#!/usr/bin/env python3
"""r34: D2 实验矩阵 — 4×4 教师标签 + GT 上界组 + 融合.

标签变体:
  alarm8 : WMFE 8×8 层告警 (教师基线, IoU≈0.59)
  alarm4 : WMFE 4×4 细化层告警 (论文 0.738 口径)  <- 主攻
  gt     : GT bbox 掩膜 (学生容量上界对照)
组:
  teacher8/teacher4 : 教师基线
  student_*          : 被动学生 (无水印无密钥推理)
  fusion_or_*        : 学生 OR 教师
输出: results/r34_distill_matrix.json (meta + 各组指标 + 逐样本 IoU)
"""
import sys, os, json, time, argparse
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs

DELTA = 15; PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))
K_LOC, T_ALARM = 9, 0.38
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')


# ---------------- 数据: img / gt / alarm8 / alarm4 ----------------
def alarm_from_fail(fail32, bs, k, min_area):
    """块级失败图 -> 像素级告警掩膜 (与论文 r14b loc_iou / r28 post 口径一致)."""
    if bs == 8:
        sm = ks.smooth_fail_map(fail32, K_LOC)
        b = (sm > T_ALARM).astype(np.uint8)
    else:
        b = (cv2.blur(fail32, (k, k)) > T_ALARM).astype(np.uint8)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    out32 = np.zeros_like(b)
    for i in range(1, nlab):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out32[lab == i] = 1
    return np.kron(out32, np.ones((bs, bs), np.uint8))


def build_dataset(n, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        return z['img'], z['gt'], z['alarm8'], z['alarm4']
    meta = json.load(open(f'{AS}/meta.json'))
    imgs, gts, a8s, a4s = [], [], [], []
    t0 = time.time()
    for s_ in meta['samples'][:n]:
        sid = s_['id']
        try:
            o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except Exception:
            continue
        key = ks.image_key(MASTER, sid)
        k4 = layer_key(key, 'bs4')
        # 双层嵌入 (8×8 主层 + 4×4 细化层, 论文口径)
        wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        wm = embed_bs(wm, k4, 4, DELTA, (1, 1))
        edited, _ = paste_edit(wm, a, s_['bbox'])
        # 两层失败图
        f8 = ks.fail_map(edited, key, DELTA, CHK, 'coord')
        f4 = fail_bs(edited, k4, 4, DELTA, (1, 1))
        alarm8 = alarm_from_fail(f8, 8, K_LOC, 8)
        alarm4 = alarm_from_fail(f4, 4, 5, 32)
        # GT bbox
        H, W = edited.shape[:2]
        gt = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = s_['bbox']
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
        imgs.append(edited); gts.append(gt); a8s.append(alarm8); a4s.append(alarm4)
        if len(imgs) >= n: break
    np.savez_compressed(cache, img=np.array(imgs), gt=np.array(gts),
                        alarm8=np.array(a8s), alarm4=np.array(a4s))
    print(f'dataset: n={len(imgs)} ({time.time()-t0:.0f}s)')
    return np.array(imgs), np.array(gts), np.array(a8s), np.array(a4s)


# ---------------- UNet (同 r33) ----------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                               nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1),
                               nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x): return self.c(x)


class SmallUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.e1 = ConvBlock(3, base); self.e2 = ConvBlock(base, base*2)
        self.e3 = ConvBlock(base*2, base*4); self.e4 = ConvBlock(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.d3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = ConvBlock(base*2, base)
        self.head = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        c1 = self.e1(x); c2 = self.e2(self.pool(c1))
        c3 = self.e3(self.pool(c2)); c4 = self.e4(self.pool(c3))
        d3 = self.d3(torch.cat([self.up3(c4), c3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), c2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), c1], 1))
        return self.head(d1)


# ---------------- 指标 ----------------
def eval_mask(pred, gt):
    ious, covs, ars, per = [], [], [], []
    for p, g in zip(pred, gt):
        inter = np.logical_and(p, g).sum(); union = np.logical_or(p, g).sum()
        iou = inter / union if union else 1.0
        ious.append(iou); per.append(float(iou))
        if g.sum():
            c = inter / g.sum(); covs.append(c); ars.append(p.sum() / g.sum())
    return dict(iou=float(np.mean(ious)), cov50=float(np.mean([c >= .5 for c in covs])),
                area_ratio=float(np.mean(ars)) if ars else float('nan'), per_iou=per)


def train_student(x_tr, y_tr, x_va, gt_va, x_te, epochs, tag, device, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float() / 255.,
        torch.from_numpy(y_tr[:, None]).float())
    dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    m = SmallUNet().to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=device))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    def lossf(logits, target):
        b = bce(logits, target)
        p = torch.sigmoid(logits).flatten(1); t = target.flatten(1)
        di = (p * t).sum(1); dsz = p.sum(1) + t.sum(1)
        return b + (1 - (2 * di + 1) / (dsz + 1)).mean()
    best_iou, best_ep, best_sd = 0., -1, None
    for ep in range(epochs):
        m.train(); tot = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); lo = lossf(m(xb), yb); lo.backward(); opt.step()
            tot += lo.item()
        sched.step()
        m.eval()
        with torch.no_grad():
            logits = []
            for i in range(0, len(x_va), 16):
                xb = torch.from_numpy(x_va[i:i+16].transpose(0, 3, 1, 2)).float().to(device) / 255.
                logits.append(torch.sigmoid(m(xb)).cpu().numpy())
            pr = (np.concatenate(logits)[:, 0] > 0.5).astype(np.uint8)
        st = eval_mask(pr, gt_va)
        if st['iou'] > best_iou:
            best_iou, best_ep = st['iou'], ep
            best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f'  [{tag}] ep{ep+1} loss={tot/len(dl):.4f} val IoU={st["iou"]:.3f}', flush=True)
    # val 选出的 best 权重 -> test 推理一次
    m.load_state_dict(best_sd); m.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(x_te), 16):
            xb = torch.from_numpy(x_te[i:i+16].transpose(0, 3, 1, 2)).float().to(device) / 255.
            out.append(torch.sigmoid(m(xb)).cpu().numpy())
    pr_te = (np.concatenate(out)[:, 0] > 0.5).astype(np.uint8)
    torch.save(best_sd, f'{BASE}/results/r34_student_{tag}.pth')
    print(f'  saved -> r34_student_{tag}.pth (val {best_iou:.3f})', flush=True)
    return best_iou, pr_te, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=1000)
    ap.add_argument('--n-test', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--labels', default='alarm4,gt')
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_all = a.n_train + a.n_test
    cache = f'{BASE}/results/r34_data_n{n_all}.npz'
    img, gt, a8, a4 = build_dataset(n_all, cache)
    # 严格三分: 训练 800 / 验证 200 (选模型) / 测试 200 (只报一次)
    x_tr, y_src = img[:800], img[:800]
    x_val, gt_val = img[800:1000], gt[800:1000]
    x_te, gt_te = img[1000:1200], gt[1000:1200]
    print(f'train={len(x_tr)} val={len(x_val)} test={len(x_te)}', flush=True)

    out = {'meta': dict(n_train=a.n_train, n_test=a.n_test, epochs=a.epochs,
                        labels=a.labels, delta=DELTA, student='SmallUNet(base=16)',
                        split='train800/val200/test200 (val selects epoch, test reported once)',
                        loss='BCE(pos=3)+Dice', sched='cosine', seed=0),
           'teachers': {}, 'students': {}, 'fusion_or': {}}
    # 教师基线
    for tn, tp in [('teacher8', a8[1000:1200]), ('teacher4', a4[1000:1200])]:
        st = eval_mask(tp, gt_te); st.pop('per_iou')
        out['teachers'][tn] = st
        print(f'[{tn}] IoU={st["iou"]:.3f} cov50={st["cov50"]:.1%}', flush=True)

    ymap_tr = {'alarm4': a4[:800].astype(np.float32),
               'gt': gt[:800].astype(np.float32)}
    for tag in a.labels.split(','):
        print(f'\n=== student_{tag} ===', flush=True)
        iou, pr, ep = train_student(x_tr, ymap_tr[tag], x_val, gt_val, x_te, a.epochs, tag, device)
        st = eval_mask(pr, gt_te); st['best_epoch'] = int(ep)
        per = st.pop('per_iou')
        out['students'][f'student_{tag}'] = st
        # 融合 (与 teacher4 OR)
        fus = np.logical_or(pr, a4[1000:1200]).astype(np.uint8)
        fs = eval_mask(fus, gt_te); fs.pop('per_iou')
        out['fusion_or'][f'fusion_{tag}_or_teacher4'] = fs
        print(f'[student_{tag}] IoU={st["iou"]:.3f} cov50={st["cov50"]:.1%} '
              f'| fusion IoU={fs["iou"]:.3f} cov50={fs["cov50"]:.1%}', flush=True)
        np.save(f'{BASE}/results/r34_pred_{tag}.npy', pr)
        out['students'][f'student_{tag}']['per_iou'] = per

    jf = f'{BASE}/results/r34_distill_matrix.json'
    json.dump(out, open(jf, 'w'), indent=1)
    print('\n->', jf, flush=True)


if __name__ == '__main__':
    main()
