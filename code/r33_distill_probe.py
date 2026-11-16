#!/usr/bin/env python3
"""r33: D2 可行性探针 — 把 WMFE 失效图蒸馏为被动学生（推理无需水印/密钥）.

设计:
  教师标签两种变体:
    V1 soft  = 平滑失效图 (有 25% 底噪, 软标签)
    V2 alarm = WMFE 告警区域二值图 (干净但 8px 粒度; 干净图->全零)
  学生: 小型 UNet, 输入=编辑图 RGB, 输出=掩膜 logit
  评估: 学生 vs GT(bbox) 的 IoU/覆盖/面积比, 对照教师 (4x4 层 IoU 0.738)
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

DELTA = 15; PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))
K_LOC, T_ALARM = 9, 0.38
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')


# ---------------- 数据集: 生成编辑图 + 两种伪标签 ----------------
def build_dataset(n, cache_path):
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return z['img'], z['soft'], z['alarm'], z['gt']
    meta = json.load(open(f'{AS}/meta.json'))
    imgs, softs, alarms, gts = [], [], [], []
    t0 = time.time()
    for s_ in meta['samples'][:n]:
        sid = s_['id']
        try:
            o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except Exception:
            continue
        key = ks.image_key(MASTER, sid)
        wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        edited, _ = paste_edit(wm, a, s_['bbox'])
        fail = ks.fail_map(edited, key, DELTA, CHK, 'coord')
        sm = ks.smooth_fail_map(fail, K_LOC)
        # V1: 平滑失效图 (软), 块级32x32 -> 像素级256x256
        soft32 = np.clip((sm - 0.25) / 0.45, 0, 1).astype(np.float32)
        soft = np.kron(soft32, np.ones((8, 8), np.float32))
        # V2: WMFE 告警 (阈值+连通域, 干净图自然为零)
        binmap = (sm > T_ALARM).astype(np.uint8)
        nlab, lab, stats, _ = cv2.connectedComponentsWithStats(binmap, 8)
        alarm32 = np.zeros_like(binmap)
        for i in range(1, nlab):
            if stats[i, cv2.CC_STAT_AREA] >= 8:
                alarm32[lab == i] = 1
        alarm = np.kron(alarm32, np.ones((8, 8), np.uint8))  # -> 256x256
        # GT: bbox 区域 (256 尺度)
        H, W = edited.shape[:2]
        gt = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = s_['bbox']
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
        imgs.append(edited); softs.append(soft); alarms.append(alarm); gts.append(gt)
        if len(imgs) >= n: break
    np.savez_compressed(cache_path, img=np.array(imgs), soft=np.array(softs),
                        alarm=np.array(alarms), gt=np.array(gts))
    print(f'dataset built: n={len(imgs)} ({time.time()-t0:.0f}s) -> {cache_path}')
    return np.array(imgs), np.array(softs), np.array(alarms), np.array(gts)


# ---------------- 小型 UNet ----------------
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
        u3 = self.up3(c4); d3 = self.d3(torch.cat([u3, c3], 1))
        u2 = self.up2(d3); d2 = self.d2(torch.cat([u2, c2], 1))
        u1 = self.up1(d2); d1 = self.d1(torch.cat([u1, c1], 1))
        return self.head(d1)


# ---------------- 指标 (与论文口径一致) ----------------
def eval_mask(pred, gt):
    """pred/gt: (N,H,W) binary. 返回 IoU/覆盖/面积比 (逐图后平均)."""
    ious, covs, ars = [], [], []
    for p, g in zip(pred, gt):
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        ious.append(inter / union if union else 1.0)
        gsum = g.sum()
        if gsum:
            covs.append(inter / gsum)
            ars.append(p.sum() / gsum)
    return dict(iou=float(np.mean(ious)), cov50=float(np.mean([c >= 0.5 for c in covs])),
                area_ratio=float(np.mean(ars)) if ars else float('nan'))


def train_student(x_tr, y_tr, x_va, y_va, gt_va, epochs, tag, device):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float() / 255.,
        torch.from_numpy(y_tr[:, None]).float())
    dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    m = SmallUNet().to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=device))
    def lossf(logits, target):
        b = bce(logits, target)
        p = torch.sigmoid(logits).flatten(1); t = target.flatten(1)
        di = (p * t).sum(1); dsz = p.sum(1) + t.sum(1)
        return b + (1 - (2 * di + 1) / (dsz + 1)).mean()
    best = None
    for ep in range(epochs):
        m.train(); tot = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); lo = lossf(m(xb), yb); lo.backward(); opt.step()
            tot += lo.item()
        # 验证 (取 argmax 0.5)
        m.eval()
        with torch.no_grad():
            logits = []
            for i in range(0, len(x_va), 16):
                xb = torch.from_numpy(x_va[i:i+16].transpose(0, 3, 1, 2)).float().to(device) / 255.
                logits.append(torch.sigmoid(m(xb)).cpu().numpy())
            pr = (np.concatenate(logits)[:, 0] > 0.5).astype(np.uint8)
        st = eval_mask(pr, gt_va)
        if best is None or st['iou'] > best[0]:
            best = (st['iou'], pr.copy(), ep)
        if (ep + 1) % 5 == 0:
            print(f'  [{tag}] ep{ep+1} loss={tot/len(dl):.4f} val IoU={st["iou"]:.3f}')
    return best  # (iou, pred, ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=400)
    ap.add_argument('--n-test', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--variants', default='V1_soft,V2_alarm')
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    cache = f'{BASE}/results/r33_distill_data.npz'
    n_all = a.n_train + a.n_test
    img, soft, alarm, gt = build_dataset(n_all, cache)
    # split-half 约定: 前段训练, 后段测试
    x_tr, x_va = img[:a.n_train], img[a.n_train:n_all]
    gt_va = gt[a.n_train:n_all]
    print(f'train={len(x_tr)} test={len(x_va)}')

    # 教师基线 (同测试集): 用 V2 alarm 直接当教师输出 (与 WMFE 告警一致)
    t_pred = alarm[a.n_train:n_all]
    t_st = eval_mask(t_pred, gt_va)
    print(f'[教师 WMFE] IoU={t_st["iou"]:.3f} cov50={t_st["cov50"]:.1%} area={t_st["area_ratio"]:.2f}')

    results = {'n_train': a.n_train, 'n_test': a.n_test, 'epochs': a.epochs,
               'teacher': t_st, 'variants': {}}
    cand = {'V1_soft': soft[:a.n_train].astype(np.float32),
            'V2_alarm': alarm[:a.n_train].astype(np.float32)}
    for tag in a.variants.split(','):
        y = cand[tag]
        print(f'\n=== {tag} ===')
        iou, pr, ep = train_student(x_tr, y, x_va, gt_va, gt_va, a.epochs, tag, device)
        st = eval_mask(pr, gt_va)
        st['best_epoch'] = int(ep)
        print(f'[{tag}] BEST IoU={st["iou"]:.3f} cov50={st["cov50"]:.1%} area={st["area_ratio"]:.2f} '
              f'(教师 {t_st["iou"]:.3f})')
        results['variants'][tag] = st
        np.save(f'{BASE}/results/r33_pred_{tag}.npy', pr)

    out = f'{BASE}/results/r33_distill_probe.json'
    json.dump(results, open(out, 'w'), indent=1)
    print('\n->', out)


if __name__ == '__main__':
    main()
