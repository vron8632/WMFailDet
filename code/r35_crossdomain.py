#!/usr/bin/env python3
"""r35: 跨域学生评估 — COCO 训练的学生直接泛化到 4 跨域.

输出: results/r35_crossdomain.json (每域: teacher4/student IoU, 逐样本)
     results/r34_student_alarm4.pth (学生权重, 供 r36/图用)
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs
from r34_distill_matrix import (ConvBlock, SmallUNet, eval_mask, alarm_from_fail,
                                DELTA, PAY, CHK, MASTER, K_LOC, T_ALARM, AS)

DOMAINS = ['butterfly', 'sat', 'doc', 'pokemon']
N_TEST = 60


def load_domain(ds, n=N_TEST):
    root = f'{BASE}/data/xdomain/{ds}_edits'
    meta = json.load(open(f'{root}/meta.json'))
    out = []
    for s_ in meta['samples']:
        sid = s_['id']
        fpo, fpa = f'{root}/images/orig_{sid}.png', f'{root}/images/att_{sid}.png'
        if not (os.path.exists(fpo) and os.path.exists(fpa)):
            continue
        try:
            o = r8.crop8(np.array(Image.open(fpo).convert('RGB')))
            a = np.array(Image.open(fpa).convert('RGB'))
            if a.shape[:2] != o.shape[:2]:
                a = np.array(Image.fromarray(a).resize((o.shape[1], o.shape[0]), Image.LANCZOS))
            a = r8.crop8(a)
        except Exception:
            continue
        out.append((str(sid), o, a, s_['bbox']))
        if len(out) >= n:
            break
    return out


def make_pair(sid, o, a, bbox):
    """-> edited, alarm4, gt"""
    key = ks.image_key(MASTER, sid)
    k4 = layer_key(key, 'bs4')
    wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
    wm = embed_bs(wm, k4, 4, DELTA, (1, 1))
    edited, _ = paste_edit(wm, a, bbox)
    f4 = fail_bs(edited, k4, 4, DELTA, (1, 1))
    alarm4 = alarm_from_fail(f4, 4, 5, 32)
    H, W = edited.shape[:2]
    gt = np.zeros((H, W), np.uint8)
    x1, y1, x2, y2 = bbox
    gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
    return edited, alarm4, gt


def train_and_save(x_tr, y_tr, x_va, gt_va, epochs=80, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float() / 255.,
        torch.from_numpy(y_tr[:, None]).float())
    dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SmallUNet().to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=device))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    def lossf(logits, target):
        b = bce(logits, target)
        p = torch.sigmoid(logits).flatten(1); t = target.flatten(1)
        di = (p * t).sum(1); dsz = p.sum(1) + t.sum(1)
        return b + (1 - (2 * di + 1) / (dsz + 1)).mean()
    best_iou, best_sd = 0., None
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); lo = lossf(m(xb), yb); lo.backward(); opt.step()
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
            best_iou = st['iou']; best_sd = {k: v for k, v in m.state_dict().items()}
        if (ep + 1) % 20 == 0:
            print(f'  ep{ep+1} val IoU={st["iou"]:.3f}', flush=True)
    torch.save(best_sd, f'{BASE}/results/r34_student_alarm4.pth')
    print(f'saved best student (IoU={best_iou:.3f}) -> r34_student_alarm4.pth', flush=True)
    return best_iou


def infer(m, imgs, device, bs=16):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            xb = torch.from_numpy(imgs[i:i+bs].transpose(0, 3, 1, 2)).float().to(device) / 255.
            out.append(torch.sigmoid(m(xb)).cpu().numpy())
    return (np.concatenate(out)[:, 0] > 0.5).astype(np.uint8)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 1) 训练 (与 r34 同配置, 保存权重)
    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    img, gt, a8, a4 = z['img'], z['gt'], z['alarm8'], z['alarm4']
    # 三分: train800/val200/test200
    x_tr, y_tr = img[:800], a4[:800].astype(np.float32)
    x_val, gt_val = img[800:1000], gt[800:1000]
    x_te, gt_te = img[1000:1200], gt[1000:1200]
    print('training student (same config as r34)...', flush=True)
    t0 = time.time()
    # 若权重已存在则跳过训练
    wpath = f'{BASE}/results/r43_best_student.pth'
    if os.path.exists(wpath):
        print('weights exist, skip training')
    else:
        train_and_save(x_tr, y_tr, x_val, gt_val, epochs=80)
    m = SmallUNet(base=32).to(device)
    m.load_state_dict(torch.load(wpath, map_location=device))

    # COCO 基准 (同测试集)
    pr_coco = infer(m, x_te, device)
    coco_st = eval_mask(pr_coco, gt_te); coco_st.pop('per_iou')
    t4 = eval_mask(a4[1000:1200], gt_te); t4.pop('per_iou')
    print(f'COCO: student={coco_st["iou"]:.3f} teacher4={t4["iou"]:.3f}', flush=True)

    out = {'meta': dict(n_test_per_domain=N_TEST, student='r34 SmallUNet alarm4',
                        train='COCO 1000, 80ep seed0'),
           'coco': {'student': coco_st, 'teacher4': t4}, 'domains': {}}
    for ds in DOMAINS:
        items = load_domain(ds)
        eds, als, gts = [], [], []
        for sid, o, a, bb in items:
            e, al, g = make_pair(sid, o, a, bb)
            eds.append(e); als.append(al); gts.append(g)
        eds = np.array(eds); als = np.array(als); gts = np.array(gts)
        pr = infer(m, eds, device)
        ss = eval_mask(pr, gts); ss.pop('per_iou')
        ts = eval_mask(als, gts); ts.pop('per_iou')
        out['domains'][ds] = {'n': len(eds), 'student': ss, 'teacher4': ts}
        print(f'[{ds}] n={len(eds)} student={ss["iou"]:.3f} (cov {ss["cov50"]:.0%}) '
              f'teacher4={ts["iou"]:.3f} (cov {ts["cov50"]:.0%})', flush=True)
        np.save(f'{BASE}/results/r35_pred_{ds}.npy', pr)
        np.save(f'{BASE}/results/r35_gt_{ds}.npy', gts)
        np.save(f'{BASE}/results/r35_alarm4_{ds}.npy', als)
    json.dump(out, open(f'{BASE}/results/r35_crossdomain.json', 'w'), indent=1)
    print('-> results/r35_crossdomain.json', flush=True)
    print(f'time {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
