#!/usr/bin/env python3
"""r60: 标签分辨率配置搜索 (B 实验 + "可学习配置"的具体化).

问题: 现用 8×8/4×4 块失效图作标签, 4px 粒度限制了学生上限 (0.851 vs GT 0.903).
假设: 更细的失效图 (2×2 = 2px 块) 或 多层组合 能提升标签质量 -> 学生上限.

配置: L8 (8x8 层), L4, L2, L8+L4, L8+L4+L2
口径: 同一批图, 同一学生架构(base32), 同一训练/测试划分
输出: results/r60_label_config.json
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
from r34_distill_matrix import SmallUNet, eval_mask, DELTA, PAY, CHK, MASTER

BIG = f'{BASE}/data/bigset'
CONFIGS = {
    'L8': [(8, None)],
    'L4': [(4, None)],
    'L2': [(2, None)],
    'L8+L4': [(8, None), (4, None)],
    'L8+L4+L2': [(8, None), (4, None), (2, None)],
}
N_TR, N_VA, N_TE = 800, 200, 200   # 探针规模 (快速看趋势)


def alarm_from(fail, bs, min_area, thr=0.38):
    if bs == 8:
        sm = ks.smooth_fail_map(fail, 5)
    else:
        k = 5 if bs >= 4 else 3
        sm = cv2.blur(fail.astype(np.float32), (k, k))
    b = (sm > thr).astype(np.uint8)
    if bs < 8:
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    out = np.zeros_like(b)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return np.kron(out, np.ones((bs, bs), np.uint8))


def build(n):
    meta = json.load(open(f'{BIG}/meta.json'))
    items = []
    for s_ in meta['samples']:
        sid = s_['id']
        fo, fa = f'{BIG}/images/orig_{sid}.png', f'{BIG}/images/att_{sid}.png'
        if os.path.exists(fo) and os.path.exists(fa):
            items.append(s_)
        if len(items) >= n:
            break
    out = {c: {'img': [], 'alarm': [], 'gt': []} for c in CONFIGS}
    t0 = time.time()
    for s_ in items:
        sid = s_['id']
        o = r8.crop8(np.array(Image.open(f'{BIG}/images/orig_{sid}.png').convert('RGB')))
        a = r8.crop8(np.array(Image.open(f'{BIG}/images/att_{sid}.png').convert('RGB')))
        key = ks.image_key(MASTER, sid)
        k4 = layer_key(key, 'bs4'); k2 = layer_key(key, 'bs2')
        base = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        wm4 = embed_bs(base, k4, 4, DELTA, (1, 1))
        wm2 = embed_bs(wm4, k2, 2, DELTA, (1, 1))
        H, W = wm2.shape[:2]
        gt = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = s_['bbox']
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
        for cfg, layers in CONFIGS.items():
            wm = base
            if any(b == 4 for b, _ in layers):
                wm = wm4
            if any(b == 2 for b, _ in layers):
                wm = wm2
            edited, _ = paste_edit(wm, a, s_['bbox'])
            acc = np.zeros((H, W), np.uint8)
            for bs, _ in layers:
                kk = key if bs == 8 else (k4 if bs == 4 else k2)
                f = ks.fail_map(edited, key, DELTA, CHK, 'coord') if bs == 8 \
                    else fail_bs(edited, kk, bs, DELTA, (1, 1))
                min_area = 8 if bs == 8 else (32 if bs == 4 else 128)
                acc = np.maximum(acc, alarm_from(f, bs, min_area))
            out[cfg]['img'].append(edited); out[cfg]['alarm'].append(acc)
            out[cfg]['gt'].append(gt)
        if len(out['L8']['img']) % 100 == 0:
            print(f'  built {len(out["L8"]["img"])}/{n} ({time.time()-t0:.0f}s)', flush=True)
    for c in CONFIGS:
        for k in ('img', 'alarm', 'gt'):
            out[c][k] = np.array(out[c][k])
    return out


def train(x_tr, y_tr, x_va, gt_va, x_te, gt_te, epochs=60, seed=0, tag=''):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float()/255.,
        torch.from_numpy(y_tr[:, None]).float()), batch_size=8, shuffle=True, drop_last=True)
    m = SmallUNet(base=32).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=dev))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    def lossf(lo_, tg):
        b = bce(lo_, tg)
        p = torch.sigmoid(lo_).flatten(1); t = tg.flatten(1)
        di = (p*t).sum(1); ds_ = p.sum(1)+t.sum(1)
        return b + (1 - (2*di+1)/(ds_+1)).mean()
    def infer(x):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(x), 8):
                xb = torch.from_numpy(x[i:i+8].transpose(0, 3, 1, 2)).float().to(dev)/255.
                out.append(torch.sigmoid(m(xb)).cpu().numpy())
        return np.concatenate(out)[:, 0]
    best, best_sd = 0., None
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        sched.step()
        s = eval_mask((infer(x_va) > .5).astype(np.uint8), gt_va)
        if s['iou'] > best:
            best = s['iou']; best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_sd)
    st = eval_mask((infer(x_te) > .5).astype(np.uint8), gt_te); st.pop('per_iou')
    print(f'  [{tag}] val={best:.3f} TEST={st["iou"]:.3f}', flush=True)
    return st, best


def main():
    d = build(N_TR + N_VA + N_TE)
    out = {'meta': dict(n_train=N_TR, n_val=N_VA, n_test=N_TE, epochs=60,
                        configs=list(CONFIGS),
                        note='label resolution ladder; probe scale'), 'runs': {}}
    for cfg in CONFIGS:
        z = d[cfg]
        x_tr, y_tr = z['img'][:N_TR], z['alarm'][:N_TR].astype(np.float32)
        x_va, gt_va = z['img'][N_TR:N_TR+N_VA], z['gt'][N_TR:N_TR+N_VA]
        x_te, gt_te = z['img'][-N_TE:], z['gt'][-N_TE:]
        # 标签本身质量 (在 val 上对 GT)
        lab_iou = eval_mask(z['alarm'][N_TR:N_TR+N_VA].astype(np.uint8), gt_va)
        print(f'=== {cfg} (label IoU vs GT = {lab_iou["iou"]:.3f}) ===', flush=True)
        st, v = train(x_tr, y_tr, x_va, gt_va, x_te, gt_te, tag=cfg)
        out['runs'][cfg] = {'test': {k: round(x, 4) for k, x in st.items()},
                            'val': round(v, 4),
                            'label_iou_vs_gt': round(lab_iou['iou'], 4),
                            'label_coverage': round(lab_iou['cov50'], 4)}
    best = max(out['runs'], key=lambda k: out['runs'][k]['test']['iou'])
    out['best'] = best
    out['best_iou'] = out['runs'][best]['test']['iou']
    json.dump(out, open(f'{BASE}/results/r60_label_config.json', 'w'), indent=1)
    print(f'-> results/r60_label_config.json  BEST={best} IoU={out["best_iou"]}')


if __name__ == '__main__':
    main()
