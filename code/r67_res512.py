#!/usr/bin/env python3
"""r67: 训练分辨率杠杆 — 512 输入/输出 (针对 small 层边界误差).

small 编辑在 256 图上仅 ~26px; 512 上采样后边界相对误差减半.
评估: 输出 512 下采样回 256 与 GT 比较.
"""
import sys, os, json
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from r34_distill_matrix import SmallUNet, eval_mask
from r64_tier_theory import tiers, eval_tiers

z = np.load(f'{BASE}/results/r61_L2_data.npz')
img, al, gt = z['img'], z['alarm'], z['gt']
N_TR, N_VA, N_TE = 2000, 400, 400
S = 512


def up(arr, size=S, interp=cv2.INTER_LINEAR):
    return np.array([cv2.resize(a, (size, size), interpolation=interp) for a in arr])


def run(base=32, epochs=80, batch=4, seed=0):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_tr = up(img[:N_TR].astype(np.float32)).astype(np.uint8)
    y_tr = up(al[:N_TR].astype(np.float32), interp=cv2.INTER_NEAREST).astype(np.float32)
    x_va = up(img[N_TR:N_TR+N_VA].astype(np.float32)).astype(np.uint8)
    gt_va256 = gt[N_TR:N_TR+N_VA]
    x_te = up(img[-N_TE:].astype(np.float32)).astype(np.uint8)
    gt_te256 = gt[-N_TE:]
    torch.manual_seed(seed); np.random.seed(seed)
    dl = DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float()/255.,
        torch.from_numpy(y_tr[:, None]).float()), batch_size=batch, shuffle=True, drop_last=True)
    m = SmallUNet(base=base).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=dev))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    def lossf(lo_, tg):
        b = bce(lo_, tg)
        p = torch.sigmoid(lo_).flatten(1); t = tg.flatten(1)
        di = (p*t).sum(1); ds_ = p.sum(1)+t.sum(1)
        return b + (1 - (2*di+1)/(ds_+1)).mean()
    def infer256(x):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(x), 2):
                xb = torch.from_numpy(x[i:i+2].transpose(0, 3, 1, 2)).float().to(dev)/255.
                p = torch.sigmoid(m(xb)).cpu().numpy()[:, 0]
                out.append(np.array([cv2.resize(q, (256, 256), interpolation=cv2.INTER_AREA) for q in p]))
        return np.concatenate(out)
    best, best_sd = 0., None
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        sched.step()
        pr = infer256(x_va)
        s = eval_mask((pr > .5).astype(np.uint8), gt_va256)
        if s['iou'] > best:
            best = s['iou']; best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if (ep + 1) % 20 == 0:
            print(f'  [512] ep{ep+1} val={s["iou"]:.3f}', flush=True)
    m.load_state_dict(best_sd)
    prob = infer256(x_te)
    st = eval_mask((prob > .5).astype(np.uint8), gt_te256); st.pop('per_iou')
    print(f'  [512] val={best:.3f} TEST={st["iou"]:.3f}', flush=True)
    return m, prob, st, best


if __name__ == '__main__':
    m, prob, st, v = run()
    out = {'meta': dict(n_train=N_TR, n_test=N_TE, resolution=S, base=32,
                        baseline_256={'overall': 0.9347, 'small': 0.8754}),
           'res512': {'overall': round(st['iou'], 4),
                      'tiers': eval_tiers(prob, gt[-N_TE:], tiers(gt[-N_TE:]), 'res512')}}
    torch.save(m.state_dict(), f'{BASE}/results/r67_res512.pth')
    json.dump(out, open(f'{BASE}/results/r67_res512.json', 'w'), indent=1)
    print('-> results/r67_res512.json')
