#!/usr/bin/env python3
"""r46: 容量与数据上限 — 回答"更多数据/更大模型能否接近 WAM(0.947)/EditGuard(0.913)".

配置 (均可比 test 200 时用 train800/val200; 扩数据配置用 train1000/val100/test100):
  base32_800  (= 主配置, 参照)
  base64_800  (容量上限, 数据不变)
  base32_1000 (数据上限, 用全部可用同域数据)
输出: results/r46_capacity.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from r34_distill_matrix import SmallUNet, eval_mask


def run(x_tr, y_tr, x_val, gt_val, x_te, gt_te, base, epochs=80, seed=0, tag=''):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float()/255.,
        torch.from_numpy(y_tr[:, None]).float()), batch_size=16, shuffle=True, drop_last=True)
    m = SmallUNet(base=base).to(dev)
    nparam = sum(p.numel() for p in m.parameters())
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
            for i in range(0, len(x), 16):
                xb = torch.from_numpy(x[i:i+16].transpose(0, 3, 1, 2)).float().to(dev)/255.
                out.append(torch.sigmoid(m(xb)).cpu().numpy())
        return np.concatenate(out)[:, 0]
    best, best_sd = 0., None
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        sched.step()
        s = eval_mask((infer(x_val) > .5).astype(np.uint8), gt_val)
        if s['iou'] > best:
            best = s['iou']; best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if (ep+1) % 20 == 0:
            print(f'  [{tag}] ep{ep+1} val={s["iou"]:.3f}', flush=True)
    m.load_state_dict(best_sd)
    st = eval_mask((infer(x_te) > .5).astype(np.uint8), gt_te); st.pop('per_iou')
    print(f'  [{tag}] base={base} params={nparam/1e6:.2f}M val={best:.3f} TEST={st["iou"]:.3f}', flush=True)
    return st, int(nparam), base, best


def main():
    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    img, gt, a4 = z['img'], z['gt'], z['alarm4']
    out = {'meta': dict(epochs=80, seed=0,
                        note='WAM 0.947 / EditGuard 0.913 are GT-mask supervised, different labels'),
           'runs': {}}
    t0 = time.time()
    # 1) base32, 800/200/200 (主配置参照)
    st, np_, b, v = run(img[:800], a4[:800].astype(np.float32), img[800:1000], gt[800:1000],
                        img[1000:1200], gt[1000:1200], 32, tag='b32_800')
    out['runs']['base32_800'] = {'test': st, 'params': np_, 'val': round(v, 4),
                                 'split': 'train800/val200/test200'}
    # 2) base64, 800/200/200 (容量上限)
    st, np_, b, v = run(img[:800], a4[:800].astype(np.float32), img[800:1000], gt[800:1000],
                        img[1000:1200], gt[1000:1200], 64, tag='b64_800')
    out['runs']['base64_800'] = {'test': st, 'params': np_, 'val': round(v, 4),
                                 'split': 'train800/val200/test200'}
    # 3) base32, 1000/100/100 (数据上限; test 变小但同分布)
    st, np_, b, v = run(img[:1000], a4[:1000].astype(np.float32), img[1000:1100], gt[1000:1100],
                        img[1100:1200], gt[1100:1200], 32, tag='b32_1000')
    out['runs']['base32_1000'] = {'test': st, 'params': np_, 'val': round(v, 4),
                                  'split': 'train1000/val100/test100 (n_test smaller!)'}
    json.dump(out, open(f'{BASE}/results/r46_capacity.json', 'w'), indent=1)
    print(f'-> results/r46_capacity.json ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
