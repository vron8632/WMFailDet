#!/usr/bin/env python3
"""r50: 辅助任务学生 (λ=0.1) 的融合评估 — 能否超过 baseline 融合 0.819.

流程: 训练 AuxUNet(λ=0.1) -> 保存 -> 网格融合 (学生概率 + 教师告警, val 选, test 报)
对照: baseline 学生融合 (r41: 0.819) / 辅助学生单用 (r47: 0.782)
输出: results/r50_aux_fusion.json
"""
import sys, os, json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from r34_distill_matrix import SmallUNet, eval_mask
from r47_aux_distill import AuxUNet


def train(x_tr, y_tr, aux_tr, x_val, gt_val, lam=0.1, epochs=80, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float()/255.,
        torch.from_numpy(y_tr[:, None]).float(),
        torch.from_numpy(aux_tr[:, None]).float()), batch_size=16, shuffle=True, drop_last=True)
    m = AuxUNet(base=32).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=dev))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    def main_loss(lo_, tg):
        b = bce(lo_, tg)
        p = torch.sigmoid(lo_).flatten(1); t = tg.flatten(1)
        di = (p*t).sum(1); ds_ = p.sum(1)+t.sum(1)
        return b + (1 - (2*di+1)/(ds_+1)).mean()
    def prob(x):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(x), 16):
                xb = torch.from_numpy(x[i:i+16].transpose(0, 3, 1, 2)).float().to(dev)/255.
                out.append(torch.sigmoid(m(xb)).cpu().numpy())
        return np.concatenate(out)[:, 0]
    best, best_sd = 0., None
    for ep in range(epochs):
        m.train()
        for xb, yb, ab in dl:
            xb, yb, ab = xb.to(dev), yb.to(dev), ab.to(dev)
            opt.zero_grad()
            mo, ao = m(xb, with_aux=True)
            loss = main_loss(mo, yb) + lam * nn.functional.mse_loss(torch.sigmoid(ao), ab)
            loss.backward(); opt.step()
        sched.step()
        s = eval_mask((prob(x_val) > .5).astype(np.uint8), gt_val)
        if s['iou'] > best:
            best = s['iou']; best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_sd)
    return m, prob, best


def main():
    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    img, gt, a4 = z['img'], z['gt'], z['alarm4']
    aux = np.load(f'{BASE}/results/r47_aux_targets.npy')
    x_tr, y_tr = img[:800], a4[:800].astype(np.float32)
    x_val, gt_val, a_val = img[800:1000], gt[800:1000], a4[800:1000]
    x_te, gt_te, a_te = img[1000:1200], gt[1000:1200], a4[1000:1200]

    m, prob, val = train(x_tr, y_tr, aux[:800], x_val, gt_val)
    torch.save(m.state_dict(), f'{BASE}/results/r50_aux_student.pth')
    Pv, Pt = prob(x_val), prob(x_te)
    st = eval_mask((Pt > .5).astype(np.uint8), gt_te); st.pop('per_iou')
    print(f'[aux student] val={val:.3f} TEST={st["iou"]:.3f}', flush=True)

    Av = a_val.astype(np.float32); At = a_te.astype(np.float32)
    best = None
    for a in (0, .25, .5, .75, 1.):
        for b in (0, .25, .5, .75, 1.):
            if a + b == 0:
                continue
            for tau in (.3, .4, .5, .6):
                s = eval_mask(((a*Pv + b*Av) > tau).astype(np.uint8), gt_val)
                if best is None or s['iou'] > best[0]:
                    best = (s['iou'], a, b, tau)
    _, a, b, tau = best
    fs = eval_mask(((a*Pt + b*At) > tau).astype(np.uint8), gt_te); fs.pop('per_iou')
    print(f'[fusion] a={a} b={b} tau={tau} TEST={fs["iou"]:.3f} (student alone {st["iou"]:.3f})')
    out = {'meta': dict(lam=0.1, n_train=800, n_val=200, n_test=200),
           'aux_student': {k: round(v, 4) for k, v in st.items()},
           'fusion': {'cfg': {'a': a, 'b': b, 'tau': tau},
                      **{k: round(v, 4) for k, v in fs.items()}},
           'reference': {'baseline_student': 0.760, 'baseline_fusion_r41': 0.819,
                         'teacher': 0.726}}
    json.dump(out, open(f'{BASE}/results/r50_aux_fusion.json', 'w'), indent=1)
    print('-> results/r50_aux_fusion.json')


if __name__ == '__main__':
    main()
