#!/usr/bin/env python3
"""r41: 多种子稳健性 + 学习式融合 + 误差分解.

  A 3 seeds (train800/val200/test200) -> test IoU mean±std
  B 融合网格 (学生概率 + 教师告警, 权重与阈值在 val 选) -> test
  C 误差分解: 逐样本胜负 + 像素级四象限
输出: results/r41_seeds_fusion.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from r34_distill_matrix import SmallUNet, eval_mask

SEEDS = [0, 1, 2]


def train_prob(x_tr, y_tr, x_val, gt_val, epochs=80, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float() / 255.,
        torch.from_numpy(y_tr[:, None]).float())
    dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    m = SmallUNet(base=32).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=dev))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    def lossf(lo_, tg):
        b = bce(lo_, tg)
        p = torch.sigmoid(lo_).flatten(1); t = tg.flatten(1)
        di = (p*t).sum(1); ds_ = p.sum(1)+t.sum(1)
        return b + (1 - (2*di+1)/(ds_+1)).mean()
    best, best_sd = 0., None
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            pr = []
            for i in range(0, len(x_val), 16):
                xb = torch.from_numpy(x_val[i:i+16].transpose(0, 3, 1, 2)).float().to(dev)/255.
                pr.append(torch.sigmoid(m(xb)).cpu().numpy())
        pr = (np.concatenate(pr)[:, 0] > 0.5).astype(np.uint8)
        s = eval_mask(pr, gt_val)
        if s['iou'] > best:
            best = s['iou']; best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_sd); m.eval()
    def prob(x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 16):
                xb = torch.from_numpy(x[i:i+16].transpose(0, 3, 1, 2)).float().to(dev)/255.
                out.append(torch.sigmoid(m(xb)).cpu().numpy())
        return np.concatenate(out)[:, 0]
    return prob, best, m


def main():
    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    img, gt, a4 = z['img'], z['gt'], z['alarm4']
    x_tr, y_tr = img[:800], a4[:800].astype(np.float32)
    x_val, gt_val, a_val = img[800:1000], gt[800:1000], a4[800:1000]
    x_te, gt_te, a_te = img[1000:1200], gt[1000:1200], a4[1000:1200]
    out = {'meta': dict(seeds=SEEDS, n_train=800, n_val=200, n_test=200,
                        split='train800/val200/test200'), 'seeds': {},
           'fusion': {}, 'error_decomposition': {}}
    t0 = time.time()
    # ---- A 多种子
    p_vals, p_tes, ious = [], [], []
    for sd in SEEDS:
        prob, val_iou, m = train_prob(x_tr, y_tr, x_val, gt_val, 80, seed=sd)
        pv, pt = prob(x_val), prob(x_te)
        st = eval_mask((pt > 0.5).astype(np.uint8), gt_te); st.pop('per_iou')
        out['seeds'][f'seed{sd}'] = {'test_iou': round(st['iou'], 4), 'val_iou': round(val_iou, 4)}
        ious.append(st['iou']); p_vals.append(pv); p_tes.append(pt)
        print(f'[seed{sd}] test IoU={st["iou"]:.4f} (val {val_iou:.4f}) {time.time()-t0:.0f}s', flush=True)
        if sd == 0:
            torch.save(m.state_dict(), f'{BASE}/results/r41_student_seed0.pth')
    arr = np.array(ious)
    out['seeds']['summary'] = {'mean': round(float(arr.mean()), 4),
                               'std': round(float(arr.std(ddof=1)), 4),
                               'min': round(float(arr.min()), 4), 'max': round(float(arr.max()), 4)}
    print(f'[seeds] {arr.mean():.4f} ± {arr.std(ddof=1):.4f}', flush=True)

    # ---- B 融合网格 (val 选, test 报)
    Pv, Pt = np.mean(p_vals, 0), np.mean(p_tes, 0)
    Av = a_val.astype(np.float32); At = a_te.astype(np.float32)
    best_cfg = None
    for a in (0, .25, .5, .75, 1.):
        for b in (0, .25, .5, .75, 1.):
            if a + b == 0:
                continue
            for tau in (.3, .4, .5, .6):
                pr = ((a*Pv + b*Av) > tau).astype(np.uint8)
                s = eval_mask(pr, gt_val)
                if best_cfg is None or s['iou'] > best_cfg[0]:
                    best_cfg = (s['iou'], a, b, tau)
    _, a, b, tau = best_cfg
    pr_te = ((a*Pt + b*At) > tau).astype(np.uint8)
    fs = eval_mask(pr_te, gt_te)
    iou_hard_or = eval_mask(np.logical_or((Pt > .5), a_te.astype(bool)).astype(np.uint8), gt_te)
    out['fusion'] = {
        'grid_best_on_val': {'a_student': a, 'b_teacher': b, 'tau': tau,
                             'val_iou': round(best_cfg[0], 4)},
        'test': {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fs.items()},
        'hard_or_test': {'iou': round(iou_hard_or['iou'], 4),
                         'cov50': round(iou_hard_or['cov50'], 4)},
        'student_only_test': {'iou': round(eval_mask((Pt > .5).astype(np.uint8), gt_te)['iou'], 4)},
        'teacher_only_test': {'iou': round(eval_mask(a_te.astype(np.uint8), gt_te)['iou'], 4)}}
    print(f"[fusion] a={a} b={b} tau={tau} -> test IoU={fs['iou']:.4f} "
          f"(student {out['fusion']['student_only_test']['iou']:.3f}, "
          f"teacher {out['fusion']['teacher_only_test']['iou']:.3f})", flush=True)

    # ---- C 误差分解
    def per_iou(p, g):
        r = []
        for pp, gg in zip(p, g):
            i = np.logical_and(pp, gg).sum(); u = np.logical_or(pp, gg).sum()
            r.append(i/u if u else 1.0)
        return np.array(r)
    si, ti = per_iou((Pt > .5).astype(np.uint8), gt_te), per_iou(a_te.astype(np.uint8), gt_te)
    pix_s = (Pt > .5); pix_t = a_te.astype(bool); g = gt_te.astype(bool)
    out['error_decomposition'] = {
        'per_sample': {'student_wins': int((si > ti).sum()), 'teacher_wins': int((ti > si).sum()),
                       'ties': int((si == ti).sum()), 'n': int(len(si))},
        'pixel_level': {
            'student_only_correct': int((pix_s & ~pix_t & g).sum()),
            'teacher_only_correct': int((~pix_s & pix_t & g).sum()),
            'both_correct': int((pix_s & pix_t & g).sum()),
            'both_wrong_fp': int((pix_s & pix_t & ~g).sum())},
        'iou_gap_when_teacher_fails': float(np.mean(si[ti < 0.3])) if (ti < 0.3).any() else None,
        'n_teacher_fails': int((ti < 0.3).sum())}
    print('[decomp] student wins %d, teacher wins %d, ties %d' % (
        out['error_decomposition']['per_sample']['student_wins'],
        out['error_decomposition']['per_sample']['teacher_wins'],
        out['error_decomposition']['per_sample']['ties']), flush=True)
    json.dump(out, open(f'{BASE}/results/r41_seeds_fusion.json', 'w'), indent=1)
    print('-> results/r41_seeds_fusion.json', flush=True)


if __name__ == '__main__':
    main()
