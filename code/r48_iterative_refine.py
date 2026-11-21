#!/usr/bin/env python3
"""r48: 新机制二 — 教师-学生迭代标签精化 (teacher-student refinement loop).

一轮 = 学生预测训练集 -> 与教师失效图交叉校验 -> 精化伪标签 -> 用精化标签重训学生.
机制含义: 零训练教师做"标签清洗/召回补全", 学生做"视觉细化", 两者互相纠错 (self-training 闭环).
对照: 原始 alarm4 标签 (baseline) / 精化 1 轮 / 精化 2 轮.
输出: results/r48_iterative_refine.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import keyed_scheme as ks
from r34_distill_matrix import SmallUNet, eval_mask, DELTA, CHK, MASTER


def train(x_tr, y_tr, x_val, gt_val, epochs=80, seed=0, init=None, tag=''):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float()/255.,
        torch.from_numpy(y_tr[:, None]).float()), batch_size=16, shuffle=True, drop_last=True)
    m = SmallUNet(base=32).to(dev)
    if init is not None:
        m.load_state_dict(init)
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
    return m, infer, best, best_sd


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    img, gt, a4 = z['img'], z['gt'], z['alarm4']
    x_tr, y_tr0 = img[:800], a4[:800].astype(np.float32)
    x_val, gt_val = img[800:1000], gt[800:1000]
    x_te, gt_te = img[1000:1200], gt[1000:1200]
    # 教师失效图 (平滑, 归一化) — 用于交叉校验
    tc = f'{BASE}/results/r47_aux_targets.npy'
    teacher_fail = np.load(tc) if os.path.exists(tc) else None
    assert teacher_fail is not None, 'run r47 first (builds failure targets)'
    tf_tr = teacher_fail[:800]

    out = {'meta': dict(n_train=800, n_val=200, n_test=200, epochs=80, seed=0,
                        mechanism='teacher-student iterative label refinement'),
           'rounds': {}}

    # round 0: baseline (原始 alarm4)
    m, infer, val, sd = train(x_tr, y_tr0, x_val, gt_val, tag='round0')
    st = eval_mask((infer(x_te) > .5).astype(np.uint8), gt_te); st.pop('per_iou')
    out['rounds']['round0_baseline'] = {'test': {k: round(v,4) for k,v in st.items()},
                                        'val': round(val, 4)}
    print(f'[round0] TEST={st["iou"]:.3f}', flush=True)

    cur_sd = sd
    for r in (1, 2):
        # 1) 当前学生对训练集的预测
        p_tr = infer(x_tr)
        # 2) 交叉校验 -> 精化标签: 视觉细化 (学生) 与可信召回 (教师) 的软融合
        refined = np.clip(0.6 * p_tr + 0.4 * tf_tr, 0, 1)
        # 只在分歧区域强化: 学生高/教师低 -> 降权; 教师高/学生低 -> 补入
        disagree_fp = (p_tr > 0.6) & (tf_tr < 0.2)
        disagree_fn = (p_tr < 0.2) & (tf_tr > 0.5)
        refined[disagree_fp] = 0.15
        refined[disagree_fn] = 0.9
        print(f'  [round{r}] label stats: fp-suppressed={disagree_fp.mean():.3f} '
              f'fn-filled={disagree_fn.mean():.3f}', flush=True)
        m, infer, val, cur_sd = train(x_tr, refined, x_val, gt_val, tag=f'round{r}')
        st = eval_mask((infer(x_te) > .5).astype(np.uint8), gt_te); st.pop('per_iou')
        out['rounds'][f'round{r}_refined'] = {'test': {k: round(v,4) for k,v in st.items()},
                                              'val': round(val, 4)}
        print(f'[round{r}] TEST={st["iou"]:.3f}', flush=True)

    torch.save(cur_sd, f'{BASE}/results/r48_student_refined.pth')
    json.dump(out, open(f'{BASE}/results/r48_iterative_refine.json', 'w'), indent=1)
    print('-> results/r48_iterative_refine.json')


if __name__ == '__main__':
    main()
