#!/usr/bin/env python3
"""r72: δ=25 的多种子确认 (r70 发现 δ=25 给出 0.9498, 需排除训练随机性).

用 r70 已生成的 δ=25 数据, 跑 3 个种子 (0/1/2), 报告均值±std.
对齐 r63 的做法 (L2/δ=15 三种子: 0.9347/0.9315/0.9343 -> 0.9335±0.0017).
输出: results/r72_delta25_seeds.json
"""
import sys, os, json
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import torch
from r61_L2_scale import train
from r34_distill_matrix import eval_mask
from r64_tier_theory import tiers, eval_tiers

N_TR, N_VA, N_TE = 2000, 400, 400


def main():
    z = np.load(f'{BASE}/results/r70_data_d25.npz')
    img, al, gt = z['img'], z['alarm'], z['gt']
    x_tr, y_tr = img[:N_TR], al[:N_TR].astype(np.float32)
    x_va, gt_va = img[N_TR:N_TR+N_VA], gt[N_TR:N_TR+N_VA]
    x_te, gt_te = img[-N_TE:], gt[-N_TE:]
    out = {'meta': dict(delta=25, n_train=N_TR, n_test=N_TE, label='2x2 layer',
                        baseline_delta15='0.9335+-0.0017 (r63)'), 'runs': {}}
    for seed in (1, 2):   # seed 0 已由 r70 给出 (0.9498)
        st, v, m = train(x_tr, y_tr, x_va, gt_va, x_te, gt_te, 32, seed=seed,
                         tag=f'd25_s{seed}')
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        m.eval(); pr = []
        with torch.no_grad():
            for i in range(0, len(x_te), 8):
                xb = torch.from_numpy(x_te[i:i+8].transpose(0, 3, 1, 2)).float().to(dev)/255.
                pr.append(torch.sigmoid(m(xb)).cpu().numpy()[:, 0])
        prob = np.concatenate(pr)
        t = eval_tiers(prob, gt_te, tiers(gt_te), f'd25_s{seed}')
        out['runs'][f'seed{seed}'] = {'iou': round(st['iou'], 4), 'val': round(v, 4),
                                      'tiers': t}
        json.dump(out, open(f'{BASE}/results/r72_delta25_seeds.json', 'w'), indent=1)
    seed0 = 0.9498
    ious = [seed0] + [v['iou'] for v in out['runs'].values()]
    out['summary'] = {'ious': ious, 'mean': round(float(np.mean(ious)), 4),
                      'std': round(float(np.std(ious, ddof=1)), 4)}
    json.dump(out, open(f'{BASE}/results/r72_delta25_seeds.json', 'w'), indent=1)
    print('summary:', out['summary'])
    print('-> results/r72_delta25_seeds.json')


if __name__ == '__main__':
    main()
