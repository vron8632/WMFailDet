#!/usr/bin/env python3
"""r74: δ=25 配置下的 GT 监督上界 (与 r70 的免费标签学生同配置对比).

r34 的 GT 上界 0.903 是 δ=15 配置; 审稿指出与 δ=25 学生比较口径不一致.
本实验用同一份 δ=25 数据, 把标签换成 GT 框掩膜, 重训同架构学生.
输出: results/r74_gt_upper_d25.json
"""
import sys, os, json
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import torch
from r61_L2_scale import train
from r34_distill_matrix import eval_mask

N_TR, N_VA, N_TE = 2000, 400, 400


def main():
    z = np.load(f'{BASE}/results/r70_data_d25.npz')
    img, gt = z['img'], z['gt']
    # GT 掩膜做标签 (与 r34 的 "精确框=GT" 一致), 软标签: 二值 -> float
    y = gt.astype(np.float32)
    x_tr, y_tr = img[:N_TR], y[:N_TR]
    x_va, gt_va = img[N_TR:N_TR+N_VA], gt[N_TR:N_TR+N_VA]
    x_te, gt_te = img[-N_TE:], gt[-N_TE:]
    st, v, m = train(x_tr, y_tr, x_va, gt_va, x_te, gt_te, 32, seed=1, tag='gt_d25')
    out = {'meta': dict(delta=25, label='GT bbox mask', n_train=N_TR, n_test=N_TE,
                        note='same data/config as r70 free-label student'),
           'gt_upper': {'iou': round(st['iou'], 4), 'val': round(v, 4),
                        'tiers': {k: round(x['iou'], 4) for k, x in
                                  __import__('r64_tier_theory', fromlist=['x']).tiers(gt_te).items()} if False else None},
           'compare': {'free_labels_d25': 0.9491, 'seeds': '0.9498/0.9499/0.9477'}}
    json.dump(out, open(f'{BASE}/results/r74_gt_upper_d25.json', 'w'), indent=1)
    print('GT upper (d25):', round(st['iou'], 4), '| free-label student: 0.9491')
    print('-> results/r74_gt_upper_d25.json')


if __name__ == '__main__':
    main()
