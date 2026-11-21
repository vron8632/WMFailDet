#!/usr/bin/env python3
"""r76: δ=25 配置下的监督来源消融 (复验 P0-3).

r38 的 0.602(噪声框)/0.269(图像级)/0.054(点) 都是 δ=15 + 800 张.
本实验用 δ=25 数据 + 2000 张, 只重测三个非 GT 标签, 与免费伪标签(0.9491)、GT(0.9872) 同配置对比.
输出: results/r76_label_ablation_d25.json
"""
import sys, os, json
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import torch
from r61_L2_scale import train

N_TR, N_VA, N_TE = 2000, 400, 400


def noisy_boxes(gt, rng, expand=0.15, jitter=3):
    """模拟人工框标注误差: 外扩 15% + 随机抖动 (在 GT 掩膜上做形态学近似)"""
    y = np.zeros_like(gt, dtype=np.float32)
    for i in range(len(gt)):
        ys, xs = np.where(gt[i] > 0)
        if len(ys) == 0:
            y[i] = gt[i]; continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        h, w = y1 - y0, x1 - x0
        dy, dx = int(h * expand), int(w * expand)
        jy, jx = rng.integers(-jitter, jitter + 1, 2)
        y0 = max(0, y0 - dy + jy); y1 = min(gt.shape[1] - 1, y1 + dy + jy)
        x0 = max(0, x0 - dx + jx); x1 = min(gt.shape[2] - 1, x1 + dx + jx)
        y[i, y0:y1 + 1, x0:x1 + 1] = 1.0
    return y


def point_labels(gt, rng):
    y = np.zeros_like(gt, dtype=np.float32)
    for i in range(len(gt)):
        ys, xs = np.where(gt[i] > 0)
        if len(ys) == 0: continue
        k = rng.integers(0, len(ys))
        cy, cx = ys[k], xs[k]
        y[i, max(0, cy-2):cy+3, max(0, cx-2):cx+3] = 1.0
    return y


def image_labels(gt):
    """图像级: 有篡改则整图 1 (无法定位)"""
    y = (gt.reshape(len(gt), -1).max(1) > 0).astype(np.float32)
    return np.repeat(y[:, None, None], gt.shape[1], 1).repeat(gt.shape[2], 2)


def main():
    z = np.load(f'{BASE}/results/r70_data_d25.npz')
    img, gt = z['img'], z['gt']
    rng = np.random.default_rng(0)
    sets = {'noisy_boxes': noisy_boxes(gt, rng),
            'point': point_labels(gt, rng),
            'image_level': image_labels(gt)}
    out = {'meta': dict(delta=25, n_train=N_TR, n_test=N_TE,
                        reference=dict(free_labels=0.9491, gt_upper=0.9872,
                                       note='all at delta=25, 2000 train')),
           'runs': {}}
    for name, y in sets.items():
        print(f'=== {name} ===', flush=True)
        st, v, m = train(img[:N_TR], y[:N_TR], img[N_TR:N_TR+N_VA], gt[N_TR:N_TR+N_VA],
                         img[-N_TE:], gt[-N_TE:], 32, seed=1, tag=name)
        out['runs'][name] = {'iou': round(st['iou'], 4), 'val': round(v, 4)}
        json.dump(out, open(f'{BASE}/results/r76_label_ablation_d25.json', 'w'), indent=1)
        print(f'  {name}: TEST={st["iou"]:.4f}', flush=True)
    print('-> results/r76_label_ablation_d25.json')


if __name__ == '__main__':
    main()
