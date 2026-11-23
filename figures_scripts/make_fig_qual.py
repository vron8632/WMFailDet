#!/usr/bin/env python3
"""定性对比图（论文 fig:qual / figs/figD1_qualitative）的绘图脚本.

数据来源: results/r78_qual_compare.npz (code/r78_qual_compare.py 产出)
  sids/imgs/gts/teacher(64x64, 4x4 块)/student/editguard

绘图约定:
  - 4 个中等编辑面积样本 (面积最接近中位数), 为便于印刷取其中较明亮的 4 个;
  - 淡色填充 + 深色描边, 保证叠加后底色不压暗, 边界清晰;
  - 列标题下的 IoU 由本脚本按样本重算, 与图严格一致。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = f'{BASE}/results/r78_qual_compare.npz'
IMG_DIR = f'{BASE}/data/bigset/images'
OUT_DIRS = [f'{BASE}/paper/cs_waterfail/figs', f'{BASE}/figs']
NAME = 'figD1_qualitative'

plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'savefig.dpi': 300, 'figure.dpi': 150})

# 填充色偏亮 (避免叠加后压暗), 描边色深 (保证边界可辨)
STYLE = {
    # alpha: 淡化程度; edge_px: 描边宽度 (0 = 不描边, 用于块状弥散的教师图)
    'gt':        dict(fill=(150, 240, 150), edge=(0, 105, 0),   alpha=0.50, edge_px=2,
                      name='Ground truth'),
    'editguard': dict(fill=(170, 226, 240), edge=(0, 110, 140), alpha=0.45, edge_px=2,
                      name='EditGuard'),
    # 教师失效图是块状弥散, 若逐块描边会形成密集线网压住底图 -> 只做淡绿填充, 不描边
    'teacher':   dict(fill=(170, 242, 175), edge=(136, 148, 214), alpha=0.40, edge_px=0,
                      name='Teacher 4$\\times$4'),
    'student':   dict(fill=(140, 240, 210), edge=(0, 95, 175),  alpha=0.50, edge_px=3,
                      name='Student (ours)'),
}


def iou(pred, gt):
    p, g = pred > 0, gt > 0
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else 1.0


def overlay(img, mask, style):
    """淡色填充 + 描边; 全图掩膜只画外框 (整幅被判篡改)。"""
    alpha, edge_px = style['alpha'], style['edge_px']
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = out[m] * (1 - alpha) + np.array(style['fill'], np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    if edge_px <= 0 and not m.all():
        return out
    mu = m.astype(np.uint8)
    edge_px = max(edge_px, 1)
    k = np.ones((2 * edge_px + 1, 2 * edge_px + 1), np.uint8)
    edge = (cv2.dilate(mu, k) - cv2.erode(mu, k)).astype(bool)
    if m.all():                      # 全图: 视为整幅掩膜, 画加粗图像边框
        bw = max(edge_px, 4)
        for c in range(3):
            out[0:bw, :, c] = style['edge'][c]
            out[-bw:, :, c] = style['edge'][c]
            out[:, 0:bw, c] = style['edge'][c]
            out[:, -bw:, c] = style['edge'][c]
    else:
        out[edge] = np.array(style['edge'], np.uint8)
    return out


def pick(imgs, gts, n=4):
    """中等编辑面积 (最接近中位数) 的 8 个候选中, 取较明亮的 n 个。"""
    areas = gts.reshape(len(gts), -1).mean(1)
    cand = np.argsort(np.abs(areas - np.median(areas)))[:max(n, 8)]
    bright = imgs[cand].reshape(len(cand), -1).mean(1)
    return cand[np.argsort(-bright)[:n]]


def main():
    z = np.load(NPZ)
    imgs, gts = z['imgs'], z['gts']
    teacher = np.kron((z['teacher'] > 0).astype(np.uint8),
                      np.ones((4, 4), np.uint8)) * 255
    masks = {'gt': gts, 'editguard': z['editguard'],
             'teacher': teacher, 'student': z['student']}

    sel = pick(imgs, gts)
    hosts = [cv2.cvtColor(cv2.imread(f'{IMG_DIR}/orig_{z["sids"][i]}.png'), cv2.COLOR_BGR2RGB)
             for i in sel]
    scores = {k: np.mean([iou(v[i], gts[i]) for i in sel]) for k, v in masks.items()}
    print('samples', [str(z['sids'][i]) for i in sel],
          {k: round(v, 3) for k, v in scores.items()})

    heads = ['(a) Host', '(b) Edited',
             '(c) Ground truth',
             f"(d) EditGuard\nIoU {scores['editguard']:.2f}",
             f"(e) Teacher 4$\\times$4\nIoU {scores['teacher']:.2f}",
             f"(f) Student (ours)\nIoU {scores['student']:.2f}"]
    order = ['gt', 'editguard', 'teacher', 'student']

    n = len(sel)
    # 物理尺寸贴近 \columnwidth (约 5.4 in), 避免插入时整体缩小导致标题不可读
    fig, axes = plt.subplots(n, 6, figsize=(5.4, 1.10 * n))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(sel):
        cells = [hosts[r], imgs[i]]   # (a) 未编辑宿主图, (b) 编辑图
        for k in order:
            if k == 'gt':            # GT 直接叠在编辑图上, 便于看区域位置
                cells.append(overlay(imgs[i], gts[i], STYLE['gt']))
            else:
                cells.append(overlay(imgs[i], masks[k][i], STYLE[k]))
        for c, cell in enumerate(cells):
            ax = axes[r, c]
            ax.imshow(cell, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5); sp.set_color('#B0B0B0')
            if r == 0:
                ax.set_title(heads[c], fontsize=6.6, pad=3)
    fig.tight_layout(w_pad=0.22, h_pad=0.18)
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(f'{d}/{NAME}.pdf', bbox_inches='tight', pad_inches=0.02)
        fig.savefig(f'{d}/{NAME}.png', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('exported', NAME)


if __name__ == '__main__':
    main()
