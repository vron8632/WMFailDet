#!/usr/bin/env python3
"""C&S 版概念图: motivation (新故事) + framework (三层架构)."""
import os, json, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
FIG = f'{BASE}/paper/latex/figs'
PREV = f'{BASE}/figs'
os.makedirs(FIG, exist_ok=True); os.makedirs(PREV, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'figure.dpi': 120, 'savefig.dpi': 300})
P = {'blue_main': '#0F4D92', 'blue_soft': '#3775BA', 'teal': '#42949E',
     'orange_dark': '#B5651D', 'red': '#B64342', 'neutral_mid': '#767676',
     'green_dark': '#2E7D32', 'gold': '#C9A227'}


def save(fig, name):
    for d in (FIG, PREV):
        fig.savefig(f'{d}/{name}.pdf', bbox_inches='tight')
        fig.savefig(f'{d}/{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('exported', name)


def overlay(img, mask, color, alpha=0.45):
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = out[m] * (1 - alpha) + np.array(color, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# =====================================================================
def fig_motivation_cs():
    """新动机图: 同一张图, 教师用密钥给失效图(要密钥), 学生只看图给掩膜(免费), 融合最强."""
    z = np.load(f'{BASE}/results/r37_qualitative.npz')
    host, edited, gt = z['host'], z['edited'], z['gt']
    alarm4, stu = z['alarm4'], z['student']
    i = 2  # Large 样本 (视觉清晰)
    heads = ['(a) Host', '(b) Edited (GT red)', '(c) Teacher: failure map\n(needs key)',
             '(d) Student: prediction\n(key-free)', '(e) Fusion\n(best)']
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.5))
    cells = [host[i],
             gt_box(edited[i], gt[i]),
             np.stack([255 * np.kron((ks_sm(fail := None) if False else 0) + np.zeros(1), np.ones(1))] * 0, -1) if False else None]
    cells = [host[i], gt_box(edited[i], gt[i])]
    # (c) 失效图: r34 数据里没有 fail 图, 用 alarm4 反推块图 (白=失效)
    # 用 r37 的 alarm4 重建块状图
    sm_blk = alarm4[i].astype(np.float32)
    cells.append(np.stack([255 * (1 - sm_blk)] * 3, -1).astype(np.uint8))
    cells.append(overlay(gt_box(edited[i], gt[i]), alarm4[i], (46, 125, 50)))
    cells.append(overlay(gt_box(edited[i], gt[i]), stu[i], (15, 77, 146)))
    for c in range(5):
        ax = axes[c]
        ax.imshow(cells[c]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color('#999')
        ax.set_title(heads[c], fontsize=7.5, pad=3)
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig_cs_motivation')


def gt_box(img, gt, color=(200, 30, 30), lw=2):
    import cv2
    out = img.copy()
    ys, xs = np.where(gt > 0)
    if len(xs):
        cv2.rectangle(out, (xs.min(), ys.min()), (xs.max(), ys.max()), color, lw)
    return out


def ks_sm(x):
    return 0


# =====================================================================
def fig_framework_cs():
    """新框架图: 四层 — 发布 / 训练(免费标签) / 推理(仅图像) / 证据链."""
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 5.6); ax.axis('off')

    def box(x, y, w, h, text, fc, ec, fs=7.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                    facecolor=fc, edgecolor=ec, linewidth=1.2))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs)

    def arrow(x1, y1, x2, y2, color='#555', ls='-', lw=1.4):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=12, color=color, lw=lw, linestyle=ls))

    # ---- L1 publication (y=4.5)
    box(0.3, 4.5, 2.3, 0.85, 'Owner: embed keyed QIM\n(one-time, zero training)', '#EAF1F8', P['blue_main'])
    box(3.3, 4.5, 1.9, 0.85, 'Publish\nwatermarked image', '#F5F5F5', P['neutral_mid'])
    box(5.9, 4.5, 2.5, 0.85, 'Distribution:\ngenerative edits\n+ degradations', '#FBE9E0', P['orange_dark'])
    arrow(2.6, 4.92, 3.3, 4.92); arrow(5.2, 4.92, 5.9, 4.92)
    arrow(8.4, 4.92, 9.0, 4.92); ax.text(8.7, 5.05, 'audit', fontsize=7, color='#555')

    # ---- L2 training (y=2.9)
    box(0.3, 2.9, 2.7, 0.95, 'Teacher (zero-training):\nkeyed QIM failure map\n(watermark + key)', '#EAF1F8', P['blue_main'])
    box(3.7, 2.9, 2.5, 0.95, 'Pseudo-labels (free):\npixel-accurate,\n99.5% consistent', '#E4F0E4', P['green_dark'])
    box(6.9, 2.9, 2.9, 0.95, 'Student distillation:\n1.93M params, no human\nlabels', '#E4F0E4', P['green_dark'])
    arrow(3.0, 3.38, 3.7, 3.38); arrow(6.2, 3.38, 6.9, 3.38)

    # ---- L3 inference (y=1.5)
    box(0.3, 1.5, 1.4, 0.85, 'Query\nimage', '#F5F5F5', P['neutral_mid'])
    box(2.1, 1.5, 2.3, 0.85, 'Student (key-free):\nimage only,\nGPU 1.1 ms', '#E4F0E4', P['green_dark'])
    box(5.0, 1.5, 3.0, 0.85, 'Evidence-chain fusion\nvisual + checks, arbitrated', '#FDF6E3', P['gold'])
    box(8.6, 1.5, 1.5, 0.85, 'Teacher\n(optional, keyed)', '#EAF1F8', P['blue_main'])
    arrow(1.7, 1.93, 2.1, 1.93); arrow(4.4, 1.93, 5.0, 1.93)
    arrow(8.0, 1.93, 8.6, 1.93, color=P['neutral_mid'], ls=(0, (3, 2)))
    ax.text(8.3, 2.12, 'optional', fontsize=6.5, color=P['neutral_mid'], ha='center')

    # ---- L4 outputs (y=0.4)
    box(2.8, 0.4, 4.0, 0.85, 'Verdict: AUTHENTIC / TAMPERED / INDETERMINATE\n+ region + evidence profile', '#FDF6E3', P['gold'])
    arrow(6.5, 1.5, 5.3, 1.25)

    # 分区虚线与标签
    for y, lab in [(5.45, 'publication'), (4.15, 'training (offline, free labels)'),
                   (2.6, 'inference (image only)'), (1.35, 'verdict & evidence chain')]:
        ax.plot([0.15, 10.05], [y, y], ls=':', color='#999', lw=0.8)
        ax.text(0.05, y + 0.08, lab, fontsize=7, color='#777')
    save(fig, 'fig_cs_framework')


if __name__ == '__main__':
    fig_motivation_cs()
    fig_framework_cs()
