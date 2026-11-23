#!/usr/bin/env python3
"""重绘 fig:combined —— (a) 块级特征 t-SNE + (b) 六轴能力画像.

只依赖 results/r70_tsne_feats.npz（避免导入 code/ 下的重依赖链），
输出到 paper/cs_waterfail/figs 与 figs/（供英文稿与仓库预览共用）。

相对早期版本的改动:
  - (a) 面板加宽（更扁的宽高比，便于嵌入单栏正文，不再独占整页）
  - 坐标轴更名为 "t-SNE dim 1 / t-SNE dim 2"（更明确地表示两个嵌入维度）
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIRS = [f'{BASE}/paper/cs_waterfail/figs', f'{BASE}/figs']
NAME = 'figT_combined'

plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 120, 'savefig.dpi': 300,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
C_CLEAN, C_EDIT, C_DEG = '#8FA396', '#3775BA', '#B8865F'
INK2 = '#5A5A5A'


def main():
    d = np.load(f'{BASE}/results/r70_tsne_feats.npz', allow_pickle=True)
    F, T = d['F'], list(d['T'])
    X = StandardScaler().fit_transform(F)
    E = TSNE(n_components=2, perplexity=30, random_state=0, init='pca').fit_transform(X)

    fig = plt.figure(figsize=(7.8, 2.62))
    gs = GridSpec(1, 2, width_ratios=[1.55, 1], wspace=0.42)

    # (a) t-SNE
    ax = fig.add_subplot(gs[0])
    colors = {'clean': C_CLEAN, 'edited': C_EDIT, 'degraded': C_DEG}
    for c in ('clean', 'degraded', 'edited'):
        ii = [i for i, t_ in enumerate(T) if t_ == c]
        ax.scatter(E[ii, 0], E[ii, 1], s=5.0, c=colors[c], alpha=0.8,
                   label=c, edgecolors='none')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('t-SNE dim 1', fontsize=7)
    ax.set_ylabel('t-SNE dim 2', fontsize=7)
    ax.legend(fontsize=6.2, frameon=False, markerscale=1.7, loc='upper right',
              handletextpad=0.2, labelspacing=0.2)
    ax.set_title('(a) Block-level features separate verdict classes',
                 fontsize=7.6, loc='left')

    # (b) radar
    dims = ['IoU', 'Coverage', 'Area', 'Speed', 'Free\nlabels', 'Key-free']

    def spd(ms):
        return min(1.0, math.log10(1000 / max(ms, 0.5)) / 3)

    ser = [('Ours (student)', [0.949, 1.0, 0.995, spd(1.1), 1.0, 1.0], '#1B3F63', True, '-', 'o'),
           ('Teacher', [0.726, 0.975, 0.960, spd(8), 1.0, 0.0], '#9AA6B2', False, ':', 's'),
           ('EditGuard', [0.927, 1.0, 0.912, spd(50), 0.0, 0.0], '#C0392B', False, '--', '^'),
           ('WAM', [0.947, 1.0, 0.950, spd(350), 0.0, 1.0], '#E2A33C', False, '-.', 'D')]
    ax = fig.add_subplot(gs[1], polar=True)
    ang = [i / len(dims) * 2 * np.pi for i in range(len(dims))] + [0]
    for name, vals, c, bold, ls, mk in ser:
        v = vals + vals[:1]
        ax.plot(ang, v, lw=2.2 if bold else 1.4, color=c, label=name, ls=ls,
                marker=mk, markersize=3.6 if bold else 3.0, alpha=1.0 if bold else 0.95)
        if bold:
            ax.fill(ang, v, color=c, alpha=0.14)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(dims, fontsize=6.4)
    ax.tick_params(axis='x', pad=1.5)
    ax.set_ylim(0, 1.0); ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(['0.5', '1.0'], fontsize=5.6, color=INK2)
    ax.spines['polar'].set_linewidth(0.5)
    ax.set_title('(b) Six deployment axes', fontsize=7.6, loc='left', pad=12)
    ax.legend(fontsize=5.8, frameon=False, loc='lower center',
              bbox_to_anchor=(0.5, -0.26), ncol=2, columnspacing=1.0,
              handletextpad=0.4, labelspacing=0.2)

    for d_ in OUT_DIRS:
        os.makedirs(d_, exist_ok=True)
        fig.savefig(f'{d_}/{NAME}.pdf', bbox_inches='tight', pad_inches=0.02)
        fig.savefig(f'{d_}/{NAME}.png', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('exported', NAME)


if __name__ == '__main__':
    main()
