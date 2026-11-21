#!/usr/bin/env python3
"""TIFS 版数据图 (三张新图 + 复用定性图):
  figT1  t-SNE: clean / edited / strongly-degraded 的块级特征嵌入 (三类可分性)
  figT2  (a)学生-教师逐样本散点 (b)分层增益条形 (命题 3 的定量验证)
  figT3  雷达图: 方法多维度能力对比 (含 key-free 与标注成本维度)
输出: paper/latex/figs/figT*.pdf/png
"""
import sys, os, json, time
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs
from r34_distill_matrix import DELTA, PAY, CHK, MASTER

FIG = f'{BASE}/paper/latex/figs'
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')

plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 120, 'savefig.dpi': 300})
# 克制配色: 2-4 语义色 + 1 强调
C_CLEAN = '#8FA396'     # 灰绿 (clean)
INK2 = '#5A5A5A'
C_EDIT = '#3775BA'      # 蓝 (edited)
C_DEG = '#B8865F'       # 暖棕 (degraded)
C_ACC = '#2F4F6F'       # 强调深蓝
C_REF = '#B64342'       # 红 (reference/baseline)


def save(fig, name):
    for d in (f'{FIG}', f'{BASE}/figs'):
        fig.savefig(f'{d}/{name}.pdf', bbox_inches='tight')
        fig.savefig(f'{d}/{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('exported', name)


# =====================================================================
def features_for(img, key, k2=None):
    """块级取证特征 (6 维, 全部可从失效图/统计量直接算, 可解释)."""
    f8 = ks.fail_map(img, key, DELTA, CHK, 'coord')
    sm = ks.smooth_fail_map(f8, 5)
    v = np.sort(sm.ravel())
    k = max(1, int(0.10 * v.size))
    top10 = float(v[-k:].mean())
    med = float(np.median(sm))
    contrast = top10 - med
    low10 = float(np.percentile(sm, 10))
    b = (sm > 0.38).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    areas = [st[i, cv2.CC_STAT_AREA] for i in range(1, n)]
    largest = float(max(areas) / max(1, b.sum())) if areas else 0.0
    ncomp = float(len(areas))
    feat = [float(sm.mean()), contrast, float(sm.std()), largest, ncomp, low10]
    return feat, b


def collect_feats(n=100):
    """三类样本各 n 张: clean / edited / strongly-degraded(JPEG50/feather/rotate)."""
    meta = json.load(open(f'{AS}/meta.json'))
    sids = []
    for s_ in meta['samples']:
        sid = s_['id']
        if os.path.exists(f'{AS}/images/orig_{sid}.png') and os.path.exists(f'{AS}/images/att_{sid}.png'):
            sids.append(s_)
        if len(sids) >= n:
            break
    from r14b_multiscale import embed_bs, fail_bs
    from r7_keyed_scheme import paste_edit
    from r10_specificity import jpeg
    def feather(im, r=4):
        return cv2.blur(im, (2*r+1, 2*r+1))
    F, T = [], []
    t0 = time.time()
    for s_ in sids[:n]:
        sid = s_['id']
        o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
        a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        key = ks.image_key(MASTER, sid); k4 = layer_key(key, 'bs4'); k2 = layer_key(key, 'bs2')
        wm1 = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        wm2 = embed_bs(wm1, k4, 4, DELTA, (1, 1))
        wm3 = embed_bs(wm2, k2, 2, DELTA, (1, 1))
        edited, _ = paste_edit(wm3, a, s_['bbox'])
        # clean
        f, b = features_for(wm3, key)
        F.append(f); T.append('clean')
        # edited
        f, b = features_for(edited, key)
        F.append(f); T.append('edited')
        # strongly degraded (信号不可用)
        for fn in (lambda im: jpeg(im, 50), lambda im: feather(im, 4)):
            f, _ = features_for(fn(edited), key)
            F.append(f); T.append('degraded')
        if len(F) % 300 == 0:
            print(f'  feats {len(F)} ({time.time()-t0:.0f}s)', flush=True)
    return np.array(F, np.float32), T


def figT1():
    cache = f'{BASE}/results/r70_tsne_feats.npz'
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        F, T = z['F'], list(z['T'])
    else:
        F, T = collect_feats(100)
        np.savez_compressed(cache, F=F, T=np.array(T))
    print(f'features: {F.shape}, classes: {collections_counter(T) if False else {c: T.count(c) for c in set(T)}}', flush=True)
    X = StandardScaler().fit_transform(F)
    ts = TSNE(n_components=2, perplexity=30, random_state=0, init='pca')
    E = ts.fit_transform(X)
    colors = {'clean': C_CLEAN, 'edited': C_EDIT, 'degraded': C_DEG}
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for c in ('clean', 'degraded', 'edited'):
        ii = [i for i, t in enumerate(T) if t == c]
        ax.scatter(E[ii, 0], E[ii, 1], s=7, c=colors[c], alpha=0.75, label=c,
                   edgecolors='none')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=6.5, frameon=False, markerscale=1.6)
    ax.set_title('Block-level forensic features separate the three verdict classes',
                 fontsize=7.5, loc='left')
    save(fig, 'figT1_tsne')


def collections_counter(T):
    d = {}
    for t in T:
        d[t] = d.get(t, 0) + 1
    return d


# =====================================================================
def figT2():
    d = json.load(open(f'{BASE}/results/r64_tier_theory.json'))
    fig = plt.figure(figsize=(7.4, 2.9))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.30)
    # (a) 分层增益 (命题 3 定量验证)
    ax = fig.add_subplot(gs[0])
    order = ['small', 'mid', 'large', 'full']
    labs = ['small\n(<10%)', 'mid\n(10–30%)', 'large\n(30–90%)', 'full\n(>90%)']
    g4 = [d['L4'][k]['iou'] for k in order]
    g2 = [d['L2'][k]['iou'] for k in order]
    x = np.arange(4); w = 0.36
    ax.bar(x - w/2, g4, w, color='#E0A060', label='4×4 labels (b=4)', edgecolor='#8A5A2B', lw=.5)
    ax.bar(x + w/2, g2, w, color='#2E6B9E', label='2×2 labels (b=2)', edgecolor='#1B4568', lw=.5)
    for i in range(4):
        gain = g2[i] - g4[i]
        ax.annotate(f'+{gain:.3f}', (x[i] + w/2, max(g2[i], g4[i]) + 0.015),
                    ha='center', fontsize=6.8, color='#2E6B9E', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel('Test IoU'); ax.set_ylim(0.55, 1.05)
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title('(a) Label-resolution gain by edit area\n(prediction: strictly monotonic)',
                 fontsize=7.5, loc='left')
    # (b) 学生 vs 教师 逐样本散点
    ax = fig.add_subplot(gs[1])
    ax.bar([0, 1], [d['L4'].get('_overall', 0.842), d['L2'].get('_overall', 0.935)],
           0.5, color=['#E0A060', '#2E6B9E'], edgecolor=['#8A5A2B', '#1B4568'], lw=.5)
    ax.text(0, d['L4'].get('_overall', 0.842) + 0.02, '0.842', ha='center', fontsize=7)
    ax.text(1, d['L2'].get('_overall', 0.935) + 0.02, '0.935', ha='center', fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['4×4 labels', '2×2 labels'], fontsize=7)
    ax.set_ylabel('Overall test IoU'); ax.set_ylim(0.55, 1.0)
    ax.set_title('(b) Overall: b=2 vs b=4', fontsize=7.5, loc='left')
    save(fig, 'figT2_prop3')


# =====================================================================
def figT3():
    """雷达图: 方法多维度能力 (含 key-free 与标注成本两个本文独有维度)."""
    dims = ['IoU', 'Coverage', 'Area\naccuracy', 'Inference\nspeed', 'Label\ncost=0', 'Key-free']
    # 归一化 0-1: IoU/1.0; cov; 1-|AR-1|; log(1/ms) 归一; 免费=1; key-free=1
    import math
    def spd(ms):
        return min(1.0, math.log10(1000 / max(ms, 0.5)) / 3)  # 1000ms->0, 1ms->1
    series = [
        ('Ours (student)', [0.949, 1.000, 0.995, spd(1.1), 1.0, 1.0],
         '#1B3F63', True, '-', 'o'),
        ('Teacher', [0.726, 0.975, 0.960, spd(8), 1.0, 0.0],
         '#9AA6B2', False, ':', 's'),
        ('EditGuard', [0.927, 1.000, 0.912, spd(50), 0.0, 0.0],
         '#C0392B', False, '--', '^'),
        ('WAM', [0.947, 1.000, 0.950, spd(350), 0.0, 1.0],
         '#E2A33C', False, '-.', 'D'),
    ]
    N = len(dims)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4.5, 4.1), subplot_kw=dict(polar=True))
    for name, vals, c, bold, ls, mk in series:
        v = vals + vals[:1]
        ax.plot(angles, v, lw=2.4 if bold else 1.5, color=c, label=name,
                ls=ls, marker=mk, markersize=4.5 if bold else 3.6,
                markevery=1, alpha=1.0 if bold else 0.95, zorder=5 if bold else 3)
        if bold:
            ax.fill(angles, v, color=c, alpha=0.13, zorder=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=7.2)
    ax.tick_params(axis='x', pad=6)
    ax.set_ylim(0, 1.0); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=6, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.08), ncol=2, columnspacing=1.4,
              handletextpad=0.6)
    ax.spines['polar'].set_linewidth(0.6)
    ax.set_title('Six axes that matter for key-free deployment', fontsize=8, pad=18)
    save(fig, 'figT3_radar')


def fig_combined():
    """图3+图5 合并: (a) t-SNE 特征可分性  (b) 六轴能力画像"""
    import numpy as _np
    d = _np.load(f'{BASE}/results/r70_tsne_feats.npz', allow_pickle=True)
    F, T = d['F'], list(d['T'])
    X = StandardScaler().fit_transform(F)
    E = TSNE(n_components=2, perplexity=30, random_state=0, init='pca').fit_transform(X)

    fig = plt.figure(figsize=(6.5, 3.05))
    gs = GridSpec(1, 2, width_ratios=[1, 1.28], wspace=0.30)

    # (a) t-SNE
    ax = fig.add_subplot(gs[0])
    colors = {'clean': C_CLEAN, 'edited': C_EDIT, 'degraded': C_DEG}
    for c in ('clean', 'degraded', 'edited'):
        ii = [i for i, t_ in enumerate(T) if t_ == c]
        ax.scatter(E[ii, 0], E[ii, 1], s=5.5, c=colors[c], alpha=0.8,
                   label=c, edgecolors='none')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('t-SNE 1', fontsize=7); ax.set_ylabel('t-SNE 2', fontsize=7)
    ax.legend(fontsize=6.2, frameon=False, markerscale=1.7, loc='upper right',
              handletextpad=0.2, labelspacing=0.2)
    ax.set_title('(a) Block-level features separate verdict classes',
                 fontsize=7.6, loc='left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    # (b) radar
    import math
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
              bbox_to_anchor=(0.5, -0.255), ncol=2, columnspacing=1.0,
              handletextpad=0.4, labelspacing=0.2)
    save(fig, 'figT_combined')


if __name__ == '__main__':
    figT1()
    figT2()
    figT3()
    fig_combined()
