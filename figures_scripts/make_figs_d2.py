#!/usr/bin/env python3
"""D2 蒸馏闭环的结果图 (4 张):
  figD1_qualitative   定性网格 4x6: host|edited+GT|failure|teacher|student|WAM
  figD2_distill       主结果: (a)各组IoU/覆盖柱 (b)教师vs学生逐样本散点
  figD3_crossdomain   跨域: 5域 x (教师4|学生)
  figD4_degradation   退化: 学生/教师保守/教师可判定 + 拒识率
输出到 paper/latex/figs (D2 系列) 与 figs/ (GitHub 预览)
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f'{BASE}/paper/latex/figs'
PREV = f'{BASE}/figs'
os.makedirs(FIG, exist_ok=True); os.makedirs(PREV, exist_ok=True)
R = f'{BASE}/results'

plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 120, 'savefig.dpi': 300})
P = {'neutral_dark': '#4D4D4D', 'neutral_mid': '#767676', 'neutral_light': '#CFCECE',
     'blue_main': '#0F4D92', 'blue_soft': '#3775BA', 'green': '#8BCF8B',
     'green_dark': '#2E7D32', 'red': '#B64342', 'gold': '#C9A227',
     'orange_soft': '#E8A15D', 'orange_dark': '#B5651D', 'teal': '#42949E',
     'violet': '#9A4D8E'}


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


def gt_box(img, gt, color=(200, 30, 30), lw=2):
    out = img.copy()
    ys, xs = np.where(gt > 0)
    if len(xs):
        import cv2
        cv2.rectangle(out, (xs.min(), ys.min()), (xs.max(), ys.max()), color, lw)
    return out


# =====================================================================
def figD1():
    z = np.load(f'{R}/r37_qualitative.npz')
    host, edited, gt = z['host'], z['edited'], z['gt']
    fail8, alarm4 = z['fail8'], z['alarm4']
    stu, wam = z['student'], z['wam']
    n = len(host)
    heads = ['(a) Host', '(b) Edited (red: GT)', '(c) Failure map',
             '(d) Teacher alarm', '(e) Student (ours)', '(f) WAM mask']
    fig, axes = plt.subplots(n, 6, figsize=(11.5, 2.05 * n))
    import cv2
    for r in range(n):
        sm = cv2.blur(fail8[r].astype(np.float32), (5, 5))
        bin_up = np.kron((sm > 0.38).astype(np.float32), np.ones((8, 8), np.float32))
        cells = [
            host[r],
            gt_box(edited[r], gt[r]),
            np.stack([255 * bin_up] * 3, -1).astype(np.uint8),  # 二值: 白=失效块(与fig1一致)
            overlay(gt_box(edited[r], gt[r]), alarm4[r], (46, 125, 50)),    # 绿=教师
            overlay(gt_box(edited[r], gt[r]), stu[r], (15, 77, 146)),       # 蓝=学生
            overlay(gt_box(edited[r], gt[r]), wam[r], (201, 162, 39)),      # 金=WAM
        ]
        for c in range(6):
            ax = axes[r, c] if n > 1 else axes[c]
            ax.imshow(cells[c]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color('#999999')
            if r == 0:
                ax.set_title(heads[c], fontsize=7.5, pad=3)
            if c == 0:
                ax.set_ylabel(['Small', 'Mid', 'Large', 'Full'][r], fontsize=7.5)
    fig.tight_layout(w_pad=0.4, h_pad=0.5)
    save(fig, 'figD1_qualitative')


# =====================================================================
def figD2():
    r34 = json.load(open(f'{R}/r34_distill_matrix.json'))
    r41 = json.load(open(f'{R}/r41_seeds_fusion.json'))
    r38 = json.load(open(f'{R}/r38_label_ablation.json'))
    r40 = json.load(open(f'{R}/r40_erasure_attack.json'))
    gt_upper = r38['labels']['bbox_gt']['iou_test']
    groups = [
        ('Teacher\n8×8', r34['teachers']['teacher8'], P['neutral_light']),
        ('Teacher\n4×4', r34['teachers']['teacher4'], P['neutral_mid']),
        ('Student\nB16', r34['students']['student_alarm4'], P['blue_soft']),
        ('Student\nB32 (ours)', {'iou': r41['seeds']['summary']['mean'], 'cov50': r40['student_original']['cov50']}, P['blue_main']),
        ('Fusion\nB32+teacher', {'iou': r41['fusion']['test']['iou'],
                                        'cov50': r41['fusion']['test'].get('cov50', float('nan'))}, P['teal']),
        ('GT-label\nupper', {'iou': gt_upper, 'cov50': r38['labels']['bbox_gt'].get('cov50', float('nan'))}, P['neutral_dark']),
    ]
    fig = plt.figure(figsize=(7.8, 2.9))
    gs = GridSpec(1, 2, width_ratios=[1.35, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(groups))
    ious = [g[1]['iou'] for g in groups]
    covs = [g[1].get('cov50') for g in groups]
    ax.bar(x, ious, 0.62, color=[g[2] for g in groups], edgecolor='#333', linewidth=0.5)
    for i, (iv, cv) in enumerate(zip(ious, covs)):
        ax.text(i, iv + 0.012, f'{iv:.3f}', ha='center', fontsize=6.8)
        if cv is not None and cv == cv:
            ax.text(i, 0.03, f'cov\n{cv:.0%}', ha='center', fontsize=5.8, color='white')
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=6.0)
    ax.set_ylabel('Localization IoU'); ax.set_ylim(0, 1.0)
    ax.set_title('(a) Groups on COCO-256 (train800/val200/test200)', fontsize=8, loc='left')
    # (b) seeds scatter
    ax = fig.add_subplot(gs[1])
    sd = [v['test_iou'] for k, v in r41['seeds'].items() if k.startswith('seed')]
    ax.axhline(0.726, color=P['red'], ls='--', lw=1.2, label='Teacher 4×4 (0.726)')
    ax.scatter(np.zeros(len(sd)) + np.random.uniform(-.06, .06, len(sd)), sd, s=34,
               color=P['blue_main'], zorder=5, label='Student seeds')
    ax.plot([-.25, .25], [np.mean(sd)] * 2, color=P['blue_main'], lw=1.6,
            label=f'mean {np.mean(sd):.3f}±{np.std(sd, ddof=1):.3f}')
    ax.set_xlim(-.4, .4); ax.set_xticks([]); ax.set_ylim(0.6, 0.85)
    ax.set_ylabel('Test IoU')
    ax.set_title('(b) Seed variance (base32)', fontsize=8, loc='left')
    ax.legend(fontsize=6, frameon=False, loc='lower right')
    save(fig, 'figD2_distill')


# =====================================================================
def figD3():
    r35 = json.load(open(f'{R}/r35_crossdomain.json'))
    r39 = json.load(open(f'{R}/r39_crossdomain_training.json'))
    doms = [('coco', 'COCO\n(in-domain)'), ('butterfly', 'Butterfly'),
            ('sat', 'Satellite'), ('doc', 'Document'), ('pokemon', 'Cartoon')]
    def key_of(d, k):
        for kk in d:
            if k in kk:
                return kk
        return None
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    x = np.arange(len(doms)); w = 0.27
    t, s1, s2 = [], [], []
    for k, _ in doms:
        if k == 'coco':
            t.append(r35['coco']['teacher4']['iou']); s1.append(r35['coco']['student']['iou'])
            s2.append(r39['joint']['coco']['iou'])
        else:
            t.append(r35['domains'][key_of(r35['domains'], k)]['teacher4']['iou'])
            s1.append(r35['domains'][key_of(r35['domains'], k)]['student']['iou'])
            s2.append(r39['joint'][key_of(r39['joint'], k)]['iou'])
    ax.bar(x - w, t, w, color=P['neutral_mid'], label='Teacher 4×4 (keyed)', edgecolor='#333', lw=.5)
    ax.bar(x, s1, w, color=P['blue_soft'], label='Student (COCO-only)', edgecolor='#333', lw=.5)
    ax.bar(x + w, s2, w, color=P['blue_main'], label='Student (multi-domain)', edgecolor='#333', lw=.5)
    for i in range(len(doms)):
        ax.text(i + w, s2[i] + 0.015, f'{s2[i]:.2f}', ha='center', fontsize=6.2, color=P['blue_main'])
    ax.set_xticks(x); ax.set_xticklabels([d[1] for d in doms], fontsize=7)
    ax.set_ylabel('Localization IoU'); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.6, frameon=False, loc='upper right', ncol=1)
    ax.set_title('Cross-domain: free pseudo-labels scale to new domains', fontsize=8, loc='left')
    save(fig, 'figD3_crossdomain')


# =====================================================================
def figD4():
    d = json.load(open(f'{R}/r36_degradation.json'))['results']
    order = ['clean', 'feather4', 'jpeg85', 'jpeg75', 'jpeg50', 'noise2',
             'sharpen', 'resize0.8', 'rotate1']
    labs = ['Clean', 'Feather', 'JPEG 85', 'JPEG 75', 'JPEG 50', 'Noise',
            'Sharpen', 'Resize', 'Rotate']
    fig = plt.figure(figsize=(7.6, 2.9))
    gs = GridSpec(1, 2, width_ratios=[1.7, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(order)); w = 0.38
    st = [d[k]['student']['iou'] for k in order]
    tc = [d[k]['teacher4_conservative']['iou'] for k in order]
    td = [d[k]['teacher4_detectable_subset']['iou'] for k in order]
    ax.bar(x - w/2, st, w, color=P['blue_main'], label='Student (passive)', edgecolor='#333', lw=0.5)
    ax.bar(x + w/2, tc, w, color=P['neutral_light'],
           label='Teacher (abstain=0, conservative)', edgecolor='#333', lw=0.5)
    ax.plot(x, td, 'o--', color=P['red'], ms=3.5, lw=1.0,
            label='Teacher, detectable subset')
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.8, rotation=25, ha='right')
    ax.set_ylabel('Localization IoU'); ax.set_ylim(0, 0.85)
    ax.legend(fontsize=6.3, frameon=False, loc='upper right')
    ax.set_title('(a) Under degradations (COCO test, n=200)', fontsize=8, loc='left')
    ax = fig.add_subplot(gs[1])
    ab = [100 * d[k]['teacher_abstain_rate'] for k in order]
    ax.bar(x, ab, 0.62, color=P['orange_dark'], edgecolor='#333', lw=0.5)
    for i, v in enumerate(ab):
        if v > 3:
            ax.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([])
    ax.set_ylabel('Teacher abstain rate (%)'); ax.set_ylim(0, 112)
    ax.set_title('(b) Teacher abstention', fontsize=8, loc='left')
    save(fig, 'figD4_degradation')


def figD5():
    r38 = json.load(open(f'{R}/r38_label_ablation.json'))
    order = [('bbox_gt', 'Exact\nboxes\n(=GT)', P['neutral_dark']),
             ('failure', 'Failure\n(free,\nours)', P['blue_main']),
             ('bbox_noisy', 'Noisy\nboxes\n(human)', P['orange_dark']),
             ('image_lvl', 'Image\nlevel', P['neutral_light']),
             ('point', 'Point', P['neutral_light'])]
    fig = plt.figure(figsize=(7.4, 2.7))
    gs = GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    labs, vals, cols = [], [], []
    for k, lab, c in order:
        if k in r38['labels']:
            labs.append(lab); vals.append(r38['labels'][k]['iou_test']); cols.append(c)
    x = np.arange(len(labs))
    ax.bar(x, vals, 0.6, color=cols, edgecolor='#333', lw=.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=6.8)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.0)
    ax.set_ylabel('Localization IoU'); ax.set_ylim(0, 1.0)
    ax.set_title('(a) Supervision source (n=800)', fontsize=8, loc='left')
    ax = fig.add_subplot(gs[1])
    sc = [('n250', 250), ('n500', 500)]
    xs = [250, 500, 800]
    ys = [r38['scaling']['n250']['iou_test'], r38['scaling']['n500']['iou_test'],
          r38['labels']['failure']['iou_test']]
    ax.plot(xs, ys, 'o-', color=P['blue_main'], ms=4, lw=1.3)
    for xx, yy in zip(xs, ys):
        ax.text(xx, yy + 0.02, f'{yy:.3f}', ha='center', fontsize=6.5)
    ax.set_xlabel('Free pseudo-labelled images'); ax.set_ylabel('Test IoU')
    ax.set_xlim(150, 900); ax.set_ylim(0.35, 0.9)
    ax.set_title('(b) Scaling of free labels', fontsize=8, loc='left')
    save(fig, 'figD5_labels')


def figD6():
    r47 = json.load(open(f'{R}/r47_aux_distill.json'))
    r48 = json.load(open(f'{R}/r48_iterative_refine.json'))
    r49 = json.load(open(f'{R}/r49_mechanism.json'))
    fig = plt.figure(figsize=(7.6, 2.6))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.34)
    # (a) aux lambda sweep
    ax = fig.add_subplot(gs[0])
    lams = ['0.0', '0.1', '0.5', '1.0']
    vals = [r47['runs'][f'lambda{l}']['test']['iou'] for l in lams]
    ax.bar(np.arange(4), vals, 0.6, color=[P['neutral_light']] + [P['blue_main']]*3,
           edgecolor='#333', lw=.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f'{v:.3f}', ha='center', fontsize=6.5)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(['0\n(base)'] + lams[1:], fontsize=6.8)
    ax.set_xlabel('aux-loss weight λ'); ax.set_ylabel('Test IoU'); ax.set_ylim(0.70, 0.82)
    ax.set_title('(a) Auxiliary failure-map head', fontsize=8, loc='left')
    # (b) iterative refinement rounds
    ax = fig.add_subplot(gs[1])
    xs = [0, 1, 2]
    ys = [r48['rounds']['round0_baseline']['test']['iou'],
          r48['rounds']['round1_refined']['test']['iou'],
          r48['rounds']['round2_refined']['test']['iou']]
    ax.plot(xs, ys, 'o-', color=P['red'], ms=5, lw=1.4)
    for xx, yy in zip(xs, ys):
        ax.text(xx, yy + 0.012, f'{yy:.3f}', ha='center', fontsize=6.5)
    ax.set_xticks(xs); ax.set_xticklabels(['round 0\n(base)', 'round 1', 'round 2'], fontsize=6.8)
    ax.set_xlabel('teacher-student refinement'); ax.set_ylabel('Test IoU')
    ax.set_ylim(0.60, 0.85)
    ax.set_title('(b) Iterative refinement (fails)', fontsize=8, loc='left')
    # (c) in vs out failure rate
    ax = fig.add_subplot(gs[2])
    ax.bar([0, 1], [r49['clean_rate_mean'], r49['out_rate_mean']], 0.55,
           color=[P['neutral_light'], P['neutral_mid']], edgecolor='#333', lw=.5,
           label='outside edit / clean')
    ax.bar([2], [r49['in_rate_mean']], 0.55, color=P['blue_main'], edgecolor='#333', lw=.5,
           label='inside edit')
    for i, v in enumerate([r49['clean_rate_mean'], r49['out_rate_mean'], r49['in_rate_mean']]):
        ax.text(i, v + 0.012, f'{v:.3f}', ha='center', fontsize=6.5)
    ax.axhline(0.5, color=P['red'], ls=':', lw=1)
    ax.text(0.05, 0.515, 'random level', color=P['red'], fontsize=6)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['clean', 'outside', 'inside'], fontsize=6.8)
    ax.set_ylabel('Block failure rate'); ax.set_ylim(0, 0.62)
    ax.set_title(f"(c) Failure saturation ({r49['consistency_frac_in_gt_out']:.0%} of images)", fontsize=8, loc='left')
    save(fig, 'figD6_mechanism')


if __name__ == '__main__':
    figD1(); figD2(); figD3(); figD4(); figD5(); figD6()
    print('ALL D2 FIGURES EXPORTED')
