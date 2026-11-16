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
    d = json.load(open(f'{R}/r34_distill_matrix.json'))
    groups = [('Teacher\n8×8', d['teachers']['teacher8'], P['neutral_light']),
              ('Teacher\n4×4', d['teachers']['teacher4'], P['neutral_mid']),
              ('Student\n(passive)', d['students']['student_alarm4'], P['blue_main']),
              ('Fusion\nOR', d['fusion_or']['fusion_alarm4_or_teacher4'], P['teal']),
              ('Student\n(GT upper)', d['students']['student_gt'], P['neutral_dark'])]
    fig = plt.figure(figsize=(7.6, 2.9))
    gs = GridSpec(1, 2, width_ratios=[1.25, 1], wspace=0.28)
    # (a) 柱状 IoU + 覆盖
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(groups))
    ious = [g[1]['iou'] for g in groups]
    covs = [g[1]['cov50'] for g in groups]
    ax.bar(x, ious, 0.62, color=[g[2] for g in groups], edgecolor='#333', linewidth=0.5)
    for i, (iv, cv) in enumerate(zip(ious, covs)):
        ax.text(i, iv + 0.012, f'{iv:.3f}', ha='center', fontsize=7)
        ax.text(i, 0.03, f'cov\n{cv:.0%}', ha='center', fontsize=6, color='white')
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=7)
    ax.set_ylabel('Localization IoU'); ax.set_ylim(0, 1.0)
    ax.set_title('(a) Groups on COCO-256 (n=200)', fontsize=8, loc='left')
    # (b) 逐样本散点 teacher4 vs student
    ax = fig.add_subplot(gs[1])
    tp = np.array(d['students']['student_alarm4'].get('per_iou', []))
    # per_iou 存的是学生; 教师的未存 -> 用学生 vs 无法散点. 改画学生 per_iou 分布 + 教师均值线
    if len(tp):
        ax.hist(tp, bins=20, color=P['blue_main'], alpha=0.75, label='Student', edgecolor='white')
        ax.axvline(d['teachers']['teacher4']['iou'], color=P['red'], ls='--', lw=1.2,
                   label=f"Teacher 4×4 ({d['teachers']['teacher4']['iou']:.3f})")
        ax.axvline(d['students']['student_alarm4']['iou'], color=P['blue_main'], ls='-', lw=1.2,
                   label=f"Student ({d['students']['student_alarm4']['iou']:.3f})")
    ax.set_xlabel('Per-sample IoU'); ax.set_ylabel('Count')
    ax.set_title('(b) Per-sample IoU distribution', fontsize=8, loc='left')
    ax.legend(fontsize=6.5, frameon=False)
    save(fig, 'figD2_distill')


# =====================================================================
def figD3():
    d = json.load(open(f'{R}/r35_crossdomain.json'))
    doms = ['coco'] + ['butterfly', 'sat', 'doc', 'pokemon']
    names = ['COCO\n(in-domain)', 'Butterfly', 'Satellite', 'Document', 'Cartoon']
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    x = np.arange(len(doms)); w = 0.36
    if 'coco' in d:
        t = [d['coco']['teacher4']['iou']] + [d['domains'][k]['teacher4']['iou'] for k in doms[1:]]
        s = [d['coco']['student']['iou']] + [d['domains'][k]['student']['iou'] for k in doms[1:]]
    else:
        t = [d['domains'][k]['teacher4']['iou'] for k in doms]
        s = [d['domains'][k]['student']['iou'] for k in doms]
    ax.bar(x - w/2, t, w, color=P['neutral_mid'], label='Teacher 4×4 (keyed)', edgecolor='#333', lw=0.5)
    ax.bar(x + w/2, s, w, color=P['blue_main'], label='Student (passive)', edgecolor='#333', lw=0.5)
    for i in range(len(doms)):
        ax.text(i - w/2, t[i] + 0.015, f'{t[i]:.2f}', ha='center', fontsize=6.5)
        ax.text(i + w/2, s[i] + 0.015, f'{s[i]:.2f}', ha='center', fontsize=6.5,
                color=P['blue_main'] if s[i] >= 0.6 else P['red'])
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel('Localization IoU'); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    ax.set_title('Cross-domain generalization (trained on COCO only, n=60/domain)',
                 fontsize=8, loc='left')
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


if __name__ == '__main__':
    figD1(); figD2(); figD3(); figD4()
    print('ALL D2 FIGURES EXPORTED')
