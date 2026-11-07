#!/usr/bin/env python3
"""Manuscript data figures (fig1/3/4/5/6) — Nature-figure standard, Python backend.

FIGURE CONTRACTS (core/contract.md five-point form)
---------------------------------------------------
Fig.1  (image plate + quant, 4 panels, 183 mm)
  Claim: watermark failure is spatially concentrated on the edited region.
  Panels: (a) host image [object] -> (b) + watermark [invisibility] ->
          (c) edited image with GT/alarm boxes [coincidence] -> (d) fail map [concentration].

Fig.3  (quantitative grid, 1 panel, 89 mm)
  Claim: the two statistics are complementary around the 1% FPR operating point.
  Panels: single FPR–TPR curve pair with operating markers.

Fig.4  (quantitative grid, 2 panels, 183 mm)
  Claim: the 4x4 refinement layer closes the small-edit coverage gap.
  Panels: (a) coverage by edit size, 8x8 vs 4x4 [main effect] ->
          (b) IoU for the two layers [precision cost].

Fig.5  (quantitative grid, 1 panel, 89 mm)
  Claim: block-level IoU is limited by the block/image ratio, not by the method.
  Panels: single line pair across 256/512/1024 with trained-method references.

Fig.6  (quantitative grid, 1 panel, 120 mm)
  Claim: contrast is immune to uniform perturbations while blur/sharpen alarm.
  Panels: single FPR spectrum, bars coloured by perturbation family.

Backend: Python (matplotlib) exclusively. Exports: PDF (vector) + PNG (600 dpi).
QA: panel-alignment gate, PDF text-size audit, collision audit.
"""
import os, glob, json, sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

SKILL_SCRIPTS = '/home/oyp/.reasonix/skills/nature-figure/scripts'
if SKILL_SCRIPTS not in sys.path:
    sys.path.insert(0, SKILL_SCRIPTS)
from audit_panel_alignment import require_matplotlib_panel_alignment

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit

R = f'{BASE}/results'
FIG = f'{BASE}/paper/latex/figs'
os.makedirs(FIG, exist_ok=True)
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
DELTA = 15
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "figure.dpi": 150,
})

# Restrained palette: one neutral family, one signal family, one accent family.
P = {
    'neutral_dark': '#4D4D4D',
    'neutral_mid': '#767676',
    'neutral_light': '#CFCECE',
    'blue_main': '#0F4D92',
    'blue_soft': '#3775BA',
    'green': '#8BCF8B',
    'green_dark': '#2E7D32',
    'red': '#B64342',
    'gold': '#C9A227',
    'orange_soft': '#E8A15D',
    'orange_dark': '#B5651D',
    'teal': '#42949E',
    'violet': '#9A4D8E',
}


def last(pat):
    fs = sorted(glob.glob(f'{R}/{pat}'))
    return json.load(open(fs[-1])) if fs else None


def save_pub(fig, name, multipanel=False):
    if multipanel:
        require_matplotlib_panel_alignment(
            fig, json_out=f'{FIG}/{name}.alignment.json',
            overlay_svg=f'{FIG}/{name}.alignment.svg',
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG}/{name}.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('exported', name)


# =====================================================================
# Fig.1  motivate: host -> +wm -> edited(with boxes) -> fail map
# =====================================================================
def fig1():
    npz = np.load(sorted(glob.glob(f'{BASE}/data/failmaps_r7_*.npz'))[-1], allow_pickle=True)
    meta = json.load(open(f'{AS}/meta.json'))
    bbox_all = {s['id']: s['bbox'] for s in meta['samples']}

    def post(pred, min_area=8):
        p = pred.astype(np.uint8)
        p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
        keep = np.zeros_like(p)
        for c in range(1, n):
            if st[c, cv2.CC_STAT_AREA] >= min_area:
                keep[lab == c] = 1
        return keep.astype(bool)

    H = W = 256
    best = None
    for i in range(len(npz['ids'])):
        sid = str(npz['ids'][i])
        x1, y1, x2, y2 = bbox_all[sid]
        gt = np.zeros((H, W), bool)
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = True
        if not (0.12 <= gt.mean() <= 0.40):
            continue
        pred = post(ks.smooth_fail_map(npz['fail'][i], 9) > 0.38, 8)
        pred_px = np.repeat(np.repeat(pred, 8, 0), 8, 1)[:H, :W]
        inter = (pred_px & gt).sum()
        iou = inter / max(1, (pred_px | gt).sum())
        rec = inter / max(1, gt.sum())
        if best is None or (iou, rec) > best[0]:
            best = ((iou, rec), i, sid, npz['fail'][i], pred)

    (score, bi, bid, fail_i, pred_i) = best
    o = np.array(Image.open(f'{AS}/images/orig_{bid}.png').convert('RGB'))
    a = np.array(Image.open(f'{AS}/images/att_{bid}.png').convert('RGB'))
    key = ks.image_key(MASTER, bid)
    wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
    bm = [s for s in meta['samples'] if s['id'] == bid][0]
    att, _ = paste_edit(wm, a, bm['bbox'])
    x1, y1, x2, y2 = bm['bbox']

    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.05),
                           gridspec_kw={'wspace': 0.04})
    ax[0].imshow(o, aspect='equal')
    ax[1].imshow(wm, aspect='equal')
    ax[2].imshow(att, aspect='equal')
    ax[2].add_patch(plt.Rectangle((x1*W, y1*H), (x2-x1)*W, (y2-y1)*H,
                                  fill=False, edgecolor=P['red'], lw=2.2, zorder=5))
    pys, pxs = np.where(pred_i > 0)
    if len(pys):
        pb = (pxs.min()*8, pys.min()*8, (pxs.max()+1)*8, (pys.max()+1)*8)
        ax[2].add_patch(plt.Rectangle((pb[0], pb[1]), pb[2]-pb[0], pb[3]-pb[1],
                                      fill=False, edgecolor=P['green_dark'], lw=2.2,
                                      ls=(0, (4, 2)), zorder=6))
    hm = ax[3].imshow(fail_i, cmap='gray', vmin=0, vmax=1,
                      interpolation='nearest', aspect='equal')

    titles = ['(a) Host image', '(b) + Watermark', '(c) Edited',
              f'(d) Fail map (IoU {score[0]:.2f})']
    for a_, t in zip(ax, titles):
        a_.set_title(t, pad=3)
        a_.set_xticks([])
        a_.set_yticks([])
    for s in ax[2].spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
    cb = fig.colorbar(hm, ax=ax[3], fraction=0.046, pad=0.02)
    cb.set_label('failure prob. (binary)', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.6)
    save_pub(fig, 'fig1_motivation', multipanel=True)
    print('  fig1 sample:', bid, 'iou', f'{score[0]:.3f}', 'cov', f'{score[1]*100:.0f}%')


# =====================================================================
# Fig.3  detection operating curves
# =====================================================================
def fig3():
    r10 = last('r10_specificity_*.json')
    wc = r10['workcurve']
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for key, color, lab, mk in [('contrast', P['blue_main'], 'Contrast (top10 − median)', 'o'),
                                ('top', P['red'], 'Top-10% block mean', 's')]:
        ts = sorted(float(t) for t in wc[key].keys())
        fpr = [wc[key][f'{t}']['fpr_clean'] for t in ts]
        tpr = [wc[key][f'{t}'].get('tpr_local') or wc[key][f'{t}'].get('tpr_edited') for t in ts]
        ax.plot(fpr, tpr, marker=mk, color=color, ms=3.2, lw=1.1, label=lab)
    for key, color, thr, lab in [('contrast', P['blue_main'], '0.15', 'contrast op.'),
                                 ('top', P['red'], '0.4', 'top op.')]:
        op = wc[key].get(thr)
        if op:
            y = op.get('tpr_local') or op.get('tpr_edited')
            ax.plot([op['fpr_clean']], [y], marker='*', ms=11, color=color,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=8)
            ax.annotate(f'{lab}\n(FPR {100*op["fpr_clean"]:.0f}%)', (op['fpr_clean'], y),
                        textcoords='offset points', xytext=(8, -6), fontsize=6, color=color)
    ax.set_xlabel('False positive rate on benign images')
    ax.set_ylabel('Detection rate on edited images')
    ax.set_xlim(-0.004, 0.20)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, loc='lower right')
    save_pub(fig, 'fig3_operating')


# =====================================================================
# Fig.4  region alarm by edit size (8x8 vs 4x4)
# =====================================================================
def fig4():
    r18 = last('r18_region_alarm_*.json')
    r28 = last('r28_tier4x4_coco_*.json')
    r14b = last('r14b_multiscale_*.json')
    t8, t4 = r18['A_tiers'], r28['tiers']
    names = ['small', 'mid', 'large', 'full']
    labels = ['Small\n(<10%)', 'Mid\n(10–30%)', 'Large\n(30–90%)', 'Full\n(>90%)']

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5),
                           gridspec_kw={'width_ratios': [1.55, 1], 'wspace': 0.28})
    x = np.arange(4)
    w = 0.36
    cov8 = [100*t8[n]['cov_ge50'] for n in names]
    cov4 = [100*t4[n]['cov_ge50'] for n in names]
    ax[0].bar(x - w/2, cov8, w, color=P['neutral_light'], label='8×8 layer')
    ax[0].bar(x + w/2, cov4, w, color=P['orange_soft'], label='4×4 layer')
    # Label only the number the panel argues: the small-edit gap closure.
    ax[0].annotate(f'{cov8[0]:.0f}%', (x[0]-w/2, cov8[0]), xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=6.5, color=P['neutral_dark'])
    ax[0].annotate(f'{cov4[0]:.0f}%', (x[0]+w/2, cov4[0]), xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=6.5, color=P['orange_dark'], fontweight='bold')

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set_ylabel('Alarm coverage (≥50%) %')
    ax[0].set_ylim(0, 108)
    ax[0].set_title('(a) Coverage by edit size', pad=3)


    iou8 = r14b['results']['d4=15_c4=(1, 1)']['iou_L8']
    iou4 = r14b['results']['d4=15_c4=(1, 1)']['iou_L4']
    ax[1].bar([0, 1], [100*iou8, 100*iou4], 0.5, color=[P['neutral_light'], P['orange_soft']])
    ax[1].annotate(f'{100*iou8:.1f}', (0, 100*iou8), ha='center', va='bottom', fontsize=6.5)
    ax[1].annotate(f'{100*iou4:.1f}', (1, 100*iou4), ha='center', va='bottom', fontsize=6.5, color=P['orange_dark'])
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(['8×8', '4×4'])
    ax[1].set_ylabel('Localization IoU (%)')
    ax[1].set_ylim(0, 108)
    ax[1].set_title('(b) Refinement layer', pad=3)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=P['neutral_light'], label='8×8 layer'),
                        Patch(color=P['orange_soft'], label='4×4 layer')],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.14), fontsize=6,
               handlelength=1.2, columnspacing=1.2)
    save_pub(fig, 'fig4_alarm', multipanel=True)


# =====================================================================
# Fig.5  resolution sweep
# =====================================================================
def fig5():
    r19 = last('r19_resolution_sweep_*.json')
    scales = sorted(int(k) for k in r19['results'].keys())
    i8 = [r19['results'][str(s)]['8x8']['iou_mean'] for s in scales]
    i4 = [r19['results'][str(s)]['4x4']['iou_mean'] for s in scales]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(scales, i8, marker='o', color=P['neutral_dark'], ms=4, lw=1.2, label='8×8 layer')
    ax.plot(scales, i4, marker='s', color=P['blue_main'], ms=4, lw=1.2, label='4×4 layer')
    ax.axhline(0.947, color=P['red'], ls='--', lw=0.9)
    ax.axhline(0.913, color=P['gold'], ls=':', lw=0.9)
    ax.text(1030, 0.955, 'WAM (256)', fontsize=6, color=P['red'], ha='right')
    ax.text(1030, 0.921, 'EditGuard (256)', fontsize=6, color=P['gold'], ha='right')

    ax.set_xlabel('Image resolution (px)')
    ax.set_ylabel('Localization IoU')
    ax.set_xticks(scales)
    ax.set_xlim(180, 1120)
    ax.set_ylim(0.55, 1.02)
    ax.legend(fontsize=6.5, loc='lower right')
    save_pub(fig, 'fig5_resolution')


# =====================================================================
# Fig.6  specificity spectrum
# =====================================================================
def _nice_name(k):
    m = {'jpeg95': 'JPEG 95', 'jpeg85': 'JPEG 85', 'jpeg75': 'JPEG 75', 'jpeg50': 'JPEG 50',
         'noise2': 'noise \u03c3=2', 'noise5': 'noise \u03c3=5', 'noise10': 'noise \u03c3=10',
         'blur3': 'blur r=3', 'sharpen': 'sharpen',
         'resize_round_0.8': 'resize 0.8\u21921.0', 'crop_arb_0.9': 'crop 0.9 (arbitrary)',
         'rotate_1': 'rotate 1\u00b0', 'brightness_1.05': 'brightness 1.05',
         'saturation_1.1': 'saturation 1.1', 'screenshot_sim': 'screenshot sim'}
    return m.get(k, k)


def fig6():
    r10 = last('r10_specificity_*.json')
    sm = r10['summary']
    groups = [
        ('JPEG', ['jpeg95', 'jpeg85', 'jpeg75', 'jpeg50'], P['blue_main']),
        ('Noise', ['noise2', 'noise5', 'noise10'], P['teal']),
        ('Blur / sharpen', ['blur3', 'sharpen'], P['red']),
        ('Geometry', ['resize_round_0.8', 'crop_arb_0.9', 'rotate_1'], P['gold']),
        ('Other', ['brightness_1.05', 'saturation_1.1', 'screenshot_sim'], P['neutral_mid']),
    ]
    names, cons, tops, cols = [], [], [], []
    for g, keys, c in groups:
        for k in keys:
            if k in sm:
                names.append(_nice_name(k))
                cons.append(100*sm[k]['contrast']['fpr'])
                tops.append(100*sm[k]['top']['fpr'])
                cols.append(c)
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.15), sharey=True,
                           gridspec_kw={'wspace': 0.06})
    ax[0].barh(y, cons, 0.62, color=cols)
    ax[1].barh(y, tops, 0.62, color=cols)
    for a_, vals, ttl, xmax in ((ax[0], cons, 'contrast FPR (%)', 112),
                                (ax[1], tops, 'top-10% FPR (%)', 100)):
        a_.axvline(1.0, color=P['neutral_mid'], ls='--', lw=0.8)
        a_.set_xlim(0, xmax)
        a_.set_xlabel(ttl, fontsize=9)
        a_.tick_params(labelsize=8)
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=8.5)
    ax[1].tick_params(axis='y', labelleft=False)
    handles = [plt.Line2D([], [], marker='s', ls='', color=c, ms=5, label=g)
               for g, _, c in groups]
    handles.append(plt.Line2D([], [], color=P['neutral_mid'], ls='--', lw=0.8, label='benign (1%)'))
    ax[1].legend(handles=handles, fontsize=8, loc='upper center',
                 bbox_to_anchor=(0.42, -0.20), ncol=3, handletextpad=0.3,
                 labelspacing=0.3, columnspacing=1.0, borderaxespad=0.0)
    ax[0].set_title('(a) contrast FPR', pad=4)
    ax[1].set_title('(b) top-10% FPR', pad=4)
    save_pub(fig, 'fig6_specificity')


if __name__ == '__main__':
    fig1()
    fig3()
    fig4()
    fig5()
    fig6()
    print('ALL FIGURES EXPORTED (PDF + PNG)')
