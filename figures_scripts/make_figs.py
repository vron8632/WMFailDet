#!/usr/bin/env python3
"""Generate all manuscript figures (English labels, Nature-style, 300 dpi)."""
import sys, os, json, glob
sys.path.insert(0,'code')
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f'{BASE}/results'
FIG = f'{BASE}/figs'
os.makedirs(FIG, exist_ok=True)
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/results/attackset_20261001_222447')
DELTA=15; PAY,CHK=(1,0),(1,1); MASTER=b'\x9a'+bytes(range(31))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
                     'font.size':8,'axes.linewidth':0.8,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':300})
C_BLUE='#2166ac'; C_RED='#b2182b'; C_GREEN='#1b7837'; C_GRAY='#878787'; C_OR='#e08214'

def post(pred, min_area=8):
    p = pred.astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
    keep = np.zeros_like(p)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == c] = 1
    return keep.astype(bool)

def last(p):
    fs=sorted(glob.glob(f'{R}/{p}')); return json.load(open(fs[-1])) if fs else None

def savefig(fig, name):
    fig.savefig(f'{FIG}/{name}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig); print('saved', name)

# ---------------- Fig.1 motivation (auto-pick a representative sample) ----------------
npz = np.load(sorted(glob.glob(f'{BASE}/data/failmaps_r7_*.npz'))[-1], allow_pickle=True)
meta = json.load(open(f'{AS}/meta.json'))
bbox_all = {s['id']: s['bbox'] for s in meta['samples']}
H256 = W256 = 256
best = None
for i in range(len(npz['ids'])):
    sid = str(npz['ids'][i]); fail = npz['fail'][i]
    x1,y1,x2,y2 = bbox_all[sid]
    gt = np.zeros((H256,W256),bool); gt[int(y1*H256):int(y2*H256),int(x1*W256):int(x2*W256)]=True
    gt_area = gt.mean()
    if not (0.12 <= gt_area <= 0.40): continue
    pred = post(ks.smooth_fail_map(fail,9) > 0.38, 8)
    pred_px = np.repeat(np.repeat(pred,8,0),8,1)[:H256,:W256]
    inter,union = (pred_px&gt).sum(),(pred_px|gt).sum()
    iou = inter/max(1,union); rec = inter/max(1,float(gt.sum()))
    if best is None or (iou,rec) > best[0]:
        best = ((iou,rec), i, sid, fail, pred, gt)
(score,bi,bid, fail_i, pred_i, gt_i) = best
o = np.array(Image.open(f'{AS}/images/orig_{bid}.png').convert('RGB'))
a = np.array(Image.open(f'{AS}/images/att_{bid}.png').convert('RGB'))
key = ks.image_key(MASTER, bid)
wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
bmeta = [s for s in meta['samples'] if s['id']==bid][0]
att, box = paste_edit(wm, a, bmeta['bbox'])
x1,y1,x2,y2 = bmeta['bbox']
W, H = W256, H256
fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.95))
ax[0].imshow(o); ax[0].set_title('(a) Host image', fontsize=8)
ax[1].imshow(wm); ax[1].set_title('(b) + embedded watermark', fontsize=8)
ax[2].imshow(att)
ax[2].add_patch(plt.Rectangle((x1*W,y1*H),(x2-x1)*W,(y2-y1)*H, fill=False, edgecolor='#b2182b', lw=2.5))
ax[2].add_patch(plt.Rectangle((x1*W,y1*H),(x2-x1)*W,(y2-y1)*H, fill=False, edgecolor='#1b7837', lw=2.5, ls='--'))
ax[2].set_title('(c) Edited (red: GT, green: alarm)', fontsize=7)
hm = ax[3].imshow(npz['fail'][bi], cmap='hot', vmin=0, vmax=1, interpolation='nearest')
ax[3].set_title(f'(d) Death map (IoU {score[0]:.2f}, cov {score[1]*100:.0f}%)', fontsize=7)
for a_ in ax:
    a_.set_xticks([]); a_.set_yticks([])
cb = fig.colorbar(hm, ax=ax[3], fraction=0.046, pad=0.02); cb.set_label('fail prob.', fontsize=7)
cb.ax.tick_params(labelsize=6)
fig.tight_layout(pad=0.4)
savefig(fig, 'fig1_motivation.png')
print('fig1 sample:', bid, 'iou', f"{score[0]:.3f}", 'gt_area', f"{gt_i.mean():.2f}")

# ---------------- Fig.3 detection operating curves ----------------
r10 = last('r10_specificity_*.json')
wc = r10['workcurve']
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for key_, color, lab in [('contrast', C_BLUE, 'Contrast (top10 − median)'),
                         ('top', C_RED, 'Top-10% block mean')]:
    xs = sorted(float(t) for t in wc[key_].keys())
    fpr = [wc[key_][f'{t}']['fpr_clean'] for t in xs]
    tpr = [wc[key_][f'{t}']['tpr_local'] if wc[key_][f'{t}'].get('tpr_local') is not None
           else wc[key_][f'{t}']['tpr_edited'] for t in xs]
    ax.plot(fpr, tpr, 'o-', color=color, ms=3, lw=1.2, label=lab)
    # recommended operating points (from text)
    if key_ == 'contrast':
        op = wc[key_].get('0.15')
        if op: ax.plot([op['fpr_clean']], [op['tpr_local'] or op['tpr_edited']], '*', color=color, ms=9, markeredgecolor='k', markeredgewidth=0.5); ax.annotate('contrast op.\n(FPR 1.7%)', (op['fpr_clean'], op['tpr_local'] or op['tpr_edited']), textcoords='offset points', xytext=(6,-14), fontsize=6)
    if key_ == 'top':
        op = wc[key_].get('0.41')
        if op: ax.plot([op['fpr_clean']], [op['tpr_edited']], '*', color=color, ms=9, markeredgecolor='k', markeredgewidth=0.5); ax.annotate('top op.\n(FPR 1.0%)', (op['fpr_clean'], op['tpr_edited']), textcoords='offset points', xytext=(6,6), fontsize=6)
    # annotate threshold at the elbow
ax.plot([0,1],[0,1],':',color=C_GRAY,lw=0.8)
ax.set_xlabel('False positive rate on benign images'); ax.set_ylabel('Detection rate on edited images')
ax.set_xlim(-0.02,0.55); ax.set_ylim(0,1.02)
ax.legend(fontsize=7, frameon=False, loc='lower right')
fig.tight_layout(pad=0.3); savefig(fig, 'fig3_operating.png')

# ---------------- Fig.4 region alarm by tier (3-color grouped) ----------------
r18 = last('r18_region_alarm_*.json'); r28 = last('r28_tier4x4_coco_*.json'); r14b = last('r14b_multiscale_*.json')
tiers8 = r18['A_tiers']; tiers4 = r28['tiers']
names = ['small','mid','large','full']
labels = ['Small\n(<10%)','Mid\n(10–30%)','Large\n(30–90%)','Full\n(>90%)']
tier_colors = ['#4477AA', '#66CCEE', '#EE6677', '#AA3377']
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
x = np.arange(4); w = 0.36
cov8 = [100*tiers8[n]['cov_ge50'] for n in names]
cov4 = [100*tiers4[n]['cov_ge50'] for n in names]
b1 = ax[0].bar(x - w/2, cov8, w, color='#BBBBBB', label='8×8 layer')
b2 = ax[0].bar(x + w/2, cov4, w, color=tier_colors, label='4×4 layer')
for xi, v in zip(x - w/2, cov8): ax[0].annotate(f'{v:.0f}', (xi, v), ha='center', va='bottom', fontsize=6)
for xi, v in zip(x + w/2, cov4): ax[0].annotate(f'{v:.0f}', (xi, v), ha='center', va='bottom', fontsize=6)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=7)
ax[0].set_ylabel('Alarm coverage (≥50%) %', fontsize=7); ax[0].set_ylim(0, 108)
ax[0].set_title('(a) Alarm coverage by edit size', fontsize=8)
ax[0].legend(fontsize=6.5, frameon=False, loc='upper left')
iou8_150 = r14b['results']['d4=15_c4=(1, 1)']['iou_L8']; iou4_150 = r14b['results']['d4=15_c4=(1, 1)']['iou_L4']
ax[1].bar([0, 1], [100*iou8_150, 100*iou4_150], 0.45, color=['#BBBBBB', '#4477AA'])
for xi, v in zip([0, 1], [100*iou8_150, 100*iou4_150]):
    ax[1].annotate(f'{v:.1f}', (xi, v), ha='center', va='bottom', fontsize=6.5)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(['8×8\n(150 imgs)', '4×4\n(150 imgs)'], fontsize=7)
ax[1].set_ylabel('Localization IoU (%)', fontsize=7); ax[1].set_ylim(0, 108)
ax[1].set_title('(b) 4×4 refinement', fontsize=8)
for a_ in ax: a_.set_ylim(0, 108)
fig.tight_layout(pad=0.4); savefig(fig, 'fig4_alarm.png')

# ---------------- Fig.5 resolution sweep ----------------
r19 = last('r19_resolution_sweep_*.json')
scales = sorted(int(k) for k in r19['results'].keys())
i8 = [r19['results'][str(s)]['8x8']['iou_mean'] for s in scales]
i4 = [r19['results'][str(s)]['4x4']['iou_mean'] for s in scales]
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.plot(scales, i8, 'o-', color=C_BLUE, ms=4, lw=1.3, label='8×8 layer')
ax.plot(scales, i4, 's-', color=C_GREEN, ms=4, lw=1.3, label='4×4 layer')
ax.axhline(0.947, color=C_RED, ls='--', lw=1.0)
ax.text(1000, 0.952, 'WAM (COCO 256)', fontsize=6.5, color=C_RED, ha='right')
ax.axhline(0.913, color=C_OR, ls=':', lw=1.0)
ax.text(1000, 0.868, 'EditGuard (COCO 256)', fontsize=6.5, color=C_OR, ha='right')
for s, v in zip(scales, i8): ax.annotate(f'{v:.3f}', (s, v), textcoords='offset points', xytext=(8, 7), fontsize=6.5, ha='left', color=C_BLUE)
for s, v in zip(scales, i4): ax.annotate(f'{v:.3f}', (s, v), textcoords='offset points', xytext=(8, -12), fontsize=6.5, ha='left', color=C_GREEN)
ax.set_xlabel('Image resolution (px)'); ax.set_ylabel('Localization IoU')
ax.set_xticks(scales); ax.set_ylim(0.55, 1.02); ax.set_xlim(180, 1150)
ax.legend(fontsize=7, frameon=False, loc='lower right')
fig.tight_layout(pad=0.3); savefig(fig, 'fig5_resolution.png')

# ---------------- Fig.6 specificity spectrum (grouped by perturbation family) ----------------
sm = r10['summary']
groups = {
    'JPEG':   ['jpeg95','jpeg85','jpeg75','jpeg50'],
    'Noise':  ['noise2','noise5','noise10'],
    'Blur/Sharp': ['blur3','sharpen'],
    'Geometry': ['resize_round_0.8','crop_arb_0.9','rotate_1'],
    'Other':  ['brightness_1.05','saturation_1.1','screenshot_sim'],
}
gcolors = {'JPEG':'#4477AA','Noise':'#66CCEE','Blur/Sharp':'#EE6677','Geometry':'#EEAA33','Other':'#BBBBBB'}
names, cons, tops, cols = [], [], [], []
for g in groups:
    for k in groups[g]:
        if k in sm:
            names.append(k); cons.append(100*sm[k]['contrast']['fpr']); tops.append(100*sm[k]['top']['fpr'])
            cols.append(gcolors[g])
y = np.arange(len(names))
fig, ax = plt.subplots(figsize=(5.2, 2.9))
ax.barh(y, cons, 0.62, color=cols, label='Contrast FPR')
ax.scatter(tops, y, marker='D', s=16, color='#B2182B', zorder=3, label='Top-10% FPR')
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
ax.axvline(1.0, color=C_GRAY, ls='--', lw=0.9)
ax.text(1.15, len(names)-0.2, 'benign (1%)', fontsize=6.5, color=C_GRAY)
for yi, c in zip(y, cons):
    if c > 12: ax.annotate(f'{c:.0f}%', (c, yi), textcoords='offset points', xytext=(4,-3), fontsize=6, color='#B2182B')
ax.set_xlabel('False alarm rate on benign processing (%)', fontsize=7)
ax.set_xscale('symlog', linthresh=1); ax.set_xlim(0, 130)
ax.legend(fontsize=6.5, frameon=False, loc='lower right')
fig.tight_layout(pad=0.3); savefig(fig, 'fig6_specificity.png')

print('ALL FIGS DONE')
