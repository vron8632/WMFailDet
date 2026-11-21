#!/usr/bin/env python3
"""r13 真实传播协议 + 轻度编辑谱（回应"协议太干净 / 检测的是篡改还是水印损坏"）.

现有协议的三个"太干净"之处:
  1 bbox 硬 paste: 编辑区像素 100% 来自 SD, 水印必然全死, 检测/定位都变简单;
  2 没有重压缩: 真实传播链几乎必然经过 JPEG / 平台缩放;
  3 编辑强度只有一档: 只有"整块重绘", 没有"局部调色/轻噪/轻模糊"这类轻度篡改.

本实验补三件事:
  A 真实化重绘: 羽化边缘 (r=2/4/8/16) + 整图 JPEG (95/85/75/50) + 重编码链 (resize+jpeg)
                + 组合 (羽化4 + jpeg85)
  B 轻度编辑谱: bbox 内亮度 ±5/10/20%、饱和度 ×1.1/1.2、高斯模糊 σ=0.8/1.5、噪声 σ=2/5、
                组合。用于回答"检测量的灵敏度下限在哪"——即多轻的改动还能被检出.
  C 方法学: 检测阈值用**前一半** clean 标定 (p99), TPR 在**后一半**测 (split-half),
           避免"阈值与评估同集".

指标: 每个条件 -> clean-vs-cond 检测 AUC、TPR@FPR1%、定位 IoU@0.5、定位 AUC、
      区内/区外失败率. 工作点固定为 G4 的 (k=9, t=0.38, open_close, min_area=8).

输出: results/r13_realistic_protocol_<ts>.json
用法: apjf-python r13_realistic_protocol.py [--n 600]
"""
import sys, os, json, argparse, datetime, time, io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

import keyed_scheme as ks
import r8_geometric as r8

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
DELTA = 15
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))
K_LOC, T_LOC = 9, 0.38


# ---------------------------------------------------------------- 基础
def bbox_px(bbox, H, W):
    x1, y1, x2, y2 = bbox
    return (int(round(x1 * W)), int(round(y1 * H)),
            int(round(x2 * W)), int(round(y2 * H)))


def paste_alpha(wm, att, bbox, radius=0):
    """羽化 paste: radius=0 退化为硬边界 (与 r6/r7 一致)."""
    H, W = wm.shape[:2]
    bx1, by1, bx2, by2 = bbox_px(bbox, H, W)
    if radius <= 0:
        out = wm.copy()
        out[by1:by2, bx1:bx2] = att[by1:by2, bx1:bx2]
        return out
    alpha = np.zeros((H, W), np.float32)
    alpha[by1:by2, bx1:bx2] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), radius / 2.0)
    out = (att.astype(np.float32) * alpha[..., None] +
           wm.astype(np.float32) * (1 - alpha[..., None]))
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg(im, q):
    buf = io.BytesIO()
    Image.fromarray(im).save(buf, format='JPEG', quality=q)
    return np.array(Image.open(buf).convert('RGB'))


def reencode(im, s, q):
    H, W = im.shape[:2]
    small = cv2.resize(im, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    return jpeg(back, q)


def light_edit(wm, bbox, kind, strength):
    """bbox 内的轻度像素操作 (不改语义, 只改像素)."""
    out = wm.copy()
    H, W = out.shape[:2]
    bx1, by1, bx2, by2 = bbox_px(bbox, H, W)
    roi = out[by1:by2, bx1:bx2].astype(np.float32)
    if kind == 'brightness':
        roi = roi * strength
    elif kind == 'saturation':
        hsv = cv2.cvtColor(np.clip(roi, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * strength, 0, 255)
        roi = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    elif kind == 'blur':
        roi = cv2.GaussianBlur(roi, (0, 0), strength)
    elif kind == 'noise':
        rng = np.random.RandomState(0)
        roi = roi + rng.normal(0, strength, roi.shape)
    elif kind == 'combo':
        roi = cv2.GaussianBlur(roi * 1.05, (0, 0), 0.8)
        rng = np.random.RandomState(0)
        roi = roi + rng.normal(0, 2, roi.shape)
    out[by1:by2, bx1:bx2] = np.clip(roi, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- 统计量/定位
def stats_of(fail):
    s = ks.smooth_fail_map(fail, K_LOC)
    vals = np.sort(s.ravel())
    n = max(1, int(0.10 * vals.size))
    top = float(vals[-n:].mean())
    return {'top': top, 'contrast': top - float(np.median(vals))}


def localize(fail):
    pred = ks.smooth_fail_map(fail, K_LOC) > T_LOC
    p = pred.astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
    keep = np.zeros_like(p)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= 8:
            keep[lab == c] = 1
    return keep.astype(bool)


def loc_metrics(fail, bbox, shape):
    n_h, n_w = fail.shape
    H, W = n_h * 8, n_w * 8
    bx1, by1, bx2, by2 = bbox_px(bbox, H, W)
    gt_px = np.zeros((H, W), bool)
    gt_px[by1:min(by2, H), bx1:min(bx2, W)] = True
    pred = localize(fail)
    pred_px = np.repeat(np.repeat(pred, 8, 0), 8, 1)
    inter, union = (pred_px & gt_px).sum(), (pred_px | gt_px).sum()
    iou = float(inter / union) if union else 0.0
    gt_b = np.zeros((n_h, n_w), bool)
    gt_b[by1 // 8:min((by2 + 7) // 8, n_h), bx1 // 8:min((bx2 + 7) // 8, n_w)] = True
    in_b, out_b = gt_b, ~gt_b
    f_sm = ks.smooth_fail_map(fail, K_LOC)
    auc = ks.auc(f_sm[out_b], f_sm[in_b]) if in_b.any() and out_b.any() else float('nan')
    return {'iou': iou, 'loc_auc': auc,
            'in_fail': float(fail[in_b].mean()) if in_b.any() else float('nan'),
            'out_fail': float(fail[out_b].mean()) if out_b.any() else float('nan')}


# ---------------------------------------------------------------- 条件
CONDS = [
    # A 重绘 (真实化)
    ('A1_hard_paste', lambda wm, att, b: paste_alpha(wm, att, b, 0)),
    ('A2_feather_r2', lambda wm, att, b: paste_alpha(wm, att, b, 2)),
    ('A3_feather_r4', lambda wm, att, b: paste_alpha(wm, att, b, 4)),
    ('A4_feather_r8', lambda wm, att, b: paste_alpha(wm, att, b, 8)),
    ('A5_feather_r16', lambda wm, att, b: paste_alpha(wm, att, b, 16)),
    ('B1_jpeg95', lambda wm, att, b: jpeg(paste_alpha(wm, att, b, 0), 95)),
    ('B2_jpeg85', lambda wm, att, b: jpeg(paste_alpha(wm, att, b, 0), 85)),
    ('B3_jpeg75', lambda wm, att, b: jpeg(paste_alpha(wm, att, b, 0), 75)),
    ('B4_jpeg50', lambda wm, att, b: jpeg(paste_alpha(wm, att, b, 0), 50)),
    ('C1_reenc_09_q85', lambda wm, att, b: reencode(paste_alpha(wm, att, b, 0), 0.9, 85)),
    ('C2_reenc_08_q75', lambda wm, att, b: reencode(paste_alpha(wm, att, b, 0), 0.8, 75)),
    ('D1_feather4_jpeg85', lambda wm, att, b: jpeg(paste_alpha(wm, att, b, 4), 85)),
    ('D2_feather4_jpeg75', lambda wm, att, b: jpeg(paste_alpha(wm, att, b, 4), 75)),
    # E 轻度编辑 (bbox 内, 不改语义)
    ('E1_bright_1.05', lambda wm, att, b: light_edit(wm, b, 'brightness', 1.05)),
    ('E2_bright_1.10', lambda wm, att, b: light_edit(wm, b, 'brightness', 1.10)),
    ('E3_bright_1.20', lambda wm, att, b: light_edit(wm, b, 'brightness', 1.20)),
    ('E4_sat_1.1', lambda wm, att, b: light_edit(wm, b, 'saturation', 1.1)),
    ('E5_sat_1.2', lambda wm, att, b: light_edit(wm, b, 'saturation', 1.2)),
    ('E6_blur_0.8', lambda wm, att, b: light_edit(wm, b, 'blur', 0.8)),
    ('E7_blur_1.5', lambda wm, att, b: light_edit(wm, b, 'blur', 1.5)),
    ('E8_noise_2', lambda wm, att, b: light_edit(wm, b, 'noise', 2)),
    ('E9_noise_5', lambda wm, att, b: light_edit(wm, b, 'noise', 5)),
    ('E10_combo', lambda wm, att, b: light_edit(wm, b, 'combo', 1.0)),
]

LIGHT = [c for c in CONDS if c[0].startswith('E')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600)
    args = ap.parse_args()
    t0 = time.time()

    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:args.n]
    rows = []
    for idx, s in enumerate(samples):
        sid = s['id']
        try:
            o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        if o.shape[0] < 128 or o.shape[1] < 128:
            continue
        key = ks.image_key(MASTER, sid)
        wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        f_clean = ks.fail_map(wm, key, DELTA, CHK, 'coord')
        rec = {'id': sid, 'clean': stats_of(f_clean), 'cond': {}}
        for name, fn in CONDS:
            im = r8.crop8(fn(wm, a, s['bbox']))
            fail = ks.fail_map(im, key, DELTA, CHK, 'coord')
            rec['cond'][name] = {**stats_of(fail), **loc_metrics(fail, s['bbox'], o.shape)}
        rows.append(rec)
        if (idx + 1) % 100 == 0:
            el = time.time() - t0
            print(f'  [{idx+1}/{len(samples)}] {el:.0f}s 预计剩余 {el/(idx+1)*(len(samples)-idx-1):.0f}s',
                  flush=True)

    n = len(rows)
    half = n // 2
    # split-half: 前一半 clean 定阈值, 后一半测 TPR
    thr = {}
    for st in ['top', 'contrast']:
        cs = np.array([r['clean'][st] for r in rows[:half]])
        thr[st] = float(np.percentile(cs, 99))
    print(f'\n阈值(前 {half} 张 clean p99): top={thr["top"]*100:.1f}%  contrast={thr["contrast"]*100:.1f}%')

    out_cond = {}
    print(f'\n{"条件":20s} {"top":>6s} {"AUCtop":>7s} {"TPR":>6s} {"con":>6s} {"AUCcon":>7s} '
          f'{"TPR":>6s} {"IoU":>6s} {"locAUC":>7s} {"in/out":>12s}')
    for name, _ in CONDS:
        v = [r['cond'][name] for r in rows]
        cs = [r['clean'] for r in rows]
        test = v[half:]
        d = {
            'top_mean': float(np.mean([x['top'] for x in v])),
            'contrast_mean': float(np.mean([x['contrast'] for x in v])),
            'iou': float(np.mean([x['iou'] for x in v])),
            'loc_auc': float(np.nanmean([x['loc_auc'] for x in v])),
            'in_fail': float(np.nanmean([x['in_fail'] for x in v])),
            'out_fail': float(np.nanmean([x['out_fail'] for x in v])),
            'detect_auc_top': ks.auc([c['top'] for c in cs], [x['top'] for x in v]),
            'detect_auc_contrast': ks.auc([c['contrast'] for c in cs],
                                          [x['contrast'] for x in v]),
            'tpr_top': float(np.mean([x['top'] > thr['top'] for x in test])),
            'tpr_contrast': float(np.mean([x['contrast'] > thr['contrast'] for x in test])),
            'family': name[0],
        }
        out_cond[name] = d
        print(f'{name:20s} {d["top_mean"]*100:5.1f}% {d["detect_auc_top"]:7.3f} '
              f'{d["tpr_top"]*100:5.1f}% {d["contrast_mean"]*100:5.1f}% '
              f'{d["detect_auc_contrast"]:7.3f} {d["tpr_contrast"]*100:5.1f}% '
              f'{d["iou"]:6.3f} {d["loc_auc"]:7.3f} '
              f'{d["in_fail"]*100:5.1f}/{d["out_fail"]*100:5.1f}')

    # 按族汇总
    fam = {}
    for name, d in out_cond.items():
        fam.setdefault(d['family'], []).append(d)
    fam_sum = {k: {'n_cond': len(v),
                   'iou_mean': float(np.mean([x['iou'] for x in v])),
                   'tpr_top_mean': float(np.mean([x['tpr_top'] for x in v])),
                   'auc_top_mean': float(np.mean([x['detect_auc_top'] for x in v])),
                   'loc_auc_mean': float(np.mean([x['loc_auc'] for x in v]))}
               for k, v in fam.items()}

    # 真实化代价: 相对于硬 paste
    base = out_cond['A1_hard_paste']
    cost = {name: {'iou_drop': base['iou'] - d['iou'],
                   'tpr_drop': base['tpr_top'] - d['tpr_top'],
                   'auc_drop': base['detect_auc_top'] - d['detect_auc_top']}
            for name, d in out_cond.items()}

    verdict = {
        'hard_paste_iou': base['iou'], 'hard_paste_tpr': base['tpr_top'],
        'worst_realistic_iou': min(out_cond[n]['iou'] for n in ['A4_feather_r8', 'B3_jpeg75',
                                                                'C2_reenc_08_q75', 'D2_feather4_jpeg75']),
        'jpeg75_tpr': out_cond['B3_jpeg75']['tpr_top'],
        'light_edit_detectable': {n: {'tpr': out_cond[n]['tpr_top'],
                                      'contrast_tpr': out_cond[n]['tpr_contrast'],
                                      'iou': out_cond[n]['iou']} for n, _ in LIGHT},
        'split_half_threshold': thr,
        'pass_realistic': bool(min(out_cond[n]['iou'] for n in
                                   ['A4_feather_r8', 'B3_jpeg75', 'D2_feather4_jpeg75']) > 0.45),
    }
    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n': n,
                    'workpoint': {'k': K_LOC, 't': T_LOC},
                    'split_half': True, 'attackset': 'attackset_20261001_222447'},
           'conditions': out_cond, 'family': fam_sum, 'cost_vs_hard': cost,
           'verdict': verdict, 'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r13_realistic_protocol_{fn}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n真实化代价(硬 paste 基线 IoU={base["iou"]:.3f}, TPR={base["tpr_top"]*100:.1f}%):')
    for name in ['A3_feather_r4', 'A4_feather_r8', 'B2_jpeg85', 'B3_jpeg75',
                 'C1_reenc_09_q85', 'D1_feather4_jpeg85', 'D2_feather4_jpeg75']:
        c = cost[name]
        print(f'  {name:20s} IoU {out_cond[name]["iou"]:.3f} (Δ{-c["iou_drop"]:+.3f})  '
              f'TPR {out_cond[name]["tpr_top"]*100:5.1f}% (Δ{-c["tpr_drop"]*100:+.1f}pt)')
    print(f'\n轻度编辑检出(Top/Contrast TPR, IoU):')
    for name, _ in LIGHT:
        print(f'  {name:18s} {out_cond[name]["tpr_top"]*100:5.1f}% / '
              f'{out_cond[name]["tpr_contrast"]*100:5.1f}%  IoU {out_cond[name]["iou"]:.3f}')
    print(f'\n结果: {fp}  ({out["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
