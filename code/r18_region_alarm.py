#!/usr/bin/env python3
"""r18 定位的正式口径: 区域告警 (region alarm) 而非像素级描边.

背景: 定位 IoU 0.61/0.74 与训练式 (WAM 0.95 / EditGuard 0.91) 差 0.2, 但差距主要来自
      块级边界误差 (±4~8 px), 而不是"找错地方". 手算显示: 预测面积 0.263 vs GT 0.260 (几乎无偏),
      区域召回 median 0.811, 86% 的图覆盖编辑区过半.
      -> 对"给人工复核提供线索"的取证场景, 区域告警比像素级 IoU 更贴近真实用法.

本实验把口径正式化 (可复现):
  A 全量 1200 (8x8 单层, 复用 r7 失败图):
      region recall / 覆盖 >=50/70/80% 的图占比 / 面积无偏 / 告警框 IoU 与覆盖率;
  B 子集 150 对比 8x8 vs 4x4 双层 (重跑嵌入):
      同一套告警指标, 看细化层是否把"覆盖"也一起提升;
  C 分层 small/mid/large/full + split-half.

输出: results/r18_region_alarm_<ts>.json
用法: apjf-python r18_region_alarm.py [--n 1200] [--n-ms 150]
"""
import sys, os, json, glob, argparse, datetime, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
DELTA = 15
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))
K_LOC, T_LOC = 9, 0.38


def post(pred, min_area=8):
    p = pred.astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
    keep = np.zeros_like(p)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == c] = 1
    return keep


def alarm_metrics(pred_blocks, bs, bbox, img_wh):
    """pred_blocks: bool 块矩阵; bs: 块大小; 返回告警指标."""
    W, H = img_wh
    pred_px = np.repeat(np.repeat(pred_blocks, bs, 0), bs, 1)
    x1, y1, x2, y2 = bbox
    gx1, gy1, gx2, gy2 = int(round(x1*W)), int(round(y1*H)), int(round(x2*W)), int(round(y2*H))
    gt = np.zeros((H, W), bool)
    gt[gy1:gy2, gx1:gx2] = True
    inter = float((pred_px & gt).sum())
    recall = inter / max(1.0, float(gt.sum()))
    precision = inter / max(1.0, float(pred_px.sum()))
    union = float((pred_px | gt).sum())
    iou = inter / max(1.0, union)
    # 告警框: 最大连通域的外接矩形
    n, lab, st, _ = cv2.connectedComponentsWithStats(pred_blocks.astype(np.uint8), connectivity=8)
    if n > 1:
        c = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh, _ = st[c]
        box = (bx*bs, by*bs, (bx+bw)*bs, (by+bh)*bs)
        bi = max(0, min(box[2], gx2) - max(box[0], gx1)) * max(0, min(box[3], gy2) - max(box[1], gy1))
        b_recall = bi / max(1.0, float(gt.sum()))
        b_prec = bi / max(1.0, float((box[2]-box[0])*(box[3]-box[1])))
        b_iou = bi / max(1.0, float(gt.sum()) + (box[2]-box[0])*(box[3]-box[1]) - bi)
    else:
        b_recall = b_prec = b_iou = 0.0
    return {'recall': recall, 'precision': precision, 'iou': iou,
            'pred_area': float(pred_px.mean()), 'gt_area': float(gt.mean()),
            'box_recall': b_recall, 'box_precision': b_prec, 'box_iou': b_iou}


def summarize(rows, tag):
    r = np.array([x['recall'] for x in rows])
    iou = np.array([x['iou'] for x in rows])
    pa = np.array([x['pred_area'] for x in rows])
    ga = np.array([x['gt_area'] for x in rows])
    br = np.array([x['box_recall'] for x in rows])
    bi = np.array([x['box_iou'] for x in rows])
    out = {
        'n': len(rows),
        'recall_mean': float(r.mean()), 'recall_median': float(np.median(r)),
        'cov_ge50': float((r >= 0.50).mean()), 'cov_ge70': float((r >= 0.70).mean()),
        'cov_ge80': float((r >= 0.80).mean()),
        'iou_mean': float(iou.mean()), 'iou_median': float(np.median(iou)),
        'pred_area': float(pa.mean()), 'gt_area': float(ga.mean()),
        'area_bias': float(pa.mean() - ga.mean()),
        'box_recall_median': float(np.median(br)), 'box_iou_median': float(np.median(bi)),
        'cov_ge50_ci95': float(1.96 * np.std((r >= 0.5).astype(float)) / np.sqrt(len(r))),
    }
    print(f'  {tag:16s} n={out["n"]:4d} recall {out["recall_mean"]:.3f}/'
          f'{out["recall_median"]:.3f}  cov>=50% {out["cov_ge50"]*100:5.1f}%  '
          f'cov>=70% {out["cov_ge70"]*100:5.1f}%  IoU {out["iou_mean"]:.3f}  '
          f'面积 {out["pred_area"]:.3f}/{out["gt_area"]:.3f}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1200)
    ap.add_argument('--n-ms', type=int, default=150)
    ap.add_argument('--npz', default=None)
    args = ap.parse_args()
    t0 = time.time()

    npz = args.npz or sorted(glob.glob(f'{BASE}/data/failmaps_r7_*.npz'))[-1]
    z = np.load(npz, allow_pickle=True)
    fail, ids = z['fail'], z['ids']
    meta = json.load(open(f'{AS}/meta.json'))
    bbox = {s['id']: s['bbox'] for s in meta['samples']}
    n = min(args.n, len(fail))
    H = W = fail.shape[1] * 8

    # ---------------- A 全量 8x8
    rows8 = []
    for i in range(n):
        sid = ids[i]
        pred = post(ks.smooth_fail_map(fail[i], K_LOC) > T_LOC, 8)
        m = alarm_metrics(pred.astype(bool), 8, bbox[sid], (W, H))
        m['id'] = sid
        rows8.append(m)
    half = len(rows8) // 2
    print('=== A 全量 8x8 (1200) ===')
    A = summarize(rows8, '8x8 全体')
    A_train = summarize(rows8[:half], '8x8 前半')
    A_test = summarize(rows8[half:], '8x8 后半')
    A['split_half_cov50'] = [A_train['cov_ge50'], A_test['cov_ge50']]

    # 分层
    tiers = {}
    areas = {s['id']: (s['bbox'][2]-s['bbox'][0])*(s['bbox'][3]-s['bbox'][1])
             for s in meta['samples']}
    for tag, lo, hi in [('small', 0, 0.10), ('mid', 0.10, 0.30),
                        ('large', 0.30, 0.90), ('full', 0.90, 1.01)]:
        rr = [r for r in rows8 if lo <= areas.get(r['id'], 0) < hi]
        if rr:
            tiers[tag] = summarize(rr, tag)

    # ---------------- B 子集 8x8 vs 4x4
    print('\n=== B 子集 150: 8x8 vs 4x4 ===')
    B8, B4 = [], []
    for s in meta['samples'][:args.n_ms]:
        sid = s['id']
        try:
            o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        key = ks.image_key(MASTER, sid)
        k8, k4 = layer_key(key, 'bs8'), layer_key(key, 'bs4')
        wm = ks.embed_keyed(o, k8, DELTA, PAY, CHK, pattern='coord')
        wm2 = embed_bs(wm, k4, 4, DELTA, (1, 1))
        att, _ = paste_edit(wm2, a, s['bbox'])
        f8 = fail_bs(att, k8, 8, DELTA, CHK)
        f4 = fail_bs(att, k4, 4, DELTA, (1, 1))
        p8 = post(ks.smooth_fail_map(f8, K_LOC) > T_LOC, 8)
        p4 = post(cv2.blur(f4, (5, 5)) > T_LOC, 32)
        B8.append(alarm_metrics(p8.astype(bool), 8, s['bbox'], (o.shape[1], o.shape[0])))
        B4.append(alarm_metrics(p4.astype(bool), 4, s['bbox'], (o.shape[1], o.shape[0])))
    Bs = {'8x8': summarize(B8, '8x8'), '4x4': summarize(B4, '4x4')}

    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n': n, 'n_ms': len(B8),
                    'workpoint': {'k': K_LOC, 't': T_LOC},
                    'note': 'recall = |pred ∩ GT| / |GT| (像素级); 告警框 = 最大连通域外接矩形'},
           'A_full_8x8': A, 'A_tiers': tiers, 'B_multiscale': Bs,
           'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r18_region_alarm_{fn}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n结论: 8x8 覆盖过半 {A["cov_ge50"]*100:.1f}% (split-half '
          f'{A["split_half_cov50"][0]*100:.1f}/{A["split_half_cov50"][1]*100:.1f}%), '
          f'recall 中位 {A["recall_median"]:.3f}, 面积偏差 {A["area_bias"]:+.3f}')
    print(f'结果: {fp}  ({out["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
