#!/usr/bin/env python3
"""r19 分辨率消融: 定位 IoU 有多少是"被 256x256 小图吃掉的".

现象: 8x8 块在 256 图上 IoU 0.61, 4x4 到 0.74; 而块级方法的边界误差固定为块大小,
      所以在更大图上, 相对误差 (块/图) 变小, IoU 应该上升.

本实验把同一批图 + 同一编辑缩放到 256/512/1024, 用同一套方法 (8x8 + 4x4) 定位:
  - 若 IoU 随分辨率单调上升 -> 说明 0.61/0.74 是"256 小图 + 8px 块"的联合限制,
    真实高清图上会更好, 论文里必须写清"分辨率-定位"关系;
  - 若不上平 -> 说明限制在方法本身, 不能拿分辨率当解释.

输出: results/r19_resolution_sweep_<ts>.json
用法: apjf-python r19_resolution_sweep.py [--n 150]
"""
import sys, os, json, argparse, datetime, time

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
SCALES = [256, 512, 1024]


def post(pred, min_area):
    p = pred.astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
    keep = np.zeros_like(p)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == c] = 1
    return keep.astype(bool)


def metrics(pred_blocks, bs, bbox, H, W):
    pred_px = np.repeat(np.repeat(pred_blocks, bs, 0), bs, 1)
    x1, y1, x2, y2 = bbox
    gx1, gy1, gx2, gy2 = int(round(x1*W)), int(round(y1*H)), int(round(x2*W)), int(round(y2*H))
    gt = np.zeros((H, W), bool)
    gt[gy1:gy2, gx1:gx2] = True
    inter = float((pred_px & gt).sum())
    union = float((pred_px | gt).sum())
    return {'iou': inter / max(1.0, union),
            'recall': inter / max(1.0, float(gt.sum())),
            'pred_area': float(pred_px.mean()), 'gt_area': float(gt.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=150)
    args = ap.parse_args()
    t0 = time.time()
    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:args.n]

    res = {}
    for S in SCALES:
        rows8, rows4, t_s = [], [], time.time()
        for s in samples:
            sid = s['id']
            try:
                o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
                a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
            except FileNotFoundError:
                continue
            oS = cv2.resize(o, (S, S), interpolation=cv2.INTER_CUBIC)
            aS = cv2.resize(a, (S, S), interpolation=cv2.INTER_CUBIC)
            key = ks.image_key(MASTER, sid)
            k8, k4 = layer_key(key, 'bs8'), layer_key(key, 'bs4')
            wm = ks.embed_keyed(oS, k8, DELTA, PAY, CHK, pattern='coord')
            wm2 = embed_bs(wm, k4, 4, DELTA, (1, 1))
            att, _ = paste_edit(wm2, aS, s['bbox'])
            f8 = fail_bs(att, k8, 8, DELTA, CHK)
            f4 = fail_bs(att, k4, 4, DELTA, (1, 1))
            p8 = post(ks.smooth_fail_map(f8, 9) > 0.38, 8)
            p4 = post(cv2.blur(f4, (5, 5)) > 0.38, 32)
            rows8.append(metrics(p8, 8, s['bbox'], S, S))
            rows4.append(metrics(p4, 4, s['bbox'], S, S))
        def sm(rows):
            r = np.array([x['recall'] for x in rows])
            return {'n': len(rows),
                    'iou_mean': float(np.mean([x['iou'] for x in rows])),
                    'recall_mean': float(r.mean()), 'recall_median': float(np.median(r)),
                    'cov_ge50': float((r >= 0.5).mean()), 'cov_ge70': float((r >= 0.7).mean()),
                    'pred_area': float(np.mean([x['pred_area'] for x in rows])),
                    'gt_area': float(np.mean([x['gt_area'] for x in rows]))}
        res[str(S)] = {'8x8': sm(rows8), '4x4': sm(rows4), 'sec': round(time.time() - t_s, 1)}
        print(f'  {S:5d}: 8x8 IoU {res[str(S)]["8x8"]["iou_mean"]:.3f} / '
              f'4x4 IoU {res[str(S)]["4x4"]["iou_mean"]:.3f}  '
              f'4x4 cov>=50% {res[str(S)]["4x4"]["cov_ge50"]*100:.1f}%  '
              f'({res[str(S)]["sec"]}s)', flush=True)

    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n': args.n,
                    'scales': SCALES, 'note': '同一批图/同一编辑, 只改分辨率'},
           'results': res, 'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r19_resolution_sweep_{fn}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'结果: {fp}  ({out["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
