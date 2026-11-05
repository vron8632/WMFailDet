#!/usr/bin/env python3
"""r20 真实高清验证: 在 640x480 的真实 IP2P 编辑样本上测定位 (不是放大图).

r19 证明 IoU 随分辨率上升 (256:0.61/0.74 -> 1024:0.87/0.84), 但"放大不增加信息"会被质疑.
本实验用 attackset_hires_sample 的**真实高清**样本 (640x480, IP2P 整图重绘 + 像素 mask paste),
同一套方法 (8x8 / 4x4) 直接跑, 与 256 的 attackset 对照.

输出: results/r20_hires_validation_<ts>.json
用法: apjf-python r20_hires_validation.py
"""
import sys, os, json, argparse, datetime, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

import keyed_scheme as ks
import r8_geometric as r8
from r14b_multiscale import layer_key, embed_bs, fail_bs

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRES = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
         '/results/attackset_hires_sample')
DELTA = 15
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))


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


def metrics(pred_blocks, bs, gt, k=5):
    """gt: 像素级 bool mask."""
    H, W = gt.shape
    f = ks.smooth_fail_map(pred_blocks.astype(np.float32), k) if bs == 8 else cv2.blur(pred_blocks.astype(np.float32), (k, k))
    pred_px = np.repeat(np.repeat(f > 0.38, bs, 0), bs, 1)[:H, :W]
    inter = float((pred_px & gt).sum())
    union = float((pred_px | gt).sum())
    return {'iou': inter / max(1.0, union),
            'recall': inter / max(1.0, float(gt.sum())),
            'pred_area': float(pred_px.mean()), 'gt_area': float(gt.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    args = ap.parse_args()
    t0 = time.time()
    meta = json.load(open(f'{HIRES}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:args.n]

    rows8, rows4 = [], []
    for s in samples:
        sid = s['id']
        try:
            o = r8.crop8(np.array(Image.open(f'{HIRES}/images/{sid}_orig.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{HIRES}/images/{sid}_att.png').convert('RGB')))
            m = r8.crop8(np.array(Image.open(f'{HIRES}/images/{sid}_mask.png').convert('L')))
        except FileNotFoundError:
            continue
        gt = m > 127
        key = ks.image_key(MASTER, sid)
        k8, k4 = layer_key(key, 'bs8'), layer_key(key, 'bs4')
        wm = ks.embed_keyed(o, key=k8, delta=DELTA, pay_coef=PAY, check_coef=CHK, pattern='coord')
        wm2 = embed_bs(wm, k4, 4, DELTA, (1, 1))
        att = wm2.copy()
        att[gt] = a[gt]
        f8 = fail_bs(att, k8, 8, DELTA, CHK)
        f4 = fail_bs(att, k4, 4, DELTA, (1, 1))
        rows8.append(metrics(f8, 8, gt))
        rows4.append(metrics(f4, 4, gt))
        print(f'  {sid:10s} {o.shape[1]}x{o.shape[0]} 8x8 IoU {rows8[-1]["iou"]:.3f} '
              f'4x4 IoU {rows4[-1]["iou"]:.3f}  gt_area {gt.mean():.3f}')

    def sm(rows):
        r = np.array([x['recall'] for x in rows])
        return {'n': len(rows), 'iou_mean': float(np.mean([x['iou'] for x in rows])),
                'recall_mean': float(r.mean()), 'recall_median': float(np.median(r)),
                'cov_ge50': float((r >= 0.5).mean()),
                'pred_area': float(np.mean([x['pred_area'] for x in rows])),
                'gt_area': float(np.mean([x['gt_area'] for x in rows]))}

    res = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n': len(rows8),
                    'data': 'attackset_hires_sample (640x480, IP2P + 像素 mask paste)'},
           '8x8': sm(rows8), '4x4': sm(rows4),
           'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r20_hires_validation_{fn}.json'
    json.dump(res, open(fp, 'w'), indent=1)
    print(f'\n真实高清 (640x480): 8x8 IoU {res["8x8"]["iou_mean"]:.3f} '
          f'(cov>=50% {res["8x8"]["cov_ge50"]*100:.0f}%)  4x4 IoU {res["4x4"]["iou_mean"]:.3f} '
          f'(cov>=50% {res["4x4"]["cov_ge50"]*100:.0f}%)')
    print(f'结果: {fp}  ({res["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
