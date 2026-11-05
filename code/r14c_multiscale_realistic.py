#!/usr/bin/env python3
"""r14c 多尺度定位在真实化条件下是否还成立（4x4 层是高频, JPEG 可能先杀它）.

r14b: 4x4 细化层把定位 IoU 从 0.587(8x8) 提到 0.738(4x4), clean 失败率 25%, PSNR 42.8dB.
但 4x4 块的 (1,1) 对应 0.25 cycle/px, 属于 JPEG 量化重灾区. 若 JPEG75 下 4x4 层崩掉,
"多尺度定位"就只是无损场景的纸面提升. 本实验就是查这件事.

条件: 硬 paste / 羽化4 / JPEG95/85/75 / 重编码(0.9+85)
指标: L8 IoU, L4 IoU, 两层 in/out 失败率, 并给出"4x4 层是否值得"(相对 L8 的增益).

输出: results/r14c_multiscale_realistic_<ts>.json
用法: apjf-python r14c_multiscale_realistic.py [--n 150]
"""
import sys, os, json, argparse, datetime, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r8d_sync_abstain import low_quantile
from r14b_multiscale import layer_key, embed_bs, fail_bs, loc_iou, psnr
from r13_realistic_protocol import jpeg, reencode, paste_alpha

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
MASTER = b'\x9a' + bytes(range(31))
PAY, CHK = (1, 0), (1, 1)
D4 = 15
C4 = (1, 1)


def conds():
    return [
        ('hard', lambda wm, att, b: paste_editor(wm, att, b, 0)),
        ('feather4', lambda wm, att, b: paste_editor(wm, att, b, 4)),
        ('jpeg95', lambda wm, att, b: jpeg(paste_editor(wm, att, b, 0), 95)),
        ('jpeg85', lambda wm, att, b: jpeg(paste_editor(wm, att, b, 0), 85)),
        ('jpeg75', lambda wm, att, b: jpeg(paste_editor(wm, att, b, 0), 75)),
        ('reenc_09_q85', lambda wm, att, b: reencode(paste_editor(wm, att, b, 0), 0.9, 85)),
    ]


def paste_editor(wm, att, bbox, radius):
    return paste_alpha(wm, att, bbox, radius)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=150)
    args = ap.parse_args()
    t0 = time.time()

    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:args.n]
    items = []
    for s in samples:
        sid = s['id']
        try:
            o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        items.append((sid, o, a, s['bbox'], ks.image_key(MASTER, sid)))
    print(f'载入 {len(items)} 图')

    res = {}
    print(f'\n{"条件":16s} {"IoU8":>6s} {"IoU4":>6s} {"Δ":>6s} {"L8 in/out":>12s} '
          f'{"L4 in/out":>12s} {"L4 clean":>9s}')
    for name, fn in conds():
        i8, i4, c8, c4, in8, out8, in4, out4 = [], [], [], [], [], [], [], []
        l4low, l8low = [], []
        for sid, o, a, bbox, key in items:
            k8, k4 = layer_key(key, 'bs8'), layer_key(key, 'bs4')
            wm = ks.embed_keyed(o, k8, 15, PAY, CHK, pattern='coord')
            wm2 = embed_bs(wm, k4, 4, D4, C4)
            att = fn(wm2, a, bbox)
            att = r8.crop8(att)
            f8c = fail_bs(wm2, k8, 8, 15, CHK)
            f4c = fail_bs(wm2, k4, 4, D4, C4)
            f8a = fail_bs(att, k8, 8, 15, CHK)
            f4a = fail_bs(att, k4, 4, D4, C4)

            H, W = o.shape[0] // 8 * 8, o.shape[1] // 8 * 8
            x1, y1, x2, y2 = bbox
            bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
            bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
            g8 = np.zeros(f8a.shape, bool)
            g8[by1 // 8:min((by2 + 7) // 8, f8a.shape[0]),
               bx1 // 8:min((bx2 + 7) // 8, f8a.shape[1])] = True
            g4 = np.zeros(f4a.shape, bool)
            g4[by1 // 4:min((by2 + 3) // 4, f4a.shape[0]),
               bx1 // 4:min((bx2 + 3) // 4, f4a.shape[1])] = True

            c8.append(float(f8c.mean()))
            c4.append(float(f4c.mean()))
            if g8.any():
                in8.append(float(f8a[g8].mean()))
            if (~g8).any():
                out8.append(float(f8a[~g8].mean()))
            if g4.any():
                in4.append(float(f4a[g4].mean()))
            if (~g4).any():
                out4.append(float(f4a[~g4].mean()))
            i8.append(loc_iou(f8a, bbox, 8, 5, 0.38, 8))
            i4.append(loc_iou(f4a, bbox, 4, 5, 0.38, 32))
            l4low.append(low_quantile(f4a))
            l8low.append(low_quantile(f8a))

        res[name] = {
            'iou_L8': float(np.mean(i8)), 'iou_L4': float(np.mean(i4)),
            'gain': float(np.mean(i4) - np.mean(i8)),
            'clean_L8': float(np.mean(c8)), 'clean_L4': float(np.mean(c4)),
            'in_out_L8': [float(np.mean(in8)), float(np.mean(out8))],
            'in_out_L4': [float(np.mean(in4)), float(np.mean(out4))],
            'low10_L4': float(np.mean(l4low)), 'low10_L8': float(np.mean(l8low)),
        }
        r = res[name]
        print(f'{name:16s} {r["iou_L8"]:6.3f} {r["iou_L4"]:6.3f} {r["gain"]:+6.3f} '
              f'{r["in_out_L8"][0]*100:5.1f}/{r["in_out_L8"][1]*100:5.1f}% '
              f'{r["in_out_L4"][0]*100:5.1f}/{r["in_out_L4"][1]*100:5.1f}% '
              f'{r["clean_L4"]*100:8.1f}%  low10(L4)={r["low10_L4"]*100:5.1f}%')

    verdict = {
        'multiscale_gain_hard': res['hard']['gain'],
        'multiscale_gain_jpeg85': res['jpeg85']['gain'],
        'multiscale_gain_jpeg75': res['jpeg75']['gain'],
        'multiscale_gain_reenc': res['reenc_09_q85']['gain'],
        'survives_jpeg75': bool(res['jpeg75']['iou_L4'] > res['jpeg75']['iou_L8'] + 0.05),
    }
    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n': len(items),
                    'delta4': D4, 'coef4': C4},
           'conditions': res, 'verdict': verdict,
           'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r14c_multiscale_realistic_{fn}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n判定: {verdict}')
    print(f'结果: {fp}  ({out["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
