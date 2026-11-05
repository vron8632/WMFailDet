#!/usr/bin/env python3
"""r14b 多尺度定位: 在 4x4 块上再嵌一层 check, 把定位分辨率从 8px 提到 4px.

背景: r14 证明后处理救不了 (背景归一化会抹掉大编辑区, 引导滤波会削峰),
      8x8 块级失败图的 IoU 上限就在 0.60 左右. 真正的瓶颈是空间分辨率.

方案: 同一张图上嵌两层独立信号
  L8: 8x8 块 (1,1)  <- 现有检测层 (检测/同步/定位用)
  L4: 4x4 块 (1,1)  <- 新增细化层 (只用于定位边界)
两个 key 由 per-image key + 不同 info 派生, 互不相关.

必须验证:
  1 4x4 层的 clean 失败率能否 ≈25% (太低/太高都不可用);
  2 嵌 4x4 层是否破坏 8x8 层 (PSNR / L8 clean fail);
  3 编辑区/区外是否仍分离 (50% vs 25%);
  4 定位 IoU 是否真的提升 (4x4 失败图 vs 8x8 失败图).

输出: results/r14b_multiscale_<ts>.json
用法: apjf-python r14b_multiscale.py [--n 120]
"""
import sys, os, json, argparse, datetime, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
MASTER = b'\x9a' + bytes(range(31))
PAY, CHK = (1, 0), (1, 1)


def layer_key(key, tag):
    return ks.hkdf_sha256(key, b'\x00' * 32, f'wm/layer/{tag}/v1'.encode(), 32)


def embed_bs(img, key, bs, delta, coef=(1, 1)):
    """在 bs x bs 块上嵌入 check (只写 check, 不写 payload)."""
    H, W = img.shape[0] // bs * bs, img.shape[1] // bs * bs
    n_h, n_w = H // bs, W // bs
    pat = ks.check_pattern(key, n_h, n_w, 'coord')
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)[:H, :W].copy()
    for i in range(n_h):
        for j in range(n_w):
            d = cv2.dct(y[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs])
            c = d[coef]
            d[coef] = (np.floor(c / delta) * delta + delta / 2) if pat[i, j] == 1 \
                else np.round(c / delta) * delta
            y[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs] = cv2.idct(d)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    out = ycrcb.copy()
    out[:H, :W, 0] = np.clip(y, 0, 255)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB).astype(np.uint8)


def fail_bs(img, key, bs, delta, coef=(1, 1)):
    H, W = img.shape[0] // bs * bs, img.shape[1] // bs * bs
    n_h, n_w = H // bs, W // bs
    pat = ks.check_pattern(key, n_h, n_w, 'coord')
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)[:H, :W]
    f = np.zeros((n_h, n_w), np.float32)
    for i in range(n_h):
        for j in range(n_w):
            c = cv2.dct(y[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs])[coef]
            f[i, j] = 1.0 if ks.qim_bit(c, delta) != pat[i, j] else 0.0
    return f


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def loc_iou(fail, bbox, bs, k=5, t=0.38, min_area=8):
    H, W = fail.shape[0] * bs, fail.shape[1] * bs
    f = ks.smooth_fail_map(fail, k) if bs == 8 else cv2.blur(fail, (k, k))
    pred = f > t
    p = pred.astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
    keep = np.zeros_like(p)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == c] = 1
    pred_px = np.repeat(np.repeat(keep.astype(bool), bs, 0), bs, 1)
    x1, y1, x2, y2 = bbox
    g = np.zeros((H, W), bool)
    g[int(round(y1 * H)):int(round(y2 * H)), int(round(x1 * W)):int(round(x2 * W))] = True
    inter, union = (pred_px & g).sum(), (pred_px | g).sum()
    return float(inter / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--delta4', type=float, nargs='+', default=[15, 20, 25])
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
    print(f'\n{"配置":26s} {"L8cl":>6s} {"L4cl":>6s} {"PSNR":>6s} {"L8in/out":>12s} '
          f'{"L4in/out":>12s} {"IoU8":>6s} {"IoU4":>6s}')
    for d4 in args.delta4:
        for c4 in [(1, 1), (0, 1), (1, 0)]:
            l8c, l4c, ps, i8, o8, i4, o4, iou8, iou4 = [], [], [], [], [], [], [], [], []
            for sid, o, a, bbox, key in items:
                k8 = layer_key(key, 'bs8')
                k4 = layer_key(key, 'bs4')
                wm = ks.embed_keyed(o, k8, 15, PAY, CHK, pattern='coord')
                wm2 = embed_bs(wm, k4, 4, d4, c4)
                att, box = paste_edit(wm2, a, bbox)

                f8c = fail_bs(wm2, k8, 8, 15, CHK)
                f4c = fail_bs(wm2, k4, 4, d4, c4)
                f8a = fail_bs(att, k8, 8, 15, CHK)
                f4a = fail_bs(att, k4, 4, d4, c4)

                H, W = o.shape[0] // 8 * 8, o.shape[1] // 8 * 8
                x1, y1, x2, y2 = bbox
                bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
                bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
                g8 = np.zeros(f8a.shape, bool)
                g8[by1 // 8:(by2 + 7) // 8, bx1 // 8:(bx2 + 7) // 8] = True
                g4 = np.zeros(f4a.shape, bool)
                g4[by1 // 4:(by2 + 3) // 4, bx1 // 4:(bx2 + 3) // 4] = True

                l8c.append(float(f8c.mean()))
                l4c.append(float(f4c.mean()))
                ps.append(psnr(o, wm2))
                i8.append(float(f8a[g8].mean()))
                o8.append(float(f8a[~g8].mean()))
                i4.append(float(f4a[g4].mean()))
                o4.append(float(f4a[~g4].mean()))
                iou8.append(loc_iou(f8a, bbox, 8, 5, 0.38, 8))
                iou4.append(loc_iou(f4a, bbox, 4, 5, 0.38, 32))
            tag = f'd4={d4}_c4={c4}'
            res[tag] = {
                'clean_fail_L8': float(np.mean(l8c)), 'clean_fail_L4': float(np.mean(l4c)),
                'psnr': float(np.mean(ps)),
                'in_out_L8': [float(np.mean(i8)), float(np.mean(o8))],
                'in_out_L4': [float(np.mean(i4)), float(np.mean(o4))],
                'iou_L8': float(np.mean(iou8)), 'iou_L4': float(np.mean(iou4)),
            }
            r = res[tag]
            print(f'{tag:26s} {r["clean_fail_L8"]*100:5.1f}% {r["clean_fail_L4"]*100:5.1f}% '
                  f'{r["psnr"]:5.1f} {r["in_out_L8"][0]*100:5.1f}/{r["in_out_L8"][1]*100:5.1f}% '
                  f'{r["in_out_L4"][0]*100:5.1f}/{r["in_out_L4"][1]*100:5.1f}% '
                  f'{r["iou_L8"]:6.3f} {r["iou_L4"]:6.3f}')

    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n': len(items),
                    'note': 'L8=8x8 checkpoint (检测层); L4=4x4 细化层; IoU 像素级'},
           'results': res, 'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r14b_multiscale_{fn}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n结果: {fp}  ({out["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
