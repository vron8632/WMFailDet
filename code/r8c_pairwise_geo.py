#!/usr/bin/env python3
"""G3-c 相对编码能否扛住重采样? 块对差值 QIM vs 绝对 QIM.

r8b 结论: crop→resize / 1° 旋转下, 绝对 QIM 的失败率与 δ 完全无关
  (δ=15/25/40 全 ≈50%) -> 不是"信号太弱", 是**绝对量化格点被插值整体平移**,
  判决随机化. 反变换 + 同步搜索也只能部分恢复 (39.2%/43.2%).
  唯一对 δ 有反应的是"缩放再放大回原尺寸"(34.5→27.6%).

本实验测相对编码 (r1_pairwise_probe 的思路, 那里因 uint8 精度未验证成功):
  把 check 位编码在**相邻块 (1,1) 系数之差**上, 而不是单块系数的绝对格点.
  重采样对相邻块的低频系数施加相近的影响, 差值应更稳.
  为避免一个块被两个块对同时使用, 只取不相交块对 (2k, 2k+1) -> 容量减半.

对照: 同 δ 下绝对 QIM (r8b 已测, 这里重跑同一批图保证配对比较).

输出: results/r8c_pairwise_geo_<ts>.json
用法: apjf-python r8c_pairwise_geo.py [--n 120]
"""
import sys, os, json, argparse, datetime, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

import keyed_scheme as ks

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
DELTA = 15
CHK = (1, 1)
MASTER = b'\x9a' + bytes(range(31))


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def y_of(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)


def write_y(img, y):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    out = ycrcb.copy()
    out[:y.shape[0], :y.shape[1], 0] = np.clip(y, 0, 255)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB).astype(np.uint8)


# ---------------------------------------------------------------- 块对差值 QIM
def embed_pair(img, key, delta, chk=CHK):
    """check 位写在不相交块对 (2k,2k+1) 的 (1,1) 系数差上."""
    H, W = img.shape[0] // 8 * 8, img.shape[1] // 8 * 8
    n_h, n_w = H // 8, W // 8
    n_p = n_w // 2
    pat = ks.check_pattern(key, n_h, n_p, 'coord')
    y = y_of(img)[:H, :W].copy()
    for i in range(n_h):
        ds = [cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.float32))
              for j in range(n_w)]
        for k in range(n_p):
            a, b = 2 * k, 2 * k + 1
            d = ds[a][chk] - ds[b][chk]
            target = (np.floor(d / delta) * delta + delta / 2) if pat[i, k] == 1 \
                else np.round(d / delta) * delta
            e = target - d
            ds[a][chk] += e / 2
            ds[b][chk] -= e / 2
        for j in range(n_w):
            y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = cv2.idct(ds[j])
    return write_y(img, y)


def fail_pair(img, key, delta, chk=CHK, off=(0, 0)):
    H, W = img.shape[0] // 8 * 8, img.shape[1] // 8 * 8
    n_h, n_w = H // 8, W // 8
    n_p = n_w // 2
    pat = ks.check_pattern(key, n_h + off[0], n_p + off[1], 'coord')
    y = y_of(img)[:H, :W]
    f = 0
    for i in range(n_h):
        ds = [cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.float32))
              for j in range(n_w)]
        for k in range(n_p):
            d = ds[2 * k][chk] - ds[2 * k + 1][chk]
            f += 1 if ks.qim_bit(d, delta) != pat[i + off[0], k + off[1]] else 0
    return f / (n_h * n_p)


# ---------------------------------------------------------------- 变换
def t_crop_resize(im, p):
    H, W = im.shape[:2]
    nh, nw = int(H * p) // 8 * 8, int(W * p) // 8 * 8
    y0, x0 = (H - nh) // 2 // 8 * 8, (W - nw) // 2 // 8 * 8
    return cv2.resize(im[y0:y0 + nh, x0:x0 + nw], (W, H), interpolation=cv2.INTER_CUBIC)


def t_rotate(im, deg):
    H, W = im.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2 - 0.5, H / 2 - 0.5), deg, 1.0)
    return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def t_resize_round(im, s):
    H, W = im.shape[:2]
    small = cv2.resize(im, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120)
    args = ap.parse_args()
    t0 = time.time()
    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:args.n]

    deltas = [15, 25, 40]
    conds = {'clean': lambda im: im, 'crop_resize_0.9': lambda im: t_crop_resize(im, 0.9),
             'rotate_2': lambda im: t_rotate(im, 2.0), 'resize_round_0.5': lambda im: t_resize_round(im, 0.5)}
    pair, absq, quality = {}, {}, {}
    for delta in deltas:
        for name, tf in conds.items():
            vp, va, ps = [], [], []
            for s in samples:
                sid = s['id']
                try:
                    o = crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
                except FileNotFoundError:
                    continue
                if o.shape[0] < 256 or o.shape[1] < 256:
                    continue
                key = ks.image_key(MASTER, sid)
                wp = embed_pair(o, key, delta, CHK)
                wa = ks.embed_keyed(o, key, delta, (1, 0), CHK, pattern='coord')
                vp.append(fail_pair(tf(wp), key, delta, CHK))
                va.append(ks.check_fail_rate(tf(wa), key, delta=delta, check_coef=CHK,
                                             pattern='coord'))
                if name == 'clean':
                    ps.append(psnr(o, wp))
            pair[f'{name}_d{delta}'] = [float(np.mean(vp)),
                                        float(1.96 * np.std(vp) / np.sqrt(len(vp)))]
            absq[f'{name}_d{delta}'] = [float(np.mean(va)),
                                        float(1.96 * np.std(va) / np.sqrt(len(va)))]
            quality[f'pair_d{delta}'] = float(np.mean(ps)) if ps else None
            print(f'  δ={delta:2d} {name:16s} pair={np.mean(vp)*100:5.1f}%  '
                  f'abs={np.mean(va)*100:5.1f}%  PSNR={np.mean(ps) if ps else float("nan"):.1f}dB')

    verdict = {}
    for delta in deltas:
        verdict[f'delta{delta}'] = {
            'pair_clean': pair[f'clean_d{delta}'][0],
            'pair_crop_resize': pair[f'crop_resize_0.9_d{delta}'][0],
            'pair_rotate': pair[f'rotate_2_d{delta}'][0],
            'abs_clean': absq[f'clean_d{delta}'][0],
            'abs_crop_resize': absq[f'crop_resize_0.9_d{delta}'][0],
            'abs_rotate': absq[f'rotate_2_d{delta}'][0],
            'pair_wins_crop_resize': bool(
                pair[f'crop_resize_0.9_d{delta}'][0] < absq[f'crop_resize_0.9_d{delta}'][0] - 0.08
                and pair[f'clean_d{delta}'][0] < 0.35),
        }
    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n': args.n, 'chk': CHK,
                    'note': 'pair = 不相交块对(2k,2k+1)差值QIM; abs = 单块绝对QIM'},
           'pair': pair, 'abs': absq, 'psnr_pair_clean': quality, 'verdict': verdict,
           'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r8c_pairwise_geo_{ts}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n判定: ' + json.dumps({k: v['pair_wins_crop_resize'] for k, v in verdict.items()}))
    print(f'结果: {fp}  ({out["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
