#!/usr/bin/env python3
"""检测器特异性（假阳性）测试：非语义扰动会不会被误判为"编辑"？

原理: 检测器的统计量 = 块级校验失败率. 干净图基线 ≈25% (单校验位天然 75% 通过),
      语义编辑后 → 50% (信号随机化).
      若普通处理 (JPEG/噪声/缩放/模糊/裁剪) 也把失败率推向 50%, 检测器就会误报,
      必须靠阈值区分. 本实验直接量出各扰动的失败率, 判断可分性.

输出: results/r5_false_positive_<ts>.json
用法: apjf-python r5_false_positive_probe.py [--n 14]
"""
import sys, os, json, argparse, glob, datetime, io

sys.path.insert(0, '/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/code')
import numpy as np
import cv2
from PIL import Image

_o = os.makedirs
def _skip(name, *a, **k):
    return None if str(name).startswith('/media/oyp/数据') else _o(name, *a, **k)
os.makedirs = _skip
try:
    import run_signal_ra as rs
finally:
    os.makedirs = _o

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRES = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
         '/results/attackset_hires_sample')
DELTA, MSGLEN = 15, 256
PAY, CHK = (1, 0), (1, 1)


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def block_fail(img):
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    H, W = y.shape
    n_h, n_w = H // 8, W // 8
    exp = rs.check_bits(H, W, n_h, n_w)
    fail = np.zeros((n_h, n_w), np.float32)
    for i in range(n_h):
        for j in range(n_w):
            c = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8])[CHK]
            r = c % DELTA
            bit = 1 if abs(r - DELTA/2) < abs(r) else 0
            fail[i, j] = 1.0 if bit != exp[i, j] else 0.0
    return fail


def jpeg(img, q):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=q)
    return np.array(Image.open(buf).convert('RGB'))


def attn_list():
    return {
        'clean': lambda im: im,
        'jpeg95': lambda im: jpeg(im, 95),
        'jpeg90': lambda im: jpeg(im, 90),
        'jpeg75': lambda im: jpeg(im, 75),
        'jpeg60': lambda im: jpeg(im, 60),
        'noise2': lambda im: np.clip(im.astype(np.float32) +
                                     np.random.RandomState(0).normal(0, 2, im.shape), 0, 255).astype(np.uint8),
        'noise5': lambda im: np.clip(im.astype(np.float32) +
                                     np.random.RandomState(0).normal(0, 5, im.shape), 0, 255).astype(np.uint8),
        'gauss_blur3': lambda im: cv2.GaussianBlur(im, (3, 3), 0.8),
        'resize_half': lambda im: cv2.resize(cv2.resize(im, (im.shape[1]//2, im.shape[0]//2),
                                                        interpolation=cv2.INTER_AREA),
                                             (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC),
        'crop90': lambda im: np.array(Image.fromarray(im).crop(
            (0, 0, int(im.shape[1]*0.9), int(im.shape[0]*0.9))).resize(
            (im.shape[1], im.shape[0]))),
        'brightness': lambda im: np.clip(im.astype(np.float32) * 1.05, 0, 255).astype(np.uint8),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=14)
    args = ap.parse_args()
    rs.PAY_COEF, rs.CHECK_COEF = PAY, CHK

    meta = json.load(open(f'{HIRES}/meta.json'))
    seen, bases = set(), []
    for s in meta['samples']:
        b = s['id'].rsplit('_', 1)[0]
        if b not in seen and os.path.exists(f'{HIRES}/images/{s["id"]}_orig.png'):
            seen.add(b)
            bases.append(s['id'])
    bases = bases[:args.n]

    rng = np.random.RandomState(42)
    msg = rng.randint(0, 2, MSGLEN).astype(np.uint8)
    atts = attn_list()
    res = {k: [] for k in atts}
    for cid in bases:
        o = crop8(np.array(Image.open(f'{HIRES}/images/{cid}_orig.png').convert('RGB')))
        wm = rs.embed_signal(o, msg, DELTA)
        for name, fn in atts.items():
            im = crop8(fn(wm))
            res[name].append(float(block_fail(im).mean()))

    print(f'{"扰动":14s} {"失败率":>8s}  {"vs clean":>9s}  {"判定":>10s}')
    clean = np.array(res['clean'])
    summary = {}
    for name in atts:
        v = np.array(res[name])
        d = (v - clean).mean() * 100
        # 判定: 是否已进入"编辑区"区间 (编辑=50%, 干净=25%)
        verdict = '误报风险高' if v.mean() > 0.42 else ('轻微上升' if v.mean() > 0.30 else '安全')
        summary[name] = {'fail_mean': float(v.mean()), 'delta_pt': float(d),
                         'verdict': verdict}
        print(f'{name:14s} {v.mean()*100:7.1f}%  {d:+8.1f}pt  {verdict:>10s}')

    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r5_false_positive_{ts}.json'
    json.dump({'n': len(bases), 'coef': f'{PAY}/{CHK}', 'delta': DELTA,
               'summary': summary, 'per_image': res}, open(fp, 'w'), indent=1)
    print(f'\n参考区间: 干净≈24.6% | 语义编辑区≈50.7% (来自 r4)')
    print(f'结果: {fp}')


if __name__ == '__main__':
    main()
