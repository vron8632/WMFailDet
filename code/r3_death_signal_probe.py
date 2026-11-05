#!/usr/bin/env python3
"""判定实验：把"水印的死亡"当证据 —— 无需恢复消息的重绘检测。

动机 (来自前面四次失败):
  - 相位假说否定 (r=0.001)、结构上限假说否定 (delta 无饱和)、
    配对属性假说弱证据 (rho=0.42)、差分编码未验证。
  - 共同点: 所有尝试都在"重绘后恢复 256 bit"这个框架内, 而该框架有硬天花板
    (delta=40/41.3dB 时仅 68%)。
  - 换问法: 不要求恢复消息, 只要求判断"水印是否已死 / 哪一块死了"。

判据: 若块级校验通过率在 clean 与 repaint 之间完全可分 (AUC≈1.0), 则
      "水印残余"本身就是一个 training-free 的重绘检测器/定位器,
      而且它只要求"检测", 不要求"恢复" —— 躲开了天花板。

数据: data/r1_attack_cache (攻击输出) + 原图; 校验位由 run_signal_ra.check_bits 固定
输出: results/r3_death_signal_<ts>.json
用法: apjf-python r3_death_signal_probe.py
"""
import sys, os, json, glob, datetime

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
CACHE = f'{BASE}/data/r1_attack_cache'
DELTA, MSGLEN = 15, 256
VARIANTS = {'V0_current': ((4, 1), (3, 2)), 'V1_lowfreq1': ((1, 0), (1, 1)),
            'V2_lowfreq2': ((2, 1), (1, 0))}


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def check_pass_map(img, check_coef):
    """逐块校验通过掩膜 (n_h, n_w) bool."""
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    H, W = y.shape
    n_h, n_w = H // 8, W // 8
    expect = rs.check_bits(H, W, n_h, n_w)
    ok = np.zeros((n_h, n_w), bool)
    for i in range(n_h):
        for j in range(n_w):
            c = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8])[check_coef]
            r = c % DELTA
            bit = 1 if abs(r - DELTA/2) < abs(r) else 0
            ok[i, j] = (bit == expect[i, j])
    return ok


def auc(neg, pos):
    """AUC = P(pos > neg) (Mann-Whitney), 无 sklearn 依赖."""
    neg, pos = np.asarray(neg), np.asarray(pos)
    if len(neg) == 0 or len(pos) == 0:
        return float('nan')
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main():
    rng = np.random.RandomState(42)
    msg = rng.randint(0, 2, MSGLEN).astype(np.uint8)
    rows = []
    for v, (pay, chk) in VARIANTS.items():
        rs.PAY_COEF, rs.CHECK_COEF = pay, chk
        seen = set()
        for f in sorted(glob.glob(f'{CACHE}/{v}/*.png')):
            cid = os.path.basename(f)[:-4]
            b = cid.rsplit('_', 1)[0]
            if b in seen:
                continue
            seen.add(b)
            op = f'{HIRES}/images/{b}_orig.png'
            if not os.path.exists(op):
                continue
            o = crop8(np.array(Image.open(op).convert('RGB')))
            wm = rs.embed_signal(o, msg, DELTA)
            att = crop8(np.array(Image.open(f).convert('RGB')))

            ok_c = check_pass_map(wm, chk)     # clean 水印图
            ok_a = check_pass_map(att, chk)    # 重绘后

            rows.append({
                'variant': v, 'img': b,
                'clean_pass': float(ok_c.mean()), 'att_pass': float(ok_a.mean()),
                'clean_block_min': float(ok_c.mean(axis=None)),
                # 块级分数分布 (用于块级 AUC / 定位)
                'clean_blocks': ok_c.ravel().astype(int).tolist(),
                'att_blocks': ok_a.ravel().astype(int).tolist(),
            })
            print(f'  {v:12s} {b:5s} clean_pass={ok_c.mean()*100:5.1f}% '
                  f'att_pass={ok_a.mean()*100:5.1f}%  Δ={(ok_c.mean()-ok_a.mean())*100:+5.1f}pt')

    # 图像级检测: 用"校验失败率"作为统计量, clean vs repaint
    print('\n=== 图像级重绘检测 (统计量 = 校验失败率) ===')
    for v in VARIANTS:
        rs_ = [r for r in rows if r['variant'] == v]
        if not rs_:
            continue
        clean = [1 - r['clean_pass'] for r in rs_]
        att = [1 - r['att_pass'] for r in rs_]
        a = auc(clean, att)
        print(f'  {v:12s} n={len(rs_)} clean失败率={np.mean(clean)*100:.1f}% '
              f'repaint失败率={np.mean(att)*100:.1f}%  AUC={a:.3f}')

    # 块级检测 (定位能力): 每块的校验是否通过, clean 块 vs repaint 块
    print('\n=== 块级检测/定位 (单位=8x8 块) ===')
    for v in VARIANTS:
        rs_ = [r for r in rows if r['variant'] == v]
        if not rs_:
            continue
        cb = np.concatenate([np.array(r['clean_blocks']) for r in rs_])
        ab = np.concatenate([np.array(r['att_blocks']) for r in rs_])
        a = auc(1 - cb, 1 - ab)
        print(f'  {v:12s} 块数={len(cb)} clean通过={cb.mean()*100:.1f}% '
              f'repaint通过={ab.mean()*100:.1f}%  块级AUC={a:.3f}')

    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r3_death_signal_{ts}.json'
    slim = [{k: val for k, val in r.items() if not k.endswith('_blocks')} for r in rows]
    json.dump({'rows': slim, 'note': 'block lists omitted for size'}, open(fp, 'w'), indent=1)
    print(f'\n结果: {fp}')


if __name__ == '__main__':
    main()
