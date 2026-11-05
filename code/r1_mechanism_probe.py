#!/usr/bin/env python3
"""机制探查：全局重绘下水印失效的成因 —— 格点相位漂移 vs 幅度衰减。

假说 A (相位): 扩散重绘把 QIM 量化格整体平移 (色调/对比度/整体偏移),
               格点相位失配 -> 判决全错. 预期: |相位漂移| 与 BA 负相关.
假说 B (幅度): 重绘把系数整体压向均值, 信号幅度衰减. 预期: 系数方差比 与 BA 正相关.

数据: data/r1_attack_cache/<variant>/<img>.png  (已缓存攻击输出, 无需重跑扩散)
      原图: attackset_hires_sample/images/<img>_orig.png
输出: results/r1_mechanism_probe_<ts>.json
用法: apjf-python r1_mechanism_probe.py
"""
import sys, os, json, glob, datetime
from collections import defaultdict

OLD_CODE = '/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/code'
sys.path.insert(0, OLD_CODE)
import numpy as np
import cv2
from PIL import Image

_orig_makedirs = os.makedirs
def _skip(name, *a, **k):
    return None if str(name).startswith('/media/oyp/数据') else _orig_makedirs(name, *a, **k)
os.makedirs = _skip
try:
    import run_signal_ra as rs
finally:
    os.makedirs = _orig_makedirs
from dct_watermark import bit_accuracy

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRES = '/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/results/attackset_hires_sample'
CACHE = f'{BASE}/data/r1_attack_cache'
DELTA = 15
MSG_LEN = 256
VARIANTS = {'V0_current': ((4, 1), (3, 2)), 'V1_lowfreq1': ((1, 0), (1, 1)),
            'V2_lowfreq2': ((2, 1), (1, 0))}


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def coef_cube(img, coef):
    """返回每个 8x8 块的指定 DCT 系数矩阵 (n_h, n_w)."""
    H, W = img.shape[:2]
    y = rs.y_channel(img)[:H, :W].astype(np.float32)
    n_h, n_w = H // 8, W // 8
    out = np.zeros((n_h, n_w), np.float32)
    for i in range(n_h):
        for j in range(n_w):
            out[i, j] = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8])[coef]
    return out


def circ_phase(vals, delta):
    """量化格相位 (0..2pi): 系数 mod delta 的圆均值方向."""
    ang = 2 * np.pi * (vals / delta)
    return float(np.angle(np.mean(np.exp(1j * ang))))


def main():
    rng = np.random.RandomState(42)
    msg = rng.randint(0, 2, MSG_LEN).astype(np.uint8)
    rows = []
    for vname, (pay, chk) in VARIANTS.items():
        rs.PAY_COEF, rs.CHECK_COEF = pay, chk
        d = f'{CACHE}/{vname}'
        if not os.path.isdir(d):
            continue
        for fp in sorted(glob.glob(f'{d}/*.png')):
            cid = os.path.basename(fp)[:-4]
            base = cid.rsplit('_', 1)[0]
            orig_p = f'{HIRES}/images/{base}_orig.png'
            if not os.path.exists(orig_p):
                continue
            o = crop8(np.array(Image.open(orig_p).convert('RGB')))
            wm = rs.embed_signal(o, msg, DELTA)
            att = crop8(np.array(Image.open(fp).convert('RGB')))

            c_o = coef_cube(o, pay)        # 原图系数
            c_w = coef_cube(wm, pay)       # 水印图系数 (已量化到格点)
            c_a = coef_cube(att, pay)      # 攻击后系数

            # 相位: 以水印图 (格点对齐) 为基准, 看攻击后漂移多少
            ph_ref = circ_phase(c_w.ravel(), DELTA)
            ph_att = circ_phase(c_a.ravel(), DELTA)
            dphi = float(np.angle(np.exp(1j * (ph_att - ph_ref))))  # -pi..pi
            # 逐块相位一致率 (格点相位保持的比例)
            ang_w = np.angle(np.exp(1j * 2 * np.pi * c_w.ravel() / DELTA))
            ang_a = np.angle(np.exp(1j * 2 * np.pi * c_a.ravel() / DELTA))
            phase_keep = float(np.mean(np.abs(np.angle(np.exp(1j * (ang_a - ang_w)))) < np.pi / 4))

            # 幅度: 系数方差比 / 均值偏移
            var_ratio = float(c_a.std() / (c_w.std() + 1e-8))
            mean_shift = float(c_a.mean() - c_w.mean())
            # 相关系数 (攻击前后系数的空间相关 = 结构保持度)
            spatial_corr = float(np.corrcoef(c_w.ravel(), c_a.ravel())[0, 1])

            me, _ = rs.decode_signal(att, MSG_LEN, DELTA, use_check=False)
            ba_uni = float(bit_accuracy(msg, me))
            me, _ = rs.decode_signal(att, MSG_LEN, DELTA, use_check=True, soft=True, dilate=1)
            ba_v2 = float(bit_accuracy(msg, me))

            rows.append({'variant': vname, 'img': base,
                         'dphi': dphi, 'abs_dphi': abs(dphi), 'phase_keep': phase_keep,
                         'var_ratio': var_ratio, 'mean_shift': mean_shift,
                         'spatial_corr': spatial_corr,
                         'ba_uni': ba_uni, 'ba_v2': ba_v2})
            print(f'{vname:12s} {base:6s} |dphi|={abs(dphi):.2f} phase_keep={phase_keep*100:.0f}% '
                  f'var_ratio={var_ratio:.2f} corr={spatial_corr:+.2f} BA={ba_uni*100:.1f}%')

    # 相关分析
    from scipy import stats
    def corr(key):
        x = np.array([r[key] for r in rows]); y = np.array([r['ba_uni'] for r in rows])
        r_, p = stats.pearsonr(x, y)
        return r_, p
    print('\n=== 与 BA 的相关 (全部变体合并, n=%d) ===' % len(rows))
    for key in ['abs_dphi', 'phase_keep', 'var_ratio', 'spatial_corr']:
        r_, p = corr(key)
        print(f'  {key:14s} r={r_:+.3f}  p={p:.2e}')

    # 变体间均值对比
    print('\n=== 变体均值 ===')
    agg = defaultdict(list)
    for r in rows:
        agg[r['variant']].append(r)
    for v, rs_ in agg.items():
        print(f"  {v:12s} n={len(rs_)} |dphi|={np.mean([x['abs_dphi'] for x in rs_]):.2f} "
              f"phase_keep={np.mean([x['phase_keep'] for x in rs_])*100:.0f}% "
              f"var_ratio={np.mean([x['var_ratio'] for x in rs_]):.2f} "
              f"corr={np.mean([x['spatial_corr'] for x in rs_]):+.2f} "
              f"BA={np.mean([x['ba_uni'] for x in rs_])*100:.1f}%")

    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r1_mechanism_probe_{ts}.json'
    json.dump({'delta': DELTA, 'rows': rows}, open(fp, 'w'), indent=1)
    print(f'\n结果: {fp}')


if __name__ == '__main__':
    main()
