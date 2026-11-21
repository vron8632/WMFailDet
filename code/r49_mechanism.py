#!/usr/bin/env python3
"""r49: 机制分析 — 为什么失效图能当监督(大样本复现假说 D/B).

1. 编辑区内 vs 区外失效率 (假说 D, 原为 n=14, 现在 n=200+)
2. 逐图一致性: 区内>区外 的比例
3. 编辑前后低频系数的空间相关性 (spatial_corr) vs 区内失效率 (假说 B 弱形式, r=0.58)
4. 不同编辑面积下失效图的可分性 (区内-区外 差值)
输出: results/r49_mechanism.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r14b_multiscale import layer_key

DELTA = 15; CHK = (1, 1); MASTER = b'\x9a' + bytes(range(31))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
N = 200


def coef_map(img, coef=(1, 1), bs=8):
    """Y 通道 8x8 DCT 的指定系数图 (n_h, n_w)."""
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    H, W = y.shape[0] // bs * bs, y.shape[1] // bs * bs
    nh, nw = H // bs, W // bs
    out = np.zeros((nh, nw), np.float32)
    for i in range(nh):
        for j in range(nw):
            out[i, j] = cv2.dct(y[i*bs:(i+1)*bs, j*bs:(j+1)*bs])[coef]
    return out


def main():
    meta = json.load(open(f'{AS}/meta.json'))
    ids = []
    for s_ in meta['samples']:
        sid = s_['id']
        if os.path.exists(f'{AS}/images/orig_{sid}.png') and os.path.exists(f'{AS}/images/att_{sid}.png'):
            ids.append(s_)
        if len(ids) >= 1200:
            break
    test = ids[1000:1200]  # 测试段 (与 r34 的 test 一致)

    rows = []
    t0 = time.time()
    for s_ in test:
        sid = s_['id']
        o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
        a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        key = ks.image_key(MASTER, sid)
        k4 = layer_key(key, 'bs4')
        wm = ks.embed_keyed(o, key, DELTA, (1, 0), CHK, pattern='coord')
        from r14b_multiscale import embed_bs
        wm = embed_bs(wm, k4, 4, DELTA, (1, 1))
        from r7_keyed_scheme import paste_edit
        edited, _ = paste_edit(wm, a, s_['bbox'])
        # 失效图 (8x8 层) 与 GT 下采样
        f8 = ks.fail_map(edited, key, DELTA, CHK, 'coord')
        H, W = edited.shape[:2]
        gt = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = s_['bbox']
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
        g8 = cv2.resize(gt, (f8.shape[1], f8.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
        if g8.sum() == 0 or g8.sum() == g8.size:
            continue
        # 假说 B: 编辑前后低频系数空间相关
        c_o = coef_map(o); c_e = coef_map(edited)
        sc = float(np.corrcoef(c_o.ravel(), c_e.ravel())[0, 1])
        rows.append(dict(in_rate=float(f8[g8].mean()), out_rate=float(f8[~g8].mean()),
                         area=float(gt.mean()), spatial_corr=sc,
                         clean_rate=float(ks.fail_map(wm, key, DELTA, CHK, 'coord').mean())))
        if len(rows) >= N:
            break

    r = {k: np.array([x[k] for x in rows]) for k in rows[0]}
    from scipy import stats
    out = {
        'meta': dict(n=len(rows), note='8x8 layer; GT downsampled to block grid'),
        'in_rate_mean': float(r['in_rate'].mean()),
        'out_rate_mean': float(r['out_rate'].mean()),
        'clean_rate_mean': float(r['clean_rate'].mean()),
        'consistency_frac_in_gt_out': float((r['in_rate'] > r['out_rate']).mean()),
        'separation_mean': float((r['in_rate'] - r['out_rate']).mean()),
        'by_area': {},
        'corr_spatialcorr_vs_inrate': float(stats.pearsonr(r['spatial_corr'], r['in_rate'])[0]),
        'corr_p_spatialcorr': float(stats.pearsonr(r['spatial_corr'], r['in_rate'])[1]),
        'corr_area_vs_inrate': float(stats.pearsonr(r['area'], r['in_rate'])[0]),
    }
    for lo, hi, name in [(0, .1, 'small'), (.1, .3, 'mid'), (.3, .9, 'large'), (.9, 1.01, 'full')]:
        m = (r['area'] >= lo) & (r['area'] < hi)
        if m.sum():
            out['by_area'][name] = dict(n=int(m.sum()),
                                        in_rate=float(r['in_rate'][m].mean()),
                                        out_rate=float(r['out_rate'][m].mean()),
                                        sep=float((r['in_rate'][m] - r['out_rate'][m]).mean()))
    json.dump(out, open(f'{BASE}/results/r49_mechanism.json', 'w'), indent=1)
    print(json.dumps(out, indent=1)[:1200])
    print(f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
