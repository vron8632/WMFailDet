#!/usr/bin/env python3
"""判定实验：水印鲁棒性是「图像 × 编辑」配对的性质，还是水印方法的性质？

核心主张: 若"结构重写率"(仅由 原图 o 与重绘输出 a 决定, 与水印方法无关) 能预测不同
          水印方法在同一批图上的 BA, 则鲁棒性应被建模为配对属性, 而非方法属性.

度量 (全部方法无关, 只用 o 和 a):
  m_ssim_block : 8x8 块级 SSIM 均值 (重写率 = 1 - 该值)
  m_ssim_p10   : 块级 SSIM 的 10% 分位 (最被重写的区域)
  m_lf_corr    : 低频结构相关 —— o/a 各自 8x8 DCT 的 (1,0)/(0,1)/(1,1) 系数图空间相关
  m_e_ratio    : 低频能量占比变化 |log(ratio)|
  m_lpips      : LPIPS(VGG) 感知距离

数据: data/r1_attack_cache/<variant>/<id>.png (已有 3 个 QIM 变体的攻击输出)
      BA 从 results/r1_global_repaint_probe_*.json 读取
输出: results/r2_pair_property_<ts>.json
用法: apjf-python r2_pair_property_probe.py
"""
import sys, os, json, glob, datetime

sys.path.insert(0, '/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/code')
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRES = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
         '/results/attackset_hires_sample')
CACHE = f'{BASE}/data/r1_attack_cache'
LF_COEFS = [(1, 0), (0, 1), (1, 1)]


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def block_ssim(img_a, img_b):
    """8x8 块级 SSIM 矩阵."""
    ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    m = ssim(ga, gb, data_range=255, full=True)[1]
    n_h, n_w = m.shape[0] // 8, m.shape[1] // 8
    return np.array([[m[i*8:(i+1)*8, j*8:(j+1)*8].mean()
                      for j in range(n_w)] for i in range(n_h)])


def lf_maps(img):
    """低频系数图 (n_h, n_w) 每个 coef."""
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    H, W = y.shape
    n_h, n_w = H // 8, W // 8
    out = {c: np.zeros((n_h, n_w), np.float32) for c in LF_COEFS}
    for i in range(n_h):
        for j in range(n_w):
            d = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8])
            for c in LF_COEFS:
                out[c][i, j] = d[c]
    return out


def main():
    # BA 真值 (已有实验)
    fp = sorted(glob.glob(f'{BASE}/results/r1_global_repaint_probe_*.json'))[-1]
    d = json.load(open(fp))
    ba = {}
    for r in d['per_image']:
        ba[(r['variant'], r['img'].rsplit('_', 1)[0])] = r['uni_ba']
    print(f'BA 来源: {os.path.basename(fp)}  ({len(ba)} 条)')

    # LPIPS (需 VGG 权重; 失败则跳过)
    lpips_fn = None
    try:
        import lpips, torch
        lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
        print('LPIPS(VGG) 已加载')
    except Exception as e:
        print(f'LPIPS 不可用, 跳过: {e}')

    rows = []
    for vdir in sorted(glob.glob(f'{CACHE}/*')):
        v = os.path.basename(vdir)
        seen = set()
        for f in sorted(glob.glob(f'{vdir}/*.png')):
            cid = os.path.basename(f)[:-4]
            b = cid.rsplit('_', 1)[0]
            if b in seen:
                continue
            seen.add(b)
            op = f'{HIRES}/images/{b}_orig.png'
            if not os.path.exists(op) or (v, b) not in ba:
                continue
            o = crop8(np.array(Image.open(op).convert('RGB')))
            a = crop8(np.array(Image.open(f).convert('RGB')))

            bs = block_ssim(o, a)
            lf_o, lf_a = lf_maps(o), lf_maps(a)
            corrs = []
            for c in LF_COEFS:
                x, y = lf_o[c].ravel(), lf_a[c].ravel()
                if x.std() > 1e-6 and y.std() > 1e-6:
                    corrs.append(float(np.corrcoef(x, y)[0, 1]))
            m_lf = float(np.mean(corrs)) if corrs else 0.0
            e_o = np.mean([lf_o[c].var() for c in LF_COEFS])
            e_a = np.mean([lf_a[c].var() for c in LF_COEFS])
            m_e = float(abs(np.log((e_a + 1e-8) / (e_o + 1e-8))))

            row = {'variant': v, 'img': b, 'ba': ba[(v, b)],
                   'm_ssim_block': float(bs.mean()),
                   'm_ssim_p10': float(np.percentile(bs, 10)),
                   'm_lf_corr': m_lf, 'm_e_ratio': m_e}
            if lpips_fn is not None:
                import torch
                t = lambda im: torch.from_numpy(
                    im.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
                with torch.no_grad():
                    row['m_lpips'] = float(lpips_fn(t(o), t(a)).item())
            rows.append(row)
            print(f"  {v:12s} {b:5s} BA={row['ba']*100:5.1f}% "
                  f"blockSSIM={row['m_ssim_block']:.3f} lf_corr={m_lf:+.2f} "
                  f"e_ratio={m_e:.3f}" + (f" lpips={row.get('m_lpips',0):.3f}"
                                          if lpips_fn else ''))

    # 相关分析: pooled + within-variant
    from scipy import stats
    metrics = ['m_ssim_block', 'm_ssim_p10', 'm_lf_corr', 'm_e_ratio']
    if lpips_fn:
        metrics.append('m_lpips')
    print('\n=== 与 BA 的相关 ===')
    print(f"{'metric':14s} {'pooled ρ':>10s} {'p':>9s}   within-variant ρ")
    for m in metrics:
        x = np.array([r[m] for r in rows]); y = np.array([r['ba'] for r in rows])
        rho, p = stats.spearmanr(x, y)
        w = []
        for v in sorted({r['variant'] for r in rows}):
            xs = np.array([r[m] for r in rows if r['variant'] == v])
            ys = np.array([r['ba'] for r in rows if r['variant'] == v])
            if len(xs) >= 4 and xs.std() > 1e-9:
                w.append(stats.spearmanr(xs, ys)[0])
        print(f'{m:14s} {rho:+10.3f} {p:9.2e}   '
              + ' '.join(f'{v[:10]}:{x:+.2f}' for v, x in
                         zip(sorted({r['variant'] for r in rows}), w)))

    print('\n=== 反例检查 (0031: 旧度量 corr=0.96 但 BA=50%) ===')
    for r in rows:
        if r['img'] == '0031':
            print(f"  {r['variant']:12s} BA={r['ba']*100:.1f}% blockSSIM={r['m_ssim_block']:.3f} "
                  f"lf_corr={r['m_lf_corr']:+.2f} e_ratio={r['m_e_ratio']:.3f}"
                  + (f" lpips={r.get('m_lpips',0):.3f}" if lpips_fn else ''))

    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_fp = f'{BASE}/results/r2_pair_property_{ts}.json'
    json.dump({'rows': rows, 'n': len(rows)}, open(out_fp, 'w'), indent=1)
    print(f'\n结果: {out_fp}')


if __name__ == '__main__':
    main()
