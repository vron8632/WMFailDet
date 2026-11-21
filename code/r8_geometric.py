#!/usr/bin/env python3
"""G3 几何失同步: ①攻击面量化 ②training-free 同步 (coord-PRF 块偏移 + 像素相位搜索).

背景 (r5): crop90 (裁剪 90% 再缩回原尺寸) 的校验失败率 49.9% ≈ 编辑区水平 -> 致命误报.
机制: 块网格一旦错位, 所有块的期望校验位都对不上, 失败率被推到随机 50%,
      与"真的被编辑"不可区分. 这不是鲁棒性问题是**同步**问题.

为什么 coord 派生能让它可解 (G0 设计):
  pattern 的 bit 只依赖块坐标 (i, j) —— `check_pattern(key, n_h, n_w, 'coord', off=(oi,oj))`
  给出"从第 (oi,oj) 块开始"的期望位. 于是裸裁剪可以把检测写成:
      min over (oi, oj, pi, pj) of fail_rate( crop(img, pi, pj), pattern[oi:oi+n_h, oj:oj+n_w] )
  用 cv2.matchTemplate(TM_SQDIFF) 在二值图上一次算出所有 (oi,oj) 的汉明距离, 成本 ~3 ms/图;
  64 个像素相位 (pi,pj) 才需要 200 ms/图.

实验:
  A 量化 (n=400, 256x256 attackset):
    clean / crop_align(p) / crop_arbitrary(p) / crop_resize(p) / resize_round(s) / rotate(θ)
    每个变换给 naive(off=(0,0)) 与 oracle(真实偏移/相位) 两种检测.
  B 同步 (n=150):
    crop_align: 是否只用块偏移搜索就回到 ~25%? 选中的偏移是否等于真值?
    crop_arb  : 加上像素相位 (64) 是否也能回到 ~25%?
    零假设: 同样搜索但用**错误密钥** -> 搜索后的最小失败率分布 (多重比较的假阳性基线).
  C 判定: 同步后 crop 的失败率应回到干净基线附近 (≤30%), 且正确密钥的搜索值显著低于零假设.

输出: results/r8_geometric_<ts>.json
用法: apjf-python r8_geometric.py [--n 400] [--n-sync 150] [--n-null 50]
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
DELTA, MSGLEN = 15, 256
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))

CROP_PS = [0.95, 0.90, 0.85, 0.80, 0.70, 0.60]
CRESIZE_PS = [0.90, 0.80, 0.70]
RESIZE_SS = [0.90, 0.80, 0.70, 0.50]
ROTATE_DEG = [1, 2, 3, 5, 10]


# ------------------------------------------------------------------ 基础
def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def y_of(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)


def pred_map(y, chk, delta=DELTA):
    """逐块 QIM 判决 bit (n_h, n_w) float32, 只做一次 DCT."""
    H, W = y.shape[:2]
    n_h, n_w = H // 8, W // 8
    out = np.empty((n_h, n_w), np.float32)
    for i in range(n_h):
        for j in range(n_w):
            c = cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.float32))[chk]
            out[i, j] = ks.qim_bit(c, delta)
    return out


def fail_off(img, key, off=(0, 0), chk=CHK, delta=DELTA):
    """给定块偏移的失败率 (naive / oracle)."""
    H, W, n_h, n_w = ks.block_grid(img.shape)
    y = y_of(img)[:H, :W]
    pred = pred_map(y, chk, delta)
    pat = ks.check_pattern(key, n_h + off[0], n_w + off[1], 'coord')[off[0]:off[0] + n_h,
                                                                   off[1]:off[1] + n_w]
    return float((pred != pat).mean())


def best_off(pred, pat):
    """所有块偏移的汉明距离 (cv2 向量化) -> (fail, oi, oj)."""
    r = cv2.matchTemplate(pat.astype(np.float32), pred.astype(np.float32), cv2.TM_SQDIFF)
    oi, oj = np.unravel_index(int(np.argmin(r)), r.shape)
    return float(r[oi, oj] / pred.size), int(oi), int(oj)


def search_sync(img, key, max_off=12, phases=1, chk=CHK, delta=DELTA):
    """搜索 (块偏移, 像素相位) 使失败率最小. phases=1 -> 只相位(0,0); 8 -> 全部 64 个."""
    y = y_of(img)
    best = (1.0, 0, 0, 0, 0)
    ph = range(phases)
    for pi in ph:
        for pj in ph:
            Hh = (y.shape[0] - pi) // 8 * 8
            Ww = (y.shape[1] - pj) // 8 * 8
            if Hh < 64 or Ww < 64:
                continue
            n_h, n_w = Hh // 8, Ww // 8
            pred = pred_map(y[pi:pi + Hh, pj:pj + Ww], chk, delta)
            pat = ks.check_pattern(key, n_h + max_off, n_w + max_off, 'coord')
            f, oi, oj = best_off(pred, pat)
            if f < best[0]:
                best = (f, oi, oj, pi, pj)
    return best


# ------------------------------------------------------------------ 几何变换
def t_crop_align(im, p):
    """8 对齐中心裁剪 -> (out, 真实块偏移)."""
    H, W = im.shape[:2]
    nh, nw = int(H * p) // 8 * 8, int(W * p) // 8 * 8
    y0, x0 = ((H - nh) // 2) // 8 * 8, ((W - nw) // 2) // 8 * 8
    return im[y0:y0 + nh, x0:x0 + nw], (y0, x0)


def t_crop_arb(im, p):
    """任意像素偏移中心裁剪 (故意 +3/+5) -> (out, (y0,x0)) 像素级真值."""
    H, W = im.shape[:2]
    nh, nw = int(H * p), int(W * p)
    y0, x0 = (H - nh) // 2 + 3, (W - nw) // 2 + 5
    return im[y0:y0 + nh, x0:x0 + nw], (y0, x0)


def t_crop_resize(im, p):
    """裁剪 8 对齐后缩回原尺寸 (r5 的 crop90 协议)."""
    out, off = t_crop_align(im, p)
    return cv2.resize(out, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC), off


def t_resize_round(im, s):
    """缩放到 s 倍再放大回原尺寸 (平台缩放)."""
    H, W = im.shape[:2]
    small = cv2.resize(im, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0)


def t_rotate(im, deg):
    """绕中心旋转 (尺寸不变, 边界复制)."""
    H, W = im.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2 - 0.5, H / 2 - 0.5), deg, 1.0)
    return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE), (0, 0)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--n-sync', type=int, default=150)
    ap.add_argument('--n-null', type=int, default=50)
    ap.add_argument('--max-off', type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()

    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:args.n]
    wms, ids, keys = [], [], []
    for s in samples:
        sid = s['id']
        try:
            o = crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        k = ks.image_key(MASTER, sid)
        wms.append(ks.embed_keyed(o, k, DELTA, PAY, CHK, pattern='coord'))
        ids.append(sid)
        keys.append(k)
    print(f'嵌入完成: n={len(wms)} ({time.time()-t0:.0f}s)')

    # ---------------- A 量化
    def run_quant(name, tf):
        naive, oracle, extra = [], [], []
        for wm, k in zip(wms, keys):
            im, info = tf(wm)
            im = crop8(im)
            naive.append(fail_off(im, k, (0, 0)))
            if name.startswith('crop_align'):
                off = (info[0] // 8, info[1] // 8)
            elif name.startswith('crop_arb'):
                # 相位错位未被纠正, 这只是"块偏移正确"的参考上界, 真正解在 B 的搜索
                off = (info[0] // 8, info[1] // 8)
            elif name.startswith('crop_resize'):
                off = (info[0] // 8, info[1] // 8)   # 缩放已破坏网格, 仅作参考
            else:
                off = (0, 0)
            oracle.append(fail_off(im, k, off))
            extra.append(im.shape[0] / 8)      # 块数 (高)
        return {'naive': [float(np.mean(naive)), float(1.96*np.std(naive)/np.sqrt(len(naive)))],
                'oracle': [float(np.mean(oracle)), float(1.96*np.std(oracle)/np.sqrt(len(oracle)))],
                'blocks_h_mean': float(np.mean(extra)), 'n': len(naive)}

    quant = {'clean': run_quant('clean', lambda im: (im, (0, 0)))}
    for p in CROP_PS:
        quant[f'crop_align_{p}'] = run_quant(f'crop_align_{p}', lambda im, p=p: t_crop_align(im, p))
    for p in CROP_PS:
        quant[f'crop_arb_{p}'] = run_quant(f'crop_arb_{p}', lambda im, p=p: t_crop_arb(im, p))
    for p in CRESIZE_PS:
        quant[f'crop_resize_{p}'] = run_quant(f'crop_resize_{p}', lambda im, p=p: t_crop_resize(im, p))
    for s in RESIZE_SS:
        quant[f'resize_round_{s}'] = run_quant(f'resize_round_{s}', lambda im, s=s: t_resize_round(im, s))
    for deg in ROTATE_DEG:
        quant[f'rotate_{deg}'] = run_quant(f'rotate_{deg}', lambda im, deg=deg: t_rotate(im, deg))

    print('\n=== A 量化 (失败率; 干净基线≈25%, 随机=50%) ===')
    print(f'{"变换":22s} {"naive":>8s} {"oracle":>8s} {"Δoracle":>9s}')
    for name, v in quant.items():
        print(f'{name:22s} {v["naive"][0]*100:7.1f}% {v["oracle"][0]*100:7.1f}% '
              f'{(v["oracle"][0]-0.25)*100:+8.1f}pt')

    # ---------------- B 同步搜索
    def run_sync(name, tf, phases, n):
        rows = []
        for wm, k in zip(wms[:n], keys[:n]):
            im, info = tf(wm)
            im = crop8(im)
            before = fail_off(im, k, (0, 0))
            f, oi, oj, pi, pj = search_sync(im, k, args.max_off, phases)
            if name.startswith('crop_align'):
                true = (info[0] // 8, info[1] // 8)
                ok = (oi, oj) == true
            elif name.startswith('crop_arb'):
                # 检测图以 (pi,pj) 取块时, 块(0,0) 对应原图 (y0+pi, x0+pj)
                # 要回到原始 8 网格, 需 (y0+pi) % 8 == 0
                true = ((-info[0]) % 8, (-info[1]) % 8)
                ok = (pi, pj) == true
            else:
                true, ok = None, None
            rows.append({'before': before, 'after': f, 'off': (oi, oj), 'phase': (pi, pj),
                         'true': true, 'match': ok})
        arr_b = np.array([r['before'] for r in rows])
        arr_a = np.array([r['after'] for r in rows])
        res = {'n': len(rows), 'before': [float(arr_b.mean()), float(1.96*arr_b.std()/np.sqrt(len(rows)))],
               'after': [float(arr_a.mean()), float(1.96*arr_a.std()/np.sqrt(len(rows)))],
               'match_rate': float(np.mean([r['match'] for r in rows])) if rows[0]['match'] is not None else None,
               'rows': rows}
        print(f'  {name:26s} n={len(rows):4d} naive={arr_b.mean()*100:5.1f}% -> '
              f'搜后={arr_a.mean()*100:5.1f}%  '
              + (f'参数命中={res["match_rate"]*100:5.1f}%' if res['match_rate'] is not None else ''))
        return res

    print('\n=== B 同步搜索 (coord-PRF 偏移/相位) ===')
    sync = {}
    for p in [0.90, 0.80, 0.60]:
        sync[f'crop_align_{p}'] = run_sync(f'crop_align_{p}',
                                           lambda im, p=p: t_crop_align(im, p), 1, args.n_sync)
    for p in [0.90, 0.80, 0.60]:
        sync[f'crop_arb_{p}'] = run_sync(f'crop_arb_{p}',
                                         lambda im, p=p: t_crop_arb(im, p), 8, args.n_sync)

    # 零假设: 错误密钥做同样搜索 (多重比较基线)
    null_rows = []
    rk = ks.random_keys(MASTER, 1)[0]
    for wm, k in zip(wms[:args.n_null], keys[:args.n_null]):
        im, info = t_crop_arb(wm, 0.80)
        im = crop8(im)
        f_correct = search_sync(im, k, args.max_off, 8)[0]
        f_wrong = search_sync(im, rk, args.max_off, 8)[0]
        null_rows.append({'correct': f_correct, 'wrong': f_wrong})
    c = np.array([r['correct'] for r in null_rows])
    w = np.array([r['wrong'] for r in null_rows])
    sync['null_test'] = {
        'n': len(null_rows),
        'correct_key_after_search': [float(c.mean()), float(1.96*c.std()/np.sqrt(len(c)))],
        'wrong_key_after_search': [float(w.mean()), float(1.96*w.std()/np.sqrt(len(w)))],
        'separated': bool(c.mean() < w.mean() - 0.05),
    }
    print(f'  零假设(n={len(null_rows)}): 正确 key 搜后 {c.mean()*100:.1f}% vs '
          f'错误 key 搜后 {w.mean()*100:.1f}%')

    # ---------------- C 判定
    def best_crop_after(p):
        return sync[f'crop_align_{p}']['after'][0], sync[f'crop_arb_{p}']['after'][0]

    verdict = {
        'crop_align_p90': best_crop_after(0.90),
        'crop_align_p80': best_crop_after(0.80),
        'crop_arb_p90': sync['crop_arb_0.9']['after'][0],
        'crop_arb_p80': sync['crop_arb_0.8']['after'][0],
        'null_gap': sync['null_test']['wrong_key_after_search'][0] - sync['null_test']['correct_key_after_search'][0],
        'pass_crop_sync': bool(sync['crop_align_0.9']['after'][0] < 0.30 and
                               sync['crop_arb_0.9']['after'][0] < 0.30 and
                               sync['null_test']['separated']),
    }

    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n': len(wms),
                    'n_sync': args.n_sync, 'n_null': args.n_null,
                    'delta': DELTA, 'max_off': args.max_off,
                    'transform': 'crop_align/crop_arb/crop_resize/resize_round/rotate',
                    'detector': 'keyed-coord PRF + QIM fail rate'},
           'A_quant': quant, 'B_sync': sync, 'C_verdict': verdict,
           'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r8_geometric_{ts}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n判定: {verdict}')
    print(f'结果: {fp}  ({out["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
