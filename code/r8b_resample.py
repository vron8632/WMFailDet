#!/usr/bin/env python3
"""G3-b 缩放/旋转为何不可救: 是"同步没找到"还是"信号已被重采样破坏"?

r8 已证明: 纯裁剪 (含任意像素偏移) 用 coord-PRF 的 (块偏移 × 像素相位) 搜索可 100% 同步,
失败率 50%→25% (基线), 参数命中 100%, 零假设 (错误密钥) 41.0%.
但 crop→resize 回原尺寸、以及 1° 旋转就把失败率推到 ~50%, 且 oracle 块偏移也救不回.

三个判定实验:
  A 反变换 + 同步: 穷举缩放因子 / 旋转角, 把图反变换后再做 (相位×偏移) 搜索.
    若仍 ~50% -> 信号在重采样中已被破坏 (同步不是瓶颈).
  B δ 扫描: δ=15/25/40 下 crop_resize/rotate/resize_round 的失败率.
    若 δ 提高能显著回落 -> 可用强度换几何鲁棒性; 若仍是 50% -> 绝对 QIM 对重采样是死刑.
  C 块大小扫描: 在 8/16/32 块上嵌入同一 check 序列 (更低频), 测 crop_resize 后存活率.
    判定"多尺度冗余嵌入"是否值得做.

输出: results/r8b_resample_<ts>.json
用法: apjf-python r8b_resample.py [--n 120] [--n-sync 30]
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


# ---------------------------------------------------------------- 任意块大小的 check 嵌入/判决
def embed_check_bs(img, key, bs, delta, chk=CHK):
    H, W = img.shape[0] // bs * bs, img.shape[1] // bs * bs
    n_h, n_w = H // bs, W // bs
    y = y_of(img)[:H, :W].copy()
    pat = ks.check_pattern(key, n_h, n_w, 'coord')
    for i in range(n_h):
        for j in range(n_w):
            d = cv2.dct(y[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs].astype(np.float32))
            c = d[chk]
            d[chk] = (np.floor(c / delta) * delta + delta / 2) if pat[i, j] == 1 \
                else np.round(c / delta) * delta
            y[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs] = cv2.idct(d)
    return write_y(img, y)


def fail_bs(img, key, bs, delta, chk=CHK):
    H, W = img.shape[0] // bs * bs, img.shape[1] // bs * bs
    n_h, n_w = H // bs, W // bs
    y = y_of(img)[:H, :W]
    pat = ks.check_pattern(key, n_h, n_w, 'coord')
    f = 0
    for i in range(n_h):
        for j in range(n_w):
            c = cv2.dct(y[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs].astype(np.float32))[chk]
            f += 1 if ks.qim_bit(c, delta) != pat[i, j] else 0
    return f / (n_h * n_w)


# ---------------------------------------------------------------- 变换
def t_crop_resize(im, p):
    H, W = im.shape[:2]
    nh, nw = int(H * p) // 8 * 8, int(W * p) // 8 * 8
    y0, x0 = (H - nh) // 2 // 8 * 8, (W - nw) // 2 // 8 * 8
    c = im[y0:y0 + nh, x0:x0 + nw]
    return cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC)


def t_rotate(im, deg):
    H, W = im.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2 - 0.5, H / 2 - 0.5), deg, 1.0)
    return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def t_resize_round(im, s):
    H, W = im.shape[:2]
    small = cv2.resize(im, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------- 同步搜索 (复用 r8 逻辑)
def pred_map(y, chk, delta):
    H, W = y.shape[:2]
    n_h, n_w = H // 8, W // 8
    out = np.empty((n_h, n_w), np.float32)
    for i in range(n_h):
        for j in range(n_w):
            c = cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.float32))[chk]
            out[i, j] = ks.qim_bit(c, delta)
    return out


def search_sync(img, key, max_off=12, phases=1, chk=CHK, delta=DELTA):
    y = y_of(img)
    best = (1.0, 0, 0, 0, 0)
    for pi in range(phases):
        for pj in range(phases):
            Hh = (y.shape[0] - pi) // 8 * 8
            Ww = (y.shape[1] - pj) // 8 * 8
            if Hh < 64 or Ww < 64:
                continue
            n_h, n_w = Hh // 8, Ww // 8
            pred = pred_map(y[pi:pi + Hh, pj:pj + Ww], chk, delta)
            pat = ks.check_pattern(key, n_h + max_off, n_w + max_off, 'coord')
            r = cv2.matchTemplate(pat.astype(np.float32), pred.astype(np.float32), cv2.TM_SQDIFF)
            oi, oj = np.unravel_index(int(np.argmin(r)), r.shape)
            f = float(r[oi, oj] / pred.size)
            if f < best[0]:
                best = (f, int(oi), int(oj), pi, pj)
    return best


def search_scale(img, key, scales, max_off=12, phases=8, topk=3):
    """穷举缩放因子反变换 + 同步搜索."""
    H, W = img.shape[:2]
    cands = []
    def undo(im, s):
        return cv2.resize(im, (max(64, int(round(W * s))), max(64, int(round(H * s)))),
                          interpolation=cv2.INTER_CUBIC)
    for s in scales:
        f = search_sync(undo(img, s), key, max_off, 1)[0]
        cands.append((f, s))
    cands.sort()
    best = (1.0, None)
    for f0, s in cands[:topk]:
        f = search_sync(undo(img, s), key, max_off, phases)[0]
        if f < best[0]:
            best = (f, s)
    return best


def search_angle(img, key, angles, max_off=12, phases=4, topk=3, crop_frac=0.85):
    """穷举旋转角反旋转 (中心裁剪去边界) + 同步搜索."""
    H, W = img.shape[:2]
    cands = []
    for a in angles:
        M = cv2.getRotationMatrix2D((W / 2 - 0.5, H / 2 - 0.5), a, 1.0)
        im2 = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        h2, w2 = int(H * crop_frac) // 8 * 8, int(W * crop_frac) // 8 * 8
        im2 = im2[(H - h2) // 2:(H - h2) // 2 + h2, (W - w2) // 2:(W - w2) // 2 + w2]
        f = search_sync(im2, key, max_off, 1)[0]
        cands.append((f, a))
    cands.sort()
    best = (1.0, None)
    for f0, a in cands[:topk]:
        M = cv2.getRotationMatrix2D((W / 2 - 0.5, H / 2 - 0.5), a, 1.0)
        im2 = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        h2, w2 = int(H * crop_frac) // 8 * 8, int(W * crop_frac) // 8 * 8
        im2 = im2[(H - h2) // 2:(H - h2) // 2 + h2, (W - w2) // 2:(W - w2) // 2 + w2]
        f = search_sync(im2, key, max_off, phases)[0]
        if f < best[0]:
            best = (f, a)
    return best


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--n-sync', type=int, default=30)
    args = ap.parse_args()
    t0 = time.time()

    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:args.n]
    rows = []
    for s in samples:
        sid = s['id']
        try:
            o = crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        if o.shape[0] < 256 or o.shape[1] < 256:
            continue
        rows.append((sid, o, ks.image_key(MASTER, sid)))
    print(f'载入 {len(rows)} 图 ({time.time()-t0:.0f}s)')

    # ---------------- A 反变换 + 同步
    # crop_resize(0.9): crop 224 -> resize 256, 反变换比例 = 224/256 = 0.875
    scales = [round(0.75 + 0.01 * i, 3) for i in range(26)]      # 0.75 .. 1.00
    angles = [round(-6 + 0.5 * i, 2) for i in range(25)]         # -6 .. +6
    rec = {'crop_resize_0.9': [], 'rotate_2': [], 'clean_control': []}
    for sid, o, key in rows[:args.n_sync]:
        wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        f_cr, s_best = search_scale(t_crop_resize(wm, 0.9), key, scales)
        f_rot, a_best = search_angle(t_rotate(wm, 2.0), key, angles)
        f_cl = search_sync(wm, key, 12, 8)[0]
        rec['crop_resize_0.9'].append({'after_search': f_cr, 'scale': s_best})
        rec['rotate_2'].append({'after_search': f_rot, 'angle': a_best})
        rec['clean_control'].append({'after_search': f_cl})
        if len(rec['clean_control']) % 10 == 0:
            print(f'  A [{len(rec["clean_control"])}/{args.n_sync}] {time.time()-t0:.0f}s', flush=True)

    A = {}
    for k, v in rec.items():
        a = np.array([x['after_search'] for x in v])
        d = {'n': len(a), 'after_search': [float(a.mean()), float(1.96 * a.std() / np.sqrt(len(a)))]}
        if 'scale' in v[0]:
            d['scale_hit'] = float(np.mean([abs(x['scale'] - 0.875) < 0.02 for x in v]))
        if 'angle' in v[0]:
            d['angle_hit'] = float(np.mean([abs(x['angle'] - 2.0) < 1.0 for x in v]))
        A[k] = d
    print('\n=== A 反变换 + 同步后的失败率 ===')
    for k, v in A.items():
        print(f'  {k:18s} n={v["n"]:3d}  {v["after_search"][0]*100:5.1f}% '
              + ''.join(f' {kk}={vv:.2f}' for kk, vv in v.items()
                        if kk in ('scale_hit', 'angle_hit')))

    # ---------------- B δ 扫描
    print('\n=== B δ 扫描 (重采样) ===')
    deltas = [15, 25, 40]
    B = {}
    for delta in deltas:
        for tname, tf in [('crop_resize_0.9', lambda im: t_crop_resize(im, 0.9)),
                          ('rotate_2', lambda im: t_rotate(im, 2.0)),
                          ('resize_round_0.5', lambda im: t_resize_round(im, 0.5))]:
            vals = []
            for sid, o, key in rows:
                wm = ks.embed_keyed(o, key, delta, PAY, CHK, pattern='coord')
                vals.append(ks.check_fail_rate(tf(wm), key, delta=delta, check_coef=CHK, pattern='coord'))
            B[f'{tname}_delta{delta}'] = [float(np.mean(vals)),
                                          float(1.96 * np.std(vals) / np.sqrt(len(vals)))]
            print(f'  {tname:18s} δ={delta:2d}: {np.mean(vals)*100:5.1f}%')

    # ---------------- C 块大小扫描
    print('\n=== C 块大小扫描 (crop_resize 0.9) ===')
    C = {}
    for bs in [8, 16, 32]:
        for tname, tf in [('clean', lambda im: im),
                          ('crop_resize_0.9', lambda im: t_crop_resize(im, 0.9)),
                          ('rotate_2', lambda im: t_rotate(im, 2.0))]:
            vals = []
            for sid, o, key in rows:
                wm = embed_check_bs(o, key, bs, DELTA, CHK)
                vals.append(fail_bs(tf(wm), key, bs, DELTA, CHK))
            C[f'bs{bs}_{tname}'] = [float(np.mean(vals)),
                                    float(1.96 * np.std(vals) / np.sqrt(len(vals)))]
            print(f'  bs={bs:2d} {tname:18s}: {np.mean(vals)*100:5.1f}%')

    verdict = {
        'crop_resize_recoverable': bool(A['crop_resize_0.9']['after_search'][0] < 0.35),
        'rotate_recoverable': bool(A['rotate_2']['after_search'][0] < 0.35),
        'delta_helps_resample': bool(B['crop_resize_0.9_delta40'][0] < B['crop_resize_0.9_delta15'][0] - 0.05),
        'bigger_block_helps_resample': bool(C['bs32_crop_resize_0.9'][0] < C['bs8_crop_resize_0.9'][0] - 0.05),
        'A': A, 'B': B, 'C': C,
    }
    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n': len(rows),
                    'n_sync': args.n_sync, 'delta': DELTA,
                    'scales': scales, 'angles': angles},
           'verdict': verdict, 'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r8b_resample_{ts}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n判定: ' + json.dumps({k: v for k, v in verdict.items() if isinstance(v, bool)}))
    print(f'结果: {fp}  ({out["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
