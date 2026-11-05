#!/usr/bin/env python3
"""r14 定位提升 v2（纯离线，复用 r7 失败图）: 背景归一化 + 引导滤波细化.

问题: r9 的最优工作点 (k=9, t=0.38, open_close, min_area=8) 只有 IoU 0.604,
      且 small 编辑只有 0.372. 损失来自两处:
  1 块级 8x8 的边界误差 (±8 px);
  2 clean 基线本身就有 ~25% 的随机失败块 (噪声底), 阈值 0.38 被噪声底抬高.

两条 training-free 的改进:
  A 背景归一化: fail_norm = fail - median_bg(fail) + 0.25  -> 压掉噪声底, 小编辑更易过阈值;
  B 引导滤波上采样: 以原图灰度 I 为引导, 把 32x32 的软失败图 guided-filter 到 256x256,
    让区域边界贴住图像边缘, 而不是方块状.

评估: 像素级 IoU (bbox 精确块级等价式), 分层 small/mid/large/full, split-half 选参.
输出: results/r14_localization_v2_<ts>.json
用法: apjf-python r14_localization_v2.py
"""
import sys, os, json, glob, argparse, datetime, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

import keyed_scheme as ks

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')


def guided_filter(I, p, r, eps):
    """单通道引导滤波 (He et al.), I/p∈[0,1] float32."""
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(I, -1, k)
    mp = cv2.boxFilter(p, -1, k)
    mIp = cv2.boxFilter(I * p, -1, k)
    cov = mIp - mI * mp
    mII = cv2.boxFilter(I * I, -1, k)
    var = mII - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def median_bg(f, k):
    """失败图的局部背景 (大窗口中值), 用于压噪声底."""
    k = k if k % 2 == 1 else k + 1
    return cv2.medianBlur((f * 255).astype(np.uint8), k).astype(np.float32) / 255.0


def iou_px(pred_px, bbox, H, W):
    x1, y1, x2, y2 = bbox
    g = np.zeros((H, W), bool)
    g[int(round(y1 * H)):int(round(y2 * H)), int(round(x1 * W)):int(round(x2 * W))] = True
    inter, union = (pred_px & g).sum(), (pred_px | g).sum()
    return float(inter / union) if union else 0.0


def post(pred, min_area=8):
    p = pred.astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
    keep = np.zeros_like(p)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == c] = 1
    return keep.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=None)
    ap.add_argument('--n', type=int, default=1200)
    args = ap.parse_args()
    t0 = time.time()

    npz_path = args.npz or sorted(glob.glob(f'{BASE}/data/failmaps_r7_*.npz'))[-1]
    z = np.load(npz_path, allow_pickle=True)
    fail, ids = z['fail'], z['ids']
    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)
    bbox = {s['id']: s['bbox'] for s in samples}
    n_img = min(len(fail), args.n)
    H, W = fail.shape[1] * 8, fail.shape[2] * 8

    # 读原图 (引导图)
    print(f'载入 {n_img} 张原图作为引导图...')
    grays = []
    for i in range(n_img):
        sid = ids[i]
        o = np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB'))[:H, :W]
        g = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        grays.append(g)
    print(f'  {time.time()-t0:.0f}s')

    area = np.array([(bbox[ids[i]][2] - bbox[ids[i]][0]) * (bbox[ids[i]][3] - bbox[ids[i]][1])
                     for i in range(n_img)])
    tr = np.arange(0, n_img // 2)
    te = np.arange(n_img // 2, n_img)

    def evaluate(pred_list, which):
        idx = tr if which == 'train' else te
        v = [iou_px(pred_list[i], bbox[ids[i]], H, W) for i in idx]
        return float(np.mean(v))

    # ---------------- 方法 A: 背景归一化 (块级)
    print('\n=== A 背景归一化 (块级) ===')
    A = {}
    for bgk in [9, 15, 21]:
        for t in [0.25, 0.28, 0.30, 0.32, 0.35]:
            preds = []
            for i in range(n_img):
                f = fail[i]
                bg = median_bg(f, bgk)
                norm = np.clip(f - bg + 0.25, 0, 1)
                s = ks.smooth_fail_map(norm, 9)
                pp = post(s > t)
                preds.append(np.repeat(np.repeat(pp, 8, 0), 8, 1))
            tr_iou = evaluate(preds, 'train')
            te_iou = evaluate(preds, 'test')
            A[f'bg{bgk}_t{t}'] = {'train': tr_iou, 'test': te_iou}
    best_A = max(A.items(), key=lambda kv: kv[1]['train'])
    print(f'  最优(训练半): {best_A[0]}  train={best_A[1]["train"]:.4f} test={best_A[1]["test"]:.4f}')

    # ---------------- 方法 B: 引导滤波上采样
    print('\n=== B 引导滤波上采样 ===')
    B = {}
    for r in [4, 8, 16]:
        for eps in [1e-3, 1e-2]:
            for t in [0.35, 0.38, 0.42]:
                preds = []
                for i in range(n_img):
                    soft = cv2.resize(fail[i], (W, H), interpolation=cv2.INTER_CUBIC)
                    gf = guided_filter(grays[i], soft, r, eps)
                    p = post(gf > t, 64)      # 像素级连通域, 面积阈值按像素
                    preds.append(p)
                tr_iou = evaluate(preds, 'train')
                te_iou = evaluate(preds, 'test')
                B[f'r{r}_e{eps}_t{t}'] = {'train': tr_iou, 'test': te_iou}
    best_B = max(B.items(), key=lambda kv: kv[1]['train'])
    print(f'  最优(训练半): {best_B[0]}  train={best_B[1]["train"]:.4f} test={best_B[1]["test"]:.4f}')

    # ---------------- 方法 C: A + B 组合
    print('\n=== C 背景归一化 + 引导滤波 ===')
    C = {}
    for bgk in [15]:
        for r in [8, 16]:
            for eps in [1e-3, 1e-2]:
                for t in [0.32, 0.35, 0.38, 0.42]:
                    preds = []
                    for i in range(n_img):
                        bg = median_bg(fail[i], bgk)
                        norm = np.clip(fail[i] - bg + 0.25, 0, 1)
                        soft = cv2.resize(norm, (W, H), interpolation=cv2.INTER_CUBIC)
                        gf = guided_filter(grays[i], soft, r, eps)
                        p = post(gf > t, 64)
                        preds.append(p)
                    tr_iou = evaluate(preds, 'train')
                    te_iou = evaluate(preds, 'test')
                    C[f'bg{bgk}_r{r}_e{eps}_t{t}'] = {'train': tr_iou, 'test': te_iou}
    best_C = max(C.items(), key=lambda kv: kv[1]['train'])
    print(f'  最优(训练半): {best_C[0]}  train={best_C[1]["train"]:.4f} test={best_C[1]["test"]:.4f}')

    # ---------------- 基线 (r9 工作点) 与分层
    base_preds = []
    for i in range(n_img):
        pp = post(ks.smooth_fail_map(fail[i], 9) > 0.38)
        base_preds.append(np.repeat(np.repeat(pp, 8, 0), 8, 1))
    base_all = float(np.mean([iou_px(base_preds[i], bbox[ids[i]], H, W) for i in range(n_img)]))

    # 用最优方法 C 配置做分层
    tag = best_C[0]
    bgk = int(tag.split('_')[0][2:])
    r = int(tag.split('_')[1][1:])
    eps = float(tag.split('_')[2][1:])
    t = float(tag.split('_')[3][1:])
    best_preds = []
    for i in range(n_img):
        bg = median_bg(fail[i], bgk)
        norm = np.clip(fail[i] - bg + 0.25, 0, 1)
        soft = cv2.resize(norm, (W, H), interpolation=cv2.INTER_CUBIC)
        gf = guided_filter(grays[i], soft, r, eps)
        best_preds.append(post(gf > t, 64))

    tiers = {}
    for name, lo, hi in [('small', 0.0, 0.10), ('mid', 0.10, 0.30),
                         ('large', 0.30, 0.90), ('full', 0.90, 1.01)]:
        idx = [i for i in range(n_img) if lo <= area[i] < hi]
        if not idx:
            continue
        tiers[name] = {
            'n': len(idx),
            'base_iou': float(np.mean([iou_px(base_preds[i], bbox[ids[i]], H, W) for i in idx])),
            'v2_iou': float(np.mean([iou_px(best_preds[i], bbox[ids[i]], H, W) for i in idx])),
        }
        print(f'  {name:6s} n={len(idx):4d} 基线 {tiers[name]["base_iou"]:.3f} -> v2 {tiers[name]["v2_iou"]:.3f}')

    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n': n_img, 'npz': npz_path,
                    'split_half': True,
                    'note': 'IoU 为像素级, split-half: 前半选参/后半评估'},
           'A_background_norm': A, 'B_guided': B, 'C_combined': C,
           'best_A': {'config': best_A[0], **best_A[1]},
           'best_B': {'config': best_B[0], **best_B[1]},
           'best_C': {'config': best_C[0], **best_C[1]},
           'baseline_r9_iou': base_all, 'tiers': tiers,
           'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r14_localization_v2_{fn}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n基线(r9) IoU={base_all:.4f}  v2 最优 test IoU={best_C[1]["test"]:.4f} '
          f'({tag})')
    print(f'结果: {fp}  ({out["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
