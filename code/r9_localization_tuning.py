#!/usr/bin/env python3
"""G4 定位精度调优 + 图像级检测统计量改进（纯离线，复用 r7 的 1200 张失败图）.

输入: data/failmaps_r7_*.npz  (keyed-coord 逐块失败图 1200x32x32 + 块级 GT + per-image key)
      attackset meta.json     (bbox, 用于像素级 IoU 与面积分层)

两部分:
  A 图像级检测统计量 (解决 r7 遗留: per-image pattern 使全图平均 AUC 从 0.9496 掉到 0.932)
    对比: 全图平均 / 最大 5x5 区域 / 平滑图 p95 / top-10% 块均值
    小编辑会被全图平均稀释, 区域统计量应该更稳 -> 决定正式方案用哪个统计量.
  B 定位后处理调优 (r6/r7 的 IoU 0.337 偏低)
    grid: 平滑窗 k in {1,3,5,7,9,11} × 阈值 t in {0.30..0.70}
    后处理: 无 / 开运算 / 闭运算 / 开+闭 / 连通域最小面积过滤
    指标: 像素级 IoU (用 bbox 精确计算, 与 r7 同口径), 块级 P/R/F1
    分层: 编辑面积 <0.1 / 0.1-0.3 / >0.3 / 全图

输出: results/r9_localization_tuning_<ts>.json
用法: apjf-python r9_localization_tuning.py [--npz data/failmaps_r7_*.npz]
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
DELTA = 15
PAY, CHK = (1, 0), (1, 1)


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def smooth(f, k):
    if k <= 1:
        return f.astype(np.float32)
    k = k if k % 2 == 1 else k + 1
    return cv2.blur(f.astype(np.float32), (k, k))


def postproc(pred, mode, min_area=0.0):
    """pred: bool 块矩阵. 返回处理后 bool 矩阵."""
    p = pred.astype(np.uint8)
    if mode in ('open', 'open_close'):
        p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if mode in ('close', 'open_close'):
        p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if min_area > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
        keep = np.zeros_like(p)
        for c in range(1, n):
            if stats[c, cv2.CC_STAT_AREA] >= min_area:
                keep[lab == c] = 1
        p = keep
    return p.astype(bool)


def pixel_iou(pred, cov, gt_area):
    """像素级 IoU 的精确块级等价式: inter = Σ pred*cov, union = 64*Σpred + |gt| - inter."""
    inter = float((pred.astype(np.float32) * cov).sum())
    union = 64.0 * float(pred.sum()) + float(gt_area) - inter
    return inter / union if union > 0 else 0.0


def block_prf(pred, gt):
    tp = float((pred & gt).sum())
    p = tp / float(pred.sum()) if pred.sum() else 1.0
    r = tp / float(gt.sum()) if gt.sum() else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def auc_ci(neg, pos, n_boot=1000, seed=0):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.randint(0, len(neg), len(neg))
        j = rng.randint(0, len(pos), len(pos))
        vals.append(ks.auc(neg[i], pos[j]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=None)
    ap.add_argument('--n-clean', type=int, default=1200, help='重算 clean 失败图的张数')
    args = ap.parse_args()
    t0 = time.time()

    npz_path = args.npz or sorted(glob.glob(f'{BASE}/data/failmaps_r7_*.npz'))[-1]
    z = np.load(npz_path, allow_pickle=True)
    fail, gt, ids, keys = z['fail'], z['gt'], z['ids'], z['keys']
    print(f'载入 {npz_path}: fail={fail.shape}, gt={gt.shape}')

    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)
    bbox = {s['id']: s['bbox'] for s in samples}

    # 每图: 像素 GT 覆盖率矩阵 cov (32x32) + gt 面积
    n_img, n_h, n_w = fail.shape
    cov = np.zeros((n_img, n_h, n_w), np.float32)
    gt_area = np.zeros(n_img, np.float64)
    area01 = np.zeros(n_img, np.float32)
    for i, sid in enumerate(ids):
        H, W = n_h * 8, n_w * 8
        x1, y1, x2, y2 = bbox[sid]
        bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
        bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
        g = np.zeros((H, W), bool)
        g[by1:by2, bx1:bx2] = True
        cov[i] = g.reshape(n_h, 8, n_w, 8).sum(axis=(1, 3))
        gt_area[i] = g.sum()
        area01[i] = (x2 - x1) * (y2 - y1)

    # ---------------- A 图像级检测统计量
    print('\n=== A 图像级检测统计量 (clean vs edited, 配对) ===')
    # clean 失败图需要重算 (npz 只存了 att)
    clean_fail = np.zeros_like(fail)
    n_done = 0
    for i, sid in enumerate(ids[:args.n_clean]):
        try:
            o = crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        k = bytes(keys[i])
        wm = ks.embed_keyed(o, k, DELTA, PAY, CHK, pattern='coord')
        clean_fail[i] = ks.fail_map(wm, k, DELTA, CHK, 'coord')
        n_done += 1
    valid = np.array([bool(clean_fail[i].sum()) for i in range(n_img)])
    print(f'  clean 失败图重算: {n_done} 张 ({time.time()-t0:.0f}s)')

    def stats_of(f):
        s = smooth(f, 5)
        return {
            'mean': float(f.mean()),
            'max_region': float(cv2.dilate(s, np.ones((5, 5), np.float32)).max()),
            'p95': float(np.percentile(s, 95)),
            'top10pct': float(np.sort(s.ravel())[-max(1, int(0.10 * s.size)):].mean()),
        }

    stat_names = ['mean', 'max_region', 'p95', 'top10pct']
    cl = {n: [] for n in stat_names}
    at = {n: [] for n in stat_names}
    for i in range(n_img):
        if not valid[i]:
            continue
        sc, sa = stats_of(clean_fail[i]), stats_of(fail[i])
        for n in stat_names:
            cl[n].append(sc[n])
            at[n].append(sa[n])
    A = {}
    for n in stat_names:
        A[n] = {
            'clean_mean': float(np.mean(cl[n])), 'att_mean': float(np.mean(at[n])),
            'auc': ks.auc(cl[n], at[n]),
            'auc_boot95': auc_ci(cl[n], at[n]),
        }
        print(f'  {n:12s} clean={A[n]["clean_mean"]*100:5.1f}% att={A[n]["att_mean"]*100:5.1f}% '
              f'AUC={A[n]["auc"]:.4f} [{A[n]["auc_boot95"][0]:.4f},{A[n]["auc_boot95"][1]:.4f}]')

    # ---------------- B 定位调优
    print('\n=== B 定位调优 (像素级 IoU, grid) ===')
    ks_list = [3, 5, 7, 8, 9, 10, 11, 13]
    ts = [round(0.30 + 0.02 * i, 2) for i in range(11)]      # 0.30..0.50
    # split-half: 前半调参, 后半评估 (避免"在测试集上调参")
    tr_idx = np.arange(0, n_img // 2)
    te_idx = np.arange(n_img // 2, n_img)
    grid = {}
    for k in ks_list:
        sm = np.stack([smooth(fail[i], k) for i in range(n_img)])
        for t in ts:
            preds = sm > t
            ious = np.array([pixel_iou(preds[i], cov[i], gt_area[i]) for i in range(n_img)])
            prf = np.array([block_prf(preds[i], gt[i]) for i in range(n_img)])
            grid[f'k{k}_t{t}'] = {
                'iou': float(ious.mean()), 'iou_ci95': float(1.96 * ious.std() / np.sqrt(n_img)),
                'iou_train': float(ious[tr_idx].mean()), 'iou_test': float(ious[te_idx].mean()),
                'prec': float(prf[:, 0].mean()), 'rec': float(prf[:, 1].mean()),
                'f1': float(prf[:, 2].mean()),
            }
    best_train = max(grid.items(), key=lambda kv: kv[1]['iou_train'])
    best = max(grid.items(), key=lambda kv: kv[1]['iou'])
    print(f'  全量最优: {best[0]}  IoU={best[1]["iou"]:.4f} P={best[1]["prec"]:.3f} '
          f'R={best[1]["rec"]:.3f} F1={best[1]["f1"]:.3f}')
    print(f'  split-half: 前半调参={best_train[0]} (train {best_train[1]["iou_train"]:.4f}) '
          f'-> 后半 test IoU={best_train[1]["iou_test"]:.4f} '
          f'(全量同配置 {best_train[1]["iou"]:.4f})')
    for k in ks_list:
        row = [grid[f'k{k}_t{t}']['iou'] for t in ts]
        print(f'  k={k:2d}  ' + ' '.join(f'{v:.3f}' for v in row))
    print(f'  (阈值列: {ts})')

    # 自适应阈值对照 (Otsu / 分位数), 用全量最优 k
    from skimage.filters import threshold_otsu
    bk, bt = best[0].split('_')[0][1:], best[0].split('_')[1][1:]
    bkp, btp = int(bk), float(bt)
    adapt = {}
    sm_best = np.stack([smooth(fail[i], bkp) for i in range(n_img)])
    for tag in ['otsu', 'q60', 'q70', 'q80']:
        ious = []
        for i in range(n_img):
            if tag == 'otsu':
                t_i = float(threshold_otsu(sm_best[i]))
            else:
                t_i = float(np.percentile(sm_best[i], float(tag[1:])))
            ious.append(pixel_iou(sm_best[i] > t_i, cov[i], gt_area[i]))
        adapt[tag] = float(np.mean(ious))
    print(f'  自适应阈值: ' + '  '.join(f'{k}={v:.3f}' for k, v in adapt.items()))

    # 后处理 (在最优 k/t 上)
    sm = sm_best
    base_pred = sm > btp
    pp = {}
    for mode, min_area in [('none', 0), ('open', 0), ('close', 0), ('open_close', 0),
                           ('open_close', 2), ('open_close', 4), ('open_close', 8),
                           ('none', 2), ('none', 4), ('none', 8)]:
        tag = f'{mode}_min{min_area}'
        ious = []
        for i in range(n_img):
            p = postproc(base_pred[i], mode, min_area)
            ious.append(pixel_iou(p, cov[i], gt_area[i]))
        pp[tag] = float(np.mean(ious))
    best_pp = max(pp.items(), key=lambda kv: kv[1])
    print(f'  后处理: 无={pp["none_min0"]:.4f}  最优={best_pp[0]} {best_pp[1]:.4f}')

    # 定位 AUC (连续分数, 与阈值无关; 平滑窗决定)
    from sklearn.metrics import roc_auc_score
    loc_auc = {}
    for k in [3, 5, 7, 9, 11]:
        vals = []
        for i in range(n_img):
            g = gt[i]
            if g.sum() < 4 or (~g).sum() < 4:
                continue
            sc = smooth(fail[i], k).ravel()
            lab = g.ravel().astype(int)
            vals.append(float(roc_auc_score(lab, sc)))
        loc_auc[f'k{k}'] = [float(np.mean(vals)),
                            float(1.96 * np.std(vals) / np.sqrt(len(vals)))]
    print('  定位 AUC: ' + '  '.join(f'{k}={v[0]:.4f}' for k, v in loc_auc.items()))

    # 分层 (最优配置 + 最优后处理)
    mode_best, min_best = best_pp[0].rsplit('_min', 1)
    min_best = float(min_best)
    tiers = {}
    for tag, lo, hi in [('small', 0.0, 0.10), ('mid', 0.10, 0.30),
                        ('large', 0.30, 0.90), ('full', 0.90, 1.01)]:
        idx = [i for i in range(n_img) if lo <= area01[i] < hi]
        if not idx:
            continue
        ious, prfs = [], []
        for i in idx:
            p = postproc(base_pred[i], mode_best, min_best)
            ious.append(pixel_iou(p, cov[i], gt_area[i]))
            prfs.append(block_prf(p, gt[i]))
        tiers[tag] = {'n': len(idx), 'iou': float(np.mean(ious)),
                      'f1': float(np.mean(np.array(prfs)[:, 2]))}
        print(f'  {tag:6s} n={len(idx):4d} IoU={tiers[tag]["iou"]:.3f} F1={tiers[tag]["f1"]:.3f}')

    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'npz': npz_path,
                    'n': int(valid.sum()), 'ks': ks_list, 'ts': ts,
                    'note': 'IoU 为像素级精确块级等价式, 与 r7/r6 同口径'},
           'A_detect_stats': A, 'B_grid': grid,
           'B_best': {'config': best[0], **best[1]},
           'B_postproc': pp, 'B_best_postproc': best_pp[0], 'B_adaptive': adapt,
           'B_split_half': {'config': best_train[0], **best_train[1]},
           'B_tiers': tiers, 'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r9_localization_tuning_{fn}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n结果: {fp}  ({out["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
