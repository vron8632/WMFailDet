#!/usr/bin/env python3
"""G5 检测器特异性扩展 + 阈值校准（r5 的升级版：新统计量 + 同步/拒答 + 工作曲线）.

r5 的旧结论: JPEG 全谱零误报, 但 crop90 失败率 49.9% ≈ 编辑区 (致命误报).
G3 已修: 纯裁剪经 coord-PRF 同步后回到基线; 重采样则由检测器**拒答**.
G4 已换统计量: 图像级检测用 top-10% 块均值 (AUC 0.9742).

本实验回答三件事:
  1 什么处理会被误报? 对 15 种非语义扰动, 用 clean 的 99% 分位定阈值, 报误报率 (FPR)
    与定位假阳性面积 (pred 区域占比);
  2 检测器工作曲线: 阈值 vs FPR(clean/扰动) vs TPR(真编辑), 给 AUC 与推荐工作点;
  3 同步/拒答在特异性上的作用: crop(任意相位) 同步后应恢复"不误报"; rotate 应被拒答
    而不是误报 (拒答率单列, 不计入 FPR).

数据: attackset_20261001 (n=300, 256x256); 真编辑 = 同 attackset 的 att 图 (bbox paste, 同 r6/r7)
输出: results/r10_specificity_<ts>.json
用法: apjf-python r10_specificity.py [--n 300]
"""
import sys, os, json, argparse, datetime, time, io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

import keyed_scheme as ks
import r8_geometric as r8
from r8d_sync_abstain import failmap_at, low_quantile, sync_fstar
from r7_keyed_scheme import paste_edit

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
DELTA = 15
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))
K_LOC, T_LOC = 9, 0.38          # G4 的定位工作点
QLOW = 0.25                     # G3 的同步判据阈值
TOP_Q = 90                      # top-10% 块均值


# ---------------------------------------------------------------- 扰动
def jpeg(im, q):
    buf = io.BytesIO()
    Image.fromarray(im).save(buf, format='JPEG', quality=q)
    return np.array(Image.open(buf).convert('RGB'))


def noise(im, sigma):
    return np.clip(im.astype(np.float32) +
                   np.random.RandomState(0).normal(0, sigma, im.shape), 0, 255).astype(np.uint8)


def sharpen(im):
    bl = cv2.GaussianBlur(im, (0, 0), 1.2)
    return np.clip(im.astype(np.float32) * 1.6 - bl.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)


def color(im, gain=1.0, sat=1.0):
    out = np.clip(im.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if sat != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def resize_round(im, s):
    H, W = im.shape[:2]
    small = cv2.resize(im, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


def crop_arb(im, p):
    return r8.t_crop_arb(im, p)[0]


def rotate(im, deg):
    return r8.t_rotate(im, deg)[0]


def screenshot(im):
    """截图/重拍近似: 缩到 0.7 + JPEG70 + 噪声 2."""
    return noise(jpeg(resize_round(im, 0.7), 70), 2)


# name -> (fn, 是否几何/需要同步)
PERTURBS = {
    'clean': (lambda im: im, False),
    'jpeg95': (lambda im: jpeg(im, 95), False),
    'jpeg75': (lambda im: jpeg(im, 75), False),
    'jpeg50': (lambda im: jpeg(im, 50), False),
    'noise2': (lambda im: noise(im, 2), False),
    'noise5': (lambda im: noise(im, 5), False),
    'noise10': (lambda im: noise(im, 10), False),
    'blur3': (lambda im: cv2.GaussianBlur(im, (3, 3), 0.8), False),
    'sharpen': (lambda im: sharpen(im), False),
    'brightness_1.05': (lambda im: color(im, 1.05), False),
    'saturation_1.1': (lambda im: color(im, 1.0, 1.1), False),
    'resize_round_0.8': (lambda im: resize_round(im, 0.8), False),
    'screenshot_sim': (lambda im: screenshot(im), False),
    'crop_arb_0.9': (lambda im: crop_arb(im, 0.9), True),
    'rotate_1': (lambda im: rotate(im, 1.0), True),
}


def stats_of(fail, k=K_LOC, q=TOP_Q):
    """两个图像级统计量.

    top      : 平滑图 top-10% 块均值 (G4 采用) -> 对局部编辑敏感, 但对全局扰动也敏感;
    contrast : top-10% - 中位数 -> 衡量失败率的**空间聚集度**:
               局部编辑 = 少数块 50%, 多数块 25% -> contrast 大;
               强全局扰动 (blur/noise) = 全图一起升到 ~50% -> contrast 小;
               全图编辑 = 全图 50% -> contrast 也小 (这是它的盲区, 必须单列).
    """
    s = ks.smooth_fail_map(fail, k)
    vals = np.sort(s.ravel())
    n = max(1, int((100 - q) / 100 * vals.size))
    top = float(vals[-n:].mean())
    med = float(np.median(vals))
    return {'top': top, 'median': med, 'contrast': top - med}


def pred_area(fail):
    pred = ks.smooth_fail_map(fail, K_LOC) > T_LOC
    p = pred.astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
    keep = np.zeros_like(p)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= 8:
            keep[lab == c] = 1
    return float(keep.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    args = ap.parse_args()
    t0 = time.time()
    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:args.n]

    rows = []
    for idx, s in enumerate(samples):
        sid = s['id']
        try:
            o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        key = ks.image_key(MASTER, sid)
        wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        rec = {'id': sid, 'perturb': {}}
        for name, (fn, geo) in PERTURBS.items():
            im = r8.crop8(fn(wm))
            if geo:
                f_star, oi, oj, pi, pj = sync_fstar(im, key, wm.shape, 8 if name.startswith('crop') else 1)
                fail = failmap_at(im, key, oi, oj, pi, pj)
                low = low_quantile(fail)
                abstain = low >= QLOW
                rec['perturb'][name] = {**stats_of(fail), 'low10': low,
                                        'abstain': bool(abstain), 'area': pred_area(fail)}
            else:
                fail = ks.fail_map(im, key, DELTA, CHK, 'coord')
                rec['perturb'][name] = {**stats_of(fail), 'abstain': False,
                                        'area': pred_area(fail)}
        # 真编辑
        att, box = paste_edit(wm, a, s['bbox'])
        fail_att = ks.fail_map(att, key, DELTA, CHK, 'coord')
        rec['edited'] = {**stats_of(fail_att), 'area': pred_area(fail_att),
                         'bbox_area': float((s['bbox'][2]-s['bbox'][0])*(s['bbox'][3]-s['bbox'][1])),
                         'in_fail': float(fail_att[block_gt(fail_att.shape, box)].mean())}
        rows.append(rec)
        if (idx + 1) % 50 == 0:
            el = time.time() - t0
            print(f'  [{idx+1}/{len(samples)}] {el:.0f}s 预计剩余 {el/(idx+1)*(len(samples)-idx-1):.0f}s',
                  flush=True)

    from sklearn.metrics import roc_auc_score
    edited_stats = np.array([r['edited']['top'] for r in rows])
    edited_con = np.array([r['edited']['contrast'] for r in rows])
    local = [r for r in rows if r['edited']['bbox_area'] < 0.9]
    full = [r for r in rows if r['edited']['bbox_area'] >= 0.9]
    print(f'\n=== 真编辑: 局部 {len(local)} / 全图 {len(full)} ===')

    summary, calib = {}, {}
    print(f'\n{"扰动":20s} {"top":>7s} {"FPR_top":>8s} {"contrast":>9s} {"FPR_con":>8s} '
          f'{"拒答":>6s} {"假阳性面积":>9s}')
    for stat_name in ['top', 'contrast']:
        cs = np.array([r['perturb']['clean'][stat_name] for r in rows])
        thr_s = float(np.percentile(cs, 99))
        es = np.array([r['edited'][stat_name] for r in rows])
        sub = 'top' if stat_name == 'top' else 'contrast'
        es_loc = np.array([r['edited'][sub] for r in local])
        es_full = np.array([r['edited'][sub] for r in full])
        calib[stat_name] = {
            'clean_mean': float(cs.mean()), 'clean_p95': float(np.percentile(cs, 95)),
            'thr_p99': thr_s,
            'tpr_all': float((es > thr_s).mean()),
            'tpr_local': float((es_loc > thr_s).mean()) if len(es_loc) else None,
            'tpr_full': float((es_full > thr_s).mean()) if len(es_full) else None,
            'auc_all': float(roc_auc_score(np.r_[np.zeros(len(cs)), np.ones(len(es))],
                                           np.r_[cs, es])),
        }
        for name in PERTURBS:
            v = [r['perturb'][name] for r in rows]
            st = np.array([x[stat_name] for x in v])
            ab = np.array([x['abstain'] for x in v], bool)
            area = np.array([x['area'] for x in v])
            fpr = float((st > thr_s).mean())
            summary.setdefault(name, {})[stat_name] = {
                'stat_mean': float(st.mean()),
                'stat_ci95': float(1.96 * st.std() / np.sqrt(len(st))),
                'fpr': fpr, 'abstain_rate': float(ab.mean()),
                'false_area_mean': float(area.mean()),
                'verdict': ('拒答' if ab.mean() > 0.5 else
                            ('误报风险高' if fpr > 0.20 else ('轻微' if fpr > 0.05 else '安全'))),
            }
        print(f'  --- 阈值 = clean p99 = {thr_s*100:.1f}% | TPR(all/local/full)='
              f'{calib[stat_name]["tpr_all"]*100:.0f}/'
              f'{(calib[stat_name]["tpr_local"] or 0)*100:.0f}/'
              f'{(calib[stat_name]["tpr_full"] or 0)*100:.0f}%  AUC={calib[stat_name]["auc_all"]:.4f}',
              flush=True)
    for name in PERTURBS:
        a, b = summary[name]['top'], summary[name]['contrast']
        print(f'{name:20s} {a["stat_mean"]*100:6.1f}% {a["fpr"]*100:7.1f}% '
              f'{b["stat_mean"]*100:8.1f}% {b["fpr"]*100:7.1f}% {a["abstain_rate"]*100:5.1f}% '
              f'{a["false_area_mean"]*100:8.2f}%')

    # ---------------- 工作曲线: 阈值扫描
    grid = {}
    for stat_name in ['top', 'contrast']:
        cs = np.array([r['perturb']['clean'][stat_name] for r in rows])
        es = np.array([r['edited'][stat_name] for r in rows])
        es_loc = np.array([r['edited'][stat_name] for r in local])
        ths = ([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60] if stat_name == 'top'
               else [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25])
        for t in ths:
            grid.setdefault(stat_name, {})[f'{t}'] = {
                'fpr_clean': float((cs > t).mean()),
                'tpr_local': float((es_loc > t).mean()) if len(es_loc) else None,
                'tpr_full': float(np.mean([r['edited'][stat_name] > t for r in full])) if len(full) else None,
                'fpr_jpeg50': float(np.mean([r['perturb']['jpeg50'][stat_name] > t for r in rows])),
                'fpr_blur3': float(np.mean([r['perturb']['blur3'][stat_name] > t for r in rows])),
                'fpr_noise10': float(np.mean([r['perturb']['noise10'][stat_name] > t for r in rows])),
                'fpr_crop': float(np.mean([r['perturb']['crop_arb_0.9'][stat_name] > t for r in rows])),
                'fpr_resize': float(np.mean([r['perturb']['resize_round_0.8'][stat_name] > t for r in rows])),
            }
    print(f'\n=== 工作曲线 (contrast: FPR_clean / TPR_local / FPR_blur3 / FPR_noise10) ===')
    for t, v in grid['contrast'].items():
        print(f'  t={t}: FPR={v["fpr_clean"]*100:5.1f}%  TPR_local={(v["tpr_local"] or 0)*100:5.1f}%  '
              f'blur3={v["fpr_blur3"]*100:5.1f}%  noise10={v["fpr_noise10"]*100:5.1f}%  '
              f'crop={v["fpr_crop"]*100:5.1f}%  resize={v["fpr_resize"]*100:5.1f}%')

    verdict = {
        'crop_no_longer_false_alarms': bool(summary['crop_arb_0.9']['top']['fpr'] < 0.05 and
                                            summary['crop_arb_0.9']['top']['abstain_rate'] < 0.10),
        'rotate_abstains': bool(summary['rotate_1']['top']['abstain_rate'] > 0.5),
        'jpeg_safe': bool(summary['jpeg75']['top']['fpr'] < 0.05),
        'contrast_immunity_blur': bool(summary['blur3']['contrast']['fpr'] < 0.10),
        'contrast_immunity_noise10': bool(summary['noise10']['contrast']['fpr'] < 0.10),
        'contrast_blind_to_full_edit': bool((calib['contrast']['tpr_full'] or 0) < 0.5),
        'top_sensitive_to_full_edit': bool((calib['top']['tpr_full'] or 0) > 0.8),
    }
    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n': len(rows),
                    'delta': DELTA, 'k_loc': K_LOC, 't_loc': T_LOC, 'qlow': QLOW,
                    'stat': f'top-{100-TOP_Q}% of {K_LOC}x{K_LOC}-smoothed fail map',
                    'thr_p99_clean': calib['top']['thr_p99'],
                    'thr_p99_contrast': calib['contrast']['thr_p99']},
           'summary': summary, 'workcurve': grid, 'verdict': verdict, 'calibration': calib,
           'edited': {'n_local': len(local), 'n_full': len(full)},
           'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r10_specificity_{fn}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n判定: ' + '  '.join(f'{k}={v}' for k, v in verdict.items()))
    print(f'结果: {fp}  ({out["runtime_sec"]}s)')


def block_gt(shape, box):
    by1, bx1, by2, bx2 = box
    n_h, n_w = int(shape[0]), int(shape[1])
    gt = np.zeros((n_h, n_w), bool)
    gt[by1 // 8:min((by2 + 7) // 8, n_h), bx1 // 8:min((bx2 + 7) // 8, n_w)] = True
    return gt


if __name__ == '__main__':
    main()
