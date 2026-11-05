#!/usr/bin/env python3
"""G6 方法对比（同协议）: TrustMark vs 我们的"死亡信号".

对比点 = **检测/定位能力**, 不是 bit accuracy:
  TrustMark 为"恢复消息"设计 (100 bit + ECC, 任意分辨率) -> 局部编辑后整图仍可能解码成功;
  我们为"检测水印死亡"设计 -> 编辑区死亡 -> 检测 + 定位.
公平性: 同一批图、同一攻击集 (attackset SD inpainting + bbox paste)、同一 clean/edited 配对.

TrustMark 的检测量:
  present  = decode 是否检测到水印 (bool)
  match    = 恢复的 secret 是否等于原始 secret
  判"被编辑" = (not present) or (not match)
TrustMark 的定位尝试:
  128x128 滑窗 (步长 64) + subimage_decode -> "有水印"图; 编辑区应无水印.
  (TrustMark 原生 localize 需要额外 BBox detector, 这里用子图解码作为公平的定位施压)

我们的方法 (r9/r10 工作点): top-10% 块均值 / contrast=top-median, 阈值 0.41 / 0.15;
  定位: 9x9 平滑 > 0.38 + open_close + min_area=8 块.

输出: results/r11_method_comparison_<ts>.json
用法: apjf-python r11_method_comparison.py [--n 200] [--n-sub 30]
"""
import sys, os, json, argparse, datetime, time

TM_ROOT = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
           '/baselines/trustmark/python')
sys.path.insert(0, TM_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from PIL import Image

import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit, block_gt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
DELTA = 15
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))
TOP_THR, CON_THR = 0.41, 0.15
K_LOC, T_LOC = 9, 0.38


def ours_stats(fail):
    s = ks.smooth_fail_map(fail, K_LOC)
    vals = np.sort(s.ravel())
    n = max(1, int(0.10 * vals.size))
    top = float(vals[-n:].mean())
    return top, top - float(np.median(vals))


def ours_localize(fail):
    pred = ks.smooth_fail_map(fail, K_LOC) > T_LOC
    p = pred.astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
    keep = np.zeros_like(p)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= 8:
            keep[lab == c] = 1
    return keep.astype(bool)


def pixel_iou(pred_blocks, bbox, shape):
    H, W = shape[0] // 8 * 8, shape[1] // 8 * 8
    x1, y1, x2, y2 = bbox
    g = np.zeros((H, W), bool)
    g[int(round(y1 * H)):int(round(y2 * H)), int(round(x1 * W)):int(round(x2 * W))] = True
    p = np.repeat(np.repeat(pred_blocks, 8, 0), 8, 1)[:H, :W]
    inter, union = (p & g).sum(), (p | g).sum()
    return float(inter / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--n-sub', type=int, default=30)
    ap.add_argument('--skip-sub', action='store_true')
    args = ap.parse_args()
    t0 = time.time()

    from trustmark import TrustMark
    tm = TrustMark(verbose=False, model_type='Q', device='cpu')
    print(f'TrustMark 已加载 ({time.time()-t0:.1f}s)')

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
        secret = f'wm{sid}'
        key = ks.image_key(MASTER, sid)

        # ---- TrustMark
        wm_tm = np.array(tm.encode(Image.fromarray(o), secret).convert('RGB'))
        att_tm, box = paste_edit(wm_tm, a, s['bbox'])
        s_c, p_c, _ = tm.decode(Image.fromarray(wm_tm))
        s_a, p_a, _ = tm.decode(Image.fromarray(att_tm))

        # ---- 我们的方法
        wm_k = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        att_k, _ = paste_edit(wm_k, a, s['bbox'])
        f_att = ks.fail_map(att_k, key, DELTA, CHK, 'coord')
        top, con = ours_stats(f_att)
        pred = ours_localize(f_att)

        rows.append({
            'id': sid,
            'tm': {'clean_present': bool(p_c), 'clean_match': bool(s_c == secret),
                   'att_present': bool(p_a), 'att_match': bool(s_a == secret)},
            'ours': {'top': top, 'contrast': con,
                     'iou': pixel_iou(pred, s['bbox'], o.shape)},
        })
        if (idx + 1) % 50 == 0:
            print(f'  [{idx+1}/{len(samples)}] {time.time()-t0:.0f}s', flush=True)

    n = len(rows)

    def rate(fn):
        return float(np.mean([fn(r) for r in rows]))

    tm_res = {
        'n': n,
        'clean_present_rate': rate(lambda r: r['tm']['clean_present']),
        'clean_match_rate': rate(lambda r: r['tm']['clean_match']),
        'att_present_rate': rate(lambda r: r['tm']['att_present']),
        'att_match_rate': rate(lambda r: r['tm']['att_match']),
        'detect_rate_lenient': rate(lambda r: (not r['tm']['att_present']) or (not r['tm']['att_match'])),
        'detect_rate_strict': rate(lambda r: not r['tm']['att_present']),
        'false_alarm_rate': rate(lambda r: (not r['tm']['clean_present']) or (not r['tm']['clean_match'])),
    }
    ours_res = {
        'n': n,
        'top_mean': float(np.mean([r['ours']['top'] for r in rows])),
        'contrast_mean': float(np.mean([r['ours']['contrast'] for r in rows])),
        'detect_rate_top': rate(lambda r: r['ours']['top'] > TOP_THR),
        'detect_rate_contrast': rate(lambda r: r['ours']['contrast'] > CON_THR),
        'iou_mean': float(np.mean([r['ours']['iou'] for r in rows])),
    }
    print(f'\n=== 检测 (n={n}, clean 误报 / edited 检出) ===')
    print(f'  TrustMark: clean present={tm_res["clean_present_rate"]*100:.1f}% '
          f'match={tm_res["clean_match_rate"]*100:.1f}% | 编辑后 present={tm_res["att_present_rate"]*100:.1f}% '
          f'match={tm_res["att_match_rate"]*100:.1f}%')
    print(f'    -> 误报率 {tm_res["false_alarm_rate"]*100:.1f}%  检出率(宽松) {tm_res["detect_rate_lenient"]*100:.1f}%  '
          f'(严格) {tm_res["detect_rate_strict"]*100:.1f}%')
    print(f'  我们: top 检出率 {ours_res["detect_rate_top"]*100:.1f}%  '
          f'contrast 检出率 {ours_res["detect_rate_contrast"]*100:.1f}%  定位 IoU {ours_res["iou_mean"]:.3f}')

    # ---- TrustMark 子图定位 (128 窗口, 步长 64)
    sub = None
    if not args.skip_sub:
        sub_rows = []
        for r, s in zip(rows[:args.n_sub], samples[:args.n_sub]):
            sid = s['id']
            o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
            secret = f'wm{sid}'
            wm_tm = np.array(tm.encode(Image.fromarray(o), secret).convert('RGB'))
            att_tm, box = paste_edit(wm_tm, a, s['bbox'])
            H, W = att_tm.shape[:2]
            win, step = 128, 64
            pr = np.zeros((3, 3), np.float32)
            gt = np.zeros((3, 3), np.float32)
            for i in range(3):
                for j in range(3):
                    y0, x0 = i * step, j * step
                    crop = att_tm[y0:y0 + win, x0:x0 + win]
                    _, present, _ = tm.subimage_decode(Image.fromarray(crop), 'text')
                    pr[i, j] = 0.0 if present else 1.0     # 1 = 无水印 = 疑似编辑
                    gt[i, j] = 1.0 if (box[1] < x0 + win and box[3] > x0 and
                                       box[0] < y0 + win and box[2] > y0) else 0.0
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(gt.ravel(), pr.ravel())) if 0 < gt.sum() < gt.size else float('nan')
            sub_rows.append({'id': sid, 'auc': auc, 'present': 1 - pr.mean(),
                             'gt_area': float(gt.mean())})
            if len(sub_rows) % 10 == 0:
                print(f'  子图定位 [{len(sub_rows)}/{args.n_sub}] {time.time()-t0:.0f}s', flush=True)
        sub = {
            'n': len(sub_rows),
            'window': 128, 'stride': 64,
            'auc_mean': float(np.nanmean([x['auc'] for x in sub_rows])),
            'pred_area_mean': float(np.mean([1 - x['present'] for x in sub_rows])),
            'gt_area_mean': float(np.mean([x['gt_area'] for x in sub_rows])),
            'auc_valid_n': int(np.sum([not np.isnan(x['auc']) for x in sub_rows])),
        }
        print(f'  TrustMark 子图定位: AUC={sub["auc_mean"]:.3f} (有效 {sub["auc_valid_n"]}/{sub["n"]}), '
              f'pred 面积={sub["pred_area_mean"]*100:.1f}% vs GT {sub["gt_area_mean"]*100:.1f}%')

    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n': n, 'n_sub': args.n_sub,
                    'protocol': 'attackset SD inpaint + bbox paste; clean/edited 配对',
                    'ours_workpoint': {'top_thr': TOP_THR, 'con_thr': CON_THR,
                                       'k_loc': K_LOC, 't_loc': T_LOC},
                    'note': 'TrustMark = 100bit+ECC 任意分辨率; 对比点是检测/定位能力而非 BA'},
           'trustmark': tm_res, 'ours': ours_res, 'trustmark_subimage_localization': sub,
           'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r11_method_comparison_{fn}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n结果: {fp}  ({out["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
