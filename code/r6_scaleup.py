#!/usr/bin/env python3
"""G1+G2 扩样实验: attackset_20261001 (1200 样本, SD inpainting + MLLM 指令, bbox paste 协议).

与 eval_attackset.py 完全一致的攻击组合 (paste_edit: bbox 内取 SD 编辑输出, bbox 外取水印图),
因此本实验同时覆盖:
  G1 样本量扩充 (8/14 -> 1200) + 置信区间
  G2 编辑器多样性 (IP2P -> SD inpainting, 上一篇主攻击)

指标:
  图像级检测: clean_fail vs att_fail 的配对 AUC (全局重绘检测的扩样版)
  定位: 块级失败图 vs bbox GT -> AUC(原始/5x5平滑), IoU@0.5
  分层: bbox 面积三分位 (小<0.1 / 中 0.1-0.3 / 大>0.3, 全图编辑单列), 编辑类型(替换/删除)

输出: results/r6_scaleup_<ts>.json
用法: apjf-python r6_scaleup.py [--n 1200]
"""
import sys, os, json, argparse, datetime, time

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
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
DELTA, MSGLEN = 15, 256
PAY, CHK = (1, 0), (1, 1)


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def block_fail(img):
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    H, W = y.shape
    n_h, n_w = H // 8, W // 8
    exp = rs.check_bits(H, W, n_h, n_w)
    fail = np.zeros((n_h, n_w), np.float32)
    for i in range(n_h):
        for j in range(n_w):
            c = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8])[CHK]
            r = c % DELTA
            bit = 1 if abs(r - DELTA/2) < abs(r) else 0
            fail[i, j] = 1.0 if bit != exp[i, j] else 0.0
    return fail


def paste_edit(wm, att, bbox01):
    """bbox 内贴回攻击图, bbox 外保持水印图 (与 eval_attackset.paste_edit 一致)."""
    H, W = wm.shape[:2]
    x1, y1, x2, y2 = bbox01
    bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
    bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
    out = wm.copy()
    out[by1:by2, bx1:bx2] = att[by1:by2, bx1:bx2]
    return out, (by1, bx1, by2, bx2)


def auc(neg, pos):
    neg, pos = np.asarray(neg, float).ravel(), np.asarray(pos, float).ravel()
    if len(neg) == 0 or len(pos) == 0:
        return float('nan')
    return float(((pos[:, None] > neg[None, :]).sum() +
                  0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1200)
    ap.add_argument('--smooth', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    rs.PAY_COEF, rs.CHECK_COEF = PAY, CHK

    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:args.n]
    print(f'r6 扩样: n={len(samples)} (SD inpainting + bbox paste, {PAY}/{CHK}, delta={DELTA})')

    rng = np.random.RandomState(args.seed)
    msg = rng.randint(0, 2, MSGLEN).astype(np.uint8)
    k = args.smooth if args.smooth % 2 == 1 else args.smooth + 1

    rows, t0 = [], time.time()
    for idx, s in enumerate(samples):
        sid = s['id']
        try:
            o = crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        wm = rs.embed_signal(o, msg, DELTA)
        att_wm, (by1, bx1, by2, bx2) = paste_edit(wm, a, s['bbox'])

        f_clean = block_fail(wm)
        f_att = block_fail(att_wm)

        # bbox -> 块级 GT
        n_h, n_w = f_att.shape
        gt = np.zeros((n_h, n_w), bool)
        gt[by1//8:(by2+7)//8, bx1//8:(bx2+7)//8] = True
        full_edit = gt.mean() > 0.95
        in_b, out_b = gt, ~gt

        f_sm = cv2.blur(f_att, (k, k))
        pred = f_sm > 0.5
        pred_px = np.repeat(np.repeat(pred, 8, 0), 8, 1)
        gt_px = np.zeros(o.shape[:2], bool)
        gt_px[by1:by2, bx1:bx2] = True
        inter, union = (pred_px & gt_px).sum(), (pred_px | gt_px).sum()

        area = (s['bbox'][2]-s['bbox'][0]) * (s['bbox'][3]-s['bbox'][1])
        tier = 'full' if full_edit else ('small' if area < 0.10 else
                                         ('mid' if area < 0.30 else 'large'))
        ins = s['instruction']
        etype = 'replace' if ('替换' in ins or '换成' in ins or '换上' in ins) else \
                ('remove' if ('删除' in ins or '移除' in ins or '去掉' in ins) else 'other')

        rows.append({
            'id': sid, 'area': float(area), 'tier': tier, 'etype': etype,
            'clean_fail': float(f_clean.mean()), 'att_fail': float(f_att.mean()),
            'att_fail_in': float(f_att[in_b].mean()) if in_b.any() else float('nan'),
            'att_fail_out': float(f_att[out_b].mean()) if out_b.any() else float('nan'),
            'auc_raw': auc(f_att[out_b], f_att[in_b]) if (in_b.any() and out_b.any()) else float('nan'),
            'auc_smooth': auc(f_sm[out_b], f_sm[in_b]) if (in_b.any() and out_b.any()) else float('nan'),
            'iou': float(inter / union) if union else 0.0,
        })
        if (idx + 1) % 50 == 0:
            el = time.time() - t0
            print(f'  [{idx+1}/{len(samples)}] 已用 {el:.0f}s, 剩余约 {el/(idx+1)*(len(samples)-idx-1):.0f}s',
                  flush=True)

    # 汇总
    def col(key, filt=None):
        v = [r[key] for r in rows if filt is None or filt(r)]
        v = [x for x in v if not (isinstance(x, float) and np.isnan(x))]
        return np.array(v)

    def ci95(v):
        return float(1.96 * v.std() / np.sqrt(len(v))) if len(v) > 1 else float('nan')

    summary = {
        'n': len(rows),
        'clean_fail': [float(col('clean_fail').mean()), ci95(col('clean_fail'))],
        'att_fail': [float(col('att_fail').mean()), ci95(col('att_fail'))],
        'detect_auc_paired': auc(col('clean_fail'), col('att_fail')),
        'loc_auc_raw_mean': float(np.nanmean(col('auc_raw'))),
        'loc_auc_smooth_mean': float(np.nanmean(col('auc_smooth'))),
        'iou_mean': float(col('iou').mean()), 'iou_ci95': ci95(col('iou')),
    }
    print('\n=== 汇总 (n=%d) ===' % len(rows))
    print(f"  干净图校验失败率 : {summary['clean_fail'][0]*100:.1f}% ±{summary['clean_fail'][1]*100:.1f}")
    print(f"  编辑后校验失败率 : {summary['att_fail'][0]*100:.1f}% ±{summary['att_fail'][1]*100:.1f}")
    print(f"  图像级检测 AUC   : {summary['detect_auc_paired']:.4f} (配对, 跨 {len(rows)} 样本)")
    print(f"  定位 AUC 原始/平滑: {summary['loc_auc_raw_mean']:.3f} / {summary['loc_auc_smooth_mean']:.3f}")
    print(f"  定位 IoU@0.5     : {summary['iou_mean']:.3f} ±{summary['iou_ci95']:.3f}")

    print('\n=== 分层: bbox 面积 ===')
    for tier in ['small', 'mid', 'large', 'full']:
        rr = [r for r in rows if r['tier'] == tier]
        if not rr:
            continue
        au = np.nanmean([r['auc_smooth'] for r in rr])
        iou = np.nanmean([r['iou'] for r in rr])
        fi = np.nanmean([r['att_fail_in'] for r in rr])
        fo = np.nanmean([r['att_fail_out'] for r in rr])
        print(f'  {tier:6s} n={len(rr):4d} in_fail={fi*100:5.1f}% out_fail={fo*100:5.1f}% '
              f'auc_sm={au:.3f} iou={iou:.3f}')
    print('\n=== 分层: 编辑类型 ===')
    for et in ['replace', 'remove', 'other']:
        rr = [r for r in rows if r['etype'] == et]
        if not rr:
            continue
        au = np.nanmean([r['auc_smooth'] for r in rr])
        print(f'  {et:8s} n={len(rr):4d} auc_sm={au:.3f} '
              f'iou={np.nanmean([r["iou"] for r in rr]):.3f}')

    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r6_scaleup_{ts}.json'
    json.dump({'meta': {'n': len(rows), 'coef': f'{PAY}/{CHK}', 'delta': DELTA,
                        'msglen': MSGLEN, 'seed': args.seed, 'smooth': k,
                        'attackset': 'attackset_20261001_222447 (SD inpaint + bbox paste)',
                        'ts': datetime.datetime.now().isoformat()},
               'summary': summary, 'rows': rows}, open(fp, 'w'), indent=1)
    print(f'\n结果: {fp}')


if __name__ == '__main__':
    main()
