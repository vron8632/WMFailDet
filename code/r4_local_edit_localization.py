#!/usr/bin/env python3
"""局部编辑下的"水印死亡=编辑证据"检测 + 定位判定实验。

对照全局重绘 (results/r3_death_signal_*): 图像级 AUC=1.000, 块级 AUC=0.62 (定位不可用).
局部编辑下预期: 被编辑区域内块校验死亡, 区域外保留 -> 块级应远好于 0.62, 定位可行.

协议 (与上一篇一致的同族局部语义编辑):
  o    = 原始图
  wm   = embed_signal(o, msg, delta=15)   低频 (1,0)/(1,1) 配置
  full = IP2P(wm, instruction)            整图语义编辑
  att  = mask 内取 full, mask 外取 wm     (paste-back, 严格局部)
数据: attackset_hires_sample/{images,meta.json} (含 _orig.png / _mask.png, GT mask 面积~10%)
输出: results/r4_local_edit_localization_<ts>.json
用法: HF_HOME=... apjf-python r4_local_edit_localization.py [--n 16]
"""
import sys, os, json, argparse, glob, datetime

os.environ.setdefault('HF_HOME', '/media/oyp/工作/Projects/042_image_forensic/huggingface')
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
HIRES = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
         '/results/attackset_hires_sample')
DEVICE = 'cuda'
DELTA, MSGLEN = 15, 256
PAY, CHK = (1, 0), (1, 1)


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def block_fail_map(img):
    """逐块校验失败掩膜 (n_h, n_w) float: 1=失败."""
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    H, W = y.shape
    n_h, n_w = H // 8, W // 8
    expect = rs.check_bits(H, W, n_h, n_w)
    fail = np.zeros((n_h, n_w), np.float32)
    for i in range(n_h):
        for j in range(n_w):
            c = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8])[CHK]
            r = c % DELTA
            bit = 1 if abs(r - DELTA/2) < abs(r) else 0
            fail[i, j] = 1.0 if bit != expect[i, j] else 0.0
    return fail


def block_mask_gt(mask_px):
    """像素 mask -> 8x8 块级 GT (块内 mask 占比 > 0.5 记为编辑块)."""
    H, W = mask_px.shape[:2]
    n_h, n_w = H // 8, W // 8
    gt = np.zeros((n_h, n_w), np.float32)
    for i in range(n_h):
        for j in range(n_w):
            gt[i, j] = mask_px[i*8:(i+1)*8, j*8:(j+1)*8].mean()
    return gt


def auc(neg, pos):
    neg, pos = np.asarray(neg, float).ravel(), np.asarray(pos, float).ravel()
    if len(neg) == 0 or len(pos) == 0:
        return float('nan')
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smooth', type=int, default=5, help='块级失败图平滑窗 (块单位)')
    args = ap.parse_args()

    rs.PAY_COEF, rs.CHECK_COEF = PAY, CHK
    meta = json.load(open(f'{HIRES}/meta.json'))
    seen, samples = set(), []
    for s in meta['samples']:
        b = s['id'].rsplit('_', 1)[0]
        mp = f'{HIRES}/images/{s["id"]}_mask.png'
        if b in seen or not os.path.exists(mp):
            continue
        seen.add(b)
        samples.append(s)
    samples = samples[:args.n]
    print(f'局部编辑检测+定位: n={len(samples)} (IP2P + mask paste-back), coef={PAY}/{CHK}')

    from diffusers import StableDiffusionInstructPix2PixPipeline
    import torch
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', torch_dtype=torch.float16,
        safety_checker=None).to(DEVICE)
    print(f'  模型已加载 ({DEVICE})')

    rng = np.random.RandomState(args.seed)
    msg = rng.randint(0, 2, MSGLEN).astype(np.uint8)
    rows = []
    for idx, s in enumerate(samples):
        cid = s['id']
        o = crop8(np.array(Image.open(f'{HIRES}/images/{cid}_orig.png').convert('RGB')))
        mp = np.array(Image.open(f'{HIRES}/images/{cid}_mask.png').convert('L'))
        mp = mp[:o.shape[0], :o.shape[1]]
        m8 = np.zeros(o.shape[:2], bool)
        m8[:mp.shape[0], :mp.shape[1]] = mp > 127

        wm = rs.embed_signal(o, msg, DELTA)
        g = torch.Generator(device=DEVICE).manual_seed(args.seed)
        full = pipe(prompt=s['instruction'], image=Image.fromarray(wm),
                    num_inference_steps=args.steps, generator=g,
                    image_guidance_scale=1.5, guidance_scale=7.5).images[0]
        full = crop8(np.array(full.convert('RGB')))
        att = np.where(m8[:, :, None], full, wm).astype(np.uint8)

        f_clean = block_fail_map(wm)
        f_att = block_fail_map(att)
        gt_blk = block_mask_gt(m8.astype(np.float32))
        in_blk = gt_blk > 0.5
        out_blk = gt_blk < 0.01

        # 平滑后的失败分数 (定位用)
        k = args.smooth if args.smooth % 2 == 1 else args.smooth + 1
        f_smooth = cv2.blur(f_att, (k, k))

        row = {
            'img': cid, 'mask_frac': float(m8.mean()),
            'clean_fail': float(f_clean.mean()),
            'att_fail_all': float(f_att.mean()),
            'att_fail_in': float(f_att[in_blk].mean()) if in_blk.any() else float('nan'),
            'att_fail_out': float(f_att[out_blk].mean()) if out_blk.any() else float('nan'),
            'blk_auc': auc(f_att[out_blk], f_att[in_blk]) if in_blk.any() and out_blk.any() else float('nan'),
            'blk_auc_smooth': auc(f_smooth[out_blk], f_smooth[in_blk]) if in_blk.any() and out_blk.any() else float('nan'),
            'n_in': int(in_blk.sum()), 'n_out': int(out_blk.sum()),
        }
        # 区域级定位: 平滑分数阈值化 -> 与 GT mask 的像素级 IoU
        thr = 0.5
        pred_blk = f_smooth > thr
        pred_px = np.repeat(np.repeat(pred_blk, 8, 0), 8, 1)[:m8.shape[0], :m8.shape[1]]
        inter = (pred_px & m8).sum()
        union = (pred_px | m8).sum()
        row['iou_thr0.5'] = float(inter / union) if union else 0.0
        row['pred_frac'] = float(pred_px.mean())
        rows.append(row)
        print(f"  [{idx+1}/{len(samples)}] {cid:8s} mask={row['mask_frac']*100:4.1f}% "
              f"clean_fail={row['clean_fail']*100:4.1f}% att_fail: in={row['att_fail_in']*100:5.1f}% "
              f"out={row['att_fail_out']*100:5.1f}% | blkAUC={row['blk_auc']:.3f} "
              f"smooth={row['blk_auc_smooth']:.3f} IoU={row['iou_thr0.5']:.3f}")

    def m(k):
        v = [r[k] for r in rows if not (isinstance(r[k], float) and np.isnan(r[k]))]
        return float(np.mean(v)) if v else float('nan')

    print('\n=== 汇总 (n=%d) ===' % len(rows))
    print(f"  干净图校验失败率      : {m('clean_fail')*100:.1f}%")
    print(f"  局部编辑后(全体)      : {m('att_fail_all')*100:.1f}%")
    print(f"  编辑区域内块失败率    : {m('att_fail_in')*100:.1f}%  (应接近 50% = 信号死亡)")
    print(f"  编辑区域外块失败率    : {m('att_fail_out')*100:.1f}%  (应接近干净图 = 信号保留)")
    print(f"  块级定位 AUC (原始)   : {m('blk_auc'):.3f}")
    print(f"  块级定位 AUC (平滑)   : {m('blk_auc_smooth'):.3f}")
    print(f"  区域级定位 IoU@0.5    : {m('iou_thr0.5'):.3f}")
    print('  对照: 全局重绘下块级 AUC = 0.62 (定位不可用)')

    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r4_local_edit_localization_{ts}.json'
    json.dump({'meta': {'n': len(rows), 'coef': f'{PAY}/{CHK}', 'delta': DELTA,
                        'smooth': args.smooth, 'ts': datetime.datetime.now().isoformat()},
               'rows': rows,
               'summary': {k: m(k) for k in ['clean_fail', 'att_fail_all', 'att_fail_in',
                                             'att_fail_out', 'blk_auc', 'blk_auc_smooth',
                                             'iou_thr0.5']}},
              open(fp, 'w'), indent=1)
    print(f'\n结果: {fp}')


if __name__ == '__main__':
    main()
