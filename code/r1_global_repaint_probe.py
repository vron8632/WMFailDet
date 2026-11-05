#!/usr/bin/env python3
"""R1 最小实验：全局重绘（full IP2P inpainting, no paste-back）下可恢复信号探查。

背景:
  已有结果 (results/full_inpaint_20260913_192207.json, n=4) 显示当前嵌入配置
  PAY=(4,1)/CHECK=(3,2), delta=15, 256bit 在全局重绘后 BA≈50.8% (完全随机).
  文献侧: BrushOff (arXiv2610.26682) 证明 relative-brightness 低频 probe 存活
  repainting; SimuFreeMark (arXiv2511.11295) 证明 AIGC 重绘保留低频结构.
  推论: 把 QIM 系数移到更低频可能存在非零恢复空间.

实验: 同一批图像、同一条消息、同一扩散采样噪声 (固定 generator seed),
      对比 3 组 (payload, check) 系数配置在全局重绘后的 bit accuracy:
        V0 current : PAY=(4,1) CHECK=(3,2)   [对照, 重跑统一协议]
        V1 lowfreq1: PAY=(1,0) CHECK=(1,1)   [更低频]
        V2 lowfreq2: PAY=(2,1) CHECK=(1,0)   [中间频带]

判据 (见 docs/RESEARCH_PLAN.md R1): 任一变体 attack BA 显著 >50% → C 保留;
全部 ≈50% → C 降级为边界诊断.

输出: ../results/r1_global_repaint_probe_<ts>.json
      攻击输出缓存: ../data/r1_attack_cache/<variant>/<imgid>.png

用法: HF_HOME=<042_image_forensic/huggingface> apjf-python r1_global_repaint_probe.py \
        [--n 12] [--msglen 256] [--delta 15] [--steps 30] [--variants V0,V1,V2]
"""
import sys, os, json, argparse, datetime

# 必须在 import diffusers/torch 前设置 (模型缓存位置)
os.environ.setdefault(
    'HF_HOME', '/media/oyp/工作/Projects/042_image_forensic/huggingface')

OLD_CODE = '/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/code'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, OLD_CODE)

import numpy as np
import cv2
from PIL import Image

# 旧项目 run_experiments.py 硬编码了已不存在的数据盘路径 (/media/oyp/数据/...),
# 其模块级 os.makedirs 会 PermissionError. 仅在 import 期间跳过该路径的建目录
# (我们不使用其 RESULTS_DIR), import 完成后立即恢复原函数.
_orig_makedirs = os.makedirs
def _makedirs_skip_legacy(name, *a, **k):
    if str(name).startswith('/media/oyp/数据'):
        return None
    return _orig_makedirs(name, *a, **k)
os.makedirs = _makedirs_skip_legacy
try:
    import run_signal_ra as rs
finally:
    os.makedirs = _orig_makedirs
from dct_watermark import bit_accuracy

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # 新项目根
HIRES = '/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/results/attackset_hires_sample'
DEVICE = 'cuda'

# 变体: name -> (PAY_COEF, CHECK_COEF)
VARIANTS = {
    'V0_current':  ((4, 1), (3, 2)),
    'V1_lowfreq1': ((1, 0), (1, 1)),
    'V2_lowfreq2': ((2, 1), (1, 0)),
}


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def psnr(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    m = np.mean((a - b) ** 2)
    return float('inf') if m == 0 else float(10 * np.log10(255.0 ** 2 / m))


def apply_variant(name):
    """monkey-patch run_signal_ra 的系数 (函数体引用模块全局字典, patch 有效)."""
    pay, chk = VARIANTS[name]
    rs.PAY_COEF, rs.CHECK_COEF = pay, chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--delta', type=int, default=15)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--dedupe', action='store_true',
                    help='按原图 base id 去重 (同 base 同 instruction 的多样本完全冗余)')
    ap.add_argument('--cache-only', action='store_true',
                    help='只生成攻击缓存 (同图同变体只跑一次扩散)')
    args = ap.parse_args()
    variants = args.variants.split(',')

    meta = json.load(open(f'{HIRES}/meta.json'))
    samples = meta['samples'][:args.n]
    if args.dedupe:
        seen, dedup = set(), []
        for s in samples:
            b = s['id'].rsplit('_', 1)[0]
            if b not in seen:
                seen.add(b)
                dedup.append(s)
        print(f'dedupe: {len(samples)} -> {len(dedup)} 唯一原图')
        samples = dedup
    print(f'R1 probe: n={len(samples)} variants={variants} '
          f'msglen={args.msglen} delta={args.delta} steps={args.steps}')

    from diffusers import StableDiffusionInstructPix2PixPipeline
    import torch
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', torch_dtype=torch.float16,
        safety_checker=None).to(DEVICE)
    print(f'  模型已加载 ({DEVICE})')

    rng = np.random.RandomState(args.seed)
    msg = rng.randint(0, 2, args.msglen).astype(np.uint8)

    cache_root = f'{BASE}/data/r1_attack_cache'
    out = {'meta': {'n': len(samples), 'variants': variants,
                    'msglen': args.msglen, 'delta': args.delta,
                    'steps': args.steps, 'seed': args.seed,
                    'protocol': 'full_ip2p_no_paste_fixed_noise',
                    'ts': datetime.datetime.now().isoformat()},
           'per_image': [], 'summary': {}}

    for v in variants:
        apply_variant(v)
        os.makedirs(f'{cache_root}/{v}', exist_ok=True)
        rows = []
        for idx, s in enumerate(samples):
            cid = s['id']
            try:
                o = crop8(np.array(Image.open(
                    f'{HIRES}/images/{cid}_orig.png').convert('RGB')))
            except FileNotFoundError:
                continue

            # 1) 嵌入 (变体系数)
            wm = rs.embed_signal(o, msg, args.delta)
            row = {'img': cid, 'variant': v}
            row['clean_psnr'] = psnr(o, wm)
            me, _ = rs.decode_signal(wm, args.msglen, args.delta, use_check=False)
            row['clean_ba'] = float(bit_accuracy(msg, me))

            # 2) 攻击: 固定采样噪声 (每变体每图相同 seed, 保证变体间可比)
            cache_p = f'{cache_root}/{v}/{cid}.png'
            if os.path.exists(cache_p):
                att = crop8(np.array(Image.open(cache_p).convert('RGB')))
                row['cached'] = True
            else:
                g = torch.Generator(device=DEVICE).manual_seed(args.seed)
                gen = pipe(prompt=s['instruction'], image=Image.fromarray(wm),
                           num_inference_steps=args.steps, generator=g,
                           image_guidance_scale=1.5,
                           guidance_scale=7.5).images[0]
                att = crop8(np.array(gen.convert('RGB')))
                Image.fromarray(att).save(cache_p)
                row['cached'] = False

            # 3) 解码: uni / v2
            me, _ = rs.decode_signal(att, args.msglen, args.delta, use_check=False)
            row['uni_ba'] = float(bit_accuracy(msg, me))
            me, _ = rs.decode_signal(att, args.msglen, args.delta, use_check=True,
                                     soft=True, dilate=1)
            row['v2_ba'] = float(bit_accuracy(msg, me))
            rows.append(row)
            out['per_image'].append(row)
            print(f'  [{v} {idx+1}/{len(samples)}] {cid} cleanPSNR={row["clean_psnr"]:.1f} '
                  f'cleanBA={row["clean_ba"]*100:.0f}% uni={row["uni_ba"]*100:.1f}% '
                  f'v2={row["v2_ba"]*100:.1f}%')

        if rows:
            out['summary'][v] = {
                'n': len(rows),
                'pay_chk': list(VARIANTS[v][0]) + list(VARIANTS[v][1]),
                'clean_psnr': float(np.mean([r['clean_psnr'] for r in rows])),
                'clean_ba': float(np.mean([r['clean_ba'] for r in rows])),
                'uni_ba': float(np.mean([r['uni_ba'] for r in rows])),
                'v2_ba': float(np.mean([r['v2_ba'] for r in rows])),
            }
            s = out['summary'][v]
            print(f'  ==> {v}: cleanPSNR={s["clean_psnr"]:.1f}dB '
                  f'uni={s["uni_ba"]*100:.1f}% v2={s["v2_ba"]*100:.1f}% '
                  f'(随机线 50%)')

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r1_global_repaint_probe_{ts}.json'
    os.makedirs(f'{BASE}/results', exist_ok=True)
    json.dump(out, open(fp, 'w'), indent=1, ensure_ascii=False)
    print(f'\n结果: {fp}')
    print('=== R1 判据 ===')
    for v, s in out['summary'].items():
        best = max(s['uni_ba'], s['v2_ba'])
        verdict = '✅ 非零恢复空间' if best > 0.53 else ('⚠️ 边缘' if best > 0.515 else '❌ 随机水平')
        print(f'  {v}: best={best*100:.1f}% {verdict}')


if __name__ == '__main__':
    main()
