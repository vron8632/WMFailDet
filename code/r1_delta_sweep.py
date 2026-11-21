#!/usr/bin/env python3
"""判定实验：全局重绘下，加大嵌入强度 (delta) 能否提升水印存活？

假说 H_amp  : 存活率由信号幅度决定 -> BA 随 delta 单调上升.
假说 H_struct: 存活率由"图像结构被重绘保留的比例"决定 -> BA 随 delta 饱和 (平台期).

这是"结构寄生"主张的判定性实验: 若 BA 在 delta 增大后迅速饱和, 则
"刻得更深"无效, 水印鲁棒性的上限由图像结构决定.

协议: 低频配置 (1,0)/(1,1) (R1 最佳), 每 delta 重新嵌入 -> 重新扩散 (固定噪声 seed),
      记录 clean PSNR (保真度代价) 与 attack BA.
输出: results/r1_delta_sweep_<ts>.json
用法: HF_HOME=... apjf-python r1_delta_sweep.py [--n 6] [--deltas 8,15,25,40]
"""
import sys, os, json, argparse, datetime

os.environ.setdefault('HF_HOME', '/media/oyp/工作/Projects/042_image_forensic/huggingface')
OLD_CODE = '/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/code'
sys.path.insert(0, OLD_CODE)

import numpy as np
from PIL import Image

_o = os.makedirs
def _skip(name, *a, **k):
    return None if str(name).startswith('/media/oyp/数据') else _o(name, *a, **k)
os.makedirs = _skip
try:
    import run_signal_ra as rs
finally:
    os.makedirs = _o
from dct_watermark import bit_accuracy

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRES = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
         '/results/attackset_hires_sample')
DEVICE = 'cuda'


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def psnr(a, b):
    m = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float('inf') if m == 0 else float(10 * np.log10(255.0 ** 2 / m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--deltas', default='8,15,25,40')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    deltas = [int(x) for x in args.deltas.split(',')]

    rs.PAY_COEF, rs.CHECK_COEF = (1, 0), (1, 1)   # R1 最优低频配置

    meta = json.load(open(f'{HIRES}/meta.json'))
    seen, samples = set(), []
    for s in meta['samples']:
        b = s['id'].rsplit('_', 1)[0]
        if b not in seen:
            seen.add(b)
            samples.append(s)
    samples = samples[:args.n]
    print(f'delta sweep: n={len(samples)} deltas={deltas} (低频 (1,0)/(1,1), 固定噪声)')

    from diffusers import StableDiffusionInstructPix2PixPipeline
    import torch
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', torch_dtype=torch.float16,
        safety_checker=None).to(DEVICE)
    print(f'  模型已加载 ({DEVICE})')

    rng = np.random.RandomState(args.seed)
    msg = rng.randint(0, 2, args.msglen).astype(np.uint8)
    out = {'meta': {'n': len(samples), 'deltas': deltas, 'coef': 'pay(1,0)/chk(1,1)',
                    'msglen': args.msglen, 'steps': args.steps,
                    'ts': datetime.datetime.now().isoformat()},
           'per_image': [], 'summary': {}}

    for delta in deltas:
        rows = []
        for idx, s in enumerate(samples):
            cid = s['id']
            o = crop8(np.array(Image.open(
                f'{HIRES}/images/{cid}_orig.png').convert('RGB')))
            wm = rs.embed_signal(o, msg, delta)
            row = {'img': cid, 'delta': delta, 'clean_psnr': psnr(o, wm)}
            g = torch.Generator(device=DEVICE).manual_seed(args.seed)
            gen = pipe(prompt=s['instruction'], image=Image.fromarray(wm),
                       num_inference_steps=args.steps, generator=g,
                       image_guidance_scale=1.5, guidance_scale=7.5).images[0]
            att = crop8(np.array(gen.convert('RGB')))
            me, _ = rs.decode_signal(att, args.msglen, delta, use_check=False)
            row['uni_ba'] = float(bit_accuracy(msg, me))
            me, _ = rs.decode_signal(att, args.msglen, delta, use_check=True,
                                     soft=True, dilate=1)
            row['v2_ba'] = float(bit_accuracy(msg, me))
            rows.append(row)
            out['per_image'].append(row)
            print(f'  [d={delta} {idx+1}/{len(samples)}] {cid} cleanPSNR={row["clean_psnr"]:.1f} '
                  f'uni={row["uni_ba"]*100:.1f}% v2={row["v2_ba"]*100:.1f}%')
        out['summary'][str(delta)] = {
            'n': len(rows),
            'clean_psnr': float(np.mean([r['clean_psnr'] for r in rows])),
            'uni_ba': float(np.mean([r['uni_ba'] for r in rows])),
            'v2_ba': float(np.mean([r['v2_ba'] for r in rows])),
        }
        s_ = out['summary'][str(delta)]
        print(f'  ==> delta={delta}: cleanPSNR={s_["clean_psnr"]:.1f}dB '
              f'uni={s_["uni_ba"]*100:.1f}% v2={s_["v2_ba"]*100:.1f}%')

    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r1_delta_sweep_{ts}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n结果: {fp}')
    print('\n=== 判定 ===')
    bas = [out['summary'][str(d)]['uni_ba'] * 100 for d in deltas]
    print('  delta ->  BA:', ' | '.join(f'{d}:{b:.1f}%' for d, b in zip(deltas, bas)))
    gain = bas[-1] - bas[0]
    print(f'  最大 delta 相对最小: {gain:+.1f}pt'
          + ('  → H_amp (幅度主导)' if gain > 8 else '  → H_struct (结构饱和)'))


if __name__ == '__main__':
    main()
