#!/usr/bin/env python3
"""判定实验：Write comparisons, not values —— 块对差分 QIM vs 逐块绝对 QIM。

动机 (来自前两个被否定的假说):
  - 格点相位漂移与 BA 无关 (r=0.001)  -> 不是相位问题
  - delta 8->40 使 BA 51.9%->68.0% 但 PSNR 50.6->41.3dB, 无饱和 -> 幅度有用但不可用,
    逐块绝对系数 QIM 存在原理性天花板.

待检验洞见: 扩散重绘会重写每个块的绝对系数, 但很少反转块与块之间的相对关系
(保语义需保相对结构). 若成立, 把消息编码在"块对差值"而非"块绝对值"上, 应在
等 PSNR 条件下显著优于绝对 QIM.

公平性: 差分嵌入把 Δ 平分到两个块, 每块期望改动 |Δ|/2 ~ delta/4,
        与绝对 QIM 的每块改动期望 delta/4 相同 -> PSNR 可比.

对照: results/r1_delta_sweep_*(delta=15, n=6, 同图同噪声 seed) = 60.4%
输出: results/r1_pairwise_probe_<ts>.json
用法: HF_HOME=... apjf-python r1_pairwise_probe.py [--n 6] [--deltas 8,15,25]
"""
import sys, os, json, argparse, datetime

os.environ.setdefault('HF_HOME', '/media/oyp/工作/Projects/042_image_forensic/huggingface')
OLD_CODE = '/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/code'
sys.path.insert(0, OLD_CODE)

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
from dct_watermark import bit_accuracy

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRES = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
         '/results/attackset_hires_sample')
DEVICE = 'cuda'
COEF = (1, 0)          # 与 R1 最优低频配置一致


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def psnr(a, b):
    m = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float('inf') if m == 0 else float(10 * np.log10(255.0 ** 2 / m))


def pair_grid(H, W):
    n_h, n_w = H // 8, W // 8
    n_pairs_per_row = n_w // 2
    return n_h, n_w, n_pairs_per_row


def embed_pairwise(img, msg, delta=15):
    """块对差分 QIM: bit 编码在水平相邻块的系数差上."""
    H, W = img.shape[0], img.shape[1]
    n_h, n_w, npr = pair_grid(H, W)
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)[:H*8//8, :W]
    y = y[:n_h*8, :n_w*8]
    L = msg_len = len(msg)
    for i in range(n_h):
        for jj in range(npr):
            j1, j2 = 2*jj, 2*jj+1
            b1 = cv2.dct(y[i*8:(i+1)*8, j1*8:(j1+1)*8].astype(np.float32))
            b2 = cv2.dct(y[i*8:(i+1)*8, j2*8:(j2+1)*8].astype(np.float32))
            bi = (i * npr + jj) % L
            d = b1[COEF] - b2[COEF]
            target_d = (np.floor(d/delta)*delta + delta/2) if msg[bi] == 1 \
                       else np.round(d/delta)*delta
            shift = target_d - d
            b1[COEF] += shift / 2.0
            b2[COEF] -= shift / 2.0
            y[i*8:(i+1)*8, j1*8:(j1+1)*8] = cv2.idct(b1)
            y[i*8:(i+1)*8, j2*8:(j2+1)*8] = cv2.idct(b2)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    out = ycrcb.copy()
    out[:n_h*8, :n_w*8, 0] = np.clip(y, 0, 255)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB).astype(np.uint8)


def decode_pairwise(img, msg_len, delta=15):
    """解码: 块对差值 mod delta 判决 + 每 bit 多数投票."""
    H, W = img.shape[0], img.shape[1]
    n_h, n_w, npr = pair_grid(H, W)
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    votes = [[] for _ in range(msg_len)]
    for i in range(n_h):
        for jj in range(npr):
            j1, j2 = 2*jj, 2*jj+1
            b1 = cv2.dct(y[i*8:(i+1)*8, j1*8:(j1+1)*8].astype(np.float32))
            b2 = cv2.dct(y[i*8:(i+1)*8, j2*8:(j2+1)*8].astype(np.float32))
            bi = (i * npr + jj) % msg_len
            d = b1[COEF] - b2[COEF]
            r = d % delta
            # 判决: 距离 bit=1 族格点 (delta/2) 更近 → 1; 距离 bit=0 族格点 (0) 更近 → 0
            # (原实现用 r > delta/2 严格不等式, 会把恰好落在 delta/2 的 bit=1 全部判错)
            votes[bi].append(1 if abs(r - delta / 2) < abs(r - delta) and abs(r - delta/2) < r
                             else 0)
    out = np.zeros(msg_len, np.uint8)
    for k, v in enumerate(votes):
        if v:
            out[k] = 1 if np.mean(v) > 0.5 else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--deltas', default='8,15,25')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    deltas = [int(x) for x in args.deltas.split(',')]

    meta = json.load(open(f'{HIRES}/meta.json'))
    seen, samples = set(), []
    for s in meta['samples']:
        b = s['id'].rsplit('_', 1)[0]
        if b not in seen:
            seen.add(b)
            samples.append(s)
    samples = samples[:args.n]
    print(f'pairwise (差分) probe: n={len(samples)} deltas={deltas} coef={COEF} 固定噪声')

    from diffusers import StableDiffusionInstructPix2PixPipeline
    import torch
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', torch_dtype=torch.float16,
        safety_checker=None).to(DEVICE)
    print(f'  模型已加载 ({DEVICE})')

    rng = np.random.RandomState(args.seed)
    msg = rng.randint(0, 2, args.msglen).astype(np.uint8)
    out = {'meta': {'n': len(samples), 'deltas': deltas, 'scheme': 'pairwise_diff_qim',
                    'coef': COEF, 'msglen': args.msglen,
                    'ts': datetime.datetime.now().isoformat()},
           'per_image': [], 'summary': {}}

    for delta in deltas:
        rows = []
        for idx, s in enumerate(samples):
            cid = s['id']
            o = crop8(np.array(Image.open(
                f'{HIRES}/images/{cid}_orig.png').convert('RGB')))
            wm = embed_pairwise(o, msg, delta)
            row = {'img': cid, 'delta': delta, 'clean_psnr': psnr(o, wm)}
            row['clean_ba'] = float(bit_accuracy(msg, decode_pairwise(wm, args.msglen, delta)))
            g = torch.Generator(device=DEVICE).manual_seed(args.seed)
            gen = pipe(prompt=s['instruction'], image=Image.fromarray(wm),
                       num_inference_steps=args.steps, generator=g,
                       image_guidance_scale=1.5, guidance_scale=7.5).images[0]
            att = crop8(np.array(gen.convert('RGB')))
            row['att_ba'] = float(bit_accuracy(msg, decode_pairwise(att, args.msglen, delta)))
            rows.append(row)
            out['per_image'].append(row)
            print(f'  [d={delta} {idx+1}/{len(samples)}] {cid} cleanPSNR={row["clean_psnr"]:.1f} '
                  f'cleanBA={row["clean_ba"]*100:.0f}% attBA={row["att_ba"]*100:.1f}%')
        out['summary'][str(delta)] = {
            'n': len(rows),
            'clean_psnr': float(np.mean([r['clean_psnr'] for r in rows])),
            'clean_ba': float(np.mean([r['clean_ba'] for r in rows])),
            'att_ba': float(np.mean([r['att_ba'] for r in rows])),
        }
        s_ = out['summary'][str(delta)]
        print(f'  ==> 差分 delta={delta}: PSNR={s_["clean_psnr"]:.1f}dB '
              f'clean={s_["clean_ba"]*100:.0f}% attack={s_["att_ba"]*100:.1f}%')

    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r1_pairwise_probe_{ts}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n结果: {fp}')
    print('\n=== 对照 (绝对 QIM, 同图同 seed, 见 r1_delta_sweep) ===')
    print('  abs  d=8: 51.9% (50.6dB) | d=15: 60.4% (48.2dB) | d=25: 65.0% (45.0dB)')
    print('  差分 ' + ' | '.join(
        f'd={d}: {out["summary"][str(d)]["att_ba"]*100:.1f}% '
        f'({out["summary"][str(d)]["clean_psnr"]:.1f}dB)' for d in deltas))


if __name__ == '__main__':
    main()
