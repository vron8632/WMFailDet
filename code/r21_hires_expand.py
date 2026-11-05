#!/usr/bin/env python3
"""r21 真实高清扩样: 16 个唯一 base × 10 个编辑 = 160 张 640x480 样本.

目的: r20 只有 30 张真实高清, 样本太少且 mask 不规则; 本脚本用 IP2P 在 16 个 base 上
      生成 10 种不同 (bbox, instruction) 的局部编辑, 矩形 bbox (与 attackset 同协议),
      用于: ①真实高清下的定位/告警指标 ②与 256 主结果的对照.

输出: data/hires_expanded/{images/{orig,att,mask}_*.png, meta.json}
用法: apjf-python r21_hires_expand.py [--per-base 10]
"""
import sys, os, json, argparse, datetime, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRES = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
         '/results/attackset_hires_sample')
OUT = f'{BASE}/data/hires_expanded'
INSTRUCTIONS = [
    'replace the object with a wooden table',
    'remove the object and fill with the background',
    'change the object to a red sofa',
    'replace it with a potted plant',
    'turn it into a bookshelf',
    'replace it with a painting on the wall',
    'change its color to blue',
    'replace the object with a glass window',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-base', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=20)
    args = ap.parse_args()
    os.makedirs(f'{OUT}/images', exist_ok=True)

    meta = json.load(open(f'{HIRES}/meta.json'))
    samples = meta['samples'] if isinstance(meta, dict) else meta
    # 去重: 每个 cid 取一张 base
    seen, bases = set(), []
    for s in samples:
        cid = s.get('cid', s['id'].split('_')[0])
        if cid in seen:
            continue
        p = f'{HIRES}/images/{s["id"]}_orig.png'
        if os.path.exists(p):
            seen.add(cid)
            bases.append((cid, s['id'], p))
    print(f'base 数: {len(bases)}')

    from diffusers import StableDiffusionInstructPix2PixPipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', torch_dtype=torch.float16, safety_checker=None).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    print('IP2P 已加载')

    rng = random.Random(args.seed)
    out_samples = []
    idx = 0
    for cid, sid, path in bases:
        o = np.array(Image.open(path).convert('RGB'))
        H, W = o.shape[:2]
        for k in range(args.per_base):
            area = rng.uniform(0.08, 0.30)
            ar = rng.uniform(0.8, 1.5)
            h = int(H * np.sqrt(area / ar)); w = int(W * ar)
            h = min(h, int(H * 0.6)); w = min(w, int(W * 0.6))
            x1 = rng.randint(8, W - w - 8); y1 = rng.randint(8, H - h - 8)
            ins = rng.choice(INSTRUCTIONS)
            g = torch.Generator(device='cuda').manual_seed(args.seed * 10000 + idx)
            full = pipe(prompt=ins, image=Image.fromarray(o), num_inference_steps=args.steps,
                        generator=g, image_guidance_scale=1.5, guidance_scale=7.5).images[0]
            full = np.array(full.resize((W, H), Image.BICUBIC))
            att = o.copy()
            att[y1:y1+h, x1:x1+w] = full[y1:y1+h, x1:x1+w]
            mask = np.zeros((H, W), np.uint8)
            mask[y1:y1+h, x1:x1+w] = 255
            new_id = f'{cid}_{k}'
            Image.fromarray(o).save(f'{OUT}/images/orig_{new_id}.png')
            Image.fromarray(att).save(f'{OUT}/images/att_{new_id}.png')
            Image.fromarray(mask).save(f'{OUT}/images/mask_{new_id}.png')
            out_samples.append({'id': new_id, 'cid': cid,
                                'bbox': [x1/W, y1/H, (x1+w)/W, (y1+h)/H],
                                'instruction': ins, 'area': (w*h)/(W*H), 'w': W, 'h': H})
            idx += 1
            if idx % 10 == 0:
                print(f'  [{idx}]', flush=True)
    json.dump({'samples': out_samples, 'source': 'attackset_hires_sample (16 base)',
               'generator': 'timbrooks/instruct-pix2pix', 'ts': datetime.datetime.now().isoformat()},
              open(f'{OUT}/meta.json', 'w'), indent=1)
    print(f'完成 {len(out_samples)} 张 -> {OUT}')


if __name__ == '__main__':
    main()
