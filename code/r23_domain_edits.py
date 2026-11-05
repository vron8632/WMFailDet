#!/usr/bin/env python3
"""r23 通用域编辑生成: 对任意域的图用 SD inpainting 生成局部编辑 (与 attackset 同协议).

输入: data/xdomain/<domain>/*.png (512x512)
输出: data/xdomain/<domain>_edits/{images/{orig,att}_*.png, meta.json} (256x256)
用法: apjf-python r23_domain_edits.py --domain sat --n 200
"""
import sys, os, json, argparse, datetime, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XD = f'{BASE}/data/xdomain'
PROMPTS = {
    'sat': ['a green field', 'a river', 'a building', 'a road', 'a forest'],
    'doc': ['a blank white area', 'a table', 'a signature', 'a stamp'],
    'art': ['a flower', 'a tree', 'a mountain', 'a face'],
    'butterfly': ['a flower petal', 'a green leaf', 'a stone surface'],
    'pokemon': ['a flower', 'a tree', 'a rock', 'a cloud'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=25)
    args = ap.parse_args()
    src = f'{XD}/{args.domain}'
    out = f'{XD}/{args.domain}_edits'
    os.makedirs(f'{out}/images', exist_ok=True)

    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting', torch_dtype=torch.float16,
        safety_checker=None).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    print(f'SD inpainting 已加载 (domain={args.domain})')

    files = sorted([f for f in os.listdir(src) if f.endswith('.png')])[:args.n]
    rng = random.Random(args.seed)
    prompts = PROMPTS.get(args.domain, PROMPTS['art'])
    samples = []
    for idx, f in enumerate(files):
        sid = f[:-4]
        img = Image.open(f'{src}/{f}').convert('RGB').resize((512, 512), Image.BICUBIC)
        area = rng.uniform(0.08, 0.30)
        ar = rng.uniform(0.7, 1.4)
        h = int(512 * np.sqrt(area / ar)); w = int(h * ar)
        h = min(h, 400); w = min(w, 400)
        x1 = rng.randint(16, 512 - w - 16); y1 = rng.randint(16, 512 - h - 16)
        mask = np.zeros((512, 512), np.uint8)
        mask[y1:y1+h, x1:x1+w] = 255
        p = rng.choice(prompts)
        g = torch.Generator(device='cuda').manual_seed(args.seed * 10000 + idx)
        out_img = pipe(prompt=p, image=img, mask_image=Image.fromarray(mask),
                       height=512, width=512, num_inference_steps=args.steps,
                       generator=g).images[0]
        edit = np.array(img).copy()
        edit[y1:y1+h, x1:x1+w] = np.array(out_img)[y1:y1+h, x1:x1+w]
        o256 = np.array(img.resize((256, 256), Image.BICUBIC))
        a256 = np.array(Image.fromarray(edit).resize((256, 256), Image.BICUBIC))
        Image.fromarray(o256).save(f'{out}/images/orig_{sid}.png')
        Image.fromarray(a256).save(f'{out}/images/att_{sid}.png')
        samples.append({'id': sid, 'bbox': [x1/512, y1/512, (x1+w)/512, (y1+h)/512],
                        'prompt': p, 'area': (w*h)/(512*512)})
        if (idx + 1) % 20 == 0:
            print(f'  [{idx+1}/{len(files)}]', flush=True)
    json.dump({'samples': samples, 'domain': args.domain,
               'source': f'data/xdomain/{args.domain}',
               'generator': 'runwayml/stable-diffusion-inpainting',
               'ts': datetime.datetime.now().isoformat()},
              open(f'{out}/meta.json', 'w'), indent=1)
    print(f'完成 {len(samples)} 张 -> {out}')


if __name__ == '__main__':
    main()
