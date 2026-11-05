#!/usr/bin/env python3
"""r16 跨域数据生成: 用 SD inpainting 对非 COCO 图像生成局部编辑.

目的: 检验"training-free 的 QIM"与"训练式 WAM"在**训练分布外**的稳定性.
WAM 训练于 SA-1B(自然场景), 本实验用 Smithsonian butterflies (特写/单一主体/不同数据集).

输出: data/xdomain/{images/orig_XXXX.png, att_XXXX.png, meta.json} (统一 256x256, 与 attackset 协议一致)
用法: apjf-python r16_xdomain_gen.py [--n 60]
"""
import sys, os, json, argparse, datetime, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f'{BASE}/data/xdomain'
OUT = f'{BASE}/data/xdomain'
PROMPTS = ['a butterfly wing pattern', 'a green leaf', 'a flower petal', 'a stone surface',
           'a patch of grass', 'a wooden texture']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(f'{OUT}/images', exist_ok=True)

    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting', torch_dtype=torch.float16,
        safety_checker=None).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    print('SD inpainting 已加载')

    files = sorted([f for f in os.listdir(SRC) if f.endswith('.png')])[:args.n]
    rng = random.Random(args.seed)
    samples = []
    for idx, f in enumerate(files):
        sid = f[:-4]
        img = Image.open(f'{SRC}/{f}').convert('RGB').resize((512, 512), Image.BICUBIC)
        # 随机 bbox (面积 0.08-0.30, 避开边界)
        area = rng.uniform(0.08, 0.30)
        ar = rng.uniform(0.7, 1.4)
        h = int(512 * np.sqrt(area / ar)); w = int(h * ar)
        h = min(h, 400); w = min(w, 400)
        x1 = rng.randint(20, 512 - w - 20); y1 = rng.randint(20, 512 - h - 20)
        box = (x1, y1, x1 + w, y1 + h)
        mask = np.zeros((512, 512), np.uint8)
        mask[box[1]:box[3], box[0]:box[2]] = 255
        prompt = rng.choice(PROMPTS)
        out = pipe(prompt=prompt, image=img, mask_image=Image.fromarray(mask),
                   height=512, width=512, num_inference_steps=30,
                   generator=torch.Generator(device='cuda').manual_seed(args.seed * 1000 + idx)).images[0]
        edit = np.array(img).copy()
        edit[box[1]:box[3], box[0]:box[2]] = np.array(out)[box[1]:box[3], box[0]:box[2]]
        # 统一 256 (与 attackset 一致)
        o256 = np.array(img.resize((256, 256), Image.BICUBIC))
        a256 = np.array(Image.fromarray(edit).resize((256, 256), Image.BICUBIC))
        Image.fromarray(o256).save(f'{OUT}/images/orig_{sid}.png')
        Image.fromarray(a256).save(f'{OUT}/images/att_{sid}.png')
        samples.append({'id': sid, 'bbox': [box[0]/512, box[1]/512, box[2]/512, box[3]/512],
                        'prompt': prompt, 'area': (w*h)/(512*512)})
        if (idx + 1) % 10 == 0:
            print(f'  [{idx+1}/{len(files)}]', flush=True)
    json.dump({'samples': samples, 'source': 'huggan/smithsonian_butterflies_subset',
               'generator': 'runwayml/stable-diffusion-inpainting',
               'ts': datetime.datetime.now().isoformat()},
              open(f'{OUT}/meta.json', 'w'), indent=1)
    print(f'完成 {len(samples)} 张 -> {OUT}')


if __name__ == '__main__':
    main()
