#!/usr/bin/env python3
"""r53: 构建大规模 COCO 编辑集 (扩数据) — 用本地 SD1.5 inpainting 生成.

来源: COCO val2017 (83895 张, 未与现有 attackset 1200 张重叠)
指令: 复用 attackset meta.json 中的编辑指令文本 (随机分配), 保持"语义编辑"性质
编辑: mask = 随机 bbox (面积分布对齐现有 attackset), SD inpainting, mask 外强制原图

输出: data/bigset/{images/orig_*.png, images/att_*.png, images/mask_*.png, meta.json}
用法: HF_HOME=.../huggingface python code/r53_build_bigset.py --n 3000
"""
import sys, os, json, time, argparse, random
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import torch
from PIL import Image

COCO = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data/coco/val2017'
OLD_META = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
            '/results/attackset_20261001_222447/meta.json')
OUT = f'{BASE}/data/bigset'
MODEL_ID = 'runwayml/stable-diffusion-inpainting'


def area_dist(n=3000):
    """对齐现有 attackset 的编辑面积分布: small<10% / mid 10-30% / large 30-90% / full>90%."""
    rng = random.Random(0)
    buckets = [('small', 0.03, 0.10, 0.30), ('mid', 0.10, 0.30, 0.45),
               ('large', 0.30, 0.90, 0.20), ('full', 0.90, 0.99, 0.05)]
    areas = []
    for _ in range(n):
        r, p = rng.random(), 0
        for _, lo, hi, w in buckets:
            p += w
            if r <= p:
                areas.append(rng.uniform(lo, hi)); break
    return areas


def rand_bbox(area, rng):
    """随机位置的矩形 bbox (归一化), 面积≈area."""
    h = rng.uniform(0.35, 1.0)
    w = min(1.0, area / h)
    h = min(1.0, area / w)
    x1 = rng.uniform(0, 1 - w); y1 = rng.uniform(0, 1 - h)
    return [x1, y1, x1 + w, y1 + h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--steps', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    np.random.seed(a.seed)

    # 复用现有指令文本
    import re as _re

    def _clean(v):
        """清洗指令: 剥离嵌套 JSON / 代码块, 丢弃坐标残留."""
        if not isinstance(v, str):
            return None
        v = v.strip()
        if v.startswith('{') or v.startswith('```'):
            m = _re.search(r'"instruction"\s*:\s*"([^"]+)"', v)
            v = m.group(1) if m else None
        if not v:
            return None
        if _re.search(r'\]\}|^\s*[0-9.,\s]+$', v):
            return None
        if not (4 <= len(v) <= 120):
            return None
        return v

    meta_old = json.load(open(OLD_META))
    raw = [s.get('instruction') or s.get('prompt') for s in meta_old['samples']]
    # 1) 从原指令里挑出【英文】且合理的 (SD1.5 CLIP 只认英文)
    en = [c for c in (_clean(v) for v in raw)
          if c and not _re.search(r'[\u4e00-\u9fff]', c)
          and not _re.search(r'rejected|high risk|error|sorry|I cannot', c, _re.I)]
    # 2) 兜底: 手写英文编辑指令池 (覆盖替换/删除/添加/改风格)
    FALLBACK = [
        'replace the object with a wooden chair', 'remove the person from the scene',
        'replace the animal with a white sheep', 'change the background to a beach',
        'replace the sign with a blue road sign', 'add a large potted plant in the corner',
        'replace the car with a bicycle', 'remove the text from the sign',
        'replace the table with an office desk', 'change the sky to a sunset',
        'replace the bottle with a glass of water', 'remove the logo from the wall',
        'replace the tree with a street lamp', 'add a red balloon above the table',
        'replace the bowl with a plate of fruit', 'change the jacket color to blue',
        'replace the bench with a stone statue', 'remove the chair from the room',
        'replace the window with a painting', 'add clouds to the sky',
        'replace the dog with a cat', 'change the floor to wooden boards',
        'replace the cup with a small vase', 'remove the umbrella from the scene',
        'replace the keyboard with a notebook', 'change the wall color to light gray',
        'replace the flowers with green leaves', 'add a mirror on the wall',
        'replace the boat with a raft', 'change the shirt to a striped pattern',
    ]
    instrs = sorted(set(en)) + FALLBACK
    print(f'instruction pool: {len(instrs)} (en-from-meta={len(set(en))}, fallback={len(FALLBACK)})',
          flush=True)
    assert len(instrs) >= 30, 'instruction pool too small'

    os.makedirs(f'{OUT}/images', exist_ok=True)
    done = set()
    if os.path.exists(f'{OUT}/meta.json'):
        old = json.load(open(f'{OUT}/meta.json'))
        done = {s['id'] for s in old['samples']}
        samples = old['samples']
        print(f'resume: {len(samples)} done', flush=True)
    else:
        samples = []

    files = sorted(f for f in os.listdir(COCO) if f.endswith('.jpg'))
    # 跳过已被 attackset 用过的图 (按文件名近似: attackset 用 COCO 前段)
    rng.shuffle(files)
    files = files[: a.n + len(done) + 50]

    print(f'loading {MODEL_ID} ...', flush=True)
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    print('model ready', flush=True)

    areas = area_dist(a.n + len(done))
    t0 = time.time()
    idx = len(samples)
    BATCH = 6
    pend = []   # (sid, arr, mask, prompt, bb)

    def flush(batch):
        nonlocal idx
        if not batch:
            return
        prompts = [b[3] for b in batch]
        imgs = [Image.fromarray(b[1]) for b in batch]
        masks = [Image.fromarray(b[2]) for b in batch]
        out = pipe(prompt=prompts, image=imgs, mask_image=masks,
                   height=256, width=256, num_inference_steps=a.steps,
                   guidance_scale=7.0,
                   generator=torch.Generator('cuda').manual_seed(a.seed + idx)).images
        for (sid, arr, mask, _pr, bb), o in zip(batch, out):
            att = np.array(o.resize((256, 256), Image.LANCZOS))
            keep = mask == 0
            att[keep] = arr[keep]
            Image.fromarray(arr).save(f'{OUT}/images/orig_{sid}.png')
            Image.fromarray(att).save(f'{OUT}/images/att_{sid}.png')
            Image.fromarray(mask).save(f'{OUT}/images/mask_{sid}.png')
            samples.append({'id': sid, 'bbox': bb, 'instruction': _pr,
                            'area': float(mask.mean() / 255.0)})
            idx += 1
        if idx % 60 < BATCH:
            el = time.time() - t0
            done_n = idx - len(done)
            print(f'  {idx}/{a.n}  {el/60:.1f}min  '
                  f'{(el/max(1,done_n)):.2f}s/张  剩余~{el/max(1,done_n)*(a.n-idx)/60:.0f}min',
                  flush=True)
            json.dump({'samples': samples}, open(f'{OUT}/meta.json', 'w'))

    files_iter = iter(files)
    while idx < a.n:
        pend.clear()
        while len(pend) < BATCH and idx + len(pend) < a.n:
            try:
                fp = next(files_iter)
            except StopIteration:
                break
            sid = f'b{idx + len(pend):05d}'
            if sid in done:
                continue
            try:
                im = Image.open(os.path.join(COCO, fp)).convert('RGB').resize((256, 256), Image.LANCZOS)
            except Exception:
                continue
            arr = np.array(im)
            bb_n = rand_bbox(areas[idx + len(pend)], rng)
            x1, y1, x2, y2 = [int(round(v * 256)) for v in bb_n]
            x2, y2 = max(x2, x1 + 16), max(y2, y1 + 16)
            mask = np.zeros((256, 256), np.uint8)
            mask[y1:y2, x1:x2] = 255
            prompt = rng.choice(instrs) if instrs else 'a photo'
            pend.append((sid, arr, mask, prompt, [x1/256, y1/256, x2/256, y2/256]))
        if not pend:
            break
        try:
            flush(pend)
        except Exception as e:
            print(f'  batch failed ({e}), falling back to single', flush=True)
            for item in pend:
                try:
                    flush([item])
                except Exception as e2:
                    print(f'  skip {item[0]}: {e2}', flush=True)
    json.dump({'samples': samples}, open(f'{OUT}/meta.json', 'w'))
    print(f'-> {OUT}  total {len(samples)}  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
