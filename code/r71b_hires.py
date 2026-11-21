#!/usr/bin/env python3
"""r71b: T3 hires (IP2P 640x480) 单跑 — 统一 resize 到 512x512 消除尺寸不一."""
import sys, os, json
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import torch
from PIL import Image
import r8_geometric as r8
from r34_distill_matrix import SmallUNet, eval_mask
from r71_tracks import make_pair, predict

ROOT = f'{BASE}/data/hires_expanded'
SZ = 512


def main():
    meta = json.load(open(f'{ROOT}/meta.json'))
    items = [(s_['id'], f'{ROOT}/images/orig_{s_["id"]}.png',
              f'{ROOT}/images/att_{s_["id"]}.png', s_['bbox'])
             for s_ in meta['samples']
             if os.path.exists(f'{ROOT}/images/orig_{s_["id"]}.png')]
    print(f'hires items: {len(items)}', flush=True)
    eds, gts = [], []
    for sid, fo, fa, bb in items:
        try:
            o = r8.crop8(np.array(Image.open(fo).convert('RGB')))
            a = np.array(Image.open(fa).convert('RGB'))
            if a.shape[:2] != o.shape[:2]:
                a = np.array(Image.fromarray(a).resize((o.shape[1], o.shape[0]), Image.LANCZOS))
            e, al, g = make_pair(str(sid), o, r8.crop8(a), bb)
            e = np.array(Image.fromarray(e).resize((SZ, SZ), Image.LANCZOS))
            g = np.array(Image.fromarray(g).resize((SZ, SZ), Image.NEAREST))
            eds.append(e); gts.append(g)
        except Exception as ex:
            print('skip', sid, ex)
    eds = np.array(eds); gts = np.array(gts)
    split = int(len(eds) * 0.67)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SmallUNet(base=32).to(dev)
    m.load_state_dict(torch.load(f'{BASE}/results/r61_L2_student.pth', map_location=dev))
    prob = predict(m, eds[split:], dev)
    st = eval_mask((prob > .5).astype(np.uint8), gts[split:]); st.pop('per_iou')
    print(f'[T3 hires_ip2p] n_test={len(eds)-split} IoU={st["iou"]:.3f}', flush=True)
    out = {'T3_cross_editor': {'hires_ip2p': {'n_test': len(eds)-split, 'res': f'{SZ}x{SZ}',
                                              **{k: round(v, 4) for k, v in st.items()}}}}
    json.dump(out, open(f'{BASE}/results/r71b_hires.json', 'w'), indent=1)
    print('-> results/r71b_hires.json')


if __name__ == '__main__':
    main()
