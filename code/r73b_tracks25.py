#!/usr/bin/env python3
"""r73b: T2/T3 用 δ=25 数据重测 (修 r73 中 make_pair 默认 δ=15 的不匹配)."""
import sys, os, json
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import torch
from PIL import Image
import r8_geometric as r8
from r34_distill_matrix import SmallUNet, eval_mask
from r71_tracks import make_pair, load_domain, predict

OUT = f'{BASE}/results/r73b_tracks25.json'


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SmallUNet(base=32).to(dev)
    m.load_state_dict(torch.load(f'{BASE}/results/r73_delta25_student.pth', map_location=dev))
    out = {'meta': dict(delta=25, note='tracks regenerated with delta=25 to match model')}

    def dump():
        json.dump(out, open(OUT, 'w'), indent=1)

    out['T2_cross_domain'] = {}
    for ds in ['butterfly', 'sat', 'doc', 'pokemon']:
        items = load_domain(ds, 60)
        eds, gts = [], []
        for sid, o, a, bb in items:
            e, al, g = make_pair(str(sid), o, a, bb, delta=25)
            eds.append(e); gts.append(g)
        eds, gts = np.array(eds), np.array(gts)
        p = predict(m, eds, dev)
        s = eval_mask((p > .5).astype(np.uint8), gts); s.pop('per_iou')
        out['T2_cross_domain'][ds] = {'n': len(eds), **{k: round(v, 4) for k, v in s.items()}}
        dump(); print(f'  [T2 {ds:10s}] IoU={s["iou"]:.3f}', flush=True)

    root = f'{BASE}/data/xdomain/lama_edits'
    meta = json.load(open(f'{root}/meta.json'))
    eds, gts = [], []
    for s_ in meta['samples']:
        sid = s_['id']
        fo, fa = f'{root}/images/orig_{sid}.png', f'{root}/images/att_{sid}.png'
        if not (os.path.exists(fo) and os.path.exists(fa)):
            continue
        o = r8.crop8(np.array(Image.open(fo).convert('RGB')))
        a2 = np.array(Image.open(fa).convert('RGB'))
        if a2.shape[:2] != o.shape[:2]:
            a2 = np.array(Image.fromarray(a2).resize((o.shape[1], o.shape[0]), Image.LANCZOS))
        e, al, g = make_pair(str(sid), o, r8.crop8(a2), s_['bbox'], delta=25)
        eds.append(e); gts.append(g)
    eds, gts = np.array(eds), np.array(gts)
    split = int(len(eds) * 0.67)
    p = predict(m, eds[split:], dev)
    s = eval_mask((p > .5).astype(np.uint8), gts[split:]); s.pop('per_iou')
    out['T3_lama'] = {'n_test': len(eds) - split, **{k: round(v, 4) for k, v in s.items()}}
    dump(); print(f'  [T3 lama] IoU={s["iou"]:.3f}', flush=True)
    print('-> results/r73b_tracks25.json')


if __name__ == '__main__':
    main()
