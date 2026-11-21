#!/usr/bin/env python3
"""r68: TruFor 基线 — 在 KeyFree-Bench 的 400 张测试样本上实测 (被动取证, key-free 可用)."""
import sys, os, json, subprocess, glob, shutil
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
from PIL import Image

TRU = f'{BASE}/data/trufor/TruFor-main/TruFor_train_test'
EXPORT = '/tmp/trufor_bigset'
OUTD = '/tmp/trufor_bigset_out'


def main():
    meta = json.load(open(f'{BASE}/data/bigset/meta.json'))
    items = []
    for s_ in meta['samples'][-400:]:
        sid = s_['id']
        fa = f'{BASE}/data/bigset/images/att_{sid}.png'
        fo = f'{BASE}/data/bigset/images/orig_{sid}.png'
        if os.path.exists(fa) and os.path.exists(fo):
            items.append((sid, fo, fa, s_['bbox']))
    print(f'items: {len(items)}', flush=True)

    os.makedirs(EXPORT, exist_ok=True)
    os.makedirs(OUTD, exist_ok=True)
    for sid, fo, fa, bb in items:
        im = Image.open(fa).convert('RGB').resize((256, 256), Image.LANCZOS)
        im.save(f'{EXPORT}/{sid}.png')

    cmd = [sys.executable, 'test.py', '-in', EXPORT, '-out', OUTD, '-g', '0',
           '-exp', 'trufor_ph3']
    print('running TruFor ...', flush=True)
    r = subprocess.run(cmd, cwd=TRU, capture_output=True, text=True, timeout=3600)
    print('exit', r.returncode, r.stdout[-500:], r.stderr[-500:], flush=True)

    files = sorted(glob.glob(f'{OUTD}/*.npz'))
    print(f'outputs: {len(files)}', flush=True)
    if not files:
        print('no outputs'); return
    rows = []
    for sid, fo, fa, bb in items:
        cand = [f for f in files if sid in os.path.basename(f)]
        if not cand:
            continue
        z = np.load(cand[0])
        keys = list(z.keys())
        arr = None
        for k in ('map', 'anom', 'score_map'):
            if k in z:
                arr = z[k]; break
        if arr is None:
            arr = z[keys[0]]
        arr = np.asarray(arr).squeeze().astype(np.float32)
        if arr.ndim != 2:
            continue
        if arr.max() > 1.5:      # 非 0-1 范围 -> 归一化
            arr = (arr - arr.min()) / max(1e-6, arr.max() - arr.min())
        H, W = 256, 256
        pr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
        gt = np.zeros((H, W), bool)
        x1, y1, x2, y2 = bb
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = True
        inter = float((pr & gt).sum()); union = float((pr | gt).sum())
        rows.append((inter / union if union else 1.0,
                     inter / max(1.0, float(gt.sum())),
                     float(pr.sum()) / max(1.0, float(gt.sum()))))
    if rows:
        a = np.array(rows, dtype=float)
        out = {'meta': dict(n=len(a), method='TruFor (ph3, official weights)',
                            protocol='key-free; same 400 test images and GT'),
               'iou': round(float(a[:, 0].mean()), 4),
               'cov50': round(float((a[:, 1] >= .5).mean()), 4),
               'area_ratio': round(float(a[:, 2].mean()), 3)}
        print('[TruFor]', out, flush=True)
        json.dump(out, open(f'{BASE}/results/r68_trufor_bigset.json', 'w'), indent=1)
        print('-> results/r68_trufor_bigset.json')


if __name__ == '__main__':
    main()
