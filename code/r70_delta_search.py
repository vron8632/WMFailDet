#!/usr/bin/env python3
"""r70: 配置搜索 (W3) — 嵌入强度 δ 对"标签质量→学生精度"的影响.

命题 1 已证: 编辑区饱和值 (≈1/2) 与 δ 无关.
但干净基线随 δ 变化 (δ=8 时 0.266 vs δ=15/25 时 0.250) -> 信噪比 (0.5 vs 0.25) 略有差异.
本实验: 固定 2×2 标签层, 扫 δ ∈ {8,15,25}, 看学生 IoU -> 回答"δ 该不该由搜索决定".
输出: results/r70_delta_search.json
"""
import sys, os, json, time
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs
from r34_distill_matrix import eval_mask, MASTER, PAY, CHK
from r61_L2_scale import train, alarm2
from r64_tier_theory import tiers, eval_tiers

BIG = f'{BASE}/data/bigset'
N_TR, N_VA, N_TE = 2000, 400, 400
DELTAS = [8, 15, 25]


def build(delta, n, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        return z['img'], z['alarm'], z['gt'], z['clean']
    meta = json.load(open(f'{BIG}/meta.json'))
    items = []
    for s_ in meta['samples']:
        if os.path.exists(f'{BIG}/images/orig_{s_["id"]}.png'):
            items.append(s_)
        if len(items) >= n:
            break
    imgs, als, gts, cleans = [], [], [], []
    t0 = time.time()
    for s_ in items:
        sid = s_['id']
        o = r8.crop8(np.array(Image.open(f'{BIG}/images/orig_{sid}.png').convert('RGB')))
        a = r8.crop8(np.array(Image.open(f'{BIG}/images/att_{sid}.png').convert('RGB')))
        key = ks.image_key(MASTER, sid)
        k4, k2 = layer_key(key, 'bs4'), layer_key(key, 'bs2')
        wm1 = ks.embed_keyed(o, key, delta, PAY, CHK, pattern='coord')
        wm2 = embed_bs(wm1, k4, 4, delta, (1, 1))
        wm3 = embed_bs(wm2, k2, 2, delta, (1, 1))
        edited, _ = paste_edit(wm3, a, s_['bbox'])
        f2 = fail_bs(edited, k2, 2, delta, (1, 1))
        # 干净基线 (用于验证命题 2 的 δ 无关性)
        fc = ks.fail_map(wm3, key, delta, CHK, 'coord')
        H, W = edited.shape[:2]
        gt = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = s_['bbox']
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
        imgs.append(edited); als.append(alarm2(f2)); gts.append(gt)
        cleans.append(float(fc.mean()))
        if len(imgs) % 500 == 0:
            print(f'    δ={delta}: {len(imgs)}/{n} ({time.time()-t0:.0f}s)', flush=True)
    imgs, als, gts = np.array(imgs), np.array(als), np.array(gts)
    np.savez_compressed(cache, img=imgs, alarm=als, gt=gts, clean=np.array(cleans))
    return imgs, als, gts, np.array(cleans)


def main():
    out = {'meta': dict(n_train=N_TR, n_test=N_TE, label='2x2 layer', deltas=DELTAS,
                        baseline_delta15=0.9347), 'runs': {}}
    for d in DELTAS:
        cache = f'{BASE}/results/r70_data_d{d}.npz'
        print(f'=== δ={d} ===', flush=True)
        img, al, gt, clean = build(d, N_TR + N_VA + N_TE, cache)
        print(f'  clean failure rate: {clean.mean():.4f} (命题 2: δ-independent)', flush=True)
        x_tr, y_tr = img[:N_TR], al[:N_TR].astype(np.float32)
        x_va, gt_va = img[N_TR:N_TR+N_VA], gt[N_TR:N_TR+N_VA]
        x_te, gt_te = img[-N_TE:], gt[-N_TE:]
        st, v, m = train(x_tr, y_tr, x_va, gt_va, x_te, gt_te, 32, tag=f'delta{d}')
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        m.eval(); pr = []
        with torch.no_grad():
            for i in range(0, len(x_te), 8):
                xb = torch.from_numpy(x_te[i:i+8].transpose(0, 3, 1, 2)).float().to(dev)/255.
                pr.append(torch.sigmoid(m(xb)).cpu().numpy()[:, 0])
        prob = np.concatenate(pr)
        t = eval_tiers(prob, gt_te, tiers(gt_te), f'delta{d}')
        out['runs'][f'delta{d}'] = {'overall': round(st['iou'], 4), 'val': round(v, 4),
                                    'clean_fail': round(float(clean.mean()), 4),
                                    'tiers': t}
        json.dump(out, open(f'{BASE}/results/r70_delta_search.json', 'w'), indent=1)
    print('-> results/r70_delta_search.json')


if __name__ == '__main__':
    main()
