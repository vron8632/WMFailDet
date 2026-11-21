#!/usr/bin/env python3
"""r71: KeyFree-Bench 的 T2/T3/T4 轨道 — 全部改用 2×2 标签重测.

T2 跨域: COCO(L2) 训练 -> EUROSAT/FUNSD/POKEMON/BUTTERFLY 测试
T3 跨编辑器: 加入 LaMa / IP2P 伪标签联合训练 -> 各自测试
T4 退化: 主模型在 9 种退化下测试 (clean/jpeg85/75/50/noise/sharpen/resize/rotate/feather)
输出: results/r71_tracks.json
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
from r34_distill_matrix import SmallUNet, eval_mask, DELTA, PAY, CHK, MASTER
from r61_L2_scale import alarm2
from r10_specificity import jpeg, noise, sharpen, resize_round, rotate
from r64_tier_theory import tiers, eval_tiers

BIG = f'{BASE}/data/bigset'
DOMAINS = ['butterfly', 'sat', 'doc', 'pokemon']


def make_pair(sid, o, a, bbox, delta=DELTA):
    """三层嵌入 + 2×2 标签."""
    key = ks.image_key(MASTER, sid)
    k4, k2 = layer_key(key, 'bs4'), layer_key(key, 'bs2')
    wm = ks.embed_keyed(o, key, delta, PAY, CHK, pattern='coord')
    wm = embed_bs(wm, k4, 4, delta, (1, 1))
    wm = embed_bs(wm, k2, 2, delta, (1, 1))
    edited, _ = paste_edit(wm, a, bbox)
    f2 = fail_bs(edited, k2, 2, delta, (1, 1))
    H, W = edited.shape[:2]
    gt = np.zeros((H, W), np.uint8)
    x1, y1, x2, y2 = bbox
    gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
    return edited, alarm2(f2), gt


def load_domain(ds, n):
    root = f'{BASE}/data/xdomain/{ds}_edits'
    meta = json.load(open(f'{root}/meta.json'))
    out = []
    for s_ in meta['samples']:
        sid = s_['id']
        fo, fa = f'{root}/images/orig_{sid}.png', f'{root}/images/att_{sid}.png'
        if not (os.path.exists(fo) and os.path.exists(fa)):
            continue
        o = r8.crop8(np.array(Image.open(fo).convert('RGB')))
        a = np.array(Image.open(fa).convert('RGB'))
        if a.shape[:2] != o.shape[:2]:
            a = np.array(Image.fromarray(a).resize((o.shape[1], o.shape[0]), Image.LANCZOS))
        out.append((str(sid), o, r8.crop8(a), s_['bbox']))
        if len(out) >= n:
            break
    return out


def predict(m, imgs, dev='cuda', bs=8):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            xb = torch.from_numpy(imgs[i:i+bs].transpose(0, 3, 1, 2)).float().to(dev) / 255.
            out.append(torch.sigmoid(m(xb)).cpu().numpy()[:, 0])
    return np.concatenate(out)


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SmallUNet(base=32).to(dev)
    m.load_state_dict(torch.load(f'{BASE}/results/r61_L2_student.pth', map_location=dev))
    out = {'meta': dict(label='2x2 layer', model='r61 L2 base32',
                        tracks=['T2 cross-domain', 'T3 cross-editor', 'T4 degradation'])}

    # ---------- T4: 退化 (主轨道测试集上)
    z = np.load(f'{BASE}/results/r61_L2_data.npz')
    x_te, gt_te = z['img'][-400:], z['gt'][-400:]
    DEGS = {
        'clean': lambda im: im, 'jpeg85': lambda im: jpeg(im, 85),
        'jpeg75': lambda im: jpeg(im, 75), 'jpeg50': lambda im: jpeg(im, 50),
        'noise2': lambda im: noise(im, 2), 'sharpen': lambda im: sharpen(im),
        'resize0.8': lambda im: resize_round(im, 0.8),
        'rotate1': lambda im: rotate(im, 1.0),
        'feather4': lambda im: cv2.blur(im, (9, 9)),
    }
    OUT = f'{BASE}/results/r71_tracks.json'
    def dump():
        json.dump(out, open(OUT, 'w'), indent=1)
    out['T4_degradation'] = {}
    for name, fn in DEGS.items():
        imgs = np.array([fn(im) for im in x_te])
        prob = predict(m, imgs, dev)
        st = eval_mask((prob > .5).astype(np.uint8), gt_te); st.pop('per_iou')
        out['T4_degradation'][name] = {k: round(v, 4) for k, v in st.items()}
        dump()
        print(f'  [T4 {name:10s}] IoU={st["iou"]:.3f} cov={st["cov50"]:.1%}', flush=True)

    # ---------- T2: 跨域 (COCO 主模型直接测)
    out['T2_cross_domain'] = {}
    for ds in DOMAINS:
        items = load_domain(ds, 60)
        eds, als, gts = [], [], []
        for sid, o, a, bb in items:
            e, al, g = make_pair(sid, o, a, bb)
            eds.append(e); als.append(al); gts.append(g)
        eds = np.array(eds); gts = np.array(gts)
        prob = predict(m, eds, dev)
        st = eval_mask((prob > .5).astype(np.uint8), gts); st.pop('per_iou')
        out['T2_cross_domain'][ds] = {'n': len(eds), **{k: round(v, 4) for k, v in st.items()}}
        dump()
        print(f'  [T2 {ds:10s}] n={len(eds)} IoU={st["iou"]:.3f}', flush=True)

    # ---------- T3: 跨编辑器 (LaMa / IP2P, 联合微调)
    out['T3_cross_editor'] = {}
    for name, root in [('lama', f'{BASE}/data/xdomain/lama_edits')]:
        meta = json.load(open(f'{root}/meta.json'))
        items = []
        for s_ in meta['samples']:
            sid = s_['id']
            fo, fa = f'{root}/images/orig_{sid}.png', f'{root}/images/att_{sid}.png'
            if os.path.exists(fo) and os.path.exists(fa):
                items.append((sid, fo, fa, s_['bbox']))
        n0 = len(items)
        split = int(n0 * 0.67)
        eds, gts = [], []
        for sid, fo, fa, bb in items:
            o = r8.crop8(np.array(Image.open(fo).convert('RGB')))
            a = np.array(Image.open(fa).convert('RGB'))
            if a.shape[:2] != o.shape[:2]:
                a = np.array(Image.fromarray(a).resize((o.shape[1], o.shape[0]), Image.LANCZOS))
            e, al, g = make_pair(str(sid), o, r8.crop8(a), bb)
            eds.append(e); gts.append(g)
        eds = np.array(eds); gts = np.array(gts)
        prob = predict(m, eds[split:], dev)
        st = eval_mask((prob > .5).astype(np.uint8), gts[split:]); st.pop('per_iou')
        out['T3_cross_editor'][name] = {'n_test': len(eds) - split,
                                        **{k: round(v, 4) for k, v in st.items()}}
        dump()
        print(f'  [T3 {name:12s}] n_test={len(eds)-split} IoU={st["iou"]:.3f}', flush=True)

    dump()
    print('-> results/r71_tracks.json')


if __name__ == '__main__':
    main()
