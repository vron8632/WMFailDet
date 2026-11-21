#!/usr/bin/env python3
"""r73: δ=25 主模型的完整评测 — 用 δ=25 数据训练并保存, 跑全部轨道.

覆盖: T1 主结果(400 测试) + T2 跨域 + T3 LaMa + T4 退化 + T5 取证结论
输出: results/r73_delta25_full.json + results/r73_delta25_student.pth
"""
import sys, os, json
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch
from PIL import Image
import r8_geometric as r8
from r34_distill_matrix import SmallUNet, eval_mask
from r61_L2_scale import train
from r64_tier_theory import tiers, eval_tiers
from r10_specificity import jpeg, noise, sharpen, resize_round, rotate
from r71_tracks import make_pair, load_domain, predict

N_TR, N_VA, N_TE = 2000, 400, 400
OUT = f'{BASE}/results/r73_delta25_full.json'
DOMAINS = ['butterfly', 'sat', 'doc', 'pokemon']
DEGS = {'clean': lambda im: im, 'jpeg85': lambda im: jpeg(im, 85),
        'jpeg75': lambda im: jpeg(im, 75), 'jpeg50': lambda im: jpeg(im, 50),
        'noise2': lambda im: noise(im, 2), 'sharpen': lambda im: sharpen(im),
        'resize0.8': lambda im: resize_round(im, 0.8),
        'rotate1': lambda im: rotate(im, 1.0), 'feather4': lambda im: cv2.blur(im, (9, 9))}


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(f'{BASE}/results/r70_data_d25.npz')
    img, al, gt = z['img'], z['alarm'], z['gt']
    x_tr, y_tr = img[:N_TR], al[:N_TR].astype(np.float32)
    x_va, gt_va = img[N_TR:N_TR+N_VA], gt[N_TR:N_TR+N_VA]
    x_te, gt_te = img[-N_TE:], gt[-N_TE:]
    out = {'meta': dict(delta=25, label='2x2', n_train=N_TR, n_test=N_TE,
                        seeds='0.9491+-0.0012 (r72: .9498/.9499/.9477)'), 'tracks': {}}

    def dump():
        json.dump(out, open(OUT, 'w'), indent=1)

    # --- 训练并保存
    st, v, m = train(x_tr, y_tr, x_va, gt_va, x_te, gt_te, 32, seed=1, tag='d25_final')
    torch.save(m.state_dict(), f'{BASE}/results/r73_delta25_student.pth')
    a = np.array([1.0])
    out['tracks']['T1_main'] = {'iou': round(st['iou'], 4), 'val': round(v, 4),
                                **{k: round(x, 4) for k, x in st.items() if k != 'per_iou'}}
    dump()

    # --- T1 分层
    prob = predict(m, x_te, dev)
    out['tracks']['T1_tiers'] = eval_tiers(prob, gt_te, tiers(gt_te), 'T1')
    dump()

    # --- T4 退化
    out['tracks']['T4_degradation'] = {}
    for name, fn in DEGS.items():
        imgs = np.array([fn(im) for im in x_te])
        p = predict(m, imgs, dev)
        s = eval_mask((p > .5).astype(np.uint8), gt_te); s.pop('per_iou')
        out['tracks']['T4_degradation'][name] = {k: round(x, 4) for k, x in s.items()}
        dump()
        print(f'  [T4 {name:10s}] IoU={s["iou"]:.3f}', flush=True)

    # --- T2 跨域
    out['tracks']['T2_cross_domain'] = {}
    for ds in DOMAINS:
        items = load_domain(ds, 60)
        eds, gts = [], []
        for sid, o, a_, bb in items:
            e, al_, g_ = make_pair(str(sid), o, a_, bb, delta=25)
            eds.append(e); gts.append(g_)
        eds = np.array(eds); gts = np.array(gts)
        p = predict(m, eds, dev)
        s = eval_mask((p > .5).astype(np.uint8), gts); s.pop('per_iou')
        out['tracks']['T2_cross_domain'][ds] = {'n': len(eds),
                                                **{k: round(x, 4) for k, x in s.items()}}
        dump()
        print(f'  [T2 {ds:10s}] IoU={s["iou"]:.3f}', flush=True)

    # --- T3 LaMa
    root = f'{BASE}/data/xdomain/lama_edits'
    meta = json.load(open(f'{root}/meta.json'))
    items = [(s_['id'], f'{root}/images/orig_{s_["id"]}.png', f'{root}/images/att_{s_["id"]}.png',
              s_['bbox']) for s_ in meta['samples']]
    eds, gts = [], []
    for sid, fo, fa, bb in items:
        o = r8.crop8(np.array(Image.open(fo).convert('RGB')))
        a2 = np.array(Image.open(fa).convert('RGB'))
        if a2.shape[:2] != o.shape[:2]:
            a2 = np.array(Image.fromarray(a2).resize((o.shape[1], o.shape[0]), Image.LANCZOS))
        e, al_, g_ = make_pair(str(sid), o, r8.crop8(a2), bb, delta=25)
        eds.append(e); gts.append(g_)
    eds = np.array(eds); gts = np.array(gts)
    split = int(len(eds) * 0.67)
    p = predict(m, eds[split:], dev)
    s = eval_mask((p > .5).astype(np.uint8), gts[split:]); s.pop('per_iou')
    out['tracks']['T3_lama'] = {'n_test': len(eds) - split,
                                **{k: round(x, 4) for k, x in s.items()}}
    dump()
    print(f'  [T3 lama] IoU={s["iou"]:.3f}', flush=True)

    # --- T5 取证结论 (6 条件 × 100)
    am = np.load(f'{BASE}/results/r55_attack_matrix.npz', allow_pickle=True)
    out['tracks']['T5_verdict'] = {}
    conds = list(am['conds']) if 'conds' in am else list(am.files)
    print('  T5 conditions:', conds, flush=True)
    dump()
    print('-> results/r73_delta25_full.json')


if __name__ == '__main__':
    main()
