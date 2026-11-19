#!/usr/bin/env python3
"""r54: 大数据集 scaling — 用 bigset (4000 张新 COCO 编辑) 训练学生.

回答: 数据从 800 -> 3200 时, 学生 IoU 是否继续上升/接近 GT 监督上界 0.903 / WAM 0.947.
配置: base32, 80ep, 三分: train N / val 400 / test 400 (测试集固定)
输出: results/r54_bigset_scaling.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs
from r34_distill_matrix import SmallUNet, eval_mask, alarm_from_fail, DELTA, PAY, CHK, MASTER

BIG = f'{BASE}/data/bigset'


def build_bigset(cache):
    """bigset -> (img, alarm4, gt)."""
    if os.path.exists(cache):
        z = np.load(cache)
        return z['img'], z['alarm'], z['gt']
    meta = json.load(open(f'{BIG}/meta.json'))
    imgs, als, gts = [], [], []
    t0 = time.time()
    for s_ in meta['samples']:
        sid = s_['id']
        fp_o, fp_a = f'{BIG}/images/orig_{sid}.png', f'{BIG}/images/att_{sid}.png'
        if not (os.path.exists(fp_o) and os.path.exists(fp_a)):
            continue
        o = r8.crop8(np.array(Image.open(fp_o).convert('RGB')))
        a = r8.crop8(np.array(Image.open(fp_a).convert('RGB')))
        key = ks.image_key(MASTER, sid); k4 = layer_key(key, 'bs4')
        wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        wm = embed_bs(wm, k4, 4, DELTA, (1, 1))
        edited, _ = paste_edit(wm, a, s_['bbox'])
        f4 = fail_bs(edited, k4, 4, DELTA, (1, 1))
        al = alarm_from_fail(f4, 4, 5, 32)
        H, W = edited.shape[:2]
        gt = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = s_['bbox']
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
        imgs.append(edited); als.append(al); gts.append(gt)
        if len(imgs) % 500 == 0:
            print(f'  built {len(imgs)} ({time.time()-t0:.0f}s)', flush=True)
    imgs = np.array(imgs); als = np.array(als); gts = np.array(gts)
    np.savez_compressed(cache, img=imgs, alarm=als, gt=gts)
    print(f'bigset: {len(imgs)} ({time.time()-t0:.0f}s)', flush=True)
    return imgs, als, gts


def train(x_tr, y_tr, x_val, gt_val, x_te, gt_te, epochs=80, seed=0, tag=''):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float()/255.,
        torch.from_numpy(y_tr[:, None]).float()), batch_size=16, shuffle=True, drop_last=True)
    m = SmallUNet(base=32).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=dev))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    def lossf(lo_, tg):
        b = bce(lo_, tg)
        p = torch.sigmoid(lo_).flatten(1); t = tg.flatten(1)
        di = (p*t).sum(1); ds_ = p.sum(1)+t.sum(1)
        return b + (1 - (2*di+1)/(ds_+1)).mean()
    def infer(x):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(x), 16):
                xb = torch.from_numpy(x[i:i+16].transpose(0, 3, 1, 2)).float().to(dev)/255.
                out.append(torch.sigmoid(m(xb)).cpu().numpy())
        return np.concatenate(out)[:, 0]
    best, best_sd = 0., None
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        sched.step()
        s = eval_mask((infer(x_val) > .5).astype(np.uint8), gt_val)
        if s['iou'] > best:
            best = s['iou']; best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if (ep+1) % 20 == 0:
            print(f'  [{tag}] ep{ep+1} val={s["iou"]:.3f}', flush=True)
    m.load_state_dict(best_sd)
    st = eval_mask((infer(x_te) > .5).astype(np.uint8), gt_te); st.pop('per_iou')
    print(f'  [{tag}] n={len(x_tr)} val={best:.3f} TEST={st["iou"]:.3f}', flush=True)
    return st, best


def main():
    cache = f'{BASE}/results/r54_bigset_data.npz'
    img, al, gt = build_bigset(cache)
    # 固定测试/验证在末尾, 训练集按需扩大
    n_te, n_val = 400, 400
    x_te, gt_te = img[-n_te:], gt[-n_te:]
    x_val, gt_val = img[-n_te-n_val:-n_te], gt[-n_te-n_val:-n_te]
    pool = len(img) - n_te - n_val
    out = {'meta': dict(pool=pool, n_val=n_val, n_test=n_te, epochs=80,
                        student='SmallUNet(base=32)', reference={'orig800': 0.775,
                        'gt_upper': 0.903, 'WAM': 0.947, 'EditGuard': 0.913}),
           'scaling': {}}
    for n in (800, 1600, 2400):
        if n > pool:
            break
        st, val = train(img[:n], al[:n].astype(np.float32), x_val, gt_val,
                        x_te, gt_te, tag=f'n{n}')
        out['scaling'][f'n{n}'] = {'test': {k: round(v, 4) for k, v in st.items()},
                                   'val': round(val, 4)}
    if pool > 2400:
        st, val = train(img[:pool], al[:pool].astype(np.float32), x_val, gt_val,
                        x_te, gt_te, tag=f'n{pool}')
        out['scaling'][f'n{pool}'] = {'test': {k: round(v, 4) for k, v in st.items()},
                                      'val': round(val, 4)}
    json.dump(out, open(f'{BASE}/results/r54_bigset_scaling.json', 'w'), indent=1)
    print('-> results/r54_bigset_scaling.json')


if __name__ == '__main__':
    main()
