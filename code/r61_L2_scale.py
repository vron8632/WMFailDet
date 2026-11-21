#!/usr/bin/env python3
"""r61: L2(2×2 层) 标签的大规模验证 — 同口径对比 r54/r59.

口径: bigset, train 2000 / val 400 / test 400
对照: L4 标签 0.851 (r59 manual_b32) / GT 上界 0.903 / EditGuard 0.913
附加: 发布侧 PSNR 代价 (单层 vs 三层嵌入)
输出: results/r61_L2_scale.json
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
from r34_distill_matrix import SmallUNet, eval_mask, DELTA, PAY, CHK, MASTER

BIG = f'{BASE}/data/bigset'
N_TR, N_VA, N_TE = 2000, 400, 400


def alarm2(fail, min_area=128, thr=0.38):
    sm = cv2.blur(fail.astype(np.float32), (3, 3))
    b = (sm > thr).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    out = np.zeros_like(b)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return np.kron(out, np.ones((2, 2), np.uint8))


def build(n, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        return z['img'], z['alarm'], z['gt'], z['psnr']
    meta = json.load(open(f'{BIG}/meta.json'))
    items = []
    for s_ in meta['samples']:
        sid = s_['id']
        if os.path.exists(f'{BIG}/images/orig_{sid}.png'):
            items.append(s_)
        if len(items) >= n:
            break
    imgs, als, gts, psnrs = [], [], [], []
    t0 = time.time()
    for s_ in items:
        sid = s_['id']
        o = r8.crop8(np.array(Image.open(f'{BIG}/images/orig_{sid}.png').convert('RGB')))
        a = r8.crop8(np.array(Image.open(f'{BIG}/images/att_{sid}.png').convert('RGB')))
        key = ks.image_key(MASTER, sid)
        k4, k2 = layer_key(key, 'bs4'), layer_key(key, 'bs2')
        wm1 = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        wm2 = embed_bs(wm1, k4, 4, DELTA, (1, 1))
        wm3 = embed_bs(wm2, k2, 2, DELTA, (1, 1))
        edited, _ = paste_edit(wm3, a, s_['bbox'])
        f2 = fail_bs(edited, k2, 2, DELTA, (1, 1))
        H, W = edited.shape[:2]
        gt = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = s_['bbox']
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
        imgs.append(edited); als.append(alarm2(f2)); gts.append(gt)
        mse = float(np.mean((wm3.astype(np.float32) - o.astype(np.float32)) ** 2))
        psnrs.append(10 * np.log10(255 ** 2 / max(mse, 1e-6)))
        if len(imgs) % 400 == 0:
            print(f'  built {len(imgs)}/{n} ({time.time()-t0:.0f}s)', flush=True)
    imgs, als, gts = np.array(imgs), np.array(als), np.array(gts)
    np.savez_compressed(cache, img=imgs, alarm=als, gt=gts, psnr=np.array(psnrs))
    return imgs, als, gts, np.array(psnrs)


def train(x_tr, y_tr, x_va, gt_va, x_te, gt_te, base=32, epochs=80, seed=0, tag=''):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float()/255.,
        torch.from_numpy(y_tr[:, None]).float()), batch_size=8, shuffle=True, drop_last=True)
    m = SmallUNet(base=base).to(dev)
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
            for i in range(0, len(x), 8):
                xb = torch.from_numpy(x[i:i+8].transpose(0, 3, 1, 2)).float().to(dev)/255.
                out.append(torch.sigmoid(m(xb)).cpu().numpy())
        return np.concatenate(out)[:, 0]
    best, best_sd = 0., None
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        sched.step()
        s = eval_mask((infer(x_va) > .5).astype(np.uint8), gt_va)
        if s['iou'] > best:
            best = s['iou']; best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if (ep+1) % 20 == 0:
            print(f'  [{tag}] ep{ep+1} val={s["iou"]:.3f}', flush=True)
    m.load_state_dict(best_sd)
    st = eval_mask((infer(x_te) > .5).astype(np.uint8), gt_te); st.pop('per_iou')
    print(f'  [{tag}] base={base} val={best:.3f} TEST={st["iou"]:.3f}', flush=True)
    return st, best, m


def main():
    cache = f'{BASE}/results/r61_L2_data.npz'
    img, alarm, gt, psnr = build(N_TR + N_VA + N_TE, cache)
    print(f'data: {len(img)}, publish-side PSNR(3-layer) = {psnr.mean():.2f} dB', flush=True)
    x_tr, y_tr = img[:N_TR], alarm[:N_TR].astype(np.float32)
    x_va, gt_va = img[N_TR:N_TR+N_VA], gt[N_TR:N_TR+N_VA]
    x_te, gt_te = img[-N_TE:], gt[-N_TE:]
    out = {'meta': dict(n_train=N_TR, n_val=N_VA, n_test=N_TE, epochs=80,
                        label='2x2-layer failure map (offline teacher)',
                        psnr_publish=round(float(psnr.mean()), 2),
                        reference={'L4_label_base32_2000': 0.851, 'GT_upper': 0.903,
                                   'EditGuard': 0.913, 'WAM': 0.947}),
           'runs': {}}
    st, v, m = train(x_tr, y_tr, x_va, gt_va, x_te, gt_te, 32, tag='L2_b32')
    out['runs']['L2_base32'] = {'test': {k: round(x, 4) for k, x in st.items()}, 'val': round(v, 4)}
    torch.save(m.state_dict(), f'{BASE}/results/r61_L2_student.pth')
    json.dump(out, open(f'{BASE}/results/r61_L2_scale.json', 'w'), indent=1)
    print('-> results/r61_L2_scale.json')


if __name__ == '__main__':
    main()
