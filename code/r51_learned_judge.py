#!/usr/bin/env python3
"""r51: 可学习失效判决器 (learned failure judge) — 替代"逐块DCT比较+固定阈值".

设计:
  输入: 退化后的水印图像 (RGB) + 期望校验位图 (32x32 上采样, 1 通道) = 4 通道
  输出: 失效概率图 (256x256), 评估时下采样回 32x32 块级
  标签: **编辑区 GT** (块级) —— 学"编辑导致的失效", 而非模仿手工判决
        (手工判决在 JPEG 下把全图判失效, 学习器应学会区分压缩 vs 编辑)

对照: 手工判决 (r36: clean 0.656/jpeg85 0.434/jpeg75 0.468/jpeg50 0.079
      /noise 0.630/sharpen 0.066/resize 0.252)
输出: results/r51_learned_judge.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs
from r34_distill_matrix import ConvBlock, eval_mask, DELTA, PAY, CHK, MASTER
from r10_specificity import jpeg, noise, sharpen, resize_round, rotate

AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
DEGS = {'clean': lambda im: im, 'jpeg85': lambda im: jpeg(im, 85),
        'jpeg75': lambda im: jpeg(im, 75), 'jpeg50': lambda im: jpeg(im, 50),
        'noise2': lambda im: noise(im, 2), 'sharpen': lambda im: sharpen(im),
        'resize0.8': lambda im: resize_round(im, 0.8)}


def bitplane(key, n=32):
    """期望校验位图 (n×n) -> 256×256 单通道 float."""
    p = ks.check_pattern(key, n, n, 'coord').astype(np.float32)
    up = np.kron(p, np.ones((256 // n, 256 // n), np.float32))
    return np.clip(up, 0, 1)


def build(n, cache):
    """-> imgs (N,256,256,3), bits (N,256,256), gts (N,32,32), degen list"""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z['img'], z['bit'], z['gt'], list(z['deg'])
    meta = json.load(open(f'{AS}/meta.json'))
    ids = []
    for s_ in meta['samples']:
        sid = s_['id']
        if os.path.exists(f'{AS}/images/orig_{sid}.png') and os.path.exists(f'{AS}/images/att_{sid}.png'):
            ids.append(s_)
        if len(ids) >= 1200:
            break
    imgs, bits, gts, degs = [], [], [], []
    t0 = time.time()
    for s_ in ids[:n]:
        sid = s_['id']
        o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
        a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        key = ks.image_key(MASTER, sid)
        k4 = layer_key(key, 'bs4')
        wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        wm = embed_bs(wm, k4, 4, DELTA, (1, 1))
        edited, _ = paste_edit(wm, a, s_['bbox'])
        # GT 块级 (32x32, 8px 块)
        H, W = edited.shape[:2]
        gt = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = s_['bbox']
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
        g32 = (cv2.resize(gt, (32, 32), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
        bp = bitplane(key, 32)
        # 每个样本 2 个变体: 编辑图 + 干净图, 各随机一个退化
        for src, lab in ((edited, g32), (wm, np.zeros_like(g32))):
            dname = str(np.random.choice(list(DEGS.keys())))
            im_deg = DEGS[dname](src)
            imgs.append(im_deg); bits.append(bp); gts.append(lab); degs.append(dname)
        if len(imgs) >= n * 2:
            break
    imgs = np.array(imgs); bits = np.array(bits); gts = np.array(gts)
    np.savez_compressed(cache, img=imgs, bit=bits, gt=gts, deg=np.array(degs))
    print(f'dataset: {len(imgs)} ({time.time()-t0:.0f}s)', flush=True)
    return imgs, bits, gts, degs


class JudgeNet(nn.Module):
    """4 通道输入 (RGB + 期望位图) -> 1 通道失效概率."""
    def __init__(self, base=16):
        super().__init__()
        self.e1 = ConvBlock(4, base); self.e2 = ConvBlock(base, base*2)
        self.e3 = ConvBlock(base*2, base*4); self.e4 = ConvBlock(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.d3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = ConvBlock(base*2, base)
        self.head = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        c1 = self.e1(x); c2 = self.e2(self.pool(c1))
        c3 = self.e3(self.pool(c2)); c4 = self.e4(self.pool(c3))
        d3 = self.d3(torch.cat([self.up3(c4), c3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), c2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), c1], 1))
        return self.head(d1)


def to_blocks(prob):
    """256x256 概率 -> 32x32 块级概率."""
    return np.array([cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA) for p in prob])


def train(x_tr, b_tr, g_tr, x_val, b_val, g_val, epochs=60, base=16, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = np.concatenate([x_tr, b_tr[..., None]], -1)  # (N,256,256,4)
    V = np.concatenate([x_val, b_val[..., None]], -1)
    Y = np.array([np.kron(g, np.ones((8, 8), np.float32)) for g in g_tr])
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X.transpose(0, 3, 1, 2)).float()/255.,
        torch.from_numpy(Y[:, None]).float())
    dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    m = JudgeNet(base=base).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=dev))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    def infer(arr):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(arr), 16):
                xb = torch.from_numpy(arr[i:i+16].transpose(0, 3, 1, 2)).float().to(dev)/255.
                out.append(torch.sigmoid(m(xb)).cpu().numpy()[:, 0])
        return np.concatenate(out)
    best, best_sd = -1., None
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); bce(m(xb), yb).backward(); opt.step()
        sched.step()
        pb = to_blocks(infer(V))
        s = eval_mask((pb > .5).astype(np.uint8), g_val)
        if s['iou'] > best:
            best = s['iou']; best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if (ep+1) % 15 == 0:
            print(f'  ep{ep+1} val IoU={s["iou"]:.3f}', flush=True)
    m.load_state_dict(best_sd)
    return m, infer, best


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = f'{BASE}/results/r51_data.npz'
    img, bit, gt, degs = build(1200, cache)
    # 三分: 前 800 张图 (1600 变体) 训练, 800-1000 (400) 验证, 1000-1200 (400) 测试
    n_tr, n_val = 1600, 400
    x_tr, b_tr, g_tr = img[:n_tr], bit[:n_tr], gt[:n_tr]
    x_val, b_val, g_val = img[n_tr:n_tr+n_val], bit[n_tr:n_tr+n_val], gt[n_tr:n_tr+n_val]
    x_te, b_te, g_te = img[n_tr+n_val:], bit[n_tr+n_val:], gt[n_tr+n_val:]
    d_te = degs[n_tr+n_val:]
    print(f'train={len(x_tr)} val={len(x_val)} test={len(x_te)}', flush=True)

    m, infer, val = train(x_tr, b_tr, g_tr, x_val, b_val, g_val)
    X_te = np.concatenate([x_te, b_te[..., None]], -1)
    pb = to_blocks(infer(X_te))
    out = {'meta': dict(n_train=len(x_tr), n_val=len(x_val), n_test=len(x_te),
                        val_iou=round(val, 4),
                        input='RGB + expected check-bitplane (4ch)',
                        label='GT edit blocks (learns edit-caused failure, not mimic manual)'),
           'by_degradation': {}}
    st = eval_mask((pb > .5).astype(np.uint8), g_te); st.pop('per_iou')
    out['overall'] = {k: round(v, 4) for k, v in st.items()}
    print(f'[learned judge] ALL TEST IoU={st["iou"]:.3f}', flush=True)
    d_te = np.array(d_te)
    for dname in DEGS:
        msk = d_te == dname
        if msk.sum() == 0:
            continue
        s = eval_mask((pb[msk] > .5).astype(np.uint8), g_te[msk]); s.pop('per_iou')
        out['by_degradation'][dname] = {'n': int(msk.sum()), **{k: round(v, 4) for k, v in s.items()}}
        print(f'  [{dname:10s}] n={msk.sum():3d} IoU={s["iou"]:.3f} cov={s["cov50"]:.1%}', flush=True)
    torch.save(m.state_dict(), f'{BASE}/results/r51_judge.pth')
    json.dump(out, open(f'{BASE}/results/r51_learned_judge.json', 'w'), indent=1)
    print('-> results/r51_learned_judge.json')


if __name__ == '__main__':
    main()
