#!/usr/bin/env python3
"""r47: 新机制一 — 辅助任务蒸馏 (auxiliary failure-map head).

学生 UNet 加第二 head: 训练时同时预测"失效图"(连续), 主 head 出篡改掩膜.
推理时只用主 head (不输入水印, 不输出失效图) — 预期: 失效信号被"内化"进表征.
对照: λ ∈ {0(baseline), 0.1, 0.5, 1.0}
输出: results/r47_aux_distill.json
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
from r14b_multiscale import layer_key
from r34_distill_matrix import ConvBlock, SmallUNet, eval_mask, DELTA, PAY, CHK, MASTER, AS


class AuxUNet(nn.Module):
    """SmallUNet + 第二个输出头 (失效图预测), 推理时只用主 head."""
    def __init__(self, base=32):
        super().__init__()
        self.backbone = SmallUNet(base=base)
        self.aux_head = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1))
    def forward(self, x, with_aux=False):
        # 复用 SmallUNet 的编码器/解码器, 在主 head 之前取出特征
        b = self.backbone
        c1 = b.e1(x); c2 = b.e2(b.pool(c1))
        c3 = b.e3(b.pool(c2)); c4 = b.e4(b.pool(c3))
        d3 = b.d3(torch.cat([b.up3(c4), c3], 1))
        d2 = b.d2(torch.cat([b.up2(d3), c2], 1))
        d1 = b.d1(torch.cat([b.up1(d2), c1], 1))
        main = b.head(d1)
        if with_aux:
            return main, self.aux_head(d1)
        return main


def fail_maps_for(sids, edited_imgs):
    """对编辑图重算平滑失效图 (256x256, 归一化), 作为 aux 回归目标."""
    out = []
    for sid, im in zip(sids, edited_imgs):
        key = ks.image_key(MASTER, sid)
        f8 = ks.fail_map(im, key, DELTA, CHK, 'coord')
        sm = ks.smooth_fail_map(f8, 5)
        sm = np.kron(sm, np.ones((8, 8), np.float32))
        out.append(np.clip((sm - 0.25) / 0.45, 0, 1))
    return np.array(out, np.float32)


def load_attackset(n):
    meta = json.load(open(f'{AS}/meta.json'))
    out = []
    for s_ in meta['samples']:
        sid = s_['id']
        if not (os.path.exists(f'{AS}/images/orig_{sid}.png') and os.path.exists(f'{AS}/images/att_{sid}.png')):
            continue
        out.append((str(sid), None, None, s_['bbox']))
        if len(out) >= n:
            break
    return out


def run(x_tr, y_tr, aux_tr, x_val, gt_val, x_te, gt_te, lam, epochs=80, seed=0, tag=''):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float()/255.,
        torch.from_numpy(y_tr[:, None]).float(),
        torch.from_numpy(aux_tr[:, None]).float()), batch_size=16, shuffle=True, drop_last=True)
    m = AuxUNet(base=32).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=dev))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    def main_loss(lo_, tg):
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
        for xb, yb, ab in dl:
            xb, yb, ab = xb.to(dev), yb.to(dev), ab.to(dev)
            opt.zero_grad()
            if lam > 0:
                mo, ao = m(xb, with_aux=True)
                loss = main_loss(mo, yb) + lam * nn.functional.mse_loss(torch.sigmoid(ao), ab)
            else:
                loss = main_loss(m(xb), yb)
            loss.backward(); opt.step()
        sched.step()
        s = eval_mask((infer(x_val) > .5).astype(np.uint8), gt_val)
        if s['iou'] > best:
            best = s['iou']; best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if (ep+1) % 20 == 0:
            print(f'  [{tag}] ep{ep+1} val={s["iou"]:.3f}', flush=True)
    m.load_state_dict(best_sd)
    st = eval_mask((infer(x_te) > .5).astype(np.uint8), gt_te); st.pop('per_iou')
    print(f'  [{tag}] λ={lam} val={best:.3f} TEST={st["iou"]:.3f}', flush=True)
    return st, best


def main():
    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    img, gt, a4 = z['img'], z['gt'], z['alarm4']
    aux_cache = f'{BASE}/results/r47_aux_targets.npy'
    if os.path.exists(aux_cache):
        aux = np.load(aux_cache)
    else:
        meta = json.load(open(f'{AS}/meta.json'))
        sids, ok = [], 0
        for s_ in meta['samples']:
            sid = s_['id']
            if os.path.exists(f'{AS}/images/orig_{sid}.png') and os.path.exists(f'{AS}/images/att_{sid}.png'):
                sids.append(sid); ok += 1
            if ok >= 1200:
                break
        aux = fail_maps_for(sids[:800], img[:800])
        np.save(aux_cache, aux)
        print('aux targets built', aux.shape, float(aux.mean()), flush=True)
    out = {'meta': dict(n_train=800, n_val=200, n_test=200, epochs=80, seed=0,
                        mechanism='auxiliary failure-map head (not used at inference)'),
           'runs': {}}
    for lam in (0.0, 0.1, 0.5, 1.0):
        st, val = run(img[:800], a4[:800].astype(np.float32), aux[:800],
                      img[800:1000], gt[800:1000], img[1000:1200], gt[1000:1200],
                      lam, tag=f'aux{lam}')
        out['runs'][f'lambda{lam}'] = {'test': {k: round(v, 4) for k, v in st.items()},
                                       'val': round(val, 4)}
    json.dump(out, open(f'{BASE}/results/r47_aux_distill.json', 'w'), indent=1)
    print('-> results/r47_aux_distill.json')


if __name__ == '__main__':
    main()
