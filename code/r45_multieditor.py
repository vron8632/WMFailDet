#!/usr/bin/env python3
"""r45: 多编辑器联合训练 — COCO + LaMa + hires(IP2P) 伪标签一起训, 修复 r42 的覆盖崩塌.

对照: 单域学生(r42: LaMa 0.265 / hires 0.246) vs 联合训练学生.
输出: results/r45_multieditor.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs
from r34_distill_matrix import SmallUNet, eval_mask, alarm_from_fail, DELTA, PAY, CHK, MASTER

S = 256


def load_any(root, n=None, scale=None):
    meta = json.load(open(f'{root}/meta.json'))
    out = []
    for s_ in meta['samples']:
        sid = s_['id']
        fpo, fpa = f'{root}/images/orig_{sid}.png', f'{root}/images/att_{sid}.png'
        if not (os.path.exists(fpo) and os.path.exists(fpa)):
            continue
        try:
            o = np.array(Image.open(fpo).convert('RGB'))
            a = np.array(Image.open(fpa).convert('RGB'))
        except Exception:
            continue
        if a.shape[:2] != o.shape[:2]:
            a = np.array(Image.fromarray(a).resize((o.shape[1], o.shape[0]), Image.LANCZOS))
        out.append((str(sid), o, a, s_['bbox']))
        if n and len(out) >= n:
            break
    return out


def make(items):
    eds, als, gts = [], [], []
    for sid, o, a, bb in items:
        o8, a8 = r8.crop8(o), r8.crop8(a)
        key = ks.image_key(MASTER, sid); k4 = layer_key(key, 'bs4')
        wm = ks.embed_keyed(o8, key, DELTA, PAY, CHK, pattern='coord')
        wm = embed_bs(wm, k4, 4, DELTA, (1, 1))
        edited, _ = paste_edit(wm, a8, bb)
        f4 = fail_bs(edited, k4, 4, DELTA, (1, 1))
        al = alarm_from_fail(f4, 4, 5, 32)
        H, W = edited.shape[:2]
        gt = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = bb
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
        edited = np.array(Image.fromarray(edited).resize((S, S), Image.LANCZOS))
        eds.append(edited); als.append(al); gts.append(gt)
    return eds, als, gts   # 尺寸可能不一, 全保持 list


def train(x_tr, y_tr, x_val, gt_val, epochs=80, seed=0):
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
        pr = (infer(x_val) > .5).astype(np.uint8)
        s = eval_mask(pr, gt_val)
        if s['iou'] > best:
            best = s['iou']; best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if (ep+1) % 20 == 0:
            print(f'  ep{ep+1} val={s["iou"]:.3f}', flush=True)
    m.load_state_dict(best_sd)
    return m, best


def evaluate(m, eds, gts, dev):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(eds), 16):
            xb = torch.from_numpy(eds[i:i+16].transpose(0, 3, 1, 2)).float().to(dev)/255.
            out.append((torch.sigmoid(m(xb)).cpu().numpy()[:, 0] > .5).astype(np.uint8))
    pr = np.concatenate(out)
    ious, covs = [], []
    for p, g in zip(pr, gts):
        import cv2
        pp = p.astype(bool)
        if pp.shape != g.shape:
            pp = cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        i = np.logical_and(pp, g).sum(); u = np.logical_or(pp, g).sum()
        ious.append(i/u if u else 1.0)
        covs.append(i/g.sum() if g.sum() else 0.)
    return dict(iou=float(np.mean(ious)), cov50=float(np.mean([c >= .5 for c in covs])))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    def to_arrays(ims, als_):
        xs, ys = [], []
        for im, al in zip(ims, als_):
            im = cv2.resize(im, (S, S), interpolation=cv2.INTER_LANCZOS4)
            al = cv2.resize(al.astype(np.float32), (S, S), interpolation=cv2.INTER_NEAREST)
            xs.append(im); ys.append(al)
        return np.array(xs), np.array(ys)

    d = {}
    for name, root in [('lama', f'{BASE}/data/xdomain/lama_edits'),
                       ('hires', f'{BASE}/data/hires_expanded')]:
        items = load_any(root)
        e, a, g = make(items)
        n = int(len(e) * 0.7)
        xa, ya = to_arrays(e, a)
        d[f'{name}_xtr'], d[f'{name}_ytr'] = xa[:n], ya[:n]
        xte, _ = to_arrays(e[n:], a[n:])
        d[f'{name}_xte'], d[f'{name}_gte'] = xte, g[n:]
        print(f'[{name}] {len(e)} (train {n}) {time.time()-t0:.0f}s', flush=True)
    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    x_tr = np.concatenate([z['img'][:800], d['lama_xtr'], d['hires_xtr']])
    y_tr = np.concatenate([z['alarm4'][:800], d['lama_ytr'], d['hires_ytr']]).astype(np.float32)
    print(f'joint: {len(x_tr)}', flush=True)
    m, val = train(x_tr, y_tr, z['img'][800:1000], z['gt'][800:1000], 80)
    out = {'meta': dict(n_joint=int(len(x_tr)), val_iou=round(val, 4)), 'joint': {}, 'cost_note': ''}
    st = evaluate(m, z['img'][1000:1200], z['gt'][1000:1200], dev)
    out['joint']['coco'] = st
    print(f'[joint] COCO {st["iou"]:.3f}', flush=True)
    for name in ['lama', 'hires']:
        st = evaluate(m, d[f'{name}_xte'], d[f'{name}_gte'], dev)
        out['joint'][name] = st
        print(f'[joint] {name} {st["iou"]:.3f}', flush=True)
    torch.save(m.state_dict(), f'{BASE}/results/r45_student_multied.pth')
    json.dump(out, open(f'{BASE}/results/r45_multieditor.json', 'w'), indent=1)
    print(f'-> results/r45_multieditor.json ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
