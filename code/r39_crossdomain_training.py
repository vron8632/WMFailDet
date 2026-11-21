#!/usr/bin/env python3
"""r39: 跨域修复实验.
  A 多域联合训练: COCO 1000 + 4 域各 120 伪标签 -> 测试各域 60 + COCO 200
  B 域内少样本微调: COCO 学生 + 各域 50 伪标签微调 10ep -> 测试该域 60
输出: results/r39_crossdomain_training.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs
from r34_distill_matrix import (SmallUNet, eval_mask, alarm_from_fail,
                                DELTA, PAY, CHK, MASTER)

DOMAINS = ['butterfly', 'sat', 'doc', 'pokemon']
N_TR = 120; N_TE = 60


def load_domain(ds, n):
    root = f'{BASE}/data/xdomain/{ds}_edits'
    meta = json.load(open(f'{root}/meta.json'))
    out = []
    for s_ in meta['samples']:
        sid = s_['id']
        fpo, fpa = f'{root}/images/orig_{sid}.png', f'{root}/images/att_{sid}.png'
        if not (os.path.exists(fpo) and os.path.exists(fpa)):
            continue
        try:
            o = r8.crop8(np.array(Image.open(fpo).convert('RGB')))
            a = np.array(Image.open(fpa).convert('RGB'))
            if a.shape[:2] != o.shape[:2]:
                a = np.array(Image.fromarray(a).resize((o.shape[1], o.shape[0]), Image.LANCZOS))
            a = r8.crop8(a)
        except Exception:
            continue
        out.append((str(sid), o, a, s_['bbox']))
        if len(out) >= n:
            break
    return out


def make_pair(sid, o, a, bbox):
    key = ks.image_key(MASTER, sid); k4 = layer_key(key, 'bs4')
    wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
    wm = embed_bs(wm, k4, 4, DELTA, (1, 1))
    edited, _ = paste_edit(wm, a, bbox)
    f4 = fail_bs(edited, k4, 4, DELTA, (1, 1))
    alarm = alarm_from_fail(f4, 4, 5, 32)
    H, W = edited.shape[:2]
    gt = np.zeros((H, W), np.uint8)
    x1, y1, x2, y2 = bbox
    gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
    return edited, alarm, gt


def train(x_tr, y_tr, x_va, gt_va, epochs, m=None, lr=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float() / 255.,
        torch.from_numpy(y_tr[:, None]).float())
    dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    if m is None:
        m = SmallUNet(base=32).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=dev))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    def lossf(lo_, tg):
        b = bce(lo_, tg)
        p = torch.sigmoid(lo_).flatten(1); t = tg.flatten(1)
        di = (p*t).sum(1); ds_ = p.sum(1)+t.sum(1)
        return b + (1 - (2*di+1)/(ds_+1)).mean()
    best = (0., None)
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            pr = []
            for i in range(0, len(x_va), 16):
                xb = torch.from_numpy(x_va[i:i+16].transpose(0, 3, 1, 2)).float().to(dev)/255.
                pr.append(torch.sigmoid(m(xb)).cpu().numpy())
        pr = (np.concatenate(pr)[:, 0] > 0.5).astype(np.uint8)
        s = eval_mask(pr, gt_va); s.pop('per_iou')
        if s['iou'] > best[0]:
            best = (s['iou'], {k: v.detach().clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1])
    return m, best[0]


def infer(m, imgs, dev, bs=16):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            xb = torch.from_numpy(imgs[i:i+bs].transpose(0, 3, 1, 2)).float().to(dev)/255.
            out.append(torch.sigmoid(m(xb)).cpu().numpy())
    return (np.concatenate(out)[:, 0] > 0.5).astype(np.uint8)


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = f'{BASE}/results/r39_xdomain_data.npz'
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        data = {d: (z[f'{d}_img'], z[f'{d}_alarm'], z[f'{d}_gt']) for d in DOMAINS}
    else:
        data = {}
        t0 = time.time()
        for d in DOMAINS:
            items = load_domain(d, N_TR + N_TE)
            eds, als, gts = [], [], []
            for sid, o, a, bb in items:
                e, al, g = make_pair(sid, o, a, bb)
                eds.append(e); als.append(al); gts.append(g)
            data[d] = (np.array(eds), np.array(als), np.array(gts))
            print(f'[{d}] generated {len(eds)} ({time.time()-t0:.0f}s)', flush=True)
        np.savez_compressed(cache, **{f'{d}_{k}': v for d in DOMAINS
                                      for k, v in zip(('img', 'alarm', 'gt'), data[d])})

    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    # 每域按实际数量动态划分 (butterfly 只有 60 张): 前 2/3 训练, 后 1/3 测试
    splits = {d: int(len(data[d][0]) * 0.67) for d in DOMAINS}
    coco_x = np.concatenate([z['img'][:800]] + [data[d][0][:splits[d]] for d in DOMAINS])
    coco_y = np.concatenate([z['alarm4'][:800]] + [data[d][1][:splits[d]] for d in DOMAINS]).astype(np.float32)
    print(f'joint training set: {len(coco_x)} splits={splits}', flush=True)

    out = {'meta': dict(domain_splits=splits, joint_n=int(len(coco_x)),
                        finetune_epochs=10), 'joint': {}, 'finetune': {}}
    # ---- A 多域联合训练
    m, _ = train(coco_x, coco_y, z['img'][800:1000], z['gt'][800:1000], 80)
    torch.save(m.state_dict(), f'{BASE}/results/r39_student_joint.pth')
    pr = infer(m, z['img'][1000:1200], dev)
    s = eval_mask(pr, z['gt'][1000:1200]); s.pop('per_iou')
    out['joint']['coco'] = s
    print(f'[joint] COCO IoU={s["iou"]:.3f}', flush=True)
    for d in DOMAINS:
        img, al, gt = data[d]
        te = slice(splits[d], len(img))
        pr = infer(m, img[te], dev)
        s = eval_mask(pr, gt[te]); s.pop('per_iou')
        out['joint'][d] = s
        print(f'[joint] {d} IoU={s["iou"]:.3f}', flush=True)

    # ---- B 域内少样本微调 (从 COCO 学生起步)
    base = SmallUNet(base=32).to(dev)
    base.load_state_dict(torch.load(f'{BASE}/results/r43_best_student.pth', map_location=dev))
    for d in DOMAINS:
        img, al, gt = data[d]
        n = len(img); nft = min(50, n // 3)
        te = slice(splits[d], n)
        fin = SmallUNet(base=32).to(dev)
        fin.load_state_dict(base.state_dict())
        fin, _ = train(img[:nft], al[:nft].astype(np.float32),
                       img[nft:2*nft], gt[nft:2*nft], 10, m=fin, lr=5e-4)
        pr = infer(fin, img[te], dev)
        s = eval_mask(pr, gt[te]); s.pop('per_iou')
        out['finetune'][d] = s
        print(f'[finetune n=50] {d} IoU={s["iou"]:.3f}', flush=True)

    json.dump(out, open(f'{BASE}/results/r39_crossdomain_training.json', 'w'), indent=1)
    print('-> results/r39_crossdomain_training.json', flush=True)


if __name__ == '__main__':
    main()
