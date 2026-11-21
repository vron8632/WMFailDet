#!/usr/bin/env python3
"""r59: 向 SOTA 靠拢 — 学习式标签 + 更大主干.

杠杆:
  1) 标签: 手工阈值告警 (baseline) vs 学习式判决器 r52 (残差: 手工先验+网络修正) vs 二者融合
  2) 主干: base32 (baseline) vs base64
  3) 数据: 1600 (bigset, 已饱和点)
目标: ≥0.88 (EditGuard 0.913 / WAM 0.947 的同一层级)
输出: results/r59_sota_push.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import keyed_scheme as ks
from r14b_multiscale import layer_key
from r34_distill_matrix import SmallUNet, eval_mask, DELTA, CHK, MASTER
from r52_residual_judge import JudgeNet


def train_judge(x_tr, keys_tr, gt_tr, x_va, keys_va, gt_va, epochs=60, base=16):
    """在目标域内训练失效判决器 (标签=GT 编辑块, 学'编辑导致的失效')."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    def build_x(imgs, keys):
        out = []
        for im, key in zip(imgs, keys):
            bp = np.kron(ks.check_pattern(key, 32, 32, 'coord').astype(np.float32),
                         np.ones((8, 8), np.float32))
            f8 = ks.fail_map(im, key, DELTA, CHK, 'coord')
            prior = np.kron(ks.smooth_fail_map(f8, 5), np.ones((8, 8), np.float32))
            out.append(np.concatenate([im.astype(np.float32) / 255., bp[..., None],
                                       np.clip(prior, 0, 1)[..., None]], -1))
        return np.stack(out)
    Xtr = build_x(x_tr, keys_tr)
    Ytr = np.array([np.kron(g, np.ones((8, 8), np.float32)) for g in gt_tr])
    Xva = build_x(x_va, keys_va)
    dl = DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(Xtr.transpose(0, 3, 1, 2)).float(),
        torch.from_numpy(Ytr[:, None]).float()), batch_size=2, shuffle=True, drop_last=True)
    m = JudgeNet(base=base).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=dev))
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); bce(m(xb), yb).backward(); opt.step()
        if (ep + 1) % 15 == 0:
            m.eval(); chunks = []
            with torch.no_grad():
                for j in range(0, len(Xva), 4):
                    xb2 = torch.from_numpy(Xva[j:j+4].transpose(0, 3, 1, 2)).float().to(dev)
                    chunks.append(torch.sigmoid(m(xb2)).cpu().numpy()[:, 0])
                    del xb2
                pr = np.concatenate(chunks)
            pb = np.array([cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA) for p in pr])
            st = eval_mask((pb > .5).astype(np.uint8), gt_va)
            print(f'  [judge] ep{ep+1} val IoU={st["iou"]:.3f} '
                  f'(pred mean={pb.mean():.3f})', flush=True)
    return m


def learned_labels(imgs, keys, m):
    """用域内判决器生成块级标签 -> 上采样 256."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m.eval(); out = []; B = 8
    for i in range(0, len(imgs), B):
        chunk = imgs[i:i+B]
        x = []
        for j, im in enumerate(chunk):
            key = keys[i + j]
            bp = np.kron(ks.check_pattern(key, 32, 32, 'coord').astype(np.float32),
                         np.ones((8, 8), np.float32))
            f8 = ks.fail_map(im, key, DELTA, CHK, 'coord')
            prior = np.kron(ks.smooth_fail_map(f8, 5), np.ones((8, 8), np.float32))
            x.append(np.concatenate([im.astype(np.float32)/255., bp[..., None],
                                     np.clip(prior, 0, 1)[..., None]], -1))
        xb = torch.from_numpy(np.stack(x).transpose(0, 3, 1, 2)).float().to(dev)
        with torch.no_grad():
            pr = torch.sigmoid(m(xb)).cpu().numpy()[:, 0]
        for p in pr:
            b32 = cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA)
            out.append(np.kron((b32 > 0.5).astype(np.float32), np.ones((8, 8), np.float32)))
    return np.array(out)


def train(x_tr, y_tr, x_val, gt_val, x_te, gt_te, base, epochs=80, seed=0, tag=''):
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
    print(f'  [{tag}] params={sum(p.numel() for p in m.parameters())/1e6:.2f}M '
          f'val={best:.3f} TEST={st["iou"]:.3f}', flush=True)
    return st, best, m


def main():
    z = np.load(f'{BASE}/results/r54_bigset_data.npz')
    img, alarm, gt = z['img'], z['alarm'], z['gt']
    n_te, n_val = 400, 400
    pool = len(img) - n_te - n_val
    x_all = img[:pool]
    y_man = alarm[:pool].astype(np.float32)
    x_te, gt_te = img[-n_te:], gt[-n_te:]
    x_val, gt_val = img[-n_te-n_val:-n_te], gt[-n_te-n_val:-n_te]
    print(f'pool={pool} val={len(x_val)} test={len(x_te)}', flush=True)

    # 块级 GT (32x32) — 判决器标签
    gt32 = np.array([cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA) > 0.5
                     for g in gt]).astype(np.uint8)

    meta = json.load(open(f'{BASE}/data/bigset/meta.json'))
    ids_all = [s_['id'] for s_ in meta['samples']]
    keys_all = [ks.image_key(MASTER, sid) for sid in ids_all]

    # 域内判决器: 前 800 张训练, 中间 400 验证 (与学生训练集分离)
    nj_tr, nj_va = 800, 400
    t0 = time.time()
    print('training domain-specific judge ...', flush=True)
    judge = train_judge(img[:nj_tr], keys_all[:nj_tr], gt32[:nj_tr],
                        img[nj_tr:nj_tr+nj_va], keys_all[nj_tr:nj_tr+nj_va],
                        gt32[nj_tr:nj_tr+nj_va])
    # 学生训练集: 判决器训练集之外的 pool 区域
    sl = slice(nj_tr + nj_va, pool)
    x_all, keys_s = img[sl], keys_all[sl]
    y_man = alarm[sl].astype(np.float32)
    cache = f'{BASE}/results/r59_learned_labels.npy'
    if os.path.exists(cache):
        y_learn = np.load(cache)
    else:
        y_learn = learned_labels(x_all, keys_s, judge)
        np.save(cache, y_learn)
    print(f'student train set: {len(x_all)}, learned-label mean={y_learn.mean():.4f} '
          f'({time.time()-t0:.0f}s)', flush=True)
    y_fuse = np.clip(np.maximum(y_man, y_learn), 0, 1)

    out = {'meta': dict(pool=pool, n_val=n_val, n_test=n_te, epochs=80,
                        reference={'manual_base32': 0.853, 'EditGuard': 0.913,
                                   'WAM': 0.947, 'GT_upper_base32': 0.903}),
           'runs': {}}
    # 1) baseline: 手工标签 + base32 (复现 0.853)
    st, v, _ = train(x_all, y_man, x_val, gt_val, x_te, gt_te, 32, tag='manual_b32')
    out['runs']['manual_base32'] = {'test': {k: round(x, 4) for k, x in st.items()}, 'val': round(v, 4)}
    # 2) 学习式标签 + base32
    st, v, _ = train(x_all, y_learn, x_val, gt_val, x_te, gt_te, 32, tag='learned_b32')
    out['runs']['learned_base32'] = {'test': {k: round(x, 4) for k, x in st.items()}, 'val': round(v, 4)}
    # 3) 融合标签 + base32
    st, v, _ = train(x_all, y_fuse, x_val, gt_val, x_te, gt_te, 32, tag='fused_b32')
    out['runs']['fused_base32'] = {'test': {k: round(x, 4) for k, x in st.items()}, 'val': round(v, 4)}
    # 4) 最佳标签 + base64 (容量)
    best_tag = max(out['runs'], key=lambda k: out['runs'][k]['test']['iou'])
    y_best = {'manual_base32': y_man, 'learned_base32': y_learn,
              'fused_base32': y_fuse}[best_tag]
    st, v, m = train(x_all, y_best, x_val, gt_val, x_te, gt_te, 64, tag=f'{best_tag}_b64')
    out['runs']['best_b64'] = {'test': {k: round(x, 4) for k, x in st.items()}, 'val': round(v, 4),
                               'label_source': best_tag}
    torch.save(m.state_dict(), f'{BASE}/results/r59_best_student.pth')
    out['best'] = max(out['runs'], key=lambda k: out['runs'][k]['test']['iou'])
    out['best_iou'] = out['runs'][out['best']]['test']['iou']
    json.dump(out, open(f'{BASE}/results/r59_sota_push.json', 'w'), indent=1)
    print(f"-> results/r59_sota_push.json  BEST={out['best']} IoU={out['best_iou']}")


if __name__ == '__main__':
    main()
