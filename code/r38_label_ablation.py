#!/usr/bin/env python3
"""r38: 标签消融 + 数据规模消融.

标签变体 (同架构/配置/seed):
  failure   : WMFE alarm4 (本文)
  bbox_gt   : 精确 bbox 矩形 (=GT, 标注的乐观上界)
  bbox_noisy: bbox 外扩15% + 抖动 (模拟人工框标注误差)
  point     : bbox 中心 8x8 单点 (点监督)
  image_lvl : 全图恒1 (图像级监督, 无定位信息)
规模: failure 标签 n=250/500/1000
输出: results/r38_label_ablation.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from r34_distill_matrix import SmallUNet, eval_mask


def bbox_mask(gt, expand=0.0, jitter=0, rng=None):
    H, W = gt.shape
    ys, xs = np.where(gt > 0)
    if not len(xs):
        return np.zeros_like(gt)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    dx, dy = int((x2 - x1) * expand), int((y2 - y1) * expand)
    if jitter and rng is not None:
        jx, jy = rng.integers(-jitter, jitter + 1, 2)
    else:
        jx = jy = 0
    x1 = max(0, x1 - dx + jx); y1 = max(0, y1 - dy + jy)
    x2 = min(W, x2 + dx + jx); y2 = min(H, y2 + dy + jy)
    m = np.zeros_like(gt); m[y1:y2, x1:x2] = 1
    return m


def point_mask(gt, k=8):
    H, W = gt.shape
    ys, xs = np.where(gt > 0)
    if not len(xs):
        return np.zeros_like(gt)
    cy, cx = int(ys.mean()), int(xs.mean())
    m = np.zeros_like(gt)
    m[max(0, cy-k//2):cy+k//2, max(0, cx-k//2):cx+k//2] = 1
    return m


def train(x_tr, y_tr, x_val, gt_val, x_te, gt_te, epochs, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr.transpose(0, 3, 1, 2)).float() / 255.,
        torch.from_numpy(y_tr[:, None]).float())
    dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    m = SmallUNet(base=32).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=dev))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    def lossf(lo_, tg):
        b = bce(lo_, tg)
        p = torch.sigmoid(lo_).flatten(1); t = tg.flatten(1)
        di = (p*t).sum(1); ds_ = p.sum(1)+t.sum(1)
        return b + (1 - (2*di+1)/(ds_+1)).mean()
    best, best_sd = 0., None
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            pr = []
            for i in range(0, len(x_val), 16):
                xb = torch.from_numpy(x_val[i:i+16].transpose(0, 3, 1, 2)).float().to(dev)/255.
                pr.append(torch.sigmoid(m(xb)).cpu().numpy())
        pr = (np.concatenate(pr)[:, 0] > 0.5).astype(np.uint8)
        s = eval_mask(pr, gt_val); s.pop('per_iou')
        if s['iou'] > best:
            best = s['iou']; best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_sd)
    m.eval()
    with torch.no_grad():
        pr = []
        for i in range(0, len(x_te), 16):
            xb = torch.from_numpy(x_te[i:i+16].transpose(0, 3, 1, 2)).float().to(dev)/255.
            pr.append(torch.sigmoid(m(xb)).cpu().numpy())
    pr = (np.concatenate(pr)[:, 0] > 0.5).astype(np.uint8)
    st = eval_mask(pr, gt_te); st.pop('per_iou')
    return st['iou'], best


def main():
    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    img, gt, a4 = z['img'], z['gt'], z['alarm4']
    # 严格三分: 训练800 / 验证200(选模型) / 测试200(只报一次)
    x_tr, gt_tr = img[:800], gt[:800]
    x_val, gt_val = img[800:1000], gt[800:1000]
    x_te, gt_te = img[1000:1200], gt[1000:1200]
    rng = np.random.default_rng(0)
    labels = {
        'failure': a4[:800].astype(np.float32),
        'bbox_gt': np.array([bbox_mask(g) for g in gt_tr], np.float32),
        'bbox_noisy': np.array([bbox_mask(g, expand=0.15, jitter=6, rng=rng) for g in gt_tr], np.float32),
        'point': np.array([point_mask(g) for g in gt_tr], np.float32),
        'image_lvl': np.ones_like(a4[:800], np.float32),
    }
    out = {'meta': dict(n_train=800, n_val=200, n_test=200, epochs=80, seed=0,
                        split='train800/val200/test200', labels=list(labels)),
           'labels': {}, 'scaling': {}}
    t0 = time.time()
    for tag, y in labels.items():
        te_iou, val_iou = train(x_tr, y, x_val, gt_val, x_te, gt_te, 80)
        out['labels'][tag] = {'iou_test': round(te_iou, 4), 'iou_val': round(val_iou, 4)}
        print(f'[{tag:10s}] test IoU={te_iou:.4f} (val {val_iou:.4f})  ({time.time()-t0:.0f}s)', flush=True)
    # 数据规模
    for n in (250, 500):
        te_iou, val_iou = train(x_tr[:n], labels['failure'][:n], x_val, gt_val, x_te, gt_te, 80)
        out['scaling'][f'n{n}'] = {'iou_test': round(te_iou, 4), 'iou_val': round(val_iou, 4)}
        print(f'[scaling n={n}] test IoU={te_iou:.4f}', flush=True)
    json.dump(out, open(f'{BASE}/results/r38_label_ablation.json', 'w'), indent=1)
    print(f'-> results/r38_label_ablation.json ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
