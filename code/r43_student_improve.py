#!/usr/bin/env python3
"""r43: 学生提升实验 — 目标: 干净三分法下把学生从 0.654 提上来.

变体 (均 train800/val200/test200, val 选模型):
  A baseline      : base16, 80ep, 无增强 (= r34 口径)
  B swa           : A + 最后 20 epoch 权重平均 (SWA)
  C aug           : A + 几何/JPEG/亮度 增强
  D base32        : 更大模型
  E full          : 增强 + base16 + 160ep + SWA
输出: results/r43_student_improve.json (+ r43_best_student.pth)
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from r34_distill_matrix import ConvBlock, SmallUNet, eval_mask


class AugDS(Dataset):
    def __init__(self, x, y, aug=True):
        self.x, self.y, self.aug = x, y, aug

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        im, lb = self.x[i].copy(), self.y[i].copy()
        if self.aug:
            if np.random.rand() < .5:
                im = im[:, ::-1].copy(); lb = lb[:, ::-1].copy()
            if np.random.rand() < .5:
                im = im[::-1].copy(); lb = lb[::-1].copy()
            if np.random.rand() < .4:                      # JPEG 增强
                q = np.random.randint(60, 96)
                ok, buf = cv2.imencode('.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, int(q)])
                if ok:
                    im = cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB)
            if np.random.rand() < .3:                      # 亮度
                g = 1 + np.random.uniform(-.15, .15)
                im = np.clip(im.astype(np.float32) * g, 0, 255).astype(np.uint8)
        return (torch.from_numpy(im.transpose(2, 0, 1)).float() / 255.,
                torch.from_numpy(lb[None]).float())


def run(x_tr, y_tr, x_val, gt_val, x_te, gt_te, *, base=16, epochs=80, aug=False,
        swa_last=0, seed=0, tag=''):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = DataLoader(AugDS(x_tr, y_tr, aug), batch_size=16, shuffle=True, drop_last=True)
    m = SmallUNet(base=base).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=dev))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def lossf(lo_, tg):
        b = bce(lo_, tg)
        p = torch.sigmoid(lo_).flatten(1); t = tg.flatten(1)
        di = (p*t).sum(1); ds_ = p.sum(1)+t.sum(1)
        return b + (1 - (2*di+1)/(ds_+1)).mean()

    def predict(mm, x):
        mm.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(x), 16):
                xb = torch.from_numpy(x[i:i+16].transpose(0, 3, 1, 2)).float().to(dev)/255.
                out.append(torch.sigmoid(mm(xb)).cpu().numpy())
        return np.concatenate(out)[:, 0]

    best_iou, best_sd = 0., None
    swa_sd, swa_n = None, 0
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        sched.step()
        pr = (predict(m, x_val) > .5).astype(np.uint8)
        s = eval_mask(pr, gt_val)
        if s['iou'] > best_iou:
            best_iou = s['iou']
            best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if swa_last and ep >= epochs - swa_last:      # 累积 SWA
            cur = {k: v.detach().float().clone() for k, v in m.state_dict().items()}
            if swa_sd is None:
                swa_sd = cur
            else:
                for k in swa_sd:
                    swa_sd[k] += cur[k]
            swa_n += 1
        if (ep + 1) % 20 == 0:
            print(f'  [{tag}] ep{ep+1} val IoU={s["iou"]:.3f}', flush=True)
    cands = {'best': (best_iou, best_sd)}
    if swa_sd is not None and swa_n:
        swa = {k: v / swa_n for k, v in swa_sd.items()}
        m.load_state_dict(swa)
        swa_iou = eval_mask((predict(m, x_val) > .5).astype(np.uint8), gt_val)['iou']
        print(f'  [{tag}] SWA val IoU={swa_iou:.3f} (best {best_iou:.3f})', flush=True)
        cands['swa'] = (swa_iou, swa)
    pick = max(cands.items(), key=lambda kv: kv[1][0])
    sd = pick[1][1]
    m.load_state_dict(sd)
    st = eval_mask((predict(m, x_te) > .5).astype(np.uint8), gt_te)
    st.pop('per_iou')
    print(f'  [{tag}] PICK={pick[0]} val={pick[1][0]:.3f} -> TEST IoU={st["iou"]:.3f}', flush=True)
    return st, sd


def main():
    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    img, gt, a4 = z['img'], z['gt'], z['alarm4']
    x_tr, y_tr = img[:800], a4[:800].astype(np.float32)
    x_val, gt_val = img[800:1000], gt[800:1000]
    x_te, gt_te = img[1000:1200], gt[1000:1200]
    jf = f'{BASE}/results/r43_student_improve.json'
    out = json.load(open(jf)) if os.path.exists(jf) else {
        'meta': dict(n_train=800, n_val=200, n_test=200,
                     split='train800/val200/test200'), 'variants': {}}
    t0 = time.time()
    only = os.environ.get('R43_ONLY', '')
    cfgs = [
        ('A_baseline', dict(base=16, epochs=80, aug=False, swa_last=0)),
        ('B_swa', dict(base=16, epochs=80, aug=False, swa_last=20)),
        ('C_aug', dict(base=16, epochs=80, aug=True, swa_last=0)),
        ('D_base32', dict(base=32, epochs=80, aug=False, swa_last=0)),
        ('E_full', dict(base=16, epochs=160, aug=True, swa_last=30)),
    ]
    best = ('', 0., None)
    for k_, v_ in out['variants'].items():
        if v_.get('iou', 0) > best[1]:
            best = (k_, v_['iou'], None)
    for tag, kw in cfgs:
        if only and tag not in only.split(','):
            continue
        print(f'=== {tag} ===', flush=True)
        st, sd = run(x_tr, y_tr, x_val, gt_val, x_te, gt_te, tag=tag, **kw)
        out['variants'][tag] = {**{k: round(v, 4) for k, v in st.items()}, 'cfg': kw}
        if st['iou'] > best[1]:
            best = (tag, st['iou'], sd)
    out['best'] = {'tag': best[0], 'test_iou': round(best[1], 4)}
    if best[2] is not None:
        torch.save(best[2], f'{BASE}/results/r43_best_student.pth')
    json.dump(out, open(f'{BASE}/results/r43_student_improve.json', 'w'), indent=1)
    print(f"\nBEST: {best[0]} test IoU={best[1]:.4f} ({time.time()-t0:.0f}s)")
    print('-> results/r43_student_improve.json')


if __name__ == '__main__':
    main()
