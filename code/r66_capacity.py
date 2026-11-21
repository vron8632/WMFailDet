#!/usr/bin/env python3
"""r66: 容量杠杆 — base64 vs base32 (L2 标签, 2000 张, 256 输入)."""
import sys, os, json
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np, torch
from r61_L2_scale import train
from r64_tier_theory import tiers, eval_tiers

z = np.load(f'{BASE}/results/r61_L2_data.npz')
img, al, gt = z['img'], z['alarm'], z['gt']
N_TR, N_VA, N_TE = 2000, 400, 400
out = {'meta': dict(n_train=N_TR, n_test=N_TE, label='L2 (2x2 layer)',
                    baseline={'overall': 0.9347, 'small': 0.8754})}

x_tr, y_tr = img[:N_TR], al[:N_TR].astype(np.float32)
x_va, gt_va = img[N_TR:N_TR+N_VA], gt[N_TR:N_TR+N_VA]
x_te, gt_te = img[-N_TE:], gt[-N_TE:]
st, v, m = train(x_tr, y_tr, x_va, gt_va, x_te, gt_te, 64, tag='L2_b64')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m.eval(); pr = []
with torch.no_grad():
    for i in range(0, len(x_te), 8):
        xb = torch.from_numpy(x_te[i:i+8].transpose(0, 3, 1, 2)).float().to(dev) / 255.
        pr.append(torch.sigmoid(m(xb)).cpu().numpy()[:, 0])
prob = np.concatenate(pr)
t = eval_tiers(prob, gt_te, tiers(gt_te), 'L2_b64')
out['base64_256'] = {'overall': round(st['iou'], 4), 'tiers': t}
torch.save(m.state_dict(), f'{BASE}/results/r66_L2_b64.pth')
json.dump(out, open(f'{BASE}/results/r66_capacity.json', 'w'), indent=1)
print('-> results/r66_capacity.json')
