#!/usr/bin/env python3
"""r64: 标签分辨率增益的分层验证 — 检验理论预测.

理论: 编辑区内有效块数 n_b = A/b^2; 小编辑在粗块下块数不足 -> 标签稀疏 -> 学不好.
      预测: 更细标签 (b=2) 的增益应主要集中在 small/mid 编辑, large/full 增益小.

实验: 同一 2000 张训练集, L2 标签 vs L4 标签, 学生(base32) 相同;
      在 400 测试集上按编辑面积分层报告 IoU.
输出: results/r64_tier_theory.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np
import torch
from r34_distill_matrix import eval_mask
from r61_L2_scale import train

N_TR, N_VA, N_TE = 2000, 400, 400


def tiers(gt_list):
    """按编辑面积分档 (与论文一致)."""
    areas = np.array([g.mean() for g in gt_list])
    return {'small': areas < 0.10, 'mid': (areas >= 0.10) & (areas < 0.30),
            'large': (areas >= 0.30) & (areas < 0.90), 'full': areas >= 0.90}


def eval_tiers(prob, gt, mask, name):
    pb = (prob > 0.5).astype(np.uint8)
    out = {}
    for k, m in mask.items():
        if m.sum() == 0:
            continue
        s = eval_mask(pb[m], gt[m]); s.pop('per_iou')
        out[k] = {'n': int(m.sum()), 'iou': round(s['iou'], 4),
                  'cov50': round(s['cov50'], 4), 'area_ratio': round(float(s['area_ratio']), 3)}
    print(f'  [{name}]', {k: v['iou'] for k, v in out.items()}, flush=True)
    return out


def main():
    # L2 标签数据
    z2 = np.load(f'{BASE}/results/r61_L2_data.npz')
    img2, al2, gt2 = z2['img'], z2['alarm'], z2['gt']
    # L4 标签数据 (r54 bigset, 同批样本顺序一致?)
    z4 = np.load(f'{BASE}/results/r54_bigset_data.npz')
    img4, al4, gt4 = z4['img'], z4['alarm'], z4['gt']
    n = min(len(img2), len(img4))
    x_te, gt_te = img2[-N_TE:], gt2[-N_TE:]
    x_va, gt_va = img2[N_TR:N_TR+N_VA], gt2[N_TR:N_TR+N_VA]
    tier_mask = tiers(gt_te)
    print('tier counts:', {k: int(v.sum()) for k, v in tier_mask.items()}, flush=True)

    out = {'meta': dict(n_train=N_TR, n_test=N_TE, student='SmallUNet base32',
                        theory='n_b = A/b^2; finer labels should mainly help small edits'),
           'L2': {}, 'L4': {}}

    # L2: 直接用 r61 已训好的权重 (seed0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    from r34_distill_matrix import SmallUNet
    m = SmallUNet(base=32).to(dev)
    m.load_state_dict(torch.load(f'{BASE}/results/r61_L2_student.pth', map_location=dev))
    m.eval()
    pr = []
    with torch.no_grad():
        for i in range(0, len(x_te), 8):
            xb = torch.from_numpy(x_te[i:i+8].transpose(0, 3, 1, 2)).float().to(dev)/255.
            pr.append(torch.sigmoid(m(xb)).cpu().numpy()[:, 0])
    prob2 = np.concatenate(pr)
    out['L2'] = eval_tiers(prob2, gt_te, tier_mask, 'L2')

    # L4: 用同一批图重训 (标签 = r54 的 alarm, 即 4×4 层告警)
    x_tr4, y_tr4 = img4[:N_TR], al4[:N_TR].astype(np.float32)
    x_va4, gt_va4 = img4[N_TR:N_TR+N_VA], gt4[N_TR:N_TR+N_VA]
    x_te4, gt_te4 = img4[-N_TE:], gt4[-N_TE:]
    st, v, m4 = train(x_tr4, y_tr4, x_va4, gt_va4, x_te4, gt_te4, 32, tag='L4_ref')
    m4.eval(); pr = []
    with torch.no_grad():
        for i in range(0, len(x_te4), 8):
            xb = torch.from_numpy(x_te4[i:i+8].transpose(0, 3, 1, 2)).float().to(dev)/255.
            pr.append(torch.sigmoid(m4(xb)).cpu().numpy()[:, 0])
    prob4 = np.concatenate(pr)
    out['L4'] = eval_tiers(prob4, gt_te4, tiers(gt_te4), 'L4')
    out['L4']['_overall'] = round(st['iou'], 4)
    out['L2']['_overall'] = round(float(eval_mask((prob2 > .5).astype(np.uint8), gt_te)['iou']), 4)

    # 增益分解
    gains = {}
    for k in ('small', 'mid', 'large', 'full'):
        if k in out['L2'] and k in out['L4']:
            gains[k] = round(out['L2'][k]['iou'] - out['L4'][k]['iou'], 4)
    out['gain_L2_minus_L4'] = gains
    out['verdict'] = 'prediction: gain(small) > gain(large); observed: ' + str(gains)
    print('gains:', gains, flush=True)
    json.dump(out, open(f'{BASE}/results/r64_tier_theory.json', 'w'), indent=1)
    print('-> results/r64_tier_theory.json')


if __name__ == '__main__':
    main()
