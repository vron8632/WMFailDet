#!/usr/bin/env python3
"""r77: 被动取证基线 — Noiseprint++ 无监督痕迹检测 (key-free 可用的同类对手).

背景: TruFor 官方权重为 epoch 2 未训练状态 (trufor.pth.tar), 不可用;
      其他被动方法 (MVSS-Net / CAT-Net / ManTraNet) 权重托管于 Google Drive, 本环境不可达.
因此用 TruFor 的指纹前端 Noiseprint++ (本地权重, 无监督, 仅需图像) 构建被动基线.

检测规则 (固定, 不调参拟合测试集):
  1. Noiseprint++ 输出 3 通道噪声指纹
  2. 指纹强度 I(x) = |fp| 通道均值
  3. 局部一致性 A(x) = |I(x) - GaussianBlur(I, 31)|
  4. 归一化后阈值 0.5 + 连通域去噪 (>=64 px)
输出: results/r77_noiseprint_baseline.json
"""
import sys, os, json, time
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
TRU = f'{BASE}/data/trufor/TruFor-main/TruFor_train_test'
sys.path.insert(0, TRU)
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch
from lib.models.DnCNN import make_net
from r34_distill_matrix import eval_mask

N_TE = 400
OUT = f'{BASE}/results/r77_noiseprint_baseline.json'


def build():
    net = make_net(3, kernels=[3] * 17, features=[64] * 16 + [1],
                   bns=[False] + [True] * 15 + [False],
                   acts=['relu'] * 16 + ['linear'], dilats=[1] * 17,
                   bn_momentum=0.1, padding=1)
    w = f'{TRU}/pretrained_models/noiseprint++/noiseprint++.th'
    net.load_state_dict(torch.load(w, map_location='cpu')['network'])
    net.eval()
    return net


def anomaly_map(net, img_u8):
    """Noiseprint++ 指纹 -> 局部不一致异常图"""
    x = torch.from_numpy(img_u8.transpose(2, 0, 1)).float().unsqueeze(0) / 255.
    with torch.no_grad():
        fp = net(x)[0].numpy()                     # (1,H,W)
    inten = np.abs(fp).mean(0).astype(np.float32)   # 指纹强度
    # 局部一致性: 与 31x31 平滑的差
    sm = cv2.GaussianBlur(inten, (31, 31), 0)
    anom = np.abs(inten - sm)
    lo, hi = np.percentile(anom, 5), np.percentile(anom, 99.5)
    anom = np.clip((anom - lo) / max(1e-6, hi - lo), 0, 1)
    return anom


def to_mask(anom, thr=0.5, min_area=64):
    b = (anom > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    out = np.zeros_like(b)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


def main():
    net = build()
    z = np.load(f'{BASE}/results/r70_data_d25.npz')
    imgs, gts = z['img'][-N_TE:], z['gt'][-N_TE:]
    print(f'n={len(imgs)}', flush=True)

    preds = []
    t0 = time.time()
    for i, im in enumerate(imgs):
        anom = anomaly_map(net, im)
        preds.append(to_mask(anom))
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(imgs)} ({time.time()-t0:.0f}s)', flush=True)
    preds = np.array(preds)

    st = eval_mask(preds, gts)
    st.pop('per_iou', None)
    # 阈值敏感性 (报告两个阈值, 不以测试集选优)
    alt = {}
    for thr in (0.4, 0.6):
        p2 = np.array([to_mask(anomaly_map(net, im), thr=thr) for im in imgs[:120]])
        s2 = eval_mask(p2, gts[:120]); s2.pop('per_iou', None)
        alt[f'thr{thr}'] = {k: round(v, 4) for k, v in s2.items()}

    out = {'meta': dict(method='Noiseprint++ (TruFor front-end), unsupervised',
                        n=len(imgs), thr=0.5, min_area=64,
                        note='TruFor official weights are epoch-2 untrained; other passive '
                             'weights are on Google Drive and unreachable here.'),
           'result': {k: round(v, 4) for k, v in st.items()},
           'threshold_sensitivity_n120': alt,
           'compare': dict(ours_free_labels=0.9491, ours_gt_upper=0.9872,
                           editguard_same_protocol=0.9272)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print('Noiseprint++ IoU =', round(st['iou'], 4), '| cov', round(st['cov50'], 3),
          '| area', round(st['area_ratio'], 3))
    print('->', OUT)


if __name__ == '__main__':
    main()
