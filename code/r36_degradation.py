#!/usr/bin/env python3
"""r36: 退化鲁棒性 — 学生 vs 教师4 谁稳.

退化作用于编辑图; GT 不变. 教师在几何退化下可能拒识(报拒识率; IoU 按拒识=0保守计).
输出: results/r36_degradation.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs
from r34_distill_matrix import (SmallUNet, eval_mask, alarm_from_fail,
                                DELTA, PAY, CHK, MASTER, K_LOC, T_ALARM)
from r10_specificity import jpeg, noise, resize_round, rotate, sharpen
from r13_realistic_protocol import *  # noqa (若有 feather 等)

N = 200
# 退化定义: name -> (fn, is_geometric_for_abstain)
DEGS = {
    'clean':      (lambda im: im, False),
    'jpeg85':     (lambda im: jpeg(im, 85), False),
    'jpeg75':     (lambda im: jpeg(im, 75), False),
    'jpeg50':     (lambda im: jpeg(im, 50), False),
    'noise2':     (lambda im: noise(im, 2), False),
    'sharpen':    (lambda im: sharpen(im), False),
    'resize0.8':  (lambda im: resize_round(im, 0.8), True),
    'rotate1':    (lambda im: rotate(im, 1.0), True),
}


def feather(im, r):
    """盒式近似羽化 (与论文 r=4 口径近似)."""
    k = 2 * r + 1
    return cv2.blur(im, (k, k))


DEGS['feather4'] = (lambda im: feather(im, 4), False)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wpath = f'{BASE}/results/r34_student_alarm4.pth'
    assert os.path.exists(wpath), 'run r35 first (weights)'
    m = SmallUNet().to(device)
    m.load_state_dict(torch.load(wpath, map_location=device))

    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    img, gt, a4 = z['img'][1000:1200], z['gt'][1000:1200], z['alarm4'][1000:1200]
    # 教师的拒识判据: low10 (平滑失效图低10%分位 < 0.20 = 可判定)
    # 这里需要失效图 -> 从 alarm 反推不行, 需重算. 简化: 用 alarm4 面积骤变+同步判据
    # -> 重算失效图代价高; 采用 r30 geo 的口径: 对几何退化直接按论文结论标注拒识,
    #    实测判据: 重新计算 low10 (需要 key/fail). 为准确起见重算.

    AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
          '/results/attackset_20261001_222447')
    meta = json.load(open(f'{AS}/meta.json'))
    # 取测试段样本 id (与 r34 数据顺序一致: 前1200个可用样本的后200)
    ids = []
    ok = 0
    for s_ in meta['samples']:
        sid = s_['id']
        if os.path.exists(f'{AS}/images/orig_{sid}.png') and os.path.exists(f'{AS}/images/att_{sid}.png'):
            ids.append(sid)
            ok += 1
        if ok >= 1200:
            break
    ids = ids[1000:1200]
    assert len(ids) == 200

    out = {'meta': dict(n=N, degradations=list(DEGS.keys()),
                        student='r34 SmallUNet alarm4 (COCO-trained)',
                        teacher='WMFE alarm4 + low10<0.20 可判定'),
           'results': {}}
    t0 = time.time()
    for dname, (fn, is_geo) in DEGS.items():
        eds, st_preds, t_preds, t_abstain = [], [], [], []
        for i, sid in enumerate(ids):
            edited = img[i]  # 已嵌水印+编辑
            deg = fn(edited)
            # 学生
            eds.append(deg)
            # 教师: 重算失效图 (key 一致)
            key = ks.image_key(MASTER, sid)
            k4 = layer_key(key, 'bs4')
            f4 = fail_bs(deg, k4, 4, DELTA, (1, 1))
            # 论文判据 (r8d 权威口径): smooth K=5 + np.percentile(10) < 0.20
            f8 = ks.fail_map(deg, key, DELTA, CHK, 'coord')
            sm8 = ks.smooth_fail_map(f8, 5)
            low10 = float(np.percentile(sm8, 10))
            abstain = low10 >= 0.25  # r8d 实验口径 (论文表3 同步率100% 对应)
            t_abstain.append(abstain)
            if abstain:
                t_preds.append(np.zeros_like(a4[i]))  # 拒识=无告警(保守)
            else:
                l10_4 = float(np.percentile(cv2.blur(f4, (5, 5)), 10))
                if l10_4 < 0.25:
                    t_preds.append(alarm_from_fail(f4, 4, 5, 32))
                else:
                    t_preds.append(alarm_from_fail(f8, 8, K_LOC, 8))  # 回退 8×8
        eds = np.array(eds)
        # 学生批量推理
        m.eval(); sp = []
        with torch.no_grad():
            for j in range(0, len(eds), 16):
                xb = torch.from_numpy(eds[j:j+16].transpose(0, 3, 1, 2)).float().to(device) / 255.
                sp.append(torch.sigmoid(m(xb)).cpu().numpy())
        sp = (np.concatenate(sp)[:, 0] > 0.5).astype(np.uint8)
        gts = gt
        ss = eval_mask(sp, gts); ss.pop('per_iou')
        ts = eval_mask(np.array(t_preds), gts); ts.pop('per_iou')
        # 教师可判定子集 IoU (排除拒识)
        keep = [i for i, a in enumerate(t_abstain) if not a]
        if keep:
            ts_det = eval_mask(np.array(t_preds)[keep], gts[keep]); ts_det.pop('per_iou')
        else:
            ts_det = {'iou': float('nan'), 'cov50': float('nan'), 'area_ratio': float('nan')}
        out['results'][dname] = {
            'student': ss, 'teacher4_conservative': ts,
            'teacher4_detectable_subset': ts_det,
            'teacher_abstain_rate': float(np.mean(t_abstain))}
        print(f'[{dname:10s}] student IoU={ss["iou"]:.3f} | teacher IoU={ts["iou"]:.3f} '
              f'(abstain {np.mean(t_abstain):.0%}, 可判定子集 {ts_det["iou"]:.3f})', flush=True)
    json.dump(out, open(f'{BASE}/results/r36_degradation.json', 'w'), indent=1)
    print(f'-> results/r36_degradation.json ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
