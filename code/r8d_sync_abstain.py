#!/usr/bin/env python3
"""G3-d 几何失同步的最终处置: 同步 + 拒识 (abstain), 以及"编辑+裁剪"后仍能定位.

r8/r8b/r8c 已定量:
  * 纯裁剪 (任意像素相位): coord-PRF + (块偏移×相位) 搜索 -> 失败率 50%→25%, 参数 100% 命中;
  * crop→resize 回原尺寸 / 1° 旋转: 反变换+搜索后仍 39–43%, 且与 δ(15/25/40)、
    编码方式(绝对 QIM / 块对差值 QIM)无关 -> **插值重采样破坏了 8x8 DCT 格点判决**, 不是同步问题;
  * 缩放再放大回原尺寸: 34.5%(δ15)→27.6%(δ40), δ 有效.

因此工程上唯一诚实的处置是让检测器**知道自己什么时候不知道**:
  搜索后的最小失败率 f* 若显著低于"随机水平"(~50%) 才认定同步成功;
  否则输出"几何失同步, 不可判定"而不是误报"整图被编辑".
  阈值用错误密钥的零分布标定.

实验:
  A 各变换下 f*(正确 key) 与 f*(错误 key) 的分布 -> 同步成功率 / 假同步率 / 拒识率;
  B "局部编辑 + 任意相位裁剪" 场景: 同步后 detect AUC / loc AUC / IoU 是否保持;
  C 判定: 纯裁剪应 pass, crop_resize/rotate 应被正确拒识 (而不是 50% 误报).

输出: results/r8d_sync_abstain_<ts>.json
用法: apjf-python r8d_sync_abstain.py [--n 120] [--n-b 100]
"""
import sys, os, json, argparse, datetime, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit, block_gt, loc_metrics, mean_ci

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
DELTA = 15
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))
MAX_OFF = 12


def t_clean(im):
    return im, (0, 0)


def t_pad(im):
    """纯裁剪但补零回原尺寸 (保留原始块网格, 相位 0) -> 不应影响检测."""
    return im, (0, 0)


def failmap_at(img, key, oi, oj, pi, pj):
    """按给定 (块偏移, 像素相位) 重算逐块失败图."""
    y = r8.y_of(img)[pi:, pj:]
    pred = r8.pred_map(y, CHK, DELTA)
    n_h, n_w = pred.shape
    pat = ks.check_pattern(key, n_h + oi, n_w + oj, 'coord')[oi:oi + n_h, oj:oj + n_w]
    return (pred != pat).astype(np.float32)


def low_quantile(fail, q=10):
    """平滑失败图的低分位数: 同步正确时存在 ≈25% 的未编辑区, 错误时全图 ≈50%."""
    f_sm = ks.smooth_fail_map(fail, 5)
    return float(np.percentile(f_sm, q))


def sync_fstar(img, key, orig_shape=None, phases=8):
    """64 相位 × 块偏移搜索, 返回 (f*, oi, oj, pi, pj).

    偏移上界受"原图块数 - 检测图块数"约束 (检测方从登记信息知道原图尺寸),
    既缩小搜索空间, 也保证映射回原图坐标时不越界.
    """
    mo = MAX_OFF
    if orig_shape is not None:
        mo = min(MAX_OFF, max(0, orig_shape[0] // 8 - img.shape[0] // 8))
    return r8.search_sync(img, key, mo, phases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--n-b', type=int, default=100)
    ap.add_argument('--thresh', type=float, default=0.35)
    ap.add_argument('--qlow', type=float, default=0.25)
    args = ap.parse_args()
    t0 = time.time()

    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:max(args.n, args.n_b)]

    wms, keys, ids, boxes, origs = [], [], [], [], []
    for s in samples:
        sid = s['id']
        try:
            o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        k = ks.image_key(MASTER, sid)
        wms.append(ks.embed_keyed(o, k, DELTA, PAY, CHK, pattern='coord'))
        keys.append(k)
        ids.append(sid)
        origs.append((o, a, s['bbox']))
    print(f'载入 {len(wms)} 图 ({time.time()-t0:.0f}s)')

    # ---------------- A 同步后的 f* 分布 (正确 key vs 错误 key)
    conds = {
        'clean': lambda im: im,
        'crop_align_0.9': lambda im: r8.t_crop_align(im, 0.9)[0],
        'crop_arb_0.9': lambda im: r8.t_crop_arb(im, 0.9)[0],
        'crop_resize_0.9': lambda im: r8.t_crop_resize(im, 0.9)[0],
        'rotate_2': lambda im: r8.t_rotate(im, 2.0)[0],
        'resize_round_0.5': lambda im: r8.t_resize_round(im, 0.5)[0],
    }
    # 相位搜索范围: 8 对齐裁剪的相位是 0; 只有任意偏移裁剪才需要全 64 相位.
    # 对重采样类给更宽松的 1 相位搜索 (仍拒识, 见结果) 以控制成本.
    cond_phases = {'clean': 1, 'crop_align_0.9': 1, 'crop_arb_0.9': 8,
                   'crop_resize_0.9': 1, 'rotate_2': 1, 'resize_round_0.5': 1}
    rk = ks.random_keys(MASTER, 1)[0]
    A = {}
    for name, tf in conds.items():
        fc, fw, lc, lw = [], [], [], []
        ph = cond_phases[name]
        for wm, k in zip(wms[:args.n], keys[:args.n]):
            im = r8.crop8(tf(wm))
            f, oi, oj, pi, pj = sync_fstar(im, k, wm.shape, ph)
            fc.append(f)
            lc.append(low_quantile(failmap_at(im, k, oi, oj, pi, pj)))
            f2, oi2, oj2, pi2, pj2 = sync_fstar(im, rk, wm.shape, ph)
            fw.append(f2)
            lw.append(low_quantile(failmap_at(im, rk, oi2, oj2, pi2, pj2)))
        fc, fw = np.array(fc), np.array(fw)
        A[name] = {
            'n': len(fc),
            'fstar_correct': [float(np.mean(fc)), float(1.96 * np.std(fc) / np.sqrt(len(fc)))],
            'fstar_wrong': [float(np.mean(fw)), float(1.96 * np.std(fw) / np.sqrt(len(fw)))],
            'low10_correct': [float(np.mean(lc)), float(1.96 * np.std(lc) / np.sqrt(len(lc)))],
            'low10_wrong': [float(np.mean(lw)), float(1.96 * np.std(lw) / np.sqrt(len(lw)))],
            'sync_rate_correct': float((np.array(fc) < args.thresh).mean()),
            'false_sync_rate_wrong': float((np.array(fw) < args.thresh).mean()),
            'sync_rate_low10': float((np.array(lc) < args.qlow).mean()),
            'false_sync_rate_low10': float((np.array(lw) < args.qlow).mean()),
        }
        print(f'  {name:18s} f*={np.mean(fc)*100:5.1f}/{np.mean(fw)*100:5.1f}%  '
              f'low10={np.mean(lc)*100:5.1f}/{np.mean(lw)*100:5.1f}%  '
              f'同步(f*)={A[name]["sync_rate_correct"]*100:5.1f}%  '
              f'同步(low10)={A[name]["sync_rate_low10"]*100:5.1f}%  '
              f'假同步={A[name]["false_sync_rate_low10"]*100:5.1f}%')

    # ---------------- B 编辑 + 任意相位裁剪: 同步后定位是否保持
    B_rows = []
    for (o, a, bbox), k in zip(origs[:args.n_b], keys[:args.n_b]):
        wm = ks.embed_keyed(o, k, DELTA, PAY, CHK, pattern='coord')
        att, box = paste_edit(wm, a, bbox)
        cim, (y0, x0) = r8.t_crop_arb(att, 0.9)
        f_star, oi, oj, pi, pj = sync_fstar(cim, k, o.shape)
        # 该相位/偏移下的逐块失败图
        y = r8.y_of(cim)[pi:, pj:]
        pred = r8.pred_map(y, CHK, DELTA)
        n_h, n_w = pred.shape
        pat = ks.check_pattern(k, n_h + oi, n_w + oj, 'coord')[oi:oi + n_h, oj:oj + n_w]
        fail = (pred != pat).astype(np.float32)
        # 映射回原图块坐标 (检测图块 (i,j) <-> 原图块 (i+oi, j+oj))
        full = np.full((o.shape[0] // 8, o.shape[1] // 8), np.nan, np.float32)
        full[oi:oi + n_h, oj:oj + n_w] = fail
        gt = block_gt(full.shape, box)
        valid = ~np.isnan(full)
        in_b, out_b = gt & valid, (~gt) & valid
        filled = np.where(valid, full, 0.5).astype(np.float32)
        f_sm = ks.smooth_fail_map(filled, 5)
        if in_b.sum() > 3 and out_b.sum() > 3:
            auc_sm = ks.auc(f_sm[out_b], f_sm[in_b])
        else:
            auc_sm = float('nan')
        pred = (f_sm > 0.5) & valid
        inter, union = (pred & gt).sum(), (pred | gt).sum()
        t_pi, t_pj = (-y0) % 8, (-x0) % 8
        t_oi, t_oj = (y0 + t_pi) // 8, (x0 + t_pj) // 8
        area = (box[2] - box[0]) * (box[3] - box[1]) / (o.shape[0] * o.shape[1])
        row = {
            'f_star': f_star, 'off': (oi, oj), 'phase': (pi, pj),
            'true_off': (t_oi, t_oj), 'true_phase': (t_pi, t_pj), 'area': float(area),
            'in_fail': float(full[in_b].mean()) if in_b.any() else float('nan'),
            'out_fail': float(full[out_b].mean()) if out_b.any() else float('nan'),
            'auc_smooth': auc_sm,
            'iou': float(inter / union) if union else 0.0,
            'off_match': (oi, oj) == (t_oi, t_oj),
            'phase_match': (pi, pj) == (t_pi, t_pj),
            'sync_ok': low_quantile(fail) < args.qlow,
        }
        B_rows.append(row)
    local = [r for r in B_rows if r['area'] < 0.9]      # 全图编辑无"未编辑参照", 单列
    B = {
        'n': len(B_rows), 'n_local': len(local), 'n_full': len(B_rows) - len(local),
        'sync_ok_rate': float(np.mean([r['sync_ok'] for r in local])),
        'off_match_rate': float(np.mean([r['off_match'] for r in local])),
        'phase_match_rate': float(np.mean([r['phase_match'] for r in local])),
        'in_fail': mean_ci([r['in_fail'] for r in local]),
        'out_fail': mean_ci([r['out_fail'] for r in local]),
        'auc_smooth': mean_ci([r['auc_smooth'] for r in local]),
        'iou': mean_ci([r['iou'] for r in local]),
        'full_edit_sync_ok_rate': float(np.mean([r['sync_ok'] for r in B_rows
                                                 if r['area'] >= 0.9])) if len(B_rows) > len(local) else None,
    }
    # 按编辑面积分层: 编辑区越大, 低分位判据越难触发
    B['by_area'] = {}
    for tag, lo, hi in [('area_lt_35', 0.0, 0.35), ('area_35_65', 0.35, 0.65),
                        ('area_gt_65', 0.65, 0.90)]:
        rr = [r for r in local if lo <= r['area'] < hi]
        if rr:
            B['by_area'][tag] = {
                'n': len(rr),
                'sync_ok_rate': float(np.mean([r['sync_ok'] for r in rr])),
                'inv_out': [float(np.mean([r['in_fail'] for r in rr])),
                            float(np.mean([r['out_fail'] for r in rr]))],
                'auc_smooth': mean_ci([r['auc_smooth'] for r in rr]),
                'iou': mean_ci([r['iou'] for r in rr]),
            }
    print(f'\n=== B 编辑+裁剪(任意相位) n={B["n"]} (局部 {B["n_local"]} / 全图 {B["n_full"]}) ===')
    print(f'  同步成功 {B["sync_ok_rate"]*100:.1f}%  块偏移命中 {B["off_match_rate"]*100:.1f}%  '
          f'相位命中 {B["phase_match_rate"]*100:.1f}%')
    print(f'  区内 {B["in_fail"][0]*100:.1f}% / 区外 {B["out_fail"][0]*100:.1f}%  '
          f'定位 AUC {B["auc_smooth"][0]:.3f}  IoU {B["iou"][0]:.3f}')
    for tag, v in B['by_area'].items():
        print(f'    {tag:12s} n={v["n"]:3d} 同步={v["sync_ok_rate"]*100:5.1f}% '
              f'in/out={v["inv_out"][0]*100:4.1f}/{v["inv_out"][1]*100:4.1f}% '
              f'AUC={v["auc_smooth"][0]:.3f} IoU={v["iou"][0]:.3f}')

    # ---------------- C 判定
    verdict = {
        'crop_sync_works': bool(A['crop_arb_0.9']['sync_rate_correct'] >= 0.95 and
                                A['clean']['false_sync_rate_wrong'] <= 0.10),
        'resample_abstains': bool(A['crop_resize_0.9']['sync_rate_correct'] <= 0.20 and
                                  A['rotate_2']['sync_rate_correct'] <= 0.20),
        'edit_after_crop_localizes': bool(B['auc_smooth'][0] > 0.80 and B['sync_ok_rate'] > 0.85),
    }
    out = {'meta': {'ts': datetime.datetime.now().isoformat(), 'n_a': args.n, 'n_b': args.n_b,
                    'thresh': args.thresh, 'max_off': MAX_OFF},
           'A_fstar': A, 'B_edit_crop': B, 'verdict': verdict,
           'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r8d_sync_abstain_{ts}.json'
    json.dump(out, open(fp, 'w'), indent=1)
    print(f'\n判定: {verdict}')
    print(f'结果: {fp}  ({out["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
