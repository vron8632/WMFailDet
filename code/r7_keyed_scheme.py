#!/usr/bin/env python3
"""G0: 密钥化方案 (code/keyed_scheme.py) 的判定实验.

三个问题, 全部在同一批数据上直接对照"旧方案 (固定 seed=42)"与"密钥化方案":

  A 密钥化不改变结论 (等价性复现)
    A1 全局重绘 (r3 协议): 3 变体 × 14 唯一图, 图像级检测 AUC;
    A2 局部编辑 (r6 协议): attackset 1200 全量, 图像级检测 AUC / 定位 AUC / IoU@0.5,
       old vs keyed(matrix) vs keyed(coord) 三条件逐图配对;
    A3 结论: |Δ| 应落在统计噪声内 (同一图像内容, 只换期望序列的派生方式).

  B 错误密钥 (无密钥第三方)
    20 个与 owner 无关的随机密钥对同一批图做检测: 失败率应 ≈ 50%,
    clean 与编辑后不可分 (AUC ≈ 0.5) -> 密钥是检测的必要条件.

  C re-embedding 洗白攻击 (密钥必要性的直接证据)
    C1 旧方案: 攻击者知道算法 + 固定 seed=42, 编辑后重新嵌入 -> 校验位被刷回,
       编辑区失败率回落到 ~25% (定位 AUC 跌到 ~0.5) = 篡改被掩盖;
    C2 密钥化: 攻击者只能用 ①旧公开算法+seed=42 或 ②随机密钥 重嵌入 ->
       正确密钥下失败率仍 ≈ 50%, 检测器继续报警 (detect AUC 保持高).
       代价是定位被破坏 -> re-embedding 从"欺骗"降级为"可检测的 DoS".

输出: results/r7_keyed_scheme_<ts>.json
      data/failmaps_r7_<ts>.npz  (A2 keyed coord 的逐块失败图 + GT, 供 G4 调优复用)
用法: apjf-python r7_keyed_scheme.py [--n 1200] [--n-wrongkey 200] [--n-reembed 200]
"""
import sys, os, json, argparse, datetime, glob, time

OLD_CODE = '/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/code'
sys.path.insert(0, OLD_CODE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from PIL import Image

_o = os.makedirs
def _skip(name, *a, **k):
    return None if str(name).startswith('/media/oyp/数据') else _o(name, *a, **k)
os.makedirs = _skip
try:
    import run_signal_ra as rs
finally:
    os.makedirs = _o

import keyed_scheme as ks

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
HIRES = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
         '/results/attackset_hires_sample')
CACHE = f'{BASE}/data/r1_attack_cache'
DELTA, MSGLEN = 15, 256
PAY, CHK = (1, 0), (1, 1)
VARIANTS = {'V0_current': ((4, 1), (3, 2)), 'V1_lowfreq1': ((1, 0), (1, 1)),
            'V2_lowfreq2': ((2, 1), (1, 0))}
MASTER = b'\x9a' + bytes(range(31))     # 演示用 master key (正式部署由 owner 保管)


# ------------------------------------------------------------------ 旧方案 (固定 seed=42)
def old_embed(img, msg, coef, delta=DELTA):
    rs.PAY_COEF, rs.CHECK_COEF = coef
    return rs.embed_signal(img, msg, delta)


def old_fail(img, coef, delta=DELTA):
    H, W, n_h, n_w = ks.block_grid(img.shape)
    return ks.fail_map_exp(img, rs.check_bits(H, W, n_h, n_w), delta, coef[1])


# ------------------------------------------------------------------ 通用
def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def paste_edit(wm, att, bbox01):
    """bbox 内取攻击图, bbox 外保持水印图 (与 eval_attackset/r6 一致)."""
    H, W = wm.shape[:2]
    x1, y1, x2, y2 = bbox01
    bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
    bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
    out = wm.copy()
    out[by1:by2, bx1:bx2] = att[by1:by2, bx1:bx2]
    return out, (by1, bx1, by2, bx2)


def block_gt(shape, box):
    by1, bx1, by2, bx2 = box
    n_h, n_w = int(shape[0]), int(shape[1])      # shape = 块级 (n_h, n_w)
    gt = np.zeros((n_h, n_w), bool)
    gt[by1 // 8:min((by2 + 7) // 8, n_h), bx1 // 8:min((bx2 + 7) // 8, n_w)] = True
    return gt


def loc_metrics(fail, box, k=5):
    """单图定位指标: 原始/平滑 AUC, IoU@0.5.

    AUC 用**块级** GT (块内 bbox 占比>0), IoU 用**像素级** GT + 8x8 块重复,
    两者都与 r6_scaleup.py 逐行同口径, 因此 old 条件可直接对齐 r6 数字.
    """
    n_h, n_w = fail.shape
    H, W = n_h * 8, n_w * 8
    by1, bx1, by2, bx2 = box
    gt_b = block_gt(fail.shape, box)
    in_b, out_b = gt_b, ~gt_b
    f_sm = ks.smooth_fail_map(fail, k)
    if in_b.any() and out_b.any():
        a_raw = ks.auc(fail[out_b], fail[in_b])
        a_sm = ks.auc(f_sm[out_b], f_sm[in_b])
    else:
        a_raw = a_sm = float('nan')
    pred_px = np.repeat(np.repeat(f_sm > 0.5, 8, 0), 8, 1)
    gt_px = np.zeros((H, W), bool)
    gt_px[by1:min(by2, H), bx1:min(bx2, W)] = True
    inter = (pred_px & gt_px).sum()
    union = (pred_px | gt_px).sum()
    return {
        'auc_raw': a_raw, 'auc_smooth': a_sm,
        'iou': float(inter / union) if union else 0.0,
        'in_fail': float(fail[in_b].mean()) if in_b.any() else float('nan'),
        'out_fail': float(fail[out_b].mean()) if out_b.any() else float('nan'),
    }


def ci95(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return float(1.96 * v.std() / np.sqrt(len(v))) if len(v) > 1 else float('nan')


def auc_ci(neg, pos, n_boot=1000, seed=0):
    """检测 AUC 的 bootstrap 95% CI (重采样图, 保配对结构)."""
    neg = np.asarray(neg, float)
    pos = np.asarray(pos, float)
    if len(neg) < 3 or len(pos) < 3:
        return [float('nan'), float('nan')]
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.randint(0, len(neg), len(neg))
        j = rng.randint(0, len(pos), len(pos))
        vals.append(ks.auc(neg[i], pos[j]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def mean_ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [float(v.mean()), ci95(v)] if len(v) else [float('nan'), float('nan')]


# ------------------------------------------------------------------ A1 全局重绘
def exp_a1():
    msg42 = np.random.RandomState(42).randint(0, 2, MSGLEN).astype(np.uint8)
    out = {}
    for v, coef in VARIANTS.items():
        rows, seen = [], set()
        for f in sorted(glob.glob(f'{CACHE}/{v}/*.png')):
            base = os.path.basename(f)[:-4].rsplit('_', 1)[0]
            if base in seen:
                continue
            seen.add(base)
            op = f'{HIRES}/images/{base}_orig.png'
            if not os.path.exists(op):
                continue
            o = crop8(np.array(Image.open(op).convert('RGB')))
            att = crop8(np.array(Image.open(f).convert('RGB')))
            key = ks.image_key(MASTER, base)

            wm_old = old_embed(o, msg42, coef)
            wm_m = ks.embed_keyed(o, key, DELTA, coef[0], coef[1], pattern='matrix')
            wm_c = ks.embed_keyed(o, key, DELTA, coef[0], coef[1], pattern='coord')

            rows.append({
                'img': base,
                'old_clean': float(old_fail(wm_old, coef).mean()),
                'old_att': float(old_fail(att, coef).mean()),
                'mtx_clean': float(ks.fail_map(wm_m, key, DELTA, coef[1], 'matrix').mean()),
                'mtx_att': float(ks.fail_map(att, key, DELTA, coef[1], 'matrix').mean()),
                'crd_clean': float(ks.fail_map(wm_c, key, DELTA, coef[1], 'coord').mean()),
                'crd_att': float(ks.fail_map(att, key, DELTA, coef[1], 'coord').mean()),
            })
        conds = {}
        for c in ['old', 'mtx', 'crd']:
            cl = [r[f'{c}_clean'] for r in rows]
            at = [r[f'{c}_att'] for r in rows]
            conds[c] = {'clean_fail': mean_ci(cl), 'att_fail': mean_ci(at),
                        'detect_auc': ks.auc(cl, at), 'n': len(rows)}
        # keyed 与 old 的逐图差值 (应≈0)
        conds['delta_crd_minus_old'] = {
            'clean': mean_ci([r['crd_clean'] - r['old_clean'] for r in rows]),
            'att': mean_ci([r['crd_att'] - r['old_att'] for r in rows]),
            'max_abs_clean': float(max(abs(r['crd_clean'] - r['old_clean']) for r in rows)),
            'max_abs_att': float(max(abs(r['crd_att'] - r['old_att']) for r in rows)),
        }
        out[v] = {'n': len(rows), 'coef': f'{coef[0]}/{coef[1]}', 'conds': conds, 'rows': rows}
        print(f'  A1 {v:12s} n={len(rows)}  old: {conds["old"]["clean_fail"][0]*100:.1f}%->'
              f'{conds["old"]["att_fail"][0]*100:.1f}% AUC={conds["old"]["detect_auc"]:.3f} | '
              f'keyed(coord): {conds["crd"]["clean_fail"][0]*100:.1f}%->'
              f'{conds["crd"]["att_fail"][0]*100:.1f}% AUC={conds["crd"]["detect_auc"]:.3f}')
    return out


# ------------------------------------------------------------------ A2 局部编辑
def exp_a2(n, save_npz=True):
    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:n]
    msg42 = np.random.RandomState(42).randint(0, 2, MSGLEN).astype(np.uint8)
    rows, t0 = [], time.time()
    fail_store = {'fail': [], 'gt': [], 'ids': []}
    for idx, s in enumerate(samples):
        sid = s['id']
        try:
            o = crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        key = ks.image_key(MASTER, sid)
        wm_old = old_embed(o, msg42, (PAY, CHK))
        wm_m = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='matrix')
        wm_c = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        att_old, box = paste_edit(wm_old, a, s['bbox'])
        att_m, _ = paste_edit(wm_m, a, s['bbox'])
        att_c, _ = paste_edit(wm_c, a, s['bbox'])

        f_old_c, f_old_a = old_fail(wm_old, (PAY, CHK)), old_fail(att_old, (PAY, CHK))
        f_mt_c, f_mt_a = (ks.fail_map(wm_m, key, DELTA, CHK, 'matrix'),
                          ks.fail_map(att_m, key, DELTA, CHK, 'matrix'))
        f_cr_c, f_cr_a = (ks.fail_map(wm_c, key, DELTA, CHK, 'coord'),
                          ks.fail_map(att_c, key, DELTA, CHK, 'coord'))
        m_old, m_c = loc_metrics(f_old_a, box), loc_metrics(f_cr_a, box)
        m_m = loc_metrics(f_mt_a, box)
        area = (s['bbox'][2] - s['bbox'][0]) * (s['bbox'][3] - s['bbox'][1])
        full = block_gt(f_cr_a.shape, box).mean() > 0.95
        tier = 'full' if full else ('small' if area < 0.10 else
                                    ('mid' if area < 0.30 else 'large'))
        ins = s['instruction']
        etype = ('replace' if ('替换' in ins or '换成' in ins or '换上' in ins) else
                 ('remove' if ('删除' in ins or '移除' in ins or '去掉' in ins) else 'other'))
        rows.append({
            'id': sid, 'area': float(area), 'tier': tier, 'etype': etype,
            'old_clean': float(f_old_c.mean()), 'old_att': float(f_old_a.mean()),
            'mtx_clean': float(f_mt_c.mean()), 'mtx_att': float(f_mt_a.mean()),
            'crd_clean': float(f_cr_c.mean()), 'crd_att': float(f_cr_a.mean()),
            'old_loc': m_old, 'mtx_loc': m_m, 'crd_loc': m_c,
        })
        fail_store['fail'].append(f_cr_a.astype(np.float32))
        fail_store['gt'].append(block_gt(f_cr_a.shape, box))
        fail_store['ids'].append(sid)
        if (idx + 1) % 300 == 0:
            el = time.time() - t0
            print(f'  A2 [{idx+1}/{len(samples)}] {el:.0f}s 预计剩余 {el/(idx+1)*(len(samples)-idx-1):.0f}s',
                  flush=True)

    def summarize(cond):
        cl = [r[f'{cond}_clean'] for r in rows]
        at = [r[f'{cond}_att'] for r in rows]
        loc = [r[f'{cond}_loc'] for r in rows]
        return {
            'n': len(rows),
            'clean_fail': mean_ci(cl), 'att_fail': mean_ci(at),
            'detect_auc_paired': ks.auc(cl, at),
            'detect_auc_boot95': auc_ci(cl, at),
            'loc_auc_raw': mean_ci([x['auc_raw'] for x in loc]),
            'loc_auc_smooth': mean_ci([x['auc_smooth'] for x in loc]),
            'iou': mean_ci([x['iou'] for x in loc]),
            'in_fail': mean_ci([x['in_fail'] for x in loc]),
            'out_fail': mean_ci([x['out_fail'] for x in loc]),
        }

    res = {'n': len(rows), 'old': summarize('old'), 'keyed_matrix': summarize('mtx'),
           'keyed_coord': summarize('crd')}
    res['paired_delta_crd_minus_old'] = {
        'detect_auc': res['keyed_coord']['detect_auc_paired'] - res['old']['detect_auc_paired'],
        'detect_auc_matrix': res['keyed_matrix']['detect_auc_paired'] - res['old']['detect_auc_paired'],
        'clean_fail': mean_ci([r['crd_clean'] - r['old_clean'] for r in rows]),
        'att_fail': mean_ci([r['crd_att'] - r['old_att'] for r in rows]),
        'loc_auc_smooth': mean_ci([r['crd_loc']['auc_smooth'] - r['old_loc']['auc_smooth']
                                   for r in rows]),
        'iou': mean_ci([r['crd_loc']['iou'] - r['old_loc']['iou'] for r in rows]),
        'note': 'crd 与 old 的 pattern 0/1 平衡性在每图上有独立方差, 检测 AUC 是跨图'
                '失败率排序量, 因此其差异主要来自 pattern 采样噪声 (见 keyed_matrix 对照).',
    }
    # 分层
    res['tiers'] = {}
    for tier in ['small', 'mid', 'large', 'full']:
        rr = [r for r in rows if r['tier'] == tier]
        if not rr:
            continue
        res['tiers'][tier] = {
            'n': len(rr),
            'old': {'in_fail': mean_ci([r['old_loc']['in_fail'] for r in rr]),
                    'out_fail': mean_ci([r['old_loc']['out_fail'] for r in rr]),
                    'loc_auc_smooth': mean_ci([r['old_loc']['auc_smooth'] for r in rr]),
                    'iou': mean_ci([r['old_loc']['iou'] for r in rr])},
            'crd': {'in_fail': mean_ci([r['crd_loc']['in_fail'] for r in rr]),
                    'out_fail': mean_ci([r['crd_loc']['out_fail'] for r in rr]),
                    'loc_auc_smooth': mean_ci([r['crd_loc']['auc_smooth'] for r in rr]),
                    'iou': mean_ci([r['crd_loc']['iou'] for r in rr])},
        }
    res['etypes'] = {}
    for et in ['replace', 'remove', 'other']:
        rr = [r for r in rows if r['etype'] == et]
        if not rr:
            continue
        res['etypes'][et] = {
            'n': len(rr),
            'old_loc_auc_smooth': mean_ci([r['old_loc']['auc_smooth'] for r in rr]),
            'crd_loc_auc_smooth': mean_ci([r['crd_loc']['auc_smooth'] for r in rr]),
        }

    npz_path = None
    if save_npz and fail_store['fail']:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        npz_path = f'{BASE}/data/failmaps_r7_{ts}.npz'
        os.makedirs(f'{BASE}/data', exist_ok=True)
        np.savez_compressed(npz_path, fail=np.stack(fail_store['fail']),
                            gt=np.stack(fail_store['gt']),
                            ids=np.array(fail_store['ids']),
                            keys=np.array([ks.image_key(MASTER, i) for i in fail_store['ids']],
                                          dtype='S32'))
        print(f'  失败图已存: {npz_path} ({len(fail_store["fail"])} 张, '
              f'{os.path.getsize(npz_path)/1e6:.1f} MB)')

    print(f'  A2 n={len(rows)}  old  : detect AUC={res["old"]["detect_auc_paired"]:.4f} '
          f'loc AUC={res["old"]["loc_auc_smooth"][0]:.3f} IoU={res["old"]["iou"][0]:.3f}')
    print(f'  A2 n={len(rows)}  keyed-matrix: detect AUC={res["keyed_matrix"]["detect_auc_paired"]:.4f} '
          f'loc AUC={res["keyed_matrix"]["loc_auc_smooth"][0]:.3f} IoU={res["keyed_matrix"]["iou"][0]:.3f}')
    print(f'  A2 n={len(rows)}  keyed-coord : detect AUC={res["keyed_coord"]["detect_auc_paired"]:.4f} '
          f'loc AUC={res["keyed_coord"]["loc_auc_smooth"][0]:.3f} IoU={res["keyed_coord"]["iou"][0]:.3f}')
    return res, npz_path


# ------------------------------------------------------------------ B 错误密钥
def exp_b(n, n_keys=20):
    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:n]
    rkeys = ks.random_keys(MASTER, n_keys)
    rows = []
    for s in samples:
        sid = s['id']
        try:
            o = crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        key = ks.image_key(MASTER, sid)
        wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        att, box = paste_edit(wm, a, s['bbox'])
        f_c = ks.fail_map(wm, key, DELTA, CHK, 'coord')
        f_a = ks.fail_map(att, key, DELTA, CHK, 'coord')
        row = {'id': sid,
               'correct_clean': float(f_c.mean()), 'correct_att': float(f_a.mean())}
        rk_clean, rk_att = [], []
        for rk in rkeys:
            rk_clean.append(float(ks.fail_map(wm, rk, DELTA, CHK, 'coord').mean()))
            rk_att.append(float(ks.fail_map(att, rk, DELTA, CHK, 'coord').mean()))
        row['rand_clean'] = rk_clean
        row['rand_att'] = rk_att
        rows.append(row)

    corr_cl = [r['correct_clean'] for r in rows]
    corr_at = [r['correct_att'] for r in rows]
    per_key_auc, per_key_clean, per_key_att = [], [], []
    for ki in range(n_keys):
        cl = [r['rand_clean'][ki] for r in rows]
        at = [r['rand_att'][ki] for r in rows]
        per_key_auc.append(ks.auc(cl, at))
        per_key_clean.append(float(np.mean(cl)))
        per_key_att.append(float(np.mean(at)))
    res = {
        'n': len(rows), 'n_random_keys': n_keys,
        'correct_key': {'clean_fail': mean_ci(corr_cl), 'att_fail': mean_ci(corr_at),
                        'detect_auc': ks.auc(corr_cl, corr_at)},
        'random_keys': {
            'per_key_clean_fail': mean_ci(per_key_clean),
            'per_key_att_fail': mean_ci(per_key_att),
            'per_key_detect_auc': mean_ci(per_key_auc),
            'per_key_auc_all': per_key_auc,
            'wrong_key_decidability': float(np.mean([abs(x - 0.5) < 0.1 for x in per_key_auc])),
        },
    }
    print(f'  B 正确密钥: clean {res["correct_key"]["clean_fail"][0]*100:.1f}% / '
          f'att {res["correct_key"]["att_fail"][0]*100:.1f}%  AUC={res["correct_key"]["detect_auc"]:.3f}')
    print(f'  B 错误密钥({n_keys}个): clean {res["random_keys"]["per_key_clean_fail"][0]*100:.1f}% / '
          f'att {res["random_keys"]["per_key_att_fail"][0]*100:.1f}%  '
          f'AUC={res["random_keys"]["per_key_detect_auc"][0]:.3f} '
          f'±{res["random_keys"]["per_key_detect_auc"][1]:.3f}')
    return res


# ------------------------------------------------------------------ C re-embedding 洗白
def exp_c(n):
    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:n]
    msg42 = np.random.RandomState(42).randint(0, 2, MSGLEN).astype(np.uint8)
    msg_atk = np.random.RandomState(7).randint(0, 2, MSGLEN).astype(np.uint8)
    rk = ks.random_keys(MASTER, 1)[0]
    rows = []
    for s in samples:
        sid = s['id']
        try:
            o = crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        key = ks.image_key(MASTER, sid)

        # ---- C1 旧方案 (固定 seed=42)
        wm42 = old_embed(o, msg42, (PAY, CHK))
        att42, box = paste_edit(wm42, a, s['bbox'])
        f_clean42 = old_fail(wm42, (PAY, CHK))
        f_att42 = old_fail(att42, (PAY, CHK))
        re42 = old_embed(att42, msg_atk, (PAY, CHK))       # 攻击者: 公开算法 + seed=42
        f_re42 = old_fail(re42, (PAY, CHK))

        # ---- C2 密钥化方案
        wm_k = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        att_k, box = paste_edit(wm_k, a, s['bbox'])
        f_clean_k = ks.fail_map(wm_k, key, DELTA, CHK, 'coord')
        f_att_k = ks.fail_map(att_k, key, DELTA, CHK, 'coord')
        # 攻击者 A: 用旧公开算法 + seed=42 重嵌入
        reA = old_embed(att_k, msg_atk, (PAY, CHK))
        f_reA = ks.fail_map(reA, key, DELTA, CHK, 'coord')
        # 攻击者 B: 用随机密钥 (同一公开算法) 重嵌入
        reB = ks.embed_keyed(att_k, rk, DELTA, PAY, CHK, pattern='coord')
        f_reB = ks.fail_map(reB, key, DELTA, CHK, 'coord')
        # 攻击者 C: 用固定 seed=42 的 matrix 模式重嵌入 (等同于旧算法同分布)
        reC = ks.embed_keyed(att_k, ks.image_key(MASTER, 'attacker-pub'), DELTA, PAY, CHK,
                             pattern='matrix')
        f_reC = ks.fail_map(reC, key, DELTA, CHK, 'coord')

        rows.append({
            'id': sid,
            'old': {'clean': float(f_clean42.mean()), 'att': float(f_att42.mean()),
                    'reembed': float(f_re42.mean()), **loc_metrics(f_att42, box),
                    'reembed_loc': loc_metrics(f_re42, box)},
            'keyed': {'clean': float(f_clean_k.mean()), 'att': float(f_att_k.mean()),
                      **loc_metrics(f_att_k, box),
                      'reA': float(f_reA.mean()), 'reA_loc': loc_metrics(f_reA, box),
                      'reB': float(f_reB.mean()), 'reB_loc': loc_metrics(f_reB, box),
                      'reC': float(f_reC.mean()), 'reC_loc': loc_metrics(f_reC, box)},
        })

    def get(r, path):
        cur = r
        for p in path.split('.'):
            cur = cur[p]
        return cur

    def block(cond_clean, cond_att, loc, re_loc=None):
        cl = [get(r, cond_clean) for r in rows]
        at = [get(r, cond_att) for r in rows]
        d = {'n': len(rows), 'clean_fail': mean_ci(cl), 'att_fail': mean_ci(at),
             'detect_auc': ks.auc(cl, at), 'detect_auc_boot95': auc_ci(cl, at),
             'loc_auc_smooth': mean_ci([get(r, loc)['auc_smooth'] for r in rows]),
             'in_fail': mean_ci([get(r, loc)['in_fail'] for r in rows]),
             'out_fail': mean_ci([get(r, loc)['out_fail'] for r in rows]),
             'iou': mean_ci([get(r, loc)['iou'] for r in rows])}
        if re_loc:
            d['reembed_loc_auc_smooth'] = mean_ci([get(r, re_loc)['auc_smooth'] for r in rows])
            d['reembed_in_fail'] = mean_ci([get(r, re_loc)['in_fail'] for r in rows])
            d['reembed_out_fail'] = mean_ci([get(r, re_loc)['out_fail'] for r in rows])
            d['reembed_iou'] = mean_ci([get(r, re_loc)['iou'] for r in rows])
        return d

    res = {
        'n': len(rows),
        'old_scheme': block('old.clean', 'old.att', 'old', 'old.reembed_loc'),
        'keyed_edited': block('keyed.clean', 'keyed.att', 'keyed'),
    }
    # 密钥化下三种重嵌入攻击的检测结果 (detect AUC 用 clean 参考)
    for tag, key_field, loc_field in [('reA_public_seed42', 'keyed.reA', 'keyed.reA_loc'),
                                      ('reB_random_key', 'keyed.reB', 'keyed.reB_loc'),
                                      ('reC_public_matrix', 'keyed.reC', 'keyed.reC_loc')]:
        cl = [get(r, 'keyed.clean') for r in rows]
        at = [get(r, key_field) for r in rows]
        res[tag] = {
            'n': len(rows),
            'fail_rate': mean_ci(at),
            'detect_auc_vs_clean': ks.auc(cl, at),
            'detect_auc_boot95': auc_ci(cl, at),
            'in_fail': mean_ci([get(r, loc_field)['in_fail'] for r in rows]),
            'out_fail': mean_ci([get(r, loc_field)['out_fail'] for r in rows]),
            'loc_auc_smooth': mean_ci([get(r, loc_field)['auc_smooth'] for r in rows]),
            'edits_masked_rate': float(np.mean(
                [get(r, loc_field)['in_fail'] < 0.35 and get(r, loc_field)['out_fail'] < 0.35
                 for r in rows])),
        }
    # 旧方案: 重嵌入后编辑区失败率回落到基线即"洗白"
    res['old_scheme']['washed_rate'] = float(np.mean(
        [get(r, 'old.reembed_loc')['in_fail'] < 0.35 and
         get(r, 'old.reembed_loc')['out_fail'] < 0.35 for r in rows]))
    res['keyed_edited']['att_mask_rate'] = float(np.mean(
        [get(r, 'keyed')['in_fail'] > 0.4 and get(r, 'keyed')['out_fail'] < 0.35
         for r in rows]))
    print(f'  C 旧方案: 编辑区 {res["old_scheme"]["in_fail"][0]*100:.1f}% / 区外 '
          f'{res["old_scheme"]["out_fail"][0]*100:.1f}%  loc AUC={res["old_scheme"]["loc_auc_smooth"][0]:.3f} '
          f'-> re-embed 后 区外 {res["old_scheme"]["reembed_out_fail"][0]*100:.1f}% '
          f'编辑区 {res["old_scheme"]["reembed_in_fail"][0]*100:.1f}% '
          f'loc AUC={res["old_scheme"]["reembed_loc_auc_smooth"][0]:.3f} '
          f'洗白率={res["old_scheme"]["washed_rate"]*100:.0f}%')
    print(f'  C 密钥化: 编辑区 {res["keyed_edited"]["in_fail"][0]*100:.1f}% / 区外 '
          f'{res["keyed_edited"]["out_fail"][0]*100:.1f}%  loc AUC={res["keyed_edited"]["loc_auc_smooth"][0]:.3f}')
    for tag in ['reA_public_seed42', 'reB_random_key', 'reC_public_matrix']:
        d = res[tag]
        print(f'    {tag}: fail={d["fail_rate"][0]*100:.1f}% detect AUC={d["detect_auc_vs_clean"]:.3f} '
              f'洗白率={d["edits_masked_rate"]*100:.0f}% loc AUC={d["loc_auc_smooth"][0]:.3f}')
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1200, help='A2 局部编辑样本数')
    ap.add_argument('--n-wrongkey', type=int, default=200)
    ap.add_argument('--n-reembed', type=int, default=200)
    ap.add_argument('--skip-a1', action='store_true')
    ap.add_argument('--no-npz', action='store_true')
    args = ap.parse_args()
    t0 = time.time()

    print('=== A1 全局重绘 (r3 协议, 3 变体) ===')
    res = {'meta': {'ts': datetime.datetime.now().isoformat(),
                    'delta': DELTA, 'msglen': MSGLEN,
                    'n_a2': args.n, 'n_wrongkey': args.n_wrongkey,
                    'n_reembed': args.n_reembed,
                    'master_key_id': ks.hkdf_sha256(MASTER, b'id', b'rid', 8).hex(),
                    'patterns': ['old(seed42)', 'keyed-matrix', 'keyed-coord'],
                    'attackset': 'attackset_20261001_222447 (SD inpaint + bbox paste)'}}
    res['A1_global_repaint'] = {} if args.skip_a1 else exp_a1()

    print('\n=== A2 局部编辑 (r6 协议, 全量) ===')
    res['A2_local_edit'], npz = exp_a2(args.n, save_npz=not args.no_npz)

    print('\n=== B 错误密钥 ===')
    res['B_wrong_key'] = exp_b(args.n_wrongkey)

    print('\n=== C re-embedding 洗白攻击 ===')
    res['C_reembedding'] = exp_c(args.n_reembed)

    # ---- 判定汇总 (文档直接引用这些字段)
    a2 = res['A2_local_edit']
    det_old, det_k = a2['old']['detect_auc_paired'], a2['keyed_coord']['detect_auc_paired']
    loc_old = a2['old']['loc_auc_smooth'][0]
    loc_k = a2['keyed_coord']['loc_auc_smooth'][0]
    a1 = res.get('A1_global_repaint', {})
    a1_aucs = [c['conds'][cond]['detect_auc'] for c in a1.values() for cond in ['old', 'crd']]
    bw = res['B_wrong_key']
    c = res['C_reembedding']
    keyed_re = [c[t]['edits_masked_rate'] for t in
                ['reA_public_seed42', 'reB_random_key', 'reC_public_matrix']]
    keyed_det = [c[t]['detect_auc_vs_clean'] for t in
                 ['reA_public_seed42', 'reB_random_key', 'reC_public_matrix']]
    res['verdicts'] = {
        'A2_keyed_preserves_conclusions': {
            'detect_auc_old': det_old, 'detect_auc_keyed': det_k,
            'detect_auc_delta': det_k - det_old,
            'detect_auc_keyed_matrix': a2['keyed_matrix']['detect_auc_paired'],
            'detect_auc_delta_matrix': a2['keyed_matrix']['detect_auc_paired'] - det_old,
            'loc_auc_old': loc_old, 'loc_auc_keyed': loc_k, 'loc_auc_delta': loc_k - loc_old,
            'A1_all_aucs': a1_aucs,
            'pass': abs(det_k - det_old) < 0.03 and abs(loc_k - loc_old) < 0.02,
        },
        'B_key_is_necessary': {
            'correct_key_auc': bw['correct_key']['detect_auc'],
            'wrong_key_auc_mean': bw['random_keys']['per_key_detect_auc'][0],
            'wrong_key_auc_ci95': bw['random_keys']['per_key_detect_auc'][1],
            'wrong_key_blind_fraction': bw['random_keys']['wrong_key_decidability'],
            'pass': abs(bw['random_keys']['per_key_detect_auc'][0] - 0.5) < 0.1,
        },
        'C_key_stops_reembedding_laundering': {
            'old_washed_rate': c['old_scheme']['washed_rate'],
            'old_loc_auc_before': c['old_scheme']['loc_auc_smooth'][0],
            'old_loc_auc_after': c['old_scheme']['reembed_loc_auc_smooth'][0],
            'old_edit_fail_before': c['old_scheme']['in_fail'][0],
            'old_edit_fail_after': c['old_scheme']['reembed_in_fail'][0],
            'keyed_washed_rate_max': max(keyed_re),
            'keyed_detect_auc_min': min(keyed_det),
            'keyed_loc_auc_after': [c[t]['loc_auc_smooth'][0] for t in
                                    ['reA_public_seed42', 'reB_random_key', 'reC_public_matrix']],
            'pass': c['old_scheme']['washed_rate'] > 0.5 and max(keyed_re) < 0.05,
        },
    }
    res['runtime_sec'] = round(time.time() - t0, 1)
    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r7_keyed_scheme_{ts}.json'
    json.dump(res, open(fp, 'w'), indent=1, ensure_ascii=False)
    print(f'\n结果: {fp}  ({res["runtime_sec"]}s)')
    if npz:
        print(f'失败图: {npz}')


if __name__ == '__main__':
    main()
