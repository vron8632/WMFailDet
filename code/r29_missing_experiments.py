#!/usr/bin/env python3
"""r29 投稿前补实验（对应模拟审稿的"必修/建议"项）.

A 2x2 细化层可行性（回应"为何不做到 2x2"）
B 跨域阈值迁移（用 COCO 阈值直接套用到 5 个跨域/编辑器配置）
C δ 与平滑窗/阈值敏感性（消融缺口）
D 高清集 base 级聚类 CI（16 base × 10 编辑的相关性）
E 局部重嵌入攻击（只对编辑区重嵌入，检验定位是否被欺骗）
F LSB 脆弱水印对照（回应"换成别的脆弱水印是否同样成立"）

输出：results/r29_*.json（每个子实验一个文件）
"""
import sys, os, json, glob, datetime, time, io, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs
from r13_realistic_protocol import jpeg

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
HIRES = f'{BASE}/data/hires_expanded'
XD = f'{BASE}/data/xdomain'
DELTA = 15
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))
THR_TOP, THR_CON = 0.41, 0.15   # COCO 上标定的工作点


def post(pred, min_area):
    p = pred.astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
    keep = np.zeros_like(p)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == c] = 1
    return keep.astype(bool)


def loc(fail, bbox, bs, k=5, t=0.38, min_area=8):
    H, W = fail.shape[0]*bs, fail.shape[1]*bs
    f = cv2.blur(fail.astype(np.float32), (k, k))
    pred = post(f > t, min_area)
    pred_px = np.repeat(np.repeat(pred, bs, 0), bs, 1)[:H, :W]
    x1, y1, x2, y2 = bbox
    gt = np.zeros((H, W), bool)
    gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = True
    inter = float((pred_px & gt).sum())
    return {'iou': inter/max(1, float((pred_px | gt).sum())),
            'recall': inter/max(1, float(gt.sum()))}


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return 99.0 if mse == 0 else float(10*np.log10(255**2/mse))


def stats(fail, k=9):
    s = ks.smooth_fail_map(fail, k)
    v = np.sort(s.ravel())
    n = max(1, int(0.10*v.size))
    top = float(v[-n:].mean())
    return top, top-float(np.median(v))


def load_coco(n):
    meta = json.load(open(f'{AS}/meta.json'))
    out = []
    for s in meta['samples'][:n]:
        sid = s['id']
        try:
            o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        out.append((sid, o, a, s['bbox']))
    return out


def save(name, obj):
    obj['meta'] = {**obj.get('meta', {}), 'ts': datetime.datetime.now().isoformat()}
    fp = f'{BASE}/results/{name}'
    json.dump(obj, open(fp, 'w'), indent=1)
    print('  ->', fp)
    return fp


# =====================================================================
def exp_A_2x2(n=150):
    print('[A] 2x2 细化层可行性')
    items = load_coco(n)
    rows = []
    for sid, o, a, bbox in items:
        key = ks.image_key(MASTER, sid)
        k8, k4, k2 = layer_key(key, 'bs8'), layer_key(key, 'bs4'), layer_key(key, 'bs2')
        wm = ks.embed_keyed(o, k8, DELTA, PAY, CHK, pattern='coord')
        wm4 = embed_bs(wm, k4, 4, DELTA, (1, 1))
        wm2 = embed_bs(wm4, k2, 2, DELTA, (1, 1))
        att, _ = paste_edit(wm2, a, bbox)
        f2c = fail_bs(wm2, k2, 2, DELTA, (1, 1))
        f2a = fail_bs(att, k2, 2, DELTA, (1, 1))
        # JPEG 85 / 75
        att85 = jpeg(att, 85)
        att75 = jpeg(att, 75)
        f85 = fail_bs(att85, k2, 2, DELTA, (1, 1))
        f75 = fail_bs(att75, k2, 2, DELTA, (1, 1))
        rows.append({
            'clean_fail': float(f2c.mean()), 'psnr': psnr(o, wm2),
            'iou': loc(f2a, bbox, 2, 5, 0.38, 128)['iou'],
            'cov50': float(loc(f2a, bbox, 2, 5, 0.38, 128)['recall'] >= 0.5),
            'clean_fail_j85': float(f85.mean()), 'iou_j85': loc(f85, bbox, 2, 5, 0.38, 128)['iou'],
            'clean_fail_j75': float(f75.mean()), 'iou_j75': loc(f75, bbox, 2, 5, 0.38, 128)['iou'],
        })
    def m(k): return float(np.mean([r[k] for r in rows]))
    res = {'n': len(rows), 'clean_fail': m('clean_fail'), 'psnr': m('psnr'),
           'iou_lossless': m('iou'), 'cov50_lossless': m('cov50'),
           'clean_after_jpeg85': m('clean_fail_j85'), 'iou_jpeg85': m('iou_j85'),
           'clean_after_jpeg75': m('clean_fail_j75'), 'iou_jpeg75': m('iou_j75')}
    print('  ', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
    return save('r29_A_2x2_feasibility.json', {'result': res})


# =====================================================================
def exp_B_threshold_transfer(n=200):
    print('[B] 跨域阈值迁移（COCO 阈值直接套用）')
    domains = [('butterfly', f'{XD}/butterfly_edits'), ('sat', f'{XD}/sat_edits'),
               ('doc', f'{XD}/doc_edits'), ('pokemon', f'{XD}/pokemon_edits'),
               ('lama', f'{XD}/lama_edits')]
    out = {}
    for tag, d in domains:
        try:
            meta = json.load(open(f'{d}/meta.json'))
        except FileNotFoundError:
            print('  skip', tag); continue
        fpr_t, tpr_t, fpr_c, tpr_c = [], [], [], []
        for s in meta['samples'][:n]:
            sid = s['id']
            o = np.array(Image.open(f'{d}/images/orig_{sid}.png').convert('RGB'))
            a = np.array(Image.open(f'{d}/images/att_{sid}.png').convert('RGB'))
            key = ks.image_key(MASTER, f'{tag}_{sid}')
            wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
            att, _ = paste_edit(wm, a, s['bbox'])
            tc, cc = stats(ks.fail_map(wm, key, DELTA, CHK, 'coord'))
            ta, ca = stats(ks.fail_map(att, key, DELTA, CHK, 'coord'))
            fpr_t.append(float(tc > THR_TOP)); tpr_t.append(float(ta > THR_TOP))
            fpr_c.append(float(cc > THR_CON)); tpr_c.append(float(ca > THR_CON))
        out[tag] = {'n': len(fpr_t),
                    'FPR_top': float(np.mean(fpr_t)), 'TPR_top': float(np.mean(tpr_t)),
                    'FPR_contrast': float(np.mean(fpr_c)), 'TPR_contrast': float(np.mean(tpr_c))}
        print('  ', tag, {k: round(v, 3) for k, v in out[tag].items() if k != 'n'})
    return save('r29_B_threshold_transfer.json', {'result': out,
                'thresholds': {'top': THR_TOP, 'contrast': THR_CON,
                               'source': 'COCO-256 clean p99 / grid search'}})


# =====================================================================
def exp_C_sensitivity(n=150):
    print('[C] δ 与平滑窗/阈值敏感性')
    items = load_coco(n)
    deltas = [8, 15, 25, 40]
    dres = {}
    for delta in deltas:
        cl, ps, ious = [], [], []
        for sid, o, a, bbox in items:
            key = ks.image_key(MASTER, sid)
            k8 = layer_key(key, 'bs8')
            wm = ks.embed_keyed(o, k8, delta, PAY, CHK, pattern='coord')
            att, _ = paste_edit(wm, a, bbox)
            cl.append(float(fail_bs(wm, k8, 8, delta, CHK).mean()))
            ps.append(psnr(o, wm))
            ious.append(loc(fail_bs(att, k8, 8, delta, CHK), bbox, 8, 5, 0.38, 8)['iou'])
        dres[str(delta)] = {'clean_fail': float(np.mean(cl)), 'psnr': float(np.mean(ps)),
                            'iou_8x8': float(np.mean(ious))}
        print('  delta', delta, {k: round(v, 3) for k, v in dres[str(delta)].items()})
    # 平滑窗 × 阈值（4x4 层，固定 delta=15）
    grid = {}
    items4 = []
    for sid, o, a, bbox in items:
        key = ks.image_key(MASTER, sid)
        k8, k4 = layer_key(key, 'bs8'), layer_key(key, 'bs4')
        wm = ks.embed_keyed(o, k8, DELTA, PAY, CHK, pattern='coord')
        wm4 = embed_bs(wm, k4, 4, DELTA, (1, 1))
        att, _ = paste_edit(wm4, a, bbox)
        items4.append((fail_bs(att, k4, 4, DELTA, (1, 1)), bbox))
    for k in [3, 5, 7, 9, 11]:
        for t in [0.32, 0.35, 0.38, 0.42, 0.46]:
            vals = [loc(f, b, 4, k, t, 32)['iou'] for f, b in items4]
            grid[f'k{k}_t{t}'] = float(np.mean(vals))
    best = max(grid.items(), key=lambda kv: kv[1])
    print('  best grid', best)
    return save('r29_C_sensitivity.json', {'delta': dres, 'grid_4x4': grid,
                'best_grid': {'config': best[0], 'iou': best[1]}})


# =====================================================================
def exp_D_cluster_ci(n=200):
    print('[D] 高清 base 级聚类 CI')
    meta = json.load(open(f'{HIRES}/meta.json'))
    rows = []
    for s in meta['samples'][:n]:
        sid = s['id']
        try:
            o = np.array(Image.open(f'{HIRES}/images/orig_{sid}.png').convert('RGB'))
            a = np.array(Image.open(f'{HIRES}/images/att_{sid}.png').convert('RGB'))
        except FileNotFoundError:
            continue
        key = ks.image_key(MASTER, sid)
        k8, k4 = layer_key(key, 'bs8'), layer_key(key, 'bs4')
        wm = ks.embed_keyed(o, k8, DELTA, PAY, CHK, pattern='coord')
        wm4 = embed_bs(wm, k4, 4, DELTA, (1, 1))
        att, _ = paste_edit(wm4, a, s['bbox'])
        r4 = loc(fail_bs(att, k4, 4, DELTA, (1, 1)), s['bbox'], 4, 5, 0.38, 32)
        rows.append({'cid': sid.split('_')[0], 'iou': r4['iou'], 'recall': r4['recall']})
    cids = sorted({r['cid'] for r in rows})
    rng = np.random.RandomState(0)
    boot = []
    for _ in range(2000):
        pick = rng.choice(cids, len(cids), replace=True)
        vals = [r['iou'] for r in rows if r['cid'] in set(pick)]
        boot.append(float(np.mean(vals)))
    res = {'n': len(rows), 'n_bases': len(cids),
           'iou_mean': float(np.mean([r['iou'] for r in rows])),
           'iou_ci95_cluster': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           'recall_mean': float(np.mean([r['recall'] for r in rows]))}
    print('  ', {k: v for k, v in res.items()})
    return save('r29_D_cluster_ci.json', {'result': res})


# =====================================================================
def exp_E_local_reembed(n=150):
    print('[E] 局部重嵌入攻击（只对编辑区重嵌入）')
    items = load_coco(n)
    rows = []
    for sid, o, a, bbox in items:
        key = ks.image_key(MASTER, sid)
        k8 = layer_key(key, 'bs8')
        wm = ks.embed_keyed(o, k8, DELTA, PAY, CHK, pattern='coord')
        att, box = paste_edit(wm, a, bbox)
        H, W = att.shape[:2]
        bx1, by1, bx2, by2 = box
        # 攻击者：只对编辑区（块对齐）重新嵌入公开 seed 的校验位
        ry1, ry2 = (by1//8)*8, min(((by2+7)//8)*8, H)
        rx1, rx2 = (bx1//8)*8, min(((bx2+7)//8)*8, W)
        region = att[ry1:ry2, rx1:rx2].copy()
        legacy_key = ks.hkdf_sha256(b'legacy-public-seed-42', b'\x00'*32, b'wm/layer/bs8/v1', 32)
        re_region = ks.embed_keyed(region, legacy_key, DELTA, PAY, CHK, pattern='coord')
        re = att.copy()
        re[ry1:ry2, rx1:rx2] = re_region
        # 检测（正确 key）
        f = ks.fail_map(re, k8, DELTA, CHK, 'coord')
        gt = np.zeros(f.shape, bool)
        gt[by1//8:(by2+7)//8, bx1//8:(bx2+7)//8] = True
        in_f = float(f[gt].mean()) if gt.any() else float('nan')
        out_f = float(f[~gt].mean()) if (~gt).any() else float('nan')
        f_sm = ks.smooth_fail_map(f, 9)
        auc = ks.auc(f_sm[~gt], f_sm[gt]) if gt.any() and (~gt).any() else float('nan')
        top, con = stats(f)
        rows.append({'in_fail': in_f, 'out_fail': out_f, 'loc_auc': auc,
                     'top': top, 'contrast': con})
    def m(k):
        v = [r[k] for r in rows]
        v = [x for x in v if not np.isnan(x)]
        return float(np.mean(v))
    res = {'n': len(rows), 'in_fail': m('in_fail'), 'out_fail': m('out_fail'),
           'loc_auc': m('loc_auc'), 'top': m('top'), 'contrast': m('contrast'),
           'edited_masked_rate': float(np.mean([r['in_fail'] < 0.35 and r['out_fail'] < 0.35
                                                for r in rows])),
           'note': '攻击者只对编辑区（块对齐）重嵌入公开 seed 的校验位'}
    print('  ', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
    return save('r29_E_local_reembed.json', {'result': res})


# =====================================================================
def embed_lsb(img, key, bs=8):
    """LSB 脆弱水印：每 bs×bs 块把 1 bit 写进块内首像素的 LSB."""
    out = img.copy()
    H, W = out.shape[0]//bs*bs, out.shape[1]//bs*bs
    n_h, n_w = H//bs, W//bs
    pat = ks.check_pattern(key, n_h, n_w, 'coord')
    for i in range(n_h):
        for j in range(n_w):
            y, x = i*bs, j*bs
            out[y, x, 0] = (out[y, x, 0] & 0xFE) | int(pat[i, j])
    return out


def fail_lsb(img, key, bs=8):
    H, W = img.shape[0]//bs*bs, img.shape[1]//bs*bs
    n_h, n_w = H//bs, W//bs
    pat = ks.check_pattern(key, n_h, n_w, 'coord')
    f = np.zeros((n_h, n_w), np.float32)
    for i in range(n_h):
        for j in range(n_w):
            f[i, j] = 1.0 if (img[i*bs, j*bs, 0] & 1) != pat[i, j] else 0.0
    return f


def exp_F_lsb(n=120):
    print('[F] LSB 脆弱水印对照')
    items = load_coco(n)
    rows = []
    for sid, o, a, bbox in items:
        key = ks.image_key(MASTER, sid)
        wm = embed_lsb(o, key)
        att, _ = paste_edit(wm, a, bbox)
        fc = fail_lsb(wm, key)
        fa = fail_lsb(att, key)
        f85 = fail_lsb(jpeg(att, 85), key)
        f75 = fail_lsb(jpeg(att, 75), key)
        fnoise = fail_lsb(np.clip(att.astype(np.float32) +
                                  np.random.RandomState(0).normal(0, 2, att.shape), 0, 255
                                  ).astype(np.uint8), key)
        rows.append({
            'clean': float(fc.mean()),
            'edited': float(fa.mean()),
            'jpeg85': float(f85.mean()), 'jpeg75': float(f75.mean()),
            'noise2': float(fnoise.mean()),
            'iou': loc(fa, bbox, 8, 5, 0.38, 8)['iou'],
        })
    def m(k): return float(np.mean([r[k] for r in rows]))
    res = {k: m(k) for k in ['clean', 'edited', 'jpeg85', 'jpeg75', 'noise2', 'iou']}
    res['n'] = len(rows)
    print('  ', {k: round(v, 3) for k, v in res.items()})
    return save('r29_F_lsb_baseline.json', {'result': res,
                'note': 'LSB 脆弱水印：每 8x8 块 1 bit 写入块首像素 LSB'})


if __name__ == '__main__':
    t0 = time.time()
    exp_A_2x2()
    exp_B_threshold_transfer()
    exp_C_sensitivity()
    exp_D_cluster_ci()
    exp_E_local_reembed()
    exp_F_lsb()
    print(f'ALL DONE {time.time()-t0:.0f}s')
