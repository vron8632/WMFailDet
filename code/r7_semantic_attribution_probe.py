#!/usr/bin/env python3
"""语义归因可行性判定：水印死亡的空间/频带模式是否携带"编辑类型"信息？

问题: 若失败图只提供"位置"，则语义归因无处着力；若失败模式本身随编辑类型变化，
      则"语义归因"（不只是输出掩膜，还要说出编辑类型/语义）有物理信号基础。

数据: attackset_20261001 (1200 样本; replace 894 / remove 294; bbox 面积分布无混淆)
特征 (全部来自死亡图, 不用 instruction 文本):
  f_in, f_out       区域内/外失败率
  f_diff            f_in - f_out
  f_edge_ratio      区域边缘带失败率 / 区域内部失败率 (整区重绘 vs 局部剥除的差异)
  f_band_*          (1,0)/(1,1)/(2,1)/(3,2) 各频带失败率 (区域内外差)
  f_blob_cnt        失败块连通域数量
  f_blob_max        最大失败连通域 / 区域面积
  f_area            区域面积
  f_clean_in        干净图上该区域的失败率 (基线校正)
判定: 5-fold CV 分类 replace vs remove 的 AUC；面积匹配子集复核。
输出: results/r7_semantic_attribution_<ts>.json
用法: apjf-python r7_semantic_attribution_probe.py [--n 1200]
"""
import sys, os, json, argparse, datetime

sys.path.insert(0, '/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/code')
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
DELTA, MSGLEN = 15, 256
PAY, CHK = (1, 0), (1, 1)
BANDS = [(1, 0), (1, 1), (2, 1), (3, 2)]


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def fail_map(img, coef, delta=DELTA):
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    H, W = y.shape
    n_h, n_w = H // 8, W // 8
    exp = rs.check_bits(H, W, n_h, n_w)
    f = np.zeros((n_h, n_w), np.float32)
    for i in range(n_h):
        for j in range(n_w):
            c = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8])[coef]
            r = c % delta
            bit = 1 if abs(r - delta/2) < abs(r) else 0
            f[i, j] = 1.0 if bit != exp[i, j] else 0.0
    return f


def paste_edit(wm, att, bbox01):
    H, W = wm.shape[:2]
    x1, y1, x2, y2 = bbox01
    bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
    bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
    out = wm.copy()
    out[by1:by2, bx1:bx2] = att[by1:by2, bx1:bx2]
    return out, (by1, bx1, by2, bx2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    rs.PAY_COEF, rs.CHECK_COEF = PAY, CHK

    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:args.n]
    rng = np.random.RandomState(args.seed)
    msg = rng.randint(0, 2, MSGLEN).astype(np.uint8)
    print(f'r7 语义归因可行性: n={len(samples)}')

    rows = []
    for idx, s in enumerate(samples):
        sid = s['id']
        try:
            o = crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        wm = rs.embed_signal(o, msg, DELTA)
        att_wm, (by1, bx1, by2, bx2) = paste_edit(wm, a, s['bbox'])

        f_clean = fail_map(wm, CHK)
        n_h, n_w = f_clean.shape
        gt = np.zeros((n_h, n_w), bool)
        gt[by1//8:(by2+7)//8, bx1//8:(bx2+7)//8] = True
        if gt.mean() > 0.95 or gt.sum() < 4:
            continue                     # 全图编辑/区域过小, 归因无意义
        in_b, out_b = gt, ~gt

        ins = s['instruction']
        etype = 'replace' if ('替换' in ins or '换成' in ins or '换上' in ins or '改变' in ins) else \
                ('remove' if ('删除' in ins or '移除' in ins or '去掉' in ins) else None)
        if etype is None:
            continue

        # 边缘带 (区域外扩 1 块) vs 区域内部
        ker = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(gt.astype(np.uint8), ker).astype(bool)
        ring = dil & ~gt
        core = cv2.erode(gt.astype(np.uint8), ker).astype(bool)

        feat = {'id': sid, 'etype': etype, 'area': float(s['bbox'][2]-s['bbox'][0]) *
                float(s['bbox'][3]-s['bbox'][1])}
        feats = {}
        for band in BANDS:
            f_att = fail_map(att_wm, band)
            fi, fo = f_att[in_b].mean(), f_att[out_b].mean()
            feats[f'f_in_{band[0]}{band[1]}'] = float(fi)
            feats[f'f_banddiff_{band[0]}{band[1]}'] = float(fi - fo)
        f_att = fail_map(att_wm, CHK)
        feat['f_in'] = float(f_att[in_b].mean())
        feat['f_out'] = float(f_att[out_b].mean())
        feat['f_diff'] = feat['f_in'] - feat['f_out']
        feat['f_ring'] = float(f_att[ring].mean()) if ring.any() else float('nan')
        feat['f_core'] = float(f_att[core].mean()) if core.any() else float('nan')
        feat['f_edge_ratio'] = float(feat['f_ring'] - feat['f_core']) if (ring.any() and core.any()) else 0.0
        feat['f_clean_in'] = float(f_clean[in_b].mean())
        # 失败块连通域（区域内）
        bw = (f_att[in_b] > 0.5).astype(np.uint8).reshape(-1)
        feat.update(feats)
        rows.append(feat)
        if (idx + 1) % 200 == 0:
            print(f'  [{idx+1}/{len(samples)}]', flush=True)

    # 分类判定
    feat_keys = [k for k in rows[0] if k not in ('id', 'etype', 'area')]
    X = np.array([[r[k] for k in feat_keys] for r in rows], float)
    y = np.array([1 if r['etype'] == 'replace' else 0 for r in rows])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f'\n样本: {len(rows)}  replace={y.sum()} remove={(1-y).sum()}  特征数={len(feat_keys)}')

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    res = {}
    for name, clf in [('RandomForest', RandomForestClassifier(n_estimators=300, random_state=0)),
                      ('LogReg', make_pipeline(StandardScaler(),
                                               LogisticRegression(max_iter=2000)))]:
        p = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
        res[name] = float(roc_auc_score(y, p))
        print(f'  {name:14s} 5-fold CV AUC = {res[name]:.3f}')

    # 面积匹配子集复核（控制面积混淆）
    areas = np.array([r['area'] for r in rows])
    lo, hi = np.percentile(areas, [25, 75])
    m = (areas >= lo) & (areas <= hi)
    if m.sum() > 100 and len(set(y[m])) == 2:
        p = cross_val_predict(RandomForestClassifier(n_estimators=300, random_state=0),
                              X[m], y[m], cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method='predict_proba')[:, 1]
        res['RF_area_matched'] = float(roc_auc_score(y[m], p))
        print(f'  RF 面积匹配(25-75分位, n={m.sum()}) AUC = {res["RF_area_matched"]:.3f}')

    # 单特征判别力 top5
    from sklearn.metrics import roc_auc_score as A
    single = []
    for i, k in enumerate(feat_keys):
        try:
            a = A(y, X[:, i])
            single.append((k, max(a, 1 - a)))
        except Exception:
            pass
    single.sort(key=lambda t: -t[1])
    print('\n  单特征判别力 top6:')
    for k, a in single[:6]:
        print(f'    {k:20s} AUC={a:.3f}')

    print('\n=== 判定 ===')
    best = max(v for k, v in res.items())
    print(f'  最高 AUC = {best:.3f} → ' + (
        '✅ 死亡模式携带编辑类型信息，语义归因有信号基础' if best > 0.65 else
        '⚠️ 弱信号 (0.55-0.65)，语义归因需额外设计' if best > 0.55 else
        '❌ 接近随机，单纯死亡图不足以支撑语义归因'))

    os.makedirs(f'{BASE}/results', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r7_semantic_attribution_{ts}.json'
    json.dump({'meta': {'n': len(rows), 'bands': BANDS, 'delta': DELTA,
                        'ts': datetime.datetime.now().isoformat()},
               'cv_auc': res, 'single_feature_auc': single, 'rows': rows},
              open(fp, 'w'), indent=1)
    print(f'\n结果: {fp}')


if __name__ == '__main__':
    main()
