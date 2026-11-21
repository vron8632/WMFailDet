#!/usr/bin/env python3
"""r57: 编辑类型归因（可解释版）— 失效图形状特征 + 平衡分类器.

检验假设: 失效图的**形状统计**是否携带编辑类型信息 (remove vs replace).
特征 (全部可从失效图直接算, 可解释): 面积/紧致度/边界复杂度/强度/连通域数/形状比
分类器: 逻辑回归 (系数可读) + 随机森林对照; 平衡采样
对照: 随机 (0.5) / 多数类
输出: results/r57_attribution_features.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs
from r34_distill_matrix import DELTA, CHK, MASTER
from r56_edit_attribution import label_of, BIG, AS


def shape_feats(fail_sm, mask_pred):
    """失效图 -> 形状特征 (可解释)."""
    b = mask_pred.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    areas = sorted([st[i, cv2.CC_STAT_AREA] for i in range(1, n)], reverse=True)
    total = float(b.sum()) + 1e-6
    ys, xs = np.where(b > 0)
    if len(xs) == 0:
        return [0.] * 8
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    bbox = float(h * w) + 1e-6
    # 边界复杂度: 周长^2 / 面积 (圆形最小)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    compact = float(per ** 2 / total)
    # 强度分布
    vals = fail_sm[b > 0]
    return [float(b.sum()),
            float(total / bbox),                       # 填充率 (紧致度)
            compact,                                   # 边界复杂度
            float(h / w),                              # 形状比
            float(len(areas)),                         # 连通域数
            float(areas[0] / total) if areas else 0.,  # 最大块占比
            float(vals.mean()),                        # 失效强度均值
            float(vals.std())]                         # 失效强度方差


def build(root, limit=None):
    meta = json.load(open(f'{root}/meta.json'))
    X, y, ids = [], [], []
    t0 = time.time()
    for s_ in meta['samples']:
        lab = label_of(s_.get('instruction'))
        if lab not in ('remove', 'replace'):
            continue
        sid = s_['id']
        fo, fa = f'{root}/images/orig_{sid}.png', f'{root}/images/att_{sid}.png'
        if not (os.path.exists(fo) and os.path.exists(fa)):
            continue
        o = r8.crop8(np.array(Image.open(fo).convert('RGB')))
        a = r8.crop8(np.array(Image.open(fa).convert('RGB')))
        key = ks.image_key(MASTER, sid); k4 = layer_key(key, 'bs4')
        wm = ks.embed_keyed(o, key, DELTA, (1, 0), CHK, pattern='coord')
        wm = embed_bs(wm, k4, 4, DELTA, (1, 1))
        edited, _ = paste_edit(wm, a, s_['bbox'])
        f8 = ks.fail_map(edited, key, DELTA, CHK, 'coord')
        sm = ks.smooth_fail_map(f8, 5)
        X.append(shape_feats(sm, (sm > 0.38)))
        y.append(0 if lab == 'remove' else 1)
        ids.append(sid)
        if limit and len(X) >= limit:
            break
    print(f'{root.split("/")[-1]}: {len(X)} ({time.time()-t0:.0f}s) '
          f'remove={y.count(0)} replace={y.count(1)}', flush=True)
    return np.array(X), np.array(y)


def main():
    X1, y1 = build(BIG)
    X2, y2 = build(AS)
    rng = np.random.default_rng(0)
    # 平衡: 每类等量
    def balance(X, y, n):
        idx = []
        for c in (0, 1):
            ii = np.where(y == c)[0]
            idx.append(rng.choice(ii, min(n, len(ii)), replace=False))
        idx = np.concatenate(idx); rng.shuffle(idx)
        return X[idx], y[idx]
    # 训练: bigset 平衡 600/类; 测试: bigset 剩下的 + attackset
    Xtr, ytr = balance(X1, y1, 600)
    mask_tr = np.zeros(len(y1), bool)
    # 简单划分: 用前 600 remove + 前 600 replace 作训练, 其余测试
    tr_idx = []
    for c in (0, 1):
        ii = np.where(y1 == c)[0][:600]
        tr_idx.append(ii)
    tr_idx = np.concatenate(tr_idx)
    te_mask = np.ones(len(y1), bool); te_mask[tr_idx] = False
    Xte, yte = X1[te_mask], y1[te_mask]
    print(f'train={len(tr_idx)} test_in={len(Xte)} test_cross={len(X2)}', flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    scaler = StandardScaler().fit(X1[tr_idx])
    out = {'meta': dict(n_train=len(tr_idx), n_test_in=int(len(Xte)), n_test_cross=int(len(X2)),
                        features=['area', 'fill_ratio', 'boundary_complex', 'aspect',
                                  'n_components', 'largest_frac', 'fail_mean', 'fail_std'],
                        task='remove vs replace (balanced)'),
           'models': {}}
    for name, mdl in (('logreg', LogisticRegression(max_iter=2000, class_weight='balanced')),
                      ('rf', RandomForestClassifier(n_estimators=300, random_state=0,
                                                    class_weight='balanced'))):
        mdl.fit(scaler.transform(X1[tr_idx]), y1[tr_idx])
        res = {}
        for tag, (X_, y_) in (('in_domain', (Xte, yte)), ('cross_domain', (X2, y2))):
            pr = mdl.predict(scaler.transform(X_))
            prob = mdl.predict_proba(scaler.transform(X_))[:, 1]
            res[tag] = {'acc': round(float(accuracy_score(y_, pr)), 4),
                        'auc': round(float(roc_auc_score(y_, prob)), 4), 'n': int(len(y_))}
        if name == 'logreg':
            res['coef'] = {f: round(float(c), 3) for f, c in
                           zip(out['meta']['features'], mdl.coef_[0])}
        out['models'][name] = res
        print(f'[{name}] in={res["in_domain"]} cross={res["cross_domain"]}', flush=True)
        if name == 'logreg':
            print(f'  coef: {res["coef"]}', flush=True)
    json.dump(out, open(f'{BASE}/results/r57_attribution_features.json', 'w'), indent=1)
    print('-> results/r57_attribution_features.json')


if __name__ == '__main__':
    main()
