#!/usr/bin/env python3
"""r75: δ=25 配置下的证据链融合 (复验 P0-2).

r55 的 0.214 -> 0.107 是 δ=15 配置. 本实验在 δ=25 下重建 T5 取证结论评测:
  可信 A : clean / jpeg85 / noise2  (水印图, 未编辑)
  篡改 T : edited / edited+jpeg75
  不可判定 I : edited+rotate1 / resize0.8 / jpeg50 / feather4
教师证据 = δ=25 的 2×2 层失效图区域统计; 学生 = r73 δ=25 权重.
输出: results/r75_forensic_d25.json
"""
import sys, os, json, time
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r14b_multiscale import layer_key, embed_bs, fail_bs
from r34_distill_matrix import SmallUNet, MASTER, PAY, CHK
from r10_specificity import jpeg, noise, resize_round, rotate
from r71_tracks import predict

D = 25
N = 50          # 每个条件的样本数
OUT = f'{BASE}/results/r75_forensic_d25.json'


def fea(alarm):
    """失效图区域统计: (面积, 最大连通块占比)"""
    a = alarm.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(a, 8)
    areas = [st[i, cv2.CC_STAT_AREA] for i in range(1, n)]
    tot = max(1, int(a.sum()))
    return float(a.mean()), float(max(areas) / tot) if areas else 0.0


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SmallUNet(base=32).to(dev)
    m.load_state_dict(torch.load(f'{BASE}/results/r73_delta25_student.pth', map_location=dev))

    z = np.load(f'{BASE}/results/r70_data_d25.npz')
    img, gt = z['img'], z['gt']
    x_te, gt_te = img[-400:], gt[-400:]

    # 干净水印图 (δ=25, 未编辑) — 需要 orig 重建
    meta = json.load(open(f'{BASE}/data/bigset/meta.json'))
    ids = [s_['id'] for s_ in meta['samples'] if os.path.exists(f'{BASE}/data/bigset/images/orig_{s_["id"]}.png')]
    # 与训练相同的取样顺序: r70 用 meta['samples'] 前 2800, 测试取最后 400
    test_ids = ids[-400:]

    CLEAN, EDIT = [], []
    t0 = time.time()
    for sid in test_ids:
        o = r8.crop8(np.array(Image.open(f'{BASE}/data/bigset/images/orig_{sid}.png').convert('RGB')))
        key = ks.image_key(MASTER, sid)
        k4, k2 = layer_key(key, 'bs4'), layer_key(key, 'bs2')
        wm = ks.embed_keyed(o, key, D, PAY, CHK, pattern='coord')
        wm = embed_bs(wm, k4, 4, D, (1, 1))
        wm = embed_bs(wm, k2, 2, D, (1, 1))
        CLEAN.append(wm)
        if len(CLEAN) % 100 == 0:
            print(f'  clean {len(CLEAN)}/400 ({time.time()-t0:.0f}s)', flush=True)
    CLEAN = np.array(CLEAN)

    def teacher_stats(imgs, ids_):
        out = []
        for im, sid in zip(imgs, ids_):
            key = ks.image_key(MASTER, sid)
            k2 = layer_key(key, 'bs2')
            f2 = fail_bs(im, k2, 2, D, (1, 1))
            out.append(fea(f2))
        return out

    conds = {}
    # --- A 类
    conds['clean'] = ('A', CLEAN[:N])
    conds['jpeg85'] = ('A', np.array([jpeg(c, 85) for c in CLEAN[:N]]))
    conds['noise2'] = ('A', np.array([noise(c, 2) for c in CLEAN[:N]]))
    # --- T 类
    conds['edited'] = ('T', x_te[:N])
    conds['edited_jpeg75'] = ('T', np.array([jpeg(e, 75) for e in x_te[:N]]))
    # --- I 类
    conds['edited_rotate1'] = ('I', np.array([rotate(e, 1.0) for e in x_te[N:2*N]]))
    conds['edited_resize08'] = ('I', np.array([resize_round(e, 0.8) for e in x_te[N:2*N]]))
    conds['edited_jpeg50'] = ('I', np.array([jpeg(e, 50) for e in x_te[N:2*N]]))
    conds['edited_feather4'] = ('I', np.array([cv2.blur(e, (9, 9)) for e in x_te[N:2*N]]))

    rows = []
    for name, (lab, imgs) in conds.items():
        ids_this = test_ids[:N] if name.startswith(('clean', 'jpeg', 'noise')) else test_ids[:N]
        tstats = teacher_stats(imgs, ids_this)
        prob = predict(m, imgs, dev)
        for i, (ar, cg) in enumerate(tstats):
            p = prob[i]
            rows.append(dict(cond=name, label=lab,
                             t_area=ar, t_clust=cg,
                             s_area=float(p.mean()), s_max=float(p.max())))
        print(f'  [{name:16s}] n={len(imgs)}', flush=True)

    # --- 决策: 教师 (面积 + 聚集度) / 学生 (面积 + 置信度) / 融合
    A_t = float(np.percentile([r['t_area'] for r in rows], 55))
    A_s = float(np.percentile([r['s_area'] for r in rows], 40))
    for r in rows:
        # 教师: 弥散(聚集度低)且面积高 -> 退化导致的伪失效 -> I
        if r['t_area'] > A_t and r['t_clust'] < 0.25:
            r['teacher'] = 'I'
        else:
            r['teacher'] = 'T' if r['t_area'] > A_t else 'A'
        # 学生
        if r['s_max'] < 0.30:
            r['student'] = 'I'
        else:
            r['student'] = 'T' if r['s_area'] > A_s else 'A'
        # 融合: 教师 I 优先; 一致采纳; 冲突以失效聚集度裁决
        if r['teacher'] == 'I':
            r['fusion'] = 'I'
        elif r['teacher'] == r['student']:
            r['fusion'] = r['teacher']
        elif r['teacher'] == 'T' and r['student'] == 'A':
            r['fusion'] = 'T' if r['t_clust'] > 0.20 else 'A'
        else:
            r['fusion'] = 'T' if r['t_clust'] > 0.15 else 'A'

    def cost(key):
        c = {'miss': 0, 'false_alarm': 0, 'conservative': 0, 'correct': 0}
        for r in rows:
            lb, pr = r['label'], r[key]
            if lb == pr:
                c['correct'] += 1
            elif lb == 'T' and pr == 'A':
                c['miss'] += 1
            elif lb == 'A' and pr == 'T':
                c['false_alarm'] += 1
            else:
                c['conservative'] += 1
        n = len(rows)
        return {k: round(v / n, 4) for k, v in c.items()} | {
            'dangerous': round((c['miss'] + c['false_alarm']) / n, 4), 'n': n}

    out = {'meta': dict(delta=25, n=len(rows), n_per_cond=N,
                        conditions=list(conds), thresholds=dict(t_area=round(A_t, 4),
                                                                s_area=round(A_s, 4)),
                        baseline_delta15=dict(teacher=0.214, fusion=0.107)),
           'cost': {k: cost(k) for k in ('teacher', 'student', 'fusion')}}
    for k, v in out['cost'].items():
        print(f'  {k:8s} dangerous={v["dangerous"]:.3f} miss={v["miss"]:.3f} '
              f'FA={v["false_alarm"]:.3f} cons={v["conservative"]:.3f}', flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('->', OUT)


if __name__ == '__main__':
    main()
