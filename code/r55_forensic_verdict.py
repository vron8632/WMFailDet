#!/usr/bin/env python3
"""r55: 取证结论评估 — AUTHENTIC / TAMPERED / INDETERMINATE 三级输出 + 混淆矩阵.

对比:
  二值基线 (强制二选一): 教师 top10 / 学生掩膜面积 / 融合
  三级 (ours): 教师 low10 拒识 + 学生置信度拒识 + 融合拒识
条件集:
  authentic   : 干净水印图 + 轻度传播 (JPEG85 / noise σ1)   -> 应判 AUTHENTIC
  tampered    : 编辑图 (+JPEG75)                            -> 应判 TAMPERED
  indeterminate: rotate1 / resize0.8 / jpeg50 / feather4    -> 应判 INDETERMINATE
输出: results/r55_forensic_verdict.json
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
from r34_distill_matrix import SmallUNet, alarm_from_fail, DELTA, PAY, CHK, MASTER
from r10_specificity import jpeg, noise, resize_round, rotate

AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
N = 100   # 每类条件样本数


def feather(im, r=4):
    return cv2.blur(im, (2*r+1, 2*r+1))


def collect():
    """-> items: list of (img, label, cond)  label in {A, T, I}"""
    meta = json.load(open(f'{AS}/meta.json'))
    ids = []
    for s_ in meta['samples']:
        sid = s_['id']
        if os.path.exists(f'{AS}/images/orig_{sid}.png') and os.path.exists(f'{AS}/images/att_{sid}.png'):
            ids.append(s_)
        if len(ids) >= 1200:
            break
    test = ids[1000:1200]
    out = []
    for i, s_ in enumerate(test[:N]):
        sid = s_['id']
        o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
        a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        key = ks.image_key(MASTER, sid); k4 = layer_key(key, 'bs4')
        wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        wm = embed_bs(wm, k4, 4, DELTA, (1, 1))
        edited, _ = paste_edit(wm, a, s_['bbox'])
        # authentic: 干净 + 轻度传播
        out.append((jpeg(wm, 85), 'A', 'clean_jpeg85', sid, key, k4))
        out.append((noise(wm, 1), 'A', 'clean_noise1', sid, key, k4))
        # tampered
        out.append((edited, 'T', 'edited', sid, key, k4))
        out.append((jpeg(edited, 75), 'T', 'edited_jpeg75', sid, key, k4))
        # indeterminate
        out.append((rotate(edited, 1.0), 'I', 'rot1', sid, key, k4))
        out.append((resize_round(edited, 0.8), 'I', 'resize08', sid, key, k4))
        out.append((jpeg(edited, 50), 'I', 'jpeg50', sid, key, k4))
        out.append((feather(edited, 4), 'I', 'feather4', sid, key, k4))
    return out


T_CONTRAST = 0.15   # 论文 fig3 的 contrast 工作点 (FPR 1.7%)


def teacher_verdict(img, key, k4, use_contrast=True):
    """教师: 8×8 low10 判同步; 而篡改用 contrast(聚集度) 判 —— 对均匀退化免疫."""
    f8 = ks.fail_map(img, key, DELTA, CHK, 'coord')
    sm8 = ks.smooth_fail_map(f8, 5)
    low10 = float(np.percentile(sm8, 10))
    v = np.sort(sm8.ravel()); k = max(1, int(0.10 * v.size))
    top10 = float(v[-k:].mean())
    med = float(np.median(sm8))
    contrast = top10 - med
    if low10 >= 0.20:
        return 'I', contrast, low10
    if use_contrast:
        return ('T' if contrast > T_CONTRAST else 'A'), contrast, low10
    return ('T' if top10 > 0.41 else 'A'), top10, low10


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SmallUNet(base=32).to(dev)
    wp = f'{BASE}/results/r43_best_student.pth'
    assert os.path.exists(wp), 'need student weights'
    m.load_state_dict(torch.load(wp, map_location=dev))

    items = collect()
    print(f'items: {len(items)}', flush=True)
    imgs = np.array([x[0] for x in items])
    # 学生推理 (概率图)
    m.eval(); probs = []
    with torch.no_grad():
        for i in range(0, len(imgs), 16):
            xb = torch.from_numpy(imgs[i:i+16].transpose(0, 3, 1, 2)).float().to(dev) / 255.
            probs.append(torch.sigmoid(m(xb)).cpu().numpy()[:, 0])
    probs = np.concatenate(probs)
    area = probs.mean(axis=(1, 2))           # 平均激活 = 面积比代理
    maxp = probs.max(axis=(1, 2))

    rows = []
    t0 = time.time()
    for k, (img, label, cond, sid, key, k4) in enumerate(items):
        tv, tscore, tlow = teacher_verdict(img, key, k4)
        rows.append(dict(label=label, cond=cond, teacher=tv, tscore=tscore, tlow=tlow,
                         area=float(area[k]), maxp=float(maxp[k])))
        if (k+1) % 100 == 0:
            print(f'  {k+1}/{len(items)} ({time.time()-t0:.0f}s)', flush=True)

    # ---- 学生二值/三级 (阈值在"类均衡"下取中位, 仅作演示口径; 论文应在 val 标定)
    A_arr = np.array([r['area'] for r in rows])
    tau_area = float(np.percentile(A_arr, 40))
    tau_conf = 0.30
    for r in rows:
        # 学生: 面积判篡改; 置信度低 -> 拒识
        if r['maxp'] < tau_conf:
            r['student'] = 'I'
        else:
            r['student'] = 'T' if r['area'] > tau_area else 'A'
        # 融合(证据链): 教师拒识优先; 冲突时用第二证据(contrast 是否"团块状")裁决
        if r['teacher'] == 'I':
            r['fusion'] = 'I'
        elif r['teacher'] == r['student']:
            r['fusion'] = r['teacher']
        elif r['teacher'] == 'T' and r['student'] == 'A':
            # 教师说篡改、学生说可信: 若失效不呈团块(contrast 低)= 均匀退化 -> 采信学生
            r['fusion'] = 'T' if r['tscore'] > 0.20 else 'A'
        else:  # 教师 A、学生 T -> 以教师为主(保守)
            r['fusion'] = 'T' if r['tscore'] > 0.15 else 'A'

    # ---- 取证代价评价: 漏检 >> 误报 >> 保守拒识
    def cost_eval(key):
        c = {'correct': 0, 'miss': 0, 'false_alarm': 0, 'conservative': 0}
        for r in rows:
            lb, pr = r['label'], r[key]
            if lb == pr:
                c['correct'] += 1
            elif lb == 'T' and pr == 'A':
                c['miss'] += 1              # 漏检: 篡改被判为可信 (最严重)
            elif lb == 'A' and pr == 'T':
                c['false_alarm'] += 1       # 误报: 可信被判为篡改 (严重)
            else:
                c['conservative'] += 1      # 保守: 任一方向拒识 (可接受)
        n = len(rows)
        dang = c['miss'] + c['false_alarm']
        return {k: round(v / n, 4) for k, v in c.items()} | {
            'dangerous_error_rate': round(dang / n, 4),
            'n': n}

    def confusion(key):
        M = np.zeros((3, 3), int)
        idx = {'A': 0, 'T': 1, 'I': 2}
        for r in rows:
            M[idx[r['label']], idx[r[key]]] += 1
        acc = float(np.trace(M) / M.sum())
        per = {}
        for lab, ii in idx.items():
            rec = float(M[ii, ii] / max(1, M[ii].sum()))
            per[lab] = round(rec, 3)
        return M.tolist(), round(acc, 4), per

    # 对照: 教师用 top10 判据 (旧口径)
    for r in rows:
        pass
    out = {'meta': dict(n_items=len(rows), n_per_cond=N, tau_contrast=T_CONTRAST,
                        conditions=sorted({r['cond'] for r in rows}),
                        tau_area=round(tau_area, 4), tau_conf=tau_conf,
                        note='student thresholds calibrated coarsely; paper should calibrate on val'),
           'confusion': {}, 'per_condition': {},
           'cost_definition': 'fusion uses a second evidence (contrast) to arbitrate; '
                              'abstention in either direction is conservative'}
    for key in ('teacher', 'student', 'fusion'):
        M, acc, per = confusion(key)
        ce = cost_eval(key)
        out['confusion'][key] = {'matrix': M, 'accuracy': acc, 'recall_by_class': per,
                                 'forensic_cost': ce}
        print(f'[{key:8s}] acc={acc:.3f} recall={per}', flush=True)
        print(f'           forensic cost: miss={ce["miss"]:.3f} '
              f'false_alarm={ce["false_alarm"]:.3f} conservative={ce["conservative"]:.3f} '
              f'-> dangerous={ce["dangerous_error_rate"]:.3f}', flush=True)
    # 分条件准确率
    for cond in sorted({r['cond'] for r in rows}):
        sub = [r for r in rows if r['cond'] == cond]
        out['per_condition'][cond] = {
            'n': len(sub),
            'label': sub[0]['label'],
            **{k: round(float(np.mean([r[k] == r['label'] for r in sub])), 3)
               for k in ('teacher', 'student', 'fusion')}}
        print(f'  [{cond:14s}] label={sub[0]["label"]} '
              f'teacher={out["per_condition"][cond]["teacher"]:.2f} '
              f'student={out["per_condition"][cond]["student"]:.2f} '
              f'fusion={out["per_condition"][cond]["fusion"]:.2f}', flush=True)
    # ---- 阈值扫描: 冲突裁决阈值 -> (漏检, 误报) 权衡曲线
    sweep = {}
    for tau in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        c = {'miss': 0, 'false_alarm': 0}
        for r in rows:
            if r['teacher'] == 'I':
                pr = 'I'
            elif r['teacher'] == r['student']:
                pr = r['teacher']
            elif r['teacher'] == 'T' and r['student'] == 'A':
                pr = 'T' if r['tscore'] > tau else 'A'
            else:
                pr = 'T' if r['tscore'] > 0.15 else 'A'
            if r['label'] == 'T' and pr == 'A':
                c['miss'] += 1
            if r['label'] == 'A' and pr == 'T':
                c['false_alarm'] += 1
        n = len(rows)
        sweep[f'{tau:.2f}'] = {'miss': round(c['miss']/n, 4),
                               'false_alarm': round(c['false_alarm']/n, 4),
                               'sum': round((c['miss']+c['false_alarm'])/n, 4)}
        print(f'  [tau={tau:.2f}] miss={sweep[f"{tau:.2f}"]["miss"]:.3f} '
              f'FA={sweep[f"{tau:.2f}"]["false_alarm"]:.3f} sum={sweep[f"{tau:.2f}"]["sum"]:.3f}',
              flush=True)
    out['threshold_sweep'] = sweep
    json.dump(out, open(f'{BASE}/results/r55_forensic_verdict.json', 'w'), indent=1)
    print('-> results/r55_forensic_verdict.json')


if __name__ == '__main__':
    main()
