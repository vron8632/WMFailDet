#!/usr/bin/env python3
"""r44: 汇总全部 D2 实验结果为 markdown 表 (写论文直接引用).

输出: results/SUMMARY_D2.md
"""
import os, json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f'{BASE}/results'


def load(n):
    p = f'{R}/{n}'
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, k=3):
    return f'{x:.{k}f}' if isinstance(x, (int, float)) else str(x)


lines = ['# D2 蒸馏闭环 — 实验结果汇总', '',
         '> 生成脚本: `code/r44_summary.py`；全部数字可追溯到 `results/*.json`', '']

# ---------- 1. 主矩阵 ----------
d = load('r34_distill_matrix.json')
if d:
    lines += ['## 1. 主矩阵（COCO-256, train800/val200/test200）', '',
              '| 组 | IoU | 覆盖≥50% | 面积比 | 备注 |', '|---|---|---|---|---|']
    for k, v in d['teachers'].items():
        lines.append(f"| 教师 {k} | {f(v['iou'])} | {v['cov50']:.1%} | {f(v['area_ratio'],2)} | 需水印+密钥 |")
    for k, v in d['students'].items():
        note = 'base16 学生（被动）' if 'alarm4' in k else 'GT 上界（标签质量参照）'
        lines.append(f"| {k} | {f(v['iou'])} | {v['cov50']:.1%} | — | {note} |")
    for k, v in d['fusion_or'].items():
        lines.append(f"| {k} | {f(v['iou'])} | {v['cov50']:.1%} | {f(v['area_ratio'],2)} | 学生∪教师 |")
    lines.append('')

# ---------- 2. 模型容量 ----------
s = load('r43_student_improve.json')
if s:
    lines += ['## 2. 学生配置消融（r43）', '', '| 配置 | test IoU | 说明 |', '|---|---|---|']
    desc = {'A_baseline': 'base16, 80ep（= r34 口径）', 'B_swa': '+ SWA',
            'C_aug': '+ 数据增强', 'D_base32': 'base32（主配置）', 'E_full': '增强+160ep+SWA'}
    for k, v in s['variants'].items():
        lines.append(f"| {k} | {f(v['iou'])} | {desc.get(k, '')} |")
    lines += [f"", f"**最佳: {s['best']['tag']} = {s['best']['test_iou']}**", '']

# ---------- 3. 多种子 + 融合 ----------
s = load('r41_seeds_fusion.json')
if s:
    lines += ['## 3. 多种子与融合（r41, base32）', '']
    sm = s['seeds'].get('summary')
    if sm:
        lines.append(f"- 3 seeds: **{sm['mean']:.4f} ± {sm['std']:.4f}** "
                     f"(min {sm['min']:.4f} / max {sm['max']:.4f})")
    fu = s.get('fusion', {})
    if fu:
        g = fu['grid_best_on_val']
        lines += ['', f"| 组 | test IoU |", '|---|---|',
                  f"| 学生（3-seed 概率平均） | {f(fu['student_only_test']['iou'])} |",
                  f"| 教师 4×4 | {f(fu['teacher_only_test']['iou'])} |",
                  f"| 硬 OR 融合 | {f(fu['hard_or_test']['iou'])} |",
                  f"| **网格融合** (a={g['a_student']}, b={g['b_teacher']}, τ={g['tau']}) "
                  f"| **{f(fu['test']['iou'])}** |"]
    ed = s.get('error_decomposition', {}).get('per_sample')
    if ed:
        lines += ['', f"- 逐样本胜负: 学生 **{ed['student_wins']}** / 教师 {ed['teacher_wins']} "
                      f"/ 平 {ed['ties']}（n={ed['n']}）"]
    lines.append('')

# ---------- 4. 跨域 ----------
s = load('r39_crossdomain_training.json')
s1 = load('r35_crossdomain.json')
if s:
    lines += ['## 4. 跨域（r35 单域学生 vs r39 多域联合）', '',
              '| 域 | 单域学生 | **多域联合** | 域内微调(≤50) | 教师 4×4 |',
              '|---|---|---|---|---|']
    coco = s['joint'].get('coco', {})
    lines.append(f"| COCO | {f(s1['coco']['student']['iou']) if s1 else '—'} | "
                 f"{f(coco.get('iou'))} | — | {f(s1['coco']['teacher4']['iou']) if s1 else '—'} |")
    for k in ['butterfly', 'sat', 'doc', 'pokemon']:
        cand = [x for x in s['joint'] if k in x]
        ck = cand[0] if cand else k
        fk = [x for x in (s['finetune'] or {}) if k in x]
        t4 = None
        if s1:
            dk = [x for x in s1['domains'] if k in x]
            if dk:
                t4 = s1['domains'][dk[0]]['teacher4']['iou']
        lines.append(f"| {k} | "
                     f"{f(s1['domains'][dk[0]]['student']['iou']) if s1 and dk else '—'} | "
                     f"**{f(s['joint'][ck]['iou'])}** | "
                     f"{f(s['finetune'][fk[0]]['iou']) if fk else '—'} | {f(t4)} |")
    lines.append('')

# ---------- 5. 退化 ----------
s = load('r36_degradation.json')
if s:
    lines += ['## 5. 退化鲁棒性（r36, base32）', '',
              '| 退化 | 学生 | 教师(拒识=0) | 教师拒识率 | 教师可判定子集 |', '|---|---|---|---|---|']
    for k, v in s['results'].items():
        lines.append(f"| {k} | **{f(v['student']['iou'])}** | {f(v['teacher4_conservative']['iou'])} | "
                     f"{v['teacher_abstain_rate']:.0%} | {f(v['teacher4_detectable_subset']['iou'])} |")
    lines.append('')

# ---------- 6. 擦除攻击 ----------
s = load('r40_erasure_attack.json')
if s:
    lines += ['## 6. 系数恢复攻击（r40）', '',
              f"- 教师: {f(s['teacher_original']['iou'])} → **{f(s['teacher_attacked']['iou'])}**",
              f"- 学生: {f(s['student_original']['iou'])} → **{f(s['student_attacked']['iou'])}**",
              f"- 攻击视觉代价 (vs 编辑图): PSNR {f(s['attack_visual_cost'],2)} dB", '']

# ---------- 7. 标签消融 ----------
s = load('r38_label_ablation.json')
if s:
    lines += ['## 7. 标签消融与数据规模（r38）', '', '| 标签 | test IoU |', '|---|---|']
    for k, v in s.get('labels', {}).items():
        lines.append(f"| {k} | {f(v.get('iou_test'))} |")
    if s.get('scaling'):
        lines += ['', '| 训练样本数 | test IoU |', '|---|---|']
        for k, v in s['scaling'].items():
            lines.append(f"| {k} | {f(v.get('iou_test'))} |")
    lines.append('')

# ---------- 7.5 多编辑器联合 ----------
s = load('r45_multieditor.json')
if s:
    lines += ['## 7.5 多编辑器联合训练（r45）', '',
              f"- 联合训练集: {s['meta']['n_joint']} (COCO800 + LaMa140 + hires98)",
              '| 域 | 单域学生(r42) | 多编辑器联合 | 教师 4×4 |', '|---|---|---|---|',
              f"| COCO | — | {f(s['joint']['coco']['iou'])} | — |",
              f"| LaMa | 0.265 | **{f(s['joint']['lama']['iou'])}** | 0.745 |",
              f"| hires | 0.246 | **{f(s['joint']['hires']['iou'])}** | 0.819 |", '']

# ---------- 8. 覆盖与成本 ----------
s = load('r42_coverage_cost.json')
if s:
    lines += ['## 8. 编辑类型覆盖与成本（r42）', '', '| 域 | n | 学生 | 教师 4×4 |', '|---|---|---|---|']
    for k, v in s.get('domains', {}).items():
        lines.append(f"| {k} | {v['n']} | {f(v['student']['iou'])} | {f(v['teacher4']['iou'])} |")
    c = s.get('cost', {})
    if c:
        lines += ['', f"- 参数量: {c['params']/1e6:.2f}M ({c['params_mb']} MB)",
                  f"- 推理: GPU {c['gpu_ms_per_image']} ms / CPU {c['cpu_ms_per_image']} ms"]
    lines.append('')

open(f'{R}/SUMMARY_D2.md', 'w').write('\n'.join(lines))
print('-> results/SUMMARY_D2.md')
print('\n'.join(lines[:20]))
