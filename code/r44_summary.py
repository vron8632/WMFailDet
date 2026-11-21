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

# ---------- 7.6 新机制探索 ----------
s = load('r47_aux_distill.json')
if s:
    lines += ['## 7.6 机制一：辅助任务蒸馏（r47）', '',
              '| λ (aux loss 权重) | val | test IoU |', '|---|---|---|']
    for k, v in s['runs'].items():
        lines.append(f"| {k.replace('lambda','')} | {f(v['val'])} | {f(v['test']['iou'])} |")
    lines.append('')
s = load('r48_iterative_refine.json')
if s:
    lines += ['## 7.7 机制二：教师-学生迭代精化（r48，负结果）', '',
              '| 轮次 | test IoU |', '|---|---|']
    for k, v in s['rounds'].items():
        lines.append(f"| {k} | {f(v['test']['iou'])} |")
    lines += ['', '结论：精化有害且逐轮恶化（误差同源 + 教师失效图含 25% 底噪）。', '']
s = load('r50_aux_fusion.json')
if s:
    lines += ['## 7.8 辅助学生的融合（r50）', '',
              f"- 辅助学生单用: {f(s['aux_student']['iou'])}",
              f"- 辅助学生融合: **{f(s['fusion']['iou'])}** (cfg a={s['fusion']['cfg']['a']}, "
              f"b={s['fusion']['cfg']['b']}, tau={s['fusion']['cfg']['tau']})",
              f"- 对照: baseline 学生 0.760 / baseline 融合 **0.819** / 教师 0.726",
              '- 解读：辅助任务提升单模型但与教师互补性下降（融合 0.819->0.802）。', '']
s = load('r49_mechanism.json')
if s:
    lines += ['## 7.9 机制分析（r49）', '',
              f"- 编辑区内失效率 **{s['in_rate_mean']:.3f}** vs 区外 **{s['out_rate_mean']:.3f}**"
              f"（干净 {s['clean_rate_mean']:.3f}），分离度 {s['separation_mean']:.3f}",
              f"- 逐图一致性（区内>区外）：**{s['consistency_frac_in_gt_out']:.1%}**（n={s['meta']['n']}）",
              f"- 区内失效率 vs 结构重写率相关：r={s['corr_spatialcorr_vs_inrate']:.3f} "
              f"(p={s['corr_p_spatialcorr']:.2f}) → 局部编辑下失效**饱和且与内容无关**，"
              f"这正是伪标签可靠的原因",
              '', '| 编辑面积档 | n | 区内 | 区外 | 分离度 |', '|---|---|---|---|---|']
    for k, v in s['by_area'].items():
        lines.append(f"| {k} | {v['n']} | {f(v['in_rate'])} | {f(v['out_rate'])} | {f(v['sep'])} |")
    lines.append('')

# ---------- 7.95 取证结论与归因 (新赛道) ----------
s = load('r55_forensic_verdict.json')
if s:
    lines += ['## 7.95 取证结论（三级输出, r55）', '',
              '| 方法 | 总体准确 | 漏检(T→A) | 误报(A→T) | 保守拒识 | 危险错误 |',
              '|---|---|---|---|---|---|']
    for k in ('teacher', 'student', 'fusion'):
        c = s['confusion'][k]
        fc = c['forensic_cost']
        lines.append(f"| {k} | {f(c['accuracy'])} | {f(fc['miss'])} | {f(fc['false_alarm'])} | "
                     f"{f(fc['conservative'])} | **{f(fc['dangerous_error_rate'])}** |")
    lines += ['', '阈值扫描（冲突裁决 τ）:', '',
              '| τ | 漏检 | 误报 | 合计 |', '|---|---|---|---|']
    for k, v in s['threshold_sweep'].items():
        lines.append(f"| {k} | {f(v['miss'])} | {f(v['false_alarm'])} | {f(v['sum'])} |")
    lines.append('')
s = load('r56_edit_attribution.json')
s2 = load('r57_attribution_features.json')
if s or s2:
    lines += ['## 7.96 编辑类型归因（r56/r57，负结果）', '']
    if s:
        lines += ['| 配置 | 域内 acc | 说明 |', '|---|---|---|']
        for k, v in s['in_domain'].items():
            lines.append(f"| {k} | {f(v['acc'])} | 小 CNN |")
    if s2:
        lines += ['', '| 分类器(8 个形状特征) | 域内 acc | AUC |', '|---|---|---|']
        for k, v in s2['models'].items():
            lines.append(f"| {k} | {f(v['in_domain']['acc'])} | {f(v['in_domain']['auc'])} |")
    lines += ['', '结论：失效图只携带**空间证据**（区域/强度/可信度），**不携带编辑语义**'
              '（remove/replace 不可分，AUC≈0.5）。归因需语义模型，超出本信号能力。', '']

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
