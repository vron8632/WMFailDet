#!/usr/bin/env python3
"""TIFS 概念图 v4 — 数字口径统一为 δ=25 主结果 + 精致排版.

修正 v3 的问题:
  1. 图上数字与正文矛盾 (旧: student 0.739/0.758, fusion 0.819; 新: 0.9491)
  2. 排版留白不均、字号层级不清
  3. 缺少任务标签 (KeyFree-Loc)
输出: paper/tifs_zh/figs/cs_motivation.{pptx,pdf,png}, cs_framework.{pptx,pdf,png}
"""
import os, subprocess
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
FIG = f'{BASE}/paper/tifs_zh/figs'
SRC = f'{BASE}/paper/latex/figs/src'

INK = RGBColor(0x22, 0x28, 0x2E); INK2 = RGBColor(0x5B, 0x66, 0x70)
BG = RGBColor(0xFC, 0xFC, 0xFA); RULE = RGBColor(0xDD, 0xDF, 0xDB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
# 四族低饱和（边框, 浅填充, 图标色）+ 一个强调
BLUE = (RGBColor(0x4A, 0x6E, 0x8E), RGBColor(0xEC, 0xF1, 0xF6), RGBColor(0x3C, 0x5C, 0x78))
GREEN = (RGBColor(0x5D, 0x7F, 0x6A), RGBColor(0xED, 0xF3, 0xEF), RGBColor(0x44, 0x66, 0x51))
GOLD = (RGBColor(0xA8, 0x92, 0x5E), RGBColor(0xFA, 0xF6, 0xEB), RGBColor(0x8A, 0x74, 0x42))
ROSE = (RGBColor(0xA0, 0x72, 0x72), RGBColor(0xF8, 0xF0, 0xF0), RGBColor(0x86, 0x58, 0x58))
GREY = (RGBColor(0x9A, 0x9E, 0x9A), RGBColor(0xF4, 0xF4, 0xF2), RGBColor(0x76, 0x7A, 0x76))
ACCENT = RGBColor(0x2A, 0x4A, 0x66)


def slide(prs, w, h):
    prs.slide_width, prs.slide_height = Inches(w), Inches(h)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(w), Inches(h))
    bg.fill.solid(); bg.fill.fore_color.rgb = BG; bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def txt(s, x, y, w, h, text, fs=10, color=INK, bold=False, align=PP_ALIGN.LEFT,
        spacing=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0; tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.alignment = align
        p.font.size = Pt(fs); p.font.color.rgb = color; p.font.bold = bold
        if spacing:
            p.space_after = Pt(spacing)
    return tb


def card(s, x, y, w, h, head, body, foot, fam, icon=None, rule=0.9, hl=False, fs_h=11,
         fs_b=9, fs_f=8.5):
    """统一卡片: 左上图标 + 标题/描述 + 底部脚注(框外)."""
    e, f, ic = fam
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.06
    shp.fill.solid(); shp.fill.fore_color.rgb = f if hl else WHITE
    shp.line.color.rgb = e; shp.line.width = Pt(rule if hl else 0.85)
    shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.10)
    tf.margin_right = Inches(0.10); tf.margin_top = Inches(0.05); tf.margin_bottom = Inches(0.05)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = head; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(fs_h); p.font.bold = True; p.font.color.rgb = e if hl else INK
    if body:
        p2 = tf.add_paragraph(); p2.text = body; p2.alignment = PP_ALIGN.CENTER
        p2.font.size = Pt(fs_b); p2.font.color.rgb = INK2
    if icon is not None:
        i_s = s.shapes.add_shape(icon, Inches(x + 0.11), Inches(y + 0.10),
                                 Inches(0.26), Inches(0.26))
        i_s.fill.solid(); i_s.fill.fore_color.rgb = ic
        i_s.line.color.rgb = e; i_s.line.width = Pt(0.7)
    if foot:
        txt(s, x - 0.05, y + h + 0.06, w + 0.10, 0.26, foot, fs=fs_f, color=INK2,
            align=PP_ALIGN.CENTER)


def arrow(s, x1, y1, x2, y2, dash=False, w=1.3):
    a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                               Inches(x2), Inches(y2))
    a.line.color.rgb = RGBColor(0x8C, 0x92, 0x8E); a.line.width = Pt(w)
    ln = a.line._get_or_add_ln()
    if dash:
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    ln.append(ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}))


def rule(s, x, y, w, color=RULE, pw=0.75):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y),
                                Inches(x + w), Inches(y))
    ln.line.color.rgb = color; ln.line.width = Pt(pw)


def band(s, y, w=12.9, x=0.3):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.34))
    shp.adjustments[0] = 0.5
    shp.fill.solid(); shp.fill.fore_color.rgb = RGBColor(0xEE, 0xF2, 0xF5)
    shp.line.color.rgb = RGBColor(0xD6, 0xDE, 0xE4); shp.line.width = Pt(0.7)
    shp.shadow.inherit = False
    return shp


def pic(s, p, x, y, w, h, border=GREY[0]):
    sh = s.shapes.add_picture(p, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.line.color.rgb = border; sh.line.width = Pt(0.9)
    return sh


# =====================================================================
def motivation():
    prs = Presentation()
    s = slide(prs, 13.333, 6.42)

    # 标题区
    txt(s, 0.34, 0.20, 9.2, 0.34, 'One edited image yields free, pixel-accurate supervision',
        fs=17, color=INK, bold=True)
    txt(s, 0.34, 0.56, 12.4, 0.24,
        'KeyFree-Loc  —  training may use the secret key, inference may not',
        fs=10, color=ACCENT)

    # 证据链 6 列
    cols = [
        (f'{SRC}/p_host.png', 'Host', 'original photo', GREY),
        (f'{SRC}/p_edited.png', 'Generative edit', 'GT = red box', ROSE),
        (f'{SRC}/p_failmap.png', 'Failure map', 'white = failed block', BLUE),
        (f'{SRC}/p_teacher.png', 'Teacher alarm', 'keyed, zero training', BLUE),
        (f'{SRC}/p_student.png', 'Student (ours)', 'image only at inference', GREEN),
        (f'{SRC}/p_fusion.png', 'Fusion', 'verdict + region', GOLD),
    ]
    x0, y0, sz, gap = 0.40, 0.86, 1.90, 0.14
    for i, (p, title, sub, fam) in enumerate(cols):
        x = x0 + i * (sz + gap)
        pic(s, p, x, y0, sz, sz, fam[0])
        hl = i == 4
        shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y0 + sz + 0.05),
                                 Inches(sz), Inches(0.50))
        shp.adjustments[0] = 0.14
        e, f, _ = fam
        shp.fill.solid(); shp.fill.fore_color.rgb = f if hl else WHITE
        shp.line.color.rgb = e; shp.line.width = Pt(1.1 if hl else 0.8)
        shp.shadow.inherit = False
        tf = shp.text_frame; tf.word_wrap = True
        tf.margin_top = tf.margin_bottom = Inches(0.01)
        p1 = tf.paragraphs[0]; p1.text = title; p1.alignment = PP_ALIGN.CENTER
        p1.font.size = Pt(9.5); p1.font.bold = True; p1.font.color.rgb = e if hl else INK
        p2 = tf.add_paragraph(); p2.text = sub; p2.alignment = PP_ALIGN.CENTER
        p2.font.size = Pt(7.8); p2.font.color.rgb = INK2
        if i < 5:
            arrow(s, x + sz + 0.02, y0 + sz / 2, x + sz + gap - 0.02, y0 + sz / 2, w=1.1)

    # 机制带
    txt(s, 0.34, 3.44, 6.0, 0.24, 'why the failure map is a reliable label   (n = 190 COCO edits)',
        fs=9, color=INK2)
    rule(s, 0.34, 3.70, 12.6)
    stats = [('0.501', 'inside edit region', 'watermark dead (= chance)', BLUE),
             ('0.259', 'outside edit region', 'intact (= clean baseline)', GREY),
             ('0.249', 'clean image', 'reference', GREY),
             ('99.5 %', 'images: inside > outside', 'independent of content', GREEN),
             ('free', 'supervision', '0.2 s / image, zero labels', GOLD)]
    yy = 3.84
    for i, (val, name, note, fam) in enumerate(stats):
        x = 0.34 + i * 2.56
        card(s, x, yy, 2.36, 0.82, val, name + '\n' + note, '', fam,
             icon=MSO_SHAPE.OCTAGON if i < 3 else (MSO_SHAPE.HEXAGON if i == 3 else MSO_SHAPE.STAR_5_POINT),
             fs_h=13, fs_b=8.2)

    # 主结果带 (δ=25 口径统一)
    txt(s, 0.34, 4.94, 6.4, 0.24, 'main results   (KeyFree-Bench T1, n_test = 400, δ = 25)',
        fs=9, color=INK2)
    rule(s, 0.34, 5.20, 12.6)
    res = [('0.9491', 'Student (ours)  ±0.0012', 'image only · 1.93 M · 1.1 ms', GREEN, True),
           ('0.9272', 'EditGuard (same protocol)', 'needs reference watermark', ROSE, False),
           ('0.947', 'WAM (reported)', 'different test set', GREY, False),
           ('0.726', 'Teacher 4×4 layer', 'needs watermark + key', BLUE, False)]
    for i, (val, name, note, fam, hl) in enumerate(res):
        x = 0.34 + i * 3.20
        card(s, x, 5.34, 3.00, 0.82, val, name + '\n' + note, '', fam,
             icon=MSO_SHAPE.STAR_5_POINT if hl else None, hl=hl, fs_h=13, fs_b=8.2)
    prs.save(f'{FIG}/cs_motivation.pptx')
    print('-> cs_motivation.pptx')


# =====================================================================
def framework():
    prs = Presentation()
    s = slide(prs, 13.333, 4.85)

    txt(s, 0.34, 0.18, 9.8, 0.34,
        'A zero-training physical signal becomes free supervision', fs=17, color=INK, bold=True)
    b = band(s, 0.58)
    tf = b.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    p.text = ('KeyFree-Loc:  training may use the secret key  ·  inference sees the image only  —  '
              'no watermark, no key, no reference')
    p.font.size = Pt(9.5); p.font.color.rgb = ACCENT; p.font.bold = True

    # ---- 三阶段卡片
    bw, bh, y = 3.30, 1.14, 1.10
    xs = [0.42, 4.80, 9.18]
    card(s, xs[0], y, bw, bh, 'Teacher',
         'keyed QIM failure map · three layers (8/4/2)\nzero training',
         'needs watermark + key', BLUE, icon=MSO_SHAPE.SUN, fs_h=14, fs_b=9)
    card(s, xs[1], y, bw, bh, 'Free pseudo-labels',
         'pixel-accurate (2×2 layer) · 0.501 in / 0.259 out\nno human annotation',
         'the same map, reused', GREEN, icon=MSO_SHAPE.FRAME, hl=True, fs_h=14, fs_b=9)
    card(s, xs[2], y, bw, bh, 'Student',
         '1.93 M params · distilled on free labels\n1.1 ms / image (GPU)',
         'needs neither watermark nor key', GOLD, icon=MSO_SHAPE.PENTAGON, fs_h=14, fs_b=9)

    for i in range(2):
        arrow(s, xs[i] + bw + 0.06, y + bh / 2, xs[i + 1] - 0.08, y + bh / 2, w=1.6)
    txt(s, xs[0] + bw + 0.02, y + bh / 2 - 0.26, 0.86, 0.22, 'labels', fs=8.5,
        color=RGBColor(0x8C, 0x92, 0x8E), align=PP_ALIGN.CENTER)

    # ---- 每阶段下的真实数据缩略图 (信息密度)
    ty, tsz = y + bh + 0.30, 0.92
    thumbs = [
        (f'{SRC}/p_failmap.png', xs[0], 'failure map (real)', BLUE),
        (f'{SRC}/p_teacher.png', xs[0] + 1.02, 'alarm overlay', BLUE),
        (f'{SRC}/p_edited.png', xs[1], 'edited input', GREEN),
        (f'{SRC}/p_student.png', xs[2], 'prediction (real)', GOLD),
        (f'{SRC}/p_fusion.png', xs[2] + 1.02, 'fused verdict', GOLD),
    ]
    for path, x, cap, fam in thumbs:
        pic(s, path, x, ty, tsz, tsz, fam[0])
        txt(s, x - 0.06, ty + tsz + 0.03, tsz + 0.12, 0.20, cap, fs=7.2, color=INK2,
            align=PP_ALIGN.CENTER)

    # ---- 结果条
    rule(s, 0.34, ty + tsz + 0.36, 12.6)
    txt(s, 0.34, ty + tsz + 0.46, 12.4, 0.26,
        'Teacher 0.726 (keyed)   ·   Student 0.9491 ± 0.0012 (key-free)   ·   '
        'EditGuard 0.9272 (same protocol, needs watermark)   ·   WAM 0.947 (reported)',
        fs=11, color=INK)
    prs.save(f'{FIG}/cs_framework.pptx')
    print('-> cs_framework.pptx')


if __name__ == '__main__':
    motivation()
    framework()
    for name in ('cs_motivation', 'cs_framework'):
        subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', FIG,
                        f'{FIG}/{name}.pptx'], capture_output=True, timeout=180)
        import pymupdf
        d = pymupdf.open(f'{FIG}/{name}.pdf')
        d[0].get_pixmap(dpi=220).save(f'{FIG}/{name}.png')
        print('exported', name, '.pdf/.png')
