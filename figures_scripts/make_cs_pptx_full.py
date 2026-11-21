#!/usr/bin/env python3
"""C&S/TIFS 版概念图 v3 — 顶会标准:
  - 真实数据集图片嵌入
  - MSO_SHAPE 图标 (key/cloud/lightning/sun/gear/verdict)
  - 多色低饱和度配色族 (蓝/绿/沙/金/玫瑰/灰)
  - motivation: 6 列真实图 + 机制数据 + 关键结果
  - framework: 四层 (publication/training/inference/verdict)
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f'{BASE}/paper/latex/figs'
SRC = f'{FIG}/src'

INK = RGBColor(0x2B, 0x2B, 0x2B); INK2 = RGBColor(0x5A, 0x5A, 0x5A)
BG = RGBColor(0xFB, 0xFB, 0xF9)
# 色族 (边框, 填充, 图标)
BLUE = (RGBColor(0x5B, 0x7C, 0x99), RGBColor(0xE9, 0xF0, 0xF5), RGBColor(0x5B, 0x7C, 0x99))
GREEN = (RGBColor(0x6E, 0x8F, 0x77), RGBColor(0xEA, 0xF2, 0xEC), RGBColor(0x4E, 0x7A, 0x59))
ORANGE = (RGBColor(0xC4, 0x8A, 0x5A), RGBColor(0xFA, 0xF0, 0xE6), RGBColor(0xB5, 0x65, 0x1D))
GOLD = (RGBColor(0xB8, 0xA0, 0x6A), RGBColor(0xFB, 0xF6, 0xE8), RGBColor(0x8E, 0x75, 0x40))
ROSE = (RGBColor(0xB0, 0x85, 0x85), RGBColor(0xF7, 0xEE, 0xEE), RGBColor(0x9A, 0x60, 0x60))
GREY = (RGBColor(0x9C, 0x9C, 0x98), RGBColor(0xF2, 0xF2, 0xF0), RGBColor(0x7A, 0x7A, 0x76))
VIOLET = (RGBColor(0x8B, 0x7C, 0xA8), RGBColor(0xF1, 0xEE, 0xF7), RGBColor(0x6B, 0x5B, 0x95))
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def new_slide(prs, w=13.333, h=7.5):
    prs.slide_width = Inches(w); prs.slide_height = Inches(h)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(w), Inches(h))
    bg.fill.solid(); bg.fill.fore_color.rgb = BG; bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def box(s, x, y, w, h, lines, fam=BLUE, fs=10.5, icon=None, icon_size=0.3,
        first_bold=True, align=PP_ALIGN.CENTER):
    e, f, ic = fam
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                             Inches(w), Inches(h))
    shp.adjustments[0] = 0.07
    shp.fill.solid(); shp.fill.fore_color.rgb = f
    shp.line.color.rgb = e; shp.line.width = Pt(1.0)
    shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.5 if icon else 0.08)
    tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    if isinstance(lines, str):
        lines = [lines]
    for i, t in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = t; p.alignment = align
        p.font.size = Pt(fs); p.font.color.rgb = INK if i == 0 else INK2
        p.font.bold = first_bold and i == 0
    if icon:
        ic_shp = s.shapes.add_shape(icon, Inches(x - 0.14), Inches(y + 0.08),
                                    Inches(icon_size), Inches(icon_size))
        ic_shp.fill.solid(); ic_shp.fill.fore_color.rgb = ic
        ic_shp.line.color.rgb = e; ic_shp.line.width = Pt(0.8)
        ic_shp.rotation = 0
    return shp


def label(s, x, y, w, h, text, fs=10, color=INK2, bold=False, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.alignment = align
        p.font.size = Pt(fs); p.font.color.rgb = color; p.font.bold = bold
    return tb


def arrow(s, x1, y1, x2, y2, color=RGBColor(0x8A, 0x8A, 0x86), w=1.5, dash=False):
    a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                               Inches(x2), Inches(y2))
    a.line.color.rgb = color; a.line.width = Pt(w)
    ln = a.line._get_or_add_ln()
    if dash:
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    ln.append(ln.makeelement(qn('a:tailEnd'),
                             {'type': 'triangle', 'w': 'med', 'len': 'med'}))
    return a


def pic(s, path, x, y, size, border=GREY[0], w=None, h=None):
    p = s.shapes.add_picture(path, Inches(x), Inches(y),
                             Inches(w if w else size), Inches(h if h else size))
    p.line.color.rgb = border; p.line.width = Pt(1.0)
    return p


def section(s, y, text):
    label(s, 0.25, y, 5.0, 0.24, text, fs=9.5, color=RGBColor(0x8A, 0x8A, 0x86))
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.25),
                                Inches(y + 0.27), Inches(13.1), Inches(y + 0.27))
    ln.line.color.rgb = RGBColor(0xD8, 0xD8, 0xD3); ln.line.width = Pt(0.75)
    ln.line._get_or_add_ln().append(
        ln.line._get_or_add_ln().makeelement(qn('a:prstDash'), {'val': 'sysDot'}))


def stat_card(s, x, y, w, h, val, name, note, fam):
    box(s, x, y, w, h, [val, name, note], fam, fs=9.5, first_bold=True)


# =====================================================================
def motivation():
    prs = Presentation(); s = new_slide(prs, h=7.0)
    label(s, 0.3, 0.15, 12.7, 0.4,
          'One edited image, two independent evidence types, one calibrated verdict',
          fs=16, color=INK, bold=True)

    cols = [
        (f'{SRC}/p_host.png', 'Host', 'original photo', GREY, MSO_SHAPE.RECTANGLE),
        (f'{SRC}/p_edited.png', 'Generative edit', 'GT = red box', ROSE, MSO_SHAPE.LIGHTNING_BOLT),
        (f'{SRC}/p_failmap.png', 'Failure map', 'white = failed block', BLUE, MSO_SHAPE.RECTANGLE),
        (f'{SRC}/p_teacher.png', 'Teacher alarm', 'keyed, zero training', BLUE, MSO_SHAPE.HEXAGON),
        (f'{SRC}/p_student.png', 'Student (ours)', 'key-free inference', GREEN, MSO_SHAPE.PENTAGON),
        (f'{SRC}/p_fusion.png', 'Fusion', 'verdict + region', GOLD, MSO_SHAPE.GEAR_6),
    ]
    x0, y0, size, gap = 0.42, 0.70, 1.86, 0.155
    for i, (path, title, sub, fam, ic) in enumerate(cols):
        x = x0 + i * (size + gap)
        pic(s, path, x, y0, size, fam[0])
        e, f, _ = fam
        cap = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                                 Inches(y0 + size + 0.03), Inches(size), Inches(0.54))
        cap.adjustments[0] = 0.12
        cap.fill.solid(); cap.fill.fore_color.rgb = f
        cap.line.color.rgb = e; cap.line.width = Pt(0.75)
        cap.shadow.inherit = False
        tf = cap.text_frame; tf.word_wrap = True
        tf.margin_top = tf.margin_bottom = Inches(0.01)
        p1 = tf.paragraphs[0]; p1.text = title; p1.alignment = PP_ALIGN.CENTER
        p1.font.size = Pt(9.5); p1.font.bold = True; p1.font.color.rgb = INK
        p2 = tf.add_paragraph(); p2.text = sub; p2.alignment = PP_ALIGN.CENTER
        p2.font.size = Pt(8); p2.font.color.rgb = INK2
        arrow(s, x + size + 0.02, y0 + size/2, x + size + gap - 0.02, y0 + size/2, w=1.3)

    # 机制层
    section(s, 3.72, 'why the failure map is a reliable label  (n = 190 COCO edits)')
    yy = 4.10
    stats = [('0.501', 'inside edit region', 'watermark dead (= chance)', BLUE),
             ('0.259', 'outside edit region', 'intact (= clean baseline)', GREY),
             ('0.249', 'clean image', 'reference', GREY),
             ('99.5 %', 'images: inside > outside', 'and independent of content', GREEN),
             ('free', 'supervision', '0.2 s / image, zero labels', GOLD)]
    for i, (val, name, note, fam) in enumerate(stats):
        x = 0.42 + i * 2.58
        box(s, x, yy, 2.42, 1.0, [val, name, note], fam, fs=9.5,
            icon=MSO_SHAPE.OCTAGON if i < 3 else MSO_SHAPE.STAR_5_POINT, icon_size=0.26)

    # 关键结果
    section(s, 5.42, 'key results  (COCO-256, n_test = 200)')
    yy = 5.78
    nums = [('0.726', 'Teacher 4×4', 'watermark + key', BLUE),
            ('0.758', 'Student (ours)', 'image only, 1.93 M', GREEN),
            ('0.819', 'Fusion', 'best overall', GOLD),
            ('0.903', 'GT-label upper', 'same architecture', GREY),
            ('0.107', 'Dangerous error', 'vs teacher 0.214', ROSE)]
    for i, (val, name, note, fam) in enumerate(nums):
        x = 0.42 + i * 2.58
        box(s, x, yy, 2.42, 1.0, [val, name, note], fam, fs=9.5,
            icon=MSO_SHAPE.STAR_5_POINT if i == 2 else None, icon_size=0.26)
    prs.save(f'{FIG}/cs_motivation_full.pptx')
    print('-> cs_motivation_full.pptx')


# =====================================================================
def framework():
    prs = Presentation(); s = new_slide(prs, h=7.5)
    label(s, 0.3, 0.14, 12.7, 0.38,
          'Watermark failure as free supervision: zero-training teacher → key-free student → evidence-chain verdict',
          fs=15, color=INK, bold=True)

    # ---- L1 publication
    section(s, 0.58, 'publication (one-time, owner side)')
    y = 0.95
    box(s, 0.45, y, 3.0, 0.85, ['Owner embeds keyed QIM', 'per-image key, zero training'],
        BLUE, icon=MSO_SHAPE.DIAMOND)
    box(s, 3.85, y, 2.4, 0.85, ['Watermarked image', 'published as usual'], GREY,
        icon=MSO_SHAPE.CLOUD)
    box(s, 6.65, y, 2.7, 0.85, ['Distribution', 'generative edits + degradations'],
        ORANGE, icon=MSO_SHAPE.LIGHTNING_BOLT)
    pic(s, f'{SRC}/p_host.png', 9.75, y - 0.03, 0.86)
    pic(s, f'{SRC}/p_edited.png', 10.75, y - 0.03, 0.86)
    arrow(s, 9.35, y + 0.42, 9.72, y + 0.42, w=1.3)
    label(s, 11.68, y + 0.24, 1.5, 0.3, 'audit target', fs=8.5)

    # ---- L2 training
    section(s, 2.12, 'training (offline; labels are free — zero human annotation)')
    y = 2.50
    box(s, 0.45, y, 3.0, 0.95, ['Teacher: zero-training', 'keyed QIM failure map',
                               'needs watermark + key'], BLUE, icon=MSO_SHAPE.SUN)
    pic(s, f'{SRC}/p_failmap.png', 3.68, y - 0.02, 0.9, BLUE[0])
    arrow(s, 3.50, y + 0.47, 3.68, y + 0.47, w=1.2)
    arrow(s, 4.62, y + 0.47, 4.85, y + 0.47, w=1.2)
    box(s, 4.85, y, 2.75, 0.95, ['Pseudo-labels (free)', '0.501 inside / 0.259 outside',
                                '99.5 % consistent'], GREEN, icon=MSO_SHAPE.FRAME)
    box(s, 7.95, y, 2.8, 0.95, ['Student distillation', '1.93 M params',
                               'no human labels'], GREEN, icon=MSO_SHAPE.PENTAGON)
    arrow(s, 7.60, y + 0.47, 7.95, y + 0.47, w=1.2)
    pic(s, f'{SRC}/p_student.png', 10.95, y - 0.02, 0.9, GREEN[0])
    arrow(s, 10.75, y + 0.47, 10.95, y + 0.47, w=1.2)

    # ---- L3 inference
    section(s, 3.80, 'inference (image only — no watermark, no key, no reference)')
    y = 4.18
    box(s, 0.45, y, 2.0, 0.9, ['Query image', 'any circulating copy'], GREY,
        icon=MSO_SHAPE.DIAMOND)
    box(s, 2.85, y, 3.0, 0.9, ['Student (key-free)', 'image only', 'GPU 1.1 ms / CPU 35.7 ms'],
        GREEN, icon=MSO_SHAPE.PENTAGON)
    box(s, 6.25, y, 3.2, 0.9, ['Evidence-chain fusion', 'student (visual) + teacher (checks)',
                              'conflict arbitrated by clustering'], GOLD,
        icon=MSO_SHAPE.GEAR_6)
    box(s, 9.85, y, 2.5, 0.9, ['Teacher (optional)', 'requires key'], BLUE,
        icon=MSO_SHAPE.SUN)
    arrow(s, 2.45, y + 0.45, 2.85, y + 0.45)
    arrow(s, 5.85, y + 0.45, 6.25, y + 0.45)
    arrow(s, 9.45, y + 0.45, 9.85, y + 0.45, dash=True, w=1.2)
    label(s, 9.05, y - 0.28, 1.2, 0.25, 'optional', fs=8.5, color=RGBColor(0x99,0x99,0x94))

    # ---- L4 verdict
    section(s, 5.42, 'verdict & evidence profile')
    y = 5.78
    box(s, 0.45, y, 3.5, 1.05,
        ['Verdict', 'AUTHENTIC / TAMPERED / INDETERMINATE'], GOLD,
        icon=MSO_SHAPE.STAR_5_POINT)
    box(s, 4.30, y, 3.3, 1.05,
        ['Evidence profile', 'failure clustering · geometry', 'attack trace'], GOLD,
        icon=MSO_SHAPE.FRAME)
    box(s, 7.95, y, 2.4, 1.05, ['Error profile', 'miss 0.016 · FA 0.091'], ROSE,
        icon=MSO_SHAPE.OCTAGON)
    box(s, 10.7, y, 2.2, 1.05, ['Region', 'alarm mask + IoU'], GOLD,
        icon=MSO_SHAPE.RECTANGLE)
    arrow(s, 3.95, y + 0.52, 4.30, y + 0.52)
    arrow(s, 7.75, y + 0.52, 7.95, y + 0.52)
    arrow(s, 10.35, y + 0.52, 10.70, y + 0.52)
    prs.save(f'{FIG}/cs_framework_full.pptx')
    print('-> cs_framework_full.pptx')


if __name__ == '__main__':
    motivation()
    framework()
