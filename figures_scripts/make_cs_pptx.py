#!/usr/bin/env python3
"""C&S 版两张概念图的可编辑 PPT:
  slide 1: motivation (教师/学生/融合 三列对比, 同一张图)
  slide 2: framework (四层: 发布/训练/推理/取证结论)
输出: figs/cs_motivation.pptx, figs/cs_framework.pptx
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f'{BASE}/figs'
os.makedirs(OUT, exist_ok=True)

INK = RGBColor(0x1A, 0x1A, 0x1A)
GREY_E, GREY_F = RGBColor(0x9A, 0x9A, 0x9A), RGBColor(0xF4, 0xF4, 0xF4)
BLU_E, BLU_F = RGBColor(0x0F, 0x4D, 0x92), RGBColor(0xE4, 0xED, 0xF9)
GRN_E, GRN_F = RGBColor(0x2F, 0x6B, 0x3F), RGBColor(0xE8, 0xF3, 0xE9)
ORA_E, ORA_F = RGBColor(0xB5, 0x65, 0x1D), RGBColor(0xFC, 0xF1, 0xE4)
GLD_E, GLD_F = RGBColor(0xB8, 0x8A, 0x1A), RGBColor(0xFD, 0xF3, 0xD9)
RED = RGBColor(0xB6, 0x43, 0x42)


def add_box(slide, x, y, w, h, lines, fill, line, fs=11, bold_first=True,
            align=PP_ALIGN.CENTER):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line; shp.line.width = Pt(1.6)
    shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    if isinstance(lines, str):
        lines = [lines]
    for i, t in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = t; p.alignment = align
        p.font.size = Pt(fs); p.font.color.rgb = INK
        p.font.bold = bold_first and i == 0
    return shp


def add_label(slide, x, y, w, text, fs=10, color=RGBColor(0x66, 0x66, 0x66),
              bold=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.3))
    tf = tb.text_frame; tf.word_wrap = False
    p = tf.paragraphs[0]; p.text = text; p.alignment = align
    p.font.size = Pt(fs); p.font.color.rgb = color; p.font.bold = bold
    return tb


def add_arrow(slide, x1, y1, x2, y2, color=None, width=2.2, dashed=False):
    c = color or RGBColor(0x55, 0x55, 0x55)
    a = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                   Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    a.line.color.rgb = c; a.line.width = Pt(width)
    ln = a.line._get_or_add_ln()
    if dashed:
        d = ln.makeelement(qn('a:prstDash'), {'val': 'dash'})
        ln.append(d)
    ln.append(ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}))
    return a


def add_section(slide, y, label, total_w=13.333):
    add_label(slide, 0.15, y - 0.06, 3.2, label, fs=10,
              color=RGBColor(0x77, 0x77, 0x77))
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.15),
                                    Inches(y + 0.28), Inches(total_w - 0.15),
                                    Inches(y + 0.28))
    ln.line.color.rgb = RGBColor(0xAA, 0xAA, 0xAA); ln.line.width = Pt(0.75)
    ln.line._get_or_add_ln().append(
        ln.line._get_or_add_ln().makeelement(qn('a:prstDash'), {'val': 'sysDot'}))


from pptx.oxml.ns import qn  # noqa: E402


# =====================================================================
def motivation():
    prs = Presentation(); prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_label(s, 0.2, 0.15, 12.9, 'Motivation — one image, two evidence types, one verdict',
              fs=15, color=INK, bold=True)

    y0, h0 = 0.75, 2.35
    # 列 1: host+edit+GT (视觉)
    add_box(s, 0.3, y0, 2.55, h0,
            ['Generative edit', 'obj replaced / removed', 'bbox + feathering'],
            GREY_F, GREY_E, fs=11)
    # 列 2: 教师
    add_box(s, 3.15, y0, 2.55, h0,
            ['Teacher evidence', 'keyed QIM failure map',
             'needs watermark + key', 'zero training'],
            BLU_F, BLU_E, fs=11)
    # 列 3: 学生
    add_box(s, 6.0, y0, 2.55, h0,
            ['Student (distilled)', 'image only at inference',
             'no watermark / no key', '1.93M params, GPU 1.1 ms'],
            GRN_F, GRN_E, fs=11)
    # 列 4: 融合
    add_box(s, 8.85, y0, 2.55, h0,
            ['Evidence-chain fusion', 'visual + checks arbitrated',
             'conflict -> contrast vote', 'miss 0.016 / FA 0.091'],
            GLD_F, GLD_E, fs=11)
    # 列 5: 结论
    add_box(s, 11.7, y0, 1.45, h0,
            ['Verdict', 'A / T / I', '+ region'], GLD_F, GLD_E, fs=11)

    ymid = y0 + h0 / 2
    add_arrow(s, 2.85, ymid, 3.15, ymid)
    add_arrow(s, 5.70, ymid, 6.00, ymid)
    add_arrow(s, 8.55, ymid, 8.85, ymid)
    add_arrow(s, 11.40, ymid, 11.70, ymid)

    # 下半: 数字与卖点
    y1 = 3.4
    add_label(s, 0.3, y1 - 0.4, 4.5, 'What the failure map is (mechanism, n=190):',
              fs=11, color=INK, bold=True)
    rows = [
        ('inside edit', 0.501, RGBColor(0x0F, 0x4D, 0x92)),
        ('outside edit', 0.259, RGBColor(0x76, 0x76, 0x76)),
        ('clean', 0.249, RGBColor(0xBB, 0xBB, 0xBB)),
    ]
    bx = 0.3
    for name, v, c in rows:
        add_box(s, bx, y1, 2.3, 0.55, [f'{name}: {v:.3f}'], RGBColor(0xFF, 0xFF, 0xFF),
                c, fs=11, bold_first=False)
        bx += 2.5
    add_label(s, 0.3, y1 + 0.62, 8.0,
              '99.5% of images satisfy inside > outside  |  independent of image content '
              '(r=0.10, p=0.16)  ->  reliable labels', fs=10.5, color=RGBColor(0x2F, 0x6B, 0x3F),
              bold=True)

    add_label(s, 0.3, 4.55, 5.5, 'Key numbers (COCO-256, n_test=200):', fs=11,
              color=INK, bold=True)
    nums = [('teacher', 0.726, 'keyed'), ('student', 0.758, 'key-free'),
            ('fusion', 0.819, 'best'), ('GT-label upper', 0.903, 'reference')]
    bx = 0.3
    for name, v, note in nums:
        add_box(s, bx, 4.9, 2.5, 0.8, [name, f'{v:.3f}', note], RGBColor(0xFF, 0xFF, 0xFF),
                RGBColor(0x99, 0x99, 0x99), fs=11, bold_first=False)
        bx += 2.7
    add_label(s, 0.3, 5.95, 12.0,
              'Free labels beat human-level noisy boxes 0.775 vs 0.602   |   '
              'dangerous error (miss+FA): fusion 0.107 vs teacher 0.214', fs=11,
              color=RGBColor(0x2F, 0x6B, 0x3F), bold=True)
    add_label(s, 0.3, 6.5, 12.0,
              'Scale: 250 -> 500 -> 800 free labels = 0.469 / 0.615 / 0.775 (not saturated)',
              fs=10.5, color=RGBColor(0x55, 0x55, 0x55))
    prs.save(f'{OUT}/cs_motivation.pptx')
    print('-> figs/cs_motivation.pptx')


# =====================================================================
def framework():
    prs = Presentation(); prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    s = prs.slides.add_slide(prs.slide_layouts[6])

    W = 13.333
    # ---- L1 publication
    add_section(s, 0.12, 'publication')
    y = 0.45
    add_box(s, 0.3, y, 2.6, 0.85, ['Owner: embed keyed QIM', 'one-time, zero training'],
            BLU_F, BLU_E, fs=11)
    add_box(s, 3.4, y, 2.1, 0.85, ['Publish', 'watermarked image'], GREY_F, GREY_E, fs=11)
    add_box(s, 6.0, y, 2.7, 0.85, ['Distribution', 'generative edits', '+ degradations'],
            ORA_F, ORA_E, fs=11)
    add_arrow(s, 2.9, y+0.42, 3.4, y+0.42)
    add_arrow(s, 5.5, y+0.42, 6.0, y+0.42)
    add_arrow(s, 8.7, y+0.42, 9.2, y+0.42)
    add_label(s, 9.25, y+0.3, 1.2, 'audit', fs=10)

    # ---- L2 training (free labels)
    add_section(s, 1.55, 'training (offline, free labels)')
    y = 1.9
    add_box(s, 0.3, y, 2.9, 0.95,
            ['Teacher (zero-training)', 'keyed QIM failure map', 'needs watermark + key'],
            BLU_F, BLU_E, fs=11)
    add_box(s, 3.7, y, 2.6, 0.95,
            ['Pseudo-labels (free)', 'pixel-accurate', '99.5% consistent'],
            GRN_F, GRN_E, fs=11)
    add_box(s, 6.8, y, 2.9, 0.95,
            ['Student distillation', '1.93M params', 'no human labels'],
            GRN_F, GRN_E, fs=11)
    add_arrow(s, 3.2, y+0.47, 3.7, y+0.47)
    add_arrow(s, 6.3, y+0.47, 6.8, y+0.47)

    # ---- L3 inference (image only)
    add_section(s, 3.1, 'inference (image only)')
    y = 3.45
    add_box(s, 0.3, y, 1.7, 0.9, ['Query', 'image'], GREY_F, GREY_E, fs=11)
    add_box(s, 2.4, y, 2.7, 0.9,
            ['Student (key-free)', 'image only', 'GPU 1.1 ms / CPU 35.7 ms'],
            GRN_F, GRN_E, fs=11)
    add_box(s, 5.5, y, 3.1, 0.9,
            ['Evidence-chain fusion', 'visual + checks, arbitrated', 'tau = 0.20'],
            GLD_F, GLD_E, fs=11)
    add_box(s, 9.0, y, 2.1, 0.9,
            ['Teacher', 'optional, keyed'], BLU_F, BLU_E, fs=11)
    add_arrow(s, 2.0, y+0.45, 2.4, y+0.45)
    add_arrow(s, 5.1, y+0.45, 5.5, y+0.45)
    a = add_arrow(s, 8.6, y+0.45, 9.0, y+0.45, color=RGBColor(0x99, 0x99, 0x99), dashed=True)
    add_label(s, 8.35, y-0.32, 1.6, 'optional', fs=9.5, color=RGBColor(0x88, 0x88, 0x88))

    # ---- L4 verdict & evidence chain
    add_section(s, 4.75, 'verdict & evidence chain')
    y = 5.1
    add_box(s, 2.2, y, 4.2, 0.9,
            ['Verdict', 'AUTHENTIC / TAMPERED / INDETERMINATE', '+ region + evidence profile'],
            GLD_F, GLD_E, fs=11)
    add_arrow(s, 7.05, y+0.45, 7.5, y+0.45)
    add_box(s, 7.5, y, 3.4, 0.9,
            ['Evidence profile', 'failure clustering / geometry', 'consistency / attack trace'],
            GLD_F, GLD_E, fs=11)
    add_label(s, 10.7, y+0.3, 2.5, 'dangerous error\n0.107 (teacher 0.214)', fs=10,
              color=RGBColor(0x2F, 0x6B, 0x3F), bold=True)

    prs.save(f'{OUT}/cs_framework.pptx')
    print('-> figs/cs_framework.pptx')


if __name__ == '__main__':
    motivation()
    framework()
