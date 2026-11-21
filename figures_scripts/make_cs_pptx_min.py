#!/usr/bin/env python3
"""极简版概念图 (只画创新点, 不画完整流程).

motivation: 3 张真实图 + 一行核心数字
framework: 3 个框 (零训练教师 -> 免费标签 -> 无密钥学生), 两侧标注需求差异
"""
import os, subprocess
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f'{BASE}/paper/latex/figs'
SRC = f'{FIG}/src'

INK = RGBColor(0x2B, 0x2B, 0x2B)
INK2 = RGBColor(0x5A, 0x5A, 0x5A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BG = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT = RGBColor(0x2F, 0x4F, 0x6F)
ACCENT_L = RGBColor(0xE8, 0xEE, 0xF3)
RED = RGBColor(0xA8, 0x4B, 0x4B)
RED_L = RGBColor(0xF4, 0xE8, 0xE8)
# 多色低饱和族 (边框, 填充, 图标)
BLUE = (RGBColor(0x5B, 0x7C, 0x99), RGBColor(0xE9, 0xF0, 0xF5), RGBColor(0x5B, 0x7C, 0x99))
GREEN = (RGBColor(0x6E, 0x8F, 0x77), RGBColor(0xEA, 0xF2, 0xEC), RGBColor(0x4E, 0x7A, 0x59))
GOLD = (RGBColor(0xB8, 0xA0, 0x6A), RGBColor(0xFB, 0xF6, 0xE8), RGBColor(0x8E, 0x75, 0x40))
GREY = (RGBColor(0xA0, 0xA0, 0x9C), RGBColor(0xF2, 0xF2, 0xF0), RGBColor(0x7A, 0x7A, 0x76))

from pptx.enum.shapes import MSO_SHAPE as _MS

def card(s, x, y, w, h, head, body, foot, fam, icon, accent=False):
    e, f, ic = fam
    shp = s.shapes.add_shape(_MS.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                             Inches(w), Inches(h))
    shp.adjustments[0] = 0.05
    shp.fill.solid(); shp.fill.fore_color.rgb = f
    shp.line.color.rgb = e; shp.line.width = Pt(1.1 if accent else 0.9)
    shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.5); tf.margin_right = Inches(0.1)
    tf.vertical_anchor = _MS_ANCH.MIDDLE
    p = tf.paragraphs[0]; p.text = head; p.alignment = _PP.CENTER
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = ACCENT if accent else INK
    p2 = tf.add_paragraph(); p2.text = body; p2.alignment = _PP.CENTER
    p2.font.size = Pt(10); p2.font.color.rgb = INK2
    ic_shp = s.shapes.add_shape(icon, Inches(x + 0.06), Inches(y + 0.10),
                                Inches(0.34), Inches(0.34))
    ic_shp.fill.solid(); ic_shp.fill.fore_color.rgb = ic
    ic_shp.line.color.rgb = e; ic_shp.line.width = Pt(0.8)
    txt(s, x, y + h + 0.05, w, 0.3, foot, fs=9.5, color=INK2, align=_PP.CENTER)


def slide(prs, h, w=13.333):
    prs.slide_width = Inches(w); prs.slide_height = Inches(h)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(w), Inches(h))
    bg.fill.solid(); bg.fill.fore_color.rgb = BG; bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def txt(s, x, y, w, h, text, fs=11, color=INK, bold=False, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text; p.alignment = align
    p.font.size = Pt(fs); p.font.color.rgb = color; p.font.bold = bold
    return tb


def pic(s, path, x, y, size, edge=RGBColor(0xD8, 0xD8, 0xD8)):
    p = s.shapes.add_picture(path, Inches(x), Inches(y), Inches(size), Inches(size))
    p.line.color.rgb = edge; p.line.width = Pt(0.75)
    return p


def arrow(s, x1, y1, x2, y2, color=RGBColor(0x9A, 0x9A, 0x9A), w=1.8):
    a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                               Inches(x2), Inches(y2))
    a.line.color.rgb = color; a.line.width = Pt(w)
    ln = a.line._get_or_add_ln()
    ln.append(ln.makeelement(qn('a:tailEnd'),
                             {'type': 'triangle', 'w': 'med', 'len': 'med'}))
    return a


# =====================================================================
def motivation():
    """3 张图 + 一行数字. 核心: 失效位置 = 篡改位置."""
    prs = Presentation()
    s = slide(prs, h=4.15)
    size = 3.0
    y = 0.30
    xs = [0.55, 3.95, 7.35]
    pic(s, f'{SRC}/p_host.png', xs[0], y, size)
    pic(s, f'{SRC}/p_edited.png', xs[1], y, size)
    pic(s, f'{SRC}/p_failmap.png', xs[2], y, size)
    for i, lab in enumerate(['host image', 'generative edit (GT: red)',
                             'watermark failure map\n(white = failed block)']):
        txt(s, xs[i], y + size + 0.04, size, 0.5, lab, fs=10.5, color=INK2,
            align=PP_ALIGN.CENTER)
    for i in range(2):
        arrow(s, xs[i] + size + 0.06, y + size / 2, xs[i + 1] - 0.10, y + size / 2)

    # 右侧: 核心数字 (两行)
    tx = 10.75
    txt(s, tx, y + 0.25, 2.4, 0.3, 'failure rate', fs=10, color=INK2, bold=True)
    txt(s, tx, y + 0.62, 2.4, 0.45, '0.501', fs=20, color=ACCENT, bold=True)
    txt(s, tx, y + 1.02, 2.4, 0.3, 'inside the edit', fs=10, color=INK2)
    txt(s, tx, y + 1.62, 2.4, 0.45, '0.259', fs=20, color=INK2, bold=True)
    txt(s, tx, y + 2.02, 2.4, 0.3, 'outside (baseline 0.249)', fs=10, color=INK2)
    txt(s, tx, y + 2.42, 2.4, 0.3, '99.5 % of images', fs=10, color=ACCENT, bold=True)

    txt(s, 0.55, y + size + 0.62, 9.6, 0.34,
        'The edit destroys the keyed watermark exactly where it acts \u2014 so the failure map '
        'is a free, pixel-accurate label.',
        fs=11, color=INK)
    prs.save(f'{FIG}/cs_motivation.pptx')
    print('-> cs_motivation.pptx')


# =====================================================================
def framework():
    """3 个框: 零训练教师 -> 免费标签 -> 无密钥学生. 只画创新点."""
    prs = Presentation()
    s = slide(prs, h=3.35)
    bw, bh = 3.25, 1.55
    y = 0.55
    xs = [0.55, 4.75, 8.95]

    def card(s, x, y, w, h, head, body, foot, fam, icon, accent=False):
        e, f, ic = fam
        shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
        shp.adjustments[0] = 0.05
        shp.fill.solid()
        shp.fill.fore_color.rgb = f if accent else WHITE
        shp.line.color.rgb = e
        shp.line.width = Pt(1.3 if accent else 1.0)
        shp.shadow.inherit = False
        tf = shp.text_frame; tf.word_wrap = True
        tf.margin_left = Inches(0.52); tf.margin_right = Inches(0.12)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.text = head; p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(13); p.font.bold = True
        p.font.color.rgb = e if accent else INK
        p2 = tf.add_paragraph(); p2.text = body; p2.alignment = PP_ALIGN.CENTER
        p2.font.size = Pt(10); p2.font.color.rgb = INK2
        ic_shp = s.shapes.add_shape(icon, Inches(x + 0.07), Inches(y + 0.13),
                                    Inches(0.36), Inches(0.36))
        ic_shp.fill.solid(); ic_shp.fill.fore_color.rgb = ic
        ic_shp.line.color.rgb = e; ic_shp.line.width = Pt(0.8)
        txt(s, x, y + h + 0.05, w, 0.3, foot, fs=9.5, color=INK2, align=PP_ALIGN.CENTER)

    card(s, xs[0], y, bw, bh, 'Teacher', 'keyed QIM failure map\nzero training',
         'needs watermark + key', BLUE, _MS.SUN)
    card(s, xs[1], y, bw, bh, 'Free pseudo-labels', 'pixel-accurate, 0.2 s/image\nno human annotation',
         'the same map, reused', GREEN, _MS.FRAME, accent=True)
    card(s, xs[2], y, bw, bh, 'Student', '1.93 M params, distilled\nimage only at inference',
         'needs neither watermark nor key', GOLD, _MS.PENTAGON)

    for i in range(2):
        arrow(s, xs[i] + bw + 0.08, y + bh / 2, xs[i + 1] - 0.10, y + bh / 2)
    txt(s, xs[0] + bw + 0.02, y + bh / 2 - 0.30, 0.9, 0.24, 'labels', fs=9,
        color=RGBColor(0x8A, 0x8A, 0x8A), align=PP_ALIGN.CENTER)

    txt(s, 0.55, y + bh + 0.50, 12.2, 0.34,
        'Contribution: a zero-training physical signal becomes free supervision, '
        'so the detector is distributable without any key.',
        fs=11, color=INK)

    # 关键数字 (小字, 右侧一行)
    txt(s, 0.55, y + bh + 0.90, 12.2, 0.3,
        'teacher 0.726 (keyed)   ·   student 0.739\u00b10.052 (key-free)   ·   '
        'fusion 0.819   ·   GT-label upper bound 0.903',
        fs=10, color=INK2)
    prs.save(f'{FIG}/cs_framework.pptx')
    print('-> cs_framework.pptx')


def to_pdf_png():
    for name in ('cs_motivation', 'cs_framework'):
        subprocess.run(['soffice', '--headless', '--convert-to', 'pdf',
                        '--outdir', FIG, f'{FIG}/{name}.pptx'], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            import pymupdf
            d = pymupdf.open(f'{FIG}/{name}.pdf')
            d[0].get_pixmap(dpi=300).save(f'{FIG}/{name}.png')
        except Exception as e:
            print('png skip', e)


if __name__ == '__main__':
    motivation(); framework(); to_pdf_png()
