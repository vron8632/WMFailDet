#!/usr/bin/env python3
"""C&S 版概念图 v2 — 真实数据图片 + 低饱和度莫兰迪配色 + 可编辑形状.

输出: paper/latex/figs/cs_motivation.pptx, cs_framework.pptx
     (并用 soffice 转 PDF/PNG)
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

# 莫兰迪低饱和配色
INK = RGBColor(0x2B, 0x2B, 0x2B)
INK2 = RGBColor(0x5A, 0x5A, 0x5A)
BG = RGBColor(0xFA, 0xFA, 0xF8)
# 色族: (边框, 填充)
BLUE = (RGBColor(0x6E, 0x86, 0x9C), RGBColor(0xEA, 0xEF, 0xF3))
GREEN = (RGBColor(0x7C, 0x97, 0x84), RGBColor(0xEC, 0xF2, 0xEE))
SAND = (RGBColor(0xC2, 0xA9, 0x8B), RGBColor(0xF7, 0xF0, 0xE8))
GOLD = (RGBColor(0xB8, 0xA0, 0x6A), RGBColor(0xFA, 0xF5, 0xE7))
ROSE = (RGBColor(0xB8, 0x91, 0x91), RGBColor(0xF8, 0xEE, 0xEE))
GREY = (RGBColor(0xA8, 0xA8, 0xA4), RGBColor(0xF3, 0xF3, 0xF0))


def new_slide(prs, w=13.333, h=6.9):
    prs.slide_width = Inches(w); prs.slide_height = Inches(h)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(w), Inches(h))
    bg.fill.solid(); bg.fill.fore_color.rgb = BG; bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def box(s, x, y, w, h, lines, fam=BLUE, fs=10.5, first_bold=True,
        align=PP_ALIGN.CENTER, radius=0.06, shadow=False):
    e, f = fam
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                             Inches(w), Inches(h))
    shp.adjustments[0] = radius
    shp.fill.solid(); shp.fill.fore_color.rgb = f
    shp.line.color.rgb = e; shp.line.width = Pt(1.0)
    shp.shadow.inherit = shadow
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    if isinstance(lines, str):
        lines = [lines]
    for i, t in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = t; p.alignment = align
        p.font.size = Pt(fs); p.font.color.rgb = INK if i == 0 else INK2
        p.font.bold = first_bold and i == 0
    return shp


def label(s, x, y, w, h, text, fs=10, color=INK2, bold=False, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text; p.alignment = align
    p.font.size = Pt(fs); p.font.color.rgb = color; p.font.bold = bold
    return tb


def arrow(s, x1, y1, x2, y2, color=RGBColor(0x8A, 0x8A, 0x86), w=1.6, dash=False):
    a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                               Inches(x2), Inches(y2))
    a.line.color.rgb = color; a.line.width = Pt(w)
    ln = a.line._get_or_add_ln()
    if dash:
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    ln.append(ln.makeelement(qn('a:tailEnd'),
                             {'type': 'triangle', 'w': 'med', 'len': 'med'}))
    return a


def pic(s, path, x, y, size, border=GREY[0]):
    p = s.shapes.add_picture(path, Inches(x), Inches(y), Inches(size), Inches(size))
    p.line.color.rgb = border; p.line.width = Pt(1.0)
    return p


def section(s, y, text, w=13.2):
    label(s, 0.2, y, 4.0, 0.24, text, fs=9.5, color=RGBColor(0x8A, 0x8A, 0x86))
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.2),
                                Inches(y + 0.26), Inches(w), Inches(y + 0.26))
    ln.line.color.rgb = RGBColor(0xD5, 0xD5, 0xD0); ln.line.width = Pt(0.75)
    ln.line._get_or_add_ln().append(
        ln.line._get_or_add_ln().makeelement(qn('a:prstDash'), {'val': 'sysDot'}))


# =====================================================================
def motivation():
    prs = Presentation()
    s = new_slide(prs)
    label(s, 0.3, 0.18, 12.7, 0.4,
          'One edited image, two independent evidence types, one calibrated verdict',
          fs=16, color=INK, bold=True)

    # ---- 上排: 真实图片 6 列
    cols = [
        (f'{SRC}/p_host.png', 'Host', 'original photo', GREY),
        (f'{SRC}/p_edited.png', 'Generative edit', 'GT = red box', ROSE),
        (f'{SRC}/p_failmap.png', 'Failure map', 'white = failed block', BLUE),
        (f'{SRC}/p_teacher.png', 'Teacher alarm', 'keyed, zero training', BLUE),
        (f'{SRC}/p_student.png', 'Student (ours)', 'key-free inference', GREEN),
        (f'{SRC}/p_fusion.png', 'Fusion', 'verdict + region', GOLD),
    ]
    x0, y0, size, gap = 0.45, 0.72, 1.85, 0.16
    for i, (path, title, sub, fam) in enumerate(cols):
        x = x0 + i * (size + gap)
        pic(s, path, x, y0, size)
        e, f = fam
        cap = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                                 Inches(y0 + size + 0.02), Inches(size), Inches(0.52))
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

    for i in range(len(cols) - 1):
        x = x0 + (i + 1) * (size + gap) - gap / 2
        arrow(s, x - 0.07, y0 + size / 2, x + 0.07, y0 + size / 2, w=1.4)

    # ---- 中部: 机制 (为什么失效图是可靠标签)
    section(s, 3.62, 'why the failure map is a reliable label  (n = 190 COCO edits)')
    yy = 4.0
    stats = [('inside edit region', '0.501', 'watermark dead (= chance)', BLUE),
             ('outside edit region', '0.259', 'intact (= clean baseline)', GREY),
             ('clean image', '0.249', 'reference', GREY)]
    for i, (name, val, note, fam) in enumerate(stats):
        x = 0.45 + i * 2.35
        box(s, x, yy, 2.2, 0.95, [f'{val}', name, note], fam, fs=9.5)
    box(s, 7.6, yy, 2.35, 0.95,
        ['99.5 %', 'images: inside > outside', 'and independent of content'],
        GREEN, fs=9.5)
    box(s, 10.1, yy, 2.85, 0.95,
        ['free supervision', '0.2 s / image, zero human labels',
         'quality > noisy human boxes'], GREEN, fs=9.5)

    # ---- 下部: 三个数字 (关键结果)
    section(s, 5.25, 'key results  (COCO-256, n_test = 200)')
    yy = 5.65
    nums = [('Teacher 4×4', '0.726', 'watermark + key', BLUE),
            ('Student (ours)', '0.758', 'image only, 1.93 M', GREEN),
            ('Fusion', '0.819', 'best overall', GOLD),
            ('GT-label upper', '0.903', 'same architecture', GREY),
            ('Dangerous error', '0.107', 'vs teacher 0.214', ROSE)]
    for i, (name, val, note, fam) in enumerate(nums):
        x = 0.45 + i * 2.55
        box(s, x, yy, 2.38, 0.95, [f'{val}', name, note], fam, fs=9.5)
    prs.save(f'{FIG}/cs_motivation.pptx')
    print('-> paper/latex/figs/cs_motivation.pptx')


# =====================================================================
def framework():
    prs = Presentation()
    s = new_slide(prs, h=6.6)
    label(s, 0.3, 0.10, 12.7, 0.35,
          'Watermark failure as free supervision: zero-training teacher → key-free student → evidence-chain verdict',
          fs=14.5, color=INK, bold=True)

    # ---- L1 publication
    section(s, 0.52, 'publication (one-time, owner side)')
    y = 0.88
    box(s, 0.45, y, 2.85, 0.78, ['Owner embeds keyed QIM', 'per-image key, zero training'], BLUE)
    box(s, 3.66, y, 2.25, 0.78, ['Watermarked image', 'published as usual'], GREY)
    box(s, 6.27, y, 2.5, 0.78, ['Distribution', 'edits + degradations'], SAND)
    arrow(s, 3.30, y+0.39, 3.66, y+0.39)
    arrow(s, 5.91, y+0.39, 6.27, y+0.39)
    pic(s, f'{SRC}/p_host.png', 9.15, y-0.03, 0.84)
    pic(s, f'{SRC}/p_edited.png', 10.12, y-0.03, 0.84)
    arrow(s, 8.77, y+0.39, 9.12, y+0.39, w=1.2)
    label(s, 11.10, y+0.24, 1.6, 0.3, 'audit target', fs=8.5)

    # ---- L2 training
    section(s, 1.90, 'training (offline; labels are free)')
    y = 2.26
    box(s, 0.45, y, 2.9, 0.85, ['Teacher: zero-training', 'keyed QIM failure map',
                               'needs watermark + key'], BLUE)
    arrow(s, 3.35, y+0.42, 3.62, y+0.42, w=1.2)
    pic(s, f'{SRC}/p_failmap.png', 3.62, y-0.02, 0.86, BLUE[0])
    arrow(s, 4.50, y+0.42, 4.78, y+0.42, w=1.2)
    box(s, 4.78, y, 2.6, 0.85, ['Pseudo-labels (free)', '0.501 inside / 0.259 outside',
                               '99.5 % consistent'], GREEN)
    arrow(s, 7.38, y+0.42, 7.66, y+0.42, w=1.2)
    box(s, 7.66, y, 2.45, 0.85, ['Student distillation', '1.93 M params', 'no human labels'], GREEN)
    arrow(s, 10.11, y+0.42, 10.38, y+0.42, w=1.2)
    pic(s, f'{SRC}/p_student.png', 10.38, y-0.02, 0.86, GREEN[0])

    # ---- L3 inference
    section(s, 3.32, 'inference (image only — no watermark, no key, no reference)')
    y = 3.68
    box(s, 0.45, y, 2.0, 0.85, ['Query image', 'any circulating copy'], GREY)
    box(s, 2.80, y, 2.85, 0.85, ['Student (key-free)', 'image only', 'GPU 1.1 ms / CPU 35.7 ms'], GREEN)
    box(s, 6.00, y, 3.15, 0.85, ['Evidence-chain fusion', 'conflict arbitrated by failure clustering',
                                'τ = 0.20'], GOLD)
    box(s, 9.50, y, 2.55, 0.85, ['Teacher (optional)', 'requires key'], BLUE)
    arrow(s, 2.45, y+0.42, 2.80, y+0.42)
    arrow(s, 5.65, y+0.42, 6.00, y+0.42)
    arrow(s, 9.15, y+0.42, 9.50, y+0.42, dash=True, w=1.2)
    label(s, 9.15, y-0.26, 1.2, 0.24, 'optional', fs=8.5, color=RGBColor(0x99,0x99,0x94))

    # ---- L4 verdict
    section(s, 4.78, 'verdict & evidence profile')
    y = 5.14
    box(s, 0.45, y, 3.5, 0.95, ['Verdict', 'AUTHENTIC / TAMPERED / INDETERMINATE'], GOLD)
    box(s, 4.25, y, 3.2, 0.95, ['Evidence profile', 'clustering · geometry · attack trace'], GOLD)
    box(s, 7.75, y, 2.35, 0.95, ['Error profile', 'miss 0.016 · FA 0.091'], ROSE)
    box(s, 10.40, y, 2.5, 0.95, ['Region', 'alarm mask + localization IoU'], GOLD)
    arrow(s, 3.95, y+0.47, 4.25, y+0.47)
    arrow(s, 7.45, y+0.47, 7.75, y+0.47)
    arrow(s, 10.10, y+0.47, 10.40, y+0.47)
    prs.save(f'{FIG}/cs_framework.pptx')
    print('-> paper/latex/figs/cs_framework.pptx')


def to_pdf_png():
    for name in ('cs_motivation', 'cs_framework'):
        src = f'{FIG}/{name}.pptx'
        subprocess.run(['soffice', '--headless', '--convert-to', 'pdf',
                        '--outdir', FIG, src], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pdf = f'{FIG}/{name}.pdf'
        try:
            import pymupdf
            d = pymupdf.open(pdf)
            d[0].get_pixmap(dpi=300).save(f'{FIG}/{name}.png')
        except Exception as e:
            print('png skip', e)
        print('->', pdf)


if __name__ == '__main__':
    motivation()
    framework()
    to_pdf_png()
