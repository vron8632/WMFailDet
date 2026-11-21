#!/usr/bin/env python3
"""TIFS/C&S 概念图 v7 — 遵循 nature-figure 的 vector-schematic 规则:

  Type    : 一种字族, 三档字号; 只有组名加粗; 全部文字同一深墨色 (不用灰字)
  Colour  : 白画布; 两个饱和填充 (深蓝=本文新组件, 赭=外部); accent 唯一;
            其余白底+灰细边框; 红只用于错误/瓶颈
  Lines   : 每边有箭头, 主路径最粗; 实线=数据流; 连接器接边中点
  Struct  : 去卡片化, 留白+细线分栏; 等间距对齐; 禁止阴影/渐变

输出: paper/cs_zh/figs/cs_{motivation,framework}.{pptx,pdf,png}
"""
import os, subprocess
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
FIG = f'{BASE}/paper/cs_zh/figs'
SRC = f'{BASE}/paper/latex/figs/src'

# --- 墨与线（唯一深色 + 两级灰线）
INK = RGBColor(0x1A, 0x1D, 0x20)
CANVAS = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xB9, 0xBE, 0xC2)      # 细边框
LINE_SOFT = RGBColor(0xD8, 0xDB, 0xDE)  # 分隔线
# --- 两个饱和填充: 本文组件=深蓝; 限制/对照=赭
BLUE = RGBColor(0x1F, 0x4E, 0x79)
BLUE_PALE = RGBColor(0xEC, 0xF2, 0xF7)
OCHRE = RGBColor(0xB0, 0x7A, 0x45)
OCHRE_PALE = RGBColor(0xF9, 0xF4, 0xEC)
RED = RGBColor(0xA5, 0x2A, 0x2A)        # 只用于瓶颈/错误
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FS_TITLE, FS_HEAD, FS_BODY, FS_SMALL = 15, 11, 8.5, 7.5


def slide(prs, w, h):
    prs.slide_width, prs.slide_height = Inches(w), Inches(h)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(w), Inches(h))
    bg.fill.solid(); bg.fill.fore_color.rgb = CANVAS; bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def txt(s, x, y, w, h, text, fs=FS_BODY, bold=False, align=PP_ALIGN.LEFT,
        color=INK, line_spacing=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.alignment = align
        p.font.size = Pt(fs); p.font.bold = bold; p.font.color.rgb = color
        if line_spacing:
            p.line_spacing = line_spacing
    return tb


def frame(s, x, y, w, h, filled=None, edge=LINE, lw=0.75):
    """白底或浅填充的矩形框：不用圆角、不用阴影。"""
    shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = filled if filled is not None else WHITE
    shp.line.color.rgb = edge; shp.line.width = Pt(lw)
    shp.shadow.inherit = False
    return shp


def card(s, x, y, w, h, title, lines, accent=False, foot=None, pale=None):
    """名称(常规或加粗) + 一行职责说明；文字全部同一墨色。"""
    frame(s, x, y, w, h, filled=(pale or WHITE),
          edge=(BLUE if accent else LINE), lw=(1.1 if accent else 0.75))
    tb = s.shapes.add_textbox(Inches(x + 0.10), Inches(y + 0.08),
                              Inches(w - 0.20), Inches(h - 0.16))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = title; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(FS_HEAD); p.font.bold = True
    p.font.color.rgb = BLUE if accent else INK
    for line in lines:
        q = tf.add_paragraph(); q.text = line; q.alignment = PP_ALIGN.CENTER
        q.font.size = Pt(FS_BODY); q.font.color.rgb = INK
        q.space_before = Pt(1.5)
    if foot:
        txt(s, x, y + h + 0.05, w, 0.22, foot, fs=FS_SMALL, align=PP_ALIGN.CENTER)


def arrow(s, x1, y1, x2, y2, heavy=False, dashed=False):
    a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                               Inches(x2), Inches(y2))
    a.line.color.rgb = INK if heavy else RGBColor(0x6E, 0x74, 0x78)
    a.line.width = Pt(1.5 if heavy else 0.9)
    ln = a.line._get_or_add_ln()
    if dashed:
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    ln.append(ln.makeelement(qn('a:tailEnd'),
                             {'type': 'triangle', 'w': 'sm' if not heavy else 'med',
                              'len': 'sm' if not heavy else 'med'}))


def hline(s, x, y, w, soft=True):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y),
                                Inches(x + w), Inches(y))
    ln.line.color.rgb = LINE_SOFT if soft else LINE
    ln.line.width = Pt(0.75)


def pic(s, path, x, y, w, h):
    sh = s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.line.color.rgb = LINE; sh.line.width = Pt(0.6)
    return sh


# =====================================================================
def motivation():
    prs = Presentation(); s = slide(prs, 13.333, 6.10)

    txt(s, 0.42, 0.20, 11.6, 0.34,
        'One edited image yields free, pixel-accurate supervision', fs=FS_TITLE, bold=True)
    txt(s, 0.42, 0.56, 12.4, 0.24,
        'KeyFree-Loc — training may use the secret key; inference may not', fs=9)

    # 证据链：图片 + 名称(无框, 细线分隔)
    cols = [(f'{SRC}/p_host.png', 'Host', 'original photo'),
            (f'{SRC}/p_edited.png', 'Generative edit', 'GT = red box'),
            (f'{SRC}/p_failmap.png', 'Failure map', 'white = failed block'),
            (f'{SRC}/p_teacher.png', 'Teacher alarm', 'keyed, zero training'),
            (f'{SRC}/p_student.png', 'Student (ours)', 'image only at inference'),
            (f'{SRC}/p_fusion.png', 'Fusion', 'verdict + region')]
    x0, y0, sz, gap = 0.42, 0.86, 1.86, 0.19
    for i, (p, title, sub) in enumerate(cols):
        x = x0 + i * (sz + gap)
        pic(s, p, x, y0, sz, sz)
        hl = (i == 4)
        txt(s, x - 0.04, y0 + sz + 0.07, sz + 0.08, 0.20, title, fs=9, bold=True,
            align=PP_ALIGN.CENTER, color=(BLUE if hl else INK))
        txt(s, x - 0.04, y0 + sz + 0.27, sz + 0.08, 0.20, sub, fs=FS_SMALL,
            align=PP_ALIGN.CENTER)
        if i < 5:
            arrow(s, x + sz + 0.03, y0 + sz / 2, x + sz + gap - 0.03, y0 + sz / 2)

    # 机制：数字行（无框、无填充，靠字号与细线）
    hline(s, 0.42, 3.30, 12.5, soft=False)
    txt(s, 0.42, 3.40, 8.0, 0.22,
        'Why the failure map is a reliable label (n = 190 COCO edits)', fs=8.5)
    stats = [('0.501', 'inside edit region', 'watermark dead (= chance)'),
             ('0.259', 'outside edit region', 'intact (= clean baseline)'),
             ('0.249', 'clean image', 'reference'),
             ('99.5%', 'inside > outside', 'independent of content'),
             ('free', 'supervision', '0.2 s / image, zero labels')]
    for i, (val, name, note) in enumerate(stats):
        x = 0.42 + i * 2.56
        txt(s, x, 3.70, 2.30, 0.34, val, fs=16, bold=True,
            color=(BLUE if i == 3 else INK))
        txt(s, x, 4.06, 2.30, 0.20, name, fs=FS_BODY)
        txt(s, x, 4.26, 2.30, 0.20, note, fs=FS_SMALL)

    # 主结果：数字行（学生唯一样式区分）
    hline(s, 0.42, 4.62, 12.5, soft=False)
    txt(s, 0.42, 4.72, 9.0, 0.22,
        'Main results (KeyFree-Bench T1, n_test = 400, δ = 25)', fs=8.5)
    res = [('0.9491', 'Student (ours) ±0.0012', 'image only · 1.93 M · 1.1 ms', True),
           ('0.9272', 'EditGuard (same protocol)', 'needs reference watermark', False),
           ('0.947', 'WAM (reported)', 'different test set', False),
           ('0.726', 'Teacher, 4×4 layer', 'needs watermark + key', False)]
    for i, (val, name, note, hl) in enumerate(res):
        x = 0.42 + i * 3.18
        txt(s, x, 5.02, 2.95, 0.34, val, fs=16, bold=True, color=(BLUE if hl else INK))
        txt(s, x, 5.38, 2.95, 0.20, name, fs=FS_BODY, bold=hl)
        txt(s, x, 5.58, 2.95, 0.20, note, fs=FS_SMALL)
    prs.save(f'{FIG}/cs_motivation.pptx')
    print('-> cs_motivation.pptx')


# =====================================================================
def framework():
    prs = Presentation(); s = slide(prs, 13.333, 4.55)

    txt(s, 0.42, 0.18, 11.6, 0.32,
        'A zero-training physical signal becomes free supervision', fs=FS_TITLE, bold=True)
    txt(s, 0.42, 0.54, 12.4, 0.22,
        'KeyFree-Loc: training may use the secret key · inference sees the image only — '
        'no watermark, no key, no reference', fs=8.5)

    # 三阶段（中间=本文新组件: 唯一 accent）
    bw, bh, y = 3.24, 1.02, 0.94
    xs = [0.42, 4.86, 9.30]
    card(s, xs[0], y, bw, bh, 'Teacher',
         ['keyed QIM failure map · layers 8/4/2', 'zero training'], foot='needs watermark + key')
    card(s, xs[1], y, bw, bh, 'Free pseudo-labels',
         ['2×2 layer · 0.501 in / 0.259 out', 'no human annotation'],
         accent=True, pale=BLUE_PALE, foot='the same map, reused')
    card(s, xs[2], y, bw, bh, 'Student',
         ['1.93 M params · distilled on free labels', '1.1 ms / image (GPU)'],
         foot='needs neither watermark nor key')
    for i in range(2):
        arrow(s, xs[i] + bw + 0.05, y + bh / 2, xs[i + 1] - 0.05, y + bh / 2, heavy=(i == 1))
    txt(s, xs[0] + bw + 0.02, y + bh / 2 - 0.30, 0.80, 0.20, 'labels', fs=FS_SMALL,
        align=PP_ALIGN.CENTER)

    # 真实数据缩略图（各阶段下方，等间距）
    ty, tsz = y + bh + 0.36, 0.96
    thumbs = [(f'{SRC}/p_failmap.png', xs[0] + 0.10, 'failure map (real)'),
              (f'{SRC}/p_teacher.png', xs[0] + 1.16, 'alarm overlay'),
              (f'{SRC}/p_edited.png', xs[1] + 0.10, 'edited input'),
              (f'{SRC}/p_host.png', xs[1] + 1.16, 'host'),
              (f'{SRC}/p_student.png', xs[2] + 0.10, 'prediction (real)'),
              (f'{SRC}/p_fusion.png', xs[2] + 1.16, 'fused verdict')]
    for p, x, cap in thumbs:
        pic(s, p, x, ty, tsz, tsz)
        txt(s, x - 0.06, ty + tsz + 0.03, tsz + 0.12, 0.18, cap, fs=6.8,
            align=PP_ALIGN.CENTER)

    # 底部：结果行（细线分隔，唯一 accent 标本文结果）
    by = ty + tsz + 0.34
    hline(s, 0.42, by, 12.5, soft=False)
    nums = [('0.726', 'Teacher (keyed)'), ('0.9491', 'Student (key-free)'),
            ('0.9272', 'EditGuard (same protocol)'), ('0.947', 'WAM (reported)')]
    for i, (v, n) in enumerate(nums):
        x = 0.42 + i * 3.20
        txt(s, x, by + 0.12, 1.10, 0.26, v, fs=13, bold=True,
            color=(BLUE if i == 1 else INK))
        txt(s, x + 1.12, by + 0.16, 2.00, 0.22, n, fs=FS_BODY)
    prs.save(f'{FIG}/cs_framework.pptx')
    print('-> cs_framework.pptx')


if __name__ == '__main__':
    motivation(); framework()
    for name in ('cs_motivation', 'cs_framework'):
        subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', FIG,
                        f'{FIG}/{name}.pptx'], capture_output=True, timeout=180)
        import pymupdf
        pymupdf.open(f'{FIG}/{name}.pdf')[0].get_pixmap(dpi=220).save(f'{FIG}/{name}.png')
        print('exported', name)
