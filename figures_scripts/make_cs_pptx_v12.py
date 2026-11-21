#!/usr/bin/env python3
"""概念图 v12 — 按参考风格重画.

动机图: 仿 teaser.pdf 风格
  (A)(B) 两幅大图并排, 图上直接贴标签, 红虚线框, 底部一句总结
  论证: 编辑在视觉上不可见 (A) 但在校验上确定失效 (B) -> 免费标签
框架图: 仿 fig_framework.pptx 风格
  分区容器 (Training / Deployment) + 紧凑子框 + 大字号 + 实心红强调 + 右侧结果条
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

NAVY = RGBColor(0x1F, 0x4E, 0x79)      # 标题/主文字（参考图的深蓝）
INK = RGBColor(0x22, 0x26, 0x2A)
GREY = RGBColor(0x5C, 0x63, 0x68)
RED = RGBColor(0xC0, 0x39, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_TRAIN = RGBColor(0xEE, 0xF2, 0xF7)   # 训练容器底
C_DEPLOY = RGBColor(0xED, 0xF3, 0xEF)  # 部署容器底
E_BLUE = RGBColor(0x4A, 0x7B, 0xA7)
E_GREEN = RGBColor(0x5A, 0x8A, 0x6A)
E_VIOL = RGBColor(0x6A, 0x59, 0x9B)
E_GREY = RGBColor(0x86, 0x8B, 0x88)


def slide(prs, w, h):
    prs.slide_width, prs.slide_height = Inches(w), Inches(h)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(w), Inches(h))
    bg.fill.solid(); bg.fill.fore_color.rgb = WHITE; bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def txt(s, x, y, w, h, text, fs=10, bold=False, color=INK, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP, wrap=True):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.alignment = align
        p.font.size = Pt(fs); p.font.bold = bold; p.font.color.rgb = color
    return tb


def container(s, x, y, w, h, title, fill, edge):
    """分区容器: 浅底 + 细边 + 左上标题（参考图 Diagnosis/Remedy）"""
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.035
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = edge; shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    txt(s, x + 0.16, y + 0.11, w - 0.3, 0.28, title, fs=13.5, bold=True, color=NAVY)
    return shp


def node(s, x, y, w, h, title, sub, edge, solid=None, fs_t=11.5, fs_s=9.2):
    """紧凑子框: 文字撑满, 字号大（参考图风格）"""
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.10
    shp.fill.solid(); shp.fill.fore_color.rgb = solid if solid else WHITE
    shp.line.color.rgb = edge; shp.line.width = Pt(1.1)
    shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = title; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(fs_t); p.font.bold = True
    p.font.color.rgb = WHITE if solid else NAVY
    if sub:
        q = tf.add_paragraph(); q.text = sub; q.alignment = PP_ALIGN.CENTER
        q.font.size = Pt(fs_s); q.font.color.rgb = WHITE if solid else INK
        q.space_before = Pt(1)
    return shp


def arrow(s, x1, y1, x2, y2, w=1.4, color=RGBColor(0x44, 0x4A, 0x4E), dashed=False):
    a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                               Inches(x2), Inches(y2))
    a.line.color.rgb = color; a.line.width = Pt(w)
    ln = a.line._get_or_add_ln()
    if dashed:
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    ln.append(ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}))


def elbow(s, pts, color=RGBColor(0x44, 0x4A, 0x4E), w=1.4):
    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        last = (i == len(pts) - 2)
        if last:
            arrow(s, x1, y1, x2, y2, w=w, color=color)
        else:
            a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                       Inches(x2), Inches(y2))
            a.line.color.rgb = color; a.line.width = Pt(w)


def tag(s, x, y, text, fs=9.5, edge=None, bold=False):
    """图上贴的白底标签（teaser 风格）"""
    w = 0.112 * len(text) + 0.18
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.28))
    shp.adjustments[0] = 0.28
    shp.fill.solid(); shp.fill.fore_color.rgb = WHITE
    shp.line.color.rgb = edge if edge else RGBColor(0x9A, 0xA0, 0xA4)
    shp.line.width = Pt(0.9)
    shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = text; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(fs); p.font.bold = bold; p.font.color.rgb = NAVY
    return shp, w


def redbox(s, x, y, w, h):
    shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.background()
    shp.line.color.rgb = RED; shp.line.width = Pt(2.2)
    shp.line._get_or_add_ln().append(
        shp.line._get_or_add_ln().makeelement(qn('a:prstDash'), {'val': 'dash'}))
    shp.shadow.inherit = False
    return shp


def pic(s, p, x, y, w, h, edge=None):
    sh = s.shapes.add_picture(p, Inches(x), Inches(y), Inches(w), Inches(h))
    if edge:
        sh.line.color.rgb = edge; sh.line.width = Pt(1.0)
    return sh


# =====================================================================
def motivation():
    """teaser 风格: (A) 视觉不可见  (B) 校验确定失效 -> 免费标签"""
    prs = Presentation(); s = slide(prs, 13.333, 6.05)
    txt(s, 0.42, 0.16, 12.5, 0.34,
        "An invisible edit leaves a free, pixel-exact forensic trace", fs=16, bold=True,
        color=RGBColor(0x1A, 0x1A, 0x1A))

    # ---- (A) 视觉: 看不出被改过
    txt(s, 0.42, 0.62, 6.0, 0.28, "(A)  Visually: the edit is imperceptible", fs=13,
        bold=True, color=NAVY)
    pic(s, f'{SRC}/p_edited.png', 0.42, 0.96, 5.95, 3.30)
    tag(s, 3.44, 4.12, "MLLM: replace the object", edge=E_BLUE)
    tag(s, 0.62, 1.10, "outside unchanged", edge=E_GREEN, bold=True)

    # ---- (B) 校验: 块级检查失效
    txt(s, 6.95, 0.62, 6.0, 0.28, "(B)  Physically: the check bit collapses", fs=13,
        bold=True, color=NAVY)
    pic(s, f'{SRC}/p_teacher.png', 6.95, 0.96, 5.95, 3.30)
    tag(s, 7.10, 1.10, "check failed: inside box", edge=RED, bold=True)
    tag(s, 8.55, 3.62, "0.501 inside / 0.259 outside", edge=E_GREY)

    # ---- 底部总结
    br = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.42), Inches(4.52),
                            Inches(12.48), Inches(0.62))
    br.adjustments[0] = 0.16
    br.fill.solid(); br.fill.fore_color.rgb = RGBColor(0xF4, 0xF6, 0xF8)
    br.line.color.rgb = RGBColor(0xD4, 0xD9, 0xDD); br.line.width = Pt(0.75)
    br.shadow.inherit = False
    tf = br.text_frame; tf.word_wrap = True
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    p.text = ("The eye cannot see the edit, but the block check fails exactly inside it — "
              "a label that costs nothing to collect")
    p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = NAVY

    # ---- 本文一句
    txt(s, 0.42, 5.34, 12.48, 0.50,
        "KeyFree-Loc:  the owner keeps the key at training time;  the deployed student sees only the image",
        fs=11, color=INK, align=PP_ALIGN.CENTER)
    prs.save(f'{FIG}/cs_motivation.pptx')
    print('-> cs_motivation.pptx')


# =====================================================================
def framework():
    """fig_framework 风格: 两个分区容器 + 紧凑大字号子框 + 右侧结果条"""
    prs = Presentation(); s = slide(prs, 13.333, 6.55)
    txt(s, 0.34, 0.14, 12.6, 0.30,
        "Watermark failure as free supervision", fs=16, bold=True, color=RGBColor(0x1A,0x1A,0x1A))

    # ================= 上容器: Training =================
    container(s, 0.34, 0.54, 9.05, 2.52, "Training  —  the owner holds the key", C_TRAIN,
              RGBColor(0xC6, 0xD4, 0xE2))
    y1, h1 = 0.98, 0.80
    node(s, 0.52, y1, 1.95, h1, "Embed", "keyed QIM 8/4/2", E_BLUE, fs_t=13, fs_s=9.8)
    node(s, 2.70, y1, 1.75, h1, "Edit", "SD / LaMa / IP2P", E_GREEN, fs_t=13, fs_s=9.8)
    node(s, 4.68, y1, 1.95, h1, "Teacher", "per-block check", E_VIOL, fs_t=13, fs_s=9.8)
    arrow(s, 2.49, y1 + h1/2, 2.68, y1 + h1/2)
    arrow(s, 4.47, y1 + h1/2, 4.66, y1 + h1/2)
    # 真实图: embed 下 host, teacher 下失效图
    pic(s, f'{SRC}/p_host.png', 0.52, y1 + h1 + 0.08, 1.10, 1.10, E_BLUE)
    txt(s, 1.70, y1 + h1 + 0.24, 1.25, 0.40, "watermarked\nhost", fs=9, color=GREY)
    pic(s, f'{SRC}/p_failmap.png', 4.68, y1 + h1 + 0.08, 1.55, 1.10, E_VIOL)


    # ================= 下容器: Deployment =================
    container(s, 0.34, 3.26, 9.05, 2.62, "Deployment  —  the image alone", C_DEPLOY,
              RGBColor(0xC9, 0xDC, 0xCE))
    y2, h2 = 3.70, 0.80
    node(s, 0.52, y2, 1.95, h2, "Distillation", "1.93 M student", E_VIOL, fs_t=13, fs_s=9.8)
    node(s, 2.70, y2, 2.10, h2, "Student (ours)", "image only · 1.1 ms", E_VIOL, solid=None, fs_t=13, fs_s=9.8)
    node(s, 5.03, y2, 1.95, h2, "Verdict", "3-way + region", E_GREY, fs_t=13, fs_s=9.8)
    arrow(s, 2.49, y2 + h2/2, 2.68, y2 + h2/2)
    arrow(s, 4.82, y2 + h2/2, 5.01, y2 + h2/2)
    pic(s, f'{SRC}/p_edited.png', 0.52, y2 + h2 + 0.08, 1.10, 1.10, E_BLUE)
    txt(s, 1.70, y2 + h2 + 0.24, 1.20, 0.40, "query\n(any copy)", fs=9, color=GREY)
    pic(s, f'{SRC}/p_student.png', 2.70, y2 + h2 + 0.08, 1.10, 1.10, E_VIOL)
    pic(s, f'{SRC}/p_fusion.png', 3.86, y2 + h2 + 0.08, 1.10, 1.10, E_GREY)
    txt(s, 5.04, y2 + h2 + 0.24, 1.60, 0.40, "prediction →\nverdict", fs=9, color=GREY)

    # ================= 唯一的跨区箭头: free labels =================
    elbow(s, [(5.45, 2.98), (5.45, 3.12), (3.75, 3.12), (3.75, 3.66)], color=RED, w=2.2)
    txt(s, 6.45, 1.98, 2.70, 0.30, "free pseudo-labels", fs=13, bold=True, color=RED)
    txt(s, 6.45, 2.34, 2.70, 0.44, "zero human annotation\npixel-accurate 2×2 layer", fs=9, color=GREY)

    # ================= 右侧结果条 =================
    container(s, 9.62, 0.54, 3.30, 5.34, "Result", C_TRAIN, RGBColor(0xC6, 0xD4, 0xE2))
    rows = [("0.9491", "Student (ours)\nimage only · key-free", E_VIOL, True),
            ("0.9272", "EditGuard\nneeds reference watermark", E_GREY, False),
            ("0.726", "Teacher 4×4\nneeds key", E_GREY, False)]
    yy = 1.06
    for i, (v, lab, col, hl) in enumerate(rows):
        txt(s, 9.80, yy, 2.95, 0.38, v, fs=20 if hl else 16, bold=True,
            color=(E_VIOL if hl else INK))
        txt(s, 9.80, yy + (0.40 if hl else 0.34), 2.95, 0.44, lab, fs=9.4, color=GREY)
        yy += 1.62 if hl else 1.46
        if i < 2:
            ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(9.80), Inches(yy - 0.62),
                                        Inches(12.74), Inches(yy - 0.62))
            ln.line.color.rgb = RGBColor(0xD4, 0xD9, 0xDD); ln.line.width = Pt(0.75)

    # ================= 底部控制句 =================
    txt(s, 0.34, 5.94, 12.6, 0.34,
        "Falsification controls:  δ search (3 configs)  ·  label resolution (8/4/2)  ·  capacity  ·  "
        "resampling  ·  resolution  ·  data scale   —   four negative results",
        fs=9.4, color=GREY, align=PP_ALIGN.CENTER)
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
