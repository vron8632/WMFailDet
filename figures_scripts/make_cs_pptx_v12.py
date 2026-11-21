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
    prs = Presentation(); s = slide(prs, 13.333, 6.62)
    txt(s, 0.42, 0.16, 12.5, 0.34,
        "An invisible edit leaves a free, pixel-exact forensic trace", fs=16, bold=True,
        color=RGBColor(0x1A, 0x1A, 0x1A))

    # ---- (A) 视觉: 看不出被改过
    txt(s, 0.75, 0.60, 4.30, 0.28, "(A)  Visually: the edit blends in", fs=12.5,
        bold=True, color=NAVY)
    pic(s, f'{SRC}/p_edited.png', 0.75, 0.94, 4.30, 4.30)
    tag(s, 2.42, 5.30, "red box = edit region", edge=E_BLUE)
    tag(s, 0.90, 1.08, "outside unchanged", edge=E_GREEN, bold=True)

    # ---- (B) 校验: 块级检查失效
    txt(s, 7.95, 0.60, 4.30, 0.28, "(B)  Physically: the block check fails", fs=12.5,
        bold=True, color=NAVY)
    pic(s, f'{SRC}/p_teacher.png', 7.95, 0.94, 4.30, 4.30)
    tag(s, 8.10, 1.08, "check failed: inside box", edge=RED, bold=True)
    tag(s, 9.55, 4.98, "0.501 inside / 0.259 outside", edge=E_GREY)

    # ---- 底部总结
    br = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.42), Inches(5.62),
                            Inches(12.48), Inches(0.58))
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
    txt(s, 0.42, 6.28, 12.48, 0.28,
        "KeyFree-Loc:  the owner keeps the key at training time;  the deployed student sees only the image",
        fs=11, color=INK, align=PP_ALIGN.CENTER)
    prs.save(f'{FIG}/cs_motivation.pptx')
    print('-> cs_motivation.pptx')


# =====================================================================
def framework():
    """只画方法: 两容器 + 七色模块 + 真实缩略 + 跨线的伪标签箭头 (无结果数字)"""
    prs = Presentation(); s = slide(prs, 13.333, 5.80)
    txt(s, 0.34, 0.12, 12.6, 0.30,
        "Watermark failure as free supervision", fs=16, bold=True, color=RGBColor(0x1A,0x1A,0x1A))
    txt(s, 0.34, 0.44, 12.6, 0.20,
        "Training keeps the key; the deployed student never sees a watermark",
        fs=10, color=GREY)

    FAM = {
        'blue':   (RGBColor(0xDC, 0xE9, 0xF5), RGBColor(0x3E, 0x6D, 0x9B)),
        'orange': (RGBColor(0xFA, 0xE7, 0xD3), RGBColor(0xBE, 0x77, 0x38)),
        'green':  (RGBColor(0xDD, 0xED, 0xE1), RGBColor(0x4B, 0x7A, 0x59)),
        'teal':   (RGBColor(0xD7, 0xEC, 0xEF), RGBColor(0x35, 0x77, 0x82)),
        'violet': (RGBColor(0xE7, 0xE1, 0xF2), RGBColor(0x67, 0x55, 0x99)),
        'rose':   (RGBColor(0xF8, 0xE2, 0xE5), RGBColor(0x9B, 0x51, 0x5C)),
        'gold':   (RGBColor(0xF8, 0xF0, 0xD7), RGBColor(0x8A, 0x71, 0x2F)),
    }

    def net_bars(x, y, w, h, color, hs=(0.20, 0.32, 0.20)):
        """U-Net 示意 (小型): 编码-瓶颈-解码 三竖条"""
        bw, gap = 0.070, 0.040
        total = len(hs) * bw + (len(hs) - 1) * gap
        x0 = x + (w - total) / 2
        for i, hh in enumerate(hs):
            bar = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                     Inches(x0 + i * (bw + gap)),
                                     Inches(y + (h - hh) / 2), Inches(bw), Inches(hh))
            bar.fill.solid(); bar.fill.fore_color.rgb = color
            bar.line.fill.background(); bar.shadow.inherit = False

    def grid_icon(x, y, size, color, n=3, cross=False):
        """块网格示意 (可带叉号表示校验失败)"""
        cell = (size - (n - 1) * 0.02) / n
        for i in range(n):
            for j in range(n):
                c = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                       Inches(x + j * (cell + 0.02)),
                                       Inches(y + i * (cell + 0.02)),
                                       Inches(cell), Inches(cell))
                c.fill.solid()
                c.fill.fore_color.rgb = color if (cross and (i + j) % 2 == 0) else WHITE
                c.line.color.rgb = color; c.line.width = Pt(0.5)
                c.shadow.inherit = False

    def mod(x, y, w, h, title, sub, key, fs_t=12.5, fs_s=9.2, icon=None):
        f, e = FAM[key]
        shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
        shp.adjustments[0] = 0.13
        shp.fill.solid(); shp.fill.fore_color.rgb = f
        shp.line.color.rgb = e; shp.line.width = Pt(1.0)
        shp.shadow.inherit = False
        tf = shp.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.04)
        tf.margin_top = tf.margin_bottom = Inches(0.02)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.text = title; p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(fs_t); p.font.bold = True; p.font.color.rgb = e
        if sub:
            q = tf.add_paragraph(); q.text = sub; q.alignment = PP_ALIGN.CENTER
            q.font.size = Pt(fs_s); q.font.color.rgb = INK; q.space_before = Pt(1)
        # 图标: 放在框内左下或右下
        if icon == 'grid':
            grid_icon(x + w - 0.42, y + h/2 - 0.17, 0.34, e)
        elif icon == 'gridx':
            grid_icon(x + w - 0.42, y + h/2 - 0.17, 0.34, e, cross=True)
        elif icon == 'net':
            net_bars(x + w - 0.46, y + h/2 - 0.16, 0.36, 0.32, e)

    # ============ Training 容器 (底 2.98) ============
    container(s, 0.34, 0.74, 12.66, 2.20, "Training  —  the owner holds the key",
              C_TRAIN, RGBColor(0xC6, 0xD4, 0xE2))
    y1, h1 = 1.14, 0.76
    xs = [0.56, 3.71, 6.86, 10.01]
    specs = [("Embed", "keyed QIM, layers 8/4/2", 'blue'),
             ("Edit", "SD / LaMa / IP2P", 'orange'),
             ("Teacher", "per-block check bit", 'green'),
             ("Failure map", "where the watermark died", 'teal')]
    for x, (t_, sub, key) in zip(xs, specs):
        mod(x, y1, 2.90, h1, t_, sub, key, icon={'Embed': 'grid', 'Edit': None,
                                                'Teacher': 'gridx', 'Failure map': None}[t_])
    for i in range(3):
        arrow(s, xs[i] + 2.92, y1 + h1/2, xs[i+1] - 0.02, y1 + h1/2)
    # 真实缩略 (在容器内, 底 2.82 < 2.94)
    thumbs1 = [(f'{SRC}/p_host.png', 0.56, "host", 'blue'),
               (f'{SRC}/p_edited.png', 3.71, "edited", 'orange'),
               (f'{SRC}/p_failmap.png', 6.86, "failure map", 'teal')]
    for path, x, cap, key in thumbs1:
        pic(s, path, x, y1 + h1 + 0.06, 0.84, 0.84, FAM[key][1])
        txt(s, x + 0.92, y1 + h1 + 0.30, 1.80, 0.24, cap, fs=8.6, color=GREY)

    # ============ 跨容器: 唯一产物 ============
    arrow(s, 6.10, 2.96, 6.10, 3.40, w=2.4, color=RED)
    txt(s, 6.32, 2.98, 3.40, 0.24, "free pseudo-labels", fs=12, bold=True, color=RED)
    txt(s, 6.32, 3.22, 3.40, 0.20, "zero annotation · pixel-accurate", fs=8.8, color=GREY)

    # ============ Deployment 容器 (底 5.66) ============
    container(s, 0.34, 3.42, 12.66, 2.20, "Deployment  —  the image alone",
              C_DEPLOY, RGBColor(0xC9, 0xDC, 0xCE))
    y2, h2 = 3.80, 0.76
    specs2 = [("Distillation", "free labels only", 'violet'),
              ("Student", "image only · 1.1 ms", 'violet'),
              ("Evidence fusion", "visual + check evidence", 'rose'),
              ("Verdict", "authentic / tampered / abstain", 'gold')]
    for x, (t_, sub, key) in zip(xs, specs2):
        mod(x, y2, 2.90, h2, t_, sub, key,
            fs_t=(12.5 if key != 'gold' else 12.0),
            icon={'Distillation': 'net', 'Student': 'net'}.get(t_))
    for i in range(3):
        arrow(s, xs[i] + 2.92, y2 + h2/2, xs[i+1] - 0.02, y2 + h2/2)
    thumbs2 = [(f'{SRC}/p_edited.png', 0.56, "query (any copy)", 'blue'),
               (f'{SRC}/p_student.png', 3.71, "prediction", 'violet'),
               (f'{SRC}/p_fusion.png', 6.86, "fused output", 'rose')]
    for path, x, cap, key in thumbs2:
        pic(s, path, x, y2 + h2 + 0.06, 0.84, 0.84, FAM[key][1])
        txt(s, x + 0.92, y2 + h2 + 0.30, 1.80, 0.24, cap, fs=8.6, color=GREY)

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
