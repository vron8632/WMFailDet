#!/usr/bin/env python3
"""概念图 v8 — CVPR/ICCV/ICLR 顶会规范.

动机图: 讲"为什么"而非堆数字
  左: 两条现有路线的代价 (主动=要密钥, 被动=要人工标注) -> 都不适配平台审计
  右: 本文关键观察 (真实图片子图: 编辑如何毁掉水印)
  下: 本文路线一句话 + 唯一一组对比数字

框架图: 五阶段 pastel 模块 + 训练/推理分界线 (顶会主流画法)
  每个模块: 色块 + 几何示意 + 真实缩略图
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

INK = RGBColor(0x1A, 0x1D, 0x20)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY_T = RGBColor(0x6E, 0x74, 0x78)
GREY_L = RGBColor(0xC9, 0xCD, 0xD0)
RED = RGBColor(0xA8, 0x30, 0x30)

# pastel 模块族 (顶会主流: 低饱和填充 + 同色系深边 + 深字)
def fam(fill, edge): return (RGBColor(*fill), RGBColor(*edge))
F_BLUE = fam((0xDD, 0xE8, 0xF4), (0x3D, 0x6A, 0x96))
F_ORNG = fam((0xFA, 0xE9, 0xD6), (0xB4, 0x6E, 0x33))
F_GREN = fam((0xDE, 0xEC, 0xE0), (0x4E, 0x7A, 0x5C))
F_VIOL = fam((0xE7, 0xE1, 0xF1), (0x6A, 0x59, 0x9B))
F_GREY = fam((0xEE, 0xEE, 0xEB), (0x82, 0x86, 0x82))
F_ROSE = fam((0xF8, 0xE4, 0xE4), (0x9E, 0x4A, 0x4A))


def slide(prs, w, h):
    prs.slide_width, prs.slide_height = Inches(w), Inches(h)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(w), Inches(h))
    bg.fill.solid(); bg.fill.fore_color.rgb = WHITE; bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def txt(s, x, y, w, h, text, fs=9, bold=False, align=PP_ALIGN.LEFT, color=INK):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.alignment = align
        p.font.size = Pt(fs); p.font.bold = bold; p.font.color.rgb = color
    return tb


def box(s, x, y, w, h, fill, edge, lw=1.0, radius=0.10):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = radius
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = edge; shp.line.width = Pt(lw)
    shp.shadow.inherit = False
    return shp


def block(s, x, y, w, h, title, lines, F, fs_t=10, fs_b=8.2, icon=None, text_w=None):
    """模块块: 色块 + 标题 + 说明 (+ 左上角几何图标)"""
    f, e = F
    box(s, x, y, w, h, f, e, 1.1)
    ml = 0.42 if icon else 0.10
    tw = (text_w - ml - 0.10) if text_w else (w - ml - 0.10)
    tb = s.shapes.add_textbox(Inches(x + ml), Inches(y + 0.05), Inches(tw), Inches(h - 0.10))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = title; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(fs_t); p.font.bold = True; p.font.color.rgb = e
    for ln_ in lines:
        q = tf.add_paragraph(); q.text = ln_; q.alignment = PP_ALIGN.CENTER
        q.font.size = Pt(fs_b); q.font.color.rgb = INK
        q.space_before = Pt(1)
    if icon:
        ic = s.shapes.add_shape(icon, Inches(x + 0.10), Inches(y + h / 2 - 0.13),
                                Inches(0.26), Inches(0.26))
        ic.fill.solid(); ic.fill.fore_color.rgb = e; ic.line.fill.background()
        ic.shadow.inherit = False


def arrow(s, x1, y1, x2, y2, w=1.2, color=GREY_T, dashed=False):
    a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                               Inches(x2), Inches(y2))
    a.line.color.rgb = color; a.line.width = Pt(w)
    ln = a.line._get_or_add_ln()
    if dashed:
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    ln.append(ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'sm', 'len': 'sm'}))


def pic(s, p, x, y, w, h, edge=GREY_L):
    sh = s.shapes.add_picture(p, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.line.color.rgb = edge; sh.line.width = Pt(0.9)
    return sh


def warn(s, x, y, size=0.26, color=RED):
    """红色叉号: 表示"不可用/代价" """
    for (dx1, dy1, dx2, dy2) in ((0, 0, 1, 1), (1, 0, 0, 1)):
        a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + dx1 * size),
                                   Inches(y + dy1 * size), Inches(x + dx2 * size),
                                   Inches(y + dy2 * size))
        a.line.color.rgb = color; a.line.width = Pt(2.0)


# =====================================================================
def motivation():
    """为什么需要 KeyFree-Loc: 两条路线各有代价, 而失效图恰好免费"""
    prs = Presentation(); s = slide(prs, 13.333, 6.6)
    txt(s, 0.40, 0.20, 12.5, 0.34,
        "Platform audit needs tamper localization without the content owner's key",
        fs=15, bold=True)

    # ---------- 左: 两条路线的代价 ----------
    txt(s, 0.40, 0.70, 6.0, 0.24, "Two existing routes, one obstacle each", fs=9.5, bold=True)

    # 路线 A: 主动水印 -> 推理要参考水印/密钥
    block(s, 0.40, 1.02, 5.90, 1.28, "Active watermarking",
          ["pixel-accurate, but the detector needs",
           "the reference watermark or the secret key"],
          F_BLUE, fs_t=10.5, icon=MSO_SHAPE.HEXAGON, text_w=4.10)
    pic(s, f'{SRC}/p_teacher.png', 4.62, 1.18, 0.92, 0.92, F_BLUE[1])
    txt(s, 5.62, 1.30, 0.66, 0.26, "needs", fs=8.5, bold=True, color=RED)
    txt(s, 5.62, 1.54, 0.66, 0.26, "key", fs=8.5, bold=True, color=RED)
    warn(s, 5.72, 1.84, 0.20)

    # 路线 B: 被动取证 -> 训练要人工像素掩膜
    block(s, 0.40, 2.42, 5.90, 1.28, "Passive forensics",
          ["runs on the image alone, but training needs",
           "manual pixel-level masks for every edit"],
          F_ORNG, fs_t=10.5, icon=MSO_SHAPE.DIAMOND, text_w=4.10)
    # 人工标注示意: 叠放的掩膜卡 + 工时
    for i in range(3):
        box(s, 4.42 + i * 0.16, 2.62 + i * 0.11, 0.56, 0.42, WHITE, F_ORNG[1], 0.9, 0.10)
    txt(s, 5.30, 2.92, 0.94, 0.26, "× hours", fs=8.5, bold=True, color=RED)

    # 结论条
    concl = box(s, 0.40, 3.84, 5.90, 0.54, F_ROSE[0], F_ROSE[1], 1.0)
    tf = concl.text_frame; tf.word_wrap = True
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    p.text = "Neither fits: a platform cannot hold every owner's key, and cannot mask every edit"
    p.font.size = Pt(9.2); p.font.bold = True; p.font.color.rgb = F_ROSE[1]

    # ---------- 右: 本文观察 ----------
    txt(s, 6.70, 0.70, 6.2, 0.24, "Our observation: edits destroy the watermark exactly where they act",
        fs=9.5, bold=True)
    seq = [(f'{SRC}/p_host.png', 'watermarked host'),
           (f'{SRC}/p_edited.png', 'after a generative edit'),
           (f'{SRC}/p_failmap.png', 'per-block check fails')]
    x0, y0, sz = 6.70, 1.02, 1.86
    for i, (p_, cap) in enumerate(seq):
        x = x0 + i * (sz + 0.34)
        pic(s, p_, x, y0, sz, sz)
        txt(s, x - 0.06, y0 + sz + 0.04, sz + 0.12, 0.34, cap, fs=8, align=PP_ALIGN.CENTER)
        if i < 2:
            arrow(s, x + sz + 0.04, y0 + sz / 2, x + sz + 0.30, y0 + sz / 2)

    obs = box(s, 6.86, 3.84, 5.92, 0.54, F_GREN[0], F_GREN[1], 1.0)
    tf = obs.text_frame; tf.word_wrap = True
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    p.text = "The failure map is spatially exact and free — it can supervise a detector that never sees a watermark"
    p.font.size = Pt(9.2); p.font.bold = True; p.font.color.rgb = F_GREN[1]

    # ---------- 底部: 本文路线 ----------
    txt(s, 0.40, 4.58, 5.0, 0.24, "KeyFree-Loc pipeline (this work)", fs=9.5, bold=True)
    y = 4.92
    block(s, 0.40, y, 3.10, 0.80, "Teacher", ["keyed, zero training"], F_BLUE, fs_t=10, fs_b=8)
    block(s, 3.90, y, 3.10, 0.80, "Free pseudo-labels", ["no annotation"], F_GREN, fs_t=10, fs_b=8)
    block(s, 7.40, y, 3.10, 0.80, "Student", ["image only, 1.93 M"], F_VIOL, fs_t=10, fs_b=8)
    for xa, xb in ((3.50, 3.88), (7.00, 7.38)):
        arrow(s, xa, y + 0.40, xb, y + 0.40, w=1.5)
    # 唯一一组对比数字
    txt(s, 10.72, y + 0.04, 2.4, 0.34, "0.9491", fs=15, bold=True, color=F_VIOL[1])
    txt(s, 10.72, y + 0.42, 2.4, 0.22, "vs 0.9272 (needs a watermark)", fs=7.8)
    prs.save(f'{FIG}/cs_motivation.pptx')
    print('-> cs_motivation.pptx')


# =====================================================================
def framework():
    """上下两排: 上=训练(可持密钥), 下=推理(无密钥); 分界线即任务本身"""
    prs = Presentation(); s = slide(prs, 13.333, 5.46)
    txt(s, 0.40, 0.16, 12.4, 0.30,
        "Watermark failure as free supervision: training keeps the key, deployment does not",
        fs=14, bold=True)

    # ---------- 上排: 训练 (key available) ----------
    txt(s, 0.40, 0.58, 6.0, 0.22, "Training  —  the owner holds the key", fs=9, bold=True)
    y1, h1 = 0.86, 1.06
    block(s, 0.40, y1, 2.30, h1, "Embed", ["keyed QIM, layers 8/4/2", "PSNR 38.2 dB"],
          F_BLUE, icon=MSO_SHAPE.HEXAGON)
    block(s, 3.02, y1, 2.30, h1, "Edit", ["SD inpainting", "LaMa / IP2P"],
          F_ORNG, icon=MSO_SHAPE.LIGHTNING_BOLT)
    block(s, 5.64, y1, 2.60, h1, "Teacher", ["per-block check", "zero training"],
          F_GREN, icon=MSO_SHAPE.SUN)
    for xa, xb in ((2.72, 3.00), (5.34, 5.62)):
        arrow(s, xa, y1 + h1 / 2, xb, y1 + h1 / 2, w=1.5)
    # 训练侧缩略
    pic(s, f'{SRC}/p_host.png', 0.40, y1 + h1 + 0.08, 0.56, 0.56, F_BLUE[1])
    pic(s, f'{SRC}/p_failmap.png', 5.64, y1 + h1 + 0.08, 0.80, 0.56, F_GREN[1])
    txt(s, 0.40, y1 + h1 + 0.66, 1.20, 0.16, "host", fs=6.4)
    txt(s, 5.64, y1 + h1 + 0.66, 2.60, 0.16, "failure map = free labels", fs=6.6)

    # ---------- 分界线 ----------
    by = 3.06
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.40), Inches(by),
                                Inches(12.93), Inches(by))
    ln.line.color.rgb = RED; ln.line.width = Pt(1.1)
    ln.line._get_or_add_ln().append(ln.line._get_or_add_ln().makeelement(qn('a:prstDash'), {'val': 'dash'}))
    txt(s, 0.40, by - 0.24, 4.2, 0.20, "▲ key available during training", fs=7.6, color=RED)
    txt(s, 9.60, by - 0.24, 3.3, 0.20, "▼ no key at inference", fs=7.6, color=RED, align=PP_ALIGN.RIGHT)

    # ---------- 下排: 推理 (no key) ----------
    y2, h2 = 3.26, 1.00
    block(s, 0.40, y2, 2.90, h2, "Distillation", ["teacher labels → 1.93 M student"],
          F_VIOL, icon=MSO_SHAPE.PENTAGON, fs_t=9.5, fs_b=7.8)
    block(s, 4.10, y2, 3.10, h2, "Student (ours)", ["image only · 1.1 ms / image", "no watermark, no key"],
          F_VIOL, icon=MSO_SHAPE.PENTAGON, fs_t=10, fs_b=7.8)
    block(s, 7.90, y2, 2.40, h2, "Verdict", ["authentic / tampered", "abstain + region"],
          F_GREY, icon=MSO_SHAPE.STAR_5_POINT, fs_t=10, fs_b=7.8)
    for xa, xb in ((3.34, 4.08), (7.24, 7.88)):
        arrow(s, xa, y2 + h2 / 2, xb, y2 + h2 / 2, w=1.5)
    pic(s, f'{SRC}/p_edited.png', 4.10, y2 + h2 + 0.10, 0.66, 0.66, F_VIOL[1])
    pic(s, f'{SRC}/p_student.png', 4.82, y2 + h2 + 0.10, 0.66, 0.66, F_VIOL[1])
    pic(s, f'{SRC}/p_fusion.png', 5.54, y2 + h2 + 0.10, 0.66, 0.66, F_VIOL[1])
    txt(s, 4.10, y2 + h2 + 0.80, 2.10, 0.18, "edited query → prediction → verdict", fs=6.6)

    # ---------- 跨分界线的唯一产物 ----------
    arrow(s, 6.90, y1 + h1 + 0.02, 6.90, y2 - 0.02, w=1.6, color=F_GREN[1])
    txt(s, 6.98, 2.72, 2.10, 0.20, "free pseudo-labels", fs=7.6, bold=True, color=F_GREN[1])

    # ---------- 结果 ----------
    txt(s, 10.50, y2 + 0.20, 2.45, 0.30, "0.9491", fs=15, bold=True, color=F_VIOL[1])
    txt(s, 10.50, y2 + 0.58, 2.45, 0.22, "±0.0012, image only", fs=7.6)
    txt(s, 10.50, y2 + 0.80, 2.45, 0.22, "EditGuard 0.9272 (needs watermark)", fs=7.2)
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
