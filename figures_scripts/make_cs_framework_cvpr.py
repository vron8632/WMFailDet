#!/usr/bin/env python3
"""框架图 — 完全按参考布局 (CVPR方法框架图) 重画, 仅标题自拟.

布局: 上=Training 容器(蛇形: 图->框->图->框->图, 下折到 Teacher -> Failure Map -> Distillation -> Student)
      虚线分界
      下=Deployment 容器(Query -> Student -> Fusion -> Verdict)
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

INK = RGBColor(0x11, 0x11, 0x11)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLUE = RGBColor(0x1F, 0x5F, 0x9E)          # 箭头/容器标签
C_TRAIN = RGBColor(0xE8, 0xF0, 0xF8)       # 训练容器浅蓝
E_TRAIN = RGBColor(0xB4, 0xC8, 0xDD)
TRAIN_TAG = RGBColor(0xB8, 0xCC, 0xE0)
C_DEPLOY = RGBColor(0xF2, 0xF2, 0xF2)      # 部署容器浅灰
E_DEPLOY = RGBColor(0xCF, 0xCF, 0xCF)
DEPLOY_TAG = RGBColor(0xC9, 0xC9, 0xC9)
GREEN = RGBColor(0x6A, 0xA8, 0x4F)
RED = RGBColor(0xC0, 0x39, 0x2B)
ORNG = RGBColor(0xE8, 0x8E, 0x2A)


def slide(prs, w, h):
    prs.slide_width, prs.slide_height = Inches(w), Inches(h)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(w), Inches(h))
    bg.fill.solid(); bg.fill.fore_color.rgb = WHITE; bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def txt(s, x, y, w, h, text, fs=11, bold=False, color=INK, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.alignment = align
        p.font.size = Pt(fs); p.font.bold = bold; p.font.color.rgb = color
    return tb


def cbox(s, x, y, w, h, fill, edge, lw=1.0, radius=0.04):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = radius
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = edge; shp.line.width = Pt(lw)
    shp.shadow.inherit = False
    return shp


def pill(s, x, y, w, h, text, fill, fs=12):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.5
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.fill.background(); shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = Inches(0.10)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = text; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(fs); p.font.bold = True; p.font.color.rgb = INK
    return shp


def bord(s, x, y, w, h, title, sub=None, fs_t=12.5, fs_s=9.8):
    """白底黑边圆角框: 标题(可多行) + 副标题, 动态垂直居中不重叠"""
    cbox(s, x, y, w, h, WHITE, INK, 1.3, 0.10)
    nl = title.count('\n') + 1
    line_h = fs_t / 72.0 * 1.32
    t_h = line_h * nl
    s_h = (fs_s / 72.0 * 1.30) if sub else 0.0
    pad = 0.03 if sub else 0.0
    total = t_h + s_h + pad
    top = y + max(0.04, (h - total) / 2)
    txt(s, x + 0.05, top, w - 0.10, t_h, title, fs=fs_t, bold=True,
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP)
    if sub:
        txt(s, x + 0.05, top + t_h + pad, w - 0.10, s_h, sub, fs=fs_s,
            color=RGBColor(0x33, 0x33, 0x33), align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP)


def arr(s, x1, y1, x2, y2, w=2.2, color=BLUE, dashed=False):
    a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                               Inches(x2), Inches(y2))
    a.line.color.rgb = color; a.line.width = Pt(w)
    ln = a.line._get_or_add_ln()
    if dashed:
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    ln.append(ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}))


def path(s, pts, w=2.2, color=BLUE):
    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        if i == len(pts) - 2:
            arr(s, x1, y1, x2, y2, w=w, color=color)
        else:
            a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                       Inches(x2), Inches(y2))
            a.line.color.rgb = color; a.line.width = Pt(w)


def photo(s, path_, x, y, size, label):
    sh = s.shapes.add_picture(path_, Inches(x), Inches(y), Inches(size), Inches(size))
    sh.line.color.rgb = INK; sh.line.width = Pt(0.75)
    txt(s, x - 0.10, y + size + 0.04, size + 0.20, 0.24, label, fs=10.5, bold=True,
        align=PP_ALIGN.CENTER)


def unet(s, x, y, w, h, color=RGBColor(0x9C, 0xC0, 0xDE), edge=BLUE):
    """U-Net 示意: 编码(下采样) -> 瓶颈 -> 解码(上采样) + skip 虚线"""
    # 每层: (相对高度, 相对宽度) —— 编码递减, 解码递增
    layers = [(0.30, 0.16), (0.22, 0.16), (0.14, 0.16), (0.08, 0.16),
              (0.14, 0.16), (0.22, 0.16), (0.30, 0.16)]
    gap = (w - len(layers) * 0.16 * w / 1.0) if False else None
    bw = 0.115
    n = len(layers)
    total = n * bw + (n - 1) * 0.055
    x0 = x + (w - total) / 2
    top = y + 0.02
    for i, (hh, _) in enumerate(layers):
        bh = hh * (h - 0.04) / 0.30
        bar = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 Inches(x0 + i * (bw + 0.055)),
                                 Inches(top + (h - 0.04 - bh) / 2),
                                 Inches(bw), Inches(bh))
        bar.adjustments[0] = 0.18
        bar.fill.solid(); bar.fill.fore_color.rgb = color
        bar.line.color.rgb = edge; bar.line.width = Pt(0.6)
        bar.shadow.inherit = False
    # skip connections (虚线): 对称层相连
    for a, b in ((0, 6), (1, 5), (2, 4)):
        xa = x0 + a * (bw + 0.055) + bw
        xb = x0 + b * (bw + 0.055)
        ya = top + 0.012
        con = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(xa + 0.006), Inches(ya),
                                      Inches(xb - 0.006), Inches(ya))
        con.line.color.rgb = edge; con.line.width = Pt(0.7)
        con.line._get_or_add_ln().append(
            con.line._get_or_add_ln().makeelement(qn('a:prstDash'), {'val': 'sysDot'}))


def grid_icon(s, x, y, size, color, n=3, filled_alt=False):
    """块网格图标: filled_alt 时交错填充 (示意校验失败)"""
    cell = (size - (n - 1) * 0.016) / n
    for i in range(n):
        for j in range(n):
            c = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                   Inches(x + j * (cell + 0.016)),
                                   Inches(y + i * (cell + 0.016)),
                                   Inches(cell), Inches(cell))
            c.fill.solid()
            c.fill.fore_color.rgb = color if (filled_alt and (i + j) % 2 == 0) else WHITE
            c.line.color.rgb = color; c.line.width = Pt(0.45)
            c.shadow.inherit = False


def bolt_icon(s, x, y, size, color):
    b = s.shapes.add_shape(MSO_SHAPE.LIGHTNING_BOLT, Inches(x), Inches(y),
                           Inches(size * 0.72), Inches(size))
    b.fill.solid(); b.fill.fore_color.rgb = color
    b.line.color.rgb = color; b.line.width = Pt(0.4)
    b.shadow.inherit = False


def main():
    prs = Presentation(); s = slide(prs, 16.0, 9.05)

    txt(s, 0.0, 0.18, 16.0, 0.38,
        "Watermark Failure Distillation for Key-Free Manipulation Localization",
        fs=19, bold=True, align=PP_ALIGN.CENTER)

    # ================= TRAINING =================
    cbox(s, 0.30, 0.68, 15.40, 5.46, C_TRAIN, E_TRAIN, 1.0, 0.02)
    pill(s, 0.46, 0.80, 4.20, 0.40, "TRAINING — THE OWNER HOLDS THE KEY", TRAIN_TAG, fs=12.5)

    y0, isz = 1.44, 1.45
    photo(s, f'{SRC}/p_host.png', 0.62, y0, isz, "Host Image")
    bord(s, 2.44, y0 + 0.16, 2.40, 1.12, "Keyed QIM Embedding", "3 layers: 8×8 / 4×4 / 2×2")
    grid_icon(s, 4.40, y0 + 0.24, 0.30, BLUE)
    photo(s, f'{SRC}/p_host.png', 5.16, y0, isz, "Watermarked Image")
    bord(s, 7.02, y0 + 0.22, 2.00, 1.00, "Generative Edit", "SD / LaMa / IP2P")
    bolt_icon(s, 8.60, y0 + 0.28, 0.32, ORNG)
    photo(s, f'{SRC}/p_edited.png', 9.36, y0, isz, "Edited Image")
    arr(s, 2.09, y0 + isz/2, 2.41, y0 + isz/2)
    arr(s, 4.86, y0 + isz/2, 5.13, y0 + isz/2)
    arr(s, 6.63, y0 + isz/2, 6.99, y0 + isz/2)
    arr(s, 9.04, y0 + isz/2, 9.33, y0 + isz/2)

    # Teacher -> Failure Map
    arr(s, 10.08, y0 + isz + 0.06, 10.08, 3.14)
    bord(s, 9.10, 3.18, 2.10, 0.94, "Teacher: Per-block\nCheck Verification")
    grid_icon(s, 10.82, 3.26, 0.28, RED, filled_alt=True)
    arr(s, 11.30, 3.65, 11.90, 3.65)
    fsz = 1.50
    sh = s.shapes.add_picture(f'{SRC}/p_failmap_orange.png', Inches(11.94), Inches(2.96),
                              Inches(fsz), Inches(fsz))
    sh.line.color.rgb = INK; sh.line.width = Pt(1.0)
    txt(s, 11.74, 4.50, fsz + 0.40, 0.22, "Failure Map", fs=10.5, bold=True, align=PP_ALIGN.CENTER)
    txt(s, 11.54, 4.74, fsz + 0.80, 0.20, "Free pseudo-labels · Zero annotation · Pixel-accurate",
        fs=8.0, color=RGBColor(0x33, 0x33, 0x33), align=PP_ALIGN.CENTER)

    # Teacher -> Distillation
    bord(s, 9.10, 4.38, 2.10, 0.62, "Distillation")
    arr(s, 10.10, 5.02, 10.10, 5.14)
    # Failure Map -> Distillation (右侧绕回, 走文字下方)
    path(s, [(13.53, 4.30), (14.30, 4.30), (14.30, 4.69), (11.24, 4.69)], w=2.6)
    # Student Network (收在容器内)
    cbox(s, 8.60, 5.20, 3.00, 0.92, WHITE, INK, 1.3, 0.10)
    txt(s, 8.60, 5.23, 3.00, 0.22, "Student Network", fs=12, bold=True, align=PP_ALIGN.CENTER)
    unet(s, 8.72, 5.46, 2.76, 0.58)

    # ================= 分界 =================
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.30), Inches(6.20),
                                Inches(15.70), Inches(6.20))
    ln.line.color.rgb = RGBColor(0x88, 0x88, 0x88); ln.line.width = Pt(1.2)
    ln.line._get_or_add_ln().append(
        ln.line._get_or_add_ln().makeelement(qn('a:prstDash'), {'val': 'sysDash'}))

    # ================= DEPLOYMENT =================
    cbox(s, 0.30, 6.24, 15.40, 2.66, C_DEPLOY, E_DEPLOY, 1.0, 0.02)
    pill(s, 0.46, 6.36, 3.95, 0.40, "DEPLOYMENT — IMAGE-ONLY INFERENCE", DEPLOY_TAG, fs=12.5)
    arr(s, 10.10, 6.10, 10.10, 6.90, w=1.7, color=RGBColor(0x66, 0x66, 0x66), dashed=True)

    y2, isz2 = 6.90, 1.42
    photo(s, f'{SRC}/p_edited.png', 0.62, y2, isz2, "Query Image\n(no key / no reference)")
    arr(s, 2.09, y2 + 0.42, 2.44, y2 + 0.42)
    cbox(s, 2.48, y2, 3.05, 1.22, WHITE, INK, 1.3, 0.10)
    txt(s, 2.48, y2 + 0.06, 3.05, 0.24, "Student Network", fs=11, bold=True, align=PP_ALIGN.CENTER)
    unet(s, 2.60, y2 + 0.30, 2.81, 0.58)
    txt(s, 2.48, y2 + 0.90, 3.05, 0.22, "Image-only · 1.1 ms per image", fs=8.6,
        color=RGBColor(0x33, 0x33, 0x33), align=PP_ALIGN.CENTER)
    arr(s, 5.58, y2 + 0.52, 5.94, y2 + 0.52)
    bord(s, 5.98, y2 + 0.10, 2.60, 1.04, "Evidence Fusion (optional)",
         "Visual + check evidence arbitration")
    arr(s, 8.64, y2 + 0.52, 9.02, y2 + 0.52)
    cbox(s, 8.98, y2 - 0.02, 3.72, 1.38, WHITE, INK, 1.3, 0.10)
    txt(s, 8.98, y2 + 0.04, 3.72, 0.24, "Verdict", fs=12.5, bold=True, align=PP_ALIGN.CENTER)
    for i, (c, lab) in enumerate(((GREEN, "Authentic"), (RED, "Tampered"), (ORNG, "Indeterminate"))):
        blk = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.06 + i * 1.22),
                                 Inches(y2 + 0.36), Inches(1.14), Inches(0.88))
        blk.adjustments[0] = 0.12
        blk.fill.solid(); blk.fill.fore_color.rgb = c
        blk.line.color.rgb = INK; blk.line.width = Pt(0.8)
        blk.shadow.inherit = False
        tf = blk.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.03)
        tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p_ = tf.paragraphs[0]; p_.text = lab; p_.alignment = PP_ALIGN.CENTER
        p_.font.size = Pt(8.6); p_.font.bold = True; p_.font.color.rgb = WHITE

    prs.save(f'{FIG}/cs_framework.pptx')
    print('-> cs_framework.pptx')
    subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', FIG,
                    f'{FIG}/cs_framework.pptx'], capture_output=True, timeout=180)
    import pymupdf
    pymupdf.open(f'{FIG}/cs_framework.pdf')[0].get_pixmap(dpi=200).save(f'{FIG}/cs_framework.png')
    print('exported cs_framework')




# =====================================================================
def motivation():
    """动机图：与框架图同一套配色（浅蓝容器 + 药丸标题 + 蓝箭头）"""
    prs = Presentation(); s = slide(prs, 16.0, 6.10)
    txt(s, 0.0, 0.16, 16.0, 0.36,
        "An invisible edit leaves a free, pixel-exact forensic trace",
        fs=19, bold=True, align=PP_ALIGN.CENTER)

    # ---- 两个容器 ----
    cbox(s, 0.30, 0.66, 7.55, 3.95, C_TRAIN, E_TRAIN, 1.0, 0.02)
    cbox(s, 8.15, 0.66, 7.55, 3.95, C_TRAIN, E_TRAIN, 1.0, 0.02)
    pill(s, 0.46, 0.78, 4.60, 0.40, "(A)  BEFORE THE EDIT — THE HOST", TRAIN_TAG, fs=12.5)
    pill(s, 8.31, 0.78, 5.35, 0.40, "(B)  AFTER THE EDIT — THE CHECK FAILS", TRAIN_TAG, fs=12.5)

    isz = 2.86
    # (A) 编辑图 + 红框
    xa, ya = 0.62, 1.38
    sh = s.shapes.add_picture(f'{SRC}/p_host.png', Inches(xa), Inches(ya),
                              Inches(isz), Inches(isz))
    sh.line.color.rgb = INK; sh.line.width = Pt(0.8)
    # 虚线红框: 将要被编辑的区域（提醒读者这是编辑前）
    rb = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(xa + 0.92), Inches(ya + 0.47),
                            Inches(1.08), Inches(1.92))
    rb.fill.background(); rb.line.color.rgb = RED; rb.line.width = Pt(2.0)
    rb.line._get_or_add_ln().append(
        rb.line._get_or_add_ln().makeelement(qn('a:prstDash'), {'val': 'dash'}))
    rb.shadow.inherit = False
    txt(s, xa + isz + 0.18, ya + 0.30, 1.42, 0.66,
        "red box =\nregion that will\nbe edited", fs=9.6, color=RGBColor(0x33,0x33,0x33))
    txt(s, xa + isz + 0.18, ya + 1.62, 1.42, 0.44, "watermark intact\neverywhere", fs=9.6,
        color=RGBColor(0x33,0x33,0x33))

    # (B) 失效高亮图 + 红框
    xb = 8.47
    sh = s.shapes.add_picture(f'{SRC}/p_teacher_orange.png', Inches(xb), Inches(ya),
                              Inches(isz), Inches(isz))
    sh.line.color.rgb = INK; sh.line.width = Pt(0.8)
    txt(s, xb + isz + 0.18, ya + 0.34, 1.55, 0.66,
        "orange = failed\nblock check\n0.501 in / 0.259 out", fs=9.6, color=RED, bold=True)
    txt(s, xb + isz + 0.18, ya + 1.55, 1.55, 0.50,
        "spatially exact,\ncosts nothing\nto collect", fs=9.6, color=RGBColor(0x33,0x33,0x33))

    # ---- 中间跨面板箭头 ----
    arr(s, 7.90, 3.10, 8.12, 3.10, w=2.6)

    # ---- 总结条 ----
    cbox(s, 0.30, 4.82, 15.40, 0.58, RGBColor(0xF4, 0xF6, 0xF8), RGBColor(0xC8, 0xD4, 0xDE), 1.0, 0.10)
    txt(s, 0.50, 4.94, 15.0, 0.32,
        "The eye cannot see the edit, but the block check fails exactly inside it — "
        "a label that costs nothing to collect",
        fs=13, bold=True, align=PP_ALIGN.CENTER)

    # ---- 任务定义行 ----
    txt(s, 0.30, 5.58, 15.40, 0.32,
        "KeyFree-Loc:  training may use the secret key  ·  inference sees the image only",
        fs=11.5, align=PP_ALIGN.CENTER, color=BLUE)

    prs.save(f'{FIG}/cs_motivation.pptx')
    print('-> cs_motivation.pptx')
    subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', FIG,
                    f'{FIG}/cs_motivation.pptx'], capture_output=True, timeout=180)
    import pymupdf
    pymupdf.open(f'{FIG}/cs_motivation.pdf')[0].get_pixmap(dpi=200).save(f'{FIG}/cs_motivation.png')
    print('exported cs_motivation')


if __name__ == '__main__':
    main()
    motivation()
