#!/usr/bin/env python3
"""生成动机图（fig:motivation / cs_motivation.pdf）——与框架图同一场景.

场景: T1 测试样本 b02608（素材由 figures_scripts/make_framework_mask.py 生成）
  (A) 宿主图 + 红框（将被编辑的区域）
  (B) 编辑图 + 冷色失效块（4×4 检测层实测: 区内 0.524 / 区外 0.260；色深∝3×3块级失效率）

输出: paper/cs_waterfail/figs/cs_motivation.pdf 与 figs/cs_motivation.pdf
"""
import os
import subprocess
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f'{BASE}/paper/latex/figs/src'
OUT_DIRS = [f'{BASE}/paper/cs_waterfail/figs', f'{BASE}/figs']

NAVY = RGBColor(0x11, 0x3A, 0x5C)
INK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x33, 0x33, 0x33)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x4B, 0x7A, 0x59)


def txt(s, x, y, w, h, text, fs=11, bold=False, color=INK, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = text; p.alignment = align
    p.font.size = Pt(fs); p.font.bold = bold; p.font.color.rgb = color
    return tb


def tag(s, x, y, text, edge, fs=10, bold=False):
    w = max(0.9, 0.085 * len(text) + 0.30)
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(x), Inches(y), Inches(w), Inches(0.30))
    sh.adjustments[0] = 0.25
    sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    sh.line.color.rgb = edge; sh.line.width = Pt(1.0)
    sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = text; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(fs); p.font.bold = bold; p.font.color.rgb = edge
    return sh


def main():
    prs = Presentation()
    W, H = 13.333, 6.62
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(H))
    bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    bg.line.fill.background(); bg.shadow.inherit = False

    txt(s, 0.42, 0.16, 12.5, 0.34,
        'An invisible edit leaves a free, pixel-exact forensic trace',
        fs=16, bold=True, color=INK)

    # ---- (A) 编辑前: 宿主图 + 红框
    txt(s, 0.75, 0.60, 4.60, 0.28, '(A)  Before the edit: the host', fs=12.5, bold=True, color=NAVY)
    pic = s.shapes.add_picture(f'{SRC}/p_scene_orig_box.png', Inches(0.75), Inches(0.94),
                               Inches(4.30), Inches(4.30))
    pic.line.color.rgb = RGBColor(0x11, 0x11, 0x11); pic.line.width = Pt(0.8)
    tag(s, 0.90, 1.08, 'watermark intact everywhere', GREEN, fs=9.5, bold=True)
    tag(s, 2.55, 4.92, 'red box = region that will be edited', RED, fs=9.5)

    # ---- (B) 编辑后: 块校验失效（橙色块）
    txt(s, 7.95, 0.60, 5.0, 0.28, '(B)  After the edit: the check fails', fs=12.5, bold=True, color=NAVY)
    pic = s.shapes.add_picture(f'{SRC}/p_scene_fail_overlay.png', Inches(7.95), Inches(0.94),
                               Inches(4.30), Inches(4.30))
    pic.line.color.rgb = RGBColor(0x11, 0x11, 0x11); pic.line.width = Pt(0.8)
    tag(s, 8.10, 1.08, 'blue = failed block check   0.524 in / 0.260 out', RED, fs=9.5, bold=True)
    tag(s, 8.10, 4.92, 'spatially exact, costs nothing to collect', GREY, fs=9.5)

    # ---- 底部一句话
    br = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.42), Inches(5.62),
                            Inches(12.48), Inches(0.58))
    br.adjustments[0] = 0.16
    br.fill.solid(); br.fill.fore_color.rgb = RGBColor(0xF4, 0xF6, 0xF8)
    br.line.color.rgb = RGBColor(0xD4, 0xD9, 0xDD); br.line.width = Pt(0.75)
    br.shadow.inherit = False
    tf = br.text_frame; tf.word_wrap = True; tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    p.text = ('The eye cannot see the edit, but the block check fails exactly inside it — '
              'a label that costs nothing to collect')
    p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = NAVY

    txt(s, 0.42, 6.28, 12.48, 0.28,
        'KeyFree-Loc:  the owner keeps the key at training time;  the deployed student sees only the image',
        fs=11, color=INK, align=PP_ALIGN.CENTER)

    os.makedirs(OUT_DIRS[0], exist_ok=True)
    pptx_path = f'{OUT_DIRS[0]}/cs_motivation.pptx'
    prs.save(pptx_path)
    subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', OUT_DIRS[0], pptx_path],
                   capture_output=True, timeout=240)
    for d in OUT_DIRS[1:]:
        if os.path.abspath(d) == os.path.abspath(OUT_DIRS[0]):
            continue
        os.makedirs(d, exist_ok=True)
        subprocess.run(['cp', f'{OUT_DIRS[0]}/cs_motivation.pdf', f'{d}/cs_motivation.pdf'], check=True)
    print('-> cs_motivation.pdf')


if __name__ == '__main__':
    main()
