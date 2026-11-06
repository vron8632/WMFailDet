#!/usr/bin/env python3
"""Editable PowerPoint architecture figure (conference style).

Canvas 13.333x7.5 in (fills 16:9), thumbnails 1.44 in, fonts 11-16 pt.
Colour code carries meaning and is declared in the legend:
  violet  = key derivation      blue  = our detection core
  green   = decision            orange= synchronisation
  red     = abstain / edit      grey  = off-the-shelf component
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = f'{BASE}/paper/framework_refs'
OUT = f'{BASE}/figs/framework.pptx'
os.makedirs(f'{BASE}/figs', exist_ok=True)

W, H = 13.333, 7.5
INK = RGBColor(0x1A, 0x1A, 0x1A)
GREY_E, GREY_F = RGBColor(0x9A, 0x9A, 0x9A), RGBColor(0xF4, 0xF4, 0xF4)
VIO_E, VIO_F = RGBColor(0x6B, 0x5B, 0x95), RGBColor(0xF1, 0xEE, 0xF8)
BLU_E, BLU_F = RGBColor(0x0F, 0x4D, 0x92), RGBColor(0xE4, 0xED, 0xF9)
GRN_E, GRN_F = RGBColor(0x2F, 0x6B, 0x3F), RGBColor(0xE8, 0xF3, 0xE9)
ORA_E, ORA_F = RGBColor(0xB5, 0x65, 0x1D), RGBColor(0xFC, 0xF1, 0xE4)
RED = RGBColor(0xB6, 0x43, 0x42)
PANEL = RGBColor(0xC8, 0xC8, 0xC8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = 'Arial'

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
slide = prs.slides.add_slide(prs.slide_layouts[6])

def X(f): return Inches(f * W)
def Y(f): return Inches((1.0 - f) * H)
def Wd(f): return Inches(f * W)
def Hd(f): return Inches(f * H)

def shape(kind, x0, y0, x1, y1, text='', ec=GREY_E, fc=GREY_F, tc=INK,
          fs=13, bold=False, lw=1.5, name=None, dash=None, align=PP_ALIGN.CENTER, nofill=False):
    s = slide.shapes.add_shape(kind, X(x0), Y(y1), Wd(x1-x0), Hd(y1-y0))
    if name: s.name = name
    if nofill:
        s.fill.background()
    else:
        s.fill.solid(); s.fill.fore_color.rgb = fc
    s.line.color.rgb = ec; s.line.width = Pt(lw)
    if dash: s.line.dash_style = dash
    s.shadow.inherit = False
    try: s.adjustments[0] = 0.05
    except Exception: pass
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln in enumerate((text.split('\n') if text else [''])):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = ln
        r.font.name = FONT; r.font.size = Pt(fs); r.font.bold = bold
        r.font.color.rgb = tc
    return s

def label(x, y, text, fs=12, color=INK, bold=False, align=PP_ALIGN.CENTER, w=0.60, name=None):
    s = slide.shapes.add_textbox(X(x - w/2), Y(y) - Hd(0.016), Wd(w), Hd(0.032))
    if name: s.name = name
    tf = s.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.name = FONT; r.font.size = Pt(fs); r.font.bold = bold
    r.font.color.rgb = color
    return s

def label_left(xl, y, text, fs=12, color=INK, bold=False, w=0.5, name=None):
    s = slide.shapes.add_textbox(X(xl), Y(y) - Hd(0.016), Wd(w), Hd(0.032))
    if name: s.name = name
    tf = s.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = text
    r.font.name = FONT; r.font.size = Pt(fs); r.font.bold = bold
    r.font.color.rgb = color
    return s

def line(x0, y0, x1, y1, color=INK, lw=1.5, head=True, dash=None, name=None):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, X(x0), Y(y0), X(x1), Y(y1))
    if name: c.name = name
    c.line.color.rgb = color; c.line.width = Pt(lw)
    if dash: c.line.dash_style = dash
    if head:
        ln = c.line._get_or_add_ln()
        ln.append(ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}))
    return c

def pic(x, y, w, path, name=None):
    h = w * W / H
    p = slide.shapes.add_picture(path, X(x), Y(y + h), Wd(w), Hd(h))
    if name: p.name = name
    return p

def grid(x0, y0, x1, y1, n, highlight, fill_hi, name):
    cw, ch = (x1 - x0) / n, (y1 - y0) / n
    for r in range(n):
        for c in range(n):
            fc = fill_hi if (r, c) in highlight else WHITE
            cell = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, X(x0 + c*cw), Y(y1 - r*ch), Wd(cw), Hd(ch))
            cell.name = f'{name}_{r}_{c}'
            cell.fill.solid(); cell.fill.fore_color.rgb = fc
            cell.line.color.rgb = RGBColor(0xBB, 0xBB, 0xBB); cell.line.width = Pt(0.6)
            cell.shadow.inherit = False
            cell.text_frame.paragraphs[0].text = ''

# ---------------- panels (fill the 16:9 canvas) ----------------
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.008, 0.045, 0.153, 0.965, '', PANEL, WHITE, lw=1.1, name='Panel Key layer')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.160, 0.655, 0.998, 0.965, '', PANEL, WHITE, lw=1.1, name='Panel Publish')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.160, 0.045, 0.998, 0.645, '', PANEL, WHITE, lw=1.1, name='Panel Audit')

label(0.0805, 0.935, '(0) Key layer', fs=16, bold=True, w=0.9, name='Title key layer')
label_left(0.172, 0.935, '(a) Publish-time embedding', fs=16, bold=True, w=4.0, name='Title publish')
label_left(0.172, 0.618, '(b) Blind auditing', fs=16, bold=True, w=3.0, name='Title audit')

# ---------------- key layer ----------------
label(0.0805, 0.898, 'K, u', fs=13, name='Key input')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.020, 0.822, 0.141, 0.870, 'HKDF', GREY_E, GREY_F, INK, 12, name='Box HKDF')
line(0.0805, 0.820, 0.0805, 0.799, BLU_E, 1.5, name='Arrow HKDF to ku')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.050, 0.757, 0.111, 0.797, 'k_u', GREY_E, GREY_F, INK, 14, name='Box ku')
line(0.0805, 0.755, 0.0805, 0.732, BLU_E, 1.5, head=False, name='Line ku split')
line(0.032, 0.732, 0.129, 0.732, BLU_E, 1.5, head=False, name='Line split bar')
line(0.032, 0.732, 0.032, 0.714, BLU_E, 1.5, name='Arrow to coord PRF')
line(0.129, 0.732, 0.129, 0.714, BLU_E, 1.5, name='Arrow to payload PRF')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.008, 0.612, 0.073, 0.712, 'coord\nPRF\nE_u(i,j)', BLU_E, BLU_F, BLU_E, 11, lw=1.8, name='Box coord PRF')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.080, 0.612, 0.145, 0.712, 'payload\nPRF\nM(i,j)', BLU_E, BLU_F, BLU_E, 11, lw=1.8, name='Box payload PRF')

grid(0.018, 0.320, 0.135, 0.520, 8, {(0, 0), (0, 1)}, BLU_E, 'Inset A coord-PRF 8x8 grid')
label(0.0765, 0.290, 'E_u(i,j) = PRF(k_u, i, j)', fs=11, color=BLU_E, w=0.9, name='Formula E_u')
line(0.032, 0.610, 0.032, 0.522, BLU_E, 1.0, head=False, dash=MSO_LINE_DASH_STYLE.DASH, name='Guide to inset A')

line(0.146, 0.700, 0.158, 0.700, BLU_E, 2.5, name='Arrow key to publish')
line(0.146, 0.362, 0.158, 0.362, BLU_E, 2.5, name='Arrow key to audit')

# ---------------- (a) publish ----------------
pic(0.168, 0.690, 0.108, f'{REFS}/ref_host.png', name='Image I')
label(0.222, 0.672, 'I', fs=13, name='Label I')
line(0.278, 0.785, 0.296, 0.785, INK, 1.5, name='Arrow I to QIM')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.300, 0.700, 0.530, 0.855,
      'QIM embedding\n8×8 DCT,  δ = 15\n(1,0) = M,   (1,1) = E_u', BLU_E, BLU_F, BLU_E, 13, bold=True, lw=1.8, name='Box QIM')
line(0.532, 0.785, 0.550, 0.785, INK, 1.5, name='Arrow QIM to W')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.554, 0.750, 0.634, 0.820, 'W', GREY_E, GREY_F, INK, 14, name='Box W')
line(0.636, 0.785, 0.654, 0.785, INK, 1.5, name='Arrow W to image')
pic(0.658, 0.690, 0.108, f'{REFS}/ref_watermarked.png', name='Image W')

line(0.712, 0.688, 0.712, 0.660, INK, 2.0, head=False, name='Channel 1')
line(0.712, 0.660, 0.222, 0.660, INK, 2.0, head=False, name='Channel 2')
line(0.222, 0.660, 0.222, 0.616, INK, 2.0, name='Channel 3 arrow')
label(0.470, 0.668, 'distribution channel  (editing · JPEG · feathering)', fs=11, w=4.2, name='Label distribution')

# ---------------- (b) audit ----------------
pic(0.168, 0.420, 0.108, f'{REFS}/ref_edited_overlay.png', name="Image I'")
label(0.222, 0.402, "I'", fs=13, name='Label I prime')
line(0.278, 0.516, 0.296, 0.516, INK, 1.5, name='Arrow I to verify')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.300, 0.450, 0.480, 0.582, 'block verify\nvs E_u → F', BLU_E, BLU_F, BLU_E, 13, bold=True, lw=1.8, name='Box verify')
line(0.482, 0.516, 0.500, 0.516, INK, 1.5, name='Arrow verify to F')
pic(0.504, 0.420, 0.108, f'{REFS}/ref_failure_map.png', name='Image F failmap')
label(0.558, 0.402, 'F', fs=13, name='Label F')

line(0.614, 0.516, 0.630, 0.516, INK, 1.5, head=False, name='Branch stem')
line(0.630, 0.545, 0.630, 0.300, INK, 1.5, head=False, name='Branch bar')
line(0.630, 0.545, 0.648, 0.545, INK, 1.5, name='Arrow to detect')
line(0.630, 0.300, 0.648, 0.300, INK, 1.5, name='Arrow to sync')

shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.652, 0.495, 0.790, 0.595, 'detect\ntop-10% / contrast', BLU_E, BLU_F, BLU_E, 13, bold=True, lw=1.8, name='Box detect')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.652, 0.250, 0.790, 0.350, 'sync\n64 phases × offsets', BLU_E, BLU_F, BLU_E, 13, bold=True, lw=1.8, name='Box sync')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.810, 0.510, 0.888, 0.580, 'EDITED', BLU_E, BLU_F, BLU_E, 13, bold=True, lw=1.8, name='Box EDITED')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.810, 0.395, 0.888, 0.465, 'BENIGN', GREY_E, GREY_F, INK, 13, lw=1.5, name='Box BENIGN')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.810, 0.250, 0.888, 0.320, 'ABSTAIN', RED, WHITE, RED, 13, bold=True, lw=1.8, name='Box ABSTAIN')
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 0.908, 0.480, 0.990, 0.610, 'region alarm\n0.38 + CC', BLU_E, BLU_F, BLU_E, 12, bold=True, lw=1.8, name='Box region alarm')

line(0.792, 0.545, 0.808, 0.545, INK, 1.5, name='Arrow detect to EDITED')
line(0.890, 0.545, 0.906, 0.545, INK, 1.5, name='Arrow EDITED to alarm')
line(0.721, 0.493, 0.721, 0.430, INK, 1.5, head=False, name='Else stem')
line(0.721, 0.430, 0.808, 0.430, INK, 1.5, name='Arrow to BENIGN')
line(0.792, 0.285, 0.808, 0.285, INK, 1.5, name='Arrow sync to ABSTAIN')
label_left(0.892, 0.285, 'degenerate', fs=10, color=RED, w=0.105, name='Label degenerate')
label(0.949, 0.452, '4×4 refine (gated)', fs=11, w=1.3, name='Label refine')

grid(0.826, 0.130, 0.871, 0.210, 4, set(), WHITE, 'Inset B sync 4x4 grid')
shape(MSO_SHAPE.RECTANGLE, 0.826, 0.130, 0.871, 0.210, '', BLU_E, WHITE, lw=1.4, nofill=True, name='Inset B phase frame')
shape(MSO_SHAPE.RECTANGLE, 0.836, 0.122, 0.881, 0.202, '', RED, WHITE, lw=1.4, dash=MSO_LINE_DASH_STYLE.DASH, nofill=True, name='Inset B offset frame')
label(0.849, 0.100, 'offset search', fs=10, w=0.9, name='Label offset search')
line(0.849, 0.248, 0.849, 0.212, GREY_E, 1.0, head=False, dash=MSO_LINE_DASH_STYLE.DASH, name='Guide to inset B')

# ---------------- legend (colour code) ----------------
LX, LY = 0.172, 0.112
items = [('our method', BLU_F, BLU_E), ('abstain / edit', WHITE, RED), ('off-the-shelf', GREY_F, GREY_E)]
for i, (txt, fc, ec) in enumerate(items):
    cx = LX + i * 0.160
    cy = LY
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, X(cx), Y(cy + 0.011), Wd(0.016), Hd(0.022))
    sh.name = f'Legend swatch {txt}'
    sh.fill.solid(); sh.fill.fore_color.rgb = fc
    sh.line.color.rgb = ec; sh.line.width = Pt(1.2)
    sh.shadow.inherit = False
    label_left(cx + 0.021, cy, txt, fs=9.5, w=0.130, name=f'Legend label {txt}')

prs.save(OUT)
print('saved', OUT, 'shapes:', len(slide.shapes))
