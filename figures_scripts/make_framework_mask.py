#!/usr/bin/env python3
"""生成方法总览图（cs_framework.pptx）里"数据集图片"所需的素材，全部取自同一场景.

场景: T1 测试集样本 b02586（COCO + SD inpainting，插入物体，与 fig:qual 第 2 行同一场景）
模型: results/r73_delta25_student.pth（学生，δ=25 主配置，2×2 标签）

输出（都被 cs_framework.pptx 引用）:
  p_scene_orig.png  Host Image / Watermarked Image （宿主图，水印不可见）
  p_scene_edit.png  Edited Image / Query Image    （编辑图）
  p_scene_fail.png  Failure Map                   （教师告警：黑底 + 橙色失败块）
  p_scene_pred.png  Predicted mask                （学生掩膜：薄荷青填充 + 蓝色描边）

改完这些图后需重新导出 PDF:
    soffice --headless --convert-to pdf --outdir paper/cs_waterfail/figs \
            paper/cs_waterfail/figs/cs_framework.pptx
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SID = 'b02608'                       # T1 测试样本（编辑区为完整矩形, 画面干净、区内/外失效率差异直观）
SRC = f'{BASE}/paper/latex/figs/src'
FILL, EDGE = np.array([140, 240, 210], np.float32), (0, 95, 175)
COOL = (28, 104, 176)                # 失效块用冷色（青蓝）
COOL_ALPHA = 0.5                     # 半透明叠加，底图保持可见
RED = (200, 30, 30)                  # 编辑区域边框
MASTER = b'\x9a' + bytes(range(31))
BS, DELTA, COEF = 4, 25, (1, 1)      # 4x4 检测层（该样本实测: 区内 0.49 / 区外 0.25）


class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.c(x)


class SmallUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.e1, self.e2 = ConvBlock(3, base), ConvBlock(base, base * 2)
        self.e3, self.e4 = ConvBlock(base * 2, base * 4), ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2); self.d3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2); self.d2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2); self.d1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        c1 = self.e1(x); c2 = self.e2(self.pool(c1))
        c3 = self.e3(self.pool(c2)); c4 = self.e4(self.pool(c3))
        d3 = self.d3(torch.cat([self.up3(c4), c3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), c2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), c1], 1))
        return self.head(d1)


def _fail_map_raw(img, uid):
    """4x4 块级失效图（网格 64x64，与 code/r14b_multiscale.fail_bs 一致）。"""
    import keyed_scheme as ks
    key = ks.hkdf_sha256(ks.image_key(MASTER, uid), b'\x00' * 32,
                         f'wm/layer/bs{BS}/v1'.encode(), 32)
    H, W = img.shape[0] // BS * BS, img.shape[1] // BS * BS
    nh, nw = H // BS, W // BS
    pat = ks.check_pattern(key, nh, nw, 'coord')
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)[:H, :W]
    f = np.zeros((nh, nw), np.float32)
    for i in range(nh):
        for j in range(nw):
            c = cv2.dct(y[i * BS:(i + 1) * BS, j * BS:(j + 1) * BS])[COEF]
            f[i, j] = 1.0 if ks.qim_bit(c, DELTA) != pat[i, j] else 0.0
    return f


def add_box(img, gt, lw=2):
    ys, xs = np.where(gt > 0)
    out = img.copy()
    if len(xs):
        cv2.rectangle(out, (xs.min(), ys.min()), (xs.max(), ys.max()), RED, lw)
    return out


def main():
    import sys; sys.path.insert(0, f'{BASE}/code')
    import json
    meta = json.load(open(f'{BASE}/data/bigset/meta.json'))
    ids = [s['id'] for s in meta['samples']
           if os.path.exists(f'{BASE}/data/bigset/images/orig_{s["id"]}.png')]
    idx = ids.index(SID)
    assert idx < 2800, 'sample must be inside r70_data_d25.npz'

    z = np.load(f'{BASE}/results/r70_data_d25.npz')
    edit, alarm, gt = z['img'][idx], z['alarm'][idx], z['gt'][idx]
    orig = np.array(Image.open(f'{BASE}/data/bigset/images/orig_{SID}.png').convert('RGB'))

    m = SmallUNet(base=32)
    m.load_state_dict(torch.load(f'{BASE}/results/r73_delta25_student.pth', map_location='cpu'))
    m.eval()
    x = torch.from_numpy(edit.transpose(2, 0, 1)[None].astype(np.float32) / 255.)
    with torch.no_grad():
        pm = (torch.sigmoid(m(x))[0, 0].numpy() > 0.5)
    inter, union = (pm & (gt > 0)).sum(), (pm | (gt > 0)).sum()
    print(f'{SID}: gt {(gt>0).mean():.3f}, pred {pm.mean():.3f}, IoU {inter/union:.3f}')

    # 1) 宿主图 / 2) 编辑图（含编辑区域红框）
    Image.fromarray(orig).save(f'{SRC}/p_scene_orig.png')
    Image.fromarray(add_box(orig, gt)).save(f'{SRC}/p_scene_orig_box.png')
    Image.fromarray(edit).save(f'{SRC}/p_scene_edit.png')
    Image.fromarray(add_box(edit, gt)).save(f'{SRC}/p_scene_edit_box.png')

    # 3) 教师失效图: 真实 4x4 块级失效（黑底橙块 / 叠在编辑图上）
    fail_map_raw = _fail_map_raw(edit, SID)
    fm = np.kron(fail_map_raw, np.ones((BS, BS), np.float32)) > 0.5
    inside = (fm & (gt > 0)).sum() / max((gt > 0).sum(), 1)
    outside = (fm & ~(gt > 0)).sum() / max((~(gt > 0)).sum(), 1)
    print(f'4x4 failure map: overall {fm.mean():.3f}, inside {inside:.3f}, outside {outside:.3f}')
    # Failure Map（示意）: 2x2 池化到 32x32 网格、每格 16 px 显示 -> 缩印时块可辨;
    # 亮度按池化后的失效率分级, 并只保留失效块（阈值 0.30）, 使"亮块集中在编辑区"一目了然。
    # Failure Map（示意）: 9x9 平滑 -> 32x32 网格、每格 16 px 显示;
    # 只画失效块（阈值 0.30，区外约 1/4、区内约 1/2 的格），亮度随局部失效率分级：
    # 区内 -> 亮青, 区外 -> 暗青（保留零星斑块, 不画成全黑）。
    sm0 = cv2.blur(fail_map_raw, (9, 9), borderType=cv2.BORDER_REFLECT)
    pool = cv2.resize(sm0, (32, 32), interpolation=cv2.INTER_AREA)
    keep = pool > 0.30
    tone = np.clip((pool - 0.30) / 0.22, 0, 1)
    grid, pad = 16, 1                           # 格间缝隙 1 px, 保留"块"的形态
    canvas = np.zeros((grid * 32, grid * 32, 3), np.uint8)
    for i in range(32):
        for j in range(32):
            if not keep[i, j]:
                continue
            v = 0.20 + 0.80 * tone[i, j]        # 区外块 ~0.2（暗青）, 区内块 ~0.9（亮青）
            canvas[i * grid + pad:(i + 1) * grid - pad,
                   j * grid + pad:(j + 1) * grid - pad] = (np.array(COOL, np.float32) * v).astype(np.uint8)
    Image.fromarray(canvas).save(f'{SRC}/p_scene_fail.png')

    # 叠加版: 9x9 平滑（与教师端一致）让失效连成片, 透明度与该局部失效率成正比（线性, 不夸大）;
    # 块按 8 px 显示, 编辑区加红框以便对照。
    sm = cv2.blur(fail_map_raw, (9, 9), borderType=cv2.BORDER_REFLECT)
    big = np.array(Image.fromarray(edit).resize((512, 512), Image.LANCZOS))
    alpha_grid = np.clip((sm - 0.22) / 0.33, 0, 1) * 0.72 + 0.10   # 区内≈0.75 / 区外≈0.19
    A = np.kron(alpha_grid, np.ones((8, 8), np.float32))[..., None]
    over = np.clip(big.astype(np.float32) * (1 - A) + np.array(COOL, np.float32) * A, 0, 255)
    over = over.astype(np.uint8)
    ys, xs = np.where(gt > 0)
    cv2.rectangle(over, (xs.min() * 2, ys.min() * 2), (xs.max() * 2, ys.max() * 2), (200, 30, 30), 4)
    Image.fromarray(over).save(f'{SRC}/p_scene_fail_overlay.png')

    # 4) 学生预测: 编辑图上叠加薄荷青填充 + 蓝色描边
    out = edit.astype(np.float32).copy()
    out[pm] = out[pm] * 0.5 + FILL * 0.5
    out = np.clip(out, 0, 255).astype(np.uint8)
    mu = pm.astype(np.uint8)
    k = np.ones((7, 7), np.uint8)
    out[(cv2.dilate(mu, k) - cv2.erode(mu, k)).astype(bool)] = EDGE
    Image.fromarray(out).save(f'{SRC}/p_scene_pred.png')
    print('-> p_scene_{orig,edit,fail,pred}.png')


if __name__ == '__main__':
    main()
