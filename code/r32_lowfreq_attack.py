#!/usr/bin/env python3
"""r32: 低频保持型编辑对抗实验 (v2, Y-channel exact).

威胁模型：攻击者已知 QIM 在 8x8 DCT 的 (1,0)/(1,1)（YCrCb 的 Y 通道）。
编辑内容后，把编辑区域内每块的这两个系数精确恢复为水印图原值（其余系数保留编辑结果）。
即"内容被改、校验位未破"。

若失效信号被绕过，编辑区失败率应回落到干净底噪 (~25%)，检测统计量随之失效。
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit

DELTA = 15; PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
K_LOC, T_LOC = 9, 0.38


def paste_lowfreq_keep(wm, att, bbox, keep=(PAY, CHK)):
    """编辑内容来自 att，但 (1,0)/(1,1) 系数精确恢复为 wm 原值（Y 通道）。"""
    H, W = wm.shape[0] // 8 * 8, wm.shape[1] // 8 * 8
    yw = cv2.cvtColor(wm, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)[:H, :W].copy()
    ya = cv2.cvtColor(att, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)[:H, :W]
    x1, y1, x2, y2 = bbox
    by1, by2 = int(y1 * H) // 8 * 8, int(np.ceil(y2 * H / 8) * 8)
    bx1, bx2 = int(x1 * W) // 8 * 8, int(np.ceil(x2 * W / 8) * 8)
    for by in range(by1, min(by2, H), 8):
        for bx in range(bx1, min(bx2, W), 8):
            Ca = cv2.dct(ya[by:by+8, bx:bx+8])
            Cw = cv2.dct(yw[by:by+8, bx:bx+8])
            for uv in keep:
                Ca[uv] = Cw[uv]                      # 精确恢复 QIM 系数
            yw[by:by+8, bx:bx+8] = cv2.idct(Ca)
    ycrcb = cv2.cvtColor(wm, cv2.COLOR_RGB2YCrCb)
    out = ycrcb.copy()
    out[:H, :W, 0] = np.clip(yw, 0, 255)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB).astype(np.uint8)


def region_fail(fail, bbox):
    h, w = fail.shape
    x1, y1, x2, y2 = bbox
    f = fail[int(y1*h):max(int(y1*h)+1, int(y2*h)), int(x1*w):max(int(x1*w)+1, int(x2*w))]
    return float(f.mean()) if f.size else float('nan')


def stats(fail):
    s = ks.smooth_fail_map(fail, K_LOC)
    v = np.sort(s.ravel()); k = max(1, int(0.10 * v.size))
    return float(v[-k:].mean())


def main(n=120):
    meta = json.load(open(f'{AS}/meta.json'))
    rows = []; t0 = time.time()
    for s_ in meta['samples'][:n]:
        sid = s_['id']
        try:
            o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except Exception:
            continue
        key = ks.image_key(MASTER, sid)
        wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        bb = s_['bbox']
        att_n, _ = paste_edit(wm, a, bb)
        att_l = paste_lowfreq_keep(wm, a, bb)
        f_n = ks.fail_map(att_n, key, DELTA, CHK, 'coord')
        f_l = ks.fail_map(att_l, key, DELTA, CHK, 'coord')
        f_c = ks.fail_map(wm, key, DELTA, CHK, 'coord')
        rows.append(dict(clean=float(f_c.mean()),
                         normal_in=region_fail(f_n, bb), lowfreq_in=region_fail(f_l, bb),
                         top_clean=stats(f_c), top_normal=stats(f_n), top_lowfreq=stats(f_l),
                         psnr_nl=float(10*np.log10(255**2/max(1e-6, np.mean(
                             (att_l.astype(float)-att_n.astype(float))**2))))))
        if len(rows) >= n: break

    def m(k): return float(np.mean([r[k] for r in rows]))
    print(f'n={len(rows)}  ({time.time()-t0:.0f}s)')
    print(f'{"指标":26s}{"干净":>9s}{"普通编辑":>10s}{"低频保持":>10s}')
    for k, lab in [('clean', '编辑区失败率'), ('normal_in', ''), ('lowfreq_in', '')]:
        pass
    print(f'{"编辑区逐块失败率":26s}{m("clean"):9.3f}{m("normal_in"):10.3f}{m("lowfreq_in"):10.3f}')
    print(f'{"top-10% 统计量":26s}{m("top_clean"):9.3f}{m("top_normal"):10.3f}{m("top_lowfreq"):10.3f}')
    print(f'{"普通 vs 低频保持 PSNR":26s}{"—":>9s}{"—":>10s}{m("psnr_nl"):10.2f} dB')
    out = dict(n=len(rows), clean_fail=m('clean'), normal_region_fail=m('normal_in'),
               lowfreq_region_fail=m('lowfreq_in'), top_clean=m('top_clean'),
               top_normal=m('top_normal'), top_lowfreq=m('top_lowfreq'),
               psnr_normal_vs_lowfreq=m('psnr_nl'))
    json.dump(out, open(f'{BASE}/results/r32_lowfreq_attack.json', 'w'), indent=1)
    print('-> results/r32_lowfreq_attack.json')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
