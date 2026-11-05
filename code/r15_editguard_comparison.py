#!/usr/bin/env python3
"""r15 方法对比: EditGuard (CVPR'24, 可逆水印 + 篡改定位) vs 我们.

为什么必须比 EditGuard: 它是"用水印做篡改定位"的同赛道方法 (训练式),
WAM/TrustMark 都不是这个任务的最直接对手.

EditGuard 接口 (本仓库 data/editguard/EditGuard-main, 已做最小 basicsr shim):
  encode: load_image(img) -> model.image_hiding()  -> container 512x512
  decode: load_image(edited) -> model.host=GT; model.image_recovery(0.2) -> mask 512x512
  定位原理: 用原始 ref 模板 vs 重建 ref 的残差 > 阈值 -> 篡改区

协议与 r6/r7/r10/r12/r13 一致: 原图 256 -> 各方法嵌入 -> bbox paste SD 编辑 -> 定位.
评估统一到 256 尺度 (EditGuard mask resize 回 256), 指标: IoU@0.5, 定位 AUC, clean 假阳性面积.

输出: results/r15_editguard_comparison_<ts>.json
用法: apjf-python r15_editguard_comparison.py [--n 100]
"""
import sys, os, json, argparse, datetime, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EG = f'{BASE}/data/editguard/EditGuard-main'
EG_CODE = f'{EG}/code'
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EG_CODE)
os.chdir(EG_CODE)

import numpy as np
import cv2
from PIL import Image
import torch

import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs, loc_iou

DELTA = 15
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    args = ap.parse_args()
    t0 = time.time()

    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})
    import options.options as option
    from models import create_model
    from test_gradio import load_image

    opt = option.dict_to_nonedict(option.parse('options/test_editguard.yml', is_train=False))
    for k in ['sdinpaint', 'controlnetinpaint', 'sdxl', 'repaint', 'degrade_shuffle']:
        opt[k] = False
    eg = create_model(opt)
    eg.load_network(f'{EG}/checkpoints/clean.pth', eg.netG)
    eg.netG.to('cuda').eval()
    print(f'EditGuard 已加载 ({time.time()-t0:.1f}s)')

    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:args.n]
    msg = np.random.RandomState(42).choice([-0.5, 0.5], (1, 64)).astype(np.float32)
    msg_t = torch.from_numpy(msg).cuda()

    rows = []
    for idx, s in enumerate(samples):
        sid = s['id']
        try:
            o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        key = ks.image_key(MASTER, sid)
        H, W = o.shape[:2]
        x1, y1, x2, y2 = s['bbox']
        bx1, by1, bx2, by2 = int(round(x1*W)), int(round(y1*H)), int(round(x2*W)), int(round(y2*H))

        # ---------- EditGuard (512)
        with torch.no_grad():
            d = load_image(o, msg)
            eg.real_H, eg.ref_L, eg.mes = d['GT'].cuda(), d['LQ'].cuda(), msg_t
            wm512 = np.array(eg.image_hiding())
            S = 512
            ebx1, eby1, ebx2, eby2 = int(x1*S), int(y1*S), int(x2*S), int(y2*S)
            att512 = np.array(Image.fromarray(a).resize((S, S), Image.BICUBIC))
            edit = wm512.copy()
            edit[eby1:eby2, ebx1:ebx2] = att512[eby1:eby2, ebx1:ebx2]
            d2 = load_image(edit, msg)
            eg.real_H = d2['GT'].cuda()
            eg.host = d2['GT'].cuda()
            eg.ref_L = d2['LQ'].cuda()
            eg.mes = msg_t
            mask512, _ = eg.image_recovery(0.2)      # (512,512) 0-3
            # clean 参考
            d3 = load_image(wm512, msg)
            eg.real_H = d3['GT'].cuda()
            eg.host = d3['GT'].cuda()
            eg.ref_L = d3['LQ'].cuda()
            eg.mes = msg_t
            mask_clean512, _ = eg.image_recovery(0.2)

        pred512 = np.array(mask512) > 0
        pred = cv2.resize(pred512.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        gt = np.zeros((H, W), bool)
        gt[by1:by2, bx1:bx2] = True
        inter, union = (pred & gt).sum(), (pred | gt).sum()
        iou_eg = float(inter / union) if union else 0.0
        in_f = float(pred[gt].mean()) if gt.any() else float('nan')
        out_f = float(pred[~gt].mean()) if (~gt).any() else float('nan')
        clean_area = float((np.array(mask_clean512) > 0).mean())

        # ---------- 我们 (8x8 + 4x4)
        k8, k4 = layer_key(key, 'bs8'), layer_key(key, 'bs4')
        wm = ks.embed_keyed(o, k8, DELTA, PAY, CHK, pattern='coord')
        wm2 = embed_bs(wm, k4, 4, DELTA, (1, 1))
        att, box = paste_edit(wm2, a, s['bbox'])
        f8 = fail_bs(att, k8, 8, DELTA, CHK)
        f4 = fail_bs(att, k4, 4, DELTA, (1, 1))
        iou8 = loc_iou(f8, s['bbox'], 8, 5, 0.38, 8)
        iou4 = loc_iou(f4, s['bbox'], 4, 5, 0.38, 32)

        rows.append({
            'id': sid,
            'eg': {'iou': iou_eg, 'in_fail': in_f, 'out_fail': out_f, 'clean_area': clean_area},
            'ours': {'iou8': iou8, 'iou4': iou4},
        })
        if (idx + 1) % 20 == 0:
            print(f'  [{idx+1}/{len(samples)}] {time.time()-t0:.0f}s', flush=True)

    def mean(path):
        v = [r[path[0]][path[1]] for r in rows]
        v = [x for x in v if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v))

    res = {
        'meta': {'ts': datetime.datetime.now().isoformat(), 'n': len(rows),
                 'protocol': 'attackset SD inpaint + bbox paste; 评估统一到 256 尺度',
                 'editguard_ckpt': f'{EG}/checkpoints/clean.pth',
                 'threshold': 0.2},
        'editguard': {'iou': mean(('eg', 'iou')), 'in_fail': mean(('eg', 'in_fail')),
                      'out_fail': mean(('eg', 'out_fail')),
                      'clean_area': mean(('eg', 'clean_area'))},
        'ours': {'iou_8x8': mean(('ours', 'iou8')), 'iou_4x4': mean(('ours', 'iou4'))},
        'runtime_sec': round(time.time() - t0, 1),
    }
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r15_editguard_comparison_{fn}.json'
    json.dump(res, open(fp, 'w'), indent=1)
    print(f'\n=== n={len(rows)} ===')
    print(f'  EditGuard : IoU {res["editguard"]["iou"]:.3f}  clean 假阳性面积 '
          f'{res["editguard"]["clean_area"]*100:.2f}%  in/out '
          f'{res["editguard"]["in_fail"]*100:.1f}/{res["editguard"]["out_fail"]*100:.1f}%')
    print(f'  我们      : IoU 8x8 {res["ours"]["iou_8x8"]:.3f} / 4x4 {res["ours"]["iou_4x4"]:.3f}')
    print(f'结果: {fp}  ({res["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
