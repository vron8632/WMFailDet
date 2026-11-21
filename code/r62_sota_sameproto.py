#!/usr/bin/env python3
"""r62: 同口径 SOTA 实测 — EditGuard / WAM 在 r61 的同一批测试样本上.

协议: 各方法使用自己的水印嵌入器(它们的设计前提), 同一批 COCO 原图 + 同一编辑(bbox paste)
      + 同一 GT, 报告 IoU/覆盖/面积比.
口径: r61 的 400 张测试样本 (bigset 的后 400 张)
输出: results/r62_sota_sameproto.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit

BIG = f'{BASE}/data/bigset'
N_TEST = 400


def region_metrics(pred, gt):
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    iou = inter / union if union else 1.0
    g = float(gt.sum())
    return iou, (inter / g if g else 0.0), (float(pred.sum()) / max(1.0, g))


def summarize(rows):
    ious = [r[0] for r in rows]
    covs = [r[1] for r in rows]
    ars = [r[2] for r in rows]
    return dict(iou=round(float(np.mean(ious)), 4),
                iou_std=round(float(np.std(ious)), 4),
                cov50=round(float(np.mean([c >= .5 for c in covs])), 4),
                area_ratio=round(float(np.mean(ars)), 3), n=len(rows))


def load_items(n):
    meta = json.load(open(f'{BIG}/meta.json'))
    items = []
    for s_ in meta['samples'][-n:]:
        sid = s_['id']
        fo, fa = f'{BIG}/images/orig_{sid}.png', f'{BIG}/images/att_{sid}.png'
        if not (os.path.exists(fo) and os.path.exists(fa)):
            continue
        o = r8.crop8(np.array(Image.open(fo).convert('RGB')))
        a = r8.crop8(np.array(Image.open(fa).convert('RGB')))
        H, W = o.shape[:2]
        gt = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = s_['bbox']
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
        items.append((sid, o, a, s_['bbox'], gt))
    return items


def run_editguard(items):
    """复用 r30 的 EditGuard 流程 (已验证可用); 返回逐样本指标."""
    EG = f'{BASE}/data/editguard/EditGuard-main'; EGC = f'{EG}/code'
    sys.path.insert(0, EGC)
    cwd = os.getcwd(); os.chdir(EGC)
    import torch
    _orig = torch.load
    torch.load = lambda *aa, **kk: _orig(*aa, **{**kk, 'weights_only': False})
    import options.options as option
    from models import create_model
    from test_gradio import load_image
    opt = option.dict_to_nonedict(option.parse('options/test_editguard.yml', is_train=False))
    for k in ['sdinpaint', 'controlnetinpaint', 'sdxl', 'repaint', 'degrade_shuffle']:
        opt[k] = False
    eg = create_model(opt)
    eg.load_network(f'{EG}/checkpoints/clean.pth', eg.netG)
    eg.netG.to('cuda').eval()
    msg = np.random.RandomState(7).choice([-0.5, 0.5], (1, 64)).astype(np.float32)
    msg_t = torch.from_numpy(msg).cuda()
    rows = []; t0 = time.time()
    for i, (sid, o, a, bb, gt) in enumerate(items):
        H, W = o.shape[:2]; x1, y1, x2, y2 = bb
        with torch.no_grad():
            d = load_image(o, msg)
            eg.real_H = d['GT'].cuda(); eg.ref_L = d['LQ'].cuda(); eg.mes = msg_t
            wm = np.array(eg.image_hiding())
            S = 512
            e1, f1, e2, f2 = int(x1*S), int(y1*S), int(x2*S), int(y2*S)
            at512 = np.array(Image.fromarray(a).resize((S, S), Image.BICUBIC))
            edit = wm.copy(); edit[f1:f2, e1:e2] = at512[f1:f2, e1:e2]
            d2 = load_image(edit, msg)
            eg.real_H = d2['GT'].cuda(); eg.host = d2['GT'].cuda()
            eg.ref_L = d2['LQ'].cuda(); eg.mes = msg_t
            mk512, _ = eg.image_recovery(0.2)
        pm = cv2.resize((np.array(mk512) > 0).astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST)
        rows.append(region_metrics(pm, gt))
        if (i + 1) % 50 == 0:
            print(f'  eg {i+1}/{len(items)} ({time.time()-t0:.0f}s)', flush=True)
    os.chdir(cwd)
    return summarize(rows) if rows else None


def main():
    items = load_items(N_TEST)
    print(f'test items: {len(items)}', flush=True)
    out = {'meta': dict(n=len(items), protocol='each method uses its own embedder; '
                        'same COCO originals, same bbox-paste edits, same GT',
                        note='r61 L2 student on this split: IoU 0.9347')}
    eg = run_editguard(items)
    if eg:
        out['editguard'] = eg
        print('[EditGuard]', eg, flush=True)
    json.dump(out, open(f'{BASE}/results/r62_sota_sameproto.json', 'w'), indent=1)
    print('-> results/r62_sota_sameproto.json')


if __name__ == '__main__':
    main()
