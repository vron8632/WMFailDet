#!/usr/bin/env python3
"""r42: 编辑类型覆盖 (LaMa 非扩散 / IP2P 高清) + 计算成本实测.

输出: results/r42_coverage_cost.json
"""
import sys, os, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs
from r34_distill_matrix import SmallUNet, eval_mask, alarm_from_fail, DELTA, PAY, CHK, MASTER


def load_any(root, n=None):
    meta = json.load(open(f'{root}/meta.json'))
    out = []
    for s_ in meta['samples']:
        sid = s_['id']
        fpo, fpa = f'{root}/images/orig_{sid}.png', f'{root}/images/att_{sid}.png'
        if not (os.path.exists(fpo) and os.path.exists(fpa)):
            continue
        try:
            o = np.array(Image.open(fpo).convert('RGB'))
            a = np.array(Image.open(fpa).convert('RGB'))
        except Exception:
            continue
        if a.shape[:2] != o.shape[:2]:
            a = np.array(Image.fromarray(a).resize((o.shape[1], o.shape[0]), Image.LANCZOS))
        out.append((str(sid), o, a, s_['bbox']))
        if n and len(out) >= n:
            break
    return out


def evaluate(m, items, dev, scale=256):
    eds, als, gts = [], [], []
    for sid, o, a, bb in items:
        o8 = r8.crop8(o); a8 = r8.crop8(a)
        key = ks.image_key(MASTER, sid); k4 = layer_key(key, 'bs4')
        wm = ks.embed_keyed(o8, key, DELTA, PAY, CHK, pattern='coord')
        wm = embed_bs(wm, k4, 4, DELTA, (1, 1))
        edited, _ = paste_edit(wm, a8, bb)
        f4 = fail_bs(edited, k4, 4, DELTA, (1, 1))
        al = alarm_from_fail(f4, 4, 5, 32)
        H, W = edited.shape[:2]
        gt = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = bb
        gt[int(y1*H):int(y2*H), int(x1*W):int(x2*W)] = 1
        if edited.shape[0] != scale:
            edited = np.array(Image.fromarray(edited).resize((scale, scale), Image.LANCZOS))
        eds.append(edited); als.append(al); gts.append(gt)
    eds = np.array(eds)
    m.eval(); pr = []
    with torch.no_grad():
        for i in range(0, len(eds), 16):
            xb = torch.from_numpy(eds[i:i+16].transpose(0, 3, 1, 2)).float().to(dev)/255.
            pr.append((torch.sigmoid(m(xb)).cpu().numpy()[:, 0] > 0.5).astype(np.uint8))
    pr = np.concatenate(pr)
    # 逐样本算 IoU (GT 尺寸可能不一致, 不能 np.array)
    def _iou(p, g):
        p = p.astype(bool)
        if p.shape != g.shape:
            p = cv2.resize(p.astype(np.uint8), (g.shape[1], g.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        i = np.logical_and(p, g).sum(); u = np.logical_or(p, g).sum()
        return (i / u if u else 1.0), (i / g.sum() if g.sum() else 0.0), (p.sum() / max(1, g.sum()))

    def agg(preds):
        ious, covs, ars = [], [], []
        for p, g in zip(preds, gts):
            i, c, a = _iou(p, g)
            ious.append(i); covs.append(c); ars.append(a)
        return dict(iou=float(np.mean(ious)),
                    cov50=float(np.mean([c >= .5 for c in covs])),
                    area_ratio=float(np.mean(ars)))

    ss = agg(pr)
    ts = agg(als)
    return ss, ts, len(eds)


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SmallUNet(base=32).to(dev)
    m.load_state_dict(torch.load(f'{BASE}/results/r43_best_student.pth', map_location=dev))
    out = {'meta': dict(student='r34 r41-seed0 alarm4 (COCO-trained)'), 'domains': {}, 'cost': {}}
    t0 = time.time()

    # LaMa (非扩散)
    items = load_any(f'{BASE}/data/xdomain/lama_edits', 200)
    ss, ts, n = evaluate(m, items, dev)
    out['domains']['lama'] = {'n': n, 'student': ss, 'teacher4': ts}
    print(f'[LaMa] n={n} student={ss["iou"]:.3f} teacher4={ts["iou"]:.3f}', flush=True)

    # hires (IP2P 640x480) -> 学生 256 推理后上采样
    items = load_any(f'{BASE}/data/hires_expanded', 140)
    ss, ts, n = evaluate(m, items, dev, scale=256)
    out['domains']['hires_ip2p'] = {'n': n, 'student': ss, 'teacher4': ts}
    print(f'[hires] n={n} student={ss["iou"]:.3f} teacher4={ts["iou"]:.3f}', flush=True)

    # 成本
    nparam = sum(p.numel() for p in m.parameters())
    x = torch.randn(1, 3, 256, 256).to(dev)
    for _ in range(5):
        with torch.no_grad():
            m(x)
    torch.cuda.synchronize() if dev == 'cuda' else None
    t1 = time.time()
    for _ in range(20):
        with torch.no_grad():
            m(x)
    torch.cuda.synchronize() if dev == 'cuda' else None
    gpu_ms = (time.time() - t1) / 20 * 1000
    mc = SmallUNet(base=32)
    xc = torch.randn(1, 3, 256, 256)
    for _ in range(2):
        with torch.no_grad():
            mc(xc)
    t2 = time.time()
    for _ in range(10):
        with torch.no_grad():
            mc(xc)
    cpu_ms = (time.time() - t2) / 10 * 1000
    out['cost'] = {'params': int(nparam), 'params_mb': round(nparam*4/1e6, 2),
                   'gpu_ms_per_image': round(gpu_ms, 2), 'cpu_ms_per_image': round(cpu_ms, 2),
                   'device': dev}
    print(f'[cost] params={nparam/1e6:.2f}M  GPU {gpu_ms:.1f} ms  CPU {cpu_ms:.1f} ms', flush=True)
    json.dump(out, open(f'{BASE}/results/r42_coverage_cost.json', 'w'), indent=1)
    print(f'-> results/r42_coverage_cost.json ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
