#!/usr/bin/env python3
"""G6-b 方法对比: Watermark Anything (WAM, Meta, ICLR'25) vs 我们的"死亡信号".

WAM = 当前最强的"局部化水印"方法 (训练式), 原生输出:
  embed  : 32-bit 消息 -> imgs_w
  detect : preds [1, 33, 256, 256] = [mask(1), bits(32)],  mask = 水印存在概率 (逐像素)

与我们的方法**直接可比**的两件事:
  1 检测: WAM 用 bit accuracy (clean 高 / 编辑后下降); 我们用 top-10% 失败率 / contrast;
  2 定位: WAM 的 mask (水印存在区域) -> 编辑区 mask 应塌陷; 我们的死亡图 -> 编辑区失败率高.
  两者都是"逐像素/逐块"的图, 因此可以同口径算 AUC 与 IoU.

协议与 r6/r7/r10/r11 完全一致:
  o -> wm -> paste_edit(wm, SD inpainting att, bbox) -> 检测/定位
  (256x256, 与 WAM 的 img_size=256 原生对齐)

输出: results/r12_method_comparison_wam_<ts>.json
用法: apjf-python r12_method_comparison_wam.py [--n 100]
"""
import sys, os, json, argparse, datetime, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = f'{BASE}/data/wam/watermark-anything-main'
CKPT = f'{BASE}/data/wam/wam_mit.pth'
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + '/notebooks')

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit, block_gt

DELTA = 15
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))
TOP_THR, CON_THR = 0.41, 0.15
K_LOC, T_LOC = 9, 0.38


def to_uint8(t):
    x = t.detach().float().clamp(0, 1)
    if x.dim() == 4:
        x = x[0]
    return (x.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def ours_stats(fail):
    s = ks.smooth_fail_map(fail, K_LOC)
    vals = np.sort(s.ravel())
    n = max(1, int(0.10 * vals.size))
    top = float(vals[-n:].mean())
    return top, top - float(np.median(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    args = ap.parse_args()
    t0 = time.time()
    os.chdir(REPO)

    import inference_utils as iu
    from watermark_anything.data.metrics import msg_predict_inference
    from sklearn.metrics import roc_auc_score

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    wam = iu.load_model_from_checkpoint(REPO + '/checkpoints/params.json', CKPT).to(dev).eval()
    print(f'WAM 已加载 ({time.time()-t0:.1f}s, device={dev})')

    meta = json.load(open(f'{AS}/meta.json'))
    samples = (meta['samples'] if isinstance(meta, dict) else meta)[:args.n]
    rows = []
    for idx, s in enumerate(samples):
        sid = s['id']
        try:
            o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        key = ks.image_key(MASTER, sid)
        x = iu.default_transform(Image.fromarray(o)).unsqueeze(0).to(dev)
        msg = wam.get_random_msg(1)
        with torch.no_grad():
            wm_t = wam.embed(x, msg)['imgs_w']
        wm_u8 = to_uint8(iu.unnormalize_img(wm_t))
        if wm_u8.shape[:2] != o.shape[:2]:
            wm_u8 = cv2.resize(wm_u8, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_CUBIC)
        att_u8, box = paste_edit(wm_u8, a, s['bbox'])

        def detect(img_u8):
            with torch.no_grad():
                preds = wam.detect(iu.default_transform(Image.fromarray(img_u8)).unsqueeze(0).to(dev))['preds']
            mask = torch.sigmoid(preds[:, 0, :, :])
            bits = preds[:, 1:, :, :]
            pm = msg_predict_inference(bits, mask)
            acc = float((pm.detach().cpu() == msg.detach().cpu()).float().mean())
            return acc, mask[0].cpu().numpy()

        acc_c, mask_c = detect(wm_u8)
        acc_a, mask_a = detect(att_u8)

        # ---- 我们的方法 (同子集)
        wm_k = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        att_k, _ = paste_edit(wm_k, a, s['bbox'])
        f_att = ks.fail_map(att_k, key, DELTA, CHK, 'coord')
        top, con = ours_stats(f_att)

        # 定位 GT (像素)
        H, W = o.shape[:2]
        x1, y1, x2, y2 = s['bbox']
        gt_px = np.zeros((H, W), bool)
        gt_px[int(round(y1*H)):int(round(y2*H)), int(round(x1*W)):int(round(x2*W))] = True

        # WAM 定位: 1-mask = "无水印"分数
        m_a = cv2.resize(mask_a, (W, H), interpolation=cv2.INTER_LINEAR)
        m_c = cv2.resize(mask_c, (W, H), interpolation=cv2.INTER_LINEAR)
        wam_score = 1.0 - m_a
        auc_wam = float(roc_auc_score(gt_px.ravel().astype(int), wam_score.ravel())) \
            if 0 < gt_px.sum() < gt_px.size else float('nan')
        pred_wam = wam_score > 0.5
        inter, union = (pred_wam & gt_px).sum(), (pred_wam | gt_px).sum()
        iou_wam = float(inter / union) if union else 0.0
        # clean 上的定位假阳性面积
        fpr_area_wam = float(((1.0 - m_c) > 0.5).mean())

        # 我们的定位
        f_sm = ks.smooth_fail_map(f_att, K_LOC)
        pred = f_sm > T_LOC
        p = pred.astype(np.uint8)
        p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
        keep = np.zeros_like(p)
        for c in range(1, n):
            if st[c, cv2.CC_STAT_AREA] >= 8:
                keep[lab == c] = 1
        pred_px = np.repeat(np.repeat(keep.astype(bool), 8, 0), 8, 1)[:H, :W]
        inter, union = (pred_px & gt_px).sum(), (pred_px | gt_px).sum()
        iou_ours = float(inter / union) if union else 0.0
        auc_ours = float(roc_auc_score(gt_px.ravel().astype(int),
                                       np.repeat(np.repeat(f_sm, 8, 0), 8, 1)[:H, :W].ravel())) \
            if 0 < gt_px.sum() < gt_px.size else float('nan')

        rows.append({
            'id': sid,
            'wam': {'acc_clean': acc_c, 'acc_att': acc_a,
                    'mask_clean': float(mask_c.mean()), 'mask_att': float(mask_a.mean()),
                    'loc_auc': auc_wam, 'iou': iou_wam, 'fpr_area_clean': fpr_area_wam},
            'ours': {'top': top, 'contrast': con, 'loc_auc': auc_ours, 'iou': iou_ours},
        })
        if (idx + 1) % 25 == 0:
            print(f'  [{idx+1}/{len(samples)}] {time.time()-t0:.0f}s', flush=True)

    def m(key, sub=None):
        v = [r[key] for r in rows]
        if sub:
            v = [x[sub] for x in v]
        v = [x for x in v if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v))

    acc_c = [r['wam']['acc_clean'] for r in rows]
    acc_a = [r['wam']['acc_att'] for r in rows]
    mk_c = [r['wam']['mask_clean'] for r in rows]
    mk_a = [r['wam']['mask_att'] for r in rows]
    labels = np.r_[np.zeros(len(mk_c)), np.ones(len(mk_a))]
    # 公平性: WAM 也可用"mask 面积塌陷"作图像级检测量 (编辑区无水印 -> mask 变低)
    auc_mask = float(roc_auc_score(labels, -np.r_[mk_c, mk_a]))
    auc_bits = float(roc_auc_score(labels, -np.r_[acc_c, acc_a]))
    res = {
        'meta': {'ts': datetime.datetime.now().isoformat(), 'n': len(rows),
                 'wam_ckpt': CKPT, 'protocol': 'attackset SD inpaint + bbox paste (256x256)',
                 'wam_msg': '32-bit random per image', 'ours_workpoint': [TOP_THR, CON_THR]},
        'wam': {
            'bit_acc_clean': float(np.mean(acc_c)),
            'bit_acc_att': float(np.mean(acc_a)),
            'detect_auc_bitec': auc_bits,
            'detect_auc_maskarea': auc_mask,
            'detect_tpr_at_acc': float(np.mean(np.array(acc_a) < np.percentile(acc_c, 5))),
            'loc_auc': m('wam', 'loc_auc'), 'iou': m('wam', 'iou'),
            'fpr_area_clean': m('wam', 'fpr_area_clean'),
            'mask_clean': m('wam', 'mask_clean'), 'mask_att': m('wam', 'mask_att'),
        },
        'ours': {
            'top_mean': m('ours', 'top'), 'contrast_mean': m('ours', 'contrast'),
            'detect_rate_top': float(np.mean([r['ours']['top'] > TOP_THR for r in rows])),
            'detect_rate_contrast': float(np.mean([r['ours']['contrast'] > CON_THR for r in rows])),
            'loc_auc': m('ours', 'loc_auc'), 'iou': m('ours', 'iou'),
        },
    }
    print(f'\n=== WAM (n={len(rows)}) ===')
    print(f'  bit acc: clean {res["wam"]["bit_acc_clean"]:.3f} -> att {res["wam"]["bit_acc_att"]:.3f}  '
          f'检测 AUC(bit_acc) {res["wam"]["detect_auc_bitec"]:.4f}  '
          f'AUC(mask 面积) {res["wam"]["detect_auc_maskarea"]:.4f}')
    print(f'  定位: AUC {res["wam"]["loc_auc"]:.3f}  IoU@0.5 {res["wam"]["iou"]:.3f}  '
          f'clean 假阳性面积 {res["wam"]["fpr_area_clean"]*100:.1f}%')
    print(f'  我们: top 检出 {res["ours"]["detect_rate_top"]*100:.1f}% / contrast '
          f'{res["ours"]["detect_rate_contrast"]*100:.1f}%  定位 AUC {res["ours"]["loc_auc"]:.3f}  '
          f'IoU {res["ours"]["iou"]:.3f}')
    res['runtime_sec'] = round(time.time() - t0, 1)
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r12_method_comparison_wam_{fn}.json'
    json.dump(res, open(fp, 'w'), indent=1)
    print(f'\n结果: {fp}  ({res["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
