#!/usr/bin/env python3
"""r24 多域对比: WAM (SA-1B 训练) vs 我们的 QIM (training-free) 在各域上的检测与定位.

域: sat (EuroSAT 卫星), doc (FUNSD 表单), art (WikiArt), butterfly (Smithsonian).
每个域用 r23 生成的 SD inpainting 局部编辑 (bbox paste, 256x256), 与 attackset 同协议.

输出: results/r24_multidomain_<domain>_<ts>.json
用法: apjf-python r24_multidomain_compare.py --domain sat
"""
import sys, os, json, argparse, datetime, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = f'{BASE}/data/wam/watermark-anything-main'
CKPT = f'{BASE}/data/wam/wam_mit.pth'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + '/notebooks')

import numpy as np
import cv2
from PIL import Image
import torch

import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs, loc_iou
from r12_method_comparison_wam import to_uint8

DELTA = 15
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True)
    ap.add_argument('--n', type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    D = f'{BASE}/data/xdomain/{args.domain}_edits'

    os.chdir(REPO)
    import inference_utils as iu
    from watermark_anything.data.metrics import msg_predict_inference
    from sklearn.metrics import roc_auc_score

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    wam = iu.load_model_from_checkpoint(REPO + '/checkpoints/params.json', CKPT).to(dev).eval()
    print(f'WAM 已加载 ({time.time()-t0:.1f}s), domain={args.domain}')

    meta = json.load(open(f'{D}/meta.json'))
    samples = meta['samples'][:args.n]
    rows = []
    for idx, s in enumerate(samples):
        sid = s['id']
        o = np.array(Image.open(f'{D}/images/orig_{sid}.png').convert('RGB'))
        a = np.array(Image.open(f'{D}/images/att_{sid}.png').convert('RGB'))
        key = ks.image_key(MASTER, f'{args.domain}_{sid}')
        H, W = o.shape[:2]
        x1, y1, x2, y2 = s['bbox']
        bx1, by1, bx2, by2 = int(round(x1*W)), int(round(y1*H)), int(round(x2*W)), int(round(y2*H))
        gt = np.zeros((H, W), bool)
        gt[by1:by2, bx1:bx2] = True

        # WAM
        with torch.no_grad():
            x = iu.default_transform(Image.fromarray(o)).unsqueeze(0).to(dev)
            msg = wam.get_random_msg(1)
            wm_t = wam.embed(x, msg)['imgs_w']
            wm_u8 = to_uint8(iu.unnormalize_img(wm_t))
            if wm_u8.shape[:2] != o.shape[:2]:
                wm_u8 = cv2.resize(wm_u8, (W, H), interpolation=cv2.INTER_CUBIC)
            att_u8, _ = paste_edit(wm_u8, a, s['bbox'])

            def detect(img_u8):
                preds = wam.detect(iu.default_transform(Image.fromarray(img_u8)).unsqueeze(0).to(dev))['preds']
                mask = torch.sigmoid(preds[:, 0, :, :])
                pm = msg_predict_inference(preds[:, 1:, :, :], mask)
                acc = float((pm.detach().cpu() == msg.detach().cpu()).float().mean())
                return acc, mask[0].cpu().numpy()

            acc_c, mask_c = detect(wm_u8)
            acc_a, mask_a = detect(att_u8)
        m_a = cv2.resize(mask_a, (W, H), interpolation=cv2.INTER_LINEAR)
        m_c = cv2.resize(mask_c, (W, H), interpolation=cv2.INTER_LINEAR)
        sc = 1.0 - m_a
        auc_wam = float(roc_auc_score(gt.ravel().astype(int), sc.ravel())) if 0 < gt.sum() < gt.size else float('nan')
        pred_w = sc > 0.5
        inter, union = (pred_w & gt).sum(), (pred_w | gt).sum()
        iou_wam = float(inter / union) if union else 0.0

        # 我们
        k8, k4 = layer_key(key, 'bs8'), layer_key(key, 'bs4')
        wm = ks.embed_keyed(o, k8, DELTA, PAY, CHK, pattern='coord')
        wm2 = embed_bs(wm, k4, 4, DELTA, (1, 1))
        att, _ = paste_edit(wm2, a, s['bbox'])
        f8c = fail_bs(wm2, k8, 8, DELTA, CHK)
        f8a = fail_bs(att, k8, 8, DELTA, CHK)
        f4a = fail_bs(att, k4, 4, DELTA, (1, 1))
        f8_sm = ks.smooth_fail_map(f8a, 9)
        vals = np.sort(f8_sm.ravel()); nt = max(1, int(0.10 * vals.size))
        top = float(vals[-nt:].mean()); con = top - float(np.median(vals))
        c_sm = ks.smooth_fail_map(f8c, 9)
        cv_ = np.sort(c_sm.ravel()); ntc = max(1, int(0.10 * cv_.size))
        top_c = float(cv_[-ntc:].mean()); con_c = top_c - float(np.median(cv_))

        rows.append({
            'id': sid,
            'wam': {'acc_clean': acc_c, 'acc_att': acc_a,
                    'mask_clean': float(m_c.mean()), 'mask_att': float(m_a.mean()),
                    'loc_auc': auc_wam, 'iou': iou_wam},
            'ours': {'clean_fail': float(f8c.mean()), 'top': top, 'contrast': con,
                     'top_clean': top_c, 'con_clean': con_c,
                     'iou8': loc_iou(f8a, s['bbox'], 8, 5, 0.38, 8),
                     'iou4': loc_iou(f4a, s['bbox'], 4, 5, 0.38, 32)},
        })
        if (idx + 1) % 40 == 0:
            print(f'  [{idx+1}/{len(samples)}] {time.time()-t0:.0f}s', flush=True)

    def mean(path):
        v = [r[path[0]][path[1]] for r in rows]
        v = [x for x in v if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v)) if v else float('nan')

    acc_c = [r['wam']['acc_clean'] for r in rows]
    acc_a = [r['wam']['acc_att'] for r in rows]
    mk_c = [r['wam']['mask_clean'] for r in rows]
    mk_a = [r['wam']['mask_att'] for r in rows]
    labels = np.r_[np.zeros(len(mk_c)), np.ones(len(mk_a))]
    res = {
        'meta': {'ts': datetime.datetime.now().isoformat(), 'n': len(rows), 'domain': args.domain,
                 'source': meta.get('source'), 'editor': meta.get('generator')},
        'wam': {
            'bit_acc_clean': float(np.mean(acc_c)), 'bit_acc_att': float(np.mean(acc_a)),
            'mask_clean': float(np.mean(mk_c)), 'mask_att': float(np.mean(mk_a)),
            'detect_auc_maskarea': float(roc_auc_score(labels, -np.r_[mk_c, mk_a])),
            'loc_auc': mean(('wam', 'loc_auc')), 'iou': mean(('wam', 'iou')),
        },
        'ours': {
            'clean_fail': mean(('ours', 'clean_fail')),
            'top_clean': mean(('ours', 'top_clean')), 'top': mean(('ours', 'top')),
            'contrast_clean': mean(('ours', 'con_clean')), 'contrast': mean(('ours', 'contrast')),
            'iou8': mean(('ours', 'iou8')), 'iou4': mean(('ours', 'iou4')),
        },
        'runtime_sec': round(time.time() - t0, 1),
    }
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r24_multidomain_{args.domain}_{fn}.json'
    json.dump(res, open(fp, 'w'), indent=1)
    print(f'\n=== {args.domain} (n={len(rows)}) ===')
    print(f'  WAM : bit_acc {res["wam"]["bit_acc_clean"]:.3f}->{res["wam"]["bit_acc_att"]:.3f}  '
          f'mask {res["wam"]["mask_clean"]:.3f}/{res["wam"]["mask_att"]:.3f}  '
          f'detect AUC {res["wam"]["detect_auc_maskarea"]:.4f}  loc AUC {res["wam"]["loc_auc"]:.3f}  '
          f'IoU {res["wam"]["iou"]:.3f}')
    print(f'  我们: clean fail {res["ours"]["clean_fail"]*100:.1f}%  '
          f'top {res["ours"]["top_clean"]*100:.1f}->{res["ours"]["top"]*100:.1f}%  '
          f'contrast {res["ours"]["contrast_clean"]*100:.1f}->{res["ours"]["contrast"]*100:.1f}%  '
          f'IoU {res["ours"]["iou8"]:.3f}/{res["ours"]["iou4"]:.3f}')
    print(f'结果: {fp}  ({res["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
