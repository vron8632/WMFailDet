#!/usr/bin/env python3
"""r30 统一对比：同一协议、同一指标体系下的多方法检测与区域告警能力。

动机：
 - 仅有 IoU 类指标（训练式占优），本文指标体系（coverage / box recall / 面积比）从未与他人同口径比较；
 - 缺"非训练类水印基线"（固定种子 QIM / LSB / TrustMark）的同口径对比；
 - 需要把"检测能力"与"定位能力"分开报告——训练式方法是为消息恢复/掩膜监督训练的，
   其"图像级检测编辑"能力未被评测过。

方法（同一 COCO attackset、同一 bbox paste 编辑协议）：
  ours_coord   : WMFE（坐标 PRF 校验位 + top-10% 检测 + 连通域告警）
  ours_matrix  : WMFE 变体（种子化矩阵校验位，尺寸耦合 -> 不能同步；其余同）
  seed42       : 经典固定种子方案（无密钥）+ 同一检测/告警管线
  lsb          : 经典 LSB 脆弱水印 + 同一检测/告警管线（非训练外部基线）
  wam          : WAM 检测（bit-EC 作检测器）+ 1-mask 作区域告警
  editguard    : EditGuard recovery mask 作区域告警（+ 掩膜面积作检测得分）
  trustmark    : TrustMark present/secret 匹配作检测 + 滑窗作告警

指标（本文定义，全部方法同一口径）：
  检测:  AUC(clean vs edited)，TPR @ FPR=1%
  区域:  coverage(>=50%)、box recall 中位数、面积比(预测/真值)
  鲁棒:  JPEG85 后仍保持上述能力（真实传播链）

输出：results/r30_unified_compare_<ts>.json
用法：apjf-python r30_unified_compare.py --n 120 [--methods ours,lsb,wam,editguard,trustmark]
"""
import sys, os, json, glob, time, datetime, argparse, traceback
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
DELTA = 15
PAY, CHK = (1, 0), (1, 1)
MASTER = b'\x9a' + bytes(range(31))
K_LOC, T_LOC = 9, 0.38

import numpy as np
import cv2
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r29_missing_experiments import embed_lsb, fail_lsb, jpeg


# ---------------- 统一指标 ----------------
def auc(neg, pos):
    neg = np.asarray(neg, float).ravel(); pos = np.asarray(pos, float).ravel()
    neg = neg[~np.isnan(neg)]; pos = pos[~np.isnan(pos)]
    if len(neg) == 0 or len(pos) == 0:
        return float('nan'), float('nan')
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    a = (gt + 0.5 * eq) / (len(pos) * len(neg))
    thr = np.quantile(neg, 0.99)
    tpr = float((pos > thr).mean())
    return float(a), tpr


def stats_from_fail(fail, thr_top=0.41):
    s = ks.smooth_fail_map(fail, K_LOC)
    vals = np.sort(s.ravel())
    n = max(1, int(0.10 * vals.size))
    return float(vals[-n:].mean())


def region_metrics(fail_or_mask, bbox, shape, is_pixel_mask=False):
    """把失效图/掩膜 -> 最大连通域外接框，算 coverage/recall/面积比。"""
    H, W = shape
    x1, y1, x2, y2 = bbox
    if is_pixel_mask:
        m = (np.asarray(fail_or_mask) > 0).astype(np.uint8)
    else:
        pred = (ks.smooth_fail_map(np.asarray(fail_or_mask), K_LOC) > T_LOC).astype(np.uint8)
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        m = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
    gy1 = max(0, min(H - 1, int(round(float(y1) * H))))
    gy2 = max(gy1 + 1, min(H, int(round(float(y2) * H))))
    gx1 = max(0, min(W - 1, int(round(float(x1) * W))))
    gx2 = max(gx1 + 1, min(W, int(round(float(x2) * W))))
    gt = np.zeros((H, W), np.uint8)
    gt[gy1:gy2, gx1:gx2] = 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return dict(rec=0.0, cov50=0.0, cov70=0.0, area_ratio=0.0)
    c = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh, _ = st[c]
    box = np.zeros((H, W), np.uint8); box[by:by + bh, bx:bx + bw] = 1
    gtn = gt.sum()
    rec = float((box & gt).sum() / max(1, gtn))
    return dict(rec=rec, cov50=float(rec >= 0.5), cov70=float(rec >= 0.7),
                area_ratio=float(box.sum() / max(1, gtn)))


def summarize(det_scores, reg_rows, tag):
    return {
        'n': len(reg_rows),
        'detect_auc': round(det_scores['auc'], 4),
        'detect_tpr_at1fpr': round(det_scores['tpr'], 4),
        'coverage_ge50': round(float(np.mean([r['cov50'] for r in reg_rows])), 4),
        'coverage_ge70': round(float(np.mean([r['cov70'] for r in reg_rows])), 4),
        'box_recall_median': round(float(np.median([r['rec'] for r in reg_rows])), 4),
        'area_ratio_mean': round(float(np.mean([r['area_ratio'] for r in reg_rows])), 4),
        'tag': tag,
    }


# ---------------- 数据 ----------------
def load_items(n, root=None):
    """root=None -> COCO attackset；否则任意含 images/orig_*.png 的数据集目录。"""
    base = root or AS
    meta = json.load(open(f'{base}/meta.json'))
    out = []
    for s in meta['samples']:
        sid = s['id']
        fpo, fpa = f'{base}/images/orig_{sid}.png', f'{base}/images/att_{sid}.png'
        if not os.path.exists(fpo) or not os.path.exists(fpa):
            continue
        try:
            o = r8.crop8(np.array(Image.open(fpo).convert('RGB')))
            a = np.array(Image.open(fpa).convert('RGB'))
            if a.shape[:2] != o.shape[:2]:
                a = np.array(Image.fromarray(a).resize((o.shape[1], o.shape[0]), Image.LANCZOS))
            a = r8.crop8(a)
        except Exception:
            continue
        out.append((str(sid), o, a, s['bbox']))
        if len(out) >= n:
            break
    return out


# ---------------- 我们的四种变体 ----------------
def run_ours(items):
    res = {}
    for name, kind in [('ours_coord', 'coord'), ('ours_matrix', 'matrix')]:
        det_c, det_a, regs = [], [], []
        for sid, o, a, bb in items:
            key = ks.image_key(MASTER, sid)
            wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern=kind)
            att, _ = paste_edit(wm, a, bb)
            f_c = ks.fail_map(wm, key, DELTA, CHK, kind)
            f_a = ks.fail_map(att, key, DELTA, CHK, kind)
            det_c.append(stats_from_fail(f_c)); det_a.append(stats_from_fail(f_a))
            regs.append(region_metrics(f_a, bb, o.shape[:2]))
        au, tp = auc(det_c, det_a)
        res[name] = summarize({'auc': au, 'tpr': tp}, regs, f'WMFE {kind}')
        # JPEG85 鲁棒性
        jc, ja, jr = [], [], []
        for sid, o, a, bb in items:
            key = ks.image_key(MASTER, sid)
            wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern=kind)
            att, _ = paste_edit(wm, a, bb)
            wj, aj = jpeg(wm, 85), jpeg(att, 85)
            f_c = ks.fail_map(wj, key, DELTA, CHK, kind)
            f_a = ks.fail_map(aj, key, DELTA, CHK, kind)
            jc.append(stats_from_fail(f_c)); ja.append(stats_from_fail(f_a))
            jr.append(region_metrics(f_a, bb, o.shape[:2]))
        au2, tp2 = auc(jc, ja)
        res[name]['after_jpeg85'] = round(au2, 4)
        if kind == 'coord':
            # 缩放往返 -> WMFE 的行为：low10 拒答率
            abst, scores = 0, []
            for sid, o, a, bb in items[:40]:
                key = ks.image_key(MASTER, sid)
                wm0 = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
                H, W = o.shape[:2]
                small = np.array(Image.fromarray(wm0).resize((int(W*0.8), int(H*0.8)), Image.BILINEAR))
                back = np.array(Image.fromarray(small).resize((W, H), Image.BILINEAR))
                fb = ks.fail_map(back, key, DELTA, CHK, 'coord')
                low10 = float(np.sort(fb.ravel())[:max(1, int(0.10*fb.size))].mean())
                # low10 判据: 平滑后最坏 10% 的失败率 < 0.20 => 信号消失 => 拒答
                smth = ks.smooth_fail_map(fb, K_LOC)
                worst = float(np.sort(smth.ravel())[-max(1, int(0.10*smth.size)):].mean())
                if worst < 0.20:
                    abst += 1
                scores.append(worst)
            res[name]['geo_abstain_rate'] = round(abst / max(1, len(items[:40])), 4)
            res[name]['geo_worst_mean'] = round(float(np.mean(scores)), 4)
    # seed42（无密钥固定种子，同一管线）：master = b'42'*32 -> 所有图共享同一 image_key
    det_c, det_a, regs = [], [], []
    FIXED = b'42' * 32
    for sid, o, a, bb in items:
        H, W = o.shape[:2]; n_h, n_w = H // 8, W // 8
        wm = ks.embed_keyed(o, FIXED, DELTA, PAY, CHK, pattern='matrix')
        att, _ = paste_edit(wm, a, bb)
        f_c = ks.fail_map(wm, FIXED, DELTA, CHK, 'matrix'); f_a = ks.fail_map(att, FIXED, DELTA, CHK, 'matrix')
        det_c.append(stats_from_fail(f_c)); det_a.append(stats_from_fail(f_a))
        regs.append(region_metrics(f_a, bb, (H, W)))
    au, tp = auc(det_c, det_a)
    res['seed42'] = summarize({'auc': au, 'tpr': tp}, regs, '经典固定种子 QIM')
    # LSB 脆弱水印（同一检测/告警管线）
    det_c, det_a, regs = [], [], []
    jc = []; ja = []
    for sid, o, a, bb in items:
        key = ks.image_key(MASTER, sid)
        wm = embed_lsb(o, key); att, _ = paste_edit(wm, a, bb)
        f_c = fail_lsb(wm, key); f_a = fail_lsb(att, key)
        det_c.append(stats_from_fail(f_c)); det_a.append(stats_from_fail(f_a))
        regs.append(region_metrics(f_a, bb, o.shape[:2]))
        ja.append(stats_from_fail(fail_lsb(jpeg(att, 85), key)))
        jc.append(stats_from_fail(fail_lsb(jpeg(wm, 85), key)))
    au, tp = auc(det_c, det_a)
    res['lsb'] = summarize({'auc': au, 'tpr': tp}, regs, 'LSB 脆弱水印')
    au2, _ = auc(jc, ja)
    res['lsb']['after_jpeg85'] = round(au2, 4)
    return res


def luma(img):
    return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)


# ---------------- 训练式方法 ----------------
def run_wam(items):
    REPO = f'{BASE}/data/wam/watermark-anything-main'
    CKPT = f'{BASE}/data/wam/wam_mit.pth'
    sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/notebooks')
    cwd = os.getcwd(); os.chdir(REPO)
    import torch
    import inference_utils as iu
    from watermark_anything.data.metrics import msg_predict_inference
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    wam = iu.load_model_from_checkpoint(REPO + '/checkpoints/params.json', CKPT).to(dev).eval()
    det_c, det_a, regs = [], [], []
    det_c_mask, det_a_mask = [], []
    wm_cache = {}
    for sid, o, a, bb in items:
        x = iu.default_transform(Image.fromarray(o)).unsqueeze(0).to(dev)
        msg = wam.get_random_msg(1)
        with torch.no_grad():
            wm_t = wam.embed(x, msg)['imgs_w']
        wm_u8 = to_uint8(iu.unnormalize_img(wm_t))
        if wm_u8.shape[:2] != o.shape[:2]:
            wm_u8 = cv2.resize(wm_u8, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_CUBIC)
        att_u8, _ = paste_edit(wm_u8, a, bb)

        def one(img):
            with torch.no_grad():
                preds = wam.detect(iu.default_transform(Image.fromarray(img)).unsqueeze(0).to(dev))['preds']
            mask = torch.sigmoid(preds[:, 0]); bits = preds[:, 1:]
            acc = float((msg_predict_inference(bits, mask).detach().cpu() == msg.detach().cpu()).float().mean())
            return acc, mask[0].cpu().numpy()
        acc_c, m_c = one(wm_u8); acc_a, m_a = one(att_u8)
        det_c.append(1 - acc_c); det_a.append(1 - acc_a)   # bit-EC 作检测得分
        det_c_mask.append(float((1.0 - m_c).mean())); det_a_mask.append(float((1.0 - m_a).mean()))
        wm_cache[sid] = wm_u8
        H, W = o.shape[:2]
        sc = cv2.resize(1.0 - m_a, (W, H), interpolation=cv2.INTER_LINEAR)
        regs.append(region_metrics(sc > 0.5, bb, (H, W), is_pixel_mask=True))
    au, tp = auc(det_c, det_a)
    out = summarize({'auc': au, 'tpr': tp}, regs, 'WAM（消息恢复作检测器）')
    out['bit_acc_clean'] = round(float(np.mean([1 - v for v in det_c])), 4)
    out['bit_acc_edited'] = round(float(np.mean([1 - v for v in det_a])), 4)
    out['detect_auc_msg'] = round(au, 4)
    # mask-area 检测得分（WAM 自带的水印存在掩膜）
    au2, tp2 = auc(det_c_mask, det_a_mask)
    out['detect_auc_maskarea'] = round(au2, 4)
    out['detect_tpr_at1fpr_maskarea'] = round(tp2, 4)
    # ---- 几何退化行为：缩放 0.8->1.0（WMFE 对此拒答），看 WAM 检测得分是否虚高 ----
    geo_clean, geo_att = [], []
    for sid, o, a, bb in items[:40]:
        H, W = o.shape[:2]
        small = np.array(Image.fromarray(wm_cache[sid]).resize((int(W*0.8), int(H*0.8)), Image.BILINEAR))
        back = np.array(Image.fromarray(small).resize((W, H), Image.BILINEAR))
        with torch.no_grad():
            preds = wam.detect(iu.default_transform(Image.fromarray(back)).unsqueeze(0).to(dev))['preds']
            m = torch.sigmoid(preds[:, 0])[0].cpu().numpy()
        geo_clean.append(float((1 - m).mean()))
    out['geo_scale_fpr1'] = round(float(np.mean(np.asarray(geo_clean) > 0.10)), 4)
    out['geo_scale_mean_score'] = round(float(np.mean(geo_clean)), 4)
    out['geo_abstain_rate'] = 0.0
    out['note_geo'] = 'WAM 无拒答通道：缩放退化图的掩膜得分如上（对照 WMFE 触发 ABSTAIN）'
    os.chdir(cwd)
    return {'wam': out}


def to_uint8(t):
    x = t.detach().float().clamp(0, 1)
    if x.dim() == 4:
        x = x[0]
    return (x.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def run_editguard(items):
    EG = f'{BASE}/data/editguard/EditGuard-main'; EGC = f'{EG}/code'
    sys.path.insert(0, EGC)
    cwd = os.getcwd()
    os.chdir(EGC)
    try:
        _orig = __import__('torch').load
        import torch
        torch.load = lambda *aa, **kk: _orig(*aa, **{**kk, 'weights_only': False})
        import options.options as option
        from models import create_model
        from test_gradio import load_image
        opt = option.dict_to_nonedict(option.parse('options/test_editguard.yml', is_train=False))
        for k in ['sdinpaint', 'controlnetinpaint', 'sdxl', 'repaint', 'degrade_shuffle']:
            opt[k] = False
        eg = create_model(opt); eg.load_network(f'{EG}/checkpoints/clean.pth', eg.netG)
        eg.netG.to('cuda').eval()
        msg = np.random.RandomState(7).choice([-0.5, 0.5], (1, 64)).astype(np.float32)
        msg_t = torch.from_numpy(msg).cuda()
        det_a, det_c, regs = [], [], []
        wm_cache = {}
        for sid, o, a, bb in items:
            H, W = o.shape[:2]; x1, y1, x2, y2 = bb
            with torch.no_grad():
                d = load_image(o, msg); eg.real_H = d['GT'].cuda(); eg.ref_L = d['LQ'].cuda(); eg.mes = msg_t
                wm = np.array(eg.image_hiding())
                S = 512
                e1, f1, e2, f2 = int(x1 * S), int(y1 * S), int(x2 * S), int(y2 * S)
                at512 = np.array(Image.fromarray(a).resize((S, S), Image.BICUBIC))
                edit = wm.copy(); edit[f1:f2, e1:e2] = at512[f1:f2, e1:e2]
                # clean 水印图自身也跑一次恢复 -> 诚实的干净侧检测得分（掩膜面积）
                d1 = load_image(wm, msg); eg.real_H = d1['GT'].cuda(); eg.host = d1['GT'].cuda()
                eg.ref_L = d1['LQ'].cuda(); eg.mes = msg_t
                mk_c, _ = eg.image_recovery(0.2)
                d2 = load_image(edit, msg); eg.real_H = d2['GT'].cuda(); eg.host = d2['GT'].cuda()
                eg.ref_L = d2['LQ'].cuda(); eg.mes = msg_t
                mk512, _ = eg.image_recovery(0.2)
            wm_cache[sid] = wm
            pm = cv2.resize((np.array(mk512) > 0).astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST)
            pc = cv2.resize((np.array(mk_c) > 0).astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST)
            regs.append(region_metrics(pm, bb, (H, W), is_pixel_mask=True))
            det_a.append(float(pm.mean()))
            det_c.append(float(pc.mean()))
        au, tp = auc(det_c, det_a)
        out = summarize({'auc': au, 'tpr': tp}, regs, 'EditGuard（recovery mask）')
        out['detect_auc_maskarea'] = round(au, 4)
        # ---- 几何退化：对 clean 水印图做缩放往返（WMFE 拒答），看 EditGuard 是否虚报篡改 ----
        geo = []
        for sid, o, a, bb in items[:30]:
            H, W = o.shape[:2]
            small = np.array(Image.fromarray(wm_cache[sid]).resize((int(W*0.8), int(H*0.8)), Image.BILINEAR))
            back = np.array(Image.fromarray(small).resize((W, H), Image.BILINEAR))
            with torch.no_grad():
                d3 = load_image(back, msg); eg.real_H = d3['GT'].cuda(); eg.host = d3['GT'].cuda()
                eg.ref_L = d3['LQ'].cuda(); eg.mes = msg_t
                mk3, _ = eg.image_recovery(0.2)
            geo.append(float((np.array(mk3) > 0).mean()))
        g = np.array(geo)
        out['geo_scale_fpr1'] = round(float((g > 0.01).mean()), 4)
        out['geo_scale_mean_mask'] = round(float(g.mean()), 4)
        out['note_geo'] = 'EditGuard 无拒答通道：缩放退化的干净图仍输出篡改掩膜（误报率如上）'
        return {'editguard': out}
    finally:
        os.chdir(cwd)


def run_trustmark(items, n_win=60):
    """TrustMark（Q 型, 同 r11）：secret 匹配作检测 + 128px 滑窗 present 作区域告警。"""
    sys.path.insert(0, '/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
                       '/code/baselines/trustmark/python')
    from trustmark import TrustMark
    tm = TrustMark(verbose=False, model_type='Q', device='cpu')
    WINS, STR = 128, 64
    det_c, det_a, regs = [], [], []
    n_ok = 0
    for idx, (sid, o, a, bb) in enumerate(items):
        H, W = o.shape[:2]
        secret = f'wm{sid}'
        try:
            wm = np.array(tm.encode(Image.fromarray(o), secret).convert('RGB'))
        except Exception:
            continue
        att, _ = paste_edit(wm, a, bb)
        s_c, p_c, _ = tm.decode(Image.fromarray(wm))
        s_a, p_a, _ = tm.decode(Image.fromarray(att))
        # 检测得分：secret 匹配 = 0, 不匹配 = 1
        det_c.append(0.0 if s_c == secret else 1.0)
        det_a.append(0.0 if s_a == secret else 1.0)
        if p_c:
            n_ok += 1
        score = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
        for yy in range(0, max(1, H - WINS + 1), STR):
            for xx in range(0, max(1, W - WINS + 1), STR):
                _, pp, _ = tm.subimage_decode(Image.fromarray(att[yy:yy+WINS, xx:xx+WINS]), secret)
                score[yy:yy+WINS, xx:xx+WINS] += float(not pp)
                cnt[yy:yy+WINS, xx:xx+WINS] += 1
        with np.errstate(invalid='ignore', divide='ignore'):
            score = np.where(cnt > 0, score / np.maximum(cnt, 1.0), 0.0)
        regs.append(region_metrics(score > 0.5, bb, (H, W), is_pixel_mask=True))
        if idx + 1 >= n_win:
            break
    au, tp = auc(det_c, det_a)
    out = summarize({'auc': au, 'tpr': tp}, regs, 'TrustMark（secret 匹配 + 滑窗）')
    out['clean_present_rate'] = round(n_ok / max(1, len(det_c)), 4)
    return {'trustmark': out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--methods', default='ours,wam,editguard')
    ap.add_argument('--dataset', default='coco',
                    help='coco | butterfly | sat | doc | pokemon | lama | all')
    ap.add_argument('--out-suffix', default='')
    a = ap.parse_args()
    t0 = time.time()
    sets = (['coco', 'butterfly', 'sat', 'doc', 'pokemon'] if a.dataset == 'all' else [a.dataset])
    for ds in sets:
        root = None if ds == 'coco' else f'{BASE}/data/xdomain/{ds}_edits'
        items = load_items(a.n, root)
        a._ds = ds
        run_all(items, a, ds)


def run_all(items, a, ds):
    t0 = time.time()
    print(f'样本: {len(items)} 张')
    out = {'meta': {'n': len(items), 'dataset': ds,
                    'protocol': 'bbox paste 编辑, 同一检测/告警指标体系 (r30)',
                    'ts': datetime.datetime.now().isoformat()},
           'delta': DELTA, 'k_loc': K_LOC, 't_loc': T_LOC}
    for m in a.methods.split(','):
        t = time.time()
        try:
            if m == 'ours':
                out.update(run_ours(items))
            elif m == 'wam':
                out.update(run_wam(items))
            elif m == 'editguard':
                out.update(run_editguard(items))
            elif m == 'trustmark':
                out.update(run_trustmark(items))
            print(f'[{m}] {time.time()-t:.1f}s')
            print(json.dumps({k: v for k, v in out.items() if k not in ('meta', 'delta', 'k_loc', 't_loc')},
                             ensure_ascii=False, indent=1, default=str))
        except Exception as e:
            print(f'[{m}] FAILED: {e}')
            traceback.print_exc()
            out[m + '_error'] = str(e)[:300]
    out['runtime_sec'] = round(time.time() - t0, 1)
    os.makedirs(f'{BASE}/results', exist_ok=True)
    fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/r30_unified_compare_{ds}{a.out_suffix}_{fn}.json'
    json.dump(out, open(fp, 'w'), indent=1, ensure_ascii=False, default=str)
    print('结果:', fp)


if __name__ == '__main__':
    main()
