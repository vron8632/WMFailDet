#!/usr/bin/env python3
"""r40: 系数恢复攻击 vs 学生 — 学生不用水印/失效图, 攻击应失效(安全优势).

攻击(r32 同构, 但针对 4×4 层): 对编辑区每 4×4 块把 (1,1) 系数恢复为未编辑水印图原值.
对照: 教师(fail4->alarm4) / 学生(direct inference) 在 原始编辑图 vs 攻击图 上.
输出: results/r40_erasure_attack.json
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

AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
N = 200


def restore_coef(img_edit, wm, bbox, bs=4, coef=(1, 1)):
    """编辑区内每 bs×bs 块把 coef 系数恢复为未编辑水印图 wm 的原值 (Y 通道)."""
    H, W = img_edit.shape[0] // bs * bs, img_edit.shape[1] // bs * bs
    ye = cv2.cvtColor(img_edit, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)[:H, :W].copy()
    yw = cv2.cvtColor(wm, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)[:H, :W]
    x1, y1, x2, y2 = bbox
    b1, b2 = int(y1*H)//bs*bs, int(np.ceil(y2*H/bs)*bs)
    a1, a2 = int(x1*W)//bs*bs, int(np.ceil(x2*W/bs)*bs)
    for i in range(b1, min(b2, H), bs):
        for j in range(a1, min(a2, W), bs):
            Ce = cv2.dct(ye[i:i+bs, j:j+bs]); Cw = cv2.dct(yw[i:i+bs, j:j+bs])
            Ce[coef] = Cw[coef]
            ye[i:i+bs, j:j+bs] = cv2.idct(Ce)
    ycrcb = cv2.cvtColor(img_edit, cv2.COLOR_RGB2YCrCb)
    out = ycrcb.copy(); out[:H, :W, 0] = np.clip(ye, 0, 255)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB).astype(np.uint8)


def infer(m, imgs, device, bs=16):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            xb = torch.from_numpy(imgs[i:i+bs].transpose(0, 3, 1, 2)).float().to(device) / 255.
            out.append(torch.sigmoid(m(xb)).cpu().numpy())
    return (np.concatenate(out)[:, 0] > 0.5).astype(np.uint8)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SmallUNet(base=32).to(device)
    m.load_state_dict(torch.load(f'{BASE}/results/r43_best_student.pth', map_location=device))
    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    edited_all, gt_all = z['img'][1000:1200], z['gt'][1000:1200]
    meta = json.load(open(f'{AS}/meta.json'))
    ids, ok = [], 0
    for s_ in meta['samples']:
        sid = s_['id']
        if os.path.exists(f'{AS}/images/orig_{sid}.png') and os.path.exists(f'{AS}/images/att_{sid}.png'):
            ids.append(s_); ok += 1
        if ok >= 1200: break
    test = ids[1000:1200]

    att_imgs, a_alarm, a_pred, o_alarm, o_pred = [], [], [], [], []
    t0 = time.time()
    for k, s_ in enumerate(test):
        sid = s_['id']
        o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
        key = ks.image_key(MASTER, sid); k4 = layer_key(key, 'bs4')
        wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        wm = embed_bs(wm, k4, 4, DELTA, (1, 1))
        # 攻击: 从 编辑图 出发, 恢复 4×4 层 (1,1)
        att = restore_coef(edited_all[k], wm, s_['bbox'], bs=4, coef=(1, 1))
        att_imgs.append(att)
        f4_att = fail_bs(att, k4, 4, DELTA, (1, 1))
        f4_org = fail_bs(edited_all[k], k4, 4, DELTA, (1, 1))
        a_alarm.append(alarm_from_fail(f4_att, 4, 5, 32))
        o_alarm.append(alarm_from_fail(f4_org, 4, 5, 32))
    att_imgs = np.array(att_imgs)
    a_pred = infer(m, att_imgs, device)
    o_pred = infer(m, edited_all, device)

    def st(p):
        s = eval_mask(p, gt_all); s.pop('per_iou'); return s
    res = {
        'meta': dict(n=N, attack='restore 4x4-layer DCT(1,1) coef in edit region',
                     student='r34 SmallUNet alarm4'),
        'teacher_original': st(np.array(o_alarm)),
        'teacher_attacked': st(np.array(a_alarm)),
        'student_original': st(o_pred),
        'student_attacked': st(a_pred),
        'attack_visual_cost': float(np.mean([
            10*np.log10(255**2/max(1e-6, np.mean((a.astype(float)-e.astype(float))**2)))
            for a, e in zip(att_imgs, edited_all)]))}
    for k, v in res.items():
        if k in ('meta', 'attack_visual_cost'): continue
        print(f'{k:20s} IoU={v["iou"]:.3f} cov={v["cov50"]:.1%}')
    print(f'attack visual cost (vs edited): PSNR {res["attack_visual_cost"]:.2f} dB')
    json.dump(res, open(f'{BASE}/results/r40_erasure_attack.json', 'w'), indent=1)
    print(f'-> results/r40_erasure_attack.json ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
