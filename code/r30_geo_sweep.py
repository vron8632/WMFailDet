#!/usr/bin/env python3
"""r30_geo_sweep: 分布外几何退化条件扫描（检验单一退化条件下的结论是否稳健）。

对干净水印图施加退化(缩放往返 0.7/0.8/0.9/1.1/1.2、旋转 1°/2°)后:
  WMFE:      low10 判据触发 ABSTAIN 的比例 (期望高) + 正常图的误拒率 (期望 0)
  EditGuard: 恢复掩膜面积比 (>>1 即误报全图; 期望接近 1)
  WAM:       存在掩膜得分 (期望 ~0, 不误报; 但它没有拒识输出)
"""
import sys, os, json, time, datetime
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r30_unified_compare import load_items, AS, MASTER, DELTA, PAY, CHK

CONDS = {'scale_0.7': 0.7, 'scale_0.8': 0.8, 'scale_0.9': 0.9,
         'scale_1.1': 1.1, 'scale_1.2': 1.2}

def roundtrip(img, f):
    H, W = img.shape[:2]
    s = np.array(Image.fromarray(img).resize((max(8, int(W*f)), max(8, int(H*f))), Image.BILINEAR))
    return np.array(Image.fromarray(s).resize((W, H), Image.BILINEAR))

def low_quantile(fail):
    return float(np.percentile(ks.smooth_fail_map(fail, 5), 10))

def main(n=60):
    t0 = time.time()
    items = load_items(n)
    out = {'meta': {'n': len(items), 'conditions': list(CONDS) + ['clean', 'rotate_1', 'rotate_2'],
                    'ts': datetime.datetime.now().isoformat()}}
    # ---- WMFE
    abst = {c: [] for c in list(CONDS) + ['clean', 'rotate_1', 'rotate_2']}
    for sid, o, a, bb in items:
        key = ks.image_key(MASTER, sid)
        wm = ks.embed_keyed(o, key, DELTA, PAY, CHK, pattern='coord')
        for name, img in ([('clean', wm)] + [(k, roundtrip(wm, f)) for k, f in CONDS.items()]
                          + [('rotate_1', r8.t_rotate(wm, 1.0)[0]), ('rotate_2', r8.t_rotate(wm, 2.0)[0])]):
            img8 = r8.crop8(img)
            f = ks.fail_map(img8, key, DELTA, CHK, 'coord')
            abst[name].append(1.0 if low_quantile(f) >= 0.20 else 0.0)   # 1 = ABSTAIN (low10>=0.20)
    out['wafe_abstain_rate'] = {k: round(float(np.mean(v)), 4) for k, v in abst.items()}
    print('WMFE abstain:', out['wafe_abstain_rate'])

    # ---- EditGuard & WAM
    eg = None
    EGC = f'{BASE}/data/editguard/EditGuard-main/code'
    sys.path.insert(0, EGC)
    cwd = os.getcwd(); os.chdir(EGC)
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
        eg = create_model(opt)
        eg.load_network(f'{BASE}/data/editguard/EditGuard-main/checkpoints/clean.pth', eg.netG)
        eg.netG.to('cuda').eval()
        msg = np.random.RandomState(7).choice([-0.5, 0.5], (1, 64)).astype(np.float32)
        msg_t = torch.from_numpy(msg).cuda()
        area = {c: [] for c in abst}
        for sid, o, a, bb in items:
            with torch.no_grad():
                d = load_image(o, msg); eg.real_H = d['GT'].cuda(); eg.ref_L = d['LQ'].cuda(); eg.mes = msg_t
                wm = np.array(eg.image_hiding())
            for name, img in ([('clean', wm)] + [(k, roundtrip(wm, f)) for k, f in CONDS.items()]
                              + [('rotate_1', r8.t_rotate(wm, 1.0)[0]), ('rotate_2', r8.t_rotate(wm, 2.0)[0])]):
                with torch.no_grad():
                    d2 = load_image(img, msg); eg.real_H = d2['GT'].cuda(); eg.host = d2['GT'].cuda()
                    eg.ref_L = d2['LQ'].cuda(); eg.mes = msg_t
                    mk, _ = eg.image_recovery(0.2)
                area[name].append(float((np.array(mk) > 0).mean()))
        out['editguard_mask_area'] = {k: round(float(np.mean(v)), 4) for k, v in area.items()}
        print('EditGuard mask area (1.0=全图误报):', out['editguard_mask_area'])
    finally:
        os.chdir(cwd)

    # ---- WAM
    REPO = f'{BASE}/data/wam/watermark-anything-main'
    sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/notebooks')
    os.chdir(REPO)
    try:
        import torch
        import inference_utils as iu
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        wam = iu.load_model_from_checkpoint(REPO + '/checkpoints/params.json',
                                            f'{BASE}/data/wam/wam_mit.pth').to(dev).eval()
        sc = {c: [] for c in abst}
        for sid, o, a, bb in items[:40]:
            x = iu.default_transform(Image.fromarray(o)).unsqueeze(0).to(dev)
            m32 = wam.get_random_msg(1)
            with torch.no_grad():
                wm_u8 = (iu.unnormalize_img(wam.embed(x, m32)['imgs_w'])[0]
                         .permute(1, 2, 0).clamp(0, 1).mul(255).round().cpu().numpy().astype(np.uint8))
            for name, img in ([('clean', wm_u8)] + [(k, roundtrip(wm_u8, f)) for k, f in CONDS.items()]
                              + [('rotate_1', r8.t_rotate(wm_u8, 1.0)[0]), ('rotate_2', r8.t_rotate(wm_u8, 2.0)[0])]):
                with torch.no_grad():
                    p = wam.detect(iu.default_transform(Image.fromarray(img)).unsqueeze(0).to(dev))['preds']
                    mm = torch.sigmoid(p[:, 0])[0].cpu().numpy()
                sc[name].append(float((1.0 - mm).mean()))
        out['wam_absence_score'] = {k: round(float(np.mean(v)), 4) for k, v in sc.items()}
        print('WAM absence score (高=报篡改):', out['wam_absence_score'])
    finally:
        os.chdir(cwd)

    out['runtime_sec'] = round(time.time() - t0, 1)
    fp = f"{BASE}/results/r30_geo_sweep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(out, open(fp, 'w'), indent=1, ensure_ascii=False)
    print('结果:', fp)

if __name__ == '__main__':
    main()
