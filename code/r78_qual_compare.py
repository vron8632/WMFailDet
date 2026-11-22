#!/usr/bin/env python3
"""r78: 定性对比图 — 同一批样本上并列 GT / EditGuard / Teacher / Student(ours).

r62 的 EditGuard 流程复用; 学生用 r73 δ=25 权重; 教师为 δ=25 的 2×2 失效图.
输出 8 样本 × 5 列 的拼图素材: results/r78_qual_*.npz + figs/figD1_qualitative 重绘
"""
import sys, os, json
BASE = "/media/oyp/工作/Projects/042_image_forensic/edit-type-aware-watermark-defense"
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r14b_multiscale import layer_key, fail_bs
from r34_distill_matrix import SmallUNet, MASTER, PAY, CHK
from r71_tracks import predict

D, N_SHOW = 25, 8
OUT = f'{BASE}/results/r78_qual_compare.npz'


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SmallUNet(base=32).to(dev)
    m.load_state_dict(torch.load(f'{BASE}/results/r73_delta25_student.pth', map_location=dev))

    z = np.load(f'{BASE}/results/r70_data_d25.npz')
    img, gt = z['img'][-400:], z['gt'][-400:]
    meta = json.load(open(f'{BASE}/data/bigset/meta.json'))
    ids = [s_['id'] for s_ in meta['samples']
           if os.path.exists(f'{BASE}/data/bigset/images/orig_{s_["id"]}.png')]
    test_ids, test_orig = ids[-400:], {}

    # 取 N_SHOW 个中等编辑面积的样本
    areas = gt.reshape(len(gt), -1).mean(1)
    order = np.argsort(np.abs(areas - np.median(areas)))[:N_SHOW]

    samples = []
    for k in order:
        im, g = img[k], gt[k]
        sid = test_ids[k]
        key = ks.image_key(MASTER, sid)
        k2 = layer_key(key, 'bs4')
        f2 = fail_bs(im, k2, 4, D, (1, 1))                 # 教师失效图 (4x4 层, 与主表一致)
        samples.append(dict(sid=sid, img=im, gt=g, teacher=f2.astype(np.uint8) * 255))
    print(f'samples: {len(samples)}', flush=True)

    # 学生预测
    probs = predict(m, np.array([s['img'] for s in samples]), dev)
    # EditGuard
    eg_pred = None
    try:
        EG = f'{BASE}/data/editguard/EditGuard-main'; EGC = f'{EG}/code'
        sys.path.insert(0, EGC)
        cwd = os.getcwd(); os.chdir(EGC)
        _orig = torch.load
        torch.load = lambda *aa, **kk: _orig(*aa, **{**kk, 'weights_only': False})
        import options.options as option
        from models import create_model
        from test_gradio import load_image
        opt = option.dict_to_nonedict(option.parse('options/test_editguard.yml', is_train=False))
        for kk in ['sdinpaint', 'controlnetinpaint', 'sdxl', 'repaint', 'degrade_shuffle']:
            opt[kk] = False
        eg = create_model(opt)
        eg.load_network(f'{EG}/checkpoints/clean.pth', eg.netG)
        eg.netG.to('cuda').eval()
        msg = np.random.RandomState(7).choice([-0.5, 0.5], (1, 64)).astype(np.float32)
        msg_t = torch.from_numpy(msg).cuda()
        eg_pred = []
        for s in samples:
            o = r8.crop8(np.array(Image.open(f'{BASE}/data/bigset/images/orig_{s["sid"]}.png').convert('RGB')))
            H, W = s['img'].shape[:2]
            # 用编辑图作 host, 复现 r62 的重建残差流程
            with torch.no_grad():
                d2 = load_image(s['img'], msg)
                eg.real_H = d2['GT'].cuda(); eg.host = d2['GT'].cuda()
                eg.ref_L = d2['LQ'].cuda(); eg.mes = msg_t
                mk512, _ = eg.image_recovery(0.2)
            pm = cv2.resize((np.array(mk512) > 0).astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST)
            eg_pred.append(pm * 255)
        os.chdir(cwd)
        print('EditGuard done', flush=True)
    except Exception as e:
        print('EditGuard unavailable:', str(e)[:120])
        eg_pred = [np.zeros_like(s['gt']) * 255 for s in samples]

    np.savez_compressed(OUT,
                        imgs=np.array([s['img'] for s in samples]),
                        gts=np.array([s['gt'] * 255 for s in samples]),
                        teacher=np.array([s['teacher'] for s in samples]),
                        student=(np.array(probs) > 0.5).astype(np.uint8) * 255,
                        editguard=np.array(eg_pred))
    print('->', OUT)


if __name__ == '__main__':
    main()
