#!/usr/bin/env python3
"""r37: 定性对比网格数据 — 4 样本(按编辑面积分层) x {宿主, 编辑GT, 失效图, 教师告警, 学生, WAM}.

输出: results/r37_qualitative.npz
  host/edited/gt/fail_heat/alarm4/student_pred/wam_mask (各 N=4, 256x256)
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
from r34_distill_matrix import SmallUNet, DELTA, PAY, CHK, MASTER

AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')


def wam_masks_for(items):
    """items: [(sid,o,a,bb)] -> 每样本 WAM 掩膜 (1-m_a>0.5)."""
    REPO = f'{BASE}/data/wam/watermark-anything-main'
    CKPT = f'{BASE}/data/wam/wam_mit.pth'
    sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/notebooks')
    cwd = os.getcwd(); os.chdir(REPO)
    import inference_utils as iu
    from watermark_anything.data.metrics import msg_predict_inference  # noqa
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    wam = iu.load_model_from_checkpoint(REPO + '/checkpoints/params.json', CKPT).to(dev).eval()
    out = []
    from r30_unified_compare import to_uint8
    for sid, o, a, bb in items:
        x = iu.default_transform(Image.fromarray(o)).unsqueeze(0).to(dev)
        msg = wam.get_random_msg(1)
        with torch.no_grad():
            wm_t = wam.embed(x, msg)['imgs_w']
        wm_u8 = to_uint8(iu.unnormalize_img(wm_t))
        if wm_u8.shape[:2] != o.shape[:2]:
            wm_u8 = cv2.resize(wm_u8, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_CUBIC)
        att_u8, _ = paste_edit(wm_u8, a, bb)
        with torch.no_grad():
            preds = wam.detect(iu.default_transform(Image.fromarray(att_u8)).unsqueeze(0).to(dev))['preds']
        mask = torch.sigmoid(preds[:, 0])[0, 0].cpu().numpy()
        H, W = o.shape[:2]
        sc = cv2.resize(1.0 - mask, (W, H), interpolation=cv2.INTER_LINEAR)
        out.append((sc > 0.5).astype(np.uint8))
    os.chdir(cwd)
    return out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(f'{BASE}/results/r34_data_n1200.npz')
    img, gt, a4 = z['img'], z['gt'], z['alarm4']
    # 测试段索引 (与 r34/r36 一致)
    meta = json.load(open(f'{AS}/meta.json'))
    ids, ok = [], 0
    for s_ in meta['samples']:
        sid = s_['id']
        if os.path.exists(f'{AS}/images/orig_{sid}.png') and os.path.exists(f'{AS}/images/att_{sid}.png'):
            ids.append(sid); ok += 1
        if ok >= 1200: break
    test_ids = ids[1000:1200]
    test_idx = list(range(1000, 1200))
    # 按 GT 面积分层选 4 个: small/mid/large/full
    areas = [(i, gt[i].sum()) for i in test_idx]
    areas.sort(key=lambda x: x[1])
    n = len(areas)
    picks = [areas[int(n * p)][0] for p in (0.15, 0.40, 0.65, 0.92)]  # small/mid/large/full
    print('picked test indices:', picks, 'areas:', [int(gt[i].sum()) for i in picks], flush=True)

    # 学生预测 (r34 已存, 顺序=测试段)
    stu_all = np.load(f'{BASE}/results/r34_pred_alarm4.npy')  # (200,256,256)
    stu_map = {test_idx[k]: stu_all[k] for k in range(len(test_idx))}

    # 失效图 + 宿主 + WAM
    hosts, edited, fails = [], [], []
    wam_items = []
    for i in picks:
        sid = test_ids[i - 1000]
        o = r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
        key = ks.image_key(MASTER, sid)
        f8 = ks.fail_map(img[i], key, DELTA, CHK, 'coord')   # 8×8 失败图 (32x32)
        hosts.append(o); edited.append(img[i]); fails.append(f8)
        # WAM 需要 att 图 (用于 paste)
        a = r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        # bbox 从 gt 反推 (矩形)
        ys, xs = np.where(gt[i] > 0)
        bb = (xs.min() / 256, ys.min() / 256, (xs.max() + 1) / 256, (ys.max() + 1) / 256)
        wam_items.append((sid, o, a, bb))
    print('running WAM on 4 items...', flush=True)
    t0 = time.time()
    wam_ms = wam_masks_for(wam_items)
    print(f'WAM done ({time.time()-t0:.0f}s)', flush=True)

    np.savez_compressed(
        f'{BASE}/results/r37_qualitative.npz',
        host=np.array(hosts), edited=np.array(edited), gt=gt[picks],
        fail8=np.array(fails), alarm4=a4[picks],
        student=np.array([stu_map[i] for i in picks]), wam=np.array(wam_ms),
        picks=np.array(picks))
    print('-> results/r37_qualitative.npz', flush=True)


if __name__ == '__main__':
    main()
