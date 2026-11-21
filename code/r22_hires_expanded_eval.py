#!/usr/bin/env python3
"""r22 真实高清 140 张上的定位/告警评估 (替代 r20 的 30 张薄弱证据)."""
import sys, os, json, glob, argparse, datetime, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r14b_multiscale import layer_key, embed_bs, fail_bs
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f'{BASE}/data/hires_expanded'
DELTA=15; PAY,CHK=(1,0),(1,1); MASTER=b'\x9a'+bytes(range(31))

def post(pred, min_area):
    p = pred.astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
    keep = np.zeros_like(p)
    for c in range(1,n):
        if st[c,cv2.CC_STAT_AREA] >= min_area: keep[lab==c]=1
    return keep.astype(bool)

def metrics(fail, bs, gt, k=5):
    H,W = gt.shape
    f = ks.smooth_fail_map(fail.astype(np.float32), k) if bs==8 else cv2.blur(fail.astype(np.float32),(k,k))
    pred = post(f>0.38, 8 if bs==8 else 32)
    pred_px = np.repeat(np.repeat(pred, bs, 0), bs, 1)[:H,:W]
    inter = float((pred_px&gt).sum()); union=float((pred_px|gt).sum())
    return {'iou': inter/max(1,union), 'recall': inter/max(1,float(gt.sum())),
            'pred_area': float(pred_px.mean()), 'gt_area': float(gt.mean())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=200); a=ap.parse_args()
    t0=time.time(); meta=json.load(open(f'{D}/meta.json'))
    rows8, rows4 = [], []
    for s in meta['samples'][:a.n]:
        sid=s['id']
        try:
            o=r8.crop8(np.array(Image.open(f'{D}/images/orig_{sid}.png').convert('RGB')))
            at=r8.crop8(np.array(Image.open(f'{D}/images/att_{sid}.png').convert('RGB')))
            m=r8.crop8(np.array(Image.open(f'{D}/images/mask_{sid}.png').convert('L')))
        except FileNotFoundError: continue
        gt=m>127; key=ks.image_key(MASTER,sid)
        k8,k4=layer_key(key,'bs8'),layer_key(key,'bs4')
        wm=ks.embed_keyed(o,k8,DELTA,PAY,CHK,pattern='coord')
        wm2=embed_bs(wm,k4,4,DELTA,(1,1))
        att=wm2.copy(); att[gt]=at[gt]
        rows8.append(metrics(fail_bs(att,k8,8,DELTA,CHK),8,gt))
        rows4.append(metrics(fail_bs(att,k4,4,DELTA,(1,1)),4,gt))
    def sm(rows):
        r=np.array([x['recall'] for x in rows])
        return {'n':len(rows),'iou_mean':float(np.mean([x['iou'] for x in rows])),
                'recall_mean':float(r.mean()),'recall_median':float(np.median(r)),
                'cov_ge50':float((r>=0.5).mean()),'cov_ge70':float((r>=0.7).mean()),
                'pred_area':float(np.mean([x['pred_area'] for x in rows])),
                'gt_area':float(np.mean([x['gt_area'] for x in rows]))}
    res={'meta':{'ts':datetime.datetime.now().isoformat(),'n':len(rows8),
                 'data':'hires_expanded (640x480, IP2P, 矩形 bbox)'},
         '8x8':sm(rows8),'4x4':sm(rows4),'runtime_sec':round(time.time()-t0,1)}
    os.makedirs(f'{BASE}/results',exist_ok=True)
    fn=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp=f'{BASE}/results/r22_hires_expanded_{fn}.json'; json.dump(res,open(fp,'w'),indent=1)
    print(f"高清 640x480 n={len(rows8)}: 8x8 IoU {res['8x8']['iou_mean']:.3f} "
          f"cov>=50% {res['8x8']['cov_ge50']*100:.0f}% | 4x4 IoU {res['4x4']['iou_mean']:.3f} "
          f"cov>=50% {res['4x4']['cov_ge50']*100:.0f}% | 面积 {res['4x4']['pred_area']:.3f}/{res['4x4']['gt_area']:.3f}")
    print(f'结果: {fp}  ({res["runtime_sec"]}s)')
if __name__=='__main__': main()
