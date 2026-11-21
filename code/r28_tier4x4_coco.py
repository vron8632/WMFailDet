#!/usr/bin/env python3
"""r28 COCO-1200 上 4×4 层的分层告警指标（补表 2 缺口）."""
import sys, os, json, datetime, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS=('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/results/attackset_20261001_222447')
DELTA=15; PAY,CHK=(1,0),(1,1); MASTER=b'\x9a'+bytes(range(31))
def post(pred, min_area):
    p=pred.astype(np.uint8)
    p=cv2.morphologyEx(p,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    p=cv2.morphologyEx(p,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(p,connectivity=8)
    keep=np.zeros_like(p)
    for c in range(1,n):
        if st[c,cv2.CC_STAT_AREA]>=min_area: keep[lab==c]=1
    return keep.astype(bool)
def main():
    t0=time.time(); meta=json.load(open(f'{AS}/meta.json'))
    samples=meta['samples']; rows=[]
    for idx,s in enumerate(samples):
        sid=s['id']
        try:
            o=r8.crop8(np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB')))
            a=r8.crop8(np.array(Image.open(f'{AS}/images/att_{sid}.png').convert('RGB')))
        except FileNotFoundError: continue
        key=ks.image_key(MASTER,sid); k8,k4=layer_key(key,'bs8'),layer_key(key,'bs4')
        wm=ks.embed_keyed(o,k8,DELTA,PAY,CHK,pattern='coord')
        wm2=embed_bs(wm,k4,4,DELTA,(1,1))
        att,_=paste_edit(wm2,a,s['bbox'])
        f4=fail_bs(att,k4,4,DELTA,(1,1))
        pred=post(cv2.blur(f4,(5,5))>0.38,32)
        pred_px=np.repeat(np.repeat(pred,4,0),4,1)[:o.shape[0],:o.shape[1]]
        x1,y1,x2,y2=s['bbox']; H,W=o.shape[:2]
        gt=np.zeros((H,W),bool); gt[int(y1*H):int(y2*H),int(x1*W):int(x2*W)]=True
        inter=float((pred_px&gt).sum()); union=float((pred_px|gt).sum())
        rows.append({'id':sid,'area':(x2-x1)*(y2-y1),
                     'iou':inter/max(1,union),'recall':inter/max(1,float(gt.sum()))})
        if (idx+1)%300==0: print(f'  [{idx+1}/{len(samples)}] {time.time()-t0:.0f}s',flush=True)
    def sm(rr):
        r=np.array([x['recall'] for x in rr]); i=np.array([x['iou'] for x in rr])
        return {'n':len(rr),'recall_mean':float(r.mean()),'recall_median':float(np.median(r)),
                'cov_ge50':float((r>=0.5).mean()),'cov_ge70':float((r>=0.7).mean()),'iou_mean':float(i.mean())}
    res={'meta':{'ts':datetime.datetime.now().isoformat(),'n':len(rows),
                 'note':'4x4 层（8x8+4x4 双层嵌入），阈值 0.38+post(32)，矩形 bbox GT'},
         'tiers':{},'overall':sm(rows)}
    for tag,lo,hi in [('small',0,0.10),('mid',0.10,0.30),('large',0.30,0.90),('full',0.90,1.01)]:
        rr=[x for x in rows if lo<=x['area']<hi]
        if rr: res['tiers'][tag]=sm(rr)
    print('overall',res['overall'])
    for k,v in res['tiers'].items(): print(k,v)
    os.makedirs(f'{BASE}/results',exist_ok=True)
    fn=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp=f'{BASE}/results/r28_tier4x4_coco_{fn}.json'; json.dump(res,open(fp,'w'),indent=1)
    print(f'结果: {fp}  ({time.time()-t0:.0f}s)')
if __name__=='__main__': main()
