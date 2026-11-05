#!/usr/bin/env python3
"""r27 WAM 在真实高清 (640x480, 140 张) 上的定位 — 补全"分辨率 × 方法"矩阵."""
import sys, os, json, datetime, time
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO=f'{BASE}/data/wam/watermark-anything-main'; CKPT=f'{BASE}/data/wam/wam_mit.pth'
D=f'{BASE}/data/hires_expanded'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, REPO+'/notebooks')
import numpy as np, cv2, torch
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r7_keyed_scheme import paste_edit
from r14b_multiscale import layer_key, embed_bs, fail_bs, loc_iou
from r12_method_comparison_wam import to_uint8
DELTA=15; PAY,CHK=(1,0),(1,1); MASTER=b'\x9a'+bytes(range(31))
os.chdir(REPO)
import inference_utils as iu
from watermark_anything.data.metrics import msg_predict_inference
from sklearn.metrics import roc_auc_score
dev='cuda'
t0=time.time()
wam=iu.load_model_from_checkpoint(REPO+'/checkpoints/params.json',CKPT).to(dev).eval()
print('WAM loaded', time.time())
meta=json.load(open(f'{D}/meta.json')); samples=meta['samples']
rows=[]
for idx,s in enumerate(samples):
    sid=s['id']
    o=np.array(Image.open(f'{D}/images/orig_{sid}.png').convert('RGB'))
    a=np.array(Image.open(f'{D}/images/att_{sid}.png').convert('RGB'))
    key=ks.image_key(MASTER,sid); H,W=o.shape[:2]
    x1,y1,x2,y2=s['bbox']; gt=np.zeros((H,W),bool)
    gt[int(y1*H):int(y2*H),int(x1*W):int(x2*W)]=True
    with torch.no_grad():
        x=iu.default_transform(Image.fromarray(o)).unsqueeze(0).to(dev)
        msg=wam.get_random_msg(1)
        wm_t=wam.embed(x,msg)['imgs_w']; wm_u8=to_uint8(iu.unnormalize_img(wm_t))
        if wm_u8.shape[:2]!=o.shape[:2]:
            wm_u8=cv2.resize(wm_u8,(W,H),interpolation=cv2.INTER_CUBIC)
        att_u8,_=paste_edit(wm_u8,a,s['bbox'])
        def detect(img):
            preds=wam.detect(iu.default_transform(Image.fromarray(img)).unsqueeze(0).to(dev))['preds']
            mask=torch.sigmoid(preds[:,0,:,:])
            pm=msg_predict_inference(preds[:,1:,:,:],mask)
            return float((pm.detach().cpu()==msg.detach().cpu()).float().mean()), mask[0].cpu().numpy()
        acc_a, mask_a = detect(att_u8)
    m_a=cv2.resize(mask_a,(W,H),interpolation=cv2.INTER_LINEAR)
    sc=1.0-m_a
    iou=sc
    pred=sc>0.5
    inter=(pred&gt).sum(); union=(pred|gt).sum()
    rows.append({'iou':float(inter/max(1,union)),'recall':float(inter/max(1,float(gt.sum()))),
                 'in':float(pred[gt].mean()) if gt.any() else float('nan'),
                 'out':float(pred[~gt].mean()) if (~gt).any() else float('nan')})
def sm(rows):
    i=np.array([r['iou'] for r in rows]); r=np.array([r['recall'] for r in rows])
    return {'n':len(rows),'iou':float(i.mean()),'iou_median':float(np.median(i)),
            'recall_median':float(np.median(r)),'cov_ge50':float((r>=0.5).mean()),
            'in':float(np.mean([x['in'] for x in rows])),
            'out':float(np.mean([x['out'] for x in rows]))}
res={'meta':{'ts':datetime.datetime.now().isoformat(),'n':len(rows),
             'data':'hires_expanded 640x480','wam_input':'256 (native) → mask resize 回 640'},
     'wam':sm(rows),'runtime_sec':round(time.time()-t0,1)}
os.makedirs(f'{BASE}/results',exist_ok=True)
fn=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
fp=f'{BASE}/results/r27_wam_hires_{fn}.json'; json.dump(res,open(fp,'w'),indent=1)
r=res['wam']
print(f"WAM 高清 640x480 n={r['n']}: IoU {r['iou']:.3f} (中位 {r['iou_median']:.3f}, "
      f"覆盖≥50% {r['cov_ge50']*100:.0f}%) in/out {r['in']*100:.1f}/{r['out']*100:.1f}%")
print(f'结果: {fp}  ({res["runtime_sec"]}s)')
