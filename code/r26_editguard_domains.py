#!/usr/bin/env python3
"""r26 EditGuard 多域: 在蝴蝶/卫星/文档/卡通上跑 EditGuard 定位 (与 r24 同协议)."""
import sys, os, json, argparse, datetime, time
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EG=f'{BASE}/data/editguard/EditGuard-main'; EG_CODE=f'{EG}/code'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EG_CODE); os.chdir(EG_CODE)
import numpy as np, cv2, torch
from PIL import Image
_orig=torch.load
torch.load=lambda *a,**k:_orig(*a,**{**k,'weights_only':False})
import options.options as option
from models import create_model
from test_gradio import load_image
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--domains',default='butterfly,sat,doc,pokemon'); a=ap.parse_args()
    t0=time.time()
    opt=option.dict_to_nonedict(option.parse('options/test_editguard.yml',is_train=False))
    for k in ['sdinpaint','controlnetinpaint','sdxl','repaint','degrade_shuffle']: opt[k]=False
    eg=create_model(opt); eg.load_network(f'{EG}/checkpoints/clean.pth', eg.netG); eg.netG.to('cuda').eval()
    print(f'EditGuard 已加载 ({time.time()-t0:.1f}s)')
    msg=np.random.RandomState(42).choice([-0.5,0.5],(1,64)).astype(np.float32)
    msg_t=torch.from_numpy(msg).cuda()
    out={'ts':datetime.datetime.now().isoformat(),'domains':{}}
    for domain in a.domains.split(','):
        D=f'{BASE}/data/xdomain/{domain}_edits'
        meta=json.load(open(f'{D}/meta.json'))
        rows=[]
        for s in meta['samples'][:150]:
            sid=s['id']
            o=np.array(Image.open(f'{D}/images/orig_{sid}.png').convert('RGB'))
            at=np.array(Image.open(f'{D}/images/att_{sid}.png').convert('RGB'))
            H,W=o.shape[:2]; x1,y1,x2,y2=s['bbox']
            bx1,by1,bx2,by2=int(x1*W),int(y1*H),int(x2*W),int(y2*H)
            with torch.no_grad():
                d=load_image(o,msg); eg.real_H=d['GT'].cuda(); eg.ref_L=d['LQ'].cuda(); eg.mes=msg_t
                wm=np.array(eg.image_hiding())
                S=512; ex1,ey1,ex2,ey2=int(x1*S),int(y1*S),int(x2*S),int(y2*S)
                at512=np.array(Image.fromarray(at).resize((S,S),Image.BICUBIC))
                edit=wm.copy(); edit[ey1:ey2,ex1:ex2]=at512[ey1:ey2,ex1:ex2]
                d2=load_image(edit,msg); eg.real_H=d2['GT'].cuda(); eg.host=d2['GT'].cuda()
                eg.ref_L=d2['LQ'].cuda(); eg.mes=msg_t
                mask512,_=eg.image_recovery(0.2)
            pred=cv2.resize((np.array(mask512)>0).astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
            gt=np.zeros((H,W),bool); gt[by1:by2,bx1:bx2]=True
            inter=(pred&gt).sum(); union=(pred|gt).sum()
            rows.append({'iou':float(inter/max(1,union)),
                         'in':float(pred[gt].mean()) if gt.any() else float('nan'),
                         'out':float(pred[~gt].mean()) if (~gt).any() else float('nan')})
        i=np.array([r['iou'] for r in rows])
        sm={'n':len(rows),'iou':float(i.mean()),'iou_median':float(np.median(i)),
            'P(iou>=0.5)':float((i>=0.5).mean()),
            'in':float(np.nanmean([r['in'] for r in rows])),
            'out':float(np.nanmean([r['out'] for r in rows]))}
        out['domains'][domain]=sm
        print(f"  {domain:10s} n={sm['n']:3d} IoU {sm['iou']:.3f} (中位 {sm['iou_median']:.3f}, "
              f"=0.5 占比 {sm['P(iou>=0.5)']*100:.0f}%) in/out {sm['in']*100:.1f}/{sm['out']*100:.1f}%",flush=True)
    out['runtime_sec']=round(time.time()-t0,1)
    os.makedirs(f'{BASE}/results',exist_ok=True)
    fn=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp=f'{BASE}/results/r26_editguard_domains_{fn}.json'; json.dump(out,open(fp,'w'),indent=1)
    print(f'结果: {fp}  ({out["runtime_sec"]}s)')
if __name__=='__main__': main()
