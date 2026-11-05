#!/usr/bin/env python3
"""r25 LaMa 编辑器: 用**非扩散** inpainting 生成局部编辑, 检验"死亡"是否 SD 特有.

数据: attackset_20261001 前 200 张原图 (256 -> 512)
编辑: 随机 bbox (0.08-0.30) + LaMa (big-lama) 填充
输出: data/xdomain/lama_edits/{images,meta.json}
"""
import sys, os, json, argparse, datetime, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS=('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness/results/attackset_20261001_222447')
OUT=f'{BASE}/data/xdomain/lama_edits'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=200)
    ap.add_argument('--seed',type=int,default=0); a=ap.parse_args()
    os.makedirs(f'{OUT}/images',exist_ok=True)
    meta=json.load(open(f'{AS}/meta.json')); samples=meta['samples'][:a.n]
    from simple_lama_inpainting import SimpleLama
    lama=SimpleLama(); print('LaMa 已加载')
    rng=random.Random(a.seed); out=[]
    for idx,s in enumerate(samples):
        sid=s['id']
        o=np.array(Image.open(f'{AS}/images/orig_{sid}.png').convert('RGB'))
        o512=Image.fromarray(o).resize((512,512), Image.BICUBIC)
        area=rng.uniform(0.08,0.30); ar=rng.uniform(0.7,1.4)
        h=int(512*np.sqrt(area/ar)); w=int(h*ar); h=min(h,400); w=min(w,400)
        x1=rng.randint(16,512-w-16); y1=rng.randint(16,512-h-16)
        mask=np.zeros((512,512),np.uint8); mask[y1:y1+h,x1:x1+w]=255
        filled=lama(o512, Image.fromarray(mask))
        edit=np.array(o512).copy(); edit[y1:y1+h,x1:x1+w]=np.array(filled)[y1:y1+h,x1:x1+w]
        o512.resize((256,256), Image.BICUBIC).save(f'{OUT}/images/orig_{sid}.png')
        Image.fromarray(edit).resize((256,256), Image.BICUBIC).save(f'{OUT}/images/att_{sid}.png')
        out.append({'id':sid,'bbox':[x1/512,y1/512,(x1+w)/512,(y1+h)/512],'area':(w*h)/(512*512)})
        if (idx+1)%50==0: print(f'  [{idx+1}/{len(samples)}]',flush=True)
    json.dump({'samples':out,'domain':'lama','source':'attackset_20261001_222447',
               'generator':'big-lama (simple-lama-inpainting)','ts':datetime.datetime.now().isoformat()},
              open(f'{OUT}/meta.json','w'),indent=1)
    print(f'完成 {len(out)} 张 -> {OUT}')
if __name__=='__main__': main()
