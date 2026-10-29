import os, sys, json
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import numpy as np, rasterio
from rasterio.windows import Window
import run_pre1023_p2pformer as R
cfg=os.environ.get("P2P_CFG", R.DEFAULT_CONFIG); ckpt=os.environ.get("P2P_CKPT","/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth")
m=R.build_model(cfg, ckpt, os.environ.get("P2P_DEV","cuda:1")); print("ready",flush=True)

def read_tile(ds,x0,y0,w,h):
    a=ds.read(window=Window(x0,y0,w,h))
    img=np.transpose(a,(1,2,0))
    if img.shape[2]==1: img=np.repeat(img,3,axis=2)
    return np.ascontiguousarray(img[:,:,:3])

ref=json.load(open("/data1/wzh26/code/detect-speedup/output/pre1023_gzda_seg/json/pre.json"))["detections"]
rc=np.array([[(d["bbox_xyxy"][0]+d["bbox_xyxy"][2])/2,(d["bbox_xyxy"][1]+d["bbox_xyxy"][3])/2,d["score"],
              d["bbox_xyxy"][2]-d["bbox_xyxy"][0], d["bbox_xyxy"][3]-d["bbox_xyxy"][1]] for d in ref])
rc=rc[rc[:,2]>=0.5]
TILES=[(7680,17920),(17280,14080),(3200,18560),(18560,11520),(5120,16000)]
with rasterio.open("/tmp/p2p_work/pre.tif") as ds:
    for crop in (256,320,384,512,640):
        imgs=[];bbs=[]
        for (x0,y0) in TILES:
            cx=max(0,min(x0+320-crop//2,ds.width-crop)); cy=max(0,min(y0+320-crop//2,ds.height-crop))
            imgs.append(read_tile(ds,cx,cy,crop,crop)); bbs.append((cx,cy,cx+crop,cy+crop))
        res=R.predict_batch(m,imgs,bbs,min_area_px=40)
        for thr in (0.05,0.1,0.2,0.3):
            n=0;hit=0;nref=0;areas=[]
            for (bx0,by0,bx1,by1),polys in zip(bbs,res):
                sel=[p for p in polys if p["score"]>=thr]
                n+=len(sel); areas+= [p["area_px"] for p in sel]
                boxes=[]
                for p in sel:
                    r=np.array(p["rings_px"][0]); boxes.append((r[:,0].min(),r[:,1].min(),r[:,0].max(),r[:,1].max()))
                boxes=np.array(boxes) if boxes else np.zeros((0,4))
                mr=rc[(rc[:,0]>=bx0)&(rc[:,0]<bx1)&(rc[:,1]>=by0)&(rc[:,1]<by1)]
                nref+=len(mr)
                for r in mr:
                    if len(boxes) and np.any((r[0]>=boxes[:,0])&(r[0]<=boxes[:,2])&(r[1]>=boxes[:,1])&(r[1]<=boxes[:,3])): hit+=1
            print(f"crop={crop} thr={thr}: dets/tile={n/len(bbs):.1f} ref/tile={nref/len(bbs):.1f} "
                  f"recall={hit/max(nref,1):.3f} area_med_px={np.median(areas) if areas else 0:.0f}",flush=True)
