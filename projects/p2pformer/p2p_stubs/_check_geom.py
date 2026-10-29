import os, sys, json
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import numpy as np, rasterio
from rasterio.windows import Window
import run_pre1023_p2pformer as R
cfg=os.environ.get("P2P_CFG", R.DEFAULT_CONFIG); ckpt=os.environ.get("P2P_CKPT","/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth")
m=R.build_model(cfg, ckpt, os.environ.get("P2P_DEV","cuda:3")); print("ready",flush=True)
def read_tile(ds,x0,y0,w,h):
    a=ds.read(window=Window(x0,y0,w,h)); img=np.transpose(a,(1,2,0))
    if img.shape[2]==1: img=np.repeat(img,3,axis=2)
    return np.ascontiguousarray(img[:,:,:3])
ref=json.load(open("/data1/wzh26/code/detect-speedup/output/pre1023_gzda_seg/json/pre.json"))["detections"]
B=np.array([d["bbox_xyxy"]+[d["score"]] for d in ref if d["score"]>=0.5])
S=384
with rasterio.open("/tmp/p2p_work/pre.tif") as ds:
    for (x0,y0) in [(7680,17920),(17280,14080),(3200,18560)]:
        cx,cy=x0+320-S//2, y0+320-S//2
        img=read_tile(ds,cx,cy,S,S)
        polys=R.predict_batch(m,[img],[(cx,cy,cx+S,cy+S)],min_area_px=40)[0]
        A=[]
        for p in polys:
            r=np.array(p["rings_px"][0])
            A.append([r[:,0].min(),r[:,1].min(),r[:,0].max(),r[:,1].max(),p["score"]])
        A=np.array(A) if A else np.zeros((0,5))
        sel=B[(B[:,0]>=cx-100)&(B[:,0]<=cx+S+100)&(B[:,1]>=cy-100)&(B[:,1]<=cy+S+100)]
        if len(A)==0 or len(sel)==0:
            print(f"window ({cx},{cy}): ours={len(A)} ref={len(sel)}",flush=True); continue
        x1=np.maximum(A[:,None,0],sel[None,:,0]); y1=np.maximum(A[:,None,1],sel[None,:,1])
        x2=np.minimum(A[:,None,2],sel[None,:,2]); y2=np.minimum(A[:,None,3],sel[None,:,3])
        inter=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None)
        aa=(A[:,2]-A[:,0])*(A[:,3]-A[:,1]); bb=(sel[:,2]-sel[:,0])*(sel[:,3]-sel[:,1])
        iou=inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-9)
        best=iou.max(1); hit=(best>=0.5)
        print(f"window ({cx},{cy}): ours={len(A)} ref={len(sel)} ours_iou>=0.5: {hit.sum()} "
              f"mean_best_iou={best.mean():.3f} ref_covered={int((iou.max(0)>=0.5).sum())}",flush=True)
