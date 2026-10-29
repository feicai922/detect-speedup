import os, sys, time, json
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import numpy as np, rasterio
from rasterio.windows import Window
import run_pre1023_p2pformer as R

SRC="/tmp/p2p_work/pre.tif"
cfg=os.environ.get("P2P_CFG", R.DEFAULT_CONFIG); ckpt=os.environ.get("P2P_CKPT","/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth")
m=R.build_model(cfg, ckpt, os.environ.get("P2P_DEV","cuda:1")); print("model ready", flush=True)

ref=json.load(open("/data1/wzh26/code/detect-speedup/output/pre1023_gzda_seg/json/pre.json"))["detections"]
refc=np.array([[(d["bbox_xyxy"][0]+d["bbox_xyxy"][2])/2,(d["bbox_xyxy"][1]+d["bbox_xyxy"][3])/2, d["score"]] for d in ref])
refc=refc[refc[:,2]>=0.4]
TILES=[(7680,17920),(17280,14080),(3200,18560),(18560,11520),(5120,16000)]
CROPS=[256,320,384,512,640]

with rasterio.open(SRC) as ds:
    for crop in CROPS:
        t1=time.time(); imgs=[]; bbs=[]
        for (x0,y0) in TILES:
            cx=max(0,min(x0+320-crop//2, ds.width-crop)); cy=max(0,min(y0+320-crop//2, ds.height-crop))
            img=ds.read(window=Window(cx,cy,crop,crop)).transpose(1,2,0)
            imgs.append(np.ascontiguousarray(img[:3])); bbs.append((cx,cy,cx+crop,cy+crop))
        res=R.predict_batch(m, imgs, bbs, min_area_px=40)
        dt=time.time()-t1
        # 统计
        for thr in (0.05,0.1,0.2,0.3):
            n=0; hit=0; nref=0
            for (bx0,by0,bx1,by1), polys in zip(bbs,res):
                sel=[p for p in polys if p["score"]>=thr]
                n+=len(sel)
                # 质心多边形的 bbox 覆盖判断
                cand=[]
                for p in sel:
                    ring=np.array(p["rings_px"][0]); cand.append((ring[:,0].min(),ring[:,1].min(),ring[:,0].max(),ring[:,1].max()))
                cand=np.array(cand) if cand else np.zeros((0,4))
                mref=refc[(refc[:,0]>=bx0)&(refc[:,0]<bx1)&(refc[:,1]>=by0)&(refc[:,1]<by1)]
                nref+=len(mref)
                for r in mref:
                    if len(cand) and np.any((r[0]>=cand[:,0])&(r[0]<=cand[:,2])&(r[1]>=cand[:,1])&(r[1]<=cand[:,3])):
                        hit+=1
            print(f"crop={crop} thr={thr}: dets={n} ref_bld={nref} recall={hit/max(nref,1):.3f}", flush=True)
        print(f"crop={crop} time {dt:.1f}s for {len(TILES)} tiles -> {dt/len(TILES):.3f}s/tile", flush=True)
