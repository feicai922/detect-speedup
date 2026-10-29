import os, sys
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import numpy as np, cv2, rasterio
from rasterio.windows import Window
import run_pre1023_p2pformer as R
from mmdet.apis import inference_detector
cfg=os.environ.get("P2P_CFG", R.DEFAULT_CONFIG); ckpt=os.environ.get("P2P_CKPT","/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth")
m=R.build_model(cfg, ckpt, os.environ.get("P2P_DEV","cuda:1"))
orig=m.construct_poly_from_lines; CAP={}
def patched(lines,scores,idxs):
    polys=orig(lines,scores,idxs); CAP['polys']=polys; return polys
m.construct_poly_from_lines=patched
with rasterio.open("/tmp/p2p_work/pre.tif") as ds:
    t=ds.read(window=Window(7840,18080,320,320)).transpose(1,2,0)
t=np.ascontiguousarray(t[:3])
res=R._parse_result(inference_detector(m, np.ascontiguousarray(t[:,:,::-1])))
b=res[0][0]; polys=CAP['polys']; masks=res[1][0]
print("n dets",len(b),flush=True)
for k in np.argsort(-b[:,4])[:8]:
    bb=b[k]; p=np.asarray(polys[k]) if polys[k] is not None and len(polys[k]) else None
    mk=np.asarray(masks[k]) if masks and masks[k] is not None else None
    print(f"score={bb[4]:.3f} det=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f})", end=" ")
    if p is not None: print(f"poly n={len(p)} x=({p[:,0].min():.0f},{p[:,0].max():.0f}) y=({p[:,1].min():.0f},{p[:,1].max():.0f})", end=" ")
    if mk is not None and mk.size:
        ys,xs=np.nonzero(mk)
        print(f"mask_area={mk.sum()} x=({xs.min()},{xs.max()}) y=({ys.min()},{ys.max()})", end="")
    print(flush=True)
