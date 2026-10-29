import os, sys
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import numpy as np, cv2
import run_pre1023_p2pformer as R
from mmdet.apis import inference_detector
cfg=os.environ.get("P2P_CFG", R.DEFAULT_CONFIG); ckpt=os.environ.get("P2P_CKPT","/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth")
m=R.build_model(cfg, ckpt, os.environ.get("P2P_DEV","cuda:1"))
orig = m.construct_poly_from_lines
CAP={}
def patched(lines, scores, idxs):
    polys = orig(lines, scores, idxs)
    CAP['lines']=lines.detach().cpu().numpy() if hasattr(lines,'detach') else lines
    CAP['polys']=polys
    return polys
m.construct_poly_from_lines = patched
img=cv2.imread("/tmp/p2p_work/P2PFormer/demo/demo.jpg")
print("img", img.shape, flush=True)
res=R._parse_result(inference_detector(m, np.ascontiguousarray(img)))
bbox_res, segm_res = res
b=bbox_res[0]
lines=CAP['lines']; polys=CAP['polys']
print("n dets", len(b), flush=True)
order=np.argsort(-b[:,4])[:5]
for k in order:
    bb=b[k]
    p=np.asarray(polys[k]) if polys[k] is not None else None
    print(f"score={bb[4]:.3f} det_bbox=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}) wh=({bb[2]-bb[0]:.0f},{bb[3]-bb[1]:.0f})", flush=True)
    if p is not None and len(p):
        print(f"    poly n={len(p)} xr=({p[:,0].min():.1f},{p[:,0].max():.1f}) yr=({p[:,1].min():.1f},{p[:,1].max():.1f})", flush=True)
    print(f"    raw lines[0]={np.round(lines[k][0],3).tolist()}", flush=True)
    print(f"    raw lines[1]={np.round(lines[k][1],3).tolist()}", flush=True)
    mk=np.asarray(segm_res[0][k]) if segm_res[0] else None
    if mk is not None and mk.size:
        ys,xs=np.nonzero(mk)
        if len(ys): print(f"    mask area={mk.sum()} xr=({xs.min()},{xs.max()}) yr=({ys.min()},{ys.max()})", flush=True)
