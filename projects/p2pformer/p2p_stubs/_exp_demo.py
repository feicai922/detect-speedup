import os, sys
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import numpy as np, cv2
import run_pre1023_p2pformer as R
cfg=os.environ.get("P2P_CFG", R.DEFAULT_CONFIG); ckpt=os.environ.get("P2P_CKPT","/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth")
m=R.build_model(cfg, ckpt, os.environ.get("P2P_DEV","cuda:1")); print("ready", flush=True)
p="/tmp/p2p_work/P2PFormer/demo/demo.jpg"
bgr=cv2.imread(p); print("demo shape", bgr.shape, "mean(BGR)", bgr.mean(axis=(0,1)), flush=True)
rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
polys=R.predict_tile(m, np.ascontiguousarray(rgb), (0,0,bgr.shape[1],bgr.shape[0]), None, min_area_px=40)
sc=sorted([x['score'] for x in polys], reverse=True)
print("n=",len(polys), "top scores", [round(s,3) for s in sc[:15]], flush=True)
print("n>=0.3:",sum(1 for s in sc if s>=0.3), "n>=0.5:",sum(1 for s in sc if s>=0.5), flush=True)
# 反向通道再试一次（若预处理通道搞反）
rgb2=bgr[:,:,::-1].copy()
polys2=R.predict_tile(m, np.ascontiguousarray(rgb2), (0,0,bgr.shape[1],bgr.shape[0]), None, min_area_px=40)
print("swap-channel n=",len(polys2),"top",[round(x['score'],3) for x in sorted([y['score'] for y in polys2],reverse=True)[:8]], flush=True)
