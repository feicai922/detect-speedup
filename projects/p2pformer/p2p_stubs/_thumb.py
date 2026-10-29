import os, sys, json
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import cv2, numpy as np
import run_pre1023_p2pformer as R
OUT="/data1/wzh26/code/detect-speedup/output/pre1023_p2pformer_r50_whumix"
SRC=os.environ.get("P2P_SRC","/tmp/p2p_work/pre.tif")
d=json.load(open(os.path.join(OUT,"json","pre.json")))
items=[p for p in d["polygons"] if p["score"]>=0.10]
print("polys for thumb:", len(items), flush=True)
R.render_overview(SRC, items, os.path.join(OUT,"vis","pre_vis_1600.jpg"),
                  max_width=1600, thresholds=(0.10,0.20,0.30))
# 放大图转 jpg（便于查看）
p=os.path.join(OUT,"vis","crops","dense1.png")
img=cv2.imread(p)
cv2.imwrite(os.path.join(OUT,"vis","crops","dense1.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY),92])
print("crop jpg", img.shape, flush=True)
