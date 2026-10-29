import os, sys, json
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import run_pre1023_p2pformer as R
OUT="/data1/wzh26/code/detect-speedup/output/pre1023_p2pformer_r50_whumix"
SRC=os.environ.get("P2P_SRC","/tmp/p2p_work/pre.tif")
d=json.load(open(os.path.join(OUT,"json","pre.json")))
items=d["polygons"]
R.render_overview(SRC, items, os.path.join(OUT,"vis","pre_vis.jpg"))
crops={"dense1":(7680,17920,600),"dense2":(17280,14080,600),"dense3":(3200,18560,600)}
R.render_crops(SRC, items, os.path.join(OUT,"vis","crops"), crops, conf=0.1)
print("vis done", flush=True)
