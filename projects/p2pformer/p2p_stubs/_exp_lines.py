import os, sys
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import numpy as np, cv2, rasterio
from rasterio.windows import Window
import run_pre1023_p2pformer as R
cfg=os.environ.get("P2P_CFG", R.DEFAULT_CONFIG); ckpt=os.environ.get("P2P_CKPT","/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth")
m=R.build_model(cfg, ckpt, os.environ.get("P2P_DEV","cuda:1"))
print("ready", flush=True)
orig = m.construct_poly_from_lines
LOG=[]
def patched(lines, scores, idxs):
    polys = orig(lines, scores, idxs)
    ls = scores.detach().cpu().numpy() if hasattr(scores,'detach') else np.asarray(scores)
    LOG.append((lines.shape, ls.copy(), [None if p is None else np.asarray(p).shape for p in polys], [None if p is None else np.asarray(p)[:4].tolist() for p in polys]))
    return polys
m.construct_poly_from_lines = patched

def report(tag):
    n=len(LOG)
    print(f"== {tag}: instances={n}", flush=True)
    for i,(shp, ls, pshp, head) in enumerate(LOG[:4]):
        print(f"  inst{i}: lines{shp} line_score max={ls.max():.3f} mean={ls.mean():.3f} n>=0.8={(ls>=0.8).sum()}",
              f"poly_shapes={pshp[:4]} first_pts={head[:2]}", flush=True)
    LOG.clear()

# demo.jpg
img = cv2.imread("/tmp/p2p_work/P2PFormer/demo/demo.jpg")
print("demo", img.shape, flush=True)
bgr = np.ascontiguousarray(img)
from mmdet.apis import inference_detector
res = R._parse_result(inference_detector(m, bgr))
report("demo.jpg")

# tif tile
with rasterio.open("/tmp/p2p_work/pre.tif") as ds:
    t = ds.read(window=Window(7840,18080,320,320)).transpose(1,2,0)
t = np.ascontiguousarray(t[:3])
res = R._parse_result(inference_detector(m, np.ascontiguousarray(t[:,:,::-1])))
report("tif crop320")
