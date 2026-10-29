import os, sys, time
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import numpy as np, rasterio
from rasterio.windows import Window
import run_pre1023_p2pformer as R

SRC = "/tmp/p2p_work/pre.tif"
cfg = os.environ.get("P2P_CFG", R.DEFAULT_CONFIG)
ckpt = os.environ.get("P2P_CKPT", "/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth")
dev = os.environ.get("P2P_DEV", "cuda:1")
m = R.build_model(cfg, ckpt, dev)
print("model ready", flush=True)

TILES = [(7680,17920),(17280,14080),(3200,18560),(8320,16000),(1920,19840),(5120,16000),(18560,11520),(640,17280)]
CROPS = [640, 384, 256]

with rasterio.open(SRC) as ds:
    for crop in CROPS:
        tot = {0.05:0, 0.1:0, 0.3:0, 0.5:0}
        allsc = []
        t1=time.time()
        imgs, bbs = [], []
        for (x0,y0) in TILES:
            cx = x0 + 320 - crop//2
            cy = y0 + 320 - crop//2
            cx = max(0, min(cx, ds.width-crop)); cy = max(0, min(cy, ds.height-crop))
            img = ds.read(window=Window(cx, cy, crop, crop)).transpose(1,2,0)
            imgs.append(np.ascontiguousarray(img[:3])); bbs.append((cx,cy,cx+crop,cy+crop))
        res = R.predict_batch(m, imgs, bbs, min_area_px=40)
        for polys in res:
            for p in polys:
                allsc.append(p["score"])
                for t in tot:
                    if p["score"] >= t: tot[t]+=1
        sc = np.array(allsc) if allsc else np.zeros(1)
        print(f"crop={crop} ({t1 and time.time()-t1:.1f}s) n={len(allsc)} "
              f">=0.05:{tot[0.05]} >=0.1:{tot[0.1]} >=0.3:{tot[0.3]} >=0.5:{tot[0.5]} "
              f"score mean={sc.mean():.3f} p90={np.percentile(sc,90):.3f} max={sc.max():.3f}", flush=True)
