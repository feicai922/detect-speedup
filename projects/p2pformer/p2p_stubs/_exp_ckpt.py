import os, sys, json, time
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import numpy as np, cv2, rasterio
from rasterio.windows import Window
import run_pre1023_p2pformer as R
BASE="/tmp/p2p_work/P2PFormer/p2pformer/configs/configs/"
CFGS=[("r50_whu_mix", BASE+"p2pformer_corner_r50_whu-mix.py", "/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth"),
      ("r50_crowdai", BASE+"p2pformer_corner_r50_crowdai.py", "/tmp/p2p_work/p2pformer_corner_r50_crowdai.pth"),
      ("dla34_whu_mix", BASE+"p2pformer_corner_dla34_whu-mix.py", "/tmp/p2p_work/p2pformer_corner_dla34_whu_mix.pth")]
img=cv2.imread("/tmp/p2p_work/P2PFormer/demo/demo.jpg")
rgb=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
with rasterio.open("/tmp/p2p_work/pre.tif") as ds:
    tiles={}
    for crop in (256,320,384):
        tiles[crop]=np.ascontiguousarray(ds.read(window=Window(7840,18080,crop,crop)).transpose(1,2,0)[:3])
for name,cfg,ckpt in CFGS:
    try:
        m=R.build_model(cfg, ckpt, "cuda:1")
    except Exception as e:
        print(name,"BUILD FAIL",repr(e)[:200],flush=True); continue
    polys=R.predict_tile(m, rgb, (0,0,rgb.shape[1],rgb.shape[0]), None, min_area_px=40)
    sc=np.array(sorted([p['score'] for p in polys],reverse=True)) if polys else np.zeros(1)
    print(f"[{name}] demo: n={len(polys)} max={sc.max():.3f} p90={np.percentile(sc,90):.3f} n>=0.3={(sc>=0.3).sum()} n>=0.5={(sc>=0.5).sum()}",flush=True)
    for crop,t in tiles.items():
        ps=R.predict_tile(m, t, (7840,18080,7840+crop,18080+crop), None, min_area_px=40)
        s=np.array([p['score'] for p in ps]) if ps else np.zeros(1)
        ar=np.array([p['area_px'] for p in ps]) if ps else np.zeros(1)
        print(f"    tif crop{crop}: n={len(ps)} max={s.max():.3f} n>=0.3={(s>=0.3).sum()} area_med={np.median(ar):.0f}",flush=True)
    del m
