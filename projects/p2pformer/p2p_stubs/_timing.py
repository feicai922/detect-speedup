import os, sys, time
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
T=time.time()
def L(m): print(f"[{time.time()-T:6.1f}s] {m}", flush=True)
import numpy as np, cv2, torch
import run_pre1023_p2pformer as R
cfg = os.environ.get("P2P_CFG", R.DEFAULT_CONFIG)
ckpt = os.environ.get("P2P_CKPT", "/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth")
dev = os.environ.get("P2P_DEV", "cuda:1")
m = R.build_model(cfg, ckpt, dev); L("model ready")
img = cv2.cvtColor(cv2.imread("/data1/wzh26/tmp/p2p_smoke/crop_rgb.png"), cv2.COLOR_BGR2RGB)
img = np.ascontiguousarray(img)
L(f"tile {img.shape}")
for t in range(2):
    t1=time.time(); polys = R.predict_tile(m, img, (9000,17000,9640,17640), None, min_area_px=40)
    sc = sorted([p['score'] for p in polys], reverse=True)[:8]
    ar = sorted([p['area_px'] for p in polys], reverse=True)[:8]
    L(f"single trial{t}: {time.time()-t1:.2f}s polys={len(polys)} scores={sc} area_px={ar}")
for bs in (2, 4):
    t1=time.time(); res = R.predict_batch(m, [img]*bs, [(9000+i*10,17000,9640+i*10,17640) for i in range(bs)], min_area_px=40)
    L(f"batch{bs}: {time.time()-t1:.2f}s per_tile={(time.time()-t1)/bs:.3f}s counts={[len(r) for r in res]}")
