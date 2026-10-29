import os, sys
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import numpy as np, cv2, rasterio
from rasterio.windows import Window
import run_pre1023_p2pformer as R
from mmdet.apis import inference_detector
cfg=os.environ.get("P2P_CFG", R.DEFAULT_CONFIG); ckpt=os.environ.get("P2P_CKPT","/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth")
m=R.build_model(cfg, ckpt, "cuda:1")
def run(tag, rgb):
    res=R._parse_result(inference_detector(m, np.ascontiguousarray(rgb[:,:,::-1])))
    b=res[0][0]
    if len(b)==0:
        print(f"{tag}: img={rgb.shape[:2]} n=0", flush=True); return
    hs=b[:,3]-b[:,1]; ws=b[:,2]-b[:,0]
    print(f"{tag}: img={rgb.shape[:2]} n={len(b)} score_max={b[:,4].max():.3f} "
          f"h_med={np.median(hs):.1f} w_med={np.median(ws):.1f} "
          f"frac_h_lt5={(hs<5).mean():.2f} y_med={np.median(b[:,1]):.0f}", flush=True)
demo=cv2.cvtColor(cv2.imread("/tmp/p2p_work/P2PFormer/demo/demo.jpg"), cv2.COLOR_BGR2RGB)
run("demo native", demo)
run("demo ->320 (pipeline upscale)", cv2.resize(demo,(320,int(320*demo.shape[0]/demo.shape[1]))))
run("demo ->640 (2x)", cv2.resize(demo,(1280,int(1280*demo.shape[0]/demo.shape[1]))))
with rasterio.open("/tmp/p2p_work/pre.tif") as ds:
    t320=ds.read(window=Window(7840,18080,320,320)).transpose(1,2,0)[:3]
    t1280=ds.read(window=Window(7840,18080,1280,1280)).transpose(1,2,0)[:3]
t320=np.ascontiguousarray(t320); t1280=np.ascontiguousarray(t1280)
run("tif320 pipeline", t320)
run("tif320 ->640 manual resize", cv2.resize(t320,(640,640),interpolation=cv2.INTER_CUBIC))
run("tif1280 (pipeline downsample)", t1280)
print("tif320 stats mean", t320.reshape(-1,3).mean(0), "std", t320.reshape(-1,3).std(0), flush=True)
print("demo stats mean", demo.reshape(-1,3).mean(0), "std", demo.reshape(-1,3).std(0), flush=True)
