import os, sys
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import numpy as np
import run_pre1023_p2pformer as R
from mmdet.apis import init_detector, inference_detector
cfg=os.environ.get("P2P_CFG", R.DEFAULT_CONFIG); ckpt=os.environ.get("P2P_CKPT","/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth")
R.import_plugins()
m=init_detector(cfg, ckpt, device="cuda:1"); m.eval()
print("ready", flush=True)
for img_path in ["/tmp/p2p_work/P2PFormer/demo/demo.jpg"]:
    res = inference_detector(m, img_path)
    print(type(res), len(res), flush=True)
    bbox_res, segm_res = res
    b = bbox_res[0]
    print("n dets", len(b), "score stats", (b[:,4].min() if len(b) else None), (b[:,4].max() if len(b) else None), flush=True)
    print("top scores", np.round(np.sort(b[:,4])[::-1][:10],3) if len(b) else None, flush=True)
    print("n masks", len(segm_res[0]) if segm_res and segm_res[0] else 0, flush=True)
