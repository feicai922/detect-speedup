import os, sys, json
sys.path.insert(0,'/data1/wzh26/code/detect-speedup')
import numpy as np, rasterio, cv2
from rasterio.windows import Window
import run_pre1023_p2pformer as R
cfg=os.environ.get("P2P_CFG", R.DEFAULT_CONFIG); ckpt=os.environ.get("P2P_CKPT","/tmp/p2p_work/p2pformer_corner_r50_whu_mix.pth")
m=R.build_model(cfg, ckpt, os.environ.get("P2P_DEV","cuda:1")); print("ready",flush=True)
SRC="/tmp/p2p_work/pre.tif"
ref=json.load(open("/data1/wzh26/code/detect-speedup/output/pre1023_gzda_seg/json/pre.json"))["detections"]
X0,Y0=7680,17920
for crop in (320,480,640):
    cx,cy=X0+320-crop//2, Y0+320-crop//2
    with rasterio.open(SRC) as ds:
        img=ds.read(window=Window(cx,cy,crop,crop)).transpose(1,2,0)
    img=np.ascontiguousarray(img[:3])
    polys=R.predict_batch(m,[img],[(cx,cy,cx+crop,cy+crop)],min_area_px=40)[0]
    polys=sorted(polys,key=lambda p:-p["score"])
    print(f"== crop={crop} window=({cx},{cy},{cx+crop},{cy+crop}) n={len(polys)}",flush=True)
    for p in polys[:6]:
        r=np.array(p["rings_px"][0])
        print(f"   score={p['score']:.3f} area_px={p['area_px']} bbox=({r[:,0].min():.0f},{r[:,1].min():.0f},{r[:,0].max():.0f},{r[:,1].max():.0f})",flush=True)
    # 该窗口内的参考建筑（conf>=0.5）
    rr=[]
    for d in ref:
        b=d["bbox_xyxy"]; ccx=(b[0]+b[2])/2; ccy=(b[1]+b[3])/2
        if cx<=ccx<cx+crop and cy<=ccy<cy+crop and d["score"]>=0.5:
            rr.append((round(d["score"],2), round(b[2]-b[0]), round(b[3]-b[1])))
    print(f"   ref buildings in window (score>=0.5): {len(rr)} sizes {rr[:8]}",flush=True)
    # 保存叠加图便于人工检查
    vis=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    for p in polys:
        if p["score"]<0.05: continue
        for ring in p["rings_px"]:
            pts=np.array([[x-cx,y-cy] for x,y in ring],np.int32)
            cv2.polylines(vis,[pts],True,(0,255,0),1,cv2.LINE_AA)
    out=f"/data1/wzh26/tmp/p2p_smoke/overlay_crop{crop}.png"
    cv2.imwrite(out,vis); print("   saved",out,flush=True)
