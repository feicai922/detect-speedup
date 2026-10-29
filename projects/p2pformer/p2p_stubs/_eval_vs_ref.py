"""与已训练 yolo26x-seg 参考结果做一致性评估（bbox IoU 贪心匹配）。"""
import json, sys
import numpy as np

OUR = "/data1/wzh26/code/detect-speedup/output/pre1023_p2pformer_r50_whumix/json/pre.json"
REF = "/data1/wzh26/code/detect-speedup/output/pre1023_gzda_seg/json/pre.json"

our = json.load(open(OUR))
dets = our["polygons"]
ref = json.load(open(REF))["detections"]

def poly_bbox(rings):
    r = np.asarray(rings[0])
    return [r[:,0].min(), r[:,1].min(), r[:,0].max(), r[:,1].max()]

A = np.array([poly_bbox(d["rings_px"]) + [d["score"]] for d in dets], dtype=np.float64) if dets else np.zeros((0,5))
B = np.array([d["bbox_xyxy"] + [d["score"]] for d in ref if d["score"] >= 0.4], dtype=np.float64)
print(f"ours={len(A)} ref={len(B)}")

def iou_mat(a, b):
    x1 = np.maximum(a[:,None,0], b[None,:,0]); y1 = np.maximum(a[:,None,1], b[None,:,1])
    x2 = np.minimum(a[:,None,2], b[None,:,2]); y2 = np.minimum(a[:,None,3], b[None,:,3])
    inter = np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None)
    aa = (a[:,2]-a[:,0])*(a[:,3]-a[:,1]); bb = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter / np.maximum(aa[:,None]+bb[None,:]-inter, 1e-9)

for thr in (0.05, 0.1, 0.15, 0.2, 0.3):
    sub = A[A[:,4] >= thr]
    if len(sub) == 0:
        print(f"thr={thr}: none"); continue
    iou = iou_mat(sub, B)
    matched_a = set(); matched_b = set()
    order = np.argsort(-sub[:,4])
    K = list(range(len(B)))
    for i in order:
        j = int(np.argmax(iou[i]))
        if iou[i, j] >= 0.5 and j not in matched_b:
            matched_a.add(int(i)); matched_b.add(j)
    tp = len(matched_a)
    prec = tp/len(sub); rec = tp/max(len(B),1)
    f1 = 2*prec*rec/max(prec+rec,1e-9)
    print(f"thr={thr:.2f}: n={len(sub):6d} TP={tp:6d} precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}")
