"""把已有 pre.json 的像素几何从"y 被垂直翻转"修正为正确坐标，并重写 geojson/shp。

修正关系：buggy 变换 y_store = y0 + (h - row)  =>  row = y0 + y1 - y_store
同时把每个环的顶点顺序反转（镜像会翻转环绕方向）。
"""
import json, os, sys
sys.path.insert(0, "/data1/wzh26/code/detect-speedup")
import numpy as np
import run_pre1023_p2pformer as R

OUT = "/data1/wzh26/code/detect-speedup/output/pre1023_p2pformer_r50_whumix"
P = os.path.join(OUT, "json", "pre.json")
d = json.load(open(P))
x0_img = 0
items = []
for p in d["polygons"]:
    tx0, ty0, tx1, ty1 = p["tile"]
    rings = []
    for ring in p["rings_px"]:
        fixed = [[float(x), float(ty0 + ty1 - y)] for x, y in ring]
        rings.append(fixed[::-1])
    items.append({"score": p["score"], "rings_px": rings, "tile": p["tile"]})
items.sort(key=lambda a: -a["score"])

from rasterio.transform import Affine
a, b, c, dd, e, f = d["transform"]
tr = Affine(a, b, c, dd, e, f)
out = []
for i, it in enumerate(items):
    rings_geo = R.px_to_geo(it["rings_px"], tr)
    area = R.ring_area_geo(rings_geo[0])
    outer = np.asarray(it["rings_px"][0])
    out.append({
        "id": i + 1,
        "score": it["score"],
        "area_m2": round(area, 2),
        "area_px": int(round(0.5 * abs(np.dot(outer[:, 0], np.roll(outer[:, 1], 1))
                                       - np.dot(outer[:, 1], np.roll(outer[:, 0], 1))))),
        "centroid_px": [round(float(outer[:, 0].mean()), 2), round(float(outer[:, 1].mean()), 2)],
        "rings_px": it["rings_px"],
        "rings_geo": rings_geo,
    })

d["polygons"] = out
d["geometry_fix"] = "y 轴翻转已修正（row = y0 + y1 - y_store），环顶点顺序已反转"
conf = d["conf_thr"]
d["stats"] = {
    "n_polygons": len(out),
    "n_after_conf": int(sum(1 for it in out if it["score"] >= conf)),
    "area_m2_total": round(sum(it["area_m2"] for it in out if it["score"] >= conf), 1),
    "score_max": max([it["score"] for it in out], default=0.0),
}
json.dump(d, open(P, "w"), ensure_ascii=False)
print("json rewritten:", len(out))

geo = {"type": "FeatureCollection", "features": [R.to_geojson_feature(it) for it in out]}
json.dump(geo, open(os.path.join(OUT, "shps", "pre_polygons.geojson"), "w"), ensure_ascii=False)
import rasterio
with rasterio.open(d["image"]) as ds:
    wkt = ds.crs.to_wkt()
for thr in (0.05, 0.1, 0.2, 0.3):
    sub = [it for it in out if it["score"] >= thr]
    if sub:
        R.write_shapefile(sub, os.path.join(OUT, "shps", f"pre_poly_conf{int(thr*100):02d}.shp"), wkt)
        print("shp", thr, len(sub))
