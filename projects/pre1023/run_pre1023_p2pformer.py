#!/usr/bin/env python
"""大图推理：开源 P2PFormer（R50 + WHU-Mix 预训练）→ 1023 测试影像 pre.tif

模型权重: /data1/wzh26/model_zoo/P2PFormer/p2pformer_corner_r50_whu_mix.pth
          （HuggingFace: zhangtao-whu/P2PFormer，TGRS 2024，
            P2PFormer: A Primitive-to-polygon Method for Regular Building
            Contour Extraction from Remote Sensing Images）
配置:     /data1/wzh26/code/P2PFormer/p2pformer/configs/configs/p2pformer_corner_r50_whu-mix.py

输入: /data1/wzh26/data/1023测试影像/1/pre.tif
      19795 x 20748, 3 波段 uint8, EPSG:4326, 约 0.6 m/px

推理方式（参考 detect-speedup 的大图切瓦思路）:
  1) rasterio 按窗口流式读瓦片（RGB→BGR，与 caffe 均值/to_rgb=False 对齐）
  2) 每瓦片送 P2PFormer（test pipeline 内部 resize 到 640）
  3) 掩膜 → 矢量多边形（rasterio.features.shapes，保留孔洞）
  4) 按瓦片"归属区"去重（只保留质心落在非重叠核心区的实例）
  5) 输出 JSON / GeoJSON / Shapefile(EPSG:4326) / 全图可视化 + 局部放大图

环境（必须是 p2pformer env，PYTHONPATH 指向 P2PFormer 仓库）:
  cd /data1/wzh26/code/detect-speedup && \
    PYTHONPATH=/data1/wzh26/code/P2PFormer \
    /home/wzh26/anaconda3/envs/p2pformer/bin/python run_pre1023_p2pformer.py \
      --devices cuda:1,cuda:3 --out ./output/pre1023_p2pformer_r50_whumix
"""


import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from copy import deepcopy

import cv2
import numpy as np

# P2PFormer 依赖可选包 `visualizer`（仅用于注意力可视化 demo），这里注入 stub
_STUB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2p_stubs")
if os.path.isdir(_STUB_DIR) and _STUB_DIR not in sys.path:
    sys.path.insert(0, _STUB_DIR)

DEFAULT_INPUT = "/data1/wzh26/data/1023测试影像/1/pre.tif"
DEFAULT_CONFIG = "/data1/wzh26/code/P2PFormer/p2pformer/configs/configs/p2pformer_corner_r50_whu-mix.py"
DEFAULT_CKPT = "/data1/wzh26/model_zoo/P2PFormer/p2pformer_corner_r50_whu_mix.pth"
OUT_ROOT = "/data1/wzh26/code/detect-speedup/output/pre1023_p2pformer_r50_whumix"

REG_PREFIXES = [
    "p2pformer.models.custom_p2pformer_plugin",
    "p2pformer.models.p2pformer_head",
    "p2pformer.models.p2pformer",
    "p2pformer.models.line_assigner",
    "p2pformer.dataset.pipelines.custom_pipelines",
]


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def import_plugins():
    import importlib
    for name in REG_PREFIXES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - 诊断用
            print(f"[warn] import {name} failed: {exc}", flush=True)


def build_model(config, checkpoint, device):
    import torch
    from mmdet.apis import init_detector

    import_plugins()
    torch.set_grad_enabled(False)
    model = init_detector(config, checkpoint, device=device)
    model.eval()
    # 单类建筑：关掉不需要的 cfg 覆盖，保留 config 内的 test_cfg
    print(f"[model] {type(model).__name__} loaded on {device} (conf_thr={model.cfg.model.test_cfg.get('score_thr')})",
          flush=True)
    return model


# --------------------------------------------------------------------------- #
# 瓦片网格
# --------------------------------------------------------------------------- #
def tile_grid(width, height, tile, stride):
    xs = list(range(0, max(width - tile, 0) + 1, stride))
    ys = list(range(0, max(height - tile, 0) + 1, stride))
    if xs[-1] + tile < width:
        xs.append(width - tile)
    if ys[-1] + tile < height:
        ys.append(height - tile)
    out = []
    for y0 in ys:
        for x0 in xs:
            out.append((x0, y0, min(x0 + tile, width), min(y0 + tile, height)))
    return out


def ownership_margin(tile, stride):
    return max((tile - stride) // 2, 0)


# --------------------------------------------------------------------------- #
# 单瓦片推理
# --------------------------------------------------------------------------- #
def _parse_result(result):
    """兼容本 repo simple_test 的返回结构：[[(bbox_results, segm_results)]]。"""
    while isinstance(result, (list, tuple)) and len(result) == 1 and isinstance(result[0], (list, tuple)):
        result = result[0]
    if isinstance(result, tuple) and len(result) == 2:
        return result
    raise ValueError(f"unexpected result structure: {type(result)} len={len(result) if hasattr(result,'__len__') else '?'}")


def _masks_to_polys(tile_rgb, tile_bbox, bbox_res, segm_res, min_area_px=30):
    from rasterio.features import shapes as rio_shapes
    from rasterio.transform import Affine

    x0, y0, x1, y1 = tile_bbox
    scores = bbox_res[0] if len(bbox_res) else np.zeros((0, 5), np.float32)
    masks = segm_res[0] if len(segm_res) else []
    polys = []
    if len(masks) == 0:
        return polys
    n = min(len(masks), len(scores))
    h, w = tile_rgb.shape[:2]
    tr = Affine(1, 0, 0, 0, 1, 0)  # 像素坐标：x=列, y=行（不做垂直翻转）
    for i in range(n):
        score = float(scores[i][-1])
        m = masks[i]
        if m is None or getattr(m, "size", 0) == 0:
            continue
        m = np.asarray(m).astype(np.uint8)
        if m.shape != (h, w):
            continue
        if int(m.sum()) < min_area_px:
            continue
        for geom, _val in rio_shapes(m, mask=m.astype(bool), transform=tr):
            rings = geom["coordinates"]
            rings_px = [[[px + x0, py + y0] for px, py in ring] for ring in rings]
            outer = np.asarray(rings_px[0], dtype=np.float64)
            if len(outer) < 3:
                continue
            area = 0.5 * abs(
                np.dot(outer[:, 0], np.roll(outer[:, 1], 1)) - np.dot(outer[:, 1], np.roll(outer[:, 0], 1))
            )
            if area < min_area_px:
                continue
            cx = float(outer[:, 0].mean())
            cy = float(outer[:, 1].mean())
            polys.append({
                "score": round(score, 4),
                "area_px": int(round(area)),
                "centroid": [round(cx, 2), round(cy, 2)],
                "rings_px": rings_px,
                "tile": [x0, y0, x1, y1],
            })
    return polys


def predict_batch(model, tiles_rgb, tiles_bbox, min_area_px=30):
    """tiles_rgb: N 张 HWC RGB uint8。返回每个瓦片的像素坐标多边形列表。"""
    import torch
    from mmdet.apis import inference_detector

    if len(tiles_rgb) == 1:
        bgr = np.ascontiguousarray(tiles_rgb[0][:, :, ::-1])
        result = _parse_result(inference_detector(model, bgr))
        return [_masks_to_polys(tiles_rgb[0], tiles_bbox[0], result[0], result[1], min_area_px)]

    bgrs = [np.ascontiguousarray(t[:, :, ::-1]) for t in tiles_rgb]
    results = inference_detector(model, bgrs)  # 批量
    out = []
    for tile, bb, res in zip(tiles_rgb, tiles_bbox, results):
        bbox_res, segm_res = _parse_result(res)
        out.append(_masks_to_polys(tile, bb, bbox_res, segm_res, min_area_px))
    return out


def predict_tile(model, tile_rgb, tile_bbox, img_bounds=None, min_area_px=30):
    return predict_batch(model, [tile_rgb], [tile_bbox], min_area_px=min_area_px)[0]


# --------------------------------------------------------------------------- #
# 坐标/几何工具
# --------------------------------------------------------------------------- #
def px_to_geo(rings_px, transform):
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    out = []
    for ring in rings_px:
        out.append([[a * px + b * py + c, d * px + e * py + f] for px, py in ring])
    return out


def ring_area_geo(ring):
    """近似面积（m²）：度 → 米。"""
    if len(ring) < 3:
        return 0.0
    lat = sum(p[1] for p in ring) / len(ring)
    mx = 111320.0 * math.cos(math.radians(lat))
    my = 110540.0
    xy = np.asarray(ring, dtype=np.float64)
    x = xy[:, 0] * mx
    y = xy[:, 1] * my
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def to_geojson_feature(item):
    return {
        "type": "Feature",
        "properties": {
            "id": item["id"],
            "score": item["score"],
            "area_m2": item["area_m2"],
            "area_px": item["area_px"],
            "centroid_px": item["centroid"],
        },
        "geometry": {"type": "Polygon", "coordinates": item["rings_geo"]},
    }


def write_shapefile(items, path, crs_wkt):
    import shapefile  # pyshp
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = shapefile.Writer(path, shapeType=shapefile.POLYGON)
    w.field("id", "N", 10, 0)
    w.field("score", "F", 12, 4)
    w.field("area_m2", "F", 14, 3)
    w.field("area_px", "N", 12, 0)
    for it in items:
        rings = it["rings_geo"]
        w.poly([[[round(x, 8), round(y, 8)] for x, y in ring] for ring in rings])
        w.record(it["id"], it["score"], round(it["area_m2"], 3), it["area_px"])
    w.close()
    with open(os.path.splitext(path)[0] + ".prj", "w", encoding="utf-8") as f:
        f.write(crs_wkt if crs_wkt else "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]")


# --------------------------------------------------------------------------- #
# 多进程 worker
# --------------------------------------------------------------------------- #
_G = {}


def _worker_init(config, checkpoint, device, tile, stride, conf, min_area_px, src, batch_size=1):
    import torch
    import rasterio
    torch.cuda.set_device(torch.device(device))
    _G["model"] = build_model(config, checkpoint, device)
    _G["src"] = src
    _G["tile"] = tile
    _G["conf"] = conf
    _G["min_area_px"] = min_area_px
    _G["batch_size"] = batch_size


def _worker_run(args):
    idx_range, bboxes = args
    import rasterio
    from rasterio.windows import Window
    model = _G["model"]
    conf = _G["conf"]
    bs = _G.get("batch_size", 1)
    out = []
    t_start = time.time()
    with rasterio.open(_G["src"]) as ds:
        for i in range(0, len(bboxes), bs):
            chunk = bboxes[i:i + bs]
            tiles, imgs = [], []
            for x0, y0, x1, y1 in chunk:
                img = ds.read(window=Window(x0, y0, x1 - x0, y1 - y0))
                img = np.transpose(img[:3], (1, 2, 0))
                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                imgs.append(img)
                tiles.append((x0, y0, x1, y1))
            batch_polys = predict_batch(model, imgs, tiles, min_area_px=_G["min_area_px"])
            for polys in batch_polys:
                for p in polys:
                    if p["score"] >= conf:
                        out.append(p)
            if (i // bs) % 20 == 0:
                done = i + len(chunk)
                rate = done / max(time.time() - t_start, 1e-6)
                print(f"[worker {idx_range[0]}-{idx_range[1]}] {done}/{len(bboxes)} tiles "
                      f"({rate:.2f} tile/s), polys={len(out)}", flush=True)
    print(f"[worker {idx_range[0]}-{idx_range[1]}] done, {len(out)} raw polys", flush=True)
    return out


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def render_overview(src, items, out_jpg, max_width=3200, thresholds=(0.05, 0.1, 0.2)):
    import rasterio
    with rasterio.open(src) as ds:
        step = max(1, int(math.ceil(max(ds.width, ds.height) / max_width)))
        arr = ds.read(out_shape=(3, ds.height // step, ds.width // step))
    img = np.transpose(arr[:3], (1, 2, 0))
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img = np.ascontiguousarray(img[:, :, ::-1])  # BGR for cv2
    colors = {0.05: (0, 200, 0), 0.1: (0, 220, 255), 0.2: (0, 0, 255)}
    for it in items:
        sc = it["score"]
        lvl = None
        for t in sorted(thresholds):
            if sc >= t:
                lvl = t
        if lvl is None:
            continue
        color = colors.get(lvl, (0, 200, 0))
        for ring in it["rings_px"]:
            pts = np.asarray([[px / step, py / step] for px, py in ring], dtype=np.int32)
            cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_jpg), exist_ok=True)
    cv2.imwrite(out_jpg, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"[vis] {out_jpg} ({img.shape[1]}x{img.shape[0]}, step={step}, polys={len(items)})", flush=True)


def render_crops(src, items, out_dir, crops, conf=0.3):
    import rasterio
    from rasterio.windows import Window
    os.makedirs(out_dir, exist_ok=True)
    for name, (cx, cy, half) in crops.items():
        x0, y0 = max(cx - half, 0), max(cy - half, 0)
        x1, y1 = x0 + 2 * half, y0 + 2 * half
        with rasterio.open(src) as ds:
            x1, y1 = min(x1, ds.width), min(y1, ds.height)
            arr = ds.read(window=Window(x0, y0, x1 - x0, y1 - y0))
        img = np.ascontiguousarray(np.transpose(arr[:3], (1, 2, 0))[:, :, ::-1])
        n = 0
        for it in items:
            if it["score"] < conf:
                continue
            for ring in it["rings_px"]:
                pts = np.asarray([[px - x0, py - y0] for px, py in ring], dtype=np.int32)
                if pts[:, 0].max() < 0 or pts[:, 1].max() < 0 or pts[:, 0].min() > (x1 - x0) or pts[:, 1].min() > (y1 - y0):
                    continue
                cv2.polylines(img, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
                n += 1
        p = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(p, img)
        print(f"[vis-crop] {p} ({n} rings)", flush=True)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--devices", default="cuda:1")
    ap.add_argument("--tile", type=int, default=384, help="原生像素瓦片边长（模型内部 resize 到 640；384 原生 ≈ 0.36 m/px，最接近 WHU 0.3 m 训练尺度）")
    ap.add_argument("--stride", type=int, default=288)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--min-area-px", type=int, default=40)
    ap.add_argument("--max-images", type=int, default=0, help="调试：只跑前 N 个瓦片")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--crops", default="", help="额外放大裁剪: name:x:y:half;...")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import rasterio
    from multiprocessing import get_context

    t0 = dt.datetime.now()
    os.makedirs(args.out, exist_ok=True)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    with rasterio.open(args.input) as ds:
        W, H, transform, crs = ds.width, ds.height, ds.transform, ds.crs
    print(f"[task] {args.input} {W}x{H} crs={crs} tile={args.tile} stride={args.stride} "
          f"devices={devices} conf={args.conf} ckpt={os.path.basename(args.checkpoint)}", flush=True)

    grid = tile_grid(W, H, args.tile, args.tile if args.stride >= args.tile else args.stride)
    if args.max_images:
        grid = grid[: args.max_images]
    print(f"[task] tiles={len(grid)}", flush=True)

    # 分片给多卡
    chunks = [[] for _ in devices]
    for i, bb in enumerate(grid):
        chunks[i % len(devices)].append(bb)

    raw = run_multi(args, chunks, devices, get_context)

    print(f"[infer] {len(raw)} raw polys in {dt.datetime.now() - t0}", flush=True)

    # ---- 去重（瓦片归属核心区）+ 面积换算 ----
    margin = ownership_margin(args.tile, min(args.stride, args.tile))
    items, drop = [], 0
    seen = []
    for p in raw:
        x0, y0, x1, y1 = p["tile"]
        cx, cy = p["centroid"]
        lm = 0 if x0 <= 0 else margin
        tm = 0 if y0 <= 0 else margin
        rm = 0 if x1 >= W else margin
        bm = 0 if y1 >= H else margin
        if not (x0 + lm - 0.5 <= cx <= x1 - rm + 0.5 and y0 + tm - 0.5 <= cy <= y1 - bm + 0.5):
            drop += 1
            continue
        rings_px = [[[round(x, 2), round(y, 2)] for x, y in ring] for ring in p["rings_px"]]
        rings_geo = px_to_geo(rings_px, transform)
        area = ring_area_geo(rings_geo[0])
        seen.append({
            "score": p["score"],
            "tile": [x0, y0, x1, y1],
            "area_m2": round(area, 2),
            "area_px": p["area_px"],
            "centroid": [round(cx, 2), round(cy, 2)],
            "rings_px": rings_px,
            "rings_geo": rings_geo,
        })
    seen.sort(key=lambda d: -d["score"])
    for i, it in enumerate(seen):
        it["id"] = i + 1
        items.append(it)
    print(f"[dedup] kept {len(items)} / {len(raw)} (dropped {drop} outside ownership core)", flush=True)

    # ---- 落盘 ----
    json_dir = os.path.join(args.out, "json")
    shp_dir = os.path.join(args.out, "shps")
    vis_dir = os.path.join(args.out, "vis")
    for d in (json_dir, shp_dir, vis_dir):
        os.makedirs(d, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.input))[0]

    payload = {
        "image": args.input,
        "width": W,
        "height": H,
        "crs": str(crs),
        "transform": list(transform)[:6],
        "model": {
            "name": "P2PFormer (R50, WHU-Mix)",
            "config": args.config,
            "checkpoint": args.checkpoint,
        },
        "tile": args.tile,
        "stride": args.stride,
        "conf_thr": args.conf,
        "min_area_px": args.min_area_px,
        "elapsed": str(dt.datetime.now() - t0),
        "stats": {
            "n_polygons": len(items),
            "n_after_conf": int(sum(1 for it in items if it["score"] >= args.conf)),
            "area_m2_total": round(sum(it["area_m2"] for it in items if it["score"] >= args.conf), 1),
            "score_max": max([it["score"] for it in items], default=0.0),
        },
        "polygons": [
            {
                "id": it["id"],
                "score": it["score"],
                "area_m2": it["area_m2"],
                "area_px": it["area_px"],
                "centroid_px": it["centroid"],
                "rings_px": it["rings_px"],
                "rings_geo": it["rings_geo"],
                "tile": it.get("tile"),
            }
            for it in items
        ],
    }
    p_json = os.path.join(json_dir, f"{name}.json")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[save] {p_json}", flush=True)

    p_geojson = os.path.join(shp_dir, f"{name}_polygons.geojson")
    with open(p_geojson, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection",
                   "features": [to_geojson_feature(it) for it in items]}, f, ensure_ascii=False)
    print(f"[save] {p_geojson}", flush=True)

    for thr in (0.05, 0.1, 0.2, 0.3):
        sub = [it for it in items if it["score"] >= thr]
        if not sub:
            continue
        p = os.path.join(shp_dir, f"{name}_poly_conf{int(thr*100):02d}.shp")
        write_shapefile(sub, p, crs.to_wkt() if crs else None)
        print(f"[save] {p} ({len(sub)})", flush=True)

    if not args.no_vis:
        render_overview(args.input, items, os.path.join(vis_dir, f"{name}_vis.jpg"))
        crops = {"crop1": (9000, 17000, 800), "crop2": (14000, 9000, 800), "crop3": (5000, 5000, 800)}
        if args.crops:
            crops = {}
            for spec in args.crops.split(";"):
                nm, cx, cy, half = spec.split(":")
                crops[nm] = (int(cx), int(cy), int(half))
        render_crops(args.input, items, os.path.join(vis_dir, "crops"), crops, conf=args.conf)

    print(f"[all] total {dt.datetime.now() - t0} | polygons={len(items)}", flush=True)
    return 0


def _worker_target(dev, bb_list, qi, kw):
    _worker_init(kw["config"], kw["checkpoint"], dev, kw["tile"], kw["stride"],
                 kw["conf"], kw["min_area_px"], kw["input"], kw.get("batch_size", 1))
    res = _worker_run(((0, len(bb_list)), bb_list))
    qi.put(res)


def run_multi(args, chunks, devices, get_context):
    """每卡一个子进程：worker 内部自行初始化对应 device 的模型。"""
    ctx = get_context("spawn")
    q = ctx.Queue()
    kw = {"config": args.config, "checkpoint": args.checkpoint, "tile": args.tile,
          "stride": args.stride, "conf": args.conf, "min_area_px": args.min_area_px,
          "input": args.input, "batch_size": args.batch_size}

    procs = []
    for dev, bb_list in zip(devices, chunks):
        if not bb_list:
            continue
        p = ctx.Process(target=_worker_target, args=(dev, bb_list, q, kw))
        p.start()
        procs.append(p)
    raw = []
    pending = len(procs)
    gotten = 0
    while gotten < pending:
        try:
            res = q.get(timeout=30)
        except Exception:
            if not any(p.is_alive() for p in procs):
                raise RuntimeError("all workers exited before reporting results")
            continue
        raw.extend(res)
        gotten += 1
    for p in procs:
        p.join()
    return raw


if __name__ == "__main__":
    sys.exit(main())
