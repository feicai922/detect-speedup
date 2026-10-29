#!/usr/bin/env python
"""大图实例分割推理：GZDA 0.6 m 训练的 yolo26x-seg → 1023 测试影像 pre.tif

模型: /data1/wzh26/code/yolo/runs/segment/gzda_building_map_0.4_0.6m-2/weights/best.pt
      （GZDA 0.6 m 子集 mAP50-95>0.40 过滤后, 300 epoch, 训练 imgsz=640 / 瓦片 1024,
        val 最优 mask mAP50-95 = 0.531，box 0.603）

输入: /data1/wzh26/data/1023测试影像/1/pre.tif（19795 x 20748, 0.6 m/px, EPSG:4326）
      训练与推理同为 0.6 m → 尺度天然对齐，imgsz 保持训练时的 640

输出: ./output/pre1023_gzda_seg/
      json/pre.json            检测 + segment_xy 掩膜多边形
      shps/all_mask_conf_*.shp 合并后的掩膜 Shapefile（按 conf 分档）
      vis/pre_mask_vis.jpg     全图可视化

运行（gpu7，两张空闲 4090）：
    cd /data1/wzh26/code/detect-speedup && \
      /home/wzh26/anaconda3/envs/yolo/bin/python run_pre1023_gzda_seg.py
"""


import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
import datetime as dt
import os
from copy import deepcopy

from detect_speedup import run_yolo

try:
    from detect_speedup.visualize_predictions import run_visualization
except ImportError as e:  # pragma: no cover
    print(f"[warn] visualization import failed: {e}", flush=True)
    run_visualization = None

COMMON = {
    "input_path": "/data1/wzh26/data/1023测试影像/1/pre.tif",
    "output_dir": "./output/pre1023_gzda_seg",
    "devices": ["cuda:0", "cuda:2"],
    "worker_count": None,
    "reader_workers": 1,
    # 训练瓦片 1024 / imgsz 640 → 推理保持一致
    "tile_size": 1024,
    "stride": 896,
    "batch_size": 4,
    "queue_size": 4,
    "save_json": True,
    "stream_json": False,
    "save_bbox_shp": False,   # 合并矢量由 visualize_predictions 统一输出
    "save_mask_shp": False,
    "global_nms": True,
    "global_iou": 0.45,
    "recursive": False,
    "skip_if_output_exists": False,
    "regularize": None,       # 不开轮廓正则化；需要时改成 REGULARIZE
}

YOLO = {
    "model_path": "/data1/wzh26/code/yolo/runs/segment/gzda_building_map_0.4_0.6m-2/weights/best.pt",
    "imgsz": 640,
    "conf": 0.3,
    "iou": 0.45,
    "max_det": 300,
    "agnostic_nms": False,
}

ENABLE_VISUALIZATION = True

VIS = {
    "geometry": "mask",
    "mask_fallback_bbox": True,   # 极少数无 mask 的结果退化成框
    "fill_alpha": 80,
    "min_mask_area": 25,
    "show_legend": False,
    "thresholds": [0.2, 0.3, 0.4, 0.5, 0.6],
    "line_width": 2,
    "jpg_quality": 90,
    "recursive": False,
}

REGULARIZE = {
    "min_area": 20.0,
    "dp_tolerance": 0.5,
    "use_pca": False,
    "angle_threshold": 10.0,
    "snap_angles": [0, 45, 90, 135],
    "enable_symmetry": False,
    "symmetry_tolerance": 2.0,
    "smooth_iterations": 0,
    "smooth_ratio": 0.25,
    "fix_topology": True,
}


def apply_env_overrides():
    if os.environ.get("PRE1023_OUT"):
        COMMON["output_dir"] = os.environ["PRE1023_OUT"]
    if os.environ.get("PRE1023_MODEL"):
        YOLO["model_path"] = os.environ["PRE1023_MODEL"]
    if os.environ.get("PRE1023_TILE"):
        COMMON["tile_size"] = int(os.environ["PRE1023_TILE"])
    if os.environ.get("PRE1023_STRIDE"):
        COMMON["stride"] = int(os.environ["PRE1023_STRIDE"])
    if os.environ.get("PRE1023_BATCH"):
        COMMON["batch_size"] = int(os.environ["PRE1023_BATCH"])
    if os.environ.get("PRE1023_CONF"):
        YOLO["conf"] = float(os.environ["PRE1023_CONF"])
    if os.environ.get("PRE1023_IMGSZ"):
        YOLO["imgsz"] = int(os.environ["PRE1023_IMGSZ"])
    if os.environ.get("PRE1023_REGULARIZE") == "1":
        COMMON["regularize"] = deepcopy(REGULARIZE)


def main():
    apply_env_overrides()
    base_dir = COMMON["output_dir"]
    common = deepcopy(COMMON)
    common["output_dir"] = os.path.join(base_dir, "json")
    if common["worker_count"] is None:
        common["worker_count"] = len(common["devices"])

    print(
        f"[task] backend=yolo(seg) input={common['input_path']} output={common['output_dir']} "
        f"devices={common['devices']} tile={common['tile_size']} stride={common['stride']} "
        f"imgsz={YOLO['imgsz']} conf={YOLO['conf']} regularize={bool(common['regularize'])}",
        flush=True,
    )
    t0 = dt.datetime.now()
    run_yolo(common=common, yolo=deepcopy(YOLO))
    print(f"[infer] done in {dt.datetime.now() - t0}", flush=True)

    if not ENABLE_VISUALIZATION or run_visualization is None:
        print("[vis] skipped", flush=True)
        return

    vis_config = deepcopy(VIS)
    vis_config["input_dir"] = common["output_dir"]
    vis_config["output_dir"] = os.path.join(base_dir, "vis")
    vis_config["shp_output_dir"] = os.path.join(base_dir, "shps")
    vis_config["image_dir"] = common["input_path"]
    inp = common["input_path"]
    if inp.lower().endswith((".tif", ".tiff")) and os.path.isfile(inp):
        vis_config["crs_source"] = inp
    print(f"[vis] json={vis_config['input_dir']} vis={vis_config['output_dir']} "
          f"shp={vis_config['shp_output_dir']}", flush=True)
    run_visualization(vis_config)
    print(f"[all] total {dt.datetime.now() - t0}", flush=True)


if __name__ == "__main__":
    main()
