#!/usr/bin/env python
"""Unified large-image inference — config-driven entry point.

Edit the config dicts below, then run:
    python run_demo_config.py
"""

import os
import datetime as dt
from copy import deepcopy

from detect_speedup import run_rfdetr, run_rtmdet, run_yolo

try:
    from detect_speedup.visualize_predictions import run_visualization
except ImportError as e:
    print(f"[warn] visualization import failed: {e}", flush=True)
    run_visualization = None

# ============================================================
# 1. 选择后端: "yolo" | "rtmdet" | "rfdetr"
# ============================================================
BACKEND = "rtmdet"

# ============================================================
# 2. 通用配置
# ============================================================
COMMON = {
    "input_path": "/data1/wzh26/code/detect-speedup/input/加拉斯加/img.tif",
    "output_dir": "./output/rtmdet_加拉斯加",  # None = 自动生成 ./output/<backend>/<MMDD_HHMM>
    # GPU 设备: ["cuda:0"] / ["cuda:0","cuda:1"]; NPU: ["npu:0","npu:1"]; CPU: ["cpu"]
    "devices": ["cuda:0","cuda:1","cuda:5","cuda:6"],
    "worker_count": None,  # None = 自动等于 len(devices)
    # 切块参数
    "tile_size": 1280,
    "stride": 1024,
    "batch_size": 4,
    # 输出控制
    "save_json": True,
    "save_bbox_shp": True,
    "save_mask_shp": True,
    "global_nms": True,
    "global_iou": 0.45,
    "recursive": False,
    "skip_if_output_exists": False,
    # 建筑物轮廓正则化（推理完成后对掩膜进行简化/直角化等）
    # None = 不启用；填 dict 即启用（参数见下方 REGULARIZE）
    "regularize": None,
}

# ============================================================
# 2.5 是否在推理后跑可视化（生成 vis/*.jpg + 合并 shps/）
#     默认 False —— 推理只产出 json/ + per-tile shp。
#     如需合并 SHP，单跑 `python run_shp_only.py <json_dir> --crs-source ...` 即可。
#     超大图建议保持 False（JPG 渲染吃内存且无价值）。
# ============================================================
ENABLE_VISUALIZATION = False

# ============================================================
# 3. 模型配置（按所选后端填写对应的）
# ============================================================

# --- YOLO ---
YOLO = {
    "model_path": "/data1/wzh26/code/yolo/runs/segment/gzda_building_seg/weights/best.pt",
    "imgsz": 1024,
    "conf": 0.3,
    "iou": 0.4,
    "max_det": 300,
    "agnostic_nms": False,
}

# --- RTMDet / MMDetection ---
RTMDET = {
    "config_file": "/data1/wzh26/code/mmdetection/work_dirs/gzda_rtmdet/gzda_rtmdet.py",
    "checkpoint_file": "/data1/wzh26/code/mmdetection/work_dirs/gzda_rtmdet/best_coco_segm_mAP_epoch_281.pth",
    "score_thr": 0.3,
    "fp16": False,
    "save_mask_overlay": False,
}

# --- RF-DETR ---
RFDETR = {
    "checkpoint": "/path/to/checkpoint_best_regular.pth",
    "num_classes": 1,
    "resolution": 1088,
    "conf_threshold": 0.3,
    "iou_threshold": 0.5,
}

# ============================================================
# 4. 可视化配置
# ============================================================
VIS = {
    "geometry": "mask",       # bbox | mask | obb  (default: instance segmentation)
    "mask_fallback_bbox": True,
    "fill_alpha": 80,          # mask fill opacity (0=outline only, 80=semi-transparent)
    "min_mask_area": 25,        # filter masks smaller than N pixels
    "show_legend": False,       # hide legend by default
    "thresholds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "line_width": 2,
    "jpg_quality": 92,
    "recursive": False,
}

# ============================================================
# 4.5 建筑物轮廓正则化配置（填给 COMMON["regularize"] 即启用）
#     参数与 building-regularize 项目一致，容差单位为米（自动投影到 UTM）。
#     设为 None 可关闭。
# ============================================================
REGULARIZE = {
    "min_area": 20.0,           # 最小面积 (m²)，过滤误检小碎块
    "dp_tolerance": 0.5,        # Douglas-Peucker 简化容差 (m)
    "use_pca": False,           # True=PCA / False=最小外接矩形 检测主方向
    "angle_threshold": 10.0,    # 直角化角度吸附阈值 (°)
    "snap_angles": [0, 45, 90, 135],
    "enable_symmetry": False,   # 是否启用对称修正
    "symmetry_tolerance": 2.0,
    "smooth_iterations": 0,     # Chaikin 平滑迭代次数 (0=不平滑)
    "smooth_ratio": 0.25,
    "fix_topology": True,
}
# REGULARIZE = None  # 关闭正则化


def main():
    # Resolve backend config
    backend = BACKEND.lower()
    if backend == "yolo":
        backend_config = deepcopy(YOLO)
        run_fn = run_yolo
    elif backend == "rtmdet":
        backend_config = deepcopy(RTMDET)
        run_fn = run_rtmdet
    elif backend in ("rfdetr", "rf-detr"):
        backend_config = deepcopy(RFDETR)
        run_fn = run_rfdetr
    else:
        raise SystemExit(f"Unknown backend: {BACKEND}. Choose yolo / rtmdet / rfdetr.")

    # Build common config — JSON/SHP 写入 json/ 子目录
    common = deepcopy(COMMON)
    common["regularize"] = REGULARIZE
    base_dir = common["output_dir"]
    if base_dir is None:
        base_dir = os.path.join(
            ".", "output", backend, dt.datetime.now().strftime("%m%d_%H%M")
        )
    common["output_dir"] = os.path.join(base_dir, "json")
    if common["worker_count"] is None:
        common["worker_count"] = len(common["devices"])

    print(
        f"[task] backend={backend}, input={common['input_path']}, "
        f"output={common['output_dir']}, devices={common['devices']}",
        flush=True,
    )

    # Run inference
    if backend == "yolo":
        run_yolo(common=common, yolo=backend_config)
    elif backend == "rtmdet":
        run_rtmdet(common=common, rtmdet=backend_config)
    elif backend in ("rfdetr", "rf-detr"):
        run_rfdetr(common=common, rfdetr=backend_config)

    # Run visualization — 输出到和 json/ 同级的 vis/ 目录
    if not ENABLE_VISUALIZATION:
        print(
            "[vis] skipped (ENABLE_VISUALIZATION=False). "
            "推理只产出 json/ + per-tile shp；"
            "如需合并 SHP 单跑 `python run_shp_only.py <json_dir> --crs-source ...`。",
            flush=True,
        )
        return

    if run_visualization is None:
        print("[vis] skipped — PIL not installed", flush=True)
        return

    vis_config = deepcopy(VIS)
    vis_config["input_dir"] = common["output_dir"]
    vis_config["output_dir"] = os.path.join(base_dir, "vis")
    vis_config["shp_output_dir"] = os.path.join(base_dir, "shps")
    vis_config["image_dir"] = common["input_path"]
    # When the source is a single TIF (not a directory of tiles), reuse it as
    # the canonical CRS source so the merged SHP always has the right WKT.
    inp = common["input_path"]
    if isinstance(inp, str) and inp.lower().endswith((".tif", ".tiff")) and os.path.isfile(inp):
        vis_config["crs_source"] = inp
    print(
        f"[vis] input_dir={vis_config['input_dir']}, vis_dir={vis_config['output_dir']}, "
        f"shp_dir={vis_config['shp_output_dir']}",
        flush=True,
    )
    run_visualization(vis_config)


if __name__ == "__main__":
    main()
