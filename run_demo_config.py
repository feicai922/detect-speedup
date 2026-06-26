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
BACKEND = "yolo"

# ============================================================
# 2. 通用配置
# ============================================================
COMMON = {
    "input_path": "/path/to/your/image_or_dir",
    "output_dir": None,  # None = 自动生成 ./output/<backend>/<MMDD_HHMM>
    # GPU 设备: ["cuda:0"] / ["cuda:0","cuda:1"]; NPU: ["npu:0","npu:1"]; CPU: ["cpu"]
    "devices": ["cuda:0"],
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
}

# ============================================================
# 3. 模型配置（按所选后端填写对应的）
# ============================================================

# --- YOLO ---
YOLO = {
    "model_path": "/path/to/best.pt",
    "imgsz": 1024,
    "conf": 0.3,
    "iou": 0.4,
    "max_det": 300,
    "agnostic_nms": False,
}

# --- RTMDet / MMDetection ---
RTMDET = {
    "config_file": "/path/to/rtmdet_config.py",
    "checkpoint_file": "/path/to/epoch_100.pth",
    "score_thr": 0.4,
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
    if run_visualization is None:
        print("[vis] skipped — PIL not installed", flush=True)
        return

    vis_config = deepcopy(VIS)
    vis_config["input_dir"] = common["output_dir"]
    vis_config["output_dir"] = os.path.join(base_dir, "vis")
    vis_config["image_dir"] = common["input_path"]
    print(
        f"[vis] input_dir={vis_config['input_dir']}, output_dir={vis_config['output_dir']}",
        flush=True,
    )
    run_visualization(vis_config)


if __name__ == "__main__":
    main()
