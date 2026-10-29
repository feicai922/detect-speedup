#!/usr/bin/env python
"""Unified large-image inference — config-driven entry point.

Edit the config dicts below, then run:
    python run_demo_config.py
"""


import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
import os
import subprocess
import sys
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
    "input_path": "/data1/wzh26/data/1023测试影像/1/pre.tif",
    "output_dir": "./output/重研-pre/pre_yolo",  # None = 自动生成 ./output/<backend>/<MMDD_HHMM>
    # GPU 设备: ["cuda:0"] / ["cuda:0","cuda:1"]; NPU: ["npu:0","npu:1"]; CPU: ["cpu"]
    "devices": ["cuda:1","cuda:0"],
    "worker_count": None,  # None = 自动等于 len(devices)
    # 切块参数
    "tile_size": 640,
    "stride": 512,
    # Per-GPU batch. Start at 12 for the 16 GB P100; increase only after a
    # stable run, because the other configured GPU has more memory.
    "batch_size": 12,
    # Keep at most four image batches buffered so raising batch_size does not
    # reintroduce the previous CPU-memory pressure.
    "queue_size": 4,
    # 输出控制
    "save_json": True,
    # This source has 3.6M overlapping tiles. Stream detections to disk so
    # inference RAM stays bounded; merge/NMS and SHP export are a post-step.
    "stream_json": True,
    "save_bbox_shp": False,
    "save_mask_shp": False,
    "global_nms": False,
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
ENABLE_VISUALIZATION = True

# Runs only after successful streaming inference. It is a two-pass, bounded-
# memory exporter and deliberately does not apply whole-image NMS.
AUTO_EXPORT_STREAM_SHP = True

# ============================================================
# 3. 模型配置（按所选后端填写对应的）
# ============================================================

# --- YOLO ---
YOLO = {
    "model_path": "/data1/wzh26/重研/code/runs/building_extraction_pre/yolo26x_seg/weights/best.pt",
    "imgsz": 640,
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
    # Keep regularization disabled for streamed huge-image inference unless it
    # was explicitly enabled in COMMON (it requires all masks in memory).
    if common.get("regularize"):
        common["regularize"] = deepcopy(REGULARIZE)
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

    if backend == "yolo" and common.get("stream_json") and AUTO_EXPORT_STREAM_SHP:
        json_dir = os.path.abspath(common["output_dir"])
        json_files = [
            name for name in os.listdir(json_dir)
            if name.lower().endswith(".json") and not name.endswith(".partial")
        ]
        shp_path = os.path.join(base_dir, "shps", "all_mask.shp")
        if len(json_files) == 1:
            command = [
                sys.executable, os.path.join(os.path.dirname(__file__), "stream_json_to_shp.py"),
                os.path.join(json_dir, json_files[0]), "--source", common["input_path"],
                "--output", shp_path, "--geometry", "mask", "--threshold", str(backend_config["conf"]),
                # A large stream can occasionally lose only its terminal `]}` on
                # an interrupted filesystem flush.  The exporter then preserves
                # all complete records but still rejects a partial final record.
                "--recover-truncated-tail",
                "--progress",
            ]
            print(f"[shp] automatically exporting streamed JSON to {shp_path}", flush=True)
            subprocess.run(command, check=True)
        elif len(json_files) > 1:
            print(
                f"[shp] found {len(json_files)} streamed JSON files; "
                "skipping merged SHP export and continuing to visualization",
                flush=True,
            )

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
