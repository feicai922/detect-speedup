#!/usr/bin/env python
"""Unified large-image inference — YOLO / RTMDet / RF-DETR.

Examples:
    python run_demo.py --yolo  --model best.pt  --input /path/to/images
    python run_demo.py --rtmdet --config xx.py --checkpoint xx.pth --input /path/to/images
    python run_demo.py --rfdetr --model checkpoint.pth --input /path/to/images --output ./out
    python run_demo.py --yolo --model best.pt --input /path/to/images --devices cuda:0,cuda:1
"""


import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
import argparse
import os
import datetime as dt
from copy import deepcopy

from detect_speedup import run_inference, run_rfdetr, run_rtmdet, run_yolo

try:
    from detect_speedup.visualize_predictions import run_visualization
except ImportError as e:
    print(f"[warn] visualization import failed: {e}", flush=True)
    run_visualization = None


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified large-image inference — YOLO / RTMDet / RF-DETR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_demo.py --model best.pt --input ./images
  python run_demo.py --rtmdet --config cfg.py --checkpoint ckpt.pth --input ./images
  python run_demo.py --rfdetr --model ckpt.pth --input ./images
  python run_demo.py --model best.pt --input ./images --devices cuda:0,cuda:1 --conf 0.5
        """,
    )

    # ---- Backend (default: yolo) ----
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--yolo", action="store_true", help="Use YOLO backend (default)")
    backend_group.add_argument("--rtmdet", action="store_true", help="Use RTMDet / MMDetection backend")
    backend_group.add_argument("--rfdetr", action="store_true", help="Use RF-DETR backend")

    # ---- I/O ----
    parser.add_argument("--input", default="./input", help="Input image file or directory (default: ./input)")
    parser.add_argument("--output", default=None, help="Output directory (default: ./output/<backend>/<MMDD_HHMM>)")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory")

    # ---- Device ----
    parser.add_argument("--devices", default="cuda:0",
                        help="Devices, comma-separated (default: cuda:0). "
                             "Examples: cuda:0 / cuda:0,cuda:1 / npu:0,npu:1 / cpu")

    # ---- Shared model params ----
    parser.add_argument("--model", help="Model/checkpoint path (YOLO .pt, RF-DETR .pth)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--tile-size", type=int, default=1280, help="Tile size (default: 1280)")
    parser.add_argument("--stride", type=int, default=1024, help="Tile stride (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--workers", type=int, default=None, help="Device worker count (default: len(devices))")

    # ---- RTMDet-specific ----
    parser.add_argument("--config", help="[RTMDet] MMDetection config file path")
    parser.add_argument("--checkpoint", help="[RTMDet] Checkpoint file path")

    # ---- RF-DETR-specific ----
    parser.add_argument("--num-classes", type=int, default=1, help="[RF-DETR] Number of classes (default: 1)")
    parser.add_argument("--resolution", type=int, default=1088, help="[RF-DETR] Input resolution (default: 1088)")

    # ---- YOLO-specific ----
    parser.add_argument("--imgsz", type=int, default=1024, help="[YOLO] Inference image size (default: 1024)")
    parser.add_argument("--iou", type=float, default=0.4, help="[YOLO] IoU threshold (default: 0.4)")
    parser.add_argument("--max-det", type=int, default=300, help="[YOLO] Max detections per tile (default: 300)")

    # ---- Visualization ----
    parser.add_argument("--no-vis", action="store_true", help="Skip visualization after inference")

    return parser.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    # Resolve backend (default: yolo)
    if args.rtmdet:
        backend = "rtmdet"
    elif args.rfdetr:
        backend = "rfdetr"
    else:
        backend = "yolo"

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    # Default output: ./output/<backend>/<MMDD_HHMM>/{json,vis}/
    if args.output is None:
        args.output = os.path.join(".", "output", backend, dt.datetime.now().strftime("%m%d_%H%M"))
    args.output = os.path.join(args.output, "json")  # JSON/SHP 写入 json/ 子目录

    # --- Common config ---
    common_config = {
        "input_path": args.input,
        "output_dir": args.output,  # JSON + SHP 写入 json/ 子目录
        "output_name": None,
        "devices": devices,
        "worker_count": args.workers or len(devices),
        "reader_workers": 2,
        "mp_start": "spawn",
        "tile_size": args.tile_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "queue_size": 16,
        "classes": None,
        "small_image_threshold": 12000,
        "prefer_single_process_for_small": True,
        "input_color": "bgr",
        "save_json": True,
        "save_bbox_shp": True,
        "save_mask_shp": True,
        "global_nms": True,
        "global_iou": 0.45,
        "recursive": args.recursive,
        "skip_if_output_exists": False,
    }

    print(
        f"[task] backend={backend}, input={common_config['input_path']}, "
        f"output={common_config['output_dir']}, devices={common_config['devices']}",
        flush=True,
    )

    # --- Backend-specific config & dispatch ---
    if backend == "yolo":
        yolo_config = {
            "model_path": args.model or "best.pt",
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "agnostic_nms": False,
        }
        run_yolo(common=common_config, yolo=yolo_config)

    elif backend == "rtmdet":
        if not args.config or not args.checkpoint:
            raise SystemExit("--rtmdet requires --config and --checkpoint")
        rtmdet_config = {
            "config_file": args.config,
            "checkpoint_file": args.checkpoint,
            "score_thr": args.conf,
            "fp16": False,
            "save_mask_overlay": False,
        }
        run_rtmdet(common=common_config, rtmdet=rtmdet_config)

    elif backend == "rfdetr":
        rfdetr_config = {
            "checkpoint": args.model or "checkpoint.pth",
            "num_classes": args.num_classes,
            "resolution": args.resolution,
            "conf_threshold": args.conf,
            "iou_threshold": 0.5,
        }
        run_rfdetr(common=common_config, rfdetr=rfdetr_config)

    else:
        run_inference(backend, common=common_config)

    # --- Visualization ---
    if args.no_vis:
        print("[vis] skipped (--no-vis)", flush=True)
    elif run_visualization is None:
        print("[vis] skipped — PIL not installed", flush=True)
    else:
        # vis 目录: 和 json/ 同级，例如 ./output/yolo/0622_1631/vis/
        vis_output_dir = os.path.join(os.path.dirname(common_config["output_dir"].rstrip("/")), "vis")
        vis_config = {
            "input_dir": common_config["output_dir"],
            "output_dir": vis_output_dir,
            "image_dir": common_config["input_path"],
            "geometry": "mask",  # instance segmentation
            "mask_fallback_bbox": True,
            "thresholds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "fill_alpha": 80,
            "min_mask_area": 25,
            "show_legend": False,
            "line_width": 2,
            "jpg_quality": 92,
            "recursive": False,
        }
        print(
            f"[vis] input_dir={vis_config['input_dir']}, output_dir={vis_config['output_dir']}",
            flush=True,
        )
        run_visualization(vis_config)


if __name__ == "__main__":
    main()
