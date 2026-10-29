#!/usr/bin/env python
"""大图实例分割推理：RF-DETR seg-medium（GZDA 训练）→ 1023 测试影像 pre.tif

模型: /data1/wzh26/runs/rfdetr_seg_medium_gzda/checkpoint_best_regular.pth
      RFDETRSegMedium, num_classes=1 (building), resolution=640, rfdetr 1.8.0,
      训练集 gzda_simplified（1024 瓦片 ~0.27 m → 缩放到 640 训练）, 30 epoch 中 best（epoch 7）
      val: bbox mAP50-95 0.372 / segm mAP50-95 0.327

输入: /data1/wzh26/data/1023测试影像/1/pre.tif（19795 x 20748, 0.6 m/px, EPSG:4326）
      tile 1024 → resolution 640，与训练时的处理一致（1024 瓦片缩放至 640）

运行（rf-detr 的 venv，必须带 PYTHONPATH 指向 detect-speedup）：
    cd /data1/wzh26/code/detect-speedup && \
      /data1/wzh26/code/rf-detr/.venv/bin/python run_pre1023_rfdetr_gzda.py
"""


import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
import datetime as dt
import os
from copy import deepcopy

COMMON = {
    "input_path": "/data1/wzh26/data/1023测试影像/1/pre.tif",
    "output_dir": "./output/pre1023_rfdetr_seg_medium_gzda",
    # gpu7 上 0/2 号 4090 被别的任务占着，用空闲的 A6000（index 6）
    "devices": [os.environ.get("PRE1023_DEV", "cuda:6")],
    "worker_count": None,
    "reader_workers": 2,
    "tile_size": 1024,
    "stride": 768,
    "batch_size": 2,
    "queue_size": 4,
    "save_json": True,
    "save_bbox_shp": False,
    "save_mask_shp": False,
    "global_nms": True,
    "global_iou": 0.45,
    "classes": [0],
    "input_color": "rgb",
    "fp16": True,
    "regularize": None,
}

RFDETR = {
    "checkpoint": "/data1/wzh26/runs/rfdetr_seg_medium_gzda/checkpoint_best_regular.pth",
    "num_classes": 1,
    "resolution": int(os.environ.get("PRE1023_RES", "640")),
    "conf_threshold": 0.3,
    "iou_threshold": 0.5,
}

ENABLE_VISUALIZATION = True

VIS = {
    "geometry": "mask",
    "mask_fallback_bbox": True,
    "fill_alpha": 80,
    "min_mask_area": 25,
    "show_legend": False,
    "thresholds": [0.2, 0.3, 0.4, 0.5, 0.6],
    "line_width": 2,
    "jpg_quality": 90,
    "recursive": False,
}


def apply_env_overrides():
    if os.environ.get("PRE1023_IN"):
        COMMON["input_path"] = os.environ["PRE1023_IN"]
    if os.environ.get("PRE1023_OUT"):
        COMMON["output_dir"] = os.environ["PRE1023_OUT"]
    if os.environ.get("PRE1023_TILE"):
        COMMON["tile_size"] = int(os.environ["PRE1023_TILE"])
    if os.environ.get("PRE1023_STRIDE"):
        COMMON["stride"] = int(os.environ["PRE1023_STRIDE"])
    if os.environ.get("PRE1023_BATCH"):
        COMMON["batch_size"] = int(os.environ["PRE1023_BATCH"])
    if os.environ.get("PRE1023_CONF"):
        RFDETR["conf_threshold"] = float(os.environ["PRE1023_CONF"])


def main():
    apply_env_overrides()
    from detect_speedup import run_rfdetr

    base_dir = COMMON["output_dir"]
    common = deepcopy(COMMON)
    common["output_dir"] = os.path.join(base_dir, "json")
    if common["worker_count"] is None:
        common["worker_count"] = len(common["devices"])

    print(
        f"[task] backend=rfdetr input={common['input_path']} output={common['output_dir']} "
        f"devices={common['devices']} tile={common['tile_size']} stride={common['stride']} "
        f"resolution={RFDETR['resolution']} conf={RFDETR['conf_threshold']}",
        flush=True,
    )
    t0 = dt.datetime.now()
    run_rfdetr(common=common, rfdetr=deepcopy(RFDETR))
    print(f"[infer] done in {dt.datetime.now() - t0}", flush=True)

    if not ENABLE_VISUALIZATION:
        return
    try:
        from detect_speedup.visualize_predictions import run_visualization
    except ImportError as exc:
        print(f"[vis] skipped: {exc}", flush=True)
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
