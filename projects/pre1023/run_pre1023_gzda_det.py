#!/usr/bin/env python
"""大图推理：GZDA 训练的 yolo26x-det 权重 → 1023 测试影像 pre.tif

模型: /data1/wzh26/runs/yolo26_building_det/gzda_simplified_det_yolo26x_1024/weights/best.pt
      （gzda_simplified 数据集, task=detect, 1 类 building, 150 epoch, imgsz=1024）

输入: /data1/wzh26/data/1023测试影像/1/pre.tif
      19795 x 20748, 3 band uint8, EPSG:4326, ~0.6 m/px

输出: ./output/pre1023_gzda_det/
      json/pre.json        检测结果（含像素坐标 + 经纬度 bbox）
      shps/*.shp           合并后的 Shapefile（按 conf 阈值分档）
      vis/*_vis.jpg        可视化

运行（在 gpu7 上，两台 4090 空闲）：
    cd /data1/wzh26/code/detect-speedup && \
      /home/wzh26/anaconda3/envs/yolo/bin/python run_pre1023_gzda_det.py
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

# ============================================================
# 配置
# ============================================================
COMMON = {
    "input_path": "/data1/wzh26/data/1023测试影像/1/pre.tif",
    "output_dir": "./output/pre1023_gzda_det",
    # gpu7 上 cuda:0 / cuda:2 是两张空闲的 4090（cuda:1/3 是 A6000，1 号正在训练）
    "devices": ["cuda:0", "cuda:2"],
    "worker_count": None,   # = len(devices)
    "reader_workers": 1,
    # 切块：训练时是 1024x1024 瓦片，推理保持 tile=imgsz=1024，stride 留 12.5% 重叠
    "tile_size": 1024,
    "stride": 896,
    "batch_size": 8,
    "queue_size": 4,
    "save_json": True,
    "stream_json": False,      # 552 块，可以全内存做 global NMS
    "save_bbox_shp": False,    # 合并 SHP 由 visualize_predictions 统一输出
    "save_mask_shp": False,    # det 模型没有 mask
    "global_nms": True,
    "global_iou": 0.45,
    "recursive": False,
    "skip_if_output_exists": False,
    "regularize": None,
}

YOLO = {
    "model_path": "/data1/wzh26/runs/yolo26_building_det/gzda_simplified_det_yolo26x_1024/weights/best.pt",
    "imgsz": 1024,
    "conf": 0.25,
    "iou": 0.45,
    "max_det": 300,
    "agnostic_nms": False,
}

ENABLE_VISUALIZATION = True

VIS = {
    "geometry": "bbox",
    "mask_fallback_bbox": True,
    "fill_alpha": 0,
    "min_mask_area": 0,
    "show_legend": False,
    "thresholds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "line_width": 2,
    "jpg_quality": 90,
    "recursive": False,
}


def apply_env_overrides():
    """允许用环境变量跑不同切块尺度，例如：
    PRE1023_OUT=./output/pre1023_gzda_det_scale2 PRE1023_TILE=512 PRE1023_STRIDE=448
    （tile 512 + imgsz 1024 会把 0.6 m 影像放大 2 倍，等效训练时的 ~0.3 m 尺度）
    """
    if os.environ.get("PRE1023_OUT"):
        COMMON["output_dir"] = os.environ["PRE1023_OUT"]
    if os.environ.get("PRE1023_TILE"):
        COMMON["tile_size"] = int(os.environ["PRE1023_TILE"])
    if os.environ.get("PRE1023_STRIDE"):
        COMMON["stride"] = int(os.environ["PRE1023_STRIDE"])
    if os.environ.get("PRE1023_BATCH"):
        COMMON["batch_size"] = int(os.environ["PRE1023_BATCH"])
    if os.environ.get("PRE1023_DEVICES"):
        COMMON["devices"] = [d for d in os.environ["PRE1023_DEVICES"].split(",") if d]
    if os.environ.get("PRE1023_CONF"):
        YOLO["conf"] = float(os.environ["PRE1023_CONF"])


def main():
    apply_env_overrides()
    base_dir = COMMON["output_dir"]
    common = deepcopy(COMMON)
    common["output_dir"] = os.path.join(base_dir, "json")
    if common["worker_count"] is None:
        common["worker_count"] = len(common["devices"])

    print(
        f"[task] backend=yolo input={common['input_path']} output={common['output_dir']} "
        f"devices={common['devices']} tile={common['tile_size']} stride={common['stride']} "
        f"imgsz={YOLO['imgsz']} conf={YOLO['conf']}",
        flush=True,
    )
    t0 = dt.datetime.now()
    run_yolo(common=common, yolo=deepcopy(YOLO))
    print(f"[infer] done in {dt.datetime.now() - t0}", flush=True)

    if not ENABLE_VISUALIZATION:
        return
    if run_visualization is None:
        print("[vis] skipped — visualization unavailable", flush=True)
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
