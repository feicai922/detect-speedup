#!/usr/bin/env python
"""大图推理：HuggingFace RT-DETR（卫星影像屋顶检测）→ 1023 测试影像 pre.tif

模型: /data1/wzh26/重研/code/runs/down-rt-detr
      来源 https://huggingface.co/Yifeng-Liu/rt-detr-finetuned-for-satellite-image-roofs-detection
      transformers 格式（RTDetrForObjectDetection，backbone rt-detr-r101vd，单类，输入 640×640）

说明: detect-speedup 的后端只覆盖 yolo / rtmdet / rfdetr，HF transformers 的 RT-DETR 不在其中，
      所以这里复用框架的**滑窗读取、批处理、global NMS、JSON 落盘、Shapefile 导出与可视化**等
      全部工具（detect_speedup.yolo_image_io_and_export / visualize_predictions），
      只把"前向推理"换成 transformers 的 RTDetrForObjectDetection。

用法（gpu7 上的 rtdetr conda 环境）：
    cd /data1/wzh26/code/detect-speedup && \
      /home/wzh26/anaconda3/envs/rtdetr/bin/python run_pre1023_rtdetr_hf.py
环境变量覆盖：PRE_IN / PRE_OUT / PRE_TILE / PRE_STRIDE / PRE_CONF / PRE_BATCH / PRE_DEV / PRE_MODEL
"""


import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
import datetime as dt
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from detect_speedup.yolo_image_io_and_export import (
    append_detection,
    count_windows,
    ensure_device,
    generate_windows,
    global_nms,
    open_reader,
)

CFG = {
    "model_dir": "/data1/wzh26/重研/code/runs/down-rt-detr",
    "input_path": "/data1/wzh26/data/1023测试影像/1/pre.tif",
    "output_dir": "./output/pre1023_rtdetr_hf",
    "device": "cuda:0",
    "tile_size": 640,
    "stride": 512,
    "batch_size": 4,
    "conf": 0.3,
    "iou": 0.5,
    "classes": {0: "roof"},
}


def env_overrides():
    m = {
        "PRE_MODEL": ("model_dir", str),
        "PRE_IN": ("input_path", str),
        "PRE_OUT": ("output_dir", str),
        "PRE_DEV": ("device", str),
        "PRE_TILE": ("tile_size", int),
        "PRE_STRIDE": ("stride", int),
        "PRE_BATCH": ("batch_size", int),
        "PRE_CONF": ("conf", float),
    }
    for key, (name, caster) in m.items():
        if os.environ.get(key):
            CFG[name] = caster(os.environ[key])


def load_model(model_dir, device):
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    t0 = time.time()
    processor = RTDetrImageProcessor.from_pretrained(model_dir)
    model = RTDetrForObjectDetection.from_pretrained(model_dir).to(device).eval()
    print(f"[model] loaded in {time.time() - t0:.1f}s from {model_dir}", flush=True)
    return processor, model


def main():
    env_overrides()
    t_start = time.time()
    device = ensure_device(CFG["device"]) or CFG["device"]
    processor, model = load_model(CFG["model_dir"], device)

    reader = open_reader(CFG["input_path"])
    width, height = reader.width, reader.height
    windows = list(generate_windows(width, height, CFG["tile_size"], CFG["stride"]))
    out_dir = Path(CFG["output_dir"])
    (out_dir / "json").mkdir(parents=True, exist_ok=True)
    print(
        f"[task] input={CFG['input_path']} ({width}x{height}) tiles={len(windows)} "
        f"(expect {count_windows(width, height, CFG['tile_size'], CFG['stride'])}) "
        f"tile={CFG['tile_size']} stride={CFG['stride']} conf={CFG['conf']} device={device}",
        flush=True,
    )

    detections = []
    t_infer = 0.0
    bs = CFG["batch_size"]
    for start in range(0, len(windows), bs):
        batch_windows = windows[start:start + bs]
        images = [reader.read(*w) for w in batch_windows]
        t0 = time.time()
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = [(w[3] - w[1], w[2] - w[0]) for w in batch_windows]
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=CFG["conf"]
        )
        t_infer += time.time() - t0
        for result, window in zip(results, batch_windows):
            boxes = result["boxes"].detach().cpu().numpy()
            scores = result["scores"].detach().cpu().numpy()
            labels = result["labels"].detach().cpu().numpy().astype(int)
            for box, score, label in zip(boxes, scores, labels):
                append_detection(
                    detections, CFG["classes"], window, width, height, box, score, label
                )
        done = min(start + bs, len(windows))
        if done % 200 == 0 or done == len(windows):
            print(
                f"[prog] {done}/{len(windows)} tiles, dets={len(detections)}, "
                f"infer={t_infer:.1f}s ({t_infer / done * 1000:.0f} ms/tile)",
                flush=True,
            )
    reader.close()

    raw = len(detections)
    merged = global_nms(detections, CFG["iou"])
    print(f"[nms] {raw} -> {len(merged)} (iou={CFG['iou']})", flush=True)

    doc = {
        "image": CFG["input_path"],
        "backend": "huggingface-rtdetr",
        "image_size": {"width": width, "height": height},
        "model": CFG["model_dir"],
        "model_name": "RTDetrForObjectDetection (r101vd)",
        "device": str(device),
        "tile_size": CFG["tile_size"],
        "stride": CFG["stride"],
        "conf": CFG["conf"],
        "iou": CFG["iou"],
        "tiles": len(windows),
        "processed_tiles": len(windows),
        "infer_seconds": round(t_infer, 1),
        "raw_detections": raw,
        "merged_detections": len(merged),
        "detections": merged,
    }
    json_path = out_dir / "json" / (Path(CFG["input_path"]).stem + ".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    print(f"[json] {json_path} ({len(merged)} dets, {json_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # 合并 Shapefile + 可视化：复用框架的 visualize_predictions
    from detect_speedup.visualize_predictions import run_visualization

    run_visualization({
        "input_dir": str(out_dir / "json"),
        "output_dir": str(out_dir / "vis"),
        "shp_output_dir": str(out_dir / "shps"),
        "image_dir": CFG["input_path"],
        "crs_source": CFG["input_path"],
        "geometry": "bbox",
        "mask_fallback_bbox": True,
        "thresholds": [0.2, 0.3, 0.4, 0.5, 0.6],
        "line_width": 2,
        "jpg_quality": 90,
        "recursive": False,
    })
    print(f"[all] total {dt.timedelta(seconds=int(time.time() - t_start))}", flush=True)


if __name__ == "__main__":
    main()
