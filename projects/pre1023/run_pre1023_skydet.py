#!/usr/bin/env python
"""大图推理：SKYDET（DINOv3-ConvNeXt + CrossFusedEncoder + DFINETransformer，gzda_v1 训练）→ pre.tif

模型: /data1/wzh26/runs/skydet_gzda_v1_640_bs8/best_stg1.pth
配置: /data1/wzh26/code/SKYDET/configs/skydet/skydet_gzda_v1_640_bs8.yml
      task=detection, num_classes=1, eval_spatial_size=640, 训练集 GZDA v1（1024 瓦片 ~0.27 m）

预处理与训练保持一致：resize 到 640×640 → /255 → DINOV3 Normalize(0.485/0.456/0.406, 0.229/0.224/0.225)
（注意 SKYDET 自带的 tools/inference/torch_inf.py 漏了 normalize，这里补上）

大图切块、global NMS、JSON 落盘、Shapefile 导出、可视化全部复用 detect-speedup 的模块。

运行（gpu7 上的 yolo 环境 + skydet 依赖目录）：
    cd /data1/wzh26/code/detect-speedup && \
      PYTHONPATH=/home/wzh26/skydet_pkgs /home/wzh26/anaconda3/envs/yolo/bin/python \
      run_pre1023_skydet.py
环境变量：PRE_IN / PRE_OUT / PRE_CKPT / PRE_CFG / PRE_TILE / PRE_STRIDE / PRE_CONF / PRE_BATCH / PRE_DEV
"""


import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

SKYDET = "/data1/wzh26/code/SKYDET"
if SKYDET not in sys.path:
    sys.path.insert(0, SKYDET)

from engine.core import YAMLConfig  # noqa: E402

from detect_speedup.yolo_image_io_and_export import (  # noqa: E402
    append_detection,
    count_windows,
    ensure_device,
    generate_windows,
    global_nms,
    open_reader,
)

CFG = {
    "cfg": "/data1/wzh26/code/SKYDET/configs/skydet/skydet_gzda_v1_640_bs8.yml",
    "ckpt": "/data1/wzh26/runs/skydet_gzda_v1_640_bs8/best_stg1.pth",
    "input_path": "/data1/wzh26/data/1023测试影像/1/pre.tif",
    "output_dir": "./output/pre1023_skydet_gzda_v1",
    "device": "cuda:0",
    "tile_size": 640,
    "stride": 512,
    "batch_size": 4,
    "conf": 0.3,
    "iou": 0.5,
    "classes": {0: "building"},
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def env_overrides():
    m = {
        "PRE_IN": ("input_path", str),
        "PRE_OUT": ("output_dir", str),
        "PRE_CKPT": ("ckpt", str),
        "PRE_CFG": ("cfg", str),
        "PRE_DEV": ("device", str),
        "PRE_TILE": ("tile_size", int),
        "PRE_STRIDE": ("stride", int),
        "PRE_BATCH": ("batch_size", int),
        "PRE_CONF": ("conf", float),
    }
    for key, (name, caster) in m.items():
        if os.environ.get(key):
            CFG[name] = caster(os.environ[key])


def build_model(cfg_path, ckpt_path, device, size):
    t0 = time.time()
    cfg = YAMLConfig(cfg_path, resume=ckpt_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    info = {k: ckpt.get(k) for k in ("epoch", "last_epoch") if k in ckpt}
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    model = Model().to(device).eval()
    print(f"[model] loaded in {time.time() - t0:.1f}s from {ckpt_path} {info}", flush=True)
    return model


def preprocess(tiles, size):
    """list of HxWx3 uint8 → NCHW float32, 与训练一致的 resize + /255 + DINOV3 normalize。"""
    batch = np.empty((len(tiles), 3, size, size), dtype=np.float32)
    for i, tile in enumerate(tiles):
        im = Image.fromarray(tile).resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        batch[i] = (arr - MEAN) / STD
    return batch


def main():
    env_overrides()
    t_start = time.time()
    device = ensure_device(CFG["device"]) or CFG["device"]
    size = 640
    model = build_model(CFG["cfg"], CFG["ckpt"], device, size)

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
    with torch.no_grad():
        for start in range(0, len(windows), bs):
            bw = windows[start:start + bs]
            tiles = [reader.read(*w) for w in bw]
            x = torch.from_numpy(preprocess(tiles, size)).to(device, non_blocking=True)
            orig = torch.tensor([[w[3] - w[1], w[2] - w[0]] for w in bw], device=device)
            t0 = time.time()
            labels, boxes, scores = model(x, orig)
            t_infer += time.time() - t0
            for lab, box, scr, window in zip(labels, boxes, scores, bw):
                lab = lab.detach().cpu().numpy().astype(int)
                box = box.detach().cpu().numpy()
                scr = scr.detach().cpu().numpy()
                keep = scr >= CFG["conf"]
                for l, b, s in zip(lab[keep], box[keep], scr[keep]):
                    append_detection(detections, CFG["classes"], window, width, height, b, s, l)
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
        "backend": "skydet",
        "image_size": {"width": width, "height": height},
        "model": CFG["ckpt"],
        "config": CFG["cfg"],
        "model_name": "SKYDET (DINOv3-ConvNeXt + CrossFusedEncoder + DFINETransformer)",
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
