"""通用大图滑窗推理循环（插件后端共用）。

职责：开图 → 生成窗口 → 逐批读取瓦片 → 调后端 ``infer()`` → 坐标回填 → global NMS
→ 写 JSON → （可选）合并 Shapefile + 可视化。全部复用 ``yolo_image_io_and_export`` 与
``visualize_predictions``，所以输出格式和原生后端完全一致。
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..yolo_image_io_and_export import (
    append_detection,
    count_windows,
    ensure_device,
    generate_starts,
    global_nms,
    open_reader,
)
from .base import TileBackend

DEFAULTS: Dict[str, Any] = {
    "input_path": None,
    "output_dir": "./output",
    "device": "cuda:0",
    "tile_size": 640,
    "stride": None,          # None → tile_size 的 80%
    "batch_size": 4,
    "conf": 0.3,
    "iou": 0.5,
    "global_nms": True,
    "global_iou": 0.45,
    "save_json": True,
    "visualize": True,
    "shp_geometry": "auto",  # auto: segment 后端用 mask，detect 后端用 bbox
    "vis_thresholds": [0.2, 0.3, 0.4, 0.5, 0.6],
    "jpg_quality": 90,
    "line_width": 2,
    "classes": None,         # None = 全部
}


def _resolve(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None})
    if out["input_path"] is None:
        raise ValueError("runner 需要 input_path")
    if out["stride"] is None:
        out["stride"] = max(1, int(out["tile_size"] * 0.8))
    return out


def run_tiled(backend: TileBackend, config: Dict[str, Any]) -> Path:
    """执行完整的大图推理，返回写出的 JSON 路径。"""
    cfg = _resolve(config)
    t_start = time.time()

    device = ensure_device(cfg["device"]) or cfg["device"]
    backend.load(device)
    print(f"[backend] {backend.describe()} on {device}", flush=True)

    reader = open_reader(cfg["input_path"])
    width, height = reader.width, reader.height
    x_starts = generate_starts(width, cfg["tile_size"], cfg["stride"])
    y_starts = generate_starts(height, cfg["tile_size"], cfg["stride"])
    windows = [
        (x, y, min(x + cfg["tile_size"], width), min(y + cfg["tile_size"], height))
        for y in y_starts
        for x in x_starts
    ]
    expected = count_windows(width, height, cfg["tile_size"], cfg["stride"])
    if len(windows) != expected:  # pragma: no cover - 自检
        raise RuntimeError(f"窗口数 {len(windows)} != 预期 {expected}")

    out_dir = Path(cfg["output_dir"])
    (out_dir / "json").mkdir(parents=True, exist_ok=True)
    print(
        f"[task] {cfg['input_path']} ({width}x{height}) tiles={len(windows)} "
        f"tile={cfg['tile_size']} stride={cfg['stride']} batch={cfg['batch_size']} "
        f"conf={cfg['conf']} -> {out_dir}",
        flush=True,
    )

    names = backend.class_names
    detections = []
    t_infer = 0.0
    bs = int(cfg["batch_size"])
    classes = cfg.get("classes")
    try:
        for start in range(0, len(windows), bs):
            batch_windows = windows[start:start + bs]
            tiles = [reader.read(*w) for w in batch_windows]
            t0 = time.time()
            preds = backend.infer(tiles, batch_windows, device)
            t_infer += time.time() - t0
            if len(preds) != len(batch_windows):
                raise RuntimeError("后端返回的预测条数与瓦片数不一致")
            for pred, window in zip(preds, batch_windows):
                keep = pred.scores >= cfg["conf"]
                if classes is not None:
                    keep &= sum(pred.labels == int(c) for c in classes) > 0
                if not keep.any():
                    continue
                seg = pred.segment_xy
                for i, (box, score, label) in enumerate(
                    zip(pred.boxes[keep], pred.scores[keep], pred.labels[keep])
                ):
                    append_detection(
                        detections,
                        names,
                        window,
                        width,
                        height,
                        box,
                        score,
                        int(label),
                        segment_xy=(seg[i] if seg is not None else None),
                    )
            done = min(start + bs, len(windows))
            if done % 200 == 0 or done == len(windows):
                print(
                    f"[prog] {done}/{len(windows)} tiles, dets={len(detections)}, "
                    f"infer={t_infer:.1f}s ({t_infer / done * 1000:.0f} ms/tile)",
                    flush=True,
                )
    finally:
        reader.close()
        backend.close()

    raw = len(detections)
    merged = global_nms(detections, cfg["global_iou"]) if cfg["global_nms"] else detections
    print(
        f"[nms] {raw} -> {len(merged)} (global_nms={cfg['global_nms']}, iou={cfg['global_iou']})",
        flush=True,
    )

    json_path = out_dir / "json" / (Path(cfg["input_path"]).stem + ".json")
    if cfg["save_json"]:
        doc = {
            "image": str(cfg["input_path"]),
            "backend": backend.name,
            "image_size": {"width": width, "height": height},
            "model": backend.config.get("checkpoint") or backend.config.get("model_dir"),
            "config": backend.config,
            "device": str(device),
            "tile_size": cfg["tile_size"],
            "stride": cfg["stride"],
            "conf": cfg["conf"],
            "iou": cfg["global_iou"],
            "tiles": len(windows),
            "processed_tiles": len(windows),
            "infer_seconds": round(t_infer, 1),
            "raw_detections": raw,
            "merged_detections": len(merged),
            "detections": merged,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        print(
            f"[json] {json_path} ({len(merged)} dets, {json_path.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )

    if cfg["visualize"]:
        from ..visualize_predictions import run_visualization

        geometry = cfg["shp_geometry"]
        if geometry == "auto":
            geometry = "mask" if backend.task == "segment" else "bbox"
        run_visualization(
            {
                "input_dir": str(out_dir / "json"),
                "output_dir": str(out_dir / "vis"),
                "shp_output_dir": str(out_dir / "shps"),
                "image_dir": str(cfg["input_path"]),
                "crs_source": str(cfg["input_path"]),
                "geometry": geometry,
                "mask_fallback_bbox": True,
                "thresholds": cfg["vis_thresholds"],
                "line_width": cfg["line_width"],
                "jpg_quality": cfg["jpg_quality"],
                "recursive": False,
            }
        )

    print(f"[all] total {dt.timedelta(seconds=int(time.time() - t_start))}", flush=True)
    return json_path
