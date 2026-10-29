#!/usr/bin/env python
"""detect-speedup 统一入口 —— 大图推理（检测 / 实例分割）

所有后端共用一个 CLI，新增后端只需在 ``detect_speedup/backends/`` 里加一个插件
（见 ``detect_speedup/backends/_template.py``）。

常用示例
--------
# 看有哪些后端
python run_large.py --list-backends

# 原生后端（yolo / rtmdet / rfdetr）：走框架原有的多进程多卡流水线
python run_large.py --backend yolo \
    --model /path/to/best.pt --input /path/to/pre.tif --out ./output/yolo_run \
    --tile 1024 --imgsz 640 --conf 0.3 --batch 8 --device cuda:0

# 插件后端（rtdetr-hf / skydet）：走通用滑窗循环
python run_large.py --backend skydet \
    --model /path/to/best_stg1.pth \
    --backend-config /data1/wzh26/code/SKYDET/configs/skydet/skydet_gzda_v1_640_bs8.yml \
    --input /path/to/pre.tif --out ./output/skydet_run --tile 640 --stride 512

python run_large.py --backend rtdetr-hf \
    --model /path/to/hf_model_dir --input /path/to/pre.tif --out ./output/rtdetr_run

# 全部参数也可写进 JSON/YAML 配置文件
python run_large.py -c my_run.yml

环境变量
--------
``DS_BACKEND / DS_INPUT / DS_OUT / DS_MODEL / DS_DEVICE`` 可覆盖对应参数，
方便在 tmux/脚本里少写一长串。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULTS = {
    "backend": "yolo",
    "input": None,
    "out": "./output/run",
    "model": None,
    "backend_config": None,
    "tile": None,
    "stride": None,
    "batch": None,
    "conf": None,
    "iou": None,
    "imgsz": None,
    "device": "cuda:0",
    "workers": None,
    "classes": None,
    "geometry": "auto",
    "no_vis": False,
    "no_json": False,
    "fp16": None,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="detect-speedup 统一大图推理入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--list-backends", action="store_true", help="列出所有可用后端后退出")
    p.add_argument("-c", "--config", help="JSON/YAML 配置文件（键同下面的参数）")
    p.add_argument("--backend", help="后端名，见 --list-backends")
    p.add_argument("-i", "--input", help="输入影像或目录（.tif/.jpg/...）")
    p.add_argument("-o", "--out", help="输出目录（下面会建 json/ shps/ vis/）")
    p.add_argument("-m", "--model", help="权重路径（yolo/rfdetr: .pt/.pth；rtdetr-hf: HF 模型目录）")
    p.add_argument("--backend-config", help="后端配置文件（skydet 的 yml、rtmdet 的 py）")
    p.add_argument("--tile", type=int, help="切块边长（默认按后端：yolo 1024 / 插件 640）")
    p.add_argument("--stride", type=int, help="切块步长（默认 tile*0.8）")
    p.add_argument("--batch", type=int, help="每个设备的 batch")
    p.add_argument("--conf", type=float, help="置信度阈值（默认 0.3）")
    p.add_argument("--iou", type=float, help="NMS IoU（默认 0.5）")
    p.add_argument("--imgsz", type=int, help="模型输入尺寸（yolo/rfdetr/插件后端的 size）")
    p.add_argument("-d", "--device", help="cuda:0 / cuda:0,cuda:2 / cpu")
    p.add_argument("--workers", type=int, help="原生后端进程数（默认 = 设备数）")
    p.add_argument("--classes", help="只保留这些类别 id，逗号分隔，如 0")
    p.add_argument("--geometry", choices=["auto", "bbox", "mask"], help="SHP/可视化几何类型")
    p.add_argument("--no-vis", action="store_true", help="不做可视化与合并 SHP")
    p.add_argument("--no-json", action="store_true", help="不写 JSON（一般不用）")
    p.add_argument("--fp16", action="store_true", help="原生后端用半精度")
    p.add_argument("--set", action="append", default=[], metavar="K=V",
                   help="额外传给后端的参数（可重复），值会自动转 int/float/bool")
    return p


def _cast(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value


def load_file_config(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yml", ".yaml")):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(f"读取 {path} 需要 PyYAML：{exc}")
        return yaml.safe_load(text) or {}
    return json.loads(text)


def resolve(args) -> dict:
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(load_file_config(args.config))
    for key in ("backend", "input", "out", "model", "backend_config", "tile", "stride",
                "batch", "conf", "iou", "imgsz", "device", "workers", "classes", "geometry"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if args.no_vis:
        cfg["no_vis"] = True
    if args.no_json:
        cfg["no_json"] = True
    if args.fp16:
        cfg["fp16"] = True

    env_map = {
        "DS_BACKEND": "backend",
        "DS_INPUT": "input",
        "DS_OUT": "out",
        "DS_MODEL": "model",
        "DS_DEVICE": "device",
        "DS_TILE": "tile",
        "DS_STRIDE": "stride",
        "DS_BATCH": "batch",
        "DS_CONF": "conf",
    }
    for env_key, name in env_map.items():
        if os.environ.get(env_key):
            cfg[name] = os.environ[env_key]
    for key in ("tile", "stride", "batch", "imgsz"):
        if isinstance(cfg.get(key), str) and cfg[key]:
            cfg[key] = int(cfg[key])
    if isinstance(cfg.get("conf"), str) and cfg["conf"]:
        cfg["conf"] = float(cfg["conf"])

    extras = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set 需要 K=V 形式，收到 {item!r}")
        key, value = item.split("=", 1)
        extras[key.strip()] = _cast(value.strip())
    cfg["extras"] = extras

    if not cfg.get("input"):
        raise SystemExit("必须指定 --input（或用 DS_INPUT / -c 配置文件）")
    cfg["classes"] = (
        [int(c) for c in str(cfg["classes"]).split(",") if str(c).strip()]
        if cfg.get("classes") not in (None, "")
        else None
    )
    cfg["devices"] = [d.strip() for d in str(cfg["device"]).split(",") if d.strip()]
    return cfg


def run_native(cfg: dict) -> None:
    from detect_speedup import run_rfdetr, run_rtmdet, run_yolo
    from detect_speedup.backends import normalize_native

    backend = normalize_native(cfg["backend"])
    out_dir = Path(cfg["out"])
    common = {
        "input_path": cfg["input"],
        "output_dir": str(out_dir / "json"),
        "devices": cfg["devices"],
        "worker_count": cfg.get("workers") or len(cfg["devices"]),
        "tile_size": cfg.get("tile") or 1024,
        "stride": cfg.get("stride"),
        "batch_size": cfg.get("batch") or 4,
        "save_json": not cfg["no_json"],
        "save_bbox_shp": False,
        "save_mask_shp": False,
        "global_nms": True,
        "global_iou": cfg.get("iou") or 0.45,
        "recursive": False,
        "classes": cfg["classes"],
        "fp16": bool(cfg.get("fp16")),
        "regularize": None,
    }
    conf = cfg.get("conf") or 0.3
    size = cfg.get("imgsz")

    if backend == "yolo":
        backend_cfg = {
            "model_path": cfg["model"] or "",
            "imgsz": size or 640,
            "conf": conf,
            "iou": cfg.get("iou") or 0.45,
            "max_det": 300,
            "agnostic_nms": False,
        }
        backend_cfg.update(cfg["extras"])
        run_yolo(common=common, yolo=backend_cfg)
    elif backend == "rtmdet":
        backend_cfg = {
            "config_file": cfg.get("backend_config") or "",
            "checkpoint_file": cfg["model"] or "",
            "score_thr": conf,
            "fp16": bool(cfg.get("fp16")),
            "save_mask_overlay": False,
        }
        backend_cfg.update(cfg["extras"])
        run_rtmdet(common=common, rtmdet=backend_cfg)
    elif backend == "rfdetr":
        backend_cfg = {
            "checkpoint": cfg["model"] or "",
            "num_classes": int(cfg["extras"].get("num_classes", 1)),
            "resolution": size or 640,
            "conf_threshold": conf,
            "iou_threshold": cfg.get("iou") or 0.5,
        }
        backend_cfg.update({k: v for k, v in cfg["extras"].items() if k != "num_classes"})
        run_rfdetr(common=common, rfdetr=backend_cfg)
    else:  # pragma: no cover
        raise SystemExit(f"不支持的原生后端 {backend}")

    if cfg["no_vis"]:
        return
    _visualize(cfg, out_dir)


def _visualize(cfg: dict, out_dir: Path) -> None:
    from detect_speedup.visualize_predictions import run_visualization

    geometry = cfg.get("geometry", "auto")
    if geometry == "auto":
        geometry = "bbox"
    run_visualization({
        "input_dir": str(out_dir / "json"),
        "output_dir": str(out_dir / "vis"),
        "shp_output_dir": str(out_dir / "shps"),
        "image_dir": cfg["input"],
        "crs_source": cfg["input"] if str(cfg["input"]).lower().endswith((".tif", ".tiff")) else None,
        "geometry": geometry,
        "mask_fallback_bbox": True,
        "thresholds": [0.2, 0.3, 0.4, 0.5, 0.6],
        "line_width": 2,
        "jpg_quality": 90,
        "recursive": False,
    })


def run_plugin(cfg: dict) -> None:
    from detect_speedup.backends import get_backend
    from detect_speedup.backends.runner import run_tiled

    backend_cfg = {}
    if cfg.get("model"):
        backend_cfg["checkpoint"] = cfg["model"]
        backend_cfg["model_dir"] = cfg["model"]
    if cfg.get("backend_config"):
        backend_cfg["config"] = cfg["backend_config"]
    if cfg.get("imgsz"):
        backend_cfg["size"] = cfg["imgsz"]
    backend_cfg.update(cfg["extras"])

    backend = get_backend(cfg["backend"], backend_cfg)
    run_tiled(
        backend,
        {
            "input_path": cfg["input"],
            "output_dir": cfg["out"],
            "device": cfg["devices"][0],
            "tile_size": cfg.get("tile") or backend.defaults.get("size", 640),
            "stride": cfg.get("stride"),
            "batch_size": cfg.get("batch") or 4,
            "conf": cfg.get("conf") or 0.3,
            "global_iou": cfg.get("iou") or 0.5,
            "save_json": not cfg["no_json"],
            "visualize": not cfg["no_vis"],
            "shp_geometry": cfg.get("geometry", "auto"),
            "classes": cfg["classes"],
        },
    )


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    from detect_speedup.backends import all_backends, is_native, plugin_names, NATIVE_BACKENDS

    if args.list_backends:
        print("原生后端（多进程多卡流水线）:", ", ".join(NATIVE_BACKENDS))
        print("插件后端（通用滑窗循环）:", ", ".join(plugin_names()) or "(无)")
        return

    cfg = resolve(args)
    backend = str(cfg["backend"]).lower()
    print(f"[run_large] backend={backend} input={cfg['input']} out={cfg['out']}", flush=True)
    if is_native(backend):
        run_native(cfg)
    else:
        run_plugin(cfg)


if __name__ == "__main__":
    main()
