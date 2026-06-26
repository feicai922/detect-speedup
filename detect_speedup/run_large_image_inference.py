from . import runtime_warning_filters  # noqa: F401
from copy import deepcopy

from .batch_multi_device_runner import run_inputs
from .model_backend_adapters import MMDetInferenceAdapter, RFDETRInferenceAdapter, YOLOInferenceAdapter

BACKENDS = {
    "yolo": YOLOInferenceAdapter,
    "rtmdet": MMDetInferenceAdapter,
    "mmdet": MMDetInferenceAdapter,
    "rfdetr": RFDETRInferenceAdapter,
    "rf-detr": RFDETRInferenceAdapter,
}

COMMON_DEFAULTS = {
    "input_path": "/path/to/image_or_dir",
    "output_dir": "./out",
    "output_name": None,
    "devices": ["cpu"],
    "worker_count": None,
    "reader_workers": 1,
    "mp_start": "spawn",
    "tile_size": 1280,
    "stride": None,
    "batch_size": 4,
    "queue_size": 8,
    "classes": None,
    "small_image_threshold": 12000,
    "prefer_single_process_for_small": True,
    "save_json": True,
    "save_bbox_shp": True,
    "save_mask_shp": True,
    "global_nms": False,
    "global_iou": 0.45,
    "recursive": False,
    "supported_suffixes": {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"},
    "skip_if_output_exists": False,
    "input_color": "rgb",
}

YOLO_DEFAULTS = {
    "model_path": "best.pt",
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.45,
    "max_det": 300,
    "agnostic_nms": False,
}

RTMDET_DEFAULTS = {
    "config_file": "/path/to/config.py",
    "checkpoint_file": "/path/to/checkpoint.pth",
    "score_thr": 0.3,
    "fp16": False,
    "save_mask_overlay": False,
}

RFDETR_DEFAULTS = {
    "checkpoint": "/path/to/checkpoint.pth",
    "num_classes": 1,
    "resolution": 640,
    "conf_threshold": 0.3,
    "iou_threshold": 0.5,
}

BACKEND_DEFAULTS = {
    "yolo": YOLO_DEFAULTS,
    "rtmdet": RTMDET_DEFAULTS,
    "mmdet": RTMDET_DEFAULTS,
    "rfdetr": RFDETR_DEFAULTS,
    "rf-detr": RFDETR_DEFAULTS,
}


def merge_config(common=None, backend=None, backend_config=None, **overrides):
    backend = (backend or "yolo").lower()
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}. Choose one of {sorted(BACKENDS)}")

    config = deepcopy(COMMON_DEFAULTS)
    config.update(deepcopy(BACKEND_DEFAULTS[backend]))
    if common:
        config.update(common)
    if backend_config:
        config.update(backend_config)
    if overrides:
        config.update(overrides)
    return config


def build_adapter(backend, config):
    backend = backend.lower()
    try:
        adapter_cls = BACKENDS[backend]
    except KeyError as exc:
        raise ValueError(f"Unsupported backend: {backend}. Choose one of {sorted(BACKENDS)}") from exc
    return adapter_cls(config)


def run_inference(backend="yolo", common=None, backend_config=None, **overrides):
    config = merge_config(common=common, backend=backend, backend_config=backend_config, **overrides)
    run_inputs(build_adapter(backend, config))


def run_yolo(common=None, yolo=None, **overrides):
    run_inference("yolo", common=common, backend_config=yolo, **overrides)


def run_rtmdet(common=None, rtmdet=None, **overrides):
    run_inference("rtmdet", common=common, backend_config=rtmdet, **overrides)


def run_rfdetr(common=None, rfdetr=None, **overrides):
    run_inference("rfdetr", common=common, backend_config=rfdetr, **overrides)
