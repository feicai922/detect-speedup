from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .batch_multi_device_runner import build_output_paths, should_skip_outputs


@dataclass
class BaseInferenceAdapter:
    config: Dict[str, Any]
    backend_name: str

    @property
    def input_path(self):
        return self.config["input_path"]

    @property
    def output_dir(self):
        return self.config["output_dir"]

    @property
    def devices(self):
        devices = self.config.get("devices")
        if isinstance(devices, str):
            return [device.strip() for device in devices.split(",") if device.strip()]
        return devices

    @property
    def mp_start(self):
        return self.config.get("mp_start", "spawn")

    @property
    def recursive(self):
        return self.config.get("recursive", False)

    @property
    def supported_suffixes(self):
        return self.config.get(
            "supported_suffixes",
            {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"},
        )

    @property
    def auto_split_threshold(self):
        """Opt-in: when > 0, inputs whose longest side exceeds this value are
        pre-split into ``<= auto_split_threshold`` tiles before inference.
        Default 0 — the backends already stream tiles via rasterio / GDAL, so
        this is only needed for PIL-only environments that hit the
        decompression-bomb check."""
        return self.config.get("auto_split_threshold", 0)

    @property
    def auto_split_workers(self):
        return int(self.config.get("auto_split_workers", 1))

    @property
    def auto_split_cache_dir(self):
        """Override the directory used to cache auto-split tiles. ``None``
        (default) places the cache alongside the source image so the output
        directory stays untouched."""
        return self.config.get("auto_split_cache_dir")

    def output_paths(self, input_file, output_dir):
        return build_output_paths(
            input_path=self.config["input_path"],
            output_dir=output_dir,
            input_file=input_file,
            output_name=self.config.get("output_name"),
            save_json=self.config.get("save_json", True),
            save_bbox_shp=self.config.get("save_bbox_shp", True),
            save_mask_shp=self.config.get("save_mask_shp", True),
        )

    def should_skip(self, input_file, output_dir):
        if not self.config.get("skip_if_output_exists", False):
            return False
        return should_skip_outputs(*self.output_paths(input_file, output_dir))


@dataclass
class YOLOInferenceAdapter(BaseInferenceAdapter):
    backend_name: str = "yolo"

    def build_config(self, input_file, output_dir):
        from .yolo_large_image_inference import MultiprocessInferConfig

        json_path, bbox_shp, mask_shp = self.output_paths(input_file, output_dir)
        return MultiprocessInferConfig(
            source=str(input_file),
            model=self.config["model_path"],
            output=str(json_path) if json_path is not None else None,
            output_bbox_shp=str(bbox_shp) if bbox_shp is not None else None,
            output_mask_shp=str(mask_shp) if mask_shp is not None else None,
            save_json=self.config.get("save_json", True),
            stream_json=self.config.get("stream_json", False),
            devices=self.devices,
            worker_count=self.config.get("worker_count"),
            reader_workers=self.config.get("reader_workers", 1),
            mp_start=self.config.get("mp_start", "spawn"),
            tile_size=self.config.get("tile_size", 1280),
            stride=self.config.get("stride"),
            batch_size=self.config.get("batch_size", 4),
            queue_size=self.config.get("queue_size", 8),
            imgsz=self.config.get("imgsz", 640),
            conf=self.config.get("conf", 0.25),
            iou=self.config.get("iou", 0.45),
            max_det=self.config.get("max_det", 300),
            agnostic_nms=self.config.get("agnostic_nms", False),
            classes=self.config.get("classes"),
            global_nms=self.config.get("global_nms", False),
            global_iou=self.config.get("global_iou", 0.45),
            small_image_threshold=self.config.get("small_image_threshold", 12000),
            prefer_single_process_for_small=self.config.get("prefer_single_process_for_small", True),
            stretch_to_uint8=self.config.get("stretch_to_uint8", False),
            input_color=self.config.get("input_color", "rgb"),
            regularize=self.config.get("regularize"),
        )

    def build_device_bound_config(self, input_file, output_dir, device):
        config = self.build_config(input_file, output_dir)
        config.device = device
        config.devices = [device]
        config.worker_count = 1
        return config

    def load_reusable_model(self, config, device):
        from .yolo_large_image_inference import load_reusable_model

        return load_reusable_model(config, device)

    def run_reused_single_process_inference(self, config, reusable_model):
        from .yolo_large_image_inference import run_reused_single_process_inference

        return run_reused_single_process_inference(config, reusable_model)

    def run_multiprocess_inference(self, config):
        from .yolo_large_image_inference import run_multiprocess_inference

        return run_multiprocess_inference(config)


@dataclass
class MMDetInferenceAdapter(BaseInferenceAdapter):
    backend_name: str = "rtmdet"

    def build_config(self, input_file, output_dir):
        from .rtmdet_large_image_inference import MultiprocessMMDetInferConfig

        json_path, bbox_shp, mask_shp = self.output_paths(input_file, output_dir)
        return MultiprocessMMDetInferConfig(
            source=str(input_file),
            config_file=str(Path(self.config["config_file"])),
            checkpoint_file=str(Path(self.config["checkpoint_file"])),
            output=str(json_path) if json_path is not None else None,
            output_bbox_shp=str(bbox_shp) if bbox_shp is not None else None,
            output_mask_shp=str(mask_shp) if mask_shp is not None else None,
            save_json=self.config.get("save_json", True),
            devices=self.devices,
            worker_count=self.config.get("worker_count"),
            reader_workers=self.config.get("reader_workers", 1),
            mp_start=self.config.get("mp_start", "spawn"),
            small_image_threshold=self.config.get("small_image_threshold", 12000),
            prefer_single_process_for_small=self.config.get("prefer_single_process_for_small", True),
            tile_size=self.config.get("tile_size", 1280),
            stride=self.config.get("stride"),
            batch_size=self.config.get("batch_size", 4),
            queue_size=self.config.get("queue_size", 8),
            score_thr=self.config.get("score_thr", 0.3),
            classes=self.config.get("classes"),
            global_nms=self.config.get("global_nms", False),
            global_iou=self.config.get("global_iou", 0.45),
            fp16=self.config.get("fp16", False),
            save_mask_overlay=self.config.get("save_mask_overlay", False),
            regularize=self.config.get("regularize"),
        )

    def build_device_bound_config(self, input_file, output_dir, device):
        config = self.build_config(input_file, output_dir)
        config.device = device
        config.devices = [device]
        config.worker_count = 1
        return config

    def load_reusable_model(self, config, device):
        from .rtmdet_large_image_inference import load_reusable_model

        return load_reusable_model(config, device)

    def run_reused_single_process_inference(self, config, reusable_model):
        from .rtmdet_large_image_inference import run_reused_single_process_inference

        return run_reused_single_process_inference(config, reusable_model)

    def run_multiprocess_inference(self, config):
        from .rtmdet_large_image_inference import run_multiprocess_inference

        return run_multiprocess_inference(config)


@dataclass
class RFDETRInferenceAdapter(BaseInferenceAdapter):
    backend_name: str = "rfdetr"

    def build_config(self, input_file, output_dir):
        from .rfdetr_large_image_inference import MultiprocessRFDETRInferConfig

        json_path, bbox_shp, mask_shp = self.output_paths(input_file, output_dir)
        return MultiprocessRFDETRInferConfig(
            source=str(input_file),
            checkpoint=str(Path(self.config["checkpoint"])),
            num_classes=self.config.get("num_classes", 1),
            resolution=self.config.get("resolution", 640),
            output=str(json_path) if json_path is not None else None,
            output_bbox_shp=str(bbox_shp) if bbox_shp is not None else None,
            output_mask_shp=str(mask_shp) if mask_shp is not None else None,
            save_json=self.config.get("save_json", True),
            devices=self.devices,
            worker_count=self.config.get("worker_count"),
            reader_workers=self.config.get("reader_workers", 1),
            mp_start=self.config.get("mp_start", "spawn"),
            small_image_threshold=self.config.get("small_image_threshold", 12000),
            prefer_single_process_for_small=self.config.get("prefer_single_process_for_small", True),
            tile_size=self.config.get("tile_size", 1280),
            stride=self.config.get("stride"),
            batch_size=self.config.get("batch_size", 4),
            queue_size=self.config.get("queue_size", 8),
            conf_threshold=self.config.get("conf_threshold", 0.3),
            iou_threshold=self.config.get("iou_threshold", 0.5),
            classes=self.config.get("classes"),
            global_nms=self.config.get("global_nms", True),
            global_iou=self.config.get("global_iou", 0.45),
            stretch_to_uint8=self.config.get("stretch_to_uint8", False),
            input_color=self.config.get("input_color", "rgb"),
            fp16=self.config.get('fp16', False),
            regularize=self.config.get("regularize"),
        )

    def build_device_bound_config(self, input_file, output_dir, device):
        config = self.build_config(input_file, output_dir)
        config.device = device
        config.devices = [device]
        config.worker_count = 1
        return config

    def load_reusable_model(self, config, device):
        from .rfdetr_large_image_inference import load_reusable_model

        return load_reusable_model(config, device)

    def run_reused_single_process_inference(self, config, reusable_model):
        from .rfdetr_large_image_inference import run_reused_single_process_inference

        return run_reused_single_process_inference(config, reusable_model)

    def run_multiprocess_inference(self, config):
        from .rfdetr_large_image_inference import run_multiprocess_inference

        return run_multiprocess_inference(config)
