from . import runtime_warning_filters  # noqa: F401
import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .process_title_utils import infer_title, read_title, set_process_title

warnings.filterwarnings(
    "ignore",
    message=r"Warning: The .* owner does not match the current owner\.",
    category=UserWarning,
    module=r"torch_npu\.utils\.collect_env",
)
warnings.filterwarnings(
    "ignore",
    message=r".*torch.Tensor.cuda and torch.nn.Module.cuda are replaced.*",
    category=ImportWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*torch.jit.script and torch.jit.script_method will be disabled.*",
    category=RuntimeWarning,
)

from .yolo_image_io_and_export import (
    bind_process_device,
    convert_results,
    count_windows,
    ensure_device,
    generate_starts,
    global_nms,
    open_reader,
    resolve_output_paths,
    tile_batches,
    tqdm,
    write_shapefile,
)
from ultralytics import YOLO

from .building_regularize import write_regularized_shapefile


@dataclass
class MultiprocessInferConfig:
    source: str
    model: str = "best.pt"
    output: Optional[str] = "./outputs/large_image_predictions.json"
    output_shp: Optional[str] = None
    output_bbox_shp: Optional[str] = None
    output_mask_shp: Optional[str] = None
    save_json: bool = True
    # Stream a normal JSON document while inference runs. This keeps huge-image
    # inference bounded in RAM, but cannot perform operations requiring every
    # detection in memory (global NMS, shapefile export, regularization).
    stream_json: bool = False
    device: str = "cpu"
    devices: Optional[List[str]] = None
    worker_count: Optional[int] = None
    reader_workers: int = 1
    queue_size: int = 8
    mp_start: str = "spawn"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 300
    tile_size: int = 1280
    stride: Optional[int] = None
    batch_size: int = 4
    global_nms: bool = False
    global_iou: float = 0.45
    agnostic_nms: bool = False
    classes: Optional[List[int]] = None
    small_image_threshold: int = 12000
    prefer_single_process_for_small: bool = True
    stretch_to_uint8: bool = False
    input_color: str = "rgb"
    # 建筑物轮廓正则化配置 (dict)，None = 不启用
    regularize: Optional[Dict[str, Any]] = None


@dataclass
class ReusableYOLOModel:
    model: Any
    device: str


class StreamingJsonWriter:
    """Atomically write prediction JSON without retaining detections in RAM."""

    def __init__(self, output_path, metadata):
        self.output_path = Path(output_path)
        self.temp_path = self.output_path.with_name(f"{self.output_path.name}.partial")
        self.file = self.temp_path.open("w", encoding="utf-8")
        self.file.write("{")
        for index, (key, value) in enumerate(metadata.items()):
            if index:
                self.file.write(",")
            self.file.write(json.dumps(key, ensure_ascii=False))
            self.file.write(":")
            json.dump(value, self.file, ensure_ascii=False)
        self.file.write(',"detections":[\n')
        self.first_detection = True
        self.detection_count = 0
        self.closed = False

    def write_detections(self, detections):
        for detection in detections:
            if not self.first_detection:
                self.file.write(",\n")
            json.dump(detection, self.file, ensure_ascii=False, separators=(",", ":"))
            self.first_detection = False
            self.detection_count += 1

    def close(self, status="done"):
        if self.closed:
            return
        self.file.write(
            f'],"status":{json.dumps(status)},"raw_detections":{self.detection_count},'
            f'"merged_detections":{self.detection_count}}}\n'
        )
        # A completed ``.json`` is the hand-off point to the post-processing
        # stage.  Flush it to the filesystem before the atomic rename so an
        # abrupt machine/filesystem failure cannot leave a seemingly completed
        # JSON whose tail was still only buffered in userspace.
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        if status == "done":
            os.replace(self.temp_path, self.output_path)
        self.closed = True


def terminate_when_parent_exits():
    """Make CUDA workers exit when an OOM kill removes their parent (Linux)."""
    if os.name != "posix":
        return
    try:
        import ctypes
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM) == 0 and os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        pass


class ProgressReporter:
    def __init__(self, total_tiles, desc_prefix="Tiles"):
        self.total_tiles = total_tiles
        self.start_time = time.monotonic()
        self.last_status_time = 0.0
        self.status_interval = 5.0
        self.read_tiles = 0
        self.infer_tiles = 0
        self.detections = 0
        self.read_bar = None
        self.infer_bar = None
        if tqdm is not None:
            self.read_bar = tqdm(
                total=total_tiles,
                desc=f"{desc_prefix} read ",
                unit="tile",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
            self.infer_bar = tqdm(
                total=total_tiles,
                desc=f"{desc_prefix} infer",
                unit="tile",
                position=1,
                leave=True,
                dynamic_ncols=True,
            )

    def write(self, message):
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message, flush=True)

    def update_read(self, tiles, queue_size=None, ready_workers=None, total_workers=None, detections=None):
        self.read_tiles = self._update_bar(self.read_bar, self.read_tiles, tiles)
        if detections is not None:
            self.detections = max(int(detections), 0)
        self._set_postfix(self.read_bar, queue_size, ready_workers, total_workers, self.detections)
        self.maybe_log_status(queue_size, ready_workers, total_workers, self.detections)

    def update_infer(self, tiles, queue_size=None, ready_workers=None, total_workers=None, detections=None):
        self.infer_tiles = self._update_bar(self.infer_bar, self.infer_tiles, tiles)
        if detections is not None:
            self.detections = max(int(detections), 0)
        self._set_postfix(self.infer_bar, queue_size, ready_workers, total_workers, self.detections)
        self.maybe_log_status(queue_size, ready_workers, total_workers, self.detections)

    def _update_bar(self, bar, current_tiles, new_tiles):
        new_tiles = min(max(int(new_tiles), 0), self.total_tiles)
        delta = new_tiles - current_tiles
        if bar is not None and delta > 0:
            bar.update(delta)
        return max(current_tiles, new_tiles)

    def _set_postfix(self, bar, queue_size=None, ready_workers=None, total_workers=None, detections=None):
        if bar is None:
            return
        postfix = {}
        if queue_size is not None:
            postfix["queue"] = queue_size
        if ready_workers is not None and total_workers is not None:
            postfix["workers"] = f"{ready_workers}/{total_workers}"
        if detections is not None:
            postfix["detections"] = int(detections)
        if postfix:
            bar.set_postfix(postfix, refresh=False)

    def maybe_log_status(self, queue_size=None, ready_workers=None, total_workers=None, detections=None, force=False):
        if tqdm is not None:
            return
        now = time.monotonic()
        if not force and (now - self.last_status_time) < self.status_interval:
            return

        elapsed = max(now - self.start_time, 1e-6)
        read_speed = self.read_tiles / elapsed
        infer_speed = self.infer_tiles / elapsed
        read_pct = (self.read_tiles / self.total_tiles * 100.0) if self.total_tiles > 0 else 0.0
        infer_pct = (self.infer_tiles / self.total_tiles * 100.0) if self.total_tiles > 0 else 0.0
        eta_text = self.format_eta(infer_speed)

        parts = [
            f"[progress] read {self.read_tiles}/{self.total_tiles} ({read_pct:.1f}%)",
            f"infer {self.infer_tiles}/{self.total_tiles} ({infer_pct:.1f}%)",
            f"read_speed {read_speed:.1f} tile/s",
            f"infer_speed {infer_speed:.1f} tile/s",
            f"eta {eta_text}",
            f"detections {int(self.detections if detections is None else detections)}",
        ]
        if queue_size is not None:
            parts.append(f"queue {queue_size}")
        if ready_workers is not None and total_workers is not None:
            parts.append(f"workers {ready_workers}/{total_workers}")

        self.write(", ".join(parts))
        self.last_status_time = now

    def format_eta(self, infer_speed):
        if self.total_tiles <= 0 or infer_speed <= 1e-6:
            return "--"
        remaining_tiles = max(self.total_tiles - self.infer_tiles, 0)
        remaining_seconds = int(round(remaining_tiles / infer_speed))
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def close(self):
        if self.read_bar is not None:
            self.read_bar.n = min(self.read_tiles, self.total_tiles)
            self.read_bar.refresh()
            self.read_bar.close()
        if self.infer_bar is not None:
            self.infer_bar.n = min(self.infer_tiles, self.total_tiles)
            self.infer_bar.refresh()
            self.infer_bar.close()
        self.maybe_log_status(detections=self.detections, force=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Large-image YOLO inference with 1 reader process and multiple predictor processes."
    )
    parser.add_argument("source", type=str, help="Path to a large image file.")
    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help="YOLO model checkpoint path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/large_image_predictions.json",
        help="Path to save merged predictions. JSON by default.",
    )
    parser.add_argument(
        "--output-shp",
        type=str,
        default=None,
        help="Optional path to save bbox predictions as a shapefile. Kept for compatibility.",
    )
    parser.add_argument(
        "--output-bbox-shp",
        type=str,
        default=None,
        help="Optional path to save bbox polygons as a shapefile.",
    )
    parser.add_argument(
        "--output-mask-shp",
        type=str,
        default=None,
        help="Optional path to save instance mask polygons as a shapefile.",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Do not write the merged prediction JSON file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Fallback device used when --devices is omitted.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="Device list for predictor workers, for example cpu or npu:0 npu:1.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Number of predictor workers. Defaults to len(--devices) or 2.",
    )
    parser.add_argument(
        "--reader-workers",
        type=int,
        default=1,
        help="Number of reader processes used for tile extraction. This is CPU I/O work, not NPU work.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=8,
        help="Max number of tile batches buffered between reader and predictors.",
    )
    parser.add_argument(
        "--mp-start",
        type=str,
        default="spawn",
        choices=("spawn", "fork", "forkserver"),
        help="Multiprocessing start method.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold used inside YOLO prediction.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Max detections per tile.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1280,
        help="Tile size for large-image window reading.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Tile stride. Defaults to tile size when omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of tiles per YOLO batch.",
    )
    parser.add_argument(
        "--global-nms",
        action="store_true",
        help="Apply an extra cross-tile global NMS after YOLO prediction.",
    )
    parser.add_argument(
        "--global-iou",
        type=float,
        default=0.45,
        help="IoU threshold for optional cross-tile global NMS.",
    )
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS inside YOLO prediction.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional class IDs to keep.",
    )
    parser.add_argument(
        "--small-image-threshold",
        type=int,
        default=12000,
        help="Prefer a lighter single-process path when both image sides are below this threshold.",
    )
    parser.add_argument(
        "--disable-small-image-fastpath",
        action="store_true",
        help="Always use the multiprocessing pipeline even for around-10000x10000 images.",
    )
    return parser.parse_args()


def build_worker_devices(config):
    base_devices = config.devices if config.devices else [config.device]
    if not base_devices:
        base_devices = ["cpu"]

    worker_count = config.worker_count
    if worker_count is None:
        worker_count = len(base_devices) if config.devices else 2
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")

    return [base_devices[i % len(base_devices)] for i in range(worker_count)]


def validate_config(config):
    if config.reader_workers <= 0:
        raise ValueError("reader_workers must be positive.")
    if config.tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.queue_size <= 0:
        raise ValueError("queue_size must be positive.")
    if (config.stride or config.tile_size) <= 0:
        raise ValueError("stride must be positive.")
    if config.stream_json:
        if not config.save_json:
            raise ValueError("stream_json requires save_json=True.")
        if config.global_nms:
            raise ValueError("stream_json is incompatible with global_nms; it requires all detections in RAM.")
        if config.output_bbox_shp or config.output_mask_shp or config.output_shp:
            raise ValueError("stream_json is incompatible with shapefile export; export it from completed JSON later.")
        if config.regularize:
            raise ValueError("stream_json is incompatible with regularize; it requires all masks in RAM.")


def partitioned_windows(width, height, tile_size, stride, reader_id, reader_count):
    x_starts = generate_starts(width, tile_size, stride)
    y_starts = generate_starts(height, tile_size, stride)
    rows_per_reader = max(1, math.ceil(len(y_starts) / reader_count))
    y_begin = reader_id * rows_per_reader
    y_end = min(len(y_starts), y_begin + rows_per_reader)

    for y1 in y_starts[y_begin:y_end]:
        y2 = min(y1 + tile_size, height)
        for x1 in x_starts:
            x2 = min(x1 + tile_size, width)
            yield (x1, y1, x2, y2)


def tile_reader_process(
    reader_id,
    source_path,
    tile_size,
    stride,
    batch_size,
    reader_count,
    active_readers,
    image_queue,
    message_queue,
    error_queue,
    stretch_to_uint8=False,
    input_color="rgb",
):
    reader = None
    try:
        terminate_when_parent_exits()
        set_process_title(read_title("yolo", reader_id=reader_id))
        reader = open_reader(
            source_path, stretch_to_uint8=stretch_to_uint8, output_color=input_color
        )
        windows = partitioned_windows(
            reader.width, reader.height, tile_size, stride, reader_id, reader_count
        )
        for batch_index, (batch_images, batch_windows) in enumerate(
            tile_batches(reader, windows, batch_size)
        ):
            image_queue.put(
                {
                    "batch_index": batch_index,
                    "images": batch_images,
                    "windows": batch_windows,
                    "reader_id": reader_id,
                }
            )
            message_queue.put(
                {
                    "type": "read",
                    "reader_id": reader_id,
                    "tiles": len(batch_windows),
                }
            )
    except Exception:
        error_queue.put(
            {
                "stage": f"reader:{reader_id}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if reader is not None:
            reader.close()
        with active_readers.get_lock():
            active_readers.value -= 1
        message_queue.put({"type": "reader_done", "reader_id": reader_id})


def predictor_process(
    worker_id,
    device,
    model_path,
    image_width,
    image_height,
    predict_kwargs,
    active_readers,
    image_queue,
    message_queue,
    error_queue,
):
    try:
        terminate_when_parent_exits()
        set_process_title(infer_title("yolo", device=device, worker_id=worker_id))
        device = bind_process_device(device)
        ensure_device(device)
        model = YOLO(model_path)

        while True:
            try:
                task = image_queue.get(timeout=1)
            except queue.Empty:
                if active_readers.value == 0:
                    break
                continue

            results = model.predict(
                source=task["images"],
                device=yolo_predict_device(device),
                imgsz=predict_kwargs["imgsz"],
                conf=predict_kwargs["conf"],
                iou=predict_kwargs["iou"],
                max_det=predict_kwargs["max_det"],
                classes=predict_kwargs["classes"],
                agnostic_nms=predict_kwargs["agnostic_nms"],
                verbose=False,
                save=False,
            )
            detections = convert_results(
                results, task["windows"], image_width, image_height
            )
            message_queue.put(
                {
                    "type": "result",
                    "worker_id": worker_id,
                    "batch_index": task["batch_index"],
                    "tiles": len(task["windows"]),
                    "detections": detections,
                }
            )
    except Exception:
        error_queue.put(
            {
                "stage": f"worker:{worker_id}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        message_queue.put({"type": "worker_done", "worker_id": worker_id, "device": device})


def get_error_nowait(error_queue):
    try:
        return error_queue.get_nowait()
    except queue.Empty:
        return None


def terminate_processes(processes):
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(timeout=2)


def join_processes(processes):
    for proc in processes:
        proc.join()


def safe_qsize(queue_obj):
    try:
        return queue_obj.qsize()
    except (NotImplementedError, AttributeError, OSError):
        return None


def resolve_runtime_output_paths(config):
    output_arg = config.output or "./outputs/large_image_predictions.json"
    output_path, output_bbox_shp_path, output_mask_shp_path = resolve_output_paths(
        output_arg, config.output_shp, config.output_bbox_shp, config.output_mask_shp
    )
    if not config.save_json:
        output_path = None
    return output_path, output_bbox_shp_path, output_mask_shp_path


def should_use_small_image_fastpath(config, reader):
    if not config.prefer_single_process_for_small:
        return False
    threshold = max(int(config.small_image_threshold), 1)
    return reader.width <= threshold and reader.height <= threshold


def maybe_prepare_output_dirs(output_path, output_bbox_shp_path, output_mask_shp_path):
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_bbox_shp_path is not None:
        output_bbox_shp_path.parent.mkdir(parents=True, exist_ok=True)
    if output_mask_shp_path is not None:
        output_mask_shp_path.parent.mkdir(parents=True, exist_ok=True)


def load_reusable_model(config, worker_device):
    worker_device = bind_process_device(worker_device)
    ensure_device(worker_device)
    return ReusableYOLOModel(model=YOLO(config.model), device=worker_device)


def yolo_predict_device(device):
    device = str(device)
    if device.startswith("npu:"):
        return device.split(":", 1)[1]
    if device == "npu":
        return "0"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    if device == "cuda":
        return "0"
    if device.startswith("gpu:"):
        return device.split(":", 1)[1]
    if device == "gpu":
        return "0"
    return device


def dump_output_files(
    output,
    merged_detections,
    meta_reader,
    output_path,
    output_bbox_shp_path,
    output_mask_shp_path,
    global_nms_enabled,
    raw_detection_count,
    regularize=None,
):
    # 先做正则化（会改写 merged_detections，写入 segment_xy_regularized），
    # 这样 JSON 与可视化都能反映正则化结果。
    if output_mask_shp_path is not None and regularize:
        write_regularized_shapefile(
            output_mask_shp_path, merged_detections, meta_reader, regularize
        )
    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"[info] saved merged predictions to {output_path}")
    if output_bbox_shp_path is not None:
        write_shapefile(output_bbox_shp_path, merged_detections, meta_reader, geometry="bbox")
        if meta_reader.crs_wkt:
            print(f"[info] saved bbox shapefile to {output_bbox_shp_path} with source CRS")
        else:
            print(f"[info] saved bbox shapefile to {output_bbox_shp_path} using pixel coordinates")
    if output_mask_shp_path is not None:
        write_shapefile(output_mask_shp_path, merged_detections, meta_reader, geometry="mask")
        mask_count = sum(1 for det in merged_detections if det.get("segment_xy"))
        if meta_reader.crs_wkt:
            print(
                f"[info] saved mask shapefile to {output_mask_shp_path} with source CRS "
                f"({mask_count} polygons)"
            )
        else:
            print(
                f"[info] saved mask shapefile to {output_mask_shp_path} using pixel coordinates "
                f"({mask_count} polygons)"
            )

    if global_nms_enabled:
        print(
            f"[info] detections before global NMS: {raw_detection_count}, "
            f"after global NMS: {len(merged_detections)}"
        )
    else:
        print(f"[info] detections exported without extra global NMS: {len(merged_detections)}")


def run_single_process_inference(config, meta_reader, stride, worker_device, reusable_model=None):
    source_path = Path(config.source)
    output_path, output_bbox_shp_path, output_mask_shp_path = resolve_runtime_output_paths(config)
    maybe_prepare_output_dirs(output_path, output_bbox_shp_path, output_mask_shp_path)

    print(
        f"[info] small-image fast path enabled for {meta_reader.width}x{meta_reader.height}. "
        f"Using single-process batched inference on {worker_device}."
    )

    reusable_model = reusable_model or load_reusable_model(config, worker_device)
    model = reusable_model.model
    total_tiles = count_windows(meta_reader.width, meta_reader.height, config.tile_size, stride)
    reporter = ProgressReporter(total_tiles)
    all_detections = []
    processed_tiles = 0

    try:
        windows = partitioned_windows(
            meta_reader.width,
            meta_reader.height,
            config.tile_size,
            stride,
            0,
            1,
        )
        for batch_images, batch_windows in tile_batches(
            meta_reader, windows, config.batch_size
        ):
            results = model.predict(
                source=batch_images,
                device=yolo_predict_device(worker_device),
                imgsz=config.imgsz,
                conf=config.conf,
                iou=config.iou,
                max_det=config.max_det,
                classes=config.classes,
                agnostic_nms=config.agnostic_nms,
                verbose=False,
                save=False,
            )
            all_detections.extend(
                convert_results(results, batch_windows, meta_reader.width, meta_reader.height)
            )
            processed_tiles += len(batch_windows)
            reporter.update_read(processed_tiles, ready_workers=1, total_workers=1, detections=len(all_detections))
            reporter.update_infer(processed_tiles, ready_workers=1, total_workers=1, detections=len(all_detections))
    finally:
        reporter.close()

    if config.global_nms:
        merged_detections = global_nms(all_detections, config.global_iou)
    else:
        merged_detections = all_detections

    if output_bbox_shp_path is not None and output_mask_shp_path is None:
        if any(det.get("segment_xy") for det in merged_detections):
            output_mask_shp_path = output_bbox_shp_path.with_name(
                f"{output_bbox_shp_path.stem}_mask.shp"
            )

    output = {
        "image": str(source_path),
        "backend": meta_reader.backend,
        "image_size": {"width": meta_reader.width, "height": meta_reader.height},
        "model": config.model,
        "devices": [worker_device],
        "reader_workers": 1,
        "worker_count": 1,
        "imgsz": config.imgsz,
        "conf": config.conf,
        "tile_size": config.tile_size,
        "stride": stride,
        "batch_size": config.batch_size,
        "queue_size": 0,
        "tiles": total_tiles,
        "global_nms": bool(config.global_nms),
        "raw_detections": len(all_detections),
        "merged_detections": len(merged_detections),
        "detections": merged_detections,
    }

    dump_output_files(
        output=output,
        merged_detections=merged_detections,
        meta_reader=meta_reader,
        output_path=output_path,
        output_bbox_shp_path=output_bbox_shp_path,
        output_mask_shp_path=output_mask_shp_path,
        global_nms_enabled=config.global_nms,
        raw_detection_count=len(all_detections),
        regularize=config.regularize,
    )
    return output


def run_multiprocess_inference(config: MultiprocessInferConfig):
    validate_config(config)
    stride = config.stride or config.tile_size
    worker_devices = build_worker_devices(config)

    if len(set(worker_devices)) == 1 and len(worker_devices) > 1:
        print(
            f"[warn] {len(worker_devices)} predictor processes share the same device "
            f"{worker_devices[0]}. This may reduce throughput instead of speeding it up."
        )
    if config.reader_workers > 1:
        print(
            f"[info] using {config.reader_workers} reader processes for tile extraction. "
            "Readers use CPU and storage I/O rather than NPU."
        )

    source_path = Path(config.source)
    output_path, output_bbox_shp_path, output_mask_shp_path = resolve_runtime_output_paths(
        config
    )
    maybe_prepare_output_dirs(output_path, output_bbox_shp_path, output_mask_shp_path)

    meta_reader = open_reader(
        str(source_path), stretch_to_uint8=config.stretch_to_uint8, output_color=config.input_color
    )
    print(
        f"[info] opened image with {meta_reader.backend}: "
        f"{source_path} ({meta_reader.width}x{meta_reader.height})"
    )
    if meta_reader.crs_wkt:
        print("[info] source image CRS detected and will be written to .prj")
    else:
        print(
            "[warn] no georeference was read from the source image. "
            "The output shapefile will use pixel coordinates."
        )

    if should_use_small_image_fastpath(config, meta_reader):
        output = run_single_process_inference(
            config=config,
            meta_reader=meta_reader,
            stride=stride,
            worker_device=worker_devices[0],
        )
        meta_reader.close()
        return output

    total_tiles = count_windows(meta_reader.width, meta_reader.height, config.tile_size, stride)
    total_row_strips = len(generate_starts(meta_reader.height, config.tile_size, stride))
    print(
        f"[info] total tiles: {total_tiles}, tile_size={config.tile_size}, stride={stride}, "
        f"readers={config.reader_workers}, workers={len(worker_devices)}, devices={worker_devices}"
    )
    if config.reader_workers > total_row_strips:
        print(
            f"[warn] reader_workers={config.reader_workers} exceeds tile rows={total_row_strips}. "
            f"Extra readers will sit idle."
        )

    ctx = mp.get_context(config.mp_start)
    image_queue = ctx.Queue(maxsize=config.queue_size)
    message_queue = ctx.Queue(
        maxsize=max(config.queue_size, len(worker_devices), config.reader_workers) * 4
    )
    error_queue = ctx.Queue()
    active_readers = ctx.Value("i", config.reader_workers)

    predict_kwargs = {
        "imgsz": config.imgsz,
        "conf": config.conf,
        "iou": config.iou,
        "max_det": config.max_det,
        "classes": config.classes,
        "agnostic_nms": config.agnostic_nms,
    }

    processes = []
    for reader_id in range(config.reader_workers):
        proc = ctx.Process(
            target=tile_reader_process,
            args=(
                reader_id,
                str(source_path),
                config.tile_size,
                stride,
                config.batch_size,
                config.reader_workers,
                active_readers,
                image_queue,
                message_queue,
                error_queue,
                config.stretch_to_uint8,
                config.input_color,
            ),
            name=f"tile-reader-{reader_id}",
        )
        processes.append(proc)

    for worker_id, device in enumerate(worker_devices):
        proc = ctx.Process(
            target=predictor_process,
            args=(
                worker_id,
                device,
                config.model,
                meta_reader.width,
                meta_reader.height,
                predict_kwargs,
                active_readers,
                image_queue,
                message_queue,
                error_queue,
            ),
            name=f"predictor-{worker_id}",
        )
        processes.append(proc)

    for proc in processes:
        proc.start()

    all_detections = []
    read_tiles = 0
    inferred_tiles = 0
    finished_readers = 0
    finished_workers = 0
    reporter = ProgressReporter(total_tiles)
    stream_writer = None
    if config.stream_json:
        stream_writer = StreamingJsonWriter(
            output_path,
            {
                "image": str(source_path),
                "backend": meta_reader.backend,
                "image_size": {"width": meta_reader.width, "height": meta_reader.height},
                "model": config.model,
                "devices": worker_devices,
                "reader_workers": config.reader_workers,
                "worker_count": len(worker_devices),
                "imgsz": config.imgsz,
                "conf": config.conf,
                "tile_size": config.tile_size,
                "stride": stride,
                "batch_size": config.batch_size,
                "queue_size": config.queue_size,
                "tiles": total_tiles,
                "global_nms": False,
            },
        )

    try:
        while finished_workers < len(worker_devices):
            err = get_error_nowait(error_queue)
            if err is not None:
                raise RuntimeError(
                    f"Multiprocess inference failed in {err['stage']}:\n{err['traceback']}"
                )

            try:
                message = message_queue.get(timeout=1)
            except queue.Empty:
                continue

            if message["type"] == "read":
                read_tiles += message["tiles"]
                reporter.update_read(
                    read_tiles,
                    queue_size=safe_qsize(image_queue),
                    ready_workers=len(worker_devices) - finished_workers,
                    total_workers=len(worker_devices),
                )
            elif message["type"] == "reader_done":
                finished_readers += 1
            elif message["type"] == "result":
                if stream_writer is not None:
                    stream_writer.write_detections(message["detections"])
                    detection_count = stream_writer.detection_count
                else:
                    all_detections.extend(message["detections"])
                    detection_count = len(all_detections)
                inferred_tiles += message["tiles"]
                reporter.update_infer(
                    inferred_tiles,
                    queue_size=safe_qsize(image_queue),
                    ready_workers=len(worker_devices) - finished_workers,
                    total_workers=len(worker_devices),
                    detections=detection_count,
                )
            elif message["type"] == "worker_done":
                finished_workers += 1
    except KeyboardInterrupt:
        terminate_processes(processes)
        if stream_writer is not None:
            stream_writer.close(status="interrupted")
        raise
    except Exception:
        terminate_processes(processes)
        if stream_writer is not None:
            stream_writer.close(status="failed")
        raise
    finally:
        reporter.close()

    join_processes(processes)

    if stream_writer is not None:
        stream_writer.close()
        meta_reader.close()
        print(
            f"[info] streamed {stream_writer.detection_count} detections to {output_path}; "
            "global NMS and shapefile export were intentionally deferred."
        )
        return {
            "image": str(source_path),
            "status": "done",
            "raw_detections": stream_writer.detection_count,
            "merged_detections": stream_writer.detection_count,
            "detections": [],
        }

    if config.global_nms:
        merged_detections = global_nms(all_detections, config.global_iou)
    else:
        merged_detections = all_detections

    if output_bbox_shp_path is not None and output_mask_shp_path is None:
        if any(det.get("segment_xy") for det in merged_detections):
            output_mask_shp_path = output_bbox_shp_path.with_name(
                f"{output_bbox_shp_path.stem}_mask.shp"
            )

    output = {
        "image": str(source_path),
        "backend": meta_reader.backend,
        "image_size": {"width": meta_reader.width, "height": meta_reader.height},
        "model": config.model,
        "devices": worker_devices,
        "reader_workers": config.reader_workers,
        "worker_count": len(worker_devices),
        "imgsz": config.imgsz,
        "conf": config.conf,
        "tile_size": config.tile_size,
        "stride": stride,
        "batch_size": config.batch_size,
        "queue_size": config.queue_size,
        "tiles": total_tiles,
        "global_nms": bool(config.global_nms),
        "raw_detections": len(all_detections),
        "merged_detections": len(merged_detections),
        "detections": merged_detections,
    }

    dump_output_files(
        output=output,
        merged_detections=merged_detections,
        meta_reader=meta_reader,
        output_path=output_path,
        output_bbox_shp_path=output_bbox_shp_path,
        output_mask_shp_path=output_mask_shp_path,
        global_nms_enabled=config.global_nms,
        raw_detection_count=len(all_detections),
        regularize=config.regularize,
    )

    meta_reader.close()
    return output


def run_reused_single_process_inference(config, reusable_model):
    validate_config(config)
    stride = config.stride or config.tile_size
    worker_devices = build_worker_devices(config)
    worker_device = worker_devices[0]
    source_path = Path(config.source)

    meta_reader = open_reader(
        str(source_path), stretch_to_uint8=config.stretch_to_uint8, output_color=config.input_color
    )
    print(
        f"[info] opened image with {meta_reader.backend}: "
        f"{source_path} ({meta_reader.width}x{meta_reader.height})"
    )
    if meta_reader.crs_wkt:
        print("[info] source image CRS detected and will be written to .prj")
    else:
        print(
            "[warn] no georeference was read from the source image. "
            "The output shapefile will use pixel coordinates."
        )

    try:
        if should_use_small_image_fastpath(config, meta_reader):
            return run_single_process_inference(
                config=config,
                meta_reader=meta_reader,
                stride=stride,
                worker_device=worker_device,
                reusable_model=reusable_model,
            )

        return run_multiprocess_inference(config)
    finally:
        meta_reader.close()


def namespace_to_config(args):
    return MultiprocessInferConfig(
        source=args.source,
        model=args.model,
        output=args.output,
        output_shp=args.output_shp,
        output_bbox_shp=args.output_bbox_shp,
        output_mask_shp=args.output_mask_shp,
        save_json=not args.no_save_json,
        device=args.device,
        devices=args.devices,
        worker_count=args.worker_count,
        reader_workers=args.reader_workers,
        queue_size=args.queue_size,
        mp_start=args.mp_start,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        tile_size=args.tile_size,
        stride=args.stride,
        batch_size=args.batch_size,
        global_nms=args.global_nms,
        global_iou=args.global_iou,
        agnostic_nms=args.agnostic_nms,
        classes=args.classes,
        small_image_threshold=args.small_image_threshold,
        prefer_single_process_for_small=not args.disable_small_image_fastpath,
        stretch_to_uint8=False,
        input_color=getattr(args, "input_color", "rgb"),
    )


def main():
    args = parse_args()
    run_multiprocess_inference(namespace_to_config(args))


if __name__ == "__main__":
    main()
