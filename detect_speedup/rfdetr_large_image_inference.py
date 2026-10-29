from . import runtime_warning_filters  # noqa: F401
import json
import math
import multiprocessing as mp
import os
import queue
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

from .process_title_utils import infer_title, read_title, set_process_title
from .yolo_image_io_and_export import (
    clip_image_point,
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

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


@dataclass
class MultiprocessRFDETRInferConfig:
    source: str
    checkpoint: str
    num_classes: int = 1
    resolution: int = 640
    output: Optional[str] = "./outputs/large_image_predictions.json"
    output_shp: Optional[str] = None
    output_bbox_shp: Optional[str] = None
    output_mask_shp: Optional[str] = None
    save_json: bool = True
    device: str = "npu:0"
    devices: Optional[List[str]] = None
    worker_count: Optional[int] = None
    reader_workers: int = 1
    queue_size: int = 8
    mp_start: str = "spawn"
    conf_threshold: float = 0.3
    iou_threshold: float = 0.5
    tile_size: int = 1280
    stride: Optional[int] = None
    batch_size: int = 4
    global_nms: bool = True
    global_iou: float = 0.45
    classes: Optional[List[int]] = None
    small_image_threshold: int = 12000
    prefer_single_process_for_small: bool = True
    stretch_to_uint8: bool = False
    input_color: str = "rgb"
    fp16: bool = False
    # 建筑物轮廓正则化配置 (dict)，None = 不启用
    regularize: Optional[dict] = None


@dataclass
class ReusableRFDETRModel:
    model: Any
    class_names: List[str]
    device: str


class ProgressReporter:
    def __init__(self, total_tiles, desc_prefix="Tiles"):
        self.total_tiles = total_tiles
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

    def _update_bar(self, bar, current, new_value):
        new_value = min(int(new_value), self.total_tiles)
        delta = max(0, new_value - current)
        if bar is not None and delta:
            bar.update(delta)
        return max(current, new_value)

    def _set_postfix(self, bar, queue_size=None, ready_workers=None, total_workers=None, detections=None):
        if bar is None:
            return
        postfix = {}
        if queue_size is not None:
            postfix["queue"] = queue_size
        if ready_workers is not None and total_workers is not None:
            postfix["workers"] = f"{ready_workers}/{total_workers}"
        if detections is not None:
            postfix["targets"] = detections
        bar.set_postfix(postfix)

    def update_read(self, tiles, queue_size=None, ready_workers=None, total_workers=None, detections=None):
        self.read_tiles = self._update_bar(self.read_bar, self.read_tiles, tiles)
        if detections is not None:
            self.detections = max(int(detections), 0)
        self._set_postfix(self.read_bar, queue_size, ready_workers, total_workers, self.detections)

    def update_infer(self, tiles, queue_size=None, ready_workers=None, total_workers=None, detections=None):
        self.infer_tiles = self._update_bar(self.infer_bar, self.infer_tiles, tiles)
        if detections is not None:
            self.detections = max(int(detections), 0)
        self._set_postfix(self.infer_bar, queue_size, ready_workers, total_workers, self.detections)

    def close(self):
        if self.read_bar is not None:
            self.read_bar.close()
        if self.infer_bar is not None:
            self.infer_bar.close()


def normalize_device(device):
    from .infer_rfdetr_npu import normalize_device as _normalize_device

    return _normalize_device(str(device))


def build_worker_devices(config):
    base_devices = config.devices if config.devices else [config.device]
    if not base_devices:
        base_devices = ["npu:0"]
    base_devices = [normalize_device(device) if str(device).startswith(("npu", "NPU")) or str(device).isdigit() else str(device) for device in base_devices]

    worker_count = config.worker_count
    if worker_count is None:
        worker_count = len(base_devices) if config.devices else 1
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
    from .infer_rfdetr_npu import load_model

    worker_device = normalize_device(worker_device)
    ensure_device(worker_device)
    model, class_names = load_model(
        checkpoint_path=config.checkpoint,
        num_classes=config.num_classes,
        device=worker_device,
        resolution=config.resolution,
    )
    return ReusableRFDETRModel(model=model, class_names=class_names, device=worker_device)


def _class_allowed(class_id, classes):
    return classes is None or int(class_id) in set(int(value) for value in classes)


def mask_geometry(mask, x_offset, y_offset):
    """掩膜按像素边界转矢量，保留一个实例的多连通区域与孔洞。"""
    import rasterio.features
    from affine import Affine
    if hasattr(mask, 'detach'):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask).squeeze()
    if mask.ndim != 2:
        raise ValueError('Expected a two-dimensional instance mask')
    mask = mask > 0
    rows, cols = np.where(mask)
    if not rows.size:
        return None, None
    left, top, right, bottom = int(cols.min()), int(rows.min()), int(cols.max())+1, int(rows.max())+1
    crop = mask[top:bottom, left:right].astype(np.uint8)
    polygons = [geom['coordinates'] for geom, value in rasterio.features.shapes(
        crop, mask=crop.astype(bool), connectivity=4,
        transform=Affine.translation(x_offset + left, y_offset + top)) if value == 1]
    if not polygons:
        return None, None
    def area(ring):
        points = np.asarray(ring)
        return abs(np.dot(points[:-1, 0], points[1:, 1]) - np.dot(points[1:, 0], points[:-1, 1])) / 2
    primary = max(polygons, key=lambda rings: area(rings[0]))[0]
    geometry = {'type': 'Polygon', 'coordinates': polygons[0]} if len(polygons) == 1 else {
        'type': 'MultiPolygon', 'coordinates': polygons}
    return primary, geometry


_MASK_POOL = None


def _mask_task(task):
    return mask_geometry(*task)


def _convert_tile_predictions(predictions, window, image_width, image_height, class_names, conf_threshold, classes):
    if predictions is None:
        return []
    det = predictions[0] if isinstance(predictions, list) else predictions
    boxes, scores = det.get('boxes'), det.get('scores')
    if boxes is None or scores is None:
        return []
    def numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    # 每张切片只拷贝一次分数、类别和框，避免逐实例 CUDA 同步。
    boxes, scores = numpy(boxes), numpy(scores)
    labels = numpy(det['labels']) if det.get('labels') is not None else np.zeros(len(scores), dtype=np.int64)
    allowed = scores >= conf_threshold
    if classes is not None:
        allowed &= np.isin(labels, list(classes))
    indices = np.flatnonzero(allowed)
    if not len(indices):
        return []
    x_offset, y_offset, _, _ = window
    masks = det.get('masks')
    if masks is None:
        geometries = [(None, None)] * len(indices)
    else:
        if hasattr(masks, 'detach'):
            selected = torch.as_tensor(indices, device=masks.device)
            masks = masks.index_select(0, selected).detach().cpu().numpy()
        else:
            masks = np.asarray(masks)[indices]
        tasks = [(mask, x_offset, y_offset) for mask in masks]
        if len(tasks) >= 8:
            global _MASK_POOL
            if _MASK_POOL is None:
                from concurrent.futures import ThreadPoolExecutor
                _MASK_POOL = ThreadPoolExecutor(max_workers=4)
            geometries = list(_MASK_POOL.map(_mask_task, tasks))
        else:
            geometries = [_mask_task(task) for task in tasks]
    detections = []
    for index, (segment, geometry) in zip(indices, geometries):
        if masks is not None and geometry is None:
            continue
        box = boxes[index]
        gx1, gy1 = clip_image_point(float(box[0])+x_offset, float(box[1])+y_offset, image_width, image_height)
        gx2, gy2 = clip_image_point(float(box[2])+x_offset, float(box[3])+y_offset, image_width, image_height)
        class_id = int(labels[index])
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else 'class_%d' % class_id
        detections.append(dict(bbox_xyxy=[gx1,gy1,gx2,gy2], score=float(scores[index]),
            class_id=class_id, class_name=class_name, tile_xyxy=[int(v) for v in window],
            segment_xy=segment, mask_geometry=geometry))
    return detections


def predict_batch(reusable_model, images, windows, image_width, image_height, config):
    from .infer_rfdetr_npu import preprocess_image

    tensors = []
    target_sizes = []
    for image in images:
        tile_h, tile_w = image.shape[:2]
        tensor, _ = preprocess_image(
            image,
            resolution=config.resolution,
            input_color=config.input_color,
            device=reusable_model.device,
        )
        tensors.append(tensor.squeeze(0))
        target_sizes.append([tile_h, tile_w])

    batch_tensor = torch.stack(tensors, dim=0)
    target_sizes_tensor = torch.tensor(target_sizes, device=reusable_model.device)
    from contextlib import nullcontext
    mixed = torch.autocast('cuda', dtype=torch.float16) if config.fp16 and str(reusable_model.device).startswith('cuda') else nullcontext()
    with torch.inference_mode(), mixed:
        outputs = reusable_model.model.model.model(batch_tensor)
        predictions = reusable_model.model.model.postprocess(outputs, target_sizes_tensor)

    detections = []
    for prediction, window in zip(predictions, windows):
        detections.extend(
            _convert_tile_predictions(
                prediction,
                window,
                image_width,
                image_height,
                reusable_model.class_names,
                config.conf_threshold,
                config.classes,
            )
        )
    return detections


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
        from .building_regularize import write_regularized_shapefile
        write_regularized_shapefile(
            output_mask_shp_path, merged_detections, meta_reader, regularize
        )
    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)
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


def build_output(source_path, meta_reader, config, devices, total_tiles, raw_detections, merged_detections, stride, queue_size):
    return {
        "image": str(source_path),
        "backend": meta_reader.backend,
        "image_size": {"width": meta_reader.width, "height": meta_reader.height},
        "checkpoint": config.checkpoint,
        "devices": devices,
        "reader_workers": config.reader_workers if queue_size else 1,
        "worker_count": len(devices),
        "resolution": config.resolution,
        "conf": config.conf_threshold,
        "iou": config.iou_threshold,
        "tile_size": config.tile_size,
        "stride": stride,
        "batch_size": config.batch_size,
        "instance_segmentation": any(d.get('mask_geometry') for d in merged_detections),
        "queue_size": queue_size,
        "tiles": total_tiles,
        "global_nms": bool(config.global_nms),
        "raw_detections": len(raw_detections),
        "merged_detections": len(merged_detections),
        "detections": merged_detections,
    }


def run_single_process_inference(config, meta_reader, stride, worker_device, reusable_model=None):
    source_path = Path(config.source)
    output_path, output_bbox_shp_path, output_mask_shp_path = resolve_runtime_output_paths(config)
    maybe_prepare_output_dirs(output_path, output_bbox_shp_path, output_mask_shp_path)
    print(
        f"[info] single-process tiled path enabled for {meta_reader.width}x{meta_reader.height}. "
        f"Using single-process batched tile inference on {worker_device}."
    )

    reusable_model = reusable_model or load_reusable_model(config, worker_device)
    total_tiles = count_windows(meta_reader.width, meta_reader.height, config.tile_size, stride)
    reporter = ProgressReporter(total_tiles)
    all_detections = []
    processed_tiles = 0

    try:
        windows = partitioned_windows(meta_reader.width, meta_reader.height, config.tile_size, stride, 0, 1)
        for batch_images, batch_windows in tile_batches(meta_reader, windows, config.batch_size):
            detections = predict_batch(
                reusable_model,
                batch_images,
                batch_windows,
                meta_reader.width,
                meta_reader.height,
                config,
            )
            all_detections.extend(detections)
            processed_tiles += len(batch_windows)
            reporter.update_read(processed_tiles, ready_workers=1, total_workers=1, detections=len(all_detections))
            reporter.update_infer(processed_tiles, ready_workers=1, total_workers=1, detections=len(all_detections))
    finally:
        reporter.close()

    if processed_tiles != total_tiles:
        raise RuntimeError('Tile coverage incomplete: %d/%d' % (processed_tiles, total_tiles))

    merged_detections = global_nms(all_detections, config.global_iou) if config.global_nms else all_detections
    output = build_output(
        source_path,
        meta_reader,
        config,
        [worker_device],
        total_tiles,
        all_detections,
        merged_detections,
        stride,
        0,
    )
    output['processed_tiles'] = processed_tiles
    dump_output_files(
        output,
        merged_detections,
        meta_reader,
        output_path,
        output_bbox_shp_path,
        output_mask_shp_path,
        config.global_nms,
        len(all_detections),
        regularize=config.regularize,
    )
    return output


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
        set_process_title(read_title("rfdetr", reader_id=reader_id))
        reader = open_reader(source_path, stretch_to_uint8=stretch_to_uint8, output_color=input_color)
        windows = partitioned_windows(reader.width, reader.height, tile_size, stride, reader_id, reader_count)
        for batch_index, (batch_images, batch_windows) in enumerate(tile_batches(reader, windows, batch_size)):
            image_queue.put(
                {
                    "batch_index": batch_index,
                    "images": batch_images,
                    "windows": batch_windows,
                    "reader_id": reader_id,
                }
            )
            message_queue.put({"type": "read", "reader_id": reader_id, "tiles": len(batch_windows)})
    except Exception:
        error_queue.put({"stage": f"reader:{reader_id}", "traceback": traceback.format_exc()})
    finally:
        if reader is not None:
            reader.close()
        with active_readers.get_lock():
            active_readers.value -= 1
        message_queue.put({"type": "reader_done", "reader_id": reader_id})


def predictor_process(
    worker_id,
    device,
    config,
    image_width,
    image_height,
    active_readers,
    image_queue,
    message_queue,
    error_queue,
):
    try:
        set_process_title(infer_title("rfdetr", device=device, worker_id=worker_id))
        reusable_model = load_reusable_model(config, device)

        while True:
            try:
                task = image_queue.get(timeout=1)
            except queue.Empty:
                if active_readers.value == 0:
                    break
                continue

            detections = predict_batch(
                reusable_model,
                task["images"],
                task["windows"],
                image_width,
                image_height,
                config,
            )
            message_queue.put(
                {
                    "type": "result",
                    "worker_id": worker_id,
                    "batch_index": task["batch_index"],
                    "windows": task["windows"],
                    "tiles": len(task["windows"]),
                    "detections": detections,
                }
            )
    except Exception:
        error_queue.put({"stage": f"worker:{worker_id}", "traceback": traceback.format_exc()})
    finally:
        message_queue.put({"type": "worker_done", "worker_id": worker_id, "device": device})


def get_error_nowait(error_queue):
    try:
        return error_queue.get_nowait()
    except queue.Empty:
        return None


def terminate_processes(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2)


def join_processes(processes):
    for process in processes:
        process.join()


def safe_qsize(queue_obj):
    try:
        return queue_obj.qsize()
    except (NotImplementedError, AttributeError, OSError):
        return None


def run_multiprocess_inference(config: MultiprocessRFDETRInferConfig):
    validate_config(config)
    stride = config.stride or config.tile_size
    worker_devices = build_worker_devices(config)

    source_path = Path(config.source)
    output_path, output_bbox_shp_path, output_mask_shp_path = resolve_runtime_output_paths(config)
    maybe_prepare_output_dirs(output_path, output_bbox_shp_path, output_mask_shp_path)

    meta_reader = open_reader(
        str(source_path),
        stretch_to_uint8=config.stretch_to_uint8,
        output_color=config.input_color,
    )
    print(
        f"[info] opened image with {meta_reader.backend}: "
        f"{source_path} ({meta_reader.width}x{meta_reader.height})"
    )
    if meta_reader.crs_wkt:
        print("[info] source image CRS detected and will be written to .prj")
    else:
        print("[warn] no georeference was read from the source image. The output shapefile will use pixel coordinates.")

    try:
        if should_use_small_image_fastpath(config, meta_reader):
            return run_single_process_inference(
                config=config,
                meta_reader=meta_reader,
                stride=stride,
                worker_device=worker_devices[0],
            )

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
        message_queue = ctx.Queue(maxsize=max(config.queue_size, len(worker_devices), config.reader_workers) * 4)
        error_queue = ctx.Queue()
        active_readers = ctx.Value("i", config.reader_workers)

        processes = []
        for reader_id in range(config.reader_workers):
            process = ctx.Process(
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
                name=f"rfdetr-reader-{reader_id}",
            )
            processes.append(process)

        for worker_id, device in enumerate(worker_devices):
            process = ctx.Process(
                target=predictor_process,
                args=(
                    worker_id,
                    device,
                    config,
                    meta_reader.width,
                    meta_reader.height,
                    active_readers,
                    image_queue,
                    message_queue,
                    error_queue,
                ),
                name=f"rfdetr-predictor-{worker_id}",
            )
            processes.append(process)

        for process in processes:
            process.start()

        all_detections = []
        seen_windows = set()
        read_tiles = 0
        inferred_tiles = 0
        finished_workers = 0
        reporter = ProgressReporter(total_tiles)

        try:
            while finished_workers < len(worker_devices):
                err = get_error_nowait(error_queue)
                if err is not None:
                    raise RuntimeError(f"Multiprocess inference failed in {err['stage']}:\n{err['traceback']}")

                try:
                    message = message_queue.get(timeout=1)
                except queue.Empty:
                    failed = [p.name for p in processes if p.exitcode not in (None, 0)]
                    if failed:
                        raise RuntimeError('Inference processes exited unexpectedly: ' + str(failed))
                    continue

                if message["type"] == "read":
                    read_tiles += message["tiles"]
                    reporter.update_read(
                        read_tiles,
                        queue_size=safe_qsize(image_queue),
                        ready_workers=len(worker_devices) - finished_workers,
                        total_workers=len(worker_devices),
                    )
                elif message["type"] == "result":
                    for window in message['windows']:
                        key = tuple(window)
                        if key in seen_windows:
                            raise RuntimeError('Duplicate image tile: ' + str(key))
                        seen_windows.add(key)
                    all_detections.extend(message["detections"])
                    inferred_tiles += message["tiles"]
                    reporter.update_infer(
                        inferred_tiles,
                        queue_size=safe_qsize(image_queue),
                        ready_workers=len(worker_devices) - finished_workers,
                        total_workers=len(worker_devices),
                        detections=len(all_detections),
                    )
                elif message["type"] == "worker_done":
                    finished_workers += 1
        except KeyboardInterrupt:
            terminate_processes(processes)
            raise
        except Exception:
            terminate_processes(processes)
            raise
        finally:
            reporter.close()

        join_processes(processes)
        err = get_error_nowait(error_queue)
        if err is not None:
            raise RuntimeError('Multiprocess inference failed: ' + str(err))
        if inferred_tiles != total_tiles or len(seen_windows) != total_tiles:
            raise RuntimeError('Tile coverage incomplete: inferred=%d unique=%d expected=%d' %
                               (inferred_tiles, len(seen_windows), total_tiles))

        merged_detections = global_nms(all_detections, config.global_iou) if config.global_nms else all_detections
        output = build_output(
            source_path,
            meta_reader,
            config,
            worker_devices,
            total_tiles,
            all_detections,
            merged_detections,
            stride,
            config.queue_size,
        )
        output['processed_tiles'] = inferred_tiles
        output['unique_tiles'] = len(seen_windows)
        dump_output_files(
            output,
            merged_detections,
            meta_reader,
            output_path,
            output_bbox_shp_path,
            output_mask_shp_path,
            config.global_nms,
            len(all_detections),
        )
        return output
    finally:
        meta_reader.close()


def run_reused_single_process_inference(config, reusable_model):
    validate_config(config)
    stride = config.stride or config.tile_size
    worker_device = reusable_model.device
    source_path = Path(config.source)

    meta_reader = open_reader(
        str(source_path),
        stretch_to_uint8=config.stretch_to_uint8,
        output_color=config.input_color,
    )
    print(
        f"[info] opened image with {meta_reader.backend}: "
        f"{source_path} ({meta_reader.width}x{meta_reader.height})"
    )
    if meta_reader.crs_wkt:
        print("[info] source image CRS detected and will be written to .prj")
    else:
        print("[warn] no georeference was read from the source image. The output shapefile will use pixel coordinates.")

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
