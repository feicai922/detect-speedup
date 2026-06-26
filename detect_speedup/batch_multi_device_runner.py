from . import runtime_warning_filters  # noqa: F401
import contextlib
import multiprocessing as mp
from numbers import Integral
import os
import traceback
from pathlib import Path

from .process_title_utils import dir_title, set_process_title

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_SUPPORTED_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


def iter_input_files(input_path, recursive=False, supported_suffixes=None):
    input_path = Path(input_path)
    suffixes = supported_suffixes or DEFAULT_SUPPORTED_SUFFIXES
    if input_path.is_file():
        yield input_path
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    for path in sorted(input_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def build_output_paths(input_path, output_dir, input_file, output_name=None, save_json=True, save_bbox_shp=True, save_mask_shp=True):
    output_dir = Path(output_dir)
    if Path(input_path).is_file() and output_name:
        stem = output_name
    else:
        stem = input_file.stem

    json_path = output_dir / f"{stem}.json" if save_json else None

    if save_bbox_shp and save_mask_shp:
        bbox_shp = output_dir / f"{stem}_bbox.shp"
        mask_shp = output_dir / f"{stem}_mask.shp"
    elif save_bbox_shp:
        bbox_shp = output_dir / f"{stem}.shp"
        mask_shp = None
    elif save_mask_shp:
        bbox_shp = None
        mask_shp = output_dir / f"{stem}.shp"
    else:
        bbox_shp = None
        mask_shp = None

    return json_path, bbox_shp, mask_shp


def should_skip_outputs(*paths):
    expected = [path for path in paths if path is not None]
    return bool(expected) and all(path.exists() for path in expected)


def partition_files_round_robin(input_files, worker_count):
    buckets = [[] for _ in range(worker_count)]
    for index, input_file in enumerate(input_files):
        buckets[index % worker_count].append(input_file)
    return buckets


def count_result_detections(result):
    if result is None:
        return 0

    for key in ("merged_detections", "raw_detections"):
        value = result.get(key)
        if isinstance(value, Integral):
            return max(int(value), 0)
        if value is not None:
            return len(value)

    detections = result.get("detections")
    return len(detections) if detections is not None else 0


def count_detections(outcome):
    value = outcome.get("detection_count")
    if isinstance(value, Integral):
        return max(int(value), 0)

    return count_result_detections(outcome)


def update_progress_postfix(progress, detected_count, last_file=None):
    if not progress:
        return

    postfix = {"targets": detected_count}
    if last_file:
        postfix["last"] = last_file
    progress.set_postfix(postfix)


def _run_one_quiet(adapter, input_file, output_dir, reusable_model=None, device=None):
    if adapter.should_skip(input_file, output_dir):
        return {"status": "skip", "file": input_file.name, "detection_count": 0, "device": device}

    config = adapter.build_device_bound_config(input_file, output_dir, device) if device else adapter.build_config(input_file, output_dir)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            if reusable_model is not None:
                result = adapter.run_reused_single_process_inference(config, reusable_model)
            else:
                result = adapter.run_multiprocess_inference(config)

    return {
        "status": "done",
        "file": input_file.name,
        "detection_count": count_result_detections(result),
        "device": device,
    }


def directory_worker_main(worker_id, device, input_files, output_dir, adapter, message_queue):
    set_process_title(dir_title(adapter.backend_name, device=device, worker_id=worker_id))
    reusable_model = None
    try:
        if input_files:
            reusable_model = adapter.load_reusable_model(
                adapter.build_device_bound_config(input_files[0], output_dir, device),
                device,
            )

        for input_file in input_files:
            message_queue.put({"type": "start", "worker_id": worker_id, "device": device, "file": input_file.name})
            outcome = _run_one_quiet(
                adapter=adapter,
                input_file=input_file,
                output_dir=output_dir,
                reusable_model=reusable_model,
                device=device,
            )
            message_queue.put(
                {
                    "type": "done",
                    "worker_id": worker_id,
                    "device": device,
                    **outcome,
                }
            )
    except Exception:
        message_queue.put(
            {
                "type": "error",
                "worker_id": worker_id,
                "device": device,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        del reusable_model


def run_one(adapter, input_file, output_dir):
    if adapter.should_skip(input_file, output_dir):
        return None

    result = adapter.run_multiprocess_inference(adapter.build_config(input_file, output_dir))
    return result


def run_directory_multi_device(adapter, input_files, output_dir):
    devices = adapter.devices or ["cpu"]
    worker_count = min(len(devices), len(input_files))
    if worker_count <= 1:
        reusable_model = adapter.load_reusable_model(
            adapter.build_device_bound_config(input_files[0], output_dir, devices[0]),
            devices[0],
        )
        try:
            iterator = tqdm(input_files, desc="Files", unit="file", dynamic_ncols=True) if tqdm else input_files
            for input_file in iterator:
                outcome = _run_one_quiet(
                    adapter=adapter,
                    input_file=input_file,
                    output_dir=output_dir,
                    reusable_model=reusable_model,
                    device=devices[0],
                )
                current_targets = count_detections(outcome)
                update_progress_postfix(iterator if tqdm else None, current_targets, input_file.name)
        finally:
            del reusable_model
        return

    ctx = mp.get_context(adapter.mp_start)
    message_queue = ctx.Queue()
    device_buckets = partition_files_round_robin(input_files, worker_count)
    processes = []

    for worker_id in range(worker_count):
        process = ctx.Process(
            target=directory_worker_main,
            args=(worker_id, devices[worker_id], device_buckets[worker_id], output_dir, adapter, message_queue),
            name=f"{adapter.backend_name}-dir-{worker_id}",
        )
        processes.append(process)
        process.start()

    completed = 0
    try:
        progress = tqdm(total=len(input_files), desc="Files", unit="file", dynamic_ncols=True) if tqdm else None
        try:
            while completed < len(input_files):
                message = message_queue.get()
                if message["type"] == "start":
                    continue

                if message["type"] == "error":
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    raise RuntimeError(
                        f"Directory inference failed on {message['device']}:\n{message['traceback']}"
                    )

                if message["type"] == "done":
                    completed += 1
                    current_targets = count_detections(message)
                    if progress:
                        progress.update(1)
                    update_progress_postfix(progress, current_targets, message.get("file"))
                    continue
        finally:
            if progress:
                progress.close()
    finally:
        for process in processes:
            process.join()


def run_inputs(adapter):
    input_path = Path(adapter.input_path)
    output_dir = Path(adapter.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(
        iter_input_files(
            input_path,
            recursive=adapter.recursive,
            supported_suffixes=adapter.supported_suffixes,
        )
    )
    if not input_files:
        raise RuntimeError(
            f"No supported input files were found in {input_path}. "
            f"Supported suffixes: {sorted(adapter.supported_suffixes)}"
        )

    print(f"[info] backend={adapter.backend_name}, found {len(input_files)} input files")
    if input_path.is_file() or len(input_files) == 1 or tqdm is None:
        for index, input_file in enumerate(input_files, start=1):
            run_one(adapter, input_file, output_dir)
        return

    run_directory_multi_device(adapter, input_files, output_dir)
