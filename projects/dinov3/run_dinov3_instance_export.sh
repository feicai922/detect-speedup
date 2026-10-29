#!/usr/bin/env bash
# Run DINOv3 semantic inference, then convert predicted labels to instances.
set -euo pipefail

CODE_DIR="${CODE_DIR:-/data1/wzh26/code_bda/bright_extended}"
PYTHON_BIN="${PYTHON_BIN:-/home/wzh26/anaconda3/envs/yolo/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data1/wzh26/data/bright_raw}"
SPLIT_DIR="${SPLIT_DIR:-/data1/wzh26/code_bda/BRIGHT/bda_benchmark/dataset/splitname/standard_ML}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/output/semantic_dinov3/infer_test}"
CHECKPOINT="${CHECKPOINT:-${CODE_DIR}/output/semantic_dinov3/best_model.pth}"
GPU_ID="${GPU_ID:-0}"
MODEL_NAME="${MODEL_NAME:-vit_base_patch16_dinov3}"
CHANNELS="${CHANNELS:-4}"
MIN_AREA="${MIN_AREA:-9}"

export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

[[ -f "$CHECKPOINT" ]] || { echo "[ERROR] checkpoint not found: $CHECKPOINT" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$CODE_DIR/semantic_dinov3/infer.py" \
  --root "$DATASET_ROOT" --split_dir "$SPLIT_DIR" \
  --checkpoint "$CHECKPOINT" --split test --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" --channels "$CHANNELS" --gpu 0

"$PYTHON_BIN" "$CODE_DIR/bright_dinov3_instances.py" \
  "$OUTPUT_DIR" "$OUTPUT_DIR/instances.geojson" --min-area "$MIN_AREA"

echo "[DONE] instance output: $OUTPUT_DIR/instances.geojson"
