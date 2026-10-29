#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-/data1/wzh26/code_bda/bright_extended}"
PYTHON_BIN="${PYTHON_BIN:-/home/wzh26/anaconda3/envs/yolo/bin/python}"
ROOT="${ROOT:-/data1/wzh26/data/bright_raw}"
OFFICIAL="${OFFICIAL:-/data1/wzh26/code_bda/BRIGHT/cvprw26}"
OUT="${OUT:-${CODE_DIR}/output/dino_maskrcnn_bright}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODEL_NAME="${MODEL_NAME:-vit_base_patch16_dinov3}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="${CODE_DIR}:${OFFICIAL}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$OUT"

"$PYTHON_BIN" "$CODE_DIR/dino_maskrcnn_bright.py" \
  --root "$ROOT" \
  --train-ann "$OFFICIAL/data/instance_annotations/train.json" \
  --val-ann "$OFFICIAL/data/instance_annotations/val.json" \
  --output-dir "$OUT" --model-name "$MODEL_NAME" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --gpu 0
