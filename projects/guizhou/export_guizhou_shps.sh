#!/usr/bin/env bash
# 顺序将两个贵州省 YOLO 流式 JSON 导出为 mask Shapefile。
# 两个目录是不同模型的推理结果，使用同一个源影像提供 CRS/地理变换。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE="/data1/wzh26/data/贵州省大图-0.6m/贵州省.tif"
THRESHOLD="${THRESHOLD:-0.3}"

OUTPUT_ROOT="$SCRIPT_DIR/output"
STREAM_EXPORTER="$SCRIPT_DIR/stream_json_to_shp.py"

if [[ ! -f "$SOURCE" ]]; then
    printf '源影像不存在: %s\n' "$SOURCE" >&2
    exit 1
fi
if [[ ! -f "$STREAM_EXPORTER" ]]; then
    printf '导出脚本不存在: %s\n' "$STREAM_EXPORTER" >&2
    exit 1
fi

models=(
    "yolo_贵州省大图"
    "yolo_贵州省大图-0.6m"
)

export_one() {
    local model_dir="$1"
    local json_path="$OUTPUT_ROOT/$model_dir/json/贵州省.json"
    local shp_path="$OUTPUT_ROOT/$model_dir/shps/all_mask.shp"

    if [[ ! -f "$json_path" ]]; then
        printf 'JSON 不存在，停止处理: %s\n' "$json_path" >&2
        return 1
    fi

    printf '\n===== 导出 %s =====\n' "$model_dir"
    "$PYTHON_BIN" "$STREAM_EXPORTER" \
        "$json_path" \
        --source "$SOURCE" \
        --output "$shp_path" \
        --threshold "$THRESHOLD" \
        --recover-truncated-tail \
        --geometry mask
}
for model_dir in "${models[@]}"; do
    export_one "$model_dir"
done

printf '\n两个模型的 SHP 导出完成。\n'
