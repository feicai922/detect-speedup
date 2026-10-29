#!/usr/bin/env bash
# =============================================================================
# 多模型 × 多影像 批量推理 + 合并 SHP 脚本
#
# 执行顺序 (按 MODELS × IMAGES 笛卡尔积):
#     1) python run_demo_config.py     # 单后端推理, 产出 json/ + per-tile shp
#     2) python run_shp_only.py        # 把 per-tile json 合并成单个 .shp
#
# 动态添加模型: 向 MODELS 数组追加一行即可。
# 动态添加影像: 向 IMAGES 数组追加一行即可。
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 路径
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_CONFIG="$SCRIPT_DIR/run_demo_config.py"
SHP_ONLY="$SCRIPT_DIR/run_shp_only.py"
BACKUP="$DEMO_CONFIG.bak.$(date +%s)"

# -----------------------------------------------------------------------------
# 1. 模型列表 — 追加条目即"动态添加"
#    格式: backend|device_csv|output_suffix|env_spec|extras
#      backend       : yolo | rtmdet | rfdetr
#      device_csv    : 逗号分隔, 例如 "cuda:0,cuda:1,cuda:5,cuda:6"
#      output_suffix : 输出目录名后缀, 例如 "rtmdet" -> ./output/rtmdet_巴伦西亚
#      env_spec      : 运行 python 用的环境, 两种写法
#                        conda:NAME              -> 激活 conda 环境 NAME
#                        uv:/abs/path/to/project -> 用 uv 项目目录 (含 pyproject.toml)
#                      留空表示直接用系统 python
#      extras        : 分号分隔的 key=value, 用于覆盖对应后端块里的字段;
#                      VALUE 必须是合法 Python 字面量 —
#                        字符串自带引号:  model_path=\"/data1/.../best.pt\"
#                        数字原样:      imgsz=640  conf=0.25
#                        布尔原样:      agnostic_nms=False
#                      留空表示沿用 run_demo_config.py 中的默认值
# -----------------------------------------------------------------------------
MODELS=(
    # "yolo|cuda:4|yolo|conda:yolo|imgsz=640;model_path=\"/data1/wzh26/code/yolo/runs/segment/kaiyuan_building_seg-2/weights/best.pt\";conf=0.3"
    # "rtmdet|cuda:0,cuda:1,cuda:5,cuda:6|rtmdet|conda:openmmlab|score_thr=0.3"
    # "rfdetr|cuda:0,cuda:1,cuda:5,cuda:6|rfdetr|uv:/data1/wzh26/code/rf-detr|conf_threshold=0.3"
    # "yolo|cuda:0,cuda:4|yolo_1|conda:yolo|model_path=\"/data1/wzh26/code/yolo/runs/segment/gzda_building_seg/weights/best.pt\""
    # "yolo|cuda:0,cuda:4|yolo_2|conda:yolo|model_path=\"/data1/wzh26/code/yolo/runs/segment/merge_building_seg/weights/best.pt\""
    "rtmdet|cuda:0,cuda:4|rtmdet|conda:openmmlab|config_file=\"/data1/wzh26/code/mmdetection/work_dirs/gzda_rtmdet/gzda_rtmdet.py\";checkpoint_file=\"/data1/wzh26/code/mmdetection/work_dirs/gzda_rtmdet/best_coco_segm_mAP_epoch_281.pth\""
)

# -----------------------------------------------------------------------------
# 2. 两张影像 — 追加条目即"动态添加"
# -----------------------------------------------------------------------------
IMAGES=(
    "/data1/wzh26/code/detect-speedup/input/加拉加斯/img.tif"
    # "/data1/wzh26/code/detect-speedup/input/首都/img.tif"
)

# -----------------------------------------------------------------------------
# 3. 通用参数 (run_shp_only.py 用)
# -----------------------------------------------------------------------------
SHP_GEOMETRY="mask"
SHP_THRESHOLD="0.3"

# -----------------------------------------------------------------------------
# 4. DEBUG 模式: 设为 1 时只打印将要执行的命令, 不真正跑 python
# -----------------------------------------------------------------------------
DEBUG="${DEBUG:-0}"

# =============================================================================
# 工具函数
# =============================================================================

# 把 "cuda:0,cuda:1" 转成 Python 列表字面量 ["cuda:0","cuda:1"]
csv_to_py_list() {
    local csv="$1"
    local out="[" first=1
    local IFS=','
    local item
    for item in $csv; do
        if [ "$first" -eq 0 ]; then out+=","; fi
        out+="\"$item\""
        first=0
    done
    out+="]"
    printf '%s' "$out"
}

# 修补 run_demo_config.py 的 BACKEND 与 COMMON 顶层字段
patch_common() {
    local backend="$1" image="$2" output_dir="$3" devices_py="$4"
    sed -i \
        -e "s|^BACKEND = .*|BACKEND = \"$backend\"|" \
        -e "s|\"input_path\": .*|\"input_path\": \"$image\",|" \
        -e "s|\"output_dir\": .*|\"output_dir\": \"$output_dir\",|" \
        -e "s|\"devices\": .*|\"devices\": $devices_py,|" \
        "$DEMO_CONFIG"
}

# 在 YOLO / RTMDET / RFDETR 块内替换指定字段
patch_backend_block() {
    local backend="$1"; shift
    local block=""
    case "$backend" in
        yolo)          block="YOLO"   ;;
        rtmdet)        block="RTMDET" ;;
        rfdetr|rf-detr) block="RFDETR" ;;
        *)             return 0 ;;
    esac
    local kv key value
    for kv in "$@"; do
        [ -z "$kv" ] && continue
        key="${kv%%=*}"
        value="${kv#*=}"
        sed -i "/^${block} = {/,/^}/ s|\"$key\": .*|\"$key\": $value,|" "$DEMO_CONFIG"
    done
}

# 打印并(可选)执行一条命令
run_cmd() {
    echo "+ $*" >&2
    if [ "$DEBUG" != "1" ]; then
        "$@"
    fi
}

# 在指定环境下跑 python:  env_spec = "conda:NAME" | "uv:/path" | ""
# 用法: run_with_env "$env_spec" script.py arg1 arg2 ...
run_with_env() {
    local env_spec="$1"; shift
    local kind="${env_spec%%:*}"
    local target="${env_spec#*:}"
    case "$kind" in
        conda)
            run_cmd conda run -n "$target" --no-capture-output python "$@"
            ;;
        uv)
            run_cmd uv run --project "$target" --no-sync -- python "$@"
            ;;
        "")
            run_cmd python "$@"
            ;;
        *)
            echo "[error] 未知的 env_spec: $env_spec (支持 conda:NAME / uv:PATH / 空)" >&2
            return 2
            ;;
    esac
}

# =============================================================================
# 主流程
# =============================================================================

# 前置检查
if [ ! -f "$DEMO_CONFIG" ]; then
    echo "[error] 找不到 $DEMO_CONFIG" >&2; exit 1
fi
if [ ! -f "$SHP_ONLY" ]; then
    echo "[error] 找不到 $SHP_ONLY" >&2; exit 1
fi
for img in "${IMAGES[@]}"; do
    if [ ! -f "$img" ]; then
        echo "[error] 影像不存在: $img" >&2; exit 1
    fi
done

# 备份 run_demo_config.py, 退出时恢复 (含异常退出)
cp "$DEMO_CONFIG" "$BACKUP"
cleanup() {
    if [ -f "$BACKUP" ]; then
        mv -f "$BACKUP" "$DEMO_CONFIG"
    fi
}
trap cleanup EXIT INT TERM

# 双层循环: 每个 (model, image) 组合 → 先推理 → 再合并 SHP
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r backend devices suffix env_spec extras <<< "$model_entry"

    # 解析 extras -> 数组
    extra_arr=()
    if [ -n "$extras" ]; then
        # 过滤空段 (尾部多余的 ; 会产生空元素)
        IFS=';' read -ra raw_arr <<< "$extras"
        for kv in "${raw_arr[@]}"; do
            [ -n "$kv" ] && extra_arr+=("$kv")
        done
    fi

    for image in "${IMAGES[@]}"; do
        parent="$(basename "$(dirname "$image")")"
        output_dir="./output/${suffix}_${parent}"
        json_dir="$output_dir/json"
        devices_py="$(csv_to_py_list "$devices")"

        echo "============================================================"
        echo "[run] backend = $backend"
        echo "      image   = $image"
        echo "      devices = [$devices]"
        echo "      env     = ${env_spec:-<system python>}"
        echo "      output  = $output_dir"
        [ "${#extra_arr[@]}" -gt 0 ] && echo "      extras  = ${extra_arr[*]}"
        echo "============================================================"

        # 1) 修补 run_demo_config.py
        patch_common "$backend" "$image" "$output_dir" "$devices_py"
        if [ "${#extra_arr[@]}" -gt 0 ]; then
            patch_backend_block "$backend" "${extra_arr[@]}"
        fi

        # 2) 推理 (per-tile json + per-tile shp) — 在该模型对应的环境里跑
        run_with_env "$env_spec" "$DEMO_CONFIG"

        # 3) 合并 SHP — 用同一 env, 避免 detect_speedup 跨环境 import 不一致
        run_with_env "$env_spec" "$SHP_ONLY" "$json_dir" \
            --crs-source "$image" \
            --geometry "$SHP_GEOMETRY" \
            --threshold "$SHP_THRESHOLD"

        echo "[ok] $backend × $parent finished -> $output_dir"
        echo
    done
done

echo "[done] all combinations finished."