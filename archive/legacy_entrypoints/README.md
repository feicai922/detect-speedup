# 被 `run_large.py` 取代的旧入口（保留备查，可删）

| 文件 | 原用途 | 现在用什么 |
|---|---|---|
| `run_demo.py` | 命令行版入口（`--yolo/--rtmdet/--rfdetr`） | `run_large.py --backend ...` |
| `run_yolo.py` / `run_rtmdet.py` / `run_rfdetr.py` | 各后端单独的 config-dict 入口 | `run_large.py --backend yolo/rtmdet/rfdetr` |
| `run_yolo_test.py` | `run_yolo.py` 的旧副本（里面的模型路径已失效） | 同上 |

仍在根目录、还能用的：`run_demo_config.py`（老的 config-dict 入口，逻辑最全，
比如 `stream_json` / `AUTO_EXPORT_STREAM_SHP` / 建筑物轮廓正则化，这些高级开关
`run_large.py` 还没有完全覆盖）、`run_vis.py`、`run_shp_only.py`、`split_tif.py`、
`stream_json_to_shp.py`、`run_all.sh`。

迁移提示：

```bash
# 旧：python run_yolo.py                  （改文件里的 COMMON/YOLO dict）
# 新：
python run_large.py --backend yolo --model best.pt --input pre.tif --out ./output/x \
    --tile 1024 --imgsz 640 --conf 0.3 --batch 8 --device cuda:0
```

需要 `stream_json`（超大图不占内存）或轮廓正则化时，暂时继续用 `run_demo_config.py`。
