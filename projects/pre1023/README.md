# pre1023 —— 1023 测试影像（`/data1/wzh26/data/1023测试影像/1/pre.tif`）的推理脚本

这些是**项目专用**的一次性脚本（不是框架通用入口）。新任务请用仓库根目录的
`run_large.py`；如果发现某个模型会长期使用，再把它做成
`detect_speedup/backends/` 里的插件（见 `backends/_template.py`）。

| 脚本 | 模型 | 输出目录 | 结果 |
|---|---|---|---|
| `run_pre1023_gzda_det.py` | yolo26x-det（gzda_simplified） | `output/pre1023_gzda_det`（原生 0.6 m）、`output/pre1023_gzda_det_scale2`（512 瓦片放大 2×） | 8394 / 9733 个框 |
| `run_pre1023_gzda_seg.py` | yolo26x-seg（GZDA 0.6 m 子集）+ 可换 yolo26n-seg | `output/pre1023_gzda_seg`、`output/pre1023_gzda_simplified_yolo26n_seg*` | 9316 / 3878 / 6609 个掩膜 |
| `run_pre1023_rfdetr_gzda.py` | RF-DETR seg-medium（gzda_simplified） | `output/pre1023_rfdetr_seg_medium_gzda`、`output/pre1023_rfdetr_qa_dense` | 整图未跑完（停在 58/528）；稠密切片 2129 个 |
| `run_pre1023_rtdetr_hf.py` | HF `RTDetrForObjectDetection`（社区屋顶权重） | `output/pre1023_rtdetr_hf` | 23363 个框 |
| `run_pre1023_skydet.py` | SKYDET（gzda_v1，640） | `output/pre1023_skydet_gzda_v1` | 29206 个框 |
| `run_pre1023_p2pformer.py` | P2PFormer（WHU-Mix，建筑轮廓） | — | 轮廓提取 |
| `run_yuantou_threshold.py` | 园头村局部补推 / 阈值导出 | — | — |
| `run_building_best_pre.py` | 重研流水线：两个 RF-DETR best 权重 + 园头村裁切 | `/data1/wzh26/重研/outputs/pre_20261028_rfdetr_best` | — |
| `finalize_building_pre_outputs.py` | 上面那套结果的收尾（矢量/图预览） | 同上 | — |

## 运行

脚本已加 `sys.path` 引导，从仓库根目录或本目录运行都可以：

```bash
cd /data1/wzh26/code/detect-speedup
/home/wzh26/anaconda3/envs/yolo/bin/python projects/pre1023/run_pre1023_skydet.py
```

各脚本顶部 docstring 写了模型路径、预处理口径与可选环境变量
（`PRE_IN / PRE_OUT / PRE_MODEL / PRE_TILE / PRE_STRIDE / PRE_CONF / PRE_BATCH / PRE_DEV`）。

## 结论汇总

- 出成果推荐 **yolo26x-seg（GZDA 0.6 m 训练）**：训练分辨率与影像一致，9316 个掩膜、密度 63/km²、
  中位边长 16.6 m，1 分 23 秒。
- SKYDET / RT-DETR 两个"外来"权重密度是它的 2.5-3 倍（158 / 197.5 个/km²），需要目视确认误检率。
- 尺度失配很关键：模型训练在 ~0.27-0.3 m，影像是 0.6 m；把瓦片半径减半（放大 2×）能显著提升召回。
- 详细对比见 `output/pre1023_*/README_RESULT.md` 与 `output/pre1023_all/README_汇总.md`。
