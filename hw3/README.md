# 数字图像处理作业三

本仓库中的 `hw3` 是《数字图像处理》课程第三次作业的提交内容，主题为 **Bundle Adjustment 与基于 COLMAP 的三维重建**。本次作业包含两个部分：

1. 使用 PyTorch 从零实现 Bundle Adjustment，联合优化共享焦距、相机外参与三维点坐标。
2. 使用 COLMAP 对给定多视角图像执行标准 SfM/MVS 流程，完成稀疏与稠密三维重建。

- 学生姓名：程开良
- 学号：SA25001013
- 课程：数字图像处理

---

## Requirements

本次作业在 Windows + Conda 环境下完成，主要使用环境为：

```bash
conda activate myenv
```

主要依赖包括：

- `torch`
- `numpy`
- `scipy`
- `matplotlib`
- `Pillow`

如需安装 Python 依赖，可在 `03_BundleAdjustment` 目录下执行：

```bash
cd hw3/03_BundleAdjustment
python -m pip install -r requirements.txt
```

Task 2 额外依赖 COLMAP。由于仓库提交时不包含体积较大的 COLMAP 二进制与压缩包，实际运行时使用了本地 CUDA 版 COLMAP。

---

## Training

### 任务一：PyTorch 实现 Bundle Adjustment

实现文件：

- `03_BundleAdjustment/train_ba.py`
- `03_BundleAdjustment/ba_utils.py`

主要完成内容：

- 读取 `points2d.npz` 中的多视角 2D 观测与可见性掩码
- 使用共享焦距、相机欧拉角、相机平移和三维点坐标作为待优化变量
- 实现针孔投影模型与可见点重投影误差
- 采用因子分解初始化，提高大规模 BA 的收敛稳定性
- 使用两阶段优化策略，先稳定相机参数，再联合优化全部变量
- 导出 loss 曲线、重投影可视化、彩色点云与指标文件

正式运行命令如下：

```bash
cd hw3/03_BundleAdjustment
python train_ba.py ^
  --data-dir data ^
  --output-dir outputs/final_run ^
  --num-iters 700 ^
  --freeze-points-iters 100 ^
  --log-every 50 ^
  --lr-euler 1e-2 ^
  --lr-trans 8e-3 ^
  --lr-points 2e-3 ^
  --smooth-weight 5e-5
```

### 任务二：COLMAP 稀疏与稠密重建

相关文件：

- `03_BundleAdjustment/run_colmap.ps1`
- `03_BundleAdjustment/render_ply_preview.py`

主要完成内容：

- 在 Windows 下修正 COLMAP 的 Qt 插件路径问题
- 支持 CPU 模式与完整 dense 流程
- 完成 `feature_extractor`、`exhaustive_matcher`、`mapper`
- 完成 `image_undistorter`、`patch_match_stereo`、`stereo_fusion`
- 导出稀疏点云、稠密点云、预览图和统计摘要

由于 COLMAP 官方二进制在中文路径下读取图像时可能触发 `BITMAP_ERROR`，最终实际运行路径使用 ASCII 目录：

```text
D:\DGP\hw3
```

完整运行命令如下：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\DGP\hw3\run_colmap.ps1 `
  -DatasetPath D:\DGP\hw3\data `
  -ColmapExe D:\程开良大学录\研究生录\课程录\研一下\数字图像处理\my-homework\hw3\03_BundleAdjustment\tools\colmap\bin\colmap.exe
```

运行完成后，最终结果已经同步回 `hw3/03_BundleAdjustment/data/colmap/` 和 `hw3/03_BundleAdjustment/outputs/`。

---

## Evaluation

### 任务一：Bundle Adjustment 结果

最终结果文件位于：

- `hw3/03_BundleAdjustment/outputs/final_run/metrics.json`
- `hw3/03_BundleAdjustment/outputs/final_run/loss_curve.png`
- `hw3/03_BundleAdjustment/outputs/final_run/reconstructed_points.obj`
- `hw3/03_BundleAdjustment/outputs/final_run/reprojection/`

关键指标如下：

| 指标 | 数值 |
| --- | ---: |
| Mean Reprojection Error | 4.6513 px |
| Median Reprojection Error | 0.4348 px |
| Max Reprojection Error | 813.0776 px |
| Visible Observations | 805089 |
| Best Training Loss | 4.4136 |
| Focal Length | 998.0685 |

结果分析：

- 中位数误差较低，说明大多数可见点已经被较好对齐。
- 平均误差高于中位数，表明仍存在少量离群观测。
- 从结果图看，重建点云已经恢复出较稳定的人头三维形状。

### 任务二：COLMAP 重建结果

最终结果文件位于：

- `hw3/03_BundleAdjustment/data/colmap/sparse/0/`
- `hw3/03_BundleAdjustment/data/colmap/dense/fused.ply`
- `hw3/03_BundleAdjustment/outputs/task2_sparse.ply`
- `hw3/03_BundleAdjustment/outputs/task2_sparse_preview.png`
- `hw3/03_BundleAdjustment/outputs/task2_sparse_stats.json`
- `hw3/03_BundleAdjustment/outputs/task2_dense_fused.ply`
- `hw3/03_BundleAdjustment/outputs/task2_dense_preview.png`
- `hw3/03_BundleAdjustment/outputs/task2_dense_stats.json`
- `hw3/03_BundleAdjustment/outputs/task2_summary.json`

稀疏重建结果如下：

| 指标 | 数值 |
| --- | ---: |
| Registered Images | 50 |
| Sparse Points | 1693 |
| Observations | 13592 |
| Mean Track Length | 8.0284 |
| Mean Observations per Image | 271.84 |
| Mean Reprojection Error | 0.6578 px |

稠密重建结果如下：

| 指标 | 数值 |
| --- | ---: |
| Dense Fused Points | 109574 |

结果分析：

- 50 张图像全部成功注册，说明 SfM 部分工作正常。
- 稀疏重建误差较低，位姿与几何恢复较稳定。
- 稠密融合点数达到 10 万级，说明 MVS 流程已经完整跑通。

---

## Results

### 1. Task 1：Bundle Adjustment

建议查看以下结果图：

- `hw3/03_BundleAdjustment/outputs/final_run/loss_curve.png`
- `hw3/03_BundleAdjustment/outputs/final_run/point_cloud_preview.png`
- `hw3/03_BundleAdjustment/outputs/final_run/reprojection/view_000_overlay.png`
- `hw3/03_BundleAdjustment/outputs/final_run/reprojection/view_025_overlay.png`

这些结果分别展示了：

- 优化收敛过程
- 三维点云整体形状
- 若干视角下的重投影对齐效果

### 2. Task 2：COLMAP 稀疏与稠密点云

建议查看以下结果图：

- `hw3/03_BundleAdjustment/outputs/task2_sparse_preview.png`
- `hw3/03_BundleAdjustment/outputs/task2_dense_preview.png`

其中：

- 稀疏点云反映了 COLMAP 在特征匹配与 BA 后恢复出的稳定结构
- 稠密点云反映了 `patch_match_stereo + stereo_fusion` 之后的更细致表面采样结果

---

## Project Structure

```text
hw3/
├── README.md
└── 03_BundleAdjustment/
    ├── README.md
    ├── ba_utils.py
    ├── train_ba.py
    ├── render_ply_preview.py
    ├── run_colmap.ps1
    ├── run_colmap.sh
    ├── requirements.txt
    ├── data/
    │   ├── images/
    │   ├── points2d.npz
    │   ├── points3d_colors.npy
    │   └── colmap/
    │       ├── database.db
    │       ├── sparse/0/
    │       └── dense/fused.ply
    └── outputs/
        ├── final_run/
        ├── task2_sparse.ply
        ├── task2_sparse_preview.png
        ├── task2_sparse_stats.json
        ├── task2_dense_fused.ply
        ├── task2_dense_preview.png
        ├── task2_dense_stats.json
        └── task2_summary.json
```

---

## Contributing

本仓库仅用于课程作业提交，不作为公共协作项目维护。

说明：

- `hw3/03_BundleAdjustment/tools/` 中的 COLMAP 二进制与下载压缩包体积较大，已在 `.gitignore` 中排除，不建议提交。
- `hw3/03_BundleAdjustment/outputs/smoke_run/` 和 `outputs/tune_run/` 是中间实验结果，不作为最终提交内容。
- 建议提交的核心内容为：源码、最终结果、说明文档，以及 `data/colmap/` 中的最终 sparse/dense 产物。
