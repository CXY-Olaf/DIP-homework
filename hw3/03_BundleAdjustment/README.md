# Assignment 3 - Bundle Adjustment

本目录给出 `hw3/03_BundleAdjustment` 的完整作业实现、运行方式和结果说明。作业分为两个部分：

1. 使用 PyTorch 从零实现 Bundle Adjustment，联合估计三维点、相机外参与共享焦距。
2. 使用 COLMAP 对给定多视角图像执行标准 SfM/MVS 流程，并导出稀疏与稠密重建结果。

---

## 1. 目录结构

```text
03_BundleAdjustment/
├── README.md
├── ba_utils.py
├── train_ba.py
├── render_ply_preview.py
├── visualize_data.py
├── run_colmap.sh
├── run_colmap.ps1
├── requirements.txt
├── data/
│   ├── images/
│   ├── points2d.npz
│   └── points3d_colors.npy
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

主要文件职责如下：

- `ba_utils.py`：数据读取、因子分解初始化、欧拉角旋转、OBJ 导出、可视化工具。
- `train_ba.py`：Task 1 的训练入口，完成 Bundle Adjustment 优化并保存结果。
- `run_colmap.ps1`：Windows 下的 COLMAP 脚本，补充了 Qt 插件路径修正与 CPU 模式支持。
- `render_ply_preview.py`：将 COLMAP 导出的 PLY 稀疏点云生成预览图和统计信息。

---

## 2. 数据说明

`data/` 中包含以下内容：

- `images/`：50 张输入图像，分辨率均为 `1024 x 1024`
- `points2d.npz`：50 个视角下的 2D 观测，每个视角形状为 `(20000, 3)`
- `points3d_colors.npy`：20000 个三维点对应的 RGB 颜色

`points2d.npz` 中每一行含义为：

- 前两维：像素坐标 `(x, y)`
- 第三维：可见性 `visibility`
  - `1.0` 表示该点在该视角可见
  - `0.0` 表示该点在该视角不可见或被遮挡

因此，Task 1 的本质是：利用多视角 2D 观测和可见性掩码，反推出一组三维结构和相机参数，使所有可见点的重投影误差最小。

---

## 3. Task 1：PyTorch 实现 Bundle Adjustment

### 3.1 参数化方式

待优化变量包括：

- 共享焦距 `f`
- 50 个相机的旋转和平移
- 20000 个三维点坐标

实现中采用以下参数化：

- 焦距使用 `softplus(raw_focal)` 保证始终为正
- 旋转使用 XYZ 欧拉角表示，再转换为旋转矩阵
- 第一个相机固定，只优化其余 49 个相机，减少规范自由度漂移
- 相机的 `z` 平移也用 `softplus` 参数化，并强制保持在物体前方

对应代码见 `train_ba.py` 中的 `BundleAdjustmentModel`。

### 3.2 投影模型

设三维点为 `P_j`，第 `i` 个相机的外参为 `R_i, T_i`，则相机坐标系下的点为：

```text
X_cam = R_i @ P_j + T_i
```

结合题目给定坐标系，投影公式写为：

```text
u = -f * Xc / Zc + cx
v =  f * Yc / Zc + cy
```

其中：

- `cx = 512`
- `cy = 512`
- 图像大小为 `1024 x 1024`

实现中对 `Zc` 加入了数值保护，避免训练初期出现除零和梯度爆炸。

### 3.3 初始化策略

初始化在这道题里非常关键。当前实现使用了一个基于因子分解的初始化流程：

1. 从各视角的可见点坐标构造测量矩阵
2. 对缺失观测做 EM 式低秩补全
3. 通过 SVD 得到仿射 motion / shape
4. 做 metric upgrade，将仿射结果提升到近似欧式结构
5. 得到初始三维点、相机旋转与共享深度

这部分逻辑实现在 `ba_utils.py` 的 `factorization_initialize` 中。和纯随机初始化相比，这样做更容易稳定收敛。

### 3.4 损失函数

总损失由以下几部分组成：

1. 可见点上的 Charbonnier 重投影误差
2. 深度约束，避免点跑到相机后方
3. 三维点中心正则
4. 三维点尺度正则
5. 相邻视角相机参数的平滑正则

对应形式可以写成：

```text
L = L_reproj + λ1 L_depth + λ2 L_center + λ3 L_scale + λ4 L_smooth
```

其中 `L_reproj` 只在 `visibility = 1` 的观测上计算。

### 3.5 优化策略

训练分两阶段进行：

1. 前 `freeze_points_iters` 轮先优化焦距和相机参数，冻结三维点
2. 后续再联合优化焦距、相机和三维点

这样做的目的，是先把相机参数带到合理范围，再放开高维点云变量，减少初期震荡。

### 3.6 运行命令

正式结果对应的命令如下：

```bash
python train_ba.py \
  --data-dir data \
  --output-dir outputs/final_run \
  --num-iters 700 \
  --freeze-points-iters 100 \
  --log-every 50 \
  --lr-euler 1e-2 \
  --lr-trans 8e-3 \
  --lr-points 2e-3 \
  --smooth-weight 5e-5
```

### 3.7 Task 1 结果

最终结果保存在 `outputs/final_run/` 中，关键指标如下：

- 平均重投影误差：`4.6513 px`
- 中位数重投影误差：`0.4348 px`
- 最大重投影误差：`813.0776 px`
- 可见观测总数：`805089`
- 最优训练损失：`4.4136`
- 优化后的共享焦距：`998.0685`
- 迭代轮数：`700`

从误差分布看：

- 50% 分位误差约为 `0.4348 px`
- 90% 分位误差约为 `1.4548 px`
- 95% 分位误差约为 `2.1086 px`

这说明绝大多数可见点已经被较好对齐。平均值偏大，主要由少量离群观测拉高。

### 3.8 Task 1 产出文件

`outputs/final_run/` 下已经包含：

- `loss_curve.png`：训练损失曲线
- `point_cloud_preview.png`：重建点云预览
- `reconstructed_points.obj`：彩色三维点云
- `metrics.json`：定量指标
- `ba_result.npz`：优化后的参数与预测结果
- `reprojection/`：若干视角的重投影叠加图

---

## 4. Task 2：COLMAP 稀疏与稠密重建

### 4.1 标准流程

COLMAP 的标准三维重建流程为：

1. `feature_extractor`
2. `exhaustive_matcher`
3. `mapper`
4. `image_undistorter`
5. `patch_match_stereo`
6. `stereo_fusion`

本次已经完成并验证全部 6 步，包括稀疏重建和稠密重建。

### 4.2 为什么实际运行放在 `D:\DGP\hw3`

在当前 Windows 环境下，官方 COLMAP 二进制在中文路径下对图像读取不稳定，会出现：

```text
BITMAP_ERROR: Failed to read the image file format
```

因此，Task 2 的可复现运行路径采用 ASCII 目录：

```text
D:\DGP\hw3
```

最终生成的稀疏模型和稠密点云会同步回当前作业目录的 `data/colmap/` 与 `outputs/` 中。这样做不是算法问题，而是 Windows 路径编码兼容性问题。

### 4.3 CUDA 环境与 Windows 脚本修正

2026-04-18 对本机环境重新核对后，确认 CUDA 条件是满足的：

- Conda 环境：`D:\anaconda3\envs\myenv`
- PyTorch：`torch 2.6.0+cu124`
- `torch.cuda.is_available() = True`
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
- `nvidia-smi` 显示驱动 CUDA 版本：`12.7`

`run_colmap.ps1` 做了两类必要修正：

1. 修正 Qt 插件路径，避免 COLMAP 误加载 Anaconda 的 PyQt 插件
2. 支持 `-UseCpu` 和 `-SkipDense`，便于在无 CUDA 环境下完成稀疏重建

最终完成 Task 2 时，使用的是官方 `with CUDA` 版 COLMAP。验证通过的命令如下：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\DGP\hw3\run_colmap.ps1 `
  -DatasetPath D:\DGP\hw3\data `
  -ColmapExe D:\程开良大学录\研究生录\课程录\研一下\数字图像处理\my-homework\hw3\03_BundleAdjustment\tools\colmap\bin\colmap.exe
```

### 4.4 本次重新验证后的稀疏重建结果

2026-04-18 重新清空 `D:\DGP\hw3\data\colmap` 后，使用 CUDA 版 COLMAP 完整重跑，`model_analyzer` 输出如下：

- Rigs: `1`
- Cameras: `1`
- Frames: `50`
- Registered frames: `50`
- Images: `50`
- Registered images: `50`
- Points: `1693`
- Observations: `13592`
- Mean track length: `8.028352`
- Mean observations per image: `271.840000`
- Mean reprojection error: `0.657770 px`

这说明 50 张图像全部成功注册，稀疏点云重建是完整成功的。

### 4.5 稠密重建结果

同一轮运行中，`patch_match_stereo` 和 `stereo_fusion` 也已成功完成，最终生成：

- `D:\DGP\hw3\data\colmap\dense\fused.ply`

日志中最终融合得到的点数为：

```text
Number of fused points: 109574
```

因此，Task 2 的 dense 部分现已完成，不再是缺项。

### 4.6 已导出的 Task 2 结果文件

当前作业目录下已同步以下结果：

- `data/colmap/sparse/0/`
- `data/colmap/dense/fused.ply`
- `outputs/task2_sparse.ply`
- `outputs/task2_sparse_preview.png`
- `outputs/task2_sparse_stats.json`
- `outputs/task2_dense_fused.ply`
- `outputs/task2_dense_preview.png`
- `outputs/task2_dense_stats.json`
- `outputs/task2_summary.json`

其中：

- `data/colmap/sparse/0/` 是同步回作业目录的稀疏模型文件
- `data/colmap/dense/fused.ply` 是同步回作业目录的稠密点云结果
- `outputs/task2_sparse*.{ply,png,json}` 是稀疏点云的导出与可视化
- `outputs/task2_dense*.{ply,png,json}` 是稠密点云的导出与可视化
- `outputs/task2_summary.json` 是本次最终完整运行后的摘要指标

---

## 5. Task 1 与 Task 2 的对比

### Task 1

- 优点：真正理解 BA 的建模、投影关系、初始化与优化细节
- 难点：需要自行处理数值稳定性、规范自由度和大规模参数优化

### Task 2

- 优点：可直接体验工业级 SfM 流程，结果稳定且误差小
- 难点：环境依赖更强，尤其是平台路径、Qt 插件、CUDA 等工程问题

从本次结果看：

- Task 1 体现了从零实现 BA 的建模与优化能力
- Task 2 体现了成熟重建系统的标准工程流程

两部分结合后，作业要求已经覆盖到“原理实现”和“工具链使用”两个层面。

---

## 6. 当前作业完成情况

截至 2026-04-18，本目录下的作业状态如下：

1. Task 1 已完成代码实现、正式训练和结果导出
2. Task 2 已完成 Windows 脚本修正、ASCII 路径下的可复现 COLMAP 稀疏与稠密重建，以及结果同步
3. 当前作业目录 `data/colmap/` 和 `outputs/` 中都已经包含 Task 2 的最终结果

如果按当前机器环境交作业，这份目录现在已经满足两部分作业要求：Task 1 完成，Task 2 的 sparse 和 dense 也都完成，且代码、结果和说明文档是一致的。
