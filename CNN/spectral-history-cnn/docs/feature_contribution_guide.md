# spectral-history-cnn 特征贡献分析指南

本文档说明在本项目中如何分析「哪些特征对分类有贡献」、应查看哪些文件，以及各文件的大小与用途。

---

## 1. 项目背景

`spectral-history-cnn` 将最终观测到的 64×64 RGB 图像转为 TV 残差的 log rFFT 幅度频谱，再用轻量 CNN 分类其处理历史（original / JPEG / downsample_x2 / x4 / x8 / x16）。

**模型输入：**

| 组成部分 | 通道数 | 说明 |
|----------|--------|------|
| log rFFT 幅度 | 1 | 可学习的信号特征，shape `[1, 64, 33]` |
| 频率坐标 (u, v, radius, cos θ, sin θ) | 5 | 固定 positional encoding |
| 轴向 proximity | 2 | 固定 |
| 正弦频带 (6 λ × 6 函数) | 36 | 固定 |
| **合计** | **44** | 见 `src/models/positional_encoding.py` |

在本项目中，「特征贡献」主要指 **64×33 频谱网格上哪些 (u, v) 频率位置** 对分类更重要。

> **注意：** v1 版本**没有**单独的 SHAP 报告、特征重要性 CSV 或 channel ablation 输出。README 中提到的 ablation 属于后续扩展方向。

---

## 2. 应查看的文件（按优先级）

### 2.1 频谱 Saliency 图（最重要）

**路径：** `outputs/v1_final64_poscnn/figures/saliency_*.png`

**含义：** 对输入频谱做**梯度 saliency**（`src/visualize.py` → `save_gradient_saliency()`）。显示模型在正确分类某类样本时，输入频谱各频率 bin 的梯度绝对值均值——亮度越高，该频率位置对预测越敏感。

**坐标轴：**

- 横轴：rFFT 水平频率 bin（0–32）
- 纵轴：垂直频率 bin（0–63）

**预期文件：**

| 文件 | 约大小 |
|------|--------|
| `saliency_original.png` | ~58 KB |
| `saliency_JPEG.png` | ~50–60 KB |
| `saliency_downsample_x2.png` | ~50–60 KB |
| `saliency_downsample_x4.png` | ~50–60 KB |
| `saliency_downsample_x8.png` | ~50–60 KB |
| `saliency_downsample_x16.png` | ~50–60 KB |

若部分 saliency 图缺失，重新运行可视化脚本即可生成。

---

### 2.2 各类别平均频谱

**路径：** `outputs/v1_final64_poscnn/figures/mean_spectrum_per_class/`

**含义：** 每个类别在测试/训练集上的平均 log rFFT 频谱。用于对比**类间频域差异**——哪些频段天然区分不同处理历史。

| 文件 | 约大小 |
|------|--------|
| `0_original.png` ~ `5_downsample_x16.png` | 各 ~47–52 KB |
| 目录合计 | ~304 KB |

---

### 2.3 示例频谱

**路径：** `outputs/v1_final64_poscnn/figures/example_spectra_per_class/`

**含义：** 每类若干条样本频谱拼图，辅助理解类内变异与平均频谱的对应关系。

| 文件 | 约大小 |
|------|--------|
| 6 张 PNG | 各 ~63–95 KB |
| 目录合计 | ~540 KB |

---

### 2.4 分类表现与混淆（辅助，非直接特征贡献）

| 文件 | 约大小 | 用途 |
|------|--------|------|
| `outputs/v1_final64_poscnn/metrics_test.json` | ~2.2 KB | 准确率、per-class P/R/F1、混淆矩阵、关键混淆对 |
| `outputs/v1_final64_poscnn/figures/confusion_matrix.png` | ~85 KB | 混淆矩阵可视化 |
| `outputs/v1_final64_poscnn/predictions_test.csv` | ~1.1 MB（9000 行） | 每条样本的 6 类概率及 metadata |

**研究上需重点关注的混淆对**（见 README）：

- JPEG vs downsample_x8 / x16
- downsample_x4 vs x8 / x16
- original vs JPEG / downsample_x2

---

## 3. 原始数据（自定义分析用）

若需自行做频段统计、正确/错误样本对比、混淆对频谱差异分析：

| 文件 | 约大小 | Shape / 说明 |
|------|--------|--------------|
| `data/processed/v1_final64_tv_rfft/test_spectra.npy` | 37 MB | `[9000, 1, 64, 33]` |
| `data/processed/v1_final64_tv_rfft/train_spectra.npy` | 170 MB | 训练集频谱 |
| `data/processed/v1_final64_tv_rfft/val_spectra.npy` | 37 MB | 验证集频谱 |
| `data/processed/v1_final64_tv_rfft/*_metadata.csv` | 1.7–8 MB | 裁剪、来源图等 metadata |
| `outputs/v1_final64_poscnn/checkpoints/best.pt` | 5.2 MB | 最佳模型权重 |

可结合 `predictions_test.csv` 筛选正确/错误样本，在 `test_spectra.npy` 上计算频段均值差或相关性。

---

## 4. 如何生成 / 补全可视化

在项目根目录执行：

```bash
cd spectral-history-cnn
export PYTHONPATH=.

python src/visualize.py \
  --config configs/v1_final64_poscnn.yaml \
  --checkpoint outputs/v1_final64_poscnn/checkpoints/best.pt \
  --split test \
  --saliency-per-class 128 \
  --examples-per-class 8
```

或使用评估脚本（含 evaluate + visualize）：

```bash
scripts/run_v1_eval.sh
```

**参数说明：**

- `--saliency-per-class`：每类最多使用多少条**正确分类**样本累积 saliency（默认 128）
- `--split`：使用 train / val / test 哪个划分

**输出目录：** `outputs/v1_final64_poscnn/figures/`

---

## 5. 推荐阅读顺序

1. **`mean_spectrum_per_class/`** — 先看各类在频域上的静态差异
2. **`saliency_*.png`** — 再看模型实际依赖哪些频率做决策
3. **`metrics_test.json` + `confusion_matrix.png`** — 结合 downsample 类之间的混淆理解 saliency 背景
4. **`predictions_test.csv` + `test_spectra.npy`** — 对特定混淆对做定量频段分析（需自行写脚本）

---

## 6. 相关源码

| 文件 | 作用 |
|------|------|
| `src/visualize.py` | 平均频谱、示例频谱、梯度 saliency |
| `src/utils/plots.py` | 频谱图绘制（magma colormap，频率轴标注） |
| `src/models/positional_encoding.py` | 44 通道中 43 个固定 positional 通道的定义 |
| `src/models/spectral_positional_cnn.py` | 频谱 + positional 拼接后进 CNN |
| `src/processing/spectrum.py` | TV 残差 → rFFT → log magnitude 预处理 |

---

## 7. 尚未实现的可扩展分析

README 建议的后续方向（v1 无现成输出）：

- positional encoding ablation（去掉部分 positional 通道看精度变化）
- residual ablation（TV / Laplacian / rank residual 对比）
- 第一层 Conv 权重可视化（从 `best.pt` 提取）
- 将 saliency 数值导出为 CSV，按频率 bin 排序

---

## 8. 目录与体积速查

```
spectral-history-cnn/
├── data/processed/v1_final64_tv_rfft/     # ~254 MB
│   ├── train_spectra.npy                  # 170 MB
│   ├── val_spectra.npy                    # 37 MB
│   ├── test_spectra.npy                   # 37 MB
│   └── *_metadata.csv, *_labels.npy
│
└── outputs/v1_final64_poscnn/             # ~13 MB
    ├── checkpoints/best.pt                # 5.2 MB
    ├── metrics_test.json                  # 2.2 KB
    ├── predictions_test.csv               # 1.1 MB
    └── figures/
        ├── saliency_*.png                 # 各 ~50–60 KB
        ├── mean_spectrum_per_class/       # ~304 KB
        ├── example_spectra_per_class/     # ~540 KB
        └── confusion_matrix.png           # ~85 KB
```

---

*文档生成日期：2026-06-17 · 对应配置：`configs/v1_final64_poscnn.yaml`*
