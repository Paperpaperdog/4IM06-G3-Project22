# 4IM06-G3-Project22 — Image Forensics

Telecom Paris IM06 课程项目：从频域/残差域痕迹判断图像是否经过 **JPEG 压缩**、**重采样（resize）** 或其它后处理，并探索 **Fourier ambiguity**（不同处理历史产生相似频谱峰值）的可分性。

本仓库整合了三位成员各自分支的工作，形成三条互补技术路线：

| 路线 | 分支来源 | 目录 / 入口 | 侧重点 |
|------|----------|-------------|--------|
| **A. 经典检测** | `zzy_raise100_resized_dataset` + `test` | 根目录 `.py` | NFA 谱相关 + 候选尺寸估计；**JPEG vs ×8 重采样** a contrario 判别 |
| **B. 可学习 Mask** | `zzy_raise100_resized_dataset`（代码）+ `xby-branch`（结果） | `spectral-mask-resampling/` | 频域重采样 + 每类 mask/reference；JPEG vs ×8/×16 歧义 |
| **C. 频谱 CNN** | `xby-branch` | `CNN/spectral-history-cnn/` | 位置编码 CNN；处理历史多类分类 |

**实验总结（深度学习）**：见 [`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md)  
**周会记录**：见 [`SUIVI.md`](SUIVI.md)

---

## 路线 A：经典重采样 / 压缩检测

### A1. 模块化 NFA 管线（`zzy_raise100_resized_dataset`）

基于 `resampling_detection.pdf` 的 clean-room 实现，侧重 **候选原图尺寸估计**。

| 文件 | 说明 |
|------|------|
| `resampling_core.py` | TV 残差、傅里叶谱、复 Pearson 相关、NFA |
| `candidate_estimation.py` | 由 NFA 峰值生成并排序候选原图尺寸 |
| `demo_resampling_detection.py` | 单图 demo（skimage camera 或本地图） |
| `synthesize_controlled_resampling_dataset.py` | RAISE TIFF 受控重采样数据集合成 |
| `run_detector_on_synthesized_dataset.py` | 批量检测已合成数据集 |
| `run_controlled_resampling_experiments.py` | 小规模一键合成+检测 |
| `detect_resampling.py` | 精简 CLI（`RESAMPLED` / `NOT_RESAMPLED` + NFA 图） |
| `RAISE_1k.csv` | RAISE 子集元数据与 TIFF 下载链接 |

快速 demo：

```bash
.venv/bin/python demo_resampling_detection.py
.venv/bin/python demo_resampling_detection.py path/to/image.png
python detect_resampling.py --image path/to/image.png --outdir outputs
```

受控数据集（100 张 RAISE，bicubic，4500 目标图）：

```bash
.venv/bin/python synthesize_controlled_resampling_dataset.py 100 --download
.venv/bin/python run_detector_on_synthesized_dataset.py \
  test_results/controlled_resampling_dataset_bicubic_raise100/metadata.csv
```

结果摘要（已入库）：`test_results/controlled_resampling_dataset_bicubic_raise100/detection_summary.csv`

### A2. JPEG vs ×8 重采样管线（`test` 分支，2026-06-16 最新提交）

> **`test` 分支本次整合的核心**：三条脚本组成完整取证实验闭环（数据集合成 → 单图检测 → 批量评估）。

| 文件 | 说明 |
|------|------|
| `create_forensic_postprocess_dataset.py` | 由原始图批量生成 `original` / `jpeg` / `resample_x8` / `mix` 四类 PNG |
| `jpeg_resample_detector.py` | 残差 + DCT/FFT 特征 + a contrario NFA，判别 JPEG / ×8 重采样 / 混合 / 原图 |
| `evaluate_detector_on_dataset.py` | 在合成数据集上批量调用检测器，输出准确率与混淆矩阵 |

**方法要点**（`jpeg_resample_detector.py`）：

1. 灰度图 → 预测残差 → DCT 块效应与 FFT 周期特征
2. 用 `null_dir` 中干净图（或相位随机 surrogate）建立零假设分布
3. 对 JPEG / 重采样特征分别算 NFA 与加权得分，输出 `Label: ...`

**典型流程**：

```bash
# 1) 合成取证数据集（×8 块级重采样 + JPEG Q85 + 混合顺序）
python create_forensic_postprocess_dataset.py \
  --input_dir path/to/raw_images \
  --output_dir dataset_x8 \
  --include_original \
  --mix_order both

# 2) 单图检测
python jpeg_resample_detector.py \
  --image dataset_x8/jpeg/example_jpeg_q85.png \
  --null_dir dataset_x8/original

# 3) 在 test split 上评估
python evaluate_detector_on_dataset.py \
  --detector jpeg_resample_detector.py \
  --dataset_root dataset_x8 \
  --split test \
  --null_dir dataset_x8/train/original \
  --max_per_class 50
```

输出标签包括：`jpeg_compression`、`8x8_resampling`、`jpeg_and_8x8_resampling`、`original_or_uncertain` 等。

### A3. 早期独立工具（`test` 分支历史提交，非本次整合重点）

| 文件 | 说明 |
|------|------|
| `ResamplingDetector.py` | 自包含 NFA 重采样 CLI（rank/TV 预处理） |
| `spai_detector_new.py` | 手工取证特征 + Random Forest（5 类） |
| `img/` | 演示图（baboon、pashmina） |

文档：[`README_ResamplingDetector.md`](README_ResamplingDetector.md)、[`README_spai_detector.md`](README_spai_detector.md)

---

## 路线 B：Spectral Mask（`spectral-mask-resampling/`）

针对 **Fourier ambiguity**：不同原始尺寸经不同路径可能落到同一观测尺寸，频谱峰值位置相似。

- **代码**：`spectral-mask-resampling/src/`（训练、评估、可视化）
- **配置**：`spectral-mask-resampling/configs/v1_fourier_ambiguity_mask.yaml`
- **最佳结果**：`outputs/v1_fourier_ambiguity_mask_clean/`（测试准确率 **56.6%**，macro F1 **0.561**）
- **汇总图**：`outputs/v1_fourier_ambiguity_mask_clean/figures/summary/`

```bash
cd spectral-mask-resampling
bash scripts/run_v1_prepare.sh
bash scripts/run_v1_train.sh
bash scripts/run_v1_eval.sh
python scripts/plot_mask_results.py   # 重新生成汇总图
```

详见 [`spectral-mask-resampling/README.md`](spectral-mask-resampling/README.md)。

---

## 路线 C：Spectral CNN（`CNN/spectral-history-cnn/`）

带频率位置编码的轻量 CNN，固定 64×64 观测、原生 64×33 频谱。

```bash
cd CNN/spectral-history-cnn
bash scripts/run_v1_pipeline_full.sh
```

- **6 类结果**（已完成）：`outputs/v1_final64_poscnn/`
- **4 类配置**（与 Mask 对齐）：`configs/v1_final64_poscnn_local.yaml`

详见 [`CNN/spectral-history-cnn/README.md`](CNN/spectral-history-cnn/README.md)。

---

## 共享数据与协议

- RAISE-1K：按源图划分 train/val/test = **700 / 150 / 150**
- `scripts/build_protocol_dataset.py`、`scripts/preprocess_raise.py`：协议数据集构建（`xby-branch`）
- 本地 TIFF 缓存默认在 `spectral-mask-resampling/data/raw/raise_tiff/`（不入库）

---

## 结果目录约定

| 路径 | 内容 | 是否入库 |
|------|------|----------|
| `test_results/` | 经典 NFA 受控实验 CSV 摘要 | 部分 CSV |
| `spectral-mask-resampling/outputs/v1_fourier_ambiguity_mask_clean/` | Mask 最佳实验 json/csv/png | 是 |
| `CNN/spectral-history-cnn/outputs/` | CNN 指标与图表 | 部分 |
| `data/`、`checkpoints/`、`*.pt`、`*.npy` | 原始数据与大文件 | 否 |

大文件请本地生成或通过外部网盘共享；仓库仅保留可复现的代码与轻量结果摘要。

---

## 分支说明

| 原分支 | 本仓库中的保留 |
|--------|----------------|
| `xby-branch` | CNN 管线、Mask 实验结果、`EXPERIMENT_SUMMARY.md` |
| `zzy_raise100_resized_dataset` | Mask 源码、RAISE 受控数据集管线、NFA 核心实现 |
| `test` | **JPEG vs ×8 重采样三脚本管线**（`create_forensic_postprocess_dataset.py`、`jpeg_resample_detector.py`、`evaluate_detector_on_dataset.py`）；早期工具见 A3 |

整合分支：`project-integration`
