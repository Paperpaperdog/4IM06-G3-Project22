# 4IM06-G3-Project22 — Image Forensics

Telecom Paris IM06 课程项目：从频域/残差域痕迹判断图像是否经过 **JPEG 压缩**、**重采样（resize）** 或其它后处理，并探索 **Fourier ambiguity**（不同处理历史产生相似频谱峰值）的可分性。

**提交分支**：`final-submission`（整合复现 + Feature / Mask / CNN 三方法，来源 `project-integration`）

| 模块 | 目录 | 说明 |
|------|------|------|
| **复现** | `pilots/`、`resampling_core.py`、NFA 脚本 | W3 先导 + NFA 重采样检测复现 |
| **Feature** | 根目录 `jpeg_resample_detector.py` 等 | DCT/FFT 统计特征判别（经典路线） |
| **Mask** | `spectral-mask-resampling/` | 可学习频域 mask，n6 六类分类 |
| **CNN** | `CNN/spectral-history-cnn/` | 位置编码频谱 CNN，n6 六类分类 |

**一键复现手册**：[`docs/EXPERIMENT_RUNBOOK.md`](docs/EXPERIMENT_RUNBOOK.md)（前置准备 → 经典 / Mask / CNN → 对比汇总）  
**实验总结（DL）**：[`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md) · **W3 先导报告**：[`REPORT.zh.md`](REPORT.zh.md)

**详细方法文档**：

| 文档 | 内容 |
|------|------|
| [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) | 总体实验报告（摘要、三方法、对比、结论） |
| [`docs/SUPPLEMENTARY_EXPERIMENTS.md`](docs/SUPPLEMENTARY_EXPERIMENTS.md) | 补充实验清单 |
| [`docs/00_project_overview.md`](docs/00_project_overview.md) | 项目概览与主要发现 |
| [`docs/01_classical_detection.md`](docs/01_classical_detection.md) | 复现 + Feature 路线 |
| [`docs/02_spectral_mask.md`](docs/02_spectral_mask.md) | Mask 路线 |
| [`docs/03_spectral_cnn.md`](docs/03_spectral_cnn.md) | CNN 路线 |

---

## 快速入口

| 目标 | 入口 | 命令示例 |
|------|------|----------|
| 复现 W3 先导 | `run_pilots.py` | `python run_pilots.py --max-images 5` |
| 单图 NFA 重采样检测 | `demo_resampling_detection.py` | `python demo_resampling_detection.py img.png` |
| RAISE 受控数据集 | `synthesize_controlled_resampling_dataset.py` | `python synthesize_controlled_resampling_dataset.py 100 --download` |
| Feature：JPEG vs 重采样 | `jpeg_resample_detector.py` | 见下方 Feature 节 |
| Feature：完整经典管线 | `scripts/analysis/run_classical_pipeline.sh` | `bash scripts/analysis/run_classical_pipeline.sh` |
| Mask 训练/评估（n6） | `spectral-mask-resampling/` | `CONFIG=configs/size_sweep/n6_mask_size64.yaml bash scripts/run_pipeline_config.sh` |
| CNN 训练/评估（n6） | `CNN/spectral-history-cnn/` | `CONFIG=configs/size_sweep/n6_poscnn_size64.yaml bash scripts/run_v1_pipeline_full.sh` |
| 三方法对比汇总 | `scripts/analysis/unified_method_comparison.py` | 见 RUNBOOK §E |

**RAISE 索引**：[`data/raise_raw/RAISE_1k.csv`](data/raise_raw/RAISE_1k.csv)  
**谱缓存说明**（不入库）：[`data/processed/README.md`](data/processed/README.md)  
**共享 n6 协议**：[`experiments/unified_protocol.py`](experiments/unified_protocol.py)

---

## 复现：W3 先导 + NFA

### W3 先导

`pilots/` + `run_pilots.py`，基于 `resampling_core.py`。

```bash
python run_pilots.py --max-images 5
```

结论：`data/pilot_results/PILOT_SUMMARY.md` · 全文：`REPORT.zh.md`

### NFA 模块化检测

| 文件 | 作用 |
|------|------|
| `resampling_core.py` | 谱相关 + NFA 核心 |
| `candidate_estimation.py` | 候选原图尺寸排序 |
| `demo_resampling_detection.py` | 单图 demo |
| `synthesize_controlled_resampling_dataset.py` | RAISE 受控合成 |
| `run_detector_on_synthesized_dataset.py` | 批量检测 |

```bash
python demo_resampling_detection.py path/to/image.png
python synthesize_controlled_resampling_dataset.py 100 --download
python run_detector_on_synthesized_dataset.py \
  test_results/controlled_resampling_dataset_bicubic_raise100/metadata.csv
```

---

## Feature：DCT/FFT 统计检测

| 文件 | 作用 |
|------|------|
| `create_forensic_postprocess_dataset.py` | 合成 original/jpeg/resample/upsample/mix |
| `jpeg_resample_detector.py` | 单图检测 |
| `evaluate_detector_on_dataset.py` | 数据集批量评估 |

```bash
python create_forensic_postprocess_dataset.py --input_dir RAW --output_dir dataset_x8 --include_original --mix_order both
python jpeg_resample_detector.py --image IMG --null_dir dataset_x8/original
python evaluate_detector_on_dataset.py --detector jpeg_resample_detector.py --dataset_root dataset_x8 --split test --null_dir dataset_x8/train/original
```

完整尺寸扫描 + 三方法对比：`bash scripts/analysis/run_classical_pipeline.sh` → `results/classical/`

早期独立 3 类 fork（仅供参考）：[`archive/features_detector_fork/`](archive/features_detector_fork/)

---

## Mask（`spectral-mask-resampling/`）

主协议 **n6**：原生 rFFT 谱、6 类、每个观测尺寸单独训练。

- **配置**：`configs/size_sweep/n6_mask_size{32,64,96,128}.yaml`
- **输出**：`results/mask/n6_mask_size*`

```bash
cd spectral-mask-resampling
CONFIG=configs/size_sweep/n6_mask_size64.yaml bash scripts/run_pipeline_config.sh
bash scripts/run_size_sweep.sh   # 四尺寸
```

详见 [`spectral-mask-resampling/README.md`](spectral-mask-resampling/README.md)。

---

## CNN（`CNN/spectral-history-cnn/`）

主协议 **n6**：6 类、原生 rFFT 谱、每尺寸单独训练。

```bash
cd CNN/spectral-history-cnn
CONFIG=configs/size_sweep/n6_poscnn_size64.yaml bash scripts/run_v1_pipeline_full.sh
bash scripts/run_size_sweep.sh   # 四尺寸
```

- **配置**：`configs/size_sweep/n6_poscnn_size{32,64,96,128}.yaml`
- **输出**：`results/cnn/n6_poscnn_size*`

详见 [`CNN/spectral-history-cnn/README.md`](CNN/spectral-history-cnn/README.md)。

---

## 结果目录

```text
results/
├── classical/          # Feature 路线输出
├── mask/               # n6_mask_size*
├── cnn/                # n6_poscnn_size*
└── comparison/         # 三方法 / 尺寸效应对比图
```

---

## 数据与入库约定

| 路径 | 入库 |
|------|------|
| `data/raise_raw/RAISE_1k.csv` | 是（索引） |
| `data/manifest.csv`、`data/pilot_results/PILOT_SUMMARY.md` | 是 |
| `test_results/**/*.csv` | 是（经典实验摘要） |
| `scripts/analysis/` | 是 |
| `results/`（metrics、figures、csv） | 是（checkpoint 默认忽略） |
| `data/processed/` 谱缓存、TIFF、`.venv` | 否 |

归档参考：[`archive/`](archive/)（早期 NFA demo、legacy CLI、features fork）
