# 4IM06-G3-Project22 — Image Forensics

Telecom Paris IM06 课程项目：从频域/残差域痕迹判断图像是否经过 **JPEG 压缩**、**重采样（resize）** 或其它后处理，并探索 **Fourier ambiguity**（不同处理历史产生相似频谱峰值）的可分性。

整合分支：`project-integration`（合并 `main`、`zzy_raise100_resized_dataset`、`test`、`xby-branch`）

| 路线 | 目录 | 一句话 |
|------|------|--------|
| **A. 经典检测** | 根目录 `pilots/`、`.py` | NFA 复现 + JPEG/×8 统计判别 |
| **B. Mask** | `spectral-mask-resampling/` | 可学习频域 mask，4 类分类 |
| **C. CNN** | `CNN/spectral-history-cnn/` | 位置编码频谱 CNN |

**实验总结（DL）**：[`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md) · **n6 运行汇总（配置/脚本/命令）**：[`docs/EXPERIMENT_RUNBOOK.md`](docs/EXPERIMENT_RUNBOOK.md) · **W3 先导报告**：[`REPORT.zh.md`](REPORT.zh.md) · **周会**：[`SUIVI.md`](SUIVI.md)

**详细方法文档**（流程图 + 数据处理 + 我们的实验）：

| 文档 | 内容 |
|------|------|
| [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) | **总体实验报告**（摘要、三路线、对比、结论） |
| [`docs/SUPPLEMENTARY_EXPERIMENTS.md`](docs/SUPPLEMENTARY_EXPERIMENTS.md) | **补充实验清单**（命令、表格模板、执行顺序） |
| [`docs/00_project_overview.md`](docs/00_project_overview.md) | 原项目、复现时间线、主要发现汇总 |
| [`docs/01_classical_detection.md`](docs/01_classical_detection.md) | 经典检测（pilots / NFA / JPEG×8） |
| [`docs/02_spectral_mask.md`](docs/02_spectral_mask.md) | Mask 路线全流程 |
| [`docs/03_spectral_cnn.md`](docs/03_spectral_cnn.md) | CNN 路线全流程 |

---

## 我该用哪个脚本？

| 目标 | 入口 | 命令示例 |
|------|------|----------|
| 复现 W3 先导（JPEG vs 重采样混淆） | `run_pilots.py` | `python run_pilots.py --max-images 5` |
| 单图 NFA 重采样检测 + NFA 曲线 | `demo_resampling_detection.py` | `python demo_resampling_detection.py img.png` |
| RAISE 受控数据集 + 候选原图尺寸 | `synthesize_controlled_resampling_dataset.py` | `python synthesize_controlled_resampling_dataset.py 100 --download` |
| JPEG vs ×8 块重采样（DCT/FFT） | `jpeg_resample_detector.py` | 见下方 A2 |
| 训练 / 评估 Mask 分类器（n6） | `spectral-mask-resampling/` | `CONFIG=configs/size_sweep/n6_mask_size64.yaml bash scripts/run_pipeline_config.sh` |
| 训练 / 评估 CNN | `CNN/spectral-history-cnn/` | `bash scripts/run_v1_pipeline_full.sh` |
| 早期独立工具（仅供参考） | `archive/legacy_test_tools/` | 见 archive README |

**RAISE 索引（唯一路径）**：[`data/raise_raw/RAISE_1k.csv`](data/raise_raw/RAISE_1k.csv)  
**参考论文**：[Resampling Detection](https://bammey.com/resampling_detection.pdf)（本地不存 PDF，请在线阅读）

---

## 路线 A：经典检测

### A0. W3 先导（`main`）

`pilots/` + `run_pilots.py`，基于 `resampling_core.py`。

```bash
python run_pilots.py --max-images 5
```

结论摘要：`data/pilot_results/PILOT_SUMMARY.md` · 全文：`REPORT.zh.md`

### A1. 模块化 NFA（`zzy`）

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

### A2. JPEG vs ×8（`test`，2026-06-16）

| 文件 | 作用 |
|------|------|
| `create_forensic_postprocess_dataset.py` | 合成 original/jpeg/resample_x8/mix |
| `jpeg_resample_detector.py` | 单图检测，输出 `Label:` |
| `evaluate_detector_on_dataset.py` | 数据集批量评估 |

```bash
python create_forensic_postprocess_dataset.py --input_dir RAW --output_dir dataset_x8 --include_original --mix_order both
python jpeg_resample_detector.py --image IMG --null_dir dataset_x8/original
python evaluate_detector_on_dataset.py --detector jpeg_resample_detector.py --dataset_root dataset_x8 --split test --null_dir dataset_x8/train/original
```

### 归档

早期 `ResamplingDetector` / `spai` / 重复 CLI / NFA 示意图 → [`archive/`](archive/)

---

## 路线 B：Mask（`spectral-mask-resampling/`）

当前主协议 `n6`：原生 rFFT 谱、6 类、每个观测尺寸单独训练。

- **配置**：`configs/size_sweep/n6_mask_size{32,64,96,128}.yaml`
- **输出**：`results/mask/n6_mask_size*`

```bash
cd spectral-mask-resampling
# 单尺寸完整管线
CONFIG=configs/size_sweep/n6_mask_size64.yaml bash scripts/run_pipeline_config.sh
# 全部尺寸
bash scripts/run_size_sweep.sh
```

历史基线 v1（512×257 网格、4 类，acc **56.6%** / macro F1 **0.561**）结果保留在
`outputs/v1_fourier_ambiguity_mask_clean/`，配置与脚本见 `archive/legacy-u6-u7` 分支。

详见 [`spectral-mask-resampling/README.md`](spectral-mask-resampling/README.md)。

---

## 路线 C：CNN（`CNN/spectral-history-cnn/`）

```bash
cd CNN/spectral-history-cnn && bash scripts/run_v1_pipeline_full.sh
```

- 6 类结果：`outputs/v1_final64_poscnn/`
- 4 类配置：`configs/v1_final64_poscnn_local.yaml`

详见 [`CNN/spectral-history-cnn/README.md`](CNN/spectral-history-cnn/README.md)。

---

## 数据与入库约定

| 路径 | 入库 |
|------|------|
| `data/raise_raw/RAISE_1k.csv` | 是（索引，**唯一路径**） |
| `data/manifest.csv`、`data/pilot_results/PILOT_SUMMARY.md` | 是 |
| `test_results/**/*.csv`（经典实验摘要） | 是（含 `nfa_candidate_topk_summary.csv`） |
| `scripts/analysis/` | 是（E2/E4 补充分析） |
| `archive/early_nfa_demos/` | 是（早期 NFA 示意图） |
| `spectral-mask-resampling/outputs/v1_fourier_ambiguity_mask_clean/` | 是（json/csv/png） |
| `CNN/spectral-history-cnn/outputs/v1_final64_poscnn/` | 是（指标、图、csv；不含 `*.pt`） |
| 本地 `data/RAISE_1k.csv`、`data/*_png/`、TIFF、checkpoint | 否 |
