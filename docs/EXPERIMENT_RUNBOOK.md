# n6 实验运行汇总

> 当前主协议：**n6**（6 类 × 4 观测尺寸，Mask/CNN 原生 rFFT 谱，每尺寸单独训练）  
> 所有**输出**统一写入项目根目录 `results/`，按方法分子文件夹。  
> 谱缓存统一在项目根 `data/processed/n6_spectra_size{N}/`（Mask 与 CNN **共用**，体积大，不入库）。

---

## 0. 协议速查

| 项目 | 内容 |
|------|------|
| **6 类** | `original` / `JPEG_Q80` / `downsample_x8` / `downsample_x16` / `upsample_x4` / `upsample_x8` |
| **观测尺寸** | 32 / 64 / 96 / 128（每个尺寸 = 独立模型 + 独立缓存 + 独立结果目录） |
| **Mask 谱** | 原生 rFFT `(o, o//2+1)`；DC 抑制 → `log1p(abs(F))`（与 CNN 一致） |
| **CNN 谱** | 原生 rFFT `(o, o//2+1)` + 位置编码（44 通道） |
| **经典路线** | DCT-FFT 检测器 +（可选）NFA 源尺寸恢复；与 B/C 只在「是否重采样」二分轴上可比 |
| **协议源码** | `experiments/unified_protocol.py`（含 `make_aligned_observed_patch` / `per_sample_seed`） |
| **Mask/CNN 样本对齐** | 同一 split JSON、同一 seed、同一每类样本数；逐样本确定性 RNG |
| **共享 split** | `spectral-mask-resampling/data/splits/raise_split_seed123.json` |

各尺寸原生谱形状：

| 观测尺寸 \(o\) | 32 | 64 | 96 | 128 |
|----------------|-----|-----|-----|-----|
| 谱 \((H, W_{rfft})\) | (32, 17) | (64, 33) | (96, 49) | (128, 65) |

---

## 1. 输出目录结构

所有实验产物写入 **`4IM06-G3-Project22/results/`**：

```text
results/
├── classical/
│   ├── forensic_pp/              # 取证后处理数据集（original/jpeg/resample/upsample/mix）
│   ├── jpeg_detector_size_sweep/ # DCT-FFT 检测器尺寸扫描 eval_size{N}.json
│   ├── classical_size_sweep/     # NFA 源尺寸恢复扫描（可选）
│   └── pipeline_logs/            # run_classical_pipeline.sh 日志
├── mask/
│   └── n6_mask_size{32,64,96,128}/   # metrics.json, checkpoints/, figures/
├── cnn/
│   └── n6_poscnn_size{32,64,96,128}/ # metrics.json, checkpoints/, figures/
└── comparison/
    ├── size_effect/              # Mask vs CNN 6 类准确率-尺寸曲线
    └── unified_comparison/         # 三方法二分轴对比
```

---

## 2. 前置准备（所有路线共用）

```bash
cd 4IM06-G3-Project22

# RAISE TIFF 缓存（Mask/CNN/经典管线均依赖）
bash spectral-mask-resampling/scripts/download_raise_tiff.sh
# 或手动放到 spectral-mask-resampling/data/raw/raise_tiff/
```

---

## 3. 实验总表

| # | 实验 | 配置 | 脚本 | 主要输出 |
|---|------|------|------|----------|
| **A** | 经典：取证数据集 + DCT-FFT 尺寸扫描 +（可选）NFA + 三方法对比 | （无单一 yaml；见 §4） | `scripts/analysis/run_classical_pipeline.sh` | `results/classical/*`, `results/comparison/unified_comparison/` |
| **B1–B4** | Mask：单尺寸完整管线 | `spectral-mask-resampling/configs/size_sweep/n6_mask_size{N}.yaml` | `spectral-mask-resampling/scripts/run_pipeline_config.sh` | `results/mask/n6_mask_size{N}/` |
| **B-all** | Mask：四尺寸顺序跑 | 同上 4 个 yaml | `spectral-mask-resampling/scripts/run_size_sweep.sh` | 同上 ×4 |
| **B-HPC** | Mask：集群 NPU 提交 | 同上 4 个 yaml | `spectral-mask-resampling/scripts/submit_size_sweep_npu.sh` | 同上 ×4 |
| **C1–C4** | CNN：单尺寸完整管线 | `CNN/spectral-history-cnn/configs/size_sweep/n6_poscnn_size{N}.yaml` | `CNN/spectral-history-cnn/scripts/run_v1_pipeline_full.sh` | `results/cnn/n6_poscnn_size{N}/` |
| **C-all** | CNN：四尺寸顺序跑 | 同上 4 个 yaml | `CNN/spectral-history-cnn/scripts/run_size_sweep.sh` | 同上 ×4 |
| **C-HPC** | CNN：集群 NPU 提交 | 同上 4 个 yaml | `CNN/spectral-history-cnn/scripts/submit_size_sweep_npu.sh` | 同上 ×4 |
| **D** | 汇总：Mask vs CNN 6 类 | — | `scripts/analysis/summarize_size_effect.py` | `results/comparison/size_effect/` |
| **E** | 汇总：三方法二分轴 | — | `scripts/analysis/unified_method_comparison.py` | `results/comparison/unified_comparison/` |

> **推荐顺序**：先跑 **B + C**（或 HPC 并行提交）→ 再跑 **A**（经典，CPU）→ 最后 **D + E**（需 B/C 的 `metrics.json`；E 还需 A 的 `jpeg_detector_size_sweep`）。

---

## 4. 路线 A：经典检测（CPU）

### 4.1 一键管线（推荐）

```bash
cd 4IM06-G3-Project22

# 前台
bash scripts/analysis/run_classical_pipeline.sh

# 后台 + 日志
bash scripts/analysis/run_classical_pipeline.sh --detach
tail -f results/classical/pipeline_logs/latest/pipeline.log
```

管线内 4 步：

| 步骤 | 做什么 | 输出 |
|------|--------|------|
| 1 | `create_forensic_postprocess_dataset.py` 生成取证集（上采样因子 4,8） | `results/classical/forensic_pp/` |
| 2 | `jpeg_detector_size_sweep.py` DCT-FFT 尺寸扫描 | `results/classical/jpeg_detector_size_sweep/` |
| 3 | `classical_size_sweep.py` NFA 源尺寸恢复（可选） | `results/classical/classical_size_sweep/` |
| 4 | `unified_method_comparison.py` 三方法对比（需 B/C 已有结果） | `results/comparison/unified_comparison/` |

常用环境变量：

```bash
# 快速试跑（限制图像数）
FORENSIC_LIMIT=100 LIMIT_IMAGES=20 bash scripts/analysis/run_classical_pipeline.sh

# 跳过 NFA 或统一对比
SKIP_NFA=1 bash scripts/analysis/run_classical_pipeline.sh
SKIP_UNIFIED=1 bash scripts/analysis/run_classical_pipeline.sh   # B/C 未跑完时先用这个
```

### 4.2 分步手动（调试）

```bash
cd 4IM06-G3-Project22

# Step 1：取证数据集
python create_forensic_postprocess_dataset.py \
  --input_dir spectral-mask-resampling/data/raw/raise_tiff \
  --output_dir results/classical/forensic_pp \
  --include_original --include_upsampling --mix_order both \
  --upsample_factors 4,8

# Step 2：DCT-FFT 尺寸扫描（与 n6 尺寸轴对齐）
python scripts/analysis/jpeg_detector_size_sweep.py \
  --dataset-root results/classical/forensic_pp \
  --null-dir results/classical/forensic_pp/original \
  --max-sizes 32,64,96,128 \
  --workers 0 \
  --outdir results/classical/jpeg_detector_size_sweep

# Step 3（可选）：NFA 源尺寸恢复
python scripts/analysis/classical_size_sweep.py \
  --image-dir spectral-mask-resampling/data/raw/raise_tiff \
  --limit-images 20 \
  --target-sizes 32,64,96,128 \
  --workers 0 \
  --outdir-root results/classical/classical_size_sweep
```

---

## 5. 路线 B：Spectral Mask（NPU / 交互节点）

### 5.1 配置一览

| 尺寸 | 配置文件 | 数据缓存 | 结果目录 |
|------|----------|----------|----------|
| 32 | `spectral-mask-resampling/configs/size_sweep/n6_mask_size32.yaml` | `data/processed/n6_spectra_size32` | `results/mask/n6_mask_size32` |
| 64 | `.../n6_mask_size64.yaml` | `data/processed/n6_spectra_size64` | `results/mask/n6_mask_size64` |
| 96 | `.../n6_mask_size96.yaml` | `data/processed/n6_spectra_size96` | `results/mask/n6_mask_size96` |
| 128 | `.../n6_mask_size128.yaml` | `data/processed/n6_spectra_size128` | `results/mask/n6_mask_size128` |

训练超参（各尺寸相同）：AdamW lr=1e-3，batch=64，**epochs=30**，device=npu。

### 5.2 预处理（与 CNN 共用，每尺寸一遍）

推荐在项目根执行（Mask/CNN 读同一目录）：

```bash
cd 4IM06-G3-Project22
SIZE=64 bash scripts/prepare_n6_spectra.sh
SIZES="32 64 96 128" bash scripts/prepare_n6_spectra.sh
```

等价于在 mask 子目录：`CONFIG=configs/size_sweep/n6_mask_size64.yaml bash scripts/run_prepare_config.sh`

### 5.3 单尺寸完整管线

```bash
cd spectral-mask-resampling

# prepare → train → eval → viz（若共享缓存已存在则跳过 prepare）
CONFIG=configs/size_sweep/n6_mask_size64.yaml \
  bash scripts/run_pipeline_config.sh

# 已有缓存，跳过 prepare
SKIP_PREPARE=1 CONFIG=configs/size_sweep/n6_mask_size64.yaml \
  bash scripts/run_pipeline_config.sh

# 只重跑 eval + viz（需已有 checkpoints/best.pt）
EVAL_ONLY=1 CONFIG=configs/size_sweep/n6_mask_size64.yaml \
  bash scripts/run_pipeline_config.sh
```

### 5.4 仅预处理（CPU，mask 子目录方式）

```bash
cd spectral-mask-resampling

CONFIG=configs/size_sweep/n6_mask_size64.yaml \
  bash scripts/run_prepare_config.sh

PREP_WORKERS=0 CONFIG=configs/size_sweep/n6_mask_size64.yaml \
  bash scripts/run_prepare_config.sh
```

### 5.5 四尺寸顺序跑（交互节点）

```bash
cd spectral-mask-resampling
bash scripts/run_size_sweep.sh

# 只跑部分尺寸
SIZES="64 128" bash scripts/run_size_sweep.sh
```

### 5.6 集群 NPU 提交

```bash
cd spectral-mask-resampling

# 需先把 scripts/vc_mask.sh 复制到集群 $CODES（与 vc_cnn_spectral_v1.sh 并列）
REPO_ROOT=/path/to/4IM06-G3-Project22 \
CODES=/path/to/Codes \
  bash scripts/submit_size_sweep_npu.sh

# 已有缓存
SKIP_PREPARE=1 bash scripts/submit_size_sweep_npu.sh

# 只重评估
EVAL_ONLY=1 bash scripts/submit_size_sweep_npu.sh
```

每个尺寸提交一个 vc 作业，作业名 `n6_mask_size{N}`，配置 `configs/size_sweep/n6_mask_size{N}.yaml`。

### 5.7 本地冒烟

```bash
cd spectral-mask-resampling

LIMIT_IMAGES=4 SAMPLES_PER_CLASS_PER_SIZE=8 \
  CONFIG=configs/size_sweep/n6_mask_size64.yaml \
  bash scripts/run_pipeline_config.sh
```

---

## 6. 路线 C：Spectral CNN（NPU / 交互节点）

### 6.1 配置一览

| 尺寸 | 配置文件 | 数据缓存 | 结果目录 |
|------|----------|----------|----------|
| 32 | `CNN/spectral-history-cnn/configs/size_sweep/n6_poscnn_size32.yaml` | `data/processed/n6_spectra_size32` | `results/cnn/n6_poscnn_size32` |
| 64 | `.../n6_poscnn_size64.yaml` | `data/processed/n6_spectra_size64` | `results/cnn/n6_poscnn_size64` |
| 96 | `.../n6_poscnn_size96.yaml` | `data/processed/n6_spectra_size96` | `results/cnn/n6_poscnn_size96` |
| 128 | `.../n6_poscnn_size128.yaml` | `data/processed/n6_spectra_size128` | `results/cnn/n6_poscnn_size128` |

训练超参（各尺寸相同）：AdamW lr=3e-4，batch=256，**epochs=50**，AMP，device=npu。

### 6.2 单尺寸完整管线（主入口）

**推荐**：`run_v1_pipeline_full.sh`（prepare → train → eval → visualize）。  
`run_v1_train.sh` / `run_v1_eval.sh` 仅作分步调试；二者与管线脚本一样通过 `CONFIG` 选择 n6 配置（默认 `n6_poscnn_size64.yaml`）。

```bash
cd CNN/spectral-history-cnn

CONFIG=configs/size_sweep/n6_poscnn_size64.yaml \
  bash scripts/run_v1_pipeline_full.sh

# 仅训练（需已有缓存）
CONFIG=configs/size_sweep/n6_poscnn_size64.yaml \
  bash scripts/run_v1_train.sh

# 已有缓存
SKIP_PREPARE=1 CONFIG=configs/size_sweep/n6_poscnn_size64.yaml \
  bash scripts/run_v1_pipeline_full.sh

# 覆盖 epoch 数
EPOCHS=50 CONFIG=configs/size_sweep/n6_poscnn_size64.yaml \
  bash scripts/run_v1_pipeline_full.sh
```

### 6.3 四尺寸顺序跑

```bash
cd CNN/spectral-history-cnn
bash scripts/run_size_sweep.sh

SIZES="64 128" bash scripts/run_size_sweep.sh
```

### 6.4 集群 NPU 提交

```bash
cd CNN/spectral-history-cnn

CNN_ROOT=/path/to/4IM06-G3-Project22/CNN/spectral-history-cnn \
REPO_ROOT=/path/to/4IM06-G3-Project22 \
CODES=/path/to/Codes \
  bash scripts/submit_size_sweep_npu.sh

SKIP_PREPARE=1 EPOCHS=50 bash scripts/submit_size_sweep_npu.sh
```

每个尺寸调用 `$CODES/vc_cnn_spectral_v1.sh`，作业名 `n6_cnn_size{N}`。

### 6.5 本地冒烟（CPU）

```bash
cd CNN/spectral-history-cnn

LIMIT_SAMPLES=50 EPOCHS=2 DEVICE=cpu \
  bash scripts/run_size_sweep.sh
```

---

## 7. 汇总分析（B/C 跑完后）

### 7.1 Mask vs CNN：6 类准确率 vs 输入尺寸

```bash
cd 4IM06-G3-Project22

python scripts/analysis/summarize_size_effect.py
# 可选：python scripts/analysis/summarize_size_effect.py --sizes 32,64,96,128
```

读取：

- `results/mask/n6_mask_size{N}/metrics.json`
- `results/cnn/n6_poscnn_size{N}/metrics.json`

输出：`results/comparison/size_effect/`（CSV + PNG）

### 7.2 三方法统一对比（需 A 的 DCT-FFT 扫描 + B/C 的 metrics）

```bash
cd 4IM06-G3-Project22

python scripts/analysis/unified_method_comparison.py \
  --sizes 32,64,96,128 \
  --classical-eval-dir results/classical/jpeg_detector_size_sweep \
  --outdir results/comparison/unified_comparison
```

输出：`results/comparison/unified_comparison/unified_comparison.{csv,png}`

---

## 8. 完整实验矩阵（需跑的全部作业）

n6 主实验共 **4 尺寸 × 2 可学习路线 + 1 经典路线 + 2 汇总**：

| 观测尺寸 | Mask 配置 | CNN 配置 | Mask 命令 | CNN 命令 |
|----------|-----------|----------|-----------|----------|
| 32 | `n6_mask_size32.yaml` | `n6_poscnn_size32.yaml` | `CONFIG=configs/size_sweep/n6_mask_size32.yaml bash scripts/run_pipeline_config.sh` | `CONFIG=configs/size_sweep/n6_poscnn_size32.yaml bash scripts/run_v1_pipeline_full.sh` |
| 64 | `n6_mask_size64.yaml` | `n6_poscnn_size64.yaml` | 同上，换 64 | 同上，换 64 |
| 96 | `n6_mask_size96.yaml` | `n6_poscnn_size96.yaml` | 同上，换 96 | 同上，换 96 |
| 128 | `n6_mask_size128.yaml` | `n6_poscnn_size128.yaml` | 同上，换 128 | 同上，换 128 |

经典 + 汇总（各跑 **1 次**）：

```bash
cd 4IM06-G3-Project22
bash scripts/analysis/run_classical_pipeline.sh          # 或 --detach
python scripts/analysis/summarize_size_effect.py         # B+C 完成后
python scripts/analysis/unified_method_comparison.py     # A+B+C 完成后
```

---

## 9. 历史基线（可选，结果已保留）

以下 **不是 n6 主实验**，结果仍在仓库中，配置/脚本在分支 `archive/legacy-u6-u7`：

| 实验 | 结果目录 | 说明 |
|------|----------|------|
| Mask v1（512×257 网格，4 类） | `spectral-mask-resampling/outputs/v1_fourier_ambiguity_mask_clean/` | acc 56.6%，见 `docs/02_spectral_mask.md` §6 |
| CNN v1（64×64 原生谱，6 类） | `CNN/spectral-history-cnn/outputs/v1_final64_poscnn/` | 见 `docs/03_spectral_cnn.md` |

重绘 v1 Mask 汇总图（无需重训）：

```bash
cd spectral-mask-resampling
python scripts/plot_mask_results.py \
  --output-dir outputs/v1_fourier_ambiguity_mask_clean
```

---

## 10. 常见问题

**Q：统一对比图里 classic 有曲线、mask/cnn 缺失？**  
A：先确认 `results/mask/n6_mask_size*/metrics.json` 与 `results/cnn/n6_poscnn_size*/metrics.json` 存在；`unified_method_comparison.py` 会跳过缺失尺寸并打印 `[skip]`。

**Q：Mask prepare 很慢？**  
A：设 `PREP_WORKERS=0`（默认）用满 CPU 核；四尺寸可并行提交 4 个 prepare 作业（不同 `CONFIG`）。

**Q：集群上 Mask 提交失败「vc wrapper not found」？**  
A：将 `spectral-mask-resampling/scripts/vc_mask.sh` 复制到 `$CODES/vc_mask.sh`，并按其中注释把 `vc submit` 行从 `vc_cnn_spectral_v1.sh` 拷过来。

**Q：经典 A-2 的 resample_x8 与 Mask/CNN 的 downsample_x8 一样吗？**  
A：不一样。经典是分块重采样；Mask/CNN 是全局 bicubic 缩放。三方法只在「是否重采样 vs 原图/JPEG」二分轴上严格可比。

**Q：`run_v1_train.sh` 和 `run_v1_pipeline_full.sh` 用哪个？**  
A：n6 主实验用 **`run_v1_pipeline_full.sh`**（或 `run_size_sweep.sh` 跑四尺寸）。`run_v1_train.sh` 只跑训练，默认 `CONFIG=configs/size_sweep/n6_poscnn_size64.yaml`；历史 v1 配置在 `archive/legacy-u6-u7` 分支。

**Q：`resample_sample_list.py` 还要跑吗？**  
A：**不用**。那是 v1 时代修补 `RAISE-1000-ms/sample_list.csv` 的遗留工具；n6 用 `split_raise.py` + `run_prepare_config.sh`，不在 runbook 主路径里。

**Q：误用了 `configs/legacy/` 里的 v1 配置怎么办？**  
A：`train.py` / `evaluate.py` / `preprocess_spectra.py` 会拒绝 legacy 路径。若确需复现旧实验：`ALLOW_LEGACY_CONFIG=1`（配置在 `CNN/.../configs/legacy/`）。

---

## 11. 相关文档

| 文档 | 内容 |
|------|------|
| [`00_project_overview.md`](00_project_overview.md) | 三路线总览 |
| [`01_classical_detection.md`](01_classical_detection.md) | 经典检测细节 |
| [`02_spectral_mask.md`](02_spectral_mask.md) | Mask 方法与 n6 协议 |
| [`03_spectral_cnn.md`](03_spectral_cnn.md) | CNN 方法与 n6 协议 |
| [`SUPPLEMENTARY_EXPERIMENTS.md`](SUPPLEMENTARY_EXPERIMENTS.md) | 补充实验 E2/E4 等 |

---

## 12. Mask / CNN 数据对齐（P0+P1）

**结论：训练用的 1 通道 log 谱缓存是同一份文件。**

| 项目 | 内容 |
|------|------|
| **共享 split** | `spectral-mask-resampling/data/splits/raise_split_seed123.json` |
| **对齐采样** | `experiments/unified_protocol.py` → `make_aligned_observed_patch` + `per_sample_seed` |
| **统一样本数** | train/val/test 各 **1000 / 类 / 尺寸** |
| **共享谱缓存** | `data/processed/n6_spectra_size{N}/`（Mask `data_dir` = CNN `processed_dir`） |
| **预处理入口** | `SIZE=N bash scripts/prepare_n6_spectra.sh`（**每个尺寸只跑一次**） |
| **训练差异** | 仅模型侧：CNN 在读取谱后加 44 通道 positional encoding；Mask 用可学习 mask |

CNN 完整管线里的 prepare 步骤若发现共享缓存已存在会自动跳过；也可先统一 preprocess 再 `SKIP_PREPARE=1` 分别训 Mask/CNN。

> 若仍使用旧的双份缓存（`n6_mask_size*` / `n6_tv_rfft_size*`），请删除后重跑 `prepare_n6_spectra.sh`。
