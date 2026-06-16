# 最小补充实验清单

> 目标：用最少实验把 `PROJECT_REPORT.md` 里「未证实」项关掉，并形成 **Mask ↔ CNN ↔ 经典** 公平对照。  
> 预计总工时：**1–2 天**（E1 占大头）；E2 零训练，约 30 分钟。

---

## 总览

| ID | 实验 | 优先级 | 训练 | 预计耗时 | 产出文件 |
|----|------|--------|------|----------|----------|
| **E1** | 4 类 CNN + early stopping | **必做** | 是 | 4–8 h | `outputs/v1_final64_poscnn4/metrics_test.json` |
| **E2** | 6 类 CNN 消融（4 类子混淆） | **必做** | 否 | 30 min | `docs/tables/e2_cnn6_ablation.md` |
| **E3** | JPEG/×8 DCT 管线定量 | 建议 | 否 | 2–4 h | `test_results/dataset_x8_eval.txt` |
| **E4** | 受控 NFA 候选尺寸 top-k | 建议 | 否 | 1 h | `test_results/nfa_candidate_topk.csv` |

完成 E1+E2 后，即可在 `PROJECT_REPORT.md` 新增 **§9 补充实验** 并更新 §6.3「未证实」列表。

---

## E1：4 类 CNN（与 Mask 公平对比）

### 目的

- 与 Mask 相同：**4 类**（original / JPEG / downsample×8 / downsample×16）
- 相同 split：**700 / 150 / 150**（`seed=123`）
- 报告 **best checkpoint**，不报告 epoch 50

### 配置（已就绪）

| 项 | 值 |
|----|-----|
| GPU 配置 | `CNN/spectral-history-cnn/configs/v1_final64_poscnn.yaml` |
| 本地/NPU 配置 | `configs/v1_final64_poscnn_local.yaml` |
| 输出目录 | `outputs/v1_final64_poscnn4` |
| 谱缓存 | `data/processed/v1_final64_tv_rfft_4cls` |
| 测试规模 | 6000（4×1500），观测固定 **64×64** |

### 命令（GPU）

```bash
cd 4IM06-G3-Project22/CNN/spectral-history-cnn
export PYTHONPATH=.

# 1) 划分 + 下载/缓存谱（若已有 processed 可 SKIP_PREPARE=1）
python src/data/split_raise.py \
  --csv ../../data/raise_raw/RAISE_1k.csv \
  --id-column File --url-column TIFF \
  --output-json data/splits/raise_split_seed123.json \
  --train 700 --val 150 --test 150 --seed 123

python src/data/preprocess_spectra.py \
  --config configs/v1_final64_poscnn.yaml \
  --split-json data/splits/raise_split_seed123.json

# 2) 训练（save_best_by: val_loss，自动存 best.pt）
export CUDA_VISIBLE_DEVICES=0
python src/train.py --config configs/v1_final64_poscnn.yaml

# 3) 仅用 best checkpoint 评估 test
python src/evaluate.py \
  --config configs/v1_final64_poscnn.yaml \
  --checkpoint outputs/v1_final64_poscnn4/checkpoints/best.pt \
  --split test
```

### 命令（本地 NPU / 已有 TIFF 缓存）

```bash
cd 4IM06-G3-Project22/CNN/spectral-history-cnn

# 一键：prepare → train → eval（谱缓存与 Mask 共用 raise_tiff）
CONFIG=configs/v1_final64_poscnn_local.yaml \
RAISE_DIR=../../spectral-mask-resampling/data/raw/raise_tiff \
bash scripts/run_v1_pipeline_full.sh
```

> 注意：`run_v1_pipeline_full.sh` 里 eval 的 checkpoint 路径若仍写 `v1_final64_poscnn`，请改为 `outputs/v1_final64_poscnn4/checkpoints/best.pt`。

### 可选：限制 epoch（防过拟合叙事）

6 类实验显示 val 最佳在 **≈epoch 5**。可在训练时：

```bash
python src/train.py --config configs/v1_final64_poscnn.yaml --epochs 15
```

仍报告 `checkpoints/best.pt`（按 val_loss 选取），并在报告中注明 best epoch。

### 验收标准

- [ ] `outputs/v1_final64_poscnn4/metrics_test.json` 存在
- [ ] `outputs/v1_final64_poscnn4/figures/confusion_matrix_test.png` 存在
- [ ] 记录 best epoch（从 `training_log.csv` 或终端输出）

### 写入报告的表格模板（E1 完成后填空）

```markdown
### 表 E1-A：Mask vs CNN（4 类，同 split，test）

| 指标 | Mask（64 观测子集） | CNN 4 类（64×64） |
|------|---------------------|-------------------|
| 测试样本数 | 4000（4×1000，仅 o=64） | 6000（4×1500） |
| 准确率 | 56.9% | ___% |
| Macro F1 | 0.561 | ___ |
| original F1 | 0.59 | ___ |
| JPEG F1 | 0.69 | ___ |
| ×8 F1 | 0.45 | ___ |
| ×16 F1 | 0.51 | ___ |
| ×8→×16 混淆 | 1902 / 5000 | ___ / 1500 |
| ×16→×8 混淆 | 1762 / 5000 | ___ / 1500 |

> Mask 的 o=64 子集取自 `accuracy_by_observed_size`；全量多尺寸 Mask 为 56.6%。
```

```markdown
### 表 E1-B：CNN 6 类 vs 4 类（说明任务设定影响）

| 类 | 6 类 F1（已有） | 4 类 F1（E1） |
|----|----------------|---------------|
| original | 0.91 | ___ |
| JPEG | 0.91 | ___ |
| ×8 | 0.23 | ___ |
| ×16 | 0.48 | ___ |
```

---

## E2：6 类 CNN 消融（零训练）

### 目的

回答：6 类设定是否**人为增加**了 ×8/×16 混淆？（×2、×4 是否充当桥梁）

### 命令

```bash
cd 4IM06-G3-Project22
python scripts/analysis/e2_cnn6_ablation.py
```

脚本读取：`CNN/spectral-history-cnn/outputs/v1_final64_poscnn/metrics_test.json`  
输出：`docs/tables/e2_cnn6_ablation.md`

### 分析内容（脚本自动完成）

1. **4 类子矩阵**：只保留 original / JPEG / ×8 / ×16，重算 acc 与 F1
2. **×8/×16 二分类**：在 true∈{×8,×16} 子集上的准确率
3. **桥梁混淆**：×8 被误判为 ×4 的数量（6 类特有）

### 写入报告的表格模板（E2，数值来自现有 6 类结果）

```markdown
### 表 E2：6 类 CNN 消融（**已完成，见 `docs/tables/e2_cnn6_ablation.md`**）

| 分析 | 结果 |
|------|------|
| 6 类总体 acc | 62.5% |
| 4 类子集 acc（去掉 ×2/×4 样本） | **76.1%** |
| 4 类 macro F1 | **0.713** |
| ×8↔×16 互相误判（6 类） | 680 + 301 = 981 |
| ×8 被误判为 ×4（桥梁） | 351 |
| ×8/×16 二分类 acc（true 仅这两类） | **53.9%** |

> 说明：6 类设定会吸收部分 ×2/×4 样本，使 4 类子集 acc 虚高；E1 的 4 类重训才是公平对比。
```

---

## E3：JPEG/×8 DCT 管线定量

### 目的

补经典路线 A2 的数字，与 W3 NFA「RAISE 不敏感」对照：**受控合成**下 DCT/FFT 能否分开 JPEG 与 ×8。

### 数据准备

`evaluate_detector_on_dataset.py` 要求目录：

```text
dataset_x8/
  train/original/    # null 分布
  test/original/
  test/jpeg/
  test/resample_x8/
  test/mix/
```

**最小方案**（RAISE test 150 张）：

```bash
cd 4IM06-G3-Project22

# 1) 从 CNN/Mask 共用缓存取出 test 源图（或任意 50–150 张 TIFF/PNG）
TEST_SRC=data/x8_eval_sources   # 自行准备：150 张 test split 原图
mkdir -p "$TEST_SRC"

# 2) 生成四类后处理（flat 结构）
python create_forensic_postprocess_dataset.py \
  --input_dir "$TEST_SRC" \
  --output_dir data/dataset_x8_flat \
  --include_original \
  --mix_order both \
  --quality 85

# 3) 整理为 evaluate 脚本要求的 split 结构
mkdir -p data/dataset_x8/train/original data/dataset_x8/test
cp -r data/dataset_x8_flat/original data/dataset_x8/train/original
for c in original jpeg resample_x8 mix; do
  mkdir -p "data/dataset_x8/test/$c"
  cp data/dataset_x8_flat/$c/* "data/dataset_x8/test/$c/" 2>/dev/null || true
done
```

### 评估

```bash
python evaluate_detector_on_dataset.py \
  --detector jpeg_resample_detector.py \
  --dataset_root data/dataset_x8 \
  --split test \
  --null_dir data/dataset_x8/train/original \
  --max_per_class 150 \
  | tee test_results/dataset_x8_eval.txt
```

### 验收标准

- [ ] `test_results/dataset_x8_eval.txt` 含 Accuracy 与混淆矩阵
- [ ] 至少 **jpeg vs resample_x8** 二分类准确率可引用

### 写入报告的表格模板（E3 完成后填空）

```markdown
### 表 E3：经典 DCT/FFT 检测（受控合成，N=___）

| true \ pred | jpeg | 8×8_resampling | mixed | uncertain |
|-------------|------|----------------|-------|-----------|
| jpeg | ___ | ___ | ___ | ___ |
| resample_x8 | ___ | ___ | ___ | ___ |
| mix | ___ | ___ | ___ | ___ |
| original | ___ | ___ | ___ | ___ |

总体准确率：___%  
JPEG vs ×8 二分类准确率：___%
```

---

## E4：受控 NFA 候选尺寸 top-k

### 目的

量化：**有 ground truth 的合成集**上，NFA 峰距离能否找回 designed peak / 源尺寸。

### 数据（已有）

`test_results/controlled_resampling_dataset_bicubic_raise100/detection_summary.csv`  
约 9000 行（100 图 × 多 source/target × vertical/horizontal）

关键列：

| 列 | 含义 |
|----|------|
| `designed_peak` | 真值峰距离 |
| `best_distance` | NFA 最佳峰 |
| `true_rank` | 真值在候选排序中的名次（1=top-1） |
| `best_nfa` | 最佳峰 NFA 值 |
| `top_candidate` | 排名第一的候选距离 |

### 命令

```bash
cd 4IM06-G3-Project22
python scripts/analysis/e4_nfa_candidate_topk.py
```

输出：`test_results/nfa_candidate_topk_summary.csv` + 终端打印

### 写入报告的表格模板（E4）

```markdown
### 表 E4：受控 NFA 候选估计（**已完成，见 `test_results/nfa_candidate_topk_summary.csv`**）

| 指标 | vertical | horizontal | 合计 |
|------|----------|------------|------|
| 样本数 N | 4500 | 4500 | 9000 |
| top-1 命中率（true_rank=1） | 11.1% | 11.1% | 11.1% |
| top-3 命中率（true_rank≤3） | 37.2% | 37.8% | 37.5% |
| best_distance = designed_peak | 22.9% | 23.6% | 23.3% |
| 显著检测率（best_nfa < 1） | 89.3% | 92.7% | 91.0% |
| 无排名（true_rank 空）占比 | 59.0% | 58.0% | 58.5% |

> 解读：受控合成上 NFA **常能检出显著峰**，但 **top-1 候选尺寸命中率仅 ~11%**——与 W3 RAISE 真实数据结论互补。
```

---

## 报告更新 checklist

完成实验后，在 `docs/PROJECT_REPORT.md` 中：

1. **新增 §9 补充实验**，粘贴表 E1–E4（填实数值）
2. **更新 §6.3**，将已完成项从「未证实」移到「已证实」
3. **更新 §8.2**，划掉已完成下一步
4. **更新 `EXPERIMENT_SUMMARY.md`** 第四节「CNN 4 类」状态

### §9 章节骨架（复制到 PROJECT_REPORT.md）

```markdown
## 9. 补充实验（2026-06-__）

本节关闭 §6.3 中「公平对比 / DCT 定量 / NFA top-k」等待证项。

### 9.1 E1：4 类 CNN 与 Mask 对比
（表 E1-A、E1-B）

**结论**：___

### 9.2 E2：6 类设定消融
（表 E2）

**结论**：___

### 9.3 E3：经典 DCT/FFT 管线
（表 E3）

**结论**：___

### 9.4 E4：受控 NFA 候选尺寸
（表 E4）

**结论**：___
```

---

## 建议执行顺序

```text
Day 1 上午：启动 E1 训练（同时跑 E2 + E4，约 1h）
Day 1 下午：E1 训练中 → 准备 E3 数据目录
Day 1 晚上：E1 eval → 填表 E1
Day 2：E3 评估 → 更新 PROJECT_REPORT §9
```

若时间只够一件事：**只做 E1**。
