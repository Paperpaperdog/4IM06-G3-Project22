# 重采样检测试点实验报告

**项目：** 4IM06-G3-Project22（图像重采样检测）  
**指导教师：** Quentin Bammey  
**报告类型：** 第 3 周（W3）试点实验说明与结果分析  
**实验日期：** 2026 年 6 月 2 日（RAISE 续跑完成：同日）  
**代码位置：** `4IM06-G3-Project22/pilots/`、`run_pilots.py`  
**数据：** [RAISE-1k](https://loki.disi.unitn.it/RAISE/download.html) 索引 `RAISE_1k.csv`，本批 **10** 张 TIFF

---

## 摘要

本项目基于 Bammey 等人的频谱相关 + a contrario（NFA）方法，目标是在真实场景中区分 **JPEG 压缩伪影** 与 **AI 生成 / 重采样** 留下的周期痕迹。第 3 周在 **RAISE-1k 前 10 张自然 RAW（TIFF→PNG）** 上完成思路 1（70 条）与思路 2（540 条）试点。

主要发现：（1）**思路 1 未达预期：** 10/10 张 RAISE 图上 `png_identity` 与 `png_resample_to_target` 最佳峰 **d 完全相同**；PNG 路径显著率 **0%**；JPEG Q90 下「仅压缩」与「压缩+重采样」聚合曲线 **一致**，d=128 处 **不显著**；（2）**JPEG/栅格假阳性成立：** `jpeg_q90_sim_x8` 在 d=48 平均 log10(NFA)≈**−71**，但仅 1/10 张最佳峰在 d=48；（3）**思路 2 未验证 k：** k=−1/0/1 的 prominence 分别为 0.920 / 0.920 / 0.920（跨 k 标准差 0.0003），**不能** 选择 k；`target_minus_ref_plus_k` 仅略优于 `ref_plus_k`（+4.7%）。

**结论（仅据本批数据）：** 下一阶段应 **优先思路 1**——在完整 `ird` 上测试 `--is_jpeg`、补受控合成对照、弄清 RAISE 上重采样峰不出现的原因；**不宜** 在当前 NFA 基线未区分重采样时投入大规模 CNN 数据集。

---

## 1. 项目背景

### 1.1 科学问题

数字图像在采集、编辑、生成、压缩、上传过程中，常会经历 **几何重采样**（缩放、插值）和 **有损压缩**（尤其是 JPEG）。重采样会在傅里叶频谱中引入 **周期性复制结构**；JPEG 则在频谱中引入与 **8×8 块网格** 相关的周期（宽度为 n 时，常关注 n/8、n/16 等距离）。

项目核心问题：

> 能否可靠地区分「**JPEG 压缩带来的周期峰**」与「**真实重采样带来的周期峰**」？进而用于识别 AI 生成链路中的上采样 / 重采样痕迹？

这在法医式图像分析、生成图像检测等场景中有直接应用价值。

### 1.2 理论基础（论文方法概要）

参考论文：[Resampling Detection](https://bammey.com/resampling_detection.pdf)（项目组已复现精简版 `resampling_core.py`）。

方法 pipeline 简述：

1. **预处理**：对灰度图做 TV 去噪，取残差 `image - TV(image)`，保留噪声级振荡与重采样痕迹。  
2. **频谱**：对残差做 2D FFT，得到复数频谱。  
3. **patch 相关**：将频谱切成不重叠 patch；对每个位移距离 **d**，计算 patch 与平移 d 后 patch 的 **复 Pearson 相关**（取模作为强度）。  
4. **a contrario NFA**：在每个 d 上统计「有多少 patch 在 [d−r, d+r] 窗口内取局部最大」；在零假设（无重采样）下该计数服从二项分布，得到 **NFA(d)**。**NFA 越小**，该距离越「异常」，越像存在重采样周期。  
5. **JPEG 处理**（完整 `ird` 实现）：若标记 `is_jpeg`，会排除与 n/16 相关的距离，减轻 JPEG 假阳性。

### 1.3 项目组进展时间线（摘自 SUIVI）

| 阶段 | 内容 |
|------|------|
| W1 | 确定目标；阅读论文；搭建仓库与沟通渠道 |
| W2 | 复现基线；对照实验；记录假设与失败案例 |
| W3 | 范围控制：暂不实现附录 E 非对齐；RAISE 先 TIFF→PNG；开展思路 1 / 2 试点 |
| 5/30 讨论 | 若试点支持，优先 **受控数据集 → 轻量 CNN → 特征解释 → 数学回溯** |

### 1.4 第 3 周待办（本次报告对应任务）

1. 将选定 RAISE TIFF 转为 PNG，建立干净实验子集。  
2. **思路 1**：JPEG / x8 / x16 混淆检查。  
3. **思路 2**：k = −1, 0, 1 对比，并考察峰值高度、宽度、旁瓣等形状指标。  
4. 比较两项试点，选定下一阶段主方向。

---

## 2. 本次工作内容

### 2.1 实现的软件流水线

在 `4IM06-G3-Project22/` 下新增可复现试点框架：

| 模块 | 文件 | 功能 |
|------|------|------|
| 配置 | `pilots/config.py` | 目标尺寸 384、参考峰 64/85/96、k 取值、检测半径等 |
| 数据准备 | `pilots/prepare_subset.py` | TIFF→PNG、中心裁剪正方形、写 `manifest.csv`；无 RAISE 时用 fallback |
| 图像变换 | `pilots/transforms.py` | 双三次缩放、JPEG 压缩、reference 尺寸公式、x8 网格模拟 |
| 指标 | `pilots/metrics.py` | 调用 `resampling_core.detect_axis`；NFA 探针、相关峰形状特征 |
| 思路 1 | `pilots/idea1_jpeg.py` | 7 种条件 × 每图，输出 `idea1_results.csv` |
| 思路 2 | `pilots/idea2_k_groups.py` | 参考尺寸组合 × 每图，输出 `idea2_results.csv` |
| 比较 | `pilots/compare.py` | 聚合统计，写 `PILOT_SUMMARY.md` 与方向建议 |
| 入口 | `run_pilots.py` | 一键顺序执行上述四步 |

**检测实现说明：** 试点使用项目组 **对齐版** `resampling_core.py`（TV + 频谱相关 + NFA），**未**接入上游 `detect_one_image.py` 的 `is_jpeg` 距离剔除，以便观察「原始 NFA 曲线」上 JPEG 与重采样峰的共存与混淆。

### 2.2 实际运行的命令与环境

```bash
cd 4IM06-G3-Project22
python run_pilots.py --max-images 3
```

- **耗时：** 约 7 分钟（思路 2 每张图 54 次检测，共 162 次，计算量大）。  
- **数据：** `data/raise_raw/` 中无 TIFF，自动 fallback 至 `../img/` 与 skimage 内置图。

### 2.3 实验子集（manifest）

通过 `RAISE_1k.csv` 下载前 **10** 张 TIFF（`data/raise_raw/tiff/`），中心裁正方形后写入 `data/raise_png/`。示例：

| image_id | 机身 | Keywords | 裁切边长 |
|----------|------|----------|----------|
| r000da54ft | Nikon D90 | nature; outdoor | 2848 |
| r001d260dt | D7000 | buildings; outdoor | 3264 |
| r006b0e4bt | D90 | Indoor | 2848 |
| r006fcc20t | D7000 | landscape; outdoor | 3264 |

完整列表见 `data/manifest.csv`。另：早期曾用 3 张 demo 图（baboon/camera）做首次试点，结论见附录 A。

### 2.4 思路 1：七种实验条件

所有结果图统一为 **384×384** 灰度。

| 条件 ID | 变换 | JPEG | 设计意图 |
|---------|------|------|----------|
| `png_identity` | 仅缩放到 384 | 无 | 基线（无重采样链） |
| `jpeg_q90_identity` | 同上 | Q=90 | 仅压缩伪影 |
| `jpeg_q75_identity` | 同上 | Q=75 | 更强压缩 |
| `png_resample_to_target` | 512/666→384 双三次 | 无 | **真实重采样** |
| `jpeg_q90_resample_to_target` | 重采样 + JPEG | Q=90 | 压缩 + 重采样叠加 |
| `png_sim_x8` | 384→48→384 | 无 | 模拟 **8 像素网格** |
| `jpeg_q90_sim_x8` | 网格模拟 + JPEG | Q=90 | 最接近「JPEG 块 + 栅格」混淆 |

**关注距离（宽 384）：**

- **d = 48**：n/8，JPEG 块相关  
- **d = 24**：n/16  
- **d = 128**：512 mod 384，真实 512→384 重采样理论周期  

**显著性判据：** `log10(NFA) < -5`（与 demo 阈值一致）。

### 2.5 思路 2：参考尺寸与 k 分组

对每个 `reference_peak ∈ {64, 85, 96}`、`k ∈ {-1, 0, 1}`、两种 pattern：

- **`ref_plus_k`：** 参考边长 = ref + k·{0,1,2}  
- **`target_minus_ref_plus_k`：** 参考边长 = (384 − ref) + k·{0,1,2}  

流程：正方形原图 → 缩放到 reference_size → 再缩放到 384×384 → 检测。

每张图 **3 × 2 × 3 × 3 = 54** 条记录，10 张图共 **540** 条。

除 NFA 外记录形状指标：

- `corr_prominence`：相关峰 prominence  
- `corr_half_width`：半高宽  
- `corr_side_ratio`：峰左右能量对数比（后文说明早期版本存在数值异常）

---

## 3. 实验结果

### 3.1 产出文件一览

```
data/
  raise_raw/RAISE_1k.csv, tiff/   # 10 张 RAISE TIFF
  raise_png/                      # 10 张正方形 PNG
  manifest.csv
  generated/idea1/, idea2/
  pilot_results/
    idea1_results.csv   # 70 行
    idea2_results.csv   # 540 行
    PILOT_SUMMARY.md
```

### 3.2 思路 1（RAISE 10 张，聚合）

| 条件 | 平均最佳 log10(NFA) | 显著率 | 平均 @d=48 | 平均 @d=128 |
|------|---------------------|--------|------------|-------------|
| jpeg_q90_sim_x8 | **−82.64** | **100%** | **−71.40** | +2.40 |
| jpeg_q75_identity | −36.03 | 100% | −23.91 | +2.44 |
| jpeg_q90_identity | −10.61 | 70% | −5.97 | +2.29 |
| jpeg_q90_resample_to_target | −10.61 | 70% | −5.97 | +2.29 |
| png_identity | −1.19 | 0% | +2.44 | +2.36 |
| png_resample_to_target | −1.19 | 0% | +2.44 | +2.36 |
| png_sim_x8 | −3.55 | 30% | −1.15 | +2.04 |

**典型单图（`idea1_results.csv`）：**

| image_id | png 最佳 d | png_resample 最佳 d | jpeg_q90_sim_x8 最佳 d |
|----------|------------|---------------------|-------------------------|
| r000da54ft | 35 | 35（相同） | 240 |
| r001d260dt | 13 | 13（相同） | **48** |
| r006b0e4bt | 158 | 158（相同） | 192 |
| r007f5116t | 312 | 312（相同） | 240 |

**要点：**

- **10/10 图** 上 `png_identity` 与 `png_resample_to_target` 的 **最佳 d 完全一致** → 2848/3264→384 在自然 RAW 上**几乎未改变 NFA 峰位置**。  
- **JPEG + x8 模拟** 仍极强（平均 d=48 处 log10(NFA)≈−71），但单图最佳 d 不总在 48（如 r000da54ft→240）。  
- **d=128** 在 RAISE 批次平均上 **不显著**（@d=128 的 log10(NFA)≈+2.3），与 demo `camera` 实验不同。

### 3.3 思路 2（RAISE 10 张，540 条）

| k | 平均 prominence | 显著率 | 平均 side_ratio（log） |
|---|-----------------|--------|-------------------------|
| −1 | 0.9199 | 67% | 6.42 |
| 0 | 0.9196 | 70% | 6.42 |
| 1 | 0.9197 | 72% | 6.42 |

| pattern | 平均 prominence |
|---------|-----------------|
| ref_plus_k | 0.8985 |
| target_minus_ref_plus_k | **0.9411** |

**解读：** k 之间 **无判别差异**（跨 k 均值标准差仅 0.0003）；`target_minus_ref_plus_k` 略优（+4.7%）仅可作为后续 reference 公式的弱先验，**不足以** 支撑「按 k 选模式」或立刻上 CNN。思路 2 显著率 67–72%，说明当前阈值下检测 **偏敏感**。

---

## 4. 结论（仅依据 RAISE-10 实验数据）

### 4.1 第 3 周待办完成情况

| 待办 | 状态 | 说明 |
|------|------|------|
| RAISE TIFF→PNG 子集 | ✅ | 10 张 |
| 思路 1 试点 | ✅ | 70 条 |
| 思路 2 试点 | ✅ | 540 条 |
| 比较并选方向 | ✅ | 见下节及 `PILOT_SUMMARY.md` |

### 4.2 思路 1：JPEG / 重采样能否靠峰值距离 d 区分？

| 问题 | 本批答案 | 依据 |
|------|----------|------|
| 无 JPEG 时，缩放到 384 与「先大方图再缩到 384」是否不同？ | **否** | **10/10** 图 `png_identity` 与 `png_resample_to_target` 最佳 **d 相同**；两组显著率 **0%** |
| 加 JPEG Q90 后，仅压缩 vs 压缩+重采样是否不同？ | **否（聚合）** | 两组平均 log10(NFA)、显著率（70%）、@d=48/@d=128 **完全一致** |
| 理论周期 128（512→384）是否出现？ | **否** | 平均 @d=128 的 log10(NFA)≈**+2.36**（不显著） |
| JPEG+8 栅格是否产生强假信号？ | **是，但不稳** | `jpeg_q90_sim_x8`：@d=48 平均 ≈**−71**；仅 **1/10** 张最佳峰在 d=48 |

**小结：** 在 RAISE 自然 RAW 上，**当前 `resampling_core` 流水线不能** 用 NFA 峰值位置区分「有无本次定义的重采样」。JPEG/栅格类干扰 **确实存在且很强**。这与 demo 图（如 `camera`）上可见 d=128 的现象 **不一致**，说明问题可能在自然图尺度、裁切或实现参数，而非「理论上一定可检」。

### 4.3 思路 2：k 与形状指标是否值得继续？

| 问题 | 本批答案 | 依据 |
|------|----------|------|
| k=−1 / 0 / 1 哪种相关模式更好？ | **无法判断** | 平均 prominence：0.9199 / 0.9196 / 0.9197 |
| 除峰位置外，形状是否有用？ | **未证明** | half-width 均在 ~1.1；k 间无分离 |
| 哪种 reference 公式略好？ | `target_minus_ref_plus_k` | 0.941 vs 0.899（+4.7%），弱效应 |

**小结：** 思路 2 **没有** 为「上 CNN / 建大数据集」提供实证支持；在 NFA 本身对重采样不敏感的前提下，标再多 reference 组合也难以产生有效标签。

### 4.4 下一阶段主方向（数据驱动，非预设）

**建议主方向：深化思路 1（基线 + JPEG 抑制 + 受控对照），暂缓大规模 CNN 数据集。**

理由概括：

1. 本批最核心的负结果是 **重采样未改变峰**——不先解决这一点，CNN 输入缺乏可靠监督信号。  
2. JPEG/栅格假阳性已证实，应优先对接论文 **`is_jpeg`** 与完整 `ird` 流程，看抑制后曲线是否可解释。  
3. 思路 2 的 k 与形状指标 **未通过** 试点筛选，不宜作为下一阶段主轴。

**建议工作项（按优先级）：**

1. 同一批 RAISE + 合成图（512→384 等已知周期）对比 `resampling_core` vs `detect_one_image.py --is_jpeg`。  
2. 画每张图的 d–log10(NFA) 曲线，归档「不可分」案例（已从 CSV 自动生成路径）。  
3. 调参：patch 比例、TV weight、半径 r，看能否在 RAISE 上唤起理论周期。  
4. 扩大 RAISE 样本（50–100 张）验证上述结论是否稳定。  
5. **暂缓：** 批量 CNN 数据集，直至「重采样 vs 非重采样」在 NFA 或替代特征上 **可重复分离**。

### 4.5 局限

| 局限 | 影响 |
|------|------|
| 仅 10 / 1000 张 RAISE | 结论为 pilot，但负结果（10/10 峰相同）已很强 |
| 未启用 `is_jpeg` | 故意观察裸 NFA；下一步必须对比抑制后行为 |
| TV 较慢 | 扩样需批处理 |
| demo 图与 RAISE 结论矛盾 | 需用合成对照查明是数据还是流程问题 |

---

## 5. 附录

### A. 如何复现实验

```bash
cd 4IM06-G3-Project22

# 全流程（可指定 RAISE 目录与图片数量）
python run_pilots.py --raise-dir data/raise_raw --max-images 10

# 分步
python -m pilots.prepare_subset --max-images 10
python -m pilots.idea1_jpeg
python -m pilots.idea2_k_groups
python -m pilots.compare
```

### B. 关键路径

- 中文进度笔记：`SUIVI.zh.md`  
- 本报告：`REPORT.zh.md`  
- 自动摘要：`data/pilot_results/PILOT_SUMMARY.md`  
- 核心检测：`resampling_core.py`  

### C. 参考文献

- Bammey et al., *Resampling Detection* — https://bammey.com/resampling_detection.pdf  
- 上游复现仓库：`ird-main`（`detect_one_image.py`, `src/ird.py`）

---

*报告由 W3 试点流水线自动生成结果后整理；若重跑实验，请同步更新第三节数值与 `PILOT_SUMMARY.md`。*
