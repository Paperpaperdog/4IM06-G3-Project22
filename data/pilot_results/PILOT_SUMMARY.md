# Pilot comparison summary

- Idea 1 rows: 70
- Idea 2 rows: 540

## Idea 1 — JPEG / x8 / x16 confusion

| Condition | Mean best log10(NFA) | Significant rate | Mean NFA @ d=48 | Mean NFA @ d=128 |
|-----------|----------------------|------------------|-----------------|------------------|
| jpeg_q75_identity | -36.03 | 100% | -23.91 | 2.44 |
| jpeg_q90_identity | -10.61 | 70% | -5.97 | 2.29 |
| jpeg_q90_resample_to_target | -10.61 | 70% | -5.97 | 2.29 |
| jpeg_q90_sim_x8 | -82.64 | 100% | -71.40 | 2.40 |
| png_identity | -1.19 | 0% | 2.44 | 2.36 |
| png_resample_to_target | -1.19 | 0% | 2.44 | 2.36 |
| png_sim_x8 | -3.55 | 30% | -1.15 | 2.04 |

**Readout:** compare `jpeg_*_identity` (compression-only) vs `*_resample_*` / `*_sim_x8` (true or grid artefacts). If JPEG-only runs peak near n/8 (48 for 384px) while resampling peaks align with expected periods (e.g. 128), location alone may separate classes; overlapping peaks motivate Idea 2 shape features.

## Idea 2 — k groups and shape metrics

| k | Mean prominence | Mean half-width | Mean side ratio | Significant rate |
|---|-----------------|-----------------|-----------------|------------------|
| -1 | 0.9199 | 1.10 | 6.419 | 67% |
| 0 | 0.9196 | 1.10 | 6.422 | 70% |
| 1 | 0.9197 | 1.10 | 6.422 | 72% |

**Pattern comparison (mean corr prominence):**
- `ref_plus_k`: 0.8985
- `target_minus_ref_plus_k`: 0.9411


## 结论（基于本批实验数据）

- **PNG 重采样不可分：** 10/10 张图上 `png_identity` 与 `png_resample_to_target` 的 **最佳峰距离 d 完全相同**。
- **PNG 路径几乎不触发显著峰：** 上述两组平均显著率 0%，平均最佳 log10(NFA)=-1.19。
- **JPEG 压缩 + 重采样仍不可分：** `jpeg_q90_identity` 与 `jpeg_q90_resample_to_target` 聚合指标一致（显著率 70%）；平均 @d=128 的 log10(NFA)=2.29（理论 512→384 周期 128 **未**成为主导峰）。
- **JPEG+栅格模拟信号极强但不稳定：** `jpeg_q90_sim_x8` 平均 @d=48 的 log10(NFA)=-71.4，但仅 **1/10** 张图最佳峰落在 d=48。
- **k 分组未验证：** k=−1/0/1 的平均 prominence 分别为 0.9199 / 0.9196 / 0.9197 （跨 k 差异 0.0001），**不能**据此选择相关模式。
- **参考尺寸公式（次要）：** `target_minus_ref_plus_k` 平均 prominence (0.9411) 高于 `ref_plus_k` (0.8985)，差别约 4.7%，属弱效应。
- **检测过敏感：** 思路 2 整体显著率约 70%，需负样本与阈值校准。

### 建议的下一阶段主方向

**优先深化思路 1（JPEG / 周期混淆 + 基线有效性），暂缓大规模 CNN 数据集。**

依据：本批 RAISE 上 **无法** 用峰值距离区分「仅缩放」与「缩放到 384 的重采样」；思路 2 的 k 与形状指标 **未** 提供可操作的判别力。在 NFA 对真实重采样仍不敏感之前，构建大量标注样本训练 CNN 的边际收益有限。

具体建议：
1. 在 `detect_one_image.py` / `src/ird.py` 上系统对比 **`--is_jpeg` 开/关**，量化 n/8、n/16 抑制能否降低 `jpeg_q90_sim_x8` 类假阳性；
2. 补充 **强受控合成图**（已知周期，如 512→384），确认实现无误后再评 RAISE；
3. 调查 RAISE 大图裁方→384 后周期被淹没的原因（内容频谱、TV 残差、patch 尺寸）；
4. 若继续 reference-size 实验，可优先试 `target_minus_ref_plus_k`，**不要** 期待 k∈{−1,0,1} 单独解决问题。
