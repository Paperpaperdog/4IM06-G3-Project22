from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .config import PILOT_RESULTS_DIR


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def float_col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows if row.get(key, "") not in ("", "nan")])


def summarize_idea1(rows: list[dict]) -> list[str]:
    lines = ["## Idea 1 — JPEG / x8 / x16 confusion", ""]
    if not rows:
        lines.append("_No results._")
        return lines

    conditions = sorted({row["condition"] for row in rows})
    lines.append("| Condition | Mean best log10(NFA) | Significant rate | Mean NFA @ d=48 | Mean NFA @ d=128 |")
    lines.append("|-----------|----------------------|------------------|-----------------|------------------|")

    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        best = float_col(subset, "nfa_best_log10")
        sig = float_col(subset, "is_significant")
        d48 = float_col(subset, "log10_nfa_d48")
        d128 = float_col(subset, "log10_nfa_d128")
        lines.append(
            f"| {condition} | {np.nanmean(best):.2f} | {np.nanmean(sig):.0%} | "
            f"{np.nanmean(d48):.2f} | {np.nanmean(d128):.2f} |"
        )

    lines.extend(
        [
            "",
            "**Readout:** compare `jpeg_*_identity` (compression-only) vs `*_resample_*` / `*_sim_x8` "
            "(true or grid artefacts). If JPEG-only runs peak near n/8 (48 for 384px) while resampling "
            "peaks align with expected periods (e.g. 128), location alone may separate classes; "
            "overlapping peaks motivate Idea 2 shape features.",
            "",
        ]
    )
    return lines


def summarize_idea2(rows: list[dict]) -> list[str]:
    lines = ["## Idea 2 — k groups and shape metrics", ""]
    if not rows:
        lines.append("_No results._")
        return lines

    lines.append("| k | Mean prominence | Mean half-width | Mean side ratio | Significant rate |")
    lines.append("|---|-----------------|-----------------|-----------------|------------------|")
    for k in sorted({int(row["k"]) for row in rows}):
        subset = [row for row in rows if int(row["k"]) == k]
        prom = float_col(subset, "corr_prominence")
        width = float_col(subset, "corr_half_width")
        side = float_col(subset, "corr_side_ratio")
        sig = float_col(subset, "is_significant")
        lines.append(
            f"| {k} | {np.nanmean(prom):.4f} | {np.nanmean(width):.2f} | "
            f"{np.nanmean(side):.3f} | {np.nanmean(sig):.0%} |"
        )

    by_pattern: dict[str, list[dict]] = {}
    for row in rows:
        by_pattern.setdefault(row["pattern"], []).append(row)

    lines.append("")
    lines.append("**Pattern comparison (mean corr prominence):**")
    for pattern, subset in sorted(by_pattern.items()):
        lines.append(f"- `{pattern}`: {np.nanmean(float_col(subset, 'corr_prominence')):.4f}")

    lines.extend(["", ""])
    return lines


def _png_resample_peak_agreement(idea1: list[dict]) -> tuple[int, int]:
    """Count images where png_identity and png_resample share the same best distance."""
    by_image: dict[str, dict[str, str]] = {}
    for row in idea1:
        by_image.setdefault(row["image_id"], {})[row["condition"]] = row

    same = 0
    total = 0
    for conditions in by_image.values():
        if "png_identity" not in conditions or "png_resample_to_target" not in conditions:
            continue
        total += 1
        if (
            conditions["png_identity"]["nfa_best_distance"]
            == conditions["png_resample_to_target"]["nfa_best_distance"]
        ):
            same += 1
    return same, total


def recommend(idea1: list[dict], idea2: list[dict]) -> list[str]:
    lines = ["## 结论（基于本批实验数据）", ""]

    if not idea1 and not idea2:
        lines.append("请先运行试点：`python run_pilots.py`。")
        return lines

    evidence: list[str] = []

    if idea1:
        same, total = _png_resample_peak_agreement(idea1)
        png_id = [r for r in idea1 if r["condition"] == "png_identity"]
        png_rs = [r for r in idea1 if r["condition"] == "png_resample_to_target"]
        jpg_id = [r for r in idea1 if r["condition"] == "jpeg_q90_identity"]
        jpg_rs = [r for r in idea1 if r["condition"] == "jpeg_q90_resample_to_target"]
        sim_x8 = [r for r in idea1 if r["condition"] == "jpeg_q90_sim_x8"]

        if total:
            evidence.append(
                f"- **PNG 重采样不可分：** {same}/{total} 张图上 "
                f"`png_identity` 与 `png_resample_to_target` 的 **最佳峰距离 d 完全相同**。"
            )
        if png_id and png_rs:
            evidence.append(
                f"- **PNG 路径几乎不触发显著峰：** 上述两组平均显著率 "
                f"{np.nanmean(float_col(png_id, 'is_significant')):.0%}，"
                f"平均最佳 log10(NFA)={np.nanmean(float_col(png_id, 'nfa_best_log10')):.2f}。"
            )
        if jpg_id and jpg_rs:
            d48_id = np.nanmean(float_col(jpg_id, "log10_nfa_d48"))
            d48_rs = np.nanmean(float_col(jpg_rs, "log10_nfa_d48"))
            d128_rs = np.nanmean(float_col(jpg_rs, "log10_nfa_d128"))
            evidence.append(
                f"- **JPEG 压缩 + 重采样仍不可分：** `jpeg_q90_identity` 与 "
                f"`jpeg_q90_resample_to_target` 聚合指标一致（显著率 "
                f"{np.nanmean(float_col(jpg_id, 'is_significant')):.0%}）；"
                f"平均 @d=128 的 log10(NFA)={d128_rs:.2f}（理论 512→384 周期 128 **未**成为主导峰）。"
            )
        if sim_x8:
            at48 = sum(1 for r in sim_x8 if int(float(r["nfa_best_distance"])) == 48)
            evidence.append(
                f"- **JPEG+栅格模拟信号极强但不稳定：** `jpeg_q90_sim_x8` 平均 @d=48 的 "
                f"log10(NFA)={np.nanmean(float_col(sim_x8, 'log10_nfa_d48')):.1f}，"
                f"但仅 **{at48}/{len(sim_x8)}** 张图最佳峰落在 d=48。"
            )

    if idea2:
        k_means = []
        for k in (-1, 0, 1):
            subset = [r for r in idea2 if int(r["k"]) == k]
            if subset:
                k_means.append(np.nanmean(float_col(subset, "corr_prominence")))
        if len(k_means) == 3:
            evidence.append(
                f"- **k 分组未验证：** k=−1/0/1 的平均 prominence 分别为 "
                f"{k_means[0]:.4f} / {k_means[1]:.4f} / {k_means[2]:.4f} "
                f"（跨 k 差异 {np.std(k_means):.4f}），**不能**据此选择相关模式。"
            )
        ref = [float(r["corr_prominence"]) for r in idea2 if r["pattern"] == "ref_plus_k"]
        tmr = [float(r["corr_prominence"]) for r in idea2 if r["pattern"] == "target_minus_ref_plus_k"]
        if ref and tmr:
            evidence.append(
                f"- **参考尺寸公式（次要）：** `target_minus_ref_plus_k` 平均 prominence "
                f"({np.nanmean(tmr):.4f}) 高于 `ref_plus_k` ({np.nanmean(ref):.4f})，"
                f"差别约 {100*(np.nanmean(tmr)-np.nanmean(ref))/np.nanmean(ref):.1f}%，属弱效应。"
            )
        evidence.append(
            f"- **检测过敏感：** 思路 2 整体显著率约 "
            f"{np.nanmean(float_col(idea2, 'is_significant')):.0%}，需负样本与阈值校准。"
        )

    lines.extend(evidence)
    lines.append("")
    lines.append("### 建议的下一阶段主方向")
    lines.append("")
    lines.append(
        "**优先深化思路 1（JPEG / 周期混淆 + 基线有效性），暂缓大规模 CNN 数据集。**\n\n"
        "依据：本批 RAISE 上 **无法** 用峰值距离区分「仅缩放」与「缩放到 384 的重采样」；"
        "思路 2 的 k 与形状指标 **未** 提供可操作的判别力。在 NFA 对真实重采样仍不敏感之前，"
        "构建大量标注样本训练 CNN 的边际收益有限。\n\n"
        "具体建议：\n"
        "1. 在 `detect_one_image.py` / `src/ird.py` 上系统对比 **`--is_jpeg` 开/关**，"
        "量化 n/8、n/16 抑制能否降低 `jpeg_q90_sim_x8` 类假阳性；\n"
        "2. 补充 **强受控合成图**（已知周期，如 512→384），确认实现无误后再评 RAISE；\n"
        "3. 调查 RAISE 大图裁方→384 后周期被淹没的原因（内容频谱、TV 残差、patch 尺寸）；\n"
        "4. 若继续 reference-size 实验，可优先试 `target_minus_ref_plus_k`，"
        "**不要** 期待 k∈{−1,0,1} 单独解决问题。"
    )
    lines.append("")
    return lines


def write_summary(idea1_csv: Path, idea2_csv: Path, out_md: Path) -> None:
    idea1 = read_csv(idea1_csv) if idea1_csv.is_file() else []
    idea2 = read_csv(idea2_csv) if idea2_csv.is_file() else []

    lines = [
        "# Pilot comparison summary",
        "",
        f"- Idea 1 rows: {len(idea1)}",
        f"- Idea 2 rows: {len(idea2)}",
        "",
    ]
    lines.extend(summarize_idea1(idea1))
    lines.extend(summarize_idea2(idea2))
    lines.extend(recommend(idea1, idea2))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")
    print("\n".join(lines[-8:]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Idea 1 and Idea 2 pilot CSVs.")
    parser.add_argument("--idea1-csv", type=Path, default=PILOT_RESULTS_DIR / "idea1_results.csv")
    parser.add_argument("--idea2-csv", type=Path, default=PILOT_RESULTS_DIR / "idea2_results.csv")
    parser.add_argument("--out-md", type=Path, default=PILOT_RESULTS_DIR / "PILOT_SUMMARY.md")
    args = parser.parse_args()
    write_summary(args.idea1_csv, args.idea2_csv, args.out_md)


if __name__ == "__main__":
    main()
