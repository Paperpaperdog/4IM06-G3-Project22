#!/usr/bin/env python3
"""E4: Summarize controlled NFA candidate ranking from detection_summary.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = (
    ROOT
    / "test_results/controlled_resampling_dataset_bicubic_raise100/detection_summary.csv"
)
OUT_PATH = ROOT / "test_results/nfa_candidate_topk_summary.csv"
EPS = 1.0  # NFA < 1 treated as significant (project convention)


def summarize_group(df: pd.DataFrame) -> dict[str, float | int]:
    n = len(df)
    if n == 0:
        return {"n": 0}

    rank = pd.to_numeric(df["true_rank"], errors="coerce")
    top1 = (rank == 1).sum()
    top3 = (rank <= 3).sum()
    no_rank = rank.isna().sum()
    peak_match = (df["best_distance"] == df["designed_peak"]).sum()
    nfa_sig = (pd.to_numeric(df["best_nfa"], errors="coerce") < EPS).sum()

    return {
        "n": n,
        "top1_count": int(top1),
        "top1_rate": top1 / n,
        "top3_count": int(top3),
        "top3_rate": top3 / n,
        "peak_match_count": int(peak_match),
        "peak_match_rate": peak_match / n,
        "nfa_sig_count": int(nfa_sig),
        "nfa_sig_rate": nfa_sig / n,
        "no_rank_count": int(no_rank),
        "no_rank_rate": no_rank / n,
    }


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    rows = []
    for axis in ["vertical", "horizontal", "all"]:
        sub = df if axis == "all" else df[df["axis"] == axis]
        stats = summarize_group(sub)
        stats["axis"] = axis
        rows.append(stats)

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {OUT_PATH}\n")
    for _, r in out.iterrows():
        print(
            f"[{r['axis']}] n={int(r['n'])} | "
            f"top1={r['top1_rate']:.1%} | top3={r['top3_rate']:.1%} | "
            f"peak_match={r['peak_match_rate']:.1%} | "
            f"nfa<{EPS}={r['nfa_sig_rate']:.1%} | no_rank={r['no_rank_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
