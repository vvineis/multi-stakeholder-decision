"""
Static paper-style radar + ranking plots from a decision-metrics CSV.

Works on any output of the framework -- including the merged CSV produced
by `fairness_baselines.py --merge_with=...`, which is the easiest way to
get a single figure that puts the compromise rules and the Fairlearn
baselines side by side.

Usage
-----
    # Lending, all rows in the CSV
    python plot_with_baselines.py \\
        results/lending/run_xxx/final_ranked_decision_metrics.csv \\
        --use-case lending --output figs/lending_radar.png

    # Lending merged with baselines, restricted to a few actors
    python plot_with_baselines.py \\
        results/lending/fairness_baselines_xxx/merged_ranked_decision_metrics.csv \\
        --use-case lending --output figs/lending_with_baselines.png \\
        --actors Oracle Bank Maximin "Nash Bargaining" FairLearn_EG_DP FairLearn_TO_EO
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ----------------------------------------------------------------------
def _load_use_case_yaml(use_case: str) -> dict:
    path = Path(__file__).parent / "conf" / "use_case" / f"{use_case}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _normalize_for_radar(df: pd.DataFrame, metrics: list[str], ranking_criteria: dict) -> pd.DataFrame:
    """Map each metric column into [0, 1] so the radar axes are visually comparable.
    For `zero`-direction metrics we plot 1 - |value| / max(|value|) (higher = closer to 0)."""
    out = df.copy()
    for m in metrics:
        col = out[m]
        if ranking_criteria.get(m) == "zero":
            denom = col.abs().max() or 1.0
            out[m] = 1.0 - col.abs() / denom
        else:
            lo, hi = col.min(), col.max()
            denom = (hi - lo) or 1.0
            out[m] = (col - lo) / denom
    return out


def radar_plot(df: pd.DataFrame, metrics: list[str], title: str) -> plt.Figure:
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(9, 8))
    cmap = plt.get_cmap("tab10")

    for idx, (_, row) in enumerate(df.iterrows()):
        values = [float(row[m]) for m in metrics] + [float(row[metrics[0]])]
        color = cmap(idx % 10)
        ax.plot(angles, values, label=str(row["Actor/Criterion"]), linewidth=1.8, color=color)
        ax.fill(angles, values, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.05), fontsize=8)
    return fig


def ranking_bar(df: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(df) + 2))
    if "Weighted Normalized-Sum" not in df.columns:
        ax.text(0.5, 0.5, "Weighted Normalized-Sum column not in CSV", ha="center")
        return fig
    sorted_df = df.sort_values("Weighted Normalized-Sum", ascending=True)
    ax.barh(sorted_df["Actor/Criterion"], sorted_df["Weighted Normalized-Sum"], color="steelblue")
    ax.set_xlabel("Weighted Normalized-Sum (higher = preferred under current weights)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="Path to decision-metrics CSV.")
    parser.add_argument("--use-case", required=True, choices=("lending", "health"))
    parser.add_argument("--output", type=Path, default=Path("figs/radar.png"))
    parser.add_argument("--actors", nargs="*",
                        help="Optional whitelist of Actor/Criterion names to plot.")
    parser.add_argument("--metrics", nargs="*",
                        help="Optional subset of evaluation metrics to plot "
                             "(default: all metrics_for_evaluation in the YAML).")
    parser.add_argument("--also-bar", action="store_true",
                        help="Also produce a ranking bar chart alongside the radar.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.actors:
        df = df[df["Actor/Criterion"].isin(args.actors)]
    if df.empty:
        raise SystemExit("No matching rows; check --actors against the CSV's Actor/Criterion column.")

    use_case_cfg = _load_use_case_yaml(args.use_case)
    metrics = args.metrics or list(use_case_cfg["criteria"]["metrics_for_evaluation"])
    ranking_criteria = dict(use_case_cfg["criteria"]["ranking_criteria"])

    plot_df = _normalize_for_radar(df, metrics, ranking_criteria)
    fig = radar_plot(plot_df, metrics, title=f"{args.use_case.title()} -- decision strategies")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved radar to {args.output}")

    if args.also_bar:
        bar_path = args.output.with_name(args.output.stem + "_ranking.png")
        bar_fig = ranking_bar(df, title=f"{args.use_case.title()} -- ranking")
        bar_fig.savefig(bar_path, dpi=200, bbox_inches="tight")
        print(f"Saved ranking bar to {bar_path}")


if __name__ == "__main__":
    main()
