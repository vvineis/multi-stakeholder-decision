"""
Stacked Dirichlet-ablation figure: three per-bucket panels arranged
vertically (top -> bottom), each showing the winner-share bars + the
weighted-normalised-sum violin for one deployment configuration.

All rows share the SAME canonical actor -> colour map used by
`plot_pareto.py`, `plot_compare_ablations.py`, and
`plot_combined_pareto_ternary.py`, so the same decision function reads
as the same hue across every paper figure. Rows also share the y-axis
ordering (canonical actor slot), so the reader can compare a decision
function vertically across buckets at a glance.

Example
-------
    python plot_stacked_dirichlet.py \\
        --inputs ablations/rf_base_10000_dirichlet.csv \\
                 ablations/rf_stricter_10000_dirichlet.csv \\
                 ablations/knn_base_10000_dirichlet.csv \\
        --labels "RF / base" "RF / strictest" "KNN / base" \\
        --output figs/stacked_dirichlet.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_ablation import _render_population_row, _style
from plot_pareto import (
    ACTOR_PALETTE,
    CANONICAL_ACTOR_ORDER,
    make_actor_color_map,
)


def _shared_axis_order(dfs: list[pd.DataFrame]) -> list[str]:
    """Ordered list of actors present in ANY CSV, in canonical order.

    Canonical actors first (in CANONICAL_ACTOR_ORDER), then any unknown
    actors alphabetically after. Same rule as make_actor_color_map so
    colours and y-positions agree.
    """
    present = set()
    for df in dfs:
        present.update(df["Actor/Criterion"].unique())
    canonical = [a for a in CANONICAL_ACTOR_ORDER if a in present]
    canonical += sorted(a for a in present if a not in set(CANONICAL_ACTOR_ORDER))
    return canonical


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inputs", nargs="+", required=True, type=Path,
                        help="One long-format Dirichlet CSV per bucket, top to bottom.")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Row labels, same length and order as --inputs.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default=None,
                        help="Optional figure suptitle.")
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise SystemExit("--inputs and --labels must have the same length.")

    dfs = [pd.read_csv(p) for p in args.inputs]
    for df, p in zip(dfs, args.inputs):
        if "is_consensus_winner" not in df.columns:
            raise SystemExit(f"{p} does not look like a multi-seed ablate_weights.py output.")

    # Shared canonical colour map + shared y-axis ordering across rows.
    all_actors = set().union(*(set(df["Actor/Criterion"].unique()) for df in dfs))
    color_map = make_actor_color_map(all_actors, palette=ACTOR_PALETTE)
    # Reversed so barh (bottom-up) puts the canonical-first actor at the TOP
    # of the axis -- matches how the reader scans the ternary/Pareto legend.
    y_order = _shared_axis_order(dfs)[::-1]

    n_rows = len(dfs)
    with _style():
        fig, axes = plt.subplots(
            n_rows, 2,
            figsize=(16, max(4.5, 0.42 * len(y_order) + 1.5) * n_rows),
            gridspec_kw={"hspace": 0.65, "wspace": 0.55},
        )
        if n_rows == 1:
            axes = axes.reshape(1, 2)

        for i, (df, label) in enumerate(zip(dfs, args.labels)):
            winners_df = df[df["is_consensus_winner"]]
            counts = winners_df["Actor/Criterion"].value_counts()
            total = int(counts.sum())
            mean_stab = (
                winners_df.groupby("Actor/Criterion")["consensus_winner_stability"].mean()
            )
            n_seeds = int(df["n_seeds"].max()) if "n_seeds" in df.columns else 1

            _render_population_row(
                axes[i, 0], axes[i, 1],
                df, color_map, y_order,
                counts, total, mean_stab, n_seeds,
                show_titles=(i == 0),  # only top row keeps the two panel titles
                show_ylabels=True,
                row_label=label,
            )

        if args.title:
            fig.suptitle(args.title, fontsize=15, y=1.001, fontweight="bold")
        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
