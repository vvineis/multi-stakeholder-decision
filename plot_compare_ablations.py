"""
Compare multiple weight ablations side-by-side.

Use this when you have run `ablate_weights.py` separately per *bucket*
(e.g. RF / base, RF / strictest, KNN / base) and want to show how the
trade-off shifts across those conditions.

Three layouts, chosen by the sweep type in the first CSV:

* **pairwise**: stacked winner-region strips, one per CSV, plus a small
  legend. Direct visual answer to "does the boundary between winner
  regions shift with model / reward variant?"

* **dirichlet**: grouped horizontal bars showing winner-share per actor,
  one cluster per CSV. Direct answer to "which compromise rule is the
  most-robust winner under each condition?"

* **ternary**: 1 x N panels of triangles, one per CSV.

Usage
-----
    python plot_compare_ablations.py \\
        --inputs ablations/rf_base_acc_vs_dp.csv \\
                 ablations/rf_strictest_acc_vs_dp.csv \\
                 ablations/knn_base_acc_vs_dp.csv \\
        --labels "RF / base" "RF / strictest" "KNN / base" \\
        --output figs/compare_acc_vs_dp.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_pareto import CANONICAL_ACTOR_ORDER, make_actor_color_map


PALETTE = plt.get_cmap("tab10").colors + plt.get_cmap("Set2").colors


def _style():
    return plt.rc_context({
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.edgecolor": "#333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _weight_columns(df):
    return [c for c in df.columns if c.startswith("weight_")]


def _detect_sweep_kind(df):
    weight_cols = _weight_columns(df)
    varied = [c for c in weight_cols if df[c].nunique() > 1]
    if len(varied) == 2 and np.allclose(df[varied].sum(axis=1).unique(), 1.0):
        return "pairwise"
    if len(varied) == 3 and np.allclose(df[varied].sum(axis=1).unique(), 1.0):
        return "ternary"
    return "dirichlet"


def _global_color_map(dfs: list[pd.DataFrame]) -> dict:
    """One consistent colour per actor across every CSV -- canonical slots."""
    actors = {a for df in dfs for a in df["Actor/Criterion"].unique()}
    return make_actor_color_map(actors, palette=PALETTE)


# ======================================================================
# Pairwise: stacked winner-region strips
# ======================================================================
def compare_pairwise(dfs: list[pd.DataFrame], labels: list[str], output: Path, title: str):
    color_map = _global_color_map(dfs)

    # Discover the varying weight column (must be the same across all CSVs).
    weight_cols = _weight_columns(dfs[0])
    varied = [c for c in weight_cols if dfs[0][c].nunique() > 1]
    if len(varied) != 2:
        raise SystemExit("Pairwise comparison requires exactly two varying weights per CSV.")
    x_col = varied[0]

    with _style():
        n = len(dfs)
        fig, axes = plt.subplots(
            n, 1, figsize=(14, max(2.8, 0.9 * n + 1.8)),
            gridspec_kw={"hspace": 0.5}, sharex=True,
        )
        if n == 1:
            axes = [axes]

        for ax, df, label in zip(axes, dfs, labels):
            ax.grid(False)
            x_vals = sorted(df[x_col].unique())
            winners_by_x = {}
            stability_by_x = {}
            for x in x_vals:
                sub = df[(df[x_col] == x) & df["is_consensus_winner"]]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                winners_by_x[x] = row["Actor/Criterion"]
                stab = row["consensus_winner_stability"]
                stability_by_x[x] = float(stab) if pd.notna(stab) else 1.0

            for i, x in enumerate(x_vals):
                left = (x_vals[i - 1] + x) / 2 if i > 0 else (
                    x - (x_vals[1] - x_vals[0]) / 2 if len(x_vals) > 1 else x - 0.5
                )
                right = (x + x_vals[i + 1]) / 2 if i < len(x_vals) - 1 else (
                    x + (x_vals[-1] - x_vals[-2]) / 2 if len(x_vals) > 1 else x + 0.5
                )
                winner = winners_by_x.get(x)
                if winner is None:
                    continue
                alpha = 0.40 + 0.55 * stability_by_x.get(x, 1.0)
                ax.axvspan(left, right, color=color_map[winner], alpha=alpha)

            # Label each contiguous region
            prev_winner = None
            region_start_x = None
            for i, x in enumerate(x_vals):
                w = winners_by_x.get(x)
                if w != prev_winner:
                    if prev_winner is not None and region_start_x is not None:
                        midx = (region_start_x + x_vals[i - 1]) / 2
                        ax.text(midx, 0.5, prev_winner, ha="center", va="center",
                                fontsize=11, fontweight="bold", color="white")
                    prev_winner = w
                    region_start_x = x
            if prev_winner is not None and region_start_x is not None:
                midx = (region_start_x + x_vals[-1]) / 2
                ax.text(midx, 0.5, prev_winner, ha="center", va="center",
                        fontsize=11, fontweight="bold", color="white")

            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_title(label, fontsize=11, loc="left", pad=4)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)

        axes[-1].set_xlabel(
            x_col.replace("weight_", "Weight on ").replace("_", " "), fontsize=11,
        )

        # Shared legend below all strips
        actors_present = sorted({a for df in dfs for a in df[df["is_consensus_winner"]]["Actor/Criterion"].unique()})
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[a]) for a in actors_present]
        fig.legend(handles, actors_present, loc="lower center",
                   bbox_to_anchor=(0.5, -0.03), ncol=min(len(actors_present), 6),
                   frameon=False, title="Consensus winner")

        fig.suptitle(title, fontsize=11, y=1.02)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved {output}")


# ======================================================================
# Dirichlet: grouped winner-share bars (color = bucket, group = actor)
# ======================================================================
def compare_dirichlet(dfs: list[pd.DataFrame], labels: list[str], output: Path, title: str):
    # Compute winner shares AND mean inter-seed stability per (bucket, actor).
    per_csv_counts = []
    per_csv_stability = []          # mean consensus_winner_stability per (bucket, actor)
    totals = []
    for df in dfs:
        winners = df[df["is_consensus_winner"]]
        counts = winners["Actor/Criterion"].value_counts()
        stab = winners.groupby("Actor/Criterion")["consensus_winner_stability"].mean()
        per_csv_counts.append(counts)
        per_csv_stability.append(stab)
        totals.append(int(counts.sum()))

    # Union of actors *evaluated* in any bucket (not just winners) so actors
    # that competed but never won -- e.g. Outcome_Pred_Model when it is
    # dominated by the compromise rules at every sample -- still get a bar
    # (at 0%) rather than being silently dropped from the y-axis.
    union_counts: dict[str, int] = {}
    for df in dfs:
        for a in df["Actor/Criterion"].unique():
            union_counts.setdefault(a, 0)
    for c in per_csv_counts:
        for a, v in c.items():
            union_counts[a] = union_counts.get(a, 0) + int(v)
    # Sort by total wins (descending); zero-win actors sort to the bottom by
    # count and then alphabetically for stability.
    actors = sorted(union_counts, key=lambda a: (-union_counts[a], a))

    n = len(dfs)
    bucket_colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    with _style():
        # 1. INCREASE figure height factor from 0.45 to 0.85 (or higher) to stretch the plot vertically
        fig, ax = plt.subplots(figsize=(8, max(5, 0.85 * len(actors) + 2)))
        ax.grid(False)
        y = np.arange(len(actors))
        
        # 2. To put more blank space between individual bars inside a block, 
        # keep bar_h the same but set height to a fraction of it (e.g., bar_h * 0.85)
        bar_h = 0.8 / n

        for i, (label, counts, total) in enumerate(zip(labels, per_csv_counts, totals)):
            shares = np.array([counts.get(a, 0) / max(total, 1) * 100 for a in actors])
            offset = (i - (n - 1) / 2) * bar_h
            
            ax.barh(
                y + offset, shares, 
                height=bar_h * 0.9,     # Leaves a 15% clear gap between single bars
                color=bucket_colors[i],
                edgecolor="white", linewidth=0.6, alpha=0.92,
                label=f"{label}  (n={total})",
            )
            # Inline labels: percentage + mean inter-seed stability of that rule
            # as consensus winner in this bucket (e.g.  "57%  s=0.80"). Label
            # every actor that won at least one config; actors with a zero
            # share (competed but never won) get no label -- an empty bar is a
            # legible zero.
            max_share = max(shares) if len(shares) else 0
            stab_series = per_csv_stability[i]
            for j, share in enumerate(shares):
                if share > 0:
                    actor = actors[j]
                    stab_val = stab_series.get(actor, float("nan"))
                    stab_str = f", stab={stab_val:.2f}" if not np.isnan(stab_val) else ""
                    ax.text(
                        share + max_share * 0.008,
                        y[j] + offset,
                        f"{share:.0f}%{stab_str}",
                        va="center", ha="left", fontsize=11, color="#222",
                    )

        ax.legend(loc="lower right", frameon=True, edgecolor="lightgray", title="Bucket")
        ax.set_yticks(y)
        ax.set_yticklabels(actors)
        ax.invert_yaxis()  # top-of-axis = most-frequent winner
        ax.set_xlabel("Share of weight configurations won (%)", fontsize=11)
        ax.set_title(title, fontsize=11, pad=15)
        ax.grid(alpha=0.25, axis="x", linestyle="--")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

     
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved {output}")


# ======================================================================
# Ternary: single-panel overlay (color = winner, marker shape = bucket)
# ======================================================================
def _barycentric_to_cartesian(wa, wb, wc):
    return 0.5 * wa + 1.0 * wc, (np.sqrt(3) / 2) * wa


BUCKET_MARKERS = ("o", "s", "^", "D", "P", "X")


def compare_ternary(dfs: list[pd.DataFrame], labels: list[str], output: Path, title: str):
    # All CSVs must share the same three varying weight columns.
    weight_cols = _weight_columns(dfs[0])
    varied = [c for c in weight_cols if dfs[0][c].nunique() > 1]
    if len(varied) != 3:
        raise SystemExit("Ternary comparison requires exactly three varying weights.")
    metric_a = varied[0][len("weight_"):]
    metric_b = varied[1][len("weight_"):]
    metric_c = varied[2][len("weight_"):]

    # Stable colour assignment: canonical actor slots (see plot_pareto.py's
    # make_actor_color_map). Same actor -> same colour across compare_ternary
    # and plot_pareto_multi and plot_combined_pareto_ternary, regardless of
    # which subset of actors happens to be present.
    all_actors_in_data = {a for df in dfs for a in df["Actor/Criterion"].unique()}
    color_map = make_actor_color_map(all_actors_in_data, palette=PALETTE)
    # The legend only shows actors that actually appear as consensus winners,
    # ordered by canonical slot so the legend visually agrees with the Pareto.
    winner_set = {a for df in dfs for a in df[df["is_consensus_winner"]]["Actor/Criterion"].unique()}
    actors_present = [a for a in CANONICAL_ACTOR_ORDER if a in winner_set]
    actors_present += sorted(a for a in winner_set if a not in set(CANONICAL_ACTOR_ORDER))

    n = len(dfs)
    bucket_markers = list(BUCKET_MARKERS)[:n]

    # Tiny radial offset per bucket so overlapping markers don't fully eclipse each other.
    jitter_radius = 0.012
    bucket_offsets = [
        (jitter_radius * np.cos(i * 2 * np.pi / n + np.pi / 2),
         jitter_radius * np.sin(i * 2 * np.pi / n + np.pi / 2))
        for i in range(n)
    ]

    with _style():
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.grid(False)

        # Triangle outline
        A = (0.5, np.sqrt(3) / 2)
        B = (0.0, 0.0)
        C = (1.0, 0.0)
        triangle = plt.Polygon([B, C, A], fill=False, edgecolor="#222", linewidth=1.5, zorder=2)
        ax.add_patch(triangle)

        # Light gridlines at every 0.2
        for level in (0.2, 0.4, 0.6, 0.8):
            xa1, ya1 = _barycentric_to_cartesian(level, 1 - level, 0)
            xa2, ya2 = _barycentric_to_cartesian(level, 0, 1 - level)
            xb1, yb1 = _barycentric_to_cartesian(1 - level, level, 0)
            xb2, yb2 = _barycentric_to_cartesian(0, level, 1 - level)
            xc1, yc1 = _barycentric_to_cartesian(1 - level, 0, level)
            xc2, yc2 = _barycentric_to_cartesian(0, 1 - level, level)
            for x1, y1, x2, y2 in [(xa1, ya1, xa2, ya2),
                                   (xb1, yb1, xb2, yb2),
                                   (xc1, yc1, xc2, yc2)]:
                ax.plot([x1, x2], [y1, y2], color="#ddd", linewidth=0.5, zorder=1)

        # Scatter all buckets in the same triangle. Marker shape = bucket; color = winner.
        for bucket_idx, (df, _) in enumerate(zip(dfs, labels)):
            marker = bucket_markers[bucket_idx]
            ox, oy = bucket_offsets[bucket_idx]
            winners = df[df["is_consensus_winner"]]
            for _, row in winners.iterrows():
                x, y = _barycentric_to_cartesian(
                    float(row[varied[0]]), float(row[varied[1]]), float(row[varied[2]]),
                )
                actor = row["Actor/Criterion"]
                stab = row.get("consensus_winner_stability", 1.0)
                alpha = 0.45 + 0.50 * (float(stab) if pd.notna(stab) else 1.0)
                ax.scatter(
                    x + ox, y + oy, c=[color_map[actor]], s=90, marker=marker,
                    alpha=alpha, edgecolors="black", linewidths=0.5, zorder=3,
                )

        # Vertex labels
        ax.annotate(metric_a.replace("_", " "), A, xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=11, fontweight="bold")
        ax.annotate(metric_b.replace("_", " "), B, xytext=(-10, -10),
                    textcoords="offset points", ha="right", va="top", fontsize=11, fontweight="bold")
        ax.annotate(metric_c.replace("_", " "), C, xytext=(10, -10),
                    textcoords="offset points", ha="left", va="top", fontsize=11, fontweight="bold")

        # Two legends: bucket (markers) and consensus-winner actor (colors).
        from matplotlib.lines import Line2D
        bucket_handles = [
            Line2D([0], [0], marker=bucket_markers[i], color="w",
                   markerfacecolor="#666", markeredgecolor="black", markersize=10,
                   label=labels[i])
            for i in range(n)
        ]
        actor_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[a], markeredgecolor="black", markersize=10,
                   label=a)
            for a in actors_present
        ]
        leg1 = ax.legend(
            handles=bucket_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
            title="Bucket (marker shape)", frameon=True, fontsize=9, title_fontsize=10,
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=actor_handles, loc="upper left", bbox_to_anchor=(1.02, 0.55),
            title="Consensus winner (color)", frameon=True, fontsize=9, title_fontsize=10,
        )

        ax.set_xlim(-0.18, 1.18)
        ax.set_ylim(-0.10, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=14, pad=12)

        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved {output}")


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inputs", nargs="+", required=True, type=Path,
                        help="Two or more long-format CSVs from ablate_weights.py.")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="One label per --inputs (same length).")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise SystemExit("--inputs and --labels must have the same number of entries.")
    if len(args.inputs) < 2:
        raise SystemExit("--inputs requires at least two CSVs (otherwise use plot_ablation.py).")

    dfs = [pd.read_csv(p) for p in args.inputs]
    for df, p in zip(dfs, args.inputs):
        if "is_consensus_winner" not in df.columns:
            raise SystemExit(f"{p} does not look like a multi-seed ablate_weights.py output.")

    kind = _detect_sweep_kind(dfs[0])
    for df, p in zip(dfs[1:], args.inputs[1:]):
        if _detect_sweep_kind(df) != kind:
            raise SystemExit(f"All CSVs must be the same sweep kind. {p} differs from {args.inputs[0]}.")

    title = args.title or f"Comparison across buckets  --  {kind} sweep"

    if kind == "pairwise":
        compare_pairwise(dfs, args.labels, args.output, title)
    elif kind == "ternary":
        compare_ternary(dfs, args.labels, args.output, title)
    else:
        compare_dirichlet(dfs, args.labels, args.output, title)


if __name__ == "__main__":
    main()
