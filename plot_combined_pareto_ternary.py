"""
Combined paper figure: multi-bucket Pareto (left) + ternary overlay (right),
sharing a single actor colour map and a single bucket marker map, with one
legend block placed below both panels.

Left panel: same content as `plot_pareto.py --metrics-globs ... --labels ...`
    -- axes normalised to [0, 1] with 1 = best, marker shape encodes the
    bucket, marker colour encodes the decision function, per-bucket
    Pareto-optimal points highlighted with a larger marker + thick black
    outline.

Right panel: same content as `plot_compare_ablations.py --inputs
<ternary CSVs> --labels ...` -- one ternary simplex, marker shape encodes
    the bucket, marker colour encodes the consensus winner at that grid
    point, opacity encodes inter-seed stability.

Both panels use the SAME (actor, colour) map and the SAME (bucket, marker)
map, so the joined legend below the figure applies to both.

Example
-------
    python plot_combined_pareto_ternary.py \\
        --use-case lending \\
        --pareto-x Accuracy --pareto-y Demographic_Parity \\
        --pareto-globs \\
            "results/lending/rf_base_10000/run_*/final_ranked_decision_metrics.csv" \\
            "results/lending/knn_base_10000/run_*/final_ranked_decision_metrics.csv" \\
            "results/lending/rf_stricter_10000/run_*/final_ranked_decision_metrics.csv" \\
        --ternary-inputs \\
            ablations/rf_base_10000_ternary.csv \\
            ablations/knn_base_10000_ternary.csv \\
            ablations/rf_stricter_10000_ternary.csv \\
        --labels "RF / base" "KNN / base" "RF / strictest" \\
        --output figs/combined_pareto_ternary.png
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

from plot_pareto import (
    BUCKET_MARKERS,
    CANONICAL_ACTOR_ORDER,
    DEFAULT_EXCLUDE as PARETO_EXCLUDE,
    _aggregate_metric_means_sems,
    _normalize_for_display,
    _sem_in_norm_space,
    make_actor_color_map,
    pareto_front_indices,
)


# Shared actor palette (must match plot_compare_ablations.py's PALETTE).
ACTOR_PALETTE = plt.get_cmap("tab10").colors + plt.get_cmap("Set2").colors


# ----------------------------------------------------------------------
def _load_use_case_yaml(use_case: str) -> dict:
    path = Path(__file__).parent / "conf" / "use_case" / f"{use_case}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ======================================================================
# Left panel: multi-bucket Pareto rendered into an existing Axes.
# ======================================================================
def _render_pareto_multi_into(
    ax,
    bucket_dfs: list[pd.DataFrame],
    bucket_labels: list[str],
    x_metric: str,
    y_metric: str,
    use_case_cfg: dict,
    actor_color: dict,
    bucket_markers: list[str],
    n_seeds_max: int,
):
    """Draw the multi-bucket Pareto plot into `ax`. No legend, no title."""
    ranking_criteria = dict(use_case_cfg["criteria"]["ranking_criteria"])
    x_dir = ranking_criteria.get(x_metric, "max")
    y_dir = ranking_criteria.get(y_metric, "max")

    rows = []
    for bucket_idx, (df, label) in enumerate(zip(bucket_dfs, bucket_labels)):
        for _, row in df.iterrows():
            rows.append({
                "Actor/Criterion": row["Actor/Criterion"],
                "bucket": label,
                "bucket_idx": bucket_idx,
                x_metric: float(row[x_metric]),
                y_metric: float(row[y_metric]),
                f"{x_metric}_sem": float(row.get(f"{x_metric}_sem", 0.0) or 0.0),
                f"{y_metric}_sem": float(row.get(f"{y_metric}_sem", 0.0) or 0.0),
            })
    combined = pd.DataFrame(rows)

    x_norm = _normalize_for_display(combined[x_metric], x_dir, x_metric).values
    y_norm = _normalize_for_display(combined[y_metric], y_dir, y_metric).values
    x_sem_n = _sem_in_norm_space(combined[f"{x_metric}_sem"], x_dir, x_metric, combined[x_metric]).values
    y_sem_n = _sem_in_norm_space(combined[f"{y_metric}_sem"], y_dir, y_metric, combined[y_metric]).values

    # Per-bucket Pareto in the normalized frame.
    is_pareto_per_bucket = np.zeros(len(combined), dtype=bool)
    for b_idx in combined["bucket_idx"].unique():
        mask = (combined["bucket_idx"] == b_idx).values
        pts = np.column_stack([x_norm[mask], y_norm[mask]])
        sub_pareto = pareto_front_indices(pts)
        pos = np.where(mask)[0]
        is_pareto_per_bucket[pos[sub_pareto]] = True

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    if n_seeds_max > 1 and (np.any(x_sem_n > 0) or np.any(y_sem_n > 0)):
        ax.errorbar(
            x_norm, y_norm, xerr=x_sem_n, yerr=y_sem_n,
            fmt="none", ecolor="lightgray", elinewidth=1.0, capsize=2.5,
            alpha=0.55, zorder=1,
        )

    for bucket_idx, _ in enumerate(bucket_labels):
        sub_mask = (combined["bucket_idx"] == bucket_idx).values
        if not sub_mask.any():
            continue
        sub_pareto = is_pareto_per_bucket[sub_mask]
        sizes = np.where(sub_pareto, 320, 200)
        edge_colors = np.where(sub_pareto, "black", "#999")
        edge_widths = np.where(sub_pareto, 2.2, 0.6)
        colors = [actor_color[a] for a in combined.loc[sub_mask, "Actor/Criterion"]]
        ax.scatter(
            x_norm[sub_mask], y_norm[sub_mask],
            s=sizes, c=colors,
            marker=bucket_markers[bucket_idx],
            edgecolors=edge_colors, linewidths=edge_widths,
            alpha=0.94, zorder=3,
        )

    ax.set_xlabel(f"{x_metric.replace('_', ' ')}", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"{y_metric.replace('_', ' ')}", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    return combined, is_pareto_per_bucket


# ======================================================================
# Right panel: ternary overlay rendered into an existing Axes.
# ======================================================================
def _barycentric_to_cartesian(wa, wb, wc):
    return 0.5 * wa + 1.0 * wc, (np.sqrt(3) / 2) * wa


def _weight_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("weight_")]


def _render_compare_ternary_into(
    ax,
    dfs: list[pd.DataFrame],
    labels: list[str],
    actor_color: dict,
    bucket_markers: list[str],
):
    """Draw the multi-bucket ternary overlay into `ax`. No legend, no title."""
    weight_cols = _weight_columns(dfs[0])
    varied = [c for c in weight_cols if dfs[0][c].nunique() > 1]
    if len(varied) != 3:
        raise SystemExit("Ternary comparison requires exactly three varying weights.")
    metric_a = varied[0][len("weight_"):]
    metric_b = varied[1][len("weight_"):]
    metric_c = varied[2][len("weight_"):]

    n = len(dfs)
    jitter_radius = 0.012
    bucket_offsets = [
        (jitter_radius * np.cos(i * 2 * np.pi / n + np.pi / 2),
         jitter_radius * np.sin(i * 2 * np.pi / n + np.pi / 2))
        for i in range(n)
    ]

    ax.grid(False)
    A = (0.5, np.sqrt(3) / 2)
    B = (0.0, 0.0)
    C = (1.0, 0.0)
    triangle = plt.Polygon([B, C, A], fill=False, edgecolor="#222",
                           linewidth=1.5, zorder=2)
    ax.add_patch(triangle)

    # Gridlines every 0.2 (constant wA / wB / wC)
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

    for bucket_idx, df in enumerate(dfs):
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
                x + ox, y + oy, c=[actor_color[actor]], s=90, marker=marker,
                alpha=alpha, edgecolors="black", linewidths=0.5, zorder=3,
            )

    # Vertex labels
    ax.annotate(metric_a.replace("_", " "), A, xytext=(0, 14),
                textcoords="offset points", ha="center", fontsize=13, fontweight="bold")
    ax.annotate(metric_b.replace("_", " "), B, xytext=(-10, -10),
                textcoords="offset points", ha="right", va="top", fontsize=13, fontweight="bold")
    ax.annotate(metric_c.replace("_", " "), C, xytext=(10, -10),
                textcoords="offset points", ha="left", va="top", fontsize=13, fontweight="bold")

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.10, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


# ======================================================================
# Orchestration
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--use-case", required=True, choices=("lending", "health"))
    parser.add_argument("--pareto-x", required=True,
                        help="Metric name for the Pareto x-axis (e.g. Accuracy).")
    parser.add_argument("--pareto-y", required=True,
                        help="Metric name for the Pareto y-axis (e.g. Demographic_Parity).")
    parser.add_argument("--pareto-globs", nargs="+", required=True,
                        help="Per-bucket glob patterns to seed CSVs for the Pareto panel.")
    parser.add_argument("--ternary-inputs", nargs="+", required=True, type=Path,
                        help="Per-bucket long-format ternary CSVs from ablate_weights.py.")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Per-bucket labels; same length and order as --pareto-globs "
                             "and --ternary-inputs.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-all", action="store_true",
                        help="Do not exclude any actors from the Pareto panel.")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help=f"Override the default Pareto-panel exclusion list "
                             f"(currently: {sorted(PARETO_EXCLUDE)}).")
    args = parser.parse_args()

    if not (len(args.pareto_globs) == len(args.ternary_inputs) == len(args.labels)):
        raise SystemExit(
            "--pareto-globs, --ternary-inputs, and --labels must have the same length."
        )
    n_buckets = len(args.labels)

    # ------------------------------------------------------------------
    # Load Pareto data (aggregate seeds per bucket).
    # ------------------------------------------------------------------
    if args.include_all:
        excluded = set()
    elif args.exclude is not None:
        excluded = set(args.exclude)
    else:
        excluded = set(PARETO_EXCLUDE)

    pareto_dfs = []
    n_seeds_max = 1
    for label, glob_pat in zip(args.labels, args.pareto_globs):
        csvs = sorted(glob.glob(glob_pat))
        if not csvs:
            print(f"WARN: no CSVs for bucket '{label}' (glob '{glob_pat}'); skipping.")
            continue
        df = _aggregate_metric_means_sems(csvs, args.pareto_x, args.pareto_y)
        if excluded:
            df = df[~df["Actor/Criterion"].isin(excluded)]
        n_seeds_max = max(n_seeds_max, df.attrs.get("n_seeds_max", len(csvs)))
        print(f"[Pareto] {label}: {len(csvs)} seeds, {len(df)} actors after filtering.")
        pareto_dfs.append(df)
    if not pareto_dfs:
        raise SystemExit("No Pareto-panel data.")

    # ------------------------------------------------------------------
    # Load ternary data.
    # ------------------------------------------------------------------
    ternary_dfs = [pd.read_csv(p) for p in args.ternary_inputs]

    # ------------------------------------------------------------------
    # Shared colour and marker maps.
    # ------------------------------------------------------------------
    # Union of all actors appearing in either panel's data. Colours come from
    # the canonical (actor -> palette slot) map so adding/dropping an actor
    # (e.g. bringing Outcome_Pred_Model back in) does not shift anyone else.
    all_actors_union = set().union(
        *(set(df["Actor/Criterion"].unique()) for df in pareto_dfs),
        *(set(df["Actor/Criterion"].unique()) for df in ternary_dfs),
    )
    actor_color = make_actor_color_map(all_actors_union, palette=ACTOR_PALETTE)
    bucket_markers = list(BUCKET_MARKERS)[:n_buckets]

    # ------------------------------------------------------------------
    # Draw.
    # ------------------------------------------------------------------
    fig, (ax_pareto, ax_ternary) = plt.subplots(1, 2, figsize=(19, 8.5))

    use_case_cfg = _load_use_case_yaml(args.use_case)
    _render_pareto_multi_into(
        ax_pareto, pareto_dfs, args.labels, args.pareto_x, args.pareto_y,
        use_case_cfg, actor_color, bucket_markers, n_seeds_max,
    )
    _render_compare_ternary_into(
        ax_ternary, ternary_dfs, args.labels, actor_color, bucket_markers,
    )

    # ------------------------------------------------------------------
    # Single shared legend block below the figure.
    # ------------------------------------------------------------------
    # Only actors that appear in EITHER panel's data get a colour entry.
    # Ordered by canonical slot so the legend agrees with the palette order
    # (Applicant, Bank, Compromise Programming, ... Regulatory, Outcome_Pred_Model).
    shown_set = set().union(
        *(set(df["Actor/Criterion"].unique()) for df in pareto_dfs),
        *(set(df[df["is_consensus_winner"]]["Actor/Criterion"].unique())
          for df in ternary_dfs),
    )
    actors_shown = [a for a in CANONICAL_ACTOR_ORDER if a in shown_set]
    actors_shown += sorted(a for a in shown_set if a not in set(CANONICAL_ACTOR_ORDER))

    bucket_handles = [
        Line2D([0], [0], marker=bucket_markers[i], color="w",
               markerfacecolor="#666", markeredgecolor="black", markersize=12,
               label=args.labels[i])
        for i in range(n_buckets)
    ]
    actor_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=actor_color[a], markeredgecolor="black", markersize=12,
               label=a)
        for a in actors_shown
    ]

    # Two stacked legends, centred under the figure.
    leg1 = fig.legend(
        handles=bucket_handles,
        loc="upper center", bbox_to_anchor=(0.5, 0.03),
        ncol=max(1, n_buckets),
        title="Bucket (marker shape)",
        title_fontsize=13, fontsize=13, frameon=True, edgecolor="lightgray",
    )
    fig.legend(
        handles=actor_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.05),
        ncol=min(max(1, len(actor_handles)), 5),
        title="Decision function (color)",
        title_fontsize=13, fontsize=13, frameon=True, edgecolor="lightgray",
    )

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
