"""
Classical 2-D Pareto frontier, multi-seed and multi-bucket aware.

Single-bucket mode
------------------
Pass `--metrics-glob` (preferred) or `--metrics-csv`. Each actor becomes a
point in `(x_metric, y_metric)` space at the mean across seeds, with SEM
error bars on both axes. The Pareto-optimal (non-dominated) set is
highlighted with a red dashed frontier and star markers.

Multi-bucket mode
-----------------
Pass `--metrics-globs` (plural) with one glob per bucket and `--labels` with
a label per bucket. Each (actor, bucket) becomes a point; colour and marker
shape both encode the bucket; the Pareto frontier is computed **across all
buckets combined**, so it directly answers "which (actor, configuration)
combinations are globally non-dominated?".

Optional overlay
----------------
`--overlay-ablation` uses an ablate_weights.py long-format CSV to scale
each actor's marker by how often it is the consensus winner across the
sweep. Only meaningful in single-bucket mode.

Reference / baseline actors (`Oracle`, `Random`, `Outcome_Pred_Model`,
`Outcome_Maxim`) are excluded by default; override with `--include-all` or
pass an explicit `--actors` list.
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


DEFAULT_EXCLUDE = {"Oracle", "Random", "Outcome_Pred_Model", "Outcome_Maxim"}
BUCKET_MARKERS = ("o", "s", "^", "D", "P", "X")
BUCKET_PALETTE = plt.get_cmap("tab10").colors


# ----------------------------------------------------------------------
def _load_use_case_yaml(use_case: str) -> dict:
    path = Path(__file__).parent / "conf" / "use_case" / f"{use_case}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _higher_better_score(series: pd.Series, direction: str) -> pd.Series:
    if direction == "max":  return series
    if direction == "min":  return -series
    if direction == "zero": return -series.abs()
    raise ValueError(f"Unknown ranking direction '{direction}'")


def pareto_front_indices(points: np.ndarray) -> np.ndarray:
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                is_pareto[i] = False
                break
    return is_pareto


def _axis_label(metric: str, direction: str) -> str:
    base = metric.replace("_", " ")
    if direction == "zero": return f"{base}  (|.|  --  0 is fairer)"
    if direction == "min":  return f"{base}  (lower is better)"
    return f"{base}  (higher is better)"


def _winner_counts_from_ablation(ablation_df: pd.DataFrame) -> pd.Series:
    """Count consensus-winner rows in the long-format ablation CSV."""
    if "is_consensus_winner" not in ablation_df.columns:
        raise SystemExit(
            "Overlay CSV does not have an `is_consensus_winner` column. "
            "Regenerate it with the multi-seed ablate_weights.py."
        )
    winners = ablation_df[ablation_df["is_consensus_winner"]]
    return winners["Actor/Criterion"].value_counts()


# ----------------------------------------------------------------------
# Multi-seed aggregation of raw metrics
# ----------------------------------------------------------------------
def _aggregate_metric_means_sems(
    csv_paths: list[str], x_metric: str, y_metric: str
) -> pd.DataFrame:
    """For each Actor/Criterion, return mean / SEM of x and y across CSVs."""
    frames = [pd.read_csv(p) for p in csv_paths]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "Actor/Criterion" not in combined.columns:
        raise SystemExit("CSVs are missing the 'Actor/Criterion' column.")
    rows = []
    n_seeds = len(frames)
    for actor, sub in combined.groupby("Actor/Criterion"):
        x_vals = sub[x_metric].dropna().values
        y_vals = sub[y_metric].dropna().values
        if len(x_vals) == 0 or len(y_vals) == 0:
            continue
        row = {
            "Actor/Criterion": actor,
            x_metric: float(np.mean(x_vals)),
            y_metric: float(np.mean(y_vals)),
            f"{x_metric}_sem": (float(np.std(x_vals, ddof=1) / math.sqrt(len(x_vals)))
                               if len(x_vals) > 1 else 0.0),
            f"{y_metric}_sem": (float(np.std(y_vals, ddof=1) / math.sqrt(len(y_vals)))
                               if len(y_vals) > 1 else 0.0),
            "n_seeds": int(min(len(x_vals), len(y_vals))),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["n_seeds_max"] = n_seeds
    return out


# ----------------------------------------------------------------------
def plot_pareto(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    use_case_cfg: dict,
    output: Path,
    win_counts: pd.Series | None = None,
    annotate: bool = True,
    n_seeds: int = 1,
):
    ranking_criteria = dict(use_case_cfg["criteria"]["ranking_criteria"])
    x_dir = ranking_criteria.get(x_metric, "max")
    y_dir = ranking_criteria.get(y_metric, "max")

    x_raw = df[x_metric].values
    y_raw = df[y_metric].values
    x_err = df[f"{x_metric}_sem"].values if f"{x_metric}_sem" in df.columns else None
    y_err = df[f"{y_metric}_sem"].values if f"{y_metric}_sem" in df.columns else None

    x_better = _higher_better_score(df[x_metric], x_dir).values
    y_better = _higher_better_score(df[y_metric], y_dir).values
    is_pareto = pareto_front_indices(np.column_stack([x_better, y_better]))

    fig, ax = plt.subplots(figsize=(10, 7))

    actors = df["Actor/Criterion"].reset_index(drop=True).tolist()

    # Marker sizes scale by winning frequency if overlay given.
    if win_counts is not None and not win_counts.empty:
        max_wins = int(win_counts.max())
        sizes = np.array(
            [260 if is_pareto[i] else 110 for i in range(len(df))], dtype=float,
        )
        for i, a in enumerate(actors):
            if a in win_counts.index and max_wins > 0:
                sizes[i] *= 1.0 + 2.0 * (win_counts[a] / max_wins) ** 0.5
    else:
        sizes = np.array([260 if is_pareto[i] else 110 for i in range(len(df))], dtype=float)

    # Per-point error bars first (so they sit under the marker)
    if n_seeds > 1 and (x_err is not None or y_err is not None):
        ax.errorbar(
            x_raw, y_raw,
            xerr=x_err if x_err is not None else None,
            yerr=y_err if y_err is not None else None,
            fmt="none", ecolor="lightgray", elinewidth=1.0, capsize=2.5, zorder=2,
        )

    for idx in range(len(df)):
        actor = actors[idx]
        is_p = bool(is_pareto[idx])
        ax.scatter(
            x_raw[idx], y_raw[idx],
            s=sizes[idx],
            c="crimson" if is_p else "steelblue",
            marker="*" if is_p else "o",
            edgecolors="black", linewidths=0.8, alpha=0.95, zorder=3,
        )
        if annotate:
            label = str(actor)
            if win_counts is not None and actor in win_counts.index:
                label += f"  ({int(win_counts[actor])} wins)"
            ax.annotate(
                label, (x_raw[idx], y_raw[idx]),
                xytext=(7, 7), textcoords="offset points",
                fontsize=8,
                fontweight="bold" if is_p else "normal",
            )

    # Pareto frontier curve
    pareto_df = df.reset_index(drop=True).loc[is_pareto].copy()
    pareto_df["__x_better__"] = x_better[is_pareto]
    pareto_df = pareto_df.sort_values("__x_better__")
    if len(pareto_df) >= 2:
        ax.plot(
            pareto_df[x_metric], pareto_df[y_metric],
            color="crimson", linestyle="--", linewidth=1.6, alpha=0.7,
            label="Pareto frontier", zorder=2,
        )

    ax.set_xlabel(_axis_label(x_metric, x_dir), fontsize=11)
    ax.set_ylabel(_axis_label(y_metric, y_dir), fontsize=11)
    seed_note = (
        f" -- mean across {n_seeds} seeds (error bars = SEM)" if n_seeds > 1 else ""
    )
    title_extra = "  (marker size = winning frequency in the sweep)" if win_counts is not None else ""
    ax.set_title(
        f"Pareto frontier: {x_metric.replace('_', ' ')} vs {y_metric.replace('_', ' ')}{seed_note}"
        f"{title_extra}",
        fontsize=12,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")
    pareto_rows = df.reset_index(drop=True).loc[is_pareto, "Actor/Criterion"].tolist()
    print(f"Pareto-optimal actors: {pareto_rows}")


# ----------------------------------------------------------------------
# Multi-bucket Pareto -- one figure, colour + marker = bucket
# ----------------------------------------------------------------------
def plot_pareto_multi(
    bucket_dfs: list[pd.DataFrame],
    bucket_labels: list[str],
    x_metric: str,
    y_metric: str,
    use_case_cfg: dict,
    output: Path,
    annotate: bool = True,
    n_seeds_max: int = 1,
):
    ranking_criteria = dict(use_case_cfg["criteria"]["ranking_criteria"])
    x_dir = ranking_criteria.get(x_metric, "max")
    y_dir = ranking_criteria.get(y_metric, "max")

    # Build a unified frame with (Actor, Bucket, raw x/y, sem x/y).
    rows = []
    for bucket_idx, (df, label) in enumerate(zip(bucket_dfs, bucket_labels)):
        for _, row in df.iterrows():
            rows.append({
                "Actor/Criterion": row["Actor/Criterion"],
                "bucket": label,
                "bucket_idx": bucket_idx,
                x_metric: row[x_metric],
                y_metric: row[y_metric],
                f"{x_metric}_sem": float(row.get(f"{x_metric}_sem", 0.0) or 0.0),
                f"{y_metric}_sem": float(row.get(f"{y_metric}_sem", 0.0) or 0.0),
            })
    combined = pd.DataFrame(rows)
    if combined.empty:
        raise SystemExit("No rows after combining buckets.")

    x_raw = combined[x_metric].values
    y_raw = combined[y_metric].values
    x_better = _higher_better_score(combined[x_metric], x_dir).values
    y_better = _higher_better_score(combined[y_metric], y_dir).values
    is_pareto = pareto_front_indices(np.column_stack([x_better, y_better]))

    fig, ax = plt.subplots(figsize=(11.5, 8))

    n = len(bucket_dfs)
    bucket_colors = [BUCKET_PALETTE[i % 10] for i in range(n)]
    bucket_markers = list(BUCKET_MARKERS)[:n]

    # Plot one bucket at a time so legend handles map cleanly to (color, marker).
    for bucket_idx, label in enumerate(bucket_labels):
        sub_mask = (combined["bucket_idx"] == bucket_idx).values
        if not sub_mask.any():
            continue
        sub = combined[sub_mask].reset_index(drop=True)
        sub_pareto = is_pareto[sub_mask]
        x_err = sub[f"{x_metric}_sem"].values
        y_err = sub[f"{y_metric}_sem"].values

        if n_seeds_max > 1 and (np.any(x_err > 0) or np.any(y_err > 0)):
            ax.errorbar(
                sub[x_metric].values, sub[y_metric].values,
                xerr=x_err, yerr=y_err,
                fmt="none", ecolor=bucket_colors[bucket_idx],
                elinewidth=1.0, capsize=2.5, alpha=0.40, zorder=2,
            )

        # Sizes / edges differentiate Pareto-optimal vs dominated.
        sizes = np.where(sub_pareto, 270, 110)
        edge_widths = np.where(sub_pareto, 1.8, 0.6)
        ax.scatter(
            sub[x_metric].values, sub[y_metric].values,
            s=sizes, c=[bucket_colors[bucket_idx]] * len(sub),
            marker=bucket_markers[bucket_idx],
            edgecolors="black", linewidths=edge_widths,
            alpha=0.92, zorder=3, label=label,
        )

    # Annotate ONLY the Pareto-optimal points (the "winners"). Dominated
    # points are still drawn but un-labelled, so the figure stays clean and
    # the eye goes straight to what matters.
    if annotate:
        pareto_indices = np.where(is_pareto)[0]
        if len(pareto_indices):
            # Offset labels radially outward from the centroid of the
            # Pareto-optimal points (so the labels sit on the outside, away
            # from the dominated cloud and away from each other).
            cx = float(combined[x_metric].iloc[pareto_indices].mean())
            cy = float(combined[y_metric].iloc[pareto_indices].mean())
            angles = np.arctan2(
                combined[y_metric].iloc[pareto_indices].values - cy,
                combined[x_metric].iloc[pareto_indices].values - cx,
            )
            order = np.argsort(angles)
            radii = [55, 75, 95]   # stagger so close-by labels separate
            for j, idx in enumerate(pareto_indices[order]):
                row = combined.iloc[idx]
                actor = row["Actor/Criterion"]
                bucket = row["bucket"]
                radius = radii[j % len(radii)]
                ang = angles[order[j]]
                dx = radius * np.cos(ang)
                dy = radius * np.sin(ang)
                ax.annotate(
                    f"{actor}\n({bucket})",
                    xy=(row[x_metric], row[y_metric]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, fontweight="bold",
                    ha="center", va="center", color="#111",
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="#444", alpha=0.95, linewidth=1.0),
                    arrowprops=dict(arrowstyle="-", color="#444",
                                    alpha=0.7, linewidth=0.9,
                                    shrinkA=0.5, shrinkB=4),
                    zorder=5,
                )

    ax.set_xlabel(_axis_label(x_metric, x_dir), fontsize=11)
    ax.set_ylabel(_axis_label(y_metric, y_dir), fontsize=11)
    seed_note = f" -- mean across {n_seeds_max} seeds; error bars = SEM" if n_seeds_max > 1 else ""
    ax.set_title(
        f"Pareto-optimal strategies across buckets: "
        f"{x_metric.replace('_', ' ')} vs {y_metric.replace('_', ' ')}{seed_note}\n"
        f"(larger markers = Pareto-optimal in the combined population)",
        fontsize=12,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=True, title="Bucket")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")
    pareto_pairs = combined.loc[is_pareto, ["Actor/Criterion", "bucket"]].values.tolist()
    print(f"Pareto-optimal (actor, bucket) pairs: {pareto_pairs}")


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_argument_group("Input (one of)")
    src.add_argument("--metrics-glob",
                     help="(single bucket) Glob across per-seed CSVs at one configuration.")
    src.add_argument("--metrics-csv", type=Path,
                     help="(single bucket, legacy) A single CSV; no SEM error bars.")
    src.add_argument("--metrics-globs", nargs="+",
                     help="(multi-bucket) One glob per bucket. Pair with --labels.")
    parser.add_argument("--labels", nargs="+",
                        help="(multi-bucket) One label per bucket, same length as --metrics-globs.")
    parser.add_argument("--use-case", required=True, choices=("lending", "health"))
    parser.add_argument("--x", required=True, help="Metric name for the x-axis.")
    parser.add_argument("--y", required=True, help="Metric name for the y-axis.")
    parser.add_argument("--output", type=Path, default=Path("figs/pareto.png"))
    parser.add_argument("--overlay-ablation", type=Path, default=None,
                        help="Optional long-format ablation CSV (from ablate_weights.py). "
                             "Used to scale marker sizes by winning frequency.")
    parser.add_argument("--actors", nargs="*",
                        help="Whitelist of Actor/Criterion rows to plot. "
                             "If omitted, the default exclusion list is applied.")
    parser.add_argument("--include-all", action="store_true",
                        help="Disable the default exclusion of Oracle / Random / "
                             "Outcome_Pred_Model / Outcome_Maxim.")
    parser.add_argument("--no-annotate", action="store_true",
                        help="Skip text labels on points.")
    args = parser.parse_args()

    n_inputs = sum(1 for v in (args.metrics_glob, args.metrics_csv, args.metrics_globs) if v)
    if n_inputs != 1:
        raise SystemExit(
            "Provide exactly one of --metrics-glob (single bucket, multi-seed), "
            "--metrics-csv (single bucket, single seed), or --metrics-globs (multi-bucket)."
        )

    use_case_cfg = _load_use_case_yaml(args.use_case)

    # --------------- multi-bucket branch ---------------
    if args.metrics_globs:
        if not args.labels or len(args.labels) != len(args.metrics_globs):
            raise SystemExit("--labels must be provided and match --metrics-globs in length.")
        if args.overlay_ablation:
            print("Note: --overlay-ablation is ignored in multi-bucket mode "
                  "(too much encoding in one figure).")
        bucket_dfs = []
        n_seeds_max = 1
        for label, glob_pat in zip(args.labels, args.metrics_globs):
            csvs = sorted(glob.glob(glob_pat))
            if not csvs:
                print(f"WARN: no CSVs for bucket '{label}' (glob '{glob_pat}'); skipping.")
                continue
            bucket_df = _aggregate_metric_means_sems(csvs, args.x, args.y)
            n_seeds_max = max(n_seeds_max, bucket_df.attrs.get("n_seeds_max", len(csvs)))
            if args.actors:
                bucket_df = bucket_df[bucket_df["Actor/Criterion"].isin(args.actors)]
            elif not args.include_all:
                bucket_df = bucket_df[~bucket_df["Actor/Criterion"].isin(DEFAULT_EXCLUDE)]
            print(f"Bucket {label!r}: {len(csvs)} seeds, {len(bucket_df)} actors after filtering.")
            bucket_dfs.append((bucket_df, label))
        if not bucket_dfs:
            raise SystemExit("No usable buckets after globbing.")
        plot_pareto_multi(
            [b for b, _ in bucket_dfs], [l for _, l in bucket_dfs],
            args.x, args.y, use_case_cfg, args.output,
            annotate=not args.no_annotate, n_seeds_max=n_seeds_max,
        )
        return

    # --------------- single-bucket branch ---------------
    if args.metrics_glob:
        csvs = sorted(glob.glob(args.metrics_glob))
        if not csvs:
            raise SystemExit(f"No CSVs matched '{args.metrics_glob}'.")
        print(f"Aggregating across {len(csvs)} seed runs.")
        df = _aggregate_metric_means_sems(csvs, args.x, args.y)
        n_seeds = df.attrs.get("n_seeds_max", len(csvs))
    else:
        df = pd.read_csv(args.metrics_csv)
        n_seeds = 1

    if args.actors:
        df = df[df["Actor/Criterion"].isin(args.actors)]
    elif not args.include_all:
        df = df[~df["Actor/Criterion"].isin(DEFAULT_EXCLUDE)]
    if df.empty:
        raise SystemExit("No rows after filtering. Check --actors / --include-all against the CSV.")

    win_counts = None
    if args.overlay_ablation is not None:
        win_counts = _winner_counts_from_ablation(pd.read_csv(args.overlay_ablation))

    plot_pareto(
        df, args.x, args.y, use_case_cfg, args.output,
        win_counts=win_counts, annotate=not args.no_annotate, n_seeds=n_seeds,
    )


if __name__ == "__main__":
    main()
