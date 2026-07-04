"""
Paper-quality plots from the long-format multi-seed CSVs produced by
`ablate_weights.py`.

The CSV is expected to have these columns:
    config_id, weight_<metric>..., Actor/Criterion,
    weighted_sum_mean, weighted_sum_sem, n_seeds,
    rank_within_config, is_consensus_winner, consensus_winner_stability

Auto-detects the sweep type from how the weight columns vary:

* **pairwise** -- two weights sum to 1, all others are zero. Plot:
    top:    Weighted-Sum (mean) vs. the varying weight, one line per actor.
            A `±SEM` band is shaded around each line so the seed-level
            uncertainty is visible.
    bottom: a "winner-region" strip color-coded by which actor wins (on the
            mean) on each segment; consensus stability is annotated below
            the strip so the reader sees whether the winner is robust.

* **ternary** -- three weights sum to 1, all others are zero. Plot: a
    triangular simplex with each grid point colored by the consensus
    winner; opacity scales with inter-seed stability. Reveals the
    *geometry* of the winner regions in a three-objective trade-off.

* **dirichlet** -- many weights vary. Plot:
    left:  horizontal bars of "how often does each rule win across the
           sweep" with absolute counts, percentages, and the mean stability
           across the configs where each rule wins.
    right: violin plot of the per-config Weighted-Sum mean, ordered so the
           rule with the highest median is at the top.

Both views use the same `tab10 + Set2` palette so the same actor always
reads as the same colour.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = plt.get_cmap("tab10").colors + plt.get_cmap("Set2").colors


def _style():
    return plt.rc_context({
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.edgecolor": "#333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ----------------------------------------------------------------------
def _weight_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("weight_")]


def _detect_sweep_kind(df: pd.DataFrame) -> str:
    weight_cols = _weight_columns(df)
    varied = [c for c in weight_cols if df[c].nunique() > 1]
    if len(varied) == 2 and np.allclose(df[varied].sum(axis=1).unique(), 1.0):
        return "pairwise"
    if len(varied) == 3 and np.allclose(df[varied].sum(axis=1).unique(), 1.0):
        return "ternary"
    return "dirichlet"


def _actor_color_map(actors: list[str]) -> dict:
    return {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(actors)}


# ======================================================================
# Pairwise sweep -- mean lines with ±SEM bands, winner strip, stability bar
# ======================================================================
def plot_pairwise(df: pd.DataFrame, output: Path, title: str):
    weight_cols = _weight_columns(df)
    varied = [c for c in weight_cols if df[c].nunique() > 1]
    x_col = varied[0]

    n_seeds = int(df["n_seeds"].max()) if "n_seeds" in df.columns else 1

    actors = sorted(
        df["Actor/Criterion"].unique(),
        key=lambda a: -df[df["Actor/Criterion"] == a]["weighted_sum_mean"].max(),
    )
    color_map = _actor_color_map(actors)

    with _style():
        fig, (ax_top, ax_strip) = plt.subplots(
            2, 1, figsize=(12, 7.2),
            gridspec_kw={"height_ratios": [5, 1.1], "hspace": 0.12},
            sharex=True,
        )

        # Consensus winner at each x value (from is_consensus_winner).
        x_vals = sorted(df[x_col].unique())
        winners_by_x: dict[float, str] = {}
        stability_by_x: dict[float, float] = {}
        for x in x_vals:
            sub = df[(df[x_col] == x) & df["is_consensus_winner"]]
            if sub.empty:
                continue
            row = sub.iloc[0]
            winners_by_x[x] = row["Actor/Criterion"]
            stab = row["consensus_winner_stability"]
            stability_by_x[x] = float(stab) if pd.notna(stab) else float("nan")

        # Lines + SEM bands + enlarged-winner markers
        for actor in actors:
            sub = df[df["Actor/Criterion"] == actor].sort_values(x_col)
            color = color_map[actor]
            ax_top.plot(
                sub[x_col], sub["weighted_sum_mean"],
                marker="o", markersize=4.5, linewidth=1.7,
                color=color, label=actor, alpha=0.95,
            )
            if "weighted_sum_sem" in sub.columns and n_seeds > 1:
                lower = sub["weighted_sum_mean"] - sub["weighted_sum_sem"]
                upper = sub["weighted_sum_mean"] + sub["weighted_sum_sem"]
                ax_top.fill_between(sub[x_col], lower, upper, color=color, alpha=0.15, linewidth=0)
            win_mask = sub[x_col].map(lambda x: winners_by_x.get(x) == actor)
            if win_mask.any():
                ax_top.scatter(
                    sub.loc[win_mask, x_col],
                    sub.loc[win_mask, "weighted_sum_mean"],
                    s=110, facecolor=color, edgecolor="black",
                    linewidth=1.0, zorder=5,
                )

        # Vertical crossover guides
        for i in range(1, len(x_vals)):
            if winners_by_x.get(x_vals[i]) != winners_by_x.get(x_vals[i - 1]):
                ax_top.axvline(x_vals[i], color="black", linestyle=":", linewidth=0.8, alpha=0.4)

        seed_note = f"  (mean ± SEM across {n_seeds} seeds)" if n_seeds > 1 else "  (single seed)"
        ax_top.set_ylabel(f"Weighted Normalized-Sum{seed_note}", fontsize=12)
        ax_top.set_title(title, fontsize=13.5, pad=15)
        ax_top.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            frameon=True, edgecolor="lightgray",
            title="Compromise rule", title_fontsize=10,
        )
        y_lo, y_hi = ax_top.get_ylim()
        pad = (y_hi - y_lo) * 0.05
        ax_top.set_ylim(y_lo - pad, y_hi + pad)

        # Winner-region strip + stability annotation
        ax_strip.grid(False)
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
            # Alpha encodes stability: opaque = stable, faint = noisy.
            stab = stability_by_x.get(x, 1.0)
            alpha = 0.35 + 0.55 * (stab if not np.isnan(stab) else 1.0)
            ax_strip.axvspan(left, right, color=color_map[winner], alpha=alpha)

        # Region labels
        prev_winner = None
        region_start_x = None
        for i, x in enumerate(x_vals):
            w = winners_by_x.get(x)
            if w != prev_winner:
                if prev_winner is not None and region_start_x is not None:
                    midx = (region_start_x + x_vals[i - 1]) / 2
                    ax_strip.text(midx, 0.5, prev_winner, ha="center", va="center",
                                  fontsize=9, fontweight="bold", color="white")
                prev_winner = w
                region_start_x = x
        if prev_winner is not None and region_start_x is not None:
            midx = (region_start_x + x_vals[-1]) / 2
            ax_strip.text(midx, 0.5, prev_winner, ha="center", va="center",
                          fontsize=9, fontweight="bold", color="white")

        ax_strip.set_ylim(0, 1)
        ax_strip.set_yticks([])
        ax_strip.set_xlabel(
            x_col.replace("weight_", "Weight on ").replace("_", " "), fontsize=12,
        )
        if n_seeds > 1:
            stab_subtitle = (
                f"Consensus winner per weight value "
                f"(strip opacity = inter-seed stability, range [0.35, 0.90])"
            )
        else:
            stab_subtitle = "Consensus winner per weight value"
        ax_strip.set_title(stab_subtitle, fontsize=10, pad=4, loc="left")
        for s in ("top", "right", "left"):
            ax_strip.spines[s].set_visible(False)

        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved {output}")


# ======================================================================
# Ternary sweep -- 3-metric simplex with winner regions
# ======================================================================
def _barycentric_to_cartesian(wa: float, wb: float, wc: float) -> tuple[float, float]:
    """Map (wA, wB, wC) with wA+wB+wC = 1 to a point in an equilateral triangle.

    Convention: vertex A at top, B bottom-left, C bottom-right.
    """
    # A = (0.5, sqrt(3)/2),  B = (0, 0),  C = (1, 0)
    x = 0.5 * wa + 0.0 * wb + 1.0 * wc
    y = (np.sqrt(3) / 2) * wa
    return x, y


def plot_ternary(df: pd.DataFrame, output: Path, title: str):
    weight_cols = _weight_columns(df)
    varied = [c for c in weight_cols if df[c].nunique() > 1]
    if len(varied) != 3:
        raise ValueError(f"plot_ternary expects 3 varying weight columns; got {len(varied)}.")

    metric_a = varied[0][len("weight_"):]
    metric_b = varied[1][len("weight_"):]
    metric_c = varied[2][len("weight_"):]

    winners = df[df["is_consensus_winner"]].copy()
    if winners.empty:
        raise SystemExit("CSV has no consensus-winner rows.")

    n_seeds = int(df["n_seeds"].max()) if "n_seeds" in df.columns else 1
    actors = sorted(winners["Actor/Criterion"].unique())
    color_map = _actor_color_map(actors)

    with _style():
        fig, ax = plt.subplots(figsize=(9.5, 8.5))
        ax.grid(False)

        # Triangle outline
        A = (0.5, np.sqrt(3) / 2)
        B = (0.0, 0.0)
        C = (1.0, 0.0)
        triangle = plt.Polygon([B, C, A], fill=False, edgecolor="#222", linewidth=1.5, zorder=2)
        ax.add_patch(triangle)

        # Light gridlines at every 0.2 (constant wA, wB, wC)
        for level in (0.2, 0.4, 0.6, 0.8):
            # constant wA (horizontal in our convention)
            xa1, ya1 = _barycentric_to_cartesian(level, 1 - level, 0)
            xa2, ya2 = _barycentric_to_cartesian(level, 0, 1 - level)
            # constant wB (parallel to AC)
            xb1, yb1 = _barycentric_to_cartesian(1 - level, level, 0)
            xb2, yb2 = _barycentric_to_cartesian(0, level, 1 - level)
            # constant wC (parallel to AB)
            xc1, yc1 = _barycentric_to_cartesian(1 - level, 0, level)
            xc2, yc2 = _barycentric_to_cartesian(0, 1 - level, level)
            for (x1, y1, x2, y2) in [(xa1, ya1, xa2, ya2),
                                     (xb1, yb1, xb2, yb2),
                                     (xc1, yc1, xc2, yc2)]:
                ax.plot([x1, x2], [y1, y2], color="#ddd", linewidth=0.6, zorder=1)

        # Scatter the grid points coloured by winner
        for _, row in winners.iterrows():
            wa = float(row[varied[0]])
            wb = float(row[varied[1]])
            wc = float(row[varied[2]])
            x, y = _barycentric_to_cartesian(wa, wb, wc)
            actor = row["Actor/Criterion"]
            color = color_map[actor]
            stab = row.get("consensus_winner_stability", 1.0)
            stab_val = float(stab) if pd.notna(stab) else 1.0
            alpha = 0.40 + 0.55 * stab_val  # range [0.40, 0.95]
            ax.scatter(x, y, c=[color], s=180, alpha=alpha,
                       edgecolors="black", linewidths=0.6, zorder=3)

        # Vertex labels (metric names)
        ax.annotate(metric_a.replace("_", " "), A,
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=13, fontweight="bold")
        ax.annotate(metric_b.replace("_", " "), B,
                    xytext=(-12, -10), textcoords="offset points",
                    ha="right", va="top", fontsize=13, fontweight="bold")
        ax.annotate(metric_c.replace("_", " "), C,
                    xytext=(12, -10), textcoords="offset points",
                    ha="left", va="top", fontsize=13, fontweight="bold")

        # Tick annotations along each edge (just numeric "weight on vertex" cues)
        for level in (0.2, 0.4, 0.6, 0.8):
            xa1, ya1 = _barycentric_to_cartesian(level, 1 - level, 0)
            ax.text(xa1 - 0.025, ya1, f"{level:.1f}", ha="right", va="center",
                    fontsize=8, color="#777")

        # Consensus-winner legend (colours)
        for actor in actors:
            ax.scatter([], [], c=[color_map[actor]], s=180, alpha=0.9,
                       edgecolors="black", linewidths=0.6, label=actor)
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            title="Consensus winner", title_fontsize=10,
            frameon=True, edgecolor="lightgray", fontsize=9,
        )

        # ---- Stability scale (only meaningful for multi-seed runs) ----
        if n_seeds > 1:
            # Inset axis to the right of the triangle.
            cax = fig.add_axes([0.91, 0.18, 0.018, 0.30])
            # Build a vertical gradient of greys at the exact alphas used by the
            # markers: alpha = 0.40 + 0.55 * stability.
            grad_n = 100
            stab_vals = np.linspace(0.0, 1.0, grad_n)
            alpha_vals = 0.40 + 0.55 * stab_vals
            # Render as a single column of rectangles so each row has its own alpha.
            # Use a neutral mid-grey base colour so the alpha is what's visible.
            base_rgb = (0.30, 0.30, 0.32)
            rgba_strip = np.tile(
                np.array(base_rgb + (1.0,)),
                (grad_n, 1, 1),
            ).reshape(grad_n, 1, 4)
            rgba_strip[:, 0, 3] = alpha_vals
            cax.imshow(rgba_strip[::-1], aspect="auto", extent=[0, 1, 0, 1])
            cax.set_xticks([])
            cax.set_yticks([0.0, 0.5, 1.0])
            cax.set_yticklabels(["0.0", "0.5", "1.0"], fontsize=8)
            cax.set_ylabel("inter-seed stability", fontsize=9, labelpad=2)
            cax.yaxis.set_label_position("right")
            cax.yaxis.tick_right()
            for s in ("top", "bottom", "left", "right"):
                cax.spines[s].set_color("#888")
                cax.spines[s].set_linewidth(0.6)

        # Plot housekeeping
        ax.set_xlim(-0.18, 1.18)
        ax.set_ylim(-0.10, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")

        # Title + subtitle
        ax.set_title(title, fontsize=14, pad=12)
        if n_seeds > 1:
            seed_note = f"Marker opacity scales with inter-seed stability   (n = {n_seeds} seeds)"
        else:
            seed_note = "Single-seed run; opacity uniform"
        ax.text(0.5, -0.05, seed_note, ha="center", va="top",
                fontsize=9, style="italic", color="#555",
                transform=ax.transAxes)

        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved {output}")


# ======================================================================
# Dirichlet sweep -- winner shares (with stability) + score violin
# ======================================================================
def plot_population(df: pd.DataFrame, output: Path, title: str):
    winners_df = df[df["is_consensus_winner"]]
    counts = winners_df["Actor/Criterion"].value_counts()
    total = int(counts.sum())
    pct = (counts / max(total, 1) * 100).round(1)
    mean_stab = (
        winners_df.groupby("Actor/Criterion")["consensus_winner_stability"].mean()
    )

    n_seeds = int(df["n_seeds"].max()) if "n_seeds" in df.columns else 1

    # Violin axis ordering: highest median at the top (ascending sort + bottom-up axis).
    actors_violin = sorted(
        df["Actor/Criterion"].unique(),
        key=lambda a: df[df["Actor/Criterion"] == a]["weighted_sum_mean"].median(),
    )
    color_map = _actor_color_map(actors_violin)

    with _style():
        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.45 * len(actors_violin) + 2)))

        _render_population_row(
            axes[0], axes[1], df, color_map, actors_violin,
            counts, total, mean_stab, n_seeds,
            show_titles=True, show_ylabels=True,
        )

        fig.suptitle(title, fontsize=14.5, y=1.02, fontweight="bold")
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved {output}")


def _render_population_row(
    ax_left, ax_right,
    df: pd.DataFrame,
    color_map: dict,
    actors_violin: list[str],
    counts: pd.Series,
    total: int,
    mean_stab: pd.Series,
    n_seeds: int,
    show_titles: bool = True,
    show_ylabels: bool = True,
    row_label: str | None = None,
):
    """Render the two-panel Dirichlet population view into a pair of axes.

    Extracted from plot_population so a driver like plot_stacked_dirichlet.py
    can stack the same content vertically for multiple buckets while sharing
    a canonical actor colour map.
    """
    # --- LEFT: winner shares + stability annotation (all evaluated actors) ---
    ax = ax_left
    ax.grid(False)
    all_actors = sorted(df["Actor/Criterion"].unique())
    counts_full = pd.Series({a: int(counts.get(a, 0)) for a in all_actors})
    counts_full = counts_full.sort_values(ascending=True)
    actors_bar = counts_full.index.tolist()
    values = counts_full.values
    pcts = [v / max(total, 1) * 100 for v in values]
    stabs = [mean_stab.get(a, float("nan")) for a in actors_bar]
    bar_colors = [color_map.get(a, "lightgray") for a in actors_bar]
    bars = ax.barh(actors_bar, values, color=bar_colors,
                   edgecolor="white", linewidth=1.5, alpha=0.92)
    max_v = max(values) if len(values) and max(values) > 0 else 1
    for rect, v, p, stab in zip(bars, values, pcts, stabs):
        if v == 0:
            continue
        stab_str = f"   stab={stab:.2f}" if not np.isnan(stab) else ""
        ax.text(
            v + max_v * 0.012, rect.get_y() + rect.get_height() / 2,
            f"{int(v)}  ({p:.1f}%){stab_str}",
            va="center", ha="left", fontsize=10, color="#222",
        )
    ax.set_xlim(0, max_v * 1.30)
    ax.set_xlabel(f"Winning configurations  (of {total})", fontsize=11.5)
    if show_titles:
        seed_subtitle = (
            f"(stab = mean across-seed agreement on the winner, {n_seeds} seeds)"
            if n_seeds > 1 else ""
        )
        ax.set_title(f"How often does each decision function win?\n{seed_subtitle}", fontsize=13)
    if row_label is not None:
        ax.set_ylabel(row_label, fontsize=13, fontweight="bold", labelpad=10)
    if not show_ylabels:
        ax.set_yticklabels([])
    ax.grid(alpha=0.25, axis="x", linestyle="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- RIGHT: score violin (mean across seeds per config) ---
    ax = ax_right
    data = [df[df["Actor/Criterion"] == a]["weighted_sum_mean"].values for a in actors_violin]
    positions = np.arange(1, len(actors_violin) + 1)
    parts = ax.violinplot(
        data, positions=positions, vert=False,
        showmeans=False, showmedians=True, showextrema=False, widths=0.85,
    )
    for body, actor in zip(parts["bodies"], actors_violin):
        body.set_facecolor(color_map[actor])
        body.set_edgecolor("#333")
        body.set_alpha(0.55)
        body.set_linewidth(0.8)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#7d1b1b")
        parts["cmedians"].set_linewidth(2.2)
    means = [np.mean(d) for d in data]
    ax.scatter(means, positions, color="black", marker="D", s=22, zorder=5, label="mean")

    ax.set_yticks(positions)
    if show_ylabels:
        ax.set_yticklabels(actors_violin)
    else:
        ax.set_yticklabels([])
    seed_subtitle_violin = f"mean over {n_seeds} seeds" if n_seeds > 1 else "single seed"
    ax.set_xlabel(f"Weighted Normalized-Sum  ({seed_subtitle_violin})", fontsize=11.5)
    if show_titles:
        ax.set_title("Score distribution across the sweep\n(ordered by median, red = median)",
                     fontsize=12.5)
    ax.grid(alpha=0.25, axis="x", linestyle="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="A long-format CSV from ablate_weights.py.")
    parser.add_argument("--output", type=Path, default=Path("figs/ablation.png"))
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {"weighted_sum_mean", "is_consensus_winner"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"CSV is missing columns {missing}. Did you regenerate with the multi-seed "
            f"ablate_weights.py? Old single-column CSVs are no longer supported."
        )

    kind = _detect_sweep_kind(df)
    label_map = {
        "pairwise":  "pairwise weight trade-off",
        "ternary":   "ternary (3-metric) weight trade-off",
        "dirichlet": "Dirichlet weight sweep",
    }
    label = label_map.get(kind, kind)
    title = args.title or f"{args.csv.stem.replace('_', ' ')}  --  {label}"

    if kind == "pairwise":
        plot_pairwise(df, args.output, title)
    elif kind == "ternary":
        plot_ternary(df, args.output, title)
    else:
        plot_population(df, args.output, title)


if __name__ == "__main__":
    main()
