"""
Weight ablation (paper edition, multi-seed aware).

Three sweep types are supported, each with a clear paper purpose:

* `pairwise`  -- vary the weight on metric A from 0 to 1 with metric B = (1 - A)
                 and all other weights = 0. Produces the canonical fair-ML
                 trade-off curve (e.g. Accuracy vs. Demographic Parity).
* `ternary`   -- triangular grid over the (A, B, C) 2-simplex with all other
                 weights = 0. Renders as a ternary plot showing the geometry
                 of winner regions in a three-objective trade-off.
* `dirichlet` -- Monte-Carlo over the simplex of all evaluation-metric weights.
                 Quantifies the *robustness* of each compromise rule: how often
                 it wins under a random sample of normative priors.

Multi-seed aggregation
----------------------
Use `--metrics-glob` (preferred) to point at the per-seed CSVs for one
configuration; the script:

  1. Re-ranks each seed under each weight configuration.
  2. Averages the weighted-normalized-sum across seeds, with SEM.
  3. Determines the **consensus winner** = the actor with the highest
     mean weighted-sum at that config.
  4. Reports **consensus stability** = fraction of seeds where the
     per-seed winner agrees with the consensus winner.

Pass `--metrics-csv` instead for a single CSV (legacy mode, `n_seeds = 1`,
SEM = 0, stability is degenerate).

Output: a single long-format CSV with one row per (config, actor):

    config_id, weight_<m>..., Actor/Criterion,
    weighted_sum_mean, weighted_sum_sem, n_seeds,
    rank_within_config, is_consensus_winner, consensus_winner_stability

Reference actors that are not compromise rules (`Oracle`, `Random`,
`Outcome_Maxim`, `Nash Social Welfare`) are excluded from the winner search
by default. `Outcome_Pred_Model` is *included* by default so it appears as a
prediction-baseline reference in the ablation plots (usually with 0% winner
share, showing the compromise rules dominate it). Override with
`--include-all` or a custom `--exclude` list.
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from utils.ranking.ranker import rerank


DEFAULT_EXCLUDE = ("Oracle", "Random", "Outcome_Maxim", "Nash Social Welfare")


# ----------------------------------------------------------------------
# Use-case YAML lookup
# ----------------------------------------------------------------------
def _ranking_spec(use_case: str) -> tuple[dict, list[str]]:
    path = Path(__file__).parent / "conf" / "use_case" / f"{use_case}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    criteria = cfg["criteria"]
    return dict(criteria["ranking_criteria"]), list(criteria["metrics_for_evaluation"])


def _strip_cached_columns(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Remove the cached normalization / ranking columns from a previous run."""
    drop = [f"{m} Normalized" for m in metrics] + [f"{m} Rank" for m in metrics]
    drop.append("Weighted Normalized-Sum")
    return df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")


# ----------------------------------------------------------------------
# Sweep generators
# ----------------------------------------------------------------------
def sweep_pairwise(metrics: list[str], metric_a: str, metric_b: str, n_steps: int = 21):
    return [
        {**{m: 0.0 for m in metrics}, metric_a: float(t), metric_b: float(1 - t)}
        for t in np.linspace(0.0, 1.0, n_steps)
    ]


def sweep_ternary(
    metrics: list[str], metric_a: str, metric_b: str, metric_c: str, n_grid: int = 10,
):
    """Triangular grid over the 3-metric simplex: (n_grid+1)*(n_grid+2)/2 points.

    Defaults give 66 grid points (n_grid=10). All other metrics get weight 0.
    """
    configs = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            k = n_grid - i - j
            if k < 0:
                continue
            cfg = {m: 0.0 for m in metrics}
            cfg[metric_a] = i / n_grid
            cfg[metric_b] = j / n_grid
            cfg[metric_c] = k / n_grid
            configs.append(cfg)
    return configs


def sweep_dirichlet(metrics: list[str], n_samples: int = 500, alpha: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_samples):
        s = rng.dirichlet([alpha] * len(metrics))
        out.append({m: float(x) for m, x in zip(metrics, s)})
    return out


# ----------------------------------------------------------------------
# Core: per-seed rerank + aggregation
# ----------------------------------------------------------------------
def _per_seed_scores(
    per_seed_dfs: list[pd.DataFrame],
    weights: dict,
    ranking_criteria: dict,
    metrics: list[str],
    exclude: set[str],
) -> list[dict[str, float]]:
    """Return one dict {actor: weighted_sum} per seed for this weight config."""
    out = []
    for df in per_seed_dfs:
        _, scores, _ = rerank(
            df,
            ranking_criteria=ranking_criteria,
            ranking_weights=weights,
            metrics_for_evaluation=metrics,
        )
        # Drop excluded actors from the per-seed score dict.
        out.append({a: s for a, s in scores.items() if a not in exclude})
    return out


def ablate_to_long(
    per_seed_dfs: list[pd.DataFrame],
    weight_configs: list[dict],
    ranking_criteria: dict,
    metrics: list[str],
    exclude: set[str],
) -> pd.DataFrame:
    """For each config and each non-excluded actor, return one row with the
    mean/SEM weighted-normalized-sum across seeds, the per-config rank,
    and a `consensus_winner_stability` indicator on the winner row."""
    per_seed_dfs = [_strip_cached_columns(df, metrics) for df in per_seed_dfs]
    n_seeds = len(per_seed_dfs)

    rows = []
    for cfg_id, weights in enumerate(weight_configs):
        per_seed = _per_seed_scores(per_seed_dfs, weights, ranking_criteria, metrics, exclude)

        # Union of actors across seeds (should be the same set in practice).
        all_actors = set().union(*per_seed)
        if not all_actors:
            continue

        # Aggregate to mean / SEM / per-seed winner.
        actor_stats: dict[str, dict] = {}
        for actor in all_actors:
            vals = np.array([s.get(actor, np.nan) for s in per_seed], dtype=float)
            vals_clean = vals[~np.isnan(vals)]
            if len(vals_clean) == 0:
                continue
            mean = float(vals_clean.mean())
            sem = (float(vals_clean.std(ddof=1)) / math.sqrt(len(vals_clean))
                   if len(vals_clean) > 1 else 0.0)
            actor_stats[actor] = {"mean": mean, "sem": sem, "n": int(len(vals_clean))}

        # Consensus winner (highest mean) and per-seed winners.
        sorted_actors = sorted(actor_stats.items(), key=lambda kv: -kv[1]["mean"])
        consensus_winner = sorted_actors[0][0] if sorted_actors else None
        per_seed_winners = [
            (max(s.items(), key=lambda kv: kv[1])[0] if s else None) for s in per_seed
        ]
        agreed = sum(1 for w in per_seed_winners if w == consensus_winner)
        stability = agreed / max(n_seeds, 1)

        for rank, (actor, stats) in enumerate(sorted_actors, start=1):
            entry = {
                "config_id": cfg_id,
                "Actor/Criterion": actor,
                "weighted_sum_mean": stats["mean"],
                "weighted_sum_sem": stats["sem"],
                "n_seeds": stats["n"],
                "rank_within_config": rank,
                "is_consensus_winner": (rank == 1),
                # Only meaningful on the winning row; NaN elsewhere keeps the column tidy.
                "consensus_winner_stability": stability if rank == 1 else float("nan"),
            }
            for m, w in weights.items():
                entry[f"weight_{m}"] = round(float(w), 6)
            rows.append(entry)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _discover_csvs(args) -> list[str]:
    if args.metrics_glob and args.metrics_csv:
        raise SystemExit("Provide either --metrics-glob OR --metrics-csv, not both.")
    if args.metrics_glob:
        matched = sorted(glob.glob(args.metrics_glob))
        if not matched:
            raise SystemExit(f"No CSVs matched '{args.metrics_glob}'.")
        return matched
    if args.metrics_csv:
        return [str(args.metrics_csv)]
    raise SystemExit("Provide --metrics-glob (multi-seed) or --metrics-csv (single seed).")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--use-case", required=True, choices=("lending", "health"))

    src = parser.add_argument_group("Input (one of)")
    src.add_argument("--metrics-glob",
                     help="Glob pattern across per-seed CSVs at one configuration "
                          "(e.g. 'results/lending/run_*seed*_Acc_0.4_Fair_0.2/"
                          "final_ranked_decision_metrics.csv').")
    src.add_argument("--metrics-csv", type=Path,
                     help="Single CSV (legacy single-seed mode).")

    parser.add_argument("--output", required=True, type=Path,
                        help="Where to write the long-format ablation CSV.")
    parser.add_argument("--sweep", required=True, choices=("pairwise", "ternary", "dirichlet"))

    parser.add_argument("--metric-a", help="(pairwise/ternary) First metric.")
    parser.add_argument("--metric-b", help="(pairwise/ternary) Second metric.")
    parser.add_argument("--metric-c", help="(ternary) Third metric.")
    parser.add_argument("--n-steps", type=int, default=21,
                        help="(pairwise) Number of grid points from 0 to 1, inclusive.")
    parser.add_argument("--n-grid", type=int, default=10,
                        help="(ternary) Subdivisions per simplex edge (n_grid=10 -> 66 points).")

    parser.add_argument("--n-samples", type=int, default=500,
                        help="(dirichlet) Number of Monte-Carlo weight samples.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="(dirichlet) Concentration parameter (1.0 = uniform on the simplex).")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                        help="Actors to exclude from the winner search. Default: "
                             f"{list(DEFAULT_EXCLUDE)}.")
    parser.add_argument("--include-all", action="store_true",
                        help="Override --exclude and consider every actor.")

    args = parser.parse_args()

    csvs = _discover_csvs(args)
    print(f"Reading {len(csvs)} seed run{'s' if len(csvs) != 1 else ''}:")
    for c in csvs:
        print(f"  - {c}")
    per_seed_dfs = [pd.read_csv(c) for c in csvs]

    ranking_criteria, metrics = _ranking_spec(args.use_case)

    if args.sweep == "pairwise":
        if not (args.metric_a and args.metric_b):
            raise SystemExit("--metric-a and --metric-b are required for sweep=pairwise.")
        weight_configs = sweep_pairwise(metrics, args.metric_a, args.metric_b, args.n_steps)
    elif args.sweep == "ternary":
        if not (args.metric_a and args.metric_b and args.metric_c):
            raise SystemExit("--metric-a, --metric-b and --metric-c are required for sweep=ternary.")
        weight_configs = sweep_ternary(
            metrics, args.metric_a, args.metric_b, args.metric_c, args.n_grid,
        )
    else:
        weight_configs = sweep_dirichlet(metrics, args.n_samples, args.alpha, args.seed)

    print(f"Generated {len(weight_configs)} weight configurations.")
    exclude = set() if args.include_all else set(args.exclude)
    if exclude:
        print(f"Excluding actors: {sorted(exclude)}")

    long_df = ablate_to_long(per_seed_dfs, weight_configs, ranking_criteria, metrics, exclude)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(long_df)} rows ({long_df['Actor/Criterion'].nunique()} actors x "
          f"{len(weight_configs)} configs) -> {args.output}")

    # Console summary.
    winners = long_df[long_df["is_consensus_winner"]]
    if not winners.empty:
        counts = winners["Actor/Criterion"].value_counts()
        mean_stability_per_actor = (
            winners.groupby("Actor/Criterion")["consensus_winner_stability"].mean()
        )
        print("\nConsensus winner distribution (mean stability across seeds):")
        for actor, count in counts.items():
            stab = mean_stability_per_actor.get(actor, float("nan"))
            print(f"  {actor:30s} {count:4d} configs  ({count / len(weight_configs) * 100:5.1f}%)"
                  f"   stability = {stab:.2f}")


if __name__ == "__main__":
    main()
