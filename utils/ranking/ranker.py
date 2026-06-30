"""
Ranker: turn a metrics DataFrame into a ranked DataFrame using configurable weights.

This was previously embedded inside `SummaryProcessor._add_ranking_and_weighted_sum_...`.
Extracting it has two benefits:

1. The ranking step is now **pure** w.r.t. the raw metrics — you can re-rank an
   existing `decision_metrics_df` under different weights without retraining the
   models. `ablate_weights.py` relies on this.
2. `SummaryProcessor` becomes a thin orchestrator over (metrics → ranking).

The math is exactly the same as the original framework, so for a given
`(metrics_df, ranking_criteria, weights, metrics_for_evaluation)` the result is identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import pandas as pd

# Small epsilon to avoid division-by-zero in normalization
_EPS = 1e-9


@dataclass(frozen=True)
class RankingConfig:
    """Plain container for the four pieces of state Ranker needs."""

    ranking_criteria: Mapping[str, str]       # metric -> {'min','max','zero'}
    ranking_weights: Mapping[str, float]      # metric -> weight in [0, 1]
    metrics_for_evaluation: Sequence[str]
    actor_criterion_col: str = "Actor/Criterion"


class Ranker:
    """Apply normalization, ranking, and a weighted-sum aggregation to a metrics df."""

    def __init__(self, config: RankingConfig):
        if not isinstance(config.ranking_weights, Mapping):
            raise TypeError("ranking_weights must be a mapping")
        self.cfg = config

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_hydra(cls, cfg) -> "Ranker":
        """Build a Ranker straight from the project's Hydra config."""
        return cls(
            RankingConfig(
                ranking_criteria=dict(cfg.criteria.ranking_criteria),
                ranking_weights=dict(cfg.ranking_weights),
                metrics_for_evaluation=list(cfg.criteria.metrics_for_evaluation),
            )
        )

    def with_weights(self, new_weights: Mapping[str, float]) -> "Ranker":
        """Return a sibling Ranker that reuses the same criteria/metrics but new weights."""
        return Ranker(
            RankingConfig(
                ranking_criteria=self.cfg.ranking_criteria,
                ranking_weights=dict(new_weights),
                metrics_for_evaluation=self.cfg.metrics_for_evaluation,
                actor_criterion_col=self.cfg.actor_criterion_col,
            )
        )

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------
    def rank(self, metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], str]:
        """
        Normalize each metric, compute a per-row rank, and produce the weighted sum.

        Returns
        -------
        ranked_df : DataFrame sorted by Weighted Normalized-Sum (descending)
        score_dict : {actor_or_criterion: weighted_sum}
        best : actor_or_criterion with the highest weighted sum
        """
        weights = self._normalized_weights()

        df = metrics_df.copy()
        for metric in self.cfg.metrics_for_evaluation:
            df[f"{metric} Normalized"] = self._normalize_column(df, metric)
            df[f"{metric} Rank"] = df[f"{metric} Normalized"].rank(
                ascending=False, method="min"
            )

        df["Weighted Normalized-Sum"] = sum(
            df[f"{metric} Normalized"] * weights.get(metric, 0.0)
            for metric in self.cfg.metrics_for_evaluation
        )

        score_dict = df.set_index(self.cfg.actor_criterion_col)[
            "Weighted Normalized-Sum"
        ].to_dict()
        best = max(score_dict, key=score_dict.get)
        ranked = df.sort_values(by="Weighted Normalized-Sum", ascending=False).reset_index(
            drop=True
        )
        return ranked, score_dict, best

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _normalized_weights(self) -> Dict[str, float]:
        total = sum(
            self.cfg.ranking_weights.get(m, 0)
            for m in self.cfg.metrics_for_evaluation
        )
        if total == 0:
            raise ValueError("Total weight cannot be zero.")
        return {
            m: self.cfg.ranking_weights.get(m, 0) / total
            for m in self.cfg.metrics_for_evaluation
        }

    def _normalize_column(self, df: pd.DataFrame, metric: str) -> pd.Series:
        direction = self.cfg.ranking_criteria[metric]
        col = df[metric]

        # Accuracy is special-cased in the original code: it is already in [0,1] and not rescaled.
        if metric == "Accuracy":
            return col.copy()

        if direction == "max":
            lo, hi = col.min(), col.max()
            return (col - lo) / max(hi - lo, _EPS)
        if direction == "min":
            lo, hi = col.min(), col.max()
            return (hi - col) / max(hi - lo, _EPS)
        if direction == "zero":
            max_abs = col.abs().max()
            return 1 - (col.abs() / max(max_abs, _EPS))

        raise ValueError(f"Unknown ranking direction '{direction}' for metric '{metric}'.")


# ------------------------------------------------------------------
# Convenience: stand-alone re-ranking without instantiating a Hydra cfg.
# ------------------------------------------------------------------
def rerank(
    metrics_df: pd.DataFrame,
    *,
    ranking_criteria: Mapping[str, str],
    ranking_weights: Mapping[str, float],
    metrics_for_evaluation: Iterable[str],
    actor_criterion_col: str = "Actor/Criterion",
) -> Tuple[pd.DataFrame, Dict[str, float], str]:
    """Functional alias for `Ranker(RankingConfig(...)).rank(metrics_df)`."""
    ranker = Ranker(
        RankingConfig(
            ranking_criteria=dict(ranking_criteria),
            ranking_weights=dict(ranking_weights),
            metrics_for_evaluation=list(metrics_for_evaluation),
            actor_criterion_col=actor_criterion_col,
        )
    )
    return ranker.rank(metrics_df)
