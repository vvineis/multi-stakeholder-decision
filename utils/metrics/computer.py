"""
MetricsCalculator: aggregates fairness, standard, case-specific, and payoff metrics
per actor / per decision criterion.

The branching on `model_type` is now centralized in the helpers below, and the
case-specific metrics class is looked up via a registry (`CASE_METRICS_REGISTRY`)
rather than hard-coded `if` chains.
"""
from __future__ import annotations

import pandas as pd

from utils.metrics.case_specific_metrics import CASE_METRICS_REGISTRY
from utils.metrics.fairness_metrics import FairnessMetrics
from utils.metrics.real_payoffs import RealPayoffMetrics
from utils.metrics.standard_metrics import StandardMetrics
from utils.rewards.get_rewards import RewardCalculator


class MetricsCalculator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_type = cfg.models.outcome.model_type
        self.case_metrics_cls = CASE_METRICS_REGISTRY[self.model_type]

    # ------------------------------------------------------------------
    def merge_predicted(self, df, decision_col):
        def _select(row):
            return row["A_outcome_binary"] if row[decision_col] == "A" else row["C_outcome_binary"]

        df["Predicted_Binary"] = df.apply(_select, axis=1)
        return df

    # ------------------------------------------------------------------
    def _make_fairness(self, df, decision_col):
        if self.model_type == "classification":
            return FairnessMetrics(cfg=self.cfg, suggestions_df=df, decision_col=decision_col)
        return FairnessMetrics(
            cfg=self.cfg, suggestions_df=df, decision_col=decision_col, outcome_col="Predicted_Binary"
        )

    def _make_standard(self, df, decision_col, true_outcome_col):
        if self.model_type == "classification":
            return StandardMetrics(
                df,
                decision_col,
                true_outcome_col,
                self.cfg.actions_outcomes.actions_set,
                self.cfg.actions_outcomes.outcomes_set,
                model_type="classification",
            )
        return StandardMetrics(
            df,
            decision_col,
            true_outcome_col,
            causal_reg_outcome_cols=[f"{a}_outcome" for a in self.cfg.actions_outcomes.actions_set],
            model_type="causal_regression",
        )

    # ------------------------------------------------------------------
    def _binarize_outcomes(self, df):
        threshold = self.cfg.case_specific_metrics.threshold_outcome
        for action in self.cfg.actions_outcomes.actions_set:
            df[f"{action}_outcome_binary"] = (df[f"{action}_outcome"] <= threshold).astype(int)
        return df

    # ------------------------------------------------------------------
    def compute_all_metrics(self, suggestions_df: pd.DataFrame, true_outcome_col: str = "True Outcome"):
        actor_list = list(self.cfg.actors.actor_list) + list(self.cfg.decision_criteria)
        metrics = {actor: {} for actor in actor_list}

        fairness_cache: dict = {}
        action_counts_cache: dict = {}

        for actor in actor_list:
            decision_col = f"{actor} Suggested Action" if actor in self.cfg.actors.actor_list else actor
            if decision_col not in suggestions_df.columns:
                continue

            # Causal-regression: derive binarized outcome columns (idempotent across actors)
            if self.model_type == "causal_regression":
                suggestions_df = self._binarize_outcomes(suggestions_df)
                suggestions_df = self.merge_predicted(suggestions_df, decision_col)

            # ---- Fairness ----
            if decision_col not in fairness_cache:
                fairness_cache[decision_col] = self._make_fairness(suggestions_df, decision_col).get_metrics(
                    self.cfg.fairness_metrics
                )

            # ---- Action counts ----
            if decision_col not in action_counts_cache:
                action_counts_cache[decision_col] = (
                    suggestions_df[decision_col].value_counts(normalize=True).to_dict()
                )

            # ---- Case-specific metrics ----
            case_metrics = self.case_metrics_cls(
                suggestions_df, decision_col, true_outcome_col, self.cfg
            )
            for metric in self.cfg.case_specific_metrics.metrics:
                try:
                    metrics[actor][metric] = case_metrics.get_metrics([metric])[metric]
                except ValueError as e:
                    print(f"Error computing metric '{metric}': {e}")

            # ---- Standard metrics ----
            standard = self._make_standard(suggestions_df, decision_col, true_outcome_col).get_metrics(
                self.cfg.standard_metrics
            )
            for metric in self.cfg.standard_metrics:
                metrics[actor][metric] = standard.get(metric)

            # ---- Fairness from cache ----
            metrics[actor].update(fairness_cache[decision_col])

            # ---- Per-action percentages ----
            counts = action_counts_cache[decision_col]
            metrics[actor].update(
                {f"Percent_{a}": counts.get(a, 0) for a in self.cfg.actions_outcomes.actions_set}
            )

            # ---- Real payoffs (classification only) ----
            if self.model_type == "classification":
                # Pick the same reward variant the training pipeline used,
                # so RealPayoff is consistent with how the rewards were generated.
                variant = self.cfg.reward_calculator.get("reward_variant", "base")
                reward_structures = RewardCalculator.get_structures_for_variant(variant)
                for reward_actor in self.cfg.actors.reward_types:
                    payoff = RealPayoffMetrics(
                        cfg=self.cfg,
                        suggestions_df=suggestions_df,
                        decision_col=decision_col,
                        true_outcome_col=true_outcome_col,
                        reward_actor=reward_actor,
                        reward_structures=reward_structures,
                    ).compute_total_real_payoff()
                    metrics[actor][f"Total Real Payoff ({reward_actor})"] = payoff

        return metrics
