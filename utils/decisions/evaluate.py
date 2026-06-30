"""
SummaryProcessor: assemble the per-row summary frame, hand it to MetricsCalculator,
then delegate ranking to Ranker.

This module is intentionally slim — the ranking math now lives in
`utils.ranking.ranker`, so the same code path is used both during a normal
pipeline run and during the weight-ablation script.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from utils.ranking.ranker import Ranker


class SummaryProcessor:
    def __init__(self, cfg, metrics_calculator, strategy, seed: int | None = None):
        self.cfg = cfg
        self.metrics_calculator = metrics_calculator
        self.strategy = strategy
        self.ranker = Ranker.from_hydra(cfg)

        self.reward_types = list(cfg.actors.reward_types)
        self.decision_criteria_list = list(cfg.decision_criteria)
        self.actions_set = list(cfg.actions_outcomes.actions_set)
        self.outcomes_set = list(cfg.actions_outcomes.outcomes_set)
        self.model_type = cfg.models.outcome.model_type
        self.mapping = cfg.actions_outcomes.mapping if self.model_type == "classification" else None

        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    def create_summary_df(
        self,
        y_val_outcome,
        X_val_outcome,
        treatment_val,
        decisions_df,
        unscaled_X_val_reward,
        expected_rewards_list,
        pred_list,
    ):
        feature_context_df = unscaled_X_val_reward.copy()
        feature_context_df["True Outcome"] = y_val_outcome.values
        feature_context_df["Real Treatment"] = (
            treatment_val.values if treatment_val is not None else None
        )

        decision_pivot = decisions_df.pivot(
            index="Row Index", columns="Decision Type", values="Best Action"
        )
        summary_df = pd.concat(
            [feature_context_df.reset_index(drop=True), decision_pivot.reset_index(drop=True)],
            axis=1,
        )

        if self.model_type == "causal_regression":
            actions_in_preds = pred_list[0].keys()
            for action in actions_in_preds:
                col = f"{action}_predicted_outcome"
                summary_df[col] = [
                    float(p[action][0]) if action in p else np.nan for p in pred_list
                ]
            summary_df["A_outcome"] = summary_df.apply(
                lambda row: row["True Outcome"] if row["Real Treatment"] == "A" else row["A_predicted_outcome"],
                axis=1,
            )
            summary_df["C_outcome"] = summary_df.apply(
                lambda row: row["True Outcome"] if row["Real Treatment"] == "C" else row["C_predicted_outcome"],
                axis=1,
            )

        # Per-actor suggested action (using `strategy`, normally MaxIndividualReward)
        if self.model_type == "classification":
            suggested = {a: [] for a in self.reward_types + ["Oracle", "Outcome_Pred_Model", "Random"]}
        else:
            suggested = {a: [] for a in self.reward_types + ["Outcome_Maxim", "Random"]}

        for idx, (expected_rewards, predicted_outcomes) in enumerate(
            zip(expected_rewards_list, pred_list)
        ):
            individual_actions = self.strategy.compute(
                expected_rewards,
                disagreement_point=None,
                ideal_point=None,
                all_actions=self.actions_set,
            )
            for actor in self.reward_types:
                suggested[actor].append(individual_actions[actor]["action"])

            if self.model_type == "classification":
                suggested["Oracle"].append(self._map_outcome_to_action(y_val_outcome.iloc[idx]))
                suggested["Outcome_Pred_Model"].append(self._map_outcome_to_action(pred_list[idx]))
            else:
                exp_outcome = {a: v[0] for a, v in predicted_outcomes.items()}
                suggested["Outcome_Maxim"].append(self._argmax_action(exp_outcome))
            suggested["Random"].append(random.choice(list(self.actions_set)))

        for actor, actions in suggested.items():
            summary_df[f"{actor} Suggested Action"] = actions
        return summary_df

    # ------------------------------------------------------------------
    def _map_outcome_to_action(self, outcome):
        return self.mapping.get(outcome, self.mapping.get("default", "Grant_lower"))

    @staticmethod
    def _argmax_action(outcomes: dict) -> str:
        max_value = max(outcomes.values())
        best = [a for a, v in outcomes.items() if v == max_value]
        return "C" if "C" in best else best[0]

    # ------------------------------------------------------------------
    @staticmethod
    def metrics_to_dataframe(metrics):
        return pd.DataFrame([{"Actor/Criterion": k, **v} for k, v in metrics.items()])

    def rank(self, metrics_df: pd.DataFrame):
        """Delegate to Ranker — preserved as a method for backward compatibility."""
        return self.ranker.rank(metrics_df)

    # ------------------------------------------------------------------
    def process_decision_metrics(
        self,
        y_val_outcome,
        X_val_outcome,
        treatment_val,
        decisions_df,
        unscaled_X_val_reward,
        expected_rewards_list,
        pred_list,
    ):
        summary_df = self.create_summary_df(
            y_val_outcome,
            X_val_outcome,
            treatment_val,
            decisions_df,
            unscaled_X_val_reward,
            expected_rewards_list,
            pred_list,
        )
        metrics_df = self.metrics_to_dataframe(
            self.metrics_calculator.compute_all_metrics(summary_df, true_outcome_col="True Outcome")
        )
        ranked_df, rank_dict, best = self.ranker.rank(metrics_df)
        return {
            "summary_df": summary_df,
            "decision_metrics_df": metrics_df,
            "ranked_decision_metrics_df": ranked_df,
            "rank_dict": rank_dict,
            "best_criterion": best,
        }
