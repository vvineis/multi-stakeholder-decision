"""
DecisionProcessor: build per-row expected rewards from the trained outcome + reward models,
then ask every solution strategy for a recommended action.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.decisions.strategies import SuggestAction, build_strategies


class DecisionProcessor:
    def __init__(self, outcome_model, reward_models, onehot_encoder, cfg):
        self.outcome_model = outcome_model
        self.reward_models = reward_models
        self.onehot_encoder = onehot_encoder
        self.cfg = cfg
        self.model_type = cfg.models.outcome.model_type
        self.feature_columns = list(cfg.context.feature_columns)
        self.categorical_columns = list(cfg.categorical_columns)
        self.actions_set = list(cfg.actions_outcomes.actions_set)
        # Strategies aware of the use case's action space (fixes Nash Bargaining default).
        self.strategies = build_strategies(cfg)

    # ------------------------------------------------------------------
    def encode_features(self, feature_context: pd.DataFrame) -> pd.DataFrame:
        available_cats = [c for c in self.categorical_columns if c in feature_context.columns]
        numerical = feature_context[self.feature_columns]

        if not available_cats:
            return pd.DataFrame(numerical.values, columns=list(numerical.columns))

        cat_encoded = self.onehot_encoder.transform(feature_context[available_cats])
        col_names = list(numerical.columns) + list(
            self.onehot_encoder.get_feature_names_out(available_cats)
        )
        combined = np.concatenate([numerical.values, cat_encoded], axis=1)
        return pd.DataFrame(combined, columns=col_names)

    # ------------------------------------------------------------------
    def compute_expected_reward(self, feature_context: pd.DataFrame):
        feature_context = feature_context[self.feature_columns]
        expected_rewards = {actor: {} for actor in self.reward_models}
        predictions_list = []

        if self.model_type == "classification":
            outcome_probs = self.outcome_model.classifier.predict_proba(feature_context)
            outcome_classes = self.outcome_model.classifier.classes_
            predictions_list = [outcome_classes[np.argmax(p)] for p in outcome_probs]

            for action in self.actions_set:
                for idx, outcome in enumerate(outcome_classes):
                    ctx = feature_context.copy()
                    ctx["Action"] = action
                    ctx["Outcome"] = outcome
                    ctx_enc = self.encode_features(ctx)
                    for actor, model in self.reward_models.items():
                        reward = model.predict(ctx_enc)[0]
                        prob = outcome_probs[0][idx]
                        expected_rewards[actor].setdefault(action, 0)
                        expected_rewards[actor][action] += prob * reward

        elif self.model_type == "causal_regression":
            pred_A, pred_C = self.outcome_model.predict_outcomes(feature_context)
            for idx in range(len(feature_context)):
                predictions_list.append({"A": [pred_A[idx]], "C": [pred_C[idx]]})

            for action in self.actions_set:
                predicted = pred_C if action == self.outcome_model.control_name else pred_A
                for actor, model in self.reward_models.items():
                    ctx = feature_context.copy()
                    ctx["Action"] = action
                    ctx["Outcome"] = predicted.flatten().astype(str)
                    ctx_enc = self.encode_features(ctx)
                    expected_rewards[actor][action] = model.predict(ctx_enc)[0]

        return expected_rewards, predictions_list

    # ------------------------------------------------------------------
    def get_decisions(self, X_val_or_test_reward: pd.DataFrame):
        all_expected_rewards, all_decision_solutions, all_clfr_preds = [], [], []
        for _, row in X_val_or_test_reward.iterrows():
            ctx_df = pd.DataFrame([row])
            expected, preds = self.compute_expected_reward(ctx_df)
            decisions = SuggestAction(expected, strategies=self.strategies).compute_all_compromise_solutions()
            all_expected_rewards.append(expected)
            all_decision_solutions.append(decisions)
            if preds is not None:
                all_clfr_preds.extend(preds)

        return (
            all_expected_rewards,
            all_decision_solutions,
            all_clfr_preds,
            self._decisions_to_df(all_decision_solutions),
        )

    @staticmethod
    def _decisions_to_df(all_decision_solutions):
        rows = [
            {"Row Index": i, "Decision Type": dtype, "Best Action": sol["action"], "Value": sol["value"]}
            for i, decisions in enumerate(all_decision_solutions)
            for dtype, sol in decisions.items()
        ]
        return pd.DataFrame(rows)
