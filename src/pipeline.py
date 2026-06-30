"""
Unified pipeline: tune outcome + reward models, build decisions, score them.

Replaces the previous `cross_validation_process.py` and `final_evaluation.py`,
which carried a lot of duplicated logic for instantiating models, running the
decision processor, and post-processing metrics. The common steps live here;
`run_cv` and `run_final` are thin compositions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from utils.decisions.processor import DecisionProcessor
from utils.decisions.evaluate import SummaryProcessor
from utils.decisions.strategies import MaxIndividualReward
from utils.metrics.computer import MetricsCalculator
from utils.models.factory import ModelFactory


# ---------------------------------------------------------------------
# State held during CV
# ---------------------------------------------------------------------
@dataclass
class CVState:
    best_hparams_outcome: List[dict] = field(default_factory=list)
    best_hparams_reward: List[dict] = field(default_factory=list)
    best_outcome_models: List = field(default_factory=list)
    best_reward_models: List = field(default_factory=list)
    fold_scores_outcome: List[float] = field(default_factory=list)
    fold_scores_reward: List[float] = field(default_factory=list)
    fold_summaries: List[pd.DataFrame] = field(default_factory=list)
    fold_decision_metrics: List[pd.DataFrame] = field(default_factory=list)
    fold_ranked_decision_metrics: List[pd.DataFrame] = field(default_factory=list)
    fold_rank_dicts: List[dict] = field(default_factory=list)
    fold_best_criteria: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.factory = ModelFactory(cfg)
        self.metrics_calculator = MetricsCalculator(cfg)
        self.summary_processor = SummaryProcessor(
            cfg=cfg,
            metrics_calculator=self.metrics_calculator,
            strategy=MaxIndividualReward(),
        )

    # ==================================================================
    # Shared hyperparameter-tuning helpers
    # ==================================================================
    def _tune_outcome(self, X_train, t_train, mu, y_train, X_val, t_val, y_val):
        """Grid-search the outcome model and return (best_params, best_model, best_score)."""
        is_classification = self.factory.model_type == "classification"
        best_score = -np.inf if is_classification else np.inf
        best_params, best_model = None, None
        logging.getLogger("causalml").setLevel(logging.WARNING)

        for params in ParameterGrid(self.factory.outcome_param_grid):
            model = self.factory.new_outcome_model()
            print(f"Trying outcome params: {params}")

            if t_train is not None:
                model.train(X_train, t_train, y_train, **params)
                score = model.evaluate(X_val, t_val, mu)
            else:
                model.train(X_train, y_train, **params)
                score = model.evaluate(X_val, y_val)

            improved = (
                (is_classification and score > best_score)
                or (not is_classification and score < best_score)
            )
            if improved:
                best_score, best_params, best_model = score, params, model

        print(f"Best outcome model: {best_model}")
        return best_params, best_model, best_score

    def _tune_reward(self, X_train, y_train_rewards, X_val, y_val_rewards):
        """Grid-search reward models. Best = lowest average MSE across actors."""
        best_params, best_models, best_score = None, {}, np.inf

        for params in ParameterGrid(self.factory.reward_param_grid):
            print(f"Trying reward params: {params}")
            wrapper = self.factory.new_reward_models(**params)
            trained = wrapper.train(X_train, y_train_rewards)
            scores = wrapper.evaluate(X_val, y_val_rewards)
            avg_mse = wrapper.average_mse(scores)
            if avg_mse < best_score:
                best_score, best_params, best_models = avg_mse, params, trained

        return best_params, best_models, best_score

    # ==================================================================
    # Decisions + metrics for a fold/test pass
    # ==================================================================
    def _score_split(self, outcome_model, reward_models, fold_dict, *, is_final: bool):
        """Run the decision processor + summary processor on one split."""
        X_val_out, t_val, mu, y_val_out = fold_dict["val_or_test_outcome"]
        X_val_rew, _ = fold_dict["val_or_test_reward"]

        decision_processor = DecisionProcessor(
            outcome_model=outcome_model,
            reward_models=reward_models,
            onehot_encoder=fold_dict["onehot_encoder"],
            cfg=self.cfg,
        )
        all_expected_rewards, _, all_predictions, decisions_df = decision_processor.get_decisions(X_val_rew)

        result = self.summary_processor.process_decision_metrics(
            y_val_outcome=y_val_out,
            X_val_outcome=X_val_out,
            treatment_val=t_val,
            decisions_df=decisions_df,
            unscaled_X_val_reward=fold_dict["unscaled_val_or_test_set"],
            expected_rewards_list=all_expected_rewards,
            pred_list=all_predictions,
        )
        return result

    # ==================================================================
    # Cross-validation entry-point
    # ==================================================================
    def run_cv(self, folds: Iterable[dict]) -> dict:
        state = CVState()

        for i, fold_dict in enumerate(folds):
            print(f"Processing fold {i + 1}/{self.cfg.cv_splits}")

            X_tr_out, t_tr, y_tr_out = fold_dict["train_outcome"]
            X_va_out, t_va, mu, y_va_out = fold_dict["val_or_test_outcome"]
            X_tr_rew, y_tr_rewards = fold_dict["train_reward"]
            X_va_rew, y_va_rewards = fold_dict["val_or_test_reward"]

            params_out, model_out, score_out = self._tune_outcome(
                X_tr_out, t_tr, mu, y_tr_out, X_va_out, t_va, y_va_out
            )
            params_rew, models_rew, score_rew = self._tune_reward(
                X_tr_rew, y_tr_rewards, X_va_rew, y_va_rewards
            )

            state.best_hparams_outcome.append(params_out)
            state.best_outcome_models.append(model_out)
            state.fold_scores_outcome.append(score_out)
            state.best_hparams_reward.append(params_rew)
            state.best_reward_models.append(models_rew)
            state.fold_scores_reward.append(score_rew)

            result = self._score_split(model_out, models_rew, fold_dict, is_final=False)
            state.fold_summaries.append(result["summary_df"])
            state.fold_decision_metrics.append(result["decision_metrics_df"])
            state.fold_ranked_decision_metrics.append(result["ranked_decision_metrics_df"])
            state.fold_rank_dicts.append(result["rank_dict"])
            state.fold_best_criteria.append(result["best_criterion"])

        suggested_out = self._select_best(state.best_hparams_outcome, state.fold_scores_outcome, maximize=True)
        suggested_rew = self._select_best(state.best_hparams_reward, state.fold_scores_reward, maximize=False)

        cv_results = {
            "best_hyperparams_outcome_per_fold": state.best_hparams_outcome,
            "best_outcome_models_per_fold": state.best_outcome_models,
            "best_hyperparams_reward_per_fold": state.best_hparams_reward,
            "best_reward_models_per_fold": state.best_reward_models,
            "suggested_params_outcome": suggested_out,
            "suggested_params_reward": suggested_rew,
            "all_fold_summaries": state.fold_summaries,
            "all_fold_decision_metrics": state.fold_decision_metrics,
            "all_fold_ranked_decision_metrics": state.fold_ranked_decision_metrics,
            "all_fold_rank_dicts": state.fold_rank_dicts,
            "all_fold_best_criteria": state.fold_best_criteria,
        }
        self._cv_results = cv_results
        return cv_results

    def aggregate_cv(self) -> dict:
        """Concatenate fold summaries and re-rank as if they were one big eval set."""
        summary_df = pd.concat(self._cv_results["all_fold_summaries"], ignore_index=True)
        decision_metrics_df = self.summary_processor.metrics_to_dataframe(
            self.metrics_calculator.compute_all_metrics(summary_df, true_outcome_col="True Outcome")
        )
        ranked, rank_dict, best = self.summary_processor.ranker.rank(decision_metrics_df)
        return {
            "summary_df": summary_df,
            "decision_metrics_df": decision_metrics_df,
            "ranked_decision_metrics_df": ranked,
            "rank_dict": rank_dict,
            "best_criterion": best,
            "suggested_params_outcome": self._cv_results["suggested_params_outcome"],
            "suggested_params_reward": self._cv_results["suggested_params_reward"],
        }

    # ==================================================================
    # Final evaluation
    # ==================================================================
    def run_final(self, data_processor, cv_results, all_train_set, test_set):
        prepared = data_processor.prepare_for_training(all_train_set, test_set)
        X_tr_out, t_tr, y_tr_out = prepared["train_outcome"]
        X_te_out, t_te, _, y_te_out = prepared["val_or_test_outcome"]
        X_tr_rew, y_tr_rewards = prepared["train_reward"]
        X_te_rew, y_te_rewards = prepared["val_or_test_reward"]

        sug_out = cv_results["suggested_params_outcome"]
        sug_rew = cv_results["suggested_params_reward"]

        outcome = self.factory.new_outcome_model()
        if self.factory.has_treatment:
            outcome.train(X_tr_out, t_tr, y_tr_out, **sug_out)
            final_score = outcome.evaluate(X_te_out, t_te, y_te_out)
            print(f"Final Outcome Model MAE: {final_score:.4f}")
        else:
            outcome.train(X_tr_out, y_tr_out, **sug_out)
            final_score = outcome.evaluate(X_te_out, y_te_out)
            print(f"Final Outcome Model Accuracy: {final_score:.4f}")

        reward_wrapper = self.factory.new_reward_models(**sug_rew)
        final_reward_models = reward_wrapper.train(X_tr_rew, y_tr_rewards)
        mse_results = reward_wrapper.evaluate(X_te_rew, y_te_rewards)
        print("Final Reward Models MSE:")
        for actor in self.cfg.actors.reward_types:
            print(f"  {actor}: {mse_results.get(f'{actor}_mse')}")

        results = self._score_split(outcome, final_reward_models, prepared, is_final=True)
        # Attach the test-set model-quality numbers for downstream reporting
        results["reward_mse_per_actor"] = {
            actor: float(mse_results.get(f"{actor}_mse"))
            for actor in self.cfg.actors.reward_types
        }
        results["outcome_score"] = float(final_score) if final_score is not None else None
        results["outcome_score_metric"] = "MAE" if self.factory.has_treatment else "Accuracy"
        print("Final evaluation on test set completed.")
        return results, sug_out, sug_rew, final_score

    # ==================================================================
    @staticmethod
    def _select_best(params_per_fold, scores_per_fold, *, maximize: bool):
        idx = int(np.argmax(scores_per_fold) if maximize else np.argmin(scores_per_fold))
        return params_per_fold[idx]
