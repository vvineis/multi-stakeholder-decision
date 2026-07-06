"""
Hydra entry-point. Pure orchestration — all the logic lives in `src/pipeline.py`.
"""
from __future__ import annotations

import os
import time

import json
import random

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.pipeline import Pipeline
from src.preprocessing import DataProcessor, load_dataset, resolve_repo_path


def _save_results(
    cfg,
    cv_results,
    final_results,
    suggested_params_outcome,
    suggested_params_reward,
    final_outcome_score,
):
    unique_id = time.strftime("%Y%m%d-%H%M%S")
    seed = int(cfg.get("seed", 111))
    folder = os.path.join(
        resolve_repo_path(cfg.result_path),
        f"run_{unique_id}_seed{seed}_Acc_{cfg.ranking_weights.Accuracy}_Fair_{cfg.ranking_weights.Demographic_Parity}",
    )
    os.makedirs(folder, exist_ok=True)

    cv_results["ranked_decision_metrics_df"].to_csv(
        os.path.join(folder, "cv_ranked_decision_metrics.csv"), index=False
    )
    final_results["ranked_decision_metrics_df"].to_csv(
        os.path.join(folder, "final_ranked_decision_metrics.csv"), index=False
    )

    # Structured run summary -- consumed by the dashboard's "Model performance"
    # panel AND by the sensitivity-analysis aggregation helpers in
    # run_sensitivity.ps1, which group runs by reward_variant, outcome_classifier,
    # and sample_size.
    outcome_cfg = cfg.models.outcome
    outcome_classifier = None
    if "classifier" in outcome_cfg:
        outcome_classifier = str(outcome_cfg.classifier.get("_target_", ""))
    elif "learner" in outcome_cfg:
        outcome_classifier = str(outcome_cfg.learner.get("_target_", ""))

    reward_variant = None
    if "reward_variant" in cfg.reward_calculator:
        reward_variant = str(cfg.reward_calculator.reward_variant)

    run_summary = {
        "seed": int(cfg.get("seed", 111)),
        "sample_size": int(cfg.sample_size),
        "cv_splits": int(cfg.cv_splits),
        "use_case": str(cfg.use_case.name),
        "model_type": str(cfg.models.outcome.model_type),
        "outcome_classifier": outcome_classifier,
        "reward_variant": reward_variant,
        "ranking_weights": dict(cfg.ranking_weights),
        "suggested_params_outcome": suggested_params_outcome,
        "suggested_params_reward": suggested_params_reward,
        "outcome_model": {
            "metric": final_results.get("outcome_score_metric", "Accuracy"),
            "value": final_results.get("outcome_score", final_outcome_score),
        },
        "reward_models_mse_per_actor": final_results.get("reward_mse_per_actor", {}),
        "best_criterion": final_results.get("best_criterion"),
    }
    with open(os.path.join(folder, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2, default=str)

    with open(os.path.join(folder, "suggested_params_and_scores.txt"), "w") as f:
        f.write(f"Suggested Params Outcome: {suggested_params_outcome}\n")
        f.write(f"Suggested Params Reward: {suggested_params_reward}\n")
        f.write(f"Final Outcome Score: {final_outcome_score}\n")
        f.write(f"samples: {cfg.sample_size}\n")
        f.write(f"cv_splits: {cfg.cv_splits}\n")
        f.write(f"seed: {seed}\n")
    print(f"Results saved to: {folder}")


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # ----- seed every RNG we know about -----
    seed = int(cfg.get("seed", 111))
    np.random.seed(seed)
    random.seed(seed)

    # ----- load & compute rewards -----
    df = load_dataset(cfg)
    reward_calculator = instantiate(cfg.reward_calculator)
    print(f"Initialized Reward Calculator: {reward_calculator}")
    df_ready = reward_calculator.compute_rewards(df)

    # ----- preprocessing -----
    data_processor = DataProcessor(df=df_ready, cfg=cfg, random_split=True)
    folds, train_set, test_set = data_processor.process()
    print(f"Train set shape: {train_set.shape}  Test set shape: {test_set.shape}")

    # ----- pipeline -----
    pipeline = Pipeline(cfg)
    pipeline.run_cv(folds)
    cv_results = pipeline.aggregate_cv()
    print("Aggregated CV Results:")
    print(cv_results)

    print("Training final models on entire training set and evaluating on test set...")
    final_results, suggested_params_outcome, suggested_params_reward, final_outcome_score = (
        pipeline.run_final(data_processor, cv_results, train_set, test_set)
    )
    print(final_results)

    # ----- persist -----
    os.makedirs(resolve_repo_path(cfg.result_path), exist_ok=True)
    _save_results(
        cfg,
        cv_results,
        final_results,
        suggested_params_outcome,
        suggested_params_reward,
        final_outcome_score,
    )


if __name__ == "__main__":
    main()
