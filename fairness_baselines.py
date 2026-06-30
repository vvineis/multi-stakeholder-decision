"""
Fairness-constrained baselines for the lending classification scenario.

Wraps two families of widely-cited methods from the Fairlearn library so that
their per-row decisions slot directly into the framework's existing metrics
pipeline (radar charts, decision-metrics CSV, dashboard).

Implemented baselines
---------------------
    FairLearn_EG_DP   ExponentiatedGradient + DemographicParity (Agarwal et al. 2018)
    FairLearn_EG_EO   ExponentiatedGradient + EqualizedOdds      (binary version)
    FairLearn_TO_DP   ThresholdOptimizer    + demographic_parity (Hardt et al. 2016)
    FairLearn_TO_EO   ThresholdOptimizer    + equalized_odds     (Hardt et al. 2016)

Each baseline produces a Series of suggested actions for the test set, using
the configured outcome -> action mapping in `cfg.actions_outcomes.mapping`,
so its metrics are computed on the SAME basis as Outcome_Pred_Model, Bank, Maximin etc.

Install
-------
    pip install fairlearn>=0.10

Usage
-----
1. Train the main framework first:
       python main.py use_case=lending sample_size=10000 cv_splits=3

2. Run the baselines on the same data split, optionally merging into the
   existing decision-metrics CSV so all rows can be re-ranked together:
       python fairness_baselines.py use_case=lending sample_size=10000 \\
           merge_with=results/lending/run_xxx/final_ranked_decision_metrics.csv

3. Or import the baselines programmatically:
       from fairness_baselines import ExponentiatedGradientDP
       baseline = ExponentiatedGradientDP(eps=0.05)
       actions = baseline.fit_predict_actions(train_df, test_df, cfg)
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import DataProcessor, load_dataset
from utils.metrics.computer import MetricsCalculator
from utils.ranking.ranker import Ranker


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------
class FairnessBaseline(ABC):
    """A fair-classification baseline that maps (train, test) -> Series of actions."""

    name: str = "FairnessBaseline"

    @abstractmethod
    def fit_predict_actions(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, cfg
    ) -> pd.Series:
        """Return a Series indexed like `test_df` whose values are action strings."""


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def _split_xys(df: pd.DataFrame, cfg, *, binary: bool):
    """Extract (X, y, sensitive features) from a raw dataframe.

    If `binary=True`, y is 1 iff the row's Outcome is the configured
    positive_outcomes_set[0], else 0 -- needed for EqualizedOdds and for
    ThresholdOptimizer which require binary labels.
    """
    feature_cols = list(cfg.context.feature_columns)
    sens_col = cfg.case_specific_metrics.positive_attribute_for_fairness
    X = df[feature_cols].copy()
    s = df[sens_col].astype(int).values
    if binary:
        pos_outcome = cfg.actions_outcomes.positive_outcomes_set[0]
        y = (df["Outcome"] == pos_outcome).astype(int).values
    else:
        y = df["Outcome"].values
    return X, y, s


def _map_outcome_to_action(predicted_outcomes, cfg) -> list:
    """Use cfg.actions_outcomes.mapping (same one Outcome_Pred_Model uses)."""
    mapping = dict(cfg.actions_outcomes.mapping)
    default = mapping.get("default", list(cfg.actions_outcomes.actions_set)[0])
    return [mapping.get(o, default) for o in predicted_outcomes]


def _map_binary_to_action(predicted_binary, cfg) -> list:
    """Binary baselines: 1 -> first positive action, 0 -> first non-positive action."""
    pos_action = cfg.actions_outcomes.positive_actions_set[0]
    neg_action = next(
        (a for a in cfg.actions_outcomes.actions_set if a not in cfg.actions_outcomes.positive_actions_set),
        "Not_Grant",
    )
    return [pos_action if int(p) == 1 else neg_action for p in predicted_binary]


def _base_estimator(n_estimators: int, random_state: int):
    """Default base classifier; RandomForest exposes sample_weight required by ExponentiatedGradient."""
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)


# ----------------------------------------------------------------------
# Reductions approach (Agarwal et al. 2018) -- in-processing
# ----------------------------------------------------------------------
@dataclass
class ExponentiatedGradientDP(FairnessBaseline):
    """ExponentiatedGradient + DemographicParity.

    Fairlearn's reductions only support binary {0, 1} labels, so the
    lending problem is collapsed to a binary task (positive outcome vs.
    rest) and the baseline emits Grant / Not_Grant. The Grant_lower action
    is not predictable from a binary fair classifier -- this is a standard
    limitation of fair-classification baselines and is fair to acknowledge
    in the paper.
    """

    eps: float = 0.05
    n_estimators: int = 50
    random_state: int = 42
    max_iter: int = 50
    name: str = field(default="FairLearn_EG_DP", init=False)

    def fit_predict_actions(self, train_df, test_df, cfg):
        from fairlearn.reductions import DemographicParity, ExponentiatedGradient

        X_tr, y_tr, s_tr = _split_xys(train_df, cfg, binary=True)
        X_te, _, _ = _split_xys(test_df, cfg, binary=True)
        clf = _base_estimator(self.n_estimators, self.random_state)
        eg = ExponentiatedGradient(
            estimator=clf,
            constraints=DemographicParity(),
            eps=self.eps,
            max_iter=self.max_iter,
        )
        eg.fit(X_tr, y_tr, sensitive_features=s_tr)
        y_pred = eg.predict(X_te)
        return pd.Series(_map_binary_to_action(y_pred, cfg), index=test_df.index)


@dataclass
class ExponentiatedGradientEO(FairnessBaseline):
    """ExponentiatedGradient + EqualizedOdds. Binary only -- positive outcome vs others."""

    eps: float = 0.05
    n_estimators: int = 50
    random_state: int = 42
    max_iter: int = 50
    name: str = field(default="FairLearn_EG_EO", init=False)

    def fit_predict_actions(self, train_df, test_df, cfg):
        from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient

        X_tr, y_tr, s_tr = _split_xys(train_df, cfg, binary=True)
        X_te, _, _ = _split_xys(test_df, cfg, binary=True)
        clf = _base_estimator(self.n_estimators, self.random_state)
        eg = ExponentiatedGradient(
            estimator=clf,
            constraints=EqualizedOdds(),
            eps=self.eps,
            max_iter=self.max_iter,
        )
        eg.fit(X_tr, y_tr, sensitive_features=s_tr)
        y_pred = eg.predict(X_te)
        return pd.Series(_map_binary_to_action(y_pred, cfg), index=test_df.index)


# ----------------------------------------------------------------------
# Post-processing (Hardt et al. 2016) -- threshold adjustment per group
# ----------------------------------------------------------------------
@dataclass
class ThresholdOptimizerDP(FairnessBaseline):
    n_estimators: int = 50
    random_state: int = 42
    name: str = field(default="FairLearn_TO_DP", init=False)

    def fit_predict_actions(self, train_df, test_df, cfg):
        from fairlearn.postprocessing import ThresholdOptimizer

        X_tr, y_tr, s_tr = _split_xys(train_df, cfg, binary=True)
        X_te, _, s_te = _split_xys(test_df, cfg, binary=True)
        clf = _base_estimator(self.n_estimators, self.random_state)
        to = ThresholdOptimizer(
            estimator=clf,
            constraints="demographic_parity",
            prefit=False,
            predict_method="predict_proba",
        )
        to.fit(X_tr, y_tr, sensitive_features=s_tr)
        y_pred = to.predict(X_te, sensitive_features=s_te)
        return pd.Series(_map_binary_to_action(y_pred, cfg), index=test_df.index)


@dataclass
class ThresholdOptimizerEO(FairnessBaseline):
    n_estimators: int = 50
    random_state: int = 42
    name: str = field(default="FairLearn_TO_EO", init=False)

    def fit_predict_actions(self, train_df, test_df, cfg):
        from fairlearn.postprocessing import ThresholdOptimizer

        X_tr, y_tr, s_tr = _split_xys(train_df, cfg, binary=True)
        X_te, _, s_te = _split_xys(test_df, cfg, binary=True)
        clf = _base_estimator(self.n_estimators, self.random_state)
        to = ThresholdOptimizer(
            estimator=clf,
            constraints="equalized_odds",
            prefit=False,
            predict_method="predict_proba",
        )
        to.fit(X_tr, y_tr, sensitive_features=s_tr)
        y_pred = to.predict(X_te, sensitive_features=s_te)
        return pd.Series(_map_binary_to_action(y_pred, cfg), index=test_df.index)


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
BASELINES = {
    "FairLearn_EG_DP": ExponentiatedGradientDP,
    "FairLearn_EG_EO": ExponentiatedGradientEO,
    "FairLearn_TO_DP": ThresholdOptimizerDP,
    "FairLearn_TO_EO": ThresholdOptimizerEO,
}


# ----------------------------------------------------------------------
# Runner: train baselines and compute metrics via the existing pipeline
# ----------------------------------------------------------------------
def _build_summary_df(test_df: pd.DataFrame, baselines: List[FairnessBaseline],
                      action_series: dict, cfg) -> pd.DataFrame:
    """Mimic the structure of SummaryProcessor.create_summary_df just enough for MetricsCalculator."""
    columns = list(cfg.context.columns_to_display)
    summary_df = test_df[columns].reset_index(drop=True).copy()
    summary_df["True Outcome"] = test_df["Outcome"].values
    summary_df["Real Treatment"] = None  # classification case
    for b in baselines:
        col = f"{b.name} Suggested Action"
        summary_df[col] = action_series[b.name].values
    return summary_df


def evaluate_fairness_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baselines: List[FairnessBaseline],
    cfg,
) -> pd.DataFrame:
    """
    Train each baseline on `train_df`, predict on `test_df`, then run the
    framework's MetricsCalculator with the baseline names appended to the actor list.

    Returns
    -------
    decision_metrics_df : one row per baseline, columns aligned with
    `final_ranked_decision_metrics.csv` (minus the live ranking columns).
    """
    print(f"Training {len(baselines)} fairness baselines...")
    action_series = {}
    for b in baselines:
        print(f"  - fitting {b.name}")
        action_series[b.name] = b.fit_predict_actions(train_df, test_df, cfg)

    summary_df = _build_summary_df(test_df, baselines, action_series, cfg)

    # Inject baseline names into a clone of the actor list so MetricsCalculator picks them up.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    extended_actors = list(cfg.actors.actor_list) + [b.name for b in baselines]
    cfg_dict["actors"]["actor_list"] = extended_actors
    # Also drop existing actor columns we don't have so the loop short-circuits cleanly.
    cfg_extended = OmegaConf.create(cfg_dict)

    calc = MetricsCalculator(cfg_extended)
    all_metrics = calc.compute_all_metrics(summary_df, true_outcome_col="True Outcome")

    rows = []
    for b in baselines:
        if b.name in all_metrics and all_metrics[b.name]:
            rows.append({"Actor/Criterion": b.name, **all_metrics[b.name]})
    return pd.DataFrame(rows)


def merge_and_rerank(
    existing_csv: str,
    baseline_metrics_df: pd.DataFrame,
    cfg,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Append baseline rows to an existing decision-metrics CSV and re-rank all of them."""
    existing = pd.read_csv(existing_csv)

    drop_cols = [
        c
        for c in existing.columns
        if any(s in c for s in ("Normalized", "Rank", "Weighted Normalized-Sum"))
    ]
    existing = existing.drop(columns=drop_cols, errors="ignore")

    combined = pd.concat([existing, baseline_metrics_df], ignore_index=True, sort=False)

    ranker = Ranker.from_hydra(cfg)
    ranked, _, best = ranker.rank(combined)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        ranked.to_csv(output_path, index=False)
    return ranked, best


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.use_case.name != "lending":
        raise SystemExit(
            f"Fairness baselines target the lending case; got use_case={cfg.use_case.name}."
        )

    print(OmegaConf.to_yaml(cfg))

    # Load data and recreate the SAME split main.py would use.
    df = load_dataset(cfg)
    reward_calculator = instantiate(cfg.reward_calculator)
    df_ready = reward_calculator.compute_rewards(df)
    dp = DataProcessor(df=df_ready, cfg=cfg, random_split=True)
    _, train_set, test_set = dp.process()
    print(f"Train: {train_set.shape}  Test: {test_set.shape}")

    # Pick baselines (CLI override: baselines=[FairLearn_EG_DP,FairLearn_TO_EO]).
    # Treat null/None as "use all available".
    raw_choice = cfg.get("baselines", None)
    chosen = list(raw_choice) if raw_choice else list(BASELINES.keys())
    unknown = [c for c in chosen if c not in BASELINES]
    if unknown:
        raise SystemExit(f"Unknown baselines: {unknown}. Available: {list(BASELINES)}")
    baselines = [BASELINES[name]() for name in chosen]
    print(f"Selected baselines: {[b.name for b in baselines]}")

    # Run + collect metrics
    metrics_df = evaluate_fairness_baselines(train_set, test_set, baselines, cfg)
    print("\nBaseline decision metrics:")
    print(metrics_df.to_string(index=False))

    # Output folder
    unique_id = time.strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(cfg.result_path, f"fairness_baselines_{unique_id}")
    os.makedirs(folder, exist_ok=True)
    baselines_csv = os.path.join(folder, "baseline_decision_metrics.csv")
    metrics_df.to_csv(baselines_csv, index=False)
    print(f"\nWrote {baselines_csv}")

    # Optional: merge into an existing run's CSV and re-rank
    merge_with = cfg.get("merge_with", None) or None  # treat null as no-merge
    if merge_with:
        out_path = os.path.join(folder, "merged_ranked_decision_metrics.csv")
        ranked, best = merge_and_rerank(merge_with, metrics_df, cfg, output_path=out_path)
        print(f"Merged + re-ranked written to {out_path}")
        print(f"Best actor/criterion after baselines: {best}")


if __name__ == "__main__":
    main()
