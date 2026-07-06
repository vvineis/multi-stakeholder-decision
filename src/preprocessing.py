"""Data processing: train/test split, K-fold, scaling, one-hot encoding, reward augmentation."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from hydra.utils import instantiate
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Repo root (parent of this src/ package). Relative config paths resolve against
# it, so they work regardless of Hydra's runtime working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_path(path) -> Path:
    """Resolve `path` against the repo root if it is relative."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# ----------------------------------------------------------------------
# Use-case-specific dataset normalization
# ----------------------------------------------------------------------
# The original framework assumes the lending CSV has already been
# normalized to use underscore-separated identifiers (Applicant_Type,
# Loan_Amount, etc.) and underscore-separated outcome values (Fully_Repaid,
# Partially_Repaid, Not_Repaid). The lending_club_data.csv that ships with
# the repo still uses the raw "Applicant Type" / "Fully Repaid" form, so we
# normalize it here on load.
_LENDING_COLUMN_RENAMES = {
    "Applicant Type": "Applicant_Type",
    "Loan Amount": "Loan_Amount",
    "Interest Rate": "Interest_Rate",
    "Credit Score": "Credit_Score",
    "Applicant ID": "Applicant_ID",
}
_LENDING_OUTCOME_RENAMES = {
    "Fully Repaid": "Fully_Repaid",
    "Partially Repaid": "Partially_Repaid",
    "Not Repaid": "Not_Repaid",
}


def normalize_lending(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns + outcome values to the underscore form the framework expects."""
    df = df.rename(columns=_LENDING_COLUMN_RENAMES)
    if "Outcome" in df.columns:
        df["Outcome"] = df["Outcome"].replace(_LENDING_OUTCOME_RENAMES)
    return df


def load_dataset(cfg) -> pd.DataFrame:
    """Load + slice + use-case-normalize the raw CSV. Use this everywhere instead of pd.read_csv."""
    df = pd.read_csv(resolve_repo_path(cfg.data_path)).iloc[: cfg.sample_size]
    if cfg.use_case.name == "lending":
        df = normalize_lending(df)
    return df


class DataProcessor:
    def __init__(self, df: pd.DataFrame, cfg, random_split: bool = True):
        self.cfg = cfg
        self.df = df
        self.random_split = random_split

        self.feature_columns = list(cfg.context.feature_columns)
        self.columns_to_display = list(cfg.context.columns_to_display)
        self.categorical_columns = list(cfg.categorical_columns)
        self.test_size = cfg.test_size
        self.reward_types = list(cfg.actors.reward_types)
        self.n_splits = cfg.cv_splits

        self.scaler = StandardScaler()
        self.onehot_encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown="ignore")
        self.reward_calculator = instantiate(cfg.reward_calculator)
        self.augmentation_params = cfg.augmentation_for_rewards.get("augmentation_parameters", {})
        self.seed = int(cfg.get("seed", 111))

        self._split_data()

    # ------------------------------------------------------------------
    def _split_data(self):
        if self.random_split:
            self.train_df, self.test_df = train_test_split(self.df, test_size=self.test_size, random_state=self.seed)
        else:
            cut = int(len(self.df) * (1 - self.test_size))
            self.train_df, self.test_df = self.df.iloc[:cut], self.df.iloc[cut:]
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

    # ------------------------------------------------------------------
    def prepare_folds(self):
        for tr_idx, va_idx in self.kf.split(self.train_df):
            yield self.prepare_for_training(self.train_df.iloc[tr_idx], self.train_df.iloc[va_idx])

    def process(self):
        return self.prepare_folds(), self.train_df, self.test_df

    # ------------------------------------------------------------------
    def prepare_for_training(self, train_df, val_df):
        train_df = self.scale_features(train_df)
        unscaled_val = val_df[self.columns_to_display].copy()
        val_df = self.scale_features(val_df, fit=False)

        # Outcome data
        X_tr_out, t_tr, _, y_tr_out = self._prepare_for_outcome_prediction(train_df)
        X_va_out, t_va, mu_va, y_va_out = self._prepare_for_outcome_prediction(val_df)

        # Reward data
        augmented_train = self.augment_train_for_reward(train_df)
        X_tr_rew, y_tr_rewards = self._prepare_for_reward_prediction(augmented_train)
        X_tr_enc, X_va_enc = self.one_hot_encode(X_tr_rew, val_df)

        X_tr_rew_combined = pd.concat(
            [X_tr_rew[self.feature_columns].reset_index(drop=True), X_tr_enc.reset_index(drop=True)],
            axis=1,
        )
        X_va_rew_combined = pd.concat(
            [val_df[self.feature_columns].reset_index(drop=True), X_va_enc.reset_index(drop=True)],
            axis=1,
        )
        y_va_rewards = {rt: val_df[f"{rt}_reward"] for rt in self.reward_types}

        return {
            "train_outcome": (X_tr_out, t_tr, y_tr_out),
            "val_or_test_outcome": (X_va_out, t_va, mu_va, y_va_out),
            "train_reward": (X_tr_rew_combined, y_tr_rewards),
            "val_or_test_reward": (X_va_rew_combined, y_va_rewards),
            "val_or_test_set": val_df,
            "unscaled_val_or_test_set": unscaled_val,
            "scaler": self.scaler,
            "onehot_encoder": self.onehot_encoder,
        }

    # ------------------------------------------------------------------
    def scale_features(self, df, fit: bool = True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            numeric = df[self.feature_columns].select_dtypes(include=["int", "float"]).columns
            non_binary = [c for c in numeric if not ((df[c].min() >= 0) and (df[c].max() <= 1))]
            if fit:
                df.loc[:, non_binary] = self.scaler.fit_transform(df[non_binary])
            else:
                df.loc[:, non_binary] = self.scaler.transform(df[non_binary])
        return df

    def one_hot_encode(self, train_df, val_df):
        train_df["Outcome"] = train_df["Outcome"].astype(str)
        val_df["Outcome"] = val_df["Outcome"].astype(str)
        self.onehot_encoder.fit(train_df[["Action", "Outcome"]])
        tr_enc = self.onehot_encoder.transform(train_df[self.categorical_columns])
        va_enc = self.onehot_encoder.transform(val_df[self.categorical_columns])
        names = self.onehot_encoder.get_feature_names_out(self.categorical_columns)
        return (
            pd.DataFrame(tr_enc, columns=names, index=train_df.index),
            pd.DataFrame(va_enc, columns=names, index=val_df.index),
        )

    # ------------------------------------------------------------------
    def augment_train_for_reward(self, df: pd.DataFrame) -> pd.DataFrame:
        possible_actions = self.augmentation_params.get("actions_set", [0, 1])
        additional = list(self.augmentation_params.get("additional_arguments", []))
        rows = []
        for _, row in df.iterrows():
            rows.append(row.copy())
            for action in [a for a in possible_actions if a != row["Action"]]:
                dup = row.copy()
                dup["Action"] = action
                extra = [dup[arg] for arg in additional]
                rewards = self.reward_calculator.get_rewards(dup["Action"], dup["Outcome"], *extra)
                for rt, val in zip(self.reward_types, rewards):
                    dup[f"{rt}_reward"] = val
                rows.append(dup)
        return pd.DataFrame(rows).reset_index(drop=True)

    # ------------------------------------------------------------------
    def _prepare_for_outcome_prediction(self, df):
        X = df[self.feature_columns]
        y = df["Outcome"]
        treatment, mu = None, None
        if self.cfg.use_case.name == "health" and "Action" in df.columns:
            treatment = df["Action"]
            mu = df["Outcome_True"]
        return X, treatment, mu, y

    def _prepare_for_reward_prediction(self, df):
        reward_features = self.feature_columns + self.categorical_columns
        X = df[reward_features]
        y_rewards = {rt: df[f"{rt}_reward"] for rt in self.reward_types}
        return X, y_rewards
