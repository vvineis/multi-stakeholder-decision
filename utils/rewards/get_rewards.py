"""
Reward calculators for lending (RewardCalculator) and healthcare (HealthRewardCalculator).

The lending reward structures come in three flavors -- 'base', 'mild', and
'strictest' -- matching the variants the original framework supported via
commented-out code blocks. Now they're picked via the `reward_variant` field
in the Hydra config, so changing the scenario is a CLI override:

    python main.py use_case=lending reward_calculator.reward_variant=strictest
"""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Lending
# ----------------------------------------------------------------------
class RewardCalculator:
    """Stakeholder rewards for the lending use case (base / mild / strictest)."""

    REWARD_STRUCTURES_BASE = {
        0: {  # Non-vulnerable applicants
            "Bank": {
                ("Grant", "Fully_Repaid"): 1.0,
                ("Grant", "Partially_Repaid"): 0.5,
                ("Grant", "Not_Repaid"): 0.0,
                ("Grant_lower", "Fully_Repaid"): 0.8,
                ("Grant_lower", "Partially_Repaid"): 1,
                ("Grant_lower", "Not_Repaid"): 0,
                ("Not_Grant", "Fully_Repaid"): 0.2,
                ("Not_Grant", "Partially_Repaid"): 0.5,
                ("Not_Grant", "Not_Repaid"): 1.0,
            },
            "Applicant": {
                ("Grant", "Fully_Repaid"): 1.0,
                ("Grant", "Partially_Repaid"): 0.5,
                ("Grant", "Not_Repaid"): 0.3,
                ("Grant_lower", "Fully_Repaid"): 0.7,
                ("Grant_lower", "Partially_Repaid"): 0.8,
                ("Grant_lower", "Not_Repaid"): 0.4,
                ("Not_Grant", "Fully_Repaid"): 0.2,
                ("Not_Grant", "Partially_Repaid"): 0.5,
                ("Not_Grant", "Not_Repaid"): 0.7,
            },
            "Regulatory": {
                ("Grant", "Fully_Repaid"): 1.0,
                ("Grant", "Partially_Repaid"): 0.2,
                ("Grant", "Not_Repaid"): 0.0,
                ("Grant_lower", "Fully_Repaid"): 0.8,
                ("Grant_lower", "Partially_Repaid"): 1,
                ("Grant_lower", "Not_Repaid"): 0.1,
                ("Not_Grant", "Fully_Repaid"): 0.5,
                ("Not_Grant", "Partially_Repaid"): 0.7,
                ("Not_Grant", "Not_Repaid"): 1.0,
            },
        },
        1: {  # Vulnerable applicants
            "Bank": {
                ("Grant", "Fully_Repaid"): 1.0,
                ("Grant", "Partially_Repaid"): 0.5,
                ("Grant", "Not_Repaid"): 0.0,
                ("Grant_lower", "Fully_Repaid"): 0.8,
                ("Grant_lower", "Partially_Repaid"): 1,
                ("Grant_lower", "Not_Repaid"): 0,
                ("Not_Grant", "Fully_Repaid"): 0.0,
                ("Not_Grant", "Partially_Repaid"): 0.2,
                ("Not_Grant", "Not_Repaid"): 1.0,
            },
            "Applicant": {
                ("Grant", "Fully_Repaid"): 1.0,
                ("Grant", "Partially_Repaid"): 0.7,
                ("Grant", "Not_Repaid"): 0.5,
                ("Grant_lower", "Fully_Repaid"): 0.5,
                ("Grant_lower", "Partially_Repaid"): 0.8,
                ("Grant_lower", "Not_Repaid"): 0.3,
                ("Not_Grant", "Fully_Repaid"): 0.0,
                ("Not_Grant", "Partially_Repaid"): 0.2,
                ("Not_Grant", "Not_Repaid"): 0.6,
            },
            "Regulatory": {
                ("Grant", "Fully_Repaid"): 1.0,
                ("Grant", "Partially_Repaid"): 0.5,
                ("Grant", "Not_Repaid"): 0.3,
                ("Grant_lower", "Fully_Repaid"): 0.7,
                ("Grant_lower", "Partially_Repaid"): 1,
                ("Grant_lower", "Not_Repaid"): 0.2,
                ("Not_Grant", "Fully_Repaid"): 0.3,
                ("Not_Grant", "Partially_Repaid"): 0.5,
                ("Not_Grant", "Not_Repaid"): 0.8,
            },
        },
    }

    # Mild variant: as in the original framework -- applicants more lenient with non-positive outcomes,
    # bank slightly stricter on partial repayment.
    REWARD_STRUCTURES_MILD = {
        0: {
            "Bank": {
                ("Grant", "Fully_Repaid"): 1.0, ("Grant", "Partially_Repaid"): 0.2, ("Grant", "Not_Repaid"): 0.0,
                ("Grant_lower", "Fully_Repaid"): 0.4, ("Grant_lower", "Partially_Repaid"): 0.5, ("Grant_lower", "Not_Repaid"): 0,
                ("Not_Grant", "Fully_Repaid"): 0.2, ("Not_Grant", "Partially_Repaid"): 0.5, ("Not_Grant", "Not_Repaid"): 1.0,
            },
            "Applicant": {
                ("Grant", "Fully_Repaid"): 1.0, ("Grant", "Partially_Repaid"): 0.7, ("Grant", "Not_Repaid"): 0.5,
                ("Grant_lower", "Fully_Repaid"): 0.7, ("Grant_lower", "Partially_Repaid"): 0.8, ("Grant_lower", "Not_Repaid"): 0.6,
                ("Not_Grant", "Fully_Repaid"): 0, ("Not_Grant", "Partially_Repaid"): 0, ("Not_Grant", "Not_Repaid"): 0,
            },
            "Regulatory": {
                ("Grant", "Fully_Repaid"): 1.0, ("Grant", "Partially_Repaid"): 0.2, ("Grant", "Not_Repaid"): 0.0,
                ("Grant_lower", "Fully_Repaid"): 0.8, ("Grant_lower", "Partially_Repaid"): 1, ("Grant_lower", "Not_Repaid"): 0.1,
                ("Not_Grant", "Fully_Repaid"): 0.5, ("Not_Grant", "Partially_Repaid"): 0.7, ("Not_Grant", "Not_Repaid"): 1.0,
            },
        },
        1: {
            "Bank": {
                ("Grant", "Fully_Repaid"): 1.0, ("Grant", "Partially_Repaid"): 0.5, ("Grant", "Not_Repaid"): 0.0,
                ("Grant_lower", "Fully_Repaid"): 0.5, ("Grant_lower", "Partially_Repaid"): 0.5, ("Grant_lower", "Not_Repaid"): 0,
                ("Not_Grant", "Fully_Repaid"): 0.0, ("Not_Grant", "Partially_Repaid"): 0.2, ("Not_Grant", "Not_Repaid"): 1.0,
            },
            "Applicant": {
                ("Grant", "Fully_Repaid"): 1.0, ("Grant", "Partially_Repaid"): 0.8, ("Grant", "Not_Repaid"): 0.7,
                ("Grant_lower", "Fully_Repaid"): 0.8, ("Grant_lower", "Partially_Repaid"): 1, ("Grant_lower", "Not_Repaid"): 0.5,
                ("Not_Grant", "Fully_Repaid"): 0.0, ("Not_Grant", "Partially_Repaid"): 0, ("Not_Grant", "Not_Repaid"): 0.2,
            },
            "Regulatory": {
                ("Grant", "Fully_Repaid"): 1.0, ("Grant", "Partially_Repaid"): 0.5, ("Grant", "Not_Repaid"): 0.3,
                ("Grant_lower", "Fully_Repaid"): 0.7, ("Grant_lower", "Partially_Repaid"): 1, ("Grant_lower", "Not_Repaid"): 0.2,
                ("Not_Grant", "Fully_Repaid"): 0.3, ("Not_Grant", "Partially_Repaid"): 0.5, ("Not_Grant", "Not_Repaid"): 0.8,
            },
        },
    }

    # Strictest variant: bank assigns zero to anything that isn't full repayment;
    # applicants prefer access to credit regardless of outcome.
    REWARD_STRUCTURES_STRICTEST = {
        0: {
            "Bank": {
                ("Grant", "Fully_Repaid"): 1.0, ("Grant", "Partially_Repaid"): 0, ("Grant", "Not_Repaid"): 0.0,
                ("Grant_lower", "Fully_Repaid"): 0.8, ("Grant_lower", "Partially_Repaid"): 0, ("Grant_lower", "Not_Repaid"): 0,
                ("Not_Grant", "Fully_Repaid"): 0.5, ("Not_Grant", "Partially_Repaid"): 1, ("Not_Grant", "Not_Repaid"): 1.0,
            },
            "Applicant": {
                ("Grant", "Fully_Repaid"): 1.0, ("Grant", "Partially_Repaid"): 0.8, ("Grant", "Not_Repaid"): 0.7,
                ("Grant_lower", "Fully_Repaid"): 0.7, ("Grant_lower", "Partially_Repaid"): 0.8, ("Grant_lower", "Not_Repaid"): 0.6,
                ("Not_Grant", "Fully_Repaid"): 0, ("Not_Grant", "Partially_Repaid"): 0, ("Not_Grant", "Not_Repaid"): 0,
            },
            "Regulatory": {
                ("Grant", "Fully_Repaid"): 1.0, ("Grant", "Partially_Repaid"): 0.2, ("Grant", "Not_Repaid"): 0.0,
                ("Grant_lower", "Fully_Repaid"): 0.8, ("Grant_lower", "Partially_Repaid"): 1, ("Grant_lower", "Not_Repaid"): 0.1,
                ("Not_Grant", "Fully_Repaid"): 0.5, ("Not_Grant", "Partially_Repaid"): 0.7, ("Not_Grant", "Not_Repaid"): 1.0,
            },
        },
        1: {
            "Bank": {
                ("Grant", "Fully_Repaid"): 1.0, ("Grant", "Partially_Repaid"): 0, ("Grant", "Not_Repaid"): 0,
                ("Grant_lower", "Fully_Repaid"): 0.8, ("Grant_lower", "Partially_Repaid"): 0, ("Grant_lower", "Not_Repaid"): 0,
                ("Not_Grant", "Fully_Repaid"): 0.5, ("Not_Grant", "Partially_Repaid"): 1, ("Not_Grant", "Not_Repaid"): 1.0,
            },
            "Applicant": {
                ("Grant", "Fully_Repaid"): 1.0, ("Grant", "Partially_Repaid"): 0.8, ("Grant", "Not_Repaid"): 0.7,
                ("Grant_lower", "Fully_Repaid"): 0.8, ("Grant_lower", "Partially_Repaid"): 1, ("Grant_lower", "Not_Repaid"): 0.6,
                ("Not_Grant", "Fully_Repaid"): 0.0, ("Not_Grant", "Partially_Repaid"): 0, ("Not_Grant", "Not_Repaid"): 0,
            },
            "Regulatory": {
                ("Grant", "Fully_Repaid"): 1.0, ("Grant", "Partially_Repaid"): 0.5, ("Grant", "Not_Repaid"): 0.3,
                ("Grant_lower", "Fully_Repaid"): 0.7, ("Grant_lower", "Partially_Repaid"): 1, ("Grant_lower", "Not_Repaid"): 0.2,
                ("Not_Grant", "Fully_Repaid"): 0.3, ("Not_Grant", "Partially_Repaid"): 0.5, ("Not_Grant", "Not_Repaid"): 0.8,
            },
        },
    }

    _VARIANT_TO_ATTR = {
        "base": "REWARD_STRUCTURES_BASE",
        "mild": "REWARD_STRUCTURES_MILD",
        "strictest": "REWARD_STRUCTURES_STRICTEST",
    }

    # Back-compat alias so old code that does `RewardCalculator.REWARD_STRUCTURES` still works.
    REWARD_STRUCTURES = REWARD_STRUCTURES_BASE

    @classmethod
    def get_structures_for_variant(cls, variant: str) -> dict:
        """Class-level lookup so MetricsCalculator can resolve real-payoff structures."""
        attr = cls._VARIANT_TO_ATTR.get(variant)
        if attr is None:
            raise ValueError(
                f"Unknown reward_variant '{variant}'. Choose from {list(cls._VARIANT_TO_ATTR)}."
            )
        return getattr(cls, attr)

    def __init__(self, reward_types, noise_level: float = 0.05, reward_variant: str = "base"):
        self.reward_types = list(reward_types)
        self.noise_level = noise_level
        self.reward_variant = reward_variant
        self.reward_structures = RewardCalculator.get_structures_for_variant(reward_variant)
        # Per-instance alias so get_rewards (which reads self.REWARD_STRUCTURES) follows the variant.
        self.REWARD_STRUCTURES = self.reward_structures

    # ------------------------------------------------------------------
    def get_rewards(self, action, outcome, applicant_type, loan_amount, interest_rate):
        base = self.REWARD_STRUCTURES[applicant_type]
        rewards = {rt: base[rt][(action, outcome)] for rt in self.reward_types}
        adjusted = self._adjust(rewards, loan_amount, interest_rate)
        return [adjusted[rt] for rt in self.reward_types]

    def _adjust(self, rewards, loan_amount, interest_rate):
        loan_factor = np.clip(loan_amount / 10000, 0.5, 1.5)
        rate_factor = np.clip(interest_rate, 0.05, 0.25)
        out = {}
        for rt, val in rewards.items():
            if rt == "Bank":
                adj = val * loan_factor * (1 + rate_factor)
            elif rt == "Applicant":
                adj = val * (2 - rate_factor) * (1 - loan_factor)
            elif rt == "Regulatory":
                adj = val * (1 - rate_factor) * loan_factor
            else:
                adj = val * loan_factor * (1 + rate_factor / 2)
            adj += np.random.uniform(-self.noise_level, self.noise_level)
            out[rt] = np.clip(adj, 0, 1)
        return out

    def compute_rewards(self, df: pd.DataFrame) -> pd.DataFrame:
        rewards = df.apply(
            lambda r: self.get_rewards(
                r["Action"], r["Outcome"], r["Applicant_Type"], r["Loan_Amount"], r["Interest_Rate"]
            ),
            axis=1,
        )
        df[["Bank_reward", "Applicant_reward", "Regulatory_reward"]] = pd.DataFrame(
            rewards.tolist(), index=df.index
        )
        return df


# ----------------------------------------------------------------------
# Healthcare
# ----------------------------------------------------------------------
class HealthRewardCalculator:
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.5,
        gamma: float = 0.6,
        noise_level: float = 0.05,
        fixed_cost: float = 100,
        base_cost: dict | None = None,
        reward_types=None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.noise_level = noise_level
        self.fixed_cost = fixed_cost
        self.base_cost = base_cost or {"A": 1, "C": 0}
        self.reward_types = list(reward_types) if reward_types else ["Healthcare_Provider", "Policy_Maker", "Parent"]
        self.min_outcome_action: dict = {}
        self.max_outcome_action: dict = {}

    # ------------------------------------------------------------------
    def compute_rewards(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["Action", "Outcome", "x23"]
        if not all(c in df.columns for c in required):
            raise ValueError(f"Input DataFrame must contain: {required}")

        df["Outcome"] = df["Outcome"].astype(float)
        self.min_outcome_action = df.groupby("Action")["Outcome"].min().to_dict()
        self.max_outcome_action = df.groupby("Action")["Outcome"].max().to_dict()

        for rt in self.reward_types:
            fn = self._reward_fn(rt)
            df[f"{rt}_reward"] = df.apply(fn, axis=1)
        return df

    def _reward_fn(self, reward_type):
        if reward_type == "Healthcare_Provider":
            return lambda row: self._healthcare_provider_reward(row["Outcome"], row["Action"])
        if reward_type == "Policy_Maker":
            return lambda row: self._policy_maker_reward(row["Outcome"], row["x23"], row["Action"])
        if reward_type == "Parent":
            return lambda row: self._parent_reward(row["Outcome"])
        raise ValueError(f"Unknown reward type: {reward_type}")

    def get_rewards(self, action, outcome, x23):
        rewards = {}
        for rt in self.reward_types:
            if rt == "Parent":
                rewards[rt] = self._parent_reward(outcome)
            elif rt == "Policy_Maker":
                rewards[rt] = self._policy_maker_reward(outcome, x23, action)
            elif rt == "Healthcare_Provider":
                rewards[rt] = self._healthcare_provider_reward(outcome, action)
            else:
                raise ValueError(f"Unknown reward type: {rt}")
        return [rewards[rt] for rt in self.reward_types]

    # ------------------------------------------------------------------
    def _healthcare_provider_reward(self, outcome, action):
        baseline = self.min_outcome_action.get("C", 2)
        improvement = max(0, outcome - baseline)
        cost = self.base_cost.get(action, 0)
        alpha = 0.8
        max_improvement = self.max_outcome_action.get("A", 12) - baseline
        normalized = improvement / (max_improvement + 1e-10)
        reward = alpha * normalized + (1 - alpha) * (1 - cost / max(self.base_cost.values()))
        reward += np.random.uniform(-self.noise_level, self.noise_level)
        return np.clip(reward, 0, 1)

    def _policy_maker_reward(self, outcome, x23, action):
        lo = self.min_outcome_action.get(action, 0)
        improvement = max(0, outcome - lo)
        denom = self.max_outcome_action.get(action, 1) - lo
        normalized = improvement / (denom + 1e-10)
        weight = 1.0 + self.beta * (x23 - 0.5)
        reward = normalized * weight + np.random.uniform(-self.noise_level, self.noise_level)
        return np.clip(reward, 0, 1)

    def _parent_reward(self, outcome):
        hi = max(self.max_outcome_action.values(), default=12)
        lo = min(self.min_outcome_action.values(), default=0)
        return np.clip((outcome - lo) / (hi - lo + 1e-10), 0, 1)
