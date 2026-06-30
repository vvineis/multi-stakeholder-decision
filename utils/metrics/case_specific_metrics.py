"""Case-specific metrics for lending and healthcare use cases."""
import pandas as pd


class LendingCaseMetrics:
    AVAILABLE = ("Total_Profit", "Total_Loss", "Unexploited_Profit")

    def __init__(self, suggestions_df, decision_col, true_outcome_col, cfg=None):
        if decision_col not in suggestions_df.columns or true_outcome_col not in suggestions_df.columns:
            raise ValueError(f"Columns {decision_col} or {true_outcome_col} not found in the DataFrame")
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.true_outcome_col = true_outcome_col

    def _potential_profit(self):
        return (self.suggestions_df["Loan_Amount"] * self.suggestions_df["Interest_Rate"] / 100).sum()

    def compute_total_profit(self):
        df = self.suggestions_df
        mask_full = (df[self.decision_col] == "Grant") & (df[self.true_outcome_col] == "Fully_Repaid")
        mask_part = (df[self.decision_col] == "Grant_lower") & (df[self.true_outcome_col] == "Partially_Repaid")
        profit_full = (df.loc[mask_full, "Loan_Amount"] * df.loc[mask_full, "Interest_Rate"] / 100).sum()
        profit_part = (df.loc[mask_part, "Loan_Amount"] * df.loc[mask_part, "Interest_Rate"] / 100).sum()
        return (profit_full + profit_part) / self._potential_profit()

    def compute_total_loss(self):
        df = self.suggestions_df
        mask_part = (df[self.decision_col] == "Grant_lower") & (df[self.true_outcome_col] == "Partially_Repaid")
        mask_nope = (df[self.decision_col] == "Grant") & (df[self.true_outcome_col] == "Not_Repaid")
        loss_part = (df.loc[mask_part, "Loan_Amount"] - df.loc[mask_part, "Recoveries"]).sum()
        loss_nope = df.loc[mask_nope, "Loan_Amount"].sum()
        potential_loss = df["Loan_Amount"].sum()
        return (loss_part + loss_nope) / potential_loss

    def compute_unexploited_profit(self):
        df = self.suggestions_df
        mask = (df[self.decision_col] == "Not_Grant") & (df[self.true_outcome_col] == "Fully_Repaid")
        unexploited = (df.loc[mask, "Loan_Amount"] * df.loc[mask, "Interest_Rate"] / 100).sum()
        return unexploited / self._potential_profit()

    def get_metrics(self, case_metrics_list):
        registry = {
            "Total_Profit": self.compute_total_profit,
            "Total_Loss": self.compute_total_loss,
            "Unexploited_Profit": self.compute_unexploited_profit,
        }
        out = {}
        for m in case_metrics_list:
            if m not in registry:
                raise ValueError(f"Metric '{m}' is not available. Choose from {list(registry)}.")
            out[m] = registry[m]()
        return out

    def compute_all_metrics(self):
        return self.get_metrics(self.AVAILABLE)


class HealthCaseMetrics:
    AVAILABLE = (
        "Percentage_treated",
        "Avg_outcome_difference",
        "Total_cognitive_score",
        "Mean_outcome_treated",
        "Mean_outcome_control",
        "Cost_effectiveness",
    )

    def __init__(self, suggestions_df, decision_col, true_outcome_col, cfg):
        if decision_col not in suggestions_df.columns or true_outcome_col not in suggestions_df.columns:
            raise ValueError(f"Columns {decision_col} or {true_outcome_col} not found in the DataFrame")
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.true_outcome_col = true_outcome_col
        self.cfg = cfg

    def compute_percentage_treated(self):
        df = self.suggestions_df
        return len(df[df[self.decision_col] == "A"]) / len(df)

    def compute_avg_outcome_difference(self):
        df = self.suggestions_df
        treated = df[df[self.decision_col] == "A"]["A_outcome"].mean()
        control = df[df[self.decision_col] == "C"]["C_outcome"].mean()
        if pd.notna(treated) and pd.notna(control):
            return treated - control
        return 0

    def compute_total_cognitive_score(self):
        df = self.suggestions_df
        if "A_outcome" not in df.columns or "C_outcome" not in df.columns:
            raise ValueError("Outcome columns (A_outcome, C_outcome) not found in the DataFrame")
        total_treated = df[df[self.decision_col] == "A"]["A_outcome"].sum()
        total_control = df[df[self.decision_col] == "C"]["C_outcome"].sum()
        return total_treated + total_control

    def compute_mean_outcome(self):
        total_treated, total_control = 0.0, 0.0
        for decision, group in self.suggestions_df.groupby(self.decision_col):
            if decision == "A":
                total_treated += group["A_outcome"].mean()
            elif decision == "C":
                total_control += group["C_outcome"].mean()
        return total_treated, total_control

    def compute_total_cost_effectiveness(self):
        df = self.suggestions_df
        required = ["A_outcome", "C_outcome", self.decision_col]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column {col} not found in the DataFrame.")

        costs = {
            "A": self.cfg.reward_calculator.base_cost.get("A", 100),
            "C": self.cfg.reward_calculator.base_cost.get("C", 10),
        }
        treated = df[df[self.decision_col] == "A"]
        improvement = (
            treated["A_outcome"].mean() - df["C_outcome"].mean() if not treated.empty else 0
        )
        return improvement / (costs["A"] - costs["C"])

    def get_metrics(self, case_metrics_list):
        registry = {
            "Percentage_treated": self.compute_percentage_treated,
            "Avg_outcome_difference": self.compute_avg_outcome_difference,
            "Total_cognitive_score": self.compute_total_cognitive_score,
            "Mean_outcome_treated": lambda: self.compute_mean_outcome()[0],
            "Mean_outcome_control": lambda: self.compute_mean_outcome()[1],
            "Cost_effectiveness": self.compute_total_cost_effectiveness,
        }
        out = {}
        for m in case_metrics_list:
            if m not in registry:
                raise ValueError(f"Metric '{m}' is not available. Choose from {list(registry)}.")
            out[m] = registry[m]()
        return out

    def compute_all_metrics(self):
        return self.get_metrics(self.AVAILABLE)


# Registry — adding a new use case is now a single line.
CASE_METRICS_REGISTRY = {
    "classification": LendingCaseMetrics,
    "causal_regression": HealthCaseMetrics,
}
