"""Group fairness metrics."""
import numpy as np


class FairnessMetrics:
    def __init__(self, cfg, suggestions_df, decision_col, outcome_col="True Outcome"):
        if decision_col not in suggestions_df.columns:
            raise ValueError(f"Column {decision_col} not found")
        if outcome_col not in suggestions_df.columns:
            raise ValueError(f"Column {outcome_col} not found")

        self.cfg = cfg
        self.df = suggestions_df
        self.decision_col = decision_col
        self.outcome_col = outcome_col

        self.group_col = cfg.case_specific_metrics.positive_attribute_for_fairness
        self.positive_group_value = cfg.case_specific_metrics.positive_group_value
        self.actions_set = list(cfg.actions_outcomes.actions_set)
        self.positive_actions_set = list(cfg.actions_outcomes.positive_actions_set)
        self.outcomes_set = list(cfg.actions_outcomes.outcomes_set)
        self.positive_outcomes_set = list(cfg.actions_outcomes.positive_outcomes_set)

    def get_metrics(self, fairness_metrics_list):
        registry = {
            "Demographic_Parity": self._dp,
            "Equal_Opportunity": self._eo,
            "Conditional_Outcome_Parity": self._cop,
        }
        out = {}
        for m in fairness_metrics_list:
            if m not in registry:
                raise ValueError(f"Metric '{m}' is not available. Choose from {list(registry)}.")
            out[m] = registry[m]()
        return out

    # ------------------------------------------------------------------
    def _pos_decision(self) -> str:
        return self.positive_actions_set[0]

    def _group_rates(self, df_pos, df_neg, decision=None):
        """Return P(decision = pos) for both groups."""
        dec = decision if decision is not None else self._pos_decision()
        pos = (df_pos[self.decision_col] == dec).mean() if len(df_pos) else np.nan
        neg = (df_neg[self.decision_col] == dec).mean() if len(df_neg) else np.nan
        return pos, neg

    def _dp(self):
        df_pos = self.df[self.df[self.group_col] == self.positive_group_value]
        df_neg = self.df[self.df[self.group_col] != self.positive_group_value]
        p, n = self._group_rates(df_pos, df_neg)
        return p - n

    def _eo(self):
        pos_outcome = self.positive_outcomes_set[0]
        df_pos = self.df[
            (self.df[self.group_col] == self.positive_group_value)
            & (self.df[self.outcome_col] == pos_outcome)
        ]
        df_neg = self.df[
            (self.df[self.group_col] != self.positive_group_value)
            & (self.df[self.outcome_col] == pos_outcome)
        ]
        p, n = self._group_rates(df_pos, df_neg)
        return p - n

    def _cop(self):
        pos_decision = self._pos_decision()
        pos_outcome = self.positive_outcomes_set[0]
        df_pred_pos = self.df[self.df[self.decision_col] == pos_decision]
        df_pos = df_pred_pos[df_pred_pos[self.group_col] == self.positive_group_value]
        df_neg = df_pred_pos[df_pred_pos[self.group_col] != self.positive_group_value]
        pos_rate = (df_pos[self.outcome_col] == pos_outcome).mean() if len(df_pos) else np.nan
        neg_rate = (df_neg[self.outcome_col] == pos_outcome).mean() if len(df_neg) else np.nan
        return pos_rate - neg_rate
