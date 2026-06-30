"""Real (ground-truth) payoffs — used only for the classification (lending) case."""


class RealPayoffMetrics:
    def __init__(self, cfg, suggestions_df, decision_col, true_outcome_col, reward_actor, reward_structures):
        if decision_col not in suggestions_df.columns or true_outcome_col not in suggestions_df.columns:
            raise ValueError(f"Columns {decision_col} or {true_outcome_col} not found in the DataFrame")
        self.df = suggestions_df
        self.decision_col = decision_col
        self.true_outcome_col = true_outcome_col
        self.reward_structures = reward_structures
        self.positive_attribute = cfg.case_specific_metrics.positive_attribute_for_fairness
        self.actor = reward_actor

    def _payoff(self, row):
        suggested = row[self.decision_col]
        true_outcome = row[self.true_outcome_col]
        applicant_type = int(row[self.positive_attribute])
        rewards = self.reward_structures[applicant_type]
        if self.actor not in rewards:
            raise KeyError(f"Actor {self.actor} not found for applicant type {applicant_type}")
        return rewards[self.actor].get((suggested, true_outcome), 0.0)

    def compute_total_real_payoff(self):
        return self.df.apply(self._payoff, axis=1).sum()
