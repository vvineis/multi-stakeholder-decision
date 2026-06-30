"""Standard predictive-performance metrics."""
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class StandardMetrics:
    def __init__(
        self,
        suggestions_df,
        decision_col,
        true_outcome_col=None,
        actions_set=None,
        outcomes_set=None,
        causal_reg_outcome_cols=None,
        model_type=None,
    ):
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.true_outcome_col = true_outcome_col
        self.model_type = model_type
        self.a_outcome_col = causal_reg_outcome_cols[0] if causal_reg_outcome_cols else None
        self.c_outcome_col = causal_reg_outcome_cols[1] if causal_reg_outcome_cols else None
        self.decision_mapping = {d: i for i, d in enumerate(actions_set)} if actions_set else None
        self.outcome_mapping = {o: i for i, o in enumerate(outcomes_set)} if outcomes_set else None
        self._y_pred = self._y_true = self._is_correct = None

    @property
    def y_pred(self):
        if self._y_pred is None and self.decision_mapping:
            self._y_pred = self.suggestions_df[self.decision_col].map(self.decision_mapping).astype(int)
        return self._y_pred

    @property
    def y_true(self):
        if self._y_true is None and self.outcome_mapping:
            self._y_true = self.suggestions_df[self.true_outcome_col].map(self.outcome_mapping).astype(int)
        return self._y_true

    @property
    def is_correct(self):
        if self._is_correct is None and self.a_outcome_col and self.c_outcome_col:
            df = self.suggestions_df
            self._is_correct = df.apply(
                lambda r: (
                    (r[self.decision_col] == "A" and r[self.a_outcome_col] >= r[self.c_outcome_col])
                    or (r[self.decision_col] == "C" and r[self.c_outcome_col] >= r[self.a_outcome_col])
                ),
                axis=1,
            )
        return self._is_correct

    def get_metrics(self, metric_list):
        if self.model_type == "classification":
            registry = {
                "Precision": lambda: precision_score(self.y_true, self.y_pred, average="macro", zero_division=0),
                "Recall": lambda: recall_score(self.y_true, self.y_pred, average="macro", zero_division=0),
                "F1 Score": lambda: f1_score(self.y_true, self.y_pred, average="macro", zero_division=0),
                "Accuracy": lambda: accuracy_score(self.y_true, self.y_pred),
            }
            return {m: registry[m]() for m in metric_list if m in registry}

        if self.model_type == "causal_regression":
            regret = self.suggestions_df.apply(
                lambda r: max(r[self.a_outcome_col], r[self.c_outcome_col])
                - (r[self.a_outcome_col] if r[self.decision_col] == "A" else r[self.c_outcome_col]),
                axis=1,
            )
            return {"Accuracy": self.is_correct.mean(), "Mean_Regret": regret.mean()}

        return {}
