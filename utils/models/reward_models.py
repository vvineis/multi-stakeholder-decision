"""Reward models: per-actor regressors wrapped behind a unified interface."""
from sklearn.base import clone
from sklearn.metrics import mean_squared_error


class RewardModels:
    """Train + evaluate one regressor per reward type."""

    def __init__(self, regressor_template, reward_types, **regressor_params):
        self.reward_types = list(reward_types)
        self.regressors = {
            rt: clone(regressor_template).set_params(**regressor_params)
            for rt in self.reward_types
        }

    def train(self, X_train, y_train_rewards):
        X_train.columns = X_train.columns.astype(str)
        trained = {}
        for rt, reg in self.regressors.items():
            reg.fit(X_train, y_train_rewards[rt])
            trained[rt] = reg
        return trained

    def evaluate(self, X_val, y_val_rewards):
        X_val.columns = X_val.columns.astype(str)
        scores = {}
        for rt, reg in self.regressors.items():
            y_pred = reg.predict(X_val)
            scores[f"{rt}_mse"] = mean_squared_error(y_val_rewards[rt], y_pred)
        print(scores)
        return scores

    def average_mse(self, scores):
        """Convenience: mean MSE across reward types from `evaluate`'s output."""
        return sum(scores[f"{rt}_mse"] for rt in self.reward_types) / len(self.reward_types)
