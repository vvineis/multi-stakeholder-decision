"""
OutcomeModel (classification) and CausalOutcomeModel (causal regression).

Behavior is identical to the original framework; only minor cleanups
(removed dead imports, consolidated prints).
"""
import warnings

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from imblearn.over_sampling import SMOTE
from causalml.inference.meta import BaseXRegressor
from xgboost import XGBRegressor


class OutcomeModel:
    """Train and evaluate an outcome-prediction classifier."""

    def __init__(
        self,
        classifier,
        use_smote: bool = True,
        smote_k_neighbors: int = 1,
        smote_random_state: int = 42,
        model_random_state: int = 111,
    ):
        self.classifier = classifier
        self.use_smote = use_smote
        self.smote_k_neighbors = smote_k_neighbors
        self.smote_random_state = smote_random_state
        self.model_random_state = model_random_state

    def train(self, X_train, y_train, **hyperparams):
        if self.use_smote:
            smote = SMOTE(
                k_neighbors=self.smote_k_neighbors,
                random_state=self.smote_random_state,
            )
            X_train, y_train = smote.fit_resample(X_train, y_train)

        if "random_state" in self.classifier.get_params():
            hyperparams["random_state"] = self.model_random_state

        self.classifier.set_params(**hyperparams)
        self.classifier.fit(X_train, y_train)
        return self.classifier

    def evaluate(self, X_val, y_val):
        y_pred = self.classifier.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"Outcome Prediction Accuracy (Validation Set): {acc * 100:.2f}%")
        return acc


class CausalOutcomeModel:
    """Train and evaluate a causal outcome model (BaseXRegressor wrapper)."""

    def __init__(self, learner=None, control_name: str = "C", random_state: int = 4242):
        if learner is None:
            raise ValueError("Learner cannot be None for CausalOutcomeModel.")
        self.learner_class = learner if isinstance(learner, type) else learner.__class__
        self.learner_instance = None
        self.control_name = control_name
        self.random_state = random_state
        self.model = None

    def train(self, X_train, treatment_train, y_train, **hyperparams):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        self.learner_instance = self.learner_class(
            random_state=self.random_state, **hyperparams
        )
        if not isinstance(self.learner_instance, XGBRegressor):
            print(f"Warning: learner is {type(self.learner_instance)}, not XGBRegressor.")

        self.model = BaseXRegressor(
            learner=self.learner_instance, control_name=self.control_name
        )
        self.model.fit(X=X_train, treatment=treatment_train, y=y_train)
        return self.model

    def predict(self, X_test, treatment: str = "A"):
        return self.model.predict(X=X_test, treatment=treatment)

    def predict_outcomes(self, X_test):
        predicted_A, predicted_C = [], []
        for i in range(len(X_test)):
            patient = X_test.iloc[i : i + 1]
            out_c = np.round(self.model.models_mu_c["A"].predict(patient), 1)
            cate = self.model.predict(patient, treatment="A")
            out_a = np.round(out_c + cate, 1)
            predicted_C.append(out_c)
            predicted_A.append(out_a)
        return np.array(predicted_A), np.array(predicted_C)

    def evaluate(self, X_test, treatment_test, y_test):
        predicted_A, predicted_C = self.predict_outcomes(X_test)
        y_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)
        t_arr = (
            treatment_test.values
            if hasattr(treatment_test, "values")
            else np.array(treatment_test)
        )

        treated = t_arr == "A"
        control = t_arr == "C"

        mae_treated = (
            np.mean(np.abs(predicted_A[treated] - y_arr[treated]))
            if treated.any()
            else None
        )
        mae_control = (
            np.mean(np.abs(predicted_C[control] - y_arr[control]))
            if control.any()
            else None
        )

        valid = [m for m in (mae_treated, mae_control) if m is not None]
        avg_mae = float(np.mean(valid)) if valid else None
        print(
            f"MAE treated={mae_treated} control={mae_control} avg={avg_mae}"
        )
        return avg_mae
