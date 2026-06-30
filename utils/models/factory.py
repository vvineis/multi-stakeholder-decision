"""
Centralized factory for outcome and reward models.

Previously, model instantiation logic (the `if "classifier" in cfg.models.outcome:
... elif "learner" in cfg.models.outcome: ...` branching and `get_model_class`
dynamic import helper) was duplicated in `cross_validation_process.py` and
`final_evaluation.py`. This module is the single source of truth.
"""
from importlib import import_module
from hydra.utils import instantiate

from utils.models.reward_models import RewardModels


def get_outcome_model_class(class_name):
    """Dynamically import a class from utils.models.outcome_model."""
    module = import_module("utils.models.outcome_model")
    return getattr(module, class_name)


class ModelFactory:
    """
    Creates fresh, untrained model objects from a Hydra config block.

    Usage:
        factory = ModelFactory(cfg)
        outcome = factory.new_outcome_model()
        reward  = factory.new_reward_models(**reward_hparams)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.outcome_cfg = cfg.models.outcome
        self.reward_cfg = cfg.models.rewards
        self.reward_types = cfg.actors.reward_types
        self._outcome_class = get_outcome_model_class(self.outcome_cfg.model_class)

    @property
    def model_type(self) -> str:
        """'classification' or 'causal_regression'."""
        return self.outcome_cfg.model_type

    @property
    def has_treatment(self) -> bool:
        return self.model_type == "causal_regression"

    def new_outcome_model(self):
        """Instantiate a fresh OutcomeModel / CausalOutcomeModel using cfg.seed."""
        seed = int(self.cfg.get("seed", 111))
        if "learner" in self.outcome_cfg:
            learner = instantiate(self.outcome_cfg.learner)
            return self._outcome_class(learner=learner, random_state=seed)
        if "classifier" in self.outcome_cfg:
            classifier = instantiate(self.outcome_cfg.classifier)
            return self._outcome_class(
                classifier=classifier,
                smote_random_state=seed,
                model_random_state=seed,
            )
        raise ValueError(
            "cfg.models.outcome must contain either a 'learner' or a 'classifier' block."
        )

    def new_reward_models(self, **hparams) -> RewardModels:
        """Instantiate a fresh RewardModels wrapper with the given hyperparameters."""
        regressor = instantiate(self.reward_cfg.regressor)
        return RewardModels(regressor, self.reward_types, **hparams)

    @property
    def outcome_param_grid(self) -> dict:
        return dict(self.outcome_cfg.param_grid)

    @property
    def reward_param_grid(self) -> dict:
        return dict(self.reward_cfg.param_grid)
