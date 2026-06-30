"""
Solution strategies for the compromise/decision step.

All strategies now share the same return contract: a `(best_action, value)` tuple.
`MaxIndividualReward` keeps its per-actor mapping, since it serves a different role
(suggesting actions for individual actors rather than picking a single group action).
"""
from abc import ABC, abstractmethod
from math import prod
from typing import Mapping, Sequence, Tuple

import numpy as np


ActorRewards = Mapping[str, Mapping[str, float]]   # actor -> {action -> reward}
DisagreementPoint = Mapping[str, float]
IdealPoint = Mapping[str, float]


class SolutionStrategy(ABC):
    """Returns (best_action, value)."""

    @abstractmethod
    def compute(
        self,
        expected_rewards: ActorRewards,
        disagreement_point: DisagreementPoint,
        ideal_point: IdealPoint,
        all_actions: Sequence[str],
    ) -> Tuple[str, float]:
        ...


class MaxIndividualReward:
    """Per-actor best action — does NOT follow the SolutionStrategy contract."""

    def compute(self, expected_rewards, disagreement_point=None, ideal_point=None, all_actions=None):
        return {
            actor: {"action": max(rewards, key=rewards.get), "value": max(rewards.values())}
            for actor, rewards in expected_rewards.items()
        }


class MaximinCriterion(SolutionStrategy):
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions):
        min_rewards = {
            action: min(actor_rewards.get(action, 0) for actor_rewards in expected_rewards.values())
            for action in all_actions
        }
        best = max(min_rewards, key=min_rewards.get)
        return best, min_rewards[best]


class KalaiSmorodinsky(SolutionStrategy):
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions):
        gains = {
            action: min(
                (rewards.get(action, disagreement_point[actor]) - disagreement_point[actor])
                / (ideal_point[actor] - disagreement_point[actor])
                for actor, rewards in expected_rewards.items()
                if ideal_point[actor] != disagreement_point[actor]
            )
            for action in all_actions
        }
        best = max(gains, key=gains.get)
        return best, gains[best]


class NashBargainingSolution(SolutionStrategy):
    """Full agreement -> partial-majority agreement -> fall back to `default_action`.

    Previously the fallback was hardcoded to 'C', which was correct for the
    health use case but meaningless for lending. Now `default_action` is
    explicit; if it's not in `all_actions` we use the lexicographically first
    available action as a last-resort safe choice.
    """

    def __init__(self, default_action: str = "C"):
        self.default_action = default_action

    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions):
        def product(action, predicate):
            if not predicate(action):
                return None
            return prod(
                max(0, expected_rewards[actor].get(action, disagreement_point[actor])
                    - disagreement_point[actor])
                for actor in expected_rewards
            )

        def full_ok(action):
            return all(
                expected_rewards[actor].get(action, disagreement_point[actor]) > disagreement_point[actor]
                for actor in expected_rewards
            )

        full = {a: v for a in all_actions if (v := product(a, full_ok)) is not None}
        if full:
            best = max(full, key=full.get)
            return best, full[best]

        n = len(expected_rewards)
        threshold = n // 2 + 1

        def partial_ok(action):
            return sum(
                expected_rewards[actor].get(action, disagreement_point[actor]) > disagreement_point[actor]
                for actor in expected_rewards
            ) >= threshold

        partial = {a: v for a in all_actions if (v := product(a, partial_ok)) is not None}
        if partial:
            best = max(partial, key=partial.get)
            return best, partial[best]

        fallback = self.default_action if self.default_action in all_actions else sorted(all_actions)[0]
        print(f"No actions meet full or partial agreement thresholds. Using {fallback} as the default action.")
        return fallback, 0


class NashSocialWelfare(SolutionStrategy):
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions, epsilon=1e-6):
        products = {
            action: prod(
                max(epsilon, actor_rewards.get(action, epsilon))
                for actor_rewards in expected_rewards.values()
            )
            for action in all_actions
        }
        best = max(products, key=products.get)
        return best, products[best]


class CompromiseProgramming(SolutionStrategy):
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions, p: int = 2):
        distances = {
            action: sum(
                abs(ideal_point[actor] - expected_rewards[actor].get(action, ideal_point[actor])) ** p
                for actor in expected_rewards
            ) ** (1 / p)
            for action in all_actions
        }
        best = min(distances, key=distances.get)
        return best, distances[best]


class ProportionalFairness(SolutionStrategy):
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions, epsilon=1e-6):
        sums = {
            action: sum(
                np.log(max(epsilon, actor_rewards.get(action, epsilon)))
                for actor_rewards in expected_rewards.values()
            )
            for action in all_actions
        }
        best = max(sums, key=sums.get)
        return best, sums[best]


# Registry -- make adding a new strategy a single-line change.
DEFAULT_STRATEGIES = {
    "Maximin": MaximinCriterion(),
    "Kalai-Smorodinsky": KalaiSmorodinsky(),
    "Nash Bargaining": NashBargainingSolution(),     # backwards-compat default
    "Nash Social Welfare": NashSocialWelfare(),
    "Compromise Programming": CompromiseProgramming(),
    "Proportional Fairness": ProportionalFairness(),
}


def build_strategies(cfg=None) -> dict:
    """
    Build a strategy registry that is aware of the use-case's action space.

    For Nash Bargaining, the no-agreement fallback action is set to the first
    *non-positive* action in `cfg.actions_outcomes.actions_set` (e.g. 'Not_Grant'
    for lending, 'C' for health). Falls back to the legacy 'C' if `cfg` is None.
    """
    if cfg is None:
        nb_default = "C"
    else:
        actions = list(cfg.actions_outcomes.actions_set)
        positive = set(cfg.actions_outcomes.positive_actions_set)
        non_positive = [a for a in actions if a not in positive]
        nb_default = non_positive[0] if non_positive else actions[0]

    return {
        "Maximin": MaximinCriterion(),
        "Kalai-Smorodinsky": KalaiSmorodinsky(),
        "Nash Bargaining": NashBargainingSolution(default_action=nb_default),
        "Nash Social Welfare": NashSocialWelfare(),
        "Compromise Programming": CompromiseProgramming(),
        "Proportional Fairness": ProportionalFairness(),
    }


class SuggestAction:
    """Apply every strategy in the registry to a single feature row's expected rewards."""

    def __init__(self, expected_rewards: ActorRewards, strategies: dict | None = None):
        self.expected_rewards = expected_rewards
        self.disagreement_point = {
            actor: min(r.values()) for actor, r in expected_rewards.items()
        }
        self.ideal_point = {
            actor: max(r.values()) for actor, r in expected_rewards.items()
        }
        self.all_actions = {a for r in expected_rewards.values() for a in r}
        self.strategies = strategies if strategies is not None else DEFAULT_STRATEGIES

    def compute_all_compromise_solutions(self):
        results = {}
        for name, strategy in self.strategies.items():
            action, value = strategy.compute(
                self.expected_rewards,
                self.disagreement_point,
                self.ideal_point,
                self.all_actions,
            )
            results[name] = {"action": action, "value": value}
        return results
