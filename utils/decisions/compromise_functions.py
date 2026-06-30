"""
Back-compat shim.

The compromise strategies were moved to `utils.decisions.strategies`. Importing them
from here continues to work so older notebooks or downstream scripts don't break.
"""
from utils.decisions.strategies import (  # noqa: F401
    SolutionStrategy,
    MaxIndividualReward,
    MaximinCriterion,
    KalaiSmorodinsky,
    NashBargainingSolution,
    NashSocialWelfare,
    CompromiseProgramming,
    ProportionalFairness,
    SuggestAction,
)
