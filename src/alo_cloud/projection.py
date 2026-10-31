"""Projected paid-run cost from the frozen call caps and the price table.

Flexible G2/G3 calls are priced at the most expensive verified model, so the
total is an upper bound on the token estimate rather than a routed forecast.
The estimate itself is cache-miss input plus output; a real cache hit costs less.
"""
from __future__ import annotations

from .contracts import MODEL_POOL, STUDY_TOTAL_LIMIT_MICROCNY, PER_MODEL_LIMIT_MICROCNY
from .freeze import TOKEN_ESTIMATE, iter_units
from .prices import charge_microcny


PINNED_ARM = {"g1-main": "mimo-v2.6-flash", "g1-strong": "mimo-v2.6-pro"}
HEADROOM_NUMERATOR = 4
HEADROOM_DENOMINATOR = 5


def call_microcny(model):
    return charge_microcny(model, dict(TOKEN_ESTIMATE))


def project():
    units = tuple(iter_units("m1")) + tuple(iter_units("m2"))
    prices = {model: call_microcny(model) for model in MODEL_POOL}
    worst = max(prices.values())
    pinned_calls = {model: 0 for model in MODEL_POOL}
    flexible_calls = 0
    total_upper = 0
    by_study = {"m1": 0, "m2": 0}
    for unit in units:
        if unit.arm in PINNED_ARM:
            model = PINNED_ARM[unit.arm]
            pinned_calls[model] += unit.request_cap
            spend = unit.request_cap * prices[model]
        else:
            flexible_calls += unit.request_cap
            spend = unit.request_cap * worst
        total_upper += spend
        by_study[unit.study] += spend
    pessimistic = {}
    for model in MODEL_POOL:
        pessimistic[model] = (pinned_calls[model] + flexible_calls) * prices[model]
    limit = STUDY_TOTAL_LIMIT_MICROCNY
    return {
        "units": len(units),
        "m1_units": sum(1 for unit in units if unit.study == "m1"),
        "m2_units": sum(1 for unit in units if unit.study == "m2"),
        "token_estimate": dict(TOKEN_ESTIMATE),
        "call_microcny": prices,
        "pinned_calls": pinned_calls,
        "flexible_calls": flexible_calls,
        "pessimistic_model_microcny": pessimistic,
        "total_upper_microcny": total_upper,
        "by_study_microcny": by_study,
        "total_limit_microcny": limit,
        "per_model_limit_microcny": PER_MODEL_LIMIT_MICROCNY,
        "headroom_microcny": limit - total_upper,
        "headroom_ratio": (limit - total_upper) / limit,
    }


def within_budget(summary=None):
    """True when the routed upper bound keeps at least 20% of the 200 CNY total.

    The single-model stress case can exceed one model's 50 CNY hard cap once
    prompts match the observed live size. The ledger still stops that model at
    50 CNY; this check is the schedule-wide headroom.
    """
    summary = project() if summary is None else summary
    total_ceiling = STUDY_TOTAL_LIMIT_MICROCNY * HEADROOM_NUMERATOR // HEADROOM_DENOMINATOR
    return summary["total_upper_microcny"] <= total_ceiling
