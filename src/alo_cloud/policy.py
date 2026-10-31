"""Versioned constrained Thompson sampling for the revised development protocol.

Final public workspace quality is a bounded continuous observation. Its Beta
pseudo-posterior is an explicitly approximate quality predictor, separate from
the native Bernoulli B/TMS stores. Resource predictions only remove infeasible
actions; there is no fixed quality/cost/time weighted reward. The service and
gateway still enforce actual hard limits before every side effect/request.
"""
from __future__ import annotations

import math
import random
from statistics import fmean, pstdev

from .errors import PilotStopped


POLICY_VERSION = "constrained-ts-quality-v1"
RESOURCE_FIELDS = ("requests", "tokens", "seconds", "cny")
UNIT_LOCAL_FIELDS = frozenset({"requests", "tokens", "seconds"})
DEFAULT_PRIOR = {"requests": 2.0, "tokens": 8192.0, "seconds": 90.0, "cny": 0.1}


def _number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


class ConstrainedThompsonPolicy:
    """Maximize sampled final quality within predicted remaining resources.

    Predictions are sample mean plus one population standard deviation, times
    the frozen safety multiplier. With no measurements the per-model prior is
    used. ``seconds`` MUST include public verifier time in adapter observations.
    These predictions are heuristics, with no chance-constraint/theorem claim.
    ``budget_remaining(model)`` is read on every selection, never cached.
    """

    def __init__(self, observations, budget_remaining, *, priors=None,
                 safety_multiplier=1.25, learn_quality=True):
        if not _number(safety_multiplier) or safety_multiplier < 1:
            raise PilotStopped("invalid_policy_safety_multiplier")
        self.observations = observations
        self.budget_remaining = budget_remaining
        self.priors = dict(priors or {})
        self.safety_multiplier = float(safety_multiplier)
        self.learn_quality = bool(learn_quality)
        for prior in self.priors.values():
            if (not isinstance(prior, dict) or set(prior) != set(RESOURCE_FIELDS)
                    or any(not _number(v) for v in prior.values())):
                raise PilotStopped("invalid_model_resource_prior")

    def resource_estimate(self, model):
        prior = self.priors.get(model, DEFAULT_PRIOR)
        rows = self.observations.get(model, [])
        result = {}
        for resource in RESOURCE_FIELDS:
            key = "cost_cny" if resource == "cny" else resource
            values = [float(row[key]) for row in rows if _number(row.get(key))]
            base = fmean(values) + pstdev(values) if values else float(prior[resource])
            # A worker action needs at least one request even after zero-cost
            # scripted controls. Scripted control data must not be imported.
            result[resource] = max(1.0 if resource == "requests" else 0.0,
                                   base * self.safety_multiplier)
        return result

    def select(self, org, pool, worker_ids, cell, topics, user_id, rng=None):
        rng = rng or random.Random(0)
        if not worker_ids or len(set(worker_ids)) != len(worker_ids):
            raise PilotStopped("empty_or_duplicate_policy_candidates")
        scores, feasible = {}, []
        for model in worker_ids:
            remaining = self.budget_remaining(model)
            if (not isinstance(remaining, dict)
                    or any(not _number(remaining.get(key)) for key in RESOURCE_FIELDS)):
                raise PilotStopped("unknown_remaining_budget")
            estimate = self.resource_estimate(model)
            violated = [key for key in RESOURCE_FIELDS if estimate[key] > remaining[key]]
            rows = self.observations.get(model, []) if self.learn_quality else []
            qualities = [float(row.get("quality_after", row.get("public_quality")))
                         for row in rows
                         if _number(row.get("quality_after", row.get("public_quality")))
                         and row.get("quality_after", row.get("public_quality")) <= 1]
            if qualities:
                alpha, beta = 1 + sum(qualities), 1 + len(qualities) - sum(qualities)
                source = "bounded_global_quality_beta_pseudo_posterior"
            else:
                belief = org.belief.query(model, cell)
                alpha, beta, source = belief.alpha, belief.beta, "native_local_bernoulli_prior"
            draw = rng.betavariate(alpha, beta) if not violated else None
            scores[model] = {"quality_draw": draw, "quality_alpha": alpha, "quality_beta": beta,
                             "quality_source": source, "quality_observations": len(qualities),
                             "resource_prediction": estimate, "remaining": dict(remaining),
                             "feasible": not violated, "violated_resources": violated}
            if not violated:
                feasible.append(model)
        if not feasible:
            # Requests, tokens and seconds are the unit's shared caps. CNY is the
            # model and study total. A unit that has spent its own allowance must
            # not stop the other units.
            if all(set(scores[model]["violated_resources"]) & UNIT_LOCAL_FIELDS
                   for model in worker_ids):
                raise PilotStopped("unit_resource_exhausted")
            raise PilotStopped("budget_infeasible_all_models")
        # Stable pool order is the preregistered exact-tie rule.
        selected = max(feasible, key=lambda model: scores[model]["quality_draw"])
        return selected, {"schema": POLICY_VERSION, "selected": selected, "scores": scores,
                          "eligible": list(worker_ids), "objective": "maximize_sampled_final_public_quality",
                          "resource_role": "hard_prediction_feasibility_filter",
                          "actual_hard_limits": "services_and_gateway",
                          "safety_multiplier": self.safety_multiplier}
