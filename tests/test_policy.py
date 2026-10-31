"""Offline constrained-Thompson checks. No prices, ledgers or model calls."""
import random

import pytest

from alo.org.organization import Organization
from alo_cloud.errors import PilotStopped
from alo_cloud.policy import DEFAULT_PRIOR, RESOURCE_FIELDS, ConstrainedThompsonPolicy


CELL = "type=generic|complexity=na|uncertainty=na"


def remaining_for(poor=False):
    def remaining(model):
        budget = {"requests": 100, "tokens": 100000, "seconds": 1000, "cny": 0 if model == "poor" else 50}
        if poor and model == "poor":
            return budget
        return budget
    return remaining


def test_resource_fields_are_the_four_prediction_dimensions():
    assert RESOURCE_FIELDS == ("requests", "tokens", "seconds", "cny")
    assert set(DEFAULT_PRIOR) == set(RESOURCE_FIELDS)


def test_estimate_uses_prior_and_request_floor_without_observations():
    policy = ConstrainedThompsonPolicy({}, remaining_for())
    estimate = policy.resource_estimate("mimo-v2.6-flash")
    assert estimate["requests"] == max(1.0, DEFAULT_PRIOR["requests"] * 1.25)
    assert estimate["cny"] == DEFAULT_PRIOR["cny"] * 1.25
    assert estimate["tokens"] == DEFAULT_PRIOR["tokens"] * 1.25
    assert estimate["seconds"] == DEFAULT_PRIOR["seconds"] * 1.25


def test_select_rejects_unknown_budget_and_infeasible_pool():
    org = Organization(":memory:")
    policy = ConstrainedThompsonPolicy({}, lambda model: {"requests": 1})
    with pytest.raises(PilotStopped, match="unknown_remaining_budget"):
        policy.select(org, {}, ["one"], CELL, (), "user", random.Random(0))
    blocked = ConstrainedThompsonPolicy({}, lambda model: {key: 0 for key in RESOURCE_FIELDS})
    with pytest.raises(PilotStopped, match="budget_infeasible_all_models"):
        blocked.select(org, {}, ["one", "two"], CELL, (), "user", random.Random(0))
    org.conn.close()


def test_select_skips_model_whose_predicted_cny_exceeds_remaining():
    org = Organization(":memory:")
    policy = ConstrainedThompsonPolicy({}, remaining_for(poor=True))
    selected, trace = policy.select(org, {}, ["poor", "rich"], CELL, (), "user", random.Random(0))
    assert selected == "rich"
    assert trace["scores"]["poor"]["feasible"] is False
    assert "cny" in trace["scores"]["poor"]["violated_resources"]
    assert trace["scores"]["rich"]["feasible"] is True
    org.conn.close()


def test_invalid_prior_and_safety_multiplier_fail_closed():
    with pytest.raises(PilotStopped, match="invalid_policy_safety_multiplier"):
        ConstrainedThompsonPolicy({}, remaining_for(), safety_multiplier=0.5)
    with pytest.raises(PilotStopped, match="invalid_model_resource_prior"):
        ConstrainedThompsonPolicy({}, remaining_for(), priors={"one": {"requests": 1}})
