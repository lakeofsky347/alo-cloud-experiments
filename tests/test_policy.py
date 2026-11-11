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
    with pytest.raises(PilotStopped, match="unit_resource_exhausted"):
        blocked.select(org, {}, ["one", "two"], CELL, (), "user", random.Random(0))
    money = ConstrainedThompsonPolicy({}, lambda model: {
        "requests": 100, "tokens": 10_000_000, "seconds": 10_000, "cny": 0})
    with pytest.raises(PilotStopped, match="budget_infeasible_all_models"):
        money.select(org, {}, ["one", "two"], CELL, (), "user", random.Random(0))
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


def _row(quality, topics, credited, requests=1):
    return {"quality_after": quality, "public_quality": quality, "topics": list(topics),
            "credited_quality": credited, "requests": requests, "tokens": 10,
            "seconds": 1, "cost_cny": 0.01}


def test_select_uses_topics_and_ignores_uncredited_workspace_quality():
    org = Organization(":memory:")
    observations = {
        "deepseek-flash": [
            _row(1.0, ["coding", "implementation"], 1.0),
            _row(1.0, ["testing", "validation"], None),
        ],
        "mimo-v2.6-flash": [
            _row(0.0, ["coding", "implementation"], 0.0),
            _row(1.0, ["testing", "validation"], 1.0),
        ],
    }
    policy = ConstrainedThompsonPolicy(observations, remaining_for())
    _selected, coding = policy.select(
        org, {}, ["deepseek-flash", "mimo-v2.6-flash"], CELL,
        ("implementation", "coding"), "user", random.Random(0))
    _selected, testing = policy.select(
        org, {}, ["deepseek-flash", "mimo-v2.6-flash"], CELL,
        ("testing", "validation"), "user", random.Random(0))
    assert coding["cell"] == CELL
    assert coding["topics"] == ["coding", "implementation"]
    assert coding["scores"]["deepseek-flash"]["quality_source"] == "role_conditioned_quality_beta_pseudo_posterior"
    assert coding["scores"]["deepseek-flash"]["quality_alpha"] == 2
    assert coding["scores"]["mimo-v2.6-flash"]["quality_beta"] == 2
    assert testing["topics"] == ["testing", "validation"]
    assert testing["scores"]["mimo-v2.6-flash"]["quality_source"] == "role_conditioned_quality_beta_pseudo_posterior"
    assert testing["scores"]["mimo-v2.6-flash"]["quality_alpha"] == 2
    assert testing["scores"]["deepseek-flash"]["quality_source"] == "native_local_bernoulli_prior"
    assert testing["scores"]["deepseek-flash"]["quality_alpha"] == 1
    global_only = ConstrainedThompsonPolicy(
        {"deepseek-flash": [{"quality_after": 1.0, "public_quality": 1.0, "requests": 1,
                             "tokens": 10, "seconds": 1, "cost_cny": 0.01}]},
        remaining_for())
    _selected, trace = global_only.select(
        org, {}, ["deepseek-flash"], CELL, ("coding",), "user", random.Random(1))
    assert trace["scores"]["deepseek-flash"]["quality_source"] == "native_local_bernoulli_prior"
    assert "bounded_global_quality_beta_pseudo_posterior" not in str(trace)
    org.conn.close()
