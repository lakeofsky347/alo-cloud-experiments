"""Synthetic amounts exercise reservations only; no account balances or prices."""
import sqlite3

import pytest

from alo_cloud.budget import BudgetLedger, BudgetStopped


def ledger(path):
    return BudgetLedger.create(path, model_limits_microcny={"one": 100, "two": 100},
        opening_commitments_microcny={"one": 40, "two": 0}, max_requests=5, prior_requests=2)


def test_opening_usage_required_and_missing_state_never_recreated(tmp_path):
    path = tmp_path / "budget.sqlite3"
    with pytest.raises(ValueError, match="complete_opening"):
        BudgetLedger.create(path, model_limits_microcny={"one": 100}, opening_commitments_microcny={}, max_requests=3, prior_requests=0)
    with pytest.raises(BudgetStopped, match="existing_regular"):
        BudgetLedger(path)
    assert not path.exists()


def test_unknown_reservation_survives_reopen_and_blocks_other_model(tmp_path):
    path = tmp_path / "budget.sqlite3"
    first = ledger(path)
    first.reserve("request-1", "one", 20, payload_sha256="a" * 64)
    first.mark_unknown("request-1")
    restored = BudgetLedger(path)
    assert restored.snapshot()["committed_or_reserved_microcny"]["one"] == 60
    assert restored.snapshot()["requests"] == 3
    with pytest.raises(BudgetStopped, match="unresolved"):
        restored.reserve("request-2", "two", 10, payload_sha256="b" * 64)
    with pytest.raises(FileExistsError):
        ledger(path)


def test_known_usage_releases_only_its_excess_and_never_replays(tmp_path):
    value = ledger(tmp_path / "budget.sqlite3")
    assert value.reserve("request-1", "one", 20, payload_sha256="a" * 64)["is_execution_permit"] is False
    value.settle_known("request-1", 7)
    assert value.snapshot()["remaining_microcny"]["one"] == 53
    with pytest.raises(BudgetStopped, match="identity_consumed"):
        value.reserve("request-1", "one", 20, payload_sha256="a" * 64)
    with pytest.raises(BudgetStopped, match="model_cny_cap_exhausted"):
        value.reserve("request-2", "one", 54, payload_sha256="b" * 64)


def test_remaining_resources_match_policy_dimensions_until_unknown(tmp_path):
    path = tmp_path / "budget.sqlite3"
    value = BudgetLedger.create(
        path, model_limits_microcny={"one": 100, "two": 100},
        opening_commitments_microcny={"one": 40, "two": 0}, max_requests=5, prior_requests=2,
        max_tokens=1000, prior_tokens=10, max_seconds_ms=5000, prior_seconds_ms=1000,
        max_tools=3, prior_tools=1)
    one = value.remaining_resources("one")
    two = value.remaining_resources("two")
    assert set(one) == {"requests", "tokens", "seconds", "cny"}
    assert one["requests"] == two["requests"] == 3
    assert one["tokens"] == two["tokens"] == 990
    assert one["seconds"] == two["seconds"] == 4
    assert one["cny"] == 60 / 1_000_000
    assert two["cny"] == 100 / 1_000_000
    value.reserve("request-1", "one", 20, payload_sha256="a" * 64, tokens=30, seconds_ms=500)
    value.settle_known("request-1", 7, tokens=5, seconds_ms=10)
    settled = value.remaining_resources("one")
    assert settled["requests"] == 2
    assert settled["tokens"] == 985
    assert settled["seconds"] == (5000 - 1010) / 1000
    assert settled["cny"] == 53 / 1_000_000
    value.reserve("request-2", "two", 10, payload_sha256="b" * 64, tokens=4, seconds_ms=20)
    value.mark_unknown("request-2")
    with pytest.raises(BudgetStopped, match="unresolved"):
        value.remaining_resources("one")
    with pytest.raises(BudgetStopped, match="unresolved"):
        value.reserve_tool("tool-1", seconds_ms=10)


def test_legacy_ledger_does_not_invent_missing_dimensions(tmp_path):
    value = ledger(tmp_path / "budget.sqlite3")
    with pytest.raises(BudgetStopped, match="resource_dimensions_unconfigured"):
        value.remaining_resources("one")


def test_v1_schema_is_refused(tmp_path):
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema', 'alo-local-budget-scaffold-v1')")
    conn.commit()
    conn.close()
    with pytest.raises(BudgetStopped, match="unsupported_budget_schema"):
        BudgetLedger(path)


def test_broken_usage_bound_is_retained_and_halts(tmp_path):
    value = ledger(tmp_path / "budget.sqlite3")
    value.reserve("request-1", "one", 20, payload_sha256="a" * 64)
    with pytest.raises(BudgetStopped, match="exceeded_frozen_bound"):
        value.settle_known("request-1", 25)
    assert value.snapshot()["committed_or_reserved_microcny"]["one"] == 65
    assert value.snapshot()["unresolved_request_ids"] == ["request-1"]
