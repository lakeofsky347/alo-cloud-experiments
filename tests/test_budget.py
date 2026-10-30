"""Synthetic amounts exercise reservations only; no account balances or prices."""
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
    with pytest.raises(BudgetStopped, match="model_limit"):
        value.reserve("request-2", "one", 54, payload_sha256="b" * 64)


def test_broken_usage_bound_is_retained_and_halts(tmp_path):
    value = ledger(tmp_path / "budget.sqlite3")
    value.reserve("request-1", "one", 20, payload_sha256="a" * 64)
    with pytest.raises(BudgetStopped, match="exceeded_frozen_bound"):
        value.settle_known("request-1", 25)
    assert value.snapshot()["committed_or_reserved_microcny"]["one"] == 65
    assert value.snapshot()["unresolved_request_ids"] == ["request-1"]
