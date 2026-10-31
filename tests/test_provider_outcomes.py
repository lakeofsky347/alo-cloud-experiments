"""Recorded vendor bodies. No network and no real credentials."""
import json
import sqlite3

import pytest

from alo_cloud.budget import BudgetLedger
from alo_cloud.errors import PilotStopped
from alo_cloud.freeze import select_units
from alo_cloud.ledger import main
from alo_cloud.prices import ceiling_for_messages, charge_microcny
from alo_cloud.run import run_schedule
from tests.test_services import KEYS, build, messages


def _usage(prompt, completion, *, reasoning=None, where="details"):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    if reasoning is None:
        return usage
    if where in {"details", "both"}:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    if where in {"top", "both"}:
        usage["reasoning_tokens"] = reasoning
    return usage


def _body(model, content, usage, *, finish_reason="stop", extra_message=None):
    message = {"content": content}
    if extra_message:
        message.update(extra_message)
    return {"model": model, "choices": [{"message": message, "finish_reason": finish_reason}], "usage": usage}


def _diag(tmp_path, request_id):
    return json.loads((tmp_path / "diagnostics" / f"{request_id}.json").read_text(encoding="utf-8"))


def test_vendor_usage_shapes_and_empty_mimo_completion_are_billed(tmp_path, monkeypatch):
    monkeypatch.delenv("ALO_RUN_DIR", raising=False)
    secret = KEYS["MIMO_API_KEY"]
    mimo = _body(
        "mimo-v2.6-flash", None,
        _usage(10, 4, reasoning=4, where="top"),
        finish_reason="length",
        extra_message={"reasoning_content": "trace " + secret},
    )
    deepseek = _body(
        "deepseek-flash", "answer",
        {**_usage(10, 6, reasoning=2, where="top"),
         "prompt_cache_hit_tokens": 4, "prompt_cache_miss_tokens": 6},
    )
    glm = _body("glm-5.3-flash", "visible", _usage(8, 5, reasoning=3, where="both"))
    services, seen, ledger = build(
        tmp_path, monkeypatch, [mimo, deepseek, glm], charge=charge_microcny, ceiling=1_000_000,
        request_ceiling=ceiling_for_messages)
    empty = services.complete_for_model("mimo-v2.6-flash", messages(), max_tokens=4096)
    assert empty["text"] == ""
    assert empty["length_stop"] is True
    assert empty["reasoning_tokens"] == 4
    assert empty["cost_microcny"] == charge_microcny(
        "mimo-v2.6-flash", {"input_tokens": 10, "output_tokens": 4, "cache_hit_tokens": None})
    assert seen[0]["body"]["max_completion_tokens"] == 8192
    assert "max_tokens" not in seen[0]["body"]
    record = _diag(tmp_path, "req-m1_b0-1")
    assert record["cause"] == "empty_completion"
    assert record["settled"] is True
    assert record["http_status"] == 200
    assert record["finish_reason"] == "length"
    blob = json.dumps(record)
    assert secret not in blob
    assert "[redacted]" in blob
    assert "Authorization" not in blob and "api-key" not in blob
    answered = services.complete_for_model("deepseek-flash", messages("next"))
    assert answered["text"] == "answer"
    assert answered["reasoning_tokens"] == 2
    assert answered["length_stop"] is False
    glm_result = services.complete_for_model("glm-5.3-flash", messages("glm"))
    assert glm_result["reasoning_tokens"] == 3
    assert ledger.snapshot()["unresolved_request_ids"] == []
    reserved = sqlite3.connect(ledger.path).execute(
        "SELECT reserved FROM requests WHERE id='req-m1_b0-1'").fetchone()[0]
    assert reserved == ceiling_for_messages("mimo-v2.6-flash", messages(), 8192)


def test_disagreed_reasoning_counts_stay_unknown(tmp_path, monkeypatch):
    monkeypatch.delenv("ALO_RUN_DIR", raising=False)
    usage = _usage(4, 4, reasoning=1, where="both")
    usage["completion_tokens_details"]["reasoning_tokens"] = 2
    blocked, seen, ledger = build(tmp_path, monkeypatch, [
        _body("deepseek-flash", "x", usage),
        _body("deepseek-flash", "y", _usage(1, 1)),
    ])
    with pytest.raises(PilotStopped, match="provider_outcome_unknown"):
        blocked.complete_for_model("deepseek-flash", messages())
    with pytest.raises(PilotStopped, match="unresolved"):
        blocked.complete_for_model("deepseek-flash", messages())
    assert len(seen) == 1
    record = _diag(tmp_path, "req-m1_b0-1")
    assert record["cause"] == "unparseable_success"
    assert record["settled"] is False
    assert ledger.snapshot()["unresolved_request_ids"] == ["req-m1_b0-1"]


def test_vendor_error_settles_zero_and_transport_failures_hold(tmp_path, monkeypatch):
    monkeypatch.delenv("ALO_RUN_DIR", raising=False)
    secret = KEYS["ZAI_API_KEY"]
    error = ({"error": {"code": "invalid_request", "message": "rejected " + secret}}, 400)
    services, _seen, ledger = build(tmp_path, monkeypatch, [
        error, _body("glm-5.3-flash", "after", _usage(2, 2)),
    ])
    failed = services.complete_for_model("glm-5.3-flash", messages())
    assert failed["vendor_error"] is True
    assert failed["cost_microcny"] == 0
    assert ledger.snapshot()["unresolved_request_ids"] == []
    record = _diag(tmp_path, "req-m1_b0-1")
    assert record["cause"] == "vendor_error"
    assert record["http_status"] == 400
    assert record["vendor_code"] == "invalid_request"
    assert secret not in json.dumps(record)
    assert services.complete_for_model("glm-5.3-flash", messages("next"))["text"] == "after"

    def boom(url, body, headers):
        raise TimeoutError(secret)

    held, _seen, held_ledger = build(tmp_path / "timeout", monkeypatch, [])
    held.transport._sender = boom
    with pytest.raises(PilotStopped, match="provider_outcome_unknown"):
        held.complete_for_model("mimo-v2.6-flash", messages(), max_tokens=128)
    timeout = _diag(tmp_path / "timeout", "req-m1_b0-1")
    assert timeout["cause"] == "timeout"
    assert timeout["response"] is None
    assert secret not in json.dumps(timeout)
    assert held_ledger.snapshot()["unresolved_request_ids"] == ["req-m1_b0-1"]

    reset, _seen, _ledger = build(tmp_path / "reset", monkeypatch, [])
    reset.transport._sender = lambda *args: (_ for _ in ()).throw(ConnectionResetError("reset"))
    with pytest.raises(PilotStopped, match="provider_outcome_unknown"):
        reset.complete_for_model("deepseek-flash", messages())
    assert _diag(tmp_path / "reset", "req-m1_b0-1")["cause"] == "connection_reset"

    opaque, _seen, _ledger = build(tmp_path / "opaque", monkeypatch, [(None, 500)])
    with pytest.raises(PilotStopped, match="provider_outcome_unknown"):
        opaque.complete_for_model("glm-5.3-flash", messages())
    assert _diag(tmp_path / "opaque", "req-m1_b0-1")["cause"] == "transport_error"


def test_resolve_worst_case_and_restart_once(tmp_path, monkeypatch):
    monkeypatch.delenv("ALO_RUN_DIR", raising=False)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    secret = "live-key-must-not-leak"
    monkeypatch.setenv("MIMO_API_KEY", secret)
    unit = select_units("m1", arms=("g1-main",), tasks=("scikit-learn__scikit-learn-13584",))[0]

    def fail(url, body, headers):
        raise TimeoutError(secret)

    with pytest.raises(PilotStopped, match="provider_outcome_unknown"):
        run_schedule((unit,), tmp_path / "run", live=False, sender=fail)
    ledger = BudgetLedger(tmp_path / "run" / "budget.sqlite3")
    request_id = ledger.snapshot()["unresolved_request_ids"][0]
    assert request_id.startswith("req-" + unit.unit_id + "-")
    assert main([
        "resolve", "--run-dir", str(tmp_path / "run"), "--request-id", request_id,
        "--worst-case", "--reason", "charge the reserved hold",
        "--restart-unit", unit.unit_id,
    ]) == 0
    settled = ledger.snapshot()
    assert settled["unresolved_request_ids"] == []
    audit = ledger.audit_rows()[0]
    assert audit["charge_microcny"] == audit["reserved_microcny"]
    assert audit["released_tokens"] > 0
    assert settled["used_tokens"] == 0
    assert secret not in (tmp_path / "run" / "diagnostics" / f"{request_id}.json").read_text(encoding="utf-8")
    assert not (tmp_path / "run" / "checkpoints" / unit.unit_id).exists()
    archived = list((tmp_path / "run" / "archive" / "restarts" / unit.unit_id).rglob("records.jsonl"))
    assert archived and "provider_outcome_unknown" in archived[0].read_text(encoding="utf-8")

    def succeed(url, body, headers):
        return _body(body["model"], '{"final":"restarted"}', _usage(3, 2))

    rows = run_schedule((unit,), tmp_path / "run", live=False, sender=succeed)
    assert rows[0]["status"] == "completed"
    assert rows[0]["restart_count"] == 1
    assert any(item.startswith(f"req-{unit.unit_id}-r1-") for item in ledger.request_ids())

    with pytest.raises(PilotStopped, match="provider_outcome_unknown"):
        run_schedule((unit,), tmp_path / "second", live=False, sender=fail)
    # Second interruption is on a unit that has already used its one restart.
    (tmp_path / "second" / "unit_restarts.json").write_text(
        json.dumps({unit.unit_id: 1}), encoding="utf-8")
    second_ledger = BudgetLedger(tmp_path / "second" / "budget.sqlite3")
    first_hold = second_ledger.snapshot()["unresolved_request_ids"][0]
    assert main([
        "resolve", "--run-dir", str(tmp_path / "second"), "--request-id", first_hold,
        "--worst-case", "--reason", "first hold on the restarted generation",
    ]) == 0
    # The record from the generation-0 failure is still "stopped". Remove it so the
    # same unit is dispatched again with restart_count 1.
    record_path = tmp_path / "second" / "records.jsonl"
    kept = []
    for line in record_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("unit_id") != unit.unit_id:
            kept.append(line)
    record_path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    with pytest.raises(PilotStopped, match="second_unknown_interruption"):
        run_schedule((unit,), tmp_path / "second", live=False, sender=fail)
    blocked = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
    assert blocked[-1]["status"] == "blocked"
    assert blocked[-1]["reason"] == "second_unknown_interruption"
    assert blocked[-1]["restart_count"] == 1
    second = second_ledger.snapshot()["unresolved_request_ids"][0]
    assert main([
        "resolve", "--run-dir", str(tmp_path / "second"), "--request-id", second,
        "--charge-microcny", 0, "--reason", "release after the second interruption",
    ]) == 0
    assert main([
        "resolve", "--run-dir", str(tmp_path / "second"), "--restart-unit", unit.unit_id,
        "--reason", "second restart is refused",
    ]) == 2
