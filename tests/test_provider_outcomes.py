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
    assert seen[0]["body"]["max_completion_tokens"] == 16384
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
    assert reserved == ceiling_for_messages("mimo-v2.6-flash", messages(), 16384)


def test_disagreed_reasoning_counts_stay_unknown(tmp_path, monkeypatch):
    monkeypatch.delenv("ALO_RUN_DIR", raising=False)
    usage = _usage(4, 4, reasoning=1, where="both")
    usage["completion_tokens_details"]["reasoning_tokens"] = 2
    blocked, seen, ledger = build(tmp_path, monkeypatch, [
        _body("deepseek-flash", "x", usage),
        _body("deepseek-flash", "y", _usage(1, 1)),
    ])
    with pytest.raises(PilotStopped, match="provider_outcome_charged"):
        blocked.complete_for_model("deepseek-flash", messages())
    assert blocked.complete_for_model("deepseek-flash", messages())["text"] == "y"
    assert len(seen) == 2
    record = _diag(tmp_path, "req-m1_b0-1")
    assert record["cause"] == "unparseable_success"
    assert record["settled"] is True
    assert ledger.snapshot()["unresolved_request_ids"] == []


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
    with pytest.raises(PilotStopped, match="provider_outcome_charged"):
        held.complete_for_model("mimo-v2.6-flash", messages(), max_tokens=128)
    timeout = _diag(tmp_path / "timeout", "req-m1_b0-1")
    assert timeout["cause"] == "timeout"
    assert timeout["settled"] is True
    assert timeout["response"] is None
    assert secret not in json.dumps(timeout)
    assert held_ledger.snapshot()["unresolved_request_ids"] == []
    assert held_ledger.audit_rows()[0]["charge_microcny"] == held_ledger.audit_rows()[0]["reserved_microcny"]
    # The settled hold does not block the next request.
    held.transport._sender = lambda url, body, headers: _body(
        body["model"], "after-timeout", _usage(2, 2))
    assert held.complete_for_model("mimo-v2.6-flash", messages("next"), max_tokens=128)["text"] == "after-timeout"

    reset, _seen, reset_ledger = build(tmp_path / "reset", monkeypatch, [])
    reset.transport._sender = lambda *args: (_ for _ in ()).throw(ConnectionResetError("reset"))
    with pytest.raises(PilotStopped, match="provider_outcome_charged"):
        reset.complete_for_model("deepseek-flash", messages())
    assert _diag(tmp_path / "reset", "req-m1_b0-1")["cause"] == "connection_reset"
    assert reset_ledger.snapshot()["unresolved_request_ids"] == []

    opaque, _seen, _ledger = build(tmp_path / "opaque", monkeypatch, [(None, 500)])
    with pytest.raises(PilotStopped, match="provider_outcome_charged"):
        opaque.complete_for_model("glm-5.3-flash", messages())
    assert _diag(tmp_path / "opaque", "req-m1_b0-1")["cause"] == "transport_error"


def test_timeout_charges_worst_case_restarts_once_and_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    live = tmp_path / "live-run" / "diagnostics"
    live.mkdir(parents=True)
    monkeypatch.setenv("ALO_RUN_DIR", str(tmp_path / "live-run"))
    secret = "live-key-must-not-leak"
    monkeypatch.setenv("MIMO_API_KEY", secret)
    unit = select_units("m1", arms=("g1-main",), tasks=("scikit-learn__scikit-learn-13584",))[0]

    def fail(url, body, headers):
        raise TimeoutError(secret)

    rows = run_schedule((unit,), tmp_path / "run", live=False, sender=fail)
    assert rows[0]["status"] == "blocked"
    assert rows[0]["reason"] == "second_unknown_interruption"
    assert rows[0]["restart_count"] == 1
    assert rows[0]["halt_schedule"] is False
    ledger = BudgetLedger(tmp_path / "run" / "budget.sqlite3")
    assert ledger.snapshot()["unresolved_request_ids"] == []
    assert all(row["charge_microcny"] == row["reserved_microcny"] for row in ledger.audit_rows())
    assert ledger.snapshot()["used_tokens"] == 0
    diagnostics = list((tmp_path / "run" / "diagnostics").glob("*.json"))
    assert diagnostics
    assert all(secret not in path.read_text(encoding="utf-8") for path in diagnostics)
    assert list(live.glob("*.json")) == []
    archived = list((tmp_path / "run" / "archive" / "restarts" / unit.unit_id).rglob("records.jsonl"))
    assert archived and "provider_outcome_charged" in archived[0].read_text(encoding="utf-8")
    assert sum(1 for line in (tmp_path / "run" / "unknown_outcomes.jsonl").read_text().splitlines() if line.strip()) == 2
    assert main([
        "archive-finished", "--run-dir", str(tmp_path / "run"),
        "--reason", "rerun under the updated caps",
    ]) == 0
    assert (tmp_path / "run" / "records.jsonl").read_text(encoding="utf-8").strip() == ""
    assert (tmp_path / "run" / "archive" / "protocol-rerun" / "records.jsonl").is_file()


def test_vendor_tool_call_and_stream_chunks():
    from alo_cloud.providers import parse_sse
    from alo_cloud.public_tools import PublicActionExtractor, coerce_public_action
    from alo_cloud.tool_loop import persistent_tool_loop
    from tests.test_tool_loop import LoopServices
    text = '<tool_call><function=list_files><parameter=arguments>{}</parameter></function></tool_call>'
    coerced = json.loads(coerce_public_action(text))
    assert coerced == {"tool": "list_files", "arguments": {}}
    services = LoopServices([
        'I will look first. ' + text,
        '{"final":"edited"}',
    ])
    result = persistent_tool_loop(services, "public task", [], max_turns=2)
    assert result["status"] == "completed"
    assert "list_files" in services.calls
    blocks = PublicActionExtractor().extract_code_blocks(
        '<tool_call><function=list_files><parameter=arguments>{}</parameter></function></tool_call>')
    assert blocks[0].language == "public_tool"
    assert json.loads(blocks[0].code)["tool"] == "list_files"
    mimo = "\n".join([
        'data: {"choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"{\\"final\\":\\"ok\\"}"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,'
        '"completion_tokens_details":{"reasoning_tokens":1}}}',
        "data: [DONE]",
    ])
    folded = parse_sse(mimo, "mimo-v2.6-pro")
    assert folded["choices"][0]["message"]["content"] == '{"final":"ok"}'
    assert folded["usage"]["completion_tokens"] == 2
    glm = (
        'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n'
        'data: {"choices":[{"finish_reason":"stop","delta":{"content":""}}],'
        '"usage":{"prompt_tokens":8,"completion_tokens":4,"total_tokens":12}}\n'
        "data: [DONE]\n"
    )
    assert parse_sse(glm, "glm-5.3-flash")["usage"]["total_tokens"] == 12
    deepseek = (
        'data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}],"usage":null}\n'
        'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5,'
        '"prompt_cache_hit_tokens":1,"prompt_cache_miss_tokens":3}}\n'
        "data: [DONE]\n"
    )
    folded = parse_sse(deepseek, "deepseek-flash")
    assert folded["choices"][0]["message"]["content"] == "a"
    assert folded["usage"]["prompt_cache_hit_tokens"] == 1


def test_trickling_keepalive_hits_the_wall_clock():
    from alo_cloud.providers import _read_sse
    clock = {"t": 0.0}

    class Tick:
        def readline(self):
            clock["t"] += 0.4
            return b": keepalive\n"

    with pytest.raises(TimeoutError, match="provider_wall_clock"):
        _read_sse(Tick(), "mimo-v2.6-pro", deadline_s=1.0, started=0.0, clock=lambda: clock["t"])
    assert clock["t"] <= 1.4


def test_restart_after_cap_minus_one_gets_a_fresh_request_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    unit = select_units("m1", arms=("g1-strong",), tasks=("scikit-learn__scikit-learn-13584",))[0]
    cap = unit.request_cap
    calls = {"n": 0}

    def sender(url, body, headers):
        del url, headers
        calls["n"] += 1
        if calls["n"] == cap:
            raise TimeoutError("trickle")
        return _body(body["model"], "not-json", _usage(4, 2))

    rows = run_schedule((unit,), tmp_path / "run", live=False, sender=sender)
    assert rows[0]["restart_count"] == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["reason"] != "cumulative_request_or_model_limit"
    assert calls["n"] == cap * 2
    ledger = BudgetLedger(tmp_path / "run" / "budget.sqlite3")
    ids = [item for item in ledger.request_ids() if unit.unit_id in item]
    first = [item for item in ids if "-r1-" not in item]
    second = [item for item in ids if "-r1-" in item]
    assert len(first) == cap
    assert len(second) == cap
    assert ledger.unit_usage(unit.unit_id, restart_count=1)["requests"] == cap
    assert ledger.snapshot()["committed_or_reserved_microcny"]["mimo-v2.6-pro"] > 0
    assert ledger.snapshot()["unresolved_request_ids"] == []
