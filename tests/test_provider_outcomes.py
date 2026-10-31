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
    with pytest.raises(PilotStopped, match="provider_deadline_charged"):
        held.complete_for_model("mimo-v2.6-flash", messages(), max_tokens=128)
    timeout = _diag(tmp_path / "timeout", "req-m1_b0-1")
    assert timeout["cause"] == "deadline_timeout"
    assert timeout["charged_timeout"] is True
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
    assert rows[0]["reason"] == "second_timeout_interruption"
    assert rows[0]["charged_timeout"] is True
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
    assert archived and "provider_deadline_charged" in archived[0].read_text(encoding="utf-8")
    unknown = tmp_path / "run" / "unknown_outcomes.jsonl"
    assert not unknown.is_file() or unknown.read_text(encoding="utf-8").strip() == ""
    counts = json.loads((tmp_path / "run" / "progress" / "timeout_counts.json").read_text(encoding="utf-8"))
    assert counts["charged_timeouts"] == 2
    assert counts["halt_reason"] is None
    assert counts["by_model"]["mimo-v2.6-flash"]["charged_timeouts_in_window"] == 2
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


def test_hold_settles_inside_the_wall_clock_margin(tmp_path, monkeypatch):
    from alo_cloud.providers import _read_sse
    state = {"t": 0.0}

    def now():
        return state["t"]

    def fake_send(url, body, headers, *, timeout_s, model, started=None, clock=None):
        del url, body, headers
        assert started == 0.0
        assert timeout_s == 600

        class Tick:
            def readline(self):
                state["t"] += 30
                return b": keepalive\n"

        return _read_sse(Tick(), model, deadline_s=timeout_s, started=started, clock=clock), 200

    monkeypatch.setattr("alo_cloud.providers.urllib_send", fake_send)
    services, _seen, ledger = build(
        tmp_path, monkeypatch, [], permit_network=True, clock=now)
    with pytest.raises(PilotStopped, match="provider_deadline_charged"):
        services.complete_for_model("mimo-v2.6-flash", messages(), max_tokens=16)
    assert state["t"] <= 600 + 30
    assert ledger.snapshot()["unresolved_request_ids"] == []
    record = _diag(tmp_path, "req-m1_b0-1")
    assert record["cause"] == "deadline_timeout"
    assert record["charged_timeout"] is True
    assert record["settled"] is True
    assert record["latency_ms"] <= (600 + 30) * 1000


def test_blocking_close_cannot_keep_the_hold_open(tmp_path, monkeypatch):
    import time
    import urllib.request

    from alo_cloud.freeze import PROVIDER_TIMEOUT_S

    deadline = 0.3
    monkeypatch.setitem(PROVIDER_TIMEOUT_S, "mimo-v2.6-flash", deadline)
    closed = {"n": 0}

    class Sock:
        def settimeout(self, value):
            return None

        def shutdown(self, how):
            return None

        def close(self):
            return None

    class Response:
        status = 200
        _sock = Sock()

        def readline(self):
            time.sleep(5)
            return b": keepalive\n"

        def close(self):
            closed["n"] += 1
            time.sleep(5)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response())
    services, _seen, ledger = build(tmp_path, monkeypatch, [], permit_network=True)
    started = time.monotonic()
    with pytest.raises(PilotStopped, match="provider_deadline_charged"):
        services.complete_for_model("mimo-v2.6-flash", messages(), max_tokens=16)
    elapsed = time.monotonic() - started
    assert elapsed <= deadline + 1.0
    assert closed["n"] == 0
    assert ledger.snapshot()["unresolved_request_ids"] == []
    record = _diag(tmp_path, "req-m1_b0-1")
    assert record["cause"] == "deadline_timeout"
    assert record["charged_timeout"] is True
    assert record["settled"] is True
    assert record["latency_ms"] <= 1000


def test_unit_request_cap_does_not_halt_the_schedule(tmp_path, monkeypatch):
    from alo_cloud.adapters import HeterogeneousALO
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    calls = {"n": 0}

    def stop_unit(self, stage, services):
        del self, stage, services
        calls["n"] += 1
        raise PilotStopped("unit_resource_exhausted")

    monkeypatch.setattr(HeterogeneousALO, "run_stage", stop_unit)
    units = select_units("m1", arms=("g2",), tasks=("scikit-learn__scikit-learn-13584",))
    rows = run_schedule(units[:2], tmp_path / "run", live=False, sender=lambda *args: (_ for _ in ()).throw(AssertionError("send")))
    assert calls["n"] == 2
    assert [row["reason"] for row in rows] == ["unit_resource_exhausted", "unit_resource_exhausted"]
    assert all(row["halt_schedule"] is False for row in rows)

    def stop_study(self, stage, services):
        del self, stage, services
        raise PilotStopped("budget_infeasible_all_models")

    monkeypatch.setattr(HeterogeneousALO, "run_stage", stop_study)
    with pytest.raises(PilotStopped, match="budget_infeasible_all_models"):
        run_schedule(units[:2], tmp_path / "global", live=False, sender=lambda *args: (_ for _ in ()).throw(AssertionError("send")))
    halted = [json.loads(line) for line in (tmp_path / "global" / "records.jsonl").read_text().splitlines() if line.strip()]
    assert len(halted) == 1
    assert halted[0]["halt_schedule"] is True


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


def test_mimo_flash_deadline_matches_pro():
    from alo_cloud.freeze import PROVIDER_TIMEOUT_S
    assert PROVIDER_TIMEOUT_S["mimo-v2.6-flash"] == 600
    assert PROVIDER_TIMEOUT_S["mimo-v2.6-pro"] == 600
    assert PROVIDER_TIMEOUT_S["deepseek-flash"] == 300
    assert PROVIDER_TIMEOUT_S["glm-5.3-flash"] == 300


def test_deadline_timeouts_do_not_count_toward_the_unknown_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    units = select_units("m1", arms=("g1-main",))[:2]

    def fail(url, body, headers):
        del url, body, headers
        raise TimeoutError("provider_wall_clock")

    rows = run_schedule(units, tmp_path / "run", live=False, sender=fail)
    assert [row["reason"] for row in rows] == ["second_timeout_interruption", "second_timeout_interruption"]
    assert all(row["halt_schedule"] is False and row["charged_timeout"] is True for row in rows)
    unknown = tmp_path / "run" / "unknown_outcomes.jsonl"
    assert not unknown.exists()
    counts = json.loads((tmp_path / "run" / "progress" / "timeout_counts.json").read_text(encoding="utf-8"))
    assert counts["charged_timeouts"] == 4
    assert counts["halt_reason"] is None
    assert counts["by_model"]["mimo-v2.6-flash"]["window_full"] is False


def test_genuine_unknowns_still_halt_after_three(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    units = select_units("m1", arms=("g1-main",))[:2]

    def fail(url, body, headers):
        del url, body, headers
        raise ConnectionResetError("reset")

    with pytest.raises(PilotStopped, match="unknown_outcome_threshold"):
        run_schedule(units, tmp_path / "run", live=False, sender=fail)
    halted = [json.loads(line) for line in (tmp_path / "run" / "records.jsonl").read_text().splitlines() if line.strip()]
    assert halted[-1]["reason"] == "unknown_outcome_threshold"
    assert halted[-1]["halt_schedule"] is True
    assert sum(1 for line in (tmp_path / "run" / "unknown_outcomes.jsonl").read_text().splitlines() if line.strip()) == 4


def test_charged_timeout_valves(tmp_path):
    from alo_cloud.contracts import MODEL_POOL, PER_MODEL_LIMIT_MICROCNY, STUDY_TOTAL_LIMIT_MICROCNY
    from alo_cloud.progress import assess_charged_timeouts

    def ledger_with(rows):
        path = tmp_path / f"budget-{len(list(tmp_path.glob('budget-*.sqlite3')))}.sqlite3"
        ledger = BudgetLedger.create(
            path,
            model_limits_microcny={model: PER_MODEL_LIMIT_MICROCNY for model in MODEL_POOL},
            opening_commitments_microcny={model: 0 for model in MODEL_POOL},
            max_requests=200, prior_requests=0, max_tokens=900_000, prior_tokens=0,
            max_seconds_ms=1_200_000, prior_seconds_ms=0, max_tools=120, prior_tools=0,
            total_limit_microcny=STUDY_TOTAL_LIMIT_MICROCNY, run_mode="dry-run")
        for request_id, held, cause in rows:
            ledger.reserve(request_id, "mimo-v2.6-flash", held, payload_sha256="a" * 64,
                           tokens=10, seconds_ms=10, unit_id="unit-a", unit_request_cap=200,
                           protocol_epoch=2)
            ledger.settle_known(request_id, held, tokens=10, seconds_ms=10)
            if cause:
                record = {
                    "schema": "alo-provider-diagnostic-v1", "request_id": request_id,
                    "cause": cause, "settled": True, "model": "mimo-v2.6-flash",
                }
                dest = path.parent / "diagnostics" / f"{request_id}.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(record), encoding="utf-8")
        return ledger

    under = [("req-e2-unit-%02d" % index, 1_000, "deadline_timeout" if index < 10 else None)
             for index in range(40)]
    quiet = assess_charged_timeouts(ledger_with(under), protocol_epoch=2)
    assert quiet["halt_reason"] is None
    assert quiet["by_model"]["mimo-v2.6-flash"]["charged_timeouts_in_window"] == 10

    over = [("req-e2-rate-%02d" % index, 1_000, "deadline_timeout" if index < 11 else None)
            for index in range(40)]
    # Historical epoch-2 diagnostics used cause "timeout" for the same settlement.
    over[0] = ("req-e2-rate-00", 1_000, "timeout")
    rated = assess_charged_timeouts(ledger_with(over), protocol_epoch=2)
    assert rated["halt_reason"] == "charged_timeout_rate"
    assert rated["by_model"]["mimo-v2.6-flash"]["window_full"] is True

    below_cap = assess_charged_timeouts(
        ledger_with([("req-e2-cny-1", 10_000_000, "deadline_timeout")]), protocol_epoch=2)
    assert below_cap["halt_reason"] is None
    above_cap = assess_charged_timeouts(
        ledger_with([("req-e2-cny-2", 10_000_001, "timeout")]), protocol_epoch=2)
    assert above_cap["halt_reason"] == "charged_timeout_cny"
    assert above_cap["charged_timeout_microcny"] == 10_000_001


def test_timeout_rate_halts_the_schedule_once_the_window_is_full(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    monkeypatch.setattr("alo_cloud.progress.CHARGED_TIMEOUT_WINDOW", 4)
    units = select_units("m1", arms=("g1-main",))[:2]

    def fail(url, body, headers):
        del url, body, headers
        raise TimeoutError("provider_wall_clock")

    with pytest.raises(PilotStopped, match="charged_timeout_rate"):
        run_schedule(units, tmp_path / "run", live=False, sender=fail)
    halted = [json.loads(line) for line in (tmp_path / "run" / "records.jsonl").read_text().splitlines() if line.strip()]
    assert halted[-1]["halt_schedule"] is True
    assert halted[-1]["reason"] == "charged_timeout_rate"
    counts = json.loads((tmp_path / "run" / "progress" / "timeout_counts.json").read_text(encoding="utf-8"))
    assert counts["halt_reason"] == "charged_timeout_rate"
    outcomes = (tmp_path / "run" / "progress" / "provider_outcomes.jsonl").read_text(encoding="utf-8")
    assert outcomes.count('"outcome": "deadline_timeout"') == 4


def test_resume_skips_a_halted_terminal_row(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    mains = select_units("m1", arms=("g1-main",))
    halted = next(unit for unit in mains if unit.task_id == "django__django-11583" and unit.phase == "B0")
    fresh = next(unit for unit in mains if unit.task_id == "scikit-learn__scikit-learn-13584" and unit.phase == "B0")
    run = tmp_path / "run"
    run.mkdir()
    (run / "protocol_epoch.json").write_text(
        json.dumps({"epoch": 2, "reason": "continue epoch 2"}) + "\n", encoding="utf-8")
    (run / "records.jsonl").write_text(json.dumps({
        "unit_id": halted.unit_id, "status": "stopped", "reason": "provider_outcome_charged",
        "halt_schedule": True, "restart_count": 1,
    }) + "\n", encoding="utf-8")

    def succeed(url, body, headers):
        del url, headers
        return _body(body["model"], '{"final":"ok"}', _usage(4, 2))

    rows = run_schedule((halted, fresh), run, live=False, sender=succeed)
    assert rows[0]["unit_id"] == halted.unit_id
    assert rows[0]["halt_schedule"] is True
    assert rows[0]["status"] == "stopped"
    assert rows[1]["unit_id"] == fresh.unit_id
    assert rows[1]["status"] == "completed"
    assert rows[1]["halt_schedule"] is False
    assert rows[1]["protocol_epoch"] == 2
    ledger = BudgetLedger(run / "budget.sqlite3")
    assert all(halted.unit_id not in request_id for request_id in ledger.request_ids())
    assert any(fresh.unit_id in request_id for request_id in ledger.request_ids())


def _racing_response(mode):
    """A response whose read unblocks when the deadline abort shuts the socket."""
    import threading

    release = threading.Event()

    class Sock:
        def __init__(self):
            self.aborted = False

        def settimeout(self, value):
            return None

        def shutdown(self, how):
            self.aborted = True
            release.set()

        def close(self):
            self.aborted = True
            release.set()

    class Response:
        status = 200

        def __init__(self):
            self._sock = Sock()
            self._sent_partial = False
            self.assembled = False

        def readline(self):
            if mode == "reset":
                release.wait(5)
                raise ConnectionResetError("reset by abort")
            if mode == "timeout":
                release.wait(5)
                raise TimeoutError("timed out")
            if mode == "empty":
                if not self._sent_partial:
                    self._sent_partial = True
                    return (
                        b'data: {"choices":[{"delta":{"content":"partial"},'
                        b'"finish_reason":"stop"}],"usage":{"prompt_tokens":3,'
                        b'"completion_tokens":1,"total_tokens":4}}\n'
                    )
                release.wait(5)
                return b""
            raise AssertionError(mode)

        def close(self):
            self.assembled = True

    return Response()


def test_deadline_race_is_a_charged_timeout(tmp_path, monkeypatch):
    import time
    import urllib.request

    from alo_cloud.freeze import PROVIDER_TIMEOUT_S
    from alo_cloud.providers import urllib_send

    monkeypatch.setattr(
        "socket.create_connection", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))

    def run_mode(mode, deadline):
        response = _racing_response(mode)

        def open_response(*args, **kwargs):
            return response

        monkeypatch.setattr(urllib.request, "urlopen", open_response)
        started = time.monotonic()
        try:
            urllib_send(
                "https://api.xiaomimimo.com/v1/chat/completions", {}, {},
                timeout_s=deadline, model="mimo-v2.6-flash")
        except TimeoutError as error:
            assert str(error) == "provider_wall_clock"
        else:
            raise AssertionError("partial body was returned as a completion")
        assert time.monotonic() - started < deadline + 1.0
        assert response.assembled is False
        return response

    run_mode("reset", 0.2)
    run_mode("timeout", 0.2)
    run_mode("empty", 0.2)

    monkeypatch.setitem(PROVIDER_TIMEOUT_S, "mimo-v2.6-flash", 0.2)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _racing_response("reset"))
    services, _seen, ledger = build(tmp_path, monkeypatch, [], permit_network=True)
    with pytest.raises(PilotStopped, match="provider_deadline_charged"):
        services.complete_for_model("mimo-v2.6-flash", messages(), max_tokens=16)
    record = _diag(tmp_path, "req-m1_b0-1")
    assert record["cause"] == "deadline_timeout"
    assert record["charged_timeout"] is True
    assert record["response"] is None
    counts = json.loads((tmp_path / "progress" / "timeout_counts.json").read_text(encoding="utf-8"))
    assert counts["charged_timeouts"] == 1
    assert counts["by_model"]["mimo-v2.6-flash"]["charged_timeouts_in_window"] == 1
    assert ledger.snapshot()["unresolved_request_ids"] == []


def test_return_at_the_deadline_is_a_charged_timeout_even_as_a_body(tmp_path, monkeypatch):
    state = {"t": 0.0}

    def now():
        return state["t"]

    def sender(url, body, headers):
        del url, headers
        state["t"] = 600
        return _body(body["model"], "partial", _usage(4, 2))

    services, seen, ledger = build(tmp_path, monkeypatch, [], clock=now)
    services.transport._sender = sender
    with pytest.raises(PilotStopped, match="provider_deadline_charged"):
        services.complete_for_model("mimo-v2.6-flash", messages(), max_tokens=16)
    record = _diag(tmp_path, "req-m1_b0-1")
    assert record["cause"] == "deadline_timeout"
    assert record["charged_timeout"] is True
    assert "partial" not in json.dumps(record)
    assert ledger.snapshot()["unresolved_request_ids"] == []
    assert len(seen) == 0


def test_pre_deadline_reset_stays_unknown(tmp_path, monkeypatch):
    import urllib.request

    from alo_cloud.providers import urllib_send

    class Response:
        status = 200
        _sock = None

        def readline(self):
            raise ConnectionResetError("peer reset")

        def close(self):
            return None

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ConnectionResetError, match="peer reset"):
        urllib_send(
            "https://api.xiaomimimo.com/v1/chat/completions", {}, {},
            timeout_s=30, model="mimo-v2.6-flash")

    def fail(url, body, headers):
        del url, body, headers
        raise ConnectionResetError("peer reset")

    services, _seen, ledger = build(tmp_path, monkeypatch, [])
    services.transport._sender = fail
    with pytest.raises(PilotStopped, match="provider_outcome_charged"):
        services.complete_for_model("deepseek-flash", messages(), max_tokens=16)
    record = _diag(tmp_path, "req-m1_b0-1")
    assert record["cause"] == "connection_reset"
    assert record["charged_timeout"] is False
    assert not (tmp_path / "progress" / "timeout_counts.json").exists()
    assert ledger.snapshot()["unresolved_request_ids"] == []
