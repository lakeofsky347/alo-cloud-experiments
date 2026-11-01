"""Client timeouts and pre-send release. No network and no credentials leave the machine."""
import json

import pytest

from alo_cloud.budget import BudgetLedger, BudgetStopped
from alo_cloud.contracts import MODEL_POOL
from alo_cloud.errors import PilotStopped
from alo_cloud.freeze import PROVIDER_TIMEOUT_S, select_units
from alo_cloud.ledger import begin_outcome_epoch, main as ledger_main, requeue_interrupted
from alo_cloud.providers import ProviderNotReached, staged_https_post
from alo_cloud.r2 import R2_CONNECT_TIMEOUT_S, R2_READ_TIMEOUT_S, R2_STREAM_IDLE_TIMEOUT_S
from alo_cloud.run import _unknown_count, blocked_fraction_exceeded, dry_run_sender, run_schedule
from tests.test_services import build, messages


def test_r2_timeouts_stay_out_of_the_freeze_and_split_connect_from_read():
    assert R2_CONNECT_TIMEOUT_S == 15
    assert R2_READ_TIMEOUT_S == PROVIDER_TIMEOUT_S
    assert R2_READ_TIMEOUT_S is not PROVIDER_TIMEOUT_S
    from alo_cloud.r2 import r2_freeze_payload
    payload = r2_freeze_payload()
    assert "read_timeout" not in payload
    assert "connect_timeout" not in payload
    assert "provider_timeout_s" not in payload
    assert "stream_idle" not in payload
    assert R2_STREAM_IDLE_TIMEOUT_S == {model: 120 for model in MODEL_POOL}


class _Sock:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)

    def shutdown(self, how):
        return None

    def close(self):
        return None


class _Lines:
    def __init__(self, lines, *, arm_on_last=False):
        self.status = 200
        self._lines = list(lines)
        self.arm_on_last = arm_on_last

    def readline(self):
        if not self._lines:
            return b""
        line = self._lines.pop(0)
        if isinstance(line, BaseException):
            raise line
        if self.arm_on_last and not self._lines and ScriptedHTTPS.clock is not None:
            ScriptedHTTPS.clock.arm()
        return line

    def close(self):
        return None


class JumpClock:
    """Stays at zero until armed, then the next call is still zero and later calls jump."""

    def __init__(self, jump_to=1000.0):
        self.now = 0.0
        self.jump_to = jump_to
        self._skip = 0
        self._jump = False

    def arm(self):
        # The reader stamps the byte time and then the chunk time before the next wait.
        self._skip = 2
        self._jump = True

    def __call__(self):
        if self._jump:
            if self._skip > 0:
                self._skip -= 1
                return self.now
            return self.jump_to
        return self.now


class ScriptedHTTPS:
    script = []
    made = []
    clock = None

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.requested = False
        self.headers = None
        self.body = None
        self.mode = ScriptedHTTPS.script.pop(0)
        ScriptedHTTPS.made.append(self)

    def connect(self):
        if self.mode == "connect_timeout":
            raise TimeoutError("connect")
        self.sock = _Sock()

    def request(self, method, path, body=None, headers=None):
        self.requested = True
        self.headers = headers
        self.body = body
        if self.mode == "request_timeout":
            raise TimeoutError("write")

    def getresponse(self):
        if self.mode == "response_timeout":
            raise TimeoutError("headers")
        if self.mode == "sse_timeout":
            return _Lines([
                b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}\n',
                socket_timeout(),
            ])
        if self.mode == "sse_progress":
            return _Lines(_progress_lines(), arm_on_last=True)
        return _Lines(self.mode)

    def close(self):
        return None


def socket_timeout():
    import socket
    return socket.timeout("body")


def _usage():
    return {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def _progress_lines():
    first = json.dumps({"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": None}]})
    second = json.dumps({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]})
    return [f"data: {first}\n".encode(), f"data: {second}\n".encode()]


def _ok_lines():
    first = json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]})
    second = json.dumps({
        "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
        "usage": _usage(),
    })
    return [
        f"data: {first}\n".encode(),
        b": keepalive\n",
        f"data: {second}\n".encode(),
        b"data: [DONE]\n",
    ]


def test_connect_timeout_is_unsent_and_a_written_request_may_have_been_sent():
    ScriptedHTTPS.script = ["connect_timeout"]
    ScriptedHTTPS.made = []
    with pytest.raises(ProviderNotReached) as missed:
        staged_https_post(
            "https://api.deepseek.com/chat/completions", {"model": "deepseek-flash"},
            {"Authorization": "Bearer secret", "Content-Type": "application/json"},
            connect_timeout_s=15, read_timeout_s=300, model="deepseek-flash",
            connector=ScriptedHTTPS)
    assert missed.value.request_sent is False
    assert ScriptedHTTPS.made[0].requested is False
    assert ScriptedHTTPS.made[0].timeout == 15

    ScriptedHTTPS.script = ["response_timeout"]
    with pytest.raises(ProviderNotReached) as sent:
        staged_https_post(
            "https://api.deepseek.com/chat/completions", {"model": "deepseek-flash", "stream": True},
            {"Authorization": "Bearer secret"},
            connect_timeout_s=15, read_timeout_s=300, model="deepseek-flash",
            connector=ScriptedHTTPS)
    assert sent.value.request_sent is True
    assert ScriptedHTTPS.made[-1].requested is True
    assert ScriptedHTTPS.made[-1].headers["Accept"] == "text/event-stream"
    assert ScriptedHTTPS.made[-1].headers["Connection"] == "keep-alive"
    assert ScriptedHTTPS.made[-1].sock.timeouts == [300]
    assert b'"stream": true' in ScriptedHTTPS.made[-1].body


def test_release_not_sent_is_zero_and_does_not_block_the_next_reserve(tmp_path):
    ledger = BudgetLedger.create(
        tmp_path / "budget.sqlite3", model_limits_microcny={"deepseek-flash": 100},
        opening_commitments_microcny={"deepseek-flash": 0}, max_requests=3, prior_requests=0,
        max_tokens=1000, prior_tokens=0, max_seconds_ms=5000, prior_seconds_ms=0)
    ledger.reserve(
        "req-unit-1", "deepseek-flash", 20, payload_sha256="a" * 64, tokens=10, seconds_ms=1000,
        unit_id="unit", unit_request_cap=1)
    released = ledger.release_not_sent("req-unit-1")
    assert released == {"status": "not_sent", "charge_microcny": 0}
    assert ledger.snapshot()["unresolved_request_ids"] == []
    assert ledger.snapshot()["committed_or_reserved_microcny"]["deepseek-flash"] == 0
    assert ledger.unit_usage("unit")["requests"] == 0
    assert ledger.blocks_dispatch() is False
    ledger.reserve(
        "req-unit-2", "deepseek-flash", 20, payload_sha256="b" * 64, tokens=10, seconds_ms=1000,
        unit_id="unit", unit_request_cap=1)
    ledger.settle_known("req-unit-2", 7, tokens=4, seconds_ms=10)
    assert ledger.unit_usage("unit")["requests"] == 1
    with pytest.raises(BudgetStopped, match="only_pending_reservation_may_be_released"):
        ledger.release_not_sent("req-unit-2")


def _install(services, script, monkeypatch):
    monkeypatch.delenv("ALO_RUN_DIR", raising=False)
    from alo_cloud.providers import OpenAICompatibleTransport
    services.unit_id = "unit-a"
    services.limits["requests"] = 1
    services.read_timeouts = dict(R2_READ_TIMEOUT_S)
    services.connect_timeout_s = float(R2_CONNECT_TIMEOUT_S)
    ScriptedHTTPS.script = list(script)
    ScriptedHTTPS.made = []
    services.transport = OpenAICompatibleTransport(
        permit_network=True, timeout_s=300, require_credentials=True,
        connect_timeout_s=R2_CONNECT_TIMEOUT_S, connector=ScriptedHTTPS)
    return services


def test_presend_retry_does_not_consume_the_unit_cap(tmp_path, monkeypatch):
    services, _seen, ledger = build(tmp_path, monkeypatch, [])
    _install(services, ["connect_timeout", "connect_timeout"], monkeypatch)
    reply = services.complete_for_model("deepseek-flash", messages(), max_tokens=128)
    assert reply["provider_not_reached"] is True
    assert reply["cost_microcny"] == 0
    assert services.resources()["model_requests"] == 0
    rows = ledger.request_rows()
    assert [row["status"] for row in rows] == ["not_sent", "not_sent"]
    assert all(row["held_microcny"] == 0 for row in rows)
    assert ledger.unit_usage("unit-a")["requests"] == 0
    assert len(ScriptedHTTPS.made) == 2

    ScriptedHTTPS.script = [_ok_lines()]
    done = services.complete_for_model("deepseek-flash", messages("next"), max_tokens=128)
    assert done["text"] == "ok"
    assert done["cost_microcny"] == 1
    assert services.resources()["model_requests"] == 1
    assert ledger.unit_usage("unit-a")["requests"] == 1
    assert ScriptedHTTPS.made[-1].sock.timeouts[0] == 300


def test_maybe_sent_timeout_stays_unknown_worst_case(tmp_path, monkeypatch):
    services, _seen, ledger = build(tmp_path, monkeypatch, [])
    _install(services, ["response_timeout"], monkeypatch)
    with pytest.raises(PilotStopped, match="provider_outcome_charged"):
        services.complete_for_model("mimo-v2.6-pro", messages(), max_tokens=128)
    assert len(ScriptedHTTPS.made) == 1
    row = ledger.request_rows()[0]
    assert row["status"] == "settled"
    assert row["held_microcny"] == 1000
    assert ledger.unit_usage("unit-a")["requests"] == 1
    assert services.resources()["model_requests"] == 0
    record = json.loads((tmp_path / "diagnostics" / f"{row['id']}.json").read_text(encoding="utf-8"))
    assert record["cause"] == "provider_not_reached"
    assert record["settled"] is True


def test_bytes_then_timeout_stays_a_charged_deadline(tmp_path, monkeypatch):
    services, _seen, ledger = build(tmp_path, monkeypatch, [])
    _install(services, ["sse_timeout"], monkeypatch)
    with pytest.raises(PilotStopped, match="provider_deadline_charged"):
        services.complete_for_model("glm-5.3-flash", messages(), max_tokens=128)
    assert ledger.request_rows()[0]["status"] == "settled"
    assert ledger.unit_usage("unit-a")["requests"] == 1
    assert ScriptedHTTPS.made[0].sock.timeouts[0] == PROVIDER_TIMEOUT_S["glm-5.3-flash"]


def test_stream_timeout_records_progress_without_changing_the_charge(tmp_path, monkeypatch):
    services, _seen, ledger = build(tmp_path, monkeypatch, [])
    _install(services, ["sse_progress"], monkeypatch)
    clock = JumpClock()
    ScriptedHTTPS.clock = clock
    services._clock = clock
    lines = _progress_lines()
    with pytest.raises(PilotStopped, match="provider_deadline_charged"):
        services.complete_for_model("glm-5.3-flash", messages(), max_tokens=128)
    row = ledger.request_rows()[0]
    assert row["status"] == "settled"
    assert row["held_microcny"] == 1000
    record = json.loads((tmp_path / "diagnostics" / f"{row['id']}.json").read_text(encoding="utf-8"))
    assert record["cause"] == "deadline_timeout"
    assert record["charged_timeout"] is True
    assert record["stream_progress"] == {
        "bytes_read": sum(len(line) for line in lines),
        "chunks": 2,
        "last_chunk_age_s": 1000.0,
        "saw_reasoning": True,
    }


def test_requeue_interrupted_selects_provider_unknown_units(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {"unit_id": "m1.deepseek", "status": "stopped", "reason": "provider_outcome_charged",
         "termination_reason": "provider_unknown"},
        {"unit_id": "m1.mimo", "status": "blocked", "reason": "second_unknown_interruption",
         "termination_reason": "blocked"},
        {"unit_id": "m1.kept", "status": "stopped", "reason": "unit_resource_exhausted",
         "termination_reason": "budget"},
    ]
    (run / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = requeue_interrupted(run)
    assert result["requeued"] == ["m1.deepseek", "m1.mimo"]
    live = (run / "records.jsonl").read_text(encoding="utf-8")
    assert "m1.kept" in live
    assert "m1.deepseek" not in live


def test_stream_idle_timeout_records_progress_and_charges(tmp_path, monkeypatch):
    services, _seen, ledger = build(tmp_path, monkeypatch, [])
    _install(services, ["sse_progress"], monkeypatch)
    services.stream_idle_timeouts = dict(R2_STREAM_IDLE_TIMEOUT_S)
    clock = JumpClock(120.0)
    ScriptedHTTPS.clock = clock
    services._clock = clock
    lines = _progress_lines()
    with pytest.raises(PilotStopped, match="provider_deadline_charged"):
        services.complete_for_model("glm-5.3-flash", messages(), max_tokens=128)
    row = ledger.request_rows()[0]
    assert row["status"] == "settled"
    assert row["held_microcny"] == 1000
    record = json.loads((tmp_path / "diagnostics" / f"{row['id']}.json").read_text(encoding="utf-8"))
    assert record["cause"] == "provider_stream_idle"
    assert record["charged_timeout"] is True
    progress = record["stream_progress"]
    assert progress["bytes_read"] == sum(len(line) for line in lines)
    assert progress["chunks"] == 2
    assert progress["saw_reasoning"] is True
    assert progress["last_chunk_age_s"] == 120.0


def test_charged_timeout_retries_the_same_messages_once(tmp_path, monkeypatch):
    body = {
        "model": "deepseek-flash",
        "choices": [{"message": {"content": '{"final":"kept"}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    services, seen, ledger = build(tmp_path, monkeypatch, [TimeoutError("stall"), body])
    services.continue_after_timeout = True
    services.unit_id = "unit-a"
    reply = services.complete_for_model("deepseek-flash", messages(), max_tokens=128)
    assert reply["text"] == '{"final":"kept"}'
    assert ledger.unit_usage("unit-a")["requests"] == 2
    assert services.resources()["model_requests"] == 2
    assert len(seen) == 2
    assert seen[0]["body"]["messages"] == seen[1]["body"]["messages"]
    assert services.charged_timeouts == 1


def test_second_timeout_stops_the_unit_without_blocking_or_restarting(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))

    def fail(url, body, headers):
        raise TimeoutError("stall")

    unit = select_units("m1", arms=("g1-main",), tasks=("scikit-learn__scikit-learn-13584",))[0]
    rows = run_schedule((unit,), tmp_path / "run", live=False, sender=fail, batch="r2")
    assert len(rows) == 1
    assert rows[0]["status"] == "stopped"
    assert rows[0]["reason"] == "provider_timeout_exhausted"
    assert rows[0]["termination_reason"] == "provider_timeout_exhausted"
    assert rows[0]["restart_count"] == 0
    assert not (tmp_path / "run" / "unit_restarts.json").exists()
    assert BudgetLedger(tmp_path / "run" / "budget.sqlite3").unit_usage(unit.unit_id)["requests"] == 2


def test_one_timeout_then_a_completion_does_not_restart_the_unit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    state = {"n": 0}

    def sender(url, body, headers):
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError("stall")
        return dry_run_sender(url, body, headers)

    unit = select_units("m1", arms=("g1-main",), tasks=("django__django-11583",))[0]
    rows = run_schedule((unit,), tmp_path / "run", live=False, sender=sender, batch="r2")
    assert rows[0]["status"] == "completed"
    assert rows[0]["termination_reason"] == "final"
    assert rows[0]["restart_count"] == 0
    assert rows[0]["charged_timeout"] is True
    assert BudgetLedger(tmp_path / "run" / "budget.sqlite3").unit_usage(unit.unit_id)["requests"] == 2


def test_unknown_cause_still_writes_a_diagnostic(tmp_path, monkeypatch):
    services, _seen, _ledger = build(tmp_path, monkeypatch, [])
    services._diagnose(
        "req-unit-a-9", "glm-5.3-flash", {"cause": "not_a_real_cause"}, latency_ms=1, settled=True)
    record = json.loads((tmp_path / "diagnostics" / "req-unit-a-9.json").read_text(encoding="utf-8"))
    assert record["cause"] == "transport_error"
    assert record["vendor_code"] == "not_a_real_cause"
    assert record["settled"] is True


def test_outcome_epoch_hides_historical_unknowns_and_blocked_rows(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    lines = [json.dumps({"unit_id": f"old-{index}", "reason": "provider_outcome_charged"}) for index in range(5)]
    (run / "unknown_outcomes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _unknown_count(run) == 5
    assert ledger_main(["outcome-epoch", "--run-dir", str(run), "--reason", "after the regression"]) == 0
    assert _unknown_count(run) == 0
    assert len((run / "unknown_outcomes.jsonl").read_text(encoding="utf-8").splitlines()) == 5
    from alo_cloud.run import _note_unknown
    _note_unknown(run, {"unit_id": "new", "reason": "provider_outcome_charged", "unknown_request_id": "req-new-1"})
    assert _unknown_count(run) == 1
    historical = [{"unit_id": f"b{index}", "status": "blocked", "reason": "second_timeout_interruption"}
                  for index in range(6)]
    assert blocked_fraction_exceeded(historical) is True
    assert blocked_fraction_exceeded(historical, outcome_epoch=1) is False
    fresh = [{"unit_id": f"n{index}", "status": "blocked", "reason": "second_timeout_interruption",
              "outcome_epoch": 1} for index in range(4)]
    assert blocked_fraction_exceeded(fresh, outcome_epoch=1) is True


def test_requeue_interrupted_selects_a_timeout_block_without_a_restart(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "unit_restarts.json").write_text(json.dumps({"m1.timeout": 1}) + "\n", encoding="utf-8")
    rows = [
        {"unit_id": "m1.timeout", "status": "blocked", "reason": "second_timeout_interruption"},
        {"unit_id": "m1.done", "status": "completed", "reason": ""},
    ]
    (run / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = requeue_interrupted(run, ["m1.done"])
    assert result["requeued"] == ["m1.done", "m1.timeout"]
    assert result["restart_counts_unchanged"] is True
    assert json.loads((run / "unit_restarts.json").read_text(encoding="utf-8")) == {"m1.timeout": 1}
    assert begin_outcome_epoch(run, reason="again")["epoch"] == 2
