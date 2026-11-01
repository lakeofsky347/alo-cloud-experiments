"""Client timeouts and pre-send release. No network and no credentials leave the machine."""
import json

import pytest

from alo_cloud.budget import BudgetLedger, BudgetStopped
from alo_cloud.errors import PilotStopped
from alo_cloud.ledger import requeue_interrupted
from alo_cloud.providers import ProviderNotReached, staged_https_post
from alo_cloud.r2 import R2_CONNECT_TIMEOUT_S, R2_READ_TIMEOUT_S
from tests.test_services import build, messages


def test_r2_timeouts_stay_out_of_the_freeze_and_split_connect_from_read():
    assert R2_CONNECT_TIMEOUT_S == 15
    assert R2_READ_TIMEOUT_S == {
        "deepseek-flash": 300,
        "mimo-v2.6-pro": 300,
        "mimo-v2.6-flash": 180,
        "glm-5.3-flash": 180,
    }
    from alo_cloud.r2 import r2_freeze_payload
    payload = r2_freeze_payload()
    assert "read_timeout" not in payload
    assert "connect_timeout" not in payload


class _Sock:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)


class _Lines:
    def __init__(self, lines):
        self.status = 200
        self._lines = list(lines)

    def readline(self):
        if not self._lines:
            return b""
        line = self._lines.pop(0)
        if isinstance(line, BaseException):
            raise line
        return line

    def close(self):
        return None


class ScriptedHTTPS:
    script = []
    made = []

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
        return _Lines(self.mode)

    def close(self):
        return None


def socket_timeout():
    import socket
    return socket.timeout("body")


def _usage():
    return {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


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


def test_bytes_then_timeout_stays_a_charged_deadline(tmp_path, monkeypatch):
    services, _seen, ledger = build(tmp_path, monkeypatch, [])
    _install(services, ["sse_timeout"], monkeypatch)
    with pytest.raises(PilotStopped, match="provider_deadline_charged"):
        services.complete_for_model("glm-5.3-flash", messages(), max_tokens=128)
    assert ledger.request_rows()[0]["status"] == "settled"
    assert ledger.unit_usage("unit-a")["requests"] == 1
    assert ScriptedHTTPS.made[0].sock.timeouts[0] == 180


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
