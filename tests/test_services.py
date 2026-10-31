"""Offline StudyServices checks. Transport replies and charges are scripted."""
import hashlib
import json
import random
from pathlib import Path

import pytest

from alo.org.organization import Organization
from alo_cloud.budget import BudgetLedger
from alo_cloud.contracts import MODEL_POOL, PER_MODEL_LIMIT_MICROCNY
from alo_cloud.errors import PilotStopped, ProtocolError
from alo_cloud.native_bridge import BoundModel
from alo_cloud.policy import RESOURCE_FIELDS, ConstrainedThompsonPolicy
from alo_cloud.providers import POOL_ENDPOINTS
from alo_cloud.sandbox import DockerSandbox
from alo_cloud.services import PROTOCOL_UNIT_LIMITS, StudyServices
from alo_cloud.tool_loop import persistent_tool_loop


KEYS = {
    "DEEPSEEK_API_KEY": "unit-test-credential-deepseek",
    "MIMO_API_KEY": "unit-test-credential-mimo",
    "ZAI_API_KEY": "unit-test-credential-zai",
}


def chat(text, model, prompt=3, completion=2, reasoning=0):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    if reasoning:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return {"model": model, "choices": [{"message": {"content": text}}], "usage": usage}


def scripted(replies):
    seen = []

    def sender(url, body, headers):
        blob = json.dumps(body, ensure_ascii=False)
        if "api-key" in headers:
            scheme, material = "api-key", headers["api-key"]
        else:
            scheme, material = "bearer", headers["Authorization"]
        seen.append({"url": url, "body": body, "blob": blob, "auth_scheme": scheme,
                     "auth_sha256": hashlib.sha256(material.encode()).hexdigest()})
        if not replies:
            raise RuntimeError("scripted transport exhausted")
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    return sender, seen


def build(tmp_path, monkeypatch, replies, **kwargs):
    for name, value in KEYS.items():
        if kwargs.get("keys", True):
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    unit = kwargs.get("unit", "m1_b0")
    limits = PROTOCOL_UNIT_LIMITS[unit]
    opening = {model: 0 for model in MODEL_POOL}
    opening.update(kwargs.get("opening", {}))
    ledger_path = Path(tmp_path) / "budget.sqlite3"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger.create(
        ledger_path,
        model_limits_microcny={model: kwargs.get("limit", PER_MODEL_LIMIT_MICROCNY) for model in MODEL_POOL},
        opening_commitments_microcny=opening,
        max_requests=limits["requests"], prior_requests=kwargs.get("prior_requests", 0),
        max_tokens=limits["tokens"], prior_tokens=kwargs.get("prior_tokens", 0),
        max_seconds_ms=limits["seconds"] * 1000,
        prior_seconds_ms=kwargs.get("prior_seconds_ms", 0),
        max_tools=limits["tools"], prior_tools=0)
    sender, seen = scripted(replies)
    services = StudyServices(
        tmp_path / "state", ledger, unit=unit,
        per_request_microcny_ceiling=kwargs.get("ceiling", 1000),
        charge_microcny=kwargs.get("charge", lambda model, usage: 1),
        request_ceiling=kwargs.get("request_ceiling"),
        sender=None if kwargs.get("permit_network") else sender,
        permit_network=bool(kwargs.get("permit_network")),
        sandbox=kwargs.get("sandbox"), clock=kwargs.get("clock"),
        shard_id=kwargs.get("shard_id", ""))
    return services, seen, ledger


def messages(text="public task"):
    return [{"role": "user", "content": text}]


def test_protocol_limits_and_payg_endpoints_are_explicit():
    assert PROTOCOL_UNIT_LIMITS["m1_b1"]["requests"] == 16
    for name in ("strong_b0", "m1_b0", "m2"):
        assert PROTOCOL_UNIT_LIMITS[name]["requests"] == 32
    for limits in PROTOCOL_UNIT_LIMITS.values():
        assert limits["tokens"] == 900_000
        assert limits["tools"] == 120
        assert limits["seconds"] == 1200
        assert limits["command_seconds"] == 60
        assert limits["max_output_tokens"] == 16384
    assert set(POOL_ENDPOINTS) == set(MODEL_POOL)
    assert POOL_ENDPOINTS["deepseek-flash"] == (
        "DEEPSEEK_API_KEY", "https://api.deepseek.com/chat/completions", "bearer")
    assert POOL_ENDPOINTS["mimo-v2.6-flash"][0] == POOL_ENDPOINTS["mimo-v2.6-pro"][0] == "MIMO_API_KEY"
    assert POOL_ENDPOINTS["mimo-v2.6-flash"][1] == "https://api.xiaomimimo.com/v1/chat/completions"
    assert POOL_ENDPOINTS["mimo-v2.6-flash"][2] == POOL_ENDPOINTS["mimo-v2.6-pro"][2] == "api-key"
    assert POOL_ENDPOINTS["glm-5.3-flash"] == (
        "ZAI_API_KEY", "https://open.bigmodel.cn/api/paas/v4/chat/completions", "bearer")
    assert all("token-plan" not in url and "api.z.ai" not in url
               for _env, url, _auth in POOL_ENDPOINTS.values())


def test_missing_sender_or_two_senders_refused(tmp_path):
    ledger = BudgetLedger.create(
        tmp_path / "budget.sqlite3",
        model_limits_microcny={model: 1 for model in MODEL_POOL},
        opening_commitments_microcny={model: 0 for model in MODEL_POOL},
        max_requests=32, prior_requests=0, max_tokens=900_000, prior_tokens=0,
        max_seconds_ms=1_200_000, prior_seconds_ms=0, max_tools=120, prior_tools=0)
    with pytest.raises(PilotStopped, match="provider_sender_required"):
        StudyServices(tmp_path / "state", ledger, unit="m1_b0", per_request_microcny_ceiling=1,
                      charge_microcny=lambda model, usage: 1)
    with pytest.raises(PilotStopped, match="one_transport_sender"):
        StudyServices(tmp_path / "state", ledger, unit="m1_b0", per_request_microcny_ceiling=1,
                      charge_microcny=lambda model, usage: 1, sender=lambda *args: {}, permit_network=True)


def test_completion_settles_four_dimensions_without_logging_the_key(tmp_path, monkeypatch):
    services, seen, ledger = build(tmp_path, monkeypatch, [
        chat("visible-reply-text", "mimo-v2.6-flash", prompt=3, completion=6, reasoning=4),
        chat("second", "deepseek-flash"),
        chat("third", "glm-5.3-flash"),
        chat("fourth", "mimo-v2.6-pro"),
        chat("rotated", "mimo-v2.6-flash"),
    ])
    first = services.complete_for_model("mimo-v2.6-flash", messages(), max_tokens=128)
    assert first["text"] == "visible-reply-text"
    assert first["currency"] == "CNY"
    assert first["cost_microcny"] == 1
    assert first["reasoning_tokens"] == 4
    assert seen[0]["url"] == POOL_ENDPOINTS["mimo-v2.6-flash"][1]
    assert seen[0]["body"]["max_completion_tokens"] == 128
    assert "max_tokens" not in seen[0]["body"]
    assert seen[0]["body"]["stream"] is True
    assert "stream_options" not in seen[0]["body"]
    assert "temperature" not in seen[0]["body"] and "thinking" not in seen[0]["body"]
    assert seen[0]["auth_scheme"] == "api-key"
    assert KEYS["MIMO_API_KEY"] not in seen[0]["blob"]
    assert seen[0]["auth_sha256"] == hashlib.sha256(KEYS["MIMO_API_KEY"].encode()).hexdigest()
    services.complete_for_model("deepseek-flash", messages("next"))
    assert seen[1]["body"]["max_tokens"] == 16384 and "thinking" not in seen[1]["body"]
    assert seen[1]["body"]["stream_options"] == {"include_usage": True}
    assert seen[1]["auth_scheme"] == "bearer"
    services.complete_for_model("glm-5.3-flash", messages("glm"))
    assert seen[2]["body"]["thinking"] == {"type": "enabled"}
    assert seen[2]["body"]["max_tokens"] == 16384
    assert "stream_options" not in seen[2]["body"]
    assert "max_completion_tokens" not in seen[2]["body"]
    assert "reasoning_effort" not in seen[2]["body"]
    services.complete_for_model("mimo-v2.6-pro", messages("pro"))
    assert [item["url"] for item in seen] == [
        POOL_ENDPOINTS[model][1] for model in
        ("mimo-v2.6-flash", "deepseek-flash", "glm-5.3-flash", "mimo-v2.6-pro")]
    remaining = services.budget_remaining("mimo-v2.6-flash")
    assert set(remaining) == set(RESOURCE_FIELDS)
    assert remaining["requests"] == 32 - 4
    assert remaining["tokens"] == 900_000 - (9 + 5 + 5 + 5)
    assert remaining["cny"] == (PER_MODEL_LIMIT_MICROCNY - 1) / 1_000_000
    assert services.budget_remaining("deepseek-flash")["requests"] == remaining["requests"]
    assert services.resources()["model_requests"] == 4
    assert services.resources()["observed_reasoning_tokens"] == 4
    rendered = repr(services) + json.dumps(services.events)
    assert "visible-reply-text" not in rendered
    for secret in KEYS.values():
        assert secret not in rendered
    monkeypatch.setenv("MIMO_API_KEY", "unit-test-credential-mimo-rotated")
    services.complete_for_model("mimo-v2.6-flash", messages("rotated"), max_tokens=32)
    assert seen[-1]["auth_sha256"] == hashlib.sha256(
        b"unit-test-credential-mimo-rotated").hexdigest()
    assert seen[-1]["auth_sha256"] != seen[0]["auth_sha256"]
    ledger.snapshot()


def test_reasoning_tokens_stay_inside_the_completion_total(tmp_path, monkeypatch):
    included = chat("answer", "glm-5.3-flash", prompt=3, completion=10, reasoning=4)
    services, _seen, ledger = build(tmp_path, monkeypatch, [included])
    result = services.complete_for_model("glm-5.3-flash", messages())
    assert result["reasoning_tokens"] == 4
    assert ledger.snapshot()["used_tokens"] == 13
    assert services.resources()["observed_reasoning_tokens"] == 4
    assert services.events[-1][1]["reasoning_tokens"] == 4

    oversized = chat("x", "deepseek-flash", prompt=1, completion=2, reasoning=9)
    blocked, blocked_seen, blocked_ledger = build(
        tmp_path / "over", monkeypatch, [oversized, chat("y", "deepseek-flash")])
    with pytest.raises(PilotStopped, match="provider_outcome_charged"):
        blocked.complete_for_model("deepseek-flash", messages())
    assert blocked.complete_for_model("deepseek-flash", messages())["text"] == "y"
    assert blocked_ledger.snapshot()["unresolved_request_ids"] == []
    assert len(blocked_seen) == 2

    unsplit = {"model": "glm-5.3-flash", "choices": [{"message": {"content": "x"}}],
               "usage": {"prompt_tokens": 8, "completion_tokens": 239, "total_tokens": 247,
                         "prompt_tokens_details": {"cached_tokens": 0}}}
    quiet, _seen, quiet_ledger = build(tmp_path / "quiet", monkeypatch, [unsplit])
    quiet_result = quiet.complete_for_model("glm-5.3-flash", messages())
    assert quiet_result["reasoning_tokens"] is None
    assert quiet.resources()["observed_reasoning_tokens"] == 0
    assert quiet_ledger.snapshot()["used_tokens"] == 247
    assert quiet.events[-1][1]["reasoning_tokens"] is None


def test_missing_key_bad_usage_and_over_charge_stop_before_another_send(tmp_path, monkeypatch):
    services, seen, ledger = build(tmp_path, monkeypatch, [])
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    with pytest.raises(PilotStopped, match="provider_credential_missing"):
        services.complete_for_model("mimo-v2.6-flash", messages())
    assert seen == []
    assert ledger.snapshot()["requests"] == 0
    monkeypatch.setenv("MIMO_API_KEY", KEYS["MIMO_API_KEY"])
    with pytest.raises(PilotStopped, match="provider_credential_in_prompt"):
        services.complete_for_model("mimo-v2.6-flash", messages(KEYS["MIMO_API_KEY"]))
    assert ledger.snapshot()["requests"] == 0

    blocked, blocked_seen, blocked_ledger = build(tmp_path / "bad", monkeypatch, [{"choices": []}])
    with pytest.raises(PilotStopped, match="provider_outcome_charged"):
        blocked.complete_for_model("glm-5.3-flash", messages())
    assert blocked_ledger.snapshot()["unresolved_request_ids"] == []
    assert len(blocked_seen) == 1

    mismatch, mismatch_seen, _ledger = build(tmp_path / "mismatch", monkeypatch, [
        chat("x", "other-model"),
    ])
    with pytest.raises(PilotStopped, match="provider_outcome_charged"):
        mismatch.complete_for_model("glm-5.3-flash", messages())
    assert mismatch_seen[0]["body"]["model"] == "glm-5.3-flash"

    inconsistent, _seen, inconsistent_ledger = build(tmp_path / "tokens", monkeypatch, [{
        "model": "deepseek-flash",
        "choices": [{"message": {"content": "x"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3},
    }])
    with pytest.raises(PilotStopped, match="provider_outcome_charged"):
        inconsistent.complete_for_model("deepseek-flash", messages())
    assert inconsistent_ledger.snapshot()["unresolved_request_ids"] == []

    costly, costly_seen, _ledger = build(
        tmp_path / "cost", monkeypatch, [chat("x", "deepseek-flash")],
        charge=lambda model, usage: 1001, ceiling=1000)
    with pytest.raises(PilotStopped, match="provider_outcome_unknown"):
        costly.complete_for_model("deepseek-flash", messages())
    with pytest.raises(PilotStopped, match="unresolved"):
        costly.complete_for_model("deepseek-flash", messages())
    assert len(costly_seen) == 1


def test_unit_caps_stop_dispatch_without_a_further_send(tmp_path, monkeypatch):
    replies = [chat('{"final":"ok"}', "mimo-v2.6-flash") for _ in range(16)]
    services, seen, _ledger = build(tmp_path, monkeypatch, replies, unit="m1_b1")
    for index in range(16):
        services.complete_for_model("mimo-v2.6-flash", messages(str(index)))
    with pytest.raises(PilotStopped, match="unit_resource_exhausted"):
        services.complete_for_model("mimo-v2.6-flash", messages("overflow"))
    assert len(seen) == 16
    with pytest.raises(PilotStopped, match="output_token_parameter"):
        services.complete_for_model("mimo-v2.6-flash", messages("too many"), max_tokens=16385)

    class Clock:
        def __init__(self):
            self.now = 0.0
        def __call__(self):
            return self.now

    clock = Clock()
    timed, timed_seen, _ledger = build(tmp_path / "clock", monkeypatch, [chat("x", "glm-5.3-flash")], clock=clock)
    assert timed.budget_remaining("glm-5.3-flash")["seconds"] == 1200
    clock.now = 100
    assert timed.budget_remaining("glm-5.3-flash")["seconds"] == 1100
    clock.now = 1200
    with pytest.raises(PilotStopped, match="unit_seconds_exhausted"):
        timed.complete_for_model("glm-5.3-flash", messages())
    assert timed_seen == []

    poor_opening = {model: 0 for model in MODEL_POOL}
    poor_opening["glm-5.3-flash"] = PER_MODEL_LIMIT_MICROCNY - 500
    poor, poor_seen, _ledger = build(tmp_path / "poor", monkeypatch, [chat("x", "glm-5.3-flash")],
                                     opening=poor_opening, ceiling=1000)
    with pytest.raises(PilotStopped, match="model_cny_cap_exhausted"):
        poor.complete_for_model("glm-5.3-flash", messages())
    assert poor_seen == []

    tools, _seen, _ledger = build(tmp_path / "tools", monkeypatch, [])
    for _ in range(120):
        assert tools.list_files() == []
    with pytest.raises(PilotStopped, match="unit_resource_exhausted"):
        tools.list_files()
    assert tools.tools == 120


def test_policy_reads_service_remaining_and_skips_an_exhausted_model(tmp_path, monkeypatch):
    opening = {model: 0 for model in MODEL_POOL}
    opening["deepseek-flash"] = PER_MODEL_LIMIT_MICROCNY
    services, _seen, _ledger = build(tmp_path, monkeypatch, [], opening=opening)
    assert services.budget_remaining("deepseek-flash")["cny"] == 0
    assert services.budget_remaining()["cny"] == 0
    assert services.budget_remaining("mimo-v2.6-flash")["cny"] == 50
    org = Organization(":memory:")
    policy = ConstrainedThompsonPolicy({}, services.budget_remaining)
    selected, trace = policy.select(org, {}, list(MODEL_POOL), "type=generic|complexity=na|uncertainty=na",
                                    (), "user", random.Random(0))
    org.conn.close()
    assert selected != "deepseek-flash"
    assert trace["scores"]["deepseek-flash"]["feasible"] is False
    assert "cny" in trace["scores"]["deepseek-flash"]["violated_resources"]
    services, seen, ledger = build(tmp_path / "unknown", monkeypatch, [{"nope": True}])
    with pytest.raises(PilotStopped, match="provider_outcome_charged"):
        services.complete_for_model("mimo-v2.6-flash", messages())
    assert ledger.snapshot()["unresolved_request_ids"] == []
    blocked = Organization(":memory:")
    try:
        selected, _trace = ConstrainedThompsonPolicy({}, services.budget_remaining).select(
            blocked, {}, ["mimo-v2.6-flash"], "type=generic|complexity=na|uncertainty=na",
            (), "user", random.Random(0))
    finally:
        blocked.conn.close()
    assert selected == "mimo-v2.6-flash"
    assert len(seen) == 1


def test_syntax_validator_uses_compile_and_ignores_unattributed_writes(tmp_path, monkeypatch):
    calls = []

    def runner(argv, timeout_s):
        calls.append((argv, timeout_s))
        (tmp_path / "state" / "workspace" / "sneak.py").write_text("print('sneak')\n")
        assert "--network=none" in argv
        assert "DEEPSEEK_API_KEY" not in argv and "MIMO_API_KEY" not in argv and "ZAI_API_KEY" not in argv
        rendered = "\n".join(argv)
        for secret in KEYS.values():
            assert secret not in rendered
        image_at = argv.index("example.invalid/alo:test")
        assert argv.index("--network=none") < image_at
        assert argv[image_at + 1] == "python"
        return {"returncode": 0, "stdout": "ok", "stderr": ""}

    # The runner closure captures tmp_path before StudyServices creates the workspace.
    services, _seen, _ledger = build(tmp_path, monkeypatch, [])
    sandbox = DockerSandbox(services.workspace.root, image="example.invalid/alo:test", runner=runner)
    services.sandbox = sandbox
    written = services.write_file("ok.py", "import pathlib\npathlib.Path('marker').write_text('x')\n")
    assert written["path"] == "ok.py"
    evidence = services.local_criterion(role_id="n", model_id="m", output="I passed", context={})
    assert evidence["status"] == "pass" and evidence["reliable"] is True
    assert evidence["scope"] == "node_local"
    assert evidence["criterion_id"] == "python_write_file_compile_syntax_v1"
    assert not (services.workspace.root / "marker").exists()
    assert "marker" not in json.dumps(evidence)
    assert services.local_criterion(role_id="n", model_id="m", output="", context={})["status"] == "unknown"

    services.write_file("broken.py", "def (\n")
    failed = services.local_criterion(role_id="n", model_id="m", output="pass", context={})
    assert failed["status"] == "fail" and failed["reliable"] is True
    assert "def (" not in json.dumps(failed)

    services.write_file("notes.txt", "not python")
    assert services.local_criterion(role_id="n", model_id="m", output="", context={})["status"] == "unknown"
    with pytest.raises(ProtocolError, match="workspace_path_rejected"):
        services.write_file("../outside.py", "print(1)\n")
    assert not (tmp_path / "outside.py").exists()

    result = services.command(["python", "-c", "print(1)"], timeout_s=5)
    assert result["returncode"] == 0
    assert (services.workspace.root / "sneak.py").is_file()
    assert services.local_criterion(role_id="n", model_id="m", output="", context={})["status"] == "unknown"
    assert calls and calls[0][1] == 5
    with pytest.raises(ProtocolError, match="command_timeout"):
        services.command(["python", "-c", "print(1)"], timeout_s=61)
    with pytest.raises(ProtocolError, match="unsupported_execute_language"):
        services.execute_code("ruby", "puts 1", timeout_s=5)
    assert len(calls) == 1

    bare, _seen, _ledger = build(tmp_path / "bare", monkeypatch, [])
    with pytest.raises(PilotStopped, match="sandbox_not_configured"):
        bare.command(["python", "--version"])
    assert bare.tools == 0
    unready = DockerSandbox(bare.workspace.root, image="example.invalid/alo:test")
    bare.sandbox = unready
    with pytest.raises(PilotStopped, match="sandbox_not_configured"):
        bare.command(["python", "--version"])
    assert bare.tools == 0

    def fail_daemon(argv, timeout_s):
        return {"returncode": 125, "stdout": "", "stderr": "Cannot connect to the Docker daemon"}

    daemon_box = DockerSandbox(services.workspace.root, image="example.invalid/alo:test", runner=fail_daemon)
    services.sandbox = daemon_box
    before_tools = services.tools
    with pytest.raises(PilotStopped, match="tool_outcome_unknown"):
        services.command(["python", "--version"], timeout_s=5)
    with pytest.raises(PilotStopped, match="unresolved"):
        services.list_files()
    assert services.tools == before_tools


def test_tool_loop_uses_the_metered_service(tmp_path, monkeypatch):
    services, seen, _ledger = build(tmp_path, monkeypatch, [
        chat('{"final":"submitted work"}', "mimo-v2.6-flash"),
    ])
    result = persistent_tool_loop(BoundModel(services, "mimo-v2.6-flash"), "public task", [], max_turns=2)
    assert result == {"status": "completed", "summary": "submitted work"}
    assert len(seen) == 1
    assert services.resources()["observed_input_tokens"] == 3
    assert services.resources()["estimated_cost_cny"] == 1 / 1_000_000


def test_permit_network_posts_through_a_replaced_urlopen(tmp_path, monkeypatch):
    captured = {}

    class Response:
        def read(self):
            return json.dumps(chat("from-mock", "glm-5.3-flash")).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["data"] = request.data.decode()
        captured["authorization"] = request.get_header("Authorization")
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("ZAI_API_KEY", KEYS["ZAI_API_KEY"])
    for name in ("DEEPSEEK_API_KEY", "MIMO_API_KEY"):
        monkeypatch.setenv(name, KEYS[name])
    services, _seen, _ledger = build(tmp_path, monkeypatch, [], permit_network=True)
    result = services.complete_for_model("glm-5.3-flash", messages("ping"), max_tokens=32)
    assert result["text"] == "from-mock"
    assert captured["url"] == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert captured["authorization"].startswith("Bearer ")
    assert KEYS["ZAI_API_KEY"] not in captured["data"]
    assert '"max_tokens": 32' in captured["data"] or '"max_tokens":32' in captured["data"]
    assert '"type": "enabled"' in captured["data"] or '"type":"enabled"' in captured["data"]


def _request_row(ledger, request_id):
    import sqlite3
    row = sqlite3.connect(ledger.path).execute(
        "SELECT held, held_tokens, held_seconds_ms, status FROM requests WHERE id=?",
        (request_id,)).fetchone()
    return {"held": row[0], "held_tokens": row[1], "held_seconds_ms": row[2], "status": row[3]}


def test_known_receipt_over_the_unit_token_cap_settles_and_stops_only_the_unit(tmp_path, monkeypatch):
    limits = PROTOCOL_UNIT_LIMITS["m1_b0"]
    services, _seen, ledger = build(
        tmp_path, monkeypatch,
        [chat("answer", "mimo-v2.6-flash", prompt=80, completion=20)],
        prior_tokens=limits["tokens"] - 30, ceiling=1000, shard_id="0/4")
    with pytest.raises(PilotStopped, match="unit_resource_exhausted"):
        services.complete_for_model("mimo-v2.6-flash", messages(), max_tokens=16)
    row = _request_row(ledger, "req-m1_b0-1")
    assert row["status"] == "settled"
    assert row["held"] == 1
    assert row["held_tokens"] == 100
    assert ledger.snapshot()["unresolved_request_ids"] == []
    assert ledger.blocks_dispatch("1/4") is False
    limits = PROTOCOL_UNIT_LIMITS["m1_b0"]
    ledger.reserve(
        "req-other-shard", "mimo-v2.6-flash", 1, payload_sha256="b" * 64,
        tokens=10, seconds_ms=10, unit_id="other-unit", unit_request_cap=limits["requests"],
        shard_id="1/4")
    diag = tmp_path / "diagnostics" / "req-m1_b0-1.json"
    assert not diag.exists()


def test_known_receipt_over_the_unit_seconds_cap_settles_and_stops_only_the_unit(tmp_path, monkeypatch):
    limits = PROTOCOL_UNIT_LIMITS["m1_b0"]
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    services, _seen, ledger = build(
        tmp_path, monkeypatch, [],
        prior_seconds_ms=limits["seconds"] * 1000 - 50,
        ceiling=1000, shard_id="0/4", clock=now)

    def sender(url, body, headers):
        del url, headers
        clock["t"] = 0.2
        return chat("answer", body["model"], prompt=2, completion=2)

    services.transport._sender = sender
    with pytest.raises(PilotStopped, match="unit_resource_exhausted"):
        services.complete_for_model("mimo-v2.6-flash", messages(), max_tokens=16)
    row = _request_row(ledger, "req-m1_b0-1")
    assert row["status"] == "settled"
    assert row["held"] == 1
    assert row["held_seconds_ms"] == 200
    assert row["held_tokens"] == 4
    assert ledger.snapshot()["unresolved_request_ids"] == []
    assert ledger.blocks_dispatch("1/4") is False
