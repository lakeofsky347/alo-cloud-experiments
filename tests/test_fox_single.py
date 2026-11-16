"""Offline fox-single-v1 checks. No socket and no real API key."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from alo_cloud.contracts import MODEL_POOL
from alo_cloud.fox_pool import (
    FOX_CHAT_URL,
    FOX_MODELS,
    FOX_USER_AGENT,
    MIMO_CONTEXT_HARD_MAX,
    TEST_EXTRAPOLATION_INSTANCES,
    FoxModel,
    apply_context_limit,
    auth_headers,
    estimate_prompt_tokens,
    estimate_size,
    model_settings,
    resolve_model,
    select_tasks,
    suspected_truncation,
)
from alo_cloud.fox_single import (
    EXPERIMENT,
    FoxServices,
    FoxStopped,
    _find_resolved,
    aggregate_report,
    interpret_call,
    main,
    post_with_retries,
    run_schedule,
    unit_eval_dir,
    write_report,
)
from alo_cloud.prices import PRICE_TABLE, UnpricedModel, charge_if_priced, charge_microcny
from alo_cloud.providers import POOL_ENDPOINTS, _same_model, model_match_kind
from alo_cloud.r2 import r2_freeze_sha256
from alo_cloud.sandbox import CREDENTIAL_ENV_NAMES
from alo_cloud.swebench_lite_frame import SWEBENCH_LITE_FRAME
from alo_cloud.workspace_prep import docker_argv_deletes_images, seed_image_tree


FROZEN_R2 = "77bddfbfd3472ae4c0d280c3dacf20810f11b5f9b5c247c17650629ebf9e241f"
WHEN = datetime(2026, 11, 13, 8, 0, tzinfo=timezone.utc)
DEV_FIRST = "sqlfluff__sqlfluff-1625"


def _clock():
    return WHEN


def _completion(model, content, usage, **extra):
    payload = {
        "model": model,
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": usage,
    }
    payload.update(extra)
    return payload


def test_freeze_and_direct_pool_unchanged():
    assert r2_freeze_sha256() == FROZEN_R2
    assert set(POOL_ENDPOINTS) == set(MODEL_POOL) == set(PRICE_TABLE)
    assert "FOXROUTER_API_KEY" in CREDENTIAL_ENV_NAMES
    internals = [row.internal for row in FOX_MODELS]
    assert internals == [
        "fox/deepseek-flash", "fox/kimi-k3.1-pro", "fox/qwen3.8-max",
        "fox/qwen3.8-flash", "fox/glm-5.3-flash", "fox/mimo-v2.6-flash",
    ]
    assert all(name not in MODEL_POOL and name not in PRICE_TABLE for name in internals)
    assert resolve_model("deepseek-flash").upstream == "deepseek-v4.1-flash-1017"
    assert resolve_model("deepseek-flash").report_name == "deepseek-flash"
    with pytest.raises(ValueError, match="use_fox_namespace"):
        resolve_model("glm-5.3-flash")
    with pytest.raises(ValueError, match="use_fox_namespace"):
        resolve_model("mimo-v2.6-flash")


def test_task_selection_and_size_estimate_does_not_call_out():
    dev = select_tasks("dev", allow_all=True)
    test = select_tasks("test", allow_all=True)
    both = select_tasks("both", allow_all=True)
    assert len(SWEBENCH_LITE_FRAME) == 300
    assert len(dev) == 23 and len(test) == 300 and len(both) == 323
    assert {row[2] for row in dev} == {"dev"}
    assert {row[2] for row in test} == {"test"}
    assert {row[0] for row in dev}.isdisjoint({row[0] for row in test})
    limited = select_tasks("test", limit=2)
    assert [row[0] for row in limited] == [row[0] for row in test[:2]]
    picked = select_tasks("dev", instance_ids=["pydicom__pydicom-901", "pydicom__pydicom-901"])
    assert [row[0] for row in picked] == ["pydicom__pydicom-901"]
    with pytest.raises(ValueError, match="instance_not_in_split"):
        select_tasks("test", instance_ids=["pydicom__pydicom-901"])
    with pytest.raises(ValueError, match="selection_required"):
        select_tasks("dev")
    dev_size = estimate_size(split="dev", allow_all=True)
    test_size = estimate_size(split="test", allow_all=True)
    assert dev_size["units"] == 6 * 23 == 138
    assert test_size["units"] == 6 * 300 == 1800
    assert dev_size["api_calls"] == 0 and test_size["api_calls"] == 0
    one = estimate_size(split="dev", models=["fox/deepseek-flash"], limit=2)
    assert one["units"] == 2 and one["models"] == 1


def test_user_agent_header_and_upstream_id(monkeypatch):
    monkeypatch.setenv("FOXROUTER_API_KEY", "unit-test-fox-credential")
    seen = {}

    def sender(url, body, headers, timeout_s):
        seen["url"] = url
        seen["body"] = body
        seen["headers"] = dict(headers)
        usage = {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
        return 200, _completion(body["model"], '{"final":"ok"}', usage)

    model = resolve_model("fox/deepseek-flash")
    result, retries, body = post_with_retries(
        model, [{"role": "user", "content": "hi"}], 16384,
        key="unit-test-fox-credential", sender=sender, timeout_s=5,
        sleep=lambda _seconds: None, clock=_clock, max_attempts=2)
    assert seen["url"] == FOX_CHAT_URL
    assert seen["headers"]["User-Agent"] == FOX_USER_AGENT == "curl/8.5.0"
    assert seen["headers"]["Authorization"] == "Bearer unit-test-fox-credential"
    assert seen["body"]["model"] == "deepseek-v4.1-flash-1017"
    assert body["model"] == "deepseek-v4.1-flash-1017"
    assert retries == []
    assert result["status"] == 200
    assert auth_headers("unit-test-fox-credential")["User-Agent"] == FOX_USER_AGENT


def test_retry_log_records_status_cloudflare_code_and_timestamp():
    attempts = {"n": 0}
    slept = []

    def sender(url, body, headers, timeout_s):
        del url, headers, timeout_s
        attempts["n"] += 1
        if attempts["n"] == 1:
            return 502, b'{"error": {"code": 1010, "message": "blocked"}}'
        if attempts["n"] == 2:
            raise TimeoutError("slow")
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        return 200, _completion(body["model"], '{"final":"ok"}', usage)

    model = resolve_model("fox/qwen3.8-flash")
    result, retries, _body = post_with_retries(
        model, [{"role": "user", "content": "hi"}], 128,
        key="", sender=sender, timeout_s=5, sleep=slept.append, clock=_clock, max_attempts=4)
    assert result["status"] == 200
    assert slept == [1.0, 2.0]
    assert [row["status"] for row in retries] == [502, None]
    assert retries[0]["cloudflare_error"] == "1010"
    assert retries[0]["timestamp"] == "2026-11-13T08:00:00Z"
    assert retries[1]["kind"] == "timeout"
    assert retries[1]["timestamp"] == "2026-11-13T08:00:00Z"

    def forbidden(url, body, headers, timeout_s):
        del url, body, headers, timeout_s
        return 403, "error code: 1010"

    result, retries, _body = post_with_retries(
        model, [{"role": "user", "content": "hi"}], 128,
        key="", sender=forbidden, timeout_s=5, sleep=slept.append, clock=_clock, max_attempts=3)
    assert result["status"] == 403
    assert retries == []
    row = interpret_call(model, [], 128, result, retries, estimated=10)
    assert row["failure"] == "provider_error"
    assert row["cloudflare_error"] == "1010"


def test_mimo_context_cap_never_sends_over_limit():
    model = resolve_model("fox/mimo-v2.6-flash")
    assert model.context_token_cap < MIMO_CONTEXT_HARD_MAX
    with pytest.raises(ValueError, match="mimo_context_cap_not_below_70k"):
        model_settings(model, context_caps={"fox/mimo-v2.6-flash": MIMO_CONTEXT_HARD_MAX})
    huge = [{"role": "user", "content": "z" * (MIMO_CONTEXT_HARD_MAX * 4)}]
    sent, record = apply_context_limit(huge, model.context_token_cap, "fail")
    assert sent is None and record["context_action"] == "context_limit"
    assert estimate_prompt_tokens(huge) > MIMO_CONTEXT_HARD_MAX
    called = {"n": 0}

    def transport(spec, messages, max_tokens):
        del spec, messages, max_tokens
        called["n"] += 1
        raise AssertionError("over-cap prompt was sent")

    services = FoxServices(
        Path("/tmp") / "fox-mimo-cap-should-not-persist", model,
        settings=model_settings(model), transport=transport,
        limits={"requests": 4, "tokens": 1000, "tools": 4, "seconds": 30})
    with pytest.raises(FoxStopped, match="context_limit"):
        services.complete_for_model(
            model.internal, huge, max_tokens=4096)
    assert called["n"] == 0
    assert services.calls[0]["failure"] == "context_limit"
    assert services.calls[0]["context_cap"] == 60_000

    older = "old " * 8000
    messages = [
        {"role": "system", "content": "keep"},
        {"role": "user", "content": older},
        {"role": "user", "content": "latest"},
    ]
    cap = estimate_prompt_tokens([messages[0], messages[2]]) + 5
    kept, trimmed = apply_context_limit(messages, cap, "trim")
    assert trimmed["context_action"] == "trimmed"
    assert kept is not None and estimate_prompt_tokens(kept) <= cap
    assert all(item.get("content") != older for item in kept)
    assert suspected_truncation(100_000, 25668) is True
    assert suspected_truncation(8_000, 3_000) is False


def test_raw_usage_cost_and_suspected_truncation_are_recorded():
    model = resolve_model("fox/glm-5.3-flash")
    usage = {
        "prompt_tokens": 8,
        "completion_tokens": 512,
        "total_tokens": 520,
        "cost": 9,
    }
    payload = _completion(
        model.upstream, "hi", usage, cost=1.25, credit=4, quota={"left": 3})
    payload["choices"][0]["message"]["reasoning_content"] = "thought"
    result = {"status": 200, "payload": payload, "raw_text": ""}
    row = interpret_call(model, [], 16384, result, [], estimated=8)
    assert row["raw_usage"] == usage
    assert row["cost"] == 1.25 and row["credit"] == 4 and row["quota"] == {"left": 3}
    assert row["usage_quirk"] == "completion_tokens_exactly_512"
    assert row["reasoning_tokens"] is None
    assert row["reasoning_content_present"] is True
    assert row["upstream_model"] == "glm-5.3-flash"
    assert "cny" not in row and "usd" not in row

    kimi = resolve_model("fox/kimi-k3.1-pro")
    kimi_usage = {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
    kimi_payload = _completion(kimi.upstream, "", kimi_usage)
    kimi_payload["choices"][0]["message"]["reasoning_content"] = "hidden"
    kimi_row = interpret_call(
        kimi, [], 16384, {"status": 200, "payload": kimi_payload, "raw_text": ""}, [], estimated=4)
    assert kimi_row["reasoning_tokens"] is None
    assert kimi_row["reasoning_content_present"] is True
    assert "reasoning_tokens" not in kimi_row["raw_usage"]

    mimo = resolve_model("fox/mimo-v2.6-flash")
    seen = {}

    def transport(spec, messages, max_tokens):
        seen["messages"] = messages
        seen["max_tokens"] = max_tokens
        reported = {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}
        body = _completion(spec.upstream, '{"final":"x"}', reported)
        return {"status": 200, "payload": body, "raw_text": ""}, []

    services = FoxServices(
        Path("/tmp") / "fox-trunc-should-not-persist", mimo,
        settings=model_settings(mimo), transport=transport,
        limits={"requests": 2, "tokens": 100_000, "tools": 2, "seconds": 30})
    services.complete_for_model(
        mimo.internal, [{"role": "user", "content": "y" * 40_000}], max_tokens=4096)
    assert seen["max_tokens"] == 16384
    assert services.calls[0]["suspected_truncation"] is True
    assert services.calls[0]["prompt_tokens"] == 100
    assert services.calls[0]["raw_usage"]["prompt_tokens"] == 100
    assert estimate_prompt_tokens(seen["messages"]) <= mimo.context_token_cap


def test_missing_price_does_not_block_and_optional_cap_uses_raw_cost(tmp_path, monkeypatch):
    assert charge_if_priced("fox/deepseek-flash", {"input_tokens": 10, "output_tokens": 10}) is None
    with pytest.raises(UnpricedModel):
        charge_microcny("fox/glm-5.3-flash", {"input_tokens": 10, "output_tokens": 10})
    assert charge_microcny("deepseek-flash", {"input_tokens": 1_000_000, "output_tokens": 0}) == 2_000_000

    def explode(model, usage):
        raise AssertionError("price table must not be required")

    monkeypatch.setattr("alo_cloud.prices.charge_microcny", explode)
    assert charge_if_priced("fox/mimo-v2.6-flash", {"input_tokens": 1, "output_tokens": 1}) is None

    def sender(url, body, headers, timeout_s, cost):
        del url, headers, timeout_s
        usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
        extra = {"cost": cost} if cost is not None else {}
        return 200, _completion(body["model"], '{"final":"noop"}', usage, **extra)

    def run(path, *, cost, budget_cap):
        def transport_sender(url, body, headers, timeout_s):
            return sender(url, body, headers, timeout_s, cost)
        return run_schedule(
            path, split="dev", models=["fox/deepseek-flash"], limit=2, live=False,
            sender=transport_sender, turn_cap=2, clock=_clock, sleep=lambda _n: None,
            budget_cap=budget_cap, seed_images=False)

    open_budget = run(tmp_path / "open", cost=None, budget_cap=0.01)
    assert open_budget["wrote"] == 2
    priced = run(tmp_path / "capped", cost=1, budget_cap=0.5)
    assert priced["wrote"] == 1
    rows = [json.loads(line) for line in (tmp_path / "capped" / "records.jsonl").read_text().splitlines()]
    assert rows[0]["list_price"] is None
    assert rows[0]["raw_cost_sum"] == 1
    assert rows[0]["upstream_model"] == "deepseek-v4.1-flash-1017"
    assert "unit-test" not in (tmp_path / "capped" / "manifest.json").read_text()


def test_report_aggregation_and_key_is_not_stored(tmp_path, monkeypatch):
    secret = "unit-test-fox-credential"
    monkeypatch.setenv("FOXROUTER_API_KEY", secret)
    calls_seen = []

    def sender(url, body, headers, timeout_s):
        del url, timeout_s
        calls_seen.append(dict(headers))
        blob = json.dumps(body["messages"])
        if '"role": "assistant"' in blob:
            content = json.dumps({"final": "fixed the return"})
        else:
            content = json.dumps({
                "tool": "edit_file",
                "arguments": {"path": "bug.py", "old_str": "return 0", "new_str": "return 1"},
            })
        usage = {"prompt_tokens": 20, "completion_tokens": 11, "total_tokens": 31}
        return 200, _completion(body["model"], content, usage, cost=0.5, credit=2)

    def scorer(command, row):
        split_at = command.index("--split")
        assert command[split_at + 1] == "dev"
        assert "return 1" in row["model_patch"]
        return {"status": "scored", "resolved": True}

    run_dir = tmp_path / "fox-single-v1"
    result = run_schedule(
        run_dir, split="dev", models=["deepseek-flash"], limit=1, live=False,
        sender=sender, scorer=scorer, turn_cap=4, clock=_clock, sleep=lambda _n: None,
        execute=True)
    assert result["wrote"] == 1
    assert calls_seen[0]["User-Agent"] == FOX_USER_AGENT
    assert calls_seen[0]["Authorization"] == "Bearer " + secret
    text_paths = [
        path for path in run_dir.rglob("*")
        if path.is_file() and ".git" not in path.parts and path.suffix in {".json", ".jsonl", ".patch", ".txt", ".md"}
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in text_paths)
    assert secret not in text
    assert "FOXROUTER_API_KEY" in (run_dir / "manifest.json").read_text(encoding="utf-8")
    record = json.loads((run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["report_model"] == "deepseek-flash"
    assert record["internal_model"] == "fox/deepseek-flash"
    assert record["upstream_model"] == "deepseek-v4.1-flash-1017"
    assert record["resolved"] is True
    assert record["failure"] is None
    assert record["max_attempts"] == 1
    assert record["criterion_id"] == "repro-v1"
    summary = write_report(run_dir)
    group = summary["by_model_split"][0]
    assert group["model"] == "deepseek-flash"
    assert group["split"] == "dev"
    assert group["instances"] == 1 and group["resolved"] == 1
    assert group["resolved_rate"] == 1
    assert group["prompt_tokens"] == 40
    assert group["completion_tokens"] == 22
    assert group["raw_cost_sum"] == 1
    assert group["raw_credit_sum"] == 4
    assert group["retries"] == 0
    assert group["cost_units"] == "as_returned"

    mixed = aggregate_report([
        {"experiment": EXPERIMENT, "report_model": "fox/glm-5.3-flash", "internal_model": "fox/glm-5.3-flash",
         "upstream_model": "glm-5.3-flash", "split": "test", "resolved": False, "failure": "tests_fail",
         "prompt_tokens": 10, "completion_tokens": 512, "reasoning_tokens": None, "retries": 2,
         "raw_cost_sum": 3, "suspected_truncation": True},
        {"experiment": EXPERIMENT, "report_model": "fox/glm-5.3-flash", "internal_model": "fox/glm-5.3-flash",
         "upstream_model": "glm-5.3-flash", "split": "test", "resolved": True, "failure": None,
         "prompt_tokens": 5, "completion_tokens": 1, "reasoning_tokens": 7, "retries": 0,
         "raw_cost_sum": None, "suspected_truncation": False},
        {"experiment": EXPERIMENT, "report_model": "fox/glm-5.3-flash", "internal_model": "fox/glm-5.3-flash",
         "upstream_model": "glm-5.3-flash", "split": "dev", "resolved": None, "failure": "timeout",
         "prompt_tokens": 1, "completion_tokens": 0, "reasoning_tokens": None, "retries": 3,
         "raw_cost_sum": None, "suspected_truncation": False},
    ], [])
    by_split = {row["split"]: row for row in mixed["by_model_split"]}
    assert by_split["test"]["instances"] == 2
    assert by_split["test"]["resolved"] == 1
    assert by_split["test"]["resolved_rate"] == 0.5
    assert by_split["test"]["prompt_tokens"] == 15
    assert by_split["test"]["completion_tokens"] == 513
    assert by_split["test"]["reasoning_tokens"] == 7
    assert by_split["test"]["raw_cost_sum"] == 3
    assert by_split["test"]["retries"] == 2
    assert by_split["test"]["failure_taxonomy"]["tests_fail"] == 1
    assert by_split["test"]["suspected_truncation"] == 1
    assert by_split["dev"]["failure_taxonomy"]["timeout"] == 1
    assert by_split["dev"]["reasoning_tokens"] is None


def test_dev_is_the_default_split_and_test_300_extrapolation_is_labeled(tmp_path, capsys):
    assert TEST_EXTRAPOLATION_INSTANCES == 300 == len(SWEBENCH_LITE_FRAME)
    assert main(["estimate", "--all"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["split"] == "dev"
    assert payload["instances"] == 23
    assert payload["units"] == 138
    assert payload["api_calls"] == 0

    run_dir = tmp_path / "observed"
    run_dir.mkdir()
    rows = []
    for prompt, cost, credit in ((10, 1, 2), (30, 3, 4)):
        rows.append({
            "experiment": "fox-single-v1",
            "report_model": "deepseek-flash",
            "internal_model": "fox/deepseek-flash",
            "upstream_model": "deepseek-v4.1-flash-1017",
            "split": "dev",
            "instance_id": f"inst-{prompt}",
            "prompt_tokens": prompt,
            "completion_tokens": 4,
            "reasoning_tokens": None,
            "raw_cost_sum": cost,
            "raw_credit_sum": credit,
        })
    (run_dir / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert main(["estimate", "--from-run-dir", str(run_dir)]) == 0
    estimate = json.loads(capsys.readouterr().out)["test_300_extrapolation"]
    assert estimate["label"] == "estimate"
    assert estimate["measured_test_split"] is False
    assert estimate["target_instances"] == 300
    deepseek = next(row for row in estimate["by_model"] if row["model"] == "deepseek-flash")
    assert deepseek["observed_instances"] == 2
    assert deepseek["observed_splits"] == ["dev"]
    spread = (200 ** 0.5)
    assert deepseek["per_instance"]["prompt_tokens"]["mean"] == pytest.approx(20)
    assert deepseek["per_instance"]["prompt_tokens"]["stdev"] == pytest.approx(spread)
    projected = deepseek["test_300_estimate"]["prompt_tokens"]
    assert projected["total_at_mean"] == pytest.approx(6000)
    assert projected["spread"] == pytest.approx(300 * spread)
    assert deepseek["test_300_estimate"]["raw_cost"]["total_at_mean"] == pytest.approx(600)
    assert deepseek["test_300_estimate"]["raw_credit"]["total_at_mean"] == pytest.approx(900)
    assert deepseek["test_300_estimate"]["reasoning_tokens"]["available"] is False
    missing = next(row for row in estimate["by_model"] if row["model"] == "fox/kimi-k3.1-pro")
    assert missing["observed_instances"] == 0
    assert missing["test_300_estimate"]["raw_cost"]["available"] is False


def test_resolved_flag_stays_inside_the_unit_eval_dir(tmp_path):
    instance = "sqlfluff__sqlfluff-1625"
    unit_a = f"fox-single-v1.fox-deepseek-flash.{instance}"
    unit_b = f"fox-single-v1.fox-kimi-k3.1-pro.{instance}"
    dir_a = unit_eval_dir(tmp_path, unit_a)
    dir_b = unit_eval_dir(tmp_path, unit_b)
    (dir_a / "report.json").write_text(
        json.dumps({instance: {"resolved": True}}), encoding="utf-8")
    (dir_b / "report.json").write_text(
        json.dumps({instance: {"resolved": False}}), encoding="utf-8")
    (tmp_path / "eval" / "aaa-shared.json").write_text(
        json.dumps({instance: {"resolved": True}}), encoding="utf-8")
    assert dir_a != dir_b
    assert _find_resolved(dir_a, instance) is True
    assert _find_resolved(dir_b, instance) is False


def test_resume_keeps_the_recorded_budget(tmp_path):
    def sender(url, body, headers, timeout_s):
        del url, headers, timeout_s
        usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
        return 200, _completion(body["model"], '{"final":"noop"}', usage, cost=1)

    kwargs = dict(
        split="dev", models=["fox/deepseek-flash"], limit=2, live=False, sender=sender,
        turn_cap=2, clock=_clock, sleep=lambda _n: None, budget_cap=0.5, seed_images=False,
    )
    run_dir = tmp_path / "fox-budget"
    first = run_schedule(run_dir, **kwargs)
    assert first["wrote"] == 1
    second = run_schedule(run_dir, **kwargs)
    assert second["wrote"] == 0
    lines = (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["raw_cost_sum"] == 1


def test_review_fixes_for_identity_limits_resume_and_unknown_scores(tmp_path):
    model = resolve_model("fox/glm-5.3-flash")
    usage = {"prompt_tokens": 8, "completion_tokens": 512, "total_tokens": 520, "reasoning_tokens": 520}
    good = _completion(model.upstream, "hi", usage)
    row = interpret_call(model, [], 16384, {"status": 200, "payload": good, "raw_text": ""}, [], estimated=8)
    assert row["ok"] is True
    assert row["reasoning_tokens"] == 520
    foreign = _completion("some-other-model", "hi", usage)
    rejected = interpret_call(
        model, [], 16384, {"status": 200, "payload": foreign, "raw_text": ""}, [], estimated=8)
    assert rejected["ok"] is False
    assert rejected["failure"] == "provider_error"
    aliased = _completion("vendor/glm-5.3-flash", "hi", usage)
    assert interpret_call(
        model, [], 16384, {"status": 200, "payload": aliased, "raw_text": ""}, [], estimated=8)["ok"] is True

    attempts = {"n": 0}

    def always_fail(url, body, headers, timeout_s):
        del url, body, headers, timeout_s
        attempts["n"] += 1
        return 502, b"error code: 520"

    _result, retries, _body = post_with_retries(
        model, [{"role": "user", "content": "hi"}], 32, key="", sender=always_fail,
        timeout_s=1, sleep=lambda _n: None, clock=_clock, max_attempts=4, max_sends=2)
    assert attempts["n"] == 2
    assert len(retries) == 2

    calls = []

    def transport(spec, messages, max_tokens, max_sends=None):
        calls.append(max_sends)
        body = _completion(spec.upstream, '{"final":"x"}', {
            "prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10,
        })
        return {"status": 200, "payload": body, "raw_text": "", "sends": 1,
                "timestamp": "2026-11-13T08:00:00Z"}, []

    services = FoxServices(
        tmp_path / "limits", model, settings=model_settings(model), transport=transport,
        limits={"requests": 4, "tokens": 10, "tools": 4, "seconds": 30})
    services.complete_for_model(model.internal, [{"role": "user", "content": "hi"}], max_tokens=32)
    assert calls == [4]
    assert services.budget_remaining()["tokens"] == 0
    with pytest.raises(FoxStopped, match="resource_limit"):
        services.complete_for_model(model.internal, [{"role": "user", "content": "hi"}], max_tokens=32)
    assert calls == [4]

    frozen = {"t": 0.0}
    stalled = FoxServices(
        tmp_path / "wall", model, settings=model_settings(model), transport=transport,
        limits={"requests": 4, "tokens": 100, "tools": 4, "seconds": 5},
        clock=lambda: frozen["t"])
    frozen["t"] = 6.0
    with pytest.raises(FoxStopped, match="resource_limit"):
        stalled.complete_for_model(model.internal, [{"role": "user", "content": "hi"}], max_tokens=32)

    def forbidden(url, body, headers, timeout_s):
        del url, body, headers, timeout_s
        return 403, "error code: 1010"

    run_dir = tmp_path / "fail-log"
    run_schedule(
        run_dir, split="dev", models=["fox/glm-5.3-flash"], limit=1, live=False,
        sender=forbidden, turn_cap=2, clock=_clock, sleep=lambda _n: None, seed_images=False)
    failures = [json.loads(line) for line in (run_dir / "failures.jsonl").read_text().splitlines()]
    provider = next(row for row in failures if row.get("kind") == "provider_error" and row.get("http_status") == 403)
    assert provider["cloudflare_error"] == "1010"
    assert provider["timestamp"] == "2026-11-13T08:00:00Z"

    summary = aggregate_report([
        {"experiment": EXPERIMENT, "report_model": "deepseek-flash", "split": "dev",
         "resolved": None, "failure": "eval_unavailable", "prompt_tokens": 1,
         "completion_tokens": 1, "reasoning_tokens": None, "retries": 0},
    ], [])
    group = summary["by_model_split"][0]
    assert group["unscored"] == 1
    assert group["resolved_rate"] is None
    assert group["resolved_rate_bounds"]["lower"] == 0
    assert group["resolved_rate_bounds"]["upper"] == 1
    assert group["resolved_rate_bounds"]["label"] == "estimate_bounds"

    broad = tmp_path / "schedule"
    kwargs = dict(
        split="dev", models=["fox/deepseek-flash"], live=False, turn_cap=2,
        clock=_clock, sleep=lambda _n: None, seed_images=False,
        sender=lambda url, body, headers, timeout_s: (
            200, _completion(body["model"], '{"final":"noop"}',
                             {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4})),
    )
    run_schedule(broad, limit=2, **kwargs)
    manifest = json.loads((broad / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["instances"]) == 2
    run_schedule(broad, limit=1, **kwargs)
    kept = json.loads((broad / "manifest.json").read_text(encoding="utf-8"))
    assert kept["instances"] == manifest["instances"]
    assert kept["units"] >= 2
    with pytest.raises(FoxStopped, match="fox_run_settings_mismatch"):
        run_schedule(broad, limit=1, turn_cap=9, **{k: v for k, v in kwargs.items() if k != "turn_cap"})


def test_image_seed_does_not_delete_or_prune_images(tmp_path):
    assert docker_argv_deletes_images(["docker", "rm", "-f", "alo-seed-1"]) is False
    assert docker_argv_deletes_images(
        ["docker", "create", "--name", "c", "swebench/sweb.eval.x86_64.example:latest"]) is False
    assert docker_argv_deletes_images(["docker", "rmi", "swebench/sweb.eval.x86_64.example:latest"]) is True
    assert docker_argv_deletes_images(["docker", "image", "rm", "swebench/sweb.eval.x86_64.example:latest"]) is True
    assert docker_argv_deletes_images(["docker", "image", "prune", "-a"]) is True
    assert docker_argv_deletes_images(["docker", "system", "prune", "-af"]) is True
    seen = []

    def docker(argv):
        seen.append(list(argv))
        if docker_argv_deletes_images(argv):
            raise AssertionError(argv)

    seed_image_tree(
        "swebench/sweb.eval.x86_64.example:latest", "/testbed", tmp_path / "tree", docker=docker)
    assert [argv[1] for argv in seen] == ["create", "cp", "rm"]
    assert all(argv[1] != "rmi" and "prune" not in argv for argv in seen)
    assert seen[0][-1].startswith("swebench/sweb.eval.x86_64.")


def _billed_usage():
    return {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "reasoning_tokens": 2,
        "cost": 0.4,
    }


def _gateway_call(model, returned, usage=None, content='{"final":"ok"}'):
    if usage is None:
        usage = _billed_usage()
    payload = _completion(returned, content, usage)
    return interpret_call(
        model, [], 16384,
        {"status": 200, "payload": payload, "raw_text": "", "timestamp": "2026-11-13T08:00:00Z", "sends": 1},
        [], estimated=8)


def test_gateway_model_ids_accept_cosmetic_aliases_and_reject_other_models():
    assert r2_freeze_sha256() == FROZEN_R2
    kimi = resolve_model("fox/kimi-k3.1-pro")
    assert kimi.accepted_aliases == ("kimi-k3p1-pro", "kimi-k3.1-pro-ta")
    assert all(
        row.accepted_aliases == ()
        for row in FOX_MODELS if row.internal != "fox/kimi-k3.1-pro")
    accepted = (
        ("fox/deepseek-flash", "deepseek-v4.1-flash"),
        ("fox/kimi-k3.1-pro", "kimi-k3-1-pro"),
        ("fox/qwen3.8-max", "qwen/qwen3.8-max"),
        ("fox/qwen3.8-flash", "qwen3.8-flash"),
        ("fox/qwen3.8-flash", "qwen/qwen3-8-flash"),
        ("fox/qwen3.8-flash", "qwen3-8-flash"),
        ("fox/glm-5.3-flash", "glm-5.3-flash"),
        ("fox/mimo-v2.6-flash", "mimo-v2.6-flash"),
    )
    for name, returned in accepted:
        model = resolve_model(name)
        row = _gateway_call(model, returned, content='{"tool":"read_file","arguments":{"path":"bug.py"}}')
        assert row["ok"] is True, (name, returned)
        assert "failure" not in row
        assert "model_mismatch" not in row
        assert row["returned_model"] == returned
        assert row["upstream_model"] == model.upstream
        assert row["prompt_tokens"] == 11
        assert row["completion_tokens"] == 7
        assert row["total_tokens"] == 18
        assert row["reasoning_tokens"] == 2
        assert row["cost"] == 0.4
        assert row["finish_reason"] == "stop"
        assert row["text"] == '{"tool":"read_file","arguments":{"path":"bug.py"}}'
        if returned == model.upstream:
            assert "model_alias_normalized" not in row
            assert model_match_kind(returned, model.upstream) == "exact"
        else:
            assert row["model_alias_normalized"] == {"requested": model.upstream, "returned": returned}
            assert model_match_kind(returned, model.upstream) == "normalized"

    rejected = (
        ("fox/qwen3.8-max", "qwen3.8-flash"),
        ("fox/qwen3.8-max", "qwen/qwen3-8-flash"),
        ("fox/qwen3.8-flash", "qwen3.8-max"),
        ("fox/kimi-k3.1-pro", "kimi-k3.1"),
        ("fox/kimi-k3.1-pro", "kimi-k3-1"),
        ("fox/deepseek-flash", "deepseek-v4.1-pro"),
        ("fox/deepseek-flash", "deepseek-v4.1-pro-1017"),
        ("fox/deepseek-flash", "deepseek-v4.1-flash-10170"),
        ("fox/deepseek-flash", "deepseek-1017-v4.1-flash"),
        ("fox/glm-5.3-flash", "glm-5.3-pro"),
        ("fox/mimo-v2.6-flash", "mimo-v2.6-pro"),
        ("fox/qwen3.8-flash", "qwen3.8-flash-pro"),
    )
    for name, returned in rejected:
        model = resolve_model(name)
        row = _gateway_call(model, returned)
        assert row["ok"] is False, (name, returned)
        assert row["failure"] == "provider_error"
        assert row["model_mismatch"] is True
        assert "model_alias_normalized" not in row
        assert row["returned_model"] == returned
        assert row["upstream_model"] == model.upstream
        assert row["prompt_tokens"] == 11
        assert row["completion_tokens"] == 7
        assert row["total_tokens"] == 18
        assert row["reasoning_tokens"] == 2
        assert row["cost"] == 0.4
        assert row["raw_usage"]["cost"] == 0.4
        assert _same_model(returned, model.upstream) is False

    assert _same_model("deepseek-v4.1-flash-latest", "deepseek-v4.1-flash-1017") is True
    assert _same_model("deepseek-v4.1-flash-20251017", "deepseek-v4.1-flash") is True
    assert _same_model("xiaomi/MiMo-V2.6-Flash", "mimo-v2.6-flash") is True
    assert _same_model("openai:glm-5.3-flash", "glm-5.3-flash") is True
    assert _same_model("vendor/org:glm-5.3-flash", "glm-5.3-flash") is True
    assert _same_model("deepseek-pro", "deepseek-flash") is False
    assert _same_model("some-other-model", "glm-5.3-flash") is False
    assert model_match_kind("glm-5.3-flash", "glm-5.3-flash") == "exact"
    assert model_match_kind("Kimi-K3.1-Pro", "kimi-k3.1-pro") == "normalized"

    aliased = FoxModel(
        "fox/qwen3.8-flash", "qwen3.8-flash", "fox/qwen3.8-flash", 128_000,
        accepted_aliases=("gateway-qwen-flash-alias",))
    alias_row = _gateway_call(aliased, "gateway-qwen-flash-alias")
    assert alias_row["ok"] is True
    assert alias_row["finish_reason"] == "stop"
    assert alias_row["model_alias_normalized"] == {
        "requested": "qwen3.8-flash", "returned": "gateway-qwen-flash-alias"}
    foreign_alias = _gateway_call(aliased, "gateway-qwen-flash-alias-pro")
    assert foreign_alias["ok"] is False and foreign_alias["failure"] == "provider_error"
    other = resolve_model("fox/qwen3.8-max")
    leaked = _gateway_call(other, "gateway-qwen-flash-alias")
    assert leaked["failure"] == "provider_error"
    with pytest.raises(ValueError, match="invalid_accepted_aliases"):
        FoxModel(
            "fox/glm-5.3-flash", "glm-5.3-flash", "fox/glm-5.3-flash", 128_000,
            accepted_aliases=("",))


def test_kimi_k3p1_alias_accepts_only_that_gateway_spelling():
    """``3p1`` is Kimi's gateway spelling of ``3.1``. ``p`` is not a separator."""
    assert r2_freeze_sha256() == FROZEN_R2
    kimi = resolve_model("fox/kimi-k3.1-pro")
    assert kimi.upstream == "kimi-k3.1-pro"
    assert kimi.accepted_aliases == ("kimi-k3p1-pro", "kimi-k3.1-pro-ta")
    qwen = resolve_model("fox/qwen3.8-max")
    assert qwen.accepted_aliases == ()
    for returned in ("kimi-k3p1-pro", "fw/kimi-k3p1-pro"):
        row = _gateway_call(kimi, returned, content='{"final":"ok"}')
        assert row["ok"] is True, returned
        assert "failure" not in row
        assert "model_mismatch" not in row
        assert row["returned_model"] == returned
        assert row["upstream_model"] == "kimi-k3.1-pro"
        assert row["model_alias_normalized"] == {
            "requested": "kimi-k3.1-pro", "returned": returned}
        assert model_match_kind(returned, kimi.upstream, kimi.accepted_aliases) == "normalized"
        assert model_match_kind(returned, kimi.upstream) == "mismatch"
    for returned in ("kimi-k3p2-pro", "fw/kimi-k3p2-pro"):
        row = _gateway_call(kimi, returned)
        assert row["ok"] is False, returned
        assert row["failure"] == "provider_error"
        assert row["model_mismatch"] is True
        assert "model_alias_normalized" not in row
        assert row["returned_model"] == returned
        assert model_match_kind(returned, kimi.upstream, kimi.accepted_aliases) == "mismatch"
    for returned in ("kimi-k3p1-pro", "fw/kimi-k3p1-pro", "qwen3p8-max"):
        row = _gateway_call(qwen, returned)
        assert row["ok"] is False, returned
        assert row["failure"] == "provider_error"
        assert row["model_mismatch"] is True
        assert "model_alias_normalized" not in row
        assert row["upstream_model"] == "qwen3.8-max"
        assert model_match_kind(returned, qwen.upstream, qwen.accepted_aliases) == "mismatch"


def test_rejected_call_usage_stays_in_the_ledger_and_failure_log(tmp_path):
    calls = {"n": 0}

    def sender(url, body, headers, timeout_s):
        del url, headers, timeout_s
        calls["n"] += 1
        assert body["model"] == "deepseek-v4.1-flash-1017"
        if calls["n"] == 1:
            usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "cost": 1.25}
            content = json.dumps({
                "tool": "edit_file",
                "arguments": {"path": "bug.py", "old_str": "return 0", "new_str": "return 1"},
            })
            return 200, _completion("deepseek-v4.1-flash", content, usage)
        usage = {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9, "cost": 2.5}
        return 200, _completion("qwen3.8-max", '{"final":"nope"}', usage)

    run_dir = tmp_path / "fox-mismatch"
    run_schedule(
        run_dir, split="dev", models=["fox/deepseek-flash"], limit=1, live=False,
        sender=sender, turn_cap=4, clock=_clock, sleep=lambda _n: None, seed_images=False)
    record = json.loads((run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert calls["n"] == 2
    assert record["failure"] == "provider_error"
    assert record["upstream_model"] == "deepseek-v4.1-flash-1017"
    assert record["prompt_tokens"] == 16
    assert record["completion_tokens"] == 7
    assert record["raw_cost_sum"] == pytest.approx(3.75)
    assert record["calls"] == 2
    call_rows = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert call_rows[0]["ok"] is True
    assert call_rows[0]["returned_model"] == "deepseek-v4.1-flash"
    assert call_rows[0]["model_alias_normalized"]["requested"] == "deepseek-v4.1-flash-1017"
    assert call_rows[0]["prompt_tokens"] == 10 and call_rows[0]["cost"] == 1.25
    assert call_rows[1]["ok"] is False
    assert call_rows[1]["failure"] == "provider_error"
    assert call_rows[1]["model_mismatch"] is True
    assert "model_alias_normalized" not in call_rows[1]
    assert call_rows[1]["upstream_model"] == "deepseek-v4.1-flash-1017"
    assert call_rows[1]["returned_model"] == "qwen3.8-max"
    assert call_rows[1]["prompt_tokens"] == 6 and call_rows[1]["cost"] == 2.5
    failures = [json.loads(line) for line in (run_dir / "failures.jsonl").read_text(encoding="utf-8").splitlines()]
    logged = [
        row for row in failures
        if row.get("model_mismatch") is True and row.get("returned_model") == "qwen3.8-max"
    ]
    assert len(logged) >= 1
    assert all(row["upstream_model"] == "deepseek-v4.1-flash-1017" for row in logged)
    assert all(row["kind"] == "provider_error" for row in logged)
    summary = write_report(run_dir)
    group = summary["by_model_split"][0]
    assert group["prompt_tokens"] == 16
    assert group["completion_tokens"] == 7
    assert group["raw_cost_sum"] == pytest.approx(3.75)
    assert group["failure_taxonomy"]["provider_error"] == 1

    def normalized(url, body, headers, timeout_s):
        del url, headers, timeout_s
        usage = {"prompt_tokens": 20, "completion_tokens": 11, "total_tokens": 31, "cost": 0.5}
        return 200, _completion("deepseek-v4.1-flash", '{"final":"noop"}', usage)

    clean = tmp_path / "fox-alias"
    run_schedule(
        clean, split="dev", models=["fox/deepseek-flash"], limit=1, live=False,
        sender=normalized, turn_cap=2, clock=_clock, sleep=lambda _n: None, seed_images=False)
    clean_record = json.loads((clean / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert clean_record["stop_reason"] is None
    assert clean_record["failure"] == "empty_patch"
    assert clean_record["upstream_model"] == "deepseek-v4.1-flash-1017"
    assert clean_record["prompt_tokens"] == 20
    assert clean_record["raw_cost_sum"] == pytest.approx(0.5)
    clean_call = json.loads((clean / "calls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert clean_call["returned_model"] == "deepseek-v4.1-flash"
    assert clean_call["model_alias_normalized"]["returned"] == "deepseek-v4.1-flash"
    assert "model_mismatch" not in clean_call


def test_cli_estimate_and_run_dir_guard(tmp_path, capsys):
    assert main(["estimate", "--split", "dev", "--all"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["units"] == 138 and payload["api_calls"] == 0
    assert main(["estimate", "--split", "test", "--all"]) == 0
    assert json.loads(capsys.readouterr().out)["units"] == 1800
    assert main(["run", "--run-dir", str(tmp_path / "fox"), "--split", "dev", "--limit", "1"]) == 2
    assert "fox_models_required" in capsys.readouterr().out
    assert main([
        "run", "--run-dir", str(tmp_path / "alo-r2-run"), "--split", "dev", "--limit", "1",
        "--models", "fox/deepseek-flash",
    ]) == 2
    assert "fox_run_dir_conflicts_with_frozen_batch" in capsys.readouterr().out
    assert main(["smoke", "--run-dir", str(tmp_path / "smoke"), "--split", "dev", "--limit", "3",
                 "--models", "fox/deepseek-flash"]) == 2
    assert "smoke_limit_is_1_or_2" in capsys.readouterr().out


def test_kimi_ta_alias_is_accepted_and_the_returned_id_is_logged():
    """Foxrouter returns ``kimi-k3.1-pro-ta`` for ``kimi-k3.1-pro`` since 2026-11-16."""
    assert r2_freeze_sha256() == FROZEN_R2
    kimi = resolve_model("fox/kimi-k3.1-pro")
    assert kimi.upstream == "kimi-k3.1-pro"
    assert model_match_kind("kimi-k3.1-pro-ta", kimi.upstream) == "mismatch"
    for returned in ("kimi-k3.1-pro-ta", "fw/kimi-k3.1-pro-ta"):
        row = _gateway_call(kimi, returned, content='{"final":"ok"}')
        assert row["ok"] is True, returned
        assert "failure" not in row
        assert "model_mismatch" not in row
        assert row["returned_model"] == returned
        assert row["upstream_model"] == "kimi-k3.1-pro"
        assert row["model_alias_normalized"] == {
            "requested": "kimi-k3.1-pro", "returned": returned}
        assert model_match_kind(returned, kimi.upstream, kimi.accepted_aliases) == "normalized"
    for returned in ("kimi-k3.1-pro-tb", "kimi-k3.2-pro-ta", "kimi-k3.1-ta"):
        row = _gateway_call(kimi, returned)
        assert row["ok"] is False, returned
        assert row["failure"] == "provider_error"
        assert row["model_mismatch"] is True
        assert row["returned_model"] == returned
    for name in ("fox/qwen3.8-max", "fox/glm-5.3-flash", "fox/mimo-v2.6-flash", "fox/deepseek-flash"):
        row = _gateway_call(resolve_model(name), "kimi-k3.1-pro-ta")
        assert row["ok"] is False, name
        assert row["model_mismatch"] is True
