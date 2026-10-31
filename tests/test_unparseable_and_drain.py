"""Settled unparseable bodies are not budget-unknowns. Drain stops between units."""
import json

import pytest

from alo_cloud.errors import PilotStopped
from alo_cloud.freeze import select_units
from alo_cloud.ledger import main
from alo_cloud.providers import parse_openai_chat_completion
from alo_cloud.run import reclassify_charged_unparseable, run_schedule
from alo_cloud.shardlock import drain_requested


def _final(url, body, headers):
    del url, headers
    return {
        "model": body["model"],
        "choices": [{"finish_reason": "stop", "message": {"content": '{"final":"ok"}'}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def test_mimo_content_list_and_prefixed_model_parse():
    parsed = parse_openai_chat_completion({
        "model": "xiaomi/MiMo-V2.6-Flash",
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": [{"type": "text", "text": "visible"}],
                        "reasoning_content": "trace"},
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7,
                  "reasoning_tokens": "1"},
    }, "mimo-v2.6-flash")
    assert parsed["text"] == "visible"
    assert parsed["reasoning_tokens"] == 1


def test_unparseable_charge_restarts_once_and_does_not_count_as_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    units = select_units("m1", arms=("g1-main",))[:2]

    def bad(url, body, headers):
        del url, headers
        return {"model": body["model"], "choices": []}

    rows = run_schedule(units, tmp_path / "run", live=False, sender=bad)
    assert [row["reason"] for row in rows] == [
        "second_unparseable_interruption", "second_unparseable_interruption"]
    assert all(row["halt_schedule"] is False and row["charged_unparseable"] is True for row in rows)
    assert all(row["unknown_request_id"] for row in rows)
    unknown = tmp_path / "run" / "unknown_outcomes.jsonl"
    assert not unknown.exists() or unknown.read_text(encoding="utf-8").strip() == ""
    counts = json.loads((tmp_path / "run" / "progress" / "unparseable_counts.json").read_text(encoding="utf-8"))
    assert counts["charged_unparseable"] == 4
    assert counts["halt_reason"] is None


def test_connection_reset_still_counts_toward_the_unknown_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    units = select_units("m1", arms=("g1-main",))[:2]

    def fail(url, body, headers):
        del url, body, headers
        raise ConnectionResetError("reset")

    with pytest.raises(PilotStopped, match="unknown_outcome_threshold"):
        run_schedule(units, tmp_path / "run", live=False, sender=fail)
    lines = [json.loads(line) for line in (tmp_path / "run" / "unknown_outcomes.jsonl").read_text().splitlines() if line.strip()]
    assert len(lines) == 4
    assert all(line["request_id"] for line in lines)


def test_reclassify_existing_null_request_rows_is_idempotent(tmp_path):
    run = tmp_path / "run"
    (run / "diagnostics").mkdir(parents=True)
    request_id = "req-e4-m1.SWE-bench_Lite.django__django-11583.g1-strong.B0-3"
    unit = "m1.SWE-bench_Lite.django__django-11583.g1-strong.B0"
    (run / "diagnostics" / f"{request_id}.json").write_text(json.dumps({
        "request_id": request_id, "unit_id": unit, "cause": "unparseable_success",
        "settled": True, "model": "mimo-v2.6-pro",
    }), encoding="utf-8")
    (run / "unknown_outcomes.jsonl").write_text(
        json.dumps({"unit_id": unit, "reason": "provider_outcome_charged", "request_id": None}) + "\n"
        + json.dumps({"unit_id": "other", "reason": "provider_outcome_charged", "request_id": "req-e4-other-1"}) + "\n",
        encoding="utf-8")
    assert reclassify_charged_unparseable(run)["reclassified"] == 1
    left = [json.loads(line) for line in (run / "unknown_outcomes.jsonl").read_text().splitlines() if line.strip()]
    assert len(left) == 1 and left[0]["unit_id"] == "other"
    audit = (run / "progress" / "reclassified_unparseable.jsonl").read_text(encoding="utf-8")
    assert request_id in audit
    assert reclassify_charged_unparseable(run)["reclassified"] == 0
    assert (run / "progress" / "reclassified_unparseable.jsonl").read_text(encoding="utf-8").count(request_id) == 1


def test_drain_finishes_the_current_unit_and_stops_before_the_next(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    run = tmp_path / "run"
    units = [unit for unit in select_units("m1", arms=("g1-main",)) if unit.phase == "B0"][:2]
    calls = {"n": 0}

    def sender(url, body, headers):
        calls["n"] += 1
        if calls["n"] == 1:
            assert main(["drain", "--run-dir", str(run)]) == 0
        return _final(url, body, headers)

    rows = run_schedule(units, run, live=False, sender=sender)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["halt_schedule"] is False
    assert drain_requested(run) is True
    text = (run / "records.jsonl").read_text(encoding="utf-8")
    assert rows[0]["unit_id"] in text
    assert units[1].unit_id not in text
    assert main(["drain", "--run-dir", str(run), "--clear"]) == 0
    assert drain_requested(run) is False
