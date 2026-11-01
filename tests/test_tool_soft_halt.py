"""Tool exceptions settle, non-budget halts drain, and infra requeues keep the restart."""
import json
from pathlib import Path

import pytest

from alo_cloud.budget import BudgetStopped
from alo_cloud.errors import PilotStopped, ProtocolError
from alo_cloud.freeze import iter_units
from alo_cloud.ledger import infra_generations, main, requeue_interrupted
from alo_cloud.run import open_run_ledger, run_schedule
from alo_cloud.sandbox import DockerSandbox
from alo_cloud.shardlock import halt_record
from alo_cloud.tool_loop import persistent_tool_loop
from alo_cloud.native_bridge import BoundModel
from tests.test_services import KEYS, build, messages


def _final(url, body, headers):
    del url, headers
    return {
        "model": body["model"],
        "choices": [{"finish_reason": "stop", "message": {"content": '{"final":"ok"}'}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _action(name, arguments=None):
    payload = json.dumps({"tool": name, "arguments": arguments or {}}, separators=(",", ":"))
    return {
        "model": "mimo-v2.6-flash",
        "choices": [{"finish_reason": "stop", "message": {"content": payload}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _b0_pair():
    return [unit for unit in iter_units("m1") if unit.arm == "g1-main" and unit.phase == "B0"][:2]


def test_settled_tool_exception_returns_to_the_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    secret = KEYS["MIMO_API_KEY"]
    services, seen, ledger = build(tmp_path, monkeypatch, [
        _action("list_files"),
        {"model": "mimo-v2.6-flash",
         "choices": [{"finish_reason": "stop", "message": {"content": '{"final":"submitted work"}'}}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
    ])

    def boom():
        raise RuntimeError("malformed tool-call markup " + secret)

    services.workspace.list_files = boom
    with pytest.raises(ProtocolError, match="tool_failed:RuntimeError"):
        services.list_files()
    assert ledger.snapshot()["unresolved_tool_ids"] == []
    recorded = json.loads(next((tmp_path / "diagnostics").glob("tool-*.json")).read_text(encoding="utf-8"))
    assert recorded["cause"] == "tool_exception" and recorded["settled"] is True
    assert recorded["vendor_code"] == "RuntimeError"
    assert "Traceback" in recorded["vendor_message"]
    assert secret not in json.dumps(recorded)
    result = persistent_tool_loop(BoundModel(services, "mimo-v2.6-flash"), "public task", [], max_turns=3)
    assert result["status"] == "completed"
    assert result["summary"] == "submitted work"
    assert result["termination"] == "final"
    assert len(seen) == 2

    units = _b0_pair()
    calls = {"n": 0}

    def sender(url, body, headers):
        calls["n"] += 1
        if calls["n"] % 2:
            return _action("list_files")
        return _final(url, body, headers)

    from alo_cloud.sandbox import Workspace
    monkeypatch.setattr(Workspace, "list_files", lambda self: (_ for _ in ()).throw(
        RuntimeError("malformed tool-call markup")))
    rows = run_schedule(units, tmp_path / "schedule", live=False, sender=sender)
    assert [row["status"] for row in rows] == ["completed", "completed"]
    assert all(row["halt_schedule"] is False for row in rows)
    assert not (tmp_path / "schedule" / "schedule_halt.json").exists()
    assert (tmp_path / "schedule" / "budget.sqlite3")
    from alo_cloud.budget import BudgetLedger
    assert BudgetLedger(tmp_path / "schedule" / "budget.sqlite3").snapshot()["unresolved_tool_ids"] == []


def test_unsettleable_tool_hold_stays_unknown(tmp_path, monkeypatch):
    services, _seen, ledger = build(tmp_path, monkeypatch, [])

    def refuse(*args, **kwargs):
        raise BudgetStopped("only_pending_reservation_may_be_settled")

    monkeypatch.setattr(ledger, "settle_tool", refuse)

    def boom():
        raise RuntimeError("malformed tool-call markup")

    services.workspace.list_files = boom
    with pytest.raises(PilotStopped, match="tool_outcome_unknown"):
        services.list_files()
    assert ledger.snapshot()["unresolved_tool_ids"]
    recorded = json.loads(next((tmp_path / "diagnostics").glob("tool-*.json")).read_text(encoding="utf-8"))
    assert recorded["settled"] is False and "Traceback" in recorded["vendor_message"]


def test_soft_halt_finishes_the_unit_and_hard_halt_stops_the_call(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    units = _b0_pair()
    run = tmp_path / "soft"
    state = {"set": False}

    def sender(url, body, headers):
        if not state["set"]:
            from alo_cloud.shardlock import set_halt
            set_halt(run, "unknown_outcome_threshold", "1/4")
            state["set"] = True
        if body["messages"][-1]["content"] == "public task" and state["set"] and state.get("tool") is None:
            state["tool"] = True
            return _action("list_files")
        return _final(url, body, headers)

    rows = run_schedule(units, run, live=False, sender=sender)
    assert [row["unit_id"] for row in rows] == [units[0].unit_id]
    assert rows[0]["status"] == "completed" and rows[0]["halt_schedule"] is False
    assert halt_record(run)["hard"] is False
    assert (run / "records.jsonl").read_text(encoding="utf-8").count(units[1].unit_id) == 0

    hard = tmp_path / "hard"
    calls = {"n": 0}

    def hard_sender(url, body, headers):
        calls["n"] += 1
        if calls["n"] == 1:
            from alo_cloud.shardlock import set_halt
            set_halt(hard, "model_cny_cap_exhausted", "0/4")
            return _action("list_files")
        return _final(url, body, headers)

    with pytest.raises(PilotStopped, match="model_cny_cap_exhausted"):
        run_schedule([units[0]], hard, live=False, sender=hard_sender)
    stopped = json.loads((hard / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert stopped["status"] == "stopped" and stopped["reason"] == "model_cny_cap_exhausted"
    assert halt_record(hard)["hard"] is True


def test_sandbox_unknown_restarts_once_without_halting(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))

    def attach(services, unit, image, *, live, memory, pids):
        del unit, image, live, memory, pids

        def runner(argv, timeout_s):
            del argv, timeout_s
            raise RuntimeError("Cannot connect to the Docker daemon")

        services.sandbox = DockerSandbox(
            services.workspace.root, image="example.invalid/alo:test", runner=runner)

    monkeypatch.setattr("alo_cloud.run._attach_sandbox", attach)
    units = _b0_pair()
    calls = {"n": 0}

    def sender(url, body, headers):
        calls["n"] += 1
        if calls["n"] <= 2:
            return _action("command", {"argv": ["python", "--version"], "timeout_s": 5})
        return _final(url, body, headers)

    rows = run_schedule(units, tmp_path / "run", live=False, sender=sender)
    assert rows[0]["reason"] == "second_tool_sandbox_interruption"
    assert rows[0]["status"] == "blocked" and rows[0]["halt_schedule"] is False
    assert rows[1]["status"] == "completed"
    assert not (tmp_path / "run" / "schedule_halt.json").exists()
    counts = json.loads((tmp_path / "run" / "unit_restarts.json").read_text(encoding="utf-8"))
    assert counts[rows[0]["unit_id"]] == 1
    assert calls["n"] == 3


def _row(unit, status, reason):
    return {"unit_id": unit.unit_id, "status": status, "reason": reason, "halt_schedule": True}


def test_requeue_interrupted_cascades_without_using_the_protocol_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    run = tmp_path / "run"
    run.mkdir()
    (run / "protocol_epoch.json").write_text(
        json.dumps({"epoch": 4, "reason": "continue"}) + "\n", encoding="utf-8")
    (run / "unit_restarts.json").write_text(json.dumps({"keep-me": 1}) + "\n", encoding="utf-8")
    pair = [unit for unit in iter_units("m1")
            if unit.task_id == "kisielk__errcheck.dacab89" and unit.arm == "g1-main"]
    b0 = next(unit for unit in pair if unit.phase == "B0")
    b1 = next(unit for unit in pair if unit.phase == "B1-self")
    assert b1.parent_unit_id == b0.unit_id
    persists = [unit for unit in iter_units("m2") if unit.m2_condition == "persist" and unit.parent_unit_id]
    child = persists[0]
    parent_id = child.parent_unit_id
    exhausted = next(unit for unit in iter_units("m1")
                     if unit.phase == "B0" and unit.unit_id not in {b0.unit_id, parent_id})
    rows = [
        _row(b0, "stopped", "unknown_outcome_threshold"),
        _row(b1, "completed", ""),
        {"unit_id": parent_id, "status": "stopped", "reason": "tool_outcome_unknown"},
        {"unit_id": child.unit_id, "status": "completed", "reason": ""},
        _row(exhausted, "stopped", "unit_resource_exhausted"),
    ]
    (run / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (run / "checkpoints" / b0.unit_id).mkdir(parents=True)
    (run / "checkpoints" / b0.unit_id / "adapter.json").write_text("{}\n", encoding="utf-8")
    (run / "sealed" / "programbench" / b1.unit_id).mkdir(parents=True)
    occupied = run / "archive" / "restarts" / "epoch-4" / b0.unit_id / "infra"
    occupied.mkdir(parents=True)
    (occupied / "kept").write_text("earlier\n", encoding="utf-8")
    ledger = open_run_ledger(run / "budget.sqlite3", "dry-run")
    tool_id = f"tool-e4-{b0.unit_id}-8"
    ledger.reserve_tool(tool_id, seconds_ms=1000, unit_id=b0.unit_id, protocol_epoch=4, shard_id="1/4")
    ledger.mark_tool_unknown(tool_id)
    assert ledger.blocks_dispatch("1/4") is True

    result = requeue_interrupted(run)
    assert tool_id not in ledger.snapshot()["unresolved_tool_ids"]
    assert ledger.blocks_dispatch("1/4") is False
    assert b0.unit_id in result["requeued"]
    assert b1.unit_id in result["requeued"]
    assert parent_id in result["requeued"]
    assert child.unit_id in result["requeued"]
    assert exhausted.unit_id not in result["requeued"]
    assert result["restart_counts_unchanged"] is True
    assert json.loads((run / "unit_restarts.json").read_text(encoding="utf-8")) == {"keep-me": 1}
    live = (run / "records.jsonl").read_text(encoding="utf-8")
    assert b0.unit_id not in live and b1.unit_id not in live
    assert exhausted.unit_id in live
    dup = run / "archive" / "restarts" / "epoch-4" / b0.unit_id / "infra-dup2"
    assert (dup / "checkpoint" / "adapter.json").is_file()
    assert (occupied / "kept").read_text(encoding="utf-8") == "earlier\n"
    assert (run / "archive" / "restarts" / "epoch-4" / b1.unit_id / "infra" / "programbench").is_dir()
    audit = [json.loads(line) for line in (run / "progress" / "infra_requeues.jsonl").read_text().splitlines()]
    cascaded = next(row for row in audit if row["unit_id"] == b1.unit_id)
    assert cascaded["source"] == "cascade" and cascaded["cascaded_from"] == b0.unit_id
    assert cascaded["unit_restarts_incremented"] is False
    assert all(row["protocol_epoch"] == 4 for row in audit)
    again = requeue_interrupted(run)
    assert again["count"] == 0
    assert len((run / "progress" / "infra_requeues.jsonl").read_text().splitlines()) == len(audit)

    named = requeue_interrupted(run, [exhausted.unit_id])
    assert named["requeued"] == [exhausted.unit_id]
    assert exhausted.unit_id not in (run / "records.jsonl").read_text(encoding="utf-8")
    assert json.loads((run / "unit_restarts.json").read_text(encoding="utf-8")) == {"keep-me": 1}

    fresh = next(unit for unit in iter_units("m1") if unit.arm == "g1-main" and unit.phase == "B0"
                 and unit.unit_id not in {b0.unit_id, exhausted.unit_id})
    old = f"req-e4-{fresh.unit_id}-1"
    ledger.reserve(old, "mimo-v2.6-flash", 10, payload_sha256="ab" * 32, tokens=10, seconds_ms=1000,
                   unit_id=fresh.unit_id, unit_request_cap=fresh.request_cap, protocol_epoch=4)
    ledger.settle_known(old, 10, tokens=10, seconds_ms=1000)
    with (run / "records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row(fresh, "stopped", "operator")) + "\n")
    assert main(["requeue-interrupted", "--run-dir", str(run), "--unit", fresh.unit_id]) == 0
    assert infra_generations(run)[fresh.unit_id] == 1
    done = run_schedule([fresh], run, live=False, sender=_final)
    assert done[0]["status"] == "completed"
    assert done[0]["infra_generation"] == 1
    assert done[0]["restart_count"] == 0
    ids = ledger.request_ids()
    assert old in ids
    assert any(item.startswith(f"req-e4-{fresh.unit_id}-infra1-") for item in ids)
    assert ledger.unit_usage(fresh.unit_id, protocol_epoch=4, restart_count=0)["requests"] == 1
    assert ledger.unit_usage(fresh.unit_id, protocol_epoch=4, restart_count=0, infra_generation=1)["requests"] == 1


def test_sharded_launcher_clears_halt_only_on_request():
    text = Path("scripts/run_sharded.sh").read_text(encoding="utf-8")
    assert 'if [ "${CLEAR_HALT:-}" = "1" ]' in text
    assert ".venv/bin/python" in text
    assert "schedule_halt.json" in text
    assert text.index("CLEAR_HALT") < text.index("clear-halt")
