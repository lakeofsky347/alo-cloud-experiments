"""Task shards share one ledger. Chains stay together and caps stay global."""
import json
import multiprocessing as mp
from pathlib import Path

import pytest

from alo_cloud.budget import BudgetLedger, BudgetStopped
from alo_cloud.collect import collect_sealed
from alo_cloud.errors import PilotStopped
from alo_cloud.freeze import filter_shard, iter_units, shard_for_task, task_groups
from alo_cloud.run import _append_record, run_schedule
from alo_cloud.shardlock import exclusive_lock, halt_reason, set_halt


def _completion(url, body, headers):
    del url, headers
    return {
        "model": body["model"],
        "choices": [{"finish_reason": "stop", "message": {"content": '{"final":"ok"}'}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def test_task_groups_keep_chains_and_cover_every_unit():
    groups = task_groups()
    flat = [key for group in groups for key in group]
    assert len(flat) == len(set(flat))
    m1 = list(iter_units("m1"))
    m2 = list(iter_units("m2"))
    everyone = m1 + m2
    parts = [filter_shard(everyone, index, 4) for index in range(4)]
    got = [unit.unit_id for part in parts for unit in part]
    assert sorted(got) == sorted(unit.unit_id for unit in everyone)
    assert len(got) == len(set(got))
    for part in parts:
        positions = [everyone.index(unit) for unit in part]
        assert positions == sorted(positions)
    for unit in m1:
        if unit.phase != "B1-self":
            continue
        parent = next(item for item in m1 if item.unit_id == unit.parent_unit_id)
        assert shard_for_task(unit.track, unit.task_id, 4) == shard_for_task(parent.track, parent.task_id, 4)
    for unit in m2:
        if unit.m2_condition != "persist" or not unit.parent_unit_id:
            continue
        parent = next(item for item in m2 if item.unit_id == unit.parent_unit_id)
        assert shard_for_task(unit.track, unit.task_id, 4) == shard_for_task(parent.track, parent.task_id, 4)
        twin = next(item for item in m2 if item.task_id == unit.task_id and item.track == unit.track
                    and item.m2_condition == "reset" and item.order_id == unit.order_id)
        assert shard_for_task(twin.track, twin.task_id, 4) == shard_for_task(unit.track, unit.task_id, 4)


def _reserve_capped(path, index, queue):
    ledger = BudgetLedger(path)
    try:
        ledger.reserve(
            f"req-cap-{index}", "one", 40, payload_sha256="a" * 64, shard_id=f"{index}/4")
        queue.put("ok")
    except BudgetStopped as error:
        queue.put(str(error))
    except Exception as error:
        queue.put(f"err:{type(error).__name__}:{error}")


def test_four_processes_cannot_overshoot_the_shared_cap(tmp_path):
    path = tmp_path / "budget.sqlite3"
    BudgetLedger.create(
        path, model_limits_microcny={"one": 1000}, opening_commitments_microcny={"one": 0},
        max_requests=10, prior_requests=0, total_limit_microcny=100)
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_reserve_capped, args=(str(path), index, queue)) for index in range(4)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(30)
        assert proc.exitcode == 0
    results = [queue.get(timeout=5) for _ in range(4)]
    assert results.count("ok") == 2
    assert all(item == "ok" or "cumulative_total_cny_limit" in item for item in results)
    ledger = BudgetLedger(path)
    assert ledger.snapshot()["total_committed_microcny"] == 80


def test_a_shard_hold_does_not_block_another_shard(tmp_path):
    path = tmp_path / "budget.sqlite3"
    ledger = BudgetLedger.create(
        path, model_limits_microcny={"one": 1000}, opening_commitments_microcny={"one": 0},
        max_requests=10, prior_requests=0, total_limit_microcny=1000)
    ledger.reserve("req-a", "one", 10, payload_sha256="a" * 64, shard_id="0/2")
    ledger.reserve("req-b", "one", 10, payload_sha256="b" * 64, shard_id="1/2")
    with pytest.raises(BudgetStopped, match="unresolved_request_blocks_subsequent_dispatch"):
        ledger.reserve("req-a2", "one", 10, payload_sha256="c" * 64, shard_id="0/2")
    with pytest.raises(BudgetStopped, match="unresolved_request_blocks_subsequent_dispatch"):
        ledger.reserve("req-legacy", "one", 10, payload_sha256="d" * 64, shard_id="")


def _append_unknown(run_dir, index, count, queue):
    from alo_cloud.run import _note_unknown
    for step in range(count):
        _note_unknown(run_dir, {"unit_id": f"u{index}", "reason": "x", "unknown_request_id": f"r{step}"})
    queue.put(index)


def test_unknown_counter_appends_do_not_drop_lines(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_append_unknown, args=(str(run), index, 20, queue)) for index in range(2)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(30)
        assert proc.exitcode == 0
    lines = [line for line in (run / "unknown_outcomes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 40


def test_halt_flag_stops_a_later_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    run = tmp_path / "run"
    set_halt(run, "charged_timeout_rate", "1/4")
    assert json.loads((run / "schedule_halt.json").read_text(encoding="utf-8"))["hard"] is False
    assert halt_reason(run) == "charged_timeout_rate"
    assert set_halt(run, "other", "2/4") == "charged_timeout_rate"
    units = list(iter_units("m1"))[:1]
    assert run_schedule(units, run, live=False, sender=_completion) == []
    assert not (run / "records.jsonl").exists() or (run / "records.jsonl").read_text(encoding="utf-8").strip() == ""
    hard = tmp_path / "hard"
    set_halt(hard, "model_cny_cap_exhausted", "0/4")
    assert json.loads((hard / "schedule_halt.json").read_text(encoding="utf-8"))["hard"] is True
    with pytest.raises(PilotStopped, match="model_cny_cap_exhausted"):
        run_schedule(units, hard, live=False, sender=_completion)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "schedule_halt.json").write_text(
        json.dumps({"halt": True, "reason": "operator", "shard": "1/4"}) + "\n", encoding="utf-8")
    with pytest.raises(PilotStopped, match="operator"):
        run_schedule(units, legacy, live=False, sender=_completion)


def test_resume_keeps_epoch_and_skips_finished_units(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    task = "scikit-learn__scikit-learn-13584"
    pair = [unit for unit in iter_units("m1")
            if unit.task_id == task and unit.phase == "B0" and unit.arm.startswith("g1")]
    done, fresh = pair
    shard = shard_for_task("SWE-bench_Lite", task, 4)
    other = next(unit for unit in iter_units("m1")
                 if unit.phase == "B0" and shard_for_task(unit.track, unit.task_id, 4) != shard)
    run = tmp_path / "run"
    run.mkdir()
    (run / "protocol_epoch.json").write_text(json.dumps({"epoch": 4, "reason": "continue"}) + "\n", encoding="utf-8")
    (run / "unit_restarts.json").write_text("{}\n", encoding="utf-8")
    (run / "unknown_outcomes.jsonl").write_text("", encoding="utf-8")
    (run / "progress").mkdir()
    (run / "progress" / "timeout_counts.json").write_text("{}\n", encoding="utf-8")
    (run / "records.jsonl").write_text(json.dumps({
        "unit_id": done.unit_id, "status": "completed", "halt_schedule": False,
        "track": done.track, "task_id": done.task_id, "study": "m1",
    }) + "\n", encoding="utf-8")
    calls = {"n": 0}

    def sender(url, body, headers):
        calls["n"] += 1
        return _completion(url, body, headers)

    rows = run_schedule(pair, run, live=False, sender=sender, shard=(shard, 4))
    assert rows[0]["unit_id"] == done.unit_id and rows[0]["status"] == "completed"
    assert any(row["unit_id"] == fresh.unit_id and row["status"] == "completed" for row in rows)
    assert all(row["unit_id"] != other.unit_id for row in rows)
    assert calls["n"] >= 1
    text = (run / "records.jsonl").read_text(encoding="utf-8")
    assert text.count(done.unit_id) == 1
    assert fresh.unit_id in text
    assert other.unit_id not in text
    assert json.loads((run / "protocol_epoch.json").read_text(encoding="utf-8"))["epoch"] == 4
    summary = collect_sealed(run)
    assert summary["units"] == 2


def test_locked_appends_are_visible_to_collect_once(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    first = {"unit_id": "unit-a", "status": "completed", "track": "SWE-bench_Lite",
             "task_id": "scikit-learn__scikit-learn-13584", "study": "m1", "arm": "g1-main", "phase": "B0"}
    second = {"unit_id": "unit-b", "status": "completed", "track": "SWE-bench_Lite",
              "task_id": "django__django-11583", "study": "m1", "arm": "g1-main", "phase": "B0"}
    _append_record(run / "records.jsonl", first)
    _append_record(run / "records.jsonl", second)
    with exclusive_lock(run, "records"):
        lines = [line for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    summary = collect_sealed(run)
    assert summary["units"] == 2
    assert summary["predictions"] == 2
