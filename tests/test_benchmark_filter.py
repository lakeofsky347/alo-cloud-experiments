"""Benchmark filter skips a track without consuming it, and lets SWE M2 start."""
import json
from pathlib import Path

import pytest

from alo_cloud.budget import BudgetLedger
from alo_cloud.collect import collect_sealed
from alo_cloud.evaluate import main as evaluate_main
from alo_cloud.evaluate import score_sealed
from alo_cloud.freeze import (
    filter_benchmarks, filter_shard, iter_units, parse_benchmarks, shard_for_unit,
)
from alo_cloud.ledger import restart_counts
from alo_cloud.run import blocked_fraction_exceeded, main


def _rows(path):
    file = Path(path) / "records.jsonl"
    if not file.is_file():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _silence_network(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))


def test_parse_benchmarks_accepts_the_box_name_and_rejects_unknown():
    assert parse_benchmarks(None) is None
    assert parse_benchmarks("") is None
    assert parse_benchmarks("SWE-bench") == ("SWE-bench_Lite",)
    assert parse_benchmarks(" SWE-bench_Lite, ProgramBench ") == ("SWE-bench_Lite", "ProgramBench")
    with pytest.raises(ValueError, match="unknown_benchmark"):
        parse_benchmarks("not-a-track")


def test_filter_skips_program_bench_without_side_effects(tmp_path, monkeypatch, capsys):
    _silence_network(monkeypatch)
    monkeypatch.setenv("ALO_BENCHMARKS", "SWE-bench")
    run = tmp_path / "run"
    run.mkdir()
    pb = next(unit for unit in iter_units("m1")
              if unit.track == "ProgramBench" and unit.arm == "g1-main" and unit.phase == "B0")
    (run / "unit_restarts.json").write_text(
        json.dumps({pb.unit_id: 1}, sort_keys=True) + "\n", encoding="utf-8")
    before = (run / "unit_restarts.json").read_text(encoding="utf-8")
    code = main([
        "--study", "m1", "--dry-run", "--run-dir", str(run), "--arm", "g1-main",
        "--tasks", "django__django-11583", "--tasks", pb.task_id,
    ])
    assert code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["skipped_by_filter"] == 2
    assert summary["benchmarks"] == ["SWE-bench_Lite"]
    rows = _rows(run)
    assert rows
    assert {row["track"] for row in rows} == {"SWE-bench_Lite"}
    assert all(row["task_id"] == "django__django-11583" for row in rows)
    assert pb.unit_id not in {row["unit_id"] for row in rows}
    ids = BudgetLedger(run / "budget.sqlite3").request_ids()
    assert ids
    assert all("ProgramBench" not in item for item in ids)
    assert (run / "unit_restarts.json").read_text(encoding="utf-8") == before
    assert restart_counts(run) == {pb.unit_id: 1}
    checkpoints = run / "checkpoints"
    if checkpoints.is_dir():
        assert all("ProgramBench" not in path.name for path in checkpoints.iterdir())
    assert not (run / "archive").exists()


def test_unset_filter_runs_program_bench_and_swe(tmp_path, monkeypatch, capsys):
    _silence_network(monkeypatch)
    monkeypatch.delenv("ALO_BENCHMARKS", raising=False)
    run = tmp_path / "run"
    pb = next(unit for unit in iter_units("m1") if unit.track == "ProgramBench" and unit.arm == "g1-main")
    code = main([
        "--study", "m1", "--dry-run", "--run-dir", str(run), "--arm", "g1-main",
        "--tasks", "django__django-11583", "--tasks", pb.task_id,
    ])
    assert code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["skipped_by_filter"] == 0
    assert "benchmarks" not in summary
    assert {row["track"] for row in _rows(run)} == {"SWE-bench_Lite", "ProgramBench"}


def test_m2_swe_proceeds_when_m1_swe_is_done_and_program_bench_is_queued(tmp_path, monkeypatch, capsys):
    _silence_network(monkeypatch)
    m1 = list(iter_units("m1"))
    m2 = list(iter_units("m2"))
    swe_m2 = [unit for unit in m2 if unit.track == "SWE-bench_Lite"]
    for unit in swe_m2:
        if not unit.parent_unit_id:
            continue
        parent = next(item for item in m2 if item.unit_id == unit.parent_unit_id)
        assert parent.track == "SWE-bench_Lite"
    shard = shard_for_unit(swe_m2[0], 4)
    pb_m1 = [unit for unit in m1 if unit.track == "ProgramBench" and shard_for_unit(unit, 4) == shard]
    swe_m1 = [unit for unit in m1 if unit.track == "SWE-bench_Lite" and shard_for_unit(unit, 4) == shard]
    assert pb_m1 and swe_m1
    run = tmp_path / "run"
    run.mkdir()
    pb = pb_m1[0]
    (run / "unit_restarts.json").write_text(json.dumps({pb.unit_id: 1}) + "\n", encoding="utf-8")
    before = (run / "unit_restarts.json").read_text(encoding="utf-8")
    lines = []
    for unit in swe_m1:
        lines.append(json.dumps({
            "unit_id": unit.unit_id, "track": unit.track, "task_id": unit.task_id,
            "study": "m1", "status": "completed", "arm": unit.arm, "phase": unit.phase,
        }))
    (run / "records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    code = main([
        "--study", "m1", "--dry-run", "--run-dir", str(run),
        "--shard", f"{shard}/4", "--benchmarks", "SWE-bench",
    ])
    assert code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["skipped_by_filter"] == len(pb_m1)
    assert {row["track"] for row in _rows(run)} == {"SWE-bench_Lite"}
    assert (run / "unit_restarts.json").read_text(encoding="utf-8") == before
    assert BudgetLedger(run / "budget.sqlite3").request_ids() == []
    first = next(unit for unit in swe_m2
                 if unit.order_id == "order-1" and unit.phase == "learning" and unit.position == 0
                 and unit.m2_condition == "persist")
    assert shard_for_unit(first, 4) == shard
    assert first.parent_unit_id == ""
    code = main([
        "--study", "m2", "--dry-run", "--run-dir", str(run),
        "--shard", f"{shard}/4", "--benchmarks", "SWE-bench",
        "--arm", "g2", "--tasks", first.task_id,
    ])
    assert code == 0
    rows = _rows(run)
    assert any(row["unit_id"] == first.unit_id and row["status"] == "completed" for row in rows)
    assert all(row.get("track") != "ProgramBench" for row in rows)
    assert all("ProgramBench" not in item for item in BudgetLedger(run / "budget.sqlite3").request_ids())
    assert restart_counts(run) == {pb.unit_id: 1}
    included, skipped = filter_benchmarks(filter_shard(m2, shard, 4), ("SWE-bench_Lite",))
    assert first in included
    assert any(unit.track == "ProgramBench" for unit in skipped)


def test_excluded_shard_exits_zero_without_a_ledger(tmp_path, monkeypatch, capsys):
    _silence_network(monkeypatch)
    m2 = list(iter_units("m2"))
    empty = None
    skipped = 0
    for index in range(4):
        part = filter_shard(m2, index, 4)
        if any(unit.track == "SWE-bench_Lite" for unit in part):
            continue
        pb = [unit for unit in part if unit.track == "ProgramBench"]
        if pb:
            empty = index
            skipped = len(pb)
            break
    assert empty is not None
    run = tmp_path / "run"
    code = main([
        "--study", "m2", "--dry-run", "--run-dir", str(run),
        "--shard", f"{empty}/4", "--benchmarks", "SWE-bench",
    ])
    assert code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["units"] == 0
    assert summary["skipped_by_filter"] == skipped
    assert not (run / "budget.sqlite3").exists()
    assert not (run / "records.jsonl").exists()
    assert not (run / "unit_restarts.json").exists()


def test_blocked_report_ignores_excluded_tracks(tmp_path):
    rows = [{"status": "blocked", "reason": "x", "track": "ProgramBench", "unit_id": f"pb{i}"}
            for i in range(6)]
    rows += [{"status": "completed", "track": "SWE-bench_Lite", "unit_id": f"swe{i}"} for i in range(4)]
    assert blocked_fraction_exceeded(rows) is True
    assert blocked_fraction_exceeded(rows, benchmarks=("SWE-bench_Lite",)) is False
    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert main(["--blocked-report", "--run-dir", str(run)]) == 1
    assert main(["--blocked-report", "--run-dir", str(run), "--benchmarks", "SWE-bench"]) == 0


def test_unknown_benchmark_does_not_touch_the_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(SystemExit) as caught:
        main(["--study", "m1", "--dry-run", "--run-dir", str(run), "--benchmarks", "nope"])
    assert caught.value.code == 2
    assert list(run.iterdir()) == []


def test_collect_and_evaluate_unrun_program_bench(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text(json.dumps({
        "unit_id": "m1.SWE-bench_Lite.django__django-11583.g1-main.B0",
        "task_id": "django__django-11583", "track": "SWE-bench_Lite",
        "arm": "g1-main", "phase": "B0", "study": "m1",
    }) + "\n", encoding="utf-8")
    summary = collect_sealed(run)
    assert summary["predictions"] == 1
    report = score_sealed(run, execute=False)
    assert report["programbench"]["status"] == "not_run"
    assert report["swebench"]["status"] == "not_executed"
    assert len(report["swebench"]["units"]) == 1
    monkeypatch.setattr("alo_cloud.evaluate.shutil.which", lambda name: "/bin/docker")

    def finished(run_dir, predictions):
        del run_dir, predictions
        return {"status": "finished", "units": [{"unit_id": "u"}]}

    def boom(*args, **kwargs):
        del args, kwargs
        raise AssertionError("scored program bench")

    monkeypatch.setattr("alo_cloud.evaluate._score_swebench", finished)
    monkeypatch.setattr("alo_cloud.evaluate._score_programbench", boom)
    executed = score_sealed(run, execute=True)
    assert executed["swebench"]["status"] == "finished"
    assert executed["programbench"]["status"] == "not_run"
    assert evaluate_main(["--run-dir", str(run), "--execute"]) == 0
    sealed = run / "sealed" / "programbench" / "old-unit" / "task"
    sealed.mkdir(parents=True)
    (sealed / "submission.tar.gz").write_bytes(b"not-a-tarball")
    filtered = score_sealed(run, execute=True, benchmarks=("SWE-bench_Lite",))
    assert filtered["swebench"]["status"] == "finished"
    assert filtered["programbench"]["status"] == "not_run"
    assert filtered["programbench"]["skipped_by_filter"] is True


def test_sharded_launcher_forwards_the_benchmark_filter():
    text = Path("scripts/run_sharded.sh").read_text(encoding="utf-8")
    assert "ALO_BENCHMARKS" in text
    assert '--benchmarks "$ALO_BENCHMARKS"' in text
    assert text.index("run_study m1") < text.index("run_study m2") < text.index('wait "$pid"')
    assert "Cleared drain.json" in text
