"""ProgramBench prompts name compile.sh. A bad prompt can be invalidated in-epoch."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from alo_cloud.evaluate import (
    PROGRAMBENCH_COMMIT, _score_programbench, programbench_docker_cpus, programbench_env,
)
from alo_cloud.freeze import iter_units, shard_for_task
from alo_cloud.ledger import main, requeue_track, track_generations
from alo_cloud.progress import assess_charged_timeouts
from alo_cloud.programbench_prompt import PROGRAMBENCH_SUBMISSION_CONTRACT
from alo_cloud.run import open_run_ledger, run_schedule, stage_for


def _final(url, body, headers):
    del url, headers
    return {
        "model": body["model"],
        "choices": [{"finish_reason": "stop", "message": {"content": '{"final":"ok"}'}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _pb_unit(arm="g1-main", phase="B0"):
    return next(unit for unit in iter_units("m1")
                if unit.track == "ProgramBench" and unit.arm == arm and unit.phase == phase)


def test_program_bench_prompt_states_the_official_contract_and_swe_does_not():
    phrase = "chmod +x ./compile.sh && ./compile.sh"
    assert phrase in PROGRAMBENCH_SUBMISSION_CONTRACT
    assert "./executable" in PROGRAMBENCH_SUBMISSION_CONTRACT
    assert "task_cleanroom_v6" in PROGRAMBENCH_SUBMISSION_CONTRACT
    assert "no network" in PROGRAMBENCH_SUBMISSION_CONTRACT
    assert "Python 3.10.12" in PROGRAMBENCH_SUBMISSION_CONTRACT
    assert "Python 3.12" not in PROGRAMBENCH_SUBMISSION_CONTRACT
    for arm in ("g1-main", "g1-strong", "g2", "g3"):
        unit = _pb_unit(arm)
        stage = stage_for(unit, {"problem_statement": "Rebuild the program from scratch."}, dry_run=True)
        assert stage["public"]["problem_statement"].startswith("Rebuild the program")
        assert phrase in stage["public"]["submission_contract"]
    swe = next(unit for unit in iter_units("m1") if unit.track == "SWE-bench_Lite")
    swe_stage = stage_for(swe, {"problem_statement": "public bug"}, dry_run=True)
    assert "submission_contract" not in swe_stage["public"]
    assert phrase not in json.dumps(swe_stage)


def test_end_of_unit_records_compile_sh_without_another_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    unit = _pb_unit()
    seen = {}

    def sender(url, body, headers):
        seen["prompt"] = json.dumps(body)
        workspace = next(Path(tmp_path).glob("units/*/state/workspace"))
        (workspace / "compile.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        return _final(url, body, headers)

    rows = run_schedule([unit], tmp_path, live=False, sender=sender)
    assert rows[0]["status"] == "completed"
    assert rows[0]["programbench_compile_sh_present"] is True
    assert rows[0]["model_requests"] == 1
    assert "chmod +x ./compile.sh && ./compile.sh" in seen["prompt"]
    recorded = json.loads((tmp_path / "diagnostics" / f"submission-{unit.unit_id}.json").read_text())
    assert recorded["cause"] == "submission_check"
    assert recorded["response"]["compile_sh_present"] is True


def test_requeue_track_archives_program_bench_and_keeps_swe(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    run = tmp_path / "run"
    run.mkdir()
    (run / "protocol_epoch.json").write_text(
        json.dumps({"epoch": 4, "reason": "continue"}) + "\n", encoding="utf-8")
    pb = _pb_unit()
    inflight = next(unit for unit in iter_units("m1")
                    if unit.track == "ProgramBench" and unit.unit_id != pb.unit_id and unit.phase == "B0")
    swe = next(unit for unit in iter_units("m1") if unit.track == "SWE-bench_Lite" and unit.phase == "B0")
    (run / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (
        {"unit_id": pb.unit_id, "track": "ProgramBench", "status": "completed", "protocol_epoch": 4},
        {"unit_id": swe.unit_id, "track": "SWE-bench_Lite", "status": "completed", "protocol_epoch": 4},
    )), encoding="utf-8")
    (run / "checkpoints" / pb.unit_id).mkdir(parents=True)
    (run / "checkpoints" / pb.unit_id / "adapter.json").write_text("{}\n", encoding="utf-8")
    (run / "checkpoints" / inflight.unit_id).mkdir(parents=True)
    (run / "checkpoints" / swe.unit_id).mkdir(parents=True)
    (run / "unit_restarts.json").write_text(
        json.dumps({pb.unit_id: 1, swe.unit_id: 1}) + "\n", encoding="utf-8")
    (run / "unknown_outcomes.jsonl").write_text(
        json.dumps({"unit_id": pb.unit_id, "reason": "provider_outcome_charged"}) + "\n"
        + json.dumps({"unit_id": swe.unit_id, "reason": "provider_outcome_charged"}) + "\n",
        encoding="utf-8")
    ledger = open_run_ledger(run / "budget.sqlite3", "dry-run")
    old = f"req-e4-{pb.unit_id}-1"
    swe_req = f"req-e4-{swe.unit_id}-1"
    for request_id, unit_id in ((old, pb.unit_id), (swe_req, swe.unit_id)):
        ledger.reserve(request_id, "mimo-v2.6-flash", 10, payload_sha256="ab" * 32,
                       tokens=10, seconds_ms=1000, unit_id=unit_id, unit_request_cap=12, protocol_epoch=4)
        ledger.settle_known(request_id, 10, tokens=10, seconds_ms=1000)
    (run / "diagnostics").mkdir()
    for request_id in (old, swe_req):
        (run / "diagnostics" / f"{request_id}.json").write_text(json.dumps({
            "cause": "deadline_timeout", "settled": True,
        }) + "\n", encoding="utf-8")
    before = assess_charged_timeouts(ledger, protocol_epoch=4)
    assert before["charged_timeouts"] == 2

    result = requeue_track(run, "ProgramBench", reason="harness_prompt_invalid")
    assert pb.unit_id in result["requeued"]
    assert inflight.unit_id in result["requeued"]
    assert swe.unit_id not in result["requeued"]
    live = (run / "records.jsonl").read_text(encoding="utf-8")
    assert pb.unit_id not in live
    assert swe.unit_id in live
    archive = run / "archive" / "invalidated" / "epoch-4" / "programbench"
    assert (archive / pb.unit_id / "checkpoint" / "adapter.json").is_file()
    assert (archive / inflight.unit_id / "checkpoint").is_dir()
    assert (run / "checkpoints" / swe.unit_id).is_dir()
    counts = json.loads((run / "unit_restarts.json").read_text(encoding="utf-8"))
    assert pb.unit_id not in counts and counts[swe.unit_id] == 1
    unknown = (run / "unknown_outcomes.jsonl").read_text(encoding="utf-8")
    assert pb.unit_id not in unknown and swe.unit_id in unknown
    assert track_generations(run)[pb.unit_id] == 1
    after = assess_charged_timeouts(ledger, protocol_epoch=4)
    assert after["charged_timeouts"] == 1
    assert old in ledger.request_ids()
    again = requeue_track(run, "ProgramBench", reason="harness_prompt_invalid")
    assert again["count"] == 0
    with pytest.raises(SystemExit):
        main(["requeue-track", "--run-dir", str(run), "--benchmark", "SWE-bench_Lite",
              "--reason", "no"])

    seen = {}

    def sender(url, body, headers):
        seen["n"] = seen.get("n", 0) + 1
        return _final(url, body, headers)

    rows = run_schedule([pb], run, live=False, sender=sender)
    assert rows[0]["status"] == "completed"
    assert rows[0]["track_generation"] == 1
    assert rows[0]["restart_count"] == 0
    assert any(item.startswith(f"req-e4-{pb.unit_id}-pbv1-") for item in ledger.request_ids())
    assert ledger.unit_usage(pb.unit_id, protocol_epoch=4, restart_count=0)["requests"] == 1
    assert ledger.unit_usage(
        pb.unit_id, protocol_epoch=4, restart_count=0, track_generation=1)["requests"] == 1
    second = requeue_track(run, "ProgramBench", reason="pb_budget_too_small")
    assert pb.unit_id in second["requeued"]
    assert swe.unit_id not in second["requeued"]
    assert (run / "archive" / "invalidated" / "epoch-4" / "programbench-pbv2" / pb.unit_id).is_dir()
    assert (archive / pb.unit_id / "checkpoint" / "adapter.json").is_file()
    assert track_generations(run)[pb.unit_id] == 2
    assert swe.unit_id in (run / "records.jsonl").read_text(encoding="utf-8")


def test_scoring_cpus_and_container_start_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("PROGRAMBENCH_DOCKER_CPUS", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert programbench_docker_cpus() == 8
    monkeypatch.setenv("PROGRAMBENCH_DOCKER_CPUS", "10")
    assert programbench_docker_cpus() == 10
    env = programbench_env({"PATH": "/usr/bin", "PROGRAMBENCH_DOCKER_CPUS": "6", "DEEPSEEK_API_KEY": "secret"})
    assert env["PROGRAMBENCH_DOCKER_CPUS"] == "6"
    assert "DEEPSEEK_API_KEY" not in env
    import alo_cloud.evaluate as evaluate

    def fake_run(argv, **kwargs):
        assert "--docker-cpus" in argv
        assert argv[argv.index("--docker-cpus") + 1] == "10"
        return subprocess.CompletedProcess(argv, 0, "", "Failed to start container: range of CPUs is from 0.01 to 8.00")

    monkeypatch.setattr(evaluate, "_harness_run", fake_run)
    monkeypatch.setattr(evaluate, "_git_head", lambda repo: PROGRAMBENCH_COMMIT)
    monkeypatch.setattr(evaluate.shutil, "which", lambda name: "/usr/bin/uv")
    unit = tmp_path / "run" / "sealed" / "programbench" / "unit-1" / "task-1"
    unit.mkdir(parents=True)
    (unit / "submission.tar.gz").write_bytes(b"")
    report = _score_programbench(tmp_path / "run", tmp_path / "repo")
    assert report["status"] == "harness_failed"
    assert report["units"][0]["container_start_failed"] is True
    assert report["units"][0]["status"] == "harness_failed"


def test_program_bench_request_caps_stay_inside_the_protocol_maxima():
    from alo_cloud.freeze import PB_REQUEST_CAPS
    from alo_cloud.services import PROTOCOL_UNIT_LIMITS
    assert PB_REQUEST_CAPS["b0"] <= PROTOCOL_UNIT_LIMITS["m1_b0"]["requests"]
    assert PB_REQUEST_CAPS["b1"] <= PROTOCOL_UNIT_LIMITS["m1_b1"]["requests"]
    pb = [unit for unit in iter_units("m1") if unit.track == "ProgramBench"]
    assert {unit.request_cap for unit in pb if unit.phase == "B0"} == {PB_REQUEST_CAPS["b0"]}
    assert {unit.request_cap for unit in pb if unit.phase == "B1-self"} == {PB_REQUEST_CAPS["b1"]}
    assert len({unit.arm for unit in pb}) == 4
    swe = [unit for unit in iter_units("m1") if unit.track == "SWE-bench_Lite" and unit.phase == "B0"]
    assert {unit.request_cap for unit in swe} == {12}
    learning = [unit for unit in iter_units("m2")
                if unit.track == "ProgramBench" and unit.phase == "learning"]
    assert {unit.request_cap for unit in learning} == {PB_REQUEST_CAPS["learning"]}
    g3 = next(unit for unit in pb if unit.arm == "g3" and unit.phase == "B0")
    assert g3.request_cap == PB_REQUEST_CAPS["b0"]
    assert g3.request_cap != PB_REQUEST_CAPS["b0"] * 4


def test_budget_notice_is_injected_when_three_requests_remain(tmp_path, monkeypatch):
    from alo_cloud.programbench_prompt import PROGRAMBENCH_BUDGET_NOTICE
    from tests.test_services import build, messages
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    replies = [
        {"model": "mimo-v2.6-flash", "choices": [{"message": {"content": "look"}}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
        {"model": "mimo-v2.6-flash", "choices": [{"message": {"content": "build"}}],
         "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
    ]
    services, seen, _ledger = build(tmp_path, monkeypatch, replies)
    services.budget_notice = PROGRAMBENCH_BUDGET_NOTICE
    services.limits["requests"] = 4
    thread = [{"role": "user", "content": "task"}]
    services.complete_for_model("mimo-v2.6-flash", thread, max_tokens=16)
    assert PROGRAMBENCH_BUDGET_NOTICE not in json.dumps(seen[0]["body"])
    services.complete_for_model("mimo-v2.6-flash", thread, max_tokens=16)
    assert PROGRAMBENCH_BUDGET_NOTICE in json.dumps(seen[1]["body"])
    assert services.budget_notice_sent is True
    recorded = json.loads(next((tmp_path / "diagnostics").glob("notice-*.json")).read_text())
    assert recorded["cause"] == "budget_notice"
    assert recorded["response"]["requests_remaining"] == 3
    assert "compile.sh" in recorded["vendor_message"]


def test_program_bench_rerun_keeps_mimo_pro_under_45_cny():
    from alo_cloud.projection import program_bench_rerun_cost
    chosen = program_bench_rerun_cost()
    assert chosen["units"] == 90
    assert chosen["pro_with_already_microcny"] <= 45_000_000
    wider = program_bench_rerun_cost(b0=40, b1=25)
    assert wider["pro_with_already_microcny"] > 45_000_000


def test_m1_program_bench_uses_every_shard_and_m2_orders_split():
    m1 = [unit.task_id for unit in iter_units("m1")
          if unit.track == "ProgramBench" and unit.phase == "B0" and unit.arm == "g1-main"]
    assert {shard_for_task("ProgramBench", task_id, 4) for task_id in m1} == {0, 1, 2, 3}
    from alo_cloud.freeze import shard_for_unit
    m2 = [unit for unit in iter_units("m2") if unit.track == "ProgramBench" and unit.m2_condition == "persist"]
    by_order = {}
    for unit in m2:
        by_order.setdefault(unit.order_id, set()).add(shard_for_unit(unit, 4))
    assert len(by_order) == 2
    assert all(len(shards) == 1 for shards in by_order.values())
    assert len({next(iter(shards)) for shards in by_order.values()}) == 2
    text = Path("scripts/run_sharded.sh").read_text(encoding="utf-8")
    assert text.index("run_study m2") < text.index('wait "$pid"')
    assert "ProgramBench M2 orders are separate shards" in text
