"""ProgramBench scores belong to one sealed attempt. All evaluator output here is synthetic."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from alo_cloud.comparison import pb_paired_report
from alo_cloud.evaluate import PROGRAMBENCH_COMMIT
from alo_cloud.fox_pb import apply_pb_score, pb_eval_plan, run_official_pb_eval


CLOG = "clog-tool__clog-cli.7066cba"
XCP = "tarka__xcp.5e5b448"
PDU = "ksxgithub__parallel-disk-usage.96978ed"


def plan_for(root, task, *, arm="A4", attempt="first"):
    source = root / "workspace"
    source.mkdir(exist_ok=True)
    (source / "source.py").write_text("print('test')\n")
    return pb_eval_plan(task, source, run_dir=root / "run", repo=root / "pb",
                        arm=arm, unit_id=f"unit-{task}", attempt_id=attempt)


def evaluate(plan, document=None, *, stdout="", change_archive=False):
    def harness(argv, *, cwd, env):
        assert "FOXROUTER_API_KEY" not in env
        task_path = Path(plan["sealed_dir"]) / plan["task_id"]
        assert (task_path / "submission.tar.gz").is_file()
        if document is not None:
            (task_path / f"{plan['task_id']}.eval.json").write_text(json.dumps(document))
        if change_archive:
            (task_path / "submission.tar.gz").write_bytes(b"changed after evaluator started")
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    return run_official_pb_eval(plan, git_head=lambda repo: PROGRAMBENCH_COMMIT, harness_run=harness)


def test_clog_cannot_supply_results_for_xcp_pdu_or_another_attempt(tmp_path):
    plan = plan_for(tmp_path, CLOG)
    # Same number of passes and failures as the reported case, without hidden test material.
    cases = [{"status": "passed"}] * 643 + [{"status": "failure"}] * (778 - 643)
    score = evaluate(plan, {"test_results": cases, "error_code": None})
    assert score["pass_rate"] == 643 / 778
    assert score["compile_success"] is True
    identity = score["official_eval_identity"]
    task_root = Path(plan["sealed_dir"]) / CLOG
    assert identity["arm"] == "A4" and identity["task_id"] == CLOG
    assert identity["unit_id"] == plan["unit_id"] and identity["attempt_id"] == "first"
    for path, field in ((task_root / "submission.tar.gz", "archive_sha256"),
                        (task_root / f"{CLOG}.eval.json", "eval_sha256")):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity[field]
    for unscored in (plan_for(tmp_path, XCP), plan_for(tmp_path, PDU),
                     plan_for(tmp_path, CLOG, attempt="retry"),
                     plan_for(tmp_path, CLOG, arm="A2")):
        missing = evaluate(unscored, stdout='{"pass_rate": 1, "compile_success": true}')
        assert missing["status"] == "unavailable" and missing["reason"] == "official_eval_missing"
        assert missing["compile_success"] is missing["pass_rate"] is missing["resolved"] is None
    stale = evaluate(plan)
    assert stale["reason"] == "official_attempt_already_exists" and stale["pass_rate"] is None


@pytest.mark.parametrize("error,compiled,rate", [
    ("compile_failed", False, 0.0),
    ("copy_executable_failed", False, 0.0),
    ("seed_git_failed", None, None),
    ("wipe_workspace_failed", None, None),
    ("hash_executable_failed", None, None),
])
def test_build_errors_are_distinct_from_infrastructure(tmp_path, error, compiled, rate):
    score = evaluate(plan_for(tmp_path, XCP), {
        "test_results": [{"status": "not_run"}], "error_code": error})
    assert score["compile_success"] is compiled and score["pass_rate"] == rate


def test_archive_change_and_task_identity_mismatch_are_unknown(tmp_path):
    plan = plan_for(tmp_path, XCP)
    good = {"test_results": [{"status": "passed"}], "error_code": None}
    assert evaluate(plan, good, change_archive=True)["reason"] == "official_archive_changed"
    other = evaluate(plan_for(tmp_path, XCP, attempt="second"), {**good, "instance_id": PDU})
    assert other["reason"] == "official_eval_identity_mismatch" and other["pass_rate"] is None


def test_build_can_be_measured_without_a_test_score(tmp_path):
    result = evaluate(plan_for(tmp_path, PDU), {
        "test_results": [{"status": "system_error"}], "executable_hash": "0" * 64})
    assert result["compile_success"] is True and result["pass_rate"] is None
    assert result["reason"] == "official_tests_unobserved"


def test_rejected_build_keeps_unit_zero_and_an_independent_official_measurement():
    official = {"status": "finished", "pass_rate": 0.75, "resolved": False,
                "compile_success": True, "official_eval_identity": {"archive_sha256": "a" * 64}}
    scored = apply_pb_score({}, execute=True, runner=lambda p: official, cost=0, wall_clock_s=0,
                            submission_check={"ok": False, "violations": []})
    assert scored["score"] == scored["pass_rate"] == 0.0 and scored["status"] == "submission_rejected"
    assert scored["official_pass_rate"] == 0.75 and scored["compile_success"] is True
    assert scored["official_eval_identity"] == official["official_eval_identity"]


def test_official_report_keeps_legacy_evidence_separate_and_refuses_ambiguous_pairs():
    rows = [
        {"arm": "A4", "task_id": CLOG, "official_compile_success": True, "compile_success": True,
         "submission_accepted": True},
        {"arm": "A4", "task_id": XCP, "official_compile_success": None, "compile_success": None,
         "submission_accepted": False},
        {"arm": "A4", "task_id": PDU, "compile_success": True},
    ]
    arm = pb_paired_report(rows)["arms"]["A4"]
    assert arm["official_compile"] == {"pass": 1, "fail": 0, "unknown": 2, "total": 3, "known": 1,
                                       "rate_among_known": 1.0, "coverage": 1 / 3}
    assert arm["submission_acceptance"] == {"accepted": 1, "rejected": 1, "unknown": 1,
                                            "accepted_and_compiled": 1}
    assert arm["legacy_units_without_official_evidence"] == 1
    with pytest.raises(ValueError, match="duplicate_pb_arm_task"):
        pb_paired_report([*rows, {**rows[0], "attempt_id": "second"}])


def test_modified_scoring_plan_is_rejected_before_harness_execution(tmp_path):
    plan = plan_for(tmp_path, XCP)
    for change in ({"task_id": PDU}, {"arm": "A2"}, {"attempt_id": "second"},
                   {"unit_id": "other"}, {"sealed_dir": str(tmp_path)},
                   {"eval_command": ["different"]}):
        result = evaluate({**plan, **change})
        assert result["reason"] == "official_attempt_identity_mismatch"
    assert not Path(plan["sealed_dir"]).exists()


def test_concurrent_scorers_cannot_share_one_attempt(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    plan = plan_for(tmp_path, XCP)
    barrier = Barrier(2)

    def score():
        barrier.wait(timeout=10)
        return evaluate(plan, {"test_results": [{"status": "passed"}]})

    with ThreadPoolExecutor(max_workers=2) as workers:
        jobs = [workers.submit(score) for _ in range(2)]
        results = [job.result(timeout=15) for job in jobs]
    assert sorted(row["status"] for row in results) == ["finished", "unavailable"]
    assert next(row for row in results if row["status"] == "unavailable")["reason"] == "official_attempt_already_exists"


def test_public_behaviour_mismatch_does_not_become_an_official_build_failure(tmp_path, monkeypatch):
    from alo_cloud.fox_pb import PbServices, run_arm

    def public_check(self):
        self.last_pb_signal = {"compile": "pass", "differential": "fail", "signal": "fail"}
        return "fail"

    monkeypatch.setattr(PbServices, "public_signal", public_check)
    run_dir = tmp_path / "run"
    run_arm(run_dir, arm="A4", tasks="kisielk__errcheck", live=False, execute=False)
    row = json.loads((run_dir / "records.jsonl").read_text())
    assert row["public_compile_success"] is True and row["public_differential"] == "fail"
    assert row["official_compile_success"] is row["compile_success"] is None
    assert row["official_pass_rate"] is None


def test_normal_entrypoint_binds_public_snapshot_and_official_archive(tmp_path):
    from alo_cloud.fox_pb import run_arm
    from alo_cloud.programbench_prompt import PB_BUILD_PROTOCOL

    run_dir = tmp_path / "new-protocol"
    expected_rate = 2 / 3
    docs = {CLOG: {"error_code": None,
                   "test_results": [{"status": s} for s in ("passed", "passed", "failure")]}}
    seen = {}

    def scorer(plan):
        seen[plan["task_id"]] = plan
        return evaluate(plan, document=docs.get(plan["task_id"]))

    run_arm(run_dir, arm="A2", tasks=[CLOG, XCP], request_cap=40, node_request_cap=20,
            pb_agent_protocol=PB_BUILD_PROTOCOL, live=False, execute=True, scorer=scorer,
            programbench_repo=tmp_path / "pb")
    rows = {row["task_id"]: row for row in map(json.loads, (run_dir / "records.jsonl").read_text().splitlines())}
    for task, row in rows.items():
        plan = seen[task]
        identity = row["official_eval_identity"]
        assert identity["arm"] == row["arm"] and identity["attempt_id"] == row["attempt_id"]
        assert row["submission_sha256"] == identity["submission_sha256"] == plan["submission_sha256"]
        assert row["submission_accepted"] is True
        assert any(part == row["attempt_id"] for part in Path(plan["sealed_dir"]).parts)
    assert rows[CLOG]["official_pass_rate"] == expected_rate
    assert rows[XCP]["official_compile_success"] is rows[XCP]["official_pass_rate"] is None
    assert rows[XCP]["harness_reason"] == "official_eval_missing"
