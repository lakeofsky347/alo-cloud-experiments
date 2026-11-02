"""Replay and follow-up cost stay offline and label records-only numbers as proxies."""
import json
from pathlib import Path

from alo_cloud.followup_cost import load_records, project_followup
from alo_cloud.signal_replay import execute_public_checks, replay_markdown, replay_run


def _record(unit_id, arm, study, task, model="deepseek-flash", requests=4, micro=1000, resolved_tokens=(20, 5)):
    return {
        "unit_id": unit_id, "arm": arm, "study": study, "task_id": task, "status": "stopped",
        "phase": "B0", "m2_condition": "",
        "model_requests": requests,
        "estimated_cost_cny": micro / 1_000_000,
        "observed_input_tokens": resolved_tokens[0],
        "observed_output_tokens": resolved_tokens[1],
        "charged_microcny": {model: micro},
        "transport": {model: {"n": requests}},
        "decision_records": [
            {"point": "assignment", "selected": model, "preference_applied": True},
        ],
        "edit_log": [
            {"tool": "command", "node": "s1", "paths": []},
            {"tool": "command", "node": "s2", "paths": []},
        ],
    }


def _transcript(returncode, argv):
    return {
        "schema": "dualbench-alo-v2",
        "worker_transcripts": {
            "s2": [
                {"role": "assistant", "content": json.dumps({"tool": "command", "arguments": {"argv": argv}})},
                {"role": "user", "content": json.dumps({"tool": "command", "result": {"returncode": returncode}})},
            ],
        },
    }


def test_records_proxy_does_not_invent_a_posterior_and_transcripts_do(tmp_path):
    run = tmp_path / "run"
    evaluation = tmp_path / "eval"
    (run / "checkpoints" / "u-fail").mkdir(parents=True)
    (run / "checkpoints" / "u-pass").mkdir(parents=True)
    records = [
        _record("u-fail", "g2", "m1", "task-fail"),
        _record("u-pass", "g2", "m1", "task-pass", micro=2000),
        _record("u-main", "g1-main", "m1", "task-fail", model="mimo-v2.6-flash", micro=500),
    ]
    (run / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    (run / "checkpoints" / "u-fail" / "adapter.json").write_text(
        json.dumps(_transcript(1, ["python3", "-m", "pytest", "tests/test_bug.py"])), encoding="utf-8")
    (run / "checkpoints" / "u-pass" / "adapter.json").write_text(
        json.dumps(_transcript(0, ["python3", "-m", "pytest", "tests/test_bug.py"])), encoding="utf-8")
    evaluation.mkdir()
    (evaluation / "m1_units.csv").write_text(
        "unit_id,arm,task,resolved,cost_cny\n"
        "u-fail,g2,task-fail,False,0.001\n"
        "u-pass,g2,task-pass,True,0.002\n"
        "u-main,g1-main,task-fail,False,0.0005\n",
        encoding="utf-8")
    report = replay_run(run, evaluation)
    assert report["records_proxy"]["proxy"] is True
    assert report["records_proxy"]["precision_recall"] is None
    assert report["records_proxy"]["posterior_updated"] is False
    assert report["records_proxy"]["command_calls"] == 4
    metrics = report["transcript"]["precision_recall"]
    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert metrics["true_negative"] == 1
    assert report["transcript"]["beta_gt_1"] is True
    assert any(row["beta"] > 1 for row in report["transcript"]["posterior"]["belief"])
    assert report["comparison"]["m1"]["oracle"]["tasks"] == 2
    text = replay_markdown(report)
    assert "Records-only proxy" in text
    assert "not a repro-v1 pass or fail" in text
    assert "--execute" in text

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "records.jsonl").write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    proxy_only = replay_run(bare, evaluation)
    assert proxy_only["adapters_found"] == 0
    assert proxy_only["transcript"]["precision_recall"] is None
    assert proxy_only["transcript"]["beta_gt_1"] is False


def test_execute_mode_is_injected_and_labels_its_source(tmp_path):
    run = tmp_path / "run"
    evaluation = tmp_path / "eval"
    unit = "m1.SWE.task.g2.B0"
    checkpoint = run / "checkpoints" / unit
    checkpoint.mkdir(parents=True)
    (run / "records.jsonl").write_text(json.dumps(_record(unit, "g2", "m1", "task")) + "\n", encoding="utf-8")
    (checkpoint / "model.patch").write_text("diff --git a/bug.py\n", encoding="utf-8")
    (checkpoint / "metadata.json").write_text(
        json.dumps({"sandbox_image": "example.invalid/alo:test"}), encoding="utf-8")
    evaluation.mkdir()
    (evaluation / "m1_units.csv").write_text(
        "unit_id,arm,task,resolved,cost_cny\n" + unit + ",g2,task,False,0.001\n", encoding="utf-8")
    seen = {}

    def runner(argv, timeout_s):
        seen["argv"] = argv
        seen["timeout"] = timeout_s
        return {"returncode": 0, "stdout": "1 1 0 1\n", "stderr": ""}

    report = replay_run(run, evaluation, execute=True, executor=runner)
    assert seen["argv"][0] == "docker"
    assert "--network=none" in seen["argv"]
    assert "example.invalid/alo:test" in seen["argv"]
    assert report["execute_replay"]["proxy"] is False
    assert report["execute_replay"]["beta_gt_1"] is True
    assert report["execute_replay"]["precision_recall"]["true_negative"] == 1
    direct = execute_public_checks(
        image="example.invalid/alo:test", patch_text="diff\n", runner=runner)
    assert direct["source"] == "execute"
    assert direct["signal"] == "fail"


def test_followup_projection_uses_observed_rates_and_names_the_cap(tmp_path):
    rows = []
    for index in range(2):
        rows.append(_record(f"m1-{index}", "g1-main", "m1", f"t{index}", model="mimo-v2.6-flash", requests=2, micro=2_000_000))
        rows.append(_record(f"g2-{index}", "g2", "m1", f"t{index}", model="deepseek-flash", requests=4, micro=4_000_000))
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = project_followup(load_records(path))
    assert report["observed"]["cny_per_request"]["deepseek-flash"] == 1
    assert report["components_cny"]["g1-deepseek-flash"]["deepseek-flash"] == 4
    assert report["components_cny"]["g2-fixed"]["deepseek-flash"] == 8
    assert report["projected_cny"]["deepseek-flash"] > report["components_cny"]["r2_repeat"]["deepseek-flash"]
    assert "cap_cny_per_model" in report
    assert report["within_cap"] is True
    assert Path(path).is_file()
