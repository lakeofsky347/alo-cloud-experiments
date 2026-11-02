"""Replay and follow-up cost stay offline and label records-only numbers as proxies."""
import json
from pathlib import Path

import pytest

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


def _transcript(returncode, argv, *, role="s2", model="deepseek-flash", trailer=""):
    assistant = json.dumps({"tool": "command", "arguments": {"argv": argv}}) + trailer
    return {
        "schema": "dualbench-alo-v2",
        "assignments": {role: {"role_id": role, "model_id": model, "identity_id": "w", "bandwidth": 1,
                               "authority": ["inspect"]}},
        "worker_transcripts": {
            role: [
                {"role": "assistant", "content": assistant},
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
        _record("u-empty", "g2", "m1", "task-empty"),
        _record("u-main", "g1-main", "m1", "task-fail", model="mimo-v2.6-flash", micro=500),
    ]
    (run / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    (run / "checkpoints" / "u-fail" / "adapter.json").write_text(
        json.dumps(_transcript(1, ["python3", "-m", "pytest", "tests/test_bug.py"],
                               role="s1", model="deepseek-flash", trailer="\n<system>stray")),
        encoding="utf-8")
    (run / "checkpoints" / "u-pass").mkdir(parents=True, exist_ok=True)
    both = _transcript(0, ["python3", "-m", "pytest", "tests/test_bug.py"], role="s2", model="mimo-v2.6-flash")
    both["assignments"]["s1"] = {"role_id": "s1", "model_id": "deepseek-flash", "identity_id": "a",
                                 "bandwidth": 1, "authority": ["inspect"]}
    both["worker_transcripts"]["s1"] = _transcript(
        0, ["python3", "alo_repro.py"], role="s1", model="deepseek-flash")["worker_transcripts"]["s1"]
    # The pass unit's definite test signal is on s2 (mimo-flash). s1's alo_repro
    # run is a single after-code of 0, which is not by itself a pass.
    (run / "checkpoints" / "u-pass" / "adapter.json").write_text(json.dumps(both), encoding="utf-8")
    (run / "checkpoints" / "u-empty").mkdir(parents=True)
    (run / "checkpoints" / "u-empty" / "adapter.json").write_text(
        json.dumps({"schema": "dualbench-alo-v2", "worker_transcripts": {},
                    "assignments": {"s2": {"model_id": "glm-5.3-flash"}}}), encoding="utf-8")
    evaluation.mkdir()
    (evaluation / "m1_units.csv").write_text(
        "unit_id,arm,task,resolved,cost_cny\n"
        "u-fail,g2,task-fail,False,0.001\n"
        "u-pass,g2,task-pass,True,0.002\n"
        "u-empty,g2,task-empty,False,0.001\n"
        "u-main,g1-main,task-fail,False,0.0005\n",
        encoding="utf-8")
    report = replay_run(run, evaluation)
    assert report["records_proxy"]["proxy"] is True
    assert report["records_proxy"]["precision_recall"] is None
    assert report["records_proxy"]["posterior_updated"] is False
    assert report["records_proxy"]["command_calls"] == 6
    metrics = report["transcript"]["precision_recall"]
    assert metrics["labeled"] + metrics["unknown_excluded"] == 3
    assert metrics["unknown_excluded"] == 1
    assert metrics["precision"] == 1
    assert metrics["true_negative"] == 1
    belief = {row["model"]: row for row in report["transcript"]["posterior"]["belief"]}
    assert belief["deepseek-flash"]["beta"] > 1
    assert belief["mimo-v2.6-flash"]["alpha"] > 1
    assert "glm-5.3-flash" not in belief
    assert report["comparison"]["m1"]["oracle"]["tasks"] == 3
    text = replay_markdown(report)
    assert "Records-only proxy" in text
    assert "not a repro-v1 pass or fail" in text
    assert "--execute" in text

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "records.jsonl").write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    proxy_only = replay_run(bare, evaluation)
    assert proxy_only["adapters_found"] == 0
    bare_metrics = proxy_only["transcript"]["precision_recall"]
    assert bare_metrics["labeled"] == 0
    assert bare_metrics["unknown_excluded"] == 1
    assert proxy_only["transcript"]["beta_gt_1"] is False


def test_execute_mode_is_injected_and_labels_its_source(tmp_path):
    run = tmp_path / "run"
    evaluation = tmp_path / "eval"
    unit = "m1.SWE.task.g2.B0"
    checkpoint = run / "checkpoints" / unit
    checkpoint.mkdir(parents=True)
    record = _record(unit, "g2", "m1", "task")
    record["edit_log"].append({"tool": "edit_file", "node": "s1", "paths": ["tests/test_bug.py"]})
    (run / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
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
        return {"returncode": 0, "stdout": json.dumps(
            {"before": None, "after": 1, "tests": 0, "applied": True, "runner": "pytest", "reason": None}
        ) + "\n", "stderr": ""}

    report = replay_run(run, evaluation, execute=True, executor=runner)
    assert seen["argv"][0] == "docker"
    assert "--network=none" in seen["argv"]
    assert "example.invalid/alo:test" in seen["argv"]
    assert "tests/test_bug.py" in seen["argv"]
    assert report["execute_replay"]["proxy"] is False
    assert report["execute_replay"]["beta_gt_1"] is False
    assert report["execute_replay"]["precision_recall"]["true_negative"] == 1
    direct = execute_public_checks(
        image="example.invalid/alo:test", patch_text="diff --git a/tests/test_bug.py b/tests/test_bug.py\n",
        test_paths=["tests/test_bug.py"], runner=runner)
    assert direct["source"] == "execute"
    assert direct["signal"] == "fail"
    assert direct["tests_returncode"] == 0
    assert direct["repro_after_returncode"] == 1
    missing = execute_public_checks(
        image="example.invalid/alo:test", patch_text="",
        runner=lambda argv, timeout_s: {"returncode": 0, "stdout": json.dumps(
            {"before": None, "after": None, "tests": None, "applied": True,
             "runner": None, "reason": "no_public_test_runner"}) + "\n", "stderr": ""})
    assert missing["signal"] == "unknown"
    assert missing["execute_reason"] == "no_public_test_runner"
    from alo_cloud.signal_replay import _EXECUTE_DRIVER
    compile(_EXECUTE_DRIVER, "<execute-driver>", "exec")
    assert "/opt/miniconda3/envs/testbed/bin/python" in _EXECUTE_DRIVER
    assert "django_runtests" in _EXECUTE_DRIVER
    assert "sympy_bin_test" in _EXECUTE_DRIVER
    assert "json.dumps" in _EXECUTE_DRIVER


def test_execute_names_the_container_and_removes_it_on_timeout():
    import subprocess
    from alo_cloud.signal_replay import REPLAY_CONTAINER_LABEL, execute_public_checks

    seen = {}
    removed = []

    def runner(argv, timeout_s):
        seen["argv"] = argv
        seen["timeout"] = timeout_s
        raise subprocess.TimeoutExpired(argv, timeout_s)

    evidence = execute_public_checks(
        image="example.invalid/alo:test", patch_text="", runner=runner, cleanup=removed.append,
        timeout_s=5)
    assert evidence["signal"] == "unknown"
    assert seen["argv"][seen["argv"].index("--name") + 1] == removed[0]
    assert REPLAY_CONTAINER_LABEL in seen["argv"]
    assert "--rm" not in seen["argv"]
    assert removed[0].startswith("alo-repro-")


def test_output_dir_defaults_outside_the_run_and_refuses_the_run(tmp_path, monkeypatch):
    import importlib.util
    from alo_cloud.signal_replay import resolve_output_dir

    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "out"
    outside.mkdir()
    assert resolve_output_dir(outside, run) == outside.resolve()
    try:
        resolve_output_dir(run / "nested", run)
    except ValueError as error:
        assert "inside" in str(error)
    else:
        raise AssertionError("output inside the run dir was accepted")
    monkeypatch.chdir(outside)
    assert resolve_output_dir(None, run) == outside.resolve()
    script = Path(__file__).resolve().parents[1] / "scripts" / "replay_signal.py"
    spec = importlib.util.spec_from_file_location("replay_signal_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "m1_units.csv").write_text("unit_id,resolved\n", encoding="utf-8")
    with pytest.raises(SystemExit) as refused:
        module.main([
            "--run-dir", str(run), "--eval-dir", str(eval_dir), "--output-dir", str(run / "inside"),
        ])
    assert refused.value.code == 2
    assert module.main([
        "--run-dir", str(run), "--eval-dir", str(eval_dir), "--output-dir", str(outside),
    ]) == 0
    assert (outside / "repro_v1_replay.json").is_file()
    assert not (run / "repro_v1_replay.json").exists()


def test_settled_ledger_keeps_deepseek_over_50_unless_the_schedule_shrinks():
    from alo_cloud.followup_cost import ledger_full_scenarios, ledger_scope_options

    full = ledger_full_scenarios()
    deepseek = [scenario["projected_cny"]["deepseek-flash"] for scenario in full.values()]
    assert min(deepseek) > 61
    assert max(deepseek) < 76
    assert all(scenario["over_cap_cny"]["deepseek-flash"] > 0 for scenario in full.values())
    assert full["light_overall_rates"]["projected_cny"]["deepseek-flash"] < full["default_overall_rates"]["projected_cny"]["deepseek-flash"]
    options = ledger_scope_options()
    assert set(options) == {
        "new_arms_only_light",
        "new_arms_only_default_requests",
        "repeat_plus_g1_deepseek_only",
        "repeat_plus_g1_and_capped_g2_fixed",
    }
    for name, option in options.items():
        assert option["projected_cny"]["deepseek-flash"] <= 50, name
        assert not option["over_cap_cny"], name
    assert options["repeat_plus_g1_and_capped_g2_fixed"]["g2_fraction"] < 1
    assert options["new_arms_only_light"]["projected_cny"]["mimo-v2.6-pro"] < 50
    from alo_cloud.followup_cost import ledger_arm_costs
    forty = ledger_arm_costs(light=True, include_g2_random=False, g2_fraction=0.40)
    assert abs(forty["projected_cny"]["deepseek-flash"] - 49.02) < 0.02
    assert abs(50 - forty["projected_cny"]["deepseek-flash"] - 0.98) < 0.02
    fraction = options["repeat_plus_g1_and_capped_g2_fixed"]["g2_fraction"]
    at_g2_rate = ledger_arm_costs(
        light=True, deepseek_rate="g2", include_g2_random=False, g2_fraction=fraction)
    assert abs(at_g2_rate["projected_cny"]["deepseek-flash"] - 51.83) < 0.05


def test_settled_ledger_reader_is_read_only(tmp_path):
    from alo_cloud.budget import BudgetLedger
    from alo_cloud.contracts import MODEL_POOL
    from alo_cloud.followup_cost import summarize_settled_ledger

    path = tmp_path / "budget.sqlite3"
    ledger = BudgetLedger.create(
        path, model_limits_microcny={model: 50_000_000 for model in MODEL_POOL},
        opening_commitments_microcny={model: 0 for model in MODEL_POOL},
        max_requests=10, prior_requests=0, total_limit_microcny=200_000_000, run_mode="dry-run")
    digest = "ab" * 32
    ledger.reserve("g1", "deepseek-flash", 1_000_000, payload_sha256=digest,
                   unit_id="m1.SWE-bench_Lite.django__django-1.g1-main.B0", unit_request_cap=4)
    ledger.settle_known("g1", 400_000)
    ledger.reserve("g2", "deepseek-flash", 2_000_000, payload_sha256=digest,
                   unit_id="m1.SWE-bench_Lite.django__django-1.g2.B0", unit_request_cap=4)
    ledger.settle_known("g2", 800_000)
    before = path.read_bytes()
    summary = summarize_settled_ledger(path)
    assert path.read_bytes() == before
    assert summary["read_only"] is True
    assert summary["g1_main_requests"] == 1
    assert summary["g2_requests"] == 1
    assert summary["ledger_cny"]["deepseek-flash"] == 1.2
    assert summary["cny_per_request"]["deepseek-flash"] == 0.6
    assert summary["g2_deepseek_cny_per_request"] == 0.8


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
