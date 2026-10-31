"""Offline checks for prices, the study ledger, the freeze, and dry-run."""
import json
from pathlib import Path
import subprocess

import pytest

from alo_cloud.budget import BudgetLedger, BudgetStopped
from alo_cloud.contracts import (
    G1_MAIN_MODEL, G1_STRONG_MODEL, MODEL_POOL, PER_MODEL_LIMIT_MICROCNY, STUDY_TOTAL_LIMIT_MICROCNY,
)
from alo_cloud.errors import PilotStopped
from alo_cloud.collect import collect_sealed
from alo_cloud.evaluate import (
    PROGRAMBENCH_COMMIT, PROGRAMBENCH_TESTS_REVISION, score_sealed, swebench_command, programbench_env,
)
from alo_cloud.images import programbench_instance_image, swebench_instance_image
from alo_cloud.sandbox import DockerSandbox
from alo_cloud.freeze import all_task_ids, freeze_sha256, iter_units, select_units
from alo_cloud.prices import UnpricedModel, charge_microcny, request_ceiling_microcny
from alo_cloud.projection import project, within_budget
from alo_cloud.run import dry_run_sender, run_schedule
from alo_cloud.services import StudyServices


def test_prices_round_up_and_missing_cache_uses_the_miss_rate():
    assert charge_microcny("deepseek-flash", {"input_tokens": 1_000_000, "output_tokens": 0}) == 2_000_000
    assert charge_microcny("mimo-v2.6-pro", {"input_tokens": 1, "output_tokens": 0,
                                             "cache_hit_tokens": 1}) == 1
    assert charge_microcny("glm-5.3-flash", {"input_tokens": 0, "output_tokens": 1}) == 3
    assert charge_microcny("mimo-v2.6-flash", {"input_tokens": 100, "output_tokens": 0}) == 100
    split = charge_microcny("mimo-v2.6-flash", {"input_tokens": 100, "output_tokens": 0,
                                                "cache_hit_tokens": 40})
    assert split == 61
    assert request_ceiling_microcny("deepseek-flash", 20, 64) == charge_microcny(
        "deepseek-flash", {"input_tokens": 20, "output_tokens": 64, "cache_hit_tokens": None})
    with pytest.raises(UnpricedModel):
        charge_microcny("not-a-model", {"input_tokens": 1, "output_tokens": 1})
    with pytest.raises(ValueError):
        charge_microcny("glm-5.3-flash", {"input_tokens": 3, "output_tokens": 0, "cache_hit_tokens": 9})


def test_reasoning_tokens_do_not_change_the_charge():
    usage = {"input_tokens": 10, "output_tokens": 4, "reasoning_tokens": 4}
    assert charge_microcny("glm-5.3-flash", usage) == charge_microcny(
        "glm-5.3-flash", {"input_tokens": 10, "output_tokens": 4})


def test_total_cap_refuses_when_the_sum_would_pass_a_single_model_cap(tmp_path):
    path = tmp_path / "budget.sqlite3"
    ledger = BudgetLedger.create(
        path, model_limits_microcny={"one": 80, "two": 80},
        opening_commitments_microcny={"one": 0, "two": 0},
        max_requests=5, prior_requests=0, total_limit_microcny=100)
    ledger.reserve("request-1", "one", 60, payload_sha256="a" * 64)
    ledger.settle_known("request-1", 60)
    with pytest.raises(BudgetStopped, match="total_cny"):
        ledger.reserve("request-2", "two", 50, payload_sha256="b" * 64)
    restored = BudgetLedger(path)
    assert restored.snapshot()["total_committed_microcny"] == 60
    assert restored.snapshot()["remaining_total_microcny"] == 40


def test_unit_scope_does_not_spend_the_next_units_request_cap(tmp_path):
    path = tmp_path / "budget.sqlite3"
    ledger = BudgetLedger.create(
        path, model_limits_microcny={model: PER_MODEL_LIMIT_MICROCNY for model in MODEL_POOL},
        opening_commitments_microcny={model: 0 for model in MODEL_POOL},
        max_requests=32, prior_requests=0, max_tokens=900_000, prior_tokens=0,
        max_seconds_ms=1_200_000, prior_seconds_ms=0, max_tools=120, prior_tools=0,
        total_limit_microcny=STUDY_TOTAL_LIMIT_MICROCNY, run_mode="dry-run")
    ledger.reserve("req-unit-a-1", "mimo-v2.6-flash", 10, payload_sha256="a" * 64,
                   tokens=5, seconds_ms=5, unit_id="unit-a", unit_request_cap=1)
    ledger.settle_known("req-unit-a-1", 4, tokens=5, seconds_ms=5)
    with pytest.raises(BudgetStopped, match="unit_resource_exhausted"):
        ledger.reserve("req-unit-a-2", "mimo-v2.6-flash", 10, payload_sha256="b" * 64,
                       tokens=5, seconds_ms=5, unit_id="unit-a", unit_request_cap=1)
    ledger.reserve("req-unit-b-1", "mimo-v2.6-pro", 10, payload_sha256="c" * 64,
                   tokens=5, seconds_ms=5, unit_id="unit-b", unit_request_cap=1)
    ledger.settle_known("req-unit-b-1", 4, tokens=5, seconds_ms=5)
    assert ledger.remaining_resources("mimo-v2.6-pro", unit_id="unit-b", unit_request_cap=1)["requests"] == 0
    assert ledger.remaining_resources("mimo-v2.6-flash", unit_id="unit-b", unit_request_cap=1)["requests"] == 0


def test_request_ceiling_is_what_gets_reserved(tmp_path, monkeypatch):
    from alo_cloud.providers import POOL_ENDPOINTS
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-credential-deepseek")
    monkeypatch.setenv("MIMO_API_KEY", "unit-test-credential-mimo")
    monkeypatch.setenv("ZAI_API_KEY", "unit-test-credential-zai")
    seen = []

    def sender(url, body, headers):
        seen.append(body["model"])
        return {"model": body["model"], "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}

    ledger = BudgetLedger.create(
        tmp_path / "budget.sqlite3",
        model_limits_microcny={model: PER_MODEL_LIMIT_MICROCNY for model in MODEL_POOL},
        opening_commitments_microcny={model: 0 for model in MODEL_POOL},
        max_requests=32, prior_requests=0, max_tokens=900_000, prior_tokens=0,
        max_seconds_ms=1_200_000, prior_seconds_ms=0, max_tools=120, prior_tools=0)

    def ceiling(model, messages, max_tokens):
        del model, messages, max_tokens
        return 80

    services = StudyServices(
        tmp_path / "state", ledger, unit="m1_b0", per_request_microcny_ceiling=100,
        charge_microcny=lambda model, usage: 1, sender=sender, request_ceiling=ceiling)
    services.complete_for_model("deepseek-flash", [{"role": "user", "content": "hi"}], max_tokens=16)
    assert seen == ["deepseek-flash"]
    assert POOL_ENDPOINTS["deepseek-flash"][1]
    assert ledger.snapshot()["committed_or_reserved_microcny"]["deepseek-flash"] == 1

    def too_high(model, messages, max_tokens):
        del model, messages, max_tokens
        return 101

    blocked = StudyServices(
        tmp_path / "state-b", ledger, unit="m1_b0", per_request_microcny_ceiling=100,
        charge_microcny=lambda model, usage: 1, sender=sender, request_ceiling=too_high)
    with pytest.raises(PilotStopped, match="ceiling"):
        blocked.complete_for_model("deepseek-flash", [{"role": "user", "content": "hi"}], max_tokens=16)
    assert seen == ["deepseek-flash"]


def test_projection_fits_under_the_budget_with_headroom():
    summary = project()
    assert within_budget(summary)
    assert summary["total_upper_microcny"] <= 160_000_000
    assert summary["headroom_ratio"] >= 0.20
    assert summary["token_estimate"]["input_tokens"] == 18000
    assert summary["token_estimate"]["output_tokens"] == 4096
    assert "108.120576" in Path("docs/cost_projection.md").read_text(encoding="utf-8")
    assert G1_STRONG_MODEL == "mimo-v2.6-pro"
    assert G1_MAIN_MODEL == "mimo-v2.6-flash"
    assert summary["m1_units"] == 96
    assert summary["m2_units"] == 84


def test_freeze_ids_are_unique_and_hash_matches_the_document():
    ids = all_task_ids()
    assert len(ids) == len(set(ids)) == 6 * 2 + 4 * 2 + 2 * 2
    for task_id in ("sqlfluff__sqlfluff-2419", "chmln__sd.87d1ba5", "sqlfluff__sqlfluff-1625"):
        assert task_id not in ids
    m1 = tuple(iter_units("m1"))
    first = [unit.arm for unit in m1 if unit.task_id == "scikit-learn__scikit-learn-13584" and unit.phase == "B0"]
    assert first == ["g1-main", "g1-strong", "g2", "g3"]
    b1 = [unit.arm for unit in m1 if unit.task_id == "scikit-learn__scikit-learn-13584" and unit.phase == "B1-self"]
    assert b1 == first
    text = Path("docs/protocol_freeze.md").read_text(encoding="utf-8")
    assert f"freeze_sha256: {freeze_sha256()}" in text
    for task_id in ids:
        assert task_id in text
    for unit in tuple(iter_units("m1")) + tuple(iter_units("m2")):
        assert len(f"req-{unit.unit_id}-1") <= 160


def test_dry_run_resumes_without_network(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("dry-run opened a socket")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    units = select_units("m1", arms=("g1-main",), tasks=("django__django-11583",))
    assert len(units) == 2
    first = run_schedule(units, tmp_path, live=False, sender=dry_run_sender)
    assert [row["status"] for row in first] == ["completed", "completed"]
    assert first[1]["phase"] == "B1-self"
    second = run_schedule(units, tmp_path, live=False, sender=dry_run_sender)
    assert [row["unit_id"] for row in second] == [row["unit_id"] for row in first]
    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    ledger = BudgetLedger(tmp_path / "budget.sqlite3")
    assert ledger.snapshot()["run_mode"] == "dry-run"
    assert ledger.snapshot()["requests"] == 2
    with pytest.raises(BudgetStopped, match="run_mode"):
        run_schedule(units, tmp_path, live=True)


def test_g2_dry_run_exports_learning_state(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    units = tuple(unit for unit in select_units("m2", arms=("g2",))
                  if unit.phase == "learning" and unit.order_id == "order-1"
                  and unit.track == "SWE-bench_Lite" and unit.position < 2)
    rows = run_schedule(units, tmp_path, live=False, sender=dry_run_sender)
    assert [row["status"] for row in rows] == ["completed"] * 4
    assert rows[3].get("parent_note") is None
    assert (tmp_path / "checkpoints" / rows[0]["unit_id"] / "learning_state.json").is_file()


def test_live_without_keys_does_not_send(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    units = select_units("m1", arms=("g1-main",), tasks=("django__django-11583",))
    with pytest.raises(PilotStopped, match="credential"):
        run_schedule(units, tmp_path, live=True)


def test_evaluator_does_not_invent_a_score(tmp_path):
    report = score_sealed(tmp_path, execute=False)
    assert report["swebench"]["status"] == "swe_predictions_missing"
    assert report["programbench"]["status"] == "programbench_submissions_missing"
    rendered = json.dumps(report)
    assert "resolved" not in rendered
    command = swebench_command(tmp_path / "preds.jsonl", run_id="alo-m1m2", instance_ids=("a",), split="test")
    assert command[2] == "swebench.harness.run_evaluation"
    assert "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2" in command
    assert PROGRAMBENCH_COMMIT in rendered
    executed = score_sealed(tmp_path, execute=True)
    assert executed["swebench"]["status"] == "unavailable"


def _load_smoke():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_test.py"
    spec = importlib.util.spec_from_file_location("alo_smoke_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_plan_stays_under_half_a_cny_and_does_not_send(capsys):
    smoke = _load_smoke()
    total = sum(amount for _model, amount in smoke.smoke_ceilings())
    assert total < 500_000
    assert smoke.main([]) == 0
    captured = capsys.readouterr().out
    assert "not_sent" in captured
    assert "DEEPSEEK_API_KEY" not in captured
    assert "Authorization" not in captured


def test_official_images_and_sandbox_limits(tmp_path):
    assert swebench_instance_image("django__django-11583") == (
        "swebench/sweb.eval.x86_64.django_1776_django-11583:latest")
    assert programbench_instance_image("ffmpeg__ffmpeg.360a402") == (
        "programbench/ffmpeg_1776_ffmpeg.360a402:task_cleanroom_v6")
    box = DockerSandbox(tmp_path, image="example.invalid/alo:test", memory="4g", pids_limit="512")
    argv = box.docker_argv(["python", "-c", "print(1)"])
    assert "--network=none" in argv
    assert "--memory=4g" in argv
    assert "--pids-limit=512" in argv
    assert "example.invalid/alo:test" in argv
    assert argv.index("--network=none") < argv.index("example.invalid/alo:test")


def _git_commit(path, filename, text):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "init", "-q", str(path)])
    subprocess.check_call(["git", "-C", str(path), "config", "user.email", "t@example.com"])
    subprocess.check_call(["git", "-C", str(path), "config", "user.name", "t"])
    (path / filename).write_text(text, encoding="utf-8")
    subprocess.check_call(["git", "-C", str(path), "add", filename])
    subprocess.check_call(["git", "-C", str(path), "commit", "-q", "-m", "base"])
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def test_image_seed_verifies_head_and_deletes_the_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    fixture = tmp_path / "fixture"
    base = _git_commit(fixture, "a.py", "base\n")
    calls = []

    def docker(argv):
        calls.append(list(argv))
        if argv[1] == "cp":
            dest = Path(argv[-1])
            shutil_copy_repo(fixture, dest)
        completed = type("Completed", (), {})()
        completed.returncode = 0
        completed.stdout = ""
        completed.stderr = ""
        return completed

    data = tmp_path / "data" / "SWE-bench_Lite"
    data.mkdir(parents=True)
    (data / "django__django-11583.json").write_text(json.dumps({
        "repo": "django/django", "base_commit": base, "problem_statement": "public bug",
        "patch": "diff --git a/secret",
    }), encoding="utf-8")
    units = select_units("m1", arms=("g1-main",), tasks=("django__django-11583",))
    rows = run_schedule(units[:1], tmp_path / "run", live=False, data_dir=tmp_path / "data",
                        sender=dry_run_sender, image_seeder=docker)
    assert rows[0]["status"] == "completed"
    assert rows[0]["sandbox_workdir"] == "/testbed"
    assert rows[0]["base_commit"] == base
    assert rows[0]["image_head"] == base
    assert "diff --git a/secret" not in json.dumps(rows[0])
    assert any(argv[:3] == ["docker", "cp", f"alo-seed-{calls[0][3][9:]}:/testbed/."] or (
        len(argv) > 2 and argv[1] == "cp" and argv[2].endswith(":/testbed/.")) for argv in calls)
    unit_id = rows[0]["unit_id"]
    meta = json.loads((tmp_path / "run" / "checkpoints" / unit_id / "metadata.json").read_text())
    assert meta["image_head"] == base
    assert meta["base_commit"] == base
    assert not (tmp_path / "run" / "units" / unit_id).exists()
    assert not (tmp_path / "run" / "checkpoints" / unit_id / "workspace").exists()
    assert (tmp_path / "run" / "checkpoints" / unit_id / "model.patch").is_file()
    patch = (tmp_path / "run" / "checkpoints" / unit_id / "model.patch").read_text(encoding="utf-8")
    assert len(patch.encode()) < 100_000
    box = DockerSandbox(
        tmp_path, image="example.invalid/swe:latest", workdir="/testbed",
        mount_points=("/testbed",), path=(
            "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/bin"))
    argv = box.docker_argv(["python", "-c", "import sklearn"])
    path_env = argv[argv.index("--env") + 1] if False else next(
        item for item in argv if item.startswith("PATH="))
    assert path_env.index("/opt/miniconda3/envs/testbed/bin") < path_env.index("/opt/miniconda3/bin")
    assert "--workdir" in argv and argv[argv.index("--workdir") + 1] == "/testbed"
    assert "dst=/testbed" in " ".join(argv)
    assert "dst=/workspace" not in " ".join(argv)


def shutil_copy_repo(source, dest):
    import shutil
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_symlink():
            continue
        target = dest / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            target.chmod(path.stat().st_mode & 0o777)


def test_collect_keeps_one_prediction_per_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("PROGRAMBENCH_HF_REVISION", "not-the-pin")
    run = tmp_path / "run"
    bases = {}
    for unit, text in (("m1.SWE.django__django-11583.g1-main.B0", "one\n"),
                       ("m1.SWE.django__django-11583.g2.B0", "two\n")):
        workspace = run / "checkpoints" / unit / "workspace"
        workspace.mkdir(parents=True)
        subprocess.check_call(["git", "init", "-q", str(workspace)])
        subprocess.check_call(["git", "-C", str(workspace), "config", "user.email", "t@example.com"])
        subprocess.check_call(["git", "-C", str(workspace), "config", "user.name", "t"])
        (workspace / "a.py").write_text("base\n", encoding="utf-8")
        subprocess.check_call(["git", "-C", str(workspace), "add", "a.py"])
        subprocess.check_call(["git", "-C", str(workspace), "commit", "-q", "-m", "base"])
        bases[unit] = subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True).strip()
        (workspace / "a.py").write_text(text, encoding="utf-8")
    program = run / "checkpoints" / "m1.ProgramBench.ffmpeg__ffmpeg.360a402.g1-main.B0" / "workspace"
    program.mkdir(parents=True)
    (program / "main.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
    with (run / "records.jsonl").open("w", encoding="utf-8") as handle:
        for unit, _text in (("m1.SWE.django__django-11583.g1-main.B0", "one\n"),
                            ("m1.SWE.django__django-11583.g2.B0", "two\n")):
            handle.write(json.dumps({
                "unit_id": unit, "task_id": "django__django-11583", "track": "SWE-bench_Lite",
                "arm": unit.split(".")[3], "phase": "B0", "study": "m1", "base_commit": bases[unit],
            }) + "\n")
        handle.write(json.dumps({
            "unit_id": "m1.ProgramBench.ffmpeg__ffmpeg.360a402.g1-main.B0",
            "task_id": "ffmpeg__ffmpeg.360a402", "track": "ProgramBench",
            "arm": "g1-main", "phase": "B0", "study": "m1",
        }) + "\n")
    summary = collect_sealed(run)
    assert summary["predictions"] == 2
    rows = [json.loads(line) for line in (run / "sealed" / "swe_predictions.jsonl").read_text().splitlines()]
    assert {row["model_name_or_path"] for row in rows} == set(bases)
    assert {row["instance_id"] for row in rows} == {"django__django-11583"}
    assert all("one" in row["model_patch"] or "two" in row["model_patch"] for row in rows)
    tarball = (run / "sealed" / "programbench" / "m1.ProgramBench.ffmpeg__ffmpeg.360a402.g1-main.B0"
               / "ffmpeg__ffmpeg.360a402" / "submission.tar.gz")
    assert tarball.is_file()
    report = score_sealed(run, execute=False, programbench_repo="/tmp/programbench-src")
    assert len(report["swebench"]["units"]) == 2
    run_ids = [item["command"][item["command"].index("--run_id") + 1] for item in report["swebench"]["units"]]
    assert len(set(run_ids)) == 2
    assert all("6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2" in item["command"]
               for item in report["swebench"]["units"])
    assert report["programbench"]["hf_revision"] == PROGRAMBENCH_TESTS_REVISION
    assert programbench_env()["PROGRAMBENCH_HF_REVISION"] == PROGRAMBENCH_TESTS_REVISION
    assert "DEEPSEEK_API_KEY" not in programbench_env({"DEEPSEEK_API_KEY": "secret", "PATH": "/usr/bin"})


def test_source_patch_skips_gitignored_build_output(tmp_path):
    from alo_cloud.collect import workspace_patch
    repo = tmp_path / "repo"
    base = _git_commit(repo, "a.py", "base\n")
    (repo / ".gitignore").write_text("*.o\n", encoding="utf-8")
    subprocess.check_call(["git", "-C", str(repo), "add", ".gitignore"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-q", "-m", "ignore"])
    base = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    (repo / "a.py").write_text("changed\n", encoding="utf-8")
    (repo / "new.py").write_text("print(1)\n", encoding="utf-8")
    (repo / "build.o").write_bytes(b"\x7fELF")
    patch = workspace_patch(repo, base)
    assert "changed" in patch
    assert "new.py" in patch
    assert "build.o" not in patch


def test_execute_only_binary_stays_in_the_image(tmp_path):
    from alo_cloud.collect import _pack_submission
    from alo_cloud.workspace_prep import release_unreadable_executable
    root = tmp_path / "ws"
    root.mkdir()
    (root / "README.md").write_text("docs\n", encoding="utf-8")
    exe = root / "executable"
    exe.write_bytes(b"#!/bin/sh\n")
    exe.chmod(0o111)
    assert release_unreadable_executable(root) is True
    assert not exe.exists()
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "README.md").write_text("docs\n", encoding="utf-8")
    binary = kept / "executable"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    assert release_unreadable_executable(kept) is False
    assert binary.is_file()
    box = DockerSandbox(root, image="programbench/example:task_cleanroom_v6", merge_image_workspace=True)
    argv = box.docker_argv(["./executable"])
    rendered = " ".join(argv)
    assert "--network=none" in argv
    assert "--read-only" not in argv
    assert "dst=/agent" in rendered
    assert "dst=/workspace" not in rendered
    assert "cp -a" not in rendered
    assert "--exclude=.git" in rendered
    assert "--exclude=executable" in rendered
    assert rendered.count("--exclude=.git") >= 2
    archive = tmp_path / "submission.tar.gz"
    (root / "main.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("x\n", encoding="utf-8")
    exe.write_bytes(b"bin")
    _pack_submission(root, archive)
    import tarfile
    names = tarfile.open(archive, "r:gz").getnames()
    assert "main.c" in names
    assert "README.md" in names
    assert "executable" not in names
    assert all(".git" not in name for name in names)


def test_resume_applies_the_sealed_patch_onto_a_fresh_seed(tmp_path):
    from alo_cloud.workspace_prep import apply_patch
    from alo_cloud.collect import workspace_patch
    origin = tmp_path / "origin"
    base = _git_commit(origin, "a.py", "base\n")
    (origin / "a.py").write_text("edited\n", encoding="utf-8")
    patch = workspace_patch(origin, base)
    fresh = tmp_path / "fresh"
    shutil_copy_repo(origin, fresh)
    subprocess.check_call(["git", "-C", str(fresh), "reset", "--hard", "HEAD"])
    assert (fresh / "a.py").read_text(encoding="utf-8") == "base\n"
    apply_patch(fresh, patch)
    assert (fresh / "a.py").read_text(encoding="utf-8") == "edited\n"


def test_live_swe_without_checkout_metadata_does_not_call_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    for name in ("DEEPSEEK_API_KEY", "MIMO_API_KEY", "ZAI_API_KEY"):
        monkeypatch.setenv(name, "present")
    data = tmp_path / "data" / "SWE-bench_Lite"
    data.mkdir(parents=True)
    (data / "django__django-11583.json").write_text(
        json.dumps({"problem_statement": "public"}), encoding="utf-8")
    units = select_units("m1", arms=("g1-main",), tasks=("django__django-11583",))
    rows = run_schedule(units[:1], tmp_path / "run", live=True, data_dir=tmp_path / "data")
    assert rows[0]["status"] == "blocked"
    assert rows[0]["reason"] == "swe_checkout_metadata_missing"
    assert rows[0]["sandbox_image"] == swebench_instance_image("django__django-11583")


def _mode_only_child(path):
    base = _git_commit(path, "a.py", "print(1)\n")
    subprocess.check_call(["git", "-C", str(path), "add", "--chmod=+x", "a.py"])
    (path / "a.py").chmod(0o755)
    subprocess.check_call(["git", "-C", str(path), "commit", "-q", "-m", "SWE-bench", "--author", "SWE-bench <swe@example.com>"])
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    return base, head


def test_mode_only_parent_is_the_patch_base(tmp_path):
    from alo_cloud.collect import workspace_patch
    from alo_cloud.workspace_prep import apply_patch, resolve_image_head
    repo = tmp_path / "repo"
    base, head = _mode_only_child(repo)
    assert head != base
    assert resolve_image_head(repo, base) == head
    (repo / "a.py").write_text("print(2)\n", encoding="utf-8")
    (repo / "a.py").chmod(0o755)
    against_head = workspace_patch(repo, head)
    against_base = workspace_patch(repo, base)
    assert "print(2)" in against_head
    assert "old mode" not in against_head
    assert "old mode" in against_base
    fresh = tmp_path / "fresh"
    shutil_copy_repo(repo, fresh)
    subprocess.check_call(["git", "-C", str(fresh), "reset", "--hard", "HEAD"])
    apply_patch(fresh, against_head)
    assert (fresh / "a.py").read_text(encoding="utf-8") == "print(2)\n"
    subprocess.check_call(["git", "-C", str(repo), "add", "a.py"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-q", "-m", "content"])
    assert resolve_image_head(repo, base) is None


def test_content_commit_on_base_is_rejected(tmp_path):
    from alo_cloud.workspace_prep import resolve_image_head
    repo = tmp_path / "repo"
    base = _git_commit(repo, "a.py", "print(1)\n")
    (repo / "a.py").write_text("print(2)\n", encoding="utf-8")
    subprocess.check_call(["git", "-C", str(repo), "add", "a.py"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-q", "-m", "content"])
    assert resolve_image_head(repo, base) is None


def test_numstat_content_blocks_a_mode_summary(tmp_path):
    from types import SimpleNamespace
    from alo_cloud.workspace_prep import resolve_image_head
    base = "b" * 40
    head = "a" * 40

    def git(argv, *, numstat="0\t0\ta.py\n", summary=" mode change 100644 => 100755 a.py\n"):
        spec = argv[3:]
        if spec == ["rev-parse", "HEAD"]:
            stdout = head + "\n"
        elif spec == ["rev-parse", "HEAD^"]:
            stdout = base + "\n"
        elif spec[:2] == ["diff", "--summary"]:
            stdout = summary
        elif spec[:2] == ["diff", "--numstat"]:
            stdout = numstat
        else:
            raise AssertionError(spec)
        return SimpleNamespace(returncode=0, stdout=stdout)

    assert resolve_image_head(tmp_path, base, git=lambda argv: git(argv)) == head
    assert resolve_image_head(
        tmp_path, base, git=lambda argv: git(argv, numstat="1\t0\ta.py\n")) is None
    assert resolve_image_head(
        tmp_path, base,
        git=lambda argv: git(argv, summary=" create mode 100755 a.py\n")) is None


def _copying_docker(source):
    from types import SimpleNamespace
    calls = []

    def docker(argv):
        calls.append(list(argv))
        if argv[1] == "cp":
            shutil_copy_repo(source, Path(argv[-1]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return docker, calls


def test_schedule_records_mode_only_image_head_and_resume_applies_it(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    fixture = tmp_path / "fixture"
    base, head = _mode_only_child(fixture)
    docker, _calls = _copying_docker(fixture)
    data = tmp_path / "data" / "SWE-bench_Lite"
    data.mkdir(parents=True)
    (data / "django__django-11583.json").write_text(json.dumps({
        "repo": "django/django", "base_commit": base, "problem_statement": "public bug",
    }), encoding="utf-8")
    units = select_units("m1", arms=("g1-main",), tasks=("django__django-11583",))
    assert [unit.phase for unit in units] == ["B0", "B1-self"]
    rows = run_schedule(units[:1], tmp_path / "run", live=False, data_dir=tmp_path / "data",
                        sender=dry_run_sender, image_seeder=docker)
    assert rows[0]["status"] == "completed"
    assert rows[0]["base_commit"] == base
    assert rows[0]["image_head"] == head
    unit_id = rows[0]["unit_id"]
    patch_path = tmp_path / "run" / "checkpoints" / unit_id / "model.patch"
    assert "old mode" not in patch_path.read_text(encoding="utf-8")
    meta = json.loads((tmp_path / "run" / "checkpoints" / unit_id / "metadata.json").read_text())
    assert meta["image_head"] == head and meta["base_commit"] == base
    edited = tmp_path / "edited"
    shutil_copy_repo(fixture, edited)
    (edited / "a.py").write_text("print(2)\n", encoding="utf-8")
    (edited / "a.py").chmod(0o755)
    from alo_cloud.collect import workspace_patch
    patch_path.write_text(workspace_patch(edited, head), encoding="utf-8")
    again = run_schedule(units, tmp_path / "run", live=False, data_dir=tmp_path / "data",
                         sender=dry_run_sender, image_seeder=docker)
    assert again[0]["unit_id"] == unit_id
    assert again[1]["status"] == "completed"
    assert again[1]["image_head"] == head
    resumed = (tmp_path / "run" / "checkpoints" / again[1]["unit_id"] / "model.patch").read_text(encoding="utf-8")
    assert "print(2)" in resumed
    assert "old mode" not in resumed


def test_blocked_fraction_stops_the_schedule(tmp_path, monkeypatch):
    from alo_cloud.run import blocked_fraction_exceeded, format_blocked_report, main
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    for name in ("DEEPSEEK_API_KEY", "MIMO_API_KEY", "ZAI_API_KEY"):
        monkeypatch.setenv(name, "present")
    one = [{"status": "blocked", "reason": "swe_image_base_commit_mismatch"}]
    one += [{"status": "completed"}] * 3
    assert blocked_fraction_exceeded(one) is False
    one.append({"status": "blocked", "reason": "b0_checkpoint_missing"})
    text, exceeded = format_blocked_report(one)
    assert exceeded is True
    assert "2 swe_image_base_commit_mismatch" not in text
    assert "1 swe_image_base_commit_mismatch" in text
    assert "1 b0_checkpoint_missing" in text
    assert "blocked_fraction_exceeded" in text
    assert blocked_fraction_exceeded([{"status": "blocked", "reason": "x"}]) is False
    data = tmp_path / "data" / "SWE-bench_Lite"
    data.mkdir(parents=True)
    (data / "scikit-learn__scikit-learn-13584.json").write_text(
        json.dumps({"problem_statement": "public"}), encoding="utf-8")
    units = select_units("m1", tasks=("scikit-learn__scikit-learn-13584",))
    with pytest.raises(PilotStopped, match="blocked_fraction_exceeded"):
        run_schedule(units, tmp_path / "run", live=True, data_dir=tmp_path / "data")
    rows = [json.loads(line) for line in (tmp_path / "run" / "records.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert {row["reason"] for row in rows} == {"swe_checkout_metadata_missing"}
    assert all(row["phase"] == "B0" for row in rows)
    assert main(["--blocked-report", "--run-dir", str(tmp_path / "run")]) == 1


def test_harness_cwd_is_inside_the_run_dir(tmp_path, monkeypatch):
    import alo_cloud.evaluate as evaluate
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(kwargs.get("cwd"))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(evaluate, "_harness_run", fake_run)
    monkeypatch.setattr(evaluate, "_git_head", lambda repo: evaluate.PROGRAMBENCH_COMMIT)
    monkeypatch.setattr(evaluate.shutil, "which", lambda name: "/usr/bin/uv")
    command = evaluate.swebench_command(
        tmp_path / "predictions.jsonl", run_id="u1", instance_ids=("django__django-11583",), split="test")
    cwd = evaluate.harness_cwd(tmp_path / "run")
    result = evaluate.execute_swebench(command, cwd=cwd)
    assert result["status"] == "finished"
    assert seen[-1] == cwd
    assert cwd == tmp_path / "run" / "eval"
    unit = tmp_path / "run" / "sealed" / "programbench" / "unit-1" / "task-1"
    unit.mkdir(parents=True)
    (unit / "submission.tar.gz").write_bytes(b"")
    report = evaluate._score_programbench(tmp_path / "run", tmp_path / "programbench-src")
    assert report["units"][0]["status"] == "finished"
    assert seen[-1] == tmp_path / "run" / "eval"


def test_repository_text_does_not_call_the_model_by_a_snapshot_suffix():
    root = Path(__file__).resolve().parents[1]
    needles = ("V4.1", "v4.1")
    for folder in ("src", "docs", "scripts"):
        for path in (root / folder).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md", ".sh", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            for needle in needles:
                assert needle not in text, path

