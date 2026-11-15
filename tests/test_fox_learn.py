"""Offline fox-pool learning stream. No socket and no API key."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import pytest

from alo_cloud.budget import BudgetLedger
from alo_cloud.contracts import MODEL_POOL
from alo_cloud.fox_learn import (
    EXPERIMENT,
    LearnStopped,
    build_stream,
    container_name_for,
    dry_gateway,
    empty_profile_hash,
    export_chain_transfer,
    iter_learning_chains,
    launch_command,
    main,
    run_chain,
    run_isolated,
    select_learning_tasks,
)
from alo_cloud.fox_pool import FOX_MODELS, resolve_alo_pool, resolve_model
from alo_cloud.fox_single import FoxStopped, _transport_for
from alo_cloud.prices import charge_if_priced, charge_microcny
from alo_cloud.r2 import iter_r2_units, r2_freeze_sha256
from alo_cloud.sandbox import DockerSandbox
from alo_cloud.fox_learn import FoxAloServices
from alo_cloud.fox_pool import model_settings


FROZEN_R2 = "77bddfbfd3472ae4c0d280c3dacf20810f11b5f9b5c247c17650629ebf9e241f"
DEV_A = "sqlfluff__sqlfluff-1625"
DEV_B = "pydicom__pydicom-1694"


def _stream(tasks=(DEV_A, DEV_B), **kwargs):
    return build_stream(split="dev", task_ids=tasks, turn_cap=8, **kwargs)


def _chain(built, chain_id):
    return next(chain for chain in built["chains"] if chain.chain_id == chain_id)


def _run(chain, run_dir, pool, **kwargs):
    return run_chain(
        chain, run_dir, pool=pool, live=False, sender=dry_gateway,
        sleep=lambda _seconds: None, **kwargs)


def _records(run_dir):
    rows = []
    for line in (Path(run_dir) / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_domestic_pool_stays_on_the_cny_ledger():
    domestic = resolve_alo_pool("domestic")
    assert domestic["kind"] == "domestic"
    assert domestic["models"] == MODEL_POOL
    assert domestic["budget"] == "cny"
    assert domestic["prices"] == "list"
    with pytest.raises(ValueError, match="domestic_pool_is_fixed"):
        resolve_alo_pool("domestic", ["deepseek-flash"])
    charge = charge_microcny("deepseek-flash", {"input_tokens": 1000, "output_tokens": 1000})
    assert type(charge) is int and charge > 0
    assert charge_if_priced("fox/deepseek-flash", {"input_tokens": 10, "output_tokens": 10}) is None
    assert r2_freeze_sha256() == FROZEN_R2
    arms = {unit.arm for unit in iter_r2_units("m2")}
    assert "g2-learn" not in arms
    assert "g2-resources" not in arms


def test_fox_pool_subset_and_alias_strictness(tmp_path):
    full = resolve_alo_pool("fox")
    assert full["models"] == tuple(row.internal for row in FOX_MODELS)
    assert full["prices"] == "none"
    assert full["budget"] == "raw_optional"
    subset = resolve_alo_pool("fox", ["fox/glm-5.3-flash", "deepseek-flash", "fox/qwen3.8-flash"])
    assert subset["models"] == (
        "fox/deepseek-flash", "fox/qwen3.8-flash", "fox/glm-5.3-flash")
    assert resolve_model("deepseek-flash").upstream == "deepseek-v4.1-flash-1017"
    with pytest.raises(ValueError):
        resolve_alo_pool("fox", ["qwen3.8-omni-flash"])
    with pytest.raises(ValueError):
        resolve_model("mimo-v2.6-flash")

    def sender(url, body, headers, timeout_s):
        del url, headers, timeout_s
        returned = sender.returned
        payload = {
            "model": returned,
            "choices": [{"message": {"content": '{"final":"x"}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            "cost": 0.25,
            "credit": 4,
            "quota": {"left": 8},
        }
        return 200, json.dumps(payload).encode("utf-8")

    specs = subset["specs"]
    settings = {row.internal: model_settings(row) for row in specs}
    transport = _transport_for(
        live=False, sender=sender, timeout_s=5, sleep=lambda _n: None,
        clock=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        max_attempts=1)
    services = FoxAloServices(
        tmp_path / "state", specs, settings_by_model=settings, transport=transport,
        limits={"requests": 8, "tokens": 1000, "seconds": 30, "tools": 4},
        budget_cap=None)
    messages = [{"role": "user", "content": "ping"}]
    sender.returned = "qwen/qwen3.8-flash"
    reply = services.complete_for_model("fox/qwen3.8-flash", messages, max_tokens=16)
    assert reply["text"]
    assert services.calls[-1]["model_alias_normalized"]["returned"] == "qwen/qwen3.8-flash"
    assert "model_mismatch" not in services.calls[-1]
    assert services.calls[-1]["cost"] == 0.25
    assert services.calls[-1]["credit"] == 4
    remaining = services.budget_remaining("fox/qwen3.8-flash")
    assert remaining["cny"] > 1_000_000
    assert "budget.sqlite3" not in {path.name for path in tmp_path.rglob("*")}

    sender.returned = "qwen3.8-omni-flash"
    with pytest.raises(FoxStopped):
        services.complete_for_model("fox/qwen3.8-flash", messages, max_tokens=16)
    assert services.calls[-1]["model_mismatch"] is True
    assert services.calls[-1]["returned_model"] == "qwen3.8-omni-flash"
    assert services.calls[-1]["cost"] == 0.25

    sender.returned = "fw/kimi-k3p1-pro"
    kimi = resolve_alo_pool("fox", ["fox/kimi-k3.1-pro"])
    kimi_settings = {row.internal: model_settings(row) for row in kimi["specs"]}
    kimi_services = FoxAloServices(
        tmp_path / "kimi", kimi["specs"], settings_by_model=kimi_settings, transport=transport,
        limits={"requests": 4, "tokens": 1000, "seconds": 30, "tools": 4})
    kimi_services.complete_for_model("fox/kimi-k3.1-pro", messages, max_tokens=16)
    assert "model_mismatch" not in kimi_services.calls[-1]
    sender.returned = "kimi-k3p2-pro"
    with pytest.raises(FoxStopped):
        kimi_services.complete_for_model("fox/kimi-k3.1-pro", messages, max_tokens=16)
    assert kimi_services.calls[-1]["model_mismatch"] is True


def test_six_chains_cover_conditions_and_orders(tmp_path):
    listed = tmp_path / "tasks.txt"
    listed.write_text(f"# subset\n{DEV_A}\n{DEV_B}\n", encoding="utf-8")
    built = build_stream(split="dev", tasks_file=listed, turn_cap=24, seed="fox-alo-learn-v1")
    assert built["turn_cap"] == 24
    assert [chain.chain_id for chain in built["chains"]] == [
        "persist-forward", "persist-reverse",
        "reset-forward", "reset-reverse",
        "resources-only-forward", "resources-only-reverse",
    ]
    ids = {chain.run_id for chain in built["chains"]}
    assert len(ids) == 6
    assert all(chain.run_id.startswith(EXPERIMENT + ".") for chain in built["chains"])
    forward = _chain(built, "persist-forward")
    reverse = _chain(built, "persist-reverse")
    assert [unit.task_id for unit in forward.units] == [DEV_A, DEV_B]
    assert [unit.task_id for unit in reverse.units] == [DEV_B, DEV_A]
    assert forward.units[0].parent_unit_id == ""
    assert forward.units[1].parent_unit_id == forward.units[0].unit_id
    assert forward.arm == "g2-learn" and forward.m2_condition == "persist"
    reset = _chain(built, "reset-forward")
    assert reset.arm == "g2-learn" and reset.m2_condition == "reset"
    assert all(unit.parent_unit_id == "" for unit in reset.units)
    resources = _chain(built, "resources-only-forward")
    assert resources.arm == "g2-resources" and resources.m2_condition == "persist"
    assert resources.strip_quality is True
    assert resources.units[1].parent_unit_id == resources.units[0].unit_id
    names = []
    workdirs = []
    for chain in built["chains"]:
        root = tmp_path / chain.chain_id
        for unit in chain.units:
            names.append(container_name_for(chain.chain_id, unit.unit_id))
            workdirs.append(str(root / "units" / unit.unit_id / "state" / "workspace"))
    assert len(names) == len(set(names)) == 12
    assert len(workdirs) == len(set(workdirs)) == 12
    assert "/" not in "".join(names)
    learn_only = build_stream(split="dev", task_ids=(DEV_A,), turn_cap=24, arms=("g2-learn",))
    assert {chain.arm for chain in learn_only["chains"]} == {"g2-learn"}
    assert len(learn_only["chains"]) == 4
    seeded = select_learning_tasks(split="dev", n=3, seed="fox-alo-learn-v1")
    again = select_learning_tasks(split="dev", n=3, seed="fox-alo-learn-v1")
    assert seeded == again
    assert len(seeded) == 3


def test_container_name_is_opt_in(tmp_path):
    plain = DockerSandbox(tmp_path, image="example.invalid/alo:test")
    assert "--name" not in plain.docker_argv(["python3", "-c", "print(1)"])
    named = DockerSandbox(
        tmp_path, image="example.invalid/alo:test", container_name="alo-fox-persist-forward-abc123")
    argv = named.docker_argv(["python3", "-c", "print(1)"])
    again = named.docker_argv(["python3", "-c", "print(1)"])
    first = argv[argv.index("--name") + 1]
    second = again[again.index("--name") + 1]
    assert first.startswith("alo-fox-persist-forward-abc123-")
    assert second.startswith("alo-fox-persist-forward-abc123-")
    assert first != second
    with pytest.raises(Exception):
        DockerSandbox(tmp_path, image="example.invalid/alo:test", container_name="has/slash")


def test_persist_reset_resources_resume_and_export(tmp_path):
    built = _stream()
    pool = built["pool"]
    persist_dir = tmp_path / "persist-forward"
    summary = _run(_chain(built, "persist-forward"), persist_dir, pool)
    assert summary["wrote"] == 2
    rows = _records(persist_dir)
    assert [row["status"] for row in rows] == ["completed", "completed"]
    assert rows[0]["profile_hash_before"] == empty_profile_hash(pool["models"], strip_quality=False)
    assert rows[1]["profile_hash_before"] == rows[0]["profile_hash_after"]
    assert rows[0]["profile_hash_after"] != rows[0]["profile_hash_before"]
    assert rows[0]["assignments"]
    assert rows[0]["node_assignments"]
    assert rows[0]["roles_by_model"]
    assert set(rows[0]["roles_by_model"]) <= set(pool["models"])
    assert rows[0]["harness_resolved"] is None
    assert rows[0]["harness_status"] == "not_executed"
    assert rows[0]["cost"] == pytest.approx(0.006)
    assert rows[0]["credit"] == pytest.approx(4.5)
    assert rows[0]["usage"]["prompt_tokens"] == 36
    assert rows[0]["prices"] == "none"
    assert "FOXROUTER_API_KEY" not in json.dumps(rows)
    assert not (persist_dir / "budget.sqlite3").exists()
    again = _run(_chain(built, "persist-forward"), persist_dir, pool)
    assert again["wrote"] == 0
    assert len(_records(persist_dir)) == 2

    package = export_chain_transfer(persist_dir)
    assert package["schema"] == "alo-cross-track-transfer-v1"
    assert package["task_family"] == "swe-repair"
    assert sum(len(rows) for rows in package["profiles"].values()) >= 1
    assert (persist_dir / "transfer_package.json").is_file()

    reset_dir = tmp_path / "reset-forward"
    _run(_chain(built, "reset-forward"), reset_dir, pool)
    reset_rows = _records(reset_dir)
    empty = empty_profile_hash(pool["models"], strip_quality=False)
    assert reset_rows[0]["profile_hash_before"] == empty
    assert reset_rows[1]["profile_hash_before"] == empty
    assert reset_rows[1]["parent_unit_id"] == ""
    assert reset_rows[0]["profile_hash_after"] != empty

    resource_dir = tmp_path / "resources-only-forward"
    _run(_chain(built, "resources-only-forward"), resource_dir, pool)
    resource_rows = _records(resource_dir)
    assert resource_rows[1]["profile_hash_before"] == resource_rows[0]["profile_hash_after"]
    carried = json.loads(
        (resource_dir / "checkpoints" / resource_rows[0]["unit_id"] / "learning_state.json").read_text(
            encoding="utf-8"))
    assert carried["belief"] == []
    assert carried["tms_outcomes"] == []
    blob = json.dumps(carried["observations"])
    assert "credited_quality" not in blob
    assert "quality_after" not in blob
    with pytest.raises(Exception):
        export_chain_transfer(reset_dir)


def test_parallel_chains_stay_isolated(tmp_path):
    built = _stream(tasks=(DEV_A,))
    pool = built["pool"]
    left = _chain(built, "persist-forward")
    right = _chain(built, "reset-reverse")
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    payloads = [
        {"chain": left, "run_dir": left_dir, "pool": pool},
        {"chain": right, "run_dir": right_dir, "pool": pool},
    ]
    with ProcessPoolExecutor(max_workers=2) as pool_exec:
        results = list(pool_exec.map(run_isolated, payloads))
    assert {row["wrote"] for row in results} == {1}
    left_ids = {row["unit_id"] for row in _records(left_dir)}
    right_ids = {row["unit_id"] for row in _records(right_dir)}
    assert left_ids.isdisjoint(right_ids)
    assert json.loads((left_dir / "manifest.json").read_text(encoding="utf-8"))["run_id"] != (
        json.loads((right_dir / "manifest.json").read_text(encoding="utf-8"))["run_id"])
    left_names = set(json.loads((left_dir / "manifest.json").read_text(encoding="utf-8"))["container_names"])
    right_names = set(json.loads((right_dir / "manifest.json").read_text(encoding="utf-8"))["container_names"])
    assert left_names.isdisjoint(right_names)
    left_work = {row["workdir"] for row in _records(left_dir)}
    right_work = {row["workdir"] for row in _records(right_dir)}
    assert left_work.isdisjoint(right_work)
    assert not any(str(right_dir) in path.as_posix() for path in left_dir.rglob("*"))
    assert not any(str(left_dir) in path.as_posix() for path in right_dir.rglob("*"))


def test_plan_and_arm_redirect(tmp_path, capsys):
    listed = tmp_path / "tasks.txt"
    listed.write_text(DEV_A + "\n" + DEV_B + "\n", encoding="utf-8")
    root = tmp_path / "launch"
    assert main([
        "plan", "--pool", "fox", "--tasks-file", str(listed), "--turn-cap", "24",
        "--root", str(root), "--models", "fox/deepseek-flash,fox/qwen3.8-flash",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_calls"] == 0
    assert payload["turn_cap"] == 24
    assert payload["models"] == ["fox/deepseek-flash", "fox/qwen3.8-flash"]
    assert len(payload["chains"]) == 6
    command = payload["chains"][0]["command"]
    assert command[:4] == ["python", "-m", "alo_cloud.fox_learn", "run"]
    assert "--pool" in command and "fox" in command
    assert "--live" in command and "--execute" in command
    assert "--turn-cap" in command and "24" in command
    assert str(root) in " ".join(command)
    assert main(["plan", "--pool", "domestic", "--n", "1"]) == 2
    assert "domestic_pool_uses_cny_ledger" in capsys.readouterr().out
    assert main([
        "run", "--pool", "fox", "--chain", "persist-forward", "--dry-run",
        "--run-dir", str(tmp_path / "cli"), "--tasks", DEV_A, "--turn-cap", "8",
        "--arm", "g2-learn", "--condition", "persist", "--order", "forward",
    ]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli["wrote"] == 1
    from alo_cloud.run import main as run_main
    with pytest.raises(SystemExit):
        run_main([
            "--study", "m2", "--arm", "g2-learn", "--dry-run",
            "--run-dir", str(tmp_path / "frozen"),
        ])


def test_optional_raw_cap_does_not_open_a_cny_ledger(tmp_path):
    built = _stream(tasks=(DEV_A, DEV_B))
    run_dir = tmp_path / "capped"
    summary = _run(_chain(built, "reset-forward"), run_dir, built["pool"], budget_cap=0.0)
    assert summary["wrote"] == 1
    assert _records(run_dir)[0]["cost"] > 0
    assert not list(run_dir.rglob("budget.sqlite3"))
    # The domestic ledger still accepts only the mainland pool.
    ledger = tmp_path / "domestic.sqlite3"
    BudgetLedger.create(
        ledger,
        model_limits_microcny={model: 50_000_000 for model in MODEL_POOL},
        opening_commitments_microcny={model: 0 for model in MODEL_POOL},
        max_requests=4, prior_requests=0, max_tokens=100, prior_tokens=0,
        max_seconds_ms=1000, prior_seconds_ms=0, max_tools=4, prior_tools=0,
        total_limit_microcny=200_000_000)
    opened = BudgetLedger(ledger)
    assert set(opened.snapshot()["limits_microcny"]) == set(MODEL_POOL)


def test_resume_rejects_a_dry_directory_for_live(tmp_path):
    built = _stream(tasks=(DEV_A,))
    chain = _chain(built, "persist-forward")
    run_dir = tmp_path / "persist-forward"
    assert _run(chain, run_dir, built["pool"])["wrote"] == 1
    with pytest.raises(LearnStopped, match="fox_run_settings_mismatch"):
        run_chain(
            chain, run_dir, pool=built["pool"], live=True, sender=None,
            sleep=lambda _seconds: None)
    assert len(_records(run_dir)) == 1
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["live"] is False


def test_plan_tasks_keep_the_selection_order(tmp_path, capsys):
    built = build_stream(split="dev", task_ids=(DEV_A, DEV_B), turn_cap=8)
    selected = f"{DEV_A},{DEV_B}"
    reverse = _chain(built, "persist-reverse")
    command = launch_command(
        reverse, tmp_path / "persist-reverse", task_ids=(DEV_A, DEV_B), live=True)
    assert command[command.index("--tasks") + 1] == selected
    replay = build_stream(split="dev", task_ids=(DEV_A, DEV_B), turn_cap=8)
    assert [unit.task_id for unit in _chain(replay, "persist-reverse").units] == [DEV_B, DEV_A]
    assert [unit.task_id for unit in _chain(replay, "persist-forward").units] == [DEV_A, DEV_B]
    assert main([
        "plan", "--pool", "fox", "--tasks", selected, "--turn-cap", "8",
        "--root", str(tmp_path / "plan"),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    for chain in payload["chains"]:
        argv = chain["command"]
        assert argv[argv.index("--tasks") + 1] == selected
        assert [row["task_id"] for row in chain["units"]] == (
            [DEV_B, DEV_A] if chain["order"] == "reverse" else [DEV_A, DEV_B])


def test_schedule_orders_are_independent_of_hash_randomization():
    one = iter_learning_chains(
        select_learning_tasks(split="dev", n=2, seed="same"), turn_cap=24, stream_seed="same")
    two = iter_learning_chains(
        select_learning_tasks(split="dev", n=2, seed="same"), turn_cap=24, stream_seed="same")
    assert [unit.unit_id for chain in one for unit in chain.units] == [
        unit.unit_id for chain in two for unit in chain.units]
    assert [unit.seed for chain in one for unit in chain.units] == [
        unit.seed for chain in two for unit in chain.units]
