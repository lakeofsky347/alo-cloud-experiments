"""Round-2 harness checks. No provider, docker daemon, or round-1 score."""
import json
import random
from pathlib import Path

import pytest

from alo_cloud.freeze import PlannedUnit
from alo_cloud.native_bridge import classify_group_turn
from alo_cloud.policy import ConstrainedThompsonPolicy
from alo_cloud.public_check import public_quality_signal, quality_from_signal
from alo_cloud.public_tools import coerce_actions, transport_summary
from alo_cloud.r2 import (
    BATCH, R1_M1, R2_COMMAND_SECONDS, cost_estimate, filter_r2_shard, iter_r2_m1_units,
    iter_r2_m2_units, m1_tasks, probe_tasks, r2_freeze_sha256, r2_manifest, smoke_unit,
    strong_task_ids,
)
from alo_cloud.run import _build_adapter, dry_run_sender, run_schedule
from alo_cloud.sandbox import MAX_READ_CHARS, Workspace
from alo_cloud.services import _map_python_argv
from alo_cloud.transport_fixtures import replay_fixtures
from alo_cloud.tool_loop import persistent_tool_loop
from tests.test_adapters import ScriptedServices, alo
from tests.test_tool_loop import LoopServices


def test_transport_fixtures_parse_at_least_ninety_percent():
    report = replay_fixtures()
    assert report["rate"] >= 0.90
    assert report["parsed"] == report["n"]
    names = {row["name"] for row in report["rows"]}
    for required in ("tool_call", "parameter_arguments", "function_tool", "tool_tag",
                     "dsml", "prose_json", "json_fence", "infer_argv", "infer_code",
                     "infer_write", "infer_read", "infer_search"):
        assert required in names


def test_first_of_two_actions_is_executed_and_the_rest_are_counted():
    text = (
        '<tool_call><function=list_files><parameter=arguments>{}</parameter></function>'
        '<function=search><parameter=text>other</parameter></function></tool_call>'
    )
    services = LoopServices([text, '{"final":"done"}'])
    services.transport_counts = {}
    services.model = "deepseek-flash"
    messages = []
    result = persistent_tool_loop(services, "task", messages, max_turns=2)
    assert result["termination"] == "final"
    assert services.calls.count("list_files") == 1
    assert not any(call == "search" or (isinstance(call, tuple) and call[0] == "search") for call in services.calls)
    blob = "\n".join(message["content"] for message in messages)
    assert "ignored 1 extra actions" in blob
    summary = transport_summary(services.transport_counts)
    assert summary["deepseek-flash"]["coerced"] >= 1


def test_unknown_argument_is_ignored():
    found = coerce_actions('{"tool":"search","arguments":{"text":"n","extra":true}}')
    assert found["actions"][0]["arguments"] == {"text": "n"}
    assert found["warnings"]


def test_timeout_clamp_warns(tmp_path, monkeypatch):
    from tests.test_services import build
    services, _seen, _ledger = build(tmp_path, monkeypatch, [])
    clamped, warnings = services.clamp_timeout(10_000)
    assert clamped == services.limits["command_seconds"] == 60
    assert warnings and "clamped" in warnings[0]
    services.limits["command_seconds"] = R2_COMMAND_SECONDS
    clamped, warnings = services.clamp_timeout(10_000)
    assert clamped == 120
    assert _map_python_argv(["python", "-m", "pytest"]) == ["python3", "-m", "pytest"]


def test_workspace_search_reads_past_the_page_and_edit_is_unique(tmp_path):
    root = tmp_path / "repo"
    workspace = Workspace(root)
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("hidden\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_mod.py").write_text("needle\n", encoding="utf-8")
    body = ("x" * 80 + "\n") * 800
    assert len(body) > MAX_READ_CHARS
    (root / "pkg.py").write_text(body + "needle\n", encoding="utf-8")
    (root / "blob.bin").write_bytes(b"\0needle")
    hits = workspace.search("needle")
    assert hits[0]["path"] == "pkg.py"
    assert hits[0]["text"]
    assert all(hit["path"] != ".git/config" for hit in hits)
    assert all(not hit["path"].endswith(".bin") for hit in hits)
    page = workspace.read_file("pkg.py")
    assert "[truncated]" in page
    assert "of " in page
    listed = workspace.list_files()
    assert ".git/config" not in listed
    assert workspace.list_files("tests") == ["tests/test_mod.py"]
    written = workspace.write_file("pkg.py", "alpha\n", append=False)
    assert written["bytes"]
    workspace.write_file("pkg.py", "beta\n", append=True)
    assert workspace.read_file("pkg.py") == "alpha\nbeta\n"
    edited = workspace.edit_file("pkg.py", "alpha", "gamma")
    assert edited["replacements"] == 1
    with pytest.raises(ValueError, match="not_found"):
        workspace.edit_file("pkg.py", "missing-text", "nope")
    workspace.write_file("pkg.py", "same\nsame\n")
    with pytest.raises(ValueError, match="not_unique"):
        workspace.edit_file("pkg.py", "same", "other")


def test_public_signal_moves_the_quality_posterior_off_beta_11(tmp_path):
    service = ScriptedServices(tmp_path, criterion="pass")
    service.public_signal = lambda: "pass"
    service.public_quality = lambda: None
    adapter = alo(max_attempts=2)
    adapter.run_stage({"prompt": "public task"}, service)
    observations = adapter.state["observations"]
    assert observations
    policy = ConstrainedThompsonPolicy(
        observations, lambda model: {"requests": 20, "tokens": 1000, "seconds": 100, "cny": 10})
    model = next(iter(observations))
    _selected, trace = policy.select(None, {}, [model], "type=generic|complexity=na|uncertainty=na", (), "u", random.Random(0))
    score = trace["scores"][model]
    assert (score["quality_alpha"], score["quality_beta"]) != (1, 1)
    assert quality_from_signal("pass") == 1.0
    assert quality_from_signal("unknown") is None
    assert public_quality_signal(tmp_path, changed_paths=["pkg.py"], patch_text="diff --git a/pkg.py") in {"pass", "fail", "unknown"}


def test_g2_reserves_verifier_turns_and_attempts():
    unit = next(unit for unit in iter_r2_m1_units() if unit.arm == "g2")
    assert unit.request_cap == 24
    adapter = _build_adapter(unit)
    assert adapter.config["max_turns"] == 24 - 1 - 3
    assert adapter.config["max_attempts"] == 2


def test_r2_schedule_is_separate_from_r1():
    m1 = m1_tasks()
    assert len(m1) == 48
    assert [task_id for task_id, _stratum in R1_M1] == [task_id for task_id, _stratum in m1[:6]]
    strong = set(strong_task_ids())
    assert len(strong) == 24
    assert {task_id for task_id, _stratum in R1_M1} <= strong
    probes = probe_tasks()
    assert len(probes) == 6
    m1_ids = {task_id for task_id, _stratum in m1}
    assert not (m1_ids & {task_id for task_id, _stratum in probes})
    units = list(iter_r2_m1_units()) + list(iter_r2_m2_units())
    assert {unit.phase for unit in units if unit.study == "m1"} == {"B0"}
    assert sum(1 for unit in units if unit.phase == "learning") == 16
    assert sum(1 for unit in units if unit.phase == "probe") == 24
    assert sum(1 for unit in units if unit.phase == "reference") == 30
    assert [unit.arm for unit in units if unit.phase == "reference"][-1] == "g1-strong"
    parts = [filter_r2_shard(units, index, 4) for index in range(4)]
    assert sorted(unit.unit_id for part in parts for unit in part) == sorted(unit.unit_id for unit in units)
    text = Path("docs/protocol_freeze_r2.md").read_text(encoding="utf-8")
    assert f"freeze_sha256: {r2_freeze_sha256()}" in text
    assert "not comparable" in text
    manifest = r2_manifest()
    assert manifest["batch"] == BATCH
    assert manifest["comparable_with_r1"] is False
    assert manifest["freeze_sha256"] == r2_freeze_sha256()
    estimate = cost_estimate()
    assert estimate["cny"]["mimo-v2.6-pro"] > 48
    assert estimate["cny"]["mimo-v2.6-pro"] < 50
    assert smoke_unit().request_cap == 40
    assert smoke_unit().task_id == "kisielk__errcheck.dacab89"


def test_missing_parent_blocks_the_chain_without_a_cold_start(tmp_path):
    calls = {"n": 0}

    def sender(url, body, headers):
        calls["n"] += 1
        return dry_run_sender(url, body, headers)

    parent = PlannedUnit(
        study="m2", arm="g2", track="SWE-bench_Lite", task_id="sympy__sympy-18621",
        stratum="short", phase="learning", position=0, seed=1, request_cap=4,
        order_id="order-1", m2_condition="persist", parent_unit_id="missing.parent",
    )
    child = PlannedUnit(
        study="m2", arm="g2", track="SWE-bench_Lite", task_id="django__django-15851",
        stratum="short", phase="learning", position=1, seed=2, request_cap=4,
        order_id="order-1", m2_condition="persist", parent_unit_id=parent.unit_id,
    )
    rows = run_schedule([parent, child], tmp_path, live=False, sender=sender, batch="r2")
    assert [row["status"] for row in rows] == ["blocked", "blocked"]
    assert [row["reason"] for row in rows] == ["parent_missing", "parent_missing"]
    assert [row["termination_reason"] for row in rows] == ["blocked", "blocked"]
    assert calls["n"] == 0
    assert (tmp_path / "r2_manifest.json").is_file()
    assert json.loads((tmp_path / "batch.json").read_text())["batch"] == "r2"


def test_m2_imports_statistics_and_does_not_apply_a_parent_patch(tmp_path):
    from alo.org.organization import Organization
    from alo_cloud.contracts import MODEL_POOL
    from alo_cloud.state import export_learning_state

    source = tmp_path / "other-repo.sqlite3"
    org = Organization(str(source))
    cell = "type=generic|complexity=na|uncertainty=na"
    org.belief.update_objective(MODEL_POOL[0], cell, True)
    org.conn.close()
    exported = export_learning_state(
        source, {"observations": {MODEL_POOL[0]: [{"public_quality": 1.0, "requests": 1}]}},
        allowed_models=MODEL_POOL, evidence_id="other-task")
    parent = PlannedUnit(
        study="m2", arm="g2", track="SWE-bench_Lite", task_id="sympy__sympy-18621",
        stratum="short", phase="learning", position=0, seed=3, request_cap=4,
        order_id="order-1", m2_condition="persist", parent_unit_id="",
    )
    # The parent checkpoint is a different task's statistics plus a patch the child must not take.
    parent_dir = tmp_path / "checkpoints" / "m2.SWE-bench_Lite.other-task.g2.learning.order-1.persist"
    parent_dir.mkdir(parents=True)
    (parent_dir / "learning_state.json").write_text(json.dumps(exported), encoding="utf-8")
    (parent_dir / "model.patch").write_text("SENTINEL_PATCH_NOT_FOR_M2\n", encoding="utf-8")
    child = PlannedUnit(
        study="m2", arm="g2", track="SWE-bench_Lite", task_id="django__django-15851",
        stratum="short", phase="learning", position=1, seed=4, request_cap=4,
        order_id="order-1", m2_condition="persist",
        parent_unit_id="m2.SWE-bench_Lite.other-task.g2.learning.order-1.persist",
    )
    rows = run_schedule([child], tmp_path, live=False, sender=dry_run_sender, batch="r2")
    assert rows[0]["status"] == "completed"
    assert rows[0]["termination_reason"] == "final"
    database = tmp_path / "checkpoints" / child.unit_id / "alo.sqlite3"
    assert database.is_file()
    restored = Organization(str(database))
    assert restored.belief.query(MODEL_POOL[0], cell).alpha == 2
    restored.conn.close()
    workspace_files = list((tmp_path / "checkpoints").rglob("model.patch"))
    assert all("SENTINEL_PATCH_NOT_FOR_M2" not in path.read_text(encoding="utf-8")
               or path.parent == parent_dir for path in workspace_files)


def test_invalid_parent_halts_rather_than_cold_starting(tmp_path):
    parent_dir = tmp_path / "checkpoints" / "m2.SWE-bench_Lite.broken.g2.learning.order-1.persist"
    parent_dir.mkdir(parents=True)
    (parent_dir / "learning_state.json").write_text("{not json", encoding="utf-8")
    unit = PlannedUnit(
        study="m2", arm="g2", track="SWE-bench_Lite", task_id="django__django-15851",
        stratum="short", phase="learning", position=1, seed=5, request_cap=4,
        order_id="order-1", m2_condition="persist",
        parent_unit_id="m2.SWE-bench_Lite.broken.g2.learning.order-1.persist",
    )
    rows = run_schedule([unit], tmp_path, live=False, sender=dry_run_sender, batch="r2")
    assert rows[0]["status"] == "blocked"
    assert rows[0]["reason"] == "parent_missing"
    assert not (tmp_path / "checkpoints" / unit.unit_id / "alo.sqlite3").exists()


def test_soft_halt_is_between_units_and_not_a_hard_stop(tmp_path):
    calls = {"n": 0}

    def sender(url, body, headers):
        calls["n"] += 1
        return dry_run_sender(url, body, headers)

    units = [unit for unit in iter_r2_m1_units() if unit.arm == "g1-main"][:2]
    rows = run_schedule(
        units, tmp_path, live=False, sender=sender, batch="r2",
        model_soft_halts={"mimo-v2.6-flash": 1})
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert calls["n"] == 1
    halt = json.loads((tmp_path / "schedule_halt.json").read_text(encoding="utf-8"))
    assert halt["hard"] is False
    assert halt["reason"] == "model_soft_halt:mimo-v2.6-flash"


def test_pb_smoke_dry_run_does_not_call_the_network(tmp_path, monkeypatch):
    from alo_cloud.run import main
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    assert main(["--pb-smoke", "--dry-run", "--run-dir", str(tmp_path)]) == 0
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["task_id"] == "kisielk__errcheck.dacab89"
    assert rows[0]["request_cap"] == 40
    assert rows[0]["termination_reason"] == "final"
    assert rows[0]["track"] == "ProgramBench"


def test_group_turn_split_and_probe_transcript_field():
    assert classify_group_turn([{"role": "system", "content": "You are in a role play game"}]) == "manager_selection"
    assert classify_group_turn([{"role": "user", "content": "fix the bug"}]) == "role"


def test_probe_transcript_is_persisted(tmp_path):
    service = ScriptedServices(tmp_path, criterion="pass")
    trained = alo()
    trained.run_stage({"prompt": "learn"}, service)
    probe = alo(probe=True)
    probe.run_stage({"prompt": "probe"}, ScriptedServices(tmp_path, criterion="pass"))
    assert "probe_transcripts" in probe.state
    assert probe.state["worker_transcripts"] == {} or "kept" not in probe.state.get("probe_transcripts", {})
    blob = json.dumps(probe.state["probe_transcripts"])
    assert "planner_messages" in blob or "worker_transcripts" in blob
