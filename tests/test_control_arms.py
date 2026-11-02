"""Control arms, routing modes, and the fixed two-node topology. Offline only."""
import json
import subprocess

from alo.org.organization import Organization
from alo.runtime.decomposer import parse_dag
from alo_cloud.adapters import FIXED_ROUTE_MODEL, FIXED_S1_S2_DAG, HeterogeneousALO
from alo_cloud.contracts import G1_DEEPSEEK_FLASH_MODEL, G1_MAIN_MODEL, G1_STRONG_MODEL, MODEL_POOL
from alo_cloud.errors import PilotStopped
from alo_cloud.r2 import (
    CONTROL_ARMS, followup_g2_config, g1_model, iter_control_units, iter_r2_m1_units,
    iter_r2_m2_units, r2_freeze_sha256, r2_shard_of,
)
from alo_cloud.repro import REPRO_CRITERION_ID
from alo_cloud.run import _build_adapter
from alo_cloud.sandbox import CRITERION_ID
from tests.test_adapters import ScriptedServices, alo

import pytest


SCRIPT = "import sys\nfrom pathlib import Path\nsys.exit(0 if 'FIXED' in Path('bug.py').read_text() else 1)\n"


def _init_repo(root, fixed):
    subprocess.check_call(["git", "init", "-q"], cwd=root)
    subprocess.check_call(["git", "config", "user.email", "t@example.com"], cwd=root)
    subprocess.check_call(["git", "config", "user.name", "t"], cwd=root)
    (root / "bug.py").write_text("BUG\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "bug.py"], cwd=root)
    subprocess.check_call(["git", "commit", "-q", "-m", "base"], cwd=root)
    if fixed:
        (root / "bug.py").write_text("FIXED\n", encoding="utf-8")
    (root / "alo_repro.py").write_text(SCRIPT, encoding="utf-8")


class ReproServices(ScriptedServices):
    def __init__(self, directory, workspace):
        super().__init__(directory)
        self.workspace = workspace
        self.edit_log = []
        self.saw_decompose = False

    def complete_for_model(self, model, messages, *, max_tokens=4096):
        content = messages[-1]["content"] if messages else ""
        if "ALO_DECOMPOSE" in content:
            self.saw_decompose = True
        return super().complete_for_model(model, messages, max_tokens=max_tokens)

    def command(self, argv, timeout_s=30):
        del timeout_s
        completed = subprocess.run(argv, cwd=self.workspace.root, capture_output=True, check=False)
        return {"returncode": completed.returncode,
                "stdout": completed.stdout.decode(errors="replace"),
                "stderr": completed.stderr.decode(errors="replace")}

    def local_criterion(self, **kwargs):
        from alo_cloud.services import StudyServices
        return StudyServices.local_criterion(self, **kwargs)


def test_frozen_g2_stays_on_thompson_syntax_and_the_planner():
    unit = next(unit for unit in iter_r2_m1_units() if unit.arm == "g2")
    adapter = _build_adapter(unit)
    assert adapter.config["routing_mode"] == "thompson"
    assert adapter.config["criterion_id"] == CRITERION_ID
    assert adapter.config["fixed_topology"] is False
    assert adapter.config["lightweight_verifier"] is False
    assert adapter.config["node_request_caps"] is None
    assert {unit.arm for unit in iter_r2_m1_units()} == {"g1-main", "g1-strong", "g2", "g3"}
    text = __import__("pathlib").Path("docs/protocol_freeze_r2.md").read_text(encoding="utf-8")
    assert f"freeze_sha256: {r2_freeze_sha256()}" in text


def test_g1_model_name_and_control_schedule_are_outside_the_freeze():
    assert g1_model("g1-main") == G1_MAIN_MODEL
    assert g1_model("g1-strong") == G1_STRONG_MODEL
    assert g1_model("g1-deepseek-flash") == "deepseek-flash"
    assert G1_DEEPSEEK_FLASH_MODEL == "deepseek-flash"
    assert "deepseek-flash-" not in G1_DEEPSEEK_FLASH_MODEL
    controls = list(iter_control_units())
    assert set(unit.arm for unit in controls) == set(CONTROL_ARMS)
    assert sum(unit.study == "m1" and unit.arm == "g1-deepseek-flash" for unit in controls) == 48
    assert sum(unit.study == "m1" and unit.arm == "g2-fixed" for unit in controls) == 48
    assert sum(unit.study == "m2" and unit.arm == "g1-deepseek-flash" for unit in controls) == 10
    assert sum(unit.study == "m2" and unit.arm == "g2-random" for unit in controls) == 40
    frozen_ids = {unit.unit_id for unit in list(iter_r2_m1_units()) + list(iter_r2_m2_units())}
    assert not frozen_ids & {unit.unit_id for unit in controls}
    sample = next(unit for unit in controls if unit.arm == "g2-fixed")
    config = followup_g2_config(sample)
    assert config["routing_mode"] == "fixed"
    assert config["planner_model"] == "deepseek-flash"
    assert config["criterion_id"] == REPRO_CRITERION_ID
    assert config["fixed_topology"] is True
    assert config["lightweight_verifier"] is True
    assert config["node_request_caps"] == {"s1": sample.request_cap, "s2": 0}
    assert config["max_attempts"] == 1
    built = _build_adapter(sample)
    assert built.config["routing_mode"] == "fixed"
    deepseek = next(unit for unit in controls if unit.arm == "g1-deepseek-flash")
    assert _build_adapter(deepseek).fixed_model == "deepseek-flash"
    homes = {r2_shard_of(unit, 4) for unit in controls if unit.task_id == sample.task_id and unit.study == "m1"}
    assert len(homes) == 1


def test_fixed_dag_parses_and_skips_the_planner(tmp_path):
    nodes = parse_dag(FIXED_S1_S2_DAG)
    assert [node.id for node in nodes] == ["s1", "s2"]
    assert nodes[1].deps == ("s1",)
    service = ScriptedServices(tmp_path)
    service.saw_decompose = False
    original = service.complete_for_model

    def complete_for_model(model, messages, *, max_tokens=4096):
        if messages and "ALO_DECOMPOSE" in messages[-1]["content"]:
            service.saw_decompose = True
        return original(model, messages, max_tokens=max_tokens)

    service.complete_for_model = complete_for_model
    adapter = alo(planner_model="deepseek-flash", fixed_topology=True, lightweight_verifier=True,
                  routing_mode="fixed", max_attempts=1)
    assert adapter.run_stage({"prompt": "public task"}, service)["status"] == "completed"
    assert service.saw_decompose is False
    assert service.calls == [FIXED_ROUTE_MODEL]
    assert all(row.get("selected") == "deepseek-flash"
               for row in adapter.state["decision_records"] if row.get("point") == "assignment")


def test_random_draws_from_the_feasible_set_and_ignores_the_preference(tmp_path):
    seen = set()
    for seed in range(12):
        service = ScriptedServices(tmp_path / str(seed))
        service.state_dir.mkdir(parents=True, exist_ok=True)
        adapter = alo(planner_model="deepseek-flash", fixed_topology=True, lightweight_verifier=True,
                      routing_mode="random", seed=seed, max_attempts=1,
                      user_preference={"first_try": "deepseek-flash", "strong": "mimo-v2.6-pro"})
        adapter.run_stage({"prompt": "public task"}, service)
        selected = [row["selected"] for row in adapter.state["decision_records"] if row.get("point") == "assignment"]
        assert selected
        assert all(model in MODEL_POOL for model in selected)
        assert all(row["preference_applied"] is False and row["routing_mode"] == "random"
                   for row in adapter.state["decision_records"] if row.get("point") == "assignment")
        seen.update(selected)
    assert len(seen) > 1


def test_fixed_model_does_not_fall_through_when_it_is_infeasible(tmp_path):
    service = ScriptedServices(tmp_path)

    def budget_remaining(model=None):
        remaining = {"requests": 100, "tokens": 100000, "seconds": 1000, "cny": 50}
        if model == "deepseek-flash":
            remaining["requests"] = 0
        return remaining

    service.budget_remaining = budget_remaining
    adapter = alo(planner_model="deepseek-flash", fixed_topology=True, routing_mode="fixed", max_attempts=1)
    with pytest.raises(PilotStopped, match="fixed_model_infeasible"):
        adapter.run_stage({"prompt": "public task"}, service)


def test_repro_failure_updates_beta_and_a_pass_does_not(tmp_path):
    from alo_cloud.sandbox import Workspace

    failed = tmp_path / "fail"
    workspace = Workspace(failed / "ws")
    _init_repo(workspace.root, fixed=False)
    service = ReproServices(failed / "state", workspace)
    service.state_dir.mkdir(parents=True, exist_ok=True)
    adapter = HeterogeneousALO(
        planner_model="deepseek-flash", fixed_topology=True, lightweight_verifier=True,
        routing_mode="fixed", criterion_id=REPRO_CRITERION_ID, max_attempts=1, seed=3)
    assert adapter.run_stage({"prompt": "bug"}, service)["status"] == "failed"
    org = Organization(str(failed / "state" / "alo.sqlite3"))
    rows = org.belief.dump()
    assert rows and any(row["beta"] > 1 for row in rows)
    org.conn.close()
    assert service.calls == ["deepseek-flash"]

    passed = tmp_path / "pass"
    workspace = Workspace(passed / "ws")
    _init_repo(workspace.root, fixed=True)
    service = ReproServices(passed / "state", workspace)
    service.state_dir.mkdir(parents=True, exist_ok=True)
    adapter = HeterogeneousALO(
        planner_model="deepseek-flash", fixed_topology=True, lightweight_verifier=True,
        routing_mode="fixed", criterion_id=REPRO_CRITERION_ID, max_attempts=1, seed=4)
    assert adapter.run_stage({"prompt": "bug"}, service)["status"] == "completed"
    org = Organization(str(passed / "state" / "alo.sqlite3"))
    rows = org.belief.dump()
    assert rows and all(row["beta"] == 1 for row in rows)
    assert any(row["alpha"] > 1 for row in rows)
    org.conn.close()
    assert (workspace.root / "bug.py").read_text(encoding="utf-8") == "FIXED\n"
    assert service.saw_decompose is False
    qualities = [row.get("quality_after") for rows in adapter.state["observations"].values() for row in rows]
    assert 0.0 not in qualities
    assert 1.0 in qualities


def test_live_belief_credits_each_node_to_the_model_that_ran_it(tmp_path):
    from alo.models.base import Completion, ModelInfo
    from alo.runtime.orchestrator import run_task
    from alo_cloud.adapters import FIXED_S1_S2_DAG

    class Worker:
        def __init__(self, name):
            self.info = ModelInfo(name, "official", name)

        def complete(self, prompt, **kwargs):
            del prompt, kwargs
            return Completion("submitted", 1, 1, 0)

    class Policy:
        def __init__(self):
            self.calls = []

        def select(self, org, pool, worker_ids, cell, topics, user_id, rng=None):
            del org, pool, cell, topics, user_id, rng
            chosen = "deepseek-flash" if not self.calls else "mimo-v2.6-flash"
            self.calls.append(chosen)
            return chosen, {"selected": chosen}

    org = Organization(str(tmp_path / "alo.sqlite3"))
    pool = {name: Worker(name) for name in ("deepseek-flash", "mimo-v2.6-flash", "planner::deepseek-flash")}
    run_task(
        org, pool, "fix the bug", planner_id="planner::deepseek-flash", fixed_dag=FIXED_S1_S2_DAG,
        max_attempts=1, criteria_fn=lambda role, output: True, assignment_policy=Policy(),
    )
    rows = {row["model"]: row for row in org.belief.dump()}
    org.conn.close()
    assert rows["deepseek-flash"]["alpha"] > 1 and rows["deepseek-flash"]["beta"] == 1
    assert rows["mimo-v2.6-flash"]["alpha"] > 1 and rows["mimo-v2.6-flash"]["beta"] == 1


def test_comparison_report_always_has_the_required_sections():
    from alo_cloud.comparison import comparison_markdown, comparison_report

    empty = comparison_report([])
    assert set(empty["all"]) >= {"oracle", "best_single_model", "per_resolved", "paired"}
    rows = [
        {"study": "m1", "arm": "g1-main", "task_id": "a", "resolved": True, "cost_cny": 1, "input_tokens": 10, "output_tokens": 5},
        {"study": "m1", "arm": "g1-main", "task_id": "b", "resolved": False, "cost_cny": 1, "input_tokens": 10, "output_tokens": 5},
        {"study": "m1", "arm": "g1-deepseek-flash", "task_id": "a", "resolved": True, "cost_cny": 2, "tokens": 30},
        {"study": "m1", "arm": "g1-deepseek-flash", "task_id": "b", "resolved": True, "cost_cny": 2, "tokens": 30},
        {"study": "m1", "arm": "g2", "task_id": "a", "resolved": False, "cost_cny": 4, "input_tokens": 1, "output_tokens": 1},
        {"study": "m1", "arm": "g2", "task_id": "b", "resolved": True, "cost_cny": 4, "input_tokens": 1, "output_tokens": 1},
    ]
    report = comparison_report(rows)["m1"]
    assert report["oracle"]["resolved"] == 2
    assert report["oracle"]["tasks"] == 2
    assert report["best_single_model"]["condition"] == "g1-deepseek-flash"
    assert report["best_single_model"]["resolved"] == 2
    assert report["per_resolved"]["g1-main"]["cny_per_resolved"] == 2
    assert report["per_resolved"]["g2"]["tokens_per_resolved"] == 4
    pair = report["paired"]["g1-main__vs__g2"]
    assert pair["only_left"] == 1
    assert pair["only_right"] == 1
    assert pair["both_resolved"] == 0
    text = comparison_markdown(comparison_report(rows))
    assert "Oracle:" in text
    assert "Best single model:" in text
    assert "g1-deepseek-flash" in text
    assert json.dumps(report)
