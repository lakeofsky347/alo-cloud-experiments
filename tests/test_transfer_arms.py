"""Opt-in learning arms, transfer package, and frozen import. Offline only."""
import json
import random
from dataclasses import replace

import pytest

from alo.org.organization import Organization
from alo.runtime.decomposer import fallback_subtasks
from alo_cloud.adapters import HeterogeneousALO, role_tool_contract
from alo_cloud.contracts import MODEL_POOL
from alo_cloud.errors import PilotStopped, ProtocolError
from alo_cloud.freeze import PlannedUnit
from alo_cloud.policy import ConstrainedThompsonPolicy
from alo_cloud.public_tools import dispatch_action
from alo_cloud.r2 import (
    USER_PREFERENCE, followup_g2_config, g1_model, iter_control_units, iter_r2_units,
    learning_g2_config, r2_freeze_payload, r2_freeze_sha256,
)
from alo_cloud.run import _build_adapter, _apply_parent, load_frozen_transfer
from alo_cloud.state import (
    QUALITY_OBSERVATION_FIELDS, StateContractError, export_learning_state,
    export_transfer_package, reject_cross_benchmark_parent,
)
from tests.test_adapters import ScriptedServices, alo


FROZEN_R2_SHA256 = "77bddfbfd3472ae4c0d280c3dacf20810f11b5f9b5c247c17650629ebf9e241f"
FROZEN_G2_KEYS = {
    "planner_model", "max_turns", "max_attempts", "seed", "policy_priors",
    "safety_multiplier", "organization_updates", "probe", "cold_prior",
    "user_preference", "routing_mode", "criterion_id", "fixed_topology",
    "lightweight_verifier", "node_request_caps", "protocol_id", "request_accounting",
    "planner_requests", "verifier_requests", "topology", "policy_version",
    "tool_loop_version",
}


def _quality_row(quality, topics, cost=0.01, requests=1):
    return {
        "public_quality": quality, "quality_before": 0.0, "quality_after": quality,
        "delta_quality": quality, "credited_quality": quality, "topics": list(topics),
        "requests": requests, "tokens": 10, "seconds": 1, "cost_cny": cost,
        "local_contract_status": "pass",
    }


def _seed_coding(quality_for):
    rows = {}
    for model in MODEL_POOL:
        quality = quality_for(model)
        coding = [_quality_row(quality, ["coding"]) for _ in range(20)]
        other = [_quality_row(1.0, ["testing", "validation"], cost=40.0, requests=80)]
        rows[model] = coding + other
    return rows


def _first_assignment(db_path):
    org = Organization(str(db_path))
    try:
        payload = org.conn.execute(
            "SELECT payload FROM audit_log WHERE event_type='assignment_policy' ORDER BY id"
        ).fetchone()[0]
    finally:
        org.conn.close()
    return json.loads(payload)


def test_freeze_hash_and_frozen_g2_configuration_stay_put():
    assert r2_freeze_sha256() == FROZEN_R2_SHA256
    payload = json.dumps(r2_freeze_payload())
    assert "g1-glm" not in payload
    assert "g2-learn" not in payload
    assert "g2-resources" not in payload
    text = __import__("pathlib").Path("docs/protocol_freeze_r2.md").read_text(encoding="utf-8")
    assert f"freeze_sha256: {FROZEN_R2_SHA256}" in text
    frozen_units = list(iter_r2_units("m1")) + list(iter_r2_units("m2"))
    assert {unit.track for unit in frozen_units} == {"SWE-bench_Lite"}
    assert "g1-glm" not in {unit.arm for unit in frozen_units}
    assert "g2-learn" not in {unit.arm for unit in frozen_units}
    unit = next(unit for unit in iter_r2_units("m1") if unit.arm == "g2")
    adapter = _build_adapter(unit, user_preference=dict(USER_PREFERENCE))
    assert set(adapter.config) == FROZEN_G2_KEYS
    assert adapter.config["planner_model"] == "deepseek-flash"
    assert adapter.config["max_attempts"] == 2
    assert adapter.config["user_preference"] == dict(USER_PREFERENCE)
    assert adapter.config["routing_mode"] == "thompson"
    assert "filter_resources_by_topics" not in adapter.config
    assert "learn_quality" not in adapter.config
    assert "fixed_model" not in adapter.config
    assert "export_mode" not in adapter.config


def test_learning_arm_follows_thompson_and_does_not_escalate(tmp_path):
    base = next(unit for unit in iter_r2_units("m1") if unit.arm == "g2")
    learn = replace(base, arm="g2-learn")
    observations = _seed_coding(lambda model: 1.0 if model == "glm-5.3-flash" else 0.0)
    learn_dir = tmp_path / "learn"
    learn_dir.mkdir()
    service = ScriptedServices(learn_dir)
    service.public_signal = lambda: "fail"
    service.criterion = "fail"
    adapter = _build_adapter(learn)
    assert adapter.config.get("user_preference") is None
    assert adapter.config["filter_resources_by_topics"] is True
    assert adapter.config["routing_mode"] == "thompson"
    assert "learn_quality" not in adapter.config
    before = json.loads(json.dumps(observations))
    adapter.state["observations"] = observations
    adapter.run_stage({"prompt": "public task"}, service)
    first = _first_assignment(learn_dir / "alo.sqlite3")
    policy = ConstrainedThompsonPolicy(
        before, service.budget_remaining, filter_resources_by_topics=True)
    org = Organization(":memory:")
    chosen, _trace = policy.select(
        org, {}, list(MODEL_POOL), first["cell"], first["topics"], "default-user",
        random.Random(learn.seed))
    org.conn.close()
    assert first["selected"] == chosen
    assert chosen != "deepseek-flash"
    reassign = [row for row in adapter.state["decision_records"] if row.get("action") == "reassign"]
    assert reassign
    assert all(row["escalated_to_strong"] is False for row in reassign)
    assert all(row["selected"] == row["counterfactual"] for row in reassign)

    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()
    frozen_service = ScriptedServices(frozen_dir)
    frozen_service.public_signal = lambda: "fail"
    frozen_service.criterion = "fail"
    frozen = _build_adapter(base, user_preference=dict(USER_PREFERENCE))
    frozen.state["observations"] = json.loads(json.dumps(before))
    frozen.run_stage({"prompt": "public task"}, frozen_service)
    preference = [row for row in frozen.state["decision_records"] if row.get("point") == "assignment"]
    assert preference[0]["selected"] == "deepseek-flash"
    assert preference[0]["preference_applied"] is True
    assert preference[0]["counterfactual"] == chosen
    escalated = [row for row in frozen.state["decision_records"] if row.get("action") == "reassign"]
    assert escalated and escalated[0]["escalated_to_strong"] is True
    assert escalated[0]["selected"] == "mimo-v2.6-pro"


def test_resources_arm_keeps_resource_rows_and_strips_quality_on_export(tmp_path):
    base = next(unit for unit in iter_r2_units("m1") if unit.arm == "g2")
    unit = replace(base, arm="g2-resources")
    assert learning_g2_config(unit)["learn_quality"] is False
    assert learning_g2_config(unit)["export_mode"] == "resources"
    assert learning_g2_config(unit)["filter_resources_by_topics"] is True
    assert "user_preference" not in learning_g2_config(unit)
    observations = {
        model: [
            _quality_row(1.0 if model == "glm-5.3-flash" else 0.0, ["coding"], cost=4.0),
            _quality_row(1.0, ["testing", "validation"], cost=40.0, requests=80),
        ]
        for model in MODEL_POOL
    }
    resource_dir = tmp_path / "resources"
    resource_dir.mkdir()
    service = ScriptedServices(resource_dir)
    adapter = _build_adapter(unit)
    assert adapter.config["learn_quality"] is False
    assert adapter.config["export_mode"] == "resources"
    adapter.state["observations"] = observations
    adapter.run_stage({"prompt": "public task"}, service)
    first = _first_assignment(resource_dir / "alo.sqlite3")
    for score in first["scores"].values():
        assert score["quality_source"] == "native_local_bernoulli_prior"
        assert score["quality_alpha"] == 1 and score["quality_beta"] == 1
        assert score["resource_prediction"]["cny"] == 5.0
        assert score["feasible"] is True
    path = tmp_path / "org.sqlite3"
    org = Organization(str(path))
    org.conn.close()
    full = export_learning_state(path, {"observations": observations}, allowed_models=MODEL_POOL)
    assert full["observations"]["glm-5.3-flash"][0]["credited_quality"] == 1.0
    stripped = export_learning_state(
        path, {"observations": observations}, allowed_models=MODEL_POOL, strip_quality=True)
    blob = json.dumps(stripped["observations"])
    for field in QUALITY_OBSERVATION_FIELDS:
        assert field not in blob
    kept = stripped["observations"]["glm-5.3-flash"][0]
    assert kept["requests"] == 1
    assert kept["cost_cny"] == 4.0
    assert kept["topics"] == ["coding"]


def test_glm_control_uses_the_fair_lock_and_fixed_model_is_a_parameter(tmp_path):
    assert g1_model("g1-glm") == "glm-5.3-flash"
    glm = next(unit for unit in iter_control_units("m1") if unit.arm == "g1-glm" and unit.phase == "B0")
    built = _build_adapter(glm)
    assert built.fixed_model == "glm-5.3-flash"
    assert built.protocol_id == "fair-control-v1"
    assert built.criterion_id == "repro-v1"
    assert built.max_attempts == 1
    assert glm.arm not in {unit.arm for unit in iter_r2_units("m1")}
    sample = next(unit for unit in iter_control_units("m1") if unit.arm == "g2-fixed" and unit.phase == "B0")
    assert followup_g2_config(sample)["fixed_model"] == "deepseek-flash"
    pinned = followup_g2_config(sample, fixed_model="glm-5.3-flash")
    assert pinned["fixed_model"] == "glm-5.3-flash"
    service = ScriptedServices(tmp_path)
    adapter = HeterogeneousALO(
        planner_model="deepseek-flash", fixed_topology=True, lightweight_verifier=True,
        routing_mode="fixed", fixed_model=pinned["fixed_model"], max_attempts=1, seed=sample.seed)
    assert adapter.run_stage({"prompt": "public task"}, service)["status"] == "completed"
    models = [payload.get("model_id") for name, payload in service.events
              if name == "alo_execution_context"]
    assert models == ["glm-5.3-flash", "glm-5.3-flash"]
    assert service.calls == ["glm-5.3-flash"]
    assert "ALO_DECOMPOSE" not in json.dumps(service.events)


def test_transfer_package_keys_match_across_tracks_and_omit_the_cell():
    state = {"observations": {"deepseek-flash": [
        _quality_row(1.0, ["implementation", "coding"], cost=0.1, requests=2),
        _quality_row(0.0, ["testing", "validation"], cost=0.02, requests=1),
        _quality_row(1.0, ["generic"], cost=9.0, requests=9),
    ], "glm-5.3-flash": [
        {**_quality_row(1.0, ["coding", "implementation"]),
         "cell": "type=writing|complexity=na|uncertainty=na", "role_id": "s1"},
    ]}}
    swe = export_transfer_package(state, allowed_models=MODEL_POOL, task_family="swe-repair")
    program = export_transfer_package(
        state, allowed_models=MODEL_POOL, task_family="program-reconstruction")

    def keys(package):
        found = []
        for model, rows in package["profiles"].items():
            for row in rows:
                found.append((model, tuple(row["role_topics"]), row["task_type"]))
                assert "cell" not in row
                assert "role_id" not in row
        return found

    assert keys(swe) == keys(program)
    assert ("deepseek-flash", ("coding", "implementation"), "coding/implementation") in keys(swe)
    assert ("deepseek-flash", ("testing", "validation"), "testing/validation") in keys(swe)
    assert "generic" not in json.dumps(swe["profiles"])
    assert "type=" not in json.dumps(swe)
    assert swe["non_transferable"] == ["cell"]
    assert swe["division_key"] == "model_x_role_x_task_type"
    assert swe["task_family"] == "swe-repair"
    assert program["task_family"] == "program-reconstruction"
    stripped = export_transfer_package(
        state, allowed_models=MODEL_POOL, task_family="swe-repair", strip_quality=True)
    blob = json.dumps(stripped["profiles"])
    for field in QUALITY_OBSERVATION_FIELDS:
        assert field not in blob
    assert stripped["profiles"]["deepseek-flash"][0]["requests"] == 2
    with pytest.raises(StateContractError, match="unregistered_task_family"):
        export_transfer_package(state, allowed_models=MODEL_POOL, task_family="other")


def test_frozen_import_does_not_write_back_or_parent_swe(tmp_path):
    state = {"observations": {"deepseek-flash": [
        _quality_row(1.0, ["coding", "implementation"]),
    ]}}
    package = export_transfer_package(state, allowed_models=MODEL_POOL, task_family="swe-repair")
    source = tmp_path / "swe-transfer.json"
    source.write_text(json.dumps(package), encoding="utf-8")
    raw = source.read_bytes()
    digest = __import__("hashlib").sha256(raw).hexdigest()
    with load_frozen_transfer(source, target_track="ProgramBench") as library:
        assert library.source_digest == digest
        assert library.observations["deepseek-flash"][0]["topics"] == ["coding", "implementation"]
        assert library.connection.execute("SELECT count(*) FROM belief").fetchone()[0] == 0
        with pytest.raises(StateContractError, match="frozen_transfer_writeback_forbidden"):
            library.write_back({"observations": {"deepseek-flash": []}})
        assert source.read_bytes() == raw
        target = library.target_state()
        assert target["may_parent_swe"] is False
        assert target["target_track"] == "ProgramBench"
        with pytest.raises(StateContractError, match="target_learning_state_cannot_parent_swe"):
            reject_cross_benchmark_parent(
                "m2.SWE-bench_Lite.task-b.g2.learning.order-1.persist",
                "SWE-bench_Lite", target)
    assert source.read_bytes() == raw
    assert __import__("hashlib").sha256(source.read_bytes()).hexdigest() == digest

    held_out = export_transfer_package(
        state, allowed_models=MODEL_POOL, task_family="program-reconstruction")
    held = tmp_path / "held.json"
    held.write_text(json.dumps(held_out), encoding="utf-8")
    before = held.read_bytes()
    with pytest.raises(StateContractError, match="target_learning_state_cannot_parent_swe"):
        with load_frozen_transfer(held, target_track="SWE-bench_Lite"):
            pass
    assert held.read_bytes() == before


def test_apply_parent_rejects_a_program_bench_learning_state(tmp_path):
    path = tmp_path / "org.sqlite3"
    org = Organization(str(path))
    org.conn.close()
    exported = export_learning_state(
        path, {"observations": {"deepseek-flash": [_quality_row(1.0, ["coding"])]}},
        allowed_models=MODEL_POOL)
    swe_id = "m2.SWE-bench_Lite.task-a.g2.learning.order-1.persist"
    swe_dir = tmp_path / "checkpoints" / swe_id
    swe_dir.mkdir(parents=True)
    (swe_dir / "learning_state.json").write_text(json.dumps(exported), encoding="utf-8")
    child = PlannedUnit(
        study="m2", arm="g2", track="SWE-bench_Lite", task_id="task-b", stratum="short",
        phase="learning", position=1, seed=1, request_cap=4, order_id="order-1",
        m2_condition="persist", parent_unit_id=swe_id)
    service = ScriptedServices(tmp_path / "child")
    service.state_dir.mkdir()
    adapter = alo(planner_model="deepseek-flash")
    assert _apply_parent(child, adapter, service, tmp_path) is None
    assert adapter.state["observations"]["deepseek-flash"][0]["topics"] == ["coding"]

    program_id = "m2.ProgramBench.task-c.g2.learning.order-1.persist"
    program_dir = tmp_path / "checkpoints" / program_id
    program_dir.mkdir(parents=True)
    (program_dir / "learning_state.json").write_text(json.dumps(exported), encoding="utf-8")
    blocked = replace(child, parent_unit_id=program_id, task_id="task-d")
    blocked_adapter = alo(planner_model="deepseek-flash")
    with pytest.raises(PilotStopped, match="target_learning_state_cannot_parent_swe"):
        _apply_parent(blocked, blocked_adapter, service, tmp_path)
    assert blocked_adapter.state["observations"] == {}


def test_fallback_generalist_keeps_the_full_tool_set(tmp_path):
    node = fallback_subtasks("keep the tools")[0]
    assert node.role.tools == ()
    assert role_tool_contract(node.role) is None
    services = type("S", (), {})()
    services.role_tools = role_tool_contract(node.role)

    def write_file(**kwargs):
        return {"wrote": kwargs["path"]}

    services.write_file = write_file
    assert dispatch_action(services, {"tool": "write_file", "arguments": {
        "path": "a.py", "content": "x"}})["wrote"] == "a.py"
    services.role_tools = ()
    with pytest.raises(ProtocolError, match="tool_outside_role_contract"):
        dispatch_action(services, {"tool": "write_file", "arguments": {
            "path": "a.py", "content": "x"}})
    services.role_tools = ("command",)
    with pytest.raises(ProtocolError, match="tool_outside_role_contract"):
        dispatch_action(services, {"tool": "write_file", "arguments": {
            "path": "a.py", "content": "x"}})

    service = ScriptedServices(tmp_path)
    seen = []
    original = service.complete_for_model

    def complete_for_model(model, messages, *, max_tokens=4096):
        content = messages[-1]["content"] if messages else ""
        if "ALO_DECOMPOSE" not in content:
            seen.append(getattr(service, "role_tools", "missing"))
        reply = original(model, messages, max_tokens=max_tokens)
        if "ALO_DECOMPOSE" in content:
            reply = dict(reply)
            reply["text"] = "not yaml"
        return reply

    service.complete_for_model = complete_for_model
    assert alo().run_stage({"prompt": "public task"}, service)["status"] == "completed"
    assert seen == [None]
