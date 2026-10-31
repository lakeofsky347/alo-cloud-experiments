"""Scripted service controls of preserved algorithms, never benchmark evidence."""
import json
import random
from types import SimpleNamespace

import pytest

from alo.org.organization import Organization
from alo_cloud.adapters import FixedSingleAgent, HeterogeneousALO, HeterogeneousGroupChat, adapter_for
from alo_cloud.contracts import G1_MAIN_MODEL, MODEL_POOL
from alo_cloud.errors import PilotStopped


PLANNER = MODEL_POOL[1]


def alo(**kwargs):
    kwargs.setdefault("planner_model", PLANNER)
    return HeterogeneousALO(**kwargs)


PLAN = """subtasks:
  - id: public_work
    goal: inspect and submit public work
    deps: []
    topics: [coding]
    role:
      capability_reqs: [coding]
      epistemic_scope:
        include: [task_description, goal, persistent_workspace]
        exclude: [billing, credentials]
      authority: [inspect, modify]
      tools: [write_file]
      output_schema: source
      success_criteria: explicit_contract
"""


class ScriptedServices:
    """A deterministic in-memory response source; no sandbox implementation."""
    available_models = MODEL_POOL

    def __init__(self, directory, *, criterion="unknown", groupchat=False):
        self.state_dir = directory
        self.criterion, self.groupchat = criterion, groupchat
        self.calls, self.events, self.tools = [], [], 0
        self.gateway = SimpleNamespace(currency="CNY")

    def complete_for_model(self, model, messages, *, max_tokens=4096):
        self.calls.append(model)
        if self.groupchat:
            text = "Verifier" if "role play game" in messages[0]["content"] else "STAGE_DONE"
        else:
            text = PLAN if "ALO_DECOMPOSE" in messages[-1]["content"] else '{"final":"scripted submission"}'
        return {"text": text, "input_tokens": 8, "output_tokens": 4, "latency_ms": 0,
                "cost_microcny": 1, "currency": "CNY"}

    def resources(self):
        return {"model_requests": len(self.calls), "observed_input_tokens": len(self.calls) * 8,
                "observed_output_tokens": len(self.calls) * 4, "estimated_cost_cny": len(self.calls) / 1_000_000}

    def public_quality(self):
        return 0.5

    def local_criterion(self, **kwargs):
        return {"status": self.criterion, "reliable": self.criterion != "unknown", "scope": "node_local",
                "criterion_id": "scripted-contract", "evidence_id": "scripted-local-observation"}

    def budget_remaining(self, model=None):
        return {"requests": 100, "tokens": 100000, "seconds": 1000, "cny": 50}

    def event(self, name, value):
        self.events.append((name, value))

    def __getattr__(self, name):
        if name in {"list_files", "read_file", "write_file", "search", "command", "execute_code"}:
            def unavailable(*args, **kwargs):
                raise AssertionError("this control does not run candidate tools")
            return unavailable
        raise AttributeError(name)


def test_fixed_model_and_transcript_survive_same_task_restore(tmp_path):
    service = ScriptedServices(tmp_path)
    initial = FixedSingleAgent(fixed_model=MODEL_POOL[1])
    assert initial.run_stage({"prompt": "public task"}, service)["status"] == "completed"
    child = FixedSingleAgent(fixed_model=MODEL_POOL[1])
    child.load_state(json.loads(json.dumps(initial.dump_state())))
    child.run_stage({"prompt": "self continuation"}, service)
    assert service.calls == [MODEL_POOL[1], MODEL_POOL[1]]
    assert len(child.messages) > len(initial.messages)


def test_real_alo_runtime_does_not_learn_capability_from_global_quality(tmp_path):
    service = ScriptedServices(tmp_path)
    adapter = alo()
    assert adapter.run_stage({"prompt": "public task"}, service)["status"] == "completed"
    org = Organization(str(tmp_path / "alo.sqlite3"))
    assert org.conn.execute("SELECT count(*) FROM belief").fetchone()[0] == 0
    org.conn.close()
    assert adapter.state["last_native_run"]["node_results"]
    assert adapter.state["rng_state"] and adapter.state["assignments"]
    assert any(name == "alo_observation" for name, _ in service.events)


def test_reliable_contract_updates_native_belief_and_freeze_disables_updates(tmp_path):
    for update in (True, False):
        directory = tmp_path / str(update)
        directory.mkdir()
        service = ScriptedServices(directory, criterion="pass")
        adapter = alo(organization_updates=update)
        adapter.run_stage({"prompt": "public task"}, service)
        org = Organization(str(directory / "alo.sqlite3"))
        assert bool(org.conn.execute("SELECT count(*) FROM belief").fetchone()[0]) is update
        org.conn.close()
        assert bool(adapter.state["observations"]) is update


def test_alo_rejects_changed_pool_after_same_task_checkpoint(tmp_path):
    service = ScriptedServices(tmp_path)
    first = alo()
    first.run_stage({"prompt": "public task"}, service)
    child = alo()
    child.load_state(json.loads(json.dumps(first.dump_state())))
    service.available_models = MODEL_POOL[:-1]
    with pytest.raises(PilotStopped, match="common_pool_changed"):
        child.run_stage({"prompt": "self continuation"}, service)


def test_optional_native_groupchat_uses_native_selector_and_restores_history(tmp_path):
    pytest.importorskip("autogen", reason="native extra not installed; GroupChat execution is untested")
    service = ScriptedServices(tmp_path, groupchat=True)
    roles = {"Planner": MODEL_POOL[0], "Engineer": MODEL_POOL[3], "Critic": MODEL_POOL[1], "Verifier": MODEL_POOL[2]}
    initial = HeterogeneousGroupChat(role_models=roles, manager_model=MODEL_POOL[1])
    assert initial.run_stage({"prompt": "public task"}, service)["status"] == "completed"
    assert MODEL_POOL[2] in service.calls and MODEL_POOL[1] in service.calls
    child = HeterogeneousGroupChat(role_models=roles, manager_model=MODEL_POOL[1])
    child.load_state(json.loads(json.dumps(initial.dump_state())))
    child.run_stage({"prompt": "self continuation"}, service)
    assert len(child.messages) > len(initial.messages)


def test_model_bindings_are_explicit_and_g1_main_is_not_pool_zero():
    assert G1_MAIN_MODEL == "mimo-v2.6-flash"
    assert MODEL_POOL[0] != G1_MAIN_MODEL
    assert G1_MAIN_MODEL in MODEL_POOL
    with pytest.raises(PilotStopped, match="g1_fixed_model_required"):
        adapter_for("G1")
    with pytest.raises(PilotStopped, match="planner_model_required"):
        adapter_for("G2")
    with pytest.raises(PilotStopped, match="groupchat_role_and_manager_binding_required"):
        adapter_for("G3")
    main = adapter_for("G1", {"fixed_model": G1_MAIN_MODEL})
    assert main.fixed_model == G1_MAIN_MODEL
    service = ScriptedServices(None)
    with pytest.raises(PilotStopped, match="fixed_model_required"):
        FixedSingleAgent().run_stage({"prompt": "public task"}, service)
    with pytest.raises(PilotStopped, match="planner_model_required"):
        HeterogeneousALO().run_stage({"prompt": "public task"}, service)
    with pytest.raises(PilotStopped, match="groupchat_role_and_manager_binding_required"):
        HeterogeneousGroupChat()._bind(MODEL_POOL)


def test_probe_does_not_write_source_and_cold_prior_ignores_posterior(tmp_path):
    learning = ScriptedServices(tmp_path, criterion="pass")
    trained = alo()
    trained.run_stage({"prompt": "public task"}, learning)
    source = tmp_path / "alo.sqlite3"
    org = Organization(str(source))
    trained_rows = org.conn.execute("SELECT count(*) FROM belief").fetchone()[0]
    org.conn.close()
    assert trained_rows > 0
    before = source.read_bytes()

    learned = ScriptedServices(tmp_path, criterion="pass")
    terminal = alo(probe=True)
    terminal.state["assignments"] = {"kept": {"role_id": "kept", "identity_id": "i",
                                              "model_id": MODEL_POOL[0], "bandwidth": 1,
                                              "authority": ("inspect",)}}
    sentinel_rng = random.Random(7).getstate()
    terminal.state["rng_state"] = sentinel_rng
    terminal.state["worker_transcripts"] = {"kept": [{"role": "user", "content": "keep"}]}
    terminal.state["planner_messages"] = [{"role": "system", "content": "keep"}]
    terminal.state["observations"] = {"kept": [{"public_quality": 0.5}]}
    terminal.run_stage({"prompt": "probe"}, learned)
    assert source.read_bytes() == before
    assert terminal.state["probe_isolation"]["prior"] == "learned_terminal"
    assert terminal.state["probe_isolation"]["belief_rows"] == trained_rows
    assert terminal.state["observations"] == {"kept": [{"public_quality": 0.5}]}
    assert terminal.state["assignments"]["kept"]["role_id"] == "kept"
    assert terminal.state["rng_state"] == sentinel_rng
    assert terminal.state["worker_transcripts"] == {"kept": [{"role": "user", "content": "keep"}]}
    assert terminal.state["planner_messages"] == [{"role": "system", "content": "keep"}]
    assert "last_native_run" not in terminal.state

    cold_service = ScriptedServices(tmp_path, criterion="pass")
    cold = alo(probe=True, cold_prior=True)
    cold.run_stage({"prompt": "cold probe"}, cold_service)
    assert source.read_bytes() == before
    assert cold.state["probe_isolation"]["prior"] == "cold"
    assert cold.state["probe_isolation"]["belief_rows"] == 0
    assert cold.config["organization_updates"] is False
    with pytest.raises(PilotStopped, match="probe_forbids_organization_updates"):
        alo(probe=True, organization_updates=True)
    with pytest.raises(PilotStopped, match="cold_prior_requires_probe"):
        alo(cold_prior=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PilotStopped, match="learned_terminal_probe_requires_source_organization"):
        alo(probe=True).run_stage({"prompt": "missing"}, ScriptedServices(empty, criterion="pass"))
