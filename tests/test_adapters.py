"""Scripted service controls of preserved algorithms, never benchmark evidence."""
import json
from types import SimpleNamespace

import pytest

from alo.org.organization import Organization
from alo_cloud.adapters import FixedSingleAgent, HeterogeneousALO, HeterogeneousGroupChat
from alo_cloud.contracts import MODEL_POOL
from alo_cloud.errors import PilotStopped


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
    adapter = HeterogeneousALO()
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
        adapter = HeterogeneousALO(organization_updates=update)
        adapter.run_stage({"prompt": "public task"}, service)
        org = Organization(str(directory / "alo.sqlite3"))
        assert bool(org.conn.execute("SELECT count(*) FROM belief").fetchone()[0]) is update
        org.conn.close()
        assert bool(adapter.state["observations"]) is update


def test_alo_rejects_changed_pool_after_same_task_checkpoint(tmp_path):
    service = ScriptedServices(tmp_path)
    first = HeterogeneousALO()
    first.run_stage({"prompt": "public task"}, service)
    child = HeterogeneousALO()
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
