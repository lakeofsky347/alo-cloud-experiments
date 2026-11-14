"""Native adapters for revised M1/M2 with dynamic official model identities.

The core ALO runtime currently accepts only Boolean criteria. Instance-local
store wrappers gate its learning writes using a separate three-valued public
local contract. Unknown does not become a B/TMS success or failure. Under
``repro-v1`` it also does not mark the DAG node successful. Protocol completion
and functional pass are separate fields.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import time

from .native_bridge import BoundModel, native_client
from .contracts import G1_DEEPSEEK_FLASH_MODEL
from .errors import PilotStopped
from .public_tools import (
    ACTION_INSTRUCTIONS, TOOLS, WORK_ORDER, PublicToolCodeExecutor, coerce_actions, dispatch_action,
)
from .native_bridge import ROLE_INSTRUCTIONS, SHARED_INSTRUCTIONS, _terminated
from .policy import ConstrainedThompsonPolicy, POLICY_VERSION, workspace_quality_credit
from .public_check import public_quality_signal, quality_from_signal
from .repro import REPRO_CRITERION_ID, REPRO_INSTRUCTION, cached_repro_v1
from .review_isolation import role_is_review, snapshot_workspace, tree_digest
from .sandbox import CRITERION_ID
from .tool_loop import TOOL_LOOP_VERSION, persistent_tool_loop

ROUTING_MODES = ("thompson", "fixed", "random")
FIXED_ROUTE_MODEL = G1_DEEPSEEK_FLASH_MODEL
EXPORT_MODES = ("full", "resources")


def role_tool_contract(role):
    """Tools dispatch will enforce, or None for the full public set.

    An empty declaration is not a restriction. Fallback generalists and any
    other role with no tool contract keep the pre-#34 tool set. A non-empty
    list of real public tools, including a review role limited to ``command``,
    is enforced. Names that are not in ``TOOLS`` are dropped. A declaration
    whose names are all unknown, such as a planner placeholder ``public_tools``,
    is unrestricted so the placeholder cannot reject every real tool.
    """
    raw = getattr(role, "tools", None)
    if not raw:
        return None
    known = tuple(name for name in raw if name in TOOLS)
    if not known:
        return None
    return known
FIXED_S1_S2_DAG = """subtasks:
  - id: s1
    goal: Write alo_repro.py and run python3 alo_repro.py before any fix so it exits non-zero. Then edit the source. Run python3 alo_repro.py again so it exits 0. Do not weaken the script.
    deps: []
    topics: [coding, implementation]
    role:
      capability_reqs: [coding]
      epistemic_scope:
        include: [task_description, goal, persistent_workspace]
        exclude: [billing, credentials]
      authority: [inspect]
      tools: [list_files, read_file, write_file, edit_file, search, command]
      output_schema: source
      success_criteria: non_empty_output
  - id: s2
    goal: Run alo_repro.py and the touched public tests. Do not reimplement the fix.
    deps: [s1]
    topics: [testing, validation]
    role:
      capability_reqs: [testing]
      epistemic_scope:
        include: [task_description, goal, persistent_workspace]
        exclude: [billing, credentials]
      authority: [inspect]
      tools: [command]
      output_schema: verification
      success_criteria: non_empty_output
"""

ROLE_ORDER = ("Planner", "Engineer", "Critic", "Verifier")
_UNSET = object()


def _db_digest(path):
    """Hash a closed SQLite file and any leftover WAL/SHM siblings."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.is_file():
            digest.update(b"\0")
            digest.update(candidate.name.encode())
            digest.update(candidate.read_bytes())
    return digest.hexdigest()


def _copy_db(source, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    for suffix in ("-wal", "-shm"):
        extra = Path(str(source) + suffix)
        if extra.is_file():
            shutil.copy2(extra, Path(str(dest) + suffix))

def _models(services):
    models = tuple(services.available_models)
    if not models or len(set(models)) != len(models) or any(not isinstance(x, str) or not x for x in models):
        raise PilotStopped("invalid_common_model_pool")
    return models


def _known_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def _delta(after, before, key):
    if not _known_number(after.get(key)) or not _known_number(before.get(key)):
        return None
    return max(0, after[key] - before[key])


def _quality(services):
    value = services.public_quality()
    return float(value) if _known_number(value) and value <= 1 else None


def _is_verifier(node_id, role_id):
    node_id = str(node_id or "")
    role_id = str(role_id or "")
    return node_id in {"s2", "verify", "verifier"} or "verif" in role_id.lower()


def _changed_paths(services):
    changed = []
    log = getattr(services, "edit_log", None)
    if isinstance(log, list):
        for row in log:
            if isinstance(row, dict):
                changed.extend(path for path in row.get("paths") or [] if isinstance(path, str))
    return changed


def _node_turn_cap(adapter, node_id, role_id):
    caps = adapter.config.get("node_request_caps") or {}
    if node_id in caps:
        return caps[node_id]
    if role_id in caps:
        return caps[role_id]
    return adapter.config["max_turns"]


class FixedSingleAgent:
    """One frozen model with a checkpoint-owned transcript and public tools."""

    def __init__(self, *, fixed_model=None, max_turns=48, criterion_id=None, protocol_id=None,
                 request_accounting=None, max_attempts=None, planner_requests=None,
                 verifier_requests=None, topology=None):
        if type(max_turns) is not int or max_turns < 1:
            raise PilotStopped("invalid_single_turn_limit")
        if criterion_id is not None and criterion_id not in {CRITERION_ID, REPRO_CRITERION_ID}:
            raise PilotStopped("invalid_criterion_id")
        self.fixed_model, self.max_turns, self.messages = fixed_model, max_turns, []
        self.criterion_id = criterion_id
        self.protocol_id = protocol_id
        self.request_accounting = request_accounting
        self.max_attempts = max_attempts
        self.planner_requests = planner_requests
        self.verifier_requests = verifier_requests
        self.topology = topology
        self.public_signal = None
        self.protocol_completed = None
        self.functional_pass = None

    def dump_state(self):
        payload = {"schema": "dualbench-single-v2", "fixed_model": self.fixed_model,
                   "max_turns": self.max_turns, "tool_loop_version": TOOL_LOOP_VERSION,
                   "messages": copy.deepcopy(self.messages)}
        if self.criterion_id:
            payload["criterion_id"] = self.criterion_id
            payload["protocol_id"] = self.protocol_id
            payload["public_signal"] = self.public_signal
            payload["protocol_completed"] = self.protocol_completed
            payload["functional_pass"] = self.functional_pass
        return payload

    def load_state(self, state):
        if (state.get("schema") != "dualbench-single-v2" or state.get("max_turns") != self.max_turns
                or state.get("tool_loop_version") != TOOL_LOOP_VERSION
                or (self.fixed_model is not None and state.get("fixed_model") != self.fixed_model)):
            raise PilotStopped("single_checkpoint_configuration_changed")
        if self.criterion_id and state.get("criterion_id") != self.criterion_id:
            raise PilotStopped("single_checkpoint_configuration_changed")
        self.fixed_model = state["fixed_model"]
        self.messages = copy.deepcopy(state["messages"])

    def run_stage(self, stage, services):
        models = _models(services)
        if not self.fixed_model:
            raise PilotStopped("fixed_model_required")
        if self.fixed_model not in models:
            raise PilotStopped("fixed_model_outside_common_pool")
        services.current_role = "g1"
        services.current_node = "g1"
        if self.criterion_id:
            try:
                services.active_criterion_id = self.criterion_id
            except Exception:
                pass
        system = "\n" + REPRO_INSTRUCTION if self.criterion_id == REPRO_CRITERION_ID else ""
        result = persistent_tool_loop(BoundModel(services, self.fixed_model),
            json.dumps(stage, ensure_ascii=False), self.messages, system=system, max_turns=self.max_turns)
        if self.criterion_id != REPRO_CRITERION_ID:
            return result
        bundle = cached_repro_v1(services, changed_paths=_changed_paths(services), messages=self.messages)
        protocol_completed = result.get("status") == "completed"
        functional_pass = bundle.get("signal") == "pass"
        self.public_signal = bundle.get("signal")
        self.protocol_completed = protocol_completed
        self.functional_pass = functional_pass
        self.repro_v1 = {
            "signal": bundle.get("signal"),
            "reason": bundle.get("reason"),
            "protocol_completed": protocol_completed,
            "functional_pass": functional_pass,
            "repro_before_returncode": bundle.get("repro_before_returncode"),
            "repro_after_returncode": bundle.get("repro_after_returncode"),
            "tests_returncode": bundle.get("tests_returncode"),
            "tests_base_returncode": bundle.get("tests_base_returncode"),
        }
        reported = dict(result)
        reported["protocol_completed"] = protocol_completed
        reported["functional_pass"] = functional_pass
        reported["public_signal"] = bundle.get("signal")
        reported["criterion_id"] = REPRO_CRITERION_ID
        # Submission is not a functional pass. Unknown stays unsuccessful.
        reported["status"] = "completed" if functional_pass else "failed"
        return reported


class HeterogeneousGroupChat:
    """AutoGen's native auto selection, broadcast and resume, with frozen bindings."""

    def __init__(self, *, role_models=None, manager_model=None, max_round=60):
        self.role_models = copy.deepcopy(role_models)
        self.manager_model = manager_model
        self.max_round = max_round
        self.messages = []
        self.model_pool = None
        if type(max_round) is not int or max_round < 2:
            raise PilotStopped("invalid_groupchat_round_limit")

    def _bind(self, models):
        if self.model_pool is not None and tuple(self.model_pool) != tuple(models):
            raise PilotStopped("groupchat_checkpoint_common_pool_changed")
        self.model_pool = tuple(models)
        # Role and selector bindings are not frozen. Pool position is not a contract.
        if not self.role_models or not self.manager_model:
            raise PilotStopped("groupchat_role_and_manager_binding_required")
        if (set(self.role_models) - {*ROLE_INSTRUCTIONS, "Verifier"}
                or any(m not in models for m in self.role_models.values()) or self.manager_model not in models):
            raise PilotStopped("groupchat_model_configuration_outside_common_pool")

    def dump_state(self):
        return {"schema": "dualbench-groupchat-v2", "role_models": copy.deepcopy(self.role_models),
                "manager_model": self.manager_model, "max_round": self.max_round,
                "model_pool": list(self.model_pool) if self.model_pool else None,
                "messages": copy.deepcopy(self.messages)}

    def load_state(self, state):
        if (state.get("schema") != "dualbench-groupchat-v2" or state.get("max_round") != self.max_round
                or (self.role_models is not None and state.get("role_models") != self.role_models)
                or (self.manager_model is not None and state.get("manager_model") != self.manager_model)):
            raise PilotStopped("groupchat_checkpoint_configuration_changed")
        self.role_models = copy.deepcopy(state["role_models"])
        self.manager_model = state["manager_model"]
        self.model_pool = tuple(state["model_pool"]) if state.get("model_pool") else None
        self.messages = copy.deepcopy(state["messages"])

    def run_stage(self, stage, services):
        import autogen as native
        from alo_cloud.public_tools import PublicActionExtractor
        if importlib.metadata.version("autogen-agentchat") != "0.2.40":
            raise PilotStopped("native_groupchat_version_changed")
        self._bind(_models(services))
        client = native_client(services)

        def config(model):
            return {"config_list": [{"model": model, "model_client_cls": client.__name__}],
                    "temperature": 0, "cache_seed": None, "max_tokens": 4096}

        instructions = SHARED_INSTRUCTIONS + ACTION_INSTRUCTIONS
        role_instructions = {**ROLE_INSTRUCTIONS,
            "Verifier": "Check the implemented behavior with public commands and diagnose regressions."}
        agents = [native.AssistantAgent(name=name,
            system_message=role_instructions[name] + " " + instructions,
            llm_config=config(model), code_execution_config=False, human_input_mode="NEVER",
            is_termination_msg=_terminated) for name in ROLE_ORDER if name in self.role_models
            for model in [self.role_models[name]]]
        if not isinstance(getattr(services, "g3_request_split", None), dict):
            services.g3_request_split = {"manager_selection": 0, "role": 0, "dispatches": 0}

        def _execute_role_text(role_name, text):
            if not isinstance(text, str) or "[action_executed]" in text:
                return text
            found = coerce_actions(text)
            if not found or not found.get("actions") or "tool" not in found["actions"][0]:
                return text
            action = found["actions"][0]
            services.current_role = role_name
            services.current_node = role_name
            try:
                result = dispatch_action(services, action)
            except (ValueError, TypeError, KeyError) as error:
                result = {"public_tool_error": type(error).__name__, "detail": str(error)[:500]}
            split = services.g3_request_split
            split["dispatches"] = split.get("dispatches", 0) + 1
            extra = len(found["actions"]) - 1
            note = f"\nignored {extra} extra actions" if extra else ""
            rendered = json.dumps({"tool": action["tool"], "result": result}, ensure_ascii=False)
            return text + "\n" + rendered + note + "\n[action_executed]"

        for agent in agents:
            agent.register_model_client(model_client_cls=client)
            original = agent.generate_reply

            def generate_reply(messages=None, sender=None, *args, _agent=agent, _original=original, **kwargs):
                reply = _original(messages, sender, *args, **kwargs)
                if isinstance(reply, str):
                    return _execute_role_text(_agent.name, reply)
                content = getattr(reply, "content", None)
                if isinstance(content, str):
                    reply.content = _execute_role_text(_agent.name, content)
                return reply

            agent.generate_reply = generate_reply
        workspace = native.UserProxyAgent(name="Workspace", human_input_mode="NEVER", llm_config=False,
            code_execution_config={"executor": PublicToolCodeExecutor(services, PublicActionExtractor()),
                                   "last_n_messages": 1},
            default_auto_reply="Use the shared public tools or submit STAGE_DONE.", is_termination_msg=_terminated)
        group = native.GroupChat(agents=[workspace, *agents], messages=[], max_round=self.max_round,
            speaker_selection_method="auto", allow_repeat_speaker=True,
            select_speaker_auto_model_client_cls=client,
            select_speaker_auto_llm_config=config(self.manager_model))
        manager = native.GroupChatManager(groupchat=group, llm_config=config(self.manager_model),
            human_input_mode="NEVER", is_termination_msg=_terminated, code_execution_config=False)
        manager.register_model_client(model_client_cls=client)
        message = {"role": "user", "name": "Workspace", "content": json.dumps(stage, ensure_ascii=False)}
        prior = len(self.messages)
        if self.messages:
            sender, message = manager.resume(copy.deepcopy(self.messages) + [message], silent=True)
        else:
            sender = workspace
        try:
            sender.initiate_chat(manager, message=message, clear_history=False, silent=True, cache=None)
        finally:
            self.messages = copy.deepcopy(group.messages)
        return {"status": "completed" if any(_terminated(x) for x in self.messages[prior:]) else "failed",
                "summary": "native GroupChat protocol submission; independent final grading is separate"}


class _BeliefGate:
    def __init__(self, store, allowed):
        self.store, self.allowed = store, allowed

    def __getattr__(self, name):
        return getattr(self.store, name)

    def update_objective(self, model, cell, success):
        return (self.store.update_objective(model, cell, success) if self.allowed()
                else self.store.query(model, cell))


class _TMSGate:
    def __init__(self, store, allowed):
        self.store, self.allowed = store, allowed

    def __getattr__(self, name):
        return getattr(self.store, name)

    def record_outcome(self, *args, **kwargs):
        if self.allowed():
            return self.store.record_outcome(*args, **kwargs)

    def record_expertise(self, *args, **kwargs):
        if self.allowed():
            return self.store.record_expertise(*args, **kwargs)


class HeterogeneousALO:
    def __init__(self, *, planner_model=None, max_turns=10, max_attempts=2, seed=20261027,
                 policy_priors=None, safety_multiplier=1.25, organization_updates=_UNSET,
                 probe=False, cold_prior=False, user_preference=None, routing_mode="thompson",
                 criterion_id=None, fixed_topology=False, lightweight_verifier=False,
                 node_request_caps=None, protocol_id=None, request_accounting=None,
                 planner_requests=None, verifier_requests=None, topology=None,
                 learn_quality=True, filter_resources_by_topics=False,
                 fixed_model=None, export_mode=None,
                 pb_mode=False, pb_dag=None, pb_prompt=None):
        if (type(max_turns) is not int or max_turns < 1
                or type(max_attempts) is not int or max_attempts < 1):
            raise PilotStopped("invalid_alo_attempt_or_turn_limit")
        if routing_mode not in ROUTING_MODES:
            raise PilotStopped("invalid_routing_mode")
        if criterion_id is None:
            criterion_id = CRITERION_ID
        elif criterion_id not in {CRITERION_ID, REPRO_CRITERION_ID}:
            raise PilotStopped("invalid_criterion_id")
        if node_request_caps is not None:
            if (not isinstance(node_request_caps, dict) or not node_request_caps
                    or any(not isinstance(key, str) or not key or type(value) is not int or value < 0
                           for key, value in node_request_caps.items())):
                raise PilotStopped("invalid_node_request_caps")
        if user_preference is not None:
            if (not isinstance(user_preference, dict) or set(user_preference) != {"first_try", "strong"}
                    or any(not isinstance(user_preference[key], str) or not user_preference[key]
                           for key in ("first_try", "strong"))
                    or user_preference["first_try"] == user_preference["strong"]):
                raise PilotStopped("invalid_user_preference")
        if export_mode is not None and export_mode not in EXPORT_MODES:
            raise PilotStopped("invalid_export_mode")
        if fixed_model is not None and (not isinstance(fixed_model, str) or not fixed_model):
            raise PilotStopped("invalid_fixed_model")
        if cold_prior and not probe:
            raise PilotStopped("cold_prior_requires_probe")
        if organization_updates is _UNSET:
            organization_updates = not probe
        elif probe and organization_updates:
            raise PilotStopped("probe_forbids_organization_updates")
        if probe:
            organization_updates = False
        self.config = {"planner_model": planner_model, "max_turns": max_turns,
                       "max_attempts": max_attempts, "seed": seed,
                       "policy_priors": copy.deepcopy(policy_priors or {}),
                       "safety_multiplier": safety_multiplier,
                       "organization_updates": bool(organization_updates),
                       "probe": bool(probe), "cold_prior": bool(cold_prior),
                       "user_preference": copy.deepcopy(user_preference),
                       "routing_mode": routing_mode,
                       "criterion_id": criterion_id,
                       "fixed_topology": bool(fixed_topology),
                       "lightweight_verifier": bool(lightweight_verifier),
                       "node_request_caps": copy.deepcopy(node_request_caps),
                       "protocol_id": protocol_id,
                       "request_accounting": request_accounting,
                       "planner_requests": planner_requests,
                       "verifier_requests": verifier_requests,
                       "topology": topology,
                       "policy_version": POLICY_VERSION,
                       "tool_loop_version": TOOL_LOOP_VERSION}
        # Opt-in fields stay out of the frozen g2 configuration when they are
        # at the historical default, so an existing checkpoint still matches.
        if learn_quality is not True:
            self.config["learn_quality"] = bool(learn_quality)
        if filter_resources_by_topics:
            self.config["filter_resources_by_topics"] = True
        if fixed_model not in (None, FIXED_ROUTE_MODEL):
            self.config["fixed_model"] = fixed_model
        if export_mode == "resources":
            self.config["export_mode"] = "resources"
        # ProgramBench fields stay out of the frozen g2 configuration unless
        # this adapter was built for that track.
        if pb_mode:
            if not isinstance(pb_dag, str) or not pb_dag.strip():
                raise PilotStopped("pb_dag_required")
            if pb_prompt is not None and not isinstance(pb_prompt, str):
                raise PilotStopped("invalid_pb_prompt")
            self.config["pb_mode"] = True
            self.config["pb_dag"] = pb_dag
            self.config["pb_prompt"] = pb_prompt or ""
        self.state = {"observations": {}, "worker_transcripts": {}, "planner_messages": []}

    def dump_state(self):
        return {"schema": "dualbench-alo-v2", "configuration": copy.deepcopy(self.config),
                **copy.deepcopy(self.state)}

    def load_state(self, state):
        if state.get("schema") != "dualbench-alo-v2" or state.get("configuration") != self.config:
            raise PilotStopped("alo_checkpoint_configuration_changed")
        self.state = copy.deepcopy({k: v for k, v in state.items() if k not in {"schema", "configuration"}})

    def run_stage(self, stage, services):
        from alo.models.base import Completion, ModelInfo
        from alo.org.organization import Assignment, Organization
        from alo.runtime.orchestrator import run_task
        from alo.runtime.remediation import Decision
        models = _models(services)
        if "model_pool" in self.state and tuple(self.state["model_pool"]) != models:
            raise PilotStopped("alo_checkpoint_common_pool_changed")
        self.state["model_pool"] = list(models)
        self.state["last_criterion_statuses"] = []
        planner_model = self.config["planner_model"]
        if not planner_model:
            raise PilotStopped("planner_model_required")
        if planner_model not in models:
            raise PilotStopped("planner_model_outside_common_pool")
        adapter, active, last = self, {}, {}
        rng = random.Random(self.config["seed"])
        if "rng_state" in self.state:
            def tuples(value):
                return tuple(tuples(x) for x in value) if isinstance(value, (list, tuple)) else value
            rng.setstate(tuples(self.state["rng_state"]))
        source_db = Path(services.state_dir) / "alo.sqlite3"
        probe = self.config["probe"]
        if probe and self.config["organization_updates"]:
            raise PilotStopped("probe_forbids_organization_updates")
        probe_dir = None
        source_digest = _db_digest(source_db) if probe else None
        observations_before = copy.deepcopy(self.state["observations"]) if probe else None
        retained = copy.deepcopy({
            "observations": self.state.get("observations", {}),
            "worker_transcripts": self.state.get("worker_transcripts", {}),
            "planner_messages": self.state.get("planner_messages", []),
            "assignments": self.state.get("assignments"),
            "rng_state": self.state.get("rng_state"),
            "last_native_run": self.state.get("last_native_run"),
            "last_criterion_statuses": self.state.get("last_criterion_statuses"),
        }) if probe else None
        if probe:
            probe_dir = tempfile.TemporaryDirectory(prefix="alo-probe-")
            db_path = Path(probe_dir.name) / "alo.sqlite3"
            if self.config["cold_prior"]:
                # Common prior only. Do not copy the last reset task's posterior.
                pass
            elif source_digest is None:
                probe_dir.cleanup()
                raise PilotStopped("learned_terminal_probe_requires_source_organization")
            else:
                _copy_db(source_db, db_path)
        else:
            db_path = source_db
        try:
            org = Organization(str(db_path))
        except BaseException:
            if probe_dir is not None:
                probe_dir.cleanup()
            raise
        learn = lambda: (
            self.config["organization_updates"]
            and self.config.get("learn_quality", True)
            and (
                last.get("public_signal") in {"pass", "fail"}
                if self.config.get("pb_mode")
                else last.get("local_contract_status") in {"pass", "fail"}
            )
        )
        org.belief, org.tms = _BeliefGate(org.belief, learn), _TMSGate(org.tms, learn)
        for role, assignment in self.state.get("assignments", {}).items():
            org._assignments[role] = Assignment(**assignment)
        try:
            services.active_criterion_id = self.config["criterion_id"]
        except Exception:
            pass
        policy = ConstrainedThompsonPolicy(self.state.setdefault("observations", {}),
            services.budget_remaining, priors=self.config["policy_priors"],
            safety_multiplier=self.config["safety_multiplier"],
            learn_quality=self.config.get("learn_quality", True),
            filter_resources_by_topics=self.config.get("filter_resources_by_topics", False))
        pref = self.config.get("user_preference")
        routing_mode = self.config["routing_mode"]

        class PreferencePolicy:
            """First feasible assignment is the preferred model. Thompson remains the counterfactual."""

            def select(self, org, pool, worker_ids, cell, topics, user_id, rng=None):
                chosen, trace = policy.select(org, pool, worker_ids, cell, topics, user_id, rng)
                if not isinstance(pref, dict):
                    return chosen, trace
                first = pref["first_try"]
                score = trace.get("scores", {}).get(first, {})
                applied = bool(isinstance(score, dict) and score.get("feasible") and first in worker_ids)
                actual = first if applied else chosen
                record = {"point": "assignment", "preference_applied": applied,
                          "escalated_to_strong": False, "counterfactual": chosen, "selected": actual}
                adapter.state.setdefault("decision_records", []).append(record)
                traced = dict(trace)
                traced.update(record)
                traced["selected"] = actual
                return actual, traced

        class RoutedPolicy:
            """fixed keeps the pinned model (default deepseek-flash). random draws uniformly."""

            def select(self, org, pool, worker_ids, cell, topics, user_id, rng=None):
                chosen, trace = policy.select(org, pool, worker_ids, cell, topics, user_id, rng)
                if routing_mode == "fixed":
                    pinned = adapter.config.get("fixed_model") or FIXED_ROUTE_MODEL
                    score = trace.get("scores", {}).get(pinned, {})
                    if not (isinstance(score, dict) and score.get("feasible")
                            and pinned in worker_ids):
                        raise PilotStopped("fixed_model_infeasible")
                    actual = pinned
                else:
                    feasible = [model for model in worker_ids
                                if trace["scores"][model]["feasible"]]
                    actual = (rng or random.Random(0)).choice(feasible)
                record = {"point": "assignment", "routing_mode": routing_mode,
                          "preference_applied": False, "escalated_to_strong": False,
                          "counterfactual": chosen, "selected": actual}
                adapter.state.setdefault("decision_records", []).append(record)
                traced = dict(trace)
                traced.update(record)
                traced["selected"] = actual
                return actual, traced

        if routing_mode == "thompson" and isinstance(pref, dict):
            assignment = PreferencePolicy()
        elif routing_mode == "thompson":
            assignment = policy
        else:
            assignment = RoutedPolicy()

        class Bridge:
            def __init__(self, model, *, planner=False):
                self.model, self.planner = model, planner
                # Provider identity is supplied by the gateway; no substring
                # inference is used for dispatch or credential selection.
                self.info = ModelInfo("planner::" + model if planner else model, "official", model)

            def complete(self, prompt, *, system="", max_tokens=4096):
                if self.planner:
                    pref = adapter.config.get("user_preference")
                    if isinstance(pref, dict) and not adapter.state.get("planner_preference_logged"):
                        adapter.state["planner_preference_logged"] = True
                        adapter.state.setdefault("decision_records", []).append({
                            "point": "planner",
                            "preference_applied": self.model == pref["first_try"],
                            "escalated_to_strong": False,
                            "counterfactual": None,
                            "selected": self.model,
                        })
                    messages = adapter.state.setdefault("planner_messages", [])
                    if not messages:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": prompt})
                    reply = services.complete_for_model(self.model, messages, max_tokens=max_tokens)
                    messages.append({"role": "assistant", "content": reply["text"]})
                    return Completion(reply["text"], reply["input_tokens"], reply["output_tokens"], reply["latency_ms"])
                messages = adapter.state.setdefault("worker_transcripts", {}).setdefault(active["role_id"], [])
                services.current_role = active.get("role_id")
                services.current_node = last.get("node_id")
                node_id = str(last.get("node_id") or "")
                role_id = str(active.get("role_id") or "")
                if adapter.config.get("lightweight_verifier") and _is_verifier(node_id, role_id):
                    # Zero model requests. This is a public re-test of the
                    # implementation, not an independent reviewer.
                    last["termination"] = "lightweight_verifier"
                    last["verifier_kind"] = "zero_request_public_retest"
                    last.update(model=self.model, submission_status="completed",
                                tool_index_after=getattr(services, "tools", None))
                    return Completion(
                        "zero-request public re-test of the implementation; not an independent reviewer",
                        0, 0, 0)
                turn_cap = _node_turn_cap(adapter, node_id, role_id)
                if type(turn_cap) is not int or turn_cap < 1:
                    raise PilotStopped("node_request_cap_invalid")
                worker_system = system + "\n" + WORK_ORDER
                if adapter.config.get("criterion_id") == REPRO_CRITERION_ID:
                    worker_system += "\n" + REPRO_INSTRUCTION
                result = persistent_tool_loop(BoundModel(services, self.model), prompt, messages,
                    system=worker_system, max_turns=turn_cap)
                last["termination"] = result.get("termination")
                last.update(model=self.model, submission_status=result["status"],
                            tool_index_after=getattr(services, "tools", None))
                return Completion(result["summary"], 0, 0, 0)

        def criterion(role, output):
            verifier_start = time.monotonic()
            quality_after = _quality(services)
            context = {key: last.get(key) for key in ("node_id", "role_id", "role_goal", "attempt",
                "tool_index_before", "tool_index_after", "submission_status")}
            signal = "unknown"
            changed = _changed_paths(services)
            if adapter.config.get("criterion_id") == REPRO_CRITERION_ID:
                role_messages = (adapter.state.get("worker_transcripts") or {}).get(role.role_id)
                bundle = cached_repro_v1(
                    services, changed_paths=changed,
                    messages=role_messages if isinstance(role_messages, list) else None)
                adapter.state.setdefault("repro_v1_measurements", []).append({
                    "node_id": last.get("node_id"),
                    "role_id": role.role_id,
                    "model_id": last.get("model"),
                    "repro_before_returncode": bundle.get("repro_before_returncode"),
                    "repro_after_returncode": bundle.get("repro_after_returncode"),
                    "recorded_before_returncode": bundle.get("recorded_before_returncode"),
                    "recorded_after_returncode": bundle.get("recorded_after_returncode"),
                    "tests_returncode": bundle.get("tests_returncode"),
                    "tests_base_returncode": bundle.get("tests_base_returncode"),
                    "same_frozen_tests": bundle.get("same_frozen_tests"),
                    "comparable_exits": bundle.get("comparable_exits"),
                    "signal": bundle.get("signal"),
                    "reason": bundle.get("reason"),
                    "protocol_completed": None,
                    "functional_pass": bundle.get("signal") == "pass",
                })
                # public_quality_signal sees the return codes inside the bundle.
                # A syntax-only pass is not reused as the repro-v1 signal.
                signal = bundle.get("signal") or "unknown"
            else:
                signal_fn = getattr(services, "public_signal", None)
                if callable(signal_fn):
                    try:
                        signal = signal_fn()
                    except Exception:
                        signal = "unknown"
                else:
                    root = getattr(getattr(services, "workspace", None), "root", None)
                    if root is not None:
                        signal = public_quality_signal(root, changed_paths=changed)
            if signal not in {"pass", "fail", "unknown"}:
                signal = "unknown"
            last["public_signal"] = signal
            signal_quality = quality_from_signal(signal)
            evidence = {"status": "unknown", "reason": "no_reliable_local_criterion"}
            hook = getattr(services, "local_criterion", None)
            if callable(hook):
                supplied = hook(role_id=role.role_id, model_id=last["model"], output=output, context=context)
                if isinstance(supplied, dict):
                    evidence = dict(supplied)
            reliable = (evidence.get("status") in {"pass", "fail"}
                        and evidence.get("reliable") is True and evidence.get("scope") == "node_local"
                        and isinstance(evidence.get("evidence_id"), str) and bool(evidence["evidence_id"])
                        and isinstance(evidence.get("criterion_id"), str) and bool(evidence["criterion_id"]))
            local_status = evidence["status"] if reliable else "unknown"
            last["local_contract_status"] = local_status
            adapter.state["last_criterion_statuses"].append({"node_id": last["node_id"],
                "role_id": role.role_id, "model_id": last["model"], "attempt": last["attempt"],
                "local_contract_status": local_status})
            if signal_quality is not None:
                quality_after = signal_quality
            workspace_changed = None
            diff_nonempty = False
            reader = getattr(services, "diff_nonempty", None)
            if callable(reader):
                try:
                    diff_nonempty = bool(reader())
                    workspace_changed = diff_nonempty
                except Exception:
                    diff_nonempty = False
                    workspace_changed = None
            declared = last.get("role_tools", ())
            # None is the unbound full set. An empty tuple is not that signal.
            credit_tools = TOOLS if declared is None else tuple(declared)
            topics = list(last.get("topics") or ())
            credited = workspace_quality_credit(
                quality_after, quality_before=last.get("quality_before"),
                role_tools=credit_tools, workspace_changed=workspace_changed)
            after = services.resources()
            before = last["resources_before"]
            input_tokens = _delta(after, before, "observed_input_tokens")
            output_tokens = _delta(after, before, "observed_output_tokens")
            observation = {"public_quality": quality_after, "quality_before": last["quality_before"],
                "quality_after": quality_after,
                "delta_quality": quality_after - last["quality_before"] if quality_after is not None and last["quality_before"] is not None else None,
                "credited_quality": credited,
                "topics": topics,
                "role_id": role.role_id,
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "tokens": input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None,
                "reasoning_tokens": _delta(after, before, "observed_reasoning_tokens"),
                "requests": _delta(after, before, "model_requests"),
                "cost_cny": _delta(after, before, "estimated_cost_cny"),
                "seconds": time.monotonic() - last["started"],
                "verifier_seconds": last["pre_verifier_seconds"] + time.monotonic() - verifier_start,
                "local_contract_status": local_status}
            record_observation = adapter.config["organization_updates"]
            # A ProgramBench unknown is not a quality or resource write.
            if adapter.config.get("pb_mode") and signal not in {"pass", "fail"}:
                record_observation = False
            if record_observation:
                adapter.state["observations"].setdefault(last["model"], []).append(observation)
            services.event("alo_observation", {"model": last["model"], "context": context,
                "local_evidence": evidence, "learning_update_allowed": learn(),
                "global_quality_scope": "role_topics_credit_not_whole_workspace", **observation})
            # Core Boolean success describes execution/contract flow. The
            # instance-local stores above separately gate objective learning.
            node_id = str(last.get("node_id") or "")
            role_id = str(role.role_id or "")
            verifier = node_id in {"s2", "verify", "verifier"} or "verif" in role_id.lower()
            last["diff_nonempty"] = bool(diff_nonempty)
            protocol_completed = last.get("submission_status") == "completed"
            functional_pass = signal == "pass"
            last["protocol_completed"] = protocol_completed
            last["functional_pass"] = functional_pass
            adapter.state.setdefault("node_outcomes", []).append({
                "node_id": last.get("node_id"),
                "role_id": role.role_id,
                "protocol_completed": protocol_completed,
                "functional_pass": functional_pass,
                "public_signal": signal,
            })
            measurements = adapter.state.get("repro_v1_measurements") or []
            if measurements and measurements[-1].get("node_id") == last.get("node_id"):
                measurements[-1]["protocol_completed"] = protocol_completed
                measurements[-1]["functional_pass"] = functional_pass
            # A non-empty implementation diff still lets s2 run. An empty diff
            # fails the node only when a user preference asks for escalation.
            pref = adapter.config.get("user_preference")
            if adapter.config.get("criterion_id") == REPRO_CRITERION_ID:
                # Unknown is not node success. B/TMS still ignore it via the gate.
                return functional_pass
            if adapter.config.get("pb_mode"):
                # Compile/differential unknown is not node success and is not stored.
                return signal == "pass"
            if isinstance(pref, dict) and not diff_nonempty:
                return False
            if signal == "fail" and verifier:
                return False
            if diff_nonempty and not verifier:
                return True
            return local_status == "pass" or (local_status == "unknown" and protocol_completed)

        @contextmanager
        def scope(phase, subtask, assignment, view, attempt):
            review_dir = None
            before_digest = None
            if phase == "worker":
                started = time.monotonic()
                before = services.resources()
                verify_start = time.monotonic()
                quality_before = _quality(services)
                pre_verifier_seconds = time.monotonic() - verify_start
                active.clear()
                active.update(role_id=subtask.role.role_id)
                try:
                    services._repro_v1_cache = None
                except Exception:
                    pass
                tools = role_tool_contract(subtask.role)
                topics = tuple(getattr(subtask, "topics", ()) or ())
                last.clear()
                last.update(started=started, resources_before=before, quality_before=quality_before,
                    pre_verifier_seconds=pre_verifier_seconds, node_id=subtask.id,
                    role_id=subtask.role.role_id, role_goal=subtask.goal, attempt=attempt,
                    tool_index_before=getattr(services, "tools", None), local_contract_status="unknown",
                    topics=topics, role_tools=tools)
                try:
                    services.role_tools = tools
                except Exception:
                    pass
                root = getattr(getattr(services, "workspace", None), "root", None)
                if role_is_review(tools) and root is not None:
                    try:
                        review_dir = tempfile.TemporaryDirectory(prefix="alo-review-")
                        snap = Path(review_dir.name) / "snapshot"
                        snapshot_workspace(root, snap)
                        services.review_snapshot = snap
                        before_digest = tree_digest(root)
                        last["review_isolation"] = "readonly_pre_review_snapshot"
                        last["pre_review_digest"] = before_digest
                    except (OSError, PilotStopped):
                        if review_dir is not None:
                            review_dir.cleanup()
                        review_dir = None
                        try:
                            services.review_snapshot = None
                        except Exception:
                            pass
                        last["review_isolation"] = "review_snapshot_unavailable"
                elif role_is_review(tools):
                    try:
                        services.review_snapshot = None
                    except Exception:
                        pass
                    last["review_isolation"] = "review_snapshot_unavailable"
                else:
                    try:
                        services.review_snapshot = None
                    except Exception:
                        pass
                    last["review_isolation"] = "implementation_workspace"
            services.event("native_alo_context", {"phase": phase,
                "node": subtask.id if subtask else None, "role_id": subtask.role.role_id if subtask else None,
                "model": assignment.model_id if assignment else planner_model,
                "view_keys": sorted(view) if view else [], "attempt": attempt})

            def _release_review():
                mutated = False
                if before_digest is not None:
                    root = getattr(getattr(services, "workspace", None), "root", None)
                    after_digest = tree_digest(root) if root is not None else None
                    last["implementation_unchanged"] = after_digest == before_digest
                    mutated = after_digest != before_digest
                try:
                    services.role_tools = None
                    services.review_snapshot = None
                except Exception:
                    pass
                if review_dir is not None:
                    review_dir.cleanup()
                return mutated

            try:
                yield
            except BaseException:
                _release_review()
                raise
            else:
                if _release_review():
                    raise PilotStopped("review_mutated_implementation")

        class BudgetedRemediation:
            include_prior_output = True

            def decide(self, ctx):
                remaining = {}
                try:
                    remaining = services.budget_remaining(last.get("model")) or {}
                except Exception:
                    remaining = {}
                budget_left = (isinstance(remaining, dict)
                               and all(isinstance(remaining.get(key), (int, float)) and remaining.get(key) > 0
                                       for key in ("requests", "cny")))
                pref = adapter.config.get("user_preference")
                quality_fail = last.get("public_signal") == "fail"
                empty_diff = not last.get("diff_nonempty")
                needs = (quality_fail or empty_diff) if isinstance(pref, dict) else quality_fail
                record = {"node_id": last.get("node_id"), "attempt": ctx.attempt,
                          "public_signal": last.get("public_signal"), "budget_left": budget_left,
                          "local_contract_status": last.get("local_contract_status"),
                          "tried_models": list(ctx.tried_models), "empty_diff": empty_diff,
                          "preference_applied": False, "escalated_to_strong": False,
                          "counterfactual": None}
                if adapter.config.get("routing_mode") == "fixed":
                    record.update(action="stop", reason="fixed_routing_no_reassign")
                    services.event("alo_decision", record)
                    adapter.state.setdefault("decision_records", []).append(record)
                    return Decision("stop", reason="fixed_routing_no_reassign")
                if not needs or not budget_left:
                    reason = "budget_exhausted" if needs and not budget_left else "public_check_not_failed"
                    if isinstance(pref, dict) and empty_diff and not quality_fail and budget_left:
                        reason = "empty_diff"
                    if not needs:
                        reason = "public_check_not_failed" if not empty_diff or not isinstance(pref, dict) else "empty_diff"
                    if needs and not budget_left:
                        reason = "budget_exhausted"
                    record.update(action="stop", reason=reason)
                    services.event("alo_decision", record)
                    adapter.state.setdefault("decision_records", []).append(record)
                    return Decision("stop", reason=reason)
                untried = [m for m in ctx.eligible_models if m not in ctx.tried_models]
                if not untried:
                    record.update(action="stop", reason="all_eligible_models_tried")
                    services.event("alo_decision", record)
                    adapter.state.setdefault("decision_records", []).append(record)
                    return Decision("stop", reason="all_eligible_models_tried")
                counterfactual, trace = policy.select(org, pool, untried, "remediation", (), "default-user", rng)
                target = counterfactual
                escalated = False
                strong = pref["strong"] if isinstance(pref, dict) else None
                strong_score = trace.get("scores", {}).get(strong) if strong else None
                if isinstance(strong_score, dict) and strong_score.get("feasible"):
                    target = strong
                    escalated = True
                record.update(action="reassign", reason="escalate_to_strong" if escalated else "constrained_ts_reassignment",
                              target_model=target, counterfactual=counterfactual,
                              escalated_to_strong=escalated, selected=target, trace=trace)
                services.event("alo_decision", record)
                services.event("alo_remediation_selection", trace)
                adapter.state.setdefault("decision_records", []).append(record)
                return Decision("reassign", target_model=target, reason=record["reason"])

        pool = {model: Bridge(model) for model in models}
        planner = Bridge(planner_model, planner=True)
        pool[planner.info.id] = planner
        try:
            if self.config.get("pb_mode"):
                task = json.dumps(stage, ensure_ascii=False) + "\n" + (self.config.get("pb_prompt") or "")
                fixed_dag = self.config.get("pb_dag")
            else:
                task = json.dumps(stage, ensure_ascii=False) + (
                    "\n" + WORK_ORDER + " "
                    "Use exactly two DAG nodes named s1 and s2. s1 localizes and edits. "
                    "s2 depends on s1 and verifies with public commands. "
                    "Every node must attempt concrete workspace work. Use task_description, "
                    "goal and persistent_workspace scopes, and choose topics from coding, implementation, "
                    "testing, validation. All roles share public tools; do not request credentials.")
                if self.config.get("criterion_id") == REPRO_CRITERION_ID:
                    task += REPRO_INSTRUCTION
                fixed_dag = FIXED_S1_S2_DAG if self.config.get("fixed_topology") else None
            result = run_task(org, pool, task, planner_id=planner.info.id, rng=rng,
                max_attempts=self.config["max_attempts"], criteria_fn=criterion, assignment_policy=assignment,
                remediation=BudgetedRemediation(), execution_scope=scope,
                extra_state={"task_description": task, "persistent_workspace": "Inspect your own files using public tools."},
                event_fn=lambda event, data: services.event("alo_" + event, data),
                include_synthetic_context=False, fixed_dag=fixed_dag)
            self.state["last_native_run"] = asdict(result)
            self.state["last_native_run"]["boolean_success_semantics"] = "protocol_flow_or_reliable_local_contract_only"
            self.state["last_native_run"]["native_cost_counter_authoritative"] = False
            return {"status": "completed" if result.overall_success else "failed",
                    "summary": "native ALO protocol/local-contract flow; unknown is excluded from capability learning",
                    "private_acceptance": "independent_final_evaluator_only"}
        finally:
            if probe:
                belief_rows = org.conn.execute("SELECT count(*) FROM belief").fetchone()[0]
                tms_rows = org.conn.execute("SELECT count(*) FROM tms_outcomes").fetchone()[0]
            org_assignments = {role: asdict(value) for role, value in org._assignments.items()}
            org_rng = rng.getstate()
            org.conn.close()
            if probe:
                # Learning fields stay as they were. The probe's own transcript is
                # kept beside them so an interrupted probe can be read later.
                probe_transcripts = {
                    "worker_transcripts": copy.deepcopy(self.state.get("worker_transcripts", {})),
                    "planner_messages": copy.deepcopy(self.state.get("planner_messages", [])),
                }
                mutated_observations = self.state["observations"] != observations_before
                mutated_source = _db_digest(source_db) != source_digest
                self.state["observations"] = retained["observations"]
                self.state["worker_transcripts"] = retained["worker_transcripts"]
                self.state["planner_messages"] = retained["planner_messages"]
                for key in ("assignments", "rng_state", "last_native_run", "last_criterion_statuses"):
                    if retained[key] is None:
                        self.state.pop(key, None)
                    else:
                        self.state[key] = retained[key]
                self.state["probe_transcripts"] = probe_transcripts
                try:
                    if mutated_observations:
                        raise PilotStopped("probe_updated_policy_observations")
                    if mutated_source:
                        raise PilotStopped("probe_mutated_source_organization")
                    self.state["probe_isolation"] = {
                        "prior": "cold" if self.config["cold_prior"] else "learned_terminal",
                        "belief_rows": belief_rows, "tms_rows": tms_rows,
                        "source_unchanged": True, "organization_updates": False,
                    }
                finally:
                    probe_dir.cleanup()
            else:
                self.state["assignments"] = org_assignments
                self.state["rng_state"] = org_rng


def adapter_for(group, config=None):
    config = dict(config or {})
    if group == "G1":
        # Pass G1_MAIN_MODEL or G1_STRONG_MODEL explicitly. Pool position is not a role.
        if not config.get("fixed_model"):
            raise PilotStopped("g1_fixed_model_required")
        return FixedSingleAgent(**config)
    if group == "G2":
        if not config.get("planner_model"):
            raise PilotStopped("planner_model_required")
        return HeterogeneousALO(**config)
    if group == "G3":
        if not config.get("role_models") or not config.get("manager_model"):
            raise PilotStopped("groupchat_role_and_manager_binding_required")
        return HeterogeneousGroupChat(**config)
    raise PilotStopped("unknown_trial_group")
