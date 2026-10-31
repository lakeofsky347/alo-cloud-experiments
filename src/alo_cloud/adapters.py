"""Native adapters for revised M1/M2 with dynamic official model identities.

The core ALO runtime currently accepts only Boolean criteria. Instance-local
store wrappers gate its learning writes using a separate three-valued public
local contract. An unknown contract can acknowledge protocol submission and
allow the DAG to continue, but never becomes a B/TMS success or failure.
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
from .errors import PilotStopped
from .public_tools import ACTION_INSTRUCTIONS, PublicToolCodeExecutor
from .native_bridge import ROLE_INSTRUCTIONS, SHARED_INSTRUCTIONS, _terminated
from .policy import ConstrainedThompsonPolicy, POLICY_VERSION
from .tool_loop import TOOL_LOOP_VERSION, persistent_tool_loop

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


class FixedSingleAgent:
    """One frozen model with a checkpoint-owned transcript and public tools."""

    def __init__(self, *, fixed_model=None, max_turns=48):
        if type(max_turns) is not int or max_turns < 1:
            raise PilotStopped("invalid_single_turn_limit")
        self.fixed_model, self.max_turns, self.messages = fixed_model, max_turns, []

    def dump_state(self):
        return {"schema": "dualbench-single-v2", "fixed_model": self.fixed_model,
                "max_turns": self.max_turns, "tool_loop_version": TOOL_LOOP_VERSION,
                "messages": copy.deepcopy(self.messages)}

    def load_state(self, state):
        if (state.get("schema") != "dualbench-single-v2" or state.get("max_turns") != self.max_turns
                or state.get("tool_loop_version") != TOOL_LOOP_VERSION
                or (self.fixed_model is not None and state.get("fixed_model") != self.fixed_model)):
            raise PilotStopped("single_checkpoint_configuration_changed")
        self.fixed_model = state["fixed_model"]
        self.messages = copy.deepcopy(state["messages"])

    def run_stage(self, stage, services):
        models = _models(services)
        if not self.fixed_model:
            raise PilotStopped("fixed_model_required")
        if self.fixed_model not in models:
            raise PilotStopped("fixed_model_outside_common_pool")
        return persistent_tool_loop(BoundModel(services, self.fixed_model),
            json.dumps(stage, ensure_ascii=False), self.messages, max_turns=self.max_turns)


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
        for agent in agents:
            agent.register_model_client(model_client_cls=client)
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
                 probe=False, cold_prior=False):
        if (type(max_turns) is not int or max_turns < 1
                or type(max_attempts) is not int or max_attempts < 1):
            raise PilotStopped("invalid_alo_attempt_or_turn_limit")
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
                       "policy_version": POLICY_VERSION,
                       "tool_loop_version": TOOL_LOOP_VERSION}
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
        learn = lambda: self.config["organization_updates"] and last.get("local_contract_status") in {"pass", "fail"}
        org.belief, org.tms = _BeliefGate(org.belief, learn), _TMSGate(org.tms, learn)
        for role, assignment in self.state.get("assignments", {}).items():
            org._assignments[role] = Assignment(**assignment)
        policy = ConstrainedThompsonPolicy(self.state.setdefault("observations", {}),
            services.budget_remaining, priors=self.config["policy_priors"],
            safety_multiplier=self.config["safety_multiplier"])

        class Bridge:
            def __init__(self, model, *, planner=False):
                self.model, self.planner = model, planner
                # Provider identity is supplied by the gateway; no substring
                # inference is used for dispatch or credential selection.
                self.info = ModelInfo("planner::" + model if planner else model, "official", model)

            def complete(self, prompt, *, system="", max_tokens=4096):
                if self.planner:
                    messages = adapter.state.setdefault("planner_messages", [])
                    if not messages:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": prompt})
                    reply = services.complete_for_model(self.model, messages, max_tokens=max_tokens)
                    messages.append({"role": "assistant", "content": reply["text"]})
                    return Completion(reply["text"], reply["input_tokens"], reply["output_tokens"], reply["latency_ms"])
                messages = adapter.state.setdefault("worker_transcripts", {}).setdefault(active["role_id"], [])
                result = persistent_tool_loop(BoundModel(services, self.model), prompt, messages,
                    system=system, max_turns=adapter.config["max_turns"])
                last.update(model=self.model, submission_status=result["status"],
                            tool_index_after=getattr(services, "tools", None))
                return Completion(result["summary"], 0, 0, 0)

        def criterion(role, output):
            verifier_start = time.monotonic()
            quality_after = _quality(services)
            context = {key: last.get(key) for key in ("node_id", "role_id", "role_goal", "attempt",
                "tool_index_before", "tool_index_after", "submission_status")}
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
            after = services.resources()
            before = last["resources_before"]
            input_tokens = _delta(after, before, "observed_input_tokens")
            output_tokens = _delta(after, before, "observed_output_tokens")
            observation = {"public_quality": quality_after, "quality_before": last["quality_before"],
                "quality_after": quality_after,
                "delta_quality": quality_after - last["quality_before"] if quality_after is not None and last["quality_before"] is not None else None,
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "tokens": input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None,
                "reasoning_tokens": _delta(after, before, "observed_reasoning_tokens"),
                "requests": _delta(after, before, "model_requests"),
                "cost_cny": _delta(after, before, "estimated_cost_cny"),
                "seconds": time.monotonic() - last["started"],
                "verifier_seconds": last["pre_verifier_seconds"] + time.monotonic() - verifier_start,
                "local_contract_status": local_status}
            if adapter.config["organization_updates"]:
                adapter.state["observations"].setdefault(last["model"], []).append(observation)
            services.event("alo_observation", {"model": last["model"], "context": context,
                "local_evidence": evidence, "learning_update_allowed": learn(),
                "global_quality_scope": "resulting_workspace_not_role_local_capability", **observation})
            # Core Boolean success describes execution/contract flow. The
            # instance-local stores above separately gate objective learning.
            return local_status == "pass" or (local_status == "unknown" and last.get("submission_status") == "completed")

        @contextmanager
        def scope(phase, subtask, assignment, view, attempt):
            if phase == "worker":
                started = time.monotonic()
                before = services.resources()
                verify_start = time.monotonic()
                quality_before = _quality(services)
                pre_verifier_seconds = time.monotonic() - verify_start
                active.clear()
                active.update(role_id=subtask.role.role_id)
                last.clear()
                last.update(started=started, resources_before=before, quality_before=quality_before,
                    pre_verifier_seconds=pre_verifier_seconds, node_id=subtask.id,
                    role_id=subtask.role.role_id, role_goal=subtask.goal, attempt=attempt,
                    tool_index_before=getattr(services, "tools", None), local_contract_status="unknown")
            services.event("native_alo_context", {"phase": phase,
                "node": subtask.id if subtask else None, "role_id": subtask.role.role_id if subtask else None,
                "model": assignment.model_id if assignment else planner_model,
                "view_keys": sorted(view) if view else [], "attempt": attempt})
            yield

        class BudgetedRemediation:
            include_prior_output = True

            def decide(self, ctx):
                if last.get("local_contract_status") == "unknown":
                    return Decision("stop", reason="unknown_local_contract_no_failure_reassignment")
                untried = [m for m in ctx.eligible_models if m not in ctx.tried_models]
                if not untried:
                    return Decision("stop", reason="all_eligible_models_tried")
                # Same exception as first assignment and later nodes. run.py
                # stops only this unit for unit_resource_exhausted, and halts
                # the schedule for a real CNY-cap failure.
                target, trace = policy.select(org, pool, untried, "remediation", (), "default-user", rng)
                services.event("alo_remediation_selection", trace)
                return Decision("reassign", target_model=target, reason="constrained_ts_reassignment")

        pool = {model: Bridge(model) for model in models}
        planner = Bridge(planner_model, planner=True)
        pool[planner.info.id] = planner
        try:
            task = json.dumps(stage, ensure_ascii=False) + (
                "\nUse at most two implementation/validation DAG nodes. Every node must attempt "
                "concrete workspace work and explicit public syntax/build/tests. Use task_description, "
                "goal and persistent_workspace scopes, and choose topics from coding, implementation, "
                "testing, validation. All roles share public tools; do not request credentials.")
            result = run_task(org, pool, task, planner_id=planner.info.id, rng=rng,
                max_attempts=self.config["max_attempts"], criteria_fn=criterion, assignment_policy=policy,
                remediation=BudgetedRemediation(), execution_scope=scope,
                extra_state={"task_description": task, "persistent_workspace": "Inspect your own files using public tools."},
                event_fn=lambda event, data: services.event("alo_" + event, data),
                include_synthetic_context=False)
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
                # Probe transcripts, assignments and RNG stay on the disposable
                # organization. Checkpoint state keeps only the pre-probe learning
                # fields plus an audit record; the caller sees run_stage's return.
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
