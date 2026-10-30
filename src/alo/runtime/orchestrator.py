"""Orchestrator - T0-T9 lifecycle.  Formal model §3.

run_task: full multi-role path
    T0 features -> T1/T2 decompose -> T3 bandit assign -> T4 compile ->
    T5 authority/permission -> T6 execute -> T7 evidence -> T8 update ->
    T9 re-org (widen on missing-context failure, audit-logged).

run_single: single-role convenience path (kept from P2-2.1).
"""
from __future__ import annotations

import hashlib
import random
import re
from contextlib import nullcontext
from dataclasses import dataclass, field

from ..models.base import ModelAdapter
from ..org.belief import cell_of
from ..org.context import Bandwidth, compile_view, widen
from ..org.organization import Organization
from ..org.role import RoleContract
from .bandit import thompson_select
from .bandwidth import choose_bandwidth
from .decomposer import Subtask, decompose, parse_dag, topo_sort
from .governance import Governance
from .remediation import FailureContext

# --- T0: feature extraction (§2.1; heuristic, no model call) ---

_DEBUG_HINTS = ("修复", "fix", "bug", "500", "错误", "报错", "崩溃", "debug")
_WRITE_HINTS = ("撰写", "文档", "draft", "write", "发布说明", "润色", "文案")
_INTERMITTENT_HINTS = ("偶发", "intermittent", "偶尔", "随机出现", "flaky")
_COMPLEX_HINTS = ("架构", "系统", "迁移", "重构", "architect", "migration")


def extract_features(task: str) -> dict[str, str]:
    t = task.lower()
    def matches(hints):
        return any(re.search(r"(?<![a-z0-9_])" + re.escape(h) + r"(?![a-z0-9_])", t)
                   if h.isascii() else h in t for h in hints)
    task_type = (
        "debugging" if matches(_DEBUG_HINTS)
        else "writing" if matches(_WRITE_HINTS)
        else "generic"
    )
    return {
        "type": task_type,
        "complexity": "high" if matches(_COMPLEX_HINTS) else "na",
        "uncertainty": "high" if matches(_INTERMITTENT_HINTS) else "na",
    }


# --- T7: evidence evaluators (machine-checkable success criteria, §1.4) ---

# Bandwidth experiment substrate: section blobs present in global state so
# that BROAD adds raw_logs/env_state and FULL additionally adds full_dump
# (§1.5; lengths differ by design to exercise the noise ground truth).
RAW_LOGS_BLOB = " ".join(["[rawlog] trace line"] * 12)
FULL_DUMP_BLOB = " ".join(["[dump] verbose environment record"] * 60)

def non_empty_criteria(role: RoleContract, output: str) -> bool:
    return bool(output.strip())


def mock_draw_criteria(role: RoleContract, output: str) -> bool:
    """Simulation ground truth: parse the MockAdapter's draw marker."""
    if "draw=fail" in output:
        return False
    return bool(output.strip())


# --- run_task: full DAG path ---


@dataclass(frozen=True)
class NodeResult:
    subtask_id: str
    role_id: str
    identity_id: str
    model_id: str
    bandwidth_final: int
    attempts: int
    success: bool
    belief_mean_after: float
    output_text: str
    selection: dict = field(default_factory=dict)
    # Populated only when a RemediationPolicy is supplied (formal model v0.3).
    model_history: tuple[str, ...] = ()
    actions: tuple[dict, ...] = ()


@dataclass(frozen=True)
class TaskRun:
    task: str
    used_fallback_decomposition: bool
    node_results: tuple[NodeResult, ...]
    overall_success: bool
    cost_usd: float
    skipped_subtasks: tuple[dict, ...] = ()


def run_task(
    org: Organization,
    pool: dict[str, ModelAdapter],
    task: str,
    *,
    user_id: str = "default-user",
    governance: Governance | None = None,
    planner_id: str | None = None,
    criteria_fn=non_empty_criteria,
    max_attempts: int = 2,
    rng: random.Random | None = None,
    selector_fn=None,
    bandwidth_fn=None,
    extra_state: dict | None = None,
    assignment_policy=None,
    fixed_dag: str | None = None,
    view_fn=None,
    retry_fixed_view: bool = False,
    event_fn=None,
    remediation=None,
    execution_scope=None,
) -> TaskRun:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if assignment_policy is not None and selector_fn is not None:
        raise ValueError("choose assignment_policy or selector_fn, not both")
    if remediation is not None and retry_fixed_view:
        raise ValueError("remediation replaces retry_fixed_view; use FixedRetryPolicy")
    # T0
    features = extract_features(task)
    cell = cell_of(features)

    # T1/T2
    planner = pool.get(planner_id) if planner_id else next(iter(pool.values()))
    assert planner is not None, "model pool is empty"
    planner_cost = 0.0
    def account_planner(completion):
        nonlocal planner_cost
        planner_cost = completion.cost(planner.info)
        if governance is not None:
            governance.charge(planner_cost, context="planner")
    if fixed_dag is None:
        with (execution_scope("planner", None, None, None, 0) if execution_scope else nullcontext()):
            subtasks, used_fallback = decompose(planner, task, on_completion=account_planner)
    else:
        # Opt-in frozen experiments: validate the DAG without a planner call.
        subtasks, used_fallback = parse_dag(fixed_dag), False
    order = topo_sort(subtasks)
    if event_fn and execution_scope is not None:
        event_fn('decomposition', {'fallback': used_fallback,
                                   'nodes': [{'id': s.id, 'deps': list(s.deps),
                                              'role_id': s.role.role_id} for s in order]})

    # Worker candidates exclude the planner (orchestration is not a worker role).
    worker_ids = [mid for mid in pool if mid != planner.info.id] or list(pool)

    # Global state: shared sections + bandwidth-differentiated sections
    # (§1.5 experiment substrate) + artifacts written as nodes complete.
    # TMS information allocation (§1.9): downstream roles only see an
    # artifact if their epistemic_scope includes it.
    global_state: dict[str, object] = {
        "goal": task,
        "constraints": [],
        "current_phase": "execute",
        "team_structure": [s.id for s in order],
        "execution_history": [],
        "raw_logs": RAW_LOGS_BLOB,
        "env_state": {"budget": "ok", "region": "prod"},
        "full_dump": FULL_DUMP_BLOB,
    }
    if extra_state:
        global_state.update(extra_state)  # experiment-controlled sections (§1.5)

    results: list[NodeResult] = []
    skipped: list[dict] = []
    unsuccessful: set[str] = set()
    total_cost = planner_cost

    # Widen is part of the ADAPTIVE treatment; arms that pin a bandwidth
    # (fixed-bandwidth baselines) must not widen (§5.2 experimental design).
    allow_widen = bandwidth_fn is None

    for sub in order:
        blocked_by = sorted(set(sub.deps) & unsuccessful)
        if blocked_by:
            unsuccessful.add(sub.id)
            detail = {'subtask': sub.id, 'blocked_by': blocked_by, 'reason': 'unsuccessful_dependency'}
            skipped.append(detail)
            (governance or Governance(org.conn)).audit('dependency_skipped', detail)
            if event_fn:
                event_fn('node_skipped', detail)
            continue
        result, cost = _run_node(
            org, pool, worker_ids, sub, features, cell, global_state,
            governance=governance, criteria_fn=criteria_fn,
            max_attempts=max_attempts, rng=rng,
            selector_fn=selector_fn, bandwidth_fn=bandwidth_fn,
            allow_widen=allow_widen,
            assignment_policy=assignment_policy, user_id=user_id,
            view_fn=view_fn, retry_fixed_view=retry_fixed_view, event_fn=event_fn,
            remediation=remediation,
            execution_scope=execution_scope,
        )
        total_cost += cost
        results.append(result)
        if event_fn:
            event_fn('node_finished', {'node_id': sub.id, 'success': result.success,
                                       'attempts': result.attempts})
        if not result.success:
            unsuccessful.add(sub.id)
        global_state[f"artifact:{sub.id}"] = result.output_text
        history = global_state.get("execution_history")
        if isinstance(history, list):
            history.append({"subtask": sub.id, "model": result.model_id, "success": result.success})

    overall = not skipped and all(r.success for r in results)
    if governance is not None:
        governance.audit("task_complete", {"task": task[:80], "overall_success": overall,
                                           "nodes": len(results), "skipped": len(skipped), "fallback": used_fallback})
    return TaskRun(
        task=task,
        used_fallback_decomposition=used_fallback,
        node_results=tuple(results),
        overall_success=overall,
        cost_usd=total_cost,
        skipped_subtasks=tuple(skipped),
    )


def _run_node(
    org: Organization,
    pool: dict[str, ModelAdapter],
    worker_ids: list[str],
    sub: Subtask,
    features: dict[str, str],
    cell: str,
    global_state: dict[str, object],
    *,
    governance: Governance | None,
    criteria_fn,
    max_attempts: int,
    rng: random.Random | None,
    selector_fn=None,
    bandwidth_fn=None,
    allow_widen: bool = True,
    assignment_policy=None,
    user_id: str = "default-user",
    view_fn=None,
    retry_fixed_view: bool = False,
    event_fn=None,
    remediation=None,
    execution_scope=None,
) -> tuple[NodeResult, float]:
    role = sub.role
    select = selector_fn or thompson_select
    choose_bw = bandwidth_fn or choose_bandwidth

    # T3: model assignment (default: Thompson over the belief posterior)
    selection = {}
    if assignment_policy is not None:
        model_id, selection = assignment_policy.select(org, pool, worker_ids, cell, sub.topics, user_id, rng)
    else:
        model_id = select(org.belief, worker_ids, cell, rng) if len(worker_ids) > 1 else worker_ids[0]
    if model_id not in worker_ids:
        raise ValueError(f"selector returned ineligible model: {model_id}")
    if selection:
        (governance or Governance(org.conn)).audit('assignment_policy', {'subtask': sub.id, **selection})
    adapter = pool[model_id]

    identity = org.spawn_identity()
    bandwidth = choose_bw(features)  # T4 initial (adaptive arm by default)
    org.assign(role, identity, model_id, bandwidth=int(bandwidth))  # audits (invariant 4)

    # T5-T7 with T9 widen loop (or an explicit RemediationPolicy)
    output_text, success, attempts, cost = "", False, 0, 0.0
    tried = [model_id]
    actions: list[dict] = []
    handoff: str | None = None
    model_first_attempt = 1
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        view = view_fn(global_state, sub, bandwidth) if view_fn else compile_view(global_state, role, bandwidth)
        prompt = role.to_prompt_header() + _render_view(view) + _render_handoff(handoff) + f"\n# Task\n{sub.goal}\n"
        # The optional integration boundary receives the actual native role,
        # assignment and compiled view for every attempt. It cannot substitute
        # a role inferred from the prompt, and exits even on model/tool errors.
        assignment = org.assignment_of(role.role_id)
        if event_fn and execution_scope is not None:
            event_fn('execution_context', {'node_id': sub.id, 'attempt_id': attempt,
                                           'role_id': role.role_id, 'identity_id': assignment.identity_id,
                                           'model_id': assignment.model_id, 'bandwidth': assignment.bandwidth,
                                           'authority': list(assignment.authority),
                                           'view_keys': sorted(view)})
        with (execution_scope("worker", sub, assignment, view, attempt) if execution_scope else nullcontext()):
            completion = adapter.complete(prompt)
        cost += completion.cost(adapter.info)
        if governance is not None:
            governance.charge(completion.cost(adapter.info), context=f"node:{sub.id}")
        output_text = completion.text
        success = criteria_fn(role, output_text)
        if event_fn:
            event_fn('criterion_result', {'node_id': sub.id, 'attempt_id': attempt,
                                          'passed': success})
        if success:
            break
        if attempt < max_attempts and remediation is not None:
            ctx = FailureContext(
                subtask_id=sub.id, attempt=attempt, max_attempts=max_attempts, model_id=model_id,
                bandwidth=int(bandwidth), eligible_models=tuple(worker_ids), tried_models=tuple(tried),
                output_sha256=hashlib.sha256(output_text.encode()).hexdigest(), allow_widen=allow_widen)
            decision = remediation.decide(ctx)
            record = {"subtask": sub.id, "attempt": attempt, "action": decision.action,
                      "from_model": model_id, "target_model": decision.target_model,
                      "bandwidth": int(bandwidth), "reason": decision.reason,
                      "output_sha256": ctx.output_sha256}
            actions.append(record)
            (governance or Governance(org.conn)).audit("remediation_decision", record)
            if event_fn and execution_scope is not None:
                event_fn('remediation_decision', record)
            if decision.action == "stop":
                break
            if decision.action == "retry":
                continue
            if decision.action == "widen":
                if not allow_widen:
                    raise ValueError("remediation policy requested widen in a fixed-bandwidth arm")
                new_bw = widen(bandwidth)
                if new_bw == bandwidth:
                    break
                if governance is not None:
                    governance.audit("reorg_widen", {"subtask": sub.id, "from": int(bandwidth),
                                                     "to": int(new_bw), "attempt": attempt})
                bandwidth = new_bw
                org.assign(role, identity, model_id, bandwidth=int(bandwidth))
                continue
            target = decision.target_model
            if target not in worker_ids or target == model_id:
                raise ValueError(f"reassign target must be another eligible model: {target!r}")
            # Formal model v0.3: one objective update per (node, model).
            _record_model_outcome(org, selector_fn, model_id, cell, sub.topics, False)
            if governance is not None:
                governance.audit("evidence", {"subtask": sub.id, "model": model_id,
                                              "objective_success": False,
                                              "attempts": attempt - model_first_attempt + 1,
                                              "superseded_by": target})
            (governance or Governance(org.conn)).audit("reorg_reassign", {
                "subtask": sub.id, "from": model_id, "to": target, "attempt": attempt,
                "handoff": {"prior_output_sha256": ctx.output_sha256, "bandwidth": int(bandwidth),
                            "prior_output_included": bool(getattr(remediation, "include_prior_output", True))}})
            handoff = output_text if getattr(remediation, "include_prior_output", True) else None
            model_id = target
            adapter = pool[model_id]
            tried.append(model_id)
            model_first_attempt = attempt + 1
            org.assign(role, identity, model_id, bandwidth=int(bandwidth))
            continue
        if attempt < max_attempts:
            if retry_fixed_view:
                continue
            if not allow_widen:
                break  # fixed-bandwidth baseline arm: no T9 widen treatment
            new_bw = widen(bandwidth)  # T9 re-org operator (§3)
            if new_bw == bandwidth:
                break  # already FULL; other operators land with reassign (v0.2)
            if governance is not None:
                governance.audit("reorg_widen", {"subtask": sub.id, "from": int(bandwidth),
                                                 "to": int(new_bw), "attempt": attempt})
            bandwidth = new_bw
            org.assign(role, identity, model_id, bandwidth=int(bandwidth))

    # T8: belief (objective ONLY), TMS directory + provenance
    # (Kept inline, in the original write order, so the default path is unchanged.)
    posterior = org.belief.update_objective(model_id, cell, success)
    # Selector learning hook (e.g. LinUCB arm update; Thompson ignores it -
    # the belief store IS its state).
    update_hook = getattr(selector_fn, "update_arm", None)
    if callable(update_hook):
        update_hook(model_id, cell, 1.0 if success else 0.0)
    org.tms.record_outcome(model_id, sub.topics, success)
    for topic in set(sub.topics):
        org.tms.record_expertise("identity", identity.id, topic, delta=1.0 if success else 0.0)
        org.tms.record_expertise("model", model_id, topic, delta=1.0 if success else 0.0)
    org.tms.register_artifact(f"artifact:{sub.id}", identity.id, role.role_id, model_id)
    if governance is not None:
        governance.audit("evidence", {"subtask": sub.id, "model": model_id,
                                      "objective_success": success, "attempts": attempts})
    return NodeResult(
        subtask_id=sub.id,
        role_id=role.role_id,
        identity_id=identity.id,
        model_id=model_id,
        bandwidth_final=int(bandwidth),
        attempts=attempts,
        success=success,
        belief_mean_after=posterior.mean,
        output_text=output_text,
        selection=selection,
        model_history=tuple(tried) if remediation is not None else (),
        actions=tuple(actions),
    ), cost


def _record_model_outcome(org, selector_fn, model_id, cell, topics, success):
    """Objective evidence for a model superseded by reassign: Belief, selector hook, TMS."""
    posterior = org.belief.update_objective(model_id, cell, success)
    update_hook = getattr(selector_fn, "update_arm", None)
    if callable(update_hook):
        update_hook(model_id, cell, 1.0 if success else 0.0)
    org.tms.record_outcome(model_id, topics, success)
    for topic in set(topics):
        org.tms.record_expertise("model", model_id, topic, delta=1.0 if success else 0.0)
    return posterior


def _render_handoff(handoff: str | None) -> str:
    if handoff is None:
        return ""
    return ("\n# Handoff\nA previous worker's output for this subtask failed acceptance. "
            "It is provided for reference only; return a complete replacement.\n"
            f"{handoff}\n")


def _render_view(view: dict[str, object]) -> str:
    lines = ["\n# Compiled View (role-specific, minimal sufficient information)"]
    for key, value in view.items():
        lines.append(f"## {key}\n{value}")
    return "\n".join(lines) + "\n"


# --- run_single: P2-2.1 convenience path (kept for CLI default) ---


@dataclass(frozen=True)
class RunResult:
    task: str
    role_id: str
    identity_id: str
    model_id: str
    output_text: str
    objective_success: bool
    belief_mean_after: float
    cost_usd: float


def run_single(
    org: Organization,
    adapter: ModelAdapter,
    task: str,
    *,
    user_id: str = "default-user",
    governance: Governance | None = None,
    criteria_fn=non_empty_criteria,
) -> RunResult:
    features = extract_features(task)
    cell = cell_of(features)
    role = RoleContract.from_dict(
        "generalist",
        {
            "goal": task,
            "capability_reqs": ["general_reasoning"],
            "epistemic_scope": {"include": ["task_description"], "exclude": ["billing", "credentials"]},
            "authority": ["inspect"],
            "tools": [],
            "output_schema": "free_text",
            "success_criteria": "non_empty_output",
        },
    )
    identity = org.spawn_identity()
    assignment = org.assign(role, identity, adapter.info.id, bandwidth=int(Bandwidth.NARROW))
    global_state = {"goal": task, "constraints": [], "task_description": task, "billing": "MUST-NOT-LEAK"}
    view = compile_view(global_state, role, Bandwidth(assignment.bandwidth))
    assert "billing" not in view, "epistemic_scope.exclude failed - leak in compiled view"
    completion = adapter.complete(role.to_prompt_header() + "\n# Task\n" + task)
    cost = completion.cost(adapter.info)
    if governance is not None:
        governance.charge(cost, context=f"run:{role.role_id}")
    objective_success = criteria_fn(role, completion.text)
    posterior = org.belief.update_objective(adapter.info.id, cell, objective_success)
    org.tms.record_outcome(adapter.info.id, (features["type"],), objective_success)
    org.tms.record_expertise("identity", identity.id, features["type"], delta=1.0 if objective_success else 0.0)
    org.tms.register_artifact(f"run:{task[:40]}", identity.id, role.role_id, adapter.info.id)
    if governance is not None:
        governance.audit("evidence", {"role": role.role_id, "model": adapter.info.id,
                                      "objective_success": objective_success})
    return RunResult(
        task=task, role_id=role.role_id, identity_id=identity.id, model_id=adapter.info.id,
        output_text=completion.text, objective_success=objective_success,
        belief_mean_after=posterior.mean, cost_usd=cost,
    )
