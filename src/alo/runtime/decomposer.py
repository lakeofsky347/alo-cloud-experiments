"""Task Decomposer + Role Generator - T1/T2.  Formal model §3.

The planner model emits a subtask DAG with one Role contract per node (YAML).
Every contract passes RoleContract validation (§1.4); the DAG passes a DFS
cycle check. Any parse/validation/cycle failure falls back to a single
generalist role - decomposition must never crash a run (robustness gate).
"""
from __future__ import annotations

from dataclasses import dataclass

import yaml

from ..models.base import ModelAdapter
from ..org.authority import GOVERNANCE_GATED, AuthorityError, validate as validate_authority
from ..org.role import ContractError, RoleContract

# Must stay inside the M2 topic allowlist. "general" is not transferable.
FALLBACK_TOPIC = "generic"

DECOMPOSE_MARKER = "ALO_DECOMPOSE"

DECOMPOSE_PROMPT = """ALO_DECOMPOSE
You are the Task Decomposer of an Adaptive LLM Organization (formal model T1/T2).
Decompose the task into a subtask DAG and generate one Role contract per subtask.
Output ONLY YAML in exactly this shape:

subtasks:
  - id: s1
    goal: <what this subtask achieves>
    deps: []                 # ids of prerequisite subtasks
    topics: [<TMS directory topics, e.g. auth, frontend>]
    role:
      capability_reqs: [<capability labels>]   # goal is taken from the subtask, do not repeat it
      epistemic_scope:
        include: [<information sections this role may see>]
        exclude: [billing, credentials]
      authority: [inspect]   # subset of inspect|modify|delegate|approve|merge|deploy|spend
      tools: [<tool names>]
      output_schema: <artifact kind>
      success_criteria: non_empty_output

Task: {task}
"""


@dataclass(frozen=True)
class Subtask:
    id: str
    goal: str
    deps: tuple[str, ...]
    topics: tuple[str, ...]
    role: RoleContract


class DecompositionError(ValueError):
    pass


def _strip_fences(text: str) -> str:
    """Real LLMs love wrapping YAML in ``` fences; peel them off."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:]  # drop ```yaml
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


def parse_dag(text: str) -> list[Subtask]:
    """Parse + validate the planner's YAML into a Subtask DAG."""
    data = yaml.safe_load(_strip_fences(text))
    if not isinstance(data, dict) or not data.get("subtasks"):
        raise DecompositionError("missing 'subtasks' list")
    if not isinstance(data["subtasks"], list):
        raise DecompositionError("subtasks must be a list")
    subtasks: dict[str, Subtask] = {}
    for raw in data["subtasks"]:
        sid = str(raw["id"])
        if sid in subtasks:
            raise DecompositionError(f"duplicate subtask id: {sid}")
        deps = raw.get("deps", [])
        if not isinstance(deps, list) or len(set(deps)) != len(deps):
            raise DecompositionError(f"invalid or duplicate dependencies for {sid}")
        topics = raw.get("topics", [])
        if not isinstance(topics, list) or any(not isinstance(t, str) or not t.strip() for t in topics):
            raise DecompositionError(f"topics must be a list of nonempty strings for {sid}")
        # goal lives at subtask level; inject into the contract (§1.4: r.goal
        # comes from the decomposition DAG node).
        role_data = dict(raw["role"])
        role_data.setdefault("goal", str(raw["goal"]))
        role = RoleContract.from_dict(sid, role_data)
        try:
            validate_authority(set(role.authority))
        except AuthorityError as error:
            raise DecompositionError(str(error)) from error
        if set(role.authority) & set(GOVERNANCE_GATED):
            raise DecompositionError(
                f"{sid} requests governance-gated authority {sorted(set(role.authority) & set(GOVERNANCE_GATED))}"
            )
        subtasks[sid] = Subtask(
            id=sid,
            goal=str(raw["goal"]),
            deps=tuple(str(d) for d in raw.get("deps", [])),
            topics=tuple(dict.fromkeys(t.strip() for t in topics)),
            role=role,
        )
    _check_dag(subtasks)
    return list(subtasks.values())


def _check_dag(subtasks: dict[str, Subtask]) -> None:
    for s in subtasks.values():
        for d in s.deps:
            if d not in subtasks:
                raise DecompositionError(f"{s.id} depends on unknown subtask {d!r}")
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(subtasks, WHITE)

    def visit(node: str) -> None:
        color[node] = GRAY
        for dep in subtasks[node].deps:
            if color[dep] == GRAY:
                raise DecompositionError(f"dependency cycle at {node!r}")
            if color[dep] == WHITE:
                visit(dep)
        color[node] = BLACK

    for sid in subtasks:
        if color[sid] == WHITE:
            visit(sid)


def topo_sort(subtasks: list[Subtask]) -> list[Subtask]:
    """Kahn's algorithm; input is guaranteed acyclic by _check_dag."""
    by_id = {s.id: s for s in subtasks}
    indegree = {s.id: len(s.deps) for s in subtasks}
    ready = sorted([sid for sid, deg in indegree.items() if deg == 0])
    order: list[Subtask] = []
    while ready:
        sid = ready.pop(0)
        order.append(by_id[sid])
        for s in subtasks:
            if sid in s.deps:
                indegree[s.id] -= 1
                if indegree[s.id] == 0:
                    ready.append(s.id)
        ready.sort()
    return order


def fallback_subtasks(task: str) -> list[Subtask]:
    """Single generalist role - decomposition must never crash a run."""
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
    return [Subtask(id="generalist", goal=task, deps=(), topics=(FALLBACK_TOPIC,), role=role)]


def decompose(adapter: ModelAdapter, task: str, *, on_completion=None) -> tuple[list[Subtask], bool]:
    """T1/T2 entry. Returns (subtasks, used_fallback)."""
    resp = adapter.complete(DECOMPOSE_PROMPT.format(task=task))
    if on_completion is not None:
        on_completion(resp)
    try:
        return parse_dag(resp.text), False
    except (DecompositionError, ContractError, AuthorityError, yaml.YAMLError, KeyError, TypeError, AttributeError):
        return fallback_subtasks(task), True
