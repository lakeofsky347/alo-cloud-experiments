"""Pure study contracts and global ordering; no task data or execution entry point."""
from __future__ import annotations

from dataclasses import dataclass


MODEL_POOL = ("deepseek-flash", "mimo-v2.6-flash", "glm-5.3-flash", "mimo-v2.6-pro")
PER_MODEL_LIMIT_MICROCNY = 50_000_000


@dataclass(frozen=True)
class TaskIdentity:
    """A task identity without its public source, hidden tests or filesystem paths."""
    track: str
    task_id: str

    def __post_init__(self):
        if self.track not in {"SWE-bench_Lite", "ProgramBench"} or not self.task_id or ":" in self.task_id:
            raise ValueError("invalid_task_identity")


@dataclass(frozen=True)
class Unit:
    """One planned run; phase/position are not new independent task samples."""
    task: TaskIdentity
    condition: str
    phase: str
    order: tuple[str, ...]


def m1_units(tasks, conditions):
    """Rotate within each track, all B0 before own B1 in the same group order.

    This creates a schedule only. It does not instantiate candidates, restore
    checkpoints, send model requests, or relax a complete formal freeze gate.
    """
    conditions = tuple(conditions)
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("unique_nonempty_conditions_required")
    positions, seen, units = {}, set(), []
    for task in tasks:
        if not isinstance(task, TaskIdentity) or task in seen:
            raise ValueError("unique_typed_task_identities_required")
        seen.add(task)
        index = positions.get(task.track, 0)
        positions[task.track] = index + 1
        offset = index % len(conditions)
        order = conditions[offset:] + conditions[:offset]
        for phase in ("B0", "B1-self"):
            units.extend(Unit(task, condition, phase, order) for condition in order)
    return tuple(units)


def m2_arm_order(first_arm, position):
    """Preserve the frozen first arm, alternating execution order by position."""
    if first_arm not in {"persist", "reset"} or type(position) is not int or position < 0:
        raise ValueError("invalid_m2_arm_order")
    order = (first_arm, "reset" if first_arm == "persist" else "persist")
    return order[::-1] if position % 2 else order


def m2_stage_order():
    """All learning streams precede once-only references, then terminal probes."""
    return ("all_learning_streams", "stateless_references", "terminal_no_update_probes", "independent_scoring")
