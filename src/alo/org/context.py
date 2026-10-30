"""Context - role-specific information view.  Formal model §1.5.

v := compile(S_global, r, b)

Compile goal (minimal sufficient information):
    minimize   |v|
    subject to P(success | r, v) >= theta_r

Bandwidth b is the core P3 experimental variable (U-shape hypothesis H3).
"""
from __future__ import annotations

from enum import IntEnum

from .role import RoleContract
from .permission import Policy, Resource, Subject, permit


class Bandwidth(IntEnum):
    NARROW = 0
    WIDE = 1
    BROAD = 2
    FULL = 3


# Sections of global state, ordered from most-shared to most-private.
# FULL sees everything; NARROW sees only the role-scoped projection.
SHARED_SECTIONS = ("goal", "constraints", "interface_contracts", "current_phase", "team_structure")
WIDE_EXTRA = ("execution_history", "related_artifacts")
BROAD_EXTRA = ("raw_logs", "env_state")


def compile_view(
    global_state: dict[str, object],
    role: RoleContract,
    bandwidth: Bandwidth,
) -> dict[str, object]:
    """Project global_state into the role's view.

    - epistemic_scope.exclude always wins (leak prevention, §1.5).
    - NARROW: shared sections + include-listed sections.
    - WIDE:   + execution history & related artifacts.
    - BROAD:  + raw logs & env state.
    - FULL:   everything except excludes (baseline arm for P3).
    """
    include = set(role.epistemic_scope["include"])
    exclude = set(role.epistemic_scope["exclude"])
    subject = Subject("context-compiler", role.role_id, frozenset(role.authority), frozenset(include))
    policy = Policy()

    view: dict[str, object] = {}
    for key, value in global_state.items():
        if key in exclude:
            continue
        if not permit(subject, "read", Resource(key, classification=key), policy):
            continue
        if bandwidth == Bandwidth.FULL:
            view[key] = value
        elif key in SHARED_SECTIONS or key in include:
            view[key] = value
        elif bandwidth >= Bandwidth.WIDE and key in WIDE_EXTRA:
            view[key] = value
        elif bandwidth >= Bandwidth.BROAD and key in BROAD_EXTRA:
            view[key] = value
    return view


def widen(bandwidth: Bandwidth, steps: int = 1) -> Bandwidth:
    """T9 re-org operator `widen`: raise bandwidth after a missing-context failure."""
    return Bandwidth(min(int(bandwidth) + steps, int(Bandwidth.FULL)))
