"""Authority - contract-level decision rights.  Formal model §1.6.

Authority answers "what may this role decide". It is granted in the Role
contract and adjusted by T9 re-organization operators (heterarchy).
Every actual operation still passes Permission (§1.7) at runtime.
"""
from __future__ import annotations

A_OPS: frozenset[str] = frozenset(
    {"inspect", "modify", "delegate", "approve", "merge", "deploy", "spend"}
)

# Operations that require governance approval before transfer (§1.6).
GOVERNANCE_GATED: frozenset[str] = frozenset({"deploy", "spend"})


class AuthorityError(ValueError):
    pass


def validate(ops: set[str] | frozenset[str]) -> frozenset[str]:
    bad = set(ops) - set(A_OPS)
    if bad:
        raise AuthorityError(f"unknown authority ops: {sorted(bad)}")
    return frozenset(ops)
