"""Permission - runtime ABAC access decision.  Formal model §1.7.

permit(subject_attrs, action, resource_attrs, env_state) -> allow|deny

Roles are generated dynamically, so a predefined role set cannot exist;
RBAC is inapplicable by construction. Policy is governed centrally
(governance layer writes; agents read-only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Resource classes that are denied unless explicitly listed in the role's
# epistemic_scope.include (least-privilege default, §1.5/§1.7).
SENSITIVE_CLASSES: frozenset[str] = frozenset({"billing", "customer_pii", "user_pii_raw", "credentials"})


@dataclass(frozen=True)
class Subject:
    identity_id: str
    role_id: str
    authority: frozenset[str]
    epistemic_include: frozenset[str]
    violations: int = 0


@dataclass(frozen=True)
class Resource:
    path: str
    classification: str = "general"  # general|billing|customer_pii|credentials|...


@dataclass
class Policy:
    """Central-governed policy. deny_classes is the org-wide sensitive set."""
    deny_classes: frozenset[str] = field(default_factory=lambda: SENSITIVE_CLASSES)
    max_violations: int = 3


def permit(subject: Subject, action: str, resource: Resource, policy: Policy) -> bool:
    # Repeat offenders are frozen out regardless of other attributes.
    if subject.violations >= policy.max_violations:
        return False
    # Sensitive resource: only readable when the role's epistemic scope
    # explicitly includes that class (compiled into the view, §1.5).
    if resource.classification in policy.deny_classes:
        if resource.classification not in subject.epistemic_include:
            return False
    # Scope and authority are independent requirements; unknown actions deny.
    required = {"read": "inspect", "inspect": "inspect", "modify": "modify",
                "merge": "merge", "deploy": "deploy", "spend": "spend",
                "delegate": "delegate", "approve": "approve"}.get(action)
    return required is not None and required in subject.authority
