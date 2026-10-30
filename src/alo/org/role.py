"""Role - dynamically generated formal contract.  Formal model §1.4.

A Role is NOT a prompt template. It is a validated contract:
    r := (goal, capability_reqs, epistemic_scope, authority, tools,
          output_schema, success_criteria)
"""
from __future__ import annotations

from dataclasses import dataclass, field

REQUIRED_FIELDS = (
    "goal",
    "capability_reqs",
    "epistemic_scope",
    "authority",
    "tools",
    "output_schema",
    "success_criteria",
)


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class RoleContract:
    role_id: str
    goal: str
    capability_reqs: tuple[str, ...]
    epistemic_scope: dict  # {"include": [...], "exclude": [...]}
    authority: frozenset[str]
    tools: tuple[str, ...]
    output_schema: str
    success_criteria: str

    @staticmethod
    def from_dict(role_id: str, d: dict) -> "RoleContract":
        missing = [k for k in REQUIRED_FIELDS if k not in d]
        if missing:
            raise ContractError(f"role {role_id!r} missing fields: {missing}")
        scope = d["epistemic_scope"]
        if not isinstance(scope, dict) or "include" not in scope or "exclude" not in scope:
            raise ContractError("epistemic_scope must have include/exclude lists")
        return RoleContract(
            role_id=role_id,
            goal=str(d["goal"]),
            capability_reqs=tuple(d["capability_reqs"]),
            epistemic_scope={"include": list(scope["include"]), "exclude": list(scope["exclude"])},
            authority=frozenset(d["authority"]),
            tools=tuple(d["tools"]),
            output_schema=str(d["output_schema"]),
            success_criteria=str(d["success_criteria"]),
        )

    def to_prompt_header(self) -> str:
        """Render the contract as the agent-facing header (view header, §1.5)."""
        return (
            f"# Role Contract: {self.role_id}\n"
            f"goal: {self.goal}\n"
            f"capability_requirements: {', '.join(self.capability_reqs)}\n"
            f"authority: {', '.join(sorted(self.authority)) or '(none)'}\n"
            f"tools: {', '.join(self.tools)}\n"
            f"output_schema: {self.output_schema}\n"
            f"success_criteria: {self.success_criteria}\n"
        )
