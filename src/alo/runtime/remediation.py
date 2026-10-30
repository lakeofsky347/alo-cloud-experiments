"""T9 remediation policies: what to do after a failed node attempt.  Formal model v0.3 §3.

The orchestrator asks a RemediationPolicy for one Decision after each failed
attempt that still has budget left. Policies only choose; the orchestrator
validates, audits and applies the action. Leaving remediation=None keeps the
original widen loop unchanged (default for every frozen experiment).

Actions
  retry     same model, same bandwidth
  widen     same model, next bandwidth tier (forbidden in fixed-bandwidth arms)
  reassign  same role and identity, another eligible model; the failed model
            receives one objective failure for this node
  stop      end the node with its current (failed) result
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..org.context import Bandwidth

ACTIONS = ("retry", "widen", "reassign", "stop")


@dataclass(frozen=True)
class FailureContext:
    subtask_id: str
    attempt: int
    max_attempts: int
    model_id: str
    bandwidth: int
    eligible_models: tuple[str, ...]
    tried_models: tuple[str, ...]
    output_sha256: str
    allow_widen: bool


@dataclass(frozen=True)
class Decision:
    action: str
    target_model: str | None = None
    reason: str = ""

    def __post_init__(self):
        if self.action not in ACTIONS:
            raise ValueError(f"unknown remediation action: {self.action!r}")
        if (self.action == "reassign") != (self.target_model is not None):
            raise ValueError("target_model is required for reassign and only for reassign")


class RemediationPolicy(Protocol):
    def decide(self, ctx: FailureContext) -> Decision: ...


def _widen_or_stop(ctx: FailureContext, reason: str) -> Decision:
    if not ctx.allow_widen:
        return Decision("stop", reason=f"{reason}; widen disabled in this arm")
    if ctx.bandwidth >= int(Bandwidth.FULL):
        return Decision("stop", reason=f"{reason}; already FULL")
    return Decision("widen", reason=reason)


@dataclass(frozen=True)
class WidenPolicy:
    """Explicit form of the original T9 loop (widen until FULL, then stop)."""

    def decide(self, ctx: FailureContext) -> Decision:
        return _widen_or_stop(ctx, "widen_policy")


@dataclass(frozen=True)
class FixedRetryPolicy:
    """Retry with the same model and view until attempts run out."""

    def decide(self, ctx: FailureContext) -> Decision:
        return Decision("retry", reason="fixed_retry")


@dataclass(frozen=True)
class ReassignPolicy:
    """Hand the node to the first untried eligible model in 'order'.

    Models absent from 'order' follow in pool order. When every eligible
    model has been tried, fall back to widen (if allowed) or stop.
    include_prior_output controls whether the failed output is handed over.
    """
    order: tuple[str, ...] = ()
    include_prior_output: bool = True

    def decide(self, ctx: FailureContext) -> Decision:
        ranked = [m for m in self.order if m in ctx.eligible_models]
        ranked += [m for m in ctx.eligible_models if m not in ranked]
        for model in ranked:
            if model not in ctx.tried_models:
                return Decision("reassign", target_model=model, reason="reassign_policy")
        return _widen_or_stop(ctx, "no untried model")
