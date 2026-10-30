"""Model adapter protocol + ModelInfo.  Formal model §1.2.

A Model is an inference resource carrying ONLY engineering metadata.
Capability information lives exclusively in the Belief store (§1.3);
adapters exposing any capability score would violate the hard boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelInfo:
    """m := (id, provider, family, ctx_window, cost_fn, latency_prior, tool_interface)"""
    id: str
    provider: str
    family: str
    ctx_window: int = 128_000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    latency_prior_ms: int = 0
    tool_interface: str = "text"


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int

    def cost(self, info: ModelInfo) -> float:
        return (
            self.input_tokens / 1000 * info.cost_per_1k_input
            + self.output_tokens / 1000 * info.cost_per_1k_output
        )


class ModelAdapter(Protocol):
    info: ModelInfo

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 2048) -> Completion:
        ...
