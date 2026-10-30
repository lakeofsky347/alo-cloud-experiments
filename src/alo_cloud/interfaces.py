"""Explicit production integration boundaries; nothing sends requests or runs code.

Adapters accept caller-provided services. A real implementation must independently
enforce permissions, cumulative accounting, deadlines and evaluation isolation.
The protocols describe requirements, not a working provider or sandbox.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .errors import IntegrationNotConfigured


class StudyServices(Protocol):
    """One task's public tools and metered model service, shared across groups."""
    state_dir: Path
    available_models: tuple[str, ...]
    tools: int

    def complete_for_model(self, model, messages, *, max_tokens=4096): ...
    def budget_remaining(self, model=None): ...
    def resources(self): ...
    def public_quality(self): ...
    def local_criterion(self, *, role_id, model_id, output, context): ...
    def event(self, event, payload): ...
    def list_files(self): ...
    def read_file(self, path): ...
    def write_file(self, path, content): ...
    def search(self, text): ...
    def command(self, argv, timeout_s=30): ...
    def execute_code(self, language, code, timeout_s=30): ...


class ProviderTransport:
    """No HTTP client, environment key reader, or default provider is installed."""

    def complete_for_model(self, model, messages, *, max_tokens=4096):
        raise IntegrationNotConfigured("provider_transport_not_implemented")


class SandboxBackend:
    """Candidate commands must eventually use an independently accepted sandbox."""

    def command(self, argv, timeout_s=30):
        raise IntegrationNotConfigured("candidate_sandbox_not_implemented")


class BenchmarkEvaluator:
    """An independent evaluator will consume sealed artifacts without model keys."""

    def score_sealed(self, candidate, task_contract):
        raise IntegrationNotConfigured("independent_benchmark_evaluator_not_implemented")
