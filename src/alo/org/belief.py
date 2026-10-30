"""Capability Belief - conditional success-probability belief.  Formal model §1.3.

B(m, x) := P(success | model=m, task_features=x)

v0.1 representation: feature-cell discretization + independent Beta posterior
per (model, cell). Upgrade path (Bayesian logistic / online GLM) noted in
formal model; cells then degrade into a prior.

HARD RULES (§1.3):
- updated ONLY by objective evidence (test pass / compile / verified success);
- subjective feedback is REJECTED here and routed to Preference (§1.8).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


class SubjectiveEvidenceError(ValueError):
    """Raised when subjective feedback is attempted on the belief store."""


def cell_of(task_features: dict[str, str]) -> str:
    """Discretize the task-feature vector into a belief cell key.

    Keep low-dimensional and interpretable (aligned with §2.1).
    """
    keys = ("type", "complexity", "uncertainty")
    return "|".join(f"{k}={task_features.get(k, 'na')}" for k in keys)


@dataclass(frozen=True)
class Posterior:
    mean: float
    var: float
    alpha: float
    beta: float


class BeliefStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def query(self, model_id: str, cell: str) -> Posterior:
        row = self.conn.execute(
            "SELECT alpha, beta FROM belief WHERE model_id = ? AND cell = ?",
            (model_id, cell),
        ).fetchone()
        a, b = (row["alpha"], row["beta"]) if row else (1.0, 1.0)
        n = a + b
        return Posterior(mean=a / n, var=a * b / (n * n * (n + 1)), alpha=a, beta=b)

    def update_objective(self, model_id: str, cell: str, success: bool) -> Posterior:
        """The ONLY write path. Caller guarantees the evidence is objective."""
        p = self.query(model_id, cell)
        a = p.alpha + (1.0 if success else 0.0)
        b = p.beta + (0.0 if success else 1.0)
        self.conn.execute(
            "INSERT INTO belief (model_id, cell, alpha, beta) VALUES (?,?,?,?) "
            "ON CONFLICT(model_id, cell) DO UPDATE SET alpha=excluded.alpha, beta=excluded.beta",
            (model_id, cell, a, b),
        )
        self.conn.commit()
        return self.query(model_id, cell)

    def update_subjective(self, *_args, **_kwargs) -> None:
        raise SubjectiveEvidenceError(
            "subjective feedback must go to PreferenceStore (dual-channel separation, §1.3/§1.8)"
        )

    def dump(self) -> list[dict]:
        """Full posterior table (experiment trajectory persistence)."""
        rows = self.conn.execute(
            "SELECT model_id, cell, alpha, beta FROM belief ORDER BY model_id, cell"
        ).fetchall()
        return [
            {"model": r["model_id"], "cell": r["cell"], "alpha": r["alpha"],
             "beta": r["beta"], "mean": r["alpha"] / (r["alpha"] + r["beta"])}
            for r in rows
        ]
