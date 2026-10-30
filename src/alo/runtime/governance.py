"""Governance - centralized audit + budget + conflict queue.  Formal model §1.9.

Centralized governance over heterogeneous execution: security policy, audit
log, budget, conflict resolution. Execution layer reads; only this layer
writes governance state (invariant 3).
"""
from __future__ import annotations

import json
import sqlite3


class BudgetExceeded(RuntimeError):
    pass


class Governance:
    def __init__(self, conn: sqlite3.Connection, budget_usd: float = 5.0):
        self.conn = conn
        self.budget_usd = budget_usd
        self.spent_usd = 0.0

    def audit(self, event_type: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO audit_log (event_type, payload) VALUES (?, ?)",
            (event_type, json.dumps(payload, ensure_ascii=False)),
        )
        self.conn.commit()

    def charge(self, cost_usd: float, context: str) -> None:
        """Governance-layer budget cap (P1 risk mitigation: runaway API cost)."""
        if self.spent_usd + cost_usd > self.budget_usd:
            self.audit("budget_exceeded", {"context": context, "attempted": cost_usd})
            raise BudgetExceeded(
                f"budget cap ${self.budget_usd:.2f} would be exceeded "
                f"(spent ${self.spent_usd:.4f} + ${cost_usd:.4f}); context: {context}"
            )
        self.spent_usd += cost_usd
        self.audit("charge", {"cost": cost_usd, "spent_total": self.spent_usd, "context": context})

    def audit_trail(self, limit: int = 100) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
