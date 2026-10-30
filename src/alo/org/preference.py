"""Preference - user-level conditional preference observations.  Formal model §1.8.

Updated ONLY by subjective feedback (override / reject / retry / edit / accept).
Never flows into BeliefStore. "User swapped Claude->GPT" licenses exactly one
conclusion: this user prefers GPT on this task kind. Nothing about objective
capability.
"""
from __future__ import annotations

import sqlite3

SIGNALS = ("accept", "edit", "retry", "reject", "override")


class PreferenceStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def record(self, user_id: str, task_cell: str, model_id: str, signal: str, detail: str = "") -> None:
        if signal not in SIGNALS:
            raise ValueError(f"unknown preference signal: {signal!r}; expected one of {SIGNALS}")
        self.conn.execute(
            "INSERT INTO preference_obs (user_id, task_cell, model_id, signal, detail) VALUES (?,?,?,?,?)",
            (user_id, task_cell, model_id, signal, detail),
        )
        self.conn.commit()

    def approval_rate(self, user_id: str, task_cell: str, model_id: str) -> float | None:
        """Empirical weighted approval rate; no prior or Bayesian calibration."""
        rows = self.conn.execute(
            "SELECT signal, COUNT(*) AS n FROM preference_obs "
            "WHERE user_id = ? AND task_cell = ? AND model_id = ? GROUP BY signal",
            (user_id, task_cell, model_id),
        ).fetchall()
        if not rows:
            return None
        score = sum({"accept": 1.0, "edit": 0.5, "retry": 0.25, "reject": 0.0, "override": 0.25}[r["signal"]] * r["n"] for r in rows)
        total = sum(r["n"] for r in rows)
        return score / total
