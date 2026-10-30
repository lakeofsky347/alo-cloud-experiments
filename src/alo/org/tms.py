"""Machine TMS - transactive memory as organizational state.  Formal model §1.9.

TMS := (Dir, Prov), implementing Wegner's three processes:
- directory updating: objective outcomes after execution
- retrieval coordination: opt-in AssignmentPolicy queries model topic history
- information allocation: scope-filtered DAG artifacts (no cross-session retrieval)

Legacy cumulative expertise remains available for inspection. Routing uses
success AND failure counts, not raw exposure-biased success totals.
"""
from __future__ import annotations

import sqlite3


class TMS:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # --- directory updating ---
    def record_expertise(self, subject_type: str, subject_id: str, topic: str, delta: float = 1.0) -> None:
        if subject_type not in ("identity", "model"):
            raise ValueError("subject_type must be identity|model")
        self.conn.execute(
            "INSERT INTO tms_dir (subject_type, subject_id, topic, expertise) VALUES (?,?,?,?) "
            "ON CONFLICT(subject_type, subject_id, topic) DO UPDATE SET expertise = expertise + excluded.expertise",
            (subject_type, subject_id, topic, delta),
        )
        self.conn.commit()

    # --- retrieval coordination ---
    def record_outcome(self, model_id: str, topics: tuple[str, ...], success: bool) -> None:
        """One objective observation per distinct topic of a completed node."""
        self.conn.executemany(
            "INSERT INTO tms_outcomes (model_id, topic, successes, failures) VALUES (?,?,?,?) "
            "ON CONFLICT(model_id, topic) DO UPDATE SET "
            "successes=successes+excluded.successes, failures=failures+excluded.failures",
            [(model_id, t, int(success), int(not success)) for t in sorted(set(topics))],
        )
        self.conn.commit()

    def model_history(self, model_ids: list[str], topics: tuple[str, ...]) -> dict[str, dict]:
        """Restrict retrieval to eligible models; identities never crowd them out.

        Average per-topic Beta(1,1) means to avoid counting overlapping topics
        as independent observations. These are descriptive routing statistics,
        not an additional capability term in the assignment utility.
        """
        result = {}
        for mid in model_ids:
            evidence = []
            for topic in sorted(set(topics)):
                row = self.conn.execute(
                    "SELECT successes, failures FROM tms_outcomes WHERE model_id=? AND topic=?",
                    (mid, topic),
                ).fetchone()
                s, f = (row['successes'], row['failures']) if row else (0, 0)
                evidence.append({'topic': topic, 'successes': s, 'failures': f,
                                 'mean': (s + 1) / (s + f + 2)})
            result[mid] = {'topics': evidence,
                           'cold': not evidence or any(e['successes'] + e['failures'] == 0 for e in evidence),
                           'score': sum(e['mean'] for e in evidence) / len(evidence) if evidence else .5}
        return result

    def who_knows(self, topic: str, limit: int = 5) -> list[tuple[str, str, float]]:
        rows = self.conn.execute(
            "SELECT subject_type, subject_id, expertise FROM tms_dir "
            "WHERE topic = ? ORDER BY expertise DESC LIMIT ?",
            (topic, limit),
        ).fetchall()
        return [(r["subject_type"], r["subject_id"], r["expertise"]) for r in rows]

    # --- provenance (credibility support) ---
    def register_artifact(self, artifact: str, identity_id: str, role_id: str, model_id: str) -> None:
        self.conn.execute(
            "INSERT INTO tms_prov (artifact, identity_id, role_id, model_id) VALUES (?,?,?,?)",
            (artifact, identity_id, role_id, model_id),
        )
        self.conn.commit()

    def provenance_of(self, artifact: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM tms_prov WHERE artifact = ? ORDER BY id", (artifact,)
        ).fetchall()
