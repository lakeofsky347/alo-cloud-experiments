"""Identity - persistent agent instance anchor.  Formal model §1.1."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Identity:
    id: str

    @staticmethod
    def spawn(conn: sqlite3.Connection, prefix: str = "worker") -> "Identity":
        # Rows are never deleted (retire only flags), so COUNT(*) is a
        # monotonic sequence across processes - an in-memory counter would
        # collide with persisted identities on the next CLI invocation.
        n = conn.execute("SELECT COUNT(*) AS n FROM identities").fetchone()["n"]
        iid = f"{prefix}-{n + 1}"
        conn.execute("INSERT INTO identities (id) VALUES (?)", (iid,))
        conn.commit()
        return Identity(id=iid)

    def retire(self, conn: sqlite3.Connection) -> None:
        # Invariant: history is preserved; only a flag flips.
        conn.execute("UPDATE identities SET retired = 1 WHERE id = ?", (self.id,))
        conn.commit()
