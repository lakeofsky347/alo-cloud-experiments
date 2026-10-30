"""Organization - the full organizational state O_t.  Formal model §1.9.

O_t := (I, M, B, Pi_u, TMS, R_active, assignments, governance)

Invariants enforced here:
1. Identities only ever accumulate (history traceable);
2. B updated only by e^obj, Pi_u only by e^sub (dual-channel separation);
3. governance is written by the governance layer only;
4. every assignments change emits an audit-log entry.
"""
from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass

from ..storage.db import connect
from .belief import BeliefStore
from .identity import Identity
from .preference import PreferenceStore
from .role import RoleContract
from .tms import TMS
from .authority import AuthorityError, GOVERNANCE_GATED


@dataclass(frozen=True)
class Assignment:
    """assignments entry: role -> (identity, model, view-bandwidth, authority)."""
    role_id: str
    identity_id: str
    model_id: str
    bandwidth: int
    authority: tuple[str, ...]


class Organization:
    def __init__(self, db_path: str = ":memory:"):
        self.conn: sqlite3.Connection = connect(db_path)
        self.belief = BeliefStore(self.conn)
        self.preference = PreferenceStore(self.conn)
        self.tms = TMS(self.conn)
        self._assignments: dict[str, Assignment] = {}

    def spawn_identity(self, prefix: str = "worker") -> Identity:
        return Identity.spawn(self.conn, prefix)

    def record_feedback(self, user_id: str, task_cell: str, model_id: str,
                        signal: str, detail: str = "") -> None:
        """Explicit subjective observation; never updates objective or TMS evidence."""
        self.preference.record(user_id, task_cell, model_id, signal, detail)
        from ..runtime.governance import Governance
        Governance(self.conn).audit('preference_feedback', {
            'user_id': user_id, 'cell': task_cell, 'model': model_id, 'signal': signal,
        })

    def assign(self, role: RoleContract, identity: Identity, model_id: str, bandwidth: int,
               *, approved_authority: frozenset[str] = frozenset()) -> Assignment:
        denied = (set(role.authority) & GOVERNANCE_GATED) - approved_authority
        if denied:
            self.conn.execute("INSERT INTO audit_log (event_type, payload) VALUES (?, ?)",
                              ("authority_denied", json.dumps({"role": role.role_id, "ops": sorted(denied)})))
            self.conn.commit()
            raise AuthorityError(f"governance approval required for: {sorted(denied)}")
        a = Assignment(
            role_id=role.role_id,
            identity_id=identity.id,
            model_id=model_id,
            bandwidth=bandwidth,
            authority=tuple(sorted(role.authority)),
        )
        self._assignments[role.role_id] = a
        # Invariant 4: audit every assignment change (payload written by caller's governance in full runtime;
        # here we emit the minimal entry so the invariant can never be skipped).
        self.conn.execute(
            "INSERT INTO audit_log (event_type, payload) VALUES (?, ?)",
            ("assign", json.dumps({"role": a.role_id, "identity": a.identity_id,
                                   "model": a.model_id, "bandwidth": a.bandwidth,
                                   "authority": a.authority,
                                   "approved_authority": sorted(approved_authority)})),
        )
        self.conn.commit()
        return a

    def assignment_of(self, role_id: str) -> Assignment | None:
        return self._assignments.get(role_id)
