"""SQLite (WAL) storage for organizational state.

Formal model ref: §1.9 Organization - all persistent components of O_t
(Belief, Preference, TMS Dir/Prov, audit log) live here.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode = WAL;

-- Capability Belief: Beta posterior per (model, feature cell).  §1.3
-- HARD RULE: updated ONLY by objective evidence.
CREATE TABLE IF NOT EXISTS belief (
    model_id TEXT NOT NULL,
    cell     TEXT NOT NULL,
    alpha    REAL NOT NULL DEFAULT 1.0,
    beta     REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (model_id, cell)
);

-- Preference observations: append-only subjective feedback.  §1.8
-- HARD RULE: rows here NEVER flow into `belief`.
CREATE TABLE IF NOT EXISTS preference_obs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    task_cell  TEXT NOT NULL,
    model_id   TEXT NOT NULL,
    signal     TEXT NOT NULL,  -- accept|edit|retry|reject|override
    detail     TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Machine TMS directory: who knows what.  §1.9 (Wegner: directory updating)
CREATE TABLE IF NOT EXISTS tms_dir (
    subject_type TEXT NOT NULL,  -- identity|model
    subject_id   TEXT NOT NULL,
    topic        TEXT NOT NULL,
    expertise    REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (subject_type, subject_id, topic)
);

-- TMS provenance: artifact -> (identity, role, model, ts).  §1.9
-- Routing evidence counts both successes and failures. Legacy cumulative
-- expertise cannot be migrated: it does not retain failed exposure counts.
CREATE TABLE IF NOT EXISTS tms_outcomes (
    model_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    successes INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (model_id, topic)
);

CREATE TABLE IF NOT EXISTS tms_prov (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact    TEXT NOT NULL,
    identity_id TEXT NOT NULL,
    role_id     TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Governance audit log: every assignments/authority/permit event.  §1.9 invariant 4
CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Identities: persistent agent instances.  §1.1
CREATE TABLE IF NOT EXISTS identities (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    retired    INTEGER NOT NULL DEFAULT 0
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn
