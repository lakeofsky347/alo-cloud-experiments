"""Strict statistical-state transfer for M2; never copy a task database.

The exported object is an allowlisted numerical sufficient-statistic view.
Task identities, assignments, artifacts, conversations and provenance remain
in the evaluation-side audit trail.  Same-task B0/B1 restore is a different
operation and may retain those fields; this module is only for new tasks.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile


SCHEMA = "alo-m2-learning-state-v2"
DEFAULT_TOPICS = (
    "bugfix", "cli", "coding", "debugging", "generic", "implementation",
    "planning", "program-reconstruction", "python", "regex", "sql",
    "testing", "validation", "writing",
)
DEFAULT_CELLS = tuple(
    f"type={kind}|complexity={complexity}|uncertainty={uncertainty}"
    for kind, complexity, uncertainty in itertools.product(
        ("debugging", "writing", "generic"), ("high", "na"), ("high", "na")
    )
)
NUMERIC_OBSERVATIONS = frozenset({
    "public_quality", "quality_before", "quality_after", "delta_quality",
    "credited_quality",
    "tokens", "input_tokens", "output_tokens", "reasoning_tokens", "seconds",
    "verifier_seconds", "cost_cny", "requests",
})
OBSERVATION_FIELDS = NUMERIC_OBSERVATIONS | {"local_contract_status", "topics"}
# Role identity stays in the task audit. It is not a cross-task field.
_DROPPED_OBSERVATION_FIELDS = frozenset({"role_id"})
TABLES = {
    "belief", "preference_obs", "tms_dir", "tms_outcomes", "tms_prov",
    "audit_log", "identities",
}


class StateContractError(ValueError):
    """The proposed cross-task transfer violates the frozen allowlist."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def state_hash(value):
    """Hash the complete envelope, including the independent-audit digest."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def _policy(allowed_models, allowed_topics, allowed_cells):
    models = tuple(sorted(set(allowed_models)))
    topics = tuple(sorted(set(DEFAULT_TOPICS if allowed_topics is None else allowed_topics)))
    cells = tuple(sorted(set(DEFAULT_CELLS if allowed_cells is None else allowed_cells)))
    if not models or any(not isinstance(m, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", m)
                         for m in models):
        raise StateContractError("invalid_frozen_model_pool")
    if any(not isinstance(t, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", t) for t in topics):
        raise StateContractError("invalid_frozen_topic_allowlist")
    if any(not isinstance(c, str) or c not in DEFAULT_CELLS for c in cells):
        raise StateContractError("unregistered_feature_cell")
    return {"model_pool": list(models), "topics": list(topics), "cells": list(cells)}


def _number(value, *, low=0, high=None, integer=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise StateContractError("non_finite_or_non_numeric_statistic")
    if value < low or (high is not None and value > high) or (integer and int(value) != value):
        raise StateContractError("statistic_out_of_range")
    return value


def _observation(row, allowed_topics=None):
    if not isinstance(row, dict) or not row:
        raise StateContractError("non_allowlisted_policy_observation")
    # Drop role identity before the allowlist check so a task-local label
    # cannot cross tasks by riding along in the observation.
    row = {key: value for key, value in row.items() if key not in _DROPPED_OBSERVATION_FIELDS}
    if set(row) - OBSERVATION_FIELDS:
        raise StateContractError("non_allowlisted_policy_observation")
    result = {}
    topics = None
    if "topics" in row:
        raw_topics = row["topics"]
        if not isinstance(raw_topics, (list, tuple)) or any(not isinstance(topic, str) for topic in raw_topics):
            raise StateContractError("invalid_observation_topics")
        allowed = set(DEFAULT_TOPICS if allowed_topics is None else allowed_topics)
        topics = [topic for topic in raw_topics if topic in allowed]
        if len(topics) != len(raw_topics):
            # An unregistered topic would make the credit key match the wrong cell.
            topics = None
    for field, value in row.items():
        if field == "topics":
            continue
        if field == "local_contract_status":
            if value not in ("pass", "fail", "unknown"):
                raise StateContractError("invalid_local_contract_status")
        elif value is not None:
            if field == "delta_quality":
                _number(value, low=-1, high=1)
            elif field in ("public_quality", "quality_before", "quality_after", "credited_quality"):
                _number(value, high=1)
            else:
                _number(value, integer=field in {"tokens", "input_tokens", "output_tokens", "reasoning_tokens", "requests"})
        if field == "credited_quality" and topics is None:
            continue
        result[field] = value
    if topics:
        result["topics"] = list(topics)
    elif "credited_quality" in result and topics is None:
        result.pop("credited_quality", None)
    return result


def _readonly_connection(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise StateContractError("source_organization_database_absent")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not TABLES <= tables:
        conn.close()
        raise StateContractError("unexpected_organization_database_schema")
    return conn


def export_learning_state(db_path, adapter_state, *, allowed_models,
                          evidence_id=None, allowed_topics=None, allowed_cells=None):
    """Export B, TMS model/topic counts and numerical policy observations.

    Only predeclared topic/cell values are transferable.  Unregistered topics
    generated in a task are omitted, so a planner cannot encode task content
    in topic names.  Counts of omitted rows are audit metadata, never evidence
    that an outcome failed.  Unknown numerical observations remain null.
    """
    policy = _policy(allowed_models, allowed_topics, allowed_cells)
    models, topics, cells = map(set, (policy["model_pool"], policy["topics"], policy["cells"]))
    if not isinstance(adapter_state, dict):
        raise StateContractError("adapter_state_must_be_mapping")
    conn = _readonly_connection(db_path)
    try:
        all_belief = [dict(r) for r in conn.execute("SELECT model_id,cell,alpha,beta FROM belief ORDER BY model_id,cell")]
        all_tms = [dict(r) for r in conn.execute("SELECT model_id,topic,successes,failures FROM tms_outcomes ORDER BY model_id,topic")]
    finally:
        conn.close()
    belief = [r for r in all_belief if r["model_id"] in models and r["cell"] in cells]
    tms = [r for r in all_tms if r["model_id"] in models and r["topic"] in topics]
    observations = {}
    source_observations = adapter_state.get("observations", {})
    if not isinstance(source_observations, dict) or set(source_observations) - models:
        raise StateContractError("unregistered_observation_model")
    for model, rows in sorted(source_observations.items()):
        if not isinstance(rows, list) or len(rows) > 100000:
            raise StateContractError("invalid_policy_observation_sequence")
        observations[model] = [_observation(row, topics) for row in rows]
    result = {
        "schema": SCHEMA, "transfer_policy": policy,
        "belief": belief, "tms_outcomes": tms, "observations": observations,
        "audit": {"evidence_sha256": hashlib.sha256(str(evidence_id).encode()).hexdigest()
                  if evidence_id is not None else None,
                  "excluded_belief_rows": len(all_belief) - len(belief),
                  "excluded_tms_rows": len(all_tms) - len(tms)},
    }
    validate_learning_state(result, allowed_models=allowed_models,
                            allowed_topics=allowed_topics, allowed_cells=allowed_cells)
    return result


def validate_learning_state(value, *, allowed_models, allowed_topics=None, allowed_cells=None):
    """Fail closed on additional fields or a different frozen policy."""
    if not isinstance(value, dict) or set(value) != {
        "schema", "transfer_policy", "belief", "tms_outcomes", "observations", "audit"
    } or value["schema"] != SCHEMA:
        raise StateContractError("invalid_learning_state_envelope")
    policy = _policy(allowed_models, allowed_topics, allowed_cells)
    if value["transfer_policy"] != policy:
        raise StateContractError("cross_task_transfer_policy_changed")
    models, topics, cells = map(set, (policy["model_pool"], policy["topics"], policy["cells"]))
    audit = value["audit"]
    if not isinstance(audit, dict) or set(audit) != {"evidence_sha256", "excluded_belief_rows", "excluded_tms_rows"}:
        raise StateContractError("non_allowlisted_audit_metadata")
    digest = audit["evidence_sha256"]
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise StateContractError("invalid_evidence_digest")
    _number(audit["excluded_belief_rows"], integer=True)
    _number(audit["excluded_tms_rows"], integer=True)
    for key, fields in (("belief", {"model_id", "cell", "alpha", "beta"}),
                        ("tms_outcomes", {"model_id", "topic", "successes", "failures"})):
        if not isinstance(value[key], list) or len(value[key]) > 100000:
            raise StateContractError("invalid_statistic_rows")
        seen = set()
        for row in value[key]:
            if not isinstance(row, dict) or set(row) != fields or row["model_id"] not in models:
                raise StateContractError("non_allowlisted_statistic")
            dimension = "cell" if key == "belief" else "topic"
            if row[dimension] not in (cells if key == "belief" else topics):
                raise StateContractError("unregistered_statistic_dimension")
            identity = row["model_id"], row[dimension]
            if identity in seen:
                raise StateContractError("duplicate_statistic_identity")
            seen.add(identity)
            if key == "belief":
                _number(row["alpha"], low=1, integer=True)
                _number(row["beta"], low=1, integer=True)
            else:
                _number(row["successes"], integer=True)
                _number(row["failures"], integer=True)
    if not isinstance(value["observations"], dict) or set(value["observations"]) - models:
        raise StateContractError("unregistered_observation_model")
    for rows in value["observations"].values():
        if not isinstance(rows, list) or len(rows) > 100000:
            raise StateContractError("invalid_policy_observation_sequence")
        for row in rows:
            _observation(row, topics)
    return True


def import_learning_state(db_path, value, *, allowed_models, mode="persist",
                          allowed_topics=None, allowed_cells=None):
    """Initialize a new task's database and return only its policy state.

    ``reset`` creates the common empty prior regardless of the supplied parent.
    An existing populated organization is rejected: same-task restoration and
    cross-task migration must never accidentally target the same database.
    """
    if mode not in {"persist", "reset"}:
        raise StateContractError("unknown_m2_state_condition")
    _policy(allowed_models, allowed_topics, allowed_cells)
    if mode == "persist":
        validate_learning_state(value, allowed_models=allowed_models,
                                allowed_topics=allowed_topics, allowed_cells=allowed_cells)
    from alo.storage.db import connect
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        if any(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in TABLES):
            raise StateContractError("target_database_must_have_empty_common_prior")
        if mode == "persist":
            with conn:
                conn.executemany("INSERT INTO belief(model_id,cell,alpha,beta) VALUES (?,?,?,?)",
                                 [(r["model_id"], r["cell"], r["alpha"], r["beta"]) for r in value["belief"]])
                conn.executemany("INSERT INTO tms_outcomes(model_id,topic,successes,failures) VALUES (?,?,?,?)",
                                 [(r["model_id"], r["topic"], r["successes"], r["failures"]) for r in value["tms_outcomes"]])
                # Directory values are derived from transferred successes,
                # never copied from legacy identity/expertise-only evidence.
                conn.executemany("INSERT INTO tms_dir(subject_type,subject_id,topic,expertise) VALUES ('model',?,?,?)",
                                 [(r["model_id"], r["topic"], r["successes"]) for r in value["tms_outcomes"]])
    finally:
        conn.close()
    return {"observations": copy.deepcopy(value["observations"]) if mode == "persist" else {}}


def readonly_probe_copy(value):
    """Return an isolated deep copy; its observations cannot reach main state."""
    return copy.deepcopy(value)


@contextmanager
def probe_state_context(value, *, allowed_models, allowed_topics=None, allowed_cells=None):
    """A disposable, read-only probe organization plus isolated policy state.

    Probes must use query-only observation logic.  Database writes fail at the
    SQLite boundary, and policy edits affect only the deep copy.  No probe data
    is returned for transfer to the main task stream.
    """
    original_hash = state_hash(value)
    isolated = readonly_probe_copy(value)
    with tempfile.TemporaryDirectory(prefix="alo-m2-probe-") as directory:
        db_path = Path(directory) / "probe.sqlite3"
        adapter_state = import_learning_state(db_path, isolated, allowed_models=allowed_models,
                                              allowed_topics=allowed_topics, allowed_cells=allowed_cells)
        conn = _readonly_connection(db_path)
        try:
            yield conn, adapter_state
        finally:
            conn.close()
            if state_hash(value) != original_hash:
                raise StateContractError("probe_mutated_main_learning_state")
