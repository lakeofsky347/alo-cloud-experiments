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
QUALITY_OBSERVATION_FIELDS = frozenset({
    "public_quality", "quality_before", "quality_after", "delta_quality",
    "credited_quality",
})
# Role topics that exist on both SWE-bench Lite and ProgramBench DAGs.
SHARED_ROLE_TOPICS = ("coding", "implementation", "testing", "validation")
TASK_FAMILIES = ("swe-repair", "program-reconstruction")
TRANSFER_SCHEMA = "alo-cross-track-transfer-v1"
TARGET_SCHEMA = "alo-frozen-transfer-target-v1"
SWE_TRACKS = frozenset({"SWE-bench_Lite", "SWE-bench"})
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


def _model_id_ok(model):
    """Domestic ids and ``fox/...`` internal names. Not a filesystem path."""
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", model):
        return False
    if model.startswith("/") or model.endswith("/") or "//" in model or ".." in model:
        return False
    return True


def _policy(allowed_models, allowed_topics, allowed_cells):
    models = tuple(sorted(set(allowed_models)))
    topics = tuple(sorted(set(DEFAULT_TOPICS if allowed_topics is None else allowed_topics)))
    cells = tuple(sorted(set(DEFAULT_CELLS if allowed_cells is None else allowed_cells)))
    # ``/`` is the fox-pool internal prefix (``fox/kimi-k3.1-pro``). Domestic
    # ids have no slash and still match. The id is a ledger key, not a path.
    if not models or any(not _model_id_ok(m) for m in models):
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


def _without_quality(row):
    """Drop quality fields inside the exporter. Callers cannot leave them in."""
    return {key: value for key, value in row.items() if key not in QUALITY_OBSERVATION_FIELDS}


def export_learning_state(db_path, adapter_state, *, allowed_models,
                          evidence_id=None, allowed_topics=None, allowed_cells=None,
                          strip_quality=False):
    """Export B, TMS model/topic counts and numerical policy observations.

    Only predeclared topic/cell values are transferable.  Unregistered topics
    generated in a task are omitted, so a planner cannot encode task content
    in topic names.  Counts of omitted rows are audit metadata, never evidence
    that an outcome failed.  Unknown numerical observations remain null.

    ``strip_quality=True`` is the resources-only export mode. Quality fields
    are removed here, after validation, so a caller that forgets to delete
    ``credited_quality`` or ``quality_after`` still cannot emit them. Belief
    and TMS rows are quality posteriors, so that mode omits them as well and
    the next task stays on the common prior.
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
    if strip_quality:
        belief = []
        tms = []
    observations = {}
    source_observations = adapter_state.get("observations", {})
    if not isinstance(source_observations, dict) or set(source_observations) - models:
        raise StateContractError("unregistered_observation_model")
    for model, rows in sorted(source_observations.items()):
        if not isinstance(rows, list) or len(rows) > 100000:
            raise StateContractError("invalid_policy_observation_sequence")
        kept = []
        for row in rows:
            cleaned = _observation(row, topics)
            if strip_quality:
                cleaned = _without_quality(cleaned)
                if not cleaned:
                    continue
            kept.append(cleaned)
        observations[model] = kept
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


def transferable_task_type(topics):
    """Shared role topics as the cross-track task type. A cell is not one."""
    from .policy import topic_key
    key = topic_key(topics)
    if not key or any(topic not in SHARED_ROLE_TOPICS for topic in key):
        return None
    return "/".join(key)


def division_key(model, role_topics):
    """Explicit model × role × task-type identity for a transferable row."""
    task_type = transferable_task_type(role_topics)
    if not isinstance(model, str) or not model or task_type is None:
        raise StateContractError("non_transferable_division_key")
    from .policy import topic_key
    role = list(topic_key(role_topics))
    return {"model": model, "role": role, "task_type": task_type}


def export_transfer_package(adapter_state, *, allowed_models, task_family=None,
                            strip_quality=False):
    """Cross-track profile. Default M2 export is a different function.

    The package keeps role topics shared by both benchmarks and an optional
    pre-registered ``task_family``. The ``extract_features`` cell is marked
    non-transferable and is not copied. Rows whose topics are not entirely
    inside the shared set are omitted.
    """
    from .policy import topic_key
    if task_family is not None and task_family not in TASK_FAMILIES:
        raise StateContractError("unregistered_task_family")
    models = set(_policy(allowed_models, None, None)["model_pool"])
    if not isinstance(adapter_state, dict):
        raise StateContractError("adapter_state_must_be_mapping")
    source = adapter_state.get("observations", {})
    if not isinstance(source, dict) or set(source) - models:
        raise StateContractError("unregistered_observation_model")
    profiles = {}
    omitted = 0
    for model, rows in sorted(source.items()):
        if not isinstance(rows, list) or len(rows) > 100000:
            raise StateContractError("invalid_policy_observation_sequence")
        kept = []
        for row in rows:
            if not isinstance(row, dict):
                raise StateContractError("non_allowlisted_policy_observation")
            body = {key: value for key, value in row.items() if key not in {"cell", "role_id"}}
            task_type = transferable_task_type(body.get("topics"))
            if task_type is None:
                omitted += 1
                continue
            cleaned = _observation(body, SHARED_ROLE_TOPICS)
            if strip_quality:
                cleaned = _without_quality(cleaned)
            if "cell" in cleaned:
                raise StateContractError("cell_is_not_transferable")
            role = list(topic_key(body.get("topics")))
            identity = division_key(model, role)
            cleaned["role_topics"] = identity["role"]
            cleaned["task_type"] = identity["task_type"]
            cleaned["topics"] = list(identity["role"])
            kept.append(cleaned)
        profiles[model] = kept
    package = {
        "schema": TRANSFER_SCHEMA,
        "division_key": "model_x_role_x_task_type",
        "task_family": task_family,
        "shared_role_topics": list(SHARED_ROLE_TOPICS),
        "non_transferable": ["cell"],
        "profiles": profiles,
        "audit": {"omitted_nonshared_rows": omitted},
    }
    validate_transfer_package(package, allowed_models=allowed_models)
    return package


def validate_transfer_package(value, *, allowed_models):
    """Fail closed when a cross-track package carries a cell or a private topic."""
    from .policy import topic_key
    if (not isinstance(value, dict) or value.get("schema") != TRANSFER_SCHEMA
            or value.get("division_key") != "model_x_role_x_task_type"
            or value.get("non_transferable") != ["cell"]
            or list(value.get("shared_role_topics") or []) != list(SHARED_ROLE_TOPICS)):
        raise StateContractError("invalid_transfer_package")
    if set(value) != {
        "schema", "division_key", "task_family", "shared_role_topics",
        "non_transferable", "profiles", "audit",
    }:
        raise StateContractError("invalid_transfer_package")
    family = value.get("task_family")
    if family is not None and family not in TASK_FAMILIES:
        raise StateContractError("unregistered_task_family")
    audit = value["audit"]
    if not isinstance(audit, dict) or set(audit) != {"omitted_nonshared_rows"}:
        raise StateContractError("non_allowlisted_audit_metadata")
    _number(audit["omitted_nonshared_rows"], integer=True)
    models = set(_policy(allowed_models, None, None)["model_pool"])
    profiles = value["profiles"]
    if not isinstance(profiles, dict) or set(profiles) - models:
        raise StateContractError("unregistered_observation_model")
    for model, rows in profiles.items():
        if not isinstance(rows, list) or len(rows) > 100000:
            raise StateContractError("invalid_policy_observation_sequence")
        for row in rows:
            if not isinstance(row, dict) or "cell" in row or "role_id" in row:
                raise StateContractError("cell_is_not_transferable")
            role = row.get("role_topics")
            if transferable_task_type(role) != row.get("task_type"):
                raise StateContractError("division_key_mismatch")
            if list(topic_key(row.get("topics"))) != list(role):
                raise StateContractError("division_key_mismatch")
            if division_key(model, role)["task_type"] != row["task_type"]:
                raise StateContractError("division_key_mismatch")
            _observation(
                {key: item for key, item in row.items() if key not in {"role_topics", "task_type"}},
                SHARED_ROLE_TOPICS)
    return True


def reject_cross_benchmark_parent(parent_unit_id, child_track, parent_state=None):
    """Refuse to let a ProgramBench learning state parent an SWE unit."""
    if child_track not in SWE_TRACKS:
        return None
    parts = parent_unit_id.split(".") if isinstance(parent_unit_id, str) else []
    parent_track = parts[1] if len(parts) > 1 else None
    if parent_track == "ProgramBench":
        raise StateContractError("target_learning_state_cannot_parent_swe")
    if isinstance(parent_state, dict):
        if parent_state.get("may_parent_swe") is False or parent_state.get("target_track") == "ProgramBench":
            raise StateContractError("target_learning_state_cannot_parent_swe")
        if parent_state.get("schema") == TARGET_SCHEMA:
            raise StateContractError("target_learning_state_cannot_parent_swe")
        if (parent_state.get("schema") == TRANSFER_SCHEMA
                and parent_state.get("task_family") == "program-reconstruction"):
            raise StateContractError("target_learning_state_cannot_parent_swe")
    return None


def _envelope_from_transfer(package, allowed_models):
    """Policy observations only. Belief cells stay at the common prior."""
    observations = {}
    for model, rows in package["profiles"].items():
        cleaned = []
        for row in rows:
            item = {key: value for key, value in row.items() if key not in {"role_topics", "task_type", "cell"}}
            item["topics"] = list(row["role_topics"])
            cleaned.append(_observation(item, SHARED_ROLE_TOPICS))
        observations[model] = cleaned
    envelope = {
        "schema": SCHEMA,
        "transfer_policy": _policy(allowed_models, None, None),
        "belief": [],
        "tms_outcomes": [],
        "observations": observations,
        "audit": {"evidence_sha256": None, "excluded_belief_rows": 0, "excluded_tms_rows": 0},
    }
    validate_learning_state(envelope, allowed_models=allowed_models)
    return envelope


class FrozenTransferLibrary:
    """Temporary probe library over a frozen transfer file. Writes to the source raise."""

    def __init__(self, source_path, source_bytes, source_digest, observations, package,
                 target_track, connection):
        self.source_path = Path(source_path)
        self.source_bytes = bytes(source_bytes)
        self.source_digest = source_digest
        self.observations = observations
        self.package = package
        self.target_track = target_track
        self.connection = connection
        self.origin_track = "SWE-bench_Lite"

    def write_back(self, *_args, **_kwargs):
        raise StateContractError("frozen_transfer_writeback_forbidden")

    def target_state(self):
        """State of the held-out track. It is not a legal parent of an SWE unit."""
        return {
            "schema": TARGET_SCHEMA,
            "source_track": self.origin_track,
            "target_track": self.target_track,
            "may_parent_swe": False,
            "source_sha256": self.source_digest,
            "division_key": "model_x_role_x_task_type",
            "non_transferable": ["cell"],
            "task_family": self.package.get("task_family"),
            "profiles": copy.deepcopy(self.package.get("profiles")),
            "observations": copy.deepcopy(self.observations),
        }

    def assert_source_unchanged(self):
        current = self.source_path.read_bytes()
        digest = hashlib.sha256(current).hexdigest()
        if current != self.source_bytes or digest != self.source_digest:
            raise StateContractError("frozen_transfer_source_mutated")


@contextmanager
def import_frozen_transfer(source_path, *, target_track, allowed_models):
    """Load a frozen package into the probe temporary library.

    Not M2 ``persist`` mode. The source file's bytes and hash are checked
    around the import. ``write_back`` raises and does not open the source
    for writing. A ProgramBench-targeted package cannot be loaded as an SWE parent.
    """
    if target_track not in {"ProgramBench", "SWE-bench_Lite"}:
        raise StateContractError("unknown_transfer_target")
    source_path = Path(source_path)
    if not source_path.is_file():
        raise StateContractError("frozen_transfer_source_missing")
    source_bytes = source_path.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    try:
        package = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StateContractError("frozen_transfer_source_unreadable") from error
    validate_transfer_package(package, allowed_models=allowed_models)
    if target_track in SWE_TRACKS and package.get("task_family") == "program-reconstruction":
        raise StateContractError("target_learning_state_cannot_parent_swe")
    reject_cross_benchmark_parent("", target_track, package)
    envelope = _envelope_from_transfer(package, allowed_models)
    try:
        with probe_state_context(envelope, allowed_models=allowed_models) as (conn, adapter_state):
            library = FrozenTransferLibrary(
                source_path, source_bytes, source_digest, adapter_state["observations"],
                package, target_track, conn)
            library.assert_source_unchanged()
            yield library
            library.assert_source_unchanged()
    finally:
        current = source_path.read_bytes()
        if current != source_bytes or hashlib.sha256(current).hexdigest() != source_digest:
            raise StateContractError("frozen_transfer_source_mutated")
