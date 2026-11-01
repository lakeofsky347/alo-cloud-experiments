"""M1/M2 paid-run controller.

``--dry-run`` uses a fake sender and never opens a socket. ``--live`` is the
only path that reads ``DEEPSEEK_API_KEY``, ``MIMO_API_KEY`` and ``ZAI_API_KEY``
and permits network sends. The ledger file is created with zero opening
commitments because this checkout records no numeric historical spend; a
resumed process reopens the same file and does not clear it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tarfile

from .adapters import adapter_for
from .budget import MICROCNY_PER_CNY, BudgetLedger, BudgetStopped
from .collect import collect_sealed
from .contracts import (
    G1_MAIN_MODEL, G1_STRONG_MODEL, MODEL_POOL, PER_MODEL_LIMIT_MICROCNY, STUDY_TOTAL_LIMIT_MICROCNY,
)
from .errors import PilotStopped
from .freeze import (
    BLOCKED_FRACTION_MIN_UNITS, G2_PLANNER_MODEL, G3_MANAGER_MODEL, G3_ROLES,
    MAX_BLOCKED_FRACTION, MAX_UNKNOWN_OUTCOMES, filter_benchmarks, filter_shard,
    requested_benchmarks, select_units,
)
from .diagnostics import write_diagnostic
from .ledger import (
    current_epoch, current_outcome_epoch, infra_generations, restart_counts, restart_unit,
    track_generations,
)
from .programbench_prompt import PROGRAMBENCH_BUDGET_NOTICE, PROGRAMBENCH_SUBMISSION_CONTRACT
from .progress import publish_timeout_counts
from .images import resolve_task_image
from .prices import ceiling_for_messages, charge_microcny
from .public_tools import transport_summary
from .sandbox import SWE_TESTBED_PATH, DockerSandbox
from .workspace_prep import (
    apply_patch, release_unreadable_executable, resolve_image_head, seed_image_tree,
)
from .services import PROTOCOL_UNIT_LIMITS, StudyServices
from .shardlock import clear_drain, drain_requested, exclusive_lock, halt_record, set_halt
from .state import export_learning_state, import_learning_state


KEY_ENVS = ("DEEPSEEK_API_KEY", "MIMO_API_KEY", "ZAI_API_KEY")
FORBIDDEN_PUBLIC_KEYS = frozenset({
    "patch", "test_patch", "fail_to_pass", "pass_to_pass", "gold_patch", "reference_patch",
    "test_content", "test_blob", "expected_output", "hints_text_hidden",
})
DRY_PLAN = """subtasks:
  - id: s1
    goal: Inspect the workspace and submit a short public note.
    deps: []
    topics: [coding]
    role:
      capability_reqs: [implement]
      epistemic_scope:
        include: [task_description]
        exclude: [credentials]
      authority: [inspect]
      tools: [list_files]
      output_schema: text
      success_criteria: non_empty_output
"""


def open_run_ledger(path, mode, opening_commitments_microcny=None):
    """Create or reopen the study ledger.

    A missing opening map means zero, which is the r1 file. An r2 file passes
    the round-1 settled totals and a reopen must still be carrying them.
    """
    path = Path(path)
    if path.exists():
        ledger = BudgetLedger(path)
        state = ledger.snapshot()
        if state.get("run_mode") != mode:
            raise BudgetStopped("ledger_run_mode_mismatch")
        if state.get("total_limit_microcny") != STUDY_TOTAL_LIMIT_MICROCNY:
            raise BudgetStopped("study_total_cap_required")
        if set(state["limits_microcny"]) != set(MODEL_POOL):
            raise BudgetStopped("ledger_model_pool_mismatch")
        if any(value != PER_MODEL_LIMIT_MICROCNY for value in state["limits_microcny"].values()):
            raise BudgetStopped("study_model_cap_must_be_50_cny")
        if opening_commitments_microcny is not None and ledger.opening_microcny() != dict(opening_commitments_microcny):
            raise BudgetStopped("prior_spend_mismatch")
        return ledger
    path.parent.mkdir(parents=True, exist_ok=True)
    limits = PROTOCOL_UNIT_LIMITS["m1_b0"]
    opening = {model: 0 for model in MODEL_POOL}
    if opening_commitments_microcny is not None:
        opening = dict(opening_commitments_microcny)
    # The file ceiling must cover the optional ProgramBench smoke (40). Paid
    # units still stop at their own request_cap. The protocol table stays 32.
    return BudgetLedger.create(
        path,
        model_limits_microcny={model: PER_MODEL_LIMIT_MICROCNY for model in MODEL_POOL},
        opening_commitments_microcny=opening,
        max_requests=max(limits["requests"], 40), prior_requests=0,
        max_tokens=limits["tokens"], prior_tokens=0,
        max_seconds_ms=limits["seconds"] * 1000, prior_seconds_ms=0,
        max_tools=limits["tools"], prior_tools=0,
        total_limit_microcny=STUDY_TOTAL_LIMIT_MICROCNY, run_mode=mode,
    )


def _strip_private(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_PUBLIC_KEYS:
                continue
            cleaned[str(key)] = _strip_private(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    if isinstance(value, str):
        return value
    if type(value) in (int, float, bool) or value is None:
        return value
    return str(value)


def load_public_payload(data_dir, track, task_id):
    """Read a staged public task file. Gold patches and tests are dropped."""
    if data_dir is None:
        return None
    path = Path(data_dir) / track / f"{task_id}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PilotStopped("public_task_payload_must_be_object")
    return _strip_private(payload)


def stage_for(unit, payload, *, dry_run):
    if payload is None:
        if not dry_run:
            raise PilotStopped("public_task_payload_missing")
        public = {"task_id": unit.task_id, "track": unit.track,
                  "note": "dry-run placeholder; benchmark text is not loaded"}
    else:
        public = dict(payload)
    if unit.track == "ProgramBench":
        public["submission_contract"] = PROGRAMBENCH_SUBMISSION_CONTRACT
    return {"task_id": unit.task_id, "track": unit.track, "phase": unit.phase,
            "arm": unit.arm, "stratum": unit.stratum, "public": public}


def dry_run_sender(url, body, headers):
    """Deterministic completion. The URL and headers are not recorded."""
    del url, headers
    messages = body.get("messages") if isinstance(body, dict) else []
    parts = []
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
    blob = "\n".join(parts)
    if "ALO_DECOMPOSE" in blob:
        text = DRY_PLAN
    elif "STAGE_DONE" in blob:
        text = "STAGE_DONE"
    else:
        text = '{"final":"dry-run submission"}'
    model = body.get("model") if isinstance(body, dict) else None
    return {"model": model, "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}}


def _build_adapter(unit, user_preference=None):
    if unit.arm == "g1-main":
        return adapter_for("G1", {"fixed_model": G1_MAIN_MODEL, "max_turns": unit.request_cap})
    if unit.arm == "g1-strong":
        return adapter_for("G1", {"fixed_model": G1_STRONG_MODEL, "max_turns": unit.request_cap})
    if unit.arm == "g2":
        # Planner uses one request. The verifier keeps at least three. s1 gets the rest.
        worker_turns = max(1, unit.request_cap - 1 - 3)
        return adapter_for("G2", {
            "planner_model": G2_PLANNER_MODEL,
            "max_turns": worker_turns,
            "max_attempts": 2,
            "seed": unit.seed,
            "probe": unit.phase == "probe",
            "cold_prior": unit.m2_condition == "cold",
            "user_preference": user_preference,
        })
    if unit.arm == "g3":
        return adapter_for("G3", {
            "role_models": dict(G3_ROLES),
            "manager_model": G3_MANAGER_MODEL,
            "max_round": max(2, unit.request_cap),
        })
    raise PilotStopped("unknown_arm")


def _restore_transcript(adapter, payload):
    """Continue B1 from B0 transcripts. The new unit keeps its own request cap."""
    schema = payload.get("schema") if isinstance(payload, dict) else None
    if schema == "dualbench-single-v2":
        if payload.get("fixed_model") != adapter.fixed_model:
            raise PilotStopped("single_checkpoint_configuration_changed")
        adapter.messages = list(payload.get("messages") or [])
        return
    if schema == "dualbench-alo-v2":
        adapter.state = {key: value for key, value in payload.items() if key not in {"schema", "configuration"}}
        return
    if schema == "dualbench-groupchat-v2":
        if (payload.get("role_models") != adapter.role_models
                or payload.get("manager_model") != adapter.manager_model):
            raise PilotStopped("groupchat_checkpoint_configuration_changed")
        adapter.messages = list(payload.get("messages") or [])
        adapter.model_pool = tuple(payload["model_pool"]) if payload.get("model_pool") else None
        return
    raise PilotStopped("unknown_checkpoint_schema")


def _apply_parent(unit, adapter, services, run_dir):
    """Restore M1 same-task state, or M2 whitelisted statistics.

    A missing or invalid M2 parent is ``parent_missing``. The caller blocks the
    unit and does not start a cold organization in its place. M2 does not copy
    ``model.patch``.
    """
    if not unit.parent_unit_id:
        return None
    parent = Path(run_dir) / "checkpoints" / unit.parent_unit_id
    if unit.study == "m1" and unit.phase == "B1-self":
        state_path = parent / "adapter.json"
        if not state_path.is_file():
            return "b0_checkpoint_missing"
        _restore_transcript(adapter, json.loads(state_path.read_text(encoding="utf-8")))
        database = parent / "alo.sqlite3"
        if database.is_file() and unit.arm == "g2":
            shutil.copy2(database, services.state_dir / "alo.sqlite3")
        return None
    if unit.study == "m2" and unit.m2_condition in {"persist", "learned"}:
        from .state import StateContractError
        export_path = parent / "learning_state.json"
        if not export_path.is_file():
            return "parent_missing"
        try:
            value = json.loads(export_path.read_text(encoding="utf-8"))
            imported = import_learning_state(
                services.state_dir / "alo.sqlite3", value, allowed_models=MODEL_POOL, mode="persist")
        except (OSError, ValueError, StateContractError, json.JSONDecodeError):
            return "parent_missing"
        adapter.state["observations"] = imported["observations"]
        return None
    return None


def _termination_reason(status, reason, outcome):
    """Every unit record names one of the closed termination set."""
    if status == "blocked" or reason in {"parent_missing", "b0_checkpoint_missing"}:
        return "blocked"
    if reason in {
        "provider_outcome_unknown", "provider_outcome_charged", "second_unknown_interruption",
        "provider_unparseable_charged", "second_unparseable_interruption",
    }:
        return "provider_unknown"
    if reason in {
        "cumulative_request_or_model_limit", "unit_resource_exhausted", "unit_seconds_exhausted",
        "model_cny_cap_exhausted", "budget_infeasible_all_models", "study_total_cny_cap_exhausted",
    }:
        return "budget"
    if isinstance(outcome, dict):
        named = outcome.get("termination")
        if named in {"turn_cap", "final", "length_stop", "budget", "provider_unknown", "blocked"}:
            return named
        if outcome.get("status") == "completed":
            return "final"
    if status == "completed":
        return "final"
    if reason == "length_stop":
        return "length_stop"
    if reason == "provider_timeout_exhausted":
        return "provider_timeout_exhausted"
    return "turn_cap"


def _charges(services):
    totals = {model: 0 for model in MODEL_POOL}
    for event, payload in services.events:
        if event == "model_settled" and isinstance(payload, dict):
            model = payload.get("model")
            amount = payload.get("charge_microcny")
            if model in totals and type(amount) is int:
                totals[model] += amount
    return {model: amount for model, amount in totals.items() if amount}


def _append_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with exclusive_lock(path.parent, "records"):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def _done_ids(path):
    done = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and isinstance(row.get("unit_id"), str):
            done[row["unit_id"]] = row
    return done


def _touched(request_id, unit_id, restart_count=0, protocol_epoch=0, infra_generation=0,
             track_generation=0):
    """True when this id belongs to the current epoch's attempt.

    Epoch 0 keeps the original ``req-<unit>`` prefix. A later protocol epoch
    uses ``req-e<N>-<unit>``, so settled rows from an archived epoch are not a
    partial retry. Within an epoch, a restart uses a different suffix, and a
    crash during the current attempt still blocks a silent retry. An infra
    requeue adds ``-infraN-`` and does not match the attempt it archived.
    """
    generation = f"-r{restart_count}" if restart_count else ""
    infra = f"-infra{infra_generation}" if infra_generation else ""
    track = f"-pbv{track_generation}" if track_generation else ""
    if protocol_epoch:
        stem = f"e{protocol_epoch}-{unit_id}{generation}{infra}{track}-"
    else:
        stem = f"{unit_id}{generation}{infra}{track}-"
    return request_id.startswith(f"req-{stem}") or request_id.startswith(f"tool-{stem}")


def missing_payloads(units, data_dir):
    missing = []
    for unit in units:
        if load_public_payload(data_dir, unit.track, unit.task_id) is None:
            missing.append(f"{unit.track}/{unit.task_id}")
    return missing


def _unknown_log(run_dir):
    return Path(run_dir) / "unknown_outcomes.jsonl"


def _line_in_outcome_epoch(row, epoch):
    """Untagged historical lines count only in epoch 0. Later epochs use the tag."""
    if not isinstance(row, dict):
        return False
    tag = row.get("outcome_epoch")
    if epoch == 0:
        return tag in (None, 0)
    return tag == epoch


def _unknown_count(run_dir):
    path = _unknown_log(run_dir)
    if not path.is_file():
        return 0
    epoch = current_outcome_epoch(run_dir)
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _line_in_outcome_epoch(row, epoch):
            count += 1
    return count


def _note_unknown(run_dir, record):
    path = _unknown_log(run_dir)
    line = json.dumps({
        "unit_id": record.get("unit_id"),
        "reason": record.get("reason"),
        "request_id": record.get("unknown_request_id"),
        "outcome_epoch": current_outcome_epoch(run_dir),
    }, ensure_ascii=False) + "\n"
    with exclusive_lock(run_dir, "unknown"):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return _unknown_count(run_dir)


def reclassify_charged_unparseable(run_dir):
    """Drop settled unparseable rows from the unknown counter. Idempotent.

    Epoch-4 rows were written as ``provider_outcome_charged`` with a null
    request id. A diagnostic with ``cause=unparseable_success`` and
    ``settled=true`` is a known worst-case charge, not an unsettled cost.
    Moved rows are listed in ``progress/reclassified_unparseable.jsonl``.
    """
    run_dir = Path(run_dir)
    unknown = _unknown_log(run_dir)
    audit = run_dir / "progress" / "reclassified_unparseable.jsonl"
    with exclusive_lock(run_dir, "unknown"):
        already = set()
        if audit.is_file():
            for line in audit.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("request_id"), str):
                    already.add(row["request_id"])
        diagnostics = []
        directory = run_dir / "diagnostics"
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                try:
                    body = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(body, dict):
                    continue
                if body.get("cause") != "unparseable_success" or body.get("settled") is not True:
                    continue
                request_id = body.get("request_id")
                if not isinstance(request_id, str) or not request_id or request_id in already:
                    continue
                diagnostics.append(body)
        if not unknown.is_file():
            return {"reclassified": 0}
        lines = [line for line in unknown.read_text(encoding="utf-8").splitlines() if line.strip()]
        kept, moved = [], []
        used = set()
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            match = None
            if isinstance(row, dict) and row.get("reason") == "provider_outcome_charged":
                for body in diagnostics:
                    request_id = body["request_id"]
                    if request_id in used:
                        continue
                    row_request = row.get("request_id")
                    if row_request not in (None, "") and row_request != request_id:
                        continue
                    row_unit = row.get("unit_id")
                    diag_unit = body.get("unit_id") or ""
                    if row_unit not in (None, "") and diag_unit and row_unit != diag_unit:
                        continue
                    match = body
                    break
            if match is None:
                kept.append(line)
                continue
            used.add(match["request_id"])
            moved.append({
                "request_id": match["request_id"],
                "unit_id": row.get("unit_id") or match.get("unit_id") or "",
                "previous_reason": "provider_outcome_charged",
                "diagnostic_cause": "unparseable_success",
                "model": match.get("model"),
            })
        if not moved:
            return {"reclassified": 0}
        body = "".join(line + "\n" for line in kept)
        unknown.write_text(body, encoding="utf-8")
        audit.parent.mkdir(parents=True, exist_ok=True)
        with audit.open("a", encoding="utf-8") as handle:
            for row in moved:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return {"reclassified": len(moved)}


def _ancestor_missing(unit, done):
    """True when a parent in this chain was blocked for a missing parent."""
    seen = set()
    parent_id = unit.parent_unit_id
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        row = done.get(parent_id)
        if not isinstance(row, dict):
            return False
        if row.get("reason") in {"parent_missing", "b0_checkpoint_missing"} or (
                row.get("status") == "blocked" and row.get("termination_reason") == "blocked"
                and row.get("reason") == "parent_missing"):
            return True
        parent_id = row.get("parent_unit_id") or ""
    return False


def _blocked_parent_record(unit):
    return {
        "schema": "alo-paid-unit-v1",
        "unit_id": unit.unit_id,
        "study": unit.study,
        "arm": unit.arm,
        "track": unit.track,
        "task_id": unit.task_id,
        "phase": unit.phase,
        "m2_condition": unit.m2_condition,
        "order_id": unit.order_id,
        "parent_unit_id": unit.parent_unit_id,
        "status": "blocked",
        "reason": "parent_missing",
        "termination_reason": "blocked",
        "halt_schedule": False,
        "charged_microcny": {},
        "sealed_files": 0,
    }


def _require_batch(run_dir, batch):
    """A run directory belongs to one batch. r1 records are not an r2 resume."""
    if batch not in {"r1", "r2"}:
        raise PilotStopped("unknown_batch")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "batch.json"
    if marker.is_file():
        try:
            found = json.loads(marker.read_text(encoding="utf-8")).get("batch")
        except (OSError, json.JSONDecodeError):
            found = None
        if found != batch:
            raise PilotStopped("batch_mismatch")
    else:
        records = run_dir / "records.jsonl"
        if records.is_file() and records.stat().st_size:
            raise PilotStopped("batch_mismatch")
        marker.write_text(json.dumps({"batch": batch, "schema": "alo-batch-v1"}, sort_keys=True) + "\n",
                          encoding="utf-8")


def _soft_model_halt(run_dir, ledger, shard_label, limits):
    """Between units, a per-model soft cap stops later units without a hard halt."""
    if not limits:
        return False
    used = ledger.snapshot().get("committed_or_reserved_microcny") or {}
    for model, micro in limits.items():
        if type(micro) is int and used.get(model, 0) >= micro:
            set_halt(run_dir, f"model_soft_halt:{model}", shard_label, hard=False)
            return True
    return False


def run_schedule(units, run_dir, *, live, data_dir=None, docker_image=None, sender=None,
                 git_runner=None, sandbox_memory=None, sandbox_pids=None, image_seeder=None,
                 shard=None, benchmarks=None, batch=None, model_soft_halts=None,
                 limit_overrides=None, reasoning_budget=None):
    """Execute units in order. Completed JSONL rows are skipped on resume.

    ``benchmarks`` limits the schedule to those tracks. Excluded units are not
    opened: no record, ledger row, requeue, or restart. An empty included set
    returns before the ledger is created.
    """
    run_dir = Path(run_dir)
    mode = "live" if live else "dry-run"
    if live and sender is not None:
        raise PilotStopped("live_run_uses_the_network_transport")
    if not live and sender is None:
        sender = dry_run_sender
    shard_label = "" if shard is None else f"{shard[0]}/{shard[1]}"
    if shard is not None:
        if batch == "r2":
            from .r2 import filter_r2_shard
            units = filter_r2_shard(units, shard[0], shard[1])
        else:
            units = filter_shard(units, shard[0], shard[1])
    units, _skipped = filter_benchmarks(units, benchmarks)
    if benchmarks and not units:
        return []
    if batch is not None:
        _require_batch(run_dir, batch)
        if batch == "r2":
            from .r2 import R2_COMMAND_SECONDS, R2_REASONING_BUDGET, r2_model_soft_halts, write_r2_manifest
            try:
                write_r2_manifest(run_dir)
            except ValueError as error:
                raise PilotStopped(str(error)) from None
            if limit_overrides is None:
                limit_overrides = {"command_seconds": R2_COMMAND_SECONDS}
            if reasoning_budget is None:
                reasoning_budget = R2_REASONING_BUDGET
            if model_soft_halts is None:
                model_soft_halts = r2_model_soft_halts()
    prior_opening = None
    read_timeouts = None
    connect_timeout_s = None
    stream_idle_timeouts = None
    continue_after_timeout = False
    if batch == "r2":
        from .r2 import (
            R2_CONNECT_TIMEOUT_S, R2_READ_TIMEOUT_S, R2_STREAM_IDLE_TIMEOUT_S, resolve_prior_spend,
        )
        read_timeouts = dict(R2_READ_TIMEOUT_S)
        connect_timeout_s = R2_CONNECT_TIMEOUT_S
        stream_idle_timeouts = dict(R2_STREAM_IDLE_TIMEOUT_S)
        continue_after_timeout = True
        try:
            prior_opening = resolve_prior_spend()["applied_microcny"]
        except ValueError as error:
            raise PilotStopped(str(error)) from None
    ledger = open_run_ledger(run_dir / "budget.sqlite3", mode, prior_opening)
    if live:
        missing = [name for name in KEY_ENVS if not os.environ.get(name)]
        if missing:
            raise PilotStopped("provider_credential_missing")
        absent = missing_payloads(units, data_dir)
        if absent:
            raise PilotStopped("public_task_payload_missing")
    if ledger.blocks_dispatch(shard_label):
        raise BudgetStopped("unresolved_request_blocks_subsequent_dispatch")
    records_path = run_dir / "records.jsonl"
    done = _done_ids(records_path)
    known_ids = ledger.request_ids()
    publish_timeout_counts(run_dir, ledger, protocol_epoch=current_epoch(run_dir))
    reclassify_charged_unparseable(run_dir)
    results = []
    try:
        for unit in units:
            if drain_requested(run_dir):
                return results
            halt = halt_record(run_dir)
            if halt:
                if halt["hard"]:
                    raise PilotStopped(halt["reason"])
                return results
            if _ancestor_missing(unit, done):
                record = _blocked_parent_record(unit)
                _append_record(records_path, record)
                done[unit.unit_id] = record
                results.append(record)
                continue
            if unit.unit_id in done:
                # The row is terminal, including one that stopped the previous
                # process with halt_schedule. Resume does not raise that flag again.
                results.append(done[unit.unit_id])
                _stop_if_blocked(results, benchmarks=benchmarks)
                continue
            epoch = current_epoch(run_dir)
            infra_generation = infra_generations(run_dir).get(unit.unit_id, 0)
            track_generation = track_generations(run_dir).get(unit.unit_id, 0)
            if any(_touched(request_id, unit.unit_id, restart_counts(run_dir).get(unit.unit_id, 0),
                            epoch, infra_generation, track_generation)
                   for request_id in known_ids):
                raise BudgetStopped("partial_unit_not_retried")
            retried = False
            while True:
                preference = None
                if batch == "r2" and unit.arm == "g2":
                    from .r2 import USER_PREFERENCE
                    preference = dict(USER_PREFERENCE)
                record = _execute_unit(
                    unit, run_dir, ledger, live=live, data_dir=data_dir,
                    docker_image=docker_image, sender=sender, git_runner=git_runner,
                    sandbox_memory=sandbox_memory, sandbox_pids=sandbox_pids,
                    image_seeder=image_seeder, unknown_so_far=_unknown_count(run_dir),
                    shard_id=shard_label, limit_overrides=limit_overrides,
                    reasoning_budget=reasoning_budget, user_preference=preference,
                    read_timeouts=read_timeouts, connect_timeout_s=connect_timeout_s,
                    stream_idle_timeouts=stream_idle_timeouts,
                    continue_after_timeout=continue_after_timeout)
                if record.get("charged_timeout"):
                    verdict = publish_timeout_counts(
                        run_dir, ledger, protocol_epoch=current_epoch(run_dir))
                    record["charged_timeout_epoch_microcny"] = verdict["charged_timeout_microcny"]
                    if verdict["halt_reason"]:
                        record["halt_schedule"] = True
                        record["reason"] = verdict["halt_reason"]
                if record.get("charged_unparseable"):
                    from .progress import publish_unparseable_counts
                    verdict = publish_unparseable_counts(
                        run_dir, ledger, protocol_epoch=current_epoch(run_dir))
                    record["charged_unparseable_epoch_microcny"] = verdict["charged_unparseable_microcny"]
                    if verdict["halt_reason"]:
                        record["halt_schedule"] = True
                        record["reason"] = verdict["halt_reason"]
                if record.get("charged_unknown"):
                    count = _note_unknown(run_dir, record)
                    if count > MAX_UNKNOWN_OUTCOMES:
                        record["halt_schedule"] = True
                        record["reason"] = "unknown_outcome_threshold"
                _append_record(records_path, record)
                known_ids = ledger.request_ids()
                if (record.get("status") == "stopped"
                        and record.get("reason") != "provider_timeout_exhausted"
                        and (record.get("charged_unknown") or record.get("charged_timeout")
                             or record.get("charged_unparseable") or record.get("tool_sandbox_unknown"))
                        and not retried and record.get("restart_count", 0) == 0
                        and not record.get("halt_schedule")):
                    if record.get("charged_timeout"):
                        restart_reason = "charged timeout; restart once"
                    elif record.get("charged_unparseable"):
                        restart_reason = "unparseable charged; restart once"
                    elif record.get("tool_sandbox_unknown"):
                        restart_reason = "tool sandbox unknown; restart once"
                    else:
                        restart_reason = "charged unknown; restart once"
                    restart_unit(run_dir, unit.unit_id, reason=restart_reason, ledger=ledger,
                                 shard_id=shard_label)
                    retried = True
                    continue
                break
            results.append(record)
            done[unit.unit_id] = record
            if _soft_model_halt(run_dir, ledger, shard_label, model_soft_halts):
                return results
            _stop_if_blocked(results, run_dir=run_dir, shard=shard_label, benchmarks=benchmarks)
            if record.get("halt_schedule"):
                reason = record.get("reason") or "schedule_stopped"
                set_halt(run_dir, reason, shard_label)
                raise PilotStopped(reason)
    finally:
        collect_sealed(run_dir)
    return results


def _want_image_seed(live, image_seeder):
    if image_seeder is not None:
        return True
    if os.environ.get("ALO_SEED_IMAGES") == "1":
        return True
    return bool(live)


def _max_blocked_fraction(maximum):
    if maximum is not None:
        return float(maximum)
    raw = os.environ.get("ALO_MAX_BLOCKED_FRACTION", "").strip()
    if not raw:
        return MAX_BLOCKED_FRACTION
    try:
        value = float(raw)
    except ValueError:
        return MAX_BLOCKED_FRACTION
    if value < 0 or value > 1:
        return MAX_BLOCKED_FRACTION
    return value


def _rows_for_benchmarks(records, benchmarks):
    """Records that count toward a gate. An empty filter keeps every row."""
    if not benchmarks:
        return records
    allowed = set(benchmarks)
    return [row for row in records if isinstance(row, dict) and row.get("track") in allowed]


def _rows_for_outcome_epoch(records, epoch):
    """Gate sample for one outcome epoch. Epoch 0 still sees untagged history."""
    if epoch is None:
        return records
    return [row for row in records if _line_in_outcome_epoch(row, epoch)]


def blocked_fraction_exceeded(records, *, maximum=None, benchmarks=None, outcome_epoch=None):
    """True when blocked units are more than the allowed share of a large enough sample."""
    records = _rows_for_outcome_epoch(_rows_for_benchmarks(records, benchmarks), outcome_epoch)
    finished = [row for row in records if isinstance(row, dict) and row.get("status")]
    if len(finished) < BLOCKED_FRACTION_MIN_UNITS:
        return False
    blocked = sum(1 for row in finished if row.get("status") == "blocked")
    return blocked / len(finished) > _max_blocked_fraction(maximum)


def format_blocked_report(records, *, maximum=None, benchmarks=None):
    """One histogram line per blocked reason, then whether the schedule must stop."""
    records = _rows_for_benchmarks(records, benchmarks)
    counts = {}
    finished = 0
    blocked = 0
    for row in records:
        if not isinstance(row, dict) or not row.get("status"):
            continue
        finished += 1
        if row.get("status") != "blocked":
            continue
        blocked += 1
        reason = row.get("reason") or "unspecified"
        counts[reason] = counts.get(reason, 0) + 1
    lines = [f"blocked {blocked} of {finished}"]
    for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"{count} {reason}")
    exceeded = blocked_fraction_exceeded(records, maximum=maximum)
    if exceeded:
        lines.append("blocked_fraction_exceeded")
    return "\n".join(lines), exceeded


def _stop_if_blocked(records, run_dir=None, shard="", benchmarks=None):
    if run_dir is not None:
        with exclusive_lock(run_dir, "records"):
            records = list(_done_ids(Path(run_dir) / "records.jsonl").values())
    epoch = current_outcome_epoch(run_dir) if run_dir is not None else 0
    records = _rows_for_outcome_epoch(records, epoch)
    exceeded = blocked_fraction_exceeded(records, benchmarks=benchmarks, outcome_epoch=epoch)
    if not exceeded:
        return
    text, _exceeded = format_blocked_report(records, benchmarks=benchmarks)
    print(text)
    if run_dir is not None:
        set_halt(run_dir, "blocked_fraction_exceeded", shard)
    raise PilotStopped("blocked_fraction_exceeded")


def _prepare_workspace(unit, services, *, live, repo, base_commit, git_runner, image, image_seeder, run_dir):
    """Copy the image tree into the one live workspace. Resume reapplies onto image HEAD.

    SWE comes from the image ``/testbed``. Official images may add one
    mode-only commit above ``base_commit``; the scored patch and a resumed
    apply both use that HEAD, which is the tree the harness patches.
    ProgramBench comes from the image ``/workspace``.
    A fresh GitHub clone is not mounted over ``/testbed``.
    """
    del repo
    if not _want_image_seed(live, image_seeder):
        return None
    if not image:
        return "sandbox_image_missing"
    root = services.workspace.root
    try:
        if unit.track == "SWE-bench_Lite":
            if not base_commit:
                return "swe_checkout_metadata_missing"
            seed_image_tree(image, "/testbed", root, docker=image_seeder)
            image_head = resolve_image_head(root, base_commit, git=git_runner)
            if not image_head:
                return "swe_image_base_commit_mismatch"
            services.image_head = image_head
            # Same-task B1 is the only restore that reapplies a parent patch.
            # M2 carries whitelisted statistics and must not copy model.patch.
            if unit.study == "m1" and unit.phase == "B1-self" and unit.parent_unit_id:
                parent_patch = Path(run_dir) / "checkpoints" / unit.parent_unit_id / "model.patch"
                if parent_patch.is_file():
                    apply_patch(root, parent_patch.read_text(encoding="utf-8"))
            return None
        if unit.track == "ProgramBench":
            seed_image_tree(image, "/workspace", root, docker=image_seeder)
            if release_unreadable_executable(root):
                services.preserve_image_executable = True
            parent_tar = (Path(run_dir) / "sealed" / "programbench" / unit.parent_unit_id
                          / unit.task_id / "submission.tar.gz")
            if unit.parent_unit_id and parent_tar.is_file():
                _extract_submission(parent_tar, root)
            return None
    except PilotStopped as error:
        return str(error)
    return None


def _extract_submission(archive, dest):
    """Overlay a previous ProgramBench submission onto a fresh image seed."""
    dest = Path(dest)
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            name = member.name
            if not name or name.startswith("/") or ".." in Path(name).parts:
                continue
            if ".git" in Path(name).parts or Path(name).name == "executable":
                continue
            target = dest / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(extracted.read())
            try:
                target.chmod(member.mode & 0o777)
            except OSError:
                pass


def _public_field(payload, key):
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _execute_unit(unit, run_dir, ledger, *, live, data_dir, docker_image, sender,
                  git_runner=None, sandbox_memory=None, sandbox_pids=None, image_seeder=None,
                  unknown_so_far=0, shard_id="", limit_overrides=None, reasoning_budget=None,
                  pro_call_cap=None, user_preference=None, read_timeouts=None,
                  connect_timeout_s=None, stream_idle_timeouts=None,
                  continue_after_timeout=False):
    record = {
        "schema": "alo-paid-unit-v1",
        "unit_id": unit.unit_id,
        "study": unit.study,
        "arm": unit.arm,
        "track": unit.track,
        "task_id": unit.task_id,
        "phase": unit.phase,
        "m2_condition": unit.m2_condition,
        "order_id": unit.order_id,
        "seed": unit.seed,
        "request_cap": unit.request_cap,
        "parent_unit_id": unit.parent_unit_id,
        "mode": "live" if live else "dry-run",
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "termination_reason": "blocked",
    }
    state_dir = Path(run_dir) / "units" / unit.unit_id / "state"
    restart_count = restart_counts(run_dir).get(unit.unit_id, 0)
    infra_generation = infra_generations(run_dir).get(unit.unit_id, 0)
    track_generation = track_generations(run_dir).get(unit.unit_id, 0)
    epoch = current_epoch(run_dir)
    record["restart_count"] = restart_count
    record["infra_generation"] = infra_generation
    record["track_generation"] = track_generation
    record["protocol_epoch"] = epoch
    services = StudyServices(
        state_dir, ledger, unit=unit.protocol_unit,
        per_request_microcny_ceiling=PER_MODEL_LIMIT_MICROCNY,
        charge_microcny=charge_microcny,
        request_ceiling=ceiling_for_messages,
        sender=None if live else sender,
        permit_network=live,
        require_credentials=live,
        unit_id=unit.unit_id,
        limits={"requests": unit.request_cap},
        limit_overrides=limit_overrides,
        reasoning_budget=reasoning_budget,
        restart_count=1 if restart_count >= 1 else 0,
        protocol_epoch=epoch,
        shard_id=shard_id,
        infra_generation=infra_generation,
        track_generation=track_generation,
        budget_notice=PROGRAMBENCH_BUDGET_NOTICE if unit.track == "ProgramBench" else None,
        read_timeouts=read_timeouts,
        connect_timeout_s=connect_timeout_s,
        stream_idle_timeouts=stream_idle_timeouts,
        continue_after_timeout=continue_after_timeout,
    )
    if type(pro_call_cap) is int:
        services.pro_call_cap = pro_call_cap
        record["pro_call_cap"] = pro_call_cap
    payload = load_public_payload(data_dir, unit.track, unit.task_id)
    repo = _public_field(payload, "repo")
    base_commit = _public_field(payload, "base_commit")
    if repo:
        record["repo"] = repo
    if base_commit:
        record["base_commit"] = base_commit
    image = resolve_task_image(unit.track, unit.task_id, fallback=docker_image)
    record["sandbox_image"] = image
    adapter = _build_adapter(unit, user_preference=user_preference)
    parent_note = _apply_parent(unit, adapter, services, run_dir)
    if parent_note in {"b0_checkpoint_missing", "parent_missing"}:
        record.update(status="blocked", reason="parent_missing" if parent_note == "parent_missing" else parent_note,
                      charged_microcny={}, sealed_files=0, termination_reason="blocked")
        _delete_live_unit(run_dir, unit.unit_id)
        return record
    if parent_note:
        record["parent_note"] = parent_note
    prepare_note = _prepare_workspace(
        unit, services, live=live, repo=repo, base_commit=base_commit, git_runner=git_runner,
        image=image, image_seeder=image_seeder, run_dir=run_dir)
    if prepare_note:
        record.update(status="blocked", reason=prepare_note, charged_microcny={}, sealed_files=0)
        _delete_live_unit(run_dir, unit.unit_id)
        return record
    image_head = getattr(services, "image_head", None)
    if image_head:
        record["image_head"] = image_head
    checkpoint = Path(run_dir) / "checkpoints" / unit.unit_id

    def _seal_partial(patch):
        checkpoint.mkdir(parents=True, exist_ok=True)
        text = patch if isinstance(patch, str) else ""
        (checkpoint / "model.patch").write_text(text, encoding="utf-8")
        record["patch_bytes"] = len(text.encode("utf-8"))

    services.seal_patch = _seal_partial
    _attach_sandbox(services, unit, image, live=live, memory=sandbox_memory, pids=sandbox_pids)
    record["sandbox_workdir"] = getattr(services.sandbox, "workdir", None) if services.sandbox else None
    record["preserve_executable_in_image"] = bool(getattr(services, "preserve_image_executable", False))
    stage = stage_for(unit, payload, dry_run=not live)
    status, reason, halt = "completed", "", False
    outcome = None
    try:
        outcome = adapter.run_stage(stage, services)
        if isinstance(outcome, dict) and outcome.get("status") in {"completed", "failed"}:
            status = outcome["status"]
    except ImportError as error:
        if "autogen" not in str(error):
            raise
        status, reason = "blocked", "native_groupchat_not_installed"
    except PilotStopped as error:
        reason = str(error)
        status = "stopped"
        halt = reason not in {
            "cumulative_request_or_model_limit", "unit_resource_exhausted",
            "unit_seconds_exhausted", "provider_outcome_charged", "provider_deadline_charged",
            "provider_unparseable_charged", "pro_share_cap",
        }
        if reason == "provider_timeout_exhausted":
            # Same messages were retried once inside the unit. That is not a restart.
            status = "stopped"
            halt = False
        elif reason == "provider_deadline_charged":
            record["charged_timeout"] = True
            if restart_count >= 1:
                status = "blocked"
                reason = "second_timeout_interruption"
            halt = False
        elif reason == "provider_unparseable_charged":
            record["charged_unparseable"] = True
            record["unknown_request_id"] = getattr(services, "last_request_id", None)
            if restart_count >= 1:
                status = "blocked"
                reason = "second_unparseable_interruption"
            halt = False
        elif reason == "provider_outcome_charged":
            record["charged_unknown"] = True
            record["unknown_request_id"] = getattr(services, "last_request_id", None)
            if restart_count >= 1:
                status = "blocked"
                reason = "second_unknown_interruption"
            halt = unknown_so_far + 1 > MAX_UNKNOWN_OUTCOMES
        elif reason == "tool_sandbox_unknown":
            record["tool_sandbox_unknown"] = True
            if restart_count >= 1:
                status = "blocked"
                reason = "second_tool_sandbox_interruption"
            halt = False
        elif reason == "provider_outcome_unknown":
            record["unknown_request_id"] = getattr(services, "last_request_id", None)
            if restart_count >= 1:
                status = "blocked"
                reason = "second_unknown_interruption"
                halt = True
    except BudgetStopped as error:
        reason = str(error)
        status, halt = "stopped", True
    sealed_files = _persist_compact(unit, services, adapter, run_dir, record)
    if getattr(services, "charged_timeouts", 0):
        record["charged_timeout"] = True
        record["charged_timeout_events"] = services.charged_timeouts
    record["outcome_epoch"] = current_outcome_epoch(run_dir)
    record["termination_reason"] = _termination_reason(status, reason, outcome)
    record["length_stops"] = getattr(services, "length_stops", 0)
    record["transport"] = transport_summary(getattr(services, "transport_counts", {}))
    record["edit_log"] = list(getattr(services, "edit_log", []) or [])
    record["g3_request_split"] = dict(getattr(services, "g3_request_split", {}) or {})
    decisions = getattr(adapter, "state", {}).get("decision_records") if hasattr(adapter, "state") else None
    if isinstance(decisions, list):
        record["decision_records"] = decisions
    _delete_live_unit(run_dir, unit.unit_id)
    resources = services.resources()
    record.update(
        status=status, reason=reason, halt_schedule=halt,
        charged_microcny=_charges(services),
        observed_input_tokens=resources["observed_input_tokens"],
        observed_output_tokens=resources["observed_output_tokens"],
        model_requests=resources["model_requests"],
        estimated_cost_cny=resources["estimated_cost_cny"],
        sealed_files=sealed_files,
        finished_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return record


def _attach_sandbox(services, unit, image, *, live, memory, pids):
    if not image:
        return
    preserve = bool(getattr(services, "preserve_image_executable", False))
    if unit.track == "SWE-bench_Lite":
        services.sandbox = DockerSandbox(
            services.workspace.root, image_resolver=lambda image=image: image,
            permit_execution=live, memory=memory, pids_limit=pids,
            workdir="/testbed", mount_points=("/testbed",), path=SWE_TESTBED_PATH)
        return
    if preserve:
        services.sandbox = DockerSandbox(
            services.workspace.root, image_resolver=lambda image=image: image,
            permit_execution=live, memory=memory, pids_limit=pids,
            merge_image_workspace=True)
        return
    services.sandbox = DockerSandbox(
        services.workspace.root, image_resolver=lambda image=image: image,
        permit_execution=live, memory=memory, pids_limit=pids)


def _persist_compact(unit, services, adapter, run_dir, record):
    """Store a patch or submission tarball, not a second and third full tree."""
    from .collect import _pack_submission, workspace_patch
    checkpoint = Path(run_dir) / "checkpoints" / unit.unit_id
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "adapter.json").write_text(
        json.dumps(adapter.dump_state(), ensure_ascii=False), encoding="utf-8")
    (checkpoint / "metadata.json").write_text(json.dumps({
        "base_commit": record.get("base_commit"),
        "image_head": record.get("image_head"),
        "sandbox_image": record.get("sandbox_image"),
        "track": unit.track,
        "task_id": unit.task_id,
        "preserve_executable_in_image": bool(getattr(services, "preserve_image_executable", False)),
    }, sort_keys=True), encoding="utf-8")
    database = services.state_dir / "alo.sqlite3"
    if database.is_file():
        shutil.copy2(database, checkpoint / "alo.sqlite3")
        if unit.arm == "g2" and unit.phase == "learning" and unit.m2_condition == "persist":
            try:
                exported = export_learning_state(
                    database, adapter.dump_state(), allowed_models=MODEL_POOL, evidence_id=unit.unit_id)
                (checkpoint / "learning_state.json").write_text(
                    json.dumps(exported, ensure_ascii=False), encoding="utf-8")
            except (ValueError, OSError) as error:
                record["learning_export_error"] = type(error).__name__
    workspace = services.workspace.root
    state = adapter.dump_state() if hasattr(adapter, "dump_state") else {}
    if isinstance(state, dict) and isinstance(state.get("probe_transcripts"), dict):
        (checkpoint / "probe_transcript.json").write_text(
            json.dumps(state["probe_transcripts"], ensure_ascii=False), encoding="utf-8")
        record["probe_transcript_saved"] = True
    if unit.track == "SWE-bench_Lite":
        patch = ""
        try:
            diff_base = record.get("image_head") or record.get("base_commit") or ""
            patch = workspace_patch(workspace, diff_base)
        except RuntimeError:
            record["patch_note"] = "git_diff_failed"
        (checkpoint / "model.patch").write_text(patch, encoding="utf-8")
        record["patch_bytes"] = len(patch.encode("utf-8"))
        return 1 if patch else 0
    if unit.track == "ProgramBench":
        compile_sh = workspace / "compile.sh"
        present = compile_sh.is_file()
        record["programbench_compile_sh_present"] = present
        paths = []
        for row in getattr(services, "edit_log", []) or []:
            if isinstance(row, dict):
                paths.extend(row.get("paths") or [])
        source_suffixes = {".py", ".c", ".cc", ".cpp", ".go", ".rs", ".java", ".sh", ".js", ".ts"}
        record["source_written"] = any(Path(path).suffix.lower() in source_suffixes or Path(path).name == "compile.sh"
                                       for path in paths if isinstance(path, str))
        build_dir = workspace / "build"
        built = build_dir.is_dir() and any(build_dir.iterdir()) if build_dir.is_dir() else False
        record["compile_ok"] = True if built or (present and record["source_written"]) else None
        record["run_ok"] = None
        record["programbench_build_dir"] = "build" if built else None
        record["programbench_budget_notice"] = bool(getattr(services, "budget_notice_sent", False))
        try:
            write_diagnostic(
                services.ledger, f"submission-{unit.unit_id}",
                cause="submission_check", model="", unit_id=unit.unit_id,
                http_status=None, vendor_code="compile.sh",
                vendor_message="present" if present else "missing",
                response={"compile_sh_present": present, "path": "compile.sh"},
                finish_reason=None, latency_ms=0, settled=True)
        except (OSError, ValueError):
            record["programbench_submission_check_error"] = "diagnostic_not_written"
        dest = (Path(run_dir) / "sealed" / "programbench" / unit.unit_id / unit.task_id
                / "submission.tar.gz")
        return _pack_submission(workspace, dest)
    return 0


def _delete_live_unit(run_dir, unit_id):
    shutil.rmtree(Path(run_dir) / "units" / unit_id, ignore_errors=True)


def _parse_shard(text):
    if text is None or text == "":
        return None
    parts = text.split("/")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise PilotStopped("invalid_shard")
    index, count = int(parts[0]), int(parts[1])
    if count < 1 or index < 0 or index >= count:
        raise PilotStopped("invalid_shard")
    return index, count


def _parser():
    parser = argparse.ArgumentParser(description="Run the frozen M1 or M2 schedule.")
    parser.add_argument("--study", choices=("m1", "m2"))
    parser.add_argument("--arm", action="append", default=[],
                        choices=("g1-main", "g1-strong", "g2", "g3"))
    parser.add_argument("--tasks", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--blocked-report", action="store_true",
                        help="Print the blocked-reason histogram and exit 1 if the fraction is too high.")
    parser.add_argument("--shard", default=None,
                        help="Task shard i/N (0-based). Whole tasks stay on one shard.")
    parser.add_argument("--benchmarks", default=None,
                        help="Comma-separated tracks to run (SWE-bench, ProgramBench). "
                             "Default is ALO_BENCHMARKS, or every track when that is unset.")
    parser.add_argument("--batch", default=None, choices=("r1", "r2"),
                        help="r2 is a separate schedule and must not resume an r1 run directory.")
    parser.add_argument("--pb-smoke", action="store_true",
                        help="Run the optional ProgramBench errcheck g1-main B0 smoke (40 requests). Not scored.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--docker-image", default=os.environ.get("ALO_SANDBOX_IMAGE") or None)
    return parser


def _schedule_status(study, results, skipped, spent, benchmarks):
    body = {
        "study": study,
        "units": len(results),
        "skipped_by_filter": skipped,
        "charged_microcny": spent,
        "charged_cny": spent / MICROCNY_PER_CNY,
    }
    if benchmarks:
        body["benchmarks"] = list(benchmarks)
    print(json.dumps(body, sort_keys=True))


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        benchmarks = requested_benchmarks(args.benchmarks)
    except ValueError:
        parser.error(
            "--benchmarks must be a comma-separated list of SWE-bench, SWE-bench_Lite, ProgramBench")
    if args.blocked_report:
        records = list(_done_ids(Path(args.run_dir) / "records.jsonl").values())
        text, exceeded = format_blocked_report(records, benchmarks=benchmarks)
        print(text)
        return 1 if exceeded else 0
    if args.pb_smoke:
        args.study = args.study or "m1"
        args.batch = "r2"
    if args.study not in {"m1", "m2"}:
        parser.error("--study is required")
    if args.live == args.dry_run:
        parser.error("pass exactly one of --dry-run or --live")
    batch = args.batch
    if args.pb_smoke:
        from .r2 import R2_COMMAND_SECONDS, R2_REASONING_BUDGET, r2_model_soft_halts, smoke_unit
        units = (smoke_unit(),)
        limit_overrides = {"command_seconds": R2_COMMAND_SECONDS, "requests": 40}
        reasoning_budget = R2_REASONING_BUDGET
        soft_halts = r2_model_soft_halts()
        batch = "r2"
    elif batch == "r2":
        from .r2 import R2_COMMAND_SECONDS, R2_REASONING_BUDGET, r2_model_soft_halts, select_r2_units
        units = select_r2_units(args.study, arms=args.arm, tasks=args.tasks)
        limit_overrides = {"command_seconds": R2_COMMAND_SECONDS}
        reasoning_budget = R2_REASONING_BUDGET
        soft_halts = r2_model_soft_halts()
    else:
        units = select_units(args.study, arms=args.arm, tasks=args.tasks)
        limit_overrides = None
        reasoning_budget = None
        soft_halts = None
    if not units:
        parser.error("no units match --arm and --tasks")
    try:
        shard = _parse_shard(args.shard)
    except PilotStopped:
        parser.error("--shard must be i/N with 0 <= i < N")
    if shard is not None and batch == "r2":
        from .r2 import filter_r2_shard
        scoped = filter_r2_shard(units, shard[0], shard[1])
    else:
        scoped = filter_shard(units, shard[0], shard[1]) if shard is not None else tuple(units)
    _included, skipped_units = filter_benchmarks(scoped, benchmarks)
    # A shard whose units are all outside the filter is finished. Do not open
    # the ledger or write a record for the excluded tracks.
    if benchmarks and not _included:
        _schedule_status(args.study, [], len(skipped_units), 0, benchmarks)
        return 0
    try:
        results = run_schedule(
            units, args.run_dir, live=args.live, data_dir=args.data_dir,
            docker_image=args.docker_image, shard=shard, benchmarks=benchmarks,
            batch=batch, model_soft_halts=soft_halts, limit_overrides=limit_overrides,
            reasoning_budget=reasoning_budget)
    except (PilotStopped, BudgetStopped) as error:
        print(str(error))
        return 2
    spent = 0
    for row in results:
        for amount in (row.get("charged_microcny") or {}).values():
            spent += amount
    _schedule_status(args.study, results, len(skipped_units), spent, benchmarks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
