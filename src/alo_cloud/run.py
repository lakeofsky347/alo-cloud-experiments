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
from .contracts import (
    G1_MAIN_MODEL, G1_STRONG_MODEL, MODEL_POOL, PER_MODEL_LIMIT_MICROCNY, STUDY_TOTAL_LIMIT_MICROCNY,
)
from .errors import PilotStopped
from .freeze import (
    G2_PLANNER_MODEL, G3_MANAGER_MODEL, G3_ROLES, select_units,
)
from .prices import ceiling_for_messages, charge_microcny
from .sandbox import DockerSandbox
from .services import PROTOCOL_UNIT_LIMITS, StudyServices
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


def open_run_ledger(path, mode):
    """Create or reopen the study ledger. Opening commitments are zero only on create."""
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
        return ledger
    path.parent.mkdir(parents=True, exist_ok=True)
    limits = PROTOCOL_UNIT_LIMITS["m1_b0"]
    return BudgetLedger.create(
        path,
        model_limits_microcny={model: PER_MODEL_LIMIT_MICROCNY for model in MODEL_POOL},
        opening_commitments_microcny={model: 0 for model in MODEL_POOL},
        max_requests=limits["requests"], prior_requests=0,
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
        public = payload
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


def _copy_tree(source, dest):
    source, dest = Path(source), Path(dest)
    if not source.is_dir():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        target = dest / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _build_adapter(unit):
    if unit.arm == "g1-main":
        return adapter_for("G1", {"fixed_model": G1_MAIN_MODEL, "max_turns": unit.request_cap})
    if unit.arm == "g1-strong":
        return adapter_for("G1", {"fixed_model": G1_STRONG_MODEL, "max_turns": unit.request_cap})
    if unit.arm == "g2":
        return adapter_for("G2", {
            "planner_model": G2_PLANNER_MODEL,
            "max_turns": max(1, unit.request_cap - 1),
            "max_attempts": 1,
            "seed": unit.seed,
            "probe": unit.phase == "probe",
            "cold_prior": unit.m2_condition == "cold",
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
    if not unit.parent_unit_id:
        return None
    parent = Path(run_dir) / "checkpoints" / unit.parent_unit_id
    if unit.study == "m1" and unit.phase == "B1-self":
        state_path = parent / "adapter.json"
        if not state_path.is_file():
            return "b0_checkpoint_missing"
        _restore_transcript(adapter, json.loads(state_path.read_text(encoding="utf-8")))
        _copy_tree(parent / "workspace", services.workspace.root)
        database = parent / "alo.sqlite3"
        if database.is_file() and unit.arm == "g2":
            shutil.copy2(database, services.state_dir / "alo.sqlite3")
        return None
    if unit.m2_condition in {"persist", "learned"}:
        export_path = parent / "learning_state.json"
        if not export_path.is_file():
            return "persist_parent_absent"
        value = json.loads(export_path.read_text(encoding="utf-8"))
        imported = import_learning_state(
            services.state_dir / "alo.sqlite3", value, allowed_models=MODEL_POOL, mode="persist")
        adapter.state["observations"] = imported["observations"]
    return None


def _seal(workspace, dest):
    dest.mkdir(parents=True, exist_ok=True)
    files = [path for path in Path(workspace).rglob("*") if path.is_file() and not path.is_symlink()]
    for path in files:
        target = dest / path.relative_to(workspace)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if files:
        with tarfile.open(dest / "submission.tar.gz", "w:gz") as archive:
            for path in files:
                archive.add(path, arcname=str(path.relative_to(workspace)))
    return len(files)


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
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
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


def _touched(request_id, unit_id):
    return request_id.startswith(f"req-{unit_id}-") or request_id.startswith(f"tool-{unit_id}-")


def missing_payloads(units, data_dir):
    missing = []
    for unit in units:
        if load_public_payload(data_dir, unit.track, unit.task_id) is None:
            missing.append(f"{unit.track}/{unit.task_id}")
    return missing


def run_schedule(units, run_dir, *, live, data_dir=None, docker_image=None, sender=None):
    """Execute units in order. Completed JSONL rows are skipped on resume."""
    run_dir = Path(run_dir)
    mode = "live" if live else "dry-run"
    if live and sender is not None:
        raise PilotStopped("live_run_uses_the_network_transport")
    if not live and sender is None:
        sender = dry_run_sender
    ledger = open_run_ledger(run_dir / "budget.sqlite3", mode)
    if live:
        missing = [name for name in KEY_ENVS if not os.environ.get(name)]
        if missing:
            raise PilotStopped("provider_credential_missing")
        absent = missing_payloads(units, data_dir)
        if absent:
            raise PilotStopped("public_task_payload_missing")
    state = ledger.snapshot()
    if state["unresolved_request_ids"] or state["unresolved_tool_ids"]:
        raise BudgetStopped("unresolved_request_blocks_subsequent_dispatch")
    records_path = run_dir / "records.jsonl"
    done = _done_ids(records_path)
    known_ids = ledger.request_ids()
    results = []
    for unit in units:
        if unit.unit_id in done:
            results.append(done[unit.unit_id])
            continue
        if any(_touched(request_id, unit.unit_id) for request_id in known_ids):
            raise BudgetStopped("partial_unit_not_retried")
        record = _execute_unit(unit, run_dir, ledger, live=live, data_dir=data_dir,
                               docker_image=docker_image, sender=sender)
        _append_record(records_path, record)
        results.append(record)
        known_ids = ledger.request_ids()
        if record["status"] == "stopped" and record.get("halt_schedule"):
            raise PilotStopped(record.get("reason") or "schedule_stopped")
    return results


def _execute_unit(unit, run_dir, ledger, *, live, data_dir, docker_image, sender):
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
        "mode": "live" if live else "dry-run",
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    state_dir = Path(run_dir) / "units" / unit.unit_id / "state"
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
    )
    if docker_image:
        services.sandbox = DockerSandbox(services.workspace.root, image=docker_image, permit_execution=live)
    adapter = _build_adapter(unit)
    parent_note = _apply_parent(unit, adapter, services, run_dir)
    if parent_note == "b0_checkpoint_missing":
        record.update(status="blocked", reason=parent_note, charged_microcny={}, sealed_files=0)
        return record
    if parent_note:
        record["parent_note"] = parent_note
    payload = load_public_payload(data_dir, unit.track, unit.task_id)
    stage = stage_for(unit, payload, dry_run=not live)
    status, reason, halt = "completed", "", False
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
        halt = reason not in {"cumulative_request_or_model_limit", "unit_seconds_exhausted"}
    except BudgetStopped as error:
        reason = str(error)
        status, halt = "stopped", True
    checkpoint = Path(run_dir) / "checkpoints" / unit.unit_id
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "adapter.json").write_text(
        json.dumps(adapter.dump_state(), ensure_ascii=False), encoding="utf-8")
    _copy_tree(services.workspace.root, checkpoint / "workspace")
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
    sealed = Path(run_dir) / "sealed" / unit.unit_id
    sealed_files = _seal(services.workspace.root, sealed)
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


def _parser():
    parser = argparse.ArgumentParser(description="Run the frozen M1 or M2 schedule.")
    parser.add_argument("--study", required=True, choices=("m1", "m2"))
    parser.add_argument("--arm", action="append", default=[],
                        choices=("g1-main", "g1-strong", "g2", "g3"))
    parser.add_argument("--tasks", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--docker-image", default=os.environ.get("ALO_SANDBOX_IMAGE") or None)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.live == args.dry_run:
        parser.error("pass exactly one of --dry-run or --live")
    units = select_units(args.study, arms=args.arm, tasks=args.tasks)
    if not units:
        parser.error("no units match --arm and --tasks")
    try:
        results = run_schedule(units, args.run_dir, live=args.live, data_dir=args.data_dir,
                               docker_image=args.docker_image)
    except (PilotStopped, BudgetStopped) as error:
        print(str(error))
        return 2
    spent = 0
    for row in results:
        for amount in (row.get("charged_microcny") or {}).values():
            spent += amount
    print(json.dumps({"study": args.study, "units": len(results),
                      "charged_microcny": spent,
                      "charged_cny": spent / MICROCNY_PER_CNY}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
