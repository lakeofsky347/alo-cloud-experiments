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
    MAX_BLOCKED_FRACTION, MAX_UNKNOWN_OUTCOMES, select_units,
)
from .ledger import restart_counts, restart_unit
from .images import resolve_task_image
from .prices import ceiling_for_messages, charge_microcny
from .sandbox import SWE_TESTBED_PATH, DockerSandbox
from .workspace_prep import (
    apply_patch, release_unreadable_executable, resolve_image_head, seed_image_tree,
)
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


def _touched(request_id, unit_id, restart_count=0):
    """True when this id belongs to the attempt that would be started now.

    A restarted unit keeps its old rows for the CNY total. Those rows use the
    previous id prefix, so they do not look like a partial retry of the new
    attempt. A crash during the new attempt still blocks a silent retry.
    """
    generation = f"-r{restart_count}" if restart_count else ""
    return (request_id.startswith(f"req-{unit_id}{generation}-")
            or request_id.startswith(f"tool-{unit_id}{generation}-"))


def missing_payloads(units, data_dir):
    missing = []
    for unit in units:
        if load_public_payload(data_dir, unit.track, unit.task_id) is None:
            missing.append(f"{unit.track}/{unit.task_id}")
    return missing


def _unknown_log(run_dir):
    return Path(run_dir) / "unknown_outcomes.jsonl"


def _unknown_count(run_dir):
    path = _unknown_log(run_dir)
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _note_unknown(run_dir, record):
    path = _unknown_log(run_dir)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "unit_id": record.get("unit_id"),
            "reason": record.get("reason"),
            "request_id": record.get("unknown_request_id"),
        }, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return _unknown_count(run_dir)


def run_schedule(units, run_dir, *, live, data_dir=None, docker_image=None, sender=None,
                 git_runner=None, sandbox_memory=None, sandbox_pids=None, image_seeder=None):
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
    try:
        for unit in units:
            if unit.unit_id in done:
                results.append(done[unit.unit_id])
                _stop_if_blocked(results)
                continue
            if any(_touched(request_id, unit.unit_id, restart_counts(run_dir).get(unit.unit_id, 0))
                   for request_id in known_ids):
                raise BudgetStopped("partial_unit_not_retried")
            retried = False
            while True:
                record = _execute_unit(
                    unit, run_dir, ledger, live=live, data_dir=data_dir,
                    docker_image=docker_image, sender=sender, git_runner=git_runner,
                    sandbox_memory=sandbox_memory, sandbox_pids=sandbox_pids,
                    image_seeder=image_seeder, unknown_so_far=_unknown_count(run_dir))
                _append_record(records_path, record)
                if record.get("charged_unknown"):
                    count = _note_unknown(run_dir, record)
                    if count > MAX_UNKNOWN_OUTCOMES:
                        record["halt_schedule"] = True
                        record["reason"] = "unknown_outcome_threshold"
                known_ids = ledger.request_ids()
                if (record.get("charged_unknown") and not retried and record.get("restart_count", 0) == 0
                        and not record.get("halt_schedule")):
                    restart_unit(run_dir, unit.unit_id, reason="charged unknown; restart once", ledger=ledger)
                    retried = True
                    continue
                break
            results.append(record)
            _stop_if_blocked(results)
            if record.get("halt_schedule"):
                raise PilotStopped(record.get("reason") or "schedule_stopped")
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


def blocked_fraction_exceeded(records, *, maximum=None):
    """True when blocked units are more than the allowed share of a large enough sample."""
    finished = [row for row in records if isinstance(row, dict) and row.get("status")]
    if len(finished) < BLOCKED_FRACTION_MIN_UNITS:
        return False
    blocked = sum(1 for row in finished if row.get("status") == "blocked")
    return blocked / len(finished) > _max_blocked_fraction(maximum)


def format_blocked_report(records, *, maximum=None):
    """One histogram line per blocked reason, then whether the schedule must stop."""
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


def _stop_if_blocked(records):
    exceeded = blocked_fraction_exceeded(records)
    if not exceeded:
        return
    text, _exceeded = format_blocked_report(records)
    print(text)
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
            parent_patch = Path(run_dir) / "checkpoints" / unit.parent_unit_id / "model.patch"
            if unit.parent_unit_id and parent_patch.is_file():
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
                  unknown_so_far=0):
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
    restart_count = restart_counts(run_dir).get(unit.unit_id, 0)
    record["restart_count"] = restart_count
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
        restart_count=1 if restart_count >= 1 else 0,
    )
    payload = load_public_payload(data_dir, unit.track, unit.task_id)
    repo = _public_field(payload, "repo")
    base_commit = _public_field(payload, "base_commit")
    if repo:
        record["repo"] = repo
    if base_commit:
        record["base_commit"] = base_commit
    image = resolve_task_image(unit.track, unit.task_id, fallback=docker_image)
    record["sandbox_image"] = image
    adapter = _build_adapter(unit)
    parent_note = _apply_parent(unit, adapter, services, run_dir)
    if parent_note == "b0_checkpoint_missing":
        record.update(status="blocked", reason=parent_note, charged_microcny={}, sealed_files=0)
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
    _attach_sandbox(services, unit, image, live=live, memory=sandbox_memory, pids=sandbox_pids)
    record["sandbox_workdir"] = getattr(services.sandbox, "workdir", None) if services.sandbox else None
    record["preserve_executable_in_image"] = bool(getattr(services, "preserve_image_executable", False))
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
        halt = reason not in {
            "cumulative_request_or_model_limit", "unit_seconds_exhausted", "provider_outcome_charged",
        }
        if reason == "provider_outcome_charged":
            record["charged_unknown"] = True
            if restart_count >= 1:
                status = "blocked"
                reason = "second_unknown_interruption"
            halt = unknown_so_far + 1 > MAX_UNKNOWN_OUTCOMES
        elif reason == "provider_outcome_unknown" and restart_count >= 1:
            status = "blocked"
            reason = "second_unknown_interruption"
            halt = True
    except BudgetStopped as error:
        reason = str(error)
        status, halt = "stopped", True
    sealed_files = _persist_compact(unit, services, adapter, run_dir, record)
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
        dest = (Path(run_dir) / "sealed" / "programbench" / unit.unit_id / unit.task_id
                / "submission.tar.gz")
        return _pack_submission(workspace, dest)
    return 0


def _delete_live_unit(run_dir, unit_id):
    shutil.rmtree(Path(run_dir) / "units" / unit_id, ignore_errors=True)


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
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--docker-image", default=os.environ.get("ALO_SANDBOX_IMAGE") or None)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.blocked_report:
        records = list(_done_ids(Path(args.run_dir) / "records.jsonl").values())
        text, exceeded = format_blocked_report(records)
        print(text)
        return 1 if exceeded else 0
    if args.study not in {"m1", "m2"}:
        parser.error("--study is required")
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
