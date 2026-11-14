"""ProgramBench transfer arms on the restricted Foxrouter pool.

Arms A1-A5 are separate run directories. PB is frozen unless
``--online-update`` is set: every task starts from that arm's initial
profile, and in-task observations do not move the assignment. ``--live`` is
the only path that reads ``FOXROUTER_API_KEY`` or opens a socket. A dry run
uses the fox-learn fixture gateway and does not score with Docker.

This experiment does not enter ``iter_r2_units`` and does not change
``r2_freeze_sha256``.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

from .adapters import adapter_for
from .comparison import pb_paired_report
from .errors import PilotStopped, ProtocolError
from .evaluate import PROGRAMBENCH_COMMIT, PROGRAMBENCH_TESTS_REVISION, programbench_command
from .fox_learn import FoxAloServices, dry_gateway, planner_for
from .fox_pool import (
    DEFAULT_READ_TIMEOUT_S,
    FOX_BY_INTERNAL,
    FOX_KEY_ENV,
    FOX_RESOURCE_PRIOR,
    model_settings,
    resolve_model,
)
from .fox_single import (
    FoxStopped,
    _append_jsonl,
    _live_secrets,
    _load_jsonl,
    _sum_amount,
    _sum_tokens,
    _transport_for,
    assert_fox_run_dir,
    redact_secrets,
)
from .freeze import PlannedUnit
from .pb_contract import (
    DEFAULT_ROLE_MAP,
    combine_pb_signal,
    is_reference_path,
    normalize_probe_owner,
    normalize_role_map,
    pb_code_violation,
    pb_command_violation,
    pb_dag,
    pb_prompt,
    submission_reference_scan,
)
from .run import load_frozen_transfer, load_public_payload, stage_for
from .state import TRANSFER_SCHEMA, StateContractError, export_transfer_package


EXPERIMENT = "fox-pb-v1"
SCHEMA = "fox-pb-v1"
UNIT_SCHEMA = "fox-pb-unit-v1"
PB_POOL = (
    "fox/deepseek-flash",
    "fox/kimi-k3.1-pro",
    "fox/mimo-v2.6-flash",
    "fox/glm-5.3-flash",
)
PB_POOL_SET = frozenset(PB_POOL)
BEST_SINGLE = "fox/deepseek-flash"
BEST_SINGLE_REPORT = "deepseek-flash"
ARMS = ("A1", "A2", "A3", "A4", "A5")
ALO_ARMS = ("A1", "A2", "A3")
G0_ARMS = ("A2", "A4")
REQUEST_CAPS = (40, 64, 80)
DEFAULT_REQUEST_CAP = 40
DEFAULT_UNIT_SECONDS = 2400
G0_TASKS = (
    ("kisielk__errcheck.dacab89", "easy"),
    ("raviqqe__muffet.a882908", "medium"),
    ("esubaalew__run.0fb9dec", "medium"),
)
_TERMINAL = frozenset({"completed", "failed", "blocked", "stopped"})
_LOCKED = (
    "arm", "models", "task_ids", "unit_ids", "request_cap", "online_update",
    "differential_signal", "role_map", "probe_script_owner", "unit_seconds",
    "read_timeout_s", "live", "frozen",
)


class PbStopped(FoxStopped):
    """The ProgramBench run stopped. The argument is a reason code."""


def _json_default(value):
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def decision_freeze(settings):
    """Run-level decision record. This is not ``r2_freeze_sha256``."""
    body = {
        "schema": "fox-pb-decision-freeze-v1",
        "role_map": dict(settings["role_map"]),
        "probe_script_owner": settings["probe_script_owner"],
        "differential_signal": bool(settings["differential_signal"]),
        "differential_kind": "blackbox_io",
        "request_cap": settings["request_cap"],
        "request_cap_candidates": list(REQUEST_CAPS),
        "online_update": bool(settings["online_update"]),
        "frozen": not bool(settings["online_update"]),
        "unit_seconds": settings["unit_seconds"],
        "read_timeout_s": settings["read_timeout_s"],
        "pool": list(PB_POOL),
        "best_single": BEST_SINGLE,
        "reference_mode": "111",
        "source_commit": PROGRAMBENCH_COMMIT,
        "tests_revision": PROGRAMBENCH_TESTS_REVISION,
    }
    digest = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
    return body, digest


def resolve_pb_pool(models=None):
    """The fox pool without qwen, and with every ``*omni*`` alias rejected."""
    if models in (None, "all"):
        names = list(PB_POOL)
    elif isinstance(models, str):
        names = [part.strip() for part in models.split(",") if part.strip()]
    else:
        names = list(models)
    if not names:
        raise PbStopped("fox_models_required")
    chosen = []
    for name in names:
        if not isinstance(name, str) or not name:
            raise PbStopped("unknown_fox_model")
        if "omni" in name.lower():
            raise PbStopped("omni_model_rejected")
        if "qwen" in name.lower():
            raise PbStopped("pb_pool_excludes_qwen")
        try:
            row = resolve_model(name)
        except ValueError as error:
            raise PbStopped(str(error)) from error
        if row.internal not in PB_POOL_SET:
            raise PbStopped("pb_pool_excludes:" + row.internal)
        if row.internal not in chosen:
            chosen.append(row.internal)
    order = {name: index for index, name in enumerate(PB_POOL)}
    chosen.sort(key=lambda name: order[name])
    return tuple(FOX_BY_INTERNAL[name] for name in chosen)


def default_pb_model_map():
    """Report names and bare ids map onto the PB fox internal ids."""
    mapping = {}
    for internal in PB_POOL:
        spec = FOX_BY_INTERNAL[internal]
        mapping[internal] = internal
        mapping[spec.report_name] = internal
        bare = internal.split("/", 1)[1]
        mapping[bare] = internal
    return mapping


def resolve_tasks(tasks, *, g0=False):
    if g0:
        return tuple(G0_TASKS)
    if isinstance(tasks, str):
        tasks = [part.strip() for part in tasks.split(",") if part.strip()]
    if not tasks:
        raise PbStopped("selection_required")
    catalog = {task_id: stratum for task_id, stratum in G0_TASKS}
    prefixes = {task_id.rsplit(".", 1)[0]: (task_id, stratum) for task_id, stratum in G0_TASKS}
    resolved = []
    for name in tasks:
        if not isinstance(name, str) or not name or "/" in name or name.startswith("."):
            raise PbStopped("unknown_pb_task")
        if name in catalog:
            resolved.append((name, catalog[name]))
        elif name in prefixes:
            resolved.append(prefixes[name])
        else:
            resolved.append((name, "unspecified"))
    if len({task_id for task_id, _stratum in resolved}) != len(resolved):
        raise PbStopped("duplicate_pb_task")
    return tuple(resolved)


def _check_request_cap(value):
    if type(value) is not int or value not in REQUEST_CAPS:
        raise PbStopped("invalid_pb_request_cap")
    return value


def _check_positive(value, code):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise PbStopped(code)
    return value


def _unit_seed(arm, task_id, position):
    digest = hashlib.sha256(f"{EXPERIMENT}|{arm}|{task_id}|{position}".encode()).hexdigest()
    return int(digest[:8], 16)


def _container_name(arm, unit_id):
    digest = hashlib.sha256(f"{arm}|{unit_id}".encode()).hexdigest()[:12]
    return f"alo-fox-pb-{arm.lower()}-{digest}"


def plan_units(arm, tasks, *, request_cap, g0=False):
    if arm not in ARMS:
        raise PbStopped("unknown_pb_arm")
    phase = "G0" if g0 else "pilot"
    units = []
    for position, (task_id, stratum) in enumerate(tasks):
        units.append(PlannedUnit(
            study="pb",
            arm=arm,
            track="ProgramBench",
            task_id=task_id,
            stratum=stratum,
            phase=phase,
            position=position,
            seed=_unit_seed(arm, task_id, position),
            request_cap=request_cap,
        ))
    return tuple(units)


class PbServices(FoxAloServices):
    """Fox pool services plus the ProgramBench reference-binary guard."""

    def __init__(self, *args, differential_enabled=False, signal_override=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.differential_enabled = bool(differential_enabled)
        self.signal_override = signal_override
        self.last_pb_signal = None

    def public_signal(self):
        if self.signal_override in {"pass", "fail", "unknown"}:
            compile_code = {"pass": 0, "fail": 1, "unknown": None}[self.signal_override]
        else:
            compile_code = None
        result = combine_pb_signal(
            compile_code,
            differential_enabled=self.differential_enabled,
            differential_signal="unknown",
        )
        self.last_pb_signal = result
        return result["signal"]

    def command(self, argv, timeout_s=30):
        violation = pb_command_violation(argv)
        if violation:
            raise ProtocolError(violation)
        return super().command(argv, timeout_s)

    def execute_code(self, language, code, timeout_s=30):
        violation = pb_code_violation(code)
        if violation:
            raise ProtocolError(violation)
        return super().execute_code(language, code, timeout_s)

    def read_file(self, path, start_line=None, end_line=None):
        if is_reference_path(path):
            raise ProtocolError("reference_binary_is_execute_only")
        return super().read_file(path, start_line=start_line, end_line=end_line)

    def write_file(self, path, content, append=False):
        if is_reference_path(path):
            raise ProtocolError("reference_binary_is_execute_only")
        return super().write_file(path, content, append=append)

    def edit_file(self, path, old_str, new_str):
        if is_reference_path(path):
            raise ProtocolError("reference_binary_is_execute_only")
        return super().edit_file(path, old_str, new_str)


def _profiles_of(document):
    if not isinstance(document, dict):
        raise PbStopped("frozen_transfer_source_unreadable")
    if document.get("schema") == TRANSFER_SCHEMA and isinstance(document.get("profiles"), dict):
        return document["profiles"]
    observations = document.get("observations")
    if isinstance(observations, dict):
        return observations
    raise PbStopped("frozen_transfer_source_unreadable")


def _observation_rows(rows):
    cleaned = []
    if not isinstance(rows, list):
        raise PbStopped("frozen_transfer_source_unreadable")
    for row in rows:
        if not isinstance(row, dict):
            raise PbStopped("frozen_transfer_source_unreadable")
        item = {
            key: value for key, value in row.items()
            if key not in {"role_topics", "task_type", "cell", "role_id"}
        }
        if "topics" not in item and isinstance(row.get("role_topics"), list):
            item["topics"] = list(row["role_topics"])
        cleaned.append(item)
    return cleaned


def _remap_profiles(profiles, model_map, allowed_models):
    allowed = list(allowed_models)
    allowed_set = set(allowed)
    remapped = {model: [] for model in allowed}
    dropped = []
    for source, rows in profiles.items():
        if not isinstance(source, str) or "omni" in source.lower():
            dropped.append(source if isinstance(source, str) else "<invalid>")
            continue
        target = model_map.get(source)
        if not isinstance(target, str) or "omni" in target.lower() or target not in allowed_set:
            dropped.append(source)
            continue
        remapped[target].extend(_observation_rows(rows))
    return remapped, dropped


@contextmanager
def open_pb_transfer(source_path, *, allowed_models, model_map=None, strip_quality=False):
    """Read-only import of a frozen SWE profile into the probe temporary store.

    The source bytes are hashed before and after. ``write_back`` on the
    yielded library raises and does not open the source for writing.
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise PbStopped("frozen_transfer_source_missing")
    source_bytes = source_path.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    try:
        document = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PbStopped("frozen_transfer_source_unreadable") from error
    mapping = default_pb_model_map()
    if model_map:
        if not isinstance(model_map, dict):
            raise PbStopped("invalid_model_map")
        for key, value in model_map.items():
            if "omni" in str(key).lower() or "omni" in str(value).lower():
                raise PbStopped("omni_model_rejected")
            if not isinstance(key, str) or not isinstance(value, str):
                raise PbStopped("invalid_model_map")
            mapping[key] = value
    remapped, dropped = _remap_profiles(_profiles_of(document), mapping, allowed_models)
    try:
        package = export_transfer_package(
            {"observations": remapped},
            allowed_models=tuple(allowed_models),
            task_family="swe-repair",
            strip_quality=bool(strip_quality),
        )
    except StateContractError as error:
        raise PbStopped("frozen_transfer_source_unreadable") from error
    temporary = tempfile.NamedTemporaryFile(prefix="alo-pb-transfer-", suffix=".json", delete=False)
    temporary_path = Path(temporary.name)
    try:
        temporary.write(_canonical(package).encode("utf-8"))
        temporary.close()
        with load_frozen_transfer(
                temporary_path, target_track="ProgramBench",
                allowed_models=tuple(allowed_models)) as library:
            if source_path.read_bytes() != source_bytes:
                raise PbStopped("frozen_transfer_source_mutated")
            library.pb_source_sha256 = source_digest
            library.pb_dropped_models = tuple(dropped)
            library.pb_strip_quality = bool(strip_quality)
            yield library
            if source_path.read_bytes() != source_bytes or hashlib.sha256(source_path.read_bytes()).hexdigest() != source_digest:
                raise PbStopped("frozen_transfer_source_mutated")
    finally:
        temporary_path.unlink(missing_ok=True)


def import_swe_profile(source_path, *, allowed_models, model_map=None, strip_quality=False):
    """Copy the imported observations out of the probe store. The source stays put."""
    with open_pb_transfer(
            source_path, allowed_models=allowed_models, model_map=model_map,
            strip_quality=strip_quality) as library:
        observations = copy.deepcopy(library.observations)
        digest = library.pb_source_sha256
        dropped = library.pb_dropped_models
    return {
        "observations": observations,
        "source_sha256": digest,
        "dropped_models": list(dropped),
        "strip_quality": bool(strip_quality),
    }


def pb_eval_plan(task_id, submission_dir, *, repo=None):
    """Official cleanroom build plus ``programbench eval``. This does not run it."""
    return {
        "task_id": task_id,
        "image_family": "task_cleanroom_v6",
        "source_commit": PROGRAMBENCH_COMMIT,
        "tests_revision": PROGRAMBENCH_TESTS_REVISION,
        "reference_mode": "111",
        "build": ["chmod", "+x", "./compile.sh"],
        "build_exec": ["./compile.sh"],
        "eval_command": programbench_command(repo, submission_dir) if repo else [],
        "submission_dir": str(submission_dir),
    }


def apply_pb_score(plan, *, execute, runner, public_signal, cost, wall_clock_s, submission_check):
    """Record the harness fields. Unknown stays null, not zero."""
    if public_signal == "pass":
        compile_success = True
    elif public_signal == "fail":
        compile_success = False
    else:
        compile_success = None
    body = {
        "resolved": None,
        "pass_rate": None,
        "compile_success": compile_success,
        "cost": cost,
        "wall_clock_s": wall_clock_s,
        "status": "not_executed",
        "submission_check": submission_check,
        "source_commit": plan.get("source_commit"),
        "image_family": plan.get("image_family"),
    }
    if isinstance(submission_check, dict) and submission_check.get("ok") is False:
        body["status"] = "submission_rejected"
        body["resolved"] = False
        return body
    if not execute:
        return body
    if runner is None:
        body["status"] = "unavailable"
        body["reason"] = "pb_eval_runner_required"
        return body
    result = runner(plan)
    if not isinstance(result, dict):
        body["status"] = "unavailable"
        return body
    if result.get("resolved") is True or result.get("resolved") is False:
        body["resolved"] = result["resolved"]
    rate = result.get("pass_rate")
    if not isinstance(rate, bool) and isinstance(rate, (int, float)):
        body["pass_rate"] = float(rate)
    if result.get("compile_success") is True or result.get("compile_success") is False:
        body["compile_success"] = result["compile_success"]
    body["status"] = result.get("status") or "finished"
    return body


def _adapter_config(arm, *, models, request_cap, seed, online_update, dag, prompt):
    if arm == "A4":
        if BEST_SINGLE not in models:
            raise PbStopped("best_single_not_in_pool")
        return "G1", {"fixed_model": BEST_SINGLE, "max_turns": request_cap}
    if arm == "A5" and BEST_SINGLE not in models:
        raise PbStopped("best_single_not_in_pool")
    config = {
        "planner_model": planner_for(tuple(models)),
        "max_turns": request_cap,
        "max_attempts": 1,
        "seed": seed,
        "probe": False,
        "organization_updates": bool(online_update) and arm in ALO_ARMS,
        "routing_mode": "fixed" if arm == "A5" else "thompson",
        "learn_quality": arm != "A3",
        "filter_resources_by_topics": True,
        "fixed_topology": True,
        "pb_mode": True,
        "pb_dag": dag,
        "pb_prompt": prompt,
        "policy_priors": {model: dict(FOX_RESOURCE_PRIOR) for model in models},
        "topology": "fixed_p1_p2",
    }
    if arm == "A3":
        config["export_mode"] = "resources"
    if arm == "A5":
        config["fixed_model"] = BEST_SINGLE
    return "G2", config


def _usage_block(calls):
    return {
        "prompt_tokens": _sum_tokens(calls, "prompt_tokens"),
        "completion_tokens": _sum_tokens(calls, "completion_tokens"),
        "reasoning_tokens": _sum_tokens(calls, "reasoning_tokens"),
        "total_tokens": _sum_tokens(calls, "total_tokens"),
        "cost": _sum_amount(calls, "cost"),
        "credit": _sum_amount(calls, "credit"),
        "quota": _sum_amount(calls, "quota"),
        "requests": len(calls),
    }


def _node_assignments(adapter):
    native = adapter.state.get("last_native_run") if hasattr(adapter, "state") else None
    nodes = native.get("node_results") if isinstance(native, dict) else None
    rows = []
    for node in nodes or ():
        if not isinstance(node, dict):
            continue
        rows.append({
            "node_id": node.get("subtask_id"),
            "role_id": node.get("role_id"),
            "model_id": node.get("model_id"),
        })
    return rows


def _first_assignment(adapter, arm):
    if arm == "A4":
        return BEST_SINGLE
    for node in _node_assignments(adapter):
        if node.get("node_id") == "p1" and isinstance(node.get("model_id"), str):
            return node["model_id"]
    nodes = _node_assignments(adapter)
    if nodes and isinstance(nodes[0].get("model_id"), str):
        return nodes[0]["model_id"]
    if arm == "A5":
        return BEST_SINGLE
    return None


def _manifest(settings, units):
    freeze_body, freeze_digest = decision_freeze(settings)
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "pool": "fox",
        "prices": "none",
        "arm": settings["arm"],
        "models": list(settings["models"]),
        "best_single": BEST_SINGLE,
        "task_ids": [unit.task_id for unit in units],
        "unit_ids": [unit.unit_id for unit in units],
        "request_cap": settings["request_cap"],
        "request_cap_candidates": list(REQUEST_CAPS),
        "online_update": bool(settings["online_update"]),
        "frozen": not bool(settings["online_update"]),
        "differential_signal": bool(settings["differential_signal"]),
        "differential_kind": "blackbox_io",
        "role_map": dict(settings["role_map"]),
        "probe_script_owner": settings["probe_script_owner"],
        "unit_seconds": settings["unit_seconds"],
        "read_timeout_s": settings["read_timeout_s"],
        "live": bool(settings["live"]),
        "g0": bool(settings["g0"]),
        "reference_guard": "execute-only-blackbox-v1",
        "reference_mode": "111",
        "source_commit": PROGRAMBENCH_COMMIT,
        "tests_revision": PROGRAMBENCH_TESTS_REVISION,
        "api_key_env": FOX_KEY_ENV,
        "api_key_recorded": False,
        "pb_freeze": freeze_body,
        "pb_freeze_sha256": freeze_digest,
        "swe_profile_sha256": settings.get("swe_profile_sha256"),
    }


def _write_manifest(run_dir, payload):
    path = Path(run_dir) / "manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("experiment") != EXPERIMENT:
            raise PbStopped("fox_run_dir_experiment_mismatch")
        for key in _LOCKED:
            if key in existing and existing.get(key) != payload.get(key):
                raise PbStopped("fox_run_settings_mismatch")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(payload) + "\n", encoding="utf-8")


def _done_ids(run_dir):
    done = set()
    for row in _load_jsonl(Path(run_dir) / "records.jsonl"):
        unit_id = row.get("unit_id")
        if isinstance(unit_id, str) and row.get("status") in _TERMINAL:
            done.add(unit_id)
    return done


def _initial_observations(settings):
    if settings["arm"] == "A2":
        return {}, None
    if settings["arm"] in {"A4", "A5"}:
        return {}, None
    path = settings.get("swe_profile")
    if not path:
        raise PbStopped("swe_profile_required")
    imported = import_swe_profile(
        path,
        allowed_models=tuple(settings["models"]),
        model_map=settings.get("model_map"),
        strip_quality=settings["arm"] == "A3",
    )
    return imported["observations"], imported["source_sha256"]


def _carried_path(run_dir):
    return Path(run_dir) / "carried_profile.json"


def _load_carried(run_dir, models):
    path = _carried_path(run_dir)
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    observations = document.get("observations")
    if not isinstance(observations, dict) or set(observations) - set(models):
        raise PbStopped("carried_profile_invalid")
    return observations


def _save_carried(run_dir, observations, models):
    body = {"schema": "fox-pb-carried-v1", "models": list(models), "observations": observations}
    _carried_path(run_dir).write_text(_canonical(body) + "\n", encoding="utf-8")


def run_arm(run_dir, *, arm, tasks=None, g0=False, models="all", request_cap=DEFAULT_REQUEST_CAP,
            unit_seconds=DEFAULT_UNIT_SECONDS, read_timeout_s=DEFAULT_READ_TIMEOUT_S,
            online_update=False, differential_signal=False, role_map=None,
            probe_script_owner="p1", swe_profile=None, model_map=None, live=False,
            sender=None, execute=False, scorer=None, signal_override=None, clock=None):
    """Run one arm in its own directory. Completed unit ids are skipped."""
    if arm not in ARMS:
        raise PbStopped("unknown_pb_arm")
    if arm in {"A1", "A3"} and not swe_profile:
        raise PbStopped("swe_profile_required")
    specs = resolve_pb_pool(models)
    model_names = tuple(row.internal for row in specs)
    request_cap = _check_request_cap(request_cap)
    unit_seconds = _check_positive(unit_seconds, "invalid_unit_seconds")
    read_timeout_s = _check_positive(read_timeout_s, "invalid_read_timeout")
    try:
        mapping = normalize_role_map(role_map)
        owner = normalize_probe_owner(probe_script_owner)
    except ValueError as error:
        raise PbStopped(str(error)) from error
    selected = resolve_tasks(tasks, g0=g0)
    units = plan_units(arm, selected, request_cap=request_cap, g0=g0)
    run_dir = assert_fox_run_dir(run_dir)
    dag = pb_dag(role_map=mapping, probe_script_owner=owner)
    prompt = pb_prompt(role_map=mapping, probe_script_owner=owner)
    settings = {
        "arm": arm,
        "models": model_names,
        "request_cap": request_cap,
        "unit_seconds": unit_seconds,
        "read_timeout_s": read_timeout_s,
        "online_update": bool(online_update),
        "differential_signal": bool(differential_signal),
        "role_map": mapping,
        "probe_script_owner": owner,
        "live": bool(live),
        "g0": bool(g0),
        "swe_profile": swe_profile,
        "model_map": model_map,
    }
    initial_observations, profile_sha = ({}, None)
    if arm in {"A1", "A3"}:
        initial_observations, profile_sha = _initial_observations(settings)
    settings["swe_profile_sha256"] = profile_sha
    _write_manifest(run_dir, _manifest(settings, units))
    if profile_sha and (run_dir / "manifest.json").is_file():
        # The source digest is recorded once. A later resume must see the same file.
        existing = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("swe_profile_sha256") not in {None, profile_sha}:
            raise PbStopped("swe_profile_changed")
    transport = _transport_for(
        live=live,
        sender=None if live else (sender or dry_gateway),
        timeout_s=read_timeout_s,
        sleep=time.sleep,
        clock=clock or (lambda: datetime.now(timezone.utc)),
        max_attempts=4,
    )
    limits = {"requests": request_cap, "tokens": 900_000, "tools": 120, "seconds": unit_seconds}
    done = _done_ids(run_dir)
    secrets = _live_secrets()
    written = []
    source_bytes = Path(swe_profile).read_bytes() if swe_profile and Path(swe_profile).is_file() else None
    for unit in units:
        if unit.unit_id in done:
            continue
        record = _run_unit(
            unit, run_dir, settings=settings, specs=specs, model_names=model_names,
            transport=transport, limits=limits, dag=dag, prompt=prompt,
            initial_observations=initial_observations, live=live, execute=execute,
            scorer=scorer, signal_override=signal_override,
        )
        calls = record.pop("_calls", []) or []
        record = redact_secrets(record, secrets)
        _append_jsonl(run_dir / "records.jsonl", record)
        for call in calls:
            call = redact_secrets(dict(call), secrets)
            call.pop("text", None)
            call["unit_id"] = record["unit_id"]
            _append_jsonl(run_dir / "calls.jsonl", call)
        written.append(record)
    if source_bytes is not None and Path(swe_profile).read_bytes() != source_bytes:
        raise PbStopped("frozen_transfer_source_mutated")
    return {
        "experiment": EXPERIMENT,
        "arm": arm,
        "run_dir": str(run_dir),
        "wrote": len(written),
        "units": len(units),
        "skipped": len(units) - len(written),
        "paid_calls": 0 if not live else None,
        "models": list(model_names),
    }


def _run_unit(unit, run_dir, *, settings, specs, model_names, transport, limits, dag, prompt,
              initial_observations, live, execute, scorer, signal_override):
    arm = settings["arm"]
    record = {
        "schema": UNIT_SCHEMA,
        "experiment": EXPERIMENT,
        "unit_id": unit.unit_id,
        "arm": arm,
        "track": "ProgramBench",
        "task_id": unit.task_id,
        "stratum": unit.stratum,
        "phase": unit.phase,
        "position": unit.position,
        "seed": unit.seed,
        "request_cap": settings["request_cap"],
        "online_update": bool(settings["online_update"]),
        "frozen": not bool(settings["online_update"]),
        "differential_signal": bool(settings["differential_signal"]),
        "role_map": dict(settings["role_map"]),
        "probe_script_owner": settings["probe_script_owner"],
        "models": list(model_names),
        "pool": "fox",
        "prices": "none",
        "container_name": _container_name(arm, unit.unit_id),
        "run_dir": str(run_dir),
    }
    state_dir = Path(run_dir) / "units" / unit.unit_id / "state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    services = PbServices(
        state_dir, specs, settings_by_model={
            row.internal: model_settings(row) for row in specs
        },
        transport=transport, limits=limits, container_name=record["container_name"],
        differential_enabled=settings["differential_signal"],
        signal_override=signal_override,
    )
    group, config = _adapter_config(
        arm, models=model_names, request_cap=settings["request_cap"], seed=unit.seed,
        online_update=settings["online_update"], dag=dag, prompt=prompt,
    )
    adapter = adapter_for(group, config)
    observations = {}
    if arm in ALO_ARMS:
        if settings["online_update"]:
            carried = _load_carried(run_dir, model_names)
            observations = copy.deepcopy(carried if carried is not None else initial_observations)
        else:
            observations = copy.deepcopy(initial_observations)
        if hasattr(adapter, "state"):
            adapter.state["observations"] = observations
    started = time.monotonic()
    payload = load_public_payload(None, "ProgramBench", unit.task_id)
    try:
        stage = stage_for(unit, payload, dry_run=not live)
    except PilotStopped as error:
        record.update(status="blocked", reason=str(error), harness_resolved=None,
                      pass_rate=None, compile_success=None, cost=None, wall_clock_s=0,
                      first_assignment=None)
        record["_calls"] = []
        return record
    status, reason = "completed", ""
    try:
        outcome = adapter.run_stage(stage, services)
        if isinstance(outcome, dict) and outcome.get("status") in {"completed", "failed"}:
            status = outcome["status"]
    except (PilotStopped, FoxStopped, PbStopped) as error:
        status, reason = "stopped", str(error)
    wall = time.monotonic() - started
    signal = None
    try:
        signal = services.public_signal()
    except Exception:
        signal = "unknown"
    submission = submission_reference_scan(services.workspace.root)
    usage = _usage_block(services.calls)
    plan = pb_eval_plan(unit.task_id, state_dir / "workspace")
    scored = apply_pb_score(
        plan, execute=bool(execute), runner=scorer, public_signal=signal,
        cost=usage["cost"], wall_clock_s=wall, submission_check=submission,
    )
    if settings["online_update"] and arm in ALO_ARMS and hasattr(adapter, "state"):
        _save_carried(run_dir, adapter.state.get("observations") or {}, model_names)
    checkpoint = Path(run_dir) / "checkpoints" / unit.unit_id
    checkpoint.mkdir(parents=True, exist_ok=True)
    if hasattr(adapter, "dump_state"):
        (checkpoint / "adapter.json").write_text(
            json.dumps(adapter.dump_state(), ensure_ascii=False, default=_json_default),
            encoding="utf-8")
    record.update(
        status=status,
        reason=reason,
        first_assignment=_first_assignment(adapter, arm) if hasattr(adapter, "state") or arm == "A4" else BEST_SINGLE if arm == "A4" else None,
        node_assignments=_node_assignments(adapter) if hasattr(adapter, "state") else [],
        harness_resolved=scored["resolved"],
        harness_status=scored["status"],
        pass_rate=scored["pass_rate"],
        compile_success=scored["compile_success"],
        submission_ok=submission.get("ok"),
        cost=usage["cost"],
        credit=usage["credit"],
        quota=usage["quota"],
        wall_clock_s=wall,
        usage=usage,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        models_used=sorted({
            call.get("internal_model") for call in services.calls
            if isinstance(call.get("internal_model"), str)
        }),
    )
    if arm == "A4":
        record["first_assignment"] = BEST_SINGLE
    record["_calls"] = list(services.calls)
    return record


def run_g0(root, **kwargs):
    """A2 and A4 on the three G0 tasks. Each arm has its own directory."""
    root = Path(root)
    kwargs = dict(kwargs)
    kwargs.pop("arm", None)
    kwargs["g0"] = True
    kwargs["tasks"] = None
    results = []
    for arm in G0_ARMS:
        results.append(run_arm(root / arm, arm=arm, **kwargs))
    return {
        "experiment": EXPERIMENT,
        "arms": list(G0_ARMS),
        "units": sum(item["units"] for item in results),
        "wrote": sum(item["wrote"] for item in results),
        "runs": results,
        "paid_calls": 0 if not kwargs.get("live") else None,
    }


def report_runs(run_dirs):
    rows = []
    for run_dir in run_dirs:
        rows.extend(_load_jsonl(Path(run_dir) / "records.jsonl"))
    return pb_paired_report(rows)


def _add_common(parser, *, arm_required):
    if arm_required:
        parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--g0", action="store_true")
    parser.add_argument("--models", default="all")
    parser.add_argument("--request-cap", type=int, default=DEFAULT_REQUEST_CAP)
    parser.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT_S)
    parser.add_argument("--unit-seconds", type=float, default=DEFAULT_UNIT_SECONDS)
    parser.add_argument("--online-update", action="store_true")
    parser.add_argument("--differential-signal", action="store_true")
    parser.add_argument("--probe-script-owner", default="p1", choices=("p1", "p2"))
    parser.add_argument("--role-map", default="s1=p1,s2=p2")
    parser.add_argument("--swe-profile", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--execute", action="store_true")


def _kwargs_from_args(args):
    return {
        "tasks": args.tasks,
        "g0": bool(args.g0),
        "models": args.models,
        "request_cap": args.request_cap,
        "unit_seconds": args.unit_seconds,
        "read_timeout_s": args.read_timeout,
        "online_update": bool(args.online_update),
        "differential_signal": bool(args.differential_signal),
        "role_map": args.role_map,
        "probe_script_owner": args.probe_script_owner,
        "swe_profile": args.swe_profile,
        "live": bool(args.live),
        "execute": bool(args.execute and args.live),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ProgramBench transfer arms on the fox pool. No API call unless --live.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run one arm. Pass --dry-run or --live.")
    run.add_argument("--run-dir", required=True)
    _add_common(run, arm_required=True)
    g0 = sub.add_parser("g0", help="Dry or live A2 and A4 on the three G0 tasks.")
    g0.add_argument("--root", required=True)
    _add_common(g0, arm_required=False)
    report = sub.add_parser("report", help="Paired PB report from one or more run directories.")
    report.add_argument("--run-dir", action="append", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            print(_canonical(report_runs(args.run_dir)))
            return 0
        if args.live == args.dry_run:
            parser.error("pass exactly one of --dry-run or --live")
        if args.command == "g0":
            result = run_g0(args.root, **_kwargs_from_args(args))
        else:
            if not args.g0 and not args.tasks:
                parser.error("pass --tasks or --g0")
            result = run_arm(args.run_dir, arm=args.arm, **_kwargs_from_args(args))
        print(_canonical(result))
        return 0
    except (PbStopped, FoxStopped, PilotStopped) as error:
        print(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
