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
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import yaml

from .adapters import adapter_for
from .comparison import pb_paired_report
from .errors import IntegrationNotConfigured, PilotStopped, ProtocolError
from .collect import _pack_submission
from .evaluate import (
    PROGRAMBENCH_COMMIT, PROGRAMBENCH_TESTS_REVISION, _container_start_failed, _git_head,
    _harness_run, harness_cwd, programbench_command, programbench_env,
)
from .images import programbench_instance_image
from .fox_learn import FoxAloServices, dry_gateway, planner_for
from .fox_pool import (
    DEFAULT_READ_TIMEOUT_S,
    DEEPSEEK_EXACT_RETURN,
    DEEPSEEK_EXACT_RETURN_VERSION,
    FOX_BY_INTERNAL,
    FoxModel,
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
    key_configured,
    redact_secrets,
)
from .freeze import PlannedUnit
from .pb_contract import (
    DEFAULT_ROLE_MAP,
    PB_SIGNAL_CANDIDATE,
    blackbox_probe_argv,
    combine_pb_signal,
    is_reference_path,
    normalize_probe_owner,
    normalize_role_map,
    pb_code_violation,
    pb_command_violation,
    pb_dag,
    pb_prompt,
    run_blackbox_differential,
    SubmissionScanDeadline,
    submission_reference_scan,
    submission_reference_scan_bounded,
)
from .sandbox import DockerSandbox
from .workspace_prep import seed_image_tree
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
# Opt-in slot substitution for one PB run. Unset leaves PB_POOL, the default
# decision freeze, and every other track unchanged. ``glm-flashx`` replaces the
# glm slot with ``glm-5.3-flashx`` served by Zhipu's official API
# (open.bigmodel.cn), keyed by ``GLM_API_KEY``. 2026-11-16: Foxrouter
# ``glm-5.3-flash`` answered HTTP 503 ``upstream_unavailable`` on every probe
# from 15:53 SGT and Foxrouter ``glm-5.3-flashx`` timed out at 16:28 and 16:30.
# The other three slots stay on Foxrouter. The substitute row lives here, not in
# ``FOX_MODELS``, so SWE / learn pools do not gain it.
ZHIPU_API_BASE = "https://open.bigmodel.cn/api/paas/v4"
GLM_KEY_ENV = "GLM_API_KEY"
GLM_FLASHX = FoxModel("fox/glm-5.3-flashx", "glm-5.3-flashx", "fox/glm-5.3-flashx", 128_000,
                      api_base=ZHIPU_API_BASE, key_env=GLM_KEY_ENV)
PB_SUBSTITUTE_MODELS = {GLM_FLASHX.internal: GLM_FLASHX}
PB_POOL_SUBSTITUTIONS = {
    "glm-flashx": {"from": "fox/glm-5.3-flash", "to": GLM_FLASHX.internal},
}


def pb_pool_for(substitute=None):
    """PB_POOL, or PB_POOL with one slot replaced by a named substitution."""
    if substitute is None:
        return PB_POOL
    rule = PB_POOL_SUBSTITUTIONS.get(substitute)
    if rule is None:
        raise PbStopped("unknown_pool_substitute")
    return tuple(rule["to"] if name == rule["from"] else name for name in PB_POOL)


def _pb_spec(internal):
    return PB_SUBSTITUTE_MODELS.get(internal) or FOX_BY_INTERNAL[internal]
BEST_SINGLE = "fox/deepseek-flash"
BEST_SINGLE_REPORT = "deepseek-flash"
ARMS = ("A1", "A2", "A3", "A4", "A5")
ALO_ARMS = ("A1", "A2", "A3")
G0_ARMS = ("A2", "A4")
REQUEST_CAPS = (40, 64, 80)
DEFAULT_REQUEST_CAP = 40
DEFAULT_UNIT_SECONDS = 2400
# The end-of-unit submission scan is bounded by the unit deadline, but never to
# less than this many seconds after it starts. A unit whose agent loop used the
# whole budget still gets its (normally sub-second) scan, so ordinary results
# do not change; only a scan that would run for minutes is cut off.
SCAN_MIN_SECONDS = 300
# Every model call is appended here as it returns (``calls.jsonl`` is still
# written once per finished unit). A killed or hung unit keeps its spend.
CALL_JOURNAL = "call_journal.jsonl"
# G0 / M18: p1 and p2 each get this many model requests. The unit request
# cap stays the upper bound. Unset on every other run, so the default
# decision freeze does not gain a field.
M18_NODE_REQUEST_CAP = 20
G0_TASKS = (
    ("kisielk__errcheck.dacab89", "easy"),
    ("raviqqe__muffet.a882908", "medium"),
    ("esubaalew__run.0fb9dec", "medium"),
)
_TERMINAL = frozenset({"completed", "failed", "blocked", "stopped"})
_LOCKED = (
    "arm", "models", "task_ids", "unit_ids", "request_cap", "node_request_cap", "pool_substitution",
    "exact_returned_opt_in",
    "online_update", "differential_signal", "role_map", "probe_script_owner",
    "unit_seconds", "read_timeout_s", "live", "frozen", "data_dir",
    "programbench_repo",
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
        "pool": list(pb_pool_for(settings.get("pool_substitute"))),
        "best_single": BEST_SINGLE,
        "reference_mode": "111",
        "source_commit": PROGRAMBENCH_COMMIT,
        "tests_revision": PROGRAMBENCH_TESTS_REVISION,
    }
    # Present only when the run sets it. The default document stays byte-identical.
    substitute = settings.get("pool_substitute")
    if substitute is not None:
        rule = PB_POOL_SUBSTITUTIONS[substitute]
        spec = PB_SUBSTITUTE_MODELS[rule["to"]]
        body["pool_substitution"] = {"name": substitute, "from": rule["from"], "to": rule["to"],
                                     "upstream": spec.upstream, "api_base": spec.api_base,
                                     "key_env": spec.key_env}
    node_cap = settings.get("node_request_cap")
    if node_cap is not None:
        body["node_request_cap"] = node_cap
    # Present only when the run sets it. The default document stays byte-identical.
    # The block records an explicit opt-in. ``identity`` stays ``uncertain``.
    opt_in = settings.get("exact_returned_opt_in")
    if opt_in is not None:
        body["exact_returned_opt_in"] = dict(opt_in)
    digest = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
    return body, digest


def resolve_pb_pool(models=None, substitute=None):
    """The fox pool without qwen, and with every ``*omni*`` alias rejected."""
    pool = pb_pool_for(substitute)
    pool_set = frozenset(pool)
    if models in (None, "all"):
        names = list(pool)
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
        row = PB_SUBSTITUTE_MODELS.get(name)
        if row is None:
            try:
                row = resolve_model(name)
            except ValueError as error:
                raise PbStopped(str(error)) from error
        if row.internal not in pool_set:
            raise PbStopped("pb_pool_excludes:" + row.internal)
        if row.internal not in chosen:
            chosen.append(row.internal)
    order = {name: index for index, name in enumerate(pool)}
    chosen.sort(key=lambda name: order[name])
    return tuple(_pb_spec(name) for name in chosen)


def default_pb_model_map(substitute=None):
    """Report names and bare ids map onto the PB fox internal ids.

    With a substitution, the replaced model's names map onto the substitute,
    so a frozen profile's rows for that slot carry over. The file is not edited.
    """
    mapping = {}
    for internal in PB_POOL:
        spec = FOX_BY_INTERNAL[internal]
        mapping[internal] = internal
        mapping[spec.report_name] = internal
        bare = internal.split("/", 1)[1]
        mapping[bare] = internal
    if substitute is not None:
        rule = PB_POOL_SUBSTITUTIONS.get(substitute)
        if rule is None:
            raise PbStopped("unknown_pool_substitute")
        source = FOX_BY_INTERNAL[rule["from"]]
        for key in (rule["from"], source.report_name, rule["from"].split("/", 1)[1]):
            mapping[key] = rule["to"]
        mapping[rule["to"]] = rule["to"]
        mapping[rule["to"].split("/", 1)[1]] = rule["to"]
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


def _check_node_request_cap(value, request_cap):
    """Per-node cap. ``None`` leaves the shared unit cap as the only limit."""
    if value is None:
        return None
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise PbStopped("invalid_node_request_cap")
    if value > request_cap:
        raise PbStopped("node_request_cap_above_total")
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


def _is_compile_invocation(argv):
    if not argv:
        return False
    base = Path(argv[0]).name
    if base == "compile.sh":
        return True
    if base in {"sh", "bash", "dash"} and len(argv) >= 3 and argv[1] == "-c":
        return "compile.sh" in argv[2]
    return False


def _side_effect_digest(root):
    """Hash readable workspace files. The execute-only reference is skipped."""
    root = Path(root)
    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "executable" or relative.startswith(".git/"):
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()


def _parse_pb_score_text(text):
    if not isinstance(text, str) or "{" not in text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    parsed = {}
    rate = value.get("pass_rate")
    passed, total = value.get("passed"), value.get("total")
    if rate is None and type(passed) is int and type(total) is int and total > 0:
        rate = passed / total
    if not isinstance(rate, bool) and isinstance(rate, (int, float)):
        parsed["pass_rate"] = float(rate)
        parsed["resolved"] = float(rate) >= 1.0
    if value.get("resolved") is True or value.get("resolved") is False:
        parsed["resolved"] = value["resolved"]
    if value.get("compile_success") is True or value.get("compile_success") is False:
        parsed["compile_success"] = value["compile_success"]
    return parsed or None


def run_official_pb_eval(plan, *, harness_run=None, git_head=None):
    """Pack one submission and run ``programbench eval`` when the checkout matches.

    A missing checkout, the wrong commit, or a missing ``uv`` is ``unavailable``.
    That is not a score of zero. Tests pass ``harness_run`` and do not start Docker.
    """
    repo = plan.get("repo") if isinstance(plan, dict) else None
    if not repo:
        return {"status": "unavailable", "reason": "programbench_checkout_required",
                "resolved": None, "pass_rate": None, "compile_success": None}
    head = (git_head or _git_head)(repo)
    if head != PROGRAMBENCH_COMMIT:
        return {"status": "unavailable", "reason": "programbench_commit_mismatch",
                "observed_commit": head, "resolved": None, "pass_rate": None,
                "compile_success": None}
    sealed = plan.get("sealed_dir")
    workspace = plan.get("submission_dir")
    task_id = plan.get("task_id")
    command = list(plan.get("eval_command") or [])
    if not sealed or not workspace or not task_id or not command:
        return {"status": "unavailable", "reason": "programbench_submissions_missing",
                "resolved": None, "pass_rate": None, "compile_success": None}
    dest = Path(sealed) / task_id / "submission.tar.gz"
    _pack_submission(workspace, dest)
    if harness_run is None and shutil.which("uv") is None:
        return {"status": "unavailable", "reason": "uv_missing", "resolved": None,
                "pass_rate": None, "compile_success": None, "command": command}
    run = harness_run or _harness_run
    run_dir = plan.get("run_dir") or sealed
    completed = run(command, cwd=harness_cwd(run_dir), env=programbench_env())
    parsed = _parse_pb_score_text(getattr(completed, "stdout", "") or "")
    if parsed is None:
        parsed = _parse_pb_score_text(getattr(completed, "stderr", "") or "")
    failed = completed.returncode != 0 or _container_start_failed(completed)
    body = {
        "status": "harness_failed" if failed and not parsed else "finished",
        "returncode": completed.returncode,
        "resolved": None,
        "pass_rate": None,
        "compile_success": None,
        "command": command,
    }
    if parsed:
        body.update(parsed)
    elif failed:
        body["status"] = "harness_failed"
    official = _read_official_eval_json(plan.get("sealed_dir"), plan.get("run_dir"))
    if official:
        body.update(official)
        if body.get("status") == "harness_failed" and official.get("pass_rate") is not None:
            body["status"] = "finished"
    return body


def _score_eval_document(data):
    """Pass rate from an official ``*.eval.json``. Test names are not kept."""
    if not isinstance(data, dict):
        return None
    results = data.get("test_results")
    error = data.get("error_code")
    compile_success = not (isinstance(error, str) and "compile" in error)
    if not isinstance(results, list) or not results:
        if error:
            return {"resolved": False, "pass_rate": 0.0, "compile_success": compile_success}
        return None
    passed = sum(1 for item in results if isinstance(item, dict) and item.get("status") == "passed")
    rate = passed / len(results)
    return {
        "pass_rate": rate,
        "resolved": rate == 1.0 and not error,
        "compile_success": compile_success,
    }


def _read_official_eval_json(*roots):
    for root in roots:
        if not root:
            continue
        path = Path(root)
        if not path.is_dir():
            continue
        for candidate in sorted(path.rglob("*.eval.json")):
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            parsed = _score_eval_document(data)
            if parsed:
                return parsed
    return None


def official_pb_scorer(plan):
    """Scorer ``main`` passes for ``--live --execute``. It does not call a model."""
    return run_official_pb_eval(plan)


_TASK_YAML_FIELDS = ("repository", "commit", "language", "difficulty")
_DOC_SUFFIXES = {".md", ".txt", ".rst", ".pod"}


def _task_yaml_candidates(task_id, *, repo, data_dir):
    found = []
    if repo:
        root = Path(repo)
        found.append(root / "src" / "programbench" / "data" / "tasks" / task_id / "task.yaml")
    if data_dir:
        base = Path(data_dir)
        found.append(base / "ProgramBench" / f"{task_id}.json")
        found.append(base / task_id / "task.yaml")
        found.append(base / "tasks" / task_id / "task.yaml")
        found.append(base / "src" / "programbench" / "data" / "tasks" / task_id / "task.yaml")
    return found


def _public_task_body(task_id, raw):
    """Public task fields only. tests.json and eval hashes stay out."""
    body = {
        "task_id": task_id,
        "reference_executable": "./executable",
        "reference_mode": "111",
        "docs": [],
    }
    if not isinstance(raw, dict):
        return body
    for key in _TASK_YAML_FIELDS:
        value = raw.get(key)
        if isinstance(value, str) and value:
            body[key] = value
    statement = raw.get("problem_statement")
    if isinstance(statement, str) and statement.strip():
        body["problem_statement"] = statement
    else:
        name = body.get("repository") or task_id
        language = body.get("language")
        parts = [f"Rebuild {name} from its documentation and the reference program."]
        if language:
            parts.append(f"Language: {language}.")
        parts.append(
            "The docs are in the workspace. The reference program is ./executable. "
            "It is execute-only (mode 111).")
        body["problem_statement"] = " ".join(parts)
    return body


def load_pb_public_payload(task_id, *, repo=None, data_dir=None):
    """Load task.yaml or a staged public JSON. Hidden tests are not read."""
    for path in _task_yaml_candidates(task_id, repo=repo, data_dir=data_dir):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            raw = json.loads(text)
        else:
            raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise PbStopped("public_task_payload_must_be_object")
        body = _public_task_body(task_id, raw)
        body["source_path"] = str(path)
        return body
    return None


def _workspace_doc_paths(root):
    paths = []
    root = Path(root)
    if not root.is_dir():
        return paths
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "executable" or relative.startswith(".git/") or path.name == "tests.json":
            continue
        if path.suffix.lower() in _DOC_SUFFIXES or path.name.upper().startswith("README"):
            paths.append(relative)
        if len(paths) >= 40:
            break
    return paths


def _prepare_pb_workspace(services, unit, *, live, image_seeder):
    """Seed the cleanroom ``/workspace`` and attach a no-network sandbox.

    A dry run attaches the sandbox with execution off, so no daemon is contacted.
    A live run copies the local image. It does not pull.
    """
    image = programbench_instance_image(unit.task_id)
    if live:
        try:
            seed_image_tree(image, "/workspace", services.workspace.root, docker=image_seeder)
        except PilotStopped as error:
            return str(error)
        # The cleanroom binary is execute-only. A host copy would hide it or
        # make the bytes readable, so the sandbox keeps the image file.
        reference = services.workspace.root / "executable"
        if reference.is_symlink() or reference.exists():
            try:
                reference.unlink()
            except OSError:
                pass
        services.preserve_image_executable = True
    name = getattr(services, "container_name", None)
    services.sandbox = DockerSandbox(
        services.workspace.root,
        image_resolver=lambda image=image: image,
        permit_execution=bool(live),
        container_name=name,
        merge_image_workspace=bool(getattr(services, "preserve_image_executable", False)),
    )
    services.sandbox_image = image
    return None


class _JournaledCalls(list):
    """``services.calls`` that also hands each appended row to ``sink``."""

    def __init__(self, sink):
        super().__init__()
        self._sink = sink

    def append(self, row):
        super().append(row)
        try:
            self._sink(row)
        except Exception:
            # The journal is a safety copy. A failed write never stops a unit.
            pass


def _call_journal_sink(run_dir, unit_id, attempt_id, secrets):
    path = Path(run_dir) / CALL_JOURNAL
    counter = {"seq": 0}

    def sink(row):
        counter["seq"] += 1
        call = redact_secrets(dict(row), secrets)
        call.pop("text", None)
        call["unit_id"] = unit_id
        call["attempt_id"] = attempt_id
        call["journal_seq"] = counter["seq"]
        _append_jsonl(path, call)
    return sink


def _scan_submission(root, *, started, unit_seconds, clock=time.monotonic, scan=None):
    """End-of-unit scan under the unit deadline. Returns (submission, failure reason).

    The deadline is the unit deadline, but at least ``SCAN_MIN_SECONDS`` from
    now, so a unit that used its whole budget is still scanned in full. The scan
    runs in a child process that is killed at the deadline (see
    ``submission_reference_scan_bounded``).
    """
    deadline = max(started + float(unit_seconds), clock() + SCAN_MIN_SECONDS)
    remaining = deadline - clock()
    if remaining <= 0:
        return None, "submission_scan_deadline"
    scan = scan or submission_reference_scan_bounded
    try:
        return scan(root, remaining), None
    except SubmissionScanDeadline:
        return None, "submission_scan_deadline"
    except Exception:
        return None, "submission_scan_error"


class PbServices(FoxAloServices):
    """Fox pool services plus the ProgramBench reference-binary guard."""

    def __init__(self, *args, differential_enabled=False, signal_override=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.differential_enabled = bool(differential_enabled)
        self.signal_override = signal_override
        self.last_pb_signal = None
        self.compile_returncode = None
        self.reference_invocations = []

    def public_signal(self):
        """Run ``compile.sh`` in the sandbox. Differential I/O is opt-in.

        The default signal is that compile result alone. ``--differential-signal``
        compares the preserved reference with the ``./executable`` that
        compile.sh produced, the same file the official scorer keeps. A missing
        script or a sandbox that cannot execute stays unknown and is not
        written into learning state.
        """
        try:
            if self.signal_override in {"pass", "fail", "unknown"}:
                compile_code = {"pass": 0, "fail": 1, "unknown": None}[self.signal_override]
                differential = None
            else:
                compile_code = self._run_compile()
                differential = self._measure_differential() if self.differential_enabled else None
            result = combine_pb_signal(
                compile_code,
                differential_enabled=self.differential_enabled,
                differential_signal=differential,
            )
            self.last_pb_signal = result
            return result["signal"]
        finally:
            staged = self.workspace.root / PB_SIGNAL_CANDIDATE
            if staged.is_file() and not staged.is_symlink():
                try:
                    staged.unlink()
                except OSError:
                    pass

    def _run_compile(self):
        root = self.workspace.root
        if not (root / "compile.sh").is_file():
            return None
        official = getattr(self.sandbox, "official_compile", None)
        if callable(official) and getattr(self.sandbox, "merge_image_workspace", False):
            try:
                code = official(60)
            except (PilotStopped, ProtocolError, IntegrationNotConfigured, OSError):
                return None
            if type(code) is not int or isinstance(code, bool):
                return None
            self.compile_returncode = code
            return code
        try:
            self._sandbox_argv(["chmod", "+x", "compile.sh"])
            result = self._sandbox_argv(["./compile.sh"])
        except (PilotStopped, ProtocolError, IntegrationNotConfigured, OSError):
            return None
        code = result.get("returncode") if isinstance(result, dict) else None
        if type(code) is not int or isinstance(code, bool):
            return None
        self.compile_returncode = code
        return code

    def _sandbox_argv(self, argv):
        violation = pb_command_violation(argv)
        if violation:
            raise ProtocolError(violation)
        if self.sandbox is None:
            raise PilotStopped("sandbox_not_configured")
        return self.sandbox.command(argv, 60)

    def _measure_differential(self):
        """Run the reference and the built program with the same inputs.

        No sandbox, no built program, or a probe that cannot be executed is
        unknown. The argv is the normal interface only.
        """
        root = self.workspace.root
        candidate = root / PB_SIGNAL_CANDIDATE
        if self.sandbox is None or not candidate.is_file():
            return "unknown"
        probes = list(self.reference_invocations) or [()]
        signals = []
        for args in probes:
            try:
                reference_argv = blackbox_probe_argv("./executable", args)
                candidate_argv = blackbox_probe_argv("./" + PB_SIGNAL_CANDIDATE, args)
                signal = run_blackbox_differential(
                    reference_argv, candidate_argv, runner=self._blackbox_io)
            except (ProtocolError, PilotStopped, OSError):
                return "unknown"
            signals.append(signal)
        if not signals or any(item == "unknown" for item in signals):
            return "unknown"
        if any(item == "fail" for item in signals):
            return "fail"
        return "pass"

    def _blackbox_io(self, argv):
        before = _side_effect_digest(self.workspace.root)
        result = self.sandbox.command(argv, 30)
        after = _side_effect_digest(self.workspace.root)
        if not isinstance(result, dict) or type(result.get("returncode")) is not int:
            return None
        return {
            "returncode": result["returncode"],
            "stdout": result.get("stdout") if isinstance(result.get("stdout"), str) else "",
            "stderr": result.get("stderr") if isinstance(result.get("stderr"), str) else "",
            "fs_before": before,
            "fs_after": after,
        }

    def _observe_command(self, argv, result):
        if not isinstance(argv, list) or not isinstance(result, dict):
            return
        code = result.get("returncode")
        if type(code) is not int or isinstance(code, bool):
            return
        if _is_compile_invocation(argv):
            self.compile_returncode = code
        if argv and is_reference_path(argv[0]):
            self.reference_invocations.append(tuple(argv[1:]))

    def command(self, argv, timeout_s=30):
        violation = pb_command_violation(argv)
        if violation:
            raise ProtocolError(violation)
        result = super().command(argv, timeout_s)
        self._observe_command(argv, result)
        return result

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
def open_pb_transfer(source_path, *, allowed_models, model_map=None, strip_quality=False,
                     substitute=None):
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
    mapping = default_pb_model_map(substitute)
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


def import_swe_profile(source_path, *, allowed_models, model_map=None, strip_quality=False,
                       substitute=None):
    """Copy the imported observations out of the probe store. The source stays put."""
    with open_pb_transfer(
            source_path, allowed_models=allowed_models, model_map=model_map,
            strip_quality=strip_quality, substitute=substitute) as library:
        observations = copy.deepcopy(library.observations)
        digest = library.pb_source_sha256
        dropped = library.pb_dropped_models
    return {
        "observations": observations,
        "source_sha256": digest,
        "dropped_models": list(dropped),
        "strip_quality": bool(strip_quality),
    }


def pb_eval_plan(task_id, submission_dir, *, repo=None, unit_id=None, run_dir=None):
    """Official cleanroom build plus ``programbench eval``. This does not run it."""
    sealed = None
    if run_dir and unit_id:
        sealed = Path(run_dir) / "sealed" / "programbench" / unit_id
    return {
        "task_id": task_id,
        "image_family": "task_cleanroom_v6",
        "source_commit": PROGRAMBENCH_COMMIT,
        "tests_revision": PROGRAMBENCH_TESTS_REVISION,
        "reference_mode": "111",
        "network": "none",
        "image": programbench_instance_image(task_id),
        "build": ["chmod", "+x", "./compile.sh"],
        "build_exec": ["./compile.sh"],
        "build_in_cleanroom": "chmod +x ./compile.sh && ./compile.sh",
        "eval_command": programbench_command(repo, sealed) if repo and sealed else [],
        "submission_dir": str(submission_dir),
        "sealed_dir": str(sealed) if sealed else None,
        "repo": str(repo) if repo else None,
        "run_dir": str(run_dir) if run_dir else None,
        "unit_id": unit_id,
    }


def _submission_rejection_reason(submission_check):
    violations = submission_check.get("violations") if isinstance(submission_check, dict) else None
    parts = []
    for item in violations or ():
        if not isinstance(item, dict):
            continue
        kind = item.get("kind") if isinstance(item.get("kind"), str) else "rejected"
        path = item.get("path") if isinstance(item.get("path"), str) else ""
        parts.append(f"{kind}:{path}" if path else kind)
    if not parts:
        return "submission_rejected"
    return "submission_rejected:" + ",".join(parts)


def _annotated_official_score(result, reason):
    """The harness score, stored beside the unit score of 0.

    A rejected submission's unit score stays 0. This field is the score the
    official runner would have reported, and it is not that unit score.
    """
    if not isinstance(result, dict):
        return {
            "annotation": "offline_official_score_not_the_unit_score",
            "reason": reason,
            "unit_score": 0,
            "pass_rate": None,
            "resolved": None,
            "compile_success": None,
            "status": None,
        }
    rate = result.get("pass_rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        rate = None
    resolved = result.get("resolved") if result.get("resolved") is True or result.get("resolved") is False else None
    compile_success = result.get("compile_success")
    if compile_success is not True and compile_success is not False:
        compile_success = None
    return {
        "annotation": "offline_official_score_not_the_unit_score",
        "reason": reason,
        "unit_score": 0,
        "pass_rate": None if rate is None else float(rate),
        "resolved": resolved,
        "compile_success": compile_success,
        "status": result.get("status") if isinstance(result.get("status"), str) else None,
    }


def apply_pb_score(plan, *, execute, runner, public_signal, cost, wall_clock_s, submission_check):
    """Record the harness fields. Unknown stays null, not zero.

    A scan rejection records score 0 and the reason. When a runner is
    available it still runs, and its score is stored as a separate
    annotation. The unit row is the caller's to write; this does not drop it.
    """
    if public_signal == "pass":
        compile_success = True
    elif public_signal == "fail":
        compile_success = False
    else:
        compile_success = None
    body = {
        "resolved": None,
        "pass_rate": None,
        "score": None,
        "compile_success": compile_success,
        "cost": cost,
        "wall_clock_s": wall_clock_s,
        "status": "not_executed",
        "submission_check": submission_check,
        "source_commit": plan.get("source_commit"),
        "image_family": plan.get("image_family"),
        "reason": None,
        "official_score_annotation": None,
    }
    rejected = isinstance(submission_check, dict) and submission_check.get("ok") is False
    if rejected:
        reason = _submission_rejection_reason(submission_check)
        body["status"] = "submission_rejected"
        body["resolved"] = False
        body["pass_rate"] = 0.0
        body["score"] = 0
        body["reason"] = reason
        if execute and runner is not None:
            try:
                result = runner(plan)
            except Exception:
                result = None
            body["official_score_annotation"] = _annotated_official_score(result, reason)
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
        body["reason"] = "pb_eval_unreadable"
        return body
    if isinstance(result.get("reason"), str):
        body["reason"] = result["reason"]
    if result.get("resolved") is True or result.get("resolved") is False:
        body["resolved"] = result["resolved"]
    rate = result.get("pass_rate")
    if not isinstance(rate, bool) and isinstance(rate, (int, float)):
        body["pass_rate"] = float(rate)
    if result.get("compile_success") is True or result.get("compile_success") is False:
        body["compile_success"] = result["compile_success"]
    body["status"] = result.get("status") or "finished"
    if body["pass_rate"] is not None:
        body["score"] = body["pass_rate"]
    return body


def _adapter_config(arm, *, models, request_cap, seed, online_update, dag, prompt,
                    node_request_cap=None):
    if arm == "A4":
        if BEST_SINGLE not in models:
            raise PbStopped("best_single_not_in_pool")
        # One tool-loop node. It keeps the unit cap. p1/p2 caps do not apply.
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
    if node_request_cap is not None:
        config["node_request_caps"] = {"p1": node_request_cap, "p2": node_request_cap}
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


def _assignment_rows_from_state(state):
    """Node rows from the checkpoint assignments.

    ``last_native_run`` is written only when the native task returns. A unit
    that stops on a cap or a gateway error still has ``assignments`` in the
    same state the checkpoint dumps.
    """
    if not isinstance(state, dict):
        return []
    raw = state.get("assignments")
    if not isinstance(raw, dict):
        return []
    node_of = {}
    contracts = state.get("role_tool_contracts")
    if isinstance(contracts, list):
        for item in contracts:
            if not isinstance(item, dict):
                continue
            role_id = item.get("role_id")
            node_id = item.get("node_id")
            if isinstance(role_id, str) and role_id and isinstance(node_id, str) and node_id:
                node_of.setdefault(role_id, node_id)
    rows = []
    for role, value in raw.items():
        if not isinstance(value, dict):
            continue
        role_id = value.get("role_id") if isinstance(value.get("role_id"), str) else role
        if not isinstance(role_id, str) or not role_id:
            continue
        model_id = value.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        rows.append({
            "node_id": node_of.get(role_id, role_id),
            "role_id": role_id,
            "model_id": model_id,
        })
    order = {node_id: index for index, node_id in enumerate(("p1", "p2", "s1", "s2"))}
    rows.sort(key=lambda row: (order.get(row["node_id"], 100), row["role_id"]))
    return rows


def _node_assignments(adapter):
    state = adapter.state if hasattr(adapter, "state") else None
    native = state.get("last_native_run") if isinstance(state, dict) else None
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
    if rows:
        return rows
    return _assignment_rows_from_state(state)


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
        "node_request_cap": settings.get("node_request_cap"),
        "pool_substitution": decision_freeze(settings)[0].get("pool_substitution"),
        "exact_returned_opt_in": freeze_body.get("exact_returned_opt_in"),
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
        "data_dir": settings.get("data_dir"),
        "programbench_repo": settings.get("programbench_repo"),
        "reference_guard": "execute-only-blackbox-v1",
        "reference_mode": "111",
        "source_commit": PROGRAMBENCH_COMMIT,
        "tests_revision": PROGRAMBENCH_TESTS_REVISION,
        "api_key_env": FOX_KEY_ENV,
        "model_endpoints": {row.internal: {"api_base": row.api_base, "key_env": row.key_env}
                            for row in (_pb_spec(name) for name in settings["models"])},
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
        # An older manifest omits the key. That omission means the opt-in was
        # off. Turning it on in that directory would change the recorded freeze
        # without rewriting the manifest, so it is refused.
        if payload.get("exact_returned_opt_in") != existing.get("exact_returned_opt_in"):
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
        substitute=settings.get("pool_substitute"),
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


def _exact_returned_document(version):
    """The one audit document, or None when the run leaves the default."""
    if version is None:
        return None
    if version != DEEPSEEK_EXACT_RETURN_VERSION:
        raise PbStopped("unknown_returned_name_opt_in")
    return dict(DEEPSEEK_EXACT_RETURN)


def _specs_with_exact_return(specs, document):
    """Copy the deepseek slot with the opt-in. The catalog row stays unset."""
    if document is None:
        return specs
    if not any(row.internal == document["slot"] for row in specs):
        raise PbStopped("exact_return_opt_in_requires_slot")
    updated = []
    for row in specs:
        if row.internal == document["slot"]:
            updated.append(replace(row, exact_returned_opt_in=document["version"]))
        else:
            updated.append(row)
    return tuple(updated)


def run_arm(run_dir, *, arm, tasks=None, g0=False, models="all", request_cap=DEFAULT_REQUEST_CAP,
            node_request_cap=None, unit_seconds=DEFAULT_UNIT_SECONDS,
            read_timeout_s=DEFAULT_READ_TIMEOUT_S,
            online_update=False, differential_signal=False, role_map=None,
            probe_script_owner="p1", swe_profile=None, model_map=None, live=False,
            sender=None, execute=False, scorer=None, signal_override=None, clock=None,
            data_dir=None, programbench_repo=None, image_seeder=None, pool_substitute=None,
            returned_name_opt_in=None):
    """Run one arm in its own directory. Completed unit ids are skipped."""
    if arm not in ARMS:
        raise PbStopped("unknown_pb_arm")
    if arm in {"A1", "A3"} and not swe_profile:
        raise PbStopped("swe_profile_required")
    specs = resolve_pb_pool(models, pool_substitute)
    exact_returned = _exact_returned_document(returned_name_opt_in)
    specs = _specs_with_exact_return(specs, exact_returned)
    model_names = tuple(row.internal for row in specs)
    request_cap = _check_request_cap(request_cap)
    node_request_cap = _check_node_request_cap(node_request_cap, request_cap)
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
        "node_request_cap": node_request_cap,
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
        "pool_substitute": pool_substitute,
        "exact_returned_opt_in": exact_returned,
        "data_dir": str(data_dir) if data_dir else None,
        "programbench_repo": str(programbench_repo) if programbench_repo else None,
        "image_seeder": image_seeder,
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
    if live and sender is None:
        # Each slot reads its own key env var. Only presence is checked here.
        for env_name in sorted({row.key_env for row in specs}):
            if not key_configured(env_name):
                raise PbStopped("credential_missing:" + env_name)
    transport = _transport_for(
        live=live,
        sender=sender if sender is not None else (None if live else dry_gateway),
        timeout_s=read_timeout_s,
        sleep=time.sleep,
        clock=clock or (lambda: datetime.now(timezone.utc)),
        max_attempts=4,
        key_envs=sorted({row.key_env for row in specs}),
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
        "node_request_cap": settings.get("node_request_cap"),
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
    if settings.get("exact_returned_opt_in"):
        record["exact_returned_opt_in"] = dict(settings["exact_returned_opt_in"])
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
    attempt_id = uuid.uuid4().hex[:12]
    record["attempt_id"] = attempt_id
    services.calls = _JournaledCalls(
        _call_journal_sink(run_dir, unit.unit_id, attempt_id, _live_secrets()))
    prepare_note = _prepare_pb_workspace(
        services, unit, live=live, image_seeder=settings.get("image_seeder"))
    record["sandbox_image"] = getattr(services, "sandbox_image", None)
    record["sandbox_workdir"] = getattr(services.sandbox, "workdir", None) if services.sandbox else None
    record["sandbox_network"] = "none"
    record["preserve_executable_in_image"] = bool(getattr(services, "preserve_image_executable", False))
    if services.sandbox is not None:
        record["sandbox_attached"] = True
        try:
            record["sandbox_network_flag"] = "--network=none" in services.sandbox.docker_argv(["true"])
        except (PilotStopped, ProtocolError, OSError):
            record["sandbox_network_flag"] = False
    if prepare_note:
        record.update(status="blocked", reason=prepare_note, harness_resolved=None,
                      pass_rate=None, compile_success=None, cost=None, wall_clock_s=0,
                      first_assignment=None)
        record["_calls"] = []
        return record
    group, config = _adapter_config(
        arm, models=model_names, request_cap=settings["request_cap"], seed=unit.seed,
        online_update=settings["online_update"], dag=dag, prompt=prompt,
        node_request_cap=settings.get("node_request_cap"),
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
    payload = load_pb_public_payload(
        unit.task_id, repo=settings.get("programbench_repo"), data_dir=settings.get("data_dir"))
    if payload is None:
        payload = load_public_payload(settings.get("data_dir"), "ProgramBench", unit.task_id)
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["reference_executable"] = payload.get("reference_executable") or "./executable"
        payload["reference_mode"] = "111"
        payload["docs"] = _workspace_doc_paths(services.workspace.root)
        record["reference_executable"] = "./executable"
        record["reference_mode"] = "111"
        record["public_repository"] = payload.get("repository")
        record["public_docs"] = list(payload["docs"])
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
    submission, scan_failure = _scan_submission(
        services.workspace.root, started=started, unit_seconds=limits["seconds"])
    scan_failed = scan_failure is not None
    if scan_failed:
        submission = {"ok": None, "violations": [], "warnings": [], "unused_probes": []}
    usage = _usage_block(services.calls)
    plan = pb_eval_plan(
        unit.task_id, services.workspace.root, repo=settings.get("programbench_repo"),
        unit_id=unit.unit_id, run_dir=run_dir)
    if execute and scorer is None:
        scorer = official_pb_scorer
    if scan_failed:
        # A scan crash or overrun is not a reference-rule rejection. The score stays null.
        scored = {
            "resolved": None, "pass_rate": None, "score": None, "compile_success": None,
            "status": "unavailable", "reason": scan_failure,
            "official_score_annotation": None,
        }
    else:
        try:
            scored = apply_pb_score(
                plan, execute=bool(execute), runner=scorer, public_signal=signal,
                cost=usage["cost"], wall_clock_s=wall, submission_check=submission,
            )
        except Exception:
            if isinstance(submission, dict) and submission.get("ok") is False:
                scored = {
                    "resolved": False, "pass_rate": 0.0, "score": 0, "compile_success": None,
                    "status": "submission_rejected",
                    "reason": _submission_rejection_reason(submission),
                    "official_score_annotation": None,
                }
            else:
                # The evaluator failed before measuring. Null is not a model zero,
                # and this is not a submission rejection.
                scored = {
                    "resolved": None, "pass_rate": None, "score": None, "compile_success": None,
                    "status": "unavailable", "reason": "evaluator_error",
                    "official_score_annotation": None,
                }
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
        resolved=scored["resolved"],
        harness_status=scored["status"],
        harness_reason=scored.get("reason"),
        pass_rate=scored["pass_rate"],
        score=scored.get("score"),
        compile_success=scored["compile_success"],
        submission_ok=submission.get("ok"),
        submission_warnings=list(submission.get("warnings") or []),
        submission_unused_probes=list(submission.get("unused_probes") or []),
        official_score_annotation=scored.get("official_score_annotation"),
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
    if scored.get("status") == "submission_rejected":
        record["pass_rate"] = 0.0
        record["score"] = 0
        record["resolved"] = False
        record["harness_resolved"] = False
        rejection = scored.get("reason") or "submission_rejected"
        record["submission_rejection_reason"] = rejection
        record["harness_reason"] = rejection
        if not record.get("reason"):
            record["reason"] = rejection
    record["_calls"] = list(services.calls)
    return record


def run_g0(root, **kwargs):
    """A2 and A4 on the three G0 tasks. Each arm has its own directory."""
    root = Path(root)
    kwargs = dict(kwargs)
    kwargs.pop("arm", None)
    kwargs["g0"] = True
    kwargs["tasks"] = None
    kwargs.setdefault("node_request_cap", M18_NODE_REQUEST_CAP)
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
    parser.add_argument("--node-request-cap", type=int, default=None)
    parser.add_argument("--pool-substitute", default=None, choices=tuple(PB_POOL_SUBSTITUTIONS))
    parser.add_argument(
        "--returned-name-opt-in", default=None, choices=(DEEPSEEK_EXACT_RETURN_VERSION,),
        help="Accept only the exact returned name deepseek-flash on fox/deepseek-flash. "
             "Default rejects that name. The upstream request id is unchanged.")
    parser.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT_S)
    parser.add_argument("--unit-seconds", type=float, default=DEFAULT_UNIT_SECONDS)
    parser.add_argument("--online-update", action="store_true")
    parser.add_argument("--differential-signal", action="store_true")
    parser.add_argument("--probe-script-owner", default="p1", choices=("p1", "p2"))
    parser.add_argument("--role-map", default="s1=p1,s2=p2")
    parser.add_argument("--swe-profile", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--programbench-repo", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--execute", action="store_true")


def _kwargs_from_args(args):
    return {
        "tasks": args.tasks,
        "g0": bool(args.g0),
        "models": args.models,
        "request_cap": args.request_cap,
        "node_request_cap": args.node_request_cap,
        "pool_substitute": args.pool_substitute,
        "returned_name_opt_in": args.returned_name_opt_in,
        "unit_seconds": args.unit_seconds,
        "read_timeout_s": args.read_timeout,
        "online_update": bool(args.online_update),
        "differential_signal": bool(args.differential_signal),
        "role_map": args.role_map,
        "probe_script_owner": args.probe_script_owner,
        "swe_profile": args.swe_profile,
        "live": bool(args.live),
        "execute": bool(args.execute and args.live),
        "data_dir": args.data_dir,
        "programbench_repo": args.programbench_repo,
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
        kwargs = _kwargs_from_args(args)
        if (args.command == "g0" or getattr(args, "g0", False)) and kwargs.get("node_request_cap") is None:
            kwargs["node_request_cap"] = M18_NODE_REQUEST_CAP
        if args.execute and args.live:
            kwargs["scorer"] = official_pb_scorer
        if args.command == "g0":
            result = run_g0(args.root, **kwargs)
        else:
            if not args.g0 and not args.tasks:
                parser.error("pass --tasks or --g0")
            result = run_arm(args.run_dir, arm=args.arm, **kwargs)
        print(_canonical(result))
        return 0
    except (PbStopped, FoxStopped, PilotStopped) as error:
        print(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
