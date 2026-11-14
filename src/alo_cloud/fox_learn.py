"""ALO learning stream on the Foxrouter pool.

Six chains: persist, reset, and resources-only, each forward and reverse.
A chain is one process, one run directory, one run id, and its own Docker
container names. Units inside a chain run in order. The domestic CNY ledger
is not opened. Token totals and raw ``cost`` / ``credit`` / ``quota`` are
stored as the gateway returned them. There is no unit price.

``--live`` is the only path that reads ``FOXROUTER_API_KEY`` or opens a
socket. Offline tests pass ``sender``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

from .adapters import adapter_for
from .errors import PilotStopped
from .freeze import PlannedUnit
from .fox_pool import (
    DEFAULT_READ_TIMEOUT_S,
    DEFAULT_TURN_CAP,
    FOX_KEY_ENV,
    FOX_RESOURCE_PRIOR,
    SWEBENCH_LITE_REVISION,
    model_settings,
    resolve_alo_pool,
    select_tasks,
)
from .fox_single import (
    FoxServices,
    FoxStopped,
    _append_jsonl,
    _live_secrets,
    _load_jsonl,
    _sum_amount,
    _sum_tokens,
    _transport_for,
    assert_fox_run_dir,
    prepare_workspace,
    redact_secrets,
    score_prediction,
)
from .r2 import learning_g2_config
from .run import load_public_payload, stage_for
from .sandbox import DockerSandbox, python_syntax_evidence
from .state import export_learning_state, export_transfer_package, import_learning_state


EXPERIMENT = "fox-alo-learn-v1"
SCHEMA = "fox-alo-learn-v1"
UNIT_SCHEMA = "fox-alo-learn-unit-v1"
DEFAULT_STREAM_SEED = "fox-alo-learn-v1"
# Uncapped headroom for the policy's ``cny`` field. It is not a price and not
# a CNY balance. An optional ``--budget-cap`` replaces it with remaining raw cost.
_UNCAPPED_CNY = 1_000_000_000.0
_CONDITIONS = (
    ("persist", "g2-learn", "persist"),
    ("reset", "g2-learn", "reset"),
    ("resources-only", "g2-resources", "persist"),
)
_ORDERS = ("forward", "reverse")
_TERMINAL = frozenset({"completed", "failed", "blocked", "stopped"})
_LOCKED = (
    "pool", "models", "chain_id", "run_id", "arm", "condition", "order",
    "turn_cap", "budget_cap", "task_ids", "unit_ids", "live",
)


class LearnStopped(FoxStopped):
    """The learning stream stopped. The argument is a reason code."""


@dataclass(frozen=True)
class LearningChain:
    chain_id: str
    run_id: str
    condition: str
    order: str
    arm: str
    m2_condition: str
    units: tuple
    tasks: tuple
    turn_cap: int

    @property
    def strip_quality(self):
        return self.arm == "g2-resources"


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def profile_digest(exported):
    """Hash the carried statistics. The per-unit evidence id is not part of it."""
    if not isinstance(exported, dict):
        raise LearnStopped("profile_export_missing")
    body = {
        "belief": exported.get("belief"),
        "tms_outcomes": exported.get("tms_outcomes"),
        "observations": exported.get("observations"),
        "transfer_policy": exported.get("transfer_policy"),
    }
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


_EMPTY_PROFILE = {}


def empty_profile_hash(models, *, strip_quality):
    key = (tuple(sorted(models)), bool(strip_quality))
    cached = _EMPTY_PROFILE.get(key)
    if cached is not None:
        return cached
    from alo.org.organization import Organization
    with tempfile.TemporaryDirectory(prefix="alo-fox-empty-") as directory:
        db_path = Path(directory) / "alo.sqlite3"
        org = Organization(str(db_path))
        org.conn.close()
        exported = export_learning_state(
            db_path, {"observations": {}}, allowed_models=models, strip_quality=strip_quality)
    digest = profile_digest(exported)
    _EMPTY_PROFILE[key] = digest
    return digest


def snapshot_profile(db_path, adapter_state, models, *, strip_quality):
    path = Path(db_path)
    if not path.is_file():
        return empty_profile_hash(models, strip_quality=strip_quality)
    exported = export_learning_state(
        path, adapter_state, allowed_models=models, strip_quality=strip_quality)
    return profile_digest(exported)


def _unit_seed(stream_seed, chain_id, position):
    digest = hashlib.sha256(f"{stream_seed}|{chain_id}|{position}".encode()).hexdigest()
    return int(digest[:8], 16)


def container_name_for(chain_id, unit_id):
    """Docker name unique to this chain and unit. No slashes."""
    digest = hashlib.sha256(f"{chain_id}|{unit_id}".encode()).hexdigest()[:12]
    slug = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in str(chain_id).lower()).strip("-")
    slug = slug[:40] or "chain"
    return f"alo-fox-{slug}-{digest}"


def workdir_for(unit_id):
    return str(Path("units") / unit_id / "state" / "workspace")


def _read_id_list(path):
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list) or not data:
            raise LearnStopped("empty_selection")
        found = []
        for item in data:
            if isinstance(item, str) and item.strip():
                found.append(item.strip())
            elif isinstance(item, dict) and isinstance(item.get("instance_id"), str):
                found.append(item["instance_id"].strip())
            else:
                raise LearnStopped("invalid_task_list")
        return found
    found = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        found.append(line.split()[0])
    if not found:
        raise LearnStopped("empty_selection")
    return found


def select_learning_tasks(*, split="dev", task_ids=(), tasks_file=None, n=None,
                          seed=DEFAULT_STREAM_SEED, allow_all=False):
    """A subset of SWE-bench Lite. One selector. No network."""
    selectors = [
        bool(task_ids),
        tasks_file is not None,
        n is not None,
        bool(allow_all),
    ]
    if sum(1 for item in selectors if item) != 1:
        raise LearnStopped("selection_required")
    if split not in {"dev", "test", "both"}:
        raise LearnStopped("unknown_split")
    if tasks_file is not None:
        task_ids = _read_id_list(tasks_file)
    try:
        if task_ids:
            return select_tasks(split, instance_ids=tuple(task_ids))
        catalog = select_tasks(split, allow_all=True)
    except ValueError as error:
        raise LearnStopped(str(error)) from None
    if allow_all:
        return catalog
    if type(n) is not int or n < 1:
        raise LearnStopped("invalid_limit")
    if not isinstance(seed, str) or not seed:
        raise LearnStopped("invalid_seed")
    ranked = tuple(sorted(
        catalog,
        key=lambda row: hashlib.sha256(f"{seed}|{row[0]}".encode()).hexdigest(),
    ))
    if n > len(ranked):
        raise LearnStopped("limit_exceeds_split")
    return ranked[:n]


def _wanted(label, selected):
    return not selected or label in selected


def iter_learning_chains(tasks, *, turn_cap=DEFAULT_TURN_CAP, stream_seed=DEFAULT_STREAM_SEED,
                         arms=(), conditions=(), orders=()):
    """Persist, reset, and resources-only, each in forward and reverse order."""
    if type(turn_cap) is not int or turn_cap < 1:
        raise LearnStopped("invalid_turn_cap")
    if not tasks:
        raise LearnStopped("empty_selection")
    arm_filter = set(arms or ())
    condition_filter = set(conditions or ())
    order_filter = set(orders or ())
    for name in arm_filter:
        if name not in {"g2-learn", "g2-resources"}:
            raise LearnStopped("unknown_arm")
    for name in condition_filter:
        if name not in {row[0] for row in _CONDITIONS}:
            raise LearnStopped("unknown_condition")
    for name in order_filter:
        if name not in _ORDERS:
            raise LearnStopped("unknown_order")
    chains = []
    for condition, arm, m2_condition in _CONDITIONS:
        if not _wanted(arm, arm_filter) or not _wanted(condition, condition_filter):
            continue
        for order in _ORDERS:
            if not _wanted(order, order_filter):
                continue
            chain_id = f"{condition}-{order}"
            ordered = tuple(tasks) if order == "forward" else tuple(reversed(tuple(tasks)))
            units = []
            previous = ""
            for position, (instance_id, _repo, split) in enumerate(ordered):
                parent = previous if m2_condition == "persist" else ""
                unit = PlannedUnit(
                    study="m2", arm=arm, track="SWE-bench_Lite", task_id=instance_id,
                    stratum=split, phase="learning", position=position,
                    seed=_unit_seed(stream_seed, chain_id, position),
                    request_cap=turn_cap, order_id=order, m2_condition=m2_condition,
                    parent_unit_id=parent,
                )
                units.append(unit)
                if m2_condition == "persist":
                    previous = unit.unit_id
            chains.append(LearningChain(
                chain_id=chain_id,
                run_id=f"{EXPERIMENT}.{chain_id}",
                condition=condition,
                order=order,
                arm=arm,
                m2_condition=m2_condition,
                units=tuple(units),
                tasks=ordered,
                turn_cap=turn_cap,
            ))
    if not chains:
        raise LearnStopped("no_chains")
    return tuple(chains)


def chain_locations(chain, run_dir):
    """Host workdir and container name for every unit. Paths stay under ``run_dir``."""
    root = Path(run_dir)
    rows = []
    for unit in chain.units:
        rows.append({
            "unit_id": unit.unit_id,
            "task_id": unit.task_id,
            "parent_unit_id": unit.parent_unit_id,
            "position": unit.position,
            "run_id": chain.run_id,
            "run_dir": str(root),
            "workdir": workdir_for(unit.unit_id),
            "container_name": container_name_for(chain.chain_id, unit.unit_id),
        })
    return rows


def launch_command(chain, run_dir, *, models="all", tasks_file=None, n=None,
                   seed=DEFAULT_STREAM_SEED, split="dev", turn_cap=DEFAULT_TURN_CAP,
                   data_dir="$FOX_DATA_DIR", budget_cap=None, live=True, allow_all=False,
                   task_ids=None):
    """One process command. Parallel chains differ in ``--chain`` and ``--run-dir``.

    ``task_ids`` is the selection order, before forward/reverse. A reverse
    chain must not write its already reversed units back into ``--tasks``,
    or the next run flips them a second time.
    """
    command = [
        "python", "-m", "alo_cloud.fox_learn", "run",
        "--pool", "fox",
        "--chain", chain.chain_id,
        "--run-dir", str(run_dir),
        "--turn-cap", str(turn_cap),
        "--split", split,
    ]
    if models in (None, "all"):
        command.extend(["--models", "all"])
    else:
        command.extend(["--models", ",".join(models)])
    if tasks_file:
        command.extend(["--tasks-file", str(tasks_file)])
    elif n is not None:
        command.extend(["--n", str(n), "--seed", str(seed)])
    elif allow_all:
        command.append("--all")
    elif task_ids:
        command.extend(["--tasks", ",".join(task_ids)])
    else:
        raise LearnStopped("selection_required")
    if budget_cap is not None:
        command.extend(["--budget-cap", str(budget_cap)])
    command.extend(["--data-dir", str(data_dir)])
    if live:
        command.extend(["--live", "--execute"])
    else:
        command.append("--dry-run")
    return command


def planner_for(models):
    """``fox/deepseek-flash`` when it is in the subset, otherwise the first fox model."""
    if "fox/deepseek-flash" in models:
        return "fox/deepseek-flash"
    if not models:
        raise LearnStopped("fox_models_required")
    return models[0]


def fox_learning_config(unit, models):
    """#35 learning config with the fox pool as the common candidate set."""
    config = learning_g2_config(unit)
    config["planner_model"] = planner_for(tuple(models))
    config["policy_priors"] = {model: dict(FOX_RESOURCE_PRIOR) for model in models}
    return config


class FoxAloServices(FoxServices):
    """Multi-model fox gateway for one ALO unit. Not a CNY ledger."""

    def __init__(self, state_dir, specs, *, settings_by_model, transport, limits,
                 budget_cap=None, spent_cost=0.0, context_policy="fail", clock=None,
                 container_name=None):
        if not specs:
            raise LearnStopped("fox_models_required")
        primary = specs[0]
        super().__init__(
            state_dir, primary, settings=settings_by_model[primary.internal],
            transport=transport, limits=limits, clock=clock, budget_cap=budget_cap,
            spent_cost=spent_cost, context_policy=context_policy)
        self.specs = {row.internal: row for row in specs}
        self.settings_by_model = dict(settings_by_model)
        self.available_models = tuple(row.internal for row in specs)
        self.container_name = container_name
        self.events = []
        self.role_tools = None
        self.review_snapshot = None
        self.active_criterion_id = None
        self.image_head = None

    def budget_remaining(self, model=None):
        remaining = super().budget_remaining(model)
        if self.budget_cap is None:
            remaining["cny"] = _UNCAPPED_CNY
        else:
            remaining["cny"] = max(0.0, float(self.budget_cap) - float(self.spent_cost))
        return remaining

    def resources(self):
        return {
            "model_requests": self._requests,
            "observed_input_tokens": _sum_tokens(self.calls, "prompt_tokens") or 0,
            "observed_output_tokens": _sum_tokens(self.calls, "completion_tokens") or 0,
            "observed_reasoning_tokens": _sum_tokens(self.calls, "reasoning_tokens") or 0,
            # Raw gateway cost, not a CNY conversion. The policy field keeps its name.
            "estimated_cost_cny": float(self.spent_cost),
        }

    def complete_for_model(self, model, messages, *, max_tokens=4096):
        spec = self.specs.get(model)
        if spec is None:
            raise LearnStopped("model_outside_fox_pool")
        self.model_spec = spec
        self.model = spec.internal
        self.settings = self.settings_by_model[spec.internal]
        started = time.monotonic()
        reply = super().complete_for_model(model, messages, max_tokens=max_tokens)
        if isinstance(reply, dict):
            reply = dict(reply)
            reply["latency_ms"] = max(0, int(round((time.monotonic() - started) * 1000)))
        return reply

    def public_quality(self):
        return None

    def event(self, event, payload):
        self.events.append((event, payload))

    def local_criterion(self, *, role_id, model_id, output, context):
        if self.active_criterion_id == "repro-v1":
            from .repro import evidence_for_service
            return evidence_for_service(self, context=context)
        del role_id, model_id, output, context
        return python_syntax_evidence(self.workspace, self.workspace.take_attributed())

    def review_command(self, argv, timeout_s=30):
        from .review_isolation import run_review_command
        timeout_s, _notes = self.clamp_timeout(timeout_s)

        def run():
            return run_review_command(self, argv, timeout_s)

        return self._tool(run)


def dry_gateway(url, body, headers, timeout_s):
    """Deterministic fox response. No socket and no key."""
    del url, headers, timeout_s
    messages = body.get("messages") if isinstance(body, dict) else []
    parts = []
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
    blob = "\n".join(parts)
    if "ALO_DECOMPOSE" in blob:
        text = (
            "subtasks:\n"
            "  - id: s1\n"
            "    goal: Inspect the workspace and submit a short public note.\n"
            "    deps: []\n"
            "    topics: [coding]\n"
            "    role:\n"
            "      capability_reqs: [implement]\n"
            "      epistemic_scope:\n"
            "        include: [task_description]\n"
            "        exclude: [credentials]\n"
            "      authority: [inspect]\n"
            "      tools: [list_files]\n"
            "      output_schema: text\n"
            "      success_criteria: non_empty_output\n"
            "  - id: s2\n"
            "    goal: Verify the public note.\n"
            "    deps: [s1]\n"
            "    topics: [validation]\n"
            "    role:\n"
            "      capability_reqs: [review]\n"
            "      epistemic_scope:\n"
            "        include: [task_description]\n"
            "        exclude: [credentials]\n"
            "      authority: [inspect]\n"
            "      tools: [list_files]\n"
            "      output_schema: text\n"
            "      success_criteria: non_empty_output\n"
        )
    elif "STAGE_DONE" in blob:
        text = "STAGE_DONE"
    else:
        text = '{"final":"dry-run submission"}'
    model = body.get("model") if isinstance(body, dict) else None
    payload = {
        "model": model,
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        "cost": 0.002,
        "credit": 1.5,
        "quota": 9,
    }
    return 200, json.dumps(payload).encode("utf-8")


def _copy_sqlite(source, dest):
    source = Path(source)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    for suffix in ("-wal", "-shm"):
        extra = Path(str(source) + suffix)
        if extra.is_file():
            shutil.copy2(extra, Path(str(dest) + suffix))


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
            "bandwidth": node.get("bandwidth_final"),
        })
    return rows


def _assignments(adapter):
    rows = []
    raw = adapter.state.get("assignments") if hasattr(adapter, "state") else None
    if isinstance(raw, dict):
        for role, value in sorted(raw.items()):
            if not isinstance(value, dict):
                continue
            rows.append({
                "role_id": value.get("role_id") or role,
                "model_id": value.get("model_id"),
                "bandwidth": value.get("bandwidth"),
                "identity_id": value.get("identity_id"),
            })
    by_model = {}
    for row in rows:
        model = row.get("model_id")
        if isinstance(model, str) and model:
            by_model.setdefault(model, []).append(row["role_id"])
    return rows, {key: values for key, values in sorted(by_model.items())}


def _compact_decisions(records):
    kept = []
    if not isinstance(records, list):
        return kept
    for row in records:
        if not isinstance(row, dict):
            continue
        item = {}
        for key in ("point", "selected", "counterfactual", "routing_mode", "node_id",
                    "action", "target_model", "reason"):
            if key in row:
                item[key] = row.get(key)
        trace = row.get("trace")
        scores = trace.get("scores") if isinstance(trace, dict) else None
        if isinstance(scores, dict):
            item["feasible"] = {
                model: bool(score.get("feasible"))
                for model, score in scores.items() if isinstance(score, dict)
            }
        if item:
            kept.append(item)
    return kept


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


def _manifest(chain, pool, *, budget_cap, split, live):
    locations = chain_locations(chain, ".")
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "pool": "fox",
        "prices": "none",
        "budget": "raw_optional",
        "models": list(pool["models"]),
        "model_specs": [{
            "internal": row.internal,
            "upstream": row.upstream,
            "report_name": row.report_name,
        } for row in pool["specs"]],
        "chain_id": chain.chain_id,
        "run_id": chain.run_id,
        "arm": chain.arm,
        "condition": chain.condition,
        "order": chain.order,
        "m2_condition": chain.m2_condition,
        "turn_cap": chain.turn_cap,
        "budget_cap": budget_cap,
        "split": split,
        "task_ids": [unit.task_id for unit in chain.units],
        "unit_ids": [unit.unit_id for unit in chain.units],
        "container_names": [row["container_name"] for row in locations],
        "workdirs": [row["workdir"] for row in locations],
        "api_key_env": FOX_KEY_ENV,
        "api_key_recorded": False,
        "dataset_revision": SWEBENCH_LITE_REVISION,
        "live": bool(live),
    }


def _same(left, right):
    return left == right


def _write_manifest(run_dir, payload):
    path = Path(run_dir) / "manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("experiment") != EXPERIMENT:
            raise LearnStopped("fox_run_dir_experiment_mismatch")
        for key in _LOCKED:
            if key in existing and not _same(existing.get(key), payload.get(key)):
                raise LearnStopped("fox_run_settings_mismatch")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(payload) + "\n", encoding="utf-8")


def _done_ids(run_dir):
    done = set()
    for row in _load_jsonl(Path(run_dir) / "records.jsonl"):
        unit_id = row.get("unit_id")
        status = row.get("status")
        if isinstance(unit_id, str) and unit_id and status in _TERMINAL:
            done.add(unit_id)
    return done


def _spent(run_dir):
    total = 0.0
    for row in _load_jsonl(Path(run_dir) / "records.jsonl"):
        amount = row.get("cost")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            usage = row.get("usage")
            amount = usage.get("cost") if isinstance(usage, dict) else None
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            continue
        total += float(amount)
    return total


def _carry_parent(unit, services, run_dir, models):
    if unit.m2_condition != "persist" or not unit.parent_unit_id:
        return None
    export_path = Path(run_dir) / "checkpoints" / unit.parent_unit_id / "learning_state.json"
    if not export_path.is_file():
        return "parent_missing"
    try:
        value = json.loads(export_path.read_text(encoding="utf-8"))
        imported = import_learning_state(
            services.state_dir / "alo.sqlite3", value, allowed_models=models, mode="persist")
    except (OSError, ValueError, json.JSONDecodeError):
        return "parent_missing"
    return imported


def _export_checkpoint(chain, unit, services, adapter, run_dir, models):
    checkpoint = Path(run_dir) / "checkpoints" / unit.unit_id
    checkpoint.mkdir(parents=True, exist_ok=True)
    state = adapter.dump_state()
    (checkpoint / "adapter.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8")
    database = services.state_dir / "alo.sqlite3"
    if not database.is_file():
        return None
    _copy_sqlite(database, checkpoint / "alo.sqlite3")
    if unit.m2_condition != "persist":
        return None
    try:
        exported = export_learning_state(
            database, state, allowed_models=models, evidence_id=unit.unit_id,
            strip_quality=chain.strip_quality)
        (checkpoint / "learning_state.json").write_text(
            json.dumps(exported, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        if chain.condition == "persist":
            package = export_transfer_package(
                state, allowed_models=models, task_family="swe-repair",
                strip_quality=False)
            (checkpoint / "transfer_package.json").write_text(
                json.dumps(package, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return exported
    except (OSError, ValueError) as error:
        return {"export_error": type(error).__name__}


def _patch_text(services):
    root = services.workspace.root
    head = getattr(services, "image_head", None) or ""
    if not isinstance(head, str) or not head or not (root / ".git").is_dir():
        return ""
    try:
        from .collect import workspace_patch
        return workspace_patch(root, head)
    except (OSError, RuntimeError):
        return ""


def _task_row(chain, unit):
    for instance_id, repo, split in chain.tasks:
        if instance_id == unit.task_id:
            return instance_id, repo, split
    raise LearnStopped("unknown_instance:" + unit.task_id)


def _public_stage(chain, unit, data_dir, *, live):
    instance_id, repo, _split = _task_row(chain, unit)
    payload = load_public_payload(data_dir, "SWE-bench_Lite", instance_id)
    if payload is None:
        if live:
            return None
        payload = {"task_id": instance_id, "repo": repo,
                   "note": "offline placeholder; benchmark text is not loaded"}
    else:
        payload = dict(payload)
        payload.setdefault("repo", repo)
    return stage_for(unit, payload, dry_run=not live)


def run_chain(chain, run_dir, *, pool, live=False, sender=None, budget_cap=None,
              data_dir=None, execute=False, scorer=None, timeout_s=DEFAULT_READ_TIMEOUT_S,
              sleep=None, clock=None, context_policy="fail", context_caps=None,
              max_tokens=None, seed_images=True):
    """Run one chain in order. Completed unit ids are skipped."""
    if pool.get("kind") != "fox":
        raise LearnStopped("domestic_pool_uses_cny_ledger")
    run_dir = assert_fox_run_dir(run_dir)
    models = tuple(pool["models"])
    if tuple(row.internal for row in pool["specs"]) != models:
        raise LearnStopped("fox_pool_mismatch")
    split = chain.tasks[0][2]
    manifest = _manifest(chain, pool, budget_cap=budget_cap, split=split, live=bool(live and sender is None))
    _write_manifest(run_dir, manifest)
    transport = _transport_for(
        live=live, sender=sender, timeout_s=timeout_s,
        sleep=sleep or time.sleep,
        clock=clock or (lambda: datetime.now(timezone.utc)),
        max_attempts=4)
    settings = {
        row.internal: model_settings(row, context_caps=context_caps, max_tokens=max_tokens)
        for row in pool["specs"]
    }
    done = _done_ids(run_dir)
    spent = _spent(run_dir)
    written = []
    secrets = _live_secrets()
    limits = {"requests": chain.turn_cap, "tokens": 900_000, "tools": 120, "seconds": 1200}
    for unit in chain.units:
        if unit.unit_id in done:
            continue
        if budget_cap is not None and spent > float(budget_cap):
            break
        record = _run_unit(
            chain, unit, run_dir, models=models, specs=pool["specs"], settings=settings,
            transport=transport, limits=limits, live=live, data_dir=data_dir,
            execute=execute, scorer=scorer, budget_cap=budget_cap, spent_cost=spent,
            context_policy=context_policy, seed_images=seed_images)
        calls = record.pop("_calls", []) or []
        record = redact_secrets(record, secrets)
        _append_jsonl(run_dir / "records.jsonl", record)
        for call in calls:
            call = redact_secrets(dict(call), secrets)
            call.pop("text", None)
            call["unit_id"] = record["unit_id"]
            _append_jsonl(run_dir / "calls.jsonl", call)
        spent = _spent(run_dir)
        written.append(record)
        if record.get("status") == "blocked" and record.get("reason") == "parent_missing":
            break
    return {
        "experiment": EXPERIMENT,
        "chain_id": chain.chain_id,
        "run_id": chain.run_id,
        "run_dir": str(run_dir),
        "wrote": len(written),
        "units": len(chain.units),
        "skipped": len(chain.units) - len(written),
    }


def _run_unit(chain, unit, run_dir, *, models, specs, settings, transport, limits,
              live, data_dir, execute, scorer, budget_cap, spent_cost, context_policy,
              seed_images):
    location = next(row for row in chain_locations(chain, run_dir) if row["unit_id"] == unit.unit_id)
    record = {
        "schema": UNIT_SCHEMA,
        "experiment": EXPERIMENT,
        "unit_id": unit.unit_id,
        "chain_id": chain.chain_id,
        "run_id": chain.run_id,
        "arm": chain.arm,
        "condition": chain.condition,
        "order": chain.order,
        "m2_condition": chain.m2_condition,
        "task_id": unit.task_id,
        "position": unit.position,
        "parent_unit_id": unit.parent_unit_id,
        "pool": "fox",
        "models": list(models),
        "turn_cap": chain.turn_cap,
        "seed": unit.seed,
        "container_name": location["container_name"],
        "workdir": location["workdir"],
        "run_dir": str(run_dir),
        "prices": "none",
    }
    state_dir = Path(run_dir) / "units" / unit.unit_id / "state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    services = FoxAloServices(
        state_dir, specs, settings_by_model=settings, transport=transport, limits=limits,
        budget_cap=budget_cap, spent_cost=spent_cost, context_policy=context_policy,
        container_name=location["container_name"])
    adapter = adapter_for("G2", fox_learning_config(unit, models))
    imported = _carry_parent(unit, services, run_dir, models)
    if imported == "parent_missing":
        record.update(status="blocked", reason="parent_missing", harness_resolved=None,
                      profile_hash_before=None, profile_hash_after=None,
                      assignments=[], roles_by_model={})
        record["_calls"] = []
        return record
    if isinstance(imported, dict):
        adapter.state["observations"] = imported.get("observations") or {}
    record["profile_hash_before"] = snapshot_profile(
        services.state_dir / "alo.sqlite3", adapter.dump_state(), models,
        strip_quality=chain.strip_quality)
    instance_id, repo, split = _task_row(chain, unit)
    base_commit = None
    payload = load_public_payload(data_dir, "SWE-bench_Lite", instance_id)
    if isinstance(payload, dict):
        base_commit = payload.get("base_commit") if isinstance(payload.get("base_commit"), str) else None
        if isinstance(payload.get("repo"), str):
            repo = payload["repo"]
    prepare_note = prepare_workspace(
        services, instance_id=instance_id, repo=repo, base_commit=base_commit or "",
        live=live, seed_images=seed_images and bool(live))
    if services.sandbox is not None:
        services.sandbox.container_name = location["container_name"]
        record["sandbox_workdir"] = services.sandbox.workdir
    else:
        record["sandbox_workdir"] = None
    if prepare_note:
        record.update(status="blocked", reason=prepare_note, harness_resolved=None,
                      profile_hash_after=record["profile_hash_before"],
                      assignments=[], roles_by_model={})
        record["_calls"] = list(services.calls)
        return record
    stage = _public_stage(chain, unit, data_dir, live=live)
    if stage is None:
        record.update(status="blocked", reason="payload_missing", harness_resolved=None,
                      profile_hash_after=record["profile_hash_before"],
                      assignments=[], roles_by_model={})
        record["_calls"] = []
        return record
    status, reason = "completed", ""
    try:
        outcome = adapter.run_stage(stage, services)
        if isinstance(outcome, dict) and outcome.get("status") in {"completed", "failed"}:
            status = outcome["status"]
    except (PilotStopped, FoxStopped) as error:
        status, reason = "stopped", str(error)
    assignments, roles = _assignments(adapter)
    try:
        record["profile_hash_after"] = snapshot_profile(
            services.state_dir / "alo.sqlite3", adapter.dump_state(), models,
            strip_quality=chain.strip_quality)
    except (OSError, ValueError):
        record["profile_hash_after"] = None
    exported = _export_checkpoint(chain, unit, services, adapter, run_dir, models)
    if isinstance(exported, dict) and exported.get("export_error"):
        record["learning_export_error"] = exported["export_error"]
    patch = _patch_text(services)
    checkpoint = Path(run_dir) / "checkpoints" / unit.unit_id
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "model.patch").write_text(patch, encoding="utf-8")
    scored = score_prediction(
        run_dir, unit_id=unit.unit_id, instance_id=instance_id, split=split,
        patch=patch, execute=execute, scorer=scorer)
    usage = _usage_block(services.calls)
    record.update(
        status=status,
        reason=reason,
        assignments=assignments,
        node_assignments=_node_assignments(adapter),
        roles_by_model=roles,
        decision_records=_compact_decisions(adapter.state.get("decision_records")),
        role_tool_contracts=list(adapter.state.get("role_tool_contracts") or []),
        harness_resolved=scored.get("resolved") if isinstance(scored, dict) else None,
        harness_status=scored.get("status") if isinstance(scored, dict) else None,
        usage=usage,
        cost=usage["cost"],
        credit=usage["credit"],
        quota=usage["quota"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        reasoning_tokens=usage["reasoning_tokens"],
    )
    record["_calls"] = list(services.calls)
    return record


def export_chain_transfer(run_dir):
    """Run ``export_transfer_package`` on the last carried profile. Offline."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise LearnStopped("manifest_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != EXPERIMENT:
        raise LearnStopped("fox_run_dir_experiment_mismatch")
    if manifest.get("condition") == "reset":
        raise LearnStopped("reset_chain_has_no_carried_profile")
    models = tuple(manifest.get("models") or ())
    if not models:
        raise LearnStopped("fox_models_required")
    unit_ids = list(manifest.get("unit_ids") or ())
    if not unit_ids:
        raise LearnStopped("empty_selection")
    last = None
    for unit_id in reversed(unit_ids):
        state_path = run_dir / "checkpoints" / unit_id / "adapter.json"
        if state_path.is_file():
            last = unit_id
            break
    if last is None:
        raise LearnStopped("profile_export_missing")
    state = json.loads((run_dir / "checkpoints" / last / "adapter.json").read_text(encoding="utf-8"))
    strip = manifest.get("arm") == "g2-resources" or manifest.get("condition") == "resources-only"
    package = export_transfer_package(
        state, allowed_models=models, task_family="swe-repair", strip_quality=strip)
    dest = run_dir / "transfer_package.json"
    dest.write_text(json.dumps(package, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return package


def run_isolated(payload):
    """One chain in this process. Used to start the six chains as separate processes."""
    return run_chain(
        payload["chain"], payload["run_dir"], pool=payload["pool"], live=False,
        sender=dry_gateway, budget_cap=payload.get("budget_cap"),
        sleep=lambda _seconds: None)


def build_stream(*, split="dev", task_ids=(), tasks_file=None, n=None, seed=DEFAULT_STREAM_SEED,
                 allow_all=False, turn_cap=DEFAULT_TURN_CAP, arms=(), conditions=(), orders=(),
                 models="all"):
    pool = resolve_alo_pool("fox", models)
    tasks = select_learning_tasks(
        split=split, task_ids=task_ids, tasks_file=tasks_file, n=n, seed=seed, allow_all=allow_all)
    chains = iter_learning_chains(
        tasks, turn_cap=turn_cap, stream_seed=seed, arms=arms, conditions=conditions, orders=orders)
    return {"pool": pool, "tasks": tasks, "chains": chains, "seed": seed,
            "turn_cap": turn_cap, "split": split}


def _parse_models(values, models_file):
    names = []
    if models_file:
        names.extend(_read_id_list(models_file))
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                names.append(part)
    if not names or names == ["all"]:
        return "all"
    if "all" in names:
        raise LearnStopped("models_all_is_exclusive")
    return names


def _add_selection(parser):
    parser.add_argument("--pool", default="fox", choices=("fox", "domestic", "mainland"))
    parser.add_argument("--models", action="append", default=[])
    parser.add_argument("--models-file", default=None)
    parser.add_argument("--split", default="dev", choices=("dev", "test", "both"))
    parser.add_argument("--tasks-file", default=None)
    parser.add_argument("--tasks", default=None, help="Comma-separated SWE-bench Lite instance ids.")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", default=DEFAULT_STREAM_SEED)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--turn-cap", type=int, default=DEFAULT_TURN_CAP)
    parser.add_argument("--arm", action="append", default=[], choices=("g2-learn", "g2-resources"))
    parser.add_argument("--condition", action="append", default=[],
                        choices=("persist", "reset", "resources-only"))
    parser.add_argument("--order", action="append", default=[], choices=("forward", "reverse"))
    parser.add_argument("--budget-cap", type=float, default=None)


def _selection(args):
    if args.pool != "fox":
        raise LearnStopped("domestic_pool_uses_cny_ledger")
    models = _parse_models(args.models, args.models_file)
    task_ids = ()
    if args.tasks:
        task_ids = tuple(part.strip() for part in args.tasks.split(",") if part.strip())
    try:
        built = build_stream(
            split=args.split, task_ids=task_ids, tasks_file=args.tasks_file, n=args.n,
            seed=args.seed, allow_all=args.all, turn_cap=args.turn_cap, arms=args.arm,
            conditions=args.condition, orders=args.order, models=models)
    except ValueError as error:
        raise LearnStopped(str(error)) from None
    return built


def _plan_payload(args, built):
    root = Path(args.root) if args.root else None
    chains = []
    names = list(built["pool"]["models"])
    model_arg = "all" if not args.models and not args.models_file else names
    for chain in built["chains"]:
        run_dir = (root / chain.chain_id) if root is not None else Path("/tmp") / EXPERIMENT / chain.chain_id
        chains.append({
            "chain_id": chain.chain_id,
            "run_id": chain.run_id,
            "arm": chain.arm,
            "condition": chain.condition,
            "order": chain.order,
            "m2_condition": chain.m2_condition,
            "run_dir": str(run_dir),
            "units": chain_locations(chain, run_dir),
            "command": launch_command(
                chain, run_dir, models=model_arg, tasks_file=args.tasks_file, n=args.n,
                seed=args.seed, split=args.split, turn_cap=args.turn_cap,
                data_dir=args.data_dir or "$FOX_DATA_DIR", budget_cap=args.budget_cap,
                live=True, allow_all=args.all,
                task_ids=[row[0] for row in built["tasks"]]),
        })
    return {
        "experiment": EXPERIMENT,
        "pool": "fox",
        "prices": "none",
        "budget": "raw_optional",
        "models": list(names),
        "turn_cap": built["turn_cap"],
        "task_ids": [row[0] for row in built["tasks"]],
        "chains": chains,
        "api_calls": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="ALO SWE-bench Lite learning stream on the fox pool.")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Print the six chain launch commands. No API call.")
    _add_selection(plan)
    plan.add_argument("--root", default=None)
    plan.add_argument("--data-dir", default=None)
    run = sub.add_parser("run", help="Run one chain. Pass --dry-run or --live.")
    _add_selection(run)
    run.add_argument("--chain", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--data-dir", default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--live", action="store_true")
    run.add_argument("--execute", action="store_true")
    export = sub.add_parser("export", help="export_transfer_package on the chain's final profile.")
    export.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            package = export_chain_transfer(args.run_dir)
            print(json.dumps({
                "schema": package.get("schema"),
                "task_family": package.get("task_family"),
                "path": str(Path(args.run_dir) / "transfer_package.json"),
                "api_calls": 0,
            }, sort_keys=True))
            return 0
        built = _selection(args)
        if args.command == "plan":
            print(json.dumps(_plan_payload(args, built), ensure_ascii=False, sort_keys=True))
            return 0
        if args.live == args.dry_run:
            parser.error("pass exactly one of --dry-run or --live")
        chosen = [chain for chain in built["chains"] if chain.chain_id == args.chain]
        if len(chosen) != 1:
            parser.error("unknown chain " + args.chain)
        result = run_chain(
            chosen[0], args.run_dir, pool=built["pool"], live=args.live,
            sender=None if args.live else dry_gateway, budget_cap=args.budget_cap,
            data_dir=args.data_dir, execute=args.execute and args.live)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (LearnStopped, FoxStopped, PilotStopped) as error:
        print(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
