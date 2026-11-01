"""Round-2 paid schedule. It is not comparable with round 1 and must not be pooled.

The task draw is deterministic from ``swebench_lite_frame`` and ``DRAW_SEED``.
It does not use round-1 scores. ProgramBench is not in the scored schedule.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import subprocess
from pathlib import Path
import sys

from .budget import MICROCNY_PER_CNY
from .contracts import G1_MAIN_MODEL, G1_STRONG_MODEL, MODEL_POOL
from .freeze import G2_BASE_SEED, PROVIDER_TIMEOUT_S, PlannedUnit, FREEZE_PATH
from .images import swebench_instance_image
from .prices import PRICE_TABLE
from .swebench_lite_frame import SWEBENCH_LITE_FRAME, SWEBENCH_LITE_REVISION


BATCH = "r2"
DRAW_SEED = "r2-task-draw-20261101-v1"
R2_COMMAND_SECONDS = 120
R2_REASONING_BUDGET = 8192
R2_M1_REQUESTS = 24
R2_M2_LEARNING_REQUESTS = 16
R2_M2_REFERENCE_REQUESTS = 16
R2_M2_PROBE_REQUESTS = 12
R2_STRONG_M1_COUNT = 24
R2_PRO_SOFT_HALT_CNY = 48
# Round 2 has a fresh 50 CNY per model. The frozen carry-in is zero.
# ALO_PRIOR_LEDGER is opt-in; unset does not read a round-1 ledger.
PRIOR_SPEND_CNY = {model: "0.000" for model in MODEL_POOL}
PRIOR_SPEND_TOTAL_CNY = "0.000"
USER_PREFERENCE = {"first_try": "deepseek-flash", "strong": "mimo-v2.6-pro"}
# Not a freeze field. These images are checked out at a commit that is not the
# task base and not a mode-only child of it, so the runner skips them.
# sphinx-doc__sphinx-7686: /testbed HEAD 2dee191, task base 752d328.
R2_IMAGE_BASE_UNAVAILABLE = ("sphinx-doc__sphinx-7686",)
# Infra only. Not part of the freeze payload. The read budget equals frozen
# provider_timeout_s, measured from connect success. Connect fails fast when
# no socket is established. A gap with no bytes is an idle timeout and does
# not extend or replace that total deadline.
R2_READ_TIMEOUT_S = dict(PROVIDER_TIMEOUT_S)
R2_CONNECT_TIMEOUT_S = 15
R2_STREAM_IDLE_TIMEOUT_S = {model: 120 for model in MODEL_POOL}
# Round-1 settled means, used only to print a cost estimate. Not a draw input.
R1_CNY_PER_REQUEST = {
    "deepseek-flash": 0.009,
    "mimo-v2.6-flash": 0.005,
    "glm-5.3-flash": 0.015,
    "mimo-v2.6-pro": 0.0335,
}

# Round-1 SWE identities. Held out of the extra draw. Strata are the r1 labels.
R1_M1 = (
    ("scikit-learn__scikit-learn-13584", "short"),
    ("django__django-11583", "short"),
    ("sympy__sympy-11870", "medium"),
    ("django__django-16408", "medium"),
    ("sympy__sympy-20322", "long"),
    ("sympy__sympy-18835", "long"),
)
R1_M2_LEARNING = (
    ("sympy__sympy-18621", "short"),
    ("django__django-15851", "short"),
    ("scikit-learn__scikit-learn-11040", "medium"),
    ("django__django-11019", "long"),
)
R1_M2_PROBES = (
    ("django__django-13590", "medium"),
    ("sympy__sympy-21055", "long"),
)
R1_HELD = {task_id for task_id, _stratum in (*R1_M1, *R1_M2_LEARNING, *R1_M2_PROBES)}

# Order-2 is the r1 permutation of the learning list. Probe seeds for the two
# old probes stay the r1 seeds. New probes get seeds recorded below.
LEARNING_ORDER_2 = (2, 0, 3, 1)
LEARNING_SEEDS = {
    "order-1": (2026102800, 2026102801, 2026102802, 2026102803),
    "order-2": (2026102810, 2026102811, 2026102812, 2026102813),
}
OLD_PROBE_SEEDS = {
    "order-1": (2026102900, 2026102901),
    "order-2": (2026102910, 2026102911),
}
NEW_PROBE_SEEDS = {
    "order-1": (2026110100, 2026110101, 2026110102, 2026110103),
    "order-2": (2026110110, 2026110111, 2026110112, 2026110113),
}

SCHEMA = "alo-protocol-freeze-r2-v1"
R2_DOC = Path(__file__).resolve().parents[2] / "docs" / "protocol_freeze_r2.md"


def _rank(seed, salt, repo, instance_id):
    return hashlib.sha256(f"{seed}|{salt}|{repo}|{instance_id}".encode()).hexdigest()


def stratified_sample(pool, count, seed, salt):
    """Largest-remainder quotas by repository, then a seeded rank inside each repo.

    ``pool`` is a sequence of ``(instance_id, repo)``. Ties break by repo name,
    then by the rank. The input order does not matter.
    """
    if type(count) is not int or count < 0:
        raise ValueError("invalid_draw_count")
    rows = sorted({(instance_id, repo) for instance_id, repo in pool})
    if count > len(rows):
        raise ValueError("draw_exceeds_pool")
    if count == 0:
        return ()
    by_repo = {}
    for instance_id, repo in rows:
        by_repo.setdefault(repo, []).append(instance_id)
    total = len(rows)
    raw = {repo: count * len(ids) / total for repo, ids in by_repo.items()}
    base = {repo: math.floor(value) for repo, value in raw.items()}
    left = count - sum(base.values())
    fractional = sorted(by_repo, key=lambda repo: (-(raw[repo] - base[repo]), repo))
    for repo in fractional:
        if left <= 0:
            break
        base[repo] += 1
        left -= 1
    chosen = []
    for repo in sorted(by_repo):
        ranked = sorted(by_repo[repo], key=lambda instance_id: _rank(seed, salt, repo, instance_id))
        chosen.extend((instance_id, repo) for instance_id in ranked[:base[repo]])
    chosen.sort(key=lambda item: _rank(seed, salt, item[1], item[0]))
    return tuple(chosen)


def _frame_lookup():
    return {instance_id: repo for instance_id, repo in SWEBENCH_LITE_FRAME}


def draw_r2_tasks():
    """Return m1 extras, new probes, and the strong subset. Pure function of the frame."""
    lookup = _frame_lookup()
    missing = sorted(task_id for task_id in R1_HELD if task_id not in lookup)
    if missing:
        raise ValueError("r1 task missing from swebench lite frame")
    eligible = [(instance_id, repo) for instance_id, repo in SWEBENCH_LITE_FRAME if instance_id not in R1_HELD]
    new_probes = stratified_sample(eligible, 4, DRAW_SEED, "m2-probes")
    probe_ids = {instance_id for instance_id, _repo in new_probes}
    m1_pool = [(instance_id, repo) for instance_id, repo in eligible if instance_id not in probe_ids]
    m1_extra = stratified_sample(m1_pool, 42, DRAW_SEED, "m1-extra")
    strong_extra = stratified_sample(
        m1_extra, R2_STRONG_M1_COUNT - len(R1_M1), DRAW_SEED, "g1-strong")
    return {"new_probes": new_probes, "m1_extra": m1_extra, "strong_extra": strong_extra}


def m1_tasks():
    drawn = draw_r2_tasks()
    rows = list(R1_M1)
    for instance_id, repo in drawn["m1_extra"]:
        rows.append((instance_id, repo.split("/")[-1]))
    return tuple(rows)


def strong_task_ids():
    drawn = draw_r2_tasks()
    return tuple(task_id for task_id, _stratum in R1_M1) + tuple(
        instance_id for instance_id, _repo in drawn["strong_extra"])


def probe_tasks():
    drawn = draw_r2_tasks()
    rows = list(R1_M2_PROBES)
    for instance_id, repo in drawn["new_probes"]:
        rows.append((instance_id, repo.split("/")[-1]))
    return tuple(rows)


def reference_tasks():
    return tuple(R1_M2_LEARNING) + tuple(probe_tasks())


def _probe_prior_order(sequence_index, probe_position):
    if (sequence_index + probe_position) % 2 == 0:
        return ("learned", "cold")
    return ("cold", "learned")


def _learning_ids(order_id):
    ids = tuple(task_id for task_id, _stratum in R1_M2_LEARNING)
    if order_id == "order-1":
        return ids
    return tuple(ids[index] for index in LEARNING_ORDER_2)


def _probe_ids():
    return tuple(task_id for task_id, _stratum in probe_tasks())


def _stratum_for(task_id):
    for collection in (R1_M1, R1_M2_LEARNING, R1_M2_PROBES, m1_tasks(), probe_tasks()):
        for identity, stratum in collection:
            if identity == task_id:
                return stratum
    return "repo"


def iter_r2_m1_units():
    strong = set(strong_task_ids())
    for index, (task_id, stratum) in enumerate(m1_tasks()):
        arms = ["g1-main", "g2", "g3"]
        if task_id in strong:
            arms.append("g1-strong")
        offset = index % len(arms)
        order = arms[offset:] + arms[:offset]
        for arm in order:
            yield PlannedUnit(
                study="m1", arm=arm, track="SWE-bench_Lite", task_id=task_id, stratum=stratum,
                phase="B0", position=index, seed=G2_BASE_SEED + index,
                request_cap=R2_M1_REQUESTS,
            )


def _learning_units(order_id, sequence_index):
    first = "persist" if order_id == "order-1" else "reset"
    other = "reset" if first == "persist" else "persist"
    previous = ""
    units = []
    for position, task_id in enumerate(_learning_ids(order_id)):
        arms = (first, other) if position % 2 == 0 else (other, first)
        for condition in arms:
            parent = previous if condition == "persist" else ""
            unit = PlannedUnit(
                study="m2", arm="g2", track="SWE-bench_Lite", task_id=task_id,
                stratum=_stratum_for(task_id), phase="learning", position=position,
                seed=LEARNING_SEEDS[order_id][position], request_cap=R2_M2_LEARNING_REQUESTS,
                order_id=order_id, m2_condition=condition, parent_unit_id=parent,
            )
            units.append(unit)
            if condition == "persist":
                previous = unit.unit_id
    return units, previous


def iter_r2_m2_units():
    """Learning, then g1-main and g3 references, then probes, then G1-strong references.

    Strong references are last so the r2-only Pro soft halt cuts them first.
    """
    learning = []
    last_persist = {}
    for sequence_index, order_id in enumerate(("order-1", "order-2")):
        units, tail = _learning_units(order_id, sequence_index)
        learning.extend(units)
        last_persist[order_id] = tail
    yield from learning
    probes = _probe_ids()
    references = tuple(task_id for task_id, _stratum in reference_tasks())
    for index, task_id in enumerate(references):
        for arm in ("g1-main", "g3"):
            yield PlannedUnit(
                study="m2", arm=arm, track="SWE-bench_Lite", task_id=task_id,
                stratum=_stratum_for(task_id), phase="reference", position=index,
                seed=G2_BASE_SEED + 100 + index, request_cap=R2_M2_REFERENCE_REQUESTS,
            )
    for sequence_index, order_id in enumerate(("order-1", "order-2")):
        old_seeds = OLD_PROBE_SEEDS[order_id]
        new_seeds = NEW_PROBE_SEEDS[order_id]
        for probe_position, task_id in enumerate(probes):
            seed = (old_seeds[probe_position] if probe_position < len(old_seeds)
                    else new_seeds[probe_position - len(old_seeds)])
            for condition in _probe_prior_order(sequence_index, probe_position):
                parent = last_persist[order_id] if condition == "learned" else ""
                yield PlannedUnit(
                    study="m2", arm="g2", track="SWE-bench_Lite", task_id=task_id,
                    stratum=_stratum_for(task_id), phase="probe", position=probe_position,
                    seed=seed + (0 if condition == "learned" else 17),
                    request_cap=R2_M2_PROBE_REQUESTS, order_id=order_id,
                    m2_condition=condition, parent_unit_id=parent,
                )
    for index, task_id in enumerate(references):
        yield PlannedUnit(
            study="m2", arm="g1-strong", track="SWE-bench_Lite", task_id=task_id,
            stratum=_stratum_for(task_id), phase="reference", position=index,
            seed=G2_BASE_SEED + 200 + index, request_cap=R2_M2_REFERENCE_REQUESTS,
        )


def iter_r2_units(study):
    if study == "m1":
        yield from iter_r2_m1_units()
    elif study == "m2":
        yield from iter_r2_m2_units()
    else:
        raise ValueError("unknown_study")


def select_r2_units(study, *, arms=(), tasks=()):
    arms, tasks = set(arms), set(tasks)
    chosen = []
    for unit in iter_r2_units(study):
        if arms and unit.arm not in arms:
            continue
        if tasks and unit.task_id not in tasks:
            continue
        chosen.append(unit)
    return tuple(chosen)


def smoke_unit():
    """Optional ProgramBench smoke. Not part of the scored r2 schedule."""
    return PlannedUnit(
        study="m1", arm="g1-main", track="ProgramBench", task_id="kisielk__errcheck.dacab89",
        stratum="easy", phase="B0", position=0, seed=G2_BASE_SEED,
        request_cap=40,
    )


def _m2_chain_key(unit):
    """Persist state and the learned probes that read it share one chain id.

    Reset learning units, cold probes, and reference units have no parent.
    Each of those is its own chain so the four shards can run them together.
    """
    if unit.m2_condition in {"persist", "learned"}:
        return ("chain", unit.order_id or "")
    return ("free", unit.unit_id)


_SHARD_ASSIGNMENT = {}


def _r2_assignment(num_shards):
    """Deterministic shard. M1 stays one task per group, in catalog order.

    M2 used to be a single group after those tasks, so with 48 M1 tasks and
    4 shards every M2 unit landed on shard 0. Chain groups and free units
    continue that same round-robin. This map is not part of the freeze payload.
    """
    cached = _SHARD_ASSIGNMENT.get(num_shards)
    if cached is not None:
        return cached
    assignment = {}
    group_index = 0
    seen_tasks = set()
    for item in iter_r2_m1_units():
        key = (item.track, item.task_id)
        if key in seen_tasks:
            continue
        seen_tasks.add(key)
        assignment[("task",) + key] = group_index % num_shards
        group_index += 1
    seen_chains = set()
    for item in iter_r2_m2_units():
        chain = _m2_chain_key(item)
        if chain[0] == "chain":
            if chain not in seen_chains:
                seen_chains.add(chain)
                assignment[chain] = group_index % num_shards
                group_index += 1
            assignment[("unit", item.unit_id)] = assignment[chain]
        else:
            assignment[("unit", item.unit_id)] = group_index % num_shards
            group_index += 1
    _SHARD_ASSIGNMENT[num_shards] = assignment
    return assignment


def r2_shard_of(unit, num_shards):
    if type(num_shards) is not int or num_shards < 1:
        raise ValueError("invalid_shard")
    assignment = _r2_assignment(num_shards)
    if unit.study == "m1":
        return assignment[("task", unit.track, unit.task_id)]
    return assignment[("unit", unit.unit_id)]


def filter_r2_shard(units, index, num_shards):
    """Shard slice in the original schedule order, so a parent still runs first."""
    if type(index) is not int or type(num_shards) is not int or num_shards < 1 or not 0 <= index < num_shards:
        raise ValueError("invalid_shard")
    return tuple(unit for unit in units if r2_shard_of(unit, num_shards) == index)


def cny_text_to_micro(text):
    """Milli-or-finer CNY text to microcny, without binary float rounding."""
    if not isinstance(text, str) or text.count(".") != 1:
        raise ValueError("prior_spend_cny")
    whole, frac = text.split(".")
    if not whole.isdigit() or not frac.isdigit() or not 1 <= len(frac) <= 6:
        raise ValueError("prior_spend_cny")
    return int(whole) * MICROCNY_PER_CNY + int(frac.ljust(6, "0"))


def frozen_prior_microcny():
    """Declared round-1 settled spend, keyed by request model name."""
    found = {model: cny_text_to_micro(PRIOR_SPEND_CNY[model]) for model in MODEL_POOL}
    if set(found) != set(MODEL_POOL):
        raise ValueError("prior_spend_pool")
    return found


def read_settled_prior(path):
    """Settled CNY totals from a ledger file. The file is opened read-only.

    Opening commitments in that file are included. An unsettled request row
    refuses the read so a partial r1 ledger cannot under-count the carry-in.
    """
    from .budget import SCHEMA
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("prior_ledger_unreadable")
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        schema = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if schema is None or schema[0] != SCHEMA:
            raise ValueError("prior_ledger_schema")
        opening = json.loads(conn.execute("SELECT value FROM meta WHERE key='opening'").fetchone()[0])
        rows = list(conn.execute("SELECT model, held, status FROM requests"))
    finally:
        conn.close()
    if any(status != "settled" for _model, _held, status in rows):
        raise ValueError("prior_ledger_unsettled")
    totals = {model: int(opening.get(model, 0)) for model in MODEL_POOL}
    for model, held, _status in rows:
        if model not in totals:
            raise ValueError("prior_ledger_model")
        totals[model] += int(held)
    return totals


def resolve_prior_spend(environ=None):
    """Zero carry-in unless ``ALO_PRIOR_LEDGER`` is set.

    Unset means the frozen zeros. A set path is read-only and its settled
    totals become the opening commitments. An unsettled row refuses the read.
    """
    source = os.environ if environ is None else environ
    frozen = frozen_prior_microcny()
    raw_path = source.get("ALO_PRIOR_LEDGER") if source is not None else None
    ledger_totals = None
    applied = dict(frozen)
    if isinstance(raw_path, str) and raw_path.strip():
        ledger_totals = read_settled_prior(raw_path.strip())
        applied = {model: int(ledger_totals[model]) for model in MODEL_POOL}
    return {
        "frozen_microcny": frozen,
        "applied_microcny": applied,
        "ledger_path": raw_path.strip() if isinstance(raw_path, str) and raw_path.strip() else None,
        "ledger_settled_microcny": ledger_totals,
    }


def r2_model_soft_halts():
    return {"mimo-v2.6-pro": R2_PRO_SOFT_HALT_CNY * MICROCNY_PER_CNY}


def _prices_for_payload():
    exported = {}
    for model in MODEL_POOL:
        row = PRICE_TABLE[model]
        charge = row["charge"]
        exported[model] = {
            "cache_miss_input_cny_per_million": charge["cache_miss_input_cny_per_million"],
            "output_cny_per_million": charge["output_cny_per_million"],
        }
    return exported


def r2_freeze_payload():
    drawn = draw_r2_tasks()
    return {
        "schema": SCHEMA,
        "batch": BATCH,
        "comparable_with_r1": False,
        "poolable_with_r1": False,
        "swebench_lite_revision": SWEBENCH_LITE_REVISION,
        "draw_seed": DRAW_SEED,
        "draw_rule": "largest_remainder_by_repo_then_sha256_rank",
        "models": {
            "pool": list(MODEL_POOL),
            "g1_main": G1_MAIN_MODEL,
            "g1_strong": G1_STRONG_MODEL,
            "g2_planner": "deepseek-flash",
            "model_name_deepseek": "deepseek-flash",
        },
        "budget_cny": {
            "per_model": 50,
            "total": 200,
            "cumulative_with_r1": False,
            "prior_spend_cny": dict(PRIOR_SPEND_CNY),
            "prior_spend_total_cny": PRIOR_SPEND_TOTAL_CNY,
            "prior_ledger_env": "ALO_PRIOR_LEDGER",
            "prior_ledger_default": "unset_is_zero",
        },
        "soft_halt_cny": {
            "mimo-v2.6-pro": R2_PRO_SOFT_HALT_CNY,
            "hard": False,
            "when": "between_units",
            "basis": "r2_spend_plus_optional_prior",
        },
        "g1_strong_m1_count": R2_STRONG_M1_COUNT,
        "g1_strong_m2_references": "last",
        "user_preference": dict(USER_PREFERENCE),
        "pro_g2_g3_share_cap": "none",
        "command_seconds": R2_COMMAND_SECONDS,
        "reasoning_budget_tokens": R2_REASONING_BUDGET,
        "request_caps": {
            "m1_b0": R2_M1_REQUESTS,
            "m2_learning": R2_M2_LEARNING_REQUESTS,
            "m2_reference": R2_M2_REFERENCE_REQUESTS,
            "m2_probe": R2_M2_PROBE_REQUESTS,
            "m1_b1": "dropped",
        },
        "program_bench_scored": False,
        "program_bench_smoke": {"task_id": "kisielk__errcheck.dacab89", "arm": "g1-main", "phase": "B0", "requests": 40},
        "m1_tasks": [{"task_id": task_id, "stratum": stratum} for task_id, stratum in m1_tasks()],
        "strong_task_ids": list(strong_task_ids()),
        "m2_learning": [task_id for task_id, _stratum in R1_M2_LEARNING],
        "m2_probes": [task_id for task_id, _stratum in probe_tasks()],
        "new_probes": [instance_id for instance_id, _repo in drawn["new_probes"]],
        "m1_extra": [instance_id for instance_id, _repo in drawn["m1_extra"]],
        "prices_cny_per_million_tokens": _prices_for_payload(),
        "parent_rule": "m1_b1_patch_only; m2_whitelist_statistics; missing_parent_blocks_chain",
        "schedule_order": "m1_then_m2_learning_then_g1main_g3_references_then_probes_then_g1strong_references",
        "r1_freeze_doc": str(FREEZE_PATH.name),
    }


def canonical_bytes(payload=None):
    body = r2_freeze_payload() if payload is None else payload
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def r2_freeze_sha256():
    return hashlib.sha256(canonical_bytes()).hexdigest()


def frame_sha256():
    raw = json.dumps(list(SWEBENCH_LITE_FRAME), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def task_ids_sha256():
    ids = [task_id for task_id, _stratum in m1_tasks()]
    ids.extend(task_id for task_id, _stratum in reference_tasks())
    raw = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def git_commit(root=None):
    root = Path(__file__).resolve().parents[2] if root is None else Path(root)
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def r2_manifest(root=None):
    return {
        "schema": "alo-r2-manifest-v1",
        "batch": BATCH,
        "comparable_with_r1": False,
        "poolable_with_r1": False,
        "commit": git_commit(root),
        "freeze_sha256": r2_freeze_sha256(),
        "frame_sha256": frame_sha256(),
        "task_ids_sha256": task_ids_sha256(),
        "swebench_lite_revision": SWEBENCH_LITE_REVISION,
        "draw_seed": DRAW_SEED,
        "prior_spend": resolve_prior_spend(),
    }


def write_r2_manifest(run_dir, root=None):
    path = Path(run_dir) / "r2_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r2_manifest(root), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def new_task_images():
    """Images for tasks that were not in the round-1 SWE schedule."""
    drawn = draw_r2_tasks()
    ids = [instance_id for instance_id, _repo in drawn["m1_extra"]]
    ids.extend(instance_id for instance_id, _repo in drawn["new_probes"])
    return tuple(swebench_instance_image(instance_id) for instance_id in ids)


def _money(calls):
    prior = {model: cny_text_to_micro(PRIOR_SPEND_CNY[model]) / MICROCNY_PER_CNY for model in MODEL_POOL}
    new = {model: calls[model] * R1_CNY_PER_REQUEST[model] for model in MODEL_POOL}
    cumulative = {model: new[model] + prior[model] for model in MODEL_POOL}
    return {
        "calls": calls,
        "cny_new": new,
        "cny_cumulative": cumulative,
        "total_cny_new": sum(new.values()),
        "total_cny_cumulative": sum(cumulative.values()),
    }


def cost_estimate():
    """Full-cap CNY at the round-1 mean price. Prior is the frozen zero carry-in.

    Expected: the first try holds. G2 planner and worker calls are
    ``deepseek-flash``. G1 is pinned. G3 is an even four-way split. A second
    figure, ``escalated``, assumes every G2 unit then spends its 3 reserved
    requests on ``mimo-v2.6-pro``. Worst case gives each model every flexible
    call it is allowed to take: G3's whole cap, and every G2 call except the
    planner and one first-try deepseek call. Not fit to round-1 scores.
    """
    units = tuple(iter_r2_m1_units()) + tuple(iter_r2_m2_units())
    expected = {model: 0.0 for model in MODEL_POOL}
    escalated = {model: 0.0 for model in MODEL_POOL}
    pinned = {model: 0.0 for model in MODEL_POOL}
    flexible = 0.0
    forced_deepseek = 0.0
    for unit in units:
        cap = unit.request_cap
        if unit.arm == "g1-main":
            expected["mimo-v2.6-flash"] += cap
            escalated["mimo-v2.6-flash"] += cap
            pinned["mimo-v2.6-flash"] += cap
        elif unit.arm == "g1-strong":
            expected["mimo-v2.6-pro"] += cap
            escalated["mimo-v2.6-pro"] += cap
            pinned["mimo-v2.6-pro"] += cap
        elif unit.arm == "g2":
            expected["deepseek-flash"] += cap
            escalated["deepseek-flash"] += cap - 3
            escalated["mimo-v2.6-pro"] += 3
            pinned["deepseek-flash"] += 2
            flexible += cap - 2
        elif unit.arm == "g3":
            flexible += cap
            for model in MODEL_POOL:
                share = cap / len(MODEL_POOL)
                expected[model] += share
                escalated[model] += share
    worst = {model: pinned[model] + flexible for model in MODEL_POOL}
    prior = {model: cny_text_to_micro(PRIOR_SPEND_CNY[model]) / MICROCNY_PER_CNY for model in MODEL_POOL}
    return {
        "expected": _money(expected),
        "escalated": _money(escalated),
        "worst_case": _money(worst),
        "prior_cny": prior,
        "pro_new_headroom_to_48_cny": R2_PRO_SOFT_HALT_CNY - prior["mimo-v2.6-pro"],
        "rates_cny_per_request": dict(R1_CNY_PER_REQUEST),
        "units": len(units),
        "user_preference": dict(USER_PREFERENCE),
        "note": "round-1 mean CNY per request at a full unit cap; not used to choose tasks",
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    digest = r2_freeze_sha256()
    if "--print" in argv:
        print(digest)
        return 0
    if "--images" in argv:
        for image in new_task_images():
            print(image)
        return 0
    if "--cost" in argv:
        print(json.dumps(cost_estimate(), sort_keys=True))
        return 0
    if "--manifest" in argv:
        print(json.dumps(r2_manifest(), sort_keys=True))
        return 0
    if "--check" not in argv:
        print("usage: python -m alo_cloud.r2 --check", file=sys.stderr)
        return 2
    text = R2_DOC.read_text(encoding="utf-8")
    if f"freeze_sha256: {digest}" not in text:
        print("r2 protocol freeze hash mismatch", file=sys.stderr)
        return 1
    for task_id, _stratum in m1_tasks():
        if task_id not in text:
            print("r2 protocol freeze is missing an m1 task id", file=sys.stderr)
            return 1
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
