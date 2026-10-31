"""Frozen paid-run schedule. The hash covers this payload, not a later edit.

Development tasks are absent on purpose: G1 Strong was selected by the owner,
so the development tournament is not part of the paid schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

from .contracts import G1_MAIN_MODEL, G1_STRONG_MODEL, MODEL_POOL
from .prices import PRICE_TABLE, RETRIEVED_ON


FREEZE_PATH = Path(__file__).resolve().parents[2] / "docs" / "protocol_freeze.md"
G2_PLANNER_MODEL = "deepseek-flash"
G3_MANAGER_MODEL = "deepseek-flash"
G3_ROLES = {
    "Planner": "deepseek-flash",
    "Engineer": "mimo-v2.6-flash",
    "Critic": "glm-5.3-flash",
    "Verifier": "mimo-v2.6-pro",
}
G2_BASE_SEED = 20261027
SELECTION_SEED = "formal-20261028-preregistered-task-selection-v1"
TOKEN_ESTIMATE = {"input_tokens": 12000, "output_tokens": 2048, "cache_hit_tokens": None}
CALL_CAPS = {
    "m1_g1_b0": 6,
    "m1_g1_b1": 3,
    "m1_g2_b0": 8,
    "m1_g2_b1": 4,
    "m1_g3_b0": 8,
    "m1_g3_b1": 4,
    "m2_g2_learning": 6,
    "m2_g1_reference": 6,
    "m2_g3_reference": 6,
    "m2_g2_probe": 4,
}
M1_CONDITIONS = ("g1-main", "g1-strong", "g2", "g3")
SCHEDULE_RULE = (
    "m1-track-rotation-v1; m2-all-learning-then-references-then-probes-v1; "
    "probe prior alternates by sequence index plus probe position"
)

# Catalog groups SWE-bench_Lite/m1, ProgramBench/m1, and the M2 orders.
# Development groups are not copied here.
TASKS = {
    "SWE-bench_Lite": {
        "m1": (
            ("scikit-learn__scikit-learn-13584", "short"),
            ("django__django-11583", "short"),
            ("sympy__sympy-11870", "medium"),
            ("django__django-16408", "medium"),
            ("sympy__sympy-20322", "long"),
            ("sympy__sympy-18835", "long"),
        ),
        "m2_learning": (
            ("sympy__sympy-18621", "short"),
            ("django__django-15851", "short"),
            ("scikit-learn__scikit-learn-11040", "medium"),
            ("django__django-11019", "long"),
        ),
        "m2_probe": (
            ("django__django-13590", "medium"),
            ("sympy__sympy-21055", "long"),
        ),
    },
    "ProgramBench": {
        "m1": (
            ("kisielk__errcheck.dacab89", "easy"),
            ("mgdm__htmlq.6e31bc8", "easy"),
            ("esubaalew__run.0fb9dec", "medium"),
            ("raviqqe__muffet.a882908", "medium"),
            ("ariga__atlas.6d81150", "hard"),
            ("unhappychoice__gittype.34b72d0", "hard"),
        ),
        "m2_learning": (
            ("arthursonzogni__json-tui.17a22b6", "easy"),
            ("xorg62__tty-clock.f2f847c", "easy"),
            ("o2sh__onefetch.e5958ce", "medium"),
            ("johnkerl__miller.8d85b46", "hard"),
        ),
        "m2_probe": (
            ("noborus__trdsql.d8c5ff6", "medium"),
            ("ffmpeg__ffmpeg.360a402", "hard"),
        ),
    },
}

M2_SEQUENCES = (
    {
        "track": "SWE-bench_Lite",
        "id": "order-1",
        "first_arm": "persist",
        "learning_ids": (
            "sympy__sympy-18621",
            "django__django-15851",
            "scikit-learn__scikit-learn-11040",
            "django__django-11019",
        ),
        "seeds": (2026102800, 2026102801, 2026102802, 2026102803),
        "probe_ids": ("django__django-13590", "sympy__sympy-21055"),
        "probe_seeds": (2026102900, 2026102901),
    },
    {
        "track": "SWE-bench_Lite",
        "id": "order-2",
        "first_arm": "reset",
        "learning_ids": (
            "scikit-learn__scikit-learn-11040",
            "sympy__sympy-18621",
            "django__django-11019",
            "django__django-15851",
        ),
        "seeds": (2026102810, 2026102811, 2026102812, 2026102813),
        "probe_ids": ("django__django-13590", "sympy__sympy-21055"),
        "probe_seeds": (2026102910, 2026102911),
    },
    {
        "track": "ProgramBench",
        "id": "order-1",
        "first_arm": "persist",
        "learning_ids": (
            "arthursonzogni__json-tui.17a22b6",
            "xorg62__tty-clock.f2f847c",
            "o2sh__onefetch.e5958ce",
            "johnkerl__miller.8d85b46",
        ),
        "seeds": (2026102900, 2026102901, 2026102902, 2026102903),
        "probe_ids": ("noborus__trdsql.d8c5ff6", "ffmpeg__ffmpeg.360a402"),
        "probe_seeds": (2026103000, 2026103001),
    },
    {
        "track": "ProgramBench",
        "id": "order-2",
        "first_arm": "reset",
        "learning_ids": (
            "o2sh__onefetch.e5958ce",
            "arthursonzogni__json-tui.17a22b6",
            "johnkerl__miller.8d85b46",
            "xorg62__tty-clock.f2f847c",
        ),
        "seeds": (2026102910, 2026102911, 2026102912, 2026102913),
        "probe_ids": ("noborus__trdsql.d8c5ff6", "ffmpeg__ffmpeg.360a402"),
        "probe_seeds": (2026103010, 2026103011),
    },
)


@dataclass(frozen=True)
class PlannedUnit:
    study: str
    arm: str
    track: str
    task_id: str
    stratum: str
    phase: str
    position: int
    seed: int
    request_cap: int
    order_id: str = ""
    m2_condition: str = ""
    parent_unit_id: str = ""

    @property
    def unit_id(self):
        parts = [self.study, self.track, self.task_id, self.arm, self.phase]
        if self.order_id:
            parts.append(self.order_id)
        if self.m2_condition:
            parts.append(self.m2_condition)
        return ".".join(parts)

    @property
    def protocol_unit(self):
        if self.study == "m1" and self.phase == "B0":
            return "m1_b0"
        if self.study == "m1" and self.phase == "B1-self":
            return "m1_b1"
        return "m2"


def _prices_for_payload():
    exported = {}
    for model in MODEL_POOL:
        row = PRICE_TABLE[model]
        charge = row["charge"]
        exported[model] = {
            "source_url": row["source_url"],
            "retrieved_on": row["retrieved_on"],
            "charge_band": row["charge_band"],
            "cache_hit_input_cny_per_million": charge["cache_hit_input_cny_per_million"],
            "cache_miss_input_cny_per_million": charge["cache_miss_input_cny_per_million"],
            "output_cny_per_million": charge["output_cny_per_million"],
        }
    return exported


def freeze_payload():
    """JSON-stable description of the paid run. Expansion rules live in code tests."""
    tasks = {
        track: {bucket: [{"task_id": task_id, "stratum": stratum} for task_id, stratum in rows]
                for bucket, rows in buckets.items()}
        for track, buckets in TASKS.items()
    }
    sequences = []
    for row in M2_SEQUENCES:
        sequences.append({
            "track": row["track"],
            "id": row["id"],
            "first_arm": row["first_arm"],
            "learning_ids": list(row["learning_ids"]),
            "seeds": list(row["seeds"]),
            "probe_ids": list(row["probe_ids"]),
            "probe_seeds": list(row["probe_seeds"]),
        })
    return {
        "schema": "alo-protocol-freeze-v2",
        "price_retrieved_on": RETRIEVED_ON,
        "models": {
            "pool": list(MODEL_POOL),
            "g1_main": G1_MAIN_MODEL,
            "g1_strong": G1_STRONG_MODEL,
            "g2_planner": G2_PLANNER_MODEL,
            "g3_manager": G3_MANAGER_MODEL,
            "g3_roles": dict(G3_ROLES),
        },
        "prices_cny_per_million_tokens": _prices_for_payload(),
        "charge_rule": "ceil_microcny; missing cache split bills all input at cache-miss; deepseek-flash uses peak",
        "budget_cny": {"per_model": 50, "total": 200, "opening_commitments": 0},
        "token_estimate": dict(TOKEN_ESTIMATE),
        "max_completion_tokens": 4096,
        "call_caps": dict(CALL_CAPS),
        "selection_seed": SELECTION_SEED,
        "g2_base_seed": G2_BASE_SEED,
        "schedule_rule": SCHEDULE_RULE,
        "m1_conditions": list(M1_CONDITIONS),
        "tasks": tasks,
        "m2_sequences": sequences,
        "params_sent": {
            "glm-5.3-flash.thinking.type": "enabled",
            "mimo.max_completion_tokens": True,
            "sampling_temperature_sent": False,
        },
        "development_tasks": "not_in_paid_schedule",
        "program_bench_source_commit": "895f65916bc5ba34a915d36da321fde0301125ad",
        "program_bench_tests_revision": "de0ddfb637590c7ecb54fa0b5301f6dc7dfbcee5",
        "program_bench_hf_revision_env": "PROGRAMBENCH_HF_REVISION",
        "swebench_lite_revision": "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2",
        "swebench_package": "4.1.0",
        "swebench_dataset_name": "princeton-nlp/SWE-bench_Lite",
        "sandbox": {
            "network": "none",
            "memory": "4g",
            "pids_limit": "512",
            "tmpfs": "1g",
            "fallback_image_env": "ALO_SANDBOX_IMAGE",
            "swebench_image": "swebench/sweb.eval.x86_64.<id>:latest (__ rewritten to _1776_)",
            "programbench_image": "programbench/<id>:task_cleanroom_v6 (__ rewritten to _1776_)",
            "swe_mounts": ["/workspace", "/testbed"],
        },
        "scoring": {
            "swe_model_name_or_path": "unit_id",
            "swe_one_command_per_unit": True,
            "programbench_layout": "sealed/programbench/<unit_id>/<instance_id>/submission.tar.gz",
        },
    }


def canonical_bytes(payload=None):
    body = freeze_payload() if payload is None else payload
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def freeze_sha256():
    return hashlib.sha256(canonical_bytes()).hexdigest()


def all_task_ids():
    found = []
    for buckets in TASKS.values():
        for rows in buckets.values():
            found.extend(task_id for task_id, _stratum in rows)
    return tuple(found)


def _m1_cap(arm, phase):
    kind = "g1" if arm.startswith("g1-") else arm
    slot = "b0" if phase == "B0" else "b1"
    return CALL_CAPS[f"m1_{kind}_{slot}"]


def _stratum(track, bucket, task_id):
    for identity, stratum in TASKS[track][bucket]:
        if identity == task_id:
            return stratum
    raise KeyError(task_id)


def iter_m1_units():
    """Rotate arms inside each track. Every arm finishes B0 before its own B1."""
    positions = {}
    for track, buckets in TASKS.items():
        for index, (task_id, stratum) in enumerate(buckets["m1"]):
            offset = positions.get(track, 0)
            positions[track] = offset + 1
            order = M1_CONDITIONS[offset % len(M1_CONDITIONS):] + M1_CONDITIONS[:offset % len(M1_CONDITIONS)]
            for phase in ("B0", "B1-self"):
                for arm in order:
                    parent = ""
                    if phase == "B1-self":
                        parent = ".".join(( "m1", track, task_id, arm, "B0"))
                    yield PlannedUnit(
                        study="m1", arm=arm, track=track, task_id=task_id, stratum=stratum,
                        phase=phase, position=index, seed=G2_BASE_SEED + index,
                        request_cap=_m1_cap(arm, phase), parent_unit_id=parent,
                    )


def _probe_prior_order(sequence_index, probe_position):
    if (sequence_index + probe_position) % 2 == 0:
        return ("learned", "cold")
    return ("cold", "learned")


def iter_m2_units():
    """All learning streams, then each reference once, then terminal probes."""
    learning = []
    for sequence_index, sequence in enumerate(M2_SEQUENCES):
        first = sequence["first_arm"]
        other = "reset" if first == "persist" else "persist"
        previous_persist = ""
        for position, task_id in enumerate(sequence["learning_ids"]):
            arms = (first, other) if position % 2 == 0 else (other, first)
            stratum = _stratum(sequence["track"], "m2_learning", task_id)
            for condition in arms:
                parent = previous_persist if condition == "persist" else ""
                unit = PlannedUnit(
                    study="m2", arm="g2", track=sequence["track"], task_id=task_id,
                    stratum=stratum, phase="learning", position=position,
                    seed=sequence["seeds"][position], request_cap=CALL_CAPS["m2_g2_learning"],
                    order_id=sequence["id"], m2_condition=condition, parent_unit_id=parent,
                )
                learning.append(unit)
                if condition == "persist":
                    previous_persist = unit.unit_id
    yield from learning
    for track, buckets in TASKS.items():
        references = list(buckets["m2_learning"]) + list(buckets["m2_probe"])
        for index, (task_id, stratum) in enumerate(references):
            bucket = "m2_learning" if index < len(buckets["m2_learning"]) else "m2_probe"
            del bucket
            for arm, cap_name in (
                ("g1-main", "m2_g1_reference"),
                ("g1-strong", "m2_g1_reference"),
                ("g3", "m2_g3_reference"),
            ):
                yield PlannedUnit(
                    study="m2", arm=arm, track=track, task_id=task_id, stratum=stratum,
                    phase="reference", position=index, seed=G2_BASE_SEED + 100 + index,
                    request_cap=CALL_CAPS[cap_name],
                )
    for sequence_index, sequence in enumerate(M2_SEQUENCES):
        for probe_position, task_id in enumerate(sequence["probe_ids"]):
            stratum = _stratum(sequence["track"], "m2_probe", task_id)
            for condition in _probe_prior_order(sequence_index, probe_position):
                parent = ""
                if condition == "learned":
                    parent = [
                        unit.unit_id for unit in learning
                        if unit.track == sequence["track"] and unit.order_id == sequence["id"]
                        and unit.m2_condition == "persist"
                    ][-1]
                yield PlannedUnit(
                    study="m2", arm="g2", track=sequence["track"], task_id=task_id,
                    stratum=stratum, phase="probe", position=probe_position,
                    seed=sequence["probe_seeds"][probe_position] + (0 if condition == "learned" else 17),
                    request_cap=CALL_CAPS["m2_g2_probe"], order_id=sequence["id"],
                    m2_condition=condition, parent_unit_id=parent,
                )


def iter_units(study):
    if study == "m1":
        yield from iter_m1_units()
    elif study == "m2":
        yield from iter_m2_units()
    else:
        raise ValueError("unknown_study")


def select_units(study, *, arms=(), tasks=()):
    arms, tasks = set(arms), set(tasks)
    chosen = []
    for unit in iter_units(study):
        if arms and unit.arm not in arms:
            continue
        if tasks and unit.task_id not in tasks:
            continue
        chosen.append(unit)
    return tuple(chosen)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    digest = freeze_sha256()
    if "--print" in argv:
        print(digest)
        return 0
    if "--check" not in argv:
        print("usage: python -m alo_cloud.freeze --check", file=sys.stderr)
        return 2
    text = FREEZE_PATH.read_text(encoding="utf-8")
    if f"freeze_sha256: {digest}" not in text:
        print("protocol freeze hash mismatch", file=sys.stderr)
        return 1
    missing = [task_id for task_id in all_task_ids() if task_id not in text]
    if missing:
        print("protocol freeze is missing task ids", file=sys.stderr)
        return 1
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
