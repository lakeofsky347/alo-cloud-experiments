"""Project the next round from r2's observed per-unit spend.

The frozen r2 schedule is 168 M1 units and 70 M2 units. The follow-up adds
three arms on the same tasks: ``g1-deepseek-flash``, ``g2-fixed`` and
``g2-random``. Prices are the observed CNY per request of each model in the
ledger, not the round-1 planning rates. The 50 CNY cap is applied to this
projection alone. It does not add the projection on top of the r2 ledger.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .contracts import G1_DEEPSEEK_FLASH_MODEL, MODEL_POOL


PER_MODEL_CAP_CNY = 50.0
R2_M1_UNITS = 168
R2_M2_UNITS = 70


def _micro_to_cny(value):
    return int(value) / 1_000_000


def load_records(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_eval_dir(path):
    path = Path(path)
    found = []
    for candidate in (
        path / "m1_units.csv", path / "m1" / "m1_units.csv", path / "m1" / "units.csv",
        path / "m2_units.csv", path / "m2" / "m2_units.csv", path / "m2" / "units.csv",
    ):
        if candidate.is_file():
            study = "m1" if "m1" in candidate.parts or candidate.name.startswith("m1") else "m2"
            with candidate.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    row["study"] = row.get("study") or study
                    found.append(row)
    return found


def _request_counts(record):
    """Attribute ``model_requests`` to models using the transport mix.

    Transport rows can be fewer than ``model_requests``. The mix is scaled up
    to the unit's request counter so the CNY-per-request rate is not inflated.
    A unit that names only one charged model and has no transport mix gives
    that model the whole counter.
    """
    counts = {}
    transport = record.get("transport") if isinstance(record.get("transport"), dict) else {}
    for model, stats in transport.items():
        if model in MODEL_POOL and isinstance(stats, dict) and type(stats.get("n")) is int:
            counts[model] = counts.get(model, 0) + stats["n"]
    target = record.get("model_requests")
    total = sum(counts.values())
    if counts and type(target) is int and target > total > 0:
        scale = target / total
        counts = {model: count * scale for model, count in counts.items()}
    if not counts and type(target) is int:
        money = record.get("charged_microcny") if isinstance(record.get("charged_microcny"), dict) else {}
        models = [model for model in MODEL_POOL if isinstance(money.get(model), int) and money[model] > 0]
        if len(models) == 1:
            counts[models[0]] = target
        else:
            counts["unknown"] = target
    return counts


def observed_rates(records):
    """CNY per recorded request, by model. A model with no requests has no rate."""
    charged = {model: 0 for model in MODEL_POOL}
    requests = {model: 0 for model in MODEL_POOL}
    for record in records:
        money = record.get("charged_microcny") if isinstance(record.get("charged_microcny"), dict) else {}
        for model, amount in money.items():
            if model in charged and isinstance(amount, int):
                charged[model] += amount
        for model, count in _request_counts(record).items():
            if model in requests:
                requests[model] += count
    rates = {}
    for model in MODEL_POOL:
        rates[model] = (charged[model] / 1_000_000 / requests[model]) if requests[model] else None
    return {"charged_cny": {model: _micro_to_cny(charged[model]) for model in MODEL_POOL},
            "requests": requests, "cny_per_request": rates}


def _arm_units(records):
    grouped = {}
    for record in records:
        if record.get("status") in {"blocked", "skipped"}:
            continue
        grouped.setdefault(record.get("arm") or "unknown", []).append(record)
    return grouped


def project_followup(records, *, cap_cny=PER_MODEL_CAP_CNY):
    """Estimate a fresh round: the r2 schedule again, plus the three new arms.

    ``g1-deepseek-flash`` repeats every non-blocked g1-main unit at the
    deepseek-flash request price and that unit's request count.
    ``g2-fixed`` puts every g2 request on deepseek-flash.
    ``g2-random`` splits every g2 request evenly across the four models.
    R5 savings are not assumed: the ledger does not split requests by node.
    """
    rates = observed_rates(records)
    price = rates["cny_per_request"]
    deepseek = G1_DEEPSEEK_FLASH_MODEL
    if price.get(deepseek) is None:
        raise ValueError("deepseek_flash_rate_missing")
    spend = {model: 0.0 for model in MODEL_POOL}
    parts = {
        "r2_repeat": {model: 0.0 for model in MODEL_POOL},
        "g1-deepseek-flash": {model: 0.0 for model in MODEL_POOL},
        "g2-fixed": {model: 0.0 for model in MODEL_POOL},
        "g2-random": {model: 0.0 for model in MODEL_POOL},
    }
    units = {"r2_repeat": 0, "g1-deepseek-flash": 0, "g2-fixed": 0, "g2-random": 0}
    studies = {"m1": set(), "m2": set()}
    for record in records:
        study = record.get("study")
        if study in studies and isinstance(record.get("unit_id"), str):
            studies[study].add(record["unit_id"])
        if record.get("status") in {"blocked", "skipped"}:
            continue
        counts = _request_counts(record)
        arm = record.get("arm")
        units["r2_repeat"] += 1
        money = record.get("charged_microcny") if isinstance(record.get("charged_microcny"), dict) else {}
        for model, amount in money.items():
            if model in spend and isinstance(amount, int):
                cny = amount / 1_000_000
                spend[model] += cny
                parts["r2_repeat"][model] += cny
        if arm == "g1-main":
            units["g1-deepseek-flash"] += 1
            calls = sum(counts.values()) or int(record.get("model_requests") or 0)
            amount = calls * price[deepseek]
            spend[deepseek] += amount
            parts["g1-deepseek-flash"][deepseek] += amount
        elif arm == "g2":
            units["g2-fixed"] += 1
            units["g2-random"] += 1
            calls = sum(counts.values()) or int(record.get("model_requests") or 0)
            amount = calls * price[deepseek]
            spend[deepseek] += amount
            parts["g2-fixed"][deepseek] += amount
            each = calls / len(MODEL_POOL)
            for model in MODEL_POOL:
                if price.get(model) is not None:
                    share = each * price[model]
                    spend[model] += share
                    parts["g2-random"][model] += share
    over = {model: round(amount - cap_cny, 6) for model, amount in spend.items() if amount > cap_cny}
    return {
        "schema": "alo-followup-cost-v1",
        "r2_units": {"m1": len(studies["m1"]), "m2": len(studies["m2"]), "rows": len(records)},
        "r2_scale": {"m1": R2_M1_UNITS, "m2": R2_M2_UNITS},
        "scale_matches_r2": len(studies["m1"]) == R2_M1_UNITS and len(studies["m2"]) == R2_M2_UNITS,
        "observed": rates,
        "projected_units": units,
        "components_cny": {
            name: {model: round(amount, 6) for model, amount in rows.items()}
            for name, rows in parts.items()
        },
        "projected_cny": {model: round(amount, 6) for model, amount in spend.items()},
        "projected_total_cny": round(sum(spend.values()), 6),
        "cap_cny_per_model": cap_cny,
        "over_cap_cny": over,
        "within_cap": not over,
        "assumption": (
            "Fresh 50 CNY per model. The repeat is the sum of per-unit charged_microcny. "
            "g1-deepseek-flash prices each g1-main unit's request count at the observed "
            "deepseek-flash CNY per request. g2-fixed prices every g2 request as deepseek-flash. "
            "g2-random splits those requests evenly. Planner and s2 calls are still included. "
            "Blocked and skipped units add no projected requests. "
            "A ledger total that also keeps timeout reservations can exceed this sum; "
            "that remainder is not split by unit in records.jsonl."
        ),
    }


def projection_markdown(report):
    lines = ["# Follow-up cost at r2 scale", ""]
    lines.append(report["assumption"])
    lines.append("")
    observed = report["observed"]["charged_cny"]
    lines.append("## Observed per-unit charges")
    lines.append("")
    lines.append("| Model | CNY | Requests | CNY per request |")
    lines.append("|---|---:|---:|---:|")
    for model in MODEL_POOL:
        rate = report["observed"]["cny_per_request"][model]
        rate_text = "" if rate is None else f"{rate:.6f}"
        lines.append(
            f"| {model} | {observed[model]:.2f} | {report['observed']['requests'][model]} | {rate_text} |")
    lines.append("")
    lines.append("## Components")
    lines.append("")
    lines.append("| Component | " + " | ".join(MODEL_POOL) + " |")
    lines.append("|---|" + "|".join("---:" for _ in MODEL_POOL) + "|")
    for name, rows in report["components_cny"].items():
        cells = " | ".join(f"{rows[model]:.2f}" for model in MODEL_POOL)
        lines.append(f"| {name} | {cells} |")
    lines.append("")
    lines.append("## Projected next round")
    lines.append("")
    lines.append("| Model | Projected CNY | Cap |")
    lines.append("|---|---:|---:|")
    for model, amount in report["projected_cny"].items():
        lines.append(f"| {model} | {amount:.2f} | {report['cap_cny_per_model']:.0f} |")
    lines.append(f"| Total | {report['projected_total_cny']:.2f} | |")
    lines.append("")
    if report["over_cap_cny"]:
        lines.append("Over the per-model cap: " + ", ".join(
            f"{model} by {amount:.2f} CNY" for model, amount in report["over_cap_cny"].items()) + ".")
    else:
        lines.append("Every model stays within the per-model cap under this projection.")
    lines.append("")
    lines.append(
        "This projection is not an authorization to start collection. "
        "It prices the repeat from per-unit charged_microcny. That sum leaves out "
        "timeout reservations and restarted attempts. Use ledger_scope_report for "
        "the settled-ledger rates.")
    lines.append("")
    return "\n".join(lines)


# Settled r2 ledger (6100 requests, 71.64 CNY). Per-unit charged_microcny in
# records.jsonl is 33.01 CNY and omits timeout reservations and restarted attempts.
R2_SETTLED_LEDGER = {
    "ledger_cny": {
        "deepseek-flash": 27.134737,
        "mimo-v2.6-flash": 15.825140,
        "glm-5.3-flash": 2.813787,
        "mimo-v2.6-pro": 25.867684,
    },
    "cny_per_request": {
        "deepseek-flash": 0.0103924691689008,
        "mimo-v2.6-flash": 0.006652013451029845,
        "glm-5.3-flash": 0.02024307194244604,
        "mimo-v2.6-pro": 0.0289671713325868,
    },
    "g2_deepseek_cny_per_request": 0.013094141810918754,
    "g1_main_requests": 1525,
    "g2_requests": 1932,
    "g2_s1_share": 0.7520912547528517,
    "timeout_reservation_cny": 24.93,
    "restart_attempt_cny": 13.70,
    "records_charged_cny": 33.01,
}


def _blank_models():
    return {model: 0.0 for model in MODEL_POOL}


def ledger_arm_costs(ledger=None, *, light=False, deepseek_rate="overall",
                     include_repeat=True, include_g1_deepseek=True,
                     include_g2_fixed=True, include_g2_random=True, g2_fraction=1.0):
    """Price a follow-up from settled per-request rates, including timeouts.

    ``light`` keeps the s1 share of G2 requests (fixed topology, no s2 model
    calls). ``deepseek_rate`` ``g2`` uses the G2-context deepseek rate for G2
    arms. g1-deepseek-flash always uses the overall deepseek rate, because
    those requests repeat G1-main. ``g2_fraction`` scales both G2 control arms.
    """
    source = R2_SETTLED_LEDGER if ledger is None else ledger
    rates = source["cny_per_request"]
    deepseek_price = (source["g2_deepseek_cny_per_request"] if deepseek_rate == "g2"
                      else rates["deepseek-flash"])
    share = source["g2_s1_share"] if light else 1.0
    if not 0 <= g2_fraction <= 1:
        raise ValueError("g2_fraction")
    g2_calls = source["g2_requests"] * share * g2_fraction
    parts = {
        "r2_repeat": _blank_models(),
        "g1-deepseek-flash": _blank_models(),
        "g2-fixed": _blank_models(),
        "g2-random": _blank_models(),
    }
    if include_repeat:
        parts["r2_repeat"] = {model: float(source["ledger_cny"][model]) for model in MODEL_POOL}
    if include_g1_deepseek:
        parts["g1-deepseek-flash"]["deepseek-flash"] = source["g1_main_requests"] * rates["deepseek-flash"]
    if include_g2_fixed:
        parts["g2-fixed"]["deepseek-flash"] = g2_calls * deepseek_price
    if include_g2_random:
        each = g2_calls / len(MODEL_POOL)
        for model in MODEL_POOL:
            price = deepseek_price if model == "deepseek-flash" else rates[model]
            parts["g2-random"][model] = each * price
    totals = {model: sum(parts[name][model] for name in parts) for model in MODEL_POOL}
    return {
        "components_cny": parts,
        "projected_cny": totals,
        "projected_total_cny": sum(totals.values()),
        "over_cap_cny": {model: totals[model] - PER_MODEL_CAP_CNY
                         for model in MODEL_POOL if totals[model] > PER_MODEL_CAP_CNY},
    }


def ledger_full_scenarios(ledger=None):
    """The four full-repeat projections. deepseek-flash exceeds 50 CNY in each."""
    specs = (
        ("default_overall_rates", False, "overall"),
        ("light_overall_rates", True, "overall"),
        ("default_g2ctx_ds_rate", False, "g2"),
        ("light_g2ctx_ds_rate", True, "g2"),
    )
    return {
        name: ledger_arm_costs(ledger, light=light, deepseek_rate=rate)
        for name, light, rate in specs
    }


def _g2_fraction_at_cap(ledger, *, light, deepseek_rate, include_random):
    """Largest G2 fraction whose deepseek spend, with the repeat and g1, stays at 50."""
    base = ledger_arm_costs(
        ledger, light=light, deepseek_rate=deepseek_rate,
        include_g2_fixed=False, include_g2_random=False)
    room = PER_MODEL_CAP_CNY - base["projected_cny"]["deepseek-flash"]
    full = ledger_arm_costs(
        ledger, light=light, deepseek_rate=deepseek_rate, include_repeat=False,
        include_g1_deepseek=False, include_g2_fixed=True, include_g2_random=include_random)
    increment = full["projected_cny"]["deepseek-flash"]
    if increment <= 0:
        return 0.0
    return max(0.0, min(1.0, room / increment))


def ledger_scope_options(ledger=None):
    """Schedules that keep every model, and deepseek-flash in particular, at or under 50 CNY.

    Dropping only g2-random from a full repeat still leaves deepseek-flash over
    the cap. The options below are the ones that fit.
    """
    source = R2_SETTLED_LEDGER if ledger is None else ledger
    fraction = _g2_fraction_at_cap(source, light=True, deepseek_rate="overall", include_random=False)
    options = {
        "new_arms_only_light": ledger_arm_costs(
            source, light=True, include_repeat=False),
        "new_arms_only_default_requests": ledger_arm_costs(
            source, light=False, include_repeat=False),
        "repeat_plus_g1_deepseek_only": ledger_arm_costs(
            source, include_g2_fixed=False, include_g2_random=False),
        "repeat_plus_g1_and_capped_g2_fixed": ledger_arm_costs(
            source, light=True, include_g2_random=False, g2_fraction=fraction),
    }
    options["repeat_plus_g1_and_capped_g2_fixed"]["g2_fraction"] = fraction
    return options
