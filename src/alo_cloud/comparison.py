"""Arm comparison that every evaluation report of this round includes.

Oracle, the best single model, CNY and tokens per resolved task, and the
paired win/loss matrix are always present. Missing measurements stay null.
They are not filled in as zero. This module does not score a benchmark and
does not call a model.
"""
from __future__ import annotations

import math


def _number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def _resolved(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def _condition(row):
    label = row.get("condition")
    if isinstance(label, str) and label.strip():
        return label.strip()
    arm = row.get("arm") or "unknown"
    phase = row.get("m2_condition") or row.get("phase") or ""
    if phase and phase not in {"B0", "reference"}:
        return f"{arm}/{phase}"
    return str(arm)


def _single_model(condition):
    name = condition.split("/", 1)[0]
    return name.startswith("g1-") or name in {"g1-main", "g1-strong", "g1-deepseek-flash"}


def _pair_key(left, right):
    return f"{left}__vs__{right}"


def _exact_mcnemar(only_left, only_right):
    """Two-sided exact test on discordant pairs. Descriptive, not a gate."""
    discordant = only_left + only_right
    if discordant <= 0:
        return None
    tail = min(only_left, only_right)
    total = 0
    for wins in range(discordant + 1):
        losses = discordant - wins
        if min(wins, losses) <= tail:
            total += math.comb(discordant, wins)
    probability = total / (2 ** discordant)
    return min(1.0, probability)


def _empty_study():
    return {
        "oracle": {"tasks": 0, "resolved": 0, "rate": None,
                   "single_model_tasks": 0, "single_model_resolved": 0, "single_model_rate": None},
        "best_single_model": None,
        "per_resolved": {},
        "paired": {},
        "conditions": {},
    }


def comparison_report(rows):
    """Build the required tables. ``rows`` are per-unit dicts.

    Each row may carry ``condition`` or ``arm``, ``task_id``, ``resolved``,
    ``cost_cny``, ``input_tokens`` and ``output_tokens``. A row with no
    resolved value is counted in cost but not in the oracle or the matrix.
    """
    grouped = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        study = str(row.get("study") or "all")
        grouped.setdefault(study, []).append(row)
    if not grouped:
        grouped = {"all": []}
    return {study: _one_study(items) for study, items in sorted(grouped.items())}


def _one_study(rows):
    by_condition = {}
    for row in rows:
        condition = _condition(row)
        task = row.get("task_id") or row.get("task")
        if not isinstance(task, str) or not task:
            continue
        resolved = _resolved(row.get("resolved"))
        cost = row.get("cost_cny")
        tokens = None
        incoming = row.get("input_tokens")
        outgoing = row.get("output_tokens")
        if _number(incoming) and _number(outgoing):
            tokens = float(incoming) + float(outgoing)
        elif _number(row.get("tokens")):
            tokens = float(row["tokens"])
        order = row.get("order") or row.get("order_id") or ""
        pair_id = f"{task}|{order}" if isinstance(order, str) and order else task
        bucket = by_condition.setdefault(condition, {})
        current = bucket.get(pair_id)
        if current is None:
            bucket[pair_id] = {"task": task, "resolved": resolved, "cost_cny": 0.0, "tokens": 0.0,
                               "cost_known": False, "tokens_known": False}
            current = bucket[pair_id]
        if resolved is True:
            current["resolved"] = True
        elif resolved is False and current["resolved"] is None:
            current["resolved"] = False
        if _number(cost):
            current["cost_cny"] += float(cost)
            current["cost_known"] = True
        if tokens is not None:
            current["tokens"] += tokens
            current["tokens_known"] = True
    report = _empty_study()
    tasks = {}
    single_tasks = {}
    for condition, tasks_for in by_condition.items():
        resolved_n = cost = tokens = 0.0
        scored = cost_units = token_units = 0
        for task, item in tasks_for.items():
            if item["cost_known"]:
                cost += item["cost_cny"]
                cost_units += 1
            if item["tokens_known"]:
                tokens += item["tokens"]
                token_units += 1
            if item["resolved"] is None:
                continue
            scored += 1
            task = item["task"]
            tasks.setdefault(task, set())
            if item["resolved"]:
                resolved_n += 1
                tasks[task].add(condition)
            else:
                tasks[task].add(None)
            if condition and _single_model(condition):
                single_tasks.setdefault(task, set())
                if item["resolved"]:
                    single_tasks[task].add(condition)
        report["conditions"][condition] = {
            "scored": scored,
            "resolved": int(resolved_n),
            "rate": (resolved_n / scored) if scored else None,
            "cost_cny": cost if cost_units else None,
            "tokens": tokens if token_units else None,
        }
        per = report["per_resolved"][condition] = {
            "resolved": int(resolved_n),
            "cny_per_resolved": None,
            "tokens_per_resolved": None,
        }
        # Total spend of the condition, including units that did not resolve.
        if resolved_n and cost_units:
            per["cny_per_resolved"] = cost / resolved_n
        if resolved_n and token_units:
            per["tokens_per_resolved"] = tokens / resolved_n
    scored_tasks = [task for task, marks in tasks.items() if marks]
    oracle_hits = sum(1 for task in scored_tasks if any(mark for mark in tasks[task]))
    report["oracle"]["tasks"] = len(scored_tasks)
    report["oracle"]["resolved"] = oracle_hits
    report["oracle"]["rate"] = (oracle_hits / len(scored_tasks)) if scored_tasks else None
    single_scored = list(single_tasks)
    single_hits = sum(1 for task in single_scored if single_tasks[task])
    report["oracle"]["single_model_tasks"] = len(single_scored)
    report["oracle"]["single_model_resolved"] = single_hits
    report["oracle"]["single_model_rate"] = (single_hits / len(single_scored)) if single_scored else None
    singles = [item for item in report["conditions"].items() if _single_model(item[0])]
    if singles:
        best = max(singles, key=lambda item: (item[1]["resolved"], item[1]["scored"], item[0]))
        report["best_single_model"] = {
            "condition": best[0],
            "resolved": best[1]["resolved"],
            "scored": best[1]["scored"],
            "rate": best[1]["rate"],
            "cny_per_resolved": report["per_resolved"][best[0]]["cny_per_resolved"],
            "tokens_per_resolved": report["per_resolved"][best[0]]["tokens_per_resolved"],
        }
    names = sorted(by_condition)
    for left in names:
        for right in names:
            if left >= right:
                continue
            both = only_left = only_right = neither = 0
            shared = set(by_condition[left]) & set(by_condition[right])
            for pair_id in shared:
                a = by_condition[left][pair_id]["resolved"]
                b = by_condition[right][pair_id]["resolved"]
                if a is None or b is None:
                    continue
                if a and b:
                    both += 1
                elif a:
                    only_left += 1
                elif b:
                    only_right += 1
                else:
                    neither += 1
            report["paired"][_pair_key(left, right)] = {
                "left": left, "right": right,
                "both_resolved": both, "only_left": only_left, "only_right": only_right,
                "neither": neither, "scored_both": both + only_left + only_right + neither,
                "mcnemar_exact_two_sided": _exact_mcnemar(only_left, only_right),
            }
    return report


def comparison_markdown(report):
    lines = ["# Arm comparison", ""]
    lines.append(
        "Oracle is the number of tasks resolved by at least one condition. "
        "The single-model oracle uses only `g1-*` arms. "
        "CNY and tokens per resolved unit divide the condition's total spend by resolved units. "
        "The paired matrix matches task and order. "
        "The exact McNemar value is descriptive.")
    lines.append("")
    for study, body in report.items():
        lines.append(f"## {study}")
        lines.append("")
        oracle = body["oracle"]
        lines.append(
            f"Oracle: {oracle['resolved']}/{oracle['tasks']}. "
            f"Single-model oracle: {oracle['single_model_resolved']}/{oracle['single_model_tasks']}.")
        best = body["best_single_model"]
        if best is None:
            lines.append("Best single model: none scored.")
        else:
            lines.append(
                f"Best single model: {best['condition']} "
                f"({best['resolved']}/{best['scored']}).")
        lines.append("")
        lines.append("| Condition | Resolved | CNY per resolved | Tokens per resolved |")
        lines.append("|---|---:|---:|---:|")
        for name, item in body["per_resolved"].items():
            cny = "" if item["cny_per_resolved"] is None else f"{item['cny_per_resolved']:.6f}"
            tokens = "" if item["tokens_per_resolved"] is None else f"{item['tokens_per_resolved']:.1f}"
            lines.append(f"| {name} | {item['resolved']} | {cny} | {tokens} |")
        lines.append("")
        lines.append("| Left | Right | Both | Only left | Only right | Neither |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for item in body["paired"].values():
            lines.append(
                f"| {item['left']} | {item['right']} | {item['both_resolved']} | "
                f"{item['only_left']} | {item['only_right']} | {item['neither']} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
