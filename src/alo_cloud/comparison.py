"""Arm comparison that every evaluation report of this round includes.

Oracle, the best single model, CNY and tokens per resolved task, and the
paired win/loss matrix are always present. Missing measurements stay null.
They are not filled in as zero. This module does not score a benchmark and
does not call a model.
"""
from __future__ import annotations

import math
from statistics import fmean, stdev


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
    return name.startswith("g1-") or name in {
        "g1-main", "g1-strong", "g1-deepseek-flash", "A4", "A5",
    }


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
        resolved_field = row.get("resolved")
        if resolved_field is None and "harness_resolved" in row:
            resolved_field = row.get("harness_resolved")
        resolved = _resolved(resolved_field)
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


PB_PAIRS = (("A1", "A2"), ("A1", "A3"), ("A1", "A4"), ("A5", "A4"))


def _mean_sd(values):
    clean = [float(value) for value in values if _number(value)]
    if not clean:
        return {"n": 0, "mean": None, "sd": None}
    if len(clean) == 1:
        return {"n": 1, "mean": clean[0], "sd": None}
    return {"n": len(clean), "mean": fmean(clean), "sd": stdev(clean)}


def _pb_score(row):
    rate = row.get("pass_rate")
    if _number(rate) and rate <= 1:
        return float(rate)
    resolved = _resolved(row.get("resolved"))
    if resolved is True:
        return 1.0
    if resolved is False:
        return 0.0
    return None


def _pb_first_assignment(row):
    named = row.get("first_assignment")
    if isinstance(named, str) and named:
        return named
    nodes = row.get("node_assignments")
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict) and node.get("node_id") in {"p1", "g1"}:
                model = node.get("model_id")
                if isinstance(model, str) and model:
                    return model
        if nodes and isinstance(nodes[0], dict):
            model = nodes[0].get("model_id")
            if isinstance(model, str) and model:
                return model
    return None


def _pb_cost(row):
    for key in ("cost", "cost_cny"):
        value = row.get(key)
        if _number(value):
            return float(value)
    usage = row.get("usage")
    if isinstance(usage, dict) and _number(usage.get("cost")):
        return float(usage["cost"])
    return None


def pb_paired_report(rows):
    """ProgramBench arm table plugged into the shared comparison report.

    Pairs are A1-A2, A1-A3, A1-A4, and A5-A4. Decision-difference is the
    fraction of tasks whose first A1 assignment differs from A2. Missing
    scores stay null.
    """
    prepared = []
    by_arm = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        arm = row.get("arm") or row.get("condition")
        task = row.get("task_id") or row.get("task")
        if not isinstance(arm, str) or not arm or not isinstance(task, str) or not task:
            continue
        item = dict(row)
        item["condition"] = arm
        item["arm"] = arm
        item["task_id"] = task
        item["study"] = "ProgramBench"
        # The runner stores the harness outcome as harness_resolved. The shared
        # comparison tables read resolved.
        if item.get("resolved") is None and "harness_resolved" in row:
            item["resolved"] = row.get("harness_resolved")
        if "cost_cny" not in item and _pb_cost(row) is not None:
            item["cost_cny"] = _pb_cost(row)
        prepared.append(item)
        arm_tasks = by_arm.setdefault(arm, {})
        if task in arm_tasks:
            previous = arm_tasks[task]
            raise ValueError(
                f"duplicate_pb_arm_task: {arm}/{task} in "
                f"{previous.get('run_dir')}/{previous.get('attempt_id')} and "
                f"{item.get('run_dir')}/{item.get('attempt_id')}. "
                "Report each lane, protocol and selected attempt separately.")
        arm_tasks[task] = item
    base = comparison_report(prepared)
    per_task = {}
    for arm, tasks in by_arm.items():
        for task, item in tasks.items():
            bucket = per_task.setdefault(task, {})
            resolved_field = item.get("resolved")
            if resolved_field is None and "harness_resolved" in item:
                resolved_field = item.get("harness_resolved")
            bucket[arm] = {
                "resolved": _resolved(resolved_field),
                "pass_rate": item.get("pass_rate") if _number(item.get("pass_rate")) else None,
                "cost": _pb_cost(item),
                "compile_success": item.get("compile_success") if type(item.get("compile_success")) is bool else None,
                "official_compile_success": item.get("official_compile_success"),
                "official_pass_rate": item.get("official_pass_rate"),
                "submission_accepted": item.get("submission_accepted", item.get("submission_ok")),
                "first_assignment": _pb_first_assignment(item),
                "wall_clock_s": item.get("wall_clock_s") if _number(item.get("wall_clock_s")) else None,
            }
    pairs = {}
    paired_body = base.get("ProgramBench", {}).get("paired", {})
    for left, right in PB_PAIRS:
        key = f"{left}-{right}"
        shared = sorted(set(by_arm.get(left, {})) & set(by_arm.get(right, {})))
        rows_for = []
        for task in shared:
            a = per_task[task].get(left, {})
            b = per_task[task].get(right, {})
            rows_for.append({
                "task_id": task,
                "left": a,
                "right": b,
            })
        lookup = paired_body.get(f"{left}__vs__{right}") or paired_body.get(f"{right}__vs__{left}")
        pairs[key] = {
            "left": left,
            "right": right,
            "tasks": rows_for,
            "comparison": lookup,
        }
    both = sorted(set(by_arm.get("A1", {})) & set(by_arm.get("A2", {})))
    compared = 0
    differ = 0
    for task in both:
        left = per_task[task]["A1"].get("first_assignment")
        right = per_task[task]["A2"].get("first_assignment")
        if not isinstance(left, str) or not isinstance(right, str):
            continue
        compared += 1
        if left != right:
            differ += 1
    arms = {}
    for arm, tasks in sorted(by_arm.items()):
        costs = []
        scores = []
        compile_hits = compile_known = 0
        official_pass = official_fail = accepted = rejected = accepted_builds = 0
        legacy_units = 0
        cost_known = 0
        for item in tasks.values():
            cost = _pb_cost(item)
            if cost is not None:
                costs.append(cost)
                cost_known += 1
            score = _pb_score(item)
            if score is not None:
                scores.append(score)
            compile = item.get("compile_success")
            if type(compile) is bool:
                compile_known += 1
                if compile is True:
                    compile_hits += 1
            official = item.get("official_compile_success")
            admissible = item.get("submission_accepted", item.get("submission_ok"))
            official_pass += official is True
            official_fail += official is False
            accepted += admissible is True
            rejected += admissible is False
            accepted_builds += admissible is True and official is True
            legacy_units += "official_compile_success" not in item
        total = len(tasks)
        official_known = official_pass + official_fail
        arms[arm] = {
            "units": len(tasks),
            "cost_per_unit": (sum(costs) / cost_known) if cost_known else None,
            "compile_rate": (compile_hits / compile_known) if compile_known else None,
            "compile_known": compile_known,
            "legacy_units_without_official_evidence": legacy_units,
            "official_compile": {
                "pass": official_pass, "fail": official_fail, "unknown": total - official_known,
                "known": official_known, "total": total,
                "rate_among_known": official_pass / official_known if official_known else None,
                "coverage": official_known / total,
            },
            "submission_acceptance": {
                "accepted": accepted, "rejected": rejected, "unknown": total - accepted - rejected,
                "accepted_and_compiled": accepted_builds,
            },
            "variance": {
                "cost": _mean_sd(costs),
                "score": _mean_sd(scores),
            },
        }
    return {
        "schema": "fox-pb-paired-v1",
        "comparison": base.get("ProgramBench", _empty_study()),
        "pairs": pairs,
        "per_task": per_task,
        "decision_difference_rate": (differ / compared) if compared else None,
        "decision_difference": {"compared": compared, "differ": differ},
        "arms": arms,
    }
