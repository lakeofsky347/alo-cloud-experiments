"""Replay the repro-v1 signal over an existing round-2 run.

Records alone do not contain command return codes. Those numbers are a proxy
and do not update B or TMS. Transcripts in ``checkpoints/<unit>/adapter.json``
carry the return codes the agent already observed. ``--execute`` reruns the
reproduction script and touched public tests inside the instance image. Hidden
SWE-bench tests are not the signal.
"""
from __future__ import annotations

import json
from pathlib import Path

from alo.org.organization import Organization

from .comparison import comparison_markdown, comparison_report
from .followup_cost import load_eval_dir, load_records
from .repro import (
    CELL, REPRO_SCRIPT, S1_TOPICS, S2_TOPICS, apply_definite_outcome, repro_v1_evidence,
)


EXECUTE_INSTRUCTIONS = """\
Optional docker replay, on the machine that has the run directory and the
SWE-bench instance images. This does not call a model API.

    python3 scripts/replay_signal.py \\
        --run-dir /workspace/alo-r2b-run \\
        --eval-dir /workspace/alo-r2-eval \\
        --execute \\
        --output-dir /tmp/alo-repro-replay

--execute applies each G2 model.patch inside that unit's sandbox image and
runs alo_repro.py before and after the patch, plus test modules the patch
touches. A missing script is unknown, not a failure. The default, without
--execute, still reads adapter.json transcripts when they are in --run-dir
and always writes the records-only proxy.
"""


def _loads(text):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _argv_from_assistant(text):
    value = _loads(text)
    if not isinstance(value, dict):
        return None
    arguments = value.get("arguments") if isinstance(value.get("arguments"), dict) else {}
    argv = arguments.get("argv")
    if value.get("tool") == "command" and isinstance(argv, list) and all(isinstance(item, str) for item in argv):
        return argv
    if value.get("tool") == "execute_code" and isinstance(arguments.get("code"), str):
        return ["python3", "-c", arguments["code"]]
    return None


def _returncode_from_user(text):
    value = _loads(text)
    if not isinstance(value, dict) or value.get("tool") not in {"command", "execute_code"}:
        return None
    result = value.get("result")
    if isinstance(result, dict) and type(result.get("returncode")) is int:
        return result["returncode"]
    return None


def _is_repro(argv):
    return any(item.endswith(REPRO_SCRIPT) or item.endswith("/" + REPRO_SCRIPT) for item in argv)


def _is_test(argv):
    if any(item in {"pytest", "py.test", "unittest"} or item.endswith("/pytest") for item in argv):
        return True
    return any(item.endswith(".py") and ("test_" in Path(item).name or item.endswith("_test.py")
                                         or "/tests/" in item or item.startswith("tests/"))
               for item in argv)


def commands_from_messages(messages):
    """Pair each command action with the return code the tool loop stored."""
    found = []
    pending = None
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            argv = _argv_from_assistant(content)
            pending = argv
        elif role == "user" and pending is not None:
            code = _returncode_from_user(content)
            if code is not None:
                found.append({"argv": pending, "returncode": code})
            pending = None
    return found


def signal_from_commands(commands):
    """Apply repro-v1 to commands the agent already ran.

    The first reproduction run is the before code and the last is the after
    code. One run is the after code only. The last test-like command supplies
    the test return code. This is not a fresh before/after harness.
    """
    repro = [item for item in commands if _is_repro(item["argv"])]
    tests = [item for item in commands if _is_test(item["argv"]) and not _is_repro(item["argv"])]
    before = after = test_code = None
    if len(repro) >= 2:
        before = repro[0]["returncode"]
        after = repro[-1]["returncode"]
    elif len(repro) == 1:
        after = repro[0]["returncode"]
    if tests:
        test_code = tests[-1]["returncode"]
    evidence = repro_v1_evidence(before=before, after=after, tests=test_code)
    evidence["source"] = "agent_transcript_returncodes"
    evidence["proxy"] = False
    evidence["controlled_before_after"] = len(repro) >= 2
    evidence["repro_commands"] = len(repro)
    evidence["test_commands"] = len(tests)
    return evidence


def signals_from_adapter(payload):
    transcripts = payload.get("worker_transcripts") if isinstance(payload, dict) else None
    if not isinstance(transcripts, dict):
        return []
    rows = []
    for role, messages in transcripts.items():
        commands = commands_from_messages(messages)
        if not commands:
            continue
        evidence = signal_from_commands(commands)
        evidence["role"] = role
        rows.append(evidence)
    return rows


def records_proxy(record):
    """What the ledger can say without return codes. Not a pass or a fail."""
    edits = record.get("edit_log") if isinstance(record.get("edit_log"), list) else []
    by_node = {}
    for row in edits:
        if not isinstance(row, dict) or row.get("tool") != "command":
            continue
        node = str(row.get("node") or "")
        by_node[node] = by_node.get(node, 0) + 1
    signals = []
    for row in record.get("decision_records") or []:
        if isinstance(row, dict) and row.get("public_signal") is not None:
            signals.append(row.get("public_signal"))
    return {
        "proxy": True,
        "source": "records_edit_log",
        "signal": None,
        "note": (
            "records.jsonl stores command tool names and not return codes. "
            "This proxy is not a repro-v1 pass or fail and does not update B or TMS."
        ),
        "command_calls_by_node": by_node,
        "command_calls": sum(by_node.values()),
        "s2_command_calls": by_node.get("s2", 0),
        "recorded_public_signals": signals,
    }


def _assigned_models(record):
    found = []
    for row in record.get("decision_records") or []:
        if isinstance(row, dict) and row.get("point") == "assignment" and isinstance(row.get("selected"), str):
            found.append(row["selected"])
    return found


def _topics_for(role):
    if role in {"s2", "verify", "verifier"} or (isinstance(role, str) and "verif" in role.lower()):
        return S2_TOPICS
    return S1_TOPICS


def precision_recall(pairs):
    """Positive class is SWE-bench ``resolved``. A pass predicts resolved.

    Unknown signals are excluded. They are counted separately.
    """
    labeled = [(signal, resolved) for signal, resolved in pairs if signal in {"pass", "fail"} and isinstance(resolved, bool)]
    unknown = sum(1 for signal, resolved in pairs if signal not in {"pass", "fail"} and isinstance(resolved, bool))
    true_positive = sum(signal == "pass" and resolved for signal, resolved in labeled)
    false_positive = sum(signal == "pass" and not resolved for signal, resolved in labeled)
    false_negative = sum(signal == "fail" and resolved for signal, resolved in labeled)
    true_negative = sum(signal == "fail" and not resolved for signal, resolved in labeled)
    predicted = true_positive + false_positive
    actual = true_positive + false_negative
    return {
        "labeled": len(labeled),
        "unknown_excluded": unknown,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": (true_positive / predicted) if predicted else None,
        "recall": (true_positive / actual) if actual else None,
    }


def _bool_resolved(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def _eval_index(rows):
    index = {}
    for row in rows:
        unit = row.get("unit_id")
        if isinstance(unit, str) and unit:
            index[unit] = row
    return index


def _comparison_rows(records, eval_index):
    rows = []
    for record in records:
        unit = record.get("unit_id")
        scored = eval_index.get(unit, {})
        resolved = _bool_resolved(scored.get("resolved"))
        cost = scored.get("cost_cny")
        try:
            cost_value = float(cost) if cost not in (None, "") else record.get("estimated_cost_cny")
        except (TypeError, ValueError):
            cost_value = record.get("estimated_cost_cny")
        condition = scored.get("condition") or record.get("arm")
        if record.get("study") == "m2" and record.get("m2_condition"):
            condition = scored.get("condition") or f"{record.get('arm')}/{record.get('m2_condition')}"
        rows.append({
            "study": record.get("study") or scored.get("study") or "all",
            "arm": record.get("arm"),
            "condition": condition,
            "task_id": record.get("task_id") or scored.get("task"),
            "order": scored.get("order") or record.get("order_id") or "",
            "unit_id": unit,
            "resolved": resolved,
            "cost_cny": cost_value,
            "input_tokens": record.get("observed_input_tokens"),
            "output_tokens": record.get("observed_output_tokens"),
        })
    return rows


def _load_adapter(run_dir, unit_id):
    path = Path(run_dir) / "checkpoints" / unit_id / "adapter.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _patch_and_image(run_dir, unit_id):
    checkpoint = Path(run_dir) / "checkpoints" / unit_id
    patch = checkpoint / "model.patch"
    meta = checkpoint / "metadata.json"
    if not patch.is_file() or not meta.is_file():
        return None, None
    try:
        image = json.loads(meta.read_text(encoding="utf-8")).get("sandbox_image")
    except (OSError, ValueError):
        return None, None
    if not isinstance(image, str) or not image:
        return None, None
    return patch.read_text(encoding="utf-8"), image


def execute_public_checks(*, image, patch_text, test_paths=(), runner=None, script_name=REPRO_SCRIPT):
    """Rerun the public script and touched tests. ``runner`` receives the docker argv.

    The default runner is the docker CLI. Tests inject ``runner`` and never
    start a daemon. A missing script stays unknown. The driver reads a mounted
    patch file and does not take hidden tests.
    """
    import subprocess
    import tempfile

    if runner is None:
        def runner(argv, timeout_s):
            try:
                completed = subprocess.run(argv, capture_output=True, timeout=timeout_s, check=False)
            except (OSError, subprocess.TimeoutExpired) as error:
                return {"returncode": None, "stdout": "", "stderr": type(error).__name__}
            return {"returncode": completed.returncode,
                    "stdout": completed.stdout.decode(errors="replace"),
                    "stderr": completed.stderr.decode(errors="replace")}
    with tempfile.TemporaryDirectory(prefix="alo-repro-") as temporary:
        patch_path = Path(temporary) / "model.patch"
        patch_path.write_text(patch_text or "", encoding="utf-8")
        argv = [
            "docker", "run", "--rm", "--network=none",
            "-v", f"{patch_path}:/tmp/alo-model.patch:ro",
            "-w", "/testbed", image, "python3", "-c", _EXECUTE_DRIVER,
            script_name, "/tmp/alo-model.patch", *list(test_paths),
        ]
        result = runner(argv, 600)
    return _parse_execute_result(result)


_EXECUTE_DRIVER = """
import pathlib, subprocess, sys
root = pathlib.Path("/testbed")
script_name, patch_path = sys.argv[1], sys.argv[2]
tests = sys.argv[3:]
script = root / script_name

def run(argv):
    completed = subprocess.run(argv, cwd=root, capture_output=True)
    return completed.returncode

before = after = None
if script.is_file():
    before = run(["python3", script_name])
patch = pathlib.Path(patch_path).read_bytes()
applied = 1
if patch.strip():
    applied = 1 if subprocess.run(
        ["git", "-C", str(root), "apply", "--whitespace=nowarn", patch_path],
        capture_output=True).returncode == 0 else 0
if script.is_file():
    after = run(["python3", script_name])
test_code = ""
existing = [path for path in tests if (root / path).is_file()]
if existing:
    test_code = str(run(["python3", "-m", "pytest", *existing]))

def field(value):
    return "" if value is None else str(value)

sys.stdout.write(field(before) + " " + field(after) + " " + test_code + " " + str(applied) + "\\n")
"""


def _parse_execute_result(result):
    if not isinstance(result, dict) or type(result.get("returncode")) is not int or result["returncode"] != 0:
        evidence = repro_v1_evidence(before=None, after=None, tests=None)
        evidence.update(source="execute", proxy=False, controlled_before_after=False,
                        execute_error="docker_replay_failed")
        return evidence
    parts = (result.get("stdout") or "").strip().split()
    if len(parts) != 4:
        evidence = repro_v1_evidence(before=None, after=None, tests=None)
        evidence.update(source="execute", proxy=False, controlled_before_after=False,
                        execute_error="unparsed_execute_output")
        return evidence

    def code(text):
        if text == "":
            return None
        if text.lstrip("-").isdigit():
            return int(text)
        return None

    before, after, tests, applied = code(parts[0]), code(parts[1]), code(parts[2]), parts[3] == "1"
    evidence = repro_v1_evidence(before=before, after=after, tests=tests)
    evidence.update(source="execute", proxy=False, controlled_before_after=applied and before is not None,
                    patch_applied=applied)
    return evidence


def replay_run(run_dir, eval_dir, *, execute=False, executor=None):
    """Replay one run directory. ``executor`` replaces docker in tests."""
    records = load_records(Path(run_dir) / "records.jsonl")
    scored = load_eval_dir(eval_dir) if eval_dir else []
    index = _eval_index(scored)
    comparison = comparison_report(_comparison_rows(records, index))
    proxy_rows = []
    resolved_s2_budget = 0
    transcript_pairs = []
    execute_pairs = []
    transcript_updates = []
    execute_updates = []
    g2 = [row for row in records if row.get("arm") == "g2"]
    adapters = 0
    for record in g2:
        proxy = records_proxy(record)
        proxy["unit_id"] = record.get("unit_id")
        proxy_rows.append(proxy)
        resolved = _bool_resolved((index.get(record.get("unit_id")) or {}).get("resolved"))
        if (resolved is True and proxy["s2_command_calls"]
                and record.get("status") == "stopped"
                and record.get("reason") == "unit_resource_exhausted"):
            resolved_s2_budget += 1
        payload = _load_adapter(run_dir, record.get("unit_id"))
        if payload is not None:
            adapters += 1
        node_signals = signals_from_adapter(payload) if payload else []
        if node_signals:
            combined = _combine(node_signals)
            transcript_pairs.append((combined["signal"], resolved))
            models = _assigned_models(record)
            model = models[-1] if models else None
            for item in node_signals:
                role_model = model
                transcript_updates.append((role_model, item["signal"], _topics_for(item.get("role"))))
        if execute:
            patch, image = _patch_and_image(run_dir, record.get("unit_id"))
            if patch is None:
                evidence = repro_v1_evidence(before=None, after=None, tests=None)
                evidence.update(source="execute", proxy=False, execute_error="patch_or_image_missing")
            else:
                evidence = execute_public_checks(
                    image=image, patch_text=patch, runner=executor,
                    test_paths=proxy.get("touched_tests") or [])
            execute_pairs.append((evidence["signal"], resolved))
            models = _assigned_models(record)
            execute_updates.append((models[-1] if models else None, evidence["signal"], S1_TOPICS))
    transcript_posterior = _posterior(transcript_updates)
    execute_posterior = _posterior(execute_updates) if execute else None
    definitive = [row for row in transcript_updates if row[1] in {"pass", "fail"}]
    return {
        "schema": "alo-repro-v1-replay",
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir) if eval_dir else None,
        "units": len(records),
        "g2_units": len(g2),
        "adapters_found": adapters,
        "execute": bool(execute),
        "records_proxy": {
            "proxy": True,
            "units": len(proxy_rows),
            "command_calls": sum(row["command_calls"] for row in proxy_rows),
            "units_with_s2_commands": sum(1 for row in proxy_rows if row["s2_command_calls"]),
            "resolved_units_stopped_in_budget_after_s2_commands": resolved_s2_budget,
            "note": proxy_rows[0]["note"] if proxy_rows else "no g2 units",
            "precision_recall": None,
            "posterior_updated": False,
        },
        "transcript": {
            "source": "agent_transcript_returncodes",
            "proxy": False,
            "note": (
                "Return codes are the commands the agent already ran. "
                "A single reproduction run is not a controlled before/after measurement. "
                "Precision and recall use only definite pass or fail against SWE-bench resolved."
            ) if adapters else "no adapter.json transcripts in this run directory",
            "precision_recall": precision_recall(transcript_pairs) if adapters else None,
            "posterior": transcript_posterior,
            "beta_gt_1": _beta_gt_1(transcript_posterior),
            "definite_updates": len(definitive),
        },
        "execute_replay": None if not execute else {
            "source": "execute",
            "proxy": False,
            "note": "Docker replay of alo_repro.py and touched public tests. Hidden tests are not run.",
            "precision_recall": precision_recall(execute_pairs),
            "posterior": execute_posterior,
            "beta_gt_1": _beta_gt_1(execute_posterior),
        },
        "comparison": comparison,
        "instructions": EXECUTE_INSTRUCTIONS,
    }


def _combine(signals):
    if any(item["signal"] == "fail" for item in signals):
        return {"signal": "fail"}
    if any(item["signal"] == "pass" for item in signals):
        return {"signal": "pass"}
    return {"signal": "unknown"}


def _posterior(updates):
    if not updates:
        return {"belief": [], "tms": [], "writes": 0}
    org = Organization(":memory:")
    writes = 0
    try:
        for model, status, topics in updates:
            if apply_definite_outcome(org.belief, org.tms, model, status, topics=topics):
                writes += 1
        belief = org.belief.dump()
        tms_rows = org.conn.execute(
            "SELECT model_id, topic, successes, failures FROM tms_outcomes ORDER BY model_id, topic"
        ).fetchall()
        tms = [{"model": row["model_id"], "topic": row["topic"],
                "successes": row["successes"], "failures": row["failures"]} for row in tms_rows]
    finally:
        org.conn.close()
    return {"belief": belief, "tms": tms, "writes": writes, "cell": CELL}


def _beta_gt_1(posterior):
    if not posterior:
        return False
    return any(row.get("beta", 1) > 1 for row in posterior.get("belief") or []) or any(
        row.get("failures", 0) > 0 for row in posterior.get("tms") or [])


def replay_markdown(report):
    lines = ["# repro-v1 replay", ""]
    proxy = report["records_proxy"]
    lines.append("## Records-only proxy")
    lines.append("")
    lines.append(proxy["note"])
    lines.append("")
    lines.append(
        f"G2 units: {report['g2_units']}. Command tool rows: {proxy['command_calls']}. "
        f"Units with an s2 command: {proxy['units_with_s2_commands']}. "
        f"Resolved units that still stopped on the unit budget after an s2 command: "
        f"{proxy.get('resolved_units_stopped_in_budget_after_s2_commands', 0)}. "
        "Precision, recall, and B/TMS are not computed from this proxy.")
    lines.append("")
    transcript = report["transcript"]
    lines.append("## Transcript return codes")
    lines.append("")
    lines.append(transcript["note"])
    lines.append("")
    _write_metrics(lines, transcript)
    if report.get("execute_replay"):
        lines.append("## Docker execute")
        lines.append("")
        lines.append(report["execute_replay"]["note"])
        lines.append("")
        _write_metrics(lines, report["execute_replay"])
    lines.append("## Comparison")
    lines.append("")
    lines.append(comparison_markdown(report["comparison"]))
    lines.append("## Execute command")
    lines.append("")
    lines.append("```")
    lines.append(report["instructions"].rstrip())
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _write_metrics(lines, section):
    metrics = section.get("precision_recall") or {}
    if not metrics:
        lines.append("No definite signal.")
        lines.append("")
        return
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    lines.append(
        f"Labeled: {metrics.get('labeled')}. Unknown excluded: {metrics.get('unknown_excluded')}. "
        f"Precision: {precision}. Recall: {recall}.")
    posterior = section.get("posterior") or {}
    lines.append(f"B/TMS writes: {posterior.get('writes', 0)}. beta>1: {section.get('beta_gt_1')}.")
    for row in posterior.get("belief") or []:
        lines.append(
            f"- belief {row.get('model')} alpha={row.get('alpha')} beta={row.get('beta')}")
    lines.append("")
