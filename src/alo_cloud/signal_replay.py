"""Replay the repro-v1 signal over an existing round-2 run.

Records alone do not contain command return codes. Those numbers are a proxy
and do not update B or TMS. Transcripts in ``checkpoints/<unit>/adapter.json``
carry the return codes the agent already observed. ``--execute`` reruns the
reproduction script and touched public tests inside the instance image. Hidden
SWE-bench tests are not the signal.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import uuid

from alo.org.organization import Organization

from .comparison import comparison_markdown, comparison_report
from .followup_cost import load_eval_dir, load_records
from .repro import (
    CELL, REPRO_SCRIPT, S1_TOPICS, S2_TOPICS, _is_repro_argv, _is_test_argv, apply_definite_outcome,
    command_results_from_messages, repro_codes_from_messages, repro_v1_evidence, touched_test_paths,
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
runs alo_repro.py before and after the patch. It also runs public test files
the agent changed. Those files may have been edited or weakened, so a pass is
the agent's version of the test. The same paths are run again at the base
commit, before the patch, and both return codes are reported. A missing script
is unknown, not a failure. Each container is named and labeled and is removed
with docker rm -f when the run ends or times out. --jobs N (default 1) runs
that many containers at once. --output-dir defaults to the current directory
and must not be inside --run-dir.
"""


def commands_from_messages(messages):
    """Pair each command action with the return code the tool loop stored.

    A JSON object followed by stray text still counts. The live tool loop
    reads the same leading object.
    """
    return command_results_from_messages(messages)


def _messages_for_commands(commands):
    messages = []
    for item in commands:
        messages.append({"role": "assistant", "content": json.dumps(
            {"tool": "command", "arguments": {"argv": item["argv"]}})})
        messages.append({"role": "user", "content": json.dumps(
            {"tool": "command", "result": {"returncode": item["returncode"]}})})
    return messages


def signal_from_commands(commands):
    """Apply repro-v1 to commands the agent already ran.

    The first reproduction run is the before code and the last is the after
    code. One run is the after code only. The last test-like command supplies
    the test return code. This is not a fresh before/after harness.
    """
    before, after, test_code = repro_codes_from_messages(_messages_for_commands(commands))
    repro_n = sum(1 for item in commands if _is_repro_argv(item["argv"]))
    test_n = sum(1 for item in commands if _is_test_argv(item["argv"]) and not _is_repro_argv(item["argv"]))
    evidence = repro_v1_evidence(before=before, after=after, tests=test_code)
    evidence["source"] = "agent_transcript_returncodes"
    evidence["proxy"] = False
    evidence["controlled_before_after"] = repro_n >= 2
    evidence["repro_commands"] = repro_n
    evidence["test_commands"] = test_n
    return evidence


def signals_from_adapter(payload):
    transcripts = payload.get("worker_transcripts") if isinstance(payload, dict) else None
    if not isinstance(transcripts, dict):
        return []
    rows = []
    for role, messages in transcripts.items():
        commands = commands_from_messages(messages if isinstance(messages, list) else [])
        if not commands:
            continue
        evidence = signal_from_commands(commands)
        evidence["role"] = role
        evidence["model_id"] = model_for_role(payload, role)
        rows.append(evidence)
    return rows


def model_for_role(payload, role):
    """Model that ran ``role``, from the checkpoint assignments.

    The unit's last assignment is not a substitute. A missing role stays
    uncredited rather than being given another node's model.
    """
    assignments = payload.get("assignments") if isinstance(payload, dict) else None
    if not isinstance(assignments, dict):
        return None
    assignment = assignments.get(role)
    if not isinstance(assignment, dict):
        return None
    model = assignment.get("model_id")
    if isinstance(model, str) and model:
        return model
    model = assignment.get("model")
    return model if isinstance(model, str) and model else None


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
        "touched_tests": touched_test_paths(_edit_paths(edits)),
    }


def _edit_paths(edits):
    paths = []
    for row in edits if isinstance(edits, list) else []:
        if isinstance(row, dict):
            paths.extend(path for path in row.get("paths") or [] if isinstance(path, str))
    return paths


def paths_from_patch(patch_text):
    """Paths named by ``diff --git a/... b/...`` headers."""
    found = []
    if not isinstance(patch_text, str):
        return found
    for line in patch_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[2].startswith("a/") and parts[3].startswith("b/"):
            found.append(parts[3][2:])
    return found


def touched_tests_for_unit(record, patch_text=""):
    paths = _edit_paths(record.get("edit_log") if isinstance(record, dict) else [])
    paths.extend(paths_from_patch(patch_text))
    return touched_test_paths(paths)


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


REPLAY_CONTAINER_LABEL = "alo.replay=repro-v1"


def container_name():
    """Docker name unique to this process and call. Safe for ``docker rm -f``."""
    return f"alo-repro-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def docker_rm(name):
    """Remove one replay container. A missing name is not an error."""
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def _default_docker_runner(argv, timeout_s):
    """Run docker. On timeout the client is killed; the caller still ``docker rm -f``."""
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    return {"returncode": proc.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace")}


def execute_public_checks(*, image, patch_text, test_paths=(), runner=None, cleanup=None,
                          timeout_s=600, script_name=REPRO_SCRIPT, container=None):
    """Rerun the public script and the agent's touched tests.

    ``tests`` in the result is the patched tree. ``tests_base`` is the same
    paths at the base commit, before the patch. The agent may have edited
    those tests, so the two codes are reported separately. Hidden tests are
    not run. Tests inject ``runner`` and never start a daemon. The default
    runner names the container, labels it, and ``cleanup`` removes it on
    exit and on timeout.
    """
    import tempfile

    name = container or container_name()
    if runner is None:
        runner = _default_docker_runner
        if cleanup is None:
            cleanup = docker_rm
    with tempfile.TemporaryDirectory(prefix="alo-repro-") as temporary:
        patch_path = Path(temporary) / "model.patch"
        patch_path.write_text(patch_text or "", encoding="utf-8")
        argv = [
            "docker", "run", "--name", name, "--label", REPLAY_CONTAINER_LABEL,
            "--network=none",
            "-v", f"{patch_path}:/tmp/alo-model.patch:ro",
            "-w", "/testbed", image, "python3", "-c", _EXECUTE_DRIVER,
            script_name, "/tmp/alo-model.patch", *list(test_paths),
        ]
        try:
            result = runner(argv, timeout_s)
        except subprocess.TimeoutExpired:
            result = {"returncode": None, "stdout": "", "stderr": "timeout"}
        finally:
            if cleanup is not None:
                cleanup(name)
    return _parse_execute_result(result)


_EXECUTE_DRIVER = r"""
import json, os, pathlib, subprocess, sys
root = pathlib.Path(os.environ.get("ALO_REPRO_ROOT", "/testbed"))
script_name, patch_path = sys.argv[1], sys.argv[2]
tests = sys.argv[3:]
script = root / script_name
testbed_python = pathlib.Path("/opt/miniconda3/envs/testbed/bin/python")
python_bin = str(testbed_python) if testbed_python.is_file() else "python3"
env = dict(os.environ)
prefix = "/opt/miniconda3/envs/testbed/bin:"
if not env.get("PATH", "").startswith(prefix):
    env["PATH"] = prefix + env.get("PATH", "")

def run(argv):
    completed = subprocess.run(argv, cwd=root, capture_output=True, env=env)
    return completed.returncode

before = after = None
# The candidate script often exists only inside model.patch. Apply once, keep
# those bytes, return to the clean tree, and run that same script on both.
extracted = None
existed = script.is_file()
original = script.read_bytes() if existed else None
patch_bytes = pathlib.Path(patch_path).read_bytes()
if patch_bytes.strip():
    probed = subprocess.run(
        ["git", "-C", str(root), "apply", "--whitespace=nowarn", patch_path],
        capture_output=True).returncode == 0
    if probed and script.is_file():
        extracted = script.read_bytes()
    subprocess.run(["git", "-C", str(root), "reset", "--hard", "HEAD"], capture_output=True)
    if existed and original is not None and not script.is_file():
        script.write_bytes(original)
    elif not existed and script.is_file():
        script.unlink()
if extracted is not None:
    script.write_bytes(extracted)
elif script.is_file():
    extracted = script.read_bytes()
if script.is_file():
    before = run([python_bin, script_name])
if extracted is not None and not existed:
    script.unlink()
elif existed and original is not None:
    script.write_bytes(original)

def run_suite(paths):
    if not paths:
        return None, None, None
    pytest_bin = pathlib.Path(python_bin).parent / "pytest"
    if pytest_bin.is_file() or run([python_bin, "-c", "import pytest"]) == 0:
        command = [str(pytest_bin), *paths] if pytest_bin.is_file() else [python_bin, "-m", "pytest", *paths]
        return run(command), "pytest", None
    if (root / "tests" / "runtests.py").is_file():
        labels = []
        for path in paths:
            label = path[6:] if path.startswith("tests/") else path
            if label.endswith(".py"):
                label = label[:-3]
            labels.append(label.replace("/", "."))
        return run([python_bin, "tests/runtests.py", "--verbosity=1", *labels]), "django_runtests", None
    if (root / "bin" / "test").is_file():
        return run([python_bin, "bin/test", *paths]), "sympy_bin_test", None
    return None, None, "no_public_test_runner"

base_paths = [path for path in tests if (root / path).is_file()]
tests_base, runner, reason = run_suite(base_paths)
patch = pathlib.Path(patch_path).read_bytes()
applied = True
if patch.strip():
    applied = subprocess.run(
        ["git", "-C", str(root), "apply", "--whitespace=nowarn", patch_path],
        capture_output=True).returncode == 0
if script.is_file():
    after = run([python_bin, script_name])
patched_paths = [path for path in tests if (root / path).is_file()]
test_code, patched_runner, patched_reason = run_suite(patched_paths)
if patched_runner:
    runner = patched_runner
if reason is None:
    reason = patched_reason
payload = {
    "before": before, "after": after, "tests": test_code, "tests_base": tests_base,
    "applied": applied, "runner": runner, "reason": reason,
    "note": "tests is the agent-changed files after the patch; tests_base is those paths at the base commit",
}
sys.stdout.write(json.dumps(payload) + "\n")
"""


def _parse_execute_payload(stdout):
    """Last JSON object on stdout. Null fields stay missing instead of collapsing."""
    if not isinstance(stdout, str):
        return None
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "applied" in payload:
            return payload
    return None


def _parse_execute_result(result):
    payload = _parse_execute_payload(result.get("stdout") if isinstance(result, dict) else None)
    if payload is None:
        evidence = repro_v1_evidence(before=None, after=None, tests=None)
        failed = (not isinstance(result, dict) or type(result.get("returncode")) is not int
                  or result.get("returncode") != 0)
        evidence.update(source="execute", proxy=False, controlled_before_after=False,
                        execute_error="docker_replay_failed" if failed else "unparsed_execute_output")
        return evidence

    def code(value):
        if value is None or value == "":
            return None
        if type(value) is int:
            return value
        if isinstance(value, str) and value.lstrip("-").isdigit():
            return int(value)
        return None

    before, after, tests = code(payload.get("before")), code(payload.get("after")), code(payload.get("tests"))
    applied = payload.get("applied") is True
    evidence = repro_v1_evidence(before=before, after=after, tests=tests)
    evidence.update(source="execute", proxy=False, controlled_before_after=applied and before is not None,
                    patch_applied=applied, execute_runner=payload.get("runner"),
                    execute_reason=payload.get("reason"),
                    tests_base_returncode=code(payload.get("tests_base")),
                    tests_note=payload.get("note"))
    return evidence


def resolve_output_dir(output, run_dir):
    """Directory for the replay report. It must not be inside ``run_dir``.

    ``output`` None is the current working directory.
    """
    destination = (Path.cwd() if output is None else Path(output)).resolve()
    run = Path(run_dir).resolve()
    if destination == run or destination.is_relative_to(run):
        raise ValueError("replay_output_inside_run_dir")
    return destination


def replay_run(run_dir, eval_dir, *, execute=False, executor=None, jobs=1):
    """Replay one run directory. ``executor`` replaces docker in tests."""
    if type(jobs) is not int or jobs < 1:
        raise ValueError("invalid_replay_jobs")
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
        # Every scored G2 unit is in the denominator. No transcript, an empty
        # transcript, and a transcript with no test command are unknown.
        if isinstance(resolved, bool):
            transcript_pairs.append((_combine(node_signals)["signal"] if node_signals else "unknown", resolved))
        for item in node_signals:
            transcript_updates.append((item.get("model_id"), item["signal"], _topics_for(item.get("role"))))
    execute_units = []
    if execute:
        def _one(record):
            patch, image = _patch_and_image(run_dir, record.get("unit_id"))
            test_paths = touched_tests_for_unit(record, patch or "")
            payload = _load_adapter(run_dir, record.get("unit_id"))
            if patch is None:
                evidence = repro_v1_evidence(before=None, after=None, tests=None)
                evidence.update(source="execute", proxy=False, execute_error="patch_or_image_missing")
            else:
                evidence = execute_public_checks(
                    image=image, patch_text=patch, runner=executor, test_paths=test_paths)
            resolved = _bool_resolved((index.get(record.get("unit_id")) or {}).get("resolved"))
            model = model_for_role(payload, "s1") if payload else None
            return record, evidence, resolved, model

        if jobs == 1:
            executed = [_one(record) for record in g2]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                executed = list(pool.map(_one, g2))
        for record, evidence, resolved, model in executed:
            if isinstance(resolved, bool):
                execute_pairs.append((evidence["signal"], resolved))
            execute_updates.append((model, evidence["signal"], S1_TOPICS))
            execute_units.append({
                "unit_id": record.get("unit_id"),
                "signal": evidence.get("signal"),
                "tests_returncode": evidence.get("tests_returncode"),
                "tests_base_returncode": evidence.get("tests_base_returncode"),
                "execute_reason": evidence.get("execute_reason") or evidence.get("execute_error"),
                "execute_runner": evidence.get("execute_runner"),
            })
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
            "precision_recall": precision_recall(transcript_pairs) if transcript_pairs else None,
            "posterior": transcript_posterior,
            "beta_gt_1": _beta_gt_1(transcript_posterior),
            "definite_updates": len(definitive),
        },
        "execute_replay": None if not execute else {
            "source": "execute",
            "proxy": False,
            "jobs": jobs,
            "note": (
                "Docker replay of alo_repro.py and public test files the agent changed. "
                "tests_returncode is those files after the patch. tests_base_returncode is "
                "the same paths at the base commit. A patched pass can be a weakened test. "
                "Hidden tests are not run."
            ),
            "precision_recall": precision_recall(execute_pairs),
            "posterior": execute_posterior,
            "beta_gt_1": _beta_gt_1(execute_posterior),
            "units": execute_units,
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
