"""Opt-in reproduction signal. The frozen syntax criterion stays the default.

``repro-v1`` is a public check. It does not read a hidden grader. A pass
requires the reproduction script to fail on the unfixed tree and pass after
the fix, and any touched public test that actually ran must pass. A syntax-only
workspace stays unknown, so B and TMS do not move. Definite pass and fail are
the only values those stores accept.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

from .public_check import _test_module, public_quality_signal


REPRO_CRITERION_ID = "repro-v1"
REPRO_SCRIPT = "alo_repro.py"
REPRO_INSTRUCTION = (
    " Write alo_repro.py at the workspace root before editing the fix. "
    "Running it on the unfixed code must exit non-zero. Running it after the fix must exit 0. "
    "Do not weaken the script so it passes on the unfixed code. "
    "Relevant public tests are test modules you touch; do not read hidden tests."
)
CELL = "type=debugging|complexity=na|uncertainty=na"
S1_TOPICS = ("coding", "implementation")
S2_TOPICS = ("testing", "validation")


def repro_v1_status(*, before, after, tests):
    """Map return codes onto pass, fail, or unknown.

    ``before`` is the reproduction script on the unfixed tree. ``after`` is the
    same script on the tree that contains the fix. ``tests`` is one return code
    for the touched public tests. Missing codes are ``None`` and are not zeros.
    """
    failed_tests = type(tests) is int and tests != 0
    failed_after = type(after) is int and after != 0
    if failed_tests or failed_after:
        return "fail"
    reproduced = type(before) is int and before != 0 and after == 0
    tests_ok = type(tests) is int and tests == 0
    if type(before) is int and before == 0 and after == 0:
        return "fail"
    if reproduced and (tests is None or tests_ok):
        return "pass"
    if tests_ok and after is None and before is None:
        return "pass"
    return "unknown"


def repro_v1_reason(*, before, after, tests, status):
    if status == "fail" and type(tests) is int and tests != 0:
        return "public_tests_failed"
    if status == "fail" and type(after) is int and after != 0:
        return "repro_failed_after_fix"
    if status == "fail" and before == 0 and after == 0:
        return "repro_did_not_fail_before_fix"
    if status == "pass" and type(before) is int and before != 0 and after == 0:
        return "repro_failed_before_and_passed_after"
    if status == "pass" and type(tests) is int and tests == 0:
        return "public_tests_passed"
    return "repro_not_run"


def repro_v1_evidence(*, before, after, tests, changed_paths=(), patch_text="", root=None):
    """Evidence for ``local_criterion``. Syntax alone cannot be a pass or a fail."""
    status = repro_v1_status(before=before, after=after, tests=tests)
    reason = repro_v1_reason(before=before, after=after, tests=tests, status=status)
    syntax_aware = "unknown"
    if root is not None and (type(after) is int or type(tests) is int):
        syntax_aware = public_quality_signal(
            root, changed_paths=changed_paths, patch_text=patch_text,
            repro_returncode=after, tests_returncode=tests)
    digest = hashlib.sha256(
        f"{before}|{after}|{tests}|{status}".encode("utf-8")).hexdigest()[:16]
    reliable = status in {"pass", "fail"}
    return {
        "status": status if reliable else "unknown",
        "reliable": reliable,
        "scope": "node_local",
        "criterion_id": REPRO_CRITERION_ID,
        "evidence_id": "repro-" + digest,
        "reason": reason,
        "repro_before_returncode": before if type(before) is int else None,
        "repro_after_returncode": after if type(after) is int else None,
        "tests_returncode": tests if type(tests) is int else None,
        "syntax_aware_signal": syntax_aware,
        "signal": status if reliable else "unknown",
    }


def touched_test_paths(changed_paths):
    found = []
    for path in changed_paths:
        if isinstance(path, str) and path.endswith(".py") and _test_module(path):
            found.append(path)
    return list(dict.fromkeys(found))


def _command_code(services, argv, timeout_s):
    command = getattr(services, "command", None)
    if not callable(command):
        return None
    try:
        result = command(list(argv), timeout_s=timeout_s)
    except Exception:
        return None
    if isinstance(result, dict) and type(result.get("returncode")) is int:
        return result["returncode"]
    return None


def _git(root, args, *, input_bytes=None):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        input=input_bytes, capture_output=True, check=False,
    )


def _restore_patch(root, patch, script_name, script_bytes):
    if patch.strip():
        restored = _git(root, ["apply", "--whitespace=nowarn"], input_bytes=patch)
        if restored.returncode != 0:
            _git(root, ["checkout", "--", "."])
            if script_bytes is not None and not (root / script_name).is_file():
                (root / script_name).write_bytes(script_bytes)
            raise RuntimeError("repro_workspace_not_restored")
    if script_bytes is not None and not (root / script_name).is_file():
        (root / script_name).write_bytes(script_bytes)


def reproduction_before(services, root, script_name, timeout_s):
    """Run the script with tracked edits reversed, then put those edits back.

    No git repository, an empty diff, or a reverse that does not apply stays
    unknown. The script file is kept across the reverse so an untracked
    reproduction can still import the unfixed sources.
    """
    root = Path(root)
    if not (root / ".git").is_dir():
        return None
    script = root / script_name
    if not script.is_file():
        return None
    script_bytes = script.read_bytes()
    diff = _git(root, ["diff", "HEAD"])
    if diff.returncode != 0:
        return None
    patch = diff.stdout
    if not patch.strip():
        return None
    reversed_diff = _git(root, ["apply", "-R", "--whitespace=nowarn"], input_bytes=patch)
    if reversed_diff.returncode != 0:
        return None
    if not script.is_file():
        script.write_bytes(script_bytes)
    try:
        return _command_code(services, ["python3", script_name], timeout_s)
    finally:
        _restore_patch(root, patch, script_name, script_bytes)


def collect_repro_v1(services, *, changed_paths=(), patch_text="", timeout_s=60, script_name=REPRO_SCRIPT):
    """Measure the script and the touched public tests. Missing pieces stay unknown."""
    workspace = getattr(services, "workspace", None)
    root = getattr(workspace, "root", None)
    before = after = tests = None
    if root is not None:
        root = Path(root)
        script = root / script_name
        if script.is_file():
            before = reproduction_before(services, root, script_name, timeout_s)
            after = _command_code(services, ["python3", script_name], timeout_s)
        existing = [path for path in touched_test_paths(changed_paths) if (root / path).is_file()]
        if existing:
            tests = _command_code(services, ["python3", "-m", "pytest", *existing], timeout_s)
    evidence = repro_v1_evidence(
        before=before, after=after, tests=tests, changed_paths=changed_paths,
        patch_text=patch_text, root=root)
    evidence["changed_paths"] = [path for path in changed_paths if isinstance(path, str)]
    return evidence


def cached_repro_v1(services, *, changed_paths=(), patch_text="", timeout_s=60):
    cached = getattr(services, "_repro_v1_cache", None)
    if isinstance(cached, dict) and cached.get("criterion_id") == REPRO_CRITERION_ID:
        return cached
    bundle = collect_repro_v1(
        services, changed_paths=changed_paths, patch_text=patch_text, timeout_s=timeout_s)
    try:
        services._repro_v1_cache = bundle
    except Exception:
        pass
    return bundle


def evidence_for_service(services, context=None):
    """``StudyServices.local_criterion`` entry for the opt-in id."""
    del context
    changed = []
    log = getattr(services, "edit_log", None)
    if isinstance(log, list):
        for row in log:
            if isinstance(row, dict):
                changed.extend(path for path in row.get("paths") or [] if isinstance(path, str))
    return cached_repro_v1(services, changed_paths=changed)


def apply_definite_outcome(belief, tms, model, status, *, cell=CELL, topics=S1_TOPICS):
    """Update B and TMS only for a definite pass or fail. Return whether it wrote."""
    if status not in {"pass", "fail"} or not isinstance(model, str) or not model:
        return False
    success = status == "pass"
    belief.update_objective(model, cell, success)
    tms.record_outcome(model, tuple(topics), success)
    return True
