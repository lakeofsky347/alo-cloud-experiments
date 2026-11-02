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
from .public_tools import leading_json_object


REPRO_CRITERION_ID = "repro-v1"
REPRO_SCRIPT = "alo_repro.py"
REPRO_INSTRUCTION = (
    " s1 must do these steps in order, and no other edit before step 3. "
    "1. Write alo_repro.py at the workspace root. It must fail on the unfixed code. "
    "2. Run exactly `python3 alo_repro.py` and keep the non-zero exit. "
    "3. Edit the source to fix the bug. Do not change alo_repro.py so step 2 would pass. "
    "4. Run exactly `python3 alo_repro.py` again. This run must exit 0. "
    "The criterion reads those two return codes from the transcript. "
    "A script that passes both before and after the fix is a failure. "
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


def _argv_of(value):
    if not isinstance(value, dict):
        return None
    arguments = value.get("arguments") if isinstance(value.get("arguments"), dict) else {}
    argv = arguments.get("argv")
    if value.get("tool") == "command" and isinstance(argv, list) and all(isinstance(item, str) for item in argv):
        return argv
    if value.get("tool") == "execute_code" and isinstance(arguments.get("code"), str):
        return ["python3", "-c", arguments["code"]]
    return None


def _returncode_of(value):
    if not isinstance(value, dict) or value.get("tool") not in {"command", "execute_code"}:
        return None
    result = value.get("result")
    if isinstance(result, dict) and type(result.get("returncode")) is int:
        return result["returncode"]
    return None


def command_results_from_messages(messages):
    """Pair each command with its return code.

    The assistant message may be a JSON object followed by stray text. The
    tool result is the next user message, also read as a leading JSON object.
    """
    found = []
    pending = None
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        value = leading_json_object(content) if isinstance(content, str) else None
        if message.get("role") == "assistant":
            pending = _argv_of(value)
        elif message.get("role") == "user" and pending is not None:
            code = _returncode_of(value)
            if code is not None:
                found.append({"argv": pending, "returncode": code})
            pending = None
    return found


def _is_repro_argv(argv):
    return any(item == REPRO_SCRIPT or item.endswith("/" + REPRO_SCRIPT) or item.endswith(REPRO_SCRIPT)
               for item in argv)


# Inspection commands can name a test file without executing it.
_NOT_TEST_RUNNERS = frozenset({
    "find", "sed", "cat", "grep", "egrep", "fgrep", "ls", "head", "tail", "wc",
    "less", "more", "rg", "awk", "cut", "sort", "diff", "echo", "printf", "cp",
    "mv", "rm", "mkdir", "touch", "chmod", "ln", "xargs", "basename",
})
_PYTEST_MODULES = frozenset({"pytest", "unittest", "nose", "nose2", "tox", "django"})
_DIRECT_TEST_RUNNERS = frozenset({
    "pytest", "py.test", "tox", "nox", "nose", "nose2", "trial",
})


def _argv_name(item):
    return Path(item).name


def _is_python_argv(item):
    name = _argv_name(item)
    return name in {"python", "python2", "python3"} or name.startswith(("python2.", "python3."))


def _is_executed_test_file(target):
    name = _argv_name(target)
    if name in {"runtests.py", "tests.py"}:
        return True
    return name.startswith("test_") or name.endswith("_test.py")


def _is_test_argv(argv):
    """True only when the command itself executes tests.

    ``find``, ``sed``, ``cat``, ``grep`` and ``ls`` can name a test file
    without running it. Those stay false.
    """
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        return False
    head = _argv_name(argv[0])
    if head in _NOT_TEST_RUNNERS:
        return False
    if head in _DIRECT_TEST_RUNNERS:
        return True
    if _is_python_argv(argv[0]):
        args = argv[1:]
        if len(args) >= 2 and args[0] == "-m" and _argv_name(args[1]) in _PYTEST_MODULES:
            return True
        positional = [item for item in args if not item.startswith("-")]
        if not positional:
            return False
        target = positional[0]
        if _argv_name(target) == "runtests.py":
            return True
        if target == "bin/test" or target.endswith("/bin/test"):
            return True
        if _argv_name(target) == "manage.py" and "test" in argv:
            return True
        if target.endswith(".py") and _is_executed_test_file(target):
            return True
        return False
    if head == "manage.py" and "test" in argv:
        return True
    return False


def repro_codes_from_messages(messages):
    """Before, after, and public-test codes recorded in one node transcript.

    Two reproduction runs are before then after. One run is after only.
    """
    commands = command_results_from_messages(messages)
    repro = [item for item in commands if _is_repro_argv(item["argv"])]
    tests = [item for item in commands if _is_test_argv(item["argv"]) and not _is_repro_argv(item["argv"])]
    before = after = test_code = None
    if len(repro) >= 2:
        before = repro[0]["returncode"]
        after = repro[-1]["returncode"]
    elif len(repro) == 1:
        after = repro[0]["returncode"]
    if tests:
        test_code = tests[-1]["returncode"]
    return before, after, test_code


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


def _restore_candidate(root, orig_head, patch, script_name, script_bytes):
    """Put back the candidate fix even if the measurement staged the unfixed tree.

    ``workspace_patch`` runs ``git add -A``. A reproduction command therefore
    stages whatever the worktree is at that moment. Restoring with
    ``git checkout -- .`` would then reload that index and drop the fix.
    Reset to the saved HEAD and reapply the saved patch instead.
    """
    reset = _git(root, ["reset", "--hard", orig_head])
    if reset.returncode != 0:
        raise RuntimeError("repro_workspace_not_restored")
    if patch.strip():
        # The measurement may have recreated alo_repro.py. A new-file hunk
        # cannot apply onto that copy, and reset --hard already removed it
        # when it was only in the patch. Apply before writing any fallback.
        applied = _git(root, ["apply", "--binary", "--whitespace=nowarn"], input_bytes=patch)
        if applied.returncode != 0:
            raise RuntimeError("repro_workspace_not_restored")
    script = root / script_name
    if script_bytes is not None and not script.is_file():
        script.write_bytes(script_bytes)
    # Match the live invariant: the index agrees with the restored worktree,
    # so a later checkout cannot revive the unfixed measurement.
    added = _git(root, ["add", "-A"])
    if added.returncode != 0:
        raise RuntimeError("repro_workspace_not_restored")


def reproduction_before(services, root, script_name, timeout_s):
    """Run the script on the unfixed tree, then restore the candidate fix.

    No git repository, an empty diff, or a reverse that does not apply stays
    unknown. The saved patch is taken against HEAD and includes staged files,
    because a live workspace stages every edit.
    """
    root = Path(root)
    if not (root / ".git").is_dir():
        return None
    script = root / script_name
    if not script.is_file():
        return None
    script_bytes = script.read_bytes()
    head = _git(root, ["rev-parse", "HEAD"])
    if head.returncode != 0:
        return None
    orig_head = head.stdout.decode().strip()
    diff = _git(root, ["diff", "--binary", "HEAD"])
    if diff.returncode != 0:
        return None
    patch = diff.stdout
    if not patch.strip():
        return None
    try:
        reversed_diff = _git(root, ["apply", "-R", "--binary", "--whitespace=nowarn"], input_bytes=patch)
        if reversed_diff.returncode != 0:
            return None
        if not script.is_file():
            script.write_bytes(script_bytes)
        measured = _command_code(services, ["python3", script_name], timeout_s)
        return measured
    finally:
        _restore_candidate(root, orig_head, patch, script_name, script_bytes)


def _prefer(measured, recorded):
    """A fresh run wins. A recorded transcript code fills a gap."""
    if type(measured) is int:
        return measured
    if type(recorded) is int:
        return recorded
    return None


def collect_repro_v1(services, *, changed_paths=(), patch_text="", timeout_s=60, script_name=REPRO_SCRIPT,
                     messages=None):
    """Measure the script and the touched public tests. Missing pieces stay unknown.

    Return codes already stored in ``messages`` are read first-class evidence.
    A harness run, when it returns an integer, replaces the recorded code.
    """
    recorded_before, recorded_after, recorded_tests = repro_codes_from_messages(messages)
    workspace = getattr(services, "workspace", None)
    root = getattr(workspace, "root", None)
    measured_before = measured_after = measured_tests = None
    if root is not None:
        root = Path(root)
        script = root / script_name
        if script.is_file():
            measured_before = reproduction_before(services, root, script_name, timeout_s)
            measured_after = _command_code(services, ["python3", script_name], timeout_s)
        existing = [path for path in touched_test_paths(changed_paths) if (root / path).is_file()]
        if existing:
            measured_tests = _command_code(services, ["python3", "-m", "pytest", *existing], timeout_s)
    before = _prefer(measured_before, recorded_before)
    after = _prefer(measured_after, recorded_after)
    tests = _prefer(measured_tests, recorded_tests)
    evidence = repro_v1_evidence(
        before=before, after=after, tests=tests, changed_paths=changed_paths,
        patch_text=patch_text, root=root)
    evidence["changed_paths"] = [path for path in changed_paths if isinstance(path, str)]
    evidence["recorded_before_returncode"] = recorded_before if type(recorded_before) is int else None
    evidence["recorded_after_returncode"] = recorded_after if type(recorded_after) is int else None
    return evidence


def cached_repro_v1(services, *, changed_paths=(), patch_text="", timeout_s=60, messages=None):
    cached = getattr(services, "_repro_v1_cache", None)
    if isinstance(cached, dict) and cached.get("criterion_id") == REPRO_CRITERION_ID:
        return cached
    bundle = collect_repro_v1(
        services, changed_paths=changed_paths, patch_text=patch_text, timeout_s=timeout_s,
        messages=messages)
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
