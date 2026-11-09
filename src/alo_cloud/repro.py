"""Opt-in reproduction signal. The frozen syntax criterion stays the default.

``repro-v1`` is a public check. It does not read a hidden grader. A pass is a
discriminating change: the reproduction script fails on the unfixed tree and
passes after the fix, or the same frozen public tests fail on the baseline and
pass after the fix under a comparable exit. A zero test code with no baseline,
a weakened suite, or an environment failure is unknown, not a pass. A
syntax-only workspace stays unknown, so B and TMS do not move. Definite pass
and fail are the only values those stores accept.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
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


_ENV_MARKERS = (
    "ModuleNotFoundError",
    "ImportError",
    "No module named",
    "ERROR collecting",
    "errors during collection",
    "command not found",
    "PermissionError",
)


def classify_test_exit(code, output, runner=None):
    """Coarse exit class for a public test command.

    ``failed`` is an assertion-style failure. ``environment`` is a missing
    dependency, a collection error, or another exit that is not the bug.
    ``unshown`` means the output does not show which of those happened.
    """
    if type(code) is not int:
        return "missing"
    text = output if isinstance(output, str) else ""
    if code == 0:
        return "passed"
    if any(marker in text for marker in _ENV_MARKERS):
        return "environment"
    if code in {2, 3, 4, 5, 127}:
        return "environment"
    if "AssertionError" in text or "FAILED" in text or " failed" in text:
        return "failed"
    if runner in {None, "pytest"} and code == 1 and ("FAILED" in text or " failed" in text):
        return "failed"
    return "unshown"


def frozen_tests_comparable(*, tests, tests_base, same_frozen_tests, comparable_exits):
    """True only when the same frozen suite failed on the baseline and passed after.

    A missing flag is not comparable. Callers that cannot show the suite and
    the exit class must pass ``False`` and stay off the pass path.
    """
    return (type(tests) is int and tests == 0
            and type(tests_base) is int and tests_base != 0
            and same_frozen_tests is True
            and comparable_exits is True)


def repro_v1_status(*, before, after, tests, tests_base=None, same_frozen_tests=None,
                    comparable_exits=None):
    """Map return codes onto pass, fail, or unknown.

    ``before`` is the reproduction script on the unfixed tree. ``after`` is the
    same script on the tree that contains the fix. ``tests`` is one return code
    for the touched public tests. ``tests_base`` is that same frozen suite on
    the baseline tree. Missing codes are ``None`` and are not zeros.

    Pass requires a discriminating change: ``before != 0`` and ``after == 0``,
    or ``tests_base != 0`` and ``tests == 0`` on the same frozen tests with a
    comparable functional exit. ``tests == 0`` alone is unknown.
    """
    failed_tests = type(tests) is int and tests != 0
    failed_after = type(after) is int and after != 0
    if failed_tests or failed_after:
        return "fail"
    if type(before) is int and before == 0 and after == 0:
        return "fail"
    reproduced = type(before) is int and before != 0 and after == 0
    tests_ok = type(tests) is int and tests == 0
    if reproduced and (tests is None or tests_ok):
        return "pass"
    if frozen_tests_comparable(
            tests=tests, tests_base=tests_base, same_frozen_tests=same_frozen_tests,
            comparable_exits=comparable_exits):
        return "pass"
    return "unknown"


def repro_v1_reason(*, before, after, tests, status, tests_base=None, same_frozen_tests=None,
                    comparable_exits=None):
    if status == "fail" and type(tests) is int and tests != 0:
        return "public_tests_failed"
    if status == "fail" and type(after) is int and after != 0:
        return "repro_failed_after_fix"
    if status == "fail" and before == 0 and after == 0:
        return "repro_did_not_fail_before_fix"
    if status == "pass" and type(before) is int and before != 0 and after == 0:
        return "repro_failed_before_and_passed_after"
    if status == "pass" and frozen_tests_comparable(
            tests=tests, tests_base=tests_base, same_frozen_tests=same_frozen_tests,
            comparable_exits=comparable_exits):
        return "frozen_tests_failed_before_and_passed_after"
    if status == "unknown" and type(tests) is int and tests == 0:
        return "public_tests_not_discriminating"
    return "repro_not_run"


def repro_v1_evidence(*, before, after, tests, changed_paths=(), patch_text="", root=None,
                      tests_base=None, same_frozen_tests=None, comparable_exits=None,
                      tests_failure_class=None, tests_base_failure_class=None):
    """Evidence for ``local_criterion``. Syntax alone cannot be a pass or a fail."""
    status = repro_v1_status(
        before=before, after=after, tests=tests, tests_base=tests_base,
        same_frozen_tests=same_frozen_tests, comparable_exits=comparable_exits)
    reason = repro_v1_reason(
        before=before, after=after, tests=tests, status=status, tests_base=tests_base,
        same_frozen_tests=same_frozen_tests, comparable_exits=comparable_exits)
    syntax_aware = "unknown"
    if root is not None and (type(after) is int or type(tests) is int):
        syntax_aware = public_quality_signal(
            root, changed_paths=changed_paths, patch_text=patch_text,
            repro_returncode=after, tests_returncode=tests)
    digest = hashlib.sha256(
        "|".join(str(part) for part in (
            before, after, tests, tests_base, status,
            same_frozen_tests is True, comparable_exits is True,
        )).encode("utf-8")).hexdigest()[:16]
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
        "tests_base_returncode": tests_base if type(tests_base) is int else None,
        "same_frozen_tests": same_frozen_tests is True,
        "comparable_exits": comparable_exits is True,
        "tests_failure_class": tests_failure_class if isinstance(tests_failure_class, str) else None,
        "tests_base_failure_class": tests_base_failure_class if isinstance(tests_base_failure_class, str) else None,
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
    result = _command_result(services, argv, timeout_s)
    if isinstance(result, dict):
        return result["returncode"]
    return None


def _command_result(services, argv, timeout_s):
    command = getattr(services, "command", None)
    if not callable(command):
        return None
    try:
        result = command(list(argv), timeout_s=timeout_s)
    except Exception:
        return None
    if not isinstance(result, dict) or type(result.get("returncode")) is not int:
        return None
    stdout = result.get("stdout") if isinstance(result.get("stdout"), str) else ""
    stderr = result.get("stderr") if isinstance(result.get("stderr"), str) else ""
    return {"returncode": result["returncode"], "output": stdout + "\n" + stderr}


def _git(root, args, *, input_bytes=None):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        input=input_bytes, capture_output=True, check=False,
    )


def _list_untracked(root):
    """Repo-relative paths of untracked, non-ignored files. Empty if git cannot list."""
    listed = _git(root, ["ls-files", "--others", "--exclude-standard", "-z"])
    if listed.returncode != 0:
        return []
    found = []
    for raw in listed.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", "surrogateescape")
        if rel and ".." not in Path(rel).parts and not rel.startswith(("/", "\\")):
            found.append(rel)
    return found


def _snapshot_untracked(root):
    """Bytes and mode of files that were never staged. ``git diff HEAD`` omits them."""
    saved = {}
    for rel in _list_untracked(root):
        path = root / rel
        if path.is_file():
            saved[rel] = (path.read_bytes(), path.stat().st_mode & 0o777)
    return saved


def _cached_added_paths(root, patch):
    """Paths the patch adds, read from the index so names are not parsed out of the diff.

    ``git apply --cached`` does not write the worktree, so an untracked copy of
    a new file does not make this probe fail. The index is returned to HEAD.
    """
    if not patch.strip():
        return []
    staged = _git(root, ["apply", "--cached", "--binary", "--whitespace=nowarn"], input_bytes=patch)
    if staged.returncode != 0:
        return []
    listed = _git(root, ["diff", "--cached", "--name-only", "--diff-filter=A", "-z"])
    cleared = _git(root, ["reset", "-q", "--mixed", "HEAD"])
    if cleared.returncode != 0:
        _git(root, ["read-tree", "HEAD"])
    if listed.returncode != 0:
        return []
    found = []
    for raw in listed.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", "surrogateescape")
        if rel and ".." not in Path(rel).parts and not rel.startswith(("/", "\\")):
            found.append(rel)
    return found


def _restore_candidate(root, orig_head, patch, script_name, script_bytes, untracked=None):
    """Put back the candidate fix even if the measurement staged the unfixed tree.

    ``workspace_patch`` runs ``git add -A``. A reproduction command therefore
    stages whatever the worktree is at that moment, including files that were
    never part of the candidate. ``git reset --hard`` then deletes those
    newly staged files, and it does not delete untracked files the command
    recreated. Snapshot the untracked files first, clear only the paths the
    saved patch adds, reapply, and write the snapshot back.
    """
    saved = {} if untracked is None else untracked
    reset = _git(root, ["reset", "--hard", orig_head])
    if reset.returncode != 0:
        raise RuntimeError("repro_workspace_not_restored")
    added_paths = _cached_added_paths(root, patch)
    if patch.strip():
        # reset --hard leaves untracked copies of files the patch adds.
        # A new-file hunk cannot apply onto that copy.
        for rel in added_paths:
            residue = root / rel
            if residue.is_file() or residue.is_symlink():
                residue.unlink()
        applied = _git(root, ["apply", "--binary", "--whitespace=nowarn"], input_bytes=patch)
        if applied.returncode != 0:
            raise RuntimeError("repro_workspace_not_restored")
    script = root / script_name
    if script_bytes is not None and not script.is_file():
        script.write_bytes(script_bytes)
    for rel, (data, mode) in saved.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)
    # A file the measurement created and left untracked is not part of the
    # candidate. Drop it before staging so it does not become part of the fix.
    allowed = set(saved) | set(added_paths)
    if script_bytes is not None:
        allowed.add(script_name)
    for rel in _list_untracked(root):
        if rel in allowed:
            continue
        extra = root / rel
        if extra.is_file() or extra.is_symlink():
            extra.unlink()
    # Match the live invariant: the index agrees with the restored worktree,
    # so a later checkout cannot revive the unfixed measurement.
    added = _git(root, ["add", "-A"])
    if added.returncode != 0:
        raise RuntimeError("repro_workspace_not_restored")


def _frozen_test_bytes(root, paths):
    """Bytes of ``paths`` at HEAD. A path that is not in HEAD is not frozen."""
    frozen = {}
    for rel in paths:
        if not isinstance(rel, str) or not rel:
            continue
        shown = _git(root, ["show", f"HEAD:{rel}"])
        if shown.returncode != 0:
            continue
        frozen[rel] = shown.stdout
    return frozen


def _clear_bytecode(root):
    """Drop cached bytecode so the baseline run cannot hide a same-size edit."""
    for directory in Path(root).rglob("__pycache__"):
        if directory.is_dir() and ".git" not in directory.parts:
            shutil.rmtree(directory, ignore_errors=True)


def _run_frozen_suite(services, root, frozen, timeout_s):
    """Run the frozen bytes on ``root`` and put the worktree copies back."""
    if not frozen:
        return None
    _clear_bytecode(root)
    saved = {}
    for rel, data in frozen.items():
        path = root / rel
        saved[rel] = path.read_bytes() if path.is_file() else None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    try:
        return _command_result(
            services, ["python3", "-m", "pytest", "-q", *list(frozen)], timeout_s)
    finally:
        for rel, current in saved.items():
            path = root / rel
            if current is None:
                if path.is_file() or path.is_symlink():
                    path.unlink()
            else:
                path.write_bytes(current)


def reproduction_before(services, root, script_name, timeout_s, on_baseline=None, require_script=True):
    """Run the script on the unfixed tree, then restore the candidate fix.

    No git repository, an empty diff, or a reverse that does not apply stays
    unknown. The saved patch is taken against HEAD and includes staged files,
    because a live workspace stages every edit. ``on_baseline`` runs while the
    tree is still the baseline, before the script, so the script cannot edit
    the suite that is being compared.
    """
    root = Path(root)
    if not (root / ".git").is_dir():
        return None
    script = root / script_name
    if require_script and not script.is_file():
        return None
    script_bytes = script.read_bytes() if script.is_file() else None
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
    # Taken before the reverse. git diff HEAD does not contain these files,
    # and a later git add -A plus reset --hard would delete them.
    untracked = _snapshot_untracked(root)
    try:
        reversed_diff = _git(root, ["apply", "-R", "--binary", "--whitespace=nowarn"], input_bytes=patch)
        if reversed_diff.returncode != 0:
            return None
        if callable(on_baseline):
            on_baseline()
        measured = None
        if script_bytes is not None:
            if not script.is_file():
                script.write_bytes(script_bytes)
            if script.is_file():
                measured = _command_code(services, ["python3", script_name], timeout_s)
        return measured
    finally:
        _restore_candidate(root, orig_head, patch, script_name, script_bytes, untracked)


def _prefer(measured, recorded):
    """A fresh run wins. A recorded transcript code fills a gap."""
    if type(measured) is int:
        return measured
    if type(recorded) is int:
        return recorded
    return None


def _suite_comparison(base_result, after_result):
    """Same runner and a shown functional failure, or not comparable."""
    if not isinstance(base_result, dict) or not isinstance(after_result, dict):
        return False, None, None
    base_class = classify_test_exit(base_result["returncode"], base_result.get("output"), "pytest")
    after_class = classify_test_exit(after_result["returncode"], after_result.get("output"), "pytest")
    comparable = base_class == "failed" and after_class == "passed"
    return comparable, base_class, after_class


def collect_repro_v1(services, *, changed_paths=(), patch_text="", timeout_s=60, script_name=REPRO_SCRIPT,
                     messages=None):
    """Measure the script and the same frozen public tests on both trees.

    Return codes already stored in ``messages`` are read first-class evidence.
    A harness run, when it returns an integer, replaces the recorded code.
    The baseline suite runs in the same process environment as the candidate
    suite. A path that is not at HEAD is not part of that frozen set.
    """
    recorded_before, recorded_after, recorded_tests = repro_codes_from_messages(messages)
    workspace = getattr(services, "workspace", None)
    root = getattr(workspace, "root", None)
    measured_before = measured_after = measured_tests = measured_base = None
    same_frozen = False
    comparable = False
    base_class = after_class = None
    if root is not None:
        root = Path(root)
        script = root / script_name
        frozen = _frozen_test_bytes(root, touched_test_paths(changed_paths)) if (root / ".git").is_dir() else {}
        holder = {}

        def on_baseline():
            if frozen:
                holder["base"] = _run_frozen_suite(services, root, frozen, timeout_s)

        if script.is_file() or frozen:
            measured_before = reproduction_before(
                services, root, script_name, timeout_s,
                on_baseline=on_baseline if frozen else None,
                require_script=script.is_file())
            if script.is_file():
                measured_after = _command_code(services, ["python3", script_name], timeout_s)
        if frozen and "base" in holder:
            after_suite = _run_frozen_suite(services, root, frozen, timeout_s)
            base_suite = holder.get("base")
            if isinstance(base_suite, dict) and isinstance(after_suite, dict):
                measured_base = base_suite["returncode"]
                measured_tests = after_suite["returncode"]
                same_frozen = True
                comparable, base_class, after_class = _suite_comparison(base_suite, after_suite)
            elif isinstance(after_suite, dict):
                measured_tests = after_suite["returncode"]
                after_class = classify_test_exit(after_suite["returncode"], after_suite.get("output"), "pytest")
        if measured_tests is None:
            # No frozen baseline. A current-tree failure is still a fail.
            # A current-tree zero is not a pass unless the script discriminates.
            existing = [path for path in touched_test_paths(changed_paths) if (root / path).is_file()]
            if existing:
                current = _command_result(
                    services, ["python3", "-m", "pytest", "-q", *existing], timeout_s)
                if isinstance(current, dict):
                    measured_tests = current["returncode"]
                    after_class = classify_test_exit(current["returncode"], current.get("output"), "pytest")
    before = _prefer(measured_before, recorded_before)
    after = _prefer(measured_after, recorded_after)
    tests = _prefer(measured_tests, recorded_tests)
    # A transcript code is not a baseline. Only the harness measurement is.
    tests_base = measured_base if type(measured_base) is int else None
    evidence = repro_v1_evidence(
        before=before, after=after, tests=tests, changed_paths=changed_paths,
        patch_text=patch_text, root=root, tests_base=tests_base,
        same_frozen_tests=same_frozen, comparable_exits=comparable,
        tests_failure_class=after_class, tests_base_failure_class=base_class)
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
