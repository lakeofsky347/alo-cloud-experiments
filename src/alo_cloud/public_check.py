"""Public quality signal for one workspace. No hidden tests and no model calls.

The signal is pass, fail, or unknown. Unknown is not a failure and is not a
zero. It is the value fed to the quality pseudo-posterior when it is pass or
fail; unknown leaves that posterior unchanged.
"""
from __future__ import annotations

import ast
from pathlib import Path


def quality_from_signal(status):
    """Map a public check onto the unit interval. Unknown stays missing."""
    if status == "pass":
        return 1.0
    if status == "fail":
        return 0.0
    return None


def _py_compile_ok(root, paths):
    root = Path(root)
    python_paths = [path for path in paths if str(path).endswith(".py")]
    if not python_paths:
        return "unknown"
    for relative in python_paths:
        candidate = root / relative
        if not candidate.is_file():
            return "unknown"
        try:
            source = candidate.read_text(encoding="utf-8")
            ast.parse(source, filename=str(relative))
        except SyntaxError:
            return "fail"
        except (OSError, UnicodeError, ValueError):
            return "unknown"
    return "pass"


def _test_module(path):
    name = Path(path).name
    parts = Path(path).parts
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return any(part in {"tests", "test"} for part in parts[:-1])


def public_quality_signal(root, *, changed_paths=(), patch_text="", repro_returncode=None,
                          tests_returncode=None):
    """Combine the public checks that do not read a hidden grader.

    A non-empty patch that is text counts as applying when it is a unified diff
    or when changed paths exist on disk. ``py_compile`` is ``ast.parse`` of
    touched Python files. A recorded public repro or touched-test command
    contributes its exit code. Missing pieces stay unknown and do not flip a
    pass into a fail.
    """
    root = Path(root)
    paths = [path for path in changed_paths if isinstance(path, str)]
    patch = patch_text if isinstance(patch_text, str) else ""
    has_patch = bool(patch.strip()) or bool(paths)
    if not has_patch and repro_returncode is None and tests_returncode is None:
        return "unknown"
    if patch.strip().startswith("diff ") or patch.strip().startswith("--- ") or paths:
        applies = "pass"
    elif patch.strip():
        applies = "unknown"
    else:
        applies = "unknown"
    compiled = _py_compile_ok(root, paths) if paths else "unknown"
    repro = "unknown"
    if type(repro_returncode) is int:
        repro = "pass" if repro_returncode == 0 else "fail"
    tests = "unknown"
    touched_tests = [path for path in paths if path.endswith(".py") and _test_module(path)]
    if type(tests_returncode) is int:
        tests = "pass" if tests_returncode == 0 else "fail"
    elif touched_tests:
        tests = compiled if compiled in {"pass", "fail"} else "unknown"
    observed = [item for item in (applies, compiled, repro, tests) if item in {"pass", "fail"}]
    if not observed:
        return "unknown"
    if any(item == "fail" for item in observed):
        return "fail"
    return "pass"
