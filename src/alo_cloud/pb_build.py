"""A public build of the exact ProgramBench submission in a disposable workspace."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time

from .errors import IntegrationNotConfigured, PilotStopped, ProtocolError
from .pb_submission import capture_submission


# Run from /workspace in the task image. A fresh synthetic repository matches
# the pinned evaluator's seed_git step. Candidate output is captured separately
# from the controller's single JSON response and is never used as a receipt.
_BUILD_DRIVER = r"""
import json, os, re, subprocess, tempfile
env = dict(os.environ, GIT_AUTHOR_DATE="2000-01-01T00:00:00Z",
           GIT_COMMITTER_DATE="2000-01-01T00:00:00Z")
git = ["git", "-c", "user.email=gold@local", "-c", "user.name=gold",
       "-c", "commit.gpgsign=false"]
steps = [("prepare", git + ["-c", "init.defaultBranch=gold", "init", "-q"]),
         ("prepare", git + ["add", "-A"]),
         ("prepare", git + ["commit", "-q", "--allow-empty", "-m", "gold"]),
         ("compile", ["sh", "-c", "chmod +x ./compile.sh && ./compile.sh"])]
with tempfile.TemporaryFile() as log:
    for stage, command in steps:
        log.seek(0)
        log.truncate()
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            break
    size = log.seek(0, os.SEEK_END)
    log.seek(0)
    if size <= 4096:
        excerpt = log.read()
    else:
        header = log.read(700)
        log.seek(0)
        error = re.compile(rb"(?:fatal |linker )?error(?:\[|:| E[0-9])|"
                           rb"no such file|no rule to make target|undefined reference|"
                           rb"module lookup disabled|could not resolve host|file not found|"
                           rb"no matching package|cannot find|command not found", re.I)
        previous = b""
        context = b""
        while line := log.readline(8192):
            match = error.search(line)
            if match:
                offset = max(0, match.start() - 350)
                context = previous + line[offset:offset + 1000] + log.read(680)
                break
            previous = line[-200:]
        log.seek(-1000, os.SEEK_END)
        excerpt = header + b"\n...[truncated]...\n"
        if context:
            excerpt += b"First error context:\n" + context + b"\n...[truncated]...\n"
        excerpt += log.read(1000)
    diagnostic = excerpt.decode("utf-8", "replace")
    print(json.dumps(dict(stage=stage, returncode=result.returncode, diagnostic=diagnostic)))
"""


def _diagnostic(value):
    return "".join(char for char in value[-4096:] if char.isprintable() or char in "\n\t")


def _compile_failure(log):
    lower = log.lower()
    if any(x in lower for x in ("module lookup disabled", "no matching package named",
                                "network is unreachable", "could not resolve host",
                                "temporary failure in name resolution", "dial tcp: lookup")):
        return "offline_dependency"
    if any(x in lower for x in ("no such file or directory", "no rule to make target",
                                "no required module provides package", "can't find crate for")):
        return "missing_build_input"
    return "compile_failed"


def _export_artifact(candidate, destination, check_deadline):
    digest = hashlib.sha256()
    # The optional destination is controller-owned. Publish only a complete
    # fresh result, never a partial or overwritten previous artifact.
    staging = tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) if destination else nullcontext()
    with staging as output:
        try:
            with candidate.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    check_deadline()
                    digest.update(chunk)
                    if output:
                        output.write(chunk)
            if output:
                output.flush()
                os.chmod(output.name, 0o700)
                check_deadline()
                os.link(output.name, destination)
        finally:
            if output:
                Path(output.name).unlink(missing_ok=True)
    return digest.hexdigest()


def run_public_build(sandbox, snapshot_root, submission_sha256, *, timeout_s=60, artifact_path=None):
    """Return public build evidence. No provider, evaluator or live tree writes."""
    result = {
        "status": "unknown", "stage": "snapshot", "reason": "build_unavailable",
        "submission_sha256": submission_sha256, "returncode": None, "diagnostic": "",
        "artifact_present": None, "artifact_executable": None, "artifact_sha256": None,
    }
    if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        return {**result, "reason": "build_timeout"}
    deadline = time.monotonic() + timeout_s

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError
        return seconds

    try:
        with tempfile.TemporaryDirectory(prefix="alo-pb-build-") as temp:
            snapshot = capture_submission(snapshot_root, temp, check_deadline=remaining)
            if snapshot.sha256 != submission_sha256:
                return {**result, "reason": "submission_hash_mismatch"}
            if "compile.sh" not in snapshot.files:
                return {**result, "status": "fail", "reason": "missing_compile_sh",
                        "artifact_present": False, "artifact_executable": False,
                        "diagnostic": "Missing ./compile.sh at the submission root."}
            export = Path(artifact_path).absolute() if artifact_path is not None else None
            original = Path(snapshot_root).resolve()
            if export and (export.parent.resolve().is_relative_to(original)
                           or (sandbox is not None and export.parent.resolve().is_relative_to(sandbox.workspace))):
                raise ValueError("public_build_output_must_be_detached")
            result["stage"] = "sandbox"
            if sandbox is None:
                return {**result, "reason": "sandbox_not_configured"}
            execution = sandbox.public_build_command(
                snapshot.root, ["python3", "-I", "-c", _BUILD_DRIVER], remaining())
            if execution.get("timed_out"):
                return {**result, "reason": "build_timeout"}
            if execution.get("failure") or execution["returncode"] != 0:
                return {**result, "reason": execution.get("failure") or "sandbox_execution_failed",
                        "diagnostic": _diagnostic(execution.get("stderr", ""))}
            payload = json.loads(execution["stdout"])
            if (not isinstance(payload, dict) or payload.get("stage") not in {"prepare", "compile"}
                    or type(payload.get("returncode")) is not int or not isinstance(payload.get("diagnostic"), str)):
                raise ValueError("invalid_build_receipt")
            stage, code = payload["stage"], payload["returncode"]
            result.update(stage=stage, diagnostic=_diagnostic(payload["diagnostic"]))
            if stage == "prepare":
                return {**result, "reason": "build_setup_failed"}
            result["returncode"] = code
            candidate = snapshot.root / "executable"
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                info = None
            runnable = (info is not None and stat.S_ISREG(info.st_mode)
                        and info.st_size > 0 and os.access(candidate, os.X_OK))
            result.update(artifact_present=info is not None, artifact_executable=runnable)
            if code != 0:
                return {**result, "status": "fail", "reason": _compile_failure(result["diagnostic"])}
            result["stage"] = "artifact"
            if not runnable:
                detail = ("./compile.sh exited 0 without creating ./executable." if info is None
                          else "./executable must be a nonempty executable regular file.")
                return {**result, "status": "fail", "diagnostic": (detail + "\n" + result["diagnostic"])[:4096],
                        "reason": "missing_artifact" if info is None else "invalid_artifact"}
            result["artifact_sha256"] = _export_artifact(candidate, export, remaining)
            return {**result, "status": "pass", "reason": "built"}
    except TimeoutError:
        return {**result, "reason": "build_timeout"}
    except (OSError, ValueError, KeyError, TypeError, IntegrationNotConfigured, PilotStopped, ProtocolError) as error:
        return {**result, "reason": "build_infrastructure_unknown",
                "diagnostic": _diagnostic(str(error))}


def _probe_tree_digest(root, remaining):
    """Hash visible file changes without reading a reference or following links."""
    digest = hashlib.sha256()
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        for name in sorted(dirs + files):
            remaining()
            path = Path(base, name)
            relative = path.relative_to(root).as_posix()
            if relative == "executable":
                continue
            mode = path.lstat().st_mode
            digest.update(os.fsencode(relative) + b"\0" + str(mode & 0o170111).encode() + b"\0")
            if stat.S_ISREG(mode):
                digest.update(bytes.fromhex(_export_artifact(path, None, remaining)))
            elif stat.S_ISLNK(mode):
                digest.update(os.fsencode(os.readlink(path)) + b"\0")
    return digest.hexdigest()


def snapshot_differential(sandbox, snapshot, artifact, artifact_sha256, invocations, wall_remaining):
    """Compare public runs from identical fresh input, including filesystem effects."""
    from .pb_contract import PB_SIGNAL_CANDIDATE, blackbox_probe_argv, run_blackbox_differential

    def remaining():
        seconds = wall_remaining()
        if seconds <= 0:
            raise TimeoutError
        return seconds

    def observe(argv):
        with tempfile.TemporaryDirectory(prefix="alo-pb-probe-") as directory:
            start = capture_submission(snapshot.root, directory, check_deadline=remaining)
            if start.sha256 != snapshot.sha256:
                raise ValueError("public_probe_snapshot_changed")
            staged = start.root / PB_SIGNAL_CANDIDATE
            staged.parent.mkdir(parents=True, exist_ok=True)
            if _export_artifact(artifact, staged, remaining) != artifact_sha256:
                raise ValueError("public_probe_artifact_changed")
            (start.root / "executable").symlink_to("/workspace/executable")
            before = _probe_tree_digest(start.root, remaining)
            response = sandbox.public_probe_command(start.root, argv, min(30, remaining()))
            if response.get("timed_out") or response.get("failure"):
                return None
            return {**response, "fs_before": before, "fs_after": _probe_tree_digest(start.root, remaining)}

    try:
        signals = []
        for args in invocations or [()]:
            # Probe scratch is outside the sealed submission. A missing fixture
            # can make both programs report identical errors and fake a match.
            if any(".pb-probe/" in arg or arg == ".pb-probe" for arg in args):
                return "unknown"
            ref = blackbox_probe_argv("./executable", args)
            candidate = blackbox_probe_argv("./" + PB_SIGNAL_CANDIDATE, args)
            signals.append(run_blackbox_differential(ref, candidate, runner=observe))
        if "unknown" in signals:
            return "unknown"
        return "fail" if "fail" in signals else "pass"
    except (OSError, ValueError, TypeError, IntegrationNotConfigured, PilotStopped):
        return "unknown"
