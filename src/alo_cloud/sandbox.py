"""Restricted workspace and an optional Docker command backend.

File tools stay on the controller so syntax attribution can name the write.
``compile()`` checks only Python files written through ``write_file``. It does
not execute them, and it does not grade behavior. Command execution is a
separate backend. The Docker argv drops capabilities, network and host
credential environment variables; building that argv does not start a daemon.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

from .errors import IntegrationNotConfigured, PilotStopped, ProtocolError


CRITERION_ID = "python_write_file_compile_syntax_v1"
MAX_FILE_BYTES = 1_048_576
MAX_SEARCH_CHARS = 10_000
MAX_TOOL_OUTPUT_CHARS = 1_048_576
CREDENTIAL_ENV_NAMES = ("DEEPSEEK_API_KEY", "MIMO_API_KEY", "ZAI_API_KEY")
DEFAULT_SANDBOX_MEMORY = "4g"
DEFAULT_SANDBOX_PIDS = "512"
DEFAULT_SANDBOX_TMPFS = "1g"
_MEMORY = re.compile(r"\d+[bBkKmMgG]")
_PIDS = re.compile(r"[1-9]\d{0,6}")
_TMPFS = re.compile(r"\d+[bBkKmMgG]")


class SandboxUnknown(PilotStopped):
    """The sandbox result is not a known pass, fail or timeout."""


def _relative(root, path):
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path or "\0" in path:
        raise ProtocolError("workspace_path_rejected")
    pure = PurePosixPath(path)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ProtocolError("workspace_path_rejected")
    candidate = (root / pure).resolve()
    base = root.resolve()
    if not candidate.is_relative_to(base):
        raise ProtocolError("workspace_path_rejected")
    return pure.as_posix(), candidate


class Workspace:
    """Controller-side files. Only successful ``write_file`` calls are attributed."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.attributed = []

    def list_files(self):
        found = []
        root = self.root.resolve()
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(name for name in dirnames
                                 if not (Path(dirpath) / name).is_symlink())
            for name in sorted(filenames):
                candidate = Path(dirpath) / name
                if candidate.is_symlink():
                    continue
                resolved = candidate.resolve()
                if resolved.is_file() and resolved.is_relative_to(root):
                    found.append(resolved.relative_to(root).as_posix())
        return found

    def read_file(self, path):
        _relative_path, candidate = _relative(self.root, path)
        if candidate.is_symlink() or not candidate.is_file():
            raise ProtocolError("workspace_file_missing")
        try:
            return candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ProtocolError("workspace_file_unreadable") from error

    def write_file(self, path, content):
        if not isinstance(content, str):
            raise ProtocolError("workspace_content_must_be_text")
        encoded = content.encode()
        if len(encoded) > MAX_FILE_BYTES:
            raise ProtocolError("workspace_content_too_large")
        relative, candidate = _relative(self.root, path)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if candidate.is_symlink():
            raise ProtocolError("workspace_path_rejected")
        try:
            candidate.write_text(content, encoding="utf-8")
        except OSError as error:
            raise SandboxUnknown("workspace_write_failed") from error
        self.attributed.append(relative)
        return {"path": relative, "bytes": len(encoded)}

    def search(self, text):
        if not isinstance(text, str) or not text or len(text) > MAX_SEARCH_CHARS:
            raise ProtocolError("workspace_search_text_rejected")
        matches = []
        for relative in self.list_files():
            try:
                content = self.read_file(relative)
            except ProtocolError:
                continue
            for index, line in enumerate(content.splitlines(), start=1):
                if text in line:
                    matches.append({"path": relative, "line": index})
                    if len(matches) >= 50:
                        return matches
        return matches

    def take_attributed(self):
        found = list(self.attributed)
        self.attributed = []
        return found


def python_syntax_evidence(workspace, relative_paths):
    """Reliable pass/fail only for attributed ``.py`` files and ``SyntaxError``."""
    python_paths = [path for path in relative_paths if path.endswith(".py")]
    digest = hashlib.sha256("\n".join(python_paths).encode()).hexdigest()[:16]
    unknown = {"status": "unknown", "reliable": False, "scope": "node_local",
               "criterion_id": CRITERION_ID, "evidence_id": "syn-" + digest,
               "reason": "no_attributed_python_write"}
    if not python_paths:
        return unknown
    failures = []
    root = workspace.root.resolve()
    try:
        for relative in python_paths:
            _relative_path, candidate = _relative(root, relative)
            if candidate.is_symlink() or not candidate.is_file() or not candidate.resolve().is_relative_to(root):
                unknown["reason"] = "attributed_python_unreadable"
                return unknown
            source = candidate.read_text(encoding="utf-8")
            compile(source, relative, "exec")
    except SyntaxError:
        failures.append(relative)
    except (OSError, UnicodeError, ValueError):
        unknown["reason"] = "attributed_python_unreadable"
        return unknown
    if failures:
        return {"status": "fail", "reliable": True, "scope": "node_local",
                "criterion_id": CRITERION_ID, "evidence_id": "syn-" + digest,
                "reason": "syntax_error", "paths": failures}
    return {"status": "pass", "reliable": True, "scope": "node_local",
            "criterion_id": CRITERION_ID, "evidence_id": "syn-" + digest,
            "reason": "compile_accepted"}


def _text(value):
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > MAX_TOOL_OUTPUT_CHARS:
        raise SandboxUnknown("sandbox_outcome_unknown")
    return value


def known_sandbox_result(result):
    if not isinstance(result, dict):
        raise SandboxUnknown("sandbox_outcome_unknown")
    stdout, stderr = _text(result.get("stdout")), _text(result.get("stderr"))
    if result.get("timed_out") is True:
        code = 124 if result.get("returncode") is None else result.get("returncode")
        if type(code) is not int:
            raise SandboxUnknown("sandbox_outcome_unknown")
        return {"returncode": code, "stdout": stdout, "stderr": stderr, "timed_out": True}
    if type(result.get("returncode")) is not int:
        raise SandboxUnknown("sandbox_outcome_unknown")
    return {"returncode": result["returncode"], "stdout": stdout, "stderr": stderr, "timed_out": False}


def _daemon_failure(result):
    stderr = result.get("stderr") if isinstance(result, dict) else ""
    text = stderr.lower() if isinstance(stderr, str) else ""
    code = result.get("returncode") if isinstance(result, dict) else None
    return code == 125 or "cannot connect to the docker daemon" in text or "is the docker daemon running" in text


def subprocess_docker_runner(argv, timeout_s):
    """Host docker CLI with an allowlisted environment. Not used by offline tests."""
    env = {"LANG": "C.UTF-8"}
    path = os.environ.get("PATH")
    if isinstance(path, str) and path:
        env["PATH"] = path
    try:
        completed = subprocess.run(argv, capture_output=True, timeout=timeout_s, env=env, check=False)
    except subprocess.TimeoutExpired as error:
        return {"returncode": 124, "stdout": _text((error.stdout or b"").decode(errors="replace")),
                "stderr": _text((error.stderr or b"").decode(errors="replace")), "timed_out": True}
    except OSError:
        raise SandboxUnknown("docker_process_not_started") from None
    return {"returncode": completed.returncode,
            "stdout": completed.stdout.decode(errors="replace"),
            "stderr": completed.stderr.decode(errors="replace"),
            "timed_out": False}


def _image_ok(image):
    return isinstance(image, str) and bool(image) and not image.startswith("-") and not any(ch.isspace() for ch in image)


def sandbox_memory(value=None):
    chosen = value if value is not None else os.environ.get("ALO_SANDBOX_MEMORY") or DEFAULT_SANDBOX_MEMORY
    if not isinstance(chosen, str) or not _MEMORY.fullmatch(chosen):
        raise PilotStopped("invalid_sandbox_memory")
    return chosen


def sandbox_pids(value=None):
    chosen = value if value is not None else os.environ.get("ALO_SANDBOX_PIDS") or DEFAULT_SANDBOX_PIDS
    if not isinstance(chosen, str) or not _PIDS.fullmatch(chosen):
        raise PilotStopped("invalid_sandbox_pids")
    return chosen


def sandbox_tmpfs(value=None):
    chosen = value if value is not None else os.environ.get("ALO_SANDBOX_TMPFS") or DEFAULT_SANDBOX_TMPFS
    if not isinstance(chosen, str) or not _TMPFS.fullmatch(chosen):
        raise PilotStopped("invalid_sandbox_tmpfs")
    return chosen


class DockerSandbox:
    """Build a no-network ``docker run`` and execute it only through ``runner`` or an explicit permit.

    ``image_resolver`` is called when a command is built, so a unit can supply
    the official per-task image. A plain ``image`` string remains the fallback.
    """

    def __init__(self, workspace, *, image=None, image_resolver=None, runner=None,
                 permit_execution=False, memory=None, pids_limit=None, tmpfs_size=None,
                 extra_mounts=()):
        if image_resolver is None and not _image_ok(image):
            raise PilotStopped("invalid_sandbox_image")
        if image_resolver is not None and not callable(image_resolver):
            raise PilotStopped("invalid_sandbox_image")
        self.workspace = Path(workspace).resolve()
        self.image = image
        self.image_resolver = image_resolver
        self.runner = runner
        self.permit_execution = bool(permit_execution)
        self.memory = sandbox_memory(memory)
        self.pids_limit = sandbox_pids(pids_limit)
        self.tmpfs_size = sandbox_tmpfs(tmpfs_size)
        self.extra_mounts = tuple(extra_mounts)
        if self.permit_execution and self.runner is not None:
            raise PilotStopped("one_sandbox_runner")
        for mount in self.extra_mounts:
            if not isinstance(mount, str) or not mount.startswith("/") or ".." in mount or "\0" in mount:
                raise PilotStopped("invalid_sandbox_mount")

    @property
    def execution_ready(self):
        return self.runner is not None or self.permit_execution

    def resolved_image(self):
        image = self.image_resolver() if self.image_resolver is not None else self.image
        if not _image_ok(image):
            raise PilotStopped("invalid_sandbox_image")
        return image

    def docker_argv(self, argv):
        image = self.resolved_image()
        command = [
            "docker", "run", "--rm",
            "--network=none",
            "--cap-drop=ALL",
            f"--pids-limit={self.pids_limit}",
            f"--memory={self.memory}",
            "--read-only",
            "--tmpfs", f"/tmp:rw,nosuid,nodev,size={self.tmpfs_size}",
            "--workdir", "/workspace",
            "--mount", f"type=bind,src={self.workspace},dst=/workspace",
        ]
        for mount in self.extra_mounts:
            command.extend(["--mount", f"type=bind,src={self.workspace},dst={mount}"])
        command.extend([
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTHONUNBUFFERED=1",
            "--env", "PYTHONPATH=/workspace",
            "--env", "LANG=C.UTF-8",
            image,
            *argv,
        ])
        rendered = "\n".join(command)
        for name in CREDENTIAL_ENV_NAMES:
            if name in rendered:
                raise SandboxUnknown("sandbox_env_rejected")
            value = os.environ.get(name)
            try:
                if value and value in rendered:
                    raise SandboxUnknown("sandbox_env_rejected")
            finally:
                value = None
        return command

    def command(self, argv, timeout_s):
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
            raise ProtocolError("sandbox_argv_rejected")
        if any("\0" in item for item in argv):
            raise ProtocolError("sandbox_argv_rejected")
        command = self.docker_argv(argv)
        if self.runner is not None:
            runner = self.runner
        elif self.permit_execution:
            runner = subprocess_docker_runner
        else:
            raise IntegrationNotConfigured("docker_execution_not_permitted")
        try:
            result = runner(command, timeout_s)
        except (SandboxUnknown, ProtocolError, PilotStopped):
            raise
        except Exception:
            raise SandboxUnknown("sandbox_outcome_unknown") from None
        if _daemon_failure(result):
            raise SandboxUnknown("docker_daemon_unavailable")
        return known_sandbox_result(result)
