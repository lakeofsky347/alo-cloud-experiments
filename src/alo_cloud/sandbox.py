"""Restricted workspace and an optional Docker command backend.

File tools stay on the controller so syntax attribution can name the write.
``compile()`` checks only Python files written through ``write_file``. It does
not execute them, and it does not grade behavior. Command execution is a
separate backend. The Docker argv drops capabilities, network and host
credential environment variables; building that argv does not start a daemon.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import uuid

from .errors import IntegrationNotConfigured, PilotStopped, ProtocolError


CRITERION_ID = "python_write_file_compile_syntax_v1"
MAX_FILE_BYTES = 1_048_576
MAX_SEARCH_CHARS = 10_000
MAX_LISTED_FILES = 200
MAX_READ_CHARS = 32_000
MAX_SEARCH_FILE_BYTES = 2_000_000
MAX_SEARCH_HITS = 100
MAX_SEARCH_LINE_CHARS = 500
MAX_TOOL_OUTPUT_CHARS = 1_048_576
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".tgz",
    ".bz2", ".xz", ".7z", ".whl", ".jar", ".so", ".o", ".a", ".pyc", ".pyo", ".exe",
    ".bin", ".dylib", ".dll", ".class", ".wasm",
}
CREDENTIAL_ENV_NAMES = ("DEEPSEEK_API_KEY", "MIMO_API_KEY", "ZAI_API_KEY", "FOXROUTER_API_KEY", "GLM_API_KEY")
DEFAULT_SANDBOX_MEMORY = "4g"
DEFAULT_SANDBOX_PIDS = "512"
DEFAULT_SANDBOX_TMPFS = "1g"
# Official SWE-bench instance images put base conda ahead of the testbed env.
SWE_TESTBED_PATH = (
    "/opt/miniconda3/envs/testbed/bin:"
    "/opt/miniconda3/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
_MEMORY = re.compile(r"\d+[bBkKmMgG]")
_PIDS = re.compile(r"[1-9]\d{0,6}")
_TMPFS = re.compile(r"\d+[bBkKmMgG]")
# Docker container names. Parallel chains pass one so their ``docker run``
# processes cannot share a name. Unset leaves the historical argv unchanged.
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}")


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

    def _file_paths(self, under=None):
        found = []
        root = self.root.resolve()
        start = root if under is None else under
        for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
            dirnames[:] = sorted(
                name for name in dirnames
                if name != ".git" and not (Path(dirpath) / name).is_symlink()
            )
            if Path(dirpath).name == ".git":
                continue
            for name in sorted(filenames):
                candidate = Path(dirpath) / name
                if candidate.is_symlink():
                    continue
                resolved = candidate.resolve()
                if resolved.is_file() and resolved.is_relative_to(root):
                    relative = resolved.relative_to(root).as_posix()
                    if relative == ".git" or relative.startswith(".git/"):
                        continue
                    found.append(relative)
        return found

    def list_files(self, path=None):
        """Return workspace paths. ``.git`` is omitted. ``path`` selects a subdirectory."""
        start = None
        if path:
            relative, candidate = _relative(self.root, path)
            if candidate.is_symlink():
                raise ProtocolError("workspace_path_rejected")
            if not candidate.exists():
                raise ProtocolError("workspace_file_missing")
            if candidate.is_file():
                return [relative]
            start = candidate
        found = self._file_paths(start)
        if len(found) <= MAX_LISTED_FILES:
            return found
        return {"paths": found[:MAX_LISTED_FILES], "truncated": True,
                "omitted": len(found) - MAX_LISTED_FILES}

    def _read_text(self, candidate):
        try:
            return candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ProtocolError("workspace_file_unreadable") from error

    def read_file(self, path, start_line=None, end_line=None):
        _relative_path, candidate = _relative(self.root, path)
        if candidate.is_symlink() or not candidate.is_file():
            raise ProtocolError("workspace_file_missing")
        text = self._read_text(candidate)
        lines = text.splitlines()
        total = len(lines)
        if start_line is None and end_line is None:
            if len(text) <= MAX_READ_CHARS:
                return text
            shown = text[:MAX_READ_CHARS]
            shown_lines = shown.count("\n") + (0 if shown.endswith("\n") else 1)
            return (shown + f"\n[truncated] showing about {shown_lines} lines of {total}. "
                    "Pass start_line and end_line to page.")
        start = 1 if start_line is None else start_line
        end = total if end_line is None else end_line
        if start > end:
            raise ProtocolError("workspace_line_range_rejected")
        chunk = lines[start - 1:end]
        body = "\n".join(chunk)
        shown_end = min(end, total)
        if len(body) > MAX_READ_CHARS:
            body = body[:MAX_READ_CHARS]
            return (body + f"\n[truncated] lines {start}-{shown_end} of {total} exceed one page. "
                    "Narrow start_line and end_line.")
        if start > 1 or shown_end < total:
            return body + f"\n[page] lines {start}-{shown_end} of {total}. Pass start_line and end_line to page."
        return body

    def write_file(self, path, content, append=False):
        if not isinstance(content, str):
            raise ProtocolError("workspace_content_must_be_text")
        if not isinstance(append, bool):
            raise ProtocolError("workspace_append_must_be_boolean")
        relative, candidate = _relative(self.root, path)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if candidate.is_symlink():
            raise ProtocolError("workspace_path_rejected")
        if append and candidate.is_file():
            try:
                content = candidate.read_text(encoding="utf-8") + content
            except (OSError, UnicodeError) as error:
                raise ProtocolError("workspace_file_unreadable") from error
        encoded = content.encode()
        if len(encoded) > MAX_FILE_BYTES:
            raise ProtocolError("workspace_content_too_large")
        try:
            candidate.write_text(content, encoding="utf-8")
        except OSError as error:
            raise SandboxUnknown("workspace_write_failed") from error
        self.attributed.append(relative)
        return {"path": relative, "bytes": len(encoded), "append": append}

    def edit_file(self, path, old_str, new_str):
        """Replace one exact match. Zero matches and repeated matches are errors."""
        if not isinstance(old_str, str) or not isinstance(new_str, str) or old_str == "":
            raise ProtocolError("edit_old_str_empty")
        relative, candidate = _relative(self.root, path)
        if candidate.is_symlink() or not candidate.is_file():
            raise ProtocolError("workspace_file_missing")
        text = self._read_text(candidate)
        count = text.count(old_str)
        if count == 0:
            raise ProtocolError("edit_old_str_not_found")
        if count > 1:
            raise ProtocolError("edit_old_str_not_unique")
        updated = text.replace(old_str, new_str, 1)
        encoded = updated.encode()
        if len(encoded) > MAX_FILE_BYTES:
            raise ProtocolError("workspace_content_too_large")
        try:
            candidate.write_text(updated, encoding="utf-8")
        except OSError as error:
            raise SandboxUnknown("workspace_write_failed") from error
        self.attributed.append(relative)
        return {"path": relative, "replacements": 1, "bytes": len(encoded)}

    def _search_skip(self, relative, candidate):
        suffix = Path(relative).suffix.lower()
        if suffix in _BINARY_SUFFIXES:
            return True
        try:
            size = candidate.stat().st_size
        except OSError:
            return True
        if size > MAX_SEARCH_FILE_BYTES:
            return True
        try:
            sample = candidate.open("rb").read(8192)
        except OSError:
            return True
        return b"\0" in sample

    @staticmethod
    def _test_path(relative):
        parts = relative.split("/")
        name = parts[-1]
        if name.startswith("test_") or name.endswith("_test.py") or name in {"tests.py", "conftest.py"}:
            return True
        return any(part in {"tests", "test", "testing"} for part in parts[:-1])

    def search(self, text):
        """Scan full file contents. Tests sort after other source. At most 100 hits."""
        if not isinstance(text, str) or not text or len(text) > MAX_SEARCH_CHARS:
            raise ProtocolError("workspace_search_text_rejected")
        matches = []
        root = self.root.resolve()
        for relative in self._file_paths():
            candidate = (root / relative).resolve()
            if self._search_skip(relative, candidate):
                continue
            try:
                content = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            for index, line in enumerate(content.splitlines(), start=1):
                if text not in line:
                    continue
                shown = line if len(line) <= MAX_SEARCH_LINE_CHARS else line[:MAX_SEARCH_LINE_CHARS]
                matches.append({"path": relative, "line": index, "text": shown,
                                "_test": self._test_path(relative)})
        matches.sort(key=lambda item: (item["_test"], item["path"], item["line"]))
        trimmed = []
        for item in matches[:MAX_SEARCH_HITS]:
            trimmed.append({"path": item["path"], "line": item["line"], "text": item["text"]})
        return trimmed

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


def _stderr_text(result):
    stderr = result.get("stderr") if isinstance(result, dict) else ""
    return stderr.lower() if isinstance(stderr, str) else ""


def container_name_in_argv(argv):
    """The ``--name`` value, when this docker argv sets one."""
    if not isinstance(argv, list) or "--name" not in argv:
        return None
    index = argv.index("--name")
    if index + 1 >= len(argv):
        return None
    name = argv[index + 1]
    if not isinstance(name, str) or not _CONTAINER_NAME.fullmatch(name):
        return None
    return name


def container_name_conflict(result):
    """True when Docker refused the run because the container name is taken.

    That refusal uses exit 125, the same code as a dead daemon. The message
    names the clash. It is not a daemon outage.
    """
    text = _stderr_text(result)
    if "is already in use" in text or "already in use by container" in text:
        return True
    if "conflict" in text and "container name" in text:
        return True
    return "you have to remove (or rename) that container" in text


def _daemon_failure(result):
    if container_name_conflict(result):
        return False
    text = _stderr_text(result)
    code = result.get("returncode") if isinstance(result, dict) else None
    if "cannot connect to the docker daemon" in text or "is the docker daemon running" in text:
        return True
    if "error during connect" in text:
        return True
    # A bare 125 is the historical daemon signal. A name clash is excluded above.
    return code == 125


def _docker_env():
    env = {"LANG": "C.UTF-8"}
    path = os.environ.get("PATH")
    if isinstance(path, str) and path:
        env["PATH"] = path
    return env


def docker_rm_f(name):
    """Force-remove one container. A missing name is not an error.

    ``docker run --rm`` does not run when the client is killed on timeout, so
    the named container would otherwise keep the name.
    """
    if not isinstance(name, str) or not _CONTAINER_NAME.fullmatch(name):
        return
    subprocess.run(
        ["docker", "rm", "-f", name], capture_output=True, check=False, timeout=30)


def run_docker_cli(argv, timeout_s, *, popen, remove):
    """Run a docker argv. On timeout, kill the client and ``remove`` the name.

    Offline tests pass ``popen`` and ``remove``. They do not start a daemon.
    """
    try:
        proc = popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_docker_env())
    except OSError:
        raise SandboxUnknown("docker_process_not_started") from None
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        try:
            stdout, stderr = proc.communicate()
        except Exception:
            stdout, stderr = b"", b""
        name = container_name_in_argv(argv)
        if name and remove is not None:
            try:
                remove(name)
            except Exception:
                pass
    code = 124 if timed_out else proc.returncode
    if type(code) is not int:
        code = 124
    return {
        "returncode": code,
        "stdout": stdout.decode(errors="replace") if isinstance(stdout, bytes) else "",
        "stderr": stderr.decode(errors="replace") if isinstance(stderr, bytes) else "",
        "timed_out": timed_out,
    }


def subprocess_docker_runner(argv, timeout_s):
    """Host docker CLI with an allowlisted environment. Not used by offline tests."""
    return run_docker_cli(argv, timeout_s, popen=subprocess.Popen, remove=docker_rm_f)


def _finish_docker_result(result, *, daemon_code):
    """Return a command result. A name clash is not a daemon failure."""
    if container_name_conflict(result):
        cleaned = known_sandbox_result(result)
        cleaned["failure"] = "container_name_conflict"
        return cleaned
    if _daemon_failure(result):
        raise SandboxUnknown(daemon_code)
    return known_sandbox_result(result)


def per_command_container_name(base):
    """A Docker name unique to one ``docker run``. The base stays recognizable.

    A timed-out command used to leave the unit's fixed name running. The next
    command then collided with it. Each command gets its own suffix.
    """
    if not isinstance(base, str) or not _CONTAINER_NAME.fullmatch(base):
        raise PilotStopped("invalid_container_name")
    suffix = uuid.uuid4().hex[:8]
    room = 63 - len(suffix) - 1
    stem = base[:room].rstrip(".-")
    if not stem or not stem[0].isalnum():
        stem = "alo"
    name = f"{stem}-{suffix}"
    if not _CONTAINER_NAME.fullmatch(name):
        raise PilotStopped("invalid_container_name")
    return name


# Image files are often mode 0444 and owned by root. Restoring the snapshot
# with tar then fails with "Permission denied" unless the tree is made
# writable first. The execute-only reference stays mode 111 and is not read.
_REVIEW_MERGE_SCRIPT = r"""
if [ -n "$(ls -A /agent 2>/dev/null)" ]; then
  find /workspace -path /workspace/executable -prune -o -exec chmod u+rwX {} +
  tar -C /agent --exclude=.git --exclude=executable -cf - . | tar -C /workspace -xf - || exit 2
fi
python3 -c 'import ctypes,os,sys
libc=ctypes.CDLL("libc.so.6", use_errno=True)
libc.mount.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_ulong,ctypes.c_char_p]
libc.mount.restype=ctypes.c_int
MS_BIND,MS_REC,MS_RDONLY,MS_REMOUNT=4096,16384,1,32
path=os.fsencode("/workspace")
if libc.mount(path, path, None, MS_BIND|MS_REC, None)!=0:
    sys.stderr.write("review_merge_tree_not_readonly\n"); sys.exit(2)
if libc.mount(path, path, None, MS_BIND|MS_REC|MS_REMOUNT|MS_RDONLY, None)!=0:
    sys.stderr.write("review_merge_tree_not_readonly\n"); sys.exit(2)
' || { echo review_merge_tree_not_readonly >&2; exit 2; }
exec setpriv --clear-groups --inh-caps -all --ambient-caps -all --bounding-set -all --reuid=__UID__ --regid=__GID__ -- "$@"
"""

# Official ProgramBench eval (commit 895f659) deletes ./executable, runs
# compile.sh, and then requires ./executable at the workspace root. The image
# reference occupies that path during the agent session. This script moves the
# reference aside without reading it, runs compile.sh as the host user, and
# copies the produced ./executable to a synced path for the black-box compare.
OFFICIAL_COMPILE_SCRIPT = r"""
set +e
if [ -n "$(ls -A /agent 2>/dev/null)" ]; then
  find /workspace -path /workspace/executable -prune -o -exec chmod u+rwX {} +
  tar -C /agent --exclude=.git --exclude=executable -cf - . | tar -C /workspace -xf -
fi
mkdir -p /opt/programbench-reference
if [ -e /workspace/executable ] && [ ! -L /workspace/executable ]; then
  mv /workspace/executable /opt/programbench-reference/executable || exit 1
  chmod 111 /opt/programbench-reference/executable
fi
setpriv --clear-groups --inh-caps -all --ambient-caps -all --bounding-set -all --reuid=__UID__ --regid=__GID__ -- sh -c 'cd /workspace && chmod +x ./compile.sh && ./compile.sh'
status=$?
if [ "$status" -eq 0 ] && [ -e /workspace/executable ] && [ ! -L /workspace/executable ] && [ -x /workspace/executable ]; then
  mkdir -p /workspace/build
  cp -f /workspace/executable /workspace/build/.alo-pb-candidate
  chown __UID__:__GID__ /workspace/build /workspace/build/.alo-pb-candidate 2>/dev/null || true
else
  if [ "$status" -eq 0 ]; then
    status=1
  fi
  rm -f /workspace/build/.alo-pb-candidate
fi
# Copy back as the host user so the workspace does not become root-owned.
setpriv --clear-groups --inh-caps -all --ambient-caps -all --bounding-set -all --reuid=__UID__ --regid=__GID__ -- sh -c 'tar -C /workspace --exclude=executable --exclude=.git -cf - . | tar -C /agent -xf -'
exit $status
"""

# A container's own exit status is infrastructure evidence. Record the actual
# program's exit separately so two launch errors cannot compare equal.
_PUBLIC_PROBE_DRIVER = r"""
import json, subprocess, sys, tempfile
with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
    child = subprocess.run(sys.argv[1:], stdout=stdout, stderr=stderr)
    stdout.seek(0)
    stderr.seek(0)
    out, err = stdout.read(262145), stderr.read(262145)
    if len(out) > 262144 or len(err) > 262144:
        raise SystemExit("public_probe_output_too_large")
    print(json.dumps(dict(returncode=child.returncode, stdout=out.decode(errors="replace"),
                          stderr=err.decode(errors="replace")), ensure_ascii=False))
"""


_OFFICIAL_CAPS = (
    "--cap-add=CHOWN", "--cap-add=DAC_OVERRIDE", "--cap-add=DAC_READ_SEARCH",
    "--cap-add=FOWNER", "--cap-add=SETGID", "--cap-add=SETUID", "--cap-add=SETPCAP",
)


def _readonly_merge_command(command):
    """Lock the populated merge tree before the review command runs.

    Setup may hold ``SYS_ADMIN`` long enough to remount ``/workspace``.
    The review command itself is exec'd through ``setpriv`` with every
    capability cleared and supplementary groups cleared. ``--regid`` is
    rejected unless a group option is present. If the remount fails, the
    command is not started.
    Nothing is copied back onto the snapshot.
    """
    command = list(command)
    if "--cap-drop=ALL" not in command or "sh" not in command:
        raise ProtocolError("review_merge_workspace_not_readonly")
    drop = command.index("--cap-drop=ALL")
    # SETPCAP is required to drop the bounding set. SETGID is required for
    # --clear-groups. DAC_OVERRIDE lets the restore rewrite mode-0444 image
    # files. All of these are dropped again by setpriv before the review command.
    for cap in ("--cap-add=CHOWN", "--cap-add=DAC_OVERRIDE", "--cap-add=DAC_READ_SEARCH",
                "--cap-add=FOWNER", "--cap-add=SETGID", "--cap-add=SETUID",
                "--cap-add=SETPCAP", "--cap-add=SYS_ADMIN"):
        if cap not in command:
            command.insert(drop + 1, cap)
    uid, gid = str(os.getuid()), str(os.getgid())
    if "--user" in command:
        index = command.index("--user")
        spec = command[index + 1]
        if not isinstance(spec, str) or ":" not in spec:
            raise ProtocolError("review_merge_workspace_not_readonly")
        uid, gid = spec.split(":", 1)
        # Setup must be root. The image USER is not root, and a dropped
        # capability set cannot rewrite root-owned files as that user.
        command[index + 1] = "0:0"
    shell = command.index("sh")
    if command[shell + 1] != "-c" or "tar -C /agent" not in command[shell + 2]:
        raise ProtocolError("review_merge_workspace_not_readonly")
    if "tar -C /workspace" not in command[shell + 2]:
        raise ProtocolError("review_merge_workspace_not_readonly")
    command[shell + 2] = _REVIEW_MERGE_SCRIPT.replace("__UID__", uid).replace("__GID__", gid)
    # Inbound populate uses `tar -C /workspace -xf`. The copy-back uses
    # `tar -C /workspace --exclude` and must not remain.
    if ("tar -C /workspace --exclude" in command[shell + 2]
            or "review_merge_tree_not_readonly" not in command[shell + 2]
            or "MS_RDONLY" not in command[shell + 2]):
        raise ProtocolError("review_merge_workspace_not_readonly")
    return command


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
                 extra_mounts=(), workdir="/workspace", mount_points=None, path=None,
                 merge_image_workspace=False, container_name=None):
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
        self.workdir = workdir if isinstance(workdir, str) and workdir.startswith("/") else None
        self.mount_points = tuple(mount_points) if mount_points is not None else ("/workspace",)
        self.path = path
        self.merge_image_workspace = bool(merge_image_workspace)
        if self.workdir is None or not self.mount_points:
            raise PilotStopped("invalid_sandbox_mount")
        if self.path is not None and (not isinstance(self.path, str) or not self.path or "\n" in self.path):
            raise PilotStopped("invalid_sandbox_path")
        if container_name is not None and (
                not isinstance(container_name, str) or not _CONTAINER_NAME.fullmatch(container_name)):
            raise PilotStopped("invalid_container_name")
        self.container_name = container_name
        if self.permit_execution and self.runner is not None:
            raise PilotStopped("one_sandbox_runner")
        for mount in (*self.extra_mounts, *self.mount_points):
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
        command = ["docker", "run", "--rm"]
        if self.container_name:
            command.extend(["--name", per_command_container_name(self.container_name)])
        command.extend([
            "--network=none",
            "--cap-drop=ALL",
            f"--pids-limit={self.pids_limit}",
            f"--memory={self.memory}",
        ])
        # The in-image /workspace must stay writable when an execute-only binary
        # cannot be copied out. Every other sandbox keeps a read-only root.
        if not self.merge_image_workspace:
            command.append("--read-only")
        command.extend([
            "--tmpfs", f"/tmp:rw,nosuid,nodev,size={self.tmpfs_size}",
            "--workdir", self.workdir,
        ])
        if self.merge_image_workspace:
            command.extend(["--mount", f"type=bind,src={self.workspace},dst=/agent"])
        else:
            for dest in self.mount_points:
                command.extend(["--mount", f"type=bind,src={self.workspace},dst={dest}"])
        for mount in self.extra_mounts:
            command.extend(["--mount", f"type=bind,src={self.workspace},dst={mount}"])
        command.extend([
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTHONUNBUFFERED=1",
            "--env", "PYTHONPATH=/workspace:/testbed",
            "--env", "LANG=C.UTF-8",
        ])
        if self.path:
            command.extend(["--env", "PATH=" + self.path])
        if self.merge_image_workspace:
            command.extend([
                image, "sh", "-c",
                # .git objects in the clean-room image are often mode 0444 and
                # owned by root. cp -a prints Permission denied for every one.
                # tar excludes .git and the reference executable in both directions.
                'if [ -n "$(ls -A /agent 2>/dev/null)" ]; then '
                'tar -C /agent --exclude=.git --exclude=executable -cf - . | tar -C /workspace -xf -; '
                'fi; '
                '"$@"; status=$?; '
                'tar -C /workspace --exclude=executable --exclude=.git -cf - . | tar -C /agent -xf -; '
                'exit $status',
                "alo-merge",
                *argv,
            ])
        else:
            command.extend([image, *argv])
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

    def readonly_argv(self, snapshot, argv):
        """Same container contract as ``docker_argv``, with the snapshot mounted read-only.

        The live workspace is not mounted. ``--cap-drop=ALL`` stays. A reviewer
        command therefore cannot write the implementation and then measure it.
        Merge mode normally copies ``/agent`` into a writable image ``/workspace``
        and copies edits back. Review populates that tree and remounts it
        read-only before the command, and it does not copy the result back.
        """
        command = self.docker_argv(argv)
        snapshot = Path(snapshot).resolve()
        live = str(self.workspace)
        rewritten = []
        for item in command:
            if item.startswith("type=bind,src=") and f"src={live}" in item:
                item = item.replace(f"src={live}", f"src={snapshot}", 1)
                if ",readonly" not in item:
                    item += ",readonly"
            rewritten.append(item)
        if self.merge_image_workspace:
            rewritten = _readonly_merge_command(rewritten)
        if not any(str(snapshot) in item and "readonly" in item for item in rewritten):
            raise ProtocolError("review_snapshot_not_mounted")
        if any(f"src={live}" in item for item in rewritten):
            raise ProtocolError("review_live_tree_still_mounted")
        return rewritten

    def readonly_command(self, snapshot, argv, timeout_s):
        if isinstance(argv, list) and argv and argv[0] in {"python", "py"}:
            argv = ["python3", *argv[1:]]
        command = self.readonly_argv(snapshot, argv)
        if self.runner is not None:
            result = self.runner(command, timeout_s)
        elif self.permit_execution:
            result = subprocess_docker_runner(command, timeout_s)
        else:
            raise PilotStopped("sandbox_not_configured")
        return _finish_docker_result(result, daemon_code="tool_sandbox_unknown")

    def command(self, argv, timeout_s):
        if isinstance(argv, list) and argv and argv[0] in {"python", "py"}:
            argv = ["python3", *argv[1:]]
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
        return _finish_docker_result(result, daemon_code="docker_daemon_unavailable")

    def public_build_command(self, detached_workspace, argv, timeout_s):
        """Build in a disposable mount that hides the entire image workspace."""
        root = Path(detached_workspace).resolve(strict=True)
        if root.is_relative_to(self.workspace) or self.workspace.is_relative_to(root):
            raise ProtocolError("public_build_requires_detached_workspace")
        box = DockerSandbox(
            root, image=self.resolved_image(), memory=self.memory, pids_limit=self.pids_limit,
            tmpfs_size=self.tmpfs_size, path=self.path, container_name=self.container_name or "alo-pb-build",
        )
        command = box.docker_argv(argv)
        # Use only a local task image. Keep tool caches writable without carrying
        # them into the submission or mounting any directory from the agent run.
        command[3:3] = ["--pull=never", "--env", "XDG_CACHE_HOME=/tmp/alo-cache"]
        return self._public_docker_result(command, timeout_s)

    def _public_docker_result(self, command, timeout_s):
        if self.runner is not None:
            runner = self.runner
        elif self.permit_execution:
            runner = subprocess_docker_runner
        else:
            raise IntegrationNotConfigured("docker_execution_not_permitted")
        result = runner(command, timeout_s)
        return _finish_docker_result(result, daemon_code="docker_daemon_unavailable")

    def public_probe_command(self, detached_workspace, argv, timeout_s):
        """Run a black-box command against a scratch tree and the local reference image."""
        root = Path(detached_workspace).resolve(strict=True)
        if root.is_relative_to(self.workspace) or self.workspace.is_relative_to(root):
            raise ProtocolError("public_probe_requires_detached_workspace")
        # Keep the image's execute-only reference at /workspace/executable.
        # Both processes run in /agent, a direct writable mount. This preserves
        # creations, deletions and renames without overlaying historical image
        # files or copying the reference bytes. The caller stages only a link.
        box = DockerSandbox(root, image=self.resolved_image(), memory=self.memory,
                            pids_limit=self.pids_limit, tmpfs_size=self.tmpfs_size, path=self.path,
                            container_name=self.container_name or "alo-pb-probe",
                            workdir="/agent", mount_points=("/agent",))
        command = box.docker_argv(["python3", "-I", "-c", _PUBLIC_PROBE_DRIVER, *argv])
        command.insert(3, "--pull=never")
        host = self._public_docker_result(command, timeout_s)
        if host.get("timed_out") or host.get("failure") or host["returncode"] != 0:
            raise SandboxUnknown("public_probe_execution_unknown")
        try:
            return known_sandbox_result(json.loads(host["stdout"]))
        except (TypeError, ValueError):
            raise SandboxUnknown("public_probe_missing_receipt") from None

    def official_compile_argv(self):
        """Docker argv that matches the official compile layout.

        Setup runs as root so it can move the execute-only reference. compile.sh
        itself runs as the host user. The produced ``./executable`` is copied to
        ``build/.alo-pb-candidate`` for a later black-box compare.
        """
        if not self.merge_image_workspace:
            raise PilotStopped("official_compile_requires_merge")
        command = self.docker_argv(["true"])
        if "--user" not in command or "--cap-drop=ALL" not in command or "sh" not in command:
            raise PilotStopped("official_compile_requires_merge")
        index = command.index("--user")
        spec = command[index + 1]
        if not isinstance(spec, str) or ":" not in spec:
            raise PilotStopped("official_compile_requires_merge")
        uid, gid = spec.split(":", 1)
        if not uid.isdigit() or not gid.isdigit():
            raise PilotStopped("official_compile_requires_merge")
        command[index + 1] = "0:0"
        drop = command.index("--cap-drop=ALL")
        for cap in _OFFICIAL_CAPS:
            if cap not in command:
                command.insert(drop + 1, cap)
        shell = command.index("sh")
        script = OFFICIAL_COMPILE_SCRIPT.replace("__UID__", uid).replace("__GID__", gid)
        command[shell:] = ["sh", "-c", script]
        return command

    def official_compile(self, timeout_s):
        """Run compile.sh the way ProgramBench eval does. Returns the status."""
        command = self.official_compile_argv()
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
        parsed = _finish_docker_result(result, daemon_code="docker_daemon_unavailable")
        if parsed.get("timed_out") or parsed.get("failure") == "container_name_conflict":
            return None
        code = parsed.get("returncode")
        if type(code) is not int or isinstance(code, bool):
            return None
        return code
