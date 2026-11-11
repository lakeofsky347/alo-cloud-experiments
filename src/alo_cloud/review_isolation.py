"""Pre-review snapshot and a read-only command for the review role.

The DAG is sequential. This module does not claim concurrent writers. A review
role's ``command`` runs against a copy taken before that role starts, with the
copy and the implementation tree mounted read-only inside a user namespace.
Capabilities are dropped before the command starts, so the command cannot
remount either tree and then measure its own edit. Lightweight s2 does not use
this path as an independent reviewer; it is a zero-request public re-test of
the implementation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .errors import ProtocolError
from .policy import MUTATING_TOOLS


def role_is_review(role_tools):
    """True when the contract cannot edit. An unbound contract is not a review."""
    if role_tools is None:
        return False
    return not (set(role_tools) & MUTATING_TOOLS)


def snapshot_workspace(source, dest):
    """Copy ``source`` to ``dest``, including ``.git``. Symlinks stay links."""
    source = Path(source)
    dest = Path(dest)
    if dest.exists():
        raise ProtocolError("review_snapshot_dest_exists")
    shutil.copytree(source, dest, symlinks=True, dirs_exist_ok=False)
    return dest


def tree_digest(root):
    """Content digest of the worktree. ``.git`` is not part of the handoff bytes."""
    import hashlib

    root = Path(root)
    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name != ".git")
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix().encode()
            digest.update(rel)
            digest.update(b"\0")
            if path.is_symlink():
                digest.update(os.readlink(path).encode())
            elif path.is_file():
                digest.update(path.read_bytes())
    return digest.hexdigest()


_ISOLATION_SETUP = r"""
import ctypes, json, os, sys
spec = json.loads(sys.argv[1])
paths = [item for item in spec.get("paths") or [] if isinstance(item, str) and item]
command = spec.get("argv") or []
if not paths or not command or any(not isinstance(item, str) for item in command):
    sys.stderr.write("review_command_not_isolated\n")
    raise SystemExit(2)
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p]
libc.mount.restype = ctypes.c_int
MS_BIND, MS_REC, MS_RDONLY, MS_REMOUNT = 4096, 16384, 1, 32

def mount(path, flags):
    raw = os.fsencode(path)
    rc = libc.mount(raw, raw, None, flags, None)
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))

try:
    seen = []
    for path in paths:
        if path in seen:
            continue
        seen.append(path)
        mount(path, MS_BIND | MS_REC)
        mount(path, MS_BIND | MS_REMOUNT | MS_RDONLY)
    os.chdir(paths[0])
except OSError:
    sys.stderr.write("review_command_not_isolated\n")
    raise SystemExit(2)
os.execvp("setpriv", [
    "setpriv", "--inh-caps", "-all", "--ambient-caps", "-all", "--bounding-set", "-all",
    *command,
])
"""


def _review_env():
    env = {}
    for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE"):
        value = os.environ.get(key)
        if isinstance(value, str) and value:
            env[key] = value
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.setdefault("LANG", "C.UTF-8")
    return env


def run_readonly_host(snapshot, argv, timeout_s, protect_paths=()):
    """Run ``argv`` with cwd ``snapshot``. Writes to the snapshot or protected trees fail."""
    snapshot = Path(snapshot).resolve()
    if not snapshot.is_dir():
        raise ProtocolError("review_snapshot_missing")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or "\0" in item for item in argv):
        raise ProtocolError("sandbox_argv_rejected")
    paths = [str(snapshot)]
    for extra in protect_paths:
        if extra is None:
            continue
        extra = Path(extra).resolve()
        if extra.is_dir() and str(extra) not in paths:
            paths.append(str(extra))
    try:
        completed = subprocess.run(
            ["unshare", "--user", "--map-root-user", "--mount", "--",
             sys.executable, "-c", _ISOLATION_SETUP,
             json.dumps({"paths": paths, "argv": list(argv)})],
            cwd=snapshot, env=_review_env(), capture_output=True,
            timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "stdout": "", "stderr": "timeout", "timed_out": True}
    except OSError as error:
        raise ProtocolError("review_command_not_isolated") from error
    stderr = completed.stderr.decode(errors="replace")
    if completed.returncode == 2 and stderr.strip() == "review_command_not_isolated":
        raise ProtocolError("review_command_not_isolated")
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.decode(errors="replace"),
        "stderr": stderr,
        "timed_out": False,
    }


def run_review_command(services, argv, timeout_s):
    """Dispatch a review ``command`` away from the live implementation tree."""
    snapshot = getattr(services, "review_snapshot", None)
    if snapshot is None:
        raise ProtocolError("review_snapshot_missing")
    sandbox = getattr(services, "sandbox", None)
    readonly = getattr(sandbox, "readonly_command", None) if sandbox is not None else None
    ready = bool(getattr(sandbox, "execution_ready", False)) if sandbox is not None else False
    if callable(readonly) and ready:
        return readonly(snapshot, list(argv), timeout_s)
    live = getattr(getattr(services, "workspace", None), "root", None)
    return run_readonly_host(snapshot, list(argv), timeout_s, protect_paths=(live,))
