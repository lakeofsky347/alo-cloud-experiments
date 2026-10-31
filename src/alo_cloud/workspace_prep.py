"""Seed a unit workspace from the task image, then keep only a patch.

``docker create`` + ``docker cp`` run on the host against an image that is
already present. The sandbox that later mounts the copy uses ``--network=none``.
Tests inject ``docker`` and ``git`` and never start a daemon.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import uuid

from .errors import PilotStopped


_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_COMMIT = re.compile(r"[0-9a-fA-F]{7,64}")


def github_clone_url(repo):
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        raise PilotStopped("swe_repo_identity_rejected")
    return f"https://github.com/{repo}.git"


def _commit_ok(commit):
    return isinstance(commit, str) and bool(_COMMIT.fullmatch(commit))


def subprocess_git(argv):
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise PilotStopped("workspace_prepare_failed")
    return completed


def prepare_swe_checkout(dest, repo, base_commit, *, runner=None):
    """Fetch ``base_commit`` into ``dest`` without printing git stderr.

    The paid run does not use this. SWE workspaces are copied from the instance
    image's ``/testbed``, which already contains that commit and its built
    extensions. This helper remains for an explicit offline fixture.
    """
    if not _commit_ok(base_commit):
        raise PilotStopped("swe_base_commit_rejected")
    dest = Path(dest)
    if (dest / ".git").is_dir():
        return {"prepared": False, "reason": "checkout_already_present"}
    dest.mkdir(parents=True, exist_ok=True)
    url = github_clone_url(repo)
    run = runner or subprocess_git
    run(["git", "init", "-q", str(dest)])
    run(["git", "-C", str(dest), "remote", "add", "origin", url])
    run(["git", "-C", str(dest), "fetch", "--quiet", "origin", base_commit])
    run(["git", "-C", str(dest), "checkout", "--quiet", "FETCH_HEAD"])
    return {"prepared": True, "repo": repo, "base_commit": base_commit}


def _docker(argv):
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise PilotStopped("image_seed_failed")
    return completed


def seed_image_tree(image, container_path, dest, *, docker=None):
    """Copy ``container_path`` out of a local image. No pull and no model network."""
    if container_path not in {"/testbed", "/workspace"}:
        raise PilotStopped("image_seed_path_rejected")
    if not isinstance(image, str) or not image or any(ch.isspace() for ch in image) or image.startswith("-"):
        raise PilotStopped("invalid_sandbox_image")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    name = "alo-seed-" + uuid.uuid4().hex[:16]
    run = docker or _docker
    created = False
    try:
        run(["docker", "create", "--name", name, image])
        created = True
        run(["docker", "cp", f"{name}:{container_path}/.", str(dest)])
    finally:
        if created:
            try:
                run(["docker", "rm", "-f", name])
            except PilotStopped:
                pass
    return dest


def release_unreadable_executable(dest):
    """Drop a host copy of ProgramBench's reference binary when it cannot be read.

    The clean-room image ships ``executable`` as root-owned and execute-only.
    A bind mount of an unreadable copy hides the in-image binary (exit 127).
    Returning True means the sandbox must leave ``/workspace/executable`` in
    the image and merge the host tree on top of it.
    """
    exe = Path(dest) / "executable"
    if not exe.is_file() and not exe.exists():
        return False
    try:
        mode = exe.stat().st_mode
    except OSError:
        return True
    readable = bool(mode & 0o444)
    executable = bool(mode & 0o111)
    if readable and executable:
        return False
    if readable and not executable:
        try:
            exe.chmod(mode | 0o111)
        except OSError:
            exe.unlink(missing_ok=True)
            return True
        return False
    try:
        exe.unlink()
    except OSError:
        pass
    return True


def head_matches(dest, base_commit, *, git=None):
    """True when ``git rev-parse HEAD`` is the image's ``base_commit``."""
    if not _commit_ok(base_commit):
        return False
    run = git or subprocess_git
    completed = run(["git", "-C", str(dest), "rev-parse", "HEAD"])
    head = (getattr(completed, "stdout", "") or "").strip().lower()
    wanted = base_commit.strip().lower()
    if not head:
        return False
    return head == wanted or head.startswith(wanted) or wanted.startswith(head)


def apply_patch(dest, patch):
    """Replay a sealed source patch onto a freshly seeded tree."""
    if not isinstance(patch, str) or not patch.strip():
        return
    completed = subprocess.run(
        ["git", "-C", str(dest), "apply", "--whitespace=nowarn", "-"],
        input=patch, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise PilotStopped("patch_apply_failed")
