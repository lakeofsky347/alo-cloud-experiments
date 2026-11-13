"""Seed a unit workspace from the task image, then keep only a patch.

``docker create`` + ``docker cp`` run on the host against an image that is
already present. The sandbox that later mounts the copy uses ``--network=none``.
Tests inject ``docker`` and ``git`` and never start a daemon.
"""
from __future__ import annotations

import os
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
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    # docker cp keeps the image owner's uid. Git then refuses the copy as
    # dubious ownership. The config is per command, not a global safe.directory.
    if argv and argv[0] == "git":
        argv = ["git", "-c", "safe.directory=*", *argv[1:]]
    completed = subprocess.run(argv, check=False, capture_output=True, text=True, env=env)
    # ``git diff A B`` exits 0 when it prints a diff. Exit 1 is also a
    # successful diff for wrappers that pass ``--exit-code``.
    if completed.returncode == 0:
        return completed
    if completed.returncode == 1 and "diff" in argv:
        return completed
    raise PilotStopped("workspace_prepare_failed")


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


def docker_argv_deletes_images(argv):
    """True when a docker argv would remove or prune images.

    ``docker rm`` deletes a container. ``docker rmi``, ``docker image rm``,
    and any ``prune`` subcommand are refused. Local SWE-bench images stay.
    """
    if not isinstance(argv, list) or not argv:
        return False
    if argv[0] != "docker":
        return False
    tokens = [item for item in argv[1:] if isinstance(item, str) and not item.startswith("-")]
    if "rmi" in tokens or "prune" in tokens:
        return True
    if len(tokens) >= 2 and tokens[0] == "image" and tokens[1] in {"rm", "remove"}:
        return True
    return False


def _docker(argv):
    if docker_argv_deletes_images(argv):
        raise PilotStopped("docker_image_delete_refused")
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
    base = docker or _docker

    def run(argv):
        # Copy a local image out. Never pull, and never delete or prune images.
        if docker_argv_deletes_images(argv):
            raise PilotStopped("docker_image_delete_refused")
        return base(argv)

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


def _same_commit(left, right):
    left, right = (left or "").strip().lower(), (right or "").strip().lower()
    if not left or not right:
        return False
    return left == right or left.startswith(right) or right.startswith(left)


def _git_stdout(run, dest, *args, strict=False):
    try:
        completed = run(["git", "-C", str(dest), *args])
    except PilotStopped as error:
        if strict:
            raise PilotStopped("swe_image_git_failed") from error
        return None
    code = getattr(completed, "returncode", 0)
    diff_with_changes = code == 1 and bool(args) and args[0] == "diff"
    if code not in (0, None) and not diff_with_changes:
        if strict:
            raise PilotStopped("swe_image_git_failed")
        return None
    return getattr(completed, "stdout", "") or ""


def _mode_only_parent_diff(text_summary, text_numstat):
    """True when HEAD^..HEAD changes file modes only.

    ``git diff --summary`` lists mode changes. ``--numstat`` is ``0`` when the
    blob is unchanged, and ``-`` for a binary file whose blob is unchanged.
    Any added or removed line blocks. A binary mode change is still mode-only.
    """
    for line in (text_numstat or "").splitlines():
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) < 3 or columns[0] not in {"0", "-"} or columns[1] not in {"0", "-"}:
            return False
    for line in (text_summary or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("mode change "):
            return False
    return True


def resolve_image_head(dest, base_commit, *, git=None):
    """Commit the scored patch is taken against.

    Official SWE-bench images add one commit on top of ``base_commit`` whose
    subject is ``SWE-bench`` and whose only effect is ``100644`` to ``100755``.
    The harness applies patches to that HEAD. A content change in that commit
    is rejected (returns None).
    """
    if not _commit_ok(base_commit):
        return None
    run = git or subprocess_git
    head = (_git_stdout(run, dest, "rev-parse", "HEAD", strict=True) or "").strip()
    if not head:
        raise PilotStopped("swe_image_git_failed")
    if _same_commit(head, base_commit):
        return head
    parent = (_git_stdout(run, dest, "rev-parse", "HEAD^", strict=True) or "").strip()
    if not parent:
        raise PilotStopped("swe_image_git_failed")
    if not _same_commit(parent, base_commit):
        return None
    summary = _git_stdout(run, dest, "diff", "--summary", "HEAD^", "HEAD", strict=True)
    numstat = _git_stdout(run, dest, "diff", "--numstat", "HEAD^", "HEAD", strict=True)
    if summary is None or numstat is None:
        raise PilotStopped("swe_image_git_failed")
    if not _mode_only_parent_diff(summary, numstat):
        return None
    return head


def head_matches(dest, base_commit, *, git=None):
    """True when the image HEAD is ``base_commit`` or a mode-only child of it."""
    return resolve_image_head(dest, base_commit, git=git) is not None


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
