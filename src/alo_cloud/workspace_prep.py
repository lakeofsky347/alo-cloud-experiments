"""Host-side git checkout for SWE workspaces.

The clone and fetch run on the host, where the network is allowed. The
sandbox that later mounts this directory is started with ``--network=none``.
Tests inject ``runner`` and never call the real git binary.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

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
    """Fetch ``base_commit`` into ``dest`` without printing git stderr."""
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
