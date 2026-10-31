"""Turn sealed workspaces into per-unit harness inputs.

SWE-bench's loader keeps one prediction per ``instance_id``. Each unit therefore
gets its own predictions file and its own ``model_name_or_path`` (the unit id,
which already names the arm, phase, and order). ProgramBench submissions are
laid out as ``sealed/programbench/<unit_id>/<instance_id>/submission.tar.gz``
so each harness invocation sees a single instance directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile


def _git(argv):
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError("git_diff_failed")
    return completed.stdout


# ProgramBench's submission is the agent's codebase. The reference binary and
# the clean-room git metadata stay out of the tarball (protocol at 895f659).
SUBMISSION_EXCLUDE = frozenset({".git", "executable"})


def workspace_patch(workspace, diff_base, *, git=_git):
    """Source diff against the image HEAD the harness patches.

    ``git add -A`` skips gitignored build output. ``diff_base`` is that HEAD,
    which equals ``base_commit`` only when the image did not add a mode-only
    commit on top of it.
    """
    workspace = Path(workspace)
    if not (workspace / ".git").is_dir():
        return ""
    if not isinstance(diff_base, str) or not diff_base:
        return ""
    git(["git", "-C", str(workspace), "add", "-A"])
    return git(["git", "-C", str(workspace), "diff", "--cached", diff_base])


def prediction_row(instance_id, unit_id, patch):
    return {
        "instance_id": instance_id,
        "model_name_or_path": unit_id,
        "model_patch": patch if isinstance(patch, str) else "",
    }


def _load_records(run_dir):
    path = Path(run_dir) / "records.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and isinstance(row.get("unit_id"), str):
            rows.append(row)
    return rows


def _write_jsonl(path, rows):
    """Rewrite one live harness index from the current records.

    This is not an archive. ``archive-finished`` copies ``swe_predictions.jsonl``
    and ``manifest.jsonl`` aside first. Per-unit files are moved, not appended,
    because the harness reads a single prediction per unit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _pack_submission(workspace, dest_tar):
    workspace = Path(workspace)
    dest_tar.parent.mkdir(parents=True, exist_ok=True)
    files = []
    if workspace.is_dir():
        for path in workspace.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(workspace)
            if any(part in SUBMISSION_EXCLUDE for part in relative.parts):
                continue
            files.append(path)
    with tarfile.open(dest_tar, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(workspace)))
    return len(files)


def collect_sealed(run_dir, *, git=_git):
    """Rewrite harness inputs from ``records.jsonl`` and sealed workspaces."""
    run_dir = Path(run_dir)
    predictions = []
    manifest = []
    for record in _load_records(run_dir):
        unit_id = record["unit_id"]
        task_id = record.get("task_id")
        track = record.get("track")
        if not isinstance(task_id, str) or not isinstance(track, str):
            continue
        sealed_patch = run_dir / "checkpoints" / unit_id / "model.patch"
        workspace = run_dir / "checkpoints" / unit_id / "workspace"
        if not workspace.is_dir():
            workspace = run_dir / "units" / unit_id / "state" / "workspace"
        source = workspace
        entry = {
            "unit_id": unit_id,
            "instance_id": task_id,
            "track": track,
            "arm": record.get("arm"),
            "phase": record.get("phase"),
            "study": record.get("study"),
            "order_id": record.get("order_id") or "",
            "m2_condition": record.get("m2_condition") or "",
        }
        if track == "SWE-bench_Lite":
            patch = ""
            note = ""
            if sealed_patch.is_file():
                patch = sealed_patch.read_text(encoding="utf-8")
            else:
                try:
                    diff_base = record.get("image_head") or record.get("base_commit") or ""
                    patch = workspace_patch(source, diff_base, git=git)
                except RuntimeError:
                    note = "git_diff_failed"
            row = prediction_row(task_id, unit_id, patch)
            predictions.append(row)
            unit_path = run_dir / "sealed" / "swe_units" / f"{unit_id}.jsonl"
            _write_jsonl(unit_path, [row])
            entry["predictions_path"] = str(unit_path.relative_to(run_dir))
            entry["patch_bytes"] = len(patch.encode("utf-8"))
            if note:
                entry["note"] = note
            elif not patch:
                entry["note"] = "empty_patch"
        elif track == "ProgramBench":
            dest = run_dir / "sealed" / "programbench" / unit_id / task_id / "submission.tar.gz"
            if dest.is_file():
                entry["submission"] = str(dest.relative_to(run_dir))
                entry["files"] = None
            else:
                count = _pack_submission(source, dest)
                entry["submission"] = str(dest.relative_to(run_dir))
                entry["files"] = count
        manifest.append(entry)
    predictions_path = run_dir / "sealed" / "swe_predictions.jsonl"
    _write_jsonl(predictions_path, predictions)
    _write_jsonl(run_dir / "sealed" / "manifest.jsonl", manifest)
    return {"predictions": len(predictions), "units": len(manifest),
            "swe_predictions": str(predictions_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build per-unit SWE and ProgramBench harness inputs.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(collect_sealed(args.run_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
