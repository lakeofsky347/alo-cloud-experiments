"""Independent scoring of sealed artifacts.

This module does not call a model and does not read provider credentials.
SWE-bench Lite is scored only by invoking the official swebench 4.1.0 harness.
ProgramBench is scored only by the ``programbench eval`` command from the
pinned source commit. Missing harnesses, Docker, or a mismatched checkout
are reported as unavailable. They are not stored as a score of zero.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


SWEBENCH_PACKAGE = "4.1.0"
SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Lite"
SWEBENCH_REVISION = "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"
PROGRAMBENCH_COMMIT = "895f65916bc5ba34a915d36da321fde0301125ad"
PROGRAMBENCH_TESTS_REVISION = "de0ddfb637590c7ecb54fa0b5301f6dc7dfbcee5"


def swebench_command(predictions_path, *, run_id, instance_ids, split):
    command = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", SWEBENCH_DATASET,
        "--split", split,
        "--predictions_path", str(predictions_path),
        "--max_workers", "1",
        "--run_id", run_id,
    ]
    if instance_ids:
        command.append("--instance_ids")
        command.extend(instance_ids)
    return command


def programbench_command(repo, submissions_dir):
    return ["uv", "run", "--project", str(repo), "programbench", "eval", str(submissions_dir)]


def _git_head(repo):
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def score_sealed(run_dir, *, execute=False, swe_predictions=None, programbench_repo=None,
                 docker_bin="docker"):
    """Read sealed files and, only with ``execute``, invoke the official harnesses.

    The returned object has no numeric score when a harness did not run.
    """
    run_dir = Path(run_dir)
    report = {
        "schema": "alo-independent-evaluation-v1",
        "swebench": {
            "package": SWEBENCH_PACKAGE,
            "dataset": SWEBENCH_DATASET,
            "revision": SWEBENCH_REVISION,
            "status": "not_run",
        },
        "programbench": {
            "source_commit": PROGRAMBENCH_COMMIT,
            "tests_revision": PROGRAMBENCH_TESTS_REVISION,
            "status": "not_run",
        },
    }
    if not execute:
        report["swebench"]["status"] = "not_executed"
        report["programbench"]["status"] = "not_executed"
        report["swebench"]["command"] = swebench_command(
            swe_predictions or (run_dir / "sealed" / "swe_predictions.jsonl"),
            run_id="alo-m1m2", instance_ids=(), split="test")
        if programbench_repo:
            report["programbench"]["command"] = programbench_command(
                programbench_repo, run_dir / "sealed" / "programbench")
        return report
    if shutil.which(docker_bin) is None:
        report["swebench"]["status"] = "unavailable"
        report["programbench"]["status"] = "unavailable"
        report["reason"] = "docker_cli_missing"
        return report
    report["swebench"] = _score_swebench(run_dir, swe_predictions)
    report["programbench"] = _score_programbench(run_dir, programbench_repo)
    return report


def _score_swebench(run_dir, predictions):
    result = {
        "package": SWEBENCH_PACKAGE,
        "dataset": SWEBENCH_DATASET,
        "revision": SWEBENCH_REVISION,
        "status": "unavailable",
    }
    try:
        import importlib.metadata as metadata
        version = metadata.version("swebench")
    except metadata.PackageNotFoundError:
        result["reason"] = "swebench_package_missing"
        return result
    if version != SWEBENCH_PACKAGE:
        result["reason"] = "swebench_version_mismatch"
        result["observed_version"] = version
        return result
    path = Path(predictions) if predictions else Path(run_dir) / "sealed" / "swe_predictions.jsonl"
    if not path.is_file():
        result["reason"] = "swe_predictions_missing"
        return result
    command = swebench_command(path, run_id="alo-m1m2", instance_ids=(), split="test")
    result["command"] = command
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    result["returncode"] = completed.returncode
    result["status"] = "finished" if completed.returncode == 0 else "harness_failed"
    # Keep harness stdout out of this report: it can echo task logs. The harness
    # writes its own report directory beside the predictions file.
    return result


def _score_programbench(run_dir, repo):
    result = {
        "source_commit": PROGRAMBENCH_COMMIT,
        "tests_revision": PROGRAMBENCH_TESTS_REVISION,
        "status": "unavailable",
    }
    if not repo:
        result["reason"] = "programbench_checkout_required"
        return result
    head = _git_head(repo)
    result["observed_commit"] = head
    if head != PROGRAMBENCH_COMMIT:
        result["reason"] = "programbench_commit_mismatch"
        return result
    submissions = Path(run_dir) / "sealed" / "programbench"
    if not submissions.is_dir():
        result["reason"] = "programbench_submissions_missing"
        return result
    command = programbench_command(repo, submissions)
    result["command"] = command
    if shutil.which("uv") is None:
        result["reason"] = "uv_missing"
        result["status"] = "unavailable"
        return result
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    result["returncode"] = completed.returncode
    result["status"] = "finished" if completed.returncode == 0 else "harness_failed"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score sealed artifacts without calling a model.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--swe-predictions", default=None)
    parser.add_argument("--programbench-repo", default=os.environ.get("PROGRAMBENCH_REPO") or None)
    args = parser.parse_args(argv)
    report = score_sealed(
        args.run_dir, execute=args.execute, swe_predictions=args.swe_predictions,
        programbench_repo=args.programbench_repo)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    failed = [report["swebench"]["status"], report["programbench"]["status"]]
    if args.execute and any(status not in {"finished", "not_run"} for status in failed):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
