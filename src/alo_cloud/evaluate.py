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

# Child process: pin datasets.load_dataset, then run the official CLI.
# argv is [revision, *official_flags]. The official parser has no revision flag.
_SWEBENCH_PINNED_LAUNCHER = r"""
import runpy
import sys
revision = sys.argv[1]
sys.argv = ["swebench.harness.run_evaluation", *sys.argv[2:]]
import swebench.harness.utils as utils
_load_dataset = utils.load_dataset

def _pinned(name, *args, **kwargs):
    kwargs["revision"] = revision
    return _load_dataset(name, *args, **kwargs)

utils.load_dataset = _pinned
runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")
"""


def swebench_command(predictions_path, *, run_id, instance_ids, split, dataset_revision=None):
    """Official harness argv. ``--dataset_revision`` is the pinned Lite revision.

    swebench 4.1.0's CLI does not accept a revision flag. ``execute_swebench``
    removes this argument and passes it to ``datasets.load_dataset`` before
    calling ``swebench.harness.run_evaluation.main``.
    """
    revision = dataset_revision or SWEBENCH_REVISION
    command = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", SWEBENCH_DATASET,
        "--dataset_revision", revision,
        "--split", split,
        "--predictions_path", str(predictions_path),
        "--max_workers", "1",
        "--run_id", run_id,
        "--namespace", "swebench",
    ]
    if instance_ids:
        command.append("--instance_ids")
        command.extend(instance_ids)
    return command


def programbench_command(repo, submissions_dir):
    return ["uv", "run", "--project", str(repo), "programbench", "eval", str(submissions_dir)]


def programbench_env(base=None):
    """Pin the test-asset revision in the child environment. Do not inherit provider keys."""
    env = dict(os.environ if base is None else base)
    env["PROGRAMBENCH_HF_REVISION"] = PROGRAMBENCH_TESTS_REVISION
    for name in ("DEEPSEEK_API_KEY", "MIMO_API_KEY", "ZAI_API_KEY"):
        env.pop(name, None)
    return env


def _read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _run_id_for(unit_id):
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in unit_id)
    return ("u" + safe)[:100]


def swebench_unit_commands(run_dir, predictions_path=None):
    """One official-harness command per prediction row (one unit, one instance)."""
    path = Path(predictions_path) if predictions_path else Path(run_dir) / "sealed" / "swe_predictions.jsonl"
    if not path.is_file():
        return None, path
    commands = []
    for row in _read_jsonl(path):
        instance_id = row.get("instance_id")
        unit_id = row.get("model_name_or_path")
        if not isinstance(instance_id, str) or not isinstance(unit_id, str):
            continue
        unit_file = Path(run_dir) / "sealed" / "swe_units" / f"{unit_id}.jsonl"
        commands.append({
            "unit_id": unit_id,
            "instance_id": instance_id,
            "command": swebench_command(
                unit_file if unit_file.is_file() else path,
                run_id=_run_id_for(unit_id),
                instance_ids=(instance_id,),
                split="test",
            ),
        })
    return commands, path


def programbench_unit_commands(run_dir, repo):
    root = Path(run_dir) / "sealed" / "programbench"
    if not root.is_dir():
        return None
    commands = []
    for unit_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        tarballs = list(unit_dir.glob("*/submission.tar.gz"))
        if not tarballs:
            continue
        commands.append({
            "unit_id": unit_dir.name,
            "instance_id": tarballs[0].parent.name,
            "command": programbench_command(repo, unit_dir) if repo else [],
            "env_revision": PROGRAMBENCH_TESTS_REVISION,
        })
    return commands


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
        swe_commands, swe_path = swebench_unit_commands(run_dir, swe_predictions)
        program_commands = programbench_unit_commands(run_dir, programbench_repo)
        if swe_commands is None:
            report["swebench"]["status"] = "swe_predictions_missing"
            report["swebench"]["command"] = swebench_command(
                swe_path, run_id="alo-m1m2", instance_ids=(), split="test")
        else:
            report["swebench"]["status"] = "not_executed"
            report["swebench"]["units"] = [
                {"unit_id": item["unit_id"], "instance_id": item["instance_id"], "command": item["command"]}
                for item in swe_commands
            ]
        if program_commands is None:
            report["programbench"]["status"] = "programbench_submissions_missing"
        else:
            report["programbench"]["status"] = "not_executed"
            report["programbench"]["units"] = program_commands
            report["programbench"]["hf_revision"] = PROGRAMBENCH_TESTS_REVISION
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
    commands, path = swebench_unit_commands(run_dir, predictions)
    if commands is None:
        result["reason"] = "swe_predictions_missing"
        return result
    if not commands:
        result["reason"] = "swe_predictions_empty"
        result["status"] = "unavailable"
        return result
    result["units"] = []
    failed = False
    for item in commands:
        unit_result = execute_swebench(item["command"])
        unit_result["unit_id"] = item["unit_id"]
        unit_result["instance_id"] = item["instance_id"]
        result["units"].append(unit_result)
        if unit_result.get("status") != "finished":
            failed = True
    result["status"] = "finished" if not failed else "harness_failed"
    result["predictions"] = str(path)
    return result


def execute_swebench(command):
    """Run one unit through the official harness with the dataset revision pinned."""
    argv = list(command)
    if "--dataset_revision" not in argv:
        return {"status": "unavailable", "reason": "dataset_revision_missing", "command": command}
    index = argv.index("--dataset_revision")
    revision = argv[index + 1]
    official = argv[:index] + argv[index + 2:]
    if revision != SWEBENCH_REVISION:
        return {"status": "unavailable", "reason": "dataset_revision_mismatch", "command": command}
    module_at = official.index("-m")
    official_flags = official[module_at + 2:]
    completed = subprocess.run(
        [sys.executable, "-c", _SWEBENCH_PINNED_LAUNCHER, revision, *official_flags],
        check=False, capture_output=True, text=True,
        env=programbench_env(),
    )
    return {
        "status": "finished" if completed.returncode == 0 else "harness_failed",
        "returncode": completed.returncode,
        "command": command,
        "dataset_revision": SWEBENCH_REVISION,
    }


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
    commands = programbench_unit_commands(run_dir, repo)
    if commands is None:
        result["reason"] = "programbench_submissions_missing"
        return result
    if not commands:
        result["reason"] = "programbench_submissions_missing"
        result["status"] = "unavailable"
        return result
    result["hf_revision"] = PROGRAMBENCH_TESTS_REVISION
    if shutil.which("uv") is None:
        result["reason"] = "uv_missing"
        result["units"] = commands
        return result
    result["units"] = []
    failed = False
    env = programbench_env()
    for item in commands:
        completed = subprocess.run(item["command"], check=False, capture_output=True, text=True, env=env)
        item = dict(item)
        item["returncode"] = completed.returncode
        item["status"] = "finished" if completed.returncode == 0 else "harness_failed"
        result["units"].append(item)
        if item["status"] != "finished":
            failed = True
    result["status"] = "finished" if not failed else "harness_failed"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score sealed artifacts without calling a model.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--swe-predictions", default=None)
    parser.add_argument("--programbench-repo", default=os.environ.get("PROGRAMBENCH_REPO") or None)
    parser.add_argument("--dataset-revision", default=SWEBENCH_REVISION)
    parser.add_argument("--programbench-hf-revision", default=PROGRAMBENCH_TESTS_REVISION)
    args = parser.parse_args(argv)
    if args.dataset_revision != SWEBENCH_REVISION or args.programbench_hf_revision != PROGRAMBENCH_TESTS_REVISION:
        print("revision does not match the frozen evaluation pins")
        return 2
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
