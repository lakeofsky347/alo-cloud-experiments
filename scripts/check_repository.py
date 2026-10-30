#!/usr/bin/env python3
"""Check tracked source files without network access or credential reads.

Reports file names and rule identifiers only, never the matching file content.
Ignored local environments are not published, so this check inspects Git's
index; inspect the staged diff before committing as a separate review step.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import json
import re
import subprocess
import sys
from urllib.parse import parse_qsl, urlsplit


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    ".gitattributes", ".gitignore", "AGENTS.md", "README.md", "LICENSE",
    "pyproject.toml", "requirements-cloud.txt", "docs", "src", "tests", "scripts", "study_records",
}
FORBIDDEN_PARTS = {
    ".cursor", ".venv", "venv", "__pycache__", ".pytest_cache", "output",
    "outputs", "results", "artifacts", "audit", "audits", "checkpoints", "logs",
    "tmp", "data", "datasets", "images", "models", "wheelhouse", "vendor",
}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".toml", ".json", ".yaml", ".yml"}
MODEL_INTERFACE_SOURCES = {
    "src/alo/models/__init__.py", "src/alo/models/base.py",
}
RECORD_DIRECTORY = "study_records/2026-10-28-m1-m2/"
ADMIN_METADATA_FILES = {"study_records/README.md"} | {
    RECORD_DIRECTORY + name for name in (
        "task_catalog.json", "selection_history.json", "m2_candidate_orders.json",
        "exposure_register.json", "storage_index.json", "source_index.json",
    )
}
LOCATOR_METADATA_FILES = {
    RECORD_DIRECTORY + "storage_index.json", RECORD_DIRECTORY + "source_index.json",
}
FORBIDDEN_RECORD_KEYS = {
    "api_key", "access_token", "refresh_token", "password", "private_key", "authorization",
    "messages", "payload", "prompt_text", "model_output", "model_response", "response_body",
    "text", "raw_content", "response_text", "reasoning_content",
    "source_rows", "source_rows_sha256", "patch", "test_patch", "gold_patch", "reference_patch",
    "test_content", "test_blob", "expected_output", "stdout", "stderr", "signed_url", "upload_url",
}
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "api_key_shape": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "machine_absolute_path": re.compile(r"(?:/" + r"Users/[^\s/]+/|/private/" + r"tmp/)"),
    "historical_output_reference": re.compile(r"(?:output/(?:audit|live|freezes)/|experiments/" + r"results/)"),
}


def record_content_problems(value) -> set[str]:
    """Administrative records may describe locations, never contain raw assets."""
    errors: set[str] = set()
    if isinstance(value, dict):
        if any(key.lower() in FORBIDDEN_RECORD_KEYS for key in value):
            errors.add("raw_private_record_field")
        for item in value.values():
            errors.update(record_content_problems(item))
    elif isinstance(value, list):
        for item in value:
            errors.update(record_content_problems(item))
    elif isinstance(value, str) and value.startswith(("http://", "https://")):
        parsed = urlsplit(value)
        keys = {key.lower() for key, _ in parse_qsl(parsed.query)}
        if parsed.username or parsed.password or keys & {
            "token", "access_token", "key", "signature", "sig", "resourcekey",
            "x-amz-signature", "x-goog-signature", "upload_id", "uploadid",
        }:
            errors.add("access_bearing_url")
    return errors


def problems(name: str, root: Path = ROOT) -> list[str]:
    path = PurePosixPath(name)
    errors: list[str] = []
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return ["unsafe_path"]
    if path.parts[0] not in TOP_LEVEL:
        errors.append("unexpected_top_level")
    if path.parts[0] == "study_records" and name not in ADMIN_METADATA_FILES:
        errors.append("undeclared_study_record")
    forbidden = FORBIDDEN_PARTS - {"models"} if name in MODEL_INTERFACE_SOURCES else FORBIDDEN_PARTS
    if any(part in forbidden for part in path.parts):
        errors.append("generated_or_external_asset_path")
    if path.name.startswith(".env") or path.name == "Dockerfile":
        errors.append("local_configuration_or_image_recipe")
    if len(path.parts) > 1 and path.suffix not in TEXT_SUFFIXES:
        errors.append("unexpected_file_type")
    source = root / name
    if source.is_symlink():
        return errors + ["symlink"]
    if not source.is_file():
        return errors + ["missing_regular_file"]
    raw = source.read_bytes()
    if len(raw) > 1024 * 1024:
        errors.append("file_exceeds_source_size_limit")
    if b"\0" in raw:
        errors.append("binary_content")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        return errors + ["not_utf8"]
    exempt_rules = {"machine_absolute_path", "historical_output_reference"} if name in LOCATOR_METADATA_FILES else set()
    errors.extend(rule for rule, pattern in PATTERNS.items() if rule not in exempt_rules and pattern.search(content))
    if name in ADMIN_METADATA_FILES and path.suffix == ".json":
        try:
            value = json.loads(content)
        except ValueError:
            errors.append("invalid_administrative_json")
        else:
            errors.extend(sorted(record_content_problems(value)))
    return errors


def main() -> int:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "-z"], cwd=ROOT, check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        print("Repository check requires an initialized Git index.", file=sys.stderr)
        return 2
    names = sorted(name for name in result.stdout.decode("utf-8").split("\0") if name)
    if not names:
        print("No tracked source files to check.", file=sys.stderr)
        return 2
    failed = False
    for name in names:
        for rule in problems(name):
            print(f"{name}: {rule}", file=sys.stderr)
            failed = True
    if failed:
        return 1
    print(f"Source repository check passed: {len(names)} tracked UTF-8 files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
