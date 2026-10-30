#!/usr/bin/env python3
"""Check tracked source files without network access or credential reads.

Reports file names and rule identifiers only, never the matching file content.
Ignored local environments are not published, so this check inspects Git's
index; inspect the staged diff before committing as a separate review step.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    ".gitattributes", ".gitignore", "AGENTS.md", "README.md", "LICENSE",
    "pyproject.toml", "requirements-cloud.txt", "docs", "src", "tests", "scripts",
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
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "api_key_shape": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "machine_absolute_path": re.compile(r"(?:/" + r"Users/[^\s/]+/|/private/" + r"tmp/)"),
    "historical_output_reference": re.compile(r"(?:output/(?:audit|live|freezes)/|experiments/" + r"results/)"),
}


def problems(name: str, root: Path = ROOT) -> list[str]:
    path = PurePosixPath(name)
    errors: list[str] = []
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return ["unsafe_path"]
    if path.parts[0] not in TOP_LEVEL:
        errors.append("unexpected_top_level")
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
    errors.extend(rule for rule, pattern in PATTERNS.items() if pattern.search(content))
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
