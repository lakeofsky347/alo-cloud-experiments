"""Canonical restore of sealed native files without audit SQLite sidecars."""
import hashlib
from pathlib import Path, PurePosixPath

from .errors import PilotStopped


def verify_native_state(directory, declared):
    directory = Path(directory)
    if not isinstance(declared, dict):
        raise PilotStopped("native_hash_manifest_invalid")
    actual = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise PilotStopped("native_state_symlink_rejected")
        if path.is_file():
            actual[str(path.relative_to(directory))] = path
    for name, digest in declared.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name not in actual:
            raise PilotStopped("native_declared_file_missing_or_unsafe")
        if hashlib.sha256(actual[name].read_bytes()).hexdigest() != digest:
            raise PilotStopped("native_declared_file_changed")
    extras = {}
    for name in actual.keys() - declared.keys():
        suffix = "-wal" if name.endswith("-wal") else "-shm" if name.endswith("-shm") else None
        base = name[:-len(suffix)] if suffix else None
        if suffix is None or base not in declared:
            raise PilotStopped("unexpected_native_state_file")
        if not actual[base].read_bytes().startswith(b"SQLite format 3\0"):
            raise PilotStopped("sidecar_requires_declared_sqlite_database")
        wal = directory / (base + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise PilotStopped("nonempty_unsealed_sqlite_wal_rejected")
        content = actual[name].read_bytes()
        extras[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                        "kind": "empty_wal" if suffix == "-wal" else "regenerable_sqlite_index"}
    return {"declared_sha256": dict(declared), "extra_audit_sidecars": extras,
            "full_directory_match": not extras, "canonical_restore_ignores_sidecars": bool(extras)}


def restore_native_state(source, destination, declared):
    source, destination = Path(source), Path(destination)
    receipt = verify_native_state(source, declared)
    destination.mkdir(parents=True, exist_ok=False)
    for name, digest in declared.items():
        content = (source / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise PilotStopped("native_source_changed_during_restore")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(content)
    verify_native_state(destination, declared)
    return receipt
