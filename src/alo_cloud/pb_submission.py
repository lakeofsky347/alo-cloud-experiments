"""The source snapshot shared by ProgramBench builds, review and evaluation."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile
import tempfile


# Names reserved by the controller or by offline language toolchains. Prune
# directories before walking them: probe fixtures can contain cycles or GiB files.
_PRIVATE_DIRS = frozenset({'.git', '.pb-probe', '.cache', '.gocache', '.gomodcache', '__pycache__', '.pytest_cache'})
_TOOL_OUTPUT_DIRS = {('build', 'gocache'), ('build', 'gomodcache'), ('target', 'debug'),
                    ('target', 'release'), ('target', '.fingerprint')}
_BLOCK = 1024 * 1024
_NATIVE_HEADERS = (b'\x7fELF', b'MZ', b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf')


def _manifest_digest(entries):
    data = json.dumps(entries, ensure_ascii=True, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(data.encode('ascii')).hexdigest()


@dataclass(frozen=True)
class SubmissionSnapshot:
    root: Path
    manifest: tuple
    sha256: str

    @property
    def files(self):
        return tuple(entry['path'] for entry in self.manifest)


def _raise_walk_error(error):
    raise error


def capture_submission(source, destination, *, check_deadline=None):
    """Materialize regular submission files in an empty directory outside the agent tree.

    Only executable permission bits affect the reproducible archive mode. The
    manifest hashes the exact bytes copied, including resources without a known
    source extension. Caller owns the destination and its lifetime.
    """
    source = Path(source).resolve(strict=True)
    root = Path(destination).resolve()
    if not source.is_dir() or root.is_relative_to(source):
        raise ValueError('invalid_submission_snapshot_directory')
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('submission_snapshot_destination_not_empty')
    entries = _submission_entries(source, root, check_deadline)
    return SubmissionSnapshot(root, tuple(entries), _manifest_digest(entries))


def submission_digest(source, *, check_deadline=None):
    """Hash the same selected input bytes as capture_submission without copying them."""
    return _manifest_digest(_submission_entries(Path(source).resolve(strict=True), None, check_deadline))


def collected_snapshot_dir(run_dir, arm, unit, attempt, task):
    for value in (arm, unit, attempt, task):
        if not isinstance(value, str) or not value or value in {'.', '..'} or Path(value).name != value:
            raise ValueError('invalid_submission_attempt_identity')
    return Path(run_dir) / 'sealed' / 'programbench-collected' / arm / unit / attempt / task


def submission_file_state(source, *, check_deadline=None):
    """Fingerprint source files for a tool receipt without copying their bytes."""
    entries = _submission_entries(Path(source).resolve(strict=True), None, check_deadline)
    return {entry['path']: entry for entry in entries}


def _submission_entries(source, root, check_deadline):
    from .pb_contract import can_be_probe_script

    entries = []
    possible_probes = []
    for directory, dirs, names in os.walk(source, onerror=_raise_walk_error):
        if check_deadline is not None:
            check_deadline()
        base = Path(directory)
        parts = base.relative_to(source).parts
        dirs[:] = sorted(name for name in dirs if name not in _PRIVATE_DIRS
                         and parts + (name,) not in _TOOL_OUTPUT_DIRS
                         and not (base / name).is_symlink())
        for name in sorted(names):
            if check_deadline is not None:
                check_deadline()
            path = base / name
            relative = path.relative_to(source)
            if relative.as_posix() == 'executable' or name == '.alo-pb-candidate':
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                continue
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as reader:
                current = os.fstat(reader.fileno())
                if not stat.S_ISREG(current.st_mode) or not os.path.samestat(info, current):
                    raise ValueError('submission_file_changed_during_capture')
                chunk = reader.read(_BLOCK)
                if parts[:1] == ('build',) and chunk.startswith(_NATIVE_HEADERS):
                    continue
                dest = None if root is None else root / relative
                if dest is not None:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                script = can_be_probe_script(path, chunk)
                mentions_reference = False
                tail = b''
                digest = hashlib.sha256()
                size = 0
                with (dest.open('xb') if dest is not None else nullcontext()) as writer:
                    while chunk:
                        if check_deadline is not None:
                            check_deadline()
                        if script and not mentions_reference:
                            mentions_reference = b'executable' in tail + chunk
                            tail = chunk[-9:]
                        digest.update(chunk)
                        if writer is not None:
                            writer.write(chunk)
                        size += len(chunk)
                        chunk = reader.read(_BLOCK)
                after = os.fstat(reader.fileno())
                if (current.st_size, current.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('submission_file_changed_during_capture')
            mode = 0o755 if current.st_mode & 0o111 else 0o644
            if dest is not None:
                dest.chmod(mode)
            entries.append(dict(path=relative.as_posix(), mode=mode, size=size, sha256=digest.hexdigest()))
            if mentions_reference and relative.as_posix() != 'compile.sh':
                possible_probes.append(relative.as_posix())
    if possible_probes:
        unused = _unused_reference_probes(root or source, source, entries, possible_probes, check_deadline)
        if root is not None:
            for name in unused:
                (root / name).unlink()
        entries = [row for row in entries if row['path'] not in unused]
    entries.sort(key=lambda entry: entry['path'])
    return entries


def _unused_reference_probes(tree, source, entries, possible_probes, check):
    from .pb_contract import (_hard_reference_kinds, _invokes_reference, reference_digest,
                              submission_build_graph)

    inputs = {row['path'] for row in entries}
    graph, complete = submission_build_graph(tree, allowed_inputs=inputs, probe_candidates=possible_probes,
                                             check=check)
    if not complete:
        return set()
    unused = set()
    digest = reference_digest(source)
    for name in possible_probes:
        if check is not None:
            check()
        if name in graph:
            continue
        try:
            text = (tree / name).read_text(encoding='utf-8')
        except UnicodeError:
            continue
        if digest and digest in text:
            continue
        if _invokes_reference(text, check) and not _hard_reference_kinds(text, check):
            unused.add(name)
    return unused



class _HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def pack_snapshot(root, dest_tar, manifest, expected_sha256):
    """Atomically publish the exact content of the checked snapshot, in manifest order."""
    if _manifest_digest(manifest) != expected_sha256:
        raise ValueError('submission_manifest_hash_mismatch')
    root = Path(root).resolve(strict=True)
    dest_tar = Path(dest_tar)
    dest_tar.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest_tar.parent, delete=False) as temp:
        name = temp.name
    try:
        with open(name, 'wb') as out, gzip.GzipFile(filename='', fileobj=out, mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as tar:
                for entry in manifest:
                    relative = Path(entry['path'])
                    if relative.is_absolute() or relative.as_posix() != entry['path']:
                        raise ValueError('submission_manifest_file_mismatch')
                    path = root / relative
                    if path.resolve() != path or not path.is_relative_to(root) or not path.is_file():
                        raise ValueError('submission_manifest_file_mismatch')
                    with path.open('rb') as stream:
                        info = os.fstat(stream.fileno())
                        mode = 0o755 if info.st_mode & 0o111 else 0o644
                        if info.st_size != entry['size'] or mode != entry['mode']:
                            raise ValueError('submission_manifest_file_mismatch')
                        member = tarfile.TarInfo(entry['path'])
                        member.size = info.st_size
                        member.mode = mode
                        reader = _HashReader(stream)
                        tar.addfile(member, reader)
                        if reader.digest.hexdigest() != entry['sha256']:
                            raise ValueError('submission_manifest_file_mismatch')
        os.replace(name, dest_tar)
    finally:
        Path(name).unlink(missing_ok=True)
    return len(manifest)


def scan_snapshot(snapshot, source, seconds=60):
    """Scan exactly the submitted files; controller-only reference metadata stays outside."""
    from .pb_contract import (REFERENCE_BASENAME, reference_digest, reference_is_execute_only,
                              submission_reference_scan_bounded)

    source = Path(source)
    ref = source / REFERENCE_BASENAME
    result = submission_reference_scan_bounded(
        snapshot.root, seconds, reference_sha256=reference_digest(source), snapshot_manifest=snapshot.manifest,
    )
    if ref.is_file() and not ref.is_symlink():
        result['reference_execute_only'] = reference_is_execute_only(ref.stat().st_mode)
        if not result['reference_execute_only']:
            result['violations'].append({'kind': 'reference_not_execute_only', 'path': REFERENCE_BASENAME})
            result['ok'] = False
    return result
