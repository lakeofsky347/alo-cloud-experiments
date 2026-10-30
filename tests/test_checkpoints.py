"""Canonical native-file restore from fresh local files, no existing checkpoints."""
import hashlib
import sqlite3

import pytest

from alo_cloud.checkpoints import restore_native_state, verify_native_state
from alo_cloud.errors import PilotStopped


def test_declared_state_restores_without_empty_sqlite_sidecars(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    database = source / "organization.sqlite3"
    conn = sqlite3.connect(database)
    conn.execute("CREATE TABLE sample(value INTEGER)")
    conn.close()
    declared = {database.name: hashlib.sha256(database.read_bytes()).hexdigest()}
    (source / "organization.sqlite3-wal").write_bytes(b"")
    result = restore_native_state(source, tmp_path / "restored", declared)
    assert result["canonical_restore_ignores_sidecars"] is True
    assert sorted(p.name for p in (tmp_path / "restored").iterdir()) == [database.name]
    verify_native_state(tmp_path / "restored", declared)


def test_corrupt_parent_cannot_be_restored_or_overwritten(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "state.txt"
    path.write_text("original")
    declared = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    path.write_text("changed")
    with pytest.raises(PilotStopped, match="declared_file_changed"):
        restore_native_state(source, tmp_path / "child", declared)
    assert not (tmp_path / "child").exists()
