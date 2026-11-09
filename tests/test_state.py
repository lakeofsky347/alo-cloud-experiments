"""New synthetic SQLite state, no imported runs, files, model outputs or fixtures."""
import json
import sqlite3

import pytest

from alo.org.organization import Organization
from alo_cloud.state import export_learning_state, import_learning_state, probe_state_context, state_hash, StateContractError


def learning_state(tmp_path):
    path = tmp_path / "organization.sqlite3"
    org = Organization(str(path))
    cell = "type=generic|complexity=na|uncertainty=na"
    org.belief.update_objective("one", cell, True)
    org.belief.update_objective("one", cell, False)
    org.tms.record_outcome("one", ("coding",), False)
    identity = org.spawn_identity()
    org.tms.register_artifact("private-to-this-task", identity.id, "role", "one")
    org.conn.close()
    state = {"observations": {"one": [{"public_quality": 0.5, "tokens": 12}]},
             "worker_transcripts": {"role": ["private-to-this-task"]}, "rng_state": [1, 2]}
    return export_learning_state(path, state, allowed_models=("one", "two"))


def test_transfer_preserves_failures_but_not_transcript_identity_or_artifacts(tmp_path):
    value = learning_state(tmp_path)
    assert "private-to-this-task" not in json.dumps(value)
    imported = import_learning_state(tmp_path / "next.sqlite3", value, allowed_models=("one", "two"))
    assert set(imported) == {"observations"}
    org = Organization(str(tmp_path / "next.sqlite3"))
    assert org.belief.query("one", "type=generic|complexity=na|uncertainty=na").beta == 2
    assert org.conn.execute("SELECT failures FROM tms_outcomes").fetchone()[0] == 1
    assert org.conn.execute("SELECT count(*) FROM identities").fetchone()[0] == 0
    org.conn.close()


def test_reset_uses_common_prior_and_probe_cannot_write_back(tmp_path):
    value = learning_state(tmp_path)
    before = state_hash(value)
    assert import_learning_state(tmp_path / "reset.sqlite3", value, allowed_models=("one", "two"), mode="reset") == {"observations": {}}
    with probe_state_context(value, allowed_models=("one", "two")) as (conn, policy):
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE belief SET alpha=100")
        policy["observations"]["one"][0]["tokens"] = 999
    assert state_hash(value) == before


def test_transfer_rejects_extra_fields_before_creating_database(tmp_path):
    value = learning_state(tmp_path)
    value["source_code"] = "must not transfer"
    target = tmp_path / "rejected.sqlite3"
    with pytest.raises(StateContractError):
        import_learning_state(target, value, allowed_models=("one", "two"))
    assert not target.exists()


def test_topic_credit_transfers_and_role_id_does_not(tmp_path):
    path = tmp_path / "organization.sqlite3"
    org = Organization(str(path))
    org.conn.close()
    exported = export_learning_state(path, {"observations": {"one": [{
        "public_quality": 1.0, "credited_quality": 1.0,
        "topics": ["coding", "implementation"], "role_id": "s1", "requests": 1,
    }]}}, allowed_models=("one", "two"))
    row = exported["observations"]["one"][0]
    assert row["topics"] == ["coding", "implementation"]
    assert row["credited_quality"] == 1.0
    assert "role_id" not in row
    dropped = export_learning_state(path, {"observations": {"one": [{
        "credited_quality": 1.0, "topics": ["not-a-topic"], "requests": 1,
    }]}}, allowed_models=("one", "two"))
    bare = dropped["observations"]["one"][0]
    assert bare == {"requests": 1}
    assert "credited_quality" not in bare
