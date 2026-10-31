"""A protocol epoch lets an archived unit run again without replaying its request ids."""
import json

from alo_cloud.freeze import select_units
from alo_cloud.ledger import current_epoch, main
from alo_cloud.run import _touched, open_run_ledger, run_schedule


UNIT = "m1.SWE-bench_Lite.scikit-learn__scikit-learn-13584.g1-main.B0"
PRIOR_CHARGE = 50_000


def _completion(url, body, headers):
    del url, headers
    return {
        "model": body["model"],
        "choices": [{"finish_reason": "stop", "message": {"content": '{"final":"ok"}'}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _seed_prior_restart(run):
    run.mkdir()
    ledger = open_run_ledger(run / "budget.sqlite3", "dry-run")
    digest = "a" * 64
    for suffix in ("r1-1", "r1-3", "r1-5"):
        request_id = f"req-{UNIT}-{suffix}"
        ledger.reserve(
            request_id, "mimo-v2.6-flash", PRIOR_CHARGE, payload_sha256=digest,
            tokens=20, seconds_ms=1000, unit_id=UNIT, unit_request_cap=12)
        ledger.settle_known(request_id, PRIOR_CHARGE, tokens=20, seconds_ms=1000)
    (run / "unit_restarts.json").write_text(json.dumps({UNIT: 1}), encoding="utf-8")
    (run / "records.jsonl").write_text("", encoding="utf-8")
    (run / "unknown_outcomes.jsonl").write_text(
        '{"unit_id":"old","reason":"provider_outcome_charged"}\n' * 2, encoding="utf-8")
    return ledger


def test_archive_with_no_records_starts_an_epoch_and_the_unit_can_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    run = tmp_path / "run"
    ledger = _seed_prior_restart(run)
    assert _touched(f"req-{UNIT}-r1-1", UNIT, 1, 0) is True
    assert _touched(f"req-{UNIT}-r1-1", UNIT, 0, 1) is False
    assert main([
        "archive-finished", "--run-dir", str(run),
        "--reason", "epoch after the archived nine units",
    ]) == 0
    assert current_epoch(run) == 1
    assert json.loads((run / "unit_restarts.json").read_text(encoding="utf-8")) == {}
    assert not (run / "unknown_outcomes.jsonl").exists()
    assert (run / "archive" / "epochs" / "0" / "unknown_outcomes.jsonl").is_file()
    # A second empty archive only advances the epoch. It does not need records.
    assert main([
        "new-epoch", "--run-dir", str(run), "--reason", "idempotent bump stays explicit",
    ]) == 0
    assert current_epoch(run) == 2
    # The box path is one bump. Put the epoch back to the first rerun generation.
    (run / "protocol_epoch.json").write_text(json.dumps({"epoch": 1}), encoding="utf-8")

    unit = select_units("m1", arms=("g1-main",), tasks=("scikit-learn__scikit-learn-13584",))[0]
    assert unit.unit_id == UNIT
    rows = run_schedule((unit,), run, live=False, sender=_completion)
    assert rows[0]["status"] == "completed"
    assert rows[0]["protocol_epoch"] == 1
    assert rows[0]["restart_count"] == 0
    ids = ledger.request_ids()
    fresh = [item for item in ids if item.startswith(f"req-e1-{UNIT}-")]
    assert fresh
    assert all(not item.startswith(f"req-e1-{UNIT}-r") for item in fresh)
    assert ledger.snapshot()["committed_or_reserved_microcny"]["mimo-v2.6-flash"] >= PRIOR_CHARGE * 3
    assert ledger.unit_usage(UNIT, protocol_epoch=1)["requests"] == len(fresh)
    assert ledger.unit_usage(UNIT, protocol_epoch=0, restart_count=1)["requests"] == 3
    assert ledger.snapshot()["unresolved_request_ids"] == []


def test_archive_finished_resets_restart_counts_for_archived_units(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text(json.dumps({"unit_id": UNIT, "status": "failed"}) + "\n", encoding="utf-8")
    (run / "checkpoints" / UNIT).mkdir(parents=True)
    (run / "checkpoints" / UNIT / "model.patch").write_text("", encoding="utf-8")
    (run / "unit_restarts.json").write_text(json.dumps({UNIT: 1, "other-unit": 1}), encoding="utf-8")
    assert main([
        "archive-finished", "--run-dir", str(run), "--reason", "protocol rerun",
    ]) == 0
    assert current_epoch(run) == 1
    assert json.loads((run / "unit_restarts.json").read_text(encoding="utf-8")) == {}
    assert (run / "archive" / "protocol-rerun" / "epoch-0" / UNIT / "checkpoint" / "model.patch").is_file()
    assert UNIT in (run / "archive" / "protocol-rerun" / "records.jsonl").read_text(encoding="utf-8")
    assert (run / "records.jsonl").read_text(encoding="utf-8") == ""


def test_archive_finished_appends_per_epoch(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    kept = run / "archive" / "protocol-rerun"
    (kept / "epoch-0").mkdir(parents=True)
    (kept / "epoch-0" / "records.jsonl").write_text('{"unit_id":"kept"}\n', encoding="utf-8")
    (kept / "records.jsonl").write_text('{"unit_id":"kept"}\n', encoding="utf-8")
    (run / "records.jsonl").write_text('{"unit_id":"added"}\n', encoding="utf-8")
    assert main(["archive-finished", "--run-dir", str(run), "--reason", "append"]) == 0
    epoch = (kept / "epoch-0" / "records.jsonl").read_text(encoding="utf-8")
    assert "kept" in epoch and "added" in epoch
    combined = (kept / "records.jsonl").read_text(encoding="utf-8")
    assert "kept" in combined and "added" in combined
    assert combined.index("kept") < combined.index("added")
    (run / "records.jsonl").write_text('{"unit_id":"later"}\n', encoding="utf-8")
    assert main(["archive-finished", "--run-dir", str(run), "--reason", "next"]) == 0
    assert "later" not in (kept / "epoch-0" / "records.jsonl").read_text(encoding="utf-8")
    assert "later" in (kept / "epoch-1" / "records.jsonl").read_text(encoding="utf-8")
    combined = (kept / "records.jsonl").read_text(encoding="utf-8")
    assert "kept" in combined and "added" in combined and "later" in combined


def test_empty_archive_from_epoch_three_parks_units_and_bumps(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "protocol_epoch.json").write_text(
        json.dumps({"epoch": 3, "reason": "epoch 3"}) + "\n", encoding="utf-8")
    (run / "records.jsonl").write_text("", encoding="utf-8")
    (run / "unit_restarts.json").write_text(json.dumps({UNIT: 1}), encoding="utf-8")
    (run / "unknown_outcomes.jsonl").write_text(
        '{"unit_id":"m1.x","reason":"provider_outcome_charged"}\n', encoding="utf-8")
    note = run / "units" / UNIT / "state"
    note.mkdir(parents=True)
    (note / "note.txt").write_text("partial", encoding="utf-8")
    (run / "checkpoints" / UNIT).mkdir(parents=True)
    (run / "checkpoints" / UNIT / "model.patch").write_text("partial-patch", encoding="utf-8")
    assert main([
        "archive-finished", "--run-dir", str(run),
        "--reason", "epoch 3 had no finished records",
    ]) == 0
    assert current_epoch(run) == 4
    assert json.loads((run / "unit_restarts.json").read_text(encoding="utf-8")) == {}
    assert not (run / "unknown_outcomes.jsonl").exists()
    assert not (run / "units").exists()
    assert not (run / "checkpoints").exists()
    assert (run / "records.jsonl").read_text(encoding="utf-8") == ""
    parked = run / "archive" / "protocol-rerun" / "epoch-3"
    assert (parked / "units" / UNIT / "state" / "note.txt").read_text(encoding="utf-8") == "partial"
    assert (parked / "checkpoints" / UNIT / "model.patch").read_text(encoding="utf-8") == "partial-patch"
    unknown = (run / "archive" / "epochs" / "3" / "unknown_outcomes.jsonl").read_text(encoding="utf-8")
    assert "provider_outcome_charged" in unknown
