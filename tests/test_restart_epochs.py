"""Restart archives are per epoch. A later epoch must not move into an old directory."""
import json

import pytest

from alo_cloud.errors import PilotStopped
from alo_cloud.freeze import select_units
from alo_cloud.ledger import current_epoch, main, restart_unit
from alo_cloud.run import open_run_ledger, run_schedule


STRONG = "m1.SWE-bench_Lite.scikit-learn__scikit-learn-13584.g1-strong.B0"
MAIN = "m1.SWE-bench_Lite.scikit-learn__scikit-learn-13584.g1-main.B0"


def _ok(url, body, headers):
    del url, headers
    return {
        "model": body["model"],
        "choices": [{"finish_reason": "stop", "message": {"content": '{"final":"ok"}'}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _b0(*arms):
    units = select_units("m1", arms=arms, tasks=("scikit-learn__scikit-learn-13584",))
    return tuple(unit for unit in units if unit.phase == "B0")


def test_occupied_restart_dir_gets_a_sibling_and_appends_nothing_over(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    ledger = open_run_ledger(run / "budget.sqlite3", "dry-run")
    (run / "records.jsonl").write_text(
        json.dumps({"unit_id": STRONG, "status": "stopped"}) + "\n"
        + json.dumps({"unit_id": MAIN, "status": "completed"}) + "\n",
        encoding="utf-8")
    (run / "checkpoints" / STRONG).mkdir(parents=True)
    (run / "checkpoints" / STRONG / "model.patch").write_text("partial", encoding="utf-8")
    (run / "sealed" / "programbench" / STRONG).mkdir(parents=True)
    (run / "sealed" / "programbench" / STRONG / "keep.txt").write_text("tar", encoding="utf-8")
    legacy = run / "archive" / "restarts" / STRONG / "1" / "checkpoint" / STRONG
    legacy.mkdir(parents=True)
    (legacy / "old.txt").write_text("keep", encoding="utf-8")
    occupied = run / "archive" / "restarts" / "epoch-0" / STRONG / "1"
    (occupied / "checkpoint" / STRONG).mkdir(parents=True)
    (occupied / "checkpoint" / STRONG / "old-epoch.txt").write_text("keep2", encoding="utf-8")
    (occupied / "records.jsonl").write_text('{"unit_id":"earlier"}\n', encoding="utf-8")
    result = restart_unit(run, STRONG, reason="charged timeout", ledger=ledger)
    assert "epoch-0" in result["archive"]
    assert result["archive"].endswith("1-dup2")
    assert (legacy / "old.txt").read_text(encoding="utf-8") == "keep"
    assert (occupied / "records.jsonl").read_text(encoding="utf-8") == '{"unit_id":"earlier"}\n'
    archived = (run / "archive" / "restarts" / "epoch-0" / STRONG / "1-dup2" / "records.jsonl"
                ).read_text(encoding="utf-8")
    assert STRONG in archived and "earlier" not in archived
    assert MAIN in (run / "records.jsonl").read_text(encoding="utf-8")
    assert STRONG not in (run / "records.jsonl").read_text(encoding="utf-8")
    assert (run / "archive" / "restarts" / "epoch-0" / STRONG / "1-dup2" / "checkpoint"
            / "model.patch").read_text(encoding="utf-8") == "partial"
    assert (run / "archive" / "restarts" / "epoch-0" / STRONG / "1-dup2" / "programbench"
            / "keep.txt").is_file()
    assert json.loads((run / "unit_restarts.json").read_text(encoding="utf-8"))[STRONG] == 1


def test_epoch_restart_then_archive_then_later_epoch_resume_and_m2(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    run = tmp_path / "run"
    b0 = _b0("g1-main", "g1-strong")
    assert [unit.unit_id for unit in b0] == [MAIN, STRONG]
    steps = iter(["ok", "timeout", "ok", "ok", "timeout", "ok", "ok"])

    def sender(url, body, headers):
        step = next(steps)
        if step == "timeout":
            raise TimeoutError("provider_wall_clock")
        return _ok(url, body, headers)

    first = run_schedule(b0, run, live=False, sender=sender)
    assert [row["unit_id"] for row in first] == [MAIN, STRONG]
    assert first[0]["status"] == "completed" and first[0]["restart_count"] == 0
    assert first[1]["status"] == "completed" and first[1]["restart_count"] == 1
    assert first[1]["halt_schedule"] is False
    epoch0 = run / "archive" / "restarts" / "epoch-0" / STRONG / "1"
    assert (epoch0 / "records.jsonl").is_file()
    assert "provider_deadline_charged" in (epoch0 / "records.jsonl").read_text(encoding="utf-8")
    legacy = run / "archive" / "restarts" / STRONG / "1" / "checkpoint" / STRONG
    legacy.mkdir(parents=True)
    (legacy / "old.txt").write_text("keep", encoding="utf-8")
    (run / "sealed" / "swe_predictions.jsonl").write_text('{"unit_id":"m1"}\n', encoding="utf-8")
    (run / "unknown_outcomes.jsonl").write_text('{"unit_id":"old","reason":"x"}\n', encoding="utf-8")
    # A previous bump already left an unknown file for this epoch number.
    prior_unknown = run / "archive" / "epochs" / "0"
    prior_unknown.mkdir(parents=True)
    (prior_unknown / "unknown_outcomes.jsonl").write_text('{"unit_id":"kept"}\n', encoding="utf-8")
    assert main(["archive-finished", "--run-dir", str(run), "--reason", "epoch 0 done"]) == 0
    assert current_epoch(run) == 1
    assert json.loads((run / "unit_restarts.json").read_text(encoding="utf-8")) == {}
    unknown = (run / "archive" / "epochs" / "0" / "unknown_outcomes.jsonl").read_text(encoding="utf-8")
    assert "kept" in unknown and '"unit_id": "old"' in unknown or '"unit_id":"old"' in unknown
    assert (run / "archive" / "protocol-rerun" / "epoch-0" / "swe_predictions.jsonl").is_file()
    assert (legacy / "old.txt").read_text(encoding="utf-8") == "keep"

    second = run_schedule(b0, run, live=False, sender=sender)
    assert second[1]["protocol_epoch"] == 1 and second[1]["restart_count"] == 1
    epoch1 = run / "archive" / "restarts" / "epoch-1" / STRONG / "1" / "records.jsonl"
    assert epoch1.is_file()
    assert (legacy / "old.txt").read_text(encoding="utf-8") == "keep"
    assert (epoch0 / "records.jsonl").is_file()

    def boom(url, body, headers):
        raise AssertionError("resume must not rerun a finished unit")

    resumed = run_schedule(b0, run, live=False, sender=boom)
    assert [row["unit_id"] for row in resumed] == [MAIN, STRONG]
    assert all(row["status"] == "completed" for row in resumed)

    m2 = select_units("m2", arms=("g1-main",))[0]
    handed = run_schedule((m2,), run, live=False, sender=sender)
    assert handed[0]["status"] == "completed"
    assert handed[0]["study"] == "m2"
    assert MAIN in (run / "records.jsonl").read_text(encoding="utf-8")
    with pytest.raises(StopIteration):
        next(steps)


def test_removed_record_with_restart_count_runs_the_one_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    run = tmp_path / "run"
    run.mkdir()
    (run / "protocol_epoch.json").write_text(
        json.dumps({"epoch": 4, "reason": "continue"}) + "\n", encoding="utf-8")
    (run / "records.jsonl").write_text(
        json.dumps({"unit_id": MAIN, "status": "completed", "halt_schedule": False}) + "\n",
        encoding="utf-8")
    (run / "unit_restarts.json").write_text(json.dumps({STRONG: 1}), encoding="utf-8")
    (run / "checkpoints" / MAIN).mkdir(parents=True)
    (run / "checkpoints" / MAIN / "model.patch").write_text("main", encoding="utf-8")
    legacy = run / "archive" / "restarts" / STRONG / "1" / "checkpoint" / STRONG
    legacy.mkdir(parents=True)
    (legacy / "old.txt").write_text("keep", encoding="utf-8")
    ledger = open_run_ledger(run / "budget.sqlite3", "dry-run")
    request_id = f"req-e4-{STRONG}-12"
    ledger.reserve(
        request_id, "mimo-v2.6-pro", 1000, payload_sha256="a" * 64,
        tokens=10, seconds_ms=10, unit_id=STRONG, unit_request_cap=12,
        protocol_epoch=4)
    ledger.settle_known(request_id, 1000, tokens=10, seconds_ms=10)
    rows = run_schedule(_b0("g1-main", "g1-strong"), run, live=False, sender=_ok)
    assert rows[0]["unit_id"] == MAIN and rows[0]["status"] == "completed"
    assert rows[1]["unit_id"] == STRONG
    assert rows[1]["restart_count"] == 1
    assert rows[1]["status"] == "completed"
    assert rows[1]["protocol_epoch"] == 4
    fresh = [item for item in ledger.request_ids() if item.startswith(f"req-e4-{STRONG}-r1-")]
    assert fresh
    assert (legacy / "old.txt").read_text(encoding="utf-8") == "keep"
    assert (run / "checkpoints" / MAIN / "model.patch").read_text(encoding="utf-8") == "main"
