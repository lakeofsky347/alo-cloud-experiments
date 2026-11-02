"""Offline checks for the opt-in repro-v1 signal. No docker and no model API."""
import subprocess
from pathlib import Path

from alo.org.organization import Organization
from alo_cloud.public_check import public_quality_signal
from alo_cloud.repro import (
    REPRO_CRITERION_ID, apply_definite_outcome, collect_repro_v1, repro_v1_evidence, repro_v1_status,
)
from alo_cloud.sandbox import CRITERION_ID
from tests.test_services import build


SCRIPT = "import sys\nfrom pathlib import Path\nsys.exit(0 if 'FIXED' in Path('bug.py').read_text() else 1)\n"


def _git(root):
    subprocess.check_call(["git", "init", "-q"], cwd=root)
    subprocess.check_call(["git", "config", "user.email", "t@example.com"], cwd=root)
    subprocess.check_call(["git", "config", "user.name", "t"], cwd=root)
    (root / "bug.py").write_text("BUG\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "bug.py"], cwd=root)
    subprocess.check_call(["git", "commit", "-q", "-m", "base"], cwd=root)


class _Workspace:
    def __init__(self, root):
        self.root = root


def _runner(root):
    def command(argv, timeout_s=30):
        del timeout_s
        completed = subprocess.run(argv, cwd=root, capture_output=True, check=False)
        return {"returncode": completed.returncode,
                "stdout": completed.stdout.decode(errors="replace"),
                "stderr": completed.stderr.decode(errors="replace")}
    return command


def test_status_table_does_not_treat_syntax_as_a_pass(tmp_path):
    (tmp_path / "pkg.py").write_text("value = 1\n", encoding="utf-8")
    assert public_quality_signal(tmp_path, changed_paths=["pkg.py"], patch_text="diff --git a/pkg.py") == "pass"
    assert repro_v1_status(before=None, after=None, tests=None) == "unknown"
    evidence = repro_v1_evidence(before=None, after=None, tests=None, changed_paths=["pkg.py"], root=tmp_path)
    assert evidence["criterion_id"] == REPRO_CRITERION_ID
    assert evidence["signal"] == "unknown"
    assert evidence["reliable"] is False
    assert repro_v1_status(before=1, after=0, tests=None) == "pass"
    assert repro_v1_status(before=1, after=0, tests=0) == "pass"
    assert repro_v1_status(before=1, after=1, tests=0) == "fail"
    assert repro_v1_status(before=0, after=0, tests=0) == "fail"
    assert repro_v1_status(before=None, after=None, tests=1) == "fail"
    assert repro_v1_status(before=None, after=None, tests=0) == "pass"
    assert repro_v1_status(before=None, after=0, tests=None) == "unknown"
    coded = repro_v1_evidence(
        before=1, after=1, tests=None, changed_paths=["pkg.py"], patch_text="diff --git a/pkg.py", root=tmp_path)
    assert coded["syntax_aware_signal"] == "fail"
    assert coded["repro_after_returncode"] == 1
    assert coded["signal"] == "fail"


def test_before_fails_and_after_passes_then_the_tree_is_restored(tmp_path):
    _git(tmp_path)
    (tmp_path / "bug.py").write_text("FIXED\n", encoding="utf-8")
    (tmp_path / "alo_repro.py").write_text(SCRIPT, encoding="utf-8")
    services = type("S", (), {})()
    services.workspace = _Workspace(tmp_path)
    services.command = _runner(tmp_path)
    evidence = collect_repro_v1(services, changed_paths=[])
    assert evidence["repro_before_returncode"] != 0
    assert evidence["repro_after_returncode"] == 0
    assert evidence["signal"] == "pass"
    assert (tmp_path / "bug.py").read_text(encoding="utf-8") == "FIXED\n"
    assert evidence["criterion_id"] == REPRO_CRITERION_ID


def test_a_script_that_still_fails_makes_beta_exceed_one(tmp_path):
    _git(tmp_path)
    (tmp_path / "alo_repro.py").write_text(SCRIPT, encoding="utf-8")
    services = type("S", (), {})()
    services.workspace = _Workspace(tmp_path)
    services.command = _runner(tmp_path)
    evidence = collect_repro_v1(services, changed_paths=[])
    assert evidence["signal"] == "fail"
    org = Organization(str(tmp_path / "alo.sqlite3"))
    assert apply_definite_outcome(org.belief, org.tms, "deepseek-flash", "unknown") is False
    assert org.conn.execute("SELECT count(*) FROM belief").fetchone()[0] == 0
    assert apply_definite_outcome(org.belief, org.tms, "deepseek-flash", evidence["signal"]) is True
    row = org.belief.dump()[0]
    assert row["beta"] > 1
    assert row["alpha"] == 1
    failures = org.conn.execute("SELECT failures FROM tms_outcomes").fetchone()[0]
    assert failures == 1
    org.conn.close()


def test_study_services_keeps_syntax_until_repro_v1_is_selected(tmp_path, monkeypatch):
    services, _seen, _ledger = build(tmp_path, monkeypatch, [])
    syntax = services.local_criterion(role_id="n", model_id="m", output="", context={})
    assert syntax["criterion_id"] == CRITERION_ID
    services.workspace.write_file("alo_repro.py", "import sys\nsys.exit(1)\n")
    services.active_criterion_id = REPRO_CRITERION_ID

    def command(argv, timeout_s=30):
        del argv, timeout_s
        return {"returncode": 1, "stdout": "", "stderr": ""}

    services.command = command
    evidence = services.local_criterion(role_id="n", model_id="m", output="", context={})
    assert evidence["criterion_id"] == REPRO_CRITERION_ID
    assert evidence["status"] == "fail"
    assert evidence["reliable"] is True
    assert evidence["scope"] == "node_local"
    assert Path(services.workspace.root, "alo_repro.py").is_file()
