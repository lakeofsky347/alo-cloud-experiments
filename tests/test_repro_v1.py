"""Offline checks for the opt-in repro-v1 signal. No docker and no model API."""
import json
import subprocess
from pathlib import Path

from alo.org.organization import Organization
from alo_cloud.adapters import HeterogeneousALO
from alo_cloud.public_check import public_quality_signal
from alo_cloud.repro import (
    REPRO_CRITERION_ID, REPRO_INSTRUCTION, apply_definite_outcome, collect_repro_v1,
    repro_v1_evidence, repro_v1_status,
)
from alo_cloud.sandbox import CRITERION_ID, Workspace
from tests.test_adapters import ScriptedServices
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


class _Flow(ScriptedServices):
    def __init__(self, directory, workspace, replies):
        super().__init__(directory)
        self.workspace = workspace
        self.replies = list(replies)
        self.edit_log = []

    def complete_for_model(self, model, messages, *, max_tokens=4096):
        self.calls.append(model)
        content = messages[-1]["content"] if messages else ""
        if "ALO_DECOMPOSE" in content:
            raise AssertionError("repro-v1 fixed topology must not call the planner")
        return {"text": self.replies.pop(0), "input_tokens": 8, "output_tokens": 4,
                "latency_ms": 0, "cost_microcny": 1, "currency": "CNY"}

    def command(self, argv, timeout_s=30):
        del timeout_s
        completed = subprocess.run(argv, cwd=self.workspace.root, capture_output=True, check=False)
        return {"returncode": completed.returncode,
                "stdout": completed.stdout.decode(errors="replace"),
                "stderr": completed.stderr.decode(errors="replace")}

    def write_file(self, path, content, append=False):
        return self.workspace.write_file(path, content, append=append)

    def edit_file(self, path, old_str, new_str):
        return self.workspace.edit_file(path, old_str, new_str)

    def local_criterion(self, **kwargs):
        from alo_cloud.services import StudyServices
        return StudyServices.local_criterion(self, **kwargs)


def _flow(tmp_path, script):
    root = tmp_path / "ws"
    workspace = Workspace(root)
    subprocess.check_call(["git", "init", "-q"], cwd=root)
    subprocess.check_call(["git", "config", "user.email", "t@example.com"], cwd=root)
    subprocess.check_call(["git", "config", "user.name", "t"], cwd=root)
    (root / "bug.py").write_text("BUG\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "bug.py"], cwd=root)
    subprocess.check_call(["git", "commit", "-q", "-m", "base"], cwd=root)
    replies = [
        json.dumps({"tool": "write_file", "arguments": {"path": "alo_repro.py", "content": script}}),
        json.dumps({"tool": "command", "arguments": {"argv": ["python3", "alo_repro.py"]}}),
        json.dumps({"tool": "edit_file", "arguments": {
            "path": "bug.py", "old_str": "BUG\n", "new_str": "FIXED\n"}}),
        json.dumps({"tool": "command", "arguments": {"argv": ["python3", "alo_repro.py"]}}),
        json.dumps({"final": "submitted the fix"}),
    ]
    service = _Flow(tmp_path / "state", workspace, replies)
    service.state_dir.mkdir(parents=True, exist_ok=True)
    adapter = HeterogeneousALO(
        planner_model="deepseek-flash", fixed_topology=True, lightweight_verifier=True,
        routing_mode="fixed", criterion_id=REPRO_CRITERION_ID, max_attempts=1, max_turns=6, seed=1)
    return adapter, service


def test_s1_fail_then_pass_is_a_criterion_pass_and_pass_then_pass_is_a_fail(tmp_path):
    discriminating = "import sys\nfrom pathlib import Path\nsys.exit(0 if 'FIXED' in Path('bug.py').read_text() else 1)\n"
    adapter, service = _flow(tmp_path / "discriminates", discriminating)
    assert adapter.run_stage({"prompt": "bug"}, service)["status"] == "completed"
    prompt = "\n".join(message["content"] for message in adapter.state["worker_transcripts"]["s1"]
                       if isinstance(message.get("content"), str))
    assert "alo_repro.py" in prompt
    assert "non-zero" in prompt
    assert REPRO_INSTRUCTION.strip()[:40] in prompt
    measured = adapter.state["repro_v1_measurements"]
    assert measured[0]["model_id"] == "deepseek-flash"
    assert measured[0]["recorded_before_returncode"] == 1
    assert measured[0]["recorded_after_returncode"] == 0
    assert measured[0]["repro_before_returncode"] != 0
    assert measured[0]["repro_after_returncode"] == 0
    assert measured[0]["signal"] == "pass"
    assert service.calls == ["deepseek-flash"] * 5

    always = "import sys\nsys.exit(0)\n"
    adapter, service = _flow(tmp_path / "does-not-discriminate", always)
    assert adapter.run_stage({"prompt": "bug"}, service)["status"] == "failed"
    measured = adapter.state["repro_v1_measurements"]
    assert measured[0]["recorded_before_returncode"] == 0
    assert measured[0]["recorded_after_returncode"] == 0
    assert measured[0]["repro_before_returncode"] == 0
    assert measured[0]["repro_after_returncode"] == 0
    assert measured[0]["signal"] == "fail"
    org = Organization(str(service.state_dir / "alo.sqlite3"))
    assert any(row["beta"] > 1 for row in org.belief.dump())
    org.conn.close()
