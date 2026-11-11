"""Offline checks for the opt-in repro-v1 signal. No docker and no model API."""
import json
import subprocess
import sys
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
        argv = list(argv)
        if argv and argv[0] in {"python", "python3"}:
            argv[0] = sys.executable
        completed = subprocess.run(argv, cwd=root, capture_output=True, check=False)
        return {"returncode": completed.returncode,
                "stdout": completed.stdout.decode(errors="replace"),
                "stderr": completed.stderr.decode(errors="replace")}
    return command


def test_only_commands_that_execute_tests_count():
    from alo_cloud.repro import _is_test_argv, repro_codes_from_messages

    assert _is_test_argv(["pytest", "tests/test_bug.py"])
    assert _is_test_argv(["python3", "-m", "pytest", "tests/test_bug.py"])
    assert _is_test_argv(["python3", "tests/test_bug.py"])
    assert _is_test_argv(["python3", "tests/runtests.py", "known_related_objects"])
    assert _is_test_argv(["python3", "bin/test", "sympy/core"])
    assert _is_test_argv(["tox", "-e", "py"])
    assert not _is_test_argv(["find", ".", "-name", "test_management_utility.py"])
    assert not _is_test_argv(["sed", "-n", "1,20p", "sympy/core/tests/test_basic.py"])
    assert not _is_test_argv(["cat", "tests/test_bug.py"])
    assert not _is_test_argv(["grep", "-n", "test_", "tests/test_bug.py"])
    assert not _is_test_argv(["ls", "tests"])
    messages = [
        {"role": "assistant", "content": json.dumps(
            {"tool": "command", "arguments": {"argv": ["sed", "-n", "1p", "tests/test_basic.py"]}})},
        {"role": "user", "content": json.dumps({"tool": "command", "result": {"returncode": 0}})},
    ]
    before, after, tests = repro_codes_from_messages(messages)
    assert (before, after, tests) == (None, None, None)


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
    assert repro_v1_status(before=None, after=None, tests=0) == "unknown"
    assert repro_v1_status(before=None, after=None, tests=0, tests_base=0) == "unknown"
    assert repro_v1_status(before=None, after=None, tests=0, tests_base=1) == "unknown"
    assert repro_v1_status(
        before=None, after=None, tests=0, tests_base=1,
        same_frozen_tests=True, comparable_exits=True) == "pass"
    assert repro_v1_status(
        before=None, after=None, tests=0, tests_base=1,
        same_frozen_tests=False, comparable_exits=True) == "unknown"
    assert repro_v1_status(
        before=None, after=None, tests=0, tests_base=1,
        same_frozen_tests=True, comparable_exits=False) == "unknown"
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


def test_staged_repro_measurement_keeps_the_candidate_fix(tmp_path):
    """Live workspaces stage every edit, including alo_repro.py.

    The before-run reverses that patch and the command stages the unfixed
    tree. The fix must still be in the worktree and the index afterwards.
    """
    root = tmp_path / "ws"
    workspace = Workspace(root)
    subprocess.check_call(["git", "init", "-q"], cwd=root)
    subprocess.check_call(["git", "config", "user.email", "t@example.com"], cwd=root)
    subprocess.check_call(["git", "config", "user.name", "t"], cwd=root)
    (root / "bug.py").write_text("BUG\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "bug.py"], cwd=root)
    subprocess.check_call(["git", "commit", "-q", "-m", "base"], cwd=root)
    (root / "bug.py").write_text("FIXED\n", encoding="utf-8")
    (root / "alo_repro.py").write_text(SCRIPT, encoding="utf-8")
    subprocess.check_call(["git", "add", "-A"], cwd=root)

    def command(argv, timeout_s=30):
        del timeout_s
        completed = subprocess.run(argv, cwd=root, capture_output=True, check=False)
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=False, capture_output=True)
        return {"returncode": completed.returncode, "stdout": "", "stderr": ""}

    services = type("S", (), {})()
    services.workspace = workspace
    services.command = command
    evidence = collect_repro_v1(services)
    assert evidence["signal"] == "pass"
    assert evidence["repro_before_returncode"] != 0
    assert evidence["repro_after_returncode"] == 0
    assert (root / "bug.py").read_text(encoding="utf-8") == "FIXED\n"
    assert (root / "alo_repro.py").is_file()
    staged = subprocess.run(
        ["git", "-C", str(root), "show", ":bug.py"], capture_output=True, text=True, check=False)
    assert staged.returncode == 0
    assert "FIXED" in staged.stdout
    assert "BUG" not in staged.stdout


def test_untracked_files_survive_and_do_not_block_reapply(tmp_path):
    """Files never staged are outside ``git diff HEAD``.

    The measurement's ``git add -A`` makes ``reset --hard`` delete them, and
    an untracked copy of a patch-added path makes the reapply fail. Both have
    to come back as the candidate left them.
    """
    root = tmp_path / "ws"
    workspace = Workspace(root)
    subprocess.check_call(["git", "init", "-q"], cwd=root)
    subprocess.check_call(["git", "config", "user.email", "t@example.com"], cwd=root)
    subprocess.check_call(["git", "config", "user.name", "t"], cwd=root)
    (root / "bug.py").write_text("BUG\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "bug.py"], cwd=root)
    subprocess.check_call(["git", "commit", "-q", "-m", "base"], cwd=root)
    (root / "bug.py").write_text("FIXED\n", encoding="utf-8")
    (root / "alo_repro.py").write_text(SCRIPT, encoding="utf-8")
    (root / "extra.py").write_text("NEW\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "bug.py", "alo_repro.py", "extra.py"], cwd=root)
    (root / "helper.py").write_text("KEEP\n", encoding="utf-8")
    helper_mode = (root / "helper.py").stat().st_mode & 0o777

    calls = {"n": 0}

    def command(argv, timeout_s=30):
        del timeout_s
        calls["n"] += 1
        completed = subprocess.run(argv, cwd=root, capture_output=True, check=False)
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=False, capture_output=True)
        # Only the before-run plants residue. The after-run must observe the restore.
        # reset --hard does not remove files created after git add -A.
        if calls["n"] == 1:
            (root / "extra.py").write_text("RESIDUE\n", encoding="utf-8")
            (root / "noise.py").write_text("NOISE\n", encoding="utf-8")
        return {"returncode": completed.returncode, "stdout": "", "stderr": ""}

    services = type("S", (), {})()
    services.workspace = workspace
    services.command = command
    evidence = collect_repro_v1(services)
    assert evidence["signal"] == "pass"
    assert evidence["repro_before_returncode"] != 0
    assert evidence["repro_after_returncode"] == 0
    assert (root / "bug.py").read_text(encoding="utf-8") == "FIXED\n"
    assert (root / "alo_repro.py").is_file()
    assert (root / "extra.py").read_text(encoding="utf-8") == "NEW\n"
    assert (root / "helper.py").read_text(encoding="utf-8") == "KEEP\n"
    assert (root / "helper.py").stat().st_mode & 0o777 == helper_mode
    assert not (root / "noise.py").exists()
    staged = subprocess.run(
        ["git", "-C", str(root), "show", ":extra.py"], capture_output=True, text=True, check=False)
    assert staged.returncode == 0
    assert staged.stdout == "NEW\n"


def _failing_suite(root):
    _git(root)
    (root / "bug.py").write_text("VALUE = 0\n", encoding="utf-8")
    (root / "test_bug.py").write_text(
        "from bug import VALUE\n\ndef test_value():\n    assert VALUE == 1\n",
        encoding="utf-8")
    subprocess.check_call(["git", "add", "bug.py", "test_bug.py"], cwd=root)
    subprocess.check_call(["git", "commit", "-q", "-m", "base"], cwd=root)


def test_live_measurement_passes_only_the_same_frozen_suite(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    _failing_suite(root)
    (root / "bug.py").write_text("VALUE = 1\n", encoding="utf-8")
    services = type("S", (), {})()
    services.workspace = _Workspace(root)
    services.command = _runner(root)
    evidence = collect_repro_v1(services, changed_paths=["test_bug.py"])
    assert evidence["tests_base_returncode"] != 0
    assert evidence["tests_returncode"] == 0
    assert evidence["same_frozen_tests"] is True
    assert evidence["comparable_exits"] is True
    assert evidence["signal"] == "pass"
    assert evidence["reason"] == "frozen_tests_failed_before_and_passed_after"
    assert (root / "bug.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert "VALUE == 1" in (root / "test_bug.py").read_text(encoding="utf-8")


def test_a_weakened_suite_or_an_import_error_is_not_a_pass(tmp_path):
    weakened = tmp_path / "weak"
    weakened.mkdir()
    _failing_suite(weakened)
    (weakened / "test_bug.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    services = type("S", (), {})()
    services.workspace = _Workspace(weakened)
    services.command = _runner(weakened)
    evidence = collect_repro_v1(services, changed_paths=["test_bug.py"])
    assert evidence["same_frozen_tests"] is True
    assert evidence["tests_returncode"] != 0
    assert evidence["signal"] == "fail"

    imported = tmp_path / "env"
    imported.mkdir()
    _git(imported)
    (imported / "bug.py").write_text("VALUE = 0\n", encoding="utf-8")
    (imported / "test_bug.py").write_text(
        "import not_a_real_module_alo_audit\n\ndef test_value():\n    assert True\n",
        encoding="utf-8")
    subprocess.check_call(["git", "add", "bug.py", "test_bug.py"], cwd=imported)
    subprocess.check_call(["git", "commit", "-q", "-m", "base"], cwd=imported)
    (imported / "not_a_real_module_alo_audit.py").write_text("VALUE = 1\n", encoding="utf-8")
    services = type("S", (), {})()
    services.workspace = _Workspace(imported)
    services.command = _runner(imported)
    evidence = collect_repro_v1(services, changed_paths=["test_bug.py", "not_a_real_module_alo_audit.py"])
    assert evidence["tests_returncode"] == 0
    assert evidence["tests_base_returncode"] != 0
    assert evidence["comparable_exits"] is False
    assert evidence["signal"] == "unknown"


def test_unknown_repro_signal_is_not_node_success_and_review_is_not_credited(tmp_path):
    from alo_cloud.sandbox import Workspace

    unknown = tmp_path / "unknown"
    workspace = Workspace(unknown / "ws")
    _git(workspace.root)
    service = _Flow(unknown / "state", workspace, [json.dumps({"final": "submitted without a script"})])
    service.state_dir.mkdir(parents=True, exist_ok=True)
    adapter = HeterogeneousALO(
        planner_model="deepseek-flash", fixed_topology=True, lightweight_verifier=True,
        routing_mode="fixed", criterion_id=REPRO_CRITERION_ID, max_attempts=1, max_turns=2, seed=1)
    assert adapter.run_stage({"prompt": "bug"}, service)["status"] == "failed"
    outcome = adapter.state["node_outcomes"][0]
    assert outcome["protocol_completed"] is True
    assert outcome["functional_pass"] is False
    assert outcome["public_signal"] == "unknown"
    skipped = adapter.state["last_native_run"]["skipped_subtasks"]
    assert any(item.get("subtask") == "s2" for item in skipped)
    org = Organization(str(service.state_dir / "alo.sqlite3"))
    assert org.conn.execute("SELECT count(*) FROM belief").fetchone()[0] == 0
    assert org.conn.execute("SELECT count(*) FROM tms_outcomes").fetchone()[0] == 0
    org.conn.close()

    passed = tmp_path / "credit"
    workspace = Workspace(passed / "ws")
    _init = passed / "ws"
    subprocess.check_call(["git", "init", "-q"], cwd=_init)
    subprocess.check_call(["git", "config", "user.email", "t@example.com"], cwd=_init)
    subprocess.check_call(["git", "config", "user.name", "t"], cwd=_init)
    (_init / "bug.py").write_text("BUG\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "bug.py"], cwd=_init)
    subprocess.check_call(["git", "commit", "-q", "-m", "base"], cwd=_init)
    (_init / "bug.py").write_text("FIXED\n", encoding="utf-8")
    (_init / "alo_repro.py").write_text(SCRIPT, encoding="utf-8")
    service = _Flow(passed / "state", workspace, [
        json.dumps({"final": "submitted the fix"}),
    ])
    service.state_dir.mkdir(parents=True, exist_ok=True)
    adapter = HeterogeneousALO(
        planner_model="deepseek-flash", fixed_topology=True, lightweight_verifier=True,
        routing_mode="fixed", criterion_id=REPRO_CRITERION_ID, max_attempts=1, max_turns=2, seed=2)
    assert adapter.run_stage({"prompt": "bug"}, service)["status"] == "completed"
    rows = {row["role_id"]: row for row in adapter.state["observations"]["deepseek-flash"]}
    assert rows["s1"]["credited_quality"] == 1.0
    assert rows["s1"]["topics"] == ["coding", "implementation"]
    assert rows["s2"]["quality_after"] == 1.0
    assert rows["s2"]["credited_quality"] is None
    assert rows["s2"]["topics"] == ["testing", "validation"]
    assert rows["s2"]["role_id"] == "s2"
    measured = adapter.state["repro_v1_measurements"]
    assert measured[0]["functional_pass"] is True
    assert measured[0]["protocol_completed"] is True
    outputs = [node["output_text"] for node in adapter.state["last_native_run"]["node_results"]]
    assert any("zero-request public re-test" in text for text in outputs)
    assert any("not an independent reviewer" in text for text in outputs)
    assert service.calls == ["deepseek-flash"]
