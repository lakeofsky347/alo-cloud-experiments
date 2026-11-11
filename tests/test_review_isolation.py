"""Review commands stay on a read-only snapshot. Offline, no model API."""
import json
from pathlib import Path

import pytest

from alo_cloud.errors import ProtocolError
from alo_cloud.public_tools import dispatch_action
from alo_cloud.review_isolation import snapshot_workspace, tree_digest
from alo_cloud.sandbox import DockerSandbox


def _tree(root, text="BUG\n"):
    root.mkdir()
    (root / "bug.py").write_text(text, encoding="utf-8")


def test_review_command_cannot_mutate_the_snapshot_or_the_implementation(tmp_path):
    live = tmp_path / "live"
    _tree(live)
    snap = tmp_path / "snap"
    snapshot_workspace(live, snap)
    services = type("S", (), {})()
    services.role_tools = ("command",)
    services.review_snapshot = snap
    services.workspace = type("W", (), {"root": live})()
    services.sandbox = None
    relative = dispatch_action(services, {"tool": "command", "arguments": {
        "argv": ["python3", "-c", "open('bug.py','w').write('PWNED')"],
    }})
    assert relative["returncode"] != 0
    assert (live / "bug.py").read_text(encoding="utf-8") == "BUG\n"
    assert (snap / "bug.py").read_text(encoding="utf-8") == "BUG\n"
    absolute = dispatch_action(services, {"tool": "command", "arguments": {
        "argv": ["python3", "-c",
                 "open(" + json.dumps(str(live / "bug.py")) + ",'w').write('PWNED')"],
    }})
    assert absolute["returncode"] != 0
    assert (live / "bug.py").read_text(encoding="utf-8") == "BUG\n"
    assert tree_digest(live) == tree_digest(snap)
    with pytest.raises(ProtocolError, match="tool_outside_role_contract"):
        dispatch_action(services, {"tool": "write_file", "arguments": {
            "path": "bug.py", "content": "PWNED\n"}})
    with pytest.raises(ProtocolError, match="tool_outside_role_contract"):
        dispatch_action(services, {"tool": "execute_code", "arguments": {
            "language": "python", "code": "open('bug.py','w').write('PWNED')"}})


def test_an_implementer_command_still_uses_the_live_tree(tmp_path):
    live = tmp_path / "live"
    _tree(live)
    seen = {}

    def command(argv, timeout_s=30):
        del timeout_s
        seen["argv"] = argv
        (live / "bug.py").write_text("EDITED\n", encoding="utf-8")
        return {"returncode": 0, "stdout": "live", "stderr": ""}

    services = type("S", (), {})()
    services.role_tools = ("write_file", "edit_file", "command")
    services.command = command
    services.review_snapshot = tmp_path / "unused"
    result = dispatch_action(services, {"tool": "command", "arguments": {
        "argv": ["python3", "bug.py"]}})
    assert result["stdout"] == "live"
    assert seen["argv"] == ["python3", "bug.py"]
    assert (live / "bug.py").read_text(encoding="utf-8") == "EDITED\n"


def test_docker_review_mounts_the_snapshot_read_only(tmp_path):
    live = tmp_path / "live"
    snap = tmp_path / "snap"
    _tree(live)
    _tree(snap, "SNAP\n")
    seen = {}

    def runner(argv, timeout_s):
        del timeout_s
        seen["argv"] = argv
        return {"returncode": 0, "stdout": "", "stderr": ""}

    sandbox = DockerSandbox(live, image="example.invalid/alo:test", runner=runner)
    sandbox.readonly_command(snap, ["python3", "-c", "print(1)"], 30)
    rendered = "\n".join(seen["argv"])
    assert f"src={snap.resolve()}" in rendered
    assert "readonly" in rendered
    assert f"src={live.resolve()}" not in rendered
    assert "--cap-drop=ALL" in seen["argv"]
    assert "--network=none" in seen["argv"]


def test_merge_review_locks_the_populated_tree_and_does_not_copy_back(tmp_path):
    live = tmp_path / "live"
    snap = tmp_path / "snap"
    _tree(live)
    _tree(snap, "SNAP\n")
    sandbox = DockerSandbox(
        live, image="example.invalid/alo:test", merge_image_workspace=True)
    argv = sandbox.readonly_argv(snap, ["python3", "-c", "print(1)"])
    rendered = "\n".join(argv)
    assert f"src={snap.resolve()}" in rendered
    assert "readonly" in rendered
    assert f"src={live.resolve()}" not in rendered
    assert "review_merge_tree_not_readonly" in rendered
    assert "MS_RDONLY" in rendered
    assert "tar -C /agent" in rendered
    assert "tar -C /workspace -xf" in rendered
    assert "tar -C /workspace --exclude" not in rendered
    assert "--cap-add=SYS_ADMIN" in argv
    assert "--cap-drop=ALL" in argv
    normal = "\n".join(sandbox.docker_argv(["./executable"]))
    assert "tar -C /workspace" in normal
    assert "--read-only" not in sandbox.docker_argv(["./executable"])


def test_missing_snapshot_does_not_fall_back_to_the_live_tree(tmp_path):
    live = tmp_path / "live"
    _tree(live)
    services = type("S", (), {})()
    services.role_tools = ("command",)
    services.review_snapshot = None
    services.workspace = type("W", (), {"root": live})()
    services.sandbox = None

    def command(*args, **kwargs):
        del args, kwargs
        raise AssertionError("live command")

    services.command = command
    with pytest.raises(ProtocolError, match="review_snapshot_missing"):
        dispatch_action(services, {"tool": "command", "arguments": {"argv": ["python3", "-c", "print(1)"]}})
    assert (live / "bug.py").read_text(encoding="utf-8") == "BUG\n"
    assert Path(live / "bug.py").is_file()
