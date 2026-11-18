"""Run the opt-in build handoff through the PB entry point with an offline sender."""
import json

import pytest

from alo_cloud import fox_pb
from alo_cloud.fox_pb import PbStopped, run_arm, run_g0
from alo_cloud.programbench_prompt import PB_BUILD_PROTOCOL
from alo_cloud.sandbox import DockerSandbox


def action(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def test_build_handoff_checks_actual_snapshot_and_preserves_the_last_four_requests(tmp_path, monkeypatch):
    requests = {"p1": [], "p2": []}
    builds = []

    def sender(url, body, headers, timeout_s):
        messages = body["messages"]
        context = "\n".join(message["content"] for message in messages)
        node = "p2" if context.rfind("Role Contract: p2") > context.rfind("Role Contract: p1") else "p1"
        requests[node].append(messages[-1]["content"])
        turn = len(requests[node])
        if node == "p2":
            text = action("pb_build") if turn == 1 else '{"final":"Verified the handoff."}'
        else:
            replies = {
                7: (action("write_file", path="main.c", content="good") + "\n"
                    + action("write_file", path="compile.sh", content="#!/bin/sh\n")),
                9: action("write_file", path="compile.sh", content="#!/bin/sh\n"),
                10: action("pb_build"),
                15: action("edit_file", path="main.c", old_str="good", new_str="broken"),
                17: action("command", argv=["./executable"]),
                18: '{"final":"Ready."}',
                19: action("edit_file", path="main.c", old_str="broken", new_str="fixed"),
                20: '{"final":"Repaired and ready for review."}',
            }
            text = replies.get(turn, action("list_files"))
        payload = {"model": body["model"], "choices": [{"message": {"content": text},
                  "finish_reason": "stop"}], "usage": {"prompt_tokens": 8, "completion_tokens": 4}}
        return 200, json.dumps(payload).encode()

    def public_build_command(self, root, argv, timeout_s):
        source = (root / "main.c").read_text()
        builds.append(source)
        if source == "broken":
            code, diagnostic = 1, "main.c:1: error: incomplete expression"
        else:
            code, diagnostic = 0, ""
            exe = root / "executable"
            exe.write_bytes(b"synthetic candidate")
            exe.chmod(0o755)
        receipt = {"stage": "compile", "returncode": code, "diagnostic": diagnostic}
        return {"returncode": 0, "stdout": json.dumps(receipt), "stderr": ""}

    capture = fox_pb._capture_final_submission

    def seal_then_change_workspace(source, directory, **kwargs):
        sealed = capture(source, directory, **kwargs)
        (source / "main.c").write_text("changed after sealing")
        return sealed

    monkeypatch.setattr(fox_pb, "_capture_final_submission", seal_then_change_workspace)
    monkeypatch.setattr(DockerSandbox, "public_build_command", public_build_command)
    run_dir = tmp_path / "repair"
    run_arm(run_dir, arm="A2", tasks="esubaalew__run", live=False, sender=sender,
            models="fox/deepseek-flash,fox/glm-5.3-flash", node_request_cap=20,
            pb_agent_protocol=PB_BUILD_PROTOCOL)
    assert len(requests["p1"]) == 20
    assert len(requests["p2"]) == 2
    first_write = json.loads(requests["p1"][7])
    assert first_write["written_files"] == [{"path": "main.c", "bytes": 4}]
    assert first_write["unexecuted_actions"] == [{"tool": "write_file", "path": "compile.sh"}]
    assert json.loads(requests["p1"][8])["build_checkpoint"]["reason"] == "missing_compile_sh"
    early = json.loads(requests["p1"][10])["result"]
    assert early["status"] == "pass"
    before_repairs = json.loads(requests["p1"][16])
    assert before_repairs["repair_window"]["requests_left"] == 4
    broken = before_repairs["build_checkpoint"]
    assert broken["status"] == "fail" and "main.c:1" in broken["diagnostic"]
    assert broken["submission_sha256"] != early["submission_sha256"]
    assert json.loads(requests["p1"][17])["public_tool_error"] == "reserved_repair_requests"
    assert json.loads(requests["p1"][18])["build_checkpoint"]["status"] == "fail"
    review = json.loads(requests["p2"][0].rsplit("Handoff build: ", 1)[1])
    assert review["status"] == "pass"
    assert review["submission_sha256"] not in {early["submission_sha256"], broken["submission_sha256"]}
    assert json.loads(requests["p2"][1])["result"] == review
    unit = json.loads((run_dir / "records.jsonl").read_text())
    checkpoint = json.loads((run_dir / "checkpoints" / unit["unit_id"] / "adapter.json").read_text())
    assert checkpoint["node_outcomes"][0]["handoff_build"]["submission_sha256"] == review["submission_sha256"]
    assert unit["public_build"]["submission_sha256"] == unit["submission_sha256"] == review["submission_sha256"]
    assert unit["public_compile_success"] is True
    assert unit["official_compile_success"] is None
    assert unit["submission_accepted"] is True
    assert "broken" in builds and "fixed" in builds and "changed after sealing" not in builds
    manifest = json.loads((run_dir / "manifest.json").read_text())
    protocol = manifest["pb_freeze"]["pb_agent_protocol"]
    assert protocol["first_build_request"] == 8 and protocol["reserved_repair_requests"] == 4
    with pytest.raises(PbStopped, match="settings_mismatch"):
        run_arm(run_dir, arm="A2", tasks="esubaalew__run", node_request_cap=20,
                models="fox/deepseek-flash,fox/glm-5.3-flash", pb_agent_protocol="node-budget-v1")


@pytest.mark.parametrize("options", [
    {"arm": "A4"}, {"node_request_cap": 16}, {"request_cap": 60}, {"probe_script_owner": "p2"},
])
def test_build_protocol_requires_the_frozen_role_and_budget_split(tmp_path, options):
    settings = dict(arm="A2", tasks="esubaalew__run", node_request_cap=20, pb_agent_protocol=PB_BUILD_PROTOCOL)
    with pytest.raises(PbStopped):
        run_arm(tmp_path / "new", **{**settings, **options})


def test_g0_routes_the_new_protocol_only_to_the_two_node_arm(tmp_path, monkeypatch):
    protocols = {}

    def record_settings(path, *, arm, **settings):
        protocols[arm] = settings.get("pb_agent_protocol")
        return {"units": 0, "wrote": 0}

    monkeypatch.setattr("alo_cloud.fox_pb.run_arm", record_settings)
    run_g0(tmp_path, pb_agent_protocol=PB_BUILD_PROTOCOL)
    assert protocols == {"A2": PB_BUILD_PROTOCOL, "A4": None}
