"""M19 node-budget-v1 (opt-in): p1 sees its own remaining node requests and gets one
compile.sh reminder at <= 6 left. Without the flag, and for A4, every request is
byte-identical to 753e54a (PR #49). Offline; replies are scripted."""
import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest

from alo_cloud.adapters import HeterogeneousALO
from alo_cloud.errors import PilotStopped
from alo_cloud.fox_pb import PbStopped, _adapter_config, decision_freeze, main, run_arm
from alo_cloud.programbench_prompt import (
    PB_AGENT_PROTOCOLS, PB_NODE_NOTICE_AT, PROGRAMBENCH_NODE_BUDGET_NOTICE)
from alo_cloud.tool_loop import persistent_tool_loop

NOTICE_PREFIX = "Node budget notice:"
LIST = json.dumps({"tool": "list_files", "arguments": {}})
MODELS = "fox/deepseek-flash,fox/glm-5.3-flash"

# sha256 of every request body (json.dumps(bodies, sort_keys=True, ensure_ascii=False))
# for the scripted replay below (list_files on every turn; node cap 20), computed
# at 753e54a. A4 must match with and without the flag; A2 must match without it.
BASE_753E54A_A4_BODIES_SHA256 = "d049fe68b38660625c305bac7754c1fc1609fbf88ed7bc76c626ac1b37715e38"
BASE_753E54A_A2_BODIES_SHA256 = "03933da1e2af62579452817b98c05490dbf9e3ef7737519e1de2fc0ba42478dd"
# M18 formal settings (cap 40, node cap 20, diff on, s1=p1/s2=p2, owner p1,
# 2400 s, read 600, glm-flashx). Before and after the opt-in field.
M18_PB_DECISION_SHA256 = "b38db09ff4321ad9249ced71f6cc39e8a095e93ba1371d51abdc915e1d53c062"
M19_PB_DECISION_SHA256 = "2b7108b9e1d36c3618963d0247947c79ac74b763ce61c8241605e76f483584b8"


class Stub:
    def __init__(self, n):
        self.replies = [LIST] * n
        self.seen = []

    def complete(self, messages, max_tokens=4096):
        self.seen.append(copy.deepcopy(messages))
        return {"text": self.replies.pop(0), "input_tokens": 3, "output_tokens": 2}

    def list_files(self):
        return []


def _budget():
    return {"node": "p1", "notice": PROGRAMBENCH_NODE_BUDGET_NOTICE, "notice_at": PB_NODE_NOTICE_AT}


def _notices(messages):
    return [i for i, m in enumerate(messages) if NOTICE_PREFIX in str(m.get("content", ""))]


def test_loop_shows_the_node_counter_and_sends_the_notice_once_at_six_left():
    stub = Stub(20)
    budget = _budget()
    messages = []
    persistent_tool_loop(stub, "do it", messages, max_turns=20, node_budget=budget)
    assert len(stub.seen) == 20
    for call, seen in enumerate(stub.seen):
        left_before = 20 - call
        assert len(_notices(seen)) == (1 if left_before <= 6 else 0), call
        if call:
            last_obs = [m for m in seen if m["role"] == "user"][-1]["content"]
            assert f"requests left for your node p1: {left_before}/20" in last_obs
    first = stub.seen[14]
    obs = json.loads(first[-1]["content"])
    assert first[-1]["role"] == "user" and obs["tool"] == "list_files"
    assert obs["requests_left"] == "requests left for your node p1: 6/20"
    assert obs["node_budget_notice"].startswith(NOTICE_PREFIX)
    assert "6 or fewer" in obs["node_budget_notice"] and "compile.sh" in obs["node_budget_notice"]
    assert all(m["role"] in {"system", "user", "assistant"} for m in messages)
    assert sum(m["role"] == "system" for m in messages) == 1   # no extra system message
    assert budget["sent"] is True and budget["sent_at_left"] == 6
    assert len(_notices(messages)) == 1


def test_resumed_transcript_with_a_notice_does_not_get_a_second_one():
    messages = []
    persistent_tool_loop(Stub(20), "do it", messages, max_turns=20, node_budget=_budget())
    stub = Stub(5)
    persistent_tool_loop(stub, "again", messages, max_turns=5, node_budget=_budget())
    assert len(_notices(messages)) == 1


def test_small_node_cap_gets_the_notice_on_its_first_call():
    stub = Stub(4)
    persistent_tool_loop(stub, "do it", [], max_turns=4, node_budget=_budget())
    assert len(_notices(stub.seen[0])) == 1
    assert stub.seen[0][-1]["role"] == "user" and stub.seen[0][-1]["content"].startswith("do it\n\n")
    assert len(_notices(stub.seen[-1])) == 1


def test_loop_without_node_budget_is_unchanged():
    a, b = Stub(8), Stub(8)
    ma, mb = [], []
    persistent_tool_loop(a, "do it", ma, max_turns=8)
    persistent_tool_loop(b, "do it", mb, max_turns=8, node_budget=None)
    assert ma == mb and a.seen == b.seen
    text = json.dumps(ma)
    assert "for your node" not in text and NOTICE_PREFIX not in text


def _sender(bodies):
    def sender(url, body, headers, timeout_s):
        del url, headers, timeout_s
        bodies.append(body)
        payload = {"model": body.get("model"),
                   "choices": [{"message": {"content": LIST}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}}
        return 200, json.dumps(payload).encode("utf-8")
    return sender


def _replay(run_dir, arm, monkeypatch, **extra):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    bodies = []
    run_arm(run_dir, arm=arm, tasks="esubaalew__run", live=False, sender=_sender(bodies),
            signal_override="unknown", models=MODELS, node_request_cap=20, **extra)
    return bodies


def _digest(bodies):
    return hashlib.sha256(json.dumps(bodies, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _role(body):
    text = "\n".join(m["content"] for m in body["messages"] if isinstance(m.get("content"), str))
    return "p2" if text.rfind("Role Contract: p2") > text.rfind("Role Contract: p1") else "p1"


def test_a4_requests_are_byte_identical_to_753e54a_with_and_without_the_flag(tmp_path, monkeypatch):
    off = _replay(tmp_path / "off", "A4", monkeypatch)
    on = _replay(tmp_path / "on", "A4", monkeypatch, pb_agent_protocol="node-budget-v1")
    assert len(off) == 40
    assert on == off
    assert _digest(off) == BASE_753E54A_A4_BODIES_SHA256
    text = json.dumps(on)
    assert "for your node" not in text and NOTICE_PREFIX not in text


def test_two_node_default_requests_are_byte_identical_to_753e54a(tmp_path, monkeypatch):
    off = _replay(tmp_path / "off", "A2", monkeypatch)
    assert _digest(off) == BASE_753E54A_A2_BODIES_SHA256


def test_flag_shows_p1_its_counter_and_reminds_it_in_its_own_turn_only(tmp_path, monkeypatch):
    off = _replay(tmp_path / "off", "A2", monkeypatch)
    on = _replay(tmp_path / "on", "A2", monkeypatch, pb_agent_protocol="node-budget-v1")
    p1 = [b for b in on if _role(b) == "p1"]
    p2 = [b for b in on if _role(b) == "p2"]
    assert len(p1) == 20 and len(p2) == 20
    for call, body in enumerate(p1):
        n = len(_notices(body["messages"]))
        assert n == (1 if 20 - call <= PB_NODE_NOTICE_AT else 0), call
        if call:
            obs = json.loads(body["messages"][-1]["content"])
            assert obs["requests_left"] == f"requests left for your node p1: {20 - call}/20 (unit: {40 - call}/40)"
            assert ("node_budget_notice" in obs) == (call == 14)
    assert all(m["role"] != "system" for m in p1[19]["messages"][1:])
    # p2 (read-only validator) sees no node counter and no reminder: identical to the default.
    assert p2 == [b for b in off if _role(b) == "p2"]
    # p1 differs from the default only by the counter wording and the one reminder.
    for call, (a, b) in enumerate(zip(p1, [b for b in off if _role(b) == "p1"])):
        stripped = []
        for m in a["messages"]:
            if NOTICE_PREFIX in str(m.get("content", "")):
                obs = json.loads(m["content"])
                del obs["node_budget_notice"]
                m = dict(m, content=json.dumps(obs, ensure_ascii=False, allow_nan=False))
            stripped.append(m)
        fixed = json.dumps(stripped).replace(
            f"requests left for your node p1: ", "requests left: ")
        for k in range(1, 21):
            fixed = fixed.replace(f"requests left: {k}/20 (unit: {20 + k}/40)", f"requests left: {20 + k}/40")
        assert fixed == json.dumps(b["messages"]), call
    row = json.loads((tmp_path / "on" / "records.jsonl").read_text(encoding="utf-8"))
    assert row["pb_agent_protocol"] == "node-budget-v1"
    manifest = json.loads((tmp_path / "on" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pb_agent_protocol"] == "node-budget-v1"
    assert manifest["pb_freeze"]["pb_agent_protocol"]["version"] == "node-budget-v1"
    off_row = json.loads((tmp_path / "off" / "records.jsonl").read_text(encoding="utf-8"))
    assert "pb_agent_protocol" not in off_row


def test_reverse_role_map_reminds_the_s1_node():
    settings = _m18(role_map={"s1": "p2", "s2": "p1"}, pb_agent_protocol="node-budget-v1")
    assert decision_freeze(settings)[0]["pb_agent_protocol"]["node_budget_nodes"] == ["p2"]
    _, config = _adapter_config("A2", models=("fox/deepseek-flash",), request_cap=40, seed=1,
                                online_update=False, dag="d", prompt="p", node_request_cap=20,
                                pb_agent_protocol="node-budget-v1", implementer_node="p2")
    assert config["pb_node_budget_nodes"] == ["p2"]


def _m18(**extra):
    settings = {"role_map": {"s1": "p1", "s2": "p2"}, "probe_script_owner": "p1",
                "differential_signal": True, "request_cap": 40, "online_update": False,
                "unit_seconds": 2400.0, "read_timeout_s": 600.0, "node_request_cap": 20,
                "pool_substitute": "glm-flashx"}
    settings.update(extra)
    return settings


def test_decision_hash_changes_only_by_the_protocol_field():
    before_body, before = decision_freeze(_m18())
    after_body, after = decision_freeze(_m18(pb_agent_protocol="node-budget-v1"))
    assert before == M18_PB_DECISION_SHA256
    assert "pb_agent_protocol" not in before_body
    assert set(after_body) - set(before_body) == {"pb_agent_protocol"}
    assert all(after_body[k] == v for k, v in before_body.items())
    assert after_body["pb_agent_protocol"] == {
        "version": "node-budget-v1", "node_budget_nodes": ["p1"], "notice_at_node_requests_left": 6}
    assert after == M19_PB_DECISION_SHA256 != before
    assert decision_freeze(_m18(pb_agent_protocol=None))[1] == before


def test_adapter_config_gates_on_node_caps_and_leaves_a4_alone():
    common = dict(models=("fox/deepseek-flash", "fox/glm-5.3-flash"), request_cap=40, seed=3,
                  online_update=False, dag="d", prompt="p")
    a4_off = _adapter_config("A4", node_request_cap=20, **common)
    a4_on = _adapter_config("A4", node_request_cap=20, pb_agent_protocol="node-budget-v1", **common)
    assert a4_on == a4_off == ("G1", {"fixed_model": "fox/deepseek-flash", "max_turns": 40})
    g_off = _adapter_config("A2", node_request_cap=20, **common)
    g_on = _adapter_config("A2", node_request_cap=20, pb_agent_protocol="node-budget-v1",
                           implementer_node="p1", **common)
    assert set(g_on[1]) - set(g_off[1]) == {"pb_agent_protocol", "pb_node_budget_nodes"}
    no_cap = _adapter_config("A2", pb_agent_protocol="node-budget-v1", **common)
    assert "pb_agent_protocol" not in no_cap[1]


def test_adapter_rejects_bad_protocol_settings():
    base = dict(planner_model="m", pb_mode=True, pb_dag="dag", node_request_caps={"p1": 20, "p2": 20})
    assert "pb_agent_protocol" not in HeterogeneousALO(**base).config
    with pytest.raises(PilotStopped):
        HeterogeneousALO(**base, pb_agent_protocol="v2", pb_node_budget_nodes=["p1"])
    with pytest.raises(PilotStopped):
        HeterogeneousALO(**base, pb_agent_protocol="node-budget-v1", pb_node_budget_nodes=["p9"])
    with pytest.raises(PilotStopped):
        HeterogeneousALO(planner_model="m", pb_mode=True, pb_dag="dag",
                         pb_agent_protocol="node-budget-v1", pb_node_budget_nodes=["p1"])
    with pytest.raises(PilotStopped):
        HeterogeneousALO(planner_model="m", pb_agent_protocol="node-budget-v1")


def test_run_arm_requires_a_node_cap_and_a_known_protocol(tmp_path):
    with pytest.raises(PbStopped, match="requires_node_request_cap"):
        run_arm(tmp_path / "a", arm="A2", tasks="esubaalew__run", pb_agent_protocol="node-budget-v1")
    with pytest.raises(PbStopped, match="unknown_pb_agent_protocol"):
        run_arm(tmp_path / "b", arm="A2", tasks="esubaalew__run", node_request_cap=20,
                pb_agent_protocol="node-budget-v2")
    assert PB_AGENT_PROTOCOLS == ("node-budget-v1",)


def test_a_run_dir_never_mixes_protocols(tmp_path):
    run_arm(tmp_path / "m18", arm="A2", tasks="esubaalew__run", node_request_cap=20)
    with pytest.raises(PbStopped, match="fox_run_settings_mismatch"):
        run_arm(tmp_path / "m18", arm="A2", tasks="esubaalew__run", node_request_cap=20,
                pb_agent_protocol="node-budget-v1")
    run_arm(tmp_path / "m19", arm="A2", tasks="esubaalew__run", node_request_cap=20,
            pb_agent_protocol="node-budget-v1")
    with pytest.raises(PbStopped, match="fox_run_settings_mismatch"):
        run_arm(tmp_path / "m19", arm="A2", tasks="esubaalew__run", node_request_cap=20)


def test_cli_flag_is_opt_in(tmp_path, capsys):
    assert main(["run", "--arm", "A2", "--tasks", "esubaalew__run", "--run-dir", str(tmp_path / "x"),
                 "--node-request-cap", "20", "--pb-agent-protocol", "node-budget-v1", "--dry-run"]) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "x" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pb_agent_protocol"] == "node-budget-v1"
    assert main(["run", "--arm", "A2", "--tasks", "esubaalew__run", "--run-dir", str(tmp_path / "y"),
                 "--node-request-cap", "20", "--dry-run"]) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "y" / "manifest.json").read_text(encoding="utf-8"))
    assert "pb_agent_protocol" not in manifest and "pb_agent_protocol" not in manifest["pb_freeze"]


_REPLAY = r'''
import hashlib, json, sys
from pathlib import Path
import urllib.request
urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network"))
from alo_cloud.fox_pb import run_arm
bodies = []
def sender(url, body, headers, timeout_s):
    bodies.append(body)
    text = json.dumps({"tool": "list_files", "arguments": {}})
    return 200, json.dumps({"model": body.get("model"), "choices": [{"message": {"content": text},
        "finish_reason": "stop"}], "usage": {"prompt_tokens": 8, "completion_tokens": 4,
        "total_tokens": 12}}).encode()
run_arm(Path(sys.argv[1]), arm=sys.argv[2], tasks="esubaalew__run", live=False, sender=sender,
        signal_override="unknown", models="fox/deepseek-flash,fox/glm-5.3-flash",
        node_request_cap=20)
print(hashlib.sha256(json.dumps(bodies, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
'''


@pytest.mark.skipif(not os.environ.get("ALO_PB_BASELINE_SRC"),
                    reason="set ALO_PB_BASELINE_SRC to a 753e54a src/ tree for the live replay")
@pytest.mark.parametrize("arm", ["A4", "A2"])
def test_live_replay_against_a_baseline_tree(tmp_path, monkeypatch, arm):
    env = dict(os.environ, PYTHONPATH=os.environ["ALO_PB_BASELINE_SRC"])
    out = subprocess.run([sys.executable, "-c", _REPLAY, str(tmp_path / "base"), arm], env=env,
                         capture_output=True, text=True, check=True, cwd=tmp_path)
    flag = {"pb_agent_protocol": "node-budget-v1"} if arm == "A4" else {}
    assert out.stdout.split()[-1] == _digest(_replay(tmp_path / "new", arm, monkeypatch, **flag))
