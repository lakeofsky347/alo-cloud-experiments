"""Offline public tool-loop checks. Replies are scripted; no provider is contacted."""
import json

from alo_cloud.tool_loop import persistent_tool_loop


class LoopServices:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.model = "mimo-v2.6-flash"

    def complete(self, messages, max_tokens=4096):
        self.calls.append(("complete", len(messages), max_tokens))
        reply = self.replies.pop(0)
        if isinstance(reply, dict):
            return reply
        return {"text": reply, "input_tokens": 3, "output_tokens": 2, "tool_calls": [{"name": "command"}]}

    def budget_remaining(self, model=None):
        return {"requests": 4, "tokens": 100, "seconds": 10, "cny": 1.5, "hidden": "not-protocol-feedback"}

    def list_files(self):
        self.calls.append("list_files")
        return ["a.py"]

    def read_file(self, path):
        raise AssertionError("not used")

    def write_file(self, path, content):
        raise AssertionError("not used")

    def search(self, text):
        raise AssertionError("not used")

    def command(self, argv, timeout_s=30):
        self.calls.append(("command", tuple(argv), timeout_s))
        return {"returncode": 0, "stdout": "ok", "stderr": ""}

    def execute_code(self, language, code, timeout_s=30):
        raise AssertionError("not used")


def test_final_completes_and_malformed_text_stays_in_the_transcript():
    services = LoopServices(["not json", '{"final":"submitted work"}'])
    messages = []
    result = persistent_tool_loop(services, "public task", messages, max_turns=2)
    assert result == {"status": "completed", "summary": "submitted work"}
    feedback = json.loads(messages[3]["content"])
    assert feedback["error_phase"] == "response_validation"
    assert feedback["tool_dispatched"] is False
    assert set(feedback["remaining_budget"]) == {"requests", "tokens", "seconds", "cny"}
    assert "hidden" not in feedback["remaining_budget"]
    assert not any(call[0] == "command" for call in services.calls if isinstance(call, tuple))


def test_provider_tool_calls_are_not_executed():
    services = LoopServices([{"text": '{"final":"submitted"}', "tool_calls": [{"name": "command", "arguments": {}}]}])
    result = persistent_tool_loop(services, "public task", [], max_turns=1)
    assert result["status"] == "completed"
    assert services.calls == [("complete", 2, 4096)]


def test_explicit_tool_then_final_and_turn_cap():
    services = LoopServices([
        '{"tool":"list_files","arguments":{}}',
        '{"tool":"command","arguments":{"argv":["python","-m","public_test"],"timeout_s":5}}',
        '{"final":"done"}',
    ])
    result = persistent_tool_loop(services, "public task", [], max_turns=3)
    assert result["status"] == "completed"
    assert "list_files" in services.calls
    assert ("command", ("python", "-m", "public_test"), 5) in services.calls
    capped = LoopServices(['{"tool":"list_files","arguments":{}}'])
    failed = persistent_tool_loop(capped, "public task", [], max_turns=1)
    assert failed["status"] == "failed"
    assert "list_files" in capped.calls
