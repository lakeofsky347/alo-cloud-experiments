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
    assert result["status"] == "completed"
    assert result["summary"] == "submitted work"
    assert result["termination"] == "final"
    feedback = json.loads(messages[3]["content"])
    assert feedback["error_phase"] == "response_validation"
    assert feedback["tool_dispatched"] is False
    assert set(feedback["remaining_budget"]) == {"requests", "tokens", "seconds", "cny"}
    assert "hidden" not in feedback["remaining_budget"]
    assert not any(call[0] == "command" for call in services.calls if isinstance(call, tuple))


def test_prose_json_is_not_a_tool_call_unless_it_has_tool_shape():
    from alo_cloud.public_tools import leading_json_object

    trailed = '{"tool":"command","arguments":{"argv":["pytest"]}}\n<system>stray'
    assert leading_json_object(trailed)["tool"] == "command"
    assert leading_json_object('  {"final":"done"}')["final"] == "done"
    assert leading_json_object('Look at {"a": 1} in the notes') is None
    buried = 'Please run {"tool":"command","arguments":{"argv":["pytest","tests/test_bug.py"]}}'
    assert leading_json_object(buried)["arguments"]["argv"][0] == "pytest"


def test_leading_json_with_trailing_system_text_still_runs():
    services = LoopServices([
        '{"tool":"command","arguments":{"argv":["python3","alo_repro.py"]}}\n<system>stray',
        '{"final":"done"}',
    ])
    result = persistent_tool_loop(services, "public task", [], max_turns=2)
    assert result["status"] == "completed"
    assert ("command", ("python3", "alo_repro.py"), 30) in services.calls


def test_provider_tool_calls_are_not_executed():
    services = LoopServices([{"text": '{"final":"submitted"}', "tool_calls": [{"name": "command", "arguments": {}}]}])
    result = persistent_tool_loop(services, "public task", [], max_turns=1)
    assert result["status"] == "completed"
    assert services.calls == [("complete", 2, 4096)]


def test_length_stop_continues_and_an_empty_turn_continues():
    stopped = LoopServices([
        {"text": "", "length_stop": True, "finish_reason": "length"},
        '{"final":"after length"}',
    ])
    messages = []
    result = persistent_tool_loop(stopped, "public task", messages, max_turns=3)
    assert result["status"] == "completed"
    assert result["summary"] == "after length"
    assert result["termination"] == "final"
    assert result["length_stops"] == 1
    assert any("Your output budget was used by reasoning" in message["content"] for message in messages)
    assert len(stopped.calls) == 2
    empty = LoopServices([
        {"text": "", "finish_reason": "stop"},
        '{"final":"after empty"}',
    ])
    messages = []
    result = persistent_tool_loop(empty, "public task", messages, max_turns=3)
    assert result["status"] == "completed"
    assert result["summary"] == "after empty"
    assert result["length_stops"] == 0
    assert "Reply with ONE short action." in messages[3]["content"]
    assert "empty_assistant_turn" not in messages[3]["content"]
    missed = LoopServices([
        {"text": "", "provider_not_reached": True},
        '{"final":"after reach"}',
    ])
    messages = []
    result = persistent_tool_loop(missed, "public task", messages, max_turns=3)
    assert result["status"] == "completed"
    assert result["summary"] == "after reach"
    assert "not reached before the request was sent" in messages[3]["content"]
    rejected = LoopServices([
        {"text": "", "vendor_error": True},
        '{"final":"after error"}',
    ])
    messages = []
    result = persistent_tool_loop(rejected, "public task", messages, max_turns=3)
    assert result["status"] == "completed"
    assert result["termination"] == "final"
    assert json.loads(messages[3]["content"])["provider_error"] == "vendor_error"


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
