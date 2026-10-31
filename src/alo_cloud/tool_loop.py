"""Strict JSON action loop with explicit, budgeted protocol correction.

Malformed output is observed and returned to the same model as a public
protocol error. It is never translated into an action or counted as a tool
success. The caller owns/persists ``messages``; completed turns update that
same list immediately, including before any tool side effect can raise.
"""
from __future__ import annotations

import json
import math

from .public_tools import (
    _unique_object, coerce_public_action, dispatch_action, parse_action,
)
from .public_tools import TOOLS_PROMPT
from .errors import ProtocolError


TOOL_LOOP_VERSION = "strict-json-public-tool-loop-v2"
FORMAT_RULE = (
    "Return exactly one JSON object per turn: no XML, no array, no Markdown/code fences, "
    "no explanatory text outside the object, no duplicate keys, and no NaN/Infinity. "
    "A tool object has exactly tool and arguments; arguments is an object, never a list. "
    "Use one of the complete examples, replacing only argument values for your task. "
    "Use final only for a nonempty submission summary; submission is not independent acceptance."
)
EXAMPLES = (
    {"tool": "list_files", "arguments": {}},
    {"tool": "read_file", "arguments": {"path": "relative/path.py"}},
    {"tool": "write_file", "arguments": {"path": "relative/path.py", "content": "print('example')\n"}},
    {"tool": "search", "arguments": {"text": "literal text"}},
    {"tool": "command", "arguments": {"argv": ["python", "-m", "public_test"], "timeout_s": 30}},
    {"tool": "execute_code", "arguments": {"language": "python", "code": "print('example')", "timeout_s": 30}},
    {"final": "Describe the changes submitted and the public checks actually run."},
)
SYSTEM_INSTRUCTIONS = (TOOLS_PROMPT + "\n" + FORMAT_RULE + "\nComplete legal response examples:\n"
                       + "\n".join(json.dumps(example, ensure_ascii=False) for example in EXAMPLES))


def _invalid_constant(_value):
    raise ProtocolError("nonfinite public response value")


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ProtocolError("nonfinite public response value")
    return value


def parse_response(text):
    """Validate final separately, delegating every tool schema to parse_action."""
    if not isinstance(text, str) or len(text.encode()) > 1_048_576:
        raise ProtocolError("invalid or oversized public response")
    value = json.loads(text, object_pairs_hook=_unique_object,
                       parse_constant=_invalid_constant, parse_float=_finite_float)
    if not isinstance(value, dict):
        raise ProtocolError("one JSON object required; no XML or array")
    if "final" in value:
        if set(value) != {"final"} or not isinstance(value["final"], str) or not value["final"].strip():
            raise ProtocolError("final requires exactly one nonempty string summary")
        return value
    return parse_action(text)


def _remaining_budget(services):
    reader = getattr(services, "budget_remaining", None)
    if not callable(reader):
        return None
    model = getattr(services, "model", None)
    value = reader(model) if model is not None else reader()
    if not isinstance(value, dict):
        return None
    # Only the four declared numerical fields can enter protocol feedback.
    return {key: value[key] for key in ("requests", "tokens", "seconds", "cny")
            if key in value and type(value[key]) in (int, float)
            and math.isfinite(value[key]) and value[key] >= 0}


def _error_feedback(services, error, *, phase):
    return {"public_tool_error": type(error).__name__, "detail": str(error)[:500],
            "protocol_version": TOOL_LOOP_VERSION, "error_phase": phase,
            "required_response": FORMAT_RULE, "valid_examples": EXAMPLES,
            "remaining_budget": _remaining_budget(services),
            "budget_rule": "The failed inference still consumes its original model/request budget. No budget reset or additional retry allowance.",
            "tool_dispatched": phase != "response_validation"}


def _append(messages, value):
    messages.append({"role": "user", "content": json.dumps(value, ensure_ascii=False, allow_nan=False)})


def persistent_tool_loop(services, prompt, messages, *, system="", max_turns=10):
    """Use at most max_turns ordinary model calls; no automatic format retry.

    Each correction is just feedback in the next already-budgeted turn. Only
    the six explicit parse_action/dispatch_action tools can execute. Provider
    tool_calls and XML-looking text are not tool transports in this loop.
    """
    if type(max_turns) is not int or max_turns < 1 or not isinstance(messages, list):
        raise ProtocolError("positive turn limit and checkpoint-owned transcript list required")
    if not messages:
        messages.append({"role": "system", "content": SYSTEM_INSTRUCTIONS + system})
    messages.append({"role": "user", "content": prompt})
    for _ in range(max_turns):
        reply = services.complete(messages)
        text = reply.get("text") if isinstance(reply, dict) else None
        # Retain the received text before parsing/dispatch. No provider-level
        # tool_calls are inspected or invoked, even if present in reply.
        messages.append({"role": "assistant", "content": text if isinstance(text, str) else ""})
        if isinstance(reply, dict) and reply.get("length_stop"):
            return {"status": "failed", "summary": "length_stop"}
        if isinstance(reply, dict) and reply.get("vendor_error"):
            _append(messages, {"provider_error": "vendor_error",
                "detail": "The provider rejected this request. It was not billed.",
                "automatic_retry": False, "protocol_version": TOOL_LOOP_VERSION})
            continue
        if text == "":
            _append(messages, {"public_tool_error": "empty_assistant_turn",
                "detail": "The assistant message was empty.",
                "protocol_version": TOOL_LOOP_VERSION, "error_phase": "response_validation",
                "required_response": FORMAT_RULE, "automatic_retry": False})
            continue
        coerced = coerce_public_action(text)
        if coerced:
            text = coerced
        try:
            action = parse_response(text)
        except (ValueError, TypeError, KeyError) as error:
            _append(messages, _error_feedback(services, error, phase="response_validation"))
            continue
        if set(action) == {"final"}:
            return {"status": "completed", "summary": action["final"]}
        try:
            result = dispatch_action(services, action)
        except (ValueError, TypeError, KeyError) as error:
            _append(messages, _error_feedback(services, error, phase="service_argument_or_operation_validation"))
            continue
        except BaseException as error:
            _append(messages, {"public_tool_interrupted": type(error).__name__, "tool": action["tool"],
                "status": "unconfirmed", "automatic_retry": False,
                "instruction": "Execution stopped. Do not assume this operation failed or repeat it without confirming its recorded side effects."})
            raise
        try:
            _append(messages, {"tool": action["tool"], "result": result})
        except (ValueError, TypeError) as error:
            _append(messages, {"public_tool_result_unserializable": type(error).__name__, "tool": action["tool"],
                "status": "unconfirmed", "automatic_retry": False})
            raise
    return {"status": "failed", "summary": "public tool turn cap reached without a valid final submission"}
