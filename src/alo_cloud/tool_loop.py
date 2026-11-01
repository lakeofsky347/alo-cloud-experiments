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
    _unique_object, coerce_actions, dispatch_action, parse_action,
)
from .public_tools import TOOLS_PROMPT, WORK_ORDER
from .errors import ProtocolError


TOOL_LOOP_VERSION = "tolerant-public-tool-loop-r2"
FORMAT_RULE = (
    "Prefer exactly one JSON object per turn: no array, no duplicate keys, and no NaN/Infinity. "
    "A tool object has tool and arguments; arguments is an object, never a list. "
    "A vendor tool tag with the same fields is accepted. Unknown extra arguments are ignored. "
    "Use final only for a nonempty submission summary; submission is not independent acceptance. "
    + WORK_ORDER
)
EXAMPLES = (
    {"tool": "list_files", "arguments": {}},
    {"tool": "read_file", "arguments": {"path": "relative/path.py"}},
    {"tool": "write_file", "arguments": {"path": "relative/path.py", "content": "print('example')\n"}},
    {"tool": "edit_file", "arguments": {"path": "relative/path.py", "old_str": "old", "new_str": "new"}},
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


LENGTH_STOP_NUDGE = "Your output budget was used by reasoning. Reply with ONE short action."
EMPTY_DIFF_NUDGE = "You have not changed any file. Make your best fix now with edit_file."


def _note_transport(services, kind):
    counts = getattr(services, "transport_counts", None)
    if not isinstance(counts, dict):
        return
    model = getattr(services, "model", None) or "unknown"
    counts.setdefault(model, []).append(kind)


def _requests_left(services):
    remaining = _remaining_budget(services) or {}
    left = remaining.get("requests")
    limits = getattr(services, "limits", None)
    cap = limits.get("requests") if isinstance(limits, dict) else None
    if isinstance(left, bool) or not isinstance(left, (int, float)):
        return None
    if isinstance(cap, bool) or not isinstance(cap, (int, float)):
        return int(left), None
    return int(left), int(cap)


def _diff_empty(services):
    reader = getattr(services, "diff_nonempty", None)
    if callable(reader):
        try:
            return not bool(reader())
        except Exception:
            return False
    return False


def _budget_line(services):
    pair = _requests_left(services)
    if pair is None:
        return None, None
    left, cap = pair
    line = f"requests left: {left}/{cap}" if cap is not None else f"requests left: {left}"
    nudge = EMPTY_DIFF_NUDGE if cap is not None and left <= 3 and _diff_empty(services) else None
    return line, nudge


def _append(messages, value, services=None):
    line, nudge = _budget_line(services) if services is not None else (None, None)
    if isinstance(value, str):
        text = value
        if line:
            text = text + "\n" + line
        if nudge:
            text = text + "\n" + nudge
    else:
        if line or nudge:
            value = dict(value)
            if line:
                value["requests_left"] = line
            if nudge:
                value["empty_diff_nudge"] = nudge
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    messages.append({"role": "user", "content": text})


def _length_stops(services):
    count = getattr(services, "length_stops", 0)
    return count if type(count) is int else 0


def _bump_length_stop(services):
    current = _length_stops(services)
    try:
        services.length_stops = current + 1
    except Exception:
        return current + 1
    return current + 1


def _finish(status, summary, termination, services):
    return {"status": status, "summary": summary, "termination": termination,
            "length_stops": _length_stops(services)}


def persistent_tool_loop(services, prompt, messages, *, system="", max_turns=10):
    """Use at most max_turns model calls. A format error is feedback, not a tool call.

    An empty ``finish_reason=length`` turn does not end the loop: the model is
    told to answer with one short action, and that next call counts as a request.
    Provider ``tool_calls`` fields are ignored. Only ``dispatch_action`` runs tools.
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
        if isinstance(reply, dict) and reply.get("length_stop") and not (isinstance(text, str) and text.strip()):
            _bump_length_stop(services)
            _append(messages, LENGTH_STOP_NUDGE, services)
            continue
        if isinstance(reply, dict) and reply.get("vendor_error"):
            _note_transport(services, "rejected")
            _append(messages, {"provider_error": "vendor_error",
                "detail": "The provider rejected this request. It was not billed.",
                "automatic_retry": False, "protocol_version": TOOL_LOOP_VERSION}, services)
            continue
        if text == "":
            _note_transport(services, "rejected")
            _append(messages, {"public_tool_error": "empty_assistant_turn",
                "detail": "The assistant message was empty.",
                "protocol_version": TOOL_LOOP_VERSION, "error_phase": "response_validation",
                "required_response": FORMAT_RULE, "automatic_retry": False}, services)
            continue
        action = None
        extra = 0
        transport = "rejected"
        try:
            action = parse_response(text)
            transport = "strict"
        except (ValueError, TypeError, KeyError):
            found = coerce_actions(text) if isinstance(text, str) else None
            if found and found.get("actions"):
                action = found["actions"][0]
                extra = len(found["actions"]) - 1
                transport = "coerced:" + str(found.get("variant") or "unknown")
        _note_transport(services, transport)
        if action is None:
            _append(messages, _error_feedback(services, ProtocolError("unparsed public action"),
                                              phase="response_validation"), services)
            continue
        if set(action) <= {"final"} or (set(action) == {"final"}):
            if isinstance(action.get("final"), str) and action["final"].strip():
                return _finish("completed", action["final"].strip(), "final", services)
            _append(messages, _error_feedback(services, ProtocolError("final requires exactly one nonempty string summary"),
                                              phase="response_validation"), services)
            continue
        if "tool" not in action:
            _append(messages, _error_feedback(services, ProtocolError("one tool and arguments required"),
                                              phase="response_validation"), services)
            continue
        try:
            result = dispatch_action(services, action)
        except (ValueError, TypeError, KeyError) as error:
            _append(messages, _error_feedback(services, error, phase="service_argument_or_operation_validation"), services)
            continue
        except BaseException as error:
            _append(messages, {"public_tool_interrupted": type(error).__name__, "tool": action["tool"],
                "status": "unconfirmed", "automatic_retry": False,
                "instruction": "Execution stopped. Do not assume this operation failed or repeat it without confirming its recorded side effects."},
                services)
            raise
        observation = {"tool": action["tool"], "result": result}
        if extra:
            observation["ignored_extra_actions"] = extra
            observation["note"] = f"ignored {extra} extra actions"
        try:
            json.dumps(observation, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError) as error:
            _append(messages, {"public_tool_result_unserializable": type(error).__name__, "tool": action["tool"],
                "status": "unconfirmed", "automatic_retry": False}, services)
            raise
        _append(messages, observation, services)
    return _finish("failed", "public tool turn cap reached without a valid final submission", "turn_cap", services)
