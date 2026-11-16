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


def _budget_line(services, node=None):
    pair = _requests_left(services)
    if pair is None:
        if node is None:
            return None, None
        return _node_line(node, None), None
    left, cap = pair
    line = f"requests left: {left}/{cap}" if cap is not None else f"requests left: {left}"
    nudge = EMPTY_DIFF_NUDGE if cap is not None and left <= 3 and _diff_empty(services) else None
    if node is not None:
        line = _node_line(node, line)
    return line, nudge


def _node_line(node, unit_line):
    """This node's own remaining requests, then the unit line it used to see."""
    left = max(node["cap"] - node["used"], 0)
    line = f"requests left for your node {node['node']}: {left}/{node['cap']}"
    if unit_line:
        line += " (" + unit_line.replace("requests left: ", "unit: ", 1) + ")"
    return line


def _node_notice(messages, node):
    """The node budget notice text, once, when this node has ``notice_at`` or fewer requests left.

    It rides in the next observation (like ``empty_diff_nudge``), so no extra
    message role is sent to the provider.
    """
    template = node.get("notice")
    if not template or node.get("sent"):
        return None
    left = node["cap"] - node["used"]
    if left > node["notice_at"]:
        return None
    prefix = template.split("{", 1)[0]
    if any(prefix in str(m.get("content", "")) for m in messages if isinstance(m, dict)):
        node["sent"] = True   # already in a resumed transcript
        return None
    node["sent"] = True
    node["sent_at_left"] = left
    return template.format(left=left, node=node["node"], cap=node["cap"])


def _append(messages, value, services=None, node=None):
    line, nudge = _budget_line(services, node) if services is not None else (
        (_node_line(node, None), None) if node is not None else (None, None))
    notice = _node_notice(messages, node) if node is not None else None
    if isinstance(value, str):
        text = value
        if line:
            text = text + "\n" + line
        if nudge:
            text = text + "\n" + nudge
        if notice:
            text = text + "\n" + notice
    else:
        if line or nudge or notice:
            value = dict(value)
            if line:
                value["requests_left"] = line
            if nudge:
                value["empty_diff_nudge"] = nudge
            if notice:
                value["node_budget_notice"] = notice
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


def persistent_tool_loop(services, prompt, messages, *, system="", max_turns=10, node_budget=None):
    """Use at most max_turns model calls. A format error is feedback, not a tool call.

    An empty ``finish_reason=length`` turn, and an empty ``finish_reason=stop``
    turn with no tool call, do not end the loop. The model is told to answer
    with one short action, and that next call counts as a request. A provider
    that was not reached before the request was sent gets the same continuation
    and does not end the unit. Provider ``tool_calls`` fields are ignored.
    Only ``dispatch_action`` runs tools.

    ``node_budget`` (ProgramBench node-budget-v1, opt-in) is a dict with
    ``node``, ``notice`` and ``notice_at``. When given, every observation shows
    this node's own remaining requests (``max_turns`` is the node cap) before
    the unit line, and ``notice`` is added once, as ``node_budget_notice`` in
    the first observation that shows ``notice_at`` or fewer node requests left
    (in the prompt if the cap itself is that small). With ``node_budget=None``
    every message is byte-identical to before.
    """
    if type(max_turns) is not int or max_turns < 1 or not isinstance(messages, list):
        raise ProtocolError("positive turn limit and checkpoint-owned transcript list required")
    if not messages:
        messages.append({"role": "system", "content": SYSTEM_INSTRUCTIONS + system})
    messages.append({"role": "user", "content": prompt})
    node = None
    if node_budget is not None:
        node = node_budget
        node.update(cap=max_turns, used=0)
        notice = _node_notice(messages, node)   # only when the node cap is <= notice_at
        if notice:
            messages[-1] = {"role": "user", "content": prompt + "\n\n" + notice}
    for turn in range(max_turns):
        reply = services.complete(messages)
        if node is not None:
            node["used"] = turn + 1
        text = reply.get("text") if isinstance(reply, dict) else None
        # Retain the received text before parsing/dispatch. No provider-level
        # tool_calls are inspected or invoked, even if present in reply.
        messages.append({"role": "assistant", "content": text if isinstance(text, str) else ""})
        if isinstance(reply, dict) and reply.get("pro_share_cap"):
            return _finish("failed", "pro_share_cap", "budget", services)
        if isinstance(reply, dict) and reply.get("provider_not_reached"):
            _append(messages, "The provider was not reached before the request was sent. "
                    "Reply with ONE short action.", services, node)
            continue
        if isinstance(reply, dict) and reply.get("vendor_error"):
            _note_transport(services, "rejected")
            _append(messages, {"provider_error": "vendor_error",
                "detail": "The provider rejected this request. It was not billed.",
                "automatic_retry": False, "protocol_version": TOOL_LOOP_VERSION}, services, node)
            continue
        empty_visible = not (isinstance(text, str) and text.strip())
        finish = reply.get("finish_reason") if isinstance(reply, dict) else None
        if isinstance(reply, dict) and empty_visible and (reply.get("length_stop") or finish == "stop"):
            if reply.get("length_stop"):
                _bump_length_stop(services)
            _append(messages, LENGTH_STOP_NUDGE, services, node)
            continue
        if text == "":
            _note_transport(services, "rejected")
            _append(messages, {"public_tool_error": "empty_assistant_turn",
                "detail": "The assistant message was empty.",
                "protocol_version": TOOL_LOOP_VERSION, "error_phase": "response_validation",
                "required_response": FORMAT_RULE, "automatic_retry": False}, services, node)
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
                                              phase="response_validation"), services, node)
            continue
        if set(action) <= {"final"} or (set(action) == {"final"}):
            if isinstance(action.get("final"), str) and action["final"].strip():
                return _finish("completed", action["final"].strip(), "final", services)
            _append(messages, _error_feedback(services, ProtocolError("final requires exactly one nonempty string summary"),
                                              phase="response_validation"), services, node)
            continue
        if "tool" not in action:
            _append(messages, _error_feedback(services, ProtocolError("one tool and arguments required"),
                                              phase="response_validation"), services, node)
            continue
        try:
            result = dispatch_action(services, action)
        except (ValueError, TypeError, KeyError) as error:
            _append(messages, _error_feedback(services, error, phase="service_argument_or_operation_validation"), services, node)
            continue
        except BaseException as error:
            _append(messages, {"public_tool_interrupted": type(error).__name__, "tool": action["tool"],
                "status": "unconfirmed", "automatic_retry": False,
                "instruction": "Execution stopped. Do not assume this operation failed or repeat it without confirming its recorded side effects."},
                services, node)
            raise
        observation = {"tool": action["tool"], "result": result}
        if extra:
            observation["ignored_extra_actions"] = extra
            observation["note"] = f"ignored {extra} extra actions"
        try:
            json.dumps(observation, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError) as error:
            _append(messages, {"public_tool_result_unserializable": type(error).__name__, "tool": action["tool"],
                "status": "unconfirmed", "automatic_retry": False}, services, node)
            raise
        _append(messages, observation, services, node)
    return _finish("failed", "public tool turn cap reached without a valid final submission", "turn_cap", services)
