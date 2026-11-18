"""Shared public action protocol. Tools delegate only to an injected sandbox.

No tool here launches a subprocess or executes code on the controller host.
The parser accepts the strict JSON object and the vendor shapes the round-1
models actually emitted. Unknown extra arguments are dropped with a warning.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import re

from .errors import ProtocolError, BaselineUnavailable
from .native_bridge import ServiceCodeExecutor
from .policy import MUTATING_TOOLS
from .review_isolation import role_is_review, run_review_command

WORK_ORDER = (
    "Work in this order: localize the bug, edit the source, then verify. "
    "Do not use the network."
)

TOOLS_PROMPT = """You maintain the current persistent multi-file workspace.
""" + WORK_ORDER + """
Reply with one action per turn. The preferred form is one JSON object:
{"final":"summary"},
{"tool":"list_files","arguments":{"path":"optional/subdir"}},
{"tool":"read_file","arguments":{"path":"relative/path","start_line":1,"end_line":200}},
{"tool":"write_file","arguments":{"path":"relative/path","content":"text","append":false}},
{"tool":"edit_file","arguments":{"path":"relative/path","old_str":"exact text","new_str":"replacement"}},
{"tool":"search","arguments":{"text":"literal text"}},
{"tool":"command","arguments":{"argv":["python3","-m","pytest","path"]}}, or
{"tool":"execute_code","arguments":{"language":"python","code":"print('example')"}}.
Only the frozen editable paths are writable. Commands use the common restricted
backend. Public feedback is available; private final scores are not. Preserve
this run's existing code, data and errors across stages; never reset its data
from a reference. A final response means submitted, not independently accepted.
edit_file replaces one exact unique match. write_file append=true adds to an
existing file. search scans full file contents. read_file pages with start_line
and end_line when the reply says the file was truncated.
"""
TOOLS = ("list_files", "read_file", "write_file", "edit_file", "search", "command", "execute_code")

PB_BUILD_TOOLS = (*TOOLS, "pb_build")

ACTION_LANGUAGE = "public_tool"

ACTION_INSTRUCTIONS = (
    " Public host actions use the same tools as every other arm. "
    + WORK_ORDER +
    " Emit one public_tool action: list_files({path?}), read_file({path,start_line?,end_line?}), "
    "write_file({path,content,append?}), edit_file({path,old_str,new_str}), "
    "search({text}), command({argv,timeout_s?}), execute_code({language,code,timeout_s?}). "
    "Use relative workspace paths; argv is a list. Each dispatched action consumes the "
    "common tool budget; execute_code consumes one command call. "
    "Prefer one JSON object. A vendor tool tag with the same fields is also read. "
    "Python/shell code blocks remain available."
)

REQUIRED = {
    "pb_build": set(),
    "list_files": set(),
    "read_file": {"path"},
    "write_file": {"path", "content"},
    "edit_file": {"path", "old_str", "new_str"},
    "search": {"text"},
    "command": {"argv"},
    "execute_code": {"language", "code"},
}

OPTIONAL = {
    "pb_build": set(),
    "list_files": {"path"},
    "read_file": {"start_line", "end_line"},
    "write_file": {"append"},
    "edit_file": set(),
    "search": set(),
    "command": {"timeout_s"},
    "execute_code": {"timeout_s"},
}

_TEXT_KEYS = {"path", "content", "text", "language", "code", "old_str", "new_str"}
_LINE_KEYS = {"start_line", "end_line"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate public action key")
        result[key] = value
    return result


def _normalize_arguments(name, arguments):
    """Drop unknown keys. Required keys and published types still fail closed."""
    if not isinstance(arguments, dict):
        raise ProtocolError("public action argument schema differs")
    if not REQUIRED[name] <= set(arguments):
        raise ProtocolError("public action argument schema differs")
    extra = sorted(set(arguments) - REQUIRED[name] - OPTIONAL[name])
    warnings = []
    if extra:
        warnings.append("ignored unknown arguments: " + ", ".join(extra))
        arguments = {key: value for key, value in arguments.items() if key not in extra}
    for key in _TEXT_KEYS & set(arguments):
        if not isinstance(arguments[key], str):
            raise ProtocolError("public action text argument required")
    if "argv" in arguments and (not isinstance(arguments["argv"], list)
                                  or any(not isinstance(item, str) for item in arguments["argv"])):
        raise ProtocolError("public action argv must be a string list")
    if "timeout_s" in arguments and (isinstance(arguments["timeout_s"], bool)
                                       or not isinstance(arguments["timeout_s"], (int, float))):
        raise ProtocolError("public action timeout must be numeric")
    if "append" in arguments and not isinstance(arguments["append"], bool):
        raise ProtocolError("public action append must be boolean")
    for key in _LINE_KEYS & set(arguments):
        if isinstance(arguments[key], bool) or type(arguments[key]) is not int or arguments[key] < 1:
            raise ProtocolError("public action line must be a positive integer")
    return arguments, warnings


def infer_tool_name(arguments):
    """Pick the tool when a vendor payload names parameters and not the tool."""
    if not isinstance(arguments, dict):
        return None
    keys = set(arguments)
    if "argv" in keys:
        return "command"
    if "code" in keys:
        return "execute_code"
    if "old_str" in keys and "new_str" in keys:
        return "edit_file"
    if "path" in keys and "content" in keys:
        return "write_file"
    if "path" in keys:
        return "read_file"
    if "text" in keys:
        return "search"
    return None


def parse_action(text, *, allowed_tools=TOOLS):
    """One JSON action. Malformed messages do not consume a service tool call.

    Unknown argument names are removed and reported in ``warnings``. Service
    argument, path, permission and budget checks stay with the services.
    """
    if not isinstance(text, str) or len(text.encode()) > 1_048_576:
        raise ProtocolError("invalid or oversized public action")
    action = json.loads(text, object_pairs_hook=_unique_object,
                        parse_constant=lambda value: (_ for _ in ()).throw(ProtocolError("nonfinite public action value")))
    if not isinstance(action, dict) or "tool" not in action or "arguments" not in action:
        raise ProtocolError("one tool and arguments required")
    name, arguments = action["tool"], action["arguments"]
    if not isinstance(name, str) or name not in allowed_tools:
        raise ProtocolError("unknown public tool")
    arguments, warnings = _normalize_arguments(name, arguments)
    result = {"tool": name, "arguments": arguments}
    if warnings:
        result["warnings"] = warnings
    return result


def _argument_value(key, raw):
    text = raw.strip()
    if key in {"argv", "arguments"} or (key == "timeout_s") or key in _LINE_KEYS or key == "append":
        value = json.loads(text)
        return value
    return text


_FUNCTION_OPEN = re.compile(r"<function=([A-Za-z0-9_.|]+)>", re.IGNORECASE)
_TOOL_OPEN = re.compile(r"<tool=([A-Za-z0-9_]+)>", re.IGNORECASE)
_PARAM_OPEN = re.compile(r"<parameter(?:=([A-Za-z0-9_]+))?>", re.IGNORECASE)
_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)
_DSML_INVOKE = re.compile(
    r"<(?:｜|\|)\s*DSML\s*(?:｜|\|)\s*invoke\b([^>]*)>(.*?)(?=<(?:｜|\|)\s*DSML\s*(?:｜|\|)\s*invoke\b|$)",
    re.DOTALL | re.IGNORECASE,
)
_DSML_PARAM = re.compile(
    r"<(?:｜|\|)\s*DSML\s*(?:｜|\|)\s*parameter\b([^>]*)>(.*?)(?=<(?:｜|\|)\s*DSML\s*(?:｜|\|)\s*parameter\b|</(?:｜|\|)\s*DSML\s*(?:｜|\|)\s*invoke>|$)",
    re.DOTALL | re.IGNORECASE,
)
_ATTR = re.compile(r"([A-Za-z0-9_]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))")


def _slice_until(body, end_tag):
    lower = body.lower()
    at = lower.find(end_tag.lower())
    if at >= 0:
        return body[:at]
    return body


def _params_from_body(body):
    """Read ``<parameter=key>value`` tags. A missing close tag ends at the next tag."""
    arguments = {}
    positions = list(_PARAM_OPEN.finditer(body))
    if not positions:
        return arguments, False
    for index, match in enumerate(positions):
        start = match.end()
        end = positions[index + 1].start() if index + 1 < len(positions) else len(body)
        chunk = body[start:end]
        chunk = _slice_until(chunk, "</parameter>")
        key = match.group(1) or "arguments"
        chunk = re.split(r"</", chunk, maxsplit=1)[0]
        arguments[key] = _argument_value(key, chunk)
    return arguments, True


def _attrs(text):
    found = {}
    for match in _ATTR.finditer(text or ""):
        found[match.group(1)] = match.group(2) if match.group(2) is not None else (
            match.group(3) if match.group(3) is not None else match.group(4))
    return found


def _action_from_parts(name, arguments, allowed_tools):
    if name in {"tool", "public_tool"}:
        name = arguments.pop("tool", None) or arguments.pop("name", None)
    if not name:
        name = infer_tool_name(arguments)
    if name == "final" or (isinstance(arguments, dict) and set(arguments) == {"final"}):
        summary = arguments.get("final") if isinstance(arguments, dict) else None
        if isinstance(summary, str) and summary.strip():
            return {"final": summary.strip()}
        return None
    if not isinstance(name, str) or name not in allowed_tools:
        return None
    normalized, warnings = _normalize_arguments(name, arguments)
    action = {"tool": name, "arguments": normalized}
    if warnings:
        action["warnings"] = warnings
    return action


def _load_arguments_blob(arguments):
    """A single ``arguments`` parameter may itself be a JSON object."""
    if set(arguments) == {"arguments"} or (len(arguments) == 1 and "arguments" in arguments):
        raw = arguments["arguments"]
        if isinstance(raw, str):
            parsed = json.loads(raw) if raw.strip().startswith("{") or raw.strip().startswith("[") else None
            if isinstance(parsed, dict):
                return parsed
        if isinstance(raw, dict):
            return raw
    return arguments


def _actions_from_function_bodies(text, allowed_tools):
    opens = list(_FUNCTION_OPEN.finditer(text))
    actions = []
    if not opens:
        return actions
    for index, match in enumerate(opens):
        start = match.end()
        end = opens[index + 1].start() if index + 1 < len(opens) else len(text)
        body = _slice_until(text[start:end], "</function>")
        name = match.group(1)
        try:
            arguments, saw_param = _params_from_body(body)
            arguments = _load_arguments_blob(arguments)
            if not saw_param and body.strip().startswith("{"):
                parsed = json.loads(body.strip())
                if isinstance(parsed, dict):
                    arguments = parsed
            action = _action_from_parts(name, arguments if isinstance(arguments, dict) else {}, allowed_tools)
        except (ProtocolError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if action:
            actions.append(action)
    return actions


def _actions_from_tool_tags(text, allowed_tools):
    opens = list(_TOOL_OPEN.finditer(text))
    actions = []
    for index, match in enumerate(opens):
        start = match.end()
        end = opens[index + 1].start() if index + 1 < len(opens) else len(text)
        body = _slice_until(text[start:end], "</tool>")
        name = match.group(1)
        try:
            arguments, saw_param = _params_from_body(body)
            arguments = _load_arguments_blob(arguments)
            if not saw_param and body.strip().startswith("{"):
                parsed = json.loads(body.strip())
                if not isinstance(parsed, dict):
                    continue
                arguments = parsed
            elif not saw_param:
                arguments = {}
            action = _action_from_parts(name, arguments if isinstance(arguments, dict) else {}, allowed_tools)
        except (ProtocolError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if action:
            actions.append(action)
    return actions


def _actions_from_dsml(text, allowed_tools):
    if "invoke" not in text.lower() or "dsml" not in text.lower():
        return []
    actions = []
    for match in _DSML_INVOKE.finditer(text):
        attrs = _attrs(match.group(1))
        name = attrs.get("name")
        arguments = {}
        try:
            for param in _DSML_PARAM.finditer(match.group(2)):
                pattrs = _attrs(param.group(1))
                key = pattrs.get("name") or "arguments"
                raw = _slice_until(param.group(2), "</")
                string_flag = (pattrs.get("string") or "").lower()
                if string_flag == "false":
                    arguments[key] = json.loads(raw.strip())
                elif key in {"argv", "timeout_s", "append", "start_line", "end_line", "arguments"}:
                    arguments[key] = json.loads(raw.strip())
                else:
                    arguments[key] = raw.strip()
            arguments = _load_arguments_blob(arguments)
            action = _action_from_parts(name, arguments if isinstance(arguments, dict) else {}, allowed_tools)
        except (ProtocolError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if action:
            actions.append(action)
    return actions


def _tool_call_shape(value):
    """A public tool call or the tool result the loop stored after it."""
    if not isinstance(value, dict):
        return False
    if "tool" in value and ("arguments" in value or "result" in value):
        return True
    return "final" in value and set(value) <= {"final"}


def _decode_object_at(text, start):
    try:
        value, end = json.JSONDecoder(object_pairs_hook=_unique_object).raw_decode(text, start)
    except (json.JSONDecodeError, ProtocolError, ValueError):
        return None, None
    if not isinstance(value, dict):
        return None, None
    return value, end


def leading_json_object(text):
    """Tool JSON at the start of ``text``, or a later object only if it is a tool call.

    A reply that is one JSON object plus stray ``<system>`` text still parses.
    A ``{`` buried in prose is used only when that object has ``tool`` and
    ``arguments`` or ``result``, or a lone ``final``. Other JSON in prose is ignored.
    """
    if not isinstance(text, str):
        return None
    stripped = text.lstrip()
    if stripped.startswith("{"):
        value, end = _decode_object_at(stripped, 0)
        if value is not None and (_tool_call_shape(value) or stripped[end:].strip() == ""):
            return value
    for value in _json_objects(text):
        if _tool_call_shape(value):
            return value
    return None


def _json_objects(text):
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    index = 0
    found = []
    while True:
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            found.append(value)
        index = end
    return found


def _action_from_json_object(value, allowed_tools):
    if not isinstance(value, dict):
        return None
    if "final" in value and set(value) <= {"final"}:
        if isinstance(value.get("final"), str) and value["final"].strip():
            return {"final": value["final"].strip()}
        return None
    arguments = value.get("arguments")
    name = value.get("tool")
    if isinstance(arguments, dict) or name:
        if not isinstance(arguments, dict):
            arguments = {key: item for key, item in value.items() if key != "tool"}
        return _action_from_parts(name, arguments, allowed_tools)
    inferred = infer_tool_name(value)
    if inferred:
        return _action_from_parts(inferred, value, allowed_tools)
    return None


def _actions_from_fences(text, allowed_tools):
    actions = []
    for language, body in _FENCE.findall(text):
        if language.lower() not in {"json", "public_tool", ""}:
            continue
        try:
            for value in _json_objects(body):
                action = _action_from_json_object(value, allowed_tools)
                if action:
                    actions.append(action)
        except (ProtocolError, ValueError, TypeError, KeyError):
            continue
    return actions


def _actions_from_prose_json(text, allowed_tools):
    actions = []
    try:
        for value in _json_objects(text):
            action = _action_from_json_object(value, allowed_tools)
            if action:
                actions.append(action)
    except (ProtocolError, ValueError, TypeError, KeyError):
        return []
    return actions


def coerce_actions(text, *, allowed_tools=TOOLS):
    """Parse every public action in one reply.

    Returns ``{"actions", "variant", "warnings"}`` or None when nothing in the
    reply matches a known shape. The caller executes the first action.
    """
    if not isinstance(text, str) or len(text.encode()) > 1_048_576:
        return None
    variant = None
    actions = []
    if "<｜" in text or "<|" in text and "dsml" in text.lower():
        actions = _actions_from_dsml(text, allowed_tools)
        variant = "dsml"
    if not actions and "<function=" in text.lower():
        actions = _actions_from_function_bodies(text, allowed_tools)
        if actions and any(name in text for name in ("<function=tool>", "<function=public_tool>",
                                                      "<function=tool ", "<function=public_tool ")):
            variant = "function_tool"
        elif actions and "<parameter=arguments>" in text.lower():
            variant = "parameter_arguments"
        elif actions:
            variant = "tool_call"
    if not actions and "<tool=" in text.lower():
        actions = _actions_from_tool_tags(text, allowed_tools)
        variant = "tool_tag"
    if not actions and "```" in text:
        actions = _actions_from_fences(text, allowed_tools)
        variant = "json_fence"
    if not actions and "{" in text:
        stripped = text.strip()
        actions = _actions_from_prose_json(text, allowed_tools)
        if actions:
            variant = "json" if stripped.startswith("{") and stripped.endswith("}") else "prose_json"
    if not actions:
        return None
    warnings = []
    for action in actions:
        warnings.extend(action.get("warnings") or [])
    return {"actions": actions, "variant": variant, "warnings": warnings}


def coerce_public_action(text):
    """First vendor action as the strict public JSON object, or None.

    Kept for callers that only need one action. Extra actions stay available
    through ``coerce_actions``.
    """
    found = coerce_actions(text)
    if not found:
        return None
    action = found["actions"][0]
    if "final" in action:
        return json.dumps({"final": action["final"]}, ensure_ascii=False)
    payload = {"tool": action["tool"], "arguments": action["arguments"]}
    return json.dumps(payload, ensure_ascii=False)


def transport_summary(counts):
    """Per-model rates. ``counts`` maps a model to a list of transport labels."""
    summary = {}
    if not isinstance(counts, dict):
        return summary
    for model, events in counts.items():
        if not isinstance(events, list):
            continue
        total = len(events)
        strict = sum(1 for item in events if item == "strict")
        rejected = sum(1 for item in events if item == "rejected")
        coerced = total - strict - rejected
        summary[model] = {
            "n": total,
            "strict": strict,
            "coerced": coerced,
            "rejected": rejected,
            "strict_rate": (strict / total) if total else None,
            "coerced_rate": (coerced / total) if total else None,
            "rejected_rate": (rejected / total) if total else None,
        }
    return summary


class PublicActionExtractor:
    """Markdown fences, vendor tags, and one JSON object, as a public_tool block."""

    def extract_code_blocks(self, message):
        text = message if isinstance(message, str) else ""
        if not isinstance(text, str):
            content = getattr(message, "content", "")
            text = content if isinstance(content, str) else ""
        if "[action_executed]" in text:
            return []
        found = coerce_actions(text)
        if found and found["actions"] and "tool" in found["actions"][0]:
            payload = {"tool": found["actions"][0]["tool"], "arguments": found["actions"][0]["arguments"]}
            return [SimpleNamespace(language=ACTION_LANGUAGE, code=json.dumps(payload, ensure_ascii=False))]
        blocks = [SimpleNamespace(language=lang or "", code=code)
                  for lang, code in _FENCE.findall(text)
                  if (lang or "").lower() not in {"json", "public_tool"}]
        return blocks


def dispatch_action(services, action):
    """Explicit dispatch. Model strings never select arbitrary attributes.

    When ``services.role_tools`` is set, the role contract is enforced here.
    A review role's ``command`` runs on the pre-review snapshot. It is not
    given the live implementation tree.
    """
    warnings = list(action.get("warnings") or [])
    arguments = dict(action["arguments"])
    clamp = getattr(services, "clamp_timeout", None)
    if "timeout_s" in arguments and callable(clamp):
        arguments["timeout_s"], timeout_warnings = clamp(arguments["timeout_s"])
        warnings.extend(timeout_warnings)
    allowed_tools = PB_BUILD_TOOLS if getattr(services, "pb_build_enabled", False) else TOOLS
    if action["tool"] not in allowed_tools:
        raise ProtocolError("unknown public tool")
    role_tools = getattr(services, "role_tools", None)
    if role_tools is not None and action["tool"] not in set(role_tools):
        raise ProtocolError("tool_outside_role_contract")
    if role_is_review(role_tools) and action["tool"] in MUTATING_TOOLS:
        raise ProtocolError("review_role_cannot_mutate")
    if action["tool"] == "command" and role_is_review(role_tools):
        reviewer = getattr(services, "review_command", None)
        if callable(reviewer):
            result = reviewer(**arguments)
        else:
            result = run_review_command(services, arguments.get("argv"), arguments.get("timeout_s", 30))
    else:
        result = getattr(services, action["tool"])(**arguments)
    if warnings and isinstance(result, dict):
        result = dict(result)
        result.setdefault("warnings", [])
        result["warnings"] = list(result["warnings"]) + warnings
    elif warnings and isinstance(result, str):
        result = result + "\n" + "; ".join(warnings)
    return result


class PublicToolCodeExecutor(ServiceCodeExecutor):
    """Opt-in host action bridge, retaining the native executor protocol."""
    def execute_code_blocks(self, code_blocks):
        blocks = list(code_blocks)
        actions = [block for block in blocks if block.language.lower() == ACTION_LANGUAGE]
        if not actions:
            return super().execute_code_blocks(blocks)
        try:
            if len(blocks) != 1:
                raise ProtocolError("one public action block required")
            action = parse_action(actions[0].code)
            result = dispatch_action(self.services, action)
            split = getattr(self.services, "g3_request_split", None)
            if isinstance(split, dict):
                split["dispatches"] = split.get("dispatches", 0) + 1
            exit_code = 0
            if action["tool"] in {"command", "execute_code"}:
                if not isinstance(result, dict):
                    raise BaselineUnavailable("tool_result_missing_exit_code")
                returncode = result.get("returncode")
                if result.get("timed_out"):
                    if returncode is not None and type(returncode) is not int:
                        raise BaselineUnavailable("tool_result_missing_exit_code")
                    exit_code = returncode or 124
                elif type(returncode) is not int:
                    raise BaselineUnavailable("tool_result_missing_exit_code")
                else:
                    exit_code = returncode
            return SimpleNamespace(exit_code=exit_code, output=json.dumps({"tool": action["tool"], "result": result}, ensure_ascii=False))
        except (ValueError, TypeError, KeyError) as exc:
            return SimpleNamespace(exit_code=2, output=json.dumps({"public_tool_error": type(exc).__name__,
                                   "detail": str(exc)[:500]}, ensure_ascii=False))
