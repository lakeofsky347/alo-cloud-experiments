"""Strict shared public action protocol; tools delegate only to an injected sandbox.

No tool here launches a subprocess or executes code on the controller host.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import re

from .errors import ProtocolError, BaselineUnavailable
from .native_bridge import ServiceCodeExecutor

TOOLS_PROMPT = """You maintain the current persistent multi-file workspace.
Reply with exactly one JSON object per turn. Either {"final":"summary"}, or
{"tool":"list_files","arguments":{}},
{"tool":"read_file","arguments":{"path":"relative/path"}},
{"tool":"write_file","arguments":{"path":"relative/path","content":"text"}},
{"tool":"search","arguments":{"text":"literal text"}},
{"tool":"command","arguments":{"argv":["python","-m","public_test"]}}, or
{"tool":"execute_code","arguments":{"language":"python","code":"print('example')"}}.
Only the frozen editable paths are writable. Commands use the common restricted
backend. Public feedback is available; private final scores are not. Preserve
this run's existing code, data and errors across stages; never reset its data
from a reference. A final response means submitted, not independently accepted.
"""
TOOLS = ("list_files", "read_file", "write_file", "search", "command", "execute_code")

ACTION_LANGUAGE = "public_tool"

ACTION_INSTRUCTIONS = (
    " Public host actions are available through this adapter extension: emit exactly "
    "one public_tool fenced block containing one JSON object with exactly tool and arguments. "
    "Allowed actions: list_files({}), read_file({path}), write_file({path,content}), "
    "search({text}), command({argv,timeout_s?}), execute_code({language,code,timeout_s?}). "
    "Use relative workspace paths; argv is a list. Each dispatched action consumes the "
    "common tool budget; execute_code consumes one command call. Workspace returns "
    "JSON tool/result or public_tool_error feedback. No action arrays or multiple "
    "blocks with a public_tool action are accepted. Python/shell code blocks remain available."
)

REQUIRED = {"list_files": set(), "read_file": {"path"}, "write_file": {"path", "content"},
            "search": {"text"}, "command": {"argv"}, "execute_code": {"language", "code"}}

OPTIONAL = {name: ({"timeout_s"} if name in {"command", "execute_code"} else set()) for name in TOOLS}

def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate public action key")
        result[key] = value
    return result

def parse_action(text):
    """One JSON action; malformed messages do not consume a service tool call.

    Service argument/value, path, permission and budget validation remains with
    the same RunServices/BoundRoleServices used by the other architectures.
    """
    if not isinstance(text, str) or len(text.encode()) > 1_048_576:
        raise ProtocolError("invalid or oversized public action")
    action = json.loads(text, object_pairs_hook=_unique_object,
                        parse_constant=lambda value: (_ for _ in ()).throw(ProtocolError("nonfinite public action value")))
    if not isinstance(action, dict) or set(action) != {"tool", "arguments"}:
        raise ProtocolError("one tool and arguments required")
    name, arguments = action["tool"], action["arguments"]
    if not isinstance(name, str) or name not in TOOLS:
        raise ProtocolError("unknown public tool")
    if not isinstance(arguments, dict) or not REQUIRED[name] <= set(arguments) or set(arguments) - REQUIRED[name] - OPTIONAL[name]:
        raise ProtocolError("public action argument schema differs")
    # Types match the published public interface. Semantic limits (timeout,
    # language, content/search length and paths) are enforced by the services.
    for key in {"path", "content", "text", "language", "code"} & set(arguments):
        if not isinstance(arguments[key], str):
            raise ProtocolError("public action text argument required")
    if "argv" in arguments and (not isinstance(arguments["argv"], list)
                                  or any(not isinstance(v, str) for v in arguments["argv"])):
        raise ProtocolError("public action argv must be a string list")
    if "timeout_s" in arguments and (isinstance(arguments["timeout_s"], bool)
                                       or not isinstance(arguments["timeout_s"], (int, float))):
        raise ProtocolError("public action timeout must be numeric")
    return action

_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=([A-Za-z_]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM = re.compile(r"<parameter(?:=([A-Za-z0-9_]+))?>(.*?)</parameter>", re.DOTALL)
_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)


def _argument_value(key, raw):
    text = raw.strip()
    if key == "argv":
        value = json.loads(text)
        if not isinstance(value, list):
            raise ProtocolError("public action argv must be a string list")
        return value
    if key == "timeout_s":
        return json.loads(text)
    return text


def coerce_public_action(text):
    """Turn one vendor tool_call into the strict public JSON object.

    The live models emit ``<tool_call><function=name>`` instead of the JSON
    object the loop asks for. One such call, naming one of the six tools, is
    the same action. Two calls, or an unknown name, are left for the JSON parser
    to reject.
    """
    if not isinstance(text, str) or "<tool_call>" not in text:
        return None
    calls = list(_TOOL_CALL.finditer(text))
    if len(calls) != 1:
        return None
    name = calls[0].group(1)
    if name not in TOOLS:
        return None
    body = calls[0].group(2).strip()
    arguments = {}
    params = list(_PARAM.finditer(body))
    try:
        if len(params) == 1 and (params[0].group(1) in {None, "arguments"}):
            raw = params[0].group(2).strip()
            if raw:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    return None
                arguments = parsed
        elif params:
            for param in params:
                key = param.group(1) or "arguments"
                arguments[key] = _argument_value(key, param.group(2))
        elif body.startswith("{"):
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                return None
            arguments = parsed
        elif body:
            return None
        action = parse_action(json.dumps({"tool": name, "arguments": arguments}))
    except (ProtocolError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return json.dumps(action, ensure_ascii=False)


class PublicActionExtractor:
    """Markdown fences, plus one vendor tool_call as a public_tool block."""

    def extract_code_blocks(self, message):
        text = message if isinstance(message, str) else ""
        if not isinstance(text, str):
            content = getattr(message, "content", "")
            text = content if isinstance(content, str) else ""
        blocks = [SimpleNamespace(language=lang or "", code=code)
                  for lang, code in _FENCE.findall(text)]
        if blocks:
            return blocks
        coerced = coerce_public_action(text)
        if coerced:
            return [SimpleNamespace(language=ACTION_LANGUAGE, code=coerced)]
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parse_action(stripped)
            except (ProtocolError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                return []
            return [SimpleNamespace(language=ACTION_LANGUAGE, code=stripped)]
        return []


def dispatch_action(services, action):
    """Explicit six-entry dispatch; model strings never select arbitrary attrs."""
    dispatch = {"list_files": services.list_files, "read_file": services.read_file,
                "write_file": services.write_file, "search": services.search,
                "command": services.command, "execute_code": services.execute_code}
    return dispatch[action["tool"]](**action["arguments"])

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
            # Same public error shape/detail ceiling as ToolLoop. Resource and
            # uncertain-execution errors propagate; they never become success.
            return SimpleNamespace(exit_code=2, output=json.dumps({"public_tool_error": type(exc).__name__,
                                   "detail": str(exc)[:500]}, ensure_ascii=False))
