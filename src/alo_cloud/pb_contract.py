"""ProgramBench role contracts, public signal, and reference-binary guardrails.

The reference ``./executable`` is execute-only (mode 111). A black-box run
through its normal interface is allowed. Reading its bytes, disassembling it,
instrumenting it, using the network, or submitting a wrapper that calls or
embeds it is not. The differential public signal is off unless a run asks
for it, and when it is on it compares stdout, stderr, exit code, and
filesystem side effects only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import shlex
import time

from .errors import ProtocolError
from .public_tools import TOOLS


PB_NODE_ORDER = ("p1", "p2")
SWE_NODE_ORDER = ("s1", "s2")
DEFAULT_ROLE_MAP = {"s1": "p1", "s2": "p2"}
SWAPPED_ROLE_MAP = {"s1": "p2", "s2": "p1"}
SWE_ROLE_TOPICS = {
    "s1": ("coding", "implementation"),
    "s2": ("testing", "validation"),
}
PROBE_SCRIPT_SENTENCE = "Write a behaviour-probe script"
# Kept out of the submission. A script here may run the reference through its
# normal interface. The submitted program may not.
PROBE_DIR = ".pb-probe"
# compile.sh's ./executable, copied aside so the black-box compare can run it
# without replacing the image reference. Not a submitted artifact.
PB_SIGNAL_CANDIDATE = "build/.alo-pb-candidate"
REFERENCE_BASENAME = "executable"
REFERENCE_MODE = 0o111
# Candidate builds live under build/. They are not the reference binary.
_CANDIDATE_PARENTS = frozenset({"build"})

_IMPLEMENT_TOOLS = (
    "list_files", "read_file", "write_file", "edit_file", "search", "command", "execute_code",
)
_VALIDATE_TOOLS = ("list_files", "read_file", "search", "command")

_INSTRUMENT = frozenset({
    "strace", "ltrace", "ptrace", "gdb", "lldb", "bpftrace", "dtrace", "sysdig",
})
_INSPECT = frozenset({
    "objdump", "llvm-objdump", "ghidra", "ghidrarun", "r2", "radare2", "rabin2",
    "ndisasm", "readelf", "eu-readelf", "nm", "strings", "hexdump", "xxd", "od",
    "objcopy", "ida", "ida64",
})
_READERS = frozenset({
    "cat", "dd", "head", "tail", "less", "more", "od", "xxd", "hexdump", "strings",
})
_COPY_TOOLS = frozenset({"cp", "mv", "ln", "install"})
_NETWORK_BIN = frozenset({
    "curl", "wget", "nc", "ncat", "netcat", "ssh", "scp", "sftp", "ftp", "ping",
    "dig", "nslookup", "socat", "telnet", "aria2c", "fetch", "pip", "pip3",
    "apt", "apt-get", "npm", "yarn", "pnpm",
})
_SHELLS = frozenset({"sh", "bash", "dash", "zsh"})
_URL = re.compile(r"https?://|ftp://|git://", re.IGNORECASE)
_REFERENCE_TEXT = re.compile(
    r"(?<![\w.])(?<!build/)(?:\./|/workspace/|/)?executable(?![\w.])",
)
_BINARY_READ = re.compile(
    r"open\s*\([^)]*executable[^)]*['\"]rb['\"]|read_bytes\s*\(",
    re.IGNORECASE | re.DOTALL,
)
_INVOKE = re.compile(
    r"(?:\bexec(?:l|v|le|lp|vpe)?\b|\bos\.system\b|\bsystem\s*\(|\bpopen\s*\(|"
    r"\bsubprocess\b|\bPopen\b|\bos\.exec[a-z_]*\b)"
    r".{0,200}(?<![\w.])(?<!build/)(?:\./|/workspace/)executable\b",
    re.IGNORECASE | re.DOTALL,
)
# objdump/strings/strace/ltrace (and the same family) aimed at the reference
# from Python or C, where the line is not a shell command.
_FORBIDDEN_TOOL_CALL = re.compile(
    r"(?is)(?:subprocess|\bos\.system\b|\bsystem\s*\(|\bpopen\s*\(|\bexecl|\bexecv|\bPopen\b)"
    r".{0,240}\b(objdump|llvm-objdump|strings|hexdump|xxd|readelf|eu-readelf|ghidra|"
    r"strace|ltrace|gdb|lldb|nm)\b"
    r".{0,160}(?<![\w.])(?<!build/)(?:\./|/workspace/)?executable\b"
)
_DIRECT_CALL = re.compile(
    r"(?m)^[ \t]*(?:(?:sudo|command|exec)[ \t]+)*(?:\./|/workspace/)executable(?:[ \t]|$)",
)
_TEXT_SUFFIXES = frozenset({
    "", ".sh", ".bash", ".py", ".c", ".h", ".cc", ".cpp", ".rs", ".go", ".js",
    ".ts", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".mak", ".cmake",
})


def normalize_role_map(value):
    """Accept the default map, the swap, or ``s1=p1,s2=p2`` text."""
    if value is None:
        return dict(DEFAULT_ROLE_MAP)
    if isinstance(value, str):
        text = value.strip()
        if text in {"", "default", "s1=p1,s2=p2", "s1:p1,s2:p2"}:
            return dict(DEFAULT_ROLE_MAP)
        if text in {"swap", "s1=p2,s2=p1", "s1:p2,s2:p1"}:
            return dict(SWAPPED_ROLE_MAP)
        parsed = {}
        for part in text.split(","):
            if not part.strip():
                continue
            if "=" in part:
                left, right = part.split("=", 1)
            elif ":" in part:
                left, right = part.split(":", 1)
            else:
                raise ValueError("invalid_role_map")
            parsed[left.strip()] = right.strip()
        value = parsed
    if not isinstance(value, dict):
        raise ValueError("invalid_role_map")
    cleaned = {str(key): str(item) for key, item in value.items()}
    if set(cleaned) != set(SWE_NODE_ORDER) or set(cleaned.values()) != set(PB_NODE_ORDER):
        raise ValueError("invalid_role_map")
    if len(set(cleaned.values())) != 2:
        raise ValueError("invalid_role_map")
    return {"s1": cleaned["s1"], "s2": cleaned["s2"]}


def normalize_probe_owner(value):
    if value is None or value == "p1":
        return "p1"
    if value == "p2":
        return "p2"
    raise ValueError("invalid_probe_script_owner")


def _yaml_list(items):
    return "[" + ", ".join(items) + "]"


def _node_block(node_id, *, goal, topics, tools, deps):
    dep_text = "[" + ", ".join(deps) + "]"
    return (
        f"  - id: {node_id}\n"
        f"    goal: {goal}\n"
        f"    deps: {dep_text}\n"
        f"    topics: {_yaml_list(topics)}\n"
        f"    role:\n"
        f"      capability_reqs: [{topics[0]}]\n"
        f"      epistemic_scope:\n"
        f"        include: [task_description, goal, persistent_workspace]\n"
        f"        exclude: [billing, credentials]\n"
        f"      authority: [inspect]\n"
        f"      tools: {_yaml_list(tools)}\n"
        f"      output_schema: source\n"
        f"      success_criteria: non_empty_output\n"
    )


def pb_prompt(*, role_map=None, probe_script_owner="p1"):
    """Worker-facing rules. The fixed DAG is what actually runs."""
    mapping = normalize_role_map(role_map)
    owner = normalize_probe_owner(probe_script_owner)
    pairs = ", ".join(f"{swe}->{pb}" for swe, pb in mapping.items())
    return (
        "ProgramBench two-node DAG p1 -> p2. "
        f"Role mapping: {pairs}. "
        f"The behaviour-probe script belongs to {owner}. "
        "The reference ./executable is execute-only (mode 111). "
        "Run it only through its normal interface and observe stdout, stderr, "
        "the exit code, and filesystem side effects. "
        "Do not read its bytes. Do not decompile or disassemble it. "
        "Do not run objdump, strings, hexdump, xxd, ghidra, readelf, "
        "strace, ltrace, gdb, or similar instrumentation on it. "
        "Do not use the network. "
        "compile.sh must produce ./executable at the workspace root. "
        f"A behaviour-probe script under {PROBE_DIR}/ is not part of the submission "
        "and may run the reference through its normal interface. "
        "The submission must not wrap, copy, embed, or call the reference binary. "
        "Use only these public tools: "
        + ", ".join(TOOLS)
        + ". Do not invent tool names such as public_tools."
    )


def pb_dag(*, role_map=None, probe_script_owner="p1"):
    """Fixed p1 -> p2 DAG. Topics follow the SWE role mapped onto each node."""
    mapping = normalize_role_map(role_map)
    owner = normalize_probe_owner(probe_script_owner)
    inverse = {pb: swe for swe, pb in mapping.items()}
    implement = (
        "Reconstruct the program from the docs and from black-box runs of the "
        "reference ./executable. Write ./compile.sh so it produces ./executable "
        "at the workspace root. "
        "The reference binary is execute-only. Do not read, decompile, or "
        "instrument it. Forbidden on the reference include objdump, strings, hexdump, "
        "ghidra, strace, and ltrace. Do not use the network. Do not wrap, copy, "
        "or call the reference from the submission."
    )
    validate = (
        "Run chmod +x ./compile.sh && ./compile.sh and check the build. "
        "Do not reimplement the program. The reference ./executable is "
        "execute-only. Do not read, decompile, or instrument it. Forbidden on "
        "the reference include objdump, strings, hexdump, ghidra, strace, and ltrace. "
        "Do not use the network."
    )
    blocks = ["subtasks:\n"]
    for node_id in PB_NODE_ORDER:
        swe = inverse[node_id]
        topics = SWE_ROLE_TOPICS[swe]
        goal = implement if swe == "s1" else validate
        if node_id == owner:
            goal = (
                goal + " " + PROBE_SCRIPT_SENTENCE
                + f" under {PROBE_DIR}/ and keep it out of the submission. "
                "That script may run the reference ./executable only through its "
                "normal interface. It must not be part of the submitted program."
            )
        if swe == "s1" or node_id == owner:
            tools = _IMPLEMENT_TOOLS
        else:
            tools = _VALIDATE_TOOLS
        deps = [] if node_id == "p1" else ["p1"]
        blocks.append(_node_block(node_id, goal=goal, topics=topics, tools=tools, deps=deps))
    text = "".join(blocks)
    for name in _tool_names(text):
        if name not in TOOLS:
            raise ProtocolError("pb_role_contract_unknown_tool")
    return text


def _tool_names(dag_text):
    names = []
    for line in dag_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("tools:"):
            body = stripped.split(":", 1)[1].strip().strip("[]")
            names.extend(part.strip() for part in body.split(",") if part.strip())
    return names


def reference_is_execute_only(mode):
    """True when the permission bits are exactly 111 (execute-only)."""
    if isinstance(mode, bool) or not isinstance(mode, int):
        return False
    return (mode & 0o777) == REFERENCE_MODE


def is_reference_path(token):
    """True for the workspace-root reference, not ``build/executable``."""
    if not isinstance(token, str) or not token or token.startswith("-"):
        return False
    text = token.strip().strip("\"'").replace("\\", "/")
    if not text or text.startswith("-"):
        return False
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if not parts or parts[-1] != REFERENCE_BASENAME:
        return False
    if any(part in _CANDIDATE_PARENTS for part in parts[:-1]):
        return False
    return True


def _basename(token):
    text = token.strip().strip("\"'")
    return Path(text.replace("\\", "/")).name.lower()


def _token_mentions_reference(token):
    if is_reference_path(token):
        return True
    if isinstance(token, str) and "=" in token and not token.startswith("-"):
        _key, value = token.split("=", 1)
        return is_reference_path(value)
    if isinstance(token, str) and token.startswith("-") and "=" in token:
        _flag, value = token.split("=", 1)
        return is_reference_path(value)
    return False


def _expand_argv(argv):
    views = [list(argv)]
    if len(argv) >= 3 and _basename(argv[0]) in _SHELLS and argv[1] == "-c":
        try:
            inner = shlex.split(argv[2])
        except ValueError:
            inner = None
        if inner:
            views.append(inner)
    return views


def _is_network(argv):
    if not argv:
        return False
    base = _basename(argv[0])
    if base in _NETWORK_BIN:
        return True
    if any(_URL.search(token) for token in argv if isinstance(token, str)):
        return True
    if base == "git" and any(token in {"clone", "fetch", "pull", "push"} for token in argv[1:]):
        return True
    if base == "cargo" and any(token in {"fetch", "install", "download"} for token in argv[1:]):
        return True
    if base == "go" and any(token in {"get", "install"} for token in argv[1:]):
        return True
    if base == "python3" and "pip" in argv and "install" in argv:
        return True
    if base in {"pip", "pip3"}:
        return True
    return False


def _targets_reference(argv):
    return any(_token_mentions_reference(token) for token in argv[1:])


def _copies_reference(argv):
    if not argv or _basename(argv[0]) not in _COPY_TOOLS:
        return False
    operands = [token for token in argv[1:] if isinstance(token, str) and not token.startswith("-")]
    if len(operands) < 2:
        return False
    sources = operands[:-1]
    return any(_token_mentions_reference(token) for token in sources)


def _path_parts(token):
    text = token.strip().strip("\"'").replace("\\", "/")
    return [part for part in text.split("/") if part not in {"", "."}]


def _is_own_build_output(token):
    """A path under ``build/``, the agent's compile output, not the reference."""
    if not isinstance(token, str) or not token:
        return False
    parts = _path_parts(token)
    return len(parts) >= 2 and "build" in parts[:-1]


def _installs_own_build(argv):
    """True when a copy installs the agent's build output onto ``./executable``.

    ``cp build/executable executable`` is the legal install. Copying the
    reference itself is not.
    """
    if not argv or _basename(argv[0]) not in _COPY_TOOLS:
        return False
    operands = [token for token in argv[1:] if isinstance(token, str) and not token.startswith("-")]
    if len(operands) != 2:
        return False
    source, dest = operands
    return _is_own_build_output(source) and is_reference_path(dest)


_REDIRECT_TOKEN = re.compile(r"^(?:\d*>>?|\d*>&)\S*$")
_SHELL_SPLIT = frozenset({"||", "&&", ";", "|"})


def _strip_redirects(tokens):
    cleaned = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        bare_redirect = token in {">", ">>", "<", ">&", "&>"} or (
            isinstance(token, str) and re.fullmatch(r"\d*(?:>>?|&>)", token))
        if bare_redirect:
            index += 2 if index + 1 < len(tokens) else 1
            continue
        if isinstance(token, str) and _REDIRECT_TOKEN.fullmatch(token):
            index += 1
            continue
        cleaned.append(token)
        index += 1
    return cleaned


# Characters that make posix ``shlex.split(comments=True)`` do more than return
# the line as one word: its whitespace, quotes, escape, and comment start. ``\s``
# is wider than shlex's whitespace, so such a line just takes the slow path.
_SHLEX_ACTIVE = re.compile(r"[\s'\"\\#]")
# Lines up to this length keep the stdlib tokenizer itself. Longer ones use
# ``_posix_shlex_split``, the same state machine in linear time.
_SHLEX_DIRECT_MAX = 4096
_SHLEX_WS = " \t\r\n"
_WORD_RUN = re.compile(r"[^ \t\r\n'\"\\#]+")
_WS_RUN = re.compile(r"[ \t\r\n]+")
_DQ_RUN = re.compile(r'[^"\\]+')
_CHECK_EVERY = 4096


def _posix_shlex_split(text, check=None):
    """``shlex.split(text, comments=True, posix=True)`` in linear time.

    This follows ``shlex.read_token`` for posix mode with ``whitespace_split``,
    the default whitespace, quotes ``'"``, escape ``\\``, escaped quotes ``"``
    and commenter ``#``. Tokens are built from slices instead of one character
    at a time. The same ``ValueError`` messages are raised. ``check`` is called
    every few thousand steps so a deadline can stop a very long line.
    """
    tokens = []
    parts = []
    quoted = False
    state = " "
    escaped_state = None
    index = 0
    size = len(text)
    steps = 0
    while True:
        steps += 1
        if check is not None and steps % _CHECK_EVERY == 0:
            check()
        if state == " ":
            if index >= size:
                return tokens
            char = text[index]
            if char in _SHLEX_WS:
                index = _WS_RUN.match(text, index).end()
            elif char == "#":
                newline = text.find("\n", index)
                index = size if newline < 0 else newline + 1
            elif char == "\\":
                escaped_state, state = "a", "\\"
                index += 1
            elif char in "'\"":
                state = char
                index += 1
            else:
                state = "a"
        elif state == "a":
            if index >= size:
                tokens.append("".join(parts))
                return tokens
            char = text[index]
            if char in _SHLEX_WS or char == "#":
                if char == "#":
                    newline = text.find("\n", index)
                    index = size if newline < 0 else newline + 1
                else:
                    index += 1
                tokens.append("".join(parts))
                parts, quoted, state = [], False, " "
            elif char in "'\"":
                state = char
                index += 1
            elif char == "\\":
                escaped_state, state = "a", "\\"
                index += 1
            else:
                end = _WORD_RUN.match(text, index).end()
                parts.append(text[index:end])
                index = end
        elif state == "'":
            quoted = True
            close = text.find("'", index)
            if close < 0:
                raise ValueError("No closing quotation")
            parts.append(text[index:close])
            index, state = close + 1, "a"
        elif state == '"':
            quoted = True
            if index >= size:
                raise ValueError("No closing quotation")
            char = text[index]
            if char == '"':
                index, state = index + 1, "a"
            elif char == "\\":
                escaped_state, state = '"', "\\"
                index += 1
            else:
                end = _DQ_RUN.match(text, index).end()
                parts.append(text[index:end])
                index = end
        else:  # escape
            if index >= size:
                raise ValueError("No escaped character")
            char = text[index]
            if escaped_state == '"' and char != "\\" and char != '"':
                parts.append("\\")
            parts.append(char)
            index += 1
            state = escaped_state


def _shlex_words(line, check=None):
    """``shlex.split(line, comments=True, posix=True)``, same result, linear time.

    shlex grows each token one character at a time, which is quadratic in the
    token length: one 5 MB word took hours. A line with none of the characters
    above is a single word, which is exactly what shlex returns for it
    (an empty line still goes to shlex, which returns ``[]``). Other short
    lines use shlex itself; long ones use ``_posix_shlex_split``.
    """
    if line and _SHLEX_ACTIVE.search(line) is None:
        return [line]
    if len(line) <= _SHLEX_DIRECT_MAX:
        return shlex.split(line, comments=True, posix=True)
    return _posix_shlex_split(line, check)


class SubmissionScanDeadline(Exception):
    """The end-of-unit submission scan ran past its deadline."""


def _deadline_check(deadline, clock):
    if deadline is None:
        return None
    clock = clock or time.monotonic

    def check():
        if clock() > deadline:
            raise SubmissionScanDeadline("submission_scan_deadline")
    return check


def _simple_commands(line, check=None):
    """Split one shell line into argv lists, dropping redirects and ``|| true``."""
    if not isinstance(line, str) or not line.strip():
        return []
    try:
        tokens = _shlex_words(line, check)
    except ValueError:
        return []
    groups = []
    current = []
    for token in tokens:
        if token in _SHELL_SPLIT:
            if current:
                groups.append(_strip_redirects(current))
            current = []
        else:
            current.append(token)
    if current:
        groups.append(_strip_redirects(current))
    return [group for group in groups if group]


def _reads_reference_bytes(argv):
    if not argv or is_reference_path(argv[0]) or not _targets_reference(argv):
        return False
    base = _basename(argv[0])
    return base in _READERS or base in _INSPECT


def is_excluded_probe(relative):
    """True for a behaviour-probe path that is not part of the submission."""
    if not isinstance(relative, str) or not relative:
        return False
    return PROBE_DIR in PurePosixPath(relative).parts


def _text_submission_violations(text, *, allow_reference_execution, check=None):
    """Violation kinds in one text file.

    Running the reference through its normal interface is ``invokes_reference``
    unless ``allow_reference_execution`` is set. The scan turns that into a
    warning when the file is outside the build graph, and a rejection when
    the build uses the file. Copying, reading, inspecting, or instrumenting
    the reference is never allowed.
    """
    if not isinstance(text, str) or not text:
        return []
    found = []
    if re.search(r"open\s*\([^)]*executable[^)]*['\"]rb['\"]", text, re.IGNORECASE | re.DOTALL):
        found.append("reads_reference_bytes")
    if re.search(r"executable.{0,80}read_bytes\s*\(|read_bytes\s*\([^)]*executable", text, re.IGNORECASE | re.DOTALL):
        found.append("reads_reference_bytes")
    if not allow_reference_execution and (_INVOKE.search(text) or _DIRECT_CALL.search(text)):
        found.append("invokes_reference")
    for line in text.splitlines():
        if check is not None:
            check()
        for argv in _simple_commands(line, check):
            if _installs_own_build(argv):
                continue
            if _reads_reference_bytes(argv):
                found.append("reads_reference_bytes")
                break
            if (argv and not is_reference_path(argv[0]) and _targets_reference(argv)
                    and _basename(argv[0]) in _INSTRUMENT):
                found.append("instruments_reference")
                break
            if _copies_reference(argv):
                found.append("copies_reference")
                break
    forbidden = _FORBIDDEN_TOOL_CALL.search(text)
    if forbidden:
        tool = forbidden.group(1).lower()
        found.append("instruments_reference" if tool in _INSTRUMENT else "inspects_reference")
    return found


def _inspect_violation(argv):
    if not argv or not _targets_reference(argv):
        return None
    # Running the reference itself is the allowed black-box interface.
    if is_reference_path(argv[0]):
        return None
    base = _basename(argv[0])
    if base in _INSTRUMENT:
        return "pb_reference_instrumentation_forbidden"
    if base in _INSPECT:
        return "pb_reference_inspection_forbidden"
    if base in _READERS:
        return "pb_reference_binary_read_forbidden"
    return None


def _text_violation(text):
    if not isinstance(text, str) or not text:
        return None
    if _URL.search(text):
        return "pb_network_forbidden"
    if _BINARY_READ.search(text):
        return "pb_reference_binary_read_forbidden"
    if not _REFERENCE_TEXT.search(text):
        return None
    lowered = text.lower()
    if any(re.search(r"(?<![\w./])" + re.escape(name) + r"(?![\w.])", lowered) for name in _INSTRUMENT):
        return "pb_reference_instrumentation_forbidden"
    if any(re.search(r"(?<![\w./])" + re.escape(name) + r"(?![\w.])", lowered) for name in _INSPECT):
        return "pb_reference_inspection_forbidden"
    return None


def _is_blackbox_program(token):
    """The reference, or the candidate under ``build/``, run as argv[0]."""
    if is_reference_path(token):
        return True
    if not isinstance(token, str) or not token:
        return False
    parts = [part for part in token.replace("\\", "/").split("/") if part not in {"", "."}]
    return bool(parts) and parts[-1] == REFERENCE_BASENAME and "build" in parts[:-1]


def pb_command_violation(argv):
    """Reason code when ``argv`` breaks the ProgramBench reference rules.

    ``['./executable', '--help']`` is a normal-interface run and is allowed,
    including when an argument is an ordinary input string. Inspection,
    instrumentation, byte reads, copies of the reference, and network
    commands are not.
    """
    if not isinstance(argv, list) or any(type(item) is not str for item in argv):
        return None
    if argv and _is_blackbox_program(argv[0]):
        return None
    if any(_URL.search(token) for token in argv):
        return "pb_network_forbidden"
    for view in _expand_argv(argv):
        if view and _is_blackbox_program(view[0]):
            continue
        if _is_network(view):
            return "pb_network_forbidden"
        if _copies_reference(view):
            return "pb_reference_copy_forbidden"
        inspected = _inspect_violation(view)
        if inspected:
            return inspected
    if len(argv) >= 3 and _basename(argv[0]) in _SHELLS | {"python", "python3"} and argv[1] == "-c":
        found = _text_violation(argv[2])
        if found:
            return found
    return None


def pb_code_violation(code):
    """Same rules for ``execute_code`` text."""
    return _text_violation(code if isinstance(code, str) else "")


def assert_pb_command(argv):
    violation = pb_command_violation(argv)
    if violation:
        raise ProtocolError(violation)
    return list(argv)


def blackbox_probe_argv(binary, args=()):
    """Argv that only runs a program through its normal interface."""
    if not isinstance(binary, str) or not binary:
        raise ProtocolError("pb_blackbox_binary_required")
    if isinstance(args, str) or not isinstance(args, (list, tuple)) or any(type(item) is not str for item in args):
        raise ProtocolError("pb_blackbox_args_must_be_strings")
    argv = [binary, *args]
    return assert_pb_command(argv)


def compile_signal(returncode):
    """``0`` is pass, any other int is fail, and a missing result is unknown."""
    if returncode is None or isinstance(returncode, bool) or type(returncode) is not int:
        return "unknown"
    return "pass" if returncode == 0 else "fail"


def _known_io(row):
    return (isinstance(row, dict)
            and type(row.get("returncode")) is int
            and not isinstance(row.get("returncode"), bool)
            and isinstance(row.get("stdout"), str)
            and isinstance(row.get("stderr"), str))


def _fs_delta(row):
    before = row.get("fs_before")
    after = row.get("fs_after")
    if before is None and after is None:
        return None
    if not isinstance(before, str) or not isinstance(after, str):
        return "unknown"
    return after if after != before else ""


def compare_blackbox(reference, candidate):
    """Pass when stdout, stderr, exit code, and filesystem deltas match.

    Missing I/O is unknown. This does not read a binary.
    """
    if not _known_io(reference) or not _known_io(candidate):
        return "unknown"
    ref_fs = _fs_delta(reference)
    cand_fs = _fs_delta(candidate)
    if ref_fs == "unknown" or cand_fs == "unknown":
        return "unknown"
    if (ref_fs is None) != (cand_fs is None):
        return "unknown"
    same = (
        reference["returncode"] == candidate["returncode"]
        and reference["stdout"] == candidate["stdout"]
        and reference["stderr"] == candidate["stderr"]
        and ref_fs == cand_fs
    )
    return "pass" if same else "fail"


def run_blackbox_differential(reference_argv, candidate_argv, *, runner):
    """Run both programs through ``runner``. The argv is checked first.

    ``runner`` returns one I/O dict per argv. No runner means unknown, which
    the caller must not write into learning state.
    """
    assert_pb_command(reference_argv)
    assert_pb_command(candidate_argv)
    if runner is None:
        return "unknown"
    reference = runner(reference_argv)
    candidate = runner(candidate_argv)
    return compare_blackbox(reference, candidate)


def combine_pb_signal(compile_returncode, *, differential_enabled=False, differential_signal=None):
    """Compile success is the signal. Differential I/O joins only when enabled."""
    compile_part = compile_signal(compile_returncode)
    enabled = bool(differential_enabled)
    if not enabled:
        return {
            "signal": compile_part,
            "compile": compile_part,
            "differential": None,
            "differential_enabled": False,
        }
    diff = differential_signal if differential_signal in {"pass", "fail", "unknown"} else "unknown"
    definite = [item for item in (compile_part, diff) if item in {"pass", "fail"}]
    if not definite:
        signal = "unknown"
    elif "fail" in definite:
        signal = "fail"
    else:
        signal = "pass"
    return {
        "signal": signal,
        "compile": compile_part,
        "differential": diff,
        "differential_enabled": True,
    }


def should_record_signal(signal):
    """Unknown is not a learning write. Same rule as a definite repro outcome."""
    return signal in {"pass", "fail"}


def signal_components_to_record(result):
    """Definite components only. An unknown differential is omitted."""
    if not isinstance(result, dict):
        return []
    rows = []
    if should_record_signal(result.get("compile")):
        rows.append(("compile", result["compile"]))
    if result.get("differential_enabled") and should_record_signal(result.get("differential")):
        rows.append(("differential", result["differential"]))
    return rows


def reference_digest(root):
    """Hash the reference when the controller can read it.

    Mode 111 is execute-only, so a hash is not available from a byte read.
    Callers that already know the digest pass it to the submission scan.
    """
    path = Path(root) / REFERENCE_BASENAME
    if not path.is_file() or path.is_symlink():
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _looks_text(path):
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES or path.name in {"compile.sh", "Makefile", "makefile"}:
        return True
    return False


_BUILD_SUFFIXES = frozenset({
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".rs", ".go", ".py",
    ".sh", ".bash", ".s", ".java", ".js", ".ts", ".mak", ".cmake", ".toml",
    ".mod", ".sum", ".f", ".f90", ".rb", ".pl", ".m", ".mm", ".swift", ".kt",
    ".cs", ".php", ".lua", ".r", ".asm", ".inc", ".in", ".pyi", ".pyx", ".pxd",
    ".zig", ".d", ".ml", ".mli", ".hs", ".lhs", ".cabal", ".gradle", ".kts",
    ".xml", ".nim",
})
_BUILD_NAMES = frozenset({
    "makefile", "gnumakefile", "cargo.toml", "cargo.lock", "go.mod", "go.sum",
    "cmakelists.txt", "meson.build", "build.gradle", "pom.xml", "package.json",
    "requirements.txt", "setup.py", "pyproject.toml", "compile.sh",
})
_PROJECT_GLOBS = {
    "cargo": ("Cargo.toml", "Cargo.lock", "build.rs", "src/**/*.rs"),
    "go": ("go.mod", "go.sum", "**/*.go"),
    "make": ("Makefile", "makefile", "GNUmakefile", "**/*.mk"),
    "cmake": ("CMakeLists.txt", "**/*.cmake", "meson.build"),
}


def _build_input_name(relative):
    """True for a source, script, or project file the build can name."""
    path = PurePosixPath(relative)
    if path.name == REFERENCE_BASENAME and "build" not in path.parts[:-1]:
        return False
    if path.suffix.lower() in _BUILD_SUFFIXES:
        return True
    return path.name.lower() in _BUILD_NAMES


def _resolve_existing(root, token, base_dir):
    if not isinstance(token, str):
        return None
    text = token.strip().strip("\"'")
    if not text or text.startswith(("-", "$")) or "://" in text or _token_is_glob(text):
        return None
    if text.startswith("/"):
        return None
    # ``../helper.h`` is a build input when it still lands inside the submission.
    candidate = (base_dir / text)
    try:
        resolved = candidate.resolve()
        root_resolved = root.resolve()
        if not resolved.is_file() or resolved.is_symlink() or not resolved.is_relative_to(root_resolved):
            return None
    except OSError:
        return None
    relative = resolved.relative_to(root_resolved).as_posix()
    if not _build_input_name(relative):
        return None
    return relative


def _token_is_glob(text):
    return any(char in text for char in "*?[")


def _token_is_unresolved_reference(text):
    """A shell variable or substitution whose files this scan cannot name."""
    if not isinstance(text, str) or not text:
        return False
    return text.startswith("$") or "${" in text or "`" in text or "$(" in text


def _brace_group(text):
    """One ``{a,b}`` group, or None when the braces are ordinary punctuation."""
    if text.count("{") != 1 or text.count("}") != 1:
        return None
    match = re.search(r"\{([^{}]*)\}", text)
    if match is None or "," not in match.group(1):
        return None
    return match


def _brace_expand(text):
    """Expand one ``{a,b}`` group. Nested braces are left unresolved."""
    match = _brace_group(text)
    if match is None:
        return [], False
    start, end = match.span()
    options = [part for part in match.group(1).split(",") if part]
    if not options:
        return [], False
    return [text[:start] + part + text[end:] for part in options], True


def _glob_matches(root, pattern, base_dir):
    """Files a relative glob names. ``False`` means the pattern did not resolve."""
    if not pattern or pattern.startswith("/") or ".." in PurePosixPath(pattern).parts:
        return [], False
    found = []
    saw_file = False
    bases = [base_dir]
    if base_dir != root:
        bases.append(root)
    try:
        root_resolved = root.resolve()
    except OSError:
        return [], False
    for base in bases:
        try:
            matches = list(base.glob(pattern))
        except (OSError, ValueError):
            return [], False
        for path in matches:
            if path.is_symlink() or not path.is_file():
                continue
            saw_file = True
            try:
                relative = path.resolve().relative_to(root_resolved).as_posix()
            except OSError:
                return [], False
            if _build_input_name(relative):
                found.append(relative)
    if not saw_file:
        return [], False
    return found, True


def _expand_build_token(root, token, base_dir):
    """Build inputs named by one shell token.

    The second value is False when the token might name inputs this scan
    cannot list. An incomplete graph must not be treated as proof that a
    file is unused.
    """
    if not isinstance(token, str):
        return [], True
    text = token.strip().strip("\"'")
    if not text or text.startswith("-") or "://" in text:
        return [], True
    if _token_is_unresolved_reference(text):
        return [], False
    if _brace_group(text) is not None:
        patterns, brace_ok = _brace_expand(text)
        if not brace_ok:
            return [], False
    elif _token_is_glob(text):
        patterns = [text]
    else:
        found = _resolve_existing(root, text, base_dir)
        if found is None and base_dir != root:
            found = _resolve_existing(root, text, root)
        return ([found] if found else []), True
    found = []
    resolved_any = False
    for pattern in patterns:
        if not _token_is_glob(pattern):
            resolved = _resolve_existing(root, pattern, base_dir)
            if resolved is None and base_dir != root:
                resolved = _resolve_existing(root, pattern, root)
            if resolved:
                found.append(resolved)
                resolved_any = True
            continue
        matches, ok = _glob_matches(root, pattern, base_dir)
        if not ok:
            continue
        found.extend(matches)
        resolved_any = True
    if not resolved_any:
        return found, False
    return found, True


def _include_paths(text):
    if not isinstance(text, str):
        return []
    return re.findall(r'#\s*include\s+"([^"]+)"', text)


def _python_modules(text):
    if not isinstance(text, str):
        return []
    names = []
    for match in re.finditer(r"(?m)^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))", text):
        name = match.group(1) or match.group(2)
        names.append(name.split(",")[0].strip().split(" as ")[0].strip())
    return names


def _project_files(root, tool):
    patterns = _PROJECT_GLOBS.get(tool) or ()
    found = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                relative = path.resolve().relative_to(root.resolve()).as_posix()
            except OSError:
                continue
            if _build_input_name(relative) or path.name.lower() in _BUILD_NAMES:
                found.append(relative)
    return found


def _build_tool(argv):
    if not argv:
        return None
    base = _basename(argv[0])
    if base in {"cargo", "rustc"}:
        return "cargo"
    if base == "go" and any(token in {"build", "install", "test"} for token in argv[1:]):
        return "go"
    if base == "make":
        return "make"
    if base in {"cmake", "meson", "ninja"}:
        return "cmake"
    return None


_SHELL_FLOW = frozenset({
    "if", "then", "else", "elif", "fi", "while", "until", "for", "do", "done",
    "case", "esac", "select", "time", "coproc",
})


def _shell_tokens(text):
    """Split a shell script. ``None`` means the quotes do not close."""
    tokens = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in " \t\r":
            index += 1
            continue
        if char == "\n":
            if not tokens or tokens[-1] != "\n":
                tokens.append("\n")
            index += 1
            continue
        if char == "#":
            while index < length and text[index] != "\n":
                index += 1
            continue
        if char in "\"'":
            quote = char
            index += 1
            buffer = []
            while index < length and text[index] != quote:
                if quote == "\"" and text[index] == "\\" and index + 1 < length:
                    buffer.append(text[index + 1])
                    index += 2
                    continue
                buffer.append(text[index])
                index += 1
            if index >= length:
                return None
            index += 1
            tokens.append("".join(buffer))
            continue
        if text.startswith("&&", index) or text.startswith("||", index):
            tokens.append(text[index:index + 2])
            index += 2
            continue
        if char in "();|&{}":
            tokens.append(char)
            index += 1
            continue
        if char in "<>":
            end = index + 1
            while end < length and text[end] in "<>&":
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        end = index + 1
        while end < length and text[end] not in " \t\r\n#\"'();|&{}<>":
            if text.startswith("&&", end) or text.startswith("||", end):
                break
            end += 1
        tokens.append(text[index:end])
        index = end
    return tokens


def _resolve_shell_dir(root, cwd, token):
    """A directory inside the submission, or None when the token is not one."""
    if not isinstance(token, str):
        return None
    text = token.strip().strip("\"'")
    if (not text or text == "-" or text.startswith(("-", "/"))
            or _token_is_unresolved_reference(text) or _token_is_glob(text)):
        return None
    candidate = cwd / text
    try:
        resolved = candidate.resolve()
        root_resolved = root.resolve()
        if (not resolved.is_dir() or resolved.is_symlink()
                or not resolved.is_relative_to(root_resolved)):
            return None
    except OSError:
        return None
    return resolved


class _BuildShell:
    """Working directory while reading one shell script for build inputs.

    ``cd``, ``pushd``, and ``popd`` move ``cwd``. A ``(...)`` subshell gets
    its own directory and stack, then the parent is restored. A change or a
    construct this walker cannot follow sets ``complete`` false.
    """

    def __init__(self, root, cwd):
        self.root = root
        self.cwd = cwd
        self.stack = []
        self.files = []
        self.complete = True

    def consume(self, tokens):
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in _SHELL_SPLIT or token in {"&", "\n", "{", "}"}:
                index += 1
                continue
            if token == "(":
                end = _matching_paren(tokens, index)
                if end is None:
                    self.complete = False
                    index += 1
                    continue
                child = _BuildShell(self.root, self.cwd)
                child.stack = list(self.stack)
                child.consume(tokens[index + 1:end])
                self.files.extend(child.files)
                self.complete = self.complete and child.complete
                index = end + 1
                continue
            if token == ")":
                self.complete = False
                index += 1
                continue
            if token in _SHELL_FLOW:
                self.complete = False
                index += 1
                continue
            argv = []
            while index < len(tokens) and tokens[index] not in _SHELL_SPLIT | {"&", "\n", "(", ")", "{", "}"}:
                argv.append(tokens[index])
                index += 1
            self._command(_strip_redirects(argv))

    def _command(self, argv):
        if not argv:
            return
        base = _basename(argv[0])
        if base == "cd":
            self._move(argv, push=False)
            return
        if base == "pushd":
            self._move(argv, push=True)
            return
        if base == "popd":
            if len(argv) != 1 or not self.stack:
                self.complete = False
                return
            self.cwd = self.stack.pop()
            return
        tool = _build_tool(argv)
        if tool:
            for found in _project_files(self.root, tool):
                if not is_excluded_probe(found):
                    self.files.append(found)
        for token in argv:
            found, token_ok = _expand_build_token(self.root, token, self.cwd)
            if not token_ok:
                self.complete = False
            self.files.extend(found)

    def _move(self, argv, *, push):
        operands = [token for token in argv[1:] if isinstance(token, str) and not token.startswith("-")]
        if len(operands) != 1:
            self.complete = False
            return
        resolved = _resolve_shell_dir(self.root, self.cwd, operands[0])
        if resolved is None:
            self.complete = False
            return
        if push:
            self.stack.append(self.cwd)
        self.cwd = resolved


def _matching_paren(tokens, open_index):
    depth = 0
    for index in range(open_index, len(tokens)):
        if tokens[index] == "(":
            depth += 1
        elif tokens[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


_SHELL_SUFFIXES = frozenset({".sh", ".bash", ".ksh", ".zsh"})


def _is_shell_script(path, text):
    """Shell completeness applies only to shell scripts, not to C or Python."""
    if path.suffix.lower() in _SHELL_SUFFIXES or path.name.lower() in {"compile.sh", "configure"}:
        return True
    first = ""
    if isinstance(text, str) and text.lstrip():
        first = text.lstrip().splitlines()[0]
    if first.startswith("#!") and re.search(r"\b(?:sh|bash|dash|ksh|zsh)\b", first):
        return True
    return False


def _shell_build_inputs(text, root, start):
    tokens = _shell_tokens(text)
    if tokens is None:
        return [], False
    walker = _BuildShell(root, start)
    walker.consume(tokens)
    return walker.files, walker.complete


def submission_build_graph(root):
    """Files reachable from ``compile.sh``, and whether that list is complete.

    A behaviour probe that ``compile.sh`` never names, and that no file in
    that graph names, is not part of the submitted program. ``cd``,
    ``pushd``, ``popd``, and ``(...)`` subshells move the directory used for
    relative paths. A glob, variable, or directory change that does not
    resolve makes the graph incomplete: callers must not treat every other
    file as unused.
    """
    root = Path(root)
    entry = root / "compile.sh"
    if not entry.is_file() or entry.is_symlink():
        return set(), True
    pending = ["compile.sh"]
    seen = set()
    complete = True
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        seen.add(relative)
        path = root / relative
        if not path.is_file() or path.is_symlink():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        base_dir = path.parent
        for include in _include_paths(text):
            found = _resolve_existing(root, include, base_dir)
            if found and found not in seen:
                pending.append(found)
            elif found is None and ".." in PurePosixPath(include).parts:
                # A parent-relative include that escapes the tree, or does not
                # exist, is not proof that every other file is unused.
                complete = False
        for module in _python_modules(text):
            parts = module.split(".")
            for candidate in (base_dir, root):
                file_name = "/".join(parts) + ".py"
                found = _resolve_existing(root, file_name, candidate)
                if found and found not in seen:
                    pending.append(found)
                package = "/".join(parts) + "/__init__.py"
                found = _resolve_existing(root, package, candidate)
                if found and found not in seen:
                    pending.append(found)
        if _is_shell_script(path, text):
            found, shell_complete = _shell_build_inputs(text, root, base_dir)
            if not shell_complete:
                complete = False
            for item in found:
                if item not in seen:
                    pending.append(item)
    return seen, complete


def _hard_reference_kinds(text, check=None):
    """Copy, read, inspect, or instrument. A normal-interface run is not here."""
    kinds = []
    for kind in _text_submission_violations(text, allow_reference_execution=True, check=check):
        if kind != "invokes_reference" and kind not in kinds:
            kinds.append(kind)
    return kinds


def _invokes_reference(text, check=None):
    return "invokes_reference" in _text_submission_violations(
        text, allow_reference_execution=False, check=check)


def submission_reference_scan(root, *, reference_sha256=None, probe_paths=(), deadline=None, clock=None):
    """Flag a submission that copies, embeds, wraps, or reads the reference.

    ``./executable`` itself is the reference and is not a submitted file.
    A ``compile.sh`` that installs the agent's own ``build/...`` output onto
    ``./executable`` is the legal output path, not a wrapper. A file the
    build graph does not use, including a root ``probe.py`` or a script under
    ``.pb-probe/``, may run the reference through its normal interface. That
    is a warning, not a rejection. A build input that embeds, copies, wraps,
    execs, reads, or runs ``objdump`` / ``strings`` / ``strace`` / ``ltrace``
    on the reference is rejected.

    ``deadline`` (a ``clock()`` value, default ``time.monotonic``) is checked
    before each file and each scanned line; past it the scan raises
    ``SubmissionScanDeadline``. ``None`` keeps the scan unbounded.
    """
    check = _deadline_check(deadline, clock)
    root = Path(root)
    digest = reference_sha256 if isinstance(reference_sha256, str) and reference_sha256 else reference_digest(root)
    mode_ok = None
    reference = root / REFERENCE_BASENAME
    if reference.is_file() and not reference.is_symlink():
        mode_ok = reference_is_execute_only(reference.stat().st_mode)
    violations = []
    warnings = []
    unused_probes = []
    if mode_ok is False:
        violations.append({"kind": "reference_not_execute_only", "path": REFERENCE_BASENAME})
    if not root.is_dir():
        return {
            "ok": not violations,
            "violations": violations,
            "warnings": warnings,
            "unused_probes": unused_probes,
            "reference_sha256": digest,
            "reference_execute_only": mode_ok,
        }
    build_graph, graph_complete = submission_build_graph(root)
    for path in sorted(root.rglob("*")):
        if check is not None:
            check()
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == REFERENCE_BASENAME or relative.startswith(".git/"):
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            violations.append({"kind": "submission_unreadable", "path": relative})
            continue
        if digest and hashlib.sha256(raw).hexdigest() == digest:
            violations.append({"kind": "embedded_reference_bytes", "path": relative})
            continue
        if not _looks_text(path) and b"\0" in raw[:8192]:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeError:
            continue
        if digest and digest in text:
            violations.append({"kind": "embedded_reference_hash", "path": relative})
            continue
        hard = _hard_reference_kinds(text, check)
        if hard:
            violations.append({"kind": hard[0], "path": relative})
            continue
        if _invokes_reference(text, check):
            # An unresolved glob or variable means the file might still be a
            # build input. Do not warn it away or drop it from the archive.
            if relative in build_graph or not graph_complete:
                violations.append({"kind": "invokes_reference", "path": relative})
            else:
                warnings.append({"kind": "unused_probe_invokes_reference", "path": relative})
                unused_probes.append(relative)
    for probe in probe_paths or ():
        if check is not None:
            check()
        path = Path(probe)
        try:
            resolved = path.resolve()
        except OSError:
            violations.append({"kind": "submission_unreadable", "path": str(probe)})
            continue
        if root.is_dir():
            try:
                if resolved.is_relative_to(root.resolve()):
                    continue
            except OSError:
                pass
        if not path.is_file() or path.is_symlink():
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            violations.append({"kind": "submission_unreadable", "path": path.name})
            continue
        if digest and hashlib.sha256(raw).hexdigest() == digest:
            violations.append({"kind": "embedded_reference_bytes", "path": path.name})
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeError:
            continue
        if digest and digest in text:
            violations.append({"kind": "embedded_reference_hash", "path": path.name})
            continue
        for kind in _text_submission_violations(text, allow_reference_execution=True, check=check):
            violations.append({"kind": kind, "path": path.name})
            break
    return {
        "ok": not violations,
        "violations": violations,
        "warnings": warnings,
        "unused_probes": unused_probes,
        "reference_sha256": digest,
        "reference_execute_only": mode_ok,
    }


# Grace after the deadline for the child to report before it is killed.
SCAN_KILL_GRACE_SECONDS = 2.0


def _bounded_scan_child(conn, root, seconds, kwargs):
    try:
        clock = time.monotonic
        result = submission_reference_scan(root, deadline=clock() + seconds, clock=clock, **kwargs)
        conn.send(("ok", result))
    except SubmissionScanDeadline:
        conn.send(("deadline", None))
    except BaseException as error:  # reported to the parent as a scan error
        conn.send(("error", type(error).__name__))
    finally:
        conn.close()


def submission_reference_scan_bounded(root, seconds, **kwargs):
    """``submission_reference_scan`` in a spawned child that is killed at the deadline.

    The child runs the same function on the same tree, so a scan that finishes
    returns the same result. The in-process deadline checks stop most overruns
    cleanly. The kill is the hard boundary for work they cannot interrupt, such
    as one slow regex over a whole file. Raises ``SubmissionScanDeadline`` on an
    overrun and ``RuntimeError`` if the child fails.
    """
    import multiprocessing

    seconds = max(float(seconds), 0.0)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_bounded_scan_child, args=(sender, str(root), seconds, kwargs), daemon=True)
    child.start()
    sender.close()
    try:
        if not receiver.poll(seconds + SCAN_KILL_GRACE_SECONDS):
            raise SubmissionScanDeadline("submission_scan_deadline")
        try:
            kind, value = receiver.recv()
        except EOFError:
            raise RuntimeError("submission_scan_child_exited") from None
    finally:
        receiver.close()
        if child.is_alive():
            child.kill()
        child.join(10)
    if kind == "deadline":
        raise SubmissionScanDeadline("submission_scan_deadline")
    if kind != "ok":
        raise RuntimeError(f"submission_scan_child_error:{value}")
    return value
