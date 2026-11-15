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
    r"(?:\bexec\b|\bos\.system\b|\bsubprocess\b|\bPopen\b|\bos\.exec[a-z_]*\b)"
    r".{0,160}(?<![\w.])(?<!build/)(?:\./|/workspace/)executable\b",
    re.IGNORECASE | re.DOTALL,
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
        if token in {">", ">>", "<", ">&", "&>"}:
            index += 2 if index + 1 < len(tokens) else 1
            continue
        if isinstance(token, str) and _REDIRECT_TOKEN.fullmatch(token):
            index += 1
            continue
        cleaned.append(token)
        index += 1
    return cleaned


def _simple_commands(line):
    """Split one shell line into argv lists, dropping redirects and ``|| true``."""
    if not isinstance(line, str) or not line.strip():
        return []
    try:
        tokens = shlex.split(line, comments=True, posix=True)
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


def _text_submission_violations(text, *, allow_reference_execution):
    """Violation kinds in one text file.

    Running the reference through its normal interface is allowed only for a
    probe that is outside the submission or excluded from it. Copying,
    wrapping, or reading the reference is never allowed.
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
        for argv in _simple_commands(line):
            if _installs_own_build(argv):
                continue
            if _reads_reference_bytes(argv):
                found.append("reads_reference_bytes")
                break
            if _copies_reference(argv):
                found.append("copies_reference")
                break
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


def submission_reference_scan(root, *, reference_sha256=None, probe_paths=()):
    """Flag a submission that copies, embeds, wraps, or reads the reference.

    ``./executable`` itself is the reference and is not a submitted file.
    A ``compile.sh`` that installs the agent's own ``build/...`` output onto
    ``./executable`` is the legal output path, not a wrapper. A behaviour-probe
    script under ``.pb-probe/`` or passed in ``probe_paths`` (outside the
    submission) may run the reference through its normal interface.
    """
    root = Path(root)
    digest = reference_sha256 if isinstance(reference_sha256, str) and reference_sha256 else reference_digest(root)
    mode_ok = None
    reference = root / REFERENCE_BASENAME
    if reference.is_file() and not reference.is_symlink():
        mode_ok = reference_is_execute_only(reference.stat().st_mode)
    violations = []
    if mode_ok is False:
        violations.append({"kind": "reference_not_execute_only", "path": REFERENCE_BASENAME})
    if not root.is_dir():
        return {
            "ok": not violations,
            "violations": violations,
            "reference_sha256": digest,
            "reference_execute_only": mode_ok,
        }
    for path in sorted(root.rglob("*")):
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
        allow_execution = is_excluded_probe(relative)
        for kind in _text_submission_violations(text, allow_reference_execution=allow_execution):
            violations.append({"kind": kind, "path": relative})
            break
    for probe in probe_paths or ():
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
        for kind in _text_submission_violations(text, allow_reference_execution=True):
            violations.append({"kind": kind, "path": path.name})
            break
    return {
        "ok": not violations,
        "violations": violations,
        "reference_sha256": digest,
        "reference_execute_only": mode_ok,
    }
