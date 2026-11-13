"""Foxrouter pool and SWE-bench Lite task selection for ``fox-single-v1``.

Internal names live under ``fox/`` so they do not collide with the mainland
pay-as-you-go ids in ``MODEL_POOL``. The upstream id is what the gateway
expects. This module does not open a socket and does not read the API key.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
import socket

from .contracts import MODEL_POOL
from .swebench_lite_frame import SWEBENCH_LITE_FRAME, SWEBENCH_LITE_REVISION


EXPERIMENT = "fox-single-v1"
# First scheduled grid. ``test`` stays selectable and is not this run.
DEFAULT_SPLIT = "dev"
TEST_EXTRAPOLATION_INSTANCES = 300
FOX_KEY_ENV = "FOXROUTER_API_KEY"
FOX_API_BASE = "https://foxrouter.org/v1"
FOX_CHAT_URL = FOX_API_BASE + "/chat/completions"
FOX_USER_AGENT = "curl/8.5.0"
# Gateway probe: mimo-v2.6-flash silently cuts ~100k-token prompts to ~25.7k.
# The cap for that model must stay strictly below this many tokens.
MIMO_CONTEXT_HARD_MAX = 70_000
DEFAULT_MAX_TOKENS = 16_384
DEFAULT_TURN_CAP = 24
DEFAULT_READ_TIMEOUT_S = 600
DEFAULT_MAX_ATTEMPTS = 4
TRANSIENT_HTTP = frozenset({429, 502, 503, 504, 520, 521, 522, 523, 524})
FAILURE_TAXONOMY = (
    "provider_error",
    "timeout",
    "context_limit",
    "suspected_truncation",
    "empty_patch",
    "apply_fail",
    "tests_fail",
    "turn_cap",
    "eval_unavailable",
    "budget_cap",
    "payload_missing",
    "image_unavailable",
    "resource_limit",
)
_CF_ERROR = re.compile(r"error code:\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class FoxModel:
    """One gateway model. ``report_name`` is what summaries print.

    ``accepted_aliases`` lists gateway ids that should count as this model even
    when separator and snapshot normalization would not. Alias comparison still
    strips a vendor prefix, so one unprefixed entry also accepts ``fw/<id>``.
    ``p`` is not a separator. ``fox/kimi-k3.1-pro`` lists ``kimi-k3p1-pro``
    because the gateway returns that spelling of ``3.1``. The other scheduled
    models leave the tuple empty.
    """

    internal: str
    upstream: str
    report_name: str
    context_token_cap: int
    max_tokens: int = DEFAULT_MAX_TOKENS
    accepted_aliases: tuple = ()

    def __post_init__(self):
        if not self.internal.startswith("fox/") or self.internal in MODEL_POOL:
            raise ValueError("fox_internal_name_collision")
        if not self.upstream or "/" in self.upstream:
            raise ValueError("fox_upstream_id_rejected")
        if self.internal == "fox/mimo-v2.6-flash" and not (
                type(self.context_token_cap) is int and 0 < self.context_token_cap < MIMO_CONTEXT_HARD_MAX):
            raise ValueError("mimo_context_cap_not_below_70k")
        if type(self.context_token_cap) is not int or self.context_token_cap < 1:
            raise ValueError("invalid_context_cap")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("invalid_max_tokens")
        aliases = self.accepted_aliases
        if type(aliases) is not tuple or any(type(item) is not str or not item.strip() for item in aliases):
            raise ValueError("invalid_accepted_aliases")


# Upstream ids are the gateway's model strings. DeepSeek is ``deepseek-flash``
# in reports; the upstream id stays metadata.
FOX_MODELS = (
    FoxModel("fox/deepseek-flash", "deepseek-v4.1-flash-1017", "deepseek-flash", 128_000),
    FoxModel("fox/kimi-k3.1-pro", "kimi-k3.1-pro", "fox/kimi-k3.1-pro", 128_000,
             accepted_aliases=("kimi-k3p1-pro",)),
    FoxModel("fox/qwen3.8-max", "qwen3.8-max", "fox/qwen3.8-max", 128_000),
    FoxModel("fox/qwen3.8-flash", "qwen3.8-flash", "fox/qwen3.8-flash", 128_000),
    FoxModel("fox/glm-5.3-flash", "glm-5.3-flash", "fox/glm-5.3-flash", 128_000),
    FoxModel("fox/mimo-v2.6-flash", "mimo-v2.6-flash", "fox/mimo-v2.6-flash", 60_000),
)
FOX_BY_INTERNAL = {row.internal: row for row in FOX_MODELS}
# ``deepseek-flash`` is the report name. The direct-endpoint ids are not aliases.
_REPORT_ALIAS = {"deepseek-flash": "fox/deepseek-flash"}

# Official SWE-bench Lite ``dev`` split at the revision pinned in
# docs/models_and_benchmarks.md. Instance id and repo only.
SWEBENCH_LITE_DEV = (
    ("sqlfluff__sqlfluff-1625", "sqlfluff/sqlfluff"),
    ("sqlfluff__sqlfluff-2419", "sqlfluff/sqlfluff"),
    ("sqlfluff__sqlfluff-1733", "sqlfluff/sqlfluff"),
    ("sqlfluff__sqlfluff-1517", "sqlfluff/sqlfluff"),
    ("sqlfluff__sqlfluff-1763", "sqlfluff/sqlfluff"),
    ("marshmallow-code__marshmallow-1359", "marshmallow-code/marshmallow"),
    ("marshmallow-code__marshmallow-1343", "marshmallow-code/marshmallow"),
    ("pvlib__pvlib-python-1707", "pvlib/pvlib-python"),
    ("pvlib__pvlib-python-1072", "pvlib/pvlib-python"),
    ("pvlib__pvlib-python-1606", "pvlib/pvlib-python"),
    ("pvlib__pvlib-python-1854", "pvlib/pvlib-python"),
    ("pvlib__pvlib-python-1154", "pvlib/pvlib-python"),
    ("pylint-dev__astroid-1978", "pylint-dev/astroid"),
    ("pylint-dev__astroid-1333", "pylint-dev/astroid"),
    ("pylint-dev__astroid-1196", "pylint-dev/astroid"),
    ("pylint-dev__astroid-1866", "pylint-dev/astroid"),
    ("pylint-dev__astroid-1268", "pylint-dev/astroid"),
    ("pyvista__pyvista-4315", "pyvista/pyvista"),
    ("pydicom__pydicom-1694", "pydicom/pydicom"),
    ("pydicom__pydicom-1413", "pydicom/pydicom"),
    ("pydicom__pydicom-901", "pydicom/pydicom"),
    ("pydicom__pydicom-1139", "pydicom/pydicom"),
    ("pydicom__pydicom-1256", "pydicom/pydicom"),
)


def context_cap_allowed(internal, cap):
    """Mimo's cap stays below the observed silent-truncation window."""
    if type(cap) is not int or cap < 1:
        return False
    if internal == "fox/mimo-v2.6-flash":
        return cap < MIMO_CONTEXT_HARD_MAX
    return True


def resolve_model(name):
    """Map a CLI name onto a fox model. Direct-endpoint ids are rejected."""
    if not isinstance(name, str) or not name:
        raise ValueError("unknown_fox_model")
    internal = _REPORT_ALIAS.get(name, name)
    row = FOX_BY_INTERNAL.get(internal)
    if row is None:
        if name in MODEL_POOL:
            raise ValueError("use_fox_namespace:" + name)
        raise ValueError("unknown_fox_model:" + name)
    return row


def resolve_models(names):
    """``all`` or a list of internal names. Order follows ``FOX_MODELS``."""
    if names in (None, "all") or names == ["all"] or names == ("all",):
        return FOX_MODELS
    if isinstance(names, str):
        names = [names]
    chosen = []
    seen = set()
    for name in names:
        row = resolve_model(name)
        if row.internal in seen:
            continue
        seen.add(row.internal)
        chosen.append(row)
    if not chosen:
        raise ValueError("fox_models_required")
    return tuple(chosen)


def model_settings(model, *, context_caps=None, max_tokens=None):
    """Apply per-model overrides. Mimo still cannot be set to 70k or above."""
    cap = model.context_token_cap
    if context_caps and model.internal in context_caps:
        cap = context_caps[model.internal]
    if not context_cap_allowed(model.internal, cap):
        raise ValueError("mimo_context_cap_not_below_70k")
    tokens = model.max_tokens if max_tokens is None else max_tokens
    if type(tokens) is not int or tokens < 1:
        raise ValueError("invalid_max_tokens")
    return {"context_token_cap": cap, "max_tokens": tokens}


def _split_rows(split):
    if split == "dev":
        return tuple((instance_id, repo, "dev") for instance_id, repo in SWEBENCH_LITE_DEV)
    if split == "test":
        return tuple((instance_id, repo, "test") for instance_id, repo in SWEBENCH_LITE_FRAME)
    if split == "both":
        return _split_rows("dev") + _split_rows("test")
    raise ValueError("unknown_split")


def split_index():
    """Instance id to ``(repo, split)``. Dev and test do not overlap."""
    index = {}
    for instance_id, repo, split in _split_rows("both"):
        if instance_id in index:
            raise ValueError("split_overlap:" + instance_id)
        index[instance_id] = (repo, split)
    return index


def select_tasks(split, *, limit=None, instance_ids=None, allow_all=False):
    """Choose instances without calling an API.

    ``allow_all`` is required when ``limit`` and ``instance_ids`` are both
    omitted, so a full dev or test run is an explicit choice.
    """
    if split not in {"dev", "test", "both"}:
        raise ValueError("unknown_split")
    catalog = split_index()
    if instance_ids:
        rows = []
        seen = set()
        for instance_id in instance_ids:
            if instance_id not in catalog:
                raise ValueError("unknown_instance:" + str(instance_id))
            repo, row_split = catalog[instance_id]
            if split != "both" and row_split != split:
                raise ValueError("instance_not_in_split:" + instance_id)
            if instance_id in seen:
                continue
            seen.add(instance_id)
            rows.append((instance_id, repo, row_split))
    else:
        rows = list(_split_rows(split))
    if limit is None:
        if not allow_all and not instance_ids:
            raise ValueError("selection_required")
    else:
        if type(limit) is not int or limit < 1:
            raise ValueError("invalid_limit")
        rows = rows[:limit]
    if not rows:
        raise ValueError("empty_selection")
    return tuple(rows)


def estimate_size(*, split, models="all", limit=None, instance_ids=None, allow_all=False):
    """Units are models times instances. This does not call a provider."""
    chosen_models = resolve_models(models)
    tasks = select_tasks(split, limit=limit, instance_ids=instance_ids, allow_all=allow_all)
    return {
        "experiment": EXPERIMENT,
        "split": split,
        "models": len(chosen_models),
        "model_names": [row.report_name for row in chosen_models],
        "instances": len(tasks),
        "units": len(chosen_models) * len(tasks),
        "api_calls": 0,
        "dataset_revision": SWEBENCH_LITE_REVISION,
    }


def estimate_prompt_tokens(messages):
    """Rough prompt size from the JSON the gateway would receive.

    One token per four characters. The context cap uses this estimate so a
    prompt over the cap is never sent unchanged.
    """
    if not isinstance(messages, list):
        return 1
    raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, (len(raw) + 3) // 4)


def apply_context_limit(messages, cap, policy):
    """Trim or refuse. Never return a body whose estimate is over ``cap``.

    ``fail`` (the default) does not call the gateway. ``trim`` drops the oldest
    non-system turns, keeps the system prompt and the latest message, and
    still fails when that remainder is over the cap.
    """
    if policy not in {"fail", "trim"}:
        raise ValueError("invalid_context_policy")
    if not isinstance(messages, list) or not messages:
        raise ValueError("invalid_model_messages")
    before = estimate_prompt_tokens(messages)
    record = {
        "context_action": "within_cap",
        "context_cap": cap,
        "estimated_prompt_tokens_before": before,
        "estimated_prompt_tokens": before,
        "trimmed_messages": 0,
    }
    if before <= cap:
        return list(messages), record
    if policy == "fail":
        record["context_action"] = "context_limit"
        return None, record
    kept = list(messages)
    dropped = 0
    while estimate_prompt_tokens(kept) > cap and len(kept) > 1:
        index = None
        for position, message in enumerate(kept[:-1]):
            if not isinstance(message, dict) or message.get("role") != "system":
                index = position
                break
        if index is None:
            break
        del kept[index]
        dropped += 1
    after = estimate_prompt_tokens(kept)
    record["trimmed_messages"] = dropped
    record["estimated_prompt_tokens"] = after
    if after > cap:
        record["context_action"] = "context_limit"
        return None, record
    record["context_action"] = "trimmed"
    return kept, record


def suspected_truncation(estimated, prompt_tokens):
    """True when the gateway's prompt count is far below the sent estimate.

    The live probe returned ``prompt_tokens=25668`` for a ~100k-token prompt
    with HTTP 200. Small estimate noise is not flagged.
    """
    if type(estimated) is not int or type(prompt_tokens) is not int:
        return False
    if estimated < 8_000 or prompt_tokens < 0:
        return False
    return prompt_tokens * 2 < estimated and estimated - prompt_tokens >= 8_000


def json_safe(value):
    """Copy a JSON value. Non-JSON values become a type tag, not a crash."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return {"unserializable_type": type(value).__name__}


def cloudflare_error_code(payload, raw_text=""):
    """Pull a Cloudflare error code out of a JSON body or an HTML/text body."""
    if isinstance(payload, dict):
        error = payload.get("error")
        if type(error) is int:
            return str(error)
        if isinstance(error, dict) and error.get("code") is not None and not isinstance(error.get("code"), bool):
            return str(error["code"])
        errors = payload.get("errors")
        if isinstance(errors, list) and errors and isinstance(errors[0], dict):
            code = errors[0].get("code")
            if code is not None and not isinstance(code, bool):
                return str(code)
    text = raw_text if isinstance(raw_text, str) else ""
    match = _CF_ERROR.search(text)
    if match:
        return match.group(1)
    return None


def transient_exception_kind(error):
    """``timeout`` or ``connection_reset`` for errors the gateway loop retries."""
    if isinstance(error, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(error, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return "connection_reset"
    reason = getattr(error, "reason", None)
    if reason is not None and reason is not error:
        return transient_exception_kind(reason)
    return None


def is_transient_status(status):
    return type(status) is int and status in TRANSIENT_HTTP


def numeric_amount(value):
    """A raw cost/credit/quota number. Strings and bools are not amounts."""
    if isinstance(value, bool) or type(value) not in (int, float):
        return None
    return float(value)


def pull_gateway_field(payload, usage, key):
    """Return ``cost``, ``credit`` or ``quota`` as the API sent it.

    A top-level field wins. Otherwise the same key inside ``usage`` is used.
    Missing means None, not zero.
    """
    if isinstance(payload, dict) and key in payload:
        return json_safe(payload[key])
    if isinstance(usage, dict) and key in usage:
        return json_safe(usage[key])
    return None


def reasoning_tokens_of(usage, completion):
    """Reported reasoning tokens, or None when the gateway omitted them.

    Kimi can send ``reasoning_content`` and no count. That stays None. A count
    above ``completion_tokens`` is left as reported when it is an int; the raw
    usage object is stored separately and is not rewritten.
    """
    if not isinstance(usage, dict):
        return None
    value = usage.get("reasoning_tokens")
    details = usage.get("completion_tokens_details")
    if value is None and isinstance(details, dict):
        value = details.get("reasoning_tokens")
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or type(value) is not int or value < 0:
        return None
    del completion
    return value


def request_body(model, messages, max_tokens):
    """Non-streaming chat completion. The body model is the upstream id."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("invalid_max_tokens")
    return {
        "model": model.upstream,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }


def auth_headers(key):
    """Bearer auth plus a browser-like User-Agent. Cloudflare 1010 blocks the default Python agent."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": FOX_USER_AGENT,
    }
    if isinstance(key, str) and key:
        headers["Authorization"] = "Bearer " + key
    return headers
