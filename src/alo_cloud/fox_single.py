"""Single-model SWE-bench Lite baseline on the Foxrouter pool.

Label ``fox-single-v1``. Output stays in the caller's run directory and is
not an r1, r2, or r3 ledger. No per-token price is assumed. The API key is
read from ``FOXROUTER_API_KEY`` only when ``--live`` sends, and it is not
written into the run metadata.

Offline tests inject ``sender``. The default live sender is the only path
that opens a gateway socket.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import http.client
import json
import os
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request

from .adapters import adapter_for
from .collect import prediction_row, workspace_patch
from .errors import PilotStopped, ProtocolError
from .evaluate import SWEBENCH_PACKAGE, SWEBENCH_REVISION, execute_swebench, swebench_command
from .fox_pool import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_READ_TIMEOUT_S,
    DEFAULT_TURN_CAP,
    DEFAULT_SPLIT,
    EXPERIMENT,
    FAILURE_TAXONOMY,
    FOX_CHAT_URL,
    TEST_EXTRAPOLATION_INSTANCES,
    FOX_KEY_ENV,
    FOX_MODELS,
    apply_context_limit,
    auth_headers,
    cloudflare_error_code,
    estimate_size,
    is_transient_status,
    json_safe,
    model_settings,
    numeric_amount,
    pull_gateway_field,
    reasoning_tokens_of,
    request_body,
    resolve_models,
    select_tasks,
    suspected_truncation,
    transient_exception_kind,
)
from .images import swebench_instance_image
from .prices import charge_if_priced
from .providers import model_match_kind, parse_openai_chat_completion, parse_sse
from .repro import REPRO_CRITERION_ID
from .run import load_public_payload, stage_for
from .sandbox import DockerSandbox, SWE_TESTBED_PATH
from .workspace_prep import resolve_image_head, seed_image_tree


_BANNED_DIR_NAMES = frozenset({
    "r1", "r2", "r3", "alo-r1-run", "alo-r2-run", "alo-r3-run",
})
_STOP_FAILURES = frozenset({
    "provider_error", "timeout", "context_limit", "budget_cap", "resource_limit",
})


class FoxStopped(PilotStopped):
    """A fox unit stopped before a submission. The argument is a taxonomy id."""


def _stamp(moment):
    if not isinstance(moment, datetime):
        raise FoxStopped("provider_error")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now():
    return datetime.now(timezone.utc)


def key_configured(env_name=FOX_KEY_ENV):
    """True when the env var holds a non-empty single-line key. The value is not returned."""
    value = os.environ.get(env_name)
    try:
        return (isinstance(value, str) and value != "" and value.strip() == value
                and "\r" not in value and "\n" not in value and "\0" not in value)
    finally:
        value = None


def _read_key(env_name=FOX_KEY_ENV):
    value = os.environ.get(env_name)
    if not (isinstance(value, str) and value != "" and value.strip() == value
            and "\r" not in value and "\n" not in value and "\0" not in value):
        value = None
        raise FoxStopped("credential_missing")
    return value


def redact_secrets(value, secrets):
    if isinstance(value, dict):
        return {str(key): redact_secrets(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret and secret in value:
                value = value.replace(secret, "[redacted]")
        return value
    return value


def assert_fox_run_dir(path):
    """Refuse directories that would mix this experiment with r1, r2, or r3."""
    resolved = Path(path).resolve()
    if resolved.name.lower() in _BANNED_DIR_NAMES:
        raise FoxStopped("fox_run_dir_conflicts_with_frozen_batch")
    for part in resolved.parts:
        if part.lower() in {"alo-r1-run", "alo-r2-run", "alo-r3-run"}:
            raise FoxStopped("fox_run_dir_conflicts_with_frozen_batch")
    return resolved


def _append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


# A blocking read this close to the deadline is the deadline. Starting another
# socket wait here would run past the attempt budget. ``urllib``'s timeout is
# only the limit for one recv, so trickle and keepalive bytes would otherwise
# keep resetting it.
_FOX_DEADLINE_EPSILON_S = 0.05


class _FoxDeadlineSocket:
    """Socket proxy that bounds the whole attempt, not each blocking call.

    Every send and recv shrinks the underlying timeout to the time still left
    and raises ``TimeoutError`` once that budget is gone. ``makefile`` keeps
    its own refcount so ``HTTPConnection`` can mark the socket closed after
    the headers and the body can still be read through the buffered file.
    """

    def __init__(self, sock, deadline):
        self._sock = sock
        self._deadline = deadline
        self._io_refs = 0
        self._closed = False

    def _remaining(self):
        return self._deadline - time.monotonic()

    def _abort(self):
        sock = self._sock
        if sock is None:
            return
        for action in (
            lambda: sock.shutdown(socket.SHUT_RDWR),
            lambda: sock.close(),
        ):
            try:
                action()
            except OSError:
                pass

    def _arm(self):
        remaining = self._remaining()
        if remaining <= _FOX_DEADLINE_EPSILON_S:
            self._abort()
            raise TimeoutError("fox_read_timeout")
        try:
            self._sock.settimeout(remaining)
        except OSError as error:
            if self._remaining() <= _FOX_DEADLINE_EPSILON_S:
                self._abort()
                raise TimeoutError("fox_read_timeout") from error
            raise

    def _io(self, method, *args, **kwargs):
        self._arm()
        try:
            return method(*args, **kwargs)
        except TimeoutError as error:
            self._abort()
            raise TimeoutError("fox_read_timeout") from error

    def recv(self, *args, **kwargs):
        return self._io(self._sock.recv, *args, **kwargs)

    def recv_into(self, *args, **kwargs):
        return self._io(self._sock.recv_into, *args, **kwargs)

    def send(self, *args, **kwargs):
        return self._io(self._sock.send, *args, **kwargs)

    def sendall(self, data, flags=0):
        view = memoryview(data)
        while len(view):
            sent = self._io(self._sock.send, view, flags)
            if not sent:
                raise TimeoutError("fox_read_timeout")
            view = view[sent:]

    def settimeout(self, value):
        remaining = self._remaining()
        if remaining <= _FOX_DEADLINE_EPSILON_S:
            self._abort()
            raise TimeoutError("fox_read_timeout")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            chosen = remaining
        else:
            chosen = min(float(value), remaining)
        self._sock.settimeout(chosen)

    def gettimeout(self):
        return self._sock.gettimeout()

    def fileno(self):
        return self._sock.fileno()

    def makefile(self, mode="r", buffering=None, *, encoding=None, errors=None, newline=None):
        return socket.socket.makefile(
            self, mode, buffering, encoding=encoding, errors=errors, newline=newline)

    def close(self):
        self._closed = True
        if self._io_refs <= 0:
            self._real_close()

    def _decref_socketios(self):
        if self._io_refs > 0:
            self._io_refs -= 1
        if self._closed:
            self.close()

    def _real_close(self):
        sock = self._sock
        self._sock = None
        if sock is None:
            return
        try:
            sock.close()
        except OSError:
            pass

    def _detach(self):
        """Hand back the raw socket without closing it."""
        sock = self._sock
        self._sock = None
        self._closed = True
        return sock

    def __getattr__(self, name):
        sock = self._sock
        if sock is None:
            raise AttributeError(name)
        return getattr(sock, name)


class _FoxDeadlineTransport:
    """Shared connect step for the HTTP and HTTPS connection subclasses."""

    def _fox_prepare_timeout(self):
        remaining = self._fox_deadline - time.monotonic()
        if remaining <= _FOX_DEADLINE_EPSILON_S:
            raise TimeoutError("fox_read_timeout")
        current = self.timeout
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            self.timeout = remaining
        else:
            self.timeout = min(float(current), remaining)

    def _fox_wrap_socket(self):
        sock = self.sock
        if sock is not None and not isinstance(sock, _FoxDeadlineSocket):
            self.sock = _FoxDeadlineSocket(sock, self._fox_deadline)

    def _fox_take_raw_socket(self):
        wrapped = self.sock
        if isinstance(wrapped, _FoxDeadlineSocket):
            self.sock = wrapped._detach()
        return self.sock


class _FoxDeadlineHTTPConnection(_FoxDeadlineTransport, http.client.HTTPConnection):
    def __init__(self, *args, deadline, **kwargs):
        self._fox_deadline = deadline
        super().__init__(*args, **kwargs)

    def connect(self):
        self._fox_prepare_timeout()
        super().connect()
        self._fox_wrap_socket()


class _FoxDeadlineHTTPSConnection(_FoxDeadlineTransport, http.client.HTTPSConnection):
    def __init__(self, *args, deadline, **kwargs):
        self._fox_deadline = deadline
        super().__init__(*args, **kwargs)

    def connect(self):
        """Connect, and keep the deadline armed while a proxy tunnel is read.

        ``HTTPSConnection.connect`` reads the proxy ``CONNECT`` response before
        returning. Those reads used to sit on the raw socket, so a proxy that
        trickled bytes could outlive ``--read-timeout``. The tunnel is read
        through the deadline socket. TLS is then negotiated on the raw socket,
        because OpenSSL uses the file descriptor rather than ``recv``.
        """
        self._fox_prepare_timeout()
        if not self._tunnel_host:
            super().connect()
            self._fox_wrap_socket()
            return
        tunnel_host = self._tunnel_host
        self._tunnel_host = None
        try:
            http.client.HTTPConnection.connect(self)
        finally:
            self._tunnel_host = tunnel_host
        self._fox_wrap_socket()
        self._tunnel()
        raw = self._fox_take_raw_socket()
        remaining = self._fox_deadline - time.monotonic()
        if remaining <= _FOX_DEADLINE_EPSILON_S:
            raise TimeoutError("fox_read_timeout")
        try:
            raw.settimeout(remaining)
        except OSError as error:
            raise TimeoutError("fox_read_timeout") from error
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=tunnel_host)
        except TimeoutError as error:
            raise TimeoutError("fox_read_timeout") from error
        self._fox_wrap_socket()


class _FoxDeadlineHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, deadline):
        super().__init__()
        self._deadline = deadline

    def http_open(self, req):
        return self.do_open(self._connection, req)

    def _connection(self, host, **kwargs):
        return _FoxDeadlineHTTPConnection(host, deadline=self._deadline, **kwargs)


class _FoxDeadlineHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, deadline):
        super().__init__()
        self._deadline = deadline

    def https_open(self, req):
        return self.do_open(self._connection, req, context=self._context)

    def _connection(self, host, **kwargs):
        return _FoxDeadlineHTTPSConnection(host, deadline=self._deadline, **kwargs)


def _close_quietly(obj):
    close = getattr(obj, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:
        return


def urllib_fox_send(url, body, headers, timeout_s):
    """POST one completion and stop when ``timeout_s`` seconds have elapsed.

    ``timeout_s`` is the wall-clock budget for this attempt (connect, an HTTPS
    proxy ``CONNECT``, request, headers, and body). It is the value of
    ``--read-timeout``, which defaults to 600 seconds. A peer that keeps the
    socket readable does not get another full window per byte. When the budget
    is spent the socket is aborted and
    ``TimeoutError`` propagates, which the retry loop already records as
    ``timeout``.
    """
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST", headers=headers)
    budget = float(timeout_s)
    deadline = time.monotonic() + budget
    opener = urllib.request.build_opener(
        _FoxDeadlineHTTPHandler(deadline),
        _FoxDeadlineHTTPSHandler(deadline),
    )
    try:
        with opener.open(request, timeout=budget) as response:
            return getattr(response, "status", 200), response.read()
    except urllib.error.HTTPError as error:
        try:
            raw = error.read()
        except TimeoutError:
            _close_quietly(error)
            raise
        except Exception:
            raw = b""
        return getattr(error, "code", None), raw


def _decode_payload(raw, upstream):
    if isinstance(raw, (bytes, bytearray)):
        text = raw.decode("utf-8", errors="replace")
    elif isinstance(raw, str):
        text = raw
    elif isinstance(raw, dict):
        return raw, ""
    else:
        return None, ""
    stripped = text.lstrip()
    if stripped.startswith("data:") or (stripped.startswith(":") and "data:" in text):
        parsed = parse_sse(text, upstream)
        return parsed, text
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None, text
    return value if isinstance(value, dict) else None, text


def post_with_retries(model, messages, max_tokens, *, key, sender, timeout_s, sleep, clock,
                      max_attempts=DEFAULT_MAX_ATTEMPTS, max_sends=None, url=FOX_CHAT_URL):
    """POST one completion. Transient failures are retried and every failure is returned.

    ``max_sends`` is the remaining worker-request allowance. Each HTTP attempt,
    including a retry, consumes one send. Retries stop when that allowance is gone.
    """
    if type(max_attempts) is not int or not 1 <= max_attempts <= 8:
        raise FoxStopped("invalid_retry_attempts")
    if max_sends is not None and (type(max_sends) is not int or max_sends < 1):
        raise FoxStopped("turn_cap")
    allowed = max_attempts if max_sends is None else min(max_attempts, max_sends)
    if allowed < 1:
        raise FoxStopped("turn_cap")
    body = request_body(model, messages, max_tokens)
    headers = auth_headers(key)
    retries = []
    last = None
    sends = 0
    for attempt in range(1, allowed + 1):
        sends += 1
        status = None
        kind = None
        payload = None
        raw_text = ""
        try:
            result = sender(url, body, headers, timeout_s)
        except Exception as error:
            kind = transient_exception_kind(error)
            if kind is None:
                raise
            last = {"status": None, "kind": kind, "payload": None, "raw_text": ""}
        else:
            if isinstance(result, tuple) and len(result) == 2:
                status, raw = result
            else:
                status, raw = 200, result
            payload, raw_text = _decode_payload(raw, model.upstream)
            last = {"status": status, "kind": "http", "payload": payload, "raw_text": raw_text}
            if not is_transient_status(status):
                last["sends"] = sends
                last["timestamp"] = _stamp(clock())
                return last, retries, body
            kind = "http"
        retries.append({
            "attempt": attempt,
            "status": status,
            "kind": kind,
            "cloudflare_error": cloudflare_error_code(payload, raw_text),
            "timestamp": _stamp(clock()),
        })
        if attempt < allowed:
            sleep(min(30.0, float(2 ** (attempt - 1))))
    if isinstance(last, dict):
        last["sends"] = sends
        last["timestamp"] = _stamp(clock())
    return last, retries, body


def _message_text(payload):
    if not isinstance(payload, dict):
        return "", False
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "", False
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return "", False
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    reasoning = message.get("reasoning_content")
    return text, isinstance(reasoning, str) and reasoning != ""


def _stored_model_id(value):
    """Keep a model id JSON-serializable. Strings are stored as returned."""
    if value is None or isinstance(value, str):
        return value
    return json_safe(value)


def interpret_call(model, sent_messages, max_tokens, result, retries, *, estimated):
    """One ledger row. Raw usage and gateway cost fields are copied, not priced.

    Usage is copied even when the call is later rejected, so a billed response
    still lands in the unit totals. A cosmetic model id is a warning. A
    different model is ``provider_error``.
    """
    payload = result.get("payload") if isinstance(result, dict) else None
    usage = payload.get("usage") if isinstance(payload, dict) and isinstance(payload.get("usage"), dict) else None
    raw_usage = json_safe(usage) if isinstance(usage, dict) else None
    text, reasoning_present = _message_text(payload)
    returned_model = payload.get("model") if isinstance(payload, dict) and "model" in payload else None
    aliases = getattr(model, "accepted_aliases", ()) or ()
    if returned_model is None:
        identity = "exact"
    else:
        identity = model_match_kind(returned_model, model.upstream, aliases)
    identity_ok = identity != "mismatch"
    parsed = None
    if isinstance(payload, dict) and identity_ok:
        try:
            parsed = parse_openai_chat_completion(payload, model.upstream, aliases)
        except PilotStopped:
            parsed = None
    prompt = completion = total = reasoning = None
    finish = None
    if isinstance(usage, dict):
        if type(usage.get("prompt_tokens")) is int:
            prompt = usage["prompt_tokens"]
        if type(usage.get("completion_tokens")) is int:
            completion = usage["completion_tokens"]
        if type(usage.get("total_tokens")) is int:
            total = usage["total_tokens"]
        reasoning = reasoning_tokens_of(usage, completion)
    if parsed is not None:
        text = parsed["text"]
        prompt = parsed["input_tokens"]
        completion = parsed["output_tokens"]
        total = parsed["total_tokens"]
        reasoning = parsed["reasoning_tokens"]
        finish = parsed["finish_reason"]
        if isinstance(usage, dict):
            reported_reasoning = reasoning_tokens_of(usage, completion)
            if reported_reasoning is not None:
                reasoning = reported_reasoning
    quirk = None
    if completion == 512:
        quirk = "completion_tokens_exactly_512"
    status = result.get("status") if isinstance(result, dict) else None
    truncated = suspected_truncation(estimated, prompt) if type(prompt) is int else False
    parsed_ok = identity_ok and (parsed is not None or bool(text) or reasoning_present)
    ok = status == 200 and parsed_ok
    row = {
        "experiment": EXPERIMENT,
        "internal_model": model.internal,
        "upstream_model": model.upstream,
        "returned_model": _stored_model_id(returned_model),
        "report_model": model.report_name,
        "max_tokens": max_tokens,
        "http_status": status,
        "ok": bool(ok and status == 200),
        "retries": retries,
        "raw_usage": raw_usage,
        "cost": pull_gateway_field(payload, usage, "cost"),
        "credit": pull_gateway_field(payload, usage, "credit"),
        "quota": pull_gateway_field(payload, usage, "quota"),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
        "reasoning_content_present": reasoning_present,
        "usage_quirk": quirk,
        "estimated_prompt_tokens": estimated,
        "suspected_truncation": truncated,
        "finish_reason": finish,
        "cloudflare_error": cloudflare_error_code(payload, result.get("raw_text") or "") if isinstance(result, dict) else None,
        "timestamp": result.get("timestamp") if isinstance(result, dict) else None,
        "sends": result.get("sends") if isinstance(result, dict) else None,
        "text": text if isinstance(text, str) else "",
    }
    if identity == "normalized":
        row["model_alias_normalized"] = {
            "requested": model.upstream,
            "returned": row["returned_model"],
        }
    if not identity_ok:
        row["ok"] = False
        row["failure"] = "provider_error"
        row["model_mismatch"] = True
        return row
    if status != 200 or parsed is None and not row["text"] and not reasoning_present:
        row["ok"] = False
        row["failure"] = "timeout" if any(item.get("kind") == "timeout" for item in retries) and status is None else "provider_error"
        if status is None and retries and retries[-1].get("kind") == "timeout":
            row["failure"] = "timeout"
        elif status is None and retries and retries[-1].get("kind") == "connection_reset":
            row["failure"] = "provider_error"
        elif status != 200:
            row["failure"] = "provider_error"
        elif parsed is None:
            row["failure"] = "provider_error"
    return row


class FoxServices:
    """Tool-loop services for one fox model. Not a CNY ``StudyServices`` ledger."""

    def __init__(self, state_dir, model, *, settings, transport, limits, clock=None,
                 budget_cap=None, spent_cost=0.0, context_policy="fail"):
        self.available_models = tuple(row.internal for row in FOX_MODELS)
        self.model_spec = model
        self.model = model.internal
        self.settings = settings
        self.transport = transport
        self.limits = dict(limits)
        self.context_policy = context_policy
        self.budget_cap = budget_cap
        self.spent_cost = float(spent_cost)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        from .sandbox import Workspace
        self.workspace = Workspace(self.state_dir / "workspace")
        self.sandbox = None
        self.image_head = None
        self.calls = []
        self.edit_log = []
        self.transport_counts = {}
        self.length_stops = 0
        self.tools = 0
        self.current_role = None
        self.current_node = None
        self.active_criterion_id = None
        self._clock = clock or time.monotonic
        self._started = self._clock()
        self._requests = 0
        self._turns = 0
        self._tokens = 0

    def _elapsed(self):
        delta = self._clock() - self._started
        if hasattr(delta, "total_seconds"):
            return float(delta.total_seconds())
        return float(delta)

    def _wall_remaining(self):
        return float(self.limits.get("seconds", 1200)) - self._elapsed()

    def _tokens_left(self):
        return int(self.limits.get("tokens", 900_000)) - self._tokens

    def _ensure_runtime(self):
        if self._wall_remaining() <= 0 or self._tokens_left() <= 0:
            raise FoxStopped("resource_limit")

    def budget_remaining(self, model=None):
        del model
        left = max(0, int(self.limits["requests"]) - self._requests)
        # No CNY field. A missing list price must not look like a zero balance.
        return {
            "requests": left,
            "tokens": max(0, self._tokens_left()),
            "seconds": max(0.0, self._wall_remaining()),
        }

    def _budget_blocked(self):
        if self.budget_cap is None:
            return False
        return self.spent_cost > float(self.budget_cap)

    def complete_for_model(self, model, messages, *, max_tokens=4096):
        if model != self.model_spec.internal:
            raise FoxStopped("model_outside_fox_pool")
        if self._budget_blocked():
            raise FoxStopped("budget_cap")
        self._ensure_runtime()
        remaining_sends = int(self.limits["requests"]) - self._requests
        if remaining_sends < 1:
            raise FoxStopped("turn_cap")
        sent_max = self.settings["max_tokens"] if max_tokens >= 4096 else max_tokens
        if type(sent_max) is not int or sent_max < 1:
            raise FoxStopped("invalid_max_tokens")
        prepared, context = apply_context_limit(
            messages, self.settings["context_token_cap"], self.context_policy)
        if prepared is None:
            self.calls.append({
                "experiment": EXPERIMENT,
                "internal_model": self.model_spec.internal,
                "upstream_model": self.model_spec.upstream,
                "report_model": self.model_spec.report_name,
                "ok": False,
                "failure": "context_limit",
                "retries": [],
                "raw_usage": None,
                "cost": None,
                "credit": None,
                "quota": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "reasoning_tokens": None,
                "returned_model": None,
                "suspected_truncation": False,
                **context,
            })
            raise FoxStopped("context_limit")
        if context["estimated_prompt_tokens"] > self._tokens_left():
            raise FoxStopped("resource_limit")
        self._turns += 1
        try:
            result, retries = self.transport(
                self.model_spec, prepared, sent_max, max_sends=remaining_sends)
        except TypeError:
            result, retries = self.transport(self.model_spec, prepared, sent_max)
        sends = result.get("sends") if isinstance(result, dict) else None
        if type(sends) is not int or sends < 1:
            sends = 1 + len(retries or [])
        self._requests += sends
        row = interpret_call(
            self.model_spec, prepared, sent_max, result, retries,
            estimated=context["estimated_prompt_tokens"])
        row.update(context)
        amount = numeric_amount(row.get("cost"))
        if amount is not None:
            self.spent_cost += amount
        used = row.get("total_tokens")
        if type(used) is not int:
            prompt_n, completion_n = row.get("prompt_tokens"), row.get("completion_tokens")
            if type(prompt_n) is int and type(completion_n) is int:
                used = prompt_n + completion_n
        if type(used) is int and used > 0:
            self._tokens += used
        self.calls.append(row)
        if not row.get("ok"):
            failure = row.get("failure") or "provider_error"
            raise FoxStopped(failure if failure in _STOP_FAILURES else "provider_error")
        reply = {
            "text": row["text"],
            "input_tokens": row["prompt_tokens"] or 0,
            "output_tokens": row["completion_tokens"] or 0,
            "reasoning_tokens": row["reasoning_tokens"],
            "finish_reason": row.get("finish_reason"),
            "length_stop": row.get("finish_reason") == "length" and not str(row["text"]).strip(),
            "vendor_error": False,
        }
        return reply

    def diff_nonempty(self):
        return any(row.get("paths") for row in self.edit_log)

    def _tool(self, action):
        self._ensure_runtime()
        if self.tools >= int(self.limits.get("tools", 120)):
            raise FoxStopped("turn_cap")
        self.tools += 1
        return action()

    def list_files(self, path=None):
        if path is None:
            return self._tool(self.workspace.list_files)
        return self._tool(lambda: self.workspace.list_files(path))

    def read_file(self, path, start_line=None, end_line=None):
        return self._tool(lambda: self.workspace.read_file(path, start_line=start_line, end_line=end_line))

    def write_file(self, path, content, append=False):
        result = self._tool(lambda: self.workspace.write_file(path, content, append=append))
        if isinstance(result, dict) and result.get("path"):
            self.edit_log.append({"paths": [result["path"]]})
        return result

    def edit_file(self, path, old_str, new_str):
        result = self._tool(lambda: self.workspace.edit_file(path, old_str, new_str))
        if isinstance(result, dict) and result.get("path"):
            self.edit_log.append({"paths": [result["path"]]})
        return result

    def search(self, text):
        return self._tool(lambda: self.workspace.search(text))

    def clamp_timeout(self, timeout_s):
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
            raise ProtocolError("public action timeout must be numeric")
        if timeout_s > 120:
            timeout_s = 120
            notes = ["timeout clamped to 120"]
        else:
            notes = []
        remaining = self._wall_remaining()
        if remaining <= 0:
            raise FoxStopped("resource_limit")
        if timeout_s > remaining:
            return remaining, notes + ["timeout clamped to remaining unit seconds"]
        return timeout_s, notes

    def command(self, argv, timeout_s=30):
        if self.sandbox is None:
            raise PilotStopped("sandbox_not_configured")

        def run():
            return self.sandbox.command(argv, timeout_s)

        return self._tool(run)

    def execute_code(self, language, code, timeout_s=30):
        if language in {"python", "py", "python3"}:
            argv = ["python3", "-c", code if isinstance(code, str) else ""]
        elif language in {"sh", "shell", "bash"}:
            argv = ["sh", "-c", code if isinstance(code, str) else ""]
        else:
            raise ProtocolError("unsupported_execute_language")
        return self.command(argv, timeout_s=timeout_s)


def _sum_amount(rows, key):
    total = 0.0
    seen = False
    for row in rows:
        amount = numeric_amount(row.get(key))
        if amount is None:
            continue
        seen = True
        total += amount
    return total if seen else None


def _sum_tokens(rows, key):
    total = 0
    seen = False
    for row in rows:
        value = row.get(key)
        if type(value) is int and value >= 0:
            seen = True
            total += value
    return total if seen else None


def classify_failure(record):
    """One primary reason for an unresolved instance."""
    if record.get("resolved") is True:
        return None
    stop = record.get("stop_reason")
    if stop in _STOP_FAILURES:
        return stop
    if record.get("diff_failed"):
        return "apply_fail"
    if record.get("image_error"):
        return "image_unavailable"
    if record.get("payload_missing"):
        return "payload_missing"
    patch = record.get("patch_bytes") or 0
    if record.get("termination") == "turn_cap" and patch <= 0:
        return "turn_cap"
    if patch <= 0:
        return "empty_patch"
    if record.get("resolved") is False:
        return "tests_fail"
    if record.get("suspected_truncation") and record.get("resolved") is None:
        return "suspected_truncation"
    if record.get("resolved") is None:
        return "eval_unavailable"
    return "provider_error"


def unit_eval_dir(run_dir, unit_id):
    """One harness output directory per unit. Models must not share a report tree."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(unit_id))[:200]
    path = Path(run_dir) / "eval" / (safe or "unit")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_resolved(root, instance_id):
    """Read ``resolved`` only from ``root``. Callers pass that unit's eval directory."""
    base = Path(root)
    if not base.is_dir():
        return None
    for path in sorted(base.rglob("*.json")):
        if path.name in {"manifest.json", "summary.json"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        row = data.get(instance_id)
        if isinstance(row, dict) and "resolved" in row:
            return bool(row["resolved"])
        if data.get("instance_id") == instance_id and "resolved" in data:
            return bool(data["resolved"])
    return None


def score_prediction(run_dir, *, unit_id, instance_id, split, patch, execute, scorer=None):
    """Write the official prediction row and, when asked, run the SWE-bench harness."""
    row = prediction_row(instance_id, unit_id, patch)
    unit_path = Path(run_dir) / "sealed" / "swe_units" / f"{unit_id}.jsonl"
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    command = swebench_command(
        unit_path, run_id=("fox" + unit_id.replace("/", "_"))[:80],
        instance_ids=(instance_id,), split=split, dataset_revision=SWEBENCH_REVISION)
    result = {"command": command, "split": split, "status": "not_executed", "resolved": None}
    if scorer is not None:
        scored = scorer(command, row)
        if isinstance(scored, dict):
            result.update(scored)
        return result
    if not execute or not str(patch or "").strip():
        return result
    try:
        import importlib.metadata as metadata
        version = metadata.version("swebench")
    except metadata.PackageNotFoundError:
        result["status"] = "swebench_package_missing"
        return result
    if version != SWEBENCH_PACKAGE:
        result["status"] = "swebench_version_mismatch"
        result["observed_version"] = version
        return result
    cwd = unit_eval_dir(run_dir, unit_id)
    harness = execute_swebench(command, cwd=cwd)
    result["harness_status"] = harness.get("status")
    result["returncode"] = harness.get("returncode")
    resolved = _find_resolved(cwd, instance_id)
    result["resolved"] = resolved
    result["status"] = "scored" if resolved is not None else harness.get("status") or "eval_unavailable"
    return result


def _git_identity():
    return ["-c", "user.email=fox-single@localhost", "-c", "user.name=fox-single"]


def _seed_offline_fixture(services):
    """One committed file so an offline chain can edit, diff, and score.

    Live runs do not use this. They copy the instance image.
    """
    root = services.workspace.root
    if (root / ".git").is_dir() or any(root.iterdir()):
        return
    import subprocess
    (root / "bug.py").write_text("def f():\n    return 0\n")
    identity = _git_identity()
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), *identity, "add", "bug.py"], check=True)
    subprocess.run(["git", "-C", str(root), *identity, "commit", "-qm", "init"], check=True)
    shown = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    services.image_head = shown.stdout.strip()


def prepare_workspace(services, *, instance_id, repo, base_commit, live, seed_images, git_runner=None):
    """Seed from the local SWE image when live. Offline runs get a one-file git fixture."""
    root = services.workspace.root
    if not live or not seed_images:
        if not live:
            _seed_offline_fixture(services)
        if not (root / ".git").is_dir() and not any(root.iterdir()):
            root.mkdir(parents=True, exist_ok=True)
        return None
    image = swebench_instance_image(instance_id)
    try:
        seed_image_tree(image, "/testbed", root)
    except PilotStopped:
        return "image_unavailable"
    if isinstance(base_commit, str) and base_commit:
        head = resolve_image_head(root, base_commit, git=git_runner)
        if not head:
            return "image_unavailable"
        services.image_head = head
    services.sandbox = DockerSandbox(
        root, image_resolver=lambda image=image: image, permit_execution=True,
        workdir="/testbed", mount_points=("/testbed",), path=SWE_TESTBED_PATH)
    return None


def _public_stage(data_dir, instance_id, repo, *, live):
    payload = load_public_payload(data_dir, "SWE-bench_Lite", instance_id)
    if payload is None:
        if live:
            return None, True
        payload = {"task_id": instance_id, "repo": repo,
                   "note": "offline placeholder; benchmark text is not loaded"}
    else:
        payload = dict(payload)
        payload.setdefault("repo", repo)

    class _Unit:
        task_id = instance_id
        track = "SWE-bench_Lite"
        phase = "B0"
        arm = EXPERIMENT
        stratum = ""

    return stage_for(_Unit(), payload, dry_run=not live), False


def run_instance(run_dir, model, task, *, settings, turn_cap, transport, live, data_dir,
                 execute, scorer=None, context_policy="fail", budget_cap=None, spent_cost=0.0,
                 seed_images=True, clock=None):
    """One model on one instance: tool loop, patch, SWE score."""
    instance_id, repo, split = task
    unit_id = f"{EXPERIMENT}.{model.internal.replace('/', '-')}.{instance_id}"
    record = {
        "experiment": EXPERIMENT,
        "schema": "fox-single-v1",
        "unit_id": unit_id,
        "internal_model": model.internal,
        "upstream_model": model.upstream,
        "report_model": model.report_name,
        "instance_id": instance_id,
        "repo": repo,
        "split": split,
        "turn_cap": turn_cap,
        "max_attempts": 1,
        "criterion_id": REPRO_CRITERION_ID,
        "protocol_id": "fair-control-v1",
        "request_accounting": "worker_model_calls_against_unit_cap",
        "planner_requests": 0,
        "verifier_requests": 0,
        "topology": "single_agent",
        "list_price": charge_if_priced(model.internal, {"input_tokens": 1, "output_tokens": 1}),
        "started_at": _stamp(clock() if clock else _now()),
    }
    if record["list_price"] is not None:
        raise FoxStopped("fox_model_must_not_have_list_price")
    services = FoxServices(
        Path(run_dir) / "units" / unit_id / "state", model,
        settings=settings, transport=transport,
        limits={"requests": turn_cap, "tokens": 900_000, "tools": 120, "seconds": 1200},
        budget_cap=budget_cap, spent_cost=spent_cost, context_policy=context_policy)
    stage, missing = _public_stage(data_dir, instance_id, repo, live=live)
    if missing:
        record.update(status="stopped", stop_reason="payload_missing", payload_missing=True,
                      patch_bytes=0, resolved=None, calls=0)
        record["failure"] = classify_failure(record)
        return record, services
    image_error = prepare_workspace(
        services, instance_id=instance_id, repo=repo,
        base_commit=(stage.get("public") or {}).get("base_commit") if isinstance(stage, dict) else None,
        live=live, seed_images=seed_images)
    if image_error:
        record.update(status="stopped", image_error=image_error, patch_bytes=0, resolved=None, calls=0)
        record["failure"] = classify_failure(record)
        return record, services
    adapter = adapter_for("G1", {
        "fixed_model": model.internal,
        "max_turns": turn_cap,
        "criterion_id": REPRO_CRITERION_ID,
        "protocol_id": "fair-control-v1",
        "request_accounting": "worker_model_calls_against_unit_cap",
        "max_attempts": 1,
        "planner_requests": 0,
        "verifier_requests": 0,
        "topology": "single_agent",
    })
    outcome = None
    stop_reason = None
    try:
        outcome = adapter.run_stage(stage, services)
    except FoxStopped as error:
        stop_reason = str(error)
    except PilotStopped as error:
        stop_reason = "provider_error" if str(error) != "sandbox_not_configured" else "provider_error"
        record["pilot_stopped"] = str(error)
    termination = outcome.get("termination") if isinstance(outcome, dict) else None
    if stop_reason == "turn_cap":
        termination = "turn_cap"
    diff_failed = False
    patch = ""
    base = services.image_head or ""
    if not base and (services.workspace.root / ".git").is_dir():
        import subprocess
        shown = subprocess.run(
            ["git", "-C", str(services.workspace.root), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True)
        if shown.returncode == 0:
            base = shown.stdout.strip()
    if base:
        try:
            patch = workspace_patch(services.workspace.root, base)
        except Exception:
            diff_failed = True
            patch = ""
    elif services.diff_nonempty():
        diff_failed = True
    checkpoint = Path(run_dir) / "checkpoints" / unit_id
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "model.patch").write_text(patch or "", encoding="utf-8")
    (checkpoint / "calls.json").write_text(
        json.dumps(redact_secrets(services.calls, _live_secrets()), ensure_ascii=False, indent=2),
        encoding="utf-8")
    scored = score_prediction(
        run_dir, unit_id=unit_id, instance_id=instance_id, split=split, patch=patch or "",
        execute=execute and bool(str(patch or "").strip()), scorer=scorer)
    record.update(
        status="stopped" if stop_reason else (outcome.get("status") if isinstance(outcome, dict) else "failed"),
        termination=termination,
        stop_reason=stop_reason,
        repro_signal=getattr(adapter, "public_signal", None),
        functional_pass=getattr(adapter, "functional_pass", None),
        protocol_completed=getattr(adapter, "protocol_completed", None),
        patch_bytes=len((patch or "").encode("utf-8")),
        diff_failed=diff_failed,
        resolved=scored.get("resolved"),
        eval_status=scored.get("status"),
        calls=len(services.calls),
        prompt_tokens=_sum_tokens(services.calls, "prompt_tokens"),
        completion_tokens=_sum_tokens(services.calls, "completion_tokens"),
        reasoning_tokens=_sum_tokens(services.calls, "reasoning_tokens"),
        raw_cost_sum=_sum_amount(services.calls, "cost"),
        raw_credit_sum=_sum_amount(services.calls, "credit"),
        raw_quota_sum=_sum_amount(services.calls, "quota"),
        retries=sum(len(call.get("retries") or []) for call in services.calls),
        suspected_truncation=any(bool(call.get("suspected_truncation")) for call in services.calls),
        context_actions=[call.get("context_action") for call in services.calls if call.get("context_action")],
        spent_cost=services.spent_cost,
    )
    record["failure"] = classify_failure(record)
    return record, services


# Every key env var a fox row may read. Redaction covers all of them.
FOX_SECRET_ENV_NAMES = (FOX_KEY_ENV, "GLM_API_KEY")


def _live_secrets():
    values = []
    for name in FOX_SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if isinstance(value, str) and value and value not in values:
            values.append(value)
        value = None
    return values


def _transport_for(*, live, sender, timeout_s, sleep, clock, max_attempts, key_envs=None):
    """``key_envs`` names the key env vars the selected rows read (default: Foxrouter only)."""
    if sender is None and not live:
        raise FoxStopped("fox_sender_required")
    required = (FOX_KEY_ENV,) if key_envs is None else tuple(key_envs)
    if live and sender is None and not all(key_configured(name) for name in required):
        raise FoxStopped("credential_missing")

    def transport(model, messages, max_tokens, max_sends=None):
        env_name = getattr(model, "key_env", FOX_KEY_ENV)
        key = _read_key(env_name) if live and sender is None else os.environ.get(env_name)
        try:
            if isinstance(key, str) and key:
                blob = json.dumps(messages, ensure_ascii=False)
                if key in blob:
                    raise FoxStopped("credential_in_prompt")
            post = sender or urllib_fox_send
            try:
                result, retries, _body = post_with_retries(
                    model, messages, max_tokens, key=key if isinstance(key, str) else "",
                    sender=post, timeout_s=timeout_s, sleep=sleep, clock=clock,
                    max_attempts=max_attempts, max_sends=max_sends,
                    url=getattr(model, "chat_url", FOX_CHAT_URL))
            except FoxStopped:
                raise
            except (OSError, urllib.error.URLError) as error:
                kind = transient_exception_kind(error) or "transport"
                return {"status": None, "payload": None, "raw_text": ""}, [{
                    "attempt": 1,
                    "status": None,
                    "kind": kind,
                    "cloudflare_error": None,
                    "timestamp": _stamp(clock()),
                }]
            return result, retries
        finally:
            key = None

    return transport


_MANIFEST_LOCKED = (
    "split", "turn_cap", "max_attempts", "criterion_id", "protocol_id",
    "context_policy", "budget_cap", "dataset_revision", "live",
)


def _same_manifest_value(left, right):
    if left == right:
        return True
    if numeric_amount(left) is not None and numeric_amount(right) is not None:
        return numeric_amount(left) == numeric_amount(right)
    return False


def _merge_manifest(existing, payload):
    """Keep the recorded schedule. A resume may add units, not relabel finished ones."""
    if existing.get("experiment") != EXPERIMENT:
        raise FoxStopped("fox_run_dir_experiment_mismatch")
    for key in _MANIFEST_LOCKED:
        if key in existing and not _same_manifest_value(existing.get(key), payload.get(key)):
            raise FoxStopped("fox_run_settings_mismatch")
    known = {row.get("internal_model"): row for row in existing.get("models") or [] if isinstance(row, dict)}
    merged_models = list(existing.get("models") or [])
    for row in payload.get("models") or []:
        if not isinstance(row, dict):
            continue
        previous = known.get(row.get("internal_model"))
        if previous is None:
            merged_models.append(row)
            continue
        for field in ("upstream_model", "report_model", "context_token_cap", "max_tokens"):
            if field in previous and previous.get(field) != row.get(field):
                raise FoxStopped("fox_run_settings_mismatch")
    instances = list(existing.get("instances") or [])
    for instance_id in payload.get("instances") or []:
        if instance_id not in instances:
            instances.append(instance_id)
    payload = dict(payload)
    payload["models"] = merged_models
    payload["instances"] = instances
    payload["units"] = max(
        int(existing.get("units") or 0),
        len(merged_models) * len(instances),
    )
    return payload


def _write_manifest(run_dir, payload):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        payload = _merge_manifest(existing, payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cumulative_raw_cost(records):
    """Sum each unit's own raw cost once. ``spent_cost`` is a running total and is not added again."""
    latest = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        unit_id = row.get("unit_id")
        if isinstance(unit_id, str) and unit_id:
            latest[unit_id] = row
    total = 0.0
    for row in latest.values():
        amount = numeric_amount(row.get("raw_cost_sum"))
        if amount is not None:
            total += amount
    return total


def run_schedule(run_dir, *, split, models="all", limit=None, instance_ids=None, allow_all=False,
                 turn_cap=DEFAULT_TURN_CAP, live=False, sender=None, execute=False, scorer=None,
                 data_dir=None, context_policy="fail", context_caps=None, max_tokens=None,
                 budget_cap=None, seed_images=True, timeout_s=DEFAULT_READ_TIMEOUT_S,
                 max_attempts=DEFAULT_MAX_ATTEMPTS, sleep=None, clock=None, smoke=False):
    """Run the selected model × instance grid. ``sender`` keeps this offline."""
    if smoke and limit is None and not instance_ids:
        limit = 1
    if smoke and type(limit) is int and limit > 2:
        raise FoxStopped("smoke_limit_is_1_or_2")
    if smoke and instance_ids and len(tuple(instance_ids)) > 2:
        raise FoxStopped("smoke_limit_is_1_or_2")
    run_dir = assert_fox_run_dir(run_dir)
    chosen = resolve_models(models)
    tasks = select_tasks(split, limit=limit, instance_ids=instance_ids, allow_all=allow_all or bool(instance_ids))
    if smoke and len(tasks) > 2:
        raise FoxStopped("smoke_limit_is_1_or_2")
    clock = clock or _now
    sleep = sleep or time.sleep
    settings = {model.internal: model_settings(model, context_caps=context_caps, max_tokens=max_tokens)
                for model in chosen}
    manifest = {
        "experiment": EXPERIMENT,
        "schema": "fox-single-v1",
        "split": split,
        "models": [{
            "internal_model": model.internal,
            "upstream_model": model.upstream,
            "report_model": model.report_name,
            "context_token_cap": settings[model.internal]["context_token_cap"],
            "max_tokens": settings[model.internal]["max_tokens"],
        } for model in chosen],
        "instances": [task[0] for task in tasks],
        "units": len(chosen) * len(tasks),
        "turn_cap": turn_cap,
        "max_attempts": 1,
        "criterion_id": REPRO_CRITERION_ID,
        "protocol_id": "fair-control-v1",
        "context_policy": context_policy,
        "budget_cap": budget_cap,
        "prices": "none",
        "api_key_env": FOX_KEY_ENV,
        "api_key_recorded": False,
        "live": bool(live and sender is None),
        "smoke": bool(smoke),
        "dataset_revision": SWEBENCH_REVISION,
    }
    _write_manifest(run_dir, manifest)
    transport = _transport_for(
        live=live, sender=sender, timeout_s=timeout_s, sleep=sleep, clock=clock,
        max_attempts=max_attempts)
    prior = _load_jsonl(run_dir / "records.jsonl")
    done = {row.get("unit_id") for row in prior}
    spent = _cumulative_raw_cost(prior)
    written = []
    secrets = _live_secrets()
    halt = False
    for model in chosen:
        if halt:
            break
        for task in tasks:
            unit_id = f"{EXPERIMENT}.{model.internal.replace('/', '-')}.{task[0]}"
            if unit_id in done:
                continue
            if budget_cap is not None and spent > float(budget_cap):
                halt = True
                break
            record, services = run_instance(
                run_dir, model, task, settings=settings[model.internal], turn_cap=turn_cap,
                transport=transport, live=live, data_dir=data_dir, execute=execute, scorer=scorer,
                context_policy=context_policy, budget_cap=budget_cap, spent_cost=spent,
                seed_images=seed_images, clock=clock)
            record = redact_secrets(record, secrets)
            _append_jsonl(run_dir / "records.jsonl", record)
            for call in services.calls:
                call = redact_secrets(dict(call), secrets)
                call["unit_id"] = record["unit_id"]
                call["instance_id"] = task[0]
                call["split"] = task[2]
                call.pop("text", None)
                _append_jsonl(run_dir / "calls.jsonl", call)
                for retry in call.get("retries") or []:
                    _append_jsonl(run_dir / "failures.jsonl", {
                        "experiment": EXPERIMENT,
                        "unit_id": record["unit_id"],
                        "kind": "retry",
                        **retry,
                    })
                if call.get("failure"):
                    failure_row = {
                        "experiment": EXPERIMENT,
                        "unit_id": record["unit_id"],
                        "kind": call["failure"],
                        "http_status": call.get("http_status"),
                        "cloudflare_error": call.get("cloudflare_error"),
                        "timestamp": call.get("timestamp"),
                        "suspected_truncation": bool(call.get("suspected_truncation")),
                        "upstream_model": call.get("upstream_model"),
                        "returned_model": call.get("returned_model"),
                    }
                    if call.get("model_mismatch"):
                        failure_row["model_mismatch"] = True
                    _append_jsonl(run_dir / "failures.jsonl", failure_row)
            if record.get("failure"):
                unit_failure = {
                    "experiment": EXPERIMENT,
                    "unit_id": record["unit_id"],
                    "kind": record["failure"],
                    "report_model": record["report_model"],
                    "split": record["split"],
                }
                mismatched = next((call for call in services.calls if call.get("model_mismatch")), None)
                if mismatched is not None:
                    unit_failure["upstream_model"] = mismatched.get("upstream_model")
                    unit_failure["returned_model"] = mismatched.get("returned_model")
                    unit_failure["model_mismatch"] = True
                _append_jsonl(run_dir / "failures.jsonl", unit_failure)
            if record.get("spent_cost") is not None:
                spent = float(record["spent_cost"])
            written.append(record)
    return {"experiment": EXPERIMENT, "wrote": len(written), "units": manifest["units"], "run_dir": str(run_dir)}


def _usage_number(value):
    if type(value) is int:
        return float(value)
    return numeric_amount(value)


def _sample_spread(values):
    """Mean and sample standard deviation. One observation has spread 0."""
    nums = [item for item in (_usage_number(value) for value in values) if item is not None]
    if not nums:
        return None
    count = len(nums)
    mean = sum(nums) / count
    if count == 1:
        stdev = 0.0
    else:
        variance = sum((item - mean) ** 2 for item in nums) / (count - 1)
        stdev = variance ** 0.5
    return {"n": count, "mean": mean, "stdev": stdev, "min": min(nums), "max": max(nums)}


def _scale_to_target(stat, instances):
    if stat is None:
        return {"available": False, "total_at_mean": None, "spread": None}
    return {
        "available": True,
        "total_at_mean": stat["mean"] * instances,
        "spread": stat["stdev"] * instances,
        "per_instance_mean": stat["mean"],
        "per_instance_stdev": stat["stdev"],
    }


def extrapolate_test_300(records):
    """Project a 300-instance test split from observed per-instance usage.

    This is an estimate, not a measurement of the test split. For each model,
    per-instance prompt, completion, and reasoning tokens, plus raw cost and
    credit, are summarized as mean ± sample standard deviation. The test-300
    total is that mean times 300. The reported spread is the per-instance
    standard deviation times 300. It is not a confidence interval.
    """
    target = TEST_EXTRAPOLATION_INSTANCES
    fields = (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("reasoning_tokens", "reasoning_tokens"),
        ("raw_cost", "raw_cost_sum"),
        ("raw_credit", "raw_credit_sum"),
    )
    by_model = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("experiment") not in (None, EXPERIMENT):
            continue
        name = record.get("report_model")
        if not isinstance(name, str) or not name:
            continue
        bucket = by_model.setdefault(name, {
            "model": name,
            "internal_model": record.get("internal_model"),
            "upstream_model": record.get("upstream_model"),
            "records": [],
            "splits": set(),
            "instance_ids": set(),
        })
        bucket["records"].append(record)
        if isinstance(record.get("split"), str):
            bucket["splits"].add(record["split"])
        if isinstance(record.get("instance_id"), str):
            bucket["instance_ids"].add(record["instance_id"])
    rows = []
    order = [model.report_name for model in FOX_MODELS]
    names = order + [name for name in by_model if name not in order]
    for name in names:
        bucket = by_model.get(name)
        observed = [] if bucket is None else bucket["records"]
        per_instance = {}
        projected = {}
        for label, key in fields:
            stat = _sample_spread([row.get(key) for row in observed])
            per_instance[label] = stat
            projected[label] = _scale_to_target(stat, target)
        model_row = next((model for model in FOX_MODELS if model.report_name == name), None)
        rows.append({
            "model": name,
            "internal_model": None if bucket is None else bucket["internal_model"],
            "upstream_model": None if model_row is None else model_row.upstream,
            "observed_records": len(observed),
            "observed_instances": 0 if bucket is None else len(bucket["instance_ids"]),
            "observed_splits": [] if bucket is None else sorted(bucket["splits"]),
            "per_instance": per_instance,
            "test_300_estimate": projected,
        })
    return {
        "label": "estimate",
        "kind": "extrapolation_from_observed_per_instance_usage",
        "measured_test_split": False,
        "target_split": "test",
        "target_instances": target,
        "method": (
            "per-instance mean ± sample standard deviation, multiplied by 300. "
            "Spread is stdev × 300, not a confidence interval."
        ),
        "by_model": rows,
    }


def aggregate_report(records, calls):
    """Per report-model × split totals. Raw cost is summed as returned."""
    groups = {}
    for record in records:
        if record.get("experiment") not in (None, EXPERIMENT):
            continue
        key = (record.get("report_model"), record.get("split"))
        bucket = groups.setdefault(key, {
            "model": record.get("report_model"),
            "internal_model": record.get("internal_model"),
            "upstream_model": record.get("upstream_model"),
            "split": record.get("split"),
            "instances": 0,
            "resolved": 0,
            "scored": 0,
            "unscored": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "reasoning_tokens_reported": False,
            "raw_cost_sum": None,
            "raw_credit_sum": None,
            "raw_quota_sum": None,
            "retries": 0,
            "suspected_truncation": 0,
            "failure_taxonomy": {name: 0 for name in FAILURE_TAXONOMY},
        })
        bucket["instances"] += 1
        if record.get("resolved") is True:
            bucket["resolved"] += 1
            bucket["scored"] += 1
        elif record.get("resolved") is False:
            bucket["scored"] += 1
        else:
            bucket["unscored"] += 1
        for token_key in ("prompt_tokens", "completion_tokens"):
            if type(record.get(token_key)) is int:
                bucket[token_key] += record[token_key]
        if type(record.get("reasoning_tokens")) is int:
            bucket["reasoning_tokens"] += record["reasoning_tokens"]
            bucket["reasoning_tokens_reported"] = True
        for amount_key, raw_key in (
            ("raw_cost_sum", "raw_cost_sum"),
            ("raw_credit_sum", "raw_credit_sum"),
            ("raw_quota_sum", "raw_quota_sum"),
        ):
            amount = numeric_amount(record.get(raw_key))
            if amount is None:
                continue
            bucket[amount_key] = (bucket[amount_key] or 0.0) + amount
        bucket["retries"] += int(record.get("retries") or 0)
        if record.get("suspected_truncation"):
            bucket["suspected_truncation"] += 1
        failure = record.get("failure")
        if failure:
            bucket["failure_taxonomy"][failure] = bucket["failure_taxonomy"].get(failure, 0) + 1
    # Calls can carry cost that a record already summed. Prefer record sums.
    # If a group has no record token totals, fill from calls.
    rows = []
    for key in sorted(groups, key=lambda item: (str(item[1]), str(item[0]))):
        bucket = groups[key]
        if not bucket["reasoning_tokens_reported"]:
            bucket["reasoning_tokens"] = None
        bucket.pop("reasoning_tokens_reported", None)
        instances = bucket["instances"]
        scored = bucket["scored"]
        unscored = bucket["unscored"]
        # A rate is a score only when every attempted instance was actually evaluated.
        if instances and unscored == 0:
            bucket["resolved_rate"] = bucket["resolved"] / instances
        else:
            bucket["resolved_rate"] = None
        bucket["resolved_rate_bounds"] = {
            "lower": (bucket["resolved"] / instances) if instances else None,
            "upper": ((bucket["resolved"] + unscored) / instances) if instances else None,
            "label": "measured" if unscored == 0 else "estimate_bounds",
        }
        bucket["cost_units"] = "as_returned"
        rows.append(bucket)
    del calls
    summary = {
        "experiment": EXPERIMENT,
        "schema": "fox-single-v1-summary",
        "prices": "none",
        "by_model_split": rows,
        "test_300_extrapolation": extrapolate_test_300(records),
    }
    return summary


def write_report(run_dir):
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment") != EXPERIMENT:
            raise FoxStopped("fox_run_dir_experiment_mismatch")
        if manifest.get("api_key_recorded"):
            raise FoxStopped("api_key_must_not_be_recorded")
    records = _load_jsonl(run_dir / "records.jsonl")
    calls = _load_jsonl(run_dir / "calls.jsonl")
    summary = aggregate_report(records, calls)
    summary["instances_recorded"] = len(records)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parse_caps(values):
    caps = {}
    for item in values or []:
        if "=" not in item:
            raise FoxStopped("invalid_context_cap")
        name, raw = item.split("=", 1)
        model = resolve_models([name])[0]
        if not raw.isdigit():
            raise FoxStopped("invalid_context_cap")
        caps[model.internal] = int(raw)
    return caps


def _parse_max_tokens(value):
    if value is None:
        return None
    if not str(value).isdigit():
        raise FoxStopped("invalid_max_tokens")
    return int(value)


def _add_selection(parser):
    parser.add_argument("--split", choices=("dev", "test", "both"), default=None,
                        help="SWE-bench Lite split. Default is dev (23). test (300) stays selectable.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--instance-ids", nargs="+", default=None)
    parser.add_argument("--all", action="store_true", help="Use every instance in the split. Required for a full run.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="fox/... names, deepseek-flash, or all")


def _selection_from_args(args, *, default_models):
    split = args.split or DEFAULT_SPLIT
    return split, args.models or default_models, args.limit, args.instance_ids, bool(args.all)


def main(argv=None):
    parser = argparse.ArgumentParser(description="fox-single-v1 SWE-bench Lite baseline. Does not call an API unless --live.")
    sub = parser.add_subparsers(dest="command", required=True)

    estimate = sub.add_parser("estimate", help="Print models × instances, or extrapolate test-300 from a run. No API call.")
    _add_selection(estimate)
    estimate.add_argument("--from-run-dir", default=None,
                          help="Extrapolate test-300 cost from observed per-instance usage in this run. Labeled as an estimate.")

    run = sub.add_parser("run", help="Run the selected grid.")
    _add_selection(run)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--live", action="store_true")
    run.add_argument("--execute", action="store_true", help="Run the SWE-bench harness when swebench==4.1.0 is installed.")
    run.add_argument("--data-dir", default=None)
    run.add_argument("--turn-cap", type=int, default=DEFAULT_TURN_CAP)
    run.add_argument("--max-tokens", default=None)
    run.add_argument("--context-policy", choices=("fail", "trim"), default="fail")
    run.add_argument("--context-cap", action="append", default=[])
    run.add_argument("--budget-cap", type=float, default=None,
                     help="Optional cap on summed raw cost fields. Default off. Missing cost does not block.")
    run.add_argument("--no-seed-images", action="store_true")
    run.add_argument(
        "--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT_S,
        help="Wall-clock seconds for one chat-completion attempt. Default 600.")
    run.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)

    smoke = sub.add_parser("smoke", help="1 or 2 instances through the full chain. Requires --live to call the gateway.")
    _add_selection(smoke)
    smoke.add_argument("--run-dir", required=True)
    smoke.add_argument("--live", action="store_true")
    smoke.add_argument("--execute", action="store_true")
    smoke.add_argument("--data-dir", default=None)
    smoke.add_argument("--turn-cap", type=int, default=4)
    smoke.add_argument("--max-tokens", default=None)
    smoke.add_argument("--context-policy", choices=("fail", "trim"), default="fail")
    smoke.add_argument("--context-cap", action="append", default=[])
    smoke.add_argument("--budget-cap", type=float, default=None)
    smoke.add_argument("--no-seed-images", action="store_true")
    smoke.add_argument(
        "--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT_S,
        help="Wall-clock seconds for one chat-completion attempt. Default 600.")
    smoke.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)

    report = sub.add_parser("report", help="Aggregate records.jsonl into summary.json.")
    report.add_argument("--run-dir", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "estimate":
            wants_grid = bool(args.all or args.limit or args.instance_ids or args.split)
            payload = {"experiment": EXPERIMENT, "api_calls": 0}
            if args.from_run_dir:
                rows = _load_jsonl(Path(args.from_run_dir) / "records.jsonl")
                payload["test_300_extrapolation"] = extrapolate_test_300(rows)
            if wants_grid or not args.from_run_dir:
                split, models, limit, instance_ids, allow_all = _selection_from_args(args, default_models="all")
                payload.update(estimate_size(
                    split=split, models=models, limit=limit, instance_ids=instance_ids, allow_all=allow_all))
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "report":
            print(json.dumps(write_report(args.run_dir), ensure_ascii=False, sort_keys=True))
            return 0
        if args.command in {"run", "smoke"} and args.live and not key_configured():
            raise FoxStopped("credential_missing")
        split, models, limit, instance_ids, allow_all = _selection_from_args(
            args, default_models=None if args.command == "run" else "all")
        if args.command in {"run", "smoke"} and not args.models:
            raise FoxStopped("fox_models_required")
        payload = run_schedule(
            args.run_dir, split=split, models=models, limit=limit, instance_ids=instance_ids,
            allow_all=allow_all, turn_cap=args.turn_cap, live=args.live, execute=args.execute,
            data_dir=args.data_dir, context_policy=args.context_policy,
            context_caps=_parse_caps(args.context_cap), max_tokens=_parse_max_tokens(args.max_tokens),
            budget_cap=args.budget_cap, seed_images=not args.no_seed_images,
            timeout_s=args.read_timeout, max_attempts=args.max_attempts, smoke=args.command == "smoke")
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except (FoxStopped, ValueError) as error:
        print(json.dumps({"experiment": EXPERIMENT, "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
