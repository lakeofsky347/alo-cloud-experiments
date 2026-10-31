"""OpenAI-compatible chat transport for the four pool models.

Keys are read from the process environment at the moment of a call and are not
stored on the instance. This module does not contain a price table, and the
network sender stays off unless the caller passes ``permit_network=True``.
Pay-as-you-go endpoints only: MiMo Token Plan and GLM Coding Plan are different
products and are not selected here.

Mainland-China pay-as-you-go URLs and the two owner parameter decisions are
recorded in ``docs/models_and_benchmarks.md``. MiMo sends
``max_completion_tokens`` and not ``max_tokens``. GLM-5.3-Flash sends
``thinking.type=enabled``. Sampling, ``reasoning_effort`` and
``clear_thinking`` are still omitted.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request

from .contracts import MODEL_POOL
from .errors import PilotStopped


# Mainland pay-as-you-go chat completions. Env names stay stable; auth scheme follows
# the vendor page cited in docs/models_and_benchmarks.md. "bearer" is
# Authorization: Bearer. "api-key" is the header used by the MiMo zh-CN curl.
POOL_ENDPOINTS = {
    "deepseek-flash": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/chat/completions", "bearer"),
    "mimo-v2.6-flash": ("MIMO_API_KEY", "https://api.xiaomimimo.com/v1/chat/completions", "api-key"),
    "mimo-v2.6-pro": ("MIMO_API_KEY", "https://api.xiaomimimo.com/v1/chat/completions", "api-key"),
    "glm-5.3-flash": ("ZAI_API_KEY", "https://open.bigmodel.cn/api/paas/v4/chat/completions", "bearer"),
}
MIMO_MODELS = frozenset({"mimo-v2.6-flash", "mimo-v2.6-pro"})


def _key_ok(value):
    return (isinstance(value, str) and value != "" and value.strip() == value
            and "\r" not in value and "\n" not in value and "\0" not in value)


def credential_configured(model):
    """True when the model's env var currently holds a non-empty single-line key."""
    if model not in POOL_ENDPOINTS:
        return False
    env_name, _url, _auth = POOL_ENDPOINTS[model]
    value = os.environ.get(env_name)
    try:
        return _key_ok(value)
    finally:
        value = None


def messages_contain_credential(model, messages):
    """Refuse to transmit a prompt that already embeds the live key."""
    if model not in POOL_ENDPOINTS:
        return False
    env_name, _url, _auth = POOL_ENDPOINTS[model]
    key = os.environ.get(env_name)
    try:
        if not _key_ok(key):
            return False
        try:
            blob = json.dumps(messages, ensure_ascii=False)
        except (TypeError, ValueError):
            return False
        return key in blob
    finally:
        key = None


def request_body(model, messages, max_tokens):
    """Chat-completions JSON. No key and no sampling freeze.

    MiMo's published field is ``max_completion_tokens`` and already includes
    reasoning tokens, so ``max_tokens`` is not sent beside it. GLM-5.3-Flash
    cannot disable thinking; the body sets ``thinking.type`` to ``enabled``.
    """
    # Streaming keeps the socket alive while a reasoning model generates.
    # DeepSeek documents stream_options.include_usage. GLM's streaming guide puts
    # usage on the last chunk without that field. MiMo documents SSE deltas and
    # does not document stream_options, so it is not sent there.
    body = {"model": model, "messages": messages, "stream": True}
    if model == "deepseek-flash":
        body["stream_options"] = {"include_usage": True}
    if model in MIMO_MODELS:
        body["max_completion_tokens"] = max_tokens
    else:
        body["max_tokens"] = max_tokens
    if model == "glm-5.3-flash":
        body["thinking"] = {"type": "enabled"}
    return body


def _auth_headers(auth, key):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if auth == "bearer":
        headers["Authorization"] = "Bearer " + key
    elif auth == "api-key":
        headers["api-key"] = key
    else:
        raise PilotStopped("unknown_auth_scheme")
    return headers


def _reasoning_tokens(usage, completion):
    """Accept the documented detail field and a top-level copy of the same count.

    DeepSeek and MiMo document ``completion_tokens_details.reasoning_tokens``.
    MiMo responses also put the same count on ``usage`` itself. GLM's published
    example has neither field. When both are present they must agree, and the
    count must already sit inside ``completion_tokens``. A missing split stays
    unreported; it is not recorded as a measured zero.
    """
    top = usage["reasoning_tokens"] if "reasoning_tokens" in usage else None
    details = usage.get("completion_tokens_details")
    detail = None
    if details is not None:
        if not isinstance(details, dict):
            raise PilotStopped("provider_outcome_unknown")
        if "reasoning_tokens" in details:
            detail = details["reasoning_tokens"]
    if top is None and detail is None:
        return None
    for value in (top, detail):
        if value is None:
            continue
        if type(value) is not int or value < 0 or value > completion:
            raise PilotStopped("provider_outcome_unknown")
    if top is not None and detail is not None and top != detail:
        raise PilotStopped("provider_outcome_unknown")
    return top if top is not None else detail


def parse_openai_chat_completion(payload, model):
    """Known usage only. Inconsistent or missing token fields are unknown.

    ``content`` may be null or empty when usage is well formed. That is a billed
    completion whose visible text is empty, not an unknown transport result.
    """
    if not isinstance(payload, dict):
        raise PilotStopped("provider_outcome_unknown")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise PilotStopped("provider_outcome_unknown")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise PilotStopped("provider_outcome_unknown")
    content = message.get("content")
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        raise PilotStopped("provider_outcome_unknown")
    finish = choices[0].get("finish_reason")
    if finish is not None and not isinstance(finish, str):
        raise PilotStopped("provider_outcome_unknown")
    returned = payload.get("model")
    if returned is not None and returned != model:
        raise PilotStopped("provider_outcome_unknown")
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise PilotStopped("provider_outcome_unknown")
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if (type(prompt) is not int or type(completion) is not int or type(total) is not int
            or prompt < 0 or completion < 0 or total != prompt + completion):
        raise PilotStopped("provider_outcome_unknown")
    reasoning = _reasoning_tokens(usage, completion)
    cache_hit = _cache_hit_tokens(usage, prompt)
    return {"text": text, "input_tokens": prompt, "output_tokens": completion,
            "reasoning_tokens": reasoning, "total_tokens": total, "cache_hit_tokens": cache_hit,
            "finish_reason": finish}


def parse_vendor_error(payload):
    """Return code and message from an OpenAI-style or GLM-style error body."""
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code", error.get("type"))
        message = error.get("message", error.get("msg"))
    elif "message" in payload or "msg" in payload:
        code = payload.get("code")
        message = payload.get("message", payload.get("msg"))
    else:
        return None
    if code is None and message is None:
        return None
    if code is not None and not isinstance(code, (str, int)):
        return None
    if message is not None and not isinstance(message, str):
        return None
    return {"code": None if code is None else str(code), "message": message or ""}


def _cache_hit_tokens(usage, prompt):
    """Return the cache-hit count, or None when the provider sent no split.

    A missing split is not a measured zero. Conflicting DeepSeek hit/miss fields
    and a cached count above the prompt are unknown.
    """
    hit = None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and "cached_tokens" in details:
        cached = details["cached_tokens"]
        if type(cached) is not int or cached < 0 or cached > prompt:
            raise PilotStopped("provider_outcome_unknown")
        hit = cached
    has_hit = "prompt_cache_hit_tokens" in usage
    has_miss = "prompt_cache_miss_tokens" in usage
    if has_hit or has_miss:
        if not has_hit or not has_miss:
            raise PilotStopped("provider_outcome_unknown")
        explicit_hit = usage["prompt_cache_hit_tokens"]
        explicit_miss = usage["prompt_cache_miss_tokens"]
        if (type(explicit_hit) is not int or type(explicit_miss) is not int
                or explicit_hit < 0 or explicit_miss < 0 or explicit_hit + explicit_miss != prompt):
            raise PilotStopped("provider_outcome_unknown")
        if hit is not None and hit != explicit_hit:
            raise PilotStopped("provider_outcome_unknown")
        hit = explicit_hit
    return hit


def _read_json_body(raw):
    if not raw:
        return None
    try:
        value = json.loads(raw.decode())
    except (UnicodeError, json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def assemble_stream(chunks, model):
    """Fold SSE chunks into one chat completion. Usage comes from the final chunk."""
    parts = []
    finish = None
    usage = None
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, dict) and chunk_usage.get("prompt_tokens") is not None:
            usage = chunk_usage
        choices = chunk.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        choice = choices[0]
        if isinstance(choice.get("finish_reason"), str):
            finish = choice["finish_reason"]
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        piece = delta.get("content")
        if not isinstance(piece, str):
            piece = message.get("content") if isinstance(message.get("content"), str) else None
        if isinstance(piece, str):
            parts.append(piece)
    return {
        "model": model,
        "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": "".join(parts)}}],
        "usage": usage,
    }


def parse_sse(text, model):
    """Parse a recorded event stream. Blank lines and SSE comments are ignored."""
    chunks = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        if not stripped.startswith("data:"):
            continue
        data = stripped[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunks.append(json.loads(data))
        except json.JSONDecodeError:
            return None
    return assemble_stream(chunks, model)


# After the deadline the caller stops waiting. This only covers the socket
# abort unblocking a read; it is not extra time for the provider.
_JOIN_SLACK_S = 0.25
# A read that lands this close to the deadline is the deadline. The abort and
# the socket timeout fire together, and the read can surface as a reset or an
# empty body. That is still our timeout.
DEADLINE_EPSILON_S = 0.05


def _past_deadline(started, clock, limit):
    return clock() - started >= limit - DEADLINE_EPSILON_S


def _socket_of(response):
    if response is None:
        return None
    sock = getattr(response, "_sock", None)
    if sock is not None:
        return sock
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None) if fp is not None else None
    if raw is None:
        return None
    sock = getattr(raw, "_sock", None)
    if sock is None:
        inner = getattr(raw, "raw", None)
        sock = getattr(inner, "_sock", None) if inner is not None else None
    return sock


def _abort_socket(response):
    """Unblock a read. Do not call ``response.close()``; that wait kept holds open."""
    sock = _socket_of(response)
    if sock is None or not hasattr(sock, "settimeout"):
        return
    for action in (
        lambda: sock.settimeout(0.05),
        lambda: sock.shutdown(socket.SHUT_RDWR),
        lambda: sock.close(),
    ):
        try:
            action()
        except OSError:
            pass


def _arm_deadline(response, seconds):
    """Bound the next socket wait by the remaining wall-clock budget."""
    sock = _socket_of(response)
    if sock is not None and hasattr(sock, "settimeout"):
        try:
            sock.settimeout(max(float(seconds), 0.001))
        except OSError:
            return


def _read_sse(response, model, *, deadline_s, started, clock=None, aborted=None):
    """Read SSE until ``[DONE]`` or the wall-clock deadline.

    ``deadline_s`` is the whole request, not the gap between bytes. A keepalive
    or a trickle still counts. The socket timeout is pulled in to whatever time
    remains so a silent poll cannot outlive the deadline either. An empty read
    after our abort is the deadline, not a completion.
    """
    clock = clock or time.monotonic
    chunks = []
    while True:
        if aborted is not None and aborted.get("value"):
            raise TimeoutError("provider_wall_clock")
        remaining = deadline_s - (clock() - started)
        if remaining <= DEADLINE_EPSILON_S:
            if aborted is not None:
                aborted["value"] = True
            _abort_socket(response)
            raise TimeoutError("provider_wall_clock")
        _arm_deadline(response, remaining)
        try:
            raw = response.readline()
        except Exception:
            if aborted is not None and aborted.get("value"):
                raise TimeoutError("provider_wall_clock") from None
            if _past_deadline(started, clock, deadline_s):
                raise TimeoutError("provider_wall_clock") from None
            raise
        if aborted is not None and aborted.get("value"):
            raise TimeoutError("provider_wall_clock")
        if not raw:
            if _past_deadline(started, clock, deadline_s):
                raise TimeoutError("provider_wall_clock")
            break
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunks.append(json.loads(data))
        except json.JSONDecodeError:
            return None
    return assemble_stream(chunks, model)


def _exchange(request, timeout_s, model, started, clock, captured, aborted):
    """One attempt. There is no retry or backoff loop."""
    response = None
    try:
        if _past_deadline(started, clock, timeout_s):
            captured["error"] = TimeoutError("provider_wall_clock")
            return
        remaining = max(timeout_s - (clock() - started), DEADLINE_EPSILON_S)
        response = urllib.request.urlopen(request, timeout=remaining)
        captured["response"] = response
        if aborted.get("value") or _past_deadline(started, clock, timeout_s):
            captured["error"] = TimeoutError("provider_wall_clock")
            return
        status = getattr(response, "status", None)
        if status != 200:
            captured["result"] = (_read_json_body(response.read()), status)
            return
        captured["result"] = (
            _read_sse(response, model, deadline_s=timeout_s, started=started, clock=clock,
                      aborted=aborted),
            status)
    except urllib.error.HTTPError as error:
        status = getattr(error, "code", None)
        try:
            raw = error.read()
        except Exception:
            raw = b""
        captured["result"] = (_read_json_body(raw), status)
    except Exception as error:
        captured["error"] = error
    finally:
        # A deadline abort must not wait on close. The caller shuts the socket
        # down and stops joining; close runs only after a finished body.
        if response is not None and captured.get("error") is None and "result" in captured:
            try:
                response.close()
            except Exception:
                pass


def urllib_send(url, body, headers, *, timeout_s, model, started=None, clock=None):
    """Real HTTP POST. Returns ``(payload, status)``. Offline tests must not call this.

    ``started`` is the moment the ledger hold was taken, on ``time.monotonic``.
    Connect, response headers, a non-200 body and the SSE stream share that
    deadline. A paused machine can make wall-clock age longer than this timer;
    the class of the result still follows monotonic time. Once the deadline is
    reached, or this function aborts the socket, the result is
    ``provider_wall_clock`` even if the read comes back as a reset or an empty
    body. A partial stream after that abort is not a completion.
    """
    clock = clock or time.monotonic
    started = clock() if started is None else started
    if _past_deadline(started, clock, timeout_s):
        raise TimeoutError("provider_wall_clock")
    data = json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data=data, method="POST", headers=headers)
    captured = {}
    aborted = {"value": False}
    worker = threading.Thread(
        target=_exchange,
        args=(request, timeout_s, model, started, clock, captured, aborted),
        daemon=True)
    worker.start()
    remaining = max(timeout_s - (clock() - started), 0)
    worker.join(remaining)
    if worker.is_alive() or _past_deadline(started, clock, timeout_s):
        aborted["value"] = True
        _abort_socket(captured.get("response"))
        worker.join(_JOIN_SLACK_S)
    # The deadline wins over whatever the worker stored after the abort.
    if aborted["value"] or _past_deadline(started, clock, timeout_s):
        raise TimeoutError("provider_wall_clock")
    if "error" in captured:
        raise captured["error"]
    if "result" not in captured:
        raise TimeoutError("provider_wall_clock")
    return captured["result"]


def _exception_cause(error):
    reason = getattr(error, "reason", error) if isinstance(error, urllib.error.URLError) else error
    # Every timeout this transport raises is the deadline we armed: the wall-clock
    # check, or the socket timeout set to the time still left. That is a charged
    # deadline, not an ambiguous provider result.
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "deadline_timeout"
    if isinstance(reason, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
        return "connection_reset"
    return "transport_error"


def interpret_response(payload, model, *, http_status):
    """Classify one HTTP result. Headers are not accepted and are not returned."""
    status = http_status
    if status is None and isinstance(payload, dict):
        status = 200
    response = payload if isinstance(payload, dict) else None
    if status != 200:
        vendor = parse_vendor_error(response) if status is not None else None
        if vendor is not None:
            return {"kind": "vendor_error", "cause": "vendor_error", "http_status": status,
                    "vendor_code": vendor["code"], "vendor_message": vendor["message"],
                    "finish_reason": None, "response": response, "parsed": None}
        return {"kind": "unknown", "cause": "transport_error", "http_status": status,
                "vendor_code": None, "vendor_message": None, "finish_reason": None,
                "response": response, "parsed": None}
    try:
        parsed = parse_openai_chat_completion(payload, model)
    except PilotStopped:
        return {"kind": "unknown", "cause": "unparseable_success", "http_status": status,
                "vendor_code": None, "vendor_message": None, "finish_reason": None,
                "response": response, "parsed": None}
    cause = "empty_completion" if parsed["text"] == "" else None
    return {"kind": "completion", "cause": cause, "http_status": status,
            "vendor_code": None, "vendor_message": None, "finish_reason": parsed["finish_reason"],
            "response": response, "parsed": parsed}


def _unknown_transport(error):
    return {"kind": "unknown", "cause": _exception_cause(error), "http_status": None,
            "vendor_code": None, "vendor_message": None, "finish_reason": None,
            "response": None, "parsed": None}


class OpenAICompatibleTransport:
    """One injected sender, or an explicit network permit. Never both, never neither at call time."""

    def __init__(self, *, sender=None, permit_network=False, timeout_s=60, require_credentials=True):
        if permit_network and sender is not None:
            raise PilotStopped("one_transport_sender")
        if permit_network and not require_credentials:
            raise PilotStopped("live_transport_requires_credentials")
        if type(timeout_s) not in (int, float) or isinstance(timeout_s, bool) or not 0 < timeout_s <= 600:
            raise PilotStopped("invalid_provider_timeout")
        self._sender = sender
        self._permit_network = bool(permit_network)
        self._require_credentials = bool(require_credentials)
        self._timeout_s = float(timeout_s)
        self.last_http_status = None

    def complete_for_model(self, model, messages, *, max_tokens=4096, timeout_s=None,
                           started=None, clock=None):
        if model not in MODEL_POOL or model not in POOL_ENDPOINTS:
            raise PilotStopped("model_outside_pool")
        env_name, url, auth = POOL_ENDPOINTS[model]
        key = os.environ.get(env_name)
        try:
            body = request_body(model, messages, max_tokens)
            encoded = json.dumps(body, ensure_ascii=False)
            if self._require_credentials:
                if not _key_ok(key):
                    raise PilotStopped("provider_credential_missing")
                if key in encoded:
                    raise PilotStopped("provider_credential_in_prompt")
                headers = _auth_headers(auth, key)
            else:
                if _key_ok(key) and key in encoded:
                    raise PilotStopped("provider_credential_in_prompt")
                headers = {"Content-Type": "application/json", "Accept": "application/json"}
            limit = self._timeout_s if timeout_s is None else timeout_s
            try:
                if (started is not None and clock is not None
                        and _past_deadline(started, clock, limit)):
                    raise TimeoutError("provider_wall_clock")
                if self._sender is not None:
                    payload = self._sender(url, body, headers)
                    status = None
                    if isinstance(payload, tuple) and len(payload) == 2 and type(payload[1]) is int:
                        payload, status = payload
                    if (started is not None and clock is not None
                            and _past_deadline(started, clock, limit)):
                        raise TimeoutError("provider_wall_clock")
                elif self._permit_network:
                    try:
                        payload, status = urllib_send(
                            url, body, headers, timeout_s=limit, model=model,
                            started=started, clock=clock)
                    except Exception as error:
                        self.last_http_status = None
                        return _unknown_transport(error)
                else:
                    raise PilotStopped("provider_network_not_permitted")
            except PilotStopped:
                raise
            except Exception as error:
                self.last_http_status = None
                return _unknown_transport(error)
            self.last_http_status = status if status is not None else 200
            return interpret_response(payload, model, http_status=status)
        finally:
            key = None
