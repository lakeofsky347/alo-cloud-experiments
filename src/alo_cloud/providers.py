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

import http.client
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
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


REASONING_BUDGET_MAX = 8192


def request_body(model, messages, max_tokens, *, reasoning_budget=None):
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
    # A numeric reasoning budget is sent only where the published body already
    # has a thinking object we can extend without inventing a MiMo field.
    # MiMo's max_completion_tokens already counts reasoning plus visible text.
    if (model == "deepseek-flash" and type(reasoning_budget) is int
            and 1 <= reasoning_budget <= REASONING_BUDGET_MAX):
        body["thinking"] = {"type": "enabled", "budget_tokens": reasoning_budget}
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
    count must already sit inside ``completion_tokens``. A count that exceeds
    completion by at most 64, or by at most one percent of completion, is
    clamped to completion. A larger gap stays unknown. A missing split stays
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

    def _as_int(value):
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return value

    top = _as_int(top)
    detail = _as_int(detail)

    def _clamp(value):
        if value is None:
            return None
        if type(value) is not int or value < 0:
            raise PilotStopped("provider_outcome_unknown")
        if value <= completion:
            return value
        over = value - completion
        within = over <= 64 or (completion > 0 and over * 100 <= completion)
        if not within:
            raise PilotStopped("provider_outcome_unknown")
        return completion

    top = _clamp(top)
    detail = _clamp(detail)
    if top is not None and detail is not None and top != detail:
        raise PilotStopped("provider_outcome_unknown")
    return top if top is not None else detail


def _same_model(returned, model):
    """Accept the pool id and a vendor prefix or different capitalisation."""
    if returned == model:
        return True
    if not isinstance(returned, str):
        return False
    folded = returned.strip().casefold().replace("_", "-")
    wanted = model.casefold()
    return folded == wanted or folded.endswith("/" + wanted) or folded.endswith(":" + wanted)


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
        text_field = choices[0].get("text")
        if isinstance(text_field, str):
            message = {"content": text_field}
        else:
            raise PilotStopped("provider_outcome_unknown")
    content = message.get("content")
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    elif isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                piece = part.get("text", part.get("content"))
                if isinstance(piece, str):
                    pieces.append(piece)
        text = "".join(pieces)
    else:
        raise PilotStopped("provider_outcome_unknown")
    finish = choices[0].get("finish_reason")
    if finish is not None and not isinstance(finish, str):
        raise PilotStopped("provider_outcome_unknown")
    returned = payload.get("model")
    if returned is not None and not _same_model(returned, model):
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


def _saw_reasoning(chunk):
    """True when one SSE event carries non-empty reasoning text."""
    if not isinstance(chunk, dict):
        return False
    choices = chunk.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return False
    choice = choices[0]
    for key in ("delta", "message"):
        body = choice.get(key)
        if not isinstance(body, dict):
            continue
        text = body.get("reasoning_content")
        if isinstance(text, str) and text:
            return True
    return False


def _stream_progress(bytes_read, data_chunks, last_chunk_at, saw_reasoning, clock):
    age = None if last_chunk_at is None else clock() - last_chunk_at
    return {
        "bytes_read": bytes_read,
        "chunks": data_chunks,
        "last_chunk_age_s": age,
        "saw_reasoning": saw_reasoning,
    }


def _wall_clock(progress):
    """Deadline timeout that still reports how far the stream got."""
    error = TimeoutError("provider_wall_clock")
    error.stream_progress = dict(progress)
    return error


def _read_sse(response, model, *, deadline_s, started, clock=None, aborted=None):
    """Read SSE until ``[DONE]`` or the wall-clock deadline.

    ``deadline_s`` is the whole request, not the gap between bytes. A keepalive
    or a trickle still counts. The socket timeout is pulled in to whatever time
    remains so a silent poll cannot outlive the deadline either. An empty read
    after our abort is the deadline, not a completion. A wall-clock timeout
    carries bytes read, data chunks, the age of the last chunk, and whether
    reasoning text was seen. That does not change the timeout class.
    """
    clock = clock or time.monotonic
    chunks = []
    bytes_read = 0
    data_chunks = 0
    last_chunk_at = None
    saw_reasoning = False

    def progress():
        return _stream_progress(bytes_read, data_chunks, last_chunk_at, saw_reasoning, clock)

    while True:
        if aborted is not None and aborted.get("value"):
            raise _wall_clock(progress())
        remaining = deadline_s - (clock() - started)
        if remaining <= DEADLINE_EPSILON_S:
            if aborted is not None:
                aborted["value"] = True
            _abort_socket(response)
            raise _wall_clock(progress())
        _arm_deadline(response, remaining)
        try:
            raw = response.readline()
        except Exception:
            if aborted is not None and aborted.get("value"):
                raise _wall_clock(progress()) from None
            if _past_deadline(started, clock, deadline_s):
                raise _wall_clock(progress()) from None
            raise
        if aborted is not None and aborted.get("value"):
            raise _wall_clock(progress())
        if not raw:
            if _past_deadline(started, clock, deadline_s):
                raise _wall_clock(progress())
            break
        if isinstance(raw, (bytes, bytearray)):
            bytes_read += len(raw)
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return None
        data_chunks += 1
        last_chunk_at = clock()
        if _saw_reasoning(parsed):
            saw_reasoning = True
        chunks.append(parsed)
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


class ProviderNotReached(Exception):
    """The provider produced no response bytes.

    ``request_sent`` is false only when this process never called ``request``.
    A failure after that call may have left the process, so the hold stays unknown.
    """

    def __init__(self, request_sent):
        super().__init__("provider_not_reached")
        self.request_sent = bool(request_sent)


class _ClientResponse:
    """Adapt ``http.client.HTTPResponse`` to the SSE reader."""

    def __init__(self, response, sock):
        self._sock = sock
        self._response = response

    def readline(self):
        return self._response.readline()

    def close(self):
        try:
            self._response.close()
        except Exception:
            return


def _not_reached(request_sent):
    return {"kind": "not_reached", "cause": "provider_not_reached",
            "request_sent": bool(request_sent), "http_status": None,
            "vendor_code": None, "vendor_message": None, "finish_reason": None,
            "response": None, "parsed": None}


def staged_https_post(url, body, headers, *, connect_timeout_s, read_timeout_s, model,
                       started=None, clock=None, connector=None):
    """POST one streaming completion with separate connect and read timeouts.

    Connect uses ``connect_timeout_s``. After the socket exists, the read
    budget is ``read_timeout_s`` measured from connect success, so a reasoning
    model can keep the stream open for the full read window. ``Connection:
    keep-alive`` and ``Accept: text/event-stream`` are set on this path.
    A failure before ``request`` is raised as ``ProviderNotReached(False)``.
    A failure after ``request`` and before any response bytes is
    ``ProviderNotReached(True)``. Bytes already on the stream use the existing
    read deadline.
    """
    clock = clock or time.monotonic
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise PilotStopped("provider_url_not_https")
    port = parts.port or 443
    path = parts.path or "/"
    if parts.query:
        path = path + "?" + parts.query
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    factory = connector or http.client.HTTPSConnection
    conn = factory(parts.hostname, port, timeout=connect_timeout_s)
    response = None
    try:
        try:
            conn.connect()
        except ProviderNotReached:
            raise
        except Exception as error:
            raise ProviderNotReached(False) from error
        sock = getattr(conn, "sock", None)
        if sock is not None and hasattr(sock, "settimeout"):
            try:
                sock.settimeout(float(read_timeout_s))
            except OSError:
                pass
        outbound = {
            "Content-Type": headers.get("Content-Type", "application/json"),
            "Accept": "text/event-stream",
            "Connection": "keep-alive",
        }
        for name, value in headers.items():
            if name.lower() in {"authorization", "api-key"}:
                outbound[name] = value
        read_started = clock()
        try:
            conn.request("POST", path, body=data, headers=outbound)
        except Exception as error:
            raise ProviderNotReached(True) from error
        try:
            response = conn.getresponse()
        except Exception as error:
            raise ProviderNotReached(True) from error
        status = getattr(response, "status", None)
        if status != 200:
            try:
                raw = response.read()
            except Exception as error:
                if _past_deadline(read_started, clock, read_timeout_s):
                    raise _wall_clock(_stream_progress(0, 0, None, False, clock)) from error
                raise
            return _read_json_body(raw), status
        adapter = _ClientResponse(response, sock)
        parsed = _read_sse(
            adapter, model, deadline_s=read_timeout_s, started=read_started, clock=clock)
        return parsed, status
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


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


def _attached_stream_progress(error):
    progress = getattr(error, "stream_progress", None)
    if not isinstance(progress, dict):
        reason = getattr(error, "reason", None)
        progress = getattr(reason, "stream_progress", None)
    if not isinstance(progress, dict):
        return None
    return dict(progress)


def _unknown_transport(error):
    outcome = {"kind": "unknown", "cause": _exception_cause(error), "http_status": None,
               "vendor_code": None, "vendor_message": None, "finish_reason": None,
               "response": None, "parsed": None}
    progress = _attached_stream_progress(error)
    if progress is not None:
        outcome["stream_progress"] = progress
    return outcome


class OpenAICompatibleTransport:
    """One injected sender, or an explicit network permit. Never both, never neither at call time."""

    def __init__(self, *, sender=None, permit_network=False, timeout_s=60, require_credentials=True,
                 connect_timeout_s=None, connector=None):
        if permit_network and sender is not None:
            raise PilotStopped("one_transport_sender")
        if permit_network and not require_credentials:
            raise PilotStopped("live_transport_requires_credentials")
        if type(timeout_s) not in (int, float) or isinstance(timeout_s, bool) or not 0 < timeout_s <= 600:
            raise PilotStopped("invalid_provider_timeout")
        if connect_timeout_s is not None and (
                type(connect_timeout_s) not in (int, float) or isinstance(connect_timeout_s, bool)
                or not 0 < connect_timeout_s <= 60):
            raise PilotStopped("invalid_connect_timeout")
        self._sender = sender
        self._permit_network = bool(permit_network)
        self._require_credentials = bool(require_credentials)
        self._timeout_s = float(timeout_s)
        self._connect_timeout_s = None if connect_timeout_s is None else float(connect_timeout_s)
        self._connector = connector
        self.last_http_status = None

    def complete_for_model(self, model, messages, *, max_tokens=4096, timeout_s=None,
                           started=None, clock=None, reasoning_budget=None):
        if model not in MODEL_POOL or model not in POOL_ENDPOINTS:
            raise PilotStopped("model_outside_pool")
        env_name, url, auth = POOL_ENDPOINTS[model]
        key = os.environ.get(env_name)
        try:
            body = request_body(model, messages, max_tokens, reasoning_budget=reasoning_budget)
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
                elif self._permit_network and self._connect_timeout_s is not None:
                    try:
                        payload, status = staged_https_post(
                            url, body, headers, connect_timeout_s=self._connect_timeout_s,
                            read_timeout_s=limit, model=model, started=started, clock=clock,
                            connector=self._connector)
                    except ProviderNotReached as error:
                        self.last_http_status = None
                        return _not_reached(error.request_sent)
                    except Exception as error:
                        self.last_http_status = None
                        return _unknown_transport(error)
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
            except ProviderNotReached as error:
                self.last_http_status = None
                return _not_reached(error.request_sent)
            except PilotStopped:
                raise
            except Exception as error:
                self.last_http_status = None
                return _unknown_transport(error)
            self.last_http_status = status if status is not None else 200
            return interpret_response(payload, model, http_status=status)
        finally:
            key = None
