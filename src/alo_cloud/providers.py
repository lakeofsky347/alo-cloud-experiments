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
    body = {"model": model, "messages": messages, "stream": False}
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


def parse_openai_chat_completion(payload, model):
    """Known usage only. Inconsistent or missing token fields are unknown."""
    if not isinstance(payload, dict):
        raise PilotStopped("provider_outcome_unknown")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise PilotStopped("provider_outcome_unknown")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
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
    # Official usage defines total as prompt + completion. reasoning_tokens, when
    # present, is a completion-detail and must not be added a second time.
    # A missing split stays unreported; it is not recorded as a measured zero.
    if "reasoning_tokens" in usage:
        raise PilotStopped("provider_outcome_unknown")
    reasoning = None
    details = usage.get("completion_tokens_details")
    if details is not None:
        if not isinstance(details, dict):
            raise PilotStopped("provider_outcome_unknown")
        if "reasoning_tokens" in details:
            reasoning = details["reasoning_tokens"]
            if type(reasoning) is not int or reasoning < 0 or reasoning > completion:
                raise PilotStopped("provider_outcome_unknown")
    cache_hit = _cache_hit_tokens(usage, prompt)
    return {"text": message["content"], "input_tokens": prompt, "output_tokens": completion,
            "reasoning_tokens": reasoning, "total_tokens": total, "cache_hit_tokens": cache_hit}


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


def urllib_send(url, body, headers, *, timeout_s):
    """Real HTTP POST. Returns ``(payload, status)``. Offline tests must not call this."""
    data = json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        status = getattr(response, "status", None)
        raw = response.read()
    return json.loads(raw.decode()), status


class OpenAICompatibleTransport:
    """One injected sender, or an explicit network permit. Never both, never neither at call time."""

    def __init__(self, *, sender=None, permit_network=False, timeout_s=60, require_credentials=True):
        if permit_network and sender is not None:
            raise PilotStopped("one_transport_sender")
        if permit_network and not require_credentials:
            raise PilotStopped("live_transport_requires_credentials")
        if type(timeout_s) not in (int, float) or isinstance(timeout_s, bool) or not 0 < timeout_s <= 60:
            raise PilotStopped("invalid_provider_timeout")
        self._sender = sender
        self._permit_network = bool(permit_network)
        self._require_credentials = bool(require_credentials)
        self._timeout_s = float(timeout_s)
        self.last_http_status = None

    def complete_for_model(self, model, messages, *, max_tokens=4096):
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
            try:
                if self._sender is not None:
                    payload = self._sender(url, body, headers)
                elif self._permit_network:
                    try:
                        payload, status = urllib_send(url, body, headers, timeout_s=self._timeout_s)
                    except urllib.error.HTTPError as error:
                        self.last_http_status = getattr(error, "code", None)
                        raise PilotStopped("provider_outcome_unknown") from None
                    self.last_http_status = status
                else:
                    raise PilotStopped("provider_network_not_permitted")
            except PilotStopped:
                raise
            except Exception:
                raise PilotStopped("provider_outcome_unknown") from None
            return parse_openai_chat_completion(payload, model)
        finally:
            key = None
