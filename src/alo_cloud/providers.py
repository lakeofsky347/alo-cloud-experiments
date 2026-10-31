"""OpenAI-compatible chat transport for the four pool models.

Keys are read from the process environment at the moment of a call and are not
stored on the instance. This module does not contain a price table, and the
network sender stays off unless the caller passes ``permit_network=True``.
Pay-as-you-go endpoints only: MiMo Token Plan and GLM Coding Plan are different
products and are not selected here. Sampling and thinking parameters are not
sent, because that contract is still unfrozen; GLM-5.3-Flash in particular
cannot be assumed to share a thinking-off switch with the other three models.
"""
from __future__ import annotations

import json
import os
import urllib.request

from .contracts import MODEL_POOL
from .errors import PilotStopped


# Official pay-as-you-go OpenAI-compatible chat completions, rechecked 2026-10-31.
# DeepSeek: https://api.deepseek.com/chat/completions with DEEPSEEK_API_KEY.
# MiMo: https://api.xiaomimimo.com/v1/chat/completions with MIMO_API_KEY.
# Z.AI: https://api.z.ai/api/paas/v4/chat/completions with ZAI_API_KEY.
POOL_ENDPOINTS = {
    "deepseek-flash": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/chat/completions"),
    "mimo-v2.6-flash": ("MIMO_API_KEY", "https://api.xiaomimimo.com/v1/chat/completions"),
    "mimo-v2.6-pro": ("MIMO_API_KEY", "https://api.xiaomimimo.com/v1/chat/completions"),
    "glm-5.3-flash": ("ZAI_API_KEY", "https://api.z.ai/api/paas/v4/chat/completions"),
}


def _key_ok(value):
    return (isinstance(value, str) and value != "" and value.strip() == value
            and "\r" not in value and "\n" not in value and "\0" not in value)


def credential_configured(model):
    """True when the model's env var currently holds a non-empty single-line key."""
    if model not in POOL_ENDPOINTS:
        return False
    env_name, _url = POOL_ENDPOINTS[model]
    value = os.environ.get(env_name)
    try:
        return _key_ok(value)
    finally:
        value = None


def messages_contain_credential(model, messages):
    """Refuse to transmit a prompt that already embeds the live key."""
    if model not in POOL_ENDPOINTS:
        return False
    env_name, _url = POOL_ENDPOINTS[model]
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
    """Chat-completions JSON. No key, no thinking switch, no temperature freeze."""
    return {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}


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
    reasoning = 0
    details = usage.get("completion_tokens_details")
    if details is not None:
        if not isinstance(details, dict) or "reasoning_tokens" not in details:
            raise PilotStopped("provider_outcome_unknown")
        reasoning = details["reasoning_tokens"]
        if type(reasoning) is not int or reasoning < 0:
            raise PilotStopped("provider_outcome_unknown")
    return {"text": message["content"], "input_tokens": prompt, "output_tokens": completion,
            "reasoning_tokens": reasoning, "total_tokens": total}


def urllib_send(url, body, key, *, timeout_s):
    """Real HTTP POST. Callers must not use this from offline tests."""
    data = json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
    return json.loads(raw.decode())


class OpenAICompatibleTransport:
    """One injected sender, or an explicit network permit. Never both, never neither at call time."""

    def __init__(self, *, sender=None, permit_network=False, timeout_s=60):
        if permit_network and sender is not None:
            raise PilotStopped("one_transport_sender")
        if type(timeout_s) not in (int, float) or isinstance(timeout_s, bool) or not 0 < timeout_s <= 60:
            raise PilotStopped("invalid_provider_timeout")
        self._sender = sender
        self._permit_network = bool(permit_network)
        self._timeout_s = float(timeout_s)

    def complete_for_model(self, model, messages, *, max_tokens=4096):
        if model not in MODEL_POOL or model not in POOL_ENDPOINTS:
            raise PilotStopped("model_outside_pool")
        env_name, url = POOL_ENDPOINTS[model]
        key = os.environ.get(env_name)
        try:
            if not _key_ok(key):
                raise PilotStopped("provider_credential_missing")
            body = request_body(model, messages, max_tokens)
            encoded = json.dumps(body, ensure_ascii=False)
            if key in encoded:
                raise PilotStopped("provider_credential_in_prompt")
            headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
                       "Accept": "application/json"}
            try:
                if self._sender is not None:
                    payload = self._sender(url, body, headers)
                elif self._permit_network:
                    payload = urllib_send(url, body, key, timeout_s=self._timeout_s)
                else:
                    raise PilotStopped("provider_network_not_permitted")
            except PilotStopped:
                raise
            except Exception:
                raise PilotStopped("provider_outcome_unknown") from None
            return parse_openai_chat_completion(payload, model)
        finally:
            key = None
