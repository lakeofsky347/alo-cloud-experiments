"""Pure AutoGen bridge preserved from the research runtime; no HTTP client or host executor.

The installed AutoGen 0.2.40 package owns native speaker selection, broadcast
and conversation state. Every model call and public tool remains delegated to
explicit caller-supplied services.
"""
from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

from .errors import BaselineUnavailable, PilotStopped

DONE = "STAGE_DONE"

SHARED_INSTRUCTIONS = (
    "Use the shared workspace and public feedback only. Preserve prior files and state. "
    "Python or shell code blocks are executed by Workspace in the common sandbox. "
    "Do not assume a self-report is acceptance. Emit STAGE_DONE alone only when the public stage work is ready."
)

ROLE_INSTRUCTIONS = {
    "Planner": "Plan cross-module changes and inspect progress.",
    "Engineer": "Implement and validate changes, using executable code blocks.",
    "Critic": "Review interfaces, migrations and regressions against public requirements.",
}

def normalize_native_messages(messages):
    """Losslessly encode native speaker names into the shared text contract.

    The native conversation remains unchanged. The durable gateway receives the
    exact normalized text. Unknown fields, function/tool calls and multimodal
    content fail closed instead of silently dropping native information.
    """
    if not isinstance(messages, list) or not messages:
        raise BaselineUnavailable("unsupported_native_message_contract")
    result = []
    for message in messages:
        if (not isinstance(message, dict) or not {"role", "content"} <= set(message)
                or set(message) - {"role", "content", "name"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)):
            raise BaselineUnavailable("unsupported_native_message_contract")
        content = message["content"]
        if "name" in message:
            if not isinstance(message["name"], str) or not message["name"]:
                raise BaselineUnavailable("unsupported_native_message_contract")
            content = "[native_speaker:" + json.dumps(message["name"], ensure_ascii=False) + "]\n" + content
        result.append({"role": message["role"], "content": content})
    return result

def _client_class(services):
    """Bind the exact same durable, budgeted service to every native client."""
    class ALOServiceModelClient:
        def __init__(self, config, **kwargs):
            self.model = config["model"]

        def create(self, params):
            if params.get("n", 1) != 1:
                raise BaselineUnavailable("native bridge requires n=1; freeze a new sampling contract for n>1")
            response = services.complete(normalize_native_messages(copy.deepcopy(params["messages"])),
                                         max_tokens=params.get("max_tokens", 2048))
            required = ("text", "input_tokens", "output_tokens")
            if (any(response.get(key) is None for key in required)
                    or not isinstance(response["text"], str)
                    or any(type(response[key]) is not int or response[key] < 0
                           for key in ("input_tokens", "output_tokens"))):
                raise BaselineUnavailable("missing_gateway_usage_or_response")
            currency = response.get("currency", getattr(getattr(services, "gateway", None), "currency", "USD"))
            if currency == "USD":
                cost = response.get("cost_usd")
            elif currency == "CNY":
                cost = response.get("cost_microcny")
                if type(cost) is not int:
                    raise BaselineUnavailable("missing_gateway_usage_or_response")
            else:
                raise BaselineUnavailable("unsupported_gateway_cost_currency")
            if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
                raise BaselineUnavailable("missing_gateway_usage_or_response")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response["text"]))],
                model=self.model,
                cost=cost,
                cost_currency=currency,
                usage={"prompt_tokens": response["input_tokens"], "completion_tokens": response["output_tokens"],
                       "total_tokens": response["input_tokens"] + response["output_tokens"],
                       "cost": cost, "model": self.model,
                       "cost_currency": currency,
                       "native_cost_counter_unit": {"CNY": "microcny"}.get(currency, currency),
                       "authoritative_accounting": "shared_gateway"},
            )

        @staticmethod
        def message_retrieval(response):
            return [choice.message.content for choice in response.choices]

        @staticmethod
        def cost(response):
            return response.cost

        @staticmethod
        def get_usage(response):
            return dict(response.usage)

    return ALOServiceModelClient

class ServiceCodeExecutor:
    """Native code-executor protocol, routed to the common isolated workspace.

    There is no subprocess, eval or host executor fallback in this bridge.
    """
    def __init__(self, services, extractor, timeout_seconds=20):
        self.services = services
        self._extractor = extractor
        self.timeout_seconds = timeout_seconds

    def __deepcopy__(self, memo):
        # Native config copies must retain one ledger/backend, not clone it.
        return self

    @property
    def code_extractor(self):
        return self._extractor

    def execute_code_blocks(self, code_blocks):
        output = []
        exit_code = 0
        for block in code_blocks:
            language = block.language.lower()
            if language in ("python", "py", "python3"):
                argv = ["python", "-c", block.code]
            elif language in ("sh", "shell", "bash"):
                argv = ["sh", "-c", block.code]
            else:
                return SimpleNamespace(exit_code=2, output="unsupported_code_language:" + language)
            shared_execute = getattr(self.services, "execute_code", None)
            if callable(shared_execute):
                result = shared_execute(argv[0], block.code, timeout_s=self.timeout_seconds)
            else:
                result = self.services.command(argv, timeout_s=self.timeout_seconds)
            if result.get("timed_out"):
                exit_code = result["returncode"] or 124
            elif type(result.get("returncode")) is not int:
                raise BaselineUnavailable("tool_result_missing_exit_code")
            else:
                exit_code = result["returncode"]
            output.append(result.get("stdout", "") + result.get("stderr", ""))
            if result.get("timed_out"):
                output.append("public_tool_timeout")
                exit_code = exit_code or 124
            if exit_code or result.get("timed_out"):
                break
        return SimpleNamespace(exit_code=exit_code, output="\n".join(output))

    def restart(self):
        # Conversation reset never resets task files, database or resource ledger.
        return None

def _terminated(message):
    return str(message.get("content", "")).strip() == DONE
class BoundModel:
    def __init__(self, services, model):
        if model not in services.available_models:
            raise PilotStopped("selected_model_not_in_common_pool")
        self.services, self.model = services, model

    def __getattr__(self, name):
        return getattr(self.services, name)

    def complete(self, messages, *, max_tokens=4096):
        return self.services.complete_for_model(self.model, messages, max_tokens=max_tokens)

def native_client(services):
    class HeterogeneousPilotModelClient:
        def __init__(self, config, **kwargs):
            self.delegate = _client_class(BoundModel(services, config["model"]))(config, **kwargs)

        def create(self, params):
            return self.delegate.create(params)

        def message_retrieval(self, response):
            return self.delegate.message_retrieval(response)

        def cost(self, response):
            return self.delegate.cost(response)

        def get_usage(self, response):
            return self.delegate.get_usage(response)

    return HeterogeneousPilotModelClient
