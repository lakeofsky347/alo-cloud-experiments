"""Metered study services. Constructing one does not permit a paid run.

The caller supplies an existing ledger, a per-request CNY ceiling and a charge
function. This module has no price table and does not select Strong, freeze the
protocol, or walk the M1/M2 schedule. Unit caps below are the enforcement
numbers from the protocol text; naming a unit does not admit tasks or open
provider credentials.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path

from .budget import MICROCNY_PER_CNY, BudgetLedger, BudgetStopped
from .contracts import MODEL_POOL, PER_MODEL_LIMIT_MICROCNY, STUDY_TOTAL_LIMIT_MICROCNY
from .diagnostics import write_diagnostic
from .errors import PilotStopped, ProtocolError
from .shardlock import halt_record
from .progress import append_provider_outcome, publish_timeout_counts
from .freeze import MIMO_MAX_COMPLETION_TOKENS, PROVIDER_TIMEOUT_S
from .providers import (
    DEADLINE_EPSILON_S, MIMO_MODELS, OpenAICompatibleTransport, credential_configured,
    messages_contain_credential, request_body,
)
from .sandbox import DockerSandbox, SandboxUnknown, Workspace, python_syntax_evidence


# protocol.md 资源、未知与停止. Enforcement table, not a collection freeze.
PROTOCOL_UNIT_LIMITS = {
    "strong_b0": {"requests": 32, "tokens": 900_000, "tools": 120, "seconds": 1200,
                  "command_seconds": 60, "max_output_tokens": MIMO_MAX_COMPLETION_TOKENS},
    "m1_b0": {"requests": 32, "tokens": 900_000, "tools": 120, "seconds": 1200,
              "command_seconds": 60, "max_output_tokens": MIMO_MAX_COMPLETION_TOKENS},
    "m1_b1": {"requests": 16, "tokens": 900_000, "tools": 120, "seconds": 1200,
              "command_seconds": 60, "max_output_tokens": MIMO_MAX_COMPLETION_TOKENS},
    "m2": {"requests": 32, "tokens": 900_000, "tools": 120, "seconds": 1200,
           "command_seconds": 60, "max_output_tokens": MIMO_MAX_COMPLETION_TOKENS},
}


def _read_timeouts(value):
    """Per-model read budgets. None keeps the protocol table."""
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise PilotStopped("invalid_read_timeouts")
    checked = {}
    for model, seconds in value.items():
        if model not in MODEL_POOL:
            raise PilotStopped("invalid_read_timeouts")
        if type(seconds) not in (int, float) or isinstance(seconds, bool) or not 0 < seconds <= 600:
            raise PilotStopped("invalid_read_timeouts")
        checked[model] = float(seconds)
    return checked


def _merge_limits(unit, limits, overrides=None):
    merged = dict(PROTOCOL_UNIT_LIMITS[unit])
    if overrides:
        if not isinstance(overrides, dict):
            raise PilotStopped("unit_limit_outside_protocol")
        for key, value in overrides.items():
            if key not in merged or type(value) is not int or value < 1:
                raise PilotStopped("unit_limit_outside_protocol")
            if value > merged[key]:
                merged[key] = value
    if limits is None:
        return merged
    if not isinstance(limits, dict):
        raise PilotStopped("unit_limit_outside_protocol")
    for key, value in limits.items():
        if key not in merged or type(value) is not int or not 1 <= value <= merged[key]:
            raise PilotStopped("unit_limit_outside_protocol")
        merged[key] = value
    return merged


def _map_python_argv(argv):
    """Use python3 inside the sandbox. ``python`` is not on every image PATH."""
    if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
        return argv
    if argv[0] in {"python", "py"}:
        return ["python3", *argv[1:]]
    return argv


def _digest(body):
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _valid_messages(messages):
    if not isinstance(messages, list) or not messages:
        return False
    for message in messages:
        if (not isinstance(message, dict) or set(message) - {"role", "content", "name"}
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)):
            return False
    return True


class StudyServices:
    """Shared public tools and one metered model gateway for a single unit."""

    available_models = MODEL_POOL

    def __init__(self, state_dir, ledger, *, unit, per_request_microcny_ceiling, charge_microcny,
                 sender=None, permit_network=False, sandbox=None, clock=None, public_quality=None,
                 unit_id=None, request_ceiling=None, require_credentials=True, limits=None,
                 restart_count=0, protocol_epoch=0, shard_id="", infra_generation=0,
                 track_generation=0, budget_notice=None, limit_overrides=None,
                 reasoning_budget=None, read_timeouts=None, connect_timeout_s=None):
        if unit not in PROTOCOL_UNIT_LIMITS:
            raise PilotStopped("unknown_protocol_unit")
        if not isinstance(ledger, BudgetLedger):
            raise PilotStopped("budget_ledger_required")
        if type(per_request_microcny_ceiling) is not int or not 1 <= per_request_microcny_ceiling <= PER_MODEL_LIMIT_MICROCNY:
            raise PilotStopped("per_request_microcny_ceiling_required")
        if not callable(charge_microcny):
            raise PilotStopped("charge_function_required")
        if request_ceiling is not None and not callable(request_ceiling):
            raise PilotStopped("request_ceiling_must_be_callable")
        if unit_id is not None and (not isinstance(unit_id, str) or not unit_id):
            raise PilotStopped("unit_id_required")
        if type(restart_count) is not int or restart_count not in {0, 1}:
            raise PilotStopped("invalid_restart_count")
        if type(infra_generation) is not int or not 0 <= infra_generation <= 99:
            raise PilotStopped("invalid_infra_generation")
        if type(track_generation) is not int or not 0 <= track_generation <= 99:
            raise PilotStopped("invalid_track_generation")
        if type(protocol_epoch) is not int or protocol_epoch < 0:
            raise PilotStopped("invalid_protocol_epoch")
        if not isinstance(shard_id, str) or (shard_id and "/" not in shard_id):
            raise PilotStopped("invalid_shard")
        self.limits = _merge_limits(unit, limits, overrides=limit_overrides)
        self.shard_id = shard_id
        self.restart_count = restart_count
        self.infra_generation = infra_generation
        self.track_generation = track_generation
        self.budget_notice = budget_notice if isinstance(budget_notice, str) and budget_notice else None
        self.budget_notice_sent = False
        if reasoning_budget is not None and (type(reasoning_budget) is not int or not 1 <= reasoning_budget <= 8192):
            raise PilotStopped("invalid_reasoning_budget")
        self.reasoning_budget = reasoning_budget
        self.read_timeouts = _read_timeouts(read_timeouts)
        if connect_timeout_s is not None and (
                type(connect_timeout_s) not in (int, float) or isinstance(connect_timeout_s, bool)
                or not 0 < connect_timeout_s <= 60):
            raise PilotStopped("invalid_connect_timeout")
        self.connect_timeout_s = None if connect_timeout_s is None else float(connect_timeout_s)
        self.transport_counts = {}
        self.pro_call_cap = None
        self._requests_by_model = {}
        self.pro_share_yields = 0
        self.length_stops = 0
        self.edit_log = []
        self.g3_request_split = {"manager_selection": 0, "role": 0, "dispatches": 0}
        self.current_role = None
        self.current_node = None
        self.seal_patch = None
        self.tool_warnings = []
        self.protocol_epoch = protocol_epoch
        self.unit = unit
        self.unit_id = unit_id
        self.ledger = ledger
        self.per_request_microcny_ceiling = per_request_microcny_ceiling
        self._charge = charge_microcny
        self._request_ceiling = request_ceiling
        self._require_credentials = bool(require_credentials)
        self.transport = OpenAICompatibleTransport(
            sender=sender, permit_network=permit_network, timeout_s=max(PROVIDER_TIMEOUT_S.values()),
            require_credentials=require_credentials, connect_timeout_s=self.connect_timeout_s)
        if sender is None and not permit_network:
            raise PilotStopped("provider_sender_required")
        self.sandbox = sandbox
        self._clock = clock or time.monotonic
        self._started = self._clock()
        self._quality = public_quality
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.workspace = Workspace(self.state_dir / "workspace")
        self.tools = 0
        self.events = []
        self._seq = 0
        self._observed = {"model_requests": 0, "input_tokens": 0, "output_tokens": 0,
                          "reasoning_tokens": 0, "microcny": 0}
        self._require_matching_ledger()

    def __repr__(self):
        return f"StudyServices(unit={self.unit!r})"

    def _require_matching_ledger(self):
        state = self.ledger.snapshot()
        if set(state["limits_microcny"]) != set(MODEL_POOL):
            raise PilotStopped("ledger_model_pool_mismatch")
        if any(value > PER_MODEL_LIMIT_MICROCNY for value in state["limits_microcny"].values()):
            raise PilotStopped("model_cny_cap_above_protocol")
        if self.unit_id:
            if any(value != PER_MODEL_LIMIT_MICROCNY for value in state["limits_microcny"].values()):
                raise PilotStopped("study_model_cap_must_be_50_cny")
            if state.get("total_limit_microcny") != STUDY_TOTAL_LIMIT_MICROCNY:
                raise PilotStopped("study_total_cap_required")
            if state["max_requests"] < self.limits["requests"]:
                raise PilotStopped("ledger_request_ceiling_below_unit")
        elif state["max_requests"] != self.limits["requests"]:
            raise PilotStopped("ledger_request_ceiling_does_not_match_unit")
        if state.get("max_tokens") != self.limits["tokens"]:
            raise PilotStopped("ledger_token_ceiling_does_not_match_unit")
        if state.get("max_seconds_ms") != self.limits["seconds"] * 1000:
            raise PilotStopped("ledger_seconds_ceiling_does_not_match_unit")
        if state.get("max_tools") != self.limits["tools"]:
            raise PilotStopped("ledger_tool_ceiling_does_not_match_unit")

    def _next_id(self, kind):
        self._seq += 1
        label = self.unit_id or self.unit
        epoch = f"e{self.protocol_epoch}-" if self.protocol_epoch else ""
        restart = f"-r{self.restart_count}" if self.restart_count else ""
        infra = f"-infra{self.infra_generation}" if self.infra_generation else ""
        track = f"-pbv{self.track_generation}" if self.track_generation else ""
        return f"{kind}-{epoch}{label}{restart}{infra}{track}-{self._seq}"

    def _resource_kwargs(self):
        fields = {"shard_id": self.shard_id} if self.shard_id else {}
        if not self.unit_id:
            return fields
        fields.update(unit_id=self.unit_id, unit_request_cap=self.limits["requests"],
                      protocol_epoch=self.protocol_epoch, restart_count=self.restart_count,
                      infra_generation=self.infra_generation, track_generation=self.track_generation)
        return fields

    def _wall_remaining(self):
        return self.limits["seconds"] - (self._clock() - self._started)

    def _ensure_wall_clock(self):
        if self._wall_remaining() <= 0:
            raise PilotStopped("unit_seconds_exhausted")

    def _ensure_not_blocked(self):
        record = halt_record(self.ledger.path.parent)
        if record and record["hard"]:
            raise PilotStopped(record["reason"])
        if self.ledger.blocks_dispatch(self.shard_id):
            raise PilotStopped("unresolved_request_blocks_subsequent_dispatch")

    def _stop(self, error):
        raise PilotStopped(str(error)) from None

    def _with_wall(self, remaining):
        remaining = dict(remaining)
        remaining["seconds"] = min(float(remaining["seconds"]), max(0.0, self._wall_remaining()))
        return remaining

    def budget_remaining(self, model=None):
        """Four policy dimensions. Without a model, CNY is the pool minimum."""
        kwargs = self._resource_kwargs()
        try:
            if model is None:
                rows = [self.ledger.remaining_resources(name, **kwargs) for name in MODEL_POOL]
                remaining = dict(rows[0])
                remaining["cny"] = min(row["cny"] for row in rows)
            else:
                remaining = self.ledger.remaining_resources(model, **kwargs)
        except BudgetStopped as error:
            self._stop(error)
        if (model == "mimo-v2.6-pro" and type(self.pro_call_cap) is int
                and self._requests_by_model.get(model, 0) >= self.pro_call_cap):
            remaining = dict(remaining)
            remaining["requests"] = 0
        return self._with_wall(remaining)

    def _worst_case_holds(self):
        state = self.ledger.snapshot()
        if "max_tokens" not in state or "max_seconds_ms" not in state:
            raise PilotStopped("resource_dimensions_unconfigured")
        if self.unit_id:
            usage = self.ledger.unit_usage(
                self.unit_id, protocol_epoch=self.protocol_epoch, restart_count=self.restart_count,
                infra_generation=self.infra_generation, track_generation=self.track_generation)
            tokens = state["max_tokens"] - usage["tokens"]
            accounted_ms = state["max_seconds_ms"] - usage["seconds_ms"]
        else:
            tokens = state["max_tokens"] - state["used_tokens"]
            accounted_ms = state["max_seconds_ms"] - state["used_seconds_ms"]
        wall_ms = int(self._wall_remaining() * 1000)
        seconds_ms = min(accounted_ms, max(wall_ms, 0))
        if tokens < 1 or seconds_ms < 1:
            raise PilotStopped("unit_resource_exhausted")
        return tokens, seconds_ms

    def _mark_request_unknown(self, request_id):
        try:
            self.ledger.mark_unknown(request_id)
        except BudgetStopped as error:
            self._stop(error)

    def _read_deadline(self, model):
        table = self.read_timeouts if isinstance(self.read_timeouts, dict) else PROVIDER_TIMEOUT_S
        seconds = table.get(model, PROVIDER_TIMEOUT_S[model])
        return float(seconds)

    def _sent_max_tokens(self, model, max_tokens):
        """Leave room for reasoning and a visible answer on every pool model.

        A caller asking for at least the historical 4096 cap is raised to the
        reasoning cap. Smaller explicit caps, used by smoke tests, stay as
        requested. The reservation uses this same number.
        """
        del model
        if max_tokens >= 4096:
            return min(self.limits["max_output_tokens"], MIMO_MAX_COMPLETION_TOKENS)
        return max_tokens

    def _diagnose(self, request_id, model, outcome, *, latency_ms, settled, cause=None,
                  charged_timeout=False):
        try:
            write_diagnostic(
                self.ledger, request_id, model=model, unit_id=self.unit_id,
                cause=cause or outcome.get("cause") or "transport_error",
                http_status=outcome.get("http_status"),
                vendor_code=outcome.get("vendor_code"),
                vendor_message=outcome.get("vendor_message"),
                response=outcome.get("response"),
                finish_reason=outcome.get("finish_reason"),
                latency_ms=latency_ms, settled=settled, charged_timeout=charged_timeout,
                stream_progress=outcome.get("stream_progress"),
            )
        except (OSError, ValueError):
            return

    def _note_outcome(self, request_id, model, outcome, charge_microcny):
        try:
            append_provider_outcome(
                self.ledger, epoch=self.protocol_epoch, model=model, request_id=request_id,
                outcome=outcome, charge_microcny=charge_microcny)
        except OSError:
            return

    def _usage_charge(self, model, parsed):
        try:
            charge = self._charge(model, {
                "input_tokens": parsed["input_tokens"], "output_tokens": parsed["output_tokens"],
                "reasoning_tokens": parsed["reasoning_tokens"], "total_tokens": parsed["total_tokens"],
                "cache_hit_tokens": parsed.get("cache_hit_tokens"),
            })
        except Exception:
            return None
        if type(charge) is not int or charge < 0:
            return None
        return charge

    def _remember(self, parsed, charge):
        self._observed["model_requests"] += 1
        self._observed["input_tokens"] += parsed["input_tokens"]
        self._observed["output_tokens"] += parsed["output_tokens"]
        if parsed["reasoning_tokens"] is not None:
            self._observed["reasoning_tokens"] += parsed["reasoning_tokens"]
        self._observed["microcny"] += charge

    def _pro_share_blocked(self, model):
        return (model == "mimo-v2.6-pro" and type(self.pro_call_cap) is int
                and self._requests_by_model.get(model, 0) >= self.pro_call_cap)

    def _unbilled_pro_share(self):
        """Refuse another Pro call before a reservation. The refusal is not a request."""
        self.pro_share_yields += 1
        if self.pro_share_yields > 4:
            raise PilotStopped("pro_share_cap")
        return {
            "text": "pro_share_cap: choose another model. This call was not billed.",
            "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": None,
            "cost_microcny": 0, "currency": "CNY", "pro_share_cap": True,
            "length_stop": False, "vendor_error": False,
        }

    def _inject_budget_notice(self, messages):
        """Tell a ProgramBench agent to finish compile.sh when 3 calls remain.

        The notice is appended once, as a system message, and a diagnostic is
        written. It does not add a model call.
        """
        if self.budget_notice is None or self.budget_notice_sent or not isinstance(messages, list):
            return
        remaining = self.limits["requests"] - self._observed["model_requests"]
        if remaining > 3 or remaining < 1:
            return
        messages.append({"role": "system", "content": self.budget_notice})
        self.budget_notice_sent = True
        label = self.unit_id or self.unit or "unit"
        try:
            write_diagnostic(
                self.ledger, f"notice-{label}-{self._observed['model_requests']}",
                cause="budget_notice", model="", unit_id=self.unit_id,
                http_status=None, vendor_code="requests_remaining",
                vendor_message=self.budget_notice,
                response={"requests_remaining": remaining},
                finish_reason=None, latency_ms=0, settled=True)
        except (OSError, ValueError):
            return

    def complete_for_model(self, model, messages, *, max_tokens=4096):
        if model not in self.available_models:
            raise PilotStopped("model_outside_pool")
        if self._pro_share_blocked(model):
            return self._unbilled_pro_share()
        if type(max_tokens) is not int or not 1 <= max_tokens <= self.limits["max_output_tokens"]:
            raise PilotStopped("output_token_parameter_outside_unit_limit")
        sent_max_tokens = self._sent_max_tokens(model, max_tokens)
        if not _valid_messages(messages):
            raise PilotStopped("invalid_model_messages")
        self._inject_budget_notice(messages)
        self._ensure_not_blocked()
        self._ensure_wall_clock()
        if self._require_credentials and not credential_configured(model):
            raise PilotStopped("provider_credential_missing")
        if messages_contain_credential(model, messages):
            raise PilotStopped("provider_credential_in_prompt")
        body = request_body(model, messages, sent_max_tokens, reasoning_budget=self.reasoning_budget)
        if self._request_ceiling is None:
            amount = self.per_request_microcny_ceiling
        else:
            try:
                amount = self._request_ceiling(model, messages, sent_max_tokens)
            except Exception:
                raise PilotStopped("per_request_microcny_ceiling_required") from None
            if type(amount) is not int or not 1 <= amount <= self.per_request_microcny_ceiling:
                raise PilotStopped("per_request_microcny_ceiling_required")
        token_hold, second_hold = self._worst_case_holds()
        deadline_s = self._read_deadline(model)
        presend_released = 0
        while True:
            request_id = self._next_id("req")
            self.last_request_id = request_id
            try:
                self.ledger.reserve(request_id, model, amount,
                                    payload_sha256=_digest(body), tokens=token_hold, seconds_ms=second_hold,
                                    **self._resource_kwargs())
            except BudgetStopped as error:
                self._stop(error)
            # The hold is open now. A provably unsent call releases it and retries once.
            started = self._clock()
            try:
                outcome = self.transport.complete_for_model(
                    model, messages, max_tokens=sent_max_tokens,
                    timeout_s=deadline_s, started=started, clock=self._clock,
                    reasoning_budget=self.reasoning_budget)
                elapsed_s = self._clock() - started
                elapsed_ms = max(0, int(round(elapsed_s * 1000)))
                if outcome.get("kind") == "not_reached" and outcome.get("request_sent") is False:
                    self.ledger.release_not_sent(request_id)
                    self._diagnose(request_id, model, outcome, latency_ms=elapsed_ms, settled=True)
                    self._note_outcome(request_id, model, "not_sent", 0)
                    if presend_released < 1:
                        presend_released += 1
                        continue
                    return {"text": "", "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": None,
                            "latency_ms": elapsed_ms, "cost_microcny": 0, "currency": "CNY",
                            "finish_reason": None, "length_stop": False, "vendor_error": False,
                            "provider_not_reached": True}
                # Monotonic time, same clock as the transport. A return at the deadline
                # is a charged timeout even when the read looks like a reset, an empty
                # body, or a partial completion. A failure before any response bytes
                # stays not_reached: unsent is released above, and a request that may
                # have been sent stays an unknown worst-case hold. Connect time is
                # outside the read budget when the two timeouts are split.
                read_limit = deadline_s
                if self.connect_timeout_s is not None:
                    read_limit = deadline_s + self.connect_timeout_s
                if (outcome.get("kind") != "not_reached"
                        and elapsed_s >= read_limit - DEADLINE_EPSILON_S):
                    progress = outcome.get("stream_progress")
                    outcome = {"kind": "unknown", "cause": "deadline_timeout", "http_status": None,
                               "vendor_code": None, "vendor_message": None, "finish_reason": None,
                               "response": None, "parsed": None}
                    if isinstance(progress, dict):
                        outcome["stream_progress"] = progress
                kind = outcome.get("kind")
                if kind == "completion":
                    parsed = outcome["parsed"]
                    charge = self._usage_charge(model, parsed)
                    if charge is None:
                        self._diagnose(request_id, model, outcome, latency_ms=elapsed_ms,
                                       settled=False, cause="charge_rejected")
                        self._mark_request_unknown(request_id)
                        raise PilotStopped("provider_outcome_unknown")
                    settled = self.ledger.settle_known(
                        request_id, charge, tokens=parsed["total_tokens"], seconds_ms=elapsed_ms)
                    if outcome.get("cause") == "empty_completion":
                        self._diagnose(request_id, model, outcome, latency_ms=elapsed_ms, settled=True)
                    self._remember(parsed, charge)
                    self._requests_by_model[model] = self._requests_by_model.get(model, 0) + 1
                    self._note_outcome(request_id, model, "completion", charge)
                    self.event("model_settled", {"model": model, "request_id": request_id,
                                                  "input_tokens": parsed["input_tokens"],
                                                  "output_tokens": parsed["output_tokens"],
                                                  "reasoning_tokens": parsed["reasoning_tokens"],
                                                  "charge_microcny": charge,
                                                  "finish_reason": parsed.get("finish_reason")})
                    if settled.get("unit_resource_exhausted"):
                        raise PilotStopped("unit_resource_exhausted")
                    return {"text": parsed["text"], "input_tokens": parsed["input_tokens"],
                            "output_tokens": parsed["output_tokens"],
                            "reasoning_tokens": parsed["reasoning_tokens"],
                            "latency_ms": elapsed_ms, "cost_microcny": charge, "currency": "CNY",
                            "finish_reason": parsed.get("finish_reason"),
                            "length_stop": parsed["text"] == "" and parsed.get("finish_reason") == "length",
                            "vendor_error": False}
                if kind == "vendor_error":
                    settled = self.ledger.settle_known(request_id, 0, tokens=0, seconds_ms=elapsed_ms)
                    self._diagnose(request_id, model, outcome, latency_ms=elapsed_ms, settled=True)
                    self._note_outcome(request_id, model, "vendor_error", 0)
                    self._observed["model_requests"] += 1
                    self.event("model_settled", {"model": model, "request_id": request_id,
                                                  "input_tokens": 0, "output_tokens": 0,
                                                  "reasoning_tokens": None, "charge_microcny": 0,
                                                  "vendor_error": True})
                    if settled.get("unit_resource_exhausted"):
                        raise PilotStopped("unit_resource_exhausted")
                    return {"text": "", "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": None,
                            "latency_ms": elapsed_ms, "cost_microcny": 0, "currency": "CNY",
                            "finish_reason": None, "length_stop": False, "vendor_error": True}
                self._mark_request_unknown(request_id)
                cause = outcome.get("cause") or "transport_error"
                try:
                    self.ledger.resolve_unknown(
                        request_id, None, reason=cause, worst_case=True)
                except BudgetStopped:
                    self._diagnose(request_id, model, outcome, latency_ms=elapsed_ms, settled=False)
                    raise PilotStopped("provider_outcome_unknown") from None
                charged_timeout = cause == "deadline_timeout"
                charged_unparseable = cause == "unparseable_success"
                self._diagnose(request_id, model, outcome, latency_ms=elapsed_ms, settled=True,
                               charged_timeout=charged_timeout)
                if charged_timeout:
                    outcome_name = "deadline_timeout"
                elif charged_unparseable:
                    outcome_name = "unparseable_charged"
                else:
                    outcome_name = "unknown"
                self._note_outcome(request_id, model, outcome_name, amount)
                if charged_timeout:
                    try:
                        publish_timeout_counts(
                            Path(self.ledger.path).parent, self.ledger,
                            protocol_epoch=self.protocol_epoch)
                    except OSError:
                        pass
                    raise PilotStopped("provider_deadline_charged")
                if charged_unparseable:
                    try:
                        from .progress import publish_unparseable_counts
                        publish_unparseable_counts(
                            Path(self.ledger.path).parent, self.ledger,
                            protocol_epoch=self.protocol_epoch)
                    except OSError:
                        pass
                    raise PilotStopped("provider_unparseable_charged")
                raise PilotStopped("provider_outcome_charged")
            except PilotStopped as error:
                if str(error) in {
                    "provider_outcome_unknown", "provider_outcome_charged", "provider_deadline_charged",
                    "provider_unparseable_charged", "unit_resource_exhausted",
                }:
                    raise
                self._diagnose(request_id, model, {"cause": "transport_error"}, latency_ms=0, settled=False)
                self._mark_request_unknown(request_id)
                raise PilotStopped("provider_outcome_unknown") from None
            except BudgetStopped:
                self._diagnose(request_id, model, {"cause": "charge_rejected"}, latency_ms=0,
                               settled=False, cause="charge_rejected")
                raise PilotStopped("provider_outcome_unknown") from None
            except Exception:
                self._diagnose(request_id, model, {"cause": "transport_error"}, latency_ms=0, settled=False)
                self._mark_request_unknown(request_id)
                raise PilotStopped("provider_outcome_unknown") from None

    def resources(self):
        return {"model_requests": self._observed["model_requests"],
                "observed_input_tokens": self._observed["input_tokens"],
                "observed_output_tokens": self._observed["output_tokens"],
                "observed_reasoning_tokens": self._observed["reasoning_tokens"],
                "estimated_cost_cny": self._observed["microcny"] / MICROCNY_PER_CNY}

    def public_quality(self):
        if self._quality is None:
            return None
        value = self._quality()
        return value

    def local_criterion(self, *, role_id, model_id, output, context):
        del role_id, model_id, output, context
        return python_syntax_evidence(self.workspace, self.workspace.take_attributed())

    def event(self, event, payload):
        self.events.append((event, payload))

    def clamp_timeout(self, timeout_s):
        """Clamp a tool timeout to the unit command limit and return warnings."""
        limit = self.limits["command_seconds"]
        if isinstance(timeout_s, bool) or type(timeout_s) not in (int, float) or not timeout_s > 0:
            raise ProtocolError("command_timeout_outside_unit_limit")
        warnings = []
        if timeout_s > limit:
            warnings.append(f"timeout_s clamped from {timeout_s} to {limit}")
            self.tool_warnings.append(warnings[-1])
            timeout_s = limit
        if timeout_s > self._wall_remaining():
            raise PilotStopped("command_timeout_exceeds_remaining_unit_seconds")
        return timeout_s, warnings

    def _validate_timeout(self, timeout_s):
        timeout_s, _warnings = self.clamp_timeout(timeout_s)
        return timeout_s

    def _close_tool(self, tool_id, started, error, *, sandbox_unknown):
        """Settle the time hold and store a redacted traceback. Return whether it settled.

        Tool calls cost no CNY. An elapsed time past the reservation is capped
        so the row can close. Only a ledger that cannot settle the row stays unknown.
        """
        elapsed_ms = max(0, int(round((self._clock() - started) * 1000)))
        settled = False
        try:
            self.ledger.settle_tool(tool_id, seconds_ms=elapsed_ms, cap_to_reservation=True)
            settled = True
        except BudgetStopped:
            try:
                self.ledger.mark_tool_unknown(tool_id)
            except BudgetStopped:
                pass
        self._diagnose(
            tool_id, "tool",
            {"cause": "tool_exception", "vendor_code": type(error).__name__,
             "vendor_message": traceback.format_exc(),
             "response": {"exception_type": type(error).__name__, "sandbox_unknown": sandbox_unknown},
             "http_status": None, "finish_reason": None},
            latency_ms=elapsed_ms, settled=settled, cause="tool_exception")
        return settled

    def _use_tool(self, action, tool="tool"):
        self._ensure_not_blocked()
        self._ensure_wall_clock()
        token_hold, second_hold = self._worst_case_holds()
        del token_hold
        before_count = len(self.workspace.attributed)
        self._active_tool = tool
        tool_id = self._next_id("tool")
        try:
            self.ledger.reserve_tool(
                tool_id, seconds_ms=second_hold, unit_id=self.unit_id or "",
                protocol_epoch=self.protocol_epoch, restart_count=self.restart_count,
                shard_id=self.shard_id, infra_generation=self.infra_generation,
                track_generation=self.track_generation)
        except BudgetStopped as error:
            self._stop(error)
        started = self._clock()
        try:
            result = action()
        except ProtocolError as error:
            if not self._close_tool(tool_id, started, error, sandbox_unknown=False):
                raise PilotStopped("tool_outcome_unknown") from None
            self.tools += 1
            raise
        except SandboxUnknown as error:
            if not self._close_tool(tool_id, started, error, sandbox_unknown=True):
                raise PilotStopped("tool_outcome_unknown") from None
            raise PilotStopped("tool_sandbox_unknown") from None
        except Exception as error:
            if not self._close_tool(tool_id, started, error, sandbox_unknown=False):
                raise PilotStopped("tool_outcome_unknown") from None
            self.tools += 1
            raise ProtocolError(f"tool_failed:{type(error).__name__}") from None
        elapsed_ms = max(0, int(round((self._clock() - started) * 1000)))
        try:
            self.ledger.settle_tool(tool_id, seconds_ms=elapsed_ms)
        except BudgetStopped:
            raise PilotStopped("tool_outcome_unknown") from None
        self.tools += 1
        self._record_edit(getattr(self, "_active_tool", "tool"), before_count)
        return result

    def _record_edit(self, tool, before_count):
        paths = list(self.workspace.attributed[before_count:])
        patch = ""
        root = self.workspace.root
        if (root / ".git").is_dir():
            try:
                from .collect import workspace_patch
                base = getattr(self, "image_head", None) or ""
                if isinstance(base, str) and base:
                    patch = workspace_patch(root, base)
            except Exception:
                patch = ""
        digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        self.edit_log.append({
            "tool": tool,
            "diff_sha256": digest,
            "paths": paths,
            "role": self.current_role,
            "node": self.current_node,
        })
        sealer = self.seal_patch
        if callable(sealer):
            try:
                sealer(patch)
            except Exception:
                self.tool_warnings.append("patch_seal_failed")

    def diff_nonempty(self):
        if any(row.get("paths") for row in self.edit_log):
            return True
        root = self.workspace.root
        if not (root / ".git").is_dir():
            return False
        try:
            from .collect import workspace_patch
            base = getattr(self, "image_head", None) or ""
            if not isinstance(base, str) or not base:
                return False
            return bool(workspace_patch(root, base).strip())
        except Exception:
            return False

    def list_files(self, path=None):
        if path is None:
            return self._use_tool(self.workspace.list_files, tool="list_files")
        return self._use_tool(lambda: self.workspace.list_files(path), tool="list_files")

    def read_file(self, path, start_line=None, end_line=None):
        return self._use_tool(
            lambda: self.workspace.read_file(path, start_line=start_line, end_line=end_line),
            tool="read_file")

    def write_file(self, path, content, append=False):
        return self._use_tool(lambda: self.workspace.write_file(path, content, append=append), tool="write_file")

    def edit_file(self, path, old_str, new_str):
        return self._use_tool(lambda: self.workspace.edit_file(path, old_str, new_str), tool="edit_file")

    def search(self, text):
        return self._use_tool(lambda: self.workspace.search(text), tool="search")

    def command(self, argv, timeout_s=30):
        timeout_s = self._validate_timeout(timeout_s)
        argv = _map_python_argv(argv)
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
            raise ProtocolError("sandbox_argv_rejected")
        if self.sandbox is None:
            raise PilotStopped("sandbox_not_configured")
        if isinstance(self.sandbox, DockerSandbox) and not self.sandbox.execution_ready:
            raise PilotStopped("sandbox_not_configured")

        def run():
            return self.sandbox.command(argv, timeout_s)

        return self._use_tool(run, tool="command")

    def execute_code(self, language, code, timeout_s=30):
        if language in {"python", "py", "python3"}:
            argv = ["python3", "-c", code if isinstance(code, str) else ""]
        elif language in {"sh", "shell", "bash"}:
            argv = ["sh", "-c", code if isinstance(code, str) else ""]
        else:
            raise ProtocolError("unsupported_execute_language")
        if not isinstance(code, str):
            raise ProtocolError("execute_code_requires_text")
        return self.command(argv, timeout_s)
