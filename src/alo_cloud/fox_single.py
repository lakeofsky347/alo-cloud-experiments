"""Single-model SWE-bench Lite baseline on the Foxrouter pool.

Label ``fox-single-v1``. Output stays in the caller's run directory and is
not an r1, r2, or r3 ledger. No per-token price is assumed. The API key is
read from ``FOXROUTER_API_KEY`` only when ``--live`` sends, and it is not
written into the run metadata.

Offline tests inject ``sender``. The default live sender is the only path
that opens a socket.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .adapters import adapter_for
from .collect import prediction_row, workspace_patch
from .errors import PilotStopped, ProtocolError
from .evaluate import SWEBENCH_PACKAGE, SWEBENCH_REVISION, execute_swebench, harness_cwd, swebench_command
from .fox_pool import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_READ_TIMEOUT_S,
    DEFAULT_TURN_CAP,
    EXPERIMENT,
    FAILURE_TAXONOMY,
    FOX_CHAT_URL,
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
from .providers import parse_openai_chat_completion, parse_sse
from .repro import REPRO_CRITERION_ID
from .run import load_public_payload, stage_for
from .sandbox import DockerSandbox, SWE_TESTBED_PATH
from .workspace_prep import resolve_image_head, seed_image_tree


_BANNED_DIR_NAMES = frozenset({
    "r1", "r2", "r3", "alo-r1-run", "alo-r2-run", "alo-r3-run",
})
_STOP_FAILURES = frozenset({"provider_error", "timeout", "context_limit", "budget_cap"})


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


def key_configured():
    """True when the env var holds a non-empty single-line key. The value is not returned."""
    value = os.environ.get(FOX_KEY_ENV)
    try:
        return (isinstance(value, str) and value != "" and value.strip() == value
                and "\r" not in value and "\n" not in value and "\0" not in value)
    finally:
        value = None


def _read_key():
    value = os.environ.get(FOX_KEY_ENV)
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


def urllib_fox_send(url, body, headers, timeout_s):
    """One HTTPS POST. Tests must inject a sender instead of calling this."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return getattr(response, "status", 200), response.read()
    except urllib.error.HTTPError as error:
        try:
            raw = error.read()
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
                      max_attempts=DEFAULT_MAX_ATTEMPTS, url=FOX_CHAT_URL):
    """POST one completion. Transient failures are retried and every failure is returned."""
    if type(max_attempts) is not int or not 1 <= max_attempts <= 8:
        raise FoxStopped("invalid_retry_attempts")
    body = request_body(model, messages, max_tokens)
    headers = auth_headers(key)
    retries = []
    last = None
    for attempt in range(1, max_attempts + 1):
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
                return last, retries, body
            kind = "http"
        retries.append({
            "attempt": attempt,
            "status": status,
            "kind": kind,
            "cloudflare_error": cloudflare_error_code(payload, raw_text),
            "timestamp": _stamp(clock()),
        })
        if attempt < max_attempts:
            sleep(min(30.0, float(2 ** (attempt - 1))))
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


def interpret_call(model, sent_messages, max_tokens, result, retries, *, estimated):
    """One ledger row. Raw usage and gateway cost fields are copied, not priced."""
    payload = result.get("payload") if isinstance(result, dict) else None
    usage = payload.get("usage") if isinstance(payload, dict) and isinstance(payload.get("usage"), dict) else None
    raw_usage = json_safe(usage) if isinstance(usage, dict) else None
    text, reasoning_present = _message_text(payload)
    parsed = None
    if isinstance(payload, dict):
        try:
            parsed = parse_openai_chat_completion(payload, model.upstream)
        except PilotStopped:
            parsed = None
    prompt = completion = total = reasoning = None
    finish = None
    if parsed is not None:
        text = parsed["text"]
        prompt = parsed["input_tokens"]
        completion = parsed["output_tokens"]
        total = parsed["total_tokens"]
        reasoning = parsed["reasoning_tokens"]
        finish = parsed["finish_reason"]
    elif isinstance(usage, dict):
        if type(usage.get("prompt_tokens")) is int:
            prompt = usage["prompt_tokens"]
        if type(usage.get("completion_tokens")) is int:
            completion = usage["completion_tokens"]
        if type(usage.get("total_tokens")) is int:
            total = usage["total_tokens"]
        reasoning = reasoning_tokens_of(usage, completion)
    quirk = None
    if completion == 512:
        quirk = "completion_tokens_exactly_512"
    status = result.get("status") if isinstance(result, dict) else None
    truncated = suspected_truncation(estimated, prompt) if type(prompt) is int else False
    parsed_ok = parsed is not None or bool(text) or reasoning_present
    ok = status == 200 and parsed_ok
    row = {
        "experiment": EXPERIMENT,
        "internal_model": model.internal,
        "upstream_model": model.upstream,
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
        "text": text if isinstance(text, str) else "",
    }
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

    def budget_remaining(self, model=None):
        del model
        left = max(0, int(self.limits["requests"]) - self._requests)
        elapsed = self._clock() - self._started
        seconds = max(0.0, float(self.limits.get("seconds", 1200)) - elapsed)
        # No CNY field. A missing list price must not look like a zero balance.
        return {"requests": left, "tokens": int(self.limits.get("tokens", 900_000)), "seconds": seconds}

    def _budget_blocked(self):
        if self.budget_cap is None:
            return False
        return self.spent_cost > float(self.budget_cap)

    def complete_for_model(self, model, messages, *, max_tokens=4096):
        if model != self.model_spec.internal:
            raise FoxStopped("model_outside_fox_pool")
        if self._budget_blocked():
            raise FoxStopped("budget_cap")
        if self._requests >= self.limits["requests"]:
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
                "suspected_truncation": False,
                **context,
            })
            raise FoxStopped("context_limit")
        self._requests += 1
        result, retries = self.transport(self.model_spec, prepared, sent_max)
        row = interpret_call(
            self.model_spec, prepared, sent_max, result, retries,
            estimated=context["estimated_prompt_tokens"])
        row.update(context)
        amount = numeric_amount(row.get("cost"))
        if amount is not None:
            self.spent_cost += amount
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
            return 120, ["timeout clamped to 120"]
        return timeout_s, []

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


def _find_resolved(root, instance_id):
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
    cwd = harness_cwd(run_dir)
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


def _live_secrets():
    value = os.environ.get(FOX_KEY_ENV)
    try:
        if isinstance(value, str) and value:
            return [value]
        return []
    finally:
        value = None


def _transport_for(*, live, sender, timeout_s, sleep, clock, max_attempts):
    if sender is None and not live:
        raise FoxStopped("fox_sender_required")
    if live and sender is None and not key_configured():
        raise FoxStopped("credential_missing")

    def transport(model, messages, max_tokens):
        key = _read_key() if live and sender is None else os.environ.get(FOX_KEY_ENV)
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
                    max_attempts=max_attempts)
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


def _write_manifest(run_dir, payload):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("experiment") != EXPERIMENT:
            raise FoxStopped("fox_run_dir_experiment_mismatch")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    done = {row.get("unit_id") for row in _load_jsonl(run_dir / "records.jsonl")}
    spent = 0.0
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
                    _append_jsonl(run_dir / "failures.jsonl", {
                        "experiment": EXPERIMENT,
                        "unit_id": record["unit_id"],
                        "kind": call["failure"],
                        "http_status": call.get("http_status"),
                        "suspected_truncation": bool(call.get("suspected_truncation")),
                    })
            if record.get("failure"):
                _append_jsonl(run_dir / "failures.jsonl", {
                    "experiment": EXPERIMENT,
                    "unit_id": record["unit_id"],
                    "kind": record["failure"],
                    "report_model": record["report_model"],
                    "split": record["split"],
                })
            if record.get("spent_cost") is not None:
                spent = float(record["spent_cost"])
            written.append(record)
    return {"experiment": EXPERIMENT, "wrote": len(written), "units": manifest["units"], "run_dir": str(run_dir)}


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
        bucket["resolved_rate"] = (bucket["resolved"] / instances) if instances else None
        bucket["cost_units"] = "as_returned"
        rows.append(bucket)
    del calls
    return {
        "experiment": EXPERIMENT,
        "schema": "fox-single-v1-summary",
        "prices": "none",
        "by_model_split": rows,
    }


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
    parser.add_argument("--split", choices=("dev", "test", "both"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--instance-ids", nargs="+", default=None)
    parser.add_argument("--all", action="store_true", help="Use every instance in the split. Required for a full run.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="fox/... names, deepseek-flash, or all")


def _selection_from_args(args, *, default_models):
    if not args.split and not args.instance_ids:
        raise FoxStopped("selection_required")
    split = args.split or "both"
    return split, args.models or default_models, args.limit, args.instance_ids, bool(args.all)


def main(argv=None):
    parser = argparse.ArgumentParser(description="fox-single-v1 SWE-bench Lite baseline. Does not call an API unless --live.")
    sub = parser.add_subparsers(dest="command", required=True)

    estimate = sub.add_parser("estimate", help="Print models × instances. No API call.")
    _add_selection(estimate)

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
    run.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT_S)
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
    smoke.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT_S)
    smoke.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)

    report = sub.add_parser("report", help="Aggregate records.jsonl into summary.json.")
    report.add_argument("--run-dir", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "estimate":
            split, models, limit, instance_ids, allow_all = _selection_from_args(args, default_models="all")
            payload = estimate_size(
                split=split, models=models, limit=limit, instance_ids=instance_ids, allow_all=allow_all)
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
