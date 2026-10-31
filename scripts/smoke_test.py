#!/usr/bin/env python3
"""Send one short completion to each pool model, or print the plan offline.

``--live`` is required before any socket is opened. The four ceilings are
summed first; the script exits without sending if that sum would exceed
0.5 CNY. Keys and authorization headers are never printed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alo_cloud.contracts import MODEL_POOL, PER_MODEL_LIMIT_MICROCNY  # noqa: E402
from alo_cloud.errors import PilotStopped  # noqa: E402
from alo_cloud.prices import ceiling_for_messages, charge_microcny  # noqa: E402
from alo_cloud.run import open_run_ledger  # noqa: E402
from alo_cloud.services import StudyServices  # noqa: E402


SMOKE_LIMIT_MICROCNY = 500_000
PROMPT = "Reply with the single word ok."
MAX_TOKENS = 64
KEY_ENVS = ("DEEPSEEK_API_KEY", "MIMO_API_KEY", "ZAI_API_KEY")


def smoke_ceilings():
    messages = [{"role": "user", "content": PROMPT}]
    return [(model, ceiling_for_messages(model, messages, MAX_TOKENS)) for model in MODEL_POOL]


def _print_row(model, *, http_status, usage, charged):
    fields = [f"model={model}", f"http_status={http_status}", f"charged_microcny={charged}"]
    if isinstance(usage, dict):
        for key in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
            if key in usage and usage[key] is not None:
                fields.append(f"{key}={usage[key]}")
    print(" ".join(fields))


def main(argv=None):
    parser = argparse.ArgumentParser(description="One tiny request to each pool model.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)
    rows = smoke_ceilings()
    total = sum(amount for _model, amount in rows)
    if total > SMOKE_LIMIT_MICROCNY or any(amount > SMOKE_LIMIT_MICROCNY for _model, amount in rows):
        print("smoke ceiling exceeds 0.5 CNY; not sent")
        return 2
    if not args.live:
        for model, amount in rows:
            _print_row(model, http_status="not_sent", usage=None, charged=amount)
        print(f"total_ceiling_microcny={total}")
        return 0
    missing = [name for name in KEY_ENVS if not os.environ.get(name)]
    if missing:
        print("provider_credential_missing")
        return 2
    run_dir = Path(args.run_dir or os.environ.get("ALO_RUN_DIR") or (Path.home() / "alo-m1-m2-run"))
    marker = run_dir / "smoke.json"
    if marker.is_file():
        print(marker.read_text(encoding="utf-8").strip())
        return 0
    ledger = open_run_ledger(run_dir / "budget.sqlite3", "live")
    messages = [{"role": "user", "content": PROMPT}]
    services = StudyServices(
        run_dir / "smoke-state", ledger, unit="m2",
        per_request_microcny_ceiling=PER_MODEL_LIMIT_MICROCNY,
        charge_microcny=charge_microcny, request_ceiling=ceiling_for_messages,
        permit_network=True, unit_id="smoke", limits={"requests": len(MODEL_POOL)},
    )
    charged = 0
    lines = []
    for model, ceiling in rows:
        if charged + ceiling > SMOKE_LIMIT_MICROCNY:
            print("smoke running total would exceed 0.5 CNY; stopped")
            return 2
        try:
            result = services.complete_for_model(model, messages, max_tokens=MAX_TOKENS)
        except PilotStopped:
            status = services.transport.last_http_status
            _print_row(model, http_status=status if status is not None else "unknown",
                       usage=None, charged=0)
            return 2
        amount = result["cost_microcny"]
        charged += amount
        if charged > SMOKE_LIMIT_MICROCNY:
            print("smoke charge exceeded 0.5 CNY; stopped")
            return 2
        usage = {
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "reasoning_tokens": result["reasoning_tokens"],
            "total_tokens": result["input_tokens"] + result["output_tokens"],
        }
        _print_row(model, http_status=services.transport.last_http_status, usage=usage, charged=amount)
        lines.append({"model": model, "http_status": services.transport.last_http_status,
                      "charged_microcny": amount, "usage": usage})
    marker.write_text(json.dumps({"charged_microcny": charged, "calls": lines}, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"total_charged_microcny={charged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
