#!/usr/bin/env python3
"""Replay synthetic tool-call fixtures. No provider call and no ledger write."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alo_cloud.transport_fixtures import replay_fixtures  # noqa: E402


def main():
    report = replay_fixtures()
    print(json.dumps({"parsed": report["parsed"], "n": report["n"], "rate": report["rate"]}, sort_keys=True))
    return 0 if report["rate"] >= 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
