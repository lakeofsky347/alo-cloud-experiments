#!/usr/bin/env python3
"""Estimate the next round from r2's observed per-unit spend. No model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alo_cloud.followup_cost import load_records, project_followup, projection_markdown  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Project follow-up cost from an r2 ledger.")
    parser.add_argument("--records", required=True, help="Path to records.jsonl")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = project_followup(load_records(args.records))
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "followup_cost.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "followup_cost.md").write_text(projection_markdown(report), encoding="utf-8")
    print(json.dumps({
        "total_cny": report["projected_total_cny"],
        "within_cap": report["within_cap"],
        "over_cap_cny": report["over_cap_cny"],
        "scale_matches_r2": report["scale_matches_r2"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
