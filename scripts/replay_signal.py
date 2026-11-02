#!/usr/bin/env python3
"""Replay the repro-v1 signal over an r2 run. No model API calls.

Records-only output is a proxy: the ledger has no command return codes.
Transcripts and ``--execute`` are labeled separately in the written report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alo_cloud.signal_replay import replay_markdown, replay_run  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay the repro-v1 signal over an existing run.")
    parser.add_argument("--run-dir", required=True, help="Directory containing records.jsonl and optional checkpoints/")
    parser.add_argument("--eval-dir", required=True, help="Directory with m1_units.csv and m2_units.csv")
    parser.add_argument("--output-dir", default=None, help="Where to write the JSON and markdown. Defaults to --run-dir.")
    parser.add_argument("--execute", action="store_true",
                        help="Rerun alo_repro.py and touched public tests in each unit's SWE image. Requires docker.")
    args = parser.parse_args(argv)
    report = replay_run(args.run_dir, args.eval_dir, execute=args.execute)
    destination = Path(args.output_dir) if args.output_dir else Path(args.run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "repro_v1_replay.json"
    markdown_path = destination / "repro_v1_replay.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(replay_markdown(report), encoding="utf-8")
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(markdown_path),
        "g2_units": report["g2_units"],
        "adapters_found": report["adapters_found"],
        "records_proxy": True,
        "transcript_beta_gt_1": report["transcript"]["beta_gt_1"],
        "execute": report["execute"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
