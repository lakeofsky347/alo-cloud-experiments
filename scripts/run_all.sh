#!/bin/sh
# Paid M1 then M2 on a Linux host with Docker. Re-running resumes.
# The ledger is $ALO_RUN_DIR/budget.sqlite3. Completed units in records.jsonl
# are skipped. An unresolved reservation stops the schedule until a person
# inspects it; this script does not clear or retry that hold.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
RUN_DIR=${ALO_RUN_DIR:-"$HOME/alo-m1-m2-run"}
DATA_DIR=${ALO_DATA_DIR:-"$RUN_DIR/public_tasks"}
PY=${PYTHON:-python3}
cd "$ROOT"

"$PY" -m alo_cloud.freeze --check

# One short completion per model. Refuses if the four ceilings exceed 0.5 CNY.
# Requires DEEPSEEK_API_KEY, MIMO_API_KEY and ZAI_API_KEY in the environment.
"$PY" scripts/smoke_test.py --live --run-dir "$RUN_DIR"

# ALO_SANDBOX_IMAGE is only the fallback. SWE and ProgramBench units use the
# official per-task images. Each unit is seeded from that image (SWE /testbed,
# ProgramBench /workspace) and the live tree is deleted after the patch or
# submission tarball is written. Set ALO_SEED_IMAGES=1 to seed during a dry-run.
if [ ! -d "$DATA_DIR/SWE-bench_Lite" ] || [ ! -d "$DATA_DIR/ProgramBench" ]; then
  echo "Stage public task JSON at $DATA_DIR/<track>/<task_id>.json"
  echo "SWE files need repo (owner/name) and base_commit. Do not include patch, test_patch, FAIL_TO_PASS or PASS_TO_PASS."
  echo "SWE-bench Lite revision 6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2."
  echo "ProgramBench source commit 895f65916bc5ba34a915d36da321fde0301125ad."
  exit 2
fi

"$PY" -m alo_cloud.run --study m1 --live --run-dir "$RUN_DIR" --data-dir "$DATA_DIR"
"$PY" -m alo_cloud.run --study m2 --live --run-dir "$RUN_DIR" --data-dir "$DATA_DIR"
"$PY" -m alo_cloud.collect --run-dir "$RUN_DIR"

# Scoring is a separate process. It does not read the provider keys.
# Revisions are passed on the command and set again inside the evaluator.
export PROGRAMBENCH_HF_REVISION=de0ddfb637590c7ecb54fa0b5301f6dc7dfbcee5
export PROGRAMBENCH_REPO="${PROGRAMBENCH_REPO:-}"
"$PY" -m alo_cloud.evaluate --run-dir "$RUN_DIR" --execute \
  --dataset-revision 6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2 \
  --programbench-hf-revision de0ddfb637590c7ecb54fa0b5301f6dc7dfbcee5

echo "Resume by re-running this script. State: $RUN_DIR"
