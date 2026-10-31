#!/bin/sh
# M1 then M2 as N task shards on one shared ledger.
# NUM_SHARDS defaults to 4. Completed units in records.jsonl are skipped.
# A halt written by any shard stops the others before their next unit.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
RUN_DIR=${ALO_RUN_DIR:-"$HOME/alo-m1-m2-run"}
DATA_DIR=${ALO_DATA_DIR:-"$RUN_DIR/public_tasks"}
N=${NUM_SHARDS:-4}
PY=${PYTHON:-python3}
cd "$ROOT"

case "$N" in
  ''|*[!0-9]*) echo "NUM_SHARDS must be a positive integer" >&2; exit 2 ;;
esac
if [ "$N" -lt 1 ]; then
  echo "NUM_SHARDS must be a positive integer" >&2
  exit 2
fi

"$PY" -m alo_cloud.freeze --check
if [ -f "$RUN_DIR/drain.json" ]; then
  "$PY" -m alo_cloud.ledger drain --run-dir "$RUN_DIR" --clear
  echo "Cleared drain.json from the previous stop."
fi
"$PY" scripts/smoke_test.py --live --run-dir "$RUN_DIR"

if [ ! -d "$DATA_DIR/SWE-bench_Lite" ] || [ ! -d "$DATA_DIR/ProgramBench" ]; then
  echo "Stage public task JSON at $DATA_DIR/<track>/<task_id>.json"
  exit 2
fi

run_study() {
  study=$1
  i=0
  pids=""
  while [ "$i" -lt "$N" ]; do
    "$PY" -m alo_cloud.run --study "$study" --live --run-dir "$RUN_DIR" \
      --data-dir "$DATA_DIR" --shard "$i/$N" &
    pids="$pids $!"
    i=$((i + 1))
  done
  failed=0
  for pid in $pids; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  return "$failed"
}

set +e
run_study m1
m1_status=$?
echo "M1 blocked-reason histogram"
"$PY" -m alo_cloud.run --blocked-report --run-dir "$RUN_DIR"
report_status=$?
set -e
if [ -f "$RUN_DIR/drain.json" ]; then
  echo "Drain requested. M2 was not started. Shards finished the current unit and exited."
  exit 0
fi
if [ "$m1_status" -ne 0 ] || [ "$report_status" -ne 0 ]; then
  echo "M1 stopped (run=$m1_status, blocked_report=$report_status). M2 was not started."
  exit 1
fi

run_study m2
"$PY" -m alo_cloud.collect --run-dir "$RUN_DIR"

export PROGRAMBENCH_HF_REVISION=de0ddfb637590c7ecb54fa0b5301f6dc7dfbcee5
export PROGRAMBENCH_REPO="${PROGRAMBENCH_REPO:-}"
"$PY" -m alo_cloud.evaluate --run-dir "$RUN_DIR" --execute \
  --dataset-revision 6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2 \
  --programbench-hf-revision de0ddfb637590c7ecb54fa0b5301f6dc7dfbcee5

echo "Resume by re-running this script. Shards: $N. State: $RUN_DIR"
