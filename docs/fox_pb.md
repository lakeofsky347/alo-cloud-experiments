# fox-pb-v1

ProgramBench transfer arms on a restricted Foxrouter pool. This is not an r1, r2, or r3 arm. It does not enter `iter_r2_units`, does not change the frozen `g2` arm, and does not change `r2_freeze_sha256`.

`python -m alo_cloud.fox_pb` is the command. `--live` is the only path that reads `FOXROUTER_API_KEY` or opens a socket. A dry run uses the fixture gateway and makes no paid call.

## Pool

The pool is the fox pool restricted to four models, in this order:

| Report name | Internal name |
| --- | --- |
| `deepseek-flash` | `fox/deepseek-flash` |
| `fox/kimi-k3.1-pro` | `fox/kimi-k3.1-pro` |
| `fox/mimo-v2.6-flash` | `fox/mimo-v2.6-flash` |
| `fox/glm-5.3-flash` | `fox/glm-5.3-flash` |

`deepseek-flash` is the report name of `fox/deepseek-flash` and is the best-single model. Qwen is not a member. Any model id containing `omni` is rejected. `--models` selects a subset. The planner is `fox/deepseek-flash` when it is in the subset, otherwise the first selected model.

## Arms

Each arm is one process and one run directory. Lanes do not share a directory.

| Arm | What it is |
| --- | --- |
| `A1` | ALO with a frozen SWE learning-stream profile imported read-only |
| `A2` | ALO cold start from the common prior. Primary control |
| `A3` | Resources-only transfer. Quality fields are stripped (`learn_quality=False`) |
| `A4` | Best single, `fox/deepseek-flash`, one tool-loop node |
| `A5` | Same model as both `p1` and `p2` |

`A1` and `A3` require `--swe-profile`, a `fox_learn export` transfer package or learning-state file. The import checks the source digest, loads it through the probe temporary store, and does not write the source. Model ids are mapped onto the PB pool. Qwen and `omni` rows are not imported.

## Decision defaults

These are recorded in `manifest.json` and in that run's `pb_freeze` block. `pb_freeze_sha256` is the hash of those decisions. It is not `r2_freeze_sha256`.

| Decision | Default | Flag |
| --- | --- | --- |
| Role map | `s1->p1`, `s2->p2` | `--role-map s1=p1,s2=p2` or `swap` |
| Behaviour-probe script | belongs to `p1` | `--probe-script-owner p1\|p2` |
| `./executable` differential | off. Compile success is the public signal | `--differential-signal` |
| Online update | off. The arm stays on its initial profile | `--online-update` |
| Request cap | 40 (also 64 or 80) | `--request-cap` |
| Per-request deadline | 600 seconds | `--read-timeout` |
| Per-unit wall clock | 2400 seconds | `--unit-seconds` |

PB is fully frozen unless `--online-update` is set. With the default, A1, A2, and A3 use the initial profile on every task, and an in-task observation does not change the next assignment. Unknown compile or differential results are not written into learning state.

The public signal is the return code of the `compile.sh` command the worker actually ran. When `--differential-signal` is on and `./build/executable` exists, the same inputs are run through `./executable` and `./build/executable`. The comparison is stdout, stderr, exit code, and filesystem side effects. It does not read the reference bytes.

`--data-dir` holds staged public tasks at `ProgramBench/<task_id>.json`. `--live` requires that file. `--programbench-repo` is the checkout `--execute` scores with `programbench eval`. A missing checkout is `unavailable`, not a zero. A live unit copies the task image's `/workspace` and attaches a `--network=none` sandbox. A dry run attaches that sandbox with execution off. An execute-only reference stays in the image.

## Reference binary

`./executable` is execute-only (mode 111). Running it and reading stdout, stderr, the exit code, and filesystem side effects is allowed. The role contract and the command filter reject, for that reference binary:

- reading its bytes (`read_file`, `cat`, `dd`, `open(..., "rb")`)
- disassembly or inspection (`objdump`, `strings`, `hexdump`, `xxd`, `ghidra`, `readelf`)
- instrumentation (`strace`, `ltrace`, `gdb`, and the same family)
- copying it into the submission
- any network client, URL fetch, or package install

A submission scan flags a built tree that copies the reference bytes, embeds its sha256, or invokes `./executable` from the submitted program. `compile.sh` may still install `build/executable` onto `./executable`. That output path is not a wrapper.

## G0

G0 is A2 and A4 on `kisielk__errcheck.dacab89`, `raviqqe__muffet.a882908`, and `esubaalew__run.0fb9dec` (six units). Short ids are accepted. The two arms get separate directories under `--root`.

```sh
python -m alo_cloud.fox_pb g0 --root /tmp/fox-pb-g0 --dry-run
```

One arm:

```sh
python -m alo_cloud.fox_pb run \
  --arm A2 \
  --tasks kisielk__errcheck,raviqqe__muffet,esubaalew__run \
  --run-dir /tmp/fox-pb-g0/A2 \
  --data-dir "$PB_DATA_DIR" \
  --dry-run
```

Live scoring also names the ProgramBench checkout:

```sh
python -m alo_cloud.fox_pb run \
  --arm A2 \
  --g0 \
  --run-dir /tmp/fox-pb-g0/A2 \
  --data-dir "$PB_DATA_DIR" \
  --programbench-repo "$PROGRAMBENCH_REPO" \
  --live --execute
```

A later run of the same directory skips a `unit_id` that already has a terminal record. The manifest refuses a different model list, task list, request cap, freeze flag, differential flag, role map, or live/dry mode. Use a new directory for `--live`.

`--execute` with `--live` is what asks for the official cleanroom build and `programbench eval`. A dry run records the plan and leaves `resolved` null. Missing scores stay null.

```sh
python -m alo_cloud.fox_pb report --run-dir /tmp/fox-pb-g0/A2 --run-dir /tmp/fox-pb-g0/A4
```

The report plugs unit rows into the shared comparison tables and adds the pairs A1-A2, A1-A3, A1-A4, and A5-A4, the A1-vs-A2 first-assignment difference rate, cost per unit, compile rate, and the mean and sample standard deviation of per-task cost and score.
