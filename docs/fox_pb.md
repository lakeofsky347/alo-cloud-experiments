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
| Request cap | 40 (also 64 or 80). Shared upper bound for the unit | `--request-cap` |
| Per-node request cap | unset, except G0/M18 which gives p1 and p2 20 each | `--node-request-cap` |
| Two-node agent protocol | unset, the M18 protocol | `--pb-agent-protocol node-budget-v1` (M19) or `--pb-agent-protocol build-checkpoint-v1` |
| Per-request deadline | 600 seconds | `--read-timeout` |
| Per-unit wall clock | 2400 seconds | `--unit-seconds` |
| Exact returned `deepseek-flash` | off. That returned string is `model_mismatch` | `--returned-name-opt-in pb-deepseek-flash-exact-return-v1` |

## Exact returned name

The fox/deepseek-flash slot requests upstream `deepseek-v4.1-flash-1017`. `accepted_aliases` stays empty on that catalog row. Alias matching folds a vendor prefix, letter case, and separators, so it is the wrong switch for this case. The default run still rejects HTTP 200 when `payload.model` is the exact string `deepseek-flash`, and records `model_mismatch` / `provider_error`. The requested upstream and the raw returned name stay on the call row.

`--returned-name-opt-in pb-deepseek-flash-exact-return-v1` is the only acceptance. It applies to a copy of the fox/deepseek-flash slot for that run. The request body `model` stays `deepseek-v4.1-flash-1017`. The raw `payload.model` must be exactly `deepseek-flash`: no prefix, no case fold, no separator fold. `fw/deepseek-flash`, `DeepSeek-Flash`, `deepseek_flash`, and `group/deepseek-flash` stay mismatches. Other slots stay mismatches if they return that string. The call row stores `exact_returned_opt_in` with `version`, `slot`, `upstream`, `returned`, `match=exact`, and `identity=uncertain`. That record is the opt-in. It is not `model_alias_normalized` and it is not evidence that the returned name and the upstream id are the same model.

The default `pb_freeze` document omits `exact_returned_opt_in`, so `pb_freeze_sha256` for an otherwise identical run stays `5a0f54fa1a88e42d43369c8f6fe8c293fbde6deb94c5d23789f7d69c1bd0c13c`. The M18 settings document (cap 40, node cap 20, differential on, glm-flashx, no agent-protocol flag) stays `b38db09ff4321ad9249ced71f6cc39e8a095e93ba1371d51abdc915e1d53c062`. With the returned-name flag, that one key is added and the hash changes. `r2_freeze_sha256` stays `77bddfbfd3472ae4c0d280c3dacf20810f11b5f9b5c247c17650629ebf9e241f`. The pool, scoring, and protocol schedule do not change. A new manifest always stores the field (`null` when off). Resuming a directory with a different value, including an older manifest that omitted the key, exits with `fox_run_settings_mismatch` and leaves the manifest bytes in place.

`alo_cloud.fox_pool.exact_returned_name_accepted(internal, upstream, returned, opt_in_version)` is the predicate. An out-of-repo guard that compares `payload.model` with the upstream id, or that rejects the exact string `deepseek-flash`, still drops the response before this process can record it. After this commit is on the box, that guard accepts a response only when the predicate is true, leaves `payload.model` unchanged, and logs the requested upstream, the raw returned name, the opt-in version, and `identity=uncertain`. Every other returned string stays rejected. The guard can check the predicate without a network call:

```sh
python -c "from alo_cloud.fox_pool import exact_returned_name_accepted as ok; raise SystemExit(0 if ok('fox/deepseek-flash','deepseek-v4.1-flash-1017','deepseek-flash','pb-deepseek-flash-exact-return-v1') else 1)"
python -c "from alo_cloud.fox_pool import exact_returned_name_accepted as ok; raise SystemExit(0 if not ok('fox/deepseek-flash','deepseek-v4.1-flash-1017','deepseek-flash',None) else 1)"
python -c "from alo_cloud.fox_pool import exact_returned_name_accepted as ok; raise SystemExit(0 if not ok('fox/deepseek-flash','deepseek-v4.1-flash-1017','fw/deepseek-flash','pb-deepseek-flash-exact-return-v1') else 1)"
```

Offline tests, still with no paid call:

```sh
python -m pytest tests/test_fox_single.py::test_deepseek_flash_exact_return_is_opt_in_and_not_an_alias tests/test_fox_pb.py::test_pb_exact_returned_deepseek_flash_is_opt_in -q -p no:cacheprovider
```

The authorized continuation after those tests is one new A4 directory for the lz4 task already on the schedule (`lz4__lz4.1519f46` in the administrative catalog; pass the same task string the historical schedule used). Use a new `--run-dir`. Do not point the flag at the historical A4 directory. Do not rerun zevv (`zevv__duc.a58fa4e`) and do not rewrite its records. Historical results stay as recorded. Primary A4 is the original 16 plus this lz4 unit, 17. The sensitivity count 18 keeps the anomalous zevv in the appendix.

```sh
python -m alo_cloud.fox_pb run \
  --arm A4 \
  --tasks lz4__lz4.1519f46 \
  --run-dir /tmp/fox-pb-a4-lz4 \
  --returned-name-opt-in pb-deepseek-flash-exact-return-v1 \
  --programbench-repo "$PROGRAMBENCH_REPO" \
  --live --execute
```

That command is the launcher shape. It is not part of the offline check, and it is a paid call only when someone runs it with `--live`.

`--pb-agent-protocol node-budget-v1` needs `--node-request-cap` and applies only to the two-node arms. The `s1` node (`p1` with the default map) sees its own remaining requests in every observation, for example `requests left for your node p1: 12/20 (unit: 32/40)`. The first observation that shows 6 or fewer node requests left also carries `node_budget_notice` (once): write the source and `compile.sh` now and run it once, because the next node cannot write files. No extra message or role is sent. The other node and A4 are unchanged; A4 requests are byte-identical with the flag on or off. Prompts and tools are not in `pb_freeze`, so the flag adds a `pb_agent_protocol` block there and the hash changes on purpose. Without the flag, the manifest, `pb_freeze_sha256` and every request are identical to `753e54a`. A run directory never mixes protocols.

### Build checkpoints

`--pb-agent-protocol build-checkpoint-v1 --request-cap 40 --node-request-cap 20` uses p1 as the implementer and probe owner, followed by p2 as the read-only reviewer. The freeze records the checkpoint, repair window and snapshot policy. Each protocol uses its own run directory.

After p1's eighth request, the harness builds the current submission and returns the first compiler diagnostic. It checks again before the last four requests. During those four requests p1 can read, write or edit source, call `pb_build`, and submit a final summary. A final summary triggers a build of the current snapshot. A failed or unavailable build returns the diagnostic to p1 within its remaining budget. A capped node hands its observed result to the run log. p2 receives a fresh build receipt for its handoff snapshot and can repeat the check. The explicit tool and scheduled checkpoints consume tool and time budgets without making model requests.

The public action is `{"tool":"pb_build","arguments":{}}`. The receipt binds the submitted file contents and modes to a SHA-256 and records build status, process exit, diagnostic and the presence of a newly built root executable. The build uses a disposable writable copy in the offline task image. The final public build reads the same sealed snapshot as the official grader and its receipt is stored in `public_build`. Behaviour probes belong in `.pb-probe/`. With the optional differential signal, each program starts from the same fresh files in its own writable `/agent` directory. Build commands still run at `/workspace`. The final scan and archive use the frozen submission manifest. A write reply reports the path and final byte size. When a reply contains multiple actions, the observation identifies each unexecuted action.

Using the M18 comparison settings (20 requests per node, p1 as probe owner, differential on, online updates off, explicit `--unit-seconds 2400 --read-timeout 600` and the `glm-flashx` pool substitution), the decision SHA-256 for `build-checkpoint-v1` is `f61837c91d8085e367889a58edfcae0b95a61751491725d816198ffc9673c747`. Each run records the hash of its own settings in `manifest.json`.

### Official score evidence

Each new run records `public_compile_success`, `public_differential`, `official_compile_success`, `submission_accepted` and `official_pass_rate`. The official rate uses the stored test cases as its denominator. Official evidence belongs to one arm, task, unit, attempt and sealed archive. An absent result gives a null official outcome. `official_eval_identity` records the matching archive and evaluation digests. A rejected submission retains its separate zero research score.

PB is fully frozen unless `--online-update` is set. With the default, A1, A2, and A3 use the initial profile on every task, and an in-task observation does not change the next assignment. Unknown compile or differential results are not written into learning state.

Under the M18 and M19 protocols the public signal runs `chmod +x compile.sh` and `./compile.sh` in the sandbox, after moving the image reference off `./executable` without reading it. That matches the official scorer at `895f659`, which deletes `./executable` and then requires `compile.sh` to create `./executable` at the workspace root. The return code is the signal, and a missing `./executable` after a zero exit is a fail. When `--differential-signal` is on, the same inputs are run through the preserved reference and that new `./executable`. The comparison is stdout, stderr, exit code, and filesystem side effects. It does not read the reference bytes. With the flag off, compile success is the only public signal.

`--programbench-repo` is a checkout of `facebookresearch/ProgramBench` at `895f659`. Public task text is `src/programbench/data/tasks/<id>/task.yaml` (repository, commit, language, difficulty). `tests.json` is not loaded. `--data-dir` can hold the same `task.yaml` or `ProgramBench/<id>.json`. The prompt names `./executable` (mode 111) and the doc paths copied out of the image. A live unit seeds `programbench/<id>:task_cleanroom_v6` `/workspace`, drops any host copy of the reference binary, and attaches a `--network=none` sandbox that keeps the image binary. A dry run attaches that sandbox with execution off and does not contact a daemon. `--live --execute` passes `official_pb_scorer`, which packs `submission.tar.gz` and runs `programbench eval` from that checkout. The plan carries the image, the cleanroom `compile.sh` build, the eval command, and commit `895f659`. A missing checkout is `unavailable`, not a zero.

## Smoke one G0 unit

Dry run, no API key and no Docker:

```sh
python -m alo_cloud.fox_pb run \
  --arm A4 \
  --tasks kisielk__errcheck \
  --run-dir /tmp/fox-pb-smoke-dry/A4 \
  --programbench-repo "$PROGRAMBENCH_REPO" \
  --dry-run
```

Live, a separate directory so the dry manifest is not resumed. This reads `FOXROUTER_API_KEY`, uses the local cleanroom image, and scores with the official evaluator:

```sh
python -m alo_cloud.fox_pb run \
  --arm A4 \
  --tasks kisielk__errcheck \
  --run-dir /tmp/fox-pb-smoke-live/A4 \
  --programbench-repo "$PROGRAMBENCH_REPO" \
  --live --execute
```

## Reference binary

`./executable` is execute-only (mode 111). Running it and reading stdout, stderr, the exit code, and filesystem side effects is allowed. The role contract and the command filter reject, for that reference binary:

- reading its bytes (`read_file`, `cat`, `dd`, `open(..., "rb")`)
- disassembly or inspection (`objdump`, `strings`, `hexdump`, `xxd`, `ghidra`, `readelf`)
- instrumentation (`strace`, `ltrace`, `gdb`, and the same family)
- copying it into the submission
- any network client, URL fetch, or package install

A submission scan flags a built tree that copies the reference bytes, embeds its sha256, reads those bytes, or invokes `./executable` from a file the build uses. `compile.sh` may install the agent's own `build/...` output onto `./executable`, including `cp build/executable executable 2>/dev/null || true`. That install is not a wrapper. A file the build graph does not reach from `compile.sh` — a root `probe.py`, a script under `.pb-probe/`, or any other unused probe — may run the reference through its normal interface. That is a warning on the unit result, and the file is left out of `submission.tar.gz`. A build input that embeds, copies, wraps, execs, or reads the reference, or that runs `objdump`, `strings`, `strace`, or `ltrace` on it, is still rejected. A rejected unit records score 0 and the reason. When an offline runner is available, its score is stored beside that zero and is not the unit score.

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
