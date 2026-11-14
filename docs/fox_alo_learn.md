# fox-alo-learn-v1

ALO multi-model learning stream on the Foxrouter pool. This is not an r1, r2, or r3 arm. It does not change `r2_freeze_sha256`, the frozen `g2` arm, scoring, or the mainland pay-as-you-go ledger.

`python -m alo_cloud.run` stays on that ledger. `--arm g2-learn` or `--arm g2-resources` there prints a pointer to this command and does not schedule a unit.

The run directory must be a new folder whose name is not `r1`, `r2`, `r3`, `alo-r1-run`, `alo-r2-run`, or `alo-r3-run`. Each chain has its own directory.

## Pool

`--pool fox` is the gateway pool. `--pool domestic` is refused here: the domestic pool is still `deepseek-flash`, `mimo-v2.6-flash`, `glm-5.3-flash`, and `mimo-v2.6-pro`, with the CNY budget ledger.

| Report name | Internal name | Upstream id |
| --- | --- | --- |
| `deepseek-flash` | `fox/deepseek-flash` | `deepseek-v4.1-flash-1017` |
| `fox/kimi-k3.1-pro` | `fox/kimi-k3.1-pro` | `kimi-k3.1-pro` |
| `fox/qwen3.8-max` | `fox/qwen3.8-max` | `qwen3.8-max` |
| `fox/qwen3.8-flash` | `fox/qwen3.8-flash` | `qwen3.8-flash` |
| `fox/glm-5.3-flash` | `fox/glm-5.3-flash` | `glm-5.3-flash` |
| `fox/mimo-v2.6-flash` | `fox/mimo-v2.6-flash` | `mimo-v2.6-flash` |

Membership is the list you pass. `--models all` is the six names above, in that order. A subset is a comma-separated `--models` list or one name per line in `--models-file`. `deepseek-flash` is accepted as the report name of `fox/deepseek-flash`. `qwen3.8-omni-flash` is not a member and is not an alias of `fox/qwen3.8-flash`.

The client, alias rule, and wall-clock `--read-timeout` (default 600 seconds, including an HTTPS proxy `CONNECT`) are the fox-single client. Requests go to `https://foxrouter.org/v1/chat/completions`. Auth is `Authorization: Bearer` from `FOXROUTER_API_KEY` only when `--live` sends. The value is not written into manifests, ledgers, or logs.

There is no CNY or USD price. A missing price does not block a unit. `--budget-cap` is optional and defaults off. When it is set, the chain stops before the next unit once the summed raw `cost` is above the cap. Missing `cost` does not count. `credit` and `quota` are stored as returned.

The planner is `fox/deepseek-flash` when that model is in the subset. Otherwise it is the first selected model in the table order.

## Chains

Six independent chains:

| Chain id | Arm | What carries to the next task |
| --- | --- | --- |
| `persist-forward` | `g2-learn` | profile, including quality |
| `persist-reverse` | `g2-learn` | profile, including quality |
| `reset-forward` | `g2-learn` | nothing; each task starts from the common prior |
| `reset-reverse` | `g2-learn` | nothing; each task starts from the common prior |
| `resources-only-forward` | `g2-resources` | resource rows only |
| `resources-only-reverse` | `g2-resources` | resource rows only |

Forward keeps the selected task order. Reverse flips it. `--arm g2-learn` keeps the four persist/reset chains. `--arm g2-resources` keeps the two resources-only chains. `--condition` and `--order` narrow that further.

Within a chain, units run one after another. Persist imports the previous unit's learning state. Reset does not. Resources-only imports the stripped export: belief and TMS rows are dropped, and quality fields are dropped, which is the #35 resources export. The next task still rebuilds the workspace, the session, and the role identities.

Each chain is one process. Give it a distinct `--run-dir`. The run id is `fox-alo-learn-v1.<chain-id>`. The host workdir is `units/<unit_id>/state/workspace` under that directory. The Docker container name is `alo-fox-<chain>-<12 hex chars>`, so parallel processes do not reuse a container name. A sandbox that is not given a name keeps the previous `docker run` argv.

`--turn-cap` defaults to 24, the same default as fox-single dev. It is the unit request ceiling. `g2-learn` still gives each worker node `max(1, turn_cap - 4)` turns, leaving the planner request and the verifier reserve from the existing learning config.

A later run of the same chain skips a `unit_id` that already has a terminal record (`completed`, `failed`, `blocked`, `stopped`). Delete that record to retry it. The manifest refuses a different model list, task list, turn cap, or budget cap in the same directory.

## Task list

SWE-bench Lite revision `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`. `--split` defaults to `dev`. Pass exactly one selector:

- `--tasks-file tasks.txt` (one instance id per line, `#` comments, or a JSON list)
- `--tasks id1,id2`
- `--n 4 --seed fox-alo-learn-v1` (stable hash order inside the split)
- `--all` (the whole split)

## What a unit records

`records.jsonl` has, per unit: the role assignments and the model chosen for each role (`assignments`, `node_assignments`, `roles_by_model`), `profile_hash_before` and `profile_hash_after`, `harness_resolved` from the existing SWE-bench scorer, and token totals plus raw `cost`, `credit`, and `quota`. The profile hash is the carried belief, TMS, and policy observations. It does not include the per-unit evidence id. `--execute` with `--live` runs that scorer. A dry run records `harness_resolved: null` and still writes the prediction row.

After a persist chain, export the final profile. Reset has nothing to carry and the export command refuses it. Resources-only exports the stripped package.

```sh
python -m alo_cloud.fox_learn export --run-dir /tmp/fox-alo-learn/persist-forward
```

## Commands

Plan the six chains. This does not call the API. `--root` is where each chain directory will sit. The printed command for each chain is the live launch:

```sh
python -m alo_cloud.fox_learn plan \
  --pool fox \
  --models all \
  --split dev \
  --all \
  --turn-cap 24 \
  --root /tmp/fox-alo-learn \
  --data-dir "$FOX_DATA_DIR"
```

A four-task subset on two models:

```sh
python -m alo_cloud.fox_learn plan \
  --pool fox \
  --models fox/deepseek-flash,fox/qwen3.8-flash \
  --n 4 \
  --seed fox-alo-learn-v1 \
  --turn-cap 24 \
  --root /tmp/fox-alo-learn \
  --data-dir "$FOX_DATA_DIR"
```

Offline check, one chain, no key and no socket:

```sh
python -m alo_cloud.fox_learn run \
  --pool fox \
  --chain persist-forward \
  --run-dir /tmp/fox-alo-learn-dry/persist-forward \
  --models fox/deepseek-flash,fox/qwen3.8-flash \
  --n 2 \
  --seed fox-alo-learn-v1 \
  --turn-cap 24 \
  --dry-run
```

Live chains. Set the key in the environment. Do not put it in the command line. Start one process per chain. The six directories, run ids, workdirs, and container names do not overlap.

```sh
export FOXROUTER_API_KEY=...
python -m alo_cloud.fox_learn run \
  --pool fox --chain persist-forward \
  --run-dir /tmp/fox-alo-learn/persist-forward \
  --models all --split dev --all --turn-cap 24 \
  --data-dir "$FOX_DATA_DIR" --live --execute

python -m alo_cloud.fox_learn run \
  --pool fox --chain persist-reverse \
  --run-dir /tmp/fox-alo-learn/persist-reverse \
  --models all --split dev --all --turn-cap 24 \
  --data-dir "$FOX_DATA_DIR" --live --execute

python -m alo_cloud.fox_learn run \
  --pool fox --chain reset-forward \
  --run-dir /tmp/fox-alo-learn/reset-forward \
  --models all --split dev --all --turn-cap 24 \
  --data-dir "$FOX_DATA_DIR" --live --execute

python -m alo_cloud.fox_learn run \
  --pool fox --chain reset-reverse \
  --run-dir /tmp/fox-alo-learn/reset-reverse \
  --models all --split dev --all --turn-cap 24 \
  --data-dir "$FOX_DATA_DIR" --live --execute

python -m alo_cloud.fox_learn run \
  --pool fox --chain resources-only-forward \
  --run-dir /tmp/fox-alo-learn/resources-only-forward \
  --models all --split dev --all --turn-cap 24 \
  --data-dir "$FOX_DATA_DIR" --live --execute

python -m alo_cloud.fox_learn run \
  --pool fox --chain resources-only-reverse \
  --run-dir /tmp/fox-alo-learn/resources-only-reverse \
  --models all --split dev --all --turn-cap 24 \
  --data-dir "$FOX_DATA_DIR" --live --execute
```

Repeat a chain command to resume. Add `--budget-cap N` only to stop on summed raw `cost`. Add `--tasks-file tasks.txt` instead of `--all` for an explicit id list. `--arm g2-learn` or `--condition persist` limits which chains `plan` prints; `run` still needs `--chain`.

`calls.jsonl` has the per-call raw usage, `cost`, `credit`, `quota`, requested upstream id, and returned id. A normalized gateway id sets `model_alias_normalized`. A different model, including `qwen3.8-omni-flash` for a `qwen3.8-flash` request, is `provider_error` with `model_mismatch`.

`--live` is what opens the socket. This repository does not start the paid run.
