# fox-single-v1

Single-model SWE-bench Lite baseline through Foxrouter. This is not an r1, r2, or r3 arm. It does not change `r2_freeze_sha256`, the mainland pay-as-you-go pool, or frozen `g2`.

The run directory must be a new folder whose name is not `r1`, `r2`, `r3`, `alo-r1-run`, `alo-r2-run`, or `alo-r3-run`. Results stay in that directory.

## Models

Gateway: `https://foxrouter.org/v1/chat/completions`. Auth is `Authorization: Bearer` from `FOXROUTER_API_KEY`. The process checks that the variable is set. It does not write the value into manifests, ledgers, or logs.

Every request sends `User-Agent: curl/8.5.0`. Cloudflare returns HTTP 403 / error 1010 for the default Python agent.

| Report name | Internal name | Upstream id | Context cap | max_tokens |
| --- | --- | --- | --- | --- |
| `deepseek-flash` | `fox/deepseek-flash` | `deepseek-v4.1-flash-1017` | 128000 | 16384 |
| `fox/kimi-k3.1-pro` | `fox/kimi-k3.1-pro` | `kimi-k3.1-pro` | 128000 | 16384 |
| `fox/qwen3.8-max` | `fox/qwen3.8-max` | `qwen3.8-max` | 128000 | 16384 |
| `fox/qwen3.8-flash` | `fox/qwen3.8-flash` | `qwen3.8-flash` | 128000 | 16384 |
| `fox/glm-5.3-flash` | `fox/glm-5.3-flash` | `glm-5.3-flash` | 128000 | 16384 |
| `fox/mimo-v2.6-flash` | `fox/mimo-v2.6-flash` | `mimo-v2.6-flash` | 60000 | 16384 |

Reports call DeepSeek `deepseek-flash`. The upstream id is metadata only. The other five keep the `fox/` name so they do not collide with the direct-endpoint `glm-5.3-flash` and `mimo-v2.6-flash`.

There is no CNY or USD price table for these models. A missing price does not block a unit. `--budget-cap` is optional and defaults off. When it is set, it compares against the summed raw `cost` numbers the API returned. Missing `cost` does not count and does not block. `credit` and `quota` are stored as returned. They are not converted.

`fox/mimo-v2.6-flash` cannot be given a context cap of 70000 or more. A prompt whose estimate is over the cap is not sent. The default policy is `fail` (`context_limit`). `--context-policy trim` drops the oldest non-system turns and still fails if the remainder is over the cap. If the returned `prompt_tokens` is far below the estimate (the live probe cut ~100k tokens down to 25668), the call is marked `suspected_truncation`.

Reasoning models get `max_tokens` 16384 unless `--max-tokens` overrides it. Token totals use the reported usage, including `reasoning_tokens` when present. Kimi often omits that count even when `reasoning_content` is present; the count stays unset rather than zero. A non-streaming completion of exactly 512 tokens is recorded as `usage_quirk=completion_tokens_exactly_512`. The raw `usage` object is stored on each call.

The gateway sometimes returns a cosmetic `model` id. Matching ignores a vendor prefix (text up to the last `/`), letter case, and whether `.`, `_`, or `-` was used. One trailing snapshot tag may also differ: `-1017`, `-20251017`, `-latest`, or `-preview`. These stay the same model: the DeepSeek upstream id `deepseek-v4.1-flash-1017` with its trailing `-1017` tag removed; `kimi-k3.1-pro` and `kimi-k3-1-pro`; `qwen3.8-max` and `qwen/qwen3.8-max`; `qwen3.8-flash` and `qwen/qwen3-8-flash` (a later probe also returned `qwen3.8-flash` exactly). `glm-5.3-flash` and `mimo-v2.6-flash` already match exactly. A different family or size is still `provider_error` (`qwen3.8-max` vs `qwen3.8-flash`, `kimi-k3.1-pro` vs `kimi-k3.1`, flash vs pro). `p` is not a separator, so `3p1` does not fold into `3.1` for every model. `FoxModel.accepted_aliases` can list extra ids for one model. Alias comparison uses the same prefix strip, so an unprefixed entry also accepts `fw/<id>`. A later live smoke returned `fw/kimi-k3p1-pro` for requested `kimi-k3.1-pro` (`3p1` is that gateway's spelling of `3.1`, after an earlier `kimi-k3-1-pro`). `fox/kimi-k3.1-pro` therefore lists `kimi-k3p1-pro`, which also accepts `fw/kimi-k3p1-pro`. From 2026-11-16 14:36 SGT the gateway returned `kimi-k3.1-pro-ta` for requested `kimi-k3.1-pro`; `-ta` is not a snapshot tag, so it is listed as a second accepted alias. Every call row keeps the gateway's `returned_model` verbatim. `kimi-k3p2-pro` still mismatches Kimi, and `kimi-k3p1-pro` does not match `qwen3.8-max`. The other five scheduled models leave the tuple empty.

Every call row stores `returned_model` and the requested `upstream_model`. A match that needed this normalization, including an accepted alias, sets `model_alias_normalized` to those two ids. That field is a warning, and the call is not failed for it. A real mismatch stays `provider_error`, sets `model_mismatch`, and writes both ids on the call row and in `failures.jsonl`. Prompt, completion, reasoning, and total tokens, plus raw `cost` / `credit` / `quota`, are copied from the response even when that call is rejected. The unit report still sums them, including calls that succeeded before the rejected one.

HTTP 429, 502, 503, 504, 520–524, timeouts, and connection resets are retried with backoff. `--read-timeout` (default 600 seconds) is the wall-clock budget for one attempt, covering connect, an HTTPS proxy `CONNECT`, headers, and the body. Keepalive or trickle bytes from the gateway do not extend that attempt: at the deadline the socket is aborted, the attempt is stored as `timeout`, and the same backoff runs. Each failed attempt is appended to `failures.jsonl` with status, Cloudflare error code when the body has one, and a UTC timestamp.

## Task set

SWE-bench Lite revision `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`.

The first run is the **dev** split: 23 instances × 6 models = 138 units. `--split` defaults to `dev`. The test split (300) is deferred and stays selectable with `--split test`.

| Split | Instances | Units with all 6 models | This run |
| --- | ---: | ---: | --- |
| `dev` (default) | 23 | 138 | first run |
| `test` | 300 | 1800 | deferred |
| `both` | 323 | 1938 | not scheduled |

`dev` is the official Lite development split (the 23 ids are in `alo_cloud.fox_pool.SWEBENCH_LITE_DEV`). `test` is the 300 ids already in `SWEBENCH_LITE_FRAME`. They do not overlap. Best-single can later be locked on `dev`. A full grid still needs `--all`. `--limit` and `--instance-ids` select a smaller set. Instance ids from `test` need `--split test` or `--split both`, because the default split is `dev`.

The agent is the existing G1 single-model loop: same public tools, `repro-v1`, `fair-control-v1`, `max_attempts=1`. Default `--turn-cap` is 24, the fair-control M1 request cap. Smoke uses 4 unless you pass another cap. The resolved bit in the summary is the SWE-bench harness result, not the repro-v1 signal. Both are stored.

## Commands

Estimate, no API call. The first command is the scheduled dev grid (138). The test line is only a size check:

```sh
python -m alo_cloud.fox_single estimate --all
python -m alo_cloud.fox_single estimate --split test --all
python -m alo_cloud.fox_single estimate --limit 5 --models fox/deepseek-flash
```

`units` in that JSON is models × instances. `--all` with no `--split` is dev.

After a dev run, extrapolate a test-300 cost from the observed per-instance tokens and raw cost/credit. The block is labeled `estimate`. It is not a measurement of the test split. Each model is mean ± sample standard deviation per instance; the test-300 total is that mean times 300, and the spread is the standard deviation times 300 (not a confidence interval):

```sh
python -m alo_cloud.fox_single estimate --from-run-dir /tmp/fox-single-v1
python -m alo_cloud.fox_single report --run-dir /tmp/fox-single-v1
```

`report` writes the same `test_300_extrapolation` object into `summary.json`.

Stage public task JSON at `$FOX_DATA_DIR/SWE-bench_Lite/<instance_id>.json` (problem statement, repo, `base_commit`). Gold patches and tests are stripped if they are present. Do not put the API key in those files.

The run machine already has the official images `swebench/sweb.eval.x86_64.<instance>:latest` (Docker 29, Python 3.12 via uv, `swebench==4.1.0`). fox-single copies `/testbed` out of a local image with `docker create` and `docker cp`, then removes only that temporary container. It does not `docker pull`, `docker rmi`, `docker image rm`, or any `docker` `prune` (including `docker image prune` and `docker system prune`).

Smoke, one dev instance, one model:

```sh
export FOXROUTER_API_KEY=...
python -m alo_cloud.fox_single smoke --live --execute \
  --run-dir /tmp/fox-single-v1-smoke \
  --data-dir "$FOX_DATA_DIR" \
  --limit 1 \
  --models fox/deepseek-flash
```

Two dev instances, every fox model:

```sh
python -m alo_cloud.fox_single smoke --live --execute \
  --run-dir /tmp/fox-single-v1-smoke \
  --data-dir "$FOX_DATA_DIR" \
  --limit 2 \
  --models all
```

The first full run is the default dev split (138 units). `--all` is what selects all 23 instances; omitting `--split` still means dev:

```sh
python -m alo_cloud.fox_single run --live --execute \
  --run-dir /tmp/fox-single-v1 \
  --data-dir "$FOX_DATA_DIR" \
  --all \
  --models all \
  --turn-cap 24
```

`--split test --all` remains available later (1800 units) and is not this run. Add `--budget-cap N` only if you want a stop on summed raw `cost`. Add `--context-cap fox/mimo-v2.6-flash=50000` to lower that model's cap. `--context-policy trim` compacts instead of failing the call.

Report:

```sh
python -m alo_cloud.fox_single report --run-dir /tmp/fox-single-v1
```

`summary.json` has one row per report-model × split: instances, resolved count, resolved rate, prompt / completion / reasoning tokens, raw cost / credit / quota sums, retry count, suspected-truncation count, and the failure taxonomy (`provider_error`, `timeout`, `context_limit`, `suspected_truncation`, `empty_patch`, `apply_fail`, `tests_fail`, `turn_cap`, `eval_unavailable`, `budget_cap`, and setup gaps such as `payload_missing`).

Per-call rows are in `calls.jsonl` (`raw_usage`, `cost`, `credit`, `quota`, requested `upstream_model`, `returned_model`, retries). A normalized gateway id adds `model_alias_normalized`. Retries and unit failures are in `failures.jsonl`. A model mismatch there includes both ids. Predictions for the harness are under `sealed/swe_units/`.

`--live` is what opens the socket. Offline tests inject a sender and do not read a real key. This repository does not start the paid run.
