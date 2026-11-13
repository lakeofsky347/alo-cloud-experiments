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

HTTP 429, 502, 503, 504, 520–524, timeouts, and connection resets are retried with backoff. Each failed attempt is appended to `failures.jsonl` with status, Cloudflare error code when the body has one, and a UTC timestamp.

## Task set

SWE-bench Lite revision `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`.

| Split | Instances | Units with all 6 models |
| --- | ---: | ---: |
| `dev` | 23 | 138 |
| `test` | 300 | 1800 |
| `both` | 323 | 1938 |

`dev` is the official Lite development split (the 23 ids are in `alo_cloud.fox_pool.SWEBENCH_LITE_DEV`). `test` is the 300 ids already in `SWEBENCH_LITE_FRAME`. They do not overlap. Best-single can later be locked on `dev`. The commands below do not start that full grid by themselves: pass `--all`, `--limit`, or `--instance-ids`.

The agent is the existing G1 single-model loop: same public tools, `repro-v1`, `fair-control-v1`, `max_attempts=1`. Default `--turn-cap` is 24, the fair-control M1 request cap. Smoke uses 4 unless you pass another cap. The resolved bit in the summary is the SWE-bench harness result, not the repro-v1 signal. Both are stored.

## Commands

Estimate, no API call:

```sh
python -m alo_cloud.fox_single estimate --split dev --all
python -m alo_cloud.fox_single estimate --split test --all
python -m alo_cloud.fox_single estimate --split dev --limit 5 --models fox/deepseek-flash
```

`units` in that JSON is models × instances.

Stage public task JSON at `$FOX_DATA_DIR/SWE-bench_Lite/<instance_id>.json` (problem statement, repo, `base_commit`). Gold patches and tests are stripped if they are present. Do not put the API key in those files.

Smoke on a machine that already has Docker, the SWE-bench instance images, `swebench==4.1.0`, and the key. One instance, one model:

```sh
export FOXROUTER_API_KEY=...
python -m alo_cloud.fox_single smoke --live --execute \
  --run-dir /tmp/fox-single-v1-smoke \
  --data-dir "$FOX_DATA_DIR" \
  --split dev --limit 1 \
  --models fox/deepseek-flash
```

Two instances, every fox model:

```sh
python -m alo_cloud.fox_single smoke --live --execute \
  --run-dir /tmp/fox-single-v1-smoke \
  --data-dir "$FOX_DATA_DIR" \
  --split dev --limit 2 \
  --models all
```

A larger run is explicit. This example is the full dev split, not a default:

```sh
python -m alo_cloud.fox_single run --live --execute \
  --run-dir /tmp/fox-single-v1 \
  --data-dir "$FOX_DATA_DIR" \
  --split dev --all \
  --models all \
  --turn-cap 24
```

Swap `--split test --all` for the 300-instance split (1800 units). Add `--budget-cap N` only if you want a stop on summed raw `cost`. Add `--context-cap fox/mimo-v2.6-flash=50000` to lower that model's cap. `--context-policy trim` compacts instead of failing the call.

Report:

```sh
python -m alo_cloud.fox_single report --run-dir /tmp/fox-single-v1
```

`summary.json` has one row per report-model × split: instances, resolved count, resolved rate, prompt / completion / reasoning tokens, raw cost / credit / quota sums, retry count, suspected-truncation count, and the failure taxonomy (`provider_error`, `timeout`, `context_limit`, `suspected_truncation`, `empty_patch`, `apply_fail`, `tests_fail`, `turn_cap`, `eval_unavailable`, `budget_cap`, and setup gaps such as `payload_missing`).

Per-call rows are in `calls.jsonl` (`raw_usage`, `cost`, `credit`, `quota`, upstream id, retries). Retries and unit failures are in `failures.jsonl`. Predictions for the harness are under `sealed/swe_units/`.

`--live` is what opens the socket. Offline tests inject a sender and do not read a real key. This repository does not start the paid run.
