# Follow-up cost at r2 scale

This is an estimate from the settled r2 ledger, not a permit to collect. The ledger has 6100 requests and 6022 settled ones. Settled spend is **71.64 CNY**.

Per-unit `charged_microcny` in `records.jsonl` sums to **33.01 CNY**. That field is the final attempt's observed charge. It leaves out:

- **24.93 CNY** charged at the full reservation for timeout or unknown outcomes (stream idle, deadline, and other)
- **13.70 CNY** from attempts that are not the final record, almost all of it in restarted units

33.01 + 24.93 + 13.70 = 71.64. Rates below divide the settled ledger total, including those two gaps, by settled requests.

| Model | Records CNY | Timeouts CNY | Restarts and other CNY | Ledger CNY | CNY per request |
|---|---:|---:|---:|---:|---:|
| deepseek-flash | 16.99 | 4.74 | 5.40 | 27.13 | 0.01039 |
| mimo-v2.6-flash | 5.72 | 6.47 | 3.64 | 15.83 | 0.00665 |
| glm-5.3-flash | 1.37 | 1.12 | 0.33 | 2.81 | 0.02024 |
| mimo-v2.6-pro | 8.93 | 12.61 | 4.32 | 25.87 | 0.02897 |

G2-context deepseek-flash is **0.01309 CNY per request**, higher than the overall 0.01039. G1-main accounts for **1525** ledger requests. G2 accounts for **1932**. The s1 share of G2 model calls, from 62 adapters (planner 62, s1 989, s2 264), is **0.752**. The new G2 arms already set `fixed_topology` and `lightweight_verifier`, so the light rows are what that code would spend. "Default" keeps planner and s2 requests.

`ledger_full_scenarios` and `ledger_scope_options` recompute these tables from the hard-coded `R2_SETTLED_LEDGER` constants in `alo_cloud.followup_cost`. They do not open a database unless you pass a ledger dict. `summarize_settled_ledger(path)` reads `budget.sqlite3` read-only (`mode=ro`, `PRAGMA query_only`) and returns settled CNY and request counts. The s1 share, 0.752, is not in the ledger, so that constant stays even when the money is read from sqlite.

## Full repeat plus all three new arms

deepseek-flash is over the 50 CNY cap in every one of these, from **61.86 to 74.61 CNY**.

| Scenario | deepseek-flash | mimo-v2.6-flash | glm-5.3-flash | mimo-v2.6-pro | Total |
|---|---:|---:|---:|---:|---:|
| Default requests, overall deepseek rate | **68.08** | 19.04 | 12.59 | 39.86 | 139.57 |
| Light G2 (s1 share), overall deepseek rate | **61.86** | 18.24 | 10.17 | 36.39 | 126.66 |
| Default requests, G2 deepseek rate | **74.61** | 19.04 | 12.59 | 39.86 | 146.09 |
| Light G2, G2 deepseek rate | **66.77** | 18.24 | 10.17 | 36.39 | 131.57 |

Dropping only `g2-random` from the light overall scenario still leaves deepseek-flash at about **58.08 CNY**. That arm is not the whole overage.

The repeat is the ledger itself (deepseek-flash 27.13). g1-deepseek-flash is 1525 requests at the overall deepseek rate (**15.85**). g2-fixed and g2-random split the 1932 G2 requests. Light multiplies those G2 requests by 0.752 before pricing.

## Options that keep deepseek-flash at or under 50 CNY

### 1. New arms only, no r2 repeat, light G2

This is the code path (`fixed_topology`, `lightweight_verifier`) without rerunning r2. deepseek-flash **34.72**.

| Model | CNY |
|---|---:|
| deepseek-flash | 34.72 |
| mimo-v2.6-flash | 2.42 |
| glm-5.3-flash | 7.35 |
| mimo-v2.6-pro | 10.52 |
| **Total** | **55.02** |

Of the deepseek spend: g1-deepseek-flash 15.85, g2-fixed 15.10, g2-random 3.78.

### 2. New arms only, no r2 repeat, full G2 request counts

Same arms if s2 model calls are not removed. deepseek-flash **40.95**.

| Model | CNY |
|---|---:|
| deepseek-flash | 40.95 |
| mimo-v2.6-flash | 3.21 |
| glm-5.3-flash | 9.78 |
| mimo-v2.6-pro | 13.99 |
| **Total** | **67.93** |

### 3. Full r2 repeat plus g1-deepseek-flash only

Drop both G2 control arms. deepseek-flash **42.98**.

| Model | CNY |
|---|---:|
| deepseek-flash | 42.98 |
| mimo-v2.6-flash | 15.83 |
| glm-5.3-flash | 2.81 |
| mimo-v2.6-pro | 25.87 |
| **Total** | **87.49** |

### 4. Full r2 repeat, g1-deepseek-flash, and a smaller g2-fixed

No `g2-random`. Light g2-fixed on **46.47%** of the G2 units (about 41 of 88) puts deepseek-flash **on the 50 CNY cap** at the overall deepseek rate (7.02 CNY of g2-fixed on top of 42.98). That schedule sits exactly on the cap. Priced at the deepseek rate observed inside G2 (0.01309 CNY per request) the same slice is **51.83 CNY**, over the cap. A **40%** slice at the overall rate is **49.02 CNY**, so the headroom is **0.98 CNY**. At that same G2 rate the 40% slice is **50.59 CNY**, which is over the cap. The largest slice that stays at 50 CNY at the G2 rate is about **36.8%**. mimo-v2.6-pro stays at the repeat's 25.87 because this option does not route random traffic onto it.

| Model | CNY at the 46.47% slice (overall rate) | CNY at 40% |
|---|---:|---:|
| deepseek-flash | 50.00 | 49.02 |
| mimo-v2.6-flash | 15.83 | 15.83 |
| glm-5.3-flash | 2.81 | 2.81 |
| mimo-v2.6-pro | 25.87 | 25.87 |

The other three models stay under 50 CNY in every option above. mimo-v2.6-pro is the next closest, and only in the full-repeat scenarios that were already over on deepseek (about 36–40).

These figures assume the r2 timeout rate repeats and that G1 requests priced as deepseek-flash cost the overall deepseek rate. They are not a token-level repricing.
