# Follow-up cost at r2 scale

This is an estimate, not a permit to collect. Reproduce it with the r2 ledger:

```sh
python3 scripts/estimate_followup_cost.py \
  --records /path/to/records.jsonl \
  --output-dir /tmp/alo-followup-cost
```

The frozen r2 schedule is 168 M1 units and 70 M2 units. The follow-up adds three arms on those same tasks: `g1-deepseek-flash`, `g2-fixed`, and `g2-random`. The cap used here is 50 CNY per model, on a fresh round. It is not added on top of the r2 ledger.

## What the attached records contain

Sum of `charged_microcny` in the attached `records.jsonl` (242 rows, 168 unique M1 units and 70 unique M2 units):

| Model | CNY | CNY per request |
|---|---:|---:|
| deepseek-flash | 16.99 | 0.014058 |
| mimo-v2.6-flash | 5.72 | 0.005017 |
| glm-5.3-flash | 1.37 | 0.014175 |
| mimo-v2.6-pro | 8.93 | 0.014292 |
| **Total** | **33.01** | |

The per-request rate divides that charge by the unit's `model_requests`, split across models in the proportion of the transport counts. Transport rows are fewer than `model_requests` on many units, so the mix is scaled up to the request counter before the division.

The round's ledger total supplied with the run is **71.64 CNY** (deepseek-flash 27.13, mimo-v2.6-pro 25.87, mimo-v2.6-flash 15.83, glm-5.3-flash 2.81). The per-unit field does not add up to that ledger. Timeout epoch counters in the same file are cumulative and do not identify the missing 38.63 CNY by unit, so the projection does not invent a split. A next round that also keeps worst-case timeout reservations would cost more than the table below, by roughly the same gap if the timeout mix repeats.

## Projection from per-unit charges

The repeat is the 33.01 CNY above. New arms use each unit's request count and the CNY-per-request rates above. `g2-fixed` prices every G2 request as `deepseek-flash`. `g2-random` splits those requests across the four models. Planner calls and s2 calls are still included. Blocked and skipped units add nothing.

| Component | deepseek-flash | mimo-v2.6-flash | glm-5.3-flash | mimo-v2.6-pro |
|---|---:|---:|---:|---:|
| r2 repeat | 16.99 | 5.72 | 1.37 | 8.93 |
| g1-deepseek-flash | 14.03 | 0 | 0 | 0 |
| g2-fixed | 22.44 | 0 | 0 | 0 |
| g2-random | 5.61 | 2.00 | 5.66 | 5.70 |
| **Total** | **59.07** | **7.72** | **7.02** | **14.64** |

Total projected spend is **88.45 CNY**. `deepseek-flash` is **9.07 CNY over** the 50 CNY cap. The other three models stay under 50. Under this mix the round cannot finish the deepseek arms without a smaller G2 request count or dropping an arm.

The fixed topology removes one planner request per G2 unit, and the lightweight verifier removes s2 model requests. Those savings are not in the table: the ledger does not split requests by node. They are the practical way to bring deepseek back under 50, because the overage is the new deepseek arms, not mimo or glm.
