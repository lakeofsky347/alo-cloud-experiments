# Round-2 paid-run freeze

Date: 2026-11-01. Batch identity `r2`. This schedule is not comparable with round 1 and must not be pooled with r1 results, scores, or ledgers. Round 2 has a fresh 50 CNY per model and a fresh 200 CNY total. The frozen carry-in is zero.

freeze_sha256: 77bddfbfd3472ae4c0d280c3dacf20810f11b5f9b5c247c17650629ebf9e241f

The round-1 freeze stays in `protocol_freeze.md`. `python -m alo_cloud.r2 --check` recomputes this hash. The draw seed is `r2-task-draw-20261101-v1`. SWE-bench Lite revision `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`. The draw is largest-remainder by repository, then SHA-256 rank. It does not use round-1 scores.

## Caps and budget

M1 is B0 only. SWE B0 is 24 requests. M2 learning and reference are 16. Probes are 12. `command_seconds` is 120. DeepSeek receives a thinking budget of 8192 tokens on the r2 path. Each model has a 50 CNY hard cap and the four models share 200 CNY, both counted on r2 spend only. `mimo-v2.6-pro` soft-halts at 48 CNY of r2 spend, between units (`hard: false`).

The frozen prior spend is 0 for every model. `ALO_PRIOR_LEDGER` is opt-in. When it is unset, the ledger opens at zero. When it is set, that file is read only and its settled totals become the opening commitments. An unsettled row refuses the run. The applied map is recorded in `r2_manifest.json`.

Model request name stays `deepseek-flash`. Endpoints stay the mainland-China pay-as-you-go URLs.

## ALO user preference

G2 only. Frozen field `user_preference`: first try `deepseek-flash`, strong preset `mimo-v2.6-pro`. The first worker assignment uses deepseek-flash when it passes the budget and resource filters; otherwise it uses the Thompson choice. If the public check fails, or the attempt leaves an empty diff, the second attempt (`max_attempts=2`) uses mimo-v2.6-pro when that model is still feasible. Thompson posteriors still update from every observation. Each decision record stores `preference_applied`, `escalated_to_strong`, and the policy's counterfactual choice. The planner is already deepseek-flash and is not a Thompson draw; it is not called a second time, because that would spend the single reserved planner request. A non-empty implementation diff still counts as success for dependencies, so s2 runs. Learned and cold probes still differ by the organization prior; the preference is an input, not transferred state. G1 and G3 are unchanged.

## M1

48 SWE-bench Lite tasks: the 6 round-1 M1 tasks, then 42 more. Arms `g1-main`, `g2`, and `g3` run on all 48. `g1-strong` runs on this 24-task subset, which includes the 6 round-1 tasks. The 18 extra strong tasks are a seeded draw (salt `g1-strong`, count 18):

- `scikit-learn__scikit-learn-13584`
- `django__django-11583`
- `sympy__sympy-11870`
- `django__django-16408`
- `sympy__sympy-20322`
- `sympy__sympy-18835`
- `sympy__sympy-22005`
- `sphinx-doc__sphinx-7686`
- `django__django-13933`
- `matplotlib__matplotlib-23913`
- `scikit-learn__scikit-learn-25747`
- `pytest-dev__pytest-5227`
- `sympy__sympy-22714`
- `django__django-14155`
- `sympy__sympy-13647`
- `django__django-13658`
- `sympy__sympy-24102`
- `django__django-13220`
- `django__django-16139`
- `django__django-14915`
- `django__django-17051`
- `mwaskom__seaborn-2848`
- `psf__requests-863`
- `astropy__astropy-7746`

All 48 M1 tasks:

- `scikit-learn__scikit-learn-13584` (short)
- `django__django-11583` (short)
- `sympy__sympy-11870` (medium)
- `django__django-16408` (medium)
- `sympy__sympy-20322` (long)
- `sympy__sympy-18835` (long)
- `django__django-14155` (django)
- `django__django-13401` (django)
- `scikit-learn__scikit-learn-13142` (scikit-learn)
- `sympy__sympy-13647` (sympy)
- `matplotlib__matplotlib-22711` (matplotlib)
- `pylint-dev__pylint-7228` (pylint)
- `django__django-14915` (django)
- `django__django-13658` (django)
- `sympy__sympy-19487` (sympy)
- `pytest-dev__pytest-5227` (pytest)
- `django__django-13710` (django)
- `sphinx-doc__sphinx-7686` (sphinx)
- `django__django-11999` (django)
- `django__django-14382` (django)
- `sympy__sympy-22005` (sympy)
- `sympy__sympy-13146` (sympy)
- `django__django-13220` (django)
- `scikit-learn__scikit-learn-15535` (scikit-learn)
- `pydata__xarray-4094` (xarray)
- `pytest-dev__pytest-8906` (pytest)
- `sympy__sympy-21627` (sympy)
- `django__django-11039` (django)
- `django__django-17051` (django)
- `sympy__sympy-24102` (sympy)
- `django__django-12125` (django)
- `sympy__sympy-18532` (sympy)
- `sympy__sympy-11897` (sympy)
- `django__django-14238` (django)
- `sympy__sympy-22714` (sympy)
- `pytest-dev__pytest-11143` (pytest)
- `sympy__sympy-16106` (sympy)
- `django__django-16139` (django)
- `django__django-16229` (django)
- `django__django-13933` (django)
- `django__django-15902` (django)
- `matplotlib__matplotlib-23964` (matplotlib)
- `matplotlib__matplotlib-23913` (matplotlib)
- `mwaskom__seaborn-2848` (seaborn)
- `sphinx-doc__sphinx-8595` (sphinx)
- `psf__requests-863` (requests)
- `scikit-learn__scikit-learn-25747` (scikit-learn)
- `astropy__astropy-7746` (astropy)

## M2

Learning tasks, unchanged from round 1:

- `sympy__sympy-18621` (short)
- `django__django-15851` (short)
- `scikit-learn__scikit-learn-11040` (medium)
- `django__django-11019` (long)

Probes, 2 round-1 probes plus 4 new ones:

- `django__django-13590` (medium)
- `sympy__sympy-21055` (long)
- `django__django-13551` (django)
- `sympy__sympy-19254` (sympy)
- `django__django-15061` (django)
- `matplotlib__matplotlib-25433` (matplotlib)

Units: 4 learning tasks x 2 orders x persist/reset = 16. Probes: 6 tasks x learned/cold x 2 orders = 24. References: `g1-main`, `g1-strong`, and `g3` on the 10 learning+probe tasks = 30. G1-strong references are scheduled last.

Order: all M1, then M2 learning, then `g1-main` and `g3` references, then probes, then `g1-strong` references.

M2 imports whitelisted ALO statistics only. A parent `model.patch` is applied only for an M1 same-task B1 restore. A missing or invalid parent is `blocked` / `parent_missing` and the rest of that chain does not cold-start.

ProgramBench is not in the scored r2 schedule. Optional smoke: `kisielk__errcheck.dacab89`, `g1-main`, B0, 40 requests.

## Cost estimate

Round-1 mean CNY per request, at a full unit cap. Prior is zero. Not fit to round-1 scores. Expected: the first try holds, so every G2 call is `deepseek-flash`, G1 is pinned, and G3 is split evenly. Escalated: every G2 unit then spends its 3 reserved requests on `mimo-v2.6-pro`. That Pro total is 44.488 CNY, under 45, so the 24-task strong subset and the G1-strong M2 references stay. Worst case: each model receives every flexible call it may take (all of G3, and every G2 call except the planner and one first-try deepseek call). Those worst-case rows are not one combined bill. Pro's worst case is above 48 CNY; the soft halt stops the run between units.

| Model | Expected calls | Expected new | Escalated new | Worst-case calls | Worst-case new |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deepseek-flash` | 2024.0 | 18.216 | 15.840 | 3008.0 | 27.072 |
| `mimo-v2.6-flash` | 1640.0 | 8.200 | 8.200 | 4144.0 | 20.720 |
| `glm-5.3-flash` | 328.0 | 4.920 | 4.920 | 2832.0 | 42.480 |
| `mimo-v2.6-pro` | 1064.0 | 35.644 | 44.488 | 3568.0 | 119.528 |
| total expected (simultaneous) |  | 66.980 | 73.448 |  |  |

