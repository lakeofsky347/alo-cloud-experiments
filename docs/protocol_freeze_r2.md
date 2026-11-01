# Round-2 paid-run freeze

Date: 2026-11-01. Batch identity `r2`. This schedule is not comparable with round 1 and must not be pooled with r1 results, scores, or ledgers. The CNY caps are cumulative with round 1: the r2 ledger is created with round 1's settled spend as its opening commitment.

freeze_sha256: 8c12f686b8de97c48668071a8c44e8ae8f24efe39f32670eb97f554683fa6fe3

The round-1 freeze stays in `protocol_freeze.md`. `python -m alo_cloud.r2 --check` recomputes this hash. The draw seed is `r2-task-draw-20261101-v1`. SWE-bench Lite revision `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`. The draw is largest-remainder by repository, then SHA-256 rank. It does not use round-1 scores.

## Caps and budget

M1 is B0 only (no B1-self). SWE B0 is 24 requests. M2 learning and reference are 16. Probes are 12. `command_seconds` for these runs is 120. DeepSeek receives a thinking budget of 8192 tokens when the r2 runner builds the request. Every model still has a 50 CNY hard cap. The four-model total is 200 CNY. Both caps include round-1 settled spend. `mimo-v2.6-pro` soft-halts at 48 CNY cumulative, between units (`hard: false`). That halt also includes the carry-in.

Frozen prior spend, milli-CNY, from the round-1 ledger: `deepseek-flash` 4.619, `mimo-v2.6-flash` 3.732, `glm-5.3-flash` 2.524, `mimo-v2.6-pro` 16.955. Those four sum to 27.830 CNY. The unrounded operator total is 27.830317 CNY; the 317 microcny residual is below one milli-CNY and is not assigned to a model. `ALO_PRIOR_LEDGER` may point at the round-1 `budget.sqlite3`. That file is read only. Settled totals replace the frozen map when they are higher, and never when they are lower. An unsettled row refuses the run.

Model request name stays `deepseek-flash`. Endpoints stay the mainland-China pay-as-you-go URLs.

## M1

48 SWE-bench Lite tasks: the 6 round-1 M1 tasks, then 42 more. Arms `g1-main`, `g2`, and `g3` run on all 48. `g1-strong` (`mimo-v2.6-pro`) runs on this 12-task subset, which includes the 6 round-1 tasks. The 6 extra strong tasks are a seeded draw (salt `g1-strong`, count 6):

- `scikit-learn__scikit-learn-13584`
- `django__django-11583`
- `sympy__sympy-11870`
- `django__django-16408`
- `sympy__sympy-20322`
- `sympy__sympy-18835`
- `sympy__sympy-22005`
- `django__django-13933`
- `matplotlib__matplotlib-23913`
- `pytest-dev__pytest-5227`
- `sympy__sympy-22714`
- `django__django-14155`

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

Units: 4 learning tasks x 2 orders x persist/reset = 16. Probes: 6 tasks x learned/cold x 2 orders = 24. References: `g1-main` and `g3` on the 10 learning+probe tasks = 20. G1-strong M2 references are not in the schedule.

Order: all M1, then M2 learning, then `g1-main` and `g3` references, then probes.

In each G2 and G3 unit, `mimo-v2.6-pro` may make at most `floor(request_cap / 5)` billed calls. A further Pro selection is refused before the provider call and does not spend CNY. G1-strong is not under that cap: every one of its calls is Pro. G1-main does not call Pro.

M2 imports whitelisted ALO statistics only. A parent `model.patch` is applied only for an M1 same-task B1 restore. A missing or invalid parent is `blocked` / `parent_missing` and the rest of that chain does not cold-start.

ProgramBench is not in the scored r2 schedule. Optional smoke: `kisielk__errcheck.dacab89`, `g1-main`, B0, 40 requests.

## Cost estimate

Round-1 mean CNY per request, at a full unit cap, plus the carry-in. Not fit to round-1 scores. Expected routing: G1 pinned; one G2 planner call on `deepseek-flash`; other G2 calls and all G3 calls split evenly and then clipped so Pro stays within `floor(cap / 5)`, with the refused calls split across the other three models. Worst case for a model: that model receives every flexible call it is allowed to receive. Pro's worst case is the G1-strong caps plus the floor(cap/5) allowance. The worst-case column is not a single bill; those rows cannot happen together.

Pro's new headroom before the 48 CNY cumulative halt is 31.045 CNY. The full-cap Pro total below is 26.733 CNY, leaving 4.312 CNY of margin.

| Model | Prior | Expected calls | Expected new | Expected cumulative | Worst-case calls | Worst-case new | Worst-case cumulative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-flash` | 4.619 | 891.3 | 8.022 | 12.641 | 2498.0 | 22.482 | 27.101 |
| `mimo-v2.6-flash` | 3.732 | 2115.3 | 10.577 | 14.309 | 3722.0 | 18.610 | 22.342 |
| `glm-5.3-flash` | 2.524 | 803.3 | 12.050 | 14.574 | 2410.0 | 36.150 | 38.674 |
| `mimo-v2.6-pro` | 16.955 | 798.0 | 26.733 | 43.688 | 798.0 | 26.733 | 43.688 |
| total (expected is simultaneous; worst case is not) | 27.830 |  | 57.382 | 85.212 |  |  |  |

