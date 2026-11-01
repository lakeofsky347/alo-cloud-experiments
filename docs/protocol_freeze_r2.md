# Round-2 paid-run freeze

Date: 2026-11-01. Batch identity `r2`. This schedule is not comparable with round 1 and must not be pooled with r1 results, scores, or ledgers.

freeze_sha256: 3e6696d68dcd6b98932ba65cd867adda07c4721cb5c820e9fa94464da73c5f38

The round-1 freeze stays in `protocol_freeze.md`. `python -m alo_cloud.r2 --check` recomputes this hash. The draw seed is `r2-task-draw-20261101-v1`. SWE-bench Lite revision `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`. The draw is largest-remainder by repository, then SHA-256 rank. It does not use round-1 scores.

## Caps and budget

M1 is B0 only (no B1-self). SWE B0 is 24 requests. M2 learning and reference are 16. Probes are 12. `command_seconds` for these runs is 120. DeepSeek receives a thinking budget of 8192 tokens when the r2 runner builds the request. Every model still has a 50 CNY hard cap. The four-model total is 200 CNY. `mimo-v2.6-pro` also has a configurable soft halt at 48 CNY cumulative, applied between units (`hard: false`).

Model request name stays `deepseek-flash`. Endpoints stay the mainland-China pay-as-you-go URLs.

## M1

48 SWE-bench Lite tasks: the 6 round-1 M1 tasks, then 42 more. Arms `g1-main`, `g2`, and `g3` run on all 48. `g1-strong` (`mimo-v2.6-pro`) runs on this 24-task subset, which includes the 6 round-1 tasks:

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

Units: 4 learning tasks x 2 orders x persist/reset = 16. Probes: 6 tasks x learned/cold x 2 orders = 24. References: `g1-main`, `g1-strong`, and `g3` on the 10 learning+probe tasks = 30.

Order: all M1, then M2 learning, then `g1-main` and `g3` references, then probes, then `g1-strong` references. The strong references are last so the pro soft halt cuts them first.

M2 imports whitelisted ALO statistics only. A parent `model.patch` is applied only for an M1 same-task B1 restore. A missing or invalid parent is `blocked` / `parent_missing` and the rest of that chain does not cold-start.

ProgramBench is not in the scored r2 schedule. Optional smoke: `kisielk__errcheck.dacab89`, `g1-main`, B0, 40 requests.

## Cost estimate

Round-1 mean CNY per request, not a score and not an input to the draw: deepseek-flash 0.009, mimo-v2.6-flash 0.005, glm-5.3-flash 0.015, mimo-v2.6-pro 0.0335. G1 calls are pinned. G2 counts one planner request on deepseek-flash and splits the rest evenly. G3 splits each cap evenly across the four roles.

- `deepseek-flash`: 7.362 CNY (818 requests)
- `mimo-v2.6-flash`: 10.210 CNY (2042 requests)
- `glm-5.3-flash`: 10.950 CNY (730 requests)
- `mimo-v2.6-pro`: 49.111 CNY (1466 requests)
- total: 77.633 CNY

The even-split `mimo-v2.6-pro` figure is about 49.1 CNY, above the 48 CNY soft halt and under the 50 CNY hard cap. The ten G1-strong reference units are 160 requests (about 5.4 CNY) and run last.

