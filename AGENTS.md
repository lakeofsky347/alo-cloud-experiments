# Working on ALO Cloud Experiments

Read `docs/experiment_plan.md`, `docs/protocol.md`, and
`docs/implementation_status.md` before changing research behavior.

This is a source-only starting point for developing the M1 self-feedback and
M2 cross-task learning experiments in a GPT/Codex cloud environment. Preserve
the distinction between implemented software, scripted controls, actual
benchmark evaluation, and model evidence.

## Development workflow

- Use Python 3.12. Prepare dependencies with `bash scripts/setup_cloud.sh`.
- Follow the exact verification commands in `docs/cloud_environment.md`.
- Keep tests deterministic and offline. Do not require provider credentials,
  model requests, Docker services, benchmark downloads, or hidden graders for
  the default development checks.
- The initial repository does not implement the complete experiment runner.
  Mark missing integrations explicitly; do not substitute a toy multi-agent
  implementation and label it G2 ALO or G3 native GroupChat.
- Keep the planned task identities, reset/continuation rules, resource
  accounting, sequence order, and no-update probes consistent across changes.
- Never treat an offline test or cloud environment installation as a model
  score, formal acceptance, or authorization to start collection.

## Repository contents

Commit plans, contracts, source code, dependency declarations, and source tests.
Keep images, archives, datasets, benchmark assets, generated reports, audit
results, logs, model responses, ledgers, checkpoints, and credentials outside
this repository. Do not copy historical run directories or Git histories into
it. Do not read or search neighboring projects to recover credentials or
private benchmark data.

Run `python3 scripts/check_repository.py` before committing. This check catches
common unwanted files and secret shapes; it is not a complete secret detector.
Inspect the staged diff as well. Store new test or setup output in a temporary
directory outside the checkout.

The current task is environment and code preparation. Adding provider access,
benchmark assets, independent scoring, persistence, or real collection is a
separate implementation stage governed by the experiment plan.
