# Working on ALO Cloud Experiments

Read `docs/experiment_plan.md`, `docs/protocol.md`, and
`docs/implementation_status.md` before changing research behavior.

This is a starting point for developing the M1 self-feedback and M2 cross-task
learning experiments in a GPT/Codex cloud environment. Read `study_records/README.md`
for the explicitly supplied task, exposure and private-storage metadata. Preserve
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
The user also authorized the administrative metadata allowlist under
`study_records/2026-10-28-m1-m2/`: task identities and ordered reserves, historical
selection states, exposure registration, proposed M2 orders, and location/hash
indexes. These records do not contain candidate answers, benchmark payloads,
private grading material, raw provider responses or the private budget ledger.
Treat them as controller-side planning metadata, not candidate prompt inputs.
Do not interpret a candidate catalog, historical QC label or scoped exposure
search as a complete formal admission or an executable task manifest.

Keep images, archives, datasets, benchmark assets, generated reports, audit
results, logs, model responses, ledgers, checkpoints, and credentials outside
this repository except for the specifically authorized administrative metadata
above. Do not copy historical run directories or Git histories into it. Location
indexes are pointers only: cloud access and contents migration are separately
unverified. Do not automatically follow them into credentials or private benchmark
data. Keep task text, answers and scoring assets out of these metadata files.

Run `python3 scripts/check_repository.py` before committing. This check catches
common unwanted files and secret shapes; it is not a complete secret detector.
Inspect the staged diff as well. Store new test or setup output in a temporary
directory outside the checkout.

The current task is environment and code preparation. Adding provider access,
benchmark assets, independent scoring, persistence, or real collection is a
separate implementation stage governed by the experiment plan.
