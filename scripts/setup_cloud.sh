#!/usr/bin/env bash
# Install this repository's development dependencies into one explicit venv.
# No sudo, system configuration edits, Docker operations or provider API calls.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_venv="$task_root/.venv"
task_python=""
task_evidence=""
task_native=0
usage() {
  echo 'Usage: setup_cloud.sh [--venv DIR] [--python PYTHON_3_12] [--with-native] [--evidence-dir NEW_EXTERNAL_DIR]'
}
while (($#)); do
  case "$1" in
    --venv|--python|--evidence-dir)
      (($# >= 2)) || { usage >&2; exit 2; }
      case "$1" in
        --venv) task_venv="$2" ;;
        --python) task_python="$2" ;;
        --evidence-dir) task_evidence="$2" ;;
      esac
      shift 2 ;;
    --with-native) task_native=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if [[ -z "$task_python" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then task_python=python3.12; else task_python=python3; fi
fi
"$task_python" - "$task_venv" "$task_root" <<'PY'
from pathlib import Path
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Python 3.12 is required. Select the environment runtime or pass --python; this script does not change system Python.')
venv=Path(sys.argv[1]).expanduser().absolute();root=Path(sys.argv[2]).resolve()
if venv.is_symlink() or (venv.exists() and not (venv/'pyvenv.cfg').is_file()):
    raise SystemExit('Refusing to use a symlink or an existing non-venv directory.')
if venv == root or root.is_relative_to(venv):
    raise SystemExit('The virtual environment must not contain the repository.')
PY
if [[ -z "$task_evidence" ]]; then
  task_evidence="$(mktemp -d "/tmp/alo-cloud-setup.XXXXXXXX")"
else
  "$task_python" - "$task_evidence" "$task_root" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]).expanduser().resolve();root=Path(sys.argv[2]).resolve()
if p == root or p.is_relative_to(root) or p.exists():
    raise SystemExit('Evidence output must be a new directory outside the repository.')
p.mkdir(parents=True,exist_ok=False)
PY
fi
task_venv="$("$task_python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().absolute())' "$task_venv")"
if [[ ! -e "$task_venv" ]]; then "$task_python" -m venv "$task_venv"; fi
task_python="$task_venv/bin/python"
"$task_python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Existing venv is not Python 3.12"'
# Inherit the platform's configured proxy/CA/package-manager settings. Do not
# unset them, add trusted-host, disable TLS checks or rewrite global pip config.
export PYTHONDONTWRITEBYTECODE=1
"$task_python" -m pip --disable-pip-version-check --no-input install \
  --requirement "$task_root/requirements-cloud.txt"
task_project="$task_root"
if ((task_native)); then task_project="$task_root[native]"; fi
"$task_python" -m pip --disable-pip-version-check --no-input install --editable "$task_project"
"$task_python" -m pip --disable-pip-version-check check > "$task_evidence/pip-check.txt"
if ((task_native)); then
  "$task_python" "$task_root/scripts/check_cloud.py" --with-native \
    --output "$task_evidence/software-check.json"
else
  "$task_python" "$task_root/scripts/check_cloud.py" \
    --output "$task_evidence/software-check.json"
fi
echo "Development dependencies installed; software check: $task_evidence/software-check.json"
echo "No Docker daemon, benchmark image, provider API, cloud account or formal experiment was checked or started."
