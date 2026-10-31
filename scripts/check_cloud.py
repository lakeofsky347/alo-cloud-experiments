#!/usr/bin/env python3
"""Check Python/software imports only; never contact Docker or a provider API."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE_PINS = {"PyYAML": "6.0.3", "pytest": "9.1.1"}
NATIVE_PINS = {"autogen-agentchat": "0.2.40", "docker": "7.2.0"}
CORE_IMPORTS = ("yaml", "pytest", "alo", "alo_cloud", "alo_cloud.contracts", "alo_cloud.state",
                "alo_cloud.budget", "alo_cloud.services", "alo_cloud.providers", "alo_cloud.sandbox")
# Preserve networking/trust configuration without reading/logging credential variables.
CHILD_ENV_NAMES = {
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TMP", "TEMP",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "PIP_CERT",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "SYSTEMROOT", "WINDIR",
}
IMPORT_PROBE = r'''
import importlib,json,socket,sys
attempts=[]
def blocked(*args,**kwargs):
    attempts.append("blocked_socket_operation")
    raise RuntimeError("Software import check blocked a socket operation")
socket.socket.connect=blocked
socket.socket.connect_ex=blocked
socket.socket.send=blocked
socket.socket.sendall=blocked
socket.socket.sendto=blocked
socket.create_connection=blocked
socket.getaddrinfo=blocked
sys.path.insert(0,sys.argv[2])
try:
    importlib.import_module(sys.argv[1])
except BaseException as error:
    print(json.dumps({"imported":False,"error_type":type(error).__name__,"socket_attempts":len(attempts)}))
    raise SystemExit(1)
print(json.dumps({"imported":True,"socket_attempts":len(attempts)}))
raise SystemExit(1 if attempts else 0)
'''


def check(*, with_native: bool = False, require_linux: bool = False) -> dict:
    pins = BASE_PINS | (NATIVE_PINS if with_native else {})
    versions = {}
    for name, expected in pins.items():
        try:
            observed = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed = None
        versions[name] = {"expected": expected, "observed": observed, "matches": observed == expected}
    env = {name: os.environ[name] for name in CHILD_ENV_NAMES if name in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    probes = []
    for module in CORE_IMPORTS + (("autogen", "docker") if with_native else ()):
        result = {"module": module, "imported": False, "socket_attempts": None}
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", IMPORT_PROBE, module, str(ROOT / "src")],
                                  cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            result["returncode"] = proc.returncode
            lines = proc.stdout.splitlines()
            if lines:
                try:
                    fields = json.loads(lines[-1])
                    result.update({k: fields[k] for k in ("imported", "socket_attempts", "error_type") if k in fields})
                except (ValueError, TypeError):
                    result["error_type"] = "UnparseableImportResult"
            if proc.returncode != 0:
                # Never copy arbitrary import output or environment values into the report.
                result.setdefault("error_type", "ProcessExit")
        except subprocess.TimeoutExpired:
            result.update(returncode=None, error_type="ImportTimeout")
        probes.append(result)
    python_ok = sys.version_info[:2] == (3, 12)
    linux_ok = platform.system() == "Linux"
    passed = (python_ok and (linux_ok or not require_linux)
              and all(v["matches"] for v in versions.values())
              and all(p["returncode"] == 0 and p["imported"] and p["socket_attempts"] == 0 for p in probes))
    return {
        "schema": "alo-cloud-software-check-v1", "status": "software_check_passed" if passed else "software_check_failed",
        "passed": passed, "python": platform.python_version(), "python_3_12": python_ok,
        "system": platform.system(), "architecture": platform.machine(), "linux_required": require_linux, "is_linux": linux_ok,
        "native_framework_checks_requested": with_native, "packages": versions, "import_probes": probes,
        "proxy_and_ca_settings_preserved": True,
        "guard_scope": "Import-only subprocesses with a Python socket guard; not an OS sandbox or a test-suite result",
        "provider_api_calls": 0, "docker_daemon_checked": False, "benchmark_images_checked": False,
        "cloud_environment_verified": False, "formal_experiment_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-native", action="store_true", help="Also import the optional AutoGen and Docker Python libraries; no daemon access")
    parser.add_argument("--require-linux", action="store_true", help="Fail the software check on a non-Linux host")
    parser.add_argument("--output", type=Path, help="Optional new JSON file outside this repository")
    args = parser.parse_args()
    if args.output:
        target = args.output.expanduser().resolve()
        if target == ROOT or target.is_relative_to(ROOT) or target.exists():
            parser.error("output must be a new file outside the repository")
    result = check(with_native=args.with_native, require_linux=args.require_linux)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            stream.write(text)
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
