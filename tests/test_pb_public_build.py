"""Public builds exercise trusted local fixtures; no daemon, providers or scorer."""
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from alo_cloud.pb_build import run_public_build
from alo_cloud.pb_submission import capture_submission
from alo_cloud.sandbox import DockerSandbox


class LocalBuildFixture:
    """Run only this test's authored input, with the production driver and cwd."""
    def __init__(self):
        self.calls = []

    def __call__(self, command, timeout):
        self.calls.append((command, timeout))
        mounts = [word for word in command if word.startswith("type=bind,")]
        assert len(mounts) == 1
        mount = dict(part.split("=", 1) for part in mounts[0].split(","))
        assert mount["dst"] == "/workspace"
        source = Path(mount["src"])
        assert not (source / "executable").exists()
        driver = command[command.index("python3") + 1:]
        result = subprocess.run(
            [sys.executable, *driver], cwd=source, capture_output=True, text=True,
            env={"PATH": os.environ["PATH"], "HOME": str(source), "LC_ALL": "C"}, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def setup_build(tmp_path, files, runner=None):
    live = tmp_path / "live"
    live.mkdir()
    for name, contents in files.items():
        location = live / name
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_text(contents)
    snapshot = capture_submission(live, tmp_path / "sealed")
    box = DockerSandbox(live, image="programbench/fixture:task_cleanroom_v6",
                        runner=runner or LocalBuildFixture(), merge_image_workspace=True)
    return box, snapshot


@pytest.mark.skipif(shutil.which("cc") is None, reason="requires local C compiler")
def test_builds_and_exports_fresh_native_artifact_on_detached_readonly_input(tmp_path):
    files = {
        "compile.sh": "#!/bin/sh\nset -eu\ngit rev-parse --verify HEAD >/dev/null\n"
                      "test ! -e executable\ntest ! -f .pb-probe/debug.py\n"
                      "cc src/main.c -o executable\nprintf changed > src/main.c\n",
        "src/main.c": "int main(void) { return 13; }\n",
        "executable": "stale input executable",
        ".pb-probe/debug.py": "probe scratch",
    }
    box, snapshot = setup_build(tmp_path, files)
    (snapshot.root / "src/main.c").chmod(0o444)
    (snapshot.root / "src").chmod(0o555)
    (snapshot.root / "compile.sh").chmod(0o444)
    out = tmp_path / "exported"
    result = run_public_build(box, snapshot.root, snapshot.sha256, timeout_s=30, artifact_path=out)
    assert result["status"] == "pass", result
    assert result["artifact_present"] and result["artifact_executable"]
    assert result["returncode"] == 0 and result["submission_sha256"] == snapshot.sha256
    assert result["artifact_sha256"] == sha256(out.read_bytes()).hexdigest()
    assert subprocess.run([str(out)]).returncode == 13
    for root in (box.workspace, snapshot.root):
        assert (root / "src/main.c").read_text() == files["src/main.c"]
    assert (box.workspace / "executable").read_text() == files["executable"]
    assert not (snapshot.root / "executable").exists()
    assert not (snapshot.root / ".git").exists()
    assert len(box.runner.calls) == 1
    command, timeout = box.runner.calls[0]
    assert 0 < timeout <= 30
    assert "--pull=never" in command and "--network=none" in command
    assert "--cap-drop=ALL" in command and "--read-only" in command
    assert str(box.workspace) not in " ".join(command) and str(snapshot.root) not in " ".join(command)


@pytest.mark.parametrize(("script", "reason", "returncode"), [
    (None, "missing_compile_sh", None),
    ("cat src/missing.c > /dev/null", "missing_build_input", 1),
    ("printf 'not an expression' > invalid.c\ncc -x c invalid.c -o executable", "compile_failed", 1),
    ("true", "missing_artifact", 0),
    ("printf data > executable", "invalid_artifact", 0),
    ("touch executable && chmod +x executable", "invalid_artifact", 0),
    ("ln -s /bin/sh executable", "invalid_artifact", 0),
    ("mkdir executable", "invalid_artifact", 0),
])
def test_identifies_build_failures(tmp_path, script, reason, returncode):
    if script and "invalid.c" in script and shutil.which("cc") is None:
        pytest.skip("requires local C compiler")
    files = {"executable": "stale", "build/.alo-pb-candidate": "stale"}
    if script:
        files["compile.sh"] = "#!/bin/sh\n" + script + "\n"
    box, snapshot = setup_build(tmp_path, files)
    out = tmp_path / "out"
    result = run_public_build(box, snapshot.root, snapshot.sha256, artifact_path=out)
    assert (result["status"], result["reason"]) == ("fail", reason), result
    assert result["returncode"] == returncode
    assert result["artifact_sha256"] is None and not out.exists()
    affected = {"missing_compile_sh": "./compile.sh", "missing_build_input": "src/missing.c",
                "compile_failed": "invalid.c", "missing_artifact": "./executable", "invalid_artifact": "./executable"}
    assert affected[reason] in result["diagnostic"]


@pytest.mark.skipif(shutil.which("go") is None, reason="requires local Go")
def test_offline_module_lookup_is_a_diagnostic_build_failure(tmp_path):
    box, snapshot = setup_build(tmp_path, {
        "compile.sh": "#!/bin/sh\nGOTOOLCHAIN=local GOPROXY=off GOSUMDB=off "
                      "go mod download example.invalid/pkg@v1.0.0\n",
        "go.mod": "module buildfixture\ngo 1.18\n",
    })
    result = run_public_build(box, snapshot.root, snapshot.sha256, timeout_s=30)
    assert result["status"] == "fail" and result["reason"] == "offline_dependency", result
    assert "module lookup disabled" in result["diagnostic"]


@pytest.mark.parametrize("outcome", [
    {"returncode": 124, "timed_out": True, "stdout": "", "stderr": ""},
    {"returncode": 125, "stdout": "", "stderr": "Cannot connect to the Docker daemon"},
    {"returncode": 127, "stdout": "", "stderr": "image tools missing"},
    {"returncode": 0, "stdout": "malformed receipt", "stderr": ""},
])
def test_infrastructure_is_unknown(tmp_path, outcome):
    box, snapshot = setup_build(tmp_path, {"compile.sh": "true"}, runner=lambda *_: outcome)
    result = run_public_build(box, snapshot.root, snapshot.sha256)
    assert result["status"] == "unknown" and result["returncode"] is None
    assert result["artifact_sha256"] is None


def test_mismatch_deadline_and_export_under_live_tree_are_refused(tmp_path):
    box, snapshot = setup_build(tmp_path, {"compile.sh": "#!/bin/sh\nexit 0\n"})
    for expected, timeout, dest in [
        ("wrong", 30, None), (snapshot.sha256, 0, None), (snapshot.sha256, 30, box.workspace / "illegal"),
    ]:
        result = run_public_build(box, snapshot.root, expected, timeout_s=timeout, artifact_path=dest)
        assert result["status"] == "unknown" and result["artifact_sha256"] is None
    assert not box.runner.calls and not (box.workspace / "illegal").exists()


def test_differential_uses_identical_inputs_for_every_observation(tmp_path):
    from alo_cloud.pb_build import snapshot_differential

    box, snapshot = setup_build(tmp_path, {"compile.sh": "#!/bin/sh\n", "input": "initial"})
    artifact = tmp_path / "built"
    artifact.write_bytes(b"fresh")
    digest = sha256(artifact.read_bytes()).hexdigest()
    seen = []

    def probe(detached, argv, timeout):
        assert Path(detached, "input").read_text() == "initial"
        assert (detached / "executable").readlink() == Path("/workspace/executable")
        assert 0 < timeout <= 10
        seen.append((detached, argv))
        (detached / "input").unlink()
        (detached / "result").write_text(argv[1])
        return {"returncode": 0, "stdout": argv[1], "stderr": "", "timed_out": False}

    box.public_probe_command = probe
    assert snapshot_differential(box, snapshot, artifact, digest, [("x",), ("y",)], lambda: 10) == "pass"
    assert len({root for root, _ in seen}) == 4
    assert (snapshot.root / "input").read_text() == "initial"
    assert not (snapshot.root / "result").exists()

    def different(detached, argv, timeout):
        result = probe(detached, argv, timeout)
        if argv[0] != "./executable":
            (detached / "result").write_text("different")
        return result

    box.public_probe_command = different
    assert snapshot_differential(box, snapshot, artifact, digest, [("x",)], lambda: 10) == "fail"
    box.public_probe_command = lambda *args: {"timed_out": True, "returncode": 124}
    assert snapshot_differential(box, snapshot, artifact, digest, [], lambda: 10) == "unknown"
    box.public_probe_command = lambda *args: pytest.fail("probe should not run")
    assert snapshot_differential(box, snapshot, artifact, "changed", [], lambda: 10) == "unknown"
    assert snapshot_differential(box, snapshot, artifact, digest, [], lambda: 0) == "unknown"
    assert snapshot_differential(box, snapshot, artifact, digest, [(".pb-probe/input",)], lambda: 10) == "unknown"


@pytest.mark.skipif(not os.getenv("ALO_PB_LOCAL_FIXTURE_IMAGE"), reason="opt-in local generic Docker image")
@pytest.mark.parametrize("same_program", [True, False])
def test_services_public_signal_with_real_no_network_containers(tmp_path, same_program):
    """Use a synthetic reference on a generic local image, without a benchmark checkout."""
    from alo_cloud.fox_pb import PbServices, model_settings, resolve_pb_pool
    from alo_cloud.public_tools import dispatch_action
    from alo_cloud.sandbox import subprocess_docker_runner

    image = os.environ["ALO_PB_LOCAL_FIXTURE_IMAGE"]
    spec = resolve_pb_pool("fox/deepseek-flash")[0]
    svc = PbServices(tmp_path / "unit", (spec,), settings_by_model={spec.internal: model_settings(spec)},
                     transport=lambda *args: pytest.fail("no model requests"),
                     limits={"requests": 40, "tokens": 1000, "tools": 10, "seconds": 90},
                     differential_enabled=True)
    svc.pb_build_enabled = True
    script = '#!/bin/sh\nset -eu\ntest -f seed\nrm seed\nprintf "%s\\n" "$1" > output\nprintf "ok\\n"\n'
    ref = tmp_path / "synthetic-reference"
    ref.write_text(script)
    ref.chmod(0o555)
    w = svc.workspace.root
    (w / "seed").write_text("seed data")
    (w / "app.sh").write_text(script if same_program else script.replace('"ok\\n"', '"different\\n"'))
    (w / "compile.sh").write_text("#!/bin/sh\nset -eu\ncp app.sh executable\nchmod +x executable\n")
    calls = []

    def local_docker(command, timeout):
        cmd = list(command)
        assert "--pull=never" in cmd and "--network=none" in cmd
        if cmd[cmd.index("--workdir") + 1] == "/agent":
            # The generic test image receives only this invented reference.
            cmd[3:3] = ["--mount", f"type=bind,src={ref},dst=/workspace/executable,readonly"]
        calls.append(cmd)
        return subprocess_docker_runner(cmd, timeout)

    svc.sandbox = DockerSandbox(w, image=image, runner=local_docker, merge_image_workspace=True)
    svc.reference_invocations = [("first",), ("second",)]
    assert svc.public_signal() == ("pass" if same_program else "fail"), svc.last_pb_signal
    assert svc.last_pb_signal["compile"] == "pass"
    assert svc.last_pb_signal["differential"] == ("pass" if same_program else "fail")
    assert len(calls) == 5
    assert (w / "seed").read_text() == "seed data" and not (w / "output").exists()
    assert not (w / "build/.alo-pb-candidate").exists()
    # A reviewer builds the fixed handoff even after the live tree diverges.
    handoff = capture_submission(w, tmp_path / "handoff")
    (w / "compile.sh").write_text("#!/bin/sh\nexit 9\n")
    svc.current_node, svc.role_tools, svc.review_snapshot = "p2", ("read_file", "pb_build"), handoff.root
    receipt = dispatch_action(svc, {"tool": "pb_build", "arguments": {}})
    assert receipt["status"] == "pass" and receipt["submission_sha256"] == handoff.sha256
    svc.current_node, svc.role_tools = "p1", ("write_file", "pb_build")
    receipt = dispatch_action(svc, {"tool": "pb_build", "arguments": {}})
    assert receipt["status"] == "fail" and receipt["returncode"] == 9
    assert receipt["submission_sha256"] != handoff.sha256


def test_long_compiler_diagnostics_preserve_first_concrete_error_and_tail(tmp_path):
    # The first filename and error are outside both the output header and tail.
    script = """#!/bin/sh
i=0
while [ "$i" -lt 350 ]; do echo "note from include search $i"; i=$((i+1)); done
echo 'src/entry.c:117:9: error: unknown type name concrete_type'
i=0
while [ "$i" -lt 350 ]; do echo "context for subsequent failures $i"; i=$((i+1)); done
echo 'final build summary'
exit 2
"""
    box, snapshot = setup_build(tmp_path, {"compile.sh": script})
    result = run_public_build(box, snapshot.root, snapshot.sha256)
    assert result["status"] == "fail" and result["returncode"] == 2
    assert "src/entry.c:117:9: error: unknown type name concrete_type" in result["diagnostic"]
    assert "final build summary" in result["diagnostic"]
    assert len(result["diagnostic"]) <= 4096


def test_identical_docker_launch_errors_cannot_be_a_differential_match(tmp_path):
    from alo_cloud.pb_build import snapshot_differential
    def failed_start(command, timeout):
        return {"returncode": 126, "stdout": "", "stderr": "docker: invalid mount config"}

    box, snapshot = setup_build(tmp_path, {"compile.sh": "#!/bin/sh\n"}, runner=failed_start)
    candidate = tmp_path / "built"
    candidate.write_bytes(b"candidate")
    assert snapshot_differential(
        box, snapshot, candidate, sha256(candidate.read_bytes()).hexdigest(), [], lambda: 30) == "unknown"
