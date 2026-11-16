"""Offline ProgramBench transfer arms. No network and no Docker."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from alo.runtime.decomposer import parse_dag
from alo_cloud.adapters import (
    ROLE_CONTRACT_ENFORCED, ROLE_CONTRACT_UNRESTRICTED_UNKNOWN, resolve_role_tool_contract,
    role_tool_contract,
)
from alo_cloud.errors import ProtocolError
from alo_cloud.evaluate import PROGRAMBENCH_COMMIT
from alo_cloud.fox_pb import (
    BEST_SINGLE,
    DEFAULT_REQUEST_CAP,
    DEFAULT_UNIT_SECONDS,
    G0_TASKS,
    PB_POOL,
    PbServices,
    PbStopped,
    decision_freeze,
    import_swe_profile,
    open_pb_transfer,
    pb_eval_plan,
    report_runs,
    resolve_pb_pool,
    run_arm,
    run_g0,
    run_official_pb_eval,
)
from alo_cloud.fox_pool import FOX_RESOURCE_PRIOR, model_settings
from alo_cloud.pb_contract import (
    DEFAULT_ROLE_MAP,
    PROBE_SCRIPT_SENTENCE,
    blackbox_probe_argv,
    combine_pb_signal,
    compare_blackbox,
    pb_command_violation,
    pb_dag,
    reference_is_execute_only,
    run_blackbox_differential,
    signal_components_to_record,
    submission_reference_scan,
)
from alo_cloud.public_tools import TOOLS, dispatch_action
from alo_cloud.r2 import r2_freeze_sha256
from alo_cloud.state import StateContractError


FROZEN_R2_SHA256 = "77bddfbfd3472ae4c0d280c3dacf20810f11b5f9b5c247c17650629ebf9e241f"
G0_IDS = {task_id for task_id, _stratum in G0_TASKS}


def _quality_row(topics):
    return {
        "public_quality": 1.0, "quality_before": 0.0, "quality_after": 1.0,
        "delta_quality": 1.0, "credited_quality": 1.0, "topics": list(topics),
        "requests": 2, "tokens": 10, "seconds": 1.0, "cost_cny": 0.01,
        "local_contract_status": "pass",
    }


def _profile(profiles):
    return {"schema": "alo-cross-track-transfer-v1", "profiles": profiles}


def test_r2_freeze_hash_is_unchanged():
    assert r2_freeze_sha256() == FROZEN_R2_SHA256


def test_role_contracts_follow_the_shared_alias_map():
    """``public_tools`` maps to the public set. Any still-unknown name is unrestricted."""
    placeholder = resolve_role_tool_contract(type("Role", (), {"tools": ("public_tools",)})())
    assert placeholder["mode"] == ROLE_CONTRACT_ENFORCED
    assert placeholder["mapped"] == TOOLS
    assert placeholder["restricts"] is False
    aliased = role_tool_contract(type("Role", (), {"tools": ("run_command",)})())
    assert aliased == ("command",)
    unknown = resolve_role_tool_contract(type("Role", (), {"tools": ("command", "not_a_public_tool")})())
    assert unknown["mode"] == ROLE_CONTRACT_UNRESTRICTED_UNKNOWN
    assert unknown["mapped"] is None
    services = type("S", (), {})()
    services.role_tools = unknown["mapped"]
    services.write_file = lambda **kwargs: {"wrote": kwargs["path"]}
    assert dispatch_action(services, {"tool": "write_file", "arguments": {
        "path": "a.py", "content": "x"}})["wrote"] == "a.py"
    services.role_tools = aliased
    with pytest.raises(ProtocolError, match="tool_outside_role_contract"):
        dispatch_action(services, {"tool": "write_file", "arguments": {
            "path": "a.py", "content": "x"}})


def test_pb_dag_defaults_and_real_tool_names():
    text = pb_dag()
    nodes = {node.id: node for node in parse_dag(text)}
    assert list(nodes) == ["p1", "p2"]
    assert nodes["p2"].deps == ("p1",)
    assert set(nodes["p1"].topics) == {"coding", "implementation"}
    assert set(nodes["p2"].topics) == {"testing", "validation"}
    assert PROBE_SCRIPT_SENTENCE in nodes["p1"].goal
    assert PROBE_SCRIPT_SENTENCE not in nodes["p2"].goal
    for node in nodes.values():
        assert node.role.tools
        assert set(node.role.tools) <= set(TOOLS)
    assert "objdump" in text and "strace" in text and "network" in text
    owned = {node.id: node for node in parse_dag(pb_dag(probe_script_owner="p2"))}
    assert PROBE_SCRIPT_SENTENCE in owned["p2"].goal
    assert "write_file" in owned["p2"].role.tools
    swapped = {node.id: node for node in parse_dag(pb_dag(role_map={"s1": "p2", "s2": "p1"}))}
    assert set(swapped["p1"].topics) == {"testing", "validation"}
    assert set(swapped["p2"].topics) == {"coding", "implementation"}
    assert normalize_map() == DEFAULT_ROLE_MAP


def normalize_map():
    from alo_cloud.pb_contract import normalize_role_map
    return normalize_role_map("s1=p1,s2=p2")


def test_public_signal_defaults_off_and_ignores_unknown():
    compile_only = combine_pb_signal(0, differential_enabled=False, differential_signal="fail")
    assert compile_only["signal"] == "pass"
    assert compile_only["differential"] is None
    assert signal_components_to_record(compile_only) == [("compile", "pass")]
    unknown = combine_pb_signal(None, differential_enabled=False)
    assert unknown["signal"] == "unknown"
    assert signal_components_to_record(unknown) == []
    both_unknown = combine_pb_signal(None, differential_enabled=True, differential_signal="unknown")
    assert both_unknown["signal"] == "unknown"
    assert signal_components_to_record(both_unknown) == []
    compile_known = combine_pb_signal(0, differential_enabled=True, differential_signal="unknown")
    assert compile_known["signal"] == "pass"
    assert signal_components_to_record(compile_known) == [("compile", "pass")]
    mismatch = combine_pb_signal(0, differential_enabled=True, differential_signal="fail")
    assert mismatch["signal"] == "fail"
    assert signal_components_to_record(mismatch) == [("compile", "pass"), ("differential", "fail")]
    assert combine_pb_signal(1)["signal"] == "fail"


def test_differential_is_black_box_io_only():
    seen = []

    def runner(argv):
        seen.append(list(argv))
        assert pb_command_violation(argv) is None
        if argv[0] == "./executable":
            return {"returncode": 0, "stdout": "ok\n", "stderr": "", "fs_before": "a", "fs_after": "a"}
        return {"returncode": 0, "stdout": "no\n", "stderr": "", "fs_before": "a", "fs_after": "b"}

    reference = blackbox_probe_argv("./executable", ["--check"])
    candidate = blackbox_probe_argv("./build/executable", ["--check"])
    assert run_blackbox_differential(reference, candidate, runner=runner) == "fail"
    assert seen == [["./executable", "--check"], ["./build/executable", "--check"]]
    assert "objdump" not in json.dumps(seen)
    same = compare_blackbox(
        {"returncode": 2, "stdout": "x", "stderr": "y", "fs_before": "a", "fs_after": "b"},
        {"returncode": 2, "stdout": "x", "stderr": "y", "fs_before": "a", "fs_after": "b"},
    )
    assert same == "pass"
    assert compare_blackbox({"returncode": 0, "stdout": "", "stderr": ""}, None) == "unknown"
    with pytest.raises(ProtocolError):
        blackbox_probe_argv("objdump", ["-d", "./executable"])


def test_command_filter_blocks_inspection_instrumentation_and_network(tmp_path):
    assert pb_command_violation(["./executable", "--version"]) is None
    assert pb_command_violation(["./executable", "https://example.test/input"]) is None
    assert pb_command_violation(["./build/executable", "arg"]) is None
    assert pb_command_violation(["./compile.sh"]) is None
    assert pb_command_violation(["objdump", "-d", "./build/executable"]) is None
    assert pb_command_violation(["objdump", "-d", "./executable"]) == "pb_reference_inspection_forbidden"
    assert pb_command_violation(["strings", "executable"]) == "pb_reference_inspection_forbidden"
    assert pb_command_violation(["hexdump", "-C", "/workspace/executable"]) == "pb_reference_inspection_forbidden"
    assert pb_command_violation(["ghidra", "./executable"]) == "pb_reference_inspection_forbidden"
    assert pb_command_violation(["strace", "./executable"]) == "pb_reference_instrumentation_forbidden"
    assert pb_command_violation(["ltrace", "-e", "open", "./executable"]) == "pb_reference_instrumentation_forbidden"
    assert pb_command_violation(["sh", "-c", "objdump -d ./executable"]) == "pb_reference_inspection_forbidden"
    assert pb_command_violation(["curl", "https://example.test"]) == "pb_network_forbidden"
    assert pb_command_violation(["wget", "http://example.test/a"]) == "pb_network_forbidden"
    assert pb_command_violation(["git", "clone", "https://example.test/repo"]) == "pb_network_forbidden"
    assert pb_command_violation(
        ["python3", "-c", "print(open('executable','rb').read())"]) == "pb_reference_binary_read_forbidden"
    assert pb_command_violation(["cp", "./executable", "build/wrapped"]) == "pb_reference_copy_forbidden"
    assert pb_command_violation(["cp", "build/app", "./executable"]) is None
    services = _services(tmp_path)
    with pytest.raises(ProtocolError, match="pb_reference_inspection_forbidden"):
        services.command(["objdump", "-d", "./executable"])
    with pytest.raises(ProtocolError, match="pb_network_forbidden"):
        services.command(["curl", "https://example.test"])
    with pytest.raises(ProtocolError, match="reference_binary_is_execute_only"):
        services.read_file("executable")
    with pytest.raises(ProtocolError, match="reference_binary_is_execute_only"):
        services.write_file("executable", "nope")


def _services(tmp_path):
    spec = resolve_pb_pool("fox/deepseek-flash")[0]
    return PbServices(
        tmp_path / "guard",
        (spec,),
        settings_by_model={spec.internal: model_settings(spec)},
        transport=lambda *args, **kwargs: ({}, []),
        limits={"requests": 40, "tokens": 1000, "tools": 10, "seconds": 2400},
    )


def test_pool_rejects_qwen_and_omni_and_keeps_the_four():
    assert tuple(row.internal for row in resolve_pb_pool("all")) == PB_POOL
    assert "qwen" not in " ".join(PB_POOL)
    with pytest.raises(PbStopped, match="omni_model_rejected"):
        resolve_pb_pool(["qwen3.8-omni-flash"])
    with pytest.raises(PbStopped, match="pb_pool_excludes_qwen"):
        resolve_pb_pool(["fox/qwen3.8-flash"])
    with pytest.raises(PbStopped, match="invalid_pb_request_cap"):
        run_arm(Path("/tmp/not-created-pb"), arm="A2", tasks="kisielk__errcheck",
                request_cap=32, live=False)


def test_submission_scan_flags_wrap_copy_and_hash(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    reference = root / "executable"
    raw = b"reference-bytes"
    reference.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    reference.chmod(0o111)
    assert reference_is_execute_only(reference.stat().st_mode)
    (root / "compile.sh").write_text(
        "#!/bin/sh\ngcc -o ./build/executable main.c\ncp build/executable ./executable\n",
        encoding="utf-8")
    (root / "build").mkdir()
    (root / "build" / "executable").write_bytes(b"built")
    clean = submission_reference_scan(root, reference_sha256=digest)
    assert clean["ok"] is True
    assert clean["reference_execute_only"] is True
    stolen = root / "wrapped"
    stolen.write_bytes(b"reference-bytes")
    copied = submission_reference_scan(root, reference_sha256=digest)
    assert copied["ok"] is False
    assert any(item["kind"] == "embedded_reference_bytes" for item in copied["violations"])
    stolen.unlink()
    (root / "note.py").write_text(f"blob = '{digest}'\n", encoding="utf-8")
    hashed = submission_reference_scan(root, reference_sha256=digest)
    assert any(item["kind"] == "embedded_reference_hash" for item in hashed["violations"])
    (root / "note.py").write_text('import subprocess\nsubprocess.run(["./executable"])\n', encoding="utf-8")
    invoked = submission_reference_scan(root, reference_sha256=digest)
    assert invoked["ok"] is True
    assert any(item["kind"] == "unused_probe_invokes_reference" and item["path"] == "note.py"
               for item in invoked["warnings"])
    assert "note.py" in invoked["unused_probes"]
    (root / "note.py").unlink()
    (root / "shim.sh").write_text("#!/bin/sh\n./executable \"$@\"\n", encoding="utf-8")
    direct = submission_reference_scan(root, reference_sha256=digest)
    assert direct["ok"] is True
    assert any(item["kind"] == "unused_probe_invokes_reference" and item["path"] == "shim.sh"
               for item in direct["warnings"])
    (root / "shim.sh").unlink()
    (root / "steal.sh").write_text("cp ./executable ./build/wrapped\n", encoding="utf-8")
    assert any(item["kind"] == "copies_reference" for item in submission_reference_scan(
        root, reference_sha256=digest)["violations"])
    (root / "steal.sh").unlink()
    (root / "compile.sh").write_text(
        "#!/bin/sh\ncp build/executable executable 2>/dev/null || true\n",
        encoding="utf-8")
    installed = submission_reference_scan(root, reference_sha256=digest)
    assert installed["ok"] is True
    assert installed["violations"] == []
    probe_dir = root / ".pb-probe"
    probe_dir.mkdir()
    (probe_dir / "run.sh").write_text("#!/bin/sh\n./executable --help\n", encoding="utf-8")
    excluded = submission_reference_scan(root, reference_sha256=digest)
    assert excluded["ok"] is True
    outside = tmp_path / "behaviour_probe.sh"
    outside.write_text("#!/bin/sh\n./executable --version\n", encoding="utf-8")
    external = submission_reference_scan(root, reference_sha256=digest, probe_paths=[outside])
    assert external["ok"] is True
    (probe_dir / "run.sh").write_text("#!/bin/sh\ncat ./executable\n", encoding="utf-8")
    read_probe = submission_reference_scan(root, reference_sha256=digest)
    assert any(item["kind"] == "reads_reference_bytes" for item in read_probe["violations"])
    (probe_dir / "run.sh").unlink()
    (root / "wrap.py").write_text('open("executable", "rb").read()\n', encoding="utf-8")
    assert any(item["kind"] == "reads_reference_bytes" for item in submission_reference_scan(
        root, reference_sha256=digest)["violations"])
    (root / "main.c").write_text('#include <stdlib.h>\nint main(void){return system("./executable");}\n', encoding="utf-8")
    (root / "compile.sh").write_text("#!/bin/sh\ngcc -o executable main.c\n", encoding="utf-8")
    wrapped = submission_reference_scan(root, reference_sha256=digest)
    assert wrapped["ok"] is False
    assert any(item["kind"] == "invokes_reference" and item["path"] == "main.c" for item in wrapped["violations"])
    (root / "compile.sh").write_text("#!/bin/sh\nstrace -o /tmp/out ./executable\n", encoding="utf-8")
    traced = submission_reference_scan(root, reference_sha256=digest)
    assert any(item["kind"] == "instruments_reference" and item["path"] == "compile.sh"
               for item in traced["violations"])
    (root / "compile.sh").write_text("#!/bin/sh\ncp ./executable ./stolen\n", encoding="utf-8")
    copied_ref = submission_reference_scan(root, reference_sha256=digest)
    assert any(item["kind"] == "copies_reference" and item["path"] == "compile.sh"
               for item in copied_ref["violations"])
    (root / "wrap.py").unlink()
    (root / "probe.py").write_text("#!/usr/bin/env python3\nimport subprocess\nsubprocess.run(['./executable'])\n",
                                   encoding="utf-8")
    (root / "compile.sh").write_text("#!/bin/sh\nprintf ok\n", encoding="utf-8")
    (root / "main.c").unlink()
    root_probe = submission_reference_scan(root, reference_sha256=digest)
    assert root_probe["ok"] is True
    assert any(item["path"] == "probe.py" for item in root_probe["warnings"])
    (root / "peek.py").write_text(
        'import subprocess\nsubprocess.run(["objdump", "-d", "./executable"])\n',
        encoding="utf-8")
    peeked = submission_reference_scan(root, reference_sha256=digest)
    assert peeked["ok"] is False
    assert any(item["kind"] == "inspects_reference" and item["path"] == "peek.py"
               for item in peeked["violations"])
    (root / "peek.py").unlink()
    reference.chmod(0o755)
    assert submission_reference_scan(root, reference_sha256=digest)["reference_execute_only"] is False


def test_import_is_read_only_and_maps_model_ids(tmp_path):
    source = tmp_path / "transfer_package.json"
    source.write_text(json.dumps(_profile({
        "deepseek-flash": [_quality_row(["coding", "implementation"])],
        "fox/qwen3.8-flash": [_quality_row(["testing", "validation"])],
        "qwen3.8-omni-flash": [_quality_row(["coding", "implementation"])],
    })), encoding="utf-8")
    before = source.read_bytes()
    with open_pb_transfer(source, allowed_models=PB_POOL, strip_quality=False) as library:
        with pytest.raises(StateContractError, match="frozen_transfer_writeback_forbidden"):
            library.write_back({"observations": {}})
        observations = json.loads(json.dumps(library.observations))
    assert source.read_bytes() == before
    assert "fox/deepseek-flash" in observations
    assert observations["fox/deepseek-flash"]
    assert "fox/qwen3.8-flash" not in observations
    assert "qwen3.8-omni-flash" not in observations
    stripped = import_swe_profile(source, allowed_models=PB_POOL, strip_quality=True)
    row = stripped["observations"]["fox/deepseek-flash"][0]
    assert "credited_quality" not in row
    assert "quality_after" not in row
    assert row["requests"] == 2
    assert source.read_bytes() == before
    with pytest.raises(PbStopped, match="omni_model_rejected"):
        import_swe_profile(
            source, allowed_models=PB_POOL,
            model_map={"deepseek-flash": "fox/qwen3.8-omni-flash"})


def test_g0_dry_run_writes_six_units_on_the_allowed_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    result = run_g0(tmp_path / "g0", live=False)
    assert result["units"] == 6
    assert result["paid_calls"] == 0
    rows = []
    for arm in ("A2", "A4"):
        manifest = json.loads((tmp_path / "g0" / arm / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["request_cap"] == DEFAULT_REQUEST_CAP
        assert manifest["request_cap_candidates"] == [40, 64, 80]
        assert manifest["online_update"] is False
        assert manifest["frozen"] is True
        assert manifest["differential_signal"] is False
        assert manifest["differential_kind"] == "blackbox_io"
        assert manifest["probe_script_owner"] == "p1"
        assert manifest["role_map"] == DEFAULT_ROLE_MAP
        assert manifest["unit_seconds"] == DEFAULT_UNIT_SECONDS
        assert manifest["read_timeout_s"] == 600
        assert manifest["live"] is False
        assert manifest["api_key_recorded"] is False
        assert manifest["pb_freeze"]["frozen"] is True
        assert manifest["pb_freeze"]["differential_signal"] is False
        assert manifest["reference_mode"] == "111"
        assert set(manifest["models"]) == set(PB_POOL)
        assert "qwen" not in " ".join(manifest["models"])
        assert all("omni" not in model for model in manifest["models"])
        assert manifest["node_request_cap"] == 20
        freeze_settings = {
            "role_map": manifest["role_map"],
            "probe_script_owner": manifest["probe_script_owner"],
            "differential_signal": manifest["differential_signal"],
            "request_cap": manifest["request_cap"],
            "online_update": manifest["online_update"],
            "unit_seconds": manifest["unit_seconds"],
            "read_timeout_s": manifest["read_timeout_s"],
        }
        if manifest.get("node_request_cap") is not None:
            freeze_settings["node_request_cap"] = manifest["node_request_cap"]
        freeze, digest = decision_freeze(freeze_settings)
        assert manifest["pb_freeze_sha256"] == digest
        assert freeze["schema"] == "fox-pb-decision-freeze-v1"
        arm_rows = [
            json.loads(line)
            for line in (tmp_path / "g0" / arm / "records.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert {row["task_id"] for row in arm_rows} == G0_IDS
        assert len(arm_rows) == 3
        for row in arm_rows:
            assert row["status"] in {"completed", "failed", "stopped", "blocked"}
            assert set(row["models_used"]) <= set(PB_POOL)
            assert all("omni" not in model for model in row["models_used"])
            assert "qwen" not in " ".join(row["models_used"])
            if arm == "A4":
                assert row["models_used"] == [BEST_SINGLE]
                assert row["first_assignment"] == BEST_SINGLE
        if arm == "A2":
            state = json.loads(next((tmp_path / "g0" / "A2" / "checkpoints").glob("*/adapter.json")).read_text())
            contracts = state["role_tool_contracts"]
            assert {item["node_id"] for item in contracts} >= {"p1", "p2"}
            assert {item["mode"] for item in contracts} == {"enforced"}
            for item in contracts:
                assert set(item["mapped"]) <= set(TOOLS)
                assert item["unknown"] == []
        rows.extend(arm_rows)
    assert len(rows) == 6
    again = run_g0(tmp_path / "g0", live=False)
    assert again["wrote"] == 0
    assert r2_freeze_sha256() == FROZEN_R2_SHA256


def test_online_update_is_opt_in_and_unknown_is_not_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    run_arm(
        tmp_path / "online", arm="A2", tasks="kisielk__errcheck", live=False,
        online_update=True, signal_override="unknown", models="fox/deepseek-flash,fox/glm-5.3-flash")
    manifest = json.loads((tmp_path / "online" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["online_update"] is True
    assert manifest["frozen"] is False
    assert manifest["models"] == ["fox/deepseek-flash", "fox/glm-5.3-flash"]
    checkpoint = next((tmp_path / "online" / "checkpoints").glob("*/adapter.json"))
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state.get("observations") in ({}, None) or all(not rows for rows in state.get("observations", {}).values())
    run_arm(
        tmp_path / "learned", arm="A2", tasks="kisielk__errcheck", live=False,
        online_update=True, signal_override="pass", models="fox/deepseek-flash")
    learned = json.loads(next((tmp_path / "learned" / "checkpoints").glob("*/adapter.json")).read_text(encoding="utf-8"))
    assert any(learned.get("observations", {}).values())


def test_scoring_chain_records_harness_fields_without_docker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))

    def scorer(plan):
        assert plan["image_family"] == "task_cleanroom_v6"
        assert plan["source_commit"].startswith("895f659")
        assert plan["build_exec"] == ["./compile.sh"]
        assert plan["reference_mode"] == "111"
        return {"status": "finished", "resolved": True, "pass_rate": 0.5, "compile_success": True}

    run_arm(
        tmp_path / "score", arm="A4", tasks="raviqqe__muffet", live=False,
        execute=True, scorer=scorer)
    row = json.loads((tmp_path / "score" / "records.jsonl").read_text(encoding="utf-8"))
    assert row["harness_resolved"] is True
    assert row["pass_rate"] == 0.5
    assert row["compile_success"] is True
    assert row["harness_status"] == "finished"
    assert isinstance(row["wall_clock_s"], float)
    plan = pb_eval_plan("raviqqe__muffet.a882908", tmp_path)
    assert plan["eval_command"] == [] or "programbench" in " ".join(plan["eval_command"])


def test_task_yaml_is_public_and_omits_hidden_tests(tmp_path):
    task_id = "kisielk__errcheck.dacab89"
    root = tmp_path / "ProgramBench" / "src" / "programbench" / "data" / "tasks" / task_id
    root.mkdir(parents=True)
    (root / "task.yaml").write_text(
        "repository: kisielk/errcheck\ncommit: dacab891ef4a1c38ecf6c4d94fd66746bb1247d5\n"
        "language: go\ndifficulty: easy\n"
        "eval_clean_hashes:\n- abcdef\n",
        encoding="utf-8")
    (root / "tests.json").write_text('{"secret":"hidden"}', encoding="utf-8")
    from alo_cloud.fox_pb import load_pb_public_payload
    payload = load_pb_public_payload(task_id, repo=tmp_path / "ProgramBench")
    assert payload["repository"] == "kisielk/errcheck"
    assert payload["language"] == "go"
    assert payload["reference_executable"] == "./executable"
    assert payload["reference_mode"] == "111"
    assert "eval_clean_hashes" not in payload
    assert "secret" not in json.dumps(payload)
    assert "hidden" not in json.dumps(payload)


def test_public_signal_runs_compile_and_optional_blackbox_probe(tmp_path):
    services = _services(tmp_path)
    assert services.public_signal() == "unknown"
    root = services.workspace.root
    (root / "build").mkdir()
    (root / "executable").write_bytes(b"reference")
    seen = []

    class Sandbox:
        def command(self, argv, timeout_s):
            seen.append(list(argv))
            text = "same" if argv[0] == "./executable" else "same"
            return {"returncode": 0, "stdout": text, "stderr": "", "timed_out": False}

    (root / "compile.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    services.sandbox = Sandbox()
    services.differential_enabled = False
    services.reference_invocations = [("input",)]
    assert services.public_signal() == "pass"
    assert [argv[0] for argv in seen] == ["chmod", "./compile.sh"]
    assert services.differential_enabled is False
    seen.clear()
    services.differential_enabled = True
    (root / "build").mkdir(exist_ok=True)
    (root / "build" / ".alo-pb-candidate").write_bytes(b"candidate")
    assert services.public_signal() == "pass"
    assert ["./executable", "input"] in seen
    assert ["./build/.alo-pb-candidate", "input"] in seen
    assert not (root / "build" / ".alo-pb-candidate").exists()
    assert all(pb_command_violation(argv) is None for argv in seen)
    seen.clear()

    class Mismatch:
        def command(self, argv, timeout_s):
            seen.append(list(argv))
            stdout = "ref" if argv[0] == "./executable" else "candidate"
            return {"returncode": 0, "stdout": stdout, "stderr": "", "timed_out": False}

    services.sandbox = Mismatch()
    services.last_pb_signal = None
    (root / "build" / ".alo-pb-candidate").write_bytes(b"candidate")
    assert services.public_signal() == "fail"
    assert services.last_pb_signal["differential"] == "fail"
    assert services.last_pb_signal["compile"] == "pass"


def test_live_unit_loads_the_public_payload_and_attaches_a_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    data = tmp_path / "public" / "ProgramBench"
    data.mkdir(parents=True)
    task_id = "kisielk__errcheck.dacab89"
    (data / f"{task_id}.json").write_text(json.dumps({
        "task_id": task_id,
        "problem_statement": "Rebuild errcheck from the bundled docs.",
    }), encoding="utf-8")

    def seeder(argv):
        if argv[:2] == ["docker", "cp"]:
            dest = Path(argv[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "README.md").write_text("task docs", encoding="utf-8")
            binary = dest / "executable"
            binary.write_bytes(b"reference")
            binary.chmod(0o111)
        return subprocess.CompletedProcess(argv, 0, "", "")

    from alo_cloud.fox_learn import dry_gateway
    run_arm(
        tmp_path / "live", arm="A4", tasks="kisielk__errcheck", live=True,
        sender=dry_gateway, data_dir=tmp_path / "public", image_seeder=seeder)
    row = json.loads((tmp_path / "live" / "records.jsonl").read_text(encoding="utf-8"))
    assert row["status"] != "blocked"
    assert row["reason"] != "public_task_payload_missing"
    assert row["sandbox_image"].startswith("programbench/")
    assert row["sandbox_workdir"] == "/workspace"
    assert row["sandbox_attached"] is True
    assert row["sandbox_network"] == "none"
    assert row["sandbox_network_flag"] is True
    assert row["reference_executable"] == "./executable"
    assert row["reference_mode"] == "111"
    assert row["preserve_executable_in_image"] is True
    workspace = next((tmp_path / "live").glob("units/*/state/workspace"))
    assert (workspace / "README.md").read_text(encoding="utf-8") == "task docs"
    assert "README.md" in row["public_docs"]
    assert not (workspace / "executable").exists()


def test_execute_uses_the_official_evaluator(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    run_arm(tmp_path / "noscore", arm="A4", tasks="esubaalew__run", live=False, execute=True)
    missing = json.loads((tmp_path / "noscore" / "records.jsonl").read_text(encoding="utf-8"))
    assert missing["harness_status"] == "unavailable"
    assert missing["harness_reason"] == "programbench_checkout_required"
    assert missing["harness_resolved"] is None
    workspace = tmp_path / "submission"
    workspace.mkdir()
    (workspace / "compile.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    plan = pb_eval_plan(
        "esubaalew__run.0fb9dec", workspace, repo=tmp_path / "ProgramBench",
        unit_id="unit-1", run_dir=tmp_path / "eval-run")
    assert plan["source_commit"].startswith("895f659")
    assert plan["network"] == "none"
    assert plan["build_in_cleanroom"] == "chmod +x ./compile.sh && ./compile.sh"
    assert plan["image"].startswith("programbench/")
    assert plan["eval_command"]
    seen = {}

    def harness(argv, *, cwd, env):
        seen["argv"] = list(argv)
        seen["env"] = env
        assert "FOXROUTER_API_KEY" not in env
        sealed = Path(plan["sealed_dir"]) / plan["task_id"]
        sealed.mkdir(parents=True, exist_ok=True)
        (sealed / f"{plan['task_id']}.eval.json").write_text(json.dumps({
            "test_results": [{"status": "passed", "name": "hidden-name"}, {"status": "failure", "name": "other"}],
            "error_code": None,
        }), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    scored = run_official_pb_eval(
        plan, harness_run=harness, git_head=lambda repo: PROGRAMBENCH_COMMIT)
    assert scored["pass_rate"] == 0.5
    assert scored["resolved"] is False
    assert scored["compile_success"] is True
    assert scored["status"] == "finished"
    assert "programbench" in seen["argv"]
    assert "eval" in seen["argv"]
    assert "hidden-name" not in json.dumps(scored)
    import inspect
    from alo_cloud.fox_pb import main as pb_main
    assert "official_pb_scorer" in inspect.getsource(pb_main)
    assert (tmp_path / "eval-run" / "sealed" / "programbench" / "unit-1"
            / "esubaalew__run.0fb9dec" / "submission.tar.gz").is_file()
    report = report_runs([tmp_path / "score"]) if False else None
    del report


def test_report_reads_harness_resolved_from_run_records(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))

    def scorer(plan):
        return {"status": "finished", "resolved": True, "pass_rate": 1.0, "compile_success": True}

    run_arm(
        tmp_path / "scored", arm="A4", tasks="raviqqe__muffet", live=False,
        execute=True, scorer=scorer)
    raw = json.loads((tmp_path / "scored" / "records.jsonl").read_text(encoding="utf-8"))
    raw.pop("resolved", None)
    from alo_cloud.comparison import pb_paired_report
    report = pb_paired_report([raw])
    assert report["comparison"]["oracle"]["tasks"] == 1
    assert report["comparison"]["oracle"]["resolved"] == 1
    task = "raviqqe__muffet.a882908"
    assert report["per_task"][task]["A4"]["resolved"] is True


def test_paired_report_covers_the_pb_contrasts():
    def row(arm, task, resolved, cost, compile_success, assignment, score):
        return {
            "arm": arm, "task_id": task, "resolved": resolved, "cost": cost,
            "compile_success": compile_success, "first_assignment": assignment,
            "pass_rate": score,
        }
    tasks = ["t1", "t2", "t3"]
    rows = []
    for task, a1, a2 in zip(tasks, ["fox/deepseek-flash", "fox/glm-5.3-flash", "fox/deepseek-flash"],
                            ["fox/kimi-k3.1-pro", "fox/glm-5.3-flash", "fox/mimo-v2.6-flash"]):
        rows.append(row("A1", task, True, 1.0, True, a1, 1.0))
        rows.append(row("A2", task, False, 2.0, True, a2, 0.0))
        rows.append(row("A3", task, False, 1.5, False, a2, 0.0))
        rows.append(row("A4", task, True, 0.5, True, BEST_SINGLE, 1.0))
        rows.append(row("A5", task, False, 0.7, True, BEST_SINGLE, 0.0))
    report = report_runs_from_rows(rows)
    assert set(report["pairs"]) == {"A1-A2", "A1-A3", "A1-A4", "A5-A4"}
    assert report["decision_difference"] == {"compared": 3, "differ": 2}
    assert report["decision_difference_rate"] == pytest.approx(2 / 3)
    assert report["arms"]["A4"]["compile_rate"] == 1
    assert report["arms"]["A2"]["cost_per_unit"] == 2
    assert report["arms"]["A4"]["variance"]["cost"]["n"] == 3
    assert report["arms"]["A4"]["variance"]["score"]["sd"] == 0
    assert report["comparison"]["best_single_model"]["condition"] == "A4"
    assert report["pairs"]["A1-A2"]["comparison"]["only_left"] == 3


def report_runs_from_rows(rows):
    from alo_cloud.comparison import pb_paired_report
    return pb_paired_report(rows)


def test_cli_dry_run_g0_and_rejects_a_changed_cap(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    from alo_cloud.fox_pb import main
    assert main(["g0", "--root", str(tmp_path / "cli"), "--dry-run"]) == 0
    assert "paid_calls" in capsys.readouterr().out
    assert main([
        "run", "--arm", "A2", "--g0", "--run-dir", str(tmp_path / "cli" / "A2"),
        "--live", "--request-cap", "64",
    ]) == 2
    assert "fox_run_settings_mismatch" in capsys.readouterr().out
    assert FOX_RESOURCE_PRIOR["cny"] == 0.0


def test_contract_matches_the_official_root_executable():
    from alo_cloud.programbench_prompt import PROGRAMBENCH_SUBMISSION_CONTRACT
    assert "produces an executable ./executable in the workspace root" in PROGRAMBENCH_SUBMISSION_CONTRACT
    assert "compile.sh must create ./executable at the workspace root" in PROGRAMBENCH_SUBMISSION_CONTRACT
    assert "cp build/executable executable" in PROGRAMBENCH_SUBMISSION_CONTRACT
    assert "./build is the writable target" not in PROGRAMBENCH_SUBMISSION_CONTRACT
    assert "write the program there" not in PROGRAMBENCH_SUBMISSION_CONTRACT
    dag = pb_dag()
    assert "produces ./executable" in dag
    assert "builds under ./build/" not in dag
    assert ".pb-probe/" in dag


def test_name_conflict_is_not_a_daemon_outage(tmp_path):
    import subprocess
    from alo_cloud.sandbox import (
        DockerSandbox, SandboxUnknown, container_name_conflict, run_docker_cli,
    )
    clash = (
        'docker: Error response from daemon: Conflict. The container name "/alo" '
        "is already in use by container \"abc\". You have to remove (or rename) "
        "that container to be able to reuse that name.\n"
    )

    def conflict(argv, timeout_s):
        del argv, timeout_s
        return {"returncode": 125, "stdout": "", "stderr": clash, "timed_out": False}

    box = DockerSandbox(
        tmp_path, image="example.invalid/alo:test", runner=conflict,
        container_name="alo-fox-pb-a2-abc123")
    result = box.command(["true"], 5)
    assert container_name_conflict(result) or result["failure"] == "container_name_conflict"
    assert result["returncode"] == 125
    assert result["failure"] == "container_name_conflict"

    def daemon(argv, timeout_s):
        del argv, timeout_s
        return {"returncode": 125, "stdout": "", "stderr": "Cannot connect to the Docker daemon",
                "timed_out": False}

    box.runner = daemon
    with pytest.raises(SandboxUnknown, match="docker_daemon_unavailable"):
        box.command(["true"], 5)

    removed = []

    class Proc:
        def __init__(self, argv, **kwargs):
            del kwargs
            self.argv = argv
            self.expired = False

        def communicate(self, timeout=None):
            del timeout
            if not self.expired:
                self.expired = True
                raise subprocess.TimeoutExpired(self.argv, 1)
            return b"", b"killed"

        def kill(self):
            self.killed = True

    argv = box.docker_argv(["sleep", "30"])
    name = argv[argv.index("--name") + 1]
    outcome = run_docker_cli(argv, 1, popen=lambda *args, **kwargs: Proc(*args, **kwargs),
                             remove=removed.append)
    assert outcome["timed_out"] is True
    assert outcome["returncode"] == 124
    assert removed == [name]


def test_official_compile_moves_the_reference_then_builds_root_executable(tmp_path):
    from alo_cloud.sandbox import OFFICIAL_COMPILE_SCRIPT, DockerSandbox
    script = OFFICIAL_COMPILE_SCRIPT
    assert script.index("mv /workspace/executable") < script.index("./compile.sh")
    assert "chmod 111 /opt/programbench-reference/executable" in script
    assert "cp -f /workspace/executable /workspace/build/.alo-pb-candidate" in script
    assert "cat /workspace/executable" not in script
    assert "objdump" not in script
    seen = {}

    def runner(argv, timeout_s):
        del timeout_s
        seen["argv"] = argv
        build = tmp_path / "build"
        build.mkdir(exist_ok=True)
        (build / ".alo-pb-candidate").write_bytes(b"built")
        return {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False}

    box = DockerSandbox(
        tmp_path, image="example.invalid/alo:test", runner=runner, merge_image_workspace=True)
    assert box.official_compile(30) == 0
    rendered = "\n".join(seen["argv"])
    assert "--network=none" in seen["argv"]
    assert "--read-only" not in seen["argv"]
    assert seen["argv"][seen["argv"].index("--user") + 1] == "0:0"
    assert "mv /workspace/executable /opt/programbench-reference/executable" in rendered
    assert "./compile.sh" in rendered
    assert ".alo-pb-candidate" in rendered
    assert (tmp_path / "build" / ".alo-pb-candidate").read_bytes() == b"built"


DEFAULT_PB_DECISION_SHA256 = "5a0f54fa1a88e42d43369c8f6fe8c293fbde6deb94c5d23789f7d69c1bd0c13c"
# G0 rerun decisions before the per-node cap: differential on, cap 40,
# s1->p1 / s2->p2, probe owner p1. This is the document PR #44 recorded.
G0_DECISION_SHA256_BEFORE_NODE_CAP = "a8d5c4949bc3b70db51fa4c8b8ad20b8eedb41aeda4ef023e31a8a4584b9deef"
G0_M18_DECISION_SHA256 = "c8f34bed293ac4bb8f2f1e7bcbd80d2efb559d90c2bf043202788dc941fbb434"


def _decision_settings(**extra):
    settings = {
        "role_map": {"s1": "p1", "s2": "p2"},
        "probe_script_owner": "p1",
        "differential_signal": False,
        "request_cap": 40,
        "online_update": False,
        "unit_seconds": 2400,
        "read_timeout_s": 600,
    }
    settings.update(extra)
    return settings


def test_default_decision_freeze_stays_and_g0_hash_gains_only_the_node_cap():
    body, digest = decision_freeze(_decision_settings())
    assert "node_request_cap" not in body
    assert digest == DEFAULT_PB_DECISION_SHA256
    before_body, before = decision_freeze(_decision_settings(differential_signal=True))
    assert "node_request_cap" not in before_body
    assert before == G0_DECISION_SHA256_BEFORE_NODE_CAP
    after_body, after = decision_freeze(_decision_settings(
        differential_signal=True, node_request_cap=20))
    assert set(after_body) - set(before_body) == {"node_request_cap"}
    for key, value in before_body.items():
        assert after_body[key] == value
    assert after_body["node_request_cap"] == 20
    assert after == G0_M18_DECISION_SHA256
    assert after != before
    assert r2_freeze_sha256() == FROZEN_R2_SHA256


def _fox_reply(body, text, *, status=200, total_tokens=12):
    model = body.get("model") if isinstance(body, dict) else None
    payload = {
        "model": model,
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": total_tokens},
    }
    return status, json.dumps(payload).encode("utf-8")


def _role_of(body):
    parts = []
    messages = body.get("messages") if isinstance(body, dict) else None
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
    text = "\n".join(parts)
    if text.rfind("Role Contract: p2") > text.rfind("Role Contract: p1"):
        return "p2"
    return "p1"


def _assistant_text(body):
    parts = []
    messages = body.get("messages") if isinstance(body, dict) else None
    if isinstance(messages, list):
        for message in messages:
            if (isinstance(message, dict) and message.get("role") == "assistant"
                    and isinstance(message.get("content"), str)):
                parts.append(message["content"])
    return "\n".join(parts)


def _no_network(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))


def test_root_probe_warns_and_is_scored_and_left_out_of_the_pack(tmp_path, monkeypatch):
    _no_network(monkeypatch)

    def sender(url, body, headers, timeout_s):
        del url, headers, timeout_s
        written = _assistant_text(body)
        if _role_of(body) == "p2":
            text = json.dumps({"final": "checked"})
        elif "probe.py" not in written:
            text = json.dumps({"tool": "write_file", "arguments": {
                "path": "probe.py",
                "content": "import subprocess\nsubprocess.run(['./executable'])\n",
            }})
        elif '"compile.sh"' not in written:
            text = json.dumps({"tool": "write_file", "arguments": {
                "path": "compile.sh",
                "content": "#!/bin/sh\nprintf ok\n",
            }})
        else:
            text = json.dumps({"final": "submitted"})
        return _fox_reply(body, text)

    def scorer(plan):
        return {"status": "finished", "resolved": False, "pass_rate": 0.4, "compile_success": True}

    run_arm(
        tmp_path / "probe", arm="A2", tasks="kisielk__errcheck", live=False,
        sender=sender, execute=True, scorer=scorer, signal_override="unknown",
        models="fox/deepseek-flash,fox/glm-5.3-flash")
    row = json.loads((tmp_path / "probe" / "records.jsonl").read_text(encoding="utf-8"))
    assert row["submission_ok"] is True
    assert row["pass_rate"] == 0.4
    assert row["score"] == 0.4
    assert row["harness_status"] == "finished"
    assert any(item["path"] == "probe.py" and item["kind"] == "unused_probe_invokes_reference"
               for item in row["submission_warnings"])
    workspace = next((tmp_path / "probe").glob("units/*/state/workspace"))
    assert (workspace / "probe.py").is_file()
    assert (workspace / "compile.sh").is_file()
    from alo_cloud.collect import _pack_submission
    import tarfile
    archive = tmp_path / "submission.tar.gz"
    _pack_submission(workspace, archive)
    names = tarfile.open(archive, "r:gz").getnames()
    assert "compile.sh" in names
    assert "probe.py" not in names
    assert all(not name.startswith(".pb-probe") for name in names)


def test_build_used_reference_wrap_is_rejected_with_score_zero(tmp_path, monkeypatch):
    _no_network(monkeypatch)

    def sender(url, body, headers, timeout_s):
        del url, headers, timeout_s
        written = _assistant_text(body)
        if _role_of(body) == "p2":
            text = json.dumps({"final": "checked"})
        elif "main.c" not in written:
            text = json.dumps({"tool": "write_file", "arguments": {
                "path": "main.c",
                "content": '#include <stdlib.h>\nint main(void){return system("./executable");}\n',
            }})
        elif '"compile.sh"' not in written:
            text = json.dumps({"tool": "write_file", "arguments": {
                "path": "compile.sh",
                "content": "#!/bin/sh\ngcc -o executable main.c\n",
            }})
        else:
            text = json.dumps({"final": "submitted"})
        return _fox_reply(body, text)

    def scorer(plan):
        return {"status": "finished", "resolved": True, "pass_rate": 0.8, "compile_success": True}

    run_arm(
        tmp_path / "reject", arm="A2", tasks="kisielk__errcheck", live=False,
        sender=sender, execute=True, scorer=scorer, signal_override="unknown",
        models="fox/deepseek-flash,fox/glm-5.3-flash")
    lines = (tmp_path / "reject" / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["submission_ok"] is False
    assert row["pass_rate"] == 0.0
    assert row["score"] == 0
    assert row["resolved"] is False
    assert row["harness_resolved"] is False
    assert row["harness_status"] == "submission_rejected"
    assert "invokes_reference" in row["harness_reason"]
    assert "main.c" in row["harness_reason"]
    assert row["reason"]
    note = row["official_score_annotation"]
    assert note["annotation"] == "offline_official_score_not_the_unit_score"
    assert note["unit_score"] == 0
    assert note["pass_rate"] == 0.8
    assert note["resolved"] is True
    assert row["pass_rate"] != note["pass_rate"]


def test_stopped_unit_records_the_checkpoint_assignment(tmp_path, monkeypatch):
    _no_network(monkeypatch)

    def sender(url, body, headers, timeout_s):
        del url, headers, timeout_s
        if _role_of(body) == "p2":
            return 500, b'{"error":{"message":"gateway down"}}'
        return _fox_reply(body, json.dumps({"final": "p1 submitted"}))

    run_arm(
        tmp_path / "stop", arm="A2", tasks="raviqqe__muffet", live=False,
        sender=sender, signal_override="unknown",
        models="fox/deepseek-flash,fox/glm-5.3-flash")
    row = json.loads((tmp_path / "stop" / "records.jsonl").read_text(encoding="utf-8"))
    assert row["status"] == "stopped"
    assert row["reason"] == "provider_error"
    checkpoint = json.loads(next((tmp_path / "stop" / "checkpoints").glob("*/adapter.json")).read_text())
    saved = {
        role: value["model_id"]
        for role, value in checkpoint["assignments"].items()
        if isinstance(value, dict) and isinstance(value.get("model_id"), str)
    }
    recorded = {item["role_id"]: item["model_id"] for item in row["node_assignments"]}
    assert saved
    assert "p1" in saved
    assert recorded == saved
    assert row["first_assignment"] == saved["p1"]
    assert row["first_assignment"] in {"fox/deepseek-flash", "fox/glm-5.3-flash"}


def test_node_request_cap_hands_p1_over_to_p2(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    counts = {"p1": 0, "p2": 0}

    def sender(url, body, headers, timeout_s):
        del url, headers, timeout_s
        role = _role_of(body)
        counts[role] += 1
        return _fox_reply(body, json.dumps({"tool": "list_files", "arguments": {}}))

    run_arm(
        tmp_path / "cap", arm="A2", tasks="esubaalew__run", live=False,
        sender=sender, node_request_cap=20, signal_override="unknown",
        models="fox/deepseek-flash,fox/glm-5.3-flash")
    row = json.loads((tmp_path / "cap" / "records.jsonl").read_text(encoding="utf-8"))
    assert counts == {"p1": 20, "p2": 20}
    assert row["status"] != "stopped"
    assert row["node_request_cap"] == 20
    assert row["request_cap"] == 40
    assert {item["node_id"] for item in row["node_assignments"]} == {"p1", "p2"}
    manifest = json.loads((tmp_path / "cap" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["node_request_cap"] == 20
    assert manifest["pb_freeze"]["node_request_cap"] == 20


def test_total_request_cap_still_bounds_both_nodes(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    counts = {"p1": 0, "p2": 0}

    def sender(url, body, headers, timeout_s):
        del url, headers, timeout_s
        role = _role_of(body)
        counts[role] += 1
        return _fox_reply(body, json.dumps({"tool": "list_files", "arguments": {}}))

    run_arm(
        tmp_path / "bound", arm="A2", tasks="esubaalew__run", live=False,
        sender=sender, node_request_cap=30, signal_override="unknown",
        models="fox/deepseek-flash,fox/glm-5.3-flash")
    row = json.loads((tmp_path / "bound" / "records.jsonl").read_text(encoding="utf-8"))
    assert counts["p1"] == 30
    assert counts["p2"] == 10
    assert row["status"] == "stopped"
    assert row["reason"] == "turn_cap"
    assert row["first_assignment"]
    assert {item["node_id"] for item in row["node_assignments"]} >= {"p1", "p2"}


def test_compile_glob_keeps_a_wrapped_source_in_the_build_and_the_archive(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "compile.sh").write_text("#!/bin/sh\ngcc *.c -o executable\n", encoding="utf-8")
    (root / "main.c").write_text(
        '#include <stdlib.h>\nint main(void){return system("./executable");}\n',
        encoding="utf-8")
    (root / "probe.py").write_text(
        "import subprocess\nsubprocess.run(['./executable'])\n", encoding="utf-8")
    scanned = submission_reference_scan(root)
    assert scanned["ok"] is False
    assert any(item["kind"] == "invokes_reference" and item["path"] == "main.c"
               for item in scanned["violations"])
    assert "main.c" not in scanned["unused_probes"]
    assert "probe.py" in scanned["unused_probes"]
    from alo_cloud.collect import _pack_submission
    import tarfile
    archive = tmp_path / "submission.tar.gz"
    _pack_submission(root, archive)
    names = tarfile.open(archive, "r:gz").getnames()
    assert "compile.sh" in names
    assert "main.c" in names
    assert "probe.py" not in names
    (root / "compile.sh").write_text("#!/bin/sh\ngcc $SRCS -o executable\n", encoding="utf-8")
    uncertain = submission_reference_scan(root)
    assert uncertain["ok"] is False
    assert any(item["path"] == "main.c" for item in uncertain["violations"])
    assert "main.c" not in uncertain["unused_probes"]
    assert "probe.py" not in uncertain["unused_probes"]


def test_compile_cd_keeps_the_source_it_builds(tmp_path):
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "compile.sh").write_text(
        "#!/bin/sh\ncd src\ngcc main.c -o ../executable\n", encoding="utf-8")
    (root / "src" / "main.c").write_text(
        '#include <stdlib.h>\nint main(void){return system("./executable");}\n',
        encoding="utf-8")
    (root / "probe.py").write_text(
        "import subprocess\nsubprocess.run(['./executable'])\n", encoding="utf-8")
    scanned = submission_reference_scan(root)
    assert scanned["ok"] is False
    assert any(item["kind"] == "invokes_reference" and item["path"] == "src/main.c"
               for item in scanned["violations"])
    assert "src/main.c" not in scanned["unused_probes"]
    assert "probe.py" in scanned["unused_probes"]
    from alo_cloud.collect import _pack_submission
    import tarfile
    archive = tmp_path / "submission.tar.gz"
    _pack_submission(root, archive)
    names = tarfile.open(archive, "r:gz").getnames()
    assert "compile.sh" in names
    assert "src/main.c" in names
    assert "probe.py" not in names
    (root / "compile.sh").write_text(
        "#!/bin/sh\npushd src >/dev/null\ngcc main.c -o ../executable\npopd\n",
        encoding="utf-8")
    pushed = submission_reference_scan(root)
    assert pushed["ok"] is False
    assert any(item["path"] == "src/main.c" for item in pushed["violations"])
    assert "probe.py" in pushed["unused_probes"]
    (root / "compile.sh").write_text(
        "#!/bin/sh\n(cd src && gcc main.c -o ../executable)\n", encoding="utf-8")
    nested = submission_reference_scan(root)
    assert nested["ok"] is False
    assert any(item["path"] == "src/main.c" for item in nested["violations"])
    assert "probe.py" in nested["unused_probes"]
    (root / "compile.sh").write_text(
        "#!/bin/sh\ncd \"$SRC\"\ngcc main.c -o ../executable\n", encoding="utf-8")
    uncertain = submission_reference_scan(root)
    assert uncertain["ok"] is False
    assert any(item["path"] == "src/main.c" for item in uncertain["violations"])
    assert "src/main.c" not in uncertain["unused_probes"]
    assert "probe.py" not in uncertain["unused_probes"]
    _pack_submission(root, archive)
    kept = tarfile.open(archive, "r:gz").getnames()
    assert "src/main.c" in kept
    assert "probe.py" in kept


def test_c_if_does_not_reject_a_legal_probe(tmp_path):
    root = tmp_path / "workspace"
    (root / ".pb-probe").mkdir(parents=True)
    (root / "compile.sh").write_text("#!/bin/sh\ngcc main.c -o executable\n", encoding="utf-8")
    (root / "main.c").write_text(
        "int main(int argc, char **argv){if (argc > 1) return 1; return 0;}\n",
        encoding="utf-8")
    (root / ".pb-probe" / "probe.py").write_text(
        "import subprocess\nsubprocess.run(['./executable'])\n", encoding="utf-8")
    scanned = submission_reference_scan(root)
    assert scanned["ok"] is True
    assert scanned["violations"] == []
    assert any(item["path"] == ".pb-probe/probe.py" for item in scanned["warnings"])


def test_parent_include_is_part_of_the_build(tmp_path):
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "compile.sh").write_text("#!/bin/sh\ngcc -o executable src/main.c\n", encoding="utf-8")
    (root / "src" / "main.c").write_text('#include "../helper.h"\nint main(void){return helper();}\n',
                                        encoding="utf-8")
    (root / "helper.h").write_text(
        '#include <stdlib.h>\nint helper(void){return system("./executable");}\n',
        encoding="utf-8")
    scanned = submission_reference_scan(root)
    assert scanned["ok"] is False
    assert any(item["kind"] == "invokes_reference" and item["path"] == "helper.h"
               for item in scanned["violations"])
    assert "helper.h" not in scanned["unused_probes"]
    from alo_cloud.collect import _pack_submission
    import tarfile
    archive = tmp_path / "submission.tar.gz"
    _pack_submission(root, archive)
    names = tarfile.open(archive, "r:gz").getnames()
    assert "compile.sh" in names
    assert "src/main.c" in names
    assert "helper.h" in names


def test_evaluator_exception_leaves_the_score_unknown(tmp_path, monkeypatch):
    _no_network(monkeypatch)

    def sender(url, body, headers, timeout_s):
        del url, headers, timeout_s
        written = _assistant_text(body)
        if _role_of(body) == "p2" or '"compile.sh"' in written:
            return _fox_reply(body, json.dumps({"final": "submitted"}))
        return _fox_reply(body, json.dumps({"tool": "write_file", "arguments": {
            "path": "compile.sh", "content": "#!/bin/sh\nprintf ok\n",
        }}))

    def scorer(plan):
        raise OSError("harness_archive_failed")

    run_arm(
        tmp_path / "eval-error", arm="A2", tasks="kisielk__errcheck", live=False,
        sender=sender, execute=True, scorer=scorer, signal_override="unknown",
        models="fox/deepseek-flash,fox/glm-5.3-flash")
    lines = (tmp_path / "eval-error" / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["submission_ok"] is True
    assert row["pass_rate"] is None
    assert row["score"] is None
    assert row["resolved"] is None
    assert row["harness_resolved"] is None
    assert row["harness_status"] == "unavailable"
    assert row["harness_reason"] == "evaluator_error"
    assert row["status"] != "submission_rejected"
    assert "submission_rejection_reason" not in row
