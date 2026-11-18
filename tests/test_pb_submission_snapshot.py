import json
import tarfile

import pytest

from alo_cloud.collect import collect_sealed
from alo_cloud.evaluate import programbench_unit_commands
from alo_cloud.fox_pb import FoxAloServices, PbServices, _capture_final_submission
from alo_cloud.pb_submission import capture_submission, pack_snapshot, scan_snapshot, submission_digest, submission_file_state
from alo_cloud.programbench_prompt import PB_BUILD_PROTOCOL


def write(root, path, data):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def test_one_manifest_filters_noninputs_and_seals_the_scanned_bytes(tmp_path):
    source = tmp_path / 'workspace'
    write(source, 'compile.sh', b'#!/bin/sh\ncc src/main.c -o executable\n')
    write(source, 'src/main.c', b'int main(void) { return 0; }\n')
    write(source, 'assets/data.bin', b'\0q')
    for name in ['executable', '.pb-probe/probe.sh', 'build/.alo-pb-candidate', '.git/config',
                 'build/gocache/aa/generated', 'target/release/binary', '.cache/blob']:
        write(source, name, b'never archived')
    (source / 'executable').chmod(0o111)
    write(source, 'build/native', b'\x7fELF\x00')
    (source / '.pb-probe/cycle').symlink_to(source, target_is_directory=True)
    (source / 'compile.sh').chmod(0o755)
    snapshot = capture_submission(source, tmp_path / 'frozen')
    assert snapshot.files == ('assets/data.bin', 'compile.sh', 'src/main.c')
    assert scan_snapshot(snapshot, source)['ok']
    assert submission_file_state(source) == {row['path']: row for row in snapshot.manifest}
    assert capture_submission(snapshot.root, tmp_path / 'again').sha256 == snapshot.sha256
    write(source, 'src/main.c', b'changed')
    assert capture_submission(source, tmp_path / 'modified').sha256 != snapshot.sha256
    tar = tmp_path / 'archive.tar.gz'
    pack_snapshot(snapshot.root, tar, snapshot.manifest, snapshot.sha256)
    original = tar.read_bytes()
    with tarfile.open(tar) as bundle:
        assert tuple(bundle.getnames()) == snapshot.files
        assert bundle.extractfile('src/main.c').read().startswith(b'int main')
        assert bundle.getmember('compile.sh').mode == 0o755
    pack_snapshot(snapshot.root, tar, snapshot.manifest, snapshot.sha256)
    assert tar.read_bytes() == original
    (snapshot.root / 'src/main.c').write_bytes(b'x' * snapshot.manifest[-1]['size'])
    with pytest.raises(RuntimeError, match='scan_child_error'):
        scan_snapshot(snapshot, source)
    with pytest.raises(ValueError, match='file_mismatch'):
        pack_snapshot(snapshot.root, tar, snapshot.manifest, snapshot.sha256)
    assert tar.read_bytes() == original


@pytest.mark.parametrize(('script', 'body', 'error'), [
    ('sh .pb-probe/probe.sh', '', 'build_depends_on_private_probe'),
    ('cc src/main.c -o executable', '#include ".pb-probe/helper.h"', 'build_depends_on_private_probe'),
    ('cc src/main.c -o executable', 'int main(void){return system("./executable");}', 'invokes_reference'),
])
def test_scanner_rejects_submitted_reference_dependencies(tmp_path, script, body, error):
    source = tmp_path / 'workspace'
    write(source, '.pb-probe/probe.sh', b'./executable --help\n')
    write(source, 'compile.sh', ('#!/bin/sh\n' + script + '\n').encode())
    write(source, 'src/main.c', body.encode())
    snapshot = capture_submission(source, tmp_path / 'frozen')
    report = scan_snapshot(snapshot, source)
    assert error in {violation['kind'] for violation in report['violations']}


def test_checkpoint_collection_reuses_frozen_manifest(tmp_path):
    import time
    source = tmp_path / 'units/u/state/workspace'
    write(source, 'compile.sh', b'#!/bin/sh\nexit 1\n')
    directory = tmp_path / 'checkpoints/u/submissions/a'
    capture, scan, failure = _capture_final_submission(source, directory, started=time.monotonic(), unit_seconds=30)
    assert failure is None and scan['ok']
    record = dict(unit_id='u', arm='A2', task_id='task', track='ProgramBench', attempt_id='a',
                  pb_agent_protocol=PB_BUILD_PROTOCOL, submission_sha256=capture.sha256,
                  submission_manifest_path='checkpoints/u/submissions/a/manifest.json')
    write(source, 'compile.sh', b'#!/bin/sh\nexit 2\n')
    directory2 = tmp_path / 'checkpoints/u/submissions/b'
    another, _, failure = _capture_final_submission(source, directory2, started=time.monotonic(), unit_seconds=30)
    assert failure is None and capture.sha256 != another.sha256
    record2 = dict(record, attempt_id='b', submission_sha256=another.sha256,
                   submission_manifest_path='checkpoints/u/submissions/b/manifest.json')
    (tmp_path / 'records.jsonl').write_text(json.dumps(record) + '\n' + json.dumps(record2) + '\n')
    write(source, 'compile.sh', b'an unrelated later edit')
    for _ in range(2):
        collect_sealed(tmp_path)
        for attempt, code in [('a', '1'), ('b', '2')]:
            dest = tmp_path / 'sealed/programbench-collected/A2/u' / attempt / 'task/submission.tar.gz'
            with tarfile.open(dest) as bundle:
                assert bundle.extractfile('compile.sh').read() == f'#!/bin/sh\nexit {code}\n'.encode()
    commands = programbench_unit_commands(tmp_path, '/fake/repo')
    assert [row['submission_sha256'] for row in commands] == [capture.sha256, another.sha256]
    assert commands[0]['command'] != commands[1]['command']
    assert not (tmp_path / 'sealed/programbench').exists()


def test_implementation_command_receipts_include_actual_files_and_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    write(tmp_path, 'old.c', b'old')
    write(tmp_path, 'gone.c', b'')
    svc = object.__new__(PbServices)
    svc.workspace = SimpleNamespace(root=tmp_path)
    svc.pb_build_enabled = True
    svc.current_node = 'p1'
    monkeypatch.setattr(svc, '_wall_remaining', lambda: 120)

    def command(self, argv, timeout_s):
        write(tmp_path, 'old.c', '新'.encode())
        write(tmp_path, 'new.rs', b'')
        write(tmp_path, '.pb-probe/probe.sh', b'probe')
        (tmp_path / 'gone.c').unlink(missing_ok=True)
        return {'returncode': 1, 'stderr': 'partial command'}

    monkeypatch.setattr(FoxAloServices, 'command', command)
    result = svc.execute_code('sh', 'echo test')
    assert result['returncode'] == 1
    assert result['changed_files'] == [{'path': 'new.rs', 'bytes': 0}, {'path': 'old.c', 'bytes': 3}]
    assert result['deleted_files'] == ['gone.c']
    svc.current_node = 'p2'
    assert 'changed_files' not in svc.command(['pwd'])


def test_unused_root_probe_and_go_cache_never_reach_build_scan_or_archive(tmp_path):
    source = tmp_path / 'workspace'
    write(source, 'compile.sh', b'#!/bin/sh\ncc main.c -o executable\n')
    write(source, 'main.c', b'int main(void){return 0;}\n')
    write(source, 'probe1.py', b'import subprocess\nsubprocess.run(["./executable", "--help"])\n')
    write(source, 'probe2.py', b'print("legitimate business asset")\n')
    write(source, '.gocache/fa/raw', b'#!/bin/sh\n./executable\n')
    frozen = capture_submission(source, tmp_path / 'frozen')
    assert frozen.files == ('compile.sh', 'main.c', 'probe2.py')
    assert sorted(submission_file_state(source)) == list(frozen.files)
    assert submission_digest(source) == frozen.sha256
    assert scan_snapshot(frozen, source)['ok']
    assert capture_submission(frozen.root, tmp_path / 'build').sha256 == frozen.sha256
    pack_snapshot(frozen.root, tmp_path / 'check.tgz', frozen.manifest, frozen.sha256)
    with tarfile.open(tmp_path / 'check.tgz') as bundle:
        assert tuple(bundle.getnames()) == frozen.files
    assert (source / 'probe1.py').exists()


@pytest.mark.parametrize(('compile_sh', 'main', 'probe', 'kind'), [
    ('python3 probe1.py', '', 'subprocess.run(["./executable"])', 'invokes_reference'),
    ('cc "$SRC" -o executable', '', 'subprocess.run(["./executable"])', 'invokes_reference'),
    ('cc main.c -o executable', 'int main(){return system("python3 probe1.py");}',
     'subprocess.run(["./executable"])', 'invokes_reference'),
    ('cc main.c -o executable', '', 'open("./executable", "rb").read()', 'reads_reference_bytes'),
])
def test_root_scripts_with_unresolved_or_real_reference_dependencies_are_rejected(
        tmp_path, compile_sh, main, probe, kind):
    source = tmp_path / 'workspace'
    write(source, 'compile.sh', ('#!/bin/sh\n' + compile_sh + '\n').encode())
    write(source, 'main.c', main.encode())
    write(source, 'probe1.py', (probe + '\n').encode())
    snapshot = capture_submission(source, tmp_path / 'frozen')
    assert 'probe1.py' in snapshot.files
    assert {'kind': kind, 'path': 'probe1.py'} in scan_snapshot(snapshot, source)['violations']
