"""The shlex fast path, the scan deadline, and the per-request call journal.

The fast path must not change one scan finding. Every corpus line is split
both ways (fast path vs plain ``shlex.split``), the linear tokenizer used for
long lines is compared with ``shlex.split`` on the same lines and on long
generated ones, and every corpus text is scanned both ways. ``ALO_PB_SCAN_CORPUS`` may name extra directories, e.g. a
copy of a live workspace; the repository itself is always part of the corpus.
"""
import json
import os
import random
import shlex
import time
from pathlib import Path

import pytest

import alo_cloud.pb_contract as pc
from alo_cloud.fox_pb import CALL_JOURNAL, SCAN_MIN_SECONDS, _scan_submission, run_arm

REPO = Path(__file__).resolve().parents[1]
MAX_CORPUS_FILE = 2_000_000


def _plain(line):
    return shlex.split(line, comments=True, posix=True)


def _split(fn, line):
    try:
        return ("ok", fn(line))
    except ValueError as error:
        return ("ValueError", str(error))


def _corpus_dirs():
    dirs = [REPO / "src", REPO / "tests", REPO / "docs", REPO / "scripts"]
    extra = os.environ.get("ALO_PB_SCAN_CORPUS")
    if extra:
        dirs.extend(Path(item) for item in extra.split(os.pathsep) if item)
    return [item for item in dirs if item.is_dir()]


def _corpus_texts():
    for root in _corpus_dirs():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
                continue
            if path.stat().st_size > MAX_CORPUS_FILE:
                continue
            try:
                yield path, path.read_bytes().decode("utf-8")
            except (OSError, UnicodeError):
                continue


_ALPHABET = list("ab09_-./=:,+@%") + [" ", "\t", "\r", "'", '"', "\\", "#", ";", "|", "&", ">", "<",
             "$", "`", "(", ")", "{", "}", "*", "?", "!", "~", "\x0b", "\x0c", "\u00a0", "\u2003",
             "\u3000", "\u00e9", "\u4e2d", "\x00"]


def _adversarial_lines(count=20000, seed=20261116):
    rng = random.Random(seed)
    for _ in range(count):
        size = rng.choice([0, 1, 2, 3, 5, 8, 13, 40, 200])
        yield "".join(rng.choice(_ALPHABET) for _ in range(size))
    yield from ["", " ", "#", "a#b", "a #b", "'", '"', "\\", "x\\", "./executable", "./executable --help",
                "cp build/executable executable 2>/dev/null || true", "cat<executable", "a\u00a0b", "a\x0bb",
                "x" * 70000, "x" * 70000 + " y", "#" + "x" * 1000]


def test_fast_path_matches_shlex_on_every_corpus_and_adversarial_line():
    checked = 0
    for _path, text in _corpus_texts():
        for line in text.splitlines():
            assert _split(pc._shlex_words, line) == _split(_plain, line), line[:200]
            checked += 1
    for line in _adversarial_lines():
        assert _split(pc._shlex_words, line) == _split(_plain, line), repr(line[:200])
        checked += 1
    assert checked > 20000


def test_linear_tokenizer_matches_shlex_on_every_line():
    checked = 0
    for _path, text in _corpus_texts():
        for line in text.splitlines():
            if len(line) <= 200_000:
                assert _split(pc._posix_shlex_split, line) == _split(_plain, line), line[:200]
                checked += 1
        # Whole texts too: newlines end comments the way shlex's readline does.
        if len(text) <= 200_000:
            assert _split(pc._posix_shlex_split, text) == _split(_plain, text), str(_path)
    for line in _adversarial_lines():
        assert _split(pc._posix_shlex_split, line) == _split(_plain, line), repr(line[:200])
        checked += 1
    rng = random.Random(4096)
    alphabet = _ALPHABET + ["\n", "xxxxxxxx", "    ", "\\\"", "''", '""', " # "]
    for _ in range(600):
        size = rng.choice([50, 500, 5000, 9000])
        line = "".join(rng.choice(alphabet) for _ in range(size))
        assert _split(pc._posix_shlex_split, line) == _split(_plain, line), repr(line[:200])
        assert _split(pc._shlex_words, line) == _split(_plain, line), repr(line[:200])
        checked += 1
    assert checked > 20000


_LONG_LINES = {
    "word_then_space": "x" * 5_000_000 + " ",
    "word_space_word": "x" * 2_500_000 + " " + "y" * 2_500_000,
    "double_quoted": '"' + "x" * 5_000_000 + '"',
    "single_quoted": "'" + "x" * 5_000_000 + "'",
    "many_words": "ab " * 1_000_000,
    "escapes": "\\x" * 1_000_000,
    "comment_tail": "./executable " + "#" + "x" * 5_000_000,
    "unclosed": '"' + "x" * 5_000_000,
}
_LONG_EXPECTED = {
    "word_then_space": ("ok", ["x" * 5_000_000]),
    "word_space_word": ("ok", ["x" * 2_500_000, "y" * 2_500_000]),
    "double_quoted": ("ok", ["x" * 5_000_000]),
    "single_quoted": ("ok", ["x" * 5_000_000]),
    "many_words": ("ok", ["ab"] * 1_000_000),
    "escapes": ("ok", ["x" * 1_000_000]),
    "comment_tail": ("ok", ["./executable"]),
    "unclosed": ("ValueError", "No closing quotation"),
}


@pytest.mark.parametrize("name", sorted(_LONG_LINES))
def test_long_slow_path_lines_are_linear(name):
    line = _LONG_LINES[name]
    started = time.monotonic()
    got = _split(pc._shlex_words, line)
    assert time.monotonic() - started < 10.0
    # Expected values are what shlex.split returns (it is too slow to run here).
    assert got == _LONG_EXPECTED[name]
    pc._simple_commands(line)


def test_scan_deadline_interrupts_one_long_line():
    line = "x" * 5_000_000 + " " + "y " * 200_000

    def expired():
        raise pc.SubmissionScanDeadline("submission_scan_deadline")
    with pytest.raises(pc.SubmissionScanDeadline):
        pc._simple_commands(line, expired)
    with pytest.raises(pc.SubmissionScanDeadline):
        pc._text_submission_violations(line, allow_reference_execution=False, check=expired)


def _fixture_workspace(tmp_path):
    root = tmp_path / "ws"
    (root / "build").mkdir(parents=True)
    (root / ".pb-probe").mkdir()
    (root / "compile.sh").write_text(
        "#!/bin/sh\nset -e\ncd \"$(dirname \"$0\")\"\nmkdir -p build\n"
        "rustc -O -o build/executable src/main.rs\ncp build/executable executable 2>/dev/null || true\n",
        encoding="utf-8")
    (root / ".pb-probe" / "probe.sh").write_text("#!/bin/sh\n./executable \"$@\" > out.txt\n", encoding="utf-8")
    (root / "steal.sh").write_text("cp ./executable ./build/wrapped\nstrings executable|head\n", encoding="utf-8")
    (root / "notes.txt").write_text("x" * 100000 + "\nplain\n'unclosed\n", encoding="utf-8")
    return root


def test_scan_findings_identical_with_and_without_fast_path(monkeypatch, tmp_path):
    texts = list(_corpus_texts())
    # Whole-tree scans: the repository and a fixture. Extra corpus trees are
    # compared file by file (a >2 MB one-line file would take hours on the
    # slow path, which is the bug being fixed).
    roots = [REPO / "src", REPO / "tests", REPO / "docs", _fixture_workspace(tmp_path)]

    def run_all():
        per_text = [(pc._text_submission_violations(text, allow_reference_execution=True),
                     pc._text_submission_violations(text, allow_reference_execution=False))
                    for _path, text in texts]
        scans = [pc.submission_reference_scan(root, reference_sha256="0" * 64) for root in roots]
        return per_text, scans

    fast = run_all()
    monkeypatch.setattr(pc, "_shlex_words", lambda line, check=None: _plain(line))
    slow = run_all()
    assert fast == slow
    assert any(scan["violations"] for scan in fast[1])


@pytest.mark.parametrize("tail", ["", " ", " ./executable"])
def test_five_megabyte_single_line_is_fast(tmp_path, tail):
    line = "x" * 5_000_000 + tail
    started = time.monotonic()
    assert pc._simple_commands(line) == [["x" * 5_000_000] + tail.split()]
    assert time.monotonic() - started < 2.0
    nested = tmp_path / "ws" / ".pbwork" / "t1" / "sub1" / "sub2"
    nested.mkdir(parents=True)
    (nested / "d.txt").write_text(line, encoding="utf-8")
    started = time.monotonic()
    result = pc.submission_reference_scan(tmp_path / "ws", reference_sha256="0" * 64)
    assert result["ok"] is True
    assert time.monotonic() - started < 20.0


def test_scan_deadline_is_checked_per_file_and_line(tmp_path):
    root = _fixture_workspace(tmp_path)
    ticks = iter(range(10_000))
    with pytest.raises(pc.SubmissionScanDeadline):
        pc.submission_reference_scan(root, reference_sha256="0" * 64, deadline=3, clock=lambda: next(ticks))
    assert pc.submission_reference_scan(root, reference_sha256="0" * 64) == pc.submission_reference_scan(
        root, reference_sha256="0" * 64, deadline=time.monotonic() + 600)


def test_unit_scan_gets_a_floor_after_the_unit_budget_is_spent(tmp_path):
    root = _fixture_workspace(tmp_path)
    # The agent loop used the whole budget long ago; the ordinary scan still runs.
    submission, failure = _scan_submission(root, started=0.0, unit_seconds=1.0)
    assert failure is None and submission == pc.submission_reference_scan(root)
    now = [1000.0]

    def clock():
        now[0] += SCAN_MIN_SECONDS / 2
        return now[0]

    submission, failure = _scan_submission(root, started=0.0, unit_seconds=1.0, clock=clock)
    assert submission is None and failure == "submission_scan_deadline"


def test_every_call_is_journaled_as_it_returns(tmp_path):
    result = run_arm(tmp_path / "A2", arm="A2", g0=True, request_cap=40, node_request_cap=20,
                     differential_signal=True, live=False)
    assert result["wrote"] >= 1
    run_dir = tmp_path / "A2"
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text().splitlines()]
    journal = [json.loads(line) for line in (run_dir / CALL_JOURNAL).read_text().splitlines()]
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    assert len(journal) == len(calls) > 0
    attempts = {record["unit_id"]: record["attempt_id"] for record in records}
    for row in journal:
        assert row["attempt_id"] == attempts[row["unit_id"]]
        assert "text" not in row

    def strip(row):
        return {k: v for k, v in row.items() if k not in {"attempt_id", "journal_seq"}}
    assert [strip(row) for row in journal] == calls
