"""Wall-clock deadline for one fox-single attempt. Local sockets only.

The fake server speaks HTTP on 127.0.0.1. It never dials a gateway and it
does not read FOXROUTER_API_KEY.
"""
from datetime import datetime, timezone
import http.client
import inspect
import json
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time

from alo_cloud.fox_pool import DEFAULT_READ_TIMEOUT_S, resolve_model
from alo_cloud.fox_single import interpret_call, post_with_retries, run_schedule, urllib_fox_send


WHEN = datetime(2026, 11, 13, 8, 0, tzinfo=timezone.utc)
TIMEOUT_S = 0.7
HOLD_S = 4.0
TRICKLE_S = 0.05
_PROXY_NAMES = (
    "http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
)
# Captured before the suite blocks connect. Tests may dial only this process's loopback server.
_REAL_CREATE_CONNECTION = socket.create_connection
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


def _clock():
    return WHEN


def _loopback_only(address):
    host = address[0] if isinstance(address, tuple) else address
    if host not in _LOOPBACK:
        raise AssertionError("network access is forbidden in source tests")


def _clear_proxies(monkeypatch):
    for name in _PROXY_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    # The suite rejects every connect. This test dials 127.0.0.1 only.
    def connect(self, address):
        _loopback_only(address)
        return _REAL_CONNECT(self, address)

    def connect_ex(self, address):
        _loopback_only(address)
        return _REAL_CONNECT_EX(self, address)

    def create_connection(address, *args, **kwargs):
        _loopback_only(address)
        return _REAL_CREATE_CONNECTION(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)


def _completion_body():
    payload = {
        "model": "deepseek-v4.1-flash-1017",
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"final":"ok"}'},
        }],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "cost": 0.25,
            "credit": 3,
            "quota": 9,
        },
    }
    return json.dumps(payload).encode("utf-8")


def _http_head(status, reason, body):
    return (
        f"HTTP/1.1 {status} {reason}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")


class _LocalGateway:
    """Accept loop. ``mode`` selects trickle, a finished body, 524, or a reset."""

    def __init__(self, mode, *, scheme="http", server_context=None):
        self.mode = mode
        self.scheme = scheme
        self.server_context = server_context
        self.ok_body = _completion_body()
        self.errors = []
        self.sessions = []
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._conns = []
        self._workers = []
        self._listener = None
        self.port = None
        self._thread = threading.Thread(target=self._serve, name="fox-deadline-fake", daemon=True)

    def start(self):
        self._thread.start()
        if not self._ready.wait(2):
            self.close()
            raise RuntimeError("local gateway did not bind")
        return self

    @property
    def url(self):
        return f"{self.scheme}://127.0.0.1:{self.port}/v1/chat/completions"

    def wait_session(self, index, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if any(item["index"] == index for item in self.sessions):
                    return
            time.sleep(0.02)
        raise AssertionError(f"session {index} was not recorded")

    def close(self):
        self._stop.set()
        with self._lock:
            conns = list(self._conns)
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        self._thread.join(2)
        for worker in self._workers:
            worker.join(2)

    def _serve(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        listener.settimeout(0.2)
        self._listener = listener
        self.port = listener.getsockname()[1]
        self._ready.set()
        index = 0
        while not self._stop.is_set():
            try:
                conn, _addr = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            index += 1
            with self._lock:
                self._conns.append(conn)
            worker = threading.Thread(
                target=self._handle, args=(conn, index), name=f"fox-deadline-{index}", daemon=True)
            self._workers.append(worker)
            worker.start()

    def _handle(self, conn, index):
        started = time.monotonic()
        sent = 0
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(2)
            if self.server_context is not None:
                conn = self.server_context.wrap_socket(conn, server_side=True)
            self._read_request(conn)
            if self.mode == "reset" and index == 1:
                self._reset(conn)
            elif self.mode == "trickle" or (self.mode == "trickle_once" and index == 1):
                sent = self._trickle(conn)
            elif self.mode == "http_524" and index == 1:
                self._send_split(conn, 524, b"error code: 524", "Gateway Timeout")
            else:
                self._send_split(conn, 200, self.ok_body, "OK")
        except OSError:
            pass
        except Exception as exc:
            self.errors.append(type(exc).__name__)
        finally:
            with self._lock:
                self.sessions.append({
                    "index": index,
                    "sent": sent,
                    "elapsed": time.monotonic() - started,
                })
            try:
                conn.close()
            except OSError:
                pass

    def _read_request(self, conn):
        conn.settimeout(2)
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 1_000_000:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        length = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":", 1)[1].strip())
        while len(rest) < length:
            chunk = conn.recv(4096)
            if not chunk:
                break
            rest += chunk

    def _trickle(self, conn):
        """Status line, then one byte at a time and no terminating header."""
        blob = b"HTTP/1.1 200 OK\r\n"
        conn.sendall(blob)
        sent = len(blob)
        deadline = time.monotonic() + HOLD_S
        while time.monotonic() < deadline and not self._stop.is_set():
            try:
                conn.sendall(b"X")
            except OSError:
                break
            sent += 1
            time.sleep(TRICKLE_S)
        return sent

    def _send_split(self, conn, status, body, reason):
        conn.sendall(_http_head(status, reason, body))
        time.sleep(0.05)
        conn.sendall(body)

    def _reset(self, conn):
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        conn.close()


def _post(server, *, timeout_s, max_attempts, sleep):
    spans = []

    def sender(url, body, headers, timeout):
        started = time.monotonic()
        try:
            return urllib_fox_send(url, body, headers, timeout)
        finally:
            spans.append(time.monotonic() - started)

    model = resolve_model("fox/deepseek-flash")
    result, retries, _body = post_with_retries(
        model, [{"role": "user", "content": "hi"}], 32,
        key="unit-test-fox-credential", sender=sender, timeout_s=timeout_s,
        sleep=sleep, clock=_clock, max_attempts=max_attempts, url=server.url)
    return model, result, retries, spans


def _near_deadline(span):
    assert TIMEOUT_S - 0.25 <= span <= TIMEOUT_S + 1.0, span


def test_read_timeout_default_stays_600_seconds():
    assert DEFAULT_READ_TIMEOUT_S == 600
    assert inspect.signature(run_schedule).parameters["timeout_s"].default == 600
    proc = subprocess.run(
        [sys.executable, "-m", "alo_cloud.fox_single", "run", "--help"],
        check=False, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--read-timeout" in proc.stdout
    assert "Wall-clock seconds" in proc.stdout


def test_finished_response_keeps_usage_cost_credit_and_quota(monkeypatch):
    _clear_proxies(monkeypatch)
    server = _LocalGateway("ok").start()
    try:
        model, result, retries, spans = _post(server, timeout_s=2, max_attempts=2, sleep=lambda _n: None)
    finally:
        server.close()
    assert not server.errors
    assert retries == []
    assert spans[0] < 1.0
    assert result["status"] == 200
    row = interpret_call(model, [], 32, result, retries, estimated=11)
    assert row["ok"] is True
    assert "failure" not in row
    assert row["prompt_tokens"] == 11
    assert row["completion_tokens"] == 7
    assert row["total_tokens"] == 18
    assert row["reasoning_tokens"] is None
    assert row["raw_usage"]["prompt_tokens"] == 11
    assert row["raw_usage"]["cost"] == 0.25
    assert row["cost"] == 0.25
    assert row["credit"] == 3
    assert row["quota"] == 9
    assert row["upstream_model"] == "deepseek-v4.1-flash-1017"
    assert row["returned_model"] == "deepseek-v4.1-flash-1017"


def test_trickle_is_cut_off_near_the_deadline_and_retried(monkeypatch):
    _clear_proxies(monkeypatch)
    slept = []
    server = _LocalGateway("trickle_once").start()
    try:
        model, result, retries, spans = _post(
            server, timeout_s=TIMEOUT_S, max_attempts=3, sleep=slept.append)
        row = interpret_call(model, [], 32, result, retries, estimated=11)
        server.wait_session(1)
    finally:
        server.close()
    assert spans[0] < HOLD_S / 2
    _near_deadline(spans[0])
    first = next(item for item in server.sessions if item["index"] == 1)
    assert first["sent"] > len(b"HTTP/1.1 200 OK\r\n")
    assert first["sent"] < 80
    assert retries[0]["kind"] == "timeout"
    assert retries[0]["status"] is None
    assert retries[0]["cloudflare_error"] is None
    assert retries[0]["attempt"] == 1
    assert retries[0]["timestamp"] == "2026-11-13T08:00:00Z"
    assert slept == [1.0]
    assert result["status"] == 200
    assert len(spans) == 2
    assert spans[1] < 1.0
    assert row["ok"] is True
    assert row["retries"][0]["kind"] == "timeout"
    assert row["prompt_tokens"] == 11
    assert row["completion_tokens"] == 7
    assert row["total_tokens"] == 18
    assert row["cost"] == 0.25
    assert row["credit"] == 3
    assert row["quota"] == 9
    assert row["raw_usage"]["prompt_tokens"] == 11


def test_deadline_expiry_matches_the_existing_timeout_ledger_row(monkeypatch):
    _clear_proxies(monkeypatch)
    slept = []
    server = _LocalGateway("trickle").start()
    try:
        model, result, retries, spans = _post(
            server, timeout_s=TIMEOUT_S, max_attempts=3, sleep=slept.append)
        for index in (1, 2, 3):
            server.wait_session(index)
    finally:
        server.close()
    assert len(spans) == 3
    for span in spans:
        assert span < HOLD_S / 2
        _near_deadline(span)
    assert slept == [1.0, 2.0]

    def boom(url, body, headers, timeout_s):
        del url, body, headers, timeout_s
        raise TimeoutError("slow")

    ref_slept = []
    ref_result, ref_retries, _body = post_with_retries(
        model, [{"role": "user", "content": "hi"}], 32,
        key="unit-test-fox-credential", sender=boom, timeout_s=TIMEOUT_S,
        sleep=ref_slept.append, clock=_clock, max_attempts=3)
    assert ref_slept == slept
    assert retries == ref_retries
    assert [item["kind"] for item in retries] == ["timeout", "timeout", "timeout"]
    row = interpret_call(model, [], 32, result, retries, estimated=11)
    ref = interpret_call(model, [], 32, ref_result, ref_retries, estimated=11)
    assert row == ref
    assert row["failure"] == "timeout"
    assert row["ok"] is False
    assert row["http_status"] is None
    assert row["raw_usage"] is None
    assert row["prompt_tokens"] is None
    assert row["completion_tokens"] is None
    assert row["total_tokens"] is None
    assert row["reasoning_tokens"] is None
    assert row["cost"] is None
    assert row["credit"] is None
    assert row["quota"] is None
    assert row["sends"] == 3


def test_prompt_524_is_still_an_http_retry(monkeypatch):
    _clear_proxies(monkeypatch)
    slept = []
    server = _LocalGateway("http_524").start()
    try:
        _model, result, retries, spans = _post(
            server, timeout_s=TIMEOUT_S, max_attempts=3, sleep=slept.append)
    finally:
        server.close()
    assert spans[0] < 0.5
    assert retries[0]["kind"] == "http"
    assert retries[0]["status"] == 524
    assert retries[0]["cloudflare_error"] == "524"
    assert slept == [1.0]
    assert result["status"] == 200


def _local_cert(tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(key), "-out", str(cert), "-days", "1",
         "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1"],
        check=True, capture_output=True)
    return cert, key


def test_https_trickle_uses_the_same_deadline(monkeypatch, tmp_path):
    _clear_proxies(monkeypatch)
    cert, key = _local_cert(tmp_path)
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert, key)

    def trusted(_http_version):
        return ssl.create_default_context(cafile=str(cert))

    monkeypatch.setattr(http.client, "_create_https_context", trusted)
    slept = []
    server = _LocalGateway("trickle_once", scheme="https", server_context=server_context).start()
    try:
        model, result, retries, spans = _post(
            server, timeout_s=TIMEOUT_S, max_attempts=2, sleep=slept.append)
        server.wait_session(1)
        row = interpret_call(model, [], 32, result, retries, estimated=11)
    finally:
        server.close()
    _near_deadline(spans[0])
    assert spans[0] < HOLD_S / 2
    assert retries[0]["kind"] == "timeout"
    assert slept == [1.0]
    assert result["status"] == 200
    assert row["ok"] is True
    assert "failure" not in row
    assert row["prompt_tokens"] == 11
    assert row["cost"] == 0.25
    assert row["credit"] == 3
    assert row["quota"] == 9


def test_fast_reset_stays_connection_reset(monkeypatch):
    _clear_proxies(monkeypatch)
    slept = []
    server = _LocalGateway("reset").start()
    try:
        _model, result, retries, spans = _post(
            server, timeout_s=TIMEOUT_S, max_attempts=2, sleep=slept.append)
    finally:
        server.close()
    assert spans[0] < 0.5
    assert retries[0]["kind"] == "connection_reset"
    assert retries[0]["status"] is None
    assert slept == [1.0]
    assert result["status"] == 200
