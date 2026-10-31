"""Default tests may not connect to a network; all responses are scripted."""
import socket

import pytest


@pytest.fixture(autouse=True)
def isolate_run_directory(monkeypatch):
    """Tests must not inherit a live run directory from the process environment."""
    monkeypatch.delenv("ALO_RUN_DIR", raising=False)


@pytest.fixture(autouse=True)
def reject_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access is forbidden in source tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket.socket, "sendto", blocked)
