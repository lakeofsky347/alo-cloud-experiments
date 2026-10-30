"""Default tests may not connect to a network; all responses are scripted."""
import socket

import pytest


@pytest.fixture(autouse=True)
def reject_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access is forbidden in source tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket.socket, "sendto", blocked)
