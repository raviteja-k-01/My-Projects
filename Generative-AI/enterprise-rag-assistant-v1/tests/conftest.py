"""The unit suite must never spend model quota or contact external services."""
import socket

import pytest


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
