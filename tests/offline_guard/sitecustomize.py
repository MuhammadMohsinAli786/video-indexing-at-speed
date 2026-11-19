"""Offline test guard: put this folder on PYTHONPATH to make every Python process (including spawned
workers) fail any network access, and log each attempt (with a stack) to $OFFLINE_LOG."""
import os
import socket
import traceback

_LOG = os.environ.get("OFFLINE_LOG")


def _blocked(what):
    def f(*a, **k):
        if _LOG:
            with open(_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"pid={os.getpid()} BLOCKED {what} args={a[:2]!r}\n{''.join(traceback.format_stack(limit=12))}\n")
        raise OSError(f"network access blocked by offline test ({what})")
    return f


socket.getaddrinfo = _blocked("getaddrinfo")
socket.gethostbyname = _blocked("gethostbyname")
socket.create_connection = _blocked("create_connection")
socket.socket.connect = _blocked("socket.connect")
socket.socket.connect_ex = _blocked("socket.connect_ex")
