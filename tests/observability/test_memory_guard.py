"""Process-level memory guard: one server per port, headroom wait, hard limit."""
import pytest

from titan.core.errors import ConfigError
from titan.observability import memory_guard as mg


def test_snapshot_reads_this_machine():
    s = mg.snapshot()
    assert 0 < s.available_gb <= s.total_gb


def test_serve_lock_is_exclusive(tmp_path):
    a = mg.ServeLock(9999, tmp_path)
    b = mg.ServeLock(9999, tmp_path)
    a.acquire()
    with pytest.raises(ConfigError, match="already holds"):
        b.acquire()
    a.release()
    b.acquire()
    b.release()


def test_wait_for_headroom_times_out(monkeypatch):
    monkeypatch.setattr(mg, "snapshot", lambda: mg.MemorySnapshot(available_gb=10.0, total_gb=128.0))
    with pytest.raises(ConfigError, match="refusing to load"):
        mg.wait_for_headroom(80.0, timeout_s=0.05, poll_s=0.01, log=lambda m: None)


def test_wait_for_headroom_returns_when_memory_frees(monkeypatch):
    seq = iter([20.0, 40.0, 90.0])
    monkeypatch.setattr(mg, "snapshot", lambda: mg.MemorySnapshot(available_gb=next(seq), total_gb=128.0))
    snap = mg.wait_for_headroom(80.0, timeout_s=5.0, poll_s=0.0, log=lambda m: None)
    assert snap.available_gb == 90.0


def test_preflight_releases_lock_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(mg, "snapshot", lambda: mg.MemorySnapshot(available_gb=1.0, total_gb=128.0))
    monkeypatch.setattr(mg, "ServeLock", lambda port, directory=None: _RealLock(port, tmp_path))
    with pytest.raises(ConfigError, match="refusing to load"):
        mg.preflight(8085, 75.0, 110.0, timeout_s=0.05, log=lambda m: None)
    probe = _RealLock(8085, tmp_path)
    probe.acquire()  # would raise if preflight had left the lock held
    probe.release()


_RealLock = mg.ServeLock
