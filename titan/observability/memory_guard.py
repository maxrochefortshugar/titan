"""Process-level memory guard for a serving process.

The scheduler's memory guard gates request admission. This one runs before the
model is loaded and answers a different question: may this process take the
GPU at all? Three checks, all cheap:

1. Exactly one server per port. A lock file keyed by the port is held for the
   life of the process, so a second `titan serve` on the same port exits before
   it maps a single weight instead of racing the first one for memory.
2. Enough memory is actually available now. macOS releases a dead process's
   pages lazily, so a server started seconds after another one stopped can
   overlap its 73 GB with the old image for long enough to swap. The check
   reads vm_stat and refuses to start until free plus inactive plus purgeable
   pages cover the weights plus a margin, waiting up to a bounded time for the
   previous image to drain.
3. A hard ceiling on MLX allocations, set with mx.set_memory_limit, so the
   process cannot grow past the configured guard whatever the workload does.

Nothing here reads the environment; the numbers come from the config.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from titan.core.errors import ConfigError

GIB = 1024 ** 3


@dataclass(frozen=True)
class MemorySnapshot:
    available_gb: float
    total_gb: float


def snapshot() -> MemorySnapshot:
    """Available memory as macOS accounts it: free, inactive, speculative and purgeable pages."""
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout)
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = 16384
    counts = {}
    for line in out.splitlines():
        if line.startswith("Mach Virtual Memory Statistics"):
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                page = int(digits)
            continue
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if value.isdigit():
            counts[key.strip()] = int(value)
    avail = sum(counts.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")) * page
    return MemorySnapshot(available_gb=avail / GIB, total_gb=total / GIB)


class ServeLock:
    """Advisory lock: one serving process per port on this machine."""

    def __init__(self, port: int, directory: Path | None = None):
        base = directory or Path.home() / ".config" / "titan" / "run"
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"serve-{port}.lock"
        self._fh = None

    def acquire(self) -> None:
        fh = open(self.path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.seek(0)
            holder = fh.read().strip() or "unknown pid"
            fh.close()
            raise ConfigError(
                f"another titan server already holds {self.path} ({holder}); "
                "one model instance at a time on this machine"
            ) from exc
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


def wait_for_headroom(need_gb: float, *, timeout_s: float = 120.0, poll_s: float = 2.0, log=print) -> MemorySnapshot:
    """Block until `need_gb` is available, or raise ConfigError after `timeout_s`."""
    deadline = time.monotonic() + timeout_s
    last = snapshot()
    while last.available_gb < need_gb:
        if time.monotonic() >= deadline:
            raise ConfigError(
                f"refusing to load: {need_gb:.1f} GB needed, {last.available_gb:.1f} GB available "
                f"after {timeout_s:.0f}s (total {last.total_gb:.0f} GB). Another model image is "
                "probably still being released; wait or stop it."
            )
        log(f"waiting for memory: need {need_gb:.1f} GB, available {last.available_gb:.1f} GB")
        time.sleep(poll_s)
        last = snapshot()
    return last


def apply_hard_limit(limit_gb: float) -> None:
    """Cap MLX allocations for this process at `limit_gb`."""
    import mlx.core as mx  # noqa: PLC0415 - keep import cost out of check-config

    mx.set_memory_limit(int(limit_gb * GIB))


def preflight(port: int, weights_gb: float, guard_gb: float, *, margin_gb: float = 8.0, timeout_s: float = 120.0, log=print) -> ServeLock:
    """Run the three checks; return the held lock (keep it alive for the process lifetime)."""
    lock = ServeLock(port)
    lock.acquire()
    try:
        snap = wait_for_headroom(weights_gb + margin_gb, timeout_s=timeout_s, log=log)
        apply_hard_limit(guard_gb)
        log(f"memory preflight ok: {snap.available_gb:.1f} GB available, hard limit {guard_gb:.0f} GB")
    except Exception:
        lock.release()
        raise
    return lock
