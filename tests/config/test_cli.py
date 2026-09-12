"""The ``titan`` command line, in process and through a subprocess.

``main`` takes its streams as arguments so the ordinary case is an in-process
call that reads what was printed. One subprocess test remains, because the thing
it checks, that ``python -m titan`` resolves and runs at all, cannot be checked
any other way.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from titan.cli import main

from .conftest import FULL, MINIMAL

REPO_ROOT = Path(__file__).resolve().parents[2]


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# check-config
# ---------------------------------------------------------------------------


def test_check_config_accepts_a_valid_file(write_config):
    path = write_config(FULL)
    code, out, err = run("check-config", "--config", str(path))
    assert code == 0
    assert out.startswith(f"ok {path}")
    assert err == ""


def test_check_config_reports_the_overrides_it_applied(write_config):
    path = write_config(MINIMAL)
    code, out, _ = run(
        "check-config", "--config", str(path), "--set", "scheduler.max_seqs=2"
    )
    assert code == 0
    assert "overrides: scheduler.max_seqs=2" in out


def test_check_config_fails_with_the_offending_key(write_config):
    path = write_config(MINIMAL + "\n[scheduler]\nmax_seq = 2\n")
    code, out, err = run("check-config", "--config", str(path))
    assert code == 2
    assert "scheduler.max_seq" in err
    assert out == ""


def test_check_config_fails_on_a_reserved_port(write_config):
    path = write_config(MINIMAL)
    code, _, err = run("check-config", "--config", str(path), "--set", "server.port=8083")
    assert code == 2
    assert "reserved" in err


def test_check_config_fails_when_there_is_no_config(monkeypatch):
    from titan.config.settings import TITAN_CONFIG_ENV

    monkeypatch.delenv(TITAN_CONFIG_ENV, raising=False)
    code, _, err = run("check-config")
    assert code == 2
    assert TITAN_CONFIG_ENV in err


def test_check_config_does_not_touch_the_model_directory(write_config):
    """The path in the config does not exist. Checking must not care."""
    path = write_config(MINIMAL)
    assert not Path("/models/qwen").exists()
    assert run("check-config", "--config", str(path))[0] == 0


# ---------------------------------------------------------------------------
# print-config
# ---------------------------------------------------------------------------


def test_print_config_redacts_the_api_key(write_config):
    path = write_config(FULL)
    code, out, _ = run("print-config", "--config", str(path))
    assert code == 0
    assert "/etc/titan/titan.key" not in out
    assert "***redacted***" in out


def test_print_config_can_show_the_key_path_when_asked(write_config):
    path = write_config(FULL)
    _, out, _ = run("print-config", "--config", str(path), "--no-redact")
    assert "/etc/titan/titan.key" in out


def test_print_config_emits_reloadable_toml(write_config):
    path = write_config(FULL)
    _, out, _ = run("print-config", "--config", str(path), "--no-redact")
    reparsed = tomllib.loads(out)
    assert reparsed["server"]["port"] == 8085
    assert reparsed["cache"]["block_tokens"] == 512


def test_print_config_json(write_config):
    import json

    path = write_config(FULL)
    _, out, _ = run("print-config", "--config", str(path), "--format", "json")
    data = json.loads(out)
    assert data["scheduler"]["memory_guard_soft_fraction"] == 0.85
    assert data["server"]["api_key_file"] == "***redacted***"


def test_print_config_shows_the_overrides_in_the_output(write_config):
    path = write_config(MINIMAL)
    _, out, _ = run(
        "print-config", "--config", str(path), "--set", "speculation.enabled=false"
    )
    data = tomllib.loads(out)
    assert data["overrides"] == ["speculation.enabled=false"]
    assert data["speculation"]["enabled"] is False


def test_print_config_fails_on_an_invalid_file(write_config):
    path = write_config(MINIMAL + "\n[cache]\nsnapshot_grid = 1500\n")
    code, _, err = run("print-config", "--config", str(path))
    assert code == 2
    assert "cache.snapshot_grid" in err


# ---------------------------------------------------------------------------
# bench and parity
# ---------------------------------------------------------------------------


def test_bench_lists_the_scripts():
    code, out, _ = run("bench")
    assert code == 0
    assert "decode_bench.py" in out
    assert "parity/greedy_parity.py" in out


def test_bench_can_name_one_script():
    code, out, _ = run("bench", "decode_bench")
    assert code == 0
    assert "decode_bench.py" in out
    assert "concurrency_sweep.py" not in out


def test_bench_rejects_an_unknown_script():
    code, _, err = run("bench", "nosuchbench")
    assert code == 2
    assert "nosuchbench" in err


def test_parity_passes_its_arguments_through(monkeypatch):
    seen: list[list[str]] = []

    def fake_call(cmd, *a, **kw):
        seen.append(cmd)
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    code, _, _ = run("parity", "--", "--only", "tools", "--reference-only")
    assert code == 0
    assert seen[0][0] == sys.executable
    assert seen[0][1].endswith("bench/parity/greedy_parity.py")
    assert seen[0][2:] == ["--only", "tools", "--reference-only"]


# ---------------------------------------------------------------------------
# the installed entry point
# ---------------------------------------------------------------------------


def test_the_module_entry_point_runs(write_config):
    path = write_config(MINIMAL)
    proc = subprocess.run(
        [sys.executable, "-m", "titan", "check-config", "--config", str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("ok ")


def test_the_entry_point_declared_in_pyproject_is_this_module():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert data["project"]["scripts"]["titan"] == "titan.cli:main"


def test_an_unknown_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit):
        main(["nosuchcommand"], out=io.StringIO(), err=io.StringIO())
