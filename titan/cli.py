"""The ``titan`` command line.

Five subcommands and no options that are not also config keys. Anything you can
change about a run is either in the TOML file or in a ``--set`` override that
gets recorded in the resolved config, so a benchmark result can always name the
configuration it came from. That is the same rule the config module is built
around, and the CLI is where it would be easiest to break.

    titan serve --config titan.toml [--set a.b=c]
    titan check-config --config titan.toml
    titan print-config --config titan.toml
    titan parity -- --only tools --reference-only
    titan bench

``check-config`` and ``print-config`` never construct a runtime, never import
MLX and never touch the model directory, so they run on a laptop against a
config for a machine that is not this one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from titan.core.errors import ConfigError

__all__ = ["main", "build_parser"]

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "bench"
PARITY_SCRIPT = BENCH_DIR / "parity" / "greedy_parity.py"


def _add_config_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="path to the TOML config; defaults to $TITAN_CONFIG",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help=(
            "override one key, repeatable. The value is parsed as TOML, so "
            "--set server.port=8085 is an integer and "
            "--set kernels.disabled='[\"topk_radix\"]' is an array."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="titan", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="load the model and serve the HTTP API")
    _add_config_options(serve)

    check = sub.add_parser(
        "check-config", help="parse and validate a config, then exit"
    )
    _add_config_options(check)

    show = sub.add_parser(
        "print-config",
        help="print the resolved config, with the API key path redacted",
    )
    _add_config_options(show)
    show.add_argument(
        "--format", choices=("toml", "json"), default="toml", help="output format"
    )
    show.add_argument(
        "--no-redact",
        action="store_true",
        help="print the API key file path as it is written in the config",
    )

    parity = sub.add_parser(
        "parity",
        help="run the greedy parity harness (bench/parity/greedy_parity.py)",
    )
    parity.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="arguments passed straight through to the harness",
    )

    bench = sub.add_parser("bench", help="list the benchmark scripts")
    bench.add_argument(
        "name", nargs="?", default=None, help="script to describe, without .py"
    )

    return parser


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _load(args: argparse.Namespace):
    from titan.config.wiring import load_config  # noqa: PLC0415

    return load_config(args.config, tuple(args.overrides))


def cmd_check_config(args: argparse.Namespace, out, err) -> int:
    config = _load(args)
    print(f"ok {config.source}", file=out)
    if config.overrides:
        print(f"overrides: {' '.join(config.overrides)}", file=out)
    return 0


def cmd_print_config(args: argparse.Namespace, out, err) -> int:
    config = _load(args)
    redact = not args.no_redact
    if args.format == "json":
        print(json.dumps(config.to_dict(redact=redact), indent=2, sort_keys=False), file=out)
    else:
        print(config.dumps(redact=redact), end="", file=out)
    return 0


def cmd_serve(args: argparse.Namespace, out, err) -> int:
    from titan.config.wiring import build_runtime  # noqa: PLC0415

    config = _load(args)
    runtime = build_runtime(config)
    runtime.start()
    print(
        f"titan serving {config.model.resolved_name()} on "
        f"http://{config.server.host}:{config.server.port}",
        file=out,
    )
    import uvicorn  # noqa: PLC0415

    try:
        uvicorn.run(
            runtime.app,
            host=config.server.host,
            port=config.server.port,
            log_level="info",
            timeout_keep_alive=int(config.server.request_timeout_s),
        )
    finally:
        runtime.stop(30.0)
    return 0


def cmd_parity(args: argparse.Namespace, out, err) -> int:
    if not PARITY_SCRIPT.exists():
        print(f"no parity harness at {PARITY_SCRIPT}", file=err)
        return 2
    passthrough = [a for a in args.args if a != "--"]
    return subprocess.call([sys.executable, str(PARITY_SCRIPT), *passthrough])


def cmd_bench(args: argparse.Namespace, out, err) -> int:
    """Placeholder. Lists what is there; running one is still a direct call.

    The bench scripts each take their own flags and several of them want a
    loaded model, so wrapping them behind one command would either lose their
    arguments or grow a second CLI. Listing them is honest about that.
    """
    scripts = sorted(
        p for p in BENCH_DIR.rglob("*.py") if not p.name.startswith("_")
    )
    if not scripts:
        print(f"no bench scripts under {BENCH_DIR}", file=err)
        return 2
    if args.name:
        matches = [p for p in scripts if p.stem == args.name]
        if not matches:
            print(f"no bench script named {args.name!r}", file=err)
            return 2
        scripts = matches
    print(f"bench scripts under {BENCH_DIR}:", file=out)
    for p in scripts:
        print(f"  {p.relative_to(BENCH_DIR)}\t{_first_line(p)}", file=out)
    print(
        "\nrun one directly, for example: "
        f"python {BENCH_DIR.name}/decode_bench.py --help",
        file=out,
    )
    return 0


def _first_line(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        stripped = line.strip().strip('"').strip("'").strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


_COMMANDS = {
    "serve": cmd_serve,
    "check-config": cmd_check_config,
    "print-config": cmd_print_config,
    "parity": cmd_parity,
    "bench": cmd_bench,
}


def main(argv: Sequence[str] | None = None, out=None, err=None) -> int:
    """Entry point. Returns a status code rather than calling ``sys.exit``.

    ``out`` and ``err`` are injectable so a test can call this in process and
    read what it printed, which is cheaper and clearer than a subprocess for
    everything except checking that the installed script exists.
    """
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return _COMMANDS[args.command](args, out, err)
    except ConfigError as exc:
        print(f"config error: {exc}", file=err)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=err)
        return 2
    except NotImplementedError as exc:
        print(f"not implemented: {exc}", file=err)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
