from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline-sentinel-operator",
        description="Launch the local Pipeline Sentinel operator API and web console",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host; loopback is the safe default")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("outputs/operator"),
        help="Persistent uploads, jobs, runs, and fusion products",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum concurrent analytic jobs; default 1 protects workstation/GPU memory",
    )
    parser.add_argument(
        "--max-upload-gib",
        type=float,
        default=4.0,
        help="Maximum browser-uploaded media request size in GiB",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the operator console in the default browser after startup",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Explicitly allow a non-loopback bind. The MVP has no authentication.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        from .service import run_server
    except ModuleNotFoundError:
        print(
            "ERROR: operator service dependencies are not installed. "
            "Install Pipeline Sentinel with the 'operator' extra.",
            file=sys.stderr,
        )
        return 2

    try:
        run_server(
            host=args.host,
            port=args.port,
            workspace=args.workspace,
            max_workers=args.max_workers,
            max_upload_gib=args.max_upload_gib,
            open_browser=args.open_browser,
            allow_remote=args.allow_remote,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
