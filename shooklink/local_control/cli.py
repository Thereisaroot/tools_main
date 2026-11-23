"""Command-line interface for the running ShookLink GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from .client import LocalControlClient, default_endpoint_path
from .updates import build_update_archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shooklinkctl",
        description="control the running ShookLink GUI",
    )
    parser.add_argument("--endpoint", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="show the live GUI connection state")
    file_parser = subparsers.add_parser("send-file", help="send a file")
    file_parser.add_argument("path", type=Path)
    for name, help_text in (
        ("pack-update", "build an offline ShookLink update ZIP"),
        ("send-update", "build an update ZIP and send it to the connected peer"),
    ):
        update_parser = subparsers.add_parser(name, help=help_text)
        update_parser.add_argument(
            "--repo", type=Path, default=Path("."),
            help="ShookLink source checkout (default: current directory)",
        )
        update_parser.add_argument(
            "--output-dir", type=Path,
            help="package directory (default: dist in the checkout)",
        )
    plain_parser = subparsers.add_parser("send-plain", help="send plain text")
    plain_parser.add_argument("text")
    secure_parser = subparsers.add_parser("send-secure", help="send secure text")
    secure_parser.add_argument("text")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[Path], LocalControlClient] = LocalControlClient,
) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        path = None
        if arguments.command in {"pack-update", "send-update"}:
            path = build_update_archive(arguments.repo, arguments.output_dir)
            print(f"prepared update: {path}", flush=True)
            if arguments.command == "pack-update":
                return 0
        elif arguments.command == "send-file":
            path = arguments.path.expanduser().resolve()

        client = client_factory(arguments.endpoint or default_endpoint_path())
        if arguments.command == "status":
            result = client.request("status")
            print(
                f"state: {result.get('state', 'unknown')}; "
                f"trusted: {str(bool(result.get('trusted'))).lower()}"
            )
        elif path is not None:
            result = client.request("send-file", {"path": str(path)})
            print(f"sent file: {result['name']} ({result['bytes']} bytes)")
        else:
            kind = "plain" if arguments.command == "send-plain" else "secure"
            client.request(arguments.command, {"text": arguments.text})
            print(f"sent {kind} text")
    except Exception as error:
        print(f"shooklinkctl: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
