"""Command-line interface for the running ShookLink GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from .client import LocalControlClient, default_endpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shooklinkctl",
        description="control the running ShookLink GUI",
    )
    parser.add_argument("--endpoint", type=Path, default=default_endpoint_path())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="show the live GUI connection state")
    file_parser = subparsers.add_parser("send-file", help="send a file")
    file_parser.add_argument("path", type=Path)
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
    client = client_factory(arguments.endpoint)
    try:
        if arguments.command == "status":
            result = client.request("status")
            print(
                f"state: {result.get('state', 'unknown')}; "
                f"trusted: {str(bool(result.get('trusted'))).lower()}"
            )
        elif arguments.command == "send-file":
            result = client.request(
                "send-file", {"path": str(arguments.path.expanduser().resolve())}
            )
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
