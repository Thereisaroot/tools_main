"""Command-line interface for the running ShookLink GUI."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
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
    probe = subparsers.add_parser("peer-update-status", help="challenge the actual remote GUI")
    probe.add_argument("--job", default="", help="observe a particular update transaction")
    keygen = subparsers.add_parser("release-keygen", help="create an offline release signing key")
    keygen.add_argument("private_key", type=Path)
    live = subparsers.add_parser("live-update", help="apply a signed release and verify its restart")
    live.add_argument("manifest", type=Path)
    live.add_argument("--timeout", type=int, default=180, help="total transaction timeout in seconds")
    file_parser = subparsers.add_parser("send-file", help="send a file")
    file_parser.add_argument("path", type=Path)
    for name, help_text in (
        ("pack-update", "build an offline ShookLink update ZIP"),
        ("send-update", "build an update ZIP and send it to the connected peer"),
        ("sign-update", "build and sign a release for managed receivers"),
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
        if name == "sign-update":
            update_parser.add_argument("--key", required=True, type=Path)
            update_parser.add_argument("--sequence", required=True, type=int)
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
        if arguments.command == "release-keygen":
            from shooklink.live_update.release import generate_key
            print(f"publisher public key: {generate_key(arguments.private_key.expanduser())}")
            return 0
        if arguments.command in {"pack-update", "send-update", "sign-update"}:
            path = build_update_archive(arguments.repo, arguments.output_dir)
            print(f"prepared update: {path}", flush=True)
            if arguments.command == "pack-update":
                return 0
            if arguments.command == "sign-update":
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
                from shooklink.live_update.release import Release, atomic_json, read_bounded

                key = Ed25519PrivateKey.from_private_bytes(read_bounded(arguments.key.expanduser(), 32))
                release = Release.sign(path, arguments.sequence, key)
                manifest = path.with_suffix(".release.json")
                atomic_json(manifest, release.as_dict())
                print(f"signed manifest: {manifest}")
                return 0
        elif arguments.command == "send-file":
            path = arguments.path.expanduser().resolve()

        client = client_factory(arguments.endpoint or default_endpoint_path())
        if arguments.command == "peer-update-status":
            result = client.request("peer-update-status", {"job": arguments.job})
            print(json.dumps(result, sort_keys=True))
        elif arguments.command == "live-update":
            from shooklink.live_update.release import Release, read_bounded

            manifest = arguments.manifest.expanduser().resolve()
            release = Release.parse(json.loads(read_bounded(manifest, 2048)))
            job = uuid.uuid4().hex
            print(f"update transaction: {job}", flush=True)
            result = client.request("live-update", {
                "job": job, "path": str(manifest.parent / (release.name + ".zip")),
                "release": release.as_dict(), "timeout": arguments.timeout,
            })
            print(f"{result['outcome']}: {result['revision']} (peer instance {result['instance']})")
        elif arguments.command == "status":
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
        detail = str(error)
        if arguments.command in {"live-update", "peer-update-status"} and detail.startswith("unknown control command:"):
            detail = "restart the controller GUI from a release with live update support"
        print(f"shooklinkctl: {detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
