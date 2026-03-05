from __future__ import annotations

import argparse
import hashlib
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from shutil import which

try:
    import certifi
except Exception:
    certifi = None

# Try several candidate URLs so script stays usable if one asset moves.
CANDIDATE_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, output: Path, *, context: ssl.SSLContext | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60, context=context) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        with output.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def download_with_curl(url: str, output: Path, *, insecure: bool = False) -> None:
    if which("curl") is None:
        raise RuntimeError("curl not found")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "--fail", "--connect-timeout", "20", "-o", str(output), url]
    if insecure:
        cmd.insert(1, "-k")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "curl failed"
        raise RuntimeError(stderr)


def _download_with_fallbacks(url: str, output: Path, *, insecure: bool = False) -> None:
    # 1) Default SSL trust store
    try:
        download(url, output)
        return
    except Exception as exc_default:
        msg = str(exc_default)
        ssl_related = "CERTIFICATE_VERIFY_FAILED" in msg or "certificate verify failed" in msg

        # 2) certifi CA bundle
        if ssl_related and certifi is not None:
            try:
                ctx = ssl.create_default_context(cafile=certifi.where())
                download(url, output, context=ctx)
                return
            except Exception:
                pass

        # 3) curl fallback (often works with system keychain)
        try:
            download_with_curl(url, output, insecure=insecure)
            return
        except Exception as exc_curl:
            raise RuntimeError(f"{exc_default} | curl fallback: {exc_curl}") from exc_curl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download default ONNX person detector model to models/person_detector.onnx"
    )
    parser.add_argument(
        "--output",
        default="models/person_detector.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output already exists",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for curl fallback (last resort)",
    )
    args = parser.parse_args()

    output = Path(args.output)

    if output.exists() and not args.force:
        print(f"[model] exists: {output}")
        print(f"[model] sha256: {sha256_file(output)}")
        return 0

    errors: list[str] = []
    for url in CANDIDATE_URLS:
        try:
            print(f"[model] trying: {url}")
            _download_with_fallbacks(url, output, insecure=args.insecure)
            print(f"[model] downloaded: {output}")
            print(f"[model] sha256: {sha256_file(output)}")
            return 0
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            errors.append(f"{url} -> {exc}")
            continue

    print("[model] failed to download from all candidate URLs", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    print(
        "[model] Please place a YOLO ONNX file manually at models/person_detector.onnx",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
