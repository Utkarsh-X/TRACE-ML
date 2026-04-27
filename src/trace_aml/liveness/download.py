"""Download helper for the MiniFASNetV2 anti-spoofing ONNX model.

Run as:
    python -m trace_aml.liveness.download

Places the model at ``models/2.7_80x80_MiniFASNetV2.onnx`` (relative to
the current working directory, i.e. the project root).
"""

from __future__ import annotations

import os
import ssl
import sys
import urllib.request
from pathlib import Path


MODEL_FILENAME = "2.7_80x80_MiniFASNetV2.onnx"
DEST_DIR = Path("models")

# Mirrors in priority order — first successful download wins.
DOWNLOAD_URLS: list[str] = [
    # InsightFace CDN mirror
    "https://github.com/deepinsight/insightface/releases/download/v0.7/2.7_80x80_MiniFASNetV2.onnx",
    # Original minivision repo (may move)
    "https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.onnx",
    # Community mirror
    "https://github.com/computervisioneng/face-anti-spoofing/raw/main/models/2.7_80x80_MiniFASNetV2.onnx",
]

# Minimum expected file size (bytes) — rejects HTML error pages pretending to be models.
MIN_SIZE_BYTES = 200_000


def download(dest_path: Path | None = None) -> Path:
    """Download MiniFASNetV2 to *dest_path* (default: ``models/2.7…onnx``).

    Returns the path to the downloaded file.
    Raises ``RuntimeError`` if all mirrors fail.
    """
    dest = dest_path or (DEST_DIR / MODEL_FILENAME)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size >= MIN_SIZE_BYTES:
        print(f"Model already present at {dest}  ({dest.stat().st_size:,} bytes)")
        return dest

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for url in DOWNLOAD_URLS:
        print(f"Trying {url[:72]}...", end=" ", flush=True)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 TRACE-AML/1.0"},
            )
            with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                data = resp.read()

            if len(data) < MIN_SIZE_BYTES:
                print(f"SKIP (only {len(data):,} bytes — likely error page)")
                continue

            dest.write_bytes(data)
            print(f"OK  ({len(data):,} bytes)")
            print(f"\nSaved to: {dest.resolve()}")
            return dest

        except Exception as exc:
            print(f"FAIL ({exc})")

    raise RuntimeError(
        f"All download mirrors failed for {MODEL_FILENAME}.\n"
        "Please download it manually from:\n"
        "  https://github.com/minivision-ai/Silent-Face-Anti-Spoofing\n"
        f"and place it at: {(DEST_DIR / MODEL_FILENAME).resolve()}"
    )


if __name__ == "__main__":
    try:
        path = download()
        sys.exit(0)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
