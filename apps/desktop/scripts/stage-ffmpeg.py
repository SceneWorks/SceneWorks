#!/usr/bin/env python3
"""Stage the static ffmpeg binary for the Rust worker (sc-3767, epic 3482).

The Rust worker shells out to ffmpeg for frame sampling (person-track), frame
extraction, timeline export, and video-gen audio mux — all via
`media_jobs::run_ffmpeg`, which honors `SCENEWORKS_FFMPEG` (set by the desktop
`setup.rs`) else falls back to `ffmpeg` on PATH. The desktop ships no system
ffmpeg, so for a packaged, Python-free Mac (epic 3482's venv strip) we must
bundle the binary as an app resource rather than resolving it from the venv's
imageio-ffmpeg.

We stage the *exact* binary the desktop runs today: imageio-ffmpeg's bundled
`ffmpeg-macos-aarch64-vX` (an evermeet.cx GPLv3 static build). The macOS arm64
imageio-ffmpeg wheel is platform-tagged (`py3-none-macosx_*_arm64`) and bundles
the binary at `imageio_ffmpeg/binaries/ffmpeg-*`, so we download that wheel (a
zip) from a pinned PyPI CDN URL, verify its sha256, then extract the binary.
Pinned URL+sha256 (not just a version) so a compromised PyPI/CDN can't flow a
malicious binary into the signed+notarized release — mirrors the
cuda_provision.rs first-run downloader (sc-8864). Bump SHA256 in lockstep with
the version.

LICENSING: the extracted ffmpeg is GPLv3 (configured `--enable-gpl` +
`--enable-libx264/x265`, which the worker's libx264 mp4 encode requires — an
LGPL build is not an option). See docs/sc-3767/ffmpeg-bundling.md for provenance
and GPL compliance. Invoked by build-sidecar.mjs on macOS only.

USAGE: python3 stage-ffmpeg.py <dest-binary-path>
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import urllib.request
import zipfile

# Pin to the version the desktop venv currently resolves (imageio-ffmpeg 0.6.0 →
# ffmpeg 7.1 macos-aarch64). Bump in lockstep with requirements-mlx.txt so the
# bundled binary matches what dev/pre-bundle runs.
IMAGEIO_FFMPEG_VERSION = "0.6.0"

# Pinned macOS arm64 wheel: URL on the PyPI CDN (files.pythonhosted.org) plus the
# sha256 of the wheel, verified after download and BEFORE extraction. This is the
# `py3-none-macosx_11_0_arm64` wheel — the only arm64 build the desktop targets.
# The sha256 was resolved from the PyPI JSON API and confirmed by downloading the
# wheel and running `shasum -a 256` (sc-8864). Update URL+SHA256 together on any
# version bump; a mismatch fails the build loudly rather than shipping an
# unverified binary into a signed release.
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/40/5c/"
    "f3d8a657d362cc93b81aab8feda487317da5b5d31c0e1fdfd5e986e55d17/"
    "imageio_ffmpeg-0.6.0-py3-none-macosx_11_0_arm64.whl"
)
WHEEL_SHA256 = "b1ae3173414b5fc5f538a726c4e48ea97edc0d2cdc11f103afee655c463fa742"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: stage-ffmpeg.py <dest-binary-path>", file=sys.stderr)
        return 2
    dest = sys.argv[1]

    # Download the pinned wheel into memory, then verify its sha256 before we
    # trust any byte of it. Fail loudly on mismatch (never extract).
    with urllib.request.urlopen(WHEEL_URL) as resp:  # noqa: S310 (pinned https URL)
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != WHEEL_SHA256:
        print(
            "stage-ffmpeg: sha256 mismatch for "
            f"{WHEEL_URL}\n  expected {WHEEL_SHA256}\n  got      {digest}",
            file=sys.stderr,
        )
        return 1

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        bins = [
            n for n in zf.namelist()
            if "imageio_ffmpeg/binaries/ffmpeg" in n and not n.endswith("/")
        ]
        if not bins:
            print("stage-ffmpeg: wheel has no ffmpeg binary", file=sys.stderr)
            return 1
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with zf.open(bins[0]) as src, open(dest, "wb") as out:
            out.write(src.read())
    os.chmod(dest, 0o755)
    print(f"stage-ffmpeg: staged {dest} (imageio-ffmpeg {IMAGEIO_FFMPEG_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
