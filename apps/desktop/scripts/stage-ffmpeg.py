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
the binary at `imageio_ffmpeg/binaries/ffmpeg-*`, so we pip-download that wheel
(a zip) and extract it. Pinned to a known version so the bundle is deterministic.

LICENSING: the extracted ffmpeg is GPLv3 (configured `--enable-gpl` +
`--enable-libx264/x265`, which the worker's libx264 mp4 encode requires — an
LGPL build is not an option). See docs/sc-3767/ffmpeg-bundling.md for provenance
and GPL compliance. Invoked by build-sidecar.mjs on macOS only.

USAGE: python3 stage-ffmpeg.py <dest-binary-path>
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import zipfile

# Pin to the version the desktop venv currently resolves (imageio-ffmpeg 0.6.0 →
# ffmpeg 7.1 macos-aarch64). Bump in lockstep with requirements-mlx.txt so the
# bundled binary matches what dev/pre-bundle runs.
IMAGEIO_FFMPEG_VERSION = "0.6.0"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: stage-ffmpeg.py <dest-binary-path>", file=sys.stderr)
        return 2
    dest = sys.argv[1]

    with tempfile.TemporaryDirectory() as tmp:
        # Let pip pick the wheel matching this (macOS arm64) build host; the
        # binary is bundled inside the platform-tagged wheel.
        subprocess.run(
            [
                sys.executable, "-m", "pip", "download",
                f"imageio-ffmpeg=={IMAGEIO_FFMPEG_VERSION}",
                "--only-binary=:all:", "--no-deps", "-d", tmp,
            ],
            check=True,
        )
        wheels = glob.glob(os.path.join(tmp, "imageio_ffmpeg-*.whl"))
        if not wheels:
            print("stage-ffmpeg: no imageio-ffmpeg wheel downloaded", file=sys.stderr)
            return 1
        with zipfile.ZipFile(wheels[0]) as zf:
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
