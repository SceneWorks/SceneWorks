#!/usr/bin/env python3
"""Stage the CoreML-enabled onnxruntime dylib for the Rust DWPose detector (sc-3487).

The Rust worker links `ort` with `load-dynamic`, so it dlopens onnxruntime at
runtime from `ORT_DYLIB_PATH` (set by the desktop `setup.rs`). For a packaged,
Python-free Mac we must bundle that dylib. The macOS arm64 onnxruntime PyPI wheel
ships exactly the CoreML-enabled `libonnxruntime.<ver>.dylib` the Python rtmlib path
uses (and that ort 2.0.0-rc.12's ORT API 24 accepts), so we download that wheel (a
zip) from a pinned PyPI CDN URL, verify its sha256, and extract its
`capi/libonnxruntime*.dylib`.

onnxruntime wheels are CPython-ABI-tagged (cp311/cp312/cp313/cp314), so unlike
imageio-ffmpeg's single `py3-none` wheel we pin a small URL+sha256 table keyed by
the build host's interpreter tag and verify against the matching entry. Pinning
URL+sha256 (not just a version) keeps a compromised PyPI/CDN from flowing a
malicious dylib into the signed+notarized release — mirrors cuda_provision.rs
(sc-8864). Bump the table in lockstep with the `ort` crate. Invoked by
build-sidecar.mjs on macOS only.

USAGE: python3 stage-onnxruntime.py <dest-dylib-path>
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import urllib.request
import zipfile

# ORT 2.0.0-rc.12 requests ONNX Runtime API 24 (>= onnxruntime 1.24); 1.26.0 is the
# version validated in the sc-3487 spike. Bump in lockstep with the `ort` crate.
ONNXRUNTIME_VERSION = "1.26.0"

# Pinned macOS arm64 wheels keyed by CPython ABI tag. Each entry is the PyPI CDN
# (files.pythonhosted.org) URL plus the sha256 of the wheel, verified after
# download and BEFORE extraction. onnxruntime ships one wheel per interpreter, so
# we select the entry matching the build host's `python3` and pin all supported
# tags. The sha256s were resolved from the PyPI JSON API and confirmed by
# downloading each wheel and running `shasum -a 256` (sc-8864). Update URL+SHA256
# together on any version bump; a mismatch fails the build loudly rather than
# shipping an unverified dylib into a signed release.
WHEELS: dict[str, dict[str, str]] = {
    "cp311": {
        "url": (
            "https://files.pythonhosted.org/packages/d4/81/"
            "29a9eb470994a75eb7b3ccf32be314d7c66675a00ac7b50294816cc2db27/"
            "onnxruntime-1.26.0-cp311-cp311-macosx_14_0_arm64.whl"
        ),
        "sha256": "ee1109ef4ef27cad90e823399e61e03b3c6c7bfe0fb820b4baf3678c15be8b3c",
    },
    "cp312": {
        "url": (
            "https://files.pythonhosted.org/packages/81/b1/"
            "d111b1df656761f980d9e298a60039a9cb66036b1d039e777537743d0ac3/"
            "onnxruntime-1.26.0-cp312-cp312-macosx_14_0_arm64.whl"
        ),
        "sha256": "05b028781b322ad74b57ce5b50aa5280bb1fe96ceec334628ade681e0b24c1ac",
    },
    "cp313": {
        "url": (
            "https://files.pythonhosted.org/packages/cf/a2/"
            "c801242685e0ce48a4ca51dfafbb588765e0446397e123be53ba5598f3f5/"
            "onnxruntime-1.26.0-cp313-cp313-macosx_14_0_arm64.whl"
        ),
        "sha256": "ccce19c5f771b8268902f77d9fed9e88f9499465d6780808faa6611a789d33f0",
    },
    "cp314": {
        "url": (
            "https://files.pythonhosted.org/packages/40/89/"
            "17546c1c20f6bfc3ae41c22152378a26edfea918af3129e2139dcd7c99f3/"
            "onnxruntime-1.26.0-cp314-cp314-macosx_14_0_arm64.whl"
        ),
        "sha256": "33a791f31432a3af1a96db5e54818b37aba5e5eefc2e6af5794c10a9118a9993",
    },
}


def _abi_tag() -> str:
    """CPython ABI tag for the running interpreter, e.g. `cp312`."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: stage-onnxruntime.py <dest-dylib-path>", file=sys.stderr)
        return 2
    dest = sys.argv[1]

    tag = _abi_tag()
    wheel = WHEELS.get(tag)
    if wheel is None:
        print(
            f"stage-onnxruntime: no pinned onnxruntime {ONNXRUNTIME_VERSION} wheel "
            f"for interpreter {tag} — add its URL+sha256 to WHEELS "
            f"(have: {', '.join(sorted(WHEELS))})",
            file=sys.stderr,
        )
        return 1

    # Download the pinned wheel into memory, then verify its sha256 before we
    # trust any byte of it. Fail loudly on mismatch (never extract).
    with urllib.request.urlopen(wheel["url"]) as resp:  # noqa: S310 (pinned https URL)
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != wheel["sha256"]:
        print(
            "stage-onnxruntime: sha256 mismatch for "
            f"{wheel['url']}\n  expected {wheel['sha256']}\n  got      {digest}",
            file=sys.stderr,
        )
        return 1

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        dylibs = [n for n in zf.namelist() if n.endswith(".dylib") and "libonnxruntime" in n]
        if not dylibs:
            print("stage-onnxruntime: wheel has no libonnxruntime dylib", file=sys.stderr)
            return 1
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with zf.open(dylibs[0]) as src, open(dest, "wb") as out:
            out.write(src.read())
    os.chmod(dest, 0o755)
    print(
        f"stage-onnxruntime: staged {dest} "
        f"(onnxruntime {ONNXRUNTIME_VERSION}, {tag})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
