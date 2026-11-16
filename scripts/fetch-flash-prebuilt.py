#!/usr/bin/env python3
"""Fetch the verified FlashAttention archive for a Linux Candle Docker build."""

import hashlib
import json
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = "x86_64-unknown-linux-gnu"
FILES = {
    "libflashattention.a", "flash-attn-prebuilt.txt", "LICENSE-MIT",
    "LICENSE-APACHE", "LICENSE-FLASH-ATTENTION", "LICENSE-CUTLASS.txt", "AUTHORS-FLASH-ATTENTION",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(crate):
    digest = hashlib.sha256()
    files = [crate / "build.rs", crate / "Cargo.toml", *(crate / "kernels").iterdir(),
             *(crate / name for name in FILES if name.startswith(("LICENSE-", "AUTHORS-")))]
    files.sort(key=lambda path: path.relative_to(crate).as_posix())
    for file in files:
        if not file.is_file():
            continue
        digest.update(file.relative_to(crate).as_posix().encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def pinned_crate():
    output = subprocess.check_output(
        ["cargo", "metadata", "--locked", "--format-version", "1", "--features", "sceneworks-worker/backend-candle"],
        cwd=ROOT,
    )
    packages = json.loads(output)["packages"]
    matches = [package for package in packages if package["name"] == "candle-flash-attn"
               and "github.com/SceneWorks/inference" in (package["source"] or "")]
    if len(matches) != 1:
        raise ValueError("expected one pinned candle-flash-attn source")
    return pathlib.Path(matches[0]["manifest_path"]).parent


def trusted_archive(source):
    pinned = json.loads((ROOT / "config/flash-attn-prebuilt.json").read_text())
    expected = pinned.get(source, {}).get(TARGET)
    if expected is not None and not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("invalid trusted FlashAttention archive digest")
    return expected


def validate(directory, source, trusted_archive_sha256):
    entries = list(directory.iterdir())
    if {entry.name for entry in entries} != FILES or len(entries) != len(FILES):
        raise ValueError("FlashAttention package has missing or unexpected files")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("FlashAttention package contains a non-file")
    fields = {}
    for line in (directory / "flash-attn-prebuilt.txt").read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise ValueError("invalid FlashAttention manifest")
        fields[key] = value
    if fields.get("format") != "1" or fields.get("source_sha256") != source or fields.get("target") != TARGET:
        raise ValueError("FlashAttention manifest source or target mismatch")
    if sha256(directory / "libflashattention.a") != fields.get("archive_sha256") or fields.get("archive_sha256") != trusted_archive_sha256:
        raise ValueError("FlashAttention archive checksum mismatch against trusted source pin")


def fetch(crate, cache):
    if platform.machine() != "x86_64":
        return None
    source = source_sha256(crate)
    trusted = trusted_archive(source)
    if trusted is None:
        return None
    destination = cache / source / TARGET
    if destination.exists():
        validate(destination, source, trusted)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    asset = f"flash-attn-{TARGET}.tar.gz"
    url = f"https://github.com/SceneWorks/inference/releases/download/flash-attn-{source}/{asset}"
    with tempfile.TemporaryDirectory(prefix="flash-attn-", dir=destination.parent) as temp:
        temporary = pathlib.Path(temp)
        bundle_path = temporary / asset
        try:
            with urllib.request.urlopen(url, timeout=300) as response, bundle_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise
        unpacked = temporary / "unpacked"
        unpacked.mkdir()
        with tarfile.open(bundle_path, "r:gz") as bundle:
            members = bundle.getmembers()
            if len(members) != len(FILES) or {member.name for member in members} != FILES or any(not member.isfile() for member in members):
                raise ValueError("FlashAttention package has unsafe archive entries")
            for member in members:
                with bundle.extractfile(member) as source_file, (unpacked / member.name).open("wb") as output:
                    shutil.copyfileobj(source_file, output)
        validate(unpacked, source, trusted)
        try:
            unpacked.rename(destination)
        except FileExistsError:
            validate(destination, source, trusted)
    return destination


if __name__ == "__main__":
    try:
        directory = fetch(pinned_crate(), pathlib.Path.home() / ".cache/sceneworks/flash-attn")
        if directory:
            print(directory)
        else:
            print("FlashAttention prebuilt unavailable; compiling from source", file=sys.stderr)
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError, urllib.error.URLError) as error:
        print(f"FlashAttention prebuilt: {error}", file=sys.stderr)
        sys.exit(2)
