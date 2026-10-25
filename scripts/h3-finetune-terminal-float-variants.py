#!/usr/bin/env python3
"""Prepare real F16/F32 source variants of an already-staged BF16 H3 transformer.

The canonical ``generalization.dense-f16/bf16/f32`` API cases need actual tensors at
those precisions. This step streams the REAL raw BF16 shards that the stager already
copied under its private root and derives two new source variants:

  <privateRoot>/derived/dense-f16    BF16 tensors -> IEEE binary16
  <privateRoot>/derived/dense-f32    BF16 tensors -> IEEE binary32 (exact bit shift)

The BF16 source itself is the dense-bf16 variant; it is not copied.

The 16-file closure (config + 14 shards + index) is derived from the trusted casebook
``docs/testing/sc-26691-h3-cases.json`` -> ``stagingAssets[<explicit set id>]``. It is
never a cache glob and never a second hand-maintained registry. The four admitted ids are
``builtin-shards``, ``builtin-ref-shards``, ``community-shards-fl2va`` and
``community-shards-ref2va``. Packed q4/q8/INT8/W4 sources are refused.

Rules this tool keeps:
  * Only BF16 tensors are converted. Every other dtype/buffer (F32, F16, I8, ...) is
    copied byte-for-byte. All keys, shapes and ``__metadata__`` are preserved.
  * Tensors are streamed in bounded chunks; no full tensor or model is materialized.
  * BF16 -> F32 is the exact bit shift ``u16 << 16``. BF16 -> F16 uses NumPy's IEEE
    round-to-nearest-even cast; finite values that would become Inf are refused.
  * The source index's weight_map must be complete and exact across all shards: a
    duplicate tensor, a missing tensor, or a dropped key refuses the run.
  * Output hashes/sizes are measured after the bytes exist; nothing is predicted.

Run under ``python3 -I`` in the parent workflow:

  python3 -I scripts/h3-finetune-terminal-float-variants.py \
      --stage-receipt /abs/stage-receipt.json \
      --stage-root /abs/private-root \
      --source-profile builtin-shards \
      [--casebook docs/testing/sc-26691-h3-cases.json] \
      [--output-root /abs/private-root/derived] [--chunk-bytes 8388608]

``numpy==2.2.6`` is a runner dependency only (private venv under ``$RUNNER_TEMP``); it is
imported lazily inside the production converter, so importing this module for tests never
needs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import sys
from pathlib import Path

DEFAULT_CASEBOOK = Path(__file__).resolve().parents[1] / "docs" / "testing" / "sc-26691-h3-cases.json"
PROFILE_PARTITIONS = {
    "builtin-shards": "FL2VA",
    "builtin-ref-shards": "Ref2VA",
    "community-shards-fl2va": "FL2VA",
    "community-shards-ref2va": "Ref2VA",
}
SOURCE_PROFILE_IDS = tuple(PROFILE_PARTITIONS)
VARIANTS = {"F16": "dense-f16", "F32": "dense-f32"}
NUMPY_VERSION = "2.2.6"
NUMPY_REQUIRED = f"numpy=={NUMPY_VERSION}"
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
MAX_CHUNK_BYTES = 64 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024 * 1024
ALLOWED_MAC_RUNNERS = frozenset({"nax-macos-2"})
ALLOWED_WINDOWS_RUNNERS = frozenset({"cuda-windows", "cuda-windows-2"})
PACKED_SUFFIXES = (".comfy_quant", ".weight_scale", ".weight_s_rel")
KNOWN_AUTHORITY = ("kind", "approvedBy", "approvedAt", "reference", "subjects", "acknowledged")
DENSE_ITEMSIZE = {"F16": 2, "F32": 4}


class FloatVariantError(RuntimeError):
    """A refusal with a stable code the workflow can report."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> None:
    raise FloatVariantError(code, message)


def _require_nonempty(value, code: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code, f"{label} must be a non-empty string")
    return value.strip()


def _require_absolute(value, label: str) -> str:
    if not isinstance(value, str) or not os.path.isabs(value):
        _fail("path", f"{label} must be an absolute path")
    return os.path.normpath(value)


def _real_directory(path, label: str) -> str:
    real = os.path.realpath(_require_nonempty(os.fspath(path), "path", label))
    if not os.path.isdir(real):
        _fail("path", f"{label} is not a directory: {real}")
    return real


def _inside(root: str, candidate: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(root), os.path.realpath(candidate)]) == os.path.realpath(root)
    except ValueError:
        return False


def _confined_absolute_file(root: str, path, label: str) -> str:
    candidate = _require_absolute(path, label)
    real = os.path.realpath(candidate)
    if real == os.path.realpath(root) or not _inside(root, real):
        _fail("path", f"{label} escapes the private root")
    if os.path.islink(candidate):
        _fail("path", f"{label} is a symlink; staged files must be regular")
    try:
        mode = os.stat(candidate, follow_symlinks=False).st_mode
    except OSError as exc:
        _fail("path", f"{label} is missing: {exc}")
    if not stat.S_ISREG(mode):
        _fail("path", f"{label} is not a regular file")
    return real


def _new_directory_under(root: str, path, label: str) -> str:
    target = _require_absolute(os.fspath(path), label)
    real_root = os.path.realpath(root)
    parent = os.path.realpath(os.path.dirname(target))
    if parent != real_root and not _inside(real_root, parent):
        _fail("path", f"{label} parent escapes the private root")
    if os.path.lexists(target):
        _fail("output", f"{label} already exists; this step creates a NEW owned directory")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if not _inside(real_root, os.path.realpath(os.path.dirname(target))):
        _fail("path", f"{label} parent escapes the private root")
    os.mkdir(target, 0o700)
    return os.path.realpath(target)


def _hash_file(path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _sha256(value, code: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
        _fail(code, f"{label} must be a full 64-hex SHA256")
    return value.lower()


def load_casebook(path=None) -> dict:
    real = os.path.realpath(_require_absolute(os.fspath(path or DEFAULT_CASEBOOK), "casebook path"))
    if os.path.islink(os.fspath(path or DEFAULT_CASEBOOK)):
        _fail("path", "casebook must not be a symlink")
    try:
        with open(real, "r", encoding="utf-8") as handle:
            book = json.load(handle)
    except OSError as exc:
        _fail("casebook", f"cannot read casebook: {exc}")
    except json.JSONDecodeError as exc:
        _fail("casebook", f"casebook is not JSON: {exc}")
    if not isinstance(book, dict):
        _fail("casebook", "casebook must be a JSON object")
    return book


def _casebook_prefix(files: list[str]) -> str:
    directories = {os.path.dirname(relative.replace("\\", "/")) for relative in files}
    if len(directories) != 1:
        _fail("profile", "selected staging set is not one directory")
    prefix = directories.pop()
    if any(part in ("", ".", "..") for part in prefix.split("/") if prefix):
        _fail("profile", "selected staging set has an unsafe prefix")
    return prefix


def profile_from_casebook(casebook: dict, source_profile: str) -> dict:
    """Derive the exact 16-file closure, pins, sizes and legal metadata from the casebook."""
    if source_profile not in PROFILE_PARTITIONS:
        _fail("profile", f"unknown source profile {source_profile!r}; admitted ids are {list(SOURCE_PROFILE_IDS)}")
    assets = casebook.get("stagingAssets")
    if not isinstance(assets, dict):
        _fail("casebook", "casebook stagingAssets object required")
    rows = assets.get(source_profile)
    if not isinstance(rows, list) or not rows:
        _fail("casebook", f"casebook stagingAssets[{source_profile!r}] missing")
    if len(rows) != 16:
        _fail("profile", f"{source_profile} must declare exactly 16 files, found {len(rows)}")
    repos, revisions, files, table = set(), set(), [], {}
    for row in rows:
        if not isinstance(row, dict):
            _fail("casebook", f"{source_profile} staging row must be an object")
        if row.get("repoType") != "model":
            _fail("profile", f"{source_profile} must be a model-repo set")
        repo = _require_nonempty(row.get("repo"), "profile", "staging repo")
        revision = _require_nonempty(row.get("revision"), "profile", "staging revision")
        relative = _require_nonempty(row.get("file"), "profile", "staging file")
        if relative.startswith("/") or "\\" in relative or ".." in relative.split("/"):
            _fail("profile", f"unsafe staging file {relative!r}")
        if any(token in relative.lower() for token in ("q4", "q8", "int8", "w4a8")):
            _fail("profile", "packed q4/q8/INT8/W4 profiles are not admitted sources for float variants")
        expected_bytes = row.get("expectedBytes")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            _fail("profile", f"{relative} expectedBytes must be positive")
        expected_sha = _sha256(row.get("expectedSha256"), "profile", f"{relative} expectedSha256")
        legal = row.get("legal")
        if not isinstance(legal, dict) or not isinstance(legal.get("licenseId"), str) or not legal["licenseId"].strip():
            _fail("profile", f"{relative} legal.licenseId required")
        legal_copy = {k: legal[k] for k in ("licenseId", "inheritedLicenseId", "attributionRequired") if k in legal}
        repos.add(repo)
        revisions.add(revision)
        files.append(relative)
        table[relative] = {"expectedBytes": expected_bytes, "expectedSha256": expected_sha, "legal": legal_copy}
    if len(repos) != 1 or len(revisions) != 1:
        _fail("profile", "selected staging set must share one repo and revision")
    prefix = _casebook_prefix(files)
    expected_names = {
        f"{prefix}/config.json" if prefix else "config.json",
        *[f"{prefix}/diffusion_pytorch_model-{index:05d}-of-00014.safetensors" if prefix else f"diffusion_pytorch_model-{index:05d}-of-00014.safetensors" for index in range(1, 15)],
        f"{prefix}/diffusion_pytorch_model.safetensors.index.json" if prefix else "diffusion_pytorch_model.safetensors.index.json",
    }
    if set(files) != expected_names:
        _fail("profile", "selected set is not the exact config + 14 shards + index closure")
    meta = ((casebook.get("stagingPlan") or {}).get("profileSemantics") or {}).get(source_profile) or {}
    partition = meta.get("partition") or PROFILE_PARTITIONS[source_profile]
    ordered = [f"{prefix}/config.json" if prefix else "config.json"]
    ordered += [f"{prefix}/diffusion_pytorch_model-{index:05d}-of-00014.safetensors" if prefix else f"diffusion_pytorch_model-{index:05d}-of-00014.safetensors" for index in range(1, 15)]
    ordered.append(f"{prefix}/diffusion_pytorch_model.safetensors.index.json" if prefix else "diffusion_pytorch_model.safetensors.index.json")
    return {
        "setId": source_profile,
        "repo": repos.pop(),
        "revision": revisions.pop(),
        "prefix": prefix,
        "partition": partition,
        "files": ordered,
        "rows": table,
    }


def verify_stage_receipt(receipt, stage_root: str, profile: dict, allow_unit_fake: bool = False) -> dict:
    """Validate the real stager receipt against EVERY casebook row (pins, sizes, legal)."""
    if not isinstance(receipt, dict):
        _fail("receipt", "stage receipt must be a JSON object")
    if receipt.get("kind") != "h3-finetune-stage-receipt":
        _fail("receipt", "stage receipt kind must be 'h3-finetune-stage-receipt'")
    if receipt.get("version") != 1:
        _fail("receipt", "stage receipt version must be 1")
    if receipt.get("status") != "completed":
        _fail("receipt", "stage receipt status must be 'completed'")
    proof_kind = receipt.get("proof_kind")
    if proof_kind not in ("frozen_stage_content", "unit_fake"):
        _fail("proof", "stage receipt proof_kind must be 'frozen_stage_content'")
    if proof_kind == "unit_fake" and not allow_unit_fake:
        _fail("proof", "unit_fake stage receipts cannot prepare campaign source variants")
    settings = receipt.get("settings")
    if not isinstance(settings, dict):
        _fail("receipt", "stage receipt settings object required")
    declared_root = _require_absolute(settings.get("privateRoot"), "settings.privateRoot")
    real_root = os.path.realpath(stage_root)
    if os.path.realpath(declared_root) != real_root:
        _fail("receipt", "settings.privateRoot differs from --stage-root")
    set_record = receipt.get("set")
    if not isinstance(set_record, dict) or not isinstance(set_record.get("ids"), list):
        _fail("receipt", "stage receipt set.ids array required")
    if profile["setId"] not in set_record["ids"]:
        _fail("profile", f"stage receipt did not select the {profile['setId']} asset set")
    authority = receipt.get("licenseAuthority")
    if not isinstance(authority, dict):
        _fail("license", "root licenseAuthority object required")
    kind = authority.get("kind")
    if kind != "user-approved" and not (allow_unit_fake and kind == "unit_fake"):
        _fail("license", "root licenseAuthority.kind must be 'user-approved'")
    if authority.get("acknowledged") is not True:
        _fail("license", "root licenseAuthority.acknowledged must be true")
    for key, label in (("approvedBy", "licenseAuthority.approvedBy"), ("reference", "licenseAuthority.reference")):
        _require_nonempty(authority.get(key), "license", label)
    if not isinstance(authority.get("approvedAt"), str) or not authority["approvedAt"].strip():
        _fail("license", "licenseAuthority.approvedAt required")
    if not isinstance(authority.get("subjects"), list):
        _fail("license", "licenseAuthority.subjects array required")
    files = receipt.get("files")
    if not isinstance(files, list):
        _fail("receipt", "stage receipt files array required")
    selected = []
    for relative in profile["files"]:
        expected = profile["rows"][relative]
        matches = [
            row for row in files
            if isinstance(row, dict)
            and row.get("repo") == profile["repo"]
            and row.get("revision") == profile["revision"]
            and row.get("file") == relative
        ]
        if len(matches) != 1:
            _fail("receipt", f"expected exactly one staged row for {relative}, found {len(matches)}")
        row = matches[0]
        if row.get("repoType") != "model":
            _fail("receipt", f"{relative} must be a model-repo row")
        if row.get("status") != "completed":
            _fail("receipt", f"{relative} status must be 'completed'")
        if row.get("actualBytes") != expected["expectedBytes"]:
            _fail("input", f"{relative} actualBytes must be the casebook {expected['expectedBytes']}")
        observed = _sha256(row.get("observedSha256"), "input", f"{relative} observedSha256")
        receipt_expected = _sha256(row.get("expectedSha256"), "input", f"{relative} expectedSha256")
        if receipt_expected != expected["expectedSha256"] or observed != expected["expectedSha256"]:
            _fail("input", f"{relative} pinned SHA does not match the casebook")
        final_path = _confined_absolute_file(real_root, row.get("finalPath"), f"{relative} finalPath")
        legal = row.get("expectedLegal")
        if not isinstance(legal, dict) or legal.get("licenseId") != expected["legal"]["licenseId"]:
            _fail("license", f"{relative} expectedLegal.licenseId must match the casebook")
        for key in ("inheritedLicenseId", "attributionRequired"):
            if key in expected["legal"] and legal.get(key) != expected["legal"][key]:
                _fail("license", f"{relative} expectedLegal.{key} must match the casebook")
        acknowledgment = row.get("explicitAcknowledgment")
        if not isinstance(acknowledgment, dict) or acknowledgment.get("acknowledged") is not True:
            _fail("license", f"{relative} explicit acknowledged licence required")
        if acknowledgment.get("licenseId") != expected["legal"]["licenseId"]:
            _fail("license", f"{relative} acknowledgment licenseId must match the casebook")
        if _sha256(acknowledgment.get("sha256"), "license", f"{relative} acknowledgment.sha256") != expected["expectedSha256"]:
            _fail("license", f"{relative} acknowledgment.sha256 must match the casebook")
        if "attributionRequired" in expected["legal"]:
            attribution = acknowledgment.get("attribution")
            if not isinstance(attribution, str) or expected["legal"]["attributionRequired"] not in attribution:
                _fail("license", f"{relative} acknowledgment must carry the required attribution")
        selected.append({
            "relative": relative,
            "basename": os.path.basename(relative),
            "finalPath": final_path,
            "expectedBytes": expected["expectedBytes"],
            "expectedSha256": expected["expectedSha256"],
            "observedSha256": observed,
            "expectedLegal": dict(expected["legal"]),
            "explicitAcknowledgment": {
                k: acknowledgment[k]
                for k in ("sha256", "acknowledged", "licenseId", "inheritedLicenseId", "inheritedAcknowledged", "attribution")
                if k in acknowledgment
            },
        })
    return {
        "profile": profile,
        "root": real_root,
        "files": selected,
        "licenseAuthority": {k: authority[k] for k in KNOWN_AUTHORITY if k in authority},
        "proofKind": proof_kind,
    }


def read_safetensors_header(path: str) -> tuple[dict, int, int]:
    """Read only the bounded JSON header; the data section is never loaded here."""
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            _fail("input", f"{os.path.basename(path)} is too short to be safetensors")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > MAX_HEADER_BYTES:
            _fail("input", f"{os.path.basename(path)} header length {header_len} is out of bounds")
        body = handle.read(header_len)
        if len(body) != header_len:
            _fail("input", f"{os.path.basename(path)} header is truncated")
    try:
        header = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail("input", f"{os.path.basename(path)} header is not valid JSON: {exc}")
    if not isinstance(header, dict):
        _fail("input", f"{os.path.basename(path)} header must be a JSON object")
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            _fail("input", f"{os.path.basename(path)} tensor {name} entry must be an object")
        for key in ("dtype", "shape", "data_offsets"):
            if key not in entry:
                _fail("input", f"{os.path.basename(path)} tensor {name} is missing {key}")
        offsets = entry["data_offsets"]
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(isinstance(v, int) and v >= 0 for v in offsets) and offsets[0] <= offsets[1]):
            _fail("input", f"{os.path.basename(path)} tensor {name} has invalid data_offsets")
        if any(name.endswith(suffix) for suffix in PACKED_SUFFIXES):
            _fail("input", f"{os.path.basename(path)} contains packed quantization tensor {name}; not an admitted FP source")
    ranges = sorted(entry["data_offsets"] for name, entry in header.items() if name != "__metadata__")
    for (_, end), (start, _) in zip(ranges, ranges[1:]):
        if end > start:
            _fail("input", f"{os.path.basename(path)} tensor data ranges overlap")
    return header, 8 + header_len, header_len


def _tensor_bytes(entry: dict) -> int:
    start, end = entry["data_offsets"]
    return end - start


def _elements(shape) -> int:
    total = 1
    for value in shape:
        if not isinstance(value, int) or value < 0:
            _fail("input", "tensor shape must be non-negative integers")
        total *= value
    return total


def build_variant_header(header: dict, target_dtype: str) -> tuple[dict, list[tuple[str, dict, int, int]], int]:
    """Return (output_header, ordered_tensors, total_tensor_bytes)."""
    output = {}
    if "__metadata__" in header:
        metadata = header["__metadata__"]
        if not isinstance(metadata, dict):
            _fail("input", "__metadata__ must be an object")
        output["__metadata__"] = metadata
    ordered = sorted(
        ((name, entry) for name, entry in header.items() if name != "__metadata__"),
        key=lambda item: (item[1]["data_offsets"][0], item[0]),
    )
    offset = 0
    result = []
    for name, entry in ordered:
        copied = dict(entry)
        source_dtype = entry["dtype"]
        if source_dtype == "BF16":
            elements = _elements(entry["shape"])
            if _tensor_bytes(entry) != elements * 2:
                _fail("input", f"BF16 tensor {name} byte length does not match its shape")
            size = elements * DENSE_ITEMSIZE[target_dtype]
            copied["dtype"] = target_dtype
        else:
            size = _tensor_bytes(entry)
        copied["data_offsets"] = [offset, offset + size]
        output[name] = copied
        result.append((name, entry, offset, size))
        offset += size
    return output, result, offset


def serialize_header(header: dict) -> bytes:
    body = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body += b" " * ((-(8 + len(body))) % 8)
    return struct.pack("<Q", len(body)) + body


def _zero_stats() -> dict:
    return {"precisionLoss": 0, "underflow": 0, "nan": 0, "inf": 0}


def _numpy_version() -> str:
    try:
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        _fail("tool", f"{NUMPY_REQUIRED} required in the runner venv under $RUNNER_TEMP ({exc})")
    version = getattr(numpy, "__version__", None)
    if version != NUMPY_VERSION:
        _fail("tool", f"{NUMPY_REQUIRED} required; runner has numpy {version!r}")
    return version


def numpy_converter(chunk: bytes, source_dtype: str, target_dtype: str) -> tuple[bytes, dict]:
    """Production BF16 converter. Imported lazily; exact F32 shift and IEEE F16 cast."""
    import numpy as np  # type: ignore[import-not-found]

    if getattr(np, "__version__", None) != NUMPY_VERSION:
        _fail("tool", f"{NUMPY_REQUIRED} required; runner has numpy {getattr(np, '__version__', None)!r}")
    if source_dtype != "BF16":
        return bytes(chunk), _zero_stats()
    if len(chunk) % 2 != 0:
        _fail("input", "BF16 chunk must contain whole 16-bit words")
    words = np.frombuffer(chunk, dtype="<u2")
    bits = words.astype(np.uint32) << np.uint32(16)
    if target_dtype == "F32":
        return bits.astype("<u4").tobytes(), _zero_stats()
    if target_dtype != "F16":
        _fail("input", f"unsupported target dtype {target_dtype}")
    values = bits.view(np.float32)
    halves = values.astype(np.float16)
    finite = np.isfinite(values)
    if np.any(finite & np.isinf(halves)):
        _fail("overflow", "a finite BF16 value overflows IEEE binary16")
    rounded = halves.astype(np.float32)
    return halves.tobytes(), {
        "precisionLoss": int(np.count_nonzero(finite & (rounded != values))),
        "underflow": int(np.count_nonzero((values != 0) & (halves == 0))),
        "nan": int(np.count_nonzero(np.isnan(values))),
        "inf": int(np.count_nonzero(np.isinf(values))),
    }


def _add_stats(total: dict, delta: dict) -> None:
    for key in ("precisionLoss", "underflow", "nan", "inf"):
        total[key] += int(delta.get(key, 0))


def convert_shard(source_path: str, destination: str, target_dtype: str, converter, chunk_bytes: int, stats: dict) -> dict:
    """Stream one source shard into one variant. Never loads a whole tensor."""
    header, data_start, _ = read_safetensors_header(source_path)
    variant_header, ordered, total = build_variant_header(header, target_dtype)
    histogram = {}
    for name, entry in variant_header.items():
        if name == "__metadata__":
            continue
        histogram[entry["dtype"]] = histogram.get(entry["dtype"], 0) + 1
    with open(source_path, "rb") as source, open(destination, "xb") as output:
        output.write(serialize_header(variant_header))
        for name, entry, _output_offset, output_size in ordered:
            source.seek(data_start + entry["data_offsets"][0])
            remaining = _tensor_bytes(entry)
            written = 0
            while remaining > 0:
                take = min(chunk_bytes, remaining)
                if entry["dtype"] == "BF16":
                    take -= take % 2
                    if take == 0:
                        take = min(2, remaining)
                chunk = source.read(take)
                if len(chunk) != take:
                    _fail("input", f"{os.path.basename(source_path)} tensor {name} data is truncated")
                if entry["dtype"] == "BF16":
                    converted, delta = converter(chunk, "BF16", target_dtype)
                    if len(converted) != take * DENSE_ITEMSIZE[target_dtype] // 2:
                        _fail("tool", f"converter returned {len(converted)} bytes for a {take}-byte BF16 chunk")
                    output.write(converted)
                    written += len(converted)
                    _add_stats(stats, delta)
                else:
                    output.write(chunk)
                    written += len(chunk)
                remaining -= len(chunk)
            if written != output_size:
                _fail("tool", f"tensor {name} wrote {written} bytes, expected {output_size}")
    stats["tensorBytes"] += total
    stats["tensors"] += len(ordered)
    for dtype_name, count in histogram.items():
        stats["dtypeHistogram"][dtype_name] = stats["dtypeHistogram"].get(dtype_name, 0) + count
    return stats


def validate_index_closure(shards: list[dict], index_path: str) -> dict:
    """Require the index weight_map to be the exact, complete, non-duplicated shard tensor set."""
    try:
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("input", f"source index is not readable JSON: {exc}")
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        _fail("input", "source index must contain weight_map")
    weight_map = index["weight_map"]
    owners = {}
    for shard in shards:
        header, _, _ = read_safetensors_header(shard["finalPath"])
        basename = shard["basename"]
        for name in header:
            if name == "__metadata__":
                continue
            if name in owners:
                _fail("input", f"duplicate tensor {name} in {owners[name]} and {basename}")
            owners[name] = basename
    missing = sorted(set(weight_map) - set(owners))
    extra = sorted(set(owners) - set(weight_map))
    if missing:
        _fail("input", f"index references {len(missing)} tensors absent from the shards ({missing[0]})")
    if extra:
        _fail("input", f"{len(extra)} shard tensors are absent from the index ({extra[0]}); keys would be dropped")
    for name, shard_name in weight_map.items():
        if owners[name] != shard_name:
            _fail("input", f"index maps {name} to {shard_name}, but it lives in {owners[name]}")
    return index


def _copy_file(source: str, destination: str) -> None:
    with open(source, "rb") as src, open(destination, "xb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(chunk)


def _update_index(source_index: dict, total_size: int) -> dict:
    updated = dict(source_index)
    metadata = dict(updated.get("metadata") or {})
    metadata["total_size"] = total_size
    updated["metadata"] = metadata
    return updated


def _write_json(path: str, value) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256_file(path: str) -> dict:
    size, sha256 = _hash_file(path)
    if size <= 0:
        _fail("output", f"{os.path.basename(path)} is empty")
    return {"path": os.path.basename(path), "bytes": size, "sha256": sha256}


def require_production_context(env=None) -> dict:
    env = os.environ if env is None else env
    if env.get("GITHUB_ACTIONS") != "true":
        _fail("guard", "Actions-only: GITHUB_ACTIONS must be 'true'")
    if env.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        _fail("guard", "workflow_dispatch-only")
    sha = env.get("GITHUB_SHA")
    if not isinstance(sha, str) or len(sha) != 40 or any(ch not in "0123456789abcdef" for ch in sha.lower()):
        _fail("guard", "GITHUB_SHA must be the exact 40-hex source revision")
    runner = env.get("RUNNER_NAME", "")
    runner_os = env.get("RUNNER_OS", "")
    if runner_os == "macOS":
        if runner not in ALLOWED_MAC_RUNNERS:
            _fail("guard", f"approved macOS runner is {sorted(ALLOWED_MAC_RUNNERS)}, got {runner!r}")
    elif runner_os == "Windows":
        if runner not in ALLOWED_WINDOWS_RUNNERS:
            _fail("guard", f"approved Windows runners are {sorted(ALLOWED_WINDOWS_RUNNERS)}, got {runner!r}")
    else:
        _fail("guard", f"approved runners are macOS/Windows, got {runner_os!r}")
    runner_temp = _require_absolute(env.get("RUNNER_TEMP"), "RUNNER_TEMP")
    return {"runner": runner, "os": runner_os, "runnerTemp": runner_temp, "sourceGhaSha": sha.lower()}


def stage_float_variants(
    receipt,
    stage_root,
    source_profile,
    output_root=None,
    *,
    casebook=None,
    casebook_path=None,
    profile_def=None,
    converter=None,
    chunk_bytes=DEFAULT_CHUNK_BYTES,
    proof_kind="unit_fake",
    allow_unit_fake=False,
    env=None,
) -> dict:
    """Stream the pinned BF16 closure into dense-f16 and dense-f32 source variants."""
    root = _real_directory(stage_root, "stage root")
    if isinstance(receipt, dict) and receipt.get("proof_kind") == "unit_fake" and not allow_unit_fake:
        _fail("proof", "unit_fake stage receipts cannot prepare campaign source variants")
    if profile_def is not None:
        if not (allow_unit_fake and converter is not None):
            _fail("profile", "profile overrides are test-only (allow_unit_fake + injected converter)")
        profile = dict(profile_def)
        if not isinstance(profile.get("files"), list) or not profile["files"] or not isinstance(profile.get("rows"), dict):
            _fail("profile", "test profile must carry files and rows")
        profile.setdefault("setId", source_profile)
        profile.setdefault("partition", PROFILE_PARTITIONS.get(source_profile, "FL2VA"))
    else:
        book = casebook if casebook is not None else load_casebook(casebook_path)
        profile = profile_from_casebook(book, source_profile)
    if not isinstance(chunk_bytes, int) or chunk_bytes < 2 or chunk_bytes > MAX_CHUNK_BYTES or chunk_bytes % 2:
        _fail("input", f"chunk_bytes must be an even integer in [2, {MAX_CHUNK_BYTES}]")
    if converter is not None:
        numpy_version = None
        converter_source = "injected-converter"
    else:
        numpy_version = _numpy_version()
        converter = numpy_converter
        converter_source = "imported"
    verified = verify_stage_receipt(receipt, root, profile, allow_unit_fake=allow_unit_fake)
    for file in verified["files"]:
        size, sha256 = _hash_file(file["finalPath"])
        if size != file["expectedBytes"]:
            _fail("input", f"{file['relative']} is {size} bytes, casebook says {file['expectedBytes']}")
        if sha256 != file["expectedSha256"]:
            _fail("input", f"{file['relative']} sha256 {sha256} differs from the casebook/stage receipt")
    derived_real = _new_directory_under(root, output_root or os.path.join(root, "derived"), "derived output root")
    variant_dirs = {dtype: _new_directory_under(root, os.path.join(derived_real, name), f"{name} output directory") for dtype, name in VARIANTS.items()}
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_gha = (env or os.environ).get("GITHUB_SHA")
    receipt_out = {
        "version": 1,
        "kind": "h3-float-variant-receipt",
        "status": "failed",
        "proof_kind": proof_kind,
        "generated": "source-variant-not-trained",
        "not_new_hf_asset": True,
        "not_model_output": True,
        "accepted_for_plan": False,
        "requires_runner_verification": True,
        "sourceProfile": {k: profile[k] for k in ("setId", "repo", "revision", "prefix", "partition")},
        "source": {
            "root": root,
            "files": [
                {
                    "relative": file["relative"],
                    "actualBytes": file["expectedBytes"],
                    "observedSha256": file["observedSha256"],
                    "expectedSha256": file["expectedSha256"],
                    "expectedLegal": file["expectedLegal"],
                    "explicitAcknowledgment": file["explicitAcknowledgment"],
                }
                for file in verified["files"]
            ],
            "licenseAuthority": verified["licenseAuthority"],
        },
        "variants": {},
        "transformation": {
            "codeSha256": code_sha,
            "sourceGhaSha": source_gha,
            "numpy": {"version": numpy_version, "source": converter_source},
            "chunkBytes": chunk_bytes,
            "bf16ToF32": "exact u16<<16 bit shift",
            "bf16ToF16": "IEEE round-to-nearest-even cast; finite-overflow refused; value-exact in F16 normal range",
        },
        "hfPins": {file["relative"]: file["expectedSha256"] for file in verified["files"]},
        "failure": None,
    }
    try:
        shards = [file for file in verified["files"] if file["relative"].endswith(".safetensors")]
        config = next(file for file in verified["files"] if file["relative"].endswith("config.json"))
        index_file = next(file for file in verified["files"] if file["relative"].endswith(".index.json"))
        if not shards:
            _fail("profile", "selected closure has no safetensors shards")
        if profile_def is None and len(shards) != 14:
            _fail("profile", f"expected 14 shards, found {len(shards)}")
        source_index = validate_index_closure(shards, index_file["finalPath"])
        variant_files = {dtype: [] for dtype in VARIANTS}
        variant_stats = {
            dtype: {"tensorBytes": 0, "tensors": 0, "precisionLoss": 0, "underflow": 0, "nan": 0, "inf": 0, "dtypeHistogram": {}}
            for dtype in VARIANTS
        }
        for shard in shards:
            for dtype, directory in variant_dirs.items():
                destination = os.path.join(directory, shard["basename"])
                convert_shard(shard["finalPath"], destination, dtype, converter, chunk_bytes, variant_stats[dtype])
                variant_files[dtype].append(_sha256_file(destination))
        for dtype, directory in variant_dirs.items():
            config_dest = os.path.join(directory, "config.json")
            _copy_file(config["finalPath"], config_dest)
            variant_files[dtype].append(_sha256_file(config_dest))
            index_dest = os.path.join(directory, os.path.basename(index_file["relative"]))
            _write_json(index_dest, _update_index(source_index, variant_stats[dtype]["tensorBytes"]))
            variant_files[dtype].append(_sha256_file(index_dest))
            config_json = json.loads(Path(config["finalPath"]).read_text(encoding="utf-8"))
            model_type = config_json.get("_class_name") or "unknown"
            model_type_path = os.path.join(directory, ".modelType")
            _write_json(model_type_path, {
                "modelType": model_type,
                "partition": profile["partition"],
                "dtype": dtype,
                "sourceProfile": profile["setId"],
                "generated": "source-variant-not-trained",
            })
            variant_files[dtype].append(_sha256_file(model_type_path))
            receipt_out["variants"][VARIANTS[dtype]] = {
                "status": "completed",
                "directory": os.path.relpath(directory, derived_real),
                "files": variant_files[dtype],
                "dtypeHistogram": variant_stats[dtype]["dtypeHistogram"],
                "tensors": variant_stats[dtype]["tensors"],
                "tensorBytes": variant_stats[dtype]["tensorBytes"],
                "conversionStats": {
                    "exact": dtype == "F32",
                    "precisionLoss": variant_stats[dtype]["precisionLoss"],
                    "underflow": variant_stats[dtype]["underflow"],
                    "nan": variant_stats[dtype]["nan"],
                    "inf": variant_stats[dtype]["inf"],
                    "note": "source-variant-not-trained; BF16->F16 is value-exact within the F16 normal range, with underflow/overflow recorded; BF16->F32 is the exact u16<<16 bit shift",
                },
                "modelType": json.loads(Path(model_type_path).read_text(encoding="utf-8")),
            }
        receipt_out["status"] = "completed"
        _write_json(os.path.join(derived_real, "float-variants-receipt.json"), receipt_out)
        return receipt_out
    except FloatVariantError as exc:
        receipt_out["failure"] = {"code": exc.code, "message": exc.message}
        try:
            _write_json(os.path.join(derived_real, "float-variants-receipt.json"), receipt_out)
        except OSError:
            pass
        raise


def assert_variant_receipt(receipt) -> dict:
    if not isinstance(receipt, dict):
        _fail("proof", "receipt object required")
    if receipt.get("proof_kind") != "campaign_native":
        _fail("proof", "unit_fake receipts cannot be campaign proof")
    if receipt.get("status") != "completed":
        _fail("proof", "completed float-variant receipt required")
    if receipt.get("generated") != "source-variant-not-trained" or receipt.get("not_model_output") is not True:
        _fail("proof", "generated source-variant provenance required")
    for name in VARIANTS.values():
        variant = receipt.get("variants", {}).get(name)
        if not isinstance(variant, dict) or variant.get("status") != "completed":
            _fail("proof", f"{name} completed variant required")
        for file in variant.get("files", []):
            if len(file.get("sha256", "")) != 64 or file.get("bytes", 0) <= 0:
                _fail("proof", f"{name} actual file hash/size required")
    return receipt


def _load_receipt(path: str) -> dict:
    real = os.path.realpath(_require_absolute(path, "--stage-receipt"))
    if os.path.islink(path):
        _fail("path", "--stage-receipt must not be a symlink")
    try:
        with open(real, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        _fail("receipt", f"cannot read stage receipt: {exc}")
    except json.JSONDecodeError as exc:
        _fail("receipt", f"stage receipt is not JSON: {exc}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-receipt", required=True, help="absolute path to the real h3-finetune-stage-receipt JSON")
    parser.add_argument("--stage-root", required=True, help="absolute private root named by settings.privateRoot")
    parser.add_argument("--source-profile", required=True, choices=SOURCE_PROFILE_IDS, help="explicit casebook staging set id")
    parser.add_argument("--casebook", help="casebook path (default: docs/testing/sc-26691-h3-cases.json)")
    parser.add_argument("--output-root", help="absolute NEW derived root (default: <stage-root>/derived)")
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES, help=f"bounded stream chunk (even, <= {MAX_CHUNK_BYTES})")
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        _fail("guard", "run this command with python3 -I")
    context = require_production_context()
    receipt = _load_receipt(args.stage_receipt)
    result = stage_float_variants(
        receipt,
        args.stage_root,
        args.source_profile,
        args.output_root,
        casebook_path=args.casebook,
        chunk_bytes=args.chunk_bytes,
        proof_kind="campaign_native",
        allow_unit_fake=False,
        env=os.environ,
    )
    print(json.dumps({"status": result["status"], "receipt": "float-variants-receipt.json", "sourceGhaSha": context["sourceGhaSha"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FloatVariantError as error:
        print(json.dumps({"status": "failed", "code": error.code, "message": error.message}), file=sys.stderr)
        raise SystemExit(2)
