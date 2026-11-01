#!/usr/bin/env python3
"""Prepare the REQUIRED LibriSpeech speech-conditioning asset from a REAL stage receipt.

This is a bounded, Actions-only asset-preparation step. It does not run a model, open a
port, touch the network, or install software. It consumes the receipt produced by
``scripts/h3-finetune-terminal-stage.mjs`` (kind ``h3-finetune-stage-receipt``, version 1),
selects exactly one known Parquet row, and derives two WAV files with the runner's existing
ffmpeg/ffprobe:

  speech-32000.wav              the row's embedded speech, 32 kHz mono PCM16
  speech-reversed.wav           an areverse diagnostic of that same speech, NOT a second
                                speaker and NOT a conditioning speaker

Run under ``python3 -I`` in the parent workflow:

  python3 -I scripts/h3-finetune-terminal-speech.py \
      --stage-receipt /abs/stage-receipt.json \
      --stage-root /abs/private-root \
      --output-dir /abs/private-root/conditioning/speech-1272-128104-0000

``pyarrow==19.0.1`` is a runner dependency only (private venv under ``$RUNNER_TEMP``). It is
imported and version-checked lazily inside the reader, so importing this module for tests
never needs it. ``ffmpeg``/``ffprobe`` must already exist on the runner PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

EXPECTED = {
    "dataset_repo": "hf-internal-testing/librispeech_asr_dummy",
    "dataset_revision": "5be91486e11a2d616f4ec5db8d3fd248585ac07a",
    "parquet_path": "clean/validation-00000-of-00001.parquet",
    "parquet_bytes": 9_192_059,
    "parquet_sha256": "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603",
    "row_id": "1272-128104-0000",
    "columns": ("id", "audio", "text", "speaker_id"),
    "sample_rate": 32_000,
    "channels": 1,
    "codec": "pcm_s16le",
    "license_id": "CC-BY-4.0",
    "mirror_license_declared": False,
}
PYARROW_VERSION = "19.0.1"
PYARROW_REQUIRED = f"pyarrow=={PYARROW_VERSION}"
# Either Mac may run the MLX lane; the request's expected_runner (exported as H3_EXPECTED_RUNNER)
# names the one, and RUNNER_NAME must equal it exactly — the same rule as
# h3-finetune-terminal-stage.mjs assertExpectedRunner (sc-26691).
ALLOWED_MAC_RUNNERS = frozenset({"nax-macos", "nax-macos-2"})
ALLOWED_WINDOWS_RUNNERS = frozenset({"cuda-windows", "cuda-windows-2"})
SOURCE_NAME = "source-audio.bin"
SPEECH_NAME = "speech-32000.wav"
REVERSED_NAME = "speech-reversed.wav"
RECEIPT_NAME = "speech-receipt.json"
KNOWN_LEGAL = ("licenseId", "inheritedLicenseId", "attributionRequired")
KNOWN_ACKNOWLEDGMENT = ("sha256", "acknowledged", "licenseId", "inheritedLicenseId", "inheritedAcknowledged", "attribution", "role")
KNOWN_AUTHORITY = ("operatorInput", "inputSha256", "kind", "approvedBy", "approvedAt", "reference", "subjects", "acknowledged")


class SpeechStageError(RuntimeError):
    """A refusal with a stable code the workflow can report."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> None:
    raise SpeechStageError(code, message)


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
    except ValueError:  # different drives on Windows
        return False


def _confined_existing(root: str, relative, label: str) -> str:
    """Resolve a receipt-relative path inside ``root`` BEFORE opening it."""
    if not isinstance(relative, str) or not relative.strip():
        _fail("path", f"{label} must be a non-empty relative path")
    if os.path.isabs(relative):
        _fail("path", f"{label} must be relative to the private root")
    parts = relative.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        _fail("path", f"{label} contains an empty, '.', or '..' component")
    candidate = os.path.join(root, *parts)
    real = os.path.realpath(candidate)
    if real == os.path.realpath(root) or not _inside(root, real):
        _fail("path", f"{label} escapes the private root (symlink/traversal refused)")
    if os.path.islink(candidate):
        _fail("path", f"{label} is a symlink; staged assets must be regular files")
    try:
        mode = os.stat(candidate, follow_symlinks=False).st_mode
    except OSError as exc:
        _fail("path", f"{label} is missing: {exc}")
    if not stat.S_ISREG(mode):
        _fail("path", f"{label} is not a regular file")
    return real


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


def _normalize_license_id(value) -> str:
    return str(value or "").strip().lower().replace("_", "-").replace(" ", "-")


def _expected_cache_relative(repo: str, revision: str, file_path: str) -> str:
    return "/".join(
        ["hf", "hub", f"datasets--{repo.replace('/', '--')}", "snapshots", revision, *file_path.split("/")]
    )


def _copy_known(record, known) -> dict:
    if not isinstance(record, dict):
        _fail("receipt", "receipt record must be an object")
    return {key: record[key] for key in known if key in record}


def verify_stage_receipt(receipt, stage_root: str, expected=None, allow_unit_fake: bool = False) -> dict:
    """Consume the REAL ``h3-finetune-stage-receipt`` v1, never an invented schema.

    Required shape (produced by ``scripts/h3-finetune-terminal-stage.mjs``):
      kind == "h3-finetune-stage-receipt", version == 1, status == "completed";
      settings.privateRoot == --stage-root (canonical);
      licenseAuthority { kind: "user-approved", approvedBy, approvedAt, reference,
                         acknowledged: true, subjects[] } (unit_fake only with the test flag);
      files[]: one dataset row for the pinned repo/revision/file with
        status "completed", actualBytes, observedSha256, finalPath (the real HF cache
        snapshot path), expectedLegal { licenseId, inheritedLicenseId?, attributionRequired },
        explicitAcknowledgment { sha256, acknowledged: true, licenseId, inherited..., attribution }.
    The raw attribution credit in expectedLegal.attributionRequired must appear verbatim in the
    explicit acknowledgment's attribution; this function copies both, it never invents consent.
    """
    expected = EXPECTED if expected is None else expected
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
        _fail("proof", "unit_fake stage receipts cannot prepare campaign assets")
    settings = receipt.get("settings")
    if not isinstance(settings, dict):
        _fail("receipt", "stage receipt settings object required")
    declared_root = _require_absolute(settings.get("privateRoot"), "settings.privateRoot")
    real_root = os.path.realpath(stage_root)
    if os.path.realpath(declared_root) != real_root:
        _fail("receipt", "settings.privateRoot differs from --stage-root")
    authority = receipt.get("licenseAuthority")
    if not isinstance(authority, dict):
        _fail("license", "root licenseAuthority object required")
    kind = authority.get("kind")
    if authority.get("acknowledged") is not True:
        _fail("license", "root licenseAuthority.acknowledged must be true")
    if kind == "operator-scoped-binding":
        operator = authority.get("operatorInput")
        if not isinstance(operator, dict) or operator.get("acknowledged") is not True:
            _fail("license", "explicit unchanged scoped operator input required")
        if operator.get("proof_kind") == "unit_fake" and not allow_unit_fake:
            _fail("license", "synthetic scoped approval is not real approval")
        if "speech" not in operator.get("assetSets", []) or "CC-BY-4.0" not in operator.get("licenseIds", []):
            _fail("license", "speech or its license is outside the approved scope")
        if not re.fullmatch(r"[a-f0-9]{64}", str(authority.get("inputSha256", ""))):
            _fail("license", "scoped input provenance hash required")
    else:
        if kind != "user-approved" and not (allow_unit_fake and kind == "unit_fake"):
            _fail("license", "root licenseAuthority.kind must be 'user-approved'")
        _require_nonempty(authority.get("approvedBy"), "license", "licenseAuthority.approvedBy")
        _require_nonempty(authority.get("reference"), "license", "licenseAuthority.reference")
        if not isinstance(authority.get("approvedAt"), str) or not authority["approvedAt"].strip():
            _fail("license", "licenseAuthority.approvedAt required")
    if not isinstance(authority.get("subjects"), list):
        _fail("license", "licenseAuthority.subjects array required")
    files = receipt.get("files")
    if not isinstance(files, list):
        _fail("receipt", "stage receipt files array required")
    matches = [
        row for row in files
        if isinstance(row, dict)
        and row.get("repoType") == "dataset"
        and row.get("repo") == expected["dataset_repo"]
        and row.get("revision") == expected["dataset_revision"]
        and row.get("file") == expected["parquet_path"]
    ]
    if len(matches) != 1:
        _fail("receipt", f"expected exactly one pinned dataset file, found {len(matches)}")
    row = matches[0]
    if row.get("status") != "completed":
        _fail("receipt", "pinned dataset file status must be 'completed'")
    if row.get("actualBytes") != expected["parquet_bytes"]:
        _fail("input", f"pinned dataset file actualBytes must be {expected['parquet_bytes']}")
    observed = row.get("observedSha256")
    if not isinstance(observed, str) or observed.lower() != expected["parquet_sha256"]:
        _fail("input", f"pinned dataset file observedSha256 must be {expected['parquet_sha256']}")
    final_path = _confined_absolute_file(real_root, row.get("finalPath"), "file.finalPath")
    relative = os.path.relpath(final_path, real_root).replace(os.sep, "/")
    wanted = _expected_cache_relative(expected["dataset_repo"], expected["dataset_revision"], expected["parquet_path"])
    if relative != wanted:
        _fail("path", f"file.finalPath is not the pinned HF cache snapshot path ({wanted})")
    legal = row.get("expectedLegal")
    if not isinstance(legal, dict):
        _fail("license", "per-file expectedLegal object required")
    legal_copy = _copy_known(legal, KNOWN_LEGAL)
    license_id = legal_copy.get("licenseId")
    if _normalize_license_id(license_id) not in ("cc-by-4.0", "cc-by-4"):
        _fail("license", "per-file expectedLegal.licenseId must be CC-BY-4.0")
    attribution_required = _require_nonempty(legal_copy.get("attributionRequired"), "license", "expectedLegal.attributionRequired")
    acknowledgment = row.get("explicitAcknowledgment")
    if not isinstance(acknowledgment, dict):
        _fail("license", "per-file explicitAcknowledgment object required")
    acknowledgment_copy = _copy_known(acknowledgment, KNOWN_ACKNOWLEDGMENT)
    if acknowledgment_copy.get("acknowledged") is not True:
        _fail("license", "per-file explicitAcknowledgment.acknowledged must be true")
    if not isinstance(acknowledgment_copy.get("sha256"), str) or acknowledgment_copy["sha256"].lower() != expected["parquet_sha256"]:
        _fail("license", "explicitAcknowledgment.sha256 must match the pinned file")
    if _normalize_license_id(acknowledgment_copy.get("licenseId")) != _normalize_license_id(license_id):
        _fail("license", "explicitAcknowledgment.licenseId must match expectedLegal.licenseId")
    inherited = legal_copy.get("inheritedLicenseId")
    if inherited is not None:
        if acknowledgment_copy.get("inheritedLicenseId") != inherited or acknowledgment_copy.get("inheritedAcknowledged") is not True:
            _fail("license", "explicitAcknowledgment must inherit and acknowledge the inherited licence")
    attribution = acknowledgment_copy.get("attribution")
    if not isinstance(attribution, str) or attribution_required not in attribution:
        _fail("license", "explicitAcknowledgment.attribution must contain the required raw credit")
    return {
        "dataset": {"repo": expected["dataset_repo"], "revision": expected["dataset_revision"]},
        "filePath": expected["parquet_path"],
        "relativePath": relative,
        "finalPath": final_path,
        "actualBytes": row["actualBytes"],
        "sha256": observed.lower(),
        "expectedLegal": legal_copy,
        "explicitAcknowledgment": acknowledgment_copy,
        "licenseAuthority": _copy_known(authority, KNOWN_AUTHORITY),
        "proofKind": proof_kind,
    }


def _pyarrow_rows(path: str, columns) -> tuple[list[dict], str]:
    """The ONLY pyarrow import: reached only when no reader is injected."""
    try:
        import pyarrow  # type: ignore[import-not-found]
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only in a misconfigured runner
        _fail("tool", f"{PYARROW_REQUIRED} required in the runner venv under $RUNNER_TEMP ({exc})")
    version = getattr(pyarrow, "__version__", None)
    if version != PYARROW_VERSION:
        _fail("tool", f"{PYARROW_REQUIRED} required; runner has pyarrow {version!r}")
    table = parquet.read_table(path, columns=list(columns))
    return table.to_pylist(), version


def read_rows(path: str, reader=None) -> tuple[list[dict], str | None]:
    """Return (rows, installed_pyarrow_version). An injected reader never claims a version."""
    if reader is not None:
        rows, version = reader(path, list(EXPECTED["columns"])), None
    else:
        rows, version = _pyarrow_rows(path, EXPECTED["columns"])
    if hasattr(rows, "to_pylist"):
        rows = rows.to_pylist()
    try:
        rows = list(rows)
    except TypeError:
        _fail("row", "Parquet reader did not return rows")
    return rows, version


def select_speech_row(rows: list[dict], expected=None) -> dict:
    """Select the EXACT unique id; no first-row/fallback/tone substitution."""
    if not isinstance(rows, list):
        _fail("row", "Parquet rows must be a list")
    expected = EXPECTED if expected is None else expected
    matches = [row for row in rows if isinstance(row, dict) and str(row.get("id")) == expected["row_id"]]
    if len(matches) != 1:
        _fail("row", f"expected exactly one row id {expected['row_id']}, found {len(matches)}")
    row = matches[0]
    audio = row.get("audio")
    if not isinstance(audio, dict):
        _fail("audio", "row audio must be an embedded struct with bytes")
    data = audio.get("bytes")
    if not isinstance(data, (bytes, bytearray, memoryview)) or len(bytes(data)) == 0:
        _fail("audio", "embedded audio.bytes must be non-empty")
    text = _require_nonempty(row.get("text"), "row", "row transcript")
    if row.get("speaker_id") is None:
        _fail("row", "row speaker_id required")
    return {"audioBytes": bytes(data), "text": text, "speakerId": row["speaker_id"]}


def _run(argv, runner=None):
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        _fail("tool", "tool argv must be a non-empty list of strings")
    if runner is not None:
        return runner(argv, cwd=None)
    # shell=False and a list argv: no shell interpolation, no script from a downloaded dir.
    return subprocess.run(argv, cwd=None, shell=False, check=False, capture_output=True, text=True)


def _run_checked(argv, runner=None, label: str = "tool") -> None:
    completed = _run(argv, runner=runner)
    if getattr(completed, "returncode", 1) != 0:
        stderr = (getattr(completed, "stderr", "") or "").strip()
        _fail("tool", f"{label} failed ({getattr(completed, 'returncode', '?')}): {stderr}")


def trusted_tool(name: str, stage_root: str) -> str:
    """Resolve an existing runner tool from PATH; refuse tools under the staged tree."""
    found = shutil.which(name)
    if not found:
        _fail("tool", f"{name} not found on the runner PATH; no local install permitted")
    real = os.path.realpath(found)
    if _inside(os.path.realpath(stage_root), real):
        _fail("tool", f"{name} resolves under the private root; refusing a downloaded tool")
    if not os.access(real, os.X_OK):
        _fail("tool", f"{name} is not executable: {real}")
    return real


def _probe_audio(ffprobe: str, path: str, runner=None) -> dict:
    completed = _run(
        [
            ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,duration,duration_ts,time_base,nb_read_frames",
            "-of", "json", path,
        ],
        runner=runner,
    )
    if getattr(completed, "returncode", 1) != 0:
        _fail("probe", f"ffprobe failed: {(getattr(completed, 'stderr', '') or '').strip()}")
    try:
        payload = json.loads(getattr(completed, "stdout", "") or "{}")
    except json.JSONDecodeError as exc:
        _fail("probe", f"ffprobe returned invalid JSON: {exc}")
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        _fail("probe", "exactly one audio stream required")
    return streams[0]


def validate_probe(stream: dict) -> dict:
    if not isinstance(stream, dict):
        _fail("probe", "audio stream metadata required")
    try:
        sample_rate = int(stream.get("sample_rate"))
        channels = int(stream.get("channels"))
        duration = float(stream.get("duration"))
    except (TypeError, ValueError):
        _fail("probe", "sample_rate, channels and duration must be numeric")
    if sample_rate != EXPECTED["sample_rate"]:
        _fail("probe", f"sample_rate must be {EXPECTED['sample_rate']}, got {sample_rate}")
    if channels != EXPECTED["channels"]:
        _fail("probe", f"channels must be {EXPECTED['channels']}, got {channels}")
    if duration <= 0:
        _fail("probe", "positive audio coverage required")
    codec = stream.get("codec_name")
    if codec is not None and codec != EXPECTED["codec"]:
        _fail("probe", f"codec must be {EXPECTED['codec']}, got {codec}")
    frames = stream.get("nb_read_frames")
    if frames is not None:
        try:
            if int(frames) <= 0:
                _fail("probe", "positive decoded audio frames required")
        except (TypeError, ValueError):
            _fail("probe", "nb_read_frames must be numeric when present")
    samples_exact = False
    samples = None
    duration_ts = stream.get("duration_ts")
    time_base = stream.get("time_base")
    if duration_ts is not None and isinstance(time_base, str):
        match = re.fullmatch(r"(\d+)/(\d+)", time_base)
        if match and int(match.group(2)) > 0:
            samples = int(round(int(duration_ts) * int(match.group(1)) / int(match.group(2)) * sample_rate))
            samples_exact = True
    if samples is None:
        samples = int(round(duration * sample_rate))
    if samples <= 0:
        _fail("probe", "positive PCM sample count required")
    return {
        "codec": codec,
        "sampleRate": sample_rate,
        "channels": channels,
        "durationSeconds": duration,
        "samples": samples,
        "samplesExact": samples_exact,
    }


def _tool_version(tool: str, runner=None) -> str:
    completed = _run([tool, "-version"], runner=runner)
    if getattr(completed, "returncode", 1) != 0:
        _fail("tool", f"{os.path.basename(tool)} -version failed")
    first = (getattr(completed, "stdout", "") or "").strip().splitlines()
    return first[0].strip() if first else "unknown"


def _write_new(path: str, data: bytes) -> None:
    with open(path, "xb") as handle:
        handle.write(data)


def _write_json(path: str, value) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _output_record(path: str, role: str, probe: dict) -> dict:
    size, sha256 = _hash_file(path)
    if size <= 0:
        _fail("output", f"{role} output is empty")
    return {
        "path": os.path.basename(path),
        "role": role,
        "bytes": size,
        "sha256": sha256,
        "codec": probe["codec"],
        "sampleRate": probe["sampleRate"],
        "channels": probe["channels"],
        "durationSeconds": probe["durationSeconds"],
        "samples": probe["samples"],
    }


def stage_speech(
    receipt,
    stage_root,
    output_dir=None,
    *,
    reader=None,
    tool_resolver=None,
    runner=None,
    prober=None,
    proof_kind: str = "unit_fake",
    expected=None,
    allow_unit_fake: bool = False,
) -> dict:
    """Produce the speech asset from the real stage receipt.

    ``proof_kind`` defaults to ``unit_fake``. Only the guarded CLI passes
    ``campaign_native`` with ``allow_unit_fake=False``; injected readers can never be
    campaign proof.
    """
    root = _real_directory(stage_root, "stage root")
    expected = EXPECTED if expected is None else expected
    verified = verify_stage_receipt(receipt, root, expected=expected, allow_unit_fake=allow_unit_fake)
    parquet = verified["finalPath"]
    size, sha256 = _hash_file(parquet)
    if size != expected["parquet_bytes"]:
        _fail("input", f"staged parquet is {size} bytes, expected {expected['parquet_bytes']}")
    if sha256 != expected["parquet_sha256"]:
        _fail("input", f"staged parquet sha256 is {sha256}, expected {expected['parquet_sha256']}")
    rows, pyarrow_version = read_rows(parquet, reader=reader)
    selected = select_speech_row(rows, expected=expected)
    if output_dir is None:
        output_dir = os.path.join(root, "conditioning", f"speech-{expected['row_id']}")
    out = _new_directory_under(root, output_dir, "conditioning output directory")
    source_path = os.path.join(out, SOURCE_NAME)
    speech_path = os.path.join(out, SPEECH_NAME)
    reversed_path = os.path.join(out, REVERSED_NAME)
    resolver = tool_resolver or trusted_tool
    ffmpeg = resolver("ffmpeg", root)
    ffprobe = resolver("ffprobe", root)
    probe = prober or (lambda path: _probe_audio(ffprobe, path, runner=runner))
    provenance = {
        "status": "failed",
        "proof_kind": proof_kind,
        "method": "generated-not-model-proof",
        "asset": {"id": expected["row_id"], "kind": "speech-conditioning"},
        "source": {
            "dataset": verified["dataset"],
            "parquet": {
                "path": verified["filePath"],
                "finalPath": verified["relativePath"],
                "actualBytes": size,
                "sha256": sha256,
            },
            "row": expected["row_id"],
            "transcript": selected["text"],
            "speakerId": selected["speakerId"],
        },
        "license": {
            "expectedLegal": verified["expectedLegal"],
            "explicitAcknowledgment": verified["explicitAcknowledgment"],
            "licenseAuthority": verified["licenseAuthority"],
            "explicitInput": True,
            "mirrorRevision": expected["dataset_revision"],
            "mirrorLicenseDeclared": expected["mirror_license_declared"],
        },
        "inputs": [
            {
                "path": verified["relativePath"],
                "bytes": size,
                "sha256": sha256,
                "kind": "staged-parquet",
            }
        ],
        "outputs": [],
        "tools": {},
        "durationSeconds": None,
        "failure": None,
    }
    try:
        _write_new(source_path, selected["audioBytes"])
        source_size, source_sha = _hash_file(source_path)
        provenance["source"]["sourceAudio"] = {"bytes": source_size, "sha256": source_sha}
        _run_checked(
            [
                ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y",
                "-i", source_path,
                "-ar", str(EXPECTED["sample_rate"]), "-ac", str(EXPECTED["channels"]),
                "-c:a", EXPECTED["codec"], speech_path,
            ],
            runner=runner, label="ffmpeg speech decode",
        )
        # areverse is a diagnostic of the same speech; it is never a second speaker.
        _run_checked(
            [
                ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y",
                "-i", speech_path, "-af", "areverse",
                "-ar", str(EXPECTED["sample_rate"]), "-ac", str(EXPECTED["channels"]),
                "-c:a", EXPECTED["codec"], reversed_path,
            ],
            runner=runner, label="ffmpeg areverse diagnostic",
        )
        speech_probe = validate_probe(probe(speech_path))
        reversed_probe = validate_probe(probe(reversed_path))
        # PCM reversal must preserve the exact sample count; when only durations are
        # available, they may differ by at most one 1/32000 s sample.
        if speech_probe["samplesExact"] and reversed_probe["samplesExact"]:
            if speech_probe["samples"] != reversed_probe["samples"]:
                _fail(
                    "probe",
                    f"areverse changed the PCM sample count ({speech_probe['samples']} -> {reversed_probe['samples']})",
                )
        elif abs(speech_probe["durationSeconds"] - reversed_probe["durationSeconds"]) > 1.0 / EXPECTED["sample_rate"] + 1e-12:
            _fail("probe", "speech and reversed durations differ by more than one sample")
        for path, role, checked in (
            (speech_path, "speech-source", speech_probe),
            (reversed_path, "diagnostic-reversed-not-speaker", reversed_probe),
        ):
            _run_checked(
                [ffmpeg, "-v", "error", "-xerror", "-i", path, "-f", "null", "-"],
                runner=runner, label=f"ffmpeg full decode {role}",
            )
            provenance["outputs"].append(_output_record(path, role, checked))
        provenance["durationSeconds"] = speech_probe["durationSeconds"]
        provenance["tools"] = {
            "ffmpeg": {"path": ffmpeg, "version": _tool_version(ffmpeg, runner=runner)},
            "ffprobe": {"path": ffprobe, "version": _tool_version(ffprobe, runner=runner)},
            "python": sys.version.split()[0],
            "pyarrow": {
                "version": pyarrow_version,
                "source": "imported" if pyarrow_version is not None else "injected-reader",
            },
        }
        provenance["status"] = "completed"
        _write_json(os.path.join(out, RECEIPT_NAME), provenance)
        return provenance
    except SpeechStageError as exc:
        provenance["failure"] = {"code": exc.code, "message": exc.message}
        try:
            _write_json(os.path.join(out, RECEIPT_NAME), provenance)
        except OSError:
            pass
        raise


def assert_campaign_receipt(receipt) -> dict:
    """Reject unit_fake/incomplete receipts; a fake preparation is never campaign proof."""
    if not isinstance(receipt, dict):
        _fail("proof", "receipt object required")
    if receipt.get("proof_kind") != "campaign_native":
        _fail("proof", "unit_fake receipts cannot be campaign proof")
    if receipt.get("status") != "completed":
        _fail("proof", "completed speech receipt required")
    if receipt.get("method") != "generated-not-model-proof":
        _fail("proof", "generated-not-model-proof method required")
    roles = {output.get("role") for output in receipt.get("outputs", []) if isinstance(output, dict)}
    if roles != {"speech-source", "diagnostic-reversed-not-speaker"}:
        _fail("proof", "both speech-source and reversed-diagnostic outputs required")
    for output in receipt["outputs"]:
        if not isinstance(output.get("sha256"), str) or len(output["sha256"]) != 64:
            _fail("proof", "output sha256 required")
        if not isinstance(output.get("bytes"), int) or output["bytes"] <= 0:
            _fail("proof", "output bytes required")
        if output.get("sampleRate") != EXPECTED["sample_rate"] or output.get("channels") != EXPECTED["channels"]:
            _fail("proof", "output speech format required")
    license_record = receipt.get("license")
    if not isinstance(license_record, dict) or license_record.get("explicitInput") is not True:
        _fail("proof", "explicit recorded licence input required")
    if not isinstance(license_record.get("explicitAcknowledgment"), dict):
        _fail("proof", "copied explicit acknowledgment required")
    if not isinstance(license_record.get("licenseAuthority"), dict):
        _fail("proof", "copied root license authority required")
    return receipt


def require_production_context(env=None) -> dict:
    """Actions/workflow_dispatch on an approved runner; never the primary dev machine."""
    env = os.environ if env is None else env
    if env.get("GITHUB_ACTIONS") != "true":
        _fail("guard", "Actions-only: GITHUB_ACTIONS must be 'true'")
    if env.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        _fail("guard", "workflow_dispatch-only")
    runner = env.get("RUNNER_NAME", "")
    runner_os = env.get("RUNNER_OS", "")
    if runner_os == "macOS":
        if runner not in ALLOWED_MAC_RUNNERS:
            _fail("guard", f"approved macOS runners are {sorted(ALLOWED_MAC_RUNNERS)}, got {runner!r}")
        expected = env.get("H3_EXPECTED_RUNNER")
        if expected not in ALLOWED_MAC_RUNNERS:
            _fail("guard", f"H3_EXPECTED_RUNNER must name an approved macOS runner {sorted(ALLOWED_MAC_RUNNERS)}, got {expected!r}")
        if runner != expected:
            _fail("guard", f"RUNNER_NAME {runner!r} must equal the request expected_runner (H3_EXPECTED_RUNNER) {expected!r}")
    elif runner_os == "Windows":
        if runner not in ALLOWED_WINDOWS_RUNNERS:
            _fail("guard", f"approved Windows runners are {sorted(ALLOWED_WINDOWS_RUNNERS)}, got {runner!r}")
        expected = env.get("H3_EXPECTED_RUNNER")
        if expected is not None and runner != expected:
            _fail("guard", f"RUNNER_NAME {runner!r} must equal the request expected_runner (H3_EXPECTED_RUNNER) {expected!r}")
    else:
        _fail("guard", f"approved runners are macOS/Windows, got {runner_os!r}")
    runner_temp = _require_absolute(env.get("RUNNER_TEMP"), "RUNNER_TEMP")
    return {"runner": runner, "os": runner_os, "runnerTemp": runner_temp}


def _load_receipt(path: str) -> dict:
    real = os.path.realpath(_require_absolute(path, "--stage-receipt"))
    if os.path.islink(path):
        _fail("path", "--stage-receipt must not be a symlink")
    try:
        with open(real, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except OSError as exc:
        _fail("receipt", f"cannot read stage receipt: {exc}")
    except json.JSONDecodeError as exc:
        _fail("receipt", f"stage receipt is not JSON: {exc}")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-receipt", required=True, help="absolute path to the real h3-finetune-stage-receipt JSON")
    parser.add_argument("--stage-root", required=True, help="absolute private root named by settings.privateRoot")
    parser.add_argument(
        "--output-dir",
        help="absolute NEW directory under --stage-root (default: <root>/conditioning/speech-1272-128104-0000)",
    )
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        _fail("guard", "run this command with python3 -I")
    require_production_context()
    receipt = _load_receipt(args.stage_receipt)
    result = stage_speech(
        receipt,
        args.stage_root,
        args.output_dir,
        proof_kind="campaign_native",
        allow_unit_fake=False,
    )
    print(json.dumps({"status": result["status"], "receipt": RECEIPT_NAME}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpeechStageError as error:
        print(json.dumps({"status": "failed", "code": error.code, "message": error.message}), file=sys.stderr)
        raise SystemExit(2)
