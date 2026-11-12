from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import pytest

# Shared, corruption-free fixtures (sc-8934 / F-132): the PNG + safetensors
# builder and the API-spawn helpers used to be copy-pasted (and had drifted)
# between this file and test_rust_api_worker_smoke.py.
from rust_api_harness import (
    PNG_1X1,
    enable_api_listening_log,
    safetensors_bytes,
    spawn_env,
    spawn_process,
    wait_for_health,
)


pytestmark = pytest.mark.parity

ROOT = Path(__file__).resolve().parents[1]
# 🔴 REGENERATE ON LINUX WITH DEFAULT FEATURES, NOT ON A MAC. These snapshots record the contract
# of the build `parity-rust` runs: Linux with NO `backend-candle`, i.e. the "neither" build. Parts
# of the model manifest are cfg-split, so a Mac serves a DIFFERENT and equally correct response —
# `encoder_route_for_model` has a real arm under
# `any(target_os = "macos", all(not(macos), feature = "backend-candle"))` and a `None`-returning
# stub otherwise, so `textEncoderOptions` is present on macOS and absent on the lane. The manifest
# test validates that optional field exactly when the running binary exposes it, then removes only
# that known cfg-split field before comparing with the Linux snapshot. This keeps the two live Rust
# runtimes under full comparison without baking a macOS-only key into the CI contract.
SNAPSHOT_PATH = ROOT / "tests" / "fixtures" / "rust_api_contract_snapshots" / "snapshots.json"
UPDATE_SNAPSHOTS = os.getenv(
    "UPDATE_SNAPSHOTS",
    os.getenv("SCENEWORKS_UPDATE_PARITY_SNAPSHOTS", ""),
).strip().lower() in {
    "1",
    "true",
    "yes",
}
UPDATED_SNAPSHOTS: dict[str, Any] = {}


TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")
PREFIXED_ID_RE = re.compile(
    r"\b(project|asset|job|timeline|track|transition|genset|generation_set|"
    r"look|character_lora|character|lora|worker)_[0-9a-f]{8,32}\b"
)
UPLOAD_SUFFIX_RE = re.compile(r"\b([a-z0-9]+(?:-[a-z0-9]+)*)-[0-9a-f]{8}(?=\.)")
TIMELINE_EXPORT_SNAPSHOT_RE = re.compile(
    r"\btimeline-exports/timeline_[0-9a-f]{32}_[0-9a-f]{32}\.json\b"
)
HEX_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
CLAIM_WORKER_RE = re.compile(r"\bclaim-worker-[a-z]\b")


def write_updated_snapshots() -> None:
    if not UPDATE_SNAPSHOTS:
        return
    # Merge into the existing on-disk snapshots rather than serializing only the
    # labels touched this run. Otherwise a filtered update (e.g.
    # `UPDATE_SNAPSHOTS=1 pytest -k character`) would clobber every untouched
    # snapshot, since UPDATED_SNAPSHOTS only accumulates labels asserted during
    # this run (sc-8862). snapshots() returns {} when the file does not yet
    # exist, so first-ever generation still works.
    #
    # Note: because this merge preserves all pre-existing labels, intentionally
    # REMOVING a snapshot must be done by editing snapshots.json manually — a
    # filtered UPDATE_SNAPSHOTS run will never delete stale entries.
    merged = dict(snapshots())
    merged.update(UPDATED_SNAPSHOTS)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


atexit.register(write_updated_snapshots)


# The portable binding contract version this harness's fake workers claim with (SW-7030).
PORTABLE_MODEL_BINDING_VERSION = 2


def portable_claim_envelope(worker_id: str, capabilities: list[str]) -> dict:
    """A production worker's registration + claim envelope for the harness fakes.

    A worker registration without `platform` can never match a frozen route's
    platform, and a claim without the portable route envelope is refused every
    bound job (`{"job": null}`). The fake harness workers therefore advertise
    the same `platform` + route envelope a real worker sends — the established
    pattern in apps/rust-api/src/tests/support.rs (`TestPortableRoute`) and the
    e2e smoke harness.

    The route is pinned to `candle:linux` regardless of the host OS: the frozen
    route set is a property of the model manifests (their per-download
    platforms), never of the API host, and the platform check compares the
    CLAIM against the worker's REGISTERED platform row — so a fake registered as
    a linux candle worker claims the candle:linux route identically on a macOS
    dev box and a Linux CI runner. That keeps the recorded snapshots
    OS-independent (`gpuId: 0`, `platform: linux` everywhere), which a
    host-derived backend could not.
    """
    backend = "candle"
    platform = "linux"
    gpu_id = "0"
    # The `candle` marker capability is the seam `worker_is_candle` keys off.
    advertise = list(capabilities) + [backend]
    registration = {
        "workerId": worker_id,
        "gpuId": gpu_id,
        "gpuName": "Fixture GPU",
        "platform": platform,
        "capabilities": advertise,
        "loadedModels": [],
    }
    claim = {
        "workerId": worker_id,
        "portableModelBindingVersion": PORTABLE_MODEL_BINDING_VERSION,
        "portableBackend": backend,
        "portablePlatform": platform,
    }
    return {"registration": registration, "claim": claim}


def portable_lora_source_claim(lora_id: str, path: Path) -> dict:
    """The `registered_digest:lora:...` source claim a production worker derives from its
    own inventory of the same LoRA bytes (SW-7030), so the harness's fake workers advertise
    the exact local source closure a character's bound LoRA entry requires at enqueue.
    Path-free: keyed by the logical lora id + the sha256 file ledger — the same envelope
    the Rust-API harness fixture (`advertise_fixture_lora_sources`) builds, and the same
    shape the e2e smoke harness derives. `sourceId` embeds sha256(("lora", id, None)) and
    the ledger; admission recomputes both from the frozen member, so any drift is a 409
    rather than a silent mismatch."""
    payload = path.read_bytes()
    sha256 = "sha256:" + hashlib.sha256(payload).hexdigest()
    files = [
        {
            "relativePath": path.name,
            "sizeBytes": path.stat().st_size,
            "sha256": sha256,
        }
    ]
    # Mirror portable_file_ledger_sha256 (crates/sceneworks-core/...): sha256 over each
    # file's (relative path, NUL, size, NUL, sha256, newline) run.
    ledger = "sha256:" + hashlib.sha256(
        path.name.encode()
        + b"\x00"
        + str(path.stat().st_size).encode()
        + b"\x00"
        + sha256.encode()
        + b"\n"
    ).hexdigest()
    identity = json.dumps(["lora", lora_id, None], separators=(",", ":")).encode()
    source_id = (
        "registered_digest:lora:"
        + hashlib.sha256(identity).hexdigest()
        + ":"
        + ledger
    )
    return {
        "schemaVersion": PORTABLE_MODEL_BINDING_VERSION,
        "sourceId": source_id,
        "provenance": {"kind": "operator_override"},
        "files": files,
        "closureSha256": ledger,
    }


def write_contract_manifests(config_dir: Path) -> None:
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "builtin.models.jsonc").write_text(
        """
        {
          // Comments are part of the manifest contract.
          "schemaVersion": 1,
          "models": [
            {
              "id": "base-model",
              "name": "Base Model",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image_diffusers",
              "capabilities": ["text_to_image"],
              "downloads": [
                {
                  "provider": "huggingface",
                  "repo": "owner/alternate-base-model",
                  "files": ["*.bin"],
                  "estimatedSizeBytes": 536870912
                },
                {
                  "provider": "huggingface",
                  "repo": "owner/base-model",
                  "files": ["*.safetensors"],
                  "default": true,
                  "estimatedSizeBytes": 12884901888
                }
              ],
              "paths": {},
              "defaults": { "width": 1024, "height": 1024 },
              "limits": {},
              "loraCompatibility": { "families": ["z-image"] },
              "ui": { "label": "Base" }
            },
            {
              // A REGISTERED builtin id: the compiled registry's row (with its typed
              // evidence) replaces whatever this file says for it (SW-7031), so the row
              // only places z_image_turbo in this catalog. It is the routed model the job
              // contracts need; its fields here are deliberately minimal because none of
              // them reach a response.
              "id": "z_image_turbo",
              "name": "Z-Image-Turbo",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image_diffusers"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    (manifest_dir / "user.models.jsonc").write_text(
        """
        {
          "schemaVersion": 1,
          "models": [
            { "id": "base-model", "name": "User Model", "ui": { "label": "User" } }
          ]
        }
        """,
        encoding="utf-8",
    )
    (manifest_dir / "builtin.loras.jsonc").write_text(
        """
        {
          "schemaVersion": 1,
          "loras": [
            {
              "id": "style-lora",
              "name": "Style LoRA",
              "family": "z-image",
              "triggerWords": ["style"],
              "compatibility": { "families": ["z-image", "wan-video"] },
              "source": { "provider": "local", "path": "loras/style.safetensors" }
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    (manifest_dir / "user.loras.jsonc").write_text(
        '{ "schemaVersion": 1, "loras": [] }\n',
        encoding="utf-8",
    )


def parse_response_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


@dataclass
class ApiResponse:
    status_code: int
    body: Any
    headers: dict[str, str]


class ServerApiHarness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runtime = "rust"
        write_contract_manifests(root / "config")
        # sc-19708: the catalog now reports the configured model-library root
        # (`modelResolution.configuredLibraryPath`). That root is environment-derived — the API
        # resolves HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE / HF_HOME and otherwise falls back to the
        # OS Hugging Face home under $HOME — so without a pin the golden would record WHOSE
        # machine recorded it (a macOS dev box bakes in /Users/<name>/..., which then fails on
        # every Linux runner, and re-recording there just flips which machine is broken).
        # HF_HUB_CACHE has top precedence, so pinning it under `root` makes the value identical on
        # every machine and lets the existing per-runtime redaction render it <runtime-root>/...,
        # exactly like every other path in the contract. It also keeps the contract run off the
        # developer's real Hugging Face cache. The assertion still discriminates: the field must
        # be present, a string, and this exact configured root. The pin lives in the shared
        # `spawn_env` helper; the e2e harness lacked it and paid a 22.6s install-state sweep
        # against the host cache on a developer box.
        env = spawn_env(root)
        env.update(
            {
                "SCENEWORKS_API_HOST": "127.0.0.1",
                "SCENEWORKS_API_PORT": "0",
                "SCENEWORKS_DATA_DIR": str(root / "data"),
                "SCENEWORKS_CONFIG_DIR": str(root / "config"),
                # sc-15504: this harness ships its OWN builtin.*.jsonc fixtures (a synthetic catalog
                # whose ids and comments are the contract under test). Opt out of the default
                # SyncFromEmbedded self-heal so the seed preserves them verbatim (IfMissing) instead of
                # overwriting them with the binary's embedded manifest.
                "SCENEWORKS_OWN_MANIFESTS": "1",
                "SCENEWORKS_JOBS_DB_PATH": str(root / "data" / "cache" / "jobs.db"),
                "SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE": "1",
            }
        )
        enable_api_listening_log(env)
        rust_binary = os.getenv("SCENEWORKS_RUST_API_BINARY")
        command = (
            [rust_binary]
            if rust_binary
            else ["cargo", "run", "-q", "-p", "sceneworks-rust-api"]
        )
        self.process = spawn_process(command, cwd=ROOT, env=env)
        self.base_url = self.process.wait_for_listening_url()
        wait_for_health(self.base_url, self.process, "rust")
        self.client = httpx.Client(base_url=self.base_url, timeout=10)

    def request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Any = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        headers: dict[str, str] | None = None,
        content: str | bytes | None = None,
    ) -> ApiResponse:
        response = self.client.request(
            method,
            path,
            json=json_payload,
            files=files,
            headers=headers,
            content=content,
        )
        return ApiResponse(response.status_code, parse_response_body(response), dict(response.headers))

    def close(self) -> None:
        self.client.close()
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


@dataclass
class ContractRuntime:
    name: str
    api: ServerApiHarness
    roots: tuple[Path, ...]
    project_id: str | None = None
    project_path: Path | None = None
    asset_id: str | None = None
    job_id: str | None = None

    def request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Any = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        headers: dict[str, str] | None = None,
        content: str | bytes | None = None,
    ) -> ApiResponse:
        response = self.api.request(
            method,
            path,
            json_payload=json_payload,
            files=files,
            headers=headers,
            content=content,
        )
        assert response.status_code < 500, (
            f"{self.name} {method} {path} returned {response.status_code}: {response.body}"
        )
        return response


@pytest.fixture()
def contract_runtimes(tmp_path):
    """Run two isolated Rust APIs to catch non-deterministic wire output before snapshot comparison."""
    runtimes = [
        ContractRuntime("rust-a", ServerApiHarness(tmp_path / "rust-a"), (tmp_path / "rust-a",)),
        ContractRuntime("rust-b", ServerApiHarness(tmp_path / "rust-b"), (tmp_path / "rust-b",)),
    ]
    try:
        yield runtimes
    finally:
        for runtime in runtimes:
            runtime.api.close()


def normalize_contract(value: Any, roots: tuple[Path, ...], path: str = "$") -> Any:
    return _normalize_contract(value, roots, path, {}, {})


def _normalize_contract(
    value: Any,
    roots: tuple[Path, ...],
    path: str,
    normalized_ids: dict[tuple[str, str], str],
    id_counts: dict[str, int],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_contract(
                item,
                roots,
                f"{path}.{key}",
                normalized_ids,
                id_counts,
            )
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        return [
            _normalize_contract(
                item,
                roots,
                f"{path}[{index}]",
                normalized_ids,
                id_counts,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        for root in roots:
            normalized = normalized.replace(str(root).replace("\\", "/"), "<runtime-root>")
        normalized = TIMESTAMP_RE.sub("<timestamp>", normalized)
        normalized = UPLOAD_SUFFIX_RE.sub(r"\1-<upload-suffix>", normalized)
        normalized = TIMELINE_EXPORT_SNAPSHOT_RE.sub(
            "timeline-exports/<timeline-export-snapshot>.json", normalized
        )

        def normalize_id(match: re.Match[str]) -> str:
            prefix = match.group(1)
            original = match.group(0)
            key = (prefix, original)
            if key not in normalized_ids:
                ordinal = id_counts.get(prefix, 0) + 1
                id_counts[prefix] = ordinal
                suffix = "" if ordinal == 1 else f"_{ordinal}"
                normalized_ids[key] = f"{prefix}_fixture{suffix}"
            return normalized_ids[key]

        normalized = PREFIXED_ID_RE.sub(normalize_id, normalized)
        normalized = CLAIM_WORKER_RE.sub("claim-worker-fixture", normalized)
        # The exact JSON parser wording/offset is intentionally ignored; the
        # stable contract is the 422 envelope.
        if path.endswith(".ctx.error"):
            return "<json-decode-error>"
        if path.endswith(".ticket") and HEX_TOKEN_RE.match(normalized):
            return "<event-ticket>"
        if normalized == "rust":
            return "<runtime>"
        if path.endswith(".startupMaintenance.status"):
            return "<maintenance-status>"
        return normalized
    if path.endswith(".loc[1]") and isinstance(value, int):
        return "<json-error-offset>"
    return value


def test_normalize_contract_preserves_distinct_id_relationships():
    normalized = normalize_contract(
        {
            "assets": [
                {"id": "asset_aaaaaaaa", "projectId": "project_11111111"},
                {"id": "asset_bbbbbbbb", "projectId": "project_11111111"},
            ],
            "selectedAssetId": "asset_aaaaaaaa",
        },
        (),
    )

    assert normalized["assets"][0]["id"] == "asset_fixture"
    assert normalized["assets"][1]["id"] == "asset_fixture_2"
    assert normalized["selectedAssetId"] == "asset_fixture"
    assert {asset["projectId"] for asset in normalized["assets"]} == {"project_fixture"}


def test_normalize_contract_preserves_the_immutable_timeline_export_path_shape():
    normalized = normalize_contract(
        "timeline-exports/timeline_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.json",
        (),
    )
    assert normalized == "timeline-exports/<timeline-export-snapshot>.json"


def diff_values(left: Any, right: Any, path: str = "$") -> list[str]:
    if path.endswith(".elapsedSeconds"):
        if left is None and right is None:
            return []
        if isinstance(left, (int, float)) and isinstance(right, (int, float)) and abs(left - right) <= 2:
            return []
        return [f"{path}: elapsed seconds drifted beyond tolerance: {left!r} != {right!r}"]
    if type(left) is not type(right):
        return [f"{path}: type {type(left).__name__} != {type(right).__name__}"]
    if isinstance(left, dict):
        diffs = []
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys - right_keys):
            diffs.append(f"{path}.{key}: only in baseline")
        for key in sorted(right_keys - left_keys):
            diffs.append(f"{path}.{key}: only in candidate")
        for key in sorted(left_keys & right_keys):
            diffs.extend(diff_values(left[key], right[key], f"{path}.{key}"))
            if len(diffs) >= 12:
                break
        return diffs
    if isinstance(left, list):
        diffs = []
        if len(left) != len(right):
            diffs.append(f"{path}: list length {len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            diffs.extend(diff_values(left_item, right_item, f"{path}[{index}]"))
            if len(diffs) >= 12:
                break
        return diffs
    if left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []


def assert_runtime_consistency(
    label: str,
    baseline_runtime: ContractRuntime,
    candidate_runtime: ContractRuntime,
    left: Any,
    right: Any,
) -> Any:
    normalized_baseline = normalize_contract(left, baseline_runtime.roots)
    normalized_candidate = normalize_contract(right, candidate_runtime.roots)
    diffs = diff_values(normalized_baseline, normalized_candidate)
    assert not diffs, (
        f"{label} contract drifted between isolated Rust runs:\n"
        + "\n".join(diffs)
        + "\n\nBaseline:\n"
        + json.dumps(normalized_baseline, indent=2, sort_keys=True)
        + "\n\nCandidate:\n"
        + json.dumps(normalized_candidate, indent=2, sort_keys=True)
    )
    return normalized_baseline


@lru_cache(maxsize=None)
def _cached_snapshots(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def snapshots() -> dict[str, Any]:
    return copy.deepcopy(_cached_snapshots(SNAPSHOT_PATH))


def test_cached_snapshots_are_isolated_between_callers():
    first = snapshots()
    first["mutation-probe"] = True
    assert "mutation-probe" not in snapshots()


# Registered builtins (SW-7031): a registered id's row comes from the compiled registry, with
# the measured memory facts of config/manifests/registry/evidence.json projected into it. Those
# facts change whenever evidence is re-measured, so no snapshot may freeze them. They are checked
# against the store (same fields, same values, nothing extra) and then replaced by a marker;
# every product field stays frozen. A registered row is frozen once, in the model manifest
# response and the unredacted worker claim; every other job snapshot folds its embedded copy to
# a marker after asserting it equals that catalog row modulo the public job redaction.
REGISTRY_DIR = ROOT / "config" / "manifests" / "registry"
REGISTRY_EVIDENCE_MARKER = "<registry-evidence>"
MANIFEST_CARRIER_RE = re.compile(r"ManifestEntr(?:y|ies)$")


@lru_cache(maxsize=None)
def registry_evidence() -> dict[str, Any]:
    return json.loads((REGISTRY_DIR / "evidence.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def registered_builtin_products() -> dict[str, Any]:
    specs = json.loads((REGISTRY_DIR / "specs.json").read_text(encoding="utf-8"))
    return {model_id: spec["product"] for model_id, spec in specs.items()}


@lru_cache(maxsize=None)
def registry_evidence_fields() -> dict[str, frozenset[str]]:
    """Every field name the evidence store projects, per location, read from the store itself
    so a newly measured field is covered without editing this harness. An unknown location
    raises: the registry's projection would need a matching rule here."""
    fields: dict[str, set[str]] = {
        "model": set(), "mlx": set(), "candle": set(), "footprint": set(),
        "depth.admission": set(), "depth.admission.lane": set(),
    }
    for record in registry_evidence().values():
        for location, values in record.items():
            if location == "downloads":
                for download in values:
                    fields["footprint"].update(download["footprint"])
            elif location == "depth":
                # SW-7032: a depth job's measured admission facts (`depth.admission`).
                assert set(values) == {"admission"}, f"unknown depth evidence {sorted(values)!r}"
                for name, value in values["admission"].items():
                    if isinstance(value, dict):
                        fields["depth.admission.lane"].update(value)
                    else:
                        fields["depth.admission"].add(name)
            else:
                fields[location].update(values)
    return {location: frozenset(names) for location, names in fields.items()}


def _mask_evidence_fields(target: dict, measured: dict, names: frozenset[str], where: str) -> None:
    present = {name: target[name] for name in names if name in target}
    assert present == measured, (
        f"{where}: projected memory evidence {present!r} does not match the evidence store {measured!r}"
    )
    for name in measured:
        target[name] = REGISTRY_EVIDENCE_MARKER


def mask_registry_evidence(entry: dict) -> None:
    """Replace a model row's evidence-projected values with a marker, in place, after asserting
    the row carries exactly the store's fields and values (none at all for an id the store does
    not measure). Also covers a catalog row's `variants[].footprint`, a copy of its download's."""
    model_id = entry.get("id")
    record = registry_evidence().get(model_id, {})
    names = registry_evidence_fields()
    _mask_evidence_fields(entry, record.get("model", {}), names["model"], f"{model_id}")
    for backend in ("mlx", "candle"):
        block = entry.get(backend)
        if isinstance(block, dict):
            _mask_evidence_fields(block, record.get(backend, {}), names[backend], f"{model_id}.{backend}")
        else:
            assert backend not in record, f"{model_id}: evidence for absent {backend} block"
    admission = (entry.get("depth") or {}).get("admission")
    measured_admission = record.get("depth", {}).get("admission", {})
    if isinstance(admission, dict):
        _mask_evidence_fields(
            admission,
            {k: v for k, v in measured_admission.items() if not isinstance(v, dict)},
            names["depth.admission"],
            f"{model_id}.depth.admission",
        )
        for backend in ("mlx", "candle"):
            lane = admission.get(backend)
            if isinstance(lane, dict):
                _mask_evidence_fields(
                    lane, measured_admission.get(backend, {}), names["depth.admission.lane"],
                    f"{model_id}.depth.admission.{backend}",
                )
            else:
                assert backend not in measured_admission, f"{model_id}: depth evidence for absent {backend} lane"
    else:
        assert not measured_admission, f"{model_id}: depth evidence for absent depth.admission"
    mask_memory_guidance(entry)
    downloads = entry.get("downloads") or []
    measured_by_index: dict[int, dict] = {}
    for measured in record.get("downloads", []):
        identity = {key: value for key, value in measured.items() if key != "footprint"}
        matches = [
            index
            for index, download in enumerate(downloads)
            if all(download.get(key) == value for key, value in identity.items())
        ]
        assert len(matches) == 1, f"{model_id}: evidence download {identity!r} matched {matches!r}"
        measured_by_index[matches[0]] = measured["footprint"]
    for variant in entry.get("variants") or []:
        footprint = variant.get("footprint")
        if footprint is None:
            continue
        owners = [index for index, download in enumerate(downloads) if download.get("variant") == variant.get("variant")]
        assert len(owners) == 1, f"{model_id}: variant {variant.get('variant')!r} has owners {owners!r}"
        assert footprint == downloads[owners[0]].get("footprint"), f"{model_id}: variant footprint drifted from its download"
        _mask_evidence_fields(
            footprint, measured_by_index.get(owners[0], {}), names["footprint"], f"{model_id}.variants"
        )
    for index, download in enumerate(downloads):
        footprint = download.get("footprint")
        if isinstance(footprint, dict):
            _mask_evidence_fields(
                footprint, measured_by_index.get(index, {}), names["footprint"], f"{model_id}.downloads[{index}]"
            )
        else:
            assert index not in measured_by_index, f"{model_id}: measured download {index} has no footprint"


MEMORY_GUIDANCE_LANE_KEYS = frozenset(
    {"suggestion", "tiers", "blanketGb", "entryFloorGb", "uncalibrated", "resolutionMinHostGb"}
)


def mask_memory_guidance(entry: dict) -> None:
    """A catalog row's `memoryGuidance` (SW-7032) is computed from the row's evidence-projected
    memory facts and the API's tier-suggestion policy, so its values move whenever evidence is
    re-measured. Freeze its lane shape (both lanes, each lane's fields) and replace the values with
    a marker; the values are pinned by the Rust guidance tests and the native catalog fixture."""
    guidance = entry.get("memoryGuidance")
    if guidance is None:
        return
    assert set(guidance) == {"mlx", "candle"}, f"{entry.get('id')}: memoryGuidance lanes {sorted(guidance)!r}"
    masked = {}
    for lane, fields in guidance.items():
        extra = set(fields) - MEMORY_GUIDANCE_LANE_KEYS - ({"endpointEstimate"} if lane == "candle" else set())
        assert MEMORY_GUIDANCE_LANE_KEYS <= set(fields) and not extra, (
            f"{entry.get('id')}.memoryGuidance.{lane}: fields {sorted(fields)!r}"
        )
        masked[lane] = {field: REGISTRY_EVIDENCE_MARKER for field in sorted(fields)}
    entry["memoryGuidance"] = masked


def _strip_public_locators(value: Any) -> Any:
    """The public job redaction as it applies to a catalog row: `paths` blocks and `path`
    strings never reach a public job payload (apps/rust-api/src/job_redaction.rs)."""
    if isinstance(value, dict):
        return {
            key: _strip_public_locators(item)
            for key, item in value.items()
            if key != "paths" and not (key.lower() == "path" and isinstance(item, str))
        }
    if isinstance(value, list):
        return [_strip_public_locators(item) for item in value]
    return value


def fold_registered_entries(value: Any, catalog: dict[str, dict], *, freeze: bool) -> Any:
    """Return `value` with every embedded registered-builtin manifest entry checked against
    the catalog row for its id. `freeze=False` (public job payloads) replaces the entry with a
    marker after asserting it equals the row modulo the public redaction; `freeze=True` (the
    unredacted worker claim) keeps it, evidence masked, after asserting it equals the row."""
    products = registered_builtin_products()

    def fold_entry(entry: Any, where: str) -> Any:
        if not isinstance(entry, dict) or entry.get("id") not in products:
            return walk(entry, where)
        model_id = entry["id"]
        required = set(products[model_id]) - ({"paths"} if not freeze else set())
        assert required <= set(entry), f"{where}: product fields {sorted(required - set(entry))} are missing"
        row = catalog.get(model_id)
        if row is None:
            # A runtime-injected entry (an auxiliary detector, a job's own video model) that
            # this harness's catalog does not list: this snapshot is its only frozen copy.
            masked = copy.deepcopy(entry)
            mask_registry_evidence(masked)
            return masked
        assert set(entry) <= set(row), f"{where}: keys {sorted(set(entry) - set(row))} are not catalog fields"
        expected = {key: row[key] for key in entry}
        if not freeze:
            expected = _strip_public_locators(expected)
        diffs = diff_values(expected, entry, where)
        assert not diffs, f"{where}: embedded {model_id} entry differs from its catalog row:\n" + "\n".join(diffs)
        if not freeze:
            return f"<registered-model-entry:{model_id}>"
        masked = copy.deepcopy(entry)
        mask_registry_evidence(masked)
        return masked

    def walk(item: Any, where: str) -> Any:
        if isinstance(item, dict):
            folded = {}
            for key, child in item.items():
                child_where = f"{where}.{key}"
                if MANIFEST_CARRIER_RE.search(key) and isinstance(child, list):
                    folded[key] = [fold_entry(entry, f"{child_where}[{i}]") for i, entry in enumerate(child)]
                elif MANIFEST_CARRIER_RE.search(key):
                    folded[key] = fold_entry(child, child_where)
                else:
                    folded[key] = walk(child, child_where)
            return folded
        if isinstance(item, list):
            return [walk(child, f"{where}[{i}]") for i, child in enumerate(item)]
        return item

    return walk(value, "$")


def contains_registered_entry(value: Any) -> bool:
    products = registered_builtin_products()
    if isinstance(value, dict):
        for key, child in value.items():
            entries = child if isinstance(child, list) else [child]
            if MANIFEST_CARRIER_RE.search(key) and any(
                isinstance(entry, dict) and entry.get("id") in products for entry in entries
            ):
                return True
            if contains_registered_entry(child):
                return True
    elif isinstance(value, list):
        return any(contains_registered_entry(child) for child in value)
    return False


def normalized_catalog(runtime: ContractRuntime) -> dict[str, dict]:
    response = runtime.request("GET", "/api/v1/models")
    assert response.status_code == 200, f"{runtime.name} model catalog failed: {response.body}"
    rows = normalize_contract(response.body, runtime.roots)
    return {row["id"]: row for row in rows}


def assert_snapshot(
    label: str,
    normalized_value: Any,
    *,
    runtime: ContractRuntime | None = None,
    freeze_registered_entries: bool = False,
) -> None:
    if contains_registered_entry(normalized_value):
        assert runtime is not None, f"{label} embeds a registered model entry; pass the runtime it came from"
        normalized_value = fold_registered_entries(
            normalized_value, normalized_catalog(runtime), freeze=freeze_registered_entries
        )
    if UPDATE_SNAPSHOTS:
        UPDATED_SNAPSHOTS[label] = normalized_value
        return
    expected = snapshots().get(label)
    assert expected is not None, f"Missing Rust API contract snapshot for {label}"
    diffs = diff_values(expected, normalized_value)
    assert not diffs, (
        f"{label} no longer matches the saved Rust API contract snapshot:\n"
        + "\n".join(diffs)
        + "\n\nExpected:\n"
        + json.dumps(expected, indent=2, sort_keys=True)
        + "\n\nActual:\n"
        + json.dumps(normalized_value, indent=2, sort_keys=True)
    )


def assert_response_contract(
    label: str,
    baseline_runtime: ContractRuntime,
    candidate_runtime: ContractRuntime,
    baseline_response: ApiResponse,
    candidate_response: ApiResponse,
    *,
    expected_status: int | None = None,
    snapshot: bool = False,
    freeze_registered_entries: bool = False,
) -> Any:
    assert baseline_response.status_code == candidate_response.status_code, (
        f"{label} status drifted: baseline {baseline_response.status_code}, candidate {candidate_response.status_code}"
    )
    if expected_status is not None:
        assert baseline_response.status_code == expected_status, (
            f"{label} returned {baseline_response.status_code}, expected {expected_status}"
        )
    else:
        assert baseline_response.status_code < 400, f"{label} unexpectedly failed with {baseline_response.status_code}"
    normalized = assert_runtime_consistency(
        label,
        baseline_runtime,
        candidate_runtime,
        baseline_response.body,
        candidate_response.body,
    )
    if snapshot:
        assert_snapshot(
            label,
            normalized,
            runtime=baseline_runtime,
            freeze_registered_entries=freeze_registered_entries,
        )
    return normalized

def db_counts(project_path: Path) -> dict[str, int]:
    with sqlite3.connect(project_path / "project.db") as connection:
        return {
            "assets": connection.execute("select count(*) from assets").fetchone()[0],
            "generationSets": connection.execute("select count(*) from generation_sets").fetchone()[0],
            "timelines": connection.execute("select count(*) from timelines").fetchone()[0],
        }


def sidecar_payload(project_path: Path, asset_id: str) -> dict[str, Any]:
    for sidecar in project_path.glob("assets/**/*.sceneworks.json"):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("id") == asset_id:
            return payload
    raise AssertionError(f"Missing sidecar for {asset_id} under {project_path}")


def character_sidecar_payload(project_path: Path, character_id: str) -> dict[str, Any]:
    for sidecar in project_path.glob("characters/*.sceneworks.character.json"):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("id") == character_id:
            return payload
    raise AssertionError(f"Missing character sidecar for {character_id} under {project_path}")


def write_person_track_sidecar(runtime: ContractRuntime) -> None:
    assert runtime.project_id is not None
    assert runtime.project_path is not None
    track_dir = runtime.project_path / "person-tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    (track_dir / "track_fixture.sceneworks.person-track.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "id": "track_fixture",
                "projectId": runtime.project_id,
                "name": "Hero",
                "createdAt": "2026-05-17T00:00:00Z",
                "sourceAssetId": "asset-video",
                "representativeFrameAssetId": "asset-frame",
                "frames": [
                    {
                        "timestamp": 0.0,
                        "box": {"x": 0.3, "y": 0.2, "width": 0.2, "height": 0.6},
                        "confidence": 0.91,
                        "detected": True,
                        "mask": None,
                    },
                    {
                        "timestamp": 0.5,
                        "box": {"x": 0.34, "y": 0.2, "width": 0.2, "height": 0.6},
                        "confidence": 0.42,
                        "detected": True,
                        "mask": None,
                        "flags": ["low_confidence"],
                    },
                ],
                "corrections": [],
                "status": {"correctionState": "ready_for_box_corrections"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def create_projects(runtimes: list[ContractRuntime], baseline_runtime: ContractRuntime, candidate_runtime: ContractRuntime) -> None:
    responses = [runtime.request("POST", "/api/v1/projects", json_payload={"name": "Parity Project"}) for runtime in runtimes]
    for runtime, response in zip(runtimes, responses):
        runtime.project_id = response.body["id"]
        runtime.project_path = Path(response.body["path"])
    assert_response_contract(
        "project create response",
        baseline_runtime,
        candidate_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=True,
    )


def upload_assets(runtimes: list[ContractRuntime], baseline_runtime: ContractRuntime, candidate_runtime: ContractRuntime) -> None:
    responses = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/assets",
            files={"file": ("fixture image.png", PNG_1X1, "image/png")},
        )
        for runtime in runtimes
    ]
    for runtime, response in zip(runtimes, responses):
        runtime.asset_id = response.body["id"]
    assert_response_contract(
        "asset upload sidecar response",
        baseline_runtime,
        candidate_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=True,
    )


def create_character_pair(
    runtimes: list[ContractRuntime],
    baseline_runtime: ContractRuntime,
    candidate_runtime: ContractRuntime,
    *,
    payload: dict[str, Any] | None = None,
    label: str = "character create response",
    snapshot: bool = True,
) -> tuple[list[str], Any]:
    responses = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/characters",
            json_payload=payload
            or {"name": "Mira", "type": "person", "description": "Lead performer"},
        )
        for runtime in runtimes
    ]
    character_ids = [response.body["id"] for response in responses]
    normalized = assert_response_contract(
        label,
        baseline_runtime,
        candidate_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=snapshot,
    )
    return character_ids, normalized


def attach_character_reference_pair(
    runtimes: list[ContractRuntime],
    baseline_runtime: ContractRuntime,
    candidate_runtime: ContractRuntime,
    character_ids: list[str],
    *,
    approved: bool = True,
    label: str = "character reference attach response",
    snapshot: bool = True,
) -> Any:
    responses = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/references",
            json_payload={
                "assetId": runtime.asset_id,
                "approved": approved,
                "role": "hero" if approved else "reference",
                "notes": "Approved face" if approved else "Primary face",
            },
        )
        for runtime, character_id in zip(runtimes, character_ids)
    ]
    return assert_response_contract(
        label,
        baseline_runtime,
        candidate_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=snapshot,
    )


def create_character_look_pair(
    runtimes: list[ContractRuntime],
    baseline_runtime: ContractRuntime,
    candidate_runtime: ContractRuntime,
    character_ids: list[str],
    *,
    snapshot: bool = True,
) -> tuple[list[str], Any]:
    responses = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/looks",
            json_payload={
                "name": "Rain coat",
                "description": "Night exterior look",
                "approvedReferenceIds": [runtime.asset_id],
                "recipeSettings": {"style": "noir"},
            },
        )
        for runtime, character_id in zip(runtimes, character_ids)
    ]
    normalized = assert_response_contract(
        "character look create response",
        baseline_runtime,
        candidate_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=snapshot,
    )
    look_ids = [response.body["looks"][0]["id"] for response in responses]
    return look_ids, normalized


def write_lora_sources(runtimes: list[ContractRuntime]) -> list[Path]:
    lora_sources = []
    for runtime in runtimes:
        # The parity harness root contains each runtime's data dir, so sourcePath is both valid
        # for the API and normalized to <runtime-root> in snapshots.
        lora_dir = runtime.roots[0] / "data" / "loras"
        lora_dir.mkdir(parents=True, exist_ok=True)
        lora_source = lora_dir / "mira.safetensors"
        lora_source.write_bytes(safetensors_bytes())
        lora_sources.append(lora_source)
    return lora_sources


def attach_character_lora_pair(
    runtimes: list[ContractRuntime],
    baseline_runtime: ContractRuntime,
    candidate_runtime: ContractRuntime,
    character_ids: list[str],
    *,
    snapshot: bool = True,
) -> tuple[list[str], Any, list[Path]]:
    responses = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/loras",
            json_payload={
                "name": "Mira LoRA",
                "sourcePath": str(lora_source),
                "triggerWords": ["mira"],
                "compatibility": {"families": ["z-image"]},
            },
        )
        for runtime, character_id, lora_source in zip(runtimes, character_ids, write_lora_sources(runtimes))
    ]
    normalized = assert_response_contract(
        "character lora attach response",
        baseline_runtime,
        candidate_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=snapshot,
    )
    lora_link_ids = [response.body["loras"][0]["id"] for response in responses]
    copied_paths = []
    for runtime, response in zip(runtimes, responses):
        copied_path = runtime.project_path / response.body["loras"][0]["projectPath"]
        assert copied_path.exists(), f"{runtime.name} did not copy character LoRA to {copied_path}"
        assert copied_path.read_bytes() == safetensors_bytes()
        copied_paths.append(copied_path)
    copied_rel_paths = [response.body["loras"][0]["projectPath"] for response in responses]
    assert_runtime_consistency("character lora copied project path", baseline_runtime, candidate_runtime, copied_rel_paths[0], copied_rel_paths[1])
    return lora_link_ids, normalized, copied_paths


def read_sse_message(lines: Any) -> dict[str, Any]:
    event = "message"
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if not data_lines:
                continue
            data = "\n".join(data_lines)
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError:
                parsed_data = data
            return {"event": event, "data": parsed_data}
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    raise AssertionError("SSE stream ended before a complete event arrived")


def collect_sse_events_during(runtime: ContractRuntime, action: Any, expected_events: tuple[str, ...]) -> tuple[Any, list[dict[str, Any]]]:
    timeout = httpx.Timeout(10.0, connect=5.0, read=5.0)
    with httpx.Client(base_url=runtime.api.base_url, timeout=timeout) as client:
        ticket = client.post("/api/v1/jobs/events/ticket").json()["ticket"]
        with client.stream("GET", f"/api/v1/jobs/events?ticket={ticket}") as response:
            response.raise_for_status()
            lines = response.iter_lines()
            ready = read_sse_message(lines)
            assert ready == {"event": "ready", "data": {"status": "connected"}}
            initial_jobs = read_sse_message(lines)
            assert initial_jobs["event"] == "jobs.snapshot"
            initial_queue = read_sse_message(lines)
            assert initial_queue["event"] == "queue.updated"
            action_result = action()
            if action_result.status_code >= 400:
                raise AssertionError(
                    f"{runtime.name} action failed before expected SSE events {expected_events}: "
                    f"{action_result.status_code} {json.dumps(action_result.body, sort_keys=True)}"
                )
            events: list[dict[str, Any]] = []
            try:
                while len(events) < len(expected_events):
                    event = read_sse_message(lines)
                    if event["event"] in expected_events:
                        events.append(event)
            except httpx.ReadTimeout as exc:
                raise AssertionError(f"{runtime.name} did not emit expected SSE events {expected_events}: {events}") from exc
    assert [event["event"] for event in events] == list(expected_events)
    return action_result, events


def test_system_manifest_and_http_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes

    for label, path, method, expected in [
        ("health response", "/api/v1/health", "GET", 200),
        ("access response", "/api/v1/access", "GET", 200),
        ("auth verify response", "/api/v1/auth/verify", "POST", 200),
        ("event ticket response", "/api/v1/jobs/events/ticket", "POST", 200),
        ("model manifest response", "/api/v1/models", "GET", 200),
        ("lora manifest response", "/api/v1/loras?modelFamily=wan-video", "GET", 200),
    ]:
        responses = [runtime.request(method, path) for runtime in contract_runtimes]
        normalized = assert_response_contract(
            label,
            baseline_runtime,
            candidate_runtime,
            responses[0],
            responses[1],
            expected_status=expected,
            snapshot=label != "model manifest response",
        )
        if label == "model manifest response":
            turbo = next(model for model in normalized if model["id"] == "z_image_turbo")
            encoder_options = turbo.pop("textEncoderOptions", None)
            if encoder_options is not None:
                assert encoder_options == [
                    {
                        "description": "Uses the text encoder paired with z_image_turbo.",
                        "id": "default",
                        "isDefault": True,
                        "label": "Model encoder (default)",
                    }
                ]
            # z_image_turbo is a registered builtin, so its row is the registry's, which
            # declares an `mlx` block. The catalog probes MLX install/conversion state for
            # such a row only on macOS (off macOS only for candle-native converters,
            # sc-20529), so these keys are host facts, not lane contract: assert the host
            # split, then remove exactly them before comparing with the Linux snapshot.
            mlx_host_state = {
                key: turbo.pop(key)
                for key in ("mlxConversionState", "mlxConvertedPath", "mlxInstallState")
                if key in turbo
            }
            if sys.platform == "darwin":
                assert mlx_host_state == {
                    "mlxConversionState": "ready",
                    "mlxConvertedPath": None,
                    "mlxInstallState": "missing",
                }, mlx_host_state
            else:
                assert mlx_host_state == {}, mlx_host_state
            for model in normalized:
                mask_registry_evidence(model)
            assert_snapshot(label, normalized)

    cors_headers = {
        "origin": "http://localhost:5173",
        "access-control-request-method": "POST",
        "access-control-request-headers": "X-SceneWorks-Token",
    }
    cors = [runtime.request("OPTIONS", "/api/v1/jobs", headers=cors_headers) for runtime in contract_runtimes]
    assert cors[0].status_code == cors[1].status_code == 200
    assert cors[0].headers.get("access-control-allow-origin") == cors[1].headers.get("access-control-allow-origin")
    assert cors[0].headers.get("access-control-allow-origin") == "http://localhost:5173"
    for response in cors:
        allow_headers = response.headers.get("access-control-allow-headers", "").lower()
        assert "x-sceneworks-token" in allow_headers


def test_project_asset_sidecar_delete_and_db_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)

    listed_projects = [runtime.request("GET", "/api/v1/projects") for runtime in contract_runtimes]
    assert_response_contract("project list response", baseline_runtime, candidate_runtime, listed_projects[0], listed_projects[1], snapshot=True)

    upload_assets(contract_runtimes, baseline_runtime, candidate_runtime)
    patched_assets = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/assets/{runtime.asset_id}/status",
            json_payload={"favorite": True, "rating": 4},
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract(
        "asset status project DB update response",
        baseline_runtime,
        candidate_runtime,
        patched_assets[0],
        patched_assets[1],
        expected_status=200,
        snapshot=True,
    )

    listed_assets = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/assets") for runtime in contract_runtimes]
    assert_response_contract("asset list response", baseline_runtime, candidate_runtime, listed_assets[0], listed_assets[1], snapshot=True)

    sidecars = [sidecar_payload(runtime.project_path, runtime.asset_id) for runtime in contract_runtimes]
    normalized_sidecar = assert_runtime_consistency("persisted upload sidecar", baseline_runtime, candidate_runtime, sidecars[0], sidecars[1])
    assert_snapshot("persisted upload sidecar", normalized_sidecar)

    counts = [db_counts(runtime.project_path) for runtime in contract_runtimes]
    assert_runtime_consistency("project DB counts after sidecar writes", baseline_runtime, candidate_runtime, counts[0], counts[1])

    reindexed = [runtime.request("POST", f"/api/v1/projects/{runtime.project_id}/reindex") for runtime in contract_runtimes]
    assert_response_contract("project reindex response", baseline_runtime, candidate_runtime, reindexed[0], reindexed[1], expected_status=200, snapshot=True)

    deleted = [
        runtime.request("DELETE", f"/api/v1/projects/{runtime.project_id}/assets/{runtime.asset_id}")
        for runtime in contract_runtimes
    ]
    assert_response_contract("asset delete response", baseline_runtime, candidate_runtime, deleted[0], deleted[1], expected_status=200, snapshot=True)

    visible_after_delete = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/assets") for runtime in contract_runtimes]
    assert_response_contract(
        "asset list after delete response",
        baseline_runtime,
        candidate_runtime,
        visible_after_delete[0],
        visible_after_delete[1],
        expected_status=200,
        snapshot=True,
    )

    purged = [
        runtime.request("DELETE", f"/api/v1/projects/{runtime.project_id}/assets/{runtime.asset_id}/purge")
        for runtime in contract_runtimes
    ]
    assert_response_contract("asset purge response", baseline_runtime, candidate_runtime, purged[0], purged[1], expected_status=200, snapshot=True)


@pytest.mark.parity
def test_character_lifecycle_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)

    character_ids, _ = create_character_pair(contract_runtimes, baseline_runtime, candidate_runtime)

    listed_characters = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/characters")
        for runtime in contract_runtimes
    ]
    assert_response_contract(
        "character list response",
        baseline_runtime,
        candidate_runtime,
        listed_characters[0],
        listed_characters[1],
        expected_status=200,
        snapshot=True,
    )

    fetched_characters = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/characters/{character_id}")
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "character detail response",
        baseline_runtime,
        candidate_runtime,
        fetched_characters[0],
        fetched_characters[1],
        expected_status=200,
        snapshot=True,
    )

    updated_characters = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}",
            json_payload={"description": "Lead performer, updated", "archived": False},
        )
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "character update response",
        baseline_runtime,
        candidate_runtime,
        updated_characters[0],
        updated_characters[1],
        expected_status=200,
        snapshot=True,
    )

    archived_characters = [
        runtime.request("POST", f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/archive")
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    normalized_post_archive_response = assert_response_contract(
        "character archive response",
        baseline_runtime,
        candidate_runtime,
        archived_characters[0],
        archived_characters[1],
        expected_status=200,
        snapshot=True,
    )

    hidden_archived = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/characters")
        for runtime in contract_runtimes
    ]
    assert_response_contract(
        "character list after archive response",
        baseline_runtime,
        candidate_runtime,
        hidden_archived[0],
        hidden_archived[1],
        expected_status=200,
        snapshot=True,
    )

    visible_archived = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/characters?includeArchived=true")
        for runtime in contract_runtimes
    ]
    assert_response_contract(
        "character list include archived response",
        baseline_runtime,
        candidate_runtime,
        visible_archived[0],
        visible_archived[1],
        expected_status=200,
        snapshot=True,
    )

    delete_archive_ids, _ = create_character_pair(
        contract_runtimes,
        baseline_runtime,
        candidate_runtime,
        payload={"name": "Delete Archive", "type": "object"},
        label="character create for delete archive response",
    )
    delete_archive_responses = [
        runtime.request(
            "DELETE",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}",
        )
        for runtime, character_id in zip(contract_runtimes, delete_archive_ids)
    ]
    normalized_delete_archive_response = assert_response_contract(
        "character delete archive response",
        baseline_runtime,
        candidate_runtime,
        delete_archive_responses[0],
        delete_archive_responses[1],
        expected_status=200,
        snapshot=True,
    )
    assert normalized_delete_archive_response == normalized_post_archive_response

    purge_ids, _ = create_character_pair(
        contract_runtimes,
        baseline_runtime,
        candidate_runtime,
        payload={"name": "Purge Me", "type": "object"},
        label="character create for purge response",
    )
    purge_responses = [
        runtime.request("DELETE", f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/purge")
        for runtime, character_id in zip(contract_runtimes, purge_ids)
    ]
    assert_response_contract(
        "character purge response",
        baseline_runtime,
        candidate_runtime,
        purge_responses[0],
        purge_responses[1],
        expected_status=200,
        snapshot=True,
    )
    for runtime, character_id in zip(contract_runtimes, purge_ids):
        assert not (runtime.project_path / "characters" / f"{character_id}.sceneworks.character.json").exists()


@pytest.mark.parity
def test_character_reference_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)
    upload_assets(contract_runtimes, baseline_runtime, candidate_runtime)
    character_ids, _ = create_character_pair(contract_runtimes, baseline_runtime, candidate_runtime)

    attach_character_reference_pair(
        contract_runtimes,
        baseline_runtime,
        candidate_runtime,
        character_ids,
        approved=False,
    )

    updated_references = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/references/{runtime.asset_id}",
            json_payload={"approved": True, "role": "hero", "notes": "Approved face"},
        )
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "character reference update response",
        baseline_runtime,
        candidate_runtime,
        updated_references[0],
        updated_references[1],
        expected_status=200,
        snapshot=True,
    )

    sidecars = [
        character_sidecar_payload(runtime.project_path, character_id)
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    normalized_sidecar = assert_runtime_consistency("persisted character sidecar", baseline_runtime, candidate_runtime, sidecars[0], sidecars[1])
    assert_snapshot("persisted character sidecar", normalized_sidecar)

    listed_assets = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/assets") for runtime in contract_runtimes]
    normalized_asset_response = assert_response_contract(
        "asset list after character reference response",
        baseline_runtime,
        candidate_runtime,
        listed_assets[0],
        listed_assets[1],
        expected_status=200,
        snapshot=True,
    )
    assert normalized_asset_response[0]["metadata"]["characterReferences"][0]["characterId"] == "character_fixture"

    asset_sidecars = [sidecar_payload(runtime.project_path, runtime.asset_id) for runtime in contract_runtimes]
    normalized_asset_sidecar = assert_runtime_consistency(
        "asset sidecar after character reference",
        baseline_runtime,
        candidate_runtime,
        asset_sidecars[0],
        asset_sidecars[1],
    )
    assert_snapshot("asset sidecar after character reference", normalized_asset_sidecar)

    removed_references = [
        runtime.request(
            "DELETE",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/references/{runtime.asset_id}",
        )
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "character reference delete response",
        baseline_runtime,
        candidate_runtime,
        removed_references[0],
        removed_references[1],
        expected_status=200,
        snapshot=True,
    )

    cleaned_assets = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/assets") for runtime in contract_runtimes]
    normalized_cleaned_asset_response = assert_response_contract(
        "asset list after character reference removal response",
        baseline_runtime,
        candidate_runtime,
        cleaned_assets[0],
        cleaned_assets[1],
        expected_status=200,
        snapshot=True,
    )
    assert normalized_cleaned_asset_response[0]["metadata"]["characterReferences"] == []

    cleaned_asset_sidecars = [sidecar_payload(runtime.project_path, runtime.asset_id) for runtime in contract_runtimes]
    normalized_cleaned_asset_sidecar = assert_runtime_consistency(
        "asset sidecar after character reference removal",
        baseline_runtime,
        candidate_runtime,
        cleaned_asset_sidecars[0],
        cleaned_asset_sidecars[1],
    )
    assert_snapshot("asset sidecar after character reference removal", normalized_cleaned_asset_sidecar)
    for asset_sidecar in cleaned_asset_sidecars:
        assert asset_sidecar["metadata"]["characterReferences"] == []


@pytest.mark.parity
def test_character_look_and_lora_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)
    upload_assets(contract_runtimes, baseline_runtime, candidate_runtime)
    character_ids, _ = create_character_pair(contract_runtimes, baseline_runtime, candidate_runtime)
    attach_character_reference_pair(contract_runtimes, baseline_runtime, candidate_runtime, character_ids, snapshot=False)

    look_ids, normalized_look_response = create_character_look_pair(
        contract_runtimes,
        baseline_runtime,
        candidate_runtime,
        character_ids,
    )

    updated_looks = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/looks/{look_id}",
            json_payload={
                "name": "Rain coat revised",
                "description": "Night exterior hero look",
                "approvedReferenceIds": [runtime.asset_id],
                "recipeSettings": {"style": "noir", "lens": "85mm"},
            },
        )
        for runtime, character_id, look_id in zip(contract_runtimes, character_ids, look_ids)
    ]
    normalized_look_update_response = assert_response_contract(
        "character look update response",
        baseline_runtime,
        candidate_runtime,
        updated_looks[0],
        updated_looks[1],
        expected_status=200,
        snapshot=True,
    )

    lora_link_ids, _, _ = attach_character_lora_pair(
        contract_runtimes,
        baseline_runtime,
        candidate_runtime,
        character_ids,
    )

    updated_loras = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/loras/{link_id}",
            json_payload={
                "name": "Mira LoRA revised",
                "triggerWords": ["mira", "rain"],
                "defaultWeight": 0.65,
                "compatibility": {"families": ["z-image"], "notes": "parity"},
                "scope": "project",
            },
        )
        for runtime, character_id, link_id in zip(contract_runtimes, character_ids, lora_link_ids)
    ]
    assert_response_contract(
        "character lora update response",
        baseline_runtime,
        candidate_runtime,
        updated_loras[0],
        updated_loras[1],
        expected_status=200,
        snapshot=True,
    )

    deleted_looks = [
        runtime.request(
            "DELETE",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/looks/{look_id}",
        )
        for runtime, character_id, look_id in zip(contract_runtimes, character_ids, look_ids)
    ]
    assert_response_contract(
        "character look delete response",
        baseline_runtime,
        candidate_runtime,
        deleted_looks[0],
        deleted_looks[1],
        expected_status=200,
        snapshot=True,
    )

    detached_loras = [
        runtime.request(
            "DELETE",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/loras/{link_id}",
        )
        for runtime, character_id, link_id in zip(contract_runtimes, character_ids, lora_link_ids)
    ]
    assert_response_contract(
        "character lora delete response",
        baseline_runtime,
        candidate_runtime,
        detached_loras[0],
        detached_loras[1],
        expected_status=200,
        snapshot=True,
    )

    assert normalized_look_response["looks"][0]["approvedReferenceIds"] == ["asset_fixture"]
    assert normalized_look_update_response["looks"][0]["recipeSettings"]["lens"] == "85mm"


@pytest.mark.parity
def test_character_test_job_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)
    upload_assets(contract_runtimes, baseline_runtime, candidate_runtime)
    character_ids, _ = create_character_pair(contract_runtimes, baseline_runtime, candidate_runtime)
    attach_character_reference_pair(contract_runtimes, baseline_runtime, candidate_runtime, character_ids, snapshot=False)
    look_ids, _ = create_character_look_pair(contract_runtimes, baseline_runtime, candidate_runtime, character_ids, snapshot=False)
    _, _, copied_lora_paths = attach_character_lora_pair(contract_runtimes, baseline_runtime, candidate_runtime, character_ids, snapshot=False)

    # SW-7030: the character's bound LoRA entry becomes a Digest member of the frozen
    # route, and admission refuses an image job whose local source closure no
    # authenticated worker advertises. Register the harness's fake worker with the
    # same portable envelope a production worker sends — platform + route on the
    # claim envelope, plus the `registered_digest:lora:...` claim over the copied
    # adapter bytes — so the test job enqueues exactly as it would against a live
    # worker inventory.
    for runtime, character_id, copied_lora_path in zip(contract_runtimes, character_ids, copied_lora_paths):
        character = runtime.request(
            "GET",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}",
        ).body
        lora_link_id = character["loras"][0]["id"]
        claim = portable_lora_source_claim(lora_link_id, copied_lora_path)
        envelope = portable_claim_envelope("character-test-worker", ["image_generate"])
        envelope["registration"]["portableSourceClaims"] = [claim]
        registration = runtime.request("POST", "/api/v1/workers/register", json_payload=envelope["registration"])
        assert registration.status_code == 200, f"{runtime.name} worker registration failed: {registration.body}"

    test_jobs = []
    event_sets = []
    for runtime, character_id, look_id in zip(contract_runtimes, character_ids, look_ids):
        response, events = collect_sse_events_during(
            runtime,
            lambda runtime=runtime, character_id=character_id, look_id=look_id: runtime.request(
                "POST",
                f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/test-jobs",
                json_payload={"prompt": "portrait in rain", "lookId": look_id, "count": 2},
            ),
            ("job.updated", "queue.updated"),
        )
        test_jobs.append(response)
        event_sets.append(events)

    assert_response_contract(
        "character test job response",
        baseline_runtime,
        candidate_runtime,
        test_jobs[0],
        test_jobs[1],
        expected_status=201,
        snapshot=True,
    )
    normalized_events = assert_runtime_consistency(
        "character test job SSE events",
        baseline_runtime,
        candidate_runtime,
        event_sets[0],
        event_sets[1],
    )
    assert_snapshot("character test job SSE events", normalized_events, runtime=baseline_runtime)


def test_timeline_and_worker_job_creation_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)
    upload_assets(contract_runtimes, baseline_runtime, candidate_runtime)

    created_timelines = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/timelines",
            json_payload={"name": "Main timeline", "aspectRatio": "16:9", "fps": 24},
        )
        for runtime in contract_runtimes
    ]
    created_contract = assert_response_contract(
        "timeline create response",
        baseline_runtime,
        candidate_runtime,
        created_timelines[0],
        created_timelines[1],
        expected_status=201,
        snapshot=True,
    )
    assert created_contract["revision"] == 1

    updated_timelines = []
    for runtime, created in zip(contract_runtimes, created_timelines):
        timeline = json.loads(json.dumps(created.body))
        timeline_id = timeline["id"]
        timeline["tracks"][0]["items"] = [
            {
                "id": "item-1",
                "trackId": "track_main",
                "assetId": runtime.asset_id,
                "type": "image",
                "displayName": "Still",
                "sourceIn": 0,
                "sourceOut": 1,
                "timelineStart": 0,
                "timelineEnd": 1,
                "speed": 1,
                "fit": "fit",
                "volume": 1,
            }
        ]
        updated_timelines.append(
            runtime.request(
                "PUT",
                f"/api/v1/projects/{runtime.project_id}/timelines/{timeline_id}",
                json_payload={"timeline": timeline, "expectedRevision": timeline["revision"]},
            )
        )
    updated_contract = assert_response_contract(
        "timeline update response",
        baseline_runtime,
        candidate_runtime,
        updated_timelines[0],
        updated_timelines[1],
        expected_status=200,
        snapshot=True,
    )
    assert updated_contract["revision"] == 2

    stale_saves = [
        runtime.request(
            "PUT",
            f"/api/v1/projects/{runtime.project_id}/timelines/{created.body['id']}",
            json_payload={"timeline": created.body, "expectedRevision": 1},
        )
        for runtime, created in zip(contract_runtimes, created_timelines)
    ]
    stale_contract = assert_response_contract(
        "stale timeline save response",
        baseline_runtime,
        candidate_runtime,
        stale_saves[0],
        stale_saves[1],
        expected_status=409,
    )
    assert stale_contract["code"] == "timeline_revision_conflict"
    assert stale_contract["context"] == {
        "currentRevision": 2,
        "expectedRevision": 1,
        "timelineId": "timeline_fixture",
    }

    listed_timelines = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/timelines") for runtime in contract_runtimes]
    assert_response_contract("timeline list response", baseline_runtime, candidate_runtime, listed_timelines[0], listed_timelines[1], expected_status=200, snapshot=True)

    detail_timelines = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/timelines/{created.body['id']}")
        for runtime, created in zip(contract_runtimes, created_timelines)
    ]
    detail_contract = assert_response_contract(
        "timeline detail response",
        baseline_runtime,
        candidate_runtime,
        detail_timelines[0],
        detail_timelines[1],
        expected_status=200,
        snapshot=True,
    )
    assert detail_contract["revision"] == 2

    frame_jobs = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/timelines/{created.body['id']}/items/item-1/frames",
            json_payload={"playheadSeconds": 0.5, "intendedUse": "reuse"},
        )
        for runtime, created in zip(contract_runtimes, created_timelines)
    ]
    assert_response_contract("timeline frame job response", baseline_runtime, candidate_runtime, frame_jobs[0], frame_jobs[1], expected_status=201, snapshot=True)

    export_jobs = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/timelines/{created.body['id']}/exports",
            json_payload={"resolution": 640, "fps": 24, "requestedGpu": "auto"},
        )
        for runtime, created in zip(contract_runtimes, created_timelines)
    ]
    export_contract = assert_response_contract(
        "timeline export job response",
        baseline_runtime,
        candidate_runtime,
        export_jobs[0],
        export_jobs[1],
        expected_status=201,
        snapshot=True,
    )
    assert export_contract["payload"]["timelineRevision"] == 2


def test_person_tracking_and_replace_person_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)
    for runtime in contract_runtimes:
        write_person_track_sidecar(runtime)

    listed_tracks = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/person-tracks")
        for runtime in contract_runtimes
    ]
    assert_response_contract("person track list response", baseline_runtime, candidate_runtime, listed_tracks[0], listed_tracks[1], expected_status=200, snapshot=True)

    track_details = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/person-tracks/track_fixture")
        for runtime in contract_runtimes
    ]
    assert_response_contract("person track detail response", baseline_runtime, candidate_runtime, track_details[0], track_details[1], expected_status=200, snapshot=True)

    correction_responses = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/person-tracks/track_fixture/corrections",
            json_payload={
                "corrections": [
                    {
                        "frameIndex": 0,
                        "box": {"x": 0.32, "y": 0.18, "width": 0.21, "height": 0.62},
                        "author": "ui",
                        "source": "manual",
                    },
                    {"frameIndex": 1, "rejected": True},
                ]
            },
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("person track corrections response", baseline_runtime, candidate_runtime, correction_responses[0], correction_responses[1], expected_status=200, snapshot=True)

    detection_jobs = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/person-tracks/detections",
            json_payload={"sourceAssetId": "asset-video", "sourceTimestamp": 1.25},
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("person detection job response", baseline_runtime, candidate_runtime, detection_jobs[0], detection_jobs[1], expected_status=201, snapshot=True)

    track_jobs = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/person-tracks/jobs",
            json_payload={
                "sourceAssetId": "asset-video",
                "representativeFrameAssetId": "asset-frame",
                "detection": {
                    "id": "person_1",
                    "box": {"x": 0.3, "y": 0.2, "width": 0.2, "height": 0.6},
                },
                "trackName": "Hero",
            },
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("person track job response", baseline_runtime, candidate_runtime, track_jobs[0], track_jobs[1], expected_status=201, snapshot=True)

    # `model` is named explicitly (sc-19570). This fixture used to omit it and inherit the
    # DEFAULT video model, which today is `ltx_2_3` — a model whose `replace_person` mode runs
    # only on the macOS MLX engine. The enqueue CONTRACT is platform-independent (201 with a job
    # snapshot on every host, which is what `expected_status` below pins), but the job's own
    # lifecycle is not: off-Mac such a pair is failed terminal at once with a
    # `platform_unreachable:` reason instead of queueing for a worker that can never exist. That
    # made `status` / `stage` / `error` / `completedAt` in the SNAPSHOT depend on which runner
    # recorded it, and this suite runs on `ubuntu-latest`.
    #
    # `wan_2_2` serves `replace_person` on BOTH lanes — candle Wan-VACE off-Mac, MLX on a Mac — so
    # the snapshot is stable everywhere and the fixture is testing the person-replace contract
    # rather than the default model's reachability. Naming it also makes the fixture immune to a
    # future change of the catalog default.
    replace_jobs = [
        runtime.request(
            "POST",
            "/api/v1/video/jobs",
            json_payload={
                "projectId": runtime.project_id,
                "projectName": "Parity Project",
                "model": "wan_2_2",
                "mode": "replace_person",
                "prompt": "hero walks through rain",
                "sourceClipAssetId": "asset-video",
                "personTrackId": "track_fixture",
                "characterId": "character_fixture",
            },
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("replace person job response", baseline_runtime, candidate_runtime, replace_jobs[0], replace_jobs[1], expected_status=201, snapshot=True)


def test_model_lora_and_image_job_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)

    model_jobs = [
        runtime.request("POST", "/api/v1/models/base-model/download", json_payload={"requestedGpu": ""})
        for runtime in contract_runtimes
    ]
    assert_response_contract("model download job response", baseline_runtime, candidate_runtime, model_jobs[0], model_jobs[1], expected_status=201, snapshot=True)

    lora_jobs = [
        runtime.request(
            "POST",
            "/api/v1/loras/import",
            json_payload={"repo": "owner/style-lora", "name": "Imported LoRA", "files": ["adapter.safetensors"]},
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("lora import job response", baseline_runtime, candidate_runtime, lora_jobs[0], lora_jobs[1], expected_status=201, snapshot=True)

    image_jobs = [
        runtime.request(
            "POST",
            "/api/v1/image/jobs",
            json_payload={
                "projectId": runtime.project_id,
                "projectName": "Parity Project",
                "prompt": "mist over hills",
                # SW-7030: `base-model` is an unrouted synthetic id (neither MLX- nor
                # candle-routed in the routing tables), so admission now fails it closed
                # (400 "No native route..."); z_image_turbo is the routed registered
                # builtin, so the job carries the registry's row and its pinned downloads.
                "model": "z_image_turbo",
                "seed": 101,
                "requestedGpu": "auto",
            },
        )
        for runtime in contract_runtimes
    ]
    for runtime, response in zip(contract_runtimes, image_jobs):
        runtime.job_id = response.body["id"]
    assert_response_contract("image job response", baseline_runtime, candidate_runtime, image_jobs[0], image_jobs[1], expected_status=201, snapshot=True)


def test_job_state_transition_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)

    # SW-7030: the fake worker uses the production portable route envelope (platform +
    # binding-version + backend/platform on the claim) exactly as a real worker does —
    # a plain claim is refused every bound job. The registration/claim pair is built by
    # `portable_claim_envelope`, which picks the backend from the host OS like a real worker.
    envelopes = [portable_claim_envelope("parity-worker", ["image_generate"]) for _ in contract_runtimes]
    workers = [
        runtime.request("POST", "/api/v1/workers/register", json_payload=envelope["registration"])
        for runtime, envelope in zip(contract_runtimes, envelopes)
    ]
    assert_response_contract("worker registration response", baseline_runtime, candidate_runtime, workers[0], workers[1], expected_status=200, snapshot=True)

    image_jobs = [
        runtime.request(
            "POST",
            "/api/v1/image/jobs",
            json_payload={
                "projectId": runtime.project_id,
                "projectName": "Parity Project",
                "prompt": "mist over hills",
                "model": "z_image_turbo",
                "seed": 101,
                "requestedGpu": "auto",
            },
        )
        for runtime in contract_runtimes
    ]
    for runtime, response in zip(contract_runtimes, image_jobs):
        runtime.job_id = response.body["id"]
    assert_response_contract("transition image job response", baseline_runtime, candidate_runtime, image_jobs[0], image_jobs[1], expected_status=201, snapshot=True)

    claimed_jobs = [
        runtime.request("POST", "/api/v1/jobs/claim", json_payload=envelope["claim"])
        for runtime, envelope in zip(contract_runtimes, envelopes)
    ]
    assert_response_contract(
        "job claim response",
        baseline_runtime,
        candidate_runtime,
        claimed_jobs[0],
        claimed_jobs[1],
        expected_status=200,
        snapshot=True,
        # The worker-facing claim is the one unredacted copy of the registered entry.
        freeze_registered_entries=True,
    )

    busy_workers = [
        runtime.request(
            "POST",
            "/api/v1/workers/parity-worker/heartbeat",
            json_payload={
                "status": "busy",
                "currentJobId": runtime.job_id,
                "loadedModels": ["Tongyi-MAI/Z-Image-Turbo"],
            },
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("busy worker heartbeat response", baseline_runtime, candidate_runtime, busy_workers[0], busy_workers[1], expected_status=200, snapshot=True)

    cancel_requests = [runtime.request("POST", f"/api/v1/jobs/{runtime.job_id}/cancel") for runtime in contract_runtimes]
    assert_response_contract("job cancel request response", baseline_runtime, candidate_runtime, cancel_requests[0], cancel_requests[1], expected_status=200, snapshot=True)

    canceled_jobs = [
        runtime.request(
            "POST",
            f"/api/v1/jobs/{runtime.job_id}/progress",
            json_payload={
                "status": "canceled",
                "stage": "canceled",
                "progress": 1,
                "message": "Worker canceled the job before completion.",
                "workerId": "parity-worker",
            },
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("job terminal progress response", baseline_runtime, candidate_runtime, canceled_jobs[0], canceled_jobs[1], expected_status=200, snapshot=True)

    queues = [runtime.request("GET", "/api/v1/queue") for runtime in contract_runtimes]
    assert_response_contract("queue summary after transitions", baseline_runtime, candidate_runtime, queues[0], queues[1], expected_status=200, snapshot=True)


def concurrent_claim_result(runtime: ContractRuntime) -> list[Any]:
    # `image_detail` (not `image_generate`) is the claim-race fixture: this test needs *a*
    # GPU-routed claimable job, and sc-12305 made the generic POST /api/v1/jobs reject the
    # generation job types, whose typed routes resolve a model manifest entry the generic
    # route cannot. image_detail is a real caller of this route (the web batch ops post it)
    # and is GPU-routed, so the two-workers-one-gpu race below is unchanged.
    # SW-7030: both workers use the production portable route envelope (`platform` on
    # registration, binding-version + backend/platform on the claim) so a frozen binding
    # is actually claimable by one of them, exactly as two real workers racing would be.
    envelopes = {
        worker_id: portable_claim_envelope(worker_id, ["image_detail"])
        for worker_id in ["claim-worker-a", "claim-worker-b"]
    }
    for worker_id in ["claim-worker-a", "claim-worker-b"]:
        runtime.request(
            "POST",
            "/api/v1/workers/register",
            json_payload=envelopes[worker_id]["registration"],
        )
    runtime.request("POST", "/api/v1/jobs", json_payload={"type": "image_detail", "payload": {}, "requestedGpu": "auto"})
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                runtime.request,
                "POST",
                "/api/v1/jobs/claim",
                json_payload=envelopes[worker_id]["claim"],
            )
            for worker_id in ["claim-worker-a", "claim-worker-b"]
        ]
    responses = [future.result() for future in futures]
    assert all(response.status_code == 200 for response in responses)
    claims = [response.body for response in responses]
    claims.sort(key=lambda claim: claim["job"]["id"] if claim["job"] else "")
    return claims


def test_concurrent_claim_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    claims = [concurrent_claim_result(runtime) for runtime in contract_runtimes]
    normalized_claims = assert_runtime_consistency("concurrent claim response set", baseline_runtime, candidate_runtime, claims[0], claims[1])
    assert_snapshot(
        "concurrent claim response set",
        normalized_claims,
        runtime=baseline_runtime,
        freeze_registered_entries=True,
    )
    for runtime_claims in claims:
        assert sum(1 for claim in runtime_claims if claim["job"] is not None) == 1
        assert sum(1 for claim in runtime_claims if claim["job"] is None) == 1


def test_error_case_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes

    missing_projects = [runtime.request("GET", "/api/v1/projects/project_missing") for runtime in contract_runtimes]
    assert_response_contract("missing project error response", baseline_runtime, candidate_runtime, missing_projects[0], missing_projects[1], expected_status=404, snapshot=True)

    invalid_statuses = [runtime.request("GET", "/api/v1/jobs?status=not_a_status") for runtime in contract_runtimes]
    assert_response_contract("invalid job status error response", baseline_runtime, candidate_runtime, invalid_statuses[0], invalid_statuses[1], expected_status=400, snapshot=True)

    malformed_bodies = [
        runtime.request(
            "POST",
            "/api/v1/projects",
            headers={"content-type": "application/json"},
            content='{"name":',
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("malformed json error response", baseline_runtime, candidate_runtime, malformed_bodies[0], malformed_bodies[1], expected_status=422, snapshot=True)

    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)
    characters = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/characters",
            json_payload={"name": "Errors", "type": "person"},
        )
        for runtime in contract_runtimes
    ]
    character_ids = [response.body["id"] for response in characters]

    missing_characters = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/characters/character_missing")
        for runtime in contract_runtimes
    ]
    assert_response_contract(
        "missing character error response",
        baseline_runtime,
        candidate_runtime,
        missing_characters[0],
        missing_characters[1],
        expected_status=404,
        snapshot=True,
    )

    missing_looks = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/looks/look_missing",
            json_payload={"name": "Missing"},
        )
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "missing character look error response",
        baseline_runtime,
        candidate_runtime,
        missing_looks[0],
        missing_looks[1],
        expected_status=404,
        snapshot=True,
    )

    missing_loras = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/loras/character_lora_missing",
            json_payload={"name": "Missing"},
        )
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "missing character lora error response",
        baseline_runtime,
        candidate_runtime,
        missing_loras[0],
        missing_loras[1],
        expected_status=404,
        snapshot=True,
    )

    invalid_lora_sources = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/loras",
            json_payload={"name": "Missing source", "sourcePath": str(runtime.roots[0] / "data" / "loras" / "missing.safetensors")},
        )
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "invalid character lora source error response",
        baseline_runtime,
        candidate_runtime,
        invalid_lora_sources[0],
        invalid_lora_sources[1],
        expected_status=400,
        snapshot=True,
    )
