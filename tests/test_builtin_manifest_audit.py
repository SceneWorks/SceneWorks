"""Structural audits of the live ``config/manifests/builtin.models.jsonc``.

These assertions guard the shipped catalog config, NOT worker runtime behaviour.
They were originally embedded in ``tests/test_worker_image_adapters.py`` (the
retired Python worker's adapter suite), which meant a future ``apps/worker``
deletion (epic 8283, Python eradication) would silently take the live-catalog
gates down with it. sc-8861 (F-059) extracts them here so the coverage survives
that deletion: this module parses the manifest file DIRECTLY and imports no
``scene_worker`` symbol at module scope.

The manifest is a JSONC file (the Rust API owns the canonical parser). Two
self-contained readers are inlined below rather than imported from
``tests/worker_runtime_shared.py`` (that helper module top-imports
``scene_worker``, which would re-couple these audits to the retired worker):

  * ``_strip_jsonc_comments`` + ``_load_builtin_models_manifest`` parse the file
    to a dict for every capability, UI-wiring, and per-model ``mlx`` audit.

The three character_image ENGINE-WIRING guards that used to live here additionally
cross-referenced the retired Python worker's ``MODEL_TARGETS`` table via a lazy
``importorskip``, so they degraded to a clean SKIP once ``apps/worker`` was deleted
(epic 8283) — losing their coverage. sc-9513 (F-059 follow-up) reimplemented them
against the Rust worker's own character-image engine wiring, reading this SAME
embedded manifest, in ``crates/sceneworks-worker/src/engines.rs`` (the tests
``character_image_capability_implies_engine_or_tuning_declaration`` /
``kolors_declares_strict_pose_controlnet`` /
``models_with_engine_block_advertise_character_image``). This module now imports no
``scene_worker`` symbol at all.
"""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "manifests" / "builtin.models.jsonc"
SCHEMA_PATH = ROOT / "packages" / "schemas" / "model-manifest.schema.json"
ENGINE_TABLE_PATH = ROOT / "crates" / "sceneworks-worker" / "src" / "engines.rs"
WORKER_SOURCE_PATH = ROOT / "crates" / "sceneworks-worker" / "src"
CONTROL_WEIGHTS_PATH = ROOT / "crates" / "sceneworks-core" / "src" / "control_weights.rs"

EXPECTED_SHIPPED_CONTROL_WEIGHTS = frozenset(
    {
        (
            "flux1_dev_control",
            "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
            "diffusion_pytorch_model.safetensors",
            "5d700aaad96c5ddcdf8a38ef9b22a82aac2c38e5",
        ),
        (
            "flux2_dev_control",
            "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union",
            "FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors",
            "b3dcd7836a0e926248dac3ccba8fc0853495764b",
        ),
        (
            "z_image_turbo_control",
            "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
            "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",
            "5155fc56d17821007d6f62ac192c09e0f0e72016",
        ),
        (
            "z_image_control",
            "alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1",
            "Z-Image-Fun-Controlnet-Union-2.1.safetensors",
            "755999a934909bd5832e20718bb7c639d2a63eb9",
        ),
        (
            "z_image_control",
            "alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1",
            "diffusion_pytorch_model.safetensors",
            "755999a934909bd5832e20718bb7c639d2a63eb9",
        ),
        (
            "qwen_image_control",
            "SceneWorks/qwen-image-2512-fun-controlnet-union",
            "q4/model.safetensors",
            "a061fbc42a4744d6a7ec206370fbd3a37d4a7cca",
        ),
        (
            "qwen_image_control",
            "SceneWorks/qwen-image-2512-fun-controlnet-union",
            "q8/model.safetensors",
            "a061fbc42a4744d6a7ec206370fbd3a37d4a7cca",
        ),
        (
            "qwen_image_control",
            "SceneWorks/qwen-image-2512-fun-controlnet-union",
            "bf16/model.safetensors",
            "a061fbc42a4744d6a7ec206370fbd3a37d4a7cca",
        ),
        (
            "kolors_control",
            "Kwai-Kolors/Kolors-ControlNet-Pose",
            "diffusion_pytorch_model.safetensors",
            "83e35a8033a89d2e75044b412d0e2474111578f7",
        ),
        (
            "krea_2_turbo_control",
            "SceneWorks/krea2-pose-controlnet-beta",
            "control_step5000.safetensors",
            "cb3a0ac7590f5ec594a4eeb43b95ee1da0b5a0ac",
        ),
    }
)


def _strip_jsonc_comments(body: str) -> str:
    """Mirror scripts/check-scaffold.mjs::stripJsoncComments so the audit reads
    the real `config/manifests/builtin.models.jsonc` without a JSONC dependency.
    Walks the body char-by-char, suppressing // line and /* block */ comments
    but leaving them intact when they appear inside string literals.
    """
    result: list[str] = []
    in_string = False
    escaped = False
    i = 0
    while i < len(body):
        char = body[i]
        nxt = body[i + 1] if i + 1 < len(body) else ""
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue
        if char == "/" and nxt == "/":
            while i < len(body) and body[i] != "\n":
                i += 1
            result.append("\n")
            continue
        if char == "/" and nxt == "*":
            i += 2
            while i < len(body) - 1 and not (body[i] == "*" and body[i + 1] == "/"):
                i += 1
            i += 2
            continue
        result.append(char)
        i += 1
    return "".join(result)


@lru_cache(maxsize=None)
def _cached_jsonc(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    return json.loads(_strip_jsonc_comments(raw))


@lru_cache(maxsize=None)
def _cached_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_builtin_models_manifest() -> dict:
    return copy.deepcopy(_cached_jsonc(MANIFEST_PATH))


def _load_schema(path: Path) -> dict:
    return copy.deepcopy(_cached_json(path))


def test_cached_manifest_and_schema_loaders_return_isolated_copies():
    first_manifest = _load_builtin_models_manifest()
    original_id = first_manifest["models"][0]["id"]
    first_manifest["models"][0]["id"] = "mutated"
    assert _load_builtin_models_manifest()["models"][0]["id"] == original_id

    first_schema = _load_schema(SCHEMA_PATH)
    original_type = first_schema["type"]
    first_schema["type"] = "mutated"
    assert _load_schema(SCHEMA_PATH)["type"] == original_type


def test_mage_flow_generation_family_is_pinned_and_complete():
    """sc-14047: all published Mage generation variants remain loadable flat snapshots."""
    models = {model["id"]: model for model in _load_builtin_models_manifest()["models"]}
    expected = {
        "mage_flow_base": ("SceneWorks/Mage-Flow-Base", "a160c0a2c9d82106687d969d161c313c7e528550", 30, 5),
        "mage_flow": ("SceneWorks/Mage-Flow", "e6719fb1abb0d3fa83ecebbf31cbf431eb054ab8", 20, 5),
        "mage_flow_turbo": ("SceneWorks/Mage-Flow-Turbo", "6fc43b7b586078997047acc9c39dbbd26044a035", 4, 1),
    }
    complete_snapshot = {
        "model_index.json",
        "scheduler/*",
        "text_encoder/*",
        "transformer/*",
        "vae/*",
    }
    for model_id, (repo, revision, steps, guidance) in expected.items():
        model = models[model_id]
        assert model["family"] == "mage-flow"
        assert model["macOnly"] is False
        assert model["candle"] == {
            "minMemoryGb": 17,
            "vramGbByTier": {"q4": 14.67, "q8": 16.95, "bf16": 20.41},
            "measured": True,
        }
        assert model["defaults"]["steps"] == steps
        assert model["defaults"]["guidanceScale"] == guidance
        downloads = model["downloads"]
        assert [entry["variant"] for entry in downloads] == ["q4", "q8", "bf16"]
        assert sum(entry.get("default") is True for entry in downloads) == 1
        for entry in downloads:
            assert entry["repo"] == repo
            assert entry["revision"] == revision
            # These are load-time quant choices over one dense mirror, so every logical tier
            # uses the same full-snapshot predicate. Physical tier provisioning belongs to sc-14059.
            assert set(entry["files"]) == complete_snapshot
        assert model["paths"]["model"] == f"${{HF_CACHE}}/{repo}"


def test_mage_flow_edit_family_is_pinned_complete_and_source_gated():
    """sc-14050: every edit variant is a complete shared-component snapshot."""
    models = {model["id"]: model for model in _load_builtin_models_manifest()["models"]}
    expected = {
        "mage_flow_edit_base": ("SceneWorks/Mage-Flow-Edit-Base", "5345fa8b0d41bdd612351f0933ef4cc9021281ab", 17495714677, 30, 5),
        "mage_flow_edit": ("SceneWorks/Mage-Flow-Edit", "d668ecc8ce5addcb3c0bbe268227eb17f832aa9f", 17495714677, 30, 5),
        "mage_flow_edit_turbo": ("SceneWorks/Mage-Flow-Edit-Turbo", "3e5ae67807083a25f3d344bc2c5ff16276415a14", 17495714653, 4, 1),
    }
    complete_snapshot = {
        "model_index.json",
        "scheduler/*",
        "text_encoder/*",
        "transformer/*",
        "vae/*",
    }
    for model_id, (repo, revision, size, steps, guidance) in expected.items():
        model = models[model_id]
        assert model["family"] == "mage-flow"
        assert model["adapter"] == "mlx_mage"
        assert model["capabilities"] == ["edit_image"]
        assert model["macOnly"] is False
        assert model["candle"] == {
            "minMemoryGb": 17,
            "vramGbByTier": {"q4": 14.67, "q8": 16.95, "bf16": 20.41},
            "measured": True,
        }
        assert model["defaults"]["steps"] == steps
        assert model["defaults"]["guidanceScale"] == guidance
        assert model["ui"]["sourceWithMultiReference"] is True
        assert model["ui"]["recommendedFor"] == ["edit_image"]
        assert [entry["variant"] for entry in model["downloads"]] == ["q4", "q8", "bf16"]
        assert sum(entry.get("default") is True for entry in model["downloads"]) == 1
        for entry in model["downloads"]:
            assert entry["repo"] == repo
            assert entry["revision"] == revision
            assert entry["estimatedSizeBytes"] == size
            assert set(entry["files"]) == complete_snapshot
        assert model["paths"]["model"] == f"${{HF_CACHE}}/{repo}"


def _model_table_default_repos() -> dict[str, str]:
    """Read the worker's pure-data ``MODEL_TABLE`` without importing Rust.

    Each row is deliberately flat, so anchoring the two string fields inside a
    ``ModelRow`` block is sufficient and keeps this catalog audit independent of
    target-specific worker compilation.
    """
    source = ENGINE_TABLE_PATH.read_text(encoding="utf-8")
    table = source.split("pub(crate) const MODEL_TABLE", maxsplit=1)[1].split("];", maxsplit=1)[0]
    rows: dict[str, str] = {}
    for block in re.findall(r"ModelRow\s*\{(.*?)\n\s*\},", table, flags=re.DOTALL):
        model_id = re.search(r'sceneworks_id:\s*"([^"]+)"', block)
        default_repo = re.search(r'default_repo:\s*"([^"]+)"', block)
        assert model_id and default_repo, f"unparseable MODEL_TABLE row:\n{block}"
        assert model_id.group(1) not in rows, f"duplicate MODEL_TABLE id: {model_id.group(1)}"
        rows[model_id.group(1)] = default_repo.group(1)
    assert rows, "MODEL_TABLE parser found no rows"
    return rows


def test_every_model_table_default_repo_is_installed_by_a_builtin_model():
    """sc-14476: a worker fallback may only name a repo the model installer stages.

    Built-ins intentionally have no top-level ``repo``; the worker lanes therefore
    resolve through ``MODEL_TABLE.default_repo``. Keep the assertion direction
    table -> downloads: co-requisites and shared components legitimately add extra
    download repos that need no table row.
    """
    manifest = _load_builtin_models_manifest()
    downloads_by_id = {
        model["id"]: {
            download["repo"]
            for download in model.get("downloads", [])
            if download.get("provider", "huggingface") == "huggingface" and download.get("repo")
        }
        for model in manifest["models"]
    }
    installed_repos = set().union(*downloads_by_id.values())
    mismatches = {
        model_id: {
            "default_repo": default_repo,
            "installed_repos": sorted(downloads_by_id.get(model_id, set())),
        }
        for model_id, default_repo in _model_table_default_repos().items()
        # Legacy request aliases may retain MODEL_TABLE rows after their standalone
        # manifest entries are collapsed (qwen_image_edit / _2509 -> _2511).
        # They are safe only while another built-in stages the identical repo.
        if default_repo not in installed_repos
    }
    assert not mismatches, (
        "MODEL_TABLE defaults must be staged by a builtin model download; "
        f"unstaged defaults: {mismatches}"
    )


def test_builtin_models_do_not_declare_a_top_level_repo():
    """Document the contract that makes each lane's catalog default authoritative."""
    offenders = [
        model["id"]
        for model in _load_builtin_models_manifest()["models"]
        if model.get("repo")
    ]
    assert not offenders, (
        "built-in repo ownership lives under downloads[]; top-level repo readers must retain an "
        f"installed fallback. Unexpected declarations: {offenders}"
    )


def _assert_exact_shipped_control_weight_registry(source: str) -> None:
    """Audit the central strict-control authority without compiling target-specific lanes."""
    table = source.split("pub const SHIPPED_CONTROL_WEIGHTS", maxsplit=1)[1].split(
        "];", maxsplit=1
    )[0]
    rows: set[tuple[str, str, str, str]] = set()
    blocks = re.findall(r"ShippedControlWeight\s*\{(.*?)\n\s*\},", table, flags=re.DOTALL)
    for block in blocks:
        fields = {}
        for field in ("engine_id", "repo", "file", "revision"):
            match = re.search(rf'{field}:\s*"([^"]+)"', block)
            assert match, f"unparseable SHIPPED_CONTROL_WEIGHTS {field}:\n{block}"
            fields[field] = match.group(1)
        row = (fields["engine_id"], fields["repo"], fields["file"], fields["revision"])
        assert row not in rows, f"duplicate shipped control-weight tuple: {row}"
        rows.add(row)

    assert rows == EXPECTED_SHIPPED_CONTROL_WEIGHTS, (
        "central strict-control authority changed; audit exact engine/repo/file/revision tuples: "
        f"added={sorted(rows - EXPECTED_SHIPPED_CONTROL_WEIGHTS)}, "
        f"removed={sorted(EXPECTED_SHIPPED_CONTROL_WEIGHTS - rows)}"
    )
    for engine_id, repo, filename, revision in rows:
        assert re.fullmatch(r"[0-9a-f]{40}", revision), (
            f"{engine_id} {repo}/{filename}: revision must be an immutable lowercase 40-hex commit"
        )


def test_shipped_control_weight_registry_is_exact_complete_and_pinned():
    """sc-13639: central authority replaces seven retired manifest repo/modelPath readers."""
    _assert_exact_shipped_control_weight_registry(
        CONTROL_WEIGHTS_PATH.read_text(encoding="utf-8")
    )


def _assert_strict_control_consumers_use_central_pinned_authority(
    sources: dict[str, str],
) -> None:
    expected_consumers = {
        "image_jobs/flux1_control.rs",
        "image_jobs/flux1_control_candle.rs",
        "image_jobs/flux2.rs",
        "image_jobs/flux2_control_candle.rs",
        "image_jobs/kolors_control.rs",
        "image_jobs/krea_control.rs",
        "image_jobs/krea_control_candle.rs",
        "image_jobs/qwen.rs",
        "image_jobs/qwen_control.rs",
        "image_jobs/zimage.rs",
        "image_jobs/zimage_control.rs",
    }
    actual_consumers = {
        path
        for path, source in sources.items()
        if path != "image_jobs/strict_control.rs"
        and "trusted_control_weight_revision(" in source
    }
    assert actual_consumers == expected_consumers, (
        "strict-control central-authority consumer inventory changed; "
        f"added={sorted(actual_consumers - expected_consumers)}, "
        f"removed={sorted(expected_consumers - actual_consumers)}"
    )
    for path in expected_consumers:
        assert "huggingface_pinned_snapshot_dir" in sources[path], (
            f"{path}: central tuple must resolve only through snapshots/<revision>, "
            "never a mutable repo cache root"
        )

    strict_control = sources["image_jobs/strict_control.rs"]
    assert (
        "sceneworks_core::control_weights::shipped_control_weight(engine_id, repo, file)"
        in strict_control
    ), "strict_control.rs must authorize the exact engine/repo/file tuple centrally"


def test_strict_control_consumers_use_central_pinned_authority():
    sources = {
        path.relative_to(WORKER_SOURCE_PATH).as_posix(): path.read_text(encoding="utf-8")
        for path in WORKER_SOURCE_PATH.rglob("*.rs")
        if "tests" not in path.relative_to(WORKER_SOURCE_PATH).parts
        and path.name != "tests.rs"
    }
    _assert_strict_control_consumers_use_central_pinned_authority(sources)


def _must_fail_assertion(callback, message: str) -> None:
    try:
        callback()
    except AssertionError:
        return
    raise AssertionError(message)


def test_control_weight_authority_audit_detects_absence_and_fallback_mutations():
    """Mutation guard: missing tuples/pins and bypassed central consumers must fail this audit."""
    registry = CONTROL_WEIGHTS_PATH.read_text(encoding="utf-8")
    table_start = registry.index("pub const SHIPPED_CONTROL_WEIGHTS")
    missing_row = registry[:table_start] + re.sub(
        r"\s*ShippedControlWeight\s*\{.*?\n\s*\},",
        "",
        registry[table_start:],
        count=1,
        flags=re.DOTALL,
    )
    _must_fail_assertion(
        lambda: _assert_exact_shipped_control_weight_registry(missing_row),
        "removing one central tuple must fail the exact registry audit",
    )
    mutable_pin = registry.replace(
        "5d700aaad96c5ddcdf8a38ef9b22a82aac2c38e5", "main", 1
    )
    _must_fail_assertion(
        lambda: _assert_exact_shipped_control_weight_registry(mutable_pin),
        "replacing one immutable pin with a mutable revision must fail the registry audit",
    )

    sources = {
        path.relative_to(WORKER_SOURCE_PATH).as_posix(): path.read_text(encoding="utf-8")
        for path in WORKER_SOURCE_PATH.rglob("*.rs")
        if "tests" not in path.relative_to(WORKER_SOURCE_PATH).parts
        and path.name != "tests.rs"
    }
    bypassed = dict(sources)
    bypassed["image_jobs/flux1_control_candle.rs"] = bypassed[
        "image_jobs/flux1_control_candle.rs"
    ].replace("trusted_control_weight_revision(", "bypassed_revision(")
    _must_fail_assertion(
        lambda: _assert_strict_control_consumers_use_central_pinned_authority(bypassed),
        "removing one central-authority consumer must fail the inventory audit",
    )
    unpinned = dict(sources)
    unpinned["image_jobs/krea_control.rs"] = unpinned["image_jobs/krea_control.rs"].replace(
        "huggingface_pinned_snapshot_dir", "huggingface_repo_cache_path"
    )
    _must_fail_assertion(
        lambda: _assert_strict_control_consumers_use_central_pinned_authority(unpinned),
        "replacing an immutable snapshot resolver with a mutable fallback must fail",
    )


def test_every_top_level_manifest_repo_reader_has_an_audited_installed_fallback():
    """sc-14476 lane inventory and regression guard.

    The marker names the effective resolution source in each lane. Explicit
    constants below are justified because their repo is staged as a co-requisite
    of a different model (InstantID/PuLID), while video resolves Wan tiers from
    the request's own downloads. Any new reader must be consciously added here.
    """
    audited_lanes = {
        "image_jobs/base.rs": "model.default_repo()",
        "image_jobs/flux_ipadapter.rs": "flux_ipadapter_default_repo(&request.model)",
        "image_jobs/instantid.rs": "INSTANTID_SDXL_REPO",
        "image_jobs/kolors_ipadapter.rs": "default_repo_for(&request.model)",
        "image_jobs/krea_edit_candle.rs": "default_repo_for(&request.model)",
        "image_jobs/pulid.rs": "PULID_FLUX_REPO",
        "image_jobs/pulid_candle.rs": "PULID_CANDLE_FLUX_REPO",
        "image_jobs/qwen_edit_candle.rs": "crate::engines::MODEL_TABLE",
        "image_jobs/sdxl_edit_candle.rs": "sdxl_edit_candle_default_repo(&request.model)",
        "image_jobs/sdxl_ipadapter.rs": "sdxl_ipadapter_default_repo(&request.model)",
        "image_jobs/zimage_edit_candle.rs": "default_repo_for(&request.model)",
        "image_jobs/zimage_identity_candle.rs": "default_repo_for(&request.model)",
        "sensenova_jobs.rs": "default_repo_for(&request.model)",
        "video_jobs/candle.rs": "candle_wan_tier_repo_from_downloads(request, engine_id)",
    }
    actual_lanes: set[str] = set()
    for path in WORKER_SOURCE_PATH.rglob("*.rs"):
        source = path.read_text(encoding="utf-8").split("\n#[cfg(test)]", maxsplit=1)[0]
        if re.search(r"\.model_manifest_entry\s*\.get\(\"repo\"\)", source):
            actual_lanes.add(path.relative_to(WORKER_SOURCE_PATH).as_posix())

    assert actual_lanes == set(audited_lanes), (
        "top-level model_manifest_entry.repo lane inventory changed; audit every added/removed lane: "
        f"added={sorted(actual_lanes - set(audited_lanes))}, "
        f"removed={sorted(set(audited_lanes) - actual_lanes)}"
    )
    for relative_path, fallback_marker in audited_lanes.items():
        source = (
            (WORKER_SOURCE_PATH / relative_path)
            .read_text(encoding="utf-8")
            .split("\n#[cfg(test)]", maxsplit=1)[0]
        )
        reads = list(re.finditer(r"\.model_manifest_entry\s*\.get\(\"repo\"\)", source))
        assert reads, f"{relative_path}: inventoried lane no longer contains a top-level repo read"
        for read in reads:
            resolution = source[read.end() : read.end() + 900]
            assert fallback_marker in resolution, (
                f"{relative_path}: top-level repo read no longer resolves through audited source "
                f"{fallback_marker!r}"
            )
    for relative_path in (
        "image_jobs/flux_ipadapter.rs",
        "image_jobs/sdxl_edit_candle.rs",
        "image_jobs/sdxl_ipadapter.rs",
        "image_jobs/zimage_control.rs",
    ):
        source = (WORKER_SOURCE_PATH / relative_path).read_text(encoding="utf-8")
        assert "default_repo_for(model)" in source, (
            f"{relative_path}: per-family fallback no longer delegates to MODEL_TABLE"
        )


def test_manifest_model_path_is_only_an_optional_override():
    """sc-14476: converted installs may inject ``modelPath``; normal installs do not.

    Every production reader must therefore inspect it inside an optional branch
    and continue to repo resolution (or decline an imported-only lane) when it
    is absent.
    """
    expected_readers = {
        "image_jobs/base.rs",
        "image_jobs/flux_ipadapter.rs",
        "image_jobs/instantid.rs",
        "image_jobs/kolors_ipadapter.rs",
        "image_jobs/krea_imported.rs",
        "image_jobs/pulid.rs",
        "image_jobs/pulid_candle.rs",
        "image_jobs/qwen_edit_candle.rs",
        "image_jobs/sdxl_edit_candle.rs",
        "image_jobs/sdxl_imported.rs",
        "image_jobs/sdxl_ipadapter.rs",
        "image_jobs/zimage_edit_candle.rs",
    }
    actual_readers: list[str] = []
    for path in WORKER_SOURCE_PATH.rglob("*.rs"):
        source = path.read_text(encoding="utf-8").split("\n#[cfg(test)]", maxsplit=1)[0]
        for match in re.finditer(
            r'request\.model_manifest_entry\.get\("modelPath"\)',
            source,
        ):
            relative_path = path.relative_to(WORKER_SOURCE_PATH).as_posix()
            actual_readers.append(relative_path)
            prefix = source[max(0, match.start() - 500) : match.start()]
            assert "if let Some(path) = request" in prefix or "let Some(raw_path) = request" in prefix, (
                f"{relative_path}: modelPath is no longer read through an "
                "optional branch"
            )
    assert len(actual_readers) == len(set(actual_readers)), (
        f"multiple production modelPath reads require individual fallback audit: {actual_readers}"
    )
    assert set(actual_readers) == expected_readers, (
        "modelPath reader inventory changed; audit every added/removed reader for an absence "
        f"fallback: added={sorted(set(actual_readers) - expected_readers)}, "
        f"removed={sorted(expected_readers - set(actual_readers))}"
    )


def test_builtin_models_manifest_satisfies_authoring_schema():
    """sc-12338: the builtin catalog's $schema is an enforced CI contract."""
    manifest = _load_builtin_models_manifest()
    schema = _load_schema(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: list(error.absolute_path),
    )
    assert not errors, "builtin.models.jsonc violates model-manifest.schema.json:\n" + "\n".join(
        f"- {'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
        for error in errors
    )


def test_builtin_schema_rejects_an_unknown_closed_model_key():
    """Mutation guard: a typo/decorative builtin key must make the CI gate fail."""
    manifest = _load_builtin_models_manifest()
    manifest["models"][0]["recommendded"] = True
    schema = _load_schema(SCHEMA_PATH)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    assert any("recommendded" in error.message for error in errors)


def _sample_audio_model_entry() -> dict:
    """A representative `type: "audio"` entry exercising every field of the new
    `audio` capability sub-block (sc-13401, epic 13400). Not a shipped model —
    real audio entries land in sc-13402 — so it lives in the test, not the
    builtin manifest.
    """
    return {
        "id": "sample_audio_speech",
        "name": "Sample Audio Speech",
        "family": "sample_audio",
        "type": "audio",
        # `audio` is a picker (non-utility) type, so the schema now requires a
        # `ui.promptGuide` with a title/path (sc-13783, reconciled with
        # scripts/check-scaffold.mjs). Kept minimal; this fixture never ships.
        "ui": {
            "promptGuide": {
                "title": "Sample Audio Prompt Guide",
                "path": "/prompt-guides/sample-audio.md",
            }
        },
        "audio": {
            "voices": [
                {
                    "id": "af_heart",
                    "label": "Heart",
                    "gender": "female",
                    "accent": "american",
                    "language": "en-US",
                },
                {"id": "bm_george", "gender": "male", "accent": "british"},
            ],
            "languages": ["en-US", "en-GB"],
            "sampleRates": [24000, 48000],
            "maxDurationSecs": 30.0,
            "editModes": ["extend", "inpaint", "cover"],
            "supportsMultiSpeaker": True,
            "maxSpeakers": 2,
            "conditioning": ["AudioEdit", "ReferenceAudio", "VoiceEmbedding"],
        },
    }


def test_schema_accepts_audio_type_and_audio_sub_block():
    """sc-13401: a `type: "audio"` entry with a populated `audio` sub-block
    validates against the authoring schema (the new sibling of mlx/candle)."""
    schema = _load_schema(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    manifest = {"schemaVersion": 1, "models": [_sample_audio_model_entry()]}
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    assert not errors, "sample audio entry must satisfy the schema:\n" + "\n".join(
        f"- {'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
        for error in errors
    )


def test_model_schema_requires_entry_identity():
    """Every model entry carries the id, display name, family, and type used by
    routing and catalog consumers."""
    schema = _load_schema(SCHEMA_PATH)
    for field in ("id", "name", "family", "type"):
        entry = _sample_audio_model_entry()
        del entry[field]
        errors = list(
            jsonschema.Draft202012Validator(schema).iter_errors(
                {"schemaVersion": 1, "models": [entry]}
            )
        )
        assert any(
            error.validator == "required"
            and field in error.validator_value
            and list(error.absolute_path) == ["models", 0]
            for error in errors
        ), f"a model entry without `{field}` must be rejected by the entry identity contract"


def test_schema_rejects_unknown_field_under_audio_sub_block():
    """Mutation guard: the `audio` block is additionalProperties:false, so a typo
    / undeclared field under it must fail validation."""
    schema = _load_schema(SCHEMA_PATH)
    entry = _sample_audio_model_entry()
    entry["audio"]["bogusField"] = True
    manifest = {"schemaVersion": 1, "models": [entry]}
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    assert any("bogusField" in error.message for error in errors), (
        "an unknown key under `audio` must be rejected by additionalProperties:false"
    )


def test_schema_rejects_audio_voice_without_id():
    """A voice object requires `id` so the picker always has a backend key."""
    schema = _load_schema(SCHEMA_PATH)
    entry = _sample_audio_model_entry()
    entry["audio"]["voices"] = [{"label": "No Id", "gender": "female"}]
    manifest = {"schemaVersion": 1, "models": [entry]}
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    # Discriminate on the jsonschema error's shape, not a substring of its
    # message: a loose `"id" in error.message` incidentally matches unrelated
    # errors (e.g. a type-enum error listing "video", which contains "id"), so
    # it could false-green under a full schema revert. Pin the `required`
    # keyword, its `["id"]` value, and the path at the voice item instead — this
    # only holds while the voice object's `required: ["id"]` is present.
    assert any(
        error.validator == "required"
        and error.validator_value == ["id"]
        and list(error.absolute_path) == ["models", 0, "audio", "voices", 0]
        for error in errors
    ), (
        "a voice entry without `id` must be rejected by the voice object's "
        "required:['id'] (a `required` error at models/0/audio/voices/0)"
    )


def test_schema_rejects_unknown_model_type():
    """Negative control: the `type` enum still rejects an out-of-set value even
    after `audio` was added, so the enum is not accidentally open."""
    schema = _load_schema(SCHEMA_PATH)
    entry = _sample_audio_model_entry()
    entry["type"] = "hologram"
    manifest = {"schemaVersion": 1, "models": [entry]}
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    assert any(
        error.validator == "enum"
        and list(error.absolute_path) == ["models", 0, "type"]
        for error in errors
    ), "the model type must fail the enum validator exactly at models/0/type"


# The seeded audio catalog (sc-13402, epic 13400). Each id is a live candle-audio
# provider registered in `crates/audio/candle-audio-catalog`; the second element is
# the `audio` capability sub-block key that MUST be populated for the Audio Studio to
# build its pickers/mode gates without probing the backend.
_SEEDED_AUDIO_MODELS = {
    "kokoro_82m": "voices",
    "moss_sfx_v2": "sampleRates",
    "acestep_v15_turbo": "editModes",
    "openvoice_v2": "conditioning",
    "chatterbox_ve": "conditioning",
}


def test_builtin_manifest_ships_the_seeded_audio_models():
    """sc-13402: the five live audio providers (Kokoro, MOSS-SFX, ACE-Step,
    OpenVoice V2, Chatterbox-VE) are seeded as `type: "audio"` entries, each
    carrying a populated `audio` capability sub-block (not just the schema-legal
    shape proven by sc-13401, but the ACTUAL shipped entries). Kokoro is the
    recommended Speech model."""
    manifest = _load_builtin_models_manifest()
    by_id = {m.get("id"): m for m in manifest["models"]}

    for model_id, required_cap_key in _SEEDED_AUDIO_MODELS.items():
        entry = by_id.get(model_id)
        assert entry is not None, f"seeded audio model {model_id} is missing from the manifest"
        assert entry.get("type") == "audio", f"{model_id} must be type:audio"
        audio = entry.get("audio")
        assert isinstance(audio, dict) and audio, f"{model_id} must carry a populated `audio` block"
        assert required_cap_key in audio, (
            f"{model_id}.audio must advertise `{required_cap_key}` (populated from backend "
            f"Capabilities, not an empty stub)"
        )
        # Every audio entry must be installable/downloadable like image/video models.
        downloads = entry.get("downloads") or []
        assert downloads and downloads[0].get("repo"), (
            f"{model_id} must define a download entry with a repo (install/download parity)"
        )
        assert entry.get("paths", {}).get("model"), f"{model_id} must define paths.model"

    # Kokoro's real voice surface: the 28 English packs the pinned snapshot ships,
    # each an object with an `id` (discriminates against an empty/placeholder list).
    kokoro_voices = by_id["kokoro_82m"]["audio"]["voices"]
    assert len(kokoro_voices) == 28, "Kokoro advertises its 28 shipped English voices"
    assert all(isinstance(v, dict) and v.get("id") for v in kokoro_voices)
    assert by_id["kokoro_82m"].get("recommended") is True, "Kokoro is the recommended Speech model"

    # ACE-Step's real edit surface (Conditioning::AudioEdit → repaint task modes). `cover` (sc-13821)
    # is the whole-clip restyle backed by the `sft_cover` coRequisite — advertised here so the Music
    # studio surfaces it, matching the backend descriptor's audio_edit_modes (Inpaint/Repaint/Extend/Cover).
    assert set(by_id["acestep_v15_turbo"]["audio"]["editModes"]) == {
        "inpaint",
        "repaint",
        "extend",
        "cover",
    }
    assert "AudioEdit" in by_id["acestep_v15_turbo"]["audio"]["conditioning"]


def _duplicate_default_downloads(manifest: dict) -> list[str]:
    """Return model/platform pairs with ambiguous primary download selection."""
    ambiguous: list[str] = []
    for model in manifest["models"]:
        downloads = model.get("downloads", [])
        for platform in ("macos", "windows", "linux"):
            defaults = [
                download
                for download in downloads
                if download.get("default") is True
                and download.get("coRequisite") is not True
                and (
                    "platforms" not in download
                    or platform in download.get("platforms", [])
                )
            ]
            if len(defaults) > 1:
                ambiguous.append(f"{model['id']}:{platform}")
    return ambiguous


def test_builtin_download_defaults_are_unique_per_platform():
    """A model may have one primary default per OS, never two applicable defaults."""
    assert not _duplicate_default_downloads(_load_builtin_models_manifest())


def test_download_default_guard_rejects_an_ambiguous_platform_mutation():
    """Mutation guard for the platform-aware replacement of schema maxContains."""
    manifest = _load_builtin_models_manifest()
    model = next(model for model in manifest["models"] if model["id"] == "wan_2_2")
    windows_download = next(
        download
        for download in model["downloads"]
        if download.get("variant") == "q8" and "windows" in download.get("platforms", [])
    )
    windows_download["default"] = True
    assert _duplicate_default_downloads(manifest) == ["wan_2_2:windows", "wan_2_2:linux"]


# ---------------------------------------------------------------------------
# F-029 download-revision pin authority (sc-13659).
#
# A download entry's optional `revision` is the immutable-commit pin the worker
# fetches into `snapshots/<sha>/`; absent means the worker resolves `main`. The
# JSON Schema constrains its FORMAT (`^[0-9a-f]{40}$`), but the
# "coRequisite: true REQUIRES a revision" invariant lives here (and in the Rust
# builtin_manifests.rs backstop) because JSON Schema cannot grandfather the
# sc-13591 pin migration still in flight.
# ---------------------------------------------------------------------------

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# `(model_id, repo)` co-requisite download pairs whose F-029 pin migration is
# still IN FLIGHT under sc-13591. Each is a KNOWN, tracked gap: the immutable
# commit SHA lives in the sc-13591 inventory but is applied by a later per-family
# story, not sc-13659 (schema + plumbing + enforcement only — it must not add
# real pins). A brand-new co-requisite may NOT join this list; pin its `revision`
# instead. Kept in lockstep with the identical Rust allowlist in
# crates/sceneworks-core/src/builtin_manifests.rs.
_COREQUISITE_REVISION_MIGRATION_PENDING: frozenset[tuple[str, str]] = frozenset(
    {
        # ("ltx_2_3", "SceneWorks/ltx-2.3-mlx") pinned in sc-13683 (the gemma coRequisite now carries
        # the full 40-hex LTX_BUNDLE_REVISION); removed here + in the Rust twin to keep both green.
        ("ltx_2_3_eros", "TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments"),
        ("wan_2_2_t2v_14b", "lightx2v/Wan2.2-Lightning"),
        ("wan_2_2_i2v_14b", "lightx2v/Wan2.2-Lightning"),
    }
)


def _corequisite_revision_gaps(manifest: dict) -> set[tuple[str, str]]:
    """`(model_id, repo)` co-requisite pairs NOT pinned to a full 40-hex SHA."""
    gaps: set[tuple[str, str]] = set()
    for model in manifest["models"]:
        for download in model.get("downloads", []):
            if download.get("coRequisite") is not True:
                continue
            revision = download.get("revision")
            if not (isinstance(revision, str) and _FULL_SHA_RE.match(revision)):
                gaps.add((model["id"], download.get("repo", "")))
    return gaps


def test_corequisite_downloads_pin_a_full_sha_revision():
    """F-029 (sc-13659): every coRequisite download pins an immutable 40-hex commit.

    A co-requisite is a FETCH-ALL companion the runtime resolves offline via a
    pinned-SHA `hf_get_pinned` reading `snapshots/<sha>/`; leaving it on `main`
    lands the wrong snapshot and hard-fails offline. The only tolerated gaps are
    the sc-13591 pins still being migrated by later stories.
    """
    manifest = _load_builtin_models_manifest()
    unexpected = _corequisite_revision_gaps(manifest) - _COREQUISITE_REVISION_MIGRATION_PENDING
    assert not unexpected, (
        "co-requisite downloads must pin a 40-hex commit SHA (F-029, sc-13659); these are "
        f"unpinned and NOT tracked for the sc-13591 migration: {sorted(unexpected)}"
    )


def test_first_party_downloads_pin_a_full_sha_revision():
    """SceneWorks controls these repositories, so the shipped catalog must never
    silently move an installed artifact when a repository's main branch changes."""
    gaps = []
    for model in _load_builtin_models_manifest()["models"]:
        for download in model.get("downloads", []):
            repo = download.get("repo", "")
            if repo.startswith("SceneWorks/") and not _FULL_SHA_RE.fullmatch(
                download.get("revision", "")
            ):
                gaps.append((model["id"], repo, download.get("variant")))
    assert not gaps, f"first-party downloads must pin immutable revisions: {gaps}"


def test_corequisite_revision_migration_allowlist_has_no_stale_entries():
    """Self-cleaning guard: an allowlist row that no longer names an unpinned
    co-requisite must be deleted, so pinning one in a later sc-13591 story forces
    its removal instead of the allowlist silently excusing an already-compliant
    entry (a test asserting a default is a false green — the allowlist must shrink).
    """
    manifest = _load_builtin_models_manifest()
    stale = _COREQUISITE_REVISION_MIGRATION_PENDING - _corequisite_revision_gaps(manifest)
    assert not stale, (
        "stale F-029 migration allowlist entries (now pinned or removed) must be deleted from "
        f"_COREQUISITE_REVISION_MIGRATION_PENDING: {sorted(stale)}"
    )


def test_corequisite_revision_guard_flags_a_new_unpinned_corequisite():
    """Mutation guard: the rule is LIVE for new entries, not decoration. A brand-new
    co-requisite with no revision (and not on the migration allowlist) is caught.
    """
    manifest = _load_builtin_models_manifest()
    kokoro = next(model for model in manifest["models"] if model["id"] == "kokoro_82m")
    kokoro.setdefault("downloads", []).append(
        {"provider": "huggingface", "repo": "example/new-corequisite", "coRequisite": True}
    )
    new_pair = ("kokoro_82m", "example/new-corequisite")
    assert new_pair in _corequisite_revision_gaps(manifest)
    assert new_pair not in _COREQUISITE_REVISION_MIGRATION_PENDING
    unexpected = _corequisite_revision_gaps(manifest) - _COREQUISITE_REVISION_MIGRATION_PENDING
    assert new_pair in unexpected


def _model_entry_with_download(download: dict) -> dict:
    """A minimal schema-valid model entry carrying a single `downloads` entry, for
    exercising the download-item schema in isolation."""
    return {
        "id": "sample_pinned_model",
        "name": "Sample Pinned Model",
        "family": "sample",
        "type": "image",
        # `image` is a picker (non-utility) type, so the schema now requires a
        # `ui.promptGuide` with a title/path (sc-13783, reconciled with
        # scripts/check-scaffold.mjs). Kept minimal; this fixture never ships.
        "ui": {
            "promptGuide": {
                "title": "Sample Prompt Guide",
                "path": "/prompt-guides/sample.md",
            }
        },
        "downloads": [download],
    }


def test_schema_pins_download_revision_to_a_40hex_sha():
    """sc-13659: the authoring schema constrains `revision` to a full 40-hex commit
    (the F-029 pin authority), accepting a valid SHA and rejecting a branch/tag/
    short/uppercase/wrong-length value via the `pattern` keyword.
    """
    schema = _load_schema(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    def revision_errors(revision: str) -> list:
        manifest = {
            "schemaVersion": 1,
            "models": [
                _model_entry_with_download(
                    {
                        "provider": "huggingface",
                        "repo": "namespace/model",
                        "files": [],
                        "revision": revision,
                    }
                )
            ],
        }
        return list(validator.iter_errors(manifest))

    assert not revision_errors("a" * 40), "a full 40-hex SHA must satisfy the schema"
    for bad in ("main", "v1.0", "abc123", "A" * 40, "g" * 40, "a" * 39, "a" * 41):
        errors = revision_errors(bad)
        # Discriminate on the failing keyword so a full schema revert (dropping the
        # pattern) turns this red rather than passing on some unrelated error.
        assert any(error.validator == "pattern" for error in errors), (
            f"revision {bad!r} must be rejected by the 40-hex pattern"
        )


def test_schema_accepts_a_component_id_on_a_corequisite_download():
    """sc-13679: a coRequisite download may carry a `componentId` — the explicit repo→component
    mapping the worker's `resolve_co_requisites` seam reads to stage `LoadSpec::components`. The
    authoring schema constrains it to lowercase snake_case (same shape as a descriptor id), accepting
    a valid id and rejecting capitals / hyphens / a leading digit / empty via the `pattern` keyword.
    """
    schema = _load_schema(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    def component_errors(component_id: str) -> list:
        manifest = {
            "schemaVersion": 1,
            "models": [
                _model_entry_with_download(
                    {
                        "provider": "huggingface",
                        "repo": "ResembleAI/chatterbox",
                        "files": ["ve.safetensors"],
                        "revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
                        "coRequisite": True,
                        "componentId": component_id,
                    }
                )
            ],
        }
        return list(validator.iter_errors(manifest))

    assert not component_errors("voice_embedding"), "a lowercase snake_case componentId must validate"
    for bad in ("Perth", "voice-embedding", "1codec", ""):
        errors = component_errors(bad)
        # Discriminate on the failing keyword so dropping the pattern turns this red rather than
        # passing on some unrelated error.
        assert any(error.validator == "pattern" for error in errors), (
            f"componentId {bad!r} must be rejected by the snake_case pattern"
        )


def test_schema_rejects_a_component_id_on_a_non_corequisite_download():
    """sc-13771: a `componentId` names a component PROVISIONED by a coRequisite, so it is valid ONLY on
    `coRequisite: true` entries. This is a MANIFEST-LOCAL invariant (checkable without any inference-side
    descriptor knowledge), so the authoring schema enforces it via an allOf conditional — catching a
    mis-tagged manifest at authoring time instead of only at runtime in `resolve_co_requisites`.

    The COMPLEMENTARY half — that a `componentId` matches some model's `ModelDescriptor::required_components`
    (and vice-versa) — is deliberately NOT enforced here: the manifest audit has no visibility into the
    inference-side descriptor `required_components` sets, so that cross-check stays a runtime error (covered
    by the `a_required_component_with_no_matching_component_id_is_a_manifest_error` unit test in
    crates/sceneworks-worker/src/model_jobs.rs).
    """
    schema = _load_schema(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    base = {
        "provider": "huggingface",
        "repo": "ResembleAI/chatterbox",
        "files": ["ve.safetensors"],
        "revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
    }

    def errors_for(download: dict) -> list:
        manifest = {
            "schemaVersion": 1,
            "models": [_model_entry_with_download(download)],
        }
        return list(validator.iter_errors(manifest))

    # Valid: componentId ON a coRequisite: true entry.
    assert not errors_for({**base, "coRequisite": True, "componentId": "voice_embedding"}), (
        "a componentId on a coRequisite: true download must validate"
    )

    # Invalid: componentId with NO coRequisite flag — the allOf conditional requires coRequisite.
    missing = errors_for({**base, "componentId": "voice_embedding"})
    assert any(
        error.validator == "required" and "coRequisite" in error.message for error in missing
    ), (
        "a componentId with no coRequisite flag must be rejected (coRequisite required); "
        f"got {[(e.validator, e.message) for e in missing]}"
    )

    # Invalid: componentId with coRequisite: false — the allOf conditional pins coRequisite to true.
    false_flag = errors_for({**base, "coRequisite": False, "componentId": "voice_embedding"})
    assert any(
        error.validator == "const" and error.absolute_path and error.absolute_path[-1] == "coRequisite"
        for error in false_flag
    ), (
        "a componentId on a coRequisite: false download must be rejected (coRequisite must be true); "
        f"got {[(e.validator, list(e.absolute_path)) for e in false_flag]}"
    )


def test_builtin_manifest_component_ids_are_all_on_corequisite_downloads():
    """sc-13771 mutation guard: the schema invariant above is LIVE against the real catalog. Every
    `componentId` in builtin.models.jsonc must sit on a `coRequisite: true` download — proving the guard
    is not merely decoration on a synthetic fixture, and that no real entry mis-tags a componentId.
    """
    manifest = _load_builtin_models_manifest()
    misplaced = [
        (model["id"], download.get("repo"), download["componentId"])
        for model in manifest["models"]
        for download in model.get("downloads", [])
        if "componentId" in download and download.get("coRequisite") is not True
    ]
    assert not misplaced, (
        "a componentId may appear only on a coRequisite: true download (sc-13771); these are "
        f"mis-tagged: {sorted(misplaced)}"
    )


def test_manifest_constraint_contract_registry_is_complete_and_live():
    """sc-12304: constraint declarations may not silently become decoration.

    The schema is the author-facing registry; this test makes its custom contract
    annotations a CI gate. It checks both directions (manifest -> registry and
    registry -> manifest), and binding entries must name production readers that
    contain the exact key. Advisory/descriptive entries are explicitly allowed not
    to reject requests, which is materially different from an accidental dead key.
    """
    manifest = _load_builtin_models_manifest()
    schema = _load_schema(SCHEMA_PATH)
    model_properties = schema["properties"]["models"]["items"]["properties"]

    declared: set[str] = set()
    for model in manifest["models"]:
        declared.update(f"limits.{key}" for key in model.get("limits", {}))
        for backend in ("mlx", "candle"):
            block = model.get(backend, {})
            if "minMemoryGb" in block:
                declared.add(f"{backend}.minMemoryGb")
            declared.update(f"{backend}.limits.{key}" for key in block.get("limits", {}))

    limits_properties = model_properties["limits"]["properties"]
    registry = {f"limits.{key}": value for key, value in limits_properties.items()}
    for backend in ("mlx", "candle"):
        backend_properties = model_properties[backend]["properties"]
        if "minMemoryGb" in backend_properties:
            registry[f"{backend}.minMemoryGb"] = backend_properties["minMemoryGb"]
        backend_limits = backend_properties.get("limits", {})
        if "$ref" in backend_limits:
            sampler_properties = schema["$defs"]["samplerLimits"]["properties"]
            for key, value in sampler_properties.items():
                registry[f"{backend}.limits.{key}"] = value

    allowed_undeclared = {
        path
        for path, contract in registry.items()
        if contract.get("x-sceneworks-allow-undeclared")
        or (
            path.startswith("candle.limits.")
            and model_properties["candle"]["properties"]["limits"].get(
                "x-sceneworks-allow-undeclared"
            )
        )
    }
    assert declared <= set(registry) and set(registry) - declared <= allowed_undeclared, (
        "constraint contract drift: every declared constraint must be registered, "
        "and every registry entry must be exercised by the builtin manifest; "
        f"unregistered={sorted(declared - set(registry))}, "
        f"undeclared={sorted(set(registry) - declared)}"
    )

    allowed_classes = {"binding", "advisory", "descriptive"}
    for path, contract in registry.items():
        classification = contract.get("x-sceneworks-contract")
        assert classification in allowed_classes, f"{path}: missing/invalid contract classification"
        readers = contract.get("x-sceneworks-readers", [])
        exemption = contract.get("x-sceneworks-reader-exemption")
        assert readers or exemption, (
            f"{path}: every contract needs anchored production readers or an explicit tracked exemption"
        )
        if exemption:
            assert re.search(r"\bsc-\d+\b", exemption), (
                f"{path}: reader exemption must cite a tracked Shortcut story"
            )
        for reader in readers:
            assert set(reader) == {"path", "anchor"}, f"{path}: malformed reader metadata"
            reader_path = ROOT / reader["path"]
            assert reader_path.is_file(), f"{path}: reader does not exist: {reader['path']}"
            assert reader["anchor"] in reader_path.read_text(encoding="utf-8"), (
                f"{path}: reader {reader['path']} no longer contains anchor {reader['anchor']!r}"
            )


# ---------------------------------------------------------------------------
# Per-model `mlx` block structural audits (parsed JSONC manifest).
# Extracted from tests/test_worker_image_adapters.py (sc-8861 / F-059).
# ---------------------------------------------------------------------------


def test_flux_manifest_has_mlx_block():
    # Manifest-driven auto-dispatch + Model Manager memory tier (sc-1970).
    # The Rust API owns the canonical jsonc parser; here we just confirm both
    # FLUX entries carry an `mlx` block and the contents look right.
    models = {model["id"]: model for model in _load_builtin_models_manifest()["models"]}

    for model_id in ("flux_schnell", "flux_dev"):
        mlx = models[model_id]["mlx"]
        assert mlx["quantize"] in {3, 4, 5, 6, 8}, (
            f"{model_id} mlx.quantize must be a supported quant level (sc-1970)"
        )
        assert mlx["minMemoryGb"] > 0, (
            f"{model_id} mlx.minMemoryGb must be a positive int (sc-1970)"
        )


def test_qwen_image_manifest_has_mlx_block():
    # sc-1972: qwen_image carries an mlx block + sampler/scheduler limits
    # override (mflux's loop is sealed on "linear" — match the wan_2_2
    # precedent of restricting the menu to default-only when the MLX path is
    # the active backend, epic 1753 §14).
    model = next(
        model
        for model in _load_builtin_models_manifest()["models"]
        if model["id"] == "qwen_image"
    )
    mlx = model["mlx"]
    assert mlx["quantize"] in {3, 4, 5, 6, 8}, (
        "qwen_image mlx.quantize must be a supported quant level (sc-1972)"
    )
    assert mlx["minMemoryGb"] > 0, (
        "qwen_image mlx.minMemoryGb must be a positive int (sc-1972)"
    )
    # MLX sampler/scheduler menu (epic 7114 P5, sc-7126): the native MLX engine now
    # routes through the unified curated sampler/scheduler framework (the old mflux
    # linear-only loop is gone), so the mlx block advertises the curated menu.
    assert {"dpmpp_2m", "uni_pc"} <= set(mlx["limits"]["samplers"]), (
        "qwen_image mlx must advertise the curated sampler menu (epic 7114)"
    )
    assert "sgm_uniform" in mlx["limits"]["schedulers"], (
        "qwen_image mlx must advertise the curated scheduler menu (epic 7114)"
    )


def test_flux2_true_v2_manifest_install_time_conversion():
    # sc-2235: the entry must declare the install-time conversion contract the
    # Rust convert job + adapter rely on.
    model = next(
        model
        for model in _load_builtin_models_manifest()["models"]
        if model["id"] == "flux2_klein_9b_true_v2"
    )
    assert model["macOnly"] is True
    assert model["adapter"] == "mlx_flux2"
    # Only the bf16 single-file is pulled (not the whole 73 GB repo).
    assert model["downloads"][0]["files"] == ["Flux2-Klein-9B-True-v2-bf16.safetensors"]
    # Undistilled defaults differ from the 4-step distill.
    assert model["defaults"]["steps"] == 24
    mlx = model["mlx"]
    assert mlx["requiresConversion"] is True
    assert mlx["converter"] == "flux2_klein_diffusers"
    assert mlx["convertSourceRepo"] == "wikeeyang/Flux2-Klein-9B-True-V2"
    # sc-14978: the convert borrows the base VAE/text-encoder/tokenizer from the ungated
    # re-host the base klein card actually installs (NOT the gated upstream, which no card
    # installs), reading them from its per-tier bf16/ subdir.
    assert mlx["convertBaseRepo"] == "SceneWorks/flux2-klein-9b-mlx"
    assert mlx["convertBaseSubdir"] == "bf16"
    assert mlx["quantize"] == 8


def test_flux2_klein_manifest_entries_present():
    # Both flux2_klein_9b and flux2_klein_9b_kv must be present in the
    # builtin manifest with the expected adapter + family + mlx block.
    models = {model["id"]: model for model in _load_builtin_models_manifest()["models"]}
    # Both ids expose the same capability set: -kv is no longer gated to
    # character_image only — it runs plain txt2img on par with the base 9B,
    # the cache just doesn't engage without a reference (sc-2173).
    for model_id in ("flux2_klein_9b", "flux2_klein_9b_kv"):
        model = models[model_id]
        assert model["adapter"] == "mlx_flux2", model_id
        assert model["family"] == "flux2-klein", model_id
        assert model["macOnly"] is True, model_id
        # sc-8711 (epic 8506): re-hosted as a public, ungated SceneWorks MLX quant-matrix
        # turnkey (q4/q8/bf16), so the entry is `gated: false` with no credentialHost — the
        # FLUX Non-Commercial LICENSE.md travels with the weights.
        assert model["gated"] is False, model_id
        # quantize records the DEFAULT tier (q4); the load Quant is forced to None so the
        # dense bf16 Qwen3 TE is preserved (DENSE_TE_TIER_MODELS).
        assert model["mlx"]["quantize"] == 4, f"{model_id}: default tier should be q4 (sc-8711)"
        assert {"text_to_image", "character_image"} <= set(model["capabilities"]), model_id


def test_z_image_turbo_manifest_has_mlx_block():
    # sc-2145: z_image_turbo carries an mlx block + sampler/scheduler limits
    # override (mflux's loop is sealed on "linear" — match the wan_2_2 /
    # qwen_image precedents of restricting the menu to default-only when the
    # MLX path is the active backend, epic 1753 §14).
    model = next(
        model
        for model in _load_builtin_models_manifest()["models"]
        if model["id"] == "z_image_turbo"
    )
    mlx = model["mlx"]
    assert mlx["quantize"] in {3, 4, 5, 6, 8}, (
        "z_image_turbo mlx.quantize must be a supported quant level (sc-2145)"
    )
    assert mlx["minMemoryGb"] > 0, (
        "z_image_turbo mlx.minMemoryGb must be a positive int (sc-2145)"
    )
    # epic 7114 P5 (sc-7126): the native MLX engine adopted the unified curated
    # sampler/scheduler framework, so the mflux linear-only restriction is gone.
    assert {"dpmpp_2m", "uni_pc"} <= set(mlx["limits"]["samplers"]), (
        "z_image_turbo mlx must advertise the curated sampler menu (epic 7114)"
    )
    assert "sgm_uniform" in mlx["limits"]["schedulers"], (
        "z_image_turbo mlx must advertise the curated scheduler menu (epic 7114)"
    )


def test_krea_2_turbo_candle_vram_tiers_match_measured_peaks():
    """sc-12126/sc-13108: never regress the directly measured standard-tier peaks."""
    manifest = _load_builtin_models_manifest()
    krea = next(model for model in manifest["models"] if model["id"] == "krea_2_turbo")
    measured_tiers = {
        tier: krea["candle"]["vramGbByTier"][tier] for tier in ("q4", "q8", "bf16")
    }

    assert measured_tiers == {
        "q4": 25.7,
        "q8": 35.2,
        "bf16": 47.2,
    }


def test_boogu_candle_vram_tiers_cover_and_pin_the_default_q8_tier():
    """sc-13533: both Boogu entries must carry a MEASURED q8 row — the tier they default to.

    `mlx.quantize: 8` makes the image-lane resolvers derive q8 for a no-pick request, and the shipped
    `base/`/`turbo/` Q8 turnkey is the ONLY variant `downloads` pulls. The candle blocks originally
    shipped only {q4, bf16} (sc-13108 measured only those two), so
    `vram_gate::predicted_peak_gb(entry, "q8")` found no row and fell through to the flat `minMemoryGb`
    floor, sizing the default tier ~2 GB UNDER its real peak and without the fit gate's 2 GB headroom —
    the permissive direction. The q8 rows below are the direct CUDA measurements (RTX PRO 6000
    Blackwell, exclusive GPU, 1024², seed 42, native path) that close it. Never regress them, and never
    drop the q8 key. Pairs with the Rust coverage lint
    `every_image_model_budgets_its_default_tier_against_a_measured_row`.
    """
    manifest = _load_builtin_models_manifest()
    expected = {
        "boogu_image": {"q4": 31.7, "q8": 42.0, "bf16": 54.4},
        "boogu_image_turbo": {"q4": 31.6, "q8": 42.1, "bf16": 54.5},
    }
    for model_id, tiers in expected.items():
        entry = next(model for model in manifest["models"] if model["id"] == model_id)
        assert {
            tier: entry["candle"]["vramGbByTier"][tier] for tier in ("q4", "q8", "bf16")
        } == tiers, model_id
        # The coarse `minMemoryGb` floor must not sit BELOW the DEFAULT (q8) tier's measured peak —
        # that under-floor (turbo shipped 40 < 42.1) was the second face of this bug, exposed whenever
        # `predicted_peak_gb` falls back to `minMemoryGb`.
        assert entry["candle"]["minMemoryGb"] >= tiers["q8"], model_id


def test_wan_2_2_candle_vram_tiers_match_measured_peaks():
    """sc-13175: never regress the measured 5B SEQUENTIAL peaks (or slide back to the resident ones).

    Re-dropped onto the sequential-offload path (sc-12757 flushes the UMT5 TE + z48 VAE off-GPU around
    the dense denoise), so these SUPERSEDE the resident numbers sc-12631 shipped (q4 46.1 / q8 48.7 /
    bf16 54.0, minMemoryGb 48). Measured on an idle RTX PRO 6000 at wan_2_2's own shipped default
    (832x480, 121 frames, 20 steps, CFG on, CANDLE_GEN_OFFLOAD=sequential), each tier in its own process.
    The peak is the tier-blind denoise attention transient, not the weights -- so q4 and q8 land on the
    SAME pool high-water and only the dense bf16 DiT is heavier; the z48 vae22 decode is the lower phase,
    which makes these card-independent. The numbers are the nvidia-smi POOL high-water (the real max
    device footprint, since cudarc never frees the pool), NOT the lower USED_MEM_HIGH concurrent-live
    (10.61/10.61/11.67 GiB) -- gating at the pool bound is the conservative answer to the sc-13174
    pool-vs-USED_MEM_HIGH caveat, so all three ship `measured: true` with no small-card packdown assumption.
    """
    manifest = _load_builtin_models_manifest()
    wan = next(model for model in manifest["models"] if model["id"] == "wan_2_2")
    candle = wan["candle"]

    assert candle["measured"] is True
    assert {tier: candle["vramGbByTier"][tier] for tier in ("q4", "q8", "bf16")} == {
        "q4": 12.1,
        "q8": 12.1,
        "bf16": 14.5,
    }
    # minMemoryGb gates the default/lightest (q4) tier + the fit gate's 2 GB headroom (12.1 + ~2).
    assert candle["minMemoryGb"] == 14
    # The re-drop's whole point: the heaviest tier's peak + the gate's 2 GB headroom still clears a 24 GB
    # card (the resident 46.1 needed ~48). If this regresses, the 5B silently walls off the card it targets.
    assert candle["vramGbByTier"]["bf16"] + 2 < 24


def test_wan_a14b_candle_all_tiers_measured_q8_admits_32gb():
    """sc-13174 (completing sc-12631): the A14B q4/q8/bf16 candle peaks are ALL MEASURED, and q8 now
    admits a 32 GB card.

    After the sequential-offload / expert-swap / bf16-TE / free-aware-tiling / finer-sdpa rework (epic
    sc-12732), the A14B renders one 14B expert at a time. Its measured `USED_MEM_HIGH` peaks at the
    1280x720/81f/4-step Lightning default are ~22 (q4) / ~28 (q8) / ~39 (bf16) GiB -- not the ~386 GiB
    OOM-floor these blocks used to carry. sc-12631 shipped q4 measured but DEFERRED q8/bf16; sc-13174
    completes them:
      * q8's live peak is ~28 GiB, but its nvidia-smi pool high-water (~34-36, which cudarc never frees)
        left it unproven whether a <=32 GB card packs down to the live peak. A GPU-memory-balloon
        emulation (64 GiB balloon -> ~31 GiB free) reproduced the SAME ~28 live peak at full GPU util with
        no spill, so q8 is gated at its live peak and now ADMITS a 32 GB RTX 5090 -- the epic goal.
      * bf16 was staged (dense fp32 diffusers, after downloading the missing transformer_2 shards) and
        measured at ~39 GiB (one bf16 expert + activations), REPLACING the old conservative derived 56
        bound: the real number admits a 48 GB card but stays refused on 32.
    Pinning the exact values (not just measured:true) mutation-checks the flip -- ripping a tier out or
    regressing q8 back to its pool bound goes RED here. This is the inverse of the sc-12631
    `..._q4_measured_admits_32gb_q8_bf16_deferred` tripwire it replaces.
    """
    manifest = _load_builtin_models_manifest()
    expected = {
        "wan_2_2_t2v_14b": {"q4": 22.13, "q8": 27.95, "bf16": 38.56},
        "wan_2_2_i2v_14b": {"q4": 22.20, "q8": 28.02, "bf16": 38.62},
    }
    for model_id, tiers in expected.items():
        entry = next(m for m in manifest["models"] if m["id"] == model_id)
        candle = entry["candle"]
        # q4/q8/bf16 are all measured now, so the block is honestly measured:true.
        assert candle["measured"] is True, f"{model_id}: q8+bf16 now measured, so measured flips to true"
        assert candle["vramGbByTier"] == tiers, (
            f"{model_id}: the measured q4/q8/bf16 peaks must not regress, got {candle['vramGbByTier']}"
        )
        assert candle["minMemoryGb"] == 24, f"{model_id}: minMemoryGb should gate q4 (~22 + 2)"
        # q4 AND q8 now fit a 32 GB card (each + the fit gate's 2 GB headroom); bf16 does not.
        assert tiers["q4"] + 2 < 32, f"{model_id}: q4 (+headroom) must fit a 32 GB card, got {tiers['q4']}"
        assert tiers["q8"] + 2 < 32, (
            f"{model_id}: q8 (+headroom) now fits a 32 GB card after the <=32 GB balloon validation, "
            f"got {tiers['q8']}"
        )
        # bf16 stays refused on a 32 GB card, but its measured peak now admits a 48 GB card (the derived
        # 56 bound refused 48).
        assert tiers["bf16"] + 2 > 32, f"{model_id}: bf16 must stay refused on a 32 GB card, got {tiers['bf16']}"
        assert tiers["bf16"] + 2 <= 48, (
            f"{model_id}: the measured bf16 peak must now admit a 48 GB card, got {tiers['bf16']}"
        )
        # Heavier tier => heavier peak (ordering sanity).
        assert tiers["q4"] < tiers["q8"] < tiers["bf16"], f"{model_id}: heavier tier => heavier peak"


def test_sdxl_manifest_has_mlx_block():
    # sdxl carries an mlx block (no `limits` override here — the MLX SDXL schedule
    # matches the torch EulerDiscrete default, and there's no per-model sampler menu
    # in the sdxl manifest entry to limit).
    model = next(
        model
        for model in _load_builtin_models_manifest()["models"]
        if model["id"] == "sdxl"
    )
    assert model["mlx"]["minMemoryGb"] > 0, (
        "sdxl mlx.minMemoryGb must be a positive int"
    )


# ---------------------------------------------------------------------------
# character_image capability / UI-wiring audits (manifest-parsed dict).
# Extracted from tests/test_worker_image_adapters.py (sc-8861 / F-059).
#
# The three character_image ENGINE-WIRING guards that used to live here
# (test_character_image_capability_implies_engine_or_tuning_declaration /
# test_kolors_declares_strict_pose_controlnet /
# test_models_with_engine_block_advertise_character_image) cross-referenced the
# retired Python worker's MODEL_TARGETS table, so they were reimplemented against
# the Rust worker's own character-image engine wiring in
# crates/sceneworks-worker/src/engines.rs (sc-9513). The manifest-only symmetry
# guard below has no worker dependency and stays here.
# ---------------------------------------------------------------------------


def test_hide_reference_strength_models_declare_a_variation_knob():
    """Symmetry guard for the sc-2017 picker UX. A model that opts out of the
    IP-Adapter reference-strength slider via `ui.hideReferenceStrength` MUST
    also declare `ui.variationStrength` — otherwise the picker shows no tuning
    control at all, and the worker silently runs at default true_cfg_scale.
    """
    manifest = _load_builtin_models_manifest()
    unbalanced: list[str] = []
    for model in manifest.get("models", []):
        ui = model.get("ui") or {}
        if not ui.get("hideReferenceStrength"):
            continue
        if not ui.get("variationStrength"):
            unbalanced.append(model["id"])
    assert not unbalanced, (
        f"Models hide the Reference-strength slider without declaring "
        f"`ui.variationStrength`: {unbalanced}. The picker would leave the user "
        f"with NO identity tuning control. Add `variationStrength` or drop "
        f"`hideReferenceStrength`."
    )


# ---------------------------------------------------------------------------
# sc-13606 (F-037): even out schema enforcement across the SIBLING builtin
# catalogs (loras / styles / control_overlays / recipe-presets).
#
# Before this, only builtin.models.jsonc got full JSON-Schema CI validation
# (sc-12338, the audits above). The lora schema required nothing per entry and
# left `source` wide open (additionalProperties:true), so a typo'd
# `source.repo`/`source.file` — the sc-12288 field class behind a silent failed
# download — passed; builtin.styles.jsonc declared a `$schema` NO check
# validated; builtin.control_overlays.jsonc had no schema at all. Authoring
# errors therefore surfaced at runtime (failed downloads, missing overlays /
# styles) instead of in CI. These tests mirror the model audit above: load each
# JSONC catalog, validate it against its schema, and prove each new constraint
# catches a deliberately-broken entry with a schema-keyword-DISCRIMINATING
# mutation check — NOT a test that merely asserts the current catalog passes.
#
# The check-scaffold registry (scripts/check-scaffold.mjs) is kept in lockstep:
# it now lists all five schema/manifest pairs so the scaffold/parity lane knows
# the styles + control-overlays pairs exist and reference a real schema; the
# DEEP jsonschema validation is this pytest lane's job (the scaffold has no
# jsonschema dependency), the same division of labour the model catalog uses.
# ---------------------------------------------------------------------------

LORA_MANIFEST_PATH = ROOT / "config" / "manifests" / "builtin.loras.jsonc"
LORA_SCHEMA_PATH = ROOT / "packages" / "schemas" / "lora-manifest.schema.json"
STYLES_MANIFEST_PATH = ROOT / "config" / "manifests" / "builtin.styles.jsonc"
STYLES_SCHEMA_PATH = ROOT / "packages" / "schemas" / "styles.schema.json"
CONTROL_OVERLAYS_MANIFEST_PATH = ROOT / "config" / "manifests" / "builtin.control_overlays.jsonc"
CONTROL_OVERLAYS_SCHEMA_PATH = ROOT / "packages" / "schemas" / "control-overlays.schema.json"
RECIPE_PRESETS_MANIFEST_PATH = ROOT / "config" / "manifests" / "builtin.recipe-presets.jsonc"
RECIPE_PRESET_SCHEMA_PATH = ROOT / "packages" / "schemas" / "recipe-preset.schema.json"


def _load_jsonc(path: Path) -> dict:
    """Generalization of `_load_builtin_models_manifest` for the sibling catalogs."""
    return copy.deepcopy(_cached_jsonc(path))


def _schema_errors(manifest: dict, schema_path: Path) -> list:
    """Validate `manifest` against the schema at `schema_path`, returning the
    (path-sorted) validation errors. Also asserts the schema itself is a legal
    Draft 2020-12 document (a broken schema is a silent all-pass otherwise)."""
    schema = _load_schema(schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    return sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: list(error.absolute_path),
    )


def _format_errors(errors) -> str:
    return "\n".join(
        f"- {'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
        for error in errors
    )


# --- LoRA catalog (builtin.loras.jsonc) ------------------------------------


def _sample_lora_entry() -> dict:
    """A minimal schema-valid LoRA entry for exercising the entry/source schema in
    isolation (never shipped — real entries live in builtin.loras.jsonc)."""
    return {
        "id": "sample_lora",
        "name": "Sample LoRA",
        "family": "krea_2",
        "compatibility": {"families": ["krea_2"]},
        "defaultWeight": 1.0,
        "source": {
            "provider": "huggingface",
            "repo": "namespace/sample-lora",
            "file": "sample.safetensors",
        },
    }


def test_builtin_loras_manifest_satisfies_authoring_schema():
    """sc-13606: builtin.loras.jsonc is now a full-jsonschema CI contract, not just
    a shallow root-key check (parity with the model catalog)."""
    errors = _schema_errors(_load_jsonc(LORA_MANIFEST_PATH), LORA_SCHEMA_PATH)
    assert not errors, "builtin.loras.jsonc violates lora-manifest.schema.json:\n" + _format_errors(errors)


def test_lora_schema_requires_entry_identity():
    """Every LoRA entry carries the complete identity consumed by catalog and
    compatibility paths: id, display name, and normalized family."""
    for field in ("id", "name", "family"):
        entry = _sample_lora_entry()
        del entry[field]
        errors = _schema_errors({"schemaVersion": 1, "loras": [entry]}, LORA_SCHEMA_PATH)
        assert any(
            error.validator == "required"
            and field in error.validator_value
            and list(error.absolute_path) == ["loras", 0]
            for error in errors
        ), f"a LoRA entry without `{field}` must be rejected by the entry identity contract"


def test_lora_schema_rejects_a_typod_source_key():
    """The sc-12288 field class: a typo'd `source.file` (or `repo`) key silently
    produced a failed download. The typed loraSource is additionalProperties:false,
    so the typo now fails at authoring time. Pin the keyword + path so a revert to
    an open source object goes red."""
    entry = _sample_lora_entry()
    entry["source"]["filez"] = entry["source"].pop("file")
    errors = _schema_errors({"schemaVersion": 1, "loras": [entry]}, LORA_SCHEMA_PATH)
    assert any(
        error.validator == "additionalProperties"
        and list(error.absolute_path) == ["loras", 0, "source"]
        and "'filez'" in error.message
        for error in errors
    ), "a typo'd key under `source` must be rejected by additionalProperties:false"


def test_lora_schema_requires_repo_on_source():
    """The typed source requires `provider`+`repo` (all builtin entries are HF
    hosted; the runtime errors without a repo). A `source` missing `repo` fails."""
    entry = _sample_lora_entry()
    del entry["source"]["repo"]
    errors = _schema_errors({"schemaVersion": 1, "loras": [entry]}, LORA_SCHEMA_PATH)
    assert any(
        error.validator == "required"
        and "repo" in error.validator_value
        and list(error.absolute_path) == ["loras", 0, "source"]
        for error in errors
    ), "a `source` without `repo` must be rejected by the typed source's required list"


def test_lora_schema_rejects_an_unknown_entry_key():
    """The entry object is additionalProperties:false, so a decorative/typo'd
    entry-level key (the model-catalog `recommendded` mutation, ported) fails."""
    entry = _sample_lora_entry()
    entry["recommendded"] = True
    errors = _schema_errors({"schemaVersion": 1, "loras": [entry]}, LORA_SCHEMA_PATH)
    assert any(
        error.validator == "additionalProperties"
        and list(error.absolute_path) == ["loras", 0]
        and "'recommendded'" in error.message
        for error in errors
    )


def test_lora_schema_rejects_a_non_40hex_source_revision():
    """`source.revision` reuses the model schema's 40-hex pin pattern, so a
    branch/tag value is rejected by the `pattern` keyword."""
    entry = _sample_lora_entry()
    entry["source"]["revision"] = "main"
    errors = _schema_errors({"schemaVersion": 1, "loras": [entry]}, LORA_SCHEMA_PATH)
    assert any(
        error.validator == "pattern"
        and list(error.absolute_path) == ["loras", 0, "source", "revision"]
        for error in errors
    ), "a non-40-hex source.revision must be rejected by the pattern"


def test_lora_source_guard_is_live_against_the_real_catalog():
    """Mutation guard: the typed-source rule is LIVE on the SHIPPED catalog, not
    decoration on a synthetic fixture. Typo the first real entry's `source.repo`
    and validation must fail (proving the guard exercises the real file)."""
    manifest = _load_jsonc(LORA_MANIFEST_PATH)
    manifest["loras"][0]["source"]["reppo"] = manifest["loras"][0]["source"].pop("repo")
    errors = _schema_errors(manifest, LORA_SCHEMA_PATH)
    assert any(
        error.validator == "additionalProperties"
        and list(error.absolute_path) == ["loras", 0, "source"]
        and "'reppo'" in error.message
        for error in errors
    )


# --- Control-overlay catalog (builtin.control_overlays.jsonc) ---------------


def _sample_control_overlay_entry() -> dict:
    """A minimal schema-valid control-overlay entry (never shipped)."""
    return {
        "id": "sample_overlay",
        "name": "Sample Overlay",
        "baseModel": "krea_2_turbo",
        "controlType": "pose",
        "source": {
            "provider": "huggingface",
            "repo": "namespace/sample-overlay",
            "file": "control.safetensors",
        },
        "files": ["control.safetensors"],
    }


def test_builtin_control_overlays_manifest_satisfies_authoring_schema():
    """sc-13606: builtin.control_overlays.jsonc previously had NO schema at all; it
    now validates against control-overlays.schema.json in CI."""
    errors = _schema_errors(
        _load_jsonc(CONTROL_OVERLAYS_MANIFEST_PATH), CONTROL_OVERLAYS_SCHEMA_PATH
    )
    assert not errors, (
        "builtin.control_overlays.jsonc violates control-overlays.schema.json:\n"
        + _format_errors(errors)
    )


def test_control_overlay_schema_requires_its_core_fields():
    """Each of id/name/baseModel/controlType/source is load-bearing (the picker
    filters by baseModel + controlType; the registry keys on id; the runtime reads
    source). Dropping any one must produce a `required` error at the entry."""
    for field in ("id", "name", "baseModel", "controlType", "source"):
        entry = _sample_control_overlay_entry()
        del entry[field]
        errors = _schema_errors(
            {"schemaVersion": 1, "controlOverlays": [entry]}, CONTROL_OVERLAYS_SCHEMA_PATH
        )
        assert any(
            error.validator == "required"
            and field in error.validator_value
            and list(error.absolute_path) == ["controlOverlays", 0]
            for error in errors
        ), f"a control overlay without `{field}` must be rejected by the entry required list"


def test_control_overlay_schema_rejects_a_typod_source_key():
    """The sc-12288 field class on the overlay source (additionalProperties:false)."""
    entry = _sample_control_overlay_entry()
    entry["source"]["repoo"] = entry["source"].pop("repo")
    errors = _schema_errors(
        {"schemaVersion": 1, "controlOverlays": [entry]}, CONTROL_OVERLAYS_SCHEMA_PATH
    )
    assert any(
        error.validator == "additionalProperties"
        and list(error.absolute_path) == ["controlOverlays", 0, "source"]
        and "'repoo'" in error.message
        for error in errors
    )


def test_control_overlay_schema_rejects_an_unknown_entry_key():
    """The overlay entry is additionalProperties:false — a typo'd field name fails."""
    entry = _sample_control_overlay_entry()
    entry["controlTyp"] = "pose"
    errors = _schema_errors(
        {"schemaVersion": 1, "controlOverlays": [entry]}, CONTROL_OVERLAYS_SCHEMA_PATH
    )
    assert any(
        error.validator == "additionalProperties"
        and list(error.absolute_path) == ["controlOverlays", 0]
        and "'controlTyp'" in error.message
        for error in errors
    )


def test_control_overlay_guard_is_live_against_the_real_catalog():
    """Mutation guard: the overlay schema is LIVE on the shipped catalog. Typo the
    real entry's `source.file` and validation must fail."""
    manifest = _load_jsonc(CONTROL_OVERLAYS_MANIFEST_PATH)
    source = manifest["controlOverlays"][0]["source"]
    source["fil"] = source.pop("file")
    errors = _schema_errors(manifest, CONTROL_OVERLAYS_SCHEMA_PATH)
    assert any(
        error.validator == "additionalProperties"
        and list(error.absolute_path) == ["controlOverlays", 0, "source"]
        and "'fil'" in error.message
        for error in errors
    )


# --- Style catalog (builtin.styles.jsonc) ----------------------------------


def test_builtin_styles_manifest_satisfies_authoring_schema():
    """sc-13606: builtin.styles.jsonc declared a `$schema` that NO check validated
    (F-037). It is now an enforced CI contract like the model catalog. (The catalog
    is machine-generated from documents/style.txt and drift-guarded by
    styleCatalog.test.js; this only validates its shape, it does not hand-edit it.)"""
    errors = _schema_errors(_load_jsonc(STYLES_MANIFEST_PATH), STYLES_SCHEMA_PATH)
    assert not errors, "builtin.styles.jsonc violates styles.schema.json:\n" + _format_errors(errors)


def test_styles_schema_rejects_an_unknown_style_key_on_the_real_catalog():
    """Mutation guard: the styles schema pins additionalProperties:false on each
    style object. Injecting a decorative/typo'd key into the first real style must
    fail — proving the now-enforced schema is live, not merely declared."""
    manifest = _load_jsonc(STYLES_MANIFEST_PATH)
    manifest["groups"][0]["styles"][0]["prromt"] = "typo"
    errors = _schema_errors(manifest, STYLES_SCHEMA_PATH)
    assert any(
        error.validator == "additionalProperties"
        and list(error.absolute_path) == ["groups", 0, "styles", 0]
        and "'prromt'" in error.message
        for error in errors
    )


def test_styles_schema_requires_a_style_prompt():
    """A style object requires id/name/prompt; dropping `prompt` from the first
    real style must be rejected by the style object's required list."""
    manifest = _load_jsonc(STYLES_MANIFEST_PATH)
    del manifest["groups"][0]["styles"][0]["prompt"]
    errors = _schema_errors(manifest, STYLES_SCHEMA_PATH)
    assert any(
        error.validator == "required"
        and "prompt" in error.validator_value
        and list(error.absolute_path) == ["groups", 0, "styles", 0]
        for error in errors
    )


# --- Recipe-preset catalog (builtin.recipe-presets.jsonc) -------------------


def test_builtin_recipe_presets_manifest_satisfies_authoring_schema():
    """sc-13606: the recipe-preset catalog now gets full jsonschema validation in
    the pytest lane (previously only the shallow check-scaffold root-key check).
    The shipped catalog is currently empty, so the discriminating guards below run
    against synthetic presets — the schema, not the catalog, carries the rules."""
    errors = _schema_errors(_load_jsonc(RECIPE_PRESETS_MANIFEST_PATH), RECIPE_PRESET_SCHEMA_PATH)
    assert not errors, (
        "builtin.recipe-presets.jsonc violates recipe-preset.schema.json:\n" + _format_errors(errors)
    )


def test_recipe_preset_schema_requires_id_and_name():
    """Every preset requires id + name. Dropping `name` from an otherwise-valid
    general preset must be rejected by the entry's required list."""
    preset = {"id": "sample_preset", "name": "Sample Preset", "kind": "general"}
    del preset["name"]
    errors = _schema_errors({"schemaVersion": 1, "presets": [preset]}, RECIPE_PRESET_SCHEMA_PATH)
    assert any(
        error.validator == "required"
        and "name" in error.validator_value
        and list(error.absolute_path) == ["presets", 0]
        for error in errors
    )


def test_recipe_preset_schema_requires_model_and_workflow_for_model_presets():
    """A preset with no `kind` is a MODEL preset; the schema's allOf conditional
    then requires model + workflow. A bare {id,name} model preset must fail — this
    only holds while that conditional exists (discriminates a revert)."""
    preset = {"id": "sample_model_preset", "name": "Sample"}
    errors = _schema_errors({"schemaVersion": 1, "presets": [preset]}, RECIPE_PRESET_SCHEMA_PATH)
    assert any(
        error.validator == "required" and set(error.validator_value) == {"model", "workflow"}
        for error in errors
    ), "a model (non-general) preset must require model + workflow via the allOf conditional"


def test_recipe_preset_schema_rejects_a_bad_id_pattern():
    """The preset `id` is pattern-constrained (lowercase slug); an id with spaces /
    capitals / punctuation is rejected by the `pattern` keyword."""
    preset = {"id": "Bad Id!", "name": "Sample", "kind": "general"}
    errors = _schema_errors({"schemaVersion": 1, "presets": [preset]}, RECIPE_PRESET_SCHEMA_PATH)
    assert any(
        error.validator == "pattern" and list(error.absolute_path) == ["presets", 0, "id"]
        for error in errors
    )
