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
import math
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
            "sdxl",
            "xinsir/controlnet-openpose-sdxl-1.0",
            "diffusion_pytorch_model.safetensors",
            "23f966cd5cfdd3f7729c903e243d87152162d2b7",
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


def test_ltx25_builtin_manifest_uses_published_tier_sizes_and_geometry():
    """sc-18781: one selected tier installs both published transformer identities."""
    manifest = _load_builtin_models_manifest()
    model = next(model for model in manifest["models"] if model["id"] == "ltx_2_5")
    co_requisites = [row for row in model["downloads"] if row.get("coRequisite")]
    tiers = {
        row["variant"]: row
        for row in model["downloads"]
        if not row.get("coRequisite")
    }
    expected_bytes = {
        "q4": 82_001_022_554,
        "q8": 83_672_119_594,
        "bf16": 145_561_735_442,
    }

    assert set(tiers) == set(expected_bytes)
    for tier, measured_bytes in expected_bytes.items():
        row = tiers[tier]
        assert row["files"] == [f"distilled/{tier}/*", f"dev/{tier}/*"]
        assert row["platforms"] == ["macos", "windows", "linux"]
        assert row["estimatedSizeBytes"] == measured_bytes
        assert row["footprint"]["diskSizeBytes"] == measured_bytes
        assert row["footprint"]["residentMemoryBytes"] is None
        assert row["footprint"]["peakMemoryBytes"] is None

    expected_co_requisites = {
        ("enhancer/*",): 23_951_746_871,
        ("distilled_lora/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",): 8_899_889_568,
    }
    assert len(co_requisites) == len(expected_co_requisites)
    for row in co_requisites:
        assert row["platforms"] == ["macos", "windows", "linux"]
        assert row["estimatedSizeBytes"] == expected_co_requisites[tuple(row["files"])]

    assert model["defaults"]["steps"] == 8
    assert model["limits"]["steps"] == [8, 30]
    assert model["limits"]["requiresDimensionsMultipleOf"] == 64
    assert "maxPixels" not in model["limits"]


def test_ltx25_builtin_manifest_declares_the_exact_backend_memory_ladders():
    """sc-18800: static declarations mirror the two provider contracts without false exemptions."""
    manifest = _load_builtin_models_manifest()
    model = next(model for model in manifest["models"] if model["id"] == "ltx_2_5")

    assert model["mlx"]["memoryStrategyCapabilities"] == {
        "bounded_decode": {
            "parameters": {"decodeTileEdge": 192, "decodeOverlap": 64},
            "overlays": ["none", "lora"],
        },
        "bounded_attention": {
            "parameters": {"attentionChunkSize": 16_777_216},
            "overlays": ["none", "lora"],
        },
        "bounded_transformer_residency": {
            "parameters": {
                "transformerWindowSize": 1,
                "transformerWindowComponent": "Dit",
            },
            "overlays": ["none"],
        },
    }
    assert model["candle"] == {
        "memoryStrategyCapabilities": {
            "bounded_decode": {
                "parameters": {"decodeTileEdge": 192, "decodeOverlap": 64},
                "overlays": ["none", "lora"],
                "tiers": ["q4", "bf16"],
            },
            "bounded_attention": {
                "parameters": {"attentionChunkSize": 16_777_216},
                "overlays": ["none", "lora"],
                "tiers": ["q4", "bf16"],
            },
            "bounded_transformer_residency": {
                "parameters": {
                    "transformerWindowSize": 1,
                    "transformerWindowComponent": "Dit",
                },
                "overlays": ["none"],
                "tiers": ["q4", "bf16"],
            },
        }
    }
    for backend in ("mlx", "candle"):
        assert "memoryStrategyStructuralExemptions" not in model[backend]
    assert "supportsSequentialOffload" not in model["candle"]


COMPONENTS_REPO = "SceneWorks/Mage-Flow-Components-mlx"
COMPONENTS_REVISION = "c936de2a107ee8d0869137e73943f6414f23adaa"
# Measured on the uploaded artifacts (sc-14980). Per-tier DiT, and the shared per-tier components.
DIT_BYTES = {"q4": 2326294167, "q8": 4374163324, "bf16": 8231571754}
TE_BYTES = {"q4": 4331077508, "q8": 4731076490, "bf16": 8887219239}
VAE_BYTES = 345053168
TIERS = ("q4", "q8", "bf16")


def _assert_mage_tier_layout(model, model_id, repo, revision):
    """sc-14980/sc-14979: the shared per-tier shape every Mage row must have.

    Supersedes the sc-14047/sc-14050 "complete flat snapshot" pin: the tiers are physically
    distinct `<tier>/` artifacts now, and the text encoder + VAE are hosted once as per-tier
    co-requisites rather than duplicated into all six variant mirrors.
    """
    downloads = model["downloads"]
    tiers = [d for d in downloads if not d.get("coRequisite")]
    co_reqs = [d for d in downloads if d.get("coRequisite")]

    assert [d["variant"] for d in tiers] == list(TIERS), model_id
    assert sum(d.get("default") is True for d in tiers) == 1, model_id
    for entry in tiers:
        tier = entry["variant"]
        assert entry["repo"] == repo, model_id
        assert entry["revision"] == revision, model_id
        # Each tier fetches ONLY its own subtree — this is what makes a per-tier delete reclaim
        # real bytes. A shared full-snapshot predicate here would silently restore 0-byte deletes.
        assert entry["files"] == [f"{tier}/*"], (model_id, tier)
        assert entry["estimatedSizeBytes"] == DIT_BYTES[tier], (model_id, tier)

    # The shared components: one mirror, one pinned revision, addressed per tier via `subdir`.
    assert len(co_reqs) == 6, model_id
    for component, sizes in (("text_encoder", TE_BYTES), ("vae", None)):
        for tier in TIERS:
            row = next(
                d
                for d in co_reqs
                if d["componentId"] == component and d["variant"] == tier
            )
            assert row["repo"] == COMPONENTS_REPO, (model_id, component, tier)
            assert row["revision"] == COMPONENTS_REVISION, (model_id, component, tier)
            assert row["subdir"] == f"{tier}/{component}", (model_id, component, tier)
            assert row["files"] == [f"{tier}/{component}/*"], (model_id, component, tier)
            expected = sizes[tier] if sizes else VAE_BYTES
            assert row["estimatedSizeBytes"] == expected, (model_id, component, tier)

    assert model["mlx"]["standardTierLayout"] is True, model_id
    assert model["mlx"]["quantize"] == 4, model_id
    assert model["paths"]["model"] == f"${{HF_CACHE}}/{repo}"


def _assert_mage_candle_ladder(model: dict, model_id: str) -> None:
    """sc-15813: every Candle Mage route declares the same truthful shared ladder contract.

    sc-20246: `memoryStrategyContract` is now ENGINE-PROJECTED per variant
    (scripts/generate-manifest-memory-declarations.mjs), so it is necessarily NOT shared — each
    variant names its own provider and its own catalog modes. The shared-ladder pin below therefore
    covers the hand-authored, measured keys, and the projected contract is asserted separately in
    shape terms: it must be wholly engine-sourced and must never claim a rung this same block
    declares structurally exempt.
    """
    contract = model["candle"].get("memoryStrategyContract")
    assert contract is not None, model_id
    exempt = set(model["candle"].get("memoryStrategyStructuralExemptions", {}))
    for implementation in contract["implementations"]:
        assert implementation["source"].startswith("config/engine-capabilities/"), model_id
        assert implementation["rung"] not in exempt, (model_id, implementation["rung"])
    assert contract["provider"] == model_id, model_id

    assert {
        key: value
        for key, value in model["candle"].items()
        if key != "memoryStrategyContract"
    } == {
        "minMemoryGb": 17,
        "vramGbByTier": {"q4": 14.67, "q8": 16.95, "bf16": 20.41},
        "vramMeasuredPixels": 1024 * 1024,
        "measured": False,
        "supportsSequentialOffload": True,
        "memoryStrategyCapabilities": {
            "bounded_attention": {
                "parameters": {"attentionChunkSize": 67_108_864},
                "overlays": ["none"],
            },
            "bounded_transformer_residency": {
                "parameters": {
                    "transformerWindowSize": 1,
                    "transformerWindowComponent": "Dit",
                },
                "overlays": ["none"],
            },
        },
        "memoryStrategyStructuralExemptions": {
            "bounded_decode": {
                "overlays": ["none", "lora"],
                "evidence": [
                    {
                        "source": "inference:crates/media/candle-gen/candle-gen-mage/src/memory_strategy.rs",
                        "reason": "The provider contract classifies bounded decode as StructurallyNotApplicable because independent tiles cannot preserve Mage CoD normalization.",
                    },
                    {
                        "source": "inference:crates/media/candle-gen/candle-gen-mage/src/vae.rs",
                        "reason": "Mage VAE group normalization reduces over height and width, so a tile observes different statistics from the full latent field.",
                    },
                ],
            },
        },
    }, model_id


def test_mage_flow_generation_family_is_pinned_and_complete():
    """sc-14047 + sc-14980: the generation variants ship physical per-tier artifacts."""
    models = {model["id"]: model for model in _load_builtin_models_manifest()["models"]}
    expected = {
        "mage_flow_base": ("SceneWorks/Mage-Flow-Base", "d642341926fcdb450c17c8fbda03759e8f731c9b", 30, 5),
        "mage_flow": ("SceneWorks/Mage-Flow", "5f6455818d8ca80ce780e9c01b9e0de1d8c5f9db", 20, 5),
        "mage_flow_turbo": ("SceneWorks/Mage-Flow-Turbo", "79016f10f96b441bebb6e6f461838adb8fb3ff5c", 4, 1),
    }
    for model_id, (repo, revision, steps, guidance) in expected.items():
        model = models[model_id]
        assert model["family"] == "mage-flow"
        assert model["macOnly"] is False
        _assert_mage_candle_ladder(model, model_id)
        assert model["defaults"]["steps"] == steps
        assert model["defaults"]["guidanceScale"] == guidance
        _assert_mage_tier_layout(model, model_id, repo, revision)


def test_mage_flow_edit_family_is_pinned_complete_and_source_gated():
    """sc-14050 + sc-14980: every edit variant ships physical per-tier artifacts."""
    models = {model["id"]: model for model in _load_builtin_models_manifest()["models"]}
    expected = {
        "mage_flow_edit_base": ("SceneWorks/Mage-Flow-Edit-Base", "6c119cdac7ce7cf8c1ab4990d9c8ca18641f2c5d", 30, 5),
        "mage_flow_edit": ("SceneWorks/Mage-Flow-Edit", "dbd4a9c07faca94491ad88ab21225d62e054d9cc", 30, 5),
        "mage_flow_edit_turbo": ("SceneWorks/Mage-Flow-Edit-Turbo", "75c11a2957aca2c78272984375502105b2b235ab", 4, 1),
    }
    for model_id, (repo, revision, steps, guidance) in expected.items():
        model = models[model_id]
        assert model["family"] == "mage-flow"
        assert model["adapter"] == "mlx_mage"
        assert model["capabilities"] == ["edit_image"]
        assert model["macOnly"] is False
        _assert_mage_candle_ladder(model, model_id)
        assert model["defaults"]["steps"] == steps
        assert model["defaults"]["guidanceScale"] == guidance
        assert model["ui"]["sourceWithMultiReference"] is True
        assert model["ui"]["recommendedFor"] == ["edit_image"]
        _assert_mage_tier_layout(model, model_id, repo, revision)


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
        "image_jobs/krea_imported.rs",
        "image_jobs/qwen.rs",
        "image_jobs/qwen_control.rs",
        "image_jobs/sdxl_control.rs",
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
    # The invariant is that the centrally-authorized tuple resolves ONLY through
    # `snapshots/<revision>`, never a mutable repo cache root. Two seams satisfy it, and a consumer
    # must use one of them:
    #
    #   * `huggingface_pinned_snapshot_dir` directly — the download-on-first-use lanes;
    #   * `resolve_hf_component_file` — the shared CACHE-ONLY resolver epic 17625 (AC9) requires of
    #     any lane added since, which itself dispatches a pinned revision to
    #     `huggingface_pinned_snapshot_dir`.
    #
    # Accepting the shared seam is not a loosening: membership in this inventory is defined by
    # calling `trusted_control_weight_revision`, and that function returns either a shipped
    # artifact's pinned revision or a catalog-authorized one it validates as 40 lowercase hex — the
    # exact form `is_pinned_hf_revision` admits. So a consumer in this set always hands
    # `resolve_hf_component_file` a pinned revision, and the mutable-root branch is unreachable
    # from here.
    pinned_resolution_seams = ("huggingface_pinned_snapshot_dir", "resolve_hf_component_file")
    for path in expected_consumers:
        assert any(seam in sources[path] for seam in pinned_resolution_seams), (
            f"{path}: central tuple must resolve only through snapshots/<revision>, "
            "never a mutable repo cache root — use `huggingface_pinned_snapshot_dir` or the "
            "shared cache-only `resolve_hf_component_file`"
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


AUDITED_TOP_LEVEL_MANIFEST_REPO_LANES = {
    "image_jobs/base.rs": "model.default_repo()",
    "image_jobs/flux1_control_candle.rs": "crate::engines::default_repo_for(&request.model)",
    "image_jobs/flux_ipadapter.rs": "flux_ipadapter_default_repo(&request.model)",
    "image_jobs/instantid.rs": "INSTANTID_SDXL_REPO",
    "image_jobs/kolors_ipadapter.rs": "default_repo_for(&request.model)",
    "image_jobs/krea_control_candle.rs": "default_repo_for(&request.model)",
    "image_jobs/krea_edit_candle.rs": "default_repo_for(&request.model)",
    "image_jobs/pulid.rs": "PULID_FLUX_REPO",
    "image_jobs/pulid_candle.rs": "PULID_CANDLE_FLUX_REPO",
    "image_jobs/qwen_edit_candle.rs": "crate::engines::MODEL_TABLE",
    "image_jobs/sdxl_edit_candle.rs": "sdxl_edit_candle_default_repo(&request.model)",
    "image_jobs/sdxl_ipadapter.rs": "sdxl_ipadapter_default_repo(&request.model)",
    "image_jobs/zimage_edit_candle.rs": "default_repo_for(&request.model)",
    "sensenova_jobs.rs": "default_repo_for(&request.model)",
    "video_jobs/candle.rs": "candle_wan_tier_repo_from_downloads(request, engine_id)",
}


def _worker_sources() -> dict[str, str]:
    return {
        path.relative_to(WORKER_SOURCE_PATH).as_posix(): path.read_text(encoding="utf-8")
        for path in WORKER_SOURCE_PATH.rglob("*.rs")
    }


def _assert_top_level_manifest_repo_readers_have_audited_installed_fallbacks(
    sources: dict[str, str], audited_lanes: dict[str, str]
) -> None:
    actual_lanes = {
        relative_path
        for relative_path, source in sources.items()
        if re.search(
            r"\.model_manifest_entry\s*\.get\(\"repo\"\)",
            source.split("\n#[cfg(test)]", maxsplit=1)[0],
        )
    }
    assert actual_lanes == set(audited_lanes), (
        "top-level model_manifest_entry.repo lane inventory changed; audit every added/removed lane: "
        f"added={sorted(actual_lanes - set(audited_lanes))}, "
        f"removed={sorted(set(audited_lanes) - actual_lanes)}"
    )
    for relative_path, fallback_marker in audited_lanes.items():
        source = sources[relative_path].split("\n#[cfg(test)]", maxsplit=1)[0]
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
        source = sources[relative_path]
        assert "default_repo_for(model)" in source, (
            f"{relative_path}: per-family fallback no longer delegates to MODEL_TABLE"
        )


def test_every_top_level_manifest_repo_reader_has_an_audited_installed_fallback():
    """sc-14476 lane inventory and regression guard.

    The marker names the effective installed resolution source in each lane. Explicit constants are
    permitted only when a built-in download stages that exact repo; video's generic lane instead
    selects from the request's own downloads. Any new reader must be consciously added here.
    """
    _assert_top_level_manifest_repo_readers_have_audited_installed_fallbacks(
        _worker_sources(), AUDITED_TOP_LEVEL_MANIFEST_REPO_LANES
    )


def _assert_minimax_h3_candle_uses_exact_installed_roots(source: str) -> None:
    production = source.split("\n#[cfg(test)]", maxsplit=1)[0]
    assert '.model_manifest_entry.get("repo")' not in production, (
        "MiniMax-H3 Candle must not accept a top-level manifest repo override; its q4/q8 and "
        "bf16/shared components come from two separately pinned installed snapshots"
    )
    for declaration, use in (
        (
            'const CANDLE_MINIMAX_H3_REPO: &str = "MiniMaxAI/MiniMax-H3";',
            "candle_minimax_h3_snapshot_dir(settings, CANDLE_MINIMAX_H3_REPO)?",
        ),
        (
            'const CANDLE_MINIMAX_H3_TIER_REPO: &str = "SceneWorks/minimax-h3-mlx";',
            "candle_minimax_h3_snapshot_dir(settings, CANDLE_MINIMAX_H3_TIER_REPO)",
        ),
    ):
        assert declaration in production, f"missing exact MiniMax-H3 repo authority: {declaration}"
        assert use in production, f"MiniMax-H3 resolver no longer consumes {use}"


def test_minimax_h3_candle_repo_audit_rejects_override_and_root_mutations():
    source = _worker_sources()["video_jobs/minimax_h3.rs"]
    _assert_minimax_h3_candle_uses_exact_installed_roots(source)

    for label, mutated in (
        (
            "upstream root",
            source.replace(
                'const CANDLE_MINIMAX_H3_REPO: &str = "MiniMaxAI/MiniMax-H3";',
                'const CANDLE_MINIMAX_H3_REPO: &str = "mutable/upstream";',
            ),
        ),
        (
            "tier rehost root",
            source.replace(
                'const CANDLE_MINIMAX_H3_TIER_REPO: &str = "SceneWorks/minimax-h3-mlx";',
                'const CANDLE_MINIMAX_H3_TIER_REPO: &str = "mutable/rehost";',
            ),
        ),
        (
            "manifest override",
            source.replace(
                "let root = candle_minimax_h3_snapshot_dir(settings, CANDLE_MINIMAX_H3_REPO)?;",
                'let root = request.model_manifest_entry.get("repo").unwrap();',
            ),
        ),
    ):
        _must_fail_assertion(
            lambda mutated=mutated: _assert_minimax_h3_candle_uses_exact_installed_roots(mutated),
            f"the MiniMax-H3 Candle audit must reject a {label} mutation",
        )


def test_manifest_model_path_is_only_an_optional_override():
    """sc-14476: converted installs may inject ``modelPath``; normal installs do not.

    Every production reader must therefore inspect it inside an optional branch
    and continue to repo resolution (or decline an imported-only lane) when it
    is absent.
    """
    expected_readers = {
        # sc-16426: gallery attribution reads the imported checkpoint only after the worker has
        # selected an imported Krea/SDXL route, and declines attribution when every path source is
        # absent. It does not affect generation's normal repo resolution.
        "image_jobs.rs",
        "image_jobs/base.rs",
        "image_jobs/flux1_control_candle.rs",
        "image_jobs/flux_ipadapter.rs",
        "image_jobs/instantid.rs",
        "image_jobs/kolors_ipadapter.rs",
        "image_jobs/krea_control_candle.rs",
        "image_jobs/krea_imported.rs",
        # sc-15036: the fine-tuned Mage-Flow base lane. Audited — `modelPath` is an optional
        # FIRST preference here, falling through to the catalog entry's `paths.model` (which is
        # what registration actually stamps) and then declining the lane entirely with `Ok(None)`
        # when neither is present, so a normal install never reaches it.
        "image_jobs/mage_finetuned.rs",
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
            statement_start = source.rfind(
                "let raw_path = request", max(0, match.start() - 500), match.start()
            )
            statement_end = source.find(";", match.end())
            statement = (
                source[statement_start : statement_end + 1]
                if statement_start >= 0 and statement_end >= 0
                else ""
            )
            compact_statement = re.sub(r"\s+", "", statement)
            # sc-16426's attribution-only reader uses idiomatic `?`, which Clippy requires. Keep
            # that exception exact: any new source, precedence change, panic/error conversion, or
            # fallback change must fail this inventory audit and be reviewed explicitly.
            question_mark_optional = relative_path == "image_jobs.rs" and compact_statement == (
                'letraw_path=request.advanced.get("modelPath")'
                '.or_else(||request.model_manifest_entry.get("modelPath"))'
                '.or_else(||{request.model_manifest_entry.get("paths")'
                '.and_then(|paths|paths.get("model"))})'
                ".and_then(Value::as_str)?;"
            )
            assert (
                "if let Some(path) = request" in prefix
                or "let Some(raw_path) = request" in prefix
                or question_mark_optional
            ), (
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


def test_memory_request_provider_mode_schema_admits_public_character_image():
    """SC-20798: provider-owned Character routes keep their public typed coordinate."""
    schema = _load_schema(SCHEMA_PATH)
    provider_modes = []

    def visit(value):
        if isinstance(value, dict):
            properties = value.get("properties", {})
            if "providerMode" in properties:
                provider_modes.append(properties["providerMode"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    assert len(provider_modes) == 1
    assert provider_modes[0]["enum"] == [
        "text_to_image",
        "image_to_image",
        "edit_image",
        "character_image",
    ]


def test_memory_declaration_withhold_is_authorable_on_both_backends():
    """sc-20246: the withhold the declaration projector honors must pass the authoring schema.

    `memoryDeclarationWithhold` is how a backend block records that it declares LESS than
    `config/engine-capabilities/capabilities.<backend>.json` dumps, on purpose, because a measurement
    says the wider claim is wrong. No committed entry needs one today — the dumps happen to agree with
    every recorded verdict — so without this test the property's FIRST real use would red
    `test_builtin_models_manifest_satisfies_authoring_schema` at the moment somebody needed it, which
    is precisely when a red is least useful. Validated against the real manifest plus a synthetic
    withhold rather than against a hand-built document, so the property is exercised where it lives.
    """
    schema = _load_schema(SCHEMA_PATH)
    validator = jsonschema.Draft202012Validator(schema)

    def validate_with(withhold, backend):
        manifest = _load_builtin_models_manifest()
        model = next(
            candidate
            for candidate in manifest["models"]
            if candidate.get(backend) is not None
        )
        model[backend]["memoryDeclarationWithhold"] = withhold
        return sorted(
            validator.iter_errors(manifest), key=lambda error: list(error.absolute_path)
        )

    for backend in ("mlx", "candle"):
        for rungs in (["bounded_decode", "bounded_attention"], "all"):
            errors = validate_with(
                {
                    "rungs": rungs,
                    "story": "SC-15525",
                    "reason": "rung 2 is withheld on quality: the production latent drifts 84/255.",
                },
                backend,
            )
            assert not errors, (backend, rungs, [error.message for error in errors])

    # Uncited or malformed withholds must be REJECTED, matching what `withheldRungs` throws on.
    for bad in (
        {"rungs": ["bounded_decode"], "reason": "no story"},
        {"rungs": ["bounded_decode"], "story": "SC-15525"},
        {"rungs": [], "story": "SC-15525", "reason": "empty"},
        {"rungs": ["not_a_rung"], "story": "SC-15525", "reason": "unknown rung"},
        {"rungs": "everything", "story": "SC-15525", "reason": "not the all literal"},
        {
            "rungs": "all",
            "story": "SC-15525",
            "reason": "extra key",
            "unexpected": True,
        },
    ):
        assert validate_with(bad, "mlx"), bad


def test_sensenova_models_do_not_advertise_lora_compatibility():
    """sc-18476: SenseNova has no diffusion-LoRA merge path, so advertise none."""
    manifest = _load_builtin_models_manifest()
    sensenova = {
        model["id"]: model
        for model in manifest["models"]
        if model.get("family") == "sensenova-u1"
    }
    assert set(sensenova) == {
        "sensenova_u1_8b",
        "sensenova_u1_8b_fast",
        "sensenova_u1_8b_infographic_v2",
        "sensenova_u1_8b_infographic_v2_fast",
        "sensenova_u1_8b_infographic_v3",
        "sensenova_u1_8b_infographic_v3_fast",
    }
    for model_id, model in sensenova.items():
        assert "loraCompatibility" not in model, (
            f"{model_id} must omit the LoRA advertisement; the SenseNova worker accepts no "
            "user adapters"
        )


def test_schema_accepts_mlx_sequential_offload_capability():
    """SC-18377: MLX staged-residency declarations are part of the authoring contract."""
    schema = _load_schema(SCHEMA_PATH)
    validator = jsonschema.Draft202012Validator(schema)
    entry = _model_entry_with_download(
        {"provider": "huggingface", "repo": "namespace/model", "files": []}
    )
    entry["mlx"] = {"supportsSequentialOffload": True}

    errors = list(validator.iter_errors({"schemaVersion": 1, "models": [entry]}))

    assert not errors, [
        (error.validator, list(error.absolute_path), error.message) for error in errors
    ]


def test_cleanup_with_model_is_boolean_and_requires_a_corequisite():
    """SC-18902: only an explicitly exclusive co-requisite may follow a cleanup tombstone."""
    schema = _load_schema(SCHEMA_PATH)
    validator = jsonschema.Draft202012Validator(schema)
    download = {
        "provider": "huggingface",
        "repo": "namespace/exclusive-adapter",
        "coRequisite": True,
        "cleanupWithModel": True,
    }
    valid = _model_entry_with_download(download)
    assert not list(validator.iter_errors({"schemaVersion": 1, "models": [valid]}))

    without_corequisite = _model_entry_with_download(
        {key: value for key, value in download.items() if key != "coRequisite"}
    )
    errors = list(
        validator.iter_errors({"schemaVersion": 1, "models": [without_corequisite]})
    )
    assert any(
        error.validator == "required"
        and "coRequisite" in error.message
        for error in errors
    ), [(error.validator, list(error.absolute_path), error.message) for error in errors]

    wrong_type = _model_entry_with_download({**download, "cleanupWithModel": "yes"})
    errors = list(validator.iter_errors({"schemaVersion": 1, "models": [wrong_type]}))
    assert any(
        error.validator == "type"
        and list(error.absolute_path)[-1:] == ["cleanupWithModel"]
        for error in errors
    ), [(error.validator, list(error.absolute_path), error.message) for error in errors]


def _cleanup_with_model_ownership_errors(manifest: dict) -> list[str]:
    owners_by_repo: dict[str, set[str]] = {}
    cleanup_rows: list[tuple[str, str]] = []
    for model in manifest["models"]:
        for download in model.get("downloads", []):
            repo = download.get("repo")
            if repo:
                owners_by_repo.setdefault(repo, set()).add(model["id"])
            if download.get("cleanupWithModel") is True:
                cleanup_rows.append((model["id"], repo))
    return [
        f"{model_id}:{repo} is referenced by {sorted(owners_by_repo.get(repo, set()))}"
        for model_id, repo in cleanup_rows
        if owners_by_repo.get(repo, set()) != {model_id}
    ]


def test_cleanup_with_model_repositories_are_exclusive_to_one_parent():
    """Deleting a cleanup tombstone may recursively remove only a repo no sibling model uses."""
    manifest = _load_builtin_models_manifest()
    assert not _cleanup_with_model_ownership_errors(manifest)

    mutated = copy.deepcopy(manifest)
    eros = next(model for model in mutated["models"] if model["id"] == "ltx_2_3_eros")
    exclusive = next(
        download for download in eros["downloads"] if download.get("cleanupWithModel") is True
    )
    sibling = next(model for model in mutated["models"] if model["id"] == "ltx_2_3")
    sibling["downloads"].append(
        {
            "provider": exclusive["provider"],
            "repo": exclusive["repo"],
            "files": exclusive.get("files", []),
        }
    )
    errors = _cleanup_with_model_ownership_errors(mutated)
    assert errors and "ltx_2_3" in errors[0] and "ltx_2_3_eros" in errors[0], errors

def test_memory_strategy_overlay_vocabularies_match_runtime_contract():
    """Static capabilities and exact provider contracts share one overlay vocabulary.

    The matrix generator and Candle selector both recognize identity-conditioned cells. Keeping
    the two authoring-schema locations exact prevents an identity-capable manifest from validating
    in one memory-strategy declaration while being rejected in the other.
    """
    schema = _load_schema(SCHEMA_PATH)
    expected = {"none", "lora", "identity", "control"}
    static_overlays = schema["$defs"]["staticMemoryStrategyCapability"]["properties"][
        "overlays"
    ]["items"]["enum"]
    contract_overlays = schema["properties"]["models"]["items"]["properties"]["mlx"][
        "properties"
    ]["memoryStrategyContract"]["properties"]["implementations"]["items"]["properties"][
        "overlays"
    ]["items"]["enum"]
    assert set(static_overlays) == set(contract_overlays) == expected


def test_measured_memory_rows_declare_their_workload_geometry():
    """sc-16020: geometry is data, not a prose assumption.

    Stated as derivations rather than catalog head-counts (sc-20799 round 2): a pinned row count
    only ever records how many rows existed on the day it was written, and every legitimate
    add/remove edits the integer instead of testing anything. What actually has to hold is that the
    populations partition cleanly and that every member of each satisfies its geometry invariant —
    which is what prevents a new row from silently escaping the normalization contract or an
    unmeasured tier gate from presenting itself as a calibrated measurement.
    """
    manifest = _load_builtin_models_manifest()
    mlx_rows = []
    candle_rows = []
    for model in manifest["models"]:
        for download in model.get("downloads", []):
            footprint = download.get("footprint", {})
            if footprint.get("peakMemoryBytes") is not None:
                mlx_rows.append((model["id"], download.get("variant"), footprint))
        candle = model.get("candle", {})
        if "vramGbByTier" in candle:
            candle_rows.append((model["id"], candle))

    # Both populations must exist at all — an `all(...)` over an empty list is vacuously true, and
    # a filter that silently stopped selecting anything is exactly the failure these guards exist
    # to catch. Existence, not cardinality.
    assert mlx_rows, "no download declares a measured peak; the mlx filter selected nothing"
    assert candle_rows, "no model declares vramGbByTier; the candle filter selected nothing"

    # Per-row invariant: every measured MLX peak is priced at the calibrated 1024² workload.
    assert all(row[2].get("measuredPixels") == 1024 * 1024 for row in mlx_rows), mlx_rows
    assert all(isinstance(row[1].get("measured"), bool) for row in candle_rows), candle_rows

    measured_rows = [row for row in candle_rows if row[1]["measured"]]
    unmeasured_rows = [row for row in candle_rows if not row[1]["measured"]]
    # measured ∪ unmeasured == candle_rows, and the two are disjoint. `measured` is asserted to be a
    # bool above, so a row cannot sit outside both; this pins that neither branch drops or
    # double-counts a row, which is the property the two head-counts were standing in for.
    assert measured_rows and unmeasured_rows
    measured_ids = {row[0] for row in measured_rows}
    unmeasured_ids = {row[0] for row in unmeasured_rows}
    assert measured_ids.isdisjoint(unmeasured_ids), measured_ids & unmeasured_ids
    assert measured_ids | unmeasured_ids == {row[0] for row in candle_rows}

    measured_image_rows = [
        row
        for row in measured_rows
        if row[0] not in {"scail2_14b", "minimax_h3", "minimax_h3_ref"}
    ]
    assert all(
        row[1].get("vramMeasuredPixels") == 1024 * 1024
        for row in measured_image_rows
    ), measured_image_rows
    scail = [row for row in measured_rows if row[0] == "scail2_14b"]
    assert len(scail) == 1
    assert scail[0][1].get("vramMeasuredPixels") == 832 * 480
    minimax_h3 = [
        row for row in measured_rows if row[0] in {"minimax_h3", "minimax_h3_ref"}
    ]
    assert {row[0] for row in minimax_h3} == {"minimax_h3", "minimax_h3_ref"}
    assert all(row[1].get("vramMeasuredPixels") == 1344 * 768 for row in minimax_h3)

    # Unmeasured rows still declare the geometry of their estimate or conservative gate, but they
    # do not enter the calibrated 1024² set. FLUX.2-dev is deliberately the sole 256² gate: its
    # durable runs establish a safe high-water, not a 1024² calibration.
    non_calibration_geometry = [
        (model_id, candle["vramMeasuredPixels"])
        for model_id, candle in unmeasured_rows
        if candle["vramMeasuredPixels"] != 1024 * 1024
    ]
    assert non_calibration_geometry == [("flux2_dev", 256 * 256)]
    flux2_dev = next(
        candle for model_id, candle in unmeasured_rows if model_id == "flux2_dev"
    )
    assert flux2_dev["measured"] is False
    assert flux2_dev["vramGbByTier"] == {"q4": 42.7, "q8": 70.8}


def test_scail2_candle_admission_matches_the_validated_shared_package_evidence():
    """sc-20744: terminal receipts promote exact q4/q8 alongside the established bf16 row."""
    manifest = _load_builtin_models_manifest()
    scail = next(model for model in manifest["models"] if model["id"] == "scail2_14b")
    candle = scail["candle"]
    assert candle == {
        "minMemoryGb": 64,
        "vramGbByTier": {"q4": 61.260, "q8": 64.928, "bf16": 102.115},
        "vramMeasuredPixels": 832 * 480,
        "measured": True,
    }
    assert candle["minMemoryGb"] == math.ceil(
        candle["vramGbByTier"]["q4"] + 2
    )
    assert "64 GB for q4, 67 GB for q8, and 105 GB for bf16" in scail["ui"]["description"]

    variants = {download["variant"]: download for download in scail["downloads"]}
    assert set(variants) == {"q4", "q8", "bf16"}
    for tier in ("q4", "q8"):
        assert variants[tier]["files"] == [f"{tier}/*"]
        assert variants[tier]["platforms"] == ["macos", "windows", "linux"]
        assert variants[tier]["revision"] == "ce88cfdb1008f395e9c820e525e6db7b6695f7b3"
    assert variants["bf16"]["platforms"] == ["macos", "windows", "linux"]
    assert variants["q4"]["default"] is True

    raw = MANIFEST_PATH.read_text(encoding="utf-8")
    scail_section = raw.split('"id": "scail2_14b"', 1)[1].split(
        "// Krea Realtime 14B", 1
    )[0]
    for exact_evidence in [
        "3174984b20334bb029170e367be234de0b3f8753",
        "ce88cfdb1008f395e9c820e525e6db7b6695f7b3",
        "31455292141",
        "93667700921",
        "9089420126",
        "de62f67b175ca91519602d5e024baf5342907b9fbe8d1297ad5abb561748bac9",
        "overallPeakGb=102.115",
    ]:
        assert exact_evidence in scail_section


def test_schema_requires_geometry_for_every_peak_memory_evidence_shape():
    """Mutation guard for both lane-owned measurement shapes."""
    schema = _load_schema(SCHEMA_PATH)
    validator = jsonschema.Draft202012Validator(schema)

    mlx_entry = _model_entry_with_download(
        {
            "provider": "huggingface",
            "repo": "namespace/model",
            "files": [],
            "footprint": {
                "diskSizeBytes": 1,
                "peakMemoryBytes": 2,
            },
        }
    )
    mlx_errors = list(
        validator.iter_errors({"schemaVersion": 1, "models": [mlx_entry]})
    )
    assert any(
        error.validator == "required"
        and "measuredPixels" in error.validator_value
        and list(error.absolute_path)[-1:] == ["footprint"]
        for error in mlx_errors
    ), [(error.validator, list(error.absolute_path), error.message) for error in mlx_errors]

    candle_entry = _model_entry_with_download(
        {"provider": "huggingface", "repo": "namespace/model", "files": []}
    )
    candle_entry["candle"] = {"vramGbByTier": {"q4": 12.5}}
    candle_errors = list(
        validator.iter_errors({"schemaVersion": 1, "models": [candle_entry]})
    )
    assert any(
        error.validator == "required"
        and "vramMeasuredPixels" in error.validator_value
        and list(error.absolute_path)[-1:] == ["candle"]
        for error in candle_errors
    ), [(error.validator, list(error.absolute_path), error.message) for error in candle_errors]
    assert any(
        error.validator == "required"
        and "measured" in error.validator_value
        and list(error.absolute_path)[-1:] == ["candle"]
        for error in candle_errors
    ), [(error.validator, list(error.absolute_path), error.message) for error in candle_errors]


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
    # sc-20529: `macOnly` is REMOVED, not flipped. For an image entry the flag is a no-op label
    # (only video entries and the vision captioner read it), and the FLUX.2-klein converter has a
    # real candle twin (sc-7459), so claiming macOS-only contradicted the shipped off-Mac convert
    # lane. Availability is driven by the routing tables, not this flag.
    assert "macOnly" not in model
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
    # SC-18460: the converted transformer is a fixed dense BF16 artifact — there is no packed tier
    # to default to. `resolve_quant` maps `<= 0` to dense/no-quant, and OMITTING the key would
    # default to q8, so 0 is the only correct declaration here. The former `== 8` described a tier
    # this entry never ships.
    assert mlx["quantize"] == 0


def test_flux2_dev_carries_no_inert_mac_only_flag():
    """sc-20530: `flux2_dev` is candle-routed off-Mac (epic 6564) AND MLX on Apple Silicon, so the
    `macOnly: true` it used to carry was inert *and* misleading — the flag is read only by the video
    catalog-withdrawal contract (`video_model_withdrawn_on_platform`, gated on `type == "video"`)
    and by the id-pinned vision captioner in the web eligibility helpers, neither of which sees an
    image entry with this id. Removed, not flipped to false, and pinned absent like the klein pair
    so it cannot creep back (sc-20529 removed `macOnly` from `flux2_klein_9b_true_v2` too — see
    `test_flux2_true_v2_manifest_install_time_conversion` above).
    """
    model = next(
        model
        for model in _load_builtin_models_manifest()["models"]
        if model["id"] == "flux2_dev"
    )
    assert model["type"] == "image"
    assert "macOnly" not in model


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
        # sc-20530: `macOnly` is REMOVED from both entries, not flipped to false. The flag is read
        # only by the video catalog-withdrawal contract (`video_model_withdrawn_on_platform`, which
        # requires `type == "video"`) and by the id-pinned vision captioner in the web eligibility
        # helpers, so on an image entry it never gated anything — it only read like a platform
        # contract these candle-routed entries do not have. Pinned absent so it cannot creep back.
        assert "macOnly" not in model, model_id
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
    """sc-12126/sc-13108/sc-15206/sc-16211: pin peaks and composition-keyed Turbo evidence."""
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
    turbo_fit = krea["candle"]["turboFit"]
    assert {
        key: turbo_fit[key]
        for key in (
            "calibrationAbi",
            "loadShape",
            "calibrationFingerprint",
            "sceneWorksRevision",
            "inferenceRevision",
            "measured",
        )
    } == {
        # sc-17097 re-measured every curve under gen_core::MEMORY_CALIBRATION_ABI 3 on the CUDA box.
        # The stamp is only allowed to move as the RESULT of that measurement, never on its own.
        "calibrationAbi": 3,
        "loadShape": "deferred_materialization",
        "calibrationFingerprint": "krea-turbo-cuda-phase-curves-v1",
        "sceneWorksRevision": "sc-15449-contract-v1",
        "inferenceRevision": "a4f409ae8ce73eda2ee8117b89b5f479666606b8",
        "measured": True,
    }
    assert turbo_fit["strategyParameters"] == {
        "resident": {},
        "threeStage": {},
        "tiledVae": {"decodeTileEdge": 512, "decodeOverlap": 128},
        "chunkedAttention": {
            "decodeTileEdge": 512,
            "decodeOverlap": 128,
            "attentionChunkSize": 134217728,
        },
        "streamedBlocks": {
            "decodeTileEdge": 512,
            "decodeOverlap": 128,
            "attentionChunkSize": 134217728,
            "transformerWindowSize": 1,
        },
    }
    assert set(turbo_fit["verification"]) == {
        "hardware",
        "stories",
        "method",
        "numericPolicy",
        "outputParity",
    }
    assert turbo_fit["verification"]["stories"] == [
        "sc-15117",
        "sc-15205",
        "sc-15206",
        "sc-17097",
    ]
    assert {
        (record["tier"], record["width"], record["height"])
        for record in turbo_fit["evidenceRecords"]
    } == {
        ("q4", 768, 768),
        ("q4", 1024, 1024),
        ("q8", 768, 768),
        ("q8", 1024, 1024),
        ("bf16", 768, 768),
        ("bf16", 1024, 1024),
    }
    for record in turbo_fit["evidenceRecords"]:
        assert set(record["predictedPeaksGb"]) == {
            "threeStage",
            "tiledVae",
            "chunkedAttention",
            "streamedBlocks",
        }
        assert set(record["observedPeaksGb"]) == set(record["predictedPeaksGb"])
        if record["evidenceScope"] == "exact_request":
            assert record["parity"]["result"] == "passed"
        else:
            assert record["evidenceScope"] == "phase_fit_only"
            assert "parity" not in record
            assert set(record["observedPhasesGb"]) == set(record["predictedPeaksGb"])
        assert len(record["sceneWorksCommit"]) == 40
        assert len(record["inferenceCommit"]) == 40
    assert turbo_fit["maxMeasuredPixels"] == 1024 * 1024
    assert set(turbo_fit["phaseCurvesByTier"]) == {"q4", "q8", "bf16"}
    for tier in ("q4", "q8", "bf16"):
        assert set(turbo_fit["phaseCurvesByTier"][tier]) == {
            "threeStage",
            "tiledVae",
            "chunkedAttention",
            "streamedBlocks",
        }
    # sc-17097 ABI-3 re-measurement. Two shape changes worth reading rather than skimming:
    # the three-stage DECODE phase is now the dominant, strongly resolution-dependent term
    # (26.51 + 9.75/MP, against the ABI-1 capture's near-flat 26.47 + 0.08/MP), and the
    # streamed-block decode is no longer the 0.30 + 3.27/MP ramp - it measured flat at 4.48.
    assert turbo_fit["phaseCurvesByTier"]["bf16"] == {
        "threeStage": {
            "text": {"fixedGb": 7.90, "perMpxGb": 0.07},
            "denoise": {"fixedGb": 22.19, "perMpxGb": 7.43},
            "decode": {"fixedGb": 26.51, "perMpxGb": 9.75},
        },
        "tiledVae": {
            "text": {"fixedGb": 8.10, "perMpxGb": 0.00},
            "denoise": {"fixedGb": 22.14, "perMpxGb": 7.36},
            "decode": {"fixedGb": 24.56, "perMpxGb": 0.07},
        },
        "chunkedAttention": {
            "text": {"fixedGb": 8.10, "perMpxGb": 0.00},
            "denoise": {"fixedGb": 25.54, "perMpxGb": 0.21},
            "decode": {"fixedGb": 24.60, "perMpxGb": 0.00},
        },
        "streamedBlocks": {
            "text": {"fixedGb": 8.10, "perMpxGb": 0.00},
            "denoise": {"fixedGb": 7.34, "perMpxGb": 1.03},
            "decode": {"fixedGb": 4.48, "perMpxGb": 0.00},
        },
    }


def _committed_phase_curves(manifest: dict) -> dict:
    """Every ``(tier, rung, phase)`` phase curve committed to the builtin manifest.

    Keyed by a readable coordinate so a failure names the curve rather than an index.
    """
    krea = next(model for model in manifest["models"] if model["id"] == "krea_2_turbo")
    curves = {}
    for tier, rungs in krea["candle"]["turboFit"]["phaseCurvesByTier"].items():
        for rung, phases in rungs.items():
            for phase, curve in phases.items():
                curves[f"{tier}.{rung}.{phase}"] = curve
    return curves


def test_image_lane_phase_curves_carry_no_temporal_coefficient():
    """sc-18812: the whole shipped image lane is still two-coefficient, all 36 curves of it.

    This is the precondition that keeps
    ``test_krea_q8_and_bf16_phase_slopes_are_fitted_from_their_own_two_points`` honest. That test
    reads ``perMpxGb`` and compares it against a measured two-point delta, which is a COMPLETE
    account of the curve only while no temporal term exists. Add ``perMpxFrameGb`` to any of these
    curves and the slope comparison would keep passing while describing a different function.

    The existing literal-dict assertion above covers bf16's 12 curves; this covers all 36, which is
    what "no image curve moved" actually requires.
    """
    curves = _committed_phase_curves(_load_builtin_models_manifest())
    assert len(curves) == 36, (
        f"3 tiers x 4 rungs x 3 phases expected, found {len(curves)} - a changed population "
        "means this guard is covering something other than what it claims"
    )
    for label, curve in curves.items():
        assert set(curve) == {"fixedGb", "perMpxGb"}, (
            f"{label} declares {sorted(curve)}; the image lane must stay two-coefficient so its "
            "fitted slope remains the whole geometry response"
        )


def test_schema_admits_the_temporal_coefficient_additively():
    """sc-18812: the temporal term is OPTIONAL, bounded, and does not disturb existing manifests.

    Three claims, each with its own direction:

    1. The unmodified shipped manifest still validates. That is the migration claim.
    2. A curve that ADDS ``perMpxFrameGb`` validates - so the change is genuinely additive and a
       video curve can ship without a schema bump.
    3. A negative or wrong-typed ``perMpxFrameGb`` is REJECTED, and so is a curve that drops
       ``perMpxGb`` in favour of it - the term extends the area form, it does not replace it.
       (Replacing it is what ``latent_tokens``/``output_voxels`` would have required, which is
       precisely why sc-18812 adopted ``cross`` instead.)
    """
    manifest = _load_builtin_models_manifest()
    schema = _load_schema(SCHEMA_PATH)
    validator = jsonschema.Draft202012Validator(schema)
    krea_index = next(
        index
        for index, model in enumerate(manifest["models"])
        if model["id"] == "krea_2_turbo"
    )
    assert not list(validator.iter_errors(manifest)), "the shipped manifest must still validate"

    def curve_of(candidate):
        return candidate["models"][krea_index]["candle"]["turboFit"]["phaseCurvesByTier"]["q8"][
            "threeStage"
        ]["decode"]

    def mutated(mutate):
        candidate = copy.deepcopy(manifest)
        mutate(candidate)
        return list(validator.iter_errors(candidate))

    assert not mutated(
        lambda candidate: curve_of(candidate).update({"perMpxFrameGb": 0.2998482076533136})
    ), "adding the temporal coefficient must not require a schema bump"
    assert not mutated(
        lambda candidate: candidate["models"][krea_index]["candle"]["turboFit"].update(
            {"maxMeasuredVoxels": 1024 * 1024 * 100}
        )
    ), "declaring the temporal envelope bound must validate"

    for label, mutate in (
        ("negative temporal coefficient", lambda c: curve_of(c).update({"perMpxFrameGb": -0.1})),
        ("non-numeric temporal coefficient", lambda c: curve_of(c).update({"perMpxFrameGb": "0.3"})),
        (
            "temporal coefficient REPLACING the area term",
            lambda c: curve_of(c).clear() or curve_of(c).update(
                {"fixedGb": 2.5, "perMpxFrameGb": 0.3}
            ),
        ),
        ("misspelled temporal coefficient", lambda c: curve_of(c).update({"perMpxFrame": 0.3})),
        (
            "zero temporal envelope bound",
            lambda c: c["models"][krea_index]["candle"]["turboFit"].update(
                {"maxMeasuredVoxels": 0}
            ),
        ),
    ):
        assert mutated(mutate), f"{label} must be rejected by the schema"

    # sc-18812 review pass: an evidence record must be ABLE to state the frame count it was
    # captured at. Without this property the matrix generator would key every record as `WxH` and
    # characterize a video capture as a one-frame design point nobody measured.
    def record_of(candidate):
        return candidate["models"][krea_index]["candle"]["turboFit"]["evidenceRecords"][0]

    assert not mutated(
        lambda candidate: record_of(candidate).update({"frames": 241})
    ), "an evidence record must be able to declare its frame count"
    for label, mutate in (
        ("zero frames", lambda c: record_of(c).update({"frames": 0})),
        ("fractional frames", lambda c: record_of(c).update({"frames": 24.5})),
        ("string frames", lambda c: record_of(c).update({"frames": "241"})),
        ("misspelled frames", lambda c: record_of(c).update({"frameCount": 241})),
    ):
        assert mutated(mutate), f"{label} must be rejected by the schema"


def _tiers_declaring_a_temporal_curve(turbo_fit: dict) -> set:
    """Tiers whose committed phase curves carry ``perMpxFrameGb`` on any rung or phase."""
    declaring = set()
    for tier, rungs in turbo_fit.get("phaseCurvesByTier", {}).items():
        for phases in rungs.values():
            if any("perMpxFrameGb" in curve for curve in phases.values()):
                declaring.add(tier)
    return declaring


def _evidence_records_missing_frames(turbo_fit: dict) -> list:
    """Evidence records that support a TEMPORAL curve without saying how many frames they measured.

    This is the fabricated-rank hazard, named: ``generate-memory-matrix.mjs#measuredGeometryKey``
    reads an absent ``frames`` as 1, which is correct for the image lane and a silent invention on
    a tier whose curve has a temporal coefficient to determine.
    """
    declaring = _tiers_declaring_a_temporal_curve(turbo_fit)
    return [
        f"{record['tier']} {record['width']}x{record['height']} "
        f"({record['sourceStory']} activity {record['sourceActivity']})"
        for record in turbo_fit.get("evidenceRecords", [])
        if record["tier"] in declaring and "frames" not in record
    ]


def test_temporal_curve_tiers_declare_frames_on_their_evidence_records():
    """sc-18812: a tier carrying ``perMpxFrameGb`` may not have frame-silent evidence.

    Inert on the shipped manifest — no committed curve declares the temporal term — so the guard is
    exercised in BOTH directions against mutated manifests rather than asserted vacuously. Without
    it, adding a temporal coefficient in one place and a video capture in another would silently
    characterize that capture as a one-frame design point: a fabricated contribution to the design
    matrix rank, which over-claims harder than the axis collapse sc-18812 set out to fix.
    """
    manifest = _load_builtin_models_manifest()
    krea = next(model for model in manifest["models"] if model["id"] == "krea_2_turbo")
    turbo_fit = krea["candle"]["turboFit"]

    # The live guard. Trivially satisfied while the image lane stays two-coefficient (which
    # ``test_image_lane_phase_curves_carry_no_temporal_coefficient`` is what enforces), and the
    # first thing to fire on the day a temporal curve ships without its evidence catching up.
    assert _evidence_records_missing_frames(turbo_fit) == []

    # Direction 1: declare the term, change nothing else. Every record of that tier is now a
    # frame-silent supporter of a temporal curve, and the audit must name them.
    declared = copy.deepcopy(turbo_fit)
    declared["phaseCurvesByTier"]["q8"]["threeStage"]["decode"]["perMpxFrameGb"] = 0.2998
    assert _tiers_declaring_a_temporal_curve(declared) == {"q8"}
    offenders = _evidence_records_missing_frames(declared)
    assert offenders, "a temporal curve with frame-silent evidence must be rejected"
    assert all(entry.startswith("q8 ") for entry in offenders), offenders
    assert len(offenders) == sum(
        1 for record in turbo_fit["evidenceRecords"] if record["tier"] == "q8"
    ), "every q8 record supports the q8 curve, so every one of them must be named"

    # Direction 2: the same manifest with the frame counts stated is accepted. So the rejection
    # above is attributable to the missing property and not to the declaration.
    repaired = copy.deepcopy(declared)
    for record in repaired["evidenceRecords"]:
        if record["tier"] == "q8":
            record["frames"] = 1
    assert _evidence_records_missing_frames(repaired) == []

    # ...and it is per TIER, not global: q4's records stay frame-silent and stay legal, because no
    # q4 curve has a temporal coefficient to determine.
    assert all(
        "frames" not in record
        for record in repaired["evidenceRecords"]
        if record["tier"] == "q4"
    )


def test_krea_q8_and_bf16_phase_slopes_are_fitted_from_their_own_two_points():
    """sc-16514: equal cross-tier slopes are allowed only when same-tier deltas prove them.

    sc-18812 precondition: this reads ``perMpxGb`` as the whole geometry response, which holds
    only while no curve carries a temporal term. ``test_image_lane_phase_curves_carry_no_temporal
    _coefficient`` is what enforces that, and it must stay green for this comparison to mean
    anything.
    """
    manifest = _load_builtin_models_manifest()
    krea = next(model for model in manifest["models"] if model["id"] == "krea_2_turbo")
    curves = krea["candle"]["turboFit"]["phaseCurvesByTier"]
    records = krea["candle"]["turboFit"]["evidenceRecords"]
    measured = {
        tier: {
            rung: tuple(
                next(
                    record["observedPhasesGb"][rung]
                    for record in records
                    if record["tier"] == tier and record["width"] == edge
                )
                for edge in (768, 1024)
            )
            for rung in ("threeStage", "tiledVae", "chunkedAttention", "streamedBlocks")
        }
        for tier in ("q8", "bf16")
    }
    megapixel_delta = (1024**2 - 768**2) / 1_000_000

    for tier, rungs in measured.items():
        for rung, (lower, upper) in rungs.items():
            for phase in ("text", "denoise", "decode"):
                measured_slope = max(
                    0.0,
                    (upper[phase] - lower[phase]) / megapixel_delta,
                )
                fitted_slope = curves[tier][rung][phase]["perMpxGb"]
                assert measured_slope <= fitted_slope + 1e-9
                assert fitted_slope < measured_slope + 0.02, (
                    f"{tier}.{rung}.{phase} slope {fitted_slope:.4f} must be the "
                    f"conservative two-decimal fit of same-tier measured slope "
                    f"{measured_slope:.4f}"
                )


def test_krea_turbo_fit_schema_rejects_stale_or_incomplete_contract_evidence():
    """sc-15449: calibrated optimization evidence stays closed and revision-bound."""
    manifest = _load_builtin_models_manifest()
    schema = _load_schema(SCHEMA_PATH)
    validator = jsonschema.Draft202012Validator(schema)
    krea_index = next(
        index
        for index, model in enumerate(manifest["models"])
        if model["id"] == "krea_2_turbo"
    )

    def assert_rejected(label, mutate):
        candidate = copy.deepcopy(manifest)
        turbo_fit = candidate["models"][krea_index]["candle"]["turboFit"]
        mutate(turbo_fit)
        assert list(validator.iter_errors(candidate)), label

    assert_rejected("unknown calibration ABI", lambda fit: fit.__setitem__("calibrationAbi", 2))
    assert_rejected(
        "superseded calibration ABI",
        lambda fit: fit.__setitem__("calibrationAbi", 1),
    )
    assert_rejected("missing load shape", lambda fit: fit.pop("loadShape"))
    assert_rejected(
        "unknown load shape",
        lambda fit: fit.__setitem__("loadShape", "lazy_materialization"),
    )
    assert_rejected(
        "malformed calibration fingerprint",
        lambda fit: fit.__setitem__("calibrationFingerprint", "Krea Turbo"),
    )
    assert_rejected(
        "stale SceneWorks contract",
        lambda fit: fit.__setitem__("sceneWorksRevision", "sc-15449-contract-v0"),
    )
    assert_rejected(
        "mutable inference revision",
        lambda fit: fit.__setitem__("inferenceRevision", "main"),
    )
    assert_rejected("estimated evidence", lambda fit: fit.__setitem__("measured", False))
    assert_rejected(
        "missing output parity evidence",
        lambda fit: fit["verification"].pop("outputParity"),
    )
    assert_rejected(
        "unknown verification evidence",
        lambda fit: fit["verification"].__setitem__("notes", "unchecked"),
    )
    assert_rejected(
        "incomplete cumulative parameters",
        lambda fit: fit["strategyParameters"]["chunkedAttention"].pop("decodeOverlap"),
    )
    assert_rejected(
        "resident optimization parameter",
        lambda fit: fit["strategyParameters"]["resident"].__setitem__("window", 1),
    )
    assert_rejected(
        "missing exact evidence records",
        lambda fit: fit.pop("evidenceRecords"),
    )
    assert_rejected(
        "evidence without observed peak",
        lambda fit: fit["evidenceRecords"][0]["observedPeaksGb"].pop("streamedBlocks"),
    )
    assert_rejected(
        "mutable artifact revision",
        lambda fit: fit["evidenceRecords"][0].__setitem__("sceneWorksCommit", "main"),
    )
    assert_rejected(
        "unexecuted parity",
        lambda fit: fit["evidenceRecords"][0]["parity"].__setitem__("result", "not_run"),
    )
    assert_rejected(
        "exact request without geometry-specific parity",
        lambda fit: fit["evidenceRecords"][0].pop("parity"),
    )
    assert_rejected(
        "phase-fit record pretending tier parity is geometry-specific",
        lambda fit: fit["evidenceRecords"][2].__setitem__(
            "parity", copy.deepcopy(fit["evidenceRecords"][3]["parity"])
        ),
    )


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
        emulation (64 GiB balloon -> ~31 GiB free) reproduced the SAME ~28 live peak with no apparent
        spill, so q8 is gated at its live peak and ADMITS a 32 GB RTX 5090 -- the epic goal.
      * bf16 was staged (dense fp32 diffusers, after downloading the missing transformer_2 shards) and
        measured at ~39 GiB (one bf16 expert + activations), REPLACING the old conservative derived 56
        bound: the real number admits a 48 GB card but stays refused on 32.

    WARNING -- the q8/bf16 <=32/<=48 admissions are UNPROVEN (sc-16091 -> sc-16118). The balloon argument
    behind them is circular: `balloon(64) + live(28) = 92 < 96` substitutes the LIVE figure for the pool
    high-water, which is only valid if the pool trimmed -- the very claim under test. Under the competing
    hypothesis it reads 64 + 34.4 = 98.4 > 95.6, a ~2.8 GiB overcommit. And "a spill would inflate wall
    time" is measured FALSE on that host (sc-15791): a 1.48 GiB overcommit completed at 1.07x, faster in a
    sibling run. Both hypotheses predict what was observed, since USED_MEM_HIGH reports LIVE bytes either
    way. bf16 has no independent evidence at all -- it was granted 48 on the same q8 inference.

    The values below are therefore pinned as SHIPPED, not as VERIFIED. This test guards against silent
    drift; it does not certify the small-card fit. sc-16118 re-validates both under an ENFORCED pool cap
    (`CUmemPoolProps.maxSize` + `cuDeviceSetMemPool`) instead of a balloon, which is not a ceiling on that
    hardware. If a tier fails there, these numbers change and this docstring's premise goes with them.

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


def test_lora_schema_accepts_a_declared_model_id_list():
    """sc-19563: the entry is additionalProperties:false, so a per-model-id key is an
    explicit schema addition rather than a free-form one. Pin that it is accepted at
    all — reverting the property would turn every shipped `modelIds` into an
    authoring error."""
    entry = _sample_lora_entry()
    entry["modelIds"] = ["minimax_h3_ref"]
    errors = _schema_errors({"schemaVersion": 1, "loras": [entry]}, LORA_SCHEMA_PATH)
    assert not errors, "a declared modelIds list must be schema-valid:\n" + _format_errors(errors)


def test_lora_schema_rejects_a_malformed_model_id_list():
    """`modelIds` is typed, not free-form: a bare string, an empty list, a non-string
    element and an empty id are each rejected. Without the typing an author could
    write `"modelIds": "minimax_h3_ref"` and the gate would read nothing — the key
    present, the constraint absent, which is worse than declaring none at all."""
    for value, keyword in (
        ("minimax_h3_ref", "type"),
        ([], "minItems"),
        ([123], "type"),
        ([""], "minLength"),
    ):
        entry = _sample_lora_entry()
        entry["modelIds"] = value
        errors = _schema_errors({"schemaVersion": 1, "loras": [entry]}, LORA_SCHEMA_PATH)
        assert any(
            error.validator == keyword
            and list(error.absolute_path)[:3] == ["loras", 0, "modelIds"]
            for error in errors
        ), f"modelIds={value!r} must be rejected by `{keyword}`; got {_format_errors(errors)}"


def test_lora_schema_accepts_a_declared_sampling_recipe():
    """sc-18726: `sampling` is an explicit schema addition on an
    additionalProperties:false entry, so pin that a well-formed block is accepted at
    all — reverting the property would turn every shipped turbo entry into an
    authoring error."""
    entry = _sample_lora_entry()
    entry["role"] = "accelerator"
    entry["sampling"] = {"steps": 4, "schedulerShift": 6.0, "audioSchedulerShift": 3.0}
    errors = _schema_errors({"schemaVersion": 1, "loras": [entry]}, LORA_SCHEMA_PATH)
    assert not errors, "a declared sampling recipe must be schema-valid:\n" + _format_errors(errors)


def test_lora_schema_rejects_a_malformed_sampling_recipe():
    """The negative arm, and the one that carries the weight (sc-18726).

    `parse_turbo_recipes` DROPS a block it cannot read rather than failing — a
    silently recipe-less accelerator renders 50 steps at the base shift, which is the
    2 h 25 m render this whole feature exists to avoid, with no error anywhere. So
    every way of writing the block wrong has to be an authoring-time red: a partial
    block (each of the three keys is load-bearing and none has a safe default), an
    out-of-band step count, a zero shift, a string where a number belongs, a typo'd
    key, and a non-object.
    """
    good = {"steps": 4, "schedulerShift": 6.0, "audioSchedulerShift": 3.0}
    cases = [
        ({key: value for key, value in good.items() if key != missing}, "required")
        for missing in good
    ]
    cases += [
        ({**good, "steps": 0}, "minimum"),
        ({**good, "steps": 4.5}, "type"),
        ({**good, "schedulerShift": 0}, "exclusiveMinimum"),
        ({**good, "audioSchedulerShift": "3.0"}, "type"),
        # The sc-12288 field class, one level in: a typo'd key is silently ignored by a
        # permissive object, and `required` alone would not catch a MISSPELLED extra.
        ({**good, "schedulerShifts": 6.0}, "additionalProperties"),
        (4, "type"),
    ]
    for value, keyword in cases:
        entry = _sample_lora_entry()
        entry["role"] = "accelerator"
        entry["sampling"] = value
        errors = _schema_errors({"schemaVersion": 1, "loras": [entry]}, LORA_SCHEMA_PATH)
        assert any(
            error.validator == keyword
            and list(error.absolute_path)[:3] == ["loras", 0, "sampling"]
            for error in errors
        ), f"sampling={value!r} must be rejected by `{keyword}`; got {_format_errors(errors)}"


def test_every_minimax_h3_lora_declares_its_partition_in_the_real_catalog():
    """sc-19563, against the SHIPPED manifest rather than a sample entry.

    Both H3 partitions are one architecture and declare one family, so family
    membership cannot express which of `minimax_h3` / `minimax_h3_ref` an adapter is
    distilled for; cross-selecting used to fold cleanly at the wrong quality.

    The final inequality is the load-bearing half: a catalog that gave all four the
    same `modelIds` would pass a presence check and enforce nothing."""
    manifest = _load_jsonc(LORA_MANIFEST_PATH)
    declared = {
        entry["id"]: entry.get("modelIds")
        for entry in manifest["loras"]
        if entry.get("family") == "minimax-h3"
    }
    assert declared, "no minimax-h3 LoRAs found; this guard would be vacuous"
    for lora_id, model_ids in declared.items():
        assert model_ids, f"{lora_id} declares no modelIds (sc-19563)"
    assert declared["minimax_h3_ref2v_turbo_4step"] == ["minimax_h3_ref"]
    fl2v = [
        ids for lora_id, ids in declared.items() if lora_id != "minimax_h3_ref2v_turbo_4step"
    ]
    assert all(
        ids == ["minimax_h3"] for ids in fl2v
    ), f"the fl2v adapters must name minimax_h3; got {fl2v}"
    assert declared["minimax_h3_ref2v_turbo_4step"] != fl2v[0], (
        "the ref2v and fl2v adapters must name DIFFERENT partitions, or the declaration "
        "enforces nothing"
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


# --------------------------------------------------------------------------------------
# sc-15299 — the `image` capability sub-block (Guidance / Negative prompt axes).
#
# The image-lane sibling of the `video` block sc-8445 shipped for Krea Realtime. Image Studio
# reads `image.supportsGuidance` / `image.supportsNegativePrompt` with ABSENT-MEANS-TRUE polarity
# (the opposite of `audio`), so these audits pin BOTH directions: the declarations that must be
# present, and the entries that must stay silent so the change is behaviour-neutral for them.
#
# Ground truth is the engine descriptor each model resolves to through
# `crates/sceneworks-worker/src/engines.rs` MODEL_TABLE, which is exactly what the worker's
# `resolve_guidance` / `resolve_negative_prompt` / `resolve_true_cfg` gate on.
# --------------------------------------------------------------------------------------

# Both axes absent: the engine descriptor is supports_guidance=false AND
# supports_negative_prompt=false, so `resolve_guidance`, `resolve_true_cfg` and
# `resolve_negative_prompt` ALL return None. Every one is a guidance-distilled student.
IMAGE_MODELS_WITHOUT_EITHER_AXIS = frozenset(
    {
        "z_image_turbo",
        "z_image_edit",  # runs on the z_image_turbo ENGINE
        "flux_schnell",
        "ideogram_4_turbo",
        "boogu_image_turbo",
        "krea_2_turbo",
        "sd3_5_large_turbo",
        "anima_turbo",
        "mage_flow_turbo",
        "mage_flow_edit_turbo",
    }
)

# Guidance is real (embedded/distilled scale, or a bespoke lane that forwards one) but there is no
# unconditional branch for a negative prompt to steer, so `resolve_negative_prompt` returns None.
IMAGE_MODELS_WITHOUT_NEGATIVE_ONLY = frozenset(
    {
        "flux_dev",
        "ideogram_4",
        "boogu_image",
        "boogu_image_edit",
        "flux2_dev",
        "sensenova_u1_8b",
        "sensenova_u1_8b_infographic_v2",
        "sensenova_u1_8b_infographic_v3",
        "sensenova_u1_8b_fast",
        "sensenova_u1_8b_infographic_v2_fast",
        "sensenova_u1_8b_infographic_v3_fast",
        "sana_sprint_1600m",
        "pulid_flux_dev",  # image_jobs/pulid.rs hard-sets negative_prompt: None
    }
)


def _image_models_by_id() -> dict:
    return {
        model["id"]: model
        for model in _load_builtin_models_manifest()["models"]
        if model.get("type") == "image"
    }


def test_cfg_free_image_models_declare_both_axes_absent():
    """sc-15299: a guidance-distilled image engine declares BOTH axes false so Image Studio
    hides Guidance and Negative prompt instead of offering knobs the worker discards."""
    models = _image_models_by_id()
    missing = sorted(IMAGE_MODELS_WITHOUT_EITHER_AXIS - set(models))
    assert not missing, f"CFG-free ids no longer in the catalog: {missing}"
    for model_id in sorted(IMAGE_MODELS_WITHOUT_EITHER_AXIS):
        block = models[model_id].get("image")
        assert block is not None, f"{model_id} is CFG-free but declares no `image` block"
        assert block.get("supportsGuidance") is False, (
            f"{model_id} is CFG-free — its engine descriptor advertises supports_guidance=false "
            "and it is not a true-CFG family, so no guidance scale reaches the engine"
        )
        assert block.get("supportsNegativePrompt") is False, (
            f"{model_id} is CFG-free — resolve_negative_prompt returns None for it"
        )


def test_negative_only_image_models_keep_their_guidance_axis():
    """sc-15299: the two keys are INDEPENDENT. An embedded-guidance engine (FLUX dev, Ideogram 4,
    SenseNova, SANA-Sprint, …) takes a real guidance scale and no negative prompt, so it must
    declare ONLY `supportsNegativePrompt: false` — declaring guidance false too would hide a live
    control."""
    models = _image_models_by_id()
    missing = sorted(IMAGE_MODELS_WITHOUT_NEGATIVE_ONLY - set(models))
    assert not missing, f"negative-free ids no longer in the catalog: {missing}"
    for model_id in sorted(IMAGE_MODELS_WITHOUT_NEGATIVE_ONLY):
        block = models[model_id].get("image")
        assert block is not None, f"{model_id} takes no negative prompt but declares no `image` block"
        assert block.get("supportsNegativePrompt") is False, f"{model_id} must declare the negative axis absent"
        assert "supportsGuidance" not in block, (
            f"{model_id} DOES take a guidance scale — declaring supportsGuidance would hide a live control"
        )


def test_guidance_taking_image_models_declare_nothing():
    """sc-15299 polarity guard: absent means TRUE, so every image entry that genuinely takes both
    axes must stay silent. Includes the true-CFG Chroma family, whose descriptor reads
    supports_guidance=false but whose Guidance control IS live — the worker forwards
    `advanced.guidanceScale` as `true_cfg` (base.rs `uses_true_cfg`/`resolve_true_cfg`). Declaring
    an `image` block for Chroma would break a working knob."""
    models = _image_models_by_id()
    declared = {model_id for model_id, model in models.items() if model.get("image") is not None}
    expected = IMAGE_MODELS_WITHOUT_EITHER_AXIS | IMAGE_MODELS_WITHOUT_NEGATIVE_ONLY
    assert declared == expected, (
        "unexpected `image` declarations — every other image entry must stay silent so it keeps "
        f"both controls: unexpected={sorted(declared - expected)}, missing={sorted(expected - declared)}"
    )
    for model_id in ("chroma1_hd", "chroma1_base", "chroma1_flash", "sana_1600m", "krea_2_raw", "sdxl"):
        assert models[model_id].get("image") is None, (
            f"{model_id} takes both axes (Chroma via true_cfg) and must declare no `image` block"
        )


def test_cfg_free_image_models_carry_no_default_negative_prompt():
    """sc-15299: a model with no negative axis must not seed one — the box is hidden and the value
    is never sent, so `ui.defaultNegativePrompt` would plant a ghost. Mirrors the krea_2_turbo rule
    already pinned in crates/sceneworks-core/src/builtin_manifests.rs."""
    models = _image_models_by_id()
    for model_id in sorted(IMAGE_MODELS_WITHOUT_EITHER_AXIS | IMAGE_MODELS_WITHOUT_NEGATIVE_ONLY):
        assert not models[model_id].get("ui", {}).get("defaultNegativePrompt"), (
            f"{model_id} declares no negative-prompt axis, so it must not declare a default negative"
        )


def test_schema_rejects_an_unknown_key_inside_the_image_block():
    """The `image` object is additionalProperties:false like its `video` sibling."""
    manifest = _load_builtin_models_manifest()
    target = next(model for model in manifest["models"] if model["id"] == "z_image_turbo")
    target["image"]["supportsGuidnce"] = False
    schema = _load_schema(SCHEMA_PATH)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    assert any("supportsGuidnce" in error.message for error in errors)


# --------------------------------------------------------------------------------------
# sc-15299 audio half — `audio.supportsGuidance` / `audio.supportsNegativePrompt` were READ by
# AudioStudio.jsx but never PRODUCIBLE: the `audio` object is additionalProperties:false and the
# schema listed neither key, so the Music tab's Guidance/Negative controls were hidden by accident
# rather than by declaration. The schema now lists them and ACE-Step declares them.
# --------------------------------------------------------------------------------------


def test_audio_block_can_declare_the_guidance_axes():
    """Mutation guard: drop either property from the schema and the shipped manifest stops
    validating — which is exactly the state this story fixed."""
    schema = _load_schema(SCHEMA_PATH)
    audio_properties = schema["properties"]["models"]["items"]["properties"]["audio"]["properties"]
    for key in ("supportsGuidance", "supportsNegativePrompt"):
        assert audio_properties[key]["type"] == "boolean", f"audio.{key} must be declarable"


def test_acestep_declares_its_distilled_guidance_axes_explicitly():
    """ACE-Step v1.5 Turbo is guidance-distilled (candle-audio-acestep's descriptor is
    supports_guidance=false / supports_negative_prompt=false, and its generate path never reads
    either). AUDIO polarity is absent-means-FALSE, so this is UI-neutral — it makes the hiding
    intentional and lets a future non-distilled music model turn the controls on."""
    models = {model["id"]: model for model in _load_builtin_models_manifest()["models"]}
    audio_block = models["acestep_v15_turbo"]["audio"]
    assert audio_block["supportsGuidance"] is False
    assert audio_block["supportsNegativePrompt"] is False


# --------------------------------------------------------------------------------------
# sc-17227 — MiniMax-H3 downstream-user licensing. The MiniMax H3 Community License grants a
# NON-TRANSFERABLE licence (§II) and defines "Licensee" as whoever uses the Works (§I.9), so a
# SceneWorks user is a Licensee in their own right; §V.2 obliges us to notify each user that the
# §V / Exhibit A restrictions apply BEFORE providing access. `MiniMaxAI/MiniMax-H3` is a PUBLIC
# repo, so the pre-existing acknowledgment gate — which keyed off `gated`, i.e. off "needs a
# Hugging Face credential" — could not express that: declaring `gated` would demand a token that
# does not exist, and not declaring it left no gate at all.
# --------------------------------------------------------------------------------------

def _huggingface_repo_key(repo):
    """Canonicalize one repo string the way the API's `huggingface_repo_key` does: lower-cased,
    trailing slash and `.git` stripped. `None` for anything that is not a non-empty string."""
    if not isinstance(repo, str):
        return None
    repo = repo.strip().rstrip("/").strip()
    if repo.lower().endswith(".git"):
        repo = repo[: -len(".git")].rstrip("/").strip()
    return repo.lower() or None


def _declared_download_repos(model):
    """Every Hugging Face repo `model`'s `downloads` rows name in their `repo` field, canonicalized.

    Mirrors the MANIFEST-row read `license_acknowledgment_repo_index` performs — which is
    `download.get("repo")` and nothing else (apps/rust-api/src/models.rs). It is deliberately NOT
    `LICENSE_GATED_REPO_PAYLOAD_KEYS`: that constant is a JOB-PAYLOAD key list, applied by
    `ensure_job_payload_license_acknowledged` to the object a client POSTs, and a manifest download
    row has no `baseRepo`/`sourceRepo` for the index to read. Applying the payload list here
    described an operation the Rust index never performs.
    """
    repos = set()
    for download in model.get("downloads", []):
        key = _huggingface_repo_key(download.get("repo"))
        if key:
            repos.add(key)
    return repos


def _license_acknowledgment_repo_index(models):
    """`owner/name` -> the id of the entry that declares it, for every repo a
    `requiresLicenseAcknowledgment` entry names. Mirrors `license_acknowledgment_repo_index` in
    apps/rust-api/src/models.rs, which is the predicate the running gate uses."""
    index = {}
    for model in models:
        if model.get("requiresLicenseAcknowledgment") is not True:
            continue
        for repo in _declared_download_repos(model):
            index.setdefault(repo, model["id"])
    return index


def _license_acknowledgment_models():
    """Every catalog entry that must carry the acknowledgment contract, DERIVED from the flag
    rather than listed.

    This replaced a hard-coded `MINIMAX_H3_IDS = ("minimax_h3", "minimax_h3_ref")`. A tuple of ids
    inside an audit whose job is to catch entries nobody remembered to list is self-defeating: a
    new entry simply is not in the tuple, every loop below skips it, and CI stays green. No entry
    ids are maintained here — add a flagged entry to the manifest and it is audited on the next run.

    What this set may be asserted on is the FAMILY-AGNOSTIC half of the contract, and only that.
    The MiniMax-specific copy assertions live under `_minimax_h3_license_models()` below: a review
    (sc-17227) caught them being made against every flagged entry, which would have turned three
    audits red for the first non-MiniMax entry to carry the flag, for reasons having nothing to do
    with it. Deriving the SET from the flag and then asserting one family's prose over it trades
    maintained ids for maintained copy; it does not remove the coupling.
    """
    models = _load_builtin_models_manifest()["models"]
    flagged = {
        model["id"]: model
        for model in models
        if model.get("requiresLicenseAcknowledgment") is True
    }
    # Without this the loops below iterate an empty dict and pass vacuously — the failure mode a
    # derived set has and a literal tuple does not.
    assert flagged, "no catalog entry declares requiresLicenseAcknowledgment"
    return flagged


# The one token that scopes the MiniMax-specific copy audits. It is a FAMILY, not a list of entry
# ids: a new MiniMax-H3 partition inherits the copy contract automatically, and an entry of any
# other family is out of scope by construction rather than by being forgotten.
MINIMAX_H3_FAMILY = "minimax-h3"


def _minimax_h3_license_models():
    """The flagged entries of the MiniMax-H3 family — the scope for assertions about MiniMax's own
    licence text, attribution string and licence URL."""
    minimax = {
        model_id: entry
        for model_id, entry in _license_acknowledgment_models().items()
        if entry.get("family") == MINIMAX_H3_FAMILY
    }
    assert minimax, f"no flagged entry is in the {MINIMAX_H3_FAMILY} family"
    return minimax


def _builtin_lora_source_repos():
    """`lora id` -> the canonicalized Hugging Face repo its download resolves to.

    Mirrors what `create_lora_download_job` reads (apps/rust-api/src/loras.rs): `source.repo`, or a
    top-level `repo` when the entry is written flat.
    """
    repos = {}
    for lora in _load_jsonc(LORA_MANIFEST_PATH)["loras"]:
        source = lora.get("source") or {}
        key = _huggingface_repo_key(source.get("repo") or lora.get("repo"))
        if key:
            repos[lora["id"]] = key
    return repos


def test_every_entry_naming_a_license_gated_repo_carries_the_flag_itself():
    """An entry that names a restricted repo in its `downloads` but does not itself declare
    `requiresLicenseAcknowledgment` is a second door onto the same weights.

    `POST /api/v1/models/:id/download` gated on the entry the PATH id names, so such an entry
    downloaded those weights ungated while `POST /api/v1/jobs` naming the same repo answered 403.
    The route now also consults the repo index (sc-17227), so this is no longer the only thing
    standing between that shape and the weights — but the manifest is where the shape is authored,
    and a shared restricted repo across two entries is one authoring decision away: the manifest
    already uses co-requisite rows naming a shared repo for several families.
    """
    models = _load_builtin_models_manifest()["models"]
    index = _license_acknowledgment_repo_index(models)
    assert index, "no repo is licence-gated; this guard would pass vacuously"

    offenders = {}
    for model in models:
        if model.get("requiresLicenseAcknowledgment") is True:
            continue
        shared = sorted(_declared_download_repos(model) & index.keys())
        if shared:
            offenders[model["id"]] = [(repo, index[repo]) for repo in shared]

    assert not offenders, (
        "these entries name a licence-gated repo without declaring "
        f"requiresLicenseAcknowledgment themselves: {offenders}"
    )


def _lora_license_remedy_offenders(lora_repos, models):
    """LoRA ids whose download is licence-gated but whose gate has no way to be CLEARED.

    A catalog LoRA naming a licence-gated repo is refused by `create_lora_download_job` until the
    caller asserts the acknowledgment. The only surface that can make that assertion is the MODEL
    card of the entry the repo index maps the repo to — a LoRA row carries no licence copy and no
    checkbox — so the remedy exists only if that entry can actually render the acceptance, i.e. it
    is in this catalog and carries the licence text and link the card shows. Where it cannot, the
    LoRA is un-downloadable through every shipped surface, which is the regression this guard
    exists to catch.
    """
    index = _license_acknowledgment_repo_index(models)
    by_id = {model["id"]: model for model in models}
    offenders = {}
    for lora_id, repo in sorted(lora_repos.items()):
        model_id = index.get(repo)
        if model_id is None:
            continue
        entry = by_id.get(model_id)
        missing = [
            field
            for field in ("licenseUrl", "licenseNotice")
            if not isinstance((entry or {}).get(field), str) or not (entry or {})[field].strip()
        ]
        if entry is None or missing:
            offenders[lora_id] = (repo, model_id, missing or ["<entry absent>"])
    return offenders


def test_every_builtin_lora_naming_a_license_gated_repo_has_a_reachable_remedy():
    """The LoRA half of the gate, which nothing scanned before (sc-17227 review MAJOR 1).

    `POST /api/v1/loras/:id/download` is repo-keyed like every other door, so a catalog LoRA whose
    `source.repo` is a repo a `requiresLicenseAcknowledgment` model declares is refused 403 without
    the assertion. `builtin.loras.jsonc` was outside every audit here — the model-side guard above
    iterates `_load_builtin_models_manifest()["models"]` only — so nothing checked that such a LoRA
    is still downloadable by a user who accepts the licence.
    """
    models = _load_builtin_models_manifest()["models"]
    index = _license_acknowledgment_repo_index(models)
    assert index, "no repo is licence-gated; this guard would pass vacuously"

    offenders = _lora_license_remedy_offenders(_builtin_lora_source_repos(), models)
    assert not offenders, (
        "these built-in LoRAs fetch a licence-gated repo with no model card able to take the "
        f"acknowledgment that clears the refusal: {offenders}"
    )

    # POSITIVE CONTROL, in the same test: the shipped LoRA catalog names no gated repo today, so
    # the assertion above passes vacuously on its own. Drive the same predicate over a SYNTHETIC
    # catalog to prove it intersects `source.repo` against the index at all, and that it reports the
    # unclearable case rather than only the clearable one.
    gated_repo, gating_model_id = sorted(index.items())[0]
    synthetic = {"synthetic_gated_lora": gated_repo, "synthetic_plain_lora": "owner/not-gated"}
    assert not _lora_license_remedy_offenders(synthetic, models), (
        "a LoRA naming a gated repo whose model carries the licence copy has a remedy and must "
        "not be reported"
    )

    stripped = [
        {key: value for key, value in model.items() if key != "licenseNotice"}
        if model["id"] == gating_model_id
        else model
        for model in models
    ]
    assert _lora_license_remedy_offenders(synthetic, stripped) == {
        "synthetic_gated_lora": (gated_repo, gating_model_id, ["licenseNotice"])
    }, "the guard must catch a gated LoRA whose gating model cannot render the acceptance"


def test_every_license_acknowledgment_entry_declares_the_credential_free_shape():
    """The FAMILY-AGNOSTIC half of the acknowledgment contract, asserted over every flagged entry.

    `requiresLicenseAcknowledgment` exists to express "the licence binds the user" WITHOUT
    "a Hugging Face credential is needed" — the two used to be one flag. So an entry that raises
    this gate must not also claim the credential shape: `gated` would make the Models screen render
    "Add token in Settings" and "Request access on Hugging Face" for a credential and an access page
    that need not exist, and `credentialHost` is the field that drives that UI.

    It must also carry the copy the gate SHOWS. A gate whose card has no licence text is a checkbox
    over nothing, and it is what the LoRA-side remedy resolves to as well
    (`_lora_license_remedy_offenders`). What that text has to SAY is family-specific and is
    asserted per family below; that it exists is not.
    """
    for model_id, entry in _license_acknowledgment_models().items():
        assert entry["requiresLicenseAcknowledgment"] is True, model_id
        assert entry.get("gated") is not True, f"{model_id}: the gated shape demands a credential"
        assert "credentialHost" not in entry, model_id
        for field in ("licenseUrl", "licenseNotice"):
            value = entry.get(field)
            assert isinstance(value, str) and value.strip(), f"{model_id}: empty {field}"


def test_minimax_h3_requires_license_acknowledgment_without_a_credential():
    """MiniMax-H3's own licence URL. Scoped to the family (sc-17227 review MAJOR 3): asserted over
    every flagged entry, this would fail the first non-MiniMax entry to raise the gate, for a reason
    having nothing to do with it. The credential-free half of what this used to assert is now
    `test_every_license_acknowledgment_entry_declares_the_credential_free_shape`.
    """
    for model_id, entry in _minimax_h3_license_models().items():
        assert entry["licenseUrl"] == "https://huggingface.co/MiniMaxAI/MiniMax-H3", model_id


def test_minimax_h3_license_notice_names_the_restrictions_it_notifies_of():
    """§V.2 requires notifying the user that the use restrictions apply — a bare "accept the
    license" checkbox does not. The notice must name the FOUR terms that decide whether the user
    may use the model at all, so assert on the SUBSTANCE, not on the field being non-empty.

    Scoped to the MiniMax-H3 family: every string below is MiniMax's own copy, and asserting it
    over the derived flag set made the flag mean "carries MiniMax's licence text" rather than
    "requires an acknowledgment" (sc-17227 review MAJOR 3)."""
    for model_id, entry in _minimax_h3_license_models().items():
        notice = entry["licenseNotice"]
        # §II / §I.9 — the licence binds the user, not only SceneWorks.
        assert "NON-TRANSFERABLE" in notice, model_id
        # §I.5 / §V.4 — the agreement's DEFAULT Applicable Territory, every excluded region named.
        # A partial list would mislead. Named as the agreement's own default scope, and nothing
        # here says how the written authorization below relates to it: which provision that
        # confirmation is given under is not something this repository has established
        # (sc-17227 records it as unresolved), so neither the copy nor this audit asserts it.
        for territory in (
            "European Union",
            "United Kingdom",
            "Republic of Korea",
            "United States of America",
        ):
            assert territory in notice, f"{model_id}: {territory} missing from licenseNotice"
        # The written authorization MiniMax gave SceneWorks, recorded verbatim on sc-17227. Pinned
        # as a FACT the notice must state — the copy it replaced pointed the reader at MiniMax to
        # ask about obtaining a licence, which stopped being the useful thing to say once the reply
        # arrived.
        assert "authorizes SceneWorks to use MiniMax H3 and MiniMax H3 Works" in notice, model_id
        assert "welcome to contact MiniMax about obtaining a licence" not in notice, model_id
        # Deliberately NOT pinned: any phrasing about which clause that authorization lands under,
        # in either direction (sc-17227 review MAJOR 4). A negative pin on the superseded
        # "the licence does not authorize use of the model" made restoring that reading a test
        # failure, which is a test asserting a clause attribution this repository has not
        # established. sc-17227's own analysis records the §II question as open; whether the reply
        # is a §II territorial extension is Michael's to determine, not this audit's to lock in.
        # The contact address survives the rewrite: it is the agreement's own, and a reader's
        # question about their OWN use still goes there.
        assert "api@minimax.io" in notice, model_id
        # §V.1 + Exhibit A item 12 — the disclosure obligation SceneWorks does NOT discharge for
        # the user (nothing in the app marks output as machine-generated), so it must be stated.
        assert "machine-generated" in notice, model_id
        # §IV.1 — the ceiling above which a separate authorization is required. The licence's own
        # measure is REVENUE ("generate more than 20 million US dollars ... in yearly revenue"),
        # not earnings; "earn" would read as profit and understate who is covered.
        assert "20 million US dollars in yearly revenue" in notice, model_id
        # §V.3 — the restriction SceneWorks' own feature set is most likely to reach: the product
        # ships a LoRA trainer, dataset captioning and a training studio. Named because the notice
        # claims to list the terms that decide whether the user may use the model at all, and a
        # reader who did not see it would reasonably conclude training on H3 output is
        # unrestricted. This assertion is what stops the set silently shrinking back to three.
        #
        # Bound to the ITEM HEADING, not to a bare "§V.3" (sc-17227 review LOW): the notice's
        # closing sentence "…is what §V.3 forbids" satisfied the loose form, so deleting the whole
        # fourth item still passed. The heading appears once, in the item itself.
        assert "(4) NO IMPROVING OTHER AI MODELS (§V.3)" in notice, model_id
        assert "improve any other artificial intelligence model" in notice, model_id
        assert "Four of its terms" in notice, model_id


def test_minimax_h3_shipped_notice_names_the_same_restrictions():
    """The §III.4 NOTICE that ships in the app (About → Licenses) is the copy a user has when
    they have only the built application and no repository checkout, so it must name the same set
    the manifest gate names — including §V.3. Pinned here for the same reason: so the bullet list
    cannot quietly lose a term."""
    notice = (ROOT / "apps" / "desktop" / "licenses" / "minimax-h3" / "NOTICE.txt").read_text(
        encoding="utf-8"
    )
    assert "Four of its terms bind every user directly" in notice
    for fragment in (
        "Applicable Territory (Sections I and V.4)",
        "Acceptable Use Policy (Section V.1 and Exhibit A)",
        "Additional Commercial Terms (Section IV.1)",
        "No improving other AI models (Section V.3)",
        "improve any other\n    artificial intelligence model",
        "20 million US dollars in\n    yearly revenue",
    ):
        assert fragment in notice, fragment
    # sc-17227 review: the notice must not assert, as fact, that the modified-files notice §III.2
    # requires is currently served on the re-hosted repository — nothing in this repository checks
    # that, and the re-host is owned by sc-17150. Verified by pinning the hedge, so restoring the
    # bare claim fails here.
    assert "is not verified by anything in this repository" in notice
    assert "re-hosted repository is where that notice is served" not in notice


def test_minimax_h3_declares_the_section_iv_2_ui_attribution():
    """§IV.2: "You shall prominently display 'MiniMax H3' on the user interface". The exact string
    with a SPACE — the hyphenated `MiniMax-H3` product name does not contain it.

    Scoped to the MiniMax-H3 family: §IV.2 is MiniMax's clause, and this attribution string is not
    something a flagged entry of another family could satisfy (sc-17227 review MAJOR 3)."""
    for model_id, entry in _minimax_h3_license_models().items():
        attribution = entry["ui"]["attribution"]
        assert "MiniMax H3" in attribution, model_id
        assert attribution == "Powered by MiniMax H3", model_id


def test_schema_accepts_license_acknowledgment_without_gated():
    """The authoring contract must permit the decoupled shape. Guard the SCHEMA, not just the
    shipped entries: without these keys the catalog stops validating (additionalProperties: false),
    which is the parity-lane failure sc-17227 had to clear."""
    schema = _load_schema(SCHEMA_PATH)
    validator = jsonschema.Draft202012Validator(schema)
    entry = _model_entry_with_download(
        {"provider": "huggingface", "repo": "namespace/model", "files": []}
    )
    entry["requiresLicenseAcknowledgment"] = True
    entry["licenseNotice"] = "Restrictions apply."
    entry.setdefault("ui", {})["attribution"] = "Powered by MiniMax H3"

    errors = list(validator.iter_errors({"schemaVersion": 1, "models": [entry]}))

    assert not errors, [
        (error.validator, list(error.absolute_path), error.message) for error in errors
    ]


def test_license_acknowledgment_schema_guard_has_teeth():
    """Mutation check: the three keys are only accepted because the schema declares them. Remove
    each one INDIVIDUALLY and the shape must be rejected — proving the test above is not passing
    on some catch-all."""
    base = _load_schema(SCHEMA_PATH)
    entry = _model_entry_with_download(
        {"provider": "huggingface", "repo": "namespace/model", "files": []}
    )
    entry["requiresLicenseAcknowledgment"] = True
    entry["licenseNotice"] = "Restrictions apply."
    entry.setdefault("ui", {})["attribution"] = "Powered by MiniMax H3"
    document = {"schemaVersion": 1, "models": [entry]}

    for holder, key in (
        (("properties", "models", "items", "properties"), "requiresLicenseAcknowledgment"),
        (("properties", "models", "items", "properties"), "licenseNotice"),
        (("properties", "models", "items", "properties", "ui", "properties"), "attribution"),
    ):
        schema = copy.deepcopy(base)
        node = schema
        for step in holder:
            node = node[step]
        assert key in node, f"{key} is not declared where this guard looks"
        del node[key]
        errors = list(jsonschema.Draft202012Validator(schema).iter_errors(document))
        assert any(
            error.validator == "additionalProperties" and key in error.message
            for error in errors
        ), f"removing {key} from the schema did not reject the entry"
