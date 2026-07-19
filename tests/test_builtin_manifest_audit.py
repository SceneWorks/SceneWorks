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
    to a dict for the capability / UI-wiring audits.
  * ``_manifest_brace_walker`` walks balanced braces so a URL containing ``//``
    inside an entry doesn't trip a naive comment strip; used by the per-model
    ``mlx`` block audits.

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

import json
import re
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "manifests" / "builtin.models.jsonc"
SCHEMA_PATH = ROOT / "packages" / "schemas" / "model-manifest.schema.json"


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


def _load_builtin_models_manifest() -> dict:
    raw = MANIFEST_PATH.read_text(encoding="utf-8")
    return json.loads(_strip_jsonc_comments(raw))


def test_builtin_models_manifest_satisfies_authoring_schema():
    """sc-12338: the builtin catalog's $schema is an enforced CI contract."""
    manifest = _load_builtin_models_manifest()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
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
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    assert any("recommendded" in error.message for error in errors)


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


def test_manifest_constraint_contract_registry_is_complete_and_live():
    """sc-12304: constraint declarations may not silently become decoration.

    The schema is the author-facing registry; this test makes its custom contract
    annotations a CI gate. It checks both directions (manifest -> registry and
    registry -> manifest), and binding entries must name production readers that
    contain the exact key. Advisory/descriptive entries are explicitly allowed not
    to reject requests, which is materially different from an accidental dead key.
    """
    manifest = _load_builtin_models_manifest()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
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


def _manifest_brace_walker():
    # Helper for the mlx-block manifest tests. Returns (raw, find_entry_block,
    # find_mlx_block) that walk balanced braces so a URL containing `//` (in
    # the entry text) doesn't trip a naive jsonc strip.
    raw = MANIFEST_PATH.read_text(encoding="utf-8")

    def find_balanced_block(start_index: int) -> str:
        depth = 0
        for index in range(start_index, len(raw)):
            ch = raw[index]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start_index : index + 1]
        raise AssertionError(f"unterminated brace block from index {start_index}")

    def find_entry_block(model_id: str) -> str:
        anchor = raw.index(f'"id": "{model_id}"')
        start = raw.rfind("{", 0, anchor)
        assert start != -1, f"entry start brace for {model_id} not found"
        return find_balanced_block(start)

    def find_mlx_block(entry_block: str) -> str:
        match = re.search(r'"mlx"\s*:\s*\{', entry_block)
        assert match, "entry block has no mlx block"
        # Resolve the entry block's position in the raw manifest, then walk
        # balanced braces from the actual opening brace so nested limits {...}
        # are captured (Qwen carries a sampler/scheduler limits override, FLUX
        # does not).
        entry_start = raw.index(entry_block)
        mlx_open = entry_start + match.end() - 1
        return find_balanced_block(mlx_open)

    return raw, find_entry_block, find_mlx_block


# ---------------------------------------------------------------------------
# Per-model `mlx` block structural audits (pure manifest; brace-walker based).
# Extracted from tests/test_worker_image_adapters.py (sc-8861 / F-059).
# ---------------------------------------------------------------------------


def test_flux_manifest_has_mlx_block():
    # Manifest-driven auto-dispatch + Model Manager memory tier (sc-1970).
    # The Rust API owns the canonical jsonc parser; here we just confirm both
    # FLUX entries carry an `mlx` block and the contents look right.
    _, find_entry_block, find_mlx_block = _manifest_brace_walker()

    for model_id in ("flux_schnell", "flux_dev"):
        block = find_entry_block(model_id)
        mlx_block = find_mlx_block(block)
        quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
        mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
        assert quant_match and int(quant_match.group(1)) in {3, 4, 5, 6, 8}, (
            f"{model_id} mlx.quantize must be a supported quant level (sc-1970)"
        )
        assert mem_match and int(mem_match.group(1)) > 0, (
            f"{model_id} mlx.minMemoryGb must be a positive int (sc-1970)"
        )


def test_qwen_image_manifest_has_mlx_block():
    # sc-1972: qwen_image carries an mlx block + sampler/scheduler limits
    # override (mflux's loop is sealed on "linear" — match the wan_2_2
    # precedent of restricting the menu to default-only when the MLX path is
    # the active backend, epic 1753 §14).
    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("qwen_image")
    mlx_block = find_mlx_block(block)
    quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
    mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
    assert quant_match and int(quant_match.group(1)) in {3, 4, 5, 6, 8}, (
        "qwen_image mlx.quantize must be a supported quant level (sc-1972)"
    )
    assert mem_match and int(mem_match.group(1)) > 0, (
        "qwen_image mlx.minMemoryGb must be a positive int (sc-1972)"
    )
    # MLX sampler/scheduler menu (epic 7114 P5, sc-7126): the native MLX engine now
    # routes through the unified curated sampler/scheduler framework (the old mflux
    # linear-only loop is gone), so the mlx block advertises the curated menu.
    assert '"dpmpp_2m"' in mlx_block and '"uni_pc"' in mlx_block, (
        "qwen_image mlx must advertise the curated sampler menu (epic 7114)"
    )
    assert '"sgm_uniform"' in mlx_block, (
        "qwen_image mlx must advertise the curated scheduler menu (epic 7114)"
    )


def test_flux2_true_v2_manifest_install_time_conversion():
    # sc-2235: the entry must declare the install-time conversion contract the
    # Rust convert job + adapter rely on.
    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("flux2_klein_9b_true_v2")
    assert '"macOnly": true' in block
    assert '"adapter": "mlx_flux2"' in block
    # Only the bf16 single-file is pulled (not the whole 73 GB repo).
    assert "Flux2-Klein-9B-True-v2-bf16.safetensors" in block
    # Undistilled defaults differ from the 4-step distill.
    assert re.search(r'"steps"\s*:\s*24', block)

    mlx_block = find_mlx_block(block)
    assert '"requiresConversion": true' in mlx_block
    assert '"converter": "flux2_klein_diffusers"' in mlx_block
    assert '"convertSourceRepo": "wikeeyang/Flux2-Klein-9B-True-V2"' in mlx_block
    assert '"convertBaseRepo": "black-forest-labs/FLUX.2-klein-9B"' in mlx_block
    assert re.search(r'"quantize"\s*:\s*8', mlx_block)


def test_flux2_klein_manifest_entries_present():
    # Both flux2_klein_9b and flux2_klein_9b_kv must be present in the
    # builtin manifest with the expected adapter + family + mlx block.
    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    # Both ids expose the same capability set: -kv is no longer gated to
    # character_image only — it runs plain txt2img on par with the base 9B,
    # the cache just doesn't engage without a reference (sc-2173).
    for model_id in ("flux2_klein_9b", "flux2_klein_9b_kv"):
        block = find_entry_block(model_id)
        assert '"adapter": "mlx_flux2"' in block, model_id
        assert '"family": "flux2-klein"' in block, model_id
        assert '"macOnly": true' in block, model_id
        # sc-8711 (epic 8506): re-hosted as a public, ungated SceneWorks MLX quant-matrix
        # turnkey (q4/q8/bf16), so the entry is `gated: false` with no credentialHost — the
        # FLUX Non-Commercial LICENSE.md travels with the weights.
        assert '"gated": false' in block, model_id
        mlx_block = find_mlx_block(block)
        quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
        assert quant_match is not None, f"{model_id}: mlx.quantize missing"
        # quantize records the DEFAULT tier (q4); the load Quant is forced to None so the
        # dense bf16 Qwen3 TE is preserved (DENSE_TE_TIER_MODELS).
        assert int(quant_match.group(1)) == 4, f"{model_id}: default tier should be q4 (sc-8711)"
        assert '"text_to_image"' in block, model_id
        assert '"character_image"' in block, model_id


def test_z_image_turbo_manifest_has_mlx_block():
    # sc-2145: z_image_turbo carries an mlx block + sampler/scheduler limits
    # override (mflux's loop is sealed on "linear" — match the wan_2_2 /
    # qwen_image precedents of restricting the menu to default-only when the
    # MLX path is the active backend, epic 1753 §14).
    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("z_image_turbo")
    mlx_block = find_mlx_block(block)
    quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
    mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
    assert quant_match and int(quant_match.group(1)) in {3, 4, 5, 6, 8}, (
        "z_image_turbo mlx.quantize must be a supported quant level (sc-2145)"
    )
    assert mem_match and int(mem_match.group(1)) > 0, (
        "z_image_turbo mlx.minMemoryGb must be a positive int (sc-2145)"
    )
    # epic 7114 P5 (sc-7126): the native MLX engine adopted the unified curated
    # sampler/scheduler framework, so the mflux linear-only restriction is gone.
    assert '"dpmpp_2m"' in mlx_block and '"uni_pc"' in mlx_block, (
        "z_image_turbo mlx must advertise the curated sampler menu (epic 7114)"
    )
    assert '"sgm_uniform"' in mlx_block, (
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


def test_wan_2_2_candle_vram_tiers_match_measured_peaks():
    """sc-12402/sc-12631: never regress the measured 5B peaks to estimates.

    Measured on an idle RTX PRO 6000 at wan_2_2's own shipped default (832x480, 121 frames,
    20 steps, CFG on) -- the schema's "video = default frames". The peak is the DENOISE
    (weights-dominated after sc-12434 chunked the sdpa); the z48 vae22 decode adds 0.0 GB, which
    is what makes these card-independent despite the decode tiler budgeting off total VRAM.
    """
    manifest = _load_builtin_models_manifest()
    wan = next(model for model in manifest["models"] if model["id"] == "wan_2_2")
    candle = wan["candle"]

    assert candle["measured"] is True
    assert {tier: candle["vramGbByTier"][tier] for tier in ("q4", "q8", "bf16")} == {
        "q4": 46.1,
        "q8": 48.7,
        "bf16": 54.0,
    }
    # minMemoryGb gates the default/lightest (q4) tier + the fit gate's 2 GB headroom.
    assert candle["minMemoryGb"] == 48


def test_wan_a14b_candle_q4_measured_admits_32gb_q8_bf16_deferred():
    """sc-12631 (post epic sc-12732): the A14B q4 candle peak is MEASURED and admits a 32 GB card.

    After the sequential-offload / expert-swap / bf16-TE / free-aware-tiling / finer-sdpa rework, the
    A14B renders one 14B expert at a time and its measured `USED_MEM_HIGH` q4 peak at the 1280x720/81f/
    4-step Lightning default is ~22 GiB -- it fits a 32 GB RTX 5090, not the ~386 GiB OOM-floor these
    blocks used to carry. This is the inverse of the old `..._are_flagged_estimated` tripwire (which
    asserted every tier exceeds a 96 GB card). (The raw 22 GiB q4 peak physically fits a 24 GB card too,
    but the gate's 2 GB headroom targets ~26 GB free, so a 24 GB card is refused -- the safe direction.)

    Per the sc-12732 handoff ("admit q4 now, re-measure q8/bf16 before admitting"), q8 and bf16 are
    DEFERRED, so the block stays `measured: false`. q8's live USED_MEM_HIGH peak was ~28 GiB but its
    nvidia-smi pool high-water (which cudarc never frees) was ~34-36 GiB and the small-card footprint is
    unvalidated, so q8 is gated at that conservative pool bound (refused on 32 GB); bf16 is DERIVED (56).
    Asserting q8/bf16 stay above a 32 GB card stops either regressing to a fits-small-card number before
    it is validated/measured. Flipping to `measured: true` is a <=32 GB-validation + bf16-stage follow-up.
    """
    manifest = _load_builtin_models_manifest()
    expected = {
        "wan_2_2_t2v_14b": {"q4": 22.13, "q8": 34.4, "bf16": 56.0},
        "wan_2_2_i2v_14b": {"q4": 22.20, "q8": 35.6, "bf16": 56.0},
    }
    for model_id, tiers in expected.items():
        entry = next(m for m in manifest["models"] if m["id"] == model_id)
        candle = entry["candle"]
        # q8/bf16 are deferred (conservative bounds), so the block is honestly flagged estimated.
        assert candle["measured"] is False, f"{model_id}: q8/bf16 deferred, so measured stays false"
        assert candle["vramGbByTier"] == tiers, (
            f"{model_id}: the measured q4 peak (and the conservative q8/bf16 bounds) must not regress, "
            f"got {candle['vramGbByTier']}"
        )
        assert candle["minMemoryGb"] == 24, f"{model_id}: minMemoryGb should gate q4 (~22 + 2)"
        # The DEFAULT (q4) tier now fits a 32 GB card -- where the old ~388 floor refused every GPU.
        assert tiers["q4"] + 2 < 32, f"{model_id}: q4 (+headroom) must fit a 32 GB card, got {tiers['q4']}"
        # q8 + bf16 stay refused on a 32 GB card until validated/measured (deferred, the safe direction).
        assert tiers["q8"] + 2 > 32, f"{model_id}: q8's conservative bound must not admit a 32 GB card"
        assert tiers["bf16"] + 2 > 32, f"{model_id}: the derived bf16 bound must not admit a 32 GB card"


def test_sdxl_manifest_has_mlx_block():
    # sdxl carries an mlx block (no `limits` override here — the MLX SDXL schedule
    # matches the torch EulerDiscrete default, and there's no per-model sampler menu
    # in the sdxl manifest entry to limit).
    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("sdxl")
    mlx_block = find_mlx_block(block)
    mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
    assert mem_match and int(mem_match.group(1)) > 0, (
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
