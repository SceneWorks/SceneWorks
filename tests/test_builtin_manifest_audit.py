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

The three character_image ENGINE-WIRING guards additionally cross-reference the
worker's ``MODEL_TARGETS`` table (which is worker-owned config, not manifest
data). They lazily ``importorskip`` it so they exercise the full manifest ↔
worker cross-check while the worker still exists, and degrade to a clean SKIP
(not a collection error) once ``apps/worker`` is deleted. Reimplementing those
against the Rust engine table post-8283 is tracked as a follow-up.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "manifests" / "builtin.models.jsonc"


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


def _model_targets() -> dict:
    """Worker-owned engine table (`ipAdapter`/`instantId`/`pulidFlux`/
    `controlNetPose`). This is the sole scene_worker dependency in this module
    and is imported lazily so its removal (epic 8283) turns the three
    engine-wiring cross-checks below into a SKIP rather than a collection error.
    sc-8861 FOLLOW_UP: reimplement those three against the Rust engine table
    (crates/sceneworks-worker) once apps/worker is deleted.
    """
    image_adapters = pytest.importorskip(
        "scene_worker.image_adapters",
        reason="scene_worker.image_adapters (MODEL_TARGETS) retired with apps/worker (epic 8283)",
    )
    return image_adapters.MODEL_TARGETS


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
# The three engine-wiring guards also cross-reference the worker MODEL_TARGETS
# table via the lazy `_model_targets()` importorskip above.
# ---------------------------------------------------------------------------


def test_character_image_capability_implies_engine_or_tuning_declaration():
    """Every builtin model that advertises `character_image` must have either
    a worker engine block (`ipAdapter` / `instantId` in MODEL_TARGETS) OR a
    `ui.variationStrength` declaration in the manifest. Otherwise the capability
    flag is dishonest — the picker shows the model in "With character" mode but
    the worker silently ignores the reference, the same shape as z_image_turbo's
    pre-sc-2005 bug. This is the cross-backbone guard for epic 2003 (sc-2018):
    adding a future character_image backbone without engine wiring will fail
    here before it ever reaches a user.
    """
    model_targets = _model_targets()
    manifest = _load_builtin_models_manifest()
    misleading: list[str] = []
    for model in manifest.get("models", []):
        capabilities = model.get("capabilities") or []
        if "character_image" not in capabilities:
            continue
        target = model_targets.get(model["id"], {})
        ui = model.get("ui") or {}
        has_engine = bool(target.get("ipAdapter") or target.get("instantId") or target.get("pulidFlux"))
        has_variation_ui = bool(ui.get("variationStrength"))
        if not (has_engine or has_variation_ui):
            misleading.append(model["id"])
    assert not misleading, (
        f"Models advertise `character_image` without engine wiring or a "
        f"`ui.variationStrength` declaration: {misleading}. Add an `ipAdapter`, "
        f"`instantId`, or `pulidFlux` block in MODEL_TARGETS for an IP-Adapter / "
        f"face-ID backbone, or declare `ui.variationStrength` for an edit-style "
        f"backbone (sc-2017), or drop the capability flag (the z_image_turbo bug, "
        f"sc-2005)."
    )


def test_kolors_declares_strict_pose_controlnet():
    """sc-2264: Kolors is the strict pose tier — the manifest must advertise
    ui.poseLibrary AND the worker target must carry the controlNetPose config so
    the pose picker offers it and the adapter can load the pose ControlNet."""
    model_targets = _model_targets()
    manifest = _load_builtin_models_manifest()
    manifest_by_id = {model["id"]: model for model in manifest.get("models", [])}
    kolors = manifest_by_id.get("kolors", {})
    assert kolors.get("ui", {}).get("poseLibrary") is True, (
        "kolors must declare ui.poseLibrary so the pose picker offers the strict tier (sc-2264)."
    )
    target = model_targets.get("kolors", {})
    assert target.get("controlNetPose", {}).get("repo") == "Kwai-Kolors/Kolors-ControlNet-Pose", (
        "kolors MODEL_TARGETS must carry the Kolors-ControlNet-Pose repo for the strict pose path."
    )
    # Identity still rides the IP-Adapter; the pose path composes both.
    assert target.get("ipAdapter"), "kolors pose path needs the IP-Adapter for identity."


def test_models_with_engine_block_advertise_character_image():
    """The reverse-drift guard. Any model that ships an `ipAdapter` or
    `instantId` block in MODEL_TARGETS exists to serve Character Studio's
    reference flow — the manifest must advertise the capability so the picker
    surfaces it. Catches the case where someone wires the worker engine but
    forgets to flip the manifest flag, leaving the engine unreachable.
    """
    model_targets = _model_targets()
    manifest = _load_builtin_models_manifest()
    manifest_by_id = {model["id"]: model for model in manifest.get("models", [])}
    unreachable: list[str] = []
    for model_id, target in model_targets.items():
        if not (target.get("ipAdapter") or target.get("instantId") or target.get("pulidFlux")):
            continue
        builtin = manifest_by_id.get(model_id)
        if builtin is None:
            # Worker-only target not exposed as a built-in (unwired path).
            continue
        capabilities = builtin.get("capabilities") or []
        if "character_image" not in capabilities:
            unreachable.append(model_id)
    assert not unreachable, (
        f"Models have engine blocks in MODEL_TARGETS but the builtin manifest "
        f"does not advertise `character_image`: {unreachable}. Add the capability "
        f"to `capabilities` and `ui.recommendedFor` so the Image Studio "
        f"\"With character\" picker surfaces the model."
    )


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
