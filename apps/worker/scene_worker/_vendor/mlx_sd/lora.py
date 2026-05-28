"""SceneWorks-authored SDXL LoRA loader for the vendored mlx-examples UNet.

sc-1975 path C: Apple's `mlx-examples/stable_diffusion` ships no LoRA support.
This module adds **kohya-with-diffusers-blocks** AND **PEFT** SDXL LoRA merging
for the mlx-examples UNet so the SceneWorks SDXL flows (`_SdxlLoraBackend`
training output, sc-1942) AND common HF community LoRAs (LCM-LoRA, pixel-art-xl,
the `nerijs` / `latent-consistency` / civitai mirror catalog, etc.) work
through the MLX backend without external conversion.

NOT vendored from upstream — entirely SceneWorks-authored. The merge is applied
pre-quantization (any future quantize step on the UNet happens AFTER LoRA
merging so the LoRA contribution lands in the bf16 reference, not the
quantized weights).

## Supported formats

**1. Kohya with diffusers block naming** — what `pipe.save_lora_weights()` writes
and almost every modern HF community SDXL LoRA ships in:

    lora_unet_<diffusers path with `_` instead of `.`>.lora_down.weight
    lora_unet_<diffusers path with `_` instead of `.`>.lora_up.weight
    lora_unet_<...>.alpha   (optional scalar)

The path swap is purely `_` ↔ `.`: e.g.
`lora_unet_down_blocks_2_attentions_1_transformer_blocks_4_attn2_to_q.lora_down.weight`
→ targets `unet.down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_q`.

**2. PEFT** — what `peft.PeftModel.save_pretrained()` writes (the format
SceneWorks's `_SdxlLoraBackend` in training_adapters.py emits):

    base_model.model.unet.<path>.lora_A.default.weight
    base_model.model.unet.<path>.lora_B.default.weight
    base_model.model.unet.<path>.alpha                 (optional scalar)

Note the swapped naming: kohya's `lora_down` is PEFT's `lora_A` (rank × in),
and kohya's `lora_up` is PEFT's `lora_B` (out × rank).

## NOT supported (yet)
- Original-SD / old-kohya block naming (`lora_unet_input_blocks_*_in_layers_*`):
  several older LoRAs (the SDXL offset example, some pre-diffusers community
  LoRAs) use this. The mapping is more involved (the original SD UNet topology
  differs from diffusers' SDXL topology) — follow-up.
- Conv layer LoRAs (only `nn.Linear` targets are merged; the diffusers SDXL
  attention LoRAs target only Linears, which is the dominant case).
- Text encoder LoRAs (rare on SDXL; SceneWorks's pipeline trains UNet-only).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open


# PEFT prefix carries the wrapped diffusers path; kohya prefix is "lora_unet_"
# directly followed by the underscore-separated diffusers path.
_PEFT_PREFIX = "base_model.model.unet."
_KOHYA_PREFIX = "lora_unet_"

# mlx-examples' Apple port renamed several attention modules relative to the
# diffusers SDXL UNet. Community / kohya LoRAs reference the diffusers leaf
# names; we walk the UNet once and register both the literal mlx-examples
# name AND the diffusers-equivalent alias so a community LoRA's
# `<path>_attn1_to_q` key still finds the matching `<path>.attn1.query_proj`
# Linear here. Trained-with-mlx-examples LoRAs (none exist in production
# today) would match via the literal path.
_DIFFUSERS_LEAF_ALIASES: dict[str, str] = {
    # mlx-examples leaf      # diffusers leaf
    "query_proj": "to_q",
    "key_proj": "to_k",
    "value_proj": "to_v",
    "out_proj": "to_out_0",
}
# NOT yet aliased: the GEGLU FF net (mlx-examples splits as linear1/2/3 while
# diffusers uses ff.net.0.proj + ff.net.2 — a non-1:1 split). FF LoRAs are
# the minority of community SDXL LoRAs; documented gap, follow-up extension.


def apply_loras_to_unet(
    unet: nn.Module,
    lora_specs: list[dict[str, Any]],
) -> int:
    """Merge a list of SDXL LoRAs into the given mlx-examples UNet.

    Each spec is ``{"path": <str path to .safetensors>, "weight": <float scale>}``
    — the same shape `lora_adapters.normalize_lora_specs` yields on the torch
    path, so the adapter layer can pass it through unchanged.

    Returns the total number of Linear modules touched across all LoRAs (for
    diagnostic logging — the adapter emits this in the worker event payload).
    A returned 0 with a non-empty `lora_specs` list is the smoke signal that
    the LoRA is in a format we don't support yet (raise upstream).
    """
    if not lora_specs:
        return 0
    # Build BOTH lookup tables ONCE per call: the dotted (diffusers) form is
    # what PEFT keys reference; the kohya form (each `.` replaced with `_`) is
    # what kohya keys reference. We can't reliably reverse-substitute a kohya
    # key without this table because legitimate diffusers names like
    # `down_blocks` and `transformer_blocks` already contain `_` — a blind
    # `_` → `.` pass would invent paths that don't exist
    # (`down.blocks.2.transformer.blocks` etc.). SDXL UNets have a few thousand
    # Linear modules; the dicts are single-digit MB and looked up once per
    # LoRA key (~2000–4000 keys per file).
    name_to_module: dict[str, nn.Linear] = {}
    kohya_to_diffusers: dict[str, str] = {}
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            name_to_module[name] = module
            kohya_to_diffusers[name.replace(".", "_")] = name
            # If this Linear is an attention proj with a renamed leaf, also
            # register the diffusers-equivalent kohya alias so community LoRAs
            # match. E.g. `down_blocks.1.attn1.query_proj` adds
            # `down_blocks_1_attn1_to_q` as an alias.
            parent, _, leaf = name.rpartition(".")
            alias = _DIFFUSERS_LEAF_ALIASES.get(leaf)
            if parent and alias is not None:
                kohya_to_diffusers[f"{parent}.{alias}".replace(".", "_")] = name

    total_touched = 0
    for spec in lora_specs:
        lora_path = Path(spec["path"])
        scale = float(spec.get("weight", 1.0))
        triples = _read_lora_triples(lora_path, kohya_to_diffusers)
        touched = _merge_triples(name_to_module, triples, scale=scale)
        total_touched += touched
    return total_touched


def _read_lora_triples(
    lora_path: Path,
    kohya_to_diffusers: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Return a `module_name -> {"down", "up", "alpha"}` map across both formats.

    `module_name` is the dotted diffusers path that resolves under `unet.<...>`.
    Format detection is per-key (not per-file) so a mixed file would still
    decode cleanly — in practice each file is single-format, but the cost is
    zero and the code is simpler this way. Tensors stay as `mx.array` until
    merge time so we don't pay an unnecessary copy.
    """
    triples: dict[str, dict[str, Any]] = defaultdict(dict)
    with safe_open(str(lora_path), framework="mlx") as handle:
        for key in handle.keys():
            module_name, role = _classify_key(key, kohya_to_diffusers)
            if module_name is None:
                continue
            tensor = handle.get_tensor(key)
            if role == "down":
                triples[module_name]["down"] = tensor
            elif role == "up":
                triples[module_name]["up"] = tensor
            elif role == "alpha":
                # alpha is a 0-d scalar; convert to a Python float lazily.
                try:
                    triples[module_name]["alpha"] = float(tensor.item())
                except (TypeError, AttributeError):
                    triples[module_name]["alpha"] = float(tensor)
    return triples


def _classify_key(
    key: str,
    kohya_to_diffusers: dict[str, str],
) -> tuple[str | None, str | None]:
    """Map a raw safetensors key onto (diffusers_module_name, role).

    Role is one of "down", "up", "alpha". Returns (None, None) for keys we
    don't recognize (text encoder LoRAs, conv keys we don't merge, kohya keys
    whose stripped stem doesn't appear in the UNet's Linear-module table).
    """
    if key.startswith(_PEFT_PREFIX):
        remainder = key[len(_PEFT_PREFIX):]
        # PEFT names map directly to dotted diffusers paths.
        # PEFT's lora_A == kohya's lora_down (rank × in_features);
        # PEFT's lora_B == kohya's lora_up   (out_features × rank).
        if remainder.endswith(".lora_A.default.weight"):
            return remainder[: -len(".lora_A.default.weight")], "down"
        if remainder.endswith(".lora_B.default.weight"):
            return remainder[: -len(".lora_B.default.weight")], "up"
        if remainder.endswith(".alpha"):
            return remainder[: -len(".alpha")], "alpha"
        return None, None
    if key.startswith(_KOHYA_PREFIX):
        # Kohya keys flatten the diffusers path with `_` instead of `.`. We can't
        # blindly reverse-substitute (`down_blocks` is a single token in diffusers
        # naming), so look up the kohya stem in the table the caller built by
        # walking `unet.named_modules()`.
        remainder = key[len(_KOHYA_PREFIX):]
        if remainder.endswith(".lora_down.weight"):
            stem = remainder[: -len(".lora_down.weight")]
            return kohya_to_diffusers.get(stem), "down"
        if remainder.endswith(".lora_up.weight"):
            stem = remainder[: -len(".lora_up.weight")]
            return kohya_to_diffusers.get(stem), "up"
        if remainder.endswith(".alpha"):
            stem = remainder[: -len(".alpha")]
            return kohya_to_diffusers.get(stem), "alpha"
        return None, None
    # Text-encoder LoRA prefixes (`lora_te1_`, `lora_te2_`, PEFT variants)
    # land here — we deliberately skip them; mlx-examples freezes the text
    # encoders at load time and SceneWorks's pipeline trains UNet-only.
    return None, None


def _merge_triples(
    name_to_module: dict[str, nn.Linear],
    triples: dict[str, dict[str, Any]],
    *,
    scale: float,
) -> int:
    """Apply each complete (down, up) triple to its matching UNet Linear.

    The merge math is the standard LoRA formula: ``W += (α / r) * (B @ A) * scale``
    where A is the down matrix (rank, in), B is the up matrix (out, rank), α is
    the per-layer alpha (default rank if missing), and `scale` is the user-set
    per-LoRA scale. Half-pair (down without up or vice versa) keys are skipped
    — they happen with conv-only LoRAs we don't merge yet.
    """
    touched = 0
    for module_name, parts in triples.items():
        if "down" not in parts or "up" not in parts:
            continue
        module = name_to_module.get(module_name)
        if module is None:
            # Key targets a name that isn't a `nn.Linear` here (e.g. a Conv,
            # or a misnamed key from a non-SDXL LoRA). Skip silently — counting
            # matches is the smoke signal upstream.
            continue
        # The `down` matrix has shape (rank, in_features) by both conventions
        # (kohya `lora_down` is rank×in; PEFT `lora_A` is rank×in). Same for
        # `up` being (out_features, rank).
        a = parts["down"]
        b = parts["up"]
        # Bail out on conv-shaped LoRAs (4D weight tensors) — we don't merge
        # those into Linear modules; they'd hit a shape mismatch anyway.
        if a.ndim != 2 or b.ndim != 2:
            continue
        rank = a.shape[0]
        alpha = parts.get("alpha", float(rank))  # PEFT/kohya default: alpha = rank
        effective_scale = (alpha / rank) * scale

        delta = (b @ a).astype(module.weight.dtype) * effective_scale
        module.weight = module.weight + delta
        touched += 1
    return touched
