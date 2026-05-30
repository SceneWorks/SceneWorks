"""Convert a FLUX.2-klein *original/single-file* transformer checkpoint into the
diffusers `transformer/` layout that mflux's `Flux2KleinWeightDefinition` loads,
then assemble a complete local diffusers model dir by borrowing the VAE / text
encoder / tokenizer / scheduler from an already-installed base FLUX.2-klein-9B.

Motivation (sc-2220 / sc-2235): community FLUX.2-klein fine-tunes such as
``wikeeyang/Flux2-Klein-9B-True-V2`` ship ONLY a transformer, as a single flat
safetensors file in the original (ComfyUI/BFL) key convention — no diffusers
subfolders, no text-encoder/VAE. mflux needs a full diffusers dir, and its
`Flux2WeightMapping.get_transformer_mapping` is a pure 1:1 rename (no permutes),
so the on-disk tensors must already match BFL's *diffusers* convention. This
module reproduces the exact transforms diffusers' own
``convert_flux2_transformer_checkpoint_to_diffusers`` applies:

  * key renames (img_in -> x_embedder, *.lin -> *.linear, ...)
  * double-block fused qkv [3*d, d] row-split into to_q/to_k/to_v (img stream)
    and add_q_proj/add_k_proj/add_v_proj (txt stream)
  * single-block linear1/linear2 -> to_qkv_mlp_proj/to_out (1:1; diffusers also
    keeps the single block fused)
  * ``final_layer.adaLN_modulation.1`` -> ``norm_out.linear`` WITH a scale/shift
    SWAP: BFL packs (shift, scale); diffusers/mflux expect (scale, shift). This
    one swap is load-bearing — that tensor modulates every output patch, so
    getting it wrong corrupts the whole image with a periodic weave (sc-2220).

The transformer math (`build_target_state_dict`) is framework-agnostic — it
takes tensor-op callables — so it is unit-testable with numpy in the main worker
venv. The mlx load/save + dir assembly live in `convert_and_assemble`, which
imports mlx lazily (only available in the mlx-flux sidecar venv).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Callable

# Borrowed-from-base components: a transformer-only fine-tune does not touch
# these, so taking them from the installed base klein-9B is correct.
BORROWED_SUBDIRS = ("vae", "text_encoder", "tokenizer", "scheduler")
BORROWED_FILES = ("model_index.json",)

# Top-level (non-block) direct renames: original -> diffusers.
_TOP_RENAMES = {
    "img_in.weight": "x_embedder.weight",
    "txt_in.weight": "context_embedder.weight",
    "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
    "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
    "final_layer.linear.weight": "proj_out.weight",
}
# Handled separately (scale/shift swap): final_layer.adaLN_modulation.1 -> norm_out.linear.
_ADALN_SOURCE = "final_layer.adaLN_modulation.1.weight"
_ADALN_TARGET = "norm_out.linear.weight"

# Per-double-block renames (original suffix -> diffusers suffix), excluding the
# fused qkv tensors which are row-split below.
_DOUBLE_RENAMES = {
    "img_attn.norm.query_norm.weight": "attn.norm_q.weight",
    "img_attn.norm.key_norm.weight": "attn.norm_k.weight",
    "img_attn.proj.weight": "attn.to_out.0.weight",
    "img_mlp.0.weight": "ff.linear_in.weight",
    "img_mlp.2.weight": "ff.linear_out.weight",
    "txt_attn.norm.query_norm.weight": "attn.norm_added_q.weight",
    "txt_attn.norm.key_norm.weight": "attn.norm_added_k.weight",
    "txt_attn.proj.weight": "attn.to_add_out.weight",
    "txt_mlp.0.weight": "ff_context.linear_in.weight",
    "txt_mlp.2.weight": "ff_context.linear_out.weight",
}
# fused qkv -> (q, k, v) target suffixes, per stream.
_DOUBLE_QKV = {
    "img_attn.qkv.weight": ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"),
    "txt_attn.qkv.weight": ("attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"),
}
# Per-single-block renames (1:1; diffusers keeps the fused single block).
_SINGLE_RENAMES = {
    "linear1.weight": "attn.to_qkv_mlp_proj.weight",
    "linear2.weight": "attn.to_out.weight",
    "norm.query_norm.weight": "attn.norm_q.weight",
    "norm.key_norm.weight": "attn.norm_k.weight",
}


def _count_blocks(keys, prefix: str) -> int:
    idxs = [int(m.group(1)) for k in keys for m in [re.match(rf"{prefix}\.(\d+)\.", k)] if m]
    return max(idxs) + 1 if idxs else 0


def build_target_state_dict(
    src: dict,
    *,
    chunk3: Callable[[object], tuple[object, object, object]],
    swap_halves: Callable[[object], object],
) -> dict:
    """Map an original-format FLUX.2-klein transformer state dict onto diffusers
    keys. ``chunk3`` row-splits a [3*d, ...] tensor into three; ``swap_halves``
    splits a [2*d, ...] tensor and swaps the halves (shift,scale -> scale,shift).
    Pure: no framework import, so numpy ops drive the unit test.
    """
    out: dict = {}
    for s, d in _TOP_RENAMES.items():
        out[d] = src[s]
    out[_ADALN_TARGET] = swap_halves(src[_ADALN_SOURCE])

    n_double = _count_blocks(src, "double_blocks")
    for i in range(n_double):
        s, d = f"double_blocks.{i}", f"transformer_blocks.{i}"
        for src_suffix, (q, k, v) in _DOUBLE_QKV.items():
            tq, tk, tv = chunk3(src[f"{s}.{src_suffix}"])
            out[f"{d}.{q}"], out[f"{d}.{k}"], out[f"{d}.{v}"] = tq, tk, tv
        for src_suffix, dst_suffix in _DOUBLE_RENAMES.items():
            out[f"{d}.{dst_suffix}"] = src[f"{s}.{src_suffix}"]

    n_single = _count_blocks(src, "single_blocks")
    for i in range(n_single):
        s, d = f"single_blocks.{i}", f"single_transformer_blocks.{i}"
        for src_suffix, dst_suffix in _SINGLE_RENAMES.items():
            out[f"{d}.{dst_suffix}"] = src[f"{s}.{src_suffix}"]

    return out


def _safetensors_header_keys(path: Path) -> set[str]:
    """Read safetensors tensor names + shapes from the header alone (no weights)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}


def _validate_against_base(produced: dict, base_transformer_dir: Path) -> None:
    """Hard guard: the produced key set + shapes must exactly match the base
    klein diffusers transformer (the ground-truth diffusers layout mflux loads).
    """
    base: dict = {}
    for shard in sorted(base_transformer_dir.glob("*.safetensors")):
        base.update(_safetensors_header_keys(shard))
    if not base:
        raise RuntimeError(f"No base transformer safetensors in {base_transformer_dir}")
    prod_keys, base_keys = set(produced), set(base)
    missing, extra = base_keys - prod_keys, prod_keys - base_keys
    bad_shape = [k for k in prod_keys & base_keys if tuple(produced[k].shape) != base[k]]
    if missing or extra or bad_shape:
        raise RuntimeError(
            f"Conversion validation FAILED vs base transformer: "
            f"{len(missing)} missing, {len(extra)} extra, {len(bad_shape)} shape mismatch. "
            f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]} shape={bad_shape[:5]}"
        )


def convert_and_assemble(source_file: str, base_dir: str, out_dir: str) -> str:
    """Convert ``source_file`` (original single-file transformer) into ``out_dir``
    as a complete diffusers model dir, borrowing components from ``base_dir`` (an
    installed base FLUX.2-klein-9B diffusers snapshot). Returns ``out_dir``.
    Imports mlx lazily — only runnable inside the mlx-flux sidecar venv.
    """
    import mlx.core as mx

    source = Path(source_file)
    base = Path(base_dir)
    out = Path(out_dir)
    base_transformer = base / "transformer"
    if not source.is_file():
        raise FileNotFoundError(f"source transformer file not found: {source}")
    if not base_transformer.is_dir():
        raise FileNotFoundError(f"base transformer dir not found: {base_transformer}")

    def chunk3(t):
        return tuple(mx.split(t, 3, axis=0))

    def swap_halves(t):
        shift, scale = mx.split(t, 2, axis=0)
        return mx.concatenate([scale, shift], axis=0)

    src = mx.load(str(source))
    produced = build_target_state_dict(src, chunk3=chunk3, swap_halves=swap_halves)
    _validate_against_base(produced, base_transformer)

    out_transformer = out / "transformer"
    out_transformer.mkdir(parents=True, exist_ok=True)
    mx.eval(list(produced.values()))
    mx.save_safetensors(str(out_transformer / "diffusion_pytorch_model.safetensors"), produced)
    shutil.copyfile(base_transformer / "config.json", out_transformer / "config.json")

    # Borrow the untouched components from the base klein snapshot. Symlink with
    # absolute targets so they survive the worker's temp->final atomic rename and
    # avoid duplicating multi-GB encoder/VAE weights on disk.
    for name in BORROWED_FILES:
        src_path = (base / name).resolve()
        dst = out / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copyfile(src_path, dst)
    for name in BORROWED_SUBDIRS:
        src_path = (base / name).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"base component missing: {src_path}")
        dst = out / name
        if dst.is_symlink() or dst.exists():
            dst.unlink() if dst.is_symlink() else shutil.rmtree(dst)
        os.symlink(src_path, dst)
    return str(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a FLUX.2-klein single-file transformer to a diffusers model dir.")
    parser.add_argument("--source-file", required=True, help="Original single-file transformer safetensors (bf16).")
    parser.add_argument("--base-dir", required=True, help="Installed base FLUX.2-klein-9B diffusers snapshot dir.")
    parser.add_argument("--out-dir", required=True, help="Output diffusers model dir to assemble.")
    args = parser.parse_args(argv)
    try:
        path = convert_and_assemble(args.source_file, args.base_dir, args.out_dir)
    except Exception as exc:  # surface a single-line error for the worker log
        sys.stderr.write(f"[mlx_flux_convert] ERROR: {exc}\n")
        return 1
    sys.stderr.write(f"[mlx_flux_convert] assembled diffusers model dir at {path}\n")
    print(json.dumps({"path": path}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
