"""sc-10610: build a tiered MLX turnkey (`q4/ q8/ bf16/`) from an SDXL checkpoint.

Every SDXL member of the catalog (`sdxl`, `realvisxl`, `realvisxl_lightning`) loads from a
`SceneWorks/*-mlx` turnkey: three tier subdirs, each a self-contained diffusers tree the worker
resolves via `image_jobs::base::standard_tier_subdir` (manifest `mlx.standardTierLayout: true`).
Until now no committed tool produced them — the recipe survived only as prose in the
`SceneWorks/realvisxl-mlx` README, and every prior turnkey was built from an upstream that was
*already* a diffusers repo. Illustrious-XL (epic 10609) is the first that is not: OnomaAI ship a
single-file LDM/A1111 checkpoint, and neither `mlx-gen-sdxl` nor `candle-gen-sdxl` can read one.

Two legs:

  1. LDM single-file -> diffusers component tree, via `StableDiffusionXLPipeline.from_single_file`.
     Skipped when `--source` is already a diffusers dir.
  2. diffusers tree -> `bf16/ q8/ q4/` tiers.

The tier spec below was *derived by reading the published `SceneWorks/realvisxl-mlx`*, not from the
README — which is wrong in three places (it claims the quantized set is chosen by module name, that
the VAE is f32-only, and that dense tensors follow the source dtype). What actually holds:

  quantize iff  `k.endswith(".weight") and w.ndim == 2 and "embeddings." not in k`
      Rank alone separates Linears (rank 2) from norms (rank 1) and convs (rank 4). The two CLIP
      embedding matrices are rank 2 but are gather lookups, not matmuls, so they stay dense.
      Verified: 743 / 72 / 193 tensors for unet / text_encoder / text_encoder_2 — exactly the
      published set.

  packing      `mx.quantize(w.astype(mx.bfloat16), group_size=64, bits={4,8})`
      Reproduces all 743 published UNet triples (`weight` u32, `scales`/`biases` bf16) bit-for-bit.
      The bfloat16 cast is load-bearing: it is what `mlx-gen`'s load-time `nn.quantize` does.

  dense residue is f32 in a quantized tier; the `bf16/` tier is f16 for unet + both text encoders
  and f32 for the VAE. `bf16` names a TIER, not a dtype — nothing in it is bfloat16.

  a quantized tier ships the VAE TWICE: f32 as `<stem>.safetensors` and f16 as
  `<stem>.fp16.safetensors`. `mlx-gen-sdxl::resolve_weight_file` probes `.fp16` first when it wants
  f16 and the plain name otherwise, so both spellings must exist for either request to resolve.

Precision note. The published tiers were cut from upstream's f32 master, while their `bf16/` tier is
a verbatim copy of upstream's `*.fp16.safetensors`. So rebuilding a quantized tier *from the bf16
tier* cannot reproduce the dense residue bit-for-bit: 60 of the UNet's 937 dense tensors carry
elements in f16's subnormal range (5.96e-8 .. 6.1e-5), where f16 holds fewer than 11 mantissa bits.
The quantized tensors are unaffected (bfloat16 is coarser still). `verify` classifies this rather
than failing on it — see `--dense-atol`. Feed the f32 master when you have one; Illustrious has only
an f16 single-file, so for it upcast-from-f16 *is* the highest-precision source that exists.

Usage:
    python scripts/build_sdxl_turnkey.py build \
        --source ~/.cache/.../Illustrious-XL-v1.0.safetensors \
        --out ./illustrious-xl-v1-mlx
    python scripts/build_sdxl_turnkey.py verify --built ./out --reference <published-snapshot>

Runs in a torch+diffusers+mlx venv (macOS/Apple Silicon); `~/sceneworks-pytorch-harness/.venv`
already has torch/diffusers/transformers — add `mlx` to it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

#: Component subdir -> weight-file stem. VAE is the only one that stays dense in every tier.
COMPONENTS: dict[str, str] = {
    "unet": "diffusion_pytorch_model",
    "text_encoder": "model",
    "text_encoder_2": "model",
    "vae": "diffusion_pytorch_model",
}
#: Components whose rank-2 weights get packed in a quantized tier.
QUANTIZED_COMPONENTS = ("unet", "text_encoder", "text_encoder_2")
#: Copied verbatim into every tier alongside the component dirs.
STATIC_TREES = ("scheduler", "tokenizer", "tokenizer_2")
GROUP_SIZE = 64
#: safetensors `__metadata__` on dense files; quantized files carry none (matches the published tiers).
DENSE_METADATA = {"format": "pt"}
#: `unet/config.json` keys that pin the architecture. If a reference turnkey agrees with the converted
#: checkpoint on all of these, its descriptors describe the same UNet and are safe to adopt (sc-10666).
ARCH_KEYS = (
    "cross_attention_dim",
    "addition_time_embed_dim",
    "projection_class_embeddings_input_dim",
    "block_out_channels",
    "transformer_layers_per_block",
    "attention_head_dim",
    "down_block_types",
    "up_block_types",
    "in_channels",
    "out_channels",
)


def quantizable(key: str, weight: mx.array) -> bool:
    """The published selection rule: rank-2 `.weight`, excluding the CLIP embedding lookups."""
    return key.endswith(".weight") and weight.ndim == 2 and "embeddings." not in key


def resolve_config_source(dense: Path, reference: Path | None) -> Path:
    """Which tree the tier descriptors are copied from: a known-good turnkey, or the converted one.

    Returns `reference` once it is proven to describe the same UNet, else `dense`. Deliberately does
    not mutate `dense` — with `--source <diffusers-dir>` that is the caller's own tree (sc-10666).

    `from_single_file` reconstructs *weights* faithfully but emits descriptors that misdescribe them:
    the text encoders get `CLIPModel` configs (`model_type: "clip"` + a nested `text_config`) though
    the stored weights are a `CLIPTextModel`; the scheduler is written as `EulerDiscreteScheduler`
    where the family uses `DDIMScheduler`; `vae/config.json` leaks diffusers' internal
    `_name_or_path: "../sdxl-vae/"`; `unet/config.json` says `upcast_attention: null` not `false`.

    None of it reaches our engines — `mlx-gen-sdxl` and `candle-gen-sdxl` hardcode the SDXL configs and
    never read these files — which is exactly why it went unnoticed. It reaches everything else:
    `transformers`/`diffusers` loading the published repo directly, and the ComfyUI external-roots
    lane (epic 10451). A published artifact must not misdescribe itself just because our loader
    doesn't look.

    Every SDXL turnkey shares one architecture, so these descriptors are architecture-only and
    identical across the family. Adopt the reference's verbatim — but only after proving the two
    really are the same UNet. Importing a config for a *different* architecture would be a far worse
    bug than the one this fixes.
    """
    if reference is None:
        return dense
    converted_unet = json.loads((dense / "unet" / "config.json").read_text())
    reference_unet = json.loads((reference / "unet" / "config.json").read_text())
    mismatched = {
        key: (converted_unet.get(key), reference_unet.get(key))
        for key in ARCH_KEYS
        if converted_unet.get(key) != reference_unet.get(key)
    }
    if mismatched:
        detail = "\n".join(f"    {k}: converted={c!r} reference={r!r}" for k, (c, r) in mismatched.items())
        raise SystemExit(
            f"--reference-configs {reference} describes a different UNet than the converted "
            f"checkpoint; refusing to adopt its descriptors:\n{detail}"
        )
    missing = [
        str(path)
        for path in [reference / "model_index.json", *(reference / c / "config.json" for c in COMPONENTS)]
        if not path.is_file()
    ]
    if missing:
        raise SystemExit(f"--reference-configs {reference} is missing descriptors: {', '.join(missing)}")
    print(f"[configs] adopting descriptors from {reference} ({len(ARCH_KEYS)} arch keys agree)", flush=True)
    return reference


def load_component(root: Path, subdir: str, stem: str) -> dict[str, mx.array]:
    """Load a component's weights, preferring the highest-precision form present.

    A diffusers repo may ship `<stem>.safetensors` (f32 master), a SHARDED master
    (`<stem>.safetensors.index.json` + `<stem>-0000N-of-0000M.safetensors`), and/or
    `<stem>.fp16.safetensors`. Always take a master when one exists: a quantized tier's dense residue
    is f32, and upcasting a f16 variant cannot recover what f16 already rounded away.

    Sharding is not hypothetical — `save_pretrained` splits any component over its 10 GB shard
    threshold, which an f32 SDXL UNet (~10.3 GB) crosses.
    """
    component = root / subdir
    single = component / f"{stem}.safetensors"
    if single.is_file():
        return mx.load(str(single))

    index = component / f"{stem}.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        weights: dict[str, mx.array] = {}
        for shard in sorted(set(weight_map.values())):
            weights.update(mx.load(str(component / shard)))
        missing = set(weight_map) - set(weights)
        if missing:
            raise SystemExit(f"{subdir}: {len(missing)} tensors named in the index are not in any shard")
        return weights

    half = component / f"{stem}.fp16.safetensors"
    if half.is_file():
        return mx.load(str(half))
    raise SystemExit(f"{subdir}: no {stem} weights (single, sharded, or .fp16) under {root}")


def write(path: Path, tensors: dict[str, mx.array], metadata: dict[str, str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), tensors, metadata=metadata or {})


def emit_dense(weights: dict[str, mx.array], dtype: mx.Dtype) -> dict[str, mx.array]:
    return {k: v.astype(dtype) for k, v in weights.items()}


def emit_quantized(weights: dict[str, mx.array], bits: int) -> dict[str, mx.array]:
    """Pack rank-2 non-embedding weights; everything else lands dense at f32."""
    out: dict[str, mx.array] = {}
    for key, value in weights.items():
        if quantizable(key, value):
            packed, scales, biases = mx.quantize(
                value.astype(mx.bfloat16), group_size=GROUP_SIZE, bits=bits
            )
            stem = key[: -len(".weight")]
            out[key] = packed
            out[f"{stem}.scales"] = scales
            out[f"{stem}.biases"] = biases
        else:
            out[key] = value.astype(mx.float32)
    return out


def copy_static(source: Path, tier: Path) -> None:
    for tree in STATIC_TREES:
        if (source / tree).is_dir():
            shutil.copytree(source / tree, tier / tree, dirs_exist_ok=True)
    if (source / "model_index.json").is_file():
        shutil.copy2(source / "model_index.json", tier / "model_index.json")
    for subdir in COMPONENTS:
        config = source / subdir / "config.json"
        if config.is_file():
            (tier / subdir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(config, tier / subdir / "config.json")


def build_tier(source: Path, out: Path, tier: str, config_source: Path | None = None) -> None:
    """Emit one tier subdir: weights from `source`, descriptors from `config_source` (default `source`)."""
    dest = out / tier
    copy_static(config_source or source, dest)
    bits = {"q4": 4, "q8": 8}.get(tier)

    for subdir, stem in COMPONENTS.items():
        weights = load_component(source, subdir, stem)

        if subdir == "vae":
            # Dense in every tier. Quantized tiers additionally ship the f16 spelling so a f16
            # request resolves without falling back (see `resolve_weight_file`).
            write(dest / subdir / f"{stem}.safetensors", emit_dense(weights, mx.float32), DENSE_METADATA)
            if bits is not None:
                write(
                    dest / subdir / f"{stem}.fp16.safetensors",
                    emit_dense(weights, mx.float16),
                    DENSE_METADATA,
                )
        elif bits is None:
            write(dest / subdir / f"{stem}.fp16.safetensors", emit_dense(weights, mx.float16), DENSE_METADATA)
        else:
            write(dest / subdir / f"{stem}.safetensors", emit_quantized(weights, bits), None)

        n = len(weights)
        print(f"  {tier}/{subdir}: {n} source tensors -> written", flush=True)
        del weights
        mx.clear_cache()


def leg1_single_file_to_diffusers(checkpoint: Path, out: Path) -> Path:
    """LDM/A1111 single file -> f32 diffusers component tree."""
    import torch
    from diffusers import StableDiffusionXLPipeline

    print(f"[leg 1] from_single_file({checkpoint.name}) -> {out}", flush=True)
    pipe = StableDiffusionXLPipeline.from_single_file(str(checkpoint), torch_dtype=torch.float32)
    # One file per component: `save_pretrained` otherwise shards anything over 10 GB, which the f32
    # SDXL UNet (~10.3 GB) crosses. `load_component` reads shards too, but a single file is cheaper.
    pipe.save_pretrained(str(out), safe_serialization=True, max_shard_size="1TB")
    del pipe
    return out


def cmd_build(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        dense = leg1_single_file_to_diffusers(source, out / "_dense")
    elif source.is_dir():
        dense = source
    else:
        raise SystemExit(f"--source is neither a file nor a directory: {source}")

    reference = Path(args.reference_configs).expanduser() if args.reference_configs else None
    config_source = resolve_config_source(dense, reference)

    for tier in args.tiers:
        print(f"[leg 2] {tier}", flush=True)
        build_tier(dense, out, tier, config_source)

    if args.prune_dense and source.is_file():
        shutil.rmtree(dense)
    print(f"\nwrote {', '.join(args.tiers)} to {out}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Diff a built turnkey against a published one, classifying dense drift separately."""
    built = Path(args.built).expanduser()
    ref = Path(args.reference).expanduser()
    failures = 0

    for tier in args.tiers:
        for subdir, stem in COMPONENTS.items():
            for name in (f"{stem}.safetensors", f"{stem}.fp16.safetensors"):
                b, r = built / tier / subdir / name, ref / tier / subdir / name
                if not r.is_file():
                    continue
                if not b.is_file():
                    print(f"FAIL {tier}/{subdir}/{name}: missing in build")
                    failures += 1
                    continue

                bt, rt = mx.load(str(b)), mx.load(str(r))
                if set(bt) != set(rt):
                    only_b, only_r = set(bt) - set(rt), set(rt) - set(bt)
                    print(f"FAIL {tier}/{subdir}/{name}: key mismatch (+{len(only_b)} -{len(only_r)})")
                    failures += 1
                    continue

                packed = {k for k in rt if k.endswith((".scales", ".biases"))}
                packed |= {k[: -len(".scales")] + ".weight" for k in rt if k.endswith(".scales")}
                exact = drift = broken = 0
                worst = 0.0
                for k in rt:
                    if mx.array_equal(bt[k], rt[k]):
                        exact += 1
                        continue
                    if k in packed or bt[k].dtype != rt[k].dtype:
                        broken += 1  # a packed tensor must match bit-for-bit
                        continue
                    delta = float(mx.max(mx.abs(bt[k].astype(mx.float32) - rt[k].astype(mx.float32))))
                    worst = max(worst, delta)
                    if delta <= args.dense_atol:
                        drift += 1
                    else:
                        broken += 1
                verdict = "FAIL" if broken else "ok"
                if broken:
                    failures += 1
                print(
                    f"{verdict:4s} {tier}/{subdir}/{name}: exact={exact} "
                    f"dense-drift={drift} (max {worst:.2e}) broken={broken}"
                )
                del bt, rt
                mx.clear_cache()

    print("\nPASS" if not failures else f"\n{failures} file(s) FAILED")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="single-file or diffusers dir -> tiered turnkey")
    b.add_argument("--source", required=True, help="LDM single-file .safetensors, or a diffusers dir")
    b.add_argument("--out", required=True)
    b.add_argument("--tiers", nargs="+", default=["bf16", "q8", "q4"], choices=["bf16", "q8", "q4"])
    b.add_argument("--prune-dense", action="store_true", help="delete the leg-1 dense tree afterwards")
    b.add_argument(
        "--reference-configs",
        help="tier dir of a known-good SDXL turnkey (e.g. <realvisxl-mlx>/bf16) whose component "
        "configs, scheduler, tokenizers and model_index.json are adopted verbatim. Aborts unless its "
        "unet/config.json agrees with the converted checkpoint on every architecture key. Use this "
        "when publishing: `from_single_file` emits descriptors that misdescribe the weights (sc-10666).",
    )
    b.set_defaults(fn=cmd_build)

    v = sub.add_parser("verify", help="diff a built turnkey against a published one")
    v.add_argument("--built", required=True)
    v.add_argument("--reference", required=True)
    v.add_argument("--tiers", nargs="+", default=["bf16", "q8", "q4"], choices=["bf16", "q8", "q4"])
    v.add_argument(
        "--dense-atol",
        type=float,
        default=0.0,
        help="tolerate dense drift up to this absolute delta; use 1e-7 when the build source is an "
        "f16 mirror but the reference was cut from an f32 master. Packed tensors always require "
        "bit-equality.",
    )
    v.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
