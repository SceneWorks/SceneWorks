"""sc-1533 spike — native MLX LoRA backprop through the LTX-2.3 transformer.

Hard go/no-go gate for epic 1532 (MLX Video LoRA Training). Answers: can we train a
LoRA *natively in MLX* against the `mlx_video` LTX transformer as shipped?

It does NOT load the 19B/48-layer Q4 checkpoint. The genuinely unknown risk is whether
`mlx.nn.value_and_grad` flows through *this LTX architecture* (patchify -> AdaLN-single ->
SPLIT RoPE -> gated attention -> cross-attn -> feed-forward -> output proj). A tiny model
with the real structural flags but small dims exercises the identical code path in seconds,
and we additionally quantize it (QLoRA) to cover the frozen-quantized-base case.

Run with the SceneWorks desktop venv interpreter:
  "/Users/michael/Library/Application Support/SceneWorks/python/venv/bin/python" \
      scripts/spikes/ltx_mlx_lora_autodiff_spike.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from mlx_video.generate import create_position_grid
from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType
from mlx_video.models.ltx.ltx import LTXModel
from mlx_video.models.ltx.transformer import Modality

TARGET_MODULE = "transformer_blocks.0.attn1.to_q"
RANK = 4
ALPHA = 8.0


def tiny_config() -> LTXModelConfig:
    """A small AudioVideo LTX config that keeps every *structural* flag of the real
    distilled model (SPLIT rope + double precision, gated attention, AdaLN coeff 6,
    middle-indices grid) but shrinks dims. inner_dim = 4*16 = 64 so nn.quantize works
    at group_size 64. cross_attention_dim must equal inner_dim (the caption projection
    emits inner_dim and cross-attn consumes it).

    Production (generate_av) uses an AudioVideo model and runs video-only by passing
    audio=None — the VideoOnly LTXModel.__call__ path mis-calls its single-modality
    preprocessor, so we faithfully mirror production with the AudioVideo type. Audio
    dims are kept tiny since the audio branch is never run at forward (audio=None)."""
    return LTXModelConfig(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=4,
        attention_head_dim=16,   # inner_dim = 64
        in_channels=64,
        out_channels=64,
        num_layers=2,
        cross_attention_dim=64,  # == inner_dim
        caption_channels=64,
        apply_gated_attention=True,
        # Audio branch (built but unused — audio=None at forward). Keep tiny.
        audio_num_attention_heads=4,
        audio_attention_head_dim=16,
        audio_in_channels=64,
        audio_out_channels=64,
        audio_cross_attention_dim=64,
        audio_caption_channels=64,
        audio_positional_embedding_max_pos=[20],
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=True,
        positional_embedding_max_pos=[20, 2048, 2048],
        use_middle_indices_grid=True,
        timestep_scale_multiplier=1000,
        norm_eps=1e-6,
    )


def synthetic_video_modality(cfg: LTXModelConfig, *, dtype=mx.float32):
    """Mirror mlx_video.generate.denoise's per-step Modality construction for a single
    latent frame (still image) at a tiny 4x4 latent resolution."""
    b, c = 1, cfg.in_channels
    f, h, w = 1, 4, 4
    num_tokens = f * h * w
    latent = mx.random.normal((b, c, f, h, w)).astype(dtype)
    latents_flat = mx.transpose(mx.reshape(latent, (b, c, -1)), (0, 2, 1))  # (B, tokens, C)
    timesteps = mx.full((b, num_tokens), 0.5, dtype=dtype)
    positions = create_position_grid(b, f, h, w)  # (B, 3, tokens, 2) float32
    context = mx.random.normal((b, 8, cfg.caption_channels)).astype(dtype)  # (B, ctx, caption_ch)
    return Modality(
        latent=latents_flat,
        timesteps=timesteps,
        positions=positions,
        context=context,
        context_mask=None,
        enabled=True,
    )


class TrainLoRALinear(nn.Module):
    """Frozen base (nn.Linear or nn.QuantizedLinear) + trainable rank-r LoRA, matching
    the inference loader's math: out = base(x) + (x @ A.T @ B.T) * (alpha/rank).
    A is [rank, in], B is [out, rank]; B zero-init so the adapter starts as identity."""

    def __init__(self, base: nn.Module, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_a = mx.random.normal((rank, in_features)) * 0.02
        self.lora_b = mx.zeros((out_features, rank))

    def __call__(self, x: mx.array) -> mx.array:
        delta = (x @ self.lora_a.T) @ self.lora_b.T
        return self.base(x) + delta * self.scale


def _get_module(model: nn.Module, path: str) -> nn.Module:
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set_module(model: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def build_lora_model(*, quantize: bool):
    cfg = tiny_config()
    model = LTXModel(cfg)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())

    if quantize:
        # QLoRA: freeze the base as a quantized layer, train fp LoRA on top.
        nn.quantize(model.transformer_blocks[0].attn1, group_size=64, bits=4)
        mx.eval(model.parameters())

    inner_dim = cfg.inner_dim  # to_q is square (inner_dim, inner_dim)
    base = _get_module(model, TARGET_MODULE)
    wrapper = TrainLoRALinear(base, inner_dim, inner_dim, RANK, ALPHA)
    _set_module(model, TARGET_MODULE, wrapper)

    # Freeze everything, then unfreeze only the two LoRA arrays on the wrapper.
    model.freeze()
    _get_module(model, TARGET_MODULE).unfreeze(recurse=False, keys=["lora_a", "lora_b"])
    return model, cfg


def run_training_case(name: str, *, quantize: bool) -> bool:
    print(f"\n=== Case: {name} (quantize={quantize}) ===")
    model, cfg = build_lora_model(quantize=quantize)

    trainable = tree_flatten(model.trainable_parameters())
    trainable_keys = [k for k, _ in trainable]
    print(f"  trainable params: {trainable_keys}")
    only_lora = trainable_keys and all("lora_" in k for k in trainable_keys)
    print(f"  PASS only-LoRA-trainable: {only_lora}")

    modality = synthetic_video_modality(cfg)
    target = mx.random.normal((1, 16, cfg.out_channels))

    def loss_fn(m):
        vx, _ = m(video=modality, audio=None)
        return mx.mean((vx - target) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=1e-2)

    losses = []
    grad_finite = True
    lora_grad_nonzero = False
    for step in range(4):
        loss, grads = loss_and_grad(model)
        mx.eval(loss, grads)
        losses.append(float(loss))
        for k, g in tree_flatten(grads):
            gmax = float(mx.max(mx.abs(g)))
            if not (gmax == gmax and gmax != float("inf")):  # NaN/inf guard
                grad_finite = False
            if gmax > 0:
                lora_grad_nonzero = True
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    print(f"  losses: {[round(x, 6) for x in losses]}")
    decreased = losses[-1] < losses[0]
    print(f"  PASS grads finite: {grad_finite}")
    print(f"  PASS lora grads non-zero (some step): {lora_grad_nonzero}")
    print(f"  PASS loss decreased: {decreased} ({losses[0]:.6f} -> {losses[-1]:.6f})")

    # Save the trained adapter in the inference-loader format and round-trip it.
    roundtrip = roundtrip_adapter(model, cfg)

    ok = bool(only_lora and grad_finite and lora_grad_nonzero and decreased and roundtrip)
    print(f"  CASE RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def roundtrip_adapter(model: nn.Module, cfg: LTXModelConfig) -> bool:
    """Save {module}.lora_A.weight / .lora_B.weight (+ alpha) and reload via the
    inference path, then apply to a fresh unwrapped model."""
    from mlx_video.lora import LoRAConfig, apply_loras_to_model, load_lora_weights, load_multiple_loras

    wrapper = _get_module(model, TARGET_MODULE)
    state = {
        f"{TARGET_MODULE}.lora_A.weight": wrapper.lora_a.astype(mx.float32),
        f"{TARGET_MODULE}.lora_B.weight": wrapper.lora_b.astype(mx.float32),
        f"{TARGET_MODULE}.alpha": mx.array(float(ALPHA), dtype=mx.float32),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spike_lora.safetensors"
        mx.save_safetensors(str(path), state)

        loaded = load_lora_weights(path)
        lw = loaded.get(TARGET_MODULE)
        parsed_ok = lw is not None and lw.rank == RANK and abs(lw.alpha - ALPHA) < 1e-6
        print(f"  PASS loader parsed adapter (rank={lw.rank if lw else None}, alpha={lw.alpha if lw else None}): {parsed_ok}")

        fresh = LTXModel(tiny_config())
        fresh.set_dtype(mx.float32)
        mx.eval(fresh.parameters())
        module_to_loras = load_multiple_loras([LoRAConfig(path=path, strength=1.0)])
        applied = apply_loras_to_model(fresh, module_to_loras, verbose=True)
        print(f"  PASS apply_loras_to_model matched modules (applied={applied}): {applied >= 1}")
        return bool(parsed_ok and applied >= 1)


REAL_REPO = "notapalindrome/ltx23-mlx-av-q4"


def real_config(caption_channels: int, audio_caption_channels: int):
    """The production AudioVideo config (mirrors generate_av.py for the Q4 distilled
    repo: caption projection off -> caption_channels = connector heads*head_dim = 4096,
    gated attention on -> adaln_embedding_coefficient = 9)."""
    return LTXModelConfig(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=32,
        attention_head_dim=128,   # inner_dim = 4096
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=caption_channels,
        caption_projection_first_linear=False,
        caption_projection_second_linear=False,
        adaln_embedding_coefficient=9,
        apply_gated_attention=True,
        audio_num_attention_heads=32,
        audio_attention_head_dim=64,
        audio_in_channels=128,
        audio_out_channels=128,
        audio_cross_attention_dim=2048,
        audio_caption_channels=audio_caption_channels,
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=True,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        audio_positional_embedding_max_pos=[20],
        use_middle_indices_grid=True,
        timestep_scale_multiplier=1000,
    )


def run_real_weights_case() -> bool:
    """Load the real cached Q4 AudioVideo LTX transformer and confirm one forward+backward
    flows finite grads to a LoRA injected into a real attn1.to_q. Offline (HF_HUB_OFFLINE)."""
    import mlx.nn as _nn
    from mlx_video.generate_av import load_unified_weights
    from mlx_video.utils import get_model_path

    print("\n=== Case: REAL Q4 AudioVideo checkpoint ===")
    model_path = Path(get_model_path(REAL_REPO))
    print(f"  model_path: {model_path}")

    sanitized = load_unified_weights(model_path, "transformer.")
    cfg = real_config(caption_channels=32 * 128, audio_caption_channels=32 * 64)
    model = LTXModel(cfg)

    # Selective quantization exactly as generate_av does (split_model.json says 4-bit/64).
    quantized_paths = {k.rsplit(".", 1)[0] for k in sanitized if k.endswith(".scales")}

    def _should_quantize(path: str, module: _nn.Module) -> bool:
        return isinstance(module, _nn.Linear) and path in quantized_paths

    _nn.quantize(model, group_size=64, bits=4, class_predicate=_should_quantize)
    model.load_weights(list(sanitized.items()), strict=False)
    mx.eval(model.parameters())
    del sanitized
    mx.clear_cache()
    print(f"  loaded transformer; peak so far {mx.get_peak_memory()/1e9:.2f} GB")

    base = _get_module(model, TARGET_MODULE)
    base_type = type(base).__name__
    is_linear = isinstance(base, (nn.Linear, nn.QuantizedLinear))
    print(f"  target {TARGET_MODULE} is {base_type} (real Linear/QuantizedLinear: {is_linear})")
    inner_dim = cfg.inner_dim
    _set_module(model, TARGET_MODULE, TrainLoRALinear(base, inner_dim, inner_dim, RANK, ALPHA))
    model.freeze()
    _get_module(model, TARGET_MODULE).unfreeze(recurse=False, keys=["lora_a", "lora_b"])

    # Small still-image latent (1 frame, 8x8) + random context at caption_channels dim.
    b, c, f, h, w = 1, cfg.in_channels, 1, 8, 8
    num_tokens = f * h * w
    latent = mx.random.normal((b, c, f, h, w)).astype(mx.bfloat16)
    latents_flat = mx.transpose(mx.reshape(latent, (b, c, -1)), (0, 2, 1))
    modality = Modality(
        latent=latents_flat,
        timesteps=mx.full((b, num_tokens), 0.5, dtype=mx.bfloat16),
        positions=create_position_grid(b, f, h, w),
        context=mx.random.normal((b, 16, cfg.caption_channels)).astype(mx.bfloat16),
        context_mask=None,
        enabled=True,
    )
    target = mx.random.normal((b, num_tokens, cfg.out_channels)).astype(mx.bfloat16)

    def loss_fn(m):
        vx, _ = m(video=modality, audio=None)
        return mx.mean((vx.astype(mx.float32) - target.astype(mx.float32)) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=1e-2)
    losses, grad_finite, lora_grad_nonzero = [], True, False
    for _ in range(2):
        loss, grads = loss_and_grad(model)
        mx.eval(loss, grads)
        losses.append(float(loss))
        for _k, g in tree_flatten(grads):
            gmax = float(mx.max(mx.abs(g)))
            if not (gmax == gmax and gmax != float("inf")):
                grad_finite = False
            if gmax > 0:
                lora_grad_nonzero = True
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    print(f"  losses: {[round(x, 6) for x in losses]}")
    print(f"  PASS grads finite: {grad_finite}")
    print(f"  PASS lora grads non-zero: {lora_grad_nonzero}")
    decreased = losses[-1] < losses[0]
    print(f"  PASS loss changed: {decreased} ({losses[0]:.6f} -> {losses[-1]:.6f})")
    print(f"  peak memory: {mx.get_peak_memory()/1e9:.2f} GB")

    ok = bool(is_linear and grad_finite and lora_grad_nonzero and decreased)
    print(f"  CASE RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    if "--real" in sys.argv:
        ok = run_real_weights_case()
        print(f"\n  REAL-WEIGHTS GO/NO-GO: {'GO' if ok else 'NO-GO'}")
        return
    print("LTX MLX LoRA autodiff spike (sc-1533)")
    print(f"mlx {mx.__version__ if hasattr(mx, '__version__') else '?'}")

    # Sanity: confirm the dict-based transformer_blocks expose the expected module path.
    probe = LTXModel(tiny_config())
    names = [n for n, _ in probe.named_modules()]
    print(f"  target module '{TARGET_MODULE}' in named_modules: {TARGET_MODULE in names}")
    del probe

    results = {
        "bf16 base": run_training_case("bf16 base", quantize=False),
        "quantized base (QLoRA)": run_training_case("quantized base (QLoRA)", quantize=True),
    }

    peak_gb = mx.get_peak_memory() / 1e9
    print("\n================ SUMMARY ================")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"  peak memory: {peak_gb:.3f} GB")
    go = all(results.values())
    print(f"\n  GO/NO-GO: {'GO — native MLX LoRA training is viable' if go else 'NO-GO — investigate failing case'}")


if __name__ == "__main__":
    main()
