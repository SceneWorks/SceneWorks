"""sc-2194 spike: LoKr round-trip on a real (tiny, config-only) diffusers model.

Retires the model-agnostic risks for epic 2193 without a 20 GB Z-Image download:
  1. Does peft.LoKrConfig attach to a diffusers transformer via add_adapter?
  2. Does it actually train (loss drops)?
  3. What key format / file size does get_peft_model_state_dict produce, vs LoRA?
  4. Can a LoKr adapter be loaded back for INFERENCE via the PEFT-injection path
     (inject_adapter_in_model + set_peft_model_state_dict) and reproduce the delta?
  5. Structurally, why can't diffusers load_lora_weights() consume it?
"""

import copy
import io

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from peft import LoKrConfig, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer  # noqa: F401  (sanity import)
from peft import inject_adapter_in_model
from safetensors.torch import save as st_save

torch.manual_seed(0)
DEVICE = "cpu"  # deterministic for allclose
TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
RANK = 8
ALPHA = 8


def build_unet():
    torch.manual_seed(0)
    return UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=8,
    ).to(DEVICE)


def fixed_batch():
    g = torch.Generator().manual_seed(42)
    sample = torch.randn(1, 4, 16, 16, generator=g)
    ts = torch.tensor([10])
    ehs = torch.randn(1, 4, 32, generator=g)
    return sample, ts, ehs


def fwd(model, batch):
    sample, ts, ehs = batch
    return model(sample, ts, ehs).sample


def state_bytes(sd):
    # mirror how trainers serialize (safetensors); measure on-disk footprint
    return len(st_save({k: v.contiguous() for k, v in sd.items()}))


def param_count(sd):
    return sum(v.numel() for v in sd.values())


def train(model, batch, steps=60):
    target = torch.randn_like(fwd(model, batch))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-2)
    first = last = None
    for i in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(fwd(model, batch), target)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    return first, last, len(trainable)


def run(kind):
    print(f"\n===== {kind.upper()} =====")
    pristine = build_unet()
    base_batch = fixed_batch()
    with torch.no_grad():
        base_out = fwd(pristine, base_batch).clone()

    model = copy.deepcopy(pristine)
    model.requires_grad_(False)
    if kind == "lokr":
        cfg = LoKrConfig(r=RANK, alpha=ALPHA, target_modules=TARGETS, decompose_factor=-1,
                         init_weights=True)
    else:
        cfg = LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=TARGETS,
                         init_lora_weights="gaussian")
    model.add_adapter(cfg)

    first, last, n = train(model, base_batch)
    print(f"  trainable tensors={n}  loss {first:.4f} -> {last:.4f}  ({'DROPS' if last < first else 'NO DROP'})")

    sd = get_peft_model_state_dict(model)
    print(f"  saved tensors={len(sd)}  params={param_count(sd):,}  bytes={state_bytes(sd):,}")
    sample_keys = list(sd.keys())[:4]
    print(f"  sample keys: {sample_keys}")

    with torch.no_grad():
        trained_out = fwd(model, base_batch).clone()
    delta_vs_base = (trained_out - base_out).abs().mean().item()

    # ---- INFERENCE RELOAD via PEFT injection (NOT load_lora_weights) ----
    fresh = copy.deepcopy(pristine)
    inject_adapter_in_model(cfg, fresh)
    missing = set_peft_model_state_dict(fresh, sd)
    with torch.no_grad():
        reloaded_out = fwd(fresh, base_batch).clone()
    reload_match = torch.allclose(trained_out, reloaded_out, atol=1e-5)
    print(f"  delta vs base (mean abs)={delta_vs_base:.5f}")
    print(f"  inference reload reproduces delta: {reload_match}  "
          f"(max diff {(trained_out - reloaded_out).abs().max().item():.2e})")
    print(f"  set_peft_model_state_dict report: {missing}")
    return {"bytes": state_bytes(sd), "params": param_count(sd), "keys": sample_keys,
            "reload_match": reload_match, "delta": delta_vs_base}


lokr = run("lokr")
lora = run("lora")

print("\n===== SUMMARY =====")
print(f"  LoRA  params={lora['params']:,}  bytes={lora['bytes']:,}")
print(f"  LoKr  params={lokr['params']:,}  bytes={lokr['bytes']:,}")
ratio = lora["bytes"] / max(lokr["bytes"], 1)
print(f"  LoKr is {ratio:.2f}x smaller on disk than LoRA at r={RANK}")
print(f"  LoKr inference-reload OK: {lokr['reload_match']}   LoRA inference-reload OK: {lora['reload_match']}")
print("\n  Why load_lora_weights() can't consume LoKr:")
print(f"    LoRA keys look like:  {lora['keys'][0]}")
print(f"    LoKr keys look like:  {lokr['keys'][0]}")
print("    diffusers' converter expects lora_A/lora_B (or kohya lora_down/lora_up);")
print("    LoKr emits lokr_w1/lokr_w2/lokr_t2 -> unrecognized -> must use PEFT injection.")
