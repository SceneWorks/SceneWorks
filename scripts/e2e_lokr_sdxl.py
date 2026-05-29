"""sc-2199 e2e: real SDXL LoKr train -> save -> inference round trip.

Exercises the actual production code paths on real cached SDXL weights:
  - build_peft_network_config (trainer) attaches a real LoKrConfig to the UNet
  - write_lokr_adapter (trainer) serializes it with routing metadata
  - apply_loras_to_pipeline (the exact entrypoint SdxlImageAdapter calls) detects
    the LoKr file, injects it via PEFT (NOT load_lora_weights), and generates.

Confirms: file routes as lokr, injection applies a real delta to generation,
and a no-LoRA baseline differs from the LoKr render. Smoke quality only
(512px / few steps) — this validates plumbing, not aesthetics.
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, "apps/worker")
sys.path.insert(0, "packages/shared")

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from peft.utils import get_peft_model_state_dict
import peft

from scene_worker.training_adapters import build_peft_network_config, write_lokr_adapter, read_run_config
from scene_worker.lora_adapters import apply_loras_to_pipeline, adapter_network_type

REPO = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16  # bf16 mandatory on MPS for SDXL (fp16 -> NaN)
OUT = "/tmp/lokr_sdxl"
os.makedirs(OUT, exist_ok=True)
PROMPT = "a photo of a teddy bear riding a skateboard"


def load_pipe():
    pipe = StableDiffusionXLPipeline.from_pretrained(REPO, torch_dtype=DTYPE, variant="fp16", use_safetensors=True)
    return pipe.to(DEVICE)


def gen(pipe, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = pipe(PROMPT, num_inference_steps=4, guidance_scale=0.0, height=512, width=512, generator=g).images[0]
    return np.asarray(img).astype(np.float32)


print(f"[device={DEVICE} dtype={DTYPE}]  loading SDXL for training...")
train_pipe = load_pipe()
unet = train_pipe.unet
unet.requires_grad_(False)

plan = {"config": {"rank": 16, "alpha": 16, "advanced": {
    "networkType": "lokr", "decomposeFactor": -1,
    "loraTargetModules": ["to_q", "to_k", "to_v", "to_out.0"],
}}}
cfg = read_run_config(plan)
unet.add_adapter(build_peft_network_config(peft, cfg))
unet.train()
trainable = [p for p in unet.parameters() if p.requires_grad]
print(f"  LoKr trainable tensors: {len(trainable)}")

# Tiny fixed-batch overfit so the adapter clearly changes the UNet output.
torch.manual_seed(0)
lat = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=DTYPE)
t = torch.tensor([500], device=DEVICE)
ehs = torch.randn(1, 77, 2048, device=DEVICE, dtype=DTYPE)
add_text = torch.randn(1, 1280, device=DEVICE, dtype=DTYPE)
add_time = torch.tensor([[512, 512, 0, 0, 512, 512]], device=DEVICE, dtype=DTYPE)
kw = {"added_cond_kwargs": {"text_embeds": add_text, "time_ids": add_time}}
target = torch.randn_like(unet(lat, t, ehs, **kw).sample)
opt = torch.optim.AdamW(trainable, lr=1e-3)
for i in range(25):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(unet(lat, t, ehs, **kw).sample.float(), target.float())
    loss.backward(); opt.step()
    if i in (0, 24):
        print(f"  step {i:2d} loss={loss.item():.4f}")

sd = get_peft_model_state_dict(unet)
path = write_lokr_adapter(sd, OUT, "bear_lokr.safetensors", rank=cfg.rank, alpha=cfg.alpha,
                          decompose_factor=cfg.decompose_factor, target_modules=cfg.lora_target_modules)
size = os.path.getsize(path)
print(f"  saved {path}  ({size/1024:.1f} KB, {len(sd)} tensors)  networkType={adapter_network_type(path)}")
del train_pipe, unet
import gc; gc.collect()
if DEVICE == "mps":
    torch.mps.empty_cache()

print("loading a FRESH SDXL pipeline for inference (no training state)...")
infer_pipe = load_pipe()
base_img = gen(infer_pipe, seed=1)

# THE production entrypoint SdxlImageAdapter uses:
lora = {"id": "bear_lokr", "installedPath": path, "weight": 1.0, "family": "sdxl"}
state = apply_loras_to_pipeline(infer_pipe, [lora], adapter_id="sdxl", model_family="sdxl")
print(f"  apply_loras_to_pipeline -> adapter_names={state.adapter_names}")
assert state.adapter_names, "no adapter applied"
# Prove it injected as a PEFT adapter on the unet (not via load_lora_weights).
print(f"  unet.peft_config adapters: {list(getattr(infer_pipe.unet, 'peft_config', {}).keys())}")

lokr_img = gen(infer_pipe, seed=1)
mad = float(np.abs(lokr_img - base_img).mean())
from PIL import Image
Image.fromarray(base_img.astype(np.uint8)).save(f"{OUT}/base.png")
Image.fromarray(lokr_img.astype(np.uint8)).save(f"{OUT}/lokr.png")
print(f"  mean abs pixel diff (base vs lokr): {mad:.3f}  -> {'DELTA APPLIED' if mad > 1.0 else 'NO VISIBLE DELTA'}")
print("E2E OK" if (state.adapter_names and mad > 1.0) else "E2E FAILED")
