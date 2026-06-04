# sc-2758 — SDXL acceleration A/B (LCM vs Lightning vs Hyper): findings + locked defaults

**Epic 2755 S3.** Gates the torch impl (sc-2759/2760/2761) and feeds the already-merged MLX impl
re-tune (sc-2769 → sc-2907). Engine-agnostic deliverable: the per-variant defaults table below.

## Method

- Stack: the **packaged SceneWorks worker venv** (torch 2.8.0, diffusers 0.39.0.dev0, peft 0.19.1,
  insightface 1.0.1), **Apple M5 Max / MPS**, fp16, 1024², seed 42 — i.e. the exact runtime the
  torch impl (sc-2760) will use.
- Bases: `sdxl` (`stabilityai/stable-diffusion-xl-base-1.0`) and `realvisxl`
  (`SG161222/RealVisXL_V5.0`). Prompts: a fox scene + a photoreal portrait. Identical prompt/seed
  across every method.
- Scripts (reproducible): [`sc2758_sdxl_acceleration_spike.py`](sc2758_sdxl_acceleration_spike.py)
  (the A/B grids) and [`sc2758_instantid_accel_spike.py`](sc2758_instantid_accel_spike.py)
  (InstantID identity). Artifacts in `/tmp/sc2758_sdxl_accel/` + `/tmp/sc2758_instantid_accel/`.
- The few-step schedulers live only in diffusers (the worker `sampler_registry` is flow-matching
  only and does not carry `LCMScheduler` — confirmed). So each method sets its scheduler directly,
  exactly as the torch `sdxl_diffusers` adapter will (sc-2760).

## LOCKED per-variant defaults (engine-agnostic — `sdxl` + `realvisxl` are identical)

| Variant       | Scheduler (diffusers, `from_config(base_scheduler.config)`)                | Default steps | Step range            | CFG (`guidance_scale`) | Acceleration LoRA (download-on-demand)                                              |
|---------------|----------------------------------------------------------------------------|:-------------:|-----------------------|:----------------------:|------------------------------------------------------------------------------------|
| **Standard**  | base `EulerDiscreteScheduler`                                              | 30            | 20–40                 | 7.0                    | — (none)                                                                            |
| **LCM**       | `LCMScheduler`                                                             | **8**         | 4–8 (one LoRA, any N) | **1.0** (≤4 → try 1.5–2.0) | `latent-consistency/lcm-lora-sdxl` / `pytorch_lora_weights.safetensors`           |
| **Lightning** | `EulerDiscreteScheduler` + `timestep_spacing="trailing"`                   | **4**         | 2 / 4 / 8 (LoRA-bound)| **1.0** (CFG off)      | `ByteDance/SDXL-Lightning` / `sdxl_lightning_{N}step_lora.safetensors`              |
| **Hyper**     | `TCDScheduler` (pass `eta=0.0` at call)                                    | **4**         | 1 / 2 / 4 / 8 (LoRA-bound) | **1.0** (CFG off)  | `ByteDance/Hyper-SD` / `Hyper-SDXL-{N}step(s)-lora.safetensors`                     |

Hard rules for the implementation:
- **Lightning & Hyper bind a step-specific LoRA**: `num_inference_steps` MUST match the LoRA's step
  grade (the 4-step LoRA at 4 steps, etc.). **LCM uses one LoRA for any step count** (true
  consistency model) — its `steps` is a free slider.
- **CFG off** (`guidance_scale` ≤ 1.0) for all three distilled methods. This both matches their
  training and halves the work (one UNet forward/step instead of two) — part of the speedup.
- **Hyper scheduler = TCD** (locked). At 4 steps `TCDScheduler(eta=0)` and
  `DDIMScheduler(timestep_spacing="trailing")` are visually equivalent; TCD is chosen because it is
  portable and matches the already-merged MLX impl (sc-2769). The official Hyper **1-step**
  `DDIMScheduler` + `timesteps=[800]` recipe is **NOT** usable on diffusers 0.39 (`set_timesteps`
  rejects custom schedules) — use TCD at 1 step if 1-step is ever exposed.
- Base SDXL acceleration LoRAs apply cleanly to **RealVisXL** (cross-checkpoint compat confirmed) —
  one LoRA set covers both bases.

## GO / NO-GO per method

| Method        | Verdict                         | Why |
|---------------|---------------------------------|-----|
| **Lightning** | **GO (strong)**                 | Sharpest few-step output; 4-step ≈ standard quality; coherent down to 2 steps. |
| **Hyper**     | **GO (strong)**                 | Equal to Lightning; widest range (a usable 1-step); 4-step the sweet spot. Matches merged MLX TCD path. |
| **LCM**       | **GO, lowest quality**          | Usable but visibly softer / lower-contrast. Needs **8 steps** (or 4 + CFG≈1.5–2). Ship as the broad-compat single-LoRA option, not the quality option. **NO-GO under InstantID (below).** |

## InstantID under acceleration — ArcFace cosine(reference, generated), RealVisXL + Kelsie ref

| Config            | Steps | ArcFace ↑  | Wall  | Verdict |
|-------------------|:-----:|:----------:|:-----:|---------|
| standard (Euler)  | 30    | 0.781      | 20.8s | baseline |
| **Lightning**     | 4     | **0.799**  | 2.0s  | **GO** — identity holds, *above* baseline, ~10× |
| **Lightning**     | 8     | 0.789      | 4.7s  | GO |
| **Hyper (TCD)**   | 4     | **0.802**  | 2.0s  | **GO** — best ArcFace, ~10× |
| **Hyper (TCD)**   | 8     | 0.793      | 3.3s  | GO |
| **LCM**           | 8     | **0.552**  | 3.3s  | **NO-GO** — identity collapses |

- **InstantID + Lightning/Hyper → GO.** Identity is carried by the per-step IdentityNet ControlNet +
  IP-Adapter conditioning, which the structure-faithful Lightning/Hyper schedulers preserve at few
  steps. 4-step is marginally *above* the 30-step baseline at ~10× speedup.
- **InstantID + LCM → NO-GO.** LCM's characteristic softening washes out the high-frequency facial
  detail ArcFace keys on; cosine drops to 0.55.

## Timing (MPS / M5 Max, 1024², mean across sdxl+realvisxl × 2 prompts)

| Variant | steps | mean wall | speedup vs 30-step |
|---|:--:|:--:|:--:|
| standard | 30 | 19.9s | 1.0× |
| LCM | 8 | 3.6s | 5.6× |
| LCM | 4 | 2.3s | 8.7× |
| Lightning | 2 | 1.4s | 14.0× |
| Lightning | 4 | 1.9s | 10.3× |
| Lightning | 8 | 3.0s | 6.6× |
| Hyper | 1 | 1.2s | 16.2× |
| Hyper | 4 | 1.9s | 10.4× |
| Hyper | 8 | 3.3s | 6.0× |

These are MPS-relative; Windows/CUDA absolute numbers differ, but the **method ranking and the
steps/CFG/scheduler/LoRA defaults are engine-agnostic** — that's what S4–S6 + sc-2769 consume.

## What downstream stories should take from this

- **sc-2759** (builtin LoRAs): repos/files confirmed above; the Lightning/Hyper entries are
  step-graded (multiple files), LCM is single-file. Flag "applying these as a plain style LoRA
  without the matching scheduler/steps degrades output" (true — proven here).
- **sc-2760** (torch scheduler paths): implement the three `from_config` scheduler builders above;
  Hyper needs `eta=0.0` threaded into the call kwargs.
- **sc-2761** (torch variant ids): default steps/CFG per the table; Lightning/Hyper variants bind a
  specific step-LoRA.
- **sc-2907** (MLX re-tune of sc-2769): align defaults — Hyper scheduler **TCD** ✔ (already matches);
  raise **LCM default from 4 → 8** steps (4 is too soft as a default); Lightning/Hyper default **4**.
- **InstantID acceleration** (new, if pursued): Hyper-4 or Lightning-4 only; never LCM.
