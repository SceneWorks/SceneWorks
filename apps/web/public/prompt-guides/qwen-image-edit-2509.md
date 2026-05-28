# Qwen Image Edit (2509) Prompt Guide

## Best For

Character Studio reference generation, localized image editing, and subject-consistency tasks where you want the **same person in a new scene, pose, or style** while preserving identity. The September 2509 iteration of Qwen-Image-Edit is built around the model card's headline use case: *"changing a person's pose while maintaining excellent identity consistency."* Apache-2.0, ungated.

For text-to-image only (no reference), use **Qwen Image**. For straightforward single-edit jobs without subject preservation, the earlier **Qwen Image Edit** (August) is interchangeable.

## How It Works

Unlike IP-Adapter models (Kolors, SDXL, FLUX), Qwen-Image-Edit doesn't blend a reference embedding into the diffusion process — it feeds the reference image into **two parallel encoders**:

- **Qwen2.5-VL** for visual *semantics* (who the subject is, what the scene contains)
- **VAE encoder** for visual *appearance* (color, texture, lighting)

The diffusion then renders the prompt while both encoders steer toward the reference. The slider is `trueCfgScale` (labeled **"Prompt strength"** in the picker) — and it does **not** trade identity for variation. The sc-2013 hardware spike measured ArcFace cosine 0.81 / 0.82 / 0.82 at trueCfgScale 2 / 4 / 6 on the same prompt and seed — face identity holds steady across the entire range. What the slider actually controls is how strictly the model follows the **prompt's scene/composition** vs the **reference's own surroundings**:

- **High `trueCfgScale` (5–6)** = prompt-dominant → the model leans into your scene description (new pose, new outfit, new setting) while keeping the subject's face/identity intact.
- **Low `trueCfgScale` (~1–2)** = reference-dominant → the output stays close to the reference's framing/styling (smaller scene changes, subtle prompt-driven adjustments).
- **Default 4.0** is the model-card sweet spot and the sc-2013 photoreal default.

If you want to loosen identity (different person, "inspired by" rather than "same character"), trueCfgScale won't get you there — switch to a resemblance-tier backbone (Kolors / SDXL / FLUX IP-Adapter).

## Prompt Shape For Character Studio

When you pick a character with an approved reference, the reference becomes the model's `image=` input — your prompt describes **what the same character is doing now**, not the reference itself.

Effective structure:

`same subject + new context + scene/lighting/composition + style anchor`

### Examples

`The same character at a sunlit beach café, reading a paperback, soft morning haze, candid editorial photograph, shallow depth of field.`

`The same person on a foggy New York street at night, wearing a long wool coat, neon storefront reflections, cinematic 35mm.`

`The same character in a sunlit kitchen, mid-laugh, holding a coffee mug, documentary photograph, warm window light.`

## Prompt Shape For Edit Mode

When using the model for localized edits (no character reference, just a source image), describe the **modification**, not the whole scene:

- `Remove the watermark in the bottom-right corner.`
- `Change the background to a snowy mountain at dusk; keep the subject and pose unchanged.`
- `Replace the green shirt with a navy turtleneck.`

The model preserves everything you don't mention.

## Tips

- **For Character Studio**: lead with "the same character/person/subject" so the model treats the reference as identity rather than as a scene to modify. Avoid "of the woman in the reference" — that often reproduces the reference's composition.
- **Negative prompts** are required for `trueCfgScale > 1` to function. Even an empty string is accepted; common defaults: `lowres, deformed, oversaturated, distorted face, watermark`.
- **Resolution**: 1024×1024 is the trained center; canonical aspect-ratio buckets (768×768, 1280×720, 720×1280) work well.
- **Don't fight the dual-control architecture**: long lists of "high quality, masterpiece, 8k" tags don't help much — the semantic+appearance encoders are doing that work from the reference.
- **Multi-image references** (Edit Plus pipeline only): supply multiple approved references for stronger identity averaging. Useful for invented characters with multiple hero shots.
- **trueCfgScale sweep**: try 2 / 4 / 6 to find the right *prompt adherence* — identity won't shift (the sc-2013 spike measured Δ0.011 cosine across the range), but scene/composition fidelity does. Photoreal characters often want ~4; stylized/painted characters often want ~3 (lets more of the reference's styling through).
- **Mac wait time**: at the model card's 50-step default this is ~16 minutes per image on MPS — the slowest engine in the picker by ~8×. Drop steps to 20 in advanced settings if you want sub-10-minute iteration (~7 min/image; small quality hit per the model card's 50-step recommendation).

## Comparison To Other Character Studio Backbones

| Backbone | Identity tier (mean ArcFace cosine on sc-2013 / sc-2009 / sc-2012 / sc-2015 spikes) | When to pick |
|---|---|---|
| **InstantID (RealVisXL)** | 0.876 — faithful face geometry (ArcFace + landmark ControlNet) | Highest-fidelity face likeness for real people; photoreal SDXL |
| **PuLID-FLUX** | 0.80 — faithful face geometry (PuLID cross-attention) | Same fidelity tier as InstantID, on FLUX's look |
| **Qwen Image Edit (2509)** | **0.75 — dual-control (semantic + appearance)** | **2nd-strongest face identity in the picker without a face-specialized engine.** Subject + outfit + setting continuity, varied poses/scenes, multi-image reference. Slowest engine on Mac (~16 min/image at 50 steps). |
| **Kolors / SDXL / RealVisXL IP-Adapter** | ~0.5 — resemblance (CLIP/face embed) | Scene-flexible "looks like" without faithful identity |
| **FLUX IP-Adapter** | ~0.5 — resemblance (XLabs CLIP-L) | Scene-flexible resemblance on FLUX's quality |
| **SenseNova-U1** | 0.33 — wardrobe + accessories preserved, face drifts | When outfit + props consistency matters more than face identity |

Qwen is the surprise face-identity engine of the epic — its dual-control architecture preserves face structure remarkably well without any explicit face-recognition signal. The tradeoffs vs InstantID/PuLID-FLUX: slightly lower identity fidelity (still 2nd-strongest), much slower on Mac, but no face detection step (works on stylized characters that ArcFace-based engines can't gate on).

## Sources

- [Qwen-Image-Edit-2509 model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- [Qwen-Image-Edit model card](https://huggingface.co/Qwen/Qwen-Image-Edit) (August iteration)
- [Diffusers QwenImage pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage)
