# Mage-Flow Prompt Guide

## Best For

Native-resolution text-to-image and instruction-based image editing from one compact 4B stack.
Mage-Flow generates any aspect from **512 to 2048 px per side** (including wide 4:1 panoramas)
without bucketing, and its editing siblings apply a written instruction to a source image while
preserving everything you do not mention. MIT-licensed, ungated.

Use a **generation** variant (Base / RL / Turbo) for text-to-image, and an **edit** variant
(Edit-Base / Edit / Edit-Turbo) when you have a source image and want to change it.

## How It Works

Mage-Flow pairs three components, all pulled in one install:

- **Mage-VAE** — a one-step codec with 128-channel, 16×-downsampled latents (compact latents mean
  a 1024² image is only ~4,096 tokens; a 2048² image ~16,384).
- **NR-MMDiT** — a 4B native-resolution multimodal diffusion transformer trained with rectified
  flow, so it renders any resolution/aspect in the 512–2048 range directly rather than through
  fixed size buckets.
- **Qwen3-VL-4B** — the text encoder for prompts, and (on the edit path) the vision encoder that
  reads your source and reference images.

All six published checkpoints share the same architecture and the same text encoder / VAE; only
the transformer weights and the recommended step/guidance defaults differ.

## Variants

| Variant | Task | Steps | Guidance (CFG) | When to pick |
|---|---|---|---|---|
| **Base** | Generate | 30 | 5.0 | Undistilled foundation behavior; the training target. |
| **RL** | Generate | 20 | 5.0 | Reward-optimized; the recommended everyday generator. |
| **Turbo** | Generate | 4 | 1.0 (off) | Fast local generation; distilled, no classifier-free guidance. |
| **Edit-Base** | Edit | 30 | 5.0 | Foundation instruction editor with true CFG. |
| **Edit** | Edit | 30 | 5.0 | Reward-optimized instruction editor; the recommended editor. |
| **Edit-Turbo** | Edit | 4 | 1.0 (off) | Fast four-step editing. |

The catalog seeds each variant's recommended steps and guidance as defaults — they are a good
starting point. On the CFG variants, **lower guidance loosens** the interpretation and **higher
guidance follows the text more strictly**. The Turbo and Edit-Turbo variants are distilled for
four-step sampling with guidance off; longer schedules and a non-zero CFG do not help them, so keep
those prompts explicit because there are fewer refinement steps to recover detail.

## Prompt Shape For Generation

Mage-Flow responds best to direct, concrete descriptions. Put the main subject first, then the
setting, composition, lighting, materials, and visual style:

`subject + setting + composition + lighting + materials + style`

> Editorial photograph of a weathered red fishing boat tied to a quiet stone harbor at dawn,
> low mist over the water, soft blue ambient light with warm cabin lamps, 50 mm lens,
> eye-level composition, natural texture and restrained color grading.

## Prompt Shape For Editing

On an edit variant your prompt is an **instruction describing the change**, not a description of
the whole scene. The model keeps everything you do not mention:

- `Change the background to a snowy mountain at dusk; keep the subject and pose unchanged.`
- `Replace the green shirt with a navy turtleneck.`
- `Add warm golden-hour light coming from the left.`

**Multiple reference images** are supported: supply more than one reference and describe how they
combine (for example, "place the subject from the first image into the setting of the second").
The edit variants read your source/reference images through the Qwen3-VL vision encoder, so
naming what is in them ("the person on the left", "the red car") helps the model target the edit.

## Resolution And Aspect Ratio

- **Default 1024×1024.** This is the trained center and the safest starting point.
- **Native range 512–2048 px per side**, any aspect ratio, including very wide 4:1 panoramas —
  Mage-Flow was trained on variable resolutions, so off-square and large canvases are first-class,
  not upscales.
- **Dimensions must be multiples of 16** (the 16× VAE downsample). The Studio resolution presets
  already satisfy this.
- **Memory scales with pixel count.** A 2048² image is ~4× the latent tokens of 1024² and its VAE
  decode alone can need several GB on top of the model, so the largest canvases want the most
  unified memory. If a large canvas will not fit, drop the resolution or pick a lower quant tier
  before reducing quality elsewhere.

## Quant Tiers

Each variant offers three load-time quantization tiers over the same installed weights:

- **q4** (default) — lowest memory, the widest device compatibility.
- **q8** — higher fidelity, more memory.
- **bf16** — maximum fidelity, the most memory.

The tier is chosen at load time from one download — you do not download a separate copy per tier.
Start at **q4**; move up to **q8** or **bf16** if you have the headroom and want the extra
fidelity. If a tier will not fit your machine it is gated with a clear message rather than crashing.

## Tips

- **Generation vs. editing are different models.** Pick a `*-Edit*` variant only when you have a
  source image; the plain variants are text-to-image and ignore a source.
- **Turbo is for speed, not schedules.** Do not raise its step count or guidance — it is distilled
  to converge in four steps with CFG off.
- **Wide panoramas** (e.g. 2048×512, 4:1) render natively — describe a horizontal composition
  ("a continuous panoramic view of …") rather than expecting the model to stitch tiles.
- **Editing preserves the unmentioned.** Keep instructions surgical; the more of the scene you
  re-describe, the more the model is free to change.

## Sources

- [Mage-Flow Base model card](https://huggingface.co/microsoft/Mage-Flow-Base)
- [Mage-Flow (RL) model card](https://huggingface.co/microsoft/Mage-Flow)
- [Mage-Flow Turbo model card](https://huggingface.co/microsoft/Mage-Flow-Turbo)
- [Mage-Flow Edit model card](https://huggingface.co/microsoft/Mage-Flow-Edit)
