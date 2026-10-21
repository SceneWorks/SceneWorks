# Supra2-IMG Prompt Guide

## Best For

Quick, small text-to-image sketches. Supra2-IMG is a compact 104-million-parameter rectified-flow
diffusion transformer that renders at exactly **256×256**, so it is fast and light on memory, but
it is not a detail or text-rendering model. Native MLX on Apple Silicon.

## How It Works

The install is three pinned upstream pieces, downloaded together and then converted on your Mac:

- **Supra2-IMG denoiser** (SupraLabs, Apache-2.0) — the DiT checkpoint, converted once at install
  into a native bundle and checked against its pinned checksum.
- **Flan-T5-Base** (Google, Apache-2.0) — the text encoder. Prompts are read up to 128 tokens; a
  longer prompt is truncated.
- **SD VAE ft-MSE** (Stability AI, MIT) — decodes the latent into the final image.

After install, generation needs no network access.

## Settings

| Setting | Default | What it does |
|---|---|---|
| Steps | 50 | Euler steps over a uniform flow schedule. Fewer is faster but softer. |
| Guidance (CFG) | 3.0 | How strictly the image follows the prompt. 1.0 or below turns guidance off. |
| Seed | Random (the release's reference default is 0) | Set one to reproduce an image. Variations render together in one pass on seed, seed+1, …, so a single render on seed + *k* reproduces image *k* of a batch. |

There is no negative prompt (guidance steers against the model's own stored unconditional
embedding), no sampler or scheduler choice, no LoRAs and no quantization tiers.

## Prompting Tips

- Describe one clear subject and a simple setting: "a red fox sitting in fresh snow, soft morning
  light".
- Short, concrete phrases work better than long paragraphs at this size.
- Style words ("watercolor", "pixel art", "studio photo") steer the overall look.
