# Iris 3B Prompt Guide

## Best For

Photographic and illustrative text-to-image at about one megapixel. Iris 3B is a 3-billion-parameter
**pixel-space** model: the network paints every pixel directly, with no VAE and no latent space, so
fine texture (skin, film grain, mosaics, fabric) is not filtered through a lossy decoder.
Apache-2.0, ungated, native MLX on Apple Silicon.

## How It Works

The generation install is two pieces, both pinned and downloaded together:

- **Iris backbone** — a hybrid dual-/single-stream diffusion transformer that predicts pixels
  directly on a 16-pixel patch grid, sampled with a FlowDPM-Solver++ over a shifted
  rectified-flow schedule.
- **Qwen3-VL-4B-Instruct** — the text encoder. Iris reads twelve of its intermediate layers rather
  than only the last one, so detailed descriptive prompts carry through well.

## Settings

| Setting | Default | What it does |
|---|---|---|
| Steps | 100 | Denoising steps. Fewer is faster but loses some fine detail; 100 is the release default. |
| Guidance (CFG) | 3.0 | How strictly the image follows the prompt. Higher follows more literally but can look harsher. 1.0 turns guidance off. |
| Seed | random | Fix it to reproduce an image. Variations use seed, seed+1, … |
| Negative prompt | empty | Things you do not want. It is the guidance unconditional, so it only applies while guidance is above 1.0 — the field is disabled when guidance is 1.0. |

There is one integrator, so there is no sampler or scheduler choice, and Iris takes no LoRAs or
quantization tiers.

## Prompt Shape

Write plain, descriptive English sentences. Put the subject first, then setting, lighting,
materials and style:

> Black and white portrait of a fisherman with a thick grey beard and deep wrinkles, piercing
> eyes, overcast light, fine grain film photograph.

> A glass sculpture of a heart filled with flowers, caustics and reflections, 3D render.

## Resolution

- **Default 1024×1024.** The release renders at about one megapixel; the Studio presets cover the
  common aspect ratios at that size.
- **Dimensions must be multiples of 16** (the patch size), from 16 up to 2048 per side.
- Memory and time grow with pixel count: every pixel is a model output, so a 2048² image is four
  times the work of 1024².

## Sources

- [Iris-3B model card](https://huggingface.co/speridlabs/iris-3b)
- [Iris-3B code](https://github.com/speridlabs/iris-3b)
- [Iris-3B paper (arXiv 2610.09450)](https://arxiv.org/abs/2610.09450)
