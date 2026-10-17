# FLUX 3 Image Klein Prompt Guide

## Best For

Photographic, illustrative and typographic text-to-image at about one megapixel in 16 steps. FLUX 3
Image Klein is the distilled 5.6B checkpoint of Black Forest Labs' FLUX 3 Image family: a flow-map
transformer that jumps along its schedule without classifier-free guidance, conditioned on the
Qwen3-VL-4B-Instruct text encoder. Native MLX on Apple Silicon.

## How It Works

The install is one gated package, pinned and downloaded together:

- **Transformer** — the 5.6B flow-map DiT (bf16): five text / image blocks, then 28 shared blocks
  over the whole token stream, sampled in 16 Euler steps without CFG.
- **Autoencoder** — a 16× image autoencoder (8× VAE with 2×2 token packing, f32).
- **Qwen3-VL-4B-Instruct** — the text encoder bundled in the package (the same weights as the Qwen
  release). FLUX 3 reads eight of its intermediate layers, so detailed, descriptive prompts carry
  through well.

## Settings

- **Steps**: 16 (the release default). The distilled checkpoint is trained for its own schedule;
  more steps do not sharpen it.
- **Guidance**: none. The checkpoint samples without classifier-free guidance, so there is no
  guidance scale and no negative prompt — describe what you want, not what to avoid.
- **Size**: any multiple of 16 pixels on each side; 1024×1024 is the release default.
- **Seed**: reproducible per seed; a batch renders one image after another from `seed`, `seed + 1`, …

## Prompting

- Write a full sentence or short paragraph: subject, setting, lighting, lens or medium, mood.
- Text in the image: put the exact words in quotes and say where they go ("a poster with the
  headline "OPEN LATE" in bold red letters at the top").
- Style words work literally ("35mm film photograph", "risograph print", "watercolour").

## Licence

The weights are distributed under the FLUX Kommunity License v1.0: non-commercial use, with a
commercial-output exception for Qualifying Users (gross annualized revenue under US$5,000,000)
subject to its content-filter and AI-disclosure conditions. It is not a licence to train,
fine-tune, distil or redistribute the weights commercially. See the licence text from the model's
About panel.
