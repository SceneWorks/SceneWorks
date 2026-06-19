# Boogu Image Prompt Guide

## Best For

General-purpose **still images** from natural-language prompts — Boogu-Image-0.1 is a flow-matching
model (a ~10.3B DiT paired with a Qwen3-VL-8B condition encoder) that reads your prompt as plain
language, so describe the image the way you'd describe it to a person. It is especially strong at
**rendering legible text in the image** (signs, labels, posters) in both **English and Chinese**.

> Three variants ship: **Boogu Image** (full quality, true-CFG, ~50 steps), **Boogu Image Turbo**
> (few-step, ~4 steps, much faster — best for iteration), and **Boogu Image Edit** (instruction-driven
> editing of a source image). They share prompting style; pick the variant by speed vs. task.

## Task Modes

| Mode | Variant | What it does | Inputs you provide |
|---|---|---|---|
| **Text → Image** | Boogu Image / Turbo | Generates an image from the prompt alone. | Prompt |
| **Image → Image (Edit)** | Boogu Image Edit | Follows a text instruction to edit a source image, keeping its overall content. | Source image + instruction |

Tips per mode:

- **Text → Image:** start with **Turbo** to explore compositions quickly, then switch to the full
  **Boogu Image** for the final render when you want maximum quality.
- **Edit:** write the prompt as an *instruction* about what to change ("make it night", "add a red
  scarf", "turn the sign into a neon sign reading OPEN"), not a full re-description of the image. The
  source provides the content; the instruction provides the change.

## Prompt Shape

`subject + scene + composition + camera/framing + aesthetic/style`

Boogu reads the whole prompt as intent, so natural descriptive language works better than a pile of
disconnected tags.

## Build The Prompt

### Subject

Name the main subject and its visible traits (color, material, clothing, expression). One or two clear
subjects render more coherently than a crowded frame.

### Scene Details

Describe background, foreground, lighting, weather, and time of day. A focused scene reads cleaner.

### Text In The Image

Boogu renders typography well. Put the exact words in quotes and say where they go:

- `a storefront with a neon sign that reads "OPEN 24 HOURS"`
- `a book cover titled "山海经" in elegant brush calligraphy`

Keep the quoted string short and explicit; long paragraphs of in-image text degrade.

### Composition & Framing

Direct framing language works: `low angle`, `wide shot`, `medium close-up`, `centered subject`,
`rule of thirds`, `shallow depth of field`, `bokeh background`.

### Style

Concise style labels work well: `cinematic`, `photorealistic`, `watercolor`, `flat illustration`,
`studio portrait`, `film noir`.

### Negative Prompt (full Boogu Image only)

The full model uses true classifier-free guidance, so a negative prompt helps reduce artifacts:
`blurry, low quality, distorted hands, extra limbs, watermark, jpeg artifacts, garbled text`. Turbo is
CFG-free — the negative prompt and guidance scale have no effect there.

## Quality & Speed Notes

- **Resolution:** use a native bucket (1024² / 768×1024 / 1024×768 / 1280×720 / 720×1280); width and
  height must be multiples of 16. 1024² is the default.
- **Steps:** Boogu Image ~25–50 (more = cleaner, slower); Boogu Image Turbo ~4 (few-step distilled —
  more steps rarely help).
- **Guidance:** Boogu Image ~2–5 (default 4); Turbo ignores guidance (CFG-free).
- **Quantization:** Q8 is the default (~23 GB, fits a 64 GB-class Mac). The full-precision bf16 build
  is also hosted for maximum quality if you have the memory.
- **Count:** keep batches modest while iterating.

## Example Prompts

`A weathered lighthouse on a rocky cliff at golden hour, waves breaking below, gulls in the distance.
Wide shot, low angle, warm directional light, shallow depth of field. Photorealistic, cinematic.`

`A cozy bakery storefront at dusk, warm window glow, a chalkboard sign that reads "FRESH BREAD".
Medium shot, centered, soft bokeh. Photorealistic.`

`Edit: change the season to winter — snow on the rooftops, bare trees, cold blue light — keep the
buildings and street layout.`

## Sources

- [Boogu-Image-0.1 (Hugging Face)](https://huggingface.co/Boogu)
- [SceneWorks/boogu-image-mlx (MLX weights)](https://huggingface.co/SceneWorks/boogu-image-mlx)
