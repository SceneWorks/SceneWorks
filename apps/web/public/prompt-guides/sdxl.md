# Stable Diffusion XL Prompt Guide

## Best For

General-purpose text-to-image across photography, illustration, concept art, and design. SDXL base 1.0 is the open, widely-supported foundation with the largest LoRA and finetune ecosystem — if you want to layer community style/character LoRAs, this is the model. CreativeML OpenRAIL++-M, commercial use OK, ungated.

SDXL uses two CLIP text encoders and **real classifier-free guidance**, so you get both a positive *and* a negative prompt. Native resolution is 1024×1024.

## Prompt Shape

SDXL responds well to both fluent natural-language descriptions and comma-separated tag lists — and to a blend of the two. A reliable structure:

`subject + key details + style + composition + lighting + quality tags`

CLIP encoders weight earlier tokens more heavily, so **lead with the subject and the most important attributes**. Keep the prompt focused; very long prompts dilute attention.

## Build The Prompt

### Subject

State the subject plainly and front-load it:

`a weathered fisherman mending a net on a wooden dock at dawn`

### Details

Add material, texture, and atmosphere:

- `hand-stitched leather`
- `condensation on cold glass`
- `drifting morning fog`
- `intricate filigree pattern`

### Style

- `editorial fashion photography`
- `cinematic film still, 35mm`
- `flat vector illustration`
- `soft gouache painting`

### Camera And Composition

- `low-angle hero shot`
- `extreme close-up macro`
- `wide establishing shot`
- `rule of thirds, centered subject`

### Lighting

- `golden hour backlight`
- `soft window light`
- `neon-lit night scene`
- `dramatic rim lighting`

### Quality Tags

SDXL was trained with aesthetic and quality signals, so a few trailing tags help:

`highly detailed, sharp focus, 8k, professional photography`

## Negative Prompts

SDXL honors a negative prompt (guidance > 1). Use it to push away common failure modes — keep it short and targeted:

`blurry, lowres, deformed, extra fingers, bad anatomy, watermark, text, jpeg artifacts, oversaturated`

## Tips

- ~30 steps at guidance 7.0 is a solid baseline; raise guidance for stronger prompt adherence, lower it (4–6) for more natural, less "baked" results.
- Native 1024×1024; the canonical aspect-ratio buckets (1152×896, 896×1152, 1216×832, 832×1216, 1344×768, 768×1344) are trained resolutions — prefer them over arbitrary sizes.
- Front-load the subject and key attributes; CLIP weights early tokens more.
- Layer community SDXL LoRAs for specific styles or characters — SDXL has the deepest LoRA ecosystem of any open model.
- Use the negative prompt actively; it is one of SDXL's strongest levers.

## Example Prompts

`A cozy independent bookstore storefront at dusk, warm interior glow spilling onto a rain-slick cobblestone street, cinematic shallow depth of field, highly detailed, sharp focus.`

`Studio product shot of a matte-black ceramic pour-over coffee set on pale oak, soft diffused side light, subtle steam rising, minimalist composition, professional photography, high detail.`

## Sources

- [SDXL base 1.0 model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [SDXL technical report](https://arxiv.org/abs/2307.01952)
- [Diffusers SDXL pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl)
