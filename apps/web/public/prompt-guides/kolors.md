# Kolors Prompt Guide

## Best For

Photorealistic text-to-image with strong prompt following in **both English and Chinese**, and notably accurate text rendering — including Chinese characters. Kolors (Kwai-Kolors) is Apache-2.0 and commercial-safe, built on a ChatGLM3 text encoder with an SDXL-style UNet.

## Prompt Shape

Unlike guidance-distilled models, Kolors uses **real classifier-free guidance**, so it has both a positive *and* a negative prompt:

`subject + setting + visual details + style + composition + lighting + any text`

Write the positive prompt as a fluent description of the finished image, and use the negative prompt to push away unwanted qualities. Recommended defaults: **~25 steps at guidance 5.0** (the model card also suggests 5.0–6.5).

## Build The Prompt

### Subject

Describe the subject as if captioning a finished photo — type, action, position, distinguishing features.

Good: `a young chef plating a dessert in a bright modern kitchen`

### Details

Add material, texture, and atmosphere:

- `glossy chocolate ganache`
- `steam rising from fresh espresso`
- `soft natural window light`
- `intricate porcelain pattern`

### Style

- `editorial food photography`
- `cinematic film still, 35mm`
- `flat vector illustration`
- `traditional ink wash painting`

### Camera And Composition

- `low-angle hero shot`
- `extreme close-up macro`
- `wide establishing shot`
- `centered product shot, studio backdrop`

### Lighting

- `golden hour backlight`
- `soft diffused studio light`
- `neon-lit night scene`
- `dramatic rim lighting`

### Text In Images

Kolors renders text well in both scripts — quote the exact words and describe the medium:

`a wooden cafe sign reading "晨光咖啡" in warm hand-painted brush lettering`

`a vintage enamel poster reading "MORNING LIGHT" in cream serif letters`

## Negative Prompts

Keep negatives short and targeted. Common starting points:

`blurry, lowres, deformed, extra fingers, watermark, text artifacts, oversaturated`

## Chinese Prompting

Kolors understands Chinese natively — you can write the whole prompt in Chinese, mix Chinese and English, or request Chinese text rendering directly. Chinese prompts often produce stronger results for culturally specific subjects.

## Tips

- ~25 steps at guidance 5.0 is the sweet spot; raise guidance toward 6.5 for stronger prompt adherence.
- Use the negative prompt — it is active, unlike distilled models.
- 1024×1024 is the native sweet spot; portrait/landscape buckets work well too.

## Example Prompts

`A cozy independent bookstore storefront at dusk, warm interior glow spilling onto a rain-slick cobblestone street, a hand-painted sign reading "PAGE & QUILL" in gold script, reflections in the wet pavement, cinematic shallow depth of field.`

`一只橘猫坐在窗台上，窗外是黄昏的城市天际线，柔和的逆光，电影感，高质量，细节丰富.`

## Sources

- [Kolors model card](https://huggingface.co/Kwai-Kolors/Kolors)
- [Kolors diffusers checkpoint](https://huggingface.co/Kwai-Kolors/Kolors-diffusers)
- [Diffusers Kolors pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/kolors)
