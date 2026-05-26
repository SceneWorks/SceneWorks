# FLUX.1 [schnell] Prompt Guide

## Best For

Fast, high-quality text-to-image generations: photography, illustration, concept art, and notably strong, legible text rendering. [schnell] is the distilled, ~4-step variant — Apache-2.0 and commercial-safe — so it favors speed while keeping FLUX's excellent prompt following.

## Prompt Shape

FLUX was trained on rich natural-language captions, so write a fluent sentence (or two) describing the finished image rather than a list of tags:

`subject + setting + visual details + style + composition + lighting + any text`

FLUX is guidance-distilled, so it does **not** use a CFG / negative prompt — put everything you want into the positive prompt and leave guidance at 0.

## Build The Prompt

### Subject

Describe the subject as if captioning a finished photo. Include type, action, position, and distinguishing features.

Good: `a weathered fisherman mending a net on a wooden dock at dawn`

### Details

Add material, texture, and atmosphere:

- `hand-stitched leather`
- `condensation on cold glass`
- `drifting morning fog`
- `intricate filigree pattern`

### Style

Use photographic or art-direction terms:

- `editorial fashion photography`
- `cinematic film still, 35mm`
- `flat vector illustration`
- `soft gouache painting`

### Camera And Composition

- `low-angle hero shot`
- `extreme close-up macro`
- `wide establishing shot`
- `centered product shot, studio backdrop`

### Lighting

- `golden hour backlight`
- `soft window light`
- `neon-lit night scene`
- `dramatic rim lighting`

### Text In Images

FLUX renders text well — quote the exact words and describe the medium:

`a vintage enamel sign reading "HARBOR CAFE" in cream serif letters`

## Tips

- Keep the prompt descriptive and positive; there is no negative prompt.
- ~4 steps is the sweet spot — more steps rarely help [schnell].
- For maximum quality (more steps, guidance control), use FLUX.1 [dev].

## Example Prompts

`A cozy independent bookstore storefront at dusk, warm interior glow spilling onto a rain-slick cobblestone street, a hand-painted sign reading "PAGE & QUILL" in gold script, reflections in the wet pavement, cinematic shallow depth of field.`

`A studio product shot of a matte-black ceramic pour-over coffee set on a pale oak table, soft diffused side light, subtle steam rising, minimalist composition, high detail.`

## Sources

- [FLUX.1 [schnell] model card](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [Diffusers Flux pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux)
