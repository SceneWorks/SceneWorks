# Qwen Image 2.1 Prompt Guide

## Best For

Large-canvas composition, posters and covers, illustration, realistic photography, and layouts with
readable text. 2.1 renders natively at 2048 px square and up to 2752 px on its long edge, so it
suits work where the detail has to survive at print size rather than being upscaled afterwards.

This is a different model from **Qwen Image** (the 2512 weights) in the same catalog — a different
checkpoint with a different text encoder. Prompts do transfer between the two, but seeds, LoRAs and
exact framing do not.

## Prompt Shape

Use the Qwen-style structure:

`subject + setting + style + camera + atmosphere + detail modifiers`

For quick ideas, the shorter shape still works:

`subject + setting + style`

## Build The Prompt

### Subject

Lead with the main subject: count, type or age, action, pose, clothing, expression, and whatever
makes it distinctive.

Good: `a ceramic fox figurine wearing a tiny raincoat, standing beside a puddle`

### Setting

Say where it is and what surrounds it — location, time of day, weather, background depth.

Good: `on a rain-slicked cobblestone street at dusk, shop windows glowing behind`

### Style

Name a medium and a treatment: `editorial photograph`, `gouache illustration`, `1970s film still`,
`technical isometric render`.

### Camera

2.1 responds to photographic direction: lens length, aperture, angle, distance.

Good: `85mm lens, shallow depth of field, low three-quarter angle`

### Text In The Image

Put the exact characters in quotation marks and say where they go and how they are set. Keep the
string short — a headline, not a paragraph.

Good: `a bookshop sign reading "NIGHT EDITION" in condensed serif capitals, centred above the door`

## Negative Prompts

2.1 is a true-CFG model: the negative prompt and the guidance scale work **together**. With the
guidance scale at its default of 1.0, classifier-free guidance is effectively off and a negative
prompt has little to push against. Raise the guidance scale as soon as you start using one.

A negative prompt is for what must be absent, not for quality adjectives:
`extra fingers, watermark, cropped text, motion blur`.

## Settings

| Control | Default | Notes |
| --- | --- | --- |
| Steps | 40 | The model's own default. Below ~20 the fine detail and text start to break down. |
| Guidance | 1.0 | True CFG. Leave at 1.0 for no negative branch; 3–5 is a reasonable band once you add one. |
| Size | 2048 x 2048 | Seven presets, from 1536 x 2752 (9:16) to 2752 x 1536 (16:9). |
| Seed | random | Fix it to iterate on a prompt without the composition moving underneath you. |

## Licence

Qwen-Image 2.1 ships under the **Qwen RESEARCH LICENSE AGREEMENT**: the weights are licensed for
research and evaluation use only, and commercial use needs a separate licence from Alibaba. You
accept these terms before the download starts; the full text is under **About → Licenses**.
