# Qwen Image 2.1 Prompt Guide

## Best For

Large-canvas composition, posters and covers, illustration, realistic photography, and layouts with
readable text. 2.1 renders natively at 2048 px square and up to 2752 px on its long edge, so it
suits work where the detail has to survive at print size rather than being upscaled afterwards.

This is a different model from **Qwen Image** (the 2512 weights) in the same catalog — a different
checkpoint with a different text encoder. Prompts do transfer between the two, but seeds and exact
framing do not.

**Adapters are not supported for this model in this release.** The engine refuses LoRA and LoKr
outright, and the catalog advertises no compatible family, so the Studio offers none — a Qwen Image
LoRA will not load here. Everything below is prompt-side.

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

## Editing with reference images

2.1 has **one pipeline**. Text-to-image is that pipeline with no reference attached; editing,
combining and local changes are the *same* call with **1–10 reference images**. There is no separate
edit model, no strength slider, and no inpainting mode.

**The order of your references is part of the prompt.** They are numbered in the order you attach
them, and each one is only visible to the instructions that follow it — so reordering two references
is a different request, not a cosmetic change. Refer to them the way you would in a sentence:

> Put the jacket from the second image on the person in the first image.

Because everything is a reference, the workflows that need special tooling elsewhere are just things
you say:

| You want | Attach | Say |
| --- | --- | --- |
| Change one region | The image, with the region marked | `Replace the area marked in red with a bay window.` |
| Use a mask | The image, then the mask | `Use the second image as a mask: change only the white area.` |
| Extract a subject | The photo | `Isolate the cyclist on a transparent background.` |
| Combine subjects | Each subject, in order | `Place the person from image 1 into the kitchen in image 2.` |
| Keep an identity | One or more shots of them | `Keep the face and hair from image 1 exactly.` |

To mark a region, draw straight **onto the reference** with the Image Editor's paint tools before
attaching it, then name the marking in the prompt ("the red outline", "the green box"). A mask is
not a special input here — it is another picture you describe.

Each reference is fitted to about 1024 px and re-encoded on **every step**, so ten references cost
noticeably more than two. Attach what the instruction actually refers to.

## Transparent backgrounds

**Turning on "Transparent background (RGBA)" does not by itself make the background transparent.**
The control keeps the model's alpha channel instead of flattening the render onto white; what goes
*into* that channel is decided by your prompt. You need both:

1. **Ask for it** — `The background is transparent. This is an RGBA image with transparency.`
   (The Studio offers exactly this wording on a button beside the toggle; edit it as you like.)
2. **Turn the toggle on**, so the transparency survives into the saved PNG.

With only the prompt you get a render that *looks* cut out but is flattened onto white. With only
the toggle you get a four-channel PNG that is fully opaque. Together you get a real cut-out that
keeps its transparency through editing, layering and export — and can be attached straight back as a
reference without losing it.

There is no matting step and no guarantee beyond what the model decodes; a difficult edge is a
prompting problem, not a settings problem.

## Settings

| Control | Default | Notes |
| --- | --- | --- |
| Steps | 40 | The model's own default. Below ~20 the fine detail and text start to break down. |
| Guidance | 1.0 | True CFG. Leave at 1.0 for no negative branch; 3–5 is a reasonable band once you add one. |
| Size | 2048 x 2048 | Seven presets, from 1536 x 2752 (9:16) to 2752 x 1536 (16:9). Custom sizes run 32–2752 px per side, in steps of 32. |
| Variations | 1 | Up to 8 per request. A 2048-square 40-step render is not cheap; raise it deliberately. |
| References | none | 1–10, in order. See *Editing with reference images*. |
| Transparent background | off | Keeps the alpha channel. Ask for transparency in the prompt too. |
| Seed | random | Fix it to iterate on a prompt without the composition moving underneath you. |

## Rewriting your prompt

2.1 ships two **official rewriters** — one for text-to-image, one for editing — as optional
downloads. They turn a short brief into the long, descriptive prompt this model was trained on, and
suggest an aspect ratio to go with it.

They are entirely optional: 2.1 generates and edits normally without either, and nothing prompts you
to install them. When one is installed, **Qwen rewriter** appears in Prompt tools; which of the two
runs is decided by your request (references attached means the editing one). The rewrite arrives in
a box you can edit, beside your original — nothing replaces your prompt until you press Apply, and
the aspect-ratio suggestion is a separate yes/no.

Both are covered by the same **Qwen RESEARCH LICENSE AGREEMENT** as the model itself.

## Licence

Qwen-Image 2.1 ships under the **Qwen RESEARCH LICENSE AGREEMENT**: the weights are licensed for
research and evaluation use only, and commercial use needs a separate licence from Alibaba. You
accept these terms before the download starts; the full text is under **About → Licenses**.
