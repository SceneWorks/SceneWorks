# Qwen Image 2.1 Turbo Prompt Guide

## Best For

Fast iteration on the same work Qwen Image 2.1 is good at: large-canvas composition, posters and
covers, illustration, realistic photography, and layouts with readable text. Turbo is the
**distilled** sibling of Qwen Image 2.1 — the same model architecture and prompt format, trained to
render in **8 steps** instead of 40 — so a draft comes back in a fraction of the time at the same
2048 px native size.

Prompts move freely between Turbo and the base model; seeds and exact framing do not, because the
weights are different. A common workflow is to explore with Turbo and render the keeper on the base
model.

**LoRA and LoKr adapters are supported.** Turbo loads adapters from the same `qwen-image-2-1`
family as the base model; adapters trained for the older Qwen Image (2512) checkpoint belong to a
different family and will not load. Turbo cannot be used as a **training** base — train on Qwen
Image 2.1 and the adapter loads on Turbo unchanged. Trained adapters inherit the research licence.

## Prompt Shape

Use the Qwen-style structure:

`subject + setting + style + camera + atmosphere + detail modifiers`

For quick ideas, the shorter shape still works:

`subject + setting + style`

Turbo rewards the same long, descriptive prompts as the base model. A short brief renders, but the
more you specify, the less the few-step schedule has to guess.

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

Turbo responds to photographic direction: lens length, aperture, angle, distance.

Good: `85mm lens, shallow depth of field, low three-quarter angle`

### Text In The Image

Put the exact characters in quotation marks and say where they go and how they are set. Keep the
string short — a headline, not a paragraph.

Good: `a bookshop sign reading "NIGHT EDITION" in condensed serif capitals, centred above the door`

## Steps And Schedule

Turbo's weights ship with their own **8-step schedule**, and that is what runs by default.

* **Other step counts are allowed.** Any count from 1 up is accepted; a count other than 8
  *resamples* the saved 8-step curve rather than switching to the base model's schedule. 8 is
  the count the schedule was distilled for.
* **Scheduler `Default` keeps the distilled schedule.** Picking any other scheduler replaces it
  with that scheduler's own curve. Every scheduler the base model offers is available, but only
  `Default` is the schedule Turbo was distilled for.
* **Samplers** work as on the base model.

## Negative Prompts

Turbo is meant to run **without** classifier-free guidance, so its guidance scale defaults to 1.0.
At 1.0 a negative prompt has little to push against. Guidance still engages if you raise it and
supply a negative prompt — the control is there — but the distilled schedule was trained CFG-free,
so raise it sparingly; a guided render also evaluates a second (negative) branch on every step.

A negative prompt is for what must be absent, not for quality adjectives:
`extra fingers, watermark, cropped text, motion blur`.

## Editing with reference images

Turbo keeps the base model's single pipeline. Text-to-image is that pipeline with no reference
attached; editing, combining and local changes are the *same* call with **1–10 reference images**.
There is no separate edit model, no strength slider, and no inpainting mode.

**The order of your references is part of the prompt.** They are numbered in the order you attach
them, and each one is only visible to the instructions that follow it — so reordering two references
is a different request, not a cosmetic change:

> Put the jacket from the second image on the person in the first image.

In the Image Editor the picture you are editing is always **image 1**, and the references you attach
start at image 2 — the reference rail shows each one's number. Each distinct picture takes one slot,
and the ten-image limit counts distinct images.

| You want | Attach | Say |
| --- | --- | --- |
| Change one region | The image, with the region marked | `Replace the area marked in red with a bay window.` |
| Use a mask | The image, then the mask | `Use the second image as a mask: change only the white area.` |
| Extract a subject | The photo | `Isolate the cyclist on a transparent background.` |
| Combine subjects | Each subject, in order | `Place the person from image 1 into the kitchen in image 2.` |
| Keep an identity | One or more shots of them | `Keep the face and hair from image 1 exactly.` |

References are scaled to about 1024 px. On the prefill path they are re-encoded on every step;
when the prompt cache runs they are encoded once (see *Prompt cache*). Either way ten references
cost noticeably more than two, so attach what the instruction actually refers to.

## Transparent backgrounds

**Turning on "Transparent background (RGBA)" does not by itself make the background transparent.**
The toggle keeps the model's alpha channel; your prompt decides what goes into it. You need both:

1. **Ask for it** — `The background is transparent. This is an RGBA image with transparency.`
   (The Studio offers exactly this wording on a button beside the toggle; edit it as you like.)
2. **Turn the toggle on**, so the transparency survives into the saved PNG.

## Settings

| Control | Default | Notes |
| --- | --- | --- |
| Steps | 8 | The distilled schedule. Other counts resample it; see *Steps And Schedule*. |
| Scheduler | Default | Default keeps the distilled schedule; any other choice replaces it. |
| Guidance | 1.0 | CFG off. Raise it only together with a negative prompt. |
| Size | 2048 x 2048 | Seven presets, from 1536 x 2752 (9:16) to 2752 x 1536 (16:9). Custom sizes run 32–2752 px per side, in steps of 32. |
| Variations | 1 | Up to 8 per request. |
| References | none | 1–10, in order. See *Editing with reference images*. |
| Transparent background | off | Keeps the alpha channel. Ask for transparency in the prompt too. |
| Seed | random | Fix it to iterate on a prompt without the composition moving underneath you. |

## Rewriting your prompt

The two **official Qwen Image 2.1 rewriters** — one for text-to-image, one for editing — work with
Turbo exactly as with the base model. They are optional downloads; when one is installed, **Qwen
rewriter** appears in Prompt tools, and which of the two runs is decided by your request
(references attached means the editing one). The rewrite arrives in a box you can edit, nothing
replaces your prompt until you press Apply, and the aspect-ratio suggestion is a separate yes/no.

## Prompt cache

Turbo (like the base model) can reuse the encoded prompt and references across steps instead of
recomputing them. Whether a render ran **cached** or **prefill** is shown on its job card in the
Queue. If a very long prompt or many large references outgrow the memory the render was admitted
with, the engine drops the cache for that render and says so there; the render still completes.

## Quant tiers

The BF16 tier downloads from Qwen's own repository. The Q8 and Q4 tiers are listed in the Model
Manager but are not yet published; they become installable once they are.

## Licence

Qwen-Image 2.1 Turbo ships under the same **Qwen RESEARCH LICENSE AGREEMENT** as the base model:
research and evaluation use only, and commercial use needs a separate licence from Alibaba. You
accept these terms for Turbo separately — accepting them for the base model does not cover it —
before the download starts; the full text is under **About → Licenses**.
