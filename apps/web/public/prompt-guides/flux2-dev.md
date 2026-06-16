# FLUX.2 [dev] Prompt Guide

## Best For

The 32B Black Forest Labs FLUX.2 [dev] checkpoint — the high-fidelity flagship of the FLUX.2 family, ported natively to MLX (Apple Silicon only). A larger, sharper sibling of FLUX.2 [klein]: a Mistral-3 text encoder and a 48-layer flow transformer give it stronger prompt adherence, finer detail, and better text rendering than the 9B klein, at the cost of speed and memory. Guidance-distilled text-to-image; reference editing and LoRAs land separately.

> **License:** FLUX [dev] Non-Commercial License v2.0 — gated Hugging Face download. Accept the license at the model card and add a Hugging Face token under Settings → Service credentials before downloading. The model is for non-commercial use; the images you generate are yours to use commercially.

> **Hardware:** 128 GB Mac only. The dense weights are too large to quantize in memory, so the model is pre-quantized to Q4 on disk at install (a one-time conversion step after download). Generation peaks ~74 GB at 1024². macOS only.

## Prompt Shape

FLUX.2 [dev] reads a fluent natural-language description, not tag lists:

`subject + setting + visual details + style + composition + lighting + any text`

There is no negative prompt — FLUX.2 [dev] is guidance-distilled (embedded guidance, not classifier-free), so describe everything you DO want in the positive prompt.

## Build The Prompt

### Subject

Describe the subject as if captioning a finished photo. Include type, action, position, and distinguishing features.

Good: `a marine biologist examining a glowing jellyfish in a research lab tank`

### Details

Add material, texture, and atmosphere:

- `brushed copper helmet`
- `wet kelp glinting under fluorescents`
- `late-afternoon golden light through high windows`

### Style + Composition

Pick concrete style cues (photo, illustration, render) and a composition (medium shot, low angle, etc.). [dev] handles photoreal and stylized equally well but doesn't infer them — say what you want. Its larger text encoder follows long, layered descriptions more faithfully than klein, so it rewards detail.

### Text

FLUX.2 [dev] renders longer, cleaner text than klein. Wrap exact strings in double quotes:

`a vintage tea-tin labeled "Earl Grey No. 3"`

## Defaults

- Resolution: 1024×1024 (also 768×768, 1280×720, 720×1280)
- Steps: 28 (24–50 is the useful range; 28 is the recommended trade-off)
- Guidance: 4.0 (embedded distilled guidance — raise for tighter prompt adherence, lower for more variation)
- Quantization: Q4 (pre-quantized on disk at install; required to fit in memory)

## Sources

- [FLUX.2 [dev] model card](https://huggingface.co/black-forest-labs/FLUX.2-dev)
- [FLUX.2 announcement](https://bfl.ai/blog/flux-2)
