# Z-Image-Turbo Prompt Guide

## Best For

Fast image drafts, photorealistic scenes, bilingual text in images, and prompts that need strong instruction following in a few steps.

## Prompt Shape

Write one detailed visual description:

`subject + action + setting + important details + style + camera/composition + lighting + text requirements`

Z-Image-Turbo responds well to long, specific prompts. If the idea is short, expand it before generating.

## Build The Prompt

### Subject

Name the main subject first. Include count, identity, pose, clothing, expression, and what must not change.

Good: `one ceramic tea cup with a cracked blue glaze, centered on a linen tablecloth`

### Details

Add the details that make the image controllable: materials, colors, props, background objects, layout, and exact relationships between objects.

For text in the image, write the exact text in quotes and say where it appears.

### Style

Use concrete visual styles: `documentary photo`, `product render`, `ink illustration`, `children's book`, `editorial fashion`, `flat poster design`.

Avoid vague praise such as `masterpiece` or `best quality`. The official prompt enhancer avoids generic quality tags and favors concrete visual description.

### Camera And Composition

Specify shot size, angle, lens feel, and framing:

- `close-up product shot`
- `eye-level portrait`
- `top-down flat lay`
- `wide establishing shot`
- `centered composition with generous negative space`

### Lighting

Describe the light source and mood:

- `soft window light from the left`
- `neon reflections on wet pavement`
- `warm backlight with a thin rim light`

### Negative Prompt

Turbo is usually guided by the positive prompt. If your workflow supports negative prompts, keep them short and concrete: `blurry text, extra fingers, distorted logo, crowded background`.

## Example Prompts

`A young woman in a red embroidered hanfu stands in a night market courtyard, holding a round fan painted with cranes. A small neon lightning sign glows above her open left palm. Behind her, a tiered pagoda is softly blurred with colorful lantern bokeh. Eye-level medium portrait, centered composition, warm lantern light mixed with cool blue night light, detailed fabric embroidery, calm expression.`

`A clean bilingual bakery poster. Large headline at the top reads "MOONCAKE MORNING" in bold cream letters. Under it, smaller Chinese text reads "新鲜出炉". Three golden mooncakes sit on a jade plate, with steam rising and osmanthus flowers scattered around. Flat editorial poster style, balanced grid layout, pale green background, crisp readable typography.`

## Sources

- [Z-Image-Turbo model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Tongyi-MAI prompting discussion](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8)
- [Z-Image prompt enhancer template](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py)
