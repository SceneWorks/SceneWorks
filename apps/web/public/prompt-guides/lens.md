# Lens Prompt Guide

## Best For

High-quality text-to-image generations with strong prompt following, detailed photography, art styles, and readable text in signs, labels, and documents.

## Prompt Shape

Lens was trained with dense captions, so write a descriptive image caption:

`subject + setting + visual details + style + composition + lighting + text if needed`

The base Lens model is slower than Lens-Turbo but favors quality.

## Build The Prompt

### Subject

Describe the subject like a caption for a finished image. Include type, position, action, and important distinguishing features.

Good: `a ruby-throated hummingbird hovering in front of a red heliconia flower`

### Details

Lens examples often include rich material, surface, location, and atmosphere details. Add texture and context:

- `hand-carved letters`
- `aged brass compass`
- `water droplets suspended around the subject`
- `ornate tile border`

### Style

Use photographic or art direction terms:

- `National Geographic wildlife photography`
- `warm still life photography`
- `high fantasy digital art`
- `loose watercolor on visible paper grain`

### Camera And Composition

Lens supports many aspect ratios, so pair composition with the selected size:

- `overhead shot`
- `aerial drone photography`
- `extreme macro`
- `telephoto wildlife photography`
- `centered product shot`

### Lighting

Be specific about light and reflections:

- `golden hour`
- `warm desk lamp lighting`
- `dramatic backlit scene`
- `soft chiaroscuro lighting`

### Text In Images

Quote exact text and describe the physical medium:

`a ceramic tile sign reading "GRAND CENTRAL" in white mosaic letters`

### Negative Prompt

Use short negatives for unwanted artifacts: `blurry text, malformed hands, distorted lettering, low detail, busy background`.

## Example Prompts

`A rustic wooden sign at a fishing village dock reading "FRESH CATCH" in hand-carved blue letters, thick hemp rope border, fishing nets and lobster traps stacked behind it, seaside morning atmosphere, warm low sunlight, shallow depth of field.`

`An artisan honey jar with a vintage botanical label reading "CLOVER HONEY" in brown serif letterpress typography, ink drawings of clover and bees, clear glass jar, kraft paper texture, soft studio product lighting, centered composition.`

## Sources

- [Microsoft Lens model card](https://huggingface.co/microsoft/Lens)
- [Microsoft Lens-Turbo model card](https://huggingface.co/microsoft/Lens-Turbo)
