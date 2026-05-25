# Qwen Image Edit Prompt Guide

## Best For

Precise instruction edits, combining reference images, replacing objects or clothing, changing pose/style, and preserving selected parts of the original image.

## Prompt Shape

Use a direct edit instruction:

`source reference + edit action + preserved elements + target details + style/composition constraints`

When multiple images are used, refer to them by order: `Image 1`, `Image 2`, `Image 3`.

## Build The Prompt

### Subject

Identify the source subject clearly. Say what should be preserved: identity, pose, outfit, lighting, background, camera angle, or composition.

Good: `Make the girl from Image 1 wear the black dress from Image 2 and sit in the pose from Image 3.`

### Details

Describe the edit with concrete visual details. For replacement edits, include color, material, shape, scale, and placement.

### Style

If changing style, describe both the desired style and what should stay realistic or unchanged.

Use: `preserve facial identity`, `keep realistic skin texture`, `change only the background to watercolor`.

### Camera And Composition

For controlled edits, ask the model to keep perspective and framing:

- `same camera angle`
- `same crop`
- `same lens perspective`
- `do not move the subject`

### Text And Layout

Quote exact text and specify its location. If editing signs, labels, posters, or UI, describe font style and alignment.

### Negative Prompt

Use a minimal negative prompt for quality issues: `blur, extra fingers, warped text, changed identity, mismatched lighting`.

## Example Prompts

`Keep the person from Image 1 with the same face, hair, and camera angle. Replace the jacket with the leather jacket from Image 2, matching the original lighting and body pose. Preserve the plain gray background and realistic photography style.`

`Edit the cafe sign so it reads "ORCHARD COFFEE" in white hand-painted script. Keep the storefront, awning, window reflections, and morning sunlight unchanged. The new lettering should follow the same perspective on the glass.`

## Sources

- [Qwen Cloud image editing docs](https://docs.qwencloud.com/developer-guides/image-generation/image-editing)
- [Qwen Cloud image prompt guide](https://docs.qwencloud.com/developer-guides/accuracy-tuning/image-generation)
- [Qwen Cloud text-to-image docs](https://docs.qwencloud.com/developer-guides/image-generation/text-to-image)
