# Lens-Turbo Prompt Guide

## Best For

Fast Lens generations, visual brainstorming, text rendering tests, and high-detail image prompts where speed matters more than maximum refinement.

## Prompt Shape

Use the same dense caption style as Lens:

`subject + setting + visual details + style + composition + lighting + text if needed`

Lens-Turbo is distilled for 4-step generation, so keep prompts clear and avoid overloading the scene.

## Build The Prompt

### Subject

Lead with one clear subject or a simple scene. If there are multiple subjects, state the count and placement.

Good: `three glass perfume bottles arranged in a triangle on a black acrylic surface`

### Details

Add the few details that matter most: material, color, texture, and background. Turbo can follow rich prompts, but crowded prompts can become visually noisy.

### Style

Use direct style labels: `editorial product photo`, `travel poster`, `macro photography`, `oil painting portrait`, `concept art`.

### Camera And Composition

Use concrete camera words:

- `top-down`
- `close-up`
- `wide angle`
- `telephoto`
- `centered composition`
- `panoramic landscape`

### Lighting

Describe the primary light and any accent light:

- `large softbox reflection`
- `warm window light`
- `cool rim light`
- `golden sunset backlight`

### Text In Images

Lens-Turbo can handle text better when the wording, font style, size, and location are explicit.

Example: `the label reads "NORTH STAR" in tall condensed white letters, centered on the bottle`.

### Negative Prompt

Use concise negatives: `blurred lettering, extra objects, low contrast, warped label`.

## Example Prompts

`A clean travel poster for a mountain railway. Large headline at the top reads "ALPINE LINE" in cream block letters. A red train curves across a snowy bridge below, pine forest and blue mountains in the background, flat vintage poster style, balanced composition, crisp readable typography.`

`A close-up editorial photograph of a black ceramic espresso cup on a marble counter, thin steam rising, amber cafe light, shallow depth of field, tiny sugar crystals scattered near the saucer, realistic texture, calm morning mood.`

## Sources

- [Microsoft Lens-Turbo model card](https://huggingface.co/microsoft/Lens-Turbo)
- [Microsoft Lens model card](https://huggingface.co/microsoft/Lens)
