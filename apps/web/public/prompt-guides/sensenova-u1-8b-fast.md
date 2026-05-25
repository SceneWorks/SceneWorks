# SenseNova-U1 8B Fast Prompt Guide

## Best For

Fast SenseNova-U1 drafts, quick layout exploration, posters, infographics, and image edits where speed matters more than final polish.

## Prompt Shape

Use the same structured brief as the base model:

`purpose + subject/content + layout + visual hierarchy + exact text + style + details`

The fast variant uses fewer steps, so be clear and avoid asking for too many unrelated ideas at once.

## Build The Prompt

### Subject

State the topic or main subject first. If the image is informational, say what the viewer should learn.

### Details

Include only the details that matter most. For graphics, specify sections and labels. For realistic scenes, specify materials, props, and background.

### Style

Use direct style labels:

- `clean infographic`
- `modern presentation slide`
- `friendly cartoon explainer`
- `realistic product photo`
- `minimal poster design`

### Camera And Composition

For graphics, use layout language. For photos, use camera language.

Good: `three equal cards across the center`, `top-down product flat lay`, `eye-level portrait`.

### Motion

This is an image model, so describe implied motion only as a still frame: `fabric caught mid-sway`, `steam rising`, `water droplets suspended`.

### Text And Typography

Quote all required text. Keep text blocks short when possible. Specify hierarchy: title, subtitle, labels, and captions.

### Editing

Fast editing is useful for drafts. For critical identity preservation or maximum-quality edits, use the base SenseNova-U1 8B model.

## Example Prompts

`A square infographic titled "MORNING ROUTINE". Four rounded panels in a 2x2 grid show: "Hydrate", "Stretch", "Plan", "Focus". Each panel has a simple icon, one short caption, and a warm pastel background. Clean modern vector style, high contrast readable text, generous spacing.`

`Keep the original portrait identity, face shape, and pose. Change the background to a bright studio wall with a soft blue gradient. Add a clean white name badge on the jacket that reads "MIRA". Preserve realistic lighting and skin texture.`

## Sources

- [SenseNova-U1 model card](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT)
- [SenseNova-U1 prompt enhancement doc](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT/blob/main/docs/prompt_enhancement.md)
- [SenseNova-U1 Infographic model card](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic)
