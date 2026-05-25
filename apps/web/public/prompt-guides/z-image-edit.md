# Z-Image-Edit Prompt Guide

## Best For

Instruction-based image edits where the source image should stay recognizable: style changes, object edits, text/layout changes, and creative transformations.

## Prompt Shape

Write an editing instruction plus the visual target:

`preserve what matters + edit action + target details + style + composition/lighting constraints`

Be explicit about what should remain unchanged.

## Build The Prompt

### Subject

Refer to the source image directly: `the person`, `the car`, `the background sign`, `the red chair`. If there are multiple subjects, identify them by position.

Good: `Keep the woman's face, pose, and camera angle unchanged. Replace the blue jacket with a black velvet blazer.`

### Details

Describe the replacement or change in concrete terms: material, color, placement, size, and relationship to existing objects.

If editing text, put exact desired text in quotes and name its location.

### Style

For style transfers, say how strong the change should be:

- `subtle film look while preserving realistic skin texture`
- `turn the scene into a flat 1960s travel poster`
- `make the product look like brushed aluminum`

### Camera And Composition

For most edits, preserve the original camera and composition unless you want a transformation. Say that directly.

Use: `keep the same framing`, `same perspective`, `same lighting direction`, `do not crop the subject`.

### Lighting

When changing materials or inserting objects, describe how they should match the existing light:

`The new glass vase should catch the same warm window light from the right.`

### What To Avoid

Avoid vague commands like `make it better`. Avoid stacking unrelated edits in one prompt. For multi-step changes, write them as a short ordered sentence.

## Example Prompts

`Keep the original portrait composition, facial identity, pose, and background unchanged. Replace the gray hoodie with a tailored emerald satin jacket, with realistic folds and highlights matching the existing soft window light. Add a small gold pin on the left lapel shaped like a crescent moon.`

`Preserve the product, camera angle, and white studio background. Change the label text to read "LUMEN TEA" in clean black serif letters, centered on the bottle. Make the cap matte silver and add a faint reflection under the bottle.`

## Sources

- [Z-Image-Turbo model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Tongyi-MAI prompting discussion](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8)
- [Z-Image prompt enhancer template](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py)
