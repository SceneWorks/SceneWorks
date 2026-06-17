# RealVisXL Lightning Prompt Guide

## Best For

Fast photoreal text-to-image — portraits, products, landscapes, and editorial-style scenes in ~5 steps instead of ~30. RealVisXL_V5.0_Lightning is a **standalone distilled** SDXL checkpoint (the SDXL-Lightning few-step distillation is baked into the weights), so it shares RealVisXL's photoreal look at roughly 6× the speed. openrail++, commercial use OK, ungated.

Use this when you want quick photoreal iterations or batches. For edits, masked inpaint, reference identity, or detail-refine, switch to the standard **RealVisXL** — Lightning is text-to-image only.

## How It Differs From Standard RealVisXL

- **~5 steps** (vs ~30). The checkpoint is distilled for a few-step Euler-trailing schedule; pushing steps much higher rarely helps and can over-cook.
- **CFG off by default (guidance 1.0).** Lightning checkpoints are trained classifier-free-guidance-free. At guidance 1.0 the model runs a single forward per step (fast) and the **negative prompt has no effect**.
- **Raise guidance toward ~1.5–2.0** only if you specifically want the negative prompt back — that re-enables real CFG at roughly 2× the per-step cost. Keep it low; high CFG on a distilled checkpoint produces harsh, over-contrasted output.
- **Text-to-image only.** Reference identity, img2img, inpaint, and detail-refine are not available on the few-step path.

## Prompt Shape

Same photoreal-leaning natural language as RealVisXL:

`subject + key details + style/medium + composition + lighting + quality tags`

CLIP encoders weight earlier tokens more heavily, so **lead with the subject and the most important attributes**. Few-step sampling rewards clear, concrete prompts — there are fewer steps to "recover" from a vague or contradictory prompt.

## Build The Prompt

### Subject

Front-load the subject in plain language:

`a candid portrait of a woman in her early 30s holding a ceramic mug`

### Details

Specific material, texture, and atmosphere — where RealVisXL pulls ahead of base SDXL:

- `fine skin texture, visible pores, light freckles`
- `soft cotton sweater, hand-knit cable pattern`
- `morning steam rising from the mug`

### Style / Medium

Photographic vocabulary rather than illustrative terms:

- `editorial portrait photography, 50mm`
- `cinematic film still, 35mm anamorphic`
- `studio product photography, matte backdrop`

### Lighting

Photoreal output lives or dies on lighting language — be specific:

- `golden hour backlight, warm rim`
- `soft north-window light, even diffusion`
- `single key light, deep shadow falloff`

### Quality Tags

Keep these short and photo-realistic — avoid stacking long lists of "8k, masterpiece, best quality" that flatten the result:

`sharp focus, natural skin tones, photorealistic, high detail`

## Negative Prompts

At the default guidance (1.0) the **negative prompt is ignored** — the distilled checkpoint runs CFG-free. If you need to push away a specific failure mode, raise guidance to ~1.5–2.0 to re-enable it, then keep the negative short and targeted:

`blurry, lowres, overprocessed skin, plastic skin, oversaturated, deformed, extra fingers, watermark, text, jpeg artifacts, painting, illustration`

## Tips

- ~5 steps is the sweet spot; try 4–6. Much higher steps waste time and can over-render.
- Leave guidance at 1.0 for the fastest, most "natural" few-step output. Only raise it (toward 2.0) when you specifically need the negative prompt.
- Native 1024×1024; the canonical SDXL buckets (1152×896, 896×1152, 1216×832, 832×1216, 1344×768, 768×1344) are trained resolutions — prefer them.
- Lighting words do more work than style words — invest in specific, physically-plausible lighting.
- Keep skin/texture tags subtle; over-tagging "ultra-realistic skin" can produce a waxy or over-rendered look — more so at low step counts.
- Layer sdxl-family LoRAs for specific styles — RealVisXL Lightning accepts the SDXL LoRA ecosystem, though heavy LoRAs may need a step or two more to settle.
- Want edits, reference likeness, or inpaint? Use the standard **RealVisXL** — the few-step path is text-to-image only.

## Example Prompts

`A candid portrait of a fisherman in his sixties on a wooden dock at dawn, weathered hands holding a coil of rope, fine skin detail, soft golden backlight, shallow depth of field, editorial documentary photography, sharp focus.`

`Studio product shot of a brushed aluminum espresso tamper on warm walnut, soft directional side light, subtle wood grain, minimalist composition, professional product photography, natural color, high detail.`

## Sources

- [RealVisXL_V5.0_Lightning model card](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning)
- [SDXL-Lightning (distillation method)](https://huggingface.co/ByteDance/SDXL-Lightning)
- [SDXL technical report](https://arxiv.org/abs/2307.01952)
