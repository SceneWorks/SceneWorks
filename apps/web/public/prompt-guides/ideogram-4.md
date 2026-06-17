# Ideogram 4 Prompt Guide

## Best For

Ideogram 4 is a 9.3B-parameter flow-matching text-to-image model with best-in-class text rendering, deep language understanding, and **explicit layout control via bounding boxes and color palettes**. Apple Silicon only (native MLX backend).

It is uniquely good at: legible typography in images, precise placement of objects and text, and adherence to a described layout — because it is prompted with a **structured JSON caption**, not free text.

> **License:** Ideogram Non-Commercial Model Agreement — gated Hugging Face download. Accept the license at the [model card](https://huggingface.co/ideogram-ai/ideogram-4-fp8) and add a Hugging Face token under Settings → Service credentials before downloading. Generations are for non-commercial use only.

> **Hardware:** Pre-quantized **Q4** (default, ~15 GB download, ~28 GB peak at 1024²) or **Q8**. The 2048² / 6:1 ceiling needs ~96 GB. macOS only.

## The key thing to know

**Ideogram 4 was trained on structured JSON captions, not free text.** A plain-text prompt produces a coherent but *prompt-agnostic* image — it will look good but ignore much of what you asked for. A JSON caption gives accurate adherence.

You don't have to write JSON by hand. SceneWorks builds the caption for you from the prompt builder, and a **magic-prompt** expander turns a plain-text idea into a full JSON caption you can review and edit before generating. This guide explains what that caption contains so you can get the most out of it.

## Caption structure

A caption has up to three top-level sections:

- **`high_level_description`** (recommended) — one sentence summarizing the whole image.
- **`style_description`** (optional) — the look: aesthetics, lighting, medium, and either a photo or an art style.
- **`compositional_deconstruction`** (required) — the layout: a `background` plus a list of `elements`.

### Style

`style_description` carries exactly one of **`photo`** (e.g. "telephoto, shallow depth of field, eye-level") or **`art_style`** (e.g. "watercolor illustration"), and the two use a slightly different key order:

- **Photo captions:** `aesthetics`, `lighting`, `photo`, `medium`, `color_palette`
- **Non-photo captions:** `aesthetics`, `lighting`, `medium`, `art_style`, `color_palette`

Where:

- `aesthetics` — mood/feel (e.g. "serene, warm, naturalistic")
- `lighting` — (e.g. "golden hour, soft backlight, long shadows")
- `medium` — (e.g. "photograph", "oil painting")
- `color_palette` (optional) — up to 16 uppercase `#RRGGBB` colors

SceneWorks emits these in the correct order for you; the distinction matters only if you paste raw JSON.

### Composition & layout

`compositional_deconstruction` has:

- **`background`** (required) — a sentence describing the scene behind the elements.
- **`elements`** (required) — a list of objects and text blocks placed on a canvas.

Each element is one of two types:

- **`obj`** — a thing in the scene. Keys in order: `type`, `bbox`, `desc`, `color_palette`.
- **`text`** — rendered text. Keys in order: `type`, `bbox`, `text`, `desc`, `color_palette`. Put the exact characters to render in `text`.

### Bounding boxes

`bbox` is `[y_min, x_min, y_max, x_max]`, integers normalized to **0–1000**, origin at the **top-left**. So `[0, 0, 1000, 1000]` is the whole frame; `[250, 320, 950, 760]` is a region in the lower-middle. In SceneWorks you can drag these on the visual canvas instead of typing numbers.

### Color palettes

Palettes are uppercase hex (`#RRGGBB`): up to **16** colors overall (on `style_description`) and up to **5** per element. Use them to lock brand colors or a consistent scheme.

## Tips

- **Be specific in `desc`.** Each element's `desc` is where detail lives — material, pose, expression, lighting on that object.
- **Key order matters.** Ideogram's caption verifier is order-sensitive; SceneWorks emits the keys in the correct order automatically, so prefer the builder over pasting raw JSON.
- **Use `text` elements for legible type.** Ideogram 4's text rendering is a headline feature — give it an explicit `text` element with a tight `bbox` rather than burying the words in a description.
- **Start from magic-prompt.** Describe your idea in plain language, let the expander draft the caption, then refine the boxes, palette, and per-element descriptions.

## Example caption

```json
{
  "high_level_description": "A photograph of a red fox sitting in a snowy forest at golden hour.",
  "style_description": {
    "aesthetics": "serene, warm, naturalistic",
    "lighting": "golden hour, soft warm backlight, long shadows",
    "photo": "telephoto, shallow depth of field, eye-level",
    "medium": "photograph"
  },
  "compositional_deconstruction": {
    "background": "A snowy forest of tall pine trees, golden sunlight filtering through the branches.",
    "elements": [
      {
        "type": "obj",
        "bbox": [250, 320, 950, 760],
        "desc": "A red fox with vivid orange fur, white chest and a thick bushy tail, sitting upright in the snow and facing the camera."
      }
    ]
  }
}
```

## Sources

- [Ideogram 4 model card](https://huggingface.co/ideogram-ai/ideogram-4-fp8)
- [Ideogram 4 blog post](https://ideogram.ai/blog/ideogram-4.0/)
