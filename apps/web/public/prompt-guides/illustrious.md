# Illustrious Prompt Guide

## Best For

Anime and illustration text-to-image. Illustrious-XL (OnomaAI) is a Danbooru-tag SDXL finetune — it shares the SDXL architecture, sdxl-family LoRA support, real CFG + negative prompt, and dual-CLIP text encoding, but is trained for high-resolution illustration rather than photorealism. Two versions are offered as separate models; see **v1.0 vs v2.0** below.

## Prompt Shape

Illustrious is trained on **Danbooru-style tags blended with natural language**. You can prompt with plain English, with comma-separated tags, or with both — tags give you the most precise control over anime-specific attributes.

A reliable structure:

`quality tags + subject count + character/appearance tags + clothing + pose/expression + setting + composition`

The CLIP encoders weight earlier tokens more heavily, so **lead with the quality tags and the subject**.

## Quality Tags

The community convention is to open with a short quality preamble, then the subject:

`masterpiece, best quality, highly detailed, 1girl, solo, ...`

Common openers: `masterpiece`, `best quality`, `highly detailed`, `absurdres`. These are learned tokens, not magic — one or two is plenty.

## Subject Count

Danbooru count tags are load-bearing. State them explicitly:

- `1girl, solo` — a single subject
- `2girls` / `1boy, 1girl` — multiple subjects

If you want one character, write `1girl, solo` (or `1boy, solo`). See the note under v2.0 about wide frames.

## Build The Prompt

### Appearance

Tag hair, eyes, and distinguishing features:

`silver hair, long hair, blue eyes, hair between eyes, ahoge`

### Clothing

`school uniform, sailor collar, pleated skirt, thighhighs`

### Pose and Expression

`standing, looking at viewer, light smile, arms behind back`

### Setting

`cherry blossom park, outdoors, day, falling petals`

### Composition

`full body`, `upper body`, `cowboy shot`, `from above`, `dutch angle`

## Negative Prompt

Anime models use a different negative-prompt convention than photoreal ones. A practical baseline:

`lowres, bad anatomy, bad hands, missing fingers, extra digits, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry`

## v1.0 vs v2.0

These are **separate models, not a version upgrade** — pick per job.

- **v1.0** handles wide and large frames well, including up to 1536×1536 and tall non-standard ratios. Reach for it when you want a big or wide composition.
- **v2.0** (the "STABLE" annealing snapshot) tends to be subtler and more consistent, but it is prone to **duplicating the subject in wide frames** — a `1girl, solo` prompt can render two characters once the frame gets wide. Its resolution picker therefore omits the widest buckets. Prefer square or tall framing with v2.0; if you see an unwanted second character, narrow the frame.

## Settings

- **Steps:** ~30
- **Guidance (CFG):** ~7.0
- **Sampler:** the default (Euler) is a good starting point; `dpmpp_2m` with a Karras schedule is a common alternative.
- **Resolution:** native 1024×1024; see the resolution picker for the safe set per version.

## License

- **v1.0** — SDXL license (CreativeML Open RAIL++-M). Commercial use OK, ungated.
- **v2.0** — CreativeML OpenRAIL-M. Commercial use OK, ungated.

Both carry behavioral-use restrictions per their respective licenses.
