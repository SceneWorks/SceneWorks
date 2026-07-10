# Anima Prompt Guide

Anima 2B (CircleStone Labs) is an anime text-to-image model. It follows the
**danbooru / booru tag convention** rather than natural-language sentences.

## Convention

- Write **comma-separated tags**, not prose.
- **Lowercase**, and use **spaces, not underscores** (`blue hair`, not `blue_hair`).
- Prefix **artists** with `@` (`@artist_name` → `@artistname`).
- Order tags roughly as:

  `[quality / meta / year / safety] [character count] [character] [series] [@artist] [general tags]`

## Prefixes

- **Positive prefix:** `masterpiece, best quality, score_7, safe,`
- **Negative prefix:** `worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration`

The negative prefix is pre-seeded into the negative-prompt box; the positive prefix
is a good way to open your prompt.

## Prompt weighting

Weighting works and needs to be **strong** to register — e.g. `(chibi:2)`. Weights
are applied to the model's T5 query tokens, so a heavier multiplier than you might use
on other models is expected.

## Variants

- **Anima 2B** — the base model (30 steps, CFG 4.5).
- **Anima 2B Aesthetic** — aesthetic fine-tune (30 steps, CFG 4.5).
- **Anima 2B Turbo** — merged CFG-free few-step student (10 steps).

Default sampler is ER-SDE-3 (`er_sde`) with a flow-match schedule (shift 3.0).

## License

Distributed under the CircleStone Labs Non-Commercial License v1.2 — generations are
for non-commercial use only.

- [Anima model card (CircleStone Labs)](https://huggingface.co/circlestone-labs/Anima)
