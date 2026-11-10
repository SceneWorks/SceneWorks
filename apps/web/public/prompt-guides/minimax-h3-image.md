# MiniMax-H3-Image Prompt Guide

MiniMax-H3-Image is MiniMax's open-weights text-to-image model. It is the MiniMax-H3
transformer without its audio stream, distilled to a fixed six-step sampler, and it decodes
through its own image VAE. This guide covers what SceneWorks exposes for it today.

## What you control

- **Prompt.** The prompt is encoded verbatim by the Qwen3-VL-32B text encoder MiniMax-H3 uses —
  no chat template is added. Up to 4096 tokens are read; anything longer is cut off.
- **Size.** Pick one of the presets. The 2048 x 2048 default is the model's own default. The
  larger presets are the model card's published sizes of about 16 megapixels each:

  | Aspect | Size |
  | --- | --- |
  | 1:1 | 4096 x 4096 |
  | 4:3 / 3:4 | 4736 x 3552 / 3552 x 4736 |
  | 3:2 / 2:3 | 5024 x 3360 / 3360 x 5024 |
  | 16:9 / 9:16 | 5472 x 3072 / 3072 x 5472 |
  | 21:9 | 6272 x 2688 |

  These edges go past the 4096 px limit the other image models use; the larger limit applies to
  this model only. A size below about 4 megapixels is generated on a larger canvas at the same
  aspect and resized back, so small outputs cost about as much as 2048 x 2048.
- **Seed.** The seed is the only other input; set one to repeat a render's starting noise.

## What is fixed, and why the controls are hidden

- **Steps are always six.** The sampler has one output head per step, so no other step count
  exists. The Steps control is hidden.
- **No guidance scale and no negative prompt.** The model is guidance-distilled: there is no
  unconditional pass for a guidance scale or a negative prompt to steer. Both controls are hidden,
  and the API refuses a request that sends either.

## Writing prompts

- Describe the scene directly: subject, setting, composition, lighting and style, in plain
  sentences. The text encoder reads up to 4096 tokens, so a long description is read in full.
- SceneWorks has not yet run its own prompt-quality comparison for this model; the model card
  is the upstream reference for prompting advice.

## Not available yet

Reference-image editing, LoRAs and the MiniMax prompt expander are separate pieces of work and
are not offered for this model yet. The Q8 and Q4 tiers are not published yet; bf16 is the only
installable tier today.

## Licence

MiniMax-H3-Image is governed by the MiniMax-H3-Image Community License Agreement, which you
accept before downloading. Its terms differ from MiniMax-H3's: commercial use needs separate
written authorization from MiniMax once annualized revenue exceeds 1 million US dollars, or for
any hosted service offered to third parties. Use is limited to the Applicable Territory, which
excludes the European Union, the United Kingdom, the Republic of Korea and the United States of
America. Read the full text under About → Licenses.
