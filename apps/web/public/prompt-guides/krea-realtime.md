# Krea Realtime 14B Prompt Guide

## Best For

**Fast text-to-video** and **restyling an existing clip**. Krea Realtime 14B is the Wan 2.1 14B
text-to-video model distilled with Self-Forcing into an *autoregressive* generator: instead of
denoising a whole clip at once, it renders three-frame blocks left to right against a rolling
attention cache, at about 6 steps per block. That makes it markedly cheaper than a full 14B
whole-clip denoise at the same length.

> It is a **24fps** model, and its durations sit on a 3-frame block lattice — 2s is 45 frames,
> 5s is 117. The Duration menu only offers lengths that land on a whole number of blocks.

## What You Provide

| Input | What it's for |
|---|---|
| **Prompt** | The whole scene: subject, action, setting, camera, and look. There is no negative prompt (see below), so everything you want has to be stated positively. |
| **Reference image** *(optional)* | Image-to-video. The still is encoded and used to warm the model's attention cache, so the clip starts from that frame and moves outward. |
| **Source clip** *(optional)* | Video-to-video. The source drives the generation, and the **Video conditioning strength** control decides how much of it survives — lower keeps more of the original. |

If you supply both, the source clip wins.

## No Negative Prompt, No Guidance

The distillation baked classifier-free guidance out of the model: it runs a single forward pass per
step with no unconditional branch. So the Video Studio shows **no negative-prompt box and no
guidance slider** for this model — that is correct, not a missing feature. Anything you would
normally push into a negative prompt has to become a positive statement instead:

- Instead of a negative "blurry, low quality" → say **"sharp, crisp focus, clean detail"**.
- Instead of a negative "no camera shake" → say **"locked-off tripod shot, steady framing"**.
- Instead of a negative "not cluttered" → say **"minimal background, single subject"**.

## Writing The Prompt

Write one continuous description rather than a keyword list. A prompt that works well names, in
roughly this order:

1. **Subject** — who or what, and its concrete appearance.
2. **Action** — one clear motion that can complete inside the clip length.
3. **Setting** — where, and the light.
4. **Camera** — shot size and movement.
5. **Look** — film stock, palette, rendering style.

> *A weathered fisherman in a yellow oilskin coat hauls a rope hand over hand, standing on the deck
> of a small boat in heavy grey swell, cold overcast light, medium shot, slow push in, grainy 16mm
> film look.*

### Keep the action inside the clip

The model generates forward in time and never revises what it has already produced. An action that
does not fit in the selected duration gets cut off mid-motion rather than compressed. At 2–3
seconds, prefer a single gesture; at 4–5 seconds you can fit a short sequence.

### Camera moves

Slow, continuous camera language works best — *slow pan left*, *gentle push in*, *slow orbit*,
*locked-off*. Cuts, whip pans, and abrupt reframes fight the left-to-right generation and tend to
smear.

## Duration And Drift

Because each block is generated from a bounded window of what came before, quality drifts as clips
get longer — colours can creep and detail can soften toward the end. Durations are capped at 5
seconds for that reason. If you want a longer sequence, generate several short clips and cut them
together rather than pushing one clip long.

## Resolution

| Bucket | Notes |
|---|---|
| **832×480** *(default)* | The model's native bucket and the fastest. Start here. |
| **480×832** | The portrait pair of the default. |
| **1280×720** / **720×1280** | True 720p. Legal and it renders, but each frame carries ~2.3× the tokens, so both the attention cache and the render time grow accordingly. Prefer it only on a large-memory Mac. |

Custom sizes must be multiples of 16 and at most 1280 on the long edge.

## Quality Tiers

Three tiers ship: **Q4** (default), **Q8**, and **bf16**. Q4 is the practical choice — a bf16 14B
transformer is roughly 28 GB resident before the attention cache is counted. Move up a tier only if
you have the memory and want the extra fidelity; it is a creative choice, not a correctness one.

## Steps

The Advanced **Steps** control defaults to **6** — the reference driver's own setting. It is steps
*per frame-block*, not per clip, so raising it multiplies across every block in the clip. Small
changes (5–8) are the useful range; large values buy little because the model was distilled to
converge in a handful of steps.

## LoRAs

Krea Realtime shares Wan 2.1's transformer layout exactly, so **Wan-family style and motion LoRAs
apply to it**, as do LoRAs trained for Krea Realtime itself. They install as forward-time residuals,
which means they work identically on the quantized Q4/Q8 tiers and on bf16.

## Sources

- [Krea Realtime 14B model card](https://huggingface.co/krea/krea-realtime-video)
- [Krea Realtime technical blog post](https://www.krea.ai/blog/krea-realtime-14b)
- [krea-ai/realtime-video inference code](https://github.com/krea-ai/realtime-video)
