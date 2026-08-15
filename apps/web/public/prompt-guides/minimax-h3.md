# MiniMax-H3 Prompt Guide

## Best For

**Short clips that need sound.** MiniMax-H3 (Hailuo 3.0) is a flow-matching diffusion transformer
that generates the video **and a synchronized stereo soundtrack in the same pass** — dialogue,
footsteps, room tone, music — rather than dubbing audio on afterwards. The clip arrives already
muxed. If a shot only needs pictures, a cheaper model will get you there faster; if the sound has to
match what is on screen, this is the one that does it natively.

Two entries ship from the same family:

| Entry | What it does |
|---|---|
| **MiniMax-H3** | Text-to-video, and first and/or last frame conditioning. |
| **MiniMax-H3 References** | Reference-driven video — up to 9 images, 3 video clips and 3 audio clips (12 files total). |

> **This is the open-weights model, not the hosted Hailuo product.** Four things MiniMax's own stack
> has, these weights do not, and each one changes how you should prompt:
>
> - **H3-Context-IR**, the hosted prompt-understanding front end. Prompts here go to the model
>   verbatim, so **prompt adherence differs from the API** no matter how the port is written. Write
>   more explicitly than you would for the hosted product.
> - **H3-Regenerate-2K**, the in-context 2K upscaler. **2K is not reachable from these weights.** The
>   checkpoint's canvas budget is about 1.03 megapixels — 1344x768 at 16:9 — and a request over the
>   budget is refused rather than refitted. A 2K frame is more than twice that area, so plan the
>   shot at the sizes below — and note that the bound is the AREA, so a wider canvas buys you a
>   shorter one rather than more pixels.
> - **Sparse-attention inference.** The open weights run *dense* attention, which is why renders are
>   measured in hours at the large canvas. It is not what caps the clip at 14.38 s — that ceiling is
>   the checkpoint's own — but it is why the long end of the range is expensive.
> - **The `<d>` dialogue markers**, and the six other special tokens the model card's examples use.
>   They are declared in the tokenizer and **do nothing** against these weights — see *Markers from
>   the upstream examples that do nothing*, below.

## Cost Comes From The Canvas, Not The Length

This is the single most important thing to internalise before your first render. Measured 50-step
timings on real weights:

| Canvas | Clip | Approx. render time |
|---|---|---|
| 576x320 | 5.17 s | ~14 minutes |
| 576x320 | 14.38 s | ~43 minutes |
| 1344x768 | 5.17 s | **~2 hours** |
| 1344x768 | 14.38 s | many hours |

The *shortest* clip at the default 1344x768 canvas already costs about two hours. The *entire*
duration range is comfortable at 576x320. **Choose the small canvas while you are still iterating on
the prompt**, and move up only once you have a shot you want at full size.

## Fourteen Clip Lengths, And Nothing In Between

The video autoencoder packs frames in groups of 17, so a legal frame count is `17n + 5`, and the
checkpoint clamps the result to roughly 5–15 seconds at 24 fps. What survives is exactly **fourteen**
renderable lengths:

`5.17 · 5.88 · 6.58 · 7.29 · 8.00 · 8.71 · 9.42 · 10.13 · 10.83 · 11.54 · 12.25 · 12.96 · 13.67 ·
14.38` seconds.

The Duration menu offers those and only those. A length outside the range is **refused**, not quietly
rounded — including the 15 seconds MiniMax's own documentation advertises, which sits between the
last rung (14.38 s) and the next one (15.08 s) and cannot be rendered at all. There is likewise **no
single-image mode**: the model will not render fewer than about five seconds, so it cannot be used as
a stills generator.

Frame rate is fixed at **24 fps** — the lattice, the duration window and every measured render assume
it.

## Sizes

There is no aspect-ratio control. The canvas comes from your first keyframe's shape, or defaults to
16:9 when there is no keyframe, and every side is rounded to a multiple of 32 inside a fixed area
budget. Ratios between 1:4 and 4:1 are accepted.

| Bucket | Use |
|---|---|
| 576x320 / 320x576 | **Start here.** Fast enough to iterate on. |
| 1024x768 / 768x1024 | 4:3 and 3:4. |
| 768x768 | Square. |
| 1344x768 / 768x1344 | Full size, 16:9 and 9:16. The default, and the expensive one. |
| 1536x672 / 672x1536 | 21:9 and its transpose. Same pixel budget as full size, same cost. **Pending an engine change — see below.** |

Every bucket in the table is the same fixed area budget or less — the 21:9 pair is 1,032,192 px,
byte for byte what 1344x768 is. **The bound is the area, not the long edge**, which is why a wider
canvas is not automatically a bigger one.

> **The 21:9 pair is not renderable yet.** The area budget already accommodates it, but the engine
> *also* caps each edge independently, and that per-edge ceiling currently sits below this pair's
> long edge — so a 21:9 request is refused today even though the menu offers it. **inference PR
> #640** raises the per-edge ceiling to the widest canvas the resolver can itself produce, which is
> what makes this pair reachable. Until it lands, treat **1344x768** as the widest canvas that
> actually renders; every other bucket above works now. **This row and #640 must not be separated at
> merge** — shipping this table without that engine change re-advertises a canvas the engine
> refuses.

A keyframe is **stretched** onto the canvas, not letterboxed — crop your reference to the target
shape first or the subject will distort.

## No Negative Prompt, No Guidance

The pipeline's entire input surface is the prompt, the conditioning media, the geometry, the step
count and a seed. There is **no guidance scale and no negative prompt** — so the Video Studio shows
neither control for this model, and that is correct rather than missing. Everything you want has to
be stated positively:

- Instead of a negative "blurry, low quality" → say **"sharp, crisp focus, clean detail"**.
- Instead of a negative "no music" → say **"no non-diegetic music, room tone only"**.
- Instead of a negative "not cluttered" → say **"minimal background, single subject"**.

## Writing The Prompt

Write one continuous paragraph rather than a keyword list, and — because the hosted prompt-rewriter
is not in the loop — say the quiet parts out loud. A prompt that works well names, in roughly this
order:

1. **Subject** — who or what, concretely.
2. **Action** — one clear motion that can start and finish inside the clip.
3. **Setting** — where, and the light.
4. **Camera** — shot size and movement.
5. **Look** — film stock, palette, rendering style.
6. **Sound** — see below. This is the part people forget.

> *A street violinist in a worn green coat draws a long slow bow across her instrument, standing
> under a subway station arch as commuters blur past behind her, warm tungsten light against cold
> tile, medium shot slowly pushing in, grainy 35mm film look. Audio: a single sustained violin note
> over station reverb, distant train rumble, footsteps passing close to camera.*

### Prompt the audio explicitly

The soundtrack is generated from the same prompt as the picture, so **whatever you do not describe,
the model invents**. Naming the sound is the difference between a usable take and a clip with
plausible-but-wrong ambience. Useful things to state:

- **Diegetic sound** — what the visible action makes. *Footsteps on gravel. A knife on a board.*
- **Ambience** — the space. *Empty warehouse reverb. Rain on a car roof. Quiet room tone.*
- **Voice** — if someone speaks, give the line and the delivery. *She says "not tonight", quietly,
  half-turned away.* Eleven languages are stable; short lines land better than long ones.
- **Music** — instrumentation and mood, if you want any. Say **"no music"** if you do not.
- **Perspective** — *close-miked* versus *distant and roomy* changes the mix noticeably.

## The Structured Schema — Worth Learning

MiniMax's own writing guides describe a labelled prompt format, and it is **not cosmetic**. In a
controlled A/B on identical seed, geometry and step count, the structured form produced things the
prose form could not:

- **Timed shot changes.** A prompt asking for `[Shot 2] At 00:03.000, the shot cuts to a closer
  low-angle view…` produced a hard cut within one frame of 3.000 s. The prose version of the same
  request produced no cut anywhere. **This has no prose equivalent** — it is the single strongest
  reason to use the schema.
- Measurably sharper frames, more motion, and a louder, wider, brighter soundtrack.

The useful labels:

| Label | What goes in it |
|---|---|
| `[Shot N] At MM:SS.mmm` | A timed shot. Framing, angle, and what changes at that moment. |
| `integrated_multimodal_description:` | The scene as a whole — picture and sound together. |
| `overall_soundscape:` | Diegetic sound and ambience. |
| `non_diegetic_music:` | Score, if any. *"Sustained low strings."* |
| `(S1)`, `(S2)` | Speaker identities, when more than one character talks. |

> `integrated_multimodal_description: A red fox picks its way along a snow-covered log in a spruce
> forest at dusk, snow-laden branches passing through the foreground.`
> `[Shot 1] At 00:00.000, a slow tracking medium shot follows the fox from the left.`
> `[Shot 2] At 00:03.000, the shot cuts to a closer low-angle view of the fox pausing, ears
> swivelling toward an off-screen sound.`
> `overall_soundscape: crunching snow underfoot, wind through branches, one distant crow.`
> `non_diegetic_music: sustained low strings, quiet.`

Part of the schema's advantage is simply that a structured prompt tends to be longer and more
specific, and you would get some of that from a detailed paragraph too. The timed cuts are the part
you cannot get any other way.

### Markers from the upstream examples that do nothing

The model card's prompt examples use a `<d>[English] …</d>` dialogue syntax, and its tokenizer
declares seven special markers in total:

`<d>` · `</d>` · `<|cutoff|>` · `<|lyrics_start|>` · `<|lyrics_end|>` · `<|caption_start|>` ·
`<|caption_end|>`

**None of them do anything here.** They are declared as strings in the tokenizer config, but the
open text encoder has no trained representation for them — the embedding rows they resolve to are
indistinguishable from the model's unused padding. They almost certainly belong to the withheld
H3-Context-IR front end, which is not in this loop.

**Nothing strips or repairs the markers.** They are not removed, not rewritten and not warned about
at submit time. Two features can still change what the text encoder sees, and both are things you
switched on yourself:

- The **Refine** button hands your prompt *and this guide* to a language model and shows you the
  rewrite as a suggestion. The box does not change until you press **Apply** — *Keep original*
  leaves it exactly as you typed it. But because the model has just read this page, a rewrite you
  do apply may well drop the markers.
- A **Style Catalog** entry or a **preset stack** is folded into the outgoing prompt at submit time.
  There is no confirmation step: the Studio previews the composed string above the Generate button,
  and then sends it.

With no style, no stack and no applied refinement, whatever is in the box is exactly what the text
encoder sees. Either way the markers cost you prompt space and contribute nothing. **Write dialogue
as plain text instead** — name the speaker, give the line, give the delivery, exactly as in the
*Prompt the audio explicitly* section above. That works; the markup does not.

### Keep the action inside the clip

At 5–14 seconds, one completed beat is the right amount of story. "Turns and smiles" works. "Walks
across the room, opens the door, and steps outside" does not — you will get a fragment of it. If the
audio has dialogue, count it out loud: a line that takes eight seconds to say will not fit in a
5.17-second clip.

## First And Last Frame

Supply a first frame, a last frame, or both. With both, the model interpolates a plausible motion
between them; the prompt still governs style, camera and — importantly — all of the sound. Keep the
two frames in the same scene and the same shape: a large jump between them reads as a cut rather than
a move.

## References

The **MiniMax-H3 References** entry conditions on media instead:

- Up to **9 images** — subjects, characters, wardrobe, locations.
- Up to **3 video clips** — motion and pacing to imitate.
- Up to **3 audio clips** — a voice to match, or a sound bed to sit in.
- **12 files in total**, so the three lists cannot all be filled at once.

**Order is meaningful.** References are labelled in the order you supply them — `<Picture 1>`,
`<Picture 2>`, `<Audio 1>`, `<Video 1>` — and the labels are what the prompt can refer to, so
reordering the same files is a genuinely different request rather than a shuffle. Name in the prompt
what each reference is *for*: "the woman from `<Picture 1>`, in the alley from `<Picture 2>`,
speaking with the voice from `<Audio 1>`" gives the model an assignment; a bare pile of files gives
it a guess.

Unlike a first/last frame, **references do not set the canvas shape** — they are encoded at their own
resolution and the output falls back to 16:9 unless you choose a size.

## Steps

Fifty steps is the reference default and what the timings above assume. Fewer steps scale the cost
almost linearly, so 25 steps roughly halves the render — a reasonable trade while you are drafting.
Below about 20 the motion starts to smear and the audio loses definition.

## Quant Tiers

Three tiers ship. All three render coherently; the difference is disk and, once block streaming
lands, memory:

- **q4** (default) — smallest. Measurably softer and a little darker than bf16, but sound.
- **q8** — visually equivalent to bf16 at about half the bytes. The value pick.
- **bf16** — full precision, and by far the largest download.

The text encoder and both autoencoders are shared across every tier and across both entries, so they
download once no matter how many tiers you install.

## Sources

- [MiniMax-H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Hailuo 3.0 prompting guide](https://hailuoai.video/)
- [MiniMax platform video generation docs](https://platform.minimax.io/docs/api-reference/video-generation)

Written for SceneWorks against the open-weights checkpoints. The cost table, the fourteen clip
lengths, the size buckets and the structured-schema A/B are measured on this build rather than
quoted, so they describe what these weights do — which is not always what the hosted product does.
