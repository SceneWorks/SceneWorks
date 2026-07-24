# ACE-Step v1.5 XL Turbo Music Guide

ACE-Step v1.5 XL Turbo is a **text-to-music** model — describe a piece and it composes and renders it. It powers the Audio Studio **Music** tab and also supports prompted audio editing of an existing clip.

## Installation

ACE-Step runs natively (Candle) on every platform. Install it once from the **Models** screen — it is about 11 GB and downloads into the shared Hugging Face cache from `ACE-Step/acestep-v15-xl-turbo-diffusers` (MIT). It ships its own Oobleck VAE, so there is no separately-licensed audio component.

## Writing the prompt

Keep the prompt to the **musical description** — genre, mood, instrumentation, production style: "dreamy lo-fi hip-hop, mellow Rhodes, vinyl crackle, tape saturation".

Tempo and key are **separate fields**, not prompt text. The model receives them on a dedicated metadata channel it was trained on, so filling them in conditions the result far more reliably than writing "80 BPM" into the description:

- **BPM** — set it whenever you have a tempo in mind. Left empty it is passed as "not specified" and the model picks one.
- **Key / scale** and **time signature** — same. Any field you leave blank is genuinely unconstrained, not defaulted to something sensible.

Prompts and lyrics in 50+ languages are supported (English, Chinese, Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian, and more).

## Lyrics improve structure, even a little

Supplying lyrics gives the model a second conditioning channel and measurably tightens rhythmic structure — songs come out with a clearer beat and more defined sections than the same prompt rendered instrumental. Use the `[verse]` / `[chorus]` tags to mark sections.

If you want an instrumental, just leave lyrics empty — that is a supported mode, not a degraded one.

## Length and coherence

Output is 48 kHz stereo, up to **10 minutes**, but musical coherence declines as the clip gets longer — a 30-second render holds its beat less consistently than a 12-second one, and the effect continues from there. This is a property of the model, not a setting to tune.

For longer pieces, prefer generating shorter sections and arranging them in the timeline, or use **Extend** to continue a clip you already like.

## Steps and guidance

The turbo checkpoint is **guidance-distilled**: CFG is baked into the weights, so there is no guidance scale to tune (the Studio hides it for this model — supplying one is rejected rather than silently ignored). It runs at a low step count by design; the reference default is 8, and raising it does not improve coherence.

## Seeds

Leaving the seed blank picks a new one each render, so the same prompt gives a different take every time. If you get a result you like, **pin its seed** to keep it and to make small prompt edits comparable. Re-rolling the seed is a legitimate way to shop for a better take.

## Editing existing audio

ACE-Step can edit a source clip through three modes:

- **Inpaint** — regenerate a bounded interior span fresh from the prompt.
- **Repaint** — regenerate a span while conditioning on the surrounding audio for continuity.
- **Extend** — continue the clip past its end, preserving the original.
