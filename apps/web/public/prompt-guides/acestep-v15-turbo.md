# ACE-Step v1.5 XL Turbo Music Guide

ACE-Step v1.5 XL Turbo is a **text-to-music** model — describe a piece and it composes and renders it. It powers the Audio Studio **Music** tab and also supports prompted audio editing of an existing clip.

## Installation

ACE-Step runs natively (Candle) on every platform. Install it once from the **Models** screen — it is about 11 GB and downloads into the shared Hugging Face cache from `ACE-Step/acestep-v15-xl-turbo-diffusers` (MIT). It ships its own Oobleck VAE, so there is no separately-licensed audio component.

## Writing the prompt

- Describe genre, mood, instrumentation, and tempo: "dreamy lo-fi hip-hop, mellow Rhodes, vinyl crackle, 80 BPM".
- Lyrics and prompts in 50+ languages are supported (English, Chinese, Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian, and more); the language tag is advisory.
- The turbo checkpoint is guidance-distilled, so it runs fast at a low step count (the reference default is around 8) — no separate CFG scale to tune.

## Editing existing audio

ACE-Step can edit a source clip through three modes:

- **Inpaint** — regenerate a bounded interior span fresh from the prompt.
- **Repaint** — regenerate a span while conditioning on the surrounding audio for continuity.
- **Extend** — continue the clip past its end, preserving the original.

## Duration

Output is 48 kHz stereo, up to **10 minutes**. Longer clips cost proportionally more time on CPU.
