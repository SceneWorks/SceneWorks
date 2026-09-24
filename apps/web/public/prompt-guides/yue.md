# YuE Lyrics-to-Song Guide

YuE (HKUST / M-A-P) is a **lyrics-to-song** model: you give it structured lyrics and a short list of genre tags, and it composes and sings a full song — vocals and accompaniment together. Every render returns a 44.1 kHz mix plus separate **vocal** and **instrumental** stems.

## Which checkpoint

There are six YuE models: one per lyric language × prompting mode.

- **Language** — pick the checkpoint that matches the language of your lyrics: **English**, **Chinese**, or **Japanese / Korean**.
- **CoT** (chain-of-thought) — lyrics + genre tags only. Start here.
- **ICL** (in-context learning) — additionally takes a **reference song clip** and follows its style. The reference can be a single mixed track, or a vocal + instrumental pair (the pair usually steers better). By default the first 30 seconds of the clip are used; you can choose a different window — a chorus is usually the most useful part.

## Installation

YuE runs natively (Candle) on every platform. Each model downloads one weight tier — **q4** by default, or **q8** / **bf16** — plus the stage-2 upsampler at the same tier and the shared xcodec codec with its two Vocos decoders. The codec and the stage-2 files are shared, so a second YuE model only adds its own stage-1 weights. At q4 a first install is about 7.4 GB; bf16 is about 17.7 GB. All weights are Apache-2.0.

## Genre tags (the prompt)

The prompt is a space-separated list of tags, not a sentence. Cover five things: **genre**, **instrument**, **mood**, **vocal gender**, and **vocal timbre**.

> inspiring female uplifting pop airy vocal electronic bright vocal

Keep it short and concrete; a long descriptive paragraph does not help.

## Lyrics

Write the lyrics in sections, each headed by a label on its own line, with a blank line between sections:

```
[verse]
Staring at the sunset, colors paint the sky
Thoughts of you keep swirling, can't deny

[chorus]
Don't let this moment fade, hold me close tonight
With you here beside me, everything's alright
```

Use `[verse]`, `[chorus]`, `[bridge]` and `[outro]`. YuE renders the song **one section at a time** — roughly 30 seconds per section — so:

- give each section enough lines to fill that time (a four-line verse or chorus is a good size);
- the song's length follows the lyrics and the number of sections you render, not a duration setting;
- rendering more sections makes a longer song and takes proportionally longer.

## Settings

The defaults are the values the model's authors published and are what it was tuned for: guidance **on** (1.5 for the first section, 1.2 after), repetition penalty **1.1**, up to **3000** tokens per section, **2** sections, seed **42**. Change them deliberately — for example raise the section count for a longer song, or turn guidance off to trade prompt adherence for variety.

## Practical notes

YuE is the heaviest audio model in the app: a 7B language model writes the song, then a 1B model and the codec render it. Expect several minutes per minute of music on a fast GPU, and longer on Apple Silicon. Progress is reported per section, and a render can be cancelled at any point.

Upstream asks (but does not require) that you credit outputs as *"Generated with YuE by HKUST/M-A-P"*, and recommends labeling shared songs as AI-generated. You are responsible for making sure a song you publish does not reproduce existing material.
