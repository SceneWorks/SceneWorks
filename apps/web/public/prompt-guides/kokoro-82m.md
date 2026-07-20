# Kokoro 82M Speech Guide

Kokoro-82M is a compact, high-quality **text-to-speech** model (StyleTTS2 lineage). You give it a script and a voice, and it synthesizes natural speech. It is the recommended model for the Audio Studio **Speech** tab.

## Installation

Kokoro runs natively (Candle) on every platform. Install it once from the **Models** screen — it is about 330 MB (the checkpoint plus the per-voice style packs) and downloads into the shared Hugging Face cache from `hexgrad/Kokoro-82M` (Apache-2.0, ungated).

## Voices

Kokoro ships **28 English voices** — 20 American (`af_*` female, `am_*` male) and 8 British (`bf_*` female, `bm_*` male). The voice prefix picks the pronunciation variant: an American voice uses the US front-end, a British voice the GB front-end. `af_heart` is the default showcase voice.

## Writing the script

- Plain prose works best — write it the way you want it read aloud.
- Punctuation drives prosody: commas and periods pace the delivery; question marks lift the intonation.
- Output is 24 kHz mono, up to about **30 seconds** per request. For longer passages, split into multiple clips.
- An optional target duration nudges the pace (the model derives a speed factor, clamped to 0.5–2.0×).

## Practical notes

Kokoro is single-speaker per request — render each speaker separately for a dialogue. It is small and fast, so iterating on wording and voice is cheap.
