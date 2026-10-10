# YuE2 Song Generation Guide (Experimental, Noncommercial)

YuE2 writes a whole song with vocals from a **style** description and **lyrics**. Before it renders any audio it can plan the song as a two-voice ABC score (melody plus chords), which you can inspect, reuse and edit. It renders 48 kHz stereo.

## Licence first

YuE2 is **experimental** and for **noncommercial use only**. Its weights (YuE2-3B and the YuE2 VAE decoders, by Multimodal Art Projection) are licensed under **CC BY-NC 4.0**, and its `qwen.tiktoken` tokenizer is under the **Tongyi Qianwen License Agreement**. Models asks you to accept these terms before the download starts, and the full texts are under **About → Licenses**.

YuE2 is a separate model from **YuE1**, the lyrics-to-song models intended for commercial use. SceneWorks never swaps one for the other: a YuE1 recipe stays YuE1, and a commercial-use workflow refuses YuE2 and points you to YuE1. Choosing YuE1 covers the model weights only. It does not clear rights in your lyrics, reference recordings or generated audio.

## Installation

YuE2 runs natively (Candle) on every platform. A base install is about 7.8 GB, downloaded straight from the upstream `m-a-p` repositories. SceneWorks does not re-host these files. The install contains:

- **YuE2-3B** and its tokenizer (about 7.3 GB), the released bf16 checkpoint;
- **one decoder**: the standard YuE2 VAE by default, or the legacy VAE if you choose it. You can add the other decoder later.

The **q8** and **q4** tiers are not separate downloads. SceneWorks derives them on your machine from the bf16 original and checks them against pinned checksums, so choosing a tier still downloads the bf16 original.

Covers of a source recording need the SheetSage2 and MERT-v2-FullSong transcription models. They are **not available yet**: the owner has not yet made a licensing decision about the transcription code. Base generation never needs them.

## Style

Describe the recording in a short phrase or sentence: genre, instrumentation, vocal character, mood and tempo feel. For example:

> Warm acoustic folk with fingerpicked guitar, soft brushed drums and an intimate male lead vocal, gentle and nostalgic.

YuE2 has no separate BPM, key, language or voice controls. Put those qualities in the style text instead.

## Lyrics

Write the lyrics with section tags such as `[Verse]`, `[Chorus]` and `[Bridge]` on their own lines, and leave a blank line between sections. The song's length follows the lyrics and the token budget. There is no target-duration control.

## Planning modes

- **Full** (the default) plans a chord-annotated melody score before rendering.
- **Melody** plans the melody without chords.
- **Off** renders straight from the style and lyrics, with no score.

You can supply your own ABC score for the Full and Melody modes, or reuse a saved plan exactly. An edited score is always a new request, and it never overwrites the original.
