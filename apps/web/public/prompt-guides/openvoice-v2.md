# OpenVoice V2 Voice Conversion Guide

OpenVoice V2 is the conversion backend for Audio Studio’s **Voice Clone** fallback. You type a script and select a target reference voice. SceneWorks first synthesizes the script with its base speech model, then OpenVoice transfers the reference clip’s timbre onto that speech.

## Installation

OpenVoice runs natively (Candle) on every platform. Install it once from the **Models** screen — the converter is about 130 MB and downloads into the shared Hugging Face cache from `myshell-ai/OpenVoiceV2` (MIT).

## How it works

You provide two user-facing inputs:

- **Script** — the exact words to speak.
- **Reference voice** — a few clean seconds of the target voice.

SceneWorks creates the source speech internally; there is no source-audio picker in this flow. The converter extracts the target’s tone color and transfers it to that generated speech. Output is 22.05 kHz.

**Refine my prompt** treats the prompt as a script: it preserves the words and meaning, making only minimal punctuation or readability adjustments. It does not describe the target voice or change the selected reference.

## Practical notes

- A clean, dry reference clip gives the best timbre transfer.
- Match strength controls how strongly the reference timbre is applied.
