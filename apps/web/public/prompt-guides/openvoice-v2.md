# OpenVoice V2 Voice Conversion Guide

OpenVoice V2 is a **tone-color voice-conversion** model. It takes source speech and a short reference clip of a target voice, then re-renders the source in the target's timbre while preserving the original content and prosody. It is one of the models behind the Audio Studio **VoiceClone** tab.

## Installation

OpenVoice runs natively (Candle) on every platform. Install it once from the **Models** screen — the converter is about 130 MB and downloads into the shared Hugging Face cache from `myshell-ai/OpenVoiceV2` (MIT).

## How it works

This is a **prompt-free** transform — there is no text prompt. You provide two inputs:

- **Source audio** — the speech whose words and delivery you want to keep.
- **Target reference** — a few seconds of the voice whose timbre you want to apply.

The converter extracts the target's tone color and transfers it onto the source. Output is 22.05 kHz; it does not resample, so keep the studio's output rate at the model's native rate.

## Practical notes

- A clean, dry reference clip gives the best timbre transfer.
- A strength control adjusts how strongly the target timbre is applied.
- Because content and prosody come from the source, record the performance you want first, then convert the voice.
