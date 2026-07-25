# Chatterbox Clone-TTS Guide

Chatterbox Clone-TTS (Resemble AI) renders a **full cloned-voice voiceover directly from your script plus a short reference clip, in a single step**. Unlike the OpenVoice conversion chain — which synthesizes a base voice and then re-timbres it — Chatterbox is a native cloned-voice generator: a T3 speech-token language model conditioned on the reference speaker drives the S3Gen token-to-waveform stack (flow-matching decoder + HiFTNet vocoder), and a PerTh provenance watermark is applied to the output. It powers the Audio Studio **Voice Clone** tab and is preferred automatically whenever it is installed.

## Installation

The model runs natively (Candle) on every platform. Install it once from the **Models** screen — the T3 checkpoint, S3Gen checkpoint, tokenizer, speaker voice encoder, and PerTh watermarker are provisioned as the model’s pinned downloads in the shared Hugging Face cache (`ResembleAI/chatterbox`, MIT).

## How it works

You provide two inputs:

- **Script** — the words to speak, typed into the prompt box. This is the text rendered in the cloned voice (not a description of the voice).
- **Reference voice** — a short library audio clip of the target voice. The model derives the speaker identity from it and uses it as the acoustic reference for the waveform, so one call produces the cloned voiceover.

There is no separate base-voice or match-strength control: the clone is rendered end to end in the reference voice at 24 kHz mono, up to about 30 seconds per generation.

**Refine my prompt** treats the prompt as a script. It preserves the words and meaning and makes only minimal punctuation or readability adjustments; it does not change the selected reference voice.

## Practical notes

A clean, single-speaker reference of a few clear seconds gives the most faithful clone; longer or noisier clips do not help. Keep scripts to natural sentences — the utterance length follows the text, so break very long copy into separate generations. English input is expected.
