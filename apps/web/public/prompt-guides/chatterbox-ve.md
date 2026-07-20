# Chatterbox Voice Encoder Guide

The Chatterbox voice encoder (Resemble AI) maps a few seconds of reference audio to a **256-dimensional voice-identity embedding**. That embedding is the identity signal for cloned-voice conditioning — the audio analogue of an ArcFace face embedding feeding InstantID. It is a building block for the Audio Studio **VoiceClone** tab, not a generator on its own.

## Installation

The encoder runs natively (Candle) on every platform. Install it once from the **Models** screen — it is a single ~5.7 MB `ve.safetensors` file that downloads into the shared Hugging Face cache from `ResembleAI/chatterbox` (MIT). Only the voice-encoder weights are fetched, not the full Chatterbox TTS model.

## How it works

This is a **prompt-free** component. You provide one input:

- **Reference audio** — a short clip (a couple of seconds is enough) of the voice to characterize.

The encoder resamples to 16 kHz, extracts mel frames, and produces an L2-normalized speaker vector averaged over short partial utterances. That vector then rides into a cloned-voice generator as its identity conditioning.

## Practical notes

A clean, single-speaker reference produces the most distinctive embedding. Longer or noisier clips do not help — a few clear seconds is the sweet spot.
