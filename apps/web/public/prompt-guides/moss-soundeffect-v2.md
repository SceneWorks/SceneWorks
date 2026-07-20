# MOSS SoundEffect v2 Guide

MOSS-SoundEffect v2.0 is a **text-to-audio** model for sound effects and ambience. You describe a sound and it synthesizes it — it is the model behind the Audio Studio **SFX** tab. It is not speech or music; it makes the world's noises.

## Installation

MOSS-SFX runs natively (Candle) on every platform. Install it once from the **Models** screen — it is about 11 GB (a Qwen3-1.7B text encoder, a 1.3B diffusion transformer, and a DAC VAE) and downloads into the shared Hugging Face cache from `OpenMOSS-Team/MOSS-SoundEffect-v2.0` (Apache-2.0).

## Writing the prompt

- Describe the source and its character: "heavy rain on a tin roof", "a distant thunderclap", "footsteps on gravel".
- Bilingual prompts (English / Chinese) are supported; the language is advisory, not a mode switch.
- Guidance (CFG) sharpens adherence to the prompt; the reference default is around 4.0, and 1.0 turns guidance off.
- A negative prompt steers the model away from unwanted qualities.

## Duration

Output is 48 kHz mono, up to **30 seconds**, with 0.1-second-granular duration control. Ask for the length you need directly.

## Practical notes

SFX generation is a diffusion process — more solver steps trade time for fidelity. Layer several short clips in the timeline to build a richer ambience rather than asking for one long busy clip.
