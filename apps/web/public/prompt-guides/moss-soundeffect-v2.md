# MOSS SoundEffect v2 Guide

MOSS-SoundEffect v2.0 is a **text-to-audio** model for sound effects and ambience. You describe a sound and it synthesizes it — it is the model behind the Audio Studio **SFX** tab. It is not speech or music; it makes the world's noises.

## Installation

MOSS-SFX runs natively (Candle) on every platform. Install it once from the **Models** screen — it is about 11 GB (a Qwen3-1.7B text encoder, a 1.3B diffusion transformer, and a DAC VAE) and downloads into the shared Hugging Face cache from `OpenMOSS-Team/MOSS-SoundEffect-v2.0` (Apache-2.0).

## Writing the prompt

- Describe the source and its character: "heavy rain on a tin roof", "a distant thunderclap", "footsteps on gravel".
- Add audible context when it matters: action, material, distance, intensity, environment, and acoustic space.
- Bilingual prompts (English / Chinese) are supported; the language is advisory, not a mode switch.
- Leave **Guidance** and **Steps** blank unless you have a specific reason. Blank uses the values the model's authors published (CFG 4.0, 100 steps), which are what it was tuned for — more steps is not a quality dial to turn up.

Example:

> A heavy wooden cellar door creaks open slowly, rusty hinges grinding, close perspective in a damp stone room with a short dark echo; no speech or music.

**Refine my prompt** rewrites this sound description only. It does not change length, language, guidance, steps, or seed. Audio Studio does not expose a negative-prompt field for this model.

## Duration

Output is 48 kHz mono, up to **30 seconds**, with whole-second duration control. Ask for the length you need directly.

One thing that is easy to get wrong: **a short clip does not render faster.** The model always denoises a full 30-second internal window and then crops to your requested length — that is how it was trained, and shortening the window degrades quality badly. A 3-second effect therefore costs the same as a 30-second one, so prefer asking for the longer clip and trimming in the timeline over rendering several short ones.

## Practical notes

SFX generation is a diffusion process over a long sequence, and attention cost grows with the square of that length. On CPU a single render takes many minutes; on Apple GPU (Metal) it is a couple of minutes. If SFX feels unusably slow, check that the GPU audio path is enabled for your build.

Layer several clips in the timeline to build a richer ambience rather than asking for one long busy clip — the model renders a single coherent event far better than a crowded scene.
