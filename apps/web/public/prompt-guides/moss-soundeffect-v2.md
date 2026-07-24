# MOSS SoundEffect v2 Guide

MOSS-SoundEffect v2.0 is a **text-to-audio** model for sound effects and ambience. You describe a sound and it synthesizes it — it is the model behind the Audio Studio **SFX** tab. It is not speech or music; it makes the world's noises.

## Installation

MOSS-SFX runs natively (Candle) on every platform. Install it once from the **Models** screen — it is about 11 GB (a Qwen3-1.7B text encoder, a 1.3B diffusion transformer, and a DAC VAE) and downloads into the shared Hugging Face cache from `OpenMOSS-Team/MOSS-SoundEffect-v2.0` (Apache-2.0).

## Writing the prompt

- Describe the source **and its character** — setting, material, and distance all condition the result: "heavy rain on a tin roof", "a distant thunderclap rolling over a valley", "footsteps on wet gravel". A bare noun phrase like "glass breaking" gives the model much less to work with than "glass shattering on a stone floor".
- Bilingual prompts (English / Chinese) are supported; the language is advisory, not a mode switch.
- A negative prompt steers the model away from unwanted qualities.

## Leave the sampling knobs alone unless you have a reason

The advanced panel exposes **guidance (CFG)** and **solver steps**. Both defaults are the values the model's authors published, and they are what the model was tuned for:

- **Steps — leave blank.** Blank resolves to the reference default of 100. More steps is not a quality dial to turn up, and fewer is not a safe speed trade.
- **Guidance — leave blank.** Blank resolves to the reference default of 4.0. 1.0 disables guidance entirely.

## Duration and render cost

Output is 48 kHz mono, up to **30 seconds**, with 0.1-second-granular duration control. Ask for the length you need directly.

One thing that is easy to get wrong: **a short clip does not render faster.** The model always denoises a full 30-second internal window and then crops to your requested length — that is how it was trained, and shortening the window degrades quality badly. A 3-second effect therefore costs the same as a 30-second one.

Because the cost is fixed, prefer **asking for the longer clip and trimming in the timeline** over rendering several short ones.

## Performance

SFX generation is a diffusion process over a long sequence, and attention cost grows with the square of that length. On CPU a single render takes many minutes; on Apple GPU (Metal) it is a couple of minutes. If SFX feels unusably slow, check that the GPU audio path is enabled for your build.

## Practical notes

Layer several clips in the timeline to build a richer ambience rather than asking for one long busy clip — the model renders a single coherent event far better than a crowded scene.
