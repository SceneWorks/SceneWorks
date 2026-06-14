# SCAIL-2 Prompt Guide

## Best For

**Character animation** — you have a reference image of a character and a driving video of someone
(or something) moving, and you want your character to perform that motion. SCAIL-2 transfers the
driving clip's motion and timing onto your character while keeping the character's identity and look.

> SCAIL-2 is heavy: it runs a segmented 14B denoise, so a clip takes several minutes. Keep clips short
> and choose a driving video whose motion completes within the selected duration.

## What You Provide

| Input | What it's for |
|---|---|
| **Reference character image** | The character to animate — its identity, face, wardrobe, and look. Use a clean, well-lit, unobstructed image where the whole subject is visible. |
| **Driving video** | The motion and timing to transfer. The output matches the driving clip's framing and aspect, so pick a bucket (e.g. 480×832 portrait, 832×480 landscape) that matches it. |
| **Prompt** | Describes the result — the scene, mood, and any details to reinforce. SCAIL-2 leans on the reference + driving for identity and motion, so the prompt is supporting guidance, not the whole story. |

The color-coded segmentation masks SCAIL-2 needs are generated for you from the driving clip and the
reference image — you don't supply masks.

## Choosing A Driving Video

The driving clip is the most important input after the reference:

- **One clear subject, fully in frame.** SCAIL-2 segments the moving person automatically; a single,
  unobstructed subject animates most cleanly.
- **Motion that fits the duration.** A gesture or step that completes within 5s reads better than a
  long action that gets cut off.
- **Steady framing.** Heavy camera shake or rapid cuts make the transferred motion noisier.
- **Match the aspect.** The output adopts the driving clip's framing — a portrait driver gives a
  portrait result.

## Choosing A Reference

- **Clean and well-lit**, with the character's face and body clearly visible.
- **Front-facing or three-quarter** views hold identity better than extreme angles.
- **Uncluttered background** — the character should be the obvious subject.

## Prompt Shape

`subject + scene + mood/style`

Because the reference carries identity and the driving clip carries motion, keep the prompt focused on
the **scene and look** you want the animated character placed into:

- Reinforce the character (`a young woman in a red jacket`) so the identity stays stable.
- Set the scene (`on a city street at dusk`, `in a softly lit studio`).
- Add a style label (`cinematic`, `photorealistic`, `warm commercial video`).

### Negative Prompt

Reduce common video artifacts: `blurry details, distorted face, extra limbs, flicker, warping,
duplicated subject, low quality`.

## Quality & Speed Notes

- **Resolution:** use a native bucket that matches the driving clip (480×832 / 832×480 / 720p). Very
  small frames look degraded.
- **Quantization:** Q4 is the default (fits ~48–64 GB Macs, faster). Opt into Q8 (advanced setting)
  for a small quality gain if you have the memory.
- **Steps:** more steps = cleaner motion but longer renders; keep them modest while iterating.
- **Duration:** longer clips stitch multiple 14B denoise segments — each segment adds minutes. Keep
  clips at 5s or less.

## Example Prompts

`A young woman with long dark hair in a green raincoat, walking through a rain-soaked city street at
night, neon reflections on the pavement. Cinematic, moody, shallow depth of field.`

`An older man in a tweed jacket gesturing as he speaks, seated in a warm wood-paneled study, soft
window light. Photorealistic, documentary feel.`

## Sources

- [SCAIL-2 model card](https://huggingface.co/zai-org/SCAIL-2)
- [SCAIL-2 GitHub repo](https://github.com/zai-org/SCAIL-2)
- [Wan2.1 GitHub repo](https://github.com/Wan-Video/Wan2.1)
