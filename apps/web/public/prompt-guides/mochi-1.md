# Mochi 1 Prompt Guide

## Best For

Text-to-video clips with strong, coherent motion at 848x480 / 30 fps. Mochi 1 is a 10B AsymmDiT trained for photorealistic motion and prompt adherence.

Mochi 1 is **text-to-video only** — it has no image, reference, or keyframe conditioning. If you need to animate an existing image, use an image-to-video model instead.

## Prompt Shape

Write one flowing, descriptive paragraph rather than a keyword list. Mochi responds to natural prose:

`subject + action + scene + camera + lighting + style`

Mochi supports a negative prompt and true CFG, so you can steer both toward and away from content.

## Build The Prompt

### Subject And Action

Lead with the subject and give it a clear, continuous action. Mochi's strength is motion — a prompt with an explicit verb ("walking", "turning", "pouring") produces far better results than a static description.

### Scene Details

Add location, time of day, weather, and props. Concrete nouns anchor the scene better than abstract mood words.

### Camera

Name the shot and any movement:

- `close-up`, `medium shot`, `wide establishing shot`
- `slow dolly in`, `tracking shot`, `static camera`, `handheld`

### Lighting And Style

Name the light and the look:

- `golden hour`, `soft overcast light`, `harsh noon sun`, `neon rim light`
- `photorealistic`, `cinematic 35mm`, `documentary`

## Settings

- **Resolution** — 848x480 (landscape) or 480x848 (portrait). This is Mochi's native and only trained bucket; other sizes are out of distribution.
- **Frames per second** — 30. Mochi is trained at 30 fps, and fps here sets the generated clip length as well as playback speed.
- **Duration** — the native design point is about 5 seconds. Shorter clips cut memory use roughly linearly, so start at 1-2 seconds if you are memory-constrained.
- **Steps** — 64 is the default and a good starting point.
- **Guidance** — 4.5 by default. Raise it for tighter prompt adherence, lower it for more natural motion.

## Tips

- Describe motion explicitly. Mochi is a motion model first; a prompt with no verb tends to produce a near-still clip.
- Prefer one connected paragraph over comma-separated tags.
- Use the negative prompt for artifacts you keep seeing (`blurry, distorted hands, static, watermark`).
- Keep the subject count low. Crowded scenes dilute motion quality at 480p.

## Limits

- No LoRA support on either backend.
- No image conditioning, first/last frame, clip extension, or person replacement.
- One clip per run.

## Licensing

Mochi 1 is Apache-2.0 — commercial use is free and the weights are ungated.

## Sources

- [Mochi 1 model card](https://huggingface.co/genmo/mochi-1-preview)
- [Genmo Mochi 1 GitHub repo](https://github.com/genmoai/mochi)
