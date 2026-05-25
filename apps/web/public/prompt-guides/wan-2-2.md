# Wan2.2 Prompt Guide

## Best For

Short text-to-video and image-to-video clips with clear subject motion, camera motion, and simple cinematic structure.

## Prompt Shape

For text-to-video:

`entity + scene + motion + aesthetic control + stylization`

For image-to-video:

`motion + camera movement`

If an image already defines the subject, setting, and style, focus the prompt on what moves and how the camera moves.

## Build The Prompt

### Subject

For text-to-video, define the main entity and visible traits. For image-to-video, refer to the source image subject directly.

### Scene Details

Describe background, foreground, lighting, weather, and time of day. Keep the scene focused so motion stays coherent.

### Style

Use concise style labels:

- `cinematic`
- `cyberpunk`
- `line art illustration`
- `wasteland style`
- `warm commercial video`

### Camera Angle And Movement

Wan benefits from direct camera language:

- `fixed camera`
- `camera pushes in`
- `camera moves left`
- `low angle`
- `wide shot`
- `medium close-up`

### Motion

Describe motion amplitude, speed, and effect:

- `slowly turns toward the window`
- `swaying gently in the wind`
- `shattering outward in small fragments`

For SceneWorks, keep clips short and write motion that can complete within the selected duration.

### Negative Prompt

Use negatives to reduce common video issues: `static frame, blurry details, subtitles, low quality, distorted hands, crowded background, extra limbs`.

## Example Prompts

`A white cat wearing sunglasses sits on a surfboard at a sunny beach. The cat looks toward the camera with a relaxed expression while small waves move around the board. The camera slowly pushes in from a medium shot to a close-up. Bright summer light, crystal blue water, soft background hills, playful commercial video style.`

`The source image comes alive with a gentle breeze. The woman's hair and scarf move softly, small leaves drift across the foreground, and sunlight flickers through the trees. Fixed camera, subtle natural motion, calm warm atmosphere.`

## Sources

- [Qwen Cloud video prompt guide](https://docs.qwencloud.com/developer-guides/accuracy-tuning/video-generation)
- [Wan2.2 GitHub repo](https://github.com/Wan-Video/Wan2.2)
- [Wan2.2 TI2V model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
