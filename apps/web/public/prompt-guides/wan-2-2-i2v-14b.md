# Wan2.2 14B Image-to-Video Prompt Guide

## Best For

Animating a still image while preserving its subject, setting, and style.

## Prompt Shape

For image-to-video, keep the prompt focused:

`subject motion + environmental motion + camera movement + preservation constraints`

The input image already defines the entity, scene, and visual style. Do not redescribe everything unless you need to override it.

## Build The Prompt

### Subject

Refer to the image subject directly:

- `the person in the image`
- `the car`
- `the foreground flowers`
- `the clouds in the background`

### Details To Preserve

Say what should stay fixed: identity, clothing, composition, background, lighting, or camera angle.

### Style

Usually preserve the source image style. If changing style, state it clearly, but expect stronger results when style remains consistent.

### Camera Angle And Movement

Use simple camera movement:

- `fixed camera`
- `slow push-in`
- `gentle pan left`
- `small handheld drift`

### Motion

Describe motion in layers:

- subject motion: `turns slightly`, `blinks`, `raises one hand`
- environment motion: `curtains sway`, `rain falls`, `dust drifts`
- camera motion: `camera slowly pulls back`

For SceneWorks, keep clips at 5 seconds or less and avoid complex action chains.

### Negative Prompt

Use negatives for drift and artifacts: `changed identity, warped face, static frame, blurry details, extra limbs, distorted hands, subtitles`.

## Example Prompts

`Keep the same subject, outfit, and background from the image. The person slowly turns their head toward the window and blinks once while their hair moves gently in a breeze. Dust particles drift through the light. Fixed camera, subtle realistic motion, preserve the original lighting and composition.`

`Animate the landscape with a slow camera push-in. The lake surface ripples gently, fog moves between the trees, and small birds cross the distant sky. Preserve the original colors, mountain shapes, and quiet sunrise mood.`

## Sources

- [Qwen Cloud video prompt guide](https://docs.qwencloud.com/developer-guides/accuracy-tuning/video-generation)
- [Wan2.2 GitHub repo](https://github.com/Wan-Video/Wan2.2)
- [Wan2.2 TI2V model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
