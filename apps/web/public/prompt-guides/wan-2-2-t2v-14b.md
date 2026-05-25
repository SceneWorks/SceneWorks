# Wan2.2 14B Text-to-Video Prompt Guide

## Best For

Higher-fidelity text-to-video clips where the whole scene must be described from scratch.

## Prompt Shape

Use the Wan text-to-video structure:

`entity + scene + motion + aesthetic control + stylization`

Because this variant is text-only, include subject, environment, style, motion, and camera information in the prompt.

## Build The Prompt

### Subject

Describe the main entity with clear visible traits: count, clothing, color, pose, expression, and role in the scene.

### Scene Details

Include location, foreground/background, time of day, weather, props, and light sources.

### Style

Name the target look:

- `cinematic realism`
- `documentary`
- `fantasy adventure`
- `stylized animation`
- `music video lighting`

### Camera Angle And Movement

Give a clear shot plan:

- `wide shot, then slow push-in`
- `low-angle tracking shot`
- `static close-up`
- `camera pans right to reveal the doorway`

### Motion

Write a simple sequence that fits the duration. For this SceneWorks variant, keep clips at 5 seconds or less and avoid asking for several scene changes.

### Atmosphere And Lighting

Describe lighting and mood through visible conditions: mist, rain, sunset, rim light, lantern glow, dust, particles, reflections.

### Negative Prompt

Use short negatives: `static, blurry, subtitles, overexposed, distorted limbs, crowded background`.

## Example Prompts

`Two anthropomorphic cats in soft boxing gear face each other on a small spotlighted stage. The orange cat bounces lightly on its feet while the gray cat raises bright blue gloves. The camera starts in a wide shot and slowly pushes in as the crowd lights shimmer in the background. Warm theatrical spotlight, playful cinematic style, clear readable action.`

`A lone astronaut walks through a jungle at dusk, white suit marked with scratches and green reflections from wet leaves. Fireflies drift around the helmet as the astronaut brushes aside hanging vines. Low-angle tracking shot, slow forward movement, humid atmosphere, muted teal and amber color palette.`

## Sources

- [Qwen Cloud video prompt guide](https://docs.qwencloud.com/developer-guides/accuracy-tuning/video-generation)
- [Wan2.2 GitHub repo](https://github.com/Wan-Video/Wan2.2)
- [Wan2.2 TI2V model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
