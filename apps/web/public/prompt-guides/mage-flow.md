# Mage-Flow Prompt Guide

Mage-Flow responds best to direct, concrete descriptions of the intended image. Put the main
subject first, then describe the setting, composition, lighting, materials, and visual style.

## Base and RL

Use Base when you want the undistilled foundation behavior. Use RL for the reward-optimized
checkpoint. Both use true classifier-free guidance; the catalog defaults are a useful starting
point, and lower guidance generally produces a looser interpretation while higher guidance follows
the text more strongly.

## Turbo

Turbo is distilled for four-step generation and does not benefit from long denoising schedules.
Keep the prompt explicit because there are fewer refinement steps.

## Example

> Editorial photograph of a weathered red fishing boat tied to a quiet stone harbor at dawn,
> low mist over the water, soft blue ambient light with warm cabin lamps, 50 mm lens,
> eye-level composition, natural texture and restrained color grading.
