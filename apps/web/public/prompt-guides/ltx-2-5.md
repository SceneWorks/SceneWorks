# LTX-2.5 Gemma 4 Prompt Guide

## Best For

LTX-2.5 produces synchronized video and audio from text or a starting image. Its Gemma 4 encoder is designed for detailed, ordered direction: describe the visible action, camera, setting, and sound as one cinematic sequence.

## Text To Video

Write one continuous, detailed paragraph of roughly 150–220 words. Start directly with an observable action or image detail — not a scene-setting preamble. Preserve every requested element, then expand it with physical, visible detail: environment, surfaces, light, colour, clothing, posture, facial cues, and where each subject is positioned.

Include exactly one clear shot scale, a viewpoint, and camera motion in natural prose. For example: “A medium shot frames the runner from a side view as the camera tracks alongside.” State that the camera remains static if it does not move. Describe events in chronological order using transitions such as “Initially”, “A moment later”, and “Simultaneously”.

Keep the description objective. Show expressions and body language instead of naming an internal emotion. Include dialogue verbatim in its original language, together with delivery, music, and environmental sound. Ask for cinematic lighting, film-grade contrast, crisp texture, and depth only where they fit the requested scene; do not invent new subjects or actions.

## Image To Video

Treat the input image as the exact first frame. Begin by faithfully describing its visible subjects, composition, lighting, clothing, and viewpoint, then continue the requested action from that frame in one continuous take. Do not replace details from the image, introduce a contradictory opening, or add a hard cut. Keep the opening shot scale and viewpoint consistent with the reference image, then describe the motion, camera movement, chronological action, and soundscape as above.

## Example

`A medium shot frames a cyclist in a red rain jacket from a side view as the camera tracks beside her across a wet bridge at blue hour. Initially, thin rain streaks through the streetlights and amber reflections ripple beneath her tires while a low electric bass pulse blends with the tire hiss on pavement. A moment later, she turns her head toward the river, loose dark hair moving beneath her helmet as the camera eases back to reveal the skyline; distant traffic and a ferry horn remain soft behind the rain.`

## Decoder And Duration Choices

Use **Conv VAE** for the fast default decode. Choose **DiffVAE** when detail in faces, textures, or on-screen elements matters more than render time. The optional duration head chooses a length from the prompt only when no manual duration is sent; set a range when you want it to choose within a bounded clip length. Temporal upsampling is off by default and increases frame count after generation.

## Sources

- [Lightricks LTX-2.5 model card](https://huggingface.co/Lightricks/LTX-2.5)
- [Lightricks LTX-2.5 Diffusers model card](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
- [Gemma 4 text-to-video prompt contract](https://github.com/Lightricks/LTX-2/blob/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma4_t2v_system_prompt.txt)
- [Gemma 4 image-to-video prompt contract](https://github.com/Lightricks/LTX-2/blob/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma4_i2v_system_prompt.txt)
