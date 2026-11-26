# Vision Captioner Guide

The Vision Captioner is a vision-language model that turns a **reference image** into a structured Ideogram-style JSON caption — a style description plus a grounded list of the elements it sees (with bounding boxes). That caption feeds image-to-prompt **variations**: you give it a picture, it describes the style and composition, and that description becomes a prompt you can regenerate from. It does not generate images itself.

The default model is an abliterated **Qwen3-VL-8B-Instruct**.

## Installation

The Vision Captioner runs in-process through native MLX on Apple Silicon and native Candle/CUDA on Windows and Linux. It is **not** auto-downloaded — install it once from the **Models** screen. It is about 18 GB and downloads into the shared Hugging Face cache, so other tools reuse the snapshot. A large-memory accelerator is recommended.

## How It Works

Point it at a reference image and it observes what is actually in the frame — it describes the style and the visible elements rather than inventing content. The output is JSON: a `style_description` plus grounded elements with bounding boxes, in the same shape Ideogram's prompt surface uses. The model is steered to emit valid JSON, and malformed output is rejected and retried.

## Practical Notes

Describe Image also offers every installed Qwen3.8-27B tier (q4, q8, or full precision) in its model selector. These are the same installed snapshots used by the batch describer; the selected tier is preserved for the request.

For models using plain-text prompts, **Describe image** produces a meticulous description of pose, clothing and accessories, the room or environment, lighting, camera angle, composition, depth of field, style and era. Its optional **Prompt options** can exclude identifying characteristics, clothing and accessories, color versus black-and-white mode, and medium or artistic style. The dataset batch describer offers the same independently selectable options; its normal default still excludes physical traits.

The goal is **style and composition variations**, not a pixel-perfect reconstruction of the source. Expect the caption to capture the look, mood, palette, and layout — then vary the specifics when you regenerate.

Captioning is a fast first pass. Review the generated caption and tweak it before generating if you want to push the result in a particular direction.
