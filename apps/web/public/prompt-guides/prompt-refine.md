# Prompt Refiner Guide

The Prompt Refiner is an 8B instruction LLM (TheDrummer/Anubis-Mini-8B-v1, a Llama-3.3 community finetune) that powers the "Refine my prompt" control in Image, Video, and Audio Studio. It rewrites your prompt to follow the selected generation model's own prompt guide — it does not generate images, video, or audio itself. The same model also backs Ideogram 4's magic-prompt expansion and reference-image captioning, so installing it once serves all of these.

## What It Does

When you click **Refine my prompt**, your current prompt and the selected model's prompt guide are sent to this model. It returns a single rewritten prompt that preserves your intent (subjects, attributes, actions, setting) while tightening phrasing and adding only details that keep the result coherent and on-guide. You review the rewrite and choose **Apply** or **Keep original** — your prompt is never changed automatically.

## Installation

The refiner runs in-process on the native worker (MLX on macOS, candle on Windows/CUDA) and is **not** auto-downloaded. Install it once from the **Models** screen (it is also offered inline the first time you refine before it is present). It is 16.1 GB — the stock bf16 checkpoint, downloaded and loaded as-is with no quantization — and downloads into the shared Hugging Face cache, so other tools reuse it.

## Practical Notes

The refiner matches the language of your prompt: a non-English prompt is rewritten in the same language.

It works without a model guide, but the rewrite is most useful when the selected generation model ships one — the refiner follows that guide's recommended structure and what-to-avoid guidance.

If a prompt is already detailed and on-guide, the refiner makes only minimal edits for fluency.
