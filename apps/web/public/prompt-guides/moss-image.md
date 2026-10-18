# MOSS-Image

Base and SFT generate images from text. Use multilingual prose, tags, or your own JSON exactly as authored. Layout JSON is text interpreted by the model, including `summary.caption`, `items`, `yxyx`/`abs_centralized` and a matching canvas. It does not supply spatial tensors or reference images.

Ordinary defaults are 50 requested steps, CFG 3, schedule shift 5, training shift 1 and context 4096. The author schedule rounds onto its training grid, so actual denoise steps can be fewer. Unconditional skipped layer 22 is the default; `[]` disables skipping. Optional skip-layer guidance has layers 0–41, a nonnegative scale and an ordered interval within 0–1.

SFT alone offers **8-step acceleration**, initially OFF. Enable it deliberately and download its selected package. BF16 + FP32 residuals uses the unmerged author adapter and replacement time weights. Full FP32 uses the already distilled published transformer, without applying a second adapter. Steps 1–8 choose trained prefixes; this is fixed-noise sampling rather than Euler. CFG, negative prompts, skip layers, SLG and schedule controls are disabled while accelerated. Switching OFF restores ordinary controls.

Neither model supports reference images, edits, masks, poses, character conditioning or general style LoRAs. Install and memory errors identify the selected components; the worker never silently substitutes ordinary SFT for an acceleration request.
