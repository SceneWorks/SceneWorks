# SceneWorks Worker

Python worker package for Diffusers/PyTorch-backed image and video inference
adapters.

In Docker Compose this worker is inference-only by default:

```text
SCENEWORKS_PYTHON_UTILITY_JOBS=0
```

Rust utility workers claim model downloads, LoRA imports, frame extraction, and
timeline exports by default. Python utility fallbacks are explicit: set
`SCENEWORKS_PYTHON_UTILITY_JOBS=1` for procedural person detection/tracking, and
set `SCENEWORKS_LEGACY_MODEL_LORA_JOBS=1` or `SCENEWORKS_LEGACY_FFMPEG_JOBS=1`
only while rolling back a specific utility job family.
