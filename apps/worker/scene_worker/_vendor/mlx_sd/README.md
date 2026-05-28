# Vendored: Apple `mlx-examples/stable_diffusion`

Source: https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion
Upstream commit: 796f5b5 (current main, 2026-05-28)
License: MIT (Apple Inc., see UPSTREAM_LICENSE)

## Why vendored
mlx-examples is a runnable-examples repo, not a PyPI package — the
`stable_diffusion/` subfolder is a runnable Python package without a
`setup.py`. Vendoring follows the established `_vendor/lens`,
`_vendor/instantid`, `_vendor/pulid_flux` precedents in this directory.
The vendored copy is imported in-process from `MlxSdxlAdapter`
(image_adapters.py) — no sidecar needed because the deps (mlx,
huggingface_hub, regex, numpy, tqdm, Pillow) are already in the worker
venv when `requirements-mlx.txt` is installed.

sc-1975.

## SceneWorks-specific patches
The upstream `model_io._MODELS` dict hard-codes only
`stabilityai/sdxl-turbo` and `stabilityai/stable-diffusion-2-1-base`.
We add `stabilityai/stable-diffusion-xl-base-1.0` (the SceneWorks SDXL
model) with the same file layout as sdxl-turbo (HF repos share the
diffusers-style `unet/`, `text_encoder/`, `vae/` subdirs). This is the
ONLY upstream patch — kept as a single block at the top of `model_io.py`
labeled `# sc-1975 patch:` so a future upstream sync (re-vendor) can
re-apply it cleanly.

## Re-vendor recipe
```
git clone --depth 1 https://github.com/ml-explore/mlx-examples.git /tmp/mlxe
rm -rf apps/worker/scene_worker/_vendor/mlx_sd/{*.py,__pycache__}
cp /tmp/mlxe/stable_diffusion/stable_diffusion/*.py apps/worker/scene_worker/_vendor/mlx_sd/
# Re-apply the SCENEWORKS_SDXL_BASE_PATCH block in model_io.py.
```
