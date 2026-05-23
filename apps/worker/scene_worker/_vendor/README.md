# Vendored third-party packages

These are not pip-installable (no `pyproject.toml`/`setup.py`), so they are
vendored here and placed on `sys.path` by the adapter that needs them. This
directory is bundled into both the Docker worker image (via `PYTHONPATH`) and
the desktop Python sidecar (via `stage-python.mjs`, which copies `scene_worker`
wholesale).

## lens

Minimal inference package for Microsoft's **Lens / Lens-Turbo** text-to-image
model. Importing `lens` registers the custom `LensPipeline`,
`LensTransformer2DModel`, and `LensGptOssEncoder` classes into the `diffusers`
and `transformers` namespaces that the model's `model_index.json` references —
there is no published pip package and no `trust_remote_code` path.

- Source: https://github.com/microsoft/Lens
- Commit: `5bf0f0cea2f4bc32ebb2b7ed2ef96d5e88b701e0` (2026-05-22)
- License: MIT (see `lens/LICENSE`)
- Consumed by: `scene_worker/image_adapters.py::LensTurboAdapter`
- Requires (pinned in `requirements.txt`): `diffusers==0.38.0`,
  `transformers>=5.8,<6`, `torch>=2.11`, `einops`, and `kernels` (for the
  mxfp4 GPT-OSS text-encoder path; without Triton the encoder dequantizes to
  bf16).

To update: re-copy `lens/` from the upstream repo at the desired commit and
update the commit hash above.
