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

## sensenova_u1

Inference package for **SenseNova-U1** (OpenSenseNova), a unified multimodal
NEO-unify model (Qwen3-based Mixture-of-Transformers; no separate VAE/encoder).
Importing `sensenova_u1` runs `register()`, which registers the custom
`neo_chat` `model_type` → `NEOChatModel`/`NEOChatConfig` into the `transformers`
Auto* registry, so the checkpoint loads via plain `AutoModel.from_pretrained`
(no `trust_remote_code`, no published pip package). Attention falls back to
torch SDPA when `flash_attn` is absent, so it runs on CUDA and MPS unchanged.

- Source: https://github.com/OpenSenseNova/SenseNova-U1
- Commit: `238d6cf3421d12989ec4a240b173d60c924a760b` (2026-05-23)
- License: Apache-2.0 (see `sensenova_u1/LICENSE`)
- Consumed by: `scene_worker/image_adapters.py::SenseNovaU1Adapter`
- Runs in the MAIN worker venv (no sidecar): its deps (torch 2.8,
  `transformers>=4.57,<4.58`, accelerate, sentencepiece, safetensors) match the
  worker stack. Pip-install is not viable — upstream `pyproject` requires
  Python <3.12 (the worker is 3.12) and pins a cu128 torch.
- Only the T2I path is wired today (`model.t2i_generate`); the `gguf` /
  `--vram_mode` offload paths are unused (the offload path is the only place
  upstream hard-excludes MPS).

To update: re-copy `src/sensenova_u1/` from the upstream repo at the desired
commit and update the commit hash above.
