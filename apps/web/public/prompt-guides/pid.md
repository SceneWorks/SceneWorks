# PiD Decoder (Qwen-Image)

**PiD** (Pixel Diffusion) is an optional, per-generation replacement for a model's
VAE decoder. Instead of decoding the latent at native resolution, PiD denoises
**directly in pixel space** and **decodes + 4× super-resolves in a single 4-step
pass** — so turning it on is effectively a "decode straight to high-res, with
detail synthesis" mode rather than a transparent decoder swap.

It's offered per latent space, so the eligible models depend on which PiD decoder
is downloaded: **Qwen-Image** (Qwen-Image, Qwen-Image-Edit, Krea 2 Turbo),
**FLUX.1** (FLUX.1, Boogu-Image, Chroma, Z-Image), **FLUX.2** (FLUX.2-dev,
FLUX.2-klein, Lens, Ideogram 4), and **SDXL** (SDXL, RealVisXL, RealVisXL
Lightning, Kolors).

## ⚠️ Non-commercial — research / evaluation only

The PiD checkpoint is licensed under the **NVIDIA License** (HuggingFace tag
`NSCLv1`). Under **§3.3**, the model and any derivative works may be used
**non-commercially — for research or evaluation purposes only**.

**This restriction applies to the images you decode with PiD.** PiD output is for
research/evaluation use only and is distinct in that respect from images produced
by the rest of the SceneWorks pipeline. When you enable PiD, SceneWorks marks the
output accordingly.

## What changes when you enable it

- **Resolution jumps to 2K/4K.** PiD decodes to roughly **4× the native pixel
  size** in one pass. Plan for larger gallery thumbnails, downloads, and dimensions.
- **It is slower than the VAE.** The 4-step pixel-diffusion decode at 4K is a
  premium path — expect noticeably longer decode time than the instant VAE decode.
- **More memory.** The PiD backbone (~1.4 B params) plus its Gemma-2-2B caption
  encoder load alongside the base model.

## When to use it

- You want a **high-resolution, detail-rich** result in one step (no separate
  upscaler pass), **and** your use is research or evaluation.

Leave PiD **off** for fast iteration, for native-resolution output, or for any use
that is not strictly research/evaluation.

## Availability

PiD is shown only for eligible models and only once its checkpoint has been
downloaded (Model Manager → "PiD Decoder (Qwen-Image)"). The PiD caption encoder
(`gemma-2-2b-it`) is a gated Google repository, so its download requires a
Hugging Face token that has accepted the Gemma terms.

## Sources

- Project: <https://research.nvidia.com/labs/sil/projects/pid/>
- Model card: <https://huggingface.co/nvidia/PiD>
- Code: <https://github.com/nv-tlabs/pid>
- Paper: arXiv:2605.23902
