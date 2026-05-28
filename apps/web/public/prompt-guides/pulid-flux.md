# PuLID-FLUX Prompt Guide

## Best For

Generating **the same person** across many different scenes, poses, and outfits from a **single reference image** — on **FLUX.1-dev**, no LoRA training required. PuLID extracts a face embedding from your character's approved reference and injects it into FLUX's DiT through a small cross-attention adapter while your prompt drives everything else.

Use it when you want FLUX-grade rendering quality with faithful likeness — the InstantID story on SDXL (RealVisXL) is the equivalent on the SDXL side. The sc-2012 spike measured **0.8016 ArcFace cosine** vs the reference at the default settings (above the InstantID-SDXL no-restore baseline of ~0.68; PuLID-FLUX does not need a face-restoration pass to clear the bar).

This is a **reference-driven** model: it only runs in the "With character" flow and always needs a clear reference face. There is no plain text-to-image or edit mode.

## How It Works

- **Identity comes from the reference image, not the prompt.** Do *not* describe the person's face, hair color, or features — the model takes those from the reference. Describing appearance only fights the reference.
- **The prompt drives the scene:** setting, action, pose, framing, wardrobe, lighting, and style.
- **Reference strength** (`idWeight`) controls how hard identity is pinned. 1.0 is the recommended photoreal baseline; lower values (0.6–0.8) loosen identity for more prompt-driven editability but the face drifts.
- **Identity start step** (`timestepToStartCfg`) controls *when* identity is injected during denoise. Lower = earlier = stronger identity but less editable. Higher = later = more editable but weaker identity. The upstream guidance: **4 for photoreal**, **0–1 for stylized**.

## Choose A Good Reference

Identity quality is set by the reference more than the prompt:

- A **clear, front-facing** photo where the face is large and well-lit.
- **One** unobstructed face (no sunglasses, heavy shadow, or extreme profile).
- Sharp focus, neutral expression works best as a baseline.

A side profile, tiny face, or low-light crop will weaken the likeness no matter how good the prompt is.

## Build The Prompt

Front-load the scene and action, then layer style and lighting. Leave the face to the reference.

### Scene & Action

Lead with **where** and **what is happening**. PuLID-FLUX renders identity onto whatever scene you describe, so a vivid scene anchors the composition.

> *On a fog-soaked pier at dawn, leaning against a weathered railing, watching boats come in.*

> *Cross-legged on a sunlit hardwood floor next to an open laptop, mid-laugh on a video call.*

### Wardrobe & Setting Details

Describe the **outfit, props, and environment**. These are the levers FLUX renders best — small specifics ("a chunky cream cable-knit sweater", "a black leather jacket over a faded band tee") read clearly. Wardrobe coming from the prompt also keeps it from being pulled from the reference image.

### Style & Lighting

FLUX-dev is photoreal by default; you can dial in cinematic looks with descriptors like *golden hour, soft natural light, shallow depth of field, 35mm film, overcast, neon-lit*. Avoid stacking too many — three or four cues land better than a long list.

## Knobs

- **Reference strength (idWeight)** — Default **1.0**. Higher (up to 1.5) tightens identity; lower (0.6–0.8) loosens for more prompt freedom but the face drifts.
- **Identity start step (timestepToStartCfg)** — Default **4** (photoreal). Drop to 0–1 for stylized/artistic scenes where you want PuLID to influence early structure too; raise above 4 if identity feels too tight and you want more editability.
- **Steps** — Default **30**. The PuLID spike validated 30; lower steps may underdevelop the FLUX render.
- **Guidance scale** — Default **4.0**. FLUX-dev's distilled guidance — between 3.0 and 5.0 reads cleanly.

## Limits

- **Person characters only.** PuLID extracts an ArcFace face embedding; non-person characters (animals, mechs, stylized cartoons) won't have a detectable face and the generation will fail. Use **FLUX.1 [dev]** with IP-Adapter for non-face references, or **InstantID (RealVisXL)** for stylized people on SDXL.
- **1024×1024 only.** The sc-2012 spike validated 1024² on Apple Silicon at ~85 GB peak unified memory and ~127s/image. Other FLUX buckets are deferred to a follow-up.
- **64 GB+ Mac required.** PuLID-FLUX needs FLUX-dev + T5-XXL + EVA-CLIP + PuLID weights resident at once. A 128 GB Mac is comfortable; 36 GB is not feasible at 1024² without offload.
- **License: FLUX.1 [dev] non-commercial (gated).** Inherits the FLUX.1-dev license — accept the gate on Hugging Face and use a token before downloading.

## Sources

- [PuLID-FLUX project (ToTheBeginning/PuLID)](https://github.com/ToTheBeginning/PuLID)
- [PuLID-FLUX adapter weights (guozinan/PuLID)](https://huggingface.co/guozinan/PuLID)
- [FLUX.1-dev model card](https://huggingface.co/black-forest-labs/FLUX.1-dev)
