# Epic: Dataset Doctor — Pre-Training Image Quality And Usefulness Evaluation

Shortcut epic: `6529`

## Goal

Before a LoRA training run starts, evaluate the training images for **technical quality** and **training usefulness**, and give the user clear, plain-language guidance (and one-tap fixes) on how to improve the set. The quality of the training images drives LoRA quality more than any hyperparameter — yet SceneWorks currently does nothing to assess them.

The feature should feel like a helpful "dataset doctor" sitting in front of the Train button, not a linter. It should be approachable enough for Simple-mode "Teach" (non-technical creators) and detailed enough for Advanced users who want raw numbers and overrides.

## Problem Statement

SceneWorks trains on whatever images the user provides. The only gates an image passes today are:

- Non-empty file + `image/*` MIME type at upload — `crates/sceneworks-core/src/project_store.rs:416-437`.
- File still exists on the worker at train time — `apps/worker/scene_worker/training_adapters.py:106-116`.
- Structural uniqueness (unique IDs/paths) — `crates/sceneworks-core/src/training_store.rs:496-540`.

The one surface labeled "health" (`apps/web/src/training/datasetHelpers.js:92-112`, rendered in `apps/web/src/screens/training/DatasetEditorPanel.jsx`) only counts items, missing captions, and **duplicate _filenames_ by string** — never pixels. We don't even measure image dimensions: width/height are written as `Value::Null` at upload (`project_store.rs:476-477`).

Between upload and training, each image is only RGB-converted, **center-cropped to a square**, and resized (`apps/worker/scene_worker/training_adapters.py:984-997`; a comment there notes aspect-ratio bucketing is future work). So a wide shot silently loses half its subject, and a small image is upscaled to mush — with zero feedback to the user.

This hurts our most important audience the most: Simple-mode "Teach" targets non-technical creators who are the least able to eyeball a dataset and the most likely to feed in near-identical selfies.

## Why This Is A Gap Worth Filling

No consumer LoRA trainer (kohya_ss, OneTrainer, etc.) does pre-training dataset evaluation — they assume curated input and only enforce mechanical resolution/bucketing. The capabilities exist only as disconnected, general-purpose tools:

- **CleanVision / fastdup** — generic image-dataset issue finders (blur, dark/light, near-duplicate, odd size/aspect).
- **LAION aesthetic predictor V2** — CLIP+MLP aesthetic score (standard SD data filter).
- **CLIP-score** — image↔caption alignment for catching mislabeled data.

Nobody has packaged these into a guided, plain-language, pre-training "is this dataset good for *my* LoRA?" check. SceneWorks has an unfair advantage: at training time we already know the **kind** (person/style/object, from Teach), the **base model**, the **preset/target resolution**, and we **auto-caption** with JoyCaption — so we can judge *usefulness*, not just generic quality.

## The Two Axes (Core Framing)

1. **Technical quality** — is each image usable on its own? (blur, resolution, exposure, duplicates) → generic, table-stakes.
2. **Training usefulness** — will this *set* teach the intended thing? (identity consistency, subject prominence, diversity/coverage, caption↔image alignment) → task-specific, under-built, **the differentiator**.

## Non-Goals

- No automatic deletion of the user's images without consent (fixes are offered, never forced).
- No hard-blocking the Train button except for genuinely untrainable sets (zero valid images, below preset minimum). Default to *warn + explain*.
- No cloud/hosted evaluation service — everything runs on-device on the native MLX build.
- **No Python runtime** — implemented entirely in Rust on the MLX/candle path (the legacy Python diffusers worker is not a target).
- No aesthetic gating for person/object LoRAs (documented bias; low-aesthetic candids are often the best identity shots).
- No claim that passing the check guarantees a good LoRA — usefulness is ultimately empirical (see Phase 4).
- No replacement of the Advanced manual training controls.

## User Outcomes

- Each training image shows a badge — good / needs attention / likely to hurt — with one plain-language reason.
- Before training, a friendly "Dataset Doctor" readout summarizes the set ("18 photos; 2 look blurry, 3 are near-duplicates, mostly the same angle — a few from other angles will help") plus a readiness meter.
- One-tap fixes are offered for fixable issues (drop duplicates, smart-crop to subject, upscale low-res, strip EXIF).
- Recommendations adapt to what the user is teaching (person vs style vs object).
- Advanced users see raw metric values, distributions, thresholds, and per-image overrides.

## Architecture

Two tiers, both on-device and **implemented entirely in Rust** (no Python):

- **Tier 0 — instant, per-image.** Cheap pure-Rust checks at upload / optionally in-browser (canvas). No new models. Resolution vs target, crop-loss under center-crop, Laplacian-variance blur, exposure clipping, exact/near-duplicate, count-vs-preset.
- **Tier 1 — async dataset-analysis job.** Modeled on the existing caption job (`apps/rust-api/src/training.rs:121-167` + worker), routed through MLX/candle (the same mechanism that already hosts JoyCaption in mlx-gen). Computes CLIP embeddings → near-dup clustering (union-find over cosine), diversity/coverage, caption alignment, aesthetic (style only). Reuses the existing face path (`crates/sceneworks-worker/src/kps_jobs.rs:306,354`, `image_jobs/instantid.rs:296`, `mlx-gen-face`/`candle-gen-face`, today generation-only) for identity consistency.

Both tiers write into a single **dataset readiness report** (per-item flags + dataset-level rollup + interpretable sub-scores + an overall readiness signal). One-tap fixes reuse assets already on disk (Real-ESRGAN upscale; smart-crop-to-subject replacing the naive center-crop; dedupe; EXIF strip). The Rust core/API stay the source of truth; the worker still trains on concrete prepared images only.

### Runtime And Crates

Every heavy crate is already in the workspace (`Cargo.lock`); the only new dependency is the small, pure-Rust `image_hasher`:

- **Tier 0:** `image` 0.25 (decode, dimensions, EXIF), `imageproc` 0.25 (Laplacian/convolution for blur); luminance histogram computed inline; perceptual hash (dHash/pHash) via the pure-Rust **`image_hasher`** crate — we use the library rather than rolling our own.
- **Tier 1:** CLIP image/text encoders via `candle-transformers` 0.10 (or the existing `mlx-gen` vision encoder that already backs JoyCaption); the LAION aesthetic head is a small MLP in `candle-nn`; embedding math via `ndarray` 0.17; near-duplicate grouping via union-find over cosine similarity (no ML clustering lib needed).
- **Identity:** `mlx-gen-face` / `candle-gen-face` for face embeddings — already Rust, already used for InstantID generation.

The legacy Python diffusers worker (`apps/worker/scene_worker/`) is explicitly **not** part of this feature's runtime.

## UI Behavior

- **Simple (`apps/web/src/screens/simple/Teach.jsx`)** — per-thumbnail badge with one plain-language reason; a "Dataset Doctor" readout + readiness meter before Train; one-tap fix buttons inline on flagged images. Bias toward warn, not block.
- **Advanced (`apps/web/src/screens/training/DatasetEditorPanel.jsx`)** — same report at higher altitude: raw metric values, per-metric distribution, thresholds, and per-image override; subsumes the old filename-only `datasetHealth`.

## Guardrails

- False positives kill trust → default to warn + show the evidence ("sharpness 12 vs median 180").
- Aesthetic score is advisory and style-only.
- Identity-outlier flags are strong-warn, not auto-remove.
- All fixes are non-destructive to the original asset where possible.

## Phasing And Stories

**P1 — Foundation + cheap wins (ships value with no new models)**

- `sc-6530` [Spike] Define metrics, thresholds & readiness scoring
- `sc-6531` Measure & persist real image dimensions at upload
- `sc-6532` Tier-0 instant per-image quality checks
- `sc-6533` Dataset readiness report — data model + API
- `sc-6534` Teach & Dataset Editor — quality badges + plain-language readout

**P2 — Embedding analysis**

- `sc-6535` Dataset analysis job — worker scaffold + MLX CLIP embeddings
- `sc-6536` Near-duplicate clustering & diversity/coverage analysis
- `sc-6537` Caption↔image alignment + aesthetic scoring (style only)

**P3 — Usefulness + remediation (the differentiator)**

- `sc-6538` Identity consistency & subject prominence (person/character)
- `sc-6539` One-tap fixes — dedupe, smart-crop, Real-ESRGAN upscale, strip EXIF
- `sc-6540` Kind-aware recommendations wired into Teach (person/style/object)

**P4 — Research**

- `sc-6541` [Research] Closed-loop: correlate dataset signals with trained-LoRA quality

## Suggested First Demo Slice

`sc-6531 → sc-6532 → sc-6534`: real dimensions → cheap per-image flags → badges + readout in Teach. A working Dataset Doctor with zero new models, suitable for demoing the concept before committing to the embedding/identity work.

## Acceptance Criteria

- Every dataset item has accurate measured dimensions.
- Each image receives typed quality flags with evidence values; Tier-0 checks run in well under a second per image.
- The Train surface renders a complete readiness report (per-item badges + dataset summary + readiness signal) from one structured payload, in both Simple and Advanced.
- Triggering Tier-1 analysis produces and persists CLIP embeddings and attaches dup-cluster, diversity, caption-alignment, and (style-only) aesthetic findings to the report.
- For person/character datasets, an outlier-identity flag fires on a set seeded with one wrong-person image.
- One-tap fixes apply non-destructively and update the report.
- Readiness verdict and recommendations differ appropriately across person/style/object on the same input set.
- Nothing hard-blocks training except a genuinely untrainable set.

## Open Questions

- Where does Tier-0 run by default — worker job, or in-browser for instant Teach feedback (or both)?
- Do we block at all, or warn-only with a confirm-anyway path?
- Should smart-crop-to-subject land here, or fold into a broader aspect-ratio-bucketing effort on the training path?
- Which CLIP/face models do we standardize on for MLX, and what's the on-device cost for a ~30-image set?
- Should the readiness report be persisted with the dataset (for provenance/repro) or recomputed on demand?
- Is aesthetic scoring worth the model weight given it's style-only and advisory?
