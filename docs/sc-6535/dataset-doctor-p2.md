# Dataset Doctor P2 — CLIP analysis (epic 6529, sc-6535/6536/6537)

Decision doc for Phase 2 of Dataset Doctor: the embedding-based "training usefulness" axis.
Sibling of the P1 spike (`docs/sc-6530/dataset-doctor-metrics.md`). Same framing — **bias to warn,
never block** unless a set is genuinely untrainable; P1 already established the gate, sub-scores, and
per-image override. P2 fills the Tier-1 slots P1 reserved.

Grounded against the actual trees (SceneWorks + mlx-gen + candle-gen), not the epic's pre-audit
references. Anchors are `file:line` at time of writing.

---

## 0. TL;DR / what changes

P2 adds an **async `dataset_analysis` job** that computes a **CLIP image embedding per dataset item**
on Apple MLX (no torch/NVIDIA), persists it keyed by content hash, and derives **set-level** findings:
near-duplicate clustering, diversity/coverage, (later) caption↔image alignment and aesthetic. These
fill `ReadinessSubScores.{diversity, alignment, aesthetic}` and add embedding-based flags to the
existing readiness report.

**The encoder already exists** as `mlx-gen-flux`'s `FluxIpImageEncoder` (canonical CLIP ViT-L/14
`visual_projection`), so the provider is mostly *reuse*. The work is the contract, the job, the
persistence, and the pure set-level math.

---

## 1. Scope split — the 6536/6537 cut is an embedding-provenance cut

Not all downstream checks need the same vector. This is the load-bearing decision:

| Check (story) | Needs | Existing encoder fits? |
|---|---|---|
| Near-dup clustering, diversity/coverage, background (6536) | image vectors in *any* internally-consistent space — cosine is meaningful regardless of head | **Yes** — `FluxIpImageEncoder::image_embeds` (768-d). |
| **Aesthetic** (6537) | the *canonical* OpenAI CLIP ViT-L/14 image embedding `visual_projection(CLS)`, L2-normalized (what the LAION MLP was trained on) | **Yes** — verified `image_embeds` is exactly that head (§3). Needs the **LAION aesthetic-MLP weights** (new hosted asset). |
| **Caption↔image alignment** (6537) | matched image **and text** embeddings in CLIP's joint contrastive space (`visual_projection` + `text_projection`) | **Image: yes. Text: NO** — the CLIP-L text path exposes the conditioning-pooled EOS hidden state, not `text_projection(eot)` (§3). Needs a CLIP-L text projection head added. |

**Therefore P2 increments on this boundary:**

- **Increment 1 = sc-6535 + sc-6536.** The image-embedding job + the set-level clustering/diversity
  math. Rides the encoder that already exists. *All downstream math is verifiable here with synthetic
  embeddings* (§6).
- **Increment 2 = sc-6537.** Aesthetic (reuse the image embed + source/host the LAION MLP) and
  alignment (add a matched CLIP-L text projection head). **Embedding source for alignment is an OPEN
  question** — do not commit to "reuse FluxIpImageEncoder" for the text side.

---

## 2. Sub-score & finding model (extends P1)

P1 reserved the slots (`crates/sceneworks-core/src/dataset_quality.rs`, `ReadinessSubScores`):
`technical: f64` (Tier-0) + `diversity / identity / alignment: Option<f64>` ("None until the embedding
job lands"). P2 fills them, with two corrections:

- **`aesthetic` is NOT reserved — P2 adds `aesthetic: Option<f64>`** to `ReadinessSubScores`.
- **`identity` is the FACE stack, not CLIP.** `identity` = face-embedding consistency (does every photo
  look like the same person?), which uses the existing `gen_core::FaceEmbedder`
  (`mlx-gen-face`/`candle-gen-face`) — a *different* encoder. CLIP must **not** fill `identity`. A
  reviewer should not "wire CLIP into identity." Face-identity is its own follow-up, out of P2's CLIP
  scope. (This is also why the epic's "looks like a different person" copy was deferred from sc-6534.)

New embedding-based `QualityCheck` variants (additive to the P1 string-enum; all **advisory**,
`Warn`/`Info`, never `Fatal` — embeddings are a soft signal):

- `NearDuplicateEmbedding` — CLIP-cosine near-dup (catches "20 frames of one burst" that pHash misses).
  Distinct from P1's `NearDuplicate` (pHash); the report can reconcile the two.
- `LowDiversity` — dataset-level: the set clusters too tightly (same pose/angle/background); won't
  generalize. Carries the recommendation hook ("add a few from other angles").
- `BackgroundContamination` — a repeated background the LoRA would bake in (6536).
- `CaptionAlignment` — low CLIP image↔caption score → re-caption action (6537, increment-2).
- `LowAesthetic` — **style datasets only**; advisory, never on person/object (6537, increment-2).

All thresholds are **placeholders pending calibration** (§8), like sc-6530 §8 — invented numbers must
not read as tuned.

---

## 3. The encoder — reuse, verified

`mlx-gen-flux/src/image_encoder.rs` loads `openai/clip-vit-large-patch14` as
`CLIPVisionModelWithProjection` and computes (`image_embeds`, `:55-66`):
`tower → CLS of last hidden state → post_layernorm → visual_projection (Linear 1024→768, no bias)`.
**This is the canonical OpenAI CLIP image embedding** — the IP-adapter `ImageProjModel` (`image_proj`)
is a *separate downstream* head, NOT applied inside `image_embeds`. Verified at `image_encoder.rs:5-12,
52-66`. The transformer body, `VisionConfig::vit_l_14()`, and `preprocess_clip_image` are re-exported
from `mlx-gen-sdxl` (`mlx-gen-sdxl/src/lib.rs:42-44`).

- **Convention**: the embedder returns the **raw** `visual_projection(pooled)` vector; callers
  L2-normalize for cosine and for the LAION MLP — exactly the `FaceEmbedder` "caller normalizes"
  convention (`gen-core/src/face.rs:17-20`).
- **Text gap (alignment)**: `mlx-gen-flux/src/text_encoder.rs:129-151` returns the CLIP-L *pooled EOS
  hidden state* for conditioning, not `text_projection(eot)`. SDXL's projected text encoder (TE2) is
  CLIP-bigG (1280-d), not matched ViT-L/14. So alignment in increment-2 needs a CLIP-L
  `text_projection` head (small: reuse the CLIP text tower + add the projection Linear).

---

## 4. The contract — `ImageEmbedder` in gen-core

A new backend-neutral trait in `gen-core` (zero tensor deps; host types only — `Vec<f32>`), modeled on
`FaceEmbedder`'s **shape** but the **`Captioner` registration** mechanism (correction to an earlier
note: `FaceEmbedder` is deliberately *not* inventory-registered — it's constructed directly; do not
copy that half).

```rust
// gen-core/src/image_embed.rs  (sketch)
pub trait ImageEmbedder: Send + Sync {
    fn descriptor(&self) -> &ImageEmbedderDescriptor;
    /// Raw embedding (caller L2-normalizes). Input: crate::media::Image (RGB8, row-major).
    fn embed(&self, image: &Image) -> Result<Vec<f32>>;
}
pub struct ImageEmbedderDescriptor {
    pub id: &'static str, pub family: &'static str, pub backend: &'static str,
    pub embedding_dim: usize,   // 768 for CLIP ViT-L/14
    pub space: &'static str,    // e.g. "clip-vit-l14" — guards cross-space cosine
    pub mac_only: bool,
}
```

- **Registration** mirrors `CaptionerRegistration` + `inventory::collect!` + `load_image_embedder(id,
  spec)` (`gen-core/src/registry.rs:44-137`). Provider does `inventory::submit! { ImageEmbedderRegistration {…} }`;
  the worker loads by id via `gen_core::load_image_embedder` exactly as `caption_jobs.rs` does
  `load_captioner`.
- **`space` tag** lets persistence and the report reject mixing embeddings from different encoders
  (a future EVA-CLIP/SigLIP swap can't silently corrupt cosine).

### Cross-repo ripple (the real cost)

gen-core lives in the **mlx-gen repo**; the worker consumes it by a pinned git rev with a **skew gate**
(one gen-core rev across mlx-gen + candle-gen + worker, `mlx-gen/CLAUDE.md` "Dependency pins"). Adding
the trait is a lockstep change: **gen-core lands first → mlx-gen + candle-gen re-pin → the worker bumps
the `sceneworks-gen-core` (+ mlx-gen/candle-gen) pins together.** The gen-core trait itself builds + tests
on Linux (the mlx-gen "contract" CI lane), so it is verifiable; the **provider is not** (§5).

---

## 5. The job — `dataset_analysis` worker job

Why a worker job when P1's Tier-0 runs synchronously in the API: Tier-0 needs no model and decodes off
the async runtime via `sceneworks-image-quality`; P2's MLX inference is `!Send` and Metal/macOS-only, so
it **must** be an async worker job — the same reason JoyCaption is a job.

Mirror the caption job end-to-end:
- **Job type**: `JobType::DatasetAnalysis => "dataset_analysis"` (`crates/sceneworks-core/src/contracts.rs:266` neighbor).
- **API**: `create_training_dataset_analysis_job` + `validate_*` + a request DTO, mirroring
  `create_training_dataset_caption_job` (`apps/rust-api/src/training.rs:248-385`) and
  `TrainingCaptionJobRequest` (`apps/rust-api/src/dto.rs:350`). Per-item payload carries
  `itemId/imagePath/contentHash`.
- **Worker**: new cfg-gated module `dataset_analysis_jobs.rs` with `run_dataset_analysis_job(api,
  settings, job)`, dispatched in `crates/sceneworks-worker/src/lib.rs:647` neighbor. Pattern from
  `caption_jobs.rs:87-322`: `spawn_blocking` (MLX is `!Send`) → `gen_core::load_image_embedder(id,
  LoadSpec::Dir(weights))` → per-item `embed(&Image)` → mpsc + throttled `interval` progress
  (Preparing→LoadingModel→Running→Saving→Completed) → derive set-level findings → write back.
  Force-link the provider: `use mlx_gen_clip as _;`. Linux/non-candle build gets a stub (like
  `caption_jobs.rs:335-345`).

### Provider crates (NOT buildable here)
`mlx-gen-clip` (+ `candle-gen-clip`): `impl ImageEmbedder` reusing `ClipVisionEncoder` /
`VisionConfig::vit_l_14()` / `preprocess_clip_image` (from `mlx-gen-sdxl`) and the `visual_projection`
head pattern from `mlx-gen-flux/src/image_encoder.rs`. Cargo.toml = `mlx-gen` + `mlx-rs` + `inventory`.
HF snapshot weights (`openai/clip-vit-large-patch14`), provisioned like the JoyCaption model download.

---

## 6. Persistence — content-hash embedding sidecar (net-new)

`content_hash` (SHA-256) is *already designated* the Tier-1 cache key
(`crates/sceneworks-core/src/training.rs:126-129`). But a 768×f32 vector per item is far too big to
inline in the manifest the way `tier0_scalars` holds 4 scalars — it would bloat the dataset JSON. There
is **no heavy-binary sidecar precedent** in the tree, so this is a design decision:

- **Recommended**: a per-dataset embeddings sidecar under `dataset_root` (e.g.
  `tier1-embeddings.safetensors` or `.npy`), a map `content_hash → Vec<f32>`, with the embedder's
  `space` + `dim` in a header. Reused across re-runs; self-invalidates on content-hash change like
  `CachedTier0Scalars::valid_for`.
- **Write path**: the worker currently writes results back **via API POST** (caption-sidecars), because
  the API process owns the project store under a lock. Faithful options: (a) a new API endpoint that
  accepts embeddings and writes the sidecar (mirrors caption-sidecars), or (b) the worker writes the
  sidecar file directly into `dataset_root` (embeddings are large binary; a direct file write may be
  cleaner than POSTing MBs of base64). **Open decision** — lean (b) for the binary, (a) for the small
  findings/sub-scores summary.

The set-level findings + sub-scores then feed `dataset_readiness_report`
(`apps/rust-api/src/training.rs:118`), which reads the cached Tier-1 data and fills the report — closing
the loop P1 left at `None`.

---

## 7. The verification boundary (weaker than P1 — flagged for the user)

P1 closed its loop in one repo (build core → Docker-verify worker). P2 does **not**:

| Layer | Verifiable here? |
|---|---|
| `gen-core` `ImageEmbedder` trait + registration | ✅ Linux (mlx-gen contract lane / `cargo` on the gen-core crate) |
| `sceneworks-core` Tier-1 types, sub-score, report wiring | ✅ local |
| Pure set-level math (cosine, dup clustering, diversity spread) in `sceneworks-image-quality`/core | ✅ local — **unit-tested with synthetic `Vec<f32>`, exactly like `evaluate_tier0` with synthetic scalars** |
| rust-api `dataset_analysis` handler + DTO + validation | ✅ Docker (Linux) |
| Worker job plumbing (the Linux-stub path, progress, write-back) | ✅ Docker — but the **MLX branch is cfg-excluded** |
| **`load_image_embedder` + real `embed()` + `mlx-gen-clip` provider** | ❌ **macOS + Metal only — cannot be built or tested in this environment** |

**The heart of 6535 (the actual CLIP forward) is spec-and-write-only here.** Everything *downstream* of
the embedding is fully verifiable with synthetic vectors, which is most of 6536's value — so the
verified-increment cadence survives by splitting on the embedding boundary (produce = unverifiable;
consume = verifiable), mirroring Tier-0's pure/decode split.

---

## 8. Calibration TODO (placeholders, not tuned)

All Tier-1 thresholds below are **guesses pending a fixture sweep** (a near-dup-heavy set + a healthy
diverse set, per 6536 acceptance):
- CLIP near-dup cosine threshold (≈0.95?).
- Diversity/coverage: how to summarize embedding spread (mean pairwise cosine? cluster count vs item
  count?) and the "too tight" floor.
- Background contamination: detection method (patch-region embeddings? a separate signal).
- Caption alignment floor (CLIPScore is unnormalized; needs a per-kind floor).
- Aesthetic: LAION MLP scale, the style-only advisory band.

---

## 9. Increment plan (proposed)

1. **Increment 1a (verifiable now)** — `sceneworks-core` Tier-1 types (`aesthetic` sub-score, new
   `QualityCheck` variants, embedding-cache types) + pure clustering/diversity math in
   `sceneworks-image-quality` with synthetic-embedding unit tests + report wiring. No provider needed.
2. **Increment 1b (write now, verify after re-pin)** — gen-core `ImageEmbedder` trait + registration
   (Linux-verifiable) → `mlx-gen-clip` + `candle-gen-clip` providers (macOS/CUDA only) → worker
   `dataset_analysis` job + rust-api handler + persistence (Docker-verify the Linux plumbing).
3. **Increment 2 (sc-6537)** — aesthetic (reuse image embed + LAION MLP asset) + alignment (CLIP-L
   text projection head). Gated on the §1/§3 text-embedding decision.

Recommend starting with **Increment 1a** — it's fully verifiable here and delivers most of 6536's logic
before any model exists.
