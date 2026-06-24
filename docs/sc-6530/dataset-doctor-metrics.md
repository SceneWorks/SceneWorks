# sc-6530 — Dataset Doctor: metrics, thresholds & readiness scoring

**Design spike for epic 6529.** Defines *what we measure*, *how we threshold it*, and
*how per-image flags roll up into a readiness signal* — the decisions that de-risk the
P1–P3 stories before any code ships. No code in this story; the outputs below feed
6531/6532/6533/6534.

> Status: **catalog + thresholds + readiness model proposed and architecture-grounded.**
> The one open item is empirical calibration on real datasets (see §8) — the threshold
> *defaults* below are starting points to be tuned against that data, not final constants.

---

## 1. Framing — the two axes

Every check is one of two kinds, and the distinction drives the default action:

- **Technical quality** — *is this image usable at all?* (resolution, blur, exposure,
  exact/near duplicates). Generic, table-stakes, kind-independent. A standard linter would
  catch these.
- **Training usefulness** — *will this **set** teach the intended thing?* (subject
  prominence, identity consistency, diversity/coverage, caption↔image alignment, count).
  Task-specific; only computable because we know the **kind** (person/style/object), the
  **target resolution**, the **preset**, and the **auto-caption**. This is the differentiator.

Guiding principle from the pitch, carried into every threshold: **bias to warn + explain,
never silently block.** A false positive that blocks the Train button destroys trust faster
than a missed soft image hurts a LoRA.

---

## 2. Check catalog

Per check: axis, the cheap local method, the tier it runs in, and the default action.
`A` = auto-fixable (one-tap, story 6539), `W` = warn + explain, `B` = block-eligible
(only when fatal — see §5).

| # | Check | Axis | Method (Rust) | Tier | Action |
|---|---|---|---|---|---|
| 1 | **Resolution vs target** | quality | stored `min(w,h)` vs training bucket (6531 dims) | 0 | W (B if ≪ bucket) |
| 2 | **Crop-loss under center-crop** | quality | aspect ratio vs the real center-crop-to-square the trainer applies | 0 | W → A (smart-crop) |
| 3 | **Blur / softness** | quality | Laplacian variance (`imageproc`) | 0 | W |
| 4 | **Exposure clipping** | quality | luminance-histogram tail mass at 0 / 255 | 0 | W |
| 5 | **Exact duplicate** | quality | SHA-256 of stored bytes (6531 content hash) | 0 | A (dedupe) |
| 6 | **Near-duplicate (pixel)** | quality | perceptual hash, Hamming distance (`image_hasher`) | 0 | W → A |
| 7 | **Count vs preset** | usefulness | item count vs preset minimum | 0 | W (B if absurdly low) |
| 8 | **Near-duplicate (semantic)** | usefulness | CLIP-cosine clustering, union-find | 1 | W → A |
| 9 | **Diversity / coverage** | usefulness | spread of CLIP embeddings (angles/scenes) | 1 | info / coach |
| 10 | **Caption↔image alignment** | usefulness | CLIP image↔text similarity | 1 | W (outliers only) |
| 11 | **Aesthetic** | quality* | LAION MLP on CLIP ViT-L/14 embedding | 1 | W — **style kind only** |
| 12 | **Identity consistency** | usefulness | ArcFace cosine vs set centroid (`*-gen-face`) | 1 | W (outliers) |
| 13 | **Subject prominence** | usefulness | face-bbox area fraction (`*-gen-face`) | 1 | W |

\* Aesthetic is nominally a quality signal but is **only surfaced for the `style` kind.**
Aesthetic predictors have a documented bias against candids/low-light, and for a
person/object LoRA the *best identity shot is often the ugly one*. Gating identity LoRAs on
aesthetics would actively harm them — so for `person`/`object` we compute it but do not show
or score it.

### Two design decisions inside the catalog

- **Blur uses an absolute floor *and* a relative-to-median band, not relative alone.**
  Story 6532 phrases blur as "Laplacian variance vs dataset median." Pure relative-to-median
  silently passes a *uniformly soft* dataset — the median is soft, so nothing deviates. We
  flag an image if its variance is below a kind-specific **absolute floor** *or* it sits far
  below the dataset median (an outlier within an otherwise sharp set). Absolute catches "all
  soft"; relative catches "this one is soft."
- **Near-duplicate is two mechanisms that MUST be reconciled into one finding.** Tier-0 pHash
  (check 6) catches *pixel*-near-exact pairs — a resave, a resize, a minor re-crop. Tier-1
  CLIP-cosine (check 8) catches *semantic*-near pairs — the same scene a half-second apart,
  different frame. They overlap. The readiness report must **merge a pHash pair and a CLIP
  pair that name the same images into a single "near-duplicate" finding**, or the user sees
  the same two photos flagged twice and stops believing the readout.

---

## 3. Thresholds (proposed defaults, to be calibrated in §8)

Defaults below are deliberately lenient (bias to warn, not block) and **vary by kind** where
the kind changes what "good" means.

| Check | Default | Kind variation |
|---|---|---|
| Min resolution | warn if `min(w,h) < bucket`; "will upscale" if `< 0.75 × bucket` | none (bucket is the target) |
| Crop-loss | warn if center-crop drops > **35%** of the longer side | person/object stricter (subject loss matters more) than style |
| Blur floor | Laplacian var below absolute floor **or** < 0.5 × dataset median | style tolerates softer (texture/bokeh); person/object stricter |
| Exposure | warn if > **5%** of pixels clipped at 0 or 255 | none |
| Near-dup (pHash) | Hamming ≤ **6** of 64 = near-dup; **0** = exact-class | none |
| Near-dup (CLIP) | cosine ≥ **0.95** = near-dup cluster | none |
| Count | warn below preset min (person ~**15**, style ~**20**, object ~**10**); block below a hard floor (~**4**) | per-kind minimums |
| Caption alignment | flag only the bottom-decile CLIP-score outliers | none |
| Aesthetic | informational percentile band | **style only** |
| Identity | warn if ArcFace cosine to centroid < **0.5**, or no face where the kind expects one | person/character only |
| Subject prominence | warn if largest face < **8%** of frame (person) | person/character only |

All thresholds live in **one config surface** (a Rust constants module + per-kind overrides),
never scattered at call sites — calibration in §8 should be a data edit, not a refactor.

> **Pilot evidence (sc-6541, see §8.1).** On the **signal (X) side** the blur **dual floor** (the
> absolute arm is load-bearing — a uniformly-soft set only trips the absolute one, a design fact),
> the non-style **diversity 0.12** floor, and the **near-dup CLIP 0.95** threshold all separated
> clean-vs-degraded *decisively*. On the **output (Y) side**, a replication across a second training
> seed and a convergence (1200-step) run revised the initial pilot: **gross prep defects cause large
> robust harm** (the center-crop confound halved identity), **blur is a weak predictor of identity**
> (a blurry photo retains facial structure; the effect is below run-to-run training variance),
> **low-diversity causes a subtle appearance mode-collapse** (visible at convergence; identity
> unaffected), and **training budget is itself a confound** (run-to-run variance swamps subtle
> dataset effects at low steps). Net: weight gross structural checks heavily; keep the absolute blur
> floor but treat blur as advisory for identity; diversity/near-dup are supported as variety
> predictors. All on one render subject — direction and mechanism, not cross-subject generalization.

---

## 4. Readiness model — sub-scores + a discrete gate, **not** a magic number

A single 0–100 "dataset score" is a trap: it's not actionable ("why 72?") and it invites
false precision. Instead:

**Per-image:** each item carries a list of typed flags `{ check, severity, evidence }`
(`severity ∈ {info, warn, fatal}`, `evidence` = the raw value, e.g. `laplacianVar: 38.2`).
The thumbnail badge is the worst severity on that item: ✓ none → ⚠ warn → ✕ fatal.

**Per-dataset:** a handful of **interpretable sub-scores**, each a plain count/ratio the UI
can name in words — not weights blended into one number:

- `technical`  — share of items with no quality flag (checks 1–6)
- `diversity`  — coverage spread (check 9); "mostly one angle" is a low-diversity signal
- `identity`   — share consistent with the set centroid (person/character; checks 12–13)
- `alignment`  — caption↔image agreement (check 10)

**The gate** is a discrete enum derived from the sub-scores + fatal flags:

- **Ready** — enough usable images, no fatal flags. Train freely.
- **Needs attention** — warnings present (soft/dup/low-diversity). Train is **enabled**;
  the readout explains what would make it stronger. This is the default for most real sets.
- **Blocked** — genuinely untrainable (see §5). Train disabled, with the one reason why.

> 6534's "readiness meter" renders **this discrete gate** (a three-state Ready/Warn/Blocked
> indicator backed by the sub-scores), not a percentage — otherwise the UI and the model
> disagree about what readiness means.

---

## 5. Block vs warn policy

**Default is warn.** Block is reserved for sets that cannot train or cannot teach anything:

- **Too few usable images** — below the hard floor (~4 after exclusions). A LoRA needs a
  minimum to converge at all.
- **Effectively one image** — every image is an exact/near duplicate of one other (e.g. 12
  copies of one selfie). Nothing to learn.
- **Wrong-subject collapse (person/character only)** — for an identity LoRA, if no image
  contains a detectable face, the kind's core promise can't be met. (A *minority* of
  off-subject images is a warn, not a block.)

Everything else — blur, exposure, low resolution, low diversity, aesthetic, caption
mismatch, a few near-dups — is **warn + explain**. Borderline calls resolve to warn.

---

## 6. Architecture decisions this spike ratifies

Grounding the metric design against the real tree (not the epic's pre-audit file references)
surfaced placement decisions that the implementation stories depend on:

1. **`sceneworks-core` is decode-free by design** — it ships *no* native image-codec build
   (`media_convert.rs` shells out to `sips`/`ffmpeg`; it hand-sniffs headers) so it stays
   cheap and correct on the Windows candle lane. The pixel-decode crates the epic names
   (`image` 0.25, `imageproc` 0.25) live in **`crates/sceneworks-worker`**, not core.
   → **Tier-0 splits across the decode boundary:**
   - **In core, header-only (6531):** dimensions + content hash. Read dims without a full
     decode (`imagesize` — pure-Rust, header-only, no codec build, faithful to core's
     existing no-decode ethos); SHA-256 the stored bytes (`sha2`, already locked).
   - **In the worker, full-decode (6532):** blur (Laplacian), exposure histogram, perceptual
     hash. These need real pixels → they belong where `image`/`imageproc` already live.
   - **The rollup/flags/readiness math is pure and lives in core (6533)** — it operates on
     stored scalars, so recompute on every dataset edit is arithmetic, never a re-decode.

2. **Persistence rides the existing JSON manifest, not a new DB.** Datasets are a JSON
   manifest on disk (`dataset.sceneworks.training-dataset.json`); `jobs.db` is only the job
   queue. Per-image scalars + flags attach to `TrainingDatasetItem` (typed fields / the
   `extra` bag). Heavy Tier-1 **embeddings** (CLIP/face) persist as a **sidecar keyed by the
   content hash**, so they survive edits and are recomputed only when bytes change.

3. **The content hash is the P1→P2 seam — make it mandatory in 6531, not "ideally."** It is
   the exact-duplicate key in Tier-0 *and* the cache key that invalidates a Tier-1 embedding
   exactly when the image bytes change. Cheap to add at upload; expensive to retrofit (you'd
   re-read every image). It's the cheapest thing that ties the two phases together.

4. **One CLIP ViT-L/14 vision provider is the keystone of all of P2.** A general CLIP *image*
   encoder does **not** exist in the tree today — current CLIP usage is *text* encoders for
   conditioning; PuLID's EVA-CLIP is a bespoke vision tower in a different embedding space.
   The LAION aesthetic MLP (check 11) is trained on **CLIP ViT-L/14** image embeddings, so
   reusing EVA-CLIP is **not** an option for aesthetic — 6535 must stand up a real ViT-L/14
   vision tower (`candle_transformers::models::clip` vision / OpenCLIP), following the
   JoyCaption/face provider pattern (`gen_core` trait + `inventory`-registered mlx-gen /
   candle-gen crates, force-linked `use … as _;`). That single model then powers near-dup
   (8), diversity (9), caption-alignment (10, paired with the text tower we already have),
   and aesthetic (11). Everything else in P2 is math on its output. *(Verify the exact
   aesthetic weights are ViT-L/14-conditioned before building — almost certainly yes.)*

5. **Face embeddings are already shipped** — `gen_core::FaceEmbedder` →
   `mlx-gen-face` / `candle-gen-face` (512-d ArcFace via `analyze()`), used today only for
   InstantID/keypoints. Identity consistency + subject prominence (12, 13) reuse this
   directly; no new model.

6. **Caption↔image alignment is weak by construction — scope it honestly.** JoyCaption
   captions are long, descriptive paragraphs; CLIP's text tower truncates at 77 tokens. A
   naive full-caption CLIP-score is unreliable. Ship a **coarse outlier flag** (or
   max-over-sentence-chunks), surfaced only for the worst outliers — **not** a per-image
   precision metric. Promising more than the signal supports would manufacture false
   positives in exactly the place trust is thinnest.

---

## 7. Story mapping

| Story | Consumes from this spike |
|---|---|
| 6531 dims at upload | §6.1 (core, header-only `imagesize` + `sha2`), §6.3 (content-hash seam) |
| 6532 Tier-0 checks | §2 checks 1–7, §3 thresholds, §6.1 (worker decode for blur/exposure/pHash) |
| 6533 readiness report | §4 sub-scores + gate, §2 near-dup reconciliation, §6.2 persistence |
| 6534 Teach/Editor UI | §4 (discrete meter + badges), §5 (warn-not-block copy) |
| 6535 analysis job | §6.4 (ViT-L/14 keystone), §6.2 (embedding sidecar) |
| 6536–6537 | checks 8–11, §3 |
| 6538 identity | checks 12–13, §6.5 (reuse `*-gen-face`) |
| 6540 kind-aware recs | §3 per-kind thresholds, §5 per-kind block floors |

---

## 8. Open item — empirical calibration (the remaining spike work)

The thresholds in §3 are reasoned defaults; the story's acceptance asks for validation on
**3–5 real datasets**: a clean one, a blurry one, a near-dup-heavy one, a wrong-person one,
and ideally a low-diversity one. The calibration loop, once the Tier-0 path (6532) computes
the raw scalars:

1. Run Tier-0 over each labelled set; dump raw distributions (Laplacian var, clip ratios,
   pHash Hamming histogram, resolution vs bucket).
2. Confirm each known-bad set trips the intended check **and** the clean set stays Ready
   (no false blocks).
3. Tune the §3 constants to maximize that separation, then freeze them in the config surface
   from §3.

This step needs the raw numbers from real images, so it lands alongside 6532 rather than
ahead of it — but the catalog, readiness model, and architecture above are the parts that
had to be settled first, and they are.

### 8.1 Closed-loop validation (sc-6541) — signals → trained-LoRA quality

The deepest validation of §8 isn't "does the bad set trip the check" but "does tripping the check
predict a worse **trained LoRA**." sc-6541 ran that closed loop on-device (native MLX:
Z-Image-Turbo LoRA train → generate → ArcFace/CLIP score). Full write-up:
[`docs/sc-6541/results.md`](../sc-6541/results.md). Scope: reduced-step LoRAs, one stylized-render
subject. The initial single-draw pilot (N = 16) was then stress-tested with a **second training
seed** and a **convergence (1200-step) run**, which revised its output-side conclusions — **direction
and mechanism, not regression-learned thresholds, and not cross-subject generalization.**

What it established (each degradation derived from one clean 21-image set, count held fixed).
**The signal (X) side is solid; the output (Y) side was revised by replication — keep them apart:**

**Solid (decisive on its own terms):**
- **X-side isolation.** Each degradation moved *only* its own signal: blur collapsed
  `blur_variance` ~75× (median 203 → 2.7) with diversity/near-dup untouched; low-diversity dropped
  `clip_diversity` 44% (0.172 → 0.097) and raised near-dup pairs 2 → 45 with `blur_variance`
  untouched. The check catalog cleanly separates known-bad from clean.
- **Checks 2/13 (crop-loss / subject-prominence) — confirmed harmful (the strongest finding).**
  The trainer's blind `center_crop_square` dropped faces on tall full-body inputs; face-centering
  the crops **doubled** learned identity (0.156 → 0.285). A clean, mechanism-level confirmation
  that these flags track real training harm, and an argument for face-aware cropping /
  aspect-bucketing in the trainer.
- **The absolute blur floor is load-bearing (confirms §2), independent of N.** The blurred set is
  *uniformly* soft (every image ≈ 2.7 vs clean median 203), so a relative-to-median rule passes it
  (nothing deviates) — only the **absolute** floor flags it. This is a design fact, not a
  statistical claim. The absolute *value* is resolution/content/subject-dependent → still needs
  multi-subject calibration before freezing a constant.

**Revised by replication (second training seed) + convergence (1200 steps) — see results.md §4:**
- **The N = 16 sign-test p-values were pseudoreplicated** (one LoRA per condition → paired outputs
  subsample a single checkpoint). A second training seed exposed it: the *control's* own identity
  swung 0.282 → 0.176 (Δ 0.106) — larger than the degradation effects — so the earlier p ≈ 0.06–0.08
  figures are not causal evidence.
- **Blur → identity: weak.** Direction held (blur was lowest-identity in both training seeds) but
  the clean–blur gap collapsed +0.093 → +0.019; at convergence (1200 steps) clean [0.331, 0.410]
  and blur [0.284, 0.315] separate by only 0.016, under the ≈0.08 between-seed swing. A blurry photo
  retains facial structure (ArcFace is blur-robust). Blur is a **weak predictor of identity** — its
  value is output sharpness + the absolute floor, not identity.
- **Low-diversity → mode-collapse: real but subtle, visible only at convergence.** The 400-step
  `same_prompt_spread` claim did not reproduce across seeds; at 1200 steps it dropped consistently
  (0.15 → 0.12) and visual inspection shows the neardup LoRA locking onto the memorized appearance of
  its 4 base images (hair/beard/wardrobe) while clean varies — an appearance-manifold collapse that
  CLIP-whole-image and ArcFace-identity structurally miss. Identity and adherence unaffected.
- **Training budget is a confound on the measurement.** Run-to-run variance swamps subtle dataset
  effects at reduced steps and only partly closes at 1200; subtle thresholds cannot be calibrated
  from low-step single-draw LoRAs.

Net: the X-side separation and the crop-confound finding are solid. On the output side, **gross prep
defects (crop) cause large robust harm, subtle pixel degradation (blur) is a weak identity
predictor, and low-diversity causes a subtle appearance mode-collapse** — all on one render subject,
establishing direction and mechanism, not learned thresholds. The §3 thresholds stand as a sensible
prioritization (weight gross structural checks; keep the absolute blur floor; diversity/near-dup
supported as variety predictors); fixing absolute constants needs additional real-photo subjects and
converged, multi-seed training. Remaining strengthening: the deferred wrong-person + subject-
prominence variants (untested), and replication on real-photo subjects.
