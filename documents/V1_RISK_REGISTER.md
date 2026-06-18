# SceneWorks v1 Risk Register (sc-1177)

> **Story:** [sc-1177 — Track top SceneWorks v1 risks and fallback plans](https://app.shortcut.com/trefry/story/1177)
> **Epic:** [1093 — SceneWorks: Research Tracks](https://app.shortcut.com/trefry/epic/1093)
> **Last updated:** 2026-06-18
> **Status:** Living document — synthesizes the empirical + web-verified findings from sc-1173/1174/1175/1176.

**Provenance:** ⚙️ = empirically measured (Apple M5 Max) · 🌐 = web-verified June 2026 · 📄 = SceneWorks code/manifest/prior research.

Each risk: **evidence · severity · owner/follow-up · mitigation · fallback decision point · Rust backend impact.**

---

## R1 — Replace Person quality 📄

**Evidence.** V1 is job/track-first with **Face Only as the safest default**; Full-Person modes are
gated per-model. Mask correction is deferred (box tracks only). On Mac, replace_person is end-to-end
via native Wan-VACE with a SAM2 MLX segmenter (mask states `active/generated/degraded/missing`)
(`documents/REPLACE_PERSON_RESEARCH.md`, `docs/mac-rust-gaps.md:104-129`).

- **Severity:** High (most visible quality surface; identity/temporal artifacts erode trust).
- **Owner / follow-up:** Replace-Person adapter owner — run a quality bar on Full-Person Keep/Replace
  Outfit before exposing them; keep them disabled per-model until they clear it.
- **Mitigation:** Ship **Face Only** as the default supported path (narrowest mask, clearest failure
  mode); procedural preview wires the full flow without a model install; sparse/by-reference masks.
- **Fallback decision point:** if Full-Person quality is below bar at v1 cut → ship Face Only only;
  Full-Person stays model-gated/off.
- **Rust backend impact:** `person_track`/`person_replace` jobs + `PersonTrack*` contracts already
  exist; mode + per-model capability gating lives in the manifest/adapter boundary; sidecar lineage
  (source clip, person track, mode, model, recipe) is the persistence surface. No new contract needed
  for Face Only; Full-Person needs honest per-model capability flags.

## R2 — Video model churn & license exposure 🌐⚙️

**Evidence.** The video-model landscape moves fast and the primary's license is non-standard:
**LTX-2.3 is NOT Apache** — custom "Community License", free commercial only under **$10M ARR** +
anti-compete clause (sc-1174). Wan2.2 is genuinely Apache-2.0. LTX ran natively here ⚙️ but is
memory-heavy (53.4 GB peak).

- **Severity:** High (primary adapter carries a commercial obligation + version volatility).
- **Owner / follow-up:** Video adapter owner — track the LTX ARR threshold as a real commercial
  obligation; keep Wan2.2 wired and exercised so a swap is a config change, not a project.
- **Mitigation:** Adapter-boundary isolation so models are swappable; **Wan2.2 TI2V-5B (Apache,
  ungated, fits 24 GB) kept as a live fallback**; pin model revisions; verify HF license-acceptance
  flow before shipping the downloader.
- **Fallback decision point:** if LTX license/terms become untenable (or ARR crosses the threshold
  without a paid license) → switch primary to Wan2.2 TI2V-5B; A14B for the quality tier.
- **Rust backend impact:** manifest `gated`/`licenseUrl` fields + adapter registry already make models
  hot-swappable; **add per-file `sha256` for supply-chain integrity** (more pointed given the license
  obligation); routing stays capability-based.

## R3 — Apple support ⚙️📄

**Evidence.** Empirically strong: the native Rust+MLX worker builds from source, passes `nax_guard`,
and runs both flagships in-process on M5 Max / macOS 26.5 (sc-1176). But: **memory-bound** (LTX 53.4 GB
of 64 GB at a *minimal* clip ⚙️), a **macOS 26.2 floor** for correct 16-bit kernels, and torch-only
holdouts each with a porting epic (`docs/mac-rust-gaps.md`).

- **Severity:** Medium (works today, but headroom + version constraints are real).
- **Owner / follow-up:** Mac runtime owner — burn down torch-only holdouts (epics 3069/3090/3401/3039,
  sc-3491); keep the self-hosted 26.2+ NAX CI runner green.
- **Mitigation:** quantized tiers (Q4/Q8); the `mac_rust_supported` oracle + warn-only
  `SCENEWORKS_MLX_REQUIRED` rollout; UI-gate features a given Mac/model can't run.
- **Fallback decision point:** if a model can't fit/port on Mac → UI-gate it on Mac (as AuraSR was)
  and keep it on Windows/Linux; never silently degrade.
- **Rust backend impact:** the CUDA-free seams already exist (engine-id table, `cfg(target_os)` gate,
  `ModelMacSupport` surface). **Add a typed precision-aware peak-memory field** — the measured LTX
  peak was ~1.7× the manifest `minMemoryGb`, so admission must key on real peak, not the DiT-only
  estimate.

## R4 — Timeline editor complexity ⚙️📄

**Evidence.** V1 is a SceneWorks-owned timeline model + FFmpeg export; all export primitives validated
on ffmpeg 8.1.2 (sc-1175 ⚙️). Risk is **scope creep** (multi-track compositing, audio mixing UI,
persistent undo) and editor-SDK temptation.

- **Severity:** Medium (schedule risk more than technical risk).
- **Owner / follow-up:** Timeline/editor owner — hold the v1 line at single main-track + fade/crossfade.
- **Mitigation:** keep FFmpeg as the authoritative renderer (browser is preview-only); ship the
  defined MVP; defer compositing/undo/generation-aware hooks explicitly (sc-1175).
- **Fallback decision point:** if the in-app editor slips → ship export of a single-track timeline with
  trims + fades only; richer editing post-v1.
- **Rust backend impact:** fully expressible on existing `Timeline*` contracts + `timeline_export`
  job; **no new versioned contract change for v1** (sc-1175). New transition types / audio mix would be
  additive (the `extra` flatten absorbs experiments).

## R5 — Model storage size ⚙️🌐

**Evidence.** Models are huge. Measured on this machine ⚙️: LTX-2.3-mlx **58 GB**, Z-Image-Turbo
**31 GB**, total HF cache **141 GB**. Web 🌐: FLUX.2-dev ~64 GB DiT + ~48 GB encoder, HunyuanImage-3.0
~169 GB. A handful of models can fill a consumer SSD.

- **Severity:** High (download time, disk pressure, and first-run UX).
- **Owner / follow-up:** Model-manager owner — keep `recommended`/`autoDownload` curated tight;
  refresh `estimatedSizeBytes` from the live HF tree; default to quantized tiers.
- **Mitigation:** on-demand download (missing-model job blocks only that path, doesn't consume GPU
  slots); quantized variants (Q4/Q8) as defaults; per-platform `downloads` so Mac pulls MLX bundles
  and Win/Linux pull torch checkpoints, not both; show sizes before download.
- **Fallback decision point:** if disk/bandwidth is a barrier for the target user → ship a minimal
  recommended set (Z-Image-Turbo + one video model) and make everything else opt-in.
- **Rust backend impact:** manifest `downloads[]` (per-platform `files`/`estimatedSizeBytes`) +
  `ModelInstallMarker` already model this; **per-file size/hash** would make pre-flight disk checks
  and integrity verification exact; download jobs already isolated from GPU scheduling.

## R6 — LoRA compatibility metadata 📄🌐

**Evidence.** LoRA ecosystems vary sharply by base: Qwen-Image LoRA is mature, Z-Image-Turbo's is
young, Wan2.2 applies LoRA as a **high/low-noise pair**, and the manifest carries a `loraCompatibility`
field. Mismatched LoRA↔base produces silent quality failures or errors.

- **Severity:** Medium (correctness/UX; wrong-base LoRA is a confusing failure).
- **Owner / follow-up:** LoRA/import owner — define and enforce `loraCompatibility` (base family +
  arch) at import and at apply time.
- **Mitigation:** validate LoRA base against the target model's family/arch before apply; encode the
  Wan high/low-noise pairing; CI capability audits (sc-2018) already enforce capability honesty.
- **Fallback decision point:** if compat metadata is unreliable at v1 → restrict LoRA to the
  best-supported base (Qwen family) and gate others behind an "experimental" flag.
- **Rust backend impact:** `loraCompatibility` manifest field + `lora_import` job + LoRA-manifest
  schema already exist; the gap is **enforcement** (an apply-time compatibility check) and reconciling
  the capability-string skew (`image_edit` vs `edit_image`; `first_last_frame`/`extend_clip` not in
  `ModelCapability`) so capability-driven gating is trustworthy.

---

## Cross-cutting follow-ups (surfaced by the spikes, not in the AC six)

- **Capability-flag enum skew** (`image_edit` vs `edit_image`; `video_extend` ≠ `extend_clip`;
  `first_last_frame`/`replace_person` not typed) — manifest strings deserialize to
  `ModelCapability::Unknown`, so any future strict capability match would silently misbehave. Reconcile
  before adding capability-gated routing. (sc-1173/1174)
- **Typed precision-aware VRAM/peak-memory** on the manifest — single biggest backend recommendation
  across sc-1174 and sc-1176; admission should check fit-by-precision, not OOM at runtime.
- **Catalog gap:** add `flux2_klein_4b` (Apache, ungated, 24 GB-friendly) — the catalog ships only the
  gated/NC klein-9B. (sc-1173)

## Sources
sc-1173 (image), sc-1174 (video), sc-1175 (timeline export), sc-1176 (Apple runtime) — this repo's
`documents/*_FEASIBILITY*.md` + `documents/FFMPEG_TIMELINE_EXPORT_MVP.md`; `documents/REPLACE_PERSON_RESEARCH.md`;
`docs/mac-rust-gaps.md`; `config/manifests/builtin.models.jsonc`; `crates/sceneworks-core/src/contracts.rs`.
Empirical storage/memory figures measured on this M5 Max.
