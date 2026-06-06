# sc-3031 — E2E visual/perf parity validation (Rust mlx-gen worker vs the Python MLX path)

**Epic 3018.** The validation gate before the cutovers (sc-3032 image, sc-3037 video) delete the
Python MLX path. This document is the parity record the story asks for.

## What "parity" means here (and what it can't mean)

The Rust engines and the Python paths are **independent implementations**, so bit-exact pixel match
is neither expected nor the bar:

- **Image:** `mlx-gen` (Rust/mlx-rs) is a port of the frozen Python **mflux** fork. They share MLX
  kernels but differ in op ordering/bindings, so same-seed output is *close, not identical*.
- **Video:** the Python path is the third-party **`mlx-video-with-audio`** package; the Rust path is
  `mlx-gen-wan` / `mlx-gen-ltx` (independent ports). These are *different engines entirely* — a
  cross-engine pixel A/B is not a meaningful parity signal.

Numeric output parity is therefore established **at the engine layer**, where `mlx-gen` already
carries an extensive golden suite dumped from the mflux fork (tolerance ~1e-2 peak / ~1e-5 mean):
per-component + full-e2e for Z-Image (incl. Q4/Q8 + img2img), Qwen (incl. edit), FLUX.1 (incl.
Q4/Q8), FLUX.2 (te/vae/e2e/edit), LTX (14 parity files incl. A/V e2e), and SDXL accel samplers, plus
per-crate `perf.rs` compiled-vs-eager A/B. See `/Users/michael/Repos/mlx-gen` `tools/golden/` +
each crate's `tests/*_real_weights.rs` / `*_parity.rs`.

So sc-3031 validates the three things the engine goldens do **not** cover — the worker integration:

1. **Settings-mapping parity (Phase A, below).** Does the SceneWorks Rust worker drive the engine
   with the *same resolved parameters* the Python adapter used (steps / quant / guidance / negative /
   seed / LoRA kind / conditioning / fit / angle / pose / frame-count)? This is where an integration
   bug would hide, and it needs no HW. **Result: clean** — see the table.
2. **Functional E2E on real weights (Phase B).** The `#[ignore]` real-weights smokes generate
   correct-shaped output through the integrated worker for every runnable model.
3. **Perf ≥ the Python path with NAX on, + streaming/cancel/peak-mem (Phase B).**

---

## Phase A — settings-mapping parity audit ✅

Source of truth: Rust `crates/sceneworks-worker/src/{image_jobs,video_jobs}.rs` +
`crates/sceneworks-core/src/{image_request,video_request}.rs` vs Python
`apps/worker/scene_worker/{image_adapters,video_adapters}.py` (the `Mlx*Adapter`s).

### Image

| Model id | steps | guidance | negative | quant default | seed | parity |
|---|---|---|---|---|---|---|
| flux_schnell | 4 | none (distilled) | n/a | Q8 | `seed+idx` / `sha256(prompt:idx)` | ✅ |
| flux_dev | 28 | 3.5 | passed | Q8 | `seed+idx` / hash | ✅ |
| qwen_image | 20 | 4.0 | passed | Q8 | `seed+idx` / hash | ✅ |
| z_image_turbo | 8 | 1.0 (distilled) | passed | Q8 | shared across pose/set, else `seed+idx`/hash | ✅ |
| sdxl / realvisxl | 30 | 7.0 | passed | **Q8** (Python vendored = dense bf16) | `seed+idx` / hash | ✅ steps/guidance/seed; quant = intentional improvement (see Δ1) |
| flux2_klein_9b / _kv | 4 | 1.0 (mandatory) | none (FLUX.2 has no CFG) | Q8 | shared across set, else `seed+idx`/hash | ✅ |
| flux2_klein_9b_true_v2 | 24 (undistilled) | 1.0 | none | Q8 | shared / `seed+idx`/hash | ✅ |

Quant resolution (all image families, identical precedence): `advanced.mlxQuantize` →
`modelManifestEntry.mlx.quantize` → **Q8 default**; `≤0 → dense bf16`, `≤4 → Q4`, else Q8.

LoRA classification (all): max 3/job; peft-LoRA accepted; peft-LoKr accepted on MLX (engine merges);
third-party LyCORIS rejected (→ torch routing). Matches Python `classify_adapter` + the sc-3021/3027
routing.

### Image — advanced flows

| Flow | Key params | parity |
|---|---|---|
| Z-Image strict pose (sc-3028) | controlScale default **0.9**, clamp `[0,2]`; skeleton stickwidth `max(6, round(min(w,h)·0.012))`; **shared seed** across the set; one image per pose | ✅ (formulas byte-identical) |
| Z-Image identity img2img-init (sc-3146) | engages iff `referenceStrength>0` + `referenceAssetId`; clamp `[0.05,1.0]`, forwarded **verbatim** (mflux `image_strength` convention, no inversion) | ✅ |
| FLUX.2 edit / KV / multi-ref (sc-3029) | engine id per variant (`flux2_klein_9b_edit` / `_kv_edit`); 1 ref → `Reference`, N → `MultiReference`; 4 steps / guidance 1.0 | ✅ |
| Character-Studio angle set (sc-3030) | 11 canonical angles in order; per-angle prompt augment with trailing-punct strip; **shared seed** | ✅ |
| Best-effort pose tier (sc-3030) | per-pose `[skeleton, reference]` MultiReference (skeleton first); pose prompt augment | ✅ |
| fit_image (sc-3030) | crop = cover+center-crop; pad/outpaint = contain+letterbox-on-black; default crop; applied to edit source only | ✅ |

### Video

| Model id | steps | guidance | negative | frame count | fps | seed | parity |
|---|---|---|---|---|---|---|---|
| wan_2_2 (TI2V-5B) | engine config default | engine config (≈5.0) | user, else None → engine default | `max(5, raw-((raw-1)%4))` | request fps | `sha256(prompt)` (no index) | ✅ |
| wan_2_2_t2v_14b | 4 (Lightning distill) | 1.0 | user / None | 4n+1 | request fps | sha256 | ✅ |
| wan_2_2_i2v_14b | engine config | engine config | user / None | 4n+1 | sha256 | ✅ |
| ltx_2_3 / ltx_2_3_eros | distilled (none) | none (engine CFG-baked 1.0) | none | `8k+1`, min 9, tie→lower | request fps | sha256 | ✅ frame/seed/fps; guidance = engine-API diff (Δ2) |

Frame-count formulas (`ltx_frame_count`, `wan_frame_count`) are **byte-identical** Rust↔Python,
including the LTX nearest-`8k+1` tie-break (`lower_delta <= upper_delta → lower`).

Video LoRA: Wan MoE high/low via the `.high_noise`→`.low_noise` sibling convention; T2V-14B Lightning
distill pair at strength 1.0; LTX per-pass residual; LoKr-on-Wan → torch (routing, sc-3036),
LoKr-on-LTX stays MLX. Conditioning: Wan i2v `Reference` (required 14B / optional 5B); LTX
`Reference`. Matches Python.

### Intentional differences (NOT regressions)

- **Δ1 — SDXL default quant.** Python MLX SDXL ran the vendored `_vendor/mlx_sd` path **dense (bf16)**;
  Rust `mlx-gen-sdxl` defaults **Q8** (validated quality-neutral, ~25% peak-mem reduction; sc-3026).
  An enhancement. `advanced.mlxQuantize=0` restores dense.
- **Δ2 — LTX guidance/CFG.** Python `mlx-video-with-audio` exposed a `cfg_scale` knob (default 3.0).
  Rust `mlx-gen-ltx` is the **distilled 2-stage A/V checkpoint** where CFG is baked — the engine forces
  1.0 and rejects guidance. Different engine; `guidanceScale` is now a no-op on the LTX path. By design
  (sc-3035).
- **Δ3 — LTX i2v `imageConditioningStrength`.** Python threaded it (default 1.0). The Rust LTX i2v path
  passes `Reference { strength: None }` (engine default). **Only non-default values diverge**; default
  behavior matches. Minor — candidate follow-up if a user actually tunes it. *(filed as a note, not a
  blocker; see "Open items")*

---

## Phase B — functional E2E + perf (real weights, this M5 Max) ⏳

Runnable smokes (weights cached locally; run serial, `--test-threads=1`):

Image (`crates/sceneworks-worker/src/image_jobs.rs`):
`zimage_real_weights_generates_one_image`, `flux_schnell_…`, `flux_dev_…`, `qwen_…`,
`flux2_klein_9b_…`, `flux2_klein_9b_kv_…`, `flux2_klein_9b_true_v2_…`, `sdxl_…`, `realvisxl_…`,
`zimage_control_real_weights_generates_one_pose`, `flux2_edit_real_weights_generates_one_image`,
`flux2_pose_tier_real_weights_generates_one_image`.

Video (`…/video_jobs.rs`): `wan_5b_real_weights` (needs `SCENEWORKS_MLX_WAN5B_DIR` →
`…/models/mlx/wan_2_2_ti2v_5b`), `ltx_real_weights_with_audio`.

Not runnable E2E here: **Wan-14B** (T2V/I2V, ~133 GB bf16 peak — exceeds a safe worker budget on
128 GB; covered by `mlx-gen-wan` engine goldens + the 5B integration smoke).

**Run (M5 Max, 2026-06-05; debug build, serial `--test-threads=1`).** All 14 runnable smokes PASS —
every MLX-native model generates correct-shaped output end-to-end through the integrated Rust worker
on real weights:

| Model / flow | result | wall (load+gen) | peak RSS |
|---|---|---|---|
| z_image_turbo (txt2img) | ✅ PASS | 6.8 s | 3.5 GB |
| flux_schnell | ✅ PASS | 6.1 s | 2.9 GB |
| flux_dev (28-step guided) | ✅ PASS | 28.7 s | 3.4 GB |
| qwen_image (true-CFG + neg) | ✅ PASS | 37.0 s | 16.2 GB |
| flux2_klein_9b | ✅ PASS | 6.5 s | 5.0 GB |
| flux2_klein_9b_kv | ✅ PASS | 6.0 s | 5.3 GB |
| flux2_klein_9b_true_v2 (24-step) | ✅ PASS | 23.8 s | 5.0 GB |
| sdxl | ✅ PASS | 6.5 s | 2.0 GB |
| realvisxl | ✅ PASS | 6.4 s | 2.1 GB |
| z_image strict-pose ControlNet | ✅ PASS | 9.9 s | 4.5 GB |
| flux2 edit (single reference) | ✅ PASS | 8.4 s | 6.5 GB |
| flux2 best-effort pose tier | ✅ PASS | 10.9 s | 7.2 GB |
| wan_2_2 TI2V-5B (T2V) | ✅ PASS | 4.7 s | 23.9 GB |
| ltx_2_3 (T2V **+ synchronized audio**) | ✅ PASS | 12.7 s | 43.9 GB |

Caveat: these are *functional* smokes at small res (256–512², few steps), debug build — wall-time is
"it works + roughly this fast" (load-dominated for the small models), not a production-res benchmark.
The wall-times track the per-family figures measured during bring-up (sc-3022–3035). Peak RSS is the
test-process max-resident (unified memory on Apple Silicon, so it includes GPU allocations).

Perf bar: per-step time ≥ the Python MLX path with NAX on. The authoritative per-step perf signal is
`mlx-gen`'s per-crate `perf.rs` (compiled-vs-eager A/B on real checkpoints) plus the **`nax-worker`**
CI lane (`tests/nax_guard.rs`, 16-bit SDPA correctness) — which confirms the shipped worker built NAX
kernels at the 26.2 deployment target (no NAX → ~2.5× regression, so a green guard is the floor).

## Phase C — spot output A/B vs the Python path ⏳

Numeric output parity is owned by the `mlx-gen` engine goldens (above). At the worker layer, a spot
A/B on 1–2 representative models (Z-Image image, LTX A/V video) provides belt-and-suspenders that the
integrated worker output is visually equivalent / not-worse than the legacy path. _Pending._

## Open items / follow-ups

- Δ3 (LTX i2v `imageConditioningStrength`) — thread it into the Rust LTX `Reference.strength` if the
  engine honors it for LTX; otherwise document as unsupported. Low priority.
