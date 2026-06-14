# sc-5576 — Wire candle Kolors + SenseNova-U1 into the worker

Epic 3692 (Candle Windows model-family expansion). Worker counterpart of the candle-gen provider
ports **sc-5485 (Kolors)** and **sc-5486 (SenseNova-U1)**, mirroring the Chroma worker wiring
(sc-5484 / [#658]). Makes the two families actually routable + executable on the Windows/CUDA candle
lane.

## Why this wasn't sc-5525

The kolors/sensenova provider stories deferred their worker wiring to a follow-up that memory had
attributed to "sc-5525". That story turned out to be the unrelated `prompt_refine` → candle `TextLlm`
cutover (merged, [#661]). This story (sc-5576) is the real worker wiring.

## Premise: additive, not a cutover

The gen-core cutover already happened. candle-gen main (`e58ff49`) and the worker both pin gen-core
**`db08076`**. So this is a pure candle-side pin bump plus link/route/advertise wiring — no mlx-gen
bump, no gen-core move.

## Changes

### 1. candle-gen pin bump (`crates/sceneworks-worker/Cargo.toml`)
- All `candle-gen-*` deps `77c70be` → `e58ff49` (candle-gen main HEAD). Same `sceneworks-gen-core`
  pin (`db08076`), so `--features backend-candle` still resolves exactly one gen-core rev (no skew),
  and no mlx-gen bump is needed. `e58ff49` adds candle-gen #54 — the SenseNova `_fast` **sm_120**
  CPU distill-LoRA merge fix (reading the I32 `alpha` through an F32 VarBuilder hit a missing
  on-device I32→F32 cast on Blackwell; the LoRA is now merged on CPU and the delta moved to the
  weight's device).
- New optional deps `candle-gen-kolors` + `candle-gen-sensenova` (`features=["cuda"]`), added to the
  `backend-candle` feature list.

### 2. Force-link (`crates/sceneworks-worker/src/image_jobs.rs`)
`use candle_gen_kolors as _;` + `use candle_gen_sensenova as _;` (Windows + `backend-candle`), so the
MSVC release linker keeps each provider's `inventory::submit!` registration.

### 3. Routing (`crates/sceneworks-worker/src/image_jobs/base.rs`)
- `is_candle_engine`: + `kolors`, `sensenova_u1_8b`, `sensenova_u1_8b_fast`.
- `candle_adapter_label`: `kolors` → `candle_kolors`; `sensenova_u1_8b` / `_fast` → `candle_sensenova`.

### 4. Router eligibility (`crates/sceneworks-core/src/jobs_store.rs`)
`CANDLE_ROUTED_MODELS` += `chroma1_hd`/`_base`/`_flash`, `kolors`, `sensenova_u1_8b`/`_fast`. Without
this a candle worker **refuses** the job (`worker_supports_job`) and it never reaches the candle lane.

> **Also fixes a Chroma gap:** sc-5484 / [#658] wired Chroma into the *worker* (`is_candle_engine`)
> but never added it to `CANDLE_ROUTED_MODELS`, so the router never let a candle worker claim a Chroma
> job — the Chroma candle lane was unreachable. Added here alongside kolors/sensenova (same one-list
> change, same default-off gate).

### 5. Capability
No code change: `engines::registry_capabilities` derives `image_generate` from any linked descriptor
whose `backend` is enabled. The `kolors` / `sensenova_u1_8b` / `_fast` `MODEL_TABLE` rows already
exist (MLX families), so linking the candle providers + `backend_candle_enabled` advertises them.

## T2I-only confinement

Both families are multi-shape on MLX (Kolors: edit / IP-reference / pose-control; SenseNova: edit /
VQA / interleave) but the candle providers are **pure T2I**. Confinement is router-enforced:
`image_request_candle_eligible` rejects `edit_image` mode, any source/reference/mask asset, poses,
and (for non-quant/LoRA families) loras/quant — so only base txt2img reaches the candle worker; every
conditioning shape falls back to the Python torch worker. The MLX-only `KolorsControl` /
`SensenovaEdit` dispatch routes never run on Windows.

## Safety

`backend_candle_enabled` is **default-off**, so production routing is unchanged until the lane is
explicitly turned on — the GPU smoke is exactly that opt-in. Providers were already real-weights
GPU-validated (SenseNova base + `_fast` in sc-5486; Kolors conformance is the remaining provider
smoke). Single gen-core rev (`db08076`) across worker + candle-gen.

## Validation

- Default lane: `cargo fmt --check`, `cargo clippy --all-targets -D warnings`, `cargo test`
  (`sceneworks-core` + `sceneworks-worker`) green.
- Candle lane: `cargo check -p sceneworks-worker --features backend-candle` (MSVC 14.44 vcvars +
  `CUDA_COMPUTE_CAP=120`) clean.
- Skew gate: `scripts/check-gen-core-skew.sh` (+ `--features backend-candle`) resolves one
  `sceneworks-gen-core` (`db08076`).
- Remaining: deployed-worker GPU job-routing smoke on the Blackwell box.

[#658]: https://github.com/michaeltrefry/SceneWorks/pull/658
[#661]: https://github.com/michaeltrefry/SceneWorks/pull/661
