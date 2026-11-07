//! Swappable diffusion samplers — the engine-agnostic seam behind the few-step acceleration
//! variants (LCM / SDXL-Lightning / Hyper-SD), sc-2769.
//!
//! As of sc-3722 the **policy** (schedules + per-step affine coefficients) lives in the
//! backend-neutral [`gen_core::sampling`] crate; this module keeps only the thin **tensor
//! application**. Each sampler type below is a wrapper holding a [`gen_core::sampling::SamplerPolicy`]
//! plus the MLX compute dtype, so the family-crate call sites are unchanged (D5). The neutral
//! coefficients (`a_x`/`a_out`/`a_noise`/`c_in`) are applied by one shared `apply_step`; a candle
//! backend implements the same ~5 lines against the same policies.
//!
//! A [`DiffusionSampler`] owns a model's **denoise schedule**: the per-step conditioning timestep,
//! the model-input scaling, the initial-noise scaling, and the per-step update. The generic denoise
//! loop drives `&dyn DiffusionSampler` so a model can swap samplers per request without the loop
//! knowing which one is running. Each model family supplies its own impls:
//! - SDXL's production default is the crate-local ancestral Euler sampler (`mlx-gen-sdxl`), which
//!   folds the input scaling into its step → [`DiffusionSampler::scale_model_input`] is identity.
//! - The acceleration samplers here are faithful ports of the **diffusers** schedulers each method
//!   is trained against (`LCMScheduler`, `EulerDiscreteScheduler(timestep_spacing="trailing")`,
//!   `TCDScheduler`); their schedule math (the DDPM `alphas_cumprod` world) is the policy layer.
//!
//! FLUX-MLX and Qwen-MLX acceleration both drive the shared [`FlowMatchSampler`] (the rectified-flow
//! world, sc-2908 / sc-2909); the Qwen-specific Lightning sigma schedule is built in
//! `mlx-gen-qwen-image` and wrapped in this same sampler (deduped in sc-2950).

use mlx_rs::ops::{add, divide, gt, maximum, minimum, multiply, sqrt, subtract, sum_axes, which};
use mlx_rs::{random, Array, Dtype};

use gen_core::guidance::GuidanceOps;
use gen_core::sampling::{
    LatentOps, LcmPolicy, LightningPolicy, SamplerPolicy, StepCoeffs, StepDtype, TcdPolicy,
    TimestepConvention,
};

use crate::array::scalar;
use crate::{CancelFlag, Progress, Result};

/// The DDPM `alphas_cumprod` noise schedule, re-exported from gen-core at the historical
/// `mlx_gen::sampler::AlphaSchedule` path (SDXL/Kolors build it for the acceleration samplers and
/// training).
pub use gen_core::sampling::{AlphaSchedule, FlowMatchPolicy};

/// A swappable denoise schedule. The generic loop calls, per step `i`:
/// `x_in = scale_model_input(latents, i)` → `eps = model(x_in, timestep(i))` → (CFG) →
/// `latents = step(eps, latents, i)`. The starting latents are `scale_initial_noise(unit_noise)`.
pub trait DiffusionSampler {
    /// Number of denoise iterations (loop count).
    fn num_steps(&self) -> usize;

    /// The conditioning timestep fed to the model at step `i` (the value the U-Net embeds).
    fn timestep(&self, i: usize) -> f32;

    /// Scale the latents into the model's expected input space at step `i`. The default is identity
    /// (samplers that fold the scaling into [`Self::step`], e.g. the ancestral Euler sampler, and
    /// the flow-match sampler whose `c_in = 1`); diffusers' Euler divides by `√(σ²+1)`.
    fn scale_model_input(&self, x: &Array, _i: usize) -> Result<Array> {
        Ok(x.clone())
    }

    /// Scale unit-normal noise into the sampler's starting latent space (the txt2img prior).
    fn scale_initial_noise(&self, noise: &Array) -> Result<Array>;

    /// One denoise step: latents at step `i` → latents at step `i+1`, given the (already
    /// CFG-combined) model output. `x` is the **un-scaled** latents (NOT the
    /// [`Self::scale_model_input`] output), matching diffusers' `step(model_output, t, sample)`.
    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array>;
}

// =================================================================================================
// Shared tensor application — the only numeric code that stays per-backend (the policy is neutral).
// =================================================================================================

/// Seed-derived per-step noise source (D6). Stochastic samplers (LCM re-noise, TCD `η>0`) draw their
/// between-step noise from a subkey split off the request seed by step index, so the trajectory is
/// deterministic for a given seed regardless of the global RNG draw order (the previous unseeded
/// `random::normal(…, None)` was order-dependent). **Same-backend determinism only** — cross-backend
/// bitwise equality is explicitly NOT a goal (RNG algorithms differ).
pub struct StepRng {
    seed: u64,
}

impl StepRng {
    /// A step-RNG keyed off the request seed. Deterministic samplers pass any value (the byte-parity
    /// branch never draws), so wrappers without a request seed use `StepRng::new(0)`.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Unit-normal noise for step `step`, drawn from a distinct subkey. The multiplier de-correlates
    /// consecutive steps; the `+1` keeps step 0 off the raw seed used for the init-noise prior.
    fn normal(&self, shape: &[i32], step: usize) -> Result<Array> {
        let sub = self
            .seed
            .wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(step as u64 + 1));
        let key = random::key(sub)?;
        Ok(random::normal::<f32>(shape, None, None, Some(&key))?)
    }
}

/// DDPM model-input scaling: `cast(c_in · x, model_dtype)`. `c_in = 1` (LCM/TCD) skips the multiply
/// to stay byte-identical to the original `x.as_dtype(model_dtype)`.
fn scale_input(c_in: f32, model_dtype: Dtype, x: &Array) -> Result<Array> {
    let scaled = if c_in == 1.0 {
        x.clone()
    } else {
        multiply(x, scalar(c_in))?
    };
    Ok(scaled.as_dtype(model_dtype)?)
}

/// Scale unit-normal noise by `init_noise_scale` (the txt2img prior). `scale = 1` (LCM/TCD/flow-match)
/// is the identity cast to f32; Lightning multiplies by its max sigma.
fn scale_initial(scale: f32, noise: &Array) -> Result<Array> {
    let n = noise.as_dtype(Dtype::Float32)?;
    if scale == 1.0 {
        Ok(n)
    } else {
        Ok(multiply(&n, scalar(scale))?)
    }
}

/// Apply one neutral [`StepCoeffs`] to the latents: `x_next = a_x·x + a_out·out + a_noise·ε`.
///
/// **Byte-parity rule (§3.3):** when `a_x == 1.0 && a_noise == 0.0`, emit exactly `x + out·a_out`
/// (the original `flow_match_euler_step` / Lightning Euler expression), NOT `x·1.0 + …` — the F-009
/// `scheduler_and_sampler_steps_are_identical` test and the FLUX golden images must stay
/// byte-identical. `StepDtype::F32` upcasts both operands (the DDPM samplers, diffusers parity);
/// `StepDtype::Latents` computes in the latents' dtype (flow-match).
fn apply_step(
    c: &StepCoeffs,
    dt: StepDtype,
    x: &Array,
    out: &Array,
    step: usize,
    rng: &StepRng,
) -> Result<Array> {
    let (x, out) = match dt {
        StepDtype::F32 => (x.as_dtype(Dtype::Float32)?, out.as_dtype(Dtype::Float32)?),
        StepDtype::Latents => (x.clone(), out.clone()),
    };
    if c.a_x == 1.0 && c.a_noise == 0.0 {
        return Ok(add(&x, &multiply(&out, scalar(c.a_out))?)?);
    }
    let mut acc = add(
        &multiply(&x, scalar(c.a_x))?,
        &multiply(&out, scalar(c.a_out))?,
    )?;
    if c.a_noise != 0.0 {
        let noise = rng.normal(acc.shape(), step)?;
        acc = add(&acc, &multiply(&noise, scalar(c.a_noise))?)?;
    }
    Ok(acc)
}

// =================================================================================================
// Unified framework backend (epic 7114 P2, sc-7118): the gen-core `LatentOps` impl over MLX `Array`.
// =================================================================================================

/// The mlx-gen backend impl of [`gen_core::sampling::LatentOps`] — the tensor primitives the unified
/// curated samplers (Euler / Heun / DPM++ 2M·SDE / UniPC / ancestral / LCM / DDIM, sc-7117) are
/// written against. Carries the same byte-parity rules as the legacy `apply_step` so an engine's
/// DEFAULT sampler stays bit-identical after it migrates onto the unified framework (the N1 gate):
/// `scale(x, 1.0)` and `axpy(1.0, x, b, y) = x + y·b` elide the multiply-by-one, and `randn_like`
/// reuses the seed-derived [`StepRng`] subkey so a stochastic sampler is deterministic per request
/// seed regardless of global RNG draw order.
#[derive(Clone, Copy, Debug, Default)]
pub struct MlxLatentOps;

/// Lift a raw MLX op error into the backend-neutral [`gen_core::Error`] (the `LatentOps` trait is
/// declared in gen-core, so its methods return `gen_core::Result`). Routes through the existing
/// mlx-gen bridge: `Exception -> mlx_gen::Error -> gen_core::Error`.
#[inline]
fn ge<T>(r: std::result::Result<T, mlx_rs::error::Exception>) -> gen_core::Result<T> {
    r.map_err(|e| crate::Error::from(e).into())
}

/// The per-eval cancel + progress preamble shared by every curated-solver driver
/// ([`run_curated_sampler`], [`run_cfgpp_sampler`], [`run_av_curated_sampler`]) — F-068 dedup. Each
/// driver's `denoise` closure calls this at the top of every model eval, then performs its own
/// (heterogeneous: single- vs two-stream) `eval` boundary.
///
/// - Cancellation: returns [`gen_core::Error::Canceled`] so a mid-render cancel surfaces as the typed
///   variant (not stringified), landing within ~1 model eval.
/// - Progress: `current` is the count of schedule nodes already descended past — monotone and robust
///   to the multi-eval solvers (Heun / DPM++ SDE evaluate twice per step; the count stays ≤ `total`),
///   which is why it is a `σ`-scan rather than a step counter. (F-131: this scan is O(sigmas) per eval;
///   left as-is because a step-index shortcut cannot reproduce the correct value at the midpoint σ the
///   2nd-order solvers pass in — behaviour must stay identical.)
#[inline]
fn step_gate(
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    sigmas: &[f32],
    total: u32,
    sigma: f32,
) -> gen_core::Result<()> {
    if cancel.is_cancelled() {
        return Err(gen_core::Error::Canceled);
    }
    let current = (sigmas.iter().filter(|&&s| s > sigma).count() as u32 + 1).min(total);
    on_progress(Progress::Step { current, total });
    // The progress callback is also the worker's phase-boundary cancellation seam. Re-check after
    // it returns so a cancel requested at `Step` stops before the next expensive model evaluation
    // and does not leave that denoise graph for request-scope cleanup to reclaim.
    if cancel.is_cancelled() {
        return Err(gen_core::Error::Canceled);
    }
    Ok(())
}

impl LatentOps for MlxLatentOps {
    type Latent = Array;

    fn scale(&self, x: &Array, scale: f32) -> gen_core::Result<Array> {
        if scale == 1.0 {
            return Ok(x.clone());
        }
        let s = ge(scalar(scale).as_dtype(x.dtype()))?;
        ge(multiply(x, s))
    }

    fn add(&self, a: &Array, b: &Array) -> gen_core::Result<Array> {
        ge(add(a, b))
    }

    fn sub(&self, a: &Array, b: &Array) -> gen_core::Result<Array> {
        ge(subtract(a, b))
    }

    fn axpy(&self, a: f32, x: &Array, b: f32, y: &Array) -> gen_core::Result<Array> {
        // Byte-parity with apply_step's a_x==1 branch: emit `x + y·b` (multiply-by-one elided), so a
        // migrated engine's default step is bit-identical to the legacy `flow_match_euler_step`.
        let sb = ge(scalar(b).as_dtype(y.dtype()))?;
        let by = ge(multiply(y, sb))?;
        if a == 1.0 {
            return ge(add(x, &by));
        }
        let sa = ge(scalar(a).as_dtype(x.dtype()))?;
        let ax = ge(multiply(x, sa))?;
        ge(add(&ax, &by))
    }

    fn randn_like(&self, x: &Array, seed: u64, step: usize) -> gen_core::Result<Array> {
        // Reuse the legacy per-step subkey derivation (D6) — same trajectory determinism guarantees.
        // `StepRng::normal` already returns `crate::Result`, which `?` lifts into `gen_core::Error`.
        Ok(StepRng::new(seed).normal(x.shape(), step)?)
    }
}

/// The guidance-axis op extension (epic 7434 P2, sc-7439) over `mlx_rs::Array` — the twin of the
/// [`LatentOps`] impl above, backing the backend-neutral [`gen_core::guidance`] library (cfg /
/// cfg_rescale / full APG) on the MLX backend. These are the exact `mlx_rs::ops` Lens `cfg_rescale`
/// and Bernini APG already use, now behind the shared trait so the math lives once in gen-core.
///
/// The reductions return the **keepdims** tensor (reduced shape, e.g. `[B,seq,1]`), not a physical
/// full-shape broadcast: MLX broadcasts it natively in the library's downstream elementwise ops, so
/// this is the cheaper equivalent of `CpuLatentOps`'s broadcast-back reference. The `shape` argument
/// is therefore unused on MLX (the `Array` carries its own shape); `axes` may be negative
/// (`sum_axes` accepts `[-1]` per-token and `[0,2,3]` per-frame alike). All ops stay lazy — no
/// `eval` boundary is introduced here (the engine's denoise loop owns per-step `eval`).
impl GuidanceOps for MlxLatentOps {
    fn mul(&self, a: &Array, b: &Array) -> gen_core::Result<Array> {
        ge(multiply(a, b))
    }

    fn div(&self, a: &Array, b: &Array) -> gen_core::Result<Array> {
        ge(divide(a, b))
    }

    fn clamp_min(&self, x: &Array, s: f32) -> gen_core::Result<Array> {
        ge(maximum(x, scalar(s)))
    }

    fn clamp_max(&self, x: &Array, s: f32) -> gen_core::Result<Array> {
        ge(minimum(x, scalar(s)))
    }

    fn select_positive(&self, sel: &Array, a: &Array, b: &Array) -> gen_core::Result<Array> {
        let positive = ge(gt(sel, scalar(0.0)))?;
        ge(which(&positive, a, b))
    }

    fn norm_over(&self, x: &Array, _shape: &[usize], axes: &[i32]) -> gen_core::Result<Array> {
        let sq = ge(multiply(x, x))?;
        ge(sqrt(&ge(sum_axes(&sq, axes, true))?))
    }

    fn dot_over(
        &self,
        a: &Array,
        b: &Array,
        _shape: &[usize],
        axes: &[i32],
    ) -> gen_core::Result<Array> {
        let prod = ge(multiply(a, b))?;
        ge(sum_axes(&prod, axes, true))
    }
}

/// Drive a curated gen-core unified [`gen_core::sampling::Sampler`] over ANY prediction type — the
/// generalized core behind [`run_flow_sampler`] (epic 7114 P3, the per-engine adoption seam).
///
/// An engine supplies its [`gen_core::sampling::ModelSampling`] (`FlowModelSampling` for the
/// rectified-flow cohort, `DiscreteModelSampling` for the ε/DDPM cohort — SDXL/Kolors,
/// `EdmModelSampling` for the v-prediction outliers — SVD), its σ schedule, and its model forward (as
/// `predict`). The `ModelSampling` recombines the raw model output into a denoised `x0` estimate and
/// supplies the `c_in` input scaling, so the curated solver (Euler / Heun / DPM++ 2M·SDE / UniPC /
/// ancestral / LCM / DDIM) never sees the prediction type — it integrates `x0` in k-diffusion sigma
/// space regardless. This is what lets one solver library serve flow, EPS, and EDM engines alike.
///
/// - `sampler_name`: the canonical curated solver name. Unknown / `None` / a non-solver alias falls
///   back to plain Euler (N3 — never hard-fail a generation over a sampling knob).
/// - `ms`: the engine's prediction-type + noise-schedule contract.
/// - `sigmas`: the descending schedule, length `num_steps + 1`, trailing `0.0`.
/// - `predict(x_in, timestep)`: the engine's model forward returning the RAW (already CFG-combined)
///   output the prediction type expects — velocity for FLOW, ε for EPS, v for V. `x_in` is the
///   `c_in`-scaled latent (`crate::ModelSampling::input_scale`; identity for FLOW) and `timestep` is the
///   conditioning value the model embeds at this σ (`crate::ModelSampling::timestep`).
///
/// Cancellation, the per-step `eval` (so a mid-render cancel lands within ~1 model eval instead of at
/// VAE decode, and peak graph memory stays bounded — the sc-5399 rationale), and progress all route
/// through the `denoise` callback, the sole per-eval hook the callback-form `Sampler` exposes.
#[allow(clippy::too_many_arguments)]
pub fn run_curated_sampler(
    sampler_name: Option<&str>,
    ms: &dyn gen_core::sampling::ModelSampling,
    sigmas: &[f32],
    latents: Array,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    mut predict: impl FnMut(&Array, f32) -> Result<Array>,
) -> Result<Array> {
    run_curated_sampler_with_latent_hook(
        sampler_name,
        ms,
        sigmas,
        latents,
        seed,
        cancel,
        on_progress,
        |_, _| {},
        &mut predict,
    )
}

/// Tracks the callback phase for curated solvers that evaluate the model twice per outer step.
///
/// Heun evaluates at the next schedule sigma for its corrector, while DPM++ SDE evaluates at an
/// off-schedule midpoint. Neither evaluation is a new denoise step. Keeping this small adapter at
/// the MLX driver seam avoids widening the backend-neutral sampler trait solely for decorative
/// output while still exposing the exact outer-step cadence to providers.
struct CuratedStepObserver {
    solver: gen_core::sampling::Solver,
    step: usize,
    correction_pending: bool,
}

impl CuratedStepObserver {
    fn new(solver: gen_core::sampling::Solver) -> Self {
        Self {
            solver,
            step: 0,
            correction_pending: false,
        }
    }

    fn is_step_start(&mut self, sigmas: &[f32], sigma: f32) -> bool {
        if self.correction_pending {
            self.correction_pending = false;
            return false;
        }

        let step = self.step;
        self.step = self.step.saturating_add(1);
        let evaluates_twice = match self.solver {
            gen_core::sampling::Solver::Heun | gen_core::sampling::Solver::DpmppSde => true,
            gen_core::sampling::Solver::Euler
            | gen_core::sampling::Solver::EulerAncestral
            | gen_core::sampling::Solver::Dpmpp2m
            | gen_core::sampling::Solver::UniPc
            | gen_core::sampling::Solver::Lcm
            | gen_core::sampling::Solver::Ddim
            | gen_core::sampling::Solver::ErSde
            | gen_core::sampling::Solver::Dpmpp2mSde => false,
        };
        let has_correction = evaluates_twice
            && sigma != 0.0
            && sigmas.get(step + 1).is_some_and(|&next| next != 0.0);
        self.correction_pending = has_correction;
        true
    }
}

/// [`run_curated_sampler`] with a best-effort observer of the raw, unscaled latent at the start of
/// each actual solver step. Heun corrector evaluations and DPM++ SDE midpoint evaluations are
/// deliberately suppressed. The legacy wrapper supplies a no-op hook, preserving the exact
/// numerical integration path and output bytes.
#[allow(clippy::too_many_arguments)]
pub fn run_curated_sampler_with_latent_hook(
    sampler_name: Option<&str>,
    ms: &dyn gen_core::sampling::ModelSampling,
    sigmas: &[f32],
    latents: Array,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    mut on_latent: impl FnMut(&Array, f32),
    mut predict: impl FnMut(&Array, f32) -> Result<Array>,
) -> Result<Array> {
    use gen_core::sampling::{denoise as gc_denoise, Sampler, Solver};

    let ops = MlxLatentOps;
    let total = sigmas.len().saturating_sub(1).max(1) as u32;
    // N3: a curated name routes to its solver; an unknown name / non-solver alias falls back to Euler.
    let solver = sampler_name
        .and_then(Solver::from_name)
        .unwrap_or(Solver::Euler);
    let sampler: Box<dyn Sampler<MlxLatentOps>> = solver.boxed();
    let mut step_observer = CuratedStepObserver::new(solver);

    let mut denoise_fn = |x: &Array, sigma: f32| -> gen_core::Result<Array> {
        step_gate(cancel, on_progress, sigmas, total, sigma)?;
        // Per-eval compute boundary: force the prior step's lazy graph now (MLX is lazy, so without
        // this the whole denoise is one un-cancellable graph that only runs at VAE decode). Output-
        // neutral. The multistep solvers reuse the previous denoised estimate, but the latent handed
        // back here is always the fresh node to integrate from, so evaluating it is safe.
        ge(mlx_rs::transforms::eval([x]))?;
        if step_observer.is_step_start(sigmas, sigma) {
            on_latent(x, sigma);
        }
        gc_denoise(&ops, ms, x, sigma, |xin, t| {
            predict(xin, t).map_err(Into::into)
        })
    };

    let out = sampler
        .sample(&ops, ms, &mut denoise_fn, latents, sigmas, seed)
        .map_err(crate::Error::from)?;
    // Force the final step's advancement (never seen by the callback, which evals only inputs).
    mlx_rs::transforms::eval([&out])?;
    Ok(out)
}

/// The CFG++ twin of [`run_curated_sampler`] (epic 7434, sc-8256). Drives a
/// [`gen_core::sampling::CfgPpSampler`] — which lands each step on the GUIDED `x0` but renoises from the
/// UNCONDITIONAL `x0` — over the engine's `predict_pair` closure, which surfaces the `(guided, uncond)`
/// raw-model pair the plain CFG path collapses into one output. `base_sampler_name` is the BASE solver
/// the user selected (`euler`/`ddim` → `euler_cfg++`, `dpmpp_2m` → `dpmpp_2m_cfg++`); a base with no
/// paper-validated CFG++ form, or `None`, falls back to `euler_cfg++` (N3 — never hard-fail over the
/// knob). The engine routes here ONLY when `req.guidance_method == "cfg_pp"` and the base solver
/// supports it, so the default path ([`run_curated_sampler`]) is byte-untouched.
///
/// Cancellation, the per-step `eval` boundary, and progress mirror [`run_curated_sampler`] exactly.
#[allow(clippy::too_many_arguments)]
pub fn run_cfgpp_sampler(
    base_sampler_name: Option<&str>,
    ms: &dyn gen_core::sampling::ModelSampling,
    sigmas: &[f32],
    latents: Array,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    mut predict_pair: impl FnMut(&Array, f32) -> Result<(Array, Array)>,
) -> Result<Array> {
    run_cfgpp_sampler_with_latent_hook(
        base_sampler_name,
        ms,
        sigmas,
        latents,
        cancel,
        on_progress,
        |_, _| {},
        &mut predict_pair,
    )
}

/// [`run_cfgpp_sampler`] with the same actual-step raw-latent observer offered by
/// [`run_curated_sampler_with_latent_hook`]. Every shipped CFG++ solver performs exactly one model
/// evaluation per outer step, so no correction/midpoint filtering is required here.
#[allow(clippy::too_many_arguments)]
pub fn run_cfgpp_sampler_with_latent_hook(
    base_sampler_name: Option<&str>,
    ms: &dyn gen_core::sampling::ModelSampling,
    sigmas: &[f32],
    latents: Array,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    mut on_latent: impl FnMut(&Array, f32),
    mut predict_pair: impl FnMut(&Array, f32) -> Result<(Array, Array)>,
) -> Result<Array> {
    use gen_core::sampling::{
        base_supports_cfgpp, cfgpp_denoise as gc_cfgpp_denoise, cfgpp_sampler_for, CfgPpSampler,
        Solver,
    };

    let ops = MlxLatentOps;
    let total = sigmas.len().saturating_sub(1).max(1) as u32;
    // N3: route the chosen base to its CFG++ variant; a base with no CFG++ form (or None) → euler_cfg++.
    let base = base_sampler_name
        .and_then(Solver::from_name)
        .filter(|solver| base_supports_cfgpp(*solver))
        .unwrap_or(Solver::Euler);
    let sampler: Box<dyn CfgPpSampler<MlxLatentOps>> = cfgpp_sampler_for::<MlxLatentOps>(base)
        .expect("base_supports_cfgpp(Euler) holds, so a sampler always resolves");

    let mut denoise_fn = |x: &Array, sigma: f32| -> gen_core::Result<(Array, Array)> {
        step_gate(cancel, on_progress, sigmas, total, sigma)?;
        ge(mlx_rs::transforms::eval([x]))?;
        on_latent(x, sigma);
        gc_cfgpp_denoise(&ops, ms, x, sigma, |xin, t| {
            predict_pair(xin, t).map_err(Into::into)
        })
    };

    let out = sampler
        .sample(&ops, &mut denoise_fn, latents, sigmas)
        .map_err(crate::Error::from)?;
    mlx_rs::transforms::eval([&out])?;
    Ok(out)
}

/// Drive a curated solver over a flow-match (rectified-flow) sigma schedule — the thin
/// [`run_curated_sampler`] wrapper for the FLOW cohort (FLUX / Qwen / Chroma / Z-Image / Boogu / LTX /
/// Wan). `conv` selects whether the model is fed the raw sigma ([`TimestepConvention::Sigma`]) or
/// `1 − σ` ([`TimestepConvention::OneMinusSigma`]); `predict` returns the RAW (already CFG-combined)
/// velocity. `euler` over FLOW reproduces the legacy [`FlowMatchSampler`] loop within the N1 tolerance.
///
/// The time-shift lives entirely in `sigmas` (resolved by [`resolve_flow_schedule`]), so
/// `FlowModelSampling::new(conv)` (mu = 0) is the correct integration contract here — its `timestep` /
/// `denoised_coeffs` are mu-independent.
#[allow(clippy::too_many_arguments)]
pub fn run_flow_sampler(
    sampler_name: Option<&str>,
    conv: TimestepConvention,
    sigmas: &[f32],
    latents: Array,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    predict: impl FnMut(&Array, f32) -> Result<Array>,
) -> Result<Array> {
    run_flow_sampler_with_latent_hook(
        sampler_name,
        conv,
        sigmas,
        latents,
        seed,
        cancel,
        on_progress,
        |_, _| {},
        predict,
    )
}

/// [`run_flow_sampler`] with a best-effort observer of the raw flow latent at each actual outer
/// solver step. Multi-evaluation solvers expose only their outer-step input: Heun correctors and
/// DPM++ SDE midpoint evaluations are suppressed by [`run_curated_sampler_with_latent_hook`]. The
/// legacy wrapper supplies an inert observer and therefore preserves the numerical path exactly.
#[allow(clippy::too_many_arguments)]
pub fn run_flow_sampler_with_latent_hook(
    sampler_name: Option<&str>,
    conv: TimestepConvention,
    sigmas: &[f32],
    latents: Array,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    on_latent: impl FnMut(&Array, f32),
    predict: impl FnMut(&Array, f32) -> Result<Array>,
) -> Result<Array> {
    let ms = gen_core::sampling::FlowModelSampling::new(conv);
    run_curated_sampler_with_latent_hook(
        sampler_name,
        &ms,
        sigmas,
        latents,
        seed,
        cancel,
        on_progress,
        on_latent,
        predict,
    )
}

// =================================================================================================
// Joint two-stream (video+audio) curated sampling — LTX's cross-modal denoise (epic 7114, sc-7122).
// =================================================================================================

/// A joint video+audio latent pair — the [`gen_core::sampling::LatentOps::Latent`] for LTX's
/// cross-modal denoise, whose two streams (`[B,128,F,H,W]` video + `[B,8,T,16]` audio) are integrated
/// **together** by one curated solver each step (the AvDiT couples them via cross-modal attention). The
/// single-`Array` [`MlxLatentOps`] cannot represent this, so the two-stream variant exists.
#[derive(Clone)]
pub struct AvLatents {
    pub video: Array,
    pub audio: Array,
}

/// [`gen_core::sampling::LatentOps`] over [`AvLatents`] — applies each solver op to BOTH streams, so the
/// gen-core curated solvers (Euler / Heun / DPM++ 2M·SDE / UniPC / ancestral / DDIM) drive LTX's joint
/// video+audio denoise. Each per-stream op reuses [`MlxLatentOps`], so the byte-parity rules
/// (`scale(x,1)`/`axpy(1,…)` elide the multiply) hold per stream.
#[derive(Clone, Copy, Debug, Default)]
pub struct MlxAvLatentOps;

impl LatentOps for MlxAvLatentOps {
    type Latent = AvLatents;

    fn scale(&self, x: &AvLatents, scale: f32) -> gen_core::Result<AvLatents> {
        Ok(AvLatents {
            video: MlxLatentOps.scale(&x.video, scale)?,
            audio: MlxLatentOps.scale(&x.audio, scale)?,
        })
    }

    fn add(&self, a: &AvLatents, b: &AvLatents) -> gen_core::Result<AvLatents> {
        Ok(AvLatents {
            video: MlxLatentOps.add(&a.video, &b.video)?,
            audio: MlxLatentOps.add(&a.audio, &b.audio)?,
        })
    }

    fn sub(&self, a: &AvLatents, b: &AvLatents) -> gen_core::Result<AvLatents> {
        Ok(AvLatents {
            video: MlxLatentOps.sub(&a.video, &b.video)?,
            audio: MlxLatentOps.sub(&a.audio, &b.audio)?,
        })
    }

    fn axpy(&self, a: f32, x: &AvLatents, b: f32, y: &AvLatents) -> gen_core::Result<AvLatents> {
        Ok(AvLatents {
            video: MlxLatentOps.axpy(a, &x.video, b, &y.video)?,
            audio: MlxLatentOps.axpy(a, &x.audio, b, &y.audio)?,
        })
    }

    fn randn_like(&self, x: &AvLatents, seed: u64, step: usize) -> gen_core::Result<AvLatents> {
        // Distinct subkeys per stream (the audio seed is XOR-shifted) so the two streams' stochastic
        // noise is decorrelated; each reuses the per-step `StepRng` derivation.
        Ok(AvLatents {
            video: MlxLatentOps.randn_like(&x.video, seed, step)?,
            audio: MlxLatentOps.randn_like(&x.audio, seed ^ 0xA5A5_5A5A_C3C3_3C3C, step)?,
        })
    }
}

/// Drive a curated unified solver over LTX's **joint video+audio** flow-match schedule — the two-stream
/// sibling of [`run_flow_sampler`] (epic 7114, sc-7122). The model is velocity-prediction over the FLOW
/// [`TimestepConvention::Sigma`] convention for BOTH streams; `predict(av_in, sigma)` returns the raw
/// `(video_velocity, audio_velocity)`. The per-step `eval`/cancel/progress contract matches
/// [`run_curated_sampler`], evaluating both streams each step. Used for LTX's distilled T2V+A path
/// (the per-token-σ I2V path with its post-step `apply_denoise_mask` blend stays native).
#[allow(clippy::too_many_arguments)]
pub fn run_av_curated_sampler(
    sampler_name: Option<&str>,
    sigmas: &[f32],
    latents: AvLatents,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    mut predict: impl FnMut(&AvLatents, f32) -> Result<AvLatents>,
) -> Result<AvLatents> {
    use gen_core::sampling::{
        denoise as gc_denoise, sampler_by_name, Euler, FlowModelSampling, Sampler,
    };

    let ops = MlxAvLatentOps;
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    let total = sigmas.len().saturating_sub(1).max(1) as u32;
    let sampler: Box<dyn Sampler<MlxAvLatentOps>> = sampler_name
        .and_then(sampler_by_name::<MlxAvLatentOps>)
        .unwrap_or_else(|| Box::new(Euler));

    let mut denoise_fn = |x: &AvLatents, sigma: f32| -> gen_core::Result<AvLatents> {
        step_gate(cancel, on_progress, sigmas, total, sigma)?;
        ge(mlx_rs::transforms::eval([&x.video, &x.audio]))?;
        gc_denoise(&ops, &ms, x, sigma, |xin, t| {
            predict(xin, t).map_err(Into::into)
        })
    };

    let out = sampler
        .sample(&ops, &ms, &mut denoise_fn, latents, sigmas, seed)
        .map_err(crate::Error::from)?;
    mlx_rs::transforms::eval([&out.video, &out.audio])?;
    Ok(out)
}

/// Resolve the descending flow sigma schedule for an engine, honoring a per-generation curated
/// `scheduler` selection (epic 7114 scheduler axis, sc-7120). The engine-side counterpart to
/// [`run_flow_sampler`]'s `sampler` knob: the *scheduler* picks the σ-schedule (where the steps land),
/// the *sampler* picks the integrator (how each step advances).
///
/// - `scheduler_name`: the canonical curated scheduler name (`normal` / `simple` / `karras` /
///   `exponential` / `sgm_uniform` / `beta` / `ddim_uniform`). `None`, an unknown name, or a native
///   alias (e.g. `linear` / `flow_match_euler`) falls back to `native` (N3 — never hard-fail a
///   generation over a scheduling knob; the engine's native schedule is the byte-exact default).
/// - `mu`: the engine's time-shift (`mlx_gen::scheduler::compute_mu(image_seq_len, steps)` for the
///   dynamic-shift models, `shift.ln()` for a static-shift model, `0.0` for an unshifted one). A curated
///   schedule is built over a [`gen_core::sampling::FlowModelSampling`] carrying this `mu` so
///   `normal` / `sgm_uniform` / … stay consistent with the engine's resolution-/config-dependent shift
///   instead of degrading to a linear σ ramp (which would starve a high-shift model of high-noise steps).
/// - `steps`: the denoise step count.
/// - `native`: the engine's exact native schedule (length `steps + 1`, trailing `0.0`), returned
///   verbatim on the default path so the per-engine N1 default-parity gate holds byte-for-byte.
///
/// Schedule construction is **convention-independent** — the σ schedule is the same noise-fraction ramp
/// however the model consumes σ — so this always builds with [`TimestepConvention::Sigma`]; the engine's
/// own conditioning convention (`Sigma` / `OneMinusSigma`) is applied separately by [`run_flow_sampler`].
/// A curated scheduler may return a length other than `steps + 1` (`ddim_uniform` / `beta` re-stride),
/// which simply changes the effective step count — the same behaviour ComfyUI / diffusers have.
pub fn resolve_flow_schedule(
    scheduler_name: Option<&str>,
    mu: f32,
    steps: usize,
    native: &[f32],
) -> Vec<f32> {
    let ms = gen_core::sampling::FlowModelSampling::with_shift(TimestepConvention::Sigma, mu);
    resolve_schedule(scheduler_name, &ms, steps, native)
}

/// Resolve a descending σ schedule honoring a per-generation curated `scheduler`, over ANY
/// [`gen_core::sampling::ModelSampling`] — the generalized core behind [`resolve_flow_schedule`] (epic
/// 7114 scheduler axis). An unset / unknown / native-aliased name returns `native` verbatim (the N1
/// byte-exact default); a curated name builds the schedule via [`gen_core::sampling::schedule_sigmas`]
/// over `ms`, which reads its σ-table / timestep↔sigma map — so `normal` / `karras` / `sgm_uniform` /
/// `simple` / `beta` / `ddim_uniform` / `exponential` land correctly for the ε/DDPM (`DiscreteModelSampling`),
/// EDM (`EdmModelSampling`), and flow (`FlowModelSampling`) contracts alike. A curated scheduler may
/// return a length other than `steps + 1` (`ddim_uniform` / `beta` re-stride), changing the effective
/// step count — the same behaviour ComfyUI / diffusers have.
pub fn resolve_schedule(
    scheduler_name: Option<&str>,
    ms: &dyn gen_core::sampling::ModelSampling,
    steps: usize,
    native: &[f32],
) -> Vec<f32> {
    use gen_core::sampling::{schedule_sigmas, Scheduler};
    match scheduler_name.and_then(Scheduler::from_name) {
        Some(sched) => schedule_sigmas(sched, ms, steps),
        None => native.to_vec(),
    }
}

/// The curated unified-framework **sampler** menu (epic 7114 decision 2) as capability strings — every
/// [`gen_core::sampling::Solver`] name, in menu order. A flow-match (or DDPM) engine advertises this
/// in its [`gen_core::generator::Capabilities`] `samplers` list (plus any legacy alias it still
/// honors, e.g. `flow_match`) so the per-generation `sampler` knob can select any curated integrator
/// and [`run_flow_sampler`] routes the name to its solver.
pub fn curated_sampler_names() -> Vec<&'static str> {
    gen_core::sampling::Solver::ALL
        .iter()
        .map(|s| s.name())
        .collect()
}

/// The curated unified-framework **scheduler** menu (epic 7114 decision 2) as capability strings —
/// every [`gen_core::sampling::Scheduler`] name, in menu order. Engines that expose the sigma-schedule
/// axis advertise this in their `schedulers` list; selecting one builds the schedule via
/// [`gen_core::sampling::schedule_sigmas`].
pub fn curated_scheduler_names() -> Vec<&'static str> {
    gen_core::sampling::Scheduler::ALL
        .iter()
        .map(|s| s.name())
        .collect()
}

// =================================================================================================
// LCM — diffusers `LCMScheduler` (epsilon prediction; SDXL world). Policy: gen_core LcmPolicy.
// =================================================================================================

/// Latent Consistency Model sampler. Predicts `x₀` from `eps`, applies the consistency boundary
/// scalings `c_skip`/`c_out`, and re-noises between steps. ~2–8 steps; CFG ≈ 1.
pub struct LcmSampler {
    policy: LcmPolicy,
    /// The compute dtype the model's forward expects (latents are cast to this in
    /// [`DiffusionSampler::scale_model_input`]); the step math runs f32.
    model_dtype: Dtype,
    rng: StepRng,
}

impl LcmSampler {
    /// Build for `num_steps` inference steps. `original_inference_steps` is diffusers' default 50.
    /// `seed` is the request seed driving the deterministic between-step re-noise (D6).
    pub fn new(
        sched: AlphaSchedule,
        num_train_timesteps: usize,
        original_inference_steps: usize,
        num_steps: usize,
        model_dtype: Dtype,
        seed: u64,
    ) -> Self {
        Self {
            policy: LcmPolicy::new(
                sched,
                num_train_timesteps,
                original_inference_steps,
                num_steps,
            ),
            model_dtype,
            rng: StepRng::new(seed),
        }
    }

    /// The deterministic consistency prediction at step `i` — diffusers' `denoised` (before the
    /// between-step re-noise). Used by the scheduler-isolation parity gate.
    pub fn denoised(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        apply_step(
            &self.policy.denoised_coeffs(i),
            StepDtype::F32,
            x,
            model_output,
            i,
            &self.rng,
        )
    }
}

impl DiffusionSampler for LcmSampler {
    fn num_steps(&self) -> usize {
        self.policy.num_steps()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.policy.coeffs(i).timestep
    }

    fn scale_model_input(&self, x: &Array, i: usize) -> Result<Array> {
        scale_input(self.policy.coeffs(i).c_in, self.model_dtype, x)
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        scale_initial(self.policy.init_noise_scale(), noise)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        apply_step(
            &self.policy.coeffs(i),
            self.policy.step_dtype(),
            x,
            model_output,
            i,
            &self.rng,
        )
    }
}

// =================================================================================================
// SDXL-Lightning — diffusers `EulerDiscreteScheduler(timestep_spacing="trailing")`. Deterministic.
// =================================================================================================

/// SDXL-Lightning sampler: trailing-spaced Euler. The latents live in diffusers' un-normalized
/// (σ-scaled) space; [`DiffusionSampler::scale_model_input`] divides by `√(σ²+1)` before the U-Net.
pub struct LightningSampler {
    policy: LightningPolicy,
    model_dtype: Dtype,
}

impl LightningSampler {
    /// Build for `num_steps` (2/4/8). Trailing-spaced timesteps + interpolated sigmas (policy layer).
    pub fn new(
        sched: &AlphaSchedule,
        num_train_timesteps: usize,
        num_steps: usize,
        model_dtype: Dtype,
    ) -> Self {
        Self {
            policy: LightningPolicy::new(sched, num_train_timesteps, num_steps),
            model_dtype,
        }
    }
}

impl DiffusionSampler for LightningSampler {
    fn num_steps(&self) -> usize {
        self.policy.num_steps()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.policy.coeffs(i).timestep
    }

    fn scale_model_input(&self, x: &Array, i: usize) -> Result<Array> {
        // x · c_in (= 1/√(σ²+1)), then cast to the model's compute dtype.
        scale_input(self.policy.coeffs(i).c_in, self.model_dtype, x)
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        // latents = randn · init_noise_sigma (the largest sigma).
        scale_initial(self.policy.init_noise_scale(), noise)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        // Euler ε-pred step, gamma=0: `x + eps·(σ_{i+1} − σ_i)`, upcast to f32 (a_x=1, a_noise=0).
        apply_step(
            &self.policy.coeffs(i),
            self.policy.step_dtype(),
            x,
            model_output,
            i,
            &StepRng::new(0),
        )
    }
}

// =================================================================================================
// Hyper-SD — diffusers `TCDScheduler` (epsilon prediction). Policy: gen_core TcdPolicy.
// =================================================================================================

/// Hyper-SD sampler: Trajectory Consistency Distillation. Like LCM but steps to an intermediate
/// noise level `s = ⌊(1−η)·t_prev⌋` and (for `η>0`) re-noises across the `t_prev`/`s` gap.
pub struct TcdSampler {
    policy: TcdPolicy,
    model_dtype: Dtype,
    rng: StepRng,
}

impl TcdSampler {
    /// Build for `num_steps`. `original_inference_steps` is diffusers' default 50; `eta` is the
    /// stochasticity (`0.0` = deterministic; ByteDance's unified LoRA recommends ~`0.3`). `seed`
    /// drives the deterministic `η>0` re-noise (D6).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sched: AlphaSchedule,
        num_train_timesteps: usize,
        original_inference_steps: usize,
        num_steps: usize,
        eta: f32,
        model_dtype: Dtype,
        seed: u64,
    ) -> Self {
        Self {
            policy: TcdPolicy::new(
                sched,
                num_train_timesteps,
                original_inference_steps,
                num_steps,
                eta,
            ),
            model_dtype,
            rng: StepRng::new(seed),
        }
    }

    /// The deterministic noised prediction `x_s` at step `i` — diffusers' `pred_noised_sample`
    /// (before the `η>0` re-noise). Used by the scheduler-isolation parity gate.
    pub fn pred_noised(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        apply_step(
            &self.policy.pred_noised_coeffs(i),
            StepDtype::F32,
            x,
            model_output,
            i,
            &self.rng,
        )
    }
}

impl DiffusionSampler for TcdSampler {
    fn num_steps(&self) -> usize {
        self.policy.num_steps()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.policy.coeffs(i).timestep
    }

    fn scale_model_input(&self, x: &Array, i: usize) -> Result<Array> {
        scale_input(self.policy.coeffs(i).c_in, self.model_dtype, x)
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        scale_initial(self.policy.init_noise_scale(), noise)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        apply_step(
            &self.policy.coeffs(i),
            self.policy.step_dtype(),
            x,
            model_output,
            i,
            &self.rng,
        )
    }
}

// =================================================================================================
// Flow-match — the rectified-flow world (FLUX.1 / Qwen-Image). Policy: gen_core FlowMatchPolicy.
// =================================================================================================

/// A flow-match (rectified-flow) Euler sampler driven by a precomputed sigma schedule. The schedule
/// is built by the model family (FLUX's `build_linear_sigmas`, Qwen's `qwen_scheduler` and its
/// Lightning builder), so this sampler is family-neutral — it owns only the flow-match update. The
/// model is velocity-prediction, the latents stay f32, and the prior is unit noise.
pub struct FlowMatchSampler {
    policy: FlowMatchPolicy,
}

impl FlowMatchSampler {
    /// Build from a precomputed sigma schedule (length `num_steps + 1`, trailing `0.0`). A schedule
    /// needs at least one step + the terminal `0`; this is debug-asserted here (the downstream `step`
    /// indexing requires it) — previously the doc promised a panic the code never enforced (F-086).
    /// FLUX/Qwen feed the raw sigma as the model timestep ([`TimestepConvention::Sigma`]).
    pub fn new(sigmas: Vec<f32>) -> Self {
        debug_assert!(
            sigmas.len() >= 2,
            "FlowMatchSampler::new: schedule needs >= 2 entries (>=1 step + terminal 0), got {}",
            sigmas.len()
        );
        Self {
            policy: FlowMatchPolicy::new(sigmas, TimestepConvention::Sigma),
        }
    }

    /// The schedule sigma at step `i` (length `num_steps + 1`, trailing `0.0`). For flow-match this
    /// equals [`DiffusionSampler::timestep`]; img2img seeds its noise blend at `sigma(start_step)`.
    pub fn sigma(&self, i: usize) -> f32 {
        self.policy.sigma_at_node(i)
    }
}

impl DiffusionSampler for FlowMatchSampler {
    fn num_steps(&self) -> usize {
        self.policy.num_steps()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.policy.coeffs(i).timestep
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        // init_noise_scale = 1 → identity cast to f32 (FLUX seeds its own noise via `create_noise`).
        scale_initial(self.policy.init_noise_scale(), noise)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        // Forward Euler on the velocity field: `x + v·(σ_{i+1} − σ_i)` (a_x=1, a_noise=0 → the
        // byte-parity branch, computed in the latents' dtype — identical to `FlowMatchEuler::step`
        // and the original `flow_match_euler_step`, F-009).
        apply_step(
            &self.policy.coeffs(i),
            self.policy.step_dtype(),
            x,
            model_output,
            i,
            &StepRng::new(0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sdxl_sched() -> AlphaSchedule {
        AlphaSchedule::scaled_linear(1000, 0.00085, 0.012)
    }

    fn scalar1(v: f32) -> Array {
        Array::from_slice(&[v], &[1])
    }
    fn val(a: &Array) -> f32 {
        a.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>()[0]
    }

    #[test]
    fn samplers_report_step_count() {
        let lcm = LcmSampler::new(sdxl_sched(), 1000, 50, 4, Dtype::Float32, 0);
        assert_eq!(lcm.num_steps(), 4);
        let light = LightningSampler::new(&sdxl_sched(), 1000, 2, Dtype::Float32);
        assert_eq!(light.num_steps(), 2);
        let tcd = TcdSampler::new(sdxl_sched(), 1000, 50, 8, 0.0, Dtype::Float32, 0);
        assert_eq!(tcd.num_steps(), 8);
    }

    // The per-step tensor application reproduces the diffusers scalars via the neutral coefficients
    // (the same references the gen_core::sampling policy goldens assert, now through MLX arrays).
    #[test]
    fn lcm_step0_denoised_matches_diffusers() {
        let s = LcmSampler::new(sdxl_sched(), 1000, 50, 4, Dtype::Float32, 0);
        let d = s.denoised(&scalar1(0.7), &scalar1(0.3), 0).unwrap();
        assert!((val(&d) - (-5.835_607)).abs() < 1e-3, "got {}", val(&d));
    }

    #[test]
    fn lightning_step0_matches_diffusers() {
        let s = LightningSampler::new(&sdxl_sched(), 1000, 4, Dtype::Float32);
        let scaled = s.scale_model_input(&scalar1(0.3), 0).unwrap();
        assert!(
            (val(&scaled) - 0.020_479_47).abs() < 1e-4,
            "scaled {}",
            val(&scaled)
        );
        let prev = s.step(&scalar1(0.7), &scalar1(0.3), 0).unwrap();
        assert!(
            (val(&prev) - (-7.073_041)).abs() < 1e-3,
            "prev {}",
            val(&prev)
        );
    }

    #[test]
    fn tcd_eta0_step0_pred_noised_matches_diffusers() {
        let s = TcdSampler::new(sdxl_sched(), 1000, 50, 4, 0.0, Dtype::Float32, 0);
        let pn = s.pred_noised(&scalar1(0.7), &scalar1(0.3), 0).unwrap();
        assert!((val(&pn) - (-0.651_963_8)).abs() < 1e-4, "got {}", val(&pn));
    }

    // Flow-match (FLUX): the sampler must reproduce the proven inline FLUX loop `x + v·(σ_{i+1}−σ_i)`
    // exactly, with `timestep(i)=σ_i` and `num_steps = len-1`. Schnell-style 4-step linear sigmas.
    #[test]
    fn flow_match_step_matches_inline_euler() {
        let sigmas = vec![1.0_f32, 0.75, 0.5, 0.25, 0.0];
        let s = FlowMatchSampler::new(sigmas.clone());
        assert_eq!(s.num_steps(), 4);
        for (i, &sig) in sigmas.iter().take(4).enumerate() {
            assert_eq!(s.timestep(i), sig);
        }
        // step 0: x=0.3, v=0.7 → 0.3 + 0.7·(0.75−1.0) = 0.125 (the exact inline-loop arithmetic).
        let out = s.step(&scalar1(0.7), &scalar1(0.3), 0).unwrap();
        assert!((val(&out) - 0.125).abs() < 1e-6, "got {}", val(&out));
        // last step integrates to σ=0: dt = 0.0 − 0.25 = −0.25.
        let last = s.step(&scalar1(0.4), &scalar1(0.2), 3).unwrap();
        assert!(
            (val(&last) - (0.2 - 0.1)).abs() < 1e-6,
            "got {}",
            val(&last)
        );
    }

    #[test]
    fn flow_match_initial_noise_is_unit_identity_f32() {
        let s = FlowMatchSampler::new(vec![1.0_f32, 0.5, 0.0]);
        let n = Array::from_slice(&[0.3_f32, -0.7, 1.1], &[3]);
        let scaled = s.scale_initial_noise(&n).unwrap();
        // init_noise_sigma = 1 → identity (×1), dtype f32.
        assert_eq!(scaled.dtype(), Dtype::Float32);
        let got = scaled.as_slice::<f32>();
        for (a, b) in got.iter().zip([0.3_f32, -0.7, 1.1]) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn step_progress_callback_can_cancel_before_model_evaluation() {
        let cancel = CancelFlag::new();
        let callback_cancel = cancel.clone();
        let result = step_gate(
            &cancel,
            &mut |progress| {
                assert!(matches!(
                    progress,
                    Progress::Step {
                        current: 1,
                        total: 1
                    }
                ));
                callback_cancel.cancel();
            },
            &[1.0, 0.0],
            1,
            1.0,
        );
        assert!(matches!(result, Err(gen_core::Error::Canceled)));
    }

    // --- Unified framework backend: MlxLatentOps (epic 7114 P2, sc-7118) ---------------------------

    use gen_core::sampling::{
        build_flow_sigmas, compute_mu, denoise, image_seq_len, Euler, FlowModelSampling, Sampler,
    };
    use mlx_rs::ops::eq;

    fn arr(v: &[f32]) -> Array {
        Array::from_slice(v, &[v.len() as i32])
    }
    fn arrays_eq(a: &Array, b: &Array) -> bool {
        eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }
    /// A reference flow velocity model `v = 0.3·x + 0.1` over MLX (matches the gen-core byte-equiv).
    fn stub_velocity(xin: &Array) -> Result<Array> {
        Ok(add(&multiply(xin, scalar(0.3))?, scalar(0.1))?)
    }

    #[test]
    fn mlx_latent_ops_scale_add_sub() {
        let ops = MlxLatentOps;
        let a = arr(&[1.0, 2.0, 3.0]);
        let b = arr(&[0.5, -1.0, 4.0]);
        assert!(arrays_eq(
            &ops.scale(&a, 2.0).unwrap(),
            &arr(&[2.0, 4.0, 6.0])
        ));
        // scale by 1.0 is a byte-identical clone (no kernel).
        assert!(arrays_eq(&ops.scale(&a, 1.0).unwrap(), &a));
        assert!(arrays_eq(&ops.add(&a, &b).unwrap(), &arr(&[1.5, 1.0, 7.0])));
        assert!(arrays_eq(
            &ops.sub(&a, &b).unwrap(),
            &arr(&[0.5, 3.0, -1.0])
        ));
    }

    #[test]
    fn mlx_axpy_a1_is_byte_identical_to_legacy_branch() {
        // axpy(1.0, x, b, y) must equal `x + y·b` exactly (the apply_step byte-parity branch).
        let ops = MlxLatentOps;
        let x = arr(&[0.3, -1.2, 2.5]);
        let y = arr(&[0.7, 0.1, -0.4]);
        let got = ops.axpy(1.0, &x, 0.25, &y).unwrap();
        let want = add(&x, multiply(&y, scalar(0.25)).unwrap()).unwrap();
        assert!(arrays_eq(&got, &want), "axpy a=1 not byte-identical");
        // General a: 2·x + (−3)·y.
        let got2 = ops.axpy(2.0, &x, -3.0, &y).unwrap();
        let want2 = add(
            multiply(&x, scalar(2.0)).unwrap(),
            multiply(&y, scalar(-3.0)).unwrap(),
        )
        .unwrap();
        assert!(arrays_eq(&got2, &want2));
    }

    #[test]
    fn mlx_randn_like_is_deterministic_shaped_and_seed_keyed() {
        let ops = MlxLatentOps;
        let x = arr(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let a = ops.randn_like(&x, 42, 0).unwrap();
        assert_eq!(a.shape(), &[5]);
        assert!(arrays_eq(&a, &ops.randn_like(&x, 42, 0).unwrap()));
        assert!(!arrays_eq(&a, &ops.randn_like(&x, 42, 1).unwrap()));
        assert!(!arrays_eq(&a, &ops.randn_like(&x, 43, 0).unwrap()));
    }

    #[test]
    fn mlx_unified_euler_matches_legacy_flowmatch() {
        // The N1 proof on the real backend: gen-core Euler over MlxLatentOps + a FLOW ModelSampling
        // reproduces the legacy FlowMatchSampler trajectory (same stub velocity, same sigmas).
        let ops = MlxLatentOps;
        let sigmas = build_flow_sigmas(8, compute_mu(image_seq_len(1024, 1024), 8));
        let x_init = arr(&[0.3, -1.1, 2.0, 0.05, -0.4, 1.7]);

        // Legacy FlowMatchSampler path (the byte-parity apply_step).
        let legacy_sampler = FlowMatchSampler::new(sigmas.clone());
        let mut legacy = x_init.clone();
        for i in 0..legacy_sampler.num_steps() {
            let v = stub_velocity(&legacy).unwrap(); // c_in=1 -> model input = x
            legacy = legacy_sampler.step(&v, &legacy, i).unwrap();
        }

        // Unified Euler over MlxLatentOps.
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let mut dn = |x: &Array, s: f32| {
            denoise(&ops, &ms, x, s, |xin, _t| {
                stub_velocity(xin).map_err(Into::into)
            })
        };
        let unified = Euler
            .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, 0)
            .unwrap();

        let (lg, un) = (legacy.as_slice::<f32>(), unified.as_slice::<f32>());
        let max = lg
            .iter()
            .zip(un)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max < 1e-4,
            "unified Euler diverged from legacy FlowMatch: {max:e}"
        );
    }

    #[test]
    fn mlx_drives_every_curated_solver_to_finite_output() {
        // The P2 deliverable: every gen-core curated sampler runs end-to-end over mlx_rs::Array.
        let ops = MlxLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = build_flow_sigmas(6, compute_mu(image_seq_len(512, 512), 6));
        let x_init = arr(&[0.2, -0.5, 1.0, 0.3]);
        for name in [
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "lcm",
            "ddim",
        ] {
            let sampler =
                gen_core::sampling::sampler_by_name::<MlxLatentOps>(name).expect("known solver");
            let mut dn = |x: &Array, s: f32| {
                denoise(&ops, &ms, x, s, |xin, _t| {
                    stub_velocity(xin).map_err(Into::into)
                })
            };
            let out = sampler
                .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, 7)
                .unwrap();
            assert!(
                out.as_slice::<f32>().iter().all(|v| v.is_finite()),
                "{name} produced non-finite output"
            );
        }
    }

    // --- Keystone: run_curated_sampler drives ε/DDPM (Discrete) + v-pred (EDM), not just FLOW --------

    use gen_core::sampling::{DiscreteModelSampling, EdmModelSampling};

    /// `run_curated_sampler` over a `DiscreteModelSampling` (ε prediction) with `euler` reproduces the
    /// legacy Kolors/diffusers Euler step `x + ε·(σ_{i+1} − σ_i)` EXACTLY: for a constant ε field the
    /// rectified integral is `x_init − ε·σ_0`. This is the keystone equivalence the DDPM cohort's curated
    /// path relies on (the `to_d` round-trip cancels: `d = (x − (x − σ·ε))/σ = ε`).
    #[test]
    fn run_curated_sampler_eps_euler_matches_legacy_discrete_step() {
        let sched = AlphaSchedule::scaled_linear(1000, 0.00085, 0.012);
        let ms = DiscreteModelSampling::sdxl(&sched);
        // A descending σ schedule (length steps+1, trailing 0) — not the native table, any valid ramp.
        let sigmas = vec![8.0_f32, 4.0, 2.0, 1.0, 0.5, 0.0];
        let x_init = arr(&[0.3, -1.1, 2.0]);
        // A constant-ε "model" (ignores the scaled input) — the linear field a consistent solver hits
        // exactly. eps must be returned in the EPS sense (the closure yields the raw model output).
        let eps = [0.7_f32, -0.2, 0.4];
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        let out = run_curated_sampler(
            Some("euler"),
            &ms,
            &sigmas,
            x_init.clone(),
            0,
            &cancel,
            &mut progress,
            |_xin, _t| Ok(arr(&eps)),
        )
        .unwrap();
        let got = out.as_slice::<f32>();
        let xi = x_init.as_slice::<f32>();
        for ((g, &x0), &e) in got.iter().zip(xi).zip(&eps) {
            let want = x0 - e * sigmas[0]; // x_init − ε·σ_0
            assert!((g - want).abs() < 2e-3, "eps euler: got {g} want {want}");
        }
    }

    #[test]
    fn raw_latent_hook_observes_only_heun_outer_step_sigmas() {
        let sched = AlphaSchedule::scaled_linear(1000, 0.00085, 0.012);
        let ms = DiscreteModelSampling::sdxl(&sched);
        let sigmas = vec![8.0_f32, 4.0, 2.0, 1.0, 0.0];
        let mut observed = Vec::new();
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        run_curated_sampler_with_latent_hook(
            Some("heun"),
            &ms,
            &sigmas,
            arr(&[0.3, -1.1, 2.0]),
            0,
            &cancel,
            &mut progress,
            |_latent, sigma| observed.push(sigma),
            |_xin, _t| Ok(arr(&[0.7, -0.2, 0.4])),
        )
        .unwrap();
        assert_eq!(observed, sigmas[..4]);
    }

    #[test]
    fn raw_latent_hook_suppresses_dpmpp_sde_midpoint_evaluations() {
        let sched = AlphaSchedule::scaled_linear(1000, 0.00085, 0.012);
        let ms = DiscreteModelSampling::sdxl(&sched);
        let sigmas = vec![8.0_f32, 4.0, 2.0, 1.0, 0.0];
        let mut observed = Vec::new();
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        run_curated_sampler_with_latent_hook(
            Some("dpmpp_sde"),
            &ms,
            &sigmas,
            arr(&[0.3, -1.1, 2.0]),
            7,
            &cancel,
            &mut progress,
            |_latent, sigma| observed.push(sigma),
            |_xin, _t| Ok(arr(&[0.7, -0.2, 0.4])),
        )
        .unwrap();
        assert_eq!(observed, sigmas[..4]);
    }

    #[test]
    fn curated_legacy_wrapper_matches_explicit_inert_hook_exactly() {
        let sched = AlphaSchedule::scaled_linear(1000, 0.00085, 0.012);
        let ms = DiscreteModelSampling::sdxl(&sched);
        let sigmas = vec![8.0_f32, 4.0, 2.0, 1.0, 0.0];
        let init = arr(&[0.3, -1.1, 2.0]);
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        let legacy = run_curated_sampler(
            Some("euler"),
            &ms,
            &sigmas,
            init.clone(),
            42,
            &cancel,
            &mut progress,
            |_xin, _t| Ok(arr(&[0.7, -0.2, 0.4])),
        )
        .unwrap();
        let mut progress = |_p: Progress| {};
        let explicit = run_curated_sampler_with_latent_hook(
            Some("euler"),
            &ms,
            &sigmas,
            init,
            42,
            &cancel,
            &mut progress,
            |_, _| {},
            |_xin, _t| Ok(arr(&[0.7, -0.2, 0.4])),
        )
        .unwrap();
        assert_eq!(legacy.as_slice::<f32>(), explicit.as_slice::<f32>());
    }

    #[test]
    fn flow_hook_observes_outer_steps_and_inert_wrapper_is_exact() {
        let sigmas = vec![1.0_f32, 0.75, 0.5, 0.25, 0.0];
        let init = arr(&[0.3, -1.1, 2.0]);
        let cancel = CancelFlag::new();

        for solver in ["heun", "dpmpp_sde"] {
            let mut observed = Vec::new();
            let mut progress = |_p: Progress| {};
            run_flow_sampler_with_latent_hook(
                Some(solver),
                TimestepConvention::Sigma,
                &sigmas,
                init.clone(),
                7,
                &cancel,
                &mut progress,
                |_latent, sigma| observed.push(sigma),
                |_xin, _t| Ok(arr(&[0.7, -0.2, 0.4])),
            )
            .unwrap();
            assert_eq!(observed, sigmas[..4], "{solver} leaked an inner evaluation");
        }

        let mut progress = |_p: Progress| {};
        let legacy = run_flow_sampler(
            Some("euler"),
            TimestepConvention::Sigma,
            &sigmas,
            init.clone(),
            42,
            &cancel,
            &mut progress,
            |_xin, _t| Ok(arr(&[0.7, -0.2, 0.4])),
        )
        .unwrap();
        let mut progress = |_p: Progress| {};
        let explicit = run_flow_sampler_with_latent_hook(
            Some("euler"),
            TimestepConvention::Sigma,
            &sigmas,
            init,
            42,
            &cancel,
            &mut progress,
            |_, _| {},
            |_xin, _t| Ok(arr(&[0.7, -0.2, 0.4])),
        )
        .unwrap();
        assert_eq!(legacy.as_slice::<f32>(), explicit.as_slice::<f32>());
    }

    #[test]
    fn cfgpp_legacy_wrapper_matches_explicit_inert_hook_exactly() {
        let sched = AlphaSchedule::scaled_linear(1000, 0.00085, 0.012);
        let ms = DiscreteModelSampling::sdxl(&sched);
        let sigmas = vec![8.0_f32, 4.0, 2.0, 1.0, 0.0];
        let init = arr(&[0.3, -1.1, 2.0]);
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        let legacy = run_cfgpp_sampler(
            Some("euler"),
            &ms,
            &sigmas,
            init.clone(),
            &cancel,
            &mut progress,
            |_xin, _t| Ok((arr(&[0.7, -0.2, 0.4]), arr(&[0.1, 0.2, -0.1]))),
        )
        .unwrap();
        let mut progress = |_p: Progress| {};
        let explicit = run_cfgpp_sampler_with_latent_hook(
            Some("euler"),
            &ms,
            &sigmas,
            init,
            &cancel,
            &mut progress,
            |_, _| {},
            |_xin, _t| Ok((arr(&[0.7, -0.2, 0.4]), arr(&[0.1, 0.2, -0.1]))),
        )
        .unwrap();
        assert_eq!(legacy.as_slice::<f32>(), explicit.as_slice::<f32>());
    }

    /// `run_av_curated_sampler` drives LTX's joint two-stream (video+audio) FLOW denoise: the curated
    /// Euler over `MlxAvLatentOps` integrates BOTH streams, and for a constant per-stream velocity it
    /// lands exactly on `x_init − v·σ_0` per stream (the rectified-flow integral) — proving the
    /// two-stream `LatentOps` + driver reproduce the legacy per-stream `euler_step`.
    #[test]
    fn run_av_curated_sampler_euler_matches_legacy_per_stream() {
        let sigmas = vec![1.0_f32, 0.75, 0.5, 0.25, 0.0];
        let v_video = [0.7_f32, -0.2, 0.4];
        let v_audio = [0.1_f32, 0.5];
        let init = AvLatents {
            video: arr(&[0.3, -1.1, 2.0]),
            audio: arr(&[0.05, -0.4]),
        };
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        let out = run_av_curated_sampler(
            Some("euler"),
            &sigmas,
            init,
            0,
            &cancel,
            &mut progress,
            |_x, _sigma| {
                Ok(AvLatents {
                    video: arr(&v_video),
                    audio: arr(&v_audio),
                })
            },
        )
        .unwrap();
        let (gv, ga) = (out.video.as_slice::<f32>(), out.audio.as_slice::<f32>());
        for ((g, &x0), &v) in gv.iter().zip(&[0.3_f32, -1.1, 2.0]).zip(&v_video) {
            assert!((g - (x0 - v * sigmas[0])).abs() < 2e-3, "video: got {g}");
        }
        for ((g, &x0), &v) in ga.iter().zip(&[0.05_f32, -0.4]).zip(&v_audio) {
            assert!((g - (x0 - v * sigmas[0])).abs() < 2e-3, "audio: got {g}");
        }
    }

    /// Every curated solver drives the two-stream AV latents to finite output (the stochastic ones too).
    #[test]
    fn run_av_curated_sampler_every_solver_is_finite() {
        let sigmas = build_flow_sigmas(6, compute_mu(image_seq_len(512, 512), 6));
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        for name in [
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "ddim",
        ] {
            let init = AvLatents {
                video: arr(&[0.2, -0.5, 1.0, 0.3]),
                audio: arr(&[0.1, -0.2]),
            };
            let out = run_av_curated_sampler(
                Some(name),
                &sigmas,
                init,
                7,
                &cancel,
                &mut progress,
                |x, _s| {
                    Ok(AvLatents {
                        video: multiply(&x.video, scalar(0.2))?,
                        audio: multiply(&x.audio, scalar(0.2))?,
                    })
                },
            )
            .unwrap();
            assert!(
                out.video.as_slice::<f32>().iter().all(|v| v.is_finite())
                    && out.audio.as_slice::<f32>().iter().all(|v| v.is_finite()),
                "{name} (AV two-stream) produced non-finite output"
            );
        }
    }

    /// `run_curated_sampler` drives a v-prediction `EdmModelSampling` (SVD's contract) to finite output
    /// over every curated solver — proving the keystone is prediction-type-agnostic (the V `x0`
    /// recombination flows through `MlxLatentOps`).
    #[test]
    fn run_curated_sampler_v_prediction_edm_is_finite_every_solver() {
        let ms = EdmModelSampling::svd();
        // A short EDM-range descending schedule (σ_max≈700 down to 0), any valid ramp.
        let sigmas = vec![80.0_f32, 20.0, 5.0, 1.0, 0.2, 0.0];
        let x_init = arr(&[0.2, -0.5, 1.0, 0.3]);
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        for name in [
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "ddim",
        ] {
            let out = run_curated_sampler(
                Some(name),
                &ms,
                &sigmas,
                x_init.clone(),
                7,
                &cancel,
                &mut progress,
                // A mild v "model": v = 0.1·x_in (the input is c_in-scaled, keeping it bounded).
                |xin, _t| Ok(multiply(xin, scalar(0.1))?),
            )
            .unwrap();
            assert!(
                out.as_slice::<f32>().iter().all(|v| v.is_finite()),
                "{name} (v-pred/EDM) produced non-finite output"
            );
        }
    }
}

/// Real-backend (MLX `Array`) parity for the gen-core guidance library over [`MlxLatentOps`]
/// (sc-7439). Proves the trait impl reproduces the bespoke Lens/Bernini MLX combine and the
/// `apg @ eta=1/nt=0/no-momentum == cfg` invariant on real `Array`, not just `CpuLatentOps`.
#[cfg(test)]
mod guidance_ops_tests {
    use super::*;
    use gen_core::guidance::{cfg, cfg_rescale, normalized_guidance};

    fn t(data: &[f32], shape: &[i32]) -> Array {
        Array::from_slice(data, shape)
    }

    fn max_abs(a: &Array, b: &Array) -> f32 {
        mlx_rs::ops::max(subtract(a, b).unwrap().abs().unwrap(), None)
            .unwrap()
            .item::<f32>()
    }

    /// A [B=1, seq=2, C=3] cond/uncond pair (per-token channel-axis geometry, the Lens case).
    fn pair() -> (Array, Array) {
        let cond = t(&[3.0, 4.0, 0.0, 1.0, 2.0, 2.0], &[1, 2, 3]);
        let uncond = t(&[0.5, -1.0, 0.25, 0.1, 0.3, -0.2], &[1, 2, 3]);
        (cond, uncond)
    }

    #[test]
    fn cfg_matches_hand_combine() {
        let ops = MlxLatentOps;
        let (cond, uncond) = pair();
        let got = cfg(&ops, &cond, &uncond, 4.0).unwrap();
        let want = add(
            &uncond,
            multiply(subtract(&cond, &uncond).unwrap(), scalar(4.0)).unwrap(),
        )
        .unwrap();
        assert!(max_abs(&got, &want) < 1e-4, "cfg over MLX != hand combine");
    }

    /// sc-7443 enabler: with the dtype-preserving `MlxLatentOps::axpy`/`scale`, gen-core `cfg` over a
    /// half-precision (fp16 / bf16) cond/uncond pair reproduces the bespoke engine combine
    /// `uncond + scalar(scale).as_dtype(eps)·(cond − uncond)` (the SDXL/Chroma/Z-Image form) **byte-for-byte
    /// AND keeps the result in the input dtype** — no f32 promotion. This is the N1 proof the unified
    /// combine is a drop-in for the half-precision image cohort. (Without the dtype-preserving change the
    /// f32 `scalar()` would promote the result to f32 and the dtype assert below would fail.)
    #[test]
    fn cfg_half_precision_is_byte_identical_and_preserves_dtype() {
        let ops = MlxLatentOps;
        let (cond_f32, uncond_f32) = pair();
        let cfg_scale = 7.5f32;
        for dt in [Dtype::Float16, Dtype::Bfloat16] {
            let cond = cond_f32.as_dtype(dt).unwrap();
            let uncond = uncond_f32.as_dtype(dt).unwrap();
            // Bespoke engine combine: the scale cast to the eps dtype (SDXL `denoise_core` form).
            let cfg_s = scalar(cfg_scale).as_dtype(dt).unwrap();
            let want = add(
                &uncond,
                multiply(subtract(&cond, &uncond).unwrap(), &cfg_s).unwrap(),
            )
            .unwrap();
            // The unified gen-core path (what sc-7443 routes the cohort through).
            let got = cfg(&ops, &cond, &uncond, cfg_scale).unwrap();
            assert_eq!(
                got.dtype(),
                dt,
                "{dt:?}: cfg must preserve dtype (no f32 promotion)"
            );
            assert_eq!(
                max_abs(&got, &want),
                0.0,
                "{dt:?}: gen-core cfg != bespoke half-precision combine"
            );
        }
    }

    /// gen-core `cfg_rescale` over MLX == the bespoke Lens formula (per-token channel-axis rescale).
    #[test]
    fn cfg_rescale_matches_lens_formula() {
        let ops = MlxLatentOps;
        let (cond, uncond) = pair();
        let shape = [1usize, 2, 3];
        let got = cfg_rescale(&ops, &cond, &uncond, 2.0, &shape, &[-1]).unwrap();
        // Hand-rolled Lens reference (mlx-gen-lens/src/schedule.rs): comb rescaled to ‖cond‖/‖comb‖.
        let comb = add(
            &uncond,
            multiply(subtract(&cond, &uncond).unwrap(), scalar(2.0)).unwrap(),
        )
        .unwrap();
        let norm =
            |x: &Array| sqrt(sum_axes(multiply(x, x).unwrap(), &[-1], true).unwrap()).unwrap();
        let cond_norm = norm(&cond);
        let comb_norm = norm(&comb);
        let ratio = divide(&cond_norm, maximum(&comb_norm, scalar(1e-12)).unwrap()).unwrap();
        // comb_norm > 0 for this pair, so the where-guard is a no-op → want = comb · ratio.
        let want = multiply(&comb, &ratio).unwrap();
        assert!(
            max_abs(&got, &want) < 1e-4,
            "cfg_rescale over MLX != Lens formula"
        );
    }

    /// The Bernini invariant on real `Array`: APG at eta=1, nt=0, no momentum == plain CFG.
    #[test]
    fn apg_reduces_to_cfg_on_mlx() {
        let ops = MlxLatentOps;
        let (cond, uncond) = pair();
        let shape = [1usize, 2, 3];
        let got =
            normalized_guidance(&ops, &cond, &uncond, 4.0, None, 1.0, 0.0, &shape, &[-1]).unwrap();
        let want = cfg(&ops, &cond, &uncond, 4.0).unwrap();
        assert!(
            max_abs(&got, &want) < 1e-4,
            "apg(eta=1,nt=0) over MLX != cfg"
        );
    }

    /// eta=0 ⇒ the APG delta is orthogonal to the conditional base: (nd · cond) per token ≈ 0.
    #[test]
    fn apg_eta0_is_orthogonal_on_mlx() {
        let ops = MlxLatentOps;
        let (cond, uncond) = pair();
        let shape = [1usize, 2, 3];
        // scale=1 so nd is recovered as (got − uncond); eta=0, nt=0.
        let got =
            normalized_guidance(&ops, &cond, &uncond, 1.0, None, 0.0, 0.0, &shape, &[-1]).unwrap();
        let nd = subtract(&got, &uncond).unwrap();
        let dot = sum_axes(multiply(&nd, &cond).unwrap(), &[-1], true).unwrap();
        let zeros = Array::zeros::<f32>(dot.shape()).unwrap();
        assert!(
            max_abs(&dot, &zeros) < 1e-3,
            "eta=0 not orthogonal to cond on MLX"
        );
    }
}
