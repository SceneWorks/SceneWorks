//! Flow-match Euler discrete scheduler — the sampler shared by the mlx-gen DiT families
//! (Z-Image, FLUX, Qwen). Port of the Python mflux fork's `FlowMatchEulerDiscreteScheduler`
//! (`models/common/schedulers/flow_match_euler_discrete_scheduler.py`).
//!
//! The schedule is a `linspace(1, 1/n, n)` run through an exponential **time-shift**, with a
//! trailing `0` appended to mark the final step. The shift's `mu` comes from one of two sources:
//! [`FlowMatchEuler::for_image`] fits it empirically from the latent sequence length (the fork's
//! `requires_sigma_shift` path, used by FLUX / Qwen / the full Z-Image model), while
//! [`FlowMatchEuler::for_static_shift`] uses a fixed `shift` pinned by a model's `scheduler_config.json`
//! (e.g. Z-Image-Turbo's `shift=3.0`) — `exp(mu) = shift`, equivalent to diffusers'
//! `use_dynamic_shifting=false`. Each denoise step is the Euler update
//! `x_{t+1} = x_t + (sigma[t+1] - sigma[t]) * v`, where `v` is the model's (already sign-flipped)
//! velocity prediction.

use mlx_rs::ops::{add, multiply};
use mlx_rs::Array;

use crate::array::scalar;
use crate::{Error, Result};

// Schedule construction (sigma tables, empirical mu) is backend-neutral policy and lives in gen-core
// (sc-3722); re-exported here at the historical `mlx_gen::scheduler::{image_seq_len, compute_mu}`
// paths (used by `tests/scheduler.rs`). Only the Euler tensor application stays in this module.
pub use gen_core::sampling::{compute_mu, image_seq_len};

/// The flow-match (rectified-flow) forward-Euler update on the velocity field:
/// `x_{i+1} = x + velocity·(σ_{i+1} − σ_i)`, with `dt = σ_{i+1} − σ_i` (negative; the schedule
/// descends to 0). This is the single numerically load-bearing line of flow-match denoising — shared
/// by [`FlowMatchEuler::step`] and [`crate::sampler::FlowMatchSampler::step`] so a fix to one can't
/// silently miss the other (F-009). Computed in the latents' dtype (no upcast), exactly matching the
/// fork's `LinearScheduler.step` / `FlowMatchEulerDiscreteScheduler.step`.
pub(crate) fn flow_match_euler_step(
    sigmas: &[f32],
    x: &Array,
    velocity: &Array,
    i: usize,
) -> Result<Array> {
    // Callers drive `i` in-contract, but an off-by-one in any consumer would index `sigmas[i+1]` out
    // of bounds and panic the denoise loop. Surface it as a typed error instead (F-042).
    if i + 1 >= sigmas.len() {
        return Err(Error::Msg(format!(
            "flow_match_euler_step: step index {i} out of range for {} sigmas",
            sigmas.len()
        )));
    }
    let dt = sigmas[i + 1] - sigmas[i];
    Ok(add(x, &multiply(velocity, scalar(dt))?)?)
}

/// A flow-match Euler denoising schedule.
pub struct FlowMatchEuler {
    /// Denoising sigmas, length `num_steps + 1` (the trailing `0.0` marks the final step).
    pub sigmas: Vec<f32>,
}

impl FlowMatchEuler {
    /// Build the schedule for `num_steps` with an explicit time-shift `mu`.
    pub fn new(num_steps: usize, mu: f32) -> Self {
        Self {
            sigmas: gen_core::sampling::build_flow_sigmas(num_steps, mu),
        }
    }

    /// Build the schedule for an image of `width`×`height`, computing the resolution-dependent
    /// `mu` from the latent sequence length (the fork's `requires_sigma_shift` path).
    pub fn for_image(num_steps: usize, width: u32, height: u32) -> Self {
        let seq_len = image_seq_len(width, height);
        Self::new(num_steps, compute_mu(seq_len, num_steps))
    }

    /// Build the schedule for a **static** time-shift `shift` (resolution- and step-independent),
    /// matching diffusers' `FlowMatchEulerDiscreteScheduler` with `use_dynamic_shifting=false`:
    /// `sigma' = shift·t / (1 + (shift-1)·t)`. The exponential time-shift used here equals that
    /// algebraic form when `exp(mu) = shift`, so this is just `new(num_steps, ln(shift))`.
    ///
    /// Used by models whose published `scheduler_config.json` pins a fixed `shift` (e.g.
    /// Z-Image-Turbo's `shift=3.0`) rather than the empirical per-resolution `mu` of
    /// [`Self::for_image`].
    pub fn for_static_shift(num_steps: usize, shift: f32) -> Self {
        Self::new(num_steps, shift.ln())
    }

    /// Wrap an already-built descending sigma schedule (length `num_steps + 1`, trailing `0.0`). Used
    /// by the epic 7114 scheduler axis ([`crate::resolve_flow_schedule`]): an engine resolves a curated
    /// `req.scheduler` into a sigma vector and wraps it here to drive the same denoise loop, with the
    /// native schedule (the `None`/default path) returned byte-for-byte.
    pub fn from_sigmas(sigmas: Vec<f32>) -> Result<Self> {
        // F-063: a schedule needs at least a start sigma + the trailing 0.0 (len >= 2); otherwise
        // `num_steps()` underflows (`len - 1` on an empty vec) and the denoise loop indexes OOB.
        // `new`/`for_image`/`for_static_shift` can't hit this (`build_flow_sigmas` clamps num_steps to
        // >= 1 and always appends 0.0), so only this request-driven entry point needs the guard.
        if sigmas.len() < 2 {
            return Err(Error::Msg(format!(
                "FlowMatchEuler::from_sigmas: need >= 2 sigmas (start + trailing 0.0), got {}",
                sigmas.len()
            )));
        }
        Ok(Self { sigmas })
    }

    /// Number of denoising steps (loop iterations). `saturating_sub` so a (guarded-against but
    /// defensively handled) empty schedule reports 0 rather than underflowing (F-063).
    pub fn num_steps(&self) -> usize {
        self.sigmas.len().saturating_sub(1)
    }

    /// The transformer timestep at step `t`: `1 - sigma[t]` (in `[0, 1]`; the model applies its
    /// own `t_scale`). Returns `1.0` for an out-of-range `t` rather than panicking (F-063).
    pub fn timestep(&self, t: usize) -> f32 {
        1.0 - self.sigmas.get(t).copied().unwrap_or(0.0)
    }

    /// One Euler step: `x_{t+1} = x_t + (sigma[t+1] - sigma[t]) * velocity`. Delegates to the shared
    /// `flow_match_euler_step` (the same update [`crate::sampler::FlowMatchSampler`] uses).
    pub fn step(&self, latents: &Array, velocity: &Array, t: usize) -> Result<Array> {
        flow_match_euler_step(&self.sigmas, latents, velocity, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_shape_and_endpoints() {
        let s = FlowMatchEuler::for_image(4, 1024, 1024);
        assert_eq!(s.sigmas.len(), 5); // num_steps + 1
        assert_eq!(s.num_steps(), 4);
        assert_eq!(*s.sigmas.last().unwrap(), 0.0);
        // sigmas strictly decreasing.
        assert!(s.sigmas.windows(2).all(|w| w[0] > w[1]));
        // timestep is 1 - sigma.
        assert!((s.timestep(0) - (1.0 - s.sigmas[0])).abs() < 1e-6);
    }

    #[test]
    fn seq_len_matches_definition() {
        assert_eq!(image_seq_len(1024, 1024), 4096);
        assert_eq!(image_seq_len(256, 256), 256);
        assert_eq!(image_seq_len(1280, 1280), 6400);
    }

    #[test]
    fn static_shift_matches_diffusers() {
        // diffusers FlowMatchEulerDiscreteScheduler with use_dynamic_shifting=false, shift=3.0:
        // sigma' = 3·t/(1+2·t) for t = linspace(1, 1/n, n); n=4 -> [1, 0.9, 0.75, 0.5, 0].
        let s = FlowMatchEuler::for_static_shift(4, 3.0);
        let expected = [1.0_f32, 0.9, 0.75, 0.5, 0.0];
        assert_eq!(s.sigmas.len(), 5);
        for (got, want) in s.sigmas.iter().zip(expected) {
            assert!(
                (got - want).abs() < 1e-5,
                "static shift: got {got} want {want}"
            );
        }
    }

    #[test]
    fn mu_large_seq_branch() {
        // > 4300 uses the linear-in-seq_len branch (independent of num_steps).
        let a = compute_mu(6400, 4);
        let b = compute_mu(6400, 8);
        assert!((a - b).abs() < 1e-6);
    }

    /// F-009: `FlowMatchEuler::step` and `FlowMatchSampler::step` now share one update, so they must
    /// produce byte-identical results for the same sigmas, latents, and velocity at every step.
    #[test]
    fn scheduler_and_sampler_steps_are_identical() {
        use crate::sampler::{DiffusionSampler, FlowMatchSampler};
        use mlx_rs::ops::eq;

        let sigmas = vec![1.0_f32, 0.9, 0.75, 0.5, 0.0];
        let euler = FlowMatchEuler {
            sigmas: sigmas.clone(),
        };
        let sampler = FlowMatchSampler::new(sigmas.clone());
        let x = Array::from_slice(&[0.1_f32, -0.2, 0.3, 0.4], &[1, 4]);
        let v = Array::from_slice(&[0.5_f32, 0.6, -0.7, 0.8], &[1, 4]);
        for i in 0..euler.num_steps() {
            let a = euler.step(&x, &v, i).unwrap();
            let b = sampler.step(&v, &x, i).unwrap();
            assert!(
                eq(&a, &b).unwrap().all(None).unwrap().item::<bool>(),
                "step {i}: scheduler and sampler diverged"
            );
        }
    }
}
