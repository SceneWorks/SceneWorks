//! sc-3048 — **Rose** + **Prodigy** optimizers for LoRA/LoKr training, Rust ports of the two
//! optimizers the SceneWorks torch trainers' picker (`build_optimizer`) exposes beyond what mlx-rs
//! ships, plus a [`TrainOptimizer`] enum that unifies them with mlx-rs [`AdamW`] behind one
//! whole-map [`step`](TrainOptimizer::step). This keeps the optimizer picker honest across the
//! epic-3039 cutover: a job that asked for `prodigy`/`rose` trains with that optimizer on MLX,
//! rather than silently collapsing to AdamW-only.
//!
//! mlx-rs ships Adam/AdamW/SGD/Lion/RMSProp/AdaGrad/Delta/AdaMax/AdaFactor but NOT these two:
//!   * **Rose** (rose-opt) — *stateless* per-slice range-of-slice normalization (no momentum /
//!     variance / step buffers): `θ -= lr · g / (|max(g)| − min(g))` reduced over all axes beyond
//!     the leading one, with gradient centralization + a coefficient-of-variation trust gate, and
//!     decoupled weight decay. A faithful port of `rose_opt.Rose` at the SceneWorks call's defaults
//!     (`compute_dtype="fp32"`, `centralize=stabilize=True`; bf16 stochastic rounding is a no-op on
//!     our f32 factors).
//!   * **Prodigy** (prodigyopt) — a learning-rate-free Adam variant whose adapted step `d` is
//!     estimated from a running numerator/denominator that **couple every parameter** through a
//!     global two-pass step. That global coupling is why the optimizer interface here is a whole-map
//!     `step(params, grads)`, not mlx-rs's per-parameter `update_single`. Ported at the picker's
//!     args (`lr = lr≥0.1 ? lr : 1.0`, `eps = 1e-6`, betas `(0.9, 0.999)`, `beta3 = √beta2`,
//!     decoupled weight decay, `d0 = 1e-6`, `slice_p = 1`).
//!
//! All trainable LoRA/LoKr factors are f32 (and 2-D), so the updates compute in f32; the per-family
//! trainers keep their own gradient accumulation + `clip_grad_norm` + LR schedule and call `step`.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_rs::ops::{
    add, divide, max_axes, mean_axes, min_axes, multiply, r#where, sign, subtract, sum_axes,
};
use mlx_rs::optimizers::{AdamW, Optimizer, OptimizerState};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use crate::train::lora::LoraParams;
use crate::Result;

/// A 1-element f32 array (broadcasts as a scalar in elementwise ops).
fn c(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Full reduction to a lazy scalar `[1]` array, WITHOUT a blocking `.item()` host sync — many of
/// these accumulate into one graph and sync once per step (F-066).
fn sum_all_lazy(x: &Array) -> Result<Array> {
    let axes: Vec<i32> = (0..x.ndim() as i32).collect();
    Ok(sum_axes(x, &axes[..], false)?.reshape(&[1])?)
}

/// Replace exact zeros with `1.0` (the reference `masked_fill_(x == 0, 1.0)` SGD/identity fallback).
fn zeros_to_one(x: &Array) -> Result<Array> {
    Ok(r#where(&x.eq(c(0.0))?, c(1.0), x)?)
}

/// Normalize an optimizer name the SceneWorks `build_optimizer` way (lowercase, strip `-`/`_`),
/// collapsing the aliases: `prodigyopt`→`prodigy`, `roseopt`→`rose`, and `adamw8bit`/`adam8bit`→
/// `adamw` (the 8-bit optimizer is a CUDA-only bitsandbytes dependency — the torch picker itself
/// falls back to AdamW when it is unavailable, which is always on Apple Silicon).
fn normalize(name: &str) -> String {
    let n: String = name
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| *c != '-' && *c != '_')
        .collect();
    match n.as_str() {
        "prodigyopt" => "prodigy".to_string(),
        "roseopt" => "rose".to_string(),
        "adamw8bit" | "adam8bit" => "adamw".to_string(),
        other => other.to_string(),
    }
}

/// The optimizers MLX training supports (after alias normalization).
pub const SUPPORTED_OPTIMIZERS: [&str; 4] = ["adamw", "adam", "rose", "prodigy"];

/// Unified training optimizer: mlx-rs [`AdamW`] (also serves `adam`, with weight-decay 0), or the
/// ported [`Rose`] / [`Prodigy`]. Each family trainer builds one with [`from_config`](Self::from_config),
/// scales the LR per step via [`set_lr_scaled`](Self::set_lr_scaled), and applies the update with
/// [`step`](Self::step) over its (clipped, accumulated) gradient map.
pub enum TrainOptimizer {
    /// mlx-rs AdamW (and `adam` ≡ AdamW with weight-decay 0). `base_lr` is the configured LR the
    /// schedule multiplier scales.
    AdamW {
        opt: AdamW,
        base_lr: f32,
    },
    Rose(Rose),
    Prodigy(Prodigy),
}

impl TrainOptimizer {
    /// Whether `name` (after alias normalization) is a supported MLX-training optimizer. Trainers
    /// call this in `validate` to reject an unsupported choice up front.
    pub fn is_supported(name: &str) -> bool {
        SUPPORTED_OPTIMIZERS.contains(&normalize(name).as_str())
    }

    /// Build the optimizer for `name`. `lr` is the configured learning rate; `weight_decay` is the
    /// already-resolved decay (the trainers pass 0 for `adam`). Errors on an unsupported name.
    pub fn from_config(name: &str, lr: f32, weight_decay: f32) -> Result<Self> {
        match normalize(name).as_str() {
            "adamw" | "adam" => {
                let mut opt = AdamW::new(lr);
                opt.weight_decay = c(weight_decay);
                Ok(TrainOptimizer::AdamW { opt, base_lr: lr })
            }
            "rose" => Ok(TrainOptimizer::Rose(Rose::new(lr, weight_decay))),
            "prodigy" => Ok(TrainOptimizer::Prodigy(Prodigy::new(lr, weight_decay))),
            other => Err(format!(
                "optimizer '{other}' is not available on MLX training (supported: {})",
                SUPPORTED_OPTIMIZERS.join(", ")
            )
            .into()),
        }
    }

    /// Scale the base learning rate by the schedule multiplier `mult` (constant/linear/cosine +
    /// warmup, computed once per optimizer update by the trainer). For AdamW/Rose this is the
    /// effective LR; for Prodigy it scales the LR knob that multiplies the adapted `d` step.
    pub fn set_lr_scaled(&mut self, mult: f32) {
        match self {
            TrainOptimizer::AdamW { opt, base_lr } => opt.lr = c(*base_lr * mult),
            TrainOptimizer::Rose(r) => r.lr = r.base_lr * mult,
            TrainOptimizer::Prodigy(p) => p.lr = p.base_lr * mult,
        }
    }

    /// Apply one optimizer step over the whole factor map, in place. AdamW/Rose update each factor
    /// independently; Prodigy runs its global two-pass `d`-adaptation across all factors. `grads` is
    /// the (clipped, accumulation-averaged) gradient map keyed identically to `params`.
    pub fn step(&mut self, params: &mut LoraParams, grads: &LoraParams) -> Result<()> {
        match self {
            TrainOptimizer::AdamW { opt, .. } => {
                for (k, g) in grads.iter() {
                    if let Some(p) = params.get(k) {
                        let mut p = p.clone();
                        opt.update_single(k, g, &mut p)?;
                        params.insert(k.clone(), p);
                    }
                }
                Ok(())
            }
            TrainOptimizer::Rose(r) => r.step(params, grads),
            TrainOptimizer::Prodigy(p) => p.step(params, grads),
        }
    }

    /// A short tag naming which optimizer this is, written into the resume metadata so a restore can
    /// verify it is loading state saved by the *same* optimizer kind (sc-9560 / F-125).
    pub fn kind_tag(&self) -> &'static str {
        match self {
            TrainOptimizer::AdamW { .. } => "adamw",
            TrainOptimizer::Rose(_) => "rose",
            TrainOptimizer::Prodigy(_) => "prodigy",
        }
    }

    /// Snapshot the optimizer's internal state to a safetensors file at `path`, for mid-schedule
    /// resume (sc-9560 / F-125). Dispatched **per optimizer** — the generic mlx-rs `OptimizerState`
    /// dump the old `save_optimizer_state` used could not capture Prodigy's global adapted-step
    /// scalars + per-parameter buffers, and is meaningless for the stateless Rose:
    ///   * **AdamW** — mlx-rs's own per-parameter moment state (`opt.state()`).
    ///   * **Rose** — stateless (no buffers); a marker keeps the resume bundle a uniform, valid file.
    ///   * **Prodigy** — the per-parameter Adam EMAs + `s` + `p0` and the global `d` / `d_max` /
    ///     `d_numerator` (the learning-rate-free adaptation state).
    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<()> {
        match self {
            TrainOptimizer::AdamW { opt, .. } => opt.state().save_safetensors(path)?,
            TrainOptimizer::Rose(_) => {
                // Nothing to persist; write a marker so resume sees a valid, uniform snapshot file.
                let marker = c(0.0);
                Array::save_safetensors(
                    vec![("__stateless".to_string(), &marker)],
                    None::<&HashMap<String, String>>,
                    path,
                )?;
            }
            TrainOptimizer::Prodigy(p) => p.save_state(path)?,
        }
        Ok(())
    }

    /// Restore optimizer state written by [`save_state`](Self::save_state) into this optimizer (built
    /// fresh from the same config). The caller is responsible for restoring into the same optimizer
    /// kind (checked via [`kind_tag`](Self::kind_tag) in the resume metadata).
    pub fn load_state(&mut self, path: impl AsRef<Path>) -> Result<()> {
        match self {
            TrainOptimizer::AdamW { opt, .. } => opt.state_mut().load_safetensors(path)?,
            TrainOptimizer::Rose(_) => {} // stateless — nothing to restore
            TrainOptimizer::Prodigy(p) => p.load_state(path)?,
        }
        Ok(())
    }
}

/// Stateless Range-Of-Slice Equilibration optimizer (rose-opt). Maintains no per-parameter buffers;
/// the only mutable state is the (schedule-scaled) learning rate.
pub struct Rose {
    base_lr: f32,
    lr: f32,
    weight_decay: f32,
    centralize: bool,
    stabilize: bool,
}

impl Rose {
    /// At the SceneWorks call's defaults: `centralize = stabilize = true`, decoupled weight decay
    /// (no schedule coupling), f32 compute.
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            base_lr: lr,
            lr,
            weight_decay,
            centralize: true,
            stabilize: true,
        }
    }

    fn step(&self, params: &mut LoraParams, grads: &LoraParams) -> Result<()> {
        for (k, g) in grads.iter() {
            if let Some(p) = params.get(k) {
                let updated = self.update_one(p, g)?;
                params.insert(k.clone(), updated);
            }
        }
        Ok(())
    }

    /// One Rose update for a single parameter — a faithful port of `Rose.step`'s per-`p` body
    /// (decoupled weight decay, then signSGD for 0-D / range-normalization for 1-D / per-leading-
    /// slice range with optional centralization + trust gating for ≥2-D).
    fn update_one(&self, param: &Array, grad: &Array) -> Result<Array> {
        let lr = self.lr;
        // Decoupled multiplicative weight decay: θ *= max(0, 1 − lr·wd).
        let mut param = if self.weight_decay != 0.0 {
            multiply(param, c((1.0 - lr * self.weight_decay).max(0.0)))?
        } else {
            param.clone()
        };
        let neg_lr = c(-lr);
        match grad.ndim() {
            0 => {
                // signSGD for scalars.
                param = add(&param, &multiply(&sign(grad)?, &neg_lr)?)?;
            }
            1 => {
                let g_max = max_axes(grad, &[0][..], false)?;
                let g_min = min_axes(grad, &[0][..], false)?;
                let denom = zeros_to_one(&subtract(&g_max.abs()?, &g_min)?)?;
                param = add(&param, &multiply(&divide(grad, &denom)?, &neg_lr)?)?;
            }
            ndim => {
                // Active axes = every axis except the leading one (per-slice scaling).
                let active: Vec<i32> = (1..ndim as i32).collect();
                let mut g = grad.clone();
                if self.centralize {
                    let mean = mean_axes(&g, &active[..], true)?;
                    g = subtract(&g, &mean)?;
                }
                // Per-slice range R = |max(g)| − min(g), reducing the active axes.
                let raw_scale = subtract(
                    &max_axes(&g, &active[..], true)?.abs()?,
                    &min_axes(&g, &active[..], true)?,
                )?;
                let denom = if self.stabilize {
                    // Coefficient-of-variation trust gate (population std/mean over the range tensor).
                    let mean = raw_scale.mean(None)?;
                    let var = subtract(&raw_scale, &mean)?.square()?.mean(None)?;
                    let std = var.sqrt()?;
                    // trust = mean / (std + mean), with (std+mean) forced to 1 when mean == 0.
                    let trust = divide(&mean, &zeros_to_one(&add(&std, &mean)?)?)?;
                    // denom = mean.lerp(raw_scale, trust) = mean + trust·(raw_scale − mean).
                    add(&mean, &multiply(&trust, &subtract(&raw_scale, &mean)?)?)?
                } else {
                    raw_scale
                };
                let denom = zeros_to_one(&denom)?;
                param = add(&param, &multiply(&divide(&g, &denom)?, &neg_lr)?)?;
            }
        }
        Ok(param)
    }
}

/// Per-parameter Prodigy state: the Adam EMAs, the `s` accumulator, and the initial parameter `p0`
/// (the `slice_p = 1` case keeps these at the full parameter shape).
struct ProdigyState {
    exp_avg: Array,
    exp_avg_sq: Array,
    s: Array,
    p0: Array,
}

/// Prodigy (prodigyopt): Adam with a learning-rate-free, globally-adapted step size `d`.
pub struct Prodigy {
    base_lr: f32,
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    beta3: f32,
    eps: f32,
    d: f32,
    d0: f32,
    d_max: f32,
    d_numerator: f32,
    d_coef: f32,
    growth_rate: f32,
    state: HashMap<Rc<str>, ProdigyState>,
}

impl Prodigy {
    /// At the picker's args: `lr = lr≥0.1 ? lr : 1.0` (LoRA LRs ≪ 0.1 ⇒ the LR knob is 1.0, the
    /// Prodigy convention), `eps = 1e-6`, betas `(0.9, 0.999)`, `beta3 = √beta2`, decoupled weight
    /// decay, `d0 = 1e-6`, `d_coef = 1`, `growth_rate = ∞`, no bias correction, no safeguard warmup.
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        let use_lr = if lr >= 0.1 { lr } else { 1.0 };
        let beta2 = 0.999;
        Self {
            base_lr: use_lr,
            lr: use_lr,
            weight_decay,
            beta1: 0.9,
            beta2,
            beta3: beta2.sqrt(),
            eps: 1e-6,
            d: 1e-6,
            d0: 1e-6,
            d_max: 1e-6,
            d_numerator: 0.0,
            d_coef: 1.0,
            growth_rate: f32::INFINITY,
            state: HashMap::new(),
        }
    }

    /// One Prodigy step — a faithful port of `Prodigy.step` (`slice_p = 1`, `beta1 > 0`, decoupled
    /// weight decay, no bias correction / safeguard warmup). Pass 1 updates the per-parameter EMAs +
    /// `s` and accumulates the global numerator/denominator; the adapted `d` is then re-estimated;
    /// pass 2 applies the Adam step at the *old* `dlr` with the *new* `d` in the denominator.
    fn step(&mut self, params: &mut LoraParams, grads: &LoraParams) -> Result<()> {
        let (beta1, beta2, beta3) = (self.beta1, self.beta2, self.beta3);
        let (d, d0, lr, eps) = (self.d, self.d0, self.lr, self.eps);
        let dlr = d * lr; // bias_correction = 1
        let d_numerator = self.d_numerator * beta3;

        // --- Pass 1: EMAs + s; accumulate the global numerator/denominator ---
        // The two per-factor reductions (⟨g, p0−p⟩ and Σ|s|) accumulate LAZILY into scalar `[1]`
        // arrays and are synced to host ONCE after the loop (F-066), rather than a blocking `.item()`
        // per factor per step (hundreds of round-trips). f32 add is IEEE-deterministic and the loop
        // order is unchanged, so the folded sums are bit-identical to the prior per-term host `+=`.
        let mut delta_numerator = c(0.0);
        let mut d_denom = c(0.0);
        for (k, g) in grads.iter() {
            let Some(p) = params.get(k) else { continue };
            // `Array::zeros` is fallible, so build the initial state with `?` rather than `.unwrap()`
            // in an `or_insert_with` closure — an allocation failure mid-training surfaces as a typed
            // `Error`, not a process abort (F-008).
            let st = match self.state.entry(k.clone()) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => e.insert(ProdigyState {
                    exp_avg: Array::zeros::<f32>(p.shape())?,
                    exp_avg_sq: Array::zeros::<f32>(p.shape())?,
                    s: Array::zeros::<f32>(p.shape())?,
                    p0: p.clone(),
                }),
            };
            // delta_numerator += (d/d0)·dlr·⟨g, p0 − p⟩  (coefficient folded exactly as the prior
            // `(d / d0) * dlr * dot`, i.e. `((d/d0)*dlr) · dot`).
            let dot = sum_all_lazy(&multiply(g, &subtract(&st.p0, p)?)?)?;
            delta_numerator = add(&delta_numerator, &multiply(&dot, c((d / d0) * dlr))?)?;
            // Adam EMAs scaled by d (Prodigy folds d into the gradient magnitude).
            st.exp_avg = add(
                &multiply(&st.exp_avg, c(beta1))?,
                &multiply(g, c(d * (1.0 - beta1)))?,
            )?;
            st.exp_avg_sq = add(
                &multiply(&st.exp_avg_sq, c(beta2))?,
                &multiply(&g.square()?, c(d * d * (1.0 - beta2)))?,
            )?;
            // s (safeguard_warmup = false ⇒ coefficient (d/d0)·dlr).
            st.s = add(
                &multiply(&st.s, c(beta3))?,
                &multiply(g, c((d / d0) * dlr))?,
            )?;
            d_denom = add(&d_denom, &sum_all_lazy(&st.s.abs()?)?)?;
        }
        // The two syncs for the whole step (F-066): fold the lazy accumulators to host scalars here.
        let delta_numerator = delta_numerator.item::<f32>();
        let d_denom = d_denom.item::<f32>();

        // No usable gradient signal this step (e.g. all-zero grads) — leave d/params unchanged.
        if d_denom == 0.0 {
            self.eval_state()?;
            return Ok(());
        }

        // --- Re-estimate the adapted step d ---
        let global_d_numerator = d_numerator + delta_numerator;
        let d_hat = self.d_coef * global_d_numerator / d_denom;
        let mut d_new = d;
        if d == d0 {
            d_new = d.max(d_hat);
        }
        let d_max = self.d_max.max(d_hat);
        d_new = d_max.min(d_new * self.growth_rate);
        self.d_numerator = global_d_numerator;
        self.d = d_new;
        self.d_max = d_max;

        // --- Pass 2: Adam step. denom uses the NEW d; dlr/weight-decay use the OLD d ---
        for (k, g) in grads.iter() {
            let Some(p) = params.get(k) else { continue };
            let st = self.state.get(k).ok_or_else(|| {
                crate::Error::Msg(format!(
                    "prodigy step: gradient key '{k}' has no pass-1 state (params and grads must \
                     share identical keys)"
                ))
            })?;
            let denom = add(&st.exp_avg_sq.sqrt()?, c(d_new * eps))?;
            let mut np = p.clone();
            if self.weight_decay != 0.0 {
                // Decoupled weight decay: θ -= decay·dlr·θ.
                np = multiply(&np, c(1.0 - self.weight_decay * dlr))?;
            }
            np = subtract(&np, &multiply(&divide(&st.exp_avg, &denom)?, c(dlr))?)?;
            let _ = g; // grad only enters pass 2 via the EMAs (beta1 > 0)
            params.insert(k.clone(), np);
        }
        self.eval_state_and(params)?;
        Ok(())
    }

    /// Materialize the per-parameter state (avoids unbounded lazy-graph growth across steps).
    fn eval_state(&self) -> Result<()> {
        let mut refs: Vec<&Array> = Vec::with_capacity(self.state.len() * 3);
        for st in self.state.values() {
            refs.push(&st.exp_avg);
            refs.push(&st.exp_avg_sq);
            refs.push(&st.s);
        }
        if !refs.is_empty() {
            eval(refs)?;
        }
        Ok(())
    }

    /// Materialize state + the updated parameters together.
    fn eval_state_and(&self, params: &LoraParams) -> Result<()> {
        let mut refs: Vec<&Array> = Vec::with_capacity(self.state.len() * 3 + params.len());
        for st in self.state.values() {
            refs.push(&st.exp_avg);
            refs.push(&st.exp_avg_sq);
            refs.push(&st.s);
        }
        refs.extend(params.values());
        eval(refs)?;
        Ok(())
    }

    /// Snapshot Prodigy's state (sc-9560 / F-125): the three global adaptation scalars as `[1]` arrays
    /// (`__d`, `__d_max`, `__d_numerator`) plus, per parameter key `k`, its four buffers under
    /// `{k}::exp_avg`, `{k}::exp_avg_sq`, `{k}::s`, `{k}::p0`. Only the *adapted* state is persisted;
    /// the base/eps/betas/decay/growth are config, re-supplied by [`Prodigy::new`] on resume.
    fn save_state(&self, path: impl AsRef<Path>) -> Result<()> {
        let d = c(self.d);
        let d_max = c(self.d_max);
        let d_num = c(self.d_numerator);
        let mut entries: Vec<(String, &Array)> = Vec::with_capacity(self.state.len() * 4 + 3);
        entries.push(("__d".to_string(), &d));
        entries.push(("__d_max".to_string(), &d_max));
        entries.push(("__d_numerator".to_string(), &d_num));
        for (k, st) in &self.state {
            entries.push((format!("{k}::exp_avg"), &st.exp_avg));
            entries.push((format!("{k}::exp_avg_sq"), &st.exp_avg_sq));
            entries.push((format!("{k}::s"), &st.s));
            entries.push((format!("{k}::p0"), &st.p0));
        }
        Array::save_safetensors(entries, None::<&HashMap<String, String>>, path)?;
        Ok(())
    }

    /// Restore state written by [`save_state`](Self::save_state) into a fresh `Prodigy::new(...)`.
    /// Rebuilds the per-parameter buffer map and the global scalars; the `{k}::exp_avg` keys carry the
    /// parameter identity (a `k` never ends in `::exp_avg`, so the base key is unambiguous).
    fn load_state(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let (tensors, _meta) = Array::load_safetensors_with_metadata(path.as_ref())?;
        let scalar = |name: &str| -> Result<f32> {
            let a = tensors.get(name).ok_or_else(|| {
                crate::Error::Msg(format!("prodigy resume: snapshot missing global '{name}'"))
            })?;
            Ok(a.item::<f32>())
        };
        self.d = scalar("__d")?;
        self.d_max = scalar("__d_max")?;
        self.d_numerator = scalar("__d_numerator")?;

        let mut state: HashMap<Rc<str>, ProdigyState> = HashMap::new();
        for (name, arr) in &tensors {
            let Some(base) = name.strip_suffix("::exp_avg") else {
                continue;
            };
            let need = |suffix: &str| -> Result<Array> {
                tensors
                    .get(&format!("{base}{suffix}"))
                    .cloned()
                    .ok_or_else(|| {
                        crate::Error::Msg(format!(
                            "prodigy resume: parameter '{base}' missing buffer '{suffix}'"
                        ))
                    })
            };
            state.insert(
                Rc::from(base),
                ProdigyState {
                    exp_avg: arr.clone(),
                    exp_avg_sq: need("::exp_avg_sq")?,
                    s: need("::s")?,
                    p0: need("::p0")?,
                },
            );
        }
        self.state = state;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(pairs: &[(&str, &[f32], &[i32])]) -> LoraParams {
        let mut m = LoraParams::new();
        for (k, data, shape) in pairs {
            m.insert(Rc::from(*k), Array::from_slice(data, shape));
        }
        m
    }

    #[test]
    fn normalize_collapses_aliases() {
        assert_eq!(normalize("AdamW"), "adamw");
        assert_eq!(normalize("prodigy-opt"), "prodigy");
        assert_eq!(normalize("rose_opt"), "rose");
        assert_eq!(normalize("adamw8bit"), "adamw");
        assert!(TrainOptimizer::is_supported("Prodigy"));
        assert!(TrainOptimizer::is_supported("rose"));
        assert!(!TrainOptimizer::is_supported("lion"));
    }

    #[test]
    fn from_config_rejects_unsupported() {
        assert!(TrainOptimizer::from_config("lion", 1e-4, 0.0).is_err());
        assert!(TrainOptimizer::from_config("adamw", 1e-4, 0.0).is_ok());
        assert!(TrainOptimizer::from_config("rose", 1e-4, 0.0).is_ok());
        assert!(TrainOptimizer::from_config("prodigy", 1e-4, 0.0).is_ok());
    }

    #[test]
    fn rose_2d_update_matches_closed_form_no_stabilize() {
        // One 2×2 grad, centralize off + stabilize off ⇒ θ -= lr · g / (|max_row| − min_row).
        let mut rose = Rose::new(0.1, 0.0);
        rose.centralize = false;
        rose.stabilize = false;
        let mut p = params(&[("w", &[0.0, 0.0, 0.0, 0.0], &[2, 2])]);
        let g = params(&[("w", &[1.0, 3.0, -2.0, -4.0], &[2, 2])]);
        rose.step(&mut p, &g).unwrap();
        // Row 0: range = |3| − 1 = 2 ⇒ θ = -0.1·[1,3]/2 = [-0.05, -0.15].
        // Row 1: range = |-2| − (-4) = 6 ⇒ θ = -0.1·[-2,-4]/6 = [0.033..., 0.066...].
        let out = p["w"].as_slice::<f32>();
        let exp = [-0.05, -0.15, 0.2 / 6.0, 0.4 / 6.0];
        for (o, e) in out.iter().zip(exp.iter()) {
            assert!((o - e).abs() < 1e-5, "rose update {out:?} vs {exp:?}");
        }
    }

    #[test]
    fn rose_decoupled_weight_decay_shrinks_then_steps() {
        // weight_decay only (zero grad on a degenerate range) still shrinks the parameter.
        let mut rose = Rose::new(0.1, 0.0);
        rose.centralize = false;
        rose.stabilize = false;
        let mut p = params(&[("w", &[2.0, 2.0, 2.0, 2.0], &[2, 2])]);
        rose.weight_decay = 0.5; // factor = 1 − 0.1·0.5 = 0.95
        let g = params(&[("w", &[0.0, 0.0, 0.0, 0.0], &[2, 2])]); // range 0 → denom 1 → no step
        rose.step(&mut p, &g).unwrap();
        for v in p["w"].as_slice::<f32>() {
            assert!(
                (v - 1.9).abs() < 1e-5,
                "wd-shrunk param should be 1.9, got {v}"
            );
        }
    }

    // Golden inputs shared by the torch-parity tests (a 2×3 param + grad). Goldens generated from
    // the SceneWorks venv's `rose_opt` / `prodigyopt` packages (CPU, fp32) — see tools note in the
    // module doc. Asserts the Rust ports match the reference to f32 precision.
    const P0: [f32; 6] = [1.0, -1.0, 0.5, 0.5, -0.5, 2.0];
    const G: [f32; 6] = [0.2, -0.1, 0.3, -0.4, 0.6, -0.2];

    #[test]
    fn rose_matches_torch_golden() {
        // rose_opt.Rose([p], lr=0.1, weight_decay=0.0, compute_dtype="fp32"), one step.
        let rose = Rose::new(0.1, 0.0); // centralize + stabilize on (SceneWorks defaults)
        let mut p = params(&[("w", &P0, &[2, 3])]);
        rose.step(&mut p, &params(&[("w", &G, &[2, 3])])).unwrap();
        let golden = [
            0.9863946, -0.952381, 0.4659864, 0.543956, -0.5659341, 2.021978,
        ];
        for (o, e) in p["w"].as_slice::<f32>().iter().zip(golden.iter()) {
            assert!((o - e).abs() < 1e-5, "Rose vs torch golden: {o} vs {e}");
        }
    }

    #[test]
    fn prodigy_matches_torch_golden() {
        // prodigyopt.Prodigy([p], lr=1.0, eps=1e-6, weight_decay=0.0), three steps (same grad).
        let mut prod = Prodigy::new(1.0, 0.0);
        let mut p = params(&[("w", &P0, &[2, 3])]);
        for _ in 0..3 {
            prod.step(&mut p, &params(&[("w", &G, &[2, 3])])).unwrap();
        }
        let golden = [
            0.9999849, -0.9999849, 0.4999848, 0.5000151, -0.5000151, 2.000015,
        ];
        for (o, e) in p["w"].as_slice::<f32>().iter().zip(golden.iter()) {
            assert!((o - e).abs() < 1e-5, "Prodigy vs torch golden: {o} vs {e}");
        }
        // Adapted step d after 3 steps (torch: 4.802397e-06).
        assert!(
            (prod.d - 4.802397e-6).abs() < 1e-9,
            "Prodigy d vs torch golden: {} vs 4.802397e-6",
            prod.d
        );
    }

    #[test]
    fn prodigy_adapts_d_upward_and_steps() {
        // Prodigy's d is unchanged on step 0 (p0 == p ⇒ numerator 0); from step 1 the accumulated
        // move makes ⟨g, p0 − p⟩ > 0, so d adapts upward. Run a few steps and check it grows while
        // parameters move and stay finite.
        let mut prod = Prodigy::new(1e-4, 0.0); // LoRA LR < 0.1 ⇒ use_lr = 1.0
        assert!(
            (prod.base_lr - 1.0).abs() < 1e-9,
            "LoRA LR < 0.1 ⇒ Prodigy LR knob 1.0"
        );
        let mut p = params(&[("w", &[1.0, -1.0, 0.5, -0.5], &[2, 2])]);
        let g = params(&[("w", &[0.2, -0.1, 0.3, -0.4], &[2, 2])]);
        let before = p["w"].as_slice::<f32>().to_vec();
        for _ in 0..4 {
            prod.step(&mut p, &g).unwrap();
        }
        assert!(
            prod.d > prod.d0,
            "d should adapt upward after the first move: {} > {}",
            prod.d,
            prod.d0
        );
        let after = p["w"].as_slice::<f32>();
        assert!(after.iter().all(|v| v.is_finite()), "finite params");
        assert!(
            before.iter().zip(after).any(|(b, a)| (b - a).abs() > 0.0),
            "parameters should move"
        );
    }

    /// F-125 core proof: for each optimizer, two continuous steps produce the SAME parameters as one
    /// step → snapshot → fresh optimizer → restore → one step. This is the resume-correctness gate that
    /// runs without real weights (the real-training-trajectory check needs Metal + a licensed
    /// snapshot). AdamW state rides mlx-rs's own serialization; Rose is stateless; Prodigy restores its
    /// global adapted-step scalars + per-parameter buffers.
    fn assert_resume_matches_uninterrupted(opt_name: &str) {
        let seed = || params(&[("w", &[0.10, -0.20, 0.30, 0.05], &[2, 2])]);
        let g = params(&[("w", &[0.40, 0.10, -0.30, 0.20], &[2, 2])]);

        // Continuous: two steps in one optimizer.
        let mut cont = seed();
        let mut o = TrainOptimizer::from_config(opt_name, 1e-3, 0.0).unwrap();
        o.set_lr_scaled(1.0);
        o.step(&mut cont, &g).unwrap();
        o.step(&mut cont, &g).unwrap();

        // Interrupted: one step, snapshot, fresh optimizer, restore, one step.
        let mut resumed = seed();
        let mut o1 = TrainOptimizer::from_config(opt_name, 1e-3, 0.0).unwrap();
        o1.set_lr_scaled(1.0);
        o1.step(&mut resumed, &g).unwrap();
        // The pid keeps two concurrent `cargo test` processes, which share `$TMPDIR`, apart.
        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp
            .path()
            .join(format!("mlxgen_resume_test_{opt_name}.safetensors"));
        o1.save_state(&path).unwrap();
        let mut o2 = TrainOptimizer::from_config(opt_name, 1e-3, 0.0).unwrap();
        o2.set_lr_scaled(1.0);
        o2.load_state(&path).unwrap();
        o2.step(&mut resumed, &g).unwrap();
        let _ = std::fs::remove_file(&path);

        let a = cont["w"].as_slice::<f32>().to_vec();
        let b = resumed["w"].as_slice::<f32>().to_vec();
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(
                (x - y).abs() <= 1e-6,
                "{opt_name} resume diverged at [{i}]: continuous {x} vs resumed {y} \
                 (optimizer state was not restored faithfully)"
            );
        }
    }

    #[test]
    fn adamw_resume_matches_uninterrupted() {
        assert_resume_matches_uninterrupted("adamw");
    }

    #[test]
    fn rose_resume_matches_uninterrupted() {
        assert_resume_matches_uninterrupted("rose");
    }

    #[test]
    fn prodigy_resume_matches_uninterrupted() {
        assert_resume_matches_uninterrupted("prodigy");
    }

    #[test]
    fn prodigy_snapshot_round_trips_globals_and_buffers() {
        let mut prod = Prodigy::new(1.0, 0.0);
        let mut p = params(&[("w", &[1.0, -1.0, 0.5, -0.5], &[2, 2])]);
        let g = params(&[("w", &[0.2, -0.1, 0.3, -0.4], &[2, 2])]);
        for _ in 0..3 {
            prod.step(&mut p, &g).unwrap();
        }
        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp.path().join("mlxgen_prodigy_roundtrip.safetensors");
        prod.save_state(&path).unwrap();
        let mut restored = Prodigy::new(1.0, 0.0);
        restored.load_state(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(restored.d, prod.d, "d");
        assert_eq!(restored.d_max, prod.d_max, "d_max");
        assert_eq!(restored.d_numerator, prod.d_numerator, "d_numerator");
        assert_eq!(restored.state.len(), prod.state.len(), "buffer count");
        let a = prod.state.get("w").unwrap();
        let b = restored.state.get("w").unwrap();
        for (name, x, y) in [
            ("exp_avg", &a.exp_avg, &b.exp_avg),
            ("exp_avg_sq", &a.exp_avg_sq, &b.exp_avg_sq),
            ("s", &a.s, &b.s),
            ("p0", &a.p0, &b.p0),
        ] {
            assert_eq!(
                x.as_slice::<f32>(),
                y.as_slice::<f32>(),
                "prodigy buffer {name} differs after round-trip"
            );
        }
    }
}
