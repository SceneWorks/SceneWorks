//! The **shared decoded-x0 perceptual auxiliary-loss path** (epic 2123 E8, sc-2125).
//!
//! Every auxiliary perceptual loss of the perceptual-character-LoRA epic — depth anchoring
//! (sc-2125), ArcFace identity + face landmarks, ViTPose/HybrIK/Sapiens body losses, and the
//! latent-space VAE-anchor / E-LatentLPIPS losses — runs through this one module instead of
//! re-implementing the plumbing per loss or per trainer:
//!
//! 1. **Recover x0** from the model prediction for the trainer's parameterisation
//!    ([`Parameterization::recover_x0`]).
//! 2. **Decode x0** with the family's small differentiable decoder ([`X0Decoder`]; TAEF1 for the
//!    Flux-VAE 16-channel families — [`super::tae::TinyDecoder`]). Losses whose
//!    [`PerceptualLoss::input`] is [`PerceptualInput::Latents`] skip the decode.
//! 3. **Build a per-image reference once per job** ([`PerceptualPath::ensure_reference`]): each
//!    loss turns the training image's own encode→decode round trip (the trainer's cached clean
//!    latent through the same decoder) into **its own reference type** — features plus any
//!    per-image data it needs at loss time (a face crop box, keypoints, a mask) — or reports the
//!    image **unusable** (`Ok(None)`, e.g. no face found), which skips that loss for that image.
//! 4. **Run the frozen auxiliary model on the live x0 with the reference in hand**
//!    ([`PerceptualLoss::loss`]) — differentiable in the live input, frozen in its own weights
//!    (captured constants, never trainable params, so autograd only produces adapter gradients).
//! 5. **Schedule**: the backend-neutral step policy lives in gen-core
//!    ([`gen_core::train::aux_schedule`], re-exported here): [`AuxAlternation`] keys alternation
//!    per optimizer window (interleaved, phase drifting per pass), [`plan_step`] builds the
//!    [`StepPlan`], and
//!    [`PerceptualPath::plan`] applies the per-image skips (an aux-only step whose claiming losses
//!    all skip the image trains the diffusion loss instead).
//!
//! ## How a trainer uses it
//! ```text
//! let path = PerceptualPath::new(Some(Box::new(taef1)), vec![AuxLoss { schedule, loss }])?;
//! // every entry's reference once, alternation over the schedule's windows:
//! let mut driver =
//!     AuxDriver::prepare(path, cache.len(), |e| clean_nchw(e), &schedule, accum, &req.cancel)?;
//! // per micro-step (the bucket schedule picks the item and its cache entry):
//! let sample = driver.sample(step, &schedule);
//! let step = sample.plan(sampled_sigma)?.expect("a driver plans every step");
//! let sigma = step.plan.noise_level;
//! // inside the traced loss closure:
//! let x0 = Parameterization::FlowNoiseMinusX0 { sigma }.recover_x0(&x_t, &pred)?;
//! let aux = step.path.aux_loss(&step.plan, step.entry, &x0_nchw)?;          // weighted sum
//! let total = combine_step_loss(step.plan.diffusion.then_some(diffusion), aux.map(|a| a.weighted))?;
//! ```
//! Memory (E7): [`perceptual_footprint_bytes`] sums the decoder's and every enabled loss's
//! [`AuxModelFootprint`] (computed from configs, before anything loads).
//!
//! ## How a new aux loss plugs in
//! Implement [`PerceptualLoss`] (name, input kind, `reference` returning its own per-image type or
//! `None` to skip, `loss` reading it back via [`reference_as`]), give it a typed
//! [`AuxLossSchedule`] + a `TrainingTechniques` flag, provide its [`AuxModelFootprint`], and push an
//! [`AuxLoss`] onto the trainer's [`PerceptualPath`]. Decode, reference caching, skipping, the
//! timestep window, alternation and accumulation come for free.

use std::any::Any;
use std::collections::HashMap;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{abs, add, divide, maximum, multiply, subtract};
use mlx_rs::Array;

pub use gen_core::train::aux_schedule::{
    combine_step_terms, perceptual_footprint_bytes, plan_step, AltKey, AuxAlternation,
    AuxModelFootprint, StepPlan,
};
pub use gen_core::train::AuxLossSchedule;

use super::tae::TinyDecoder;
use crate::{Error, Result};

/// How a trainer's model output relates to the clean sample, with the per-step noise scalar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Parameterization {
    /// Flow matching where the model regresses `noise − x0` (Z-Image's negated forward, the
    /// `x_t = (1−σ)·x0 + σ·noise` convention): `x0 = x_t − σ·pred`.
    FlowNoiseMinusX0 { sigma: f32 },
    /// Flow matching where the model regresses `x0 − noise` (the raw diffusers flow output):
    /// `x0 = x_t + σ·pred`.
    FlowX0MinusNoise { sigma: f32 },
    /// ε-prediction: `x0 = (x_t − √(1−ᾱ)·ε) / √ᾱ`.
    Epsilon { alpha_bar: f32 },
    /// v-prediction: `x0 = √ᾱ·x_t − √(1−ᾱ)·v`.
    VPrediction { alpha_bar: f32 },
}

impl Parameterization {
    /// The gain on a step's weighted aux term that makes a given aux-loss weight mean the same on
    /// every parameterisation (sc-2124). The aux gradient reaching the model output is
    /// `∂L/∂pred = (∂x0/∂pred)ᵀ · ∂L/∂x0`, and `|∂x0/∂pred|` is the per-step scalar
    ///
    /// - flow (either sign): `σ ∈ [0, 1]`;
    /// - v-prediction: `√(1−ᾱ) ∈ [0, 1]`;
    /// - ε-prediction: `σ_ve = √((1−ᾱ)/ᾱ)`, unbounded (SDXL: ≈0.03 … ≈14.6).
    ///
    /// Flow and v-prediction are already bounded by 1, so their gain is exactly `1` (the term is
    /// untouched — bit-identical to the ungained path). ε-prediction gets `min(1, 1/σ_ve) =
    /// min(1, √ᾱ/√(1−ᾱ))`, so its effective Jacobian is `min(σ_ve, 1)`: the same as flow's `σ ≈ σ_ve`
    /// at low noise and capped at flow's `1` at high noise, instead of amplifying the aux gradient
    /// up to ~14.6× (the real-weight A/B's adversarial SDXL faces).
    pub fn aux_gain(&self) -> f32 {
        match *self {
            Self::FlowNoiseMinusX0 { .. } | Self::FlowX0MinusNoise { .. } => 1.0,
            Self::VPrediction { .. } => 1.0,
            Self::Epsilon { alpha_bar } => {
                let ab = alpha_bar.clamp(0.0, 1.0);
                let noise = (1.0 - ab).sqrt();
                if noise <= ab.sqrt() {
                    1.0
                } else {
                    ab.sqrt() / noise
                }
            }
        }
    }

    /// Recover the model's x0 estimate from the noisy input and its prediction. Pure MLX ops —
    /// differentiable in `pred`.
    pub fn recover_x0(&self, x_t: &Array, pred: &Array) -> Result<Array> {
        let s = |v: f32| Array::from_f32(v);
        Ok(match *self {
            Self::FlowNoiseMinusX0 { sigma } => subtract(x_t, &multiply(pred, s(sigma))?)?,
            Self::FlowX0MinusNoise { sigma } => add(x_t, &multiply(pred, s(sigma))?)?,
            Self::Epsilon { alpha_bar } => {
                let sa = alpha_bar.max(1e-8).sqrt();
                let s1 = (1.0 - alpha_bar).max(0.0).sqrt();
                divide(&subtract(x_t, &multiply(pred, s(s1))?)?, s(sa))?
            }
            Self::VPrediction { alpha_bar } => {
                let sa = alpha_bar.max(0.0).sqrt();
                let s1 = (1.0 - alpha_bar).max(0.0).sqrt();
                subtract(&multiply(x_t, s(sa))?, &multiply(pred, s(s1))?)?
            }
        })
    }
}

/// A frozen, differentiable latent → pixel decoder for the decoded-x0 losses.
pub trait X0Decoder {
    /// Model-space latents NCHW `[B, C, h, w]` → pixels NHWC `[B, H, W, 3]` in `[0, 1]`,
    /// differentiable in `latents`.
    fn decode(&self, latents: &Array) -> Result<Array>;
}

impl X0Decoder for TinyDecoder {
    fn decode(&self, latents: &Array) -> Result<Array> {
        TinyDecoder::decode(self, latents)
    }
}

/// What a [`PerceptualLoss`] consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerceptualInput {
    /// The decoded x0 pixels (NHWC `[B, H, W, 3]`, `[0, 1]`) — depth, identity, body losses.
    DecodedPixels,
    /// The x0 latent itself (NCHW) — the VAE-anchor / E-LatentLPIPS losses, which skip the decode.
    Latents,
}

/// A loss's per-image reference: whatever type the loss defines (features, crop box, keypoints,
/// mask, …), read back in [`PerceptualLoss::loss`] with [`reference_as`].
pub type LossReference = Box<dyn Any>;

/// Borrow a [`LossReference`] as the loss's own type `T` (a mismatch is a loss bug ⇒ error).
pub fn reference_as<'a, T: 'static>(loss: &str, reference: &'a dyn Any) -> Result<&'a T> {
    reference.downcast_ref::<T>().ok_or_else(|| {
        Error::Msg(format!(
            "perceptual path: the '{loss}' loss was handed a reference of another type"
        ))
    })
}

/// One frozen auxiliary perceptual model + its comparison.
pub trait PerceptualLoss {
    /// Short name for errors/diagnostics (e.g. `"depth"`).
    fn name(&self) -> &'static str;
    /// What [`reference`](Self::reference) and [`loss`](Self::loss) consume.
    fn input(&self) -> PerceptualInput {
        PerceptualInput::DecodedPixels
    }
    /// Build image's reference from its clean input (decoded round trip, or the clean latent for
    /// [`PerceptualInput::Latents`]). Called once per image per job, outside autograd; the loss
    /// must return evaluated, gradient-free data. `Ok(None)` ⇒ the image is **unusable** for this
    /// loss (e.g. no face detected): the loss is skipped for that image on every step.
    fn reference(&self, clean: &Array) -> Result<Option<LossReference>>;
    /// [`reference`](Self::reference) with the image's **subject mask** in hand — `[H, W]` f32 in
    /// `[0, 1]` at the decoded pixel size (1 = subject), when the trainer has one (sc-24832: the
    /// normal loss restricted to the subject reads it). The default ignores the mask.
    fn reference_with_mask(
        &self,
        clean: &Array,
        subject_mask: Option<&Array>,
    ) -> Result<Option<LossReference>> {
        let _ = subject_mask;
        self.reference(clean)
    }
    /// The unweighted scalar loss of the live input for one image, given that image's reference
    /// (crop boxes / keypoints / masks it carries are applied to the live input here).
    /// Differentiable in `live`.
    fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array>;
    /// The step weight of this loss at the plan's noise level `t ∈ [0, 1]` (flow `σ`, or `t / T`),
    /// multiplied into its scheduled weight by [`PerceptualPath::aux_loss`] — e.g. upstream's
    /// `t_ratio` scaling of the face losses. Default `1.0` (no timestep weighting).
    fn timestep_weight(&self, noise_level: f32) -> f32 {
        let _ = noise_level;
        1.0
    }
}

/// A scheduled auxiliary loss.
pub struct AuxLoss {
    pub schedule: AuxLossSchedule,
    pub loss: Box<dyn PerceptualLoss>,
}

/// The weighted aux contribution of one step, plus each contributing loss's raw value.
pub struct AuxTerms {
    /// `Σ weight_i · loss_i` over the plan's aux losses — the term added to the step loss.
    pub weighted: Array,
    /// `(loss index, unweighted loss)` for diagnostics.
    pub per_loss: Vec<(usize, Array)>,
}

/// Sum the step's loss terms ([`combine_step_terms`]): the diffusion loss (when the plan has it)
/// plus the weighted aux sum (when the plan has any aux loss). Both absent errors.
pub fn combine_step_loss(diffusion: Option<Array>, aux: Option<Array>) -> Result<Array> {
    combine_step_terms(
        diffusion,
        aux,
        |d, a| Ok(add(&d, &a)?),
        || {
            Error::Msg(
                "perceptual path: a training step with neither a diffusion nor an aux loss term"
                    .into(),
            )
        },
    )
}

/// The trainer-owned perceptual path: decoder + scheduled losses + the per-image reference cache.
pub struct PerceptualPath {
    decoder: Option<Box<dyn X0Decoder>>,
    losses: Vec<AuxLoss>,
    /// image index → one reference per loss (same order as `losses`; `None` = unusable ⇒ skipped).
    references: HashMap<usize, Vec<Option<LossReference>>>,
    reference_computations: usize,
    /// The job's subject masks for the losses that read one (sc-24832), see
    /// [`attach_subject_masks`](Self::attach_subject_masks).
    subject_masks: Option<gen_core::train::subject_mask::PerceptualSubjectMasks>,
}

impl PerceptualPath {
    /// Build the path. A loss that consumes decoded pixels requires a decoder.
    pub fn new(decoder: Option<Box<dyn X0Decoder>>, losses: Vec<AuxLoss>) -> Result<Self> {
        if decoder.is_none() {
            if let Some(l) = losses
                .iter()
                .find(|l| l.loss.input() == PerceptualInput::DecodedPixels)
            {
                return Err(Error::Msg(format!(
                    "perceptual path: the '{}' loss decodes x0 but no x0 decoder was provided",
                    l.loss.name()
                )));
            }
        }
        Ok(Self {
            decoder,
            losses,
            references: HashMap::new(),
            reference_computations: 0,
            subject_masks: None,
        })
    }

    /// Whether any loss is enabled (otherwise every step is the plain diffusion step).
    pub fn is_active(&self) -> bool {
        self.losses.iter().any(|l| l.schedule.is_enabled())
    }

    /// The losses, in index order.
    pub fn losses(&self) -> &[AuxLoss] {
        &self.losses
    }

    fn needs_pixels<'a>(&self, idxs: impl IntoIterator<Item = &'a usize>) -> bool {
        idxs.into_iter()
            .any(|&i| self.losses[i].loss.input() == PerceptualInput::DecodedPixels)
    }

    fn decode(&self, latents: &Array) -> Result<Array> {
        self.decoder
            .as_ref()
            .ok_or_else(|| Error::Msg("perceptual path: no x0 decoder".into()))?
            .decode(latents)
    }

    fn references_of(&self, image: usize) -> Result<&[Option<LossReference>]> {
        self.references
            .get(&image)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                Error::Msg(format!(
                "perceptual path: no cached reference for image {image} (ensure_reference first)"
            ))
            })
    }

    /// Build and cache image `image`'s per-loss references from its clean latent (NCHW, model
    /// space — the trainer's cached VAE encode of the training image), once per image per job: a
    /// second call for the same image is a no-op. `image` is the trainer's reference key: a
    /// trainer that caches one latent per (item, resolution bucket) keys it per cache entry, so
    /// each bucket's reference matches that bucket's decode size (Z-Image does). The round-trip decode is gradient-stopped.
    pub fn ensure_reference(&mut self, image: usize, clean_latents: &Array) -> Result<()> {
        self.ensure_reference_with_mask(image, clean_latents, None)
    }

    /// Hand the path the job's subject masks (sc-24832): from then on every
    /// [`ensure_reference`](Self::ensure_reference) passes its reference key's item mask — cropped
    /// with the trainer's crop rule and area-averaged onto that reference's decoded pixel grid — to
    /// each loss's [`PerceptualLoss::reference_with_mask`]. `None` (no loss reads masks) is a no-op.
    /// Every trainer calls this once with [`PerceptualSubjectMasks::load`](gen_core::train::subject_mask::PerceptualSubjectMasks::load)
    /// before preparing references.
    pub fn attach_subject_masks(
        &mut self,
        masks: Option<gen_core::train::subject_mask::PerceptualSubjectMasks>,
    ) {
        self.subject_masks = masks;
    }

    /// [`ensure_reference`](Self::ensure_reference) with the image's subject mask (`[H, W]` f32 at
    /// the decoded pixel size), handed to every loss's
    /// [`PerceptualLoss::reference_with_mask`] (sc-24832).
    pub fn ensure_reference_with_mask(
        &mut self,
        image: usize,
        clean_latents: &Array,
        subject_mask: Option<&Array>,
    ) -> Result<()> {
        if self.references.contains_key(&image) {
            return Ok(());
        }
        let all: Vec<usize> = (0..self.losses.len()).collect();
        let pixels = if self.needs_pixels(&all) {
            let px = mlx_rs::stop_gradient(self.decode(clean_latents)?)?;
            px.eval()?;
            Some(px)
        } else {
            None
        };
        // An explicit mask wins; else the attached job masks (sc-24832), resampled onto this
        // reference's decoded pixel grid.
        let attached = match (subject_mask, &self.subject_masks, &pixels) {
            (None, Some(masks), Some(px)) => {
                let sh = px.shape();
                let (h, w) = (sh[1] as usize, sh[2] as usize);
                let v = masks
                    .pixel_mask(image, w, h)
                    .map_err(|e| Error::Msg(e.to_string()))?;
                Some(Array::from_slice(&v, &[h as i32, w as i32]))
            }
            _ => None,
        };
        let subject_mask = subject_mask.or(attached.as_ref());
        let mut refs = Vec::with_capacity(self.losses.len());
        for l in &self.losses {
            let input = match l.loss.input() {
                PerceptualInput::DecodedPixels => pixels.as_ref().expect("decoded above"),
                PerceptualInput::Latents => clean_latents,
            };
            refs.push(l.loss.reference_with_mask(input, subject_mask)?);
        }
        self.references.insert(image, refs);
        self.reference_computations += 1;
        Ok(())
    }

    /// How many images have had their references built this job (each exactly once).
    pub fn reference_computations(&self) -> usize {
        self.reference_computations
    }

    /// Whether loss `loss` has a usable reference for `image` (`false` ⇒ skipped for that image).
    pub fn is_usable(&self, image: usize, loss: usize) -> Result<bool> {
        Ok(self.references_of(image)?[loss].is_some())
    }

    /// The plan for alternation `key` (from [`AuxAlternation::key`]) on `image` at sampled noise
    /// level `raw_t`: [`plan_step`], minus the losses this image is unusable for (an aux-only step
    /// left with no claiming loss trains the diffusion loss instead). Requires the image's
    /// reference ([`ensure_reference`](Self::ensure_reference)).
    pub fn plan(&self, key: impl Into<AltKey>, image: usize, raw_t: f32) -> Result<StepPlan> {
        let refs = self.references_of(image)?;
        let schedules: Vec<AuxLossSchedule> = self.losses.iter().map(|l| l.schedule).collect();
        Ok(plan_step(&schedules, key, raw_t).without_skipped(|i| refs[i].is_none()))
    }

    /// The weighted aux term for `plan` on image `image`'s live x0 latent (NCHW, model space),
    /// differentiable in `x0`. `None` when the plan has no aux loss. A loss the plan names but the
    /// image has no usable reference for is an error (plans from [`plan`](Self::plan) never do).
    pub fn aux_loss(&self, plan: &StepPlan, image: usize, x0: &Array) -> Result<Option<AuxTerms>> {
        if plan.aux.is_empty() {
            return Ok(None);
        }
        let refs = self.references_of(image)?;
        let pixels = if self.needs_pixels(&plan.aux) {
            Some(self.decode(x0)?)
        } else {
            None
        };
        let mut weighted: Option<Array> = None;
        let mut per_loss = Vec::with_capacity(plan.aux.len());
        for &i in &plan.aux {
            let l = &self.losses[i];
            let reference = refs[i].as_deref().ok_or_else(|| {
                Error::Msg(format!(
                    "perceptual path: the '{}' loss is skipped for image {image} but the plan \
                     names it",
                    l.loss.name()
                ))
            })?;
            let input = match l.loss.input() {
                PerceptualInput::DecodedPixels => pixels.as_ref().expect("decoded above"),
                PerceptualInput::Latents => x0,
            };
            let raw = l.loss.loss(input, reference)?;
            let step_weight = l.schedule.weight * l.loss.timestep_weight(plan.noise_level);
            let w = multiply(&raw, Array::from_f32(step_weight))?;
            weighted = Some(match weighted {
                Some(acc) => add(&acc, &w)?,
                None => w,
            });
            per_loss.push((i, raw));
        }
        Ok(Some(AuxTerms {
            weighted: weighted.expect("plan.aux is non-empty"),
            per_loss,
        }))
    }
}

// ----------------------------------------------------------------------------------------------
// Epic 2123 E8 — the step-level glue between a trainer loop and its `PerceptualPath` (the MLX twin
// of `candle_gen::train::flow_match::AuxDriver`), so every MLX trainer prepares references, keys
// the alternation the same way.
// ----------------------------------------------------------------------------------------------

/// A trainer's perceptual path plus its alternation, ready for the loop: every cache entry's
/// reference is built. The alternation is a pure function of the step, so a resumed run needs no
/// replay.
pub struct AuxDriver {
    path: PerceptualPath,
    alternation: AuxAlternation,
}

impl AuxDriver {
    /// Build each of the `n_entries` item-major cache entries' references from `clean(entry)` (the
    /// entry's clean latent in the path's NCHW model-space layout, once per entry, before the loop)
    /// and alternate over the `schedule`'s epochs with `accum` micro-steps per update.
    pub fn prepare(
        path: PerceptualPath,
        n_entries: usize,
        clean: impl FnMut(usize) -> Result<Array>,
        schedule: &gen_core::BucketSchedule,
        accum: u32,
        cancel: &gen_core::runtime::CancelFlag,
    ) -> Result<Self> {
        Self::prepare_keyed(path, n_entries, clean, schedule.epoch_len(), accum, cancel)
    }

    /// [`prepare`](Self::prepare) for a loop whose step → sample mapping is not the plain bucket
    /// schedule walk (Wan's interleaved experts, LTX-2.5's round-robin): its visit order repeats
    /// every `epoch_steps` micro-steps, with `window` micro-steps per alternation window.
    pub fn prepare_keyed(
        mut path: PerceptualPath,
        n_entries: usize,
        mut clean: impl FnMut(usize) -> Result<Array>,
        epoch_steps: usize,
        window: u32,
        cancel: &gen_core::runtime::CancelFlag,
    ) -> Result<Self> {
        for entry in 0..n_entries {
            // A cancel during the (per-entry decode + frozen-model) reference build stops here,
            // before any DiT work.
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            path.ensure_reference(entry, &clean(entry)?)?;
        }
        Ok(Self {
            path,
            alternation: AuxAlternation::new(epoch_steps, window),
        })
    }

    /// The shared path.
    pub fn path(&self) -> &PerceptualPath {
        &self.path
    }

    /// The prepared path, consuming the driver (for a caller that plans by explicit keys).
    pub fn into_path(self) -> PerceptualPath {
        self.path
    }

    /// The alternation key of 1-based micro-step `step` — for the
    /// [`prepare_keyed`](Self::prepare_keyed) loops.
    pub fn key(&self, step: u32) -> AltKey {
        self.alternation.key(step)
    }

    /// Micro-step `step`'s (1-based) [`StepSample`].
    pub fn sample(&mut self, step: u32, schedule: &gen_core::BucketSchedule) -> StepSample<'_> {
        let k = (step - 1) as usize;
        let (item, _) = schedule.sample(k);
        let key = self.alternation.key(step);
        StepSample {
            item,
            entry: schedule.cache_index(k),
            perceptual: Some((&self.path, key)),
        }
    }
}

/// Micro-step `step`'s sample: its [`AuxDriver::sample`] when the trainer has a perceptual path,
/// else the plain (item, entry) of the bucket schedule.
pub fn step_sample<'a>(
    aux: Option<&'a mut AuxDriver>,
    step: u32,
    schedule: &gen_core::BucketSchedule,
) -> StepSample<'a> {
    match aux {
        Some(a) => a.sample(step, schedule),
        None => {
            let k = (step - 1) as usize;
            StepSample::plain(schedule.sample(k).0, schedule.cache_index(k))
        }
    }
}

/// What one micro-step trains on: the real dataset item (alternation key), the item-major cache
/// entry (reference key), and — perceptual losses on — the shared path with the step's
/// alternation key.
#[derive(Clone, Copy)]
pub struct StepSample<'a> {
    /// The real dataset item index (`schedule.sample(k).0`).
    pub item: usize,
    /// The cache entry (`schedule.cache_index(k)`).
    pub entry: usize,
    perceptual: Option<(&'a PerceptualPath, AltKey)>,
}

impl<'a> StepSample<'a> {
    /// A sample with no perceptual path: every step is the plain diffusion step.
    pub fn plain(item: usize, entry: usize) -> Self {
        Self {
            item,
            entry,
            perceptual: None,
        }
    }

    /// Plan the step at the sampled noise level `raw_t` (the trainer's `σ`/`t ∈ [0, 1]`, `1` =
    /// pure noise). `None` without a perceptual path — the trainer then runs its legacy step.
    pub fn plan(&self, raw_t: f32) -> Result<Option<AuxStep<'a>>> {
        let Some((path, key)) = self.perceptual else {
            return Ok(None);
        };
        Ok(Some(AuxStep {
            path,
            plan: path.plan(key, self.entry, raw_t)?,
            entry: self.entry,
        }))
    }
}

/// One planned perceptual step: the [`StepPlan`] and the path (and cache entry) to evaluate its
/// aux term on.
pub struct AuxStep<'a> {
    /// The shared path.
    pub path: &'a PerceptualPath,
    /// The step's plan (aux-only steps carry the remapped noise level).
    pub plan: StepPlan,
    /// The cache entry whose reference the aux term compares against.
    pub entry: usize,
}

// ----------------------------------------------------------------------------------------------
// Depth-consistency losses (MiDaS SSI-L1 + multi-scale gradient matching), the comparison half of
// depth anchoring — a port of ai-toolkit-perceptual `toolkit/depth_consistency.py`
// (`ssi_l1`, `multiscale_grad_loss`, `compute_depth_consistency_loss`, fork commit 6e01a6e).
// ----------------------------------------------------------------------------------------------

/// Upstream `ssi_weight` default.
pub const DEPTH_SSI_WEIGHT: f32 = 1.0;
/// Upstream `grad_weight` default.
pub const DEPTH_GRAD_WEIGHT: f32 = 0.5;
/// Upstream `grad_scales` default.
pub const DEPTH_GRAD_SCALES: usize = 4;

fn flat_rows(x: &Array) -> Result<Array> {
    let sh = x.shape();
    if sh.len() != 3 {
        return Err(Error::Msg(format!(
            "depth loss expects [B, H, W] maps, got shape {sh:?}"
        )));
    }
    Ok(x.reshape(&[sh[0], sh[1] * sh[2]])?)
}

/// Scale-and-shift-invariant L1 between depth maps `[B, H, W]`: per sample, solve
/// `min_{s,t} ‖s·pred + t − target‖²` in closed form (differentiable in `pred`), then the mean
/// absolute error of the aligned prediction. Returns `(loss, s, t)` with `s`/`t` (`[B, 1]`)
/// gradient-stopped.
pub fn ssi_l1(pred: &Array, target: &Array) -> Result<(Array, Array, Array)> {
    let p = flat_rows(pred)?;
    let g = flat_rows(target)?;
    let mean = |a: &Array| a.mean_axes(&[1], true);
    let mean_p = mean(&p)?;
    let mean_g = mean(&g)?;
    let var_p = subtract(&mean(&multiply(&p, &p)?)?, &multiply(&mean_p, &mean_p)?)?;
    let cov = subtract(&mean(&multiply(&p, &g)?)?, &multiply(&mean_p, &mean_g)?)?;
    let s = divide(&cov, &maximum(&var_p, Array::from_f32(1e-6))?)?;
    let t = subtract(&mean_g, &multiply(&s, &mean_p)?)?;
    let aligned = add(&multiply(&s, &p)?, &t)?;
    let loss = abs(&subtract(&aligned, &g)?)?.mean(None)?;
    Ok((loss, mlx_rs::stop_gradient(&s)?, mlx_rs::stop_gradient(&t)?))
}

/// 2×2 average pool of `[B, H, W]` (odd trailing row/col dropped, like torch `avg_pool2d`).
fn avg_pool2(x: &Array) -> Result<Option<Array>> {
    let sh = x.shape();
    let (b, h, w) = (sh[0], sh[1] / 2, sh[2] / 2);
    if h == 0 || w == 0 {
        return Ok(None);
    }
    let cropped = x.index((.., ..2 * h, ..2 * w));
    Ok(Some(
        cropped
            .reshape(&[b, h, 2, w, 2])?
            .mean_axes(&[2, 4], false)?,
    ))
}

/// Multi-scale L1 gradient-matching loss between depth maps `[B, H, W]` (MiDaS): at each of
/// `scales` 2×-pooled levels, the mean |Δx| + |Δy| of `pred − target`, averaged over levels.
pub fn multiscale_grad_loss(pred: &Array, target: &Array, scales: usize) -> Result<Array> {
    let mut p = pred.clone();
    let mut g = target.clone();
    let mut total: Option<Array> = None;
    for k in 0..scales {
        if k > 0 {
            match (avg_pool2(&p)?, avg_pool2(&g)?) {
                (Some(pp), Some(gg)) => {
                    p = pp;
                    g = gg;
                }
                _ => break,
            }
        }
        let d = subtract(&p, &g)?;
        let sh = d.shape();
        let mut level = Array::from_f32(0.0);
        if sh[2] > 1 {
            let dx = abs(&subtract(d.index((.., .., 1..)), d.index((.., .., ..-1)))?)?;
            level = add(&level, &dx.mean(None)?)?;
        }
        if sh[1] > 1 {
            let dy = abs(&subtract(d.index((.., 1.., ..)), d.index((.., ..-1, ..)))?)?;
            level = add(&level, &dy.mean(None)?)?;
        }
        total = Some(match total {
            Some(acc) => add(&acc, &level)?,
            None => level,
        });
    }
    let total = total.unwrap_or_else(|| Array::from_f32(0.0));
    Ok(divide(&total, Array::from_f32(scales as f32))?)
}

/// The full depth-consistency loss: `DEPTH_SSI_WEIGHT · ssi_l1 + DEPTH_GRAD_WEIGHT ·
/// multiscale_grad(s·pred + t, target)`, with `s`/`t` the gradient-stopped SSI alignment (so the
/// prediction's arbitrary depth scale never leaks into the gradient term). Invariant to any
/// positive or negative scale and any shift of `pred`.
pub fn depth_consistency_loss(pred: &Array, target: &Array) -> Result<Array> {
    let (ssi, s, t) = ssi_l1(pred, target)?;
    let sh = pred.shape();
    let s3 = s.reshape(&[sh[0], 1, 1])?;
    let t3 = t.reshape(&[sh[0], 1, 1])?;
    let aligned = add(&multiply(pred, &s3)?, &t3)?;
    let grad = multiscale_grad_loss(&aligned, target, DEPTH_GRAD_SCALES)?;
    Ok(add(
        &multiply(&ssi, Array::from_f32(DEPTH_SSI_WEIGHT))?,
        &multiply(&grad, Array::from_f32(DEPTH_GRAD_WEIGHT))?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::random;
    use mlx_rs::transforms::{eval, grad};

    fn sched(weight: f32, t_min: f32, t_max: f32, every_n: u32) -> AuxLossSchedule {
        AuxLossSchedule {
            weight,
            t_min,
            t_max,
            every_n,
        }
    }

    fn map(seed: u64, shape: &[i32]) -> Array {
        random::uniform::<_, f32>(0.0f32, 1.0f32, shape, Some(&random::key(seed).unwrap())).unwrap()
    }

    fn scalar(a: &Array) -> f32 {
        eval([a]).unwrap();
        a.item::<f32>()
    }

    /// "Decoder" for the plumbing tests: NCHW `[1, 3, H, W]` → NHWC, unchanged values.
    struct Identity;
    impl X0Decoder for Identity {
        fn decode(&self, latents: &Array) -> Result<Array> {
            Ok(latents.transpose_axes(&[0, 2, 3, 1])?)
        }
    }

    /// The toy per-image crop loss the face/body losses (S10/S11) are shaped like: at reference
    /// time it finds the bright region of channel 0 (the "face"), stores its bounding box plus the
    /// clean crop's mean colour, and reports images with no bright region as unusable; at loss time
    /// it crops the LIVE pixels with the stored box and compares means.
    struct BrightCrop;
    struct CropRef {
        /// `(y0, y1, x0, x1)`, half-open.
        bbox: (i32, i32, i32, i32),
        mean: Array,
    }
    impl BrightCrop {
        fn crop(px: &Array, b: (i32, i32, i32, i32)) -> Array {
            px.index((.., b.0..b.1, b.2..b.3, ..))
        }
    }
    impl PerceptualLoss for BrightCrop {
        fn name(&self) -> &'static str {
            "bright-crop"
        }
        fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
            // Read pixels by index: the decode is a transposed view, so its raw buffer is not in
            // NHWC order.
            let (h, w) = (clean.shape()[1], clean.shape()[2]);
            let (mut y0, mut y1, mut x0, mut x1) = (h, 0, w, 0);
            for y in 0..h {
                for x in 0..w {
                    if clean.index((0, y, x, 0)).item::<f32>() > 0.5 {
                        (y0, y1, x0, x1) = (y0.min(y), y1.max(y + 1), x0.min(x), x1.max(x + 1));
                    }
                }
            }
            if y1 == 0 {
                return Ok(None);
            }
            let bbox = (y0, y1, x0, x1);
            let mean = mlx_rs::stop_gradient(Self::crop(clean, bbox).mean(None)?)?;
            mean.eval()?;
            Ok(Some(Box::new(CropRef { bbox, mean })))
        }
        fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
            let r = reference_as::<CropRef>(self.name(), reference)?;
            let m = Self::crop(live, r.bbox).mean(None)?;
            Ok(subtract(&m, &r.mean)?.square()?)
        }
    }

    /// A clean NCHW image `[1, 3, 6, 6]`: zeros, with a bright 2×3 patch at rows 1..3, cols 2..5
    /// when `face` is set.
    fn clean_image(face: bool) -> (Array, Vec<f32>) {
        let mut v = vec![0.1f32; 3 * 36];
        if face {
            for y in 1..3 {
                for x in 2..5 {
                    v[y * 6 + x] = 0.9; // channel 0 plane of NCHW
                }
            }
        }
        (Array::from_slice(&v, &[1, 3, 6, 6]), v)
    }

    fn crop_path() -> PerceptualPath {
        PerceptualPath::new(
            Some(Box::new(Identity)),
            vec![AuxLoss {
                schedule: sched(0.5, 0.0, 1.0, 2),
                loss: Box::new(BrightCrop),
            }],
        )
        .unwrap()
    }

    /// Review major (S10/S11 shape): a loss builds its own per-image reference (a crop box computed
    /// at reference time) and applies it to the LIVE decode. Changing live pixels outside the box
    /// leaves the loss at zero; changing them inside raises it. Mutation: crop the live pixels with
    /// the full frame instead of `r.bbox` ⇒ the outside change moves the loss ⇒ red.
    #[test]
    fn a_crop_loss_applies_its_reference_box_to_the_live_decode() {
        let mut path = crop_path();
        let (clean, v) = clean_image(true);
        path.ensure_reference(0, &clean).unwrap();
        let plan = path.plan(2, 0, 0.5).unwrap();
        assert!(!plan.diffusion && plan.aux == vec![0], "{plan:?}");
        let at = |v: &[f32]| {
            let x0 = Array::from_slice(v, &[1, 3, 6, 6]);
            scalar(&path.aux_loss(&plan, 0, &x0).unwrap().unwrap().weighted)
        };
        assert!(at(&v).abs() < 1e-10, "self loss");
        let mut outside = v.clone();
        outside[5 * 6] = 0.0; // row 5, col 0: outside the box
        assert!(
            at(&outside).abs() < 1e-10,
            "outside-box change must not count"
        );
        let mut inside = v.clone();
        inside[6 + 3] = 0.1; // row 1, col 3: inside the box
        assert!(at(&inside) > 1e-4, "inside-box change must count");
    }

    /// Review major: an image a loss cannot use (no "face") is skipped — no error. On a step its
    /// alternation claims, the step trains the diffusion loss instead and the aux term is absent
    /// (zero contribution). Mutation: drop `without_skipped` in `PerceptualPath::plan` ⇒ the plan
    /// names the loss ⇒ `aux_loss` errors ⇒ red.
    #[test]
    fn an_unusable_image_is_skipped_and_falls_back_to_diffusion() {
        let mut path = crop_path();
        let (clean, _) = clean_image(false);
        path.ensure_reference(1, &clean).unwrap();
        assert!(!path.is_usable(1, 0).unwrap());
        let plan = path.plan(2, 1, 0.5).unwrap();
        assert!(plan.diffusion && plan.aux.is_empty(), "{plan:?}");
        assert!(path.aux_loss(&plan, 1, &clean).unwrap().is_none());
        // An image never prepared is an error, never a silent skip.
        assert!(path.plan(2, 7, 0.5).is_err());
    }

    #[test]
    fn a_pixel_loss_without_a_decoder_is_refused() {
        let r = PerceptualPath::new(
            None,
            vec![AuxLoss {
                schedule: sched(0.1, 0.0, 1.0, 2),
                loss: Box::new(BrightCrop),
            }],
        );
        assert!(r.is_err());
    }

    /// sc-24831: a loss's `timestep_weight` scales its term by the plan's noise level, on top of
    /// its scheduled weight; the default is 1. Mutation: ignore `timestep_weight` in `aux_loss` ⇒
    /// the weighted term equals the unweighted one ⇒ red.
    #[test]
    fn timestep_weight_scales_the_term_by_the_plan_noise_level() {
        struct TWeighted;
        impl PerceptualLoss for TWeighted {
            fn name(&self) -> &'static str {
                "t-weighted"
            }
            fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
                BrightCrop.reference(clean)
            }
            fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
                BrightCrop.loss(live, reference)
            }
            fn timestep_weight(&self, t: f32) -> f32 {
                t * t
            }
        }
        assert_eq!(BrightCrop.timestep_weight(0.3), 1.0);
        let mut path = PerceptualPath::new(
            Some(Box::new(Identity)),
            vec![AuxLoss {
                schedule: sched(0.5, 0.0, 1.0, 1),
                loss: Box::new(TWeighted),
            }],
        )
        .unwrap();
        let (clean, mut v) = clean_image(true);
        path.ensure_reference(0, &clean).unwrap();
        v[6 + 3] = 0.1;
        let x0 = Array::from_slice(&v, &[1, 3, 6, 6]);
        let plan = path.plan(0, 0, 0.6).unwrap();
        let terms = path.aux_loss(&plan, 0, &x0).unwrap().unwrap();
        let raw = scalar(&terms.per_loss[0].1);
        let t = plan.noise_level;
        assert!(raw > 1e-4);
        assert!((scalar(&terms.weighted) - 0.5 * t * t * raw).abs() < 1e-7);
    }

    /// A latent-input toy loss for the [`AuxDriver`] tests: its reference is the clean latent's
    /// value (so each entry's reference is identifiable), its loss `|live − reference|`.
    struct EntryValue;
    impl PerceptualLoss for EntryValue {
        fn name(&self) -> &'static str {
            "entry-value"
        }
        fn input(&self) -> PerceptualInput {
            PerceptualInput::Latents
        }
        fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
            Ok(Some(Box::new(scalar(&clean.sum(None)?))))
        }
        fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
            let r = *reference_as::<f32>(self.name(), reference)?;
            Ok(abs(subtract(&live.sum(None)?, Array::from_f32(r))?)?)
        }
    }

    fn entry_path() -> PerceptualPath {
        PerceptualPath::new(
            None,
            vec![AuxLoss {
                schedule: sched(1.0, 0.0, 1.0, 2),
                loss: Box::new(EntryValue),
            }],
        )
        .unwrap()
    }

    /// Entry `e`'s clean latent: a `[1, 1, 1, 1]` holding `e`.
    fn entry_latent(e: usize) -> Result<Array> {
        Ok(Array::from_slice(&[e as f32], &[1, 1, 1, 1]))
    }

    /// 3 items × 2 buckets (repeats 1 and 2): an item-major cache of 6 entries.
    fn two_bucket_schedule() -> gen_core::BucketSchedule {
        let buckets = [
            gen_core::train::ResolutionBucket {
                resolution: 512,
                repeats: 1,
            },
            gen_core::train::ResolutionBucket {
                resolution: 768,
                repeats: 2,
            },
        ];
        gen_core::BucketSchedule::new(3, &buckets, 7)
    }

    /// A cancel tripped during the reference build stops it between entries with a typed
    /// `Canceled` — no further entry is decoded, so no DiT work follows (feature-end review round
    /// 2). Mutation: drop the per-entry `cancel` check ⇒ all 6 entries build and `prepare` succeeds
    /// ⇒ red.
    #[test]
    fn aux_driver_prepare_stops_on_cancel_between_entries() {
        let schedule = two_bucket_schedule();
        let cancel = gen_core::runtime::CancelFlag::default();
        let mut calls = 0;
        let result = AuxDriver::prepare(
            entry_path(),
            6,
            |e| {
                calls += 1;
                cancel.cancel();
                entry_latent(e)
            },
            &schedule,
            1,
            &cancel,
        );
        assert!(matches!(result, Err(Error::Canceled)), "{:?}", result.err());
        assert_eq!(calls, 1, "the build stopped at the next entry");
    }

    /// [`AuxDriver::prepare`] builds every item-major entry's reference exactly once, each from its
    /// OWN entry's clean latent (the aux term on entry `e`'s live latent `e` is zero). Mutations:
    /// build only `0..n_entries - 1` ⇒ entry 5 has no reference ⇒ red; call `clean` twice per
    /// entry ⇒ 12 calls ⇒ red; pass `clean(0)` for every entry ⇒ entry 3's term is non-zero ⇒ red.
    #[test]
    fn aux_driver_builds_each_entry_reference_once() {
        let schedule = two_bucket_schedule();
        let mut calls = 0;
        let d = AuxDriver::prepare(
            entry_path(),
            6,
            |e| {
                calls += 1;
                entry_latent(e)
            },
            &schedule,
            1,
            &Default::default(),
        )
        .unwrap();
        assert_eq!(calls, 6);
        assert_eq!(d.path().reference_computations(), 6);
        let plan = plan_step(&[sched(1.0, 0.0, 1.0, 1)], 1, 0.5);
        for e in 0..6 {
            assert!(d.path().is_usable(e, 0).unwrap(), "entry {e}");
            let live = entry_latent(e).unwrap();
            let terms = d.path().aux_loss(&plan, e, &live).unwrap().unwrap();
            assert_eq!(scalar(&terms.weighted), 0.0, "entry {e}");
        }
    }

    /// The driver alternates over the schedule's epochs (sc-2124): every step plans exactly what
    /// the gen-core alternation over `epoch_len` micro-steps claims, no two aux steps run back to
    /// back, every item gets both kinds and an aux step lands on a bucket-1 entry; every sample
    /// reports the schedule's item and item-major entry. Mutation: build the alternation over the
    /// cache length (`n_entries`, an even period) instead of `schedule.epoch_len()` ⇒ the plans
    /// drift off the expected keys ⇒ red.
    #[test]
    fn aux_driver_interleaves_over_the_schedule() {
        let schedule = two_bucket_schedule();
        let mut d = AuxDriver::prepare(
            entry_path(),
            6,
            entry_latent,
            &schedule,
            1,
            &Default::default(),
        )
        .unwrap();
        let expected = AuxAlternation::new(schedule.epoch_len(), 1);
        let mut kinds = [Vec::new(), Vec::new(), Vec::new()];
        let mut flags = Vec::new();
        let mut aux_on_bucket1 = false;
        for step in 1..=36u32 {
            let k = (step - 1) as usize;
            let s = d.sample(step, &schedule);
            assert_eq!(s.item, schedule.sample(k).0);
            assert_eq!(s.entry, schedule.cache_index(k));
            let p = s.plan(0.3).unwrap().expect("a driver plans every step");
            assert_eq!(p.entry, s.entry);
            assert_eq!(
                !p.plan.diffusion,
                expected.key(step).claims(2),
                "step {step}"
            );
            kinds[s.item].push(p.plan.diffusion);
            flags.push(p.plan.diffusion);
            aux_on_bucket1 |= !p.plan.diffusion && s.entry % 2 == 1;
        }
        assert!(
            flags.windows(2).all(|w| w[0] || w[1]),
            "two aux steps in a row: {flags:?}"
        );
        for (item, k) in kinds.iter().enumerate() {
            assert!(
                k.contains(&true) && k.contains(&false),
                "item {item}: {k:?}"
            );
        }
        assert!(
            aux_on_bucket1,
            "the alternation must span an item's buckets"
        );
        // Without a driver the sample is the plain schedule pick and plans nothing.
        let plain = step_sample(None, 4, &schedule);
        assert_eq!(
            (plain.item, plain.entry),
            (schedule.sample(3).0, schedule.cache_index(3))
        );
        assert!(plain.plan(0.3).unwrap().is_none());
    }

    /// Resume needs no replay (sc-2124): the alternation is a pure function of the step, so a
    /// driver first sampled at step 6 (accumulation 2) plans steps 6.. exactly as one walked from
    /// step 1, and a [`AuxDriver::prepare_keyed`] driver's keys are the gen-core alternation over
    /// its `epoch_steps`. Mutation: make the key depend on how many steps were sampled ⇒ red.
    #[test]
    fn aux_driver_resume_plans_like_a_full_walk() {
        let schedule = two_bucket_schedule();
        let plans = |start: u32| {
            let mut d = AuxDriver::prepare(
                entry_path(),
                6,
                entry_latent,
                &schedule,
                2,
                &Default::default(),
            )
            .unwrap();
            (start + 1..=24)
                .map(|step| {
                    let s = d.sample(step, &schedule);
                    let p = s.plan(0.3).unwrap().unwrap();
                    (s.item, s.entry, p.plan.diffusion, p.plan.aux.clone())
                })
                .collect::<Vec<_>>()
        };
        let full = plans(0);
        assert!(full.iter().any(|p| !p.2) && full.iter().any(|p| p.2));
        assert_eq!(plans(5), full[5..].to_vec());

        let d = AuxDriver::prepare_keyed(entry_path(), 3, entry_latent, 3, 2, &Default::default())
            .unwrap();
        let alternation = AuxAlternation::new(3, 2);
        for step in 1..=12 {
            assert_eq!(d.key(step), alternation.key(step), "step {step}");
        }
    }

    /// References are built once per image per job however often `ensure_reference` is called.
    /// Mutation: drop the `contains_key` early return ⇒ counter 6 ⇒ red.
    #[test]
    fn references_are_computed_once_per_image() {
        let mut path = crop_path();
        let imgs = [clean_image(true).0, clean_image(false).0];
        for _epoch in 0..3 {
            for (i, l) in imgs.iter().enumerate() {
                path.ensure_reference(i, l).unwrap();
            }
        }
        assert_eq!(path.reference_computations(), 2);
        assert!(combine_step_loss(None, None).is_err());
    }

    /// AC3: the SSI loss (and the full depth loss) is invariant to scale and shift of the
    /// predicted depth. Mutation: replace `aligned = s·p + t` with `aligned = p` in `ssi_l1` ⇒ the
    /// scaled/shifted losses differ ⇒ red.
    #[test]
    fn ssi_and_depth_loss_are_scale_and_shift_invariant() {
        let pred = map(1, &[2, 12, 10]);
        let target = map(2, &[2, 12, 10]);
        let (base, _, _) = ssi_l1(&pred, &target).unwrap();
        let base = scalar(&base);
        let full = scalar(&depth_consistency_loss(&pred, &target).unwrap());
        assert!(
            base > 0.01,
            "a non-trivial pair must have a non-zero loss: {base}"
        );
        for (a, b) in [(3.0f32, 0.0f32), (0.25, 7.0), (1.0, -2.5), (-2.0, 1.0)] {
            let moved = add(
                multiply(&pred, Array::from_f32(a)).unwrap(),
                Array::from_f32(b),
            )
            .unwrap();
            let (l, _, _) = ssi_l1(&moved, &target).unwrap();
            let l = scalar(&l);
            assert!(
                (l - base).abs() <= 1e-4 * base.max(1.0),
                "ssi under ({a}, {b}): {l} vs {base}"
            );
            let f = scalar(&depth_consistency_loss(&moved, &target).unwrap());
            assert!(
                (f - full).abs() <= 1e-4 * full.max(1.0),
                "depth loss under ({a}, {b}): {f} vs {full}"
            );
        }
    }

    /// The self-loss is zero (the cached round-trip reference is a reachable target), and the loss
    /// is differentiable in the prediction.
    #[test]
    fn depth_loss_has_a_zero_floor_and_flows_gradient() {
        let target = map(3, &[1, 9, 9]);
        let self_loss = scalar(&depth_consistency_loss(&target, &target).unwrap());
        assert!(self_loss.abs() < 1e-5, "self loss {self_loss}");
        let f = |p: &Array| -> mlx_rs::error::Result<Array> {
            depth_consistency_loss(p, &target)
                .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))
        };
        let g = grad(f)(&map(4, &[1, 9, 9])).unwrap();
        assert!(scalar(&g.abs().unwrap().sum(None).unwrap()) > 0.0);
    }

    #[test]
    fn multiscale_grad_loss_sees_structure_not_offset() {
        let t = map(5, &[1, 16, 16]);
        let shifted = add(&t, Array::from_f32(0.3)).unwrap();
        assert!(scalar(&multiscale_grad_loss(&shifted, &t, 4).unwrap()) < 1e-6);
        let other = map(6, &[1, 16, 16]);
        assert!(scalar(&multiscale_grad_loss(&other, &t, 4).unwrap()) > 0.01);
    }

    /// sc-2124: the aux gain normalises the ε Jacobian `σ_ve = √((1−ᾱ)/ᾱ)` to `min(σ_ve, 1)` —
    /// exactly `1/σ_ve` at high noise (SDXL's last timestep, σ_ve ≈ 14.6), exactly `1` at low noise —
    /// and is exactly `1` (no change at all) for flow and v-prediction at every noise level.
    /// Mutations: return `1` for `Epsilon` ⇒ red; return `1/σ_ve` unclamped ⇒ the low-noise case
    /// (gain > 1) reds; scale the flow gain by `σ` ⇒ red.
    #[test]
    fn aux_gain_normalises_epsilon_and_leaves_flow_alone() {
        for ab in [0.004_7f32, 0.05, 0.2, 0.45] {
            let sigma_ve = ((1.0 - ab) / ab).sqrt();
            assert!(sigma_ve > 1.0);
            let g = Parameterization::Epsilon { alpha_bar: ab }.aux_gain();
            assert!(
                (g - 1.0 / sigma_ve).abs() <= 1e-6,
                "ᾱ={ab}: {g} vs {}",
                1.0 / sigma_ve
            );
            assert!(g < 1.0);
        }
        for ab in [0.5f32, 0.8, 0.999, 1.0] {
            assert_eq!(
                Parameterization::Epsilon { alpha_bar: ab }.aux_gain(),
                1.0,
                "ᾱ={ab}"
            );
        }
        assert_eq!(Parameterization::Epsilon { alpha_bar: 0.0 }.aux_gain(), 0.0);
        for v in [0.0f32, 0.3, 0.5, 0.97, 1.0] {
            assert_eq!(
                Parameterization::FlowNoiseMinusX0 { sigma: v }.aux_gain(),
                1.0
            );
            assert_eq!(
                Parameterization::FlowX0MinusNoise { sigma: v }.aux_gain(),
                1.0
            );
            assert_eq!(
                Parameterization::VPrediction { alpha_bar: v }.aux_gain(),
                1.0
            );
        }
    }

    #[test]
    fn recover_x0_inverts_each_parameterisation() {
        let x0 = map(7, &[1, 2, 3, 3]);
        let n = map(8, &[1, 2, 3, 3]);
        let close = |a: &Array, b: &Array| {
            let d = scalar(&subtract(a, b).unwrap().abs().unwrap().max(None).unwrap());
            assert!(d < 1e-5, "max diff {d}");
        };
        let sigma = 0.3f32;
        let lerp = |a: f32, b: f32| {
            add(
                multiply(&x0, Array::from_f32(a)).unwrap(),
                multiply(&n, Array::from_f32(b)).unwrap(),
            )
            .unwrap()
        };
        let x_t = lerp(1.0 - sigma, sigma);
        let v_neg = subtract(&n, &x0).unwrap();
        close(
            &Parameterization::FlowNoiseMinusX0 { sigma }
                .recover_x0(&x_t, &v_neg)
                .unwrap(),
            &x0,
        );
        let v = subtract(&x0, &n).unwrap();
        close(
            &Parameterization::FlowX0MinusNoise { sigma }
                .recover_x0(&x_t, &v)
                .unwrap(),
            &x0,
        );
        let ab = 0.64f32;
        let x_t = lerp(ab.sqrt(), (1.0 - ab).sqrt());
        close(
            &Parameterization::Epsilon { alpha_bar: ab }
                .recover_x0(&x_t, &n)
                .unwrap(),
            &x0,
        );
        let vp = subtract(
            multiply(&n, Array::from_f32(ab.sqrt())).unwrap(),
            multiply(&x0, Array::from_f32((1.0 - ab).sqrt())).unwrap(),
        )
        .unwrap();
        close(
            &Parameterization::VPrediction { alpha_bar: ab }
                .recover_x0(&x_t, &vp)
                .unwrap(),
            &x0,
        );
    }
}
