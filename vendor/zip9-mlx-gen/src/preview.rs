//! Shared per-step latent preview machinery.
//!
//! Providers own their latent layout and linear RGB fit. This module only numbers schedule
//! positions, validates an already-unpacked `[1, C, h, w]` latent, projects it with a caller-owned
//! `C x 3` fit, and emits the resulting RGB8 frame.

use std::cell::Cell;

use mlx_rs::ops::{add, matmul, maximum, minimum, multiply, round};
use mlx_rs::{Array, Dtype};

use crate::array::scalar;
use crate::{Error, Image, PreviewFrame, PreviewSink, Result};

/// Numbers preview frames by schedule position.
///
/// The sampler driver invokes providers once per actual outer solver step. Keying the resulting
/// frames to schedule positions keeps numbering monotonic and bounded even for truncated schedules.
pub struct PreviewCounter {
    emitted: Cell<u32>,
    total: u32,
}

impl PreviewCounter {
    /// Build a counter for a sampler schedule (`n + 1` sigma entries for `n` steps).
    pub fn new(sigmas: &[f32]) -> Self {
        Self {
            emitted: Cell::new(0),
            total: sigmas.len().saturating_sub(1).max(1) as u32,
        }
    }

    /// Total denoise steps represented by this schedule.
    pub fn total(&self) -> u32 {
        self.total
    }

    /// Return the 1-based frame number for `sigma`, or `None` if that position was emitted.
    ///
    /// An off-schedule solver evaluation advances one position beyond the last emission. Frame
    /// numbering remains monotonic and bounded in both cases.
    pub fn next(&self, sigmas: &[f32], sigma: f32) -> Option<u32> {
        let candidate = match sigmas.iter().position(|s| *s == sigma) {
            Some(index) => index as u32 + 1,
            None => self.emitted.get().saturating_add(1),
        }
        .min(self.total);
        if candidate <= self.emitted.get() {
            return None;
        }
        self.emitted.set(candidate);
        Some(candidate)
    }
}

/// Run a provider-owned projection and emit it for this schedule position.
///
/// Projection failures are deliberately swallowed: previews are decorative and losing a frame
/// must never fail the caller's render. The projection closure runs only after the counter advances,
/// preserving schedule-position consumption when family-specific unpacking fails, and is never
/// invoked for an inert sink or an already-emitted position.
pub fn emit_preview<F>(
    sink: &PreviewSink,
    counter: &PreviewCounter,
    sigmas: &[f32],
    sigma: f32,
    project: F,
) where
    F: FnOnce() -> Result<Image>,
{
    emit_preview_with_sigma(sink, counter, sigmas, sigma, |_| project());
}

/// Run a provider-owned projection that also receives the schedule sigma for this frame.
///
/// This is additive to [`emit_preview`]: existing providers keep their zero-argument projection
/// closure and therefore cannot accidentally change behaviour. Discrete epsilon-prediction families
/// whose running latent is in raw VE sigma-space use this seam to apply the same input scaling as
/// their denoiser before projecting. The optional argument lets those projectors fail closed if a
/// sigma-less provider calls them directly; this schedule-keyed emitter always supplies `Some`.
pub fn emit_preview_with_sigma<F>(
    sink: &PreviewSink,
    counter: &PreviewCounter,
    sigmas: &[f32],
    sigma: f32,
    project: F,
) where
    F: FnOnce(Option<f32>) -> Result<Image>,
{
    if !sink.is_active() {
        return;
    }
    let Some(current) = counter.next(sigmas, sigma) else {
        return;
    };
    if let Ok(image) = project(Some(sigma)) {
        sink.emit(PreviewFrame {
            current,
            total: counter.total(),
            image,
        });
    }
}

/// Project an unpacked `[1, C, h, w]` latent to a latent-resolution RGB8 image.
///
/// `factors` must contain exactly one RGB row per latent channel. The fit remains provider-owned:
/// sharing this operation does not imply that latent spaces share coefficients.
pub fn project_latents(latents: &Array, factors: &[[f32; 3]], bias: [f32; 3]) -> Result<Image> {
    let shape = latents.shape();
    if shape.len() != 4 || shape[0] != 1 {
        return Err(format!("preview latent must have shape [1, C, h, w], got {shape:?}").into());
    }
    if shape[1] == 0 || shape[2] == 0 || shape[3] == 0 {
        return Err(format!(
            "preview latent channel and spatial dimensions must be non-zero, got {shape:?}"
        )
        .into());
    }
    let channels = shape[1] as usize;
    if factors.len() != channels {
        return Err(format!(
            "preview factor row count {} does not match latent channel count {channels}",
            factors.len()
        )
        .into());
    }

    let (h, w) = (shape[2], shape[3]);
    let x = latents.as_dtype(Dtype::Float32)?;
    let x = x.reshape(&[shape[1], h * w])?.transpose_axes(&[1, 0])?;

    let factor_values: Vec<f32> = factors.iter().flatten().copied().collect();
    let factors = Array::from_slice(&factor_values, &[shape[1], 3]);
    let bias = Array::from_slice(&bias, &[3]);
    let rgb = add(&matmul(&x, &factors)?, &bias)?;
    let rgb = minimum(&maximum(&rgb, scalar(0.0))?, scalar(1.0))?;
    let rgb = round(&multiply(&rgb, scalar(255.0))?, 0)?;
    let pixels = rgb
        .try_as_slice::<f32>()
        .map_err(|error| Error::Msg(format!("preview projection readback failed: {error}")))?
        .iter()
        .map(|&v| v as u8)
        .collect();
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    /// An 8-step schedule has 9 sigmas. Euler evaluates once per step, at sigmas[0..8].
    const SIGMAS: [f32; 9] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0];

    #[test]
    fn euler_cadence_numbers_every_step_once() {
        let counter = PreviewCounter::new(&SIGMAS);
        assert_eq!(counter.total(), 8);
        let frames: Vec<_> = SIGMAS[..8]
            .iter()
            .filter_map(|&s| counter.next(&SIGMAS, s))
            .collect();
        assert_eq!(frames, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn duplicate_schedule_positions_are_not_emitted_twice() {
        let counter = PreviewCounter::new(&SIGMAS);
        let mut frames = Vec::new();
        for &sigma in &SIGMAS[..8] {
            frames.extend(counter.next(&SIGMAS, sigma));
            frames.extend(counter.next(&SIGMAS, sigma));
        }
        assert_eq!(frames, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(frames.iter().all(|&f| f <= counter.total()));
    }

    #[test]
    fn off_schedule_sigma_falls_back_to_monotonic_numbering() {
        let counter = PreviewCounter::new(&SIGMAS);
        assert_eq!(counter.next(&SIGMAS, 0.95), Some(1));
        assert_eq!(counter.next(&SIGMAS, 0.85), Some(2));
        assert_eq!(counter.next(&SIGMAS, 0.9), None);
        assert_eq!(counter.next(&SIGMAS, 0.8), Some(3));
    }

    #[test]
    fn single_sigma_schedule_has_a_total_of_one() {
        let sigmas = [1.0_f32];
        let counter = PreviewCounter::new(&sigmas);
        assert_eq!(counter.total(), 1);
        assert_eq!(counter.next(&sigmas, 1.0), Some(1));
        assert_eq!(counter.next(&sigmas, 1.0), None);
    }

    #[test]
    fn projection_accepts_every_shipped_latent_channel_count() {
        for channels in [4, 12, 16, 32, 128] {
            let latents = Array::zeros::<f32>(&[1, channels, 2, 3]).unwrap();
            let factors = vec![[0.0; 3]; channels as usize];
            let image = project_latents(&latents, &factors, [0.0, 0.5, 1.0]).unwrap();
            assert_eq!((image.width, image.height), (3, 2));
            assert_eq!(image.pixels, [0, 128, 255].repeat(6));
        }
    }

    #[test]
    fn projection_rejects_invalid_layout_and_factor_rows() {
        let rank_three = Array::zeros::<f32>(&[4, 2, 3]).unwrap();
        let error = project_latents(&rank_three, &[[0.0; 3]; 4], [0.0; 3]).unwrap_err();
        assert!(error.to_string().contains("[1, C, h, w]"));

        let batched = Array::zeros::<f32>(&[2, 4, 2, 3]).unwrap();
        let error = project_latents(&batched, &[[0.0; 3]; 4], [0.0; 3]).unwrap_err();
        assert!(error.to_string().contains("[1, C, h, w]"));

        let four_channels = Array::zeros::<f32>(&[1, 4, 2, 3]).unwrap();
        let error = project_latents(&four_channels, &[[0.0; 3]; 3], [0.0; 3]).unwrap_err();
        assert!(error
            .to_string()
            .contains("3 does not match latent channel count 4"));
    }

    #[test]
    fn emit_preview_swallows_projection_errors_and_continues_the_schedule() {
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        let counter = PreviewCounter::new(&SIGMAS);
        let latents = Array::zeros::<f32>(&[1, 4, 2, 3]).unwrap();

        emit_preview(&sink, &counter, &SIGMAS, SIGMAS[0], || {
            Err(Error::Msg("synthetic projection failure".into()))
        });
        emit_preview(&sink, &counter, &SIGMAS, SIGMAS[1], || {
            project_latents(&latents, &[[0.0; 3]; 4], [0.0; 3])
        });

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].current, 2);
        assert_eq!(frames[0].total, 8);
    }

    #[test]
    fn sigma_aware_emission_forwards_the_schedule_sigma() {
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        let counter = PreviewCounter::new(&SIGMAS);
        let observed = Cell::new(None);
        let latents = Array::zeros::<f32>(&[1, 4, 2, 3]).unwrap();

        emit_preview_with_sigma(&sink, &counter, &SIGMAS, SIGMAS[2], |sigma| {
            observed.set(sigma);
            project_latents(&latents, &[[0.0; 3]; 4], [0.0; 3])
        });

        assert_eq!(observed.get(), Some(SIGMAS[2]));
        assert_eq!(frames.lock().unwrap().len(), 1);
    }

    #[test]
    fn inert_sink_does_not_advance_the_counter() {
        let counter = PreviewCounter::new(&SIGMAS);
        emit_preview(
            &PreviewSink::default(),
            &counter,
            &SIGMAS,
            SIGMAS[0],
            || panic!("an inert preview sink must not invoke projection"),
        );
        assert_eq!(counter.next(&SIGMAS, SIGMAS[0]), Some(1));
    }

    #[test]
    fn projection_rejects_zero_channel_or_spatial_dimensions() {
        for shape in [[1, 0, 2, 3], [1, 4, 0, 3], [1, 4, 2, 0]] {
            let latents = Array::zeros::<f32>(&shape).unwrap();
            let factors = vec![[0.0; 3]; shape[1] as usize];
            let error = project_latents(&latents, &factors, [0.0; 3]).unwrap_err();
            assert!(error.to_string().contains("must be non-zero"));
        }
    }
}
