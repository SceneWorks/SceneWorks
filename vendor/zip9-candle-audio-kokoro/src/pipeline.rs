//! The assembled Kokoro synthesis pipeline (reference `KModel.forward_with_tokens`,
//! sc-12836): phoneme ids + voice style vector + speed → 24 kHz mono samples.
//!
//! Stage order (each a cancellation/progress boundary for the provider):
//!
//! 1. PLBERT contextual encoding + `bert_encoder` projection,
//! 2. duration prediction (`DurationEncoder` → BiLSTM → `duration_proj` → sigmoid-sum),
//! 3. alignment expansion + F0/energy prediction,
//! 4. text encoding + alignment (`t_en @ aln`),
//! 5. decoder + iSTFT-Net vocoder.

use std::path::Path;

use candle_audio::candle_core::{DType, Device, IndexOp, Tensor};
use candle_audio::{AudioError, Result};
use candle_nn::{linear, Linear, Module};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::albert::Plbert;
use crate::config::KokoroConfig;
use crate::decoder::Decoder;
use crate::predictor::ProsodyPredictor;
use crate::text_encoder::TextEncoder;
use crate::weights::{section_var_builder, split_ref_s};

/// The pipeline's checkpoint filename inside a `hexgrad/Kokoro-82M` snapshot.
pub const CHECKPOINT_FILE: &str = "kokoro-v1_0.pth";

/// Number of synthesis stages (the provider's `Progress::Step` total).
pub const STAGES: u32 = 5;

/// One waveform frame per duration unit: `prod(upsample_rates) · gen_istft_hop_size ·
/// F0-upsample(×2)` = 600 samples at 24 kHz → 40 duration frames per second.
pub const SAMPLES_PER_FRAME: usize = 600;

/// Called after each completed stage (`1..=STAGES`); returning an error (typically
/// [`AudioError::Canceled`]) aborts synthesis.
pub type StageSink<'a> = &'a mut dyn FnMut(u32) -> Result<()>;

pub struct KokoroPipeline {
    pub config: KokoroConfig,
    bert: Plbert,
    bert_encoder: Linear,
    predictor: ProsodyPredictor,
    text_encoder: TextEncoder,
    decoder: Decoder,
    device: Device,
}

impl KokoroPipeline {
    /// Assemble the pipeline from a snapshot directory holding `config.json` +
    /// `kokoro-v1_0.pth`.
    pub fn from_snapshot(root: &Path, device: &Device) -> Result<Self> {
        let config = KokoroConfig::from_file(&root.join("config.json"))?;
        let pth = root.join(CHECKPOINT_FILE);
        if !pth.is_file() {
            return Err(AudioError::Msg(format!(
                "kokoro snapshot {} has no {CHECKPOINT_FILE}",
                root.display()
            )));
        }
        let bert = Plbert::new(
            config.n_token,
            &config.plbert,
            section_var_builder(&pth, "bert", device)?,
        )?;
        let bert_encoder = linear(
            config.plbert.hidden_size,
            config.hidden_dim,
            section_var_builder(&pth, "bert_encoder", device)?,
        )?;
        let predictor = ProsodyPredictor::new(
            config.style_dim,
            config.hidden_dim,
            config.n_layer,
            config.max_dur,
            section_var_builder(&pth, "predictor", device)?,
        )?;
        let text_encoder = TextEncoder::new(
            config.hidden_dim,
            config.text_encoder_kernel_size,
            config.n_layer,
            config.n_token,
            section_var_builder(&pth, "text_encoder", device)?,
        )?;
        let decoder = Decoder::new(
            config.hidden_dim,
            config.style_dim,
            &config.istftnet,
            section_var_builder(&pth, "decoder", device)?,
        )?;
        Ok(Self {
            config,
            bert,
            bert_encoder,
            predictor,
            text_encoder,
            decoder,
            device: device.clone(),
        })
    }

    /// Synthesize `tokens` (phoneme ids WITHOUT the boundary sentinels — they are added here)
    /// with the voice's 256-wide `ref_s` style row. `speed` > 1 speaks faster. Deterministic
    /// per `seed`. `cancel` is a cheap cooperative-cancellation probe threaded INSIDE the
    /// dominant stage-5 decoder/vocoder (the stage boundaries alone are too coarse there);
    /// when it fires, synthesis returns the typed [`AudioError::Canceled`].
    pub fn synthesize(
        &self,
        tokens: &[u32],
        ref_s: &[f32],
        speed: f32,
        seed: u64,
        stage: StageSink<'_>,
        cancel: crate::decoder::CancelProbe<'_>,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(AudioError::Msg(
                "kokoro: no phonemes survived tokenization (unsupported script?)".into(),
            ));
        }
        if tokens.len() + 2 > self.bert.context_length() {
            return Err(AudioError::Msg(format!(
                "kokoro: {} phoneme tokens exceed the model context of {} (split the script)",
                tokens.len() + 2,
                self.bert.context_length()
            )));
        }
        if !(0.1..=10.0).contains(&speed) || !speed.is_finite() {
            return Err(AudioError::Msg(format!(
                "kokoro: speed {speed} out of range"
            )));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let (style_decoder, style_prosody) = split_ref_s(ref_s, &self.device)?;

        // Stage 1: PLBERT + projection. Boundary sentinels (id 0) wrap the tokens.
        let mut ids: Vec<u32> = Vec::with_capacity(tokens.len() + 2);
        ids.push(0);
        ids.extend_from_slice(tokens);
        ids.push(0);
        let t = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, t), &self.device)?;
        let bert_dur = self.bert.forward(&input_ids)?; // [1, T, 768]
        let d_en = self
            .bert_encoder
            .forward(&bert_dur)? // [1, T, 512]
            .transpose(1, 2)?
            .contiguous()?; // [1, 512, T]
        stage(1)?;

        // Stage 2: durations.
        let d = self.predictor.text_encoder.forward(&d_en, &style_prosody)?; // [1, T, 640]
        let raw_durations = self.predictor.durations(&d)?;
        let pred_dur: Vec<usize> = raw_durations
            .iter()
            .map(|&x| ((x / speed).round_ties_even().max(1.0)) as usize)
            .collect();
        let total_frames: usize = pred_dur.iter().sum();
        stage(2)?;

        // Frame → token alignment map (the reference's one-hot `pred_aln_trg`, applied as a
        // repeat-gather instead of a matmul).
        let frame_token: Vec<usize> = pred_dur
            .iter()
            .enumerate()
            .flat_map(|(t, &n)| std::iter::repeat_n(t, n))
            .collect();

        // Stage 3: F0 / energy over the aligned duration features.
        let en = gather_frames(&d.i(0)?.to_vec2::<f32>()?, &frame_token, &self.device)?;
        let (f0, n_curve) = self.predictor.f0n_train(&en, &style_prosody)?;
        stage(3)?;

        // Stage 4: text encoding + alignment.
        let t_en = self.text_encoder.forward(&input_ids)?; // [1, 512, T]
        let t_en_rows = t_en.i(0)?.t()?.to_vec2::<f32>()?; // [T][512]
        let asr = gather_frames(&t_en_rows, &frame_token, &self.device)?; // [1, 512, F]
        stage(4)?;

        // Stage 5: decoder + vocoder (cancellation probed inside — the dominant-cost stage).
        let samples =
            self.decoder
                .forward(&asr, &f0, &n_curve, &style_decoder, &mut rng, cancel)?;
        if samples.len() != total_frames * SAMPLES_PER_FRAME {
            return Err(AudioError::Msg(format!(
                "kokoro: vocoder produced {} samples for {} frames (expected {})",
                samples.len(),
                total_frames,
                total_frames * SAMPLES_PER_FRAME
            )));
        }
        if !samples.iter().all(|s| s.is_finite()) {
            return Err(AudioError::Msg("kokoro: non-finite samples".into()));
        }
        stage(5)?;
        Ok(samples)
    }

    /// Natural (speed-1) duration estimate in seconds for `tokens` — the duration head only,
    /// used to honor `target_duration` by deriving a speed factor.
    pub fn natural_duration_secs(&self, raw_durations: &[f32]) -> f32 {
        let frames: f32 = raw_durations
            .iter()
            .map(|&x| x.round_ties_even().max(1.0))
            .sum();
        frames * SAMPLES_PER_FRAME as f32 / crate::decoder::SAMPLE_RATE as f32
    }

    /// Run stages 1–2 only and return the raw (speed-1) per-token durations — the cheap probe
    /// [`crate::model`] uses to derive a speed factor for `target_duration`.
    pub fn raw_durations(&self, tokens: &[u32], ref_s: &[f32]) -> Result<Vec<f32>> {
        let mut ids: Vec<u32> = Vec::with_capacity(tokens.len() + 2);
        ids.push(0);
        ids.extend_from_slice(tokens);
        ids.push(0);
        let t = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, t), &self.device)?;
        let (_, style_prosody) = split_ref_s(ref_s, &self.device)?;
        let bert_dur = self.bert.forward(&input_ids)?;
        let d_en = self
            .bert_encoder
            .forward(&bert_dur)?
            .transpose(1, 2)?
            .contiguous()?;
        let d = self.predictor.text_encoder.forward(&d_en, &style_prosody)?;
        self.predictor.durations(&d)
    }
}

/// Expand per-token rows (`[T][C]`) into channel-major aligned frames `[1, C, F]` via the
/// frame→token map (the `x @ pred_aln_trg` product, batch-1).
fn gather_frames(rows: &[Vec<f32>], frame_token: &[usize], device: &Device) -> Result<Tensor> {
    let f = frame_token.len();
    if f == 0 {
        return Err(AudioError::Msg("kokoro: empty alignment".into()));
    }
    let c = rows[0].len();
    let mut flat = vec![0.0f32; c * f];
    for (fi, &ti) in frame_token.iter().enumerate() {
        let row = &rows[ti];
        for (ci, &v) in row.iter().enumerate() {
            flat[ci * f + fi] = v;
        }
    }
    Ok(Tensor::from_vec(flat, (1, c, f), device)?.to_dtype(DType::F32)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_frames_repeats_token_rows() {
        let dev = Device::Cpu;
        let rows = vec![vec![1.0f32, 10.0], vec![2.0, 20.0]];
        let t = gather_frames(&rows, &[0, 0, 1], &dev).unwrap();
        assert_eq!(t.dims(), &[1, 2, 3]);
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, [1.0, 1.0, 2.0, 10.0, 10.0, 20.0]);
    }
}
