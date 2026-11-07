//! StyleTTS2 `ProsodyPredictor` (modules.py) — the duration / F0 / energy heads (sc-12836).
//!
//! - [`DurationEncoder`]: 3 × (BiLSTM over `[text ‖ style]` + AdaLayerNorm, re-appending the
//!   style after each norm) producing the 640-wide `d` features.
//! - Duration head: BiLSTM + `duration_proj` linear → per-token `sigmoid(…).sum(-1)` frame
//!   counts (each frame = 600 samples = 25 ms at 24 kHz).
//! - [`ProsodyPredictor::f0n_train`]: the shared BiLSTM over aligned features, then the F0 and
//!   N (energy) `AdainResBlk1d` stacks (the middle block upsamples time ×2) and their 1×1
//!   projections — the pitch/energy curves the vocoder's harmonic source consumes.
//!
//! Batch-1, no padding mask (single unpadded sequence). Dropout layers are inference no-ops.

use candle_audio::candle_core::{IndexOp, Tensor};
use candle_audio::Result;
use candle_nn::{conv1d, linear, Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

use crate::nn::{AdaLayerNorm, AdainResBlk1d, BiLstm};

/// `DurationEncoder`: alternating BiLSTM / AdaLayerNorm blocks over `[d_model + style]`.
pub struct DurationEncoder {
    /// `(lstm, ada_layer_norm)` pairs, in checkpoint order (`lstms.0` = LSTM, `lstms.1` =
    /// AdaLayerNorm, `lstms.2` = LSTM, …).
    blocks: Vec<(BiLstm, AdaLayerNorm)>,
}

impl DurationEncoder {
    pub fn new(sty_dim: usize, d_model: usize, nlayers: usize, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::with_capacity(nlayers);
        for layer in 0..nlayers {
            let lstm = BiLstm::new(
                d_model + sty_dim,
                d_model / 2,
                vb.pp(format!("lstms.{}", 2 * layer)),
            )?;
            let norm =
                AdaLayerNorm::new(sty_dim, d_model, vb.pp(format!("lstms.{}", 2 * layer + 1)))?;
            blocks.push((lstm, norm));
        }
        Ok(Self { blocks })
    }

    /// `x: [1, d_model, T]` (the bert-encoded text), `s: [1, sty_dim]` →
    /// `d: [1, T, d_model + sty_dim]` (style re-appended after the last norm — the reference
    /// output feeds both the duration LSTM and the alignment product).
    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let t = x.dim(2)?;
        // s expanded over time: [1, sty, T].
        let s_time = s
            .unsqueeze(2)?
            .broadcast_as((1, s.dim(1)?, t))?
            .contiguous()?;
        // Start: [1, d_model + sty, T].
        let mut x = Tensor::cat(&[x, &s_time], 1)?;
        for (lstm, norm) in &self.blocks {
            // LSTM over [1, T, C].
            let seq = x.transpose(1, 2)?.contiguous()?;
            let out = lstm.forward(&seq)?; // [1, T, d_model]
            let out = out.transpose(1, 2)?.contiguous()?; // [1, d_model, T]
                                                          // AdaLayerNorm + re-append style on the channel dim.
            let normed = norm.forward(&out, s)?;
            x = Tensor::cat(&[&normed, &s_time], 1)?;
        }
        Ok(x.transpose(1, 2)?.contiguous()?)
    }
}

pub struct ProsodyPredictor {
    pub text_encoder: DurationEncoder,
    lstm: BiLstm,
    duration_proj: Linear,
    shared: BiLstm,
    f0_blocks: Vec<AdainResBlk1d>,
    n_blocks: Vec<AdainResBlk1d>,
    f0_proj: Conv1d,
    n_proj: Conv1d,
}

impl ProsodyPredictor {
    pub fn new(
        style_dim: usize,
        d_hid: usize,
        nlayers: usize,
        max_dur: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let text_encoder = DurationEncoder::new(style_dim, d_hid, nlayers, vb.pp("text_encoder"))?;
        let lstm = BiLstm::new(d_hid + style_dim, d_hid / 2, vb.pp("lstm"))?;
        let duration_proj = linear(d_hid, max_dur, vb.pp("duration_proj.linear_layer"))?;
        let shared = BiLstm::new(d_hid + style_dim, d_hid / 2, vb.pp("shared"))?;
        let build_stack = |name: &str| -> Result<Vec<AdainResBlk1d>> {
            Ok(vec![
                AdainResBlk1d::new(d_hid, d_hid, style_dim, false, vb.pp(format!("{name}.0")))?,
                AdainResBlk1d::new(
                    d_hid,
                    d_hid / 2,
                    style_dim,
                    true,
                    vb.pp(format!("{name}.1")),
                )?,
                AdainResBlk1d::new(
                    d_hid / 2,
                    d_hid / 2,
                    style_dim,
                    false,
                    vb.pp(format!("{name}.2")),
                )?,
            ])
        };
        let f0_blocks = build_stack("F0")?;
        let n_blocks = build_stack("N")?;
        let f0_proj = conv1d(d_hid / 2, 1, 1, Conv1dConfig::default(), vb.pp("F0_proj"))?;
        let n_proj = conv1d(d_hid / 2, 1, 1, Conv1dConfig::default(), vb.pp("N_proj"))?;
        Ok(Self {
            text_encoder,
            lstm,
            duration_proj,
            shared,
            f0_blocks,
            n_blocks,
            f0_proj,
            n_proj,
        })
    }

    /// Per-token raw frame durations from the duration-encoder features `d: [1, T, 640]` —
    /// `sigmoid(duration_proj(lstm(d))).sum(-1)`, before the speed divide / rounding.
    pub fn durations(&self, d: &Tensor) -> Result<Vec<f32>> {
        let x = self.lstm.forward(d)?; // [1, T, d_hid]
        let logits = self.duration_proj.forward(&x)?; // [1, T, max_dur]
        let probs = candle_nn::ops::sigmoid(&logits)?;
        let dur = probs.sum(2)?.i(0)?; // [T]
        Ok(dur.to_vec1()?)
    }

    /// The F0/N heads over the aligned features `en: [1, 640, F]` and prosody style `s` →
    /// `(f0, n)` curves, each `[2·F]` (the middle block upsamples time ×2).
    pub fn f0n_train(&self, en: &Tensor, s: &Tensor) -> Result<(Vec<f32>, Vec<f32>)> {
        let x = self.shared.forward(&en.transpose(1, 2)?.contiguous()?)?; // [1, F, d_hid]
        let x = x.transpose(1, 2)?.contiguous()?; // [1, d_hid, F]

        let mut f0 = x.clone();
        for block in &self.f0_blocks {
            f0 = block.forward(&f0, s)?;
        }
        let f0 = self.f0_proj.forward(&f0)?; // [1, 1, 2F]
        let mut n = x;
        for block in &self.n_blocks {
            n = block.forward(&n, s)?;
        }
        let n = self.n_proj.forward(&n)?; // [1, 1, 2F]
        Ok((f0.flatten_all()?.to_vec1()?, n.flatten_all()?.to_vec1()?))
    }
}
