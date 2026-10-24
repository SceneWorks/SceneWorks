//! Shared neural building blocks for the Kokoro port (sc-12836) — faithful candle ports of the
//! StyleTTS2 modules the acoustic model and vocoder are assembled from
//! (`hexgrad/kokoro`'s `modules.py` / `istftnet.py`):
//!
//! - [`BiLstm`] — a single-layer bidirectional LSTM over `[1, T, C]` (torch `nn.LSTM(…,
//!   bidirectional=True)`; the checkpoint's `weight_ih_l0` / `…_l0_reverse` naming maps onto two
//!   candle LSTMs).
//! - [`ChannelLayerNorm`] — StyleTTS2's `LayerNorm` over the channel dim of `[1, C, T]` with
//!   `gamma` / `beta` parameter names.
//! - [`AdaLayerNorm`] — layer norm (no affine) + style-projected `(1 + γ)·x + β`.
//! - [`AdaIn1d`] — instance norm (no affine — the checkpoint carries no norm weights; the
//!   reference loads with `strict=False` leaving its `affine=True` params at identity) + the
//!   style-projected affine.
//! - [`AdainResBlk1d`] — the styled residual block (optional nearest×2 upsample + depthwise
//!   transposed-conv pool) used by the prosody predictor F0/N heads and the decoder.
//! - [`AdaInResBlock1`] — the vocoder's Snake-activated styled resblock (HiFi-GAN shape).
//!
//! All blocks run batch-1 inference; weight-norm pairs are already resolved to plain `weight`
//! tensors by [`crate::weights`].

use candle_audio::candle_core::{Device, IndexOp, Tensor};
use candle_audio::ops::nearest_upsample1d;
use candle_audio::Result;
use candle_nn::{
    conv1d, conv1d_no_bias, conv_transpose1d, linear, lstm, ops, Conv1d, Conv1dConfig,
    ConvTranspose1d, ConvTranspose1dConfig, LSTMConfig, Linear, Module, LSTM, RNN,
};

/// LeakyReLU slope used across StyleTTS2 blocks (`nn.LeakyReLU(0.2)`).
pub const LRELU_SLOPE_BLOCKS: f64 = 0.2;
/// LeakyReLU slope inside the vocoder generator (`F.leaky_relu(x, 0.1)`).
pub const LRELU_SLOPE_GENERATOR: f64 = 0.1;

/// Reverse a `[1, T, C]` tensor along its time dimension.
pub fn reverse_time(x: &Tensor) -> Result<Tensor> {
    let t = x.dim(1)?;
    let idx: Vec<u32> = (0..t as u32).rev().collect();
    let idx = Tensor::from_vec(idx, t, x.device())?;
    Ok(x.index_select(&idx, 1)?)
}

/// A single-layer bidirectional LSTM (`nn.LSTM(input, hidden, 1, batch_first=True,
/// bidirectional=True)`): forward and reverse passes concatenated on the feature dim.
pub struct BiLstm {
    fwd: LSTM,
    bwd: LSTM,
}

impl BiLstm {
    /// Build from a var-builder holding torch names (`weight_ih_l0`, `weight_hh_l0`,
    /// `bias_ih_l0`, `bias_hh_l0` and their `_reverse` twins).
    pub fn new(in_dim: usize, hidden_dim: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        let fwd = lstm(in_dim, hidden_dim, LSTMConfig::default(), vb.clone())?;
        let bwd = lstm(
            in_dim,
            hidden_dim,
            LSTMConfig {
                direction: candle_nn::rnn::Direction::Backward,
                ..LSTMConfig::default()
            },
            vb,
        )?;
        Ok(Self { fwd, bwd })
    }

    /// `[1, T, in] → [1, T, 2·hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let states = self.fwd.seq(x)?;
        let fwd_out = self.fwd.states_to_tensor(&states)?; // [1, T, hidden]
        let rev_in = reverse_time(x)?;
        let states = self.bwd.seq(&rev_in)?;
        let bwd_out = reverse_time(&self.bwd.states_to_tensor(&states)?)?;
        Ok(Tensor::cat(&[&fwd_out, &bwd_out], 2)?)
    }
}

/// StyleTTS2's `LayerNorm` (modules.py): normalize the channel dim of `[1, C, T]` with
/// `gamma` / `beta` affine parameters, eps `1e-5`.
pub struct ChannelLayerNorm {
    gamma: Tensor,
    beta: Tensor,
}

impl ChannelLayerNorm {
    pub fn new(channels: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        Ok(Self {
            gamma: vb.get(channels, "gamma")?,
            beta: vb.get(channels, "beta")?,
        })
    }

    /// `[1, C, T] → [1, C, T]` (normalized over C).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xt = x.transpose(1, 2)?; // [1, T, C]
        let mean = xt.mean_keepdim(2)?;
        let centered = xt.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(2)?;
        let normed = centered.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
        let out = normed
            .broadcast_mul(&self.gamma)?
            .broadcast_add(&self.beta)?;
        Ok(out.transpose(1, 2)?)
    }
}

/// Layer-norm a `[1, T, C]` tensor over its last dim without affine (eps `1e-5`).
fn layer_norm_no_affine(x: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(2)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(2)?;
    Ok(centered.broadcast_div(&(var + 1e-5)?.sqrt()?)?)
}

/// `AdaLayerNorm` (modules.py): style-conditioned layer norm over channels.
pub struct AdaLayerNorm {
    fc: Linear,
}

impl AdaLayerNorm {
    pub fn new(style_dim: usize, channels: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        Ok(Self {
            fc: linear(style_dim, channels * 2, vb.pp("fc"))?,
        })
    }

    /// `x: [1, C, T]`, `s: [1, style_dim] → [1, C, T]`.
    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let h = self.fc.forward(s)?; // [1, 2C]
        let c = h.dim(1)? / 2;
        let gamma = h.i((.., ..c))?.unsqueeze(1)?; // [1, 1, C]
        let beta = h.i((.., c..))?.unsqueeze(1)?;
        let xt = x.transpose(1, 2)?; // [1, T, C]
        let normed = layer_norm_no_affine(&xt)?;
        let out = normed
            .broadcast_mul(&(gamma + 1.0)?)?
            .broadcast_add(&beta)?;
        Ok(out.transpose(1, 2)?)
    }
}

/// `AdaIN1d` (istftnet.py): instance norm over time (no affine — see module docs) + the
/// style-projected `(1 + γ)·x̂ + β` affine.
pub struct AdaIn1d {
    fc: Linear,
}

impl AdaIn1d {
    pub fn new(style_dim: usize, num_features: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        Ok(Self {
            fc: linear(style_dim, num_features * 2, vb.pp("fc"))?,
        })
    }

    /// `x: [1, C, T]`, `s: [1, style_dim] → [1, C, T]`.
    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        // Instance norm: per-channel over time, biased variance, eps 1e-5.
        let mean = x.mean_keepdim(2)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(2)?;
        let normed = centered.broadcast_div(&(var + 1e-5)?.sqrt()?)?;

        let h = self.fc.forward(s)?; // [1, 2C]
        let c = h.dim(1)? / 2;
        let gamma = h.i((.., ..c))?.unsqueeze(2)?; // [1, C, 1]
        let beta = h.i((.., c..))?.unsqueeze(2)?;
        Ok(normed
            .broadcast_mul(&(gamma + 1.0)?)?
            .broadcast_add(&beta)?)
    }
}

/// `AdainResBlk1d` (istftnet.py): the styled residual block. `upsample` doubles time via
/// nearest interpolation on the shortcut and a depthwise stride-2 transposed conv (`pool`) on
/// the residual path; a `conv1x1` shortcut projects when `dim_in != dim_out`. Output is
/// `(residual + shortcut) / √2`.
pub struct AdainResBlk1d {
    conv1: Conv1d,
    conv2: Conv1d,
    norm1: AdaIn1d,
    norm2: AdaIn1d,
    conv1x1: Option<Conv1d>,
    pool: Option<ConvTranspose1d>,
}

impl AdainResBlk1d {
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        style_dim: usize,
        upsample: bool,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let conv_cfg = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = conv1d(dim_in, dim_out, 3, conv_cfg, vb.pp("conv1"))?;
        let conv2 = conv1d(dim_out, dim_out, 3, conv_cfg, vb.pp("conv2"))?;
        let norm1 = AdaIn1d::new(style_dim, dim_in, vb.pp("norm1"))?;
        let norm2 = AdaIn1d::new(style_dim, dim_out, vb.pp("norm2"))?;
        let conv1x1 = if dim_in != dim_out {
            Some(conv1d_no_bias(
                dim_in,
                dim_out,
                1,
                Conv1dConfig::default(),
                vb.pp("conv1x1"),
            )?)
        } else {
            None
        };
        let pool = if upsample {
            Some(conv_transpose1d(
                dim_in,
                dim_in,
                3,
                ConvTranspose1dConfig {
                    padding: 1,
                    output_padding: 1,
                    stride: 2,
                    dilation: 1,
                    groups: dim_in,
                },
                vb.pp("pool"),
            )?)
        } else {
            None
        };
        Ok(Self {
            conv1,
            conv2,
            norm1,
            norm2,
            conv1x1,
            pool,
        })
    }

    fn shortcut(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.pool.is_some() {
            // Nearest ×2 time upsample. candle's CUDA/Metal backends don't implement
            // `upsample_nearest1d` (sc-13886 / sc-13691), so route through the backend-agnostic,
            // bit-identical shared helper instead.
            nearest_upsample1d(x, 2)?
        } else {
            x.clone()
        };
        Ok(match &self.conv1x1 {
            Some(c) => c.forward(&x)?,
            None => x,
        })
    }

    fn residual(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let x = self.norm1.forward(x, s)?;
        let x = ops::leaky_relu(&x, LRELU_SLOPE_BLOCKS)?;
        let x = match &self.pool {
            Some(p) => p.forward(&x)?,
            None => x,
        };
        let x = self.conv1.forward(&x)?;
        let x = self.norm2.forward(&x, s)?;
        let x = ops::leaky_relu(&x, LRELU_SLOPE_BLOCKS)?;
        Ok(self.conv2.forward(&x)?)
    }

    /// `x: [1, C_in, T]`, `s: [1, style_dim] → [1, C_out, T·(1|2)]`.
    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let out = (self.residual(x, s)? + self.shortcut(x)?)?;
        Ok((out * (1.0 / 2f64.sqrt()))?)
    }
}

/// Snake activation `x + sin²(αx)/α` (the vocoder's periodic activation; istftnet.py inline).
fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let s = x.broadcast_mul(alpha)?.sin()?.sqr()?;
    Ok(x.broadcast_add(&s.broadcast_div(alpha)?)?)
}

/// `AdaINResBlock1` (istftnet.py): the vocoder's HiFi-GAN-shaped styled resblock — three
/// dilated/plain conv pairs, each conv preceded by AdaIN + Snake.
pub struct AdaInResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIn1d>,
    adain2: Vec<AdaIn1d>,
    alpha1: Vec<Tensor>,
    alpha2: Vec<Tensor>,
}

impl AdaInResBlock1 {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        style_dim: usize,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let get_padding = |d: usize| (kernel_size * d - d) / 2;
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut adain1 = Vec::new();
        let mut adain2 = Vec::new();
        let mut alpha1 = Vec::new();
        let mut alpha2 = Vec::new();
        for (i, &d) in dilations.iter().enumerate() {
            convs1.push(conv1d(
                channels,
                channels,
                kernel_size,
                Conv1dConfig {
                    padding: get_padding(d),
                    dilation: d,
                    ..Default::default()
                },
                vb.pp(format!("convs1.{i}")),
            )?);
            convs2.push(conv1d(
                channels,
                channels,
                kernel_size,
                Conv1dConfig {
                    padding: get_padding(1),
                    ..Default::default()
                },
                vb.pp(format!("convs2.{i}")),
            )?);
            adain1.push(AdaIn1d::new(
                style_dim,
                channels,
                vb.pp(format!("adain1.{i}")),
            )?);
            adain2.push(AdaIn1d::new(
                style_dim,
                channels,
                vb.pp(format!("adain2.{i}")),
            )?);
            alpha1.push(vb.get((1, channels, 1), &format!("alpha1.{i}"))?);
            alpha2.push(vb.get((1, channels, 1), &format!("alpha2.{i}"))?);
        }
        Ok(Self {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
        })
    }

    /// `x: [1, C, T]`, `s: [1, style_dim] → [1, C, T]`.
    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let xt = self.adain1[i].forward(&x, s)?;
            let xt = snake(&xt, &self.alpha1[i])?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = self.adain2[i].forward(&xt, s)?;
            let xt = snake(&xt, &self.alpha2[i])?;
            let xt = self.convs2[i].forward(&xt)?;
            x = (xt + x)?;
        }
        Ok(x)
    }
}

/// Left-reflection-pad a `[1, C, T]` tensor by one sample (`nn.ReflectionPad1d((1, 0))`).
pub fn reflection_pad_left1(x: &Tensor) -> Result<Tensor> {
    let first_two = x.i((.., .., 1..2))?; // reflect: index 1 mirrored to the left of index 0
    Ok(Tensor::cat(&[&first_two, x], 2)?)
}

/// Build a `[1, C]` tensor on `device` from a slice (style vectors, small vectors).
pub fn row_tensor(values: &[f32], device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_slice(values, (1, values.len()), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::{DType, Device};
    use std::collections::HashMap;

    fn vb_from(pairs: Vec<(&str, Tensor)>) -> candle_nn::VarBuilder<'static> {
        let map: HashMap<String, Tensor> =
            pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    #[test]
    fn reverse_time_reverses() {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 3, 1), &dev).unwrap();
        let r: Vec<f32> = reverse_time(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(r, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn channel_layer_norm_normalizes_channels() {
        let dev = Device::Cpu;
        let vb = vb_from(vec![
            ("gamma", Tensor::ones(2, DType::F32, &dev).unwrap()),
            ("beta", Tensor::zeros(2, DType::F32, &dev).unwrap()),
        ]);
        let ln = ChannelLayerNorm::new(2, vb).unwrap();
        // x[1, 2, 1] with channels [3, 5] → normalized to [-1, 1].
        let x = Tensor::from_slice(&[3.0f32, 5.0], (1, 2, 1), &dev).unwrap();
        let y: Vec<f32> = ln
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            (y[0] + 1.0).abs() < 1e-3 && (y[1] - 1.0).abs() < 1e-3,
            "{y:?}"
        );
    }

    #[test]
    fn snake_is_identity_at_zero() {
        let dev = Device::Cpu;
        let alpha = Tensor::ones((1, 1, 1), DType::F32, &dev).unwrap();
        let x = Tensor::zeros((1, 1, 4), DType::F32, &dev).unwrap();
        let y: Vec<f32> = snake(&x, &alpha)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(y, [0.0; 4]);
    }

    #[test]
    fn reflection_pad_left_mirrors_index_one() {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[10.0f32, 20.0, 30.0], (1, 1, 3), &dev).unwrap();
        let y: Vec<f32> = reflection_pad_left1(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(y, [20.0, 10.0, 20.0, 30.0]);
    }
}
