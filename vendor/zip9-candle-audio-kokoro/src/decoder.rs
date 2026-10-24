//! StyleTTS2 `Decoder` + iSTFT-Net `Generator` (istftnet.py) — aligned text features + F0/N
//! curves + decoder style → 24 kHz waveform (sc-12836).
//!
//! The conv/resblock stacks run on candle tensors; the harmonic-source synthesis
//! (`SourceModuleHnNSF`) and the tiny n_fft=20 forward/inverse STFTs are host `f32` DSP —
//! `n_fft = 20` is not a power of two, so the radix-2 helpers in `candle_audio::dsp` do not
//! apply and a naive O(n²) real DFT (11 bins × 20 points) is used instead; at 20 points it is
//! costless and numerically exact.
//!
//! Reference-faithfulness notes:
//! - The sine generator's random initial harmonic phase is dropped: in the reference
//!   `SineGen._f02sine` (non-pulse path) the `rand_ini` perturbation lands on sample 0 only and
//!   the immediately following ×1/300 linear downsample never samples index 0, so it has no
//!   effect on the output.
//! - The additive source noise (`noise_amp · randn`) IS kept, drawn from the request-seeded RNG
//!   — the reproducibility law (same request + seed ⇒ byte-identical samples) holds.
//! - `phase = sin(conv_post's upper bins)` (bounded phase) is the reference behavior, ported
//!   as-is.

use candle_audio::candle_core::{IndexOp, Tensor};
use candle_audio::dsp::hann_window;
use candle_audio::{AudioError, Result};
use candle_nn::{
    conv1d, conv_transpose1d, ops, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig,
    Module, VarBuilder,
};
use rand::rngs::StdRng;
use rand::Rng;

use crate::config::IstftNetConfig;
use crate::nn::{reflection_pad_left1, AdaInResBlock1, AdainResBlk1d, LRELU_SLOPE_GENERATOR};

/// Kokoro's native output sample rate (Hz).
pub const SAMPLE_RATE: u32 = 24_000;

/// A cheap cooperative-cancellation probe threaded through the decoder/vocoder — the vocoder
/// dominates synthesis cost, so cancellation must be observable *inside* it, not only at the
/// pipeline's stage boundaries (sc-12836 review). Returns `true` when the request is cancelled.
pub type CancelProbe<'a> = &'a dyn Fn() -> bool;

/// Bail with the typed [`AudioError::Canceled`] when the probe fires.
fn check_cancel(cancel: CancelProbe<'_>) -> Result<()> {
    if cancel() {
        return Err(AudioError::Canceled);
    }
    Ok(())
}

/// SineGen amplitude / noise constants (istftnet.py `SourceModuleHnNSF` defaults as
/// instantiated by `Generator`: harmonic_num = 8, voiced threshold = 10).
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;
const HARMONICS: usize = 9; // fundamental + 8 overtones

// ---------------------------------------------------------------------------------------------
// Host DSP: naive real DFT STFT pair for the vocoder's tiny n_fft (20).
// ---------------------------------------------------------------------------------------------

/// Precomputed cos/sin tables for an `n`-point one-sided real DFT.
struct SmallRdft {
    n: usize,
    n_bins: usize,
    cos: Vec<f32>, // [n_bins][n]
    sin: Vec<f32>,
}

impl SmallRdft {
    fn new(n: usize) -> Self {
        let n_bins = n / 2 + 1;
        let mut cos = vec![0.0f32; n_bins * n];
        let mut sin = vec![0.0f32; n_bins * n];
        for k in 0..n_bins {
            for j in 0..n {
                let ang = -2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
                cos[k * n + j] = ang.cos() as f32;
                sin[k * n + j] = ang.sin() as f32;
            }
        }
        Self {
            n,
            n_bins,
            cos,
            sin,
        }
    }

    /// Forward DFT of one windowed frame → (re, im) per one-sided bin.
    fn forward(&self, frame: &[f32], re: &mut [f32], im: &mut [f32]) {
        for k in 0..self.n_bins {
            let (mut r, mut i) = (0.0f32, 0.0f32);
            for (j, &x) in frame.iter().enumerate().take(self.n) {
                r += x * self.cos[k * self.n + j];
                i += x * self.sin[k * self.n + j];
            }
            re[k] = r;
            im[k] = i;
        }
    }

    /// Inverse one-sided DFT of a bin set → an `n`-long real frame (with the 1/n scale).
    fn inverse(&self, re: &[f32], im: &[f32], frame: &mut [f32]) {
        let n = self.n;
        for (j, out) in frame.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..self.n_bins {
                // Conjugate-symmetric expansion: interior bins count twice. The tables carry the
                // forward (negative-angle) sines, so the inverse's `- im·sin(+θ)` term appears
                // here with a plus.
                let w = if k == 0 || k == n / 2 { 1.0 } else { 2.0 };
                acc += w * (re[k] * self.cos[k * n + j] + im[k] * self.sin[k * n + j]);
            }
            *out = acc / n as f32;
        }
    }
}

/// `torch.stft(center=True, pad_mode="reflect", onesided=True)` for a small n_fft: returns
/// `(magnitude, phase)` bin-major `[n_bins, n_frames]`.
fn stft_small(samples: &[f32], n_fft: usize, hop: usize) -> Result<(Vec<f32>, Vec<f32>, usize)> {
    if samples.len() <= n_fft / 2 {
        return Err(AudioError::Msg(format!(
            "vocoder stft: {} samples too short for n_fft {n_fft}",
            samples.len()
        )));
    }
    let window = hann_window(n_fft);
    let rdft = SmallRdft::new(n_fft);
    let pad = n_fft / 2;
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    padded.extend((1..=pad).rev().map(|i| samples[i]));
    padded.extend_from_slice(samples);
    padded.extend((0..pad).map(|i| samples[samples.len() - 2 - i]));

    let n_bins = n_fft / 2 + 1;
    let n_frames = 1 + (padded.len() - n_fft) / hop;
    let mut mag = vec![0.0f32; n_bins * n_frames];
    let mut phase = vec![0.0f32; n_bins * n_frames];
    let mut frame = vec![0.0f32; n_fft];
    let mut re = vec![0.0f32; n_bins];
    let mut im = vec![0.0f32; n_bins];
    for t in 0..n_frames {
        let start = t * hop;
        for (dst, (x, w)) in frame
            .iter_mut()
            .zip(padded[start..start + n_fft].iter().zip(&window))
        {
            *dst = x * w;
        }
        rdft.forward(&frame, &mut re, &mut im);
        for k in 0..n_bins {
            mag[k * n_frames + t] = (re[k] * re[k] + im[k] * im[k]).sqrt();
            phase[k * n_frames + t] = im[k].atan2(re[k]);
        }
    }
    Ok((mag, phase, n_frames))
}

/// `torch.istft(center=True)` for a small n_fft: bin-major magnitude/phase `[n_bins,
/// n_frames]` → time samples (windowed overlap-add, squared-window normalization, the
/// centering pad trimmed).
fn istft_small(
    magnitude: &[f32],
    phase: &[f32],
    n_frames: usize,
    n_fft: usize,
    hop: usize,
) -> Vec<f32> {
    let window = hann_window(n_fft);
    let rdft = SmallRdft::new(n_fft);
    let n_bins = n_fft / 2 + 1;
    let out_len = n_fft + (n_frames.saturating_sub(1)) * hop;
    let mut out = vec![0.0f32; out_len];
    let mut wsum = vec![0.0f32; out_len];
    let mut re = vec![0.0f32; n_bins];
    let mut im = vec![0.0f32; n_bins];
    let mut frame = vec![0.0f32; n_fft];
    for t in 0..n_frames {
        for k in 0..n_bins {
            let m = magnitude[k * n_frames + t];
            let p = phase[k * n_frames + t];
            re[k] = m * p.cos();
            im[k] = m * p.sin();
        }
        rdft.inverse(&re, &im, &mut frame);
        let start = t * hop;
        for (i, (x, w)) in frame.iter().zip(&window).enumerate() {
            out[start + i] += x * w;
            wsum[start + i] += w * w;
        }
    }
    for (x, w) in out.iter_mut().zip(&wsum) {
        if *w > 1e-8 {
            *x /= *w;
        }
    }
    let pad = n_fft / 2;
    let end = out_len.saturating_sub(pad);
    out[pad.min(end)..end].to_vec()
}

// ---------------------------------------------------------------------------------------------
// Harmonic source (SourceModuleHnNSF, host side).
// ---------------------------------------------------------------------------------------------

/// The reference `_f02sine` phase path, exact for a per-frame-constant F0 (which is what the
/// nearest ×`upsample_scale` `f0_upsamp` produces): per-frame `rad = (f0·k / sr) mod 1`,
/// cumulative phase at frame rate, then ×`upsample_scale` linear interpolation back to sample
/// rate (`align_corners=False` coordinates, edges clamped).
fn harmonic_sines(f0_frames: &[f32], upsample_scale: usize, harmonic: usize) -> Vec<f32> {
    let n_frames = f0_frames.len();
    let mut phase_frames = Vec::with_capacity(n_frames);
    let mut acc = 0.0f64;
    for &f0 in f0_frames {
        let rad = ((f0 as f64) * (harmonic as f64) / SAMPLE_RATE as f64).rem_euclid(1.0);
        acc += rad;
        phase_frames.push(acc * 2.0 * std::f64::consts::PI);
    }
    let n_out = n_frames * upsample_scale;
    let scale = upsample_scale as f64;
    let mut sines = Vec::with_capacity(n_out);
    for j in 0..n_out {
        let c = (j as f64 + 0.5) / scale - 0.5;
        let phase = if c <= 0.0 {
            phase_frames[0]
        } else if c >= (n_frames - 1) as f64 {
            phase_frames[n_frames - 1]
        } else {
            let i0 = c.floor() as usize;
            let frac = c - i0 as f64;
            phase_frames[i0] * (1.0 - frac) + phase_frames[i0 + 1] * frac
        };
        sines.push((phase * scale).sin() as f32);
    }
    sines
}

/// `SourceModuleHnNSF`: F0 frames (the model's per-frame pitch curve) → the merged harmonic
/// excitation at sample rate. `l_linear` is the checkpoint's 9→1 harmonic merge.
fn harmonic_source(
    f0_frames: &[f32],
    upsample_scale: usize,
    l_weight: &[f32],
    l_bias: f32,
    rng: &mut StdRng,
) -> Vec<f32> {
    let n_out = f0_frames.len() * upsample_scale;
    // Per-harmonic sine banks (voicing/noise applied on the merged fly below).
    let banks: Vec<Vec<f32>> = (1..=HARMONICS)
        .map(|k| harmonic_sines(f0_frames, upsample_scale, k))
        .collect();
    let mut out = Vec::with_capacity(n_out);
    for j in 0..n_out {
        let f0 = f0_frames[j / upsample_scale];
        let uv = if f0 > VOICED_THRESHOLD { 1.0f32 } else { 0.0 };
        let noise_amp = uv * NOISE_STD + (1.0 - uv) * SINE_AMP / 3.0;
        let mut acc = l_bias;
        for (k, bank) in banks.iter().enumerate() {
            let sine = bank[j] * SINE_AMP;
            // Gaussian noise, exactly like the reference `randn_like` draw, from the
            // request-seeded RNG (reproducibility law).
            let noise: f32 = noise_amp * rng.sample::<f32, _>(rand_distr::StandardNormal);
            let s = sine * uv + noise;
            acc += l_weight[k] * s;
        }
        out.push(acc.tanh());
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Generator (vocoder head).
// ---------------------------------------------------------------------------------------------

pub struct Generator {
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<AdaInResBlock1>,
    noise_convs: Vec<Conv1d>,
    noise_res: Vec<AdaInResBlock1>,
    conv_post: Conv1d,
    m_source_weight: Vec<f32>,
    m_source_bias: f32,
    n_fft: usize,
    hop: usize,
    upsample_scale: usize,
    num_kernels: usize,
}

impl Generator {
    pub fn new(style_dim: usize, cfg: &IstftNetConfig, vb: VarBuilder) -> Result<Self> {
        let num_upsamples = cfg.upsample_rates.len();
        let num_kernels = cfg.resblock_kernel_sizes.len();
        let mut ups = Vec::new();
        for (i, (&u, &k)) in cfg
            .upsample_rates
            .iter()
            .zip(&cfg.upsample_kernel_sizes)
            .enumerate()
        {
            ups.push(conv_transpose1d(
                cfg.upsample_initial_channel / (1 << i),
                cfg.upsample_initial_channel / (1 << (i + 1)),
                k,
                ConvTranspose1dConfig {
                    padding: (k - u) / 2,
                    stride: u,
                    ..Default::default()
                },
                vb.pp(format!("ups.{i}")),
            )?);
        }
        let mut resblocks = Vec::new();
        let mut noise_convs = Vec::new();
        let mut noise_res = Vec::new();
        for i in 0..num_upsamples {
            let ch = cfg.upsample_initial_channel / (1 << (i + 1));
            for (j, (&k, d)) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(&cfg.resblock_dilation_sizes)
                .enumerate()
            {
                resblocks.push(AdaInResBlock1::new(
                    ch,
                    k,
                    d,
                    style_dim,
                    vb.pp(format!("resblocks.{}", i * num_kernels + j)),
                )?);
            }
            if i + 1 < num_upsamples {
                let stride_f0: usize = cfg.upsample_rates[i + 1..].iter().product();
                noise_convs.push(conv1d(
                    cfg.gen_istft_n_fft + 2,
                    ch,
                    stride_f0 * 2,
                    Conv1dConfig {
                        stride: stride_f0,
                        padding: stride_f0.div_ceil(2),
                        ..Default::default()
                    },
                    vb.pp(format!("noise_convs.{i}")),
                )?);
                noise_res.push(AdaInResBlock1::new(
                    ch,
                    7,
                    &[1, 3, 5],
                    style_dim,
                    vb.pp(format!("noise_res.{i}")),
                )?);
            } else {
                noise_convs.push(conv1d(
                    cfg.gen_istft_n_fft + 2,
                    ch,
                    1,
                    Conv1dConfig::default(),
                    vb.pp(format!("noise_convs.{i}")),
                )?);
                noise_res.push(AdaInResBlock1::new(
                    ch,
                    11,
                    &[1, 3, 5],
                    style_dim,
                    vb.pp(format!("noise_res.{i}")),
                )?);
            }
        }
        let last_ch = cfg.upsample_initial_channel / (1 << num_upsamples);
        let conv_post = conv1d(
            last_ch,
            cfg.gen_istft_n_fft + 2,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_post"),
        )?;
        let m_source_weight: Vec<f32> = vb
            .get((1, HARMONICS), "m_source.l_linear.weight")?
            .flatten_all()?
            .to_vec1()?;
        let m_source_bias: f32 = vb
            .get(1, "m_source.l_linear.bias")?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        let upsample_scale: usize =
            cfg.upsample_rates.iter().product::<usize>() * cfg.gen_istft_hop_size;
        Ok(Self {
            ups,
            resblocks,
            noise_convs,
            noise_res,
            conv_post,
            m_source_weight,
            m_source_bias,
            n_fft: cfg.gen_istft_n_fft,
            hop: cfg.gen_istft_hop_size,
            upsample_scale,
            num_kernels,
        })
    }

    /// `x: [1, C, F]` decoder features, `s: [1, style]`, `f0_frames`: the per-frame F0 curve
    /// (length F) → waveform samples (`F · upsample_scale` of them). `cancel` is probed
    /// between every heavy sub-operation (noise branch, upsample, each resblock, the head).
    pub fn forward(
        &self,
        x: &Tensor,
        s: &Tensor,
        f0_frames: &[f32],
        rng: &mut StdRng,
        cancel: CancelProbe<'_>,
    ) -> Result<Vec<f32>> {
        // Harmonic excitation + its spectrogram (host DSP).
        let har = harmonic_source(
            f0_frames,
            self.upsample_scale,
            &self.m_source_weight,
            self.m_source_bias,
            rng,
        );
        check_cancel(cancel)?;
        let (har_mag, har_phase, har_frames) = stft_small(&har, self.n_fft, self.hop)?;
        let n_bins = self.n_fft / 2 + 1;
        let mut har_cat = Vec::with_capacity(2 * n_bins * har_frames);
        har_cat.extend_from_slice(&har_mag);
        har_cat.extend_from_slice(&har_phase);
        let har = Tensor::from_vec(har_cat, (1, 2 * n_bins, har_frames), x.device())?;
        check_cancel(cancel)?;

        let mut x = x.clone();
        let n_up = self.ups.len();
        for i in 0..n_up {
            x = ops::leaky_relu(&x, LRELU_SLOPE_GENERATOR)?;
            let x_source = self.noise_convs[i].forward(&har)?;
            let x_source = self.noise_res[i].forward(&x_source, s)?;
            check_cancel(cancel)?;
            x = self.ups[i].forward(&x)?;
            if i == n_up - 1 {
                x = reflection_pad_left1(&x)?;
            }
            x = (x + x_source)?;
            check_cancel(cancel)?;
            let mut xs: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let r = self.resblocks[i * self.num_kernels + j].forward(&x, s)?;
                check_cancel(cancel)?;
                xs = Some(match xs {
                    Some(acc) => (acc + r)?,
                    None => r,
                });
            }
            x = (xs.expect("num_kernels >= 1") / self.num_kernels as f64)?;
        }
        check_cancel(cancel)?;
        // F.leaky_relu's default slope (0.01) — distinct from the 0.1 used inside the loop.
        let x = ops::leaky_relu(&x, 0.01)?;
        let x = self.conv_post.forward(&x)?; // [1, n_fft + 2, Fr]
        check_cancel(cancel)?;
        let frames = x.dim(2)?;
        let spec = x
            .i((0, ..n_bins, ..))?
            .exp()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let phase = x
            .i((0, n_bins.., ..))?
            .sin()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(istft_small(&spec, &phase, frames, self.n_fft, self.hop))
    }
}

// ---------------------------------------------------------------------------------------------
// Decoder (the styled feature stack in front of the vocoder head).
// ---------------------------------------------------------------------------------------------

pub struct Decoder {
    encode: AdainResBlk1d,
    decode: Vec<AdainResBlk1d>,
    f0_conv: Conv1d,
    n_conv: Conv1d,
    asr_res: Conv1d,
    pub generator: Generator,
}

impl Decoder {
    pub fn new(
        dim_in: usize,
        style_dim: usize,
        cfg: &IstftNetConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let encode = AdainResBlk1d::new(dim_in + 2, 1024, style_dim, false, vb.pp("encode"))?;
        let decode = vec![
            AdainResBlk1d::new(1024 + 2 + 64, 1024, style_dim, false, vb.pp("decode.0"))?,
            AdainResBlk1d::new(1024 + 2 + 64, 1024, style_dim, false, vb.pp("decode.1"))?,
            AdainResBlk1d::new(1024 + 2 + 64, 1024, style_dim, false, vb.pp("decode.2"))?,
            AdainResBlk1d::new(1024 + 2 + 64, 512, style_dim, true, vb.pp("decode.3"))?,
        ];
        let stride2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let f0_conv = conv1d(1, 1, 3, stride2, vb.pp("F0_conv"))?;
        let n_conv = conv1d(1, 1, 3, stride2, vb.pp("N_conv"))?;
        let asr_res = conv1d(512, 64, 1, Conv1dConfig::default(), vb.pp("asr_res.0"))?;
        let generator = Generator::new(style_dim, cfg, vb.pp("generator"))?;
        Ok(Self {
            encode,
            decode,
            f0_conv,
            n_conv,
            asr_res,
            generator,
        })
    }

    /// `asr: [1, 512, F]` aligned text features, `f0_curve` / `n_curve`: the predictor's 2F
    /// pitch/energy curves, `s: [1, 128]` decoder style → waveform samples. `cancel` is
    /// probed after every styled block and throughout the vocoder head.
    pub fn forward(
        &self,
        asr: &Tensor,
        f0_curve: &[f32],
        n_curve: &[f32],
        s: &Tensor,
        rng: &mut StdRng,
        cancel: CancelProbe<'_>,
    ) -> Result<Vec<f32>> {
        let device = asr.device();
        let f0_t = Tensor::from_slice(f0_curve, (1, 1, f0_curve.len()), device)?;
        let n_t = Tensor::from_slice(n_curve, (1, 1, n_curve.len()), device)?;
        let f0_down = self.f0_conv.forward(&f0_t)?; // [1, 1, F]
        let n_down = self.n_conv.forward(&n_t)?;
        let mut x = Tensor::cat(&[asr, &f0_down, &n_down], 1)?;
        x = self.encode.forward(&x, s)?;
        check_cancel(cancel)?;
        let asr_res = self.asr_res.forward(asr)?; // [1, 64, F]
        let mut res = true;
        for (i, block) in self.decode.iter().enumerate() {
            if res {
                x = Tensor::cat(&[&x, &asr_res, &f0_down, &n_down], 1)?;
            }
            x = block.forward(&x, s)?;
            check_cancel(cancel)?;
            if i == self.decode.len() - 1 {
                res = false;
            }
        }
        let _ = res;
        self.generator.forward(&x, s, f0_curve, rng, cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_rdft_round_trips() {
        let rdft = SmallRdft::new(20);
        let frame: Vec<f32> = (0..20).map(|i| ((i * 3 % 7) as f32 - 3.0) / 3.0).collect();
        let mut re = vec![0.0; 11];
        let mut im = vec![0.0; 11];
        rdft.forward(&frame, &mut re, &mut im);
        let mut back = vec![0.0; 20];
        rdft.inverse(&re, &im, &mut back);
        for (a, b) in frame.iter().zip(&back) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn stft_istft_small_round_trip_reconstructs() {
        let signal: Vec<f32> = (0..2000)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 24000.0).sin())
            .collect();
        let (mag, phase, frames) = stft_small(&signal, 20, 5).unwrap();
        assert_eq!(frames, 1 + signal.len() / 5);
        let out = istft_small(&mag, &phase, frames, 20, 5);
        assert!(out.len() >= signal.len());
        let mut worst = 0.0f32;
        for i in 40..signal.len() - 40 {
            worst = worst.max((signal[i] - out[i]).abs());
        }
        assert!(worst < 1e-3, "worst interior error {worst}");
    }

    #[test]
    fn harmonic_sines_track_the_requested_frequency() {
        // 100 Hz over 40 frames at ×300 → 12000 samples; the fundamental must cross zero
        // upward ~100·(12000/24000) = 50 times.
        let f0 = vec![100.0f32; 40];
        let s = harmonic_sines(&f0, 300, 1);
        assert_eq!(s.len(), 12_000);
        let crossings = s.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count() as i64;
        assert!((crossings - 50).abs() <= 2, "{crossings} upward crossings");
        assert!(s.iter().all(|x| x.is_finite() && x.abs() <= 1.0 + 1e-6));
    }

    #[test]
    fn unvoiced_frames_produce_bounded_noise() {
        use rand::SeedableRng;
        let f0 = vec![0.0f32; 8];
        let mut rng = StdRng::seed_from_u64(7);
        let out = harmonic_source(&f0, 300, &[0.1; HARMONICS], 0.0, &mut rng);
        assert_eq!(out.len(), 2400);
        assert!(out.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
        // Not silent: the unvoiced branch injects sine_amp/3-scale noise.
        let rms = (out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 1e-4, "rms {rms}");
    }
}
