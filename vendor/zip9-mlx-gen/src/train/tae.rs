//! **Tiny AutoEncoder decoder** (TAESD architecture, madebyollin/taesd) — the small, frozen,
//! fully differentiable latent → pixel decoder the shared perceptual-loss path
//! ([`super::perceptual`]) runs a trainer's x0 prediction through (epic 2123 E8, sc-2125).
//!
//! One module graph covers the whole TAESD family (reference: `taesd.py` in
//! github.com/madebyollin/taesd — `Decoder`, `Decoder(use_midblock_gn=True)` and `F16Decoder`):
//!
//! | checkpoint (HF repo / file) | VAE family | latent ch | variant |
//! |---|---|---|---|
//! | `madebyollin/taef1` `diffusion_pytorch_model.safetensors` | FLUX.1 (Z-Image) | 16 | `Decoder` |
//! | `madebyollin/taesdxl` `diffusion_pytorch_model.safetensors` | SDXL (SDXL, Illustrious, Kolors) | 4 | `Decoder` |
//! | `madebyollin/taesd3` `diffusion_pytorch_model.safetensors` | SD3 / SD3.5 | 16 | `Decoder` |
//! | `madebyollin/taef2` `taef2.safetensors` | FLUX.2 (Lens) | 32 | `Decoder`, mid-block GroupNorm pool (`arch_variant = "flux_2"`) |
//! | `madebyollin/taeqi2_1` `taeqi2_1.safetensors` | Qwen-Image-2.1 | 64 | `F16Decoder` (16×, RGBA, 2×2 pixel-shuffle head) |
//!
//! **Input normalization.** Every TAESD-family decoder consumes the diffusion model's
//! **normalized sampler latent directly**: the diffusers checkpoints (taef1 / taesdxl / taesd3)
//! ship `scaling_factor = 1.0` / `shift_factor = 0.0`, so a diffusers pipeline hands them its own
//! latent unchanged (SDXL: `z·0.13025`; SD3: `(z − 0.0609)·1.5305`; FLUX.1: `(z − 0.1159)·0.3611`).
//! The two repos without a diffusers integration say the same in their published wrappers:
//! TAEQI2.1 "consumes / produces normalized latents directly" (its wrapper sets
//! `latents_mean = 0`, `latents_std = 1`), and TAEF2's wrapper substitutes an identity
//! `BatchNorm2d(128, affine=False, eps=0)` for the FLUX.2 VAE's latent batch-norm, so it decodes the
//! transformer's batch-normalized latent **unpatchified** to `[32, h/8, w/8]` (the trainer adapter
//! undoes the 2×2 patchify, not the normalization). No de-normalization is applied here.
//!
//! Architecture (`Decoder`, `num_decoder_blocks = [3, 3, 3, 1]`):
//! `tanh(x/3)·3` → conv3×3(latent→C₀)+ReLU → 3×Block → ↑2 → conv3×3(no bias, C₀→C₁) → 3×Block → ↑2 →
//! conv3×3(C₁→C₂) → 3×Block → ↑2 → conv3×3(C₂→C₃) → 1×Block → conv3×3(C₃→out). A Block is
//! `ReLU(conv(ReLU(conv(ReLU(conv(x))))) + x)`; with the mid-block pool (TAEF2, first stage only)
//! it first does `x += conv1×1(ReLU(GroupNorm₄(conv1×1(x))))` (hidden width 4·C₀, no conv bias).
//! `F16Decoder` widens the low-resolution stages (`C = [256, 128, 64, 64]`), projects to
//! `4·image_channels` and pixel-shuffles 2×2 (16× total upscale). The raw output is pixels in
//! `[0, 1]` (diffusers' `AutoencoderTiny.decode` then maps it to `[-1, 1]`; we keep `[0, 1]`,
//! clamped, which is what every perceptual model consumes). An RGBA decoder (TAEQI2.1) is reduced
//! to its RGB channels — straight alpha, so they are the colour the perceptual models see.
//!
//! Weight keys: the diffusers layout `decoder.layers.{i}.…` (verified against the published
//! safetensors headers of all five repos — taef2/taeqi2_1 use the same layout, which their READMEs'
//! `convert_diffusers_sd_to_taesd` maps onto `taesd.py`); torch OIHW conv weights are permuted to
//! MLX OHWI and cast to f32 at load (taeqi2_1 ships f16).

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{add, clip, multiply, tanh};
use mlx_rs::{random, Array, Dtype};

use crate::nn::{conv2d, upsample_nearest};
use crate::weights::Weights;
use crate::{Error, Result};

/// TAESD-family decoder hyperparameters of the plain `Decoder` graph (uniform width, RGB, 8×).
/// The structural variants (TAEF2's mid-block pool, TAEQI2.1's `F16Decoder`) are described by
/// [`TinyDecoderSpec`]; every API taking a decoder description accepts either.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TinyDecoderConfig {
    /// Latent channels the decoder consumes (16 for TAEF1/TAESD3, 4 for TAESD/TAESDXL).
    pub latent_channels: i32,
    /// Hidden width of every stage (64 for every shipped plain checkpoint).
    pub channels: i32,
    /// Blocks per stage (`[3, 3, 3, 1]` for every shipped checkpoint); one ×2 upsample between
    /// consecutive stages, so the decode upscales by `2^(len-1)` (8).
    pub blocks: [usize; 4],
}

impl TinyDecoderConfig {
    /// `madebyollin/taef1` — the FLUX.1 16-channel latent API (Z-Image's VAE family).
    pub fn taef1() -> Self {
        Self {
            latent_channels: 16,
            channels: 64,
            blocks: [3, 3, 3, 1],
        }
    }

    /// `madebyollin/taesdxl` — the SDXL 4-channel latent API (SDXL, Illustrious, Kolors).
    pub fn taesdxl() -> Self {
        Self {
            latent_channels: 4,
            ..Self::taef1()
        }
    }

    /// `madebyollin/taesd3` — the SD3 16-channel latent API (SD3 / SD3.5).
    pub fn taesd3() -> Self {
        Self::taef1()
    }

    /// Spatial upscale factor latent → pixels (8).
    pub fn upscale(&self) -> i32 {
        TinyDecoderSpec::from(self.clone()).upscale()
    }

    /// Exact parameter count of the decoder graph (for estimating memory before loading).
    pub fn param_count(&self) -> u64 {
        TinyDecoderSpec::from(self.clone()).param_count()
    }

    /// The pre-load memory figures of this decoder for `out_h × out_w` training images (resident
    /// f32 weights + one differentiable decode); the shared estimator
    /// [`super::perceptual::perceptual_footprint_bytes`] sums it with the losses'.
    pub fn footprint(&self, out_h: u32, out_w: u32) -> super::perceptual::AuxModelFootprint {
        TinyDecoderSpec::from(self.clone()).footprint(out_h, out_w)
    }

    /// Conservative upper bound on the **training working set** of one differentiable decode to an
    /// `out_h × out_w` image, in bytes (see [`TinyDecoderSpec::training_working_set_bytes`]).
    pub fn training_working_set_bytes(&self, out_h: u32, out_w: u32) -> u64 {
        TinyDecoderSpec::from(self.clone()).training_working_set_bytes(out_h, out_w)
    }
}

/// Full structural description of a TAESD-family decoder: the plain [`TinyDecoderConfig`] graph
/// plus the `taesd.py` variants (per-stage widths, mid-block GroupNorm pool, RGBA + pixel-shuffle
/// head).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TinyDecoderSpec {
    /// Latent channels the decoder consumes.
    pub latent_channels: i32,
    /// Width of each stage (`[C; 4]` for the plain decoder, `[256, 128, 64, 64]` for F16).
    pub stage_channels: [i32; 4],
    /// Blocks per stage (`[3, 3, 3, 1]` for every shipped checkpoint).
    pub blocks: [usize; 4],
    /// The first stage's blocks carry the mid-block GroupNorm pool (`use_midblock_gn`, TAEF2).
    pub midblock_gn: bool,
    /// Image channels the decoder produces (3 RGB, 4 RGBA); [`TinyDecoder::decode`] returns RGB.
    pub image_channels: i32,
    /// The head projects to `4·image_channels` and pixel-shuffles 2×2 (`F16Decoder`).
    pub pixel_shuffle: bool,
}

impl From<TinyDecoderConfig> for TinyDecoderSpec {
    fn from(c: TinyDecoderConfig) -> Self {
        Self {
            latent_channels: c.latent_channels,
            stage_channels: [c.channels; 4],
            blocks: c.blocks,
            midblock_gn: false,
            image_channels: 3,
            pixel_shuffle: false,
        }
    }
}

/// GroupNorm group count of the mid-block pool (`nn.GroupNorm(4, 4·C)`).
pub(crate) const MIDBLOCK_GN_GROUPS: i32 = 4;
/// torch `nn.GroupNorm` default eps.
pub(crate) const MIDBLOCK_GN_EPS: f32 = 1e-5;

impl TinyDecoderSpec {
    /// `madebyollin/taef1` (plain decoder, 16 latent channels).
    pub fn taef1() -> Self {
        TinyDecoderConfig::taef1().into()
    }

    /// `madebyollin/taesdxl` (plain decoder, 4 latent channels).
    pub fn taesdxl() -> Self {
        TinyDecoderConfig::taesdxl().into()
    }

    /// `madebyollin/taesd3` (plain decoder, 16 latent channels).
    pub fn taesd3() -> Self {
        TinyDecoderConfig::taesd3().into()
    }

    /// `madebyollin/taef2` — the FLUX.2 32-channel latent API (Lens): the plain decoder with the
    /// first stage's mid-block GroupNorm pool (`taesd.py` `arch_variant = "flux_2"`).
    pub fn taef2() -> Self {
        Self {
            latent_channels: 32,
            midblock_gn: true,
            ..Self::taef1()
        }
    }

    /// `madebyollin/taeqi2_1` — the Qwen-Image-2.1 64-channel, 16× latent API: `F16Decoder`
    /// (`arch_variant = "f16"`, RGBA, `image_channels = 4`).
    pub fn taeqi2_1() -> Self {
        Self {
            latent_channels: 64,
            stage_channels: [256, 128, 64, 64],
            blocks: [3, 3, 3, 1],
            midblock_gn: false,
            image_channels: 4,
            pixel_shuffle: true,
        }
    }

    /// The plain-config view (`channels` = the last stage's width).
    pub fn config(&self) -> TinyDecoderConfig {
        TinyDecoderConfig {
            latent_channels: self.latent_channels,
            channels: self.stage_channels[3],
            blocks: self.blocks,
        }
    }

    /// Spatial upscale factor latent → pixels (8, or 16 with the pixel-shuffle head).
    pub fn upscale(&self) -> i32 {
        (1 << (self.blocks.len() - 1)) * if self.pixel_shuffle { 2 } else { 1 }
    }

    /// Channels of the final conv (before any pixel shuffle).
    fn head_channels(&self) -> i32 {
        self.image_channels * if self.pixel_shuffle { 4 } else { 1 }
    }

    /// Exact parameter count of the decoder graph (for estimating memory before loading).
    pub fn param_count(&self) -> u64 {
        let conv = |i: i32, o: i32, bias: bool| {
            let (i, o) = (i as u64, o as u64);
            o * i * 9 + if bias { o } else { 0 }
        };
        let s = self.stage_channels;
        let stages = self.blocks.len();
        let mut n = conv(self.latent_channels, s[0], true);
        for (i, &nb) in self.blocks.iter().enumerate() {
            n += nb as u64 * 3 * conv(s[i], s[i], true);
            if i + 1 < stages {
                n += conv(s[i], s[i + 1], false);
            }
        }
        if self.midblock_gn {
            let (c, g) = (s[0] as u64, 4 * s[0] as u64);
            // conv1×1(c→4c, no bias) + GroupNorm affine (2·4c) + conv1×1(4c→c, no bias).
            n += self.blocks[0] as u64 * (c * g + 2 * g + g * c);
        }
        n + conv(s[stages - 1], self.head_channels(), true)
    }

    /// The pre-load memory figures of this decoder for `out_h × out_w` training images (resident
    /// f32 weights + one differentiable decode); the shared estimator
    /// [`super::perceptual::perceptual_footprint_bytes`] sums it with the losses'.
    pub fn footprint(&self, out_h: u32, out_w: u32) -> super::perceptual::AuxModelFootprint {
        super::perceptual::AuxModelFootprint {
            param_bytes: self.param_count() * 4,
            working_set_bytes: self.training_working_set_bytes(out_h, out_w),
            reference_bytes_per_image: 0,
        }
    }

    /// Conservative upper bound on the **training working set** of one differentiable decode to an
    /// `out_h × out_w` image, in bytes: every intermediate the backward retains (each Block keeps
    /// its three conv outputs, two ReLUs, the residual sum and the fused ReLU; each stage keeps its
    /// upsample + conv; a mid-block pool keeps its 1×1 conv, norm, ReLU (at 4× width) and the
    /// residual), f32, ×2 for the matching cotangent buffers in the backward. Used by the trainer
    /// memory estimate (epic 2123 E7); not a measured value.
    pub fn training_working_set_bytes(&self, out_h: u32, out_w: u32) -> u64 {
        let per_block = 7u64;
        let full = out_h as u64 * out_w as u64;
        // The last stage runs at the output resolution, or ¼ of it under the 2×2 pixel shuffle.
        let last_stage = if self.pixel_shuffle { full / 4 } else { full };
        let stages = self.blocks.len();
        let mut channel_pixels = 0u64;
        for (i, &n) in self.blocks.iter().enumerate() {
            // Stage i runs at last_stage / 4^(stages-1-i) pixels.
            let pixels = last_stage / (1u64 << (2 * (stages - 1 - i)));
            let c = self.stage_channels[i] as u64;
            let stage_tensors = per_block * n as u64 + 2; // + the stage conv and its upsample
            channel_pixels += stage_tensors * c * pixels;
            if self.midblock_gn && i == 0 {
                channel_pixels += n as u64 * (3 * 4 * c + c) * pixels;
            }
        }
        // The head (+ its pixel shuffle and RGB slice) at the output resolution.
        channel_pixels += 3 * self.head_channels() as u64 * last_stage;
        channel_pixels * 4 * 2
    }
}

/// The optional mid-block GroupNorm pool of a TAEF2 first-stage Block.
struct Pool {
    /// conv1×1 C → 4C (no bias), MLX OHWI.
    expand: Array,
    gn_weight: Array,
    gn_bias: Array,
    /// conv1×1 4C → C (no bias), MLX OHWI.
    project: Array,
}

/// PyTorch `nn.GroupNorm` over NHWC `x`, composed from differentiable MLX ops (groups are
/// contiguous channel ranges, statistics over (H, W, channels-in-group), biased variance).
fn group_norm_nhwc(
    x: &Array,
    weight: &Array,
    bias: &Array,
    groups: i32,
    eps: f32,
) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let g = x.reshape(&[b, h, w, groups, c / groups])?;
    let mean = g.mean_axes(&[1, 2, 4], true)?;
    let centered = mlx_rs::ops::subtract(&g, &mean)?;
    let var = centered.square()?.mean_axes(&[1, 2, 4], true)?;
    let normed = mlx_rs::ops::divide(
        &centered,
        &mlx_rs::ops::sqrt(&add(&var, Array::from_f32(eps))?)?,
    )?
    .reshape(&[b, h, w, c])?;
    Ok(add(&multiply(&normed, weight)?, bias)?)
}

impl Pool {
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = conv2d(x, &self.expand, None, 1, 0)?;
        let h = group_norm_nhwc(
            &h,
            &self.gn_weight,
            &self.gn_bias,
            MIDBLOCK_GN_GROUPS,
            MIDBLOCK_GN_EPS,
        )?;
        let h = mlx_rs::nn::relu(&h)?;
        conv2d(&h, &self.project, None, 1, 0)
    }
}

/// One TAESD residual block: three 3×3 convs (ReLU between), identity skip, fused ReLU; with a
/// mid-block pool, `x += pool(x)` first (`taesd.py` `Block.forward`).
struct Block {
    convs: [(Array, Array); 3],
    pool: Option<Pool>,
}

impl Block {
    fn forward(&self, x: &Array) -> Result<Array> {
        let pooled;
        let x = match &self.pool {
            Some(p) => {
                pooled = add(x, &p.forward(x)?)?;
                &pooled
            }
            None => x,
        };
        let mut h = conv2d(x, &self.convs[0].0, Some(&self.convs[0].1), 1, 1)?;
        h = mlx_rs::nn::relu(&h)?;
        h = conv2d(&h, &self.convs[1].0, Some(&self.convs[1].1), 1, 1)?;
        h = mlx_rs::nn::relu(&h)?;
        h = conv2d(&h, &self.convs[2].0, Some(&self.convs[2].1), 1, 1)?;
        Ok(mlx_rs::nn::relu(&add(&h, x)?)?)
    }
}

enum Layer {
    /// 3×3 conv with optional bias.
    Conv(Array, Option<Array>),
    Relu,
    Block(Block),
    Upsample,
}

/// torch `nn.PixelShuffle(2)` on NHWC: channel `c·4 + i·2 + j` → pixel `(2h + i, 2w + j)` of
/// output channel `c`.
fn pixel_shuffle2(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3] / 4);
    Ok(x.reshape(&[b, h, w, c, 2, 2])?
        .transpose_axes(&[0, 1, 4, 2, 5, 3])?
        .reshape(&[b, 2 * h, 2 * w, c])?)
}

/// The loaded, frozen TAESD-family decoder.
pub struct TinyDecoder {
    layers: Vec<Layer>,
    spec: TinyDecoderSpec,
    cfg: TinyDecoderConfig,
    param_bytes: u64,
}

/// Permute a torch conv weight `[out, in, kH, kW]` → MLX `[out, kH, kW, in]`.
fn ohwi(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// The key every TAESD-family checkpoint in the diffusers layout carries (the first decoder conv).
pub(crate) const DECODER_PROBE_KEY: &str = "decoder.layers.0.weight";

/// The file a decoder checkpoint directory is read from: the diffusers
/// `diffusion_pytorch_model.safetensors` when present (taef1 / taesdxl / taesd3 — the taesd/taesdxl
/// repos also ship raw `taesd.py`-layout `*_{encoder,decoder}.safetensors` whose keys collide);
/// otherwise the ONE `.safetensors` in the directory that carries the diffusers-layout decoder
/// (`decoder.layers.0.weight` — taef2 ships only `taef2.safetensors`, taeqi2_1 only
/// `taeqi2_1.safetensors`, both in that layout per their published headers). Unrelated
/// `.safetensors` files are skipped; none or several decoder files is a named error.
fn read_decoder_dir(dir: &std::path::Path) -> Result<Weights> {
    let diffusers = dir.join("diffusion_pytorch_model.safetensors");
    if diffusers.is_file() {
        return Weights::from_file(diffusers);
    }
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| {
            Error::Msg(format!(
                "tiny decoder: cannot read checkpoint directory {}: {e}",
                dir.display()
            ))
        })?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .filter(|p| !gen_core::weightsmeta::is_hidden_file(p))
        .collect();
    files.sort();
    let mut found: Vec<(std::path::PathBuf, Weights)> = Vec::new();
    for f in files {
        let w = Weights::from_file(&f)?;
        if w.get(DECODER_PROBE_KEY).is_some() {
            found.push((f, w));
        }
    }
    match found.len() {
        1 => Ok(found.pop().expect("one").1),
        0 => Err(Error::Msg(format!(
            "tiny decoder: no .safetensors file in {} carries a diffusers-layout decoder \
             ({DECODER_PROBE_KEY})",
            dir.display()
        ))),
        _ => Err(Error::Msg(format!(
            "tiny decoder: several decoder checkpoints in {}: {:?}",
            dir.display(),
            found
                .iter()
                .map(|(p, _)| p.display().to_string())
                .collect::<Vec<_>>()
        ))),
    }
}

impl TinyDecoder {
    /// Load the decoder half of a TAESD-family checkpoint directory: the diffusers
    /// `diffusion_pytorch_model.safetensors` when present, otherwise the one `.safetensors` that
    /// carries the diffusers-layout decoder (`decoder.layers.0.weight`; taef2 / taeqi2_1 ship a
    /// single `<variant>.safetensors`). Unrelated files are skipped; none or several is an error.
    pub fn from_dir(
        dir: impl AsRef<std::path::Path>,
        cfg: impl Into<TinyDecoderSpec>,
    ) -> Result<Self> {
        let w = read_decoder_dir(dir.as_ref())?;
        Self::from_weights(&w, cfg)
    }

    /// Build from already-read weights carrying the diffusers `decoder.layers.{i}.…` keys.
    pub fn from_weights(w: &Weights, cfg: impl Into<TinyDecoderSpec>) -> Result<Self> {
        let spec: TinyDecoderSpec = cfg.into();
        struct Take<'a> {
            w: &'a Weights,
            bytes: u64,
        }
        impl Take<'_> {
            fn get(&mut self, key: &str) -> Result<Array> {
                let a = self.w.require(key)?.as_dtype(Dtype::Float32)?;
                self.bytes += a.nbytes() as u64;
                Ok(a)
            }
            fn conv(&mut self, idx: usize, bias: bool) -> Result<Layer> {
                let weight = ohwi(&self.get(&format!("decoder.layers.{idx}.weight"))?)?;
                let b = if bias {
                    Some(self.get(&format!("decoder.layers.{idx}.bias"))?)
                } else {
                    None
                };
                Ok(Layer::Conv(weight, b))
            }
            fn leg(&mut self, idx: usize, j: usize) -> Result<(Array, Array)> {
                let p = format!("decoder.layers.{idx}.conv.{j}");
                Ok((
                    ohwi(&self.get(&format!("{p}.weight"))?)?,
                    self.get(&format!("{p}.bias"))?,
                ))
            }
            fn pool(&mut self, idx: usize) -> Result<Pool> {
                let p = format!("decoder.layers.{idx}.pool");
                Ok(Pool {
                    expand: ohwi(&self.get(&format!("{p}.0.weight"))?)?,
                    gn_weight: self.get(&format!("{p}.1.weight"))?,
                    gn_bias: self.get(&format!("{p}.1.bias"))?,
                    project: ohwi(&self.get(&format!("{p}.3.weight"))?)?,
                })
            }
        }
        let mut t = Take { w, bytes: 0 };
        let mut layers = vec![t.conv(0, true)?, Layer::Relu];
        let mut idx = 2usize;
        let stages = spec.blocks.len();
        for (stage, &n) in spec.blocks.iter().enumerate() {
            for _ in 0..n {
                let convs = [t.leg(idx, 0)?, t.leg(idx, 2)?, t.leg(idx, 4)?];
                let pool = if spec.midblock_gn && stage == 0 {
                    Some(t.pool(idx)?)
                } else {
                    None
                };
                layers.push(Layer::Block(Block { convs, pool }));
                idx += 1;
            }
            let last = stage == stages - 1;
            if !last {
                layers.push(Layer::Upsample);
                idx += 1;
            }
            // Stage-closing conv: bias only on the final (C→image) projection.
            layers.push(t.conv(idx, last)?);
            idx += 1;
        }
        Ok(Self {
            layers,
            cfg: spec.config(),
            spec,
            param_bytes: t.bytes,
        })
    }

    /// The loaded configuration's plain view (`channels` = the last stage's width).
    pub fn config(&self) -> &TinyDecoderConfig {
        &self.cfg
    }

    /// The loaded structural description.
    pub fn spec(&self) -> &TinyDecoderSpec {
        &self.spec
    }

    /// Resident parameter bytes (f32).
    pub fn param_bytes(&self) -> u64 {
        self.param_bytes
    }

    /// Model-space latents NCHW `[B, C, h, w]` → RGB pixels NHWC `[B, s·h, s·w, 3]` in `[0, 1]`
    /// (`s` = [`TinyDecoderSpec::upscale`]; an RGBA decoder's alpha is dropped). Pure MLX ops end
    /// to end (no host round trip, no stop-gradient), so it is differentiable in `latents`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let sh = latents.shape();
        if sh.len() != 4 || sh[1] != self.spec.latent_channels {
            return Err(Error::Msg(format!(
                "tiny decoder expects NCHW latents with {} channels, got shape {sh:?}",
                self.spec.latent_channels
            )));
        }
        let x = latents.transpose_axes(&[0, 2, 3, 1])?;
        // Clamp(): tanh(x / 3) * 3.
        let three = Array::from_f32(3.0);
        let mut h = multiply(&tanh(&mlx_rs::ops::divide(&x, &three)?)?, &three)?;
        for layer in &self.layers {
            h = match layer {
                Layer::Conv(w, b) => conv2d(&h, w, b.as_ref(), 1, 1)?,
                Layer::Relu => mlx_rs::nn::relu(&h)?,
                Layer::Block(block) => block.forward(&h)?,
                Layer::Upsample => upsample_nearest(&h, 2)?,
            };
        }
        if self.spec.pixel_shuffle {
            h = pixel_shuffle2(&h)?;
        }
        if self.spec.image_channels > 3 {
            h = h.index((.., .., .., ..3));
        }
        Ok(clip(&h, (&Array::from_f32(0.0), &Array::from_f32(1.0)))?)
    }
}

/// A complete random-init decoder checkpoint for `cfg` in the diffusers key layout (torch OIHW
/// conv weights) — for tests of the shared perceptual path and the trainers that use it, which
/// must never download real weights. Deterministic in `seed`. Accepts a [`TinyDecoderConfig`] or
/// a [`TinyDecoderSpec`].
pub fn synthetic_tiny_decoder_weights<C: Clone + Into<TinyDecoderSpec>>(
    cfg: &C,
    seed: u64,
) -> Result<Weights> {
    let spec: TinyDecoderSpec = cfg.clone().into();
    let mut w = Weights::empty();
    let mut n = 0u64;
    let mut rnd = |shape: &[i32]| -> Result<Array> {
        n += 1;
        let fan_in: i32 = shape[1..].iter().product();
        let std = (2.0 / fan_in as f32).sqrt();
        let key = random::key(seed.wrapping_mul(1_000_003).wrapping_add(n))?;
        Ok(multiply(
            &random::normal::<f32>(shape, None, None, Some(&key))?,
            Array::from_f32(std),
        )?)
    };
    let s = spec.stage_channels;
    let mut idx = 0usize;
    w.insert(
        format!("decoder.layers.{idx}.weight"),
        rnd(&[s[0], spec.latent_channels, 3, 3])?,
    );
    w.insert(
        format!("decoder.layers.{idx}.bias"),
        rnd(&[s[0], 1])?.reshape(&[s[0]])?,
    );
    idx += 2; // conv + relu
    let stages = spec.blocks.len();
    for (stage, &nb) in spec.blocks.iter().enumerate() {
        let c = s[stage];
        for _ in 0..nb {
            for j in [0, 2, 4] {
                w.insert(
                    format!("decoder.layers.{idx}.conv.{j}.weight"),
                    rnd(&[c, c, 3, 3])?,
                );
                w.insert(
                    format!("decoder.layers.{idx}.conv.{j}.bias"),
                    rnd(&[c, 1])?.reshape(&[c])?,
                );
            }
            if spec.midblock_gn && stage == 0 {
                let g = 4 * c;
                let p = format!("decoder.layers.{idx}.pool");
                w.insert(format!("{p}.0.weight"), rnd(&[g, c, 1, 1])?);
                w.insert(
                    format!("{p}.1.weight"),
                    add(&rnd(&[g, 1])?.reshape(&[g])?, Array::from_f32(1.0))?,
                );
                w.insert(format!("{p}.1.bias"), rnd(&[g, 1])?.reshape(&[g])?);
                w.insert(format!("{p}.3.weight"), rnd(&[c, g, 1, 1])?);
            }
            idx += 1;
        }
        let last = stage == stages - 1;
        if !last {
            idx += 1; // upsample
        }
        let out = if last {
            spec.head_channels()
        } else {
            s[stage + 1]
        };
        w.insert(
            format!("decoder.layers.{idx}.weight"),
            rnd(&[out, c, 3, 3])?,
        );
        if last {
            w.insert(
                format!("decoder.layers.{idx}.bias"),
                rnd(&[out, 1])?.reshape(&[out])?,
            );
        }
        idx += 1;
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::{eval, grad};

    fn tiny_cfg() -> TinyDecoderConfig {
        TinyDecoderConfig {
            latent_channels: 4,
            channels: 8,
            blocks: [3, 3, 3, 1],
        }
    }

    /// Every key the synthetic builder emits must be consumed by the loader and vice versa, and
    /// the listed keys (a sample of the published safetensors header) must exist.
    fn assert_layout(spec: TinyDecoderSpec, present: &[&str], absent: &[&str]) -> TinyDecoder {
        let w = synthetic_tiny_decoder_weights(&spec, 1).unwrap();
        for key in present {
            assert!(w.get(key).is_some(), "missing {key}");
        }
        for key in absent {
            assert!(w.get(key).is_none(), "unexpected {key}");
        }
        let dec = TinyDecoder::from_weights(&w, spec.clone()).unwrap();
        assert!(w.unused_keys().is_empty(), "unused: {:?}", w.unused_keys());
        assert_eq!(dec.param_bytes(), spec.param_count() * 4);
        dec
    }

    /// Keys shared by all five published headers (`decoder.layers.{0,2..4,6,7..9,11,12..14,16,17,18}`).
    const COMMON_KEYS: [&str; 10] = [
        "decoder.layers.0.weight",
        "decoder.layers.0.bias",
        "decoder.layers.2.conv.0.weight",
        "decoder.layers.4.conv.4.bias",
        "decoder.layers.6.weight",
        "decoder.layers.11.weight",
        "decoder.layers.16.weight",
        "decoder.layers.17.conv.2.weight",
        "decoder.layers.18.weight",
        "decoder.layers.18.bias",
    ];

    #[test]
    fn taef1_layout_indices_match_the_diffusers_checkpoint() {
        // The real TAEF1 safetensors keys: conv 0, blocks 2-4, up 5, conv 6, blocks 7-9, up 10,
        // conv 11, blocks 12-14, up 15, conv 16, block 17, conv 18 (+bias).
        let cfg = TinyDecoderConfig::taef1();
        let dec = assert_layout(
            cfg.clone().into(),
            &COMMON_KEYS,
            &["decoder.layers.6.bias", "decoder.layers.2.pool.0.weight"],
        );
        // 16·64·9+64 + 10 blocks·3·(64·64·9+64) + 3·64·64·9 + 3·64·9+3 params, f32.
        let params = 16 * 64 * 9 + 64 + 30 * (64 * 64 * 9 + 64) + 3 * 64 * 64 * 9 + 3 * 64 * 9 + 3;
        assert_eq!(dec.param_bytes(), params as u64 * 4);
        assert_eq!(dec.config().param_count(), params as u64);
        assert_eq!(cfg.param_count(), params as u64);
    }

    /// taesdxl / taesd3 (`diffusion_pytorch_model.safetensors`, header read at revisions b20258aa /
    /// d58dcacc): the plain layout; `decoder.layers.0.weight` is `[64, 4, 3, 3]` / `[64, 16, 3, 3]`.
    #[test]
    fn taesdxl_and_taesd3_layouts_match_the_published_headers() {
        for (spec, latent) in [
            (TinyDecoderSpec::taesdxl(), 4),
            (TinyDecoderSpec::taesd3(), 16),
        ] {
            assert_eq!(spec.latent_channels, latent);
            let w = synthetic_tiny_decoder_weights(&spec, 3).unwrap();
            assert_eq!(
                w.get("decoder.layers.0.weight").unwrap().shape(),
                &[64, latent, 3, 3]
            );
            assert_layout(spec, &COMMON_KEYS, &["decoder.layers.6.bias"]);
        }
        assert_eq!(TinyDecoderConfig::taesdxl().latent_channels, 4);
        assert_eq!(TinyDecoderConfig::taesd3().upscale(), 8);
    }

    /// taef2 (`taef2.safetensors`, header read at revision bd244ebf): the plain layout plus
    /// `decoder.layers.{2,3,4}.pool.{0,1,3}` — `pool.0` `[256, 64, 1, 1]`, `pool.1` GroupNorm
    /// weight/bias `[256]`, `pool.3` `[64, 256, 1, 1]` — and `decoder.layers.0.weight`
    /// `[64, 32, 3, 3]`. Later stages carry no pool.
    #[test]
    fn taef2_layout_matches_the_published_header() {
        let spec = TinyDecoderSpec::taef2();
        let w = synthetic_tiny_decoder_weights(&spec, 4).unwrap();
        assert_eq!(
            w.get("decoder.layers.0.weight").unwrap().shape(),
            &[64, 32, 3, 3]
        );
        assert_eq!(
            w.get("decoder.layers.3.pool.0.weight").unwrap().shape(),
            &[256, 64, 1, 1]
        );
        assert_eq!(
            w.get("decoder.layers.3.pool.1.bias").unwrap().shape(),
            &[256]
        );
        assert_eq!(
            w.get("decoder.layers.4.pool.3.weight").unwrap().shape(),
            &[64, 256, 1, 1]
        );
        let mut present = COMMON_KEYS.to_vec();
        present.extend([
            "decoder.layers.2.pool.0.weight",
            "decoder.layers.2.pool.1.weight",
            "decoder.layers.2.pool.1.bias",
            "decoder.layers.4.pool.3.weight",
        ]);
        assert_layout(
            spec.clone(),
            &present,
            &[
                "decoder.layers.7.pool.0.weight",
                "decoder.layers.2.pool.3.bias",
            ],
        );
        // 3 pools × (64·256 + 2·256 + 256·64) on top of the plain 32-channel graph.
        let plain = TinyDecoderSpec {
            midblock_gn: false,
            ..spec.clone()
        };
        assert_eq!(
            spec.param_count() - plain.param_count(),
            3 * (64 * 256 + 2 * 256 + 256 * 64)
        );
        assert_eq!(spec.upscale(), 8);
    }

    /// taeqi2_1 (`taeqi2_1.safetensors`, f16, header read at revision 379cc3f2): `F16Decoder` —
    /// conv 0 `[256, 64, 3, 3]`, stage convs 6 `[128, 256]`, 11 `[64, 128]`, 16 `[64, 64]`, head 18
    /// `[16, 64, 3, 3]` + bias `[16]` (4 RGBA × 2×2 shuffle).
    #[test]
    fn taeqi2_1_layout_matches_the_published_header() {
        let spec = TinyDecoderSpec::taeqi2_1();
        let w = synthetic_tiny_decoder_weights(&spec, 5).unwrap();
        for (key, shape) in [
            ("decoder.layers.0.weight", vec![256, 64, 3, 3]),
            ("decoder.layers.2.conv.0.weight", vec![256, 256, 3, 3]),
            ("decoder.layers.6.weight", vec![128, 256, 3, 3]),
            ("decoder.layers.7.conv.0.weight", vec![128, 128, 3, 3]),
            ("decoder.layers.11.weight", vec![64, 128, 3, 3]),
            ("decoder.layers.16.weight", vec![64, 64, 3, 3]),
            ("decoder.layers.18.weight", vec![16, 64, 3, 3]),
            ("decoder.layers.18.bias", vec![16]),
        ] {
            assert_eq!(w.get(key).unwrap().shape(), shape.as_slice(), "{key}");
        }
        assert_layout(spec.clone(), &COMMON_KEYS, &["decoder.layers.6.bias"]);
        assert_eq!(spec.upscale(), 16);
        assert_eq!(spec.config().channels, 64);
    }

    /// The real taeqi2_1 ships f16; the loader casts to f32 (mixed-dtype convs would otherwise
    /// promote/err) and counts f32 bytes. Mutation: drop the `as_dtype(Float32)` ⇒ param bytes
    /// halve ⇒ red.
    #[test]
    fn f16_checkpoints_load_as_f32() {
        let spec = TinyDecoderSpec {
            latent_channels: 8,
            stage_channels: [16, 8, 8, 8],
            ..TinyDecoderSpec::taeqi2_1()
        };
        let src = synthetic_tiny_decoder_weights(&spec, 6).unwrap();
        let mut half = Weights::empty();
        for key in src.keys() {
            half.insert(
                key.to_string(),
                src.get(key).unwrap().as_dtype(Dtype::Float16).unwrap(),
            );
        }
        let dec = TinyDecoder::from_weights(&half, spec.clone()).unwrap();
        assert_eq!(dec.param_bytes(), spec.param_count() * 4);
        let z = random::normal::<f32>(&[1, 8, 2, 2], None, None, Some(&random::key(1).unwrap()))
            .unwrap();
        assert_eq!(dec.decode(&z).unwrap().dtype(), Dtype::Float32);
    }

    #[test]
    fn decode_upscales_by_eight_into_unit_range_and_is_differentiable() {
        let cfg = tiny_cfg();
        let dec = TinyDecoder::from_weights(&synthetic_tiny_decoder_weights(&cfg, 2).unwrap(), cfg)
            .unwrap();
        let z = random::normal::<f32>(&[1, 4, 3, 2], None, None, Some(&random::key(5).unwrap()))
            .unwrap();
        let px = dec.decode(&z).unwrap();
        assert_eq!(px.shape(), &[1, 24, 16, 3]);
        eval([&px]).unwrap();
        let v = px.as_slice::<f32>();
        assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
        // d(sum(pixels))/d(latents) flows (no host round trip / stop-gradient in the graph).
        let f = |z: &Array| -> mlx_rs::error::Result<Array> {
            dec.decode(z)
                .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))?
                .sum(None)
        };
        let g = grad(f)(&z).unwrap();
        eval([&g]).unwrap();
        assert!(g.abs().unwrap().sum(None).unwrap().item::<f32>() > 0.0);
    }

    fn grad_flows(dec: &TinyDecoder, z: &Array) -> f32 {
        let f = |z: &Array| -> mlx_rs::error::Result<Array> {
            dec.decode(z)
                .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))?
                .sum(None)
        };
        let g = grad(f)(z).unwrap();
        eval([&g]).unwrap();
        g.abs().unwrap().sum(None).unwrap().item::<f32>()
    }

    /// A tiny F16 decoder (pixel-shuffle head, RGBA) upscales 16× and returns RGB only, in range,
    /// differentiably. Mutations: drop the pixel shuffle ⇒ 8× ⇒ red; drop the RGB slice ⇒ 4
    /// channels ⇒ red.
    #[test]
    fn f16_decode_upscales_by_sixteen_to_rgb() {
        let spec = TinyDecoderSpec {
            latent_channels: 8,
            stage_channels: [16, 8, 8, 8],
            ..TinyDecoderSpec::taeqi2_1()
        };
        let dec =
            TinyDecoder::from_weights(&synthetic_tiny_decoder_weights(&spec, 7).unwrap(), spec)
                .unwrap();
        let z = random::normal::<f32>(&[1, 8, 2, 3], None, None, Some(&random::key(8).unwrap()))
            .unwrap();
        let px = dec.decode(&z).unwrap();
        assert_eq!(px.shape(), &[1, 32, 48, 3]);
        eval([&px]).unwrap();
        assert!(px.as_slice::<f32>().iter().all(|x| (0.0..=1.0).contains(x)));
        assert!(grad_flows(&dec, &z) > 0.0);
    }

    /// `pixel_shuffle2` is torch `PixelShuffle(2)`: input channel `c·4 + i·2 + j` at `(h, w)` lands
    /// at `(2h + i, 2w + j)` of channel `c`. Mutation: swap the `i`/`j` axes in the transpose ⇒ red.
    #[test]
    fn pixel_shuffle_matches_torch_channel_order() {
        // [1, 1, 1, 8]: two output channels × (i, j) ∈ {0,1}².
        let x = Array::from_slice(&[0f32, 1., 2., 3., 4., 5., 6., 7.], &[1, 1, 1, 8]);
        // Materialize row-major before reading the host slice (the shuffle is a strided view).
        let y = crate::array::contiguous(&pixel_shuffle2(&x).unwrap()).unwrap();
        assert_eq!(y.shape(), &[1, 2, 2, 2]);
        eval([&y]).unwrap();
        // y[0, i, j, c] = c·4 + i·2 + j.
        let v = y.as_slice::<f32>().to_vec();
        let at = |i: usize, j: usize, c: usize| v[(i * 2 + j) * 2 + c];
        for (i, j, c) in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1), (0, 1, 1)] {
            assert_eq!(at(i, j, c), (c * 4 + i * 2 + j) as f32, "({i},{j},{c})");
        }
    }

    /// The mid-block pool participates: a TAEF2-style decoder's output differs from the same
    /// weights without the pool, and it stays differentiable. Mutation: skip `x += pool(x)` in
    /// `Block::forward` ⇒ outputs equal ⇒ red.
    #[test]
    fn midblock_pool_changes_the_decode_and_is_differentiable() {
        let spec = TinyDecoderSpec {
            latent_channels: 4,
            stage_channels: [8; 4],
            midblock_gn: true,
            ..TinyDecoderSpec::taef2()
        };
        let w = synthetic_tiny_decoder_weights(&spec, 9).unwrap();
        let with = TinyDecoder::from_weights(&w, spec.clone()).unwrap();
        let without = TinyDecoder::from_weights(
            &w,
            TinyDecoderSpec {
                midblock_gn: false,
                ..spec
            },
        )
        .unwrap();
        let z = random::normal::<f32>(&[1, 4, 3, 3], None, None, Some(&random::key(10).unwrap()))
            .unwrap();
        let a = with.decode(&z).unwrap();
        let b = without.decode(&z).unwrap();
        eval([&a, &b]).unwrap();
        assert_eq!(a.shape(), &[1, 24, 24, 3]);
        let diff = mlx_rs::ops::subtract(&a, &b)
            .unwrap()
            .abs()
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>();
        assert!(diff > 0.0, "pool had no effect");
        assert!(grad_flows(&with, &z) > 0.0);
    }

    /// The composed GroupNorm normalizes each contiguous channel group over (H, W, group channels)
    /// like torch. Mutation: reduce over axes `[1, 2]` only ⇒ red.
    #[test]
    fn group_norm_matches_torch_semantics() {
        // [1, 1, 2, 4], 2 groups: group 0 = channels {0,1}, group 1 = {2,3}.
        let x = Array::from_slice(&[1f32, 3., 10., 10., 5., 7., 20., 40.], &[1, 1, 2, 4]);
        let ones = Array::from_slice(&[1f32; 4], &[4]);
        let zeros = Array::from_slice(&[0f32; 4], &[4]);
        let y = group_norm_nhwc(&x, &ones, &zeros, 2, 0.0).unwrap();
        eval([&y]).unwrap();
        let v = y.as_slice::<f32>();
        // group 0 values {1,3,5,7}: mean 4, var 5.
        let s = 5f32.sqrt();
        for (got, want) in [
            (v[0], -3.0 / s),
            (v[1], -1.0 / s),
            (v[4], 1.0 / s),
            (v[5], 3.0 / s),
        ] {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
        // group 1 values {10,10,20,40}: mean 20, var 150.
        let s1 = 150f32.sqrt();
        assert!((v[2] - (-10.0 / s1)).abs() < 1e-5);
        assert!((v[7] - (20.0 / s1)).abs() < 1e-5);
    }

    #[test]
    fn decode_refuses_the_wrong_latent_layout() {
        let cfg = tiny_cfg();
        let dec = TinyDecoder::from_weights(&synthetic_tiny_decoder_weights(&cfg, 2).unwrap(), cfg)
            .unwrap();
        let z = Array::zeros::<f32>(&[1, 3, 2, 2]).unwrap();
        assert!(dec.decode(&z).is_err());
    }

    #[test]
    fn working_set_scales_with_output_area() {
        for spec in [
            TinyDecoderSpec::taef1(),
            TinyDecoderSpec::taef2(),
            TinyDecoderSpec::taeqi2_1(),
        ] {
            let a = spec.training_working_set_bytes(512, 512);
            let b = spec.training_working_set_bytes(1024, 1024);
            assert_eq!(b, a * 4, "{spec:?}");
            assert!(a > 0);
        }
        let cfg = TinyDecoderConfig::taef1();
        assert_eq!(
            cfg.training_working_set_bytes(512, 512),
            TinyDecoderSpec::taef1().training_working_set_bytes(512, 512)
        );
        // The wider/pooled variants cost more than the plain decoder at the same output size.
        let plain = TinyDecoderSpec::taef1().footprint(1024, 1024);
        let f2 = TinyDecoderSpec::taef2().footprint(1024, 1024);
        let qi = TinyDecoderSpec::taeqi2_1().footprint(1024, 1024);
        assert!(
            f2.param_bytes > plain.param_bytes && f2.working_set_bytes > plain.working_set_bytes
        );
        assert!(qi.param_bytes > plain.param_bytes);
    }

    /// `from_dir` reads the diffusers file when the directory also holds the raw
    /// `taesd.py`-layout encoder/decoder files (whose keys collide), and otherwise the single file.
    #[test]
    fn from_dir_prefers_the_diffusers_file() {
        let cfg = tiny_cfg();
        let w = synthetic_tiny_decoder_weights(&cfg, 2).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let save = |w: &Weights, name: &str| {
            let pairs: Vec<(String, Array)> = w
                .keys()
                .map(|k| (k.to_string(), w.get(k).unwrap().clone()))
                .collect();
            Array::save_safetensors(
                pairs.iter().map(|(k, v)| (k.as_str(), v)),
                None,
                dir.path().join(name),
            )
            .unwrap();
        };
        save(&w, "single.safetensors");
        assert!(TinyDecoder::from_dir(dir.path(), cfg.clone()).is_ok());
        std::fs::rename(
            dir.path().join("single.safetensors"),
            dir.path().join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();
        // Two raw-layout files sharing a key: loading the whole dir would be a duplicate-key error.
        let mut raw = Weights::empty();
        raw.insert("1.weight", Array::zeros::<f32>(&[1]).unwrap());
        save(&raw, "taesd_encoder.safetensors");
        save(&raw, "taesd_decoder.safetensors");
        assert!(Weights::from_dir(dir.path()).is_err());
        assert!(TinyDecoder::from_dir(dir.path(), cfg).is_ok());
    }

    fn save_to(dir: &std::path::Path, w: &Weights, name: &str) {
        let pairs: Vec<(String, Array)> = w
            .keys()
            .map(|k| (k.to_string(), w.get(k).unwrap().clone()))
            .collect();
        Array::save_safetensors(
            pairs.iter().map(|(k, v)| (k.as_str(), v)),
            None,
            dir.join(name),
        )
        .unwrap();
    }

    /// The taef2 / taeqi2_1 snapshot shape (one `<variant>.safetensors`, diffusers decoder keys plus
    /// `encoder.*`, no diffusers file): `from_dir` picks the decoder file even with an unrelated
    /// `.safetensors` beside it (whose keys would otherwise merge in, or collide), and refuses a
    /// directory with no decoder file or two. Mutations: read every file (`Weights::from_dir`) ⇒ a
    /// colliding unrelated key errors ⇒ red; accept the first file instead of the probed one ⇒
    /// `aaa_unrelated` is picked ⇒ red.
    #[test]
    fn from_dir_picks_the_variant_file_among_unrelated_safetensors() {
        let spec = TinyDecoderSpec {
            latent_channels: 4,
            stage_channels: [8; 4],
            ..TinyDecoderSpec::taef2()
        };
        let mut w = synthetic_tiny_decoder_weights(&spec, 3).unwrap();
        w.insert("encoder.0.weight", Array::zeros::<f32>(&[1]).unwrap());
        let dir = tempfile::tempdir().unwrap();
        let mut unrelated = Weights::empty();
        // Shares a key with the decoder file, so merging the directory would be a collision.
        unrelated.insert("encoder.0.weight", Array::zeros::<f32>(&[2]).unwrap());
        save_to(dir.path(), &unrelated, "aaa_unrelated.safetensors");
        let only_unrelated = TinyDecoder::from_dir(dir.path(), spec.clone())
            .err()
            .expect("no decoder file")
            .to_string();
        assert!(
            only_unrelated.contains("decoder.layers.0.weight"),
            "{only_unrelated}"
        );
        save_to(dir.path(), &w, "taef2.safetensors");
        let dec = TinyDecoder::from_dir(dir.path(), spec.clone()).unwrap();
        assert_eq!(dec.spec(), &spec);
        save_to(dir.path(), &w, "taef2_copy.safetensors");
        let two = TinyDecoder::from_dir(dir.path(), spec)
            .err()
            .expect("ambiguous")
            .to_string();
        assert!(two.contains("several"), "{two}");
    }
}
