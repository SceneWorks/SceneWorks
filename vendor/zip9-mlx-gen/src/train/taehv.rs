//! **TAEHV tiny video decoder** (madebyollin/taehv `taehv.py`) — the small, frozen, fully
//! differentiable latent → pixel decoder the shared perceptual-loss path ([`super::perceptual`])
//! runs a **video/Wan-VAE-family** trainer's x0 prediction through (epic 2123 E8, sc-24830).
//!
//! Faithful port of the reference's non-`_super` decoder (`TAEHV.decoder`, upstream rev
//! `011dfc2112197741c540e0bdd5b7b67bcc930771`). Three checkpoints are wired, each the upstream
//! name-guessed architecture:
//!
//! | variant | base VAE | latent ch | patch | `decoder_time_upscale` | t-upscale | `frames_to_trim` | px / latent |
//! |---|---|---|---|---|---|---|---|
//! | `taew2_1` | Wan 2.1, Wan 2.2 14B, Qwen-Image | 16 | 1 | (F, T, T) | 4 | 3 | 8 |
//! | `taew2_2` | Wan 2.2 TI2V 5B | 48 | 2 | (F, T, T) | 4 | 3 | 16 |
//! | `taeltx2_3` | LTX-2.3 / LTX-2.5 | 128 | 4 | (T, T, T) | 8 | 7 | 32 |
//!
//! (`decoder_space_upscale = (T, T, T)` for all three.)
//!
//! **Latent space.** TAEHV applies no latent scale/shift: it decodes the diffusion model's own
//! **normalized** latent. The upstream diffusers wrappers set `latents_mean = 0`, `latents_std = 1`
//! (Wan) and the TAELTX demo feeds `(z − latents_mean) / latents_std` (LTX); i.e. exactly the
//! per-channel-normalized latent the trainers cache and the DiT regresses.
//!
//! Decoder graph (`decoder.{i}` = the reference `nn.Sequential` index; `n_f = [256, 128, 64, 64]`):
//! `0` Clamp `tanh(x/3)·3` → `1` conv3×3(latent→256)+bias → `2` ReLU → per stage `s ∈ 0..3` at
//! base `b = 3 + 6s`: `b..b+2` 3×`MemBlock(n_f[s])` → `b+3` nearest ↑2 → `b+4` `TGrow(n_f[s], stride)`
//! (1×1 conv, no bias, `n_f[s] → n_f[s]·stride`, then split into `stride` consecutive frames) →
//! `b+5` conv3×3(`n_f[s] → n_f[s+1]`, no bias) → `21` ReLU → `22` conv3×3(64 → 3·patch²)+bias →
//! pixel-shuffle(patch) → clamp [0, 1] → drop the first `frames_to_trim` output frames.
//! A MemBlock is `ReLU(conv(ReLU(conv(ReLU(conv(cat[x, past]))))) + x)` where `past` is the same
//! block's input one frame earlier in the clip (zeros for a clip's first frame) — the reference's
//! parallel (`apply_model_with_memblocks_parallel`) semantics.
//!
//! **Per-frame decode (the perceptual path).** [`TaehvDecoder::decode_frames`] — and its
//! [`X0Decoder`] impl — decodes each of `N` latent frames **independently as a `T = 1` clip**:
//! the decoder grows the single latent into `t_upscale` frames and the reference trim keeps only
//! the **last** one. Because no MemBlock follows the final TGrow, that frame equals the last
//! TGrow output chunk of the last stage-2 frame, so the per-frame path computes the full-resolution
//! tail for that one frame only (the TGrow 1×1 conv restricted to its last `n_f[2]` output
//! channels — the same slice the reference's `patch_tgrow_layers` takes). It is bit-for-bit the
//! same math as [`TaehvDecoder::decode_video`] with `T = 1` (tested).
//! [`TaehvDecoder::decode_clip_last_frames`] is the same tail on `T`-frame clips: the MemBlocks
//! carry each clip's earlier frames and only the clip's final output frame is kept — a video
//! trainer decodes latent frame `k > 0` with its predecessor this way (LTX-2.5).
//!
//! Weight keys: the reference `state_dict` layout (`decoder.{i}.…`, torch OIHW conv weights
//! permuted to MLX OHWI at load, cast to f32). Shipped checkpoints also carry the 64 `encoder.*`
//! keys, which the decoder never reads; any unread `decoder.*` key is an error (wrong variant,
//! e.g. a `_super` checkpoint).

use std::path::Path;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{add, clip, concatenate_axis, divide, multiply, tanh};
use mlx_rs::{random, Array, Dtype};

use super::perceptual::{AuxModelFootprint, X0Decoder};
use crate::nn::{conv2d, upsample_nearest};
use crate::weights::Weights;
use crate::{Error, Result};

/// TAEHV decoder hyperparameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaehvConfig {
    /// Checkpoint name (`taew2_1`, `taew2_2`, `taeltx2_3`); the checkpoint file is
    /// `<name>.safetensors`.
    pub name: &'static str,
    /// Latent channels the decoder consumes.
    pub latent_channels: i32,
    /// Output pixel-shuffle patch size.
    pub patch_size: i32,
    /// Per-stage temporal ×2 upsampling (TGrow stride 2 when true).
    pub decoder_time_upscale: [bool; 3],
    /// Per-stage spatial ×2 nearest upsampling.
    pub decoder_space_upscale: [bool; 3],
    /// Stage widths `n_f` (`[256, 128, 64, 64]` for every shipped non-super checkpoint).
    pub channels: [i32; 4],
}

impl TaehvConfig {
    /// `taew2_1` — Wan 2.1 / Wan 2.2 14B / Qwen-Image VAE (16 channels).
    pub fn taew2_1() -> Self {
        Self {
            name: "taew2_1",
            latent_channels: 16,
            patch_size: 1,
            decoder_time_upscale: [false, true, true],
            decoder_space_upscale: [true, true, true],
            channels: [256, 128, 64, 64],
        }
    }

    /// `taew2_2` — Wan 2.2 TI2V 5B VAE (48 channels, 16× spatial).
    pub fn taew2_2() -> Self {
        Self {
            name: "taew2_2",
            latent_channels: 48,
            patch_size: 2,
            ..Self::taew2_1()
        }
    }

    /// `taeltx2_3` — LTX-2.3 / LTX-2.5 VAE (128 channels, 32× spatial, 8× temporal).
    pub fn taeltx2_3() -> Self {
        Self {
            name: "taeltx2_3",
            latent_channels: 128,
            patch_size: 4,
            decoder_time_upscale: [true, true, true],
            ..Self::taew2_1()
        }
    }

    /// The Wan-VAE-family variant for a Wan/Qwen-Image VAE `z_dim` (16 → `taew2_1`,
    /// 48 → `taew2_2`); `None` for any other channel count.
    pub fn for_wan_z_dim(z_dim: i32) -> Option<Self> {
        match z_dim {
            16 => Some(Self::taew2_1()),
            48 => Some(Self::taew2_2()),
            _ => None,
        }
    }

    /// Human-readable decoder name for errors (`TAEW2.1`, `TAEW2.2`, `TAELTX2.3`).
    pub fn display_name(&self) -> String {
        match self.name {
            "taew2_1" => "TAEW2.1".into(),
            "taew2_2" => "TAEW2.2".into(),
            "taeltx2_3" => "TAELTX2.3".into(),
            other => other.to_uppercase(),
        }
    }

    /// The checkpoint file name (`<name>.safetensors`, the upstream `safetensors/` file).
    pub fn file_name(&self) -> String {
        format!("{}.safetensors", self.name)
    }

    fn time_stride(&self, stage: usize) -> i32 {
        if self.decoder_time_upscale[stage] {
            2
        } else {
            1
        }
    }

    fn space_scale(&self, stage: usize) -> i32 {
        if self.decoder_space_upscale[stage] {
            2
        } else {
            1
        }
    }

    /// Output frames per latent frame (`2^#time-upscaling stages`).
    pub fn t_upscale(&self) -> i32 {
        (0..3).map(|s| self.time_stride(s)).product()
    }

    /// Leading output frames the reference drops (`t_upscale − 1`).
    pub fn frames_to_trim(&self) -> i32 {
        self.t_upscale() - 1
    }

    /// Pixels per latent along each spatial axis (`2^#space-upscaling stages · patch`).
    pub fn spatial_upscale(&self) -> i32 {
        (0..3).map(|s| self.space_scale(s)).product::<i32>() * self.patch_size
    }

    /// Exact parameter count of the decoder graph.
    pub fn param_count(&self) -> u64 {
        let conv3 = |i: i32, o: i32, bias: bool| {
            (o as u64) * (i as u64) * 9 + if bias { o as u64 } else { 0 }
        };
        let nf = self.channels;
        let mut n = conv3(self.latent_channels, nf[0], true);
        for s in 0..3 {
            let c = nf[s];
            n += 3 * (conv3(2 * c, c, true) + 2 * conv3(c, c, true));
            n += (c as u64) * (c as u64) * self.time_stride(s) as u64;
            n += conv3(c, nf[s + 1], false);
        }
        n + conv3(nf[3], 3 * self.patch_size * self.patch_size, true)
    }

    /// Pre-load memory figures for decoding ONE latent frame to an `out_h × out_w` image (resident
    /// f32 weights + one differentiable per-frame decode). A trainer decoding `k` frames per step
    /// scales `working_set_bytes` by `k`.
    pub fn footprint(&self, out_h: u32, out_w: u32) -> AuxModelFootprint {
        AuxModelFootprint {
            param_bytes: self.param_count() * 4,
            working_set_bytes: self.training_working_set_bytes(out_h, out_w),
            reference_bytes_per_image: 0,
        }
    }

    /// Conservative upper bound on the training working set of one per-frame
    /// ([`TaehvDecoder::decode_frames`]) differentiable decode to `out_h × out_w`, in bytes: every
    /// intermediate the backward retains, f32, ×2 for the cotangents. Not a measured value.
    pub fn training_working_set_bytes(&self, out_h: u32, out_w: u32) -> u64 {
        self.clip_training_working_set_bytes(out_h, out_w, 1)
    }

    /// [`training_working_set_bytes`](Self::training_working_set_bytes) for one
    /// [`TaehvDecoder::decode_clip_last_frames`] clip of `clip_frames` latent frames: every frame
    /// of the clip runs the head and the MemBlock stages (the memory crosses frames); only the
    /// clip's final output frame runs the full-resolution tail.
    pub fn clip_training_working_set_bytes(&self, out_h: u32, out_w: u32, clip_frames: u32) -> u64 {
        let clip = clip_frames.max(1) as u64;
        let up = self.spatial_upscale() as u64;
        let (lh, lw) = (
            (out_h as u64).div_ceil(up).max(1),
            (out_w as u64).div_ceil(up).max(1),
        );
        let nf = self.channels.map(|c| c as u64);
        // Clamp + conv_in + ReLU at the latent resolution.
        let mut area = lh * lw;
        let mut elems = 3 * clip * area * nf[0].max(self.latent_channels as u64);
        let mut frames = clip;
        for s in 0..3 {
            let c = nf[s];
            // Each MemBlock retains: concat (2C), the zero-padded past (2C: pad + slice), three conv
            // outputs, two ReLUs, the residual sum and the fused ReLU (~11C), per frame.
            elems += 3 * 11 * frames * area * c;
            let sc = self.space_scale(s) as u64;
            let up_area = area * sc * sc;
            if s < 2 {
                let stride = self.time_stride(s) as u64;
                // Upsample, TGrow output + its frame-split copy, the stage conv.
                elems += frames * up_area * c; // upsample
                elems += 2 * frames * stride * up_area * c; // TGrow (+ split copy)
                frames *= stride;
                elems += frames * up_area * nf[s + 1]; // stage conv
            } else {
                // Per-frame tail: one frame at full resolution — last-frame slice, upsample, the
                // TGrow chunk, conv, ReLU, the output conv, pixel-shuffle copy and clamp.
                let p2 = (self.patch_size * self.patch_size) as u64;
                elems += area * c; // last-frame slice
                elems += 4 * up_area * nf[3]; // upsample, TGrow chunk, conv, ReLU
                elems += 3 * up_area * 3 * p2; // output conv, shuffle copy, clamp
            }
            area = up_area;
        }
        elems * 4 * 2
    }
}

/// One TAEHV MemBlock (`n_in == n_out` everywhere in the decoder, so the skip is the identity).
struct MemBlock {
    convs: [(Array, Array); 3],
}

impl MemBlock {
    /// `x`: `[n·T, h, w, C]` (clip-major). `past` is `x` shifted one frame later inside each clip
    /// with zeros for each clip's first frame.
    fn forward(&self, x: &Array, clips: i32) -> Result<Array> {
        let sh = x.shape();
        let (nt, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
        let t = nt / clips;
        let past = if t == 1 {
            Array::zeros::<f32>(&[nt, h, w, c])?
        } else {
            let xr = x.reshape(&[clips, t, h, w, c])?;
            let zero = Array::zeros::<f32>(&[clips, 1, h, w, c])?;
            let prev = xr.index((.., ..t - 1));
            concatenate_axis(&[&zero, &prev], 1)?.reshape(&[nt, h, w, c])?
        };
        let cat = concatenate_axis(&[x, &past], 3)?;
        let mut y = conv2d(&cat, &self.convs[0].0, Some(&self.convs[0].1), 1, 1)?;
        y = mlx_rs::nn::relu(&y)?;
        y = conv2d(&y, &self.convs[1].0, Some(&self.convs[1].1), 1, 1)?;
        y = mlx_rs::nn::relu(&y)?;
        y = conv2d(&y, &self.convs[2].0, Some(&self.convs[2].1), 1, 1)?;
        Ok(mlx_rs::nn::relu(&add(&y, x)?)?)
    }
}

struct Stage {
    blocks: [MemBlock; 3],
    /// TGrow 1×1 conv, OHWI `[C·stride, 1, 1, C]`.
    tgrow: Array,
    stride: i32,
    space: i32,
    /// Stage-closing 3×3 conv (no bias), OHWI.
    conv: Array,
}

/// The loaded, frozen TAEHV decoder.
pub struct TaehvDecoder {
    cfg: TaehvConfig,
    conv_in: (Array, Array),
    stages: [Stage; 3],
    conv_out: (Array, Array),
    param_bytes: u64,
}

/// Permute a torch conv weight `[out, in, kH, kW]` → MLX `[out, kH, kW, in]`.
fn ohwi(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// TGrow: per-frame 1×1 conv to `C·stride` channels, then split the channel axis into `stride`
/// consecutive frames (torch `reshape(-1, C, H, W)` on NCHW).
fn tgrow(x: &Array, w: &Array, stride: i32) -> Result<Array> {
    let y = conv2d(x, w, None, 1, 0)?;
    if stride == 1 {
        return Ok(y);
    }
    let sh = y.shape();
    let (nt, h, wd, cs) = (sh[0], sh[1], sh[2], sh[3]);
    let c = cs / stride;
    Ok(y.reshape(&[nt, h, wd, stride, c])?
        .transpose_axes(&[0, 3, 1, 2, 4])?
        .reshape(&[nt * stride, h, wd, c])?)
}

/// `F.pixel_shuffle(x, p)` on NHWC: channel `c·p² + i·p + j` → pixel `(h·p + i, w·p + j)`, channel
/// `c`.
fn pixel_shuffle(x: &Array, p: i32) -> Result<Array> {
    if p == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, h, w, cp) = (sh[0], sh[1], sh[2], sh[3]);
    let c = cp / (p * p);
    Ok(x.reshape(&[b, h, w, c, p, p])?
        .transpose_axes(&[0, 1, 4, 2, 5, 3])?
        .reshape(&[b, h * p, w * p, c])?)
}

impl TaehvDecoder {
    /// Load from a checkpoint file, or from a directory holding `<name>.safetensors` directly, under
    /// `vae/` (the `Kijai/LTX2.3_comfy` mirror layout) or under `safetensors/` (the upstream
    /// `madebyollin/taehv` repo layout). The file is picked by name only — never "the only
    /// safetensors in the dir". A missing checkpoint is an error naming the decoder.
    pub fn from_path(path: impl AsRef<Path>, cfg: TaehvConfig) -> Result<Self> {
        let path = path.as_ref();
        let missing = || {
            Error::Msg(format!(
                "{} decoder checkpoint ({}) not found at {}",
                cfg.display_name(),
                cfg.file_name(),
                path.display()
            ))
        };
        let file = if path.is_file() {
            path.to_path_buf()
        } else {
            // By NAME only: a snapshot root (e.g. a multi-component mirror repo) can hold other
            // safetensors, so "the only/first safetensors in the dir" would load the wrong file.
            ["", "vae", "safetensors"]
                .iter()
                .map(|sub| path.join(sub).join(cfg.file_name()))
                .find(|p| p.is_file())
                .ok_or_else(missing)?
        };
        let w = Weights::from_file(&file)
            .map_err(|e| Error::Msg(format!("{}: {e}", cfg.display_name())))?;
        Self::from_weights(&w, cfg)
    }

    /// Build from already-read weights in the reference `decoder.{i}.…` layout. Every tensor's
    /// shape is checked against `cfg`; an unread `decoder.*` key is an error.
    pub fn from_weights(w: &Weights, cfg: TaehvConfig) -> Result<Self> {
        let name = cfg.display_name();
        let mut bytes = 0u64;
        let mut get = |key: &str, shape: &[i32]| -> Result<Array> {
            let a = w.require(key)?;
            if a.shape() != shape {
                return Err(Error::Msg(format!(
                    "{name}: `{key}` has shape {:?}, expected {shape:?} (wrong TAEHV variant?)",
                    a.shape()
                )));
            }
            let a = a.as_dtype(Dtype::Float32)?;
            bytes += a.nbytes() as u64;
            Ok(a)
        };
        let nf = cfg.channels;
        let conv_in = (
            ohwi(&get(
                "decoder.1.weight",
                &[nf[0], cfg.latent_channels, 3, 3],
            )?)?,
            get("decoder.1.bias", &[nf[0]])?,
        );
        let mut stages = Vec::with_capacity(3);
        for s in 0..3 {
            let base = 3 + 6 * s;
            let c = nf[s];
            let mut blocks = Vec::with_capacity(3);
            for b in 0..3 {
                let p = format!("decoder.{}.conv", base + b);
                let leg = |get: &mut dyn FnMut(&str, &[i32]) -> Result<Array>,
                           j: usize,
                           cin: i32|
                 -> Result<(Array, Array)> {
                    Ok((
                        ohwi(&get(&format!("{p}.{j}.weight"), &[c, cin, 3, 3])?)?,
                        get(&format!("{p}.{j}.bias"), &[c])?,
                    ))
                };
                blocks.push(MemBlock {
                    convs: [
                        leg(&mut get, 0, 2 * c)?,
                        leg(&mut get, 2, c)?,
                        leg(&mut get, 4, c)?,
                    ],
                });
            }
            let stride = cfg.time_stride(s);
            let tgrow = ohwi(&get(
                &format!("decoder.{}.conv.weight", base + 4),
                &[c * stride, c, 1, 1],
            )?)?;
            let conv = ohwi(&get(
                &format!("decoder.{}.weight", base + 5),
                &[nf[s + 1], c, 3, 3],
            )?)?;
            let blocks: [MemBlock; 3] = blocks
                .try_into()
                .map_err(|_| Error::Msg("taehv: three MemBlocks per stage".into()))?;
            stages.push(Stage {
                blocks,
                tgrow,
                stride,
                space: cfg.space_scale(s),
                conv,
            });
        }
        let out_c = 3 * cfg.patch_size * cfg.patch_size;
        let conv_out = (
            ohwi(&get("decoder.22.weight", &[out_c, nf[3], 3, 3])?)?,
            get("decoder.22.bias", &[out_c])?,
        );
        let stray: Vec<&str> = w
            .unused_keys()
            .into_iter()
            .filter(|k| k.starts_with("decoder."))
            .collect();
        if !stray.is_empty() {
            return Err(Error::Msg(format!(
                "{name}: checkpoint carries decoder keys this architecture does not read \
                 (wrong TAEHV variant?): {stray:?}"
            )));
        }
        let stages: [Stage; 3] = stages
            .try_into()
            .map_err(|_| Error::Msg("taehv: three stages".into()))?;
        Ok(Self {
            cfg,
            conv_in,
            stages,
            conv_out,
            param_bytes: bytes,
        })
    }

    /// The loaded configuration.
    pub fn config(&self) -> &TaehvConfig {
        &self.cfg
    }

    /// Resident parameter bytes (f32).
    pub fn param_bytes(&self) -> u64 {
        self.param_bytes
    }

    /// Clamp → conv_in → ReLU on NHWC `[n, h, w, C]`.
    fn head(&self, x: &Array) -> Result<Array> {
        let three = Array::from_f32(3.0);
        let h = multiply(&tanh(&divide(x, &three)?)?, &three)?;
        let h = conv2d(&h, &self.conv_in.0, Some(&self.conv_in.1), 1, 1)?;
        Ok(mlx_rs::nn::relu(&h)?)
    }

    /// ReLU → output conv → pixel-shuffle → clamp [0, 1] on NHWC.
    fn tail(&self, x: &Array) -> Result<Array> {
        let h = mlx_rs::nn::relu(x)?;
        let h = conv2d(&h, &self.conv_out.0, Some(&self.conv_out.1), 1, 1)?;
        let h = pixel_shuffle(&h, self.cfg.patch_size)?;
        Ok(clip(&h, (&Array::from_f32(0.0), &Array::from_f32(1.0)))?)
    }

    fn check_channels(&self, c: i32, sh: &[i32]) -> Result<()> {
        if c != self.cfg.latent_channels {
            return Err(Error::Msg(format!(
                "{} expects {} latent channels, got shape {sh:?}",
                self.cfg.display_name(),
                self.cfg.latent_channels
            )));
        }
        Ok(())
    }

    /// The reference `decode_video` (parallel mode): latents `[N, T, C, h, w]` (NTCHW) → frames
    /// NHWC `[N, T·t_upscale − frames_to_trim, H, W, 3]` in `[0, 1]`. Differentiable.
    pub fn decode_video(&self, latents: &Array) -> Result<Array> {
        let sh = latents.shape().to_vec();
        if sh.len() != 5 {
            return Err(Error::Msg(format!(
                "{} decode_video expects NTCHW latents, got shape {sh:?}",
                self.cfg.display_name()
            )));
        }
        self.check_channels(sh[2], &sh)?;
        let (n, t, c, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let x = latents
            .reshape(&[n * t, c, h, w])?
            .transpose_axes(&[0, 2, 3, 1])?;
        let mut x = self.head(&x)?;
        for st in &self.stages {
            for b in &st.blocks {
                x = b.forward(&x, n)?;
            }
            if st.space > 1 {
                x = upsample_nearest(&x, st.space)?;
            }
            x = tgrow(&x, &st.tgrow, st.stride)?;
            x = conv2d(&x, &st.conv, None, 1, 1)?;
        }
        let x = self.tail(&x)?;
        let osh = x.shape().to_vec();
        let frames = osh[0] / n;
        let x = x.reshape(&[n, frames, osh[1], osh[2], osh[3]])?;
        Ok(x.index((.., self.cfg.frames_to_trim()..)))
    }

    /// Decode `N` latent frames **each as an independent `T = 1` clip** (the reference's
    /// `decode_video(z[:, None])`, i.e. the last of the `t_upscale` grown frames after the
    /// `frames_to_trim` trim): NCHW `[N, C, h, w]` → NHWC `[N, H, W, 3]` in `[0, 1]`. Pure MLX
    /// ops (no host round trip, no stop-gradient) — differentiable in `latents`.
    pub fn decode_frames(&self, latents: &Array) -> Result<Array> {
        let sh = latents.shape().to_vec();
        if sh.len() != 4 {
            return Err(Error::Msg(format!(
                "{} decode_frames expects NCHW latents, got shape {sh:?}",
                self.cfg.display_name()
            )));
        }
        self.decode_clip_last_frames(&latents.reshape(&[sh[0], 1, sh[1], sh[2], sh[3]])?)
    }

    /// Decode `N` clips of `T` latent frames each and keep **only each clip's final output
    /// frame** — the last of the last latent frame's `t_upscale` grown frames, i.e.
    /// `decode_video(latents)[:, -1]`, with the MemBlocks carrying the clip's earlier frames:
    /// NTCHW `[N, T, C, h, w]` → NHWC `[N, H, W, 3]` in `[0, 1]`. Only that frame's
    /// full-resolution tail is computed (the last stage-2 frame through the last `n_f[2]` TGrow
    /// output channels — no MemBlock follows the final TGrow). Pure MLX ops — differentiable in
    /// `latents`.
    pub fn decode_clip_last_frames(&self, latents: &Array) -> Result<Array> {
        let sh = latents.shape().to_vec();
        if sh.len() != 5 {
            return Err(Error::Msg(format!(
                "{} decode_clip_last_frames expects NTCHW latents, got shape {sh:?}",
                self.cfg.display_name()
            )));
        }
        self.check_channels(sh[2], &sh)?;
        let (n, t, c, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let x = latents
            .reshape(&[n * t, c, h, w])?
            .transpose_axes(&[0, 2, 3, 1])?;
        let mut x = self.head(&x)?;
        let last = self.stages.len() - 1;
        for (s, st) in self.stages.iter().enumerate() {
            for b in &st.blocks {
                x = b.forward(&x, n)?;
            }
            if s == last {
                // Only the clip's final output frame survives the trim: the last frame's last
                // TGrow chunk (no MemBlock follows, so nothing else feeds it).
                let xs = x.shape().to_vec();
                let f = xs[0] / n;
                x = x
                    .reshape(&[n, f, xs[1], xs[2], xs[3]])?
                    .index((.., f - 1..))
                    .reshape(&[n, xs[1], xs[2], xs[3]])?;
            }
            if st.space > 1 {
                x = upsample_nearest(&x, st.space)?;
            }
            if s == last {
                let c = self.cfg.channels[s];
                let w_last = st.tgrow.index((st.stride - 1) * c..);
                x = conv2d(&x, &w_last, None, 1, 0)?;
            } else {
                x = tgrow(&x, &st.tgrow, st.stride)?;
            }
            x = conv2d(&x, &st.conv, None, 1, 1)?;
        }
        self.tail(&x)
    }
}

impl X0Decoder for TaehvDecoder {
    fn decode(&self, latents: &Array) -> Result<Array> {
        self.decode_frames(latents)
    }
}

/// A complete random-init TAEHV decoder checkpoint for `cfg` in the reference key layout (torch
/// OIHW conv weights, decoder keys only) — for tests, which must never download real weights.
/// Deterministic in `seed`.
pub fn synthetic_taehv_weights(cfg: &TaehvConfig, seed: u64) -> Result<Weights> {
    let mut w = Weights::empty();
    let mut n = 0u64;
    let mut rnd = |shape: &[i32]| -> Result<Array> {
        n += 1;
        let fan_in: i32 = shape[1..].iter().product::<i32>().max(1);
        let std = (2.0 / fan_in as f32).sqrt();
        let key = random::key(seed.wrapping_mul(1_000_003).wrapping_add(n))?;
        Ok(multiply(
            &random::normal::<f32>(shape, None, None, Some(&key))?,
            Array::from_f32(std),
        )?)
    };
    let bias = |w: &mut Weights,
                key: String,
                c: i32,
                rnd: &mut dyn FnMut(&[i32]) -> Result<Array>|
     -> Result<()> {
        w.insert(
            key,
            multiply(&rnd(&[c, 1])?.reshape(&[c])?, Array::from_f32(0.1))?,
        );
        Ok(())
    };
    let nf = cfg.channels;
    w.insert(
        "decoder.1.weight",
        rnd(&[nf[0], cfg.latent_channels, 3, 3])?,
    );
    bias(&mut w, "decoder.1.bias".into(), nf[0], &mut rnd)?;
    for s in 0..3 {
        let base = 3 + 6 * s;
        let c = nf[s];
        for b in 0..3 {
            for (j, cin) in [(0, 2 * c), (2, c), (4, c)] {
                w.insert(
                    format!("decoder.{}.conv.{j}.weight", base + b),
                    rnd(&[c, cin, 3, 3])?,
                );
                bias(
                    &mut w,
                    format!("decoder.{}.conv.{j}.bias", base + b),
                    c,
                    &mut rnd,
                )?;
            }
        }
        w.insert(
            format!("decoder.{}.conv.weight", base + 4),
            rnd(&[c * cfg.time_stride(s), c, 1, 1])?,
        );
        w.insert(
            format!("decoder.{}.weight", base + 5),
            rnd(&[nf[s + 1], c, 3, 3])?,
        );
    }
    let out_c = 3 * cfg.patch_size * cfg.patch_size;
    w.insert("decoder.22.weight", rnd(&[out_c, nf[3], 3, 3])?);
    bias(&mut w, "decoder.22.bias".into(), out_c, &mut rnd)?;
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{abs, subtract};
    use mlx_rs::transforms::{eval, grad};

    /// A variant's real hyperparameters with tiny stage widths.
    fn tiny(cfg: TaehvConfig) -> TaehvConfig {
        TaehvConfig {
            channels: [8, 6, 4, 4],
            ..cfg
        }
    }

    fn load(cfg: &TaehvConfig, seed: u64) -> TaehvDecoder {
        TaehvDecoder::from_weights(&synthetic_taehv_weights(cfg, seed).unwrap(), cfg.clone())
            .unwrap()
    }

    fn latents(shape: &[i32], seed: u64) -> Array {
        random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap())).unwrap()
    }

    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        let d = abs(subtract(a, b).unwrap()).unwrap().max(None).unwrap();
        eval([&d]).unwrap();
        d.item::<f32>()
    }

    #[test]
    fn variant_hyperparameters_match_the_reference() {
        // taehv.py name guessing: taew2_1 → patch 1 / 16ch / time (F,T,T); taew2_2 → patch 2 /
        // 48ch; taeltx → patch 4 / 128ch / time (T,T,T). frames_to_trim = t_upscale − 1.
        let w21 = TaehvConfig::taew2_1();
        assert_eq!((w21.latent_channels, w21.patch_size), (16, 1));
        assert_eq!(
            (w21.t_upscale(), w21.frames_to_trim(), w21.spatial_upscale()),
            (4, 3, 8)
        );
        let w22 = TaehvConfig::taew2_2();
        assert_eq!((w22.latent_channels, w22.patch_size), (48, 2));
        assert_eq!(
            (w22.t_upscale(), w22.frames_to_trim(), w22.spatial_upscale()),
            (4, 3, 16)
        );
        let ltx = TaehvConfig::taeltx2_3();
        assert_eq!((ltx.latent_channels, ltx.patch_size), (128, 4));
        assert_eq!(
            (ltx.t_upscale(), ltx.frames_to_trim(), ltx.spatial_upscale()),
            (8, 7, 32)
        );
        assert_eq!(TaehvConfig::for_wan_z_dim(16), Some(w21));
        assert_eq!(TaehvConfig::for_wan_z_dim(48), Some(w22));
        assert_eq!(TaehvConfig::for_wan_z_dim(128), None);
    }

    #[test]
    fn full_size_key_layout_matches_the_shipped_checkpoints_and_is_fully_consumed() {
        // Shapes from the upstream safetensors headers (rev 011dfc21): decoder.1.weight
        // [256, z, 3, 3]; TGrow decoder.7.conv.weight [256·s0, 256, 1, 1] (256 for Wan, 512 for
        // LTX); decoder.22.weight [3·p², 64, 3, 3] (3 / 12 / 48). 64 decoder tensors per variant.
        for (cfg, tgrow0, out_c) in [
            (TaehvConfig::taew2_1(), 256, 3),
            (TaehvConfig::taew2_2(), 256, 12),
            (TaehvConfig::taeltx2_3(), 512, 48),
        ] {
            let w = synthetic_taehv_weights(&cfg, 1).unwrap();
            assert_eq!(w.len(), 64, "{}", cfg.name);
            assert_eq!(
                w.get("decoder.7.conv.weight").unwrap().shape(),
                &[tgrow0, 256, 1, 1]
            );
            assert_eq!(
                w.get("decoder.13.conv.weight").unwrap().shape(),
                &[256, 128, 1, 1]
            );
            assert_eq!(
                w.get("decoder.19.conv.weight").unwrap().shape(),
                &[128, 64, 1, 1]
            );
            assert_eq!(
                w.get("decoder.22.weight").unwrap().shape(),
                &[out_c, 64, 3, 3]
            );
            assert!(
                w.get("decoder.8.bias").is_none(),
                "stage convs carry no bias"
            );
            let dec = TaehvDecoder::from_weights(&w, cfg.clone()).unwrap();
            assert!(w.unused_keys().is_empty(), "unused: {:?}", w.unused_keys());
            assert_eq!(dec.param_bytes(), cfg.param_count() * 4);
        }
    }

    #[test]
    fn encoder_keys_are_ignored_but_stray_decoder_keys_and_wrong_shapes_refuse() {
        let cfg = tiny(TaehvConfig::taew2_1());
        let mut w = synthetic_taehv_weights(&cfg, 3).unwrap();
        w.insert(
            "encoder.0.weight",
            Array::zeros::<f32>(&[8, 3, 3, 3]).unwrap(),
        );
        assert!(TaehvDecoder::from_weights(&w, cfg.clone()).is_ok());
        w.insert("decoder.23.weight", Array::zeros::<f32>(&[1]).unwrap());
        let err = TaehvDecoder::from_weights(&w, cfg.clone()).err().unwrap();
        assert!(err.to_string().contains("decoder.23.weight"), "{err}");
        // A taew2_2 checkpoint loaded as taew2_1 (48 vs 16 latent channels) is a named error.
        let w22 = synthetic_taehv_weights(&tiny(TaehvConfig::taew2_2()), 3).unwrap();
        let err = TaehvDecoder::from_weights(&w22, cfg).err().unwrap();
        assert!(err.to_string().contains("TAEW2.1"), "{err}");
    }

    #[test]
    fn missing_checkpoint_names_the_decoder() {
        let dir = tempfile::tempdir().unwrap();
        let err = TaehvDecoder::from_path(dir.path(), TaehvConfig::taeltx2_3())
            .err()
            .unwrap()
            .to_string();
        assert!(
            err.contains("TAELTX2.3") && err.contains("taeltx2_3.safetensors"),
            "{err}"
        );
    }

    #[test]
    fn checkpoint_is_picked_by_name_never_by_being_the_only_file() {
        let cfg = tiny(TaehvConfig::taew2_1());
        let w = synthetic_taehv_weights(&cfg, 2).unwrap();
        let pairs: Vec<(String, Array)> = w
            .keys()
            .map(|k| (k.to_string(), w.require(k).unwrap().clone()))
            .collect();
        let refs: Vec<(&str, &Array)> = pairs.iter().map(|(k, a)| (k.as_str(), a)).collect();
        let dir = tempfile::tempdir().unwrap();
        // A mirror snapshot root holding an unrelated component: not taken as the decoder.
        Array::save_safetensors(
            vec![("x", &Array::zeros::<f32>(&[1]).unwrap())],
            None,
            dir.path().join("other_model.safetensors"),
        )
        .unwrap();
        let err = TaehvDecoder::from_path(dir.path(), cfg.clone())
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("taew2_1.safetensors"), "{err}");
        for sub in ["", "vae", "safetensors"] {
            let d = dir.path().join(format!("layout{sub}"));
            std::fs::create_dir_all(d.join(sub)).unwrap();
            Array::save_safetensors(refs.clone(), None, d.join(sub).join("taew2_1.safetensors"))
                .unwrap();
            let dec = TaehvDecoder::from_path(&d, cfg.clone()).unwrap();
            assert_eq!(dec.param_bytes(), cfg.param_count() * 4, "layout {sub:?}");
        }
        // The file path itself.
        let f = dir.path().join("layout/taew2_1.safetensors");
        assert!(TaehvDecoder::from_path(&f, cfg).is_ok());
    }

    #[test]
    fn per_frame_shapes_per_variant() {
        for (cfg, up) in [
            (tiny(TaehvConfig::taew2_1()), 8),
            (tiny(TaehvConfig::taew2_2()), 16),
            (tiny(TaehvConfig::taeltx2_3()), 32),
        ] {
            let dec = load(&cfg, 4);
            let z = latents(&[3, cfg.latent_channels, 2, 3], 9);
            let px = dec.decode_frames(&z).unwrap();
            assert_eq!(px.shape(), &[3, 2 * up, 3 * up, 3], "{}", cfg.name);
            eval([&px]).unwrap();
            assert!(px.as_slice::<f32>().iter().all(|v| (0.0..=1.0).contains(v)));
        }
    }

    #[test]
    fn per_frame_decode_equals_the_reference_t1_clip_with_its_trim() {
        // The per-frame path must equal decode_video on N independent T=1 clips (t_upscale
        // grown frames, first frames_to_trim dropped ⇒ exactly one frame per clip).
        for cfg in [tiny(TaehvConfig::taew2_1()), tiny(TaehvConfig::taeltx2_3())] {
            let dec = load(&cfg, 5);
            let z = latents(&[2, cfg.latent_channels, 2, 2], 11);
            let fast = dec.decode_frames(&z).unwrap();
            let sh = z.shape().to_vec();
            let video = dec
                .decode_video(&z.reshape(&[sh[0], 1, sh[1], sh[2], sh[3]]).unwrap())
                .unwrap();
            let vs = video.shape().to_vec();
            assert_eq!(vs[1], 1, "{}: T=1 ⇒ one frame after the trim", cfg.name);
            let video = video.reshape(&[vs[0], vs[2], vs[3], vs[4]]).unwrap();
            assert!(max_abs_diff(&fast, &video) < 1e-5, "{}", cfg.name);
        }
    }

    #[test]
    fn video_decode_grows_and_trims_and_memory_crosses_frames() {
        let cfg = tiny(TaehvConfig::taew2_1());
        let dec = load(&cfg, 6);
        let z = latents(&[1, 3, cfg.latent_channels, 2, 2], 12);
        let v = dec.decode_video(&z).unwrap();
        // 3 latent frames × 4 − 3 trimmed = 9 frames.
        assert_eq!(v.shape(), &[1, 9, 16, 16, 3]);
        // The last output frame of a 3-frame clip differs from decoding the last latent alone:
        // MemBlocks carry the earlier frames' state (≈1.0 apart here; with the memory cut the two
        // agree up to Metal small-channel conv rounding, < 1e-2, so the bar sits well between).
        let alone = dec
            .decode_frames(
                &z.index((.., 2))
                    .reshape(&[1, cfg.latent_channels, 2, 2])
                    .unwrap(),
            )
            .unwrap();
        let last = v.index((.., 8)).reshape(&[1, 16, 16, 3]).unwrap();
        assert!(max_abs_diff(&alone, &last) > 0.1);
    }

    /// `decode_clip_last_frames` on `T = 2` clips is exactly the last frame of the reference
    /// `decode_video` of the same clips (the last latent frame's last grown frame, decoded with its
    /// predecessor's memory), one frame per clip. Mutations: keep the clip's first stage-2 frame
    /// (`f - 1` → `0`) ⇒ red; cut the cross-frame memory (MemBlocks run with `clips = n·t`) ⇒ red.
    #[test]
    fn clip_last_frame_decode_equals_the_reference_clips_last_frame() {
        for cfg in [tiny(TaehvConfig::taew2_1()), tiny(TaehvConfig::taeltx2_3())] {
            let dec = load(&cfg, 8);
            let z = latents(&[2, 2, cfg.latent_channels, 2, 2], 14);
            let fast = dec.decode_clip_last_frames(&z).unwrap();
            let up = cfg.spatial_upscale();
            assert_eq!(fast.shape(), &[2, 2 * up, 2 * up, 3], "{}", cfg.name);
            let video = dec.decode_video(&z).unwrap();
            let vs = video.shape().to_vec();
            assert_eq!(
                vs[1],
                2 * cfg.t_upscale() - cfg.frames_to_trim(),
                "{}",
                cfg.name
            );
            let last = video
                .index((.., vs[1] - 1))
                .reshape(&[vs[0], vs[2], vs[3], vs[4]])
                .unwrap();
            assert!(max_abs_diff(&fast, &last) < 1e-5, "{}", cfg.name);
        }
    }

    /// A `T`-frame clip's working set grows with `T` (head + MemBlock stages run on every frame)
    /// but stays under `T`× the per-frame figure (one full-resolution tail per clip). Mutation:
    /// ignore `clip_frames` ⇒ red.
    #[test]
    fn clip_working_set_grows_with_the_clip_length() {
        let cfg = TaehvConfig::taeltx2_3();
        let one = cfg.clip_training_working_set_bytes(512, 512, 1);
        let two = cfg.clip_training_working_set_bytes(512, 512, 2);
        assert_eq!(one, cfg.training_working_set_bytes(512, 512));
        assert!(two > one && two < 2 * one, "{one} {two}");
    }

    #[test]
    fn per_frame_decode_is_differentiable() {
        let cfg = tiny(TaehvConfig::taew2_2());
        let dec = load(&cfg, 7);
        let z = latents(&[2, cfg.latent_channels, 2, 2], 13);
        let f = |z: &Array| -> mlx_rs::error::Result<Array> {
            dec.decode(z)
                .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))?
                .sum(None)
        };
        let g = grad(f)(&z).unwrap();
        eval([&g]).unwrap();
        assert_eq!(g.shape(), z.shape());
        assert!(g.abs().unwrap().sum(None).unwrap().item::<f32>() > 0.0);
    }

    #[test]
    fn footprint_scales_with_area_and_counts_f32_weights() {
        for cfg in [
            TaehvConfig::taew2_1(),
            TaehvConfig::taew2_2(),
            TaehvConfig::taeltx2_3(),
        ] {
            let a = cfg.footprint(512, 512);
            let b = cfg.footprint(1024, 1024);
            assert_eq!(a.param_bytes, cfg.param_count() * 4);
            assert_eq!(b.working_set_bytes, a.working_set_bytes * 4, "{}", cfg.name);
            assert!(a.working_set_bytes > 0);
        }
    }

    /// splitmix64(seed·2³² + j) → uniform [−1, 1) (top 53 bits) · scale·√3 + offset, in f64 — the
    /// exact generator of the upstream-parity script (see the golden test below).
    fn splitmix_uniform(shape: &[i32], seed: u64, scale: f64, offset: f64) -> Array {
        let n: i32 = shape.iter().product();
        let v: Vec<f32> = (0..n as u64)
            .map(|j| {
                let mut z = (j.wrapping_add(seed << 32)).wrapping_add(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
                (u * scale * 3f64.sqrt() + offset) as f32
            })
            .collect();
        Array::from_slice(&v, shape)
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn full_size_decode_matches_upstream_taehv_py_golden_values() {
        // Golden values from the UPSTREAM reference (madebyollin/taehv taehv.py @ 011dfc21,
        // `TAEHV(checkpoint_path=None, arch_name=name).double()`, `decode_video(parallel=True)`)
        // with deterministic splitmix64 weights: decoder.22.bias = 0.5 + U·0.05, other biases
        // U·0.05, weights U·√(1/fan_in) (U uniform, unit variance), key seed = Σ key bytes mod 997;
        // latents U (seed 7) of shape [1, 2, C, 2, 3]. Rows: (variant, video mean, video samples
        // (flat NTHWC index, value), frame-1-alone mean, frame-1-alone samples).
        #[allow(clippy::type_complexity)]
        let golden: &[(&str, f32, &[(usize, f32)], f32, &[(usize, f32)])] = &[
            (
                "taew2_1",
                0.5453534,
                &[
                    (0, 0.5278884),
                    (383, 0.5084928),
                    (767, 0.6362164),
                    (1151, 0.4839259),
                    (1535, 0.2605670),
                    (1919, 0.2949821),
                    (2303, 0.4986200),
                    (2687, 0.7761875),
                    (3071, 0.6646734),
                    (3455, 0.3471640),
                    (3839, 0.6241932),
                    (4223, 0.4906073),
                    (4607, 0.4165053),
                    (4991, 0.5359723),
                    (5375, 0.6930664),
                    (5759, 0.4709628),
                ],
                0.5575177,
                &[
                    (0, 0.4963038),
                    (76, 0.5129942),
                    (153, 0.4917771),
                    (230, 0.5424019),
                    (306, 0.5916426),
                    (383, 0.4937346),
                    (460, 0.5383758),
                    (537, 0.6685046),
                    (613, 0.6094354),
                    (690, 0.4107044),
                    (767, 0.6319227),
                    (844, 0.6082115),
                    (920, 0.7683413),
                    (997, 0.6360722),
                    (1074, 0.3444287),
                    (1151, 0.4861233),
                ],
            ),
            (
                "taew2_2",
                0.4599152,
                &[
                    (0, 0.5360567),
                    (1535, 0.1798162),
                    (3071, 0.5886377),
                    (4607, 0.5701227),
                    (6143, 0.4623908),
                    (7679, 0.3602138),
                    (9215, 0.3585008),
                    (10751, 0.4661903),
                    (12287, 0.6765762),
                    (13823, 0.4666952),
                    (15359, 0.5489842),
                    (16895, 0.1881633),
                    (18431, 0.4183862),
                    (19967, 0.1838195),
                    (21503, 0.3284745),
                    (23039, 0.5554451),
                ],
                0.4340227,
                &[
                    (0, 0.5102840),
                    (307, 0.4135604),
                    (614, 0.4558109),
                    (921, 0.5233188),
                    (1228, 0.7189682),
                    (1535, 0.2424411),
                    (1842, 0.4115649),
                    (2149, 0.3088579),
                    (2457, 0.1860975),
                    (2764, 0.3281009),
                    (3071, 0.3731965),
                    (3378, 0.6654694),
                    (3685, 0.2662813),
                    (3992, 0.5445945),
                    (4299, 0.6135635),
                    (4607, 0.5529289),
                ],
            ),
            (
                "taeltx2_3",
                0.4776780,
                &[
                    (0, 0.5787073),
                    (11059, 0.7197583),
                    (22118, 0.3439939),
                    (33177, 0.4524901),
                    (44236, 0.4893995),
                    (55295, 0.2780566),
                    (66354, 0.5553743),
                    (77413, 0.8660297),
                    (88473, 0.6473224),
                    (99532, 0.3645075),
                    (110591, 0.4368899),
                    (121650, 0.1053346),
                    (132709, 0.8250001),
                    (143768, 0.4557823),
                    (154827, 0.7226046),
                    (165887, 0.3213555),
                ],
                0.4809216,
                &[
                    (0, 0.5433555),
                    (1228, 0.0000000),
                    (2457, 0.3800583),
                    (3686, 0.4027431),
                    (4914, 0.5836142),
                    (6143, 0.6946127),
                    (7372, 0.4173975),
                    (8601, 0.1996691),
                    (9829, 0.7288970),
                    (11058, 0.5623746),
                    (12287, 0.6670548),
                    (13516, 0.6050438),
                    (14744, 0.5309826),
                    (15973, 0.3135534),
                    (17202, 0.2649713),
                    (18431, 0.3214565),
                ],
            ),
        ];
        for &(name, mean, samples, mean1, samples1) in golden {
            let cfg = match name {
                "taew2_1" => TaehvConfig::taew2_1(),
                "taew2_2" => TaehvConfig::taew2_2(),
                _ => TaehvConfig::taeltx2_3(),
            };
            let shapes = synthetic_taehv_weights(&cfg, 0).unwrap();
            let mut w = Weights::empty();
            for k in shapes.keys() {
                let shape = shapes.get(k).unwrap().shape().to_vec();
                let seed = k.bytes().map(u64::from).sum::<u64>() % 997;
                let fan_in: i32 = shape[1..].iter().product::<i32>().max(1);
                let v = match k {
                    "decoder.22.bias" => splitmix_uniform(&shape, seed, 0.05, 0.5),
                    _ if k.ends_with("bias") => splitmix_uniform(&shape, seed, 0.05, 0.0),
                    _ => splitmix_uniform(&shape, seed, (1.0 / fan_in as f64).sqrt(), 0.0),
                };
                w.insert(k.to_string(), v);
            }
            let dec = TaehvDecoder::from_weights(&w, cfg.clone()).unwrap();
            let c = cfg.latent_channels;
            let z = splitmix_uniform(&[1, 2, c, 2, 3], 7, 1.0, 0.0);
            let video = dec.decode_video(&z).unwrap();
            let frame1 = dec
                .decode_frames(&z.index((.., 1..)).reshape(&[1, c, 2, 3]).unwrap())
                .unwrap();
            for (out, m, smp, what) in [
                (&video, mean, samples, "video"),
                (&frame1, mean1, samples1, "frame 1 alone"),
            ] {
                let flat = out.flatten(None, None).unwrap();
                eval([&flat]).unwrap();
                let v = flat.as_slice::<f32>();
                let got_mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
                assert!(
                    (got_mean as f32 - m).abs() < 1e-4,
                    "{name} {what} mean {got_mean} vs {m}"
                );
                for &(i, want) in smp {
                    assert!(
                        (v[i] - want).abs() < 2e-4,
                        "{name} {what} [{i}] {} vs {want}",
                        v[i]
                    );
                }
            }
        }
    }
}
