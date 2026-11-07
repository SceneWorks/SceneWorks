//! **E-LatentLPIPS** (epic 2123, sc-24833) — the learned latent-space perceptual metric of Kang et
//! al., *Diffusion2GAN* (ECCV 2024, `mingukkang/elatentlpips` @ f64cd65), as a frozen
//! [`PerceptualLoss`] on the shared perceptual path ([`super::perceptual`]) with
//! [`PerceptualInput::Latents`] input: it compares the model's x0 latent with the training image's
//! clean latent directly, no decode.
//!
//! Network (`ELatentLPIPS(net="vgg16")`): a VGG16-BN trunk whose first convolution takes the
//! family's latent channels and whose **first three max-pools are identities**
//! (`LatentVGG16BN`), sliced at torchvision feature indices `0-7 / 7-14 / 14-24 / 24-34 / 34-44`
//! (`relu1_2 … relu5_3`; slices 4 and 5 end with the two remaining 2×2 max-pools). Each slice's
//! output is unit-normalized over channels (`x / sqrt(Σc x² + 1e-8)`), the squared difference of the
//! two inputs' normalized features goes through a bias-free 1×1 "lin" head to one channel, and is
//! spatially averaged; the five layer terms are summed, plus the mean absolute latent difference
//! (`add_l1_loss=True`). This is exactly how ai-toolkit-perceptual calls it (`normalize=False` on
//! model-space latents, `ensembling=False`, `augment=None`), so the augmentation ensemble is not
//! part of the technique. BatchNorm runs in eval mode (running statistics, `eps = 1e-5`).
//!
//! Weights: the published `elatentlpips_ckpt/<family>_latest_vgg16_tuned.pth` state dicts
//! (`net.slice{s}.{i}.*`, `lin{k}.model.1.weight`), one per [`LatentLpipsFamily`]; a
//! `<stem>.safetensors` with the same keys is accepted too. Parity against the upstream PyTorch
//! module is pinned by `latent_perceptual_parity.json` (see [`super::formula`]).

use std::any::Any;
use std::path::{Path, PathBuf};

use mlx_rs::ops::{abs, add, divide, maximum, multiply, sqrt, subtract};
use mlx_rs::{Array, Dtype};

use gen_core::train::LatentLpipsFamily;

use super::formula::{formula_array, Role};
use super::perceptual::{
    reference_as, AuxModelFootprint, LossReference, PerceptualInput, PerceptualLoss,
};
use crate::nn::conv2d;
use crate::weights::Weights;
use crate::{Error, Result};

/// Output channels of the five trunk slices (`relu1_2 … relu5_3`).
pub const LPIPS_CHANNELS: [i32; 5] = [64, 128, 256, 512, 512];
/// torchvision `vgg16_bn` feature-index bounds of the five slices.
pub(crate) const SLICE_BOUNDS: [(usize, usize); 5] =
    [(0, 7), (7, 14), (14, 24), (24, 34), (34, 44)];
/// The `vgg16_bn` "D" config: conv widths, `0` = max-pool.
pub(crate) const VGG16_CFG: [i32; 18] = [
    64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0,
];
/// How many leading max-pools `LatentVGG16BN` replaces with identity.
const IDENTITY_POOLS: usize = 3;
pub(crate) const BN_EPS: f32 = 1e-5;
const NORMALIZE_EPS: f32 = 1e-8;

/// One feature-index layer of the latent VGG16-BN trunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerKind {
    /// 3×3 conv, padding 1, with bias: `(in, out)`.
    Conv(i32, i32),
    /// BatchNorm over `c` channels.
    Bn(i32),
    Relu,
    /// A max-pool `LatentVGG16BN` turned into `nn.Identity`.
    Identity,
    /// 2×2 / stride-2 max-pool.
    MaxPool,
}

/// The 44 feature-index layers of `LatentVGG16BN(latent_channels)`.
fn trunk_layers(latent_channels: i32) -> Vec<LayerKind> {
    let mut out = Vec::with_capacity(44);
    let mut cin = latent_channels;
    let mut pools = 0;
    for &v in &VGG16_CFG {
        if v == 0 {
            out.push(if pools < IDENTITY_POOLS {
                LayerKind::Identity
            } else {
                LayerKind::MaxPool
            });
            pools += 1;
        } else {
            out.push(LayerKind::Conv(cin, v));
            out.push(LayerKind::Bn(v));
            out.push(LayerKind::Relu);
            cin = v;
        }
    }
    out
}

/// The slice a feature index belongs to (1-based, as in the checkpoint keys).
fn slice_of(index: usize) -> usize {
    SLICE_BOUNDS
        .iter()
        .position(|&(lo, hi)| (lo..hi).contains(&index))
        .expect("feature index < 44")
        + 1
}

fn layer_key(index: usize) -> String {
    format!("net.slice{}.{index}", slice_of(index))
}

/// Every float tensor of the checkpoint the network reads, with its shape and formula role.
fn checkpoint_tensors(latent_channels: i32) -> Vec<(String, Vec<i32>, Role)> {
    let mut out = Vec::new();
    for (i, layer) in trunk_layers(latent_channels).into_iter().enumerate() {
        let k = layer_key(i);
        match layer {
            LayerKind::Conv(cin, cout) => {
                out.push((format!("{k}.weight"), vec![cout, cin, 3, 3], Role::Conv));
                out.push((format!("{k}.bias"), vec![cout], Role::Bias));
            }
            LayerKind::Bn(c) => {
                out.push((format!("{k}.weight"), vec![c], Role::NormWeight));
                out.push((format!("{k}.bias"), vec![c], Role::Bias));
                out.push((format!("{k}.running_mean"), vec![c], Role::BnMean));
                out.push((format!("{k}.running_var"), vec![c], Role::BnVar));
            }
            _ => {}
        }
    }
    for (k, &c) in LPIPS_CHANNELS.iter().enumerate() {
        out.push((
            format!("lin{k}.model.1.weight"),
            vec![1, c, 1, 1],
            Role::Lin,
        ));
    }
    out
}

/// Exact parameter count of the network for `latent_channels` (conv + BN affine + BN running
/// stats + lin heads) — for the pre-load memory estimate.
pub fn param_count(latent_channels: i32) -> u64 {
    checkpoint_tensors(latent_channels)
        .iter()
        .map(|(_, s, _)| s.iter().map(|&d| d as u64).product::<u64>())
        .sum()
}

/// The deterministic formula checkpoint (every key the network reads) — the weights the parity
/// fixture was produced with, and synthetic weights for any test that must not download.
pub fn formula_weights(latent_channels: i32) -> Weights {
    let mut w = Weights::empty();
    for (k, shape, role) in checkpoint_tensors(latent_channels) {
        let a = formula_array(&k, &shape, role);
        w.insert(k, a);
    }
    w
}

enum Op {
    /// 3×3 conv (MLX OHWI weight) + bias, padding 1.
    Conv(Array, Array),
    /// Eval-mode BatchNorm folded to `x · scale + shift`.
    Affine(Array, Array),
    Relu,
    MaxPool,
}

/// The loaded, frozen E-LatentLPIPS network.
pub struct LatentLpips {
    latent_channels: i32,
    slices: Vec<Vec<Op>>,
    /// Lin-head weights, one `[C]` vector per slice.
    lins: Vec<Array>,
}

fn f32w(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.as_dtype(Dtype::Float32)?)
}

/// 2×2 / stride-2 max-pool over NHWC (an odd trailing row/col is dropped, like torch).
fn max_pool2(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w, c) = (sh[0], sh[1] / 2, sh[2] / 2, sh[3]);
    use mlx_rs::ops::indexing::IndexOp;
    let cropped = x.index((.., ..2 * h, ..2 * w, ..));
    Ok(cropped
        .reshape(&[b, h, 2, w, 2, c])?
        .max_axes(&[2, 4], false)?)
}

impl LatentLpips {
    /// Build from a checkpoint map (`net.slice{s}.{i}.*`, `lin{k}.model.1.weight`, torch layouts).
    pub fn from_weights(w: &Weights, latent_channels: i32) -> Result<Self> {
        let mut slices: Vec<Vec<Op>> = (0..5).map(|_| Vec::new()).collect();
        for (i, layer) in trunk_layers(latent_channels).into_iter().enumerate() {
            let k = layer_key(i);
            let op = match layer {
                LayerKind::Conv(cin, cout) => {
                    let cw = f32w(w, &format!("{k}.weight"))?;
                    if cw.shape() != [cout, cin, 3, 3] {
                        return Err(Error::Msg(format!(
                            "E-LatentLPIPS: {k}.weight has shape {:?}, expected [{cout}, {cin}, 3, 3] \
                             (a checkpoint for another latent family?)",
                            cw.shape()
                        )));
                    }
                    Op::Conv(
                        cw.transpose_axes(&[0, 2, 3, 1])?,
                        f32w(w, &format!("{k}.bias"))?,
                    )
                }
                LayerKind::Bn(_) => {
                    let g = f32w(w, &format!("{k}.weight"))?;
                    let b = f32w(w, &format!("{k}.bias"))?;
                    let m = f32w(w, &format!("{k}.running_mean"))?;
                    let v = f32w(w, &format!("{k}.running_var"))?;
                    let scale = divide(&g, &sqrt(&add(&v, Array::from_f32(BN_EPS))?)?)?;
                    let shift = subtract(&b, &multiply(&m, &scale)?)?;
                    Op::Affine(scale, shift)
                }
                LayerKind::Relu => Op::Relu,
                LayerKind::MaxPool => Op::MaxPool,
                LayerKind::Identity => continue,
            };
            slices[slice_of(i) - 1].push(op);
        }
        let lins = LPIPS_CHANNELS
            .iter()
            .enumerate()
            .map(|(k, &c)| Ok(f32w(w, &format!("lin{k}.model.1.weight"))?.reshape(&[c])?))
            .collect::<Result<Vec<_>>>()?;
        let net = Self {
            latent_channels,
            slices,
            lins,
        };
        // Freeze: materialize the folded constants once.
        for s in &net.slices {
            for op in s {
                match op {
                    Op::Conv(a, b) | Op::Affine(a, b) => mlx_rs::transforms::eval([a, b])?,
                    _ => {}
                }
            }
        }
        Ok(net)
    }

    /// Load `family`'s published checkpoint from `dir` (a `Mingguksky/elatentlpips` snapshot):
    /// [`LatentLpipsFamily::checkpoint_file`], else the same file name at the top level, else a
    /// same-stem `.safetensors`.
    pub fn from_dir(dir: impl AsRef<Path>, family: LatentLpipsFamily) -> Result<Self> {
        let path = resolve_checkpoint(dir.as_ref(), family)?;
        let w = if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            Weights::from_file(&path)?
        } else {
            Weights::from_map(crate::pth::load_pth_f32(&path)?)
        };
        Self::from_weights(&w, family.latent_channels() as i32).map_err(|e| {
            Error::Msg(format!(
                "E-LatentLPIPS ({}) from {}: {e}",
                family.as_str(),
                path.display()
            ))
        })
    }

    /// Latent channels the first convolution consumes.
    pub fn latent_channels(&self) -> i32 {
        self.latent_channels
    }

    /// The five unit-normalized slice outputs of an NHWC latent.
    fn features(&self, x: &Array) -> Result<Vec<Array>> {
        let mut h = x.clone();
        let mut out = Vec::with_capacity(5);
        for slice in &self.slices {
            for op in slice {
                h = match op {
                    Op::Conv(w, b) => conv2d(&h, w, Some(b), 1, 1)?,
                    Op::Affine(s, t) => add(&multiply(&h, s)?, t)?,
                    Op::Relu => maximum(&h, Array::from_f32(0.0))?,
                    Op::MaxPool => max_pool2(&h)?,
                };
            }
            let norm = sqrt(&add(
                &multiply(&h, &h)?.sum_axes(&[3], true)?,
                Array::from_f32(NORMALIZE_EPS),
            )?)?;
            out.push(divide(&h, &norm)?);
        }
        Ok(out)
    }

    /// The per-layer terms `[5]` and the L1 term of `in0` vs `in1` (NCHW model-space latents), each
    /// averaged over the batch. Differentiable in both inputs.
    pub fn distance_terms(&self, in0: &Array, in1: &Array) -> Result<(Vec<Array>, Array)> {
        if in0.shape() != in1.shape() {
            return Err(Error::Msg(format!(
                "E-LatentLPIPS: input shapes differ ({:?} vs {:?})",
                in0.shape(),
                in1.shape()
            )));
        }
        let c = in0.shape().get(1).copied().unwrap_or(0);
        if in0.ndim() != 4 || c != self.latent_channels {
            return Err(Error::Msg(format!(
                "E-LatentLPIPS: expected an NCHW latent with {} channels, got {:?}",
                self.latent_channels,
                in0.shape()
            )));
        }
        let x0 = in0.as_dtype(Dtype::Float32)?;
        let x1 = in1.as_dtype(Dtype::Float32)?;
        let l1 = abs(&subtract(&x0, &x1)?)?.mean(None)?;
        let f0 = self.features(&x0.transpose_axes(&[0, 2, 3, 1])?)?;
        let f1 = self.features(&x1.transpose_axes(&[0, 2, 3, 1])?)?;
        let mut layers = Vec::with_capacity(5);
        for ((a, b), lin) in f0.iter().zip(&f1).zip(&self.lins) {
            let d = subtract(a, b)?;
            let weighted = multiply(&multiply(&d, &d)?, lin)?.sum_axes(&[3], false)?; // [B, h, w]
            layers.push(weighted.mean(None)?);
        }
        Ok((layers, l1))
    }

    /// E-LatentLPIPS(`in0`, `in1`) with the L1 term — the scalar upstream returns (batch-averaged).
    pub fn distance(&self, in0: &Array, in1: &Array) -> Result<Array> {
        let (layers, l1) = self.distance_terms(in0, in1)?;
        let mut total = l1;
        for l in layers {
            total = add(&total, &l)?;
        }
        Ok(total)
    }
}

/// The checkpoint file for `family` under `dir` (see [`LatentLpips::from_dir`]).
pub fn resolve_checkpoint(dir: &Path, family: LatentLpipsFamily) -> Result<PathBuf> {
    let rel = family.checkpoint_file();
    let name = Path::new(&rel)
        .file_name()
        .expect("checkpoint_file has a file name")
        .to_owned();
    let stem = Path::new(&name).with_extension("safetensors");
    let candidates = [dir.join(&rel), dir.join(&name), dir.join(stem)];
    candidates
        .iter()
        .find(|p| p.is_file())
        .cloned()
        .ok_or_else(|| {
            Error::Msg(format!(
                "E-LatentLPIPS: no {} checkpoint under {} (looked for {})",
                family.as_str(),
                dir.display(),
                rel
            ))
        })
}

/// The E-LatentLPIPS auxiliary loss: frozen network + the clean latent as the per-image reference.
/// The reference stays resident: it is the latent itself, at most `16 · (H/8) · (W/8)` f32 —
/// ~1 MB per (item, bucket) entry at 1024², the size of the trainer's own cached latent — so,
/// unlike the VAE anchor's ~0.5 GB of taps, it needs no on-disk spill.
pub struct LatentLpipsLoss {
    net: LatentLpips,
}

/// The per-image E-LatentLPIPS reference: the image's clean model-space latent (NCHW).
pub struct LatentLpipsReference {
    pub latent: Array,
}

impl LatentLpipsLoss {
    /// Wrap a loaded network.
    pub fn new(net: LatentLpips) -> Self {
        Self { net }
    }

    /// Load `family`'s checkpoint from `dir`.
    pub fn from_dir(dir: impl AsRef<Path>, family: LatentLpipsFamily) -> Result<Self> {
        Ok(Self::new(LatentLpips::from_dir(dir, family)?))
    }

    /// The wrapped network.
    pub fn net(&self) -> &LatentLpips {
        &self.net
    }
}

impl PerceptualLoss for LatentLpipsLoss {
    fn name(&self) -> &'static str {
        "latent_lpips"
    }

    fn input(&self) -> PerceptualInput {
        PerceptualInput::Latents
    }

    /// The clean latent itself (evaluated, gradient-stopped). Every image is usable.
    fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
        let latent = mlx_rs::stop_gradient(clean.as_dtype(Dtype::Float32)?)?;
        latent.eval()?;
        Ok(Some(Box::new(LatentLpipsReference { latent })))
    }

    /// E-LatentLPIPS of the live x0 latent against the clean latent (upstream order:
    /// `model(x0_pred, target)`).
    fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
        let r = reference_as::<LatentLpipsReference>(self.name(), reference)?;
        self.net.distance(live, &r.latent)
    }
}

/// The pre-load memory figures of E-LatentLPIPS for `family` on a `latent_h × latent_w` latent
/// (epic 2123 E7): resident f32 weights; a working set holding every trunk activation (conv, BN and
/// ReLU outputs per conv layer) for both inputs, ×2 for the backward's cotangents; and the cached
/// clean latent per image.
pub fn latent_lpips_footprint(
    family: LatentLpipsFamily,
    latent_h: u32,
    latent_w: u32,
) -> AuxModelFootprint {
    let c = family.latent_channels() as i32;
    let mut pixels = latent_h as u64 * latent_w as u64;
    let mut floats = 0u64;
    for layer in trunk_layers(c) {
        match layer {
            LayerKind::Conv(_, cout) => floats += 3 * cout as u64 * pixels,
            LayerKind::MaxPool => {
                pixels /= 4;
            }
            _ => {}
        }
    }
    AuxModelFootprint {
        param_bytes: param_count(c) * 4,
        working_set_bytes: floats * 4 * 2 * 2,
        reference_bytes_per_image: c as u64 * latent_h as u64 * latent_w as u64 * 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::formula::{input_wave, PARITY_FIXTURE_REL};
    use crate::train::perceptual::{AuxLoss, AuxLossSchedule, PerceptualPath};
    use mlx_rs::transforms::{eval, grad};

    fn fixture() -> serde_json::Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(PARITY_FIXTURE_REL);
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
    }

    fn scalar(a: &Array) -> f32 {
        eval([a]).unwrap();
        a.item::<f32>()
    }

    fn wave_latent(
        c: i32,
        h: i32,
        w: i32,
        spec: &serde_json::Value,
        base: Option<&[f64]>,
    ) -> (Array, Vec<f64>) {
        let n = (c * h * w) as usize;
        let p: Vec<f64> = spec
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let mut v = input_wave(n, p[0], p[1], p[2]);
        if let Some(b) = base {
            for (x, y) in v.iter_mut().zip(b) {
                *x += y;
            }
        }
        let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        (Array::from_slice(&f, &[1, c, h, w]), v)
    }

    fn net(c: i32) -> LatentLpips {
        LatentLpips::from_weights(&formula_weights(c), c).unwrap()
    }

    /// The fixture is the one the committed producer script made (its bytes are pinned in the
    /// fixture). Mutation: edit the producer without regenerating ⇒ red.
    #[test]
    fn the_parity_fixture_matches_its_producer() {
        use sha2::{Digest, Sha256};
        let fx = fixture();
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(PARITY_FIXTURE_REL)
            .with_file_name(fx["producer"].as_str().unwrap());
        let digest = Sha256::digest(std::fs::read(script).unwrap());
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, fx["producer_sha256"].as_str().unwrap());
    }

    /// Parity with the upstream PyTorch `ELatentLPIPS.forward` (normalize=False, add_l1_loss=True,
    /// ensembling=False) on the formula weights, for a 4-channel and a 16-channel family: the
    /// total, each of the five layer terms, the L1 term and the self-distance. Mutations: keep the
    /// first max-pool (`IDENTITY_POOLS = 2`), drop the channel normalization, or drop the L1 term
    /// ⇒ red.
    #[test]
    fn matches_the_upstream_reference_on_formula_weights() {
        let fx = fixture();
        for case in fx["latent_lpips"].as_array().unwrap() {
            let c = case["latent_channels"].as_i64().unwrap() as i32;
            let (h, w) = (
                case["h"].as_i64().unwrap() as i32,
                case["w"].as_i64().unwrap() as i32,
            );
            let (in0, base) = wave_latent(c, h, w, &case["inputs"]["in0"], None);
            let (in1, _) = wave_latent(c, h, w, &case["inputs"]["in1_delta"], Some(&base));
            let n = net(c);
            let (layers, l1) = n.distance_terms(&in0, &in1).unwrap();
            let close = |got: f32, want: f64, what: &str| {
                let tol = 2e-4 * want.abs().max(1e-3);
                assert!(
                    (got as f64 - want).abs() <= tol,
                    "{} {what}: {got} vs upstream {want}",
                    case["family"]
                );
            };
            for (k, l) in layers.iter().enumerate() {
                close(
                    scalar(l),
                    case["per_layer"][k].as_f64().unwrap(),
                    &format!("layer {k}"),
                );
            }
            close(scalar(&l1), case["l1"].as_f64().unwrap(), "l1");
            close(
                scalar(&n.distance(&in0, &in1).unwrap()),
                case["value"].as_f64().unwrap(),
                "total",
            );
            assert_eq!(scalar(&n.distance(&in0, &in0).unwrap()), 0.0);
        }
    }

    /// AC2: zero for identical latents, positive for a perturbed one (its learned layer terms too),
    /// growing with the perturbation; differentiable in the live latent. Mutation: subtract the
    /// layer terms in `distance` ⇒ the distance falls below its L1 term ⇒ red.
    #[test]
    fn is_zero_for_identical_latents_and_positive_when_perturbed() {
        let n = net(4);
        let x = mlx_rs::random::normal::<f32>(
            &[1, 4, 8, 8],
            None,
            None,
            Some(&mlx_rs::random::key(3).unwrap()),
        )
        .unwrap();
        let d = mlx_rs::random::normal::<f32>(
            &[1, 4, 8, 8],
            None,
            None,
            Some(&mlx_rs::random::key(4).unwrap()),
        )
        .unwrap();
        assert_eq!(scalar(&n.distance(&x, &x).unwrap()), 0.0);
        let small = add(&x, multiply(&d, Array::from_f32(0.05)).unwrap()).unwrap();
        let big = add(&x, multiply(&d, Array::from_f32(0.5)).unwrap()).unwrap();
        let (s, b) = (
            scalar(&n.distance(&small, &x).unwrap()),
            scalar(&n.distance(&big, &x).unwrap()),
        );
        assert!(s > 0.0 && b > s, "small {s} big {b}");
        // The learned part is itself positive: every layer term > 0, and the distance exceeds its
        // L1 term alone.
        let (layers, l1) = n.distance_terms(&big, &x).unwrap();
        for (k, l) in layers.iter().enumerate() {
            assert!(scalar(l) > 0.0, "layer {k}");
        }
        assert!(b > scalar(&l1), "distance {b} must exceed its L1 term");
        let g = grad(|p: &Array| -> mlx_rs::error::Result<Array> {
            n.distance(p, &x)
                .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))
        })(&small)
        .unwrap();
        assert!(scalar(&g.abs().unwrap().sum(None).unwrap()) > 0.0);
    }

    /// A checkpoint for another latent family (wrong first-conv width) is refused, never silently
    /// run. Mutation: drop the shape check ⇒ the conv fails later or runs ⇒ the message assertion reds.
    #[test]
    fn a_checkpoint_for_another_family_is_refused() {
        let err = LatentLpips::from_weights(&formula_weights(16), 4)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("another latent family"), "{err}");
        let tmp = tempfile::tempdir().unwrap();
        let err = LatentLpips::from_dir(tmp.path(), LatentLpipsFamily::Sdxl)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("no sdxl checkpoint"), "{err}");
    }

    /// A same-stem `.safetensors` rehost loads through `from_dir` and computes what the in-memory
    /// formula network computes (the published `.pth` path is covered by `pth`'s state-dict test).
    #[test]
    fn loads_a_safetensors_rehost_from_a_snapshot_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sdxl_latest_vgg16_tuned.safetensors");
        let w = formula_weights(4).into_tensors();
        let mut named: Vec<(&String, &Array)> = w.iter().collect();
        named.sort_unstable_by_key(|(k, _)| *k);
        Array::save_safetensors(
            named.into_iter().map(|(k, v)| (k.as_str(), v)),
            None::<&std::collections::HashMap<String, String>>,
            &path,
        )
        .unwrap();
        assert_eq!(
            resolve_checkpoint(tmp.path(), LatentLpipsFamily::Sdxl).unwrap(),
            path
        );
        let loss = LatentLpipsLoss::from_dir(tmp.path(), LatentLpipsFamily::Sdxl).unwrap();
        let x = mlx_rs::random::normal::<f32>(
            &[1, 4, 8, 8],
            None,
            None,
            Some(&mlx_rs::random::key(1).unwrap()),
        )
        .unwrap();
        let y = add(&x, Array::from_f32(0.1)).unwrap();
        let (a, b) = (
            scalar(&loss.net().distance(&x, &y).unwrap()),
            scalar(&net(4).distance(&x, &y).unwrap()),
        );
        assert!((a - b).abs() <= 1e-6 * b.abs().max(1.0), "{a} vs {b}");
    }

    /// The parameter count is the published checkpoints' size: both files exceed `param_count · 4`
    /// by the same 35 446 bytes (pickle, zip headers, int64 `num_batches_tracked` counters), so the
    /// window catches a missing lin head (5.9 KB) or BN running stat (17 KB). Mutation: drop the lin
    /// heads or the running variances from `checkpoint_tensors` ⇒ red.
    #[test]
    fn param_count_matches_the_published_checkpoint_sizes() {
        for (c, bytes) in [(4, 58_969_974u64), (16, 58_997_622)] {
            let ours = param_count(c) * 4;
            assert!(
                ours <= bytes && (30_000..40_000).contains(&(bytes - ours)),
                "{c}: {ours} vs {bytes}"
            );
        }
    }

    /// The shared path drives it as a latent loss: no decoder needed, reference = clean latent,
    /// zero at the clean latent and positive off it. Mutation: report `DecodedPixels` from `input`
    /// ⇒ `PerceptualPath::new(None, …)` refuses ⇒ red.
    #[test]
    fn runs_on_the_shared_path_without_a_decoder() {
        let mut path = PerceptualPath::new(
            None,
            vec![AuxLoss {
                schedule: AuxLossSchedule {
                    weight: 0.5,
                    t_min: 0.0,
                    t_max: 0.5,
                    every_n: 1,
                },
                loss: Box::new(LatentLpipsLoss::new(net(4))),
            }],
        )
        .unwrap();
        let clean = mlx_rs::random::normal::<f32>(
            &[1, 4, 8, 8],
            None,
            None,
            Some(&mlx_rs::random::key(9).unwrap()),
        )
        .unwrap();
        path.ensure_reference(0, &clean).unwrap();
        let plan = path.plan(1, 0, 0.3).unwrap();
        assert!(plan.diffusion && plan.aux == vec![0], "{plan:?}");
        let at_clean = path.aux_loss(&plan, 0, &clean).unwrap().unwrap();
        assert_eq!(scalar(&at_clean.weighted), 0.0);
        let off = add(&clean, Array::from_f32(0.2)).unwrap();
        assert!(scalar(&path.aux_loss(&plan, 0, &off).unwrap().unwrap().weighted) > 0.0);
        // Outside the [0, 0.5] window the loss does not fire.
        assert!(path.plan(1, 0, 0.8).unwrap().aux.is_empty());
    }

    /// E7: the footprint carries the weights, a positive working set that grows with the latent,
    /// and one clean latent per image. Mutation: drop the per-image reference term ⇒ red.
    #[test]
    fn footprint_scales_with_the_latent() {
        let a = latent_lpips_footprint(LatentLpipsFamily::Flux, 64, 64);
        let b = latent_lpips_footprint(LatentLpipsFamily::Flux, 128, 128);
        assert_eq!(a.param_bytes, param_count(16) * 4);
        assert_eq!(a.reference_bytes_per_image, 16 * 64 * 64 * 4);
        assert!(b.working_set_bytes == 4 * a.working_set_bytes && a.working_set_bytes > 0);
    }
}
