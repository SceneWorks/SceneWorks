//! Family-agnostic LoRA/LoKr **training** machinery (epic 3039) — the adapter-factor lifecycle the
//! spike (sc-3042) proved, hoisted out of the first family trainer (Z-Image, sc-3044) so every
//! family crate (Z-Image, SDXL sc-3045, Wan sc-3046, LTX sc-3047) shares one implementation.
//!
//! The model crates do NOT use mlx-rs's `Module`/`ModuleParameters` system (hand-rolled `&self`
//! forwards over raw `Array`s, [`crate::adapters`]), so training uses **functional autograd**: the
//! trainable factors live OUTSIDE the model in a [`LoraParams`] map, and each step are re-injected
//! into the target [`AdaptableLinear`](crate::adapters::AdaptableLinear)s as a single
//! [`Adapter`] via [`AdaptableLinear::set_adapters`](crate::adapters::AdaptableLinear::set_adapters).
//! The injection mirrors the inference reload op-for-op — for LoRA: transpose the `[r,in]`/`[out,r]`
//! factors, fold `alpha/rank` into `b`, `scale = 1`; for LoKr: reconstruct the delta with the SAME
//! [`reconstruct_lokr_delta`] the loader uses — so the trained adapter round-trips through the
//! inference path.
//!
//! Everything here is generic over the adapter host ([`AdaptableHost`]); the two genuine per-family
//! differences are passed in by the caller:
//!   * **LoKr reconstruct dtype** — `Bfloat16` for the bf16-residual families (Z-Image/Qwen),
//!     `Float32` for the f32-everywhere SDXL merge path. Training must reconstruct at the dtype the
//!     inference loader uses, so the adapter round-trips.
//!   * **PEFT save-key prefix** — `""` for the DiT families (`{path}.lora_A.weight`),
//!     `"base_model.model.unet."` for SDXL (what `peft.save_pretrained()` / the SceneWorks
//!     `_SdxlLoraBackend` emit, and what the SDXL loader's PEFT classifier expects).
//!
//! The model forward, the noise/target construction, and the text/VAE encoding stay in the family
//! crate (they are model-specific); this module owns only the adapter factors.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::multiply;
use mlx_rs::{random, Array, Dtype};

use crate::adapters::{reconstruct_lokr_delta, AdaptableHost, Adapter};
use crate::Result;

/// The trainable LoRA/LoKr factor map — keyed by `{path}.<factor>` (e.g. `…to_q.lora_a`,
/// `…to_q.lokr_w1`). The autograd arguments (`keyed_value_and_grad`) and the optimizer-stepped
/// state.
pub type LoraParams = HashMap<Rc<str>, Array>;

/// One LoRA-trained Linear: its dotted module path (e.g. `down_blocks.1.…attn1.to_q`) plus the
/// pre-built parameter-map keys and the `[out, in]` dims read off the base weight.
pub struct LoraTarget {
    pub path: String,
    a_key: Rc<str>,
    b_key: Rc<str>,
    pub in_f: i32,
    pub out_f: i32,
}

/// One LoKr-trained Linear: its path, the base `[out,in]` shape, and the factor-map keys —
/// `lokr_w1` always, then either full `lokr_w2` or low-rank `lokr_w2_a`/`lokr_w2_b`.
pub struct LokrTarget {
    path: String,
    base_shape: Vec<i32>,
    w1_key: Rc<str>,
    w2_key: Option<Rc<str>>,
    w2a_key: Option<Rc<str>>,
    w2b_key: Option<Rc<str>>,
}

/// LyCORIS dimension factorization: split `dimension` into `(a, b)`, `a*b = dimension`, `a <= b`,
/// with `a` as close to `factor` (or balanced/√dimension when `factor < 0`) as a divisor allows.
/// The LoKr weight `[out,in]` then factors as `kron(w1, w2)` with `w1 = [fac(out).0, fac(in).0]`,
/// `w2 = [fac(out).1, fac(in).1]`. Port of LyCORIS `factorization`.
pub fn factorization(dimension: i32, factor: i32) -> (i32, i32) {
    if factor > 0 && dimension % factor == 0 {
        let n = dimension / factor;
        return if factor > n { (n, factor) } else { (factor, n) };
    }
    let factor = if factor < 0 { dimension } else { factor };
    let (mut m, mut n) = (1i32, dimension);
    let mut length = m + n;
    while m < n {
        let mut new_m = m + 1;
        while dimension % new_m != 0 {
            new_m += 1;
        }
        let new_n = dimension / new_m;
        if new_m + new_n > length || new_m > factor {
            break;
        }
        m = new_m;
        n = new_n;
        length = m + n;
    }
    if m > n {
        (n, m)
    } else {
        (m, n)
    }
}

/// Resolve each `target_paths` entry on `host`, read its `[out,in]` dims, and initialise the
/// trainable LoRA factors the Python `_MlxLoRALinear` way — `A ~ N(0, 0.02)` `[rank,in]`,
/// `B = 0` `[out,rank]` — keyed `{path}.lora_a` / `{path}.lora_b`. The `B = 0` init makes the
/// adapter start as an exact no-op (it only learns from the gradient).
pub fn build_lora_targets<H: AdaptableHost>(
    host: &mut H,
    target_paths: &[String],
    rank: i32,
    seed: u64,
) -> Result<(Vec<LoraTarget>, LoraParams)> {
    // F-065: training-side twin of the F-002/F-010 load guard. rank <= 0 yields a 0·inf = NaN
    // no-op delta (and a degenerate `[0,in]`/`[out,0]` factor shape); reject it up front rather
    // than training a silently-corrupt adapter.
    if rank <= 0 {
        return Err(crate::Error::Msg(format!(
            "LoRA training: rank must be > 0, got {rank}"
        )));
    }
    let mut targets = Vec::with_capacity(target_paths.len());
    let mut params: LoraParams = HashMap::new();
    for (i, path) in target_paths.iter().enumerate() {
        let segs: Vec<&str> = path.split('.').collect();
        // SC-18319 — target *sizing* only reads the base shape, so it goes through the PROBE half.
        // `bind_lora_params` below is the mutation that installs the trainable stack (and unfuses).
        let facts = host.adaptable_facts(&segs).ok_or_else(|| -> crate::Error {
            format!("LoRA target does not resolve on the model: {path}").into()
        })?;
        let shape = facts.base_shape; // [out, in]
        let (out_f, in_f) = (shape[0], shape[1]);

        let a_key: Rc<str> = Rc::from(format!("{path}.lora_a"));
        let b_key: Rc<str> = Rc::from(format!("{path}.lora_b"));
        // Distinct subkey per target so the RNG init differs per layer.
        let ka = random::key(seed.wrapping_add(2 * i as u64 + 1))?;
        let a = multiply(
            &random::normal::<f32>(&[rank, in_f], None, None, Some(&ka))?,
            Array::from_slice(&[0.02f32], &[1]),
        )?;
        let b = Array::zeros::<f32>(&[out_f, rank])?;
        mlx_rs::transforms::eval([&a, &b])?;
        params.insert(a_key.clone(), a);
        params.insert(b_key.clone(), b);
        targets.push(LoraTarget {
            path: path.clone(),
            a_key,
            b_key,
            in_f,
            out_f,
        });
    }
    Ok((targets, params))
}

/// Inject the current trainable factors as one `Adapter::Lora` per target — EXACTLY as the inference
/// reload (`install_lora_groups`): transpose `[r,in]`→`[in,r]` and `[out,r]`→`[r,out]`, fold
/// `alpha/rank` into `b`, `scale = 1`. Differentiable (the transposes/fold are traced).
pub fn install_training_lora<H: AdaptableHost>(
    host: &mut H,
    params: &LoraParams,
    targets: &[LoraTarget],
    alpha: f32,
) -> MlxResult<()> {
    install_training_lora_as(host, params, targets, alpha, None)
}

/// [`install_training_lora`] with an optional compute-dtype cast: when `dtype` is `Some`, the folded
/// factors are cast to it inside the traced graph (sc-4887 bf16 mixed-precision training — the
/// residual must match the bf16 activation stream or every adapted Linear silently re-promotes the
/// whole chain to f32). The trainable leaves stay f32; the gradient flows back through the `astype`
/// VJP, so the optimizer still sees f32 grads (master-weights pattern). `None` = factor dtype as-is.
pub fn install_training_lora_as<H: AdaptableHost>(
    host: &mut H,
    params: &LoraParams,
    targets: &[LoraTarget],
    alpha: f32,
    dtype: Option<Dtype>,
) -> MlxResult<()> {
    for t in targets {
        // `.get().ok_or_else()?` rather than a direct index: a bookkeeping bug that drops a key from
        // the optimizer-stepped params map must surface as a typed error, not a panic (F-008).
        let a = params
            .get(&t.a_key)
            .ok_or_else(|| Exception::custom(format!("LoRA param missing: {}", t.a_key)))?
            .t(); // [r,in] -> [in,r]
        let b_t = params
            .get(&t.b_key)
            .ok_or_else(|| Exception::custom(format!("LoRA param missing: {}", t.b_key)))?
            .t(); // [out,r] -> [r,out]
        let rank = a.shape()[1] as f32;
        let b = b_t.multiply(Array::from_slice(&[alpha / rank], &[1]))?;
        let (a, b) = match dtype {
            Some(dt) => (a.as_dtype(dt)?, b.as_dtype(dt)?),
            None => (a, b),
        };
        let segs: Vec<&str> = t.path.split('.').collect();
        let lin = host
            .adaptable_mut(&segs)
            .ok_or_else(|| Exception::custom(format!("LoRA target not found: {}", t.path)))?;
        lin.set_training_adapters(vec![Adapter::Lora { a, b, scale: 1.0 }]);
    }
    Ok(())
}

/// Clear every listed path's adapter stack (back to the bare frozen base).
pub fn clear_adapters<H: AdaptableHost>(host: &mut H, paths: &[String]) {
    for path in paths {
        let segs: Vec<&str> = path.split('.').collect();
        if let Some(lin) = host.adaptable_mut(&segs) {
            lin.set_adapters(Vec::new());
        }
    }
}

/// Write the trainable LoRA factors as PEFT-format safetensors — `{prefix}{path}.lora_A.weight`
/// `[r,in]`, `{prefix}{path}.lora_B.weight` `[out,r]`, scalar `{prefix}{path}.alpha`. `key_prefix`
/// is `""` for the DiT families (bare dotted paths) and `"base_model.model.unet."` for SDXL (what
/// `peft.save_pretrained()` emits and the SDXL loader's PEFT classifier expects). Metadata records
/// the network type/rank/alpha (the epic-2193 reload contract).
pub fn save_lora_peft(
    params: &LoraParams,
    targets: &[LoraTarget],
    alpha: f32,
    rank: u32,
    key_prefix: &str,
    path: impl AsRef<Path>,
) -> Result<()> {
    save_lora_peft_with_meta(params, targets, alpha, rank, key_prefix, &[], path)
}

/// [`save_lora_peft`] plus caller-supplied `__metadata__` entries (sc-14057).
///
/// `extra` is written **before** the `networkType`/`rank`/`alpha` contract keys, so a caller can
/// never accidentally overwrite them. Its purpose is family *provenance*: SceneWorks' importer
/// identifies an adapter's architecture from the safetensors header, and for a family whose tensor
/// names are shared with a sibling (Mage-Flow's NR-MMDiT block leaves are spelled identically to
/// Qwen-Image's) the keys alone cannot say which one wrote the file. A `family` / `baseModel` stamp
/// — the same pair the candle Krea trainer writes — makes our own adapters self-identifying on
/// re-import instead of landing family-less.
pub fn save_lora_peft_with_meta(
    params: &LoraParams,
    targets: &[LoraTarget],
    alpha: f32,
    rank: u32,
    key_prefix: &str,
    extra: &[(&str, &str)],
    path: impl AsRef<Path>,
) -> Result<()> {
    let alphas: Vec<(String, Array)> = targets
        .iter()
        .map(|t| {
            (
                format!("{key_prefix}{}.alpha", t.path),
                Array::from_slice(&[alpha], &[1]),
            )
        })
        .collect();
    let mut entries: Vec<(String, &Array)> = Vec::with_capacity(targets.len() * 3);
    for t in targets {
        // `.get().ok_or_else()?` over a direct index so a missing param surfaces as a typed error
        // rather than a panic mid-checkpoint (F-008).
        let a = params
            .get(&t.a_key)
            .ok_or_else(|| crate::Error::Msg(format!("LoRA param missing: {}", t.a_key)))?;
        let b = params
            .get(&t.b_key)
            .ok_or_else(|| crate::Error::Msg(format!("LoRA param missing: {}", t.b_key)))?;
        entries.push((format!("{key_prefix}{}.lora_A.weight", t.path), a));
        entries.push((format!("{key_prefix}{}.lora_B.weight", t.path), b));
    }
    for (k, v) in &alphas {
        entries.push((k.clone(), v));
    }
    let mut meta: HashMap<String, String> = extra
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    meta.insert("networkType".to_string(), "lora".to_string());
    meta.insert("rank".to_string(), rank.to_string());
    meta.insert("alpha".to_string(), alpha.to_string());
    Array::save_safetensors(entries, Some(&meta), path)?;
    Ok(())
}

/// Initialise trainable LoKr factors per target, matching **PEFT `LoKrConfig(init_weights=True)`'s
/// `reset_adapter_parameters`** (sc-5179) — the init the Python LoRA/LoKr trainers
/// (`lens_train_runner.py` and the SDXL/Wan/etc. backends) use, so a *Python LoKr learning rate
/// transfers* to this native path.
///
/// The weight `[out,in]` factors as `kron(w1[out_a,in_a], w2[out_b,in_b])` (`decompose_both` is off,
/// PEFT's default — `w1` is never further low-ranked). `w2` is low-ranked to `rank` when
/// `rank < max(out_b,in_b)/2` (PEFT's `use_w2` predicate). The **first** factor `w1` is
/// **zero-initialised**; the **second** factor `w2` (full, or both `w2_a`/`w2_b` low-rank) is
/// **kaiming-uniform `a=√5`** ⇒ `U(±1/√fan_in)`, `fan_in` = the factor's input dim. Because the zeroed
/// factor is `w1`, the initial delta `kron(0, w2)·(alpha/rank) = 0` (the LoKr analog of LoRA's
/// `B = 0`), while the *non*-zero factor's **fan-in-scaled** magnitude is what makes the LoKr
/// lr-sensitivity track PEFT. `factor` is the decompose knob (`-1` = balanced/auto).
///
/// NOTE (sc-5179): this *replaced* the prior `w1 ~ N(0,0.02)` / zeroed-`w2` init — the **opposite**
/// zeroed factor and a **fixed** 0.02 scale (~4–5× below PEFT's fan-in-scaled kaiming), which is why
/// LoKr formerly needed a ~10× higher, non-transferable lr than LoRA. The save/round-trip format
/// (`lokr_w1` + `lokr_w2`|`lokr_w2_a`/`lokr_w2_b`) and `reconstruct_lokr_delta` are unchanged, so prior
/// adapters still load; only newly-*trained* LoKr dynamics change. (The `factor > 0` factorization
/// sort still differs from PEFT — a separate pre-existing divergence; the default `factor = -1` matches.)
pub fn build_lokr_targets<H: AdaptableHost>(
    host: &mut H,
    target_paths: &[String],
    rank: i32,
    factor: i32,
    seed: u64,
) -> Result<(Vec<LokrTarget>, LoraParams)> {
    // F-065: training-side twin of the F-010 load guard. rank <= 0 (or a degenerate factorization)
    // yields a NaN-scale reconstructed delta; reject it rather than training a corrupt adapter.
    if rank <= 0 {
        return Err(crate::Error::Msg(format!(
            "LoKr training: rank must be > 0, got {rank}"
        )));
    }
    let mut targets = Vec::with_capacity(target_paths.len());
    let mut params = LoraParams::new();
    // `torch.nn.init.kaiming_uniform_(a=√5)` on a 2-D `[d0, d1]` factor ⇒ `U(±1/√fan_in)` with
    // `fan_in = d1` (gain `√(2/(1+5)) = 1/√3`, bound `√3·gain/√fan_in = 1/√fan_in`).
    let kaiming = |shape: [i32; 2], key: &Array| -> Result<Array> {
        let bound = 1.0f32 / (shape[1] as f32).sqrt();
        Ok(random::uniform::<_, f32>(
            -bound,
            bound,
            &shape[..],
            Some(key),
        )?)
    };
    for (i, path) in target_paths.iter().enumerate() {
        let segs: Vec<&str> = path.split('.').collect();
        // SC-18319 — factor sizing is shape-only, so the PROBE half; `bind_lokr_params` mutates.
        let facts = host.adaptable_facts(&segs).ok_or_else(|| -> crate::Error {
            format!("LoKr target does not resolve on the model: {path}").into()
        })?;
        let shape = facts.base_shape; // [out, in]
        let (out_a, out_b) = factorization(shape[0], factor);
        let (in_a, in_b) = factorization(shape[1], factor);

        // w1 = zeros [out_a, in_a] — the zeroed factor, so the initial delta is exactly 0.
        let w1 = Array::zeros::<f32>(&[out_a, in_a])?;
        let w1_key: Rc<str> = Rc::from(format!("{path}.lokr_w1"));
        mlx_rs::transforms::eval([&w1])?;
        params.insert(w1_key.clone(), w1);

        let (mut w2_key, mut w2a_key, mut w2b_key) = (None, None, None);
        // PEFT `use_w2 = not(r < max(out_b,in_b)/2)`: a low-rank w2 = w2_a @ w2_b only when r is below
        // half the larger factor dim, else a full w2. Both factors are kaiming-init (the delta is held
        // at 0 by the zeroed w1, not by zeroing the second factor).
        if rank > 0 && (rank as f32) < (out_b.max(in_b) as f32) / 2.0 {
            let ka = random::key(seed.wrapping_add(7 * i as u64 + 3))?;
            let kb = random::key(seed.wrapping_add(7 * i as u64 + 5))?;
            let w2a = kaiming([out_b, rank], &ka)?; // fan_in = rank
            let w2b = kaiming([rank, in_b], &kb)?; // fan_in = in_b
            let ak: Rc<str> = Rc::from(format!("{path}.lokr_w2_a"));
            let bk: Rc<str> = Rc::from(format!("{path}.lokr_w2_b"));
            mlx_rs::transforms::eval([&w2a, &w2b])?;
            params.insert(ak.clone(), w2a);
            params.insert(bk.clone(), w2b);
            w2a_key = Some(ak);
            w2b_key = Some(bk);
        } else {
            let k2 = random::key(seed.wrapping_add(7 * i as u64 + 3))?;
            let w2 = kaiming([out_b, in_b], &k2)?; // fan_in = in_b
            let wk: Rc<str> = Rc::from(format!("{path}.lokr_w2"));
            mlx_rs::transforms::eval([&w2])?;
            params.insert(wk.clone(), w2);
            w2_key = Some(wk);
        }
        targets.push(LokrTarget {
            path: path.clone(),
            base_shape: shape,
            w1_key,
            w2_key,
            w2a_key,
            w2b_key,
        });
    }
    Ok((targets, params))
}

/// Inject each target's LoKr delta — reconstructed from the trainable factors EXACTLY as the
/// inference loader (`reconstruct_lokr_delta` at `lokr_dtype`; `Adapter::Lokr` residual `x·ΔWᵀ`) —
/// so the trained adapter round-trips. `lokr_dtype` is the dtype the inference loader reconstructs
/// at (`Bfloat16` for Z-Image/Qwen, `Float32` for SDXL). Differentiable (kron/matmul/cast traced).
pub fn install_training_lokr<H: AdaptableHost>(
    host: &mut H,
    params: &LoraParams,
    targets: &[LokrTarget],
    alpha: f32,
    rank: f32,
    lokr_dtype: Dtype,
) -> MlxResult<()> {
    for t in targets {
        let w1 = params.get(t.w1_key.as_ref());
        let w2 = t.w2_key.as_ref().and_then(|k| params.get(k.as_ref()));
        let w2a = t.w2a_key.as_ref().and_then(|k| params.get(k.as_ref()));
        let w2b = t.w2b_key.as_ref().and_then(|k| params.get(k.as_ref()));
        let delta = reconstruct_lokr_delta(
            alpha,
            rank,
            &t.base_shape,
            w1,
            None,
            None,
            w2,
            w2a,
            w2b,
            lokr_dtype,
        )
        .map_err(|e| Exception::custom(e.to_string()))?;
        let segs: Vec<&str> = t.path.split('.').collect();
        let lin = host
            .adaptable_mut(&segs)
            .ok_or_else(|| Exception::custom(format!("LoKr target not found: {}", t.path)))?;
        lin.set_training_adapters(vec![Adapter::Lokr { delta, scale: 1.0 }]);
    }
    Ok(())
}

/// Write the trainable LoKr factors as safetensors — `{path}.lokr_w1` + (`lokr_w2` | `lokr_w2_a` +
/// `lokr_w2_b`) — with `networkType=lokr` + `rank`/`alpha`/`decomposeFactor` metadata (the epic-2193
/// reload contract). The loaders reconstruct the delta from these shapes (the SDXL LoKr classifier
/// also accepts a `base_model.model.unet.` prefix, but the bare keys resolve directly for every
/// family, so no prefix is written).
pub fn save_lokr(
    params: &LoraParams,
    targets: &[LokrTarget],
    alpha: f32,
    rank: f32,
    decompose_factor: i32,
    path: impl AsRef<Path>,
) -> Result<()> {
    save_lokr_with_meta(params, targets, alpha, rank, decompose_factor, &[], path)
}

/// [`save_lokr`] plus caller-supplied `__metadata__` entries — the LoKr twin of
/// [`save_lora_peft_with_meta`] (sc-14057), so a family stamp rides both adapter kinds.
pub fn save_lokr_with_meta(
    params: &LoraParams,
    targets: &[LokrTarget],
    alpha: f32,
    rank: f32,
    decompose_factor: i32,
    extra: &[(&str, &str)],
    path: impl AsRef<Path>,
) -> Result<()> {
    let mut entries: Vec<(String, &Array)> = Vec::with_capacity(targets.len() * 3);
    for t in targets {
        let keys = [
            Some(&t.w1_key),
            t.w2_key.as_ref(),
            t.w2a_key.as_ref(),
            t.w2b_key.as_ref(),
        ];
        for key in keys.into_iter().flatten() {
            // `.get().ok_or_else()?` over a direct index so a missing factor surfaces as a typed
            // error rather than a panic mid-checkpoint (F-008).
            let v = params
                .get(key.as_ref())
                .ok_or_else(|| crate::Error::Msg(format!("LoKr factor missing: {key}")))?;
            entries.push((key.to_string(), v));
        }
    }
    let mut meta: HashMap<String, String> = extra
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    meta.insert("networkType".to_string(), "lokr".to_string());
    meta.insert("rank".to_string(), (rank as i64).to_string());
    // Lossless (sc-24158): `f32` Display renders an integral alpha as before (`4`) and keeps a
    // fractional one (`0.5`) — `as i64` truncated it, silently rescaling the reloaded LoKr.
    meta.insert("alpha".to_string(), alpha.to_string());
    meta.insert("decomposeFactor".to_string(), decompose_factor.to_string());
    Array::save_safetensors(entries, Some(&meta), path)?;
    Ok(())
}

/// Sum `grads` into the running accumulator (gradient accumulation across micro-steps).
pub fn accumulate_grads(acc: &mut Option<LoraParams>, grads: LoraParams) -> Result<()> {
    use mlx_rs::ops::add;
    match acc {
        None => *acc = Some(grads),
        Some(a) => {
            for (k, g) in grads {
                let entry = a
                    .get(&k)
                    .ok_or_else(|| crate::Error::Msg(format!("grad key {k} vanished")))?;
                let summed = add(entry, &g)?;
                a.insert(k, summed);
            }
        }
    }
    Ok(())
}

/// Divide accumulated gradients by `accum` (the mean over the accumulation window).
pub fn average_grads(grads: LoraParams, accum: u32) -> Result<LoraParams> {
    if accum <= 1 {
        return Ok(grads);
    }
    let inv = Array::from_slice(&[1.0 / accum as f32], &[1]);
    let mut out = HashMap::with_capacity(grads.len());
    for (k, g) in grads {
        out.insert(k, multiply(&g, &inv)?);
    }
    Ok(out)
}

/// **Weight noising** (epic 2123, sc-24826; relative mode of ai-toolkit-perceptual): permanently
/// perturb every trainable adapter tensor in `params` by
/// `w ← w + N(0, 1) · sigma · rms(w)`, where `rms(w) = sqrt(mean(w²))` is taken per tensor.
///
/// Call it right after each **real optimizer update** (never on a gradient-accumulation
/// micro-step), passing the 0-based index of the update that just fired. `params` holds only the
/// adapter factors (LoRA A/B, LoKr w1/w2/w2_a/w2_b) — the frozen base weights live inside the
/// model's `AdaptableLinear`s and are never reachable from here (epic 2123 E5).
///
/// Determinism (E4): the noise for tensor `i` (keys visited in sorted order, so `HashMap`
/// iteration order cannot leak in) at update `update_idx` is drawn from the RNG key
/// [`gen_core::train::technique_noise_key`]`(seed, WEIGHT_NOISE_SALT, update_idx, i)` only — the
/// same seeded run (or a resumed one) adds the same noise.
///
/// `sigma == 0.0` returns immediately without touching `params` or drawing any randomness, so an
/// off run is bit-identical to a run without this call (E1). A negative / non-finite `sigma` is an
/// error (the gen-core technique floor refuses it before training; this is the kernel's own guard).
pub fn apply_weight_noise(
    params: &mut LoraParams,
    sigma: f32,
    seed: u64,
    update_idx: u32,
) -> Result<()> {
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(crate::Error::Msg(format!(
            "weight noise: sigma must be a finite value >= 0, got {sigma}"
        )));
    }
    if sigma == 0.0 {
        return Ok(());
    }
    let mut keys: Vec<Rc<str>> = params.keys().cloned().collect();
    keys.sort();
    let sigma_arr = Array::from_slice(&[sigma], &[1]);
    for (i, key) in keys.iter().enumerate() {
        let w = &params[key];
        let dtype = w.dtype();
        let wf = w.as_dtype(Dtype::Float32)?;
        let rms = wf.square()?.mean(None)?.sqrt()?;
        let rng = random::key(gen_core::train::technique_noise_key(
            seed,
            gen_core::train::WEIGHT_NOISE_SALT,
            update_idx,
            i,
        ))?;
        let noise = random::normal::<f32>(w.shape(), None, None, Some(&rng))?;
        let delta = multiply(&noise, &multiply(&rms, &sigma_arr)?)?;
        let noised = mlx_rs::ops::add(&wf, &delta)?.as_dtype(dtype)?;
        params.insert(key.clone(), noised);
    }
    mlx_rs::transforms::eval(params.values())?;
    Ok(())
}

/// **Gradient noise** (epic 2123, sc-24827; the `neelakantan` mode of ai-toolkit-perceptual):
/// add `N(0, 1) · σ_t` to every adapter gradient in `grads`, with
/// `σ_t = eta / (1 + update_idx)^gamma` ([`gen_core::train::gradient_noise_std`]).
///
/// Call it once per **real optimizer update**, on the window-averaged gradients **after** the
/// global-norm clip and **before** the optimizer step (upstream's placement — the clip must not eat
/// the noise). `grads` is keyed like the adapter [`LoraParams`] — it never holds a base weight.
/// Seeded like [`apply_weight_noise`] but on the independent
/// [`GRADIENT_NOISE_SALT`](gen_core::train::GRADIENT_NOISE_SALT) stream (E4).
///
/// `eta == 0.0` returns before any RNG draw (E1); a malformed eta/gamma is an error.
pub fn apply_gradient_noise(
    grads: &mut LoraParams,
    eta: f32,
    gamma: f32,
    seed: u64,
    update_idx: u32,
) -> Result<()> {
    if !eta.is_finite() || eta < 0.0 || !gamma.is_finite() || gamma < 0.0 {
        return Err(crate::Error::Msg(format!(
            "gradient noise: eta and gamma must be finite values >= 0, got eta {eta}, gamma {gamma}"
        )));
    }
    if eta == 0.0 {
        return Ok(());
    }
    let std = gen_core::train::gradient_noise_std(eta, gamma, update_idx);
    let std_arr = Array::from_slice(&[std], &[1]);
    let mut keys: Vec<Rc<str>> = grads.keys().cloned().collect();
    keys.sort();
    for (i, key) in keys.iter().enumerate() {
        let g = &grads[key];
        let dtype = g.dtype();
        let rng = random::key(gen_core::train::technique_noise_key(
            seed,
            gen_core::train::GRADIENT_NOISE_SALT,
            update_idx,
            i,
        ))?;
        let noise = random::normal::<f32>(g.shape(), None, None, Some(&rng))?;
        let noised = mlx_rs::ops::add(&g.as_dtype(Dtype::Float32)?, &multiply(&noise, &std_arr)?)?
            .as_dtype(dtype)?;
        grads.insert(key.clone(), noised);
    }
    Ok(())
}

/// One real **adapter optimizer update** — the step every MLX LoRA/LoKr trainer fires once per
/// gradient-accumulation window (epic 2123 S2, sc-24827): clip the window-averaged `avg_grads` to
/// unit global norm → [`apply_gradient_noise`] → optimizer step → materialize →
/// [`apply_weight_noise`]. `update_idx` is the 0-based index of this update (drives the noise
/// schedule and RNG streams); `noise_seed` is the job seed (a multi-expert trainer passes its
/// per-expert seed so the experts draw independent noise). With both techniques off this is
/// exactly the pre-epic-2123 update (clip → step → eval), bit for bit.
pub fn adapter_optimizer_update(
    opt: &mut crate::train::optim::TrainOptimizer,
    params: &mut LoraParams,
    avg_grads: &LoraParams,
    cfg: &gen_core::TrainingConfig,
    update_idx: u32,
    noise_seed: u64,
) -> Result<()> {
    let grads = clip_and_noise_grads(avg_grads, cfg, update_idx, noise_seed)?;
    opt.step(params, &grads)?;
    mlx_rs::transforms::eval(params.values())?;
    apply_weight_noise(params, cfg.weight_noise_sigma, noise_seed, update_idx)?;
    Ok(())
}

/// The gradient half of [`adapter_optimizer_update`]: clip `avg_grads` to unit global norm, then
/// [`apply_gradient_noise`] on the clipped gradients (upstream's order — the clip must not shrink
/// the noise). Returns the gradients the optimizer steps on.
pub fn clip_and_noise_grads(
    avg_grads: &LoraParams,
    cfg: &gen_core::TrainingConfig,
    update_idx: u32,
    noise_seed: u64,
) -> Result<LoraParams> {
    let (clipped, _norm) = mlx_rs::optimizers::clip_grad_norm(avg_grads, 1.0)?;
    let mut clipped: LoraParams = clipped
        .into_iter()
        .map(|(k, v)| (k, v.into_owned()))
        .collect();
    apply_gradient_noise(
        &mut clipped,
        cfg.gradient_noise_eta,
        cfg.gradient_noise_gamma,
        noise_seed,
        update_idx,
    )?;
    Ok(clipped)
}

/// The trainable adapter kind — dispatches the per-step inject and the save the train loop calls,
/// so one loop drives both LoRA and LoKr. `install` takes the LoKr reconstruct dtype and `save` the
/// PEFT key prefix (the two per-family differences); both are no-ops for the other variant.
pub enum TrainAdapter {
    Lora { targets: Vec<LoraTarget> },
    Lokr { targets: Vec<LokrTarget> },
}

impl TrainAdapter {
    /// The dotted paths this adapter trains (for clearing the stack back to the bare base).
    pub fn paths(&self) -> Vec<String> {
        match self {
            TrainAdapter::Lora { targets } => targets.iter().map(|t| t.path.clone()).collect(),
            TrainAdapter::Lokr { targets } => targets.iter().map(|t| t.path.clone()).collect(),
        }
    }

    /// Number of trained target modules.
    pub fn len(&self) -> usize {
        match self {
            TrainAdapter::Lora { targets } => targets.len(),
            TrainAdapter::Lokr { targets } => targets.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn install<H: AdaptableHost>(
        &self,
        host: &mut H,
        params: &LoraParams,
        alpha: f32,
        rank: f32,
        lokr_dtype: Dtype,
    ) -> MlxResult<()> {
        self.install_as(host, params, alpha, rank, None, lokr_dtype)
    }

    /// [`install`](Self::install) with an optional LoRA compute-dtype cast (sc-4887 bf16 training);
    /// LoKr keeps its own `lokr_dtype` (the bf16-residual round-trip contract is independent).
    pub fn install_as<H: AdaptableHost>(
        &self,
        host: &mut H,
        params: &LoraParams,
        alpha: f32,
        rank: f32,
        lora_dtype: Option<Dtype>,
        lokr_dtype: Dtype,
    ) -> MlxResult<()> {
        match self {
            TrainAdapter::Lora { targets } => {
                install_training_lora_as(host, params, targets, alpha, lora_dtype)
            }
            TrainAdapter::Lokr { targets } => {
                install_training_lokr(host, params, targets, alpha, rank, lokr_dtype)
            }
        }
    }

    pub fn save(
        &self,
        params: &LoraParams,
        alpha: f32,
        rank: f32,
        decompose_factor: i32,
        key_prefix: &str,
        path: &Path,
    ) -> Result<()> {
        self.save_with_meta(params, alpha, rank, decompose_factor, key_prefix, &[], path)
    }

    /// [`save`](Self::save) with extra `__metadata__` entries (sc-14057) — one call site for both
    /// adapter kinds, so a family/provenance stamp cannot be applied to LoRA and forgotten for
    /// LoKr. See [`save_lora_peft_with_meta`].
    #[allow(clippy::too_many_arguments)]
    pub fn save_with_meta(
        &self,
        params: &LoraParams,
        alpha: f32,
        rank: f32,
        decompose_factor: i32,
        key_prefix: &str,
        extra: &[(&str, &str)],
        path: &Path,
    ) -> Result<()> {
        match self {
            TrainAdapter::Lora { targets } => save_lora_peft_with_meta(
                params,
                targets,
                alpha,
                rank as u32,
                key_prefix,
                extra,
                path,
            ),
            TrainAdapter::Lokr { targets } => {
                save_lokr_with_meta(params, targets, alpha, rank, decompose_factor, extra, path)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise_params() -> LoraParams {
        let mut p: LoraParams = HashMap::new();
        // Two tensors with very different scales so "∝ rms" is distinguishable from absolute noise,
        // plus an all-zero tensor (LoRA `B` at init) whose rms — and thus noise — is zero.
        let small = multiply(
            random::normal::<f32>(&[64, 32], None, None, Some(&random::key(11).unwrap())).unwrap(),
            Array::from_slice(&[0.01f32], &[1]),
        )
        .unwrap();
        let big = multiply(
            random::normal::<f32>(&[32, 64], None, None, Some(&random::key(12).unwrap())).unwrap(),
            Array::from_slice(&[3.0f32], &[1]),
        )
        .unwrap();
        p.insert(Rc::from("blk.to_q.lora_a"), small);
        p.insert(Rc::from("blk.to_k.lora_a"), big);
        p.insert(
            Rc::from("blk.to_q.lora_b"),
            Array::zeros::<f32>(&[32, 64]).unwrap(),
        );
        p
    }

    fn host_vec(a: &Array) -> Vec<f32> {
        let a = a.as_dtype(Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&a]).unwrap();
        a.as_slice::<f32>().to_vec()
    }

    fn rms(v: &[f32]) -> f64 {
        (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    }

    /// sc-24826 AC1: the per-tensor delta is noise whose RMS is `sigma · rms(w)` (relative mode),
    /// a zero tensor stays zero, and the noise is not identical across tensors.
    #[test]
    fn weight_noise_delta_is_proportional_to_each_tensors_rms() {
        let before = noise_params();
        let mut after = noise_params();
        let sigma = 0.0125f32;
        apply_weight_noise(&mut after, sigma, 7, 0).unwrap();
        for key in ["blk.to_q.lora_a", "blk.to_k.lora_a"] {
            let w0 = host_vec(&before[key]);
            let w1 = host_vec(&after[key]);
            let delta: Vec<f32> = w0.iter().zip(&w1).map(|(a, b)| b - a).collect();
            let ratio = rms(&delta) / (sigma as f64 * rms(&w0));
            // N(0,1) over 2048 elements: the sample RMS is within a few percent of 1.
            assert!(
                (0.9..1.1).contains(&ratio),
                "{key}: delta rms / (sigma·rms(w)) = {ratio}, expected ≈ 1"
            );
        }
        assert!(
            host_vec(&after["blk.to_q.lora_b"])
                .iter()
                .all(|x| *x == 0.0),
            "a zero tensor has rms 0, so relative noise must leave it untouched"
        );
    }

    /// sc-24826 AC3 (E1): sigma 0 is bit-identical to never calling the helper.
    #[test]
    fn weight_noise_sigma_zero_is_bit_identical() {
        let before = noise_params();
        let mut after = noise_params();
        apply_weight_noise(&mut after, 0.0, 7, 3).unwrap();
        for (k, v) in &before {
            assert_eq!(host_vec(v), host_vec(&after[k]), "{k} changed at sigma 0");
        }
    }

    /// sc-24826 (E4): same (seed, update) ⇒ identical noise; a different seed or update ⇒ different.
    #[test]
    fn weight_noise_is_derived_from_seed_and_update_index() {
        let run = |seed: u64, update: u32| {
            let mut p = noise_params();
            apply_weight_noise(&mut p, 0.0125, seed, update).unwrap();
            host_vec(&p["blk.to_k.lora_a"])
        };
        assert_eq!(
            run(7, 2),
            run(7, 2),
            "same seed/update must add the same noise"
        );
        assert_ne!(
            run(7, 2),
            run(8, 2),
            "the noise must depend on the job seed"
        );
        assert_ne!(
            run(7, 2),
            run(7, 3),
            "each optimizer update must draw fresh noise"
        );
    }

    #[test]
    fn weight_noise_rejects_malformed_sigma() {
        for bad in [-0.1f32, f32::NAN] {
            let mut p = noise_params();
            assert!(apply_weight_noise(&mut p, bad, 0, 0).is_err(), "{bad}");
        }
    }

    #[test]
    fn save_lora_peft_errors_on_missing_param() {
        // F-008: a target whose factor key isn't in the (empty) params map must surface a typed
        // error, not a panic from a direct map index.
        let target = LoraTarget {
            path: "blocks.0.attn.to_q".into(),
            a_key: Rc::from("blocks.0.attn.to_q.lora_a"),
            b_key: Rc::from("blocks.0.attn.to_q.lora_b"),
            in_f: 8,
            out_f: 8,
        };
        let params: LoraParams = HashMap::new();
        let err = save_lora_peft(
            &params,
            &[target],
            1.0,
            4,
            "",
            "/tmp/unused_sc4019.safetensors",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("LoRA param missing"), "got: {err}");
    }

    /// sc-24158: a fractional LoKr `alpha` survives save → metadata → [`parse_lokr`] (the writer used
    /// `as i64`, truncating `2.5` to `2` and silently rescaling the reloaded LoKr).
    ///
    /// [`parse_lokr`]: crate::adapters::loader::parse_lokr
    #[test]
    fn save_lokr_writes_a_fractional_alpha_losslessly() {
        use crate::adapters::AdaptableLinear;
        use crate::weights::Weights;
        struct OneLin(AdaptableLinear);
        impl AdaptableHost for OneLin {
            fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
                (path == ["to_q"]).then_some(&mut self.0)
            }
            fn adaptable_paths(&self) -> Vec<String> {
                vec!["to_q".to_string()]
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let mut host = OneLin(AdaptableLinear::dense(
            mlx_rs::random::normal::<f32>(&[8, 8], None, None, None).unwrap(),
            None,
        ));
        let (targets, params) =
            build_lokr_targets(&mut host, &["to_q".into()], 4, -1, 0).expect("lokr targets");
        let path = dir.path().join("lokr.safetensors");
        TrainAdapter::Lokr { targets }
            .save_with_meta(&params, 2.5, 4.0, -1, "", &[], &path)
            .expect("save");
        let w = Weights::from_file(&path).expect("reload");
        assert_eq!(w.metadata("alpha"), Some("2.5"));
        assert_eq!(w.metadata("rank"), Some("4"));
        let file = crate::adapters::loader::parse_lokr(&w).expect("parse");
        assert_eq!((file.rank, file.alpha), (4.0, 2.5));
    }

    /// sc-14057: caller-supplied `__metadata__` entries ride BOTH adapter kinds, and can never
    /// displace the `networkType`/`rank`/`alpha` reload contract even when they collide by name.
    #[test]
    fn save_with_meta_stamps_extra_metadata_on_both_adapter_kinds() {
        use crate::adapters::AdaptableLinear;
        use crate::weights::Weights;
        struct OneLin(AdaptableLinear);
        impl AdaptableHost for OneLin {
            fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
                (path == ["to_q"]).then_some(&mut self.0)
            }
            fn adaptable_paths(&self) -> Vec<String> {
                vec!["to_q".to_string()]
            }
        }
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let extra = [("family", "mage_flow"), ("networkType", "IGNORED")];

        for (kind, file) in [("lora", "a.safetensors"), ("lokr", "b.safetensors")] {
            let mut host = OneLin(AdaptableLinear::dense(
                mlx_rs::random::normal::<f32>(&[8, 8], None, None, None).unwrap(),
                None,
            ));
            let (adapter, params) = if kind == "lora" {
                let (targets, params) =
                    build_lora_targets(&mut host, &["to_q".into()], 4, 0).expect("lora targets");
                (TrainAdapter::Lora { targets }, params)
            } else {
                let (targets, params) = build_lokr_targets(&mut host, &["to_q".into()], 4, -1, 0)
                    .expect("lokr targets");
                (TrainAdapter::Lokr { targets }, params)
            };
            let path = dir.join(file);
            adapter
                .save_with_meta(&params, 8.0, 4.0, -1, "", &extra, &path)
                .expect("save");
            let w = Weights::from_file(&path).expect("reload");
            assert_eq!(
                w.metadata("family"),
                Some("mage_flow"),
                "{kind}: the provenance stamp must survive"
            );
            assert_eq!(
                w.metadata("networkType"),
                Some(kind),
                "{kind}: the reload contract must win over a colliding extra key"
            );
            assert_eq!(w.metadata("rank"), Some("4"));
        }
    }

    #[test]
    fn factorization_balances_and_respects_factor() {
        // Balanced (factor = -1): the squarest divisor pair, a <= b.
        assert_eq!(factorization(8, -1), (2, 4));
        assert_eq!(factorization(2560, -1), (40, 64));
        // A prime dimension cannot split → (1, p).
        assert_eq!(factorization(7, -1), (1, 7));
        // Explicit factor that divides: the pair straddles `factor`.
        assert_eq!(factorization(64, 8), (8, 8));
        let (a, b) = factorization(2048, 16);
        assert_eq!(a * b, 2048);
        assert!(a <= b);
    }

    #[test]
    fn build_lokr_init_matches_peft_reset_adapter_parameters() {
        // sc-5179 — the native LoKr init must match PEFT `LoKrConfig(init_weights=True)`'s
        // `reset_adapter_parameters` (w1 zero-init, w2 kaiming-uniform a=√5), so a Python LoKr lr
        // transfers. A minimal one-Linear host stands in for a model.
        use crate::adapters::AdaptableLinear;
        struct OneLin(AdaptableLinear);
        impl AdaptableHost for OneLin {
            fn adaptable_mut(&mut self, p: &[&str]) -> Option<&mut AdaptableLinear> {
                if p == ["w"] {
                    Some(&mut self.0)
                } else {
                    None
                }
            }
            fn adaptable_paths(&self) -> Vec<String> {
                vec!["w".into()]
            }
        }
        // out=64 → fac(-1)=(8,8); in=48 → (6,8). out_b = in_b = 8. rank 3 < max(8,8)/2 = 4 → low-rank.
        let host_w = Array::zeros::<f32>(&[64, 48]).unwrap();
        let mut host = OneLin(AdaptableLinear::dense(host_w, None));
        let rank = 3;
        let (targets, params) =
            build_lokr_targets(&mut host, &["w".to_string()], rank, -1, 7).unwrap();
        assert_eq!(targets.len(), 1);
        let t = &targets[0];

        // w1 is the ZEROED factor (PEFT), shape [out_a=8, in_a=6].
        let w1 = &params[&t.w1_key];
        assert_eq!(w1.shape(), &[8, 6]);
        assert_eq!(
            w1.abs().unwrap().sum(None).unwrap().item::<f32>(),
            0.0,
            "w1 must be zero-init (PEFT reset_adapter_parameters), not N(0,0.02)"
        );

        // w2 is low-rank kaiming: no full w2; w2_a [8,3] bound 1/√3, w2_b [3,8] bound 1/√8.
        assert!(
            t.w2_key.is_none() && t.w2a_key.is_some() && t.w2b_key.is_some(),
            "rank 3 < max(8,8)/2 must take the low-rank w2 path"
        );
        let w2a = &params[t.w2a_key.as_ref().unwrap()];
        let w2b = &params[t.w2b_key.as_ref().unwrap()];
        let maxabs = |a: &Array| a.abs().unwrap().max(None).unwrap().item::<f32>();
        let bound_a = 1.0f32 / (rank as f32).sqrt(); // ≈ 0.577
        let bound_b = 1.0f32 / 8f32.sqrt(); // ≈ 0.354
                                            // Within the kaiming bound, and clearly spread (NOT the old fixed-0.02 scale, whose max would
                                            // be ~0.06 — well under bound·0.3 here).
        assert!(
            maxabs(w2a) <= bound_a + 1e-6 && maxabs(w2a) > bound_a * 0.3,
            "w2_a must be kaiming U(±1/√rank): max {} vs bound {bound_a}",
            maxabs(w2a)
        );
        assert!(
            maxabs(w2b) <= bound_b + 1e-6 && maxabs(w2b) > bound_b * 0.3,
            "w2_b must be kaiming U(±1/√in_b): max {} vs bound {bound_b}",
            maxabs(w2b)
        );

        // Initial delta is exactly 0 (w1 = 0) — the adapter starts as a no-op, like LoRA's B = 0.
        let delta = reconstruct_lokr_delta(
            1.0,
            rank as f32,
            &[64, 48],
            Some(w1),
            None,
            None,
            None,
            Some(w2a),
            Some(w2b),
            Dtype::Float32,
        )
        .unwrap();
        assert_eq!(
            delta.abs().unwrap().max(None).unwrap().item::<f32>(),
            0.0,
            "initial LoKr delta must be exactly 0"
        );
    }

    #[test]
    fn average_grads_is_identity_for_unit_accum() {
        let mut p: LoraParams = HashMap::new();
        p.insert(Rc::from("x"), Array::from_slice(&[2.0f32, 4.0], &[2]));
        let out = average_grads(p, 1).unwrap();
        assert_eq!(out["x"].as_slice::<f32>(), &[2.0, 4.0]);
    }

    #[test]
    fn average_grads_divides_by_accum() {
        let mut p: LoraParams = HashMap::new();
        p.insert(Rc::from("x"), Array::from_slice(&[2.0f32, 4.0], &[2]));
        let out = average_grads(p, 2).unwrap();
        assert_eq!(out["x"].as_slice::<f32>(), &[1.0, 2.0]);
    }

    #[test]
    fn accumulate_grads_sums_into_running_total() {
        let mk = |v: f32| {
            let mut p: LoraParams = HashMap::new();
            p.insert(Rc::from("x"), Array::from_slice(&[v], &[1]));
            p
        };
        let mut acc: Option<LoraParams> = None;
        accumulate_grads(&mut acc, mk(1.0)).unwrap();
        accumulate_grads(&mut acc, mk(2.5)).unwrap();
        let acc = acc.unwrap();
        assert_eq!(acc["x"].as_slice::<f32>(), &[3.5]);
    }
}

/// Epic 2123 S2 (sc-24827) — gradient noise + the shared adapter optimizer update, on tiny
/// synthetic factor maps (seconds, < 1 MB).
#[cfg(test)]
mod adapter_noise_tests {
    use super::*;
    use crate::train::optim::TrainOptimizer;
    use gen_core::TrainingConfig;

    fn randn(shape: &[i32], seed: u64, scale: f32) -> Array {
        multiply(
            random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap())).unwrap(),
            Array::from_slice(&[scale], &[1]),
        )
        .unwrap()
    }

    fn host(a: &Array) -> Vec<f32> {
        let a = a.as_dtype(Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&a]).unwrap();
        a.as_slice::<f32>().to_vec()
    }

    fn rms(v: &[f32]) -> f64 {
        (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Two SEPARATE Metal executions of the same seeded graph (optimizer steps, norm/rms
    /// reductions) agree only to the last few ulps — Metal reductions are not bit-deterministic
    /// across runs (run 37255363552 differed by 1 ulp). Compare such results within
    /// `1e-6 + 1e-5·max|want|`; every mutation these tests guard (wrong noise index, noise before
    /// the step, a drifted seed, noise leaking into an "off" run) moves values by ≥ 1e-4.
    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        let scale = want.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let tol = 1e-6 + 1e-5 * scale;
        let diff = max_abs_diff(got, want);
        assert!(diff <= tol, "{what}: max |diff| {diff} > tolerance {tol}");
    }

    /// Zero gradients over two adapter tensors (8192 elements each) — whatever `apply_gradient_noise`
    /// leaves in them IS the noise.
    fn zero_grads() -> LoraParams {
        let mut g: LoraParams = HashMap::new();
        g.insert(
            Rc::from("blk.to_q.lora_a"),
            Array::zeros::<f32>(&[64, 128]).unwrap(),
        );
        g.insert(
            Rc::from("blk.to_q.lora_b"),
            Array::zeros::<f32>(&[128, 64]).unwrap(),
        );
        g
    }

    fn params() -> LoraParams {
        let mut p: LoraParams = HashMap::new();
        p.insert(Rc::from("blk.to_q.lora_a"), randn(&[8, 64], 21, 0.02));
        p.insert(Rc::from("blk.to_q.lora_b"), randn(&[64, 8], 22, 0.5));
        p.insert(Rc::from("blk.to_k.lokr_w1"), randn(&[4, 4], 23, 1.0));
        p
    }

    /// Fixed synthetic gradients (keys sorted, so `HashMap` order cannot change which draw lands
    /// on which factor).
    fn grads(seed: u64, scale: f32) -> LoraParams {
        let p = params();
        let mut keys: Vec<_> = p.keys().cloned().collect();
        keys.sort();
        keys.into_iter()
            .enumerate()
            .map(|(i, k)| {
                let g = randn(p[&k].shape(), seed + i as u64, scale);
                (k, g)
            })
            .collect()
    }

    /// AC3 annealing: the measured per-element std of the injected noise equals
    /// `eta / (1 + t)^gamma` at t = 0, 9, 99, 999 (16384 samples each ⇒ ~0.6% sampling error), so it
    /// shrinks with the update index exactly per the formula.
    ///
    /// *Mutation that reds this:* any change to the exponent in `gen_core::train::gradient_noise_std`
    /// (e.g. `powf(gamma)` → `powf(gamma * 0.5)`, or dropping it) — at t = 99 the measured std moves
    /// by > 2x.
    #[test]
    fn gradient_noise_std_anneals_with_the_update_index() {
        let (eta, gamma) = (0.01f32, 0.55f32);
        let mut prev = f64::INFINITY;
        for t in [0u32, 9, 99, 999] {
            let mut g = zero_grads();
            apply_gradient_noise(&mut g, eta, gamma, 7, t).unwrap();
            let all: Vec<f32> = g.values().flat_map(host).collect();
            let measured = rms(&all);
            let want = eta as f64 / (1.0 + t as f64).powf(gamma as f64);
            assert!(
                (measured / want - 1.0).abs() < 0.03,
                "t={t}: measured std {measured}, formula {want}"
            );
            assert!(measured < prev, "the noise must shrink with t");
            prev = measured;
        }
    }

    /// E1: eta 0 leaves the gradients bit-identical and draws nothing.
    #[test]
    fn gradient_noise_eta_zero_is_bit_identical() {
        let before = grads(5, 1.0);
        let mut after = grads(5, 1.0);
        apply_gradient_noise(&mut after, 0.0, 0.55, 7, 3).unwrap();
        for (k, v) in &before {
            assert_eq!(host(v), host(&after[k]), "{k} changed at eta 0");
        }
    }

    /// E4: same (seed, update) ⇒ same gradient noise; a different seed or update ⇒ different; and
    /// the gradient-noise stream is independent of the weight-noise stream for the same tensor.
    #[test]
    fn gradient_noise_is_seeded_and_on_its_own_stream() {
        let draw = |seed: u64, t: u32| {
            let mut g = zero_grads();
            apply_gradient_noise(&mut g, 1.0, 0.0, seed, t).unwrap();
            host(&g["blk.to_q.lora_a"])
        };
        assert_eq!(draw(7, 2), draw(7, 2));
        assert_ne!(draw(7, 2), draw(8, 2));
        assert_ne!(draw(7, 2), draw(7, 3));
        // Weight noise on a ones tensor (rms 1, sigma 1) adds exactly its unit draw.
        let mut w: LoraParams = HashMap::new();
        w.insert(
            Rc::from("blk.to_q.lora_a"),
            Array::ones::<f32>(&[64, 128]).unwrap(),
        );
        w.insert(
            Rc::from("blk.to_q.lora_b"),
            Array::ones::<f32>(&[128, 64]).unwrap(),
        );
        apply_weight_noise(&mut w, 1.0, 7, 2).unwrap();
        let weight_draw: Vec<f32> = host(&w["blk.to_q.lora_a"])
            .iter()
            .map(|x| x - 1.0)
            .collect();
        let grad_draw = draw(7, 2);
        let close = weight_draw
            .iter()
            .zip(&grad_draw)
            .filter(|(a, b)| (*a - *b).abs() < 1e-4)
            .count();
        assert!(
            close < 16,
            "weight and gradient noise must not share a stream"
        );
    }

    #[test]
    fn gradient_noise_rejects_malformed_knobs() {
        for (eta, gamma) in [
            (-0.1f32, 0.55f32),
            (f32::NAN, 0.55),
            (0.01, -1.0),
            (0.01, f32::NAN),
        ] {
            let mut g = zero_grads();
            assert!(
                apply_gradient_noise(&mut g, eta, gamma, 0, 0).is_err(),
                "{eta} {gamma}"
            );
        }
    }

    /// Placement: the noise is added AFTER the unit-norm clip, so the clip cannot shrink it. One
    /// tensor carries a huge gradient (norm ≫ 1, so the clip scales everything by ~1e-4) and the
    /// other a zero gradient; the zero tensor's clipped-and-noised gradient has std `eta`, not
    /// `eta · 1e-4`.
    ///
    /// *Mutation that reds this:* applying the noise before `clip_grad_norm` in
    /// `clip_and_noise_grads`.
    #[test]
    fn gradient_noise_is_added_after_the_clip() {
        let mut g: LoraParams = HashMap::new();
        g.insert(
            Rc::from("big"),
            multiply(
                Array::ones::<f32>(&[64, 64]).unwrap(),
                Array::from_slice(&[1.0e3f32], &[1]),
            )
            .unwrap(),
        );
        g.insert(Rc::from("zero"), Array::zeros::<f32>(&[128, 128]).unwrap());
        let cfg = TrainingConfig {
            gradient_noise_eta: 0.01,
            gradient_noise_gamma: 0.55,
            ..Default::default()
        };
        let out = clip_and_noise_grads(&g, &cfg, 0, 7).unwrap();
        let measured = rms(&host(&out["zero"]));
        assert!(
            (measured / 0.01 - 1.0).abs() < 0.03,
            "post-clip noise std {measured}, want 0.01"
        );
    }

    fn legacy_update(opt: &mut TrainOptimizer, p: &mut LoraParams, g: &LoraParams) {
        let (clipped, _) = mlx_rs::optimizers::clip_grad_norm(g, 1.0).unwrap();
        let clipped: LoraParams = clipped
            .into_iter()
            .map(|(k, v)| (k, v.into_owned()))
            .collect();
        opt.step(p, &clipped).unwrap();
        mlx_rs::transforms::eval(p.values()).unwrap();
    }

    /// E1: with both techniques off the shared update is the pre-epic-2123 update (clip → step →
    /// eval, no noise). Compared within [`assert_close`]: the two runs are separate Metal
    /// executions.
    ///
    /// *Mutation that reds this:* any noise leaking into an "off" update (e.g. forcing a non-zero
    /// eta inside `adapter_optimizer_update`).
    #[test]
    fn techniques_off_update_matches_the_legacy_update() {
        let cfg = TrainingConfig::default();
        let (mut new, mut old) = (params(), params());
        let mut o1 = TrainOptimizer::from_config("adamw", 1e-2, 0.0).unwrap();
        let mut o2 = TrainOptimizer::from_config("adamw", 1e-2, 0.0).unwrap();
        for t in 0..3 {
            let g = grads(100 + t as u64, 3.0);
            adapter_optimizer_update(&mut o1, &mut new, &g, &cfg, t, cfg.seed).unwrap();
            legacy_update(&mut o2, &mut old, &g);
        }
        for (k, v) in &old {
            assert_close(
                &host(&new[k]),
                &host(v),
                &format!("{k} with techniques off"),
            );
        }
    }

    /// Weight noise fires AFTER the optimizer step, on the update's own index: the shared update
    /// with sigma > 0 equals the legacy update followed by `apply_weight_noise(.., update_idx)`.
    ///
    /// *Mutation that reds this:* applying weight noise before `opt.step`, or passing a different
    /// index (e.g. `update_idx + 1`).
    #[test]
    fn weight_noise_follows_the_step_on_the_update_index() {
        let cfg = TrainingConfig {
            weight_noise_sigma: 0.0125,
            ..Default::default()
        };
        let (mut new, mut old) = (params(), params());
        let mut o1 = TrainOptimizer::from_config("adamw", 1e-2, 0.0).unwrap();
        let mut o2 = TrainOptimizer::from_config("adamw", 1e-2, 0.0).unwrap();
        for t in 0..3 {
            let g = grads(200 + t as u64, 3.0);
            adapter_optimizer_update(&mut o1, &mut new, &g, &cfg, t, 42).unwrap();
            legacy_update(&mut o2, &mut old, &g);
            apply_weight_noise(&mut old, 0.0125, 42, t).unwrap();
        }
        for (k, v) in &old {
            assert_close(&host(&new[k]), &host(v), k);
        }
    }

    /// Gradient noise reaches the optimizer: with eta > 0 the stepped factors differ from the clean
    /// update by far more than run-to-run Metal jitter, and two seeded runs agree within
    /// [`assert_close`] (E4).
    ///
    /// *Mutations that red this:* dropping the gradient noise (no difference from clean); an
    /// unseeded / drifting noise stream (the two noisy runs diverge far beyond the tolerance).
    #[test]
    fn gradient_noise_changes_the_update_reproducibly() {
        let run = |eta: f32| {
            let cfg = TrainingConfig {
                gradient_noise_eta: eta,
                ..Default::default()
            };
            let mut p = params();
            let mut o = TrainOptimizer::from_config("adamw", 1e-2, 0.0).unwrap();
            for t in 0..2 {
                adapter_optimizer_update(&mut o, &mut p, &grads(300 + t as u64, 3.0), &cfg, t, 9)
                    .unwrap();
            }
            host(&p["blk.to_q.lora_a"])
        };
        let (clean, a, b) = (run(0.0), run(0.05), run(0.05));
        let effect = max_abs_diff(&a, &clean);
        assert!(
            effect > 1e-3,
            "gradient noise must reach the step (max |diff| {effect})"
        );
        assert_close(&b, &a, "seeded gradient noise must reproduce");
    }
}
