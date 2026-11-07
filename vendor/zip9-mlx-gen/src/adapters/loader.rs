//! Adapter-file loaders — read a trained LoRA/LoKr `.safetensors` and install it onto a
//! model tree via [`AdaptableHost`]. Closes out sc-2343's loader piece.
//!
//! **LoKr** is generic and faithfully ported from the fork's `LoKrLoader.apply`: keys are
//! bare module paths (`‹path›.lokr_w1`/`lokr_w2`, full or low-rank `_a`/`_b`) and the file
//! carries `networkType=lokr` + `alpha`/`rank` in safetensors metadata, so the delta and
//! target path are fully determined by the file — no per-model mapping table.
//!
//! **LoRA** here covers two on-disk conventions, both family-agnostic:
//! - **PEFT/diffusers** (`‹prefix›‹path›.lora_A/B.weight` + optional `‹path›.alpha`): dotted module
//!   paths resolve directly via [`AdaptableHost::adaptable_mut`] ([`apply_lora_peft`]).
//! - **kohya / sd-scripts** (`lora_unet_‹path, .→_›.lora_down/up.weight` + optional `.alpha`,
//!   sc-2618): the flattened module path can't be re-split blindly, so it resolves through a
//!   `flattened → dotted` table built from [`AdaptableHost::adaptable_paths`]
//!   ([`apply_lora_kohya`]). kohya `lora_down`/`lora_up` == PEFT `lora_A`/`lora_B`, so both feed the
//!   shared `install_lora_groups` and a kohya file yields the identical adapter to its PEFT twin.
//!
//! - **BFL / ComfyUI** (`lora_unet_double_blocks_*` / `diffusion_model.…` / `base_model.model.…`,
//!   sc-2743): a *fused* source linear (`…img_attn.qkv`, `…linear1`) is row-sliced into the model's
//!   *split* targets (`attn.to_q/to_k/to_v`, …) via per-target [`LoraRowSlice`] transforms, with BFL
//!   module renames (`img_in`→`x_embedder`). This is fused→split weight surgery, orthogonal to the
//!   kohya underscore form; the host supplies its table via [`AdaptableHost::bfl_targets`]
//!   ([`apply_lora_bfl`]). Only FLUX.2/FLUX.1 expose one; for other hosts a BFL file's keys surface as
//!   unmatched (loud), never silently dropped.

use std::collections::{BTreeMap, BTreeSet};

use mlx_rs::memory::{get_active_memory, get_memory_limit};
use mlx_rs::ops::add;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use super::{
    build_lokr_factors, reconstruct_loha_delta, reconstruct_lokr_delta,
    reconstruct_lokr_delta_scaled, AdaptableHost, Adapter, DiffPatchPart, LokrFactors,
};
use crate::runtime::{AdapterKind, AdapterSpec};
use crate::weights::Weights;
use crate::{Error, Result};

// The format predicates, factor-suffix tables, rank/alpha parsing, and key-alias resolution are
// backend-neutral string/metadata logic and live in gen-core (sc-3722); this module supplies the
// `Weights`/`Array` half (factor grouping + delta reconstruction). The historical
// `mlx_gen::adapters::loader::{KOHYA_PREFIX, COMMON_LORA_PREFIXES, resolve_lokr_path}` paths stay
// resolvable via these re-exports.
use gen_core::weightsmeta as wmeta;
pub use gen_core::weightsmeta::{resolve_lokr_path, COMMON_LORA_PREFIXES, KOHYA_PREFIX};

/// PEFT LoKr per-module factor suffixes (gen-core's table) — each factor is full (`lokr_w1`/
/// `lokr_w2`) or low-rank (`_a`/`_b`). Exact-suffix matched.
use gen_core::weightsmeta::LOKR_SUFFIXES;

/// `true` if the file's `networkType` metadata marks it a LoKr adapter.
pub fn is_lokr(w: &Weights) -> bool {
    wmeta::is_lokr_network_type(w.metadata("networkType"))
}

/// A parsed LoKr file: the global `(alpha, rank)` from metadata plus every module's Kronecker
/// factors grouped by path. The factor map is keyed by the bare factor name (`lokr_w1`,
/// `lokr_w1_a`, `lokr_w1_b`, `lokr_w2`, `lokr_w2_a`, `lokr_w2_b`); a module is full or low-rank.
///
/// This is the format-parsing half of a LoKr install, factored out of [`apply_lokr`] so the video
/// providers (LTX/Wan) — which install onto their crate-local `Linear`s as a forward-time residual
/// or an in-place weight merge, rather than the core [`AdaptableHost`] — reuse the exact same factor
/// grouping + metadata read and differ only in the install step. Each provider then maps the bare
/// module `path` through its own key→module table and calls [`reconstruct_lokr_delta`].
#[derive(Debug)]
pub struct LokrFile {
    pub alpha: f32,
    pub rank: f32,
    /// `module path → { factor name → tensor }`.
    pub groups: BTreeMap<String, BTreeMap<String, Array>>,
}

impl LokrFile {
    /// Reconstruct one module's `[out,in]` delta at `out_dtype` from its grouped factors, baking in
    /// `alpha/rank` (the user `strength` is applied separately). `base_shape` is the target linear's
    /// logical weight shape. Returns the [`reconstruct_lokr_delta`] result.
    pub fn delta(
        &self,
        factors: &BTreeMap<String, Array>,
        base_shape: &[i32],
        out_dtype: Dtype,
    ) -> Result<Array> {
        reconstruct_lokr_delta(
            self.alpha,
            self.rank,
            base_shape,
            factors.get("lokr_w1"),
            factors.get("lokr_w1_a"),
            factors.get("lokr_w1_b"),
            factors.get("lokr_w2"),
            factors.get("lokr_w2_a"),
            factors.get("lokr_w2_b"),
            out_dtype,
        )
    }

    /// The **structured** (deferred, allocation-free) counterpart to [`delta`](Self::delta) for a
    /// packed base (sc-10578, extending sc-10050 from the video providers to every family on the
    /// shared install path): build the small `[a,c]`/`[b,d]` Kronecker factors so the residual applies
    /// via the vec-trick without ever materializing the `[out,in]` delta.
    ///
    /// **Scale differs from [`delta`](Self::delta) by design.** `delta` bakes only `alpha/rank` and
    /// leaves the user `strength` to [`Adapter::Lokr`]'s `scale` field; the structured residual carries
    /// no separate scale (it is baked into `factors.w2`), so the FULL `alpha/rank · strength` is
    /// applied here. Mismatching these two is a silent mis-scale, not a crash.
    ///
    /// Returns `Ok(None)` when the module has no 2-D matrix form (a conv-shaped factor); the peft
    /// LoKr format never carries a tucker `lokr_t2` — that is third-party-only, hence the `None`
    /// passed through. The caller then materializes (dense base) or errors (packed base).
    pub fn factors(
        &self,
        factors: &BTreeMap<String, Array>,
        strength: f32,
        base_shape: &[i32],
        out_dtype: Dtype,
    ) -> Result<Option<LokrFactors>> {
        build_lokr_factors(
            (self.alpha / self.rank) * strength,
            base_shape,
            factors.get("lokr_w1"),
            factors.get("lokr_w1_a"),
            factors.get("lokr_w1_b"),
            factors.get("lokr_w2"),
            None, // peft LoKr never carries a tucker `lokr_t2` (lycoris conv-only).
            factors.get("lokr_w2_a"),
            factors.get("lokr_w2_b"),
            out_dtype,
        )
    }
}

/// Parse a LoKr `.safetensors` into [`LokrFile`]: read `rank`/`alpha` from metadata (alpha defaults
/// to rank, i.e. scale 1.0, matching PEFT) and group every `‹path›.lokr_*` tensor by module path.
/// Shared by [`apply_lokr`] (core `AdaptableHost` install) and the video providers' crate-local
/// residual/merge installers.
pub fn parse_lokr(w: &Weights) -> Result<LokrFile> {
    // rank/alpha (alpha defaults to rank ⇒ scale 1.0, matching PEFT) — parsed in gen-core.
    let (rank, alpha) = wmeta::parse_rank_alpha(w.metadata("rank"), w.metadata("alpha"));

    let keys: Vec<String> = w.keys().map(str::to_string).collect();
    let mut groups: BTreeMap<String, BTreeMap<String, Array>> = BTreeMap::new();
    for key in &keys {
        if let Some((path, factor)) = wmeta::split_factor_key(key, &LOKR_SUFFIXES) {
            groups
                .entry(path.to_string())
                .or_default()
                .insert(factor.to_string(), w.require(key)?.clone());
        }
    }
    Ok(LokrFile {
        alpha,
        rank,
        groups,
    })
}

/// Read a scalar adapter value (an `alpha`) as `f32`, regardless of its on-disk dtype. Trained
/// adapters store `alpha` in their compute dtype: real kohya/BFL FLUX LoRAs ship it **bf16** (sc-2657),
/// and `Array::as_slice::<f32>()` `unwrap`s a hard dtype-mismatch (it never casts), so reading a bf16
/// scalar that way panics. Cast to f32 first (exact for the small integer alphas these files carry, and
/// a no-op when already f32); a `[]`- or `[1]`-shaped scalar both read as a one-element slice.
///
/// A size-0 `alpha` tensor (a malformed third-party adapter file) has no data pointer to borrow, so
/// `as_slice` — `try_as_slice().unwrap()` — would panic rather than fall through to the trailing
/// `first()`. Guard it like [`crate::array::host_i32`] so one bad file fails its single job with a
/// typed error instead of aborting the worker.
fn scalar_alpha(a: &Array) -> Result<Option<f32>> {
    if a.size() == 0 {
        // The callers only reach here for an `alpha` key that *exists* (`w.require(..)`), so a
        // present-but-empty tensor is a malformed file — NOT an absent alpha. Returning `Ok(None)`
        // here would be indistinguishable from "no alpha key", silently falling back to the
        // `alpha == rank ⇒ scale 1.0` default and mis-scaling the adapter while reporting success.
        // Fail this one job with a typed error instead (F-031, matching the contract documented above).
        return Err(Error::Msg(
            "scalar_alpha: alpha tensor present but empty (size 0) — malformed adapter file".into(),
        ));
    }
    Ok(a.as_dtype(Dtype::Float32)?
        .try_as_slice::<f32>()
        .map_err(|e| Error::Msg(format!("scalar_alpha: not a readable scalar array: {e}")))?
        .first()
        .copied())
}

/// Outcome of installing an adapter file: how many target modules were adapted, and any
/// adapter keys that matched no module in the host (surfaced, never silently dropped).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ApplyReport {
    pub applied: usize,
    pub unmatched_paths: Vec<String>,
    /// ComfyUI/lightx2v diff-patch (`.diff`/`.diff_b`) deltas that did **not** land — skipped
    /// (shape-incompatible, or a weight `.diff` aimed at a packed Linear) plus stems that matched no
    /// module (sc-15326). Always empty on [`apply_adapters_strict`], which does not read diff-patch
    /// keys at all; populated by [`apply_adapters_strict_with_diff_patch`].
    ///
    /// Carried on the report rather than left to `stderr` because a partial adapter install is a fact a
    /// *user* has to be told — the render is changed but not by the whole LoRA, which looks like a
    /// success. A provider stamps this into the asset's provenance; an engine log line is not a
    /// product surface.
    pub diff_patch_unapplied: Vec<String>,
}

/// How a LyCORIS applier resolves a file's raw module key to the host's dotted module path
/// (the one axis the non-BFL LyCORIS installers differ on — see [`install_lycoris_groups`]).
enum LycorisKeyResolution<'t> {
    /// Keys are already bare dotted module paths (the peft LoKr convention, [`apply_lokr`]).
    Dotted,
    /// Trainer raw keys: a kohya/lycoris `<PREFIX>_<flattened>` stem resolves through the
    /// host-derived flattened→dotted table ([`kohya_table`]); anything else is treated as
    /// already-dotted after stripping a common `transformer.`/`diffusion_model.` namespace
    /// ([`strip_common_lora_prefix`], sc-8395).
    Thirdparty(&'t BTreeMap<String, String>),
}

/// Bytes → GiB base for the sc-10678 pre-flight guard's messages.
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
/// The materialized `[out,in]` LoKr/LoHa delta dtype is bf16 by default (PARITY-BF16, sc-2609): 2
/// bytes/element; a host whose [`AdaptableHost::lycoris_dtype`] is f32 materializes 4.
const DELTA_DTYPE_BYTES: usize = 2;
/// Fraction of the MLX memory limit the guard treats as usable, leaving headroom for the forward-pass
/// activation working set (mirrors the sc-4874 training pre-flight's 0.85).
const MATERIALIZE_BUDGET_HEADROOM: f64 = 0.85;

/// Bytes a materialized `[out, in]` bf16 delta occupies — the memory a non-deferrable adapter (LoHa /
/// tucker-LoKr) forces onto a packed base per target (sc-10678). Pure (no MLX state) ⇒ deterministically
/// unit-testable.
fn projected_delta_bytes(base_shape: &[i32], dtype: Dtype) -> usize {
    let element = match dtype {
        Dtype::Float32 => 4,
        _ => DELTA_DTYPE_BYTES,
    };
    base_shape
        .iter()
        .map(|&d| d.max(0) as usize)
        .product::<usize>()
        * element
}

/// Whether materializing `projected_new` more bytes on top of the `active` resident set would cross the
/// safe budget (`limit × headroom`). `limit == 0` means MLX has no cap set (unlimited) ⇒ never guard.
/// Pure (the caller supplies the live `get_active_memory()` / `get_memory_limit()`) ⇒ deterministically
/// unit-testable without touching global MLX memory state.
fn materialization_exceeds_budget(active: usize, projected_new: usize, limit: usize) -> bool {
    if limit == 0 {
        return false;
    }
    let safe = (limit as f64 * MATERIALIZE_BUDGET_HEADROOM) as usize;
    active.saturating_add(projected_new) > safe
}

/// What one resolved LyCORIS group will do in the install pass — computed up front so the packed +
/// non-deferrable **materialization** (LoHa / tucker-LoKr) can be projected and budget-checked BEFORE any
/// `[out,in]` delta is allocated (sc-10678). Deferrable / dense / unmatched groups carry no such cost.
enum LycorisPlan<F> {
    /// Key resolved to no module (surfaced in `unmatched_paths` under its raw spelling).
    Unmatched(String),
    /// Dense base → materialize `[out,in]` and stack as `Adapter::Lokr` (unchanged, fork-parity).
    Dense { dotted: String, delta: F },
    /// Packed base with a deferrable Kronecker form → stack `Adapter::LokrStructured` (allocation-free).
    Deferred {
        dotted: String,
        factors: LokrFactors,
    },
    /// Packed base, NOT deferrable → materialize `[out,in]` (the budget-guarded case).
    Materialize {
        dotted: String,
        base_shape: Vec<i32>,
        delta: F,
    },
}

/// Shared install skeleton for the non-BFL LyCORIS appliers ([`apply_lokr`],
/// [`apply_lokr_thirdparty`], [`apply_loha_thirdparty`] — F-067): resolve each module's raw key to
/// a host dotted path per `resolution`, then stack a residual onto the target. A key that resolves to
/// no module is surfaced in `unmatched_paths` under its raw spelling, never silently dropped. The BFL
/// fused→split twin is [`install_bfl_lycoris`].
///
/// **On a packed base, prefer the deferred residual** (sc-10578):
///   * **Dense base** — `delta(base_shape)` materializes the `[out,in]` `ΔW` (the closure bakes in the
///     format's `alpha/rank` scale and dtype) and it stacks as an [`Adapter::Lokr`] at the user
///     `scale`. Byte-for-byte the pre-sc-10578 behavior, which every family's fork-parity goldens
///     depend on.
///   * **Packed base, deferrable module** — `factors(scale, base_shape)` builds the small Kronecker
///     factors and it stacks as an [`Adapter::LokrStructured`], applying the residual via the vec-trick
///     **without ever allocating `[out,in]`**. Note the scale: the structured residual carries none of
///     its own (it is baked into `factors.w2`), so the user `scale` is passed *in* here, whereas the
///     dense path passes it to `Adapter::Lokr`. Getting that backwards is a silent mis-scale.
///   * **Packed base, NOT deferrable** (`factors` → `Ok(None)`: LoHa's Hadamard product, which has no
///     deferred form at all; a tucker/CP `lokr_t2`; a conv-shaped factor) — **fall back to the
///     materialized delta**, exactly as before. It is correct: [`Adapter::Lokr`]'s residual is
///     `scale · x·ΔWᵀ`, which never reads the base, so it rides over `quantized_matmul` unchanged.
///     It is merely memory-hungry.
///
/// **The materialized fallback is budget-guarded, not silent (sc-10678).** An earlier cut of sc-10578
/// rejected the non-deferrable packed combination outright, reasoning from sc-10051 (LoHa on quantized
/// Wan: steer to bf16, do not materialize). That was wrong as a blanket rule *here*: this shared path is
/// reached by eleven provider crates, adapters install **after** `.quantize()` in every one, and the
/// worker defaults `spec.quantize` to `Some(Q8)` for **every** MLX model — so `is_quantized()` is true on
/// the default path almost everywhere, and erroring unconditionally would turn a working third-party LoHa
/// that *fits* into a hard failure. So the fallback stays: a materialization that fits the memory budget
/// runs, on the user's chosen (quantized) tier. But a materialization that would **not** fit is refused
/// UP FRONT with a catchable, actionable error ([`materialization_exceeds_budget`]) rather than left to a
/// silent mid-run SIGKILL when the OS hard-kills the worker. The tier is never switched (it is a creative
/// choice, not a perf knob); bf16 is only *suggested*, framed as a different render. This is the sc-10678
/// decision (E: pre-flight memory guard), mirroring the sc-4874 training pre-flight.
fn install_lycoris_groups<K, F, G>(
    host: &mut impl AdaptableHost,
    groups: impl IntoIterator<Item = (K, F, G)>,
    scale: f32,
    resolution: LycorisKeyResolution<'_>,
) -> Result<ApplyReport>
where
    K: AsRef<str> + Into<String>,
    F: FnOnce(&[i32]) -> Result<Array>,
    G: FnOnce(f32, &[i32]) -> Result<Option<LokrFactors>>,
{
    let mut report = ApplyReport::default();

    // Pass 1 — resolve every group to a concrete plan, consuming the `factors` probe (which is cheap:
    // it builds the small Kronecker factors or returns `None`, but never the `[out,in]` delta). Sum the
    // bytes EVERY **materialization** will allocate — the packed + non-deferrable case AND the dense-base
    // case, both of which stack a full `[out,in]` bf16 delta per target (F-011, sc-11129) — so we can
    // budget-check them before the (potentially multi-GB) reconstruction runs. Only deferred (structured,
    // allocation-free) and unmatched groups add nothing to `projected_materialize`.
    let mut plans: Vec<LycorisPlan<F>> = Vec::new();
    let mut projected_materialize: usize = 0;
    let dtype = host.lycoris_dtype();
    // Resolved module path → the raw key that claimed it. Two raw spellings of one module in one file
    // (`lycoris_X` beside `lora_unet_X`, `diffusion_model.X` beside `X`) would otherwise BOTH install
    // and double-apply the delta — refuse instead (sc-24158).
    let mut claimed: BTreeMap<String, String> = BTreeMap::new();
    for (raw, delta, factors) in groups {
        let dotted: String = match &resolution {
            LycorisKeyResolution::Dotted => raw.as_ref().to_string(),
            LycorisKeyResolution::Thirdparty(table) => resolve_lokr_path(raw.as_ref(), table)
                .unwrap_or_else(|| strip_common_lora_prefix(raw.as_ref()))
                .to_string(),
        };
        if let Some(previous) = claimed.insert(dotted.clone(), raw.as_ref().to_string()) {
            return Err(Error::Msg(format!(
                "LyCORIS adapter carries both `{previous}` and `{}`, which resolve to the same \
                 module `{dotted}`; refusing a double apply",
                raw.as_ref()
            )));
        }
        let parts: Vec<&str> = dotted.split('.').collect();
        // SC-18319 — pass 1 only *reads* (is the base packed, how big is it), so it goes through the
        // PROBE half of the host surface. Taking the `&mut` here would unfuse every `FusedQkvProjection`
        // it walked past — dismantling the fusion during a scan that installs nothing.
        match host.adaptable_facts(&parts) {
            // Packed base: take the deferred form when the module has one, else plan a materialization.
            Some(facts) if facts.is_quantized => {
                let base_shape = facts.base_shape;
                match factors(scale, &base_shape)? {
                    Some(f) => plans.push(LycorisPlan::Deferred { dotted, factors: f }),
                    None => {
                        projected_materialize += projected_delta_bytes(&base_shape, dtype);
                        plans.push(LycorisPlan::Materialize {
                            dotted,
                            base_shape,
                            delta,
                        });
                    }
                }
            }
            Some(facts) => {
                // Dense base still materializes the same `[out,in]` bf16 delta as a stacked residual —
                // the OOM the guard prevents on packed tiers is reachable here too (F-011). Project it.
                projected_materialize += projected_delta_bytes(&facts.base_shape, dtype);
                plans.push(LycorisPlan::Dense { dotted, delta });
            }
            None => plans.push(LycorisPlan::Unmatched(raw.into())),
        }
    }

    // sc-10678 — pre-flight memory guard. A non-deferrable adapter (LoHa / tucker-LoKr) must materialize
    // a dense `[out,in]` delta per target — on a PACKED (q4/q8) base (no deferred form) OR a dense base
    // (F-011: the stacked residual is the same footprint). Over a whole model that can dwarf the base
    // footprint; left unchecked it OOMs the worker via an uncatchable SIGKILL mid-run. Refuse UP FRONT —
    // but only when the projected materialization would actually exceed the budget, so an adapter that
    // fits still runs on the user's chosen tier (a plain LoRA or an all-deferred LoKr never reach this
    // branch: `projected_materialize == 0`).
    if projected_materialize > 0 {
        let active = get_active_memory();
        let limit = get_memory_limit();
        if materialization_exceeds_budget(active, projected_materialize, limit) {
            return Err(Error::Msg(format!(
                "This adapter (LoHa / tucker-LoKr) has no allocation-free form, so it must materialize a \
                 dense `[out,in]` delta for every target — ~{:.1} GB on top of the ~{:.1} GB already \
                 resident, over the ~{:.1} GB budget. Use a plain LoRA/LoKr instead (near-zero extra \
                 memory on any tier). On a quantized tier, the bf16 tier merges this adapter into the \
                 dense weight in place — note bf16 renders differently from the quantized tier.",
                projected_materialize as f64 / GIB,
                active as f64 / GIB,
                (limit as f64 * MATERIALIZE_BUDGET_HEADROOM) / GIB,
            )));
        }
    }

    // Pass 2 — install. Behaviour is identical to the pre-guard single loop: deferred residual on a
    // deferrable packed module, materialized `Adapter::Lokr` on a non-deferrable packed or dense module,
    // and `unmatched_paths` under the raw key. Re-resolving by `dotted` is a cheap map lookup (the host
    // is not structurally mutated between passes).
    // A pass-2 re-resolution that returns `None` (the module vanished between passes — unreachable on a
    // stable host today, but lazy/offloaded module trees are now a live pattern, epic 10834) must be
    // SURFACED in `unmatched_paths`, never silently dropped, honoring the function's contract (F-012,
    // sc-11129). The dotted path is the resolved spelling (the raw key is gone by pass 2).
    for plan in plans {
        match plan {
            LycorisPlan::Unmatched(raw) => report.unmatched_paths.push(raw),
            LycorisPlan::Deferred { dotted, factors } => {
                let parts: Vec<&str> = dotted.split('.').collect();
                if let Some(lin) = host.adaptable_mut(&parts) {
                    lin.push(Adapter::LokrStructured { factors });
                    report.applied += 1;
                } else {
                    report.unmatched_paths.push(dotted);
                }
            }
            LycorisPlan::Materialize {
                dotted,
                base_shape,
                delta,
            } => {
                let delta = delta(&base_shape)?;
                let parts: Vec<&str> = dotted.split('.').collect();
                if let Some(lin) = host.adaptable_mut(&parts) {
                    lin.push(Adapter::Lokr { delta, scale });
                    report.applied += 1;
                } else {
                    report.unmatched_paths.push(dotted);
                }
            }
            LycorisPlan::Dense { dotted, delta } => {
                let parts: Vec<&str> = dotted.split('.').collect();
                if let Some(lin) = host.adaptable_mut(&parts) {
                    let base_shape = lin.base_shape();
                    let delta = delta(&base_shape)?;
                    lin.push(Adapter::Lokr { delta, scale });
                    report.applied += 1;
                } else {
                    report.unmatched_paths.push(dotted);
                }
            }
        }
    }
    Ok(report)
}

/// Install a LoKr adapter file onto `host`. `scale` is the user-facing strength (the
/// `alpha/rank` factor is baked into the reconstructed delta, mirroring the fork).
pub fn apply_lokr(host: &mut impl AdaptableHost, w: &Weights, scale: f32) -> Result<ApplyReport> {
    let file = parse_lokr(w)?;
    let file = &file;
    // Fork-parity residual path keeps the delta at bf16 (PARITY-BF16, sc-2609) unless the host
    // computes wider ([`AdaptableHost::lycoris_dtype`]).
    let dtype = host.lycoris_dtype();
    let groups = file.groups.iter().map(|(path, factors)| {
        (
            path.as_str(),
            move |base: &[i32]| file.delta(factors, base, dtype),
            move |strength: f32, base: &[i32]| file.factors(factors, strength, base, dtype),
        )
    });
    install_lycoris_groups(host, groups, scale, LycorisKeyResolution::Dotted)
}

// ---- Third-party LyCORIS LoKr (sc-3642) ----------------------------------------------------------

/// Third-party LoKr factor suffixes — the PEFT set plus `lokr_t2` (the lycoris tucker/CP factor).
/// `.lokr_w1_a`/`_b` precede the bare `.lokr_w1` so exact-suffix matching never mis-binds. (gen-core.)
use gen_core::weightsmeta::LOKR_TP_SUFFIXES;

/// `true` if any key is a LoKr factor (`*.lokr_w…`), regardless of `networkType` metadata. This is
/// how a **third-party** LyCORIS LoKr (kohya / ai-toolkit / lycoris-lib) is recognized — those files
/// ship the Kronecker factors but NOT SceneWorks' peft `networkType=lokr` stamp that [`is_lokr`]
/// keys off. (A `lokr_t2` tucker factor always co-occurs with `lokr_w2_a`, so `.lokr_w` suffices.)
pub fn is_lokr_keys(w: &Weights) -> bool {
    wmeta::keys_contain_lokr(w.keys())
}

/// One module's third-party LoKr factors. Unlike the peft [`LokrFile`] (one global `(alpha, rank)`
/// from metadata), a third-party file carries **per-module** factor shapes + an optional per-module
/// `.alpha` scalar, so rank/alpha/scale are derived per module here.
#[derive(Default)]
pub struct ThirdPartyLokr {
    w1: Option<Array>,
    w1_a: Option<Array>,
    w1_b: Option<Array>,
    w2: Option<Array>,
    w2_a: Option<Array>,
    w2_b: Option<Array>,
    t2: Option<Array>,
    alpha: Option<f32>,
}

impl ThirdPartyLokr {
    /// The factorization rank (`lora_dim`). lycoris lays the factors out inconsistently, so derive in
    /// a fixed order from whichever decomposed factor is present: `lokr_w1_a` is `[shape0, dim]`
    /// (dim = `shape[1]`); the tucker `lokr_t2` is `[dim, dim, kH, kW]` (dim = `shape[0]`); the
    /// non-tucker `lokr_w2_a` is `[shape0, dim]` (dim = `shape[1]`). `None` when **both** factors are
    /// full — lycoris then forces `alpha = lora_dim` ⇒ scale 1, so rank is unused.
    ///
    /// F-010: a corrupt/adversarial third-party file can carry a 1-D/0-D factor (the shape index used
    /// to panic) or a zero dim (→ `scale = alpha/0` = inf/NaN baked into the reconstructed delta and
    /// silently installed). `Result` so both are rejected with a typed error instead; mirrors the
    /// F-002 LoRA rank-0 guard.
    fn rank(&self) -> Result<Option<f32>> {
        fn leading_dim(a: &Array, axis: usize, what: &str) -> Result<f32> {
            let shp = a.shape();
            if shp.len() < 2 {
                return Err(Error::Msg(format!(
                    "lokr adapter: {what} factor must be 2-D, got {}-D shape {:?}",
                    shp.len(),
                    shp
                )));
            }
            let r = shp[axis] as f32;
            if r <= 0.0 {
                return Err(Error::Msg(format!(
                    "lokr adapter: {what} factor has zero rank (zero leading dim)"
                )));
            }
            Ok(r)
        }
        if let Some(a) = &self.w1_a {
            return Ok(Some(leading_dim(a, 1, "lokr_w1_a")?));
        }
        if let Some(t) = &self.t2 {
            return Ok(Some(leading_dim(t, 0, "lokr_t2")?));
        }
        self.w2_a
            .as_ref()
            .map(|a| leading_dim(a, 1, "lokr_w2_a"))
            .transpose()
    }

    /// LyCORIS `scale`: `alpha / lora_dim` (alpha defaulting to `lora_dim`), EXCEPT both-full forces
    /// scale 1 (mirrors `LokrModule.__init__`: `if use_w1 and use_w2: alpha = lora_dim`).
    fn scale(&self) -> Result<f32> {
        match self.rank()? {
            None => Ok(1.0),
            Some(r) => Ok(self.alpha.unwrap_or(r) / r),
        }
    }

    /// Reconstruct this module's `ΔW` (lycoris per-module scale baked in) at `out_dtype`. `pub` so the
    /// merge-path providers (SDXL/Wan/LTX, sc-3671) reuse the exact derivation + reconstruction and
    /// differ only in how they install it (in-place merge vs forward residual).
    pub fn delta(&self, base_shape: &[i32], out_dtype: Dtype) -> Result<Array> {
        reconstruct_lokr_delta_scaled(
            self.scale()?,
            base_shape,
            self.w1.as_ref(),
            self.w1_a.as_ref(),
            self.w1_b.as_ref(),
            self.w2.as_ref(),
            self.t2.as_ref(),
            self.w2_a.as_ref(),
            self.w2_b.as_ref(),
            out_dtype,
        )
    }

    /// The **structured** (deferred, allocation-free) counterpart to [`delta`](Self::delta) for a
    /// packed base (sc-10050): build the small `[a,c]`/`[b,d]` Kronecker factors (the FULL scale
    /// `lycoris_per_module_scale · strength` baked into `w2`) via [`build_lokr_factors`], so the residual
    /// applies the vec-trick without ever materializing the `[out,in]` delta. The user `strength` is
    /// baked in here (the structured residual carries no separate scale). Returns `Ok(None)` for a
    /// **tucker/CP** `w2` (`lokr_t2`, conv-only) that has no 2-D matrix form — the caller then falls back
    /// (materialize on dense, or a clear error on packed). `pub` so the video providers (Wan/LTX) reuse
    /// the exact derivation.
    pub fn factors(
        &self,
        strength: f32,
        base_shape: &[i32],
        out_dtype: Dtype,
    ) -> Result<Option<LokrFactors>> {
        build_lokr_factors(
            self.scale()? * strength,
            base_shape,
            self.w1.as_ref(),
            self.w1_a.as_ref(),
            self.w1_b.as_ref(),
            self.w2.as_ref(),
            self.t2.as_ref(),
            self.w2_a.as_ref(),
            self.w2_b.as_ref(),
            out_dtype,
        )
    }
}

/// Group a third-party LoKr file's tensors by raw module key (the part before `.lokr_*`/`.alpha`).
/// The raw key is whatever the trainer wrote — a `<PREFIX>_<flattened.path>` (kohya/lycoris) or, more
/// rarely, a dotted path; resolution to the host's module map happens in [`apply_lokr_thirdparty`]
/// (or the merge-path providers' own tables, sc-3671).
pub fn parse_lokr_thirdparty(w: &Weights) -> Result<BTreeMap<String, ThirdPartyLokr>> {
    let mut groups: BTreeMap<String, ThirdPartyLokr> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        if let Some(raw) = key.strip_suffix(".alpha") {
            if let Some(a) = scalar_alpha(w.require(&key)?)? {
                groups.entry(raw.to_string()).or_default().alpha = Some(a);
            }
            continue;
        }
        for suffix in LOKR_TP_SUFFIXES {
            if let Some(raw) = key.strip_suffix(suffix) {
                let g = groups.entry(raw.to_string()).or_default();
                let t = w.require(&key)?.clone();
                match &suffix[1..] {
                    "lokr_w1" => g.w1 = Some(t),
                    "lokr_w1_a" => g.w1_a = Some(t),
                    "lokr_w1_b" => g.w1_b = Some(t),
                    "lokr_w2" => g.w2 = Some(t),
                    "lokr_w2_a" => g.w2_a = Some(t),
                    "lokr_w2_b" => g.w2_b = Some(t),
                    "lokr_t2" => g.t2 = Some(t),
                    _ => {}
                }
                break;
            }
        }
    }
    Ok(groups)
}

// `resolve_lokr_path` — resolve a third-party LoKr raw module key (`<PREFIX>_<stem>`, the diffusers
// path flattened with `.`→`_`) to a host dotted path, longest-stem-wins — is defined in gen-core
// (`weightsmeta`) and re-exported at the top of this module so the merge-path providers (sc-3671)
// resolve third-party keys against their own module tables.

/// Strip a leading [`COMMON_LORA_PREFIXES`] namespace (`transformer.` / `diffusion_model.`) from a
/// dotted third-party key, so a host whose [`AdaptableHost::adaptable_mut`] routes from the bare
/// module path still resolves it. The PEFT path passes a detected prefix to the appliers, but the
/// third-party LoKr/LoHa dotted-fallback path has no equivalent — without this, e.g. ostris
/// ai-toolkit's Krea-2 LoKr (`diffusion_model.blocks.N.attn.wq…`, sc-8395) matches no target even
/// though the Krea host already aliases `blocks`/`txtfusion`/`wq…`/`mlp` (sc-8185). Returns the key
/// unchanged when it carries no such prefix.
fn strip_common_lora_prefix(raw: &str) -> &str {
    for prefix in COMMON_LORA_PREFIXES {
        if let Some(rest) = raw.strip_prefix(prefix) {
            return rest;
        }
    }
    raw
}

/// Install a third-party LyCORIS **LoKr** file (LoHa is sc-3643) onto `host`. Reconstructs each
/// module's Kronecker delta from its per-module factors (full / low-rank / tucker) at the lycoris
/// scale and stacks it as an [`Adapter::Lokr`] residual at the user `scale` — the same install as
/// peft [`apply_lokr`], differing only in (a) per-module rank/alpha derivation and (b) resolving the
/// trainer's flattened key names to the host's dotted module map. Unresolved paths are surfaced in
/// `unmatched_paths`, never silently dropped.
pub fn apply_lokr_thirdparty(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
) -> Result<ApplyReport> {
    let table = kohya_table(&host.adaptable_paths());
    let groups = parse_lokr_thirdparty(w)?;
    // The host's LyCORIS dtype (bf16 by default, PARITY-BF16, sc-2609) — same as peft `apply_lokr`.
    let dtype = host.lycoris_dtype();
    let groups = groups.iter().map(|(raw, g)| {
        (
            raw.as_str(),
            move |base: &[i32]| g.delta(base, dtype),
            move |strength: f32, base: &[i32]| g.factors(strength, base, dtype),
        )
    });
    install_lycoris_groups(
        host,
        groups,
        scale,
        LycorisKeyResolution::Thirdparty(&table),
    )
}

// ---- Third-party LyCORIS LoHa (sc-3643) ----------------------------------------------------------

/// Third-party LoHa factor suffixes (the two Hadamard low-rank pairs + optional tucker factors).
/// (gen-core.)
use gen_core::weightsmeta::LOHA_TP_SUFFIXES;

/// `true` if any key is a LoHa factor (`*.hada_w…`) — how a third-party LyCORIS LoHa (kohya /
/// ai-toolkit / lycoris-lib) is recognized. Mutually exclusive with [`is_lokr_keys`] (`lokr_*`).
pub fn is_loha_keys(w: &Weights) -> bool {
    wmeta::keys_contain_loha(w.keys())
}

/// One module's third-party LoHa factors — two low-rank Hadamard pairs, optional tucker `t1`/`t2`,
/// and an optional per-module `.alpha` (rank/scale derived per module, like [`ThirdPartyLokr`]).
#[derive(Default)]
pub struct ThirdPartyLoha {
    w1_a: Option<Array>,
    w1_b: Option<Array>,
    w2_a: Option<Array>,
    w2_b: Option<Array>,
    t1: Option<Array>,
    t2: Option<Array>,
    alpha: Option<f32>,
}

impl ThirdPartyLoha {
    /// rank (`lora_dim`) = `hada_w1_b.shape[0]` (lycoris stores `hada_w1_b` as `[lora_dim, …]` in
    /// both the tucker and non-tucker layouts).
    ///
    /// F-010: a 1-D/0-D factor panics on the shape index; a zero leading dim → `scale = alpha/0` =
    /// inf/NaN baked into the reconstructed delta. `Result` so both are rejected (F-002 LoRA twin).
    fn rank(&self) -> Result<Option<f32>> {
        Ok(match &self.w1_b {
            None => None,
            Some(b) => {
                let shp = b.shape();
                if shp.len() < 2 {
                    return Err(Error::Msg(format!(
                        "loha adapter: hada_w1_b must be 2-D, got {}-D shape {:?}",
                        shp.len(),
                        shp
                    )));
                }
                let r = shp[0] as f32;
                if r <= 0.0 {
                    return Err(Error::Msg(
                        "loha adapter: hada_w1_b has zero rank (zero leading dim)".into(),
                    ));
                }
                Some(r)
            }
        })
    }

    /// LyCORIS `scale = alpha / lora_dim` (alpha defaulting to `lora_dim`). LoHa is always decomposed
    /// (no both-full case), so — unlike LoKr — there is no forced-1 branch.
    fn scale(&self) -> Result<f32> {
        match self.rank()? {
            None => Ok(1.0),
            Some(r) => Ok(self.alpha.unwrap_or(r) / r),
        }
    }

    /// Reconstruct this module's LoHa `ΔW` (lycoris per-module scale baked in) at `out_dtype`. `pub`
    /// for the merge-path providers (sc-3671).
    pub fn delta(&self, base_shape: &[i32], out_dtype: Dtype) -> Result<Array> {
        let (w1_a, w1_b, w2_a, w2_b) = match (&self.w1_a, &self.w1_b, &self.w2_a, &self.w2_b) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return Err("LoHa: a hada_w1/w2 a/b factor is missing".into()),
        };
        reconstruct_loha_delta(
            self.scale()?,
            base_shape,
            w1_a,
            w1_b,
            w2_a,
            w2_b,
            self.t1.as_ref(),
            self.t2.as_ref(),
            out_dtype,
        )
    }
}

/// Group a third-party LoHa file's tensors by raw module key (the part before `.hada_*`/`.alpha`).
/// `pub` for the merge-path providers (sc-3671).
pub fn parse_loha_thirdparty(w: &Weights) -> Result<BTreeMap<String, ThirdPartyLoha>> {
    let mut groups: BTreeMap<String, ThirdPartyLoha> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        if let Some(raw) = key.strip_suffix(".alpha") {
            if let Some(a) = scalar_alpha(w.require(&key)?)? {
                groups.entry(raw.to_string()).or_default().alpha = Some(a);
            }
            continue;
        }
        for suffix in LOHA_TP_SUFFIXES {
            if let Some(raw) = key.strip_suffix(suffix) {
                let g = groups.entry(raw.to_string()).or_default();
                let t = w.require(&key)?.clone();
                match &suffix[1..] {
                    "hada_w1_a" => g.w1_a = Some(t),
                    "hada_w1_b" => g.w1_b = Some(t),
                    "hada_w2_a" => g.w2_a = Some(t),
                    "hada_w2_b" => g.w2_b = Some(t),
                    "hada_t1" => g.t1 = Some(t),
                    "hada_t2" => g.t2 = Some(t),
                    _ => {}
                }
                break;
            }
        }
    }
    Ok(groups)
}

/// Install a third-party LyCORIS **LoHa** file onto `host`. Reconstructs each module's Hadamard delta
/// and stacks it as an [`Adapter::Lokr`] residual (the reconstructed `ΔW` applies through the same
/// `scale · x·ΔWᵀ` forward path — no distinct adapter variant needed). Module-key resolution
/// (flattened-prefixed → dotted via `kohya_table`) and unmatched-path surfacing mirror
/// [`apply_lokr_thirdparty`].
///
/// The Hadamard product has **no deferred (unmaterialized) form** — unlike LoKr's Kronecker vec-trick —
/// so the factors builder below is unconditionally `None`: there is nothing to build. On a packed Q4/Q8
/// base, `install_lycoris_groups` therefore falls back to the materialized `[out,in]` delta. That is
/// CORRECT (the residual `scale · x·ΔWᵀ` never reads the base, so it rides over `quantized_matmul`
/// unchanged) but holds a dense tensor per target. Refusing the trade instead — sc-10051's call for
/// quantized Wan — would break every third-party LoHa on the DEFAULT tier of every MLX model, because
/// adapters install *after* `.quantize()` and the worker defaults `spec.quantize` to `Some(Q8)`.
/// Whether a packed base should refuse a multi-GB delta is sc-10678, not this path's call to make.
pub fn apply_loha_thirdparty(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
) -> Result<ApplyReport> {
    let table = kohya_table(&host.adaptable_paths());
    let groups = parse_loha_thirdparty(w)?;
    // Same residual dtype as the LoKr paths (the host's, bf16 by default — PARITY-BF16, sc-2609).
    let dtype = host.lycoris_dtype();
    let groups = groups.iter().map(|(raw, g)| {
        (
            raw.as_str(),
            move |base: &[i32]| g.delta(base, dtype),
            // Hadamard has no Kronecker structure to defer — never deferrable, on any base.
            |_strength: f32, _base: &[i32]| -> Result<Option<LokrFactors>> { Ok(None) },
        )
    });
    install_lycoris_groups(
        host,
        groups,
        scale,
        LycorisKeyResolution::Thirdparty(&table),
    )
}

/// Install a PEFT/diffusers-format LoRA file onto `host`. The down/up factors carry the file's
/// namespace prefix on a **dotted** module path, in either of two interchangeable spellings:
/// - PEFT: `‹prefix›‹path›.lora_A.weight` / `.lora_B.weight`;
/// - diffusers/ComfyUI/ai-toolkit (e.g. the lightx2v Qwen-Image-Lightning LoRAs, sc-2909):
///   `‹prefix›‹path›.lora_down.weight` / `.lora_up.weight` — `lora_down`==`lora_A`, `lora_up`==`lora_B`
///   (identical shapes), differing from the kohya format only in that the path stays dotted (no
///   `lora_unet_` flattening), so it routes here rather than to [`apply_lora_kohya`].
///
/// Both store the down factor as `[r, in]` and the up factor as `[out, r]`; we transpose to the
/// residual form `x·A·B` (`A: [in, r]`, `B: [r, out]`) and fold `alpha/rank` into `B`, matching the
/// fork. `‹prefix›‹path›.alpha` is optional (and may be bare — see below). `strip_prefix` removes a
/// leading namespace such as `"base_model.model."` or `"transformer."`.
pub fn apply_lora_peft(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    strip_prefix: Option<&str>,
) -> Result<ApplyReport> {
    let prefix = strip_prefix.unwrap_or("");
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        // The down/up factors always carry the file's namespace prefix. `lora_A`/`lora_B` (PEFT) and
        // `lora_down`/`lora_up` (diffusers/ComfyUI) are interchangeable spellings of the same role,
        // and a PEFT export that kept its adapter name writes `lora_A.default` (sc-24158 — the
        // candle `install_dotted_adapters` and the kohya loader below already accept it).
        if let Some(rest) = key.strip_prefix(prefix) {
            if let Some(path) = rest
                .strip_suffix(".lora_A.weight")
                .or_else(|| rest.strip_suffix(".lora_A.default.weight"))
                .or_else(|| rest.strip_suffix(".lora_down.weight"))
            {
                let slot = &mut groups.entry(path.to_string()).or_default().a;
                if slot.is_some() {
                    // Two spellings of one factor (`lora_A` beside `lora_A.default` / `lora_down`):
                    // refuse rather than keep whichever key sorts last (sc-24158).
                    return Err(format!(
                        "LoRA down/A factor for `{path}` is spelled twice (second: `{key}`); \
                         refusing an ambiguous apply"
                    )
                    .into());
                }
                *slot = Some(w.require(&key)?.clone());
                continue;
            }
            if let Some(path) = rest
                .strip_suffix(".lora_B.weight")
                .or_else(|| rest.strip_suffix(".lora_B.default.weight"))
                .or_else(|| rest.strip_suffix(".lora_up.weight"))
            {
                let slot = &mut groups.entry(path.to_string()).or_default().b;
                if slot.is_some() {
                    return Err(format!(
                        "LoRA up/B factor for `{path}` is spelled twice (second: `{key}`); \
                         refusing an ambiguous apply"
                    )
                    .into());
                }
                *slot = Some(w.require(&key)?.clone());
                continue;
            }
        }
        // `alpha` may be prefixed (`<prefix><path>.alpha`) OR bare (`<path>.alpha`): some trainers
        // pair prefixed `lora_A/B` with a bare `alpha` — notably the fork's `QwenLoRAMapping`, whose
        // alpha patterns are bare-only. Resolve to the same `<path>` either way (rather than
        // stripping the A/B prefix off the alpha key and dropping a bare one) so the `alpha/rank`
        // fold is kept; a prefixed and a bare alpha that *disagree* for one path is a hard error (no
        // silent pick). Without this, a prefixed-A/B + bare-alpha file applied at the wrong
        // (unscaled) strength while reporting success (sc-2528 adversarial review).
        if let Some(path) = key
            .strip_prefix(prefix)
            .and_then(|r| r.strip_suffix(".alpha"))
            .or_else(|| key.strip_suffix(".alpha"))
        {
            if let Some(new) = scalar_alpha(w.require(&key)?)? {
                let slot = &mut groups.entry(path.to_string()).or_default().alpha;
                match *slot {
                    Some(existing) if existing != new => {
                        return Err(format!(
                            "LoRA alpha conflict for `{path}`: {existing} vs {new} \
                             (prefixed and bare alpha keys disagree)"
                        )
                        .into());
                    }
                    _ => *slot = Some(new),
                }
            }
        }
    }

    // PEFT/diffusers `save_lora_adapter` files carry no per-target `.alpha` tensor — `lora_alpha`/`r`
    // (+ per-module overrides) live in the `lora_adapter_metadata` header blob (sc-5513). `None` for a
    // file without the blob, in which case the per-target `.alpha` or the factor rank is used exactly
    // as before. (kohya / BFL loaders ship a `.alpha` tensor and pass `None` here.)
    let cfg = wmeta::LoraAdapterMeta::from_metadata(w.metadata(wmeta::LORA_ADAPTER_METADATA_KEY));
    install_lora_groups(host, groups, scale, cfg.as_ref())
}

/// Install grouped `(down=A, up=B, alpha)` LoRA factors onto `host`, one residual per resolved module
/// path. Shared by the PEFT/diffusers loader ([`apply_lora_peft`]) and the kohya loader
/// ([`apply_lora_kohya`]): both conventions agree on the math (`A: [r,in]`, `B: [out,r]`, transpose to
/// the residual form `x·A·B`, fold `alpha/rank` into `B`) and differ only in how keys map to `path`.
/// A path with a missing `down` or `up` half is skipped (its partner targeted a non-LoRA key);
/// a path that resolves to no module is surfaced in `unmatched_paths`, never silently dropped.
fn install_lora_groups(
    host: &mut impl AdaptableHost,
    groups: BTreeMap<String, LoraParts>,
    scale: f32,
    meta: Option<&wmeta::LoraAdapterMeta>,
) -> Result<ApplyReport> {
    let mut report = ApplyReport::default();
    for (path, parts) in groups {
        let (Some(a_raw), Some(b_raw)) = (parts.a, parts.b) else {
            continue;
        };
        let parents: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&parents) {
            Some(lin) => {
                let a = a_raw.t(); // [r, in] -> [in, r]
                let mut b = b_raw.t(); // [out, r] -> [r, out]
                                       // A Linear LoRA's factors are 2-D; a malformed 1-D/scalar factor would panic on the
                                       // `a.shape()[1]` rank read below. Reject it with a typed error up front (F-034).
                if a.shape().len() != 2 || b.shape().len() != 2 {
                    return Err(Error::Msg(format!(
                        "lora adapter at '{path}' has non-2-D factors (down {:?}, up {:?})",
                        a_raw.shape(),
                        b_raw.shape()
                    )));
                }
                // Effective scaling. Precedence: per-target `.alpha` tensor (kohya / SceneWorks
                // trainer / BFL) → the PEFT/diffusers `lora_adapter_metadata` blob's
                // `alpha_pattern`/`lora_alpha` (sc-5513 — that format ships NO `.alpha` tensor) → no
                // fold (the pre-existing `alpha == rank ⇒ scale 1.0` default). The denominator honors
                // the blob `r`/`rank_pattern` when given (always `> 0`), else the factor's stored
                // leading dim (which equals it for a well-formed PEFT file).
                let (cfg_alpha, cfg_rank) = meta.map_or((None, None), |m| m.effective(&path));
                if let Some(alpha) = parts.alpha.or(cfg_alpha) {
                    let factor_rank = a.shape()[1] as f32; // r
                    if factor_rank == 0.0 {
                        // Zero rank (empty/malformed factor) → non-finite alpha/rank → a NaN residual
                        // folded into the linear, silently corrupting inference. Reject the adapter
                        // instead of installing it (sc-5252/F-002).
                        return Err(Error::Msg(format!(
                            "lora adapter at '{path}' has zero rank (empty down/up factor)"
                        )));
                    }
                    let rank = cfg_rank.unwrap_or(factor_rank);
                    b = b.multiply(Array::from_slice(&[alpha / rank], &[1]))?;
                }
                lin.push(Adapter::Lora { a, b, scale });
                report.applied += 1;
            }
            None => report.unmatched_paths.push(path),
        }
    }
    Ok(report)
}

#[derive(Default)]
struct LoraParts {
    a: Option<Array>,
    b: Option<Array>,
    alpha: Option<f32>,
}

// `KOHYA_PREFIX` (`lora_unet_`) is defined in gen-core and re-exported at the top of this module.

/// kohya factor suffixes mapped to a [`LoraParts`] role. `lora_down`==PEFT `lora_A`,
/// `lora_up`==PEFT `lora_B`; the optional `.default` infix is the peft-export form some kohya
/// converters emit. Order is irrelevant (exact-suffix match).
const KOHYA_SUFFIXES: [(&str, KohyaRole); 5] = [
    (".lora_down.weight", KohyaRole::Down),
    (".lora_up.weight", KohyaRole::Up),
    (".lora_down.default.weight", KohyaRole::Down),
    (".lora_up.default.weight", KohyaRole::Up),
    (".alpha", KohyaRole::Alpha),
];

#[derive(Clone, Copy)]
enum KohyaRole {
    Down,
    Up,
    Alpha,
}

/// `true` if `w` is a kohya-format LoRA — any key carries the `lora_unet_` prefix. (kohya files are
/// the only convention that flattens the module path; PEFT/diffusers keep dots, LoKr is bare.)
pub fn is_kohya(w: &Weights) -> bool {
    wmeta::keys_are_kohya(w.keys())
}

/// Build the kohya `flattened-stem → dotted-path` lookup from a host's routable target paths
/// (`AdaptableHost::adaptable_paths`). The stem is the dotted path with `.`→`_` (the kohya
/// flattening), WITHOUT the `lora_unet_` prefix. Mirrors the SDXL matcher (sc-2639) and the fork's
/// explicit `lora_unet_…` patterns, generalized over any [`AdaptableHost`].
fn kohya_table(paths: &[String]) -> BTreeMap<String, String> {
    wmeta::kohya_table(paths)
}

/// Install a kohya-format LoRA (`lora_unet_<flattened path>.lora_down/up.weight` + optional `.alpha`)
/// onto `host`. The flattened stem is resolved against `table` (built from
/// [`AdaptableHost::adaptable_paths`]) — blind `_`→`.` splitting is impossible because module names
/// contain underscores (`to_out.0`, `feed_forward.w1`, `img_mlp.net.0.proj`). Resolved factors are
/// installed through the same `install_lora_groups` path as PEFT, so a kohya file produces the
/// identical adapter to the equivalent PEFT file.
///
/// `lora_unet_` keys whose stem is NOT in the table (off-surface) are surfaced in `unmatched_paths`
/// so the strict policy fails loudly rather than silently dropping them. The BFL fused→split kohya
/// form (`lora_unet_double_blocks_*`, sc-2743) is routed to [`apply_lora_bfl`] *before* this loader
/// for a host that exposes [`AdaptableHost::bfl_targets`]; reaching here it has no table entry and is
/// likewise surfaced. Keys without the `lora_unet_` prefix (e.g. a bundled text-encoder `lora_te_…`)
/// are not denoiser targets and are ignored, matching the PEFT loader's treatment of out-of-namespace
/// keys.
pub fn apply_lora_kohya(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    table: &BTreeMap<String, String>,
) -> Result<ApplyReport> {
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    let mut unresolved: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some(rem) = key.strip_prefix(KOHYA_PREFIX) else {
            continue; // not a denoiser kohya key (e.g. text-encoder `lora_te_…`) — ignore.
        };
        let Some((stem, role)) = KOHYA_SUFFIXES
            .iter()
            .find_map(|(suf, role)| rem.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue; // a `lora_unet_` key with an unrecognized suffix — ignore.
        };
        let Some(path) = table.get(stem) else {
            unresolved.insert(stem.to_string());
            continue;
        };
        let parts = groups.entry(path.clone()).or_default();
        match role {
            KohyaRole::Down => parts.a = Some(w.require(&key)?.clone()),
            KohyaRole::Up => parts.b = Some(w.require(&key)?.clone()),
            KohyaRole::Alpha => parts.alpha = scalar_alpha(w.require(&key)?)?,
        }
    }

    // kohya / BFL files carry a per-target `.alpha` tensor, not the `lora_adapter_metadata` blob — no
    // blob to honor here (sc-5513).
    let mut report = install_lora_groups(host, groups, scale, None)?;
    report.unmatched_paths.extend(unresolved);
    Ok(report)
}

// ---- BFL / ComfyUI fused→split LoRA (sc-2743) ----------------------------------------------------

/// A row-slice over a raw LoRA factor (applied BEFORE the `[r,in]`/`[out,r]`→residual transpose),
/// porting the fork's `LoraTransforms` byte-for-byte (sc-2743). The BFL/ComfyUI checkpoints store a
/// block's q/k/v (and, for FLUX.1, the qkv+mlp) concatenated along dim-0 of a single *fused* linear;
/// each diffusers split target slices out its own rows. Indices/divisibility match `LoraTransforms`
/// exactly (verified against the fork venv).
#[derive(Clone, Debug)]
pub enum LoraRowSlice {
    /// Chunk `index` of `n` equal dim-0 chunks (`chunk = shape[0] / n`), ALWAYS sliced — the fork's
    /// `_split_qkv_up` (the up factor `[n·out, r]` → `[out, r]`). `n=3` for qkv.
    Chunk { n: i32, index: i32 },
    /// Chunk `index` of `n` equal dim-0 chunks IFF `shape[0] % n == 0`, else the whole tensor — the
    /// fork's `_split_qkv_down`/`_split_qkv_mlp_down` (the down factor is *shared* across q/k/v when
    /// the rank isn't divisible by `n`, which is the usual fused-qkv LoRA, and sliced when it is).
    ChunkIfDivisible { n: i32, index: i32 },
    /// The dim-0 slice `[Σdims[..index] .. Σdims[..=index]]` — the fork's `_split_qkv_mlp_up` with
    /// config-derived `dims` (FLUX.1 `linear1` = `[q,k,v,mlp]`, e.g. `[3072,3072,3072,12288]`). FLUX.2
    /// keeps qkv+mlp fused (`to_qkv_mlp_proj`) so it never uses this; FLUX.1 (sc-2657) will.
    Dims { dims: Vec<i32>, index: i32 },
}

impl LoraRowSlice {
    fn apply(&self, t: &Array) -> Result<Array> {
        let rows = t.shape()[0];
        // `n`/`index`/`dims` come from a static `bfl_targets()` table built in code, not the file —
        // but a miswritten entry (n<=0, index out of [0,n), or an out-of-range `dims` index) would
        // divide-by-zero / index out of bounds and panic. Reject it with a typed error (F-033).
        let (start, end) = match self {
            LoraRowSlice::Chunk { n, index } => {
                if *n <= 0 || *index < 0 || *index >= *n {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::Chunk: invalid chunk spec (n={n}, index={index})"
                    )));
                }
                // F-060: `Chunk` ALWAYS slices (vs `ChunkIfDivisible`'s fall-through), so a non-divisible
                // `rows` silently mis-slices the fused qkv delta (truncating division drops the tail).
                // The fork's `_split_qkv_up` is only ever emitted for an exactly-divisible fused linear,
                // so a non-divisible shape means a mis-converted/mismatched adapter — reject it.
                if rows % n != 0 {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::Chunk: rows ({rows}) not divisible by n ({n}); \
                         the fused factor is mis-shaped for this split"
                    )));
                }
                let chunk = rows / n;
                (index * chunk, (index + 1) * chunk)
            }
            LoraRowSlice::ChunkIfDivisible { n, index } => {
                if *n <= 0 || *index < 0 || *index >= *n {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::ChunkIfDivisible: invalid chunk spec (n={n}, index={index})"
                    )));
                }
                if rows % n != 0 {
                    return Ok(t.clone());
                }
                let chunk = rows / n;
                (index * chunk, (index + 1) * chunk)
            }
            LoraRowSlice::Dims { dims, index } => {
                let i = *index as usize;
                if *index < 0 || i >= dims.len() {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::Dims: index {index} out of range for {} dims",
                        dims.len()
                    )));
                }
                let start: i32 = dims[..i].iter().sum();
                (start, start + dims[i])
            }
        };
        // `t[start:end, :]` — byte-identical to the fork's slicing.
        Ok(t.try_index((start..end, ..))?)
    }
}

/// One BFL/ComfyUI adapter target: a set of source key spellings (across the `lora_unet_` /
/// `diffusion_model.` / `base_model.model.` prefix conventions) mapping to a diffusers module
/// `target_path`, with an optional [`LoraRowSlice`] on the up/down factor. A *fused* source (BFL
/// `…img_attn.qkv`) is named by SEVERAL `BflTarget`s — one per split destination (`to_q`/`to_k`/`to_v`)
/// — that share its key spellings but slice different rows; the loader fans the one source tensor into
/// all of them. A plain rename (BFL `img_in` → `x_embedder`) is a `BflTarget` with no slice. Mirrors a
/// fork `LoRATarget` restricted to its BFL patterns + up/down transforms.
#[derive(Clone, Debug)]
pub struct BflTarget {
    /// Diffusers module path that [`AdaptableHost::adaptable_mut`] resolves (concrete, no `{block}`).
    pub target_path: String,
    /// Source up-factor (`lora_up`/`lora_B`) key spellings.
    pub up_keys: Vec<String>,
    /// Source down-factor (`lora_down`/`lora_A`) key spellings.
    pub down_keys: Vec<String>,
    /// Source `alpha` key spellings (no transform, no transpose).
    pub alpha_keys: Vec<String>,
    /// Row-slice applied to the up factor (the qkv split). `None` for a plain rename.
    pub up_slice: Option<LoraRowSlice>,
    /// Row-slice applied to the down factor (shared-or-split). `None` for a plain rename.
    pub down_slice: Option<LoraRowSlice>,
}

/// One contribution of a source key to a target: which target/role it feeds and how to slice it.
struct BflSlot {
    target: String,
    role: KohyaRole,
    slice: Option<LoraRowSlice>,
}

/// Invert `targets` into `source_key → [contribution, …]`. One fused source key (e.g. a qkv
/// `lora_up`) contributes to multiple targets (q/k/v) with different slices, so the value is a list.
fn bfl_index(targets: &[BflTarget]) -> BTreeMap<String, Vec<BflSlot>> {
    let mut index: BTreeMap<String, Vec<BflSlot>> = BTreeMap::new();
    let mut push = |key: &str, target: &str, role: KohyaRole, slice: Option<LoraRowSlice>| {
        index.entry(key.to_string()).or_default().push(BflSlot {
            target: target.to_string(),
            role,
            slice,
        });
    };
    for t in targets {
        for k in &t.up_keys {
            push(k, &t.target_path, KohyaRole::Up, t.up_slice.clone());
        }
        for k in &t.down_keys {
            push(k, &t.target_path, KohyaRole::Down, t.down_slice.clone());
        }
        for k in &t.alpha_keys {
            push(k, &t.target_path, KohyaRole::Alpha, None);
        }
    }
    index
}

/// `true` if any key in `w` is a known BFL source key for `targets` — i.e. the file uses the BFL /
/// ComfyUI naming (`double_blocks`/`single_blocks`/`img_in`/… across the three prefix conventions),
/// which the diffusers/peft/standard-kohya paths cannot resolve. Precise: a standard diffusers/peft
/// or standard-kohya file shares none of these spellings, so it is never misrouted here.
pub fn is_bfl(w: &Weights, targets: &[BflTarget]) -> bool {
    if targets.is_empty() {
        return false;
    }
    let index = bfl_index(targets);
    w.keys().any(|k| index.contains_key(k))
}

/// Recognized LoRA factor suffixes — a key ending in one of these is adapter-shaped (vs. a base
/// weight or some bundled extra). Used to surface BFL-named keys that resolve to no target.
const LORA_FACTOR_SUFFIXES: [&str; 5] = [
    ".lora_up.weight",
    ".lora_down.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    ".alpha",
];

/// Install a BFL / ComfyUI fused→split LoRA onto `host` (sc-2743). Each file key is matched against
/// the inverted [`BflTarget`] index; a matched *fused* source is row-sliced per destination and fanned
/// into the diffusers split targets (`…img_attn.qkv` → `attn.to_q/to_k/to_v`), a plain rename is copied
/// through. Resolved factors feed the same `install_lora_groups` path as PEFT/kohya (transpose +
/// `alpha/rank` fold), so a BFL file yields the byte-identical adapter to the equivalent diffusers
/// split-target LoRA.
///
/// An adapter-shaped key that matches NO target — an off-surface BFL key (e.g. a block out of range)
/// — is surfaced in `unmatched_paths` (loud, never silently dropped). A bundled text-encoder key
/// (`lora_te_…`/`text_encoder.…`) is not a denoiser target and is ignored, matching the PEFT/kohya
/// loaders' treatment of out-of-namespace keys.
pub fn apply_lora_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    targets: &[BflTarget],
) -> Result<ApplyReport> {
    let index = bfl_index(targets);
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    let mut unresolved: BTreeSet<String> = BTreeSet::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some(slots) = index.get(&key) else {
            // Not a BFL source key: surface it if it's an adapter-shaped denoiser key (off-surface),
            // ignore a bundled text-encoder adapter (`lora_te_…`/`…text_encoder.…`).
            let adapter_shaped = LORA_FACTOR_SUFFIXES.iter().any(|s| key.ends_with(s));
            let foreign_te = key.starts_with("lora_te") || key.contains("text_encoder");
            if adapter_shaped && !foreign_te {
                unresolved.insert(key);
            }
            continue;
        };
        let v = w.require(&key)?;
        for slot in slots {
            let parts = groups.entry(slot.target.clone()).or_default();
            match slot.role {
                KohyaRole::Down => {
                    parts.a = Some(match &slot.slice {
                        Some(s) => s.apply(v)?,
                        None => v.clone(),
                    });
                }
                KohyaRole::Up => {
                    parts.b = Some(match &slot.slice {
                        Some(s) => s.apply(v)?,
                        None => v.clone(),
                    });
                }
                KohyaRole::Alpha => {
                    // Detect a genuine conflict rather than nondeterministic last-wins (F-014,
                    // sc-11129): a file with two `.alpha` spellings mapping to one target resolved by
                    // HashMap visit order, installing at a nondeterministic scale across runs while
                    // reporting success. Mirror `apply_lora_peft`'s hard "alpha conflict" error. A
                    // repeated identical value (the same key visited twice) is not a conflict.
                    if let Some(new) = scalar_alpha(v)? {
                        match parts.alpha {
                            Some(existing) if existing != new => {
                                return Err(Error::Msg(format!(
                                    "LoRA alpha conflict for `{}`: {existing} vs {new} \
                                     (duplicate alpha keys for one BFL target)",
                                    slot.target
                                )));
                            }
                            _ => parts.alpha = Some(new),
                        }
                    }
                }
            }
        }
    }

    // kohya / BFL files carry a per-target `.alpha` tensor, not the `lora_adapter_metadata` blob — no
    // blob to honor here (sc-5513).
    let mut report = install_lora_groups(host, groups, scale, None)?;
    report.unmatched_paths.extend(unresolved);
    Ok(report)
}

// ---- BFL / ComfyUI fused→split LyCORIS LoKr / LoHa (sc-8345) --------------------------------------

/// One BFL/ComfyUI LyCORIS destination: a split diffusers `target_path` plus the row-slice that carves
/// this destination's rows out of the *fused* reconstructed `[out,in]` delta (`None` = a plain rename,
/// the whole delta). Unlike [`BflTarget`] — which slices the raw LoRA up/down *factors* before the
/// residual transpose — a LyCORIS adapter reconstructs the full Kronecker/Hadamard delta first and
/// then slices its rows, so only the out-dim (`up`) slice is relevant here.
#[derive(Clone, Debug)]
struct BflLycorisTarget {
    target_path: String,
    out_slice: Option<LoraRowSlice>,
}

/// Build a `prefixed-module-path → [split target, …]` map from a host's [`BflTarget`] list, keyed by
/// the module path *as a LyCORIS file spells it* — every `up_key` minus its `.lora_up.weight` /
/// `.lora_B.weight` factor suffix, i.e. the `lora_unet_<flat>` / `diffusion_model.<dotted>` /
/// `base_model.model.<dotted>` BFL spellings. A fused qkv source maps to its three split targets (each
/// carrying its own out-dim slice); a rename maps to one target with no slice. The LyCORIS analog of
/// [`bfl_index`] (which keys by full factor key + role because LoRA slices factors, not the delta).
fn bfl_lycoris_module_map(targets: &[BflTarget]) -> BTreeMap<String, Vec<BflLycorisTarget>> {
    let mut map: BTreeMap<String, Vec<BflLycorisTarget>> = BTreeMap::new();
    for t in targets {
        for up in &t.up_keys {
            let Some(module) = up
                .strip_suffix(".lora_up.weight")
                .or_else(|| up.strip_suffix(".lora_B.weight"))
            else {
                continue;
            };
            let entry = map.entry(module.to_string()).or_default();
            // The same module key appears under both the `lora_up` and `lora_B` spellings; keep one
            // entry per destination.
            if entry.iter().all(|e| e.target_path != t.target_path) {
                entry.push(BflLycorisTarget {
                    target_path: t.target_path.clone(),
                    out_slice: t.up_slice.clone(),
                });
            }
        }
    }
    map
}

/// The LyCORIS module path a key belongs to — `key` minus a trailing `.lokr_*` / `.hada_*` / `.alpha`
/// factor suffix — or `None` if `key` is not a LyCORIS factor key.
fn lycoris_module_of(key: &str) -> Option<&str> {
    if let Some(module) = key.strip_suffix(".alpha") {
        return Some(module);
    }
    LOKR_TP_SUFFIXES
        .iter()
        .chain(LOHA_TP_SUFFIXES.iter())
        .find_map(|suffix| key.strip_suffix(suffix))
}

/// `true` if any LyCORIS factor key in `w` names a module in the BFL map — i.e. the file uses the
/// BFL/ComfyUI fused naming a host's `bfl_targets()` covers. A diffusers/bare/standard-kohya LyCORIS
/// file (modules like `transformer.…` or a `lora_unet_<diffusers-flat>` that resolves through the
/// kohya table) shares none of these spellings, so it is never misrouted here and stays on the
/// existing third-party/peft path. Empty map (a host with no BFL surface — every engine but FLUX.1/
/// FLUX.2) ⇒ always `false`.
fn is_bfl_lycoris(w: &Weights, map: &BTreeMap<String, Vec<BflLycorisTarget>>) -> bool {
    if map.is_empty() {
        return false;
    }
    w.keys()
        .any(|k| lycoris_module_of(k).is_some_and(|module| map.contains_key(module)))
}

/// Install grouped LyCORIS deltas onto a host's BFL fused→split targets (sc-8345). For each source
/// module the `reconstruct` closure rebuilds the FULL fused `[out,in]` delta (the host-fused qkv shape,
/// with the format's `alpha/rank` scale already baked in); each destination then row-slices its share
/// out of that delta and stacks it as an [`Adapter::Lokr`] residual at the user `scale`. The fused
/// `out` is the SUM of the destinations' out dims (3·inner for a qkv split, Σdims for FLUX.1's qkv+mlp,
/// the target's own out for a rename), so it is derived from the resolved targets rather than parsed
/// from the slice. A module absent from `map` is surfaced in `unmatched_paths`, never silently dropped.
fn install_bfl_lycoris<I, F>(
    host: &mut impl AdaptableHost,
    map: &BTreeMap<String, Vec<BflLycorisTarget>>,
    groups: I,
    scale: f32,
) -> Result<ApplyReport>
where
    I: IntoIterator<Item = (String, F)>,
    F: FnOnce(&[i32]) -> Result<Array>,
{
    let mut report = ApplyReport::default();

    // One resolved BFL module: its split destinations, the fused `[Σout, in]` shape to reconstruct at,
    // and the (deferred) reconstruction closure — held so pass 1 can budget-check the total before any
    // pass-2 allocation.
    struct BflPlan<F> {
        targets: Vec<BflLycorisTarget>,
        fused_shape: [i32; 2],
        reconstruct: F,
    }

    // Pass 1 — resolve each module's fused shape and sum the bytes the reconstruction will materialize
    // (F-010, sc-11129). This BFL fused→split path has NO deferred/allocation-free form: each
    // destination row-slices the fully reconstructed `[Σout,in]` bf16 delta, so every install
    // materializes. Its plain-path twin (`install_lycoris_groups`) already applies the sc-10678 memory
    // guard; this sibling did not, so a full-coverage BFL LoKr on FLUX could hold multiple GB of
    // resident deltas with no up-front refusal and OOM the worker mid-load via an uncatchable SIGKILL.
    let mut plans: Vec<BflPlan<F>> = Vec::new();
    let mut projected_materialize: usize = 0;
    let dtype = host.lycoris_dtype();
    for (module, reconstruct) in groups {
        let Some(targets) = map.get(&module) else {
            report.unmatched_paths.push(module);
            continue;
        };
        // Fused reconstruction shape: rows = Σ destination out dims, cols = the shared in dim. Every
        // fused destination MUST share one in_dim — the reconstruction uses the first destination's
        // input dim for all rows, so a miswritten `bfl_targets()` entry fusing mismatched in-dims would
        // otherwise surface as an opaque kron/reshape error far from the cause (F-013, sc-11129).
        let mut fused_out = 0i32;
        let mut in_dim: Option<i32> = None;
        let mut resolvable = true;
        for tgt in targets {
            let parts: Vec<&str> = tgt.target_path.split('.').collect();
            // SC-18319 — shape-only resolution, so the PROBE half. See `install_lycoris_groups`.
            match host.adaptable_facts(&parts).map(|facts| facts.base_shape) {
                Some(shape) if shape.len() == 2 => {
                    fused_out += shape[0];
                    match in_dim {
                        None => in_dim = Some(shape[1]),
                        Some(existing) if existing != shape[1] => {
                            return Err(Error::Msg(format!(
                                "BFL LyCORIS module '{module}': fused destination '{}' has in_dim {} \
                                 but a prior destination in the same fused group had in_dim {existing} \
                                 — a fused qkv/mlp split must share one input dim",
                                tgt.target_path,
                                shape[1],
                            )));
                        }
                        Some(_) => {}
                    }
                }
                _ => {
                    resolvable = false;
                    break;
                }
            }
        }
        let (Some(in_dim), true) = (in_dim, resolvable) else {
            // A destination that didn't resolve (or a non-2-D linear) — surface the module rather than
            // install a partial, mis-shaped delta.
            report.unmatched_paths.push(module);
            continue;
        };
        projected_materialize += projected_delta_bytes(&[fused_out, in_dim], dtype);
        plans.push(BflPlan {
            targets: targets.clone(),
            fused_shape: [fused_out, in_dim],
            reconstruct,
        });
    }

    // sc-10678 / F-010 — pre-flight memory guard, the same policy as `install_lycoris_groups`.
    if projected_materialize > 0 {
        let active = get_active_memory();
        let limit = get_memory_limit();
        if materialization_exceeds_budget(active, projected_materialize, limit) {
            return Err(Error::Msg(format!(
                "This BFL/ComfyUI LyCORIS adapter must reconstruct a fused dense `[out,in]` delta for \
                 every target (the fused→split path has no allocation-free form) — ~{:.1} GB on top of \
                 the ~{:.1} GB already resident, over the ~{:.1} GB budget. Use a plain LoRA instead \
                 (near-zero extra memory on any tier). On a quantized tier, the bf16 tier merges this \
                 adapter into the dense weight in place — note bf16 renders differently.",
                projected_materialize as f64 / GIB,
                active as f64 / GIB,
                (limit as f64 * MATERIALIZE_BUDGET_HEADROOM) / GIB,
            )));
        }
    }

    // Pass 2 — reconstruct each fused delta and row-slice every destination's share.
    for plan in plans {
        let BflPlan {
            targets,
            fused_shape,
            reconstruct,
        } = plan;
        let delta = reconstruct(&fused_shape)?;
        for tgt in &targets {
            let parts: Vec<&str> = tgt.target_path.split('.').collect();
            let Some(lin) = host.adaptable_mut(&parts) else {
                report.unmatched_paths.push(tgt.target_path.clone());
                continue;
            };
            let piece = match &tgt.out_slice {
                Some(slice) => slice.apply(&delta)?,
                None => delta.clone(),
            };
            lin.push(Adapter::Lokr {
                delta: piece,
                scale,
            });
            report.applied += 1;
        }
    }
    Ok(report)
}

/// Install a metadata-stamped (peft) LoKr file in BFL/ComfyUI fused naming (sc-8345). Same Kronecker
/// reconstruction + `alpha/rank` fold as [`apply_lokr`], but the fused qkv source is rebuilt at the
/// host-fused shape and row-sliced into the split targets via [`install_bfl_lycoris`].
fn apply_lokr_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    map: &BTreeMap<String, Vec<BflLycorisTarget>>,
) -> Result<ApplyReport> {
    let file = parse_lokr(w)?;
    let (alpha, rank) = (file.alpha, file.rank);
    let dtype = host.lycoris_dtype();
    let groups = file.groups.into_iter().map(|(module, factors)| {
        (module, move |base: &[i32]| {
            reconstruct_lokr_delta(
                alpha,
                rank,
                base,
                factors.get("lokr_w1"),
                factors.get("lokr_w1_a"),
                factors.get("lokr_w1_b"),
                factors.get("lokr_w2"),
                factors.get("lokr_w2_a"),
                factors.get("lokr_w2_b"),
                dtype,
            )
        })
    });
    install_bfl_lycoris(host, map, groups, scale)
}

/// Install a third-party LyCORIS **LoKr** file in BFL/ComfyUI fused naming (sc-8345). Per-module
/// lycoris scale + tucker-capable Kronecker reconstruction (same as [`apply_lokr_thirdparty`]), fused→
/// split via [`install_bfl_lycoris`].
fn apply_lokr_thirdparty_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    map: &BTreeMap<String, Vec<BflLycorisTarget>>,
) -> Result<ApplyReport> {
    let dtype = host.lycoris_dtype();
    let groups = parse_lokr_thirdparty(w)?
        .into_iter()
        .map(|(module, g)| (module, move |base: &[i32]| g.delta(base, dtype)));
    install_bfl_lycoris(host, map, groups, scale)
}

/// Install a third-party LyCORIS **LoHa** file in BFL/ComfyUI fused naming (sc-8345). Hadamard
/// reconstruction (same as [`apply_loha_thirdparty`]), fused→split via [`install_bfl_lycoris`].
fn apply_loha_thirdparty_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    map: &BTreeMap<String, Vec<BflLycorisTarget>>,
) -> Result<ApplyReport> {
    let dtype = host.lycoris_dtype();
    let groups = parse_loha_thirdparty(w)?
        .into_iter()
        .map(|(module, g)| (module, move |base: &[i32]| g.delta(base, dtype)));
    install_bfl_lycoris(host, map, groups, scale)
}

/// Load and install every adapter in `specs` onto `host`, stacking in order. Each spec's file is
/// read, dispatched to the LoKr or PEFT-LoRA loader by its [`AdapterKind`], applied at `spec.scale`,
/// and its [`ApplyReport`] merged into the combined result — unmatched target paths are surfaced,
/// never silently dropped. `lora_strip_prefix` is the per-family namespace stripped from PEFT-LoRA
/// keys (e.g. `"transformer."`); it does not apply to LoKr (whose keys are bare module paths).
///
/// This is the load-time seam (sc-2534): a provider calls it inside `load()` with its model's
/// [`AdaptableHost`] while the model is still mutable. Empty `specs` is a no-op (empty report).
///
/// **Scope (F-035):** this fixed-prefix variant routes PEFT/diffusers LoRA, metadata-stamped LoKr,
/// and keyless third-party LyCORIS LoKr/LoHa. Unlike [`apply_adapter_specs_autoprefix`] it does NOT
/// detect BFL/ComfyUI fused→split or kohya-flattened files — both require walking the host module
/// tree (`bfl_targets()` / `kohya_table()`), which only the autoprefix path does. Callers that may
/// receive BFL or kohya files must use the autoprefix variant; here such a file would resolve no
/// targets and be reported as fully unmatched rather than applied.
pub fn apply_adapter_specs(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    lora_strip_prefix: Option<&str>,
) -> Result<ApplyReport> {
    let mut combined = ApplyReport::default();
    for spec in specs {
        let w = Weights::from_file(&spec.path)?;
        let report = match spec.kind {
            AdapterKind::Lokr => {
                // A keys-only third-party LyCORIS LoKr (kohya / ai-toolkit / lycoris-lib) carries
                // `lokr_*` keys but no `networkType` stamp, and its per-module `.alpha` scalars +
                // optional tucker `lokr_t2` are invisible to the peft parser — routing it to
                // `apply_lokr` would silently drop both and install a mis-scaled delta. Detect by
                // keys and route to the third-party applier, mirroring the `Lora` arm below and
                // the autoprefix dispatch (F-012).
                if !is_lokr(&w) && is_lokr_keys(&w) {
                    apply_lokr_thirdparty(host, &w, spec.scale)?
                } else {
                    apply_lokr(host, &w, spec.scale)?
                }
            }
            AdapterKind::Lora => {
                // The file's metadata is authoritative; a kind/metadata mismatch is a caller error
                // (the PEFT-LoRA loader would find no `lora_A/B` keys and apply nothing) — surface it.
                if is_lokr(&w) {
                    return Err(format!(
                        "adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )
                    .into());
                }
                // A third-party LyCORIS LoKr (sc-3642) / LoHa (sc-3643) carries `lokr_*` / `hada_*`
                // keys but no `networkType` stamp, so a caller can't know to set a non-Lora kind —
                // detect + route by keys.
                if is_lokr_keys(&w) {
                    apply_lokr_thirdparty(host, &w, spec.scale)?
                } else if is_loha_keys(&w) {
                    apply_loha_thirdparty(host, &w, spec.scale)?
                } else {
                    apply_lora_peft(host, &w, spec.scale, lora_strip_prefix)?
                }
            }
        };
        // Materialize the factors this file contributed at load (sc-24245): left lazy, the first
        // forward's command buffers wait on the safetensors reads — past the GPU watchdog on a
        // cold page cache (see `mlx_gen_qwen_image::loader::load_transformer_with`).
        w.materialize_accessed()?;
        combined.applied += report.applied;
        combined.unmatched_paths.extend(report.unmatched_paths);
    }
    Ok(combined)
}

// `COMMON_LORA_PREFIXES` — the LoRA key namespace prefixes diffusers/peft adapter files use, tried
// in order; the first a key begins with is stripped. LoKr files are bare (no prefix); kohya
// `lora_unet_…` files flatten the module dots to underscores and resolve through a separate
// flattened→dotted table ([`apply_lora_kohya`], sc-2618), not this prefix strip. SceneWorks' trained
// LoRAs use `transformer.` (peft `save_lora_weights`) or `diffusion_model.` (sd-scripts export) —
// both observed on real files. Defined in gen-core (`weightsmeta`) and re-exported above.

/// The LoRA namespace prefix present in `w`'s keys, if any (see [`COMMON_LORA_PREFIXES`]).
pub fn detect_lora_prefix(w: &Weights) -> Option<&'static str> {
    wmeta::detect_lora_prefix(w.keys())
}

/// Host-derived lookup tables shared across one [`apply_adapter_specs_autoprefix`] call's specs.
/// Each is a model-tree walk, so it is built lazily and at most once — the first time a spec's
/// classification (or its apply arm) needs it.
#[derive(Default)]
struct HostTables {
    kohya: Option<BTreeMap<String, String>>,
    bfl: Option<Vec<BflTarget>>,
    bfl_lyc: Option<BTreeMap<String, Vec<BflLycorisTarget>>>,
}

impl HostTables {
    fn kohya(&mut self, host: &impl AdaptableHost) -> &BTreeMap<String, String> {
        self.kohya
            .get_or_insert_with(|| kohya_table(&host.adaptable_paths()))
    }

    fn bfl(&mut self, host: &impl AdaptableHost) -> &[BflTarget] {
        self.bfl.get_or_insert_with(|| host.bfl_targets())
    }

    fn bfl_lyc(&mut self, host: &impl AdaptableHost) -> &BTreeMap<String, Vec<BflLycorisTarget>> {
        if self.bfl_lyc.is_none() {
            let map = bfl_lycoris_module_map(self.bfl(host));
            self.bfl_lyc = Some(map);
        }
        self.bfl_lyc.as_ref().unwrap()
    }
}

/// The on-disk format of one adapter file, as resolved by [`classify_adapter_format`] — the single
/// routing truth for [`apply_adapter_specs_autoprefix`]'s dispatch (F-069). Variant order mirrors
/// the detection precedence:
///
/// 1. **Third-party LyCORIS keys** (`lokr_*` without a `networkType` stamp, or `hada_*`) — before
///    BFL-LoRA/kohya, because a kohya-flattened LyCORIS file also carries the `lora_unet_` prefix,
///    which `is_kohya` would otherwise claim and then apply nothing (sc-3642/sc-3643).
/// 2. **BFL/ComfyUI fused→split LoRA naming** — before kohya because it shares the `lora_unet_`
///    prefix (sc-2743). Skipped (with kohya) for a metadata-stamped LoKr, whose keys stay dotted.
/// 3. **kohya-flattened LoRA** (`lora_unet_…`).
/// 4. The spec's declared [`AdapterKind`]: a peft/metadata LoKr (BFL-named or bare) or a
///    PEFT/diffusers LoRA — where a `Lora` declaration against `networkType=lokr` metadata is a
///    caller error, surfaced as [`Self::LoraKindMismatch`].
///
/// Each LyCORIS format carries a BFL twin because a LyCORIS file can ship in BFL/ComfyUI fused
/// naming on a FLUX host (sc-8345); every other host has an empty BFL surface, so the `*Bfl`
/// variants never classify there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdapterFormat {
    /// Third-party LyCORIS LoKr (`lokr_*` keys, no `networkType` stamp) in BFL/ComfyUI fused naming.
    ThirdpartyLokrBfl,
    /// Third-party LyCORIS LoKr in bare / diffusers / kohya-flattened naming.
    ThirdpartyLokr,
    /// Third-party LyCORIS LoHa (`hada_*` keys) in BFL/ComfyUI fused naming.
    ThirdpartyLohaBfl,
    /// Third-party LyCORIS LoHa in bare / diffusers / kohya-flattened naming.
    ThirdpartyLoha,
    /// BFL/ComfyUI fused→split LoRA (sc-2743).
    BflLora,
    /// kohya / sd-scripts flattened LoRA (`lora_unet_…`).
    KohyaLora,
    /// Metadata-stamped (peft) LoKr in BFL/ComfyUI fused naming (sc-8345).
    PeftLokrBfl,
    /// Declared-`Lokr` fallback: a metadata-stamped or bare dotted-path LoKr.
    PeftLokr,
    /// PEFT/diffusers LoRA — the common fallback.
    PeftLora,
    /// Declared `Lora` but the file's metadata says `networkType=lokr` — a caller error (the
    /// metadata is authoritative), surfaced as a hard error by the dispatch.
    LoraKindMismatch,
}

/// Classify one adapter file for the autoprefix dispatch (F-069). A pure function of the file's
/// keys/metadata, the declared `kind`, and the host's kohya/BFL surface; `tables` caches the
/// host-tree walks across specs. See [`AdapterFormat`] for the precedence rationale.
fn classify_adapter_format(
    w: &Weights,
    kind: AdapterKind,
    host: &impl AdaptableHost,
    tables: &mut HostTables,
) -> AdapterFormat {
    // The `networkType=lokr` stamp gates `lokr_*` keys: a stamped (peft) file routes by its stamp
    // (global metadata alpha/rank), a keys-only file by the third-party per-module derivation.
    let is_lokr_meta = is_lokr(w);
    if !is_lokr_meta && is_lokr_keys(w) {
        return if is_bfl_lycoris(w, tables.bfl_lyc(host)) {
            AdapterFormat::ThirdpartyLokrBfl
        } else {
            AdapterFormat::ThirdpartyLokr
        };
    }
    if is_loha_keys(w) {
        return if is_bfl_lycoris(w, tables.bfl_lyc(host)) {
            AdapterFormat::ThirdpartyLohaBfl
        } else {
            AdapterFormat::ThirdpartyLoha
        };
    }
    if !is_lokr_meta {
        if is_bfl(w, tables.bfl(host)) {
            return AdapterFormat::BflLora;
        }
        if is_kohya(w) {
            return AdapterFormat::KohyaLora;
        }
    }
    // No format-discriminating keys left — fall back to the spec's declared kind.
    match kind {
        AdapterKind::Lokr => {
            if is_lokr_meta && is_bfl_lycoris(w, tables.bfl_lyc(host)) {
                AdapterFormat::PeftLokrBfl
            } else {
                AdapterFormat::PeftLokr
            }
        }
        AdapterKind::Lora if is_lokr_meta => AdapterFormat::LoraKindMismatch,
        AdapterKind::Lora => AdapterFormat::PeftLora,
    }
}

/// [`apply_adapter_specs`] with per-file LoRA-prefix **auto-detection** ([`detect_lora_prefix`])
/// instead of a fixed prefix — the common provider path, since LoRA files vary
/// (`transformer.` / `diffusion_model.` / bare) while LoKr keys are bare. The host's key→module map
/// must match the (prefix-stripped) diffusers module paths.
pub fn apply_adapter_specs_autoprefix(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    let mut combined = ApplyReport::default();
    for report in apply_adapter_specs_autoprefix_reported(host, specs)? {
        combined.applied += report.applied;
        combined.unmatched_paths.extend(report.unmatched_paths);
    }
    Ok(combined)
}

/// Per-spec form of [`apply_adapter_specs_autoprefix`]. The host is mutated in the same input order
/// and the expensive host lookup tables are still shared across the batch; only the report reduction
/// is deferred so a strict batch caller can retain file attribution.
fn apply_adapter_specs_autoprefix_reported(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<Vec<ApplyReport>> {
    let mut tables = HostTables::default();
    let mut reports = Vec::with_capacity(specs.len());
    for spec in specs {
        // Load + classify the file once per spec (F-004); `classify_adapter_format` is the single
        // source of routing truth (F-069) — the arms below only fetch the table the applier needs
        // and apply.
        let w = Weights::from_file(&spec.path)?;
        let report = match classify_adapter_format(&w, spec.kind, host, &mut tables) {
            AdapterFormat::ThirdpartyLokrBfl => {
                let map = tables.bfl_lyc(host);
                apply_lokr_thirdparty_bfl(host, &w, spec.scale, map)?
            }
            AdapterFormat::ThirdpartyLokr => apply_lokr_thirdparty(host, &w, spec.scale)?,
            AdapterFormat::ThirdpartyLohaBfl => {
                let map = tables.bfl_lyc(host);
                apply_loha_thirdparty_bfl(host, &w, spec.scale, map)?
            }
            AdapterFormat::ThirdpartyLoha => apply_loha_thirdparty(host, &w, spec.scale)?,
            AdapterFormat::BflLora => {
                let targets = tables.bfl(host);
                apply_lora_bfl(host, &w, spec.scale, targets)?
            }
            AdapterFormat::KohyaLora => {
                let table = tables.kohya(host);
                apply_lora_kohya(host, &w, spec.scale, table)?
            }
            AdapterFormat::PeftLokrBfl => {
                let map = tables.bfl_lyc(host);
                apply_lokr_bfl(host, &w, spec.scale, map)?
            }
            AdapterFormat::PeftLokr => apply_lokr(host, &w, spec.scale)?,
            AdapterFormat::PeftLora => {
                apply_lora_peft(host, &w, spec.scale, detect_lora_prefix(&w))?
            }
            AdapterFormat::LoraKindMismatch => {
                return Err(format!(
                    "adapter {} declared Lora but its metadata says networkType=lokr",
                    spec.path.display()
                )
                .into());
            }
        };
        // Materialize the factors this file contributed at load (sc-24245): left lazy, the first
        // forward's command buffers wait on the safetensors reads — past the GPU watchdog on a
        // cold page cache (see `mlx_gen_qwen_image::loader::load_transformer_with`).
        w.materialize_accessed()?;
        reports.push(report);
    }
    Ok(reports)
}

// ---- ComfyUI / lightx2v diff-patch (full-weight/bias deltas), fold-after-build (sc-13825) --------
//
// The ComfyUI "diff patch" mechanism (`comfy/lora.py`): a `‹weight-key›.diff` is a **full-rank weight
// delta** applied `W += strength·diff`, and `‹weight-key›.diff_b` its **bias delta** `b += strength·
// diff_b`. It is the degenerate, already-materialized case of the same additive fold a LoRA/LoKr does,
// with δ handed over raw instead of rebuilt from `B·A` / Kronecker factors, plus a bias channel
// low-rank adapters cannot express. The community Krea "filter-bypass" ships one such tensor,
// `diffusion_model.txtfusion.projector.diff` (`[1, num_text_layers]`), a delta on the 12→1
// `text_fusion.projector` collapse (MLX parity for candle sc-13726).
//
// **Fold-after-build (the MLX seam).** Unlike candle — which folds into the raw weight map before the
// DiT builds — every Krea MLX lane (Resident/Sequential, pose control, the monolithic `KreaPipeline`)
// receives its adapter specs *after* the DiT is constructed, so this folds into the built host's dense
// `AdaptableLinear` bases via [`AdaptableLinear::merge_dense_delta`] / `merge_dense_bias_delta`,
// resolving each stem through the same [`strip_common_lora_prefix`] + host alias
// (`txtfusion`→`text_fusion`, sc-8185) the low-rank pass uses. A `.diff` target must be **dense**: the
// projector (and the other dense front-end projections) stays dense on every tier
// (`TextFusionTransformer::quantize` skips it), so the real bypass folds cleanly on dense, Q4, and Q8.
// A (theoretical) `.diff` aimed at a *quantized* front-end module is surfaced-and-skipped rather than
// unpacked — candle's before-build fold can unpack a packed base, this after-build fold deliberately
// cannot, and says so loudly instead of half-applying.
//
// **Module-coupled shape-aware skip.** A `.diff` whose shape ≠ the base weight (a cross-architecture or
// out-of-surface delta) is skipped as a whole module — its coupled `.diff_b` dropped with it, never a
// half-patch — and surfaced, mirroring ComfyUI's own warn-and-skip and the SCAIL2 diff-patch contract.
//
// ## Two channels that cover every tier, one that does not (sc-15326)
//
// The "a dense delta cannot fold into a packed base" rule above is true only of a **weight** delta into
// a **Linear**. Measured against the real lightx2v `Wan2.1-T2V-14B` cfg-step-distill file (647
// diff-patch keys), taking that rule as the whole story would have folded **7 of 647** on the default Q4
// tier and **407 of 647** on bf16 — the same LoRA rendering differently per tier, on a product where the
// quant tier is a *creative* choice. Two of the three channels in fact fold on **every** tier
// (tier-independent in *coverage* — see the per-channel dtype note below), and this fold uses them:
//
// * **`.diff_b` into a Linear (407 keys).** `QuantizedLinear` packs only the weight; the bias stays a
//   plain dense vector and the forward is `quantized_matmul(…) + b`. So `b += scale·δ_b` folds on Q4,
//   Q8 and bf16 alike — see [`AdaptableLinear::merge_bias_delta`]. Tier-independent in *coverage*:
//   every such key lands on every tier, though the fold dtype still follows the base (a quantize casts
//   the bias to bf16, so an f32-on-disk base folds in f32 dense and bf16 packed).
// * **`.diff` / `.diff_b` into a norm (240 keys).** RMSNorm/LayerNorm parameters are never packed by any
//   family's quantize predicate, so they are dense on every tier. They are not `AdaptableLinear`s at any
//   surface width, so they route through the separate [`AdaptableHost::diff_patch_param_mut`] surface.
// * **`.diff` into a Linear (0 keys in that file).** The only genuinely tier-dependent case, and the
//   only one still skipped-and-surfaced over a packed base.
//
// Result for that file: **647 of 647 on every tier**, with no tier divergence at all.

/// One module's diff-patch deltas: a full-rank weight delta and/or a bias delta.
#[derive(Default)]
struct DiffParts {
    diff: Option<Array>,   // weight delta, base-shaped
    diff_b: Option<Array>, // bias delta, base-shaped
}

/// What a diff-patch fold did: the number of weight/bias deltas folded, and the targets deliberately
/// skipped (shape-incompatible / packed-weight `.diff` / no base bias — coupled parts dropped, never
/// half-applied) or resolved to no module. Surfaced loudly by the caller **and** carried out on
/// [`ApplyReport`] so a provider can tell the user exactly what did not land; a file that folded
/// **nothing** is caught by the combined zero-match guard, not here.
#[derive(Debug, Default)]
pub struct DiffPatchReport {
    /// Count of weight + bias deltas folded (into a dense Linear weight, into any Linear's dense bias
    /// — packed or not — or into a host norm parameter).
    pub folded: usize,
    /// Targets whose weight/bias delta could not fold (shape mismatch, a weight `.diff` aimed at a
    /// packed Linear, or a `.diff_b` with no base bias) — coupled deltas dropped together, never
    /// half-applied.
    pub skipped: Vec<String>,
    /// Diff-patch stems that resolved to no adaptable module on the host.
    pub unmatched: Vec<String>,
    /// Deltas that resolved to a host target and shape-checked clean but were **deliberately not
    /// written**, because their spec's `scale` is 0 (sc-15265). Counted separately from
    /// [`folded`](Self::folded) so the host's arrays stay byte-identical to a never-installed adapter
    /// while the zero-match guard still sees that the file's targets *did* resolve.
    pub disabled: usize,
}

impl DiffPatchReport {
    /// Record one delta that resolved and shape-checked: [`folded`](Self::folded) when the spec is
    /// live, [`disabled`](Self::disabled) when its scale is 0 and nothing was written.
    fn record(&mut self, live: bool) {
        if live {
            self.folded += 1;
        } else {
            self.disabled += 1;
        }
    }
}

/// `true` if any of `keys` is a diff-patch delta name (`.diff` / `.diff_b`) — the structural marker of
/// a ComfyUI/lightx2v diff-patch file, which the forward-time residual loader cannot consume. The
/// header-only twin of [`has_diff_patch_keys`] (which needs a fully-loaded [`Weights`]): a caller with
/// just the safetensors key list (e.g. from a [`gen_core::weightsmeta::CheckpointMeta`]) can detect a
/// diff-patch adapter WITHOUT loading its tensors — used by the Krea multi-phase guard (sc-13884) to
/// reject a diff-patch adapter, whose delta folds irreversibly into the dense base and so cannot be
/// toggled off per phase.
pub fn has_diff_patch_key_names<'a>(mut keys: impl Iterator<Item = &'a str>) -> bool {
    keys.any(|k| k.ends_with(".diff") || k.ends_with(".diff_b"))
}

/// `true` if any tensor key in `w` is a diff-patch delta (`.diff` / `.diff_b`) — the structural marker
/// of a ComfyUI/lightx2v diff-patch file, which the forward-time residual loader cannot consume.
pub fn has_diff_patch_keys(w: &Weights) -> bool {
    has_diff_patch_key_names(w.keys())
}

/// Fold every ComfyUI/lightx2v **diff-patch** (`.diff` weight / `.diff_b` bias) delta the `specs` carry
/// into `host`'s matched **dense** `AdaptableLinear` bases at each spec's `scale`: `‹stem›.diff` →
/// `W += scale·δ`, `‹stem›.diff_b` → `b += scale·δ_b`. Each stem resolves through the same
/// `strip_common_lora_prefix` + host module alias the low-rank pass uses, so
/// `diffusion_model.txtfusion.projector` reaches `text_fusion.projector`. Non-diff keys (`lora_*` /
/// `lokr_*`) are ignored here — they ride the disjoint forward-time residual pass. A no-op for specs
/// carrying no `.diff`/`.diff_b`. See the module note above for the fold-after-build + coupled-skip
/// contract.
pub fn fold_diff_patch_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<DiffPatchReport> {
    let mut combined = DiffPatchReport::default();
    for report in fold_diff_patch_adapters_reported(host, specs)? {
        combined.folded += report.folded;
        combined.disabled += report.disabled;
        combined.skipped.extend(report.skipped);
        combined.unmatched.extend(report.unmatched);
    }
    Ok(combined)
}

/// Per-spec form of [`fold_diff_patch_adapters`]. All diff-patch specs are still folded in one
/// ordered pass before the low-rank pass; retaining the individual reports does not alter mutation
/// order.
fn fold_diff_patch_adapters_reported(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<Vec<DiffPatchReport>> {
    let mut reports = Vec::with_capacity(specs.len());
    for spec in specs {
        let mut report = DiffPatchReport::default();
        let w = Weights::from_file(&spec.path)?;
        // Group each module's `.diff`/`.diff_b` by its resolved (prefix-stripped) dotted stem.
        let mut groups: BTreeMap<String, DiffParts> = BTreeMap::new();
        for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
            if let Some(stem) = key.strip_suffix(".diff_b") {
                groups
                    .entry(strip_common_lora_prefix(stem).to_string())
                    .or_default()
                    .diff_b = Some(w.require(&key)?.clone());
            } else if let Some(stem) = key.strip_suffix(".diff") {
                groups
                    .entry(strip_common_lora_prefix(stem).to_string())
                    .or_default()
                    .diff = Some(w.require(&key)?.clone());
            }
        }
        for (stem, parts) in groups {
            fold_one_diff_module(host, &stem, &parts, spec.scale, &mut report)?;
        }
        // Materialize the factors this file contributed at load (sc-24245): left lazy, the first
        // forward's command buffers wait on the safetensors reads — past the GPU watchdog on a
        // cold page cache (see `mlx_gen_qwen_image::loader::load_transformer_with`).
        w.materialize_accessed()?;
        reports.push(report);
    }
    Ok(reports)
}

/// Scale a delta by `scale` in f32 (the fold math runs in f32; `merge_dense_delta`/`_bias_delta` cast
/// back to the base dtype on write).
fn scaled_f32(delta: &Array, scale: f32) -> Result<Array> {
    Ok(delta
        .as_dtype(Dtype::Float32)?
        .multiply(crate::array::scalar(scale))?)
}

/// Record a whole module as skipped — the weight delta and, when present, its coupled bias delta
/// (never a half-patch).
fn skip_whole_module(stem: &str, parts: &DiffParts, report: &mut DiffPatchReport) {
    report.skipped.push(stem.to_string());
    if parts.diff_b.is_some() {
        report.skipped.push(format!("{stem}.bias"));
    }
}

/// Fold one module's `.diff`/`.diff_b` into the host's base at that dotted stem.
///
/// Routes to whichever of the two host surfaces owns the stem: an [`AdaptableLinear`]
/// ([`AdaptableHost::adaptable_mut`]) first, then a dense non-Linear parameter — a norm
/// ([`AdaptableHost::diff_patch_param_mut`], sc-15326). An unresolved path on both is recorded as
/// unmatched.
///
/// On a Linear: a weight-delta shape ≠ the base, or a weight delta aimed at a **packed** base (this
/// after-build fold does not unpack), skips the **whole** module — its coupled bias dropped too, never a
/// half-patch. The bias delta itself is *not* tier-gated: a quantized Linear keeps its bias dense, so
/// `.diff_b` folds identically on Q4/Q8/bf16 (sc-15326). A `.diff_b` with no shape-matching base bias is
/// surfaced but does not undo the weight fold.
fn fold_one_diff_module(
    host: &mut impl AdaptableHost,
    stem: &str,
    parts: &DiffParts,
    scale: f32,
    report: &mut DiffPatchReport,
) -> Result<()> {
    // sc-15265: a scale-0 install must be byte-identical to no install at all. Resolve and shape-check
    // exactly as usual — so `skipped`/`unmatched` stay truthful and the zero-match guard still sees the
    // targets — but write nothing. (`δ · 0.0` is not reliably inert: it flips `-0.0` to `+0.0`, and
    // `Inf · 0.0` is NaN, which would poison the host at precisely the setting that turns the adapter
    // OFF.)
    let live = scale != 0.0;
    let segs: Vec<&str> = stem.split('.').collect();
    // SC-18319 — every *decision* this function makes (does the path resolve to a Linear at all, is
    // the base packed, do the weight and bias shapes match) is answerable from the PROBE half, so the
    // `&mut` handle is taken only on the branches that actually write. That matters twice over here:
    // the quantized/shape-mismatch arm SKIPS, and a `scale == 0` install writes nothing at all — under
    // the old `&mut`-first resolution both of those still unfused every `FusedQkvProjection` they
    // touched, in exchange for no mutation whatsoever.
    let Some(facts) = host.adaptable_facts(&segs) else {
        return fold_one_diff_param(host, stem, &segs, parts, scale, report);
    };

    if let Some(diff) = &parts.diff {
        // A weight `.diff` is the one genuinely tier-dependent channel: a dense delta cannot fold into
        // a packed weight, and unpacking to make it fit would defeat the tier the user chose. Skip the
        // whole module (coupled bias with it) and surface it.
        if facts.is_quantized || diff.shape() != facts.base_shape.as_slice() {
            skip_whole_module(stem, parts, report);
            return Ok(());
        }
    }
    // Bias delta (`.diff_b`): folds onto `{stem}.bias` when the base carries a shape-matching bias —
    // packed or not, since `QuantizedLinear` never packs its bias.
    let bias_ok = parts
        .diff_b
        .as_ref()
        .is_some_and(|d| facts.bias_shape.as_deref() == Some(d.shape()));

    if let Some(diff) = &parts.diff {
        if live {
            host.adaptable_mut(&segs)
                .expect("resolved through the probe above")
                .merge_dense_delta(&scaled_f32(diff, scale)?)?;
        }
        report.record(live);
    }

    if let Some(diff_b) = &parts.diff_b {
        if bias_ok {
            if live {
                host.adaptable_mut(&segs)
                    .expect("resolved through the probe above")
                    .merge_bias_delta(&scaled_f32(diff_b, scale)?)?;
            }
            report.record(live);
        } else {
            report.skipped.push(format!("{stem}.bias"));
        }
    }
    Ok(())
}

/// Fold one module's deltas through the host's **non-Linear dense parameter** surface — the norms
/// (sc-15326). Reached only when the stem resolves to no [`AdaptableLinear`].
///
/// Unlike the Linear path this is tier-independent by construction: every parameter reachable through
/// [`AdaptableHost::diff_patch_param_mut`] is dense on every quant tier (norm weights/biases are never
/// in a family's quantize predicate), so `p += scale·δ` is the same arithmetic at Q4, Q8 and bf16.
///
/// The two failure modes are split exactly as on the Linear path (sc-15326):
///
/// * A **shape mismatch** on either half skips the **whole** module — a delta that happens to fit one
///   half of some other architecture's module is not a delta for this one, and a half-patched norm is
///   worse than an unpatched one.
/// * A half the **host does not expose** skips only *that* half. Every qk-RMSNorm is weight-only, so a
///   lightning/step-distill file carrying `norm_q.diff_b` next to a perfectly valid `norm_q.diff` still
///   lands the weight delta and surfaces just the bias — mirroring the Linear path, where a `.diff_b`
///   with no shape-matching base bias is surfaced but does not undo the weight fold.
///
/// A stem with neither half present on the host is unmatched.
fn fold_one_diff_param(
    host: &mut impl AdaptableHost,
    stem: &str,
    segs: &[&str],
    parts: &DiffParts,
    scale: f32,
    report: &mut DiffPatchReport,
) -> Result<()> {
    // A module the host exposes at all — probed on whichever half the file actually carries, so a
    // weight-only norm (RMSNorm) and a weight+bias norm (affine LayerNorm) both resolve.
    let known = host
        .diff_patch_param_mut(segs, DiffPatchPart::Weight)
        .is_some()
        || host
            .diff_patch_param_mut(segs, DiffPatchPart::Bias)
            .is_some();
    if !known {
        report.unmatched.push(stem.to_string());
        return Ok(());
    }

    let halves = [
        (DiffPatchPart::Weight, parts.diff.as_ref(), stem.to_string()),
        (
            DiffPatchPart::Bias,
            parts.diff_b.as_ref(),
            format!("{stem}.bias"),
        ),
    ];

    // Pre-check both halves BEFORE mutating either, so a shape-mismatched module is skipped whole
    // rather than half-patched — while a half the host simply does not expose only takes itself out.
    let mut absent = [false; 2];
    for (i, (part, delta, _)) in halves.iter().enumerate() {
        let Some(delta) = delta else { continue };
        match host.diff_patch_param_mut(segs, *part) {
            Some(p) if p.shape() == delta.shape() => {}
            // Present but the wrong shape: not a delta for this module at all.
            Some(_) => {
                skip_whole_module(stem, parts, report);
                return Ok(());
            }
            // Not a channel this module has (a `.diff_b` on a weight-only RMSNorm). Surfaced below;
            // the other half still folds.
            None => absent[i] = true,
        }
    }

    for (i, (part, delta, label)) in halves.iter().enumerate() {
        let Some(delta) = delta else { continue };
        if absent[i] {
            report.skipped.push(label.clone());
            continue;
        }
        // Scale 0 (sc-15265): resolved and shape-checked, deliberately not written.
        if scale != 0.0 {
            let scaled = scaled_f32(delta, scale)?;
            let p = host
                .diff_patch_param_mut(segs, *part)
                .expect("shape-checked");
            *p = add(&*p, &scaled.as_dtype(p.dtype())?)?;
        }
        report.record(scale != 0.0);
    }
    Ok(())
}

/// Provider-facing load-time adapter install: [`apply_adapter_specs_autoprefix`] plus a strict
/// no-silent-drop policy — errors if a non-empty spec list matched nothing, or if any adapter
/// target resolved to no module. `model` names the model in the error (e.g. `"z_image_turbo"`).
/// Both Z-Image and Qwen providers call this; the only per-family piece is the model's
/// `AdaptableHost` key→module map.
pub fn apply_adapters_strict(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    model: &str,
) -> Result<ApplyReport> {
    apply_adapters_strict_inner(host, specs, model, 0)
}

/// Diff-patch-aware strict install (sc-13825, MLX parity for candle sc-13726): fold any ComfyUI
/// `.diff`/`.diff_b` **diff-patch** deltas into the host's dense bases first ([`fold_diff_patch_adapters`]),
/// then run the ordinary low-rank residual pass, relaxing the zero-match guard by the count folded. The
/// two passes are disjoint by key suffix, so a mixed lightning file (both `.diff` and `lora_*`) has its
/// weight/bias deltas folded AND its low-rank factors stacked as residuals. Skipped / unmatched
/// diff-patch targets are surfaced on `stderr` **and** returned on
/// [`ApplyReport::diff_patch_unapplied`], so a provider can put a partial install in front of the user
/// instead of leaving it in an engine log (sc-15326); a file that folds nothing **and** matches no
/// low-rank target still errors via the combined guard. Krea's filter-bypass and Krea Realtime's
/// step-distill entry.
///
/// A spec at **scale 0** folds nothing (sc-15265 — a disabled adapter is byte-identical to an
/// uninstalled one), but its resolved targets still relax the guard, so disabling a *pure* diff-patch
/// file does not turn into a "matched nothing" error.
pub fn apply_adapters_strict_with_diff_patch(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    model: &str,
) -> Result<ApplyReport> {
    let mut combined = ApplyReport::default();
    for report in apply_adapters_strict_with_diff_patch_reported(host, specs, model)? {
        combined.applied += report.applied;
        combined.unmatched_paths.extend(report.unmatched_paths);
        combined
            .diff_patch_unapplied
            .extend(report.diff_patch_unapplied);
    }
    Ok(combined)
}

/// Per-spec report form of [`apply_adapters_strict_with_diff_patch`].
///
/// This preserves the batch install contract exactly: every spec's diff-patch deltas fold in input
/// order first, then every spec's low-rank residuals install in input order, and the strict guards
/// evaluate the batch totals. In particular, a wholly unsupported diff-patch file is reported as
/// unapplied without failing the load when another file in the same batch applies; the batch only
/// fails the zero-match guard when nothing across the entire batch lands.
pub fn apply_adapters_strict_with_diff_patch_reported(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    model: &str,
) -> Result<Vec<ApplyReport>> {
    let diff_reports = fold_diff_patch_adapters_reported(host, specs)?;
    let skipped: Vec<&str> = diff_reports
        .iter()
        .flat_map(|report| report.skipped.iter().map(String::as_str))
        .collect();
    if !skipped.is_empty() {
        eprintln!(
            "{model} adapters: {} diff-patch target(s) skipped (shape-incompatible / a weight `.diff` \
             on a packed Linear / no base bias — coupled parts dropped, never half-applied): {:?}",
            skipped.len(),
            skipped
        );
    }
    let unmatched: Vec<&str> = diff_reports
        .iter()
        .flat_map(|report| report.unmatched.iter().map(String::as_str))
        .collect();
    if !unmatched.is_empty() {
        eprintln!(
            "{model} adapters: {} diff-patch target(s) matched no module, skipped: {:?}",
            unmatched.len(),
            unmatched
        );
    }

    let mut reports = apply_adapter_specs_autoprefix_reported(host, specs)?;
    debug_assert_eq!(reports.len(), diff_reports.len());
    for (report, diff) in reports.iter_mut().zip(diff_reports) {
        // Diff-patch deltas that resolved count toward the total install, so the returned report is
        // truthful. A scale-0 delta is included for the same reason the low-rank pass counts a scale-0
        // adapter as applied: the target matched and was accepted, it is simply inert.
        report.applied += diff.folded + diff.disabled;
        // Everything that did NOT land, carried to the provider so a partial install can reach the
        // user with its originating file intact.
        report.diff_patch_unapplied = diff.skipped;
        report.diff_patch_unapplied.extend(diff.unmatched);
    }

    let applied: usize = reports.iter().map(|report| report.applied).sum();
    if !specs.is_empty() && applied == 0 {
        return Err(format!(
            "{model} adapters: no target modules matched across {} adapter file(s) — check the \
             format/prefix (expected diffusers/peft LoRA, kohya `lora_unet_` LoRA, BFL/ComfyUI \
             fused→split LoRA — for a host with a BFL surface — LoKr keys, or a ComfyUI/lightx2v \
             `<module>.diff`/`.diff_b` diff-patch)",
            specs.len()
        )
        .into());
    }
    let low_rank_unmatched: Vec<&str> = reports
        .iter()
        .flat_map(|report| report.unmatched_paths.iter().map(String::as_str))
        .collect();
    if !low_rank_unmatched.is_empty() {
        return Err(format!(
            "{model} adapters: {} adapter target(s) matched no module (surfaced, not silently \
             dropped): {:?}",
            low_rank_unmatched.len(),
            low_rank_unmatched
        )
        .into());
    }
    Ok(reports)
}

/// Core of [`apply_adapters_strict`]: `pre_applied` is the count of diff-patch targets a prior pass
/// already resolved — folded, or deliberately left unwritten at scale 0 (0 for the plain path). It only
/// relaxes the zero-match guard — a diff-patch-only file whose delta already folded resolves zero
/// low-rank residuals here, and must not read as "matched nothing".
fn apply_adapters_strict_inner(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    model: &str,
    pre_applied: usize,
) -> Result<ApplyReport> {
    let report = apply_adapter_specs_autoprefix(host, specs)?;
    if !specs.is_empty() && report.applied == 0 && pre_applied == 0 {
        return Err(format!(
            "{model} adapters: no target modules matched across {} adapter file(s) — check the \
             format/prefix (expected diffusers/peft LoRA, kohya `lora_unet_` LoRA, BFL/ComfyUI \
             fused→split LoRA — for a host with a BFL surface — LoKr keys, or a ComfyUI/lightx2v \
             `<module>.diff`/`.diff_b` diff-patch)",
            specs.len()
        )
        .into());
    }
    if !report.unmatched_paths.is_empty() {
        return Err(format!(
            "{model} adapters: {} adapter target(s) matched no module (surfaced, not silently \
             dropped): {:?}",
            report.unmatched_paths.len(),
            report.unmatched_paths
        )
        .into());
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::{AdaptableLinear, Adapter};
    use crate::runtime::{AdapterKind, AdapterSpec};
    use mlx_rs::ops::indexing::TryIndexOp;
    use mlx_rs::ops::{all_close, array_eq};
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// A host whose modules live at arbitrary dotted paths — including segment names with internal
    /// underscores (`feed_forward`, `to_out.0`) so the kohya flattening is genuinely ambiguous and a
    /// blind `_`→`.` split would mis-route. `adaptable_paths` returns the registered paths, so it
    /// exercises the real `flattened → dotted` table path.
    struct MultiHost {
        mods: HashMap<String, AdaptableLinear>,
        paths: Vec<String>,
    }
    impl MultiHost {
        fn new(specs: &[(&str, Array)]) -> Self {
            let mut mods = HashMap::new();
            let mut paths = Vec::new();
            for (p, w) in specs {
                mods.insert((*p).to_string(), AdaptableLinear::dense(w.clone(), None));
                paths.push((*p).to_string());
            }
            Self { mods, paths }
        }
    }
    impl AdaptableHost for MultiHost {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            self.mods.get_mut(&path.join("."))
        }
        fn adaptable_paths(&self) -> Vec<String> {
            self.paths.clone()
        }
    }

    /// Minimal host with a single adaptable linear at path `["lin"]`.
    struct OneLinear {
        lin: AdaptableLinear,
    }
    impl AdaptableHost for OneLinear {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            match path {
                ["lin"] => Some(&mut self.lin),
                _ => None,
            }
        }
    }

    fn scratch_file(tmp: &tempfile::TempDir, name: &str) -> PathBuf {
        // Per-process scratch dir — a fixed `$TMPDIR` name races a second concurrent `cargo test`.
        let dir = tmp.path().join("mlx_gen_loader_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// Mimics the Krea host's native-dialect routing (sc-8185 aliases `blocks`/`wq…`) but,
    /// like every `AdaptableHost`, routes from the *bare* module path — it has no
    /// `diffusion_model.` arm. So it only resolves an ai-toolkit key once the
    /// `strip_common_lora_prefix` step (sc-8395) removes that namespace.
    struct BareModuleHost {
        lin: AdaptableLinear,
    }
    impl AdaptableHost for BareModuleHost {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            match path {
                ["blocks", "0", "attn", "wq"] => Some(&mut self.lin),
                _ => None,
            }
        }
        fn adaptable_paths(&self) -> Vec<String> {
            vec!["blocks.0.attn.wq".to_string()]
        }
    }

    #[test]
    fn strip_common_lora_prefix_removes_known_namespaces() {
        assert_eq!(
            strip_common_lora_prefix("diffusion_model.blocks.0.attn.wq"),
            "blocks.0.attn.wq"
        );
        assert_eq!(strip_common_lora_prefix("transformer.x.y"), "x.y");
        // No known prefix → returned unchanged.
        assert_eq!(
            strip_common_lora_prefix("blocks.0.attn.wq"),
            "blocks.0.attn.wq"
        );
    }

    #[test]
    fn thirdparty_lokr_resolves_diffusion_model_prefixed_keys() {
        let tmp = tempfile::tempdir().unwrap();
        // sc-8395: ostris ai-toolkit writes Krea-2 LoKr keys as
        // `diffusion_model.‹native path›.lokr_w*`. Before the prefix strip the dotted
        // fallback handed the host `["diffusion_model", …]`, which matches no arm →
        // "no target modules matched". After the strip it resolves via the bare path.
        let weight = Array::from_slice(
            &(0..16).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 4],
        );
        // Both factors full ⇒ lycoris forces scale 1; w1[2,2] ⊗ w2[2,2] = ΔW[4,4].
        let w1 = Array::from_slice(&[0.5f32, 0.6, 0.7, 0.8], &[2, 2]);
        let w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let path = scratch_file(&tmp, "krea_aitoolkit_lokr.safetensors");
        Array::save_safetensors(
            vec![
                ("diffusion_model.blocks.0.attn.wq.lokr_w1", &w1),
                ("diffusion_model.blocks.0.attn.wq.lokr_w2", &w2),
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(is_lokr_keys(&w));

        let mut host = BareModuleHost {
            lin: AdaptableLinear::dense(weight, None),
        };
        let report = apply_lokr_thirdparty(&mut host, &w, 1.0).unwrap();
        assert_eq!(
            report.applied, 1,
            "the diffusion_model.-prefixed key must resolve after the strip"
        );
        assert!(report.unmatched_paths.is_empty());
    }

    #[test]
    fn lora_peft_transposes_and_folds_alpha() {
        let tmp = tempfile::tempdir().unwrap();
        // base [out=4, in=3]; PEFT lora_A [r=2, in=3], lora_B [out=4, r=2], alpha=4 (rank=2).
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        let path = scratch_file(&tmp, "lora.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_A.weight", &a_raw),
                ("lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha),
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_lora_peft(&mut host, &w, 0.5, None).unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.unmatched_paths.is_empty());

        // Reference: a = A^T [in,r], b = B^T * (alpha/rank=2.0) [r,out], scale 0.5.
        let mut expected = AdaptableLinear::dense(weight, None);
        let b_scaled = b_raw
            .t()
            .multiply(Array::from_slice(&[2.0f32], &[1]))
            .unwrap();
        expected.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_scaled,
            scale: 0.5,
        });

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let got = host.lin.forward(&x).unwrap();
        let want = expected.forward(&x).unwrap();
        assert!(all_close(&got, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn lora_peft_honors_lora_adapter_metadata_alpha() {
        let tmp = tempfile::tempdir().unwrap();
        // sc-5513: a diffusers / PEFT `save_lora_adapter` file carries NO per-target `.alpha` tensor —
        // the scaling lives in the `lora_adapter_metadata` header blob. With `lora_alpha = 16`, `r = 8`
        // the PEFT loader must fold `(16/8) = 2.0` (the metadata strength), not the pre-sc-5513
        // `alpha = rank` default (factor 1.0). Proves the blob is read and applied.
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        // PEFT factors with a TRUE rank of 8 (matching the blob `r`): A [r=8, in=3], B [out=4, r=8].
        let a_raw = Array::from_slice(
            &(0..24).map(|i| i as f32 * 0.03 - 0.3).collect::<Vec<_>>(),
            &[8, 3],
        );
        let b_raw = Array::from_slice(
            &(0..32).map(|i| 0.4 - i as f32 * 0.02).collect::<Vec<_>>(),
            &[4, 8],
        );

        let path = scratch_file(&tmp, "lora_adapter_metadata.safetensors");
        let meta = HashMap::from([(
            "lora_adapter_metadata".to_string(),
            r#"{"lora_alpha": 16, "r": 8}"#.to_string(),
        )]);
        // Deliberately NO `lin.alpha` tensor — the scaling must come from the blob.
        Array::save_safetensors(
            vec![("lin.lora_A.weight", &a_raw), ("lin.lora_B.weight", &b_raw)],
            Some(&meta),
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(w.metadata("lora_adapter_metadata").is_some());

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_lora_peft(&mut host, &w, 1.0, None).unwrap();
        assert_eq!(report.applied, 1);

        // Reference: alpha 16 over rank 8 ⇒ factor 2.0 folded into B (scale 1.0).
        let mut expected = AdaptableLinear::dense(weight.clone(), None);
        expected.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_raw
                .t()
                .multiply(Array::from_slice(&[2.0f32], &[1]))
                .unwrap(),
            scale: 1.0,
        });
        // The pre-sc-5513 default (alpha = rank ⇒ factor 1.0) would diverge by a full factor of 2.
        let mut buggy = AdaptableLinear::dense(weight, None);
        buggy.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_raw.t(),
            scale: 1.0,
        });

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let got = host.lin.forward(&x).unwrap();
        let want = expected.forward(&x).unwrap();
        let old = buggy.forward(&x).unwrap();
        assert!(
            all_close(&got, &want, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>(),
            "metadata-alpha fold must match (16/8)·scale"
        );
        assert!(
            !all_close(&got, &old, 1e-4, 1e-4, false)
                .unwrap()
                .item::<bool>(),
            "metadata alpha must differ from the alpha=rank default"
        );
    }

    /// sc-5513 **live torch-PEFT A/B** (the epic 3641 / sc-3671 on-device harness; torch IS available
    /// on the Mac at `~/mlx-flux-venv`). `#[ignore]` — gated on a real diffusers `save_lora_adapter`
    /// file, generated with peft 0.19 + diffusers 0.37:
    /// ```text
    /// ~/mlx-flux-venv/bin/python - <<'PY'
    /// import os, torch
    /// from diffusers import UNet2DConditionModel
    /// from peft import LoraConfig
    /// torch.manual_seed(0)
    /// unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
    ///     block_out_channels=(16,32), down_block_types=("CrossAttnDownBlock2D","DownBlock2D"),
    ///     up_block_types=("UpBlock2D","CrossAttnUpBlock2D"), cross_attention_dim=16,
    ///     attention_head_dim=2, norm_num_groups=4)
    /// unet.add_adapter(LoraConfig(r=8, lora_alpha=16, target_modules=["to_q","to_k","to_v"],
    ///     alpha_pattern={"to_k":8}, rank_pattern={"to_k":16}, init_lora_weights=False))
    /// unet.save_lora_adapter("/tmp/sc5513_adapter")
    /// PY
    /// SC5513_PEFT_ADAPTER=/tmp/sc5513_adapter/pytorch_lora_weights.safetensors \
    ///   cargo test -p mlx-gen peft_lora_adapter_metadata_ab -- --ignored --nocapture
    /// ```
    /// peft's authoritative per-module scaling (`mod.scaling['default']`) is then `to_q`/`to_v` = 16/8 =
    /// 2.0 and `to_k` = 8/16 = 0.5 (the override is deliberately discriminating). The Rust core loader
    /// must install each residual at exactly that scaling — proving the `lora_adapter_metadata` blob is
    /// honored on a genuine torch file (which carries NO per-target `.alpha` tensor, the bug's premise).
    #[test]
    #[ignore = "needs a diffusers save_lora_adapter file via SC5513_PEFT_ADAPTER (see doc comment)"]
    fn peft_lora_adapter_metadata_ab() {
        let Ok(path) = std::env::var("SC5513_PEFT_ADAPTER") else {
            eprintln!("SC5513_PEFT_ADAPTER unset — skipping live torch A/B");
            return;
        };
        let w = Weights::from_file(&path).unwrap();
        // The whole premise: a real diffusers `save_lora_adapter` file ships NO per-target `.alpha`
        // tensor — the scaling lives in the `lora_adapter_metadata` blob.
        assert!(
            !w.keys().any(|k| k.ends_with(".alpha")),
            "diffusers save_lora_adapter must not ship a per-target .alpha tensor"
        );
        assert!(w.metadata(wmeta::LORA_ADAPTER_METADATA_KEY).is_some());

        let block = "down_blocks.0.attentions.0.transformer_blocks.0.attn1";
        // peft ground truth (confirmed independently via `mod.scaling`): global 2.0, `to_k` override 0.5.
        for (leaf, peft_scale) in [("to_q", 2.0f32), ("to_k", 0.5f32)] {
            let module = format!("{block}.{leaf}");
            let a_raw = w
                .require(&format!("{module}.lora_A.weight"))
                .unwrap()
                .clone();
            let b_raw = w
                .require(&format!("{module}.lora_B.weight"))
                .unwrap()
                .clone();
            let out = b_raw.shape()[0];
            let inp = a_raw.shape()[1];
            // Zero base so the forward IS the pure (scaled) residual.
            let zero = Array::from_slice(&vec![0f32; (out * inp) as usize], &[out, inp]);
            let mut host = MultiHost::new(&[(module.as_str(), zero.clone())]);
            let report = apply_lora_peft(&mut host, &w, 1.0, None).unwrap();
            assert!(report.applied >= 1, "{leaf}: not applied");

            // Reference at peft's ground-truth scaling: residual = (x·Aᵀ·Bᵀ)·peft_scale.
            let mut expect = AdaptableLinear::dense(zero, None);
            expect.push(Adapter::Lora {
                a: a_raw.t(),
                b: b_raw
                    .t()
                    .multiply(Array::from_slice(&[peft_scale], &[1]))
                    .unwrap(),
                scale: 1.0,
            });
            let x = Array::from_slice(
                &(0..inp).map(|i| (i as f32 * 0.3).sin()).collect::<Vec<_>>(),
                &[1, inp],
            );
            let segs: Vec<&str> = module.split('.').collect();
            let got = host.adaptable_mut(&segs).unwrap().forward(&x).unwrap();
            let want = expect.forward(&x).unwrap();
            assert!(
                all_close(&got, &want, 1e-4, 1e-4, false)
                    .unwrap()
                    .item::<bool>(),
                "{leaf}: Rust apply diverged from peft scaling {peft_scale}"
            );
            println!("OK {leaf}: Rust apply matches peft scaling {peft_scale}");
        }
    }

    #[test]
    fn lora_bf16_scalar_alpha_reads_without_panic() {
        let tmp = tempfile::tempdir().unwrap();
        // sc-2657: real kohya/BFL FLUX LoRAs ship `alpha` as a **bf16 scalar of shape []**. The alpha
        // read used `as_slice::<f32>()`, which `unwrap`s a dtype mismatch and would panic on bf16 — a
        // latent bug masked by every prior test synthesizing f32 alpha. The fix casts to f32 first.
        // Here a bf16 `[]`-shaped alpha must load AND fold identically to its f32 `[1]`-shaped twin.
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        // alpha = 16 (exactly representable in bf16), stored bf16 and 0-d — like the real file.
        let alpha_bf16 = Array::from_slice(&[16.0f32], &[1])
            .reshape(&[] as &[i32])
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        let path = scratch_file(&tmp, "lora_bf16_alpha.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_A.weight", &a_raw),
                ("lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha_bf16),
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_lora_peft(&mut host, &w, 0.5, None).unwrap();
        assert_eq!(report.applied, 1, "bf16 alpha LoRA should apply, not panic");

        // Reference: identical fold with alpha=16, rank=2 → factor 8.
        let mut expected = AdaptableLinear::dense(weight, None);
        expected.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_raw
                .t()
                .multiply(Array::from_slice(&[8.0f32], &[1]))
                .unwrap(),
            scale: 0.5,
        });
        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let got = host.lin.forward(&x).unwrap();
        let want = expected.forward(&x).unwrap();
        assert!(all_close(&got, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn scalar_alpha_empty_tensor_errors_not_panic() {
        // sc-3959 added the no-panic guard for a malformed third-party adapter with a zero-length
        // `.alpha` (before it, `as_slice::<f32>()` panicked on the size-0 array, aborting the worker).
        // F-031 then tightened it from a silent `Ok(None)` to a typed error: the callers only reach
        // `scalar_alpha` for a *present* alpha key, so present-but-empty is a malformed file — returning
        // `None` would be indistinguishable from "no alpha" and silently mis-scale the adapter.
        let empty = Array::from_slice(&[] as &[f32], &[0]);
        assert_eq!(empty.size(), 0);
        let err = scalar_alpha(&empty)
            .expect_err("size-0 alpha must be a typed error, not Ok")
            .to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn lora_peft_folds_bare_alpha_under_a_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        // Prefixed `lora_A/B` (`transformer.lin.lora_{A,B}.weight`) + a BARE `lin.alpha` — the
        // fork's Qwen convention (bare-only alpha patterns). The bare alpha must NOT be dropped:
        // the residual folds alpha/rank into B exactly as the all-bare case does. (sc-2528 review.)
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]); // [r=2, in=3]
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let alpha = Array::from_slice(&[4.0f32], &[1]); // rank=2 -> factor 2

        let path = scratch_file(&tmp, "lora_prefixed_bare_alpha.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.lin.lora_A.weight", &a_raw),
                ("transformer.lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha), // BARE — no `transformer.` prefix
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_lora_peft(&mut host, &w, 0.5, Some("transformer.")).unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.unmatched_paths.is_empty());

        // Reference: B scaled by alpha/rank = 2 (the bare alpha was honored).
        let mut expected = AdaptableLinear::dense(weight, None);
        let b_scaled = b_raw
            .t()
            .multiply(Array::from_slice(&[2.0f32], &[1]))
            .unwrap();
        expected.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_scaled,
            scale: 0.5,
        });
        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let got = host.lin.forward(&x).unwrap();
        let want = expected.forward(&x).unwrap();
        assert!(
            all_close(&got, &want, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>(),
            "bare alpha under a prefix was dropped or mis-folded"
        );
    }

    #[test]
    fn lora_peft_conflicting_alpha_errors() {
        let tmp = tempfile::tempdir().unwrap();
        // A prefixed alpha and a bare alpha that disagree for the same path -> hard error, no
        // silent pick.
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let path = scratch_file(&tmp, "lora_conflicting_alpha.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.lin.lora_A.weight", &a_raw),
                ("transformer.lin.lora_B.weight", &b_raw),
                ("transformer.lin.alpha", &Array::from_slice(&[4.0f32], &[1])),
                ("lin.alpha", &Array::from_slice(&[8.0f32], &[1])), // disagrees
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight, None),
        };
        assert!(apply_lora_peft(&mut host, &w, 1.0, Some("transformer.")).is_err());
    }

    #[test]
    fn unmatched_paths_are_reported_not_dropped() {
        let tmp = tempfile::tempdir().unwrap();
        // A LoKr file targeting a path the host doesn't have -> applied 0, path reported.
        let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let path = scratch_file(&tmp, "lokr_miss.safetensors");
        Array::save_safetensors(
            vec![
                ("missing.path.lokr_w1", &dummy),
                ("missing.path.lokr_w2", &dummy),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(is_lokr(&w));

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
        };
        let report = apply_lokr(&mut host, &w, 1.0).unwrap();
        assert_eq!(report.applied, 0);
        assert_eq!(report.unmatched_paths, vec!["missing.path".to_string()]);
    }

    // ---- BFL/ComfyUI fused→split LyCORIS (sc-8345) ------------------------------------------------

    /// A host with a BFL surface: the three split attention projections (`to_q/to_k/to_v`, the fused-
    /// qkv destinations) plus a rename destination (`to_out`), and a `bfl_targets()` mapping the fused
    /// `diffusion_model.double_blocks.0.img_attn.{qkv,proj}` BFL names onto them — the minimal shape of
    /// a FLUX `Flux2Transformer` for the LyCORIS fused→split path.
    struct BflHost {
        mods: HashMap<String, AdaptableLinear>,
    }
    impl BflHost {
        fn new() -> Self {
            let mut mods = HashMap::new();
            // qkv splits: each [out=2, in=3] → the fused source is [6, 3].
            for dst in ["to_q", "to_k", "to_v"] {
                mods.insert(
                    format!("transformer_blocks.0.attn.{dst}"),
                    AdaptableLinear::dense(Array::from_slice(&[0.0f32; 6], &[2, 3]), None),
                );
            }
            // rename dest: [out=4, in=4].
            mods.insert(
                "transformer_blocks.0.attn.to_out".to_string(),
                AdaptableLinear::dense(Array::from_slice(&[0.0f32; 16], &[4, 4]), None),
            );
            Self { mods }
        }
    }
    impl AdaptableHost for BflHost {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            self.mods.get_mut(&path.join("."))
        }
        fn adaptable_paths(&self) -> Vec<String> {
            self.mods.keys().cloned().collect()
        }
        fn bfl_targets(&self) -> Vec<BflTarget> {
            let block_keys = |module: &str| {
                (
                    vec![
                        format!("diffusion_model.{module}.lora_B.weight"),
                        format!("diffusion_model.{module}.lora_up.weight"),
                    ],
                    vec![
                        format!("diffusion_model.{module}.lora_A.weight"),
                        format!("diffusion_model.{module}.lora_down.weight"),
                    ],
                    vec![format!("diffusion_model.{module}.alpha")],
                )
            };
            let mut out = Vec::new();
            let (up, down, alpha) = block_keys("double_blocks.0.img_attn.qkv");
            for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
                out.push(BflTarget {
                    target_path: format!("transformer_blocks.0.attn.{dst}"),
                    up_keys: up.clone(),
                    down_keys: down.clone(),
                    alpha_keys: alpha.clone(),
                    up_slice: Some(LoraRowSlice::Chunk {
                        n: 3,
                        index: idx as i32,
                    }),
                    down_slice: Some(LoraRowSlice::ChunkIfDivisible {
                        n: 3,
                        index: idx as i32,
                    }),
                });
            }
            let (up, down, alpha) = block_keys("double_blocks.0.img_attn.proj");
            out.push(BflTarget {
                target_path: "transformer_blocks.0.attn.to_out".to_string(),
                up_keys: up,
                down_keys: down,
                alpha_keys: alpha,
                up_slice: None,
                down_slice: None,
            });
            out
        }
    }

    /// A `networkType=lokr` file in BFL/ComfyUI fused naming (`diffusion_model.…img_attn.qkv` fused,
    /// `…img_attn.proj` renamed) must apply onto a FLUX-shaped host: the fused qkv delta is rebuilt at
    /// `[6,3]` and row-sliced into `to_q/to_k/to_v`, and the proj rename lands whole on `to_out`. Before
    /// sc-8345 every target surfaced as unmatched (the strict apply errored). Exercises the full
    /// `apply_adapters_strict` dispatch, not just the leaf applier.
    #[test]
    fn bfl_named_lokr_fused_qkv_and_rename_resolve() {
        let tmp = tempfile::tempdir().unwrap();
        // Fused qkv LoKr: kron(w1[3,1], w2[2,3]) → [6,3].
        let qkv_w1 = Array::from_slice(&[1.0f32, 0.5, -0.25], &[3, 1]);
        let qkv_w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
        // Proj rename LoKr: kron(w1[2,2], w2[2,2]) → [4,4].
        let proj_w1 = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
        let proj_w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let path = scratch_file(&tmp, "bfl_lokr_fused.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1",
                    &qkv_w1,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2",
                    &qkv_w2,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lokr_w1",
                    &proj_w1,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lokr_w2",
                    &proj_w2,
                ),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();

        let mut host = BflHost::new();
        let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lokr);
        let report =
            apply_adapters_strict(&mut host, std::slice::from_ref(&spec), "flux2_klein_9b")
                .unwrap();
        assert_eq!(report.applied, 4);
        assert!(report.unmatched_paths.is_empty());

        // The fused qkv delta, reconstructed independently at the fused shape, row-sliced into thirds.
        let full = reconstruct_lokr_delta(
            1.0,
            1.0,
            &[6, 3],
            Some(&qkv_w1),
            None,
            None,
            Some(&qkv_w2),
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap();
        for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            let lin = host
                .adaptable_mut(&["transformer_blocks", "0", "attn", dst])
                .unwrap();
            let Adapter::Lokr { delta, scale } = &lin.adapters()[0] else {
                panic!("expected a LoKr adapter on {dst}");
            };
            assert_eq!(*scale, 1.0);
            let start = idx as i32 * 2;
            let want = full.try_index((start..start + 2, ..)).unwrap();
            assert!(
                all_close(delta, &want, 1e-5, 1e-5, false)
                    .unwrap()
                    .item::<bool>(),
                "qkv split {dst} delta mismatch"
            );
        }

        // Proj rename lands whole on to_out.
        let proj_full = reconstruct_lokr_delta(
            1.0,
            1.0,
            &[4, 4],
            Some(&proj_w1),
            None,
            None,
            Some(&proj_w2),
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap();
        let lin = host
            .adaptable_mut(&["transformer_blocks", "0", "attn", "to_out"])
            .unwrap();
        let Adapter::Lokr { delta, .. } = &lin.adapters()[0] else {
            panic!("expected a LoKr adapter on to_out");
        };
        assert!(all_close(delta, &proj_full, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    /// A bare diffusers-named LoKr (split `to_q` directly, no BFL fused name) on the SAME BFL host must
    /// still route through the ordinary `apply_lokr` path, NOT the fused→split one — `is_bfl_lycoris`
    /// keys only off the BFL spellings, so non-BFL LyCORIS is untouched by sc-8345.
    #[test]
    fn bare_diffusers_lokr_on_bfl_host_stays_on_plain_path() {
        let tmp = tempfile::tempdir().unwrap();
        // kron(w1[2,1], w2[1,3]) → [2,3], the shape of the split to_q.
        let w1 = Array::from_slice(&[1.0f32, 0.5], &[2, 1]);
        let w2 = Array::from_slice(&[0.1f32, 0.2, 0.3], &[1, 3]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let path = scratch_file(&tmp, "bare_lokr_on_bfl_host.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer_blocks.0.attn.to_q.lokr_w1", &w1),
                ("transformer_blocks.0.attn.to_q.lokr_w2", &w2),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();
        let mut host = BflHost::new();
        let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lokr);
        let report =
            apply_adapter_specs_autoprefix(&mut host, std::slice::from_ref(&spec)).unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.unmatched_paths.is_empty());
        let lin = host
            .adaptable_mut(&["transformer_blocks", "0", "attn", "to_q"])
            .unwrap();
        assert!(matches!(lin.adapters()[0], Adapter::Lokr { .. }));
    }

    /// A third-party LyCORIS LoKr (no `networkType` stamp — detected by `lokr_*` keys) in BFL fused
    /// naming routes through the fused→split applier too (sc-8345). Both-full factors ⇒ lycoris scale 1.
    #[test]
    fn bfl_named_thirdparty_lokr_fused_qkv_resolves() {
        let tmp = tempfile::tempdir().unwrap();
        // kron(w1[3,1], w2[2,3]) → [6,3]; both factors full ⇒ scale 1.0.
        let qkv_w1 = Array::from_slice(&[1.0f32, 0.5, -0.25], &[3, 1]);
        let qkv_w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
        let proj_w1 = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
        let proj_w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let path = scratch_file(&tmp, "bfl_tp_lokr_fused.safetensors");
        // NO `networkType` metadata → is_lokr() false, is_lokr_keys() true (the third-party path).
        Array::save_safetensors(
            vec![
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1",
                    &qkv_w1,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2",
                    &qkv_w2,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lokr_w1",
                    &proj_w1,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lokr_w2",
                    &proj_w2,
                ),
            ],
            None,
            &path,
        )
        .unwrap();
        let mut host = BflHost::new();
        let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lora); // kind irrelevant — keys route it
        let report =
            apply_adapters_strict(&mut host, std::slice::from_ref(&spec), "flux2_klein_9b")
                .unwrap();
        assert_eq!(report.applied, 4);
        assert!(report.unmatched_paths.is_empty());

        let full = reconstruct_lokr_delta_scaled(
            1.0,
            &[6, 3],
            Some(&qkv_w1),
            None,
            None,
            Some(&qkv_w2),
            None,
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap();
        for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            let lin = host
                .adaptable_mut(&["transformer_blocks", "0", "attn", dst])
                .unwrap();
            let Adapter::Lokr { delta, .. } = &lin.adapters()[0] else {
                panic!("expected a LoKr adapter on {dst}");
            };
            let start = idx as i32 * 2;
            let want = full.try_index((start..start + 2, ..)).unwrap();
            assert!(all_close(delta, &want, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>());
        }
    }

    /// A third-party LyCORIS LoHa (`hada_*` keys) in BFL fused naming routes through the fused→split
    /// applier (sc-8345); the Hadamard delta is rebuilt at the fused shape, then row-sliced.
    #[test]
    fn bfl_named_loha_fused_qkv_resolves() {
        let tmp = tempfile::tempdir().unwrap();
        // (w1_a@w1_b) ⊙ (w2_a@w2_b) at [6,3], rank r=1 ⇒ scale 1.0.
        let w1_a = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[6, 1]);
        let w1_b = Array::from_slice(&[1.0f32, -1.0, 0.5], &[1, 3]);
        let w2_a = Array::from_slice(&[0.6f32, 0.5, 0.4, 0.3, 0.2, 0.1], &[6, 1]);
        let w2_b = Array::from_slice(&[0.2f32, 0.4, -0.2], &[1, 3]);
        let path = scratch_file(&tmp, "bfl_loha_fused.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.hada_w1_a",
                    &w1_a,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.hada_w1_b",
                    &w1_b,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.hada_w2_a",
                    &w2_a,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.hada_w2_b",
                    &w2_b,
                ),
            ],
            None,
            &path,
        )
        .unwrap();
        let mut host = BflHost::new();
        let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lora);
        let report =
            apply_adapters_strict(&mut host, std::slice::from_ref(&spec), "flux2_klein_9b")
                .unwrap();
        // Only the fused qkv is present → the three splits; proj/to_out untouched (no factors for it).
        assert_eq!(report.applied, 3);
        assert!(report.unmatched_paths.is_empty());

        let full = reconstruct_loha_delta(
            1.0,
            &[6, 3],
            &w1_a,
            &w1_b,
            &w2_a,
            &w2_b,
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap();
        for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            let lin = host
                .adaptable_mut(&["transformer_blocks", "0", "attn", dst])
                .unwrap();
            let Adapter::Lokr { delta, .. } = &lin.adapters()[0] else {
                panic!("expected an installed delta on {dst}");
            };
            let start = idx as i32 * 2;
            let want = full.try_index((start..start + 2, ..)).unwrap();
            assert!(all_close(delta, &want, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>());
        }
    }

    /// The load-time connector stacks a mixed LoRA + LoKr spec list and is equivalent to calling
    /// the underlying loaders directly, in order.
    #[test]
    fn apply_specs_stacks_mixed_lora_and_lokr() {
        let tmp = tempfile::tempdir().unwrap();
        // base [out=4, in=2].
        let base_vals: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let weight = Array::from_slice(&base_vals, &[4, 2]);

        // PEFT LoRA file targeting ["lin"]: lora_A [r=2, in=2], lora_B [out=4, r=2].
        let a_raw = Array::from_slice(&[0.1f32, 0.2, -0.1, -0.2], &[2, 2]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let lora_path = scratch_file(&tmp, "specs_lora.safetensors");
        Array::save_safetensors(
            vec![("lin.lora_A.weight", &a_raw), ("lin.lora_B.weight", &b_raw)],
            None,
            &lora_path,
        )
        .unwrap();

        // LoKr file targeting ["lin"]: kron(w1[2,1], w2[2,2]) -> [4,2]; alpha==rank -> factor 1.
        let w1 = Array::from_slice(&[1.0f32, 0.5], &[2, 1]);
        let w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let lokr_path = scratch_file(&tmp, "specs_lokr.safetensors");
        Array::save_safetensors(
            vec![("lin.lokr_w1", &w1), ("lin.lokr_w2", &w2)],
            Some(&meta),
            &lokr_path,
        )
        .unwrap();

        let specs = vec![
            AdapterSpec {
                path: lora_path.clone(),
                scale: 0.5,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            },
            AdapterSpec {
                path: lokr_path.clone(),
                scale: 1.0,
                kind: AdapterKind::Lokr,
                pass_scales: None,
                moe_expert: None,
            },
        ];

        let mut via_specs = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_adapter_specs(&mut via_specs, &specs, None).unwrap();
        assert_eq!(report.applied, 2);
        assert!(report.unmatched_paths.is_empty());

        // Reference: the same files through the underlying loaders directly, in order.
        let mut via_loaders = OneLinear {
            lin: AdaptableLinear::dense(weight, None),
        };
        apply_lora_peft(
            &mut via_loaders,
            &Weights::from_file(&lora_path).unwrap(),
            0.5,
            None,
        )
        .unwrap();
        apply_lokr(
            &mut via_loaders,
            &Weights::from_file(&lokr_path).unwrap(),
            1.0,
        )
        .unwrap();

        let x = Array::from_slice(&[1.0f32, -2.0], &[1, 2]);
        let got = via_specs.lin.forward(&x).unwrap();
        let want = via_loaders.lin.forward(&x).unwrap();
        assert!(all_close(&got, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());

        // Both adapters actually moved the output off the bare base.
        let base = AdaptableLinear::dense(Array::from_slice(&base_vals, &[4, 2]), None)
            .forward(&x)
            .unwrap();
        assert!(!all_close(&got, &base, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn apply_specs_empty_is_noop() {
        let weight = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_adapter_specs(&mut host, &[], None).unwrap();
        assert_eq!(report, ApplyReport::default());

        let x = Array::from_slice(&[1.0f32, -1.0], &[1, 2]);
        let got = host.lin.forward(&x).unwrap();
        let want = AdaptableLinear::dense(weight, None).forward(&x).unwrap();
        assert!(all_close(&got, &want, 1e-6, 1e-6, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn apply_specs_reports_unmatched_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let path = scratch_file(&tmp, "specs_miss.safetensors");
        Array::save_safetensors(
            vec![("nope.here.lokr_w1", &dummy), ("nope.here.lokr_w2", &dummy)],
            Some(&meta),
            &path,
        )
        .unwrap();

        let specs = vec![AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lokr,
            pass_scales: None,
            moe_expert: None,
        }];
        let mut host = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
        };
        let report = apply_adapter_specs(&mut host, &specs, None).unwrap();
        assert_eq!(report.applied, 0);
        assert_eq!(report.unmatched_paths, vec!["nope.here".to_string()]);
    }

    #[test]
    fn apply_specs_kind_metadata_mismatch_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        let path = scratch_file(&tmp, "specs_mismatch.safetensors");
        Array::save_safetensors(vec![("lin.lokr_w1", &dummy)], Some(&meta), &path).unwrap();

        // Declared Lora but the file's metadata says LoKr -> a loud error, not a silent no-op.
        let specs = vec![AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }];
        let mut host = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
        };
        assert!(apply_adapter_specs(&mut host, &specs, None).is_err());
    }

    #[test]
    fn detect_lora_prefix_variants() {
        let tmp = tempfile::tempdir().unwrap();
        let a = Array::from_slice(&[0.0f32], &[1, 1]);
        let bare = scratch_file(&tmp, "detect_bare.safetensors");
        Array::save_safetensors(vec![("lin.lora_A.weight", &a)], None, &bare).unwrap();
        assert_eq!(
            detect_lora_prefix(&Weights::from_file(&bare).unwrap()),
            None
        );

        let tf = scratch_file(&tmp, "detect_tf.safetensors");
        Array::save_safetensors(vec![("transformer.lin.lora_A.weight", &a)], None, &tf).unwrap();
        assert_eq!(
            detect_lora_prefix(&Weights::from_file(&tf).unwrap()),
            Some("transformer.")
        );

        let dm = scratch_file(&tmp, "detect_dm.safetensors");
        Array::save_safetensors(vec![("diffusion_model.lin.lora_A.weight", &a)], None, &dm)
            .unwrap();
        assert_eq!(
            detect_lora_prefix(&Weights::from_file(&dm).unwrap()),
            Some("diffusion_model.")
        );
    }

    #[test]
    fn autoprefix_strips_detected_prefix_and_applies() {
        let tmp = tempfile::tempdir().unwrap();
        // base [out=2, in=2]; a `transformer.`-prefixed peft LoRA on path ["lin"].
        let weight = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let a = Array::from_slice(&[0.1f32, 0.2, -0.1, -0.2], &[2, 2]); // [r=2, in=2]
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]); // [out=2, r=2]
        let path = scratch_file(&tmp, "autoprefix_lora.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.lin.lora_A.weight", &a),
                ("transformer.lin.lora_B.weight", &b),
            ],
            None,
            &path,
        )
        .unwrap();

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight, None),
        };
        let specs = vec![AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }];
        let report = apply_adapter_specs_autoprefix(&mut host, &specs).unwrap();
        assert_eq!(
            report.applied, 1,
            "transformer.-prefixed key should resolve to lin"
        );
        assert!(report.unmatched_paths.is_empty());

        // Strict wrapper: a bare-but-unmatched target errors rather than silently dropping.
        let miss = scratch_file(&tmp, "autoprefix_miss.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.nope.lora_A.weight", &a),
                ("transformer.nope.lora_B.weight", &b),
            ],
            None,
            &miss,
        )
        .unwrap();
        let mut host2 = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]), None),
        };
        let specs2 = vec![AdapterSpec {
            path: miss,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }];
        assert!(apply_adapters_strict(&mut host2, &specs2, "test").is_err());
    }

    /// sc-2909: a diffusers/ComfyUI LoRA spelled with `lora_down`/`lora_up` factor suffixes on a
    /// **dotted, un-prefixed** path (the lightx2v Qwen-Image-Lightning format) routes through the
    /// PEFT loader (no `lora_unet_` prefix → not kohya) and installs the BYTE-IDENTICAL adapter to
    /// its `lora_A`/`lora_B` twin — and `apply_adapter_specs_autoprefix` resolves it end-to-end.
    #[test]
    fn diffusers_lora_down_up_equals_peft_ab() {
        let tmp = tempfile::tempdir().unwrap();
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]); // [r, in]
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]); // [out, r]
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        // down==A, up==B, bare alpha, no namespace prefix — exactly the lightx2v Lightning spelling.
        let down_path = scratch_file(&tmp, "diffusers_down_up.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_down.weight", &a_raw),
                ("lin.lora_up.weight", &b_raw),
                ("lin.alpha", &alpha),
            ],
            None,
            &down_path,
        )
        .unwrap();
        // Detected as un-prefixed (not kohya, not BFL) and resolved through the strict seam.
        let w = Weights::from_file(&down_path).unwrap();
        assert!(!is_kohya(&w), "dotted-path lora_down is NOT kohya");
        assert_eq!(detect_lora_prefix(&w), None, "no namespace prefix");

        let mut via_down = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_adapter_specs_autoprefix(
            &mut via_down,
            &[AdapterSpec::new(down_path, 0.5, AdapterKind::Lora)],
        )
        .unwrap();
        assert_eq!(report.applied, 1, "lora_down/up resolved to lin");
        assert!(report.unmatched_paths.is_empty());

        // The `lora_A`/`lora_B` twin must install the identical adapter.
        let ab_path = scratch_file(&tmp, "diffusers_ab_twin.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_A.weight", &a_raw),
                ("lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha),
            ],
            None,
            &ab_path,
        )
        .unwrap();
        let mut via_ab = OneLinear {
            lin: AdaptableLinear::dense(weight, None),
        };
        apply_lora_peft(
            &mut via_ab,
            &Weights::from_file(&ab_path).unwrap(),
            0.5,
            None,
        )
        .unwrap();

        let pull = |h: &mut OneLinear| match h.adaptable_mut(&["lin"]).unwrap().adapters() {
            [Adapter::Lora { a, b, scale }] => (a.clone(), b.clone(), *scale),
            _ => panic!("expected one LoRA"),
        };
        let (da, db, ds) = pull(&mut via_down);
        let (pa, pb, ps) = pull(&mut via_ab);
        assert_eq!(ds, ps);
        assert!(
            array_eq(&da, &pa, false).unwrap().item::<bool>()
                && array_eq(&db, &pb, false).unwrap().item::<bool>(),
            "lora_down/up and lora_A/B installed different adapters"
        );
    }

    // ---- kohya `lora_unet_` LoRA (sc-2618) ----

    /// Two modules whose flattened kohya stems are ambiguous under a blind `_`→`.` split: the
    /// segment `to_out.0` and the segment name `feed_forward` both contain the separator char.
    fn kohya_two_module_host() -> MultiHost {
        let w_out = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let w_ff = Array::from_slice(
            &(0..15).map(|i| i as f32 * 0.07).collect::<Vec<_>>(),
            &[5, 3],
        );
        MultiHost::new(&[
            ("blocks.0.attn.to_out.0", w_out),
            ("blocks.0.feed_forward.w1", w_ff),
        ])
    }

    /// The same (down, up, alpha) factors written in BOTH conventions and applied through the
    /// provider seam must yield byte-identical adapters — a kohya file is interchangeable with its
    /// PEFT twin. This is the sc-2618 gate at the core level (no model weights needed).
    #[test]
    fn kohya_equiv_to_peft_bit_exact() {
        let tmp = tempfile::tempdir().unwrap();
        // out=4/in=3 and out=5/in=3, rank=2; alpha=4 (≠ rank → exercises the alpha/rank fold).
        let a_out = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]); // [r,in]
        let b_out = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]); // [out,r]
        let a_ff = Array::from_slice(&[0.05f32, -0.15, 0.2, 0.3, -0.25, 0.1], &[2, 3]);
        let b_ff = Array::from_slice(
            &[0.2f32, -0.2, 0.1, 0.3, -0.1, 0.4, 0.15, -0.35, 0.05, 0.25],
            &[5, 2],
        );
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        let kohya_path = scratch_file(&tmp, "equiv_kohya.safetensors");
        Array::save_safetensors(
            vec![
                ("lora_unet_blocks_0_attn_to_out_0.lora_down.weight", &a_out),
                ("lora_unet_blocks_0_attn_to_out_0.lora_up.weight", &b_out),
                ("lora_unet_blocks_0_attn_to_out_0.alpha", &alpha),
                ("lora_unet_blocks_0_feed_forward_w1.lora_down.weight", &a_ff),
                ("lora_unet_blocks_0_feed_forward_w1.lora_up.weight", &b_ff),
                ("lora_unet_blocks_0_feed_forward_w1.alpha", &alpha),
            ],
            None as Option<&HashMap<String, String>>,
            &kohya_path,
        )
        .unwrap();

        let peft_path = scratch_file(&tmp, "equiv_peft.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.blocks.0.attn.to_out.0.lora_A.weight", &a_out),
                ("transformer.blocks.0.attn.to_out.0.lora_B.weight", &b_out),
                ("transformer.blocks.0.attn.to_out.0.alpha", &alpha),
                ("transformer.blocks.0.feed_forward.w1.lora_A.weight", &a_ff),
                ("transformer.blocks.0.feed_forward.w1.lora_B.weight", &b_ff),
                ("transformer.blocks.0.feed_forward.w1.alpha", &alpha),
            ],
            None as Option<&HashMap<String, String>>,
            &peft_path,
        )
        .unwrap();

        let mut via_kohya = kohya_two_module_host();
        let rep_k = apply_adapters_strict(
            &mut via_kohya,
            &[AdapterSpec {
                path: kohya_path,
                scale: 0.75,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
            "test",
        )
        .unwrap();
        assert_eq!(rep_k.applied, 2, "both kohya modules resolve");

        let mut via_peft = kohya_two_module_host();
        apply_adapters_strict(
            &mut via_peft,
            &[AdapterSpec {
                path: peft_path,
                scale: 0.75,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
            "test",
        )
        .unwrap();

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        for p in ["blocks.0.attn.to_out.0", "blocks.0.feed_forward.w1"] {
            let gk = via_kohya.mods.get(p).unwrap().forward(&x).unwrap();
            let gp = via_peft.mods.get(p).unwrap().forward(&x).unwrap();
            assert!(
                array_eq(&gk, &gp, false).unwrap().item::<bool>(),
                "kohya and peft adapters diverged at {p}"
            );
            // And both actually moved off the bare base.
            let base = AdaptableLinear::dense(
                via_kohya
                    .mods
                    .get(p)
                    .unwrap()
                    .dense_weight()
                    .unwrap()
                    .0
                    .clone(),
                None,
            )
            .forward(&x)
            .unwrap();
            assert!(
                !array_eq(&gk, &base, false).unwrap().item::<bool>(),
                "adapter at {p} was a no-op"
            );
        }
    }

    /// The flattened stem `blocks_0_feed_forward_w1` must resolve to `blocks.0.feed_forward.w1`
    /// (the table), NOT the blind split `blocks.0.feed.forward.w1` — proving the disambiguation does
    /// real work.
    #[test]
    fn kohya_table_disambiguates_underscore_segment_names() {
        let mut host = kohya_two_module_host();
        // The blind `_`→`.` split target does not exist; the correct dotted path does.
        assert!(host
            .adaptable_mut(&["blocks", "0", "feed", "forward", "w1"])
            .is_none());
        assert!(host
            .adaptable_mut(&["blocks", "0", "feed_forward", "w1"])
            .is_some());

        let table = kohya_table(&host.adaptable_paths());
        assert_eq!(
            table.get("blocks_0_feed_forward_w1").map(String::as_str),
            Some("blocks.0.feed_forward.w1")
        );
        assert_eq!(
            table.get("blocks_0_attn_to_out_0").map(String::as_str),
            Some("blocks.0.attn.to_out.0")
        );
    }

    /// A `lora_unet_` key whose stem is off-surface (e.g. FLUX.2 BFL `double_blocks_*`, sc-2743) is
    /// surfaced in `unmatched_paths` and fails the strict policy — loud, never silently dropped.
    #[test]
    fn kohya_offsurface_stem_surfaced_and_strict_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let path = scratch_file(&tmp, "kohya_offsurface.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight",
                    &a,
                ),
                ("lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight", &b),
            ],
            None as Option<&HashMap<String, String>>,
            &path,
        )
        .unwrap();

        let mut host = kohya_two_module_host();
        let table = kohya_table(&host.adaptable_paths());
        let report =
            apply_lora_kohya(&mut host, &Weights::from_file(&path).unwrap(), 1.0, &table).unwrap();
        assert_eq!(report.applied, 0);
        assert_eq!(
            report.unmatched_paths,
            vec!["double_blocks_0_img_attn_qkv".to_string()]
        );

        // Through the strict provider seam it is a hard error.
        let mut host2 = kohya_two_module_host();
        assert!(apply_adapters_strict(
            &mut host2,
            &[AdapterSpec {
                path,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
            "test",
        )
        .is_err());
    }

    /// A kohya adapter at `scale = 0` is a bit-exact no-op (the scale-0 invariant), and `is_kohya`
    /// detects the format.
    #[test]
    fn kohya_scale_zero_is_bit_exact_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let path = scratch_file(&tmp, "kohya_scale0.safetensors");
        Array::save_safetensors(
            vec![
                ("lora_unet_blocks_0_attn_to_out_0.lora_down.weight", &a),
                ("lora_unet_blocks_0_attn_to_out_0.lora_up.weight", &b),
            ],
            None as Option<&HashMap<String, String>>,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(is_kohya(&w));

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let mut host = kohya_two_module_host();
        let base = host
            .mods
            .get("blocks.0.attn.to_out.0")
            .unwrap()
            .forward(&x)
            .unwrap();
        let table = kohya_table(&host.adaptable_paths());
        apply_lora_kohya(&mut host, &w, 0.0, &table).unwrap();
        let out = host
            .mods
            .get("blocks.0.attn.to_out.0")
            .unwrap()
            .forward(&x)
            .unwrap();
        assert!(array_eq(&out, &base, false).unwrap().item::<bool>());
    }

    // ---- BFL / ComfyUI fused→split LoRA (sc-2743) ----

    /// The [`LoraRowSlice`] variants are byte-faithful to the fork's `LoraTransforms`. Expected values
    /// are pinned to the mflux venv (`LoraTransforms.split_*` on the same inputs, sc-2743): up always
    /// slices, down is shared unless the rank is divisible, and the qkv-mlp `dims` slice matches the
    /// `[3072,3072,3072,12288]` boundaries.
    #[test]
    fn lora_row_slice_matches_fork_transforms() {
        // arange(6,2): split_q_up = rows[0:2], split_v_up = rows[4:6].
        let t6 = Array::from_slice(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[6, 2]);
        let q_up = LoraRowSlice::Chunk { n: 3, index: 0 }.apply(&t6).unwrap();
        assert_eq!(q_up.shape(), &[2, 2]);
        assert_eq!(q_up.as_slice::<f32>(), &[0.0, 1.0, 2.0, 3.0]);
        let v_up = LoraRowSlice::Chunk { n: 3, index: 2 }.apply(&t6).unwrap();
        assert_eq!(v_up.as_slice::<f32>(), &[8.0, 9.0, 10.0, 11.0]);

        // down: ChunkIfDivisible — whole when rank%3!=0 (the usual fused-qkv LoRA), sliced when ==0.
        let d4 = Array::from_slice(&(0..8).map(|i| i as f32).collect::<Vec<_>>(), &[4, 2]);
        let d4q = LoraRowSlice::ChunkIfDivisible { n: 3, index: 0 }
            .apply(&d4)
            .unwrap();
        assert_eq!(d4q.shape(), &[4, 2], "rank 4 not ÷3 → shared whole");
        assert_eq!(d4q.as_slice::<f32>(), d4.as_slice::<f32>());
        let d6q = LoraRowSlice::ChunkIfDivisible { n: 3, index: 0 }
            .apply(&t6)
            .unwrap();
        assert_eq!(
            d6q.as_slice::<f32>(),
            &[0.0, 1.0, 2.0, 3.0],
            "rank 6 ÷3 → sliced"
        );

        // F-060: `Chunk` ALWAYS slices (vs `ChunkIfDivisible`'s fall-through), so a non-divisible
        // rows count must be rejected instead of silently truncating the fused qkv delta.
        let err = LoraRowSlice::Chunk { n: 3, index: 0 }
            .apply(&d4)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not divisible"), "{err}");

        // qkv-mlp up `dims` (FLUX.1 `linear1`): q = rows[0:3072], mlp = rows[9216:21504].
        let dims = vec![3072, 3072, 3072, 12288];
        let total: i32 = dims.iter().sum();
        let big = Array::from_slice(
            &(0..total).map(|i| i as f32).collect::<Vec<_>>(),
            &[total, 1],
        );
        let q = LoraRowSlice::Dims {
            dims: dims.clone(),
            index: 0,
        }
        .apply(&big)
        .unwrap();
        assert_eq!(q.shape(), &[3072, 1]);
        assert_eq!(q.as_slice::<f32>()[0], 0.0);
        let mlp = LoraRowSlice::Dims {
            dims: dims.clone(),
            index: 3,
        }
        .apply(&big)
        .unwrap();
        assert_eq!(mlp.shape(), &[12288, 1]);
        assert_eq!(mlp.as_slice::<f32>()[0], 9216.0);
    }

    /// A host with three separate per-head linears at `blk.attn.to_{q,k,v}` (`[inner,in]` each).
    fn three_qkv_host(inner: i32, inp: i32) -> MultiHost {
        let zeros = || Array::from_slice(&vec![0.0f32; (inner * inp) as usize], &[inner, inp]);
        MultiHost::new(&[
            ("blk.attn.to_q", zeros()),
            ("blk.attn.to_k", zeros()),
            ("blk.attn.to_v", zeros()),
        ])
    }

    /// The sc-2743 gate at the core level: a BFL *fused* qkv LoRA, split via [`apply_lora_bfl`],
    /// installs the BYTE-IDENTICAL adapter at each of `to_q/to_k/to_v` as the equivalent *diffusers
    /// split-target* LoRA (the fork-verified PEFT path). The fused up `[3·inner, r]` is row-sliced into
    /// per-head `[inner, r]`; the down `[r, in]` (rank not ÷3) is shared. No model weights needed.
    #[test]
    fn bfl_fused_qkv_equals_diffusers_split() {
        let tmp = tempfile::tempdir().unwrap();
        let (inner, inp, r) = (4i32, 3i32, 2i32);
        // Per-head up factors, then the fused up = their dim-0 concat (row-major, so flat concat).
        let bq: Vec<f32> = (0..inner * r)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
            .collect();
        let bk: Vec<f32> = (0..inner * r)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.02)
            .collect();
        let bv: Vec<f32> = (0..inner * r)
            .map(|i| ((i % 3) as f32 - 1.0) * 0.03)
            .collect();
        let mut fused = Vec::new();
        fused.extend_from_slice(&bq);
        fused.extend_from_slice(&bk);
        fused.extend_from_slice(&bv);
        let b_fused = Array::from_slice(&fused, &[3 * inner, r]);
        let b_q = Array::from_slice(&bq, &[inner, r]);
        let b_k = Array::from_slice(&bk, &[inner, r]);
        let b_v = Array::from_slice(&bv, &[inner, r]);
        // Shared down [r, in] (rank 2 not ÷3 → shared across q/k/v) + alpha ≠ rank.
        let a = Array::from_slice(
            &(0..r * inp)
                .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
                .collect::<Vec<_>>(),
            &[r, inp],
        );
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        // BFL file: one fused qkv linear (kohya `lora_unet_` spelling).
        let up_key = "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight";
        let down_key = "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight";
        let alpha_key = "lora_unet_double_blocks_0_img_attn_qkv.alpha";
        let bfl_path = scratch_file(&tmp, "bfl_qkv.safetensors");
        Array::save_safetensors(
            vec![(up_key, &b_fused), (down_key, &a), (alpha_key, &alpha)],
            None as Option<&HashMap<String, String>>,
            &bfl_path,
        )
        .unwrap();
        let wb = Weights::from_file(&bfl_path).unwrap();

        let mk = |idx: i32, tgt: &str| BflTarget {
            target_path: tgt.to_string(),
            up_keys: vec![up_key.to_string()],
            down_keys: vec![down_key.to_string()],
            alpha_keys: vec![alpha_key.to_string()],
            up_slice: Some(LoraRowSlice::Chunk { n: 3, index: idx }),
            down_slice: Some(LoraRowSlice::ChunkIfDivisible { n: 3, index: idx }),
        };
        let targets = vec![
            mk(0, "blk.attn.to_q"),
            mk(1, "blk.attn.to_k"),
            mk(2, "blk.attn.to_v"),
        ];

        let mut host_bfl = three_qkv_host(inner, inp);
        let rep = apply_lora_bfl(&mut host_bfl, &wb, 0.7, &targets).unwrap();
        assert_eq!(rep.applied, 3, "all three split targets installed");
        assert!(rep.unmatched_paths.is_empty());

        // Equivalent diffusers split-target file: per-head up, SHARED down, same alpha.
        let peft_path = scratch_file(&tmp, "bfl_split_peft.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.blk.attn.to_q.lora_B.weight", &b_q),
                ("transformer.blk.attn.to_q.lora_A.weight", &a),
                ("transformer.blk.attn.to_q.alpha", &alpha),
                ("transformer.blk.attn.to_k.lora_B.weight", &b_k),
                ("transformer.blk.attn.to_k.lora_A.weight", &a),
                ("transformer.blk.attn.to_k.alpha", &alpha),
                ("transformer.blk.attn.to_v.lora_B.weight", &b_v),
                ("transformer.blk.attn.to_v.lora_A.weight", &a),
                ("transformer.blk.attn.to_v.alpha", &alpha),
            ],
            None as Option<&HashMap<String, String>>,
            &peft_path,
        )
        .unwrap();
        let wp = Weights::from_file(&peft_path).unwrap();
        let mut host_peft = three_qkv_host(inner, inp);
        apply_lora_peft(&mut host_peft, &wp, 0.7, Some("transformer.")).unwrap();

        for p in ["blk.attn.to_q", "blk.attn.to_k", "blk.attn.to_v"] {
            let pull = |h: &MultiHost| match h.mods.get(p).unwrap().adapters() {
                [Adapter::Lora { a, b, scale }] => (a.clone(), b.clone(), *scale),
                _ => panic!("expected one LoRA at {p}"),
            };
            let (ba, bb, bs) = pull(&host_bfl);
            let (pa, pb, ps) = pull(&host_peft);
            assert_eq!(bs, ps, "scale differs at {p}");
            assert!(
                array_eq(&ba, &pa, false).unwrap().item::<bool>()
                    && array_eq(&bb, &pb, false).unwrap().item::<bool>(),
                "BFL split and diffusers split installed different adapters at {p}"
            );
        }
    }

    /// `is_bfl` detects a BFL file; an off-surface adapter-shaped key is surfaced (not dropped) while a
    /// bundled text-encoder key is ignored; and a scale-0 BFL adapter is a bit-exact no-op.
    #[test]
    fn bfl_detection_unmatched_and_scale_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let up = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
            &[4, 2],
        );
        let down = Array::from_slice(&[0.1f32, 0.2, -0.1, -0.2, 0.3, -0.3], &[2, 3]);
        let targets = vec![BflTarget {
            target_path: "blk.attn.to_out".to_string(),
            up_keys: vec!["lora_unet_double_blocks_0_img_attn_proj.lora_up.weight".to_string()],
            down_keys: vec!["lora_unet_double_blocks_0_img_attn_proj.lora_down.weight".to_string()],
            alpha_keys: vec![],
            up_slice: None,
            down_slice: None,
        }];

        let path = scratch_file(&tmp, "bfl_detect.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight",
                    &up,
                ),
                (
                    "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight",
                    &down,
                ),
                // off-surface BFL key (no target) → surfaced, not silently dropped.
                (
                    "lora_unet_double_blocks_9_img_attn_proj.lora_up.weight",
                    &up,
                ),
                (
                    "lora_unet_double_blocks_9_img_attn_proj.lora_down.weight",
                    &down,
                ),
                // bundled text-encoder key → ignored (out of denoiser namespace).
                ("lora_te_text_model_layer_0.lora_up.weight", &up),
            ],
            None as Option<&HashMap<String, String>>,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(is_bfl(&w, &targets), "a BFL source key marks the file BFL");

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let mut host = MultiHost::new(&[(
            "blk.attn.to_out",
            Array::from_slice(
                &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
                &[4, 3],
            ),
        )]);
        let base = host
            .mods
            .get("blk.attn.to_out")
            .unwrap()
            .forward(&x)
            .unwrap();

        // scale 0 → bit-exact no-op; the off-surface block-9 key is surfaced, the TE key ignored.
        let rep = apply_lora_bfl(&mut host, &w, 0.0, &targets).unwrap();
        assert_eq!(rep.applied, 1, "the on-surface proj target installed");
        // Both block-9 keys (up + down) are surfaced (sorted: down < up); the `lora_te_` key ignored.
        assert_eq!(
            rep.unmatched_paths,
            vec![
                "lora_unet_double_blocks_9_img_attn_proj.lora_down.weight".to_string(),
                "lora_unet_double_blocks_9_img_attn_proj.lora_up.weight".to_string(),
            ],
            "off-surface BFL keys surfaced; TE key ignored"
        );
        let out = host
            .mods
            .get("blk.attn.to_out")
            .unwrap()
            .forward(&x)
            .unwrap();
        assert!(
            array_eq(&out, &base, false).unwrap().item::<bool>(),
            "scale-0 BFL adapter must be a bit-exact no-op"
        );
    }

    /// sc-3642: a third-party (non-peft / lycoris) LoKr reconstructs the SAME per-module delta the
    /// `lycoris` library produces. Fixtures (real lycoris adapters + ground-truth deltas) come from
    /// `tools/sc3642_lokr_reference.py` via `~/mlx-flux-venv` — the on-device A/B. Covers the four
    /// shapes: full-w1 + decomposed-w2, both-decomposed, both-full (scale forced to 1), and conv
    /// `lokr_t2` tucker.
    #[test]
    fn thirdparty_lokr_matches_lycoris_reference() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3642_lokr");
        for name in [
            "linear_w1full_w2lr",
            "linear_bothlr",
            "linear_bothfull",
            "conv_tucker",
        ] {
            let w = Weights::from_file(dir.join(format!("{name}.safetensors"))).unwrap();
            let exp = Weights::from_file(dir.join(format!("{name}.expected.safetensors"))).unwrap();
            assert!(is_lokr_keys(&w), "{name}: not detected as LoKr by keys");
            assert!(
                !is_lokr(&w),
                "{name}: a third-party file has no networkType metadata"
            );

            // Reconstruct the flattened→dotted table from the expected (dotted) module paths.
            let table: BTreeMap<String, String> = exp
                .keys()
                .map(|d| (d.replace('.', "_"), d.to_string()))
                .collect();
            let groups = parse_lokr_thirdparty(&w).unwrap();
            assert!(!groups.is_empty(), "{name}: parsed no LoKr modules");
            for (raw, g) in &groups {
                let dotted = resolve_lokr_path(raw, &table)
                    .unwrap_or_else(|| panic!("{name}: cannot resolve raw key {raw:?}"));
                let want = exp.require(dotted).unwrap();
                // Reconstruct in f32 (lycoris computes f32) and compare to the ground truth.
                let got = g.delta(want.shape(), Dtype::Float32).unwrap();
                assert_eq!(
                    got.shape(),
                    want.shape(),
                    "{name}/{dotted}: reconstructed delta shape mismatch"
                );
                assert!(
                    all_close(&got, want, 1e-4, 1e-5, false)
                        .unwrap()
                        .item::<bool>(),
                    "{name}/{dotted}: reconstructed LoKr delta diverged from the lycoris reference"
                );
            }
        }
    }

    /// sc-3642: the third-party LoKr installs through the autoprefix dispatch even when the caller
    /// labels the spec `AdapterKind::Lora` (a third-party file carries no `networkType` to tell the
    /// caller otherwise) — detection-by-keys routes it to `apply_lokr_thirdparty`, resolving the
    /// `lycoris_`-prefixed flattened key to the host's dotted module.
    #[test]
    fn thirdparty_lokr_routes_and_installs_via_autoprefix() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3642_lokr");
        let exp = Weights::from_file(dir.join("linear_w1full_w2lr.expected.safetensors")).unwrap();
        // Host with the one Linear the fixture targets (dotted path "proj"), sized to the delta.
        let delta_shape = exp.require("proj").unwrap().shape().to_vec();
        let base = Array::zeros::<f32>(&delta_shape).unwrap();
        let mut host = MultiHost::new(&[("proj", base)]);
        let spec = AdapterSpec::new(
            dir.join("linear_w1full_w2lr.safetensors"),
            1.0,
            AdapterKind::Lora, // deliberately mislabeled — detection must override
        );
        let report =
            apply_adapter_specs_autoprefix(&mut host, std::slice::from_ref(&spec)).unwrap();
        assert_eq!(report.applied, 1, "third-party LoKr was not installed");
        assert!(
            report.unmatched_paths.is_empty(),
            "unexpected unmatched: {:?}",
            report.unmatched_paths
        );
    }

    /// sc-3643: a third-party (non-peft / lycoris) LoHa reconstructs the SAME per-module delta the
    /// `lycoris` library produces. Fixtures from `tools/sc3643_loha_reference.py` via `~/mlx-flux-venv`.
    /// Covers linear, conv (kernel folded into the factors), and conv `hada_t1/t2` tucker.
    #[test]
    fn thirdparty_loha_matches_lycoris_reference() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3643_loha");
        for name in ["linear", "conv_notucker", "conv_tucker"] {
            let w = Weights::from_file(dir.join(format!("{name}.safetensors"))).unwrap();
            let exp = Weights::from_file(dir.join(format!("{name}.expected.safetensors"))).unwrap();
            assert!(is_loha_keys(&w), "{name}: not detected as LoHa by keys");
            assert!(!is_lokr_keys(&w), "{name}: must not look like LoKr");
            assert!(
                !is_lokr(&w),
                "{name}: a third-party file has no networkType metadata"
            );

            let table: BTreeMap<String, String> = exp
                .keys()
                .map(|d| (d.replace('.', "_"), d.to_string()))
                .collect();
            let groups = parse_loha_thirdparty(&w).unwrap();
            assert!(!groups.is_empty(), "{name}: parsed no LoHa modules");
            for (raw, g) in &groups {
                let dotted = resolve_lokr_path(raw, &table)
                    .unwrap_or_else(|| panic!("{name}: cannot resolve raw key {raw:?}"));
                let want = exp.require(dotted).unwrap();
                let got = g.delta(want.shape(), Dtype::Float32).unwrap();
                assert_eq!(
                    got.shape(),
                    want.shape(),
                    "{name}/{dotted}: reconstructed delta shape mismatch"
                );
                assert!(
                    all_close(&got, want, 1e-4, 1e-5, false)
                        .unwrap()
                        .item::<bool>(),
                    "{name}/{dotted}: reconstructed LoHa delta diverged from the lycoris reference"
                );
            }
        }
    }

    /// sc-3643: a third-party LoHa installs through the autoprefix dispatch even when the caller
    /// labels the spec `AdapterKind::Lora` — detection-by-keys routes it to `apply_loha_thirdparty`.
    #[test]
    fn thirdparty_loha_routes_and_installs_via_autoprefix() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3643_loha");
        let exp = Weights::from_file(dir.join("linear.expected.safetensors")).unwrap();
        let delta_shape = exp.require("proj").unwrap().shape().to_vec();
        let base = Array::zeros::<f32>(&delta_shape).unwrap();
        let mut host = MultiHost::new(&[("proj", base)]);
        let spec = AdapterSpec::new(dir.join("linear.safetensors"), 1.0, AdapterKind::Lora);
        let report =
            apply_adapter_specs_autoprefix(&mut host, std::slice::from_ref(&spec)).unwrap();
        assert_eq!(report.applied, 1, "third-party LoHa was not installed");
        assert!(
            report.unmatched_paths.is_empty(),
            "unexpected unmatched: {:?}",
            report.unmatched_paths
        );
    }

    /// F-012: the fixed-prefix `apply_adapter_specs` must route a keys-only third-party LoKr
    /// declared `AdapterKind::Lokr` to `apply_lokr_thirdparty` — the peft applier's global-metadata
    /// parse ignores the per-module `.alpha` (and a tucker `lokr_t2`), so it would install the
    /// delta at the wrong scale while reporting success.
    #[test]
    fn fixed_prefix_lokr_kind_routes_keys_only_file_to_thirdparty() {
        let tmp = tempfile::tempdir().unwrap();
        // Keys-only LoKr (no `networkType` stamp): w1 full [2,2], w2 decomposed [2,r=2]·[2,2], and
        // a per-module alpha=1 over rank 2 ⇒ lycoris scale 0.5 — discriminating, because the peft
        // parse drops the `.alpha` tensor and defaults alpha=rank ⇒ scale 1.
        let base = Array::from_slice(
            &(0..16).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 4],
        );
        let w1 = Array::from_slice(&[0.5f32, 0.6, 0.7, 0.8], &[2, 2]);
        let w2_a = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let w2_b = Array::from_slice(&[0.4f32, 0.3, 0.2, 0.1], &[2, 2]);
        let alpha = Array::from_slice(&[1.0f32], &[1]);
        let path = scratch_file(&tmp, "f012_keysonly_lokr.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lokr_w1", &w1),
                ("lin.lokr_w2_a", &w2_a),
                ("lin.lokr_w2_b", &w2_b),
                ("lin.alpha", &alpha),
            ],
            None as Option<&HashMap<String, String>>,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(
            !is_lokr(&w) && is_lokr_keys(&w),
            "premise: a keys-only third-party file"
        );

        // Reference: the third-party applier called directly (per-module alpha honored).
        let mut want = OneLinear {
            lin: AdaptableLinear::dense(base.clone(), None),
        };
        apply_lokr_thirdparty(&mut want, &w, 1.0).unwrap();

        let mut got = OneLinear {
            lin: AdaptableLinear::dense(base.clone(), None),
        };
        let report = apply_adapter_specs(
            &mut got,
            &[AdapterSpec::new(path, 1.0, AdapterKind::Lokr)],
            None,
        )
        .unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.unmatched_paths.is_empty());

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5, 0.25], &[1, 4]);
        let got_y = got.lin.forward(&x).unwrap();
        let want_y = want.lin.forward(&x).unwrap();
        assert!(
            array_eq(&got_y, &want_y, false).unwrap().item::<bool>(),
            "fixed-prefix Lokr dispatch must install the identical third-party delta"
        );
        // The pre-F-012 route: the peft applier drops the per-module alpha (scale 1, not 0.5).
        let mut peft = OneLinear {
            lin: AdaptableLinear::dense(base, None),
        };
        apply_lokr(&mut peft, &w, 1.0).unwrap();
        let peft_y = peft.lin.forward(&x).unwrap();
        assert!(
            !all_close(&got_y, &peft_y, 1e-4, 1e-4, false)
                .unwrap()
                .item::<bool>(),
            "the peft-path result must differ (it mis-scales) for this test to discriminate"
        );
    }

    // ---- F-069 format classification --------------------------------------------------------------

    /// Save a tiny safetensors of `keys` (dummy `[1,1]` tensors — classification reads only
    /// keys/metadata) and load it back as `Weights`.
    fn classify_weights(
        tmp: &tempfile::TempDir,
        name: &str,
        keys: &[&str],
        meta: Option<&HashMap<String, String>>,
    ) -> Weights {
        let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
        let entries: Vec<(&str, &Array)> = keys.iter().map(|k| (*k, &dummy)).collect();
        let path = scratch_file(tmp, name);
        Array::save_safetensors(entries, meta, &path).unwrap();
        Weights::from_file(&path).unwrap()
    }

    /// F-069: `classify_adapter_format` is the single routing truth — one minimal key-set per
    /// [`AdapterFormat`] variant, plus the precedence corners (LyCORIS keys beat kohya's shared
    /// `lora_unet_` prefix; BFL naming beats kohya; the `networkType=lokr` stamp gates `lokr_*`
    /// keys; a BFL-named LyCORIS file on a host without a BFL surface stays third-party).
    #[test]
    fn classify_adapter_format_per_format() {
        let tmp = tempfile::tempdir().unwrap();
        let lokr_meta: HashMap<String, String> = [
            ("networkType".to_string(), "lokr".to_string()),
            ("alpha".to_string(), "1.0".to_string()),
            ("rank".to_string(), "1".to_string()),
        ]
        .into_iter()
        .collect();
        // A host without a BFL surface (the common case) and the FLUX-shaped `BflHost` whose
        // `bfl_targets()` names `diffusion_model.double_blocks.…`. Tables are per-host.
        let plain = MultiHost::new(&[(
            "blocks.0.attn.to_q",
            Array::from_slice(&[0.0f32; 4], &[2, 2]),
        )]);
        let flux = BflHost::new();
        let mut tp = HostTables::default();
        let mut tf = HostTables::default();
        let bfl_qkv = "diffusion_model.double_blocks.0.img_attn.qkv";

        // Third-party LoKr: `lokr_*` keys, no stamp — regardless of the declared kind.
        let w = classify_weights(
            &tmp,
            "cls_tp_lokr.safetensors",
            &["lin.lokr_w1", "lin.lokr_w2"],
            None,
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lora, &plain, &mut tp),
            AdapterFormat::ThirdpartyLokr
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lokr, &plain, &mut tp),
            AdapterFormat::ThirdpartyLokr
        );

        // Precedence: a kohya-flattened LoKr also carries `lora_unet_`, but LyCORIS keys win
        // (is_kohya would claim it and apply nothing).
        let w = classify_weights(
            &tmp,
            "cls_tp_lokr_kohya.safetensors",
            &[
                "lora_unet_blocks_0_attn_to_q.lokr_w1",
                "lora_unet_blocks_0_attn_to_q.lokr_w2",
            ],
            None,
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lora, &plain, &mut tp),
            AdapterFormat::ThirdpartyLokr
        );

        // Third-party LoKr in BFL/ComfyUI fused naming — only on a host with a BFL surface.
        let w = classify_weights(
            &tmp,
            "cls_tp_lokr_bfl.safetensors",
            &[&format!("{bfl_qkv}.lokr_w1"), &format!("{bfl_qkv}.lokr_w2")],
            None,
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lokr, &flux, &mut tf),
            AdapterFormat::ThirdpartyLokrBfl
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lokr, &plain, &mut tp),
            AdapterFormat::ThirdpartyLokr,
            "no BFL surface ⇒ the same file stays on the plain third-party path"
        );

        // Third-party LoHa (`hada_*` keys) + its BFL twin.
        let w = classify_weights(
            &tmp,
            "cls_tp_loha.safetensors",
            &[
                "lin.hada_w1_a",
                "lin.hada_w1_b",
                "lin.hada_w2_a",
                "lin.hada_w2_b",
            ],
            None,
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lora, &plain, &mut tp),
            AdapterFormat::ThirdpartyLoha
        );
        let w = classify_weights(
            &tmp,
            "cls_tp_loha_bfl.safetensors",
            &[
                &format!("{bfl_qkv}.hada_w1_a"),
                &format!("{bfl_qkv}.hada_w1_b"),
            ],
            None,
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lora, &flux, &mut tf),
            AdapterFormat::ThirdpartyLohaBfl
        );

        // BFL fused→split LoRA — detected before kohya despite sharing no `lora_unet_` here
        // (the `diffusion_model.` spelling), and before the PEFT fallback.
        let w = classify_weights(
            &tmp,
            "cls_bfl_lora.safetensors",
            &[
                &format!("{bfl_qkv}.lora_up.weight"),
                &format!("{bfl_qkv}.lora_down.weight"),
            ],
            None,
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lora, &flux, &mut tf),
            AdapterFormat::BflLora
        );

        // kohya-flattened LoRA.
        let w = classify_weights(
            &tmp,
            "cls_kohya.safetensors",
            &[
                "lora_unet_blocks_0_attn_to_q.lora_down.weight",
                "lora_unet_blocks_0_attn_to_q.lora_up.weight",
            ],
            None,
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lora, &plain, &mut tp),
            AdapterFormat::KohyaLora
        );

        // peft LoKr: the `networkType=lokr` stamp gates the `lokr_*` keys off the third-party path.
        let w = classify_weights(
            &tmp,
            "cls_peft_lokr.safetensors",
            &["lin.lokr_w1", "lin.lokr_w2"],
            Some(&lokr_meta),
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lokr, &plain, &mut tp),
            AdapterFormat::PeftLokr
        );

        // peft LoKr in BFL/ComfyUI fused naming (sc-8345).
        let w = classify_weights(
            &tmp,
            "cls_peft_lokr_bfl.safetensors",
            &[&format!("{bfl_qkv}.lokr_w1"), &format!("{bfl_qkv}.lokr_w2")],
            Some(&lokr_meta),
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lokr, &flux, &mut tf),
            AdapterFormat::PeftLokrBfl
        );

        // PEFT/diffusers LoRA — the common fallback.
        let w = classify_weights(
            &tmp,
            "cls_peft_lora.safetensors",
            &[
                "transformer.lin.lora_A.weight",
                "transformer.lin.lora_B.weight",
            ],
            None,
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lora, &plain, &mut tp),
            AdapterFormat::PeftLora
        );

        // Declared `Lora` against `networkType=lokr` metadata — the caller error.
        let w = classify_weights(
            &tmp,
            "cls_mismatch.safetensors",
            &["lin.lokr_w1", "lin.lokr_w2"],
            Some(&lokr_meta),
        );
        assert_eq!(
            classify_adapter_format(&w, AdapterKind::Lora, &plain, &mut tp),
            AdapterFormat::LoraKindMismatch
        );
    }

    // ---- Packed (Q4/Q8) base: structured LoKr, never a materialized delta (sc-10578) -------------

    /// `[out, in] = [64, 64]` — `in` is exactly one group at the default `group_size = 64`, so the
    /// base quantizes cleanly and `base_shape()` recovers `[64, 64]` off the `[64, 1]` scales grid.
    const PK_OUT: i32 = 64;
    const PK_IN: i32 = 64;

    /// A deterministic base weight whose entries are multiples of `1/16` — exact in bf16, so the
    /// `PARITY-BF16` downcast inside `quantize()` introduces no rounding of its own.
    fn pk_base_weight() -> Array {
        let v: Vec<f32> = (0..(PK_OUT * PK_IN) as usize)
            .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
            .collect();
        Array::from_slice(&v, &[PK_OUT, PK_IN])
    }

    /// A peft LoKr over `lin`: `w1 [8,8] ⊗ w2 [8,8] → ΔW [64,64]`, `alpha/rank = 8/4 = 2.0`.
    /// Factor entries are ±powers of two so `kron` and both scale multiplies stay exact in bf16 —
    /// any residual mismatch is then a real disagreement, not float noise.
    fn pk_lokr_file(tmp: &tempfile::TempDir, name: &str) -> Weights {
        let w1: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.25 })
            .collect();
        let w2: Vec<f32> = (0..64)
            .map(|i| if i % 3 == 0 { 0.25 } else { -0.5 })
            .collect();
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("rank".to_string(), "4".to_string());
        meta.insert("alpha".to_string(), "8.0".to_string());
        let path = scratch_file(tmp, name);
        Array::save_safetensors(
            vec![
                ("lin.lokr_w1", &Array::from_slice(&w1, &[8, 8])),
                ("lin.lokr_w2", &Array::from_slice(&w2, &[8, 8])),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();
        Weights::from_file(&path).unwrap()
    }

    /// `ΔW = kron(w1, w2)` at the full `alpha/rank · strength` scale, applied as `x·ΔWᵀ` in f32 —
    /// an INDEPENDENT reference, computed from the LoKr definition rather than by re-running either
    /// install path. Both the dense (materialized) and packed (structured) residuals must match it.
    fn pk_reference_residual(w: &Weights, x: &Array, full_scale: f32) -> Array {
        use mlx_rs::ops::{kron, matmul, multiply};
        let w1 = w
            .require("lin.lokr_w1")
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let w2 = w
            .require("lin.lokr_w2")
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let delta = multiply(kron(&w1, &w2).unwrap(), Array::from_f32(full_scale)).unwrap();
        matmul(x, delta.t()).unwrap()
    }

    fn pk_x() -> Array {
        let v: Vec<f32> = (0..(2 * PK_IN) as usize)
            .map(|i| ((i % 9) as f32 - 4.0) / 8.0)
            .collect();
        Array::from_slice(&v, &[2, PK_IN])
    }

    /// The load-bearing test. On a packed base a LoKr must install as the STRUCTURED Kronecker form
    /// (no `[out,in]` delta ever allocated) and must reproduce the same residual as the dense,
    /// materialized path. `alpha/rank = 2.0` and `strength = 2.5` are both ≠ 1 and differ from each
    /// other, so a scale threaded through the wrong path cannot coincidentally pass.
    #[test]
    fn lokr_on_packed_base_installs_structured_and_matches_dense() {
        let tmp = tempfile::tempdir().unwrap();
        let w = pk_lokr_file(&tmp, "sc10578_packed_lokr.safetensors");
        let x = pk_x();
        let strength = 2.5f32;
        let reference = pk_reference_residual(&w, &x, 2.0 * strength);

        // Dense base → materialized `Adapter::Lokr` (unchanged pre-sc-10578 behavior).
        let mut dense = OneLinear {
            lin: AdaptableLinear::dense(pk_base_weight(), None),
        };
        apply_lokr(&mut dense, &w, strength).unwrap();
        let dense_adapter = &dense.lin.adapters()[0];
        assert!(
            matches!(dense_adapter, Adapter::Lokr { .. }),
            "a DENSE base must keep the materialized delta — the other families' fork-parity \
             goldens depend on it"
        );
        let dense_residual = dense_adapter.residual(&x).unwrap();

        // Packed base → structured `Adapter::LokrStructured`.
        for bits in [4, 8] {
            let mut packed = OneLinear {
                lin: AdaptableLinear::dense(pk_base_weight(), None),
            };
            packed.lin.quantize(bits, Some(64)).unwrap();
            assert!(packed.lin.is_quantized());
            assert_eq!(packed.lin.base_shape(), vec![PK_OUT, PK_IN]);

            apply_lokr(&mut packed, &w, strength).unwrap();
            let adapter = &packed.lin.adapters()[0];
            let factors = match adapter {
                Adapter::LokrStructured { factors } => factors,
                Adapter::Lokr { .. } => panic!(
                    "q{bits}: a packed base installed a MATERIALIZED delta — this is the memory \
                     regression sc-10578 exists to prevent"
                ),
                _ => panic!("q{bits}: unexpected adapter variant"),
            };
            // The whole point: only the small factors are held, never `[out, in]`.
            assert_eq!(factors.w1.shape(), &[8, 8]);
            assert_eq!(factors.w2.shape(), &[8, 8]);
            assert!(
                (factors.w1.size() + factors.w2.size()) * 16 < (PK_OUT * PK_IN) as usize,
                "q{bits}: structured factors must be far smaller than the [out,in] delta"
            );

            let structured_residual = adapter.residual(&x).unwrap();
            assert!(
                all_close(&structured_residual, &reference, 1e-3, 1e-4, None)
                    .unwrap()
                    .item::<bool>(),
                "q{bits}: structured residual must match the independent kron reference"
            );
            assert!(
                all_close(&structured_residual, &dense_residual, 1e-3, 1e-4, None)
                    .unwrap()
                    .item::<bool>(),
                "q{bits}: structured and materialized residuals must agree"
            );
        }
    }

    /// MUTATION CHECK for the test above. `LokrFile::factors` bakes the FULL `alpha/rank · strength`
    /// into `w2` because the structured residual carries no separate scale, whereas `delta` bakes only
    /// `alpha/rank` and leaves `strength` to `Adapter::Lokr::scale`. Dropping `strength` here is the
    /// exact silent mis-scale that distinction invites — confirm the assertion above would catch it.
    #[test]
    fn structured_lokr_residual_is_sensitive_to_the_strength_factor() {
        let tmp = tempfile::tempdir().unwrap();
        let w = pk_lokr_file(&tmp, "sc10578_packed_lokr_mut.safetensors");
        let x = pk_x();
        let strength = 2.5f32;
        let reference = pk_reference_residual(&w, &x, 2.0 * strength);

        let file = parse_lokr(&w).unwrap();
        let group = file.groups.get("lin").unwrap();

        // Correct: full scale = (alpha/rank) · strength.
        let good = file
            .factors(group, strength, &[PK_OUT, PK_IN], Dtype::Bfloat16)
            .unwrap()
            .expect("a linear LoKr is deferrable");
        assert!(
            all_close(good.residual(&x).unwrap(), &reference, 1e-3, 1e-4, None)
                .unwrap()
                .item::<bool>()
        );

        // Mutated: `strength` forgotten (scale = alpha/rank only). Must FAIL the same assertion —
        // otherwise the test above is incapable of catching a dropped strength.
        let bad = file
            .factors(group, 1.0, &[PK_OUT, PK_IN], Dtype::Bfloat16)
            .unwrap()
            .expect("a linear LoKr is deferrable");
        assert!(
            !all_close(bad.residual(&x).unwrap(), &reference, 1e-3, 1e-4, None)
                .unwrap()
                .item::<bool>(),
            "dropping `strength` must change the residual — otherwise the scale assertions are vacuous"
        );
    }

    /// LoHa has no deferred form at all (Hadamard, not Kronecker), so on a packed base it falls back to
    /// the materialized `[out,in]` delta — the pre-sc-10578 behavior, preserved deliberately. Erroring
    /// instead (my first cut) would break every third-party LoHa on the DEFAULT tier of every MLX model:
    /// adapters install AFTER `.quantize()`, and the worker defaults `spec.quantize` to `Some(Q8)`. The
    /// fallback is correct — `Adapter::Lokr`'s residual never reads the base — just memory-hungry.
    /// Whether a packed base should refuse a multi-GB delta is sc-10678.
    #[test]
    fn loha_on_packed_base_falls_back_to_materialization() {
        let tmp = tempfile::tempdir().unwrap();
        let r = 4;
        let a: Vec<f32> = (0..(PK_OUT * r) as usize)
            .map(|i| (i % 5) as f32 * 0.1)
            .collect();
        let b: Vec<f32> = (0..(r * PK_IN) as usize)
            .map(|i| (i % 7) as f32 * 0.1)
            .collect();
        let (wa, wb) = (
            Array::from_slice(&a, &[PK_OUT, r]),
            Array::from_slice(&b, &[r, PK_IN]),
        );
        let path = scratch_file(&tmp, "sc10578_packed_loha.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.hada_w1_a", &wa),
                ("lin.hada_w1_b", &wb),
                ("lin.hada_w2_a", &wa),
                ("lin.hada_w2_b", &wb),
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(is_loha_keys(&w));

        // Dense base: still works, materialized as before.
        let mut dense = OneLinear {
            lin: AdaptableLinear::dense(pk_base_weight(), None),
        };
        assert_eq!(
            apply_loha_thirdparty(&mut dense, &w, 1.0).unwrap().applied,
            1
        );

        // Packed base: falls back to the materialized delta — installed, NOT rejected.
        let mut packed = OneLinear {
            lin: AdaptableLinear::dense(pk_base_weight(), None),
        };
        packed.lin.quantize(8, Some(64)).unwrap();
        assert_eq!(
            apply_loha_thirdparty(&mut packed, &w, 1.0).unwrap().applied,
            1,
            "a LoHa must still install on a packed base — erroring here would break every \
             third-party LoHa on the default tier of every MLX model (adapters install AFTER \
             .quantize(), and the worker defaults spec.quantize to Some(Q8))"
        );
        match packed.lin.adapters() {
            [Adapter::Lokr { delta, .. }] => assert_eq!(delta.shape(), &[PK_OUT, PK_IN]),
            _ => panic!("LoHa on a packed base must install the materialized delta"),
        }
        // And it must be CORRECT, not merely present: the residual is base-independent, so the packed
        // and dense installs produce the same one. (Catches a fallback that silently mis-scales.)
        let x = pk_x();
        assert!(
            all_close(
                packed.lin.adapters()[0].residual(&x).unwrap(),
                dense.lin.adapters()[0].residual(&x).unwrap(),
                1e-5,
                1e-6,
                None
            )
            .unwrap()
            .item::<bool>(),
            "the packed fallback residual must equal the dense one"
        );
        assert!(
            packed.lin.is_quantized(),
            "the fallback must not dequantize the base"
        );
    }

    /// sc-10678 pre-flight guard — the projection is exact (`out·in·2` bytes for the bf16 delta) and the
    /// budget check is a pure comparison, so both are deterministically testable with **no** global MLX
    /// memory state (the `install_lycoris_groups` wiring reads `get_active_memory()`/`get_memory_limit()`
    /// and hands the values to these; the fits-path accumulation is exercised by
    /// `loha_on_packed_base_falls_back_to_materialization`, whose tiny delta stays under any real budget).
    #[test]
    fn projected_delta_bytes_is_out_times_in_times_bf16() {
        assert_eq!(
            projected_delta_bytes(&[2048, 2048], Dtype::Bfloat16),
            2048 * 2048 * 2
        );
        assert_eq!(
            projected_delta_bytes(&[64, 128], Dtype::Bfloat16),
            64 * 128 * 2
        );
        assert_eq!(
            projected_delta_bytes(&[64, 128], Dtype::Float32),
            64 * 128 * 4
        );
        // A whole DiT's worth of non-deferrable deltas (Anima: 448 targets ≈ 2048² each) ≈ 3.75 GiB —
        // the memory-lie that motivated the guard (a ~1 GB q4 DiT would balloon past bf16).
        let whole = 448 * projected_delta_bytes(&[2048, 2048], Dtype::Bfloat16);
        let gib = whole as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!((3.5..4.0).contains(&gib), "448×2048² bf16 ≈ {gib:.2} GiB");
    }

    #[test]
    fn materialization_exceeds_budget_respects_headroom_and_unlimited() {
        let gib = 1024usize * 1024 * 1024;
        // 4 GiB of new deltas on a 1 GiB resident set, 16 GiB limit → 5 GiB < 16·0.85=13.6 → fits.
        assert!(!materialization_exceeds_budget(gib, 4 * gib, 16 * gib));
        // Same deltas against a 4 GiB limit → 5 GiB > 4·0.85=3.4 → refuse.
        assert!(materialization_exceeds_budget(gib, 4 * gib, 4 * gib));
        // The 0.85 headroom bites between 8 and 9 GiB against a 10 GiB limit (safe = 8.5) — dropping the
        // headroom (using the raw limit) would flip the second assert, so this pins it.
        assert!(
            !materialization_exceeds_budget(0, 8 * gib, 10 * gib),
            "8 < 8.5 fits"
        );
        assert!(
            materialization_exceeds_budget(0, 9 * gib, 10 * gib),
            "9 > 8.5 refuses"
        );
        // limit == 0 ⇒ MLX has no cap set (unlimited) ⇒ never guard, even for a huge projection.
        assert!(!materialization_exceeds_budget(gib, 100 * gib, 0));
        // Nothing resident + nothing projected → never exceeds. NOTE the plain-LoRA / all-deferred /
        // dense case (`projected == 0`) is short-circuited in `install_lycoris_groups` BEFORE this is
        // called (`if projected_materialize > 0`), so the guard never fires on the resident set alone —
        // this pure fn, given a resident set already over budget, WOULD return true, which is why the
        // `projected > 0` gate lives at the call site, not here.
        assert!(!materialization_exceeds_budget(0, 0, gib));
    }

    /// sc-10678 — the over-budget REFUSAL path, end to end through `apply_loha_thirdparty`. `#[ignore]`d
    /// because it mutates the PROCESS-GLOBAL MLX memory limit (`set_memory_limit`), which would clobber
    /// concurrent allocations in the default multi-threaded runner. Run it ALONE:
    /// `cargo test -p mlx-gen --lib loha_on_packed_over_budget_is_refused -- --ignored --exact`.
    /// Sets the limit below the resident set so any materialization exceeds it, applies a LoHa to a
    /// packed base, and asserts a catchable, ACTIONABLE error (not a mid-run SIGKILL), leaving the base
    /// untouched — then restores the limit. The guard fires in pass 1 (shape-only, no allocation), so the
    /// tiny limit never spuriously errors before the refusal.
    #[test]
    #[ignore = "mutates the process-global MLX memory limit; run alone"]
    fn loha_on_packed_over_budget_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let r = 4;
        let a: Vec<f32> = (0..(PK_OUT * r) as usize)
            .map(|i| (i % 5) as f32 * 0.1)
            .collect();
        let b: Vec<f32> = (0..(r * PK_IN) as usize)
            .map(|i| (i % 7) as f32 * 0.1)
            .collect();
        let (wa, wb) = (
            Array::from_slice(&a, &[PK_OUT, r]),
            Array::from_slice(&b, &[r, PK_IN]),
        );
        let path = scratch_file(&tmp, "sc10678_over_budget_loha.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.hada_w1_a", &wa),
                ("lin.hada_w1_b", &wb),
                ("lin.hada_w2_a", &wa),
                ("lin.hada_w2_b", &wb),
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();

        let mut packed = OneLinear {
            lin: AdaptableLinear::dense(pk_base_weight(), None),
        };
        packed.lin.quantize(8, Some(64)).unwrap();

        // Force any materialization over budget: limit = 1 byte ⇒ safe = 0. Save + restore.
        let prev = mlx_rs::memory::set_memory_limit(1);
        let result = apply_loha_thirdparty(&mut packed, &w, 1.0);
        mlx_rs::memory::set_memory_limit(prev);

        let err = result
            .expect_err("a LoHa over budget on a packed base must be refused, not materialized");
        let msg = err.to_string();
        assert!(
            msg.contains("quantized") && msg.contains("plain LoRA/LoKr"),
            "the refusal must be actionable (name the quantized-tier issue + the plain-LoRA escape): {msg}"
        );
        // A refused install leaks no partial adapter, and never dequantizes the base.
        assert!(
            packed.lin.adapters().is_empty(),
            "a refused install must not leave a partial adapter"
        );
        assert!(packed.lin.is_quantized());
    }

    /// A tucker/CP `lokr_t2` has no 2-D matrix form, so `build_lokr_factors` returns `None` and the
    /// packed base falls back to the materialized path — **exactly what a dense base does**. Here that
    /// fallback then fails, because a conv-shaped `ΔW` cannot reshape onto a Linear `[64,64]` target
    /// (lycoris only emits `lokr_t2` for conv layers). The point of the test is that packed and dense
    /// behave IDENTICALLY: sc-10578 introduced no new failure mode for non-deferrable modules.
    #[test]
    fn tucker_lokr_on_packed_base_matches_dense_behavior() {
        let tmp = tempfile::tempdir().unwrap();
        let t2 = Array::from_slice(&[0.1f32; 2 * 2 * 3 * 3], &[2, 2, 3, 3]);
        let w2a = Array::from_slice(&[0.2f32; 2 * 8], &[2, 8]);
        let w2b = Array::from_slice(&[0.3f32; 2 * 8], &[2, 8]);
        let w1 = Array::from_slice(&[0.4f32; 8 * 8], &[8, 8]);
        let path = scratch_file(&tmp, "sc10578_packed_tucker_lokr.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lokr_w1", &w1),
                ("lin.lokr_t2", &t2),
                ("lin.lokr_w2_a", &w2a),
                ("lin.lokr_w2_b", &w2b),
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();

        // Dense base: the materialized path rejects a conv delta on a Linear target.
        let mut dense = OneLinear {
            lin: AdaptableLinear::dense(pk_base_weight(), None),
        };
        let dense_err = apply_lokr_thirdparty(&mut dense, &w, 1.0)
            .expect_err("a conv/tucker delta cannot reshape onto a Linear base")
            .to_string();

        // Packed base: falls back to that same path, so it fails the same way — not with a new,
        // sc-10578-specific rejection.
        let mut packed = OneLinear {
            lin: AdaptableLinear::dense(pk_base_weight(), None),
        };
        packed.lin.quantize(4, Some(64)).unwrap();
        let e = apply_lokr_thirdparty(&mut packed, &w, 1.0)
            .expect_err("a tucker LoKr on a packed base fails via the materialized fallback")
            .to_string();
        assert_eq!(
            e, dense_err,
            "packed must fail identically to dense — sc-10578 must add no new failure mode"
        );
        assert!(
            packed.lin.adapters().is_empty(),
            "nothing installed on failure"
        );
    }

    /// Write a third-party LoHa file targeting `lin` at `[PK_OUT, PK_IN]` (rank 4).
    fn write_loha_lin(tmp: &tempfile::TempDir, name: &str) -> Weights {
        let r = 4;
        let a: Vec<f32> = (0..(PK_OUT * r) as usize)
            .map(|i| (i % 5) as f32 * 0.1)
            .collect();
        let b: Vec<f32> = (0..(r * PK_IN) as usize)
            .map(|i| (i % 7) as f32 * 0.1)
            .collect();
        let (wa, wb) = (
            Array::from_slice(&a, &[PK_OUT, r]),
            Array::from_slice(&b, &[r, PK_IN]),
        );
        let path = scratch_file(tmp, name);
        Array::save_safetensors(
            vec![
                ("lin.hada_w1_a", &wa),
                ("lin.hada_w1_b", &wb),
                ("lin.hada_w2_a", &wa),
                ("lin.hada_w2_b", &wb),
            ],
            None,
            &path,
        )
        .unwrap();
        Weights::from_file(&path).unwrap()
    }

    #[test]
    #[ignore = "mutates the process-global MLX memory limit; run alone"]
    fn dense_base_loha_over_budget_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        // F-011 (sc-11129): a DENSE-base LoHa materializes the same `[out,in]` bf16 delta as a stacked
        // residual, so it must be included in the sc-10678 budget projection — the OOM the packed guard
        // prevents is reachable on the dense tier too. Before the fix a dense group added 0 to the
        // projection and this install went through unchecked.
        let w = write_loha_lin(&tmp, "f011_dense_over_budget_loha.safetensors");
        let mut dense = OneLinear {
            lin: AdaptableLinear::dense(pk_base_weight(), None),
        };
        let prev = mlx_rs::memory::set_memory_limit(1);
        let result = apply_loha_thirdparty(&mut dense, &w, 1.0);
        mlx_rs::memory::set_memory_limit(prev);
        result.expect_err("a dense LoHa over budget must now be refused up front (F-011)");
        assert!(
            dense.lin.adapters().is_empty(),
            "a refused install must leave no partial adapter on the dense base"
        );
    }

    #[test]
    fn dense_base_loha_within_budget_still_installs() {
        let tmp = tempfile::tempdir().unwrap();
        // F-011 regression: counting the dense materialization must only refuse GENUINELY over-budget
        // runs — a LoHa that fits still installs on the dense base (the default, unlimited-limit path).
        let w = write_loha_lin(&tmp, "f011_dense_ok_loha.safetensors");
        let mut dense = OneLinear {
            lin: AdaptableLinear::dense(pk_base_weight(), None),
        };
        assert_eq!(
            apply_loha_thirdparty(&mut dense, &w, 1.0).unwrap().applied,
            1,
            "a within-budget dense LoHa must still install"
        );
    }

    #[test]
    fn lycoris_pass2_resolution_miss_is_surfaced_not_dropped() {
        let tmp = tempfile::tempdir().unwrap();
        // F-012 (sc-11129): if a module resolves in pass 1 but vanishes by pass 2 (a lazy/offloaded
        // host, epic 10834), the plan must be SURFACED in `unmatched_paths`, never silently dropped —
        // honoring `install_lycoris_groups`'s contract. Modeled with a host whose `adaptable_mut`
        // returns the linear only on its FIRST call (pass 1) and `None` after (pass 2).
        struct VanishingHost {
            lin: AdaptableLinear,
            calls: usize,
        }
        impl AdaptableHost for VanishingHost {
            fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
                if path != ["lin"] {
                    return None;
                }
                self.calls += 1;
                if self.calls == 1 {
                    Some(&mut self.lin)
                } else {
                    None
                }
            }
        }
        let w = pk_lokr_file(&tmp, "f012_vanishing_lokr.safetensors");
        let mut host = VanishingHost {
            lin: AdaptableLinear::dense(pk_base_weight(), None),
            calls: 0,
        };
        let report = apply_lokr(&mut host, &w, 1.0).unwrap();
        assert_eq!(report.applied, 0, "the vanished target installs nothing");
        assert_eq!(
            report.unmatched_paths,
            vec!["lin".to_string()],
            "a pass-2 resolution miss must be surfaced in unmatched_paths"
        );
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn bfl_lycoris_fused_mismatched_in_dim_errors() {
        // F-013 (sc-11129): a fused BFL destination set that does not share one input dim must fail with
        // a NAMED validation error (module + dims), not an opaque downstream kron/reshape error. Drive
        // `install_bfl_lycoris` directly with a host whose two fused destinations disagree on in_dim.
        struct MismatchHost {
            q: AdaptableLinear,
            k: AdaptableLinear,
        }
        impl AdaptableHost for MismatchHost {
            fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
                match path {
                    ["attn", "to_q"] => Some(&mut self.q),
                    ["attn", "to_k"] => Some(&mut self.k),
                    _ => None,
                }
            }
            fn adaptable_paths(&self) -> Vec<String> {
                vec!["attn.to_q".to_string(), "attn.to_k".to_string()]
            }
            fn bfl_targets(&self) -> Vec<BflTarget> {
                let up = vec!["diffusion_model.blk.qkv.lora_B.weight".to_string()];
                let down = vec!["diffusion_model.blk.qkv.lora_A.weight".to_string()];
                let alpha = vec!["diffusion_model.blk.qkv.alpha".to_string()];
                ["to_q", "to_k"]
                    .iter()
                    .enumerate()
                    .map(|(idx, dst)| BflTarget {
                        target_path: format!("attn.{dst}"),
                        up_keys: up.clone(),
                        down_keys: down.clone(),
                        alpha_keys: alpha.clone(),
                        up_slice: Some(LoraRowSlice::Chunk {
                            n: 2,
                            index: idx as i32,
                        }),
                        down_slice: None,
                    })
                    .collect()
            }
        }
        // to_q in=3, to_k in=4 — the fused qkv cannot share one input dim.
        let mut host = MismatchHost {
            q: AdaptableLinear::dense(Array::from_slice(&[0.0f32; 6], &[2, 3]), None),
            k: AdaptableLinear::dense(Array::from_slice(&[0.0f32; 8], &[2, 4]), None),
        };
        let targets = host.bfl_targets();
        let map = bfl_lycoris_module_map(&targets);
        let groups: Vec<(String, fn(&[i32]) -> Result<Array>)> = vec![(
            "diffusion_model.blk.qkv".to_string(),
            // Never reached — the in_dim guard fires in pass 1, before any reconstruction.
            (|_: &[i32]| unreachable!("reconstruct must not run when the in_dim guard fires"))
                as fn(&[i32]) -> Result<Array>,
        )];
        let err = install_bfl_lycoris(&mut host, &map, groups, 1.0)
            .expect_err("mismatched fused in-dims must be a named validation error");
        let msg = err.to_string();
        assert!(
            msg.contains("in_dim") && msg.contains("diffusion_model.blk.qkv"),
            "the error must name the module and the mismatched dims: {msg}"
        );
    }

    #[test]
    #[ignore = "mutates the process-global MLX memory limit; run alone"]
    #[allow(clippy::type_complexity)]
    fn bfl_lycoris_over_budget_is_refused() {
        // F-010 (sc-11129): the BFL fused→split path materializes the whole fused delta with no
        // deferred form, so it must apply the sc-10678 memory guard like its plain-path twin — refusing
        // UP FRONT rather than OOMing the worker mid-reconstruct. Drive `install_bfl_lycoris` with a
        // 1-byte memory limit so any projected materialization is over budget.
        let mut host = BflHost::new();
        let targets = host.bfl_targets();
        let map = bfl_lycoris_module_map(&targets);
        let groups: Vec<(String, fn(&[i32]) -> Result<Array>)> = vec![(
            "diffusion_model.double_blocks.0.img_attn.qkv".to_string(),
            // Never reached — the budget guard fires in pass 1, before any reconstruction.
            (|_: &[i32]| unreachable!("reconstruct must not run when the budget guard fires"))
                as fn(&[i32]) -> Result<Array>,
        )];
        let prev = mlx_rs::memory::set_memory_limit(1);
        let result = install_bfl_lycoris(&mut host, &map, groups, 1.0);
        mlx_rs::memory::set_memory_limit(prev);
        let err =
            result.expect_err("a BFL LyCORIS materialization over budget must be refused (F-010)");
        assert!(
            err.to_string().contains("budget"),
            "the refusal must be the actionable budget error: {err}"
        );
        // Nothing installed on refusal.
        for dst in ["to_q", "to_k", "to_v"] {
            assert!(
                host.adaptable_mut(&["transformer_blocks", "0", "attn", dst])
                    .unwrap()
                    .adapters()
                    .is_empty(),
                "a refused BFL install must leave no partial adapter on {dst}"
            );
        }
    }

    #[test]
    fn bfl_lora_conflicting_alpha_errors() {
        let tmp = tempfile::tempdir().unwrap();
        // F-014 (sc-11129): two `.alpha` spellings mapping to one BFL target must be a hard conflict
        // error (mirroring `apply_lora_peft`), not a nondeterministic HashMap-order last-wins scale.
        let a = Array::from_slice(&[0.1f32, 0.2, 0.3], &[1, 3]);
        let b = Array::from_slice(&[0.4f32, 0.5], &[2, 1]);
        let alpha4 = Array::from_slice(&[4.0f32], &[1]);
        let alpha8 = Array::from_slice(&[8.0f32], &[1]);
        let path = scratch_file(&tmp, "f014_bfl_alpha_conflict.safetensors");
        Array::save_safetensors(
            vec![
                ("diffusion_model.m.lora_A.weight", &a),
                ("diffusion_model.m.lora_B.weight", &b),
                ("diffusion_model.m.alpha", &alpha4),
                ("diffusion_model.m2.alpha", &alpha8),
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();

        // One target whose two alpha_keys both resolve to it — the conflict shape.
        let targets = vec![BflTarget {
            target_path: "attn.to_q".to_string(),
            up_keys: vec!["diffusion_model.m.lora_B.weight".to_string()],
            down_keys: vec!["diffusion_model.m.lora_A.weight".to_string()],
            alpha_keys: vec![
                "diffusion_model.m.alpha".to_string(),
                "diffusion_model.m2.alpha".to_string(),
            ],
            up_slice: None,
            down_slice: None,
        }];
        let mut host = MultiHost::new(&[("attn.to_q", Array::from_slice(&[0.0f32; 6], &[2, 3]))]);
        let err = apply_lora_bfl(&mut host, &w, 1.0, &targets)
            .expect_err("conflicting alphas for one BFL target must error");
        assert!(
            err.to_string().contains("alpha conflict"),
            "expected a named alpha-conflict error, got: {err}"
        );
    }

    // ---- ComfyUI/lightx2v diff-patch fold (sc-13825, MLX parity for candle sc-13726) -------------

    /// A host with one biased dense `AdaptableLinear` at the Krea projector path, carrying the
    /// `txtfusion`→`text_fusion` alias the real DiT uses — so a `diffusion_model.txtfusion.projector.*`
    /// diff-patch key resolves after [`strip_common_lora_prefix`], exactly as on the DiT.
    struct DiffHost {
        lin: AdaptableLinear,
    }
    impl AdaptableHost for DiffHost {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            match path {
                ["text_fusion" | "txtfusion", "projector"] => Some(&mut self.lin),
                _ => None,
            }
        }
        fn adaptable_paths(&self) -> Vec<String> {
            vec!["text_fusion.projector".to_string()]
        }
    }

    fn save_one(tmp: &tempfile::TempDir, name: &str, entries: Vec<(&str, &Array)>) -> PathBuf {
        let path = scratch_file(tmp, name);
        Array::save_safetensors(entries, None, &path).unwrap();
        path
    }

    #[test]
    fn has_diff_patch_keys_detects_diff_and_diff_b() {
        let tmp = tempfile::tempdir().unwrap();
        let dp = save_one(
            &tmp,
            "dp_detect.safetensors",
            vec![(
                "diffusion_model.txtfusion.projector.diff",
                &Array::from_slice(&[0.1f32; 3], &[1, 3]),
            )],
        );
        assert!(has_diff_patch_keys(&Weights::from_file(&dp).unwrap()));

        // A plain low-rank file carries no `.diff`/`.diff_b` → not a diff-patch file.
        let plain = save_one(
            &tmp,
            "dp_plain.safetensors",
            vec![
                (
                    "diffusion_model.txtfusion.projector.lora_down.weight",
                    &Array::from_slice(&[0.1f32; 3], &[1, 3]),
                ),
                (
                    "diffusion_model.txtfusion.projector.lora_up.weight",
                    &Array::from_slice(&[0.1f32; 1], &[1, 1]),
                ),
            ],
        );
        assert!(!has_diff_patch_keys(&Weights::from_file(&plain).unwrap()));

        // The header-only twin (sc-13884 multi-phase guard) makes the SAME diff-patch-vs-low-rank
        // distinction from just the key names — no tensor load: a `.diff`/`.diff_b` marks a diff-patch
        // (rejected from multi-phase), a low-rank `lora_*` file does not (the toggleable turbo-LoRA case).
        assert!(has_diff_patch_key_names(
            ["diffusion_model.txtfusion.projector.diff"].into_iter()
        ));
        assert!(has_diff_patch_key_names(
            ["some.module.diff_b", "other.lora_down.weight"].into_iter()
        ));
        assert!(!has_diff_patch_key_names(
            [
                "blk.0.attn.to_q.lora_down.weight",
                "blk.0.attn.to_q.lora_up.weight"
            ]
            .into_iter()
        ));
        assert!(!has_diff_patch_key_names(std::iter::empty()));
    }

    /// The community filter-bypass: a single `diffusion_model.txtfusion.projector.diff` folds
    /// `W += scale·δ` into the dense projector base, resolving through the prefix-strip + `txtfusion`
    /// alias. Untargeted state stays put.
    #[test]
    fn fold_diff_patch_folds_projector_weight_delta() {
        let tmp = tempfile::tempdir().unwrap();
        let base_w = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]);
        let delta = Array::from_slice(&[10.0f32, 20.0, 30.0], &[1, 3]);
        let dp = save_one(
            &tmp,
            "dp_proj.safetensors",
            vec![("diffusion_model.txtfusion.projector.diff", &delta)],
        );

        let mut host = DiffHost {
            lin: AdaptableLinear::dense(base_w, None),
        };
        let report =
            fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 0.5, AdapterKind::Lora)])
                .unwrap();
        assert_eq!(report.folded, 1, "the projector diff must fold");
        assert!(report.skipped.is_empty());
        assert!(report.unmatched.is_empty());

        // W + 0.5·δ = [1+5, 2+10, 3+15].
        let (w, _) = host.lin.dense_weight().unwrap();
        let want = Array::from_slice(&[6.0f32, 12.0, 18.0], &[1, 3]);
        assert!(all_close(w, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    /// A `.diff_b` bias delta folds into the base bias alongside the `.diff` weight delta — the channel
    /// low-rank adapters cannot express. Both fold at `scale`, counted as two.
    #[test]
    fn fold_diff_patch_folds_weight_and_bias_delta() {
        let tmp = tempfile::tempdir().unwrap();
        let dw = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let db = Array::from_slice(&[5.0f32, 6.0], &[2]);
        let dp = save_one(
            &tmp,
            "dp_wb.safetensors",
            vec![
                ("diffusion_model.txtfusion.projector.diff", &dw),
                ("diffusion_model.txtfusion.projector.diff_b", &db),
            ],
        );
        let mut host = DiffHost {
            lin: AdaptableLinear::dense(
                Array::from_slice(&[0.0f32; 4], &[2, 2]),
                Some(Array::from_slice(&[0.0f32; 2], &[2])),
            ),
        };
        let report =
            fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 1.0, AdapterKind::Lora)])
                .unwrap();
        assert_eq!(report.folded, 2, "weight + bias delta both fold");

        let (w, b) = host.lin.dense_weight().unwrap();
        assert!(all_close(w, &dw, 1e-5, 1e-5, false).unwrap().item::<bool>());
        assert!(all_close(b.unwrap(), &db, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    /// Module-coupled shape-aware skip: a `.diff` whose shape ≠ the base is skipped as a whole module —
    /// its coupled `.diff_b` dropped too, never a half-patch — surfaced, never folded; base untouched.
    #[test]
    fn fold_diff_patch_shape_mismatch_skips_whole_module() {
        let tmp = tempfile::tempdir().unwrap();
        let base_w = Array::from_slice(&[0.0f32; 4], &[2, 2]);
        let base_b = Array::from_slice(&[0.0f32; 2], &[2]);
        let dp = save_one(
            &tmp,
            "dp_mm.safetensors",
            vec![
                // [3,3] cannot fold into the [2,2] base; its coupled [2] bias must drop with it.
                (
                    "diffusion_model.txtfusion.projector.diff",
                    &Array::from_slice(&[1.0f32; 9], &[3, 3]),
                ),
                (
                    "diffusion_model.txtfusion.projector.diff_b",
                    &Array::from_slice(&[9.0f32; 2], &[2]),
                ),
            ],
        );
        let mut host = DiffHost {
            lin: AdaptableLinear::dense(base_w.clone(), Some(base_b.clone())),
        };
        let report =
            fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 1.0, AdapterKind::Lora)])
                .unwrap();
        assert_eq!(report.folded, 0, "shape-mismatched module folds nothing");
        assert_eq!(
            report.skipped.len(),
            2,
            "weight + coupled bias both surfaced as skipped"
        );

        let (w, b) = host.lin.dense_weight().unwrap();
        assert!(
            array_eq(w, &base_w, false).unwrap().item::<bool>(),
            "weight untouched"
        );
        assert!(
            array_eq(b.unwrap(), &base_b, false).unwrap().item::<bool>(),
            "bias untouched"
        );
    }

    /// The strict diff-patch entry tolerates a **diff-only** file (the low-rank pass matches nothing,
    /// but the fold pre-applied it — so no "matched nothing" error), and still errors when a diff-patch
    /// file resolves to no module at all (folds nothing AND matches no low-rank target).
    #[test]
    fn strict_with_diff_patch_tolerates_diff_only_and_errors_on_all_unmatched() {
        let tmp = tempfile::tempdir().unwrap();
        // (a) diff-only, targets the projector → folds; applied counts the fold.
        let ok = save_one(
            &tmp,
            "dp_only.safetensors",
            vec![(
                "diffusion_model.txtfusion.projector.diff",
                &Array::from_slice(&[1.0f32; 3], &[1, 3]),
            )],
        );
        let mut host = DiffHost {
            lin: AdaptableLinear::dense(Array::from_slice(&[0.0f32; 3], &[1, 3]), None),
        };
        let report = apply_adapters_strict_with_diff_patch(
            &mut host,
            &[AdapterSpec::new(ok, 1.0, AdapterKind::Lora)],
            "krea_2",
        )
        .unwrap();
        assert_eq!(
            report.applied, 1,
            "the folded diff-patch counts toward applied"
        );

        // (b) diff-patch stem resolves to no module → folds nothing, no low-rank target → error.
        let bad = save_one(
            &tmp,
            "dp_bad.safetensors",
            vec![(
                "diffusion_model.blocks.99.unknown.diff",
                &Array::from_slice(&[0.1f32; 3], &[1, 3]),
            )],
        );
        let mut host2 = DiffHost {
            lin: AdaptableLinear::dense(Array::from_slice(&[0.0f32; 3], &[1, 3]), None),
        };
        let err = apply_adapters_strict_with_diff_patch(
            &mut host2,
            &[AdapterSpec::new(bad, 1.0, AdapterKind::Lora)],
            "krea_2",
        )
        .expect_err("an all-unmatched diff-patch file must still error");
        assert!(
            err.to_string().contains("no target modules matched"),
            "expected the no-match error, got: {err}"
        );
    }
    // ---- sc-15326: tier-independent diff-patch channels + no-silent-drop reporting ----------------

    /// A host with one `[64,64]` biased Linear (quantizable — `in % group_size == 0`) plus an RMSNorm
    /// gain and an affine LayerNorm's weight+bias, exposed on the two DIFFERENT diff-patch surfaces:
    /// the Linear through `adaptable_mut`, the norms through `diff_patch_param_mut`. Mirrors the shape
    /// of the real Krea Realtime / Wan host, whose lightx2v step-distill deltas split the same way.
    struct TierHost {
        lin: AdaptableLinear,
        norm_q: Array,
        norm3_w: Array,
        norm3_b: Array,
    }
    impl TierHost {
        fn new() -> Self {
            Self {
                lin: AdaptableLinear::dense(
                    Array::from_slice(&[0.25f32; 64 * 64], &[64, 64]),
                    Some(Array::from_slice(&[1.0f32; 64], &[64])),
                ),
                norm_q: Array::from_slice(&[1.0f32; 8], &[8]),
                norm3_w: Array::from_slice(&[1.0f32; 8], &[8]),
                norm3_b: Array::from_slice(&[0.0f32; 8], &[8]),
            }
        }
    }
    impl AdaptableHost for TierHost {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            match path {
                ["blocks", "0", "self_attn", "q"] => Some(&mut self.lin),
                _ => None,
            }
        }
        fn adaptable_paths(&self) -> Vec<String> {
            vec!["blocks.0.self_attn.q".to_string()]
        }
        fn diff_patch_param_mut(
            &mut self,
            path: &[&str],
            part: DiffPatchPart,
        ) -> Option<&mut Array> {
            match (path, part) {
                (["blocks", "0", "self_attn", "norm_q"], DiffPatchPart::Weight) => {
                    Some(&mut self.norm_q)
                }
                (["blocks", "0", "norm3"], DiffPatchPart::Weight) => Some(&mut self.norm3_w),
                (["blocks", "0", "norm3"], DiffPatchPart::Bias) => Some(&mut self.norm3_b),
                _ => None,
            }
        }
    }

    /// **The sc-15326 decision, gated.** A `.diff_b` bias delta must fold IDENTICALLY over a packed Q4
    /// base and a dense one: `QuantizedLinear` packs only the weight, so `b += scale·δ_b` is the same
    /// arithmetic either way. This is what makes applying a lightx2v step-distill LoRA's 407 Linear
    /// bias deltas tier-independent — without it, switching Krea Realtime to the diff-patch installer
    /// would have folded 7 of 647 keys at the default Q4 against 407 at bf16, i.e. the same LoRA
    /// rendering differently per tier on a product where the quant tier is a *creative* choice.
    ///
    /// Discriminating: it compares the two tiers' resulting biases element-wise, so a tier-gated bias
    /// fold (the pre-sc-15326 behaviour) fails on the fold COUNT and a wrongly-scaled one fails on the
    /// values.
    #[test]
    fn diff_patch_bias_delta_folds_identically_on_a_packed_and_a_dense_base() {
        let tmp = tempfile::tempdir().unwrap();
        let db = Array::from_slice(&(0..64).map(|i| i as f32).collect::<Vec<_>>(), &[64]);
        let dp = save_one(
            &tmp,
            "dp_tier_bias.safetensors",
            vec![("diffusion_model.blocks.0.self_attn.q.diff_b", &db)],
        );
        let spec = |p: &PathBuf| vec![AdapterSpec::new(p.clone(), 0.5, AdapterKind::Lora)];

        let mut dense = TierHost::new();
        let r_dense = fold_diff_patch_adapters(&mut dense, &spec(&dp)).unwrap();

        let mut packed = TierHost::new();
        packed.lin.quantize(4, None).unwrap();
        assert!(
            packed.lin.is_quantized(),
            "the Q4 arm must actually be packed"
        );
        let r_packed = fold_diff_patch_adapters(&mut packed, &spec(&dp)).unwrap();

        assert_eq!(r_dense.folded, 1, "the bias delta folds on the dense base");
        assert_eq!(
            r_packed.folded, 1,
            "the bias delta must ALSO fold on the packed base — a QuantizedLinear keeps its bias dense"
        );
        assert!(
            r_packed.skipped.is_empty() && r_packed.unmatched.is_empty(),
            "nothing may be skipped on the packed tier: {r_packed:?}"
        );

        // b + 0.5·δ_b, bit-for-bit the same on both tiers.
        let want = Array::from_slice(
            &(0..64).map(|i| 1.0 + 0.5 * i as f32).collect::<Vec<_>>(),
            &[64],
        );
        for (label, b) in [
            ("dense", dense.lin.bias().unwrap()),
            ("packed", packed.lin.bias().unwrap()),
        ] {
            assert!(
                all_close(b, &want, 1e-5, 1e-5, false)
                    .unwrap()
                    .item::<bool>(),
                "{label} bias is not b + 0.5·δ_b"
            );
        }
    }

    /// The one channel that genuinely CANNOT be tier-independent: a full-rank **weight** `.diff` aimed
    /// at a packed Linear. It is skipped as a whole module (its coupled `.diff_b` dropped with it,
    /// never a half-patch) and surfaced — this fold does not unpack a base the user deliberately chose.
    /// The real lightx2v T2V file carries zero such keys, which is why the tier stays undivided in
    /// practice.
    #[test]
    fn diff_patch_weight_delta_on_a_packed_linear_skips_the_whole_module() {
        let tmp = tempfile::tempdir().unwrap();
        let dw = Array::from_slice(&[0.5f32; 64 * 64], &[64, 64]);
        let db = Array::from_slice(&[7.0f32; 64], &[64]);
        let dp = save_one(
            &tmp,
            "dp_tier_weight.safetensors",
            vec![
                ("diffusion_model.blocks.0.self_attn.q.diff", &dw),
                ("diffusion_model.blocks.0.self_attn.q.diff_b", &db),
            ],
        );
        let mut host = TierHost::new();
        host.lin.quantize(4, None).unwrap();
        let before = host.lin.bias().unwrap().clone();
        let report =
            fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 1.0, AdapterKind::Lora)])
                .unwrap();
        assert_eq!(
            report.folded, 0,
            "a weight .diff cannot fold into a packed base"
        );
        assert_eq!(
            report.skipped.len(),
            2,
            "weight + coupled bias both surfaced as skipped, never half-applied: {report:?}"
        );
        assert!(
            array_eq(host.lin.bias().unwrap(), &before, false)
                .unwrap()
                .item::<bool>(),
            "the coupled bias must NOT be folded when its weight delta is skipped"
        );
    }

    /// The norm surface (sc-15326): `.diff` on an RMSNorm gain and `.diff`+`.diff_b` on an affine
    /// LayerNorm fold through `AdaptableHost::diff_patch_param_mut` — the 200 `.diff` + 40
    /// `norm3.diff_b` deltas a lightx2v step-distill file carries, which are not `AdaptableLinear`s at
    /// any surface width and were dropped without a word before this existed.
    #[test]
    fn diff_patch_norm_deltas_fold_through_the_param_surface() {
        let tmp = tempfile::tempdir().unwrap();
        let d_nq = Array::from_slice(&[0.5f32; 8], &[8]);
        let d_n3w = Array::from_slice(&[2.0f32; 8], &[8]);
        let d_n3b = Array::from_slice(&[3.0f32; 8], &[8]);
        let dp = save_one(
            &tmp,
            "dp_norms.safetensors",
            vec![
                ("diffusion_model.blocks.0.self_attn.norm_q.diff", &d_nq),
                ("diffusion_model.blocks.0.norm3.diff", &d_n3w),
                ("diffusion_model.blocks.0.norm3.diff_b", &d_n3b),
            ],
        );
        let mut host = TierHost::new();
        let report =
            fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 2.0, AdapterKind::Lora)])
                .unwrap();
        assert_eq!(report.folded, 3, "all three norm deltas fold: {report:?}");
        assert!(report.skipped.is_empty() && report.unmatched.is_empty());

        for (label, got, want) in [
            ("norm_q", &host.norm_q, 1.0 + 2.0 * 0.5),
            ("norm3.weight", &host.norm3_w, 1.0 + 2.0 * 2.0),
            ("norm3.bias", &host.norm3_b, 0.0 + 2.0 * 3.0),
        ] {
            let expect = Array::from_slice(&[want; 8], &[8]);
            assert!(
                all_close(got, &expect, 1e-5, 1e-5, false)
                    .unwrap()
                    .item::<bool>(),
                "{label} is not p + 2.0·δ"
            );
        }
    }

    /// The norm path honors the same two rules the Linear path does: a shape-mismatched delta skips the
    /// WHOLE module (coupled bias dropped with it, never a half-patch), and a stem the host exposes on
    /// neither half is reported unmatched rather than dropped.
    #[test]
    fn diff_patch_norm_shape_mismatch_skips_whole_module_and_unknown_is_unmatched() {
        let tmp = tempfile::tempdir().unwrap();
        let dp = save_one(
            &tmp,
            "dp_norm_bad.safetensors",
            vec![
                // [4] cannot fold into the [8] norm3; its coupled [8] bias must drop with it.
                (
                    "diffusion_model.blocks.0.norm3.diff",
                    &Array::from_slice(&[1.0f32; 4], &[4]),
                ),
                (
                    "diffusion_model.blocks.0.norm3.diff_b",
                    &Array::from_slice(&[1.0f32; 8], &[8]),
                ),
                (
                    "diffusion_model.blocks.0.nowhere.diff",
                    &Array::from_slice(&[1.0f32; 8], &[8]),
                ),
            ],
        );
        let mut host = TierHost::new();
        let (w0, b0) = (host.norm3_w.clone(), host.norm3_b.clone());
        let report =
            fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 1.0, AdapterKind::Lora)])
                .unwrap();
        assert_eq!(report.folded, 0, "nothing may fold: {report:?}");
        assert_eq!(
            report.skipped.len(),
            2,
            "norm3 weight + coupled bias skipped"
        );
        assert_eq!(
            report.unmatched,
            vec!["blocks.0.nowhere".to_string()],
            "an unexposed stem is unmatched, not dropped"
        );
        assert!(array_eq(&host.norm3_w, &w0, false).unwrap().item::<bool>());
        assert!(
            array_eq(&host.norm3_b, &b0, false).unwrap().item::<bool>(),
            "the coupled bias must not be folded when its weight delta is skipped"
        );
    }

    /// sc-15326 review — on the norm path, a `.diff_b` whose target has **no bias channel** must take
    /// only itself out, not the module's perfectly valid weight `.diff`.
    ///
    /// Every qk-RMSNorm is weight-only, so a lightning / step-distill variant carrying `norm_q.diff_b`
    /// next to `norm_q.diff` would otherwise lose all 200 valid norm weight deltas. This is exactly the
    /// Linear path's rule ("a `.diff_b` with no shape-matching base bias is surfaced but does not undo
    /// the weight fold"), which the norm path claimed to mirror and did not. A **shape** mismatch still
    /// skips the whole module — that split is the point, and the sibling test above pins the other half.
    #[test]
    fn diff_patch_norm_bias_with_no_host_part_does_not_undo_the_weight_fold() {
        let tmp = tempfile::tempdir().unwrap();
        let dp = save_one(
            &tmp,
            "dp_norm_qk_bias.safetensors",
            vec![
                (
                    "diffusion_model.blocks.0.self_attn.norm_q.diff",
                    &Array::from_slice(&[0.5f32; 8], &[8]),
                ),
                // An RMSNorm has no β. Correctly shaped, simply nowhere to land.
                (
                    "diffusion_model.blocks.0.self_attn.norm_q.diff_b",
                    &Array::from_slice(&[9.0f32; 8], &[8]),
                ),
            ],
        );
        let mut host = TierHost::new();
        let report =
            fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 2.0, AdapterKind::Lora)])
                .unwrap();
        assert_eq!(
            report.folded, 1,
            "the weight .diff must still fold: {report:?}"
        );
        assert_eq!(
            report.skipped,
            vec!["blocks.0.self_attn.norm_q.bias".to_string()],
            "only the bias half is surfaced, and the weight stem is NOT: {report:?}"
        );
        assert!(report.unmatched.is_empty(), "{report:?}");
        let expect = Array::from_slice(&[1.0 + 2.0 * 0.5f32; 8], &[8]);
        assert!(
            all_close(&host.norm_q, &expect, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>(),
            "norm_q must carry w + 2.0·δ, not the pre-fold value"
        );
    }

    /// sc-15265's invariant, held on the diff-patch path too: **installing an adapter at scale 0 is
    /// byte-identical to never installing one.** The fold resolves and shape-checks as usual — so the
    /// skipped/unmatched surfaces stay truthful and the zero-match guard still sees the targets — but
    /// writes nothing, and the resolved deltas land on `disabled` rather than `folded`.
    ///
    /// Also pins the consequence: a **pure** diff-patch file at scale 0 folds nothing, so without
    /// counting `disabled` toward the guard relaxation, disabling it would turn into a spurious
    /// "no target modules matched" error.
    ///
    /// Discriminating on two axes, because array equality alone is **not** enough — `\u{3b4} \u{b7} 0.0` is
    /// exact zero for finite deltas, so a fold that runs anyway leaves the arrays bit-identical. So this
    /// asserts the report split (`folded == 0`, `disabled == 3`) AND carries one **non-finite** delta
    /// entry: `Inf \u{b7} 0.0 = NaN`, which a still-running fold would write into the host at precisely the
    /// setting the user chose to turn the adapter OFF. That is sc-15265's own stated rationale for
    /// skipping the residual outright rather than trusting the multiply.
    #[test]
    fn diff_patch_at_scale_zero_writes_nothing_and_still_clears_the_zero_match_guard() {
        let tmp = tempfile::tempdir().unwrap();
        let mut nq_delta = [0.5f32; 8];
        nq_delta[3] = f32::INFINITY;
        let dp = save_one(
            &tmp,
            "dp_scale_zero.safetensors",
            vec![
                (
                    "diffusion_model.blocks.0.self_attn.norm_q.diff",
                    &Array::from_slice(&nq_delta, &[8]),
                ),
                (
                    "diffusion_model.blocks.0.norm3.diff_b",
                    &Array::from_slice(&[3.0f32; 8], &[8]),
                ),
                // A Linear bias delta — the other channel that folds on every tier.
                (
                    "diffusion_model.blocks.0.self_attn.q.diff_b",
                    &Array::from_slice(&[7.0f32; 64], &[64]),
                ),
            ],
        );
        // The fold itself: everything resolves, nothing is written.
        let mut probe = TierHost::new();
        let dp_report = fold_diff_patch_adapters(
            &mut probe,
            &[AdapterSpec::new(dp.clone(), 0.0, AdapterKind::Lora)],
        )
        .unwrap();
        assert_eq!(
            (dp_report.folded, dp_report.disabled),
            (0, 3),
            "a scale-0 spec resolves its targets but folds none of them: {dp_report:?}"
        );
        assert!(
            dp_report.skipped.is_empty() && dp_report.unmatched.is_empty(),
            "resolution/shape surfacing stays truthful at scale 0: {dp_report:?}"
        );

        let mut host = TierHost::new();
        let (nq0, n3b0) = (host.norm_q.clone(), host.norm3_b.clone());
        let b0 = host.lin.bias().unwrap().clone();

        let specs = [AdapterSpec::new(dp, 0.0, AdapterKind::Lora)];
        let report = apply_adapters_strict_with_diff_patch(&mut host, &specs, "tier_host")
            .expect("a disabled pure diff-patch file must install, not read as `matched nothing`");
        assert_eq!(
            report.applied, 3,
            "all three targets resolved and were accepted: {report:?}"
        );
        assert!(report.diff_patch_unapplied.is_empty(), "{report:?}");

        for (label, got, want) in [
            ("norm_q", &host.norm_q, &nq0),
            ("norm3.bias", &host.norm3_b, &n3b0),
            ("q.bias", host.lin.bias().unwrap(), &b0),
        ] {
            assert!(
                array_eq(got, want, false).unwrap().item::<bool>(),
                "{label} moved at scale 0 — a disabled adapter must be byte-identical to none"
            );
        }
    }

    /// **No silent drop, and it reaches the caller.** Everything the diff-patch pass could not land —
    /// skipped AND unmatched — comes back on `ApplyReport::diff_patch_unapplied`, so a provider can
    /// stamp a partial install into user-visible provenance instead of it living in an engine `stderr`
    /// line nobody reads. A clean file leaves the list empty.
    #[test]
    fn strict_with_diff_patch_reports_every_unapplied_delta_on_the_report() {
        let tmp = tempfile::tempdir().unwrap();
        // A file that lands its norm delta but cannot land a packed-Linear weight delta or an
        // out-of-surface stem.
        let mixed = save_one(
            &tmp,
            "dp_report.safetensors",
            vec![
                (
                    "diffusion_model.blocks.0.self_attn.norm_q.diff",
                    &Array::from_slice(&[0.5f32; 8], &[8]),
                ),
                (
                    "diffusion_model.blocks.0.self_attn.q.diff",
                    &Array::from_slice(&[0.5f32; 64 * 64], &[64, 64]),
                ),
                (
                    "diffusion_model.blocks.0.nowhere.diff",
                    &Array::from_slice(&[1.0f32; 8], &[8]),
                ),
            ],
        );
        let mut host = TierHost::new();
        host.lin.quantize(4, None).unwrap();
        let report = apply_adapters_strict_with_diff_patch(
            &mut host,
            &[AdapterSpec::new(mixed, 1.0, AdapterKind::Lora)],
            "krea_realtime_14b",
        )
        .unwrap();
        assert_eq!(report.applied, 1, "only the norm delta landed");
        let mut unapplied = report.diff_patch_unapplied.clone();
        unapplied.sort();
        assert_eq!(
            unapplied,
            vec![
                "blocks.0.nowhere".to_string(),
                "blocks.0.self_attn.q".to_string()
            ],
            "both the packed-weight skip and the unmatched stem must be reported to the caller"
        );

        // A file whose every delta lands reports NOTHING unapplied — the list is a real signal, not
        // noise present on every render.
        let clean = save_one(
            &tmp,
            "dp_report_clean.safetensors",
            vec![(
                "diffusion_model.blocks.0.norm3.diff",
                &Array::from_slice(&[0.5f32; 8], &[8]),
            )],
        );
        let mut host2 = TierHost::new();
        let clean_report = apply_adapters_strict_with_diff_patch(
            &mut host2,
            &[AdapterSpec::new(clean, 1.0, AdapterKind::Lora)],
            "krea_realtime_14b",
        )
        .unwrap();
        assert!(
            clean_report.diff_patch_unapplied.is_empty(),
            "a fully-applied file must report nothing unapplied: {clean_report:?}"
        );
    }
}
