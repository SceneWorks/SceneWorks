//! **The parameterized QK-norm + RoPE + layout primitive** and the **adapter/quant-aware fused QKV
//! projection** (SC-18319, epic 18304 P4).
//!
//! Every DiT/MMDiT family in this tree runs the same five-stage prologue between its attention
//! projections and [`scaled_dot_product_attention`](mlx_rs::fast::scaled_dot_product_attention):
//!
//! ```text
//!   project → [full-dim QK-norm] → head split → [per-head QK-norm] → RoPE → [repeat-KV] → BHSD
//! ```
//!
//! Before this module each family open-coded that prologue, so a fix (or a fusion) had to be
//! travelled by hand ~20 times. What differs between families is *not* the pipeline — it is a
//! bounded set of **knobs**, each one demanded by a named, surveyed family. This module carries the
//! pipeline once and the knobs as data.
//!
//! ## The knob set, and who demands each
//!
//! | # | knob | type | demanded by |
//! |---|------|------|-------------|
//! | 1 | per-head vs full-dim QK-norm, post- vs pre-head-split | [`QkNormPlacement`] | LTX (full-dim, pre-split), Wan, SANA vs everyone else |
//! | 2 | interleaved/adjacent-pair vs half-split rotation | [`RopeStyle`] | Lens/Mage/PID/Boogu/Mochi vs Ideogram/Anima vs LTX |
//! | 3 | RoPE optional | [`RopeStyle::None`] per call | Anima DiT cross-attention, LTX text cross-attention, Krea |
//! | 4 | RoPE partial (leading `rot` channels, tail passthrough) | [`RopeSpec::rot_dims`] | SeedVR2 |
//! | 5 | RoPE on one stream only | [`RopeSpec::q`]/[`RopeSpec::k`] independently `None` | Mage (image only), Mochi |
//! | 6 | separate q and k position tables | [`RopeSpec::q`]/[`RopeSpec::k`] carry different tables | Anima conditioner, LTX cross-modal (`k_pe`) |
//! | 7 | RoPE = none for a whole family | [`RopeStyle::None`] | SD3 |
//! | 8 | concat-then-RoPE vs RoPE-then-concat | call order: [`StreamOrder::join`] before or after [`apply_rope`] | Chroma vs FLUX.1 |
//! | 9 | fused vs separate QKV | [`QkvSource`] / [`FusedQkvProjection`] | Lens/Ideogram/PID fused; Mage/SD3/Chroma/Boogu separate |
//! | 10 | GQA ratio, repeat-KV placed **after** RoPE | [`AttnPrepSpec::kv_heads`] | Boogu (28/7), Krea |
//! | 11 | stream order `[img,txt]` vs `[txt,img]` | [`StreamOrder`] | Lens/Ideogram (`[img,txt]`) vs Mage/Chroma/PID (`[txt,img]`) |
//! | 12 | dtype handed to SDPA after the rotation | [`RopeDtype`] | Ideogram/Anima/PID (promotion stands ⇒ f32 SDPA) vs Lens/Boogu/Mage/LTX (restored ⇒ bf16 SDPA) |
//!
//! Knob 12 was **not** in the original survey. It surfaced when the per-family fixtures were run at
//! each family's real dtype rather than in f32: every family builds f32 `(cos, sin)` tables, so the
//! rotation promotes a bf16 stream, and whether the family casts the result back decides whether its
//! attention runs in bf16 or f32. In an f32 fixture the two are indistinguishable. See [`RopeDtype`].
//!
//! Two further fields exist for **bit-exactness**, not expressiveness: [`AttnPrepSpec::rotation_axes`]
//! (whether a family rotates in `[B,S,H,D]` or after the transpose to `[B,H,S,D]`) and
//! [`NormDtype`]'s precision policies. Both rotations and RMSNorm act on the last axis and broadcast
//! over the head axis, so the two axis orders are numerically identical — but keeping each family's
//! *own* order means the migration is provably a no-op rather than "should be a no-op".
//!
//! ## Fusion is opt-in
//!
//! Knob 9's [`FusedQkvProjection`] is gated on [`set_fused_qkv`], which **defaults to `false`**
//! (Michael, 2026-08-16). A family that adopted the fused projection therefore runs its three
//! separate projections — bit-identically to its pre-P4 code — until a caller enables the toggle
//! before loading weights. The reason is reproducibility: the fused and unfused arms differ in the
//! GEMM's `N`, MLX selects its Metal tile shape from `(M, N, K)`, and off the self-hosted NAX build
//! the two can land a few ULP apart, which a chaos sampler turns into a different image at the same
//! seed. Nothing else in this module is gated — the prologue primitive always runs.
//!
//! ## What this module deliberately does NOT do
//!
//! * **It does not pack QK-norm scales.** `config/tier-integrity.jsonc` (SceneWorks) and its
//!   executable counterpart `gen_core::tier_integrity` forbid declaring or packing per-channel /
//!   per-token vectors — RMSNorm/LayerNorm/QK-norm scales, modulation and bias vectors — because no
//!   packer in the tree targets them at any tier. QK-norm weights are therefore *borrowed by
//!   reference* at call time ([`QkNormSpec`]) and never folded into a fused matrix.
//! * **It does not re-fuse what is already fused.** Dense `linear+bias` is already one `addmm`
//!   (`adapters.rs`, sc-2779); quantized `linear+bias` already goes through
//!   [`nn::quantized_matmul_with_bias`](crate::nn::quantized_matmul_with_bias); FLUX.2's single block
//!   already ships one `[q|k|v|mlp]` matrix; Qwen's vision tower already ships a fused `qkv`; the
//!   elementwise glue is already compiled; and MLX's SDPA is already one fused Metal kernel.
//! * **It does not fuse across a family boundary.** Nothing here builds a matrix shared by two
//!   families — see the group-size discussion on [`FusedQkvProjection`].
//!
//! ## Structural exemptions
//!
//! Eleven attention paths in this tree are *not* expressible here and stay on their native code.
//! Each one, with its specific deviation and file reference, is recorded in
//! [`EXEMPTIONS`] — a compiled-in table, so the justification cannot drift away from the code.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, mean_axis, multiply, rsqrt, split, split_sections,
    stack_axis,
};
use mlx_rs::{Array, Dtype};

use crate::adapters::{AdaptableLinear, LinearFacts};
use crate::nn::rope_rotate;
use crate::{Error, Result};

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Knobs
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Knob 2 + 3 + 7 — which rotation kernel (if any) the stream uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeStyle {
    /// No rotation at all. SD3 (knob 7) and Anima's DiT cross-attention / Krea's optional path
    /// (knob 3) select this; the tables are then ignored.
    None,
    /// **Adjacent-pair complex** (a.k.a. "interleaved"): lanes `(x[2i], x[2i+1])` are the real and
    /// imaginary parts of one complex number, rotated by `(cos[i], sin[i])` from a `[.., head_dim/2]`
    /// table. Lens, Mage, PID, Boogu, SeedVR2. Routed through the shared compiled-glue
    /// [`rope_rotate`] so the fusion the elementwise glue already provides is preserved.
    AdjacentPair,
    /// **Half-split** (`rotate_half`, the HF/diffusers convention): lane `i` pairs with lane
    /// `i + head_dim/2`, and the `cos`/`sin` tables are the **full** `head_dim` wide. Ideogram,
    /// Anima (`nn::apply_text_rope`), and the two vision towers (which are exempt for other
    /// reasons).
    RotateHalf,
    /// **Half-split, paired form** — the GPT-NeoX `apply_split_rotary_emb`: split the head into
    /// halves `(a, b)` and rotate the pair through the shared [`rope_rotate`] kernel, giving
    /// `concat(a·cos − b·sin, b·cos + a·sin)` from a **half**-width (`head_dim/2`) table. LTX
    /// (`mlx-gen-ltx/src/rope.rs`).
    ///
    /// This is the *same rotation* as [`RopeStyle::RotateHalf`] — feed `RotateHalf` the
    /// `[cos, cos]`/`[sin, sin]` tiling of this table and the numbers agree — but it is a separate
    /// arm rather than a table-preprocessing step for two reasons that both matter here: the table
    /// is genuinely half-width on disk (tiling it would allocate `head_dim`-wide tables for every
    /// block of a video DiT), and the op sequence is one fused `rope_rotate` instead of
    /// `split → negate → concat → mul → mul → add`, so collapsing the two arms would make the LTX
    /// migration a *numerically-should-match* rather than a bit-exact no-op.
    HalvesPaired,
}

/// Knob 1 — where the QK-norm sits relative to the head split, and over which axis it reduces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkNormPlacement {
    /// Per-head RMSNorm over `head_dim`, applied **after** the head split. The common case: Lens,
    /// Ideogram, PID, Chroma, Anima, SD3, Mochi, Mage, Boogu.
    PerHeadPostSplit,
    /// RMSNorm over the **full** `heads · head_dim` axis, applied **before** the head split. LTX
    /// (`transformer.rs` `norm_q`/`norm_k` over the flat projection) and Wan.
    FullDimPreSplit,
}

/// Knob 11 — which stream leads the joint sequence. Both orders are live in this tree and swapping
/// them yields a *running* model with garbage output and no shape error, so the order is named
/// rather than positional.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamOrder {
    /// `[image, text]` — Lens (`_build_joint_attention_mask` orders image first), Ideogram.
    ImageFirst,
    /// `[text, image]` — Mage (the scatter offsets), Chroma, PID.
    TextFirst,
}

/// Which axis order a family runs its QK-norm and rotation in. Not a semantic knob — the two are
/// numerically identical (both ops act on the last axis and broadcast over the head axis) — but
/// preserving each family's own order makes the migration a provable no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationAxes {
    /// `[B, S, H, D]` — rotate before the SDPA transpose (Lens, Mage, Boogu).
    TokenMajor,
    /// `[B, H, S, D]` — transpose first, then rotate (Ideogram, PID, Chroma).
    HeadMajor,
}

/// A `(cos, sin)` position table. Knobs 5 and 6 are expressed by *which* of
/// [`RopeSpec::q`] / [`RopeSpec::k`] is `Some`, and by them carrying **different** tables.
#[derive(Clone, Copy, Debug)]
pub struct RopeTables<'a> {
    pub cos: &'a Array,
    pub sin: &'a Array,
}

impl<'a> RopeTables<'a> {
    pub fn new(cos: &'a Array, sin: &'a Array) -> Self {
        Self { cos, sin }
    }
}

/// The precision policy a family's QK-norm runs under. Not a semantic knob, but the three policies
/// are genuinely different numbers on a bf16 stream, so a migration cannot pick one for a family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormDtype {
    /// No casts — the stream's own dtype in, the promoted result out (Lens, Ideogram, Mage, Boogu,
    /// SD3, Mochi).
    Native,
    /// Cast the stream to f32 before the norm and **leave the result f32** (Chroma's `proj_heads`,
    /// which deliberately runs the rest of the attention prologue in f32). Because that is a
    /// *stream-wide* policy rather than a norm detail, it promotes the **value** stream too — in
    /// Chroma, `proj_heads(.., None)` casts `v` to f32 even though it is never normalized.
    PromoteToF32,
    /// Cast both the stream and the weight to f32, normalize, then cast the result **back** to the
    /// stream's dtype (PID's `backbone::layers::rms` — the reference PixDiT `RMSNorm` computes in
    /// fp32 and casts back, which is load-bearing over the stack's ~60 norms on a bf16 decode).
    F32RoundTrip,
    /// The **eager** f32 formulation — `x·rsqrt(mean(x²) + eps)` followed by a separate multiply by
    /// the f32 weight, result left f32 — rather than MLX's fused `fast::rms_norm`.
    ///
    /// Mochi's `MochiRMSNorm(dim_head, eps, True)` is built as `RMSNorm(0, eps, False)` *then* a
    /// weight multiply, and the two are the same arithmetic but not necessarily the same rounding:
    /// the fused kernel is one pass over the row while this is three ops. Mochi's whole DiT is
    /// parity-pinned against that formulation, so it keeps it.
    EagerF32,
}

/// The QK-norm half of the spec. The weights are **borrowed**, never owned or packed — see the
/// module doc's tier-integrity note.
#[derive(Clone, Copy, Debug)]
pub struct QkNormSpec<'a> {
    /// `None` leaves the query stream unnormalized (SDXL, SVD, the PuLID tower).
    pub q: Option<&'a Array>,
    /// `None` leaves the key stream unnormalized.
    pub k: Option<&'a Array>,
    pub eps: f32,
    /// Knob 1.
    pub placement: QkNormPlacement,
    /// Precision policy — see [`NormDtype`].
    pub dtype: NormDtype,
}

impl<'a> QkNormSpec<'a> {
    /// No QK-norm on either stream — SD3's cross-attention, SDXL, SVD.
    pub fn none() -> Self {
        Self {
            q: None,
            k: None,
            eps: 0.0,
            placement: QkNormPlacement::PerHeadPostSplit,
            dtype: NormDtype::Native,
        }
    }

    /// The common case: per-head RMSNorm over `head_dim` on both streams, applied after the head
    /// split.
    pub fn per_head(q: &'a Array, k: &'a Array, eps: f32) -> Self {
        Self {
            q: Some(q),
            k: Some(k),
            eps,
            placement: QkNormPlacement::PerHeadPostSplit,
            dtype: NormDtype::Native,
        }
    }

    /// Knob 1's other arm: RMSNorm over the whole `heads · head_dim` projection, before the split.
    pub fn full_dim_pre_split(q: &'a Array, k: &'a Array, eps: f32) -> Self {
        Self {
            q: Some(q),
            k: Some(k),
            eps,
            placement: QkNormPlacement::FullDimPreSplit,
            dtype: NormDtype::Native,
        }
    }

    /// Select a non-default precision policy — see [`NormDtype`].
    pub fn with_dtype(mut self, dtype: NormDtype) -> Self {
        self.dtype = dtype;
        self
    }
}

/// The RoPE half of the spec (knobs 2–7).
#[derive(Clone, Copy, Debug)]
pub struct RopeSpec<'a> {
    pub style: RopeStyle,
    /// Knob 5/6 — `None` leaves the query stream unrotated; a table different from [`Self::k`]'s is
    /// the "separate q and k position tables" case.
    pub q: Option<RopeTables<'a>>,
    /// Knob 5/6 — `None` leaves the key stream unrotated (Mage's text stream is never rotated).
    pub k: Option<RopeTables<'a>>,
    /// Knob 4 — rotate only the leading `rot` channels of `head_dim` and pass the tail through
    /// unchanged (SeedVR2). `None` rotates the whole head.
    pub rot_dims: Option<i32>,
    /// What dtype the rotation hands to SDPA — see [`RopeDtype`]. **Not cosmetic**: it decides
    /// whether the attention runs in bf16 or f32.
    pub dtype: RopeDtype,
}

impl Default for RopeSpec<'_> {
    fn default() -> Self {
        Self {
            style: RopeStyle::None,
            q: None,
            k: None,
            rot_dims: None,
            dtype: RopeDtype::Promoted,
        }
    }
}

/// The **twelfth** knob, and the one a naive migration silently breaks: what dtype the rotated
/// stream is in when SDPA receives it.
///
/// Every family in this tree builds its `(cos, sin)` tables in **f32** while the DiT itself may run
/// **bf16**, so the rotation's multiplies promote. What the families do *next* splits them cleanly in
/// two, and the two are different models — one runs the attention in f32, the other in bf16:
///
/// * some cast the result back (`x.float() … .type_as(x)`), keeping the stream bf16;
/// * some do not, so the f32 promotion stands and SDPA consumes f32.
///
/// There is no safe default here, which is why this is an enum rather than a `bool` with a
/// convenient falsy value. `mlx-gen/tests/qkv_family_fixtures.rs` pins each family's choice at its
/// real dtype; getting it wrong is invisible in an f32 fixture and changes every bf16 render.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeDtype {
    /// **Leave the promotion standing.** No explicit casts: the rotation runs in whatever dtype
    /// MLX's promotion picks (f32 tables over a bf16 stream ⇒ f32) and the result *keeps* it, so a
    /// bf16 projection feeds an f32 SDPA. Ideogram (`transformer/block.rs`), Anima
    /// (`nn::apply_text_rope`), PID (`backbone/rope.rs`) — none of which cast back. Also Chroma and
    /// Mochi, whose streams are already f32 by the time they rotate, making it a no-op there.
    Promoted,
    /// **Restore the stream's dtype.** Compute promoted, then cast the result back to the input's
    /// dtype — the reference `x.float() … .type_as(x)` idiom. Lens (`.as_dtype(x.dtype())`), Boogu
    /// (`rope.rs`), Mage (`mage_layers.py:15-21`), LTX (`rope::apply_split_rotary_emb`).
    RestoreInput,
}

/// The whole prologue spec: geometry plus every knob.
#[derive(Clone, Copy, Debug)]
pub struct AttnPrepSpec<'a> {
    /// Query heads.
    pub heads: i32,
    /// Key/value heads. Knob 10 — `kv_heads < heads` is GQA; the repeat happens **after** RoPE,
    /// which is where Boogu and Krea place it and is *not* interchangeable with repeating before
    /// (the table is indexed per kv head).
    pub kv_heads: i32,
    pub head_dim: i32,
    pub qk_norm: QkNormSpec<'a>,
    pub rope: RopeSpec<'a>,
    /// Bit-exactness field, see [`RotationAxes`].
    pub rotation_axes: RotationAxes,
}

impl<'a> AttnPrepSpec<'a> {
    /// The common single-stream shape: `heads == kv_heads`, per-head QK-norm, token-major rotation.
    pub fn new(heads: i32, head_dim: i32) -> Self {
        Self {
            heads,
            kv_heads: heads,
            head_dim,
            qk_norm: QkNormSpec::none(),
            rope: RopeSpec::default(),
            rotation_axes: RotationAxes::TokenMajor,
        }
    }

    pub fn with_kv_heads(mut self, kv_heads: i32) -> Self {
        self.kv_heads = kv_heads;
        self
    }

    pub fn with_qk_norm(mut self, spec: QkNormSpec<'a>) -> Self {
        self.qk_norm = spec;
        self
    }

    pub fn with_rope(mut self, spec: RopeSpec<'a>) -> Self {
        self.rope = spec;
        self
    }

    pub fn with_rotation_axes(mut self, axes: RotationAxes) -> Self {
        self.rotation_axes = axes;
        self
    }

    /// `heads / kv_heads` — the GQA repeat factor (1 when not GQA).
    pub fn kv_groups(&self) -> i32 {
        if self.kv_heads <= 0 {
            1
        } else {
            self.heads / self.kv_heads
        }
    }
}

/// Knob 9 — how the caller's projections arrive.
pub enum QkvSource<'a> {
    /// One `[B, S, (heads + 2·kv_heads)·head_dim]` projection output, split on the last axis in
    /// `[q, k, v]` order (Lens, Ideogram, PID, SeedVR2 — and everything
    /// [`FusedQkvProjection`] packs).
    Packed(&'a Array),
    /// Three separate `[B, S, ·]` projection outputs (Mage, SANA, SD3, Chroma, Boogu).
    Separate {
        q: &'a Array,
        k: &'a Array,
        v: &'a Array,
    },
}

/// A prepared q/k/v triple. [`prepare`] returns it in **BHSD** (`[B, H, S, D]`) — the layout
/// [`sdpa_budgeted_bhsd`](crate::attention::sdpa_budgeted_bhsd) requires by contract.
#[derive(Clone, Debug)]
pub struct QkvHeads {
    pub q: Array,
    pub k: Array,
    pub v: Array,
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The primitive
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Split a `[B, S, ·]` projection into `[B, S, H, D]` heads.
pub fn split_heads(x: &Array, heads: i32, head_dim: i32) -> Result<Array> {
    let sh = x.shape();
    if sh.len() != 3 {
        return Err(Error::Msg(format!(
            "qkv::split_heads expects [B, S, heads·head_dim], got {sh:?}"
        )));
    }
    Ok(x.reshape(&[sh[0], sh[1], heads, head_dim])?)
}

/// `[B, S, H, D] ↔ [B, H, S, D]` — the SDPA transpose (its own inverse).
pub fn transpose_heads(x: &Array) -> Result<Array> {
    Ok(x.transpose_axes(&[0, 2, 1, 3])?)
}

/// `[B, H, S, D] → [B, S, H·D]` — the post-SDPA merge every migrated family shares.
pub fn merge_heads(x: &Array) -> Result<Array> {
    let sh = x.shape();
    if sh.len() != 4 {
        return Err(Error::Msg(format!(
            "qkv::merge_heads expects [B, H, S, D], got {sh:?}"
        )));
    }
    Ok(x.transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[sh[0], sh[2], sh[1] * sh[3]])?)
}

/// Knob 10 — repeat each KV head `groups` times so it matches the query head count. Placed
/// **after** RoPE by every GQA family here. `axis` names which of the two layouts `x` is in.
pub fn repeat_kv(x: &Array, groups: i32, axes: RotationAxes) -> Result<Array> {
    if groups <= 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    if sh.len() != 4 {
        return Err(Error::Msg(format!(
            "qkv::repeat_kv expects a rank-4 stream, got {sh:?}"
        )));
    }
    // The head axis is 2 in token-major `[B,S,H,D]` and 1 in head-major `[B,H,S,D]`.
    let head_axis = match axes {
        RotationAxes::TokenMajor => 2usize,
        RotationAxes::HeadMajor => 1usize,
    };
    let expanded = x.expand_dims((head_axis + 1) as i32)?;
    let mut target: Vec<i32> = sh.to_vec();
    target.insert(head_axis + 1, groups);
    let broadcast = broadcast_to(&expanded, &target[..])?;
    let mut merged: Vec<i32> = sh.to_vec();
    merged[head_axis] = sh[head_axis] * groups;
    Ok(broadcast.reshape(&merged[..])?)
}

/// Knob 1 — RMSNorm over the last axis, under the precision policy from `spec`.
///
/// The weight is borrowed; nothing here mutates or packs it (tier-integrity).
fn qk_rms(x: &Array, weight: &Array, eps: f32, dtype: NormDtype) -> Result<Array> {
    match dtype {
        NormDtype::Native => Ok(rms_norm(x, weight, eps)?),
        NormDtype::PromoteToF32 => Ok(rms_norm(&x.as_dtype(Dtype::Float32)?, weight, eps)?),
        NormDtype::F32RoundTrip => {
            let out = rms_norm(
                &x.as_dtype(Dtype::Float32)?,
                &weight.as_dtype(Dtype::Float32)?,
                eps,
            )?;
            Ok(out.as_dtype(x.dtype())?)
        }
        NormDtype::EagerF32 => {
            let xf = x.as_dtype(Dtype::Float32)?;
            let ms = mean_axis(&xf.square()?, -1, true)?;
            let normed = multiply(&xf, &rsqrt(&add(&ms, Array::from_f32(eps))?)?)?;
            Ok(multiply(&normed, &weight.as_dtype(Dtype::Float32)?)?)
        }
    }
}

/// Knob 2/4 — the rotation itself, over a rank-4 stream in either layout.
///
/// `x` is `[B, S, H, D]` (token-major) or `[B, H, S, D]` (head-major); the table is broadcast over
/// the head axis and indexed by the token axis. Exposed publicly because knob 8
/// ("concat-then-RoPE", Chroma) needs to rotate a stream that has *already* been joined, which is a
/// call-order choice rather than a parameter.
pub fn apply_rope(
    x: &Array,
    tables: RopeTables<'_>,
    style: RopeStyle,
    axes: RotationAxes,
    rot_dims: Option<i32>,
    dtype: RopeDtype,
) -> Result<Array> {
    if style == RopeStyle::None {
        return Ok(x.clone());
    }
    let sh = x.shape();
    if sh.len() != 4 {
        return Err(Error::Msg(format!(
            "qkv::apply_rope expects a rank-4 stream, got {sh:?}"
        )));
    }
    let head_dim = sh[3];
    // Knob 4 — partial RoPE: rotate the leading `rot` channels, pass the tail through untouched.
    if let Some(rot) = rot_dims {
        if rot < 0 || rot > head_dim {
            return Err(Error::Msg(format!(
                "qkv::apply_rope: rot_dims {rot} outside [0, {head_dim}]"
            )));
        }
        if rot == 0 {
            return Ok(x.clone());
        }
        if rot != head_dim {
            let mut parts = split_sections(x, &[rot], 3)?;
            let tail = parts.swap_remove(1);
            let head = parts.swap_remove(0);
            let rotated = apply_rope(&head, tables, style, axes, None, dtype)?;
            // The tail must match the rotated head's dtype — the rotation can widen the head.
            let tail = if tail.dtype() == rotated.dtype() {
                tail
            } else {
                tail.as_dtype(rotated.dtype())?
            };
            return Ok(concatenate_axis(&[&rotated, &tail], 3)?);
        }
    }

    let in_dtype = x.dtype();
    // `RestoreInput` is the `x.float() … .type_as(x)` idiom: promote the stream (and, for symmetry
    // with the reference implementations, the tables) to f32, then cast the RESULT back at the end.
    // `Promoted` casts nothing and lets MLX's own promotion stand all the way into SDPA. The
    // arithmetic is identical either way — bf16→f32 is exact and both multiply in f32 — so the ONLY
    // difference is the final cast, which is exactly the thing that differs between families.
    let promote = dtype == RopeDtype::RestoreInput;
    let x = if promote && in_dtype != Dtype::Float32 {
        x.as_dtype(Dtype::Float32)?
    } else {
        x.clone()
    };
    let sh = x.shape();
    let (d0, d1, d2) = (sh[0], sh[1], sh[2]);
    let widen = |t: &Array| -> Result<Array> {
        Ok(if promote && t.dtype() != Dtype::Float32 {
            t.as_dtype(Dtype::Float32)?
        } else {
            t.clone()
        })
    };
    // Reshape both tables to the rank-4 broadcast shape this stream needs.
    let tables_for = |width: i32| -> Result<(Array, Array)> {
        let target = expect_table(tables, axes, [d0, d1, d2], width)?;
        Ok(match target {
            // Already broadcast-shaped (LTX's `[B, H, T, head_dim/2]`) — pass through untouched, so
            // the migrated call is the *identical* op sequence with no inserted reshape.
            None => (widen(tables.cos)?, widen(tables.sin)?),
            Some(shape) => (
                widen(tables.cos)?.reshape(&shape[..])?,
                widen(tables.sin)?.reshape(&shape[..])?,
            ),
        })
    };

    let out = match style {
        RopeStyle::None => unreachable!("handled above"),
        RopeStyle::AdjacentPair => {
            let half = head_dim / 2;
            let (cos, sin) = tables_for(half)?;
            let x5 = x.reshape(&[d0, d1, d2, half, 2])?;
            let lanes = split(&x5, 2, 4)?;
            let real = lanes[0].reshape(&[d0, d1, d2, half])?;
            let imag = lanes[1].reshape(&[d0, d1, d2, half])?;
            // The shared compiled-glue kernel (`nn::rope_rotate`) — one fused op with glue on,
            // the identical six eager ops with it off.
            let (out_real, out_imag) = rope_rotate(&real, &imag, &cos, &sin)?;
            stack_axis(&[out_real, out_imag], 4)?.reshape(&[d0, d1, d2, head_dim])?
        }
        RopeStyle::RotateHalf => {
            let (cos, sin) = tables_for(head_dim)?;
            let halves = split(&x, 2, 3)?;
            let rotated = concatenate_axis(&[&halves[1].negative()?, &halves[0]], 3)?;
            add(&multiply(&x, &cos)?, &multiply(&rotated, &sin)?)?
        }
        RopeStyle::HalvesPaired => {
            let (cos, sin) = tables_for(head_dim / 2)?;
            let halves = split(&x, 2, 3)?;
            let (first, second) = rope_rotate(&halves[0], &halves[1], &cos, &sin)?;
            concatenate_axis(&[&first, &second], 3)?
        }
    };
    // The one line that separates the two families of families. Under `Promoted` the promotion
    // reaches SDPA; under `RestoreInput` it stops here.
    Ok(match dtype {
        RopeDtype::Promoted => out,
        RopeDtype::RestoreInput if out.dtype() == in_dtype => out,
        RopeDtype::RestoreInput => out.as_dtype(in_dtype)?,
    })
}

/// How a family's `(cos, sin)` table is indexed. All four are live in this tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TableLayout {
    /// `[tokens, width]` — one row per token, broadcast over batch AND heads. Lens, PID, Mage,
    /// Boogu, Chroma, Anima.
    Shared,
    /// `[batch, tokens, width]` — Ideogram's 3-D MRoPE, whose table depends on each sample's grid.
    PerBatch,
    /// `[tokens, heads, width]` — **Mochi**, whose `pos_frequencies` are per attention head, so
    /// every head rotates by its own angles.
    PerHead,
    /// Already rank-4 and elementwise broadcast-compatible with the stream — **LTX**, whose
    /// `precompute_split_freqs_cis` emits `[B, H, T, head_dim/2]` (per batch *and* per head, which
    /// none of the three rank-≤3 layouts can express). Used verbatim, with no reshape.
    PreBroadcast,
}

/// Resolve the table to the rank-4 broadcast shape this stream needs, or `None` when it is already
/// in that shape ([`TableLayout::PreBroadcast`]).
///
/// For the rank-≤3 layouts only the element count is load-bearing — a table arrives variously as
/// `[S, w]`, `[1, S, w]`, `[B, S, w]` or `[S, H, w]` across the families — but a *wrong-length*
/// table is the classic silent-garbage RoPE bug (every token reads row 0 under a permissive
/// broadcast), so a mismatch is an error, never a reshape.
fn expect_table(
    tables: RopeTables<'_>,
    axes: RotationAxes,
    dims: [i32; 3],
    width: i32,
) -> Result<Option<[i32; 4]>> {
    let [d0, d1, d2] = dims;
    let (tokens, heads) = match axes {
        RotationAxes::TokenMajor => (d1, d2),
        RotationAxes::HeadMajor => (d2, d1),
    };
    let shared = tokens * width;
    let mut layout: Option<TableLayout> = None;
    for (name, t) in [("cos", tables.cos), ("sin", tables.sin)] {
        let sh = t.shape();
        let n: i32 = sh.iter().product();
        // A rank-4 table is taken to be pre-broadcast — but only if every axis genuinely lines up,
        // so a rank-4 table of the wrong geometry is still an error rather than a bad broadcast.
        let found = if sh.len() == 4 {
            let target = [d0, d1, d2, width];
            if sh
                .iter()
                .zip(target.iter())
                .all(|(&a, &b)| a == 1 || a == b)
            {
                TableLayout::PreBroadcast
            } else {
                return Err(Error::Msg(format!(
                    "qkv::apply_rope: {name} table {sh:?} is rank-4 but does not broadcast against \
                     the stream's {target:?}"
                )));
            }
        // The `Shared` arm is checked first, so a `batch == 1` (or `heads == 1`) table that is
        // ambiguous between the layouts resolves to the cheapest broadcast, which is identical.
        } else if n == shared {
            TableLayout::Shared
        } else if n == d0 * shared {
            TableLayout::PerBatch
        } else if n == heads * shared {
            TableLayout::PerHead
        } else {
            return Err(Error::Msg(format!(
                "qkv::apply_rope: {name} table has {n} elements but the stream needs {tokens} × \
                 {width}, or {d0} × that per batch, or {heads} × that per head; table shape {sh:?}"
            )));
        };
        match layout {
            None => layout = Some(found),
            Some(first) if first == found => {}
            Some(first) => {
                return Err(Error::Msg(format!(
                "qkv::apply_rope: cos and sin tables disagree on layout ({first:?} vs {found:?})"
            )))
            }
        }
    }
    let layout = layout.expect("both tables were classified");
    Ok(match (axes, layout) {
        (_, TableLayout::PreBroadcast) => None,
        (RotationAxes::TokenMajor, TableLayout::Shared) => Some([1, tokens, 1, width]),
        (RotationAxes::HeadMajor, TableLayout::Shared) => Some([1, 1, tokens, width]),
        (RotationAxes::TokenMajor, TableLayout::PerBatch) => Some([d0, tokens, 1, width]),
        (RotationAxes::HeadMajor, TableLayout::PerBatch) => Some([d0, 1, tokens, width]),
        (RotationAxes::TokenMajor, TableLayout::PerHead) => Some([1, tokens, heads, width]),
        (RotationAxes::HeadMajor, TableLayout::PerHead) => {
            // A per-head rank-≤3 table in head-major layout would need a transpose, which would
            // change the op sequence of whichever family adopted it. No family does today (LTX, the
            // only per-head *and* head-major family, ships its table pre-broadcast); refuse rather
            // than silently reshape into a wrong broadcast.
            return Err(Error::Msg(
                "qkv::apply_rope: a rank-≤3 per-head RoPE table is only supported in token-major \
                 layout (Mochi's `[seq, heads, head_dim/2]` tables); a head-major family must ship \
                 a pre-broadcast `[B, H, T, width]` table (LTX)"
                    .to_string(),
            ));
        }
    })
}

/// **The primitive.** Runs one stream's full prologue and returns q/k/v in **BHSD**.
///
/// Stages, in order (each optional stage is skipped, not defaulted, when its knob says so):
/// 1. split `src` into q/k/v (knob 9);
/// 2. full-dim QK-norm on the flat `[B, S, H·D]` projection, if [`QkNormPlacement::FullDimPreSplit`]
///    (knob 1);
/// 3. head split to `[B, S, H, D]`;
/// 4. per-head QK-norm over `head_dim`, if [`QkNormPlacement::PerHeadPostSplit`] (knob 1);
/// 5. transpose to `[B, H, S, D]` *first* when [`RotationAxes::HeadMajor`];
/// 6. RoPE per stream, style, table and `rot_dims` (knobs 2–7);
/// 7. repeat-KV (knob 10) — **after** RoPE;
/// 8. transpose to BHSD if not already there.
pub fn prepare(src: QkvSource<'_>, spec: &AttnPrepSpec<'_>) -> Result<QkvHeads> {
    let (mut q, mut k, v) = match src {
        QkvSource::Packed(packed) => {
            let q_out = spec.heads * spec.head_dim;
            let kv_out = spec.kv_heads * spec.head_dim;
            let sh = packed.shape();
            let last = *sh.last().ok_or_else(|| {
                Error::Msg("qkv::prepare: packed projection has no axes".to_string())
            })?;
            if last != q_out + 2 * kv_out {
                return Err(Error::Msg(format!(
                    "qkv::prepare: packed projection is {last} wide, expected {} \
                     (heads {} · head_dim {} + 2 · kv_heads {} · head_dim)",
                    q_out + 2 * kv_out,
                    spec.heads,
                    spec.head_dim,
                    spec.kv_heads
                )));
            }
            let axis = (sh.len() - 1) as i32;
            let mut parts = split_sections(packed, &[q_out, q_out + kv_out], axis)?;
            let v = parts.swap_remove(2);
            let k = parts.swap_remove(1);
            let q = parts.swap_remove(0);
            (q, k, v)
        }
        QkvSource::Separate { q, k, v } => (q.clone(), k.clone(), v.clone()),
    };

    // Stage 2 — full-dim QK-norm on the flat projection, before any head split (knob 1).
    if spec.qk_norm.placement == QkNormPlacement::FullDimPreSplit {
        if let Some(w) = spec.qk_norm.q {
            q = qk_rms(&q, w, spec.qk_norm.eps, spec.qk_norm.dtype)?;
        }
        if let Some(w) = spec.qk_norm.k {
            k = qk_rms(&k, w, spec.qk_norm.eps, spec.qk_norm.dtype)?;
        }
    }

    // Stage 3 — head split.
    let mut q = split_heads(&q, spec.heads, spec.head_dim)?;
    let mut k = split_heads(&k, spec.kv_heads, spec.head_dim)?;
    let mut v = split_heads(&v, spec.kv_heads, spec.head_dim)?;

    // Stage 4 — per-head QK-norm over head_dim (knob 1).
    if spec.qk_norm.placement == QkNormPlacement::PerHeadPostSplit {
        if let Some(w) = spec.qk_norm.q {
            q = qk_rms(&q, w, spec.qk_norm.eps, spec.qk_norm.dtype)?;
        }
        if let Some(w) = spec.qk_norm.k {
            k = qk_rms(&k, w, spec.qk_norm.eps, spec.qk_norm.dtype)?;
        }
    }

    // Stage 4b — `PromoteToF32` is a stream-wide policy, so an unnormalized stream (always `v`,
    // and `q`/`k` when their weight is `None`) is promoted explicitly rather than left narrow.
    if spec.qk_norm.dtype == NormDtype::PromoteToF32 {
        for s in [&mut q, &mut k, &mut v] {
            if s.dtype() != Dtype::Float32 {
                *s = s.as_dtype(Dtype::Float32)?;
            }
        }
    }

    // Stage 5 — head-major families transpose before rotating.
    if spec.rotation_axes == RotationAxes::HeadMajor {
        q = transpose_heads(&q)?;
        k = transpose_heads(&k)?;
        v = transpose_heads(&v)?;
    }

    // Stage 6 — RoPE (knobs 2–7).
    if spec.rope.style != RopeStyle::None {
        if let Some(t) = spec.rope.q {
            q = apply_rope(
                &q,
                t,
                spec.rope.style,
                spec.rotation_axes,
                spec.rope.rot_dims,
                spec.rope.dtype,
            )?;
        }
        if let Some(t) = spec.rope.k {
            k = apply_rope(
                &k,
                t,
                spec.rope.style,
                spec.rotation_axes,
                spec.rope.rot_dims,
                spec.rope.dtype,
            )?;
        }
    }

    // Stage 7 — GQA repeat-KV, AFTER the rotation (knob 10).
    let groups = spec.kv_groups();
    if groups > 1 {
        k = repeat_kv(&k, groups, spec.rotation_axes)?;
        v = repeat_kv(&v, groups, spec.rotation_axes)?;
    }

    // Stage 8 — hand back BHSD, the shared attention seam's contract.
    if spec.rotation_axes == RotationAxes::TokenMajor {
        q = transpose_heads(&q)?;
        k = transpose_heads(&k)?;
        v = transpose_heads(&v)?;
    }
    Ok(QkvHeads { q, k, v })
}

impl StreamOrder {
    /// Knob 11 — concatenate two prepared streams along the **token** axis of BHSD (axis 2) in this
    /// order. Knob 8's "concat-then-RoPE" is this call followed by [`apply_rope`]; "RoPE-then-concat"
    /// is [`prepare`] (which rotates) followed by this call.
    pub fn join(&self, image: &QkvHeads, text: &QkvHeads) -> Result<QkvHeads> {
        let (a, b) = match self {
            StreamOrder::ImageFirst => (image, text),
            StreamOrder::TextFirst => (text, image),
        };
        Ok(QkvHeads {
            q: concatenate_axis(&[&a.q, &b.q], 2)?,
            k: concatenate_axis(&[&a.k, &b.k], 2)?,
            v: concatenate_axis(&[&a.v, &b.v], 2)?,
        })
    }

    /// The token count that leads the joint sequence — the boundary a caller splits the attention
    /// output back at.
    pub fn lead_tokens(&self, image_tokens: i32, text_tokens: i32) -> i32 {
        match self {
            StreamOrder::ImageFirst => image_tokens,
            StreamOrder::TextFirst => text_tokens,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The adapter/quant-aware fused QKV projection
// ─────────────────────────────────────────────────────────────────────────────────────────────

thread_local! {
    /// Request/thread-scoped, matching [`crate::nn::set_compile_glue`]'s scope (sc-18316): a render
    /// loop is synchronous, so a per-thread toggle cannot race a concurrent render.
    ///
    /// **Defaults to `false` — fusion is opt-in.** See [`set_fused_qkv`] for why.
    static FUSED_QKV: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Enable/disable read-side QKV projection fusion on the current thread.
/// **Fusion is off unless a caller turns it on** (Michael, 2026-08-16).
///
/// ## Why opt-in
///
/// The fused and the unfused arm are *algebraically* identical, and their **tensors** are equal
/// byte-for-byte (see [`FusedQkvProjection`]). Their **outputs** are not equal on every host: MLX
/// picks its Metal GEMM tile shape — and therefore its accumulation order — from `(M, N, K)`, and
/// the two arms differ in `N` by construction (`3·out` versus `out`). Float addition is not
/// associative, so off the self-hosted macOS 26.2 NAX build the projection can land a few ULP apart,
/// and a chaos sampler turns a few ULP into a visibly different image at the same seed. That is a
/// reproducibility regression, and the speedup it would buy is P6 output that does not exist yet, so
/// defaulting on would trade a measured guarantee for an unmeasured one. When P6 has numbers, the
/// decision to flip this constant is a one-line, deliberate edit with a test
/// (`qkv/tests.rs`'s `fusion_is_opt_in_so_an_eligible_triple_is_split_by_default`) that must be
/// changed alongside it.
///
/// ## How to opt in
///
/// Call this (or hold a [`FusedQkvGuard`], which restores the prior value on drop) **before the
/// weights are loaded**, and keep it set across every [`FusedQkvProjection::repack`] the load path
/// runs — `quantize` and `cast_weights` both re-pack, so a guard scoped to the constructor alone is
/// not enough:
///
/// ```ignore
/// let _fused = mlx_gen::qkv::FusedQkvGuard::set(true);
/// let model = Family::from_weights(&weights, /* … */)?; // packs where the predicate allows
/// ```
///
/// The toggle is read when a [`FusedQkvProjection`] chooses its representation — at
/// [`FusedQkvProjection::new`] and [`FusedQkvProjection::repack`] — **not** on every forward. That
/// is deliberate: a projection holds *either* one packed matrix *or* three separate bases, never
/// both (see [`FusedQkvProjection`]), so "unfused" is a different object rather than a different
/// branch over the same tensors. A benchmark A/B therefore sets the toggle before loading each arm,
/// which is exactly what production does, and the fused-off baseline costs no extra slicing.
pub fn set_fused_qkv(on: bool) {
    FUSED_QKV.set(on);
}

/// Whether QKV projection fusion is currently enabled on this thread. `false` unless a caller opted
/// in — see [`set_fused_qkv`].
pub fn fused_qkv() -> bool {
    FUSED_QKV.get()
}

/// RAII guard restoring the prior [`fused_qkv`] value on drop — including on an early `?`. **This
/// is the opt-in handle**: hold one across the whole load (construction *and* every `quantize` /
/// `cast_weights`, each of which re-packs), not just across the constructor.
#[must_use = "dropping the guard restores the prior fused-QKV setting"]
pub struct FusedQkvGuard {
    prev: bool,
}

impl FusedQkvGuard {
    pub fn set(on: bool) -> Self {
        Self {
            prev: FUSED_QKV.replace(on),
        }
    }
}

impl Drop for FusedQkvGuard {
    fn drop(&mut self) {
        FUSED_QKV.set(self.prev);
    }
}

/// Why a q/k/v triple could not be packed into one matrix. Surfaced (rather than swallowed) so a
/// family that *expects* fusion can assert it, and so the Boogu case is legible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoFusion {
    /// Fusion is switched off on this thread — the P6 baseline arm **and the default**, because
    /// fusion is opt-in ([`set_fused_qkv`]). A projection that a caller expected to be packed and
    /// finds reporting this reason did not hit an unsafe shape: nobody turned the toggle on before
    /// the weights were loaded.
    Disabled,
    /// One or more of q/k/v carries a LoRA/LoKr adapter. **The designed-against failure is silently
    /// dropping it**, so fusion is refused instead — see [`FusedQkvProjection`].
    AdaptersInstalled,
    /// A dense base and a quantized base in the same triple.
    MixedBases,
    /// Quantized bases disagree on `bits` or `group_size`.
    QuantMismatch { bits: (i32, i32), group: (i32, i32) },
    /// A projection's `out_features` is not a multiple of the quantization group size.
    /// **This is Boogu**: `GROUP_SIZE = 32` (explicit, non-default) with kv `out = 7 · 120 = 840`,
    /// and `840 % 32 = 8`.
    OutFeaturesNotGroupAligned { out: i32, group: i32 },
    /// A projection's `in_features` is not a multiple of the group size — the input axis is where
    /// the affine groups actually run, so this can never be packed around.
    ///
    /// **Not reachable through any current construction path**, and deliberately kept anyway: MLX's
    /// `quantize` rejects a misaligned last dimension outright, and for a base loaded pre-quantized
    /// `AdaptableLinear::base_shape` *derives* `in` as `scales_cols · group_size`, so it is a
    /// multiple of the group by arithmetic. This arm is the belt-and-braces assertion that the
    /// derivation's own stated precondition ("exact only when `in % group_size == 0`") still holds
    /// — a guard, not a branch with a test behind it, because a test for it would have to fake a
    /// state neither path can produce.
    InFeaturesNotGroupAligned { inp: i32, group: i32 },
    /// [`FusedQkvProjection::unfuse`] was called explicitly; call
    /// [`repack`](FusedQkvProjection::repack) to re-attempt.
    Unfused,
    /// The three projections do not read the same input width, so they cannot share one matmul.
    InputWidthMismatch,
    /// Some projections carry a dense bias and others do not.
    BiasMismatch,
}

/// Which of the three projections a host's routing arm addresses — the selector for
/// [`FusedQkvProjection::part_mut`] (mutate) and [`FusedQkvProjection::part_facts`] (probe).
///
/// Named rather than positional: the pack order is `[q | k | v]` on the output axis and a swapped
/// index yields a *running* model with garbage attention and no shape error whenever the three share
/// an out width — which is every non-GQA family here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkvPart {
    Q,
    K,
    V,
}

impl QkvPart {
    /// The projection's position in the packed row order.
    fn index(self) -> usize {
        match self {
            QkvPart::Q => 0,
            QkvPart::K => 1,
            QkvPart::V => 2,
        }
    }
}

/// Three separate q/k/v projections, each applying its own adapter stack.
///
/// Boxed inside [`Backing`] so the enum is not sized by its larger variant: a `FusedQkvProjection`
/// is held per attention block, and paying the split variant's footprint on every packed block would
/// give back part of what packing is for.
#[derive(Clone)]
struct SplitQkv {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
}

/// The representation a [`FusedQkvProjection`] currently holds. **Exactly one of the two** — the
/// packed matrix is not a cache over the bases, it *replaces* them.
#[derive(Clone)]
enum Backing {
    Split(Box<SplitQkv>),
    /// One packed `[q_out + k_out + v_out, in]` matrix plus the two split points.
    Packed {
        linear: AdaptableLinear,
        splits: [i32; 2],
    },
}

/// **Adapter- and quantization-aware fused QKV projection.**
///
/// Three logically separate `to_q` / `to_k` / `to_v` projections presented as one call, backed by a
/// single packed matrix when — and only when — a caller has **opted in** and that is provably safe.
///
/// ## Opt-in (the default is split)
///
/// [`set_fused_qkv`] defaults to `false`, so a `FusedQkvProjection` built with no explicit opt-in
/// holds the three separate bases and reports [`NoFusion::Disabled`]. Everything below — the packing
/// predicate, the adapter guard, the effective-bits argument — describes what happens **once fusion
/// is enabled**; with it off this type is a thin pass-through over three [`AdaptableLinear`]s and is
/// bit-identical to the code each family open-coded before P4.
///
/// The default is off because the fused and unfused arms can differ by a few ULP on a host whose
/// Metal GEMM kernel selection differs across the two output widths, and a chaos sampler cascades
/// that into a visibly different image at the same seed — a reproducibility regression traded
/// against a speedup P6 has not yet measured. [`set_fused_qkv`] carries the full argument.
///
/// ## One representation, not two
///
/// The packed matrix **replaces** the three bases; it is not built alongside them. That is a
/// memory-correctness requirement, not a style choice: a DiT's q/k/v projections are a large share
/// of its resident weights, so retaining a packed *copy* on top of the originals would raise weight
/// residency by roughly the size of the whole attention projection set in every block — the opposite
/// of what epic 18304 is for. The backing is therefore an either/or, [`Self::unfuse`] slices
/// the packed matrix back into three projections when one is needed (exactly, by row range), and
/// [`Self::repack`] returns to the packed form.
///
/// ## Adapters (the designed-against failure)
///
/// [`AdaptableLinear`]'s contract is `base(x) + Σ adapter.residual(x)`, and the base weight is never
/// mutated. A packed matrix is built from the **bases only**, so a fused forward that ignored the
/// adapter stack would *silently drop every installed LoRA/LoKr* — a running model producing
/// un-adapted output with no error.
///
/// This type makes that unrepresentable rather than merely checked. [`Self::parts_mut`], the only
/// way to reach an [`AdaptableLinear`] to install an adapter on, **unpacks first** and leaves the
/// projection in the split representation. A subsequent [`Self::repack`] re-runs the safety predicate,
/// which refuses while any live adapter is installed ([`NoFusion::AdaptersInstalled`]). So there is
/// no ordering in which a packed matrix and a live adapter coexist, and no forward-time check that
/// could be forgotten.
///
/// Applying the residuals *inside* the fusion was rejected: the three adapters have independent
/// ranks and scales and their residuals are per-projection, so folding them would mean three
/// low-rank matmuls plus the packed one — strictly more work than the unfused path it replaced.
///
/// A *disabled* adapter (`scale == 0`) does not block the pack: `AdaptableLinear::apply_adapters`
/// rule 1 skips it outright, so "installed at scale 0" is byte-identical to "never installed".
///
/// ## Quantization — effective bits must not change
///
/// MLX's group-wise affine quantization runs its groups along the **input** axis of a `[out, in]`
/// weight (`scales`/`biases` are `[out, in/group]`). Packing therefore concatenates on the **output**
/// axis (axis 0) of the weight, the scales and the biases together, which keeps every row's own
/// groups intact and every stored scale attached to exactly the codes it came from — the packed
/// matrix dequantizes to the byte-identical concatenation of the three originals, and
/// [`Self::unfuse`] inverts it by the same row ranges. That byte-level claim is exact on every
/// host and is asserted on the **tensors** (`qkv/tests.rs`'s
/// `unfusing_recovers_the_three_bases_exactly`), not inferred from a forward — see
/// [`Self::forward_packed`] for why a forward comparison is the weaker instrument.
///
/// The guard is deliberately **stricter** than that argument requires: a triple is packed only when
/// every part shares `bits` and `group_size`, every `in_features` is group-aligned (which the affine
/// format needs anyway), **and** every `out_features` is group-aligned. The `out` rule is the
/// conservative one, and it is the one that matters in practice: a family whose `out_features` is
/// not a multiple of its own group size is a family whose packer chose a non-default group, and
/// mixing it with anything else — or reshaping the packed output head-major — would cross a group
/// boundary. Boogu is exactly that family (`mlx-gen-boogu/src/quant.rs`: `GROUP_SIZE = 32`, kv
/// `out = 840`, `840 % 32 = 8`, `head_dim = 120`, `120 % 32 ≠ 0`), so its kv projections are
/// refused by construction and reported as [`NoFusion::OutFeaturesNotGroupAligned`]. Boogu also
/// never shares a packed matrix with any other family, because nothing here builds a matrix that
/// spans two families at all.
///
/// Nothing about the *activation* path changes: a packed base still goes through
/// [`nn::quantized_matmul_with_bias`](crate::nn::quantized_matmul_with_bias) with the activations
/// fed as-is, exactly as an unpacked one does. QK-norm scales are never packed (see the module doc's
/// tier-integrity note) — this type only ever touches projection weights.
///
/// ## Activation receipt
///
/// Every **forward** records
/// [`FUSED_ATTENTION_PRIMITIVES`](crate::diagnostics::FUSED_ATTENTION_PRIMITIVES) as
/// [`Applied`](crate::diagnostics::ToggleDisposition::Applied) or
/// [`Fallback`](crate::diagnostics::ToggleDisposition::Fallback) on the active diagnostic scope.
/// The P6 matrix requires that receipt rather than inferring the toggle from a flag or from timing,
/// and it is what proves in-process that the fused path is not inert. It is recorded from the
/// forward rather than from construction because a diagnostic scope is per request while a model is
/// loaded outside one.
#[derive(Clone)]
pub struct FusedQkvProjection {
    backing: Backing,
    /// Why the pack was refused, when it was. `None` while packed.
    refusal: Option<NoFusion>,
}

impl std::fmt::Debug for FusedQkvProjection {
    /// Reports the live representation, not the weights: which backing is held, the packed shape when
    /// there is one, and the refusal when there is not. Hand-written because [`AdaptableLinear`] is not
    /// `Debug` (an `Array` would dump its buffer).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedQkvProjection")
            .field("fused", &self.fusion_engaged())
            .field("packed_shape", &self.packed_shape())
            .field("refusal", &self.refusal)
            .finish()
    }
}

impl FusedQkvProjection {
    /// Wrap three projections and attempt the pack immediately, honouring [`fused_qkv`] — which is
    /// `false` unless the caller opted in, so this leaves the split backing by default.
    pub fn new(q: AdaptableLinear, k: AdaptableLinear, v: AdaptableLinear) -> Self {
        let mut me = Self {
            backing: Backing::Split(Box::new(SplitQkv { q, k, v })),
            refusal: None,
        };
        me.repack();
        me
    }

    /// Attempt to return to the packed representation. A no-op when already packed.
    ///
    /// **Must** be called after anything that changes a base weight or an adapter stack — the packed
    /// matrix is built from the bases, so it is not a view of them.
    pub fn repack(&mut self) {
        if let Backing::Packed { .. } = self.backing {
            self.refusal = None;
            return;
        }
        if !fused_qkv() {
            self.refusal = Some(NoFusion::Disabled);
            return;
        }
        let Backing::Split(parts) = &self.backing else {
            unreachable!("the packed arm returned above");
        };
        match try_pack(&parts.q, &parts.k, &parts.v) {
            Ok((linear, splits)) => {
                self.backing = Backing::Packed { linear, splits };
                self.refusal = None;
            }
            Err(why) => {
                self.refusal = Some(why);
            }
        }
    }

    /// Drop back to three separate projections, slicing the packed matrix by row range if needed.
    /// Idempotent. This is what [`Self::parts_mut`] runs first, and what a caller that is about to
    /// install adapters should call explicitly.
    pub fn unfuse(&mut self) -> Result<()> {
        if let Backing::Packed { linear, splits } = &self.backing {
            let (q, k, v) = split_parts(linear, *splits)?;
            self.backing = Backing::Split(Box::new(SplitQkv { q, k, v }));
            self.refusal = Some(NoFusion::Unfused);
        }
        Ok(())
    }

    /// `true` when this projection currently holds one packed matrix. Tests and the P6 receipt
    /// assert this to prove the fusion is not inert.
    pub fn fusion_engaged(&self) -> bool {
        matches!(self.backing, Backing::Packed { .. })
    }

    /// Why the pack was refused, if it was.
    pub fn refusal(&self) -> Option<&NoFusion> {
        self.refusal.as_ref()
    }

    /// The packed matrix's logical `[out, in]` shape, or `None` while split. Lets a caller (and the
    /// tests) observe *which* representation is live rather than infer it.
    pub fn packed_shape(&self) -> Option<Vec<i32>> {
        match &self.backing {
            Backing::Packed { linear, .. } => Some(linear.base_shape()),
            Backing::Split(_) => None,
        }
    }

    /// The packed matrix's declared `(bits, group_size)`, or `None` when split or still dense. The
    /// **effective-bits receipt**: packing must carry its parts' quantization spec verbatim, never
    /// re-quantize and never regroup.
    pub fn packed_quant_spec(&self) -> Option<(i32, i32)> {
        match &self.backing {
            Backing::Packed { linear, .. } => linear
                .quantized_params()
                .map(|(_, _, _, _, group, bits)| (bits, group)),
            Backing::Split(_) => None,
        }
    }

    /// The three projections, **unpacking first** so an adapter installed through them can never be
    /// stranded behind a stale packed matrix. Call [`Self::repack`] afterwards to re-attempt fusion;
    /// it will refuse while a live adapter is installed.
    pub fn parts_mut(
        &mut self,
    ) -> Result<(
        &mut AdaptableLinear,
        &mut AdaptableLinear,
        &mut AdaptableLinear,
    )> {
        self.unfuse()?;
        match &mut self.backing {
            Backing::Split(p) => Ok((&mut p.q, &mut p.k, &mut p.v)),
            Backing::Packed { .. } => unreachable!("unfuse leaves the split backing"),
        }
    }

    /// **The mutation half** of a host's `to_q`/`to_k`/`to_v` routing — one projection, with the
    /// same unpack-first contract as [`Self::parts_mut`]. A family's
    /// [`AdaptableHost::adaptable_mut`](crate::adapters::AdaptableHost::adaptable_mut) arm resolves
    /// through here, so installing an adapter leaves the block split and a later [`Self::repack`]
    /// refuses while that adapter is live.
    pub fn part_mut(&mut self, part: QkvPart) -> Result<&mut AdaptableLinear> {
        self.unfuse()?;
        match &mut self.backing {
            Backing::Split(p) => Ok(match part {
                QkvPart::Q => &mut p.q,
                QkvPart::K => &mut p.k,
                QkvPart::V => &mut p.v,
            }),
            Backing::Packed { .. } => unreachable!("unfuse leaves the split backing"),
        }
    }

    /// **The probe half** — one projection's [`LinearFacts`] **without unfusing**, the answer a
    /// family's
    /// [`AdaptableHost::adaptable_facts`](crate::adapters::AdaptableHost::adaptable_facts) arm hands
    /// back (SC-18319).
    ///
    /// While packed the facts are *derived*, never sliced: each part's `out` is its row range
    /// (`splits` is exactly the row concatenation the pack performed), `in` is the shared input
    /// width, and `is_quantized` / bias-presence are the packed matrix's own — because packing
    /// carries its parts' quantization spec verbatim and concatenates the dense bias on the same
    /// axis. `has_live_adapters` is `false` by construction: the pack predicate refuses a triple carrying
    /// any live adapter, so a *packed* backing is itself the proof that none is installed.
    ///
    /// While split it is the part's own snapshot, so the two representations answer identically —
    /// which is what makes a probe invisible to its caller.
    pub fn part_facts(&self, part: QkvPart) -> LinearFacts {
        let idx = part.index();
        match &self.backing {
            Backing::Split(p) => LinearFacts::of(match part {
                QkvPart::Q => &p.q,
                QkvPart::K => &p.k,
                QkvPart::V => &p.v,
            }),
            Backing::Packed { linear, splits } => {
                let shape = linear.base_shape();
                let (rows, inp) = (shape[0], shape[1]);
                let bounds = [0, splits[0], splits[1], rows];
                let out = bounds[idx + 1] - bounds[idx];
                LinearFacts {
                    base_shape: vec![out, inp],
                    is_quantized: linear.is_quantized(),
                    // A packed backing *is* the no-adapter proof — `try_pack` refuses a triple
                    // carrying any live adapter, and `parts_mut`/`part_mut` (the only route to an
                    // installer) leave the backing split. So neither count can be non-zero here.
                    has_live_adapters: false,
                    adapter_count: 0,
                    bias_shape: linear.bias().map(|_| vec![out]),
                }
            }
        }
    }

    /// Quantize all three bases and re-pack.
    pub fn quantize(&mut self, bits: i32, group_size: Option<i32>) -> Result<()> {
        {
            let (q, k, v) = self.parts_mut()?;
            q.quantize(bits, group_size)?;
            k.quantize(bits, group_size)?;
            v.quantize(bits, group_size)?;
        }
        self.repack();
        Ok(())
    }

    /// Cast all three dense bases and re-pack.
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        {
            let (q, k, v) = self.parts_mut()?;
            q.cast_weights(dtype)?;
            k.cast_weights(dtype)?;
            v.cast_weights(dtype)?;
        }
        self.repack();
        Ok(())
    }

    /// One `[.., q_out + k_out + v_out]` projection output, ready for [`QkvSource::Packed`].
    ///
    /// Packed: one matmul. Split: three forwards concatenated — **and split is what a caller gets
    /// without an explicit [`set_fused_qkv(true)`](set_fused_qkv)**, because fusion is opt-in. The
    /// two are **algebraically identical** — the packed matrix is the row-wise concatenation of the
    /// three bases and a matmul's output rows are independent — and they are bit-identical whenever
    /// MLX selects the same Metal GEMM kernel for both output widths.
    ///
    /// They are **not** bit-identical in general, and that is a property of the backend rather than
    /// of this type: MLX chooses its GEMM tile shape (and therefore its accumulation order) from
    /// `(M, N, K)`, the two arms differ in `N` by construction, and float addition is not
    /// associative. The self-hosted macOS 26.2 NAX build agrees to the bit across every geometry
    /// this tree ships; hosted CI (`macos-15-arm64`, `MACOSX_DEPLOYMENT_TARGET=15.0`) compiles a
    /// different kernel set and can land a few ULP apart. `qkv/tests.rs`'s `agree` helper is where
    /// that bound lives, and it is set far below what a *slicing* defect would produce, so a wrong
    /// row range still fails loudly.
    ///
    /// **That residual ULP divergence is exactly why the default is off**: it reaches the sampler,
    /// and a chaos sampler turns it into a different image at the same seed. A caller that opts in
    /// is accepting that trade for the geometries it has measured.
    ///
    /// The byte-level guarantee this feature actually rests on is one level down and is exact
    /// everywhere: the packed **tensors** are the concatenation of the parts' tensors, and
    /// [`Self::unfuse`] recovers each part byte-for-byte.
    pub fn forward_packed(&self, x: &Array) -> Result<Array> {
        match &self.backing {
            Backing::Packed { linear, .. } => {
                record_fusion(true);
                linear.forward(x)
            }
            Backing::Split(p) => {
                record_fusion(false);
                let q = p.q.forward(x)?;
                let k = p.k.forward(x)?;
                let v = p.v.forward(x)?;
                let axis = (q.shape().len() - 1) as i32;
                Ok(concatenate_axis(&[&q, &k, &v], axis)?)
            }
        }
    }

    /// The three projection outputs, split back out.
    pub fn forward(&self, x: &Array) -> Result<(Array, Array, Array)> {
        match &self.backing {
            Backing::Packed { linear, splits } => {
                record_fusion(true);
                let out = linear.forward(x)?;
                let axis = (out.shape().len() - 1) as i32;
                let mut parts = split_sections(&out, &splits[..], axis)?;
                let v = parts.swap_remove(2);
                let k = parts.swap_remove(1);
                let q = parts.swap_remove(0);
                Ok((q, k, v))
            }
            Backing::Split(p) => {
                record_fusion(false);
                Ok((p.q.forward(x)?, p.k.forward(x)?, p.v.forward(x)?))
            }
        }
    }

    /// Evaluate every retained array — the residency seam's `materialize_weights` for this
    /// composite. Exactly one representation is retained, so this evaluates exactly one copy.
    pub fn materialize_weights(&self) -> Result<()> {
        match &self.backing {
            Backing::Packed { linear, .. } => linear.materialize_weights(),
            Backing::Split(p) => {
                p.q.materialize_weights()?;
                p.k.materialize_weights()?;
                p.v.materialize_weights()
            }
        }
    }
}

/// Record the P4 activation receipt on whatever diagnostic scope is active (a no-op outside one).
///
/// Emitted from the **forward**, not from construction: a diagnostic scope is per *request*, and a
/// model is loaded outside it, so a load-time receipt would never appear in the request the P6
/// matrix is measuring. Per-forward recording is the same cadence `nn::rope_rotate` already uses for
/// its compile disposition, so the cost is precedented — one thread-local counter bump per attention
/// block per step.
fn record_fusion(applied: bool) {
    crate::diagnostics::record_toggle(
        crate::diagnostics::FUSED_ATTENTION_PRIMITIVES,
        if applied {
            crate::diagnostics::ToggleDisposition::Applied
        } else {
            crate::diagnostics::ToggleDisposition::Fallback
        },
    );
}

/// The safety predicate. Returns the packed matrix and its split points, or the specific reason it
/// was refused.
fn try_pack(
    q: &AdaptableLinear,
    k: &AdaptableLinear,
    v: &AdaptableLinear,
) -> std::result::Result<(AdaptableLinear, [i32; 2]), NoFusion> {
    let parts = [q, k, v];
    if parts
        .iter()
        .any(|l| l.adapters().iter().any(|a| !a.is_disabled()))
    {
        return Err(NoFusion::AdaptersInstalled);
    }

    let shapes: Vec<Vec<i32>> = parts.iter().map(|l| l.base_shape()).collect();
    if shapes.iter().any(|s| s.len() != 2) {
        return Err(NoFusion::InputWidthMismatch);
    }
    let inp = shapes[0][1];
    if shapes.iter().any(|s| s[1] != inp) {
        return Err(NoFusion::InputWidthMismatch);
    }
    let splits = [shapes[0][0], shapes[0][0] + shapes[1][0]];

    let quantized = parts.iter().filter(|l| l.is_quantized()).count();
    if quantized != 0 && quantized != parts.len() {
        return Err(NoFusion::MixedBases);
    }

    if quantized == 0 {
        let mut weights = Vec::with_capacity(3);
        let mut biases = Vec::with_capacity(3);
        for l in parts {
            let (w, b) = l.dense_weight().ok_or(NoFusion::MixedBases)?;
            weights.push(w.clone());
            if let Some(b) = b {
                biases.push(b.clone());
            }
        }
        if !biases.is_empty() && biases.len() != parts.len() {
            return Err(NoFusion::BiasMismatch);
        }
        let weight = cat_rows(&weights).map_err(|_| NoFusion::InputWidthMismatch)?;
        let bias = if biases.is_empty() {
            None
        } else {
            Some(cat_rows(&biases).map_err(|_| NoFusion::BiasMismatch)?)
        };
        return Ok((AdaptableLinear::dense(weight, bias), splits));
    }

    // Quantized: every part must agree on bits/group, and every out/in must be group-aligned.
    let mut packed_w = Vec::with_capacity(3);
    let mut packed_s = Vec::with_capacity(3);
    let mut packed_b = Vec::with_capacity(3);
    let mut dense_bias = Vec::with_capacity(3);
    let mut spec: Option<(i32, i32)> = None;
    for (l, shape) in parts.iter().zip(&shapes) {
        let (w, s, b, bias, group, bits) = l.quantized_params().ok_or(NoFusion::MixedBases)?;
        match spec {
            None => spec = Some((bits, group)),
            Some((eb, eg)) if eb == bits && eg == group => {}
            Some((eb, eg)) => {
                return Err(NoFusion::QuantMismatch {
                    bits: (eb, bits),
                    group: (eg, group),
                })
            }
        }
        if shape[1] % group != 0 {
            return Err(NoFusion::InFeaturesNotGroupAligned {
                inp: shape[1],
                group,
            });
        }
        if shape[0] % group != 0 {
            return Err(NoFusion::OutFeaturesNotGroupAligned {
                out: shape[0],
                group,
            });
        }
        packed_w.push(w.clone());
        packed_s.push(s.clone());
        packed_b.push(b.clone());
        if let Some(bias) = bias {
            dense_bias.push(bias.clone());
        }
    }
    if !dense_bias.is_empty() && dense_bias.len() != parts.len() {
        return Err(NoFusion::BiasMismatch);
    }
    let (bits, group) = spec.expect("a non-empty quantized triple sets the spec");
    let bias = if dense_bias.is_empty() {
        None
    } else {
        Some(cat_rows(&dense_bias).map_err(|_| NoFusion::BiasMismatch)?)
    };
    Ok((
        AdaptableLinear::from_quantized_parts(
            cat_rows(&packed_w).map_err(|_| NoFusion::InputWidthMismatch)?,
            cat_rows(&packed_s).map_err(|_| NoFusion::InputWidthMismatch)?,
            cat_rows(&packed_b).map_err(|_| NoFusion::InputWidthMismatch)?,
            bias,
            group,
            bits,
        ),
        splits,
    ))
}

/// Concatenate on the **output** axis (row-major axis 0) — the only axis that is safe for a
/// group-quantized weight, and the axis every packed tensor (`weight`, `scales`, `biases`, dense
/// `bias`) shares.
fn cat_rows(parts: &[Array]) -> Result<Array> {
    let refs: Vec<&Array> = parts.iter().collect();
    Ok(concatenate_axis(&refs, 0)?)
}

/// The exact inverse of [`try_pack`]'s row concatenation: slice the packed matrix back into three
/// projections at `splits`. Every packed tensor is row-major on the output axis, so this recovers
/// byte-identical parts for a dense *and* a quantized base — which is what lets the packed matrix be
/// the sole retained copy.
fn split_parts(
    linear: &AdaptableLinear,
    splits: [i32; 2],
) -> Result<(AdaptableLinear, AdaptableLinear, AdaptableLinear)> {
    let rows = |a: &Array| -> Result<Vec<Array>> { Ok(split_sections(a, &splits[..], 0)?) };
    if let Some((w, bias)) = linear.dense_weight() {
        let mut ws = rows(w)?;
        let mut bs = match bias {
            Some(b) => rows(b)?.into_iter().map(Some).collect(),
            None => vec![None, None, None],
        };
        let (w2, w1, w0) = (ws.swap_remove(2), ws.swap_remove(1), ws.swap_remove(0));
        let (b2, b1, b0) = (bs.swap_remove(2), bs.swap_remove(1), bs.swap_remove(0));
        return Ok((
            AdaptableLinear::dense(w0, b0),
            AdaptableLinear::dense(w1, b1),
            AdaptableLinear::dense(w2, b2),
        ));
    }
    let (w, s, b, bias, group, bits) = linear.quantized_params().ok_or_else(|| {
        Error::Msg("qkv::split_parts: base is neither dense nor quantized".into())
    })?;
    let mut ws = rows(w)?;
    let mut ss = rows(s)?;
    let mut bs = rows(b)?;
    let mut biases: Vec<Option<Array>> = match bias {
        Some(x) => rows(x)?.into_iter().map(Some).collect(),
        None => vec![None, None, None],
    };
    let mut take = |i: usize| {
        AdaptableLinear::from_quantized_parts(
            ws.swap_remove(i),
            ss.swap_remove(i),
            bs.swap_remove(i),
            biases.swap_remove(i),
            group,
            bits,
        )
    };
    let v = take(2);
    let k = take(1);
    let q = take(0);
    Ok((q, k, v))
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Structural exemptions
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// One attention path that is **not** expressible through [`prepare`], with the specific deviation
/// that makes it so and where to read it.
#[derive(Clone, Copy, Debug)]
pub struct Exemption {
    /// The family crate (or the sub-module inside it) that owns the path.
    pub path: &'static str,
    /// The deviation, stated concretely enough to re-check.
    pub deviation: &'static str,
}

/// **The documented structural exemptions** (SC-18319's survey outcome). The story explicitly
/// permits leaving a family on its native path *provided the deviation is documented* — this table
/// is that documentation, compiled in so it cannot drift into a stale markdown file, and asserted
/// non-empty and well-formed by this module's tests.
///
/// None of these is a scope reduction: each is a path whose arithmetic differs from the shared
/// pipeline in a way a parameter cannot express without the "parameter" becoming "run this family's
/// code".
///
/// **The bar for landing on this list is "a parameter cannot express it", not "a parameter does not
/// yet express it".** Two families the survey initially flagged are *not* here, because the honest
/// answer was to grow the primitive instead:
///
/// * **Anima** needed knob 3 (RoPE optional — the same code path runs self-attention with a rotation
///   and cross-attention without one) and knob 6 (its conditioner gives q and k genuinely different
///   position tables). Both are parameters; anima is migrated.
/// * **LTX** needed knob 1's `FullDimPreSplit` arm, knob 6 (`k_pe`), and two things the primitive
///   did not have: [`RopeStyle::HalvesPaired`] (the GPT-NeoX rotate-halves form over a *half*-width
///   table through one `rope_rotate`, rather than the tiled full-width expression) and
///   a **pre-broadcast** rank-4 table layout (its `[B, H, T, head_dim/2]` tables are per batch **and** per
///   head, which no rank-≤3 layout can carry). Both were added; LTX is migrated.
pub const EXEMPTIONS: &[Exemption] = &[
    Exemption {
        path: "mlx-gen-qwen-image/src/text_encoder/vision/attention.rs",
        deviation: "`[seq, embed]` with no batch axis and block-diagonal windowed SDPA driven by \
                    `cu_seqlens`; not BHSD, so it cannot reach `sdpa_budgeted_bhsd` at all. Its \
                    `rotate_half` rotation is explicitly documented as non-interchangeable with the \
                    MMDiT interleaved one. No adapter and no quantization seam. Its `qkv` is \
                    already fused.",
    },
    Exemption {
        path: "mlx-gen-flux2/src/text_encoder/.. (Pixtral vision tower)",
        deviation: "Same non-BHSD windowed shape as the Qwen tower, and no adapter seam (raw \
                    `Array` weights).",
    },
    Exemption {
        path: "mlx-gen-wan/src/transformer.rs",
        deviation: "QK-norm is full-dim on the FLAT `[B, S, H·D]` projection pre-reshape, combined \
                    with a 3-axis asymmetric `[22|21|21]` RoPE (`rope.rs`) and an f32-RoPE-over-bf16 \
                    cast dance. The placement knob covers the norm; the asymmetric per-axis table \
                    split is a Wan-specific table construction, not a rotation style.",
    },
    Exemption {
        path: "mlx-gen-wan/src/vace.rs",
        deviation: "A self-contained diffusers-layout duplicate of the Wan block that is \
                    dtype-generic over f32/bf16; it inherits Wan's exemption and adds a second \
                    weight layout.",
    },
    Exemption {
        path: "mlx-gen-krea/src/block.rs",
        deviation: "Four simultaneous deviations: a `+1`-centred `RmsScale` (not RMSNorm), a fifth \
                    `to_gate` projection with a sigmoid applied BETWEEN SDPA and `to_out`, optional \
                    RoPE, and the GQA repeat placed between RoPE and the transpose rather than \
                    after both.",
    },
    Exemption {
        path: "mlx-gen-flux/src/transformer.rs (FLUX.1 IP-Adapter / XLabs seam)",
        deviation: "The `double_block_ip` seam taps the RMS-normed, PRE-RoPE query and feeds a \
                    second SDPA from it. A primitive that fuses QK-norm and RoPE cannot serve this \
                    without exposing that intermediate as an output, which would defeat the fusion.",
    },
    Exemption {
        path: "mlx-gen-sdxl/src/unet/transformer.rs (and mlx-gen-kolors, which only loads it)",
        deviation: "No RoPE, no QK-norm, cross-attention K/V from a different sequence, and a \
                    second IP-branch SDPA sharing the query. It exercises none of the primitive's \
                    parameters, so routing it through would add a call with every knob off.",
    },
    Exemption {
        path: "mlx-gen-sana/src/transformer.rs",
        deviation: "`attn1` is ReLU **linear** attention — no softmax and no score tensor, so it is \
                    not SDPA at all; `attn2` hand-rolls matmul→mask→softmax→matmul for a chunked \
                    mask. No RoPE anywhere. This exempts the PROLOGUE only: `attn1`'s q/k/v are \
                    ordinary `AdaptableLinear`s reading one stream and DO go through \
                    `FusedQkvProjection` (SC-18319 P4). `attn2` is cross-attention and must not.",
    },
    Exemption {
        path: "mlx-gen-seedvr2/src/dit.rs",
        deviation: "Partial RoPE (the knob exists here) combined with window-partitioned attention, \
                    a cached window PERMUTATION applied to the packed QKV between the projection \
                    and the QK-norm, and text outputs summed across windows. The mid-region tap is \
                    the blocker: fusing projection→QK-norm would hide the raw packed QKV the \
                    permutation needs.",
    },
    Exemption {
        path: "mlx-gen-pulid/src/.. (EVA-CLIP tower)",
        deviation: "No QK-norm at all (a post-attention LayerNorm instead), RoPE on patch tokens \
                    only with the CLS token passed through, and an asymmetric q/v-only bias.",
    },
    Exemption {
        path: "mlx-gen-svd/src/..",
        deviation: "No QK-norm, no RoPE, no adapters, no quantization — the primitive degenerates \
                    to a no-op wrapper, so using it would be pure indirection.",
    },
];

#[cfg(test)]
mod tests;
