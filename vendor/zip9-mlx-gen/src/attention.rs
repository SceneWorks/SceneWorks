//! **Bounded attention** (rung 3 of the SC-15448 ladder): the MLX **kernel**.
//!
//! The **planner** — [`AttentionBudget`], [`CONSTRAINED_ATTN_SCORES_BUDGET`] and the query-block
//! arithmetic — lives in [`gen_core::attention_budget`] and is re-exported below, so this module keeps
//! its public shape while the chunk boundaries come from the same place every backend's do (SC-15793).
//! That is the shared-ladder split rungs 2 (`gen_core::tiling`) and 4 (`gen_core::block_window`)
//! already use.
//!
//! What stays here is genuinely MLX's: [`sdpa_budgeted_bhsd`], [`sdpa_head_budgeted_bhsd`], the
//! `slice_axis` helper, and the per-chunk `eval`. The kernels must NOT merge — measured like-for-like on the denoise phase, this
//! rung is worth **−32% on candle and −1.7% here**, roughly an order of magnitude apart, precisely
//! because the kernels differ; see the attribution table below and the [`gen_core::attention_budget`]
//! module doc. (MLX's often-quoted −5.0% is the whole-request figure, not the denoise phase — the two
//! are not interchangeable when comparing backends.)
//!
//! There are **two** levers, and [`AttentionChunkAxis`] is where a caller picks between them. The
//! head axis ([`sdpa_head_budgeted_bhsd`], sc-18661) splits complete heads and is the stronger one —
//! it leaves the query GEMM's `M` untouched and reconstructs the unbounded output exactly, provided
//! the budget admits one whole head. Read [`AttentionChunkAxis`] before relying on that: below that
//! budget it falls back to query rows and inherits their weaker contract under the same name.
//!
//! The original lever is query-row chunking: every query row's softmax is over **all** keys and is independent
//! of the other rows, so running the attention over blocks of query rows leaves each query's complete
//! key/value domain intact while bounding the per-call attention scratch to `chunk_rows × Sk` scores
//! instead of `Sq × Sk`. Precision, schedule, seed, conditioning, and the output contract are
//! untouched: the same [`scaled_dot_product_attention`] is called with the same `scale`, the same
//! dtypes, and the same k/v.
//!
//! ## What this bounds on MLX — measured (SC-15615)
//!
//! candle's CPU/CUDA `attention_basic` **materializes** the whole `[B,H,Sq,Sk]` scores tensor, so
//! bounding it there is a large unconditional win — that is what bought the Candle Z-Image 8 GB fit
//! (staged Q4 denoise 8.394 → 5.709 GB, SC-15256).
//!
//! **MLX never materializes it.** `fast::scaled_dot_product_attention` dispatches to a fused Metal
//! kernel that streams the scores. Measured in isolation on Apple M5 Max at the production Z-Image
//! DiT shape (B=1, H=30, D=128, f32), the unbounded peak is exactly `4 · B·H·Sq·D · sizeof(dtype)` —
//! q, k, v and the output, nothing else (241.9 MiB at Sq=4128, 961.9 MiB at Sq=16416; a materialized
//! score tensor at Sq=16416 would alone be 30 GiB). `at_the_production_shape_sdpa_streams_the_scores_and_chunking_is_exact`
//! pins that. So there is **no score tensor for this helper to bound**, and in isolation — with q/k/v
//! already materialized and pinned alive — chunking only *adds* transients (+50% peak).
//!
//! Inside a real DiT forward the result inverts, for a different reason. There q/k/v are intermediates
//! in a deep lazy graph, and `AttentionBudget::eval_per_chunk` cuts that graph at every chunk
//! boundary, letting MLX free upstream transients earlier. Measured on the real Z-Image-turbo q4 DiT
//! (1024², 4 steps, staged residency):
//!
//! | Arm | Denoise peak | vs unbounded |
//! |---|---:|---:|
//! | unbounded | 4.7746 GiB | — |
//! | 64 Mi chunk, lazy (no per-chunk eval) | 4.7708 GiB | −0.08% (noise) |
//! | 64 Mi chunk + per-chunk eval | 4.6944 GiB | **−1.7%** |
//! | never-chunks budget + eval flag (control) | 4.7747 GiB | +0.00% |
//!
//! The control arm is the important one: setting `eval_per_chunk` on a budget that never chunks is
//! inert, so the saving is genuinely produced by chunking — but by the **graph cut** it forces, not by
//! bounding a score matrix. It is therefore real, quality-preserving (`sum|out|` identical to the last
//! digit across all four arms) and *small* — nothing like CUDA's 32%. End to end through the staged
//! generate it measured 4.898 → 4.653 GiB (−5.0%) with a bit-identical image.
//!
//! Whether a given family/geometry benefits stays a measurement, never an assumption:
//! [`AttentionBudget::UNBOUNDED`] is the default and every caller keeps the single-call path until a
//! measured rung selects otherwise. A family whose attention *is* materialized (a hand-rolled
//! `matmul → softmax → matmul`, or a shape the fused kernel rejects) gets the full CUDA-style saving
//! from this same seam instead of forking its own copy.
//!
//! ## Laziness
//!
//! MLX is lazy, and per the table above the laziness is where the whole MLX-side saving lives.
//! `AttentionBudget::eval_per_chunk` forces each chunk's output to be materialized before the next
//! chunk is built — the same lever `mlx_gen_flux2::chunk::MemoryConfig::eval_per_block` uses one level
//! up. It is **bit-exact** (it only forces materialization) and costs one synchronization per chunk.
//! [`AttentionBudget::CONSTRAINED`] therefore sets it; a lazily-chunked budget measured as no better
//! than unbounded.
//!
//! `eval` is not valid inside an autograd trace, so `eval_per_chunk` must stay
//! `false` on training paths. The trainers never select a bounded rung (the default is
//! [`AttentionBudget::UNBOUNDED`], whose fast path is a single unchunked call), so this is structural
//! rather than a runtime guard.

use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use crate::{Error, Result};

/// The rung-3 **planner**, re-exported from its shared home so MLX provider code, the SC-15615
/// evidence and `mlx-gen-z-image`'s published `attention_chunk_sizes` all keep their current import
/// paths. The arithmetic lives in [`gen_core::attention_budget`] (SC-15793) alongside the rung-2
/// (`gen_core::tiling`) and rung-4 (`gen_core::block_window`) planners; only the kernel below is MLX's.
pub use gen_core::attention_budget::{
    AttentionBudget, AttentionPlan, CONSTRAINED_ATTN_SCORES_BUDGET,
};

/// Bounded scaled-dot-product attention over the 4-D DiT shape `q, k, v: [B, H, Sq, D]` (the key
/// length `Sk` may differ from `Sq`), returning `[B, H, Sq, D]` — a drop-in around an existing
/// [`scaled_dot_product_attention`] call.
///
/// When `budget` leaves the whole call within bounds (always, for
/// [`AttentionBudget::UNBOUNDED`]) this is exactly the caller's original single call. Otherwise the
/// query rows are split into blocks, each block attending over the **complete** k/v, and the block
/// outputs are concatenated back along the query axis.
///
/// `mask` is passed through to each chunk. A `[.., 1, Sk]` mask (or `None`) broadcasts identically
/// onto every chunk; a per-query `[.., Sq, Sk]` mask is narrowed to the chunk's own rows. Z-Image's
/// DiT is intentionally maskless, so the mask path exists for the other families that adopt this.
pub fn sdpa_budgeted_bhsd(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: Option<&Array>,
    plan: AttentionPlan<'_>,
) -> Result<Array> {
    let qs = q.shape();
    if qs.len() != 4 {
        return Err(Error::Msg(format!(
            "sdpa_budgeted_bhsd expects q as [B, H, Sq, D], got {qs:?}"
        )));
    }
    // `k`/`v` are indexed on the same axes as `q`, so they must have the same rank; a 3-D `k` would
    // otherwise silently read the head axis as the key length and produce a wrong (not failed) chunk
    // plan.
    for (name, a) in [("k", k), ("v", v)] {
        if a.shape().len() != 4 {
            return Err(Error::Msg(format!(
                "sdpa_budgeted_bhsd expects {name} as [B, H, Sk, D], got {:?}",
                a.shape()
            )));
        }
    }
    if let Some(m) = mask {
        // The narrowing below indexes `len() - 2`, so a rank-1 mask would underflow.
        if m.shape().len() < 2 {
            return Err(Error::Msg(format!(
                "sdpa_budgeted_bhsd expects a mask with a query and a key axis, got {:?}",
                m.shape()
            )));
        }
    }
    let (b, h, sq) = (qs[0], qs[1], qs[2]);
    let sk = k.shape()[2];

    let block = plan.budget.query_block(b, h, sq, sk);
    if block >= sq {
        record_chunk_count(1);
        return Ok(scaled_dot_product_attention(
            q,
            k,
            v,
            scale,
            mask.map(ScaledDotProductAttentionMask::Array),
            None,
        )?);
    }

    let mut outs: Vec<Array> = Vec::with_capacity(sq.div_euclid(block) as usize + 1);
    let mut start = 0;
    while start < sq {
        // Between-chunk cancellation: a bounded call is the only place inside a DiT forward with a
        // boundary to check at. Checked BEFORE the chunk so a cancel that arrived during the previous
        // chunk stops the next launch; the partial `outs` drop here and the caller's request scope
        // does the synchronize-and-release.
        if plan.is_cancelled() {
            return Err(Error::Canceled);
        }
        let len = block.min(sq - start);
        let q_chunk = slice_axis(q, 2, start, len)?;
        // Narrow only a per-query mask; a broadcast `[.., 1, Sk]` mask is shared by every chunk.
        let chunk_mask = match mask {
            Some(m) if m.shape()[m.shape().len() - 2] == sq => {
                Some(slice_axis(m, (m.shape().len() - 2) as i32, start, len)?)
            }
            other => other.cloned(),
        };
        let out = scaled_dot_product_attention(
            &q_chunk,
            k,
            v,
            scale,
            chunk_mask
                .as_ref()
                .map(ScaledDotProductAttentionMask::Array),
            None,
        )?;
        if plan.budget.eval_per_chunk() {
            mlx_rs::transforms::eval([&out])?;
        }
        outs.push(out);
        start += len;
    }
    record_chunk_count(outs.len());
    let refs: Vec<&Array> = outs.iter().collect();
    Ok(concatenate_axis(&refs, 2)?)
}

/// Which axis a bounded call splits, when the budget forces a split at all.
///
/// The two axes **share a score budget but not a numerical contract**, which is
/// [`gen_core::attention_budget`]'s own wording and the reason this is an explicit choice rather
/// than an internal heuristic:
///
/// | axis | what a chunk is | the query GEMM's `M` | agreement with the unbounded call |
/// |---|---|---|---|
/// | [`QueryRows`](Self::QueryRows) | `block` query rows, all `H` heads, complete k/v | **changes** | numerical |
/// | [`Heads`](Self::Heads) | `heads_per_chunk` complete heads, all `Sq` rows | unchanged | **exact**, while the budget admits one whole head — see below |
///
/// # The head axis is bit-exact — but only while it does not fall back (sc-18661)
///
/// `mlx_gen_sdxl::memory_strategy::ATTENTION_SUPPORT` predicted that "a head-axis implementation
/// would be bit-exact by construction". Measured, that holds: heads are fully independent, there is
/// no accumulator and no running max to split, `M` is untouched, and MLX dispatches the same Metal
/// specialization. Exactly `0` relative max-abs at every shape sc-18661 measured — the production
/// MiniMax-H3 DiT (bf16, `H = 56`, `D = 128`, `Sq` 7 586 and 20 022), a synthetic `f32` `H = 5`
/// sweep across four heads-per-chunk boundaries, and the committed H3 DiT fixture (`f32`, `H = 4`,
/// `D = 24`).
///
/// **The trap is the budget.** [`Heads`](Self::Heads) falls back to query rows *inside* a head chunk
/// when one head still exceeds the budget — the order the shared planner prescribes — and such a
/// call silently inherits the query axis's weaker numerical contract while still reporting itself as
/// the head axis. The condition is exact and worth stating: the head axis keeps bit identity iff the
/// budget admits one whole head, `max_score_elements >= B · Sq · Sk`. Below that it is the query axis
/// wearing a different name. Measured on the H3 fixture at `Sq = 29`: `0` at a one-head budget,
/// `5.9e-4` at half of one — the *same* number the query-row arm reports there.
///
/// Callers that need the distinction should read it off the plan rather than assume it:
/// `AttentionBudget::head_chunks(..).chunks_heads()` with a query block equal to `Sq` is the
/// bit-exact configuration, and `mlx_gen_minimax_h3::dit::layers::planned_attention_chunks` is a
/// worked example of surfacing both counts.
///
/// Until sc-18661 only the query-row axis existed on MLX. MiniMax-H3 is the family that forced the
/// point — 56 heads against Z-Image's 30, so the head axis alone buys a 56x reduction in the score
/// domain.
///
/// Neither axis is a default: [`BoundedAttention::UNBOUNDED`] is, and a family selects an axis only
/// on its own measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionChunkAxis {
    /// Split query rows. Every chunk sees all `H` heads and the complete k/v.
    QueryRows,
    /// Split complete heads first, and fall back to query rows **inside** a head chunk only when one
    /// head still exceeds the budget — the order [`AttentionBudget::head_chunks`] prescribes.
    Heads,
}

/// A [`AttentionPlan`] plus the axis its chunks are cut on — everything a bounded MLX attention call
/// needs that is not a tensor.
///
/// `Copy`, so it threads through a whole DiT forward chain as cheaply as the budget alone. Existing
/// callers that predate the head axis keep [`sdpa_budgeted_bhsd`], which is exactly this value at
/// [`AttentionChunkAxis::QueryRows`].
#[derive(Clone, Copy, Debug)]
pub struct BoundedAttention<'a> {
    /// The budget and the request's cancel flag.
    pub plan: AttentionPlan<'a>,
    /// The axis chunks are cut on when the budget forces a split.
    pub axis: AttentionChunkAxis,
}

impl<'a> BoundedAttention<'a> {
    /// The unbounded, uncancellable plan — byte-identical to the pre-rung forward on either axis.
    pub const UNBOUNDED: Self = Self {
        plan: AttentionPlan::UNBOUNDED,
        axis: AttentionChunkAxis::QueryRows,
    };

    /// A bounded plan on `axis`.
    pub const fn new(plan: AttentionPlan<'a>, axis: AttentionChunkAxis) -> Self {
        Self { plan, axis }
    }

    /// A bounded plan at `budget`, cutting complete heads.
    pub const fn heads(budget: AttentionBudget) -> Self {
        Self::new(AttentionPlan::budgeted(budget), AttentionChunkAxis::Heads)
    }

    /// A bounded plan at `budget`, cutting query rows.
    pub const fn query_rows(budget: AttentionBudget) -> Self {
        Self::new(
            AttentionPlan::budgeted(budget),
            AttentionChunkAxis::QueryRows,
        )
    }

    /// The same plan with `cancel` attached, checked between chunks.
    pub fn with_cancel(mut self, cancel: &'a crate::CancelFlag) -> Self {
        self.plan = self.plan.with_cancel(cancel);
        self
    }
}

/// Bounded SDPA over `q, k, v: [B, H, Sq, D]` on the caller's chosen [`AttentionChunkAxis`].
///
/// [`AttentionChunkAxis::QueryRows`] is exactly [`sdpa_budgeted_bhsd`]. [`AttentionChunkAxis::Heads`]
/// consumes [`AttentionBudget::head_chunks`] first and delegates each head chunk to
/// [`sdpa_budgeted_bhsd`], which re-plans the query axis **inside** the chunk — with `H` already
/// narrowed the query planner usually reports the whole `Sq`, i.e. one un-chunked fused call per head
/// group, which is what makes this axis bit-exact.
pub fn sdpa_bounded_bhsd(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: Option<&Array>,
    bounded: BoundedAttention<'_>,
) -> Result<Array> {
    match bounded.axis {
        AttentionChunkAxis::QueryRows => sdpa_budgeted_bhsd(q, k, v, scale, mask, bounded.plan),
        AttentionChunkAxis::Heads => sdpa_head_budgeted_bhsd(q, k, v, scale, mask, bounded.plan),
    }
}

/// Bounded SDPA that splits **complete heads** — the bit-exact axis (sc-18661).
///
/// Each chunk is a complete fused SDPA over `heads_per_chunk` whole heads, every one of them seeing
/// its full `Sq` query rows and the complete k/v for those heads. The query GEMM's `M` dimension is
/// therefore identical to the unbounded call's, so MLX dispatches the same Metal specialization and
/// the reconstruction is bit-exact rather than tolerance-equivalent — the property the query-row axis
/// cannot offer (`mlx_gen_sdxl::memory_strategy::ATTENTION_SUPPORT` predicted this; sc-18661 measured
/// it).
///
/// **That holds only while the budget admits one whole head** (`max_score_elements >= B · Sq · Sk`).
/// Below it a single head still exceeds the budget, this kernel falls back to query rows *inside* the
/// head chunk — the order the shared planner prescribes — and the call inherits the query axis's
/// weaker numerical contract under this function's name. See [`AttentionChunkAxis`].
///
/// `mask` is narrowed on the head axis when it carries one head entry per head; a broadcast `[.., 1,
/// .., ..]` mask (or `None`) is shared by every chunk unchanged. A per-query mask is narrowed by the
/// inner [`sdpa_budgeted_bhsd`] call if the query axis also has to split.
///
/// Cancellation follows the rung's shared contract: checked **between** chunks, before the next
/// launch, and never on the unbounded fast path.
pub fn sdpa_head_budgeted_bhsd(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: Option<&Array>,
    plan: AttentionPlan<'_>,
) -> Result<Array> {
    let qs = q.shape();
    if qs.len() != 4 {
        return Err(Error::Msg(format!(
            "sdpa_head_budgeted_bhsd expects q as [B, H, Sq, D], got {qs:?}"
        )));
    }
    for (name, a) in [("k", k), ("v", v)] {
        if a.shape().len() != 4 {
            return Err(Error::Msg(format!(
                "sdpa_head_budgeted_bhsd expects {name} as [B, H, Sk, D], got {:?}",
                a.shape()
            )));
        }
    }
    let (b, h, sq) = (qs[0], qs[1], qs[2]);
    let sk = k.shape()[2];
    // `k`/`v` carry the same head axis as `q`; a k/v with a different `H` would make the head slices
    // below pair a query head with the wrong key head — wrong output, not a failure.
    for (name, a) in [("k", k), ("v", v)] {
        if a.shape()[1] != h {
            return Err(Error::Msg(format!(
                "sdpa_head_budgeted_bhsd expects {name} to carry q's {h} heads, got {:?}",
                a.shape()
            )));
        }
    }

    let heads = plan.budget.head_chunks(
        b.max(0) as u64,
        h.max(0) as u64,
        sq.max(0) as u64,
        sk.max(0) as u64,
    );
    if !heads.chunks_heads() {
        // In budget, unbounded, or a single head: there is nothing for this axis to split, and the
        // query-row kernel owns the remaining decision (which for an in-budget call is the
        // byte-identical single fused call).
        record_head_chunk_count(1);
        return sdpa_budgeted_bhsd(q, k, v, scale, mask, plan);
    }
    let per_chunk = (heads.heads_per_chunk().max(1) as i32).min(h);

    // A mask with a real head axis must be narrowed with the heads; a broadcast one is shared.
    let mask_head_axis = mask.and_then(|m| {
        let ms = m.shape();
        // The head axis is the second-from-last-two, i.e. `len() - 3`; ranks below 3 have none.
        (ms.len() >= 3 && ms[ms.len() - 3] == h).then_some((ms.len() - 3) as i32)
    });

    let mut outs: Vec<Array> = Vec::with_capacity(h.div_euclid(per_chunk) as usize + 1);
    let mut start = 0;
    while start < h {
        if plan.is_cancelled() {
            return Err(Error::Canceled);
        }
        let len = per_chunk.min(h - start);
        let q_chunk = slice_axis(q, 1, start, len)?;
        let k_chunk = slice_axis(k, 1, start, len)?;
        let v_chunk = slice_axis(v, 1, start, len)?;
        let chunk_mask = match (mask, mask_head_axis) {
            (Some(m), Some(axis)) => Some(slice_axis(m, axis, start, len)?),
            (other, _) => other.cloned(),
        };
        // The inner call re-plans the query axis against the NARROWED head count, so it only chunks
        // query rows when a single head still exceeds the budget — the order the shared planner
        // prescribes, and the reason this axis stays bit-exact whenever it can.
        let out = sdpa_budgeted_bhsd(
            &q_chunk,
            &k_chunk,
            &v_chunk,
            scale,
            chunk_mask.as_ref(),
            plan,
        )?;
        if plan.budget.eval_per_chunk() {
            mlx_rs::transforms::eval([&out])?;
        }
        outs.push(out);
        start += len;
    }
    record_head_chunk_count(outs.len());
    let refs: Vec<&Array> = outs.iter().collect();
    Ok(concatenate_axis(&refs, 1)?)
}

/// Test-only observation of how many chunks the last [`sdpa_budgeted_bhsd`] call actually ran.
///
/// Without this, every equivalence test in this module — and every one in `mlx-gen-z-image` — passes
/// with the chunking deleted, because they all assert *chunked == unbounded*, which is trivially true
/// when the chunking never engages. This lets a test assert that the lever it is exercising is
/// actually pulled. Compiled out entirely in release; `RUST_TEST_THREADS=1` is forced repo-wide, so a
/// process-global counter is safe.
#[cfg(test)]
mod chunk_probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static LAST_CHUNK_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub(super) static LAST_HEAD_CHUNK_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Chunks run by the most recent [`super::sdpa_budgeted_bhsd`] call (`1` = the unchunked path).
    ///
    /// Under [`super::sdpa_head_budgeted_bhsd`] this reports the **last head chunk's** query-axis
    /// decision, which is the one that says whether the head axis alone was enough to fit the budget.
    pub fn last_chunk_count() -> usize {
        LAST_CHUNK_COUNT.load(Ordering::Relaxed)
    }

    /// Head chunks run by the most recent [`super::sdpa_head_budgeted_bhsd`] call (`1` = the head
    /// axis did not split).
    pub fn last_head_chunk_count() -> usize {
        LAST_HEAD_CHUNK_COUNT.load(Ordering::Relaxed)
    }

    /// Reset before a call so a stale value cannot satisfy an assertion.
    pub fn reset() {
        LAST_CHUNK_COUNT.store(0, Ordering::Relaxed);
        LAST_HEAD_CHUNK_COUNT.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
pub use chunk_probe::{last_chunk_count, last_head_chunk_count, reset as reset_chunk_count};

#[cfg(test)]
fn record_chunk_count(n: usize) {
    chunk_probe::LAST_CHUNK_COUNT.store(n, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(test))]
#[inline(always)]
fn record_chunk_count(_n: usize) {}

#[cfg(test)]
fn record_head_chunk_count(n: usize) {
    chunk_probe::LAST_HEAD_CHUNK_COUNT.store(n, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(test))]
#[inline(always)]
fn record_head_chunk_count(_n: usize) {}

/// Contiguous `[.., start..start+len, ..]` slice along `axis`, via boundary splits (no host index
/// vector and no gather, unlike `take_axis`).
///
/// `pub` because it is genuinely shared: it is numerics-neutral plumbing every query-row chunker
/// needs, and a provider whose attention kernel cannot be this one (SANA's `attn2` materializes its
/// scores, so routing it through the fused kernel above would change resident-path numerics) still
/// needs to slice its query axis the same way. Forking THAT would be a fork with no arithmetic
/// reason behind it.
pub fn slice_axis(a: &Array, axis: i32, start: i32, len: i32) -> Result<Array> {
    Ok(a.split_axis(&[start, start + len], axis)?.swap_remove(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CancelFlag;
    use mlx_rs::Dtype;

    fn flat(a: &Array) -> Vec<f32> {
        a.reshape(&[-1])
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    fn arange(shape: &[i32], scale: f32, offset: f32) -> Array {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * scale + offset).sin())
            .collect();
        Array::from_slice(&data, shape)
    }

    /// Peak-relative max delta of the chunked forward against the unbounded one.
    ///
    /// Asserts on the way through that the chunked arm **actually chunked** (`expect_chunks`). Without
    /// that, every equivalence assertion in this module is a false green: `chunked == unbounded` is
    /// trivially true when the chunking never engages, so deleting the lever would leave the whole
    /// suite passing.
    fn chunked_delta(
        q: &Array,
        k: &Array,
        v: &Array,
        scale: f32,
        mask: Option<&Array>,
        budget: AttentionBudget,
        expect_chunks: usize,
    ) -> f32 {
        reset_chunk_count();
        let full = sdpa_budgeted_bhsd(q, k, v, scale, mask, AttentionPlan::UNBOUNDED).unwrap();
        assert_eq!(
            last_chunk_count(),
            1,
            "the unbounded arm must be a single un-chunked call"
        );
        reset_chunk_count();
        let chunked =
            sdpa_budgeted_bhsd(q, k, v, scale, mask, AttentionPlan::budgeted(budget)).unwrap();
        assert_eq!(
            last_chunk_count(),
            expect_chunks,
            "the budgeted arm did not chunk as expected — the equivalence below would be vacuous"
        );
        assert_eq!(chunked.shape(), full.shape());
        let (a, c) = (flat(&full), flat(&chunked));
        let peak = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
        a.iter()
            .zip(&c)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
            / peak
    }

    /// The documented numerical-equivalence bound (SC-15615; the CUDA twin is `candle_gen_z_image`'s
    /// `attention_query_chunking_matches_the_unbounded_forward`).
    ///
    /// Query-row chunking does not change the math — each query still attends over the complete
    /// k/v — but it *does* change which Metal kernel specialization MLX picks for the
    /// `[.., block, Sk]` matmul, and MLX's Metal matmul runs in reduced precision (the sc-2338
    /// parity class). So the agreement is **numerical, not bit-exact in general**: measured on
    /// Apple M5 Max it is exactly `0` for some block sizes and ~1e-3 peak-relative for others, with
    /// no monotonic relationship to block size. At the *production* shape and budget it is exactly
    /// zero — see `at_the_production_shape_sdpa_streams_the_scores_and_chunking_is_exact`.
    const PRODUCTION_TOLERANCE: f32 = 2e-3;

    #[test]
    fn unbounded_never_chunks() {
        let budget = AttentionBudget::UNBOUNDED;
        assert!(budget.is_unbounded());
        assert_eq!(budget.query_block(1, 30, 4128, 4128), 4128);
        assert_eq!(budget.max_score_elements(), u64::MAX);
        assert_eq!(AttentionBudget::default(), AttentionBudget::UNBOUNDED);
    }

    /// **The kernel-to-planner binding.** `mlx_gen` re-exports gen-core's `AttentionBudget`, so
    /// re-asserting the planner's arithmetic here would be a tautology — it would pass with the MLX
    /// kernel deleted. What is worth pinning, and what this pins, is that
    /// [`sdpa_budgeted_bhsd`] *actually sizes its chunks from the shared planner*: for each shared
    /// conformance case, the kernel must run exactly `ceil(Sq / shared_boundary)` chunks.
    ///
    /// A kernel that reintroduced local chunk arithmetic — the failure this story exists to prevent —
    /// fails here even though the planner itself is untouched.
    ///
    /// The shared table's rows are production geometries (Sq up to 65536) and cannot be launched in a
    /// unit test — an earlier version of this test iterated them and silently exercised only the one
    /// row small enough to run, which happened to be a *non-chunking* row. So this sweeps runnable
    /// shapes across the **whole range of block sizes** — a single row, ragged multi-row splits, and
    /// the un-chunked call — and requires the kernel's chunk count to equal the planner's
    /// `ceil(Sq / block)` at every one.
    ///
    /// The production 4128-row geometry is bound at real size by
    /// [`at_the_production_shape_sdpa_streams_the_scores_and_chunking_is_exact`] (541 rows ⇒ 8 chunks);
    /// the planner's agreement with the shared table is pinned by
    /// [`the_production_geometry_plans_the_shared_boundary`].
    #[test]
    fn the_kernel_chunks_exactly_as_the_shared_planner_declares() {
        // Sq = 37 is prime, so every block size below it leaves a ragged tail — a kernel that rounded
        // the tail differently from `ceil` shows up here.
        let (b, h, sq, d) = (1i32, 3i32, 37i32, 8i32);
        let q = arange(&[b, h, sq, d], 0.017, 0.3);
        let k = arange(&[b, h, sq, d], 0.013, -0.7);
        let v = arange(&[b, h, sq, d], 0.011, 1.1);
        let scale = (d as f32).powf(-0.5);
        let rows_per_query = (b * h * sq) as u64;

        let mut chunked_arms = 0;
        for rows in [1i32, 2, 5, 18, 36, 37] {
            let budget = AttentionBudget::from_score_elements(rows_per_query * rows as u64, false);
            let block = budget.query_block(b, h, sq, sq);
            assert_eq!(
                block, rows,
                "the planner did not produce the {rows}-row block this arm targets"
            );
            reset_chunk_count();
            sdpa_budgeted_bhsd(&q, &k, &v, scale, None, AttentionPlan::budgeted(budget)).unwrap();
            let expected = (sq as usize).div_ceil(block as usize);
            assert_eq!(
                last_chunk_count(),
                expected,
                "at a {block}-row block the kernel ran {} chunks, the shared planner implies \
                 {expected}",
                last_chunk_count()
            );
            if expected > 1 {
                chunked_arms += 1;
            }
        }
        assert!(
            chunked_arms >= 5,
            "only {chunked_arms} arms actually chunked — this test would pass with chunking gone"
        );
    }

    /// The production geometry the MLX kernel ships at must be the boundary the shared table declares.
    /// Pinned against the table by value (not by a name lookup) so a reworded case cannot silently
    /// drop the assertion.
    #[test]
    fn the_production_geometry_plans_the_shared_boundary() {
        // Z-Image-turbo unified stack at 1024²: B=1, H=30, Sq=Sk=4128 → 64 Mi / (30·4128) = 541 rows.
        let budget = AttentionBudget::CONSTRAINED;
        assert_eq!(budget.query_block(1, 30, 4128, 4128), 541);
        let shared = gen_core::attention_budget::CROSS_BACKEND_CHUNK_CASES
            .iter()
            .find(|c| {
                c.budget == CONSTRAINED_ATTN_SCORES_BUDGET
                    && c.rows_per_query == 30 * 4128
                    && c.sq == 4128
            })
            .expect("the shared table must carry the production z-image geometry");
        assert_eq!(
            budget.query_block(1, 30, 4128, 4128) as u64,
            shared.expect_rows,
            "the MLX 4-D adapter diverged from the shared table at the production geometry"
        );
        // A call that already fits the budget is a single un-chunked pass.
        assert_eq!(budget.query_block(1, 30, 32, 32), 32);
    }

    #[test]
    fn chunked_matches_the_unbounded_forward_within_the_documented_tolerance() {
        // [B=1, H=3, Sq=Sk=37, D=8] — 37 is prime, so every chunk size leaves a ragged tail.
        let (b, h, s, d) = (1, 3, 37, 8);
        let q = arange(&[b, h, s, d], 0.017, 0.3);
        let k = arange(&[b, h, s, d], 0.013, -0.7);
        let v = arange(&[b, h, s, d], 0.011, 1.1);
        let scale = (d as f32).powf(-0.5);

        // Sweep block sizes by choosing budgets that yield 1, 5, 18, 37 (== the whole call) rows.
        for rows in [1i32, 5, 18, 37] {
            let budget = AttentionBudget::from_score_elements(
                (b * h * s * rows) as u64,
                rows % 2 == 1, // exercise both eval_per_chunk settings
            );
            assert_eq!(budget.query_block(b, h, s, s), rows, "rows {rows}");
            let expected_chunks = (s as usize).div_ceil(rows as usize);
            let max_rel = chunked_delta(&q, &k, &v, scale, None, budget, expected_chunks);
            assert!(
                max_rel < PRODUCTION_TOLERANCE,
                "rows {rows}: max relative delta {max_rel:e} exceeds {PRODUCTION_TOLERANCE:e}"
            );
        }

        // Mutation check — the metric is not vacuously zero: perturbing `v` must move it far past
        // the tolerance, so a comparison that accidentally compared an array to itself fails here.
        let v_moved = mlx_rs::ops::add(&v, Array::from_slice(&[0.5f32], &[1])).unwrap();
        let full = sdpa_budgeted_bhsd(&q, &k, &v, scale, None, AttentionPlan::UNBOUNDED).unwrap();
        let moved =
            sdpa_budgeted_bhsd(&q, &k, &v_moved, scale, None, AttentionPlan::UNBOUNDED).unwrap();
        let (a, c) = (flat(&full), flat(&moved));
        let peak = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
        let moved_rel = a
            .iter()
            .zip(&c)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
            / peak;
        assert!(
            moved_rel > 100.0 * PRODUCTION_TOLERANCE,
            "mutation check: perturbing v moved the metric only {moved_rel:e}"
        );
    }

    /// Two claims at the **production** Z-Image-turbo DiT shape (1024² → 4128 unified tokens, 30
    /// heads, head_dim 128, f32 — the dtype the DiT stream runs at after `apply_pad`), asserted in one
    /// test so the ~250 MB q/k/v allocation happens once rather than twice:
    ///
    /// 1. MLX's fused SDPA **streams** the `[B,H,Sq,Sk]` scores rather than materializing them, so
    ///    there is no score tensor for query-row chunking to bound (unlike candle's
    ///    `attention_basic`). This is the load-bearing fact behind the whole SC-15615 finding, and it
    ///    is pinned by arithmetic, not a magic constant: with q/k/v already evaluated, the unbounded
    ///    call's peak-active bytes must stay within one operand's slack of `4 · B·H·Sq·D · 4`.
    ///    A materialized score tensor would add `B·H·Sq·Sk · 4` — 1.9 GiB here, ~8× the budget below.
    /// 2. At the production 64 Mi budget the chunked and unbounded forwards agree **exactly** — the
    ///    strongest form of the equivalence claim, at the geometry that ships.
    #[test]
    fn at_the_production_shape_sdpa_streams_the_scores_and_chunking_is_exact() {
        use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

        let (b, h, s, d) = (1i64, 30i64, 4128i64, 128i64);
        let shape = [b as i32, h as i32, s as i32, d as i32];
        let q = arange(&shape, 0.0007, 0.3);
        let k = arange(&shape, 0.0011, -0.7);
        let v = arange(&shape, 0.0013, 1.1);
        let scale = (d as f32).powf(-0.5);
        mlx_rs::transforms::eval([&q, &k, &v]).unwrap();

        // (1) No score materialization on the unbounded path.
        clear_cache();
        reset_peak_memory();
        reset_chunk_count();
        let unbounded =
            sdpa_budgeted_bhsd(&q, &k, &v, scale, None, AttentionPlan::UNBOUNDED).unwrap();
        mlx_rs::transforms::eval([&unbounded]).unwrap();
        let peak = get_peak_memory() as i64;
        assert_eq!(last_chunk_count(), 1, "the unbounded arm must not chunk");
        let one_operand = b * h * s * d * 4; // f32
        let scores = b * h * s * s * 4;
        assert!(
            peak <= 5 * one_operand,
            "unbounded SDPA peaked at {peak} B, more than q+k+v+out ({} B) plus one operand of \
             slack — MLX appears to be materializing the {scores} B score tensor, which would make \
             query-row chunking a real memory rung on this backend. Re-measure SC-15615.",
            4 * one_operand
        );

        // (2) Exact agreement at the production budget. 4128 rows / 541 per chunk = 8 chunks.
        let budget = AttentionBudget::CONSTRAINED;
        assert_eq!(
            budget.query_block(shape[0], shape[1], shape[2], shape[2]),
            541
        );
        reset_chunk_count();
        let chunked =
            sdpa_budgeted_bhsd(&q, &k, &v, scale, None, AttentionPlan::budgeted(budget)).unwrap();
        assert_eq!(
            last_chunk_count(),
            8,
            "the budgeted arm did not chunk — the exactness assertion below would be vacuous"
        );
        assert_eq!(chunked.shape(), unbounded.shape());
        let (a, c) = (flat(&unbounded), flat(&chunked));
        let peak_abs = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
        let max_rel = a
            .iter()
            .zip(&c)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
            / peak_abs;
        assert_eq!(
            max_rel, 0.0,
            "production-shape chunking must be exact, got {max_rel:e}"
        );
        drop((unbounded, chunked));
        clear_cache();
    }

    #[test]
    fn chunked_matches_with_a_broadcast_and_a_per_query_mask() {
        let (b, h, s, d) = (1, 2, 24, 8);
        let q = arange(&[b, h, s, d], 0.019, 0.2);
        let k = arange(&[b, h, s, d], 0.023, -0.4);
        let v = arange(&[b, h, s, d], 0.007, 0.9);
        let scale = (d as f32).powf(-0.5);
        let budget = AttentionBudget::from_score_elements((b * h * s * 7) as u64, false);
        assert_eq!(budget.query_block(b, h, s, s), 7);

        for mask in [
            arange(&[b, 1, 1, s], 0.31, 0.0),  // broadcast over query rows
            arange(&[b, 1, s, s], 0.29, -0.1), // per-query rows -> must be narrowed
        ] {
            let max_rel = chunked_delta(&q, &k, &v, scale, Some(&mask), budget, 4);
            assert!(
                max_rel < PRODUCTION_TOLERANCE,
                "mask {:?}: max relative delta {max_rel:e}",
                mask.shape()
            );
        }
    }

    /// A per-query mask that is NOT narrowed per chunk is the classic chunking bug (every chunk would
    /// see row 0's mask). Pin that the narrowing actually happens by using a mask whose rows differ
    /// strongly: an un-narrowed implementation would blow past the tolerance.
    #[test]
    fn a_per_query_mask_is_narrowed_to_each_chunks_own_rows() {
        let (b, h, s, d) = (1, 2, 24, 8);
        let q = arange(&[b, h, s, d], 0.019, 0.2);
        let k = arange(&[b, h, s, d], 0.023, -0.4);
        let v = arange(&[b, h, s, d], 0.007, 0.9);
        let scale = (d as f32).powf(-0.5);
        let budget = AttentionBudget::from_score_elements((b * h * s * 7) as u64, false);

        // A strongly row-dependent additive mask: row i suppresses all keys except key i.
        let mut data = vec![-1.0e4f32; (b * s * s) as usize];
        for i in 0..s {
            data[(i * s + i) as usize] = 0.0;
        }
        let mask = Array::from_slice(&data, &[b, 1, s, s]);
        let max_rel = chunked_delta(&q, &k, &v, scale, Some(&mask), budget, 4);
        assert!(
            max_rel < PRODUCTION_TOLERANCE,
            "row-dependent mask: max relative delta {max_rel:e}"
        );
    }

    #[test]
    fn rejects_a_non_4d_query() {
        let q = arange(&[2, 3, 4], 0.1, 0.0);
        let err = sdpa_budgeted_bhsd(&q, &q, &q, 1.0, None, AttentionPlan::UNBOUNDED)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[B, H, Sq, D]"), "{err}");
    }

    /// `k`/`v` are indexed on the same axes as `q`; a lower-rank one would silently read the head
    /// axis as the key length and produce a wrong (not failed) chunk plan. Same for a rank-1 mask,
    /// whose narrowing would underflow `len() - 2`.
    #[test]
    fn rejects_mismatched_key_value_and_mask_ranks() {
        let q = arange(&[1, 2, 8, 4], 0.1, 0.0);
        let flat = arange(&[1, 2, 8], 0.1, 0.0);
        for (label, k, v) in [("k", &flat, &q), ("v", &q, &flat)] {
            let err = sdpa_budgeted_bhsd(&q, k, v, 1.0, None, AttentionPlan::UNBOUNDED)
                .unwrap_err()
                .to_string();
            assert!(err.contains("[B, H, Sk, D]"), "{label}: {err}");
        }
        let bad_mask = arange(&[8], 0.1, 0.0);
        let err = sdpa_budgeted_bhsd(&q, &q, &q, 1.0, Some(&bad_mask), AttentionPlan::UNBOUNDED)
            .unwrap_err()
            .to_string();
        assert!(err.contains("query and a key axis"), "{err}");
    }

    /// Between-chunk cancellation (SC-15615 scope: "per-chunk cancellation checks"). A bounded call
    /// splits one atomic kernel launch into N, and the boundaries are the only place inside a DiT
    /// forward a cancel can land; the unbounded fast path has no boundary and must be unaffected.
    #[test]
    fn a_cancel_stops_a_bounded_call_between_chunks() {
        let (b, h, s, d) = (1, 2, 24, 8);
        let q = arange(&[b, h, s, d], 0.019, 0.2);
        let k = arange(&[b, h, s, d], 0.023, -0.4);
        let v = arange(&[b, h, s, d], 0.007, 0.9);
        let scale = (d as f32).powf(-0.5);
        let budget = AttentionBudget::from_score_elements((b * h * s * 7) as u64, false);

        let cancel = CancelFlag::default();
        cancel.cancel();
        let plan = AttentionPlan::budgeted(budget).with_cancel(&cancel);
        let err = sdpa_budgeted_bhsd(&q, &k, &v, scale, None, plan).unwrap_err();
        assert!(
            matches!(err, Error::Canceled),
            "a bounded call must return Error::Canceled, got {err}"
        );

        // An un-tripped flag changes nothing, and the UNBOUNDED fast path never consults it (there is
        // no chunk boundary to consult it at) — so a cancelled flag cannot break an unselected request.
        let live = CancelFlag::default();
        let plan = AttentionPlan::budgeted(budget).with_cancel(&live);
        assert!(sdpa_budgeted_bhsd(&q, &k, &v, scale, None, plan).is_ok());
        let plan = AttentionPlan::UNBOUNDED.with_cancel(&cancel);
        assert!(
            sdpa_budgeted_bhsd(&q, &k, &v, scale, None, plan).is_ok(),
            "a cancelled flag must not affect the unbounded fast path"
        );
    }

    // ── The head axis (sc-18661) ────────────────────────────────────────────────────────────────

    /// **Exact reconstruction on the head axis**, pinned at `0.0` rather than at a tolerance.
    ///
    /// Splitting complete heads leaves every call's `Sq` — and therefore the query GEMM's `M` — at
    /// its unbounded value, so the arithmetic is untouched, there is no accumulator to split, and MLX
    /// dispatches the same Metal specialization. Every heads-per-chunk from 1 to 4 over a **prime**
    /// head count is swept, so every arm leaves a ragged tail and a single lucky boundary cannot
    /// carry the claim.
    ///
    /// Each arm additionally asserts that the query axis stayed **whole** inside the head chunk. That
    /// is the condition the exactness depends on ([`AttentionChunkAxis`]): a budget below one head's
    /// score domain makes this kernel fall back to query rows and inherit their weaker contract while
    /// still reporting itself as the head axis, and the zeros below would then be measuring the wrong
    /// thing. The final arm shows the same budget genuinely binding on the query axis, so the zeros
    /// are the axis's doing rather than a budget that never engaged.
    #[test]
    fn the_head_axis_reconstructs_the_unbounded_forward_exactly() {
        // 5 heads is prime, so every heads-per-chunk below it leaves a ragged tail.
        let (b, h, s, d) = (1, 5, 24, 8);
        let q = arange(&[b, h, s, d], 0.019, 0.2);
        let k = arange(&[b, h, s, d], 0.023, -0.4);
        let v = arange(&[b, h, s, d], 0.007, 0.9);
        let scale = (d as f32).powf(-0.5);
        let full = sdpa_budgeted_bhsd(&q, &k, &v, scale, None, AttentionPlan::UNBOUNDED).unwrap();

        let mut split_arms = 0;
        for heads_per_chunk in [1u64, 2, 3, 4] {
            // `head_chunks` splits when `B·H·Sq·Sk` exceeds the budget and yields
            // `budget / (B·Sq·Sk)` heads per chunk.
            let per_head = (b * s * s) as u64;
            let budget = AttentionBudget::from_score_elements(per_head * heads_per_chunk, true);
            reset_chunk_count();
            let chunked =
                sdpa_head_budgeted_bhsd(&q, &k, &v, scale, None, AttentionPlan::budgeted(budget))
                    .unwrap();
            let expected = (h as usize).div_ceil(heads_per_chunk as usize);
            assert_eq!(
                last_head_chunk_count(),
                expected,
                "at {heads_per_chunk} heads per chunk the kernel ran {} head chunks, the shared \
                 planner implies {expected}",
                last_head_chunk_count()
            );
            assert_eq!(
                last_chunk_count(),
                1,
                "with H narrowed to {heads_per_chunk} the query axis must stay whole, or the \
                 exactness below is not the head axis's property"
            );
            assert_eq!(chunked.shape(), full.shape());
            let (a, c) = (flat(&full), flat(&chunked));
            let max_abs = a
                .iter()
                .zip(&c)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert_eq!(
                max_abs, 0.0,
                "heads_per_chunk {heads_per_chunk}: head-axis chunking must be bit-exact, got a \
                 max absolute difference of {max_abs:e}"
            );
            split_arms += 1;
        }
        assert_eq!(
            split_arms, 4,
            "every arm must have split, or this test passes with the head axis gone"
        );

        // **The contrast, and the anti-vacuity guard in one.** The same budget on the query axis
        // must actually chunk — proving the budget was binding and that the zero above is the
        // *axis's* doing rather than a budget that never engaged.
        let budget = AttentionBudget::from_score_elements((b * s * s) as u64, true);
        reset_chunk_count();
        sdpa_budgeted_bhsd(&q, &k, &v, scale, None, AttentionPlan::budgeted(budget)).unwrap();
        assert!(
            last_chunk_count() > 1,
            "the query axis did not chunk at the budget the head axis split five ways — the \
             comparison above is not between two axes at one budget"
        );
    }

    /// An in-budget, unbounded or single-head call must take the query-row kernel's own fast path
    /// rather than entering a head loop that would concatenate one chunk back together.
    #[test]
    fn the_head_axis_delegates_when_there_is_nothing_to_split() {
        let (b, s, d) = (1, 12, 8);
        let scale = (d as f32).powf(-0.5);
        for (label, h, budget) in [
            ("unbounded", 4, AttentionBudget::UNBOUNDED),
            (
                "single head",
                1,
                AttentionBudget::from_score_elements(1, true),
            ),
            (
                "in budget",
                4,
                AttentionBudget::from_score_elements(u64::MAX - 1, true),
            ),
        ] {
            let q = arange(&[b, h, s, d], 0.019, 0.2);
            reset_chunk_count();
            let out =
                sdpa_head_budgeted_bhsd(&q, &q, &q, scale, None, AttentionPlan::budgeted(budget))
                    .unwrap();
            assert_eq!(out.shape(), &[b, h, s, d], "{label}");
            assert_eq!(last_head_chunk_count(), 1, "{label}: must not split heads");
        }
    }

    /// A mask with a real head axis must be narrowed with the heads. Un-narrowed, every chunk would
    /// see head 0's mask — the head-axis twin of the classic per-query chunking bug — so the mask is
    /// made strongly head-dependent and the result compared against the unbounded call.
    #[test]
    fn a_per_head_mask_is_narrowed_to_each_chunks_own_heads() {
        let (b, h, s, d) = (1, 4, 10, 8);
        let q = arange(&[b, h, s, d], 0.019, 0.2);
        let k = arange(&[b, h, s, d], 0.023, -0.4);
        let v = arange(&[b, h, s, d], 0.007, 0.9);
        let scale = (d as f32).powf(-0.5);

        // Head `i` may attend only to key `i`; an un-narrowed mask makes every chunk head 0's.
        let mut data = vec![-1.0e4f32; (b * h * s) as usize];
        for head in 0..h {
            for row in 0..s {
                data[(head * s + row) as usize] = if row == head { 0.0 } else { -1.0e4 };
            }
        }
        let mask = Array::from_slice(&data, &[b, h, 1, s]);
        let full =
            sdpa_budgeted_bhsd(&q, &k, &v, scale, Some(&mask), AttentionPlan::UNBOUNDED).unwrap();

        let budget = AttentionBudget::from_score_elements((b * s * s) as u64, false);
        reset_chunk_count();
        let chunked = sdpa_head_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            Some(&mask),
            AttentionPlan::budgeted(budget),
        )
        .unwrap();
        assert_eq!(last_head_chunk_count(), 4, "the head axis must have split");
        let (a, c) = (flat(&full), flat(&chunked));
        let max_abs = a
            .iter()
            .zip(&c)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert_eq!(max_abs, 0.0, "a per-head mask was not narrowed per chunk");

        // A broadcast mask has no head axis to narrow and must be shared unchanged.
        let broadcast = arange(&[b, 1, 1, s], 0.31, 0.0);
        let full = sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            Some(&broadcast),
            AttentionPlan::UNBOUNDED,
        )
        .unwrap();
        let chunked = sdpa_head_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            Some(&broadcast),
            AttentionPlan::budgeted(budget),
        )
        .unwrap();
        let (a, c) = (flat(&full), flat(&chunked));
        let max_abs = a
            .iter()
            .zip(&c)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert_eq!(max_abs, 0.0, "a broadcast mask must survive head chunking");
    }

    /// A single head that still exceeds the budget must fall back to query rows **inside** the head
    /// chunk — the order the shared planner prescribes — rather than silently exceeding it.
    #[test]
    fn one_head_over_budget_falls_back_to_query_rows_inside_the_chunk() {
        let (b, h, s, d) = (1, 3, 20, 8);
        let q = arange(&[b, h, s, d], 0.019, 0.2);
        let scale = (d as f32).powf(-0.5);
        // Half of one head's score domain: the head axis can only reach one head per chunk, and one
        // head is still twice the budget.
        let budget = AttentionBudget::from_score_elements(((b * s * s) / 2) as u64, false);
        reset_chunk_count();
        let out = sdpa_head_budgeted_bhsd(&q, &q, &q, scale, None, AttentionPlan::budgeted(budget))
            .unwrap();
        assert_eq!(out.shape(), &[b, h, s, d]);
        assert_eq!(last_head_chunk_count(), h as usize, "one head per chunk");
        assert!(
            last_chunk_count() > 1,
            "one head still over budget must chunk query rows inside the head chunk, ran {}",
            last_chunk_count()
        );
    }

    /// Between-chunk cancellation on the head axis, and inertness of a cancelled flag on the
    /// unbounded fast path — the rung's shared contract, asserted on the new axis too.
    #[test]
    fn a_cancel_stops_a_head_bounded_call_between_chunks() {
        let (b, h, s, d) = (1, 4, 12, 8);
        let q = arange(&[b, h, s, d], 0.019, 0.2);
        let scale = (d as f32).powf(-0.5);
        let budget = AttentionBudget::from_score_elements((b * s * s) as u64, false);

        let cancel = CancelFlag::default();
        cancel.cancel();
        let plan = AttentionPlan::budgeted(budget).with_cancel(&cancel);
        assert!(
            matches!(
                sdpa_head_budgeted_bhsd(&q, &q, &q, scale, None, plan).unwrap_err(),
                Error::Canceled
            ),
            "a head-bounded call must return Error::Canceled"
        );
        let plan = AttentionPlan::UNBOUNDED.with_cancel(&cancel);
        assert!(
            sdpa_head_budgeted_bhsd(&q, &q, &q, scale, None, plan).is_ok(),
            "a cancelled flag must not affect the unbounded fast path"
        );
    }

    /// A k/v whose head axis disagrees with q would pair each query head with the wrong key head —
    /// wrong output, not a failure — so it is refused rather than sliced.
    #[test]
    fn the_head_axis_rejects_a_key_value_with_a_different_head_count() {
        let q = arange(&[1, 4, 8, 4], 0.1, 0.0);
        let wrong = arange(&[1, 2, 8, 4], 0.1, 0.0);
        for (label, k, v) in [("k", &wrong, &q), ("v", &q, &wrong)] {
            let err = sdpa_head_budgeted_bhsd(&q, k, v, 1.0, None, AttentionPlan::UNBOUNDED)
                .unwrap_err()
                .to_string();
            assert!(err.contains("q's 4 heads"), "{label}: {err}");
        }
        let flat3 = arange(&[1, 4, 8], 0.1, 0.0);
        let err = sdpa_head_budgeted_bhsd(&q, &flat3, &q, 1.0, None, AttentionPlan::UNBOUNDED)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[B, H, Sk, D]"), "{err}");
    }

    /// [`sdpa_bounded_bhsd`] must route to the axis it is handed and to no other. Asserted through
    /// the two probes, which are the only observable that distinguishes the axes at equal output.
    #[test]
    fn the_axis_selector_routes_to_the_axis_it_is_given() {
        let (b, h, s, d) = (1, 4, 16, 8);
        let q = arange(&[b, h, s, d], 0.019, 0.2);
        let scale = (d as f32).powf(-0.5);
        let budget = AttentionBudget::from_score_elements((b * s * s) as u64, false);

        reset_chunk_count();
        sdpa_bounded_bhsd(&q, &q, &q, scale, None, BoundedAttention::heads(budget)).unwrap();
        assert_eq!(last_head_chunk_count(), 4);
        assert_eq!(last_chunk_count(), 1);

        reset_chunk_count();
        sdpa_bounded_bhsd(
            &q,
            &q,
            &q,
            scale,
            None,
            BoundedAttention::query_rows(budget),
        )
        .unwrap();
        assert_eq!(
            last_head_chunk_count(),
            0,
            "the query-row route must not touch the head probe"
        );
        assert!(last_chunk_count() > 1);

        reset_chunk_count();
        sdpa_bounded_bhsd(&q, &q, &q, scale, None, BoundedAttention::UNBOUNDED).unwrap();
        assert_eq!(last_chunk_count(), 1);
    }

    #[test]
    fn the_chunk_probe_reports_the_unchunked_path_as_one_chunk() {
        // Guards the guard: if `record_chunk_count` were dropped from the fast path, every
        // `expect_chunks` assertion above would compare against a stale value instead of failing.
        let q = arange(&[1, 2, 8, 4], 0.1, 0.0);
        reset_chunk_count();
        assert_eq!(last_chunk_count(), 0);
        sdpa_budgeted_bhsd(&q, &q, &q, 1.0, None, AttentionPlan::UNBOUNDED).unwrap();
        assert_eq!(last_chunk_count(), 1);
    }
}
