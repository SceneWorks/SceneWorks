//! Rung 4 — **bounded transformer residency**, the MLX half (SC-15750, hoisted by SC-15790).
//!
//! The schedule itself lives in [`gen_core::block_window`]: window arithmetic, loop order, release
//! discipline and the cancellation contract are identical on every backend. This module supplies the
//! two MLX-specific operations and keeps the call shape the providers already use.
//!
//! MLX's half is small on purpose:
//! - the view is [`Weights`] — `Array` handles from `load_safetensors`, which are **lazy per tensor**
//!   (1073 handles cost 0.0 MiB until something is evaluated), so re-opening per window is nearly free;
//! - the release is `mlx_rs::memory::clear_cache()`.
//!
//! ## Measured on MLX (SC-15744 / SC-15750, real z-image-turbo q4, Apple/Metal)
//!
//! These are MLX numbers and establish only that the shape works — they predict nothing about another
//! backend, which is the rule rung 3 taught the hard way (~6x different value across backends).
//!
//! One block materializes at **97.3 MiB** against a header-computed 97.1 MiB — no hidden copy.
//! Dropping returns it: active falls to 8.0 MiB and stays flat across all 30 blocks. Packed-quant
//! survives it — `quantized_matmul` works on a per-block materialized `weight`/`scales`/`biases`
//! triple, because each is an independent file entry rather than a slice of one packed buffer.
//!
//! | window | peak | vs resident |
//! |---:|---:|---:|
//! | 1 | 8.0 MiB | 29.9x |
//! | 2 | 15.9 MiB | 15.0x |
//! | 4 | 31.8 MiB | 7.5x |
//! | 8 | 63.6 MiB | 3.7x |
//! | 30 (resident) | 238.4 MiB | — |
//!
//! Linear in window size, which is the contract. Re-materializing every block once per denoise step
//! cost ~0.309 s/step with a warm page cache on a 128 GB machine — the OPTIMISTIC bound, since the
//! hosts this rung exists for are small Macs under pressure where the page cache is evicted first.
//!
//! ## The trap, and why it is MLX's reason and not a universal one
//!
//! MLX is lazy, so the carried activation is an unevaluated graph node still referencing the window's
//! weights: **dropping before forcing evaluation frees nothing**. Measured at window = 1, **8.0 MiB
//! with the guard vs 238.4 MiB without** — identical to fully resident, with correct output either
//! way. Silent, not loud. `block_window_without_materialize_frees_nothing` pins it.
//!
//! An eager backend has no lazy graph to cut and may not need this at all — or may need it for an
//! unrelated reason. [`gen_core::block_window`] carries the obligation's shape; the reason is per
//! backend and must be measured there.

use std::ops::Range;

use gen_core::block_window::BlockWindowBackend;
use gen_core::runtime::CancelFlag;

use crate::weights::Weights;
use crate::Result;

/// Re-exported so providers keep a single import path. The type is backend-neutral arithmetic.
pub use gen_core::block_window::BlockPlan;

/// Binds MLX's two operations to the shared driver: a fresh lazily-loaded [`Weights`] view per
/// window, and `clear_cache()` to return the allocator's holdings after the window drops.
struct MlxWindowBackend<F, R> {
    open: F,
    release: R,
}

impl<F, R> BlockWindowBackend for MlxWindowBackend<F, R>
where
    F: FnMut() -> Result<Weights>,
    R: Fn(),
{
    type View = Weights;

    fn open_view(&mut self) -> gen_core::Result<Weights> {
        (self.open)().map_err(Into::into)
    }

    fn release(&self) {
        (self.release)();
    }
}

fn run_windowed_with_release<S>(
    plan: &BlockPlan,
    cancel: &CancelFlag,
    init: S,
    open: impl FnMut() -> Result<Weights>,
    mut apply: impl FnMut(S, &mut Weights, Range<usize>) -> Result<S>,
    materialize: impl Fn(&S) -> Result<()>,
    release: impl Fn(),
) -> Result<S> {
    let mut backend = MlxWindowBackend { open, release };
    gen_core::block_window::run_windowed(
        &mut backend,
        plan,
        cancel,
        init,
        |state, view, range| apply(state, view, range).map_err(Into::into),
        |state| materialize(state).map_err(Into::into),
    )
    .map_err(Into::into)
}

/// Run one denoise step's worth of block windows over MLX weights.
///
/// Thin adapter over [`gen_core::block_window::run_windowed`] — see there for the lifecycle and the
/// ordering guarantees. The signature is unchanged from before the hoist, so provider code needs no
/// edits.
///
/// `apply` MUST take its tensors out of the view (`Weights::remove`) rather than cloning them:
/// `Array` is refcounted, so a clone left in the map keeps the materialized buffer resident and the
/// bound silently does not hold.
pub fn run_windowed<S>(
    plan: &BlockPlan,
    cancel: &CancelFlag,
    init: S,
    open: impl FnMut() -> Result<Weights>,
    mut apply: impl FnMut(S, &mut Weights, Range<usize>) -> Result<S>,
    materialize: impl Fn(&S) -> Result<()>,
) -> Result<S> {
    run_windowed_with_release(
        plan,
        cancel,
        init,
        open,
        |state, view, range| apply(state, view, range),
        materialize,
        mlx_rs::memory::clear_cache,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// The adapter must forward MLX's synchronization callback before it clears MLX's allocator
    /// cache. `release` is injected here only so the otherwise-global clear is observable without
    /// relying on process-wide allocator counters or real weights.
    #[test]
    fn mlx_adapter_materializes_before_releasing_each_window() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let open_log = Rc::clone(&log);
        let apply_log = Rc::clone(&log);
        let materialize_log = Rc::clone(&log);
        let release_log = Rc::clone(&log);

        let out = run_windowed_with_release(
            &BlockPlan::new(2, 1).unwrap(),
            &CancelFlag::default(),
            0usize,
            move || {
                open_log.borrow_mut().push("open");
                Ok(Weights::empty())
            },
            move |state, _view, _range| {
                apply_log.borrow_mut().push("apply");
                Ok(state + 1)
            },
            move |_| {
                materialize_log.borrow_mut().push("materialize");
                Ok(())
            },
            move || release_log.borrow_mut().push("clear_cache"),
        )
        .unwrap();

        assert_eq!(out, 2);
        assert_eq!(
            log.borrow().as_slice(),
            [
                "open",
                "apply",
                "materialize",
                "clear_cache",
                "open",
                "apply",
                "materialize",
                "clear_cache"
            ]
        );
    }

    /// The MLX error bridge and backend release must both run for cancellation raised during the
    /// materialization callback, not only for a flag cancelled before the first view opened.
    #[test]
    fn mlx_adapter_releases_after_typed_cancel_during_materialize() {
        let releases = Rc::new(RefCell::new(0usize));
        let released = Rc::clone(&releases);
        let err = run_windowed_with_release(
            &BlockPlan::new(2, 1).unwrap(),
            &CancelFlag::default(),
            0usize,
            || Ok(Weights::empty()),
            |state, _view, _range| Ok(state),
            |_| Err(Error::Canceled),
            move || *released.borrow_mut() += 1,
        )
        .unwrap_err();

        assert!(matches!(err, Error::Canceled));
        assert_eq!(
            *releases.borrow(),
            1,
            "the active MLX window must be released"
        );
    }

    /// The cancellation path now crosses a crate boundary — `gen_core::Error::Canceled` has to come
    /// back as `mlx_gen::Error::Canceled`, not collapse into `Error::Msg`.
    ///
    /// A real regression risk rather than a hypothetical: the pre-hoist driver returned
    /// `Error::Msg("cancelled")` and SC-15754 had to fix it to the typed variant, because a
    /// stringified cancellation reports a cancelled render as a FAILED job (sc-4481). The hoist
    /// re-routes that path through two `From` conversions, either of which could silently flatten it.
    #[test]
    fn cancellation_survives_the_gen_core_boundary_as_the_typed_variant() {
        let cancel = CancelFlag::default();
        cancel.cancel();

        let err = run_windowed(
            &BlockPlan::new(4, 2).unwrap(),
            &cancel,
            0usize,
            || Ok(Weights::empty()),
            |s, _v: &mut Weights, _r| Ok(s),
            |_| Ok(()),
        )
        .unwrap_err();

        assert!(
            matches!(err, Error::Canceled),
            "cancellation must round-trip as Error::Canceled, not a message: {err:?}"
        );
    }

    /// A provider error must arrive intact rather than being flattened by the round trip.
    #[test]
    fn a_provider_error_survives_the_boundary() {
        let err = run_windowed(
            &BlockPlan::new(2, 1).unwrap(),
            &CancelFlag::default(),
            0usize,
            || Ok(Weights::empty()),
            |_s, _v: &mut Weights, _r| -> Result<usize> {
                Err(Error::MissingTensor(
                    "layers.0.attention.to_k.weight".into(),
                ))
            },
            |_| Ok(()),
        )
        .unwrap_err();

        assert!(
            matches!(err, Error::MissingTensor(ref k) if k.ends_with("to_k.weight")),
            "expected the typed MissingTensor to survive, got {err:?}"
        );
    }
}
