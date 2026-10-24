//! sc-15750 — rung 4 (`mlx_gen::block_residency`) bounds transformer weight residency on real
//! packed-quantized weights, and the materialize guard that makes it work is load-bearing.
//!
//! `#[ignore]`d and env-gated. Point `MLX_GEN_BLOCK_WEIGHTS` at a packed-quantized transformer
//! `.safetensors` whose blocks are keyed `layers.<n>.…` (measured against `SceneWorks/z-image-turbo-mlx`
//! q4), then:
//!   MLX_GEN_BLOCK_WEIGHTS=/path/to/transformer/model.safetensors \
//!     cargo test -p mlx-gen --release --test integration block_residency_real_weights:: -- --ignored --nocapture
//!
//! The path is supplied, never derived: inference does not resolve HF caches itself — the caller owns
//! artifact resolution and hands down a concrete path. `scripts/check-workspace.py` enforces that.
//!
//! The "forward pass" here is a chain of `quantized_matmul` calls through each block's `attention.to_k`
//! triple. That is not the real Z-Image block, and deliberately so: this is the SHARED primitive's
//! test, so it must not depend on any one family's block. What it does reproduce faithfully is the
//! property the primitive has to survive — a **lazy graph chain** where each window's output still
//! references that window's weights until something forces evaluation.

use std::ops::Range;

use gen_core::runtime::CancelFlag;
use mlx_gen::block_residency::{run_windowed, BlockPlan};
use mlx_gen::weights::Weights;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

const N_BLOCKS: usize = 30;
const WIDTH: i32 = 3840;

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// The caller-supplied weights path, or `None` to skip. Env-only by design — see the module docs.
fn path() -> Option<String> {
    let p = std::env::var("MLX_GEN_BLOCK_WEIGHTS").ok()?;
    std::path::Path::new(&p).exists().then_some(p)
}

/// Chain one window's blocks into the carried activation, taking each block's tensors OUT of the
/// view so the drop in `run_windowed` is a real release.
fn apply_window(state: Array, view: &mut Weights, range: Range<usize>) -> mlx_gen::Result<Array> {
    let mut cur = state;
    for i in range {
        let base = format!("layers.{i}.attention.to_k");
        let (Some(w), Some(scales), Some(biases)) = (
            view.remove(&format!("{base}.weight")),
            view.remove(&format!("{base}.scales")),
            view.remove(&format!("{base}.biases")),
        ) else {
            continue;
        };
        cur = mlx_rs::ops::quantized_matmul(
            &cur,
            &w,
            &scales,
            Some(&biases),
            Some(true),
            Some(64),
            Some(4),
        )?;
    }
    Ok(cur)
}

fn x0() -> Array {
    Array::zeros::<f32>(&[1, WIDTH])
        .and_then(|a| a.as_dtype(Dtype::Bfloat16))
        .expect("x0")
}

/// Run a full sweep at `window`, returning the MLX peak (MiB). `materialize` off reproduces the trap.
fn sweep(p: &str, window: usize, materialize_on: bool) -> f64 {
    let plan = BlockPlan::new(N_BLOCKS, window).expect("plan");
    let cancel = CancelFlag::default();
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();

    let out = run_windowed(
        &plan,
        &cancel,
        x0(),
        || Weights::from_file(p),
        apply_window,
        |s: &Array| {
            if materialize_on {
                eval([s])?;
            }
            Ok(())
        },
    )
    .expect("windowed run");
    eval([&out]).expect("final eval");
    mib(mlx_rs::memory::get_peak_memory())
}

#[test]
#[ignore = "set MLX_GEN_BLOCK_WEIGHTS to a packed-quantized transformer safetensors"]
fn block_window_bounds_transformer_residency() {
    let Some(p) = path() else {
        println!("SKIP: set MLX_GEN_BLOCK_WEIGHTS to a packed-quantized transformer safetensors");
        return;
    };

    println!("\n  window   peak (MiB)   vs resident");
    let resident = sweep(&p, N_BLOCKS, true);
    let mut rows = Vec::new();
    for w in [1usize, 2, 4, 8] {
        let peak = sweep(&p, w, true);
        println!(
            "  {w:>6}   {peak:>10.1}   {:>6.1}x",
            resident / peak.max(1.0)
        );
        rows.push((w, peak));
    }
    println!("  {N_BLOCKS:>6}   {resident:>10.1}   (resident baseline)");

    let (_, w1) = rows[0];
    assert!(
        w1 < resident * 0.5,
        "window=1 peak {w1:.1} MiB should be far below resident {resident:.1} MiB"
    );
    // Peak must be monotonically non-decreasing in window size — that is the whole contract.
    for pair in rows.windows(2) {
        let ((wa, pa), (wb, pb)) = (pair[0], pair[1]);
        assert!(
            pb >= pa * 0.9,
            "window {wb} peak {pb:.1} should not be materially below window {wa} peak {pa:.1}"
        );
    }
}

/// MUTATION CHECK — without the materialize call, the carried activation is an unevaluated graph node
/// that still references every window's weights, so dropping frees nothing and the bound silently
/// does not hold. If this test ever passes with a bounded peak, the guard in `run_windowed` has
/// stopped doing anything and the whole rung is decorative.
#[test]
#[ignore = "set MLX_GEN_BLOCK_WEIGHTS to a packed-quantized transformer safetensors"]
fn block_window_without_materialize_frees_nothing() {
    let Some(p) = path() else {
        println!("SKIP: set MLX_GEN_BLOCK_WEIGHTS to a packed-quantized transformer safetensors");
        return;
    };

    let guarded = sweep(&p, 1, true);
    let unguarded = sweep(&p, 1, false);
    println!("\n  window=1 WITH materialize:    {guarded:>9.1} MiB");
    println!("  window=1 WITHOUT materialize: {unguarded:>9.1} MiB");
    println!(
        "  guard is worth:               {:>9.1} MiB",
        unguarded - guarded
    );

    assert!(
        unguarded > guarded * 2.0,
        "the materialize guard must be load-bearing: unguarded {unguarded:.1} MiB vs guarded \
         {guarded:.1} MiB. If these are close, dropping a window is already freeing its weights and \
         this module's central claim needs re-deriving."
    );
}
