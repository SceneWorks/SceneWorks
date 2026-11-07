//! SPIKE sc-15744 — can MLX materialize packed-quantized transformer blocks ONE AT A TIME?
//!
//! Rung 4 (bounded transformer residency) is the only memory-ladder rung with no shared primitive in
//! `mlx-gen`. Before building one, establish whether MLX can even express it: materialize a single
//! transformer block from a safetensors file, use it, release it, and keep peak resident memory near
//! ONE BLOCK rather than the whole model.
//!
//! Throwaway measurement code — the ANSWER is the deliverable, not this file.
//!
//! Env-gated. Point `MLX_GEN_BLOCK_WEIGHTS` at a packed-quantized transformer `.safetensors` whose
//! blocks are keyed `layers.<n>.…`, then:
//!   MLX_GEN_BLOCK_WEIGHTS=/path/to/transformer/model.safetensors \
//!     cargo test -p mlx-gen --release --test integration rung4_block_residency_spike:: -- --ignored --nocapture
//!
//! The path is supplied, never derived: inference does not resolve HF caches itself — the caller owns
//! artifact resolution and hands down a concrete path. `scripts/check-workspace.py` enforces that.
//!
//! Subject measured: z-image-turbo q4 transformer — 3.23 GiB, 1073 tensors, `layers.0..29`
//! (30 blocks), ~97 MiB/block, 2.85 GiB of block weights. Packed-quant triple per linear:
//! `weight U32[out, in/8]` + `scales`/`biases` BF16[out, in/group_size].

use std::collections::{BTreeMap, HashMap};

use mlx_rs::memory::{
    clear_cache, get_active_memory, get_cache_memory, get_peak_memory, reset_peak_memory,
};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// The caller-supplied weights path, or `None` to skip. Env-only by design — see the module docs.
fn weights_path() -> Option<String> {
    let p = std::env::var("MLX_GEN_BLOCK_WEIGHTS").ok()?;
    std::path::Path::new(&p).exists().then_some(p)
}

fn snap(label: &str) -> (f64, f64, f64) {
    let (a, c, p) = (
        mib(get_active_memory()),
        mib(get_cache_memory()),
        mib(get_peak_memory()),
    );
    println!("  {label:<44} active {a:>9.1}  cache {c:>9.1}  peak {p:>9.1}   (MiB)");
    (a, c, p)
}

/// Split `layers.<n>.<rest>` into (n, full key). Non-block tensors return `None`.
fn block_index(key: &str) -> Option<usize> {
    let rest = key.strip_prefix("layers.")?;
    let (idx, _) = rest.split_once('.')?;
    idx.parse().ok()
}

#[test]
#[ignore = "set MLX_GEN_BLOCK_WEIGHTS to a packed-quantized transformer safetensors"]
fn rung4_per_block_materialization_spike() {
    let Some(path) = weights_path() else {
        println!("SKIP: set MLX_GEN_BLOCK_WEIGHTS to a packed-quantized transformer safetensors");
        return;
    };

    println!("\n=== Q1: is load_safetensors LAZY? ===");
    clear_cache();
    reset_peak_memory();
    let (base_active, _, _) = snap("baseline (pre-load)");

    let (mut tensors, _meta): (HashMap<String, Array>, HashMap<String, String>) =
        Array::load_safetensors_with_metadata(&path).expect("load safetensors");
    let n_tensors = tensors.len();
    let (after_load, _, _) = snap("after load_safetensors (no eval)");
    println!(
        "  -> {n_tensors} tensor handles; delta {:.1} MiB",
        after_load - base_active
    );
    println!(
        "  -> LAZY: {}",
        if after_load - base_active < 64.0 {
            "YES"
        } else {
            "NO"
        }
    );

    // Group block keys.
    let mut blocks: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for key in tensors.keys() {
        if let Some(idx) = block_index(key) {
            blocks.entry(idx).or_default().push(key.clone());
        }
    }
    let n_blocks = blocks.len();
    println!("\n=== Q2/Q3/Q5: per-block materialize -> use -> release ({n_blocks} blocks) ===");

    reset_peak_memory();
    let mut max_active: f64 = 0.0;
    let mut per_block_cost: Vec<f64> = Vec::new();

    for (idx, keys) in blocks.iter() {
        // Take this block's arrays OUT of the map so dropping them is a real release.
        let mut held: Vec<(String, Array)> = Vec::with_capacity(keys.len());
        for k in keys {
            if let Some(a) = tensors.remove(k) {
                held.push((k.clone(), a));
            }
        }

        let before = mib(get_active_memory());

        // Materialize just this block.
        for (_, a) in &held {
            eval([a]).expect("eval block tensor");
        }
        let after_eval = mib(get_active_memory());
        let cost = after_eval - before;
        per_block_cost.push(cost);
        max_active = max_active.max(after_eval);

        // Q4: exercise the PACKED-QUANT path on this block, not just the raw tensors.
        let mm = held
            .iter()
            .find(|(k, _)| k.ends_with("attention.to_k.weight"))
            .map(|(k, _)| k.clone());
        if let Some(wkey) = mm {
            let base = wkey.trim_end_matches(".weight").to_owned();
            let w = &held.iter().find(|(k, _)| *k == wkey).unwrap().1;
            let s = held.iter().find(|(k, _)| *k == format!("{base}.scales"));
            let b = held.iter().find(|(k, _)| *k == format!("{base}.biases"));
            if let (Some((_, scales)), Some((_, biases))) = (s, b) {
                let in_features = scales.shape()[1] * 64; // group_size 64
                let x = Array::zeros::<f32>(&[1, in_features])
                    .and_then(|a| a.as_dtype(mlx_rs::Dtype::Bfloat16))
                    .expect("x");
                let out = mlx_rs::ops::quantized_matmul(
                    &x,
                    w,
                    scales,
                    Some(biases),
                    Some(true),
                    Some(64),
                    Some(4),
                )
                .expect("quantized_matmul");
                eval([&out]).expect("eval matmul");
                if *idx == 0 {
                    println!(
                        "  block 0 quantized_matmul OK: x{:?} @ w{:?} -> {:?}",
                        x.shape(),
                        w.shape(),
                        out.shape()
                    );
                }
                max_active = max_active.max(mib(get_active_memory()));
            }
        }

        // Release the block.
        drop(held);
        clear_cache();
        let after_drop = mib(get_active_memory());

        if *idx < 3 || *idx == n_blocks - 1 {
            println!(
                "  block {idx:<2}  +{cost:>7.1} on eval   ->  {after_eval:>8.1} active   ->  after drop {after_drop:>8.1}"
            );
        }
    }

    let bounded_peak = mib(get_peak_memory());
    println!(
        "\n  per-block eval cost: min {:.1} / max {:.1} MiB",
        per_block_cost.iter().cloned().fold(f64::INFINITY, f64::min),
        per_block_cost.iter().cloned().fold(0.0, f64::max)
    );
    println!("  MAX ACTIVE across the sweep: {max_active:.1} MiB");
    println!("  PEAK (MLX counter) across the sweep: {bounded_peak:.1} MiB");

    // Control: what does holding every block at once cost?
    println!("\n=== control: materialize ALL blocks resident ===");
    drop(tensors);
    clear_cache();
    reset_peak_memory();
    snap("after releasing everything");

    let (all, _): (HashMap<String, Array>, HashMap<String, String>) =
        Array::load_safetensors_with_metadata(&path).expect("reload");
    let block_arrays: Vec<&Array> = all
        .iter()
        .filter(|(k, _)| block_index(k).is_some())
        .map(|(_, v)| v)
        .collect();
    for a in &block_arrays {
        eval([*a]).expect("eval all");
    }
    let (resident_active, _, resident_peak) = snap("all 30 blocks resident");

    println!("\n=== VERDICT ===");
    println!("  bounded (one block at a time) : peak {bounded_peak:>9.1} MiB");
    println!("  resident (all blocks)         : peak {resident_peak:>9.1} MiB (active {resident_active:.1})");
    if bounded_peak > 0.0 {
        println!(
            "  reduction factor              : {:.1}x",
            resident_active / bounded_peak.max(1.0)
        );
    }
    println!(
        "  RUNG 4 EXPRESSIBLE ON MLX     : {}",
        if bounded_peak < resident_active * 0.5 {
            "YES"
        } else {
            "NO"
        }
    );
}

/// The cost question the memory answer does not settle: a DiT runs N denoise steps, so a block-window
/// schedule RE-MATERIALIZES every block once per step. If that means re-reading 2.85 GiB from disk per
/// step, rung 4 buys memory at an unacceptable throughput price — and the machines that need it are
/// exactly the ones whose OS page cache is under pressure, so warm-cache numbers are the optimistic
/// bound, not the expected one.
#[test]
#[ignore = "set MLX_GEN_BLOCK_WEIGHTS to a packed-quantized transformer safetensors"]
fn rung4_rematerialization_cost_per_step() {
    let Some(path) = weights_path() else {
        println!("SKIP: set MLX_GEN_BLOCK_WEIGHTS to a packed-quantized transformer safetensors");
        return;
    };
    const STEPS: usize = 8; // z-image-turbo default

    let (mut tensors, _): (HashMap<String, Array>, HashMap<String, String>) =
        Array::load_safetensors_with_metadata(&path).expect("load");
    let mut blocks: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for key in tensors.keys() {
        if let Some(idx) = block_index(key) {
            blocks.entry(idx).or_default().push(key.clone());
        }
    }
    // Keep only block keys, owned per index, so each step can re-load from the file.
    let block_keys: Vec<Vec<String>> = blocks.values().cloned().collect();
    drop(std::mem::take(&mut tensors));
    clear_cache();

    println!(
        "\n=== re-materialization cost: {STEPS} steps x {} blocks ===",
        block_keys.len()
    );
    let mut step_times = Vec::new();
    for step in 0..STEPS {
        let t0 = std::time::Instant::now();
        // Each step re-opens the file and walks every block, exactly as a block-window schedule would.
        let (map, _): (HashMap<String, Array>, HashMap<String, String>) =
            Array::load_safetensors_with_metadata(&path).expect("reload");
        for keys in &block_keys {
            let held: Vec<&Array> = keys.iter().filter_map(|k| map.get(k)).collect();
            for a in &held {
                eval([*a]).expect("eval");
            }
            drop(held);
        }
        drop(map);
        clear_cache();
        let dt = t0.elapsed();
        step_times.push(dt.as_secs_f64());
        println!(
            "  step {step}: {:.3}s   (active after {:.1} MiB)",
            dt.as_secs_f64(),
            mib(get_active_memory())
        );
    }
    let total: f64 = step_times.iter().sum();
    let mean = total / STEPS as f64;
    println!("\n  mean per-step re-materialization: {mean:.3}s");
    println!("  total over {STEPS} steps: {total:.3}s");
    println!("  (compare: a resident DiT pays this ONCE at load)");
    println!(
        "  throughput verdict: {}",
        if mean < 0.5 {
            "cheap enough to schedule per step"
        } else {
            "EXPENSIVE — needs a block WINDOW, not per-block"
        }
    );
}
