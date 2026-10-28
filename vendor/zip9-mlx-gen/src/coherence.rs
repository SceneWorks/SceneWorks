//! GPU-view coherence of freshly loaded weight buffers (sc-22414).
//!
//! # The defect this closes
//!
//! A safetensors `Load` runs on the CPU stream: MLX allocates a Metal shared buffer and `pread`s
//! the file into it. Under a full page cache with dirty write-back pressure, the **GPU's view of
//! that buffer can lag the CPU's**: the first Metal kernel to read it sees exact zeros for pages
//! the CPU already holds correctly, and a later kernel sees the real bytes. Measured on Mac2
//! (Mac17,6, 128 GB, macOS 26.6.1; 6/6 reproductions, see SceneWorks
//! `docs/calibration/sc-18791/mac2-cold-load/RUNBOOK.md`): LTX-2.5's once-loaded 2 GB Gemma 4
//! `embed_tokens` table read as zeros on render 1's first pass while every CPU-side hash of the
//! same buffer was correct, every layer weight was correct at both views, and the same handle read
//! correctly minutes later. An all-zero embedding maps through RMSNorm, SDPA and the residual adds
//! to an all-zero stack, so the render is deterministic garbage rather than a crash — the
//! measured-vs-warm parity breach that blocked the SC-18791 campaign.
//!
//! Ruled out there by experiment: the OS version, the sampler, a CPU→GPU fence race (a
//! materialization barrier before the forward changed nothing), read errors and short reads in
//! MLX's reader, uncached reads alone, read churn alone, and user-space staging of the read (the
//! CPU view was never wrong). Time heals it: every variant that slowed the load down passed.
//!
//! # The fix
//!
//! Verify the GPU's view against the CPU's at the load boundary, before any graph consumes the
//! buffer, and wait for them to agree. [`byte_checksum`] is a **wrapping integer sum of the raw
//! bytes** read through a given stream: modular addition is order-independent, so a CPU reduction
//! and a GPU reduction over the same bytes are bit-identical regardless of how each backend
//! partitions the sum (a float sum would not be). [`verify_gpu_view`] evaluates each array on its
//! own stream, takes the CPU checksum as the reference, and re-reads the GPU checksum with a
//! bounded back-off until it matches. Exhausting the budget is a typed
//! [`Error::IncoherentLoad`], never a silent retry-forever and never a silent render.
//!
//! The check is a GPU read of every loaded byte plus a CPU read of the same — cheap against the
//! load itself, and the GPU read is the one that would otherwise have happened inside the first
//! forward.
//!
//! # Mirrored, not shared
//!
//! This is a deliberate twin of `mlx_llm::primitives::coherence`: `mlx-gen` does not depend on
//! `mlx-llm`, so — exactly like the two `Weights` types — the semantics are mirrored and the names
//! kept identical. [`Weights::materialize`](crate::weights::Weights::materialize) and
//! [`Weights::materialize_accessed`](crate::weights::Weights::materialize_accessed) are the two
//! seams, and both call [`verify_gpu_view`] — so a loader is guarded exactly when it materializes
//! through one of them. The LTX-2.5 render path does at every load (text encoder, connector, DiT
//! resident or streamed, VAEs, vocoder, upsamplers, duration head, adapters, enhancer), as do the
//! block-window streams (krea, z-image, qwen-image, flux2, sdxl, boogu). A provider that never
//! materializes — evaluating its checkpoint inside its first forward — is **not** guarded by this
//! module; guarding it means adding the `materialize` call at its load seam, which also changes
//! when its weights become resident, so that is a per-provider decision rather than a blanket one.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use mlx_rs::{Array, Dtype, StreamOrDevice};

use crate::error::{Error, Result};

/// GPU re-reads attempted before [`verify_gpu_view`] gives up on one array.
pub const MAX_RETRIES: u32 = 12;

/// Back-off before retry `attempt` (1-based), in milliseconds: 50 ms doubling to a 2 s ceiling —
/// about 15 s total across [`MAX_RETRIES`]. The Mac2 evidence put the healing time well under a
/// second. Kept as an integer so the schedule is a clock-free reading for its tests.
pub fn backoff_millis(attempt: u32) -> u64 {
    let ms = 50u64.saturating_mul(1u64 << attempt.saturating_sub(1).min(20));
    ms.min(2_000)
}

/// [`backoff_millis`] as the `Duration` the retry loop sleeps.
pub fn backoff(attempt: u32) -> Duration {
    Duration::from_millis(backoff_millis(attempt))
}

static RETRIES: AtomicU64 = AtomicU64::new(0);

/// Number of GPU re-reads this process has needed so far — the incidence counter. Zero on a
/// healthy host; every increment is one observed CPU/GPU view divergence.
pub fn retries() -> u64 {
    RETRIES.load(Ordering::Relaxed)
}

/// Wrapping `u32` sum of the raw bytes of `a`, read through `stream`.
///
/// The array is evaluated first (a lazy `Load` runs on its own stream either way). Sizes that are
/// not a multiple of four bytes widen a byte view to `u32` before summing so the checksum keeps
/// its full width; a wrapping `u8` sum would pass an all-zero read one time in 256.
pub fn byte_checksum(a: &Array, stream: StreamOrDevice) -> Result<u64> {
    a.eval()?;
    // A zero-element tensor has no bytes for either view to disagree on, and Metal has no
    // `init_reduce_sum` kernel for `uint32`, so an empty GPU reduction would *error* rather than
    // return 0 — turning a load that was fine into a refusal.
    if a.nbytes() == 0 {
        return Ok(0);
    }
    let flat = a.flatten_device(None, None, &stream)?;
    let words = if flat.nbytes() % 4 == 0 {
        flat.view_dtype_device(Dtype::Uint32, &stream)?
    } else {
        flat.view_dtype_device(Dtype::Uint8, &stream)?
            .as_dtype_device(Dtype::Uint32, &stream)?
    };
    let sum = words.sum_device(false, &stream)?;
    Ok(u64::from(sum.try_item::<u32>()?))
}

/// Verify that the GPU reads every one of `arrays` as the bytes the CPU holds, waiting for the
/// views to converge. `name` is only for the error.
///
/// A process whose MLX default device is the CPU has no GPU view to verify: its forwards read the
/// buffers on the CPU stream, and probing the GPU would put Metal work into a process that pinned
/// itself off it (the CPU-pinned test children, sc-25690).
pub fn verify_gpu_view<'a>(arrays: impl IntoIterator<Item = (&'a str, &'a Array)>) -> Result<()> {
    if mlx_rs::Device::try_default()? == mlx_rs::Device::cpu() {
        return Ok(());
    }
    verify_gpu_view_with(
        arrays,
        |a| byte_checksum(a, StreamOrDevice::gpu()),
        std::thread::sleep,
    )
}

/// [`verify_gpu_view`] with the GPU read and the wait injectable — the seam the tests drive a
/// divergence through, since a real one cannot be provoked on demand.
pub(crate) fn verify_gpu_view_with<'a>(
    arrays: impl IntoIterator<Item = (&'a str, &'a Array)>,
    mut gpu_checksum: impl FnMut(&Array) -> Result<u64>,
    mut sleep: impl FnMut(Duration),
) -> Result<()> {
    for (name, a) in arrays {
        // The reference. MLX's CPU reduce accumulates `uint32` through `int32_t` (wrapping in
        // practice; the 8 MiB agreement test below is what pins that across a pin bump).
        let cpu = byte_checksum(a, StreamOrDevice::cpu())?;
        let mut attempt = 0u32;
        loop {
            let gpu = gpu_checksum(a)?;
            if gpu == cpu {
                break;
            }
            attempt += 1;
            RETRIES.fetch_add(1, Ordering::Relaxed);
            if attempt > MAX_RETRIES {
                return Err(Error::IncoherentLoad {
                    name: name.to_string(),
                    bytes: a.nbytes(),
                    cpu,
                    gpu,
                    attempts: attempt,
                });
            }
            sleep(backoff(attempt));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            })
            .collect()
    }

    fn named(a: &Array) -> [(&str, &Array); 1] {
        [("t", a)]
    }

    /// The checksum is the same number from either stream, for every byte-size residue and for a
    /// non-byte dtype. MUTATION: replace the wrapping `u32` sum with a float sum and the
    /// large-array case goes RED (reduction order differs between backends).
    #[test]
    fn cpu_and_gpu_checksums_agree_for_every_size_residue() {
        for (n, seed) in [(4096usize, 1u64), (4097, 2), (4098, 3), (4099, 4), (1, 5)] {
            let bytes = random_bytes(n, seed);
            let a = Array::from_slice(&bytes, &[n as i32]);
            let cpu = byte_checksum(&a, StreamOrDevice::cpu()).unwrap();
            let gpu = byte_checksum(&a, StreamOrDevice::gpu()).unwrap();
            assert_eq!(cpu, gpu, "n={n}");
            assert_ne!(cpu, 0, "n={n}: a random payload must not checksum to zero");
        }
        // 2-D bf16, the shape of a real embedding table; 8 MiB so the GPU reduction is partitioned.
        let words: Vec<u32> = random_bytes(8 << 20, 9)
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let table = Array::from_slice(&words, &[2048, 1024])
            .view_dtype_device(Dtype::Bfloat16, StreamOrDevice::cpu())
            .unwrap();
        assert_eq!(table.shape(), &[2048, 2048]);
        let cpu = byte_checksum(&table, StreamOrDevice::cpu()).unwrap();
        let gpu = byte_checksum(&table, StreamOrDevice::gpu()).unwrap();
        assert_eq!(cpu, gpu);
        let expected = words.iter().fold(0u32, |acc, w| acc.wrapping_add(*w));
        assert_eq!(
            cpu,
            u64::from(expected),
            "checksum is the wrapping u32 word sum"
        );
    }

    /// A zeroed read is distinguishable from the data: the checksum of zeros is 0, the data's is
    /// not — and a partial zeroing changes it too.
    #[test]
    fn a_zeroed_or_partially_zeroed_view_changes_the_checksum() {
        let mut bytes = random_bytes(1 << 16, 11);
        let full = byte_checksum(
            &Array::from_slice(&bytes, &[bytes.len() as i32]),
            StreamOrDevice::gpu(),
        )
        .unwrap();
        for b in bytes.iter_mut().skip(4096).take(4096) {
            *b = 0;
        }
        let partial = byte_checksum(
            &Array::from_slice(&bytes, &[bytes.len() as i32]),
            StreamOrDevice::gpu(),
        )
        .unwrap();
        let zeros = byte_checksum(
            &Array::zeros::<u8>(&[1 << 16]).unwrap(),
            StreamOrDevice::gpu(),
        )
        .unwrap();
        assert_ne!(full, partial);
        assert_ne!(full, zeros);
        assert_eq!(zeros, 0);
    }

    /// A healthy array passes with no wait and no retry counted.
    #[test]
    fn a_coherent_array_passes_without_waiting() {
        let a = Array::from_slice(&random_bytes(1 << 12, 21), &[1 << 12]);
        let before = retries();
        let slept = RefCell::new(Vec::new());
        verify_gpu_view_with(
            named(&a),
            |a| byte_checksum(a, StreamOrDevice::gpu()),
            |d| slept.borrow_mut().push(d),
        )
        .unwrap();
        assert!(slept.borrow().is_empty());
        assert_eq!(retries(), before);
        verify_gpu_view(named(&a)).unwrap();
    }

    /// A GPU view that reads zeros for a while and then converges is waited for, not failed:
    /// the waits follow the back-off schedule and the incidence counter records each retry.
    /// MUTATION: drop the retry loop (return the error on the first mismatch) — RED.
    #[test]
    fn a_lagging_gpu_view_is_retried_until_it_converges() {
        let a = Array::from_slice(&random_bytes(1 << 12, 31), &[1 << 12]);
        let before = retries();
        let reads = RefCell::new(0u32);
        let slept = RefCell::new(Vec::new());
        verify_gpu_view_with(
            named(&a),
            |a| {
                *reads.borrow_mut() += 1;
                if *reads.borrow() <= 3 {
                    Ok(0)
                } else {
                    byte_checksum(a, StreamOrDevice::gpu())
                }
            },
            |d| slept.borrow_mut().push(d),
        )
        .unwrap();
        assert_eq!(*reads.borrow(), 4);
        assert_eq!(
            *slept.borrow(),
            vec![backoff(1), backoff(2), backoff(3)],
            "one back-off per divergent read"
        );
        assert_eq!(retries() - before, 3);
    }

    /// A GPU view that never converges is a typed, named error after exactly the budget — never
    /// a silent pass and never an unbounded wait. MUTATION: make the loop unbounded — the test
    /// hangs; make it return `Ok` on exhaustion — RED.
    #[test]
    fn a_gpu_view_that_never_converges_is_a_typed_error_after_the_budget() {
        let a = Array::from_slice(&random_bytes(1 << 12, 41), &[1 << 12]);
        let reads = RefCell::new(0u32);
        let slept = RefCell::new(Vec::new());
        let err = verify_gpu_view_with(
            [("model.embed_tokens.weight", &a)],
            |_| {
                *reads.borrow_mut() += 1;
                Ok(0)
            },
            |d| slept.borrow_mut().push(d),
        )
        .unwrap_err();
        assert_eq!(*reads.borrow(), MAX_RETRIES + 1);
        match err {
            Error::IncoherentLoad {
                name,
                bytes,
                cpu,
                gpu,
                attempts,
            } => {
                assert_eq!(name, "model.embed_tokens.weight");
                assert_eq!(bytes, 1 << 12);
                assert_ne!(cpu, 0);
                assert_eq!(gpu, 0);
                assert_eq!(attempts, MAX_RETRIES + 1);
            }
            other => panic!("expected IncoherentLoad, got {other:?}"),
        }
        let schedule: Vec<Duration> = (1..=MAX_RETRIES).map(backoff).collect();
        assert_eq!(
            *slept.borrow(),
            schedule,
            "every retry waited its scheduled back-off"
        );
        let budget_millis: u64 = (1..=MAX_RETRIES).map(backoff_millis).sum();
        assert!(
            (10_000..=30_000).contains(&budget_millis),
            "total wait budget {budget_millis} ms must stay in the 10–30 s band"
        );
    }

    /// A zero-element tensor checksums to 0 on both streams instead of erroring (Metal has no
    /// `uint32` init-reduce kernel). MUTATION: drop the `nbytes() == 0` early return — RED.
    #[test]
    fn a_zero_element_tensor_checksums_to_zero_on_both_streams() {
        for a in [
            Array::zeros::<f32>(&[0]).unwrap(),
            Array::zeros::<u8>(&[0]).unwrap(),
            Array::zeros::<f32>(&[4, 0, 8]).unwrap(),
        ] {
            assert_eq!(byte_checksum(&a, StreamOrDevice::gpu()).unwrap(), 0);
            assert_eq!(byte_checksum(&a, StreamOrDevice::cpu()).unwrap(), 0);
            verify_gpu_view(named(&a)).unwrap();
        }
    }

    /// The in-repo cost measurement AC3 asks for. Ignored: it allocates 1 GiB. Recorded on
    /// Mac17,6 (2026-09-02, release): CPU 18 GB/s cold / 68 GB/s warm, GPU 250–330 GB/s.
    /// Run: `cargo test --release --lib -p <crate> checksum_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "1 GiB allocation; throughput probe, run on demand"]
    fn checksum_throughput_1gib() {
        let n = 1usize << 30;
        let a = Array::ones::<u8>(&[n as i32]).unwrap();
        a.eval().unwrap();
        for (label, stream) in [
            ("cpu", StreamOrDevice::cpu as fn() -> StreamOrDevice),
            ("gpu", StreamOrDevice::gpu),
        ] {
            for pass in 0..2 {
                let t = std::time::Instant::now();
                let c = byte_checksum(&a, stream()).unwrap();
                let s = t.elapsed().as_secs_f64();
                eprintln!(
                    "{label} pass {pass}: checksum={c:#x} {s:.3}s {:.1} GB/s",
                    n as f64 / s / 1e9
                );
            }
        }
    }

    /// The schedule: doubling from 50 ms, capped at 2 s.
    #[test]
    fn backoff_doubles_from_50ms_and_caps_at_2s() {
        assert_eq!(backoff_millis(1), 50);
        assert_eq!(backoff_millis(2), 100);
        assert_eq!(backoff_millis(6), 1600);
        assert_eq!(backoff_millis(7), 2000);
        assert_eq!(backoff_millis(40), 2000);
    }
}
