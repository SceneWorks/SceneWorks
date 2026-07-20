//! Weightless CUDA quantized-matmul smoke for the SceneWorks workspace build (sc-7544 / sc-13510).
//!
//! candle-kernels compiles its GGUF `QMatMul` kernels (`mmq_gguf/*`, `moe/*`, `mmvq_gguf`) into a
//! static `libmoe.a` of SASS with no PTX. Un-patched, that archive holds a single
//! `CUDA_COMPUTE_CAP`-derived cubin; at the cap=80 packaging baseline a Blackwell (sm_120) GPU
//! then has no compatible code and nothing to JIT, so every quantized matmul **silently returns
//! zeros** (quantized models render solid black while dense models work). The fix is the root
//! Cargo.toml `[patch]` onto the inference repo's vendored multi-arch candle-kernels.
//!
//! The inference repo has carried this exact smoke since sc-7544 — but it guards the *inference*
//! workspace, and the regression (sc-13510) happened in the *SceneWorks* workspace build, which
//! resolved candle-kernels from upstream with nothing watching. This copy closes that gap: it
//! exercises the actual kernels linked into the worker's own graph, on the GPU, in the
//! windows-candle CI lane (`cargo test -p sceneworks-worker --features backend-candle` on the
//! self-hosted CUDA box). It is weightless (no checkpoints) and skips gracefully when no CUDA
//! device is available, so non-GPU builds stay green.
//!
//! Note the lane builds at the box's native cap=120, where even a single-arch build would pass —
//! the cap=80 *fatbin* guarantees are held by `candle_kernels_patch_guard.rs` (lockfile) and the
//! packaging-time `cuobjdump` check in apps/desktop/scripts/build-sidecar.mjs. This smoke proves
//! the linked quant kernels actually launch and compute correctly on real hardware.
#![cfg(all(not(target_os = "macos"), feature = "backend-candle"))]

use runtime_cuda::media::candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use runtime_cuda::media::candle_core::{Device, Module, Tensor};

/// Deterministic, launch-portable pseudo-random f32 in roughly [-1, 1] (splitmix64-style hash of
/// the index). Avoids a device RNG so the CPU reference and the CUDA result quantize
/// byte-identical data.
fn pseudo_random(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Cosine similarity of two tensors over all elements (flattened, on the CPU).
fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap();
    let b = b.flatten_all().unwrap();
    let dot = (&a * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let na = (&a * &a)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    let nb = (&b * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    dot / (na * nb).max(1e-12)
}

fn all_finite(t: &Tensor) -> bool {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite())
}

/// The GGUF Q4_0/Q8_0 `QMatMul` on the CUDA device matches the CPU reference (cos≈1, all-finite).
///
/// On the broken single-arch packaging the CUDA result is all-zeros/garbage (cos≈0) — this fails
/// loudly. With the multi-arch fatbin (native sm_120 cubin) it passes.
#[test]
fn cuda_qmatmul_matches_cpu() {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP cuda_qmatmul_matches_cpu: no CUDA device ({e})");
            return;
        }
    };
    eprintln!("[quant-smoke] device={device:?}");

    // out=N, in=K, rows=M. K is a multiple of 32 (Q4_0/Q8_0 block) and 256 (k-quant QK_K), so the
    // shapes stay valid for every GGUF dtype should we extend the sweep later.
    let (n, k, m) = (512usize, 1024usize, 8usize);
    let w_cpu = Tensor::from_vec(pseudo_random(n * k), (n, k), &Device::Cpu).expect("w");
    let x_cpu = Tensor::from_vec(pseudo_random(m * k), (m, k), &Device::Cpu).expect("x");

    // Q4 quantization-noise floor is wider; Q8 is near-lossless.
    for (dtype, min_cos, label) in [
        (GgmlDType::Q8_0, 0.999f32, "Q8_0"),
        (GgmlDType::Q4_0, 0.99f32, "Q4_0"),
    ] {
        // CPU reference: quantize + matmul entirely on the CPU.
        let mm_cpu = QMatMul::from_qtensor(QTensor::quantize(&w_cpu, dtype).expect("cpu quantize"))
            .expect("cpu qmatmul");
        let y_cpu = mm_cpu.forward(&x_cpu).expect("cpu forward");

        // CUDA: quantize the SAME cpu source straight onto the device, matmul on the device.
        let mm_cuda = QMatMul::from_qtensor(
            QTensor::quantize_onto(&w_cpu, dtype, &device).expect("cuda quantize_onto"),
        )
        .expect("cuda qmatmul");
        let x_cuda = x_cpu.to_device(&device).expect("x->cuda");
        let y_cuda = mm_cuda
            .forward(&x_cuda)
            .expect("cuda forward")
            .to_device(&Device::Cpu)
            .expect("y->cpu");

        let cos = cosine(&y_cpu, &y_cuda);
        let finite = all_finite(&y_cuda);
        eprintln!("[quant-smoke] {label}: cos(CUDA, CPU)={cos:.5} all_finite={finite}");

        assert!(
            finite,
            "{label} CUDA QMatMul produced non-finite values — likely no compatible cubin for \
             this arch (single-arch SASS build on a newer GPU). The root Cargo.toml [patch] onto \
             the vendored multi-arch candle-kernels is probably not in effect (sc-13510)."
        );
        assert!(
            cos > min_cos,
            "{label} CUDA QMatMul does not match the CPU reference (cos={cos:.5} <= {min_cos}). \
             On Blackwell sm_120 this means candle-kernels' libmoe.a has no native sm_120 cubin \
             (the quant kernels silently no-op). The root Cargo.toml [patch] onto the vendored \
             multi-arch candle-kernels is probably not in effect (sc-13510)."
        );
    }
}
