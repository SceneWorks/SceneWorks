//! MLX unified-memory pre-load fit-gate (epic 10834 Phase 0, sc-10835).
//!
//! The unified-memory sibling of the candle `vram_gate.rs` (epic 10765, sc-10766). Before an MLX
//! generation loads, predict the model's whole-model peak footprint and compare it against the
//! machine's unified-memory budget, so a Mac that cannot fit the model is rejected with an
//! actionable message instead of being SIGKILL'd / Metal-OOM'd mid-render. Also honors
//! [`MLX_MEMORY_CAP_ENV`], which emulates a smaller Mac on big hardware so the Phase 1 sequential
//! residency work (sc-10839) can be validated on the dev box's 128 GB machine without 16/32 GB
//! hardware.
//!
//! ## Why this budgets on PREDICTED (on-disk) bytes, not live allocator deltas
//! MLX materializes weights lazily on first forward, so `get_active_memory()` reads ~0 right after
//! `load` — a post-load delta would see nothing. And a wired-memory overcommit SIGKILLs the process
//! rather than returning a catchable error, so we cannot "load and catch the OOM." This is therefore
//! a pre-load ADMISSION check keyed off the summed on-disk component weight bytes plus a fixed
//! headroom, never a post-allocation accounting number — the same conclusion the candle gate reached
//! for a different (caching-allocator) reason.
//!
//! Generalizes the per-model `flux2_dev_edit_memory_guard` (`image_jobs/flux2.rs`): that one gates a
//! single activation-bound edit path; this gates the base weight-fit for every MLX image model.
//!
//! The pure decision logic is cross-platform and unit-tested on every lane; only the live
//! `sysctl hw.memsize` probe is macOS-only (it returns `None` elsewhere, so the gate no-ops).

use std::path::Path;
use std::sync::OnceLock;

use crate::{WorkerError, WorkerResult};

/// Emulate a smaller Mac: force the total-unified-memory budget (GB). Set e.g.
/// `SCENEWORKS_MLX_MEMORY_CAP_GB=16` to make the gate treat this machine as a 16 GB Mac, so a model
/// that would OOM there is rejected (and, once Phase 1 lands, run under sequential residency) exactly
/// as on real small hardware. Unset / non-positive ⇒ use the real `sysctl hw.memsize` total.
pub(crate) const MLX_MEMORY_CAP_ENV: &str = "SCENEWORKS_MLX_MEMORY_CAP_GB";

/// Headroom (GB) added on top of the summed on-disk component weights to cover MLX Metal activation
/// spikes during denoise/decode plus the OS + other apps drawing from the same unified pool.
///
/// PROVISIONAL — calibrate against `get_peak_memory` telemetry per the sc-10835 measurement
/// sub-task. Anchor: the candle FLUX A/B (sc-10769) measured ~11 GB of transient over ~33 GB of
/// resident weights at 1024²; a Mac additionally shares the pool with the OS + app. 10 GB is a
/// deliberately conservative floor that keeps the gate from over-rejecting borderline models while
/// still catching the genuinely-too-big ones; the real activation term scales with resolution and is
/// refined once measured.
const HEADROOM_GB: f64 = 10.0;

/// Bytes per binary gigabyte (GiB) — matches `gpu::total_unified_memory_gb`, which divides
/// `hw.memsize` by 1024³, and the epic's measured on-disk table.
const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// A usable unified-memory budget for the machine, in GB. Single field (no free/total split): on
/// unified memory the whole pool is the budget, and current pressure is absorbed by [`HEADROOM_GB`]
/// rather than a live "free" reading that fluctuates with the OS.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MemoryBudget {
    pub total_gb: f64,
}

/// The gate outcome. `Unknown` = no signal (the RAM probe failed, or the weights dir had no
/// measurable `.safetensors`) ⇒ never block, mirroring the flux2 guard's `available_gb == None`
/// short-circuit and the candle gate's `Unknown`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FitDecision {
    Unknown,
    Fits,
    TooBig { needed_gb: f64, available_gb: f64 },
}

/// Read the small-Mac cap from the environment. `Some(gb)` only for a positive number.
pub(crate) fn mlx_memory_cap_gb() -> Option<f64> {
    parse_memory_cap(std::env::var(MLX_MEMORY_CAP_ENV).ok().as_deref())
}

/// Parse the cap value: a positive, finite float (GB), else `None`.
pub(crate) fn parse_memory_cap(raw: Option<&str>) -> Option<f64> {
    let value = raw?.trim().parse::<f64>().ok()?;
    (value.is_finite() && value > 0.0).then_some(value)
}

/// Resolve the budget: the emulation cap overrides the real probed total (emulating a smaller Mac);
/// otherwise the real total. `None` from both ⇒ no budget ⇒ the gate no-ops (`Unknown`).
pub(crate) fn resolve_budget(real_total_gb: Option<f64>, cap: Option<f64>) -> Option<MemoryBudget> {
    cap.or(real_total_gb)
        .map(|total_gb| MemoryBudget { total_gb })
}

/// Predicted whole-model peak (GiB) = summed component weight bytes + [`HEADROOM_GB`]. `None` when
/// `weight_bytes == 0` (nothing measured ⇒ no signal ⇒ never block).
pub(crate) fn predicted_peak_gb(weight_bytes: u64) -> Option<f64> {
    (weight_bytes > 0).then(|| weight_bytes as f64 / BYTES_PER_GIB + HEADROOM_GB)
}

/// Decide whether the predicted peak fits the budget. Missing either input ⇒ `Unknown` (never
/// block), exactly like the flux2 guard and the candle gate.
pub(crate) fn fit_decision(needed_gb: Option<f64>, budget: Option<MemoryBudget>) -> FitDecision {
    let (Some(needed_gb), Some(budget)) = (needed_gb, budget) else {
        return FitDecision::Unknown;
    };
    if budget.total_gb + f64::EPSILON < needed_gb {
        FitDecision::TooBig {
            needed_gb,
            available_gb: budget.total_gb,
        }
    } else {
        FitDecision::Fits
    }
}

/// Sum the on-disk bytes of every `.safetensors` weight file under `dir` (recursively), following
/// symlinks (the HF cache stores each shard as a symlink into `blobs/`). AppleDouble `._*` sidecars
/// are skipped — they masquerade as `.safetensors` and would double-count (and corrupt globs, per
/// the AppleDouble sidecar gotcha). Returns 0 if the directory is missing or holds no weights, which
/// the gate treats as "no signal".
pub(crate) fn sum_safetensors_bytes(dir: &Path) -> u64 {
    fn walk(dir: &Path, total: &mut u64) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            // `metadata()` follows symlinks (HF blobs); `file_type()` on the DirEntry does not, so
            // resolve the target kind via `metadata` for symlinked shard files.
            let Ok(meta) = std::fs::metadata(&path) else {
                continue;
            };
            if meta.is_dir() {
                walk(&path, total);
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".safetensors") && !name.starts_with("._") {
                *total += meta.len();
            }
        }
    }
    let mut total = 0;
    walk(dir, &mut total);
    total
}

/// Total unified memory (GiB) via a blocking `sysctl hw.memsize`, cached process-wide (physical RAM
/// never changes at runtime). The blocking sibling of `gpu::total_unified_memory_gb` — the gate runs
/// on the generator-cache thread, which is already blocking on the weight load, so a one-shot
/// subprocess probe there is free. `None` off macOS or when the probe fails ⇒ the gate no-ops (a
/// cached `None` is a deliberate fail-open, consistent with `Unknown` never blocking).
fn probe_total_unified_memory_gib() -> Option<f64> {
    static TOTAL_GIB: OnceLock<Option<f64>> = OnceLock::new();
    *TOTAL_GIB.get_or_init(sysctl_total_unified_memory_gib)
}

#[cfg(target_os = "macos")]
fn sysctl_total_unified_memory_gib() -> Option<f64> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let bytes: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    Some(bytes as f64 / BYTES_PER_GIB)
}

#[cfg(not(target_os = "macos"))]
fn sysctl_total_unified_memory_gib() -> Option<f64> {
    None
}

/// Pre-load admission gate: reject if the model's predicted peak won't fit the machine's
/// unified-memory budget (or the [`MLX_MEMORY_CAP_ENV`] emulated budget). Called on the generator
/// cache's cold-load path, before `gen_core::load` allocates — never on a warm cache hit, so an
/// already-resident model is never re-gated. Any missing signal (probe failed, no cap, unmeasurable
/// weights) ⇒ `Ok`, so the gate never blocks a machine that can actually fit the model.
///
/// `model_label` is a human-facing model name for the rejection message (the engine id).
pub(crate) fn ensure_fits(weight_bytes: u64, model_label: &str) -> WorkerResult<()> {
    let budget = resolve_budget(probe_total_unified_memory_gib(), mlx_memory_cap_gb());
    fit_result(weight_bytes, budget, model_label)
}

/// Map a fit decision to a worker result: `TooBig` → an actionable [`WorkerError::InvalidPayload`],
/// else `Ok`. Split from [`ensure_fits`] so the rejection message is testable without the live probe
/// or the `MLX_MEMORY_CAP_ENV` global.
fn fit_result(
    weight_bytes: u64,
    budget: Option<MemoryBudget>,
    model_label: &str,
) -> WorkerResult<()> {
    match fit_decision(predicted_peak_gb(weight_bytes), budget) {
        FitDecision::TooBig {
            needed_gb,
            available_gb,
        } => Err(WorkerError::InvalidPayload(format!(
            "{model_label} needs ~{needed} GB of unified memory (model weights plus headroom for \
             activations and the OS) but this machine has ~{available} GB. Choose a smaller quant \
             tier, lower the output resolution, or run on a Mac with more memory.",
            needed = needed_gb.round() as i64,
            available = available_gb.round() as i64,
        ))),
        FitDecision::Fits | FitDecision::Unknown => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_memory_cap_accepts_positive_numbers_only() {
        assert_eq!(parse_memory_cap(Some("16")), Some(16.0));
        assert_eq!(parse_memory_cap(Some("  32.5 ")), Some(32.5));
        assert_eq!(parse_memory_cap(Some("0")), None);
        assert_eq!(parse_memory_cap(Some("-8")), None);
        assert_eq!(parse_memory_cap(Some("nan")), None);
        assert_eq!(parse_memory_cap(Some("inf")), None);
        assert_eq!(parse_memory_cap(Some("abc")), None);
        assert_eq!(parse_memory_cap(Some("")), None);
        assert_eq!(parse_memory_cap(None), None);
    }

    #[test]
    fn resolve_budget_prefers_the_emulation_cap() {
        // Cap overrides the real total (emulate a smaller Mac on big hardware).
        assert_eq!(
            resolve_budget(Some(128.0), Some(16.0)),
            Some(MemoryBudget { total_gb: 16.0 })
        );
        // No cap ⇒ the real total.
        assert_eq!(
            resolve_budget(Some(128.0), None),
            Some(MemoryBudget { total_gb: 128.0 })
        );
        // A cap with no real reading still yields a budget (exercisable in a no-probe unit test).
        assert_eq!(
            resolve_budget(None, Some(16.0)),
            Some(MemoryBudget { total_gb: 16.0 })
        );
        // No signal at all ⇒ no budget ⇒ gate no-ops.
        assert_eq!(resolve_budget(None, None), None);
    }

    #[test]
    fn predicted_peak_is_weights_plus_headroom_and_zero_is_no_signal() {
        // 20 GiB of weights ⇒ 20 + headroom.
        let bytes = 20 * 1024 * 1024 * 1024_u64;
        assert_eq!(predicted_peak_gb(bytes), Some(20.0 + HEADROOM_GB));
        // No measurable weights ⇒ no signal.
        assert_eq!(predicted_peak_gb(0), None);
    }

    #[test]
    fn fit_decision_rejects_only_a_genuine_overflow() {
        let budget = MemoryBudget { total_gb: 16.0 };
        // qwen-image q8: ~36 GiB weights + 10 headroom = 46 ⇒ too big for a 16 GB Mac.
        assert_eq!(
            fit_decision(predicted_peak_gb(36 * 1024 * 1024 * 1024_u64), Some(budget)),
            FitDecision::TooBig {
                needed_gb: 46.0,
                available_gb: 16.0,
            }
        );
        // A ~3 GiB model fits.
        assert_eq!(
            fit_decision(predicted_peak_gb(3 * 1024 * 1024 * 1024_u64), Some(budget)),
            FitDecision::Fits
        );
        // Exactly-fits is not a rejection: budget 46, need 46.
        assert_eq!(
            fit_decision(Some(46.0), Some(MemoryBudget { total_gb: 46.0 })),
            FitDecision::Fits
        );
        // Missing either input ⇒ never block.
        assert_eq!(fit_decision(None, Some(budget)), FitDecision::Unknown);
        assert_eq!(fit_decision(Some(8.0), None), FitDecision::Unknown);
    }

    #[test]
    fn sum_safetensors_skips_appledouble_and_nonweights_and_recurses() {
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_sum_{}_{}",
            std::process::id(),
            line!()
        ));
        let te = root.join("text_encoder");
        let dit = root.join("transformer");
        std::fs::create_dir_all(&te).expect("mk te");
        std::fs::create_dir_all(&dit).expect("mk dit");
        std::fs::write(te.join("model.safetensors"), vec![0u8; 1000]).expect("te weights");
        std::fs::write(dit.join("diffusion.safetensors"), vec![0u8; 2000]).expect("dit weights");
        // AppleDouble sidecar + a non-weight file must NOT be counted.
        std::fs::write(te.join("._model.safetensors"), vec![0u8; 500]).expect("sidecar");
        std::fs::write(dit.join("config.json"), vec![0u8; 700]).expect("config");

        assert_eq!(sum_safetensors_bytes(&root), 3000);
        // Missing dir ⇒ 0 (no signal).
        assert_eq!(sum_safetensors_bytes(&root.join("nope")), 0);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn fit_result_rejects_over_budget_with_an_actionable_message() {
        // qwen-image q8 ≈ 36 GiB of weights ⇒ predicted ~46 GB; a 16 GB Mac can't fit it.
        let bytes = 36 * 1024 * 1024 * 1024_u64;
        let error = fit_result(bytes, Some(MemoryBudget { total_gb: 16.0 }), "qwen-image")
            .expect_err("an over-budget model must be rejected");
        let WorkerError::InvalidPayload(message) = error else {
            panic!("expected InvalidPayload, got {error:?}");
        };
        assert!(message.contains("qwen-image"), "names the model: {message}");
        assert!(
            message.contains("unified memory"),
            "explains the limit: {message}"
        );
        assert!(message.contains("46"), "states what it needs: {message}");
        assert!(message.contains("16"), "states the budget: {message}");
    }

    #[test]
    fn fit_result_allows_a_fit_and_never_blocks_without_signal() {
        let bytes = 36 * 1024 * 1024 * 1024_u64;
        // Fits a 128 GB Mac.
        assert!(fit_result(bytes, Some(MemoryBudget { total_gb: 128.0 }), "qwen-image").is_ok());
        // No budget signal ⇒ never block, even a huge model.
        assert!(fit_result(999 * 1024 * 1024 * 1024_u64, None, "qwen-image").is_ok());
    }

    // HF cache stores each shard as a symlink into `blobs/`; the gate must follow those to the real
    // byte size. The synthetic test above uses plain files, so exercise the symlink layout here.
    #[cfg(unix)]
    #[test]
    fn sum_safetensors_follows_hf_cache_symlinks() {
        use std::os::unix::fs::symlink;
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_symlink_{}_{}",
            std::process::id(),
            line!()
        ));
        let blobs = root.join("blobs");
        let snap = root.join("snapshots/hash/transformer");
        std::fs::create_dir_all(&blobs).expect("mk blobs");
        std::fs::create_dir_all(&snap).expect("mk snap");
        std::fs::write(blobs.join("weightblob"), vec![0u8; 4096]).expect("weight blob");
        std::fs::write(blobs.join("sidecarblob"), vec![0u8; 500]).expect("sidecar blob");
        symlink(blobs.join("weightblob"), snap.join("diffusion.safetensors")).expect("weight link");
        symlink(
            blobs.join("sidecarblob"),
            snap.join("._diffusion.safetensors"),
        )
        .expect("sidecar link");

        // Summing the SNAPSHOT dir follows the symlink to the 4096-byte blob and skips the `._`
        // sidecar; the `blobs/` dir is not under the snapshot, so nothing is double-counted.
        assert_eq!(sum_safetensors_bytes(&root.join("snapshots/hash")), 4096);

        std::fs::remove_dir_all(&root).ok();
    }
}
