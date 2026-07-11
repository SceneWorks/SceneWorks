//! MLX unified-memory pre-load fit-gate (epic 10834 Phase 0 sc-10835; Phase 1 residency selection
//! sc-10839).
//!
//! The unified-memory sibling of the candle `vram_gate.rs` (epic 10765, sc-10766/sc-10821). Before an
//! MLX generation loads, predict the model's whole-model peak footprint and compare it against the
//! machine's unified-memory budget. Three outcomes, mirroring the candle gate:
//!  - **Fits** (or no signal) → load resident (the warm, cross-job path).
//!  - **Won't fit resident, but the provider supports sequential component residency (sc-10839) and
//!    the staged max-single-component peak WILL fit** → select [`OffloadPolicy::Sequential`] so the
//!    engine drops the text encoder(s) before the DiT loads, bounding peak to the largest component.
//!  - **Won't fit even staged** → reject with an actionable message instead of a SIGKILL / Metal-OOM
//!    mid-render.
//!
//! Also honors [`MLX_MEMORY_CAP_ENV`], which emulates a smaller Mac on big hardware so the sequential
//! residency selection can be validated on the dev box's 128 GB machine without 16/32 GB hardware.
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

use gen_core::{LoadSpec, OffloadPolicy, WeightsSource};

use crate::{WorkerError, WorkerResult};

/// The MLX engine ids whose provider honors [`OffloadPolicy::Sequential`] — the sc-10839 Phase 1
/// providers (SDXL + Z-Image-turbo). The gate only SELECTS sequential residency for these; for any
/// other engine `Sequential` would be a no-op (the advisory contract treats it as `Resident`), so
/// predicting "it fits staged" and then holding everything resident would SIGKILL — hence the
/// allowlist. Extended per family as the fan-out (sc-10840) wires each engine.
const SEQUENTIAL_CAPABLE_ENGINES: &[&str] = &["sdxl", "z_image_turbo"];

/// Whether `engine_id`'s provider drops components in phase order under [`OffloadPolicy::Sequential`].
pub(crate) fn engine_supports_sequential(engine_id: &str) -> bool {
    SEQUENTIAL_CAPABLE_ENGINES.contains(&engine_id)
}

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
    /// The predicted RESIDENT peak won't fit, but the provider supports sequential component residency
    /// (sc-10839) — run with [`OffloadPolicy::Sequential`] instead of rejecting. Sequential peak is
    /// always ≤ resident. The caller then runs the second-stage [`sequential_overflow_gb`] check and
    /// rejects only if even the staged max-single-component peak won't fit.
    Offload {
        needed_gb: f64,
        available_gb: f64,
    },
    TooBig {
        needed_gb: f64,
        available_gb: f64,
    },
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

/// Predicted SEQUENTIAL peak (GiB) = the largest single working set + [`HEADROOM_GB`] (sc-10839). The
/// `Sequential` schedule drops the text encoder(s) before the DiT loads and keeps the tiny VAE
/// co-resident with the DiT, so the peak is `max(text-encoders, everything-else) + headroom` rather
/// than the resident sum. `everything-else = total − text_encoders` (the DiT + VAE + any control/IP).
/// `None` when nothing was measured (`total == 0`). When the text encoders are unmeasured
/// (`te_bytes == 0`) this equals the resident peak — no claimed saving, so the second-stage overflow
/// check then rejects exactly as the resident gate would (the safe fallback).
pub(crate) fn predicted_sequential_peak_gb(total_bytes: u64, te_bytes: u64) -> Option<f64> {
    if total_bytes == 0 {
        return None;
    }
    let rest_bytes = total_bytes.saturating_sub(te_bytes);
    let staged_max = te_bytes.max(rest_bytes);
    Some(staged_max as f64 / BYTES_PER_GIB + HEADROOM_GB)
}

/// Fold sequential-residency capability into a fit decision (sc-10839, mirroring the candle
/// `vram_gate::resolve_offload`): a [`FitDecision::TooBig`] on a provider that drops components in
/// phase order becomes [`FitDecision::Offload`] — load→use→drop so peak is the largest single
/// component (≤ resident) rather than a reject. Every other outcome (Fits / Unknown / TooBig on a
/// non-capable provider) is unchanged.
pub(crate) fn resolve_offload(decision: FitDecision, sequential_capable: bool) -> FitDecision {
    match decision {
        FitDecision::TooBig {
            needed_gb,
            available_gb,
        } if sequential_capable => FitDecision::Offload {
            needed_gb,
            available_gb,
        },
        other => other,
    }
}

/// Second-stage gate for a [`FitDecision::Offload`] (sc-10839): sequential residency was selected
/// because the RESIDENT peak won't fit, on the promise that the staged working set will. If the
/// predicted staged peak ([`predicted_sequential_peak_gb`]) STILL exceeds the budget, return
/// `Some(needed_gb)` so the caller rejects before load with an actionable message instead of a
/// reactive Metal-OOM / SIGKILL. `None` (staged fits, or no budget) keeps the sequential run. Unlike
/// the candle gate — where the sequential peak is only sometimes measured — the MLX staged peak is
/// always derivable from the on-disk component split, so this check always applies.
pub(crate) fn sequential_overflow_gb(
    sequential_needed_gb: Option<f64>,
    budget: Option<MemoryBudget>,
) -> Option<f64> {
    let (needed_gb, budget) = (sequential_needed_gb?, budget?);
    (budget.total_gb + f64::EPSILON < needed_gb).then_some(needed_gb)
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

/// Sum the on-disk `.safetensors` bytes of the model's TEXT-ENCODER component(s) — the phase-A
/// component the `Sequential` residency drops before the DiT loads (sc-10839). Matches the diffusers
/// snapshot's top-level `text_encoder` / `text_encoder_2` / `text_encoder_*` subdirs (SDXL has both
/// CLIP encoders; Z-Image the single Qwen encoder), reusing [`sum_safetensors_bytes`] per subdir so
/// the HF-cache symlink + AppleDouble handling is shared. `0` if the dir is missing or has no
/// recognizable text-encoder subdir — which makes the staged estimate fall back to the resident sum
/// (no claimed saving), the safe direction.
pub(crate) fn sum_text_encoder_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut total = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !std::fs::metadata(&path).is_ok_and(|meta| meta.is_dir()) {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == "text_encoder" || name.starts_with("text_encoder_") {
            total += sum_safetensors_bytes(&path);
        }
    }
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

/// The residency-selection outcome (sc-10839) — the pure decision, split from the [`LoadSpec`]/IO so
/// the whole three-way selection is deterministically unit-testable without the live probe or the
/// `MLX_MEMORY_CAP_ENV` global. See [`decide_residency`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ResidencyOutcome {
    /// Fits resident (or no signal) — load with everything co-resident (the warm cross-job path).
    Resident,
    /// Won't fit resident but the provider stages components and the staged peak fits — load with
    /// [`OffloadPolicy::Sequential`].
    Sequential,
    /// Won't fit even staged (or the provider can't stage) — reject. `staged_gb` is `Some` when the
    /// staged path was attempted and still overflows (so the message can name it).
    Reject {
        needed_gb: f64,
        available_gb: f64,
        staged_gb: Option<f64>,
    },
}

/// The pure residency decision: given the model's whole-model + text-encoder on-disk bytes, the
/// (possibly emulated) budget, and whether the provider stages components, choose Resident /
/// Sequential / Reject (sc-10839). No IO, no globals — the live [`apply_residency_policy`] resolves
/// those and delegates here.
pub(crate) fn decide_residency(
    total_bytes: u64,
    te_bytes: u64,
    budget: Option<MemoryBudget>,
    sequential_capable: bool,
) -> ResidencyOutcome {
    let resident = fit_decision(predicted_peak_gb(total_bytes), budget);
    match resolve_offload(resident, sequential_capable) {
        FitDecision::Fits | FitDecision::Unknown => ResidencyOutcome::Resident,
        FitDecision::Offload {
            needed_gb,
            available_gb,
        } => {
            // Second stage: reject if even the staged max-single-component peak won't fit.
            let staged = predicted_sequential_peak_gb(total_bytes, te_bytes);
            match sequential_overflow_gb(staged, budget) {
                Some(_) => ResidencyOutcome::Reject {
                    needed_gb,
                    available_gb,
                    staged_gb: staged,
                },
                None => ResidencyOutcome::Sequential,
            }
        }
        FitDecision::TooBig {
            needed_gb,
            available_gb,
        } => ResidencyOutcome::Reject {
            needed_gb,
            available_gb,
            staged_gb: None,
        },
    }
}

/// Pre-load admission + residency-selection gate (sc-10835 Phase 0, sc-10839 Phase 1). Called on the
/// generator cache's cold-load path, before `gen_core::load` allocates — never on a warm cache hit,
/// so an already-resident model is never re-gated. Resolves the budget + on-disk component bytes,
/// delegates the choice to [`decide_residency`], and returns the [`LoadSpec`] to load with:
///  - fits resident (or no signal / unmeasurable weights) ⇒ the spec unchanged (warm resident path);
///  - won't fit resident but the provider stages components and the staged peak fits ⇒ the spec with
///    [`OffloadPolicy::Sequential`] set (drop the text encoder(s) before the DiT loads);
///  - won't fit even staged ⇒ [`WorkerError::InvalidPayload`] with an actionable message.
///
/// `engine_id` is both the [`engine_supports_sequential`] key and the human-facing model name in the
/// rejection message.
pub(crate) fn apply_residency_policy(spec: LoadSpec, engine_id: &str) -> WorkerResult<LoadSpec> {
    // Respect an offload policy already chosen upstream (defensive: the MLX cache seam normally sees
    // the default `Resident`, but never downgrade a `Sequential` set by another gate).
    if spec.offload_policy == OffloadPolicy::Sequential {
        return Ok(spec);
    }
    let budget = resolve_budget(probe_total_unified_memory_gib(), mlx_memory_cap_gb());
    let (total_bytes, te_bytes) = match &spec.weights {
        WeightsSource::Dir(dir) => (sum_safetensors_bytes(dir), sum_text_encoder_bytes(dir)),
        // A single-file source has no component split to stage — resident-or-reject only.
        WeightsSource::File(file) => (std::fs::metadata(file).map_or(0, |meta| meta.len()), 0),
    };
    match decide_residency(
        total_bytes,
        te_bytes,
        budget,
        engine_supports_sequential(engine_id),
    ) {
        ResidencyOutcome::Resident => Ok(spec),
        ResidencyOutcome::Sequential => {
            tracing::info!(
                event = "mlx_sequential_residency_selected",
                engine = %engine_id,
                total_gb = (total_bytes as f64 / BYTES_PER_GIB).round() as i64,
                text_encoder_gb = (te_bytes as f64 / BYTES_PER_GIB).round() as i64,
                budget_gb = budget.map(|b| b.total_gb.round() as i64),
            );
            Ok(spec.with_offload_policy(OffloadPolicy::Sequential))
        }
        ResidencyOutcome::Reject {
            needed_gb,
            available_gb,
            staged_gb,
        } => Err(too_big_error(engine_id, needed_gb, available_gb, staged_gb)),
    }
}

/// Build the actionable over-budget rejection. `staged_gb` is `Some` when sequential residency was
/// tried and its staged peak still overflows (so the message names both the resident and the staged
/// requirement — telling the user even one-component-at-a-time won't fit); `None` for a plain resident
/// reject on a non-staging provider. Split out so the message is testable without the live probe.
fn too_big_error(
    model_label: &str,
    needed_gb: f64,
    available_gb: f64,
    staged_gb: Option<f64>,
) -> WorkerError {
    let staged_note = match staged_gb {
        Some(staged) => format!(
            " (~{} GB even loading one component at a time)",
            staged.round() as i64
        ),
        None => String::new(),
    };
    WorkerError::InvalidPayload(format!(
        "{model_label} needs ~{needed} GB of unified memory{staged_note} (model weights plus \
         headroom for activations and the OS) but this machine has ~{available} GB. Choose a \
         smaller quant tier, lower the output resolution, or run on a Mac with more memory.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
    ))
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
    fn engine_supports_sequential_is_the_wired_provider_allowlist() {
        assert!(engine_supports_sequential("sdxl"));
        assert!(engine_supports_sequential("z_image_turbo"));
        // Not-yet-wired providers must NOT be offered sequential (they'd ignore it and SIGKILL).
        assert!(!engine_supports_sequential("qwen-image"));
        assert!(!engine_supports_sequential("z_image_turbo_control"));
        assert!(!engine_supports_sequential("flux"));
    }

    #[test]
    fn predicted_sequential_peak_is_largest_component_plus_headroom() {
        let gib = 1024 * 1024 * 1024_u64;
        // illustrious q8-class: total ~5 GiB, text encoders ~1 GiB ⇒ staged = max(1, 4) + headroom.
        let total = 5 * gib;
        let te = gib;
        assert_eq!(
            predicted_sequential_peak_gb(total, te),
            Some(4.0 + HEADROOM_GB)
        );
        // TE-dominant (lens-class): total 17, TE 13 ⇒ staged = max(13, 4) + headroom = 13 + headroom.
        assert_eq!(
            predicted_sequential_peak_gb(17 * gib, 13 * gib),
            Some(13.0 + HEADROOM_GB)
        );
        // Unmeasured text encoders ⇒ staged == resident sum (no claimed saving), the safe fallback.
        assert_eq!(
            predicted_sequential_peak_gb(20 * gib, 0),
            Some(20.0 + HEADROOM_GB)
        );
        // Nothing measured ⇒ no signal.
        assert_eq!(predicted_sequential_peak_gb(0, 0), None);
    }

    #[test]
    fn resolve_offload_rewrites_toobig_only_when_capable() {
        let too_big = FitDecision::TooBig {
            needed_gb: 46.0,
            available_gb: 16.0,
        };
        // Sequential-capable provider ⇒ Offload (carrying the resident numbers).
        assert_eq!(
            resolve_offload(too_big.clone(), true),
            FitDecision::Offload {
                needed_gb: 46.0,
                available_gb: 16.0,
            }
        );
        // Non-capable ⇒ still a reject.
        assert!(matches!(
            resolve_offload(too_big, false),
            FitDecision::TooBig { .. }
        ));
        // Fits / Unknown are never rewritten.
        assert_eq!(resolve_offload(FitDecision::Fits, true), FitDecision::Fits);
        assert_eq!(
            resolve_offload(FitDecision::Unknown, true),
            FitDecision::Unknown
        );
    }

    #[test]
    fn sequential_overflow_rejects_only_a_genuine_staged_overflow() {
        let budget = Some(MemoryBudget { total_gb: 16.0 });
        // Staged still needs 20 > 16 ⇒ reject even sequentially.
        assert_eq!(sequential_overflow_gb(Some(20.0), budget), Some(20.0));
        // Staged fits (14 <= 16) ⇒ run sequentially, no reject.
        assert_eq!(sequential_overflow_gb(Some(14.0), budget), None);
        // Exactly-fits is not an overflow.
        assert_eq!(sequential_overflow_gb(Some(16.0), budget), None);
        // No staged estimate or no budget ⇒ best-effort run (no reject).
        assert_eq!(sequential_overflow_gb(None, budget), None);
        assert_eq!(sequential_overflow_gb(Some(20.0), None), None);
    }

    #[test]
    fn too_big_error_names_model_budget_and_optional_staged() {
        // Plain resident reject (non-staging provider): no staged note.
        let WorkerError::InvalidPayload(resident) = too_big_error("qwen-image", 46.0, 16.0, None)
        else {
            panic!("expected InvalidPayload");
        };
        assert!(
            resident.contains("qwen-image"),
            "names the model: {resident}"
        );
        assert!(resident.contains("unified memory"), "explains: {resident}");
        assert!(resident.contains("46"), "states what it needs: {resident}");
        assert!(resident.contains("16"), "states the budget: {resident}");
        assert!(
            !resident.contains("one component at a time"),
            "no staged note when not attempted: {resident}"
        );
        // Staged reject: the message also names the one-at-a-time requirement.
        let WorkerError::InvalidPayload(staged) = too_big_error("sdxl", 46.0, 16.0, Some(24.0))
        else {
            panic!("expected InvalidPayload");
        };
        assert!(
            staged.contains("one component at a time"),
            "names the staged path: {staged}"
        );
        assert!(
            staged.contains("24"),
            "states the staged requirement: {staged}"
        );
    }

    #[test]
    fn decide_residency_picks_resident_sequential_or_reject_by_budget() {
        let gib = 1024 * 1024 * 1024_u64;
        // illustrious q8-class: total ~5 GiB (TE ~1, DiT+VAE ~4). Resident peak = 5+10 = 15;
        // staged peak = max(1, 4)+10 = 14.
        let total = 5 * gib;
        let te = gib;

        // Roomy budget (128 GB Mac) ⇒ Resident (keep the warm path).
        assert_eq!(
            decide_residency(total, te, Some(MemoryBudget { total_gb: 128.0 }), true),
            ResidencyOutcome::Resident
        );
        // Budget between staged (14) and resident (15): resident won't fit, staged will, provider
        // stages ⇒ Sequential. This is the fit-gate SELECTING sequential residency.
        assert_eq!(
            decide_residency(total, te, Some(MemoryBudget { total_gb: 14.5 }), true),
            ResidencyOutcome::Sequential
        );
        // Same budget but a provider that can't stage ⇒ reject (never a silent Resident that OOMs).
        assert!(matches!(
            decide_residency(total, te, Some(MemoryBudget { total_gb: 14.5 }), false),
            ResidencyOutcome::Reject {
                staged_gb: None,
                ..
            }
        ));
        // Budget below even the staged peak ⇒ reject, naming the staged requirement.
        assert!(matches!(
            decide_residency(total, te, Some(MemoryBudget { total_gb: 12.0 }), true),
            ResidencyOutcome::Reject {
                staged_gb: Some(_),
                ..
            }
        ));
        // No budget signal ⇒ Resident (never block).
        assert_eq!(
            decide_residency(total, te, None, true),
            ResidencyOutcome::Resident
        );
        // Unmeasured weights ⇒ Resident (no signal).
        assert_eq!(
            decide_residency(0, 0, Some(MemoryBudget { total_gb: 8.0 }), true),
            ResidencyOutcome::Resident
        );
    }

    #[test]
    fn sum_text_encoder_bytes_sums_only_text_encoder_subdirs() {
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_te_{}_{}",
            std::process::id(),
            line!()
        ));
        // SDXL-shaped tree: two CLIP encoders + the U-Net + VAE.
        for (sub, bytes) in [
            ("text_encoder", 1000usize),
            ("text_encoder_2", 3000),
            ("unet", 9000),
            ("vae", 400),
        ] {
            let dir = root.join(sub);
            std::fs::create_dir_all(&dir).expect("mk subdir");
            std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).expect("weights");
        }
        // Only the two text-encoder subdirs count (1000 + 3000); unet/vae are excluded.
        assert_eq!(sum_text_encoder_bytes(&root), 4000);
        // The whole-model sum includes everything.
        assert_eq!(sum_safetensors_bytes(&root), 13400);
        // Missing dir ⇒ 0.
        assert_eq!(sum_text_encoder_bytes(&root.join("nope")), 0);

        std::fs::remove_dir_all(&root).ok();
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
