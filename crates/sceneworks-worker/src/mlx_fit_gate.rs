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

/// Whether `engine_id`'s provider drops components in phase order under [`OffloadPolicy::Sequential`]
/// — derived at query time from the engine's REGISTERED descriptor
/// [`Capabilities`](gen_core::Capabilities)`::supports_sequential_offload` bit, not a hand-maintained
/// allowlist (sc-10840, epic 10834).
///
/// Why the descriptor bit is the right source of truth: [`OffloadPolicy::Sequential`] is *advisory* —
/// a provider that has NOT wired the load→use→drop residency lifecycle silently treats it as
/// `Resident` (never an error), so predicting "it fits staged" and then holding everything resident
/// would SIGKILL. `supports_sequential_offload` is precisely the provider's own machine-readable
/// attestation that it wired that lifecycle (the gen-core discovery signal, sc-11126). Reading it
/// per-engine makes the gate self-maintaining: every family the mlx-gen Phase-1 fan-out wires
/// (sc-10840 — sd3/sana/flux/flux2/chroma/ideogram/kolors/anima/boogu/bernini alongside the earlier
/// sdxl/z-image/qwen/lens/krea families) is covered the moment its descriptor advertises the bit, with
/// no lockstep edit here. An engine that does not separate a text encoder (e.g. sensenova's fused MoT,
/// `footprint` te=0) leaves the bit `false` and is correctly never offered `Sequential` — a no-op that
/// would OOM.
///
/// This is a pre-load, weights-free registry lookup (`(descriptor)()` allocates no tensors), the same
/// query shape the worker already uses for family/guidance/quant capability advertisement and the
/// analogous [`gen_core::footprint`] size seam (sc-10894). An id with no registered generator — or a
/// registered one that does not advertise the bit — yields `false` (the safe default: never select a
/// residency policy the provider won't honor). Sees exactly the providers the binary force-links (the
/// sc-4482 dead-strip rule): on macOS every `mlx_gen_*` provider is anchored in `image_jobs`, so the
/// live worker resolves them; off-Mac the MLX gate no-ops anyway (no `sysctl` budget).
pub(crate) fn engine_supports_sequential(engine_id: &str) -> bool {
    gen_core::registry::generators()
        .find(|reg| (reg.descriptor)().id == engine_id)
        .is_some_and(|reg| (reg.descriptor)().capabilities.supports_sequential_offload)
}

/// Emulate a smaller Mac: force the total-unified-memory budget (GB). Set e.g.
/// `SCENEWORKS_MLX_MEMORY_CAP_GB=16` to make the gate treat this machine as a 16 GB Mac, so a model
/// that would OOM there is rejected (and, once Phase 1 lands, run under sequential residency) exactly
/// as on real small hardware. Unset / non-positive ⇒ use the real `sysctl hw.memsize` total.
pub(crate) const MLX_MEMORY_CAP_ENV: &str = "SCENEWORKS_MLX_MEMORY_CAP_GB";

/// Headroom (GiB) added on top of the summed on-disk component weights to cover the MLX Metal
/// activation transient during denoise/decode plus the OS + other apps drawing from the same unified
/// pool (the gate budgets against TOTAL physical RAM, so the OS share must come out of this headroom).
///
/// CALIBRATED (sc-10863) from real `get_peak_memory` footprints measured through
/// `footprint_measure.rs` (one tier per process; peak = load + one 1024² generation, RESIDENT
/// hold-all path, no memory cap). Measured `transient = peak − resident` and `headroom = peak − disk`:
///
/// | model            | disk GiB | resident | peak  | transient | headroom(peak−disk) |
/// |------------------|---------:|---------:|------:|----------:|--------------------:|
/// | illustrious q8   |     5.01 |     4.74 | 18.78 |     14.04 |               13.77 |
/// | lens q4          |    17.67 |    16.46 | 30.50 |     14.04 |               12.83 |
/// | qwen-image q8    |    35.90 |    33.45 | 41.11 |      7.66 |                5.20 |
/// | lens-turbo bf16  |    28.43 |    45.67 | 75.55 |     29.88 |               47.12 |
///
/// FINDING — the transient is NOT a function of on-disk weight bytes: qwen-image q8 has the LARGEST
/// weights (35.9 GiB) but the SMALLEST transient (7.66 — its VAE decode is tiled, sc-11747), while
/// illustrious q8 has the smallest weights but a 14 GiB transient. It is architecture- and
/// resolution-bound (dominated by the VAE decode + attention at the output resolution), so a
/// disk-SCALED predictor (`Σweights · k`) would over-reject the large-but-efficient models and
/// under-predict the small ones — the wrong shape. And the load-time gate cannot see the request
/// resolution (the generator is cached across resolutions), so a per-request `f(resolution)` term is
/// not threadable at this seam. A conservative CONSTANT is therefore the right structure.
///
/// 18 GiB = the max measured transient at 1024² (14.04, illustrious q8 & lens q4) + ~4 GiB macOS/app
/// reserve. This replaces the provisional 10.0, which UNDER-predicted 3 of the 4 measured tiers
/// (illustrious 15.0<18.8, lens 27.7<30.5, lens-turbo 38.4<75.6) — i.e. was a latent SIGKILL risk on
/// Macs sized between the predicted and the real peak. All three resident≈disk tiers are now covered
/// with margin without over-rejecting a model that fits (illustrious q8: 5.01+18=23.0 still fits a
/// 24 GiB Mac, where its real 18.8 GiB peak + OS does too).
///
/// NOT covered by this constant (surfaced sc-10863, tracked follow-ups — see the story): (1) IN-MEMORY
/// WEIGHT EXPANSION — lens-turbo bf16's headroom is 47 because its mxfp4-on-disk gpt-oss text encoder
/// loads to bf16 (resident 45.67 = 1.61× disk 28.43), so `sum_safetensors_bytes` under-counts the
/// in-memory weights and the gate under-predicts this tier REGARDLESS of headroom (pre-existing: 10
/// under-predicted it by 37 GiB, 18 by 33). A blanket bf16 expansion factor can't fix it (a bf16 tier
/// whose encoder is bf16-on-disk would then be over-rejected ~1.6×); the real fix needs the weight-byte
/// input to reflect in-memory size, per family, backed by bf16 measurements across models. (2) Output
/// RESOLUTION > 1024² grows the VAE-decode transient past 14 GiB — all four points are 1024², so 18 is
/// a 1024²-worst-case; a higher-res campaign is a follow-up.
const HEADROOM_GB: f64 = 18.0;

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

/// On-disk `.safetensors` bytes of a [`WeightsSource`]: the recursive sum for a `Dir`, the file length
/// for a single-file `File`. Used to fold a separate control/overlay checkpoint ([`LoadSpec::control`])
/// into the fit total — its weights are not under the base `spec.weights` tree.
fn weights_source_bytes(src: &WeightsSource) -> u64 {
    match src {
        WeightsSource::Dir(dir) => sum_safetensors_bytes(dir),
        WeightsSource::File(file) => std::fs::metadata(file).map_or(0, |meta| meta.len()),
    }
}

/// Resolve the TEXT-ENCODER on-disk bytes for the staged split (sc-10894), preferring the provider-owned
/// per-component footprint over the `text_encoder*` subdir scan.
///
/// The subdir scan ([`sum_text_encoder_bytes`]) only recognizes the *diffusers* `text_encoder*` naming;
/// it returns **zero** for a family whose encoder lives elsewhere — boogu's `mllm/`, bernini's flat
/// `t5_encoder.safetensors`, anima's `text_encoders/` under a `split_files/` root — or that has no
/// separable encoder at all (sensenova's flat unified MoT). A zero text-encoder collapses the staged
/// (`max(te, rest)`) peak back to the resident peak, so no `Sequential` saving is ever selected. The
/// provider's `gen_core::footprint` computes the split from the exact subdirs *its own* loader resolves,
/// so it is authoritative per family. `footprint_te` is `Some` when the provider declared a footprint,
/// `None` otherwise (or the query errored) — in which case this falls back to the subdir scan, the
/// historical behavior. The whole-model `total` stays the recursive [`sum_safetensors_bytes`] sum, so
/// `rest = total − te` accounts for the DiT + VAE + anything else regardless of the footprint's own
/// dit/vae split (and keeps the sc-11006 control-branch folding intact).
pub(crate) fn resolve_text_encoder_bytes(footprint_te: Option<u64>, dir: &Path) -> u64 {
    footprint_te.unwrap_or_else(|| sum_text_encoder_bytes(dir))
}

/// Sum the on-disk `.safetensors` bytes of the model's TEXT-ENCODER component(s) — the phase-A
/// component the `Sequential` residency drops before the DiT loads (sc-10839). Matches the diffusers
/// snapshot's top-level `text_encoder` / `text_encoder_2` / `text_encoder_*` subdirs (SDXL has both
/// CLIP encoders; Z-Image the single Qwen encoder), reusing [`sum_safetensors_bytes`] per subdir so
/// the HF-cache symlink + AppleDouble handling is shared. `0` if the dir is missing or has no
/// recognizable text-encoder subdir — which makes the staged estimate fall back to the resident sum
/// (no claimed saving), the safe direction. Superseded, when a provider declares a footprint, by
/// [`resolve_text_encoder_bytes`] (sc-10894); still the fallback for providers that do not.
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
    // Prefer the provider-owned per-component footprint for the text-encoder split (sc-10894): it is
    // authoritative per family, unlike the `text_encoder*` subdir guess which reads ZERO for boogu
    // (`mllm/`), bernini (flat `t5_encoder.safetensors`), anima (`text_encoders/` under `split_files/`),
    // etc. `Ok(None)` (provider declares no footprint) or an `Err` (unknown id / single-file source)
    // fall open to the subdir scan. Query BEFORE the match so both `Dir`/`File` arms see it.
    let footprint_te = gen_core::footprint(engine_id, &spec)
        .ok()
        .flatten()
        .map(|fp| fp.text_encoder);
    let (mut total_bytes, te_bytes) = match &spec.weights {
        WeightsSource::Dir(dir) => (
            sum_safetensors_bytes(dir),
            resolve_text_encoder_bytes(footprint_te, dir),
        ),
        // A single-file source has no diffusers component tree; honor a footprint TE if the provider
        // somehow computed one, else 0 (resident-or-reject only).
        WeightsSource::File(file) => (
            std::fs::metadata(file).map_or(0, |meta| meta.len()),
            footprint_te.unwrap_or(0),
        ),
    };
    // A ControlNet/overlay checkpoint (`spec.control`, e.g. qwen_image_control's VACE branch) lives in
    // a SEPARATE weights source, not under `spec.weights`, so it is missing from the sums above. It is
    // a HEAVY-side component (loaded + quantized with the base DiT, never dropped with the text
    // encoder), so fold its bytes into `total_bytes` only — leaving `te_bytes` alone keeps the staged
    // split `rest = total − te` counting it on the DiT side. Without this the staged-peak prediction
    // under-counts and could select Sequential when even the staged working set won't fit (→ SIGKILL).
    if let Some(control) = &spec.control {
        total_bytes += weights_source_bytes(control);
    }
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
        // qwen-image q8: ~36 GiB weights + 18 headroom (sc-10863) = 54 ⇒ too big for a 16 GB Mac.
        assert_eq!(
            fit_decision(predicted_peak_gb(36 * 1024 * 1024 * 1024_u64), Some(budget)),
            FitDecision::TooBig {
                needed_gb: 36.0 + HEADROOM_GB,
                available_gb: 16.0,
            }
        );
        // A ~3 GiB model fits a roomy budget (3 + 18 headroom = 21 < 32).
        assert_eq!(
            fit_decision(
                predicted_peak_gb(3 * 1024 * 1024 * 1024_u64),
                Some(MemoryBudget { total_gb: 32.0 })
            ),
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
    fn weights_source_bytes_counts_both_file_and_dir_control_checkpoints() {
        // The qwen_image_control VACE branch ships either as a single `.safetensors` File or as a Dir
        // of shards; both must be counted so `apply_residency_policy` folds the control branch into the
        // heavy side of the staged-peak split (else the DiT-phase working set is under-counted).
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_ctrl_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&root).expect("mk root");

        // Single-file control checkpoint ⇒ its file length.
        let file = root.join("control.safetensors");
        std::fs::write(&file, vec![0u8; 4096]).expect("control file");
        assert_eq!(
            weights_source_bytes(&WeightsSource::File(file.clone())),
            4096
        );

        // Dir control checkpoint ⇒ the recursive `.safetensors` sum (AppleDouble sidecars skipped).
        let dir = root.join("control_dir");
        std::fs::create_dir_all(&dir).expect("mk control dir");
        std::fs::write(dir.join("part-1.safetensors"), vec![0u8; 1000]).expect("shard 1");
        std::fs::write(dir.join("part-2.safetensors"), vec![0u8; 2000]).expect("shard 2");
        std::fs::write(dir.join("._part-1.safetensors"), vec![0u8; 999]).expect("sidecar");
        assert_eq!(weights_source_bytes(&WeightsSource::Dir(dir)), 3000);

        std::fs::remove_dir_all(&root).ok();
    }

    /// The gate derives sequential-capability from each engine's REGISTERED descriptor bit
    /// (`Capabilities::supports_sequential_offload`) rather than a hand-maintained allowlist (sc-10840,
    /// epic 10834). This exercises the LIVE registry, so it must see the force-linked `mlx_gen_*`
    /// providers — anchored (`use mlx_gen_* as _;` in `image_jobs`) only on macOS, the sole platform the
    /// MLX gate runs on. Off-Mac the image registry is empty, so this is macOS-gated exactly like the
    /// `engines.rs` descriptor sweeps. At the pinned mlx-gen `45428fa` every image engine advertises the
    /// bit, so every wired id resolves true through the shared registry query.
    #[cfg(target_os = "macos")]
    #[test]
    fn engine_supports_sequential_is_derived_from_the_registered_capability() {
        // The earlier-wired families (sdxl / z-image / qwen / lens / krea) still resolve true — proving
        // dropping the hardcoded allowlist introduced no regression for the already-covered engines.
        for id in [
            "sdxl",
            "z_image",
            "z_image_control",
            "z_image_turbo",
            "z_image_turbo_control",
            "qwen_image",
            "qwen_image_edit",
            "qwen_image_control",
            "lens",
            "lens_turbo",
            "krea_2_turbo",
            "krea_2_raw",
            "krea_2_edit",
            "krea_2_turbo_edit",
            "krea_2_turbo_control",
        ] {
            assert!(
                engine_supports_sequential(id),
                "{id}: earlier-wired family must stay sequential-capable"
            );
        }
        // The sc-10840 Phase-1 fan-out families are AUTO-covered by the capability query with no
        // allowlist edit here — the whole point of deriving from the descriptor bit. A newly-wired
        // engine (e.g. `sd3_5_large`) is sequential-capable the moment its provider advertises the bit.
        for id in [
            "sd3_5_large",
            "sd3_5_large_turbo",
            "sd3_5_medium",
            "sana_1600m",
            "sana_sprint_1600m",
            "flux1_schnell",
            "flux1_dev",
            "flux1_dev_control",
            "flux2_klein_9b",
            "flux2_klein_9b_edit",
            "flux2_klein_9b_kv_edit",
            "flux2_dev",
            "flux2_dev_control",
            "flux2_dev_edit",
            "chroma1_base",
            "chroma1_flash",
            "chroma1_hd",
            "ideogram_4",
            "ideogram_4_turbo",
            "kolors",
            "anima_base",
            "anima_aesthetic",
            "anima_turbo",
            "boogu_image",
            "boogu_image_turbo",
            "boogu_image_edit",
            "bernini",
        ] {
            assert!(
                engine_supports_sequential(id),
                "{id}: sc-10840 fan-out engine must be sequential-capable at mlx-gen 45428fa"
            );
        }
        // A REGISTERED engine that does NOT advertise the bit stays false: sensenova's encoder is fused
        // into a unified MoT (`footprint` te=0) — no separable text encoder to drop, so residency buys
        // nothing and Sequential would be a no-op that OOMs. This proves the query reads the descriptor
        // BIT, not mere registry membership.
        assert!(!engine_supports_sequential("sensenova_u1_8b"));
    }

    /// An id with no registered generator is never sequential-capable (the safe default: never select a
    /// residency policy the provider won't honor) — a cross-platform invariant that holds even off-Mac
    /// where the image registry is empty.
    #[test]
    fn engine_supports_sequential_is_false_for_an_unregistered_id() {
        assert!(!engine_supports_sequential("no_such_engine_xyz"));
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
        // illustrious q8-class: total ~5 GiB (TE ~1, DiT+VAE ~4). With HEADROOM_GB=18 (sc-10863):
        // resident peak = 5+18 = 23; staged peak = max(1, 4)+18 = 22.
        let total = 5 * gib;
        let te = gib;

        // Roomy budget (128 GB Mac) ⇒ Resident (keep the warm path).
        assert_eq!(
            decide_residency(total, te, Some(MemoryBudget { total_gb: 128.0 }), true),
            ResidencyOutcome::Resident
        );
        // Budget between staged (22) and resident (23): resident won't fit, staged will, provider
        // stages ⇒ Sequential. This is the fit-gate SELECTING sequential residency.
        assert_eq!(
            decide_residency(total, te, Some(MemoryBudget { total_gb: 22.5 }), true),
            ResidencyOutcome::Sequential
        );
        // Same budget but a provider that can't stage ⇒ reject (never a silent Resident that OOMs).
        assert!(matches!(
            decide_residency(total, te, Some(MemoryBudget { total_gb: 22.5 }), false),
            ResidencyOutcome::Reject {
                staged_gb: None,
                ..
            }
        ));
        // Budget below even the staged peak (22) ⇒ reject, naming the staged requirement.
        assert!(matches!(
            decide_residency(total, te, Some(MemoryBudget { total_gb: 20.0 }), true),
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

    /// sc-10894: on a boogu-style snapshot (text encoder under `mllm/`, not `text_encoder*`), the
    /// subdir scan reads ZERO, but `resolve_text_encoder_bytes` PREFERS a provider footprint value when
    /// present and only falls back to the scan when it is `None`.
    #[test]
    fn resolve_text_encoder_prefers_footprint_over_subdir_scan() {
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_resolve_{}_{}",
            std::process::id(),
            line!()
        ));
        // Encoder under `mllm/`, DiT `transformer/`, VAE `vae/` — NO `text_encoder*` subdir.
        for (sub, bytes) in [("mllm", 1500usize), ("transformer", 9000), ("vae", 400)] {
            let dir = root.join(sub);
            std::fs::create_dir_all(&dir).expect("mk subdir");
            std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).expect("weights");
        }
        // The historical subdir scan finds no `text_encoder*` → 0 (the bug this seam fixes).
        assert_eq!(sum_text_encoder_bytes(&root), 0);
        // The whole-model sum still sees every component.
        assert_eq!(sum_safetensors_bytes(&root), 10900);
        // No footprint declared ⇒ fall back to the (zero) subdir scan.
        assert_eq!(resolve_text_encoder_bytes(None, &root), 0);
        // A provider footprint (the `mllm/` bytes) is preferred, even though the scan reads zero.
        assert_eq!(resolve_text_encoder_bytes(Some(1500), &root), 1500);

        std::fs::remove_dir_all(&root).ok();
    }

    /// sc-10894 end-to-end: a non-zero footprint text encoder flips the residency decision from Reject to
    /// Sequential where the zero-reading subdir scan (the fallback) would reject. This is the whole point
    /// of the seam — the staged peak is only real when the text-encoder split is measured.
    #[test]
    fn footprint_text_encoder_flips_reject_to_sequential() {
        let gib = 1024 * 1024 * 1024_u64;
        // boogu-class: whole model 22 GiB (mllm 13 + transformer 8 + vae 1). No `text_encoder*` subdir,
        // so the subdir scan reads 0.
        let total = 22 * gib;
        // Budget between the staged (31) and resident (40) peaks under HEADROOM_GB=18 (sc-10863).
        let budget = Some(MemoryBudget { total_gb: 34.0 });

        // Fallback path (footprint None on a dir with no `text_encoder*`) → te = 0 → staged == resident
        // peak (22 + 18 = 40) > 34 → Reject, even though one-component-at-a-time WOULD fit.
        let te_fallback = resolve_text_encoder_bytes(None, std::path::Path::new("/nonexistent"));
        assert_eq!(te_fallback, 0);
        assert!(matches!(
            decide_residency(total, te_fallback, budget, true),
            ResidencyOutcome::Reject { .. }
        ));

        // Provider footprint (te = 13 GiB) → staged = max(13, 22 − 13 = 9) + 18 = 31 ≤ 34 → Sequential.
        let te_footprint =
            resolve_text_encoder_bytes(Some(13 * gib), std::path::Path::new("/ignored"));
        assert_eq!(te_footprint, 13 * gib);
        assert_eq!(
            decide_residency(total, te_footprint, budget, true),
            ResidencyOutcome::Sequential
        );
    }
}
