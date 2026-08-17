//! Candle/CUDA VRAM fit-gate + small-card emulation (epic 10765 Phase 0, sc-10766).
//!
//! The dynamic complement to the static per-tier `candle.minMemoryGb` manifest gate (sc-9094): before a
//! candle generation loads, predict the selected tier's peak VRAM and compare it against a LIVE budget
//! so a card that can't fit the model is rejected with an actionable message instead of being
//! SIGKILL'd / Metal-OOM'd mid-render. Also honors `SCENEWORKS_CUDA_VRAM_CAP_GB`, which emulates a
//! smaller card on big hardware so the Phase 1 offload paths (sc-10769) can be validated on the dev
//! box's 96 GB RTX PRO 6000s.
//!
//! ## Caching-allocator caveat (why this budgets on PREDICTED peak, not resident deltas)
//! candle's CUDA backend uses cudarc, with no `empty_cache` and no reclaim on `Device::synchronize()`.
//! A freed WITHIN-DEVICE component — the text encoders dropped before the DiT under Phase 1 sequential
//! residency, with the device still alive — returns to candle's in-process pool (reused by the next
//! allocation, which is what keeps a small card from OOMing) WITHOUT the DRIVER-resident `nvidia-smi`
//! free number dropping back. A full generator/device DROP is different: it destroys the device's pool
//! and returns its VRAM to the driver, so `free` RISES (GPU-measured, sc-13960). Either way this gate is
//! a pre-load ADMISSION check keyed off the manifest's measured per-tier peak — never a post-free
//! accounting number, since it runs while the model it is about to replace is still resident.
//!
//! Everything here is pure and unit-tested; the live `nvidia-smi` reading lives in [`crate::gpu`] and
//! the wiring is in `generate_candle_stream` (image_jobs/base.rs).

use super::*;
#[cfg(test)]
use gen_core::MemoryStrategy;
#[cfg(test)]
use serde_json::Value;

use crate::fit_gate::BYTES_PER_GIB;
pub(crate) use crate::fit_gate::{resolve_offload, FitDecision};

/// Emulate a smaller card: cap usable VRAM (GB). Set e.g. `SCENEWORKS_CUDA_VRAM_CAP_GB=10` to make the
/// fit-gate treat this GPU as a 10 GB card, so a too-big model is rejected (and, once Phase 1 lands,
/// offloaded) exactly as it would be on real small hardware. Unset / non-positive ⇒ use the real
/// live free VRAM.
pub(crate) const CUDA_VRAM_CAP_ENV: &str = "SCENEWORKS_CUDA_VRAM_CAP_GB";

/// Fixed transient/runtime headroom (GB) added on top of a per-tier MEASURED peak (`candle.vramGbByTier`)
/// to cover allocator slack + activation spikes not captured by the steady peak. Not added on top of
/// `candle.minMemoryGb`, which the manifest already pads over the measured peak (sc-9094).
/// Candle/CUDA's dedicated-VRAM allocator/context slack. This aliases the backend-neutral typed
/// policy so the generic gate, control gate, Mochi, and the web mirror cannot assign independent
/// meanings or values to the same dedicated pool.
pub(crate) const HEADROOM_GB: f64 = crate::fit_gate::dedicated_vram_reserve().gb;

/// A live (or capped) VRAM budget for the selected GPU, in GB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VramBudget {
    pub free_gb: f64,
    pub total_gb: f64,
}

/// Read the small-card cap from the environment. `Some(gb)` only for a positive number.
pub(crate) fn cuda_vram_cap_gb() -> Option<f64> {
    parse_vram_cap(std::env::var(CUDA_VRAM_CAP_ENV).ok().as_deref())
}

/// Parse the cap value: a positive float (GB), else `None`.
pub(crate) fn parse_vram_cap(raw: Option<&str>) -> Option<f64> {
    let value = raw?.trim().parse::<f64>().ok()?;
    (value.is_finite() && value > 0.0).then_some(value)
}

/// Apply the small-card cap to a real budget. The cap emulates a card whose TOTAL VRAM is `cap` GB:
/// `total := cap`, `free := min(real_free, cap)`. With no real reading a cap still yields a full budget
/// (`free = total = cap`) so the gate is exercisable in a no-GPU unit test. No cap ⇒ the real budget
/// unchanged.
pub(crate) fn apply_vram_cap(real: Option<VramBudget>, cap: Option<f64>) -> Option<VramBudget> {
    match cap {
        Some(cap) => {
            let free_gb = real.map_or(cap, |budget| budget.free_gb.min(cap));
            Some(VramBudget {
                free_gb,
                total_gb: cap,
            })
        }
        None => real,
    }
}

/// Fold this process's RECLAIMABLE VRAM into the live budget (sc-11023): add `reclaimable_gb` to
/// `free_gb`, clamped to the physical `total_gb`. The candle generator cache is a single exclusive slot
/// that evicts its current occupant BEFORE loading the incoming model. This budget is read while that
/// occupant is still resident (so raw `free` still counts the model it is about to replace); crediting
/// the reclaimable pool predicts the `free` the imminent evict will PRODUCE, so a warm same-model
/// re-gate or a swap-in isn't falsely rejected. That prediction holds whether evicting returns the VRAM
/// to the driver (GPU-measured sc-13960: a full generator drop frees most of it back — `nvidia-smi`
/// `free` rises) or a within-device component free keeps it pooled in-process — and the clamp to
/// `total_gb` is the safety net that keeps it honest against a stale high-water. A no-op when
/// `reclaimable_gb == 0` (nothing loaded yet), so a genuine cold first-load is gated exactly as before.
pub(crate) fn with_reclaimable(budget: VramBudget, reclaimable_gb: f64) -> VramBudget {
    VramBudget {
        free_gb: (budget.free_gb + reclaimable_gb.max(0.0)).min(budget.total_gb),
        total_gb: budget.total_gb,
    }
}

/// Per-GPU high-water of the peak VRAM (GB) this process has admitted a load at (sc-11023) — a bound on
/// what evicting the current resident can free back for a swap-in. Keyed by `gpu_id` (a worker process
/// is pinned to one card, but keying is defensive + explicit). It only ever grows (monotonic MAX), and
/// [`with_reclaimable`] clamps the credit to `total_gb`, so a later gate can only under-credit, never
/// over-admit — the safety that holds whether an evict frees to the driver or pools in-process.
fn reclaimable_pool_store() -> &'static std::sync::Mutex<std::collections::HashMap<String, f64>> {
    static STORE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, f64>>> =
        std::sync::OnceLock::new();
    STORE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Record the peak VRAM (GB) an admitted load will occupy on `gpu_id`, as a monotonic high-water
/// (sc-11023). Called only after the fit-gate ADMITS a load, so it reflects a real allocation attempt.
/// A non-positive peak is ignored.
pub(crate) fn note_loaded_peak(gpu_id: &str, peak_gb: f64) {
    // Ignore a non-positive, NaN, or infinite peak (defensive — a real measured peak is finite > 0).
    if !peak_gb.is_finite() || peak_gb <= 0.0 {
        return;
    }
    let mut store = reclaimable_pool_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let entry = store.entry(gpu_id.to_owned()).or_insert(0.0);
    if peak_gb > *entry {
        *entry = peak_gb;
    }
}

/// The reclaimable in-process VRAM pool (GB) for `gpu_id` — the [`note_loaded_peak`] high-water, or
/// `0.0` when nothing has loaded on this card yet (so [`with_reclaimable`] is a no-op on a cold start).
pub(crate) fn reclaimable_pool_gb(gpu_id: &str) -> f64 {
    reclaimable_pool_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(gpu_id)
        .copied()
        .unwrap_or(0.0)
}

/// The tier key (`nvfp4`/`bf16`/`q8`/`q4`) the request selected — derived the SAME way the tier-subdir
/// resolvers pick their folder (`advanced.mlxQuantize` → manifest `mlx.quantize` → `q8` default;
/// `<= 0` ⇒ `bf16`; `<= 4` ⇒ `q4`; else `q8`). Deliberately NOT derived from the resolved `Quant`,
/// which is `None` on packed-tier candle families whose tier is the resolved subdir, not a load-time
/// quantize (see `resolve_quant`'s candle-lane note).
///
/// **`nvfp4` (sc-11042) is passed IN, not parsed here.** NVFP4 carries no `mlxQuantize` by design (no
/// integer is honest for a ~4.5-effective-bit tier), so a bits-derived key can only ever return `q8`
/// for it — which sized an NVFP4 render against `candle.vramGbByTier["q8"]`, roughly 2× its real
/// footprint. That failed CONSERVATIVE (a spurious `TooBig`/`Offload`, never an OOM), but it was the
/// fourth bits-derived site of the same aliasing and is now keyed off the tier IDENTITY like the other
/// three. The caller passes `image_jobs::base::nvfp4_selected` — the same predicate behind the load
/// quant and the recorded label, so all four agree on what ran by construction. This module stays pure
/// (no GPU probe, no filesystem) precisely because the flag arrives resolved.
///
/// Mirrors [`preferred_tier`](crate::image_jobs::base)'s `(bits, floor, nvfp4)` shape: `nvfp4 == false`
/// leaves every existing mapping byte-identical.
pub(crate) fn requested_tier_key(
    advanced: &JsonObject,
    manifest_entry: &JsonObject,
    nvfp4: bool,
) -> &'static str {
    // The distinct NVFP4 tier short-circuits the bits map: it is a tier identity, not a point on the
    // bits ladder, and it has no honest `mlxQuantize` integer to be derived from.
    if nvfp4 {
        return NVFP4_TIER;
    }
    let raw = advanced.get("mlxQuantize").and_then(quant_int).or_else(|| {
        manifest_entry
            .get("mlx")
            .and_then(|mlx| mlx.get("quantize"))
            .and_then(quant_int)
    });
    match raw {
        None => "q8",
        Some(bits) if bits <= 0 => "bf16",
        Some(bits) if bits <= 4 => "q4",
        Some(_) => "q8",
    }
}

/// Predicted peak VRAM (GB) for `tier_key` from the manifest `candle` block:
/// `candle.vramGbByTier[tier_key]` (measured peak) + [`HEADROOM_GB`], else `candle.minMemoryGb`
/// (already padded), else `None` — an unmeasured model (no `candle` block, e.g. the dense sc-3675
/// families) skips the gate entirely.
///
/// **An `nvfp4` tier with no measured row degrades to the `q8` row, NOT to `minMemoryGb` (sc-11042).**
/// sc-11043 owns the convert-at-install loop and **must add an `nvfp4` row to `vramGbByTier` when it
/// converts a tier** — this is the documented behavior until it does. `minMemoryGb` is the WRONG floor
/// to land on here: per the manifest schema it is "the measured overall-peak of the DEFAULT (lightest,
/// typically q4) hosted tier", which heavier tiers are explicitly allowed to exceed — so falling
/// through to it would size an FP4 render against the lightest tier's number and fail PERMISSIVELY
/// (an under-prediction admits a load that can OOM). The `q8` row instead OVER-predicts (q8's weights
/// are ~2× NVFP4's ~4.5 effective bits), which is both safe and exactly the number this gate already
/// used for an NVFP4 request before the tier had its own key — the bits-derived
/// [`requested_tier_key`] returned `"q8"` for it. So the sizing is no less conservative than the
/// status quo, and a missing row can only ever cost a spurious `TooBig`/`Offload`, never an OOM.
pub(crate) fn predicted_peak_gb(manifest_entry: &JsonObject, tier_key: &str) -> Option<f64> {
    let candle = manifest_entry.get("candle")?;
    let measured = |key: &str| {
        candle
            .get("vramGbByTier")
            .and_then(|tiers| tiers.get(key))
            .and_then(json_f64)
    };
    if let Some(gb) = measured(tier_key) {
        return Some(gb + HEADROOM_GB);
    }
    // NVFP4 with no measured row → the q8 row (a deliberate over-prediction; see the note above).
    if tier_key == NVFP4_TIER {
        if let Some(gb) = measured("q8") {
            return Some(gb + HEADROOM_GB);
        }
    }
    candle.get("minMemoryGb").and_then(json_f64)
}

/// Resident prediction with a load-exact independently resident adapter stack. Callers pass zero
/// for a dense/folded load; packed providers pass the measured source bytes retained as residuals.
pub(crate) fn predicted_peak_gb_with_adapter_bytes(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
) -> Option<f64> {
    predicted_peak_gb(manifest_entry, tier_key)
        .map(|peak| peak + adapter_bytes as f64 / BYTES_PER_GIB)
}

/// Predicted SEQUENTIAL peak VRAM (GB) for `tier_key`: `candle.sequentialPeakGb[tier_key]` (the measured
/// largest single working set of the sequential-residency path, sc-10856) + [`HEADROOM_GB`], mirroring
/// [`predicted_peak_gb`]'s headroom. `None` when unmeasured (no `sequentialPeakGb`, or no entry for this
/// tier) — then the [`FitDecision::Offload`] path keeps today's best-effort behavior: run sequentially
/// and lean on the reactive Metal/CUDA-OOM containment backstop rather than reject.
///
/// An `nvfp4` tier with no measured row falls back to the `q8` row, mirroring [`predicted_peak_gb`]'s
/// NVFP4 note (q8's sequential working set is an over-prediction of NVFP4's, so the second-stage gate
/// degrades conservatively rather than best-effort). sc-11043 must add the real `nvfp4` rows.
pub(crate) fn predicted_sequential_peak_gb(
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    let sequential = manifest_entry.get("candle")?.get("sequentialPeakGb")?;
    let measured = |key: &str| sequential.get(key).and_then(json_f64);
    measured(tier_key)
        .or_else(|| (tier_key == NVFP4_TIER).then(|| measured("q8")).flatten())
        .map(|gb| gb + HEADROOM_GB)
}

/// Sequential prediction with the same adapter residency charged in every lifecycle policy.
pub(crate) fn predicted_sequential_peak_gb_with_adapter_bytes(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
) -> Option<f64> {
    predicted_sequential_peak_gb(manifest_entry, tier_key)
        .map(|peak| peak + adapter_bytes as f64 / BYTES_PER_GIB)
}

/// Decide whether the predicted peak fits the (possibly capped) live budget. Missing either input ⇒
/// `Unknown` (never block). Compares against `free_gb` — what is actually allocatable now — mirroring
/// the flux2 edit guard's use of `available_gb`.
pub(crate) fn fit_decision(needed_gb: Option<f64>, budget: Option<VramBudget>) -> FitDecision {
    let (Some(needed_gb), Some(budget)) = (needed_gb, budget) else {
        return FitDecision::Unknown;
    };
    if budget.free_gb + f64::EPSILON < needed_gb {
        FitDecision::TooBig {
            needed_gb,
            available_gb: budget.free_gb,
        }
    } else {
        FitDecision::Fits
    }
}

/// Second-stage gate for an [`FitDecision::Offload`] (epic 10765, sc-10856): sequential residency was
/// selected because the RESIDENT peak won't fit, on the promise that the sequential working set will.
/// If the tier's MEASURED sequential peak is known (`sequential_needed_gb` = [`predicted_sequential_peak_gb`])
/// and STILL exceeds the budget, this returns `Some(needed_gb)` so the caller can reject before load with
/// an actionable message instead of running into a reactive Metal/CUDA OOM. `None` — the sequential peak
/// fits, is unmeasured for this tier, or there is no live budget — keeps today's best-effort run.
pub(crate) fn sequential_overflow_gb(
    sequential_needed_gb: Option<f64>,
    budget: Option<VramBudget>,
) -> Option<f64> {
    let (needed_gb, budget) = (sequential_needed_gb?, budget?);
    (budget.free_gb + f64::EPSILON < needed_gb).then_some(needed_gb)
}

/// The resolved residency PLAN for a sequential-capable candle image lane: the pure composition of
/// [`fit_decision`] → [`resolve_offload`] → [`sequential_overflow_gb`] that base.rs's txt2img gate and
/// the bespoke Qwen-Edit gate share. Carries only the ACTION — never a budget-derived number — so two
/// plans computed against different budgets compare equal **iff they take the same action**. That is
/// the invariant the sc-13960 evict-then-reclaim two-pass leans on: gate once against raw free and once
/// against `free + reclaimable_pool`, and evict only when the plan actually improves (never for two
/// rejects that differ only in their reported free-VRAM number).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoadPlan {
    /// Load at full residency — the whole-model resident peak fits (or the tier is unmeasured, so the
    /// gate never blocks).
    Resident,
    /// Load with sequential component residency — the resident peak overflows but the staged working
    /// set fits (or its peak is unmeasured, so best-effort staging).
    Sequential,
    /// Reject before OOM — the resident peak overflows AND either the measured sequential peak also
    /// overflows or the engine cannot stage.
    Reject,
}

/// Translate the shared strategy vocabulary to Krea's historical manifest keys at the evidence-read
/// boundary. This is deliberately a function, not a second rung enum: selection order and execution
/// composition remain owned by [`gen_core::MemoryStrategy`] and the provider contract.
fn krea_turbo_manifest_key(strategy: gen_core::MemoryStrategy) -> &'static str {
    match strategy {
        gen_core::MemoryStrategy::Resident => "resident",
        gen_core::MemoryStrategy::StagedResidency => "threeStage",
        gen_core::MemoryStrategy::BoundedDecode => "tiledVae",
        gen_core::MemoryStrategy::BoundedAttention => "chunkedAttention",
        gen_core::MemoryStrategy::BoundedTransformerResidency => "streamedBlocks",
    }
}

/// Per-phase prediction for one Krea Turbo rung at the requested geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct KreaTurboPhasePeaks {
    pub text_gb: f64,
    pub denoise_gb: f64,
    pub decode_gb: f64,
}

impl KreaTurboPhasePeaks {
    pub(crate) fn peak_gb(self) -> f64 {
        self.text_gb.max(self.denoise_gb).max(self.decode_gb)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum KreaTurboFit {
    Resident {
        peak_gb: f64,
        needed_gb: f64,
        selection: gen_core::MemorySelection,
    },
    Fits {
        phases: KreaTurboPhasePeaks,
        needed_gb: f64,
        selection: gen_core::MemorySelection,
        memory: gen_core::GenerationMemory,
        /// The rung was carried by a synthesized fitted-curve ESTIMATE, not by an exact measured
        /// record of this cell (sc-18097). True exactly when the request geometry has no
        /// `exact_request` record: a rung's measured candidates come only from that record, and a
        /// rung whose record exists but is structurally excluded emits no estimate either (the
        /// whole-tier fail-closed rule in `krea_turbo_fit_with_runtime`), so those are the only
        /// two states. Consumed by [`krea_turbo_smaller_fit_with_runtime`], which — mirroring
        /// `mlx_fit_gate::verified_lower_alternative` — must not offer an estimate-backed
        /// geometry as refusal advice.
        estimate_scoped: bool,
    },
    Reject {
        phases: KreaTurboPhasePeaks,
        needed_gb: f64,
    },
    Unverified {
        reason: gen_core::MemoryEvidenceVerdict,
    },
}

pub(crate) const KREA_TURBO_SCENEWORKS_REVISION: &str = "sc-15449-contract-v1";
// sc-17097: the inference revision the shipped Krea phase curves are declared COMPATIBLE with. It must
// stay equal to the manifest's `turboFit.inferenceRevision` - `candidate_exclusion` compares the two and
// stales every optimized rung when they diverge.
//
// The curves were captured against `277f4238`; each evidence record keeps
// that exact commit in its own `inferenceCommit` receipt, which is never rewritten (sc-16482: a receipt
// testifies to its own run). This constant moved to the sc-15819 closeout pin only after verifying the
// range is a single commit whose diff against BOTH `candle-gen-krea` and `gen-core/src/memory_strategy.rs`
// is empty - the measured path and the calibration identity (ABI 3, fingerprint, deferred load shape) are
// byte-for-byte unchanged, so the captures remain valid rather than merely re-stamped.

#[derive(Clone, Debug)]
pub(crate) struct KreaRuntimeEvidenceContext {
    resolved_route: String,
    backend: String,
    gpu_id: String,
    compute_capability: f32,
    artifact_provider: String,
    artifact_repository: String,
    resolved_revision: String,
    tier_root: String,
    resolved_artifact_root: std::path::PathBuf,
}

impl KreaRuntimeEvidenceContext {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inspect(
        resolved_route: &str,
        backend: &str,
        gpu_id: &str,
        compute_capability: Option<f32>,
        artifact_provider: &str,
        artifact_repository: &str,
        resolved_revision: &str,
        tier_root: &str,
        resolved_artifact_root: &std::path::Path,
        pinned_snapshot_root: &std::path::Path,
    ) -> Option<Self> {
        let compute_capability = compute_capability?;
        let expected = pinned_snapshot_root.join(tier_root);
        if resolved_artifact_root.canonicalize().ok()? != expected.canonicalize().ok()? {
            return None;
        }
        let components = sceneworks_core::mlx_tier_completeness::tier_declared_components(
            resolved_artifact_root,
        )?;
        if components.is_empty()
            || !components.iter().all(|component| {
                std::fs::read_dir(resolved_artifact_root.join(component)).is_ok_and(|entries| {
                    entries
                        .flatten()
                        .any(|entry| !sceneworks_core::lora_family::is_hidden_file(&entry.path()))
                })
            })
        {
            return None;
        }
        let parse_json = |path: &std::path::Path| {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        };
        let weights_loadable = |component: &str| {
            let dir = resolved_artifact_root.join(component);
            if parse_json(&dir.join("config.json")).is_none() {
                return false;
            }
            let index = std::fs::read_dir(&dir).ok().and_then(|entries| {
                entries.flatten().map(|entry| entry.path()).find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.ends_with(".safetensors.index.json"))
                })
            });
            if let Some(index) = index {
                return parse_json(&index)
                    .and_then(|value| value.get("weight_map")?.as_object().cloned())
                    .is_some_and(|weight_map| {
                        !weight_map.is_empty()
                            && weight_map.values().all(|shard| {
                                shard
                                    .as_str()
                                    .is_some_and(|shard| dir.join(shard).is_file())
                            })
                    });
            }
            sceneworks_core::mlx_tier_completeness::dir_has_visible_file_ending(
                &dir,
                ".safetensors",
            )
        };
        if !["transformer", "text_encoder", "vae"]
            .into_iter()
            .all(weights_loadable)
            || parse_json(&resolved_artifact_root.join("tokenizer/tokenizer.json")).is_none()
            || parse_json(&resolved_artifact_root.join("scheduler/scheduler_config.json")).is_none()
        {
            return None;
        }
        Some(Self {
            resolved_route: resolved_route.to_owned(),
            backend: backend.to_owned(),
            gpu_id: gpu_id.to_owned(),
            compute_capability,
            artifact_provider: artifact_provider.to_owned(),
            artifact_repository: artifact_repository.to_owned(),
            resolved_revision: resolved_revision.to_owned(),
            tier_root: tier_root.to_owned(),
            resolved_artifact_root: resolved_artifact_root.to_owned(),
        })
    }

    #[cfg(test)]
    pub(crate) fn verified_for_test(tier_root: &str) -> Self {
        Self {
            resolved_route: "krea_2_turbo".into(),
            backend: "candle".into(),
            gpu_id: "test-gpu".into(),
            compute_capability: 12.0,
            artifact_provider: "huggingface".into(),
            artifact_repository: "SceneWorks/krea-2-turbo-mlx".into(),
            resolved_revision: "d009674080cc1bccf2b629d834c34bf5eccdb723".into(),
            tier_root: tier_root.into(),
            resolved_artifact_root: krea_test_artifact_root(tier_root),
        }
    }
}

/// Sparse, weights-free Krea source used by the Candle-only memory-contract tests.
///
/// The provider's memory lookup now shares its complete production load-spec validation, including
/// the encoder and tokenizer contracts. An empty path therefore no longer represents a valid source.
/// Keep the tests on the real lookup seam with a structurally truthful fixture; no model tensor is
/// materialized and no memory evidence is measured or rewritten here.
#[cfg(test)]
fn krea_test_artifact_root(tier: &str) -> std::path::PathBuf {
    struct Fixture {
        _temp: tempfile::TempDir,
        root: std::path::PathBuf,
    }

    static FIXTURE: std::sync::OnceLock<Fixture> = std::sync::OnceLock::new();
    let fixture = FIXTURE.get_or_init(|| {
        let temp = tempfile::tempdir().expect("Krea memory-contract fixture root");
        let root = temp.path().to_path_buf();
        let contract = crate::inference_runtime::media_encoder_contract("krea_2_turbo")
            .expect("Krea encoder contract");
        for (tier, bits) in [("q4", Some(4)), ("q8", Some(8)), ("bf16", None)] {
            let tier_root = root.join(tier);
            gen_core_testkit::write_encoder_contract_fixture(
                &tier_root.join("text_encoder"),
                contract,
            )
            .expect("write sparse Krea encoder fixture");
            if let Some(bits) = bits {
                let transformer = tier_root.join("transformer");
                std::fs::create_dir_all(&transformer).expect("Krea transformer fixture dir");
                std::fs::write(
                    transformer.join("config.json"),
                    serde_json::to_vec(&serde_json::json!({
                        "quantization": { "bits": bits, "group_size": 64 }
                    }))
                    .expect("Krea transformer fixture config"),
                )
                .expect("write Krea transformer fixture config");
            }
        }
        Fixture { _temp: temp, root }
    });
    assert!(
        matches!(tier, "q4" | "q8" | "bf16"),
        "unsupported Krea fixture tier {tier}"
    );
    fixture.root.join(tier)
}

#[cfg(test)]
fn krea_test_load_spec(tier: &str) -> gen_core::LoadSpec {
    gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(krea_test_artifact_root(tier)))
}

/// Read a measured phase curve `fixedGb + perMpxGb * megapixels`. The manifest stores fixed weight /
/// allocator residency separately from the geometry-dependent activation slope. Invalid or
/// incomplete evidence fails closed to `None`; callers retain the established sequential gate
/// instead of inventing a fit.
///
/// SC-16514 recovered the q8/bf16 768² captures from SC-15205 activity 15272 and SC-15206 activity
/// 15314 into `turboFit.evidenceRecords`. Every tier now carries 768² and 1024² phase cells, and every
/// `perMpxGb` is fitted from that tier's own phase delta. The recovered cells are explicitly
/// `phase_fit_only`: the cited activities do not establish geometry-specific 768² output parity, so
/// they characterize the bounded curve without authorizing exact runtime admission. Q8's 7.98
/// denoise slope equals q4's because both measured rises are 3.658 GiB, not because the coefficient
/// was borrowed. Zero geometry-sensitive slopes remain only where the two samples are flat or
/// decrease; the manifest names each such pair. `maxMeasuredPixels` remains 1024² because larger
/// attention shapes have not been validated, so the curve is fitted within that bound rather than
/// extrapolated beyond it.
fn krea_phase_curve(phase: &JsonObject, pixels: u64) -> Option<f64> {
    let fixed = phase.get("fixedGb").and_then(json_f64)?;
    let per_mpx = phase.get("perMpxGb").and_then(json_f64)?;
    if !fixed.is_finite() || !per_mpx.is_finite() || fixed < 0.0 || per_mpx < 0.0 {
        return None;
    }
    Some(fixed + per_mpx * pixels as f64 / 1_000_000.0)
}

/// The typed materialization shape the shipped Krea Turbo curves were measured under (sc-17097).
///
/// `None` for a missing or unrecognized value, which fails the fit closed rather than defaulting: a
/// silently assumed shape is what let calibration ABI 2's load-shape axis do no work on this route.
pub(crate) fn krea_turbo_load_shape(turbo_fit: &Value) -> Option<gen_core::LoadShape> {
    match turbo_fit.get("loadShape")?.as_str()? {
        "eager_materialization" => Some(gen_core::LoadShape::EagerMaterialization),
        "deferred_materialization" => Some(gen_core::LoadShape::DeferredMaterialization),
        _ => None,
    }
}

fn krea_rung_phase_peaks(
    manifest_entry: &JsonObject,
    tier: &str,
    strategy: gen_core::MemoryStrategy,
    width: u32,
    height: u32,
) -> Option<KreaTurboPhasePeaks> {
    let turbo_fit = manifest_entry.get("candle")?.get("turboFit")?;
    let pixels = u64::from(width).checked_mul(u64::from(height))?;
    let max_measured_pixels = turbo_fit.get("maxMeasuredPixels")?.as_u64()?;
    if pixels > max_measured_pixels {
        return None;
    }
    let rung = turbo_fit
        .get("phaseCurvesByTier")?
        .get(tier)?
        .get(krea_turbo_manifest_key(strategy))?
        .as_object()?;
    let phase = |name: &str| {
        rung.get(name)
            .and_then(Value::as_object)
            .and_then(|curve| krea_phase_curve(curve, pixels))
    };
    Some(KreaTurboPhasePeaks {
        text_gb: phase("text")?,
        denoise_gb: phase("denoise")?,
        decode_gb: phase("decode")?,
    })
}

/// The per-phase predicted peaks a measured evidence record declares for one rung — the measured
/// basis a fitted-curve estimate extrapolates from (sc-18097). `None` when the record does not
/// carry a complete phase triple for the rung, which fails the estimate closed.
fn krea_record_phase_peaks(record: &Value, manifest_rung: &str) -> Option<KreaTurboPhasePeaks> {
    let phases = record.get("predictedPhasesGb")?.get(manifest_rung)?;
    Some(KreaTurboPhasePeaks {
        text_gb: phases.get("text").and_then(json_f64)?,
        denoise_gb: phases.get("denoise").and_then(json_f64)?,
        decode_gb: phases.get("decode").and_then(json_f64)?,
    })
}

/// The phase index (0 text, 1 denoise, 2 decode) carrying the peak of a phase triple (sc-18097).
/// Ties resolve to the LATER phase deterministically, mirroring the MLX gate's `binding_phase`
/// (sc-18096); the comparison below only ever contrasts two triples produced by the same per-phase
/// curves, so tie handling cannot manufacture a flip on its own.
fn krea_binding_phase(peaks: KreaTurboPhasePeaks) -> u8 {
    let mut phase = 0_u8;
    let mut peak = peaks.text_gb;
    if peaks.denoise_gb >= peak {
        phase = 1;
        peak = peaks.denoise_gb;
    }
    if peaks.decode_gb >= peak {
        phase = 2;
    }
    phase
}

fn krea_rung_parameters(
    turbo_fit: &Value,
    strategy: gen_core::MemoryStrategy,
) -> Option<gen_core::MemoryStrategyParameters> {
    let parameters = turbo_fit
        .get("strategyParameters")?
        .get(krea_turbo_manifest_key(strategy))?
        .as_object()?;
    let value = |name| {
        parameters
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
    };
    Some(gen_core::MemoryStrategyParameters {
        decode_tile_edge: value("decodeTileEdge"),
        decode_overlap: value("decodeOverlap"),
        attention_chunk_size: value("attentionChunkSize"),
        transformer_window_size: value("transformerWindowSize"),
        // Rung 4's window scope (SC-15794). `Dit` is the only scope that is correct here:
        // `candle-gen-krea` leaves `transformer_window_components` empty, which the contract reads
        // as the DiT-only pre-SC-15794 behaviour, and every measured `strategyParameters` row in
        // the manifest was collected against DiT block streaming. It is also the published default
        // SC-15794 upheld -- a text-encoder scope cut z-image conditioning 46.5% but moved the
        // request peak 0.0%, so widening the scope buys an admission gate nothing.
        //
        // The scope is declared only on the rung that owns it. `validate_selected_parameters`
        // rejects any `Some(..)` below `BoundedTransformerResidency` as "irrelevant: the selection
        // does not engage its owning strategy rung", and the three cheaper rungs here pair with
        // cheaper strategies
        // (see `rung_pairs`); a blanket `Some` would fail `validate_selection` on three of the
        // four candidates and silently downgrade their evidence verdicts. `None` on those rungs
        // carries the identical DiT meaning without tripping that check.
        transformer_window_component: (strategy
            == gen_core::MemoryStrategy::BoundedTransformerResidency)
            .then_some(gen_core::TransformerComponent::Dit),
    })
}

/// Select the least-cost measured Krea Turbo fit rung for this tier, geometry, and live budget.
/// `allow_streamed_blocks` is false when the job carries load-time adapters: the provider preserves
/// those jobs on its existing resident/staged paths rather than silently omitting their residuals.
///
/// Returns `None` when live budget or complete measured manifest evidence is absent. This is distinct
/// from `Reject`: unknown evidence must not masquerade as proof that a configuration cannot fit.
///
/// Since sc-18097 (epic 18093 R1b) an in-envelope request geometry with no exact measured record
/// no longer freezes to `Unverified`: each optimized rung carries a fitted-curve ESTIMATE
/// candidate anchored to a verified measured record, graded by the shared selector behind the
/// candle estimate margin — see the synthesis block below for the anchoring and binding-phase
/// rules.
pub(crate) fn krea_turbo_fit_with_runtime(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    allow_streamed_blocks: bool,
    runtime: Option<&KreaRuntimeEvidenceContext>,
) -> Option<KreaTurboFit> {
    use crate::memory_strategy::{self, Budget, Candidate, RequestScope, Selection};
    use gen_core::{
        MemoryConformanceState, MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey,
        MemoryEvidenceVerdict, MemoryGeometry, MemoryParityContract, MemoryParityResult,
        MemorySelection, MemoryStrategy, MemoryStrategyParameters,
    };

    let budget = budget?;
    let turbo_fit = manifest_entry.get("candle")?.get("turboFit")?;
    let calibration_fingerprint = turbo_fit.get("calibrationFingerprint")?.as_str()?;
    let calibration_abi = turbo_fit.get("calibrationAbi")?.as_u64()? as u32;
    // sc-17097: calibration ABI 2 added the typed load shape, but this route never read it - the
    // worker took the shape from the provider alone, so the axis could not detect drift here. The
    // manifest now states the shape its curves were MEASURED under and the handshake below compares
    // the two; eager and deferred measurements are not interchangeable.
    let declared_load_shape = krea_turbo_load_shape(turbo_fit)?;
    let scene_works_revision = turbo_fit.get("sceneWorksRevision")?.as_str()?;
    let inference_revision = turbo_fit.get("inferenceRevision")?.as_str()?;
    let max_pixels = turbo_fit.get("maxMeasuredPixels")?.as_u64()?;
    let geometry = MemoryGeometry {
        width,
        height,
        batch: 1,
        frames: 1,
        reference_count: 0,
    };
    let pixels = u64::from(width).checked_mul(u64::from(height))?;
    if pixels > max_pixels {
        return Some(KreaTurboFit::Unverified {
            reason: MemoryEvidenceVerdict::OutOfEnvelope,
        });
    }
    // Runtime artifact identity is a required evidence dimension. Preserve the explicit unverified
    // result when it is unavailable instead of attempting provider source validation against an
    // empty path and collapsing to `None`; callers keep the same resident-or-reject fallback, while
    // diagnostics retain the truthful reason.
    if runtime.is_none() {
        return Some(KreaTurboFit::Unverified {
            reason: MemoryEvidenceVerdict::Unverified,
        });
    }
    let load_root = runtime
        .map(|runtime| runtime.resolved_artifact_root.clone())
        .unwrap_or_default();
    let provider_contract = crate::inference_runtime::media()
        .memory_strategy_contract(
            "krea_2_turbo",
            &gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(load_root)),
        )
        .ok()
        .flatten()?;
    let numeric_tier = gen_core::MemoryNumericTier {
        precision: gen_core::Precision::Bf16,
        quant: match tier {
            "q4" => Some(gen_core::Quant::Q4),
            "q8" => Some(gen_core::Quant::Q8),
            "bf16" => None,
            _ => return None,
        },
        component_precision_floors: &[],
    };
    let measured_closure_digest = turbo_fit
        .get("inferenceClosureDigest")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let live_closure_digest =
        sceneworks_core::memory_calibration::packaged_closure_digest("candle", "krea_2_turbo")
            .unwrap_or_default();
    let request = RequestScope {
        resolved_route: "krea_2_turbo",
        backend: "candle",
        tier: numeric_tier,
        mode: "text_to_image",
        overlay: (!allow_streamed_blocks).then_some("adapter"),
        geometry,
        // sc-17774: one mechanism, read not frozen.
        expected_closure_digest: &live_closure_digest,
    };
    let resident_peak_gb = manifest_entry
        .get("candle")?
        .get("vramGbByTier")?
        .get(tier)
        .and_then(json_f64)?;
    if !resident_peak_gb.is_finite() || resident_peak_gb < 0.0 {
        return None;
    }
    let resident_parameters = turbo_fit
        .get("strategyParameters")?
        .get("resident")?
        .as_object()?;
    if !resident_parameters.is_empty() {
        return None;
    }
    let bytes = |gb: f64| (gb * BYTES_PER_GIB).round().clamp(0.0, u64::MAX as f64) as u64;
    let tier_records = turbo_fit
        .get("evidenceRecords")?
        .as_array()?
        .iter()
        .filter(|record| {
            record.get("evidenceScope").and_then(Value::as_str) == Some("exact_request")
                && record.get("tier").and_then(Value::as_str) == Some(tier)
        })
        .filter_map(|record| {
            let record_width = u32::try_from(record.get("width")?.as_u64()?).ok()?;
            let record_height = u32::try_from(record.get("height")?.as_u64()?).ok()?;
            Some((record, record_width, record_height))
        })
        .collect::<Vec<_>>();
    let evidence_record = tier_records
        .iter()
        .find(|(_, record_width, record_height)| *record_width == width && *record_height == height)
        .map(|(record, _, _)| *record);
    // `record`/`at_geometry` are parameters rather than captures (sc-18097): the request cell's
    // candidates are graded against the request's own record and geometry, while the estimate
    // synthesis below re-grades a DIFFERENT record at its own measured geometry to decide whether
    // it is a verified extrapolation anchor.
    let make_evidence = |selection: MemorySelection,
                         manifest_rung: Option<&str>,
                         fallback_predicted_peak_gb: f64,
                         fallback_phases: Option<KreaTurboPhasePeaks>,
                         evidence_record: Option<&Value>,
                         at_geometry: MemoryGeometry| {
        let record_peak = |field: &str| {
            let rung = manifest_rung?;
            evidence_record?.get(field)?.get(rung).and_then(json_f64)
        };
        let record_phase = |phase: &str| {
            let rung = manifest_rung?;
            evidence_record?
                .get("predictedPhasesGb")?
                .get(rung)?
                .get(phase)
                .and_then(json_f64)
        };
        let engaged_composition = manifest_rung.map_or_else(
            || provider_contract.engaged_composition(selection.strategy),
            |rung| match evidence_record {
                Some(record) => record
                    .get("measuredCompositions")
                    .and_then(|compositions| compositions.get(rung))
                    .and_then(Value::as_array)
                    .and_then(|composition| {
                        composition
                            .iter()
                            .map(|rung| match rung.as_str()? {
                                "resident" => Some(MemoryStrategy::Resident),
                                "staged_residency" => Some(MemoryStrategy::StagedResidency),
                                "bounded_decode" => Some(MemoryStrategy::BoundedDecode),
                                "bounded_attention" => Some(MemoryStrategy::BoundedAttention),
                                "bounded_transformer_residency" => {
                                    Some(MemoryStrategy::BoundedTransformerResidency)
                                }
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .unwrap_or_default(),
                None => provider_contract.engaged_composition(selection.strategy),
            },
        );
        let predicted_peak_gb =
            record_peak("predictedPeaksGb").unwrap_or(fallback_predicted_peak_gb);
        let observed_peak_gb = record_peak("observedPeaksGb");
        let harness_version = evidence_record
            .and_then(|record| record.get("harnessVersion"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let parity = evidence_record
            .and_then(|record| record.get("parity"))
            .and_then(Value::as_object);
        let verdict = |condition, failure| {
            if condition {
                MemoryEvidenceVerdict::Satisfied
            } else {
                failure
            }
        };
        let static_valid = provider_contract.conformance_errors().is_empty()
            && provider_contract.validate_selection(&selection).is_ok();
        let declared_calibration = provider_contract
            .calibration
            .as_ref()
            .is_some_and(|identity| {
                identity.abi == calibration_abi
                    && identity.fingerprint == calibration_fingerprint
                    && identity.load_shape == declared_load_shape
            });
        let valid_commit = |field: &str| {
            evidence_record
                .and_then(|record| record.get(field))
                .and_then(Value::as_str)
                .is_some_and(|revision| {
                    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
                })
        };
        // sc-17774: `compatibleInferenceRevision` is gone. It was a per-record hand-declared "this
        // capture is also valid at revision X" claim — the same one-shot hatch `flux2_dev` had, in a
        // second spelling. Compatibility is now decided by the lane's closure digest below.
        let historical_compatible = valid_commit("sceneWorksCommit")
            && valid_commit("inferenceCommit")
            && evidence_record
                .and_then(|record| record.get("compatibleSceneWorksRevision"))
                .and_then(Value::as_str)
                == Some(KREA_TURBO_SCENEWORKS_REVISION);
        let loadability = evidence_record
            .and_then(|record| record.get("loadability"))
            .and_then(Value::as_object);
        let expected_compute_capability = loadability
            .and_then(|loadability| loadability.get("computeCapability"))
            .and_then(json_f64);
        // sc-17774: the lane's own compile closure, not a frozen inference SHA. `inference_revision`
        // stays parsed above as capture provenance for the receipt.
        let current_environment = scene_works_revision == KREA_TURBO_SCENEWORKS_REVISION
            && turbo_fit
                .get("inferenceClosureDigest")
                .and_then(Value::as_str)
                .is_some_and(|declared| {
                    sceneworks_core::memory_calibration::packaged_closure_digest(
                        "candle",
                        "krea_2_turbo",
                    )
                    .is_some_and(|live| live == declared)
                })
            && turbo_fit.get("measured").and_then(Value::as_bool) == Some(true)
            && runtime.is_some_and(|runtime| {
                !runtime.gpu_id.trim().is_empty()
                    && expected_compute_capability.is_some_and(|expected| {
                        (f64::from(runtime.compute_capability) - expected).abs() < f64::EPSILON
                    })
            });
        let loadability_matches_record =
            loadability
                .zip(runtime)
                .is_some_and(|(loadability, runtime)| {
                    loadability.get("provider").and_then(Value::as_str)
                        == Some(runtime.artifact_provider.as_str())
                        && loadability.get("repository").and_then(Value::as_str)
                            == Some(runtime.artifact_repository.as_str())
                        && loadability.get("resolvedRevision").and_then(Value::as_str)
                            == Some(runtime.resolved_revision.as_str())
                        && loadability.get("tierRoot").and_then(Value::as_str)
                            == Some(runtime.tier_root.as_str())
                        && loadability.get("route").and_then(Value::as_str)
                            == Some(runtime.resolved_route.as_str())
                        && loadability.get("backend").and_then(Value::as_str)
                            == Some(runtime.backend.as_str())
                        && runtime.resolved_route == provider_contract.provider_id
                        && runtime.backend == provider_contract.backend.backend_id()
                        && runtime.tier_root == tier
                        && manifest_entry
                            .get("downloads")
                            .and_then(Value::as_array)
                            .is_some_and(|downloads| {
                                downloads.iter().any(|download| {
                                    download.get("provider").and_then(Value::as_str)
                                        == loadability.get("provider").and_then(Value::as_str)
                                        && download.get("repo").and_then(Value::as_str)
                                            == loadability.get("repository").and_then(Value::as_str)
                                        && download.get("revision").and_then(Value::as_str)
                                            == loadability
                                                .get("resolvedRevision")
                                                .and_then(Value::as_str)
                                        && download.get("variant").and_then(Value::as_str)
                                            == Some(tier)
                                })
                            })
                });
        let phases_match = fallback_phases.is_some_and(|phases| {
            [
                ("text", phases.text_gb),
                ("denoise", phases.denoise_gb),
                ("decode", phases.decode_gb),
            ]
            .into_iter()
            .all(|(name, predicted)| {
                record_phase(name).is_some_and(|recorded| (recorded - predicted).abs() <= 0.01)
            })
        });
        let exact_parameters = manifest_rung.is_some()
            && record_peak("predictedPeaksGb").is_some_and(|record_peak_gb| {
                (record_peak_gb - fallback_predicted_peak_gb).abs() <= 0.01
            })
            && phases_match
            && provider_contract.validate_selection(&selection).is_ok();
        let evidence_dimensions = MemoryEvidenceDimensions {
            static_implementation: verdict(static_valid, MemoryEvidenceVerdict::Invalid),
            declared_calibration: verdict(
                declared_calibration,
                MemoryEvidenceVerdict::FingerprintMismatch,
            ),
            historical_verification: verdict(
                historical_compatible,
                if evidence_record.is_some() {
                    MemoryEvidenceVerdict::Stale
                } else {
                    MemoryEvidenceVerdict::Missing
                },
            ),
            current_environment_verification: verdict(
                current_environment,
                if evidence_record.is_some() {
                    MemoryEvidenceVerdict::Stale
                } else {
                    MemoryEvidenceVerdict::Missing
                },
            ),
            canonical_route_loadability: verdict(
                loadability_matches_record,
                if evidence_record.is_some() {
                    MemoryEvidenceVerdict::Unverified
                } else {
                    MemoryEvidenceVerdict::Missing
                },
            ),
            exact_strategy_parameters: verdict(
                exact_parameters,
                MemoryEvidenceVerdict::OutOfEnvelope,
            ),
        };
        let verified = observed_peak_gb.is_some()
            && !harness_version.is_empty()
            && parity.is_some_and(|parity| {
                parity.get("contract").and_then(Value::as_str) == Some("golden")
                    && parity.get("result").and_then(Value::as_str) == Some("passed")
            })
            && evidence_dimensions.all_satisfied();
        let parity_contract = parity.map_or(MemoryParityContract::Exact, |parity| {
            MemoryParityContract::Golden {
                fixture: parity
                    .get("fixture")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                metric: parity
                    .get("metric")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                maximum_error: parity
                    .get("maximumError")
                    .and_then(json_f64)
                    .unwrap_or_default(),
            }
        });
        MemoryEvidence {
            key: MemoryEvidenceKey {
                resolved_route: "krea_2_turbo".to_owned(),
                backend: gen_core::MemoryBackend::Candle,
                tier: numeric_tier,
                load_shape: provider_contract.load_shape,
                mode: gen_core::MemoryMode::TextToImage,
                // The existing measurements cover ordinary T2I only.
                overlay: None,
                geometry: at_geometry,
                strategy: selection.strategy,
                engaged_composition,
                parameters: selection.parameters,
            },
            conformance: if verified {
                MemoryConformanceState::Verified
            } else {
                MemoryConformanceState::ImplementedUnverified
            },
            dimensions: evidence_dimensions,
            calibration_abi,
            calibration_fingerprint: calibration_fingerprint.to_owned(),
            sceneworks_revision: scene_works_revision.to_owned(),
            inference_revision: inference_revision.to_owned(),
            harness_version: harness_version.to_owned(),
            predicted_peak_bytes: bytes(predicted_peak_gb),
            observed_peak_bytes: observed_peak_gb.map(bytes),
            parity: parity_contract,
            parity_result: if verified {
                MemoryParityResult::Passed
            } else {
                MemoryParityResult::NotRun
            },
        }
    };
    let resident_selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: MemoryStrategyParameters::default(),
        tier: numeric_tier,
    };

    let optimized_strategies = [
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ];
    let mut evidence = vec![make_evidence(
        resident_selection,
        None,
        resident_peak_gb,
        None,
        evidence_record,
        geometry,
    )];
    let mut selections = vec![resident_selection];
    let mut measured = Vec::new();
    for strategy in optimized_strategies {
        let phases = krea_rung_phase_peaks(manifest_entry, tier, strategy, width, height)?;
        let phase_peak_gb = phases.peak_gb();
        let needed_gb = phase_peak_gb + HEADROOM_GB;
        let parameters = krea_rung_parameters(turbo_fit, strategy)?;
        let selection = MemorySelection {
            strategy,
            parameters,
            tier: numeric_tier,
        };
        measured.push((strategy, phases, needed_gb, selection));
        evidence.push(make_evidence(
            selection,
            Some(krea_turbo_manifest_key(strategy)),
            phase_peak_gb,
            Some(phases),
            evidence_record,
            geometry,
        ));
        selections.push(selection);
    }
    // ── sc-18097 (epic 18093 R1b): fitted-curve estimate candidates for unmeasured cells. ──
    //
    // The candle mirror of `mlx_fit_gate::synthesize_estimate_ladder`'s fitted arm. The manifest's
    // per-phase curves ARE the fitted model over this tier's measured cells, so an in-envelope
    // request geometry nobody measured gets an estimate candidate per optimized rung at the
    // curve-predicted peak, graded by the shared selector behind the candle ESTIMATE margin
    // (`crate::ladder_margin_policy::CANDLE_ESTIMATE_MARGIN`). Where the exact request cell has a
    // verified record, the selector's measured-supersedes-estimate rule keeps admission
    // byte-for-byte unchanged.
    //
    // A rung's fitted estimate is emitted only when EVERY `exact_request` record of this tier —
    // graded at ITS OWN geometry through the same `make_evidence` conjuncts as the request path —
    // passes the full measured eligibility predicate (`optimized_eligibility`). Whole-tier rather
    // than best-anchor, because the curve is fitted across every cell (see the fail-closed note at
    // the loop head). That check carries every restriction sc-18096 established for extrapolation
    // bases —
    // closure-CURRENT capture (a stale record may serve its own cell behind the stale margin but
    // may not seed an extrapolation; the estimate margin was derived over same-closure re-capture
    // variance and cannot also absorb closure drift), the loaded contract's calibration identity
    // (a drifted provider must not receive fitted candidates built from another identity's
    // records), artifact + hardware loadability, measured-composition agreement, parity, and
    // curve↔record phase agreement at the anchor cell (so a mutated curve that no longer describes
    // the measurement cannot smuggle its numbers back in as an estimate).
    //
    // `ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE` (sc-18094) is honored at this synthesis
    // seam: if the curve moves the request peak onto a different phase than the anchor record's
    // binding phase, the fitted candidate is NOT emitted. No weights+headroom floor path exists on
    // this lane (the constraint's scope exemption is therefore never exercised here): every rung
    // always has a curve, and a floor from the resident row could never admit deeper than the
    // resident baseline it equals, so a suppressed rung honestly falls out of estimate admission.
    let mut estimates: Vec<(MemorySelection, MemoryEvidence)> = Vec::new();
    for (strategy, phases, _, selection) in &measured {
        let manifest_rung = krea_turbo_manifest_key(*strategy);
        let record_is_eligible = |record: &Value, record_width: u32, record_height: u32| {
            let Some(anchor_phases) =
                krea_rung_phase_peaks(manifest_entry, tier, *strategy, record_width, record_height)
            else {
                return false;
            };
            let anchor_geometry = MemoryGeometry {
                width: record_width,
                height: record_height,
                batch: 1,
                frames: 1,
                reference_count: 0,
            };
            // The FULL measured-eligibility predicate, not just `Verified` conformance: it
            // additionally requires the record's measured composition to agree with the loaded
            // contract's and the calibration identity to match (the mirror of
            // `mlx_fit_gate::collect_estimate_bases`' engaged-composition filter).
            make_evidence(
                *selection,
                Some(manifest_rung),
                anchor_phases.peak_gb(),
                Some(anchor_phases),
                Some(record),
                anchor_geometry,
            )
            .optimized_eligibility(&provider_contract)
            .is_ok()
        };
        // FAIL CLOSED ON THE WHOLE TIER (sc-18097 review, major finding). The tier's phase curves
        // are fitted across EVERY measured cell, and the measured path's own
        // `exact_strategy_parameters` conjunct requires the record peak and the curve to agree
        // within 0.01 GiB — so the curve evaluated at an excluded cell's geometry reproduces that
        // cell's number. Anchoring on a surviving SIBLING record would therefore let a
        // structurally excluded cell (composition mismatch, drifted identity, stale capture) be
        // re-admitted at its own numbers behind nothing but the 4% estimate margin, laundering
        // the per-cell structural exclusion the epic requires to keep excluding. So a rung emits
        // fitted candidates only when EVERY `exact_request` record of the tier is eligible for it:
        // one bad row disqualifies the tier's curve for that rung, not merely that row.
        if let Some((_, bad_width, bad_height)) = tier_records
            .iter()
            .find(|(record, width, height)| !record_is_eligible(record, *width, *height))
        {
            tracing::info!(
                route = "krea_2_turbo",
                backend = "candle",
                ?strategy,
                ineligible_record_geometry = format!("{bad_width}x{bad_height}"),
                "fitted-curve estimates suppressed for this rung: a measured record of this tier \
                 is structurally excluded, and the tier's curve is fitted across it"
            );
            continue;
        }
        let anchor = tier_records
            .iter()
            .copied()
            // Prefer the largest measured cell: the curve is fitted within `maxMeasuredPixels`,
            // and the top sample is the anchor closest to that envelope.
            .max_by_key(|(_, record_width, record_height)| {
                u64::from(*record_width) * u64::from(*record_height)
            });
        let Some((anchor_record, anchor_width, anchor_height)) = anchor else {
            continue;
        };
        let measured_binding =
            krea_record_phase_peaks(anchor_record, manifest_rung).map(krea_binding_phase);
        let request_binding = krea_binding_phase(*phases);
        if crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE
            && measured_binding != Some(request_binding)
        {
            // The pinned sc-18094 constraint: the corpus shows a per-phase re-capture spread no
            // margin in the policy absorbs, so a curve that moves the request peak onto a
            // different phase than the one the anchor measured is refused rather than margined.
            tracing::info!(
                route = "krea_2_turbo",
                backend = "candle",
                ?strategy,
                anchor_geometry = format!("{anchor_width}x{anchor_height}"),
                ?measured_binding,
                request_binding,
                "fitted-curve estimate rejected: the curve moves the request peak onto a \
                 different phase than the measured anchor's \
                 (ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE)"
            );
            continue;
        }
        let predicted_peak_bytes = bytes(phases.peak_gb());
        tracing::info!(
            route = "krea_2_turbo",
            backend = "candle",
            ?strategy,
            anchor_geometry = format!("{anchor_width}x{anchor_height}"),
            raw_peak_bytes = predicted_peak_bytes,
            "synthesized fitted-curve estimate candidate from the anchored phase curves"
        );
        estimates.push((
            *selection,
            MemoryEvidence {
                key: MemoryEvidenceKey {
                    resolved_route: "krea_2_turbo".to_owned(),
                    backend: gen_core::MemoryBackend::Candle,
                    tier: numeric_tier,
                    load_shape: provider_contract.load_shape,
                    mode: gen_core::MemoryMode::TextToImage,
                    overlay: None,
                    geometry,
                    strategy: selection.strategy,
                    engaged_composition: provider_contract.engaged_composition(selection.strategy),
                    parameters: selection.parameters,
                },
                conformance: MemoryConformanceState::ImplementedUnverified,
                dimensions: MemoryEvidenceDimensions {
                    static_implementation: MemoryEvidenceVerdict::Satisfied,
                    declared_calibration: MemoryEvidenceVerdict::Missing,
                    historical_verification: MemoryEvidenceVerdict::Missing,
                    current_environment_verification: MemoryEvidenceVerdict::Missing,
                    canonical_route_loadability: MemoryEvidenceVerdict::Unverified,
                    exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
                },
                calibration_abi,
                calibration_fingerprint: calibration_fingerprint.to_owned(),
                sceneworks_revision: scene_works_revision.to_owned(),
                inference_revision: inference_revision.to_owned(),
                harness_version: String::new(),
                predicted_peak_bytes,
                observed_peak_bytes: None,
                parity: MemoryParityContract::Exact,
                parity_result: MemoryParityResult::NotRun,
            },
        ));
    }
    let mut candidates = selections
        .iter()
        .zip(&evidence)
        .map(|(selection, evidence)| Candidate {
            selection: *selection,
            evidence,
            closure_digest: &measured_closure_digest,
            basis: memory_strategy::CandidateBasis::Measured,
        })
        .collect::<Vec<_>>();
    // Synthesized under (and anchored to) the live closure — there is nothing for currency to
    // invalidate, exactly like the MLX gate's synthesized candidates (sc-18096).
    candidates.extend(estimates.iter().map(|(selection, evidence)| Candidate {
        selection: *selection,
        evidence,
        closure_digest: &live_closure_digest,
        basis: memory_strategy::CandidateBasis::EstimateFittedCurve,
    }));
    let selection = memory_strategy::select_strategy(
        request,
        &provider_contract,
        Some(Budget {
            available_gb: budget.free_gb,
            reclaimable_gb: 0.0,
            total_gb: budget.total_gb,
            reserved_headroom_gb: HEADROOM_GB,
        }),
        &candidates,
    );
    match selection {
        Selection::Selected {
            selection:
                selected @ MemorySelection {
                    strategy: MemoryStrategy::Resident,
                    ..
                },
            needed_gb,
            ..
        } => Some(KreaTurboFit::Resident {
            peak_gb: resident_peak_gb,
            needed_gb: needed_gb + HEADROOM_GB,
            selection: selected,
        }),
        Selection::Selected {
            selection: selected,
            needed_gb,
            ..
        } => {
            let (_, phases, _, _) = measured
                .into_iter()
                .find(|(_, _, _, selection)| selection.strategy == selected.strategy)?;
            let memory = gen_core::GenerationMemory {
                tile_vae_decode: provider_contract
                    .engages(selected.strategy, MemoryStrategy::BoundedDecode),
                chunk_attention: provider_contract
                    .engages(selected.strategy, MemoryStrategy::BoundedAttention),
                stream_transformer_blocks: provider_contract.engages(
                    selected.strategy,
                    MemoryStrategy::BoundedTransformerResidency,
                ),
                ..Default::default()
            };
            Some(KreaTurboFit::Fits {
                phases,
                needed_gb: needed_gb + HEADROOM_GB,
                selection: selected,
                memory,
                // See the field doc: with no `exact_request` record at this geometry the rung's
                // measured candidates are structurally excluded, so only a synthesized estimate
                // can have carried it.
                estimate_scoped: evidence_record.is_none(),
            })
        }
        Selection::Reject { needed_gb, .. } => {
            let (_, phases, _, _) = measured.last().copied()?;
            Some(KreaTurboFit::Reject {
                phases,
                needed_gb: needed_gb + HEADROOM_GB,
            })
        }
        // sc-18097 narrowed this arm's meaning: an in-envelope unmeasured geometry now carries a
        // fitted-curve estimate per rung, so the selector lands here only when a rung has NO
        // eligible candidate at all — a stale-closure or otherwise unverifiable manifest (whose
        // records may not seed extrapolation), a mutated record/curve pair, a binding-phase flip,
        // or an overlay the measurements do not cover. Those remain the explicit fallback to the
        // established generic gate, exactly as before.
        Selection::Unverified { reason } => Some(KreaTurboFit::Unverified { reason }),
    }
}

#[cfg(test)]
fn krea_turbo_fit(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    allow_streamed_blocks: bool,
) -> Option<KreaTurboFit> {
    let runtime = KreaRuntimeEvidenceContext::verified_for_test(tier);
    krea_turbo_fit_with_runtime(
        manifest_entry,
        tier,
        width,
        height,
        budget,
        allow_streamed_blocks,
        Some(&runtime),
    )
}

/// Highest lower-pixel manifest bucket that the deepest available Krea Turbo rung can actually fit.
/// Used only to make rejection copy truthful: if this returns `None`, lowering resolution is not
/// presented as an escape hatch.
///
/// The advice is restricted to MEASURED admissions (sc-18097): an estimate-scoped `Fits` — the
/// fitted-curve admission of a geometry nobody measured — is not offered here, mirroring
/// `mlx_fit_gate::verified_lower_alternative` ("no formula, interpolation, tier heuristic, or
/// aspect-ratio rewrite is admitted"). Naming a fallback that itself rests on an estimate invites
/// a second refusal at the geometry the user was just told to switch to. `Resident` keeps its
/// pre-sc-18097 acceptance: that verdict is the manifest's geometry-independent `vramGbByTier`
/// row, which this helper has always been allowed to quote.
pub(crate) fn krea_turbo_smaller_fit_with_runtime(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    allow_streamed_blocks: bool,
    runtime: Option<&KreaRuntimeEvidenceContext>,
) -> Option<(u32, u32)> {
    let current_pixels = u64::from(width) * u64::from(height);
    let resolutions = manifest_entry
        .get("limits")?
        .get("resolutions")?
        .as_array()?;
    let mut candidates = resolutions
        .iter()
        .filter_map(Value::as_str)
        .filter_map(|resolution| resolution.split_once('x'))
        .filter_map(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        .filter(|(w, h)| u64::from(*w) * u64::from(*h) < current_pixels)
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(w, h)| std::cmp::Reverse(u64::from(*w) * u64::from(*h)));
    candidates.into_iter().find(|(w, h)| {
        matches!(
            krea_turbo_fit_with_runtime(
                manifest_entry,
                tier,
                *w,
                *h,
                budget,
                allow_streamed_blocks,
                runtime,
            ),
            Some(
                KreaTurboFit::Resident { .. }
                    | KreaTurboFit::Fits {
                        estimate_scoped: false,
                        ..
                    }
            )
        )
    })
}

#[cfg(test)]
fn krea_turbo_smaller_fit(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    allow_streamed_blocks: bool,
) -> Option<(u32, u32)> {
    let runtime = KreaRuntimeEvidenceContext::verified_for_test(tier);
    krea_turbo_smaller_fit_with_runtime(
        manifest_entry,
        tier,
        width,
        height,
        budget,
        allow_streamed_blocks,
        Some(&runtime),
    )
}

/// Resolve the [`LoadPlan`] for `needed` (resident peak) and `sequential_needed` (measured staged peak,
/// [`predicted_sequential_peak_gb`]) against `budget`. `None` peaks are unmeasured ⇒ never block (admit
/// resident, or stage best-effort). `sequential_capable` is the provider's staging capability: a lane
/// that cannot stage turns a resident overflow straight into [`LoadPlan::Reject`].
///
/// Monotonic in the budget: a larger `free_gb` can only move the plan `Reject → Sequential → Resident`,
/// never the other way — which is what makes "plan changed after reclaim ⇒ plan improved" sound.
pub(crate) fn load_plan(
    needed: Option<f64>,
    sequential_needed: Option<f64>,
    budget: Option<VramBudget>,
    sequential_capable: bool,
) -> LoadPlan {
    match resolve_offload(fit_decision(needed, budget), sequential_capable) {
        FitDecision::Offload { .. } => {
            if sequential_overflow_gb(sequential_needed, budget).is_some() {
                LoadPlan::Reject
            } else {
                LoadPlan::Sequential
            }
        }
        // Reached only when the engine cannot stage (resolve_offload leaves TooBig intact); a
        // sequential-capable lane rewrites TooBig → Offload above.
        FitDecision::TooBig { .. } => LoadPlan::Reject,
        // Fits (resident) or Unknown (unmeasured / no budget) — admit resident, never block.
        _ => LoadPlan::Resident,
    }
}

// ---------------------------------------------------------------------------------------------
// Mochi 1: the FRAME-DEPENDENT decode fit gate, candle/CUDA half (epic 1788 / sc-12306)
// ---------------------------------------------------------------------------------------------
//
// The candle twin of `mlx_fit_gate`'s Mochi gate. Both call the SAME arithmetic
// (`crate::fit_gate::mochi_needed_gb`); this half supplies the CUDA budget, the CUDA reserve, and the
// CUDA-worded message.
//
// Why Mochi needs its own gate here rather than riding `fit_decision` above:
//
//  1. `predicted_peak_gb` is MANIFEST-driven (`candle.vramGbByTier`), and **`mochi_1` has no `candle`
//     block at all** — it ships `mlx` only (builtin.models.jsonc:7233). So the generic gate returns
//     `None` ⇒ `Unknown` ⇒ admit. It could not protect Mochi even if the video lane called it. (No
//     candle VIDEO model has such a block; closing that is sc-12344.)
//
//  2. Even given a block, the generic gate is deliberately RESOLUTION-BLIND: a per-tier constant
//     calibrated at load time, where the seam cannot see the request geometry (the generator is cached
//     across resolutions). Mochi's AsymmVAE decode is UNTILED (candle-gen-mochi pipeline.rs:151 —
//     `vae.decode(&latents)` materializes the whole clip; sc-12291), so its peak grows LINEARLY IN CLIP
//     LENGTH: a 7-frame and a 151-frame request differ by ~55 GiB on the same model and the same tier.
//     A constant cannot express that.
//
//  3. Unlike the MLX lane, a CUDA OOM here is CATCHABLE (`classify_engine_error`) rather than MLX's
//     unmappable `exit(-1)` (sc-12178/12179). So this gate is not preventing a process kill — it is
//     preventing a raw allocator error that arrives only AFTER the full 64-step denoise has burned
//     minutes of GPU time, and that says nothing about the one lever that actually fixes it. At the
//     shipped 5 s / 151-frame default Mochi needs ~81 GB; an RTX 5090 has 32 GB, so on consumer
//     hardware this is not an edge case but the DEFAULT path.

/// The pure candle Mochi admission decision: `Some(error)` when the predicted peak overflows the VRAM
/// budget, `None` to admit. Missing either signal (unmeasurable weights / no budget) admits — the gate
/// never blocks without evidence, exactly like [`fit_decision`].
///
/// Budgets against `free_gb` (what is allocatable now), like [`fit_decision`] and unlike the MLX gate,
/// which has only a unified `total_gb`. Reserves [`HEADROOM_GB`] rather than the MLX gate's
/// the MLX unified reserve: the OS does not draw from discrete VRAM, so the term covers allocator slack and
/// CUDA context overhead instead. The two constants agree at 2.0 today but mean different things.
///
/// Pure (no GPU probe, no env) so the whole decision is unit-testable without CUDA; the caller resolves
/// the budget.
pub(crate) fn mochi_fit_error(
    model_label: &str,
    weight_bytes: u64,
    frames: u32,
    width: u32,
    height: u32,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    let (needed_gb, budget) = (
        crate::fit_gate::mochi_needed_gb(weight_bytes, frames, width, height, HEADROOM_GB)?,
        budget?,
    );
    (budget.free_gb + f64::EPSILON < needed_gb).then(|| {
        mochi_too_big_error(
            model_label,
            needed_gb,
            budget.free_gb,
            frames,
            width,
            height,
            weight_bytes as f64 / BYTES_PER_GIB,
            gpu_id,
        )
    })
}

/// Build Mochi's actionable over-budget rejection for the candle lane. Follows this module's existing
/// reject convention — name the model, state what it needs and what the GPU has — and adds the lever
/// that is UNIQUE to Mochi: the clip length.
///
/// The generic candle message's advice ("choose a smaller quant tier, lower the resolution") is nearly
/// useless here: Mochi has one trained bucket (848×480), and the decode dwarfs the tier delta (q4→bf16
/// is ~11 GiB against a ~60 GiB decode), so neither lever closes a 49 GB gap. The message therefore
/// leads with shortening the clip — the only lever that moves the dominant term.
///
/// Deliberately does NOT reuse `mlx_fit_gate::mochi_too_big_error`: that one says "unified memory" and
/// "run on a Mac with more memory", both false on CUDA. The shared thing between the lanes is the
/// arithmetic, not the prose.
#[allow(clippy::too_many_arguments)]
fn mochi_too_big_error(
    model_label: &str,
    needed_gb: f64,
    available_gb: f64,
    frames: u32,
    width: u32,
    height: u32,
    weights_gb: f64,
    gpu_id: &str,
) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{model_label} needs ~{needed} GB of VRAM to render a {frames}-frame {width}x{height} clip \
         (~{weights} GB of weights, held resident for the whole run, plus an untiled VAE decode whose \
         peak grows with clip length) but GPU {gpu_id} has ~{available} GB available. Shorten the \
         clip — the decode peak scales roughly linearly with duration — or run on a card with more \
         VRAM.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
        weights = weights_gb.round() as i64,
    ))
}

// ---------------------------------------------------------------------------------------------
// SVD candle-video long-burst gate (sc-14492)
// ---------------------------------------------------------------------------------------------

/// Pure SVD CUDA admission decision for the candle lane.
///
/// A 32 GB RTX PRO 4500 completed the supported/default 25-frame, decode-chunk-8, 25-step 1024x576
/// render after sc-14625's sequential CFG/residency and live-free VAE tiling changes. That complete
/// measured profile (or lower settings) is admitted on 32 GB; requests beyond any measured dimension
/// remain rejected on that card class. Missing telemetry admits, matching every other fit gate: no
/// invented signal means no rejection.
///
/// The check keys off physical `total_gb`, not momentary `free_gb`. This story establishes a hardware
/// boundary, not a trustworthy transient-working-set threshold for a busy card.
pub(crate) fn svd_fit_error(
    frames: u32,
    decode_chunk_size: u32,
    steps: u32,
    width: u32,
    height: u32,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    let budget = budget?;
    if !crate::fit_gate::svd_profile_needs_larger_card(
        frames,
        decode_chunk_size,
        steps,
        width,
        height,
        budget.total_gb,
    ) {
        return None;
    }
    Some(WorkerError::InvalidPayload(format!(
        "Stable Video Diffusion's {frames}-frame, decode-chunk-{decode_chunk_size}, {steps}-step \
         {width}x{height} CUDA profile is outside the measured admission envelope on \
         GPU {gpu_id}'s {total} GB VRAM class. A real 32 GB RTX PRO 4500 completed up to \
         {validated_width}x{validated_height}, {validated_frames} frames, \
         decodeChunkSize={validated_chunk}, and {validated_steps} steps at a 17.521 GiB peak, so \
         even that bounded profile requires at least an {minimum} GB physical-card class. Requests \
         beyond any measured dimension remain unvalidated through 32 GB. Use a card and recipe \
         inside that measured envelope, or choose a larger GPU. The job was rejected before model \
         load.",
        total = budget.total_gb.round() as i64,
        validated_width = crate::fit_gate::SVD_32GB_VALIDATED_MAX_WIDTH,
        validated_height = crate::fit_gate::SVD_32GB_VALIDATED_MAX_HEIGHT,
        validated_frames = crate::fit_gate::SVD_32GB_VALIDATED_MAX_FRAMES,
        validated_chunk = crate::fit_gate::SVD_32GB_VALIDATED_MAX_DECODE_CHUNK,
        validated_steps = crate::fit_gate::SVD_32GB_VALIDATED_MAX_STEPS,
        minimum = crate::fit_gate::SVD_VALIDATED_PROFILE_MIN_VRAM_GB.round() as i64,
    )))
}

// ---------------------------------------------------------------------------------------------
// The Wan candle video weights-floor gate (epic 1788 / sc-12344)
// ---------------------------------------------------------------------------------------------
//
// Mochi above gets a bespoke gate because its decode peak is frame-dependent. Every OTHER candle video
// engine was admitted with NO pre-flight check at all until this: `predicted_peak_gb` is manifest-driven
// off `candle.vramGbByTier`, and **no candle video model carries a `candle` block**, so wiring the
// generic gate would return `None` ⇒ `Unknown` ⇒ admit for every one of them — dead code that reads as
// coverage (the failure `tests/gpu_and_manifest.rs`'s flux2 test warns about).
//
// So this budgets on the one signal that exists WITHOUT a measurement campaign: the on-disk weight
// bytes of the components the loader actually reads. Three facts make that a sound admission check for
// **Wan** rather than a guess:
//
//  1. **The engines this FLOOR still gates hold all components resident for the whole run.** ltx / svd /
//     mochi declare `supports_sequential_offload: false` (candle-gen-ltx / -svd / -mochi), so
//     `resolve_offload` is a no-op for them and Σweights is a genuine LOWER BOUND. **The Wan A14B T2V/I2V
//     and the dense TI2V-5B are the exception as of sc-12631 / epic sc-12732 (5B: sc-13175):** the candle
//     engine stages components **one-resident-at-a-time** (`render_sequential`,
//     `supports_sequential_offload: true`) — the A14B swaps TE → high-expert → low-expert → VAE (only ONE
//     14B expert live at a time, not both), the 5B flushes its TE + z48 VAE off-GPU around the dense
//     denoise (sc-12757) — so each is sized by its MEASURED sequential `candle.vramGbByTier` peak and this
//     floor never gates it (`wan_video_fit_error` takes the measured branch), while its load is forced
//     `OffloadPolicy::Sequential` (`video_jobs::candle_wan_offload_policy`) so the sized peak is the one
//     actually loaded. The floor stays the fallback for a Wan tier that was never measured.
//  2. **There is no host paging on CUDA.** Weights that do not fit VRAM cannot be demand-paged the way
//     MLX's unified pool swaps, so Σweights is a genuine LOWER BOUND on the job's need. This is the
//     asymmetry that makes the weights floor safe HERE but not on MLX's legacy transition override
//     exists to stop a pageable transient from wall-rejecting a small Mac (sc-12179), and that whole
//     class of false reject cannot arise on a discrete card.
//  3. Therefore rejecting when the weights alone overflow can never wall-reject a machine that would
//     have rendered — the sc-12179 regression this lane must not repeat.
//
// ⚠️ **(2) holds only where the on-disk bytes ARE the loaded set** — which is why [`wan_weight_bytes`]
// sums NAMED component subdirs and why `ltx`/`svd` are exempt. See that function's note; getting this
// wrong turns the floor from a lower bound into an over-count, i.e. straight into the sc-12179
// wall-reject.
//
// ⚠️ **The manifest's `footprint.peakMemoryBytes` for these models is an MLX measurement and MUST NOT be
// reused here.** `wan_2_2_t2v_14b`'s recorded 24.5 GiB peak sits BELOW its own `diskSizeBytes` because,
// as that manifest comment says, "A14B is MoE — only ONE 14B expert is resident at a time, not both".
// The candle A14B now ALSO holds one expert at a time (sc-12733, per (1)), but that number is still an
// MLX measurement at a different geometry/dtype and must not be reused: the candle lane is sized by its
// own MEASURED sequential `candle.vramGbByTier` (sc-12631). (Before sc-12732 the candle engine co-located
// both experts, which is why the pre-rework floor under-predicted it by a whole ~28 GiB bf16 expert.)
//
// ## This is a FLOOR — it under-predicts, deliberately — and it is now the FALLBACK, not the gate
// It counts weights only: no activation transient, no VAE decode, no attention working set. A MARGINAL
// job is still admitted and can still CUDA-OOM (catchably, via `classify_engine_error` — not MLX's
// unmappable `exit(-1)`). That was the honest bound from the data that existed before a measurement.
// [`wan_video_fit_error`] now prefers the MEASURED per-tier peak (`candle.vramGbByTier`, sc-12402) and
// only falls back here when a model/tier was never measured.
//
// ⚠️ **Point (1) above is WRONG for the DENSE tier, and sc-12402 measured how wrong.** "On-disk bytes ARE
// the loaded set" holds only for the PACKED q4/q8 tiers. `candle-gen-wan` casts dtypes on load
// (`wan14b.rs:49-52`: `DIT_DTYPE = BF16`, `ENC_DTYPE = F32`, applied per-tensor by `mmap_var_builder`),
// so the dense `Wan-AI/*-Diffusers` experts — which ship **fp32**, not bf16: `transformer/…index.json`
// declares `total_size` 57,153,966,336 = 53.2 GiB for ONE 14B expert — HALVE on load, while the UMT5 TE
// DOUBLES (bf16 10.58 → f32 21.16 GiB) on EVERY tier. Measured error vs the real device set:
//
//   * packed q4/q8 — floor UNDER-counts the co-resident set by the transient it omits. Safe direction:
//     it just admits, the pre-gate status quo. For the sequential-offload Wan engines the relationship
//     even INVERTS: the 5B's on-disk q4 floor is 18.15 GiB, but its MEASURED SEQUENTIAL peak is only
//     ~12.1 GB pool high-water / 10.61 GiB USED_MEM_HIGH (sc-13175 — the denoise transient, with the bf16
//     TE + z48 VAE flushed off-GPU), so there the floor OVER-counts and the measured `candle.vramGbByTier`
//     block must win (it does; see below).
//   * dense bf16 — floor OVER-counts by ~44 GiB (117.5 GiB summed vs a ~75 GiB device set), which
//     wall-rejects a 96 GB card that would render. That is the sc-12179 class this gate's note claims
//     cannot arise here; it can, on this one tier.
//
// So the floor is kept for its ORIGINAL job — refusing the clearly-impossible where nothing was measured
// — and a measurement supersedes it wherever one exists. The dense over-count is why
// [`wan_video_fit_error`] does NOT `max` the two.
//
// Deliberately NOT reusing Mochi's `mochi_decode_peak_gb`: that term is specific to Mochi's untiled
// AsymmVAE (peak linear in clip length). Wan tiles/chunks differently, so applying it here would
// over-predict badly and wall-reject working hardware — hence this gate is frame-blind by construction.

/// The exact component subdirs a candle **Wan** engine's loader reads, or `None` for an engine whose
/// on-disk bytes are NOT its loaded set (see below) ⇒ no signal ⇒ the gate admits.
///
/// Each list is the set of dirs the provider enumerates with `sorted_safetensors`, which loads **every**
/// `.safetensors` it finds — no variant selection — so the sum over these dirs IS the resident set:
///  * 5B (`candle-gen-wan` lib.rs:122/141/124) — `transformer` + `text_encoder` + `vae`;
///  * A14B T2V/I2V (wan14b.rs:288/301/302/328) — the same plus `transformer_2`, the second MoE expert,
///    which is co-resident (see the module note) and must be counted.
///
/// Packed q4/q8 tiers and the dense `Wan-AI/*-Diffusers` snapshot use the SAME subdir names (the tier is
/// detected from tensor content — the `proj_out.scales` marker — not from the directory layout), so one
/// list serves both.
///
/// **Why NAMED subdirs and not a blind recursive sum of `model_dir`.** A blind sum cannot go inert,
/// which is attractive — but it can OVER-count, and over-counting is the direction that hurts: it
/// wall-rejects hardware that renders fine (sc-12179), whereas an under-count merely admits (the
/// pre-gate status quo). Naming the dirs makes a wrong name read `0` for that component — permissive,
/// the safe direction — and it also makes the tier-ROOT fallback safe: if no tier subdir resolved and
/// `model_dir` is a root holding `q4/`+`q8/`, there is no top-level `transformer/`, so this reads 0 and
/// admits rather than summing two tiers at once.
///
/// **Why `ltx_2_3_distilled` and `svd_xt` are `None` (sc-12344's recorded exemptions).** Their on-disk
/// bytes are categorically not their loaded set, so ANY floor built from a directory sum would
/// wall-reject working cards:
///  * **ltx dense** — `ltx_checkpoint()` (candle-gen-ltx lib.rs:129-159) picks exactly ONE root-level
///    file by substring rank (`distilled` > `bf16` > largest), skipping the `fp8`/`mixed`/lora/upscaler
///    siblings shipped beside it. The hosted `Lightricks/LTX-2.3` is ~146 GiB on disk
///    (`estimatedSizeBytes: 157004895813`) against a single-file load — summing it would refuse LTX on
///    every GPU in existence.
///  * **ltx packed tier** — reads 3 exact files (`transformer` / `connector` / `vae_decoder`,
///    tier.rs:152-173) while the tier dir also ships `vae_encoder` + `audio_vae` + `vocoder` +
///    `upsampler`, which the T2V render never loads (tier.rs:30).
///  * **svd_xt** — resolves ONE exact file per component (candle-gen-svd lib.rs:128-141), but the
///    upstream `stabilityai/stable-video-diffusion-img2vid-xt` snapshot ships `X.safetensors` AND
///    `X.fp16.safetensors` side by side in each of `unet`/`vae`/`image_encoder`, so a dir sum roughly
///    DOUBLES a ~8.9 GiB model — enough to false-reject a small card that runs it today.
///
/// Closing those two honestly needs the provider to own its split (`register_generators! { …;
/// footprint = … }` — gen-core's `PerComponentBytes`, whose own doc explains that a consumer guessing
/// component sizes is exactly this failure). None of the video crates registers one today, so that is a
/// cross-repo change on the inference monorepo: **sc-12397**. Not a fudge factor — the LTX dense case is
/// off by ~7x, not by a constant.
fn wan_weight_components(engine_id: &str) -> Option<&'static [&'static str]> {
    match engine_id {
        "wan2_2_ti2v_5b" => Some(&["transformer", "text_encoder", "vae"]),
        "wan2_2_t2v_14b" | "wan2_2_i2v_14b" => {
            Some(&["transformer", "transformer_2", "text_encoder", "vae"])
        }
        // `mochi_1` has its own frame-dependent gate above; `ltx_2_3_distilled` / `svd_xt` are exempt.
        _ => None,
    }
}

/// The on-disk bytes a candle Wan load holds resident: [`wan_weight_components`] summed under
/// `model_dir`. `0` for a non-Wan engine, or when nothing could be scanned ⇒ no signal ⇒ admit.
///
/// Reuses [`crate::mlx_fit_gate::sum_safetensors_bytes`] per component so the HF-cache symlink handling
/// (shards are symlinks into `blobs/`) and the AppleDouble `._*` skip are shared with the MLX lane
/// rather than re-implemented — the same reason sc-12306's Mochi gate reuses `mochi_resident_bytes`.
pub(crate) fn wan_weight_bytes(engine_id: &str, model_dir: &Path) -> u64 {
    wan_weight_components(engine_id).map_or(0, |components| {
        components
            .iter()
            .map(|component| crate::mlx_fit_gate::sum_safetensors_bytes(&model_dir.join(component)))
            .sum()
    })
}

/// Exact resident-weight floor for the dedicated sequential Wan2.2 VACE-Fun provider. The shared
/// UMT5+VAE preparation phase is co-resident, then each complete expert is loaded one at a time. The
/// peak is therefore `max(shared, high + adapters, low + adapters)`, never the sum of both experts.
/// Missing any required component fails closed because this route has no calibrated CUDA peak yet.
pub(crate) fn wan_vace_fun_sequential_weight_bytes(
    model_dir: &Path,
    adapter_stage_bytes: u64,
) -> Result<u64, WorkerError> {
    let component = |name: &str| crate::mlx_fit_gate::sum_safetensors_bytes(&model_dir.join(name));
    let high = component("transformer");
    let low = component("transformer_2");
    let text_encoder = component("text_encoder");
    let vae = component("vae");
    if [high, low, text_encoder, vae].contains(&0) {
        return Err(WorkerError::InvalidPayload(
            "Wan2.2 VACE-Fun admission cannot verify transformer, transformer_2, text_encoder, and vae weights; repair the model install before retrying."
                .to_owned(),
        ));
    }
    let shared = text_encoder.checked_add(vae).ok_or_else(|| {
        WorkerError::InvalidPayload("Wan2.2 VACE-Fun shared footprint overflowed u64.".to_owned())
    })?;
    let high = high.checked_add(adapter_stage_bytes).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Wan2.2 VACE-Fun high-expert footprint overflowed u64.".to_owned(),
        )
    })?;
    let low = low.checked_add(adapter_stage_bytes).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Wan2.2 VACE-Fun low-expert footprint overflowed u64.".to_owned(),
        )
    })?;
    Ok(shared.max(high).max(low))
}

/// The pure Wan candle video admission decision: `Some(error)` when the model's RESIDENT WEIGHTS alone
/// cannot fit the VRAM budget, `None` to admit. The non-Mochi twin of [`mochi_fit_error`], and pure for
/// the same reason — the caller resolves the budget, so the whole decision is unit-testable with no CUDA
/// driver and no GPU.
///
/// Missing either signal admits: no budget (`nvidia-smi` unreadable) or unmeasurable weights
/// (`weight_bytes == 0` — an exempt engine, or a dir that could not be scanned) ⇒ `None`, exactly like
/// [`fit_decision`]'s [`FitDecision::Unknown`]. A gate that blocks without evidence is a regression, not
/// a safety net.
pub(crate) fn video_weights_fit_error(
    model_label: &str,
    weight_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    let (needed_gb, budget) = (video_weights_needed_gb(weight_bytes)?, budget?);
    (budget.free_gb + f64::EPSILON < needed_gb).then(|| {
        video_weights_too_big_error(
            model_label,
            needed_gb,
            budget.free_gb,
            weight_bytes as f64 / BYTES_PER_GIB,
            gpu_id,
        )
    })
}

/// The candle **video** admission decision (sc-12402): budget on the MEASURED per-tier peak
/// (`candle.vramGbByTier[tier_key]` + [`HEADROOM_GB`], via [`predicted_peak_gb`]) when the manifest
/// carries one, else fall back to the sc-12344 on-disk weights FLOOR ([`video_weights_fit_error`]).
///
/// This is the seam that makes the measured blocks LIVE. Until sc-12402 the video lane called the
/// floor directly and nothing read `candle.vramGbByTier` — so adding the blocks alone would have been
/// dead data that reads as coverage (the failure `tests/gpu_and_manifest.rs`'s flux2 test exists to
/// catch). Pure, like both halves it composes: the caller resolves the budget and the tier.
///
/// ## Why a measurement REPLACES the floor rather than composing with it (`max`)
///
/// A `max(measured, floor)` looks safer and is not: **the floor OVER-counts the dense tier**, so
/// maxing would re-introduce the exact wall-reject the measurement exists to remove.
///
/// The floor's premise — `wan_weight_components`' "on-disk bytes ARE the loaded set" — holds only for
/// the PACKED q4/q8 tiers. The dense `Wan-AI/*-Diffusers` snapshot the manifest routes `bf16` to ships
/// its experts in **fp32** (`transformer/…index.json` declares `total_size` 57,153,966,336 = 53.2 GiB
/// for ONE 14B expert), and `candle-gen-wan` casts on load — `DIT_DTYPE = BF16`, `ENC_DTYPE = F32`
/// (wan14b.rs:49-52), applied per-tensor by `mmap_var_builder`. So the experts HALVE (57.2 → 28.6 GB
/// each) while the UMT5 TE DOUBLES (11.4 → 22.7 GB). Net, the dense floor sums ~117 GiB against a real
/// ~81 GB resident set and refuses a 96 GB card that renders it. (The packed tiers fail the other way
/// — the f32 TE doubling makes their floor UNDER-count — which is safe: a floor that under-counts just
/// admits, the pre-gate status quo.)
///
/// A real measurement is not a better proxy for the peak, it IS the peak; deferring to a byte-sum that
/// is provably wrong about dtype would be superstition. Where no measurement exists the floor is still
/// the best available signal, so it stays as the fallback.
///
/// Missing either signal admits, exactly like [`fit_decision`]'s [`FitDecision::Unknown`]: no budget
/// (`nvidia-smi` unreadable) ⇒ `None`, and an engine with neither a `candle` block nor countable
/// weights (`ltx`/`svd`) ⇒ `None` through the floor.
#[cfg(any(test, doc))]
pub(crate) fn wan_video_fit_error(
    model_label: &str,
    manifest_entry: &JsonObject,
    tier_key: &str,
    weight_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    wan_video_fit_error_with_adapter_bytes(
        model_label,
        manifest_entry,
        tier_key,
        weight_bytes,
        0,
        gpu_id,
        budget,
    )
}

/// Adapter-aware Wan admission. Packed callers pass the independently resident user stack; dense
/// callers pass zero because their factors are folded. The calibrated Lightning stack is already in
/// the manifest peak and must not be included here.
pub(crate) fn wan_video_fit_error_with_adapter_bytes(
    model_label: &str,
    manifest_entry: &JsonObject,
    tier_key: &str,
    weight_bytes: u64,
    adapter_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    // Unmeasured (no `candle` block, or no row for this tier and no `minMemoryGb`) ⇒ the sc-12344
    // floor, byte-for-byte the shipped behavior.
    let Some(needed_gb) = predicted_peak_gb(manifest_entry, tier_key) else {
        return video_weights_fit_error(
            model_label,
            weight_bytes.saturating_add(adapter_bytes),
            gpu_id,
            budget,
        );
    };
    let needed_gb = needed_gb + adapter_bytes as f64 / BYTES_PER_GIB;
    let budget = budget?;
    (budget.free_gb + f64::EPSILON < needed_gb)
        .then(|| video_peak_too_big_error(model_label, tier_key, needed_gb, budget.free_gb, gpu_id))
}

/// Fail-closed admission for the shared SCAIL-2 Candle bf16 package. Unlike the generic Wan gate,
/// SCAIL has no lower-memory Candle tier and no sequential/offload lifecycle, so falling back to
/// on-disk bytes or admitting an absent row would turn an incomplete catalog into a real OOM. The
/// exact production render owns `candle.vramGbByTier.bf16`; [`HEADROOM_GB`] remains the common CUDA
/// reserve added by every measured-peak gate.
#[cfg(test)]
pub(crate) fn scail2_video_fit_error(
    manifest_entry: &JsonObject,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    scail2_video_fit_error_with_adapter_bytes(manifest_entry, 0, gpu_id, budget)
}

pub(crate) fn scail2_video_fit_error_with_adapter_bytes(
    manifest_entry: &JsonObject,
    adapter_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    const MEASURED_PIXELS: u64 = 832 * 480;
    let candle = manifest_entry.get("candle");
    let measured = candle
        .and_then(|value| value.get("measured"))
        .and_then(Value::as_bool);
    let peak_gb = candle
        .and_then(|value| value.get("vramGbByTier"))
        .and_then(|tiers| tiers.get("bf16"))
        .and_then(json_f64)
        .filter(|value| value.is_finite() && *value > 0.0);
    let min_memory_gb = candle
        .and_then(|value| value.get("minMemoryGb"))
        .and_then(Value::as_u64);
    let measured_pixels = candle
        .and_then(|value| value.get("vramMeasuredPixels"))
        .and_then(Value::as_u64);
    let (true, Some(peak_gb), Some(min_memory_gb), Some(MEASURED_PIXELS)) = (
        measured == Some(true),
        peak_gb,
        min_memory_gb,
        measured_pixels,
    ) else {
        return Some(WorkerError::InvalidPayload(
            "SCAIL-2 Candle admission is unavailable because the installed catalog has no complete \
             measured bf16 CUDA row (positive peak, 832x480 geometry, and minMemoryGb floor). \
             Refusing to load the 47.2 GB shared package; update SceneWorks before retrying."
                .to_owned(),
        ));
    };
    let base_needed_gb = peak_gb + HEADROOM_GB;
    if (min_memory_gb as f64) + f64::EPSILON < base_needed_gb.ceil() {
        return Some(WorkerError::InvalidPayload(format!(
            "SCAIL-2 Candle admission is unavailable because catalog minMemoryGb={min_memory_gb} \
             is below the measured bf16 CUDA peak plus reserve (~{} GB). Refusing to load the \
             47.2 GB shared package; update SceneWorks before retrying.",
            base_needed_gb.ceil() as u64,
        )));
    }
    let needed_gb = base_needed_gb + adapter_bytes as f64 / BYTES_PER_GIB;
    let Some(budget) = budget else {
        return Some(WorkerError::InvalidPayload(
            "SCAIL-2 Candle admission could not read free GPU VRAM from nvidia-smi. Refusing to \
             load the 47.2 GB shared package without proving that the measured bf16 render peak \
             fits; verify the NVIDIA driver/runtime and retry."
                .to_owned(),
        ));
    };
    (budget.free_gb + f64::EPSILON < needed_gb).then(|| {
        WorkerError::InvalidPayload(format!(
            "SCAIL-2 shared bf16 needs ~{needed} GB of free GPU VRAM (the measured 832x480, \
             81-frame render peak plus CUDA reserve), but GPU {gpu_id} has ~{available} GB \
             available. Its Candle provider has no lower-memory tier or sequential offload; use a \
             GPU with more free VRAM, or use the MLX q4/q8 tiers on macOS.",
            needed = needed_gb.ceil() as i64,
            available = budget.free_gb.round() as i64,
        ))
    })
}

/// Build the MEASURED-peak candle video rejection (sc-12402). The measured sibling of
/// [`video_weights_too_big_error`], and worded apart from it on purpose: that one can only honestly
/// speak about weights, this one is the whole render's ceiling (weights + denoise + the budget-tiled
/// VAE decode), so it names the render rather than the weights.
///
/// Names the TIER, because the tier is the lever that moves this number and the gate now knows which
/// one it sized. Like the floor's message it does NOT offer "lower the resolution": the measured peak
/// is a per-tier CONSTANT taken at the model's default geometry (the manifest schema's "video =
/// default frames"), and Wan's decode is budget-tiled (`auto_tiling_budgeted_wan22`) so the weights
/// dominate and a smaller clip cannot move the number the user was just shown. That is the same
/// resolution-blind contract the image lane's `vramGbByTier` gate has always had, and the reason Mochi
/// needs its own frame-scaled gate instead of this one (its untiled AsymmVAE decode IS the peak).
fn video_peak_too_big_error(
    model_label: &str,
    tier_key: &str,
    needed_gb: f64,
    available_gb: f64,
    gpu_id: &str,
) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{model_label} needs ~{needed} GB of VRAM to render at its {tier_key} tier (the measured peak of \
         the whole render — the resident weights plus the denoise and VAE-decode transient, at whatever \
         component residency the engine uses) but GPU {gpu_id} has ~{available} GB available. \
         Select a smaller quant tier, or run on a GPU with more VRAM.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
    ))
}

/// The predicted resident FLOOR (GiB) for a candle video job: the on-disk weights every component of
/// this engine holds for the whole run, plus [`HEADROOM_GB`].
///
/// [`HEADROOM_GB`] is the CUDA reserve — allocator slack + CUDA context overhead — not MLX's
/// unified reserve (the OS does not draw from discrete VRAM). The two may agree numerically but mean
/// different things; this is the same split `fit_gate::mochi_needed_gb` takes its `reserve_gb`
/// parameter for (sc-12306).
///
/// `None` when nothing was measured (`weight_bytes == 0`) ⇒ no signal ⇒ never block.
fn video_weights_needed_gb(weight_bytes: u64) -> Option<f64> {
    (weight_bytes > 0).then(|| weight_bytes as f64 / BYTES_PER_GIB + HEADROOM_GB)
}

/// Build the generic candle video weights-floor rejection. Names the model, what its weights alone
/// need, and what the card has.
///
/// The levers are WEIGHTS levers, and only those. It deliberately does NOT say "lower the output
/// resolution" like the image lane's `vram_reject_tail`: resolution has exactly ZERO effect on weight
/// bytes, so offering it here would send the user to a knob that cannot move the number they were just
/// shown. Nor does it say Mochi's "shorten the clip" — same reason (there is no decode term in this
/// floor). The tier IS the lever that moves weights, plus the card itself.
fn video_weights_too_big_error(
    model_label: &str,
    needed_gb: f64,
    available_gb: f64,
    weights_gb: f64,
    gpu_id: &str,
) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{model_label} needs at least ~{needed} GB of VRAM just to hold its weights (~{weights} GB, \
         every component held resident for the whole run — this engine does not stage components) but \
         GPU {gpu_id} has ~{available} GB available. Select a smaller quant tier, or run on a GPU with \
         more VRAM.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
        weights = weights_gb.round() as i64,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(value: Value) -> JsonObject {
        value.as_object().expect("object literal").clone()
    }

    fn krea_fit_manifest() -> JsonObject {
        let curve = |fixed: f64, per_mpx: f64| json!({ "fixedGb": fixed, "perMpxGb": per_mpx });
        obj(json!({
            "downloads": [{
                "provider": "huggingface",
                "repo": "SceneWorks/krea-2-turbo-mlx",
                "revision": "d009674080cc1bccf2b629d834c34bf5eccdb723",
                "variant": "q4"
            }],
            "limits": {
                "resolutions": ["768x768", "1024x1024", "1536x1536", "2048x2048"]
            },
            "candle": {
                "vramGbByTier": { "q4": 30.0 },
                "turboFit": {
                    // This is a SYNTHETIC ladder fixture (30 GiB flat q4 curves), not the shipped
                    // manifest, so tracking the current constants is authoring rather than surgery -
                    // `builtin_krea_turbo_calibration_abi_tracks_the_pinned_provider` and
                    // `builtin_krea_turbo_resident_admission_passes_the_provider_handshake` are the
                    // tests that read the SHIPPED values, and neither patches anything.
                    "calibrationAbi": gen_core::MEMORY_CALIBRATION_ABI,
                    "loadShape": "deferred_materialization",
                    "calibrationFingerprint": "krea-turbo-cuda-phase-curves-v1",
                    "sceneWorksRevision": "sc-15449-contract-v1",
                    "inferenceRevision": "a4f409ae8ce73eda2ee8117b89b5f479666606b8",
                    // sc-17774: the fixture must carry the LIVE digest for this lane, because that
                    // is what currency now compares. Reading it from the packaged table rather than
                    // freezing a literal keeps the fixture honest across a pin bump.
                    "inferenceClosureDigest": sceneworks_core::memory_calibration::packaged_closure_digest("candle", "krea_2_turbo").unwrap_or_default(),
                    "measured": true,
                    "maxMeasuredPixels": 1048576,
                    "evidenceRecords": [{
                        "evidenceScope": "exact_request",
                        "tier": "q4",
                        "width": 1024,
                        "height": 1024,
                        "harnessVersion": "unit-test",
                        "sceneWorksCommit": "edcab1247988548aeb5b8a5a8eb8b981826c8b8e",
                        "inferenceCommit": "0ef859f947a1bcd108a37e472ef57f6fab7b6a58",
                        "compatibleSceneWorksRevision": "sc-15449-contract-v1",
                        "measuredCompositions": {
                            "threeStage": ["resident", "staged_residency"],
                            "tiledVae": ["resident", "staged_residency", "bounded_decode"],
                            "chunkedAttention": [
                                "resident",
                                "staged_residency",
                                "bounded_decode",
                                "bounded_attention"
                            ],
                            "streamedBlocks": [
                                "resident",
                                "staged_residency",
                                "bounded_decode",
                                "bounded_attention",
                                "bounded_transformer_residency"
                            ]
                        },
                        "loadability": {
                            "provider": "huggingface",
                            "repository": "SceneWorks/krea-2-turbo-mlx",
                            "resolvedRevision": "d009674080cc1bccf2b629d834c34bf5eccdb723",
                            "tierRoot": "q4",
                            "route": "krea_2_turbo",
                            "backend": "candle",
                            "computeCapability": 12.0
                        },
                        "predictedPeaksGb": {
                            "threeStage": 17.291456,
                            "tiledVae": 16.097152,
                            "chunkedAttention": 14.097152,
                            "streamedBlocks": 11.097152
                        },
                        "predictedPhasesGb": {
                            "threeStage": {
                                "text": 7.524288,
                                "denoise": 16.097152,
                                "decode": 17.291456
                            },
                            "tiledVae": {
                                "text": 7.524288,
                                "denoise": 16.097152,
                                "decode": 11.097152
                            },
                            "chunkedAttention": {
                                "text": 7.524288,
                                "denoise": 14.097152,
                                "decode": 11.097152
                            },
                            "streamedBlocks": {
                                "text": 7.524288,
                                "denoise": 8.048576,
                                "decode": 11.097152
                            }
                        },
                        "observedPeaksGb": {
                            "threeStage": 17.0,
                            "tiledVae": 16.0,
                            "chunkedAttention": 14.0,
                            "streamedBlocks": 11.0
                        },
                        "parity": {
                            "contract": "golden",
                            "result": "passed",
                            "fixture": "unit-test",
                            "metric": "max_abs",
                            "maximumError": 0.0
                        }
                    }],
                    "strategyParameters": {
                        "resident": {},
                        "threeStage": {},
                        "tiledVae": {
                            "decodeTileEdge": 512,
                            "decodeOverlap": 128
                        },
                        "chunkedAttention": {
                            "decodeTileEdge": 512,
                            "decodeOverlap": 128,
                            "attentionChunkSize": 134217728
                        },
                        "streamedBlocks": {
                            "decodeTileEdge": 512,
                            "decodeOverlap": 128,
                            "attentionChunkSize": 134217728,
                            "transformerWindowSize": 1
                        }
                    },
                    "phaseCurvesByTier": {
                        "q4": {
                            "threeStage": {
                                "text": curve(7.0, 0.5),
                                "denoise": curve(14.0, 2.0),
                                "decode": curve(11.0, 6.0)
                            },
                            "tiledVae": {
                                "text": curve(7.0, 0.5),
                                "denoise": curve(14.0, 2.0),
                                "decode": curve(9.0, 2.0)
                            },
                            "chunkedAttention": {
                                "text": curve(7.0, 0.5),
                                "denoise": curve(12.0, 2.0),
                                "decode": curve(9.0, 2.0)
                            },
                            "streamedBlocks": {
                                "text": curve(7.0, 0.5),
                                "denoise": curve(7.0, 1.0),
                                "decode": curve(9.0, 2.0)
                            }
                        }
                    }
                }
            }
        }))
    }

    /// The shipped Krea entry with its `turboFit` closure digest overridden to the LIVE one.
    ///
    /// sc-17774: the shipped ladder declares the digest it was actually measured under, and that is
    /// currently behind the pin — `candle-gen-krea` itself has not moved, but `gen-core` has, so the
    /// currency gate (correctly, for a source-level unit) reports it stale. Every test below is about
    /// which RUNG the ladder selects, which is a different axis; leaving them to trip over currency
    /// would stop them testing the ladder at all. Currency itself is covered by
    /// `krea_control_fit::tests::a_stale_control_closure_falls_back_instead_of_reporting_a_fit`.
    ///
    /// Read, never frozen — a literal would go stale on the next pin bump.
    fn builtin_krea_turbo_manifest_at_live_closure() -> JsonObject {
        let mut manifest = builtin_krea_turbo_manifest();
        if let Some(fit) = manifest
            .get_mut("candle")
            .and_then(Value::as_object_mut)
            .and_then(|candle| candle.get_mut("turboFit"))
            .and_then(Value::as_object_mut)
        {
            fit.insert(
                "inferenceClosureDigest".to_owned(),
                json!(
                    sceneworks_core::memory_calibration::packaged_closure_digest(
                        "candle",
                        "krea_2_turbo"
                    )
                    .unwrap_or_default()
                ),
            );
        }
        manifest
    }

    fn builtin_krea_turbo_manifest() -> JsonObject {
        let jsonc = include_str!("../../../config/manifests/builtin.models.jsonc");
        let parsed: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(jsonc))
                .expect("builtin model manifest parses");
        parsed["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|model| model["id"].as_str() == Some("krea_2_turbo"))
            .and_then(Value::as_object)
            .expect("Krea 2 Turbo manifest entry")
            .clone()
    }

    fn builtin_krea_turbo_manifest_with_original_fingerprint() -> JsonObject {
        let mut manifest = builtin_krea_turbo_manifest_at_live_closure();
        manifest["candle"]["turboFit"]["calibrationFingerprint"] =
            Value::String("krea-turbo-cuda-phase-curves-v1".into());
        manifest
    }

    /// sc-17097: NO shipped route may carry a `calibrationAbi` stamp the pinned gen-core has moved
    /// past, whatever its manifest shape.
    ///
    /// The per-route tests below cover `krea_2_turbo` in depth because it is the one stamp that reaches
    /// a provider run context. This sweep is the cheap net for the next ABI bump: it walks the whole
    /// builtin manifest for any key named `calibrationAbi` and fails on the first stale one, so a route
    /// added later cannot ship a stale stamp just because nobody wrote it a bespoke test.
    ///
    /// Deliberately NOT swept: `mlx.calibrations[].abi` and `memoryStrategyContract.abi`. The former is
    /// a historical receipt binding whose ABI-1 rows are correct provenance and demote fail-closed to
    /// the legacy estimator (sc-16482 forbids backfilling them); the latter is an unrelated
    /// static-contract version that the schema independently pins.
    #[test]
    fn no_builtin_route_ships_a_stale_calibration_abi_stamp() {
        let jsonc = include_str!("../../../config/manifests/builtin.models.jsonc");
        let parsed: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(jsonc))
                .expect("builtin model manifest parses");

        // Two shapes carry a live calibration ABI: `turboFit.calibrationAbi` (Krea's bespoke opt-in)
        // and `<backend>.calibrations[].abi` (the packaged-evidence bindings, read into
        // `MemoryEvidence::calibration_abi` by `candle_memory_strategy::binding`). Both are swept.
        fn walk(value: &Value, path: &str, found: &mut Vec<(String, u64)>) {
            match value {
                Value::Object(map) => {
                    for (key, child) in map {
                        let child_path = format!("{path}.{key}");
                        let is_stamp = key == "calibrationAbi"
                            // `mlx.calibrations[]` rows are HISTORICAL receipts: sc-16482 forbids
                            // backfilling them and `mlx_fit_gate` demotes a stale one to
                            // `AdmissionPath::Legacy`, so an ABI-1 row there is correct provenance,
                            // not drift. The candle rows have no such carve-out.
                            || (key == "abi" && path.ends_with("]") && path.contains(".candle.calibrations"));
                        if is_stamp {
                            found.push((
                                child_path.clone(),
                                child
                                    .as_u64()
                                    .expect("a calibration ABI stamp is an integer"),
                            ));
                        }
                        walk(child, &child_path, found);
                    }
                }
                Value::Array(items) => {
                    for (index, child) in items.iter().enumerate() {
                        walk(child, &format!("{path}[{index}]"), found);
                    }
                }
                _ => {}
            }
        }

        let mut found = Vec::new();
        walk(&parsed, "manifest", &mut found);
        assert!(
            !found.is_empty(),
            "the sweep found no calibrationAbi stamps at all — it has stopped inspecting what it \
             claims to guard"
        );
        let stale = found
            .iter()
            .filter(|(_, abi)| *abi != u64::from(gen_core::MEMORY_CALIBRATION_ABI))
            .collect::<Vec<_>>();
        assert!(
            stale.is_empty(),
            "{stale:?} are stamped at a calibration ABI the pinned gen-core ({}) has moved past; \
             re-measure those fits, do not re-stamp them",
            gen_core::MEMORY_CALIBRATION_ABI
        );
    }

    /// sc-17097: the shipped `krea_2_turbo` opt-in must be stamped at the calibration ABI the pinned
    /// provider actually declares.
    ///
    /// A stale stamp is not a soft degrade. [`gen_core::MemoryRunContext`] documents that "resident
    /// requests carry this handshake too", and `optimized_eligibility` returns `Ok(())` for the
    /// non-optimized resident rung *before* it ever reaches the ABI comparison — so on any card where
    /// Krea fits resident (every tier on a 96 GB RTX PRO 6000) the selector still returns
    /// `KreaTurboFit::Resident`, the worker still builds a run context out of this stamp, and
    /// `standard_memory_strategy_safety_check` rejects the whole request with
    /// `unsupported: krea_2_turbo: calibration handshake mismatch`.
    #[test]
    fn builtin_krea_turbo_calibration_abi_tracks_the_pinned_provider() {
        let manifest = builtin_krea_turbo_manifest_at_live_closure();
        let shipped = manifest["candle"]["turboFit"]["calibrationAbi"]
            .as_u64()
            .expect("shipped calibrationAbi");
        assert_eq!(
            u32::try_from(shipped).expect("calibrationAbi fits u32"),
            gen_core::MEMORY_CALIBRATION_ABI,
            "builtin.models.jsonc stamps calibrationAbi {shipped} while the pinned gen-core declares \
             {}; every Krea Turbo candle t2i that fits resident fails the provider handshake until \
             the fit is re-measured under the current ABI",
            gen_core::MEMORY_CALIBRATION_ABI
        );
    }

    /// sc-17097 end-to-end regression: drive the exact seam `image_jobs::base` uses — the shipped
    /// manifest's own stamp into a real [`gen_core::MemoryRunContext`], checked against the real
    /// pinned provider contract — and require the provider to ACCEPT.
    ///
    /// This is the production failure reproduced in-process: it goes RED the moment the shipped stamp
    /// drifts from the provider, whatever the reason, and no fixture surgery can hide it because the
    /// manifest is read unpatched.
    #[test]
    fn builtin_krea_turbo_resident_admission_passes_the_provider_handshake() {
        let manifest = builtin_krea_turbo_manifest_at_live_closure();
        let turbo_fit = manifest["candle"]["turboFit"]
            .as_object()
            .expect("Krea turbo fit");
        let provider_contract = crate::inference_runtime::media()
            .memory_strategy_contract("krea_2_turbo", &krea_test_load_spec("q4"))
            .expect("Krea contract lookup succeeds")
            .expect("Krea contract exists");
        let identity = provider_contract
            .calibration
            .as_ref()
            .expect("Krea provider declares calibration");

        // A card that comfortably holds the resident tier, which is what makes this a hard reject
        // rather than a fallback: the selector picks Resident and the worker builds a context.
        let budget = Some(VramBudget {
            free_gb: 90.0,
            total_gb: 96.0,
        });
        let fit = krea_turbo_fit(&manifest, "q4", 1024, 1024, budget, true);
        let Some(KreaTurboFit::Resident {
            peak_gb, selection, ..
        }) = fit
        else {
            panic!("expected the shipped Krea manifest to select the resident rung, got {fit:?}");
        };

        let gb_to_bytes = |gb: f64| {
            (gb * 1024.0 * 1024.0 * 1024.0)
                .round()
                .clamp(0.0, u64::MAX as f64) as u64
        };
        // Field-for-field the context `image_jobs::base` builds for a plain Krea Turbo t2i.
        let context = gen_core::MemoryRunContext {
            selection,
            optimization_authority: gen_core::MemoryOptimizationAuthority::Calibrated,
            calibration_abi: u32::try_from(
                turbo_fit["calibrationAbi"]
                    .as_u64()
                    .expect("shipped calibrationAbi"),
            )
            .expect("calibrationAbi fits u32"),
            calibration_fingerprint: turbo_fit["calibrationFingerprint"]
                .as_str()
                .expect("shipped calibrationFingerprint")
                .to_owned(),
            // From the MANIFEST, exactly as `image_jobs::base` does. Reading it back off `identity`
            // would compare the provider against itself and could never catch a shape mismatch -
            // which is the whole reason calibration ABI 2 added this axis.
            load_shape: krea_turbo_load_shape(&manifest["candle"]["turboFit"])
                .expect("the shipped turbo fit declares a load shape"),
            mode: gen_core::MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: gen_core::MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: gen_core::MemoryBudget {
                total_bytes: gb_to_bytes(96.0),
                committed_bytes: gb_to_bytes(6.0),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: gb_to_bytes(2.0),
            },
            predicted_peak_bytes: gb_to_bytes(peak_gb),
            cache_state: gen_core::MemoryCacheState::Cold,
            evidence_revision: format!(
                "{KREA_TURBO_SCENEWORKS_REVISION}@{}",
                sceneworks_core::memory_calibration::packaged_closure_digest(
                    "candle",
                    "krea_2_turbo"
                )
                .unwrap_or_default()
            ),
        };

        let decision = gen_core::standard_memory_strategy_safety_check(
            &provider_contract,
            &context,
            None,
            None,
        );
        assert_eq!(
            decision,
            gen_core::MemorySafetyDecision::Accept,
            "the shipped Krea Turbo opt-in must satisfy the pinned provider's calibration handshake \
             (manifest ABI {}, provider ABI {})",
            context.calibration_abi,
            identity.abi
        );
    }

    #[test]
    fn parse_vram_cap_accepts_positive_numbers_only() {
        assert_eq!(parse_vram_cap(Some("10")), Some(10.0));
        assert_eq!(parse_vram_cap(Some("  8.5 ")), Some(8.5));
        assert_eq!(parse_vram_cap(Some("0")), None);
        assert_eq!(parse_vram_cap(Some("-4")), None);
        assert_eq!(parse_vram_cap(Some("nan")), None);
        assert_eq!(parse_vram_cap(Some("abc")), None);
        assert_eq!(parse_vram_cap(Some("")), None);
        assert_eq!(parse_vram_cap(None), None);
    }

    #[test]
    fn apply_vram_cap_emulates_a_smaller_card() {
        let real = VramBudget {
            free_gb: 90.0,
            total_gb: 96.0,
        };
        // No cap ⇒ unchanged.
        assert_eq!(apply_vram_cap(Some(real), None), Some(real));
        // Cap below real free ⇒ total = cap, free clamped to cap.
        assert_eq!(
            apply_vram_cap(Some(real), Some(10.0)),
            Some(VramBudget {
                free_gb: 10.0,
                total_gb: 10.0,
            })
        );
        // Real free already below the cap ⇒ free preserved, total = cap.
        assert_eq!(
            apply_vram_cap(
                Some(VramBudget {
                    free_gb: 6.0,
                    total_gb: 96.0,
                }),
                Some(10.0)
            ),
            Some(VramBudget {
                free_gb: 6.0,
                total_gb: 10.0,
            })
        );
        // No live reading + a cap ⇒ synthetic full budget (exercisable in a no-GPU test).
        assert_eq!(
            apply_vram_cap(None, Some(10.0)),
            Some(VramBudget {
                free_gb: 10.0,
                total_gb: 10.0,
            })
        );
        // No reading, no cap ⇒ None.
        assert_eq!(apply_vram_cap(None, None), None);
    }

    #[test]
    fn krea_turbo_selects_the_least_cost_sufficient_rung() {
        let manifest = krea_fit_manifest();
        let fit = |free_gb| {
            krea_turbo_fit(
                &manifest,
                "q4",
                1024,
                1024,
                Some(VramBudget {
                    free_gb,
                    total_gb: free_gb,
                }),
                true,
            )
        };
        assert!(matches!(
            fit(20.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            fit(19.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedDecode,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            fit(17.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedAttention,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            fit(13.5),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedTransformerResidency,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(fit(10.0), Some(KreaTurboFit::Reject { .. })));
    }

    #[test]
    fn krea_admission_uses_record_peak_not_a_lower_mutated_curve() {
        let mut manifest = krea_fit_manifest();
        let curves = manifest
            .get_mut("candle")
            .and_then(Value::as_object_mut)
            .and_then(|candle| candle.get_mut("turboFit"))
            .and_then(Value::as_object_mut)
            .and_then(|fit| fit.get_mut("phaseCurvesByTier"))
            .and_then(Value::as_object_mut)
            .and_then(|tiers| tiers.get_mut("q4"))
            .and_then(Value::as_object_mut)
            .and_then(|tier| tier.get_mut("threeStage"))
            .and_then(Value::as_object_mut)
            .expect("three-stage curves");
        for phase in ["text", "denoise", "decode"] {
            curves.insert(phase.to_owned(), json!({ "fixedGb": 1.0, "perMpxGb": 0.0 }));
        }
        assert!(matches!(
            krea_turbo_fit(
                &manifest,
                "q4",
                1024,
                1024,
                Some(VramBudget {
                    free_gb: 19.0,
                    total_gb: 19.0,
                }),
                true,
            ),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedDecode,
                    ..
                },
                ..
            })
        ));
    }

    #[test]
    fn krea_non_max_phase_mutation_excludes_only_the_unbound_rung() {
        let mut manifest = krea_fit_manifest();
        manifest["candle"]["turboFit"]["phaseCurvesByTier"]["q4"]["threeStage"]["text"]
            ["fixedGb"] = json!(8.0);
        assert!(matches!(
            krea_turbo_fit(
                &manifest,
                "q4",
                1024,
                1024,
                Some(VramBudget {
                    free_gb: 20.0,
                    total_gb: 20.0,
                }),
                true,
            ),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedDecode,
                    ..
                },
                ..
            })
        ));
    }

    #[test]
    fn krea_evidence_dimension_mutations_fail_closed_with_specific_reasons() {
        let fit = |manifest: &JsonObject| {
            krea_turbo_fit(
                manifest,
                "q4",
                1024,
                1024,
                Some(VramBudget {
                    free_gb: 20.0,
                    total_gb: 20.0,
                }),
                true,
            )
        };

        // sc-17774: mutate the term that DECIDES currency. This used to set
        // `compatibleInferenceRevision`, which is deleted — leaving the mutation inert and the
        // fail-closed assertion passing vacuously.
        let mut stale = krea_fit_manifest();
        stale["candle"]["turboFit"]["inferenceClosureDigest"] = Value::String("1".repeat(64));
        assert_eq!(
            fit(&stale),
            Some(KreaTurboFit::Unverified {
                reason: gen_core::MemoryEvidenceVerdict::Stale,
            })
        );

        let mut unloadable = krea_fit_manifest();
        unloadable["candle"]["turboFit"]["evidenceRecords"][0]["loadability"]["resolvedRevision"] =
            Value::String("2222222222222222222222222222222222222222".into());
        assert_eq!(
            fit(&unloadable),
            Some(KreaTurboFit::Unverified {
                reason: gen_core::MemoryEvidenceVerdict::Unverified,
            })
        );

        let mut fingerprint = krea_fit_manifest();
        fingerprint["candle"]["turboFit"]["calibrationFingerprint"] =
            Value::String("mutated-fingerprint".into());
        assert_eq!(
            fit(&fingerprint),
            Some(KreaTurboFit::Unverified {
                reason: gen_core::MemoryEvidenceVerdict::FingerprintMismatch,
            })
        );

        let mut legacy = krea_fit_manifest();
        legacy["candle"]["turboFit"]["evidenceRecords"][0]
            .as_object_mut()
            .expect("evidence record")
            .remove("measuredCompositions");
        assert_eq!(
            fit(&legacy),
            Some(KreaTurboFit::Unverified {
                reason: gen_core::MemoryEvidenceVerdict::Invalid,
            }),
            "composition-agnostic legacy evidence must not remain eligible"
        );
    }

    #[test]
    fn krea_runtime_context_is_required_and_hardware_and_artifact_bound() {
        let manifest = krea_fit_manifest();
        let budget = Some(VramBudget {
            free_gb: 20.0,
            total_gb: 20.0,
        });
        let run = |runtime: Option<&KreaRuntimeEvidenceContext>| {
            krea_turbo_fit_with_runtime(&manifest, "q4", 1024, 1024, budget, true, runtime)
        };
        assert!(matches!(run(None), Some(KreaTurboFit::Unverified { .. })));

        let mut wrong_hardware = KreaRuntimeEvidenceContext::verified_for_test("q4");
        wrong_hardware.compute_capability = 8.9;
        assert!(matches!(
            run(Some(&wrong_hardware)),
            Some(KreaTurboFit::Unverified {
                reason: gen_core::MemoryEvidenceVerdict::Stale,
            })
        ));

        let mut wrong_artifact = KreaRuntimeEvidenceContext::verified_for_test("q4");
        wrong_artifact.resolved_revision = "2222222222222222222222222222222222222222".into();
        assert!(matches!(
            run(Some(&wrong_artifact)),
            Some(KreaTurboFit::Unverified {
                reason: gen_core::MemoryEvidenceVerdict::Unverified,
            })
        ));
    }

    #[test]
    fn runtime_artifact_inspection_rejects_a_missing_index_shard() {
        let temp = tempfile::tempdir().expect("tempdir");
        let revision = "d009674080cc1bccf2b629d834c34bf5eccdb723";
        let snapshot = temp.path().join("snapshots").join(revision);
        let tier = snapshot.join("q4");
        for component in [
            "transformer",
            "text_encoder",
            "vae",
            "tokenizer",
            "scheduler",
        ] {
            std::fs::create_dir_all(tier.join(component)).expect("component dir");
        }
        std::fs::write(
            tier.join("model_index.json"),
            r#"{
              "transformer": ["lib", "Transformer"],
              "text_encoder": ["lib", "TextEncoder"],
              "vae": ["lib", "Vae"],
              "tokenizer": ["lib", "Tokenizer"],
              "scheduler": ["lib", "Scheduler"]
            }"#,
        )
        .expect("model index");
        for component in ["transformer", "text_encoder", "vae"] {
            std::fs::write(tier.join(component).join("config.json"), "{}").expect("config");
        }
        let transformer_shard = tier.join("transformer/model-00001-of-00001.safetensors");
        std::fs::write(
            tier.join("transformer/model.safetensors.index.json"),
            r#"{"weight_map":{"tensor":"model-00001-of-00001.safetensors"}}"#,
        )
        .expect("weight index");
        std::fs::write(&transformer_shard, b"weights").expect("transformer shard");
        std::fs::write(tier.join("text_encoder/model.safetensors"), b"weights")
            .expect("text weights");
        std::fs::write(tier.join("vae/model.safetensors"), b"weights").expect("vae weights");
        std::fs::write(tier.join("tokenizer/tokenizer.json"), "{}").expect("tokenizer");
        std::fs::write(tier.join("scheduler/scheduler_config.json"), "{}").expect("scheduler");

        let inspect = || {
            KreaRuntimeEvidenceContext::inspect(
                "krea_2_turbo",
                "candle",
                "0",
                Some(12.0),
                "huggingface",
                "SceneWorks/krea-2-turbo-mlx",
                revision,
                "q4",
                &tier,
                &snapshot,
            )
        };
        assert!(inspect().is_some());
        std::fs::remove_file(transformer_shard).expect("remove shard");
        assert!(inspect().is_none());
    }

    #[test]
    fn krea_turbo_fit_is_tier_and_resolution_aware_and_never_guesses() {
        let manifest = krea_fit_manifest();
        let budget = Some(VramBudget {
            free_gb: 17.0,
            total_gb: 17.0,
        });
        assert!(matches!(
            krea_turbo_fit(&manifest, "q4", 1024, 1024, budget, true),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedAttention,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            krea_turbo_fit(&manifest, "q4", 2048, 2048, budget, true),
            Some(KreaTurboFit::Unverified { .. })
        ));
        assert_eq!(
            krea_turbo_smaller_fit(&manifest, "q4", 2048, 2048, budget, true),
            Some((1024, 1024))
        );
        assert_eq!(
            krea_turbo_fit(&manifest, "q8", 1024, 1024, budget, true),
            None,
            "the synthetic fixture intentionally carries only q4 evidence"
        );
        assert_eq!(
            krea_turbo_fit(&manifest, "q4", 1024, 1024, None, true),
            None
        );
    }

    #[test]
    fn krea_turbo_block_streaming_is_not_selected_for_adapter_jobs() {
        let manifest = krea_fit_manifest();
        let budget = Some(VramBudget {
            free_gb: 13.5,
            total_gb: 13.5,
        });
        assert!(matches!(
            krea_turbo_fit(&manifest, "q4", 1024, 1024, budget, false),
            Some(KreaTurboFit::Unverified { .. })
        ));
    }

    /// sc-18097 headline (epic 18093 R1b): an in-envelope geometry with NO exact measured record —
    /// the cell that used to freeze to `Unverified` and fall back to the resident-only generic
    /// gate — now admits through fitted-curve estimates, deep rungs included, with the measured
    /// strategy parameters, and refuses honestly below the widened margins.
    ///
    /// Fixture arithmetic at 896² (0.802816 Mpx; fixture curves in [`krea_fit_manifest`]), all
    /// binding phases matching the 1024² anchor record's:
    ///   threeStage  peak 15.817 (decode-bound) → widened ×1.04 ≈ 16.450
    ///   tiledVae    peak 15.606 (denoise)      → ≈ 16.230
    ///   chunkedAttention peak 13.606 (denoise) → ≈ 14.150
    ///   streamedBlocks   peak 10.606 (decode)  → ≈ 11.030
    #[test]
    fn krea_turbo_unmeasured_geometry_admits_by_fitted_estimate_and_refuses_below_margin() {
        let manifest = krea_fit_manifest();
        let fit = |free_gb: f64| {
            krea_turbo_fit(
                &manifest,
                "q4",
                896,
                896,
                Some(VramBudget {
                    free_gb,
                    total_gb: free_gb,
                }),
                true,
            )
        };

        // 20 GiB free (18 effective): the cheapest fitting estimate rung is the staged floor.
        match fit(20.0) {
            Some(KreaTurboFit::Fits { selection, .. }) => {
                assert_eq!(selection.strategy, MemoryStrategy::StagedResidency);
            }
            other => panic!("896² must admit by the staged fitted estimate, got {other:?}"),
        }

        // 13.2 GiB free (11.2 effective): only the deep rung's widened estimate (~11.03) fits —
        // and the selection must carry the measured sweep parameters and translate to the engine
        // knobs the engaged composition names.
        match fit(13.2) {
            Some(KreaTurboFit::Fits {
                selection, memory, ..
            }) => {
                assert_eq!(
                    selection.strategy,
                    MemoryStrategy::BoundedTransformerResidency
                );
                assert_eq!(selection.parameters.decode_tile_edge, Some(512));
                assert_eq!(selection.parameters.decode_overlap, Some(128));
                assert_eq!(selection.parameters.attention_chunk_size, Some(134_217_728));
                assert_eq!(selection.parameters.transformer_window_size, Some(1));
                assert!(memory.tile_vae_decode);
                assert!(memory.chunk_attention);
                assert!(memory.stream_transformer_blocks);
            }
            other => panic!("only the deep estimate rung fits 11.2 GiB effective, got {other:?}"),
        }

        // Margin mutation arm: at 12.9 GiB free (10.9 effective) the RAW deep-rung peak (10.606)
        // fits but the widened one (~11.03) does not — a selector whose estimate margin is zeroed
        // admits here and flips this arm red. The refusal quotes the widened requirement plus the
        // 2 GiB admission headroom, recomputed from the POLICY constant so a narrower margin
        // cannot sneak in.
        match fit(12.9) {
            Some(KreaTurboFit::Reject { needed_gb, .. }) => {
                let streamed_peak_gb = 9.0 + 2.0 * 0.802816;
                let expected = streamed_peak_gb
                    * (1.0 + crate::ladder_margin_policy::CANDLE_ESTIMATE_MARGIN)
                    + HEADROOM_GB;
                assert!(
                    (needed_gb - expected).abs() < 1e-3,
                    "the refusal must quote the margin-widened deep-rung estimate: needed \
                     {needed_gb}, expected {expected}"
                );
            }
            other => panic!("below every widened estimate the request must reject, got {other:?}"),
        }
    }

    /// sc-18097 review: refusal ADVICE stays measured-only. `krea_turbo_smaller_fit_with_runtime`
    /// mirrors `mlx_fit_gate::verified_lower_alternative` and must not name a geometry whose own
    /// admission rests on a fitted estimate — telling a user to drop to a resolution that was
    /// itself only guessed invites a second refusal there.
    ///
    /// The shipped-data assertions elsewhere cannot pin this (the reviewer's finding): q8/bf16's
    /// binding streamed floors are resolution-INDEPENDENT (`text` is `fixedGb` 5.01 / `perMpxGb`
    /// 0), so no smaller geometry fits those budgets with or without the filter, and the fixture's
    /// 2048²→1024² case resolves to a cell that HAS an exact record. This fixture arm is built to
    /// discriminate:
    ///
    /// * 1024² carries an exact record, so its streamed rung is graded at the MEASURED 11.097 GiB
    ///   — which does not fit a 10.9 GiB effective budget: the largest lower bucket rejects.
    /// * 768² has no record, so its streamed rung is a fitted estimate at the curve peak 10.180
    ///   GiB, widened to 10.587 — which DOES fit. Both facts are asserted directly below, so the
    ///   discrimination is visible in this test rather than inferred: with the filter removed the
    ///   helper necessarily returns `Some((768, 768))` from the very call it makes.
    #[test]
    fn krea_turbo_smaller_fit_never_names_an_estimate_backed_geometry() {
        let manifest = krea_fit_manifest();
        let budget = |free_gb: f64| {
            Some(VramBudget {
                free_gb,
                total_gb: free_gb,
            })
        };

        // The two facts that make the filter load-bearing at this budget.
        assert!(
            matches!(
                krea_turbo_fit(&manifest, "q4", 1024, 1024, budget(12.9), true),
                Some(KreaTurboFit::Reject { .. })
            ),
            "the largest lower bucket must reject at its MEASURED peak, or the helper would stop \
             there and never reach the estimate-backed one"
        );
        match krea_turbo_fit(&manifest, "q4", 768, 768, budget(12.9), true) {
            Some(KreaTurboFit::Fits {
                estimate_scoped,
                selection,
                ..
            }) => {
                assert!(
                    estimate_scoped,
                    "768² has no exact record, so its admission must be estimate-scoped"
                );
                assert_eq!(
                    selection.strategy,
                    MemoryStrategy::BoundedTransformerResidency
                );
            }
            other => panic!(
                "768² must ADMIT by fitted estimate here — that is what the filter has to \
                 suppress: {other:?}"
            ),
        }
        // …so the advice must be silent rather than name 768².
        assert_eq!(
            krea_turbo_smaller_fit(&manifest, "q4", 1536, 1536, budget(12.9), true),
            None,
            "an estimate-backed geometry may not be offered as the lower-resolution escape hatch"
        );

        // Control: a MEASURED lower cell is still named, so the filter withholds estimates rather
        // than silencing the advice outright. At 14 GiB free (12 effective) 1024²'s measured
        // streamed rung (11.097) fits.
        assert_eq!(
            krea_turbo_smaller_fit(&manifest, "q4", 1536, 1536, budget(14.0), true),
            Some((1024, 1024)),
            "a measured lower cell must still be offered"
        );
    }

    /// sc-18094/sc-18097: `ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE` is honored at the
    /// candle synthesis seam. At 512² the fixture's threeStage curve moves the request peak onto
    /// DENOISE (14.524 against 12.573 decode) while its 1024² anchor record binds on DECODE
    /// (17.291) — so the staged fitted estimate is refused and the ladder's first admissible rung
    /// is bounded decode, whose binding phase is denoise at BOTH geometries. Both rungs' widened
    /// peaks fit the 17.5 GiB effective budget and the staged rung is walked first, so a mutation
    /// that drops the binding-phase gate selects `StagedResidency` and turns this red.
    #[test]
    fn krea_turbo_fitted_estimates_honor_the_measured_binding_phase_constraint() {
        let manifest = krea_fit_manifest();
        match krea_turbo_fit(
            &manifest,
            "q4",
            512,
            512,
            Some(VramBudget {
                free_gb: 19.5,
                total_gb: 19.5,
            }),
            true,
        ) {
            Some(KreaTurboFit::Fits { selection, .. }) => {
                assert_eq!(
                    selection.strategy,
                    MemoryStrategy::BoundedDecode,
                    "a binding-phase flip must refuse the staged FITTED estimate per \
                     ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE"
                );
            }
            other => panic!("the no-flip bounded-decode estimate must admit 512², got {other:?}"),
        }
    }

    /// sc-18097: the estimate bases obey the sc-18096 restrictions — a stale-closure manifest may
    /// not seed fitted extrapolation (its measured cells keep serving their OWN geometry behind
    /// the stale margin, but the estimate margin was derived over same-closure re-capture variance
    /// and cannot also absorb closure drift), and a calibration fingerprint that drifted from the
    /// loaded provider's identity loses the bases entirely. Both mutations are WELL-FORMED (the
    /// digest is a valid 64-hex string, the fingerprint keeps the shipped token grammar), so the
    /// refusals below are the anchor-eligibility gate's work, not a parse failure — and the
    /// registered contract is asserted conformance-CLEAN so the fingerprint arm cannot pass by a
    /// grammar-conformance accident (the sc-18096 finding).
    #[test]
    fn krea_turbo_estimate_bases_require_current_closure_and_loaded_identity() {
        let admit = |manifest: &JsonObject| {
            krea_turbo_fit(
                manifest,
                "q4",
                896,
                896,
                Some(VramBudget {
                    free_gb: 20.0,
                    total_gb: 20.0,
                }),
                true,
            )
        };
        // Control point: the unmutated manifest admits this cell by fitted estimate.
        assert!(
            matches!(admit(&krea_fit_manifest()), Some(KreaTurboFit::Fits { .. })),
            "the unmutated fixture must admit 896² by estimate, or the arms below prove nothing"
        );

        let mut stale = krea_fit_manifest();
        stale["candle"]["turboFit"]["inferenceClosureDigest"] = Value::String("1".repeat(64));
        assert!(
            matches!(admit(&stale), Some(KreaTurboFit::Unverified { .. })),
            "a stale-closure record must not seed a fitted extrapolation"
        );

        let provider_contract = crate::inference_runtime::media()
            .memory_strategy_contract("krea_2_turbo", &krea_test_load_spec("q4"))
            .expect("Krea contract lookup succeeds")
            .expect("Krea contract exists");
        assert!(
            provider_contract.conformance_errors().is_empty(),
            "the loaded contract must be conformance-clean so the fingerprint arm exercises the \
             anchor identity gate, not format validation"
        );
        let mut drifted = krea_fit_manifest();
        drifted["candle"]["turboFit"]["calibrationFingerprint"] =
            Value::String("krea-turbo-cuda-phase-curves-v2".into());
        assert_eq!(
            admit(&drifted),
            Some(KreaTurboFit::Unverified {
                reason: gen_core::MemoryEvidenceVerdict::FingerprintMismatch,
            }),
            "a fingerprint drifted from the loaded identity loses the fitted bases"
        );
    }

    #[test]
    fn builtin_krea_evidence_is_keyed_by_the_measured_engaged_composition() {
        let manifest = builtin_krea_turbo_manifest_with_original_fingerprint();
        let turbo_fit = manifest["candle"]["turboFit"]
            .as_object()
            .expect("Krea turbo fit");
        let manifest_fingerprint = turbo_fit["calibrationFingerprint"]
            .as_str()
            .expect("manifest calibration fingerprint");
        assert_eq!(manifest_fingerprint, "krea-turbo-cuda-phase-curves-v1");

        let provider_contract = crate::inference_runtime::media()
            .memory_strategy_contract("krea_2_turbo", &krea_test_load_spec("q4"))
            .expect("Krea contract lookup succeeds")
            .expect("Krea contract exists");
        let provider_fingerprint = provider_contract
            .calibration
            .as_ref()
            .expect("Krea provider declares calibration")
            .fingerprint
            .as_str();
        assert_eq!(
            manifest_fingerprint, provider_fingerprint,
            "composition identity, not a blanket fingerprint tombstone, invalidates mismatched rows"
        );

        for tier in ["q4", "q8", "bf16"] {
            let measured = turbo_fit["evidenceRecords"]
                .as_array()
                .expect("evidence records")
                .iter()
                .find(|record| {
                    record["tier"].as_str() == Some(tier) && record["width"].as_u64() == Some(1024)
                })
                .expect("1024 evidence record");
            for manifest_rung in [
                "threeStage",
                "tiledVae",
                "chunkedAttention",
                "streamedBlocks",
            ] {
                assert_eq!(
                    turbo_fit["engagedCompositions"][manifest_rung],
                    measured["measuredCompositions"][manifest_rung],
                    "{tier} {manifest_rung} matrix identity must match its measured composition"
                );
            }
            assert_eq!(
                measured["measuredCompositions"]["tiledVae"],
                json!(["resident", "staged_residency", "bounded_decode"])
            );
            assert_eq!(
                measured["measuredCompositions"]["chunkedAttention"],
                json!([
                    "resident",
                    "staged_residency",
                    "bounded_decode",
                    "bounded_attention"
                ])
            );
        }

        use gen_core::MemoryStrategy::{
            BoundedAttention, BoundedDecode, BoundedTransformerResidency, Resident, StagedResidency,
        };
        for (strategy, expected) in [
            (Resident, vec![Resident]),
            (StagedResidency, vec![Resident, StagedResidency]),
            (
                BoundedDecode,
                vec![Resident, StagedResidency, BoundedDecode],
            ),
            (
                BoundedAttention,
                vec![Resident, StagedResidency, BoundedDecode, BoundedAttention],
            ),
            (
                BoundedTransformerResidency,
                vec![
                    Resident,
                    StagedResidency,
                    BoundedDecode,
                    BoundedAttention,
                    BoundedTransformerResidency,
                ],
            ),
        ] {
            assert_eq!(
                provider_contract.engaged_composition(strategy),
                expected,
                "provider composition must match the measured and published ladder for {strategy:?}"
            );
        }

        for (tier, probe_rung) in [
            ("q4", MemoryStrategy::BoundedDecode),
            ("q8", MemoryStrategy::BoundedAttention),
            ("bf16", MemoryStrategy::BoundedAttention),
        ] {
            let phases = krea_rung_phase_peaks(&manifest, tier, probe_rung, 1024, 1024)
                .expect("probed rung phase peaks");
            let required = phases.peak_gb() + HEADROOM_GB + 0.01;
            match krea_turbo_fit(
                &manifest,
                tier,
                1024,
                1024,
                Some(VramBudget {
                    free_gb: required,
                    total_gb: required,
                }),
                true,
            ) {
                Some(KreaTurboFit::Fits { selection, .. }) => assert_eq!(
                    selection.strategy, probe_rung,
                    "{tier} must select the least-cost useful rung at its measured boundary"
                ),
                other => panic!("{tier} expected {probe_rung:?}, got {other:?}"),
            }

            // sc-18097: the fail-closed claim survives estimate admission because the synthesis
            // fails closed on the WHOLE TIER — the tier's phase curves are fitted across every
            // measured cell, and the measured path requires each record's peak to agree with the
            // curve within 0.01 GiB, so a sibling-anchored estimate would reproduce the excluded
            // cell's own number behind nothing but the 4% margin. Both corruption shapes are
            // therefore asserted: every record of the tier (below) and, on the one tier with two
            // exact_request cells, the request's own record alone (the second arm) — a per-cell
            // structural exclusion must not be laundered by a surviving sibling.
            let mutate_compositions = |record: &mut Value| {
                record["measuredCompositions"]["tiledVae"] = json!(["resident", "bounded_decode"]);
                record["measuredCompositions"]["chunkedAttention"] =
                    json!(["resident", "bounded_decode", "bounded_attention"]);
                record["measuredCompositions"]["streamedBlocks"] = json!([
                    "resident",
                    "bounded_decode",
                    "bounded_attention",
                    "bounded_transformer_residency"
                ]);
            };
            let mut no_compatible_deeper_row = manifest.clone();
            for record in no_compatible_deeper_row["candle"]["turboFit"]["evidenceRecords"]
                .as_array_mut()
                .expect("evidence records")
                .iter_mut()
                .filter(|record| record["tier"].as_str() == Some(tier))
            {
                mutate_compositions(record);
            }
            assert_eq!(
                krea_turbo_fit(
                    &no_compatible_deeper_row,
                    tier,
                    1024,
                    1024,
                    Some(VramBudget {
                        free_gb: required,
                        total_gb: required,
                    }),
                    true,
                ),
                Some(KreaTurboFit::Unverified {
                    reason: gen_core::MemoryEvidenceVerdict::CompositionMismatch,
                }),
                "{tier} mismatched rows must fail closed when no exact deeper composition can fit"
            );

            // The laundering arm, on the one tier with a second `exact_request` cell (q4 ships
            // 768² as well as 1024²): corrupting ONLY the request's own record must NOT be
            // rescued by the surviving 768² sibling. The tier's curve at 1024² reproduces the
            // corrupted record's peak to within the measured path's own 0.01 GiB agreement
            // conjunct, so a best-anchor synthesis would re-admit the excluded cell at its own
            // number behind nothing but the estimate margin. Whole-tier fail-closed is what stops
            // that, and this arm is its mutation check: switching the synthesis back to
            // "any eligible anchor wins" turns this green-to-red.
            if tier == "q4" {
                let mut single_corrupted = manifest.clone();
                mutate_compositions(
                    single_corrupted["candle"]["turboFit"]["evidenceRecords"]
                        .as_array_mut()
                        .expect("evidence records")
                        .iter_mut()
                        .find(|record| {
                            record["tier"].as_str() == Some(tier)
                                && record["width"].as_u64() == Some(1024)
                        })
                        .expect("1024 evidence record"),
                );
                assert_eq!(
                    krea_turbo_fit(
                        &single_corrupted,
                        tier,
                        1024,
                        1024,
                        Some(VramBudget {
                            free_gb: required,
                            total_gb: required,
                        }),
                        true,
                    ),
                    Some(KreaTurboFit::Unverified {
                        reason: gen_core::MemoryEvidenceVerdict::CompositionMismatch,
                    }),
                    "{tier}: a per-cell structural exclusion must not be laundered into an \
                     estimate anchored on a surviving sibling record"
                );
            }
        }
    }

    #[test]
    fn historical_builtin_krea_q4_curves_select_24_16_and_12_gb_rungs() {
        let manifest = builtin_krea_turbo_manifest_with_original_fingerprint();
        let fit = |free_gb| {
            krea_turbo_fit(
                &manifest,
                "q4",
                1024,
                1024,
                Some(VramBudget {
                    free_gb,
                    total_gb: free_gb,
                }),
                true,
            )
        };
        assert!(matches!(fit(96.0), Some(KreaTurboFit::Resident { .. })));
        assert!(matches!(
            fit(24.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    ..
                },
                ..
            })
        ));
        // sc-17097: bounded-decode re-measured 15.41 -> 13.32 GiB at 1024^2, so a 16 GiB card now
        // stops one rung EARLIER than it did under the ABI-1 curves - it no longer has to chunk
        // attention to fit. The ladder order is unchanged; the rung that fits moved.
        assert!(matches!(
            fit(16.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedDecode,
                    ..
                },
                ..
            })
        ));
        // Moving the 16 GiB probe down to bounded-decode left bounded-attention unprobed: after the
        // sc-17097 re-measurement it is selected only for free_gb in [12.54, 15.32). Probe inside
        // that window so every rung the ladder can still reach keeps a golden.
        assert!(matches!(
            fit(14.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedAttention,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            fit(12.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedTransformerResidency,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            krea_turbo_fit(
                &manifest,
                "q4",
                2048,
                2048,
                Some(VramBudget {
                    free_gb: 12.0,
                    total_gb: 12.0,
                }),
                true,
            ),
            Some(KreaTurboFit::Unverified { .. })
        ));
        assert!(matches!(
            krea_turbo_fit(
                &manifest,
                "q4",
                2048,
                2048,
                Some(VramBudget {
                    free_gb: 17.0,
                    total_gb: 17.0,
                }),
                true,
            ),
            Some(KreaTurboFit::Unverified { .. })
        ));
    }

    #[test]
    fn historical_builtin_krea_q8_curves_keep_q8_and_select_only_measured_useful_rungs() {
        let manifest = builtin_krea_turbo_manifest_with_original_fingerprint();
        let fit = |free_gb| {
            krea_turbo_fit(
                &manifest,
                "q8",
                1024,
                1024,
                Some(VramBudget {
                    free_gb,
                    total_gb: free_gb,
                }),
                true,
            )
        };
        assert!(matches!(
            fit(32.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    ..
                },
                ..
            })
        ));
        // sc-17097: the Q8 tiled-VAE rung is no longer a measured no-op (see below), so 24 GiB now
        // fits bounded-decode and 20 GiB fits bounded-attention - each card stops one rung earlier.
        assert!(matches!(
            fit(24.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedDecode,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            fit(20.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedAttention,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            fit(12.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedTransformerResidency,
                    ..
                },
                ..
            })
        ));

        let three_stage =
            krea_rung_phase_peaks(&manifest, "q8", MemoryStrategy::StagedResidency, 1024, 1024)
                .expect("Q8 three-stage evidence");
        let tiled =
            krea_rung_phase_peaks(&manifest, "q8", MemoryStrategy::BoundedDecode, 1024, 1024)
                .expect("Q8 tiled-VAE evidence");
        // sc-17097 INVERTS this assertion, and the inversion is the finding. Under the ABI-1
        // capture the Q8 decode peak was identical with and without tiling (16.514 -> 16.514), so
        // tiled VAE was a measured no-op that had to be prevented from displacing the cheaper
        // three-stage rung. Re-measured under ABI 3 it is a real saving: three-stage peaks at 25.92
        // GiB (decode-bound) against tiled VAE's 19.04. Keeping the old direction would now assert
        // that a rung which demonstrably helps must not be offered.
        assert!(
            tiled.peak_gb() < three_stage.peak_gb(),
            "the re-measured Q8 tiled-VAE rung is a real decode saving ({:.3} GiB against \
             three-stage {:.3}) and must be selectable",
            tiled.peak_gb(),
            three_stage.peak_gb()
        );

        // sc-17097: the Q8 streamed-block floor re-measured 6.85 -> 5.01 GiB, so the admit/reject
        // boundary moved from ~8.95 to ~7.01 GiB (peak plus the 2 GiB reserve). Probed 0.01 GiB clear
        // of the boundary on each side rather than exactly on it: the byte-exact tie is decided by
        // f64 rounding of `free_gb - headroom`, which is not the behaviour under test.
        assert!(matches!(
            fit(7.02),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedTransformerResidency,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(fit(7.00), Some(KreaTurboFit::Reject { .. })));
        let budget = Some(VramBudget {
            free_gb: 7.00,
            total_gb: 7.00,
        });
        assert_eq!(
            krea_turbo_smaller_fit(&manifest, "q8", 1024, 1024, budget, true),
            None,
            "lower-aspect curves without exact parity records must not be recommended"
        );
        let no_escape_budget = Some(VramBudget {
            free_gb: 6.5,
            total_gb: 6.5,
        });
        assert_eq!(
            krea_turbo_smaller_fit(&manifest, "q8", 1024, 1024, no_escape_budget, true,),
            None,
            "do not claim a lower-resolution escape when no shipped measured shape fits"
        );
    }

    #[test]
    fn builtin_krea_q8_curves_conservatively_cover_every_measured_phase_sample() {
        let manifest = builtin_krea_turbo_manifest_at_live_closure();
        let samples = [
            (
                MemoryStrategy::StagedResidency,
                768,
                [5.003, 15.690, 21.411],
            ),
            (
                MemoryStrategy::StagedResidency,
                1024,
                [5.003, 19.159, 25.911],
            ),
            (MemoryStrategy::BoundedDecode, 768, [5.003, 15.503, 13.692]),
            (MemoryStrategy::BoundedDecode, 1024, [5.003, 19.034, 13.724]),
            (
                MemoryStrategy::BoundedAttention,
                768,
                [5.003, 14.878, 13.661],
            ),
            (
                MemoryStrategy::BoundedAttention,
                1024,
                [5.003, 15.253, 13.724],
            ),
            (
                MemoryStrategy::BoundedTransformerResidency,
                768,
                [5.003, 4.878, 4.411],
            ),
            (
                MemoryStrategy::BoundedTransformerResidency,
                1024,
                [5.003, 4.878, 3.661],
            ),
        ];
        for (rung, edge, measured) in samples {
            let predicted = krea_rung_phase_peaks(&manifest, "q8", rung, edge, edge)
                .expect("every published Q8 matrix cell has a curve");
            for (phase, predicted_gb, measured_gb) in [
                ("text", predicted.text_gb, measured[0]),
                ("denoise", predicted.denoise_gb, measured[1]),
                ("decode", predicted.decode_gb, measured[2]),
            ] {
                assert!(
                    predicted_gb >= measured_gb && predicted_gb < measured_gb + 1.0,
                    "{rung:?} {edge}² {phase}: curve {predicted_gb:.3} must conservatively cover \
                     measured {measured_gb:.3} without an unrelated tier-derived margin"
                );
            }
        }
    }

    /// sc-18097 repin of `q8_and_bf16_768_phase_fit_records_do_not_overclaim_exact_runtime_admission`.
    ///
    /// The 768² cells carry `phase_fit_only` records: they characterize the fitted curves without
    /// authorizing EXACT runtime admission, and pre-18097 the gate therefore froze to
    /// `Unverified { OutOfEnvelope }`. The estimate ladder retires the freeze without weakening
    /// the original claim — 768² is admitted (or refused) by a fitted-curve ESTIMATE graded behind
    /// the candle estimate margin, never by an exact-verified claim, and the sc-18094
    /// binding-phase constraint decides per rung whether the extrapolation is even allowed:
    ///
    /// * q8: every shipped rung keeps its anchor's binding phase at 768² (streamed-blocks binds on
    ///   the area-flat text phase at both geometries), so the deep rung admits by estimate on a
    ///   card that fits its widened floor — and refuses below it (the margin mutation arm).
    /// * bf16: the streamed-blocks anchor binds on DENOISE at 1024² (8.420 against 8.100 text)
    ///   while the fitted curve binds on TEXT at 768² — a binding-phase flip, so
    ///   `ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE` refuses the fitted candidate and
    ///   the 12 GiB request stays `Unverified` on the SHIPPED data: a live demonstration that the
    ///   constraint is enforced at this synthesis seam, on real curves.
    #[test]
    fn q8_and_bf16_768_phase_fit_cells_are_estimate_graded_never_exactly_admitted() {
        let manifest = builtin_krea_turbo_manifest_with_original_fingerprint();
        let fit = |tier: &str, free_gb: f64| {
            krea_turbo_fit(
                &manifest,
                tier,
                768,
                768,
                Some(VramBudget {
                    free_gb,
                    total_gb: free_gb,
                }),
                true,
            )
        };

        // q8 at 12 GiB free (10 GiB effective): the streamed-blocks fitted estimate (~5.01 GiB
        // text-bound peak, widened ~5.21) is the only rung that fits, and it must carry the
        // measured strategy parameters.
        match fit("q8", 12.0) {
            Some(KreaTurboFit::Fits {
                selection, memory, ..
            }) => {
                assert_eq!(
                    selection.strategy,
                    MemoryStrategy::BoundedTransformerResidency,
                    "the deep rung's fitted estimate must admit the 768² q8 request"
                );
                assert_eq!(selection.parameters.transformer_window_size, Some(1));
                assert!(memory.tile_vae_decode);
                assert!(memory.chunk_attention);
                assert!(memory.stream_transformer_blocks);
            }
            other => panic!("q8 768² must admit by fitted estimate, got {other:?}"),
        }
        // Margin mutation arm (q8): at 7.15 GiB free (5.15 effective) the RAW streamed peak
        // (~5.01) fits but the widened one (~5.21) does not — a selector whose estimate margin is
        // zeroed admits here and flips this arm red. Every rung carries an eligible estimate, so
        // the refusal is the honest margins-based `Reject`, not `Unverified`.
        assert!(
            matches!(fit("q8", 7.15), Some(KreaTurboFit::Reject { .. })),
            "below the widened deep-rung estimate the q8 768² request must reject"
        );

        // bf16 at 12 GiB free: the binding-phase flip suppresses the streamed-blocks fitted
        // estimate and no shallower rung fits, so the request remains a structural refusal —
        // phase-fit-only evidence still cannot overclaim past the sc-18094 constraint.
        let bf16 = fit("bf16", 12.0);
        assert!(
            matches!(bf16, Some(KreaTurboFit::Unverified { .. })),
            "bf16 768² must stay refused: the anchor's binding phase flips at this geometry; \
             got {bf16:?}"
        );
    }

    #[test]
    fn builtin_krea_q8_and_bf16_slopes_are_fitted_from_same_tier_samples() {
        let manifest = builtin_krea_turbo_manifest_at_live_closure();
        let turbo_fit = &manifest["candle"]["turboFit"];
        let samples = [
            ("q8", "threeStage", MemoryStrategy::StagedResidency),
            ("q8", "tiledVae", MemoryStrategy::BoundedDecode),
            ("q8", "chunkedAttention", MemoryStrategy::BoundedAttention),
            (
                "q8",
                "streamedBlocks",
                MemoryStrategy::BoundedTransformerResidency,
            ),
            ("bf16", "threeStage", MemoryStrategy::StagedResidency),
            ("bf16", "tiledVae", MemoryStrategy::BoundedDecode),
            ("bf16", "chunkedAttention", MemoryStrategy::BoundedAttention),
            (
                "bf16",
                "streamedBlocks",
                MemoryStrategy::BoundedTransformerResidency,
            ),
        ];
        let megapixel_delta = (1024_f64.powi(2) - 768_f64.powi(2)) / 1_000_000.0;

        for (tier, manifest_rung, rung) in samples {
            let record = |edge| {
                turbo_fit["evidenceRecords"]
                    .as_array()
                    .expect("evidence records")
                    .iter()
                    .find(|record| {
                        record["tier"] == tier
                            && record["width"] == edge
                            && record["height"] == edge
                    })
                    .expect("same-tier geometry record")
            };
            let measured_phase = |record: &Value, phase| {
                record["observedPhasesGb"][manifest_rung][phase]
                    .as_f64()
                    .expect("machine-readable measured phase")
            };
            let measured_768 = record(768);
            let measured_1024 = record(1024);
            let predicted_768 = krea_rung_phase_peaks(&manifest, tier, rung, 768, 768)
                .expect("768² fitted phase vector");
            let predicted_1024 = krea_rung_phase_peaks(&manifest, tier, rung, 1024, 1024)
                .expect("1024² fitted phase vector");
            for (phase, lower_prediction, upper_prediction, lower_measured, upper_measured) in [
                (
                    "text",
                    predicted_768.text_gb,
                    predicted_1024.text_gb,
                    measured_phase(measured_768, "text"),
                    measured_phase(measured_1024, "text"),
                ),
                (
                    "denoise",
                    predicted_768.denoise_gb,
                    predicted_1024.denoise_gb,
                    measured_phase(measured_768, "denoise"),
                    measured_phase(measured_1024, "denoise"),
                ),
                (
                    "decode",
                    predicted_768.decode_gb,
                    predicted_1024.decode_gb,
                    measured_phase(measured_768, "decode"),
                    measured_phase(measured_1024, "decode"),
                ),
            ] {
                let fitted_slope = (upper_prediction - lower_prediction) / megapixel_delta;
                let measured_slope = ((upper_measured - lower_measured) / megapixel_delta).max(0.0);
                assert!(
                    fitted_slope + 1e-9 >= measured_slope && fitted_slope < measured_slope + 0.02,
                    "{tier} {rung:?} {phase}: slope {fitted_slope:.4} must be the conservative \
                     two-decimal fit of this tier's measured {measured_slope:.4}, not another \
                     tier's coefficient"
                );
            }
        }
    }

    #[test]
    fn historical_builtin_krea_bf16_curves_keep_bf16_and_select_only_measured_useful_rungs() {
        let manifest = builtin_krea_turbo_manifest_with_original_fingerprint();
        let fit = |free_gb| {
            krea_turbo_fit(
                &manifest,
                "bf16",
                1024,
                1024,
                Some(VramBudget {
                    free_gb,
                    total_gb: free_gb,
                }),
                true,
            )
        };
        assert!(matches!(
            fit(48.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    ..
                },
                ..
            })
        ));
        // Newly reachable: under the ABI-1 curves the BF16 tiled-VAE and three-stage peaks were tied
        // at 32.1157 GiB, so bounded-decode could never be selected. The re-measurement separates
        // them (29.86 against 36.73), making this a live rung for free_gb in [31.86, 38.73).
        assert!(matches!(
            fit(33.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedDecode,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            fit(31.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedAttention,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            fit(12.0),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedTransformerResidency,
                    ..
                },
                ..
            })
        ));

        let three_stage = krea_rung_phase_peaks(
            &manifest,
            "bf16",
            MemoryStrategy::StagedResidency,
            1024,
            1024,
        )
        .expect("BF16 three-stage evidence");
        let tiled =
            krea_rung_phase_peaks(&manifest, "bf16", MemoryStrategy::BoundedDecode, 1024, 1024)
                .expect("BF16 tiled-VAE evidence");
        // sc-17097 inverts this for the same reason as the Q8 sibling: the ABI-1 BF16 decode peak
        // was flat across tiling (26.446 -> 26.446), so tiled VAE measured as a no-op. Re-measured
        // it is a real saving - three-stage 36.73 GiB against tiled VAE 29.86.
        assert!(
            tiled.peak_gb() < three_stage.peak_gb(),
            "the re-measured BF16 tiled-VAE rung is a real decode saving ({:.3} GiB against \
             three-stage {:.3}) and must be selectable",
            tiled.peak_gb(),
            three_stage.peak_gb()
        );

        // sc-17097: the BF16 streamed floor re-measured 8.53 -> 8.42 GiB, moving the boundary from
        // ~10.63 to ~10.42. Probed 0.01 GiB clear on each side, as for Q8.
        assert!(matches!(
            fit(10.43),
            Some(KreaTurboFit::Fits {
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedTransformerResidency,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(fit(10.41), Some(KreaTurboFit::Reject { .. })));
        let immediate_below = Some(VramBudget {
            free_gb: 10.41,
            total_gb: 10.41,
        });
        assert_eq!(
            krea_turbo_smaller_fit(&manifest, "bf16", 1024, 1024, immediate_below, true),
            None,
            "BF16's measured streamed text floor is resolution-independent, so reject copy must not \
             promise that lowering resolution can cross the immediate-below boundary"
        );
    }

    #[test]
    fn builtin_krea_bf16_curves_conservatively_cover_every_measured_phase_sample() {
        let manifest = builtin_krea_turbo_manifest_at_live_closure();
        let samples = [
            (
                MemoryStrategy::StagedResidency,
                768,
                [7.940, 26.565, 32.255],
            ),
            (
                MemoryStrategy::StagedResidency,
                1024,
                [7.972, 29.972, 36.724],
            ),
            (MemoryStrategy::BoundedDecode, 768, [8.097, 26.472, 24.599]),
            (MemoryStrategy::BoundedDecode, 1024, [8.097, 29.847, 24.630]),
            (
                MemoryStrategy::BoundedAttention,
                768,
                [8.097, 25.659, 24.599],
            ),
            (
                MemoryStrategy::BoundedAttention,
                1024,
                [8.097, 25.753, 24.599],
            ),
            (
                MemoryStrategy::BoundedTransformerResidency,
                768,
                [8.097, 7.940, 4.474],
            ),
            (
                MemoryStrategy::BoundedTransformerResidency,
                1024,
                [7.940, 8.408, 3.848],
            ),
        ];
        for (rung, edge, measured) in samples {
            let predicted = krea_rung_phase_peaks(&manifest, "bf16", rung, edge, edge)
                .expect("every published BF16 matrix cell has a curve");
            for (phase, predicted_gb, measured_gb) in [
                ("text", predicted.text_gb, measured[0]),
                ("denoise", predicted.denoise_gb, measured[1]),
                ("decode", predicted.decode_gb, measured[2]),
            ] {
                assert!(
                    predicted_gb >= measured_gb && predicted_gb < measured_gb + 1.0,
                    "{rung:?} {edge}² {phase}: curve {predicted_gb:.3} must conservatively cover \
                     measured {measured_gb:.3} without a lower-tier-derived margin"
                );
            }
        }
    }

    /// sc-14625: the complete default tuple passed on a real 32 GB RTX PRO 4500. Admit exactly that
    /// measured boundary without generalizing any one setting or retaining the old 48 GB guess.
    #[test]
    fn svd_cuda_preflight_admits_the_complete_validated_32gb_recipe() {
        let card_32 = apply_vram_cap(None, Some(32.0));
        assert!(
            svd_fit_error(25, 8, 25, 1024, 576, "0", card_32).is_none(),
            "the real-hardware-validated default profile must be admitted on 32 GB"
        );
        let too_small = svd_fit_error(25, 8, 25, 1024, 576, "0", apply_vram_cap(None, Some(16.0)))
            .expect("the 17.521 GiB measured peak cannot be admitted on a 16 GB card");
        assert!(
            too_small.to_string().contains("at least an 18 GB"),
            "the rejection must explain the measured minimum: {too_small}"
        );
        let over = svd_fit_error(26, 8, 25, 1024, 576, "0", card_32)
            .expect("a recipe beyond the measured frame boundary must still reject on 32 GB");
        let message = over.to_string();
        assert!(
            !message.contains("48 GB")
                && message.contains("1024x576")
                && message.contains("25 frames")
                && message.contains("decodeChunkSize=8")
                && message.contains("25 steps"),
            "rejection must report only the complete measured 32 GB boundary: {message}"
        );
        assert!(
            svd_fit_error(25, 9, 25, 1024, 576, "0", card_32).is_some()
                && svd_fit_error(25, 8, 26, 1024, 576, "0", card_32).is_some()
                && svd_fit_error(25, 8, 25, 1025, 576, "0", card_32).is_some(),
            "changing any unproven setting must reject instead of extrapolating the measured tuple"
        );
        assert!(
            svd_fit_error(26, 9, 26, 1025, 577, "0", apply_vram_cap(None, Some(48.0))).is_none(),
            "larger cards remain admitted without claiming an unmeasured minimum"
        );
        assert!(
            svd_fit_error(26, 9, 26, 1025, 577, "0", None).is_none(),
            "missing telemetry must preserve the fit-gate convention: admit without invented evidence"
        );
    }

    #[test]
    fn requested_tier_key_mirrors_the_subdir_resolvers() {
        let empty = obj(json!({}));
        // No pick anywhere ⇒ q8 default (sc-10726).
        assert_eq!(requested_tier_key(&empty, &empty, false), "q8");
        // advanced.mlxQuantize wins, number or numeric string.
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": 0})), &empty, false),
            "bf16"
        );
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": 4})), &empty, false),
            "q4"
        );
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": 8})), &empty, false),
            "q8"
        );
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": "4"})), &empty, false),
            "q4"
        );
        // Falls back to the manifest mlx.quantize when advanced is silent.
        assert_eq!(
            requested_tier_key(&empty, &obj(json!({"mlx": {"quantize": 4}})), false),
            "q4"
        );
    }

    /// sc-11042 (epic 11037 SC#5): the fit gate keys off the tier IDENTITY, never bits.
    ///
    /// The FOURTH bits-derived site of the same aliasing. NVFP4 carries no `mlxQuantize` by design (no
    /// integer is honest for a ~4.5-effective-bit tier), so a bits-derived key returned `"q8"` for it
    /// and sized an NVFP4 render against `vramGbByTier["q8"]` — ~2× its real footprint. Conservative
    /// (spurious TooBig/Offload, never an OOM), but wrong, and it made this the one selection site that
    /// disagreed with the load quant + the recorded label about which tier was running.
    #[test]
    fn requested_tier_key_honors_the_nvfp4_tier_identity() {
        let empty = obj(json!({}));
        // The selected NVFP4 tier short-circuits the bits map — including the `mlxQuantize: null` an
        // NVFP4 request actually carries (sc-12006), which `quant_int` reads as "no pick" ⇒ `q8`.
        assert_eq!(requested_tier_key(&empty, &empty, true), NVFP4_TIER);
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": null})), &empty, true),
            NVFP4_TIER
        );
        // …and it is NOT the q4 whose numerics it must never be confused with, nor the q8 the
        // bits-derived key produced.
        assert_ne!(requested_tier_key(&empty, &empty, true), "q4");
        assert_ne!(requested_tier_key(&empty, &empty, true), "q8");

        // `nvfp4 = false` ⇒ every existing mapping is byte-identical (the caller's `nvfp4_selected` is
        // false for every request that didn't explicitly pick the tier on eligible hardware with the
        // tier installed — i.e. all of them today).
        for (advanced, manifest, expected) in [
            (json!({}), json!({}), "q8"),
            (json!({"mlxQuantize": 0}), json!({}), "bf16"),
            (json!({"mlxQuantize": 4}), json!({}), "q4"),
            (json!({"mlxQuantize": 8}), json!({}), "q8"),
            (json!({}), json!({"mlx": {"quantize": 4}}), "q4"),
            // A `quantTier: "nvfp4"` label with the gate closed (not Blackwell, or the tier isn't
            // installed) sizes the tier that will actually load — never nvfp4.
            (json!({"quantTier": "nvfp4"}), json!({}), "q8"),
        ] {
            assert_eq!(
                requested_tier_key(&obj(advanced.clone()), &obj(manifest.clone()), false),
                expected,
                "advanced={advanced} manifest={manifest}"
            );
        }
    }

    /// sc-11042: an `nvfp4` tier with no measured `vramGbByTier` row degrades CONSERVATIVELY — to the
    /// `q8` row, not to `minMemoryGb`.
    ///
    /// `minMemoryGb` is the manifest's DEFAULT (lightest, typically q4) tier peak, which heavier tiers
    /// are explicitly allowed to exceed — landing there would UNDER-predict and admit a load that can
    /// OOM. The q8 row over-predicts (q8's weights are ~2× NVFP4's), which is exactly the number this
    /// gate already used for an NVFP4 request when the bits-derived key returned `"q8"`. sc-11043 must
    /// add the real `nvfp4` rows when it converts a tier; until then this is the documented behavior.
    /// sc-12425 — **the INT8-ConvRot row must be READ, and must exceed the q8 row it used to alias to.**
    ///
    /// The NVFP4 aliasing described above (`requested_tier_key` returning `"q8"` for a tier with no
    /// honest `mlxQuantize`) hit ConvRot identically — but with the sign flipped, and the flip is the
    /// bug. For NVFP4, q8 OVER-predicts, so the aliasing only cost a spurious `TooBig`/`Offload`
    /// ("never an OOM"). For INT8-ConvRot q8 **under**-predicts: measured on a real trunk (sc-12381,
    /// exclusive sm_120, 1024²/8-step) the tier peaks at **42.9 GB**, while q8's row predicts
    /// 35.9 + 2.0 = 37.9 — the gate admitted a load that OOMs by ~5 GB. `image_jobs::base` now names the
    /// tier by IDENTITY ([`INT8_CONVROT_TIER`], off a RESOLVED ConvRot load) rather than handing it to
    /// the bits-derived key.
    ///
    /// Until sc-12425 this row was **dead** — nothing looked it up — which is exactly why its unmeasured
    /// 31.0 estimate survived: a row nothing reads cannot be wrong out loud. It is load-bearing now.
    #[test]
    fn int8_convrot_is_sized_by_its_own_measured_row_not_the_q8_row_it_aliased_to() {
        // The shipping Krea 2 Turbo shape, with sc-12425's corrected row.
        let manifest = obj(json!({
            "candle": {
                "minMemoryGb": 32,
                "vramGbByTier": { "q4": 26.4, "q8": 35.9, "bf16": 55.6, "int8-convrot": 42.9 }
            }
        }));
        /// The real-trunk overall-peak (sc-12381). The gate must predict at least this.
        const MEASURED_PEAK_GB: f64 = 42.9;

        // The row is consulted at all — the regression that would silently restore the dead row.
        let convrot = predicted_peak_gb(&manifest, INT8_CONVROT_TIER).expect("convrot row");
        assert_eq!(convrot, MEASURED_PEAK_GB + HEADROOM_GB);

        // The OOM sc-12425 fixes: q8's row under-predicts a 42.9 GB render.
        let via_q8 = predicted_peak_gb(&manifest, "q8").expect("q8 row");
        assert!(
            convrot > via_q8,
            "INT8-ConvRot must out-predict q8 ({convrot} vs {via_q8}); were q8 >= it, the pre-sc-12425 \
             aliasing would have been harmless and this story would not exist"
        );
        assert!(
            convrot >= MEASURED_PEAK_GB,
            "the gate must predict at least the MEASURED peak ({MEASURED_PEAK_GB} GB); {convrot} would \
             admit a render that OOMs. Re-measure with `krea-convrot-vram` (sc-12381), don't re-estimate."
        );
        assert!(
            via_q8 < MEASURED_PEAK_GB,
            "sanity: q8's row is expected to UNDER-predict the measured ConvRot peak — that under-\
             prediction IS the defect. If this trips, the q8 row moved and the story's framing needs \
             re-reading, not a threshold bump."
        );
    }

    #[test]
    fn nvfp4_without_a_measured_row_degrades_to_q8_not_min_memory() {
        let manifest = obj(json!({
            "candle": {
                "minMemoryGb": 40,
                "vramGbByTier": { "q4": 47.2, "q8": 58, "bf16": 72 },
                "sequentialPeakGb": { "q4": 20.0, "q8": 26.0 }
            }
        }));
        // No `nvfp4` row ⇒ the q8 row + headroom, NOT the lighter (permissive) minMemoryGb…
        assert_eq!(
            predicted_peak_gb(&manifest, NVFP4_TIER),
            Some(58.0 + HEADROOM_GB)
        );
        assert_ne!(predicted_peak_gb(&manifest, NVFP4_TIER), Some(40.0));
        // …and the same for the second-stage sequential gate.
        assert_eq!(
            predicted_sequential_peak_gb(&manifest, NVFP4_TIER),
            Some(26.0 + HEADROOM_GB)
        );

        // Once sc-11043 measures the tier, its own row wins outright.
        let measured = obj(json!({
            "candle": {
                "minMemoryGb": 40,
                "vramGbByTier": { "q8": 58, "nvfp4": 31.5 },
                "sequentialPeakGb": { "q8": 26.0, "nvfp4": 14.0 }
            }
        }));
        assert_eq!(
            predicted_peak_gb(&measured, NVFP4_TIER),
            Some(31.5 + HEADROOM_GB)
        );
        assert_eq!(
            predicted_sequential_peak_gb(&measured, NVFP4_TIER),
            Some(14.0 + HEADROOM_GB)
        );

        // The q8 back-stop is NVFP4-only: every other tier keeps the unchanged
        // measured-row → minMemoryGb chain.
        let sparse = obj(json!({ "candle": { "minMemoryGb": 40, "vramGbByTier": { "q8": 58 } } }));
        assert_eq!(predicted_peak_gb(&sparse, "q4"), Some(40.0));
        assert_eq!(predicted_peak_gb(&sparse, "bf16"), Some(40.0));
        assert_eq!(predicted_sequential_peak_gb(&sparse, "q4"), None);
        // No q8 row either ⇒ nvfp4 falls all the way through to minMemoryGb rather than panicking.
        let no_q8 = obj(json!({ "candle": { "minMemoryGb": 40, "vramGbByTier": {} } }));
        assert_eq!(predicted_peak_gb(&no_q8, NVFP4_TIER), Some(40.0));
        assert_eq!(predicted_sequential_peak_gb(&no_q8, NVFP4_TIER), None);
    }

    #[test]
    fn predicted_peak_prefers_measured_tier_then_min_memory() {
        let manifest = obj(json!({
            "candle": {
                "minMemoryGb": 56,
                "vramGbByTier": { "q4": 47.2, "q8": 58, "bf16": 72 }
            }
        }));
        // Measured tier peak + headroom.
        assert_eq!(predicted_peak_gb(&manifest, "q4"), Some(47.2 + HEADROOM_GB));
        // Missing tier in vramGbByTier ⇒ minMemoryGb (no extra headroom, already padded).
        let sparse = obj(json!({ "candle": { "minMemoryGb": 40, "vramGbByTier": {} } }));
        assert_eq!(predicted_peak_gb(&sparse, "q4"), Some(40.0));
        // No candle block ⇒ unmeasured ⇒ None (gate no-ops).
        assert_eq!(predicted_peak_gb(&obj(json!({})), "q4"), None);
    }

    #[test]
    fn adapter_bytes_change_resident_and_sequential_fit_boundaries() {
        let manifest = obj(json!({
            "candle": {
                "vramGbByTier": { "q4": 7.0 },
                "sequentialPeakGb": { "q4": 6.0 }
            }
        }));
        let one_gib = BYTES_PER_GIB as u64;
        assert_eq!(
            predicted_peak_gb_with_adapter_bytes(&manifest, "q4", 0),
            Some(7.0 + HEADROOM_GB)
        );
        assert_eq!(
            predicted_peak_gb_with_adapter_bytes(&manifest, "q4", one_gib),
            Some(8.0 + HEADROOM_GB)
        );
        assert_eq!(
            predicted_sequential_peak_gb_with_adapter_bytes(&manifest, "q4", one_gib),
            Some(7.0 + HEADROOM_GB)
        );
        let boundary = VramBudget {
            free_gb: 7.5 + HEADROOM_GB,
            total_gb: 16.0,
        };
        assert_eq!(
            fit_decision(
                predicted_peak_gb_with_adapter_bytes(&manifest, "q4", 0),
                Some(boundary)
            ),
            FitDecision::Fits
        );
        assert!(matches!(
            fit_decision(
                predicted_peak_gb_with_adapter_bytes(&manifest, "q4", one_gib),
                Some(boundary)
            ),
            FitDecision::TooBig { .. }
        ));
    }

    /// sc-12090 numeric regression: `krea_2_turbo` Q4-only on a ~30 GB card. Budgeting the tier the
    /// disk-probing resolver returns (`q4`) ADMITS (26.4 + 2 = 28.4 ≤ 30), where the old manifest
    /// re-derivation budgeted `q8` (35.9 + 2 = 37.9) and false-rejected. This pins the fit math the
    /// gate now feeds the ON-DISK tier (`tier_key_from_resolved_dir`), not the manifest's q8 default.
    #[test]
    fn resolved_q4_admits_where_manifest_q8_would_reject() {
        // Krea 2 Turbo candle tiers (builtin.models.jsonc, measured — sc-12126).
        let manifest = obj(json!({
            "candle": { "vramGbByTier": { "q4": 26.4, "q8": 35.9 } }
        }));
        let budget = VramBudget {
            free_gb: 30.0,
            total_gb: 32.0,
        };
        // The resolved on-disk tier is q4 (only q4 installed) → admits.
        assert_eq!(
            fit_decision(predicted_peak_gb(&manifest, "q4"), Some(budget)),
            FitDecision::Fits
        );
        // The old manifest-derived q8 (never installed) would have false-rejected the SAME job.
        assert!(matches!(
            fit_decision(predicted_peak_gb(&manifest, "q8"), Some(budget)),
            FitDecision::TooBig { .. }
        ));
    }

    #[test]
    fn fit_decision_rejects_only_a_genuine_overflow() {
        let budget = VramBudget {
            free_gb: 10.0,
            total_gb: 10.0,
        };
        assert_eq!(fit_decision(Some(8.0), Some(budget)), FitDecision::Fits);
        // Exactly-fits is not a rejection.
        assert_eq!(fit_decision(Some(10.0), Some(budget)), FitDecision::Fits);
        assert_eq!(
            fit_decision(Some(49.2), Some(budget)),
            FitDecision::TooBig {
                needed_gb: 49.2,
                available_gb: 10.0,
            }
        );
        // Missing either input ⇒ never block.
        assert_eq!(fit_decision(None, Some(budget)), FitDecision::Unknown);
        assert_eq!(fit_decision(Some(8.0), None), FitDecision::Unknown);
    }

    #[test]
    fn resolve_offload_rewrites_toobig_only_when_sequential_capable() {
        let budget = VramBudget {
            free_gb: 10.0,
            total_gb: 10.0,
        };
        let too_big = fit_decision(Some(40.0), Some(budget));
        assert!(matches!(too_big, FitDecision::TooBig { .. }));
        // Sequential-capable (the candle FLUX lane) ⇒ Offload instead of reject, carrying the numbers.
        assert_eq!(
            resolve_offload(too_big.clone(), true),
            FitDecision::Offload {
                needed_gb: 40.0,
                available_gb: 10.0,
            }
        );
        // Not capable ⇒ unchanged TooBig (still rejects).
        assert!(matches!(
            resolve_offload(too_big, false),
            FitDecision::TooBig { .. }
        ));
        // Fits / Unknown are never rewritten, regardless of capability.
        assert_eq!(resolve_offload(FitDecision::Fits, true), FitDecision::Fits);
        assert_eq!(
            resolve_offload(FitDecision::Unknown, true),
            FitDecision::Unknown
        );
    }

    #[test]
    fn predicted_sequential_peak_reads_the_measured_tier_plus_headroom() {
        let manifest = obj(json!({
            "candle": {
                "vramGbByTier": { "q4": 47.2, "q8": 58.0, "bf16": 72.0 },
                "sequentialPeakGb": { "q4": 30.0, "q8": 40.0, "bf16": 55.0 }
            }
        }));
        assert_eq!(
            predicted_sequential_peak_gb(&manifest, "q4"),
            Some(30.0 + HEADROOM_GB)
        );
        assert_eq!(
            predicted_sequential_peak_gb(&manifest, "bf16"),
            Some(55.0 + HEADROOM_GB)
        );
        // Tier absent from sequentialPeakGb ⇒ None (best-effort run, no reject).
        let sparse = obj(json!({ "candle": { "sequentialPeakGb": { "q4": 30.0 } } }));
        assert_eq!(predicted_sequential_peak_gb(&sparse, "q8"), None);
        // No sequentialPeakGb block ⇒ None (today's behavior: resident-only gate).
        let resident_only = obj(json!({ "candle": { "vramGbByTier": { "q4": 47.2 } } }));
        assert_eq!(predicted_sequential_peak_gb(&resident_only, "q4"), None);
        // No candle block ⇒ None.
        assert_eq!(predicted_sequential_peak_gb(&obj(json!({})), "q4"), None);
    }

    #[test]
    fn sequential_overflow_rejects_only_a_measured_genuine_overflow() {
        let budget = VramBudget {
            free_gb: 10.0,
            total_gb: 10.0,
        };
        // Measured sequential peak still exceeds the budget ⇒ reject, carrying the number.
        assert_eq!(sequential_overflow_gb(Some(32.0), Some(budget)), Some(32.0));
        // Sequential peak fits ⇒ proceed (None). Exactly-fits is not an overflow.
        assert_eq!(sequential_overflow_gb(Some(8.0), Some(budget)), None);
        assert_eq!(sequential_overflow_gb(Some(10.0), Some(budget)), None);
        // Unmeasured tier ⇒ best-effort run (None), even when the card is tiny.
        assert_eq!(sequential_overflow_gb(None, Some(budget)), None);
        // No live budget ⇒ never block (None).
        assert_eq!(sequential_overflow_gb(Some(32.0), None), None);
    }

    #[test]
    fn with_reclaimable_adds_the_pool_and_clamps_to_total() {
        let budget = VramBudget {
            free_gb: 14.0,
            total_gb: 96.0,
        };
        // A resident ~82 GB model is reclaimable → the incoming load sees free + 82, capped to total.
        // This is the sc-11023 fix: raw free (14) would reject a bf16 re-load that actually fits.
        assert_eq!(
            with_reclaimable(budget, 82.0),
            VramBudget {
                free_gb: 96.0,
                total_gb: 96.0,
            }
        );
        // Partial reclaim stays below total.
        assert_eq!(
            with_reclaimable(budget, 20.0),
            VramBudget {
                free_gb: 34.0,
                total_gb: 96.0,
            }
        );
        // Nothing reclaimable (cold start) ⇒ budget unchanged.
        assert_eq!(with_reclaimable(budget, 0.0), budget);
        // Negative is treated as zero (defensive).
        assert_eq!(with_reclaimable(budget, -5.0), budget);
        // Never exceeds the physical total, even with a huge/stale high-water.
        assert_eq!(
            with_reclaimable(budget, 1000.0),
            VramBudget {
                free_gb: 96.0,
                total_gb: 96.0,
            }
        );
    }

    /// [`load_plan`] resolves the residency ACTION for a sequential-capable lane and ignores the
    /// budget's reported numbers in its identity — the invariant the sc-13960 two-pass relies on.
    #[test]
    fn load_plan_resolves_resident_sequential_and_reject() {
        let needed = Some(56.7); // resident peak
        let seq = Some(30.0); // measured staged peak
        let big = Some(VramBudget {
            free_gb: 90.0,
            total_gb: 96.0,
        });
        let mid = Some(VramBudget {
            free_gb: 40.0,
            total_gb: 96.0,
        });
        let tiny = Some(VramBudget {
            free_gb: 10.0,
            total_gb: 96.0,
        });
        // Resident peak fits ⇒ Resident.
        assert_eq!(load_plan(needed, seq, big, true), LoadPlan::Resident);
        // Resident overflows but the staged peak fits ⇒ Sequential.
        assert_eq!(load_plan(needed, seq, mid, true), LoadPlan::Sequential);
        // Even the staged peak overflows ⇒ Reject.
        assert_eq!(load_plan(needed, seq, tiny, true), LoadPlan::Reject);
        // A NON-sequential-capable lane rejects a resident overflow outright (no staging).
        assert_eq!(load_plan(needed, seq, mid, false), LoadPlan::Reject);
        // Unmeasured peak or no budget ⇒ never block (Resident).
        assert_eq!(load_plan(None, seq, mid, true), LoadPlan::Resident);
        assert_eq!(load_plan(needed, seq, None, true), LoadPlan::Resident);
        // Two Rejects against different budgets are the SAME plan (no budget number in the identity),
        // so the two-pass never evicts for a reclaim that still can't fit.
        assert_eq!(
            load_plan(needed, seq, tiny, true),
            load_plan(needed, seq, mid, false)
        );
    }

    /// sc-13960: on a warm worker, folding the reclaimable pool FLIPS a bespoke edit lane's plan from a
    /// needless sequential downtier (and from an outright reject) back to a full-residency admit — the
    /// exact repeated-render scenarios the story names. Pins the arithmetic the two-pass evict-reclaim
    /// gate performs (`load_plan(raw)` vs `load_plan(with_reclaimable(raw, pool))`).
    #[test]
    fn reclaim_flips_a_warm_edit_gate_back_to_resident() {
        // Two q4 Qwen-Edits (peak 56.7) back-to-back on a 96 GB card: after edit #1 drops, cudarc holds
        // ~56.7 GB in-pool, so edit #2's RAW free is ~39.3 GB.
        let q4_needed = Some(56.7);
        let q4_seq = Some(30.0);
        let raw = Some(VramBudget {
            free_gb: 39.3,
            total_gb: 96.0,
        });
        // RAW free needlessly downtiers to sequential…
        assert_eq!(
            load_plan(q4_needed, q4_seq, raw, true),
            LoadPlan::Sequential
        );
        // …but crediting the 56.7 GB pool the first edit left behind readmits it RESIDENT.
        let reclaimed = raw.map(|b| with_reclaimable(b, 56.7));
        assert_eq!(
            load_plan(q4_needed, q4_seq, reclaimed, true),
            LoadPlan::Resident
        );

        // A repeated bf16 edit (81.7 resident / 52.2 sequential): after edit #1 drops, RAW free ~14.3 GB
        // — the staged 52.2 overflows, so RAW REJECTS the second render.
        let bf16_needed = Some(81.7);
        let bf16_seq = Some(52.2);
        let raw = Some(VramBudget {
            free_gb: 14.3,
            total_gb: 96.0,
        });
        assert_eq!(
            load_plan(bf16_needed, bf16_seq, raw, true),
            LoadPlan::Reject
        );
        // Reclaiming the 81.7 GB pool the first edit left behind readmits it RESIDENT.
        let reclaimed = raw.map(|b| with_reclaimable(b, 81.7));
        assert_eq!(
            load_plan(bf16_needed, bf16_seq, reclaimed, true),
            LoadPlan::Resident
        );

        // A cold pool (reclaimable 0) is a no-op: the raw plan stands, so a genuine first-load on a
        // constrained card is gated exactly as before.
        let reclaimed_cold = raw.map(|b| with_reclaimable(b, 0.0));
        assert_eq!(
            load_plan(bf16_needed, bf16_seq, reclaimed_cold, true),
            load_plan(bf16_needed, bf16_seq, raw, true)
        );
    }

    #[test]
    fn reclaimable_pool_is_a_per_gpu_monotonic_high_water() {
        // Distinct gpu_ids so this test can't race the process-global against other tests.
        let gpu = "test-reclaimable-gpu-a";
        let other = "test-reclaimable-gpu-b";
        // Nothing loaded yet ⇒ 0 (so `with_reclaimable` no-ops on a cold card).
        assert_eq!(reclaimable_pool_gb(gpu), 0.0);
        note_loaded_peak(gpu, 30.0);
        assert_eq!(reclaimable_pool_gb(gpu), 30.0);
        // A bigger load raises the high-water…
        note_loaded_peak(gpu, 82.0);
        assert_eq!(reclaimable_pool_gb(gpu), 82.0);
        // …a smaller later load does NOT lower it — the cudarc pool never returns pages to the driver.
        note_loaded_peak(gpu, 54.0);
        assert_eq!(reclaimable_pool_gb(gpu), 82.0);
        // Non-positive peaks are ignored.
        note_loaded_peak(gpu, 0.0);
        note_loaded_peak(gpu, -1.0);
        assert_eq!(reclaimable_pool_gb(gpu), 82.0);
        // Keyed per GPU — a different card is independent.
        assert_eq!(reclaimable_pool_gb(other), 0.0);
    }

    // -----------------------------------------------------------------------------------------
    // The Wan candle video weights-floor gate (sc-12344).
    // -----------------------------------------------------------------------------------------

    /// An RTX 5090 — the biggest consumer NVIDIA card, and the hardware the story names. 32 GB total,
    /// all free (a cold card with nothing loaded). `apply_vram_cap(None, ..)` synthesizes the budget, so
    /// the whole decision is exercisable here with no CUDA driver and no GPU.
    fn rtx_5090() -> Option<VramBudget> {
        apply_vram_cap(None, Some(32.0))
    }

    /// Wan2.2 T2V-A14B candle tier bytes — the SHIPPED hosted sizes, straight from this platform's
    /// `downloads[]` `estimatedSizeBytes` in `builtin.models.jsonc`: q4/q8 from the packed
    /// `SceneWorks/wan2.2-t2v-a14b-candle`, bf16 from the dense `Wan-AI/Wan2.2-T2V-A14B-Diffusers` the
    /// manifest routes the bf16 variant to. Real numbers, so these tests prove the REAL jobs are
    /// admitted/refused rather than that arithmetic is arithmetic.
    const WAN_A14B_CANDLE_Q4_BYTES: u64 = 29_788_704_888; // 27.74 GiB
    const WAN_A14B_CANDLE_Q8_BYTES: u64 = 44_071_949_832; // 41.05 GiB
    const WAN_A14B_CANDLE_BF16_BYTES: u64 = 72_000_000_000; // 67.06 GiB

    /// THE story (sc-12344): Wan A14B at bf16 is a ~67 GiB dual-expert MoE and was admitted on EVERY
    /// consumer card with no pre-flight check. It is now refused before the load + denoise.
    ///
    /// Kills the mutations a compile alone would not:
    ///   * dropping `transformer_2` from the A14B component list ⇒ ~half the bytes ⇒ ADMITS.
    ///   * sizing off the manifest's `footprint.peakMemoryBytes` (24.5 GiB — an MLX measurement that
    ///     assumes only ONE expert is resident) ⇒ ADMITS. The candle engine co-locates both.
    ///   * reusing the Mochi message ⇒ the "Shorten the clip" assert fails.
    #[test]
    fn wan_a14b_bf16_is_refused_on_an_rtx_5090() {
        let message = video_weights_fit_error(
            "wan2_2_t2v_14b",
            WAN_A14B_CANDLE_BF16_BYTES,
            "0",
            rtx_5090(),
        )
        .expect(
            "wan A14B bf16 needs ~67 GiB of weights + 2 GiB headroom = ~69 GB and NO consumer NVIDIA \
             card has that — admitting it burns the load + denoise before a raw CUDA OOM",
        )
        .to_string();
        assert!(
            message.contains("wan2_2_t2v_14b"),
            "names the model: {message}"
        );
        assert!(
            message.contains("69") && message.contains("32"),
            "states what it needs and what the card has: {message}"
        );
        assert!(
            message.contains("smaller quant tier"),
            "names the lever that actually moves weight bytes: {message}"
        );
        // The levers must be WEIGHTS levers. Resolution cannot change weight bytes, and the clip length
        // is Mochi's decode lever — offering either here sends the user to a knob that cannot move the
        // number they were just shown.
        assert!(
            !message.contains("resolution"),
            "resolution has zero effect on a weights floor: {message}"
        );
        assert!(
            !message.contains("Shorten the clip"),
            "Mochi's decode-lever prose leaking into the weights floor: {message}"
        );
        assert!(
            message.contains("VRAM") && !message.contains("unified memory"),
            "must be CUDA-worded, not the MLX lane's Mac prose: {message}"
        );
    }

    /// The other half of the contract: on the SAME 32 GB card the q4 tier is ADMITTED, and bf16 fits the
    /// 96 GB dev box. Without this the reject test above would pass against a gate that blanket-refuses
    /// Wan on every card — so this pair is what proves the gate discriminates by BUDGET and by TIER
    /// rather than wall-rejecting the lane.
    #[test]
    fn wan_a14b_admits_the_tier_that_fits_the_card() {
        // q4 = 27.74 + 2 = 29.74 GB ≤ 32 — the tier a 5090 actually runs.
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_CANDLE_Q4_BYTES, "0", rtx_5090())
                .is_none(),
            "q4 fits a 32 GB card — refusing it wall-rejects hardware that works today"
        );
        // q8 = 43.05 does NOT fit the same card…
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_CANDLE_Q8_BYTES, "0", rtx_5090())
                .is_some(),
            "q8 does not fit 32 GB"
        );
        // …but does fit a 48 GB card, where bf16 (69.06) still does not — two tiers, one card, two
        // verdicts: the gate reads the TIER's bytes, not the model id.
        let card_48 = apply_vram_cap(None, Some(48.0));
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_CANDLE_Q8_BYTES, "0", card_48)
                .is_none(),
            "q8 fits 48 GB"
        );
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_CANDLE_BF16_BYTES, "0", card_48)
                .is_some(),
            "bf16 does not fit the SAME 48 GB card"
        );
        // The 96 GB RTX PRO 6000 runs bf16 — the gate must not refuse the box this tier exists for.
        assert!(
            video_weights_fit_error(
                "wan2_2_t2v_14b",
                WAN_A14B_CANDLE_BF16_BYTES,
                "0",
                apply_vram_cap(None, Some(96.0))
            )
            .is_none(),
            "bf16 fits a 96 GB card"
        );
    }

    /// The CUDA reserve is real: a tier whose raw weights fit the card but whose weights + [`HEADROOM_GB`]
    /// do not must be refused. Pinned on a NON-default budget chosen so the two answers differ — q4's
    /// 27.74 GiB fits a 29 GB card outright, and only the reserve rejects it. Deleting the `+ HEADROOM_GB`
    /// from `video_weights_needed_gb` flips this to admit; nothing else in the suite would notice.
    #[test]
    fn video_weights_needed_gb_reserves_the_cuda_headroom() {
        let card_29 = apply_vram_cap(None, Some(29.0));
        assert!(
            WAN_A14B_CANDLE_Q4_BYTES as f64 / BYTES_PER_GIB < 29.0,
            "fixture guard: the raw weights must FIT the card, else this proves nothing"
        );
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_CANDLE_Q4_BYTES, "0", card_29)
                .is_some(),
            "weights alone fit 29 GB but weights + 2 GB of allocator/context reserve do not"
        );
        // And the reserve is the CUDA allocator one, not MLX's foreign-demand reserve, so pin the
        // arithmetic rather than the constant's name.
        assert_eq!(
            video_weights_needed_gb(WAN_A14B_CANDLE_Q4_BYTES),
            Some(WAN_A14B_CANDLE_Q4_BYTES as f64 / BYTES_PER_GIB + 2.0)
        );
    }

    /// The gate NO-OPS without a budget signal (the story's explicit AC) and without a weight signal — a
    /// worker on a card `nvidia-smi` cannot read, or pointed at a dir it cannot scan, must keep rendering
    /// exactly as it did before this gate existed. A fit gate that blocks on missing evidence is a
    /// regression, not a safety net (sc-12179: never wall-reject a machine that worked).
    #[test]
    fn video_weights_fit_error_no_ops_without_a_budget_or_weight_signal() {
        // No budget ⇒ admit, even for a model no card could hold.
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_CANDLE_BF16_BYTES, "0", None)
                .is_none()
        );
        // No weight signal ⇒ admit, even on a tiny card.
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", 0, "0", apply_vram_cap(None, Some(1.0)))
                .is_none()
        );
        assert_eq!(video_weights_needed_gb(0), None);
    }

    /// The component lists ARE the gate's correctness: they must name exactly the dirs the loader reads.
    ///
    /// Pins the A14B's second expert in particular — `transformer_2` is co-resident (wan14b.rs:337-343),
    /// and dropping it from the list halves the prediction and silently un-gates the bf16 tier this story
    /// exists to catch.
    #[test]
    fn wan_weight_bytes_sums_exactly_the_components_the_loader_reads() {
        let root_guard = tempfile::Builder::new()
            .prefix("sc12344_wan_components_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
        for (relative, len) in [
            ("transformer/model.safetensors", 1_000_u64),
            ("transformer_2/model.safetensors", 2_000),
            ("text_encoder/model.safetensors", 400),
            ("vae/model.safetensors", 30),
            // NOT read by the loader — a decoy that must not be counted. A blind recursive sum of the
            // dir would swallow it; naming the components is what keeps the floor from over-counting.
            ("upsampler/model.safetensors", 9_000_000),
        ] {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(&path).unwrap().set_len(len).unwrap();
        }

        // A14B counts BOTH experts + TE + VAE, and nothing else.
        assert_eq!(
            wan_weight_bytes("wan2_2_t2v_14b", root),
            1_000 + 2_000 + 400 + 30
        );
        assert_eq!(
            wan_weight_bytes("wan2_2_i2v_14b", root),
            1_000 + 2_000 + 400 + 30
        );
        // The 5B is single-expert: `transformer_2` must NOT be counted (it would not exist on a real 5B
        // snapshot; counting it here would mean the list, not the disk, decided).
        assert_eq!(wan_weight_bytes("wan2_2_ti2v_5b", root), 1_000 + 400 + 30);
    }

    #[test]
    fn vace_fun_floor_uses_shared_or_one_expert_and_fails_closed_when_incomplete() {
        let root_guard = tempfile::Builder::new()
            .prefix("sc18478_vace_fun_components_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
        for (relative, len) in [
            ("transformer/model.safetensors", 1_000_u64),
            ("transformer_2/model.safetensors", 2_000),
            ("text_encoder/model.safetensors", 400),
            ("vae/model.safetensors", 30),
        ] {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(path).unwrap().set_len(len).unwrap();
        }
        assert_eq!(
            wan_vace_fun_sequential_weight_bytes(root, 100).expect("complete layout"),
            2_100,
            "sequential VACE-Fun holds shared components or one expert plus its adapter stack, never both experts"
        );
        std::fs::remove_file(root.join("transformer_2/model.safetensors")).unwrap();
        assert!(
            wan_vace_fun_sequential_weight_bytes(root, 100).is_err(),
            "an incomplete dual-expert layout must fail before load"
        );
    }

    /// The EXEMPTIONS and the tier-ROOT fallback both read `0` ⇒ no signal ⇒ admit.
    ///
    /// This is the sc-12179 guard. `ltx`/`svd` are exempt because their on-disk bytes are not their
    /// loaded set (LTX dense is a ~146 GiB repo that loads ONE root file; SVD ships both dtype variants
    /// per component) — a floor built from a dir sum would refuse cards that render fine. And when NO wan
    /// tier subdir resolved, `model_dir` is a ROOT holding `q4/`+`q8/`: there is no top-level
    /// `transformer/`, so this reads 0 and admits rather than summing two tiers at once.
    #[test]
    fn wan_weight_bytes_is_zero_for_exempt_engines_and_a_tier_root() {
        let root_guard = tempfile::Builder::new()
            .prefix("sc12344_wan_exempt_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
        // A tier ROOT: the components live one level DOWN, under each tier.
        for tier in ["q4", "q8"] {
            for component in ["transformer", "transformer_2", "text_encoder", "vae"] {
                let path = root.join(tier).join(component).join("model.safetensors");
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::File::create(&path)
                    .unwrap()
                    .set_len(5_000_000_000)
                    .unwrap();
            }
        }

        // No top-level component dirs ⇒ no signal ⇒ admit. A blind recursive sum would read ~37 GiB
        // here — both tiers at once — and wall-reject a card that runs either one.
        assert_eq!(wan_weight_bytes("wan2_2_t2v_14b", root), 0);
        assert!(
            video_weights_fit_error(
                "wan2_2_t2v_14b",
                wan_weight_bytes("wan2_2_t2v_14b", root),
                "0",
                rtx_5090()
            )
            .is_none(),
            "an unrecognized layout must ADMIT, never reject on a number nothing verified"
        );

        // The exempt engines read 0 even pointed at a fully-populated component tree.
        let populated = root.join("q4");
        assert_eq!(wan_weight_bytes("ltx_2_3_distilled", &populated), 0);
        assert_eq!(wan_weight_bytes("svd_xt", &populated), 0);
        // Mochi has its own frame-dependent gate; it must not also ride this one.
        assert_eq!(wan_weight_bytes("mochi_1", &populated), 0);
        // …and the wan engines DO read that same tree, so the zeros above are the exemption, not a
        // broken fixture.
        assert!(wan_weight_bytes("wan2_2_t2v_14b", &populated) > 0);
    }

    // -----------------------------------------------------------------------------------------
    // The MEASURED per-tier peak supersedes the floor (sc-12402).
    // -----------------------------------------------------------------------------------------

    #[test]
    fn scail2_requires_a_measured_bf16_row_and_splits_low_from_high_cards() {
        let entry = obj(json!({
            "candle": {
                "minMemoryGb": 105,
                "vramGbByTier": { "bf16": 102.115 },
                "vramMeasuredPixels": 399360,
                "measured": true
            }
        }));
        let low = apply_vram_cap(None, Some(104.0));
        let message = scail2_video_fit_error(&entry, "0", low)
            .expect("102.115 GB measured + 2 GB reserve cannot fit 104 GB")
            .to_string();
        assert!(message.contains("SCAIL-2 shared bf16"), "{message}");
        assert!(
            message.contains("105") && message.contains("104"),
            "{message}"
        );
        assert!(
            message.contains("no lower-memory tier or sequential offload"),
            "{message}"
        );
        assert!(message.contains("MLX q4/q8"), "{message}");

        let high = apply_vram_cap(None, Some(105.0));
        assert!(
            scail2_video_fit_error(&entry, "0", high).is_none(),
            "a card above measured peak + reserve must be admitted"
        );
        let adapter_message = scail2_video_fit_error_with_adapter_bytes(
            &entry,
            BYTES_PER_GIB as u64,
            "0",
            apply_vram_cap(None, Some(105.0)),
        )
        .expect("a 1 GiB adapter source must move the same card over the cold-load boundary")
        .to_string();
        assert!(adapter_message.contains("106"), "{adapter_message}");
        let no_probe = scail2_video_fit_error(&entry, "0", None)
            .expect("an unknown live budget cannot safely admit the dense F32 stack")
            .to_string();
        assert!(
            no_probe.contains("could not read free GPU VRAM"),
            "{no_probe}"
        );
        assert!(no_probe.contains("nvidia-smi"), "{no_probe}");
        assert!(no_probe.contains("Refusing to load"), "{no_probe}");
    }

    #[test]
    fn scail2_admission_fails_closed_on_missing_or_unmeasured_catalog_truth() {
        for entry in [
            obj(json!({})),
            obj(json!({ "candle": { "measured": true } })),
            obj(json!({
                "candle": {
                    "minMemoryGb": 105,
                    "measured": false,
                    "vramGbByTier": { "bf16": 102.115 },
                    "vramMeasuredPixels": 399360
                }
            })),
            obj(json!({
                "candle": {
                    "minMemoryGb": 105,
                    "measured": true,
                    "vramGbByTier": { "bf16": 0 },
                    "vramMeasuredPixels": 399360
                }
            })),
            obj(json!({
                "candle": {
                    "measured": true,
                    "vramGbByTier": { "bf16": 102.115 },
                    "vramMeasuredPixels": 399360
                }
            })),
            obj(json!({
                "candle": {
                    "minMemoryGb": 105,
                    "measured": true,
                    "vramGbByTier": { "bf16": 102.115 }
                }
            })),
            obj(json!({
                "candle": {
                    "minMemoryGb": 105,
                    "measured": true,
                    "vramGbByTier": { "bf16": 102.115 },
                    "vramMeasuredPixels": 399361
                }
            })),
        ] {
            let message = scail2_video_fit_error(&entry, "0", None)
                .expect("catalog omission must refuse even when no GPU budget is observable")
                .to_string();
            assert!(
                message.contains("no complete measured bf16 CUDA row"),
                "{message}"
            );
            assert!(message.contains("Refusing to load"), "{message}");
        }

        let under_floor = obj(json!({
            "candle": {
                "minMemoryGb": 104,
                "measured": true,
                "vramGbByTier": { "bf16": 102.115 },
                "vramMeasuredPixels": 399360
            }
        }));
        let message = scail2_video_fit_error(&under_floor, "0", None)
            .expect("a floor below measured peak + reserve must refuse before probing the GPU")
            .to_string();
        assert!(message.contains("minMemoryGb=104"), "{message}");
        assert!(message.contains("~105 GB"), "{message}");
        assert!(message.contains("Refusing to load"), "{message}");
    }

    /// The 5B's FORMER RESIDENT candle block (q4 46.1 / q8 48.7 / bf16 54.0), as sc-12402/sc-12631 shipped
    /// it. sc-13175 RE-DROPPED the live `wan_2_2` onto sequential offload, so the shipped block is now the
    /// ~10-12 GiB sequential peak (gated ≤24 GB) — see `gpu_and_manifest::wan_candle_blocks_drive_the_video_fit_gate_and_reject`
    /// and `test_builtin_manifest_audit::test_wan_2_2_candle_vram_tiers_match_measured_peaks` for the live
    /// numbers. This synthetic fixture is RETAINED because the gate paths below need a measured peak that
    /// EXCEEDS both the on-disk weights floor and a consumer card — a shape no live candle video model has
    /// once its TE + VAE offload — so it stays a hard-coded resident-scale example, NOT a live-manifest read.
    fn wan_5b_entry() -> JsonObject {
        obj(json!({
            "candle": {
                "minMemoryGb": 48,
                "vramGbByTier": { "q4": 46.1, "q8": 48.7, "bf16": 54.0 },
                "measured": true
            }
        }))
    }

    /// The gate path sc-12402 was built for: a tier the weights FLOOR admits, but whose real peak OOMs, is
    /// refused. Exercised with the 5B's FORMER RESIDENT shape (`wan_5b_entry`) — post-sc-13175 the live 5B
    /// offloads and this under-count case no longer describes it, but the gate arithmetic it pins is the same.
    ///
    /// That resident q4's floor is 16.13 GiB of on-disk weights + 2 = ~18 GB, so an RTX 5090 (32 GB) is
    /// ADMITTED by the floor — and the job would die in the denoise, because the resident peak is 46.1 GB
    /// (attention + co-resident TE/VAE, not the weights alone). The floor under-counts by ~2.4x, which is
    /// not a tuning error but a category error — it counts weights and the peak is not weights.
    ///
    /// Kills the mutations a compile alone would not:
    ///   * dropping the `predicted_peak_gb` branch (⇒ back to the floor) ⇒ the 5090 ADMITS;
    ///   * keying the manifest lookup off the request instead of the RESOLVED tier ⇒ wrong row;
    ///   * `max`-ing the measured peak with the floor ⇒ still passes here, but fails the dense test below.
    #[test]
    fn measured_peak_refuses_a_5b_job_the_weights_floor_would_admit() {
        let entry = wan_5b_entry();
        // The on-disk floor for the shipped q4 tier: 16.13 GiB.
        const WAN_5B_Q4_DISK_BYTES: u64 = 17_315_750_512;

        // Floor alone: ~18 GB needed vs 32 GB free ⇒ admits (the shipped behavior, and the bug).
        assert!(
            video_weights_fit_error("wan_2_2", WAN_5B_Q4_DISK_BYTES, "0", rtx_5090()).is_none(),
            "precondition: the weights floor admits this job on a 5090 — that IS the sc-12402 bug"
        );

        // Measured peak: 46.1 + 2 headroom = 48.1 GB vs 32 GB ⇒ REFUSED before the load + denoise.
        let message = wan_video_fit_error(
            "wan_2_2",
            &entry,
            "q4",
            WAN_5B_Q4_DISK_BYTES,
            "0",
            rtx_5090(),
        )
        .expect("the measured 46.1 GB peak cannot fit a 32 GB card — refuse before the OOM")
        .to_string();
        assert!(message.contains("wan_2_2"), "names the model: {message}");
        assert!(message.contains("q4"), "names the sized tier: {message}");
        assert!(
            message.contains("48") && message.contains("32"),
            "states what it needs and what the card has: {message}"
        );
        // The measured message is about the RENDER, not the weights — the floor's wording would be a
        // lie here (weights are 16 GiB of a 46 GB peak).
        assert!(
            !message.contains("just to hold its weights"),
            "must not reuse the weights-floor wording for a measured peak: {message}"
        );
        // Resolution is NOT offered as a lever: the measured peak is a per-tier constant at the model's
        // default geometry, so it cannot move the number the user was just shown.
        assert!(
            !message.contains("resolution"),
            "the resolution-blind gate must not send the user to a knob it cannot honor: {message}"
        );
    }

    /// The other half of the story's acceptance: a tier that FITS is still ADMITTED. A gate that
    /// refuses everything would pass the test above and be worthless.
    #[test]
    fn measured_peak_admits_a_tier_that_fits_the_card() {
        let entry = wan_5b_entry();
        const WAN_5B_Q4_DISK_BYTES: u64 = 17_315_750_512;
        // The dev box: 96 GB. q4's peak 46.1 + 2 = 48.1 ≤ 95.6 ⇒ ADMIT (the former resident shape rendered
        // there; `wan_5b_entry`).
        let card96 = apply_vram_cap(None, Some(95.6));
        assert!(
            wan_video_fit_error("wan_2_2", &entry, "q4", WAN_5B_Q4_DISK_BYTES, "0", card96)
                .is_none(),
            "the q4 job renders on this card — the gate must not wall-reject it"
        );
        // Same model, different tier ⇒ opposite verdict on a card sized BETWEEN the tiers: a card with
        // 52 GB free admits q4 (48.1) and q8 (50.7) but refuses bf16 (54.0 + 2 = 56.0 > 52). The gate
        // reads the TIER, not the model — this resident ladder spans only ~8 GB (everything but the DiT is
        // tier-independent), so the tiers land close but the gate still splits them.
        let card52 = apply_vram_cap(None, Some(52.0));
        assert!(
            wan_video_fit_error("wan_2_2", &entry, "q4", WAN_5B_Q4_DISK_BYTES, "0", card52)
                .is_none(),
            "q4's measured 48.1 GB need fits a 52 GB card"
        );
        assert!(
            wan_video_fit_error("wan_2_2", &entry, "bf16", WAN_5B_Q4_DISK_BYTES, "0", card52)
                .is_some(),
            "bf16's measured 54.0 GB peak + 2 headroom overflows a 52 GB card — refuse"
        );
    }

    #[test]
    fn wan_additive_adapter_bytes_flip_the_measured_and_floor_boundaries() {
        let entry = wan_5b_entry();
        let one_gib = BYTES_PER_GIB as u64;
        let boundary = apply_vram_cap(None, Some(48.6));
        assert!(wan_video_fit_error_with_adapter_bytes(
            "wan_2_2", &entry, "q4", 0, 0, "0", boundary,
        )
        .is_none());
        assert!(
            wan_video_fit_error_with_adapter_bytes(
                "wan_2_2", &entry, "q4", 0, one_gib, "0", boundary,
            )
            .is_some(),
            "a packed adapter must be added to the measured render peak"
        );

        let no_measurement = obj(json!({}));
        let floor_boundary = apply_vram_cap(None, Some(3.5));
        assert!(wan_video_fit_error_with_adapter_bytes(
            "wan",
            &no_measurement,
            "q4",
            one_gib,
            0,
            "0",
            floor_boundary,
        )
        .is_none());
        assert!(
            wan_video_fit_error_with_adapter_bytes(
                "wan",
                &no_measurement,
                "q4",
                one_gib,
                one_gib,
                "0",
                floor_boundary,
            )
            .is_some(),
            "the unmeasured weights floor must also include additive adapters"
        );
    }

    /// sc-12402 regression pin: the measured peak REPLACES the floor, it does not compose with it.
    ///
    /// The dense `bf16` tier is the case that makes `max(measured, floor)` wrong. `Wan-AI/*-Diffusers`
    /// ships its experts in **fp32** (~117.5 GiB summed for A14B T2V) and `candle-gen-wan` loads them as
    /// bf16 (`wan14b.rs:49-52`), so the on-disk floor OVER-counts by ~44 GiB and would wall-reject a
    /// 96 GB card that renders the job at ~75 GB — the sc-12179 class `wan_weight_components` claims
    /// cannot arise on this lane. A measurement is not a better proxy for the peak; it IS the peak.
    #[test]
    fn a_measurement_supersedes_an_overcounting_dense_floor() {
        // A14B T2V dense: 117.51 GiB of fp32 on disk, ~75 GB resident once cast to bf16.
        const WAN_A14B_DENSE_DISK_BYTES: u64 = 126_170_000_000;
        let entry = obj(json!({
            "candle": { "vramGbByTier": { "bf16": 75.3 }, "measured": false }
        }));
        let card96 = apply_vram_cap(None, Some(95.6));

        // The floor alone WALL-REJECTS this card (117.5 + 2 = ~120 > 95.6) — the bug.
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_DENSE_DISK_BYTES, "0", card96)
                .is_some(),
            "precondition: the on-disk floor over-counts the fp32 dense tier and wall-rejects a 96 GB card"
        );
        // The measured/derived peak admits it: 75.3 + 2 = 77.3 ≤ 95.6.
        assert!(
            wan_video_fit_error(
                "wan2_2_t2v_14b",
                &entry,
                "bf16",
                WAN_A14B_DENSE_DISK_BYTES,
                "0",
                card96
            )
            .is_none(),
            "a `max(measured, floor)` would keep the floor's 120 GB over-count and wall-reject here"
        );
    }

    /// An unmeasured model keeps the sc-12344 floor EXACTLY — the fallback is not a silent un-gating.
    #[test]
    fn an_unmeasured_model_falls_back_to_the_weights_floor() {
        let no_block = obj(json!({}));
        const WAN_A14B_BF16_BYTES: u64 = 72_000_000_000;

        // No `candle` block ⇒ the floor decides, byte-identical to `video_weights_fit_error`.
        let via_wan = wan_video_fit_error(
            "wan2_2_t2v_14b",
            &no_block,
            "bf16",
            WAN_A14B_BF16_BYTES,
            "0",
            rtx_5090(),
        );
        let via_floor =
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_BF16_BYTES, "0", rtx_5090());
        assert_eq!(
            via_wan.map(|e| e.to_string()),
            via_floor.map(|e| e.to_string()),
            "an unmeasured model must behave exactly as it did before sc-12402"
        );

        // An exempt engine (0 weights) + no block ⇒ admit, never block without evidence.
        assert!(
            wan_video_fit_error("ltx_2_3_distilled", &no_block, "bf16", 0, "0", rtx_5090())
                .is_none()
        );
        // No live budget ⇒ admit, even with a measured row that would overflow.
        assert!(
            wan_video_fit_error("wan_2_2", &wan_5b_entry(), "bf16", 0, "0", None).is_none(),
            "no budget signal ⇒ never block"
        );
    }

    /// Live real-hardware validation (sc-10766): exercises the REAL `nvidia-smi` VRAM reading on GPU 0
    /// plus the full cap → predict → decide chain — the one piece the pure tests above can't cover.
    /// Ignored by default (needs a CUDA GPU); run on the box with
    /// `cargo test -p sceneworks-worker --lib --features backend-candle -- --ignored live_cuda_budget`.
    #[tokio::test]
    #[ignore]
    async fn live_cuda_budget_drives_a_real_fit_decision() {
        let real = crate::gpu::nvidia_vram_budget_gb("0")
            .await
            .expect("GPU 0 should report a live VRAM budget on a CUDA box");
        assert!(
            real.total_gb > 0.0
                && real.free_gb >= 0.0
                && real.free_gb <= real.total_gb + f64::EPSILON,
            "sane live budget: {real:?}"
        );
        eprintln!("live CUDA budget GPU0: {real:?}");
        // Emulate a 10 GB card: a ~47 GB q4 model must be rejected before load…
        let capped = apply_vram_cap(Some(real), Some(10.0)).expect("capped budget");
        let big = obj(json!({ "candle": { "vramGbByTier": { "q4": 47.2 } } }));
        assert!(matches!(
            fit_decision(predicted_peak_gb(&big, "q4"), Some(capped)),
            FitDecision::TooBig { .. }
        ));
        // …while a ~4 GB model fits that same emulated card.
        let small = obj(json!({ "candle": { "vramGbByTier": { "q4": 4.0 } } }));
        assert_eq!(
            fit_decision(predicted_peak_gb(&small, "q4"), Some(capped)),
            FitDecision::Fits
        );
    }

    /// GPU repro of the sc-11023 warm-swap false reject: occupy REAL VRAM on GPU 0 to mimic a
    /// resident model (nvidia-smi `free` drops the way a cached model makes it), then show that the
    /// SAME live budget REJECTS the incoming bf16 tier on raw `free` (the bug) but FITS once the
    /// reclaimable high-water is folded in (the fix). Run on an idle GPU 0:
    /// `cargo test -p sceneworks-worker --lib --features backend-candle -- --ignored --nocapture \
    ///  reclaimable_high_water_flips_a_real_warm_reject`
    #[tokio::test]
    #[ignore]
    async fn reclaimable_high_water_flips_a_real_warm_reject() {
        use runtime_cuda::media::candle_core::{DType, Device, Tensor};

        let gpu = "0";
        // Qwen-Image bf16 published tier peaks (builtin.models.jsonc `candle` block, sc-10969).
        let manifest = obj(json!({
            "candle": { "vramGbByTier": { "bf16": 82.5 }, "sequentialPeakGb": { "bf16": 71.7 } }
        }));
        let needed = predicted_peak_gb(&manifest, "bf16").expect("bf16 resident peak");
        let seq_needed = predicted_sequential_peak_gb(&manifest, "bf16").expect("bf16 seq peak");

        let cold = crate::gpu::nvidia_vram_budget_gb(gpu)
            .await
            .expect("GPU 0 live budget — run on a CUDA box");
        eprintln!("cold budget: {cold:?} (needed resident={needed}, sequential={seq_needed})");
        assert!(
            cold.free_gb > seq_needed,
            "GPU 0 must start with enough free VRAM to host bf16 for the repro: {cold:?}"
        );

        // Mimic run 1 admitting a bf16 load: the live gate records the model's predicted peak as the
        // reclaimable high-water (note_loaded_peak) AND the model actually occupies VRAM. Occupy enough
        // (8 GB f32 chunks) to push `free` below the sequential threshold, so raw-free rejects even
        // sequentially — capped defensively so we never exhaust the card.
        note_loaded_peak(gpu, needed);
        let device = Device::new_cuda(0).expect("cuda:0");
        let chunk_elems = 8usize * 1024 * 1024 * 1024 / 4; // 8 GB of f32
        let mut hogs: Vec<Tensor> = Vec::new();
        loop {
            let now = crate::gpu::nvidia_vram_budget_gb(gpu)
                .await
                .expect("live budget");
            if now.free_gb < seq_needed - 4.0 || hogs.len() >= 8 {
                break;
            }
            hogs.push(Tensor::zeros(chunk_elems, DType::F32, &device).expect("VRAM hog chunk"));
        }
        let warm = crate::gpu::nvidia_vram_budget_gb(gpu)
            .await
            .expect("GPU 0 live budget (warm)");
        eprintln!("warm budget after {} GB hog: {warm:?}", hogs.len() * 8);
        assert!(
            warm.free_gb < seq_needed,
            "the hog must push free below the sequential need to arm the reject: {warm:?}"
        );

        // BEFORE the fix (raw free): sequential residency is selected, then the second-stage overflow
        // check rejects — the exact "even with sequential residency" 2nd-run failure.
        let raw_decision = resolve_offload(fit_decision(Some(needed), Some(warm)), true);
        assert!(
            matches!(raw_decision, FitDecision::Offload { .. }),
            "raw free must not fit resident: {raw_decision:?}"
        );
        assert!(
            sequential_overflow_gb(Some(seq_needed), Some(warm)).is_some(),
            "raw-free gate must reject even sequentially — this is the sc-11023 bug"
        );

        // AFTER the fix: fold the reclaimable high-water (the resident model we are about to evict)
        // back into the budget → the incoming bf16 fits resident, no false reject.
        let reclaimable = reclaimable_pool_gb(gpu);
        let fixed = with_reclaimable(warm, reclaimable);
        eprintln!("reclaimable={reclaimable} → augmented budget: {fixed:?}");
        assert_eq!(
            fit_decision(Some(needed), Some(fixed)),
            FitDecision::Fits,
            "with the resident model counted reclaimable, the warm re-gate must FIT"
        );
        // Keep the hog alive to the end so the warm reading isn't reclaimed early.
        drop(hogs);
    }
}
