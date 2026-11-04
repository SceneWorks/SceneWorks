//! mlx_fit_gate residency owner.
use super::*;

pub(super) fn gib_to_bytes(gib: f64) -> u64 {
    sceneworks_core::memory_anchor::gib_to_bytes(gib)
}

pub(super) fn decimal_gb_to_bytes(gb: f64) -> u64 {
    (gb * 1_000_000_000.0).round().clamp(0.0, u64::MAX as f64) as u64
}

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
/// (sc-10840 — sd3/sana/flux/flux2/chroma/ideogram/kolors/anima/boogu alongside the earlier
/// sdxl/z-image/qwen/lens/krea families) is covered the moment its descriptor advertises the bit, with
/// no lockstep edit here. Providers that stage unconditionally (for example Bernini) deliberately
/// leave the selectable-control bit `false`; an engine that cannot benefit from staging (for example
/// SenseNova's fused MoT, `footprint` te=0) does the same. Neither is offered a no-op `Sequential`
/// selection.
///
/// This is a pre-load, weights-free registry lookup (`(descriptor)()` allocates no tensors), the same
/// query shape the worker already uses for family/guidance/quant capability advertisement and the
/// analogous `ProviderRegistry::footprint` size seam (sc-10894). An id with no registered generator — or a
/// registered one that does not advertise the bit — yields `false` (the safe default: never select a
/// residency policy the provider won't honor). Sees exactly the providers the selected runtime bundle
/// carries: MLX providers are explicitly anchored on macOS, while the CUDA bundle exposes its explicit
/// Candle catalog. The same query is shared by the MLX fit gate (sc-10840) and Candle fit gate
/// (sc-12130), so adding a truthful provider capability needs no worker allowlist edit.
/// **Two descriptor bits, one question (sc-19721).** gen-core split the attestation in two, and this
/// gate's question is the disjunction:
///
/// * `supports_sequential_offload` — the SELECTABLE [`OffloadPolicy::Sequential`] is honoured.
/// * `unconditionally_engages_staged_residency` — the provider stages eligible components through a
///   load/use/drop lifecycle on EVERY generation, with or without a policy request.
///
/// Either one makes "peak is the dominant component, not the sum" true, which is the only thing this
/// gate reads the bit for. The pair is not redundant: MLX Bernini holds no component weights on the
/// generator at all, so it never had a selectable control to honour — it advertised
/// `supports_sequential_offload: true` purely to reach this gate, and inference corrected that to the
/// second bit. Reading only the first would have charged Bernini the SUM of the Qwen2.5-VL planner,
/// the UMT5-XXL encoder and the two MoE experts — a co-residency `generate_impl` never creates — and
/// refused a model that fits, silently, on a pin bump. The `false` default still means "no
/// attestation", so an unwired engine (sensenova's fused MoT) is still never offered `Sequential`.
///
/// Requesting `Sequential` from a provider that stages unconditionally but exposes no selectable
/// control is safe in the direction that matters: the policy is *advisory* (gen-core treats an
/// unwired request as `Resident`, never an error), and here the staged behaviour the prediction
/// assumes is what the provider physically does regardless.
pub(crate) fn engine_supports_sequential(engine_id: &str) -> bool {
    crate::inference_runtime::media()
        .generators()
        .find(|reg| (reg.descriptor)().id == engine_id)
        .is_some_and(|reg| {
            let capabilities = (reg.descriptor)().capabilities;
            capabilities.supports_sequential_offload
                || capabilities.unconditionally_engages_staged_residency
        })
}

/// Whether `engine_id` physically stages heavyweight phases, either because the provider exposes
/// the selectable [`OffloadPolicy::Sequential`] control or because it does so unconditionally.
///
/// This is static descriptor truth for publication and capability reporting. Load-time selection
/// must continue to use [`engine_supports_sequential`]: an unconditional-only provider does not
/// consume `OffloadPolicy::Sequential`, so treating this broader predicate as authorization to set
/// that control would turn an accurate declaration into a no-op request.
#[cfg(test)]
pub(crate) fn engine_engages_staged_residency(engine_id: &str) -> bool {
    crate::inference_runtime::media()
        .generators()
        .find(|reg| (reg.descriptor)().id == engine_id)
        .is_some_and(|reg| {
            (reg.descriptor)()
                .capabilities
                .staged_residency_availability()
                != StagedResidencyAvailability::Absent
        })
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
/// 18 GiB = the max COMMON-CASE transient at 1024² (14.04, illustrious q8 & lens q4 — the three
/// resident≈disk tiers; lens-turbo's larger 29.88 transient is a separate architecture outlier, below)
/// plus a ~4 GiB macOS/app reserve. This replaces the provisional 10.0, which UNDER-predicted 3 of the
/// 4 measured tiers (illustrious 15.0<18.8, lens 27.7<30.5, lens-turbo 38.4<75.6) — i.e. was a latent
/// SIGKILL risk on Macs sized between the predicted and the real peak. All three resident≈disk tiers
/// are now covered with margin without over-rejecting a model that fits (illustrious q8: 5.01+18=23.0
/// still fits a 24 GiB Mac, where its real 18.8 GiB peak + OS does too).
///
/// NOT covered by this constant (surfaced sc-10863, tracked follow-ups — see the story): (1) the
/// lens-turbo bf16 OUTLIER, whose 47.12 GiB headroom (peak 75.55 − disk 28.43) is NOT one effect but
/// TWO that must be modeled together. It DECOMPOSES as (a) 17.24 GiB IN-MEMORY WEIGHT EXPANSION
/// (resident 45.67 − disk 28.43) — its mxfp4-on-disk gpt-oss text encoder expands loading to bf16
/// (45.67 = 1.61× disk 28.43), so `sum_safetensors_bytes` under-counts the in-memory weights — PLUS
/// (b) a 29.88 GiB ACTIVATION TRANSIENT (peak 75.55 − resident 45.67), which is architecture-bound (the
/// large gpt-oss encoder's activations) and ~2× the ~14 GiB the other three tiers show at the same
/// 1024². HEADROOM=18 covers the common-case transient (~14) + ~4 GiB OS/app reserve, but UNDER-predicts
/// this class by ~29 GiB even AFTER a weight-byte correction — because the 29.88 transient ALONE exceeds
/// 18 (75.55 − (28.43 + 18) = 29.12; the old provisional 10 under-predicted it by ~37: 75.55 −
/// (28.43 + 10) = 37.12). So correcting only the weight bytes is INSUFFICIENT: both the in-memory weight
/// size AND the outsized transient must be modeled for this bf16/MXFP4 source. (A blanket bf16 expansion factor
/// also can't fix the weight half — a bf16 tier whose encoder is bf16-on-disk would then be
/// over-rejected ~1.6× — so that fix needs per-family in-memory weight sizing plus a per-architecture
/// transient term, backed by bf16 measurements across models.) Tracked in sc-11924. (2) Output
/// RESOLUTION > 1024² grows the VAE-decode transient past 14 GiB — all four points are 1024², so 18 is
/// a 1024²-worst-case; a higher-res campaign is a follow-up.
///
/// Declared in `sceneworks_core::memory_anchor` since sc-22738 so the memory adapter's LTX-2.3
/// admission floors its projection with the SAME allowance the video gate floors with
/// (`video_admission::floor_phase_peaks`); this is the worker's read of it, not a second
/// declaration.
pub(super) const HEADROOM_GB: f64 = sceneworks_core::memory_anchor::MLX_GENERIC_HEADROOM_GB;

/// Lens dense/bf16's measured 1024² activation transient. Its gpt-oss encoder is the only current
/// MLX family whose architecture-bound transient exceeds the generic calibration (sc-11924).
pub(super) const LENS_DENSE_HEADROOM_GB: f64 = 29.88;

/// The macOS/app share inside [`HEADROOM_GB`] — the part of that flat allowance that covers the OS and
/// other apps drawing from the same unified pool, as opposed to this request's activation transient.
///
/// This is the sc-10863 decomposition read back out and given a name: `HEADROOM_GB` 18 = the max
/// common-case measured 1024² transient (14.04, illustrious q8 & lens q4) + this ~4 GiB reserve. It
/// was previously only prose in that constant's doc comment, which is precisely how it came to be
/// multiplied by megapixels — the request estimator could not tell the two halves apart.
///
/// It applies to [`HEADROOM_GB`] ONLY. See [`HeadroomAllowance`] for why that distinction is load
/// bearing rather than pedantic.
pub(crate) const OS_APP_RESERVE_GB: f64 = 4.0;

impl HeadroomAllowance {
    /// The sc-10863 calibration: a 1024² transient plus a macOS/app reserve.
    pub(super) const GENERIC: Self = Self {
        total_gb: HEADROOM_GB,
        os_reserve_gb: OS_APP_RESERVE_GB,
    };
    /// sc-11924's lens-dense measurement: a bare activation transient, no reserve folded in.
    pub(super) const LENS_DENSE: Self = Self {
        total_gb: LENS_DENSE_HEADROOM_GB,
        os_reserve_gb: 0.0,
    };
}

/// The numeric tier a `LoadSpec` loads, exactly as [`MlxRequestPlan::for_spec_and_manifest`]
/// derives it (sc-18814 factored it out so the video gate cannot drift into a second derivation).
/// The manifest-driven `fixed_artifact_tier` override is deliberately NOT applied here: a video
/// route carries no calibration binding to resolve one from.
pub(crate) fn spec_numeric_tier(engine_id: &str, spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: active_component_floors(
            declared_component_floors(engine_id),
            spec.quantize,
        ),
    }
}

/// The activation-headroom allowance this load already charges — the same
/// [`HeadroomAllowance`] the load-time residency gate uses (sc-18814 exposes it so the video gate
/// reuses that number instead of inventing one).
pub(crate) fn spec_headroom_bytes(engine_id: &str, spec: &LoadSpec) -> u64 {
    gib_to_bytes(spec_component_bytes(engine_id, spec).2.total_gb)
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
pub(crate) fn resolve_budget(
    real_total_gb: Option<f64>,
    cap: Option<f64>,
) -> Option<MlxMemoryBudget> {
    cap.or(real_total_gb)
        .map(|total_gb| MlxMemoryBudget { total_gb })
}

/// Predicted whole-model peak (GiB) = summed component weight bytes + [`HEADROOM_GB`]. `None` when
/// `weight_bytes == 0` (nothing measured ⇒ no signal ⇒ never block).
#[cfg(test)]
pub(crate) fn predicted_peak_gb(weight_bytes: u64) -> Option<f64> {
    predicted_peak_gb_with_headroom(weight_bytes, HEADROOM_GB)
}

pub(super) fn predicted_peak_gb_with_headroom(weight_bytes: u64, headroom_gb: f64) -> Option<f64> {
    (weight_bytes > 0).then(|| weight_bytes as f64 / BYTES_PER_GIB + headroom_gb)
}

/// Decide whether the predicted peak fits the budget. Missing either input ⇒ `Unknown` (never
/// block), exactly like the flux2 guard and the candle gate.
pub(crate) fn fit_decision(needed_gb: Option<f64>, budget: Option<MlxMemoryBudget>) -> FitDecision {
    crate::fit_gate::fit_decision(needed_gb, budget.map(|budget| budget.total_gb))
}

/// Predicted SEQUENTIAL peak (GiB) = the largest single working set + [`HEADROOM_GB`] (sc-10839). The
/// `Sequential` schedule drops the text encoder(s) before the DiT loads and keeps the tiny VAE
/// co-resident with the DiT, so the peak is `max(text-encoders, everything-else) + headroom` rather
/// than the resident sum. `everything-else = total − text_encoders` (the DiT + VAE + any control/IP).
/// `None` when nothing was measured (`total == 0`). When the text encoders are unmeasured
/// (`te_bytes == 0`) this equals the resident peak — no claimed saving, so the second-stage overflow
/// check then rejects exactly as the resident gate would (the safe fallback).
#[cfg(test)]
pub(crate) fn predicted_sequential_peak_gb(total_bytes: u64, te_bytes: u64) -> Option<f64> {
    predicted_sequential_peak_gb_with_headroom(total_bytes, te_bytes, HEADROOM_GB)
}

pub(super) fn predicted_sequential_peak_gb_with_headroom(
    total_bytes: u64,
    te_bytes: u64,
    headroom_gb: f64,
) -> Option<f64> {
    if total_bytes == 0 {
        return None;
    }
    let rest_bytes = total_bytes.saturating_sub(te_bytes);
    let staged_max = te_bytes.max(rest_bytes);
    Some(staged_max as f64 / BYTES_PER_GIB + headroom_gb)
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
    budget: Option<MlxMemoryBudget>,
) -> Option<f64> {
    crate::fit_gate::sequential_overflow_gb(
        sequential_needed_gb,
        budget.map(|budget| budget.total_gb),
    )
}

/// Sum the on-disk bytes of every `.safetensors` weight file under `dir` (recursively), following
/// symlinks (the HF cache stores each shard as a symlink into `blobs/`). AppleDouble `._*` sidecars
/// are skipped — they masquerade as `.safetensors` and would double-count (and corrupt globs, per
/// the AppleDouble sidecar gotcha). Returns 0 if the directory is missing or holds no weights, which
/// the gate treats as "no signal".
pub(crate) fn sum_safetensors_bytes(dir: &Path) -> u64 {
    fn walk(
        dir: &Path,
        visited_dirs: &mut std::collections::HashSet<std::path::PathBuf>,
        total: &mut u64,
    ) {
        // `metadata` below intentionally follows file symlinks because Hugging Face
        // snapshots link shards into `blobs/`. Directory links are different: an
        // operator-provided model root may contain a link/junction cycle. Canonical
        // directory identity makes that traversal finite and also avoids double-counting
        // an aliased subtree.
        let Ok(canonical_dir) = std::fs::canonicalize(dir) else {
            return;
        };
        if !visited_dirs.insert(canonical_dir.clone()) {
            return;
        }
        let Ok(entries) = std::fs::read_dir(canonical_dir) else {
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
                walk(&path, visited_dirs, total);
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
    walk(dir, &mut std::collections::HashSet::new(), &mut total);
    total
}

/// On-disk `.safetensors` bytes of a [`WeightsSource`]: the recursive sum for a `Dir`, the file length
/// for a single-file `File`. Used to fold a separate control/overlay checkpoint ([`LoadSpec::control`])
/// into the fit total — its weights are not under the base `spec.weights` tree.
pub(super) fn weights_source_bytes(src: &WeightsSource) -> u64 {
    match src {
        WeightsSource::Dir(dir) => sum_safetensors_bytes(dir),
        WeightsSource::File(file) => std::fs::metadata(file).map_or(0, |meta| meta.len()),
    }
}

pub(super) fn text_encoder_discovery_roots(
    source: &WeightsSource,
) -> gen_core::Result<Vec<PathBuf>> {
    let entry = std::path::absolute(crate::weights_source::WeightsSourceExt::path(source))?;
    let lexical_root = match source {
        WeightsSource::Dir(_) => entry.clone(),
        WeightsSource::File(_) => entry.parent().map(Path::to_path_buf).ok_or_else(|| {
            gen_core::Error::Unsupported(
                "selected text-encoder file has no lexical parent directory".into(),
            )
        })?,
    };
    let canonical = std::fs::canonicalize(&entry)?;
    let canonical_root = match source {
        WeightsSource::Dir(_) => canonical,
        WeightsSource::File(_) => canonical.parent().map(Path::to_path_buf).ok_or_else(|| {
            gen_core::Error::Unsupported(
                "selected text-encoder file has no canonical parent directory".into(),
            )
        })?,
    };
    let mut roots = vec![lexical_root, canonical_root];
    roots.sort();
    roots.dedup();
    Ok(roots)
}

/// Resolve path-free planning facts exactly as gen-core's loader does. A prepared spec reuses its
/// retained acquisition receipt; compatibility mode uses bounded metadata discovery. Neither path
/// acquires or discards an executable artifact seal.
pub(super) fn resolved_text_encoder_source(
    spec: &LoadSpec,
    source: &WeightsSource,
) -> gen_core::Result<ResolvedTextEncoderSource> {
    let facts = match spec.prepared_text_encoder_planning_facts()? {
        Some(facts) => facts,
        None => gen_core::text_encoder_planning_facts_for_discovery(
            source,
            &text_encoder_discovery_roots(source)?,
        )?,
    };
    let path = match (source, facts.source_layout()) {
        (WeightsSource::File(path), gen_core::TextEncoderSourceLayout::File)
        | (WeightsSource::Dir(path), gen_core::TextEncoderSourceLayout::DirectDirectory) => {
            path.clone()
        }
        (WeightsSource::Dir(path), gen_core::TextEncoderSourceLayout::CompleteSnapshot) => {
            path.join("text_encoder")
        }
        _ => {
            return Err(gen_core::Error::Unsupported(
                "selected text-encoder planning layout does not match its LoadSpec source".into(),
            ));
        }
    };
    Ok(ResolvedTextEncoderSource {
        path,
        direct_shard_bytes: facts.direct_shard_bytes(),
    })
}

pub(super) fn adapter_path_bytes(path: &Path) -> u64 {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => sum_safetensors_bytes(path),
        Ok(metadata)
            if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors") =>
        {
            metadata.len()
        }
        _ => 0,
    }
}

pub(super) fn adapter_source_bytes_for_gate(engine_id: &str, spec: &LoadSpec) -> u64 {
    adapter_source_bytes_for_gate_where(engine_id, spec, |_| true)
}

pub(super) fn external_adapter_source_bytes_for_gate(engine_id: &str, spec: &LoadSpec) -> u64 {
    adapter_source_bytes_for_gate_where(engine_id, spec, |path| match &spec.weights {
        WeightsSource::Dir(root) => !path.starts_with(root),
        WeightsSource::File(_) => true,
    })
}

pub(super) fn adapter_source_bytes_for_gate_where(
    engine_id: &str,
    spec: &LoadSpec,
    include: impl Fn(&Path) -> bool,
) -> u64 {
    let source_bytes = spec.adapters.iter().fold(0_u64, |total, adapter| {
        if !include(&adapter.path) {
            return total;
        }
        let bytes = adapter_path_bytes(&adapter.path);
        total.saturating_add(bytes)
    });
    adapter_resident_source_bytes(engine_id, spec, source_bytes)
}

pub(super) fn adapter_resident_source_bytes(
    engine_id: &str,
    spec: &LoadSpec,
    source_bytes: u64,
) -> u64 {
    if matches!(engine_id, "wan_vace" | "wan2_2_vace_fun_14b") {
        return 0;
    }
    if !matches!(
        engine_id,
        "wan2_2_ti2v_5b" | "wan2_2_t2v_14b" | "wan2_2_i2v_14b"
    ) {
        return source_bytes;
    }

    // Wan dense loads fold factors into the mutable base before optional load-time quantization, so
    // the adapter files do not remain independently resident. A pre-packed snapshot declares its
    // quantization in the root config and installs adapters as forward-time residuals instead.
    let prepacked = spec.quantize.is_none()
        && matches!(&spec.weights, WeightsSource::Dir(root) if packed_quant_bits(root, "").is_some());
    if prepacked {
        source_bytes
    } else {
        0
    }
}

pub(super) fn prepared_file_sizes(spec: &LoadSpec) -> gen_core::Result<BTreeMap<PathBuf, u64>> {
    spec.validate_prepared_file_pins()?;
    if !spec.prepared_file_pins().is_prepared() {
        return Err(gen_core::Error::Unsupported(
            "prepared accounting requires a finalized prepared LoadSpec".into(),
        ));
    }
    Ok(spec
        .prepared_file_pins()
        .iter()
        .map(|(path, pin)| (path.clone(), pin.target_fingerprint().size))
        .collect())
}

pub(super) fn prepared_source_bytes(
    source: &WeightsSource,
    file_sizes: &BTreeMap<PathBuf, u64>,
) -> u64 {
    match source {
        WeightsSource::Dir(dir) => sum_safetensors_bytes(dir),
        WeightsSource::File(file) => std::path::absolute(file)
            .ok()
            .and_then(|file| file_sizes.get(&file).copied())
            .unwrap_or(0),
    }
}

pub(super) fn prepared_adapter_source_bytes_for_gate_where(
    engine_id: &str,
    spec: &LoadSpec,
    file_sizes: &BTreeMap<PathBuf, u64>,
    include: impl Fn(&Path) -> bool,
) -> u64 {
    let source_bytes = spec.adapters.iter().fold(0_u64, |total, adapter| {
        if include(&adapter.path) {
            let bytes = std::path::absolute(&adapter.path)
                .ok()
                .and_then(|path| file_sizes.get(&path).copied())
                .unwrap_or(0);
            total.saturating_add(bytes)
        } else {
            total
        }
    });
    adapter_resident_source_bytes(engine_id, spec, source_bytes)
}

/// Resolve the TEXT-ENCODER on-disk bytes for the staged split (sc-10894), preferring the provider-owned
/// per-component footprint over the `text_encoder*` subdir scan.
///
/// The subdir scan ([`sum_text_encoder_bytes`]) only recognizes the *diffusers* `text_encoder*` naming;
/// it returns **zero** for a family whose encoder lives elsewhere — boogu's `mllm/`, bernini's flat
/// `t5_encoder.safetensors`, anima's `text_encoders/` under a `split_files/` root — or that has no
/// separable encoder at all (sensenova's flat unified MoT). A zero text-encoder collapses the staged
/// (`max(te, rest)`) peak back to the resident peak, so no `Sequential` saving is ever selected. The
/// provider's `ProviderRegistry::footprint` computes the split from the exact subdirs *its own* loader resolves,
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

/// Total unified memory (GiB) via the shared bounded physical-memory probe, cached process-wide (physical RAM
/// never changes at runtime). The blocking sibling of `gpu::total_unified_memory_gb` — the gate runs
/// on the generator-cache thread, which is already blocking on the weight load, so the bounded
/// subprocess read does not require a Tokio runtime. `None` off macOS or on probe failure retains
/// fail-open admission; failures are retried so a transient launch failure cannot disable the gate.
pub(crate) fn probe_total_unified_memory_gib() -> Option<f64> {
    probe_total_unified_memory_bytes().map(|bytes| bytes as f64 / BYTES_PER_GIB)
}

/// Total unified memory (bytes) — the raw probe [`probe_total_unified_memory_gib`] scales. Cached
/// process-wide after a successful bounded read, so the `sysctl` subprocess is shared however many
/// callers ask. Also read by [`crate::generator_cache::apply_gpu_memory_limit`] at worker startup,
/// which needs the byte figure rather than the GiB one.
pub(crate) fn probe_total_unified_memory_bytes() -> Option<u64> {
    crate::gpu::total_unified_memory_bytes()
}

/// The pure residency decision: given the model's whole-model + text-encoder on-disk bytes, the
/// (possibly emulated) budget, and whether the provider stages components, choose Resident /
/// Sequential / Reject (sc-10839). No IO, no globals — the live [`apply_residency_policy`] resolves
/// those and delegates here.
#[cfg(test)]
pub(crate) fn decide_residency(
    total_bytes: u64,
    te_bytes: u64,
    budget: Option<MlxMemoryBudget>,
    sequential_capable: bool,
) -> ResidencyOutcome {
    decide_residency_with_headroom(
        total_bytes,
        te_bytes,
        budget,
        sequential_capable,
        HEADROOM_GB,
    )
}

pub(super) fn decide_residency_with_headroom(
    total_bytes: u64,
    te_bytes: u64,
    budget: Option<MlxMemoryBudget>,
    sequential_capable: bool,
    headroom_gb: f64,
) -> ResidencyOutcome {
    let peak = decide_residency_by_peak_with_headroom(
        total_bytes,
        te_bytes,
        budget,
        sequential_capable,
        headroom_gb,
    );
    match peak {
        ResidencyOutcome::Reject { .. } => {
            weights_floor_load_admission(total_bytes, te_bytes, budget, sequential_capable)
                .unwrap_or(peak)
        }
        admitted => admitted,
    }
}

/// The PEAK-based residency decision (the pre-sc-12179 logic): compare the predicted whole-model peak
/// (`Σweights + HEADROOM_GB`) — and, for a sequential-capable provider, the staged max-component peak
/// — against the budget. This is the right signal for SELECTING Resident vs Sequential and for the
/// rejection message's `needed`/`staged` numbers, but it rejects too aggressively on small Macs
/// because the flat headroom bundles a pageable 1024² activation transient (sc-12179); the caller
/// folds in the [`weights_floor_load_admission`] floor before honoring a reject.
#[cfg(test)]
pub(super) fn decide_residency_by_peak(
    total_bytes: u64,
    te_bytes: u64,
    budget: Option<MlxMemoryBudget>,
    sequential_capable: bool,
) -> ResidencyOutcome {
    decide_residency_by_peak_with_headroom(
        total_bytes,
        te_bytes,
        budget,
        sequential_capable,
        HEADROOM_GB,
    )
}

pub(super) fn decide_residency_by_peak_with_headroom(
    total_bytes: u64,
    te_bytes: u64,
    budget: Option<MlxMemoryBudget>,
    sequential_capable: bool,
    headroom_gb: f64,
) -> ResidencyOutcome {
    // Generic MLX has no provider-supplied memory-strategy contract or request-scoped evidence yet.
    // Enter the shared selector explicitly as ImplementedUnverified, then keep the established
    // cold-load gate. This adopts one selector API without manufacturing VERIFIED evidence or
    // copying optimized selection logic; request-aware providers can promote only after exposing
    // their own contract, fingerprint, and exact request evidence.
    let shared_observation = generic_mlx_shared_observation(total_bytes, budget, headroom_gb);
    debug_assert!(matches!(
        shared_observation,
        crate::memory_strategy::Selection::Selected {
            selection: gen_core::MemorySelection {
                strategy: gen_core::MemoryStrategy::Resident,
                ..
            },
            ..
        } | crate::memory_strategy::Selection::Reject { .. }
            | crate::memory_strategy::Selection::Unverified { .. }
    ));
    let resident = fit_decision(
        predicted_peak_gb_with_headroom(total_bytes, headroom_gb),
        budget,
    );
    match resolve_offload(resident, sequential_capable) {
        FitDecision::Fits | FitDecision::Unknown => ResidencyOutcome::Resident,
        FitDecision::Offload {
            needed_gb,
            available_gb,
        } => {
            let staged =
                predicted_sequential_peak_gb_with_headroom(total_bytes, te_bytes, headroom_gb);
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

pub(super) fn generic_mlx_shared_observation(
    total_bytes: u64,
    budget: Option<MlxMemoryBudget>,
    headroom_gb: f64,
) -> crate::memory_strategy::Selection {
    use crate::memory_strategy::{Budget, Candidate, RequestScope};
    use gen_core::{
        MemoryBackendRealization, MemoryEvidenceKey, MemoryGeometry, MemoryNumericTier,
        MemoryProviderContract, MemorySelection, MemoryStrategy,
    };

    let tier = MemoryNumericTier {
        precision: gen_core::Precision::Bf16,
        quant: None,
        component_precision_floors: &[],
    };
    let geometry = MemoryGeometry {
        width: 1,
        height: 1,
        batch: 1,
        frames: 1,
        reference_count: 0,
    };
    let selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: Default::default(),
        tier,
    };
    let headroom_bytes = (headroom_gb * BYTES_PER_GIB)
        .max(0.0)
        .ceil()
        .clamp(0.0, u64::MAX as f64) as u64;
    let evidence = sceneworks_core::memory_contracts::unverified_estimate(
        MemoryEvidenceKey {
            model_family: "generic_mlx_cold_load".into(),
            resolved_route: "generic_mlx_cold_load".into(),
            backend: gen_core::MemoryBackend::Mlx,
            tier,
            // The generic estimator models the historical bulk cold load; no generic route
            // defers transformer materialization.
            load_shape: gen_core::LoadShape::EagerMaterialization,
            mode: memory_mode_from_mode_key("image_generation"),
            reference_shape: MemoryReferenceShape::None,
            overlay: Some("resolved_load_spec".into()),
            geometry,
            frames_per_second: None,
            strategy: MemoryStrategy::Resident,
            engaged_composition: vec![MemoryStrategy::Resident],
            parameters: Default::default(),
        },
        gen_core::MEMORY_CALIBRATION_ABI,
        String::new(),
        "sc-15449-contract-v1".into(),
        // sc-22508: the peak is the WHOLE floor — resolved-spec weights plus the activation
        // headroom — because the allowance declared below must be a slice of the peak it grades,
        // not a second charge sitting outside it (`memory_strategy::Candidate
        // ::unmodeled_activation_bytes`). Headroom used to live only in the budget's
        // `reserved_headroom_gb`, which made the declaration an out-of-band charge.
        //
        // Moving it is exactly value-preserving, not a policy change: writing the MLX allowance as
        // A (`ladder_margin_policy::FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE`), the comparison was
        // `weights + A*headroom <= total - headroom` and is now
        // `(weights + headroom) + A*headroom <= total`, the same inequality for ANY A —
        // which is the point of writing it symbolically: the identity is independent of whatever
        // the corpus derives A to be, and the earlier prose that spelled it `0.17` went stale the
        // moment that derivation moved. `reserved_headroom_gb` is zeroed below so headroom is
        // counted once.
        total_bytes.saturating_add(headroom_bytes),
    );
    let contract = MemoryProviderContract::compatibility_default(
        "generic_mlx_cold_load",
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: false,
            cache_eviction: true,
        },
    );
    crate::memory_strategy::select_strategy(
        RequestScope {
            resolved_route: "generic_mlx_cold_load",
            backend: "mlx",
            tier,
            mode: "image_generation",
            overlay: Some("resolved_load_spec"),
            geometry,
        },
        &contract,
        budget.map(|budget| Budget {
            available_gb: budget.total_gb,
            reclaimable_gb: 0.0,
            total_gb: budget.total_gb,
            // Zero, because the headroom is now IN the candidate's peak (see the peak's comment).
            // Leaving it here as well would charge the term twice.
            reserved_headroom_gb: 0.0,
        }),
        &[Candidate {
            selection,
            evidence: &evidence,
            // The generic cold-load estimate is exactly a weights+headroom floor (sc-18096).
            basis: crate::memory_strategy::CandidateBasis::EstimateFloor,
            // sc-22508: the headroom half of that floor is the only uncertain term; the weights
            // half is the resolved load spec's counted bytes. This is a genuine slice of
            // `predicted_peak_bytes` above, which is what the field's contract requires.
            unmodeled_activation_bytes: Some(headroom_bytes),
        }],
    )
}

/// The largest single component's on-disk weight bytes (GiB) — the wired residency the `Sequential`
/// schedule holds at peak (text encoder(s) dropped before the DiT loads). WEIGHTS ONLY, no activation
/// headroom — contrast [`predicted_sequential_peak_gb`], which adds [`HEADROOM_GB`] for the peak
/// estimate. `rest = total − te` (the DiT + VAE + any folded control branch).
pub(super) fn staged_weights_gb(total_bytes: u64, te_bytes: u64) -> f64 {
    let rest_bytes = total_bytes.saturating_sub(te_bytes);
    te_bytes.max(rest_bytes) as f64 / BYTES_PER_GIB
}

/// LOAD-time weights-floor admission (formerly `legacy_admission_override`, and before that
/// `weights_fit_floor`).
///
/// sc-18096 retired this function's old "Decision 1 / Decision 2 transition freeze" meaning — the
/// rule that unmeasured cells "remain byte-for-byte legacy … until that cell's calibration story
/// opts it into evidence" is gone: the REQUEST-scoped gate now synthesizes estimate-backed
/// candidates for every implemented rung, so an unmeasured cell is no longer frozen to
/// resident/sequential behavior. What survives, byte-for-byte (including the 8 GiB q4 guard), is
/// the load-time floor itself: a spec whose bare staged weights fit under the legacy ceiling is
/// ADMITTED for loading even when the peak-based prediction (weights + flat headroom) rejects,
/// because bounding the render transient is the request gate's job, not the loader's. The
/// remaining honest reject is a spec whose staged WEIGHTS do not fit — the load materializes
/// those bytes before any request-time rung can bound anything, so no ladder admission could
/// rescue it.
pub(super) fn weights_floor_load_admission(
    total_bytes: u64,
    te_bytes: u64,
    budget: Option<MlxMemoryBudget>,
    sequential_capable: bool,
) -> Option<ResidencyOutcome> {
    let budget = budget?;
    let reserve = crate::fit_gate::legacy_unified_reserve(budget.total_gb);
    let ceiling_gb = budget.total_gb - reserve.gb;
    if sequential_capable {
        (staged_weights_gb(total_bytes, te_bytes) <= ceiling_gb)
            .then_some(ResidencyOutcome::Sequential)
    } else {
        (total_bytes as f64 / BYTES_PER_GIB <= ceiling_gb).then_some(ResidencyOutcome::Resident)
    }
}

pub(super) fn with_selected_sequential_shape(engine_id: &str, spec: LoadSpec) -> LoadSpec {
    let spec = spec.with_offload_policy(OffloadPolicy::Sequential);
    crate::memory_route_registry::apply_registered_load_shape(
        crate::memory_route_registry::MemoryRouteBackend::Mlx,
        engine_id,
        crate::memory_route_registry::MemoryRouteMode::TextToImage,
        spec,
        true,
    )
}

/// Pre-load admission + residency-selection gate (sc-10835 Phase 0, sc-10839 Phase 1). Called on the
/// generator cache's cold-load path, before `crate::inference_runtime::load` allocates — never on a warm cache hit,
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
    if spec.prepared_file_pins().is_prepared() {
        spec.validate_prepared_file_pins().map_err(|error| {
            crate::classify_engine_error("MLX residency source validation failed", error)
        })?;
    }
    // MOSS's generic stream admits the authenticated SELECTED precision closure plus request
    // workspace inside the cache transaction. The generic source sum includes the unselected SFT
    // transformer during full-FP32 acceleration and must not choose a different load policy.
    if sceneworks_core::moss_image::is_moss(engine_id) {
        return Ok(spec);
    }
    // Respect an offload policy already chosen upstream (defensive: the MLX cache seam normally sees
    // the default `Resident`, but never downgrade a `Sequential` set by another gate).
    if spec.offload_policy == OffloadPolicy::Sequential {
        return Ok(spec);
    }
    let outcome = if spec.prepared_file_pins().is_prepared() {
        try_decide_residency_for_spec(engine_id, &spec)?
    } else {
        decide_residency_for_spec(engine_id, &spec)
    };
    match outcome {
        ResidencyOutcome::Resident => Ok(spec),
        ResidencyOutcome::Sequential => {
            let (total_bytes, te_bytes, _) = if spec.prepared_file_pins().is_prepared() {
                spec_component_bytes_checked(engine_id, &spec)?
            } else {
                spec_component_bytes(engine_id, &spec)
            };
            tracing::info!(
                event = "mlx_sequential_residency_selected",
                engine = %engine_id,
                total_gb = (total_bytes as f64 / BYTES_PER_GIB).round() as i64,
                text_encoder_gb = (te_bytes as f64 / BYTES_PER_GIB).round() as i64,
            );
            let spec = with_selected_sequential_shape(engine_id, spec);
            // Z-Image's shipped rung-4 evidence was captured under an independent deferred loader
            // shape, while its lower rungs are eager. Production reaches both honestly by coupling
            // the deferred shape only to the cold-load branch that selected Sequential residency.
            // Resident hosts retain the eager shape and its four lower-rung bindings.
            Ok(spec)
        }
        ResidencyOutcome::Reject {
            needed_gb,
            available_gb,
            staged_gb,
        } => Err(too_big_error(engine_id, needed_gb, available_gb, staged_gb)),
    }
}

/// The `(total, text-encoder)` on-disk component bytes a `spec` loads (sc-10894 seam). The whole-model
/// sum plus the staged text-encoder split, preferring the provider-owned per-component footprint over
/// the `text_encoder*` subdir scan (which reads ZERO for boogu `mllm/`, bernini flat `t5_encoder`,
/// anima `text_encoders/`, etc.), and folding a separate `spec.control` (qwen_image_control's VACE
/// branch) into the HEAVY side so the staged split `rest = total − te` counts it on the DiT side.
///
/// sc-16014-resolution: rehosted-q4-q8. Lens q4/q8 use provider-detected MLX affine packs, so their
/// provider footprint stays disk-derived. The bf16 artifact remains MXFP4 and receives both its
/// measured materialization delta and the architecture-specific activation headroom. A genuinely
/// bf16-on-disk encoder receives the same activation headroom but no invented weight expansion.
pub(super) fn spec_component_bytes(
    engine_id: &str,
    spec: &LoadSpec,
) -> (u64, u64, HeadroomAllowance) {
    let footprint = crate::inference_runtime::media()
        .footprint(engine_id, spec)
        .ok()
        .flatten();
    spec_component_bytes_with_provider_footprint(engine_id, spec, footprint)
}

pub(super) fn spec_component_bytes_checked(
    engine_id: &str,
    spec: &LoadSpec,
) -> WorkerResult<(u64, u64, HeadroomAllowance)> {
    let footprint = crate::inference_runtime::media()
        .footprint(engine_id, spec)
        .map_err(|error| crate::classify_engine_error("MLX footprint query failed", error))?;
    spec_component_bytes_with_provider_footprint_checked(engine_id, spec, footprint).map_err(
        |error| crate::classify_engine_error("MLX prepared source accounting failed", error),
    )
}

/// Provider-injected core of [`spec_component_bytes`]. The live path obtains `footprint` from the
/// active registry; keeping the arithmetic independent of registry composition lets platform-neutral
/// tests exercise the exact Lens materialization and overlay accounting used on macOS.
pub(super) fn spec_component_bytes_with_provider_footprint(
    engine_id: &str,
    spec: &LoadSpec,
    footprint: Option<PerComponentBytes>,
) -> (u64, u64, HeadroomAllowance) {
    let external_adapter_bytes = external_adapter_source_bytes_for_gate(engine_id, spec);
    // The infallible compatibility/planning seam predates prepared specs. Valid selected encoders
    // resolve through gen-core; an invalid ad-hoc spec retains the old conservative recursive
    // estimate and will still fail contract validation before a real provider load.
    let selected_text_encoder = spec.text_encoder.as_ref().map(|source| {
        resolved_text_encoder_source(spec, source).unwrap_or_else(|_| ResolvedTextEncoderSource {
            path: crate::weights_source::WeightsSourceExt::path(source).to_path_buf(),
            direct_shard_bytes: weights_source_bytes(source),
        })
    });
    spec_component_bytes_with_provider_footprint_and_sizes(
        engine_id,
        spec,
        footprint,
        selected_text_encoder.as_ref(),
        weights_source_bytes,
        external_adapter_bytes,
        adapter_path_bytes,
    )
}

pub(super) fn spec_component_bytes_with_provider_footprint_and_sizes(
    engine_id: &str,
    spec: &LoadSpec,
    footprint: Option<PerComponentBytes>,
    selected_text_encoder: Option<&ResolvedTextEncoderSource>,
    source_bytes: impl Fn(&WeightsSource) -> u64,
    external_adapter_bytes: u64,
    adapter_source_bytes: impl Fn(&Path) -> u64,
) -> (u64, u64, HeadroomAllowance) {
    let footprint_te = footprint.map(|fp| fp.text_encoder);
    // A provider that accepts a primary single-file checkpoint owns the meaning of its named
    // companions. Its footprint is the exact consumed assembly (imported DiT + companion TE/VAE),
    // not a split of `spec.weights` alone. Treat it as authoritative so a `base_snapshot` supplying
    // tokenizer/config does not recursively re-price the snapshot transformer the primary File
    // replaces.
    let provider_owned_file_footprint = matches!(&spec.weights, WeightsSource::File(_))
        .then_some(footprint)
        .flatten();
    let mut headroom = HeadroomAllowance::GENERIC;
    let (mut total_bytes, mut te_bytes) = match &spec.weights {
        WeightsSource::Dir(dir) => {
            let mut total = sum_safetensors_bytes(dir);
            let te = resolve_text_encoder_bytes(footprint_te, dir);
            if matches!(engine_id, "lens" | "lens_turbo") {
                let disk_te = sum_safetensors_bytes(&dir.join("text_encoder"));
                let materialized_expansion = te.saturating_sub(disk_te);
                total = total.saturating_add(materialized_expansion);
                if spec.quantize.is_none() && packed_quant_bits(dir, "text_encoder").is_none() {
                    headroom = HeadroomAllowance::LENS_DENSE;
                }
            }
            (total, te)
        }
        // A single-file source has no component tree of its own. A File-aware provider footprint
        // prices the complete consumed assembly; otherwise keep the primary file here and add only
        // the well-known companion subtrees below.
        WeightsSource::File(_) => provider_owned_file_footprint.map_or_else(
            || (source_bytes(&spec.weights), footprint_te.unwrap_or(0)),
            |components| {
                (
                    components
                        .text_encoder
                        .saturating_add(components.dit)
                        .saturating_add(components.vae),
                    components.text_encoder,
                )
            },
        ),
    };
    // Every directory source in this set was priced recursively. A selected encoder at the same
    // path or below one is already covered; a selected encoder above one is not, because its exact
    // loader inventory contains direct shards only and excludes nested descendants.
    let mut recursively_priced_paths =
        vec![crate::weights_source::WeightsSourceExt::path(&spec.weights)];
    if let Some(control) = &spec.control {
        let bytes = source_bytes(control);
        total_bytes = total_bytes.saturating_add(bytes);
        recursively_priced_paths.push(crate::weights_source::WeightsSourceExt::path(control));
    }
    // Read the actual adapter sources at the same pre-load seam as controls. Provider-specific
    // residency matters: packed Wan keeps additive residuals, while dense Wan folds them into the
    // base and adds zero independent bytes. Other providers conservatively retain the source bytes;
    // a typed component contract may replace them with a more exact resident measurement below.
    total_bytes = total_bytes.saturating_add(external_adapter_bytes);
    if external_adapter_bytes > 0 {
        recursively_priced_paths.extend(spec.adapters.iter().filter_map(|adapter| {
            let inside = match &spec.weights {
                WeightsSource::Dir(root) => adapter.path.starts_with(root),
                WeightsSource::File(_) => false,
            };
            (!inside && adapter_source_bytes(&adapter.path) > 0).then_some(adapter.path.as_path())
        }));
    }
    // Caller-provisioned components (epic 13657) are staged from a DIFFERENT snapshot than
    // `spec.weights`, so the dir scan above cannot see them (sc-15154). Mage-Flow's per-tier dir
    // holds the DiT alone — its text encoder and VAE are bit-identical across the six variants and
    // hosted once in a shared mirror — so an unstaged sum scored a q4 edit install at 2.17 GiB
    // against a real 6.52 GiB, which both under-quoted the over-budget message and let the
    // the legacy override admit tiers that do not fit. A component resolved to a path INSIDE the
    // weights dir is skipped, because the scan already counted it.
    for (component_id, source) in &spec.components {
        if provider_owned_file_footprint.is_some() {
            // The File-aware footprint already reflects every named source the provider consumes.
            continue;
        }
        if matches!(&spec.weights, WeightsSource::File(_))
            && component_id == gen_core::BASE_SNAPSHOT_COMPONENT
        {
            // Compatibility fallback for a File-capable provider that has not published its
            // footprint yet. The companion snapshot supplies TE/VAE/tokenizer/config only; its
            // transformer is replaced by `spec.weights`. Explicit ComfyUI components replace the
            // corresponding snapshot subtree and are counted normally on their own map entries.
            if let WeightsSource::Dir(root) = source {
                if !spec
                    .components
                    .contains_key(gen_core::COMFYUI_TEXT_ENCODER_COMPONENT)
                {
                    let companion_te = sum_safetensors_bytes(&root.join("text_encoder"));
                    total_bytes = total_bytes.saturating_add(companion_te);
                    te_bytes = te_bytes.saturating_add(companion_te);
                }
                if !spec
                    .components
                    .contains_key(gen_core::COMFYUI_VAE_COMPONENT)
                {
                    total_bytes =
                        total_bytes.saturating_add(sum_safetensors_bytes(&root.join("vae")));
                }
            }
            continue;
        }
        let inside = match source {
            WeightsSource::Dir(path) | WeightsSource::File(path) => match &spec.weights {
                WeightsSource::Dir(root) => path.starts_with(root),
                WeightsSource::File(_) => false,
            },
        };
        if !inside {
            let component_bytes = source_bytes(source);
            total_bytes = total_bytes.saturating_add(component_bytes);
            recursively_priced_paths.push(crate::weights_source::WeightsSourceExt::path(source));
            if matches!(&spec.weights, WeightsSource::File(_))
                && component_id == gen_core::COMFYUI_TEXT_ENCODER_COMPONENT
            {
                te_bytes = te_bytes.saturating_add(component_bytes);
            }
            // A component staged from OUTSIDE a `Dir` weights root SUPERSEDES any identically named
            // subtree INSIDE that root: the loader reads the staged copy, so the in-root copy is dead
            // weight the `sum_safetensors_bytes(dir)` above already (wrongly) counted. Subtract it.
            //
            // MiniMax-H3 is the motivating case (sc-17137). `spec.weights` is the shared UPSTREAM
            // snapshot — the VAEs and tokenizer a render genuinely loads — but that same snapshot also
            // carries the dense `transformer/`, `transformer_ref/` and `text_encoder/`, ~180 GB a
            // q4/q8 render never touches because the tiered DiT (`components["transformer"]`) and the
            // packed encoder (`components["text_encoder"]`) are staged in from `SceneWorks/
            // minimax-h3-mlx`. Without this subtraction the flat sum quotes ~250 GB for a model whose
            // real q4 resident set is ~55 GB, so a 128 GB Mac refuses a model that fits with room to
            // spare. The `<name>_ref` sibling covers a DiT split into base/reference partitions (the
            // reference partition is likewise staged from the tier repo, so the upstream copy is dead).
            //
            // No-op for every model whose weights dir has no subdir named like a staged component —
            // i.e. whose `spec.weights` IS its DiT rather than a companion root — so it cannot move any
            // other model's admission. Exact cancellation: the subtree scan re-walks the same files the
            // whole-tree scan counted (`sum_safetensors_bytes` dedups by canonical dir per call).
            if let WeightsSource::Dir(root) = &spec.weights {
                let superseded = sum_safetensors_bytes(&root.join(component_id)).saturating_add(
                    sum_safetensors_bytes(&root.join(format!("{component_id}_ref"))),
                );
                total_bytes = total_bytes.saturating_sub(superseded);
                // The dir scan measured the DENSE upstream encoder into `te_bytes`; once superseded,
                // the resident (and sequential-droppable) text encoder is the staged packed one. Keep
                // the `Σweights − te` staged peak coherent rather than subtracting bytes no longer
                // resident. Guarded to the packed-TE component key MiniMax-H3 stages (sc-19120).
                if superseded > 0 && component_id == "text_encoder" {
                    te_bytes = component_bytes;
                }
            }
        }
    }
    // The optional selected encoder is a typed source, not a named component. Include its exact
    // resolved direct-shard bytes in the resident total. Equal/child paths are already covered by a
    // recursively priced source. A selected encoder that is a parent of the base/control/component
    // still adds all of its direct shards: gen-core's loader does not recurse into those descendants.
    // A selection-aware provider owns the staged/materialized TE footprint; exact direct-shard bytes
    // are the fallback only when that provider fact is absent or zero. Provider-owned File
    // footprints are complete assemblies and remain authoritative.
    if provider_owned_file_footprint.is_none() {
        if let Some(selected) = selected_text_encoder {
            let already_priced = recursively_priced_paths
                .iter()
                .any(|path| selected.path.starts_with(path));
            let additional_bytes = if already_priced {
                0
            } else {
                selected.direct_shard_bytes
            };
            total_bytes = total_bytes.saturating_add(additional_bytes);
            te_bytes = footprint_te
                .filter(|bytes| *bytes > 0)
                .unwrap_or(selected.direct_shard_bytes);
        }
    }
    (total_bytes, te_bytes, headroom)
}

pub(super) fn spec_component_bytes_with_provider_footprint_checked(
    engine_id: &str,
    spec: &LoadSpec,
    footprint: Option<PerComponentBytes>,
) -> gen_core::Result<(u64, u64, HeadroomAllowance)> {
    let file_sizes = prepared_file_sizes(spec)?;
    let selected_text_encoder = spec
        .text_encoder
        .as_ref()
        .map(|source| resolved_text_encoder_source(spec, source))
        .transpose()?;
    // Fill a File slot the provider left EMPTY with the exact prepared target size — and only such a
    // slot. A non-zero value in a File slot is a provider fact about the assembly it will actually
    // materialize (the checkpoint-import lane prices the DECODED plan: an fp8/int8 single-file import
    // is resident at its decode width, not at its on-disk width), so overwriting it with the raw file
    // length under-prices the modeled side of the resident-attribution guard by exactly the decode
    // delta and refuses imports that fit (sc-20651).
    //
    // The overwrite was introduced to stop compatibility-mode provider code from smuggling its own
    // `stat` into prepared admission identity. It never bought that: `prepared_file_sizes` above runs
    // `validate_prepared_file_pins` → `ensure_unchanged` on every configured File, so a prepared spec
    // whose file no longer matches its token is a hard error before any byte is priced, and for a
    // genuinely byte-preserving provider the substituted value is the value already there. What the
    // overwrite did buy was silently discarding every plan-priced fact. The infallible sibling
    // `spec_component_bytes_with_provider_footprint` never had it; the two seams now agree.
    let footprint = footprint.map(|mut footprint| {
        let fill = |slot: &mut u64, source: &WeightsSource| {
            if *slot == 0 {
                *slot = prepared_source_bytes(source, &file_sizes);
            }
        };
        if matches!(spec.weights, WeightsSource::File(_)) {
            fill(&mut footprint.dit, &spec.weights);
        }
        if let Some(source @ WeightsSource::File(_)) = spec
            .components
            .get(gen_core::COMFYUI_TEXT_ENCODER_COMPONENT)
        {
            fill(&mut footprint.text_encoder, source);
        }
        if let Some(source @ WeightsSource::File(_)) =
            spec.components.get(gen_core::COMFYUI_VAE_COMPONENT)
        {
            fill(&mut footprint.vae, source);
        }
        footprint
    });
    let external_adapter_bytes =
        prepared_adapter_source_bytes_for_gate_where(engine_id, spec, &file_sizes, |path| {
            match &spec.weights {
                WeightsSource::Dir(root) => !path.starts_with(root),
                WeightsSource::File(_) => true,
            }
        });
    Ok(spec_component_bytes_with_provider_footprint_and_sizes(
        engine_id,
        spec,
        footprint,
        selected_text_encoder.as_ref(),
        |source| prepared_source_bytes(source, &file_sizes),
        external_adapter_bytes,
        |path| {
            std::path::absolute(path)
                .ok()
                .and_then(|path| file_sizes.get(&path).copied())
                .unwrap_or(0)
        },
    ))
}

pub(super) fn packed_quant_bits(root: &std::path::Path, component: &str) -> Option<i64> {
    let config = std::fs::read(root.join(component).join("config.json")).ok()?;
    serde_json::from_slice::<serde_json::Value>(&config)
        .ok()?
        .get("quantization")?
        .get("bits")?
        .as_i64()
}

/// The residency outcome (Resident / Sequential / Reject) a `spec` would take against this machine's
/// unified-memory budget — the pure decision behind [`apply_residency_policy`], factored out so the
/// capability downtier (sc-10733) can evaluate a candidate tier's fit at the base.rs seam WITHOUT
/// building the final spec twice. Same budget + component-byte + sequential-capability inputs the live
/// gate uses, so the seam's downtier choice and the cache's admission never disagree.
pub(crate) fn decide_residency_for_spec(engine_id: &str, spec: &LoadSpec) -> ResidencyOutcome {
    let budget = resolve_budget(probe_total_unified_memory_gib(), mlx_memory_cap_gb());
    let (total_bytes, te_bytes, headroom) = spec_component_bytes(engine_id, spec);
    // The LOAD-time gate is resolution-blind, so it budgets the family's whole flat allowance and
    // has no use for the reserve split (sc-16195 changes only the request-scoped estimator).
    decide_residency_with_headroom(
        total_bytes,
        te_bytes,
        budget,
        engine_supports_sequential(engine_id),
        headroom.total_gb,
    )
}

pub(super) fn try_decide_residency_for_spec(
    engine_id: &str,
    spec: &LoadSpec,
) -> WorkerResult<ResidencyOutcome> {
    let budget = resolve_budget(probe_total_unified_memory_gib(), mlx_memory_cap_gb());
    let (total_bytes, te_bytes, headroom) = spec_component_bytes_checked(engine_id, spec)?;
    Ok(decide_residency_with_headroom(
        total_bytes,
        te_bytes,
        budget,
        engine_supports_sequential(engine_id),
        headroom.total_gb,
    ))
}

/// The residency outcome for a candidate tier's WEIGHTS DIR — a bare-`Dir` convenience over
/// [`decide_residency_for_spec`], kept for the live real-weights gate below.
///
/// The base.rs capability downtier (sc-10733) used to call this and now builds its probe spec with
/// `tier_probe_spec` instead, because a bare `Dir` under-counts any model whose components are
/// caller-staged from another snapshot (sc-15154).
#[cfg(test)]
pub(crate) fn residency_for_dir(
    engine_id: &str,
    weights_dir: &std::path::Path,
) -> ResidencyOutcome {
    let spec = LoadSpec::new(WeightsSource::Dir(weights_dir.to_path_buf()));
    decide_residency_for_spec(engine_id, &spec)
}

/// Build the actionable over-budget rejection. `staged_gb` is `Some` when sequential residency was
/// tried and its staged peak still overflows (so the message names both the resident and the staged
/// requirement — telling the user even one-component-at-a-time won't fit); `None` for a plain resident
/// reject on a non-staging provider. Split out so the message is testable without the live probe.
pub(super) fn too_big_error(
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
