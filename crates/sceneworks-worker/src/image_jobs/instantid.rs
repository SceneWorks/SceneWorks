/// The SceneWorks model id for native InstantID (production = InstantID on RealVisXL_V5.0).
const INSTANTID_MODEL: &str = "instantid_realvisxl";
/// SDXL base for InstantID when the manifest omits `repo`: the SceneWorks quant-matrix turnkey
/// re-host (sc-9965, epic 8506) — the SAME `SceneWorks/realvisxl-mlx` the plain `realvisxl` model
/// loads, with self-contained `bf16/` (default — the validated fp16 identity envelope) + packed
/// `q8/` / `q4/` tier subdirs. Was upstream `SG161222/RealVisXL_V5.0` (dense diffusers root, never
/// tier-aware); `resolve_instantid_sdxl_base` now resolves the selected tier via
/// [`instantid_tier_subdir`]. The mlx-gen SDXL loaders packed-detect the UNet + both CLIP tiers
/// (sc-8746), so Q4/Q8 load packed with no install-time convert; the candle lane runs dense f16 and
/// always loads `bf16/`. The InstantID adapter weights (IdentityNet / ip-adapter / face stack) are
/// separate on-demand dense loads and stay dense.
const INSTANTID_SDXL_REPO: &str = "SceneWorks/realvisxl-mlx";
/// Stock InstantID checkpoint repo — the IdentityNet `ControlNetModel/` lives here.
const INSTANTID_CONTROLNET_REPO: &str = "InstantX/InstantID";
/// Pinned revision for the stock InstantX IdentityNet repo (sc-9879, F-077 follow-up).
const INSTANTID_CONTROLNET_REVISION: &str = "57b32dfee076092ad2930c71fd6d439c2c3b1820";
/// Converted-weights bundle (download-on-first-use): the MLX `ip-adapter.safetensors`
/// (`tools/convert_instantid.py`) + the native face stack `scrfd_10g.safetensors`
/// (`convert_scrfd.py`) + `arcface_iresnet100.safetensors` (`convert_glintr100.py`). Public
/// repo, mirroring the YOLO11 / SAM2 `SceneWorks/*-mlx` uploads (sc-3633 / sc-3707).
const INSTANTID_MLX_REPO: &str = "SceneWorks/instantid-mlx";
/// Pinned revision for the first-party converted InstantID bundle (sc-9879, F-077 follow-up). Even
/// though `SceneWorks/instantid-mlx` is a first-party repo, fetching the mutable `main` branch means a
/// re-push (or a compromised token) could silently swap the ip-adapter / SCRFD / ArcFace weights we
/// load. Pin the exact commit for defense-in-depth (mirrors sc-8879/sc-9682). HF's tree API still reports
/// each file's `lfs.oid`, which `ensure_hf_cached_file` verifies against. NOTE: the candle PuLID lane
/// reuses this SAME repo (`pulid_candle.rs` `PULID_CANDLE_FACE_REPO`); its pin must match this sha.
pub(crate) const INSTANTID_MLX_REVISION: &str =
    "bca0cacf8e5e04529bb2b326a521361b02be84fd";
const INSTANTID_IP_ADAPTER_FILE: &str = "ip-adapter.safetensors";
// `pub(crate)` so the Dataset Doctor face pass (sc-6538) can join them under the bundle dir that
// `ensure_face_stack_dir` stages, to load the MLX `FaceAnalysis` (SCRFD + ArcFace) directly.
pub(crate) const INSTANTID_SCRFD_FILE: &str = "scrfd_10g.safetensors";
pub(crate) const INSTANTID_ARCFACE_FILE: &str = "arcface_iresnet100.safetensors";
/// The IdentityNet weight file inside `ControlNetModel/` (a stock diffusers SDXL ControlNet).
const INSTANTID_CONTROLNET_FILES: [&str; 2] =
    ["config.json", "diffusion_pytorch_model.safetensors"];
/// Torch-parity defaults (the `instantid_realvisxl` MODEL_TARGETS): RealVisXL is tuned for a
/// low CFG; the engine's own `InstantIdRequest::default` guidance (5.0) is for base SDXL.
const INSTANTID_DEFAULT_STEPS: u32 = 30;
const INSTANTID_DEFAULT_GUIDANCE: f32 = 3.0;
const INSTANTID_IP_SCALE: f32 = 0.8;
const INSTANTID_CONTROLNET_SCALE: f32 = 0.8;
/// Angle-set default `controlnetConditioningScale` (sc-8354). The sc-8222 real-weight A/B (full
/// 5-point cn sweep, paired on the built-in 11-angle set) found the landmark-ControlNet lock is the
/// only viable lever that sharpens angle-set output: lowering it from 0.80 to 0.65 lifts median blur
/// variance ~+11% and clears the most angles above the person blur floor (9/11) at a small identity
/// cost (−0.012 mean ArcFace cosine, well inside the InstantID ~0.82 envelope). Identity / pose modes
/// keep the 0.80 default — this softer lock is angle-set-only.
const INSTANTID_ANGLE_CONTROLNET_SCALE: f32 = 0.65;
/// xinsir OpenPose-SDXL ControlNet (the pose-mode second branch, sc-3117). Loads via the stock
/// `load_controlnet` (no conversion) — `image_adapters.py:615-617` parity.
const INSTANTID_OPENPOSE_REPO: &str = "xinsir/controlnet-openpose-sdxl-1.0";
/// Pinned revision for the xinsir OpenPose-SDXL ControlNet (sc-9879, F-077 follow-up).
const INSTANTID_OPENPOSE_REVISION: &str = "23f966cd5cfdd3f7729c903e243d87152162d2b7";
/// Pinned commit revisions for the MLX PuLID-FLUX download repos (sc-11168, F-007 — completes the sc-9879
/// rollout on the MLX lane). The MLX PuLID route (`pulid.rs`) fetches its adapter + EVA/BiSeNet bundle
/// through `ensure_instantid_file`, so those repos route through `instantid_revision` below and were
/// falling back to the mutable `main` branch — the candle twin (`pulid_candle.rs`) already pins them, so
/// the MLX side was the last unpinned lane. Pin each to its exact commit for defense-in-depth. The repo
/// NAMES are the macOS-only `pulid.rs` `PULID_ADAPTER_REPO` / `PULID_MLX_REPO` consts; they are matched
/// here as string literals because `instantid.rs` also compiles on the candle lane where those macOS-only
/// consts do not exist (a macOS test ties the literals back to the consts). These shas MUST equal the
/// candle `PULID_CANDLE_ADAPTER_REVISION` / `PULID_CANDLE_MLX_REVISION` (same repos). PuLID's third repo
/// (SCRFD / ArcFace) IS `SceneWorks/instantid-mlx`, already pinned above by `INSTANTID_MLX_REVISION`.
const PULID_ADAPTER_REVISION: &str = "492b1451255dc9d9bc3c857259690b5f8b998d4a";
const PULID_MLX_REVISION: &str = "78ef91f977eae16d66fb191caf003154b7a0a0b8";
/// Torch-parity default OpenPose lock (`instantid_adapter.py::_openpose_scale`, default 0.7).
const INSTANTID_OPENPOSE_SCALE: f32 = 0.7;
/// The face-restore re-render side (the engine's production crop size, sc-3380).
const INSTANTID_FACE_RESTORE_SIDE: u32 = 1024;
/// The adapter/engine id recorded on InstantID assets + telemetry, selected by backend: the native
/// MLX provider on macOS, the candle (Windows/CUDA) provider off-Mac (sc-5491). Distinguishes the two
/// lanes in the asset sidecar + the `instantIdEngine` raw-settings key.
#[cfg(target_os = "macos")]
const INSTANTID_ENGINE: &str = "mlx_instantid";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const INSTANTID_ENGINE: &str = "candle_instantid";

// ---------------------------------------------------------------------------
// Request-scoped memory admission (sc-20799, epic 15448)
// ---------------------------------------------------------------------------
//
// Both InstantID engines ship the SAME bespoke admission surface — an
// `instantid`-owned `MemoryProviderContract`, an `InstantIdMemoryIdentity` that
// names the exact composition, `InstantId::load_with_memory_context`, and a
// per-request `begin_memory_request` scope. Until this, the worker called the
// unadmitted `InstantId::load` on both lanes, so the whole surface was
// unreachable and the memory matrix reported InstantID `Missing` on both
// backends. The candle-only sc-16069 `admit_conditioning_paths` floor stays: it
// is a pre-load on-disk floor, complementary to (not a substitute for) the
// provider's own request-scoped handshake.
//
// The two crates' `memory_strategy` modules are byte-identical in shape (the
// route/identity/overlay-key/evidence-revision are deliberately backend-neutral
// so cross-backend evidence is not split); only `provider_contract` differs — the
// MLX one takes the numeric tier, the candle one is dense-only.
#[cfg(target_os = "macos")]
use runtime_macos::providers::instantid::memory_strategy as instantid_memory;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::instantid::memory_strategy as instantid_memory;

use instantid_memory::{InstantIdMemoryIdentity, InstantIdRoute};

/// Bytes per binary gigabyte — the currency the GB-denominated worker probes
/// (`gpu::total_unified_memory_gb`, `vram_gate::VramBudget`) are converted into for
/// [`gen_core::MemoryBudget`], which is byte-denominated.
const INSTANTID_BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// GB → bytes. The float→int cast saturates (a negative reading lands on 0, an absurd one on
/// `u64::MAX`), so a nonsense probe can never become a giant budget.
fn instantid_gib_bytes(gb: f64) -> u64 {
    (gb * INSTANTID_BYTES_PER_GIB).ceil() as u64
}

/// The single path inside a [`gen_core::WeightsSource`]. Local rather than
/// `conditioning_fit::weights_source_path` because that module is candle-only and this seam runs on
/// BOTH lanes.
fn instantid_source_path(source: &WeightsSource) -> &Path {
    match source {
        WeightsSource::Dir(path) | WeightsSource::File(path) => path.as_path(),
    }
}

/// The exact, ORDERED artifact set one InstantID load consumes, each tagged with the role it plays.
///
/// This is the input to BOTH the admission fingerprint and the priced floor, so the two can never
/// describe different artifact sets. The role tag is part of the hashed stream, so swapping two
/// files between roles changes the fingerprint even though the path multiset is unchanged.
#[allow(clippy::too_many_arguments)]
fn instantid_artifact_entries<'a>(
    sdxl_base: &'a Path,
    identitynet: &'a WeightsSource,
    ip_adapter: &'a Path,
    scrfd: &'a Path,
    arcface: &'a Path,
    openpose: Option<&'a WeightsSource>,
    adapters: &'a [AdapterSpec],
    pid: Option<&'a gen_core::PidWeights>,
) -> Vec<(&'static str, &'a Path)> {
    let mut entries: Vec<(&'static str, &Path)> = vec![
        ("sdxl_base", sdxl_base),
        ("identitynet", instantid_source_path(identitynet)),
        ("ip_adapter", ip_adapter),
        ("scrfd", scrfd),
        ("arcface", arcface),
    ];
    if let Some(openpose) = openpose {
        entries.push(("openpose", instantid_source_path(openpose)));
    }
    entries.extend(
        adapters
            .iter()
            .map(|adapter| ("adapter", adapter.path.as_path())),
    );
    if let Some(pid) = pid {
        entries.push(("pid_checkpoint", instantid_source_path(&pid.checkpoint)));
        entries.push(("pid_gemma", instantid_source_path(&pid.gemma)));
    }
    entries
}

/// A deterministic digest of the exact artifact identities [`instantid_artifact_entries`] named.
///
/// It is the `artifact_fingerprint` axis of [`InstantIdMemoryIdentity`], so it lands verbatim inside
/// the provider's `overlay_key()` — two different artifact sets therefore never share an overlay key
/// and can never price each other. Hex, so it can never contain the `=` the shared safety gate
/// rejects in an overlay.
///
/// The digest is over the ORDERED paths with a record separator between them. Order carries the
/// role (each role sits at a fixed position in [`instantid_artifact_entries`]), so hashing the role
/// name too would add nothing; the separator is what stops two different splits of the same
/// concatenated text — `["a", "bc"]` and `["ab", "c"]` — from colliding.
fn instantid_artifact_fingerprint(entries: &[(&'static str, &Path)]) -> String {
    let mut hasher = Sha256::new();
    for (_role, path) in entries {
        hasher.update(path.to_string_lossy().as_bytes());
        hasher.update(b"\x1e");
    }
    format!("{:x}", hasher.finalize())
}

/// Drop every entry CONTAINED in an earlier-kept entry so a recursive directory scan and an explicit
/// file inside it are not both counted. Same shortest-path-first containment rule (and same reason)
/// as `conditioning_fit::dedupe_contained`, reimplemented here because that module is candle-only
/// and this pricing must be identical on both lanes. An OVER-count is the one direction this must
/// never have: it would refuse a render that fits.
fn instantid_priced_entries<'a>(entries: &[(&'static str, &'a Path)]) -> Vec<&'a Path> {
    let mut ordered: Vec<&Path> = entries.iter().map(|(_, path)| *path).collect();
    ordered.sort_by_key(|path| path.as_os_str().len());
    let mut kept: Vec<&Path> = Vec::with_capacity(ordered.len());
    for path in ordered {
        if kept.iter().any(|keeper| path.starts_with(keeper)) {
            continue;
        }
        kept.push(path);
    }
    kept
}

/// On-disk bytes of the whole InstantID artifact set: a file's own length, a directory's recursive
/// `.safetensors` sum. Missing/unreadable contributes 0 (less evidence, never a phantom
/// requirement), so a fully unreadable set sums to 0 and the admission carries no floor at all.
fn instantid_artifact_bytes(entries: &[(&'static str, &Path)]) -> u64 {
    instantid_priced_entries(entries)
        .into_iter()
        .map(|path| match std::fs::metadata(path) {
            Ok(meta) if meta.is_file() => meta.len(),
            Ok(meta) if meta.is_dir() => crate::mlx_fit_gate::sum_safetensors_bytes(path),
            _ => 0,
        })
        .fold(0u64, u64::saturating_add)
}

/// The manifest's own memory declaration for this backend + tier, if it carries one:
/// `<backend>.vramGbByTier[<tier>]` then `<backend>.minMemoryGb`. `instantid_realvisxl` declares
/// NEITHER today (it has no `mlx` or `candle` memory block at all), so the caller falls back to the
/// priced artifact bytes — this reader exists so a measured figure, once added, is honored rather
/// than ignored. Never a constant of our own invention.
fn instantid_manifest_peak_gb(
    manifest_entry: &JsonObject,
    backend_key: &str,
    tier_key: &str,
) -> Option<f64> {
    let block = manifest_entry.get(backend_key)?;
    block
        .get("vramGbByTier")
        .and_then(|tiers| tiers.get(tier_key))
        .and_then(Value::as_f64)
        .or_else(|| block.get("minMemoryGb").and_then(Value::as_f64))
}

/// The manifest block key + resolved tier key for THIS backend's InstantID load.
///
/// MLX honors the packed q4/q8 tiers ([`instantid_quant`] / [`instantid_tier_subdir`]); the candle
/// stack is dense-only and always loads `bf16/`, which is exactly what the candle provider's
/// `resolved_numeric_tier()` declares.
#[cfg(target_os = "macos")]
fn instantid_memory_backend_keys(request: &ImageRequest) -> (&'static str, &'static str) {
    (
        "mlx",
        match instantid_quant(request).0 {
            Some(4) => "q4",
            Some(8) => "q8",
            _ => "bf16",
        },
    )
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn instantid_memory_backend_keys(request: &ImageRequest) -> (&'static str, &'static str) {
    let _ = request;
    ("candle", "bf16")
}

/// The numeric tier this load actually materializes, in the provider's currency.
///
/// MLX: `Bf16` precision (the dense-default sentinel both crates use) with the packed quant the
/// request selected, matching `provider_contract(tier)`'s per-request tier validation. Candle: the
/// provider's own `resolved_numeric_tier()` (Bf16 / no quant).
#[cfg(target_os = "macos")]
fn instantid_memory_tier(request: &ImageRequest) -> gen_core::MemoryNumericTier {
    gen_core::MemoryNumericTier {
        quant: match instantid_quant(request).0 {
            Some(4) => Some(Quant::Q4),
            Some(8) => Some(Quant::Q8),
            _ => None,
        },
        ..instantid_memory::dense_numeric_tier()
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn instantid_memory_tier(request: &ImageRequest) -> gen_core::MemoryNumericTier {
    let _ = request;
    instantid_memory::resolved_numeric_tier()
}

/// The worker-owned live memory budget for an InstantID admission.
///
/// macOS: total unified memory (`sysctl hw.memsize`) — the same reading the MLX fit gate sizes
/// against; there is no per-process committed reading at this seam, so `committed_bytes` is 0.
/// Candle: the live NVML total/free, through the same `apply_vram_cap` emulation knob the SenseNova
/// direct-admission lane uses, with `total - free` charged as committed.
///
/// A host with NO readable budget yields a budget that admits exactly this request — the worker's
/// standing "no evidence never blocks" convention (`FitDecision::Unknown`,
/// `ConditioningFit::NoBudget`), not a refusal on an unread host.
#[cfg(target_os = "macos")]
async fn instantid_memory_budget(
    _settings: &Settings,
    predicted_peak_bytes: u64,
) -> gen_core::MemoryBudget {
    let total_bytes = crate::gpu::total_unified_memory_gb()
        .await
        .map(instantid_gib_bytes)
        .unwrap_or(predicted_peak_bytes);
    gen_core::MemoryBudget {
        total_bytes,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn instantid_memory_budget(
    settings: &Settings,
    predicted_peak_bytes: u64,
) -> gen_core::MemoryBudget {
    let budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    match budget {
        Some(budget) => gen_core::MemoryBudget {
            total_bytes: instantid_gib_bytes(budget.total_gb),
            committed_bytes: instantid_gib_bytes((budget.total_gb - budget.free_gb).max(0.0)),
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        None => gen_core::MemoryBudget {
            total_bytes: predicted_peak_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
    }
}

/// Build the exact `MemoryRunContext` the InstantID provider admits.
///
/// Every field is pinned by the provider's own safety check: `character_image` mode, one reference,
/// batch/frames 1, `use_pid` / `has_phases` equal to the identity's, `overlay` equal to the
/// identity's `overlay_key()`, the contract's calibration handshake (abi + fingerprint +
/// `EagerMaterialization` load shape), the loaded numeric tier, and a budget that fits the predicted
/// peak. `Resident` is the honest strategy: this lane loads the whole stack up front (the provider's
/// `StagedResidency` arm releases components between phases, which this worker does not drive), and
/// `Resident` is not an "optimized" strategy, so `MemoryOptimizationAuthority::Resident` is the
/// matching authority.
///
/// A contract with no calibration identity is an error, not a default: the InstantID contract
/// declares one, and silently substituting a blank fingerprint would make the handshake vacuous.
fn instantid_memory_context(
    contract: &gen_core::MemoryProviderContract,
    tier: gen_core::MemoryNumericTier,
    identity: &InstantIdMemoryIdentity,
    width: u32,
    height: u32,
    budget: gen_core::MemoryBudget,
    predicted_peak_bytes: u64,
) -> WorkerResult<gen_core::MemoryRunContext> {
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "InstantID memory contract declares no calibration identity".to_owned(),
        )
    })?;
    Ok(gen_core::MemoryRunContext {
        selection: gen_core::MemorySelection {
            strategy: gen_core::MemoryStrategy::Resident,
            parameters: Default::default(),
            tier,
        },
        optimization_authority: gen_core::MemoryOptimizationAuthority::Resident,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: gen_core::MemoryMode::Other("character_image".to_owned()),
        has_reference: true,
        use_pid: identity.use_pid,
        has_phases: identity.face_restore,
        geometry: gen_core::MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 1,
        },
        overlay: Some(identity.overlay_key()),
        budget,
        predicted_peak_bytes,
        cache_state: gen_core::MemoryCacheState::Cold,
        evidence_revision: instantid_memory::REQUEST_EVIDENCE_REVISION.to_owned(),
    })
}

/// Fail-closed pre-load validation of the admission context, on the ASYNC side — so a refusal is a
/// typed [`WorkerError::InvalidPayload`] raised before any weights are touched, rather than a
/// stringified engine error from inside the blocking load. `load_with_memory_context` re-validates
/// the same context inside the load (defense in depth); this call is what makes the refusal typed
/// and pre-load. There is deliberately NO fallback to the unadmitted `InstantId::load` path.
#[cfg(target_os = "macos")]
fn instantid_validate_admission(
    contract: &gen_core::MemoryProviderContract,
    tier: gen_core::MemoryNumericTier,
    identity: &InstantIdMemoryIdentity,
    context: &gen_core::MemoryRunContext,
) -> WorkerResult<()> {
    instantid_memory::validate_context(contract, tier, identity, context)
        .map_err(|error| WorkerError::InvalidPayload(format!("InstantID memory admission: {error}")))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn instantid_validate_admission(
    contract: &gen_core::MemoryProviderContract,
    tier: gen_core::MemoryNumericTier,
    identity: &InstantIdMemoryIdentity,
    context: &gen_core::MemoryRunContext,
) -> WorkerResult<()> {
    let _ = tier;
    instantid_memory::validate_context(contract, identity, context)
        .map_err(|error| WorkerError::InvalidPayload(format!("InstantID memory admission: {error}")))
}

/// The provider contract for this lane (MLX validates the request tier, candle is dense-only).
#[cfg(target_os = "macos")]
fn instantid_provider_contract(
    tier: gen_core::MemoryNumericTier,
) -> gen_core::MemoryProviderContract {
    instantid_memory::provider_contract(tier)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn instantid_provider_contract(
    tier: gen_core::MemoryNumericTier,
) -> gen_core::MemoryProviderContract {
    let _ = tier;
    instantid_memory::provider_contract()
}

/// The provider route this job's iteration mode drives. The route is an admission IDENTITY axis
/// (it lands in `overlay_key()`), so an angle set can never be priced by identity-mode evidence.
fn instantid_memory_route(mode: &InstantIdMode) -> InstantIdRoute {
    match mode {
        InstantIdMode::Identity => InstantIdRoute::Identity,
        InstantIdMode::AngleSet => InstantIdRoute::Angle,
        InstantIdMode::PoseSet(_) => InstantIdRoute::Pose,
    }
}

/// How an InstantID character job batches its iterations (torch-parity precedence: a pose set
/// wins over an angle set, which wins over plain identity — `instantid_adapter.py:655`).
enum InstantIdMode {
    /// `count` images at the reference's natural head pose (engine `generate`, W×H letterboxed).
    Identity,
    /// The 11-view Character-Studio set, shared seed (engine `generate_with_kps` from the
    /// worker-owned [`INSTANTID_ANGLE_KPS`] presets, square).
    AngleSet,
    /// `n` pose-library poses, shared seed — MultiControlNet IdentityNet + OpenPose (engine
    /// `generate_pose`, square).
    PoseSet(usize),
}

/// The 11-view Character-Studio angle set flag.
fn instantid_angle_set(request: &ImageRequest) -> bool {
    advanced::flag(&request.advanced, "angleSet")
}

/// Classify the InstantID iteration mode (pose set > angle set > plain identity).
fn instantid_mode(request: &ImageRequest) -> InstantIdMode {
    let poses = pose_entries(request).len();
    if poses > 0 {
        InstantIdMode::PoseSet(poses)
    } else if instantid_angle_set(request) {
        InstantIdMode::AngleSet
    } else {
        InstantIdMode::Identity
    }
}

/// Per-image InstantID action (the engine entry point this iteration calls). `Send` (it is moved
/// into the blocking task): `BodyPoint = Option<(f64, f64)>`, `&'static str`, and the unit variant
/// are all `Send`.
enum InstantIdAction {
    /// `generate` — the reference's natural head pose, W×H letterboxed.
    Identity,
    /// `generate_with_kps` — a Character-Studio view from worker-owned landmark presets (square).
    /// Carries the normalized 5-point kps directly (sc-4424) rather than an angle name, so the
    /// worker owns the framing presets and arbitrary/user-defined kps flow through the same path.
    Angle([(f32, f32); 5]),
    /// `generate_pose` — MultiControlNet IdentityNet + OpenPose on these COCO-18 keypoints (square).
    Pose(Vec<BodyPoint>),
}

/// Bridge the worker's gallery-normalized keypoints (`openpose_skeleton::Keypoint = Option<(f32,
/// f32)>`) to the engine's `BodyPoint = Option<(f64, f64)>`. `parse_poses` already applied the
/// COCO-18 normalize + conf<=0 drop, so this is just the f32→f64 widening.
fn pose_to_body_points(keypoints: &[crate::openpose_skeleton::Keypoint]) -> Vec<BodyPoint> {
    keypoints
        .iter()
        .map(|point| point.map(|(x, y)| (x as f64, y as f64)))
        .collect()
}

/// Resolve the RealVisXL (SDXL) base snapshot for InstantID: an explicit `modelPath` dir
/// (advanced or manifest) wins, else the selected quant tier subdir of the HF cache snapshot for the
/// manifest `repo` (default `SceneWorks/realvisxl-mlx`). The big base is staged by the normal
/// model-download flow; `None` here means it is not present, so the native lane refuses the job and
/// no fallback is attempted. An explicit `modelPath` override loads verbatim (no tier resolution) —
/// it is a fully-assembled diffusers dir the caller vouches for.
fn resolve_instantid_sdxl_base(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    if let Some(path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
    {
        return resolve_app_managed_model_dir(settings, &path, "InstantID SDXL modelPath")
            .map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(INSTANTID_SDXL_REPO);
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo)
        .map(|root| instantid_tier_subdir(&root, request)))
}

/// Pick the engine-complete tier subdir of the `SceneWorks/realvisxl-mlx` turnkey `root` for the
/// InstantID SDXL backbone (sc-9965, epic 8506). The turnkey ships self-contained `bf16/` (dense —
/// the default, the validated fp16 identity envelope: ArcFace-cosine ~0.82 @1024²) + packed `q8/` /
/// `q4/` subdirs, each a complete diffusers tree (`unet/` + `text_encoder{,_2}/` + `vae/` +
/// tokenizer(s) + scheduler + model_index.json).
///
/// Tier selection mirrors [`instantid_quant`]'s bit mapping so the resolved tier and the applied
/// load-quant agree (the load-time `.quantize()` no-ops on an already-packed base): `Some(4)` →
/// `q4/`, `Some(8)` → `q8/`, `None` (the default, `mlxQuantize` unset / `<=0`) → `bf16/`. The candle
/// InstantID lane runs dense f16 with no packed path (`candle-gen-sdxl::load_instantid_unet` reads
/// the dense `unet/diffusion_pytorch_model.fp16.safetensors`, no `.scales` detect) and the worker
/// already forces `recipe_bits -> None` there, so it always loads `bf16/`.
///
/// Falls back preferred → `bf16` → `q4` → `q8` → `root` so a partially-downloaded turnkey surfaces
/// as a load error rather than a silent half-load — the same philosophy as
/// [`standard_tier_subdir`]. Tier presence is filename-agnostic (a `unet/` `*.safetensors`, packed
/// single-file or dense), so it holds for every tier regardless of the packed backbone filename.
fn instantid_tier_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    // Which tier the request wants. Only the MLX lane honors the packed Q4/Q8 tiers; candle (and the
    // no-face build) is dense-only, so `preferred` is `bf16` there.
    #[cfg(target_os = "macos")]
    let preferred = match instantid_quant(request).0 {
        Some(4) => "q4",
        Some(8) => "q8",
        _ => "bf16",
    };
    #[cfg(not(target_os = "macos"))]
    let preferred = {
        let _ = request;
        "bf16"
    };
    // A tier is "present" when its `unet/` backbone holds any `*.safetensors` (packed single-file OR
    // a dense `*.fp16.safetensors`). InstantID is always an SDXL turnkey — the backbone lives under
    // `unet/`, never `transformer/` — so this one-component probe covers every tier.
    let present = |name: &str| -> Option<PathBuf> {
        let unet = root.join(name).join("unet");
        // A hidden `._*.safetensors` AppleDouble sidecar is not a backbone (SceneWorks#1333).
        let has_backbone = std::fs::read_dir(&unet)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| !sceneworks_core::lora_family::is_hidden_file(&entry.path()))
            .any(|entry| entry.file_name().to_string_lossy().ends_with(".safetensors"));
        has_backbone.then(|| root.join(name))
    };
    // sc-12279 generalized: the `SceneWorks/realvisxl-mlx` base ships a per-tier `model_index.json`, so
    // route the chain through the shared completeness guard — a torn tier (e.g. `unet/` present but
    // `tokenizer/` missing) now falls through to a complete sibling instead of crashing the SDXL loader
    // on the absent tokenizer. `instantid.rs` is `include!`d into the `image_jobs` module, so
    // `pick_loadable_tier` / `tier_components_present` (base.rs) are directly in scope.
    pick_loadable_tier(&[preferred, "bf16", "q8", "q4"], &present, &tier_components_present)
        .unwrap_or_else(|| root.to_path_buf())
}

/// True when this is a native-MLX-eligible InstantID job: the production model in
/// `character_image` mode with a reference face whose SDXL base resolves locally. ALL InstantID
/// modes are now native (sc-3345 identity + angle set; sc-3381 pose mode + face-restore via the
/// #193 engine). Mirrors `jobs_store::instantid_mlx_eligible` so the worker and the router agree.
fn instantid_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == INSTANTID_MODEL
        && request.mode == "character_image"
        && non_empty(&request.reference_asset_id)
        && matches!(resolve_instantid_sdxl_base(request, settings), Ok(Some(_)))
}

/// The number of images an InstantID job produces: `n` for a pose set, the active angle
/// collection's length for an angle set (sc-4450 — variable N, not fixed 11), else `request.count`.
fn instantid_image_count(request: &ImageRequest, settings: &Settings) -> u32 {
    match instantid_mode(request) {
        InstantIdMode::PoseSet(count) => count as u32,
        InstantIdMode::AngleSet => active_angle_collection(request, settings).1.len() as u32,
        InstantIdMode::Identity => request.count,
    }
}

/// Resolve the active angle-set collection for this job (sc-4450): the per-generation override
/// (`advanced.keypointCollectionId`) → the user default → the built-in 11. Built-in fallback on
/// any store error so angle generation never hard-fails on a Key Point Library hiccup.
///
/// ACCEPTED TRADEOFF (sc-8953 / F-151): for an angle-set job this runs its `ProjectStore` lookup
/// twice — once via `instantid_image_count` at plan time and once here at generation time. It is a
/// single small indexed read against the local SQLite store, negligible next to the generation, so
/// the duplicate is left as-is; the fix (resolve once and thread the collection through the plan)
/// is deferred until it matters.
fn active_angle_collection(
    request: &ImageRequest,
    settings: &Settings,
) -> (
    String,
    Vec<sceneworks_core::project_store::ResolvedAnglePreset>,
) {
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let override_id = advanced::str(&request.advanced, "keypointCollectionId", "");
    let override_id = override_id.trim();
    let override_id = (!override_id.is_empty()).then_some(override_id);
    store
        .resolve_angle_collection(override_id)
        .unwrap_or_else(|_| {
            (
                sceneworks_core::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID.to_owned(),
                builtin_angle_presets(),
            )
        })
}

/// The built-in 11 as resolved angle presets (the worker-side fallback when the store is
/// unreachable, sc-4450).
fn builtin_angle_presets() -> Vec<sceneworks_core::project_store::ResolvedAnglePreset> {
    use sceneworks_core::{angle_kps, project_store::ResolvedAnglePreset};
    angle_kps::BUILTIN_ANGLE_KPS
        .iter()
        .map(|(angle, kps)| ResolvedAnglePreset {
            preset_id: angle_kps::builtin_preset_id(angle),
            name: angle_kps::builtin_angle_display_name(angle),
            angle: Some((*angle).to_owned()),
            kps: *kps,
        })
        .collect()
}

/// Resolve InstantID denoise steps: `advanced.steps` (clamped 1..=80) → manifest `steps` →
/// the torch-parity default (30).
fn instantid_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", INSTANTID_DEFAULT_STEPS, 1..=80)
}

/// Resolve InstantID guidance: `advanced.guidanceScale` → manifest `guidanceScale` → the
/// RealVisXL-tuned default (3.0). Clamped to a sane CFG range.
fn instantid_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        INSTANTID_DEFAULT_GUIDANCE,
        0.0..=30.0,
    )
}

/// Resolve InstantID quantization. **fp16 (dense) is the default** — the validated identity
/// envelope (ArcFace-cosine 0.82 @1024²); Q8/Q4 only on an explicit `advanced.mlxQuantize` /
/// manifest opt-in (identity drops to ~0.64 @512² and full-res quant is unvalidated). Returns
/// the engine `bits` (`Some(4)`/`Some(8)`/`None`) + the recipe bit count.
fn instantid_quant(request: &ImageRequest) -> (Option<i32>, Option<i64>) {
    let raw = request
        .advanced
        .get("mlxQuantize")
        .and_then(quant_int)
        .or_else(|| {
            request
                .model_manifest_entry
                .get("mlx")
                .and_then(|mlx| mlx.get("quantize"))
                .and_then(quant_int)
        });
    match raw {
        Some(bits) if bits > 0 && bits <= 4 => (Some(4), Some(4)),
        Some(bits) if bits > 4 => (Some(8), Some(8)),
        // None / 0 / negative → fp16 (the default + the validated InstantID envelope).
        _ => (None, None),
    }
}

/// Flat telemetry recorded on InstantID assets (parity with `mlx_raw_settings` + the torch
/// `InstantIDAdapter` recipe keys).
#[allow(clippy::too_many_arguments)]
fn instantid_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: f32,
    ip_scale: f32,
    controlnet_scale: f32,
    angle_set: bool,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("ipAdapterScale".to_owned(), json!(ip_scale));
    raw.insert(
        "controlnetConditioningScale".to_owned(),
        json!(controlnet_scale),
    );
    raw.insert(
        "instantIdEngine".to_owned(),
        Value::String(INSTANTID_ENGINE.to_owned()),
    );
    if angle_set {
        raw.insert("angleSet".to_owned(), Value::Bool(true));
    }
    raw
}

/// The pinned commit revision for each InstantID download repo (sc-9879, F-077 follow-up). Every
/// InstantID weight repo is a fixed, non-overridable const (the env pins point at local snapshot dirs,
/// never another HF repo), so fetching the mutable `main` branch means a re-push (or a compromised token)
/// could silently swap the weights we load. Pin each to its exact commit for defense-in-depth (mirrors
/// sc-8879/sc-9682); the `lfs.oid` sha256 verify in `ensure_hf_cached_file` is retained. Any repo not in
/// this table (a caller passing something unexpected) falls back to `main` rather than a wrong sha.
fn instantid_revision(repo: &str) -> &'static str {
    match repo {
        INSTANTID_MLX_REPO => INSTANTID_MLX_REVISION,
        INSTANTID_CONTROLNET_REPO => INSTANTID_CONTROLNET_REVISION,
        INSTANTID_OPENPOSE_REPO => INSTANTID_OPENPOSE_REVISION,
        // The MLX PuLID lane (`pulid.rs`) fetches its adapter + EVA/BiSeNet bundle through
        // `ensure_instantid_file`, so pin those two repos here too (sc-11168 / F-007). Matched as string
        // literals because `pulid.rs`'s `PULID_ADAPTER_REPO` / `PULID_MLX_REPO` consts are macOS-only.
        "guozinan/PuLID" => PULID_ADAPTER_REVISION,
        "SceneWorks/pulid-flux-mlx" => PULID_MLX_REVISION,
        _ => "main",
    }
}

/// Resolve a single InstantID weight file: return it if already present in `dir`, else
/// download `url` into `dir` (atomic `.tmp` + rename, so a partial download is never mistaken
/// for a complete one).
async fn ensure_instantid_file(
    context: &DownloadContext<'_>,
    repo: &str,
    dir: &Path,
    name: &str,
) -> WorkerResult<PathBuf> {
    ensure_hf_cached_file(context, repo, instantid_revision(repo), name, &dir.join(name)).await
}

/// Resolve only the SCRFD detector weights (`scrfd_10g.safetensors`) from the same converted
/// bundle InstantID uses — for the standalone kps-extraction capability (sc-4433), which needs
/// face detection but neither ArcFace nor the SDXL/IdentityNet stack. Shares the env override
/// (`SCENEWORKS_INSTANTID_WEIGHTS`) + app cache + download-on-first-use with
/// [`ensure_instantid_weights`], so a prior InstantID run leaves it already cached.
// The standalone kps-extraction capability is a macOS path; the candle lane only loads SCRFD via the
// InstantID face stack (`with_face`), so this helper is unused off-Mac — allow it dead there.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) async fn ensure_scrfd_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PathBuf> {
    let client = crate::downloads::streaming_download_client();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "KPS extraction canceled while fetching SCRFD weights.",
        fresh_download: false,
    };
    let bundle_dir = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| settings.data_dir.join("cache").join("instantid-mlx"));
    ensure_instantid_file(
        &context,
        INSTANTID_MLX_REPO,
        &bundle_dir,
        INSTANTID_SCRFD_FILE,
    )
    .await
}

/// Resolve the face-stack DIRECTORY (`scrfd_10g.safetensors` and `arcface_iresnet100.safetensors`),
/// staging BOTH files. The off-Mac kps-extraction capability (sc-5497, epic 5482) loads them through
/// `runtime_cuda::providers::face::load`, and the Dataset Doctor face pass (sc-6538) loads them on BOTH backends:
/// candle from the dir, MLX by joining the two canonical file names ([`INSTANTID_SCRFD_FILE`],
/// [`INSTANTID_ARCFACE_FILE`]) — so it is no longer candle-only. (The Mac kps path still loads SCRFD
/// alone via [`ensure_scrfd_weights`].) Shares the env override (`SCENEWORKS_INSTANTID_WEIGHTS`), the
/// app cache, and download-on-first-use with [`ensure_instantid_weights`], so a prior
/// InstantID/PuLID/extraction run leaves it already cached. Returns the bundle dir (which IS the candle
/// face stack's load dir, exactly the `face_dir` the candle InstantID path resolves).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn ensure_face_stack_dir(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PathBuf> {
    let client = crate::downloads::streaming_download_client();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "Canceled while fetching face-stack weights.",
        fresh_download: false,
    };
    let bundle_dir = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| settings.data_dir.join("cache").join("instantid-mlx"));
    ensure_instantid_file(&context, INSTANTID_MLX_REPO, &bundle_dir, INSTANTID_SCRFD_FILE).await?;
    ensure_instantid_file(
        &context,
        INSTANTID_MLX_REPO,
        &bundle_dir,
        INSTANTID_ARCFACE_FILE,
    )
    .await?;
    Ok(bundle_dir)
}

/// Resolve all InstantID weight inputs, downloading the small converted bundle + the stock
/// IdentityNet on first use. Returns `(identitynet_dir, ip_adapter, scrfd, arcface)` — all
/// `Send` paths; the `!Send` MLX load happens on the blocking thread. Resolution order favours
/// an env override / the HF cache before any network fetch.
async fn ensure_instantid_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<(WeightsSource, PathBuf, PathBuf, PathBuf)> {
    let client = crate::downloads::streaming_download_client();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "InstantID generation canceled while fetching weights.",
        fresh_download: false,
    };

    // Converted bundle (ip-adapter + face stack): an env-pinned dir (pre-staged for local
    // validation) wins, else the app cache (download missing files from SceneWorks/instantid-mlx).
    let bundle_dir = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| settings.data_dir.join("cache").join("instantid-mlx"));
    let ip_adapter = ensure_instantid_file(
        &context,
        INSTANTID_MLX_REPO,
        &bundle_dir,
        INSTANTID_IP_ADAPTER_FILE,
    )
    .await?;
    let scrfd = ensure_instantid_file(
        &context,
        INSTANTID_MLX_REPO,
        &bundle_dir,
        INSTANTID_SCRFD_FILE,
    )
    .await?;
    let arcface = ensure_instantid_file(
        &context,
        INSTANTID_MLX_REPO,
        &bundle_dir,
        INSTANTID_ARCFACE_FILE,
    )
    .await?;

    // IdentityNet (stock InstantX ControlNetModel): env override → HF cache snapshot →
    // download the two files into the app cache.
    if let Ok(dir) = std::env::var("SCENEWORKS_INSTANTID_CONTROLNET") {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            return Ok((WeightsSource::Dir(dir), ip_adapter, scrfd, arcface));
        }
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, INSTANTID_CONTROLNET_REPO)
    {
        let controlnet = snapshot.join("ControlNetModel");
        if controlnet
            .join("diffusion_pytorch_model.safetensors")
            .exists()
        {
            return Ok((WeightsSource::Dir(controlnet), ip_adapter, scrfd, arcface));
        }
    }
    let controlnet_dir = settings.data_dir.join("cache").join("instantid-controlnet");
    for file in INSTANTID_CONTROLNET_FILES {
        let source = format!("ControlNetModel/{file}");
        ensure_hf_cached_file(
            &context,
            INSTANTID_CONTROLNET_REPO,
            INSTANTID_CONTROLNET_REVISION,
            &source,
            &controlnet_dir.join(file),
        )
        .await?;
    }
    Ok((
        WeightsSource::Dir(controlnet_dir),
        ip_adapter,
        scrfd,
        arcface,
    ))
}

/// Resolve the xinsir OpenPose-SDXL ControlNet dir for pose mode: env override
/// (`SCENEWORKS_INSTANTID_OPENPOSE`) → HF cache snapshot → download the two files on first use. A
/// stock diffusers SDXL ControlNet (loads via `with_openpose`/`load_controlnet`, no conversion).
async fn ensure_instantid_openpose(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<WeightsSource> {
    if let Ok(dir) = std::env::var("SCENEWORKS_INSTANTID_OPENPOSE") {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            return Ok(WeightsSource::Dir(dir));
        }
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, INSTANTID_OPENPOSE_REPO) {
        if snapshot
            .join("diffusion_pytorch_model.safetensors")
            .exists()
        {
            return Ok(WeightsSource::Dir(snapshot));
        }
    }
    let client = crate::downloads::streaming_download_client();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "InstantID generation canceled while fetching OpenPose weights.",
        fresh_download: false,
    };
    let dir = settings.data_dir.join("cache").join("instantid-openpose");
    for file in INSTANTID_CONTROLNET_FILES {
        ensure_instantid_file(&context, INSTANTID_OPENPOSE_REPO, &dir, file).await?;
    }
    Ok(WeightsSource::Dir(dir))
}

/// Real InstantID generation: resolve the reference + weights on the async side, then load the
/// bespoke `InstantId` provider once + generate each image on the blocking thread (the MLX
/// model is `!Send`). Three modes (torch parity): single identity (`generate`), the 11-view angle
/// set (`generate_with_kps` from the worker-owned [`INSTANTID_ANGLE_KPS`] presets — sc-4424), and
/// the pose-library set (`generate_pose`, MultiControlNet IdentityNet with xinsir OpenPose —
/// sc-3117). `advanced.faceRestore` adds the ADetailer-style re-render pass (`restore_face`,
/// sc-3380) on each output. The engine `generate*` take the per-job `CancelFlag` (via
/// `InstantIdRequest.cancel`) and a `Progress` callback (sc-4380/sc-4382), so streaming is
/// per-step (`Step`/`Decoding` events) and cancellation is honoured mid-denoise — same contract
/// as the registry families. Reuses [`consume_gen_events`] for the asset writes.
async fn generate_instantid_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let sdxl_base = resolve_instantid_sdxl_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("InstantID base (RealVisXL) not found".to_owned())
    })?;
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("InstantID requires a reference face image".to_owned())
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;

    let (controlnet, ip_adapter, scrfd_path, arcface_path) =
        ensure_instantid_weights(api, settings, job).await?;

    // User style/character LoRAs (sc-6038). InstantID is a stock SDXL (RealVisXL) UNet, so SDXL
    // adapters apply on top of IdentityNet + the identity IP-Adapter — and the manifest advertises
    // `families:[sdxl]`, so the picker offers them. Resolved + path-confined exactly like every other
    // SDXL-family path (base.rs/sdxl.rs); merged onto the UNet by the engine `InstantIdPaths.adapters`
    // seam. Shared across all three modes (identity / angle set / pose) since they share the one load.
    let adapters = resolve_adapters(request, settings)?;
    let adapter_count = adapters.len();

    let steps = instantid_steps(request);
    let guidance = instantid_guidance(request);
    let (quant_bits, recipe_bits) = instantid_quant(request);
    // The candle InstantID stack runs dense f16 — there is no quantized path. Ignore the MLX quant
    // knob entirely on this lane: don't apply it (the `quantize` step in the load closure is macOS-
    // only) and don't record it as applied (`recipe_bits` -> None). `let _` consumes the otherwise-
    // unused `quant_bits`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let recipe_bits: Option<i64> = {
        let _ = (quant_bits, recipe_bits);
        None
    };
    let ip_scale = advanced::f32_clamped(
        &request.advanced,
        "ipAdapterScale",
        INSTANTID_IP_SCALE,
        0.0..=1.0,
    );
    let mode = instantid_mode(request);
    let angle_set = matches!(mode, InstantIdMode::AngleSet);
    let pose_set = matches!(mode, InstantIdMode::PoseSet(_));
    // The angle set runs a softer landmark lock by default (sc-8354) so the off-axis views clear the
    // blur floor; identity/pose modes keep the standard 0.80. An explicit slider value (now surfaced
    // on the Angle Set card) still wins via the same `controlnetConditioningScale` key.
    let controlnet_default = if angle_set {
        INSTANTID_ANGLE_CONTROLNET_SCALE
    } else {
        INSTANTID_CONTROLNET_SCALE
    };
    let controlnet_scale = advanced::f32_clamped(
        &request.advanced,
        "controlnetConditioningScale",
        controlnet_default,
        0.0..=2.0,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(INSTANTID_SDXL_REPO)
        .to_owned();
    // The active Key Point Library collection drives the angle set (sc-4450): per-generation
    // override > user default > built-in 11. Resolved once (and only for angle jobs).
    let angle_collection = angle_set.then(|| active_angle_collection(request, settings));
    let openpose_scale = advanced::f32_clamped(
        &request.advanced,
        "openPoseScale",
        INSTANTID_OPENPOSE_SCALE,
        0.0..=2.0,
    );
    let face_restore = advanced::flag(&request.advanced, "faceRestore");
    // PiD decoder overlay (epic 7840, sc-8371 mlx / sc-8373 candle): decode Angles/Poses through the
    // `sdxl` PiD super-resolving student (2K/4K) instead of the SDXL VAE when the request opted in
    // (`advanced.usePid`) AND the checkpoint + Gemma snapshots are cached. `resolve_pid_weights`
    // returns `None` for a non-eligible model / missing opt-in / absent snapshots → native VAE.
    // InstantID composes the SDXL VAE, so BOTH the mlx and candle engines share the one `sdxl` student
    // via `with_pid` — the candle lane (sc-8373) now carries the same `InstantId::with_pid` +
    // `InstantIdRequest.use_pid` seam as macOS, so this resolves on either face backend.
    #[cfg(any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle")))]
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, &request.model)?;
    #[cfg(any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle")))]
    let use_pid = pid_weights.is_some();
    // Load the xinsir OpenPose ControlNet only for pose mode (it is the MultiControlNet second
    // branch; identity/angle modes don't need it).
    let openpose = if pose_set {
        Some(ensure_instantid_openpose(api, settings, job).await?)
    } else {
        None
    };

    let mut raw_settings = instantid_raw_settings(
        request,
        &repo,
        steps,
        recipe_bits,
        guidance,
        ip_scale,
        controlnet_scale,
        angle_set,
    );
    if pose_set {
        raw_settings.insert("poseLibrary".to_owned(), Value::Bool(true));
        raw_settings.insert("openPoseScale".to_owned(), json!(openpose_scale));
    }
    if face_restore {
        raw_settings.insert("faceRestore".to_owned(), Value::Bool(true));
    }
    // Mark PiD output on the asset sidecar (epic 7840): the NSCLv1 non-commercial restriction flows
    // to PiD-decoded output, distinct from the rest of the pipeline. Only set when PiD actually ran.
    #[cfg(any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle")))]
    if use_pid {
        raw_settings.insert("usePid".to_owned(), Value::Bool(true));
    }
    // Record how many user LoRAs were merged onto the SDXL UNet (sc-6038) so the asset sidecar shows
    // the adapters were applied (they previously rode the request but were silently dropped).
    if adapter_count > 0 {
        raw_settings.insert("appliedLoraCount".to_owned(), json!(adapter_count));
    }
    // Record which collection + ordered presets produced the set, so each asset (by index) maps
    // back to the preset that rendered it (sc-4450).
    if let Some((collection_id, presets)) = &angle_collection {
        raw_settings.insert("keypointCollectionId".to_owned(), json!(collection_id));
        raw_settings.insert(
            "anglePresetIds".to_owned(),
            json!(presets
                .iter()
                .map(|preset| preset.preset_id.clone())
                .collect::<Vec<_>>()),
        );
    }

    // Per-image work items: (seed, prompt, action). Pose + angle sets share one seed (only the
    // pose changes across the set — noise-derived attributes stay constant); single identity is
    // per-seed at the reference's natural pose.
    //
    // PiD output tier (sc-10054): when PiD runs, 2K caps the effective base so its fixed 4× lands on
    // ~2048 (default 4K/native leaves the requested dims untouched). The skeleton/keypoints render at
    // this same base, so control + latent stay aligned. Gated to the PiD-capable face backends (where
    // `use_pid` + the helpers exist); the no-face build keeps the requested dims verbatim.
    #[cfg(any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle")))]
    let (width, height) =
        pid_effective_dims(request.width, request.height, use_pid, pid_output_tier(request));
    #[cfg(not(any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle"))))]
    let (width, height) = (request.width, request.height);
    let work: Vec<(i64, String, InstantIdAction)> = match &mode {
        InstantIdMode::PoseSet(_) => {
            let set_seed = resolve_seed(request, 0);
            parse_poses(request)
                .into_iter()
                .map(|pose| {
                    (
                        set_seed,
                        request.prompt.clone(),
                        InstantIdAction::Pose(pose_to_body_points(&pose.keypoints)),
                    )
                })
                .collect()
        }
        InstantIdMode::AngleSet => {
            let set_seed = resolve_seed(request, 0);
            // One image per preset in the active collection's order (sc-4450). Built-in presets
            // carry their canonical angle so the prompt still gets the per-angle clause; custom
            // presets render to their kps with the base prompt.
            let presets = angle_collection
                .as_ref()
                .map(|(_, presets)| presets.clone())
                .unwrap_or_else(builtin_angle_presets);
            presets
                .into_iter()
                .map(|preset| {
                    let prompt = match &preset.angle {
                        Some(angle) => augment_prompt_for_angle(&request.prompt, angle),
                        None => request.prompt.clone(),
                    };
                    (set_seed, prompt, InstantIdAction::Angle(preset.kps))
                })
                .collect()
        }
        InstantIdMode::Identity => (0..request.count as usize)
            .map(|index| {
                (
                    resolve_seed(request, index),
                    request.prompt.clone(),
                    InstantIdAction::Identity,
                )
            })
            .collect(),
    };
    let total = work.len();

    // Curated unified-sampler selection (epic 7114, sc-7432). InstantID builds its bespoke request
    // OUTSIDE base.rs's generic plumbing, so read the per-generation knob here and N3-normalize it
    // against the shared curated menu both engines honor — mlx #538 / candle #130 route a curated
    // solver/scheduler through the additive `denoise_curated` path; an unknown name drops back to the
    // engine default + emits an event rather than hard-failing `validate_request`. N1: with neither set
    // the request carries `None` ⇒ the bespoke ancestral default loop runs byte-for-byte unchanged.
    let (curated_samplers, curated_schedulers) = curated_image_menu();
    let (sampler, scheduler, _shift) = read_advanced_sampling_knobs(&request.advanced);
    let sampler = normalize_sampling_knob(
        sampler,
        &curated_samplers,
        "sampler",
        &request.model,
        &job.id,
        backend,
    );
    let scheduler = normalize_sampling_knob(
        scheduler,
        &curated_schedulers,
        "scheduler",
        &request.model,
        &job.id,
        backend,
    );

    let negative_prompt = request.negative_prompt.clone();
    // The reference asset id is the `sourceAssetId` recorded on each `faceLikeness` block
    // (sc-4408/sc-4409/sc-4410/sc-4411) so a consumer can attribute the score back to the source
    // identity face. Captured as an owned String for the blocking closure. Consumed on every InstantID
    // mode now (single identity / angle / pose all score, sc-4411); allow it unused on the
    // non-face-backend build.
    #[cfg_attr(
        not(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        )),
        allow(unused_variables)
    )]
    let face_likeness_source_ref = reference_id.to_owned();
    // InstantID reuses the candle SDXL conditioner + VAE (candle-gen-instantid, sc-13663), so it stages
    // the SAME three SDXL components (epic 13657, sc-13682): CLIP-L/bigG tokenizers + fp16-fix VAE.
    // Resolved before the blocking closure (a missing one fails fast) and moved in. `InstantIdPaths`
    // carries these fields ONLY on the candle build — the macOS MLX InstantID lane is self-contained — so
    // both the resolve and the struct fields are candle-gated.
    // sc-13739: the candle `InstantIdPaths` no longer takes the three SDXL components as flat fields —
    // it takes an `SdxlComponents` built from a `LoadSpec` (the same load-time gate the SDXL engine
    // runs). Resolve the three from the manifest coRequisites, stage them under their component ids, and
    // build the gated `SdxlComponents` here; a missing/misconfigured one fails the job before the load.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let sdxl = {
        let (tokenizer_clip_l, tokenizer_clip_bigg, vae_fp16_fix) =
            resolve_sdxl_components(&request.model_manifest_entry, settings)?;
        SdxlComponents::from_spec(
            &gen_core::LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::new()))
                .with_component(COMPONENT_TOKENIZER_CLIP_L, tokenizer_clip_l)
                .with_component(COMPONENT_TOKENIZER_CLIP_BIGG, tokenizer_clip_bigg)
                .with_component(COMPONENT_VAE_FP16_FIX, vae_fp16_fix),
        )
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("InstantID SDXL components: {error}"))
        })?
    };
    // Conditioning-overlay VRAM admission (sc-16069, epic 15448) — candle only: the macOS MLX InstantID
    // lane loads a registered generator and is already admitted by
    // `mlx_fit_gate::apply_residency_policy` in `generator_cache`. The candle lane is not: it is claimed
    // by `resolve_candle_image_route`'s FIRST arm (`instantid_realvisxl` is not an `is_candle_engine`
    // txt2img id) and loads a bespoke `InstantIdPaths` through the UNcached `start_gen_stream`, so before
    // this it allocated with no pre-flight check at all.
    //
    // InstantID's overlay is its whole identity stack: IdentityNet, the identity IP-Adapter, the SCRFD
    // detector + ArcFace embedder, the pose-mode OpenPose ControlNet branch, and an opted-in PiD decoder
    // pair. Priced before anything is moved into the load closure. (The `sdxl` components built above are
    // already borrowed by value into `SdxlComponents`, so the fp16-fix VAE is not re-summed here; it is a
    // few hundred MB against tens of GB, and a lower floor only ever admits.)
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        // The OpenPose branch resolves to a DIRECTORY, and on the warm path that directory is the whole
        // `xinsir/controlnet-openpose-sdxl-1.0` HF snapshot — which may carry checkpoints this render does
        // not load. Price the two files the lane actually names (`INSTANTID_CONTROLNET_FILES`) instead of
        // scanning the snapshot, so an alternate checkpoint sitting beside them cannot inflate the floor
        // into refusing a render that fits (sc-16069 review). A file that is absent contributes 0, so the
        // freshly-downloaded layout (where both files ARE the directory's contents) prices identically.
        let openpose_dir = openpose
            .as_ref()
            .map(crate::conditioning_fit::weights_source_path);
        let openpose_files: Vec<std::path::PathBuf> = openpose_dir
            .map(|dir| {
                INSTANTID_CONTROLNET_FILES
                    .iter()
                    .map(|file| dir.join(file))
                    .collect()
            })
            .unwrap_or_default();
        let mut overlays = vec![
            crate::conditioning_fit::weights_source_path(&controlnet),
            ip_adapter.as_path(),
            scrfd_path.as_path(),
            arcface_path.as_path(),
        ];
        overlays.extend(openpose_files.iter().map(std::path::PathBuf::as_path));
        overlays.extend(crate::conditioning_fit::pid_paths(pid_weights.as_ref()));
        admit_conditioning_paths(
            settings,
            "InstantID",
            "face-identity conditioning stack",
            &sdxl_base,
            &overlays,
        )
        .await?;
    }

    // ---- Request-scoped memory admission (sc-20799) --------------------------------------------
    // The bespoke InstantID admission surface, driven identically on both backends. Built from the
    // artifacts THIS job resolved (so the overlay key is bound to the exact artifact set) and
    // validated here, before any weight file is opened.
    let memory_route = instantid_memory_route(&mode);
    let artifact_entries = instantid_artifact_entries(
        &sdxl_base,
        &controlnet,
        &ip_adapter,
        &scrfd_path,
        &arcface_path,
        openpose.as_ref(),
        &adapters,
        pid_weights.as_ref(),
    );
    let memory_identity = InstantIdMemoryIdentity {
        route: memory_route,
        adapter_count,
        use_pid,
        face_restore,
        artifact_fingerprint: instantid_artifact_fingerprint(&artifact_entries),
    };
    // Predicted peak: the manifest's own measured figure when it declares one, else the priced
    // on-disk bytes of exactly the artifact set the overlay key names. No invented constant, and no
    // borrowed evidence from another provider.
    let (manifest_backend_key, manifest_tier_key) = instantid_memory_backend_keys(request);
    let predicted_peak_bytes = instantid_manifest_peak_gb(
        &request.model_manifest_entry,
        manifest_backend_key,
        manifest_tier_key,
    )
    .map(instantid_gib_bytes)
    .unwrap_or_else(|| instantid_artifact_bytes(&artifact_entries));
    drop(artifact_entries);
    let memory_tier = instantid_memory_tier(request);
    let memory_contract = instantid_provider_contract(memory_tier);
    let memory_context = instantid_memory_context(
        &memory_contract,
        memory_tier,
        &memory_identity,
        width,
        height,
        instantid_memory_budget(settings, predicted_peak_bytes).await,
        predicted_peak_bytes,
    )?;
    instantid_validate_admission(
        &memory_contract,
        memory_tier,
        &memory_identity,
        &memory_context,
    )?;
    // The load closure needs its own copies; the originals are moved into the per-item drive closure
    // below, which revalidates the SAME context against each request through `begin_memory_request`.
    let load_memory_identity = memory_identity.clone();
    let load_memory_context = memory_context.clone();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "instantid",
        adapter_count,
        move || {
            let paths = InstantIdPaths {
                sdxl_base,
                identitynet: controlnet,
                ip_adapter,
                // User LoRA/LoKr adapters (sc-6038), resolved above and merged onto the SDXL UNet by
                // both engine lanes (mlx-gen #477 / candle-gen #86 both carry the field; worker mlx
                // pin now 19d5522, candle pin c98609f). Populated for BOTH backends — superseding the
                // earlier candle-only `Vec::new()` stopgap from #730.
                adapters,
                // The caller-staged SDXL components as an `SdxlComponents` (candle only — see above).
                #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
                sdxl,
                // c6d prices the full Candle composition at load time. Seed the paths with the
                // already-resolved sources; the later builders still attach those same sources
                // so their load/reload behavior remains unchanged while the admission contract
                // sees the complete resident stack.
                #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
                openpose: openpose.clone(),
                #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
                face_dir: Some(
                    scrfd_path
                        .parent()
                        .unwrap_or(scrfd_path.as_path())
                        .to_path_buf(),
                ),
            };
            // Admitted load (sc-20799): the provider revalidates the exact route/composition/budget
            // handshake and RETAINS it, which is what makes `begin_memory_request` reachable per
            // item. There is no unadmitted fallback — a refusal here fails the job. The MLX entry
            // takes the numeric tier (its contract validates the tier request-by-request); the
            // candle entry is dense-only and takes none.
            #[cfg(target_os = "macos")]
            let model = InstantId::load_with_memory_context(
                &paths,
                memory_tier,
                load_memory_identity,
                load_memory_context,
            )
            .map_err(|error| WorkerError::Engine(format!("InstantID load failed: {error}")))?;
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            let model =
                InstantId::load_with_memory_context(&paths, load_memory_identity, load_memory_context)
                    .map_err(|error| {
                        WorkerError::Engine(format!("InstantID load failed: {error}"))
                    })?;
            // Attach OpenPose (pose mode) BEFORE quantize so it quantizes with the stack; quantize
            // before with_face (the engine's documented order). `with_openpose` is backend-neutral
            // (both engines take `&WeightsSource` and consume+return `self`).
            let model = match &openpose {
                Some(source) => model.with_openpose(source).map_err(|error| {
                    WorkerError::Engine(format!("InstantID OpenPose load failed: {error}"))
                })?,
                None => model,
            };
            // Quantization is an MLX-only knob — the candle InstantID stack runs dense f16 and has no
            // `quantize` method (the candle lane already forced `quant_bits` out, above).
            #[cfg(target_os = "macos")]
            let model = match quant_bits {
                Some(bits) => model.quantize(bits).map_err(|error| {
                    WorkerError::Engine(format!("InstantID quantize failed: {error}"))
                })?,
                None => model,
            };
            // Attach the SCRFD + ArcFace face stack. The MLX engine loads the two weight files
            // explicitly; the candle FaceEmbedder (sc-5490) loads the pair from THEIR DIRECTORY by the
            // canonical `scrfd_10g.safetensors` + `arcface_iresnet100.safetensors` names (exactly what
            // `ensure_instantid_weights` stages), so it takes the dir, not the two paths.
            #[cfg(target_os = "macos")]
            let model = {
                let scrfd = Weights::from_file(&scrfd_path).map_err(|error| {
                    WorkerError::Engine(format!("InstantID SCRFD weights {scrfd_path:?}: {error}"))
                })?;
                let arcface = Weights::from_file(&arcface_path).map_err(|error| {
                    WorkerError::Engine(format!(
                        "InstantID ArcFace weights {arcface_path:?}: {error}"
                    ))
                })?;
                model.with_face(&scrfd, &arcface).map_err(|error| {
                    WorkerError::Engine(format!("InstantID face stack: {error}"))
                })?
            };
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            let model = {
                let face_dir = scrfd_path.parent().unwrap_or(scrfd_path.as_path());
                // `arcface_path` is staged in the same dir; `with_face(dir)` resolves it by name.
                let _ = &arcface_path;
                model.with_face(face_dir).map_err(|error| {
                    WorkerError::Engine(format!("InstantID face stack: {error}"))
                })?
            };
            // Attach the optional PiD super-resolving decoder (epic 7840, sc-8371 mlx / sc-8373 candle):
            // the `sdxl` student InstantID reuses (it composes the SDXL VAE). `pid_weights` is `Some`
            // only when this generation opted in AND the snapshots are cached, so this is a no-op for a
            // native-VAE generation. Both face backends expose the same `with_pid(&PidWeights)` seam, so
            // one arm serves the mlx and candle lanes.
            // `mut` since the pinned inference revision: both engines' `largest_face` took `&self`
            // and now takes `&mut self` (candle-gen-instantid/src/model.rs:870,
            // mlx-gen-instantid/src/model.rs:830). It now brackets the detection in
            // `prepare_conditioning_phase()` / `release_conditioning_components()`, so the call
            // stages the conditioning components in and back out rather than assuming them
            // resident — inference epic sc-20762's staged memory lifecycle.
            #[cfg(any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle")))]
            let mut model = match &pid_weights {
                Some(pid) => model.with_pid(pid).map_err(|error| {
                    WorkerError::Engine(format!("InstantID PiD decoder load failed: {error}"))
                })?,
                None => model,
            };
            // Face-restore needs the reference identity embedding (imposed on the re-rendered crop).
            // Detect it once on the raw reference. The candle `largest_face` takes the neutral
            // `gen_core::Image`; the MLX engine takes raw RGB bytes + dims.
            let restore_embedding = if face_restore {
                #[cfg(target_os = "macos")]
                let embedding = model
                    .largest_face(
                        &reference.pixels,
                        reference.height as usize,
                        reference.width as usize,
                    )
                    .map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "InstantID face-restore reference: {error}"
                        ))
                    })?
                    .embedding;
                #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
                let embedding = model
                    .largest_face(&reference)
                    .map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "InstantID face-restore reference: {error}"
                        ))
                    })?
                    .embedding;
                Some(embedding)
            } else {
                None
            };
            // Identity-likeness scorer (epic 4406, sc-4409 angles / sc-4410 poses / sc-4411 plain
            // With-Character): for EVERY InstantID character_image job — single identity (the plain
            // With-Character generation, sc-4411), an angle set, OR a pose set — embed the source
            // identity face ONCE here and reuse it across every output image, through the SHARED
            // generator-agnostic seam (the same `build_face_likeness_scorer` the FLUX.2 / Qwen /
            // SenseNova lanes call). It loads a SEPARATE SCRFD + ArcFace stack from the engine's
            // `with_face`, but from the SAME staged antelopev2 bundle, so no extra weights. All three
            // modes carry the character `referenceAssetId` (InstantID requires a reference face), so the
            // scorer is always built — the plain identity case is the general With-Character scoring
            // sc-4411 wires. Construction is non-fatal (weights / no source face → `None` → scores
            // omitted; generation still renders).
            #[cfg(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            ))]
            let scorer: Option<crate::face_likeness::FaceLikenessScorer> = {
                let face_weights_dir = scrfd_path.parent().unwrap_or(scrfd_path.as_path());
                crate::face_likeness::build_face_likeness_scorer(face_weights_dir, &reference)
            };
            #[cfg(not(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            )))]
            let scorer: Option<()> = None;
            Ok((model, reference, restore_embedding, scorer))
        },
        move |(model, reference, restore_embedding, scorer), tx, cancel| {
            // The candle `generate*` / `restore_face` take `&mut self` (each call sets the face IP
            // tokens on the UNet before the denoise), so the per-item closure mutates `model`; the MLX
            // engine's are `&self`. Bind `mut` for the candle lane and allow the unused-mut on macOS.
            #[allow(unused_mut)]
            let mut model = model;
            // The scorer + its source-ref are moved into the per-item closure below; they live for
            // the whole set so the source identity is embedded exactly ONCE (at construction, above)
            // and reused across every angle.
            drive_gen_items_scored(
                tx,
                work,
                move |_index, (seed, prompt, action), _preview, on_progress| {
                    if cancel.is_cancelled() {
                        return Ok(None);
                    }
                    // Per-step progress → GenEvent::Step, so `consume_gen_events` streams step
                    // updates, fires `image_inference_start`, and polls the cancel API (sc-4382 —
                    // without Step events an InstantID job could never be cancelled).
                    // Angle + pose sets use a square canvas (the engine forces `req.height =
                    // req.width` for the canonical landmark/skeleton — the sc-2009 kps-aspect rule);
                    // single identity keeps the requested W×H (the engine letterboxes the reference).
                    let req = InstantIdRequest {
                        prompt,
                        negative: negative_prompt.clone(),
                        width,
                        height,
                        steps: steps as usize,
                        guidance,
                        ip_adapter_scale: ip_scale,
                        controlnet_scale,
                        openpose_scale,
                        seed: seed as u64,
                        // PiD opt-in (sc-8371 mlx / sc-8373 candle): both engines' `InstantIdRequest`
                        // carry this field; the engine errors if set without a `with_pid` load, so the
                        // two stay in lockstep — `use_pid` is `pid_weights.is_some()`.
                        #[cfg(any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle")))]
                        use_pid,
                        sampler: sampler.clone(),
                        scheduler: scheduler.clone(),
                        cancel: cancel.clone(),
                    };
                    // ONE request scope per generated item, held across BOTH engine calls for that
                    // item. The admitted identity carries `face_restore` as `has_phases`, i.e. the
                    // context that was admitted already describes the restore pass — so the restore
                    // re-render belongs INSIDE this scope, not in a second one it was never admitted
                    // for. `begin_memory_request` re-checks the route/composition and the request's
                    // own geometry + PiD route against the admitted context, so a request that
                    // crossed its admission fails here rather than allocating.
                    let mut scope = model
                        .begin_memory_request(&memory_context, &memory_identity, &req, memory_route)
                        .map_err(|error| {
                            WorkerError::InvalidPayload(format!(
                                "InstantID memory request refused: {error}"
                            ))
                        })?;
                    let result = match &action {
                        InstantIdAction::Identity => {
                            model.generate(&req, &reference, &mut *on_progress)
                        }
                        InstantIdAction::Angle(kps) => {
                            model.generate_with_kps(&req, &reference, kps, &mut *on_progress)
                        }
                        InstantIdAction::Pose(keypoints) => {
                            model.generate_pose(&req, &reference, keypoints, &mut *on_progress)
                        }
                    };
                    let mut out = match result {
                        Ok(out) => out,
                        // A cancel tripped mid-denoise surfaces as the engine's cancelled error —
                        // stop cleanly (consume_gen_events posts the Canceled update). The scope
                        // still gets its terminal `finish`, which is what synchronizes the device and
                        // releases the request-local allocations. A cleanup error on an already-
                        // failing path is not allowed to mask the reason we are leaving.
                        Err(_) if cancel.is_cancelled() => {
                            let _ = scope.finish(gen_core::MemoryRunOutcome::Canceled);
                            return Ok(None);
                        }
                        Err(error) => {
                            let message = format!("InstantID generation failed: {error}");
                            let _ = scope.finish(gen_core::MemoryRunOutcome::Error {
                                message: message.clone(),
                            });
                            return Err(WorkerError::Engine(message));
                        }
                    };
                    // Optional ADetailer-style face-restore re-render (sc-3380), imposing the
                    // reference identity on the cropped face with the gender-neutral restore prompt.
                    if let Some(embedding) = &restore_embedding {
                        let restore_req = InstantIdRequest {
                            prompt: FACE_RESTORE_PROMPT.to_owned(),
                            negative: negative_prompt.clone(),
                            width: INSTANTID_FACE_RESTORE_SIDE,
                            height: INSTANTID_FACE_RESTORE_SIDE,
                            steps: steps as usize,
                            guidance,
                            ip_adapter_scale: ip_scale,
                            controlnet_scale,
                            openpose_scale,
                            seed: seed as u64,
                            // Face-restore re-render always decodes on the native VAE (sc-8371 mlx /
                            // sc-8373 candle): the engine forces this internally too (its paste-back
                            // assumes a side×side crop a 4× PiD decode would corrupt), but be explicit
                            // at the seam.
                            #[cfg(any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle")))]
                            use_pid: false,
                            sampler: sampler.clone(),
                            scheduler: scheduler.clone(),
                            cancel: cancel.clone(),
                        };
                        out = match model.restore_face(
                            &restore_req,
                            &out,
                            embedding,
                            &mut *on_progress,
                        ) {
                            Ok(out) => out,
                            Err(_) if cancel.is_cancelled() => {
                                let _ = scope.finish(gen_core::MemoryRunOutcome::Canceled);
                                return Ok(None);
                            }
                            Err(error) => {
                                let message = format!("InstantID face-restore failed: {error}");
                                let _ = scope.finish(gen_core::MemoryRunOutcome::Error {
                                    message: message.clone(),
                                });
                                return Err(WorkerError::InvalidPayload(message));
                            }
                        };
                    }
                    // Both engine calls for this item are done: run the scope's terminal cleanup
                    // (synchronize + release the request-local allocations) and drop it, so the next
                    // item begins from a released state. A cleanup failure fails the job — it means
                    // the release the admission promised did not happen.
                    scope
                        .finish(gen_core::MemoryRunOutcome::Complete)
                        .map_err(|error| {
                            WorkerError::Engine(format!(
                                "InstantID memory scope cleanup failed: {error}"
                            ))
                        })?;
                    drop(scope);
                    // Identity-likeness post-pass (sc-4409 angles / sc-4410 poses / sc-4411 plain
                    // With-Character): score this finished image against the per-job cached source
                    // embedding, on this blocking thread (the `!Send` face stack lives here). CRITICAL
                    // ordering (sc-4410): this runs AFTER the optional face-restore re-render above, so
                    // `out` is the FINAL post-restore image — the score reflects exactly what the user
                    // sees. `score_or_null` makes per-image scoring non-fatal (a backend error → a
                    // logged `null`), and a full-body / turned / profile result with no reliable frontal
                    // face records an honest `detected:false` N/A — never a misleading low number. `None`
                    // scorer (a failed construction) ⇒ no block ⇒ the field is omitted. The Image build +
                    // pixel clone is paid ONLY when a scorer exists.
                    #[cfg(any(
                        target_os = "macos",
                        all(not(target_os = "macos"), feature = "backend-candle")
                    ))]
                    let face_likeness = scorer.as_ref().and_then(|scorer| {
                        crate::face_likeness::score_generated_image(
                            Some(scorer),
                            &Image {
                                width: out.width,
                                height: out.height,
                                pixels: out.pixels.clone(),
                            },
                            Some(face_likeness_source_ref.as_str()),
                        )
                    });
                    #[cfg(not(any(
                        target_os = "macos",
                        all(not(target_os = "macos"), feature = "backend-candle")
                    )))]
                    let face_likeness: Option<JsonObject> = {
                        let _ = (&scorer, &face_likeness_source_ref);
                        None
                    };
                    Ok(Some((seed, out.width, out.height, out.pixels, face_likeness)))
                },
            )
        },
    );

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        INSTANTID_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// Tile-ControlNet detail refine (macOS, epic 3041 / sc-3060): the standalone
// `image_detail` job (Image Editor, epic 2427). Faithful port of the Python
// `run_image_detail` + `_refine_tiled_detail` (image_adapters.py) onto the engine's
// SDXL tile-ControlNet path: each tile is img2img-refined with itself as both the
// init (Reference) and the ControlNet conditioning (Control, control=same), then
// recomposed with a raised-cosine feather over the overlap. Unlike the diffusers
// pipeline, the engine requires width/height ∈ [512, 2048] and multiples of 8, so a
// tile is run at the nearest valid size and the result resized back before blending.
// ---------------------------------------------------------------------------

/// Request-scoped memory admission (sc-20799). Every assertion here is weights-free and
/// host-independent: budgets and geometries are constructed literally, paths live under a
/// `tempfile::tempdir()`, and no absolute host path or host RAM figure is ever asserted.
#[cfg(test)]
mod instantid_memory_tests {
    use super::*;
    use serde_json::json;

    fn entry<'a>(role: &'static str, path: &'a Path) -> (&'static str, &'a Path) {
        (role, path)
    }

    /// The dense tier each backend's provider declares. The two crates spell the same Bf16/no-quant
    /// tier differently — MLX `dense_numeric_tier()`, candle `resolved_numeric_tier()` (candle is
    /// dense-only, so it has no separate "dense" name) — so the tests name it once here.
    #[cfg(target_os = "macos")]
    fn dense_tier() -> gen_core::MemoryNumericTier {
        instantid_memory::dense_numeric_tier()
    }

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    fn dense_tier() -> gen_core::MemoryNumericTier {
        instantid_memory::resolved_numeric_tier()
    }

    /// A composition identity whose fingerprint is a literal, so the assertions below never depend
    /// on this machine's paths.
    fn identity(
        route: InstantIdRoute,
        adapter_count: usize,
        use_pid: bool,
        face_restore: bool,
    ) -> InstantIdMemoryIdentity {
        InstantIdMemoryIdentity {
            route,
            adapter_count,
            use_pid,
            face_restore,
            artifact_fingerprint: "fingerprint-a".to_owned(),
        }
    }

    fn budget(total_bytes: u64) -> gen_core::MemoryBudget {
        gen_core::MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        }
    }

    /// The exact context the production seam builds, plus the contract it is validated against.
    fn admitted(
        identity: &InstantIdMemoryIdentity,
        predicted_peak_bytes: u64,
        total_bytes: u64,
    ) -> (
        gen_core::MemoryProviderContract,
        gen_core::MemoryNumericTier,
        gen_core::MemoryRunContext,
    ) {
        let tier = dense_tier();
        let contract = instantid_provider_contract(tier);
        let context = instantid_memory_context(
            &contract,
            tier,
            identity,
            1024,
            1024,
            budget(total_bytes),
            predicted_peak_bytes,
        )
        .expect("the InstantID contract declares a calibration identity");
        (contract, tier, context)
    }

    /// The whole point of the change: the context the worker builds is the one the provider's own
    /// bespoke safety check ACCEPTS — on this backend, for every route.
    #[test]
    fn worker_built_context_is_accepted_by_the_provider_on_every_route() {
        for route in [
            InstantIdRoute::Identity,
            InstantIdRoute::Angle,
            InstantIdRoute::Pose,
        ] {
            let identity = identity(route, 2, true, true);
            let (contract, tier, context) = admitted(&identity, 8, 64);
            instantid_validate_admission(&contract, tier, &identity, &context)
                .unwrap_or_else(|error| panic!("{route:?} must be admitted, got {error}"));
        }
    }

    /// Every axis the provider pins is really pinned by the context we build: crossing any one of
    /// them fails CLOSED rather than admitting on a neighbour's evidence.
    #[test]
    fn every_crossed_admission_axis_fails_closed() {
        let identity = identity(InstantIdRoute::Angle, 1, true, true);
        let (contract, tier, accepted) = admitted(&identity, 8, 64);

        let crossings: Vec<(&str, gen_core::MemoryRunContext)> = vec![
            ("mode", {
                let mut context = accepted.clone();
                context.mode = gen_core::MemoryMode::TextToImage;
                context
            }),
            ("reference_count", {
                let mut context = accepted.clone();
                context.geometry.reference_count = 2;
                context
            }),
            ("batch", {
                let mut context = accepted.clone();
                context.geometry.batch = 2;
                context
            }),
            ("frames", {
                let mut context = accepted.clone();
                context.geometry.frames = 2;
                context
            }),
            ("use_pid", {
                let mut context = accepted.clone();
                context.use_pid = !context.use_pid;
                context
            }),
            ("has_phases", {
                let mut context = accepted.clone();
                context.has_phases = !context.has_phases;
                context
            }),
            ("overlay", {
                let mut context = accepted.clone();
                context.overlay = None;
                context
            }),
            ("evidence_revision", {
                let mut context = accepted.clone();
                context.evidence_revision = "borrowed-sdxl-evidence".to_owned();
                context
            }),
            ("calibration_fingerprint", {
                let mut context = accepted.clone();
                context.calibration_fingerprint = "some-other-provider".to_owned();
                context
            }),
            ("calibration_abi", {
                let mut context = accepted.clone();
                context.calibration_abi = context.calibration_abi.wrapping_add(1);
                context
            }),
            ("load_shape", {
                let mut context = accepted.clone();
                context.load_shape = gen_core::LoadShape::DeferredMaterialization;
                context
            }),
            ("budget", {
                let mut context = accepted.clone();
                context.predicted_peak_bytes = context.budget.total_bytes + 1;
                context
            }),
        ];
        for (axis, context) in crossings {
            assert!(
                instantid_validate_admission(&contract, tier, &identity, &context).is_err(),
                "crossing {axis} must fail closed"
            );
        }
    }

    /// The overlay key is the identity, so a job whose composition differs in ANY axis cannot be
    /// admitted against another job's context.
    #[test]
    fn a_context_built_for_one_composition_rejects_another() {
        let admitted_identity = identity(InstantIdRoute::Identity, 0, false, false);
        let (contract, tier, context) = admitted(&admitted_identity, 8, 64);
        for other in [
            identity(InstantIdRoute::Pose, 0, false, false),
            identity(InstantIdRoute::Identity, 1, false, false),
            InstantIdMemoryIdentity {
                artifact_fingerprint: "fingerprint-b".to_owned(),
                ..identity(InstantIdRoute::Identity, 0, false, false)
            },
        ] {
            assert!(
                instantid_validate_admission(&contract, tier, &other, &context).is_err(),
                "a context admitted for {admitted_identity:?} must reject {other:?}"
            );
        }
    }

    /// The iteration mode is what selects the route axis, so the three modes must not collapse.
    #[test]
    fn each_iteration_mode_maps_to_its_own_route() {
        assert_eq!(
            instantid_memory_route(&InstantIdMode::Identity),
            InstantIdRoute::Identity
        );
        assert_eq!(
            instantid_memory_route(&InstantIdMode::AngleSet),
            InstantIdRoute::Angle
        );
        assert_eq!(
            instantid_memory_route(&InstantIdMode::PoseSet(3)),
            InstantIdRoute::Pose
        );
    }

    /// The fingerprint keys the overlay to the EXACT artifact set: two different sets never share
    /// one, and the digest is order- and role-sensitive (a path multiset alone is not the identity).
    #[test]
    fn artifact_fingerprint_separates_every_artifact_set() {
        let a = PathBuf::from("/models/a.safetensors");
        let b = PathBuf::from("/models/b.safetensors");
        let base = |ip: &Path, scrfd: &Path| {
            instantid_artifact_fingerprint(&[
                entry("sdxl_base", Path::new("/models/bf16")),
                entry("ip_adapter", ip),
                entry("scrfd", scrfd),
            ])
        };
        let reference = base(&a, &b);
        assert_eq!(reference, base(&a, &b), "the digest must be deterministic");
        assert_eq!(reference.len(), 64, "sha256 hex");
        assert!(
            reference.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
            "the digest lands inside the overlay key, which must never carry '='"
        );
        assert_ne!(
            reference,
            base(&b, &a),
            "swapping two files between roles is a different artifact set"
        );
        // The record separator is load-bearing: without it these two entry lists concatenate to the
        // same text and two different artifact sets would share one overlay key.
        assert_ne!(
            instantid_artifact_fingerprint(&[
                entry("sdxl_base", Path::new("a")),
                entry("ip_adapter", Path::new("bc")),
            ]),
            instantid_artifact_fingerprint(&[
                entry("sdxl_base", Path::new("ab")),
                entry("ip_adapter", Path::new("c")),
            ]),
            "artifact paths must be delimited, not concatenated"
        );
        assert_ne!(
            reference,
            instantid_artifact_fingerprint(&[
                entry("sdxl_base", Path::new("/models/q4")),
                entry("ip_adapter", &a),
                entry("scrfd", &b),
            ]),
            "a different base tier is a different artifact set"
        );
        assert_ne!(
            reference,
            instantid_artifact_fingerprint(&[
                entry("sdxl_base", Path::new("/models/bf16")),
                entry("ip_adapter", &a),
                entry("scrfd", &b),
                entry("openpose", Path::new("/models/openpose")),
            ]),
            "an added branch is a different artifact set"
        );
    }

    /// The entry list is what both the fingerprint and the priced floor read, so it must name every
    /// optional artifact this job actually loads — and only those.
    #[test]
    fn artifact_entries_name_every_optional_branch_exactly_once() {
        let identitynet = WeightsSource::Dir(PathBuf::from("/w/identitynet"));
        let openpose = WeightsSource::Dir(PathBuf::from("/w/openpose"));
        let pid = gen_core::PidWeights {
            checkpoint: WeightsSource::File(PathBuf::from("/w/pid.safetensors")),
            gemma: WeightsSource::Dir(PathBuf::from("/w/gemma")),
        };
        let adapters = vec![
            AdapterSpec::new(PathBuf::from("/w/lora-a.safetensors"), 1.0, AdapterKind::Lora),
            AdapterSpec::new(PathBuf::from("/w/lora-b.safetensors"), 1.0, AdapterKind::Lora),
        ];
        let bare = instantid_artifact_entries(
            Path::new("/w/base"),
            &identitynet,
            Path::new("/w/ip.safetensors"),
            Path::new("/w/scrfd.safetensors"),
            Path::new("/w/arcface.safetensors"),
            None,
            &[],
            None,
        );
        assert_eq!(
            bare.iter().map(|(role, _)| *role).collect::<Vec<_>>(),
            vec!["sdxl_base", "identitynet", "ip_adapter", "scrfd", "arcface"]
        );
        let full = instantid_artifact_entries(
            Path::new("/w/base"),
            &identitynet,
            Path::new("/w/ip.safetensors"),
            Path::new("/w/scrfd.safetensors"),
            Path::new("/w/arcface.safetensors"),
            Some(&openpose),
            &adapters,
            Some(&pid),
        );
        assert_eq!(
            full.iter().map(|(role, _)| *role).collect::<Vec<_>>(),
            vec![
                "sdxl_base",
                "identitynet",
                "ip_adapter",
                "scrfd",
                "arcface",
                "openpose",
                "adapter",
                "adapter",
                "pid_checkpoint",
                "pid_gemma",
            ]
        );
        assert_eq!(
            full[6].1,
            Path::new("/w/lora-a.safetensors"),
            "adapters keep their declared order — a reorder is a different fingerprint"
        );
    }

    /// An over-count is the one direction the floor must never have: on a fresh install the face
    /// files live INSIDE the bundle directory, and a naive sum would charge them twice.
    #[test]
    fn priced_bytes_count_a_nested_artifact_once() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = tmp.path().join("bundle");
        std::fs::create_dir_all(&bundle).unwrap();
        std::fs::write(bundle.join("scrfd.safetensors"), vec![7u8; 2048]).unwrap();
        let nested = bundle.join("scrfd.safetensors");
        let outside = tmp.path().join("ip.safetensors");
        std::fs::write(&outside, vec![7u8; 1024]).unwrap();

        let dir_only = instantid_artifact_bytes(&[entry("identitynet", &bundle)]);
        assert_eq!(dir_only, 2048);
        assert_eq!(
            instantid_artifact_bytes(&[entry("identitynet", &bundle), entry("scrfd", &nested)]),
            2048,
            "a file inside a summed directory must not be charged twice"
        );
        assert_eq!(
            instantid_artifact_bytes(&[
                entry("identitynet", &bundle),
                entry("scrfd", &nested),
                entry("ip_adapter", &outside),
            ]),
            3072,
            "an artifact outside the directory must still be charged"
        );
        assert_eq!(
            instantid_artifact_bytes(&[entry("ip_adapter", &tmp.path().join("absent"))]),
            0,
            "an unreadable artifact contributes no evidence, never a phantom requirement"
        );
    }

    /// The predicted peak comes from the manifest when the manifest declares one — the measured
    /// per-tier row first, then the padded floor. Never a constant of ours.
    #[test]
    fn manifest_peak_prefers_the_measured_tier_row_then_min_memory() {
        let declared = json!({
            "candle": { "vramGbByTier": { "bf16": 21.5, "q4": 9.0 }, "minMemoryGb": 12.0 }
        });
        let declared = declared.as_object().unwrap();
        assert_eq!(
            instantid_manifest_peak_gb(declared, "candle", "bf16"),
            Some(21.5)
        );
        assert_eq!(
            instantid_manifest_peak_gb(declared, "candle", "q8"),
            Some(12.0),
            "an unmeasured tier falls back to the declared floor"
        );
        assert_eq!(
            instantid_manifest_peak_gb(declared, "mlx", "bf16"),
            None,
            "another backend's block is not this backend's evidence"
        );
        let silent = json!({});
        assert_eq!(
            instantid_manifest_peak_gb(silent.as_object().unwrap(), "candle", "bf16"),
            None,
            "instantid_realvisxl declares nothing today, so the caller must price the artifacts"
        );
    }

    /// GB→byte conversion is the currency bridge between the worker's GB-denominated probes and the
    /// byte-denominated contract; getting the scale wrong silently mis-sizes every budget.
    #[test]
    fn gib_bytes_converts_in_binary_gigabytes() {
        assert_eq!(instantid_gib_bytes(1.0), 1024 * 1024 * 1024);
        assert_eq!(instantid_gib_bytes(0.5), 512 * 1024 * 1024);
        assert_eq!(instantid_gib_bytes(0.0), 0);
    }

    /// The declared numeric tier must be the tier the lane actually loads, because the provider
    /// rejects a selection whose tier differs from the loaded one.
    #[test]
    fn declared_tier_matches_the_tier_this_lane_loads() {
        let request = |advanced: serde_json::Value| {
            ImageRequest::from_payload(
                json!({ "model": "instantid_realvisxl", "advanced": advanced })
                    .as_object()
                    .unwrap(),
            )
        };
        let dense = instantid_memory_tier(&request(json!({})));
        assert_eq!(dense.quant, None);
        assert_eq!(
            instantid_memory_backend_keys(&request(json!({}))).1,
            "bf16"
        );
        #[cfg(target_os = "macos")]
        {
            assert_eq!(
                instantid_memory_tier(&request(json!({ "mlxQuantize": 4 }))).quant,
                Some(Quant::Q4)
            );
            assert_eq!(
                instantid_memory_tier(&request(json!({ "mlxQuantize": 8 }))).quant,
                Some(Quant::Q8)
            );
            assert_eq!(
                instantid_memory_backend_keys(&request(json!({ "mlxQuantize": 4 }))),
                ("mlx", "q4")
            );
            // The MLX contract validates the tier request-by-request: a context built for the dense
            // tier must not admit a q4 load.
            let identity = identity(InstantIdRoute::Identity, 0, false, false);
            let (contract, _, context) = admitted(&identity, 8, 64);
            let crossed = gen_core::MemoryNumericTier {
                quant: Some(Quant::Q4),
                ..instantid_memory::dense_numeric_tier()
            };
            assert!(
                instantid_validate_admission(&contract, crossed, &identity, &context).is_err(),
                "a dense-admitted context must not price a q4 load"
            );
        }
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        {
            assert_eq!(
                instantid_memory_tier(&request(json!({ "mlxQuantize": 4 }))).quant,
                None,
                "the candle InstantID stack is dense-only"
            );
            assert_eq!(
                instantid_memory_backend_keys(&request(json!({ "mlxQuantize": 4 }))),
                ("candle", "bf16")
            );
        }
    }
}

#[cfg(test)]
mod instantid_tier_tests {
    use super::*;
    use serde_json::json;

    fn request(advanced: serde_json::Value) -> ImageRequest {
        ImageRequest::from_payload(
            json!({ "model": "instantid_realvisxl", "advanced": advanced })
                .as_object()
                .unwrap(),
        )
    }

    /// Seed a present `<tier>/unet/<file>` so [`instantid_tier_subdir`]'s probe sees it downloaded
    /// (InstantID's backbone always lives under `unet/`).
    fn seed_unet(root: &Path, tier: &str, file: &str) {
        let dir = root.join(tier).join("unet");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file), b"x").unwrap();
    }

    /// The InstantID backbone defaults to the DENSE `bf16/` tier (the validated fp16 identity
    /// envelope), NOT the standard-tier `q4/` default — quant degrades ArcFace identity, so it is
    /// opt-in. On MLX, `mlxQuantize` 4/8 select the packed `q4/`/`q8/` tiers (mirroring
    /// [`instantid_quant`]); off-Mac (candle / no-face) the backbone is dense-only, so every request
    /// resolves `bf16/` regardless of the knob.
    #[test]
    fn defaults_to_bf16_and_selects_packed_tiers_only_on_mlx() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_unet(root, "q4", "diffusion_pytorch_model.safetensors");
        seed_unet(root, "q8", "diffusion_pytorch_model.safetensors");
        seed_unet(root, "bf16", "diffusion_pytorch_model.fp16.safetensors");

        // Unset / opt-out → bf16 on every backend.
        assert_eq!(
            instantid_tier_subdir(root, &request(json!({}))),
            root.join("bf16")
        );
        assert_eq!(
            instantid_tier_subdir(root, &request(json!({ "mlxQuantize": 0 }))),
            root.join("bf16")
        );

        // Q4/Q8 opt-in resolves the packed tier ONLY on the MLX lane; the candle lane is dense-only.
        #[cfg(target_os = "macos")]
        {
            assert_eq!(
                instantid_tier_subdir(root, &request(json!({ "mlxQuantize": 4 }))),
                root.join("q4")
            );
            assert_eq!(
                instantid_tier_subdir(root, &request(json!({ "mlxQuantize": 8 }))),
                root.join("q8")
            );
            assert_eq!(
                instantid_tier_subdir(root, &request(json!({ "mlxQuantize": "8" }))),
                root.join("q8")
            );
        }
        #[cfg(not(target_os = "macos"))]
        {
            assert_eq!(
                instantid_tier_subdir(root, &request(json!({ "mlxQuantize": 4 }))),
                root.join("bf16")
            );
            assert_eq!(
                instantid_tier_subdir(root, &request(json!({ "mlxQuantize": 8 }))),
                root.join("bf16")
            );
        }
    }

    /// A partial turnkey falls back to a present tier rather than a half-empty subdir (so the engine
    /// surfaces a clear missing-weights error), and an absent turnkey resolves to the repo root.
    #[test]
    fn falls_back_when_preferred_tier_absent() {
        // Only q8 downloaded: an unset (bf16-preferred) request falls through bf16 → q8.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_unet(root, "q8", "diffusion_pytorch_model.safetensors");
        assert_eq!(
            instantid_tier_subdir(root, &request(json!({}))),
            root.join("q8")
        );
        // Nothing present → the repo root (engine surfaces the missing-weights error).
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(
            instantid_tier_subdir(empty.path(), &request(json!({}))),
            empty.path().to_path_buf()
        );
    }

    #[test]
    fn fallback_prefers_q8_over_q4_and_skips_a_torn_q8() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_unet(root, "q8", "diffusion_pytorch_model.safetensors");
        seed_unet(root, "q4", "diffusion_pytorch_model.safetensors");

        assert_eq!(
            instantid_tier_subdir(root, &request(json!({}))),
            root.join("q8"),
            "dense fallback should preserve the highest available fidelity"
        );

        std::fs::write(
            root.join("q8").join("model_index.json"),
            br#"{"unet":["diffusers","UNet2DConditionModel"],"tokenizer":["transformers","CLIPTokenizer"]}"#,
        )
        .unwrap();
        assert_eq!(
            instantid_tier_subdir(root, &request(json!({}))),
            root.join("q4"),
            "a q8 tier missing a model-index component must not block a complete q4 fallback"
        );
    }
}
