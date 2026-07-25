#[allow(unused_imports)]
use super::prelude::*;
#[cfg(target_os = "macos")]
use super::wan::{advanced_opt_f32, advanced_opt_u32};
#[cfg(target_os = "macos")]
use super::wan::{generate_video_using, VideoGenInput};

// ---------------------------------------------------------------------------
// Mochi 1 (epic 1788 / sc-11992): a 10B AsymmDiT text-to-video model with a 6×-temporal AsymmVAE,
// served natively on BOTH backends — `mlx-gen-mochi` (macOS) and `candle-gen-mochi` (Windows/Linux
// CUDA). Two facts shape this whole block, and both DEVIATE from the LTX template it otherwise
// mirrors:
//
//  1. ONE engine id, TWO backends. Both providers register `MODEL_ID = "mochi_1"` (unlike LTX's
//     `ltx_2_3` + `ltx_2_3_distilled` split), so `mochi_engine_id` is lane-agnostic.
//  2. ONE repo, TWO backends. Every tier lives in `SceneWorks/mochi-1-mlx` with no `platforms` tag,
//     because A6's `.scales`-detect seam lets candle ingest the mlx-affine tiers 1:1 (CUDA-validated
//     on Blackwell, sc-11990). So the tier resolver + the on-demand fetches below are compiled for
//     the SUPERSET of both lanes and serve candle exactly as they serve MLX.
//
// A tier IS a directory, not a requant toggle: both descriptors advertise `supported_quants: &[]`, so
// `advanced.mlxQuantize` selects WHICH DIR to load and the provider only ASSERTS the tier's baked-in
// level from its `split_model.json` (a mismatch is a hard load error, never a silent bf16 run).
// Installed layout — the tier dirs are SIBLINGS of the shared components, which the provider resolves
// from the tier dir's PARENT (`resolve_component_root`):
//
//     <model_dir>/{q4,q8,bf16}/{split_model.json, transformer/}
//     <model_dir>/{text_encoder,tokenizer,vae}/          <- the shared coRequisite (~10.4 GB)
//
// Text-to-video ONLY (`conditioning: []` on both descriptors), no LoRA/LoKr, no sampler/scheduler
// axis (one fixed flow-match Euler integrator), `max_count: 1`.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Mochi asset (the candle sibling is [`CANDLE_MOCHI_ADAPTER`]).
#[cfg(target_os = "macos")]
pub(super) const MOCHI_ADAPTER: &str = "mlx_mochi";

/// The ONE turnkey repo serving BOTH backends (epic 1788 / A6 sc-11990). Deliberately NOT the LTX
/// shape (MLX rehost for macOS + upstream torch repo off-Mac): candle ingests these mlx-affine tiers
/// directly, `SceneWorks/mochi-1-candle` was never published, and pointing every platform here also
/// sidesteps sc-12113 (upstream `genmo/mochi-1-preview` ships both a bf16 and an fp32 `vae/` and the
/// loader's dir-glob picks fp32 non-deterministically — this rehost ships bf16 only).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const MOCHI_REPO: &str = "SceneWorks/mochi-1-mlx";

/// Pinned revision for the fixed [`MOCHI_REPO`]. The repo is a hard-coded const (no manifest/payload
/// override reaches the on-demand `q8/*` + `bf16/*` fetches below), so pulling the mutable `main`
/// branch would let an upstream re-push silently swap a checkpoint we load. Pin the exact commit for
/// defense-in-depth, mirroring [`LTX_BUNDLE_REVISION`] / the SeedVR2 + Real-ESRGAN pins. This is the
/// same revision B1's manifest sizes were read from (HF API rev `90a87786`); the native downloader
/// still verifies each file's own hash on download.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const MOCHI_REVISION: &str = "90a87786d9b5b592c3b3c083e5fcdf130007b1de";

/// Operator override for the Mochi model root (or a single tier dir) — the smoke's seam and the
/// escape hatch for a hand-staged conversion, mirroring `$SCENEWORKS_MLX_LTX_DIR`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const MOCHI_DIR_ENV: &str = "SCENEWORKS_MLX_MOCHI_DIR";

/// SceneWorks Mochi model id → the gen-core registry id, or `None` if not a Mochi family id.
///
/// ONE id covers BOTH backends (both providers register `MODEL_ID = "mochi_1"`) — contrast
/// [`ltx_engine_id`], which folds two sceneworks ids onto one engine, and the candle LTX arm, which
/// maps to a DIFFERENT `ltx_2_3_distilled` id. Despite that, this is the **MLX lane's** resolver: it
/// is the `resolve_video_route` / `mochi_available` / `ensure_video_engine_weights` key. The candle
/// lane resolves the same id through its own [`candle_video_engine_id`] table and then keys off the
/// resolved `engine_id` (mirroring its local `is_ltx`), so it never calls this — hence the macOS gate
/// rather than the superset one the shared tier helpers below carry.
#[cfg(target_os = "macos")]
pub(super) fn mochi_engine_id(model: &str) -> Option<&'static str> {
    (model == "mochi_1").then_some("mochi_1")
}

/// Whether `dir` is a loadable Mochi TIER dir: the `split_model.json` the provider reads to resolve
/// the tier's baked-in quant, plus the AsymmDiT weights themselves. A tier dir mid-download (manifest
/// written, `transformer/` absent, or vice versa) fails this, so it never half-loads.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn mochi_tier_dir_is_complete(dir: &Path) -> bool {
    dir.join("split_model.json").is_file()
        && dir.join("transformer").join("model.safetensors").is_file()
}

/// Whether `root` carries the SHARED T5-XXL text encoder + tokenizer + AsymmVAE the provider resolves
/// from the tier dir's PARENT. These ship as a `coRequisite` (sc-9696) — a separate ~10.4 GB download
/// tracked once and excluded from the tier matrix — so a user can genuinely have a complete `q4/` and
/// no `text_encoder/`. Checking the tier dir alone would then resolve "successfully" and dead-end
/// inside the provider on a missing-file error.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_shared_is_complete(root: &Path) -> bool {
    ["text_encoder", "tokenizer", "vae"]
        .iter()
        .all(|component| root.join(component).is_dir())
}

/// Parse `advanced.mlxQuantize` (int or numeric string) → the requested bit width, if present.
/// Mirrors [`ltx_quant_bits`].
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_quant_bits(request: &VideoRequest) -> Option<i64> {
    request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
}

/// Whether the request opts into the Q8 tier (`advanced.mlxQuantize >= 8`, int or string).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn mochi_wants_q8(request: &VideoRequest) -> bool {
    mochi_quant_bits(request).is_some_and(|bits| (8..=15).contains(&bits))
}

/// Whether the request opts into the dense **bf16** tier (`advanced.mlxQuantize <= 0`, or an explicit
/// `>= 16`). Never the default: absent ⇒ q4, so the ~20 GB dense tier is a deliberate opt-in
/// (mirrors [`ltx_wants_bf16`] / [`resolve_mlx_dense_quant`]'s `<= 0` rule, and additionally accepts
/// a literal `16` since bf16 IS 16-bit and a caller may say so).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn mochi_wants_bf16(request: &VideoRequest) -> bool {
    mochi_quant_bits(request).is_some_and(|bits| bits <= 0 || bits >= 16)
}

/// The Mochi tier search order — preferred tier first, then the always-SMALLER fallback tiers so a
/// partially-installed model still loads: `mlxQuantize <= 0`/`>= 16` ⇒ bf16, `8..=15` ⇒ q8, and — with
/// an explicit `1..=4` OR no explicit pick — the **q4** default (q4-first).
///
/// bf16 stays OUT of the default order so a default job never loads the dense tier by accident, and
/// the video lane keeps the pre-sc-10726 q4-first default (sc-10859) rather than epic 10721's app-wide
/// Q8 — see [`wan_tier_order`] / [`ltx_bundle_tier_order`] for that carve-out's rationale. For Mochi
/// the argument is sharper still: the untiled decode already dominates the footprint, so a silent Q8
/// buys quality the machine may not have the memory to spend.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn mochi_tier_order(request: &VideoRequest) -> &'static [&'static str] {
    if mochi_wants_bf16(request) {
        &["bf16", "q8", "q4"]
    } else if mochi_wants_q8(request) {
        &["q8", "q4"]
    } else {
        &["q4", "q8"]
    }
}

/// Pick the first COMPLETE tier subdir of a Mochi model `root`, trying `order` (preferred first).
/// Requires the shared co-requisite at `root` too — a tier dir without the T5/VAE siblings is not
/// loadable, so it must not resolve.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn mochi_tier_subdir(root: &Path, order: &[&str]) -> Option<PathBuf> {
    if !mochi_shared_is_complete(root) {
        return None;
    }
    order
        .iter()
        .map(|tier| root.join(tier))
        .find(|dir| mochi_tier_dir_is_complete(dir))
}

/// The tier's baked-in quant level, read from the resolved tier dir's `split_model.json` — the same
/// manifest the provider reads. `Some(Q4)`/`Some(Q8)` for a packed tier, `None` for dense bf16.
///
/// Threaded onto the `LoadSpec` so (a) the asset record + metrics report the tier that was actually
/// loaded rather than the tier that was asked for, and (b) the provider's assert becomes a real
/// cross-check that our tier selection and its manifest read agree. Deriving this from the RESOLVED
/// dir rather than from `advanced.mlxQuantize` is what keeps a fallback (requested q8, only q4
/// installed) from turning into the provider's hard "spec.quantize disagrees with the tier" error.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn mochi_tier_quant(tier_dir: &Path) -> Option<Quant> {
    let raw = std::fs::read_to_string(tier_dir.join("split_model.json")).ok()?;
    let manifest: Value = serde_json::from_str(&raw).ok()?;
    if !manifest.get("quantized").and_then(Value::as_bool)? {
        return None;
    }
    match manifest.get("quantization_bits").and_then(Value::as_i64)? {
        4 => Some(Quant::Q4),
        8 => Some(Quant::Q8),
        _ => None,
    }
}

/// The dir the PRE-DOWNLOAD fit gate measures: the REQUESTED tier's path, whether or not it has been
/// downloaded yet. `None` when no model root is locatable at all ⇒ no signal ⇒ [`mochi_precheck`]
/// skips (never blocks without evidence, matching the gate's own fail-open rule).
///
/// Deliberately NOT [`resolve_mochi_model_dir`]: that one requires a COMPLETE tier and falls back to a
/// smaller installed one, and it hard-errors when nothing is installed — which is correct AFTER the
/// download and wrong before it (a first-ever q8 job would be rejected as "not downloaded" instead of
/// downloading). This returns a bare path and lets the gate's on-disk scan decide.
///
/// Naming the REQUESTED tier (not the fallback) is what keeps the weights term a strict lower bound on
/// what THIS job will hold resident: the scan finds the shared co-requisites plus this tier's own bytes
/// if present, never a different tier's.
///
/// **Shared by both lanes** (sc-12373): every input is lane-neutral — the tier ORDER the request asked
/// for, the `MOCHI_DIR_ENV` override, and the one hosted [`MOCHI_REPO`] snapshot that serves candle and
/// MLX alike. Widened from macOS-only rather than copied, so the two pre-download gates cannot disagree
/// about which directory they are measuring.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn mochi_precheck_dir(settings: &Settings, request: &VideoRequest) -> Option<PathBuf> {
    // The preferred tier — `mochi_tier_order`'s head is the one the request actually asked for; the
    // rest of the order is fallback, which the pre-check must not measure.
    let requested = mochi_tier_order(request).first().copied()?;
    let root = match std::env::var(MOCHI_DIR_ENV) {
        Ok(override_dir) => {
            let root = PathBuf::from(override_dir.trim());
            // A self-contained dir that IS one tier — the components live under it, so it is already
            // the thing to measure (and no download can change it).
            if mochi_tier_dir_is_complete(&root) && mochi_shared_is_complete(&root) {
                return Some(root);
            }
            root
        }
        Err(_) => huggingface_snapshot_dir(&settings.data_dir, MOCHI_REPO)?,
    };
    Some(root.join(requested))
}

/// PRE-DOWNLOAD Mochi admission check (sc-12322) — the same gate as [`mochi_preflight`], run BEFORE
/// `ensure_mochi_*_present` fetches a ~13-20 GiB tier, so a job that cannot fit is refused without
/// paying for the download first.
///
/// This is sound because the weights term is a conservative FLOOR here, not zero. Mochi's
/// `text_encoder/` + `tokenizer/` + `vae/` are a `coRequisite: true` download (~9.7 GiB) that is
/// already on disk before an on-demand `<tier>/*` fetch, and `mochi_resident_bytes` sums those shared
/// siblings from the tier dir's PARENT even when the tier dir itself is absent. So:
///
/// ```text
/// floor = shared co-requisites + (requested tier, if already downloaded)  ≤  actual resident weights
/// ⇒ needed(floor) ≤ needed(actual) ⇒ {pre-check refuses} ⊆ {full gate refuses}
/// ```
///
/// i.e. it can only refuse when refusal is CERTAIN — a config that fits is never rejected early, which
/// is the property that makes an early gate safe at all. Anything it admits still faces the unchanged
/// full check after the download.
///
/// The floor is what makes this bite on the machine the story is named for: at the shipped 151-frame
/// default the shared components alone give 9.73 + 60.56 decode + 2 OS = 72.29 GiB > 64. A
/// WEIGHTS-FREE pre-check (decode + OS only) would total 62.56 and ADMIT on a 64 GB Mac — downloading
/// the tier just to refuse it, which is the whole bug.
///
/// Lives beside [`mochi_preflight`] and for the same reason: one seam, asserted directly across
/// budgets and tier layouts. Its ORDER against the `ensure_mochi_*_present` fetches — the thing that
/// makes it worth having at all — is pinned separately, at the call site, by
/// `generate_mochi_using_refuses_before_paying_for_the_tier_download` (sc-12318).
#[cfg(target_os = "macos")]
pub(super) fn mochi_precheck(
    model: &str,
    engine_id: &str,
    tier_dir: Option<&Path>,
    raw_frames: u32,
    width: u32,
    height: u32,
) -> WorkerResult<()> {
    let Some(tier_dir) = tier_dir else {
        return Ok(());
    };
    // The COERCED count — the frames the decode will really pay for, exactly as `mochi_preflight`
    // computes it. Pre-checking the raw request would gate on a length the engine never renders.
    let frames = video_frame_count(model, raw_frames);
    crate::mlx_fit_gate::mochi_fit_check(engine_id, tier_dir, frames, width, height)
}

/// PRE-DOWNLOAD Mochi admission check for the candle lane (sc-12373) — the CUDA twin of
/// [`mochi_precheck`], run BEFORE `ensure_mochi_*_present` fetches a ~13–20 GiB tier.
///
/// **The soundness argument is [`mochi_precheck`]'s, verbatim, and it is lane-neutral:** the weights
/// term here is a conservative FLOOR, not zero. Mochi's `text_encoder/` + `tokenizer/` + `vae/` are a
/// `coRequisite: true` download (~9.7 GiB) already on disk before an on-demand `<tier>/*` fetch, and
/// `mochi_resident_bytes` sums those shared siblings from the tier dir's PARENT even when the tier dir
/// itself is absent. So `floor ≤ actual` ⇒ `needed(floor) ≤ needed(actual)` ⇒ `{pre-check refuses} ⊆
/// {full gate refuses}`: it can only refuse when refusal is CERTAIN. Anything it admits still faces the
/// unchanged [`mochi_vram_preflight`] after the download.
///
/// What is NOT shared with the MLX twin, and why this is a sibling rather than a widened cfg: the
/// budget is a `VramBudget` gated on `free_gb` (not unified `total_gb`), the reserve is
/// `vram_gate::HEADROOM_GB` (not `OS_RESERVE_GB` — the OS does not draw from discrete VRAM), and the
/// message is CUDA-worded. `budget` arrives resolved, so this stays free of the GPU probe and the whole
/// decision is unit-testable without CUDA.
///
/// Its ORDER against the fetches — the entire point — is pinned at the call site by
/// `generate_candle_video_using_refuses_before_paying_for_the_tier_download`, not by this function's
/// existence: clippy's dead-code pass proves the call exists, never that it runs first.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn mochi_vram_precheck(
    model: &str,
    engine_id: &str,
    tier_dir: Option<&Path>,
    raw_frames: u32,
    width: u32,
    height: u32,
    gpu_id: &str,
    budget: Option<crate::vram_gate::VramBudget>,
) -> WorkerResult<()> {
    let Some(tier_dir) = tier_dir else {
        return Ok(());
    };
    // The COERCED count — the frames the decode will really pay for, exactly as `mochi_vram_preflight`
    // computes it. Pre-checking the raw request would gate on a length the engine never renders.
    let frames = video_frame_count(model, raw_frames);
    match crate::vram_gate::mochi_fit_error(
        engine_id,
        crate::mlx_fit_gate::mochi_resident_bytes(tier_dir),
        frames,
        width,
        height,
        gpu_id,
        budget,
    ) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Resolve the Mochi tier dir to load: the `$SCENEWORKS_MLX_MOCHI_DIR` override (accepted as either a
/// model root carrying tier subdirs or a single self-contained tier dir), then the turnkey
/// [`MOCHI_REPO`] snapshot's tier subdir. Only a COMPLETE tier + its shared co-requisite counts.
///
/// Shared by BOTH lanes (one repo, both backends), so the candle generation arm calls this instead of
/// `candle_video_snapshot_dir` — the flat-snapshot resolver has no notion of the tier layout.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_mochi_model_dir(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<PathBuf> {
    let order = mochi_tier_order(request);
    if let Ok(override_dir) = std::env::var(MOCHI_DIR_ENV) {
        let root = PathBuf::from(override_dir.trim());
        // A model root with `q4/`/`q8/`/`bf16/` subdirs …
        if let Some(dir) = mochi_tier_subdir(&root, order) {
            return Ok(dir);
        }
        // … or a self-contained dir that IS one tier (components under it, the provider's
        // `resolve_component_root` self-resolution case).
        if mochi_tier_dir_is_complete(&root) && mochi_shared_is_complete(&root) {
            return Ok(root);
        }
    }
    let requested_variant = mochi_tier_order(request).first().copied();
    let receipt_dir = crate::model_jobs::huggingface_receipt_weights_dir(
        &settings.data_dir,
        MOCHI_REPO,
        Some(&request.model),
        requested_variant,
    );
    if let Some(receipt_dir) = receipt_dir {
        // A receipt tier is already the exact, atomic file set selected at install time.  Do not run
        // it through the current-manifest fallback order (or self-heal it with newly named files).
        let receipt_root = receipt_dir.parent();
        let shared_receipt = crate::model_jobs::huggingface_receipt_weights_dir(
            &settings.data_dir,
            MOCHI_REPO,
            Some(&request.model),
            Some("default"),
        );
        if mochi_tier_dir_is_complete(&receipt_dir)
            && shared_receipt.as_deref() == receipt_root
            && receipt_root.is_some_and(mochi_shared_is_complete)
        {
            return Ok(receipt_dir);
        }
    }
    if let Some(root) = huggingface_snapshot_dir(&settings.data_dir, MOCHI_REPO) {
        if let Some(dir) = mochi_tier_subdir(&root, order) {
            return Ok(dir);
        }
        // The snapshot exists but nothing loadable resolved — say WHICH half is missing rather than a
        // blanket "not found", since the tiers and the shared components are separate downloads.
        if !mochi_shared_is_complete(&root) {
            return Err(WorkerError::InvalidPayload(format!(
                "{}: the shared Mochi text encoder / tokenizer / VAE are not installed (expected \
                 text_encoder/, tokenizer/ and vae/ beside the tier dirs under {}). Re-download \
                 Mochi 1 in the Model Manager — the shared components install alongside whichever \
                 tier you pick.",
                request.model,
                root.display()
            )));
        }
        return Err(WorkerError::InvalidPayload(format!(
            "{}: no complete Mochi tier found under {} (looked for {order:?}). Download the tier you \
             selected in the Model Manager, or set ${MOCHI_DIR_ENV} to a model dir.",
            request.model,
            root.display()
        )));
    }
    Err(WorkerError::InvalidPayload(format!(
        "{}: Mochi 1 is not downloaded (no {MOCHI_REPO} snapshot under {}). Install it from the \
         Model Manager, or set ${MOCHI_DIR_ENV} to a model dir.",
        request.model,
        settings.data_dir.display()
    )))
}

/// Whether the linked Mochi engine can serve this request now: a Mochi model id, in the ONE mode it
/// supports, with resolvable weights.
///
/// **text_to_video only** — both descriptors declare `conditioning: []`, so there is no i2v /
/// first-last / extend / bridge / replace surface. This gate matters because `VideoRequest`'s mode
/// DEFAULTS to `image_to_video` when the payload omits one, so a mode check is what keeps a
/// conditioning-shaped Mochi job out of the route rather than letting it reach the engine's
/// `validate_request` (or, worse, an unrelated arm of the ladder).
#[cfg(target_os = "macos")]
pub(super) fn mochi_available(request: &VideoRequest, settings: &Settings) -> bool {
    mochi_engine_id(&request.model).is_some()
        && request.mode == "text_to_video"
        && resolve_mochi_model_dir(settings, request).is_ok()
}

/// Fetch the `q8/` tier on demand (mirrors [`ensure_ltx_q8_present`]). The default install is the lean
/// q4 tier + the shared co-requisite; a job that toggles `advanced.mlxQuantize: 8` without a
/// pre-installed q8 pulls just `q8/*` from the FIXED [`MOCHI_REVISION`] before any compute. No-op when
/// q8 isn't requested, when the snapshot isn't downloaded at all (resolve surfaces the clear
/// "not downloaded" error), or when `q8/` is already complete. Fails loud on a real download error.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn ensure_mochi_q8_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    if !mochi_wants_q8(request) {
        return Ok(());
    }
    ensure_mochi_tier_present(api, settings, job, "q8").await
}

/// Fetch the dense `bf16/` tier on demand (~20 GB) — the [`ensure_mochi_q8_present`] sibling for the
/// power-user tier ([`mochi_wants_bf16`]).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn ensure_mochi_bf16_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    if !mochi_wants_bf16(request) {
        return Ok(());
    }
    ensure_mochi_tier_present(api, settings, job, "bf16").await
}

/// The shared body behind the two on-demand tier fetches: pull `<tier>/*` from the pinned
/// [`MOCHI_REVISION`] when the snapshot exists but that tier doesn't.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn ensure_mochi_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    tier: &str,
) -> WorkerResult<()> {
    let tier_receipt = crate::model_jobs::huggingface_receipt_weights_dir(
        &settings.data_dir,
        MOCHI_REPO,
        None,
        Some(tier),
    );
    let shared_receipt = crate::model_jobs::huggingface_receipt_weights_dir(
        &settings.data_dir,
        MOCHI_REPO,
        None,
        Some("default"),
    );
    if let (Some(tier_dir), Some(shared_root)) = (tier_receipt, shared_receipt) {
        if tier_dir.parent() == Some(shared_root.as_path())
            && mochi_tier_dir_is_complete(&tier_dir)
            && mochi_shared_is_complete(&shared_root)
        {
            return Ok(());
        }
    }
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, MOCHI_REPO) else {
        return Ok(());
    };
    if mochi_tier_dir_is_complete(&root.join(tier)) {
        return Ok(());
    }
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        MOCHI_REPO,
        MOCHI_REVISION,
        &[format!("{tier}/*")],
    )
    .await
    .map(|_| ())
}

/// The `rawSettings` recorded on a Mochi asset. `mochiTier` names the tier that was actually LOADED
/// (from the resolved dir's `split_model.json`), not the one the request asked for — those differ
/// when a requested tier isn't installed and the order falls back.
#[cfg(target_os = "macos")]
pub(super) fn mochi_raw_settings(request: &VideoRequest, quant: Option<Quant>) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert(
        "mochiTier".to_owned(),
        Value::String(
            match quant {
                Some(Quant::Q4) => "q4",
                Some(Quant::Q8) => "q8",
                _ => "bf16",
            }
            .to_owned(),
        ),
    );
    Value::Object(raw)
}

/// Everything a Mochi generation must settle BEFORE the load, resolved as ONE unit by
/// [`mochi_preflight`] (sc-11992).
///
/// These travel together on purpose. Both fields are things [`generate_mochi`] cannot build its
/// [`VideoGenInput`] without, and [`mochi_preflight`] is the only thing that produces them — so the
/// generation arm cannot obtain a frame count or a quant marker on a path that skipped the fit gate.
/// Splitting them back into free-standing calls in the caller is what let the gate be silently dropped
/// (the mutation that survived the first round of this story's review).
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MochiPreflight {
    /// The requested frame count coerced onto Mochi's `6k+1` lattice.
    pub(super) frames: u32,
    /// The tier dir's quant marker (`None` = the dense bf16 tier).
    pub(super) quant: Option<Quant>,
}

/// The Mochi pre-flight: coerce the clip length onto the engine's lattice, refuse a job that cannot
/// fit this machine, and report the tier's quant — the whole "before we load anything" decision, as a
/// pure, synchronously testable seam (sc-11992).
///
/// Both decisions here are invisible in the output when wrong — a mis-coerced frame count is a hard
/// engine reject, a skipped gate is a `SIGKILL` — so they live in one seam where `mochi_preflight_*`
/// can assert them directly by model id, budget, and clip length.
///
/// This function was originally justified by the claim that [`generate_mochi`] is "unreachable from a
/// unit test" because it is `async` and takes a live `ApiClient`/`JobSnapshot`. **That was wrong**
/// (sc-12318): `ApiClient::new` does no I/O, `JobSnapshot` deserializes from a literal, and
/// `generate_mochi_using` now drives the whole arm against a stub engine. Keep the seam anyway — it
/// gives the gate cheap, direct coverage across budgets and clip lengths, and returning
/// `{frames, quant}` as a unit is what stops the arm obtaining either on a path that skipped the gate.
/// But do NOT reason from that claim again: a decision put here is *additionally* covered, not
/// *only* coverable here.
///
/// `Ok(MochiPreflight)` admits; `Err` is the actionable pre-crash rejection.
#[cfg(target_os = "macos")]
pub(super) fn mochi_preflight(
    model: &str,
    engine_id: &str,
    tier_dir: &Path,
    raw_frames: u32,
    width: u32,
    height: u32,
) -> WorkerResult<MochiPreflight> {
    // The shared stride ladder, dispatched BY MODEL — so Mochi lands on `mochi_frame_count`'s 6k+1
    // lattice and never on Wan's 4k+1 stride. NOT `wan_frame_count` (see `video_frame_count`): the
    // shipped 5 s @ 30 fps default takes `wan_frame_count(150) = 149`, and `149 % 6 == 5` is OFF the
    // lattice, which the engine's `validate_request` hard-rejects.
    let frames = video_frame_count(model, raw_frames);

    // PRE-FLIGHT FIT GATE (epic AC: "an unsupported environment shows an actionable error, not a
    // crash"). This MUST run before the load: Mochi holds ~18.7 GiB of weights resident for the whole
    // run (`supports_sequential_offload: false`) and its AsymmVAE decode is UNTILED, so the peak grows
    // linearly with clip length (sc-12291) — ~60 GiB of decode alone at the shipped 5 s default. MLX's
    // default error handler is `exit(-1)`, a hard process kill that CANNOT be mapped to a job error
    // after the fact, so an over-budget job has to be refused here or it takes the worker down with
    // it. The generic `mlx_fit_gate` cannot cover this: it is resolution-blind by construction and its
    // weights-fit floor would admit on the 18.7 GiB weights alone.
    //
    // It takes the COERCED `frames` — the count the decode will actually pay for — not the raw
    // request, and not a constant.
    crate::mlx_fit_gate::mochi_fit_check(engine_id, tier_dir, frames, width, height)?;

    Ok(MochiPreflight {
        frames,
        quant: mochi_tier_quant(tier_dir),
    })
}

/// Resolve a Mochi request into a [`VideoGenInput`] and run it (epic 1788 / sc-11992) — the MLX lane.
///
/// Text-to-video only, true CFG (not distilled, so negative prompt + guidance are real), frames snap
/// to `6k+1`, and the tier dir carries the quant. Progress/heartbeat bridging, cooperative cancel, the
/// forward-progress stall watchdog and the completion metrics all come from the shared
/// [`generate_video`] funnel — the same one Wan/LTX/SVD use — so Mochi inherits the "no background
/// heartbeat during a job, the progress callback IS the keepalive" contract without re-implementing it.
///
/// Deliberately thin: every pre-load decision lives in [`mochi_precheck`] (before the download) or
/// [`mochi_preflight`] (after it), each a pure seam its own tests assert directly.
///
/// The ORDER of those two against the `ensure_mochi_*_present` fetches is the one decision this arm
/// itself owns — the pre-check must precede the download to be worth having (sc-12322) — and it is no
/// longer unpinned: `generate_mochi_using`'s tests (sc-12318) drive this whole body against a stub
/// engine, including `generate_mochi_using_refuses_before_paying_for_the_tier_download`, which fails if
/// the pre-check is moved after the fetches.
#[cfg(target_os = "macos")]
pub(super) async fn generate_mochi(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    generate_mochi_using(
        api,
        settings,
        job,
        request,
        engine_id,
        backend,
        crate::inference_runtime::load,
    )
    .await
}

/// [`generate_mochi`] with the engine loader supplied by the caller (sc-12318) — the seam that makes
/// this arm's pre-load decisions assertable.
///
/// Everything `generate_mochi` does lives here; `generate_mochi` is the one-line delegation that binds
/// the real `inference_runtime::load`. That delegation carries no logic, so the only thing it can drift
/// on is the loader itself — whereas the decisions below are invisible in the output when wrong (a
/// mis-coerced count is a hard engine reject; a skipped gate is a `SIGKILL`; a pre-check that runs too
/// late is a ~13-20 GiB download the answer never depended on), and are covered by
/// `generate_mochi_using_*`.
#[cfg(target_os = "macos")]
pub(super) async fn generate_mochi_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    engine_id: &'static str,
    backend: &str,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<(DecodedVideo, Value)> {
    // Refuse a job that cannot possibly fit BEFORE paying for its tier download (sc-12322). Runs on a
    // weights FLOOR (the already-present shared co-requisites), so it only ever refuses when the full
    // gate below would too — see `mochi_precheck`. MUST stay ahead of the fetches below: that ordering
    // is the whole point, and `generate_mochi_using_refuses_before_paying_for_the_tier_download` is
    // what holds it there.
    mochi_precheck(
        &request.model,
        engine_id,
        mochi_precheck_dir(settings, request).as_deref(),
        request.raw_frame_count(),
        request.width,
        request.height,
    )?;
    // A tier the job toggled but never downloaded is fetched before any compute (no-op otherwise).
    ensure_mochi_q8_present(api, settings, job, request).await?;
    ensure_mochi_bf16_present(api, settings, job, request).await?;
    let model_dir = resolve_mochi_model_dir(settings, request)?;
    // Frame lattice + fit gate + tier quant, as one gated unit — see `mochi_preflight`.
    let MochiPreflight { frames, quant } = mochi_preflight(
        &request.model,
        engine_id,
        &model_dir,
        request.raw_frame_count(),
        request.width,
        request.height,
    )?;

    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        // No adapter path on either backend (`supports_lora`/`supports_lokr` = false).
        adapters: Vec::new(),
        // t2v only — `conditioning: []` on the descriptor.
        conditioning: Vec::new(),
        prompt: request.prompt.clone(),
        // Not distilled ⇒ true CFG, so a negative prompt is real conditioning here (contrast the
        // distilled LTX path, which passes `None`).
        negative_prompt: non_empty_negative_prompt(request),
        width: request.width,
        height: request.height,
        frames,
        fps: request.fps,
        // `None` ⇒ the engine's own DEFAULT_STEPS (64) / DEFAULT_GUIDANCE (4.5).
        steps: advanced_opt_u32(request, "steps"),
        guidance: advanced_opt_f32(request, "guidanceScale"),
        seed: resolve_video_seed(request) as u64,
        ..VideoGenInput::default()
    };
    let raw_settings = mochi_raw_settings(request, quant);
    let decoded = generate_video_using(
        api,
        settings,
        job,
        backend,
        &request.advanced,
        input,
        load_generator,
    )
    .await?;
    Ok((decoded, raw_settings))
}
