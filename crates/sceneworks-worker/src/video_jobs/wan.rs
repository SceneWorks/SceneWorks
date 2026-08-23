#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use super::ltx::resolve_keyframe_conditioning;
#[allow(unused_imports)]
use super::prelude::*;
#[cfg(target_os = "macos")]
use super::{
    bernini::{bernini_engine_id, resolve_bernini_model_dir},
    krea_realtime::{krea_realtime_engine_id, resolve_krea_realtime_tier_dir_and_quant},
    ltx::{ltx_engine_id, resolve_ltx_model_dir},
    mochi::{mochi_engine_id, resolve_mochi_model_dir},
    scail2::{resolve_scail2_model_dir, scail2_engine_id},
    svd::{resolve_svd_model_dir, svd_engine_id},
};
#[cfg(target_os = "macos")]
use super::{ltx::resolve_clip_media_path, vace::FRAME_PAD_COLOR};

// ---------------------------------------------------------------------------
// Real MLX Wan2.2 generation (macOS, via mlx-gen-wan, sc-3034): T2V/TI2V (5B
// dense, z48 VAE), T2V/I2V (A14B dual-expert MoE) + MoE/Lightning LoRA. Decodes
// the engine's `GenerationOutput::Video { frames, fps, audio: None }` into a
// `DecodedVideo` and reuses the [`encode_media`] pipeline above. LTX (sc-3035) and
// every other model keep the procedural stub.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Wan asset (mirrors the image `mlx_*` convention).
#[cfg(target_os = "macos")]
pub(super) const WAN_ADAPTER: &str = "mlx_wan";

/// Resolve a Wan request into a [`VideoGenInput`] and run it (sc-3034).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_wan(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let (steps, guidance) = wan_sampling(engine_id, request);
    let negative_prompt = non_empty_negative_prompt(request);
    let conditioning = match request.mode.as_str() {
        "extend_clip" | "video_bridge" => {
            resolve_wan_clip_conditioning(api, settings, job, request, project_path, engine_id)
                .await?
        }
        _ => resolve_wan_conditioning(settings, request, project_path, engine_id)?,
    };
    ensure_wan_tier_present(api, settings, job, request).await?;
    ensure_wan_lightning_present(api, settings, job, request, engine_id).await?;
    let (model_dir, quant) = if wan_tier_repo(&request.model).is_some() {
        resolve_wan_tier_dir_and_quant(settings, request, engine_id)?
    } else {
        (
            resolve_wan_model_dir(settings, &request.model, engine_id)?,
            resolve_wan_quant(request),
        )
    };
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        adapters: resolve_wan_adapters(settings, request, engine_id)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps,
        guidance,
        seed: resolve_video_seed(request) as u64,
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}

/// Raw-settings recorded on a real MLX Wan asset: the request's `advanced` knobs plus
/// the real-inference markers (mirrors the image `mlx_raw_settings`). Also records the
/// effective sampler the worker actually dispatched (sc-4997) — the 5B interim default / the
/// 14B Lightning preset — so the chosen steps/CFG is inspectable on the asset, not silent.
#[cfg(target_os = "macos")]
pub(super) fn wan_raw_settings(request: &VideoRequest, engine_id: &str) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    let (steps, guidance) = wan_sampling(engine_id, request);
    if let Some(steps) = steps {
        raw.insert("effectiveSteps".to_owned(), json!(steps));
    }
    if let Some(guidance) = guidance {
        raw.insert("effectiveGuidanceScale".to_owned(), json!(guidance));
    }
    Value::Object(raw)
}

/// SceneWorks Wan model id → mlx-gen registry id, or `None` if `model` is not a Wan
/// family id this worker serves.
#[cfg(target_os = "macos")]
pub(super) fn wan_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "wan_2_2" => Some("wan2_2_ti2v_5b"),
        "wan_2_2_t2v_14b" => Some("wan2_2_t2v_14b"),
        "wan_2_2_i2v_14b" => Some("wan2_2_i2v_14b"),
        _ => None,
    }
}

/// Whether the linked Wan engine can serve this request now: a Wan model id with
/// resolvable on-disk weights. Off macOS / non-Wan / weights-absent → the stub
/// (mirrors the image `mlx_available` weights gate).
/// Fail-loud gate for the stub fallback (sc-4176): when the requested model id
/// maps to an MLX video engine family (Wan/LTX/SVD) but its weights/snapshot
/// can't be resolved, surface the resolver's precise re-download error instead
/// of silently degrading the job to procedural stub output. Non-engine model
/// ids pass through (the stub is their intended path).
#[cfg(target_os = "macos")]
pub(crate) fn ensure_video_engine_weights(
    request: &VideoRequest,
    settings: &Settings,
) -> WorkerResult<()> {
    if let Some(engine_id) = wan_engine_id(&request.model) {
        resolve_wan_model_dir(settings, &request.model, engine_id)?;
    }
    if ltx_engine_id(&request.model).is_some() {
        resolve_ltx_model_dir(settings, request)?;
    }
    if svd_engine_id(&request.model).is_some() {
        if request.source_asset_id.is_none() {
            return Err(WorkerError::InvalidPayload(
                "SVD image-to-video requires a source image asset.".to_owned(),
            ));
        }
        resolve_svd_model_dir(settings)?;
    }
    if bernini_engine_id(&request.model).is_some() {
        resolve_bernini_model_dir(settings)?;
    }
    if scail2_engine_id(&request.model).is_some() {
        resolve_scail2_model_dir(settings)?;
    }
    // Mochi 1 (epic 1788 / sc-11992). Without this arm a Mochi job whose weights don't resolve falls
    // to `VideoRoute::Stub` and the user is handed a PROCEDURAL FAKE VIDEO instead of the resolver's
    // precise "download the tier" / "the shared components are missing" error — exactly the silent
    // degradation sc-4176 added this gate to prevent.
    if mochi_engine_id(&request.model).is_some() {
        resolve_mochi_model_dir(settings, request)?;
    }
    // Krea Realtime 14B (epic 8431 / sc-8443). Without this arm a Krea job whose weights don't resolve
    // falls to `VideoRoute::Stub` and the user is handed a PROCEDURAL FAKE VIDEO instead of the
    // resolver's precise "download the snapshot" error — the silent degradation sc-4176 added this gate
    // to prevent. It asks the TIER resolver, not `resolve_krea_realtime_model_dir` (sc-15258): every
    // published Krea file lives under a `q4/`/`q8/`/`bf16/` prefix, so a resolvable snapshot ROOT is no
    // evidence of loadable weights — a torn install would otherwise pass this gate and, since
    // `krea_realtime_available` refuses it, get the fake clip anyway.
    if krea_realtime_engine_id(&request.model).is_some() {
        resolve_krea_realtime_tier_dir_and_quant(settings, request)?;
    }
    // MiniMax-H3 (epic 17137 / sc-19508). Without an arm here a MiniMax-H3 job whose engine or
    // weights are absent falls to `VideoRoute::Stub` and the user is handed a PROCEDURAL FAKE CLIP
    // — silently, since a generated-looking mp4 at the requested geometry is indistinguishable from
    // a real render until watched. That is the degradation sc-4176 added this gate to prevent, and
    // the same arm Mochi and Krea each needed.
    //
    // sc-17159 filled this slot with an UNCONDITIONAL refusal carrying a hard-coded "not in the
    // pinned inference revision" string, because at that commit there was no dispatch arm to fall
    // through to. sc-19508 substitutes the real gate rather than deleting the guard: it now asks
    // the REGISTRY whether the engine is linked, checks the conditioning shape against the entry's
    // DiT partition, and then runs the real tier + shared-component resolver — so a job still fails
    // loudly at the current pin, an unprovisioned install still fails loudly after the pin bump,
    // and each failure names its own cause instead of one string covering all three.
    //
    // `crates/sceneworks-worker/src/pinned_engine_geometry.rs` still carries the two mechanisms
    // that force the GEOMETRY tie-in to be revisited on a pin move: `REV_WITHOUT_MINIMAX_H3` and
    // `minimax_h3_arrival_tripwire`. Neither is touched here — this arm no longer depends on the
    // pin being any particular revision, which is the point.
    super::minimax_h3::ensure_minimax_h3_renderable(request, settings)?;
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn wan_available(request: &VideoRequest, settings: &Settings) -> bool {
    match wan_engine_id(&request.model) {
        Some(engine_id) => resolve_wan_model_dir(settings, &request.model, engine_id).is_ok(),
        None => false,
    }
}

/// Resolve the converted MLX snapshot directory for a Wan model (mirrors the Python
/// `_resolve_wan_mlx`): an env override, then the app-managed `<data>/models/mlx/<id>`,
/// then (T2V-14B only) the turnkey HF MLX snapshot. Errors clearly if none is present.
#[cfg(target_os = "macos")]
pub(super) fn resolve_wan_model_dir(
    settings: &Settings,
    model: &str,
    _engine_id: &str,
) -> WorkerResult<PathBuf> {
    let (env, local_id, hf_repo): (&str, &str, Option<&str>) = match model {
        "wan_2_2" => (
            "SCENEWORKS_MLX_WAN5B_DIR",
            "wan_2_2",
            Some("SceneWorks/wan2.2-ti2v-5b-mlx"),
        ),
        "wan_2_2_t2v_14b" => (
            "SCENEWORKS_MLX_WAN14B_T2V_DIR",
            "wan_2_2_t2v_14b",
            Some("SceneWorks/wan2.2-t2v-a14b-mlx"),
        ),
        "wan_2_2_i2v_14b" => (
            "SCENEWORKS_MLX_WAN14B_I2V_DIR",
            "wan_2_2_i2v_14b",
            Some("SceneWorks/wan2.2-i2v-a14b-mlx"),
        ),
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "not a Wan model: {other}"
            )))
        }
    };
    if let Some(dir) = local_mlx_dir(settings, env, local_id) {
        return Ok(dir);
    }
    if let Some(repo) = hf_repo {
        if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, repo) {
            return Ok(dir);
        }
    }
    Err(WorkerError::InvalidPayload(format!(
        "{model}: no MLX weights found. Convert/download the Wan checkpoint into {}{}.",
        settings
            .data_dir
            .join("models")
            .join("mlx")
            .join(local_id)
            .display(),
        hf_repo
            .map(|repo| format!(" (or download the turnkey repo {repo})"))
            .unwrap_or_default(),
    )))
}

/// A locally-converted MLX dir for the model (env override, then
/// `<data>/models/mlx/<id>`), counted only when it holds a `config.json` — mirrors the
/// Python `_local_mlx_dir`, so a locally-quantized conversion supersedes a turnkey download.
#[cfg(target_os = "macos")]
pub(super) fn local_mlx_dir(settings: &Settings, env: &str, local_id: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(override_dir) = std::env::var(env) {
        let trimmed = override_dir.trim();
        if !trimmed.is_empty() {
            candidates.push(PathBuf::from(trimmed));
        }
    }
    candidates.push(settings.data_dir.join("models").join("mlx").join(local_id));
    candidates
        .into_iter()
        .find(|dir| dir.join("config.json").is_file())
}

/// The turnkey SceneWorks Wan2.2 **T2V-A14B** MLX repo (sc-9942, epic 8506). Hosts the quant matrix
/// as self-contained tier subdirs `q4/` (default) + `q8/` + `bf16/`, each a COMPLETE dual-expert
/// snapshot (both MoE experts + UMT5 T5 encoder + z16 VAE + tokenizer + `config.json`). This replaces
/// the flat dense-bf16 layout (which quantized at LOAD, staging the full bf16 experts first); the
/// worker now descends into the chosen tier so a pre-packed snapshot loads with no install-time
/// convert peak. The flat root files are kept for back-compat with already-shipped workers that
/// resolve the repo root (a cleanup story drops them once those age out); a new worker only ever
/// resolves the tier subdirs.
#[cfg(target_os = "macos")]
pub(super) const WAN_T2V_14B_REPO: &str = "SceneWorks/wan2.2-t2v-a14b-mlx";

/// Pinned revision for [`WAN_T2V_14B_REPO`] (mirrors [`LTX_BUNDLE_REVISION`], sc-9879). The repo is a
/// hard-coded const — no manifest/payload override reaches the on-demand `q8/*` + `bf16/*` fetches —
/// so pulling the mutable `main` branch would let an upstream re-push silently swap a checkpoint we
/// load. Pin the exact commit that adds the `q4/`/`q8/`/`bf16/` tier subdirs for defense-in-depth
/// (the native downloader still verifies each file's own hash on download). This is the commit that added the
/// `q4/`/`q8/`/`bf16/` tier subdirs (sc-9942).
#[cfg(target_os = "macos")]
pub(super) const WAN_T2V_14B_REVISION: &str = "991eb255c544bbb2e1f1e07da4355c2f0a5337b7";

/// The turnkey SceneWorks Wan2.2 **I2V-A14B** MLX repo (sc-9943, epic 8506). The image→video sibling
/// of [`WAN_T2V_14B_REPO`]: same self-contained `q4/`/`q8/`/`bf16/` tier layout (both MoE experts +
/// UMT5 T5 + z16 VAE + tokenizer + `config.json`), differing only in the experts' `in_dim` (36
/// image-concat conditioning vs 16 text-only). The worker descends into the chosen tier so a
/// pre-packed snapshot loads with no install-time convert peak; the legacy flat root files stay for
/// already-shipped workers.
#[cfg(target_os = "macos")]
pub(super) const WAN_I2V_14B_REPO: &str = "SceneWorks/wan2.2-i2v-a14b-mlx";

/// Pinned revision for [`WAN_I2V_14B_REPO`] (mirrors [`WAN_T2V_14B_REVISION`]). The commit that adds
/// the `q4/`/`q8/`/`bf16/` tier subdirs to the I2V-A14B repo (sc-9943); pinning the exact commit (not
/// the mutable `main`) stops an upstream re-push from silently swapping a checkpoint the on-demand
/// `q8/*` + `bf16/*` fetch loads (the native downloader still verifies each file's own hash on download).
#[cfg(target_os = "macos")]
pub(super) const WAN_I2V_14B_REVISION: &str = "c6c786170031eccc3a1fac0f98f1ad4ff988271e";

/// The turnkey SceneWorks Wan2.2 **TI2V-5B** MLX repo (sc-9941, epic 8506). The single-expert sibling
/// of the A14B repos: same self-contained `q4/`/`q8/`/`bf16/` tier layout, but ONE transformer
/// (`model.safetensors`) rather than the dual `high/low_noise_model` MoE experts (still + UMT5 T5 +
/// z16 VAE + tokenizer + `config.json`). The worker descends into the chosen tier so a pre-packed
/// snapshot loads with no install-time convert peak; the legacy flat root files stay for
/// already-shipped workers (cleanup sc-9977).
#[cfg(target_os = "macos")]
pub(super) const WAN_TI2V_5B_REPO: &str = "SceneWorks/wan2.2-ti2v-5b-mlx";

/// Pinned revision for [`WAN_TI2V_5B_REPO`] (mirrors [`WAN_T2V_14B_REVISION`]). The commit that adds
/// the `q4/`/`q8/`/`bf16/` tier subdirs to the TI2V-5B repo (sc-9941); pinning the exact commit (not
/// the mutable `main`) stops an upstream re-push from silently swapping a checkpoint the on-demand
/// `q8/*` + `bf16/*` fetch loads (the native downloader still verifies each file's own hash on download).
#[cfg(target_os = "macos")]
pub(super) const WAN_TI2V_5B_REVISION: &str = "bb1b055249614cf9d7cf4373fbdbc184b77dee88";

/// Pinned commit revision for the A14B Lightning distill-LoRA repo `lightx2v/Wan2.2-Lightning` (sc-11168 /
/// F-007 — completes the sc-9879 rollout). Both the MLX (`ensure_wan_lightning_present`) and candle
/// (formerly separate) self-heal fetches were pulling the mutable `main` branch, so an
/// upstream re-push (or a compromised token) could silently swap the high/low distill weights we load.
/// Pin the exact commit for defense-in-depth (the native downloader still verifies each file's own hash on
/// download). Shared by BOTH lanes so the twins agree. Gated to the lanes that actually fetch it (macOS
/// MLX or the candle build) so a Linux-non-candle build doesn't flag it dead.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const WAN_LIGHTNING_REVISION: &str = "18bccf8884ec0a078eed79785eb4ef13ea16ce1e";

/// Architecture-specific directory in `lightx2v/Wan2.2-Lightning`.
///
/// This mapping is shared by both backends and by both the cache-healing and resolution paths so a
/// new architecture cannot silently fetch one pair and load another.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn wan_lightning_subdir(engine_id: &str) -> Option<&'static str> {
    match engine_id {
        "wan2_2_t2v_14b" => Some("Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1"),
        "wan2_2_i2v_14b" => Some("Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"),
        _ => None,
    }
}

/// The files that make an **A14B** (dual-expert MoE) Wan tier subdir COMPLETE: both experts + the T5
/// encoder + VAE + tokenizer + `config.json`.
#[cfg(target_os = "macos")]
pub(super) const WAN_A14B_TIER_FILES: &[&str] = &[
    "high_noise_model.safetensors",
    "low_noise_model.safetensors",
    "t5_encoder.safetensors",
    "vae.safetensors",
    "tokenizer.json",
    "config.json",
];

/// The files that make a **TI2V-5B** (single-expert) Wan tier subdir COMPLETE: the one transformer
/// (`model.safetensors`) + the T5 encoder + VAE + tokenizer + `config.json`.
#[cfg(target_os = "macos")]
pub(super) const WAN_TI2V_5B_TIER_FILES: &[&str] = &[
    "model.safetensors",
    "t5_encoder.safetensors",
    "vae.safetensors",
    "tokenizer.json",
    "config.json",
];

/// The tier-completeness file set for a Wan quant-matrix model: the single-expert TI2V-5B ships one
/// `model.safetensors`, the A14B MoE models ship the two `high/low_noise_model.safetensors` experts.
#[cfg(target_os = "macos")]
pub(super) fn wan_tier_files(model: &str) -> &'static [&'static str] {
    if model == "wan_2_2" {
        WAN_TI2V_5B_TIER_FILES
    } else {
        WAN_A14B_TIER_FILES
    }
}

/// Map a Wan quant-matrix video model id to its `(quant-matrix repo, pinned revision)` for the
/// on-demand tier fetch, or `None` for a model with no hosted tier matrix. The TI2V-5B (sc-9941),
/// T2V-A14B (sc-9942) and I2V-A14B (sc-9943) turnkeys host the SAME self-contained
/// `q4/`/`q8/`/`bf16/` tier layout (epic 8506); only the repo + pinned commit (and the single- vs
/// dual-expert file set, see [`wan_tier_files`]) differ, so the whole tier-resolve/fetch path is
/// shared and keyed only here. `request.model` is `"wan_2_2"` for the TI2V-5B engine.
#[cfg(target_os = "macos")]
pub(super) fn wan_tier_repo(model: &str) -> Option<(&'static str, &'static str)> {
    match model {
        "wan_2_2" => Some((WAN_TI2V_5B_REPO, WAN_TI2V_5B_REVISION)),
        "wan_2_2_t2v_14b" => Some((WAN_T2V_14B_REPO, WAN_T2V_14B_REVISION)),
        "wan_2_2_i2v_14b" => Some((WAN_I2V_14B_REPO, WAN_I2V_14B_REVISION)),
        _ => None,
    }
}

/// Parse `advanced.mlxQuantize` (int or numeric string) for the Wan quant-matrix tier selector.
#[cfg(target_os = "macos")]
fn wan_quant_bits(request: &VideoRequest) -> Option<i64> {
    request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
}

/// The Wan2.2 quant-matrix tier search order for a request — preferred tier first, then the
/// always-smaller fallback tiers so a repo missing the preferred subdir still loads (mirrors
/// [`ltx_bundle_tier_order`]): `mlxQuantize <= 0` ⇒ `bf16`, `>= 8` ⇒ `q8`, an explicit `1..=4` ⇒
/// `q4`, and — with NO explicit `mlxQuantize` — the **`q4`** default (q4-first).
///
/// The video lane deliberately does NOT take epic 10721's app-wide **Q8** default (sc-10726); it keeps
/// the pre-sc-10726 q4-first default (sc-10859). Rationale: the MLX video lane has no user Q8 lever
/// (Video Studio's quant control targets the GGUF import path), so a silent Q8 default gives no UI-accessible
/// quality benefit and only ever surfaces as an *accidental* default when the Q8 tier landed on disk via
/// a side lane — where it risks a video-runtime OOM at heavy res/frame counts (the install-fit clamp
/// doesn't help: the sc-8516 budget is 1024²-image-calibrated). Q8/bf16 stay reachable on an explicit
/// pick; `bf16` stays OUT of the default order so a default job never pulls the huge dense tier. The
/// precise "highest tier that fits the video-runtime budget" default is the deferred sc-10733 (S8).
#[cfg(target_os = "macos")]
pub(super) fn wan_tier_order(request: &VideoRequest) -> &'static [&'static str] {
    match wan_quant_bits(request) {
        Some(b) if b <= 0 => &["bf16", "q8", "q4"],
        Some(b) if b >= 8 => &["q8", "q4"],
        // No explicit pick (`None`) OR an explicit `1..=4` ⇒ q4-first (sc-10859 video carve-out).
        _ => &["q4", "q8"],
    }
}

/// Whether `dir` is a COMPLETE self-contained Wan2.2 tier snapshot, given the model's expected tier
/// file set (`files`, from [`wan_tier_files`]): the transformer(s), the T5 encoder, VAE, tokenizer,
/// and `config.json`. A partially-downloaded tier fails this so [`wan_tier_subdir`] falls through to
/// a smaller complete tier rather than half-loading.
#[cfg(target_os = "macos")]
pub(super) fn wan_tier_is_complete(dir: &Path, files: &[&str]) -> bool {
    files.iter().all(|file| dir.join(file).is_file())
}

/// Descend a resolved Wan2.2 quant-matrix repo `root` into the requested quant tier subdir
/// (sc-9941 TI2V-5B / sc-9942 T2V / sc-9943 I2V, epic 8506), mirroring [`ltx_bundle_subdir`]. Returns
/// the first COMPLETE tier in [`wan_tier_order`] (all of a model's weights — one transformer for the
/// 5B, both experts for the A14B — live in the SAME subdir, so one resolution covers the model), or
/// `None` when the repo has no complete tier subdir — a legacy flat snapshot, where the caller keeps
/// the root + load-time quant.
#[cfg(target_os = "macos")]
pub(super) fn wan_tier_subdir(root: &Path, request: &VideoRequest) -> Option<PathBuf> {
    let files = wan_tier_files(&request.model);
    wan_tier_order(request)
        .iter()
        .map(|tier| root.join(tier))
        .find(|dir| wan_tier_is_complete(dir, files))
}

/// Resolve the Wan2.2 `(model_dir, load-time quant)` for a generation, descending into the
/// quant-matrix tier subdir when the turnkey ships them (sc-9941 TI2V-5B / sc-9942 T2V / sc-9943 I2V).
/// A pre-packed
/// tier's `config.json` is authoritative — [`WanTransformer::from_weights`] builds the experts at the
/// stored bits and `resolve_load_time_quant` rejects a conflicting `spec.quantize` as a hard error —
/// so a resolved tier loads with `quant = None`: `mlxQuantize` selects WHICH tier, never a load-time
/// requant (the `bf16/` tier is dense, so `None` ⇒ dense too). A legacy flat snapshot (no tier
/// subdirs) keeps today's behavior: load the root and quantize at load per [`resolve_wan_quant`].
#[cfg(target_os = "macos")]
pub(super) fn resolve_wan_tier_dir_and_quant(
    settings: &Settings,
    request: &VideoRequest,
    engine_id: &'static str,
) -> WorkerResult<(PathBuf, Option<Quant>)> {
    let root = resolve_wan_model_dir(settings, &request.model, engine_id)?;
    match wan_tier_subdir(&root, request) {
        Some(tier) => Ok((tier, None)),
        None => Ok((root, resolve_wan_quant(request))),
    }
}

/// On-demand fetch of a non-default Wan2.2 quant-matrix tier subdir (sc-9941 TI2V-5B / sc-9942 T2V /
/// sc-9943 I2V, mirrors [`ensure_ltx_q8_present`] / [`ensure_ltx_bf16_present`]). The macOS default
/// download is the lean `q4/` tier; a job that opts into a heavier tier (`mlxQuantize <= 0` ⇒ `bf16`,
/// `>= 8` ⇒ `q8`) pulls just that subdir from that model's FIXED [`wan_tier_repo`] revision the first
/// time it is requested so [`wan_tier_subdir`] can resolve it. No-op for a model with no hosted tier
/// matrix, a `q4` (default)
/// job, when the repo snapshot isn't downloaded yet (resolve surfaces the clear error), or when the
/// tier is already complete. Fails loud on a real download error — fast, before any compute; a
/// tier that isn't published yet stays absent so resolve falls back to a smaller complete
/// tier.
#[cfg(target_os = "macos")]
pub(super) async fn ensure_wan_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    let Some((repo, revision)) = wan_tier_repo(&request.model) else {
        return Ok(());
    };
    let tier = match wan_quant_bits(request) {
        Some(b) if b <= 0 => "bf16",
        Some(b) if b >= 8 => "q8",
        // q4 default — ships with the base install, nothing to fetch on demand.
        _ => return Ok(()),
    };
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, repo) else {
        return Ok(());
    };
    if wan_tier_is_complete(&root.join(tier), wan_tier_files(&request.model)) {
        return Ok(());
    }
    let files = vec![format!("{tier}/*")];
    crate::model_jobs::ensure_hf_files_cached(api, settings, job, repo, revision, &files)
        .await
        .map(|_| ())
}

/// On-demand fetch of the 4-step Lightning distill LoRA pair (`lightx2v/Wan2.2-Lightning`) for the
/// A14B MoE models (sc-10030). Normally the pair installs as a manifest `coRequisite` alongside the
/// model (sc-9696), but a worker that installed the model BEFORE the coRequisite was added has the
/// tiers without the LoRA — and [`resolve_wan_adapters`] then hard-errors when the toggle is on. This
/// self-heals that case: it pulls just the per-architecture high/low pair the first time a gen needs
/// it (twin of [`ensure_wan_tier_present`] / the candle `ensure_qwen_lightning_lora_cached`). No-op
/// when the Lightning toggle is off (sc-10047 — the native multi-step recipe needs no LoRA), for a
/// non-A14B engine, or when the pair is already cached. A pair still missing after the fetch makes
/// resolve surface the clear "fetch it via the model manager" error. Fails loud on a real download error —
/// fast, before any compute.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn ensure_wan_lightning_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    engine_id: &str,
) -> WorkerResult<()> {
    // sc-10047: Lightning is a default-on toggle now. When the job opted out (`advanced.lightning`
    // = false), the native multi-step CFG recipe runs with no Lightning adapter, so we need nothing
    // here. Default-on (or explicitly on) still wants the pair present and self-heals if absent.
    if !wan_lightning_on(engine_id, request) {
        return Ok(());
    }
    // Per-architecture subdir (NOT cross-compatible, sc-4997); must match `resolve_lightning_loras`.
    let Some(subdir) = wan_lightning_subdir(engine_id) else {
        return Ok(());
    };
    const REPO: &str = "lightx2v/Wan2.2-Lightning";
    // Fast path: both halves already materialized in the hub cache (the common case after install).
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, REPO) {
        let base = snapshot.join(subdir);
        if base.join("high_noise_model.safetensors").is_file()
            && base.join("low_noise_model.safetensors").is_file()
        {
            return Ok(());
        }
    }
    let files = vec![
        format!("{subdir}/high_noise_model.safetensors"),
        format!("{subdir}/low_noise_model.safetensors"),
    ];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        REPO,
        WAN_LIGHTNING_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// The 4-step Lightning distill LoRA pair (high/low) for an A14B MoE model
/// (`lightx2v/Wan2.2-Lightning`, the rank-64 Seko distill). The subdir is architecture-specific:
/// T2V-A14B (V1.1) and I2V-A14B (V1) ship distinct LoRAs that are NOT cross-compatible (sc-4997).
/// Errors if not downloaded / the per-architecture subdir is missing.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_lightning_loras(
    settings: &Settings,
    engine_id: &str,
) -> WorkerResult<(PathBuf, PathBuf)> {
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, "lightx2v/Wan2.2-Lightning")
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{engine_id}: the Lightning distill LoRA (lightx2v/Wan2.2-Lightning) is not \
                 downloaded — fetch it via the model manager"
            ))
        })?;
    let base = wan_lightning_subdir(engine_id).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{engine_id}: no Lightning distill LoRA — only the A14B MoE models bake Lightning"
        ))
    })?;
    let high = snapshot.join(base).join("high_noise_model.safetensors");
    let low = snapshot.join(base).join("low_noise_model.safetensors");
    for file in [&high, &low] {
        if !file.is_file() {
            return Err(WorkerError::InvalidPayload(format!(
                "{engine_id}: Lightning LoRA file missing: {}",
                file.display()
            )));
        }
    }
    Ok((high, low))
}

/// The `.low_noise.safetensors` sibling of a Wan A14B MoE high-noise LoRA file, or
/// `None` when the file is not the high-noise half of a pair (port of the Python
/// `wan_moe_low_noise_sibling`; case-insensitive `.high_noise.safetensors` suffix).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn wan_moe_low_noise_sibling(primary: &Path) -> Option<PathBuf> {
    const HIGH: &str = ".high_noise.safetensors";
    let name = primary.file_name()?.to_str()?;
    if !name.to_ascii_lowercase().ends_with(HIGH) {
        return None;
    }
    let stem = &name[..name.len() - HIGH.len()];
    let sibling = primary.with_file_name(format!("{stem}.low_noise.safetensors"));
    sibling.is_file().then_some(sibling)
}

/// Build the adapter specs for a Wan generation (sc-3034): the Lightning distill pair
/// (both A14B MoE models — T2V + I2V — tagged high/low, sc-4997) followed by the user LoRAs.
/// On the MoE models a user
/// `*.high_noise.safetensors` with a `.low_noise` sibling tags high→High / low→Low; a
/// single-file LoRA is shared (both experts on MoE, the single model on the 5B). peft LoKr AND
/// third-party LyCORIS (LoHa / non-peft LoKr) both apply on the MLX Wan/LTX paths now (epic 3641,
/// sc-3671) — `classify_adapter` returns `Lora` for third-party and the engine detects + merges it.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_wan_adapters(
    settings: &Settings,
    request: &VideoRequest,
    engine_id: &str,
) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let is_wan_a14b = engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b";
    let is_moe = is_wan_a14b || matches!(engine_id, "bernini" | "wan2_2_vace_fun_14b");
    let mut specs: Vec<AdapterSpec> = Vec::new();

    // Lightning distill (both A14B MoE models — T2V + I2V, sc-4997): 4-step, applied per-expert at
    // strength 1.0 through the standard adapter path. As of sc-10047 this is a **default-on toggle**
    // (`advanced.lightning`) rather than mandatory — the mlx-gen additive path (epic 10043) applies
    // it on the quantized tiers, so the pair is added only when the toggle is on. When off, the
    // native multi-step CFG recipe runs ([`wan_sampling`]) with no Lightning adapter. User LoRAs
    // below are honored in both states. The subdir is resolved per architecture (not cross-compatible).
    if is_wan_a14b && wan_lightning_on(engine_id, request) {
        let (high, low) = resolve_lightning_loras(settings, engine_id)?;
        specs.push(moe_adapter(
            high,
            1.0,
            gen_core::AdapterKind::Lora,
            gen_core::MoeExpert::High,
        ));
        specs.push(moe_adapter(
            low,
            1.0,
            gen_core::AdapterKind::Lora,
            gen_core::MoeExpert::Low,
        ));
    }

    for lora in &request.loras {
        let path = crate::image_jobs::lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        let file = resolve_lora_file(
            settings,
            path,
            crate::image_jobs::declared_adapter_file(lora),
        )?;
        let kind = crate::image_jobs::classify_adapter(&file)?;
        let scale = lora_scale(lora);
        match (is_moe, wan_moe_low_noise_sibling(&file)) {
            (true, Some(low)) => {
                // A MoE pair → high half to the high-noise expert, the sibling to the low.
                let low_kind = crate::image_jobs::classify_adapter(&low)?;
                specs.push(moe_adapter(file, scale, kind, gen_core::MoeExpert::High));
                specs.push(moe_adapter(low, scale, low_kind, gen_core::MoeExpert::Low));
            }
            _ => {
                // Single-file → shared (both experts on MoE; the dense single model on the 5B).
                specs.push(AdapterSpec {
                    path: file,
                    scale,
                    kind,
                    pass_scales: None,
                    moe_expert: None,
                });
            }
        }
    }
    Ok(specs)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn moe_adapter(
    path: PathBuf,
    scale: f32,
    kind: gen_core::AdapterKind,
    expert: gen_core::MoeExpert,
) -> AdapterSpec {
    AdapterSpec {
        path,
        scale,
        kind,
        pass_scales: None,
        moe_expert: Some(expert),
    }
}

/// Build the adapter specs for a Wan-VACE generation (sc-3893 worker routing). Unlike the base Wan
/// path, VACE-1.3B is a **single dense** transformer: no Lightning distill, no MoE high/low experts.
/// So every user LoRA/LoKr is applied shared with `moe_expert: None` — the engine `wan_vace` provider
/// merges diffusers-named LoRA/LoKr (mlx-gen #184) and rejects `moe_expert` tags. `classify_adapter`
/// tags SceneWorks peft LoKr as `Lokr` and everything else (incl. third-party LyCORIS LoHa / non-peft
/// LoKr) as `Lora`, which the engine then detects + merges by key sniff (epic 3641). Delegates to
/// the shared [`resolve_dense_adapters`] (sc-8830).
#[cfg(target_os = "macos")]
pub(super) fn resolve_wan_vace_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
    resolve_dense_adapters(settings, request, MAX_JOB_LORAS)
}

/// Build the adapter specs for a SCAIL-2 generation (sc-5451 inference LoRA path, mlx-gen #462).
/// SCAIL-2 is a single **dense** Wan2.1-14B-I2V transformer — like Wan-VACE, no Lightning distill and
/// no MoE high/low experts — so every LoRA is applied shared with `moe_expert: None`. The engine
/// installs a standard `lora_down/up` (PEFT/diffusers/kohya/LoKr) adapter as a forward-time residual
/// over the (Q4/Q8) base; `classify_adapter` tags SceneWorks peft LoKr as `Lokr` and everything else
/// (incl. third-party LyCORIS) as `Lora`. This carries both a user-selected SCAIL-2 LoRA and the
/// bundled Bias-Aware DPO quality LoRA (both surface through `request.loras`). A lightx2v diff-patch
/// "lightning" LoRA installs via the engine's in-place diff-patch merge (sc-5684); selecting it makes
/// the worker apply the step-distill recipe (`scail2_sampling`, sc-5700). Delegates to the shared
/// [`resolve_dense_adapters`] (sc-8830) — the MLX Wan-VACE / SCAIL-2 twin.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_scail2_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
    resolve_dense_adapters(settings, request, MAX_JOB_LORAS)
}

/// The first-frame conditioning for a Wan generation: required for I2V-14B, optional for
/// the TI2V-5B (present → image-conditioned mask-blend, absent → pure T2V), and ignored
/// by the T2V-14B (text-only). Loads `source_asset_id` to an in-memory RGB8 image.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_wan_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &str,
) -> WorkerResult<Vec<Conditioning>> {
    // first_last_frame is Wan-native only on the TI2V-5B mask-blend keyframe path (sc-3357);
    // the routing gate (`video_mode_is_mlx_eligible`) already restricts FLF to `wan_2_2`, but
    // guard here too so a mis-routed 14B MoE job fails clearly instead of silently dropping it.
    if request.mode == "first_last_frame" {
        if engine_id != "wan2_2_ti2v_5b" {
            return Err(WorkerError::InvalidPayload(format!(
                "first_last_frame is only supported on wan_2_2 (TI2V-5B), not {engine_id}."
            )));
        }
        return resolve_keyframe_conditioning(settings, request, project_path);
    }
    if engine_id == "wan2_2_ti2v_5b" {
        match request.mode.as_str() {
            "text_to_video"
                if request.source_asset_id.is_some() || request.last_frame_asset_id.is_some() =>
            {
                return Err(WorkerError::InvalidPayload(
                    "wan_2_2 text-to-video must not carry sourceAssetId or lastFrameAssetId; select image_to_video or first_last_frame explicitly."
                        .to_owned(),
                ));
            }
            "image_to_video" if request.last_frame_asset_id.is_some() => {
                return Err(WorkerError::InvalidPayload(
                    "wan_2_2 image-to-video must not carry lastFrameAssetId; select first_last_frame explicitly."
                        .to_owned(),
                ));
            }
            _ => {}
        }
    }
    let required = engine_id == "wan2_2_i2v_14b"
        || (engine_id == "wan2_2_ti2v_5b" && request.mode == "image_to_video");
    let accepts = required || engine_id == "wan2_2_ti2v_5b";
    if !accepts {
        return Ok(Vec::new());
    }
    match request.source_asset_id.as_deref() {
        Some(asset_id) => {
            let image = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                asset_id,
                project_path,
            )?;
            // Pre-fit to the output W×H by the chosen crop/pad mode (sc-6139) — see
            // `resolve_ltx_conditioning`; without it the provider VAE-encodes a stretched
            // first frame into its channel-concat `y`.
            let image = crate::image_jobs::fit_engine_image(
                image,
                request.width,
                request.height,
                &request.fit_mode,
            )?;
            Ok(vec![Conditioning::Reference {
                image,
                strength: None,
            }])
        }
        None if required => Err(WorkerError::InvalidPayload(format!(
            "{}: image-to-video requires a source image (sourceAssetId).",
            request.model
        ))),
        None => Ok(Vec::new()),
    }
}

/// Which boundary frame of a source clip to extract for Wan-native clip conditioning (sc-3357).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ClipFramePosition {
    /// The clip's first decoded frame (the right-side clip's head for `video_bridge`).
    First,
    /// The clip's last decoded frame (the source tail for `extend_clip` / the left-side clip).
    Last,
}

/// Build the Wan-native boundary [`Conditioning::Keyframe`] set for extend_clip / video_bridge
/// (sc-3357). Wan TI2V-5B has no in-context clip-append path (LTX's IC-LoRA `VideoClip`); its only
/// clip primitive is the single-frame mask-blend `Keyframe` (the same one Wan FLF rides). So the
/// faithful Wan-native form — matching the torch Wan reference, which routed these modes to plain
/// i2v (`_pipeline_kind` → `"image"`, never IC-LoRA/VACE) — pins the clip *boundary* frame(s):
/// - **extend_clip** → the source clip's last frame pinned at latent frame `0` (continue from it),
///   strength `videoConditioningStrength`.
/// - **video_bridge** → the left clip's last frame at `0` (`videoConditioningStrength`) + the right
///   clip's first frame at latent frame `-1` (the engine's negative-from-end index), strength
///   `bridgeRightVideoConditioningStrength`. Mechanically identical to first_last_frame.
///
/// Both strengths default to `1.0` (fully pinned), mirroring [`build_video_clip_conditioning`] and
/// the torch `_advanced_float` defaults. This is the single-frame fidelity ceiling for Wan; richer
/// motion-tail continuity is the LTX IC-LoRA path or native Wan-VACE (sc-3385 routing matrix).
#[cfg(target_os = "macos")]
pub(super) fn build_wan_boundary_conditioning(
    request: &VideoRequest,
    left_frame: Image,
    right_frame: Option<Image>,
) -> WorkerResult<Vec<Conditioning>> {
    let mut conditioning = vec![Conditioning::Keyframe {
        image: left_frame,
        frame_idx: 0,
        strength: advanced::f32(&request.advanced, "videoConditioningStrength", 1.0),
    }];
    if request.mode == "video_bridge" {
        let right = right_frame.ok_or_else(|| {
            WorkerError::InvalidPayload(
                "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                    .to_owned(),
            )
        })?;
        conditioning.push(Conditioning::Keyframe {
            image: right,
            frame_idx: -1,
            strength: advanced::f32(
                &request.advanced,
                "bridgeRightVideoConditioningStrength",
                1.0,
            ),
        });
    }
    Ok(conditioning)
}

/// Resolve extend_clip / video_bridge into Wan-native boundary [`Conditioning::Keyframe`]s
/// (sc-3357). Wan-native clip conditioning is **only** the TI2V-5B mask-blend keyframe path, so
/// guard the engine (the routing gate `video_mode_is_mlx_eligible` already restricts these to
/// `wan_2_2`, but fail clearly here too if a 14B MoE job is mis-routed). Extracts the boundary
/// frame(s) — the source clip's last frame (+ the right clip's first frame for bridge) — then maps
/// them via [`build_wan_boundary_conditioning`]. Unlike the LTX path this needs **no** IC-LoRA.
#[cfg(target_os = "macos")]
pub(super) async fn resolve_wan_clip_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &str,
) -> WorkerResult<Vec<Conditioning>> {
    if engine_id != "wan2_2_ti2v_5b" {
        return Err(WorkerError::InvalidPayload(format!(
            "{} is only supported on wan_2_2 (TI2V-5B), not {engine_id}.",
            request.mode.replace('_', " ")
        )));
    }
    let left_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} requires a source clip (sourceClipAssetId).",
            request.mode.replace('_', " ")
        ))
    })?;
    let left_frame = extract_clip_boundary_frame(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        left_id,
        request.width,
        request.height,
        ClipFramePosition::Last,
    )
    .await?;
    let right_frame = if request.mode == "video_bridge" {
        let right_id = request
            .bridge_right_clip_asset_id
            .as_deref()
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
        Some(
            extract_clip_boundary_frame(
                api,
                settings,
                job,
                &request.project_id,
                project_path,
                right_id,
                request.width,
                request.height,
                ClipFramePosition::First,
            )
            .await?,
        )
    } else {
        None
    };
    build_wan_boundary_conditioning(request, left_frame, right_frame)
}

/// Decode a single boundary frame (first or last) of a source clip into an [`Image`], fit to the
/// output `width`×`height` by contain+pad (letterbox, `FRAME_PAD_COLOR`) so a clip whose aspect
/// differs from the output is not distorted — sc-6229, matching the `load_source_video_frames`
/// recipe (sc-3357, the Wan boundary-keyframe conditioning input). The last frame
/// uses ffmpeg `-sseof` to seek near the end + `-update 1` so each decoded frame overwrites the lone
/// output, leaving the final frame; the first frame is a plain `-frames:v 1`. Extracted via the
/// shared [`run_ffmpeg`] (binary resolution + heartbeat/cancel), then loaded off the async runtime.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn extract_clip_boundary_frame(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_id: &str,
    project_path: &Path,
    asset_id: &str,
    width: u32,
    height: u32,
    position: ClipFramePosition,
) -> WorkerResult<Image> {
    let clip_path = resolve_clip_media_path(settings, project_id, asset_id, project_path)?;
    let frames_dir = project_path
        .join("assets")
        .join(".cond_clips")
        .join(Uuid::new_v4().simple().to_string());
    tokio::fs::create_dir_all(&frames_dir).await?;
    let out = frames_dir.join("boundary.png");
    let mut args = vec!["ffmpeg".to_owned(), "-nostdin".to_owned(), "-y".to_owned()];
    if position == ClipFramePosition::Last {
        // Seek to ~2s before EOF; short clips clamp to the start (whole clip decoded). `-update 1`
        // overwrites the single output per frame, so the final decoded frame is what remains.
        args.push("-sseof".to_owned());
        args.push("-2".to_owned());
    }
    args.push("-i".to_owned());
    args.push(clip_path.display().to_string());
    args.push("-vf".to_owned());
    // Contain+pad (letterbox) to the output dims so a source clip whose aspect differs from the
    // requested W×H is not stretched (sc-6229); reuses the `FRAME_PAD_COLOR` recipe.
    args.push(format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,\
         pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={FRAME_PAD_COLOR},format=rgb24"
    ));
    if position == ClipFramePosition::Last {
        args.push("-update".to_owned());
        args.push("1".to_owned());
    } else {
        args.push("-frames:v".to_owned());
        args.push("1".to_owned());
    }
    args.push(out.display().to_string());
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let result = run_ffmpeg(args, Some(ctx)).await;
    let load = async {
        result?;
        let path = out.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<Image> {
            let decoded = crate::image_decode::decode_image_any(&path)
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "boundary conditioning frame {}: {error}",
                        path.display()
                    ))
                })?
                .to_rgb8();
            Ok(Image {
                width: decoded.width(),
                height: decoded.height(),
                pixels: decoded.into_raw(),
            })
        })
        .await
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?
    };
    let frame = load.await;
    let _ = tokio::fs::remove_dir_all(&frames_dir).await;
    frame
}

/// Map `advanced.mlxQuantize` to a quant level (≤0 → dense, ≤4 → Q4, else Q8). Absent →
/// `None`: dense bf16, or the engine builds it quantized from a pre-quantized snapshot.
#[cfg(target_os = "macos")]
pub(super) fn resolve_wan_quant(request: &VideoRequest) -> Option<Quant> {
    let bits = request.advanced.get("mlxQuantize").and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    })?;
    match bits {
        b if b <= 0 => None,
        b if b <= 4 => Some(Quant::Q4),
        _ => Some(Quant::Q8),
    }
}

/// Interim step count for the dense TI2V-5B until a 5B distill LoRA ships (sc-4999): half the
/// engine's 40-step default, so an out-of-the-box 1280×720 job no longer runs the ~40-min /
/// GPU-wedging 40-step+CFG schedule that wedged the GPU (sc-4986 / sc-4997). CFG is retained
/// (no 5B distill exists, so dropping it would hurt prompt adherence); the user can still dial
/// `steps`/`guidanceScale` lower from VideoStudio, and the engine pre-flight guard (sc-4986) is
/// the memory backstop. The full few-step / no-CFG preset lands once the 5B distill LoRA exists.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const WAN5B_INTERIM_STEPS: u32 = 20;

/// An optional positive-integer `advanced` knob (`steps`); accepts a number or a numeric string.
/// Shared by the MLX path and the candle video lane (sc-5097).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn advanced_opt_u32(request: &VideoRequest, key: &str) -> Option<u32> {
    request.advanced.get(key).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as u32)
    })
}

/// An optional float `advanced` knob (`guidanceScale`); accepts a number or a numeric string.
/// Shared by the MLX path and the candle video lane (sc-5097).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn advanced_opt_f32(request: &VideoRequest, key: &str) -> Option<f32> {
    request.advanced.get(key).and_then(|value| {
        value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as f32)
    })
}

/// `true` if the A14B MoE Lightning distill is engaged for this request (sc-10047). The Lightning
/// 4-step distill is now a **default-on toggle** (`advanced.lightning`) rather than mandatory: the
/// mlx-gen additive path (epic 10043) applies the high/low pair on the quantized tiers, so a job can
/// opt out and run the native multi-step CFG recipe instead. Only the two A14B MoE models (T2V + I2V)
/// bake Lightning — for every other engine (the dense 5B, non-Wan) this is irrelevant and returns
/// `false`. Backward compatible: an absent flag on an A14B job defaults to `true` (the prior
/// always-on behavior). A strict-bool `false` opts out; `true` (or absent) opts in.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn wan_lightning_on(engine_id: &str, request: &VideoRequest) -> bool {
    let is_moe = engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b";
    if !is_moe {
        return false;
    }
    // Absent ⇒ default-on for A14B; only an explicit strict-bool `false` opts out.
    request
        .advanced
        .get("lightning")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true)
}

/// Per-model sampling for the base Wan path (sc-3034 / sc-4997 / sc-10047). On the A14B MoE models
/// (T2V + I2V) the recipe is now conditional on the Lightning toggle ([`wan_lightning_on`]):
/// - toggle **on** (default) → the 4-step Lightning distill preset: forced 4 steps / CFG-off
///   (guide 1.0), unchanged from before.
/// - toggle **off** → the native Wan2.2 A14B multi-step + CFG recipe: honor an explicit user
///   `steps`/`guidanceScale`, else `None` so the engine's own config.json A14B defaults (40 steps,
///   dual CFG) stand exactly.
///
/// The dense TI2V-5B has no distill LoRA yet (sc-4999) and no toggle: honor an explicit user
/// `steps`/`guidanceScale`, else apply the interim default ([`WAN5B_INTERIM_STEPS`], CFG retained).
/// `None` ⇒ the engine config default.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn wan_sampling(engine_id: &str, request: &VideoRequest) -> (Option<u32>, Option<f32>) {
    if engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b" {
        if wan_lightning_on(engine_id, request) {
            // Lightning distill (default): 4 steps / CFG-off. The distill is applied as an
            // adapter (resolve_wan_adapters), so a user `steps`/`guidanceScale` can't break it.
            return (Some(4), Some(1.0));
        }
        // Toggle off: native multi-step CFG. Honor an explicit user override, else `None` so the
        // engine's config.json A14B non-distill defaults (multi-step + CFG on) stand exactly.
        let steps = advanced_opt_u32(request, "steps");
        let guidance = advanced_opt_f32(request, "guidanceScale");
        return (steps, guidance);
    }
    // wan2_2_ti2v_5b (dense): user override wins, else the interim default; CFG left to the
    // engine (guide 5.0) unless the user disables it via `guidanceScale`.
    let steps = advanced_opt_u32(request, "steps").or(Some(WAN5B_INTERIM_STEPS));
    let guidance = advanced_opt_f32(request, "guidanceScale");
    (steps, guidance)
}

/// The lightx2v lightning step-distill recipe (sc-5684 / sc-5700): 8 steps, CFG off, scheduler shift 1.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SCAIL2_LIGHTNING_STEPS: u32 = 8;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SCAIL2_LIGHTNING_GUIDANCE: f32 = 1.0;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SCAIL2_LIGHTNING_SHIFT: f32 = 1.0;

/// SCAIL-2 sampling recipe `(steps, guidance, scheduler_shift)`. When a lightx2v diff-patch
/// "lightning" LoRA is selected (`lightning`), apply the step-distill recipe so the toggle yields the
/// ~10× fewer-DiT-passes speedup: CFG off (guidance 1.0 → the engine short-circuits to a single DiT
/// forward per step) and scheduler shift 1.0 are the lightning invariants (forced), and the step count
/// defaults to 8 but honors an explicit user `advanced.steps` override. Without a lightning LoRA, return
/// all-`None` so the engine's quality defaults (40 steps, guide 5.0, shift 5.0) stand exactly as before
/// — this path is unchanged. The chosen knobs are recorded as `effective*` in [`scail2_raw_settings`]
/// so what actually ran is inspectable on the asset (mirrors [`wan_raw_settings`]).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn scail2_sampling(
    request: &VideoRequest,
    lightning: bool,
) -> (Option<u32>, Option<f32>, Option<f32>) {
    if !lightning {
        return (None, None, None);
    }
    (
        advanced_opt_u32(request, "steps").or(Some(SCAIL2_LIGHTNING_STEPS)),
        Some(SCAIL2_LIGHTNING_GUIDANCE),
        Some(SCAIL2_LIGHTNING_SHIFT),
    )
}

/// `true` if any resolved adapter is a lightx2v diff-patch ("lightning") LoRA — the engine's own
/// detector (a file carrying full-rank `.diff`/`.diff_b` tensors), so the recipe keys off the actual
/// format, not a catalog id or filename. A file that can't be read is treated as non-lightning (the
/// engine surfaces the real load error downstream).
#[cfg(target_os = "macos")]
pub(super) fn scail2_adapters_have_lightning(adapters: &[AdapterSpec]) -> bool {
    adapters
        .iter()
        .any(|a| runtime_macos::providers::scail2::has_diff_patch_keys(&a.path).unwrap_or(false))
}

/// In-place ComfyUI Wan2.2 A14B experts for the sc-10671 base lane (epic 10451 Phase 2c). When set on a
/// [`VideoGenInput`], [`generate_video`] builds the two experts from these files (key remap +
/// scaled-fp8 dequant, `runtime_cuda::providers::wan::load_from_comfyui_experts`) via the uncached bespoke load path
/// instead of the registry snapshot. The UMT5 TE + VAE are read in place too when `te_file` / `vae_file`
/// are set (sc-10909), else they come from `model_dir` (a resident Wan snapshot tier); the tiny
/// tokenizer always comes from `model_dir`. Read in place, never copied.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone)]
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
pub(super) struct ComfyuiWanExperts {
    /// The high-noise expert file (ComfyUI `*_high_noise_*`), read in place → candle `transformer/`.
    high_file: PathBuf,
    /// The low-noise expert file (ComfyUI `*_low_noise_*`), read in place → candle `transformer_2/`.
    low_file: PathBuf,
    /// The UMT5-XXL text encoder (`umt5_xxl_fp8_e4m3fn_scaled`, companion scaled-fp8), read in place
    /// when present (sc-10909); `None` ⇒ the snapshot `text_encoder/`.
    te_file: Option<PathBuf>,
    /// The Wan VAE (`wan_2.1_vae.safetensors`, native WAN-VAE keys), read in place when present
    /// (sc-10909); `None` ⇒ the snapshot `vae/`.
    vae_file: Option<PathBuf>,
    /// I2V (channel-concat) vs T2V — selects the Wan config (`patch_embedding` in-channels differ).
    i2v: bool,
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl ComfyuiWanExperts {
    pub(super) fn new(
        high_file: PathBuf,
        low_file: PathBuf,
        te_file: Option<PathBuf>,
        vae_file: Option<PathBuf>,
        i2v: bool,
    ) -> Self {
        Self {
            high_file,
            low_file,
            te_file,
            vae_file,
            i2v,
        }
    }
}

/// The resolved inputs for one video generation (engine load + request build), shared by
/// Wan (sc-3034) and LTX (sc-3035) — split out so the engine call is unit-testable on real
/// weights without the API/job plumbing. The LTX-only knobs (`video_mode` no_audio,
/// prompt-enhance) default off for Wan; the Wan-only `moe_expert` rides on `adapters`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) struct VideoGenInput {
    pub(super) engine_id: &'static str,
    pub(super) model_dir: PathBuf,
    pub(super) quant: Option<Quant>,
    pub(super) adapters: Vec<AdapterSpec>,
    pub(super) conditioning: Vec<Conditioning>,
    pub(super) prompt: String,
    pub(super) negative_prompt: Option<String>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) frames: u32,
    pub(super) fps: u32,
    pub(super) steps: Option<u32>,
    pub(super) guidance: Option<f32>,
    /// Flow-matching scheduler shift (`req.scheduler_shift`); `None` ⇒ the engine default. Set by the
    /// SCAIL-2 lightning recipe (shift 1.0, sc-5700); the other models leave it at the engine default.
    pub(super) scheduler_shift: Option<f32>,
    /// Per-generation sampler / scheduler (epic 7114 P5, sc-7127). Left `None` by the handlers; the
    /// shared funnel [`generate_video`] reads them from the job's `advanced` block and N3-guards them
    /// against the resolved engine descriptor's advertised surface before they reach `req`. A video
    /// engine that does not advertise the curated sampler/scheduler axis (everything but the Wan
    /// fold-in + the SVD/LTX sampler-only outliers, until candle adoption) leaves these `None`.
    pub(super) sampler: Option<String>,
    pub(super) scheduler: Option<String>,
    pub(super) seed: u64,
    /// Per-request control-clip conditioning scale (Wan-VACE `conditioning_scale`, sc-3441 /
    /// sc-3521); `None` ⇒ the engine default (1.0). Unused by the non-control paths.
    pub(super) control_scale: Option<f32>,
    // LTX-only knobs (sc-3035); left at defaults by Wan + the other models.
    pub(super) video_mode: Option<String>,
    pub(super) enhance_prompt: bool,
    pub(super) use_uncensored_enhancer: bool,
    pub(super) enhance_max_tokens: Option<u32>,
    pub(super) enhance_temperature: Option<f32>,
    // SVD-only micro-conditioning knobs (sc-3523); `None` on the other models.
    pub(super) motion_bucket_id: Option<f32>,
    pub(super) noise_aug_strength: Option<f32>,
    pub(super) decode_chunk_size: Option<u32>,
    // SVD motion-conditioning fps, decoupled from the output `fps` (sc-3764); `None` elsewhere.
    pub(super) conditioning_fps: Option<u32>,
    // SeedVR2 input pre-blur (sc-4816); `None` on the other models.
    pub(super) softness: Option<f32>,
    // LTX-only external Gemma-3 text-encoder snapshot dir (sc-8827): rides `LoadSpec::text_encoder` so
    // the LTX provider locates its Gemma encoder from the spec instead of the process-global
    // `$LTX_GEMMA_DIR` env var (the old `set_var` seam was unsound on the multithreaded runtime,
    // F-025). `None` on every other model (they bundle their TE) and when no override resolves.
    pub(super) text_encoder_dir: Option<PathBuf>,
    /// LTX-only optional **amoral 4-bit Gemma enhancer** snapshot dir (sc-2845 `useUncensoredEnhancer`).
    /// `Some` ⇒ staged in `LoadSpec::components["uncensored_enhancer"]` so the MLX LTX provider loads it
    /// on demand when a request sets the flag (sc-13664 deleted the provider's `$LTX_UNCENSORED_GEMMA_DIR`
    /// / HF-cache scan). `None` on every other model and when the enhancer is off or unprovisioned.
    pub(super) uncensored_enhancer_dir: Option<PathBuf>,
    /// In-place ComfyUI Wan MoE experts (epic 10451 Phase 2c, sc-10671). `Some` ⇒ [`generate_video`]
    /// takes the bespoke uncached load path (`load_from_comfyui_experts`) instead of the registry
    /// snapshot; `None` on every other job.
    pub(super) comfyui: Option<ComfyuiWanExperts>,
    /// MiniMax-H3's tiered DiT directory (epic 17137, sc-19508) — staged in
    /// `LoadSpec::components["transformer"]`. `None` on every other model.
    ///
    /// MiniMax-H3 is the first video family whose tiered weights live in a DIFFERENT repo from its
    /// shared components: the pre-quantized DiT partitions come from `SceneWorks/minimax-h3-mlx`
    /// while the text encoder, tokenizer and both VAEs come from the upstream `MiniMaxAI/MiniMax-H3`
    /// snapshot. `model_dir` (⇒ `spec.weights`) can only name one root, so the other has to ride the
    /// components map — the same mechanism LTX's optional `uncensored_enhancer` uses.
    pub(super) dit_component_dir: Option<PathBuf>,
    /// MiniMax-H3's PER-TIER PACKED text encoder (epic 17137, sc-19120 / sc-19506) — staged in
    /// `LoadSpec::components["text_encoder"]`. `None` on every other model, and `None` for H3 when
    /// the selected tier ships no packed text encoder (the dense bf16 tier, or a q4/q8 install
    /// predating the packed co-requisite).
    ///
    /// NOT the same field as [`Self::text_encoder_dir`], and the distinction is load-bearing rather
    /// than stylistic: that one rides `LoadSpec::text_encoder`, which is what the LTX provider
    /// reads. `mlx-gen-minimax-h3::resolve_text_encoder_dir` reads
    /// `spec.components["text_encoder"]` and falls back to `<weights>/text_encoder` — the UPSTREAM
    /// DENSE bf16 Qwen3-VL-32B — when the key is absent. sc-19120 published q4/q8 packed text
    /// encoders and wired them into the manifest as per-tier co-requisites, but nothing staged them,
    /// so a q4 render loaded a 53 GB dense conditioner and the tier bought nothing on the largest
    /// component in the family. This field is that staging.
    pub(super) text_encoder_component_dir: Option<PathBuf>,
    /// Residency policy for the load (sc-12631). Defaults to [`OffloadPolicy::Resident`] — the historical
    /// video behavior (every component held for the whole run). The candle A14B (two 14B experts swapped
    /// one-resident-at-a-time) and the dense 5B (TE/VAE flushed off-GPU around the denoise, sc-13175) flip
    /// this to [`OffloadPolicy::Sequential`] so the measured `candle.vramGbByTier` peak (the SEQUENTIAL
    /// working set) is the one actually loaded; see `candle_video_offload_policy`. Left `Resident` on the
    /// MLX (macOS) path and the resident-only LTX candle engine. SVD-XT also selects Sequential in
    /// sc-14625 for its conditioner → UNet → VAE lifecycle.
    pub(super) offload_policy: OffloadPolicy,
    /// Per-request memory-rung knobs selected by the video memory gate (sc-18814), or `None` for
    /// the provider's own defaults. Set ONLY by [`generate_video_using`], from
    /// `crate::video_admission::admit_video_generation`; every handler leaves it at the
    /// `Default` so a route the gate makes no decision on is byte-identical to before the gate
    /// existed. The image lane's equivalent is `mlx_fit_gate::evaluate_request`'s
    /// `MlxRequestEvaluation::memory`.
    pub(super) memory: Option<gen_core::GenerationMemory>,
    /// Contract/evidence handshake for provider safety and the request-scoped lifecycle. Kept
    /// separate from `memory`: a Resident selection still carries context while preserving the
    /// provider's request defaults with `memory == None`.
    pub(super) memory_context: Option<gen_core::MemoryRunContext>,
    /// Optional fallible admission consumed only by the serialized generator-cache cold-miss path.
    /// SCAIL Candle and the uncalibrated dual-expert VACE-Fun lane set it; other video families leave
    /// it `None`. It is deliberately outside
    /// `LoadSpec`/the cache key: request-time free VRAM must be re-evaluated for a miss, while an exact
    /// resident key (including precision/adapters/layout) bypasses the cold-load gate.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    pub(super) cold_load_admission: Option<crate::generator_cache::GeneratorColdLoadAdmission>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl Default for VideoGenInput {
    fn default() -> Self {
        Self {
            memory: None,
            memory_context: None,
            engine_id: "",
            model_dir: PathBuf::new(),
            quant: None,
            adapters: Vec::new(),
            conditioning: Vec::new(),
            prompt: String::new(),
            negative_prompt: None,
            width: 0,
            height: 0,
            frames: 0,
            fps: 0,
            steps: None,
            guidance: None,
            scheduler_shift: None,
            sampler: None,
            scheduler: None,
            seed: 0,
            control_scale: None,
            video_mode: None,
            enhance_prompt: false,
            use_uncensored_enhancer: false,
            enhance_max_tokens: None,
            enhance_temperature: None,
            motion_bucket_id: None,
            noise_aug_strength: None,
            decode_chunk_size: None,
            conditioning_fps: None,
            softness: None,
            text_encoder_dir: None,
            uncensored_enhancer_dir: None,
            comfyui: None,
            dit_component_dir: None,
            text_encoder_component_dir: None,
            offload_policy: OffloadPolicy::Resident,
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            cold_load_admission: None,
        }
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn video_load_spec(input: &VideoGenInput) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(input.model_dir.clone()));
    spec.quantize = input.quant;
    spec.precision = Precision::Bf16;
    spec.adapters = input.adapters.clone();
    // LTX's external Gemma-3 text encoder rides the spec (sc-8827); `None` retains the provider's
    // legacy fallback. Video providers otherwise have no image-only control/PiD/identity sources.
    spec.text_encoder = input.text_encoder_dir.clone().map(WeightsSource::Dir);
    // Never downgrade a Sequential policy selected by the candle A14B route.
    spec.offload_policy = input.offload_policy;
    // Named model components (epic 13657). Video providers advertise no `required_components`, so
    // the map is empty by default. Three OPTIONAL components ride it:
    //
    // * LTX-2.3's `uncensored_enhancer` (sc-2845 / sc-13664): when a `useUncensoredEnhancer` job
    //   resolved the amoral 4-bit Gemma snapshot, stage it here so the provider loads it on demand
    //   instead of the deleted `$LTX_UNCENSORED_GEMMA_DIR` / HF-cache scan.
    // * MiniMax-H3's tiered DiT (`"transformer"`, sc-19508): its quantized partitions live in a
    //   different repo from its shared components, and `weights` can only name one root.
    // * MiniMax-H3's per-tier PACKED text encoder (`"text_encoder"`, sc-19120 / sc-19506): the
    //   packed conditioner ships beside the DiT tiers in `SceneWorks/minimax-h3-mlx`, while the
    //   dense bf16 one comes from upstream, so it is the same different-repo problem the DiT
    //   has. Absent ⇒ `mlx-gen-minimax-h3` falls back to `<weights>/text_encoder`, the dense
    //   upstream copy. It does NOT read `LoadSpec::text_encoder` — that field has zero hits in
    //   the whole engine crate, so staging there would resolve nothing and hard-error inside
    //   the engine at the `config.json` probe.
    //
    // All absent ⇒ empty map, the video load path unchanged. They are collected rather than
    // branched so adding a fourth cannot silently drop one.
    spec.components = [
        input
            .uncensored_enhancer_dir
            .clone()
            .map(|dir| ("uncensored_enhancer".to_owned(), WeightsSource::Dir(dir))),
        input
            .dit_component_dir
            .clone()
            .map(|dir| ("transformer".to_owned(), WeightsSource::Dir(dir))),
        input
            .text_encoder_component_dir
            .clone()
            .map(|dir| ("text_encoder".to_owned(), WeightsSource::Dir(dir))),
    ]
    .into_iter()
    .flatten()
    .collect::<BTreeMap<_, _>>();
    spec
}

/// Provider-only request modifiers share the overlay evidence axis with adapters and enhancers.
/// A catalog `text_to_video` request can still select a different provider workload (LTX's
/// `no_audio` is the live case), so the resolved provider mode must not borrow the ordinary
/// no-overlay curve until a receipt names this exact carrier.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn video_admission_overlay(input: &VideoGenInput) -> Option<String> {
    let mut overlays = Vec::new();
    if !input.adapters.is_empty() {
        overlays.push(format!("adapters:{}", input.adapters.len()));
    }
    if input.use_uncensored_enhancer || input.uncensored_enhancer_dir.is_some() {
        overlays.push("enhancer:uncensored".to_owned());
    } else if input.enhance_prompt {
        overlays.push("enhancer:standard".to_owned());
    }
    if let Some(video_mode) = input.video_mode.as_deref() {
        overlays.push(format!("provider_video_mode:{video_mode}"));
    }
    if matches!(input.engine_id, "ltx_2_3" | "ltx_2_3_distilled") {
        let mut references = input.conditioning.iter().filter_map(|conditioning| {
            let Conditioning::Reference { image, strength } = conditioning else {
                return None;
            };
            Some((image, strength.unwrap_or(1.0)))
        });
        if let (Some((image, strength)), None) = (references.next(), references.next()) {
            overlays.push(format!(
                "reference:image:{}x{}:strength:{:08x}",
                image.width,
                image.height,
                strength.to_bits()
            ));
        }
        let keyframes = input
            .conditioning
            .iter()
            .filter_map(|conditioning| {
                let Conditioning::Keyframe {
                    image,
                    frame_idx,
                    strength,
                } = conditioning
                else {
                    return None;
                };
                Some((image, *frame_idx, *strength))
            })
            .collect::<Vec<_>>();
        if let [(first, 0, first_strength), (last, -1, last_strength)] = keyframes.as_slice() {
            overlays.push(format!(
                "keyframe:first:image:{}x{}:frame:0:strength:{:08x}",
                first.width,
                first.height,
                first_strength.to_bits()
            ));
            overlays.push(format!(
                "keyframe:last:image:{}x{}:frame:-1:strength:{:08x}",
                last.width,
                last.height,
                last_strength.to_bits()
            ));
        }
        let clips = input
            .conditioning
            .iter()
            .filter_map(|conditioning| {
                let Conditioning::VideoClip {
                    frames,
                    frame_idx,
                    strength,
                } = conditioning
                else {
                    return None;
                };
                Some((frames, *frame_idx, *strength))
            })
            .collect::<Vec<_>>();
        if let [(frames, 0, strength)] = clips.as_slice() {
            if let Some(image) = frames.first() {
                if frames
                    .iter()
                    .all(|frame| frame.width == image.width && frame.height == image.height)
                {
                    overlays.push(format!(
                        "clip:append:frames:{}:image:{}x{}:frame:0:strength:{:08x}",
                        frames.len(),
                        image.width,
                        image.height,
                        strength.to_bits()
                    ));
                }
            }
        }
    }
    (!overlays.is_empty()).then(|| overlays.join("+"))
}

/// Whether the resolved provider input is inside the promoted SC-18810 calibration surface.
/// This check runs before the live-budget probe and before contract selection, so unsupported
/// I2V/keyframe/clip, overlay, enhancer, no-audio, and out-of-envelope FPS requests keep the
/// historical direct-generate path instead of reaching provider safety with invented coverage.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
/// Apply the admission result at the single loaded-video handoff. Keeping the provider knobs and
/// lifecycle context in one operation makes it impossible for the generation request to carry an
/// optimized rung while silently bypassing its safety/begin/configure/finish contract.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn apply_video_admission_outcome(
    input: &mut VideoGenInput,
    outcome: crate::video_admission::VideoAdmissionOutcome,
) -> WorkerResult<()> {
    if let Some(refusal) = outcome.refusal {
        return Err(WorkerError::InvalidPayload(refusal));
    }
    input.memory = outcome.memory;
    input.memory_context = outcome.context;
    Ok(())
}

/// Run one generation to a [`DecodedVideo`] (RGB8 frames + fps + optional audio) against an already
/// loaded video generator, streaming denoise progress via `on_progress` and honoring `cancel`.
/// The engine fills the audio track (LTX) or leaves it `None` (Wan).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn run_loaded_video_generation(
    generator: &dyn Generator,
    input: VideoGenInput,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<DecodedVideo> {
    let memory_context = input.memory_context;
    let mut req = GenerationRequest {
        prompt: input.prompt,
        negative_prompt: input.negative_prompt,
        width: input.width,
        height: input.height,
        frames: Some(input.frames),
        fps: Some(input.fps),
        steps: input.steps,
        guidance: input.guidance,
        scheduler_shift: input.scheduler_shift,
        // Per-generation sampler / scheduler (sc-7127), already N3-guarded against the engine's
        // advertised surface in `generate_video`, so an unsupported name was dropped to `None` (the
        // engine default) before reaching here — `validate_request` only ever sees an advertised name.
        sampler: input.sampler,
        scheduler: input.scheduler,
        seed: Some(input.seed),
        conditioning: input.conditioning,
        control_scale: input.control_scale,
        video_mode: input.video_mode,
        enhance_prompt: input.enhance_prompt,
        use_uncensored_enhancer: input.use_uncensored_enhancer,
        enhance_max_tokens: input.enhance_max_tokens,
        enhance_temperature: input.enhance_temperature,
        motion_bucket_id: input.motion_bucket_id,
        noise_aug_strength: input.noise_aug_strength,
        decode_chunk_size: input.decode_chunk_size,
        conditioning_fps: input.conditioning_fps,
        softness: input.softness,
        // The video memory gate's selection (sc-18814). `None` — every route the gate does not
        // decide, plus a selected resident rung — leaves the provider's own defaults in place.
        memory: input.memory,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = crate::memory_strategy::generate_with_scope(
        generator,
        &mut req,
        memory_context.as_ref(),
        on_progress,
    )
    .map_err(|error| crate::classify_engine_error("video generation failed", error))?;
    match output {
        GenerationOutput::Video { frames, fps, audio } => Ok(DecodedVideo {
            frames: frames
                .into_iter()
                .map(|image| RgbFrame {
                    width: image.width,
                    height: image.height,
                    pixels: image.pixels,
                })
                .collect(),
            fps,
            audio: audio.map(|track| AudioTrack {
                samples: track.samples,
                sample_rate: track.sample_rate,
                channels: track.channels,
            }),
            adapter_apply_reports: generator.adapter_apply_reports(),
        }),
        GenerationOutput::Images(_) => Err(WorkerError::Engine(
            "video model returned images, expected video frames".to_owned(),
        )),
        // `GenerationOutput::Audio` arrived with the candle-audio lane (sc-12834); no video engine
        // produces it, so it is as much an engine contract violation here as `Images`.
        GenerationOutput::Audio(_) => Err(WorkerError::Engine(
            "video model returned audio, expected video frames".to_owned(),
        )),
    }
}

#[cfg(all(target_os = "macos", test))]
fn load_video_generation_for_tests(input: &VideoGenInput) -> WorkerResult<Box<dyn Generator>> {
    let spec = video_load_spec(input);
    crate::inference_runtime::load(input.engine_id, &spec)
        .map_err(|error| crate::classify_engine_error("video load failed", error))
}

#[cfg(all(target_os = "macos", test))]
pub(super) fn run_video_generation(
    input: VideoGenInput,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<DecodedVideo> {
    let generator = load_video_generation_for_tests(&input)?;
    run_loaded_video_generation(generator.as_ref(), input, cancel, on_progress)
}

/// Forward-progress watchdog: if the engine emits no progress event (no denoise `Step`, no
/// `Decoding`) for this long — covering both the silent cold model-load phase and the gap
/// between steps — the generation is treated as wedged and the job is failed with a clear
/// error instead of heartbeating indefinitely. Tuned well above any legitimate single load or
/// step on the current video models; override via `SCENEWORKS_VIDEO_STALL_SECS` for an
/// unusually large/slow model or disk.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const VIDEO_STALL_TIMEOUT: Duration = Duration::from_secs(600);

/// Grace period granted after a stall is detected and engine cancellation is requested, before
/// the still-running blocking task is abandoned. A cooperative engine bails between steps well
/// within this window (the manual-cancel path proves it honors the flag); the abandon escape
/// only matters for a hard Metal wedge that never re-checks cancel, and keeps the watchdog from
/// itself re-hanging on the join.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const VIDEO_STALL_GRACE: Duration = Duration::from_secs(60);

/// The effective forward-progress stall timeout: `SCENEWORKS_VIDEO_STALL_SECS` (a positive
/// integer number of seconds) when set, else [`VIDEO_STALL_TIMEOUT`].
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn video_stall_timeout() -> Duration {
    parse_stall_timeout(std::env::var("SCENEWORKS_VIDEO_STALL_SECS").ok())
}

/// Parse the `SCENEWORKS_VIDEO_STALL_SECS` override (a positive integer number of seconds),
/// falling back to [`VIDEO_STALL_TIMEOUT`] when unset, blank, non-numeric, or zero.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn parse_stall_timeout(raw: Option<String>) -> Duration {
    raw.and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map(Duration::from_secs)
        .unwrap_or(VIDEO_STALL_TIMEOUT)
}

/// First-detection handling for the in-loop video cancel poller (sc-5516): trip the engine
/// `CancelFlag` and post a NON-terminal "Cancelling…" update (indeterminate progress bar —
/// `running` + fraction 0.0 renders the "Working" animation, not a backward jump). The terminal
/// `Canceled` is posted only after the blocking generation actually stops (see `generate_video`),
/// so the worker row — and therefore the next queued job — is not freed until the GPU is genuinely
/// idle, and the UI honestly shows "Cancelling…" until completion. Best-effort: a failed status
/// update here is non-fatal because the post-run terminal write is what ultimately frees the
/// worker. Mirrors the image path's `begin_image_cancel` (sc-5515).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn begin_video_cancel(
    api: &ApiClient,
    job_id: &str,
    cancel: &CancelFlag,
    backend: &str,
) {
    cancel.cancel();
    let _ = update_job(
        api,
        job_id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.0,
            "Cancelling — finishing the current step…",
            None,
            backend,
        ),
    )
    .await;
}

/// The `(samplers, schedulers)` a video engine advertises (epic 7114), read from its registered
/// gen-core descriptor by engine id — the same `Capabilities` surface `validate_request` enforces, so
/// the N3 guard in [`generate_video`] mirrors the image lane's `model.descriptor.capabilities` read.
/// Empty (so every name N3-falls back to the engine default) when the id isn't registered on this
/// backend — e.g. a candle video engine before it adopts the unified framework.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn video_engine_sampling_surface(engine_id: &str) -> (Vec<&'static str>, Vec<&'static str>) {
    crate::inference_runtime::generators()
        .map(|reg| (reg.descriptor)())
        .find(|descriptor| descriptor.id == engine_id)
        .map(|descriptor| {
            (
                descriptor.capabilities.samplers,
                descriptor.capabilities.schedulers,
            )
        })
        .unwrap_or_default()
}

/// The effective video settings captured from the resolved [`VideoGenInput`] just
/// before it moves into the blocking generation task (epic 10402, sc-10418). Sourced
/// from the single resolved funnel so it reflects exactly what reached the engine —
/// the tier-resolved quant, the N3-guarded sampler/scheduler, and the recipe-resolved
/// steps/guidance — rather than re-deriving them from the sparse `advanced` payload
/// (the video engines' quant/guidance rules are engine-specific and would drift from
/// what actually ran). Captured like `log_engine_id`, before the move at the spawn.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) struct VideoSettingsSnapshot {
    pub(super) quant: Option<Quant>,
    pub(super) sampler: Option<String>,
    pub(super) scheduler: Option<String>,
    pub(super) scheduler_shift: Option<f32>,
    pub(super) guidance: Option<f32>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) seed: u64,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl VideoSettingsSnapshot {
    fn from_input(input: &VideoGenInput) -> Self {
        Self {
            quant: input.quant,
            sampler: input.sampler.clone(),
            scheduler: input.scheduler.clone(),
            scheduler_shift: input.scheduler_shift,
            guidance: input.guidance,
            width: input.width,
            height: input.height,
            seed: input.seed,
        }
    }
}

/// Normalized quant label + bit-width for a resolved video [`Quant`] (epic 10402,
/// sc-10418): `Q8` → ("q8", 8), `Q4` → ("q4", 4), `Nvfp4` → ("nvfp4", None), `None`
/// (dense/bf16) → ("bf16", None). Mirrors the image lane's `effective_quant_label` mapping
/// so the Stats charts group video and image runs on the same tier labels.
///
/// **MATCH THE VARIANT — never derive a tier label from [`Quant::bits`] (sc-11042, epic 11037 SC#5).**
/// This function used to `format!("q{bits}")` from `q.bits()`, which was correct only while `Quant` was
/// `{Q4, Q8}`. `Quant::Nvfp4::bits()` returns **4** (its E2M1 elements are 4-bit), so the bits-derived
/// form stamped an NVFP4 video render as `"q4"` + bits 4 in Stats telemetry — falsely reporting one
/// creative choice as another, exactly the tier aliasing this epic forbids. **The compiler could not
/// catch it**: reading `.bits()` raises no E0004 when a variant is added, so it compiled silently on the
/// `backend-candle` lane. The explicit arms below make any future `Quant` variant a hard compile error
/// here instead of a silent mislabel — do not collapse them back into a catch-all or a `bits()` map.
///
/// NVFP4 reports **no** bit count: it is ~4.5 EFFECTIVE bits/weight (E2M1 elements + FP8-E4M3 block
/// scales + an FP32 per-tensor scale), so `Some(4)` would re-introduce the same `q4` aliasing in the
/// `quant_bits` column that the label fix removes. `None` is the honest "no integer width applies" —
/// the same signal the dense/bf16 arm uses, and the same reason `flux2_comfyui_raw_settings` writes
/// `mlxQuantize: null` for this tier.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn video_quant_label(quant: Option<Quant>) -> (Option<String>, Option<u32>) {
    match quant {
        Some(Quant::Q4) => (Some("q4".to_owned()), Some(4)),
        Some(Quant::Q8) => (Some("q8".to_owned()), Some(8)),
        Some(Quant::Nvfp4) => (Some("nvfp4".to_owned()), None),
        None => (Some("bf16".to_owned()), None),
    }
}

/// Fold the effective video settings + model + observed step count into the
/// phase-timing metrics block for a finished video job (epic 10402, sc-10418). A
/// video job produces one output (sc-10426). Sampler/scheduler fall back to
/// "default" (engine-native) so the comparison charts always have a non-blank group,
/// mirroring the image lane. Guidance / scheduler-shift stay `None` when the engine's
/// own config default was used (not overridden) — an honest "not captured" rather
/// than a fabricated value the worker can't know without loading the engine config.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn build_video_metrics(
    mut metrics: GenerationMetrics,
    settings: &VideoSettingsSnapshot,
    model: Option<String>,
    effective_steps: Option<u32>,
) -> GenerationMetrics {
    let (quant_label, quant_bits) = video_quant_label(settings.quant);
    metrics.model = model;
    metrics.quant_label = quant_label;
    metrics.quant_bits = quant_bits;
    metrics.sampler = Some(
        settings
            .sampler
            .clone()
            .unwrap_or_else(|| "default".to_owned()),
    );
    metrics.scheduler = Some(
        settings
            .scheduler
            .clone()
            .unwrap_or_else(|| "default".to_owned()),
    );
    metrics.scheduler_shift = settings
        .scheduler_shift
        .and_then(|shift| serde_json::Number::from_f64(shift as f64));
    metrics.steps = effective_steps;
    metrics.image_count = Some(1); // one video output per job (sc-10426)
    metrics.guidance_scale = settings
        .guidance
        .and_then(|scale| serde_json::Number::from_f64(scale as f64));
    metrics.guidance_method = Some("cfg".to_owned());
    metrics.width = Some(settings.width);
    metrics.height = Some(settings.height);
    metrics.seed = Some(settings.seed as i64);
    metrics
}

/// Drive a `run_video_generation` on a blocking thread, forwarding its streamed denoise
/// progress to the async worker (Generating stage ~0.25..0.58) + polling cancel ~every 2s.
/// The shared blocking + mpsc + cancel plumbing for Wan and LTX. A forward-progress watchdog
/// ([`video_stall_timeout`]) fails a wedged job loudly rather than letting it look alive forever.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn generate_video(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    backend: &str,
    advanced: &JsonObject,
    input: VideoGenInput,
) -> WorkerResult<DecodedVideo> {
    generate_video_using(
        api,
        settings,
        job,
        backend,
        advanced,
        input,
        crate::inference_runtime::load,
    )
    .await
}

/// [`generate_video`] with the engine loader supplied by the caller (sc-12318).
///
/// The `_using` half of the same pair [`crate::generator_cache::with_cached_generator`] already splits
/// one level down, and it exists for the same reason: with the loader threaded in, a test can drive an
/// async per-family arm (`generate_mochi`, `generate_candle_video`) against a stub `Generator` and
/// assert on the [`VideoGenInput`] that actually reached the engine. Without it, every decision an arm
/// makes inline — the frame lattice, the Mochi fit gate — is reachable only as the free function it
/// delegates to, never as the call itself.
///
/// SCOPE: the injected loader covers the registry **cached** path only. The in-place ComfyUI Wan MoE
/// branch builds its generator from per-file expert weights through `with_uncached_generator`, which has
/// no `(engine_id, spec)` key to load from, so it ignores `load_generator` and stays uncovered here.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn generate_video_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    backend: &str,
    advanced: &JsonObject,
    mut input: VideoGenInput,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<DecodedVideo> {
    // Per-generation sampler / scheduler axis for video (epic 7114 P5, sc-7127). The handlers leave
    // `input.sampler`/`scheduler` `None`; read them from the caller's already-parsed `advanced` block
    // here — the single funnel every Wan / LTX / SVD path passes through — and N3-guard each against
    // the resolved engine descriptor's advertised surface. A name the engine does not advertise (every
    // video engine but the Wan fold-in + the SVD/LTX sampler-only outliers, until candle adoption) is
    // dropped to the engine default + a `sampling_knob_unsupported` event, never a hard-fail. Taking
    // `advanced` by reference avoids re-parsing the whole payload into a throwaway VideoRequest per
    // generation (F-118).
    {
        let (raw_sampler, raw_scheduler, raw_shift) =
            crate::image_jobs::read_advanced_sampling_knobs(advanced);
        let (samplers, schedulers) = video_engine_sampling_surface(input.engine_id);
        input.sampler = crate::image_jobs::normalize_sampling_knob(
            raw_sampler,
            &samplers,
            "sampler",
            input.engine_id,
            &job.id,
            backend,
        );
        input.scheduler = crate::image_jobs::normalize_sampling_knob(
            raw_scheduler,
            &schedulers,
            "scheduler",
            input.engine_id,
            &job.id,
            backend,
        );
        // Schedule shift: only when the handler hasn't already forced it (the SCAIL-2 lightning recipe
        // sets shift 1.0), so the user knob can't clobber a model's required recipe. Parity with the
        // image lane's `advanced.schedulerShift` / `timestepShift` read.
        if input.scheduler_shift.is_none() {
            input.scheduler_shift = raw_shift;
        }
    }
    // The video memory gate (sc-18814, epic 18803). Resolved here — the ONE funnel every video
    // family on both lanes passes through — so no per-family edit is needed and neither lane can
    // silently miss it. The budget probe is async on the candle lane, so it happens before the
    // blocking task is spawned; the selection itself runs inside `run`, where the loaded
    // generator (and therefore the provider's memory contract) is in scope. That is the same
    // position the image lane calls `mlx_fit_gate::evaluate_request` from: after the load, before
    // `generate`.
    // The catalog model id, read straight off the payload rather than by re-parsing the whole
    // request into a throwaway `VideoRequest` (F-118, the same reason `advanced` arrives by
    // reference). Read through the SAME function `VideoRequest::from_payload` resolves `model`
    // with, so the two cannot diverge: a bare `.unwrap_or("ltx_2_3")` kept a present-but-empty
    // `model` as `""` while the parse resolved it to `ltx_2_3`, and the two ids grade different
    // families through `video_admission_surface`.
    let admission_model_id = sceneworks_core::video_request::payload_model_id(&job.payload);
    // The fitted-curve identity uses the catalog family, not the provider descriptor's internal
    // family. Resolve it exactly like the asset path's `resolve_family`, without re-parsing the
    // entire payload into another `VideoRequest` merely for this admission-only key.
    let admission_manifest_entry = job
        .payload
        .get("modelManifestEntry")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let admission_model_family =
        super::resolve_catalog_video_family(&admission_model_id, &admission_manifest_entry);
    // Bind fitted curves to the same request mode `VideoRequest::from_payload` resolves. The
    // promoted LTX curve is T2V; every other mode falls back until its own curve exists.
    let admission_mode = sceneworks_core::video_request::payload_video_mode(&job.payload);
    // Curve currency is the provider's live packaged compile closure, independent of whether an
    // exact per-cell calibration binding exists in the manifest (sc-19020). An undeclared provider
    // keeps the established sentinel and therefore cannot match a closure-bound fitted curve.
    let admission_closure_digest = sceneworks_core::memory_calibration::packaged_closure_digest(
        crate::video_admission::LANE.as_key(),
        input.engine_id,
    )
    .unwrap_or_else(|| crate::mlx_fit_gate::UNCALIBRATED_CLOSURE.to_owned());

    let cancel = CancelFlag::new();
    let stall_timeout = video_stall_timeout();
    let log_engine_id = input.engine_id;
    // Snapshot the effective settings before `input` moves into the blocking task
    // (sc-10418), so the completion-time metrics POST reports exactly what reached
    // the engine (resolved quant / sampler / scheduler / guidance / dims / seed).
    let video_settings = VideoSettingsSnapshot::from_input(&input);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Progress>(64);
    let blocking = {
        let cancel = cancel.clone();
        let spec = video_load_spec(&input);
        let engine_id = input.engine_id;
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        let cold_load_admission = input.cold_load_admission.take();
        // sc-10671: an in-place ComfyUI Wan MoE takes the bespoke **uncached** load path
        // (`load_from_comfyui_experts` — two experts read in place + remapped + dequant'd), which frees
        // any resident cached generator first; every other job takes the registry cached path. On the
        // non-candle/macOS build `comfyui` is always `None`, so only the cached path is compiled.
        let comfyui_load = input.comfyui.as_ref().map(|e| {
            (
                e.high_file.clone(),
                e.low_file.clone(),
                e.te_file.clone(),
                e.vae_file.clone(),
                input.model_dir.clone(),
                e.i2v,
            )
        });
        // The admission tier/headroom read the model DIRECTORY (`spec_component_bytes` sums the
        // snapshot's safetensors and asks the registry for a footprint), so they are filesystem
        // work and must not run on a reactor thread. The clone is cheap (`LoadSpec` is `Clone`;
        // `spec` itself moves into the loader) and lets both derivations happen inside `run`, which
        // executes on the generator cache thread — the same position the image lane derives its
        // own from, inside the blocking closure.
        let admission_spec = spec.clone();
        let admission_geometry = (
            input.width,
            input.height,
            input.frames,
            input.decode_chunk_size,
        );
        tokio::spawn(async move {
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            let cold_load_cancel = cancel.clone();
            let run = move |generator: &dyn Generator,
                            cache_state: gen_core::MemoryCacheState,
                            loaded_policy: crate::generator_cache::ExecutionPolicy,
                            warm_policy: crate::execution_planner::WarmPolicyProposal,
                            _external_committed_bytes: u64,
                            provider_resident_bytes: u64| {
                // The video lane has no request-scoped memory block for a policy switch to act
                // through; admission below keeps using the LOADED policy. Decline truthfully.
                warm_policy.decline(
                    crate::execution_planner::ServedAsIsReason::RouteHasNoRequestScopedMemory,
                );
                let load_policy = loaded_policy.offload_policy;
                let mut input = input;
                let admission_tier =
                    crate::mlx_fit_gate::resolved_video_numeric_tier(engine_id, &admission_spec)?;
                let spec_headroom_bytes =
                    crate::mlx_fit_gate::spec_headroom_bytes(engine_id, &admission_spec);
                let reference_count = u32::try_from(
                    input
                        .conditioning
                        .iter()
                        .filter(|conditioning| {
                            matches!(
                                conditioning,
                                gen_core::Conditioning::Reference { .. }
                                    | gen_core::Conditioning::Keyframe { .. }
                            )
                        })
                        .count(),
                )
                .unwrap_or(u32::MAX);
                // This is the provider-facing carrier, not a synonym for the user-visible mode:
                // Wan turns clip extension/bridging into pinned keyframes, while I2V reaches the
                // reference-image encoder. A future measured row must name that real residency
                // surface before request-scoped selection can use it.
                let admission_reference_shape = if reference_count == 0 {
                    "none"
                } else {
                    match admission_mode.as_str() {
                        "image_to_video" => "image",
                        "first_last_frame" => "keyframe",
                        "extend_clip" => "none",
                        _ => "other",
                    }
                };
                let admission_overlay = video_admission_overlay(&input);
                let mut admission_inputs = crate::video_admission::VideoAdmissionInputs {
                    model_id: &admission_model_id,
                    model_family: &admission_model_family,
                    route: engine_id,
                    mode: &admission_mode,
                    reference_count,
                    reference_shape: admission_reference_shape,
                    overlay: admission_overlay.as_deref(),
                    lane: crate::video_admission::LANE,
                    tier: admission_tier,
                    width: admission_geometry.0,
                    height: admission_geometry.1,
                    frames: admission_geometry.2,
                    decode_chunk_size: admission_geometry.3,
                    fps: input.fps,
                    runtime: None,
                    headroom_bytes: spec_headroom_bytes,
                    expected_closure_digest: &admission_closure_digest,
                };
                // Evidence is the preflight: an unsupported request stays direct generation without
                // attempting a platform memory probe that could fail independently of admission.
                let admission_runtime =
                    if crate::video_admission::packaged_video_evidence_covers_request(
                        generator,
                        &admission_inputs,
                    ) {
                        crate::video_admission::live_video_runtime_state(
                            engine_id,
                            cache_state,
                            load_policy,
                            provider_resident_bytes,
                        )?
                    } else {
                        None
                    };
                let admission_headroom_bytes = match admission_runtime {
                    Some(runtime) => spec_headroom_bytes
                        .checked_sub(runtime.budget.reserved_headroom_bytes)
                        .ok_or_else(|| {
                            WorkerError::InvalidPayload(format!(
                                "{engine_id} live memory reserve {} exceeds fallback headroom {}; \
                                 refusing an inconsistent video budget",
                                runtime.budget.reserved_headroom_bytes, spec_headroom_bytes,
                            ))
                        })?,
                    // Unsupported surfaces and lanes without a canonical post-load snapshot fail
                    // open before selection, so this value is observationally inert there.
                    None => spec_headroom_bytes,
                };
                admission_inputs.runtime = admission_runtime;
                admission_inputs.headroom_bytes = admission_headroom_bytes;
                let outcome =
                    crate::video_admission::admit_video_generation(generator, admission_inputs);
                apply_video_admission_outcome(&mut input, outcome)?;
                let mut on_progress = |progress: Progress| {
                    // A closed channel means the consumer loop returned early (POST failure /
                    // 409); trip the engine flag so the denoise bails instead of running unheard
                    // (sc-8804, F-003 — the swallowed-closed-channel leak).
                    if tx.blocking_send(progress).is_err() {
                        cancel.cancel();
                    }
                };
                run_loaded_video_generation(generator, input, &cancel, &mut on_progress)
            };
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            let result = match comfyui_load {
                Some((high, low, te, vae, snapshot, i2v)) => {
                    debug_assert!(cold_load_admission.is_none());
                    crate::generator_cache::with_uncached_generator(
                        move || {
                            runtime_cuda::providers::wan::wan14b::load_from_comfyui_experts_with_offload(
                                high,
                                low,
                                te,
                                vae,
                                snapshot,
                                i2v,
                                OffloadPolicy::Sequential,
                            )
                            .map_err(|error| {
                                crate::classify_engine_error("video load failed", error)
                            })
                        },
                        // The uncached in-place route loads eagerly under the hardcoded policy and
                        // has no request-scoped planner seam, so the loaded policy is synthesized
                        // here and the proposal is inert (the run closure declines it regardless).
                        move |generator, cache_state, load_policy, external, provider| {
                            run(
                                generator,
                                cache_state,
                                crate::generator_cache::ExecutionPolicy {
                                    offload_policy: load_policy,
                                    load_shape: gen_core::LoadShape::EagerMaterialization,
                                    load_shape_declaration_result:
                                        gen_core::LoadShapeDeclarationResult::NotEvaluated,
                                },
                                crate::execution_planner::WarmPolicyProposal::inert(engine_id),
                                external,
                                provider,
                            )
                        },
                    )
                    .await
                }
                None => match cold_load_admission {
                    Some(admission) => {
                        crate::generator_cache::with_cached_generator_for_request_using_cold_admission(
                            engine_id,
                            spec,
                            "video load failed",
                            cold_load_cancel,
                            admission,
                            load_generator,
                            run,
                        )
                        .await
                    }
                    None => {
                        crate::generator_cache::with_cached_generator_for_request_using(
                            engine_id,
                            spec,
                            "video load failed",
                            load_generator,
                            run,
                        )
                        .await
                    }
                },
            };
            #[cfg(not(all(not(target_os = "macos"), feature = "backend-candle")))]
            let result = {
                let _ = comfyui_load;
                crate::generator_cache::with_cached_generator_for_request_using(
                    engine_id,
                    spec,
                    "video load failed",
                    load_generator,
                    run,
                )
                .await
            };
            result
        })
    };

    // Bind the blocking generation task to its cancel flag (sc-8804, F-003): every `update_job`/
    // `heartbeat` `?` in the loop below returns early on a transient POST failure or a 409
    // (stale-sweep reclaim); on that early return this guard trips the engine `CancelFlag` and
    // aborts the still-running denoise instead of leaving it burning GPU memory alongside the next
    // claimed job. The stall/abandon watchdog and final join reach through `guard.handle_mut()` /
    // `guard.into_handle()`. `cancel` is kept alongside (it's `Clone`) for the in-loop pollers.
    let mut guard = CancelJoinGuard::new(cancel.clone(), blocking);
    let mut canceled = false;
    // Set when the watchdog (not the user) tripped, so the job is failed with a stall error
    // rather than reported as a clean user cancellation.
    let mut stalled = false;
    // Once a stall is detected we request engine cancel and wait at most `VIDEO_STALL_GRACE`
    // for the blocking task to unwind; past this deadline we abandon it (a hard Metal wedge)
    // so the watchdog never re-hangs on the join.
    let mut abandon_deadline: Option<Instant> = None;
    let mut abandoned = false;
    let mut last_cancel = Instant::now();
    // Time of the most recent progress event; the forward-progress watchdog fails the job if
    // this goes stale for `stall_timeout` (covers both the silent load phase and step-to-step).
    let mut last_progress = Instant::now();
    // Interval arm so the cold model-load phase (crate::inference_runtime::load emits no progress)
    // still heartbeats and polls cancel, instead of looking dead to the API's
    // staleness check until the first denoise step (sc-4276 / F-MLXW-12; mirrors
    // the caption-job select!-with-interval).
    let mut interval = tokio::time::interval(crate::progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Per-phase wall-clock (epic 10402, sc-10405): load = start→first Step,
    // sample = Step→Decoding, decode = Decoding→engine-return (video emits no
    // per-frame decode-done event). Posted best-effort at clean completion.
    let mut phase_timer = crate::job_metrics::PhaseTimer::new(Instant::now());
    // Effective denoise step count from the Step event (sc-10406).
    let mut video_effective_steps: Option<u32> = None;
    // Run the progress loop capturing its Result so any `?`-error path performs the explicit awaited
    // bounded-join teardown BEFORE returning, instead of drop-and-run (sc-8804, F-003). The stall/
    // abandon watchdog inside the loop still handles the hard-wedge case via `abandoned`.
    let loop_result: WorkerResult<()> = async {
        loop {
            tokio::select! {
                maybe_progress = rx.recv() => {
                    let Some(progress) = maybe_progress else {
                        break;
                    };
                    last_progress = Instant::now(); // forward progress — reset the stall watchdog.
                    if canceled {
                        continue; // drain so the blocking sender never blocks.
                    }
                    // sc-9618: a process shutdown is a cancel checkpoint too — short-circuit the API
                    // poll so a quit stops the gen at this frame step, matching a user cancel.
                    if shutdown_requested() {
                        begin_video_cancel(api, &job.id, &cancel, backend).await;
                        canceled = true;
                        continue;
                    }
                    if last_cancel.elapsed() >= Duration::from_secs(2) {
                        last_cancel = Instant::now();
                        if cancel_requested_peek(api, &job.id).await {
                            begin_video_cancel(api, &job.id, &cancel, backend).await;
                            canceled = true;
                            continue;
                        }
                        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                    }
                    // Phase-boundary capture (sc-10405), borrowing so `progress` is
                    // still owned by the fraction/message match below.
                    match &progress {
                        Progress::Step { total, .. } => {
                            phase_timer.mark_sample_step(Instant::now());
                            if video_effective_steps.is_none() {
                                video_effective_steps = Some(*total);
                            }
                        }
                        Progress::Decoding => phase_timer.mark_decoding(Instant::now()),
                        Progress::Loading(_) => {}
                    }
                    let (status, stage, fraction, message) = match progress {
                        Progress::Step { current, total } => (
                            JobStatus::Running,
                            ProgressStage::Generating,
                            0.25 + 0.30 * (current as f64 / total.max(1) as f64),
                            format!("Generating frames — step {current}/{total}."),
                        ),
                        Progress::Decoding => (
                            JobStatus::Running,
                            ProgressStage::Generating,
                            0.58,
                            "Decoding frames.".to_owned(),
                        ),
                        Progress::Loading(phase) => (
                            JobStatus::LoadingModel,
                            ProgressStage::LoadingModel,
                            0.24,
                            match phase {
                                LoadPhase::TextEncoder => "Loading text encoder.",
                                LoadPhase::Renderer => "Loading render components.",
                            }
                            .to_owned(),
                        ),
                    };
                    update_job(
                        api,
                        &job.id,
                        video_progress(
                            status,
                            stage,
                            fraction,
                            &message,
                            None,
                            backend,
                        ),
                    )
                    .await?;
                }
                _ = interval.tick() => {
                    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                    // sc-9618: honor a process shutdown on every tick (local flag read, unthrottled).
                    if !canceled && (shutdown_requested()
                        || (last_cancel.elapsed() >= Duration::from_secs(2) && {
                            last_cancel = Instant::now();
                            cancel_requested_peek(api, &job.id).await
                        }))
                    {
                        begin_video_cancel(api, &job.id, &cancel, backend).await;
                        canceled = true;
                    }
                    // Forward-progress watchdog: a wedged engine keeps this async loop heartbeating
                    // (the block runs on a separate thread), so the API sees a healthy job forever.
                    // If no progress has arrived for `stall_timeout`, request engine cancel and start
                    // the abandon countdown.
                    if !canceled && last_progress.elapsed() >= stall_timeout {
                        tracing::warn!(
                            event = "rust_worker_video_stalled",
                            jobId = %job.id,
                            engine = %log_engine_id,
                            stallSeconds = stall_timeout.as_secs(),
                            "no progress within the stall window — requesting engine cancel"
                        );
                        cancel.cancel();
                        canceled = true;
                        stalled = true;
                        abandon_deadline = Some(Instant::now() + VIDEO_STALL_GRACE);
                    }
                    if let Some(deadline) = abandon_deadline {
                        if Instant::now() >= deadline {
                            abandoned = true;
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }
    .await;
    if let Err(error) = loop_result {
        guard.cancel_and_join().await;
        return Err(error);
    }

    if abandoned {
        // The engine never honored the cancel flag within the grace window (a hard Metal wedge).
        // Detach the still-running blocking task instead of awaiting it — awaiting would re-hang
        // the very failure path this watchdog exists to break. The thread (and the GPU it holds)
        // leaks until the worker is restarted by the supervisor.
        tracing::error!(
            event = "rust_worker_video_abandoned",
            jobId = %job.id,
            engine = %log_engine_id,
            graceSeconds = VIDEO_STALL_GRACE.as_secs(),
            "engine did not respond to cancellation within the grace window — exiting the worker \
             so the supervisor can recover the wedged GPU task"
        );
        guard.handle_mut().abort();
        std::process::exit(70);
    }
    // Loop exited cleanly — reclaim the handle (disarming the drop-guard) and join the finished task.
    let result = guard
        .into_handle()
        .await
        .map_err(|error| task_join_error("video task join", error))?;
    if stalled {
        return Err(WorkerError::Engine(format!(
            "Video generation stalled: no progress for {}s. The job was canceled.",
            stall_timeout.as_secs()
        )));
    }
    if canceled {
        // Reached only on a genuine user cancel — the stall/abandon watchdog returns above.
        // Generation has actually stopped now, so post the TERMINAL Canceled here (not at the
        // earlier cancel poll, which only tripped the flag + showed "Cancelling…"). This terminal
        // write is what frees the worker row (`jobs_store::update_job_progress`), so it lands as
        // the worker returns to its claim loop — the next queued job waits only until the GPU is
        // genuinely free (sc-5516; mirrors the image path sc-5515).
        update_job(
            api,
            &job.id,
            video_progress(
                JobStatus::Canceled,
                ProgressStage::Canceled,
                1.0,
                CANCEL_MESSAGE,
                None,
                backend,
            ),
        )
        .await?;
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
    }
    // Post the video metrics (epic 10402): the resolved effective settings
    // (quant / sampler / scheduler / guidance / dims / seed, sc-10418) + model +
    // effective steps (sc-10406) folded with the per-phase timing (sc-10405).
    // into_metrics closes the decode span still open at completion (video emits no
    // decode-done event). Best-effort; coalesce-merges with the S2 hardware block
    // server-side.
    let timing = phase_timer.into_metrics(Instant::now()).unwrap_or_default();
    let model = job
        .payload
        .get("model")
        .and_then(|value| value.as_str())
        .map(str::to_owned);
    let metrics = build_video_metrics(timing, &video_settings, model, video_effective_steps);
    crate::job_metrics::post_generation_metrics(api, &job.id, &metrics).await;
    result
}
