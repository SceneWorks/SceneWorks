use super::{
    admit_conditioning_paths, attach_manifest_text_encoder, gate_tier_key, gate_with_evict_reclaim,
    krea_model_subdir, lora_label, nvfp4_host_eligible, nvfp4_selected, pose_entries,
    resolve_adapters, resolve_advanced_or_manifest_u32, resolve_text_style_gain,
    run_candle_strict_control, trusted_control_weight_revision, AdapterSpec, ApiClient, CancelFlag,
    CandleStrictControl, Image, ImagePlan, ImageRequest, JobSnapshot, JsonObject, Path, PathBuf,
    Progress, Settings, Value, WorkerError, WorkerResult,
};
use super::{advanced, huggingface_snapshot_dir};
use super::{
    ensure_hf_cached_file, resolve_app_managed_model_dir, safe_weight_filename, DownloadContext,
};
use crate::conditioning_fit::ConditioningAdmission;
use serde_json::json;

// Candle (Windows/CUDA) Krea 2 pose-ControlNet route (sc-8464, epic 8459) — `krea_2_turbo` +
// `advanced.poses` off-Mac via `runtime_cuda::providers::krea::Krea2Control`. The first Krea backbone control lane and
// the deployable form of the sc-8460 spike: a trained control-branch overlay loaded on the frozen Krea 2
// Turbo base (dense bf16), rendering one image per library pose, each conditioned on a full DWPose
// skeleton (rendered cross-platform by `openpose_skeleton::draw_wholebody`, the SAME renderer training
// used). True pose lock via a residual added to the single CFG-free guidance forward, scaled by
// `control_scale`; `control_scale = 0` is engine-proven byte-identical to base txt2img.
//
// **Candle-only.** There is no MLX Krea control twin yet (8459 S5 / sc-8465); this whole file is gated to
// the Windows/CUDA candle build (the module declaration in image_jobs.rs carries the cfg). It is a child module of
// the `image_jobs` module, so it shares that module's imports (`parse_poses`/`pose_entries`/`Settings`/
// `WorkerResult`/`huggingface_snapshot_dir`/`start_gen_stream`/… all in scope unqualified).
//
// The base is any complete Krea 2 Turbo diffusers snapshot (`transformer/ text_encoder/ vae/ tokenizer/`):
// the legacy dense `krea/Krea-2-Turbo`, OR — the common case now that the dense download is retired — the
// installed `SceneWorks/krea-2-turbo-mlx` tier the txt2img lane uses (q8 default / q4 / bf16). The control
// branch is a composable-forward overlay (`KreaTrainDit`) trained against the bf16 base; the packed q4/q8
// tiers are key-compatible and load via candle-gen's dequant-on-load (composable DiT reconstructs the dense
// grid from the packed triple — candle-gen #471, sc-11727), so q8 renders ≈ bf16 and q4 stays pose-locked.
// Peak VRAM ≈ dense (dequant-to-bf16 in VRAM), well within a single 96 GB card.

/// The dense Krea 2 Turbo diffusers repo when the manifest omits `repo` — a bring-your-own / legacy base
/// (the manifest download entry was retired in favor of the `SceneWorks/krea-2-turbo-mlx` tiers, sc-9092).
/// The control provider loads the dense bf16 composable base the overlay trained on; the packed mlx tiers
/// below are key-compatible (the bf16 tier mirrors this tree) and load via candle-gen's dequant-on-load.
const KREA_CONTROL_BASE_REPO: &str = "krea/Krea-2-Turbo";
/// The `SceneWorks/krea-2-turbo-mlx` turnkey (q8 default / q4 / bf16 self-contained subdirs) — the SAME
/// base the txt2img `krea_2_turbo` lane installs and loads. Now that the dense `krea/Krea-2-Turbo` download
/// is retired, this is what a user actually has on disk, so the control base resolves the installed tier
/// here (via the shared [`krea_model_subdir`]) when the legacy dense repo is absent. candle-gen packed-
/// detects the tier and the composable control DiT dequantizes it on load (candle-gen #471, sc-11727):
/// q8 renders ≈ bf16, q4 stays pose-locked (mild haze) — GPU-proven.
const KREA_CONTROL_MLX_REPO: &str = "SceneWorks/krea-2-turbo-mlx";
/// Pose ControlNet conditioning-scale default (candle-gen `Krea2Control::DEFAULT_CONTROL_SCALE`). The S0
/// spike found the usable band ~0.5–0.85 for the distilled CFG-free base; ship a comfortable mid.
const KREA_CONTROL_DEFAULT_SCALE: f32 = runtime_cuda::providers::krea::DEFAULT_CONTROL_SCALE;
/// Hard cap on the exposed `control_scale` — above ~0.85 the frozen CFG-free base over-drives to halftone
/// (S0 finding: graceful soft-haze, never confetti, but not a usable range).
const KREA_CONTROL_SCALE_CAP: f32 = 0.85;
/// Denoise-steps default — the distilled Turbo schedule (8-step CFG-free).
const KREA_CONTROL_DEFAULT_STEPS: u32 = 8;
/// The adapter/engine id recorded on candle Krea control assets (distinct from the `candle_krea` txt2img
/// lane).
pub(super) const KREA_CONTROL_ENGINE: &str = "candle_krea_control";
/// The [`STRICT_CONTROL_ENGINES`] catalog id this lane validates `advanced.controlMode` against (the Krea
/// pose-only row — `{Pose}`).
pub(super) const KREA_CONTROL_ENGINE_ID: &str = "krea_2_turbo_control";
/// Env override pointing directly at a Krea 2 Turbo dense diffusers snapshot dir (validation / bring-your-
/// own base) — bypasses the HF-cache resolve.
const KREA_CONTROL_BASE_ENV: &str = "SCENEWORKS_KREA_CONTROL_BASE";
/// Env override pointing directly at a trained control-branch overlay `.safetensors` (validation against
/// the spike checkpoint / bring-your-own) — bypasses the hosted-overlay resolve + download.
const KREA_CONTROL_WEIGHTS_ENV: &str = "SCENEWORKS_CONTROLNET_KREA";
/// Default published Krea pose control-branch overlay repo (sc-8466) — the S0 spike (5,000-step)
/// checkpoint, hosted so the overlay downloads/provisions like the other control repos (the FLUX.2
/// `FLUX2_CONTROL_CANDLE_REPO` precedent) when the user hasn't selected a studio-trained overlay
/// (B4/sc-10165). EXPERIMENTAL / not-for-production: an 8-step CFG-free feasibility overlay, usable
/// pose-lock ~0.5–0.85 (S0). A studio-trained overlay (resolved to `controlWeights.path`) always overrides.
const KREA_CONTROL_OVERLAY_REPO: &str = "SceneWorks/krea2-pose-controlnet-beta";
/// The overlay weight file within [`KREA_CONTROL_OVERLAY_REPO`] (the final 5k-step checkpoint; the repo
/// also carries the 4.5k for comparison).
const KREA_CONTROL_OVERLAY_FILE: &str = "control_step5000.safetensors";
/// Pinned revision for the default overlay repo (defense-in-depth: `main` moving under us can't swap the
/// checkpoint we load — mirrors `FLUX2_CONTROL_CANDLE_REVISION` / sc-9879). Registered overlays carry
/// their own catalog-authorized immutable revision. `ensure_hf_cached_file` still verifies the file's
/// `lfs.oid` from HF's tree API.
pub(super) const KREA_CONTROL_OVERLAY_REVISION: &str = "cb3a0ac7590f5ec594a4eeb43b95ee1da0b5a0ac";

/// The Krea control fit-ladder tier for the base directory the resolver will actually load.
///
/// Standard turnkey basenames are authoritative because `krea_model_subdir` can clamp/fall back away
/// from the requested bits. Opaque dense roots deliberately fall through to the request key; NVFP4 is
/// likewise used only when no standard tier basename resolved.
fn krea_control_gate_tier(
    convrot_resolved: bool,
    resolved_base: &Path,
    advanced: &JsonObject,
    manifest_entry: &JsonObject,
    nvfp4: bool,
) -> &'static str {
    gate_tier_key(
        convrot_resolved,
        resolved_base,
        advanced,
        manifest_entry,
        nvfp4,
    )
}

/// Verify that the live request is inside sc-16013's exact rendered-device envelope: 1024² on sm_120,
/// the pinned shipping base tier, and the pinned default control overlay. Custom/legacy artifacts and
/// adapters may still run best-effort, but cannot inherit the calibrated hard-reject verdict.
fn krea_control_runtime_evidence_verified(
    request: &ImageRequest,
    settings: &Settings,
    tier: &str,
    base: &Path,
    control: &Path,
) -> bool {
    if request.width != 1024
        || request.height != 1024
        || crate::gpu::cached_compute_cap() != Some(12.0)
    {
        return false;
    }
    let Some(download) = request
        .model_manifest_entry
        .get("downloads")
        .and_then(Value::as_array)
        .and_then(|downloads| {
            downloads
                .iter()
                .find(|download| download.get("variant").and_then(Value::as_str) == Some(tier))
        })
    else {
        return false;
    };
    let (Some(provider), Some(repository), Some(revision)) = (
        download.get("provider").and_then(Value::as_str),
        download.get("repo").and_then(Value::as_str),
        download.get("revision").and_then(Value::as_str),
    ) else {
        return false;
    };
    let Some(pinned_base_root) = crate::model_jobs::huggingface_pinned_snapshot_dir(
        &settings.data_dir,
        repository,
        revision,
    ) else {
        return false;
    };
    if crate::vram_gate::KreaRuntimeEvidenceContext::inspect(
        KREA_CONTROL_ENGINE_ID,
        "candle",
        &settings.gpu_id,
        crate::gpu::cached_compute_cap(),
        provider,
        repository,
        revision,
        tier,
        base,
        &pinned_base_root,
    )
    .is_none()
    {
        return false;
    }
    let Some(pinned_control_root) = crate::model_jobs::huggingface_pinned_snapshot_dir(
        &settings.data_dir,
        KREA_CONTROL_OVERLAY_REPO,
        KREA_CONTROL_OVERLAY_REVISION,
    ) else {
        return false;
    };
    match (
        control.canonicalize().ok(),
        pinned_control_root
            .join(KREA_CONTROL_OVERLAY_FILE)
            .canonicalize()
            .ok(),
    ) {
        (Some(actual), Some(expected)) => actual == expected,
        _ => false,
    }
}

/// Model ids the candle Krea strict-pose control route accepts (the deployed base the overlay applies on).
fn is_krea_control_model(model: &str) -> bool {
    model == "krea_2_turbo"
}

/// Resolve the Krea 2 Turbo dense diffusers snapshot: the `SCENEWORKS_KREA_CONTROL_BASE` env → an explicit
/// `modelPath` (advanced or manifest) → the HF cache snapshot for the manifest `repo` (default
/// `krea/Krea-2-Turbo`). `None` ⇒ not present locally (the job is not candle-runnable). Mirrors
/// `resolve_flux2_control_base`.
pub(super) fn resolve_krea_control_base(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    if super::wants_krea_convrot(request) {
        return Ok(super::resolve_krea_convrot(request, settings).map(|(root, _)| root));
    }
    if let Ok(env_dir) = std::env::var(KREA_CONTROL_BASE_ENV) {
        let p = PathBuf::from(env_dir.trim());
        if p.is_dir() {
            return Ok(Some(p));
        }
    }
    if let Some(path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
    {
        return resolve_app_managed_model_dir(settings, &path, "Krea control modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            crate::engines::default_repo_for(&request.model).unwrap_or(KREA_CONTROL_BASE_REPO)
        });
    if let Some(root) = huggingface_snapshot_dir(&settings.data_dir, repo) {
        if repo == KREA_CONTROL_MLX_REPO {
            let tier = krea_model_subdir(&root, request);
            if tier.join("transformer").is_dir() {
                return Ok(Some(tier));
            }
        } else {
            // Explicit legacy / bring-your-own dense diffusers base.
            return Ok(Some(root));
        }
    }
    // The installed `SceneWorks/krea-2-turbo-mlx` tier the user actually has (q8 default / q4 / bf16),
    // resolved EXACTLY like the txt2img lane (`krea_model_subdir` honours `advanced.mlxQuantize` and falls
    // back to any downloaded tier — so a q4-only or q8-only install resolves). candle-gen `from_dir`
    // packed-detects the tier; the composable control DiT dequantizes the packed base on load (candle-gen
    // #471, sc-11727). Gate on `transformer/` so a partial download surfaces "base not installed" rather
    // than half-loading.
    if let Some(root) = huggingface_snapshot_dir(&settings.data_dir, KREA_CONTROL_MLX_REPO) {
        let tier = krea_model_subdir(&root, request);
        if tier.join("transformer").is_dir() {
            return Ok(Some(tier));
        }
    }
    Ok(None)
}

/// True when this is a candle-eligible Krea 2 strict-pose job: `krea_2_turbo` with a non-empty
/// `advanced.poses`, not edit mode, whose dense base resolves locally. Mirrors
/// `jobs_store::krea_control_candle_eligible` so the worker and router agree. The overlay weights are NOT
/// part of the gate: they are resolved on first use in the stream.
pub(super) fn krea_control_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_krea_control_model(&request.model)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && if super::wants_krea_convrot(request) {
            super::resolve_krea_convrot_dit(settings).is_some()
        } else {
            matches!(resolve_krea_control_base(request, settings), Ok(Some(_)))
        }
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → default (8).
fn krea_control_candle_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", KREA_CONTROL_DEFAULT_STEPS, 1..=50)
}

/// The (repo, filename) of the hosted control overlay — `advanced.controlWeights.{repo,filename}`
/// overrides (a not-yet-cached registered/hosted overlay the API passed through), else the default
/// published beta overlay. Mirrors `flux2_control_candle_repo_file`; the filename must be a plain
/// component (sc-8821 / F-019).
pub(super) fn krea_control_overlay_repo_file(
    request: &ImageRequest,
) -> WorkerResult<(String, String)> {
    let cw = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object);
    let pick = |key: &str, default: &str| {
        cw.and_then(|m| m.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or(default)
            .to_owned()
    };
    let repo = pick("repo", KREA_CONTROL_OVERLAY_REPO);
    let file = safe_weight_filename(
        &pick("filename", KREA_CONTROL_OVERLAY_FILE),
        "advanced.controlWeights.filename",
    )?;
    trusted_control_weight_revision(request, KREA_CONTROL_ENGINE_ID, &repo, &file)?;
    Ok((repo, file))
}

/// Confine a payload-supplied `advanced.controlWeights.path` to an app-managed root (sc-11168 / F-006).
/// The API writes this key for a studio-trained / registered LOCAL overlay (B4/sc-10165), but the value
/// arrives untrusted across the LAN boundary (epic 4484), so — like every other on-disk model input — it
/// must resolve under the app data dir / HF hub cache (or a declared external root) via the house
/// `normalize_app_managed_model_path`; without this a crafted job could point the loader at any file on
/// disk (an arbitrary-file-read primitive). Returns `Ok(None)` when the payload carries no path, `Ok(Some)`
/// for a confined path (whether or not it exists — the caller checks `is_file`), and the same
/// `InvalidPayload` rejection as the sibling lanes for an out-of-root path. Mirrors the MLX twin.
pub(super) fn krea_control_payload_overlay_path(
    settings: &Settings,
    request: &ImageRequest,
) -> WorkerResult<Option<PathBuf>> {
    let Some(path) = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object)
        .and_then(|cw| cw.get("path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    Ok(Some(crate::paths::normalize_app_managed_model_path(
        settings,
        path,
        "Krea control overlay",
    )?))
}

/// Resolve the control-branch overlay checkpoint the `Krea2Control` provider loads, downloading on first
/// use. Order (most specific wins): the `SCENEWORKS_CONTROLNET_KREA` env (validation / bring-your-own) → an
/// `advanced.controlWeights.path` (a studio-trained or registered LOCAL overlay the API resolved,
/// B4/sc-10165) → an `advanced.controlWeights.{repo,filename}` hosted override / the default published
/// overlay repo (`SceneWorks/krea2-pose-controlnet-beta`, sc-8466), fetched into the app cache. The
/// ~6.6 GB overlay is lazy-fetched only on the first Krea pose job (vs bloating the base download),
/// mirroring `ensure_flux2_control_candle_weights`.
async fn ensure_krea_control_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    // 1. Env override — a local overlay `.safetensors` (validation against the spike, bring-your-own).
    if let Ok(p) = std::env::var(KREA_CONTROL_WEIGHTS_ENV) {
        let p = PathBuf::from(p.trim());
        if p.is_file() {
            return Ok(p);
        }
    }
    // 2. A LOCAL overlay path the API resolved from a studio-trained / registered overlay selection
    //    (B4/sc-10165 `resolve_control_overlay_selection` writes `advanced.controlWeights.path`),
    //    confined to an app-managed root (sc-11168 / F-006).
    if let Some(p) = krea_control_payload_overlay_path(settings, request)? {
        if p.is_file() {
            return Ok(p);
        }
    }
    // 3. A hosted overlay: a `controlWeights.{repo,filename}` override (a not-yet-cached registered/hosted
    //    overlay the API passed through) or the default published beta overlay — HF cache, else download.
    let (repo, file) = krea_control_overlay_repo_file(request)?;
    let revision = trusted_control_weight_revision(request, KREA_CONTROL_ENGINE_ID, &repo, &file)?;
    if let Some(snapshot) =
        crate::model_jobs::huggingface_pinned_snapshot_dir(&settings.data_dir, &repo, &revision)
    {
        let f = snapshot.join(&file);
        if f.is_file() {
            return Ok(f);
        }
    }
    let client = crate::downloads::streaming_download_client();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "Krea 2 strict-pose generation canceled while fetching control overlay.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-krea")
        .join(&file);
    // Pin the exact commit for the default overlay repo so `main` moving under us can't swap the
    // checkpoint (sc-8466 / sc-9879). Registered overlays carry their own immutable pin.
    ensure_hf_cached_file(&context, &repo, &revision, &file, &dst).await?;
    Ok(dst)
}

/// Flat telemetry recorded on candle Krea control assets.
fn krea_control_candle_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(KREA_CONTROL_ENGINE.to_owned()),
    );
    // User LoRA labels applied on top of the pose control branch (sc-11721) — mirrors the
    // `image_settings_metrics` `loras` field so the control-lane asset records what rode alongside the
    // pose lock. Omitted when no LoRA was requested (the sc-4408 omit-when-absent contract).
    let loras: Vec<Value> = request
        .loras
        .iter()
        .filter_map(lora_label)
        .map(Value::String)
        .collect();
    if !loras.is_empty() {
        raw.insert("loras".to_owned(), Value::Array(loras));
    }
    raw
}

/// The per-lane half of the candle Krea strict-control [`CandleStrictControl`] driver: the resolved base +
/// overlay paths + request numerics. Krea 2 Turbo is CFG-free (no guidance / negative pass) and bf16
/// (no quant tier). Moved onto the blocking thread, loaded once, drives every pose.
pub(super) struct KreaStrictControl {
    base: PathBuf,
    /// Complete API-prepared load receipt retained through the bespoke provider construction.
    load_spec: gen_core::LoadSpec,
    /// Immutable INT8-ConvRot DiT selected by the request. `None` loads `base/transformer`.
    convrot_dit: Option<PathBuf>,
    control: PathBuf,
    /// User LoRA/LoKr adapters applied additively to the frozen base DiT (sc-11721) — a character/style
    /// adapter reshapes the subject while the control branch keeps the pose lock. Empty ⇒ stock control.
    adapters: Vec<AdapterSpec>,
    /// The tier the control branch is packed to — **tier integrity, not a rung** (sc-15799). Derived
    /// from the resolved BASE tier by `krea_control_fit::control_branch_tier_for_key`, so it is the same
    /// on a 16 GB card and a 96 GB one: a q8 base carries a q8 branch, a q4 base floors at q8 (the
    /// declared, measured exception), and only a dense base carries a dense branch. Folds the ~6.6 GB
    /// published bf16 branch onto the GPU packed (dequant-on-forward) so a packed render never holds
    /// precision it did not ask for.
    branch_tier: Option<gen_core::Quant>,
    /// Force the seam-free tiled VAE decode (sc-11744) — the fit ladder's first speed-cost rung after
    /// sequential residency, engaged only when the predicted decode-phase peak exceeds free VRAM.
    /// `false` (the big-card default) is the monolithic full-speed decode. A *speed* cost, no quality cost.
    tile_vae_decode: bool,
    /// Engage sc-6217-style query-row attention chunking on the composable base stack + control branch
    /// (sc-11745, candle-gen #496) — the fit ladder's DEEPEST rung, engaged only when the predicted
    /// denoise-phase activation peak exceeds free VRAM. `false` (the big-card default) is the unchunked
    /// full-speed forward. A *speed* cost (~+6%), byte-identical output.
    chunk_attention: bool,
    /// Request-scoped residency selection for this bespoke provider. It is not a registered generator,
    /// so the control fit gate owns this decision instead of consulting `supports_sequential_offload`.
    stage_residency: bool,
    prompt: String,
    width: u32,
    height: u32,
    steps: u32,
    control_scale: f32,
    /// Krea "text style" tap-reweight gain (sc-12009) — self-gates on `ui.textStyleGain` (Krea only);
    /// `None`/g≈1 is a byte-exact no-op. Applied to the pose-control lane's CFG-free context by the
    /// engine (inference sc-12009, `Krea2ControlRequest.text_style_gain`).
    text_style_gain: Option<f32>,
}

/// Routing/wiring fixture for the Krea strict-control provider — dummy paths, no load. Mirrors
/// [`super::qwen_control::qwen_strict_control_test_fixture`].
#[cfg(test)]
pub(super) fn krea_strict_control_test_fixture(path: PathBuf) -> KreaStrictControl {
    KreaStrictControl {
        base: path.clone(),
        load_spec: gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(path.clone()))
            .with_control(gen_core::WeightsSource::File(path.clone())),
        convrot_dit: None,
        control: path,
        adapters: Vec::new(),
        branch_tier: None,
        tile_vae_decode: false,
        chunk_attention: false,
        stage_residency: false,
        prompt: "p".to_owned(),
        width: 1024,
        height: 1024,
        steps: 8,
        control_scale: 0.7,
        text_style_gain: None,
    }
}

impl KreaStrictControl {
    /// Build this lane's bespoke request. Split out of [`CandleStrictControl::generate_one`] so the
    /// preview wiring is reachable without a loaded 20 GB provider — see
    /// `candle_strict_control_requests_carry_the_live_preview_sink` in `image_jobs::tests`, which
    /// calls this and asserts an emitted frame reaches the sink the driver supplied.
    ///
    /// `preview` is the job's live sink and is **cloned onto the request**, never defaulted (epic 16948,
    /// sc-16962). Krea 2 emits per-step latent previews from every render route as of inference
    /// `f94c0b1c` (sc-16950); before this the lane passed `Default::default()` and the user saw nothing.
    pub(super) fn control_request(
        &self,
        seed: u64,
        cancel: &CancelFlag,
        preview: &gen_core::PreviewSink,
    ) -> runtime_cuda::providers::krea::Krea2ControlRequest {
        runtime_cuda::providers::krea::Krea2ControlRequest {
            prompt: self.prompt.clone(),
            width: self.width,
            height: self.height,
            steps: self.steps as usize,
            control_scale: self.control_scale,
            text_style_gain: self.text_style_gain,
            seed,
            tile_vae_decode: self.tile_vae_decode,
            stage_residency: self.stage_residency,
            cancel: cancel.clone(),
            preview: preview.clone(),
        }
    }
}

impl CandleStrictControl for KreaStrictControl {
    type Model = runtime_cuda::providers::krea::Krea2Control;

    fn engine_id(&self) -> &'static str {
        KREA_CONTROL_ENGINE_ID
    }

    fn engine_label(&self) -> &'static str {
        KREA_CONTROL_ENGINE
    }

    fn stream_tag(&self) -> &'static str {
        "krea_control"
    }

    fn out_width(&self) -> u32 {
        self.width
    }

    fn out_height(&self) -> u32 {
        self.height
    }

    /// This lane is admitted in its OWN preamble — `generate_candle_krea_control_stream` runs both the
    /// shared weights floor ([`admit_conditioning_paths`]) and the measured [`crate::krea_control_fit`]
    /// ladder there — so the shared driver's gate stands down (sc-16069).
    ///
    /// **It is an ORDERING requirement, not an exemption.** The preamble records the peak it admitted at
    /// with [`crate::vram_gate::note_loaded_peak`]. A rejection arriving after that call would leave a
    /// reclaimable high-water standing for a load that never allocated — over-crediting the pool, which
    /// lets the NEXT gate over-admit an OOM. So the floor has to run BEFORE the ladder, which is before
    /// control ever reaches the driver. Re-gating in the driver would add a second `nvidia-smi` probe and
    /// risk a second generator eviction for no new information.
    ///
    /// The floor and the ladder answer different questions and both are needed here. Current measured
    /// tiers use the ladder for transient-aware rejection; at a tier with no priced row it cannot judge
    /// the render at all
    /// (`KreaControlFit::Unverified`). In both cases the floor is the only check that can still refuse a
    /// host which cannot hold the weights. Its footprint prices the files each route actually loads.
    fn conditioning_admission(&self) -> ConditioningAdmission {
        ConditioningAdmission::GatedInPreamble {
            gate: "krea_control_fit",
        }
    }

    fn load(&self) -> WorkerResult<Self::Model> {
        let paths = runtime_cuda::providers::krea::Krea2ControlPaths {
            root: self.base.clone(),
            convrot_dit: self.convrot_dit.clone(),
            native_dit: None,
            control: self.control.clone(),
            adapters: self.adapters.clone(),
            // Tier integrity (sc-15799): the branch's tier is a function of the base tier, decided
            // before the fit ladder runs and identical on every card.
            branch_tier: self.branch_tier,
            // Unchunked (full speed) by default; the fit ladder (sc-11745) forces query-row attention
            // chunking only to bound the denoise activation peak on a constrained card — byte-identical.
            chunk_attention: self.chunk_attention,
            // Compatibility-only load field; request-scoped residency is authoritative.
            offload_policy: gen_core::OffloadPolicy::Resident,
        };
        runtime_cuda::providers::krea::Krea2Control::load_with_spec(&paths, &self.load_spec)
            .map_err(|error| {
                WorkerError::Engine(format!("Krea 2 strict-pose control load failed: {error}"))
            })
    }

    fn generate_one(
        &self,
        model: &Self::Model,
        control: &Image,
        seed: u64,
        cancel: &CancelFlag,
        preview: &gen_core::PreviewSink,
        on_progress: &mut dyn FnMut(Progress),
    ) -> WorkerResult<Image> {
        let req = self.control_request(seed, cancel, preview);
        model.generate(&req, control, on_progress).map_err(|error| {
            WorkerError::Engine(format!("Krea 2 strict-pose generation failed: {error}"))
        })
    }
}

/// Real candle Krea 2 strict-pose generation: one image per pose, each conditioned on a full DWPose
/// skeleton (`controlMode` unset ⇒ pose) via a trained control-branch overlay on the frozen Turbo base
/// (sc-8464; engine sc-8462). Resolves the dense base + the overlay, then hands a [`KreaStrictControl`] to
/// the shared [`run_candle_strict_control`] driver (validation against `krea_2_turbo_control`'s
/// `supported_kinds` = {Pose}, per-pose skeleton rendering, scoring). Krea is CFG-free bf16. The pose path
/// is byte-preserved; `control_scale = 0` is byte-identical to base.
pub(super) async fn generate_candle_krea_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    if super::wants_krea_convrot(request) {
        super::ensure_krea_convrot_base_present(api, settings, job, request).await?;
    }
    let (base, convrot_dit) = if super::wants_krea_convrot(request) {
        super::resolve_krea_convrot(request, settings).map(|(root, dit)| (root, Some(dit)))
    } else {
        resolve_krea_control_base(request, settings)?.map(|root| (root, None))
    }
    .ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Krea 2 Turbo control base weights not found for the selected tier".to_owned(),
        )
    })?;
    let control = ensure_krea_control_weights(api, settings, job, request).await?;
    // User LoRA/LoKr adapters ride additively on the frozen base DiT (sc-11721 / candle-gen sc-11720):
    // resolved + path-confined by the shared helper (enforces MAX_JOB_LORAS + `normalize_app_managed_
    // lora_path`), then installed on the base at load — the pose control branch is never adapted.
    let adapters = resolve_adapters(request, settings)?;
    if convrot_dit.is_some() && !adapters.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 INT8-ConvRot pose control does not support LoRA/LoKr or diff-patch adapters"
                .to_owned(),
        ));
    }
    let adapter_bytes =
        gen_core::adapter_stack_resident_bytes(&adapters, gen_core::AdapterResidencyMode::Additive)
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "Krea 2 cannot determine the resident size of the requested adapter stack."
                        .to_owned(),
                )
            })?;

    let steps = krea_control_candle_steps(request);
    let control_scale = advanced::f32_clamped(
        &request.advanced,
        "controlScale",
        KREA_CONTROL_DEFAULT_SCALE,
        0.0..=KREA_CONTROL_SCALE_CAP,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            crate::engines::default_repo_for(&request.model).unwrap_or(KREA_CONTROL_BASE_REPO)
        })
        .to_owned();

    let pose_count = pose_entries(request).len();
    let raw_settings =
        krea_control_candle_raw_settings(request, &repo, steps, control_scale, pose_count);

    // VRAM fit ladder (sc-11754, epic 8459 → epic 10765). The control lane is diverted around the base.rs
    // `generate_candle_stream` fit-gate, so it gets its own here: predict the control-lane peak (base tier
    // + the shipping per-tier control branch + activations + the end-of-render VAE-decode spike) and
    // compare it against live/capped free VRAM. On a big card nothing engages. On a constrained card (or
    // one emulated via `SCENEWORKS_CUDA_VRAM_CAP_GB`) the ladder first stages the text and heavy phases
    // sequentially (sc-12176), then engages VAE-decode tiling (sc-11744) and attention chunking
    // (sc-11745) until it fits, else rejects-before-OOM. Branch precision is fixed before this ladder.
    //
    // sc-13588 / sc-13960: this lane loads through the UNcached `start_gen_stream` (it never evicts the
    // single-slot generator cache), so a resident txt2img generator stays co-resident and its cudarc pool
    // pages are NOT free — the ladder gates against RAW live free first (crediting those pages against a
    // raw gate would over-admit an OOM). `gate_with_evict_reclaim` (base.rs) adds the missing lever: only
    // when reclaiming the pool would readmit this render at a higher residency does it EVICT that
    // generator (making the credit honest, at the cost of the warm txt2img cache) and act on the reclaimed
    // plan — fixing the repeated-control-render needless downtier / reject. Same treatment as
    // `qwen_edit_candle.rs`; base.rs already gets it for free via the evicting cache. The admitted peak is
    // recorded (`note_loaded_peak` below) so a repeated control render can reclaim these pooled pages.
    // sc-11042 / sc-13619: budget the tier `resolve_krea_control_base` ACTUALLY selected. The turnkey
    // resolver clamps/falls back to an installed tier, so request bits can disagree with the directory
    // that loads (for example, requested q4 with only q8 installed). Dense/opaque roots have no tier
    // basename and deliberately retain the request/NVFP4 fallback.
    let tier = krea_control_gate_tier(
        convrot_dit.is_some(),
        &base,
        &request.advanced,
        &request.model_manifest_entry,
        nvfp4_selected(request, nvfp4_host_eligible(), Some(&base)),
    );
    // Tier integrity (sc-15799): the control branch's tier is derived from the resolved BASE tier, NOT
    // from the fit ladder, so it is decided here — before any budget is read — and holds on every path
    // below, `Unknown` included. A branch that only packs when VRAM is tight is precisely the defect this
    // removes: it left ~3.3 GB of precision a q8 render never requested resident on every roomy card
    // (the branch is 3.30 B params ≈ 6.6 GB bf16, ~3.3 GB at q8).
    let branch_tier = crate::krea_control_fit::control_branch_tier_for_key(tier);

    let mut memory_spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(base.clone()));
    memory_spec = match tier {
        "q4" => memory_spec.with_quant(gen_core::Quant::Q4),
        "q8" | "int8-convrot" => memory_spec.with_quant(gen_core::Quant::Q8),
        _ => memory_spec,
    };
    memory_spec = memory_spec.with_control(gen_core::WeightsSource::File(control.clone()));
    if !adapters.is_empty() {
        memory_spec = memory_spec.with_adapters(adapters.clone());
    }
    if let Some(convrot_dit) = convrot_dit.as_ref() {
        memory_spec = memory_spec.with_component(
            gen_core::KREA_CONVROT_DIT_COMPONENT,
            gen_core::WeightsSource::File(convrot_dit.clone()),
        );
    }
    memory_spec =
        attach_manifest_text_encoder(memory_spec, KREA_CONTROL_ENGINE_ID, request, settings)?;
    let selected_text_encoder = memory_spec.text_encoder.clone();
    let admitted_text_encoder = selected_text_encoder
        .as_ref()
        .map(|source| match source {
            gen_core::WeightsSource::Dir(path) | gen_core::WeightsSource::File(path) => {
                path.clone()
            }
        })
        .unwrap_or_else(|| base.join("text_encoder"));

    // Conditioning-overlay weights FLOOR (sc-16069) — run HERE, before the ladder, and therefore before
    // the `note_loaded_peak` below. Ordering is the whole reason this lane gates itself instead of through
    // `run_candle_strict_control` (`ConditioningAdmission::GatedInPreamble`): a rejection after
    // `note_loaded_peak` would leave a reclaimable high-water standing for a load that never allocated,
    // over-crediting the pool for the next gate.
    //
    // It is NOT redundant with the ladder. Current measured tiers receive transient-aware admission,
    // while a future or malformed tier it cannot price (`Unverified`) makes no memory claim at
    // all. Without this floor those paths reach allocation with no hard check, which is exactly the
    // sc-16069 defect the rest of this story removes everywhere else.
    //
    // The footprint counts the BASE ONLY when the branch is packed. The branch is folded onto the GPU at
    // `branch_tier` (sc-15799) — ~3.3 GB at q8, ~1.7 GB at q4 against a ~6.6 GB published bf16 checkpoint
    // — so pricing the file would over-count a packed base by several GB and could refuse a render that
    // fits, the one direction this gate must never take. A DENSE branch (`branch_tier == None`, i.e. a
    // bf16 base) loads at exactly the published bytes, so there it is counted and the floor is tighter.
    if let Some(convrot_dit) = convrot_dit.as_deref() {
        // The bf16 surface supplies only tokenizer/Qwen3-VL/VAE on this route; its dense transformer is
        // not loaded. Price the actual ConvRot DiT plus the two weight-bearing shared component dirs so
        // the floor neither hides the int8 trunk nor charges the unused bf16 trunk.
        let shared_components = [admitted_text_encoder.clone(), base.join("vae")];
        let shared_component_paths: Vec<&Path> =
            shared_components.iter().map(PathBuf::as_path).collect();
        admit_conditioning_paths(
            settings,
            "Krea 2 INT8-ConvRot",
            "pose-ControlNet shared components",
            convrot_dit,
            &shared_component_paths,
        )
        .await?;
    } else {
        // Price the exact component set so a replacement does not silently keep charging the
        // bundled encoder while omitting the selected file. Tokenizer/config files are negligible;
        // the weight-bearing set is transformer + selected encoder + VAE (+ dense branch).
        let transformer = base.join("transformer");
        let vae = base.join("vae");
        let mut component_paths = vec![admitted_text_encoder.as_path(), vae.as_path()];
        if branch_tier.is_none() {
            component_paths.push(control.as_path());
        }
        admit_conditioning_paths(
            settings,
            "Krea 2",
            "pose-ControlNet branch",
            &transformer,
            &component_paths,
        )
        .await?;
    }
    let provider_memory_contract = crate::inference_runtime::media()
        .memory_strategy_contract(KREA_CONTROL_ENGINE_ID, &memory_spec)
        .ok()
        .flatten();
    let memory_geometry = gen_core::MemoryGeometry {
        width: request.width,
        height: request.height,
        batch: 1,
        frames: 1,
        // The separately supplied pose image is one reference in the provider's own authoritative
        // evidence probe. A zero here makes a current measured cell structurally ineligible.
        reference_count: crate::krea_control_fit::KREA_CONTROL_REFERENCE_COUNT,
    };
    let runtime_evidence_verified = adapter_bytes == 0
        && selected_text_encoder.is_none()
        && krea_control_runtime_evidence_verified(request, settings, tier, &base, &control);

    let raw_budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    // sc-13960 two-pass: walk the ladder against raw free, then — only if reclaiming the cudarc pool
    // would step it off a needless rung (a strict improvement; the reclaimed budget only grows) — evict
    // the resident generator and act on the reclaimed walk.
    let (fit, _budget) = gate_with_evict_reclaim(
        &settings.gpu_id,
        raw_budget,
        // sc-16069: ONE seam (`fit_ladder_for_entry`) reads every manifest row for the resolved tier,
        // instead of the lane hand-threading six of them in. That is what makes the "block present but no
        // row for THIS tier" case expressible at all: as six separate `Option`s a missing row arrived as
        // an indistinguishable `None` and the ladder took the zero-adaptation big-card path in silence.
        // It also removes the standing risk of pairing one tier's peak with another tier's savings.
        |budget| {
            crate::krea_control_fit::fit_ladder_for_entry_with_runtime(
                &request.model_manifest_entry,
                tier,
                budget,
                adapter_bytes,
                memory_geometry,
                provider_memory_contract.as_ref(),
                runtime_evidence_verified,
            )
        },
        // Two non-fits differing only in their reported free number are the same non-outcome — a reclaim
        // that still won't fit must NOT trigger a pointless evict. `BestEffort` is the superseded-evidence
        // form of the same non-outcome (it always carries every measured rung, so only its numbers can
        // move), so it is paired here too. Any other change is a strict improvement, worth evicting the
        // warm txt2img cache for.
        |raw, reclaimed| {
            !matches!(
                (raw, reclaimed),
                (
                    crate::krea_control_fit::KreaControlFit::TooBig { .. },
                    crate::krea_control_fit::KreaControlFit::TooBig { .. }
                ) | (
                    crate::krea_control_fit::KreaControlFit::BestEffort { .. },
                    crate::krea_control_fit::KreaControlFit::BestEffort { .. }
                ) | (
                    // sc-16069: `Unverified` is budget-INDEPENDENT (the tier has no row to price at any
                    // budget), so reclaiming can never change it. Evicting the warm txt2img cache for it
                    // would be pure loss.
                    crate::krea_control_fit::KreaControlFit::Unverified { .. },
                    crate::krea_control_fit::KreaControlFit::Unverified { .. }
                )
            ) && raw != reclaimed
        },
    )
    .await?;
    // The peak this admitted load actually leaves in the cudarc pool (sc-13960) — recorded below as the
    // reclaimable high-water. Computed before the by-value match consumes `fit`; `None` on a reject.
    let incurred_peak = crate::krea_control_fit::incurred_peak_gb_with_adapter_bytes(
        &fit,
        &request.model_manifest_entry,
        tier,
        adapter_bytes,
    );
    let (offload_policy, tile_vae_decode, chunk_attention) = match fit {
        // Big-card fast path (or no signal): monolithic full-speed decode, unchunked attention.
        crate::krea_control_fit::KreaControlFit::Unknown
        | crate::krea_control_fit::KreaControlFit::Fits {
            offload_policy: gen_core::OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
            // Measured or estimate-scoped (sc-18097) — the knobs are identical; only
            // `incurred_peak_gb_with_adapter_bytes` (computed above) distinguishes them.
            estimate_scoped: _,
        } => (gen_core::OffloadPolicy::Resident, false, false),
        // Constrained card: the fit ladder engaged the cheapest sufficient set of rungs to fit —
        // sequential residency (sc-12176), the seam-free tiled VAE decode (sc-11744), and query-row
        // attention chunking (sc-11745). All three are speed-only; the branch tier is not among them.
        crate::krea_control_fit::KreaControlFit::Fits {
            offload_policy,
            tile_vae_decode: tile,
            chunk_attention: chunk,
            estimate_scoped: _,
        } => {
            tracing::info!(
                model = %request.model,
                tier,
                offload_policy = ?offload_policy,
                tile_vae_decode = tile,
                chunk_attention = chunk,
                branch_tier = ?branch_tier,
                "Krea control VRAM fit ladder: predicted peak exceeds free VRAM — engaging rungs \
                 (sequential residency, VAE-decode tiling, attention chunking) to fit"
            );
            (offload_policy, tile, chunk)
        }
        // The `candle.control` block exists but carries NO peak row for the resolved base tier
        // (sc-16069) — so the render cannot be priced. Previously this collapsed into `Unknown` and took
        // the big-card fast path with no log: no staging, no tiling, no chunking, no reject, no trace.
        // Now it is an explicit, named, logged decision that stages residency (the cheapest adaptation,
        // no quality cost) and still never rejects — there is no evidence to reject on. The hard check for
        // this tier is the `conditioning_fit` weights floor already run in this function's preamble above
        // (it must precede `note_loaded_peak`, hence not in `run_candle_strict_control`). Records no
        // reclaimable peak (`incurred_peak_gb` → None).
        crate::krea_control_fit::KreaControlFit::Unverified {
            offload_policy,
            tile_vae_decode: tile,
            chunk_attention: chunk,
            tier_key,
        } => {
            tracing::warn!(
                model = %request.model,
                tier = %tier_key,
                offload_policy = ?offload_policy,
                branch_tier = ?branch_tier,
                "Krea control VRAM fit ladder: the candle.control block has NO peak row for this base \
                 tier, so this render cannot be priced. Staging component residency (the cheapest, \
                 quality-free adaptation) and declining to reject on absent evidence. The on-disk \
                 weights floor already applied in this lane's preamble is the only hard check for this \
                 tier until it is measured, so a render whose TRANSIENTS overflow can still reach a \
                 reactive CUDA OOM. Add direct peakGbByTier and sequentialPeakGbByTier measurements \
                 for it."
            );
            (offload_policy, tile, chunk)
        }
        // Generic stale-evidence fallback: engage every speed-only rung and let the reactive CUDA-OOM
        // backstop decide rather than reject from a superseded upper bound.
        crate::krea_control_fit::KreaControlFit::BestEffort {
            offload_policy,
            tile_vae_decode: tile,
            chunk_attention: chunk,
            needed_gb,
            available_gb,
        } => {
            tracing::warn!(
                model = %request.model,
                tier,
                offload_policy = ?offload_policy,
                tile_vae_decode = tile,
                chunk_attention = chunk,
                branch_tier = ?branch_tier,
                needed_gb,
                available_gb,
                "Krea control VRAM fit ladder: predicted peak exceeds free VRAM at every rung, but the \
                 control-lane evidence is not current. Admitting best-effort with every speed-only rung \
                 engaged rather than rejecting from stale evidence."
            );
            (offload_policy, tile, chunk)
        }
        // Won't fit even at the deepest rung, on CURRENT evidence ⇒ reject before the reactive CUDA OOM.
        crate::krea_control_fit::KreaControlFit::TooBig {
            needed_gb,
            available_gb,
        } => {
            return Err(WorkerError::InvalidPayload(format!(
                "Krea 2 pose-ControlNet at the {tier} base tier needs ~{needed} GB of VRAM (with \
                 headroom, sequential residency + tiled VAE decode + attention chunking) but \
                 GPU {gpu} has ~{available} GB available. Lower the output resolution or run on a card \
                 with more VRAM.",
                needed = needed_gb.round() as i64,
                available = available_gb.round() as i64,
                gpu = settings.gpu_id,
            )));
        }
    };
    // sc-13960: record the admitted control peak so a repeated control render (or a following
    // txt2img/edit gate) can reclaim these pooled pages — the control lane recorded nothing before this,
    // which is why its own repeated renders could not reclaim. Dropped when `run_candle_strict_control`
    // returns; `None` (unmeasured / rejected) ⇒ no-op.
    if let Some(peak_gb) = incurred_peak {
        crate::vram_gate::note_loaded_peak(&settings.gpu_id, peak_gb);
    }

    let provider = KreaStrictControl {
        base,
        load_spec: memory_spec,
        convrot_dit,
        control,
        adapters,
        branch_tier,
        tile_vae_decode,
        chunk_attention,
        stage_residency: matches!(offload_policy, gen_core::OffloadPolicy::Sequential),
        prompt: request.prompt.clone(),
        width: request.width,
        height: request.height,
        steps,
        control_scale,
        text_style_gain: resolve_text_style_gain(request),
    };

    run_candle_strict_control(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        provider,
        raw_settings,
        asset_writes,
    )
    .await
}

#[cfg(test)]
mod krea_control_tier_reconcile_tests {
    use super::super::tier_resolver::NVFP4_TIER;
    use super::*;
    use serde_json::json;

    fn request(bits: i64) -> ImageRequest {
        ImageRequest::from_payload(
            json!({
                "model": "krea_2_turbo",
                "advanced": { "mlxQuantize": bits },
                "modelManifestEntry": {}
            })
            .as_object()
            .unwrap(),
        )
    }

    #[test]
    fn fit_ladder_sizes_the_resolved_base_tier_before_request_fallback() {
        let req = request(4);

        // Installed-tier fallback: a q4 request that resolved q8 must budget q8.
        assert_eq!(
            krea_control_gate_tier(
                false,
                Path::new("/cache/SceneWorks/krea-2-turbo-mlx/q8"),
                &req.advanced,
                &req.model_manifest_entry,
                false,
            ),
            "q8"
        );
        assert_eq!(
            crate::vram_gate::requested_tier_key(&req.advanced, &req.model_manifest_entry, false,),
            "q4",
            "the request key intentionally differs in this regression"
        );

        // Every recognized resolved tier is authoritative, independent of the requested q4.
        assert_eq!(
            krea_control_gate_tier(
                false,
                Path::new("/cache/SceneWorks/krea-2-turbo-mlx/q4"),
                &req.advanced,
                &req.model_manifest_entry,
                false,
            ),
            "q4"
        );
        assert_eq!(
            krea_control_gate_tier(
                false,
                Path::new("/cache/SceneWorks/krea-2-turbo-mlx/bf16"),
                &req.advanced,
                &req.model_manifest_entry,
                false,
            ),
            "bf16"
        );

        // An eligible NVFP4 selection survives only when the resolved directory is itself NVFP4/opaque;
        // a standard installed fallback remains authoritative.
        assert_eq!(
            krea_control_gate_tier(
                false,
                Path::new("/cache/SceneWorks/krea-2-turbo-mlx/nvfp4"),
                &req.advanced,
                &req.model_manifest_entry,
                true,
            ),
            NVFP4_TIER
        );
        assert_eq!(
            krea_control_gate_tier(
                false,
                Path::new("/cache/SceneWorks/krea-2-turbo-mlx/q8"),
                &req.advanced,
                &req.model_manifest_entry,
                true,
            ),
            "q8"
        );

        // Bring-your-own dense snapshots have opaque basenames, preserving the request-derived fallback.
        assert_eq!(
            krea_control_gate_tier(
                false,
                Path::new("/models/krea-dense-snapshot"),
                &req.advanced,
                &req.model_manifest_entry,
                false,
            ),
            "q4"
        );
        assert_eq!(
            krea_control_gate_tier(
                false,
                Path::new("/models/krea-dense-snapshot"),
                &req.advanced,
                &req.model_manifest_entry,
                true,
            ),
            NVFP4_TIER
        );

        // SC-16453: the immutable ConvRot DiT identity wins even though its shared tokenizer/TE/VAE
        // surface is the `bf16/` directory. Dropping this identity aliases the load back to bf16.
        assert_eq!(
            krea_control_gate_tier(
                true,
                Path::new("/cache/SceneWorks/krea-2-turbo-mlx/bf16"),
                &req.advanced,
                &req.model_manifest_entry,
                false,
            ),
            super::super::tier_resolver::INT8_CONVROT_TIER
        );
    }
}
