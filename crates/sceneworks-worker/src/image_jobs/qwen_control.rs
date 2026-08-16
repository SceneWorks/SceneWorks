use super::advanced;
use super::{
    ensure_hf_cached_file, huggingface_snapshot_dir, resolve_app_managed_model_dir,
    safe_weight_filename, standard_tier_subdir, DownloadContext,
};
use super::{
    pose_entries, resolve_adapters, resolve_advanced_or_manifest_f32,
    resolve_advanced_or_manifest_u32, run_candle_strict_control, trusted_control_weight_revision,
    ApiClient, CancelFlag, CandleStrictControl, Image, ImagePlan, ImageRequest, JobSnapshot,
    JsonObject, Path, PathBuf, Progress, QwenFunControl, QwenFunControlPaths,
    QwenFunControlRequest, Settings, Value, WorkerError, WorkerResult,
};
use crate::conditioning_fit::{ConditioningAdmission, ConditioningFootprint};
use serde_json::json;

// Candle (Windows/CUDA) Qwen-Image 2512-Fun-Controlnet-Union (strict control) route (sc-5489 origin /
// sc-8350 repoint, epic 8236) — `qwen_image` + `advanced.poses` off-Mac via
// `runtime_cuda::providers::qwen_image::QwenFunControl`. The candle sibling of the MLX Qwen 2512-Fun strict-control path
// (qwen.rs `generate_qwen_control_stream`): one image per pose (or, with `advanced.controlMode =
// canny|depth` + a source, an auto-derived canny / Depth-Anything-V2 map), each fed to the VACE-style
// 2512-Fun-Controlnet-Union branch overlaid on the Qwen-Image-2512 base. sc-9870: the control overlay is
// now the SceneWorks PACKED tier (`SceneWorks/qwen-image-2512-fun-controlnet-union`, per-quant q4/q8/bf16
// subdirs), resolved per `advanced.mlxQuantize`, replacing the dense alibaba-pai overlay staging.
//
// **sc-8350 source swap.** This lane previously loaded the InstantX `Qwen-Image-ControlNet-Union`
// checkpoint (`QwenControl`, a residual-ControlNet on the `Qwen/Qwen-Image` base). It now rides the
// 2512-Fun-Union VACE engine (`QwenFunControl`) on the `Qwen/Qwen-Image-2512` base — input-agnostic
// (pose/canny/depth, no mode index), matching the `STRICT_CONTROL_ENGINES` `qwen_image_control` row. The
// candle-gen InstantX `control.rs` engine (`QwenControl`) stays in the crate but is no longer used by the
// worker.
//
// **Candle-only.** macOS keeps the MLX `qwen_image_control` registry generator; the candle `QwenFunControl`
// is a bespoke provider, so this whole file is gated to the Windows/CUDA candle build (the module
// declaration in image_jobs.rs carries the cfg). It is a child module of the `image_jobs` module, so it
// shares that
// module's imports (`parse_poses`/`Settings`/`WorkerResult`/`huggingface_snapshot_dir`/
// `ensure_hf_cached_file`/`start_gen_stream`/… all in scope unqualified).

/// Default 2512-Fun-Controlnet-Union weights (Apache-2.0, input-agnostic VACE control). As of sc-9870
/// (epic 8236) this points at the SceneWorks PACKED control tier — a per-quant matrix whose q4/ q8/ bf16/
/// subdirs each ship a single `model.safetensors` overlay — NOT the old dense alibaba-pai overlay
/// (sc-8350). The exact subdir is selected per `advanced.mlxQuantize` by [`qwen_control_tier_subdir`] so
/// the control overlay tier tracks the base transformer tier for a coherent A/B. The candle
/// `QwenFunControl` engine already packed-detects the overlay (sc-9869), so nothing downstream changes.
/// Same repo the MLX path uses (`qwen.rs` — the shared `STRICT_CONTROL_ENGINES` `qwen_image_control` row).
const QWEN_CONTROL_REPO: &str = "SceneWorks/qwen-image-2512-fun-controlnet-union";
/// Pinned revision for the default `QWEN_CONTROL_REPO` (sc-9879, F-077 follow-up). Fetching the mutable
/// `main` branch means a re-push (or a compromised token) could silently swap the ControlNet overlay we
/// load; pin the exact commit for defense-in-depth (mirrors the other candle control lanes, e.g.
/// `FLUX1_CONTROL_CANDLE_REVISION`). Registered overlays carry their own catalog-authorized immutable
/// revision. The pin is the packed-tier repo's `main` HEAD as of the sc-9870 repoint. HF's tree API still
/// reports each tier file's `lfs.oid`, which `ensure_hf_cached_file` verifies against.
#[cfg(test)]
pub(super) const QWEN_CONTROL_REVISION: &str = "a061fbc42a4744d6a7ec206370fbd3a37d4a7cca";
/// The single packed control file inside each tier subdir (`q4/`, `q8/`, `bf16/`). Deterministic —
/// the packed tier ships exactly one `model.safetensors` per subdir, so the sc-8350 two-overlay
/// ambiguity is naturally resolved.
pub(super) const QWEN_CONTROL_FILE: &str = "model.safetensors";
/// The Qwen-Image-2512 base diffusers repo when the manifest omits `repo` (the 2512-Fun base, sc-8350).
pub(super) const QWEN_CONTROL_DEFAULT_REPO: &str = "Qwen/Qwen-Image-2512";
/// ControlNet conditioning-scale default (the strict-pose tier).
pub(super) const QWEN_CONTROL_DEFAULT_SCALE: f32 = 1.0;
/// Denoise-steps default (Qwen-Image production).
const QWEN_CONTROL_DEFAULT_STEPS: u32 = 30;
/// CFG default.
const QWEN_CONTROL_DEFAULT_GUIDANCE: f32 = 4.0;
/// The adapter/engine id recorded on candle Qwen control assets (distinct from the txt2img
/// `candle_qwen` lane).
pub(super) const QWEN_CONTROL_ENGINE: &str = "candle_qwen_control";
/// The [`STRICT_CONTROL_ENGINES`] catalog id this candle lane validates `advanced.controlMode` against
/// (the `qwen_image_control` row — `{Pose, Canny, Depth}`). As of sc-8350 the candle lane loads the
/// 2512-Fun-Controlnet-Union checkpoint on the Qwen-Image-2512 base (`QwenFunControl`); sc-9870 repoints
/// the control overlay at the packed tier, matching the table's `qwen_image_control` repo
/// (`SceneWorks/qwen-image-2512-fun-controlnet-union`) exactly — consistent with the MLX `qwen.rs` lane.
pub(super) const QWEN_CONTROL_ENGINE_ID: &str = "qwen_image_control";

/// Model ids the candle Qwen ControlNet route accepts.
fn is_qwen_control_model(model: &str) -> bool {
    model == "qwen_image"
}

/// Resolve the Qwen-Image-2512 base (diffusers) snapshot: an explicit `modelPath` (advanced or manifest) →
/// the HF cache snapshot for the manifest `repo` (default `Qwen/Qwen-Image-2512`, sc-8350). `None` ⇒ not
/// present locally (the candle lane refuses the job; no fallback is attempted). Mirrors
/// `resolve_kolors_ipadapter_base`.
fn resolve_qwen_control_base(
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
        return resolve_app_managed_model_dir(settings, &path, "Qwen control modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            crate::engines::default_repo_for(&request.model).unwrap_or(QWEN_CONTROL_DEFAULT_REPO)
        });
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo)
        .map(|root| standard_tier_subdir(&root, request)))
}

/// True when this is a candle-eligible Qwen strict-pose job: `qwen_image` with a non-empty
/// `advanced.poses`, not edit mode, whose base resolves locally. Mirrors
/// `jobs_store::qwen_control_candle_eligible` so the worker and router agree.
pub(super) fn qwen_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_qwen_control_model(&request.model)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_qwen_control_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=100) → manifest `steps` → default (30).
fn qwen_control_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", QWEN_CONTROL_DEFAULT_STEPS, 1..=100)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → default (4.0), clamped.
fn qwen_control_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        QWEN_CONTROL_DEFAULT_GUIDANCE,
        0.0..=30.0,
    )
}

/// The packed control tier subdir the request's `advanced.mlxQuantize` selects (sc-9870): `bf16` (opt
/// out of quantization, `<= 0` / "none"), `q8` (`> 4`), `q4` for an explicit Q4 pick (`1..=4`), else —
/// with NO explicit `mlxQuantize` — the **`q8`** default (sc-10726) — the SAME mapping
/// [`standard_tier_subdir`] uses for the base transformer tier, so the control overlay tier tracks the
/// base tier for a coherent A/B (a q8-default base pairs with the q8 control overlay). The whole control
/// matrix (q4/q8/bf16) installs as co-requisites alongside the base, so the q8 overlay is on disk
/// whenever the base is.
///
/// The Q8 default is CLAMPED to what's installed (sc-10726): `snapshot` is the control repo's HF cache
/// snapshot, and — with NO explicit pick — the resolver picks the highest CLEAN overlay tier whose
/// `<tier>/model.safetensors` is actually on disk (q8 → bf16 → q4), so a legacy/partial install that
/// carries only the q4 overlay resolves q4 rather than forcing a NEW q8 fetch (via
/// [`ensure_qwen_control_weights`]) on the plain default path. An EXPLICIT pick is returned as-is (it
/// may legitimately fetch its tier on demand), and an absent `snapshot` falls back to `q8`, the
/// fresh-install co-requisite the base download brings. Mirrors the MLX `qwen_control_tier_subdir`.
fn qwen_control_tier_subdir(request: &ImageRequest, snapshot: Option<&Path>) -> &'static str {
    let bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    match bits {
        Some(b) if b <= 0 => "bf16",
        Some(b) if b > 4 => "q8",
        Some(_) => "q4",
        // Default: the app-wide Q8 default CLAMPED to the installed overlay — never a new fetch.
        None => qwen_control_installed_default_tier(snapshot),
    }
}

/// The highest CLEAN control overlay tier (q8 → bf16 → q4) whose `<tier>/model.safetensors` is present
/// in the control repo `snapshot`, else `"q8"` (the fresh-install co-requisite default). Keeps a plain
/// default job from forcing a NEW on-demand fetch on a legacy/partial install that carries only q4
/// (sc-10726) — the default tier only wins when it is actually on disk. Mirrors the MLX
/// `qwen_control_installed_default_tier`.
fn qwen_control_installed_default_tier(snapshot: Option<&Path>) -> &'static str {
    let present = |tier: &str| {
        snapshot
            .map(|root| root.join(tier).join(QWEN_CONTROL_FILE).is_file())
            .unwrap_or(false)
    };
    if present("q8") {
        "q8"
    } else if present("bf16") {
        "bf16"
    } else if present("q4") {
        "q4"
    } else {
        "q8"
    }
}

/// The (repo, repo-relative file path) of the ControlNet weights.
///
/// Default (sc-9870): the SceneWorks packed control tier — repo [`QWEN_CONTROL_REPO`], file
/// `<tier>/model.safetensors` where `<tier>` is [`qwen_control_tier_subdir`] (per `advanced.mlxQuantize`).
/// Deterministic single-file resolution — each tier subdir ships exactly one `model.safetensors`.
///
/// A registered overlay may provide a catalog-authorized repo/file/revision tuple. When a filename is
/// present the tier subdir is NOT applied (the tuple addresses a specific file directly).
pub(super) fn qwen_control_repo_file(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<(String, String)> {
    let cw = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object);
    let pick = |key: &str| {
        cw.and_then(|m| m.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_owned)
    };
    let repo = pick("repo").unwrap_or_else(|| QWEN_CONTROL_REPO.to_owned());
    let file = match pick("filename") {
        // Explicit override — a plain-component file in the override repo (no tier subdir).
        Some(name)
            if sceneworks_core::control_weights::shipped_control_weight(
                QWEN_CONTROL_ENGINE_ID,
                &repo,
                &name,
            )
            .is_some() =>
        {
            name
        }
        Some(name) => safe_weight_filename(&name, "advanced.controlWeights.filename")?,
        // Default packed tier — `<tier>/model.safetensors` selected by `advanced.mlxQuantize`, whose
        // DEFAULT tier is clamped to the installed overlay so a plain default job never forces a new
        // fetch (sc-10726).
        None => {
            let snapshot = (repo == QWEN_CONTROL_REPO)
                .then(|| {
                    sceneworks_core::control_weights::default_control_revision(
                        QWEN_CONTROL_ENGINE_ID,
                    )
                    .expect("shipped Qwen control revision")
                })
                .and_then(|revision| {
                    crate::model_jobs::huggingface_pinned_snapshot_dir(
                        &settings.data_dir,
                        &repo,
                        revision,
                    )
                });
            format!(
                "{}/{QWEN_CONTROL_FILE}",
                qwen_control_tier_subdir(request, snapshot.as_deref())
            )
        }
    };
    trusted_control_weight_revision(request, QWEN_CONTROL_ENGINE_ID, &repo, &file)?;
    Ok((repo, file))
}

/// Resolve the 2512-Fun-Controlnet-Union weight **file** the `QwenFunControl` provider loads (sc-8350),
/// downloading on first use. Order: an env-pinned file (`SCENEWORKS_CONTROLNET_QWEN`) → a whole-repo HF
/// cache snapshot → download into the app cache.
async fn ensure_qwen_control_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    let (repo, file) = qwen_control_repo_file(request, settings)?;
    if let Ok(p) = std::env::var("SCENEWORKS_CONTROLNET_QWEN") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
    }
    let revision = trusted_control_weight_revision(request, QWEN_CONTROL_ENGINE_ID, &repo, &file)?;
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
        cancel_message: "Qwen strict-pose generation canceled while fetching control weights.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-qwen")
        .join(&file);
    // Pin the exact commit for the default control repo so `main` moving under us can't swap the
    // ControlNet overlay (sc-9879). sc-9870 (merged concurrently) repointed `QWEN_CONTROL_REPO` from
    // `alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union` to the first-party SceneWorks PACKED tier
    // (`SceneWorks/qwen-image-2512-fun-controlnet-union`, per-quant `<tier>/model.safetensors`); the pin
    // below is that repo's verified `main` HEAD. Registered overlays carry their own immutable pin.
    ensure_hf_cached_file(&context, &repo, &revision, &file, &dst).await?;
    Ok(dst)
}

/// Flat telemetry recorded on candle Qwen control assets.
pub(super) fn qwen_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    guidance: f32,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(QWEN_CONTROL_ENGINE.to_owned()),
    );
    raw
}

/// The per-lane half of the candle Qwen 2512-Fun strict-control [`CandleStrictControl`] driver (sc-8304 /
/// sc-8350): the resolved base + 2512-Fun-Union control weight paths + the request numerics. Qwen runs
/// true CFG, so it carries a negative prompt + guidance. Moved onto the blocking thread, loaded once,
/// drives every pose.
pub(super) struct QwenStrictControl {
    qwen_base: PathBuf,
    controlnet: PathBuf,
    prompt: String,
    negative: String,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    control_scale: f32,
    adapters: Vec<gen_core::AdapterSpec>,
}

#[cfg(test)]
pub(super) fn qwen_strict_control_test_fixture(path: PathBuf) -> QwenStrictControl {
    QwenStrictControl {
        qwen_base: path.clone(),
        controlnet: path,
        prompt: "p".to_owned(),
        negative: "n".to_owned(),
        width: 512,
        height: 512,
        steps: 30,
        guidance: 4.0,
        control_scale: 1.0,
        adapters: Vec::new(),
    }
}

impl QwenStrictControl {
    /// Build this lane's bespoke request. Split out of [`CandleStrictControl::generate_one`] so the
    /// preview wiring is reachable without a loaded provider — see
    /// `candle_strict_control_requests_carry_the_live_preview_sink` in `image_jobs::tests`, which
    /// calls this and asserts an emitted frame reaches the sink the driver supplied.
    ///
    /// `preview` is the job's live sink and is **cloned onto the request**, never defaulted (epic 16948,
    /// sc-16962). At inference `5b6d6aa`, Qwen-Image emits per-step latent previews from t2i, edit
    /// and `control_fun` (introduced by sc-16952). Frames are of the developing target only — the
    /// control hint's
    /// VACE latents never reach the sampler's running latent.
    pub(super) fn control_request(
        &self,
        seed: u64,
        cancel: &CancelFlag,
        preview: &gen_core::PreviewSink,
    ) -> QwenFunControlRequest {
        QwenFunControlRequest {
            prompt: self.prompt.clone(),
            negative: self.negative.clone(),
            width: self.width,
            height: self.height,
            steps: self.steps as usize,
            guidance: self.guidance,
            control_scale: self.control_scale,
            seed,
            cancel: cancel.clone(),
            preview: preview.clone(),
        }
    }
}

impl CandleStrictControl for QwenStrictControl {
    type Model = QwenFunControl;

    fn engine_id(&self) -> &'static str {
        QWEN_CONTROL_ENGINE_ID
    }

    fn engine_label(&self) -> &'static str {
        QWEN_CONTROL_ENGINE
    }

    fn stream_tag(&self) -> &'static str {
        "qwen_control"
    }

    fn out_width(&self) -> u32 {
        self.width
    }

    fn out_height(&self) -> u32 {
        self.height
    }

    /// The Qwen-Image-2512 base tier dir + the packed 2512-Fun-Controlnet-Union overlay file, exactly
    /// the two paths [`Self::load`] hands `QwenFunControlPaths` (sc-16069).
    fn conditioning_admission(&self) -> ConditioningAdmission {
        let mut overlays = vec![self.controlnet.as_path()];
        overlays.extend(self.adapters.iter().map(|adapter| adapter.path.as_path()));
        ConditioningAdmission::Floor(ConditioningFootprint::from_paths(
            "Qwen-Image",
            "strict-pose ControlNet branch",
            &self.qwen_base,
            &overlays,
        ))
    }

    fn load(&self) -> WorkerResult<Self::Model> {
        let paths = QwenFunControlPaths {
            qwen_base: self.qwen_base.clone(),
            controlnet: self.controlnet.clone(),
            adapters: self.adapters.clone(),
        };
        QwenFunControl::load(&paths).map_err(|error| {
            WorkerError::Engine(format!("Qwen 2512-Fun strict-control load failed: {error}"))
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
            WorkerError::Engine(format!(
                "Qwen 2512-Fun strict-control generation failed: {error}"
            ))
        })
    }
}

/// Real candle Qwen strict-pose generation: one image per pose, each conditioned on a full DWPose skeleton
/// (`controlMode` unset) or a canny/depth control map. Resolves the Qwen-Image-2512 base + 2512-Fun control weights, then
/// hands a [`QwenStrictControl`] to the shared [`run_candle_strict_control`] driver (validation against
/// `qwen_image_control`'s `supported_kinds`, per-pose preprocessing, scoring). `generate` takes the
/// per-job `CancelFlag` + a `Progress` callback (per-step streaming + mid-denoise cancel). The pose path
/// is byte-preserved.
pub(super) async fn generate_candle_qwen_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let qwen_base = resolve_qwen_control_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("Qwen-Image base weights not found".to_owned())
    })?;
    let controlnet = ensure_qwen_control_weights(api, settings, job, request).await?;

    let steps = qwen_control_steps(request);
    let guidance = qwen_control_guidance(request);
    let control_scale = advanced::f32_clamped(
        &request.advanced,
        "controlScale",
        QWEN_CONTROL_DEFAULT_SCALE,
        0.0..=2.0,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            crate::engines::default_repo_for(&request.model).unwrap_or(QWEN_CONTROL_DEFAULT_REPO)
        })
        .to_owned();

    let pose_count = pose_entries(request).len();
    let raw_settings =
        qwen_control_raw_settings(request, &repo, steps, guidance, control_scale, pose_count);

    let adapters = resolve_adapters(request, settings)?;
    let provider = QwenStrictControl {
        qwen_base,
        controlnet,
        prompt: request.prompt.clone(),
        negative: request.negative_prompt.clone(),
        width: request.width,
        height: request.height,
        steps,
        guidance,
        control_scale,
        adapters,
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
mod qwen_control_tier_tests {
    use super::*;
    use serde_json::json;

    fn request(advanced: serde_json::Value) -> ImageRequest {
        ImageRequest::from_payload(
            json!({ "model": "qwen_image", "advanced": advanced })
                .as_object()
                .unwrap(),
        )
    }

    /// Write a present `<tier>/model.safetensors` overlay so [`qwen_control_tier_subdir`]'s clamp sees
    /// the tier as installed.
    fn seed_tier(root: &Path, tier: &str) {
        let dir = root.join(tier);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(QWEN_CONTROL_FILE), b"x").unwrap();
    }

    /// sc-10726: the plain default (no `advanced.mlxQuantize`) resolves the app-wide Q8 default when the
    /// q8 overlay is on disk, and every explicit pick is honored as-is (an explicit request may fetch its
    /// tier on demand, so it is NOT clamped to what's installed). Mirrors the MLX lane's test.
    #[test]
    fn default_prefers_installed_q8_and_honors_explicit_picks() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_tier(root, "q4");
        seed_tier(root, "q8");
        seed_tier(root, "bf16");

        assert_eq!(
            qwen_control_tier_subdir(&request(json!({})), Some(root)),
            "q8"
        );
        assert_eq!(
            qwen_control_tier_subdir(&request(json!({ "mlxQuantize": 4 })), Some(root)),
            "q4"
        );
        assert_eq!(
            qwen_control_tier_subdir(&request(json!({ "mlxQuantize": 8 })), Some(root)),
            "q8"
        );
        assert_eq!(
            qwen_control_tier_subdir(&request(json!({ "mlxQuantize": 0 })), Some(root)),
            "bf16"
        );
        assert_eq!(
            qwen_control_tier_subdir(&request(json!({ "mlxQuantize": "0" })), Some(root)),
            "bf16"
        );
    }

    /// sc-10726 acceptance #3: on a legacy/partial install carrying ONLY the q4 overlay, the plain
    /// default must resolve q4 — NOT the app-wide q8 default — so it never forces a NEW q8 on-demand
    /// fetch (via [`ensure_qwen_control_weights`]). An EXPLICIT q8 pick is still returned verbatim.
    #[test]
    fn default_clamps_to_only_installed_q4() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_tier(root, "q4");

        assert_eq!(
            qwen_control_tier_subdir(&request(json!({})), Some(root)),
            "q4"
        );
        seed_tier(root, "bf16");
        assert_eq!(
            qwen_control_tier_subdir(&request(json!({})), Some(root)),
            "bf16"
        );
        assert_eq!(
            qwen_control_tier_subdir(&request(json!({ "mlxQuantize": 8 })), Some(root)),
            "q8"
        );
    }

    /// With NO snapshot cached yet (or an empty snapshot), the default falls back to `q8` — the
    /// fresh-install co-requisite the base download brings alongside the base tier.
    #[test]
    fn default_falls_back_to_q8_when_nothing_installed() {
        assert_eq!(qwen_control_tier_subdir(&request(json!({})), None), "q8");
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(
            qwen_control_tier_subdir(&request(json!({})), Some(empty.path())),
            "q8"
        );
    }
}
