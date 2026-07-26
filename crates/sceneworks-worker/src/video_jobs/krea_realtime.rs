#[cfg(target_os = "macos")]
use super::ltx::extract_clip_frames;
#[allow(unused_imports)]
use super::prelude::*;
#[cfg(target_os = "macos")]
use super::wan::{generate_video_using, local_mlx_dir, VideoGenInput};

// ---------------------------------------------------------------------------
// Real MLX Krea Realtime 14B generation (macOS, via mlx-gen-krea-realtime, epic 8431 / sc-8443 S10):
// an autoregressive, self-forcing, real-time video denoiser whose SHIPPED checkpoint is Wan 2.1 T2V
// 14B weight-for-weight (the S1 audit fact the engine crate is built on). What makes it distinct is
// its inference REGIME — a short per-frame-block few-step Self-Forcing schedule + a rolling causal KV
// cache — not its weights, so the DiT / z16 VAE / UMT5 text encoder / RoPE / schedulers are all reused
// from stock Wan inside the engine.
//
// Surface (descriptor `pipeline::descriptor`, sc-8440 S7): CFG-OFF video (no negative-prompt / guidance
// / true-cfg axis — the AR few-step denoise runs a single batch-1 forward per step), sampler
// `self_forcing`, `supported_quants = []` (dense bf16 load — no load-time quant tier wired yet),
// `supports_lora = false`, `mac_only = true`. It advertises two conditioning shapes the worker maps its
// own reference inputs onto here:
//   * i2v — a `Reference` still VAE-encoded to WARM the AR KV cache from clean context. The engine's
//     `run` reads only the reference IMAGE (`Conditioning::Reference { image, .. }`), never its
//     `strength` — the cache-warm has no strength analog — so **i2v `strength` is a NO-OP**, faithful
//     to the sibling `bernini`/reference paths which also emit `strength: None`. Documented + pinned
//     below so a future edit can't silently start honoring it.
//   * v2v — a `VideoClip` source that drives a strength-CONTROLLED AR init (`FewStepSchedule::for_
//     strength`): a lower strength preserves more of the source. So **v2v DOES honor `strength`**
//     (`videoConditioningStrength`, default 1.0 — the same knob + default the Wan/LTX v2v paths use).
//
// This is the non-gated worker DISPATCH wiring (routing + VideoRequest→GenerationRequest mapping +
// heartbeat-funnel routing). The real-weight watchable-clip e2e needs the real ~28 GB DiT and is the
// GATED S13 validation; the MLX rehost of the turnkey snapshot is the gated S2 remainder (sc-8435).
// LoRA passthrough is the S15 seam (see `generate_krea_realtime`). Mac-only: the off-Mac (candle) lane
// has NO Krea Realtime engine, so a non-mac job fails loud in `run_video_generate_job` (mirroring the
// `wan_2_2_vace_fun_14b` mac-only guard) rather than silently routing to a different backend / the stub.
// ---------------------------------------------------------------------------

/// The SceneWorks/model catalog id for Krea Realtime 14B — identical to the engine registry id
/// (`mlx_gen_krea_realtime::MODEL_ID`), so the SceneWorks id maps to the engine 1:1 (no `_distilled`-
/// style split). Named once here so the route, the raw-settings, and the mac-only guard agree.
#[cfg(target_os = "macos")]
pub(super) const KREA_REALTIME_MODEL_ID: &str = "krea_realtime_14b";

/// Adapter id recorded on a real MLX Krea Realtime asset (mirrors the `mlx_*` convention).
#[cfg(target_os = "macos")]
pub(super) const KREA_REALTIME_ADAPTER: &str = "mlx_krea_realtime";

/// SceneWorks Krea Realtime model id → mlx-gen registry id, or `None` if `model` is not Krea Realtime.
/// The id IS the engine id (`KREA_REALTIME_MODEL_ID`); no earlier predicate in `resolve_video_route`
/// can match it (it is not a Wan/LTX/SVD/Bernini/SCAIL-2/Mochi id), so appending the krea arm leaves
/// every pre-existing route byte-identical.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_engine_id(model: &str) -> Option<&'static str> {
    (model == KREA_REALTIME_MODEL_ID).then_some(KREA_REALTIME_MODEL_ID)
}

/// Whether the linked Krea Realtime engine can serve this request now (resolvable weights). `mac_only`
/// is implicit — this fn is macOS-only, and a non-mac job never reaches the route (it is rejected
/// loudly in `run_video_generate_job`). Routed by weight availability like the other MLX engines: an
/// unresolved snapshot falls to `VideoRoute::Stub`, whose fail-loud gate (`ensure_video_engine_
/// weights`) then surfaces the resolver's precise error instead of a procedural fake clip.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_available(_request: &VideoRequest, settings: &Settings) -> bool {
    resolve_krea_realtime_model_dir(settings).is_ok()
}

/// The turnkey SceneWorks Krea Realtime MLX repo (the converted transformer-only DiT snapshot +
/// the stock Wan `t5_encoder`/`vae`/`tokenizer`). **Provisional**: the MLX rehost to the SceneWorks HF
/// org is the gated S2 remainder (sc-8435), so the repo may not be published yet — `resolve_krea_
/// realtime_model_dir` only ever LOOKS UP an already-downloaded snapshot in the local cache (no
/// network fetch here; there is no on-demand quant-tier fetch because the engine advertises
/// `supported_quants = []` and loads dense bf16), so an absent repo simply means the env override /
/// app-managed dir are the resolvable sources until the rehost + download lands.
#[cfg(target_os = "macos")]
pub(super) const KREA_REALTIME_REPO: &str = "SceneWorks/krea-realtime-14b-mlx";

/// Resolve the Krea Realtime MLX snapshot dir: env override (`SCENEWORKS_MLX_KREA_REALTIME_DIR`) →
/// app-managed `<data>/models/mlx/krea_realtime` → the turnkey `SceneWorks/krea-realtime-14b-mlx`
/// snapshot if already downloaded (mirrors `resolve_bernini_model_dir`). Errors clearly if none is
/// present (no stub fallback).
#[cfg(target_os = "macos")]
pub(super) fn resolve_krea_realtime_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Some(dir) = local_mlx_dir(
        settings,
        "SCENEWORKS_MLX_KREA_REALTIME_DIR",
        "krea_realtime",
    ) {
        return Ok(dir);
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, KREA_REALTIME_REPO) {
        return Ok(dir);
    }
    Err(WorkerError::InvalidPayload(format!(
        "krea_realtime_14b: no MLX weights found. Download the turnkey {KREA_REALTIME_REPO} snapshot \
         via the Model Manager, set $SCENEWORKS_MLX_KREA_REALTIME_DIR, or place a converted snapshot \
         at {}.",
        settings
            .data_dir
            .join("models")
            .join("mlx")
            .join("krea_realtime")
            .display(),
    )))
}

/// The Krea Realtime video task a request resolves to, from its supplied media: a source clip → `v2v`,
/// else a reference still → `i2v`, else `t2v`. Routed on the MEDIA (matching how the engine's own `run`
/// routes on the conditioning it is handed), not the SceneWorks `mode` string, so the mapping is faithful
/// regardless of which mode label the caller used. Recorded on the asset for observability.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_video_task(request: &VideoRequest) -> &'static str {
    if request.source_clip_asset_id.is_some() {
        "v2v"
    } else if request.source_asset_id.is_some() {
        "i2v"
    } else {
        "t2v"
    }
}

/// Raw-settings recorded on a real MLX Krea Realtime asset (mirrors `bernini_raw_settings`).
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    // The engine task the supplied media resolved to (lineage / observability).
    raw.insert(
        "kreaRealtimeTask".to_owned(),
        Value::String(krea_realtime_video_task(request).to_owned()),
    );
    Value::Object(raw)
}

/// Build the Krea Realtime conditioning from the (already loaded) reference media — the PURE i2v/v2v
/// shape decision the engine's `run` mirrors. A source `clip` → one `VideoClip` whose `strength` is
/// HONORED (v2v drives a strength-controlled AR init). Else a reference `still` → one `Reference` with
/// `strength: None` — i2v warms the AR KV cache from the still and the engine reads only the image, so
/// **the i2v strength is a NO-OP** (faithful to the sibling `bernini`/reference paths). Else empty
/// (t2v). Clip takes precedence over a still, exactly as the engine's `run` prefers `VideoClip`.
///
/// Kept pure (loaded media in, conditioning out) so the invariant this story exists to wire — i2v drops
/// strength, v2v carries it — is unit-testable without a GPU, weights, or on-disk assets.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_conditioning(
    still: Option<Image>,
    clip: Option<Vec<Image>>,
    v2v_strength: f32,
) -> Vec<Conditioning> {
    if let Some(frames) = clip {
        // v2v: the source clip drives the strength-controlled AR init. `frame_idx` is inert for Krea
        // (its `run` reads only `frames` + `strength`); carried at 0 for the shared `VideoClip` contract.
        return vec![Conditioning::VideoClip {
            frames,
            frame_idx: 0,
            strength: v2v_strength,
        }];
    }
    if let Some(image) = still {
        // i2v: the still warms the AR KV cache. `strength` is a NO-OP for the cache-warm (the engine
        // reads only the image), so it is dropped to `None` — do NOT wire a strength here.
        return vec![Conditioning::Reference {
            image,
            strength: None,
        }];
    }
    Vec::new()
}

/// Resolve a Krea Realtime request's supplied media into the engine conditioning: a source clip
/// (`sourceClipAssetId`) → v2v `VideoClip`, else a reference still (`sourceAssetId`) → i2v `Reference`,
/// else t2v (empty). The clip decodes to the output frame count via the shared `extract_clip_frames`;
/// the still loads via the shared `load_reference_image`. The shape + strength decision itself lives in
/// the pure [`krea_realtime_conditioning`].
#[cfg(target_os = "macos")]
pub(super) async fn resolve_krea_realtime_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    // v2v takes precedence over i2v (matches the engine's `run`): a source clip → v2v.
    if let Some(clip_id) = request.source_clip_asset_id.as_deref() {
        let frames = extract_clip_frames(
            api,
            settings,
            job,
            &request.project_id,
            project_path,
            clip_id,
            request.width,
            request.height,
            wan_frame_count(request.raw_frame_count()),
        )
        .await?;
        // v2v HONORS strength (`videoConditioningStrength`, default 1.0 — the Wan/LTX v2v convention).
        let strength = advanced::f32(&request.advanced, "videoConditioningStrength", 1.0);
        return Ok(krea_realtime_conditioning(None, Some(frames), strength));
    }
    // i2v: a reference still (the I2V conditioning source, like SVD's `sourceAssetId`).
    if let Some(ref_id) = request.source_asset_id.as_deref() {
        let still = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            ref_id,
            project_path,
        )?;
        // i2v strength is a NO-OP (the pure helper drops it); the `1.0` here is never read.
        return Ok(krea_realtime_conditioning(Some(still), None, 1.0));
    }
    // t2v: no conditioning.
    Ok(Vec::new())
}

/// Real MLX Krea Realtime generation (epic 8431 / sc-8443 S10): build the `VideoGenInput` and run the
/// shared `generate_video` heartbeat funnel. The supplied media resolves to the engine conditioning
/// ([`resolve_krea_realtime_conditioning`]) — empty for t2v, a `Reference` still for i2v (strength
/// no-op), a `VideoClip` source for v2v (strength honored). CFG off: no negative prompt / guidance
/// (the engine advertises none), `steps` from the advanced `steps` knob (else the engine default),
/// dense bf16 load (`supported_quants = []`, so `quant = None`). Frame count uses the Wan 1-mod-4
/// stride coercion — the DiT + z16 VAE ARE stock Wan 2.1, so the Wan stride is exactly right.
///
/// Runs through the `generate_video` → `generate_video_using` funnel so a long/multi-minute AR job
/// drives `heartbeat(...)` and is NOT marked `interrupted` by the ~90 s API stale-sweep, and so the
/// funnel's per-step cancel poll trips the engine's `CancelFlag` promptly (the S8 half of this story's
/// DoD).
///
/// **LoRA passthrough is the S15 seam.** The engine advertises `supports_lora = false`, so no adapter
/// is wired (`adapters` left empty): forwarding a LoRA to an engine whose load path ignores it would
/// violate capability honesty. A LoRA-bearing request is NOT rejected — it runs the base DiT — so the
/// worker never breaks on one; the full dense-LoRA integration (via the shared `resolve_dense_adapters`)
/// lands in sc-8443's S15.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_krea_realtime(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    generate_krea_realtime_using(
        api,
        settings,
        job,
        request,
        project_path,
        engine_id,
        backend,
        crate::inference_runtime::load,
    )
    .await
}

/// [`generate_krea_realtime`] with the engine loader supplied by the caller (mirrors
/// `generate_mochi_using`, sc-12318). With the loader threaded in, a test can drive this arm against a
/// stub `Generator` and assert on the `GenerationRequest` that actually reached the engine (prompt /
/// steps / seed / frame count / fps and the resolved conditioning) without weights or a GPU.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_krea_realtime_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<DecodedVideo> {
    let conditioning =
        resolve_krea_realtime_conditioning(api, settings, job, request, project_path).await?;
    let model_dir = resolve_krea_realtime_model_dir(settings)?;
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        // Dense bf16 load: the engine advertises `supported_quants = []`, so never request a load-time
        // quant (a Q4/Q8 request would be capability-dishonest until a tier is wired).
        quant: None,
        conditioning,
        prompt: request.prompt.clone(),
        // CFG off: the engine advertises no negative-prompt / guidance axis, so leave both unset.
        negative_prompt: None,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps: super::wan::advanced_opt_u32(request, "steps"),
        seed: resolve_video_seed(request) as u64,
        // No `video_mode`: the engine routes purely on the supplied conditioning, not a task string.
        // No adapters: `supports_lora = false` (LoRA is the S15 seam — see the doc comment).
        ..VideoGenInput::default()
    };
    generate_video_using(
        api,
        settings,
        job,
        backend,
        &request.advanced,
        input,
        load_generator,
    )
    .await
}
