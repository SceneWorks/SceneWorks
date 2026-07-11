//! Backend-gap classification, unavailability/unsupported errors, and the mac/candle
//! "can the Rust flow run this?" oracles. Moved out of `jobs_store.rs` (sc-8816) with no
//! behavior change.

use serde_json::{Map, Value};

use crate::contracts::{JobSnapshot, JobType};
use crate::jobs_store::routing::candle::{
    image_job_is_candle_eligible, image_request_candle_pose_reject,
    training_job_is_candle_eligible, upscale_job_is_candle_eligible, video_job_is_candle_eligible,
    video_upscale_job_is_candle_eligible,
};
use crate::jobs_store::routing::catalog::{
    CANDLE_ROUTED_MODELS, CANDLE_VIDEO_ROUTED_MODELS, MLX_ROUTED_MODELS, VIDEO_MLX_ROUTED_MODELS,
};
use crate::jobs_store::routing::mlx::{
    caption_job_is_mlx_eligible, job_is_mlx_eligible, training_job_is_mlx_eligible,
    understanding_job_is_mlx_eligible, upscale_job_is_mlx_eligible, video_job_is_mlx_eligible,
    video_mode_is_mlx_eligible, video_upscale_job_is_mlx_eligible,
};

/// True when *any* MLX-routing predicate (image/detail, video, or training) claims this
/// job — the union an `mlx` worker would want. Used both to classify a claim for routing
/// observability (sc-3449) and to identify the jobs the macOS grace sweep must fail when
/// no `mlx` worker is alive (sc-3483).
pub(crate) fn job_is_any_mlx_eligible(job: &JobSnapshot) -> bool {
    job_is_mlx_eligible(job)
        || video_job_is_mlx_eligible(job)
        || video_upscale_job_is_mlx_eligible(job)
        || training_job_is_mlx_eligible(job)
        || caption_job_is_mlx_eligible(job)
        || understanding_job_is_mlx_eligible(job)
}

/// True when *any* candle-routing predicate (image, video, image/video upscale, caption,
/// understanding, training) claims this job — the union a candle (Windows/CUDA) worker would want,
/// the off-Mac twin of [`job_is_any_mlx_eligible`] (sc-5502, epic 5483). Used to identify the jobs
/// the Phase-7 candle grace sweep must fail when no live candle worker is alive
/// ([`JobsStore::fail_stranded_candle_jobs`]). Deliberately excludes the unsupported-pose shapes
/// the candle worker only *owns to reject* ([`image_job_candle_pose_reject`]) — those are gaps,
/// not served, so they must strand/enforce-fail rather than wait for a candle worker. Training
/// (sc-7817) is included only for the candle-trainable kernels; the rest stay on the torch worker.
pub(crate) fn job_is_any_candle_eligible(job: &JobSnapshot) -> bool {
    image_job_is_candle_eligible(job)
        || video_job_is_candle_eligible(job)
        || upscale_job_is_candle_eligible(job)
        || video_upscale_job_is_candle_eligible(job)
        || caption_job_is_mlx_eligible(job)
        || understanding_job_is_mlx_eligible(job)
        || training_job_is_candle_eligible(job)
}

/// Actionable terminal error for an MLX-eligible job stranded on macOS with no live `mlx`
/// worker (sc-3483). Names the model + job type so the job card and the System → Logs
/// surface point at the real gap, never a generic failure. Prefixed `mlx_unavailable:` so
/// the cause is greppable in logs and distinguishable from `mlx_unsupported` (sc-3484).
pub(crate) fn mlx_unavailable_error(job: &JobSnapshot, grace_seconds: u64) -> String {
    let model = job
        .payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("(unknown)");
    format!(
        "mlx_unavailable: the MLX GPU worker is required on macOS but no live worker \
         claimed this job within {grace_seconds}s (model={model}, type={job_type}). There \
         is no fallback worker on Mac — check System → Logs and confirm the MLX \
         worker is running.",
        job_type = job.job_type.as_str()
    )
}

/// Actionable terminal error for a candle-eligible job stranded off-Mac with no live candle
/// (CUDA) worker (sc-5502, epic 5483) — the Windows/Linux twin of [`mlx_unavailable_error`].
/// Names the model + job type and is prefixed `candle_unavailable:` so the cause is greppable in
/// logs and distinguishable from `candle_unsupported`.
pub(crate) fn candle_unavailable_error(job: &JobSnapshot, grace_seconds: u64) -> String {
    let model = job
        .payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("(unknown)");
    format!(
        "candle_unavailable: the candle GPU worker is required off-Mac but no live worker \
         claimed this job within {grace_seconds}s (model={model}, type={job_type}). There is no \
         fallback worker on this deployment — check System → Logs and confirm the \
         candle worker is running.",
        job_type = job.job_type.as_str()
    )
}

/// Human, actionable terminal error attributing a worker's abnormal death to its
/// terminating signal or non-zero exit code (sc-4881 signals; sc-6320 non-signal
/// exits). Signal 9 (an uncatchable SIGKILL — almost always an OS memory-pressure
/// OOM kill) carries a remediation hint tailored to the dead job's kind (sc-5567):
/// training points at gradient checkpointing (the sc-4874 first-step OOM),
/// image/video generation at the knobs that actually shrink the working set (batch
/// count, resolution, frame count). Other uncatchable deaths (SIGABRT GPU/Metal
/// abort, SIGSEGV) name themselves. A non-signal non-zero exit is a self-terminated
/// process — exit code 101 is the Rust panic code and self-names; other codes report
/// the raw code — so the job card and System → Logs show a real cause instead of a
/// frozen progress bar. A signal takes precedence when both are somehow present.
/// `job_type` is the failed job's kind when one was active (`None` when the worker
/// died idle).
pub(crate) fn termination_failure_error(
    signal: Option<i32>,
    exit_code: Option<i32>,
    job_type: Option<&JobType>,
) -> String {
    if let Some(signal) = signal {
        let hint = match signal {
            9 => oom_remediation_hint(job_type),
            6 => ", likely a GPU/Metal command-buffer abort or assertion",
            11 => " (segmentation fault)",
            _ => "",
        };
        return match signal_name(signal) {
            Some(name) => format!("Worker terminated by signal {signal} ({name}){hint}."),
            None => format!("Worker terminated by signal {signal}{hint}."),
        };
    }
    match exit_code {
        // 101 is the Rust panic exit code — a panic that unwound to a process exit
        // rather than aborting on a signal. Name it so the cause is unmistakable.
        Some(101) => "Worker process panicked and exited (code 101). \
                      Check System → Logs for the panic message."
            .to_owned(),
        Some(code) => format!(
            "Worker process exited unexpectedly (code {code}). Check System → Logs for the cause."
        ),
        // Defensive: the supervisor only calls this for an abnormal exit (a signal
        // or a non-zero code), so neither-present shouldn't reach here — report it
        // generically rather than fabricate a signal or code.
        None => {
            "Worker process terminated unexpectedly. Check System → Logs for the cause.".to_owned()
        }
    }
}

/// Signal-9 (SIGKILL/OOM) remediation hint keyed to the dead job's kind so the guidance
/// is actionable rather than training-centric (sc-5567). The `_` arm covers the long tail
/// of non-generation job types (and is required anyway — `JobType` is `#[non_exhaustive]`).
pub(crate) fn oom_remediation_hint(job_type: Option<&JobType>) -> &'static str {
    match job_type {
        // LoRA / ControlNet-branch training: the sc-4874 first-training-step OOM — gradient
        // checkpointing is the real lever; resolution is secondary.
        Some(JobType::LoraTrain | JobType::ControlTraining) => {
            ", likely out-of-memory during the first training step \
             — enable Gradient Checkpointing or reduce resolution"
        }
        // Video generation/edit: working set scales with resolution AND frame count.
        Some(
            JobType::VideoGenerate
            | JobType::VideoExtend
            | JobType::VideoBridge
            | JobType::VideoUpscale
            | JobType::PersonReplace,
        ) => ", likely out-of-memory — reduce the resolution, frame count, or batch count",
        // Image generation/edit: a multi-image batch stacks per-image working set — count
        // is the first knob, then resolution (sc-5567).
        Some(
            JobType::ImageGenerate
            | JobType::ImageEdit
            | JobType::ImageUpscale
            | JobType::ImageDetail
            | JobType::ImageSegment
            | JobType::ImageVqa
            | JobType::ImageInterleave,
        ) => ", likely out-of-memory — reduce the image count or resolution",
        _ => ", likely out-of-memory — reduce the resolution or batch count",
    }
}

/// Conventional name for the common terminating signals we attribute (sc-4881).
pub(crate) fn signal_name(signal: i32) -> Option<&'static str> {
    Some(match signal {
        1 => "SIGHUP",
        2 => "SIGINT",
        6 => "SIGABRT",
        9 => "SIGKILL",
        11 => "SIGSEGV",
        15 => "SIGTERM",
        _ => return None,
    })
}

/// Why the Rust/MLX flow can't run a job on macOS (epic 3482 / sc-3484) — the inverse of the
/// `*_mlx_eligible` predicates, extended across every job type. Feature-precise so the
/// `mlx_unsupported` Logs event + the gap inventory name the exact surface to port or drop.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UnsupportedReason {
    /// Model id involved, when the gap is model-specific (e.g. "kolors", "qwen_image").
    pub model: Option<String>,
    /// The specific capability that isn't in the Rust/MLX flow (e.g. "strict-pose ControlNet",
    /// "third-party LyCORIS LoRA", "image_upscale (Real-ESRGAN)").
    pub feature: String,
    /// Actionable human-readable explanation.
    pub detail: String,
    /// Closing story/epic ("epic 3401", "sc-3489"), `"drop-candidate"`, or `None` when not yet
    /// triaged — the roadmap pointer. "where known" per the story.
    pub suggested_epic: Option<String>,
}

impl UnsupportedReason {
    pub(crate) fn new(
        model: Option<&str>,
        feature: &str,
        detail: &str,
        suggested_epic: Option<&str>,
    ) -> Self {
        Self {
            model: model.map(str::to_owned),
            feature: feature.to_owned(),
            detail: detail.to_owned(),
            suggested_epic: suggested_epic.map(str::to_owned),
        }
    }

    /// Terminal job error for an enforced `mlx_unsupported` failure (sc-3484): greppable
    /// prefix, names feature + model + roadmap pointer.
    pub fn error_message(&self) -> String {
        let model = self
            .model
            .as_deref()
            .map(|m| format!(" ({m})"))
            .unwrap_or_default();
        let pointer = self
            .suggested_epic
            .as_deref()
            .map(|epic| format!(" [{epic}]"))
            .unwrap_or_default();
        format!(
            "mlx_unsupported: {feature}{model} is not in the MLX flow on macOS — {detail}{pointer}",
            feature = self.feature,
            detail = self.detail,
        )
    }

    /// Terminal job error for an enforced `candle_unsupported` failure (sc-5502, epic 5483) — the
    /// off-Mac twin of [`Self::error_message`]: same feature/model/detail/pointer, a candle-flavored
    /// prefix and flow name so the two backends' gap events are independently greppable.
    pub fn candle_error_message(&self) -> String {
        let model = self
            .model
            .as_deref()
            .map(|m| format!(" ({m})"))
            .unwrap_or_default();
        let pointer = self
            .suggested_epic
            .as_deref()
            .map(|epic| format!(" [{epic}]"))
            .unwrap_or_default();
        format!(
            "candle_unsupported: {feature}{model} is not in the candle/CUDA flow off-Mac — {detail}{pointer}",
            feature = self.feature,
            detail = self.detail,
        )
    }
}

/// macOS "can the Rust/MLX flow run this?" oracle (sc-3484). `Ok(())` = the in-process mlx
/// worker — or an MLX-agnostic in-process path (downloads, ffmpeg, prompt refine) — runs it
/// with no Python torch dependency. `Err` names the exact Python-torch gap. This is the epic's
/// *forcing function*: under mlx-required **enforce** mode an `Err` job fails terminal with
/// `mlx_unsupported`, and the set of `Err`s IS the port-or-drop roadmap. Consistent with
/// routing by construction — anything `job_is_any_mlx_eligible` accepts is `Ok`.
pub fn mac_rust_supported(job: &JobSnapshot) -> Result<(), UnsupportedReason> {
    if job_is_any_mlx_eligible(job) {
        return Ok(());
    }
    let model = job.payload.get("model").and_then(Value::as_str);
    match job.job_type {
        // In-process macOS job types with no Python torch dependency: MLX-agnostic metadata/utility
        // work + ffmpeg, plus prompt refine — now served by the native MLX `prompt_refine` TextLlm
        // provider (sc-5552, the mlx twin of the candle sc-5525 cutover), so the worker advertises +
        // claims it and this `Ok` is backed by a real capability (no longer the pre-sc-5552 strand).
        JobType::Placeholder
        | JobType::ModelDownload
        | JobType::ModelImport
        | JobType::LoraImport
        | JobType::LoraDownload
        | JobType::FrameExtract
        | JobType::TimelineExport
        | JobType::PromptRefine
        // sc-6535: dataset_analysis is a native Rust/MLX job (the CLIP image embedder), not a
        // Python-torch gap — so it's `Ok` here. Its real capability is gated by the worker's
        // advertisement once `mlx-gen-clip` is linked; until then it queues, never enforce-fails.
        | JobType::DatasetAnalysis
        // sc-6539: dataset_upscale runs the Real-ESRGAN ONNX engine natively in the Rust worker
        // (the same engine as image_upscale) — not a Python-torch gap. Its real availability is the
        // worker's capability advertisement, so it queues rather than enforce-fails here.
        | JobType::DatasetUpscale
        // sc-6538: dataset_face_analysis runs the native SCRFD+ArcFace stack (mlx-gen-face) in the Rust
        // worker — not a Python-torch gap. Gated by the worker's capability advertisement, so it queues
        // rather than enforce-fails here.
        | JobType::DatasetFaceAnalysis
        // sc-4415: face_likeness_compare runs the same native SCRFD+ArcFace stack to score two existing
        // assets on demand — a native Rust/MLX job, not a Python-torch gap. Gated by the worker's
        // capability advertisement, so it queues rather than enforce-fails here.
        | JobType::FaceLikenessCompare => Ok(()),

        // Forward-compat: an unrecognized job type isn't a known Python-torch gap, so don't
        // enforce-fail it (it would otherwise break a newer job type this build doesn't model).
        JobType::Unknown(_) => Ok(()),

        JobType::ImageGenerate | JobType::ImageEdit => Err(classify_image_gap(&job.payload)),

        JobType::ImageDetail => Err(UnsupportedReason::new(
            model,
            "non-SDXL tile-detail refine",
            "image_detail is ported to MLX only for the SDXL/RealVisXL backbones (sc-3060); other models / third-party LyCORIS are not available in the native flow on Mac.",
            Some("epic 3041"),
        )),

        // SenseNova-U1 VQA + Document-Studio interleave are ported to the Rust MLX worker
        // (sc-3905, via the concrete `T2iModel` — the `Generator` contract can't express
        // text / text+image output); eligible jobs early-return `Ok` above. This arm is
        // reached only for an understanding job on a model with no in-process path.
        JobType::ImageVqa | JobType::ImageInterleave => Err(UnsupportedReason::new(
            model,
            "image understanding / interleave on this model",
            "image VQA / interleaved generation runs on MLX for the SenseNova-U1 model (sensenova_u1_8b[_fast]); other models have no in-process understanding path and are not available in the native flow on Mac.",
            Some("epic 3180"),
        )),

        JobType::VideoGenerate => Err(classify_video_gap(&job.payload)),

        // Reached only for ineligible extend/bridge jobs (the eligible LTX IC-LoRA path + the Wan
        // TI2V-5B boundary-keyframe path early-return `Ok` via `job_is_any_mlx_eligible`). The
        // remaining gap is an engine with no in-context / keyframe path: the 14B Wan MoE engines
        // and any non-MLX video model (sc-3522 / sc-3357).
        JobType::VideoExtend | JobType::VideoBridge => Err(UnsupportedReason::new(
            model,
            "extend / bridge on this engine",
            "extend_clip / video_bridge run on MLX on the LTX IC-LoRA path (ltx_2_3 / ltx_2_3_eros) \
             and Wan TI2V-5B (wan_2_2, single-frame boundary keyframe conditioning); other engines \
             (the 14B Wan MoE) have no keyframe path, so they are not available in the native flow on Mac.",
            Some("epic 3040"),
        )),

        // replace_person → native Wan-VACE (the replace-capable models) or native SCAIL-2
        // (scail2_14b, sc-5452) is MLX-eligible (handled by the early `job_is_any_mlx_eligible` Ok
        // above). This arm is only reached for a replace_person job on a model with no MLX video
        // engine — that stays torch.
        JobType::PersonReplace => Err(UnsupportedReason::new(
            model,
            "replace_person",
            "person replacement runs on native Wan-VACE (the replace-capable MLX video models) or native SCAIL-2 (scail2_14b); this model has no MLX video engine, so it is not available in the native flow on Mac.",
            Some("epic 3040"),
        )),

        // Person detection + tracking are now ported to the Rust worker (epic 3482,
        // sc-3488): native-MLX YOLO11 detection (sc-3633), SORT/ByteTrack track assembly
        // (sc-3634), and SAM2 per-frame segmentation (sc-3709) all run in-process on the
        // macOS MLX worker, so the Replace-Person detect → track → mask flow is
        // Python-free. (replace_person end-to-end still needs the video-gen/inpaint half,
        // a tracked torch gap on `PersonReplace` below — epic 3040.)
        JobType::PersonDetect | JobType::PersonTrack => Ok(()),

        // DWPose pose detection is now ported to the Rust worker (sc-3487): RTMW
        // whole-body via `ort`/CoreML on the macOS MLX worker, so the Pose Library
        // "create from photo" flow + InstantID pose conditioning run Python-free.
        JobType::PoseDetect => Ok(()),

        // SCRFD 5-point landmark extraction is native-MLX on the Rust worker (sc-4433,
        // epic 4422): the same SCRFD detector the InstantID face stack already runs
        // in-process, so the Key Point Library "extract kps from this image" flow is
        // Python-free on Mac.
        JobType::KpsExtract => Ok(()),

        // Smart-select segmentation (epic 6087, sc-6105): native-MLX SAM3 box-prompt
        // segmentation runs in-process on the macOS Rust worker (the box-PVS path of the
        // sc-4926 SAM3 stack — `segment_jobs::run_image_segment_job`), so the Image Editor
        // smart-select tool is Python-free on Mac. Mac-only by construction: the capability
        // is advertised only by `mlx_gpu`, so no torch/candle worker ever claims it.
        JobType::ImageSegment => Ok(()),

        // Real-ESRGAN image upscaling is ported to the Rust worker (sc-3489) and SeedVR2 (the
        // native-MLX one-step diffusion upscaler, epic 4811 / sc-4815) runs in-process via
        // `mlx-gen-seedvr2`, so the upscale tool runs Python-free. The AuraSR engine (`aura-sr`,
        // a 617M-param torch-only GigaGAN) was DROPPED on Mac after the sc-3668 port-or-drop
        // spike (no viable Rust path; only a marginal, ~35-50x-slower quality difference vs
        // Real-ESRGAN x4) and is now dropped as an offered engine off-Mac too (sc-5499 — the
        // Python torch backend that served it is retired in Phase 7). The UI hides the AuraSR
        // engine option on every platform, so this Err is a defensive submit-time guard.
        JobType::ImageUpscale => {
            if upscale_job_is_mlx_eligible(job) {
                Ok(())
            } else {
                Err(UnsupportedReason::new(
                    model,
                    "image_upscale (AuraSR)",
                    "the upscaler runs Real-ESRGAN (+ SeedVR2); the AuraSR engine is dropped as an offered engine (sc-3668 / sc-5499).",
                    Some("sc-5499"),
                ))
            }
        }

        // Video upscaling is net-new on Mac (epic 4811 / sc-4816): the native-MLX SeedVR2
        // engine is the only path (there is no torch video upscaler), so a SeedVR2 job is
        // supported and anything else has no in-process engine. Eligible jobs early-return
        // `Ok` above via `job_is_any_mlx_eligible`; this arm is the defensive guard.
        JobType::VideoUpscale => {
            if video_upscale_job_is_mlx_eligible(job) {
                Ok(())
            } else {
                Err(UnsupportedReason::new(
                    model,
                    "video_upscale (non-SeedVR2 engine)",
                    "video upscaling runs on the native-MLX SeedVR2 engine (seedvr2); no other engine is available.",
                    Some("epic 4811"),
                ))
            }
        }

        JobType::ModelConvert => classify_convert_gap(&job.payload),

        JobType::LoraTrain => Err(classify_training_gap(&job.payload)),

        // ControlNet Training Studio (epic 10159): the control-branch trainer is candle-only — there is
        // no MLX control trainer yet (that is B5/sc-10177) — so a `control_training` job stranded on Mac
        // with no candle worker is a real gap, not a torch fallback.
        JobType::ControlTraining => Err(UnsupportedReason::new(
            None,
            "ControlNet branch training",
            "ControlNet training runs on the candle/CUDA control trainer (krea_control); there is no native MLX control trainer yet, so it is not available in the flow on Mac.",
            Some("epic 10159"),
        )),

        JobType::TrainingCaption => Err(UnsupportedReason::new(
            None,
            "dataset captioning",
            "this dataset captioning job is not in the MLX JoyCaption flow.",
            Some("sc-3556"),
        )),
    }
}

/// Off-Mac "can the candle/CUDA flow run this?" oracle (sc-5502, epic 5483) — the Windows/Linux
/// twin of [`mac_rust_supported`]. `Ok(())` = the candle worker (or an MLX-agnostic in-process
/// Rust path: downloads, ffmpeg, prompt refine — sc-5525) runs it with zero torch; `Err` names the
/// exact torch gap. Under `candle_required` **enforce** an `Err` job fails terminal with
/// `candle_unsupported`, so the set of `Err`s is the off-Mac port-or-drop roadmap. Consistent with
/// routing by construction — anything [`job_is_any_candle_eligible`] accepts is `Ok`.
///
/// **Scope (this slice):** biased toward `Ok` exactly like the MLX oracle ("never over-gate a
/// valid combination"). It enforce-fails only the **generation** gaps that have crisp candle
/// eligibility predicates — the shapes that would otherwise silently mis-serve as an unconditioned
/// torch T2I (the sc-5968 concern, generalized from poses to the whole image/video surface). The
/// CV-aux / segment / detail / training / convert / infra job types route by capability and their
/// candle parity is still landing (Phase 5, epic 5482; the training cutover); they stay `Ok` here
/// so the enforce sweep never kills a job the co-resident torch worker still serves. Each converts
/// to an `Err` arm as its phase epic closes and torch is retired for that surface (sc-5503).
pub fn candle_supported(job: &JobSnapshot) -> Result<(), UnsupportedReason> {
    if job_is_any_candle_eligible(job) {
        return Ok(());
    }
    let model = job.payload.get("model").and_then(Value::as_str);
    match job.job_type {
        // Reached only for an ineligible image shape (the eligible candle lanes early-return `Ok`
        // above): a torch-only family, or a conditioned shape with no candle lane — incl. the
        // sc-5968 strict-pose-on-an-unwired-family trap.
        JobType::ImageGenerate | JobType::ImageEdit => Err(classify_candle_image_gap(&job.payload)),

        // SenseNova-U1 VQA / interleave run on candle (sc-5501); eligible jobs early-return `Ok`.
        // This arm is reached only for an understanding job on a model with no candle path.
        JobType::ImageVqa | JobType::ImageInterleave => Err(UnsupportedReason::new(
            model,
            "image understanding / interleave on this model",
            "image VQA / interleaved generation runs on candle for the SenseNova-U1 model \
             (sensenova_u1_8b[_fast]); other models have no candle understanding path off-Mac.",
            Some("epic 3180"),
        )),

        JobType::VideoGenerate => Err(classify_candle_video_gap(&job.payload)),

        // Wan-VACE extend/bridge/replace run on candle (sc-5494); eligible jobs early-return `Ok`.
        // This arm is the gap: an engine with no candle keyframe/clip path.
        JobType::VideoExtend | JobType::VideoBridge => Err(UnsupportedReason::new(
            model,
            "extend / bridge on this engine",
            "extend_clip / video_bridge run on candle only on the Wan-VACE path (the \
             replace-capable Wan models); other engines have no candle keyframe/clip path off-Mac.",
            Some("epic 5481"),
        )),

        JobType::PersonReplace => Err(UnsupportedReason::new(
            model,
            "person replacement on this engine",
            "replace_person runs on candle via native Wan-VACE (the replace-capable Wan models); \
             this model has no candle video engine off-Mac.",
            Some("epic 5481"),
        )),

        // image_upscale eligible engines (Real-ESRGAN + SeedVR2) early-return `Ok`; this arm is the
        // dropped AuraSR engine (sc-3668 / sc-5499 — a defensive submit-time guard, the UI hides it).
        JobType::ImageUpscale => Err(UnsupportedReason::new(
            model,
            "image_upscale (AuraSR)",
            "the candle upscaler runs Real-ESRGAN (+ SeedVR2); the AuraSR engine is dropped as an \
             offered engine (sc-3668 / sc-5499).",
            Some("sc-5499"),
        )),

        // video_upscale SeedVR2 early-returns `Ok`; this arm is the defensive non-SeedVR2 guard.
        JobType::VideoUpscale => Err(UnsupportedReason::new(
            model,
            "video_upscale (non-SeedVR2 engine)",
            "video upscaling runs on the candle SeedVR2 engine (seedvr2); no other engine is \
             available off-Mac.",
            Some("epic 5482"),
        )),

        // JoyCaption early-returns `Ok`; this arm is a non-JoyCaption captioner.
        JobType::TrainingCaption => Err(UnsupportedReason::new(
            None,
            "dataset captioning",
            "this dataset captioning job is not in the candle JoyCaption flow.",
            Some("sc-5098"),
        )),

        // Not enforce-failed by this slice (biased to `Ok`). The MLX-agnostic in-process job types
        // (downloads, model import/convert, ffmpeg, prompt refine — sc-5525) run with zero torch
        // off-Mac. The CV-aux / segment / tile-detail / training surfaces route by capability and
        // are still co-served by the Python torch worker until their phase epics close (Phase 5
        // epic 5482 for person/pose/kps; the candle SAM3 segment of sc-5062; the training cutover
        // for lora_train) — leaving them `Ok` keeps the enforce sweep from killing a job torch
        // still serves. An unrecognized job type is never a known gap (forward-compat).
        JobType::Placeholder
        | JobType::ModelDownload
        | JobType::ModelImport
        | JobType::ModelConvert
        | JobType::LoraImport
        | JobType::LoraDownload
        | JobType::FrameExtract
        | JobType::TimelineExport
        | JobType::PromptRefine
        | JobType::ImageDetail
        | JobType::PersonDetect
        | JobType::PersonTrack
        | JobType::PoseDetect
        | JobType::KpsExtract
        | JobType::ImageSegment
        | JobType::LoraTrain
        // epic 10159: control_training routes by capability (only a candle worker with the
        // krea_control trainer advertises it); candle-eligible ones already early-returned Ok via
        // `job_is_any_candle_eligible`, so reaching here means "no candle worker yet" — leave Ok
        // rather than enforce-fail, the same "parity landing later" treatment as lora_train.
        | JobType::ControlTraining
        // sc-6535: a candle CLIP embedder (`candle-gen-clip`) is future work; until then
        // dataset_analysis routes by capability (no candle worker advertises it) rather than
        // enforce-failing — the same "parity landing later" treatment as the surfaces above.
        | JobType::DatasetAnalysis
        // sc-6539: dataset_upscale parity on candle routes by capability, like dataset_analysis.
        | JobType::DatasetUpscale
        // sc-6538: dataset_face_analysis on the candle lane (candle-gen-face) routes by capability too.
        | JobType::DatasetFaceAnalysis
        // sc-4415: face_likeness_compare on the candle lane (candle-gen-face) routes by capability too.
        | JobType::FaceLikenessCompare
        | JobType::Unknown(_) => Ok(()),
    }
}

/// Name the precise gap for a candle-ineligible image job (sc-5502) — the candle-worded,
/// candle-parity twin of [`classify_image_gap`]. The strict-pose-on-an-unwired-family case (the
/// canonical sc-5968 silent-T2I trap) is named precisely; the rest report whether the model has no
/// candle engine at all or is a candle txt2img family asked for a conditioned shape with no lane.
pub(crate) fn classify_candle_image_gap(payload: &Map<String, Value>) -> UnsupportedReason {
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
        return UnsupportedReason::new(None, "image generation", "no model specified.", None);
    };
    // The sc-5968 case generalized: a candle family with no strict-pose lane asked for poses —
    // it would otherwise silently render an unconditioned image, so it is a hard gap off-Mac.
    if image_request_candle_pose_reject(model, payload) {
        return UnsupportedReason::new(
            Some(model),
            "strict-pose ControlNet",
            "this model has no candle strict-pose lane (candle serves strict pose for qwen_image / \
             kolors / z_image_turbo, and SDXL via InstantID); the pose request would otherwise \
             silently render an unconditioned image, so it is rejected off-Mac.",
            Some("sc-5489"),
        );
    }
    if !CANDLE_ROUTED_MODELS.contains(&model) {
        return UnsupportedReason::new(
            Some(model),
            "unsupported image model / shape",
            "this model (or its requested conditioning shape) has no candle/CUDA lane off-Mac \
             until its port lands.",
            Some("epic 3692"),
        );
    }
    // A candle txt2img family but a conditioned shape (edit / reference / inpaint / LoRA / quant)
    // with no candle lane for it (the candle identity/control/edit lanes early-return `Ok`).
    UnsupportedReason::new(
        Some(model),
        "conditioned shape on a txt2img candle family",
        "this candle family serves text-to-image; the requested edit / reference / inpaint / LoRA / \
         quant shape has no candle lane for it off-Mac.",
        Some("epic 5480"),
    )
}

/// Name the precise gap for a candle-ineligible `video_generate` job (sc-5502) — the candle-worded
/// twin of [`classify_video_gap`]: a torch-only video model or an advanced/conditioned mode with no
/// candle lane.
pub(crate) fn classify_candle_video_gap(payload: &Map<String, Value>) -> UnsupportedReason {
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
        return UnsupportedReason::new(None, "video generation", "no model specified.", None);
    };
    if !CANDLE_VIDEO_ROUTED_MODELS.contains(&model) {
        return UnsupportedReason::new(
            Some(model),
            "unsupported video model",
            "this video model has no candle/CUDA engine off-Mac.",
            Some("epic 5095"),
        );
    }
    UnsupportedReason::new(
        Some(model),
        "advanced / conditioned video mode",
        "this video_generate mode is not candle-eligible on this model (candle serves base \
         text-to-video, the 14B I2V + SVD image-to-video, and Wan-VACE extend/bridge/replace); \
         other conditioned modes + LoRAs have no candle path off-Mac.",
        Some("epic 5481"),
    )
}

/// Name the precise gap for an ineligible `image_generate` / `image_edit` job: a torch-only
/// model, or a torch-only feature on an otherwise-MLX family. Mirrors the per-family
/// `*_mlx_eligible` gates so the reason matches why routing refused it.
pub(crate) fn classify_image_gap(payload: &Map<String, Value>) -> UnsupportedReason {
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
        return UnsupportedReason::new(None, "image generation", "no model specified.", None);
    };
    if !MLX_ROUTED_MODELS.contains(&model) {
        // No whole-model torch-only image family remains: every one was ported to MLX and moved
        // into `MLX_ROUTED_MODELS`, so anything reaching here is an unported model with no port
        // epic yet. Kolors (epic 3090 / sc-3875), InstantID (epic 3109 / sc-3345), PuLID-FLUX
        // (epic 3069 / sc-3344), z_image_edit (epic 3529 / sc-3923), Chroma (epic 3531 /
        // sc-3843), SenseNova-U1 (epic 3180 / sc-3900), and finally Lens / Lens-Turbo (epic 3164
        // / sc-5105 — the LAST one) all routed. Models with a partial surface (e.g. InstantID
        // pose-library, PuLID reference-less) are named per-feature below, not here. The old
        // `torch_only_image_model_epic` seam was retired once it became permanently `None`
        // (sc-8951); a future torch-only image model reintroduces per-model epic mapping here.
        // Keep in sync with `docs/mac-rust-gaps.md` §1.
        return UnsupportedReason::new(
            Some(model),
            "unsupported image model",
            "this model has no MLX engine and no port epic yet — file a porting epic and drop it on Mac (epic 3482 policy).",
            None,
        );
    }
    // Third-party LyCORIS (LoHa / non-peft LoKr) now applies on every MLX provider (epic 3641,
    // sc-3642/3643/3671), so it is no longer an image gap.
    match model {
        "qwen_image" => UnsupportedReason::new(
            Some(model),
            "reference / edit conditioning",
            "base Qwen-Image reference / edit_image conditioning is not available in the native flow on Mac unless it is the strict-pose ControlNet tier.",
            Some("epic 3401"),
        ),
        "flux_schnell" | "flux_dev" => UnsupportedReason::new(
            Some(model),
            "reference (XLabs IP-Adapter)",
            "FLUX.1 reference is the XLabs IP-Adapter (not img2img-init); it is not available in the native flow on Mac until the MLX port lands. (FLUX.1 edit_image has no path on any platform — a future Kontext capability, not an eradication gap; see sc-3535.)",
            Some("epic 3621"),
        ),
        "qwen_image_edit"
        | "qwen_image_edit_2509"
        | "qwen_image_edit_2511"
        | "qwen_image_edit_2511_lightning" => UnsupportedReason::new(
            Some(model),
            "edit without a reference/source image",
            "the Qwen-Image-Edit model needs edit_image+sourceAssetId or character_image+referenceAssetId to route to MLX.",
            None,
        ),
        "sensenova_u1_8b" | "sensenova_u1_8b_fast" => {
            let has_poses = payload
                .get("advanced")
                .and_then(Value::as_object)
                .and_then(|advanced| advanced.get("poses"))
                .and_then(Value::as_array)
                .is_some_and(|poses| !poses.is_empty());
            if has_poses {
                UnsupportedReason::new(
                    Some(model),
                    "strict pose (ControlNet)",
                    "SenseNova-U1 has no ControlNet/skeleton conditioning — the strict-pose tier is not an MLX path; it is not available in the native flow on Mac (dropped on Mac).",
                    Some("epic 3180"),
                )
            } else {
                UnsupportedReason::new(
                    Some(model),
                    "edit/character without a reference",
                    "SenseNova-U1 edit needs edit_image+sourceAssetId, and Character Studio needs character_image+referenceAssetId, to route to MLX.",
                    None,
                )
            }
        }
        // InstantID (sc-3345 identity + angle set; sc-3381 pose mode + face-restore): the full
        // surface runs on MLX for `character_image` + `referenceAssetId`. Only a non-character /
        // reference-less job has no InstantID path. Mirrors `instantid_mlx_eligible`.
        "instantid_realvisxl" => UnsupportedReason::new(
            Some(model),
            "InstantID without a character reference",
            "InstantID runs on MLX for character_image with a referenceAssetId (single identity, the 11-view angle set, pose-library mode, and face-restore); a non-character / reference-less job has no InstantID path.",
            None,
        ),
        // PuLID-FLUX (sc-3344): runs on MLX only for character_image with a referenceAssetId (the
        // face it injects). A non-character / reference-less job has no PuLID path. Mirrors
        // `pulid_flux_mlx_eligible`.
        "pulid_flux_dev" => UnsupportedReason::new(
            Some(model),
            "PuLID-FLUX without a character reference",
            "PuLID-FLUX runs on MLX for character_image with a referenceAssetId (the reference face drives the identity injection); a non-character / reference-less job has no PuLID-FLUX path.",
            None,
        ),
        // Kolors (epic 3090) runs its full surface on MLX now — T2I (sc-3875), img2img (sc-4765),
        // the IP-Adapter-Plus reference (sc-4767) and the strict-pose tier (sc-4766 / engine sc-5012)
        // — so a kolors job is never gap-classified; any residual falls to the defensive arm below.
        // flux2 / sdxl / realvisxl only fall out via LyCORIS (handled above) — defensive.
        _ => UnsupportedReason::new(
            Some(model),
            "unsupported configuration",
            "this model/feature combination is not in the MLX flow.",
            None,
        ),
    }
}

/// Name the precise gap for an ineligible `video_generate` job: a torch-only model (incl. SVD) or
/// an advanced mode. Mirrors `video_job_is_mlx_eligible`. (Third-party LyCORIS and LoKr-on-Wan now
/// apply on the MLX Wan/LTX paths — epic 3641 sc-3671 — so neither is a video gap anymore.)
pub(crate) fn classify_video_gap(payload: &Map<String, Value>) -> UnsupportedReason {
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
        return UnsupportedReason::new(None, "video generation", "no model specified.", None);
    };
    if !VIDEO_MLX_ROUTED_MODELS.contains(&model) {
        return UnsupportedReason::new(
            Some(model),
            "unsupported video model",
            "this video model has no MLX engine; it is not available in the native flow on Mac.",
            Some("epic 3040"),
        );
    }
    let mode = payload
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("image_to_video");
    if !video_mode_is_mlx_eligible(model, mode) {
        return UnsupportedReason::new(
            Some(model),
            "advanced video mode",
            "this video_generate mode is not MLX-eligible on this model (first_last_frame / \
             extend_clip / video_bridge / replace_person route to MLX only on the capable engines — \
             LTX + Wan TI2V-5B for the keyframe/clip modes, Wan-VACE for replace_person).",
            Some("epic 3040"),
        );
    }
    UnsupportedReason::new(
        Some(model),
        "unsupported video configuration",
        "this video configuration is not in the MLX flow.",
        None,
    )
}

/// Name the precise gap for an ineligible `lora_train` job. Mirrors `training_job_is_mlx_eligible`:
/// a kernel with no native mlx-gen Rust trainer, or LoKr-on-Wan.
pub(crate) fn classify_training_gap(payload: &Map<String, Value>) -> UnsupportedReason {
    let kernel = payload
        .get("plan")
        .and_then(Value::as_object)
        .and_then(|plan| plan.get("target"))
        .and_then(Value::as_object)
        .and_then(|target| target.get("kernel"))
        .and_then(Value::as_str);
    match kernel {
        // `kolors_lora` (sc-4568/sc-4732) and `lens_lora` (sc-5148/sc-5180) are no longer gaps —
        // both have native mlx-gen Rust trainers and route to the mlx worker, so they never reach
        // this classifier.
        Some("wan_lora") | Some("wan_moe_lora") => UnsupportedReason::new(
            None,
            "LoKr-on-Wan training",
            "Wan LoKr training is not available in the native flow on Mac (no Kronecker merge in the mlx Wan path).",
            Some("epic 3039"),
        ),
        _ => UnsupportedReason::new(
            None,
            "LoRA/LoKr training",
            "this training kernel has no native mlx-gen trainer.",
            Some("epic 3039"),
        ),
    }
}

/// The `mlx.converter` discriminators the native, in-process mlx-gen converters handle — the single
/// source of truth the convert-gap gate derives its allow-list from, mirroring the
/// `resolve_convert_plan` match arms in `sceneworks-worker` (each id here is a real arm there).
///
/// A `model_convert` job copies its `converter` payload verbatim from a model's `mlx.converter`
/// manifest field (`create_model_convert_job` in `apps/rust-api`), so every reachable converter — a
/// builtin model's OR a user model's — is either named here or is genuinely unsupported. Deriving the
/// gate from this const (rather than a second, hand-maintained copy of the list inside the gate) is
/// what fixes sc-10573: `ltx_video` was a live, shipped converter (the LTX-2.3 "10 Eros"
/// install-convert) yet the gate's old hardcoded list omitted it, so `mac_rust_supported`
/// mis-classified a valid LTX conversion as an mlx gap — a spurious `mlx_unsupported` warn event
/// today, and a wrongful terminal failure once enforce mode ships (sc-3492).
///
/// Two drift guards (see their tests) keep this list honest so it can't go stale again:
/// - `sceneworks-core` (`every_builtin_manifest_converter_is_in_the_convert_gap_allowlist`): every
///   `mlx.converter` in the embedded `builtin.models.jsonc` is listed here.
/// - `sceneworks-worker` (`native_converters_match_resolve_convert_plan_arms`): every id here is a
///   real `resolve_convert_plan` arm (not its "Unknown MLX converter." fallback), and a bogus id is
///   rejected — so the list can neither omit a live arm nor over-claim a nonexistent one.
pub const NATIVE_CONVERTERS: &[&str] = &[
    "flux2_klein_diffusers",
    "ltx_video",
    "flux2_dev_quant",
    "sd3_5_large_quant",
    "sd3_5_large_turbo_quant",
    "sd3_5_medium_quant",
    "anima_quant",
];

/// `model_convert` is supported for the in-process Rust converters enumerated in
/// [`NATIVE_CONVERTERS`] (FLUX.2-klein `flux2_klein_diffusers` sc-3136; LTX-2.3 `ltx_video` sc-3240;
/// FLUX.2-dev `flux2_dev_quant` sc-5921; the SD3.5 `sd3_5_*_quant` variants sc-7871; Anima
/// `anima_quant` sc-10517). The allow-list is derived from that const so it can never drift from the
/// worker's real converter set again (sc-10573). The default/absent converter is the retired Python
/// mlx-video path (sc-3491 / sc-3224) — still a gap.
pub(crate) fn classify_convert_gap(payload: &Map<String, Value>) -> Result<(), UnsupportedReason> {
    let converter = payload
        .get("converter")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if NATIVE_CONVERTERS.contains(&converter) {
        return Ok(());
    }
    Err(UnsupportedReason::new(
        payload.get("model").and_then(Value::as_str),
        "Wan/LTX model conversion (mlx_video)",
        "installing a non-turnkey Wan/LTX checkpoint converts via the native mlx_video path.",
        Some("sc-3491 / sc-3224"),
    ))
}
