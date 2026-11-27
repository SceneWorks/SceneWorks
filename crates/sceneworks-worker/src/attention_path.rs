//! Per-job FlashAttention-2 scope for every candle generation job (SW-7276, feature SW-F379).
//!
//! candle-gen routes every candle model's attention through one shared FlashAttention-2 dispatch
//! behind a single process-wide toggle (`runtime_cuda::attention`). Each GPU worker is its own child
//! process that runs one job at a time, so a toggle set at job start is correct for the whole job
//! and cannot race another job. Because it is process-wide, a stale "off" from one job would
//! otherwise leak into the next, so:
//!
//! 1. [`AttentionJobScope`] is opened once per candle generation job at the utility-dispatch seam. It
//!    puts the toggle back to the default (on), resets the per-path counters, logs one structured
//!    `attention_path` event at job end, and restores the default when it ends or is dropped, so a
//!    job type that never states a preference cannot inherit the previous job's choice;
//! 2. the image and video lanes then set the toggle from their typed, resolved request
//!    (`advanced.flashAttn`, default on) through [`apply_image_request`] /
//!    [`apply_video_request`], right before the candle route runs, for EVERY model family.
//!    Production code never reads the raw `advanced` map (the `source_coverage` gate), so the
//!    dispatch seam cannot read the payload itself.
//!
//! The counters are recorded in the log event only: `GenerationMetrics` has no generic per-job
//! execution-facts map (its one provider observation, `veda_execution`, is H3-specific), and adding
//! a contract field plus a server merge for a diagnostic counter is not warranted.
use runtime_cuda::attention::{self, AttentionPathCounts};
use sceneworks_core::contracts::{JobSnapshot, JobType};
use sceneworks_core::image_request::ImageRequest;
use sceneworks_core::video_request::VideoRequest;

/// Job types that run candle generation models and therefore reach the shared attention helpers.
/// Everything else (analysis, downloads, training, audio, ...) keeps the default-on toggle.
pub(crate) fn job_runs_candle_attention(job_type: &JobType) -> bool {
    matches!(
        job_type,
        JobType::ImageGenerate
            | JobType::ImageEdit
            | JobType::ImageDetail
            | JobType::ImageInterleave
            | JobType::ImageVqa
            | JobType::ImageUpscale
            | JobType::DatasetUpscale
            | JobType::VideoGenerate
            | JobType::VideoExtend
            | JobType::VideoBridge
            | JobType::VideoUpscale
            | JobType::PersonReplace
    )
}

/// Set the shared toggle from a request's `flashAttn`; absent means the default, on. Returns the
/// value that was set.
fn apply(requested: Option<bool>) -> bool {
    let enabled = requested.unwrap_or(true);
    attention::set_flash_attention_enabled(enabled);
    enabled
}

/// The image lanes' entry: `advanced.flashAttn` of the resolved image request.
pub(crate) fn apply_image_request(request: &ImageRequest) -> bool {
    apply(request.advanced_bool("flashAttn"))
}

/// The video lanes' entry: `advanced.flashAttn` of the resolved video request.
pub(crate) fn apply_video_request(request: &VideoRequest) -> bool {
    apply(
        request
            .options()
            .flash_attn
            .as_ref()
            .map(|bound| bound.value),
    )
}

fn log_attention_path(job_id: &str, flash_requested: bool, counts: &AttentionPathCounts) {
    tracing::info!(
        event = "attention_path",
        jobId = %job_id,
        flashRequested = flash_requested,
        flashCompiled = attention::FLASH_ATTN_COMPILED,
        flash = counts.flash,
        notCompiled = counts.not_compiled,
        notCuda = counts.not_cuda,
        unsupportedArch = counts.unsupported_arch,
        dtype = counts.dtype,
        mixedDtypeOrDevice = counts.mixed_dtype_or_device,
        masked = counts.masked,
        shape = counts.shape,
        headDim = counts.head_dim,
        indexOverflow = counts.index_overflow,
        disabled = counts.disabled,
        gradTracked = counts.grad_tracked,
        materialized = counts.materialized,
        budgeted = counts.budgeted(),
        total = counts.total(),
        "candle attention path counts"
    );
}

/// Owns the toggle for one job. See the module docs.
pub(crate) struct AttentionJobScope {
    job_id: String,
}

impl AttentionJobScope {
    /// `None` for a job type that does not run candle generation models.
    pub(crate) fn begin(job: &JobSnapshot) -> Option<Self> {
        if !job_runs_candle_attention(&job.job_type) {
            return None;
        }
        apply(None);
        // Discard whatever the previous job (or a warm-up) left behind.
        let _ = attention::take_attention_path_counts();
        Some(Self {
            job_id: job.id.clone(),
        })
    }

    /// Emit the job's `attention_path` event; the drop then restores the default.
    pub(crate) fn finish(self) {
        log_attention_path(
            &self.job_id,
            attention::flash_attention_enabled(),
            &attention::attention_path_counts(),
        );
    }
}

impl Drop for AttentionJobScope {
    fn drop(&mut self) {
        apply(None);
    }
}

/// The toggle is process-wide and `cargo test` runs tests on threads: every test that flips it
/// (here and in `image_jobs::sdxl_control`) takes this lock.
#[cfg(test)]
pub(crate) static TOGGLE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::TOGGLE_TEST_LOCK as TOGGLE;
    use super::*;
    use serde_json::{json, Value};

    fn image_request(value: Value) -> ImageRequest {
        let payload = value.as_object().expect("payload");
        ImageRequest::from_resolved(payload, payload).expect("resolved image request")
    }

    fn video_request(value: Value) -> VideoRequest {
        let payload = value.as_object().expect("payload");
        VideoRequest::from_resolved(payload, payload).expect("resolved video request")
    }

    #[test]
    fn the_image_helper_sets_the_shared_toggle_for_a_non_sdxl_model() {
        let _guard = TOGGLE.lock().unwrap_or_else(|e| e.into_inner());
        // flux2 / qwen_image are neither SDXL nor Z-Image: the old per-model branch skipped them.
        for model in ["flux2_dev", "qwen_image", "krea2"] {
            attention::set_flash_attention_enabled(true);
            let off = image_request(json!({ "model": model, "advanced": { "flashAttn": false } }));
            assert!(!apply_image_request(&off), "{model}");
            assert!(
                !attention::flash_attention_enabled(),
                "{model}: off request"
            );

            // A later job that says nothing must not inherit the stale off: absent means on.
            let absent = image_request(json!({ "model": model, "advanced": {} }));
            assert!(apply_image_request(&absent), "{model}");
            assert!(attention::flash_attention_enabled(), "{model}: default on");

            let on = image_request(json!({ "model": model, "advanced": { "flashAttn": true } }));
            assert!(apply_image_request(&on), "{model}");
        }
    }

    #[test]
    fn the_video_helper_sets_the_shared_toggle_and_defaults_on() {
        let _guard = TOGGLE.lock().unwrap_or_else(|e| e.into_inner());
        let off = video_request(json!({ "projectId": "p", "advanced": { "flashAttn": false } }));
        assert!(!apply_video_request(&off));
        assert!(!attention::flash_attention_enabled());
        let absent = video_request(json!({ "projectId": "p", "advanced": {} }));
        assert!(apply_video_request(&absent));
        assert!(attention::flash_attention_enabled());
    }

    #[test]
    fn the_scope_resets_a_stale_off_and_the_counters() {
        let _guard = TOGGLE.lock().unwrap_or_else(|e| e.into_inner());
        // A previous job left the toggle off.
        attention::set_flash_attention_enabled(false);
        // `begin` needs a job; build the smallest valid snapshot through serde.
        let job: JobSnapshot = serde_json::from_value(json!({
            "id": "job_1", "type": "image_generate", "status": "running", "projectId": null,
            "projectName": null, "payload": {}, "result": {}, "requestedGpu": "auto",
            "assignedGpu": null, "workerId": null, "progress": 0, "stage": "running", "message": "",
            "error": null, "etaSeconds": null, "elapsedSeconds": null, "attempts": 1,
            "sourceJobId": null, "duplicateOfJobId": null, "cancelRequested": false,
            "createdAt": "t", "updatedAt": "t", "startedAt": null, "completedAt": null,
            "canceledAt": null, "lastHeartbeatAt": null
        }))
        .expect("job snapshot");
        let scope = AttentionJobScope::begin(&job).expect("image jobs open a scope");
        assert!(
            attention::flash_attention_enabled(),
            "stale off must not leak in"
        );
        assert_eq!(attention::attention_path_counts().total(), 0);
        apply(Some(false));
        scope.finish();
        assert!(
            attention::flash_attention_enabled(),
            "drop restores the default"
        );
    }

    /// The typed handlers that bypass `run_image_generate_job` still honour an explicit
    /// `flashAttn: false`: resolve each exactly as its handler does, then apply.
    #[test]
    fn explicit_false_reaches_the_toggle_through_every_typed_image_handler() {
        let _guard = TOGGLE.lock().unwrap_or_else(|e| e.into_inner());
        for (kind, model, extra) in [
            (
                "vqa",
                "sensenova_u1_8b",
                json!({"sourceAssetId":"s","question":"What?"}),
            ),
            ("interleave", "sensenova_u1_8b", json!({"prompt":"tree"})),
            (
                "imageDetail",
                "realvisxl",
                json!({"sourceAssetId":"s","prompt":"tree"}),
            ),
            ("image", "flux_dev", json!({"prompt":"tree"})),
        ] {
            let entry = sceneworks_core::registry::builtin_model_spec(model)
                .unwrap()
                .project(None);
            let mut payload = json!({"projectId":"p","model":model,"advanced":{"flashAttn":false}});
            for (key, value) in extra.as_object().unwrap() {
                payload[key] = value.clone();
            }
            let resolved = sceneworks_core::request_validation::resolve(
                kind,
                payload.as_object().unwrap(),
                entry.as_object().unwrap(),
            )
            .unwrap_or_else(|error| panic!("{kind}: {error:?}"));
            let request =
                ImageRequest::from_resolved(payload.as_object().unwrap(), &resolved).unwrap();
            attention::set_flash_attention_enabled(true);
            assert!(!apply_image_request(&request), "{kind}");
            assert!(!attention::flash_attention_enabled(), "{kind}");
        }
    }

    /// Every covered handler with a typed request calls the shared helper. (Upscale jobs carry no
    /// `advanced` options at all, so they run at the scope's default.)
    #[test]
    fn every_typed_candle_handler_calls_the_shared_helper() {
        for (source, handler, call) in [
            (
                include_str!("image_jobs.rs"),
                "image_jobs.rs",
                "apply_image_request(&execution)",
            ),
            (
                include_str!("video_jobs/mod.rs"),
                "video_jobs/mod.rs",
                "apply_video_request(&request)",
            ),
            (
                include_str!("image_jobs/detail.rs"),
                "detail.rs",
                "apply_image_request(&request)",
            ),
        ] {
            assert!(source.contains(call), "{handler}");
        }
        let sensenova = include_str!("sensenova_jobs.rs");
        assert_eq!(
            sensenova
                .matches("attention_path::apply_image_request(&request)")
                .count(),
            2,
            "vqa and interleave"
        );
    }

    #[test]
    fn only_candle_generation_job_types_open_a_scope() {
        for covered in [
            JobType::ImageGenerate,
            JobType::ImageEdit,
            JobType::VideoGenerate,
            JobType::VideoExtend,
            JobType::VideoBridge,
            JobType::PersonReplace,
            JobType::ImageUpscale,
            JobType::VideoUpscale,
        ] {
            assert!(job_runs_candle_attention(&covered), "{covered:?}");
        }
        for uncovered in [
            JobType::ModelDownload,
            JobType::LoraTrain,
            JobType::PersonDetect,
        ] {
            assert!(!job_runs_candle_attention(&uncovered), "{uncovered:?}");
        }
    }
}
