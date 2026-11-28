//! Per-job FlashAttention-2 scope for every candle generation job (SW-7276, feature SW-F379), and
//! the per-job lossy-attention opt-ins (SW-6132, feature SW-F287).
//!
//! candle-gen routes every candle model's attention through one shared FlashAttention-2 dispatch
//! behind a single process-wide toggle (`runtime_cuda::attention`). Each GPU worker is its own child
//! process that runs one job at a time, so a toggle set at job start is correct for the whole job
//! and cannot race another job. Because it is process-wide, a stale "off" from one job would
//! otherwise leak into the next, so:
//!
//! 1. [`AttentionJobScope`] is opened once per candle generation job at the utility-dispatch seam. It
//!    puts the toggle back to the default (on) and the lossy opt-ins back to OFF, resets the
//!    per-path counters, logs one structured `attention_path` event at job end, and restores both
//!    defaults when it ends or is dropped (success, error, cancel and unwind alike), so a job type
//!    that never states a preference cannot inherit the previous job's choice;
//! 2. the image and video lanes then set the toggle and the opt-ins from their typed, resolved
//!    request (`advanced.flashAttn`, default on; `advanced.lossyAttention`, default off) through
//!    [`apply_image_request`] / [`apply_video_request`], right before the candle route runs, for
//!    EVERY model family. Production code never reads the raw `advanced` map (the
//!    `source_coverage` gate), so the dispatch seam cannot read the payload itself;
//! 3. every candle load seam (`inference_runtime::load` / `load_transform` and the generator
//!    cache's request seams, warm hits included) calls [`admit_engine`] with the engine id before
//!    it loads or renders: `admit_lossy_attention` turns a provider or kernel that cannot take the
//!    requested opt-in into a typed `Unsupported`, the job's refusal. The API refuses the same
//!    requests earlier from the checked-in projection of the same table
//!    (`config/engine-capabilities/attention/shared-attention-routing.candle.json`, pinned to the
//!    linked `SHARED_ATTENTION_ROUTING` by `shared_attention_facts_match_the_linked_routing_table`).
//!
//! The counters are recorded per job: the `attention_path` log event, the job's generation metrics
//! (`attentionExecution`, returned by [`AttentionJobScope::finish`]) and, for a job that opted in
//! to a lossy kernel, its asset recipe (`rawAdapterSettings.attentionExecution`), so a user can see
//! whether the kernel actually ran.
use std::sync::Mutex;

use runtime_cuda::attention::{
    self, AttentionPathCounts, Int8AttentionConfig, KCentering, LossyAttentionOptIns, PvPrecision,
    SolConfig, SolSelection,
};
use sceneworks_core::contracts::{GenerationMetrics, JobSnapshot, JobType};
use sceneworks_core::image_request::ImageRequest;
use sceneworks_core::options::lossy_attention::{
    KCenteringWire, LossyAttentionWire, PvPrecisionWire, SolSelectionWire, ATTENTION_EXECUTION_KEY,
};
use sceneworks_core::video_request::VideoRequest;
use serde_json::{json, Map, Value};

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

/// What the current job asked for in `advanced.lossyAttention` (`None`: nothing). Kept beside the
/// engine switch so admission checks the REQUEST even if the switch refused it, and so the recipe
/// receipt can name what was requested.
static REQUESTED_LOSSY: Mutex<Option<LossyAttentionWire>> = Mutex::new(None);

fn requested_lossy_slot() -> std::sync::MutexGuard<'static, Option<LossyAttentionWire>> {
    REQUESTED_LOSSY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The engine's `LossyAttentionOptIns` for a request's wire (the defaults are inference's; see
/// `the_wire_maps_onto_the_inference_configs_and_their_defaults`). An invalid selection cannot
/// reach here (`parse_wire_value` is the API's shape gate); should one, it maps to an invalid
/// `SolConfig`, which `admit_lossy_attention` refuses with a typed error.
pub(crate) fn opt_ins_from_wire(wire: &LossyAttentionWire) -> LossyAttentionOptIns {
    let int8 = wire.int8.map(|int8| Int8AttentionConfig {
        pv: match int8.pv_precision() {
            PvPrecisionWire::ValueDtype => PvPrecision::ValueDtype,
            PvPrecisionWire::Int8 => PvPrecision::Int8,
        },
        k_centering: match int8.k_centering() {
            KCenteringWire::Off => KCentering::Off,
            KCenteringWire::Always => KCentering::Always,
            KCenteringWire::Adaptive => KCentering::Adaptive,
        },
    });
    let sol = wire.sol.map(|sol| SolConfig {
        selection: match sol.selection() {
            Ok(SolSelectionWire::TopK(ratio)) => SolSelection::TopK(ratio),
            Ok(SolSelectionWire::Tau(tau)) => SolSelection::Tau(tau),
            // Out of range: `SolConfig::validate` refuses it in admission.
            Err(_) => SolSelection::TopK(f32::NAN),
        },
        sink_blocks: sol.sink_blocks(),
        sink_q: sol.sink_q(),
        tail: sol.tail(),
        query_chunk_blocks: sol.query_chunk_blocks() as usize,
    });
    LossyAttentionOptIns { int8, sol }
}

/// Set the job-level lossy opt-ins from the request (absent: OFF) and remember the request.
fn apply_lossy(wire: Option<&LossyAttentionWire>) {
    let wire = wire.copied().filter(LossyAttentionWire::any);
    let opt_ins = wire
        .as_ref()
        .map(opt_ins_from_wire)
        .unwrap_or(LossyAttentionOptIns::OFF);
    if let Err(refusal) = attention::set_lossy_attention(opt_ins) {
        // Unreachable after shape validation; the switch stays OFF and admission refuses the
        // remembered request with the same reason before anything loads.
        tracing::warn!(error = %refusal, "lossy attention opt-in refused by the engine switch");
        let _ = attention::set_lossy_attention(LossyAttentionOptIns::OFF);
    }
    *requested_lossy_slot() = wire;
}

/// Reset the job-level opt-ins to OFF and forget the request.
fn reset_lossy() {
    let _ = attention::set_lossy_attention(LossyAttentionOptIns::OFF);
    *requested_lossy_slot() = None;
}

/// The current job's requested lossy attention (`None`: not requested).
pub(crate) fn requested_lossy_attention() -> Option<LossyAttentionWire> {
    *requested_lossy_slot()
}

/// The typed refusal of the current job's lossy-attention request for `engine_id`, before it
/// loads or renders (`Ok` when nothing is requested). See the module docs.
pub(crate) fn admit_engine(engine_id: &str) -> gen_core::Result<()> {
    let Some(wire) = requested_lossy_attention() else {
        return Ok(());
    };
    attention::admit_lossy_attention(engine_id, &opt_ins_from_wire(&wire))
}

/// The image lanes' entry: `advanced.flashAttn` and `advanced.lossyAttention` of the resolved
/// image request.
pub(crate) fn apply_image_request(request: &ImageRequest) -> bool {
    apply_lossy(request.lossy_attention());
    apply(request.advanced_bool("flashAttn"))
}

/// The video lanes' entry: `advanced.flashAttn` and `advanced.lossyAttention` of the resolved
/// video request.
pub(crate) fn apply_video_request(request: &VideoRequest) -> bool {
    let options = request.options();
    apply_lossy(options.lossy_attention.as_ref().map(|bound| &bound.value));
    apply(options.flash_attn.as_ref().map(|bound| bound.value))
}

/// The standalone lanes' entry (image upscale / Iris restoration, video upscale, dataset upscale):
/// they name an engine and have no typed request, so the shared core rule
/// (`sceneworks_core::lossy_attention::standalone_lossy_attention`, the API's enqueue check)
/// parses and admits the option for that engine; the opt-in is then set exactly as the studio
/// lanes set theirs. An engine without candle attention (Real-ESRGAN) is a typed refusal.
pub(crate) fn apply_standalone_request(job: &JobSnapshot) -> crate::WorkerResult<()> {
    let wire =
        sceneworks_core::lossy_attention::standalone_lossy_attention(&job.job_type, &job.payload)
            .map_err(|refusal| crate::WorkerError::InvalidPayload(refusal.detail))?;
    apply_lossy(wire.as_ref());
    Ok(())
}

fn log_attention_path(job_id: &str, flash_requested: bool, counts: &AttentionPathCounts) {
    tracing::info!(
        event = "attention_path",
        jobId = %job_id,
        flashRequested = flash_requested,
        flashCompiled = attention::FLASH_ATTN_COMPILED,
        lossyRequested = requested_lossy_attention().is_some(),
        flash = counts.flash,
        int8 = counts.int8,
        sol = counts.sol,
        int8Refusals = counts.int8_refusals.total(),
        solRefusals = counts.sol_refusals.total(),
        lossyF32Cast = counts.lossy_adapted.f32_cast,
        lossyHeadPad = counts.lossy_adapted.head_pad,
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

/// `name: count` for each nonzero count (a refusal tally lists only the reasons that fired).
fn nonzero(pairs: &[(&str, u64)]) -> Value {
    Value::Object(
        pairs
            .iter()
            .filter(|(_, count)| *count > 0)
            .map(|(name, count)| ((*name).to_owned(), json!(count)))
            .collect::<Map<_, _>>(),
    )
}

/// The job's attention execution receipt (`attentionExecution`): the request, and the paths the
/// engine took — FlashAttention, INT8 / Sol-Attn calls, each lossy refusal by reason, and the
/// operand adaptations (f32 run at bf16, head dim zero-padded).
pub(crate) fn attention_execution(
    requested: Option<&LossyAttentionWire>,
    counts: &AttentionPathCounts,
) -> Value {
    let r = &counts.int8_refusals;
    let s = &counts.sol_refusals;
    json!({
        "lossyRequested": requested
            .map(|wire| serde_json::to_value(wire).expect("serializable lossy attention wire")),
        "flash": counts.flash,
        "int8": counts.int8,
        "sol": counts.sol,
        "budgeted": counts.budgeted(),
        "total": counts.total(),
        "int8Refusals": nonzero(&[
            ("notCompiled", r.not_compiled),
            ("notCuda", r.not_cuda),
            ("unsupportedArch", r.unsupported_arch),
            ("dtype", r.dtype),
            ("mixedDtypeOrDevice", r.mixed_dtype_or_device),
            ("masked", r.masked),
            ("headDim", r.head_dim),
            ("shape", r.shape),
            ("indexOverflow", r.index_overflow),
            ("compile", r.compile),
        ]),
        "solRefusals": nonzero(&[
            ("notCompiled", s.not_compiled),
            ("notCuda", s.not_cuda),
            ("unsupportedArch", s.unsupported_arch),
            ("dtype", s.dtype),
            ("mixedDtypeOrDevice", s.mixed_dtype_or_device),
            ("masked", s.masked),
            ("headDim", s.head_dim),
            ("shape", s.shape),
            ("crossAttention", s.cross_attention),
            ("sequenceTooLong", s.sequence_too_long),
            ("indexOverflow", s.index_overflow),
            ("compile", s.compile),
            ("config", s.config),
        ]),
        "lossyAdapted": {
            "f32Cast": counts.lossy_adapted.f32_cast,
            "headPad": counts.lossy_adapted.head_pad,
        },
    })
}

/// Stamp this job's receipt onto an asset's raw settings when the job opted in to a lossy kernel
/// (replacing any request-carried copy: only this generation is authority). A job that did not
/// opt in records nothing in its recipe; its counts still reach the job's metrics.
pub(crate) fn record_attention_execution(raw_settings: &mut Value) {
    let Some(raw) = raw_settings.as_object_mut() else {
        return;
    };
    raw.remove(ATTENTION_EXECUTION_KEY);
    if let Some(wire) = requested_lossy_attention() {
        raw.insert(
            ATTENTION_EXECUTION_KEY.to_owned(),
            attention_execution(Some(&wire), &attention::attention_path_counts()),
        );
    }
}

/// [`record_attention_execution`] on every asset fact's `rawAdapterSettings`.
pub(crate) fn record_attention_execution_on_assets(asset_writes: &mut [Value]) {
    if requested_lossy_attention().is_none() {
        return;
    }
    for fact in asset_writes {
        if let Some(fact) = fact.as_object_mut() {
            let raw = fact
                .entry("rawAdapterSettings".to_owned())
                .or_insert_with(|| json!({}));
            record_attention_execution(raw);
        }
    }
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
        reset_lossy();
        // Discard whatever the previous job (or a warm-up) left behind.
        let _ = attention::take_attention_path_counts();
        Some(Self {
            job_id: job.id.clone(),
        })
    }

    /// Emit the job's `attention_path` event and return its generation-metrics block
    /// (`attentionExecution`, posted by the caller); the drop then restores the defaults.
    pub(crate) fn finish(self) -> GenerationMetrics {
        let counts = attention::attention_path_counts();
        log_attention_path(&self.job_id, attention::flash_attention_enabled(), &counts);
        GenerationMetrics {
            attention_execution: Some(attention_execution(
                requested_lossy_attention().as_ref(),
                &counts,
            )),
            ..Default::default()
        }
    }
}

impl Drop for AttentionJobScope {
    fn drop(&mut self) {
        apply(None);
        reset_lossy();
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
    use runtime_cuda::attention::{LossyServe, SharedAttentionRouting};
    use sceneworks_core::options::lossy_attention::{
        KernelServeWire, ProviderAttentionRouting, SharedAttentionFacts,
    };
    use serde_json::{json, Value};

    /// The linked `SHARED_ATTENTION_ROUTING` in the checked-in facts schema
    /// (`config/engine-capabilities/attention/shared-attention-routing.candle.json`).
    fn shared_attention_facts() -> SharedAttentionFacts {
        let serve = |serve: LossyServe| match serve {
            LossyServe::Served => KernelServeWire {
                served: true,
                reason: None,
            },
            LossyServe::Never(why) => KernelServeWire {
                served: false,
                reason: Some(why.to_owned()),
            },
        };
        SharedAttentionFacts {
            backend: "candle".to_owned(),
            source: "runtime_cuda::attention::SHARED_ATTENTION_ROUTING".to_owned(),
            providers: attention::SHARED_ATTENTION_ROUTING
                .iter()
                .map(|(id, routing)| {
                    let routing = match *routing {
                        SharedAttentionRouting::SharedHelpers { int8, sol } => {
                            ProviderAttentionRouting::SharedHelpers {
                                int8: serve(int8),
                                sol: serve(sol),
                            }
                        }
                        SharedAttentionRouting::Bespoke => ProviderAttentionRouting::Bespoke,
                    };
                    ((*id).to_owned(), routing)
                })
                .collect(),
        }
    }

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

    // ---------------------------------------------------------------------------------------------
    // SW-6132: lossy attention opt-ins.
    // ---------------------------------------------------------------------------------------------

    fn job(job_type: &str, id: &str) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": id, "type": job_type, "status": "running", "projectId": null,
            "projectName": null, "payload": {}, "result": {}, "requestedGpu": "auto",
            "assignedGpu": null, "workerId": null, "progress": 0, "stage": "running", "message": "",
            "error": null, "etaSeconds": null, "elapsedSeconds": null, "attempts": 1,
            "sourceJobId": null, "duplicateOfJobId": null, "cancelRequested": false,
            "createdAt": "t", "updatedAt": "t", "startedAt": null, "completedAt": null,
            "canceledAt": null, "lastHeartbeatAt": null
        }))
        .expect("job snapshot")
    }

    fn wire(value: Value) -> LossyAttentionWire {
        sceneworks_core::options::lossy_attention::parse_wire_value(&value).expect("valid wire")
    }

    #[test]
    fn the_wire_maps_onto_the_inference_configs_and_their_defaults() {
        // `int8: {}` is inference's `Int8AttentionConfig::default()`.
        let defaults = opt_ins_from_wire(&wire(json!({ "int8": {} })));
        assert_eq!(defaults.int8, Some(Int8AttentionConfig::default()));
        assert_eq!(defaults.sol, None);
        // `sol: { topkRatio }` is inference's `SolConfig::top_k(ratio)`.
        let sol = opt_ins_from_wire(&wire(json!({ "sol": { "topkRatio": 0.25 } })));
        assert_eq!(sol.sol, Some(SolConfig::top_k(0.25)));
        assert_eq!(sol.int8, None);
        // Every field, both kernels.
        let full = opt_ins_from_wire(&wire(json!({
            "int8": { "pvPrecision": "int8", "kCentering": "off" },
            "sol": { "tau": 0.5, "sinkBlocks": [0, 2], "sinkQ": [1, 4], "tail": false,
                     "queryChunkBlocks": 8 }
        })));
        assert_eq!(
            full.int8,
            Some(Int8AttentionConfig {
                k_centering: KCentering::Off,
                pv: PvPrecision::Int8,
            })
        );
        assert_eq!(
            full.sol,
            Some(SolConfig {
                selection: SolSelection::Tau(0.5),
                sink_blocks: (0, 2),
                sink_q: (1, 4),
                tail: false,
                query_chunk_blocks: 8,
            })
        );
        for (spelling, centering) in [
            ("off", KCentering::Off),
            ("always", KCentering::Always),
            ("adaptive", KCentering::Adaptive),
        ] {
            let opt_ins = opt_ins_from_wire(&wire(json!({ "int8": { "kCentering": spelling } })));
            assert_eq!(opt_ins.int8.unwrap().k_centering, centering, "{spelling}");
        }
        assert_eq!(
            opt_ins_from_wire(&LossyAttentionWire::default()),
            LossyAttentionOptIns::OFF
        );
    }

    /// The options crate's range rule is inference's `SolConfig::validate`: every probe the wire
    /// accepts the engine accepts, and every probe the wire refuses the engine refuses too.
    #[test]
    fn the_wire_range_rule_agrees_with_sol_config_validate() {
        let ratios = [-1.0, 0.0, 1e-50, 1e-6, 0.1, 0.5, 1.0, 1.0000001, 1.5];
        let sinks: [[u32; 2]; 5] = [[0, 0], [0, 4], [3, 1], [0, i32::MAX as u32], [0, u32::MAX]];
        for ratio in ratios {
            for sink in sinks {
                let raw =
                    json!({ "sol": { "topkRatio": ratio, "sinkBlocks": sink, "sinkQ": sink } });
                let parsed = sceneworks_core::options::lossy_attention::parse_wire_value(&raw);
                let engine = SolConfig {
                    selection: SolSelection::TopK(ratio as f32),
                    sink_blocks: (sink[0], sink[1]),
                    sink_q: (sink[0], sink[1]),
                    ..SolConfig::top_k(1.0)
                }
                .validate();
                assert_eq!(parsed.is_ok(), engine.is_ok(), "{raw}");
                if let Ok(parsed) = parsed {
                    assert!(opt_ins_from_wire(&parsed).validate().is_ok(), "{raw}");
                }
            }
        }
        for tau in [f64::MAX, -3.0, 0.0, 2.5] {
            let raw = json!({ "sol": { "tau": tau } });
            let parsed = sceneworks_core::options::lossy_attention::parse_wire_value(&raw);
            let engine = SolConfig {
                selection: SolSelection::Tau(tau as f32),
                ..SolConfig::top_k(1.0)
            }
            .validate();
            assert_eq!(parsed.is_ok(), engine.is_ok(), "{raw}");
        }
    }

    /// The API refuses unsupported families from the checked-in projection of the linked
    /// `SHARED_ATTENTION_ROUTING`; this pins the two together. Regenerate with
    /// `SCENEWORKS_WRITE_ATTENTION_FACTS=1 cargo test -p sceneworks-worker --features backend-candle
    /// --lib shared_attention_facts_match_the_linked_routing_table`.
    #[test]
    fn shared_attention_facts_match_the_linked_routing_table() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../config/engine-capabilities/attention/shared-attention-routing.candle.json",
        );
        let linked = shared_attention_facts();
        if std::env::var_os("SCENEWORKS_WRITE_ATTENTION_FACTS").is_some() {
            let text = serde_json::to_string_pretty(&linked).expect("facts serialize") + "\n";
            std::fs::write(&path, text).expect("write shared attention facts");
        }
        let committed: SharedAttentionFacts = serde_json::from_str(
            &std::fs::read_to_string(&path).expect("read shared attention facts"),
        )
        .expect("parse shared attention facts");
        assert_eq!(
            committed, linked,
            "config/engine-capabilities/attention/shared-attention-routing.candle.json is stale; \
             regenerate it (see this test's docs)"
        );
        assert_eq!(
            committed,
            *sceneworks_core::registry::shared_attention_facts(),
            "sceneworks-core embeds the same file"
        );
    }

    /// Every candle-routed builtin's projected capability is the linked inference answer for its
    /// engines, so the API's refusal and the worker's admission agree model by model.
    #[test]
    fn the_catalog_capability_agrees_with_inference_admission_for_every_builtin() {
        let mut checked = 0;
        for spec in sceneworks_core::registry::builtin_model_specs() {
            let engines = spec.candle_attention_provider_ids();
            let capability = spec.lossy_attention_capability();
            for (kernel, opt_ins) in [
                (
                    "int8",
                    LossyAttentionOptIns {
                        int8: Some(Int8AttentionConfig::default()),
                        sol: None,
                    },
                ),
                (
                    "sol",
                    LossyAttentionOptIns {
                        int8: None,
                        sol: Some(SolConfig::top_k(0.5)),
                    },
                ),
            ] {
                let served = capability.as_ref().is_some_and(|capability| match kernel {
                    "int8" => capability.int8.served,
                    _ => capability.sol.served,
                });
                let compiled = match kernel {
                    "int8" => attention::INT8_ATTN_COMPILED,
                    _ => attention::SOL_ATTN_COMPILED,
                };
                let admitted = !engines.is_empty()
                    && engines
                        .iter()
                        .all(|engine| attention::admit_lossy_attention(engine, &opt_ins).is_ok());
                assert_eq!(
                    served && compiled,
                    admitted,
                    "{} ({engines:?}) {kernel}",
                    spec.product.id
                );
                checked += 1;
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn admission_refuses_by_provider_and_kernel() {
        let _guard = TOGGLE.lock().unwrap_or_else(|e| e.into_inner());
        let int8 = wire(json!({ "int8": {} }));
        let sol = wire(json!({ "sol": { "topkRatio": 0.5 } }));
        let cases: [(&str, &LossyAttentionWire, bool); 8] = [
            ("krea_2_turbo", &int8, attention::INT8_ATTN_COMPILED),
            ("krea_2_turbo", &sol, attention::SOL_ATTN_COMPILED),
            ("wan2_2_t2v_14b", &sol, attention::SOL_ATTN_COMPILED),
            ("qwen_image_2_1", &int8, attention::INT8_ATTN_COMPILED),
            ("qwen_image_2_1", &sol, false),
            ("ideogram_4", &int8, false),
            ("sana_1600m", &int8, false),
            ("not_a_provider", &int8, false),
        ];
        for (engine, request, admitted) in cases {
            apply_lossy(Some(request));
            let result = admit_engine(engine);
            assert_eq!(result.is_ok(), admitted, "{engine} {request:?}: {result:?}");
            if let Err(error) = result {
                assert!(
                    matches!(error, gen_core::Error::Unsupported(_)),
                    "typed refusal: {error:?}"
                );
            }
        }
        // Nothing requested: every engine is admitted, bespoke ones included.
        reset_lossy();
        assert!(admit_engine("sana_1600m").is_ok());
        assert!(admit_engine("not_a_provider").is_ok());
    }

    #[test]
    fn a_lossy_job_followed_by_a_plain_job_leaves_the_switch_off() {
        let _guard = TOGGLE.lock().unwrap_or_else(|e| e.into_inner());
        let lossy = image_request(json!({
            "model": "krea2",
            "advanced": { "lossyAttention": { "int8": { "pvPrecision": "int8" } } }
        }));
        let plain = image_request(json!({ "model": "krea2", "advanced": {} }));

        let scope = AttentionJobScope::begin(&job("image_generate", "job_lossy")).unwrap();
        apply_image_request(&lossy);
        assert!(attention::lossy_attention().int8.is_some());
        assert!(requested_lossy_attention().is_some());
        let metrics = scope.finish();
        let receipt = metrics.attention_execution.expect("receipt");
        assert_eq!(receipt["lossyRequested"]["int8"]["pvPrecision"], "int8");
        assert_eq!(attention::lossy_attention(), LossyAttentionOptIns::OFF);
        assert!(requested_lossy_attention().is_none());

        let scope = AttentionJobScope::begin(&job("image_generate", "job_plain")).unwrap();
        apply_image_request(&plain);
        assert_eq!(attention::lossy_attention(), LossyAttentionOptIns::OFF);
        assert!(requested_lossy_attention().is_none());
        assert!(admit_engine("sana_1600m").is_ok(), "no opt-in, no refusal");
        let receipt = scope.finish().attention_execution.expect("receipt");
        assert_eq!(receipt["lossyRequested"], Value::Null);
        assert_eq!(receipt["int8"], 0);
        assert_eq!(receipt["sol"], 0);
        assert_eq!(attention::lossy_attention(), LossyAttentionOptIns::OFF);
    }

    #[test]
    fn a_failed_canceled_or_panicking_job_resets_the_opt_ins() {
        let _guard = TOGGLE.lock().unwrap_or_else(|e| e.into_inner());
        let lossy = video_request(json!({
            "projectId": "p",
            "advanced": { "lossyAttention": { "sol": { "topkRatio": 0.5 } } }
        }));
        // An error or a cancel drops the scope without `finish`.
        {
            let _scope = AttentionJobScope::begin(&job("video_generate", "job_err")).unwrap();
            apply_video_request(&lossy);
            assert!(attention::lossy_attention().sol.is_some());
        }
        assert_eq!(attention::lossy_attention(), LossyAttentionOptIns::OFF);
        assert!(requested_lossy_attention().is_none());
        // A panic unwinds through the scope.
        let unwound = std::panic::catch_unwind(|| {
            let _scope = AttentionJobScope::begin(&job("video_generate", "job_panic")).unwrap();
            apply_video_request(&lossy);
            panic!("engine panic");
        });
        assert!(unwound.is_err());
        assert_eq!(attention::lossy_attention(), LossyAttentionOptIns::OFF);
        assert!(requested_lossy_attention().is_none());
        // A stale opt-in left by anything else is cleared when the next job's scope opens.
        apply_video_request(&lossy);
        let scope = AttentionJobScope::begin(&job("video_generate", "job_next")).unwrap();
        assert_eq!(attention::lossy_attention(), LossyAttentionOptIns::OFF);
        drop(scope);
    }

    #[test]
    fn only_an_opted_in_job_stamps_its_recipe() {
        let _guard = TOGGLE.lock().unwrap_or_else(|e| e.into_inner());
        reset_lossy();
        let mut raw = json!({ "attentionExecution": { "forged": true }, "steps": 8 });
        record_attention_execution(&mut raw);
        assert!(
            raw.get("attentionExecution").is_none(),
            "a forged receipt is dropped"
        );
        apply_lossy(Some(&wire(json!({ "int8": {} }))));
        let mut assets = vec![json!({ "rawAdapterSettings": { "steps": 8 } }), json!({})];
        record_attention_execution_on_assets(&mut assets);
        for asset in &assets {
            let receipt = &asset["rawAdapterSettings"]["attentionExecution"];
            assert_eq!(receipt["lossyRequested"], json!({ "int8": {} }));
            for key in ["flash", "int8", "sol", "budgeted", "total"] {
                assert!(receipt[key].is_u64(), "{key}");
            }
            assert!(receipt["int8Refusals"].is_object());
            assert!(receipt["lossyAdapted"]["f32Cast"].is_u64());
        }
        reset_lossy();
    }

    #[test]
    fn the_receipt_lists_only_the_refusals_that_fired() {
        let mut counts = AttentionPathCounts {
            int8: 228,
            flash: 2,
            ..Default::default()
        };
        counts.int8_refusals.masked = 35;
        counts.int8_refusals.dtype = 1;
        counts.lossy_adapted.f32_cast = 4;
        let receipt = attention_execution(None, &counts);
        assert_eq!(receipt["int8"], 228);
        assert_eq!(receipt["int8Refusals"], json!({ "masked": 35, "dtype": 1 }));
        assert_eq!(receipt["solRefusals"], json!({}));
        assert_eq!(
            receipt["lossyAdapted"],
            json!({ "f32Cast": 4, "headPad": 0 })
        );
        assert_eq!(receipt["total"], 230);
    }

    fn standalone_job(job_type: &str, payload: Value) -> JobSnapshot {
        let mut job = job(job_type, "job_standalone");
        job.payload = payload.as_object().expect("payload").clone();
        job
    }

    /// Codex P2 on #3528: a standalone SeedVR2 upscale used to reset the opt-in and never apply
    /// it. The shared standalone helper now sets the job's switch (what the engine reads per
    /// attention call) and admits the engine before it loads.
    #[test]
    fn a_standalone_seedvr2_upscale_applies_the_requested_switch() {
        let _guard = TOGGLE.lock().unwrap_or_else(|e| e.into_inner());
        let job = standalone_job(
            "image_upscale",
            json!({ "sourceAssetId": "a", "engine": "seedvr2",
                    "advanced": { "lossyAttention": { "int8": { "kCentering": "always" } } } }),
        );
        let scope = AttentionJobScope::begin(&job).expect("upscale jobs open a scope");
        crate::inference_runtime::apply_standalone_lossy_attention(&job).unwrap();
        let switch = attention::lossy_attention();
        assert_eq!(
            switch.int8.map(|config| config.k_centering),
            Some(KCentering::Always)
        );
        assert!(requested_lossy_attention().is_some());
        assert_eq!(
            admit_engine("seedvr2").is_ok(),
            attention::INT8_ATTN_COMPILED,
            "seedvr2 serves INT8"
        );
        let receipt = scope.finish().attention_execution.expect("receipt");
        assert_eq!(receipt["lossyRequested"]["int8"]["kCentering"], "always");
        assert_eq!(attention::lossy_attention(), LossyAttentionOptIns::OFF);
    }

    #[test]
    fn every_standalone_handler_applies_or_refuses_through_the_shared_helper() {
        let _guard = TOGGLE.lock().unwrap_or_else(|e| e.into_inner());
        let sol = json!({ "sol": { "topkRatio": 0.5 } });
        // Served engines: Iris restoration and SeedVR2 video upscaling.
        for (job_type, payload) in [
            (
                "image_upscale",
                json!({ "engine": "iris", "advanced": { "lossyAttention": sol } }),
            ),
            (
                "video_upscale",
                json!({ "sourceAssetId": "v", "advanced": { "lossyAttention": sol } }),
            ),
        ] {
            let job = standalone_job(job_type, payload);
            let _scope = AttentionJobScope::begin(&job).expect("scope");
            crate::inference_runtime::apply_standalone_lossy_attention(&job).unwrap();
            assert!(attention::lossy_attention().sol.is_some(), "{job_type}");
        }
        assert_eq!(attention::lossy_attention(), LossyAttentionOptIns::OFF);
        // Real-ESRGAN (image default, dataset upscale) has no candle attention: a typed refusal,
        // and the switch stays OFF.
        for (job_type, payload) in [
            (
                "image_upscale",
                json!({ "advanced": { "lossyAttention": sol } }),
            ),
            (
                "dataset_upscale",
                json!({ "advanced": { "lossyAttention": sol } }),
            ),
        ] {
            let job = standalone_job(job_type, payload);
            let _scope = AttentionJobScope::begin(&job).expect("scope");
            let error = crate::inference_runtime::apply_standalone_lossy_attention(&job)
                .expect_err(job_type);
            assert!(
                matches!(error, crate::WorkerError::InvalidPayload(_)),
                "{job_type}: {error:?}"
            );
            assert_eq!(attention::lossy_attention(), LossyAttentionOptIns::OFF);
        }
        // Each standalone handler calls the shared helper before its engine loads.
        for (source, handler, calls) in [
            (include_str!("upscale_jobs.rs"), "upscale_jobs.rs", 2),
            (
                include_str!("video_jobs/seedvr2.rs"),
                "video_jobs/seedvr2.rs",
                1,
            ),
        ] {
            assert_eq!(
                source
                    .matches("crate::inference_runtime::apply_standalone_lossy_attention(job)?")
                    .count(),
                calls,
                "{handler}"
            );
        }
    }
}
