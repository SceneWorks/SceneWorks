//! Dataset prompt-batch describe job (SW-7082, feature SW-F362 "Create Prompt Batch from Dataset").
//!
//! The `dataset_batch_describe` job runs a vision-language describer over every image of a training
//! dataset (or a subset of its item ids) and returns, per image in dataset order, six description
//! fields (`description`, `style`, `camera`, `composition`, `lighting`, `environment`) plus the
//! image's pixel width and height. Training Studio assembles the prompt-batch text from those fields
//! in the web app; this job only describes.
//!
//! **The describing prompt.** The user edits only the instruction text (the `[SYSTEM]` block of
//! `image_batch_describe_v1.txt` is the default). The fixed JSON output contract (the asset's
//! `[OUTPUT]` block) is ALWAYS appended by this job after the instruction, so no edit can remove the
//! six fields the parser reads.
//!
//! **The describer seam.** The model is a payload field (`model`). Today the Vision Captioner
//! (Qwen3-VL, JSON-constrained decode through the resident refine-model cache) is the only accepted
//! value; anything else is refused. A second describer (JoyCaption, SW-7083) implements
//! [`BatchDescriber`] — its own decode and, since it has no constrained decoding, its own
//! [`BatchDescriber::parse`] — and joins [`DescribeModel`]; the job loop, progress, cancel and result
//! shape stay as they are.
//!
//! **Failure and cancel.** An image whose output cannot be produced (unreadable image, generation
//! error, reply that is not the six-field object) is returned flagged `failed` with its error rather
//! than dropped, and the job still completes. A model that cannot load fails the whole job. On cancel
//! the job ends `canceled` and its result carries every image already processed.

use super::*;

#[cfg(native_lane)]
use futures_util::future::BoxFuture;
#[cfg(native_lane)]
use gen_core::core_llm::CancelFlag;
#[cfg(native_lane)]
pub(crate) use sceneworks_core::dataset_batch_describe::{DESCRIBE_FIELDS, VISION_CAPTIONER_REPO};

/// The batch describer prompt asset: `[SYSTEM]` is the default editable instruction, `[OUTPUT]` the
/// fixed JSON output contract appended after any instruction, `[USER]` the user turn.
#[cfg(native_lane)]
const IMAGE_BATCH_DESCRIBE_V1: &str = include_str!("image_batch_describe_v1.txt");

#[cfg(native_lane)]
const CANCEL_MESSAGE: &str = "Dataset batch describe canceled by user.";

/// Reply budget. Six fields of detailed prose; a reply that reaches the cap is cut off mid-object
/// and so fails to parse, which flags that image. Mirrors the caption tasks' 4096 ceiling: a reply
/// that has said everything emits EOS far below it.
#[cfg(native_lane)]
const DESCRIBE_MAX_NEW_TOKENS: u32 = 4096;

/// Longest error kept on a failed item (an engine error can embed a long backtrace).
#[cfg(native_lane)]
const MAX_ITEM_ERROR_CHARS: usize = 1_000;

/// Which describer the job's `model` field selects. One variant today; JoyCaption joins in SW-7083.
#[cfg(native_lane)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DescribeModel {
    VisionCaptioner,
}

#[cfg(native_lane)]
impl DescribeModel {
    /// Resolve the payload's `model`. Absent or blank means the Vision Captioner; any other value is
    /// refused (the API refuses the same request with a 400 before queuing it).
    pub(crate) fn from_payload(model: Option<&str>) -> WorkerResult<Self> {
        match model.map(str::trim).filter(|value| !value.is_empty()) {
            None | Some(VISION_CAPTIONER_REPO) => Ok(Self::VisionCaptioner),
            Some(other) => Err(WorkerError::InvalidPayload(format!(
                "Dataset batch describe does not support model {other:?}; use the Vision Captioner \
                 ({VISION_CAPTIONER_REPO})."
            ))),
        }
    }

    pub(crate) fn repo(self) -> &'static str {
        match self {
            Self::VisionCaptioner => VISION_CAPTIONER_REPO,
        }
    }
}

/// The messages one image is described with: the system text is the instruction followed by the
/// fixed output contract; the user turn is the asset's `[USER]` text (the image rides beside it).
#[cfg(native_lane)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DescribePrompt {
    /// The editable describing instruction actually used (the default when the job sent none).
    pub(crate) instruction: String,
    /// The fixed JSON output contract, never editable.
    pub(crate) output_contract: String,
    pub(crate) user: String,
}

#[cfg(native_lane)]
impl DescribePrompt {
    /// Build the prompt for a job's optional instruction. Returns the prompt and whether the
    /// default instruction was used.
    pub(crate) fn for_instruction(instruction: Option<&str>) -> (Self, bool) {
        let custom = instruction
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let is_default = custom.is_none();
        let prompt = Self {
            instruction: custom.unwrap_or_else(default_describe_instruction),
            output_contract: crate::prompt_refine_jobs::magic_section_from(
                IMAGE_BATCH_DESCRIBE_V1,
                "OUTPUT",
            ),
            user: crate::prompt_refine_jobs::magic_section_from(IMAGE_BATCH_DESCRIBE_V1, "USER"),
        };
        (prompt, is_default)
    }

    /// The system message: instruction, then the output contract. The contract is appended last so
    /// it is the final word on the reply's shape whatever the instruction says.
    pub(crate) fn system(&self) -> String {
        format!("{}\n\n{}", self.instruction, self.output_contract)
    }
}

/// The default editable describing instruction: the asset's `[SYSTEM]` block (it holds no output
/// contract — that is the separate `[OUTPUT]` block).
#[cfg(native_lane)]
pub(crate) fn default_describe_instruction() -> String {
    crate::prompt_refine_jobs::magic_section_from(IMAGE_BATCH_DESCRIBE_V1, "SYSTEM")
}

/// The six parsed fields of one described image (each may be empty).
#[cfg(native_lane)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DescribedFields {
    pub(crate) fields: [String; 6],
}

/// Parse a describer reply into the six fields: the outermost JSON object (reasoning blocks and
/// code fences stripped). A missing or `null` field is an empty string; a field of any other
/// non-string type, a reply that is not an object, or an object with all six fields empty is an
/// error (the image is flagged, not dropped).
#[cfg(native_lane)]
pub(crate) fn parse_described_fields(raw: &str) -> Result<DescribedFields, String> {
    let cleaned = crate::prompt_refine_jobs::clean_json_output(raw);
    let value: Value = serde_json::from_str(&cleaned)
        .map_err(|error| format!("The describer reply is not a JSON object ({error})."))?;
    let Value::Object(object) = value else {
        return Err("The describer reply is not a JSON object.".to_owned());
    };
    let mut fields: [String; 6] = Default::default();
    for (slot, name) in fields.iter_mut().zip(DESCRIBE_FIELDS) {
        *slot = match object.get(name) {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(text)) => text.trim().to_owned(),
            Some(other) => {
                return Err(format!(
                    "The describer reply's {name:?} field is not a string ({other})."
                ))
            }
        };
    }
    if fields.iter().all(String::is_empty) {
        return Err("The describer reply has none of the six description fields.".to_owned());
    }
    Ok(DescribedFields { fields })
}

/// What a describer returns for one image. The outer `Err` fails the whole job (the model could not
/// load — every image would fail the same way); the inner `Err` is THIS image's failure, recorded on
/// its item.
#[cfg(native_lane)]
pub(crate) type DescribeFuture = BoxFuture<'static, WorkerResult<Result<String, String>>>;

/// One image describer behind the batch job (the SW-7083 JoyCaption seam).
#[cfg(native_lane)]
pub(crate) trait BatchDescriber: Send + Sync {
    /// Describe one decoded image with `prompt`, returning the raw reply. The future must stop
    /// promptly once `cancel` is tripped (a describer with its own cancel type bridges it by polling
    /// `cancel.is_cancelled()` from its progress callback).
    fn describe(
        &self,
        image: image::DynamicImage,
        prompt: &DescribePrompt,
        cancel: CancelFlag,
    ) -> DescribeFuture;

    /// Turn a raw reply into the six fields. The default is the strict JSON-object parse the
    /// constrained Vision Captioner needs; a describer without constrained decoding overrides it.
    fn parse(&self, raw: &str) -> Result<DescribedFields, String> {
        parse_described_fields(raw)
    }
}

/// The Vision Captioner (Qwen3-VL) describer: a JSON-constrained decode on the resident refine-model
/// cache, the model loaded once and reused across the dataset's images.
#[cfg(native_lane)]
struct VisionCaptionerDescriber {
    weights_dir: PathBuf,
    job_id: String,
}

#[cfg(native_lane)]
impl BatchDescriber for VisionCaptionerDescriber {
    fn describe(
        &self,
        image: image::DynamicImage,
        prompt: &DescribePrompt,
        cancel: CancelFlag,
    ) -> DescribeFuture {
        use gen_core::core_llm::{
            Constraint, Content, Message, ModelRequirements, Role, Sampling, StreamEvent,
            TextLlmRequest, ThinkingMode,
        };
        let spec = gen_core::core_llm::LoadSpec {
            source: self.weights_dir.to_string_lossy().into_owned(),
            // A snapshot directory carries its own projector (see `catalog_semantic_jobs`).
            ..Default::default()
        };
        // Resolve on the JSON constraint alone, exactly like the `image_caption` task: no provider
        // statically advertises vision + Json for a Qwen-VL snapshot; `mlx-llama`/`candle-llama`
        // flip to vision at load (see `prompt_refine_jobs`).
        let requirements = ModelRequirements::default().with_constraint(Constraint::Json);
        let system = prompt.system();
        let user = prompt.user.clone();
        let job_id = self.job_id.clone();
        let kv_compression = crate::llm_kv_cache::worker_default_policy();
        Box::pin(crate::refine_model_cache::with_cached_refiner(
            spec,
            requirements,
            "Vision Captioner load failed",
            move |model| -> WorkerResult<Result<String, String>> {
                let image = match crate::prompt_refine_jobs::vision_image_ref(image) {
                    Ok(image) => image,
                    Err(error) => {
                        return Ok(Err(format!("The image could not be prepared: {error}")))
                    }
                };
                // The Vision Captioner has no thinking mode; a no-think request keeps the JSON
                // mask safe to apply from the first token.
                let thinking = ThinkingMode::Disabled;
                let constraint = crate::prompt_refine_jobs::json_decode_constraint(
                    true,
                    model.descriptor().capabilities.supports_thinking,
                    thinking,
                );
                let request = TextLlmRequest {
                    progress: Some(crate::load_progress::text_observer()),
                    messages: vec![
                        Message::system(system),
                        Message {
                            role: Role::User,
                            content: vec![Content::Image(image), Content::text(user)],
                            thinking: None,
                            tool_calls: Vec::new(),
                        },
                    ],
                    sampling: Sampling {
                        temperature: 0.4,
                        top_p: 0.9,
                        ..Sampling::default()
                    },
                    max_new_tokens: DESCRIBE_MAX_NEW_TOKENS,
                    constraint,
                    cancel,
                    thinking,
                    kv_compression,
                    ..Default::default()
                };
                // sc-24029: bound MLX's process-global freed-buffer cache across the decode.
                let mut cache_bound = crate::mlx_decode_cache::DecodeCacheBound::mlx();
                let generation =
                    crate::load_progress::generate_text(model, &request, &mut |event| {
                        if matches!(event, StreamEvent::Token { .. }) {
                            cache_bound.note_event();
                        }
                    });
                crate::llm_kv_cache::KvCacheRecord::of(
                    crate::llm_kv_cache::NATIVE_LLM_ENGINE,
                    request.kv_compression,
                    &generation,
                    |output| output.kv_cache.as_ref(),
                )
                .emit(&job_id);
                Ok(generation
                    .map(|output| output.text)
                    .map_err(|error| format!("Vision Captioner generation failed: {error}")))
            },
        ))
    }
}

/// One dataset image to describe, as the API queued it.
#[cfg(native_lane)]
#[derive(Clone, Debug)]
struct DescribeItem {
    item_id: String,
    image_path: PathBuf,
    /// The image's path relative to the dataset root (echoed so the UI can show a thumbnail).
    relative_path: String,
    asset_id: Option<String>,
}

/// One item of the job result, in dataset order.
#[cfg(native_lane)]
#[derive(Clone, Debug)]
struct DescribedItem {
    item: DescribeItem,
    size: Option<(u32, u32)>,
    outcome: Result<DescribedFields, String>,
}

#[cfg(native_lane)]
impl DescribedItem {
    fn to_json(&self) -> Value {
        let mut object = JsonObject::new();
        object.insert("itemId".to_owned(), json!(self.item.item_id));
        object.insert("imagePath".to_owned(), json!(self.item.relative_path));
        object.insert("assetId".to_owned(), json!(self.item.asset_id));
        object.insert("width".to_owned(), json!(self.size.map(|size| size.0)));
        object.insert("height".to_owned(), json!(self.size.map(|size| size.1)));
        let empty = DescribedFields::default();
        let fields = self.outcome.as_ref().unwrap_or(&empty);
        for (name, text) in DESCRIBE_FIELDS.iter().zip(&fields.fields) {
            object.insert((*name).to_owned(), json!(text));
        }
        object.insert("failed".to_owned(), json!(self.outcome.is_err()));
        object.insert(
            "error".to_owned(),
            json!(self
                .outcome
                .as_ref()
                .err()
                .map(|error| bounded_error(error))),
        );
        Value::Object(object)
    }
}

#[cfg(native_lane)]
fn bounded_error(error: &str) -> String {
    if error.chars().count() <= MAX_ITEM_ERROR_CHARS {
        return error.to_owned();
    }
    let mut cut: String = error.chars().take(MAX_ITEM_ERROR_CHARS).collect();
    cut.push('…');
    cut
}

/// The validated job request.
#[cfg(native_lane)]
struct BatchDescribeRequest {
    model: DescribeModel,
    prompt: DescribePrompt,
    default_prompt: bool,
    dataset_id: String,
    dataset_version: Value,
    items: Vec<DescribeItem>,
}

#[cfg(native_lane)]
fn parse_batch_describe_request(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<BatchDescribeRequest> {
    let model = DescribeModel::from_payload(payload.get("model").and_then(Value::as_str))?;
    let instruction = match payload.get("prompt") {
        None | Some(Value::Null) => None,
        Some(Value::String(text))
            if text.chars().count()
                > sceneworks_core::dataset_batch_describe::MAX_DESCRIBE_PROMPT_CHARS =>
        {
            return Err(WorkerError::InvalidPayload(format!(
                "Dataset batch describe prompt is longer than {} characters.",
                sceneworks_core::dataset_batch_describe::MAX_DESCRIBE_PROMPT_CHARS
            )))
        }
        Some(Value::String(text)) => Some(text.as_str()),
        Some(other) => {
            return Err(WorkerError::InvalidPayload(format!(
                "Dataset batch describe prompt must be a string, got {other}."
            )))
        }
    };
    let (prompt, default_prompt) = DescribePrompt::for_instruction(instruction);
    let items = parse_dataset_items(
        settings,
        payload,
        "Batch describe",
        DatasetItemPolicy::Strict,
        |item| {
            Ok(DescribeItem {
                item_id: item.item_id.to_owned(),
                image_path: item.image_path()?,
                relative_path: item
                    .object
                    .get("relativePath")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                asset_id: crate::payload::trimmed_string(item.object.get("assetId"))
                    .map(str::to_owned),
            })
        },
    )?;
    if items.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Dataset batch describe job has no images to describe.".to_owned(),
        ));
    }
    Ok(BatchDescribeRequest {
        model,
        prompt,
        default_prompt,
        dataset_id: required_payload_string(payload, "datasetId")?.to_owned(),
        dataset_version: payload
            .get("datasetVersion")
            .cloned()
            .unwrap_or(Value::Null),
        items,
    })
}

/// The job result: the request echo plus every processed item in dataset order.
#[cfg(native_lane)]
fn batch_describe_result(
    request: &BatchDescribeRequest,
    described: &[DescribedItem],
    canceled: bool,
) -> JsonObject {
    let failed = described
        .iter()
        .filter(|item| item.outcome.is_err())
        .count();
    let mut result = JsonObject::new();
    result.insert("model".to_owned(), json!(request.model.repo()));
    result.insert("datasetId".to_owned(), json!(request.dataset_id));
    result.insert("datasetVersion".to_owned(), request.dataset_version.clone());
    result.insert("prompt".to_owned(), json!(request.prompt.instruction));
    result.insert(
        "promptSource".to_owned(),
        json!(if request.default_prompt {
            "default"
        } else {
            "custom"
        }),
    );
    result.insert("itemCount".to_owned(), json!(request.items.len()));
    result.insert("processedCount".to_owned(), json!(described.len()));
    result.insert("describedCount".to_owned(), json!(described.len() - failed));
    result.insert("failedCount".to_owned(), json!(failed));
    result.insert("canceled".to_owned(), json!(canceled));
    result.insert(
        "items".to_owned(),
        Value::Array(described.iter().map(DescribedItem::to_json).collect()),
    );
    result
}

/// Progress before image `index` (0-based) of `total` starts: 0.08 at the first image, 0.96 once
/// every image is done.
#[cfg(native_lane)]
fn image_progress(done: usize, total: usize) -> f64 {
    0.08 + 0.88 * (done as f64 / total.max(1) as f64)
}

#[cfg(native_lane)]
pub(crate) async fn run_dataset_batch_describe_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = parse_batch_describe_request(settings, &job.payload)?;
    let describer: Box<dyn BatchDescriber> = match request.model {
        DescribeModel::VisionCaptioner => Box::new(VisionCaptionerDescriber {
            weights_dir: resolve_app_managed_model_dir(
                settings,
                VISION_CAPTIONER_REPO,
                "Vision Captioner model path",
            )?,
            job_id: job.id.clone(),
        }),
    };
    run_batch_describe(api, settings, job, request, describer.as_ref()).await
}

/// The job loop, describer injected so a test drives the real progress/cancel/result path with a
/// stub describer and no model.
#[cfg(native_lane)]
async fn run_batch_describe(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: BatchDescribeRequest,
    describer: &dyn BatchDescriber,
) -> WorkerResult<()> {
    let backend = backend_label(&settings.gpu_id);
    let total = request.items.len();
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload_for_backend(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.04,
            "Preparing dataset batch describe.",
            None,
            backend,
        ),
    )
    .await?;

    let mut described: Vec<DescribedItem> = Vec::with_capacity(total);
    let mut canceled = false;
    for (index, item) in request.items.iter().enumerate() {
        // Honour a cancel between images promptly, not only on the supervisor's heartbeat tick.
        if cancel_requested_peek(api, &job.id).await {
            canceled = true;
            break;
        }
        update_job(
            api,
            &job.id,
            progress_payload_for_backend(
                JobStatus::Running,
                ProgressStage::Running,
                image_progress(index, total),
                &format!("Describing image {} of {total}.", index + 1),
                None,
                backend,
            ),
        )
        .await?;

        let path = item.image_path.clone();
        let decoded = crate::load_progress::spawn_blocking(move || {
            crate::image_decode::decode_image_any(&path)
        })
        .await
        .map_err(|error| task_join_error("batch describe image decode", error))?;
        let outcome = match decoded {
            Err(error) => DescribedItem {
                item: item.clone(),
                size: None,
                outcome: Err(format!("The image could not be read: {error}")),
            },
            Ok(image) => {
                let size = (image.width(), image.height());
                let cancel = CancelFlag::new();
                let generation = describer.describe(image, &request.prompt, cancel.clone());
                let (was_canceled, reply) = StreamSupervisor::new(
                    cancel,
                    generation,
                    progress_report_interval(settings),
                    None,
                )
                .wait(api, settings, &job.id)
                .await?;
                if was_canceled {
                    // This image was interrupted mid-description: it has no result to return.
                    canceled = true;
                    break;
                }
                DescribedItem {
                    item: item.clone(),
                    size: Some(size),
                    outcome: reply?.and_then(|raw| describer.parse(&raw)),
                }
            }
        };
        described.push(outcome);
        let done = described.len();
        update_job(
            api,
            &job.id,
            progress_payload_for_backend(
                JobStatus::Running,
                ProgressStage::Running,
                image_progress(done, total),
                &format!("Described {done} of {total} images."),
                Some(batch_describe_result(&request, &described, false)),
                backend,
            ),
        )
        .await?;
    }

    let result = batch_describe_result(&request, &described, canceled);
    if canceled {
        update_job(
            api,
            &job.id,
            progress_payload_for_backend(
                JobStatus::Canceled,
                ProgressStage::Canceled,
                1.0,
                CANCEL_MESSAGE,
                Some(result),
                backend,
            ),
        )
        .await?;
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
    }
    let failed = described
        .iter()
        .filter(|item| item.outcome.is_err())
        .count();
    let message = if failed == 0 {
        format!("Described {total} image(s).")
    } else {
        format!(
            "Described {} of {total} image(s); {failed} failed.",
            total - failed
        )
    };
    update_job(
        api,
        &job.id,
        progress_payload_for_backend(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &message,
            Some(result),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// Without a native vision provider (neither the macOS MLX worker nor the candle build) the
/// `prompt_refine` capability this job routes by is never advertised, so the job never arrives here.
#[cfg(not(native_lane))]
pub(crate) async fn run_dataset_batch_describe_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "Dataset batch describe needs the macOS MLX worker or the candle backend.".to_owned(),
    ))
}

#[cfg(all(test, native_lane))]
#[path = "batch_describe_jobs_tests.rs"]
mod tests;
