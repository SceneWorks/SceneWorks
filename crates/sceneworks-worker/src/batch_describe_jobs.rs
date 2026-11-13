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
//! **The describer seam.** The model is a payload field (`model`): the Vision Captioner (Qwen3-VL,
//! JSON-constrained decode through the resident refine-model cache, the default) or JoyCaption
//! (SW-7083, the `training_caption` captioner); anything else is refused. Each implements
//! [`BatchDescriber`]; the job loop, progress, cancel and result shape are shared.
//!
//! **JoyCaption.** JoyCaption (LLaVA) takes ONE user prompt beside the image and has no constrained
//! decoding, so it gets the instruction, the output contract and the user line concatenated into
//! that prompt (the contract still follows the instruction, so it stays the last word on shape). Its
//! providers refuse a prompt over 4,000 characters, so its default instruction is the asset's
//! condensed `[JOYCAPTION_SYSTEM]` block and a custom instruction is capped at
//! `MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS` (sceneworks-core). Its reply is parsed leniently (first JSON object found
//! anywhere in the reply), and an image whose reply still does not parse is described again, at most
//! [`JOYCAPTION_PARSE_RETRIES`] more times, before it is flagged.
//!
//! **Failure and cancel.** An image whose output cannot be produced (unreadable image, generation
//! error, reply that is not the six-field object) is returned flagged `failed` with its error rather
//! than dropped, and the job still completes. A model that cannot load fails the whole job. On cancel
//! the job ends `canceled` and every image already processed stays readable.
//!
//! **Where the items go.** Each finished image is appended, as it finishes, to the job's per-item
//! results (`POST /api/v1/jobs/:job_id/result-items`, stored beside the job row); the progress and
//! terminal posts carry only counts. A cumulative item list in the progress result would grow on
//! every post until a large dataset exceeded the API's JSON body limit (and every `job.updated`
//! event would carry it). One item is bounded too: each field is capped at
//! [`MAX_DESCRIBE_FIELD_CHARS`], and an item whose JSON still exceeded the store's
//! `MAX_JOB_RESULT_ITEM_BYTES` would be replaced by a flagged item without its text.

use super::*;

#[cfg(native_lane)]
use futures_util::future::BoxFuture;
#[cfg(native_lane)]
use gen_core::core_llm::CancelFlag;
#[cfg(native_lane)]
pub(crate) use sceneworks_core::dataset_batch_describe::{
    DESCRIBE_FIELDS, JOYCAPTION_REPO, QWEN38_27B_DESCRIBER_REPO, VISION_CAPTIONER_REPO,
};

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

/// JoyCaption's reply budget: the providers' own ceiling (mlx-gen and candle-gen JoyCaption both
/// advertise `max_new_tokens` 1024). Its default instruction asks for a reply that fits it.
#[cfg(native_lane)]
const JOYCAPTION_MAX_NEW_TOKENS: u32 = 1024;

/// How many more times a JoyCaption image is described when its reply does not parse. JoyCaption
/// samples (temperature 0.6, a fresh seed per call), so a second attempt is a different reply.
#[cfg(native_lane)]
pub(crate) const JOYCAPTION_PARSE_RETRIES: usize = 2;

/// Longest error kept on a failed item (an engine error can embed a long backtrace).
#[cfg(native_lane)]
const MAX_ITEM_ERROR_CHARS: usize = 1_000;

/// Longest text kept in one description field. The reply budgets (4,096 tokens for the Vision
/// Captioner, 1,024 for JoyCaption) already bound a parsed reply far below this in practice; the cap
/// makes the bound independent of the tokenizer, so six fields, each JSON-escaped at worst six
/// bytes per character, stay well under the store's per-item limit (`MAX_JOB_RESULT_ITEM_BYTES`).
#[cfg(native_lane)]
pub(crate) const MAX_DESCRIBE_FIELD_CHARS: usize = 16_000;

/// Which describer the job's `model` field selects.
#[cfg(native_lane)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DescribeModel {
    VisionCaptioner,
    JoyCaption,
    /// The Qwen3.8 27B describer (SW-F365): the Vision Captioner's describing path (same prompt,
    /// JSON-constrained decode, no thinking) over a larger model that comes in tiers.
    Qwen38,
}

#[cfg(native_lane)]
impl DescribeModel {
    /// Resolve the payload's `model`. Absent or blank means the Vision Captioner; a value that is
    /// neither describer is refused (the API refuses the same request with a 400 before queuing it).
    pub(crate) fn from_payload(model: Option<&str>) -> WorkerResult<Self> {
        match model.map(str::trim).filter(|value| !value.is_empty()) {
            None | Some(VISION_CAPTIONER_REPO) => Ok(Self::VisionCaptioner),
            Some(JOYCAPTION_REPO) => Ok(Self::JoyCaption),
            Some(QWEN38_27B_DESCRIBER_REPO) => Ok(Self::Qwen38),
            Some(other) => Err(WorkerError::InvalidPayload(format!(
                "Dataset batch describe does not support model {other:?}; use the Vision Captioner \
                 ({VISION_CAPTIONER_REPO}), JoyCaption ({JOYCAPTION_REPO}) or the Qwen3.8 27B \
                 describer ({QWEN38_27B_DESCRIBER_REPO})."
            ))),
        }
    }

    pub(crate) fn repo(self) -> &'static str {
        match self {
            Self::VisionCaptioner => VISION_CAPTIONER_REPO,
            Self::JoyCaption => JOYCAPTION_REPO,
            Self::Qwen38 => QWEN38_27B_DESCRIBER_REPO,
        }
    }
}

/// Where a Qwen3.8 27B tier's snapshot lives. Full precision (`bf16`) is the upstream repository's
/// snapshot. A quantised tier is the `<tier>/` folder of this engine's re-host repository: it was
/// quantised ahead of time, so loading it never reads or load-quantises the full-precision files.
#[cfg(native_lane)]
fn qwen38_describer_weights_dir(settings: &Settings, tier: &str) -> WorkerResult<PathBuf> {
    use sceneworks_core::dataset_batch_describe::{
        QWEN38_27B_DESCRIBER_CANDLE_REPO, QWEN38_27B_DESCRIBER_MLX_REPO,
    };
    let label = "Qwen3.8 27B describer model path";
    if tier == "bf16" {
        return resolve_app_managed_model_dir(settings, QWEN38_27B_DESCRIBER_REPO, label);
    }
    let repo = if cfg!(target_os = "macos") {
        QWEN38_27B_DESCRIBER_MLX_REPO
    } else {
        QWEN38_27B_DESCRIBER_CANDLE_REPO
    };
    let dir = resolve_app_managed_model_dir(settings, repo, label)?.join(tier);
    if dir.join("config.json").is_file() {
        Ok(dir)
    } else {
        Err(WorkerError::InvalidPayload(format!(
            "The {tier} tier of the Qwen3.8 27B describer is not installed (no snapshot at {}).",
            dir.display()
        )))
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
    /// Build `model`'s prompt for a job's optional instruction. Returns the prompt and whether the
    /// model's default instruction was used.
    pub(crate) fn for_instruction(model: DescribeModel, instruction: Option<&str>) -> (Self, bool) {
        let custom = instruction
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let is_default = custom.is_none();
        let prompt = Self {
            instruction: custom.unwrap_or_else(|| default_describe_instruction(model)),
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

    /// The one prompt a single-turn describer (JoyCaption) receives: the system text (instruction,
    /// then contract), then the user line.
    pub(crate) fn single_turn(&self) -> String {
        format!("{}\n\n{}", self.system(), self.user)
    }
}

/// `model`'s default editable describing instruction: the asset's `[SYSTEM]` block for the Vision
/// Captioner, its condensed `[JOYCAPTION_SYSTEM]` block for JoyCaption (neither holds the output
/// contract — that is the separate `[OUTPUT]` block).
#[cfg(native_lane)]
pub(crate) fn default_describe_instruction(model: DescribeModel) -> String {
    sceneworks_core::dataset_batch_describe::default_describe_instruction(
        IMAGE_BATCH_DESCRIBE_V1,
        Some(model.repo()),
    )
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
    described_fields_from(|name| object.get(name))
}

/// The lenient parse for a describer without constrained decoding (JoyCaption): reasoning blocks
/// and code fences stripped, then the FIRST complete JSON object anywhere in the reply (leading
/// prose, trailing prose, or a second object after it are ignored), with field names matched
/// ignoring ASCII case. The field rules are the strict parse's: missing or `null` is empty, any
/// other non-string is an error, all six empty is an error.
#[cfg(native_lane)]
pub(crate) fn parse_lenient_described_fields(raw: &str) -> Result<DescribedFields, String> {
    let text = crate::text_clean::strip_reasoning_and_fence(raw);
    let object = text
        .match_indices('{')
        .find_map(|(start, _)| {
            match serde_json::Deserializer::from_str(&text[start..])
                .into_iter::<Value>()
                .next()
            {
                Some(Ok(Value::Object(object))) => Some(object),
                _ => None,
            }
        })
        .ok_or_else(|| "The describer reply holds no complete JSON object.".to_owned())?;
    described_fields_from(|name| {
        object.get(name).or_else(|| {
            object
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, value)| value)
        })
    })
}

/// The six fields of a parsed reply object, each read through `field`.
#[cfg(native_lane)]
fn described_fields_from<'a>(
    field: impl Fn(&str) -> Option<&'a Value>,
) -> Result<DescribedFields, String> {
    let mut fields: [String; 6] = Default::default();
    for (slot, name) in fields.iter_mut().zip(DESCRIBE_FIELDS) {
        *slot = match field(name) {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(text)) => bounded_text(text.trim(), MAX_DESCRIBE_FIELD_CHARS),
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

    /// How many more times an image is described when its reply does not [`Self::parse`]. A
    /// generation error is never retried. The constrained Vision Captioner's reply always has the
    /// object's shape, so it is not retried.
    fn parse_retries(&self) -> usize {
        0
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

/// Loads the JoyCaption captioner: `crate::inference_runtime::load_captioner` in a job, a stub in a
/// test.
#[cfg(native_lane)]
pub(crate) type CaptionerLoader = Box<
    dyn FnOnce(&str, &gen_core::LoadSpec) -> gen_core::Result<Box<dyn gen_core::Captioner>> + Send,
>;

/// One image for the JoyCaption thread to describe, with where to send the reply.
#[cfg(native_lane)]
struct JoyCaptionCall {
    image: image::DynamicImage,
    prompt: String,
    cancel: CancelFlag,
    reply: tokio::sync::oneshot::Sender<WorkerResult<Result<String, String>>>,
}

/// The JoyCaption describer: the existing `training_caption` captioner (mlx-gen-joycaption on macOS,
/// candle-gen-joycaption off-Mac, loaded through the same `load_captioner` seam). A `Captioner` is
/// not `Send`, so one blocking thread owns it for the whole job: it loads the model on the first
/// image, describes every image the job sends it, and drops the model when the job drops this
/// describer.
#[cfg(native_lane)]
pub(crate) struct JoyCaptionDescriber {
    calls: std::sync::mpsc::Sender<JoyCaptionCall>,
}

#[cfg(native_lane)]
impl JoyCaptionDescriber {
    pub(crate) fn spawn(
        weights_dir: PathBuf,
        job_id: String,
        backend: String,
        load: CaptionerLoader,
    ) -> Self {
        let (calls, receiver) = std::sync::mpsc::channel::<JoyCaptionCall>();
        // Detached: the thread ends when `calls` is dropped (the job finished with this describer).
        drop(crate::load_progress::spawn_blocking(move || {
            joycaption_thread(receiver, weights_dir, job_id, backend, load)
        }));
        Self { calls }
    }
}

#[cfg(native_lane)]
fn joycaption_thread(
    calls: std::sync::mpsc::Receiver<JoyCaptionCall>,
    weights_dir: PathBuf,
    job_id: String,
    backend: String,
    load: CaptionerLoader,
) {
    use gen_core::{
        CaptionOptions, CaptionRequest, CaptionSampling, Image, LoadSpec, Progress, WeightsSource,
    };
    let backend = backend.to_uppercase();
    let mut load = Some(load);
    let mut captioner: Option<Box<dyn gen_core::Captioner>> = None;
    while let Ok(call) = calls.recv() {
        if let Some(loader) = load.take() {
            emit_event(
                "caption_pipeline_load_start",
                json!({ "jobId": job_id, "engine": JOYCAPTION_REPO }),
            );
            match loader(
                JOYCAPTION_REPO,
                &LoadSpec::new(WeightsSource::Dir(weights_dir.clone())),
            ) {
                Ok(model) => {
                    emit_event(
                        "caption_pipeline_load_complete",
                        json!({ "jobId": job_id, "engine": JOYCAPTION_REPO }),
                    );
                    captioner = Some(model);
                }
                Err(error) => {
                    // A model that cannot load fails the whole job; the thread ends with it.
                    let _ = call.reply.send(Err(WorkerError::Engine(format!(
                        "JoyCaption {backend} load failed: {error}"
                    ))));
                    return;
                }
            }
        }
        let Some(model) = captioner.as_ref() else {
            return;
        };
        let rgb = call.image.to_rgb8();
        // The job's cancel is a `core_llm::CancelFlag`; the captioner polls its own gen-core flag.
        // Bridge them before the call and on every progress event the captioner emits.
        let caption_cancel = gen_core::CancelFlag::new();
        if call.cancel.is_cancelled() {
            caption_cancel.cancel();
        }
        let request = CaptionRequest {
            image: Image {
                width: rgb.width(),
                height: rgb.height(),
                pixels: rgb.into_raw(),
            },
            // The gen-core contract: `custom_prompt` is the source of `prompt`.
            options: CaptionOptions {
                custom_prompt: call.prompt.clone(),
                ..CaptionOptions::default()
            },
            prompt: call.prompt,
            // The training caption job's sampling defaults, with the providers' full reply budget.
            sampling: CaptionSampling {
                temperature: 0.6,
                top_p: 0.9,
                max_new_tokens: JOYCAPTION_MAX_NEW_TOKENS,
                ..CaptionSampling::default()
            },
            cancel: caption_cancel.clone(),
            ..Default::default()
        };
        // sc-24029: bound MLX's process-global freed-buffer cache across the decode.
        let mut cache_bound = crate::mlx_decode_cache::DecodeCacheBound::mlx();
        let mut on_progress = |_: Progress| {
            cache_bound.note_event();
            if call.cancel.is_cancelled() {
                caption_cancel.cancel();
            }
        };
        let reply = match model.caption(&request, &mut on_progress) {
            Ok(output) => Ok(output.text),
            Err(gen_core::Error::Canceled) => Err("JoyCaption generation canceled.".to_owned()),
            Err(error) => Err(format!("JoyCaption {backend} generation failed: {error}")),
        };
        drop(cache_bound);
        // The job may have stopped waiting (canceled); nothing to do then.
        let _ = call.reply.send(Ok(reply));
    }
}

#[cfg(native_lane)]
impl BatchDescriber for JoyCaptionDescriber {
    fn describe(
        &self,
        image: image::DynamicImage,
        prompt: &DescribePrompt,
        cancel: CancelFlag,
    ) -> DescribeFuture {
        let (reply, answer) = tokio::sync::oneshot::channel();
        let sent = self.calls.send(JoyCaptionCall {
            image,
            prompt: prompt.single_turn(),
            cancel,
            reply,
        });
        Box::pin(async move {
            let stopped =
                || WorkerError::Engine("The JoyCaption describer stopped unexpectedly.".to_owned());
            sent.map_err(|_| stopped())?;
            answer.await.map_err(|_| stopped())?
        })
    }

    fn parse(&self, raw: &str) -> Result<DescribedFields, String> {
        parse_lenient_described_fields(raw)
    }

    fn parse_retries(&self) -> usize {
        JOYCAPTION_PARSE_RETRIES
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
    bounded_text(error, MAX_ITEM_ERROR_CHARS)
}

/// `text` cut to at most `max` characters, with an ellipsis marking a cut.
#[cfg(native_lane)]
fn bounded_text(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let mut cut: String = text.chars().take(max).collect();
    cut.push('…');
    cut
}

/// The item object appended to the job's per-item results: `item`'s JSON, or, if that would exceed
/// the store's per-item limit (not reachable with the field caps; kept so the append can never be
/// refused for size), a flagged item that keeps the id and size but no text.
#[cfg(native_lane)]
fn stored_item_json(item: &DescribedItem) -> Value {
    let value = item.to_json();
    if value.to_string().len() <= sceneworks_core::jobs_store::MAX_JOB_RESULT_ITEM_BYTES {
        return value;
    }
    DescribedItem {
        item: DescribeItem {
            relative_path: String::new(),
            ..item.item.clone()
        },
        size: item.size,
        outcome: Err("The description was too large to store.".to_owned()),
    }
    .to_json()
}

/// Append one finished image to the job's per-item results, as the worker that owns the job. A 409
/// (the job was canceled, swept or reclaimed) propagates like a rejected progress post.
#[cfg(native_lane)]
async fn append_result_item(
    api: &ApiClient,
    job_id: &str,
    position: usize,
    item: &DescribedItem,
) -> WorkerResult<()> {
    let _: Value = api
        .post_json(
            &format!("/api/v1/jobs/{job_id}/result-items"),
            &json!({
                "workerId": api.worker_id,
                "items": [{ "position": position, "item": stored_item_json(item) }],
            }),
        )
        .await?;
    Ok(())
}

/// The validated job request.
#[cfg(native_lane)]
struct BatchDescribeRequest {
    model: DescribeModel,
    /// The model tier described with: `Some` for the Qwen3.8 27B describer only.
    tier: Option<&'static str>,
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
    // The tier travels as `variant`, the request vocabulary the model-source seam already reads
    // (`requested_runtime_variant`), so the source a job is admitted against is the tier it loads.
    // A non-string value is refused, never read as "no tier": that would run the default size.
    let requested_tier = match payload.get("variant") {
        None | Some(Value::Null) => None,
        Some(Value::String(tier)) => Some(tier.as_str()),
        Some(other) => {
            return Err(WorkerError::InvalidPayload(format!(
                "Dataset batch describe variant (the model tier) must be a string, got {other}."
            )))
        }
    };
    let tier = sceneworks_core::dataset_batch_describe::resolve_describe_tier(
        Some(model.repo()),
        requested_tier,
    )
    .map_err(WorkerError::InvalidPayload)?;
    let limit =
        sceneworks_core::dataset_batch_describe::max_describe_prompt_chars(Some(model.repo()));
    let instruction = match payload.get("prompt") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) if text.trim().chars().count() > limit => {
            return Err(WorkerError::InvalidPayload(format!(
                "Dataset batch describe prompt is longer than {limit} characters for this model."
            )))
        }
        Some(Value::String(text)) => Some(text.as_str()),
        Some(other) => {
            return Err(WorkerError::InvalidPayload(format!(
                "Dataset batch describe prompt must be a string, got {other}."
            )))
        }
    };
    let (prompt, default_prompt) = DescribePrompt::for_instruction(model, instruction);
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
        tier,
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

/// The job result: the request echo plus counts. The items themselves are the job's per-item
/// results (`/result-items`), never this object, so every post that carries it stays small.
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
    if let Some(tier) = request.tier {
        result.insert("variant".to_owned(), json!(tier));
    }
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
        DescribeModel::Qwen38 => Box::new(VisionCaptionerDescriber {
            weights_dir: qwen38_describer_weights_dir(
                settings,
                request.tier.unwrap_or(
                    sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_DEFAULT_TIER,
                ),
            )?,
            job_id: job.id.clone(),
        }),
        DescribeModel::JoyCaption => Box::new(JoyCaptionDescriber::spawn(
            resolve_app_managed_model_dir(settings, JOYCAPTION_REPO, "JoyCaption model path")?,
            job.id.clone(),
            backend_label(&settings.gpu_id).to_owned(),
            Box::new(crate::inference_runtime::load_captioner),
        )),
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
                let described = describe_with_retries(
                    api,
                    settings,
                    job,
                    describer,
                    &request.prompt,
                    image,
                    (index, total),
                )
                .await?;
                let Some(outcome) = described else {
                    // This image was interrupted mid-description: it has no result to return.
                    canceled = true;
                    break;
                };
                DescribedItem {
                    item: item.clone(),
                    size: Some(size),
                    outcome,
                }
            }
        };
        // Store the item before the post that counts it, so a reader that sees `processedCount`
        // reach N can always read N items.
        append_result_item(api, &job.id, described.len(), &outcome).await?;
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

/// Describe one image, describing it again (at most [`BatchDescriber::parse_retries`] more times)
/// while its reply does not parse. A generation error is returned at once, never retried. Returns
/// `None` when the job was canceled during an attempt or before a retry.
#[cfg(native_lane)]
async fn describe_with_retries(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    describer: &dyn BatchDescriber,
    prompt: &DescribePrompt,
    image: image::DynamicImage,
    (index, total): (usize, usize),
) -> WorkerResult<Option<Result<DescribedFields, String>>> {
    let retries = describer.parse_retries();
    let mut image = Some(image);
    let mut attempt = 0;
    loop {
        // Keep a copy only while another attempt may need the image.
        let input = if attempt < retries {
            image.clone()
        } else {
            image.take()
        }
        .expect("the image is kept until the last attempt");
        let cancel = CancelFlag::new();
        let generation = describer.describe(input, prompt, cancel.clone());
        let (was_canceled, reply) =
            StreamSupervisor::new(cancel, generation, progress_report_interval(settings), None)
                .wait(api, settings, &job.id)
                .await?;
        if was_canceled {
            return Ok(None);
        }
        let raw = match reply? {
            Ok(raw) => raw,
            Err(error) => return Ok(Some(Err(error))),
        };
        match describer.parse(&raw) {
            Ok(fields) => return Ok(Some(Ok(fields))),
            Err(error) if attempt >= retries => {
                return Ok(Some(Err(if attempt == 0 {
                    error
                } else {
                    format!("{error} (after {} attempts)", attempt + 1)
                })))
            }
            Err(_) => {}
        }
        attempt += 1;
        // Honour a cancel before spending another attempt on this image.
        if cancel_requested_peek(api, &job.id).await {
            return Ok(None);
        }
        update_job(
            api,
            &job.id,
            progress_payload_for_backend(
                JobStatus::Running,
                ProgressStage::Running,
                image_progress(index, total),
                &format!(
                    "Describing image {} of {total} again (attempt {}): the reply was not the \
                     six-field object.",
                    index + 1,
                    attempt + 1
                ),
                None,
                backend_label(&settings.gpu_id),
            ),
        )
        .await?;
    }
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
