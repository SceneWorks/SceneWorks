//! Dataset prompt-batch describe job (SW-7082, feature SW-F362 "Create Prompt Batch from Dataset").
//!
//! The `dataset_batch_describe` job runs a vision-language describer over every image of a training
//! dataset (or a subset of its item ids) and returns, per image in dataset order, seven description
//! fields (`description`, `pose`, `style`, `camera`, `composition`, `lighting`, `environment`) plus
//! the image's pixel width and height. Training Studio assembles the prompt-batch text from those
//! fields in the web app; this job only describes.
//!
//! **Pose.** The Vision Captioner writes `pose` as one object per person (a `label` and the eleven
//! [`POSE_SLOTS`]); this job flattens it into one stored line per person,
//! `<label> — Body line: …; Stance: …; …`, so every stored field is text. JoyCaption writes `pose`
//! as plain text, stored as written.
//!
//! **The describing prompt.** The user edits only the instruction text (the `[SYSTEM]` block of
//! `image_batch_describe_v1.txt` is the default). The fixed JSON output contract (the asset's
//! `[OUTPUT]` block, `[JOYCAPTION_OUTPUT]` for JoyCaption) is ALWAYS appended by this job after the
//! instruction, so no edit (and no preset saved before a field was added) can remove a field the
//! parser reads.
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
//! error, reply that is not the description object) is returned flagged `failed` with its error rather
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
//!
//! **Fit to (SW-7163).** With a `fitTo` target the Vision Captioner also condenses each described
//! image into one prompt within the target image model's prompt token limit (see
//! [`condense`]); a `condenseOnly` job condenses rows already described (their seven fields ride
//! in the payload) without opening an image or describing again. JoyCaption does not condense.

use super::*;

#[cfg(native_lane)]
#[path = "batch_describe_condense.rs"]
mod condense;
#[cfg(native_lane)]
use condense::{condense_fields, CondenseFuture, CondenseReply, Condensed, FitTarget};

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

/// Reply budget. Seven fields of detailed prose; a reply that reaches the cap is cut off mid-object
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
/// makes the bound independent of the tokenizer, so seven fields, each JSON-escaped at worst six
/// bytes per character, stay well under the store's per-item limit (`MAX_JOB_RESULT_ITEM_BYTES`).
#[cfg(native_lane)]
pub(crate) const MAX_DESCRIBE_FIELD_CHARS: usize =
    sceneworks_core::dataset_batch_describe::MAX_DESCRIBE_FIELD_CHARS;

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

    /// The speculative option this model's describe and condense requests carry. The Qwen3.8
    /// 27B asks for `auto`: its native MTP head runs, and the engine's monitor demotes it to plain
    /// decoding when it measures slower (on CUDA the q8 tier keeps it — t=0.4 + JSON 50.4 vs 39.4
    /// tok/s plain — and the q4 tier drops it under the JSON constraint). `None` keeps the
    /// backend's default for the other describers.
    pub(crate) fn speculative(self) -> Option<gen_core::core_llm::Speculative> {
        match self {
            Self::Qwen38 => Some(gen_core::core_llm::Speculative::Auto),
            Self::VisionCaptioner | Self::JoyCaption => None,
        }
    }

    /// The name a failure message gives this model.
    pub(crate) fn display_name(self) -> &'static str {
        match self {
            Self::VisionCaptioner => "Vision Captioner",
            Self::JoyCaption => "JoyCaption",
            Self::Qwen38 => "Qwen3.8 27B describer",
        }
    }

    /// The asset section holding this model's fixed output contract: the Vision Captioner's asks
    /// for pose as named slots, JoyCaption's (shorter, to fit its prompt limit) as plain text. The
    /// Qwen3.8 27B describer is a Vision Captioner-class describer and uses the slot contract.
    pub(crate) fn output_section(self) -> &'static str {
        match self {
            Self::VisionCaptioner | Self::Qwen38 => "OUTPUT",
            Self::JoyCaption => "JOYCAPTION_OUTPUT",
        }
    }
}

/// Where a Qwen3.8 27B tier's snapshot lives. Full precision (`bf16`) is the upstream repository's
/// snapshot. A quantised tier is the `<tier>/` folder of this engine's re-host repository: it was
/// quantised ahead of time, so loading it never reads or load-quantises the full-precision files.
#[cfg(native_lane)]
pub(crate) fn qwen38_describer_weights_dir(
    settings: &Settings,
    tier: &str,
) -> WorkerResult<PathBuf> {
    use sceneworks_core::dataset_batch_describe::{
        QWEN38_27B_DESCRIBER_CANDLE_REPO, QWEN38_27B_DESCRIBER_MLX_REPO,
    };
    let label = "Qwen3.8 27B describer model path";
    let not_installed = || {
        WorkerError::InvalidPayload(format!(
            "The {tier} tier of the Qwen3.8 27B describer is not installed. Download that tier \
             in the Model Manager, or choose a tier that is installed."
        ))
    };
    let repo = if tier == "bf16" {
        QWEN38_27B_DESCRIBER_REPO
    } else if cfg!(target_os = "macos") {
        QWEN38_27B_DESCRIBER_MLX_REPO
    } else {
        QWEN38_27B_DESCRIBER_CANDLE_REPO
    };
    let snapshot =
        resolve_app_managed_model_dir(settings, repo, label).map_err(|_| not_installed())?;
    // Installed means every file the catalog row pins is in this snapshot, the same rule the API's
    // install state and the job's tier check apply: a cancelled download leaves a folder with
    // some of them, never an install.
    let files = sceneworks_core::dataset_batch_describe::qwen38_tier_files(repo, tier);
    if files.is_empty() || !qwen38_snapshot_holds(&snapshot, &files) {
        return Err(not_installed());
    }
    Ok(if tier == "bf16" {
        snapshot
    } else {
        snapshot.join(tier)
    })
}

/// Whether `snapshot` holds every file in `files`. A `model-*.safetensors` shard glob is satisfied
/// by the shards the snapshot's `model.safetensors.index.json` names, every one of them present.
#[cfg(native_lane)]
fn qwen38_snapshot_holds(snapshot: &Path, files: &[String]) -> bool {
    files.iter().all(|file| {
        if !file.contains('*') {
            return snapshot.join(file).is_file();
        }
        let Some(shards) = std::fs::read_to_string(snapshot.join("model.safetensors.index.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|index| {
                index.get("weight_map")?.as_object().map(|map| {
                    map.values()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect::<std::collections::BTreeSet<_>>()
                })
            })
        else {
            return false;
        };
        !shards.is_empty() && shards.iter().all(|shard| snapshot.join(shard).is_file())
    })
}

/// The catalog's pinned snapshot of a Qwen3.8 27B tier, as [`qwen38_describer_weights_dir`]
/// resolves it on a platform whose quantised tiers come from `rehost_repo`.
#[cfg(all(test, native_lane))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Qwen38PinnedSnapshot {
    pub(crate) repo: String,
    pub(crate) revision: String,
    /// The weights folder relative to the hub cache:
    /// `models--<repo>/snapshots/<revision>`, then `<tier>/` for a quantised tier.
    pub(crate) relative_dir: PathBuf,
}

/// Read a tier's pinned snapshot from the catalog `entry`'s download rows: full precision (`bf16`)
/// is the upstream repository's row at its root, a quantised tier is `rehost_repo`'s row for that
/// variant, in its `<tier>/` folder. `None` when the catalog has no such row.
#[cfg(all(test, native_lane))]
pub(crate) fn qwen38_pinned_snapshot(
    entry: &JsonObject,
    tier: &str,
    rehost_repo: &str,
) -> Option<Qwen38PinnedSnapshot> {
    let repo = if tier == "bf16" {
        QWEN38_27B_DESCRIBER_REPO
    } else {
        rehost_repo
    };
    let revision = entry
        .get("downloads")?
        .as_array()?
        .iter()
        .find(|row| {
            row.get("repo").and_then(Value::as_str) == Some(repo)
                && row.get("variant").and_then(Value::as_str) == Some(tier)
        })?
        .get("revision")?
        .as_str()?
        .to_owned();
    let mut relative_dir = PathBuf::from(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(&revision);
    if tier != "bf16" {
        relative_dir.push(tier);
    }
    Some(Qwen38PinnedSnapshot {
        repo: repo.to_owned(),
        revision,
        relative_dir,
    })
}

/// The memory (GB) a Qwen3.8 27B tier needs on `lane` (`mlx` or `candle`): that lane's
/// `minMemoryGbByTier[tier]` floor in the catalog entry the job carries, else the lane's
/// `minMemoryGb`. `None` when the job carries no such entry or floor (then nothing is refused).
/// The MLX lane reads it; the Candle lane reads its floor through the CUDA fit gate instead.
#[cfg(native_lane)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn qwen38_tier_floor_gb(payload: &JsonObject, lane: &str, tier: &str) -> Option<f64> {
    let block = qwen38_catalog_entry(payload)?.get(lane)?;
    block
        .get("minMemoryGbByTier")
        .and_then(|floors| floors.get(tier))
        .and_then(Value::as_f64)
        .or_else(|| block.get("minMemoryGb").and_then(Value::as_f64))
}

/// The Qwen3.8 27B catalog entry the API attached to the job.
#[cfg(native_lane)]
fn qwen38_catalog_entry(payload: &JsonObject) -> Option<&JsonObject> {
    use sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_MODEL_ID;
    payload
        .get("modelManifestEntries")?
        .as_array()?
        .iter()
        .filter_map(Value::as_object)
        .find(|entry| {
            entry.get("id").and_then(Value::as_str) == Some(QWEN38_27B_DESCRIBER_MODEL_ID)
        })
}

/// Refuse a tier whose memory floor exceeds what this worker has, with the two numbers. A missing
/// floor or budget refuses nothing, like every other fit gate.
#[cfg(native_lane)]
pub(crate) fn qwen38_fit_refusal(
    tier: &str,
    needed_gb: Option<f64>,
    available_gb: Option<f64>,
) -> WorkerResult<()> {
    match crate::fit_gate::fit_decision(needed_gb, available_gb) {
        crate::fit_gate::FitDecision::TooBig {
            needed_gb,
            available_gb,
        } => Err(WorkerError::InvalidPayload(format!(
            "The {tier} tier of the Qwen3.8 27B describer needs {needed_gb:.1} GB of memory, but \
             this worker has {available_gb:.1} GB for it. Choose a smaller tier."
        ))),
        _ => Ok(()),
    }
}

/// The Candle decision once the refine cache has answered: a tier already resident there is reused
/// and needs no new VRAM, so only a cold load is checked against the free VRAM it will see (with
/// any other cached refiner's weights credited, since the load drops it first). Pure, so it runs
/// without a GPU.
#[cfg(native_lane)]
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn qwen38_cached_fit_refusal(
    access: crate::cache_thread::CacheAccess,
    tier: &str,
    needed_gb: Option<f64>,
    available_gb: Option<f64>,
) -> WorkerResult<()> {
    match access {
        crate::cache_thread::CacheAccess::Warm => Ok(()),
        crate::cache_thread::CacheAccess::Cold => qwen38_fit_refusal(tier, needed_gb, available_gb),
    }
}

/// The free VRAM (GB) a cold Candle load is checked against. The cached image/video generator is
/// evicted first (as the depth, training and image admissions do), then `probe_budget` reads the
/// card, and `refiner_credit_gb` (the weights of a different cached refiner, which the load's own
/// evict-before-load drops) is added, capped at the card's total. The refiner itself is not
/// evicted here, so a job this check refuses leaves it resident. A warm tier is reused and needs
/// no reading, so nothing is evicted for it. The evict and the probe are injected so the order is
/// testable without a GPU.
#[cfg(native_lane)]
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) async fn qwen38_free_gb_after_eviction<Evict, Probe>(
    access: crate::cache_thread::CacheAccess,
    refiner_credit_gb: f64,
    evict_generator: impl FnOnce() -> Evict,
    probe_budget: impl FnOnce() -> Probe,
) -> WorkerResult<Option<f64>>
where
    Evict: std::future::Future<Output = WorkerResult<bool>>,
    Probe: std::future::Future<Output = WorkerResult<Option<crate::fit_gate::VramBudget>>>,
{
    if access == crate::cache_thread::CacheAccess::Warm {
        return Ok(None);
    }
    if evict_generator().await? {
        tracing::info!(
            event = "qwen38_describer_evicted_cached_generator",
            "Qwen3.8 27B describer admission: evicted the cached generator before reading free VRAM"
        );
    }
    Ok(probe_budget()
        .await?
        .map(|budget| crate::fit_gate::with_reclaimable(budget, refiner_credit_gb).free_gb))
}

/// The weights (GB) a cached refiner at `source` holds: the bytes of its safetensors payload, what
/// dropping it releases. 0 when it cannot be read, so an unreadable entry is never over-credited.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn cached_refiner_weights_gb(source: &Path) -> f64 {
    gen_core::core_llm::checkpoint_payload_bytes(source)
        .map(|bytes| bytes as f64 / (1u64 << 30) as f64)
        .unwrap_or(0.0)
}

/// Whether this worker can hold the job's tier, checked before any image.
///
/// MLX: the tier's floor against TOTAL unified memory (the catalog's MLX floors are whole-machine
/// sizes). A model held in the refine cache or the generator cache does not lower that total, so
/// nothing is evicted here: eviction could not change the decision.
///
/// Candle: the CUDA fit gate's own floor reader against FREE VRAM, which a cached model does lower.
/// So the refine cache is asked first, without evicting: the same tier resident there is reused
/// (admitted with no check); a different cached refiner stays resident and its weights are
/// credited, because the load's own evict-before-load drops it. A cold load evicts the cached
/// image/video generator before free VRAM is read fresh ([`qwen38_free_gb_after_eviction`]).
#[cfg(native_lane)]
async fn qwen38_memory_admission(
    settings: &Settings,
    payload: &JsonObject,
    tier: &str,
    weights_dir: &Path,
) -> WorkerResult<()> {
    #[cfg(target_os = "macos")]
    {
        let _ = (settings, weights_dir);
        qwen38_fit_refusal(
            tier,
            qwen38_tier_floor_gb(payload, "mlx", tier),
            crate::fit_gate::unified_total_gb(),
        )
    }
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        let (spec, requirements) = vision_describer_load(weights_dir);
        let (access, cached_refiner) =
            crate::refine_model_cache::refiner_admission_access(&spec, &requirements).await?;
        let needed_gb = qwen38_catalog_entry(payload)
            .and_then(|entry| crate::vram_gate::predicted_peak_gb(entry, tier));
        let credit_gb = cached_refiner
            .as_deref()
            .map_or(0.0, cached_refiner_weights_gb);
        let gpu_id = settings.gpu_id.clone();
        let available_gb = qwen38_free_gb_after_eviction(
            access,
            credit_gb,
            crate::generator_cache::evict_cached_generator,
            || async move {
                crate::load_progress::spawn_blocking(move || {
                    crate::vram_gate::live_capped_budget_fresh_blocking(&gpu_id)
                })
                .await
                .map_err(|error| task_join_error("describer VRAM probe", error))
            },
        )
        .await?;
        qwen38_cached_fit_refusal(access, tier, needed_gb, available_gb)
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
                model.output_section(),
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

/// The parsed fields of one described image, in [`DESCRIBE_FIELDS`] order (each may be empty).
#[cfg(native_lane)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DescribedFields {
    pub(crate) fields: [String; DESCRIBE_FIELDS.len()],
    /// Set when the reply was cut off at the reply budget and only part of it was kept (see
    /// [`recover_cut_off_reply`]): what the cut cost, shown on the image's row.
    pub(crate) notice: Option<String>,
}

/// The named slots of one pose entry, in order: the key the Vision Captioner writes and the label
/// of its clause in the stored line.
#[cfg(native_lane)]
pub(crate) const POSE_SLOTS: [(&str, &str); 11] = [
    ("body_line", "Body line"),
    ("stance", "Stance"),
    ("facing", "Facing"),
    ("torso", "Torso"),
    ("head", "Head"),
    ("left_arm", "Left arm"),
    ("right_arm", "Right arm"),
    ("left_leg", "Left leg"),
    ("right_leg", "Right leg"),
    ("depth", "Depth"),
    ("placement", "Placement"),
];

/// What a pose slot the model left missing or blank is stored as. The prompt tells the model to
/// write "hidden" for a hidden limb; a missing slot is a model error for that slot only, so the
/// image keeps the rest of its pose instead of failing.
#[cfg(native_lane)]
pub(crate) const POSE_SLOT_NOT_STATED: &str = "not stated";

/// Parse a describer reply into the fields: the outermost JSON object (reasoning blocks and code
/// fences stripped). A missing or `null` field is an empty string; `pose` may also be a list of
/// per-person slot objects (see [`flatten_pose`]); a field of any other non-string type, a reply
/// that is not an object, or an object with every field empty is an error (the image is flagged,
/// not dropped).
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
/// ignoring ASCII case. The field rules are the strict parse's: missing or `null` is empty, `pose`
/// may be plain text or slot objects, any other non-string is an error, all empty is an error.
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

/// The fields of a parsed reply object, each read through `field`.
#[cfg(native_lane)]
fn described_fields_from<'a>(
    field: impl Fn(&str) -> Option<&'a Value>,
) -> Result<DescribedFields, String> {
    let mut fields: [String; DESCRIBE_FIELDS.len()] = Default::default();
    for (slot, name) in fields.iter_mut().zip(DESCRIBE_FIELDS) {
        *slot = match field(name) {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(text)) => bounded_text(text.trim(), MAX_DESCRIBE_FIELD_CHARS),
            Some(value @ (Value::Array(_) | Value::Object(_))) if name == "pose" => {
                bounded_text(&flatten_pose(value)?, MAX_DESCRIBE_FIELD_CHARS)
            }
            Some(other) => {
                return Err(format!(
                    "The describer reply's {name:?} field is not a string ({other})."
                ))
            }
        };
    }
    if fields.iter().all(String::is_empty) {
        return Err("The describer reply has none of the description fields.".to_owned());
    }
    Ok(DescribedFields {
        fields,
        notice: None,
    })
}

/// The stored text of a structured `pose`: one line per entry, in reply order. An entry object is
/// `<label> — Body line: …; Stance: …; …` with every [`POSE_SLOTS`] clause (a missing, null or
/// blank slot reads [`POSE_SLOT_NOT_STATED`]); an entry without a label (an animal) is the clauses
/// alone. The list form the contract asks for, a lone entry object, and a map keyed by label
/// (`{"woman_1": {slots…}}`, each key the label of its entry) are all accepted. A text entry is
/// kept as written, and an empty entry is skipped, so an empty list is an empty field.
#[cfg(native_lane)]
pub(crate) fn flatten_pose(value: &Value) -> Result<String, String> {
    let entries: Vec<(Option<&str>, &Value)> = match value {
        Value::Array(entries) => entries.iter().map(|entry| (None, entry)).collect(),
        Value::Object(map) if is_pose_map(map) => map
            .iter()
            .map(|(label, entry)| (Some(label.as_str()), entry))
            .collect(),
        entry => vec![(None, entry)],
    };
    let mut lines = Vec::new();
    for (label, entry) in entries {
        let line = match entry {
            Value::Null => String::new(),
            Value::String(text) => match label.map(one_line).filter(|label| !label.is_empty()) {
                Some(label) if !text.trim().is_empty() => format!("{label} — {}", one_line(text)),
                _ => one_line(text),
            },
            Value::Object(object) => pose_line(object, label),
            other => {
                return Err(format!(
                    "The describer reply's \"pose\" entry is not an object ({other})."
                ))
            }
        };
        if !line.is_empty() {
            lines.push(line);
        }
    }
    Ok(lines.join("\n"))
}

/// Whether a `pose` object is a map keyed by label rather than one entry (see
/// [`is_lone_pose_entry`]).
#[cfg(native_lane)]
fn is_pose_map(map: &JsonObject) -> bool {
    !map.is_empty()
        && !is_lone_pose_entry(
            map.iter()
                .map(|(key, value)| (key.as_str(), Some(value.is_object()))),
        )
}

/// Whether a `pose` object's members make it ONE entry rather than a map keyed by label: it has a
/// `label` member, or a slot-named member whose value is not an object. A map's values are the
/// entries (objects), so a label that happens to be a slot name still reads as a map key. Each
/// member is its key and whether its value is an object (`None` when the value is not known yet,
/// as for a member cut off before its value began).
#[cfg(native_lane)]
fn is_lone_pose_entry<'a>(members: impl IntoIterator<Item = (&'a str, Option<bool>)>) -> bool {
    members.into_iter().any(|(key, is_object)| {
        key.eq_ignore_ascii_case("label")
            || (is_object == Some(false)
                && POSE_SLOTS
                    .iter()
                    .any(|(slot, _)| key.eq_ignore_ascii_case(slot)))
    })
}

/// One person's stored pose line; empty when the entry has no label and no slot text at all.
/// `map_label` is the entry's key when the pose came as a map keyed by label.
#[cfg(native_lane)]
fn pose_line(entry: &JsonObject, map_label: Option<&str>) -> String {
    let text = |key: &str| {
        entry
            .get(key)
            .or_else(|| {
                entry
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(key))
                    .map(|(_, value)| value)
            })
            .map(slot_text)
            .unwrap_or_default()
    };
    let slots: Vec<(&str, String)> = POSE_SLOTS
        .iter()
        .map(|(key, name)| (*name, text(key)))
        .collect();
    let mut label = text("label");
    if label.is_empty() {
        label = map_label.map(one_line).unwrap_or_default();
    }
    if label.is_empty() && slots.iter().all(|(_, value)| value.is_empty()) {
        return String::new();
    }
    let clauses = slots
        .iter()
        .map(|(name, value)| {
            let value = if value.is_empty() {
                POSE_SLOT_NOT_STATED
            } else {
                value
            };
            format!("{name}: {value}")
        })
        .collect::<Vec<_>>()
        .join("; ");
    if label.is_empty() {
        clauses
    } else {
        format!("{label} — {clauses}")
    }
}

/// One slot's text: a string as written; a list's items joined with ", "; a nested object as
/// `key: value` clauses joined with ", "; a number or boolean as written. Empty for null or blank.
#[cfg(native_lane)]
fn slot_text(value: &Value) -> String {
    let joined = |parts: Vec<String>| {
        parts
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(", ")
    };
    match value {
        Value::Null => String::new(),
        Value::String(text) => one_line(text),
        Value::Bool(_) | Value::Number(_) => value.to_string(),
        Value::Array(items) => joined(items.iter().map(slot_text).collect()),
        Value::Object(members) => joined(
            members
                .iter()
                .map(|(key, value)| {
                    let text = slot_text(value);
                    if text.is_empty() {
                        text
                    } else {
                        format!("{}: {text}", one_line(&key.replace('_', " ")))
                    }
                })
                .collect(),
        ),
    }
}

/// `text` with every whitespace run (newlines included) collapsed to one space, so a slot can
/// never break the one-line-per-person layout.
#[cfg(native_lane)]
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The error of a reply cut off at the reply budget before its description was complete.
#[cfg(native_lane)]
pub(crate) const CUT_OFF_BEFORE_DESCRIPTION: &str =
    "The reply reached the length limit before the description was complete.";

/// What survives of a reply that did not parse because it was cut off at the reply budget: the
/// JSON object it started ends before it closes. Every complete top-level field is kept, and a cut
/// `pose` keeps only its complete entries (an entry cut part-way is dropped, never half-kept). The
/// image is described, with a [`DescribedFields::notice`] saying what the cut cost; a reply cut
/// before `description` is complete is a failure with [`CUT_OFF_BEFORE_DESCRIPTION`]. `None` when
/// the reply is not a cut-off object (it closes, or is malformed), so its own parse error stands.
#[cfg(native_lane)]
pub(crate) fn recover_cut_off_reply(raw: &str) -> Option<Result<DescribedFields, String>> {
    let text = crate::text_clean::strip_reasoning_and_fence(raw);
    let open = text
        .match_indices('{')
        .find_map(|(start, _)| open_members(&text[start + 1..], true))?;
    let mut object = JsonObject::new();
    for (key, value) in open.complete {
        object.insert(key.unwrap_or_default(), value);
    }
    let has =
        |object: &JsonObject, name: &str| object.keys().any(|key| key.eq_ignore_ascii_case(name));
    // A pose cut part-way keeps its complete entries; `pose_cut` is how many.
    let mut pose_cut: Option<usize> = None;
    if let Some(cut) = open.cut.filter(|cut| {
        cut.key
            .as_deref()
            .is_some_and(|key| key.eq_ignore_ascii_case("pose"))
    }) {
        let partial = cut.partial.trim_start();
        let is_object = partial.starts_with('{');
        let mut entries = if is_object || partial.starts_with('[') {
            open_members(&partial[1..], is_object)
                .map(|inner| {
                    // An object that never closed is either one entry cut part-way (dropped
                    // whole, never half-kept) or a map keyed by label (its complete entries kept,
                    // the cut last one dropped).
                    if is_object {
                        let cut_member = inner.cut.as_ref().and_then(|member| {
                            let started = member.partial.trim_start();
                            member.key.as_deref().map(|key| {
                                (key, (!started.is_empty()).then(|| started.starts_with('{')))
                            })
                        });
                        let members = inner
                            .complete
                            .iter()
                            .map(|(key, value)| {
                                (key.as_deref().unwrap_or_default(), Some(value.is_object()))
                            })
                            .chain(cut_member);
                        if is_lone_pose_entry(members) {
                            return Vec::new();
                        }
                    }
                    inner.complete
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        if is_object {
            // A map entry that is not an entry object or text cannot be a person.
            entries.retain(|(_, value)| value.is_object() || value.is_string());
        }
        let is_map = is_object;
        pose_cut = Some(entries.len());
        let value = if is_map {
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key.unwrap_or_default(), value))
                    .collect(),
            )
        } else {
            Value::Array(entries.into_iter().map(|(_, value)| value).collect())
        };
        object.insert("pose".to_owned(), value);
    }
    if !has(&object, "description") {
        return Some(Err(CUT_OFF_BEFORE_DESCRIPTION.to_owned()));
    }
    let mut parts = Vec::new();
    match pose_cut {
        Some(0) => parts.push(
            "the pose was cut off before any person was complete, so it is left out".to_owned(),
        ),
        Some(count) => parts.push(format!(
            "the pose covers only the first {count} {} in this image",
            if count == 1 { "person" } else { "people" }
        )),
        None if !has(&object, "pose") => parts.push("the pose is missing".to_owned()),
        None => {}
    }
    let missing: Vec<&str> = DESCRIBE_FIELDS
        .iter()
        .copied()
        .filter(|field| *field != "pose" && !has(&object, field))
        .collect();
    if missing.is_empty() {
        if !parts.is_empty() {
            parts.push("the other fields are complete".to_owned());
        }
    } else {
        parts.push(format!(
            "{} {} cut off",
            missing.join(", "),
            if missing.len() == 1 { "was" } else { "were" }
        ));
    }
    let notice = (!parts.is_empty())
        .then(|| format!("The reply reached the length limit: {}.", parts.join("; ")));
    Some(
        described_fields_from(|name| {
            object.get(name).or_else(|| {
                object
                    .iter()
                    .find(|(key, _)| key.eq_ignore_ascii_case(name))
                    .map(|(_, value)| value)
            })
        })
        .map(|fields| DescribedFields { notice, ..fields }),
    )
}

/// The members of a JSON container that the text ends inside: every complete member (with its key
/// for an object) and the member cut off part-way, if any.
#[cfg(native_lane)]
struct OpenMembers<'a> {
    complete: Vec<(Option<String>, Value)>,
    cut: Option<CutMember<'a>>,
}

/// A member the text ends inside: its key (when it got that far) and the text of its value so far.
#[cfg(native_lane)]
struct CutMember<'a> {
    key: Option<String>,
    partial: &'a str,
}

/// Read a container from just after its opening bracket (`keyed` for an object). `None` when the
/// container closes or the text is malformed: only a container the text ends inside is cut off.
#[cfg(native_lane)]
fn open_members(text: &str, keyed: bool) -> Option<OpenMembers<'_>> {
    let skip_whitespace = |at: usize| text.len() - text[at..].trim_start().len();
    let mut complete = Vec::new();
    let mut at = 0;
    loop {
        at = skip_whitespace(at);
        let rest = &text[at..];
        if rest.is_empty() {
            return Some(OpenMembers {
                complete,
                cut: None,
            });
        }
        if rest.starts_with(',') {
            at += 1;
            continue;
        }
        if rest.starts_with(['}', ']']) {
            return None;
        }
        let key = if keyed {
            let key = match next_json_value(rest) {
                NextValue::Complete(Value::String(key), length) => {
                    at += length;
                    key
                }
                NextValue::CutOff => {
                    let cut = Some(CutMember {
                        key: None,
                        partial: "",
                    });
                    return Some(OpenMembers { complete, cut });
                }
                _ => return None,
            };
            at = skip_whitespace(at);
            if text[at..].is_empty() {
                let cut = Some(CutMember {
                    key: Some(key),
                    partial: "",
                });
                return Some(OpenMembers { complete, cut });
            }
            if !text[at..].starts_with(':') {
                return None;
            }
            at = skip_whitespace(at + 1);
            Some(key)
        } else {
            None
        };
        match next_json_value(&text[at..]) {
            NextValue::Complete(value, length) => {
                complete.push((key, value));
                at += length;
            }
            NextValue::CutOff => {
                let cut = Some(CutMember {
                    key,
                    partial: &text[at..],
                });
                return Some(OpenMembers { complete, cut });
            }
            NextValue::Malformed => return None,
        }
    }
}

#[cfg(native_lane)]
enum NextValue {
    /// A complete value and the bytes it spans.
    Complete(Value, usize),
    /// The text ends inside (or before) the value.
    CutOff,
    Malformed,
}

/// The JSON value at the start of `text`.
#[cfg(native_lane)]
fn next_json_value(text: &str) -> NextValue {
    let mut stream = serde_json::Deserializer::from_str(text).into_iter::<Value>();
    match stream.next() {
        Some(Ok(value)) => NextValue::Complete(value, stream.byte_offset()),
        None => NextValue::CutOff,
        Some(Err(error)) if error.is_eof() => NextValue::CutOff,
        Some(Err(_)) => NextValue::Malformed,
    }
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

    /// Turn a raw reply into the description fields. The default is the strict JSON-object parse the
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

    /// What survives of a reply that still does not parse after the last retry because it was cut
    /// off at the reply budget ([`recover_cut_off_reply`]); `None` keeps the parse error.
    fn recover(&self, raw: &str) -> Option<Result<DescribedFields, String>> {
        recover_cut_off_reply(raw)
    }

    /// Whether this describer runs the "Fit to" condensing step ([`Self::condense`]). JoyCaption
    /// has no text-only path, so its rows are never condensed.
    fn condenses(&self) -> bool {
        false
    }

    /// One text-only generation with the loaded model: `system` is the condensing instruction,
    /// `user` the fields. Only called when [`Self::condenses`].
    fn condense(&self, _system: String, _user: String, _cancel: CancelFlag) -> CondenseFuture {
        Box::pin(async { Ok(Err("This describer does not condense.".to_owned())) })
    }
}

/// The Vision Captioner (Qwen3-VL) describer: a JSON-constrained decode on the resident refine-model
/// cache, the model loaded once and reused across the dataset's images.
#[cfg(native_lane)]
pub(crate) struct VisionCaptionerDescriber {
    /// Which model is loaded: the Vision Captioner or the Qwen3.8 27B describer. Its name leads the
    /// load and generation failure messages.
    pub(crate) model: DescribeModel,
    pub(crate) weights_dir: PathBuf,
    pub(crate) job_id: String,
}

/// One `describe_generation_stats` event per generation: what the decode cost and which path it
/// ran (speculative proposer and acceptance, CUDA-graph replays), so a slow batch can be read from
/// the worker log rather than guessed at.
#[cfg(native_lane)]
fn emit_generation_stats(job_id: &str, phase: &str, output: &gen_core::core_llm::TextLlmOutput) {
    let timings = output.timings.as_ref();
    let decode_seconds = timings.map(|t| t.decode.as_secs_f64());
    let generated = output.usage.generated_tokens;
    let decode = output.decode.as_ref();
    crate::emit_event(
        "describe_generation_stats",
        json!({
            "jobId": job_id,
            "phase": phase,
            "promptTokens": output.usage.prompt_tokens,
            "generatedTokens": generated,
            "prefillSeconds": timings.map(|t| t.prefill.as_secs_f64()),
            "decodeSeconds": decode_seconds,
            "decodeTokensPerSecond": decode_seconds
                .filter(|seconds| *seconds > 0.0)
                .map(|seconds| f64::from(generated) / seconds),
            "proposer": decode.map(|d| format!("{:?}", d.proposer)),
            "draftTokens": decode.and_then(|d| d.draft_tokens),
            "proposedTokens": decode.map(|d| d.proposed_tokens),
            "acceptedTokens": decode.map(|d| d.accepted_tokens),
            "targetForwards": decode.map(|d| d.target_forwards),
            "sampler": decode.map(|d| d.sampler.clone()),
            "cudaGraphsReplayed": decode.map(|d| d.cuda_graphs.replayed),
            "cudaGraphsEager": decode.map(|d| d.cuda_graphs.eager),
            "fallbacks": decode.map(|d| d.fallbacks.clone()).unwrap_or_default(),
        }),
    );
}

/// The refine-cache load a Vision Captioner-class describer makes for `weights_dir`: its spec and
/// requirements, which together are the cache key. Shared with the 27B's memory check so that
/// check asks the cache about exactly the model the describe call loads.
#[cfg(native_lane)]
pub(crate) fn vision_describer_load(
    weights_dir: &Path,
) -> (
    gen_core::core_llm::LoadSpec,
    gen_core::core_llm::ModelRequirements,
) {
    use gen_core::core_llm::{Constraint, ModelRequirements};
    let spec = gen_core::core_llm::LoadSpec {
        source: weights_dir.to_string_lossy().into_owned(),
        // A snapshot directory carries its own projector (see `catalog_semantic_jobs`).
        ..Default::default()
    };
    // Resolve on the JSON constraint alone, exactly like the `image_caption` task: no provider
    // statically advertises vision + Json for a Qwen-VL snapshot; `mlx-llama`/`candle-llama`
    // flip to vision at load (see `prompt_refine_jobs`).
    (
        spec,
        ModelRequirements::default().with_constraint(Constraint::Json),
    )
}

#[cfg(native_lane)]
impl VisionCaptionerDescriber {
    /// The cache key both generations use: the condensing step must hit the model the describe
    /// call loaded, so its spec and requirements are the describe call's exactly. The Qwen3.8 27B
    /// memory admission asks the refine cache about the same key ([`vision_describer_load`]).
    fn load_key(
        &self,
    ) -> (
        gen_core::core_llm::LoadSpec,
        gen_core::core_llm::ModelRequirements,
    ) {
        vision_describer_load(&self.weights_dir)
    }
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
            Content, Message, Role, Sampling, StreamEvent, TextLlmRequest, ThinkingMode,
        };
        let (spec, requirements) = self.load_key();
        let system = prompt.system();
        let user = prompt.user.clone();
        let job_id = self.job_id.clone();
        let kv_compression = crate::llm_kv_cache::worker_default_policy();
        let name = self.model.display_name();
        let speculative = self.model.speculative();
        Box::pin(crate::refine_model_cache::with_cached_refiner(
            spec,
            requirements,
            format!("{name} load failed"),
            move |model| -> WorkerResult<Result<String, String>> {
                let image = match crate::prompt_refine_jobs::vision_image_ref(image) {
                    Ok(image) => image,
                    Err(error) => {
                        return Ok(Err(format!("The image could not be prepared: {error}")))
                    }
                };
                // Thinking is always off: the 8B Vision Captioner has no thinking mode, and the
                // Qwen3.8 27B has one that this request disables. A no-think request keeps the
                // JSON mask safe to apply from the first token.
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
                    speculative,
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
                if let Ok(output) = &generation {
                    emit_generation_stats(&job_id, "describe", output);
                }
                Ok(generation
                    .map(|output| output.text)
                    .map_err(|error| format!("{name} generation failed: {error}")))
            },
        ))
    }

    fn condenses(&self) -> bool {
        true
    }

    /// The text-only rewrite on the model the describe call left resident (same cache key, so no
    /// second load and no new memory admission): plain text, thinking disabled.
    fn condense(&self, system: String, user: String, cancel: CancelFlag) -> CondenseFuture {
        use gen_core::core_llm::{
            FinishReason, Message, Sampling, StreamEvent, TextLlmRequest, ThinkingMode,
        };
        let (spec, requirements) = self.load_key();
        let job_id = self.job_id.clone();
        let kv_compression = crate::llm_kv_cache::worker_default_policy();
        let name = self.model.display_name();
        let speculative = self.model.speculative();
        Box::pin(crate::refine_model_cache::with_cached_refiner(
            spec,
            requirements,
            format!("{name} load failed"),
            move |model| -> WorkerResult<Result<CondenseReply, String>> {
                let request = TextLlmRequest {
                    progress: Some(crate::load_progress::text_observer()),
                    messages: vec![Message::system(system), Message::user(user)],
                    sampling: Sampling {
                        temperature: 0.4,
                        top_p: 0.9,
                        ..Sampling::default()
                    },
                    max_new_tokens: condense::CONDENSE_MAX_NEW_TOKENS,
                    constraint: None,
                    cancel,
                    thinking: ThinkingMode::Disabled,
                    speculative,
                    kv_compression,
                    ..Default::default()
                };
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
                if let Ok(output) = &generation {
                    emit_generation_stats(&job_id, "condense", output);
                }
                Ok(generation
                    .map(|output| CondenseReply {
                        generated_tokens: output.usage.generated_tokens as usize,
                        cut_off: output.finish_reason == Some(FinishReason::Length),
                        text: output.text,
                    })
                    .map_err(|error| format!("{name} generation failed: {error}")))
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
                    let _ = call.reply.send(Err(crate::generation_error(
                        &format!("JoyCaption {backend} load failed"),
                        error,
                    )));
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
    /// A condense-only job's row as the dialog holds it: the seven fields (the user may have
    /// edited them) and the image's pixel size. The image itself is never opened.
    given: Option<GivenRow>,
}

/// A condense-only or measure-only row as the dialog holds it.
#[cfg(native_lane)]
#[derive(Clone, Debug)]
struct GivenRow {
    /// The seven fields (the user may have edited them) and the row's cut-off notice, kept.
    fields: DescribedFields,
    size: Option<(u32, u32)>,
    /// Measure-only: the user's edited condensed prompt to re-count.
    edited_text: Option<String>,
}

/// What a job does with its items.
#[cfg(native_lane)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    /// Describe each image (and condense it when the job has a target).
    Describe,
    /// Condense the given rows only: no image is opened and nothing is described.
    CondenseOnly,
    /// Re-count the given rows' edited condensed prompts: nothing is generated and no model loads.
    MeasureOnly,
}

/// One item of the job result, in dataset order.
#[cfg(native_lane)]
#[derive(Clone, Debug)]
struct DescribedItem {
    item: DescribeItem,
    size: Option<(u32, u32)>,
    outcome: Result<DescribedFields, String>,
    /// The "Fit to" condensed prompt; absent without a target (the item's shape is then exactly
    /// what it was before the step existed).
    condensed: Option<Condensed>,
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
        object.insert("notice".to_owned(), json!(fields.notice));
        if let Some(condensed) = &self.condensed {
            object.insert("condensed".to_owned(), condensed.to_json());
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
pub(crate) fn bounded_error(error: &str) -> String {
    bounded_text(error, MAX_ITEM_ERROR_CHARS)
}

/// `text` cut to at most `max` characters, with an ellipsis marking a cut.
#[cfg(native_lane)]
pub(crate) fn bounded_text(text: &str, max: usize) -> String {
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
        condensed: None,
    }
    .to_json()
}

/// Append one finished image to the job's per-item results, as the worker that owns the job. A 409
/// (the job was canceled, swept or reclaimed) propagates like a rejected progress post.
#[cfg(native_lane)]
///
/// `provisional` marks a checkpoint the job will replace at the same position (an image stored
/// before its condensing step): a reader re-reads that position until it is final. A final item
/// carries no `provisional` key.
async fn append_result_item(
    api: &ApiClient,
    job_id: &str,
    position: usize,
    item: &DescribedItem,
    provisional: bool,
) -> WorkerResult<()> {
    let mut stored = stored_item_json(item);
    if provisional {
        stored["provisional"] = json!(true);
    }
    let _: Value = api
        .post_json(
            &format!("/api/v1/jobs/{job_id}/result-items"),
            &json!({
                "workerId": api.worker_id,
                "items": [{ "position": position, "item": stored }],
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
    /// The "Fit to" target, when the job has one.
    fit: Option<FitTarget>,
    mode: RunMode,
}

/// A condense-only or measure-only item's given row: a `fields` object of the seven text fields (a
/// missing one is empty; an unknown name, a non-string, an over-long value, an over-long row or a
/// row with every field empty is refused), its cut-off `notice`, its `width` and `height`, and for
/// measure-only its `condensedText`.
#[cfg(native_lane)]
fn given_row(
    item: &crate::analysis_jobs_common::DatasetItem<'_>,
    mode: RunMode,
) -> WorkerResult<GivenRow> {
    let invalid = |detail: String| {
        WorkerError::InvalidPayload(format!(
            "Dataset batch describe item {} {detail}",
            item.item_id
        ))
    };
    let object = item
        .object
        .get("fields")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("has no fields to condense.".to_owned()))?;
    if let Some(unknown) = object
        .keys()
        .find(|key| !DESCRIBE_FIELDS.contains(&key.as_str()))
    {
        return Err(invalid(format!("has an unknown field {unknown:?}.")));
    }
    let mut fields: [String; DESCRIBE_FIELDS.len()] = Default::default();
    for (slot, name) in fields.iter_mut().zip(DESCRIBE_FIELDS) {
        *slot = match object.get(name) {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(text)) if text.chars().count() <= MAX_DESCRIBE_FIELD_CHARS => {
                text.trim().to_owned()
            }
            Some(_) => {
                return Err(invalid(format!(
                    "field {name:?} must be text of at most {MAX_DESCRIBE_FIELD_CHARS} characters."
                )))
            }
        };
    }
    let total: usize = fields.iter().map(|field| field.chars().count()).sum();
    if total > sceneworks_core::dataset_batch_describe::MAX_CONDENSE_ROW_CHARS {
        return Err(invalid(format!(
            "has {total} characters across its fields, more than {}.",
            sceneworks_core::dataset_batch_describe::MAX_CONDENSE_ROW_CHARS
        )));
    }
    if fields.iter().all(String::is_empty) {
        return Err(invalid("has no text in any field to condense.".to_owned()));
    }
    let dimension = |key: &str| {
        item.object
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0)
    };
    let text = |key: &str| {
        crate::payload::trimmed_string(item.object.get(key))
            .map(|text| bounded_text(text, MAX_DESCRIBE_FIELD_CHARS))
    };
    let edited_text = match mode {
        RunMode::MeasureOnly => Some(
            text("condensedText")
                .ok_or_else(|| invalid("has no condensed prompt to measure.".to_owned()))?,
        ),
        _ => None,
    };
    Ok(GivenRow {
        fields: DescribedFields {
            fields,
            notice: text("notice").map(|notice| bounded_error(&notice)),
        },
        size: dimension("width").zip(dimension("height")),
        edited_text,
    })
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
    let fit = condense::parse_fit_target(payload)?;
    let flag = |key: &str| payload.get(key).and_then(Value::as_bool) == Some(true);
    let mode = match (flag("condenseOnly"), flag("measureOnly")) {
        (false, false) => RunMode::Describe,
        (true, false) => RunMode::CondenseOnly,
        (false, true) => RunMode::MeasureOnly,
        (true, true) => {
            return Err(WorkerError::InvalidPayload(
                "Dataset batch describe is condenseOnly or measureOnly, not both.".to_owned(),
            ))
        }
    };
    let condenses =
        sceneworks_core::dataset_batch_describe::describer_condenses(Some(model.repo()));
    if mode != RunMode::Describe && (fit.is_none() || !condenses) {
        return Err(WorkerError::InvalidPayload(
            "Dataset batch describe condenseOnly and measureOnly need a fitTo target and a \
             describer that condenses."
                .to_owned(),
        ));
    }
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
                given: if mode == RunMode::Describe {
                    None
                } else {
                    Some(given_row(&item, mode)?)
                },
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
        fit,
        mode,
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
    result.insert(
        "fitTo".to_owned(),
        json!(request.fit.as_ref().map(|fit| fit.model_id.as_str())),
    );
    result.insert(
        "condenseOnly".to_owned(),
        json!(request.mode == RunMode::CondenseOnly),
    );
    result.insert(
        "measureOnly".to_owned(),
        json!(request.mode == RunMode::MeasureOnly),
    );
    result
}

/// Progress before image `index` (0-based) of `total` starts: 0.08 at the first image, 0.96 once
/// every image is done.
#[cfg(native_lane)]
pub(crate) fn image_progress(done: usize, total: usize) -> f64 {
    0.08 + 0.88 * (done as f64 / total.max(1) as f64)
}

#[cfg(native_lane)]
pub(crate) async fn run_dataset_batch_describe_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let mut request = parse_batch_describe_request(settings, &job.payload)?;
    let condenses =
        sceneworks_core::dataset_batch_describe::describer_condenses(Some(request.model.repo()));
    if let Some(fit) = request.fit.as_mut().filter(|_| condenses) {
        match condense::exact_token_counter(settings, fit) {
            Ok(counter) => fit.counter = Some(counter),
            Err(reason) => fit.estimate_reason = Some(reason),
        }
    }
    if request.mode == RunMode::MeasureOnly {
        // Counting needs only the target's tokenizer: no describer is resolved or loaded.
        return run_batch_describe(api, settings, job, request, &NoDescriber).await;
    }
    let describer: Box<dyn BatchDescriber> = match request.model {
        DescribeModel::VisionCaptioner => Box::new(VisionCaptionerDescriber {
            model: DescribeModel::VisionCaptioner,
            weights_dir: resolve_app_managed_model_dir(
                settings,
                VISION_CAPTIONER_REPO,
                "Vision Captioner model path",
            )?,
            job_id: job.id.clone(),
        }),
        DescribeModel::Qwen38 => {
            // Both refusals come before any image: the tier must be installed and fit here.
            let tier = request.tier.unwrap_or(
                sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_DEFAULT_TIER,
            );
            let weights_dir = qwen38_describer_weights_dir(settings, tier)?;
            qwen38_memory_admission(settings, &job.payload, tier, &weights_dir).await?;
            Box::new(VisionCaptionerDescriber {
                model: DescribeModel::Qwen38,
                weights_dir,
                job_id: job.id.clone(),
            })
        }
        DescribeModel::JoyCaption => Box::new(JoyCaptionDescriber::spawn(
            resolve_app_managed_model_dir(settings, JOYCAPTION_REPO, "JoyCaption model path")?,
            job.id.clone(),
            backend_label(&settings.gpu_id).to_owned(),
            Box::new(crate::inference_runtime::load_captioner),
        )),
    };
    run_batch_describe(api, settings, job, request, describer.as_ref()).await
}

/// The describer of a measure-only job, which never describes or generates.
#[cfg(native_lane)]
struct NoDescriber;

#[cfg(native_lane)]
impl BatchDescriber for NoDescriber {
    fn describe(
        &self,
        _: image::DynamicImage,
        _: &DescribePrompt,
        _: CancelFlag,
    ) -> DescribeFuture {
        Box::pin(async { Ok(Err("A measure-only job describes nothing.".to_owned())) })
    }
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
        let condensing = request.fit.as_ref().filter(|_| describer.condenses());
        let verb = match request.mode {
            RunMode::Describe => "Describing",
            RunMode::CondenseOnly => "Condensing",
            RunMode::MeasureOnly => "Measuring",
        };
        update_job(
            api,
            &job.id,
            progress_payload_for_backend(
                JobStatus::Running,
                ProgressStage::Running,
                image_progress(index, total),
                &format!("{verb} image {} of {total}.", index + 1),
                None,
                backend,
            ),
        )
        .await?;

        let mut outcome = if let Some(given) = &item.given {
            // Condense- or measure-only: the row as the dialog holds it; no image is opened.
            DescribedItem {
                item: item.clone(),
                size: given.size,
                outcome: Ok(given.fields.clone()),
                condensed: match (&request.fit, &given.edited_text) {
                    (Some(fit), Some(text)) => {
                        Some(condense::measure_edited(fit, text, &given.fields.fields))
                    }
                    _ => None,
                },
            }
        } else {
            let Some(outcome) = describe_item(
                api,
                settings,
                job,
                &request,
                describer,
                item,
                (index, total),
            )
            .await?
            else {
                canceled = true;
                break;
            };
            outcome
        };
        let condense_now = request.mode != RunMode::MeasureOnly && outcome.outcome.is_ok();
        if let (Some(fit), true) = (condensing, condense_now) {
            if request.mode == RunMode::Describe {
                // Store the description first: a model that fails while condensing (and so
                // fails the job) must not lose the image just described. The condensed item
                // replaces it below (the store keys items by id); until then it is provisional.
                append_result_item(api, &job.id, described.len(), &outcome, true).await?;
                update_job(
                    api,
                    &job.id,
                    progress_payload_for_backend(
                        JobStatus::Running,
                        ProgressStage::Running,
                        image_progress(index, total),
                        &format!(
                            "Condensing image {} of {total} for {}.",
                            index + 1,
                            fit.name
                        ),
                        None,
                        backend,
                    ),
                )
                .await?;
            }
            let fields = outcome
                .outcome
                .as_ref()
                .map(|fields| fields.fields.clone())
                .unwrap_or_default();
            match condense_fields(api, settings, job, describer, fit, &fields, (index, total))
                .await?
            {
                Some(condensed) => outcome.condensed = Some(condensed),
                None => {
                    // Canceled while condensing: a newly described image is already stored
                    // with its fields; a condense-only row is unchanged.
                    if request.mode == RunMode::Describe {
                        described.push(outcome);
                    }
                    canceled = true;
                    break;
                }
            }
        }
        // Store the item before the post that counts it, so a reader that sees `processedCount`
        // reach N can always read N items.
        append_result_item(api, &job.id, described.len(), &outcome, false).await?;
        described.push(outcome);
        let done = described.len();
        let finished = match request.mode {
            RunMode::Describe => "Described",
            RunMode::CondenseOnly => "Condensed",
            RunMode::MeasureOnly => "Measured",
        };
        update_job(
            api,
            &job.id,
            progress_payload_for_backend(
                JobStatus::Running,
                ProgressStage::Running,
                image_progress(done, total),
                &format!("{finished} {done} of {total} images."),
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
    let message = if request.mode == RunMode::CondenseOnly {
        format!("Condensed {total} image(s).")
    } else if request.mode == RunMode::MeasureOnly {
        format!("Measured {total} image(s).")
    } else if failed == 0 {
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

/// Decode and describe one image. `None` when the job was canceled mid-description (the image has
/// no result to return).
#[cfg(native_lane)]
async fn describe_item(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &BatchDescribeRequest,
    describer: &dyn BatchDescriber,
    item: &DescribeItem,
    (index, total): (usize, usize),
) -> WorkerResult<Option<DescribedItem>> {
    let path = item.image_path.clone();
    let decoded =
        crate::load_progress::spawn_blocking(move || crate::image_decode::decode_image_any(&path))
            .await
            .map_err(|error| task_join_error("batch describe image decode", error))?;
    let image = match decoded {
        Err(error) => {
            return Ok(Some(DescribedItem {
                item: item.clone(),
                size: None,
                outcome: Err(format!("The image could not be read: {error}")),
                condensed: None,
            }))
        }
        Ok(image) => image,
    };
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
    Ok(described.map(|outcome| DescribedItem {
        item: item.clone(),
        size: Some(size),
        outcome,
        condensed: None,
    }))
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
        if let Some(fatal) = crate::process_fault(&reply) {
            return Err(fatal);
        }
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
                if let Some(recovered) = describer.recover(&raw) {
                    return Ok(Some(recovered));
                }
                return Ok(Some(Err(if attempt == 0 {
                    error
                } else {
                    format!("{error} (after {} attempts)", attempt + 1)
                })));
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
                     description object.",
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
