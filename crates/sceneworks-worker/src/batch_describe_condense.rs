//! The "Fit to" condensing step of the dataset batch describe job (SW-7163).
//!
//! A described image is seven fields (often 2,400 to 4,300 characters for one person), longer than
//! an image model reads. With a "Fit to" target (a catalog image model that declares
//! `image.maxPromptTokens`), each described image gets a second, text-only generation by the same
//! describer while it is still loaded: the fixed `[CONDENSE]` instruction of the prompt asset asks
//! for ONE prompt of at most N words that keeps every field's content, every person label and each
//! person's pose. The seven fields stay stored; the condensed prompt is stored beside them.
//!
//! **Size.** The limit comes from the catalog entry the API carried in the job
//! (`fitToManifestEntry`), never from a client number. When the target's own tokenizer is installed
//! the worker counts with it exactly (Krea 2: the snapshot's `tokenizer/tokenizer.json`, template
//! included, the same ids the engine admits against its 1,024-token cap); otherwise the count is
//! the describer's own token count of the reply, an estimate, held to
//! `CONDENSE_ESTIMATE_PERCENT` of the limit. The prompt must also fit the route's
//! character limit beside its `[W:H]` line, with each label wrapped as `{{label}}`.
//!
//! **Retry.** A reply over either limit, cut off at its budget, or missing a person label is
//! rewritten once with a stricter word target. A result still over is stored with `over: true` and
//! a plain notice; one still missing a label is stored `failed` (the batch block then uses the seven
//! fields). Text is never truncated to make it fit.

use super::*;

use sceneworks_core::dataset_batch_describe::{
    condense_char_bound, condense_token_bound, condense_word_target, CONDENSE_RETRY_PERCENT,
    LABEL_WRAP_CHARS,
};

/// Reply budget of one condensing generation. The target is at most about 600 words (about 800
/// tokens), so a reply that has said everything stops well below this; one that reaches it is cut
/// off mid-sentence and is treated as over the limit.
pub(crate) const CONDENSE_MAX_NEW_TOKENS: u32 = 1536;

/// One condensing reply: the text, how many tokens the describer generated for it (its own
/// tokenizer), and whether it stopped at the reply budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CondenseReply {
    pub(crate) text: String,
    pub(crate) generated_tokens: usize,
    pub(crate) cut_off: bool,
}

/// What a describer returns for one condensing generation. The outer `Err` fails the whole job (the
/// model could not load); the inner `Err` is this image's condensing failure.
pub(crate) type CondenseFuture = BoxFuture<'static, WorkerResult<Result<CondenseReply, String>>>;

/// Counts a prompt's tokens exactly as the target image model's engine does.
pub(crate) trait PromptTokenCounter: Send + Sync {
    fn count(&self, text: &str) -> Result<usize, String>;
}

/// A job's validated "Fit to" target: the catalog entry's id, name and declared limit, and the
/// exact counter when the target's tokenizer is installed.
pub(crate) struct FitTarget {
    pub(crate) model_id: String,
    pub(crate) name: String,
    pub(crate) limit: usize,
    pub(crate) entry: Value,
    pub(crate) counter: Option<Box<dyn PromptTokenCounter>>,
    /// Why the count is an estimate, shown on each condensed row (`None` while exact).
    pub(crate) estimate_reason: Option<String>,
}

/// The target named by the payload's `fitTo`, resolved from the catalog entry the API carried
/// (`fitToManifestEntry`, whose id must be `fitTo`). `None` when the job has no target.
pub(crate) fn parse_fit_target(payload: &JsonObject) -> WorkerResult<Option<FitTarget>> {
    let Some(model_id) = crate::payload::trimmed_string(payload.get("fitTo")) else {
        return Ok(None);
    };
    let entry = payload
        .get("fitToManifestEntry")
        .filter(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Dataset batch describe fitTo {model_id:?} carries no catalog entry."
            ))
        })?;
    let limit =
        sceneworks_core::dataset_batch_describe::prompt_token_limit(entry).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Dataset batch describe fitTo {model_id:?} declares no image.maxPromptTokens."
            ))
        })?;
    Ok(Some(FitTarget {
        model_id: model_id.to_owned(),
        name: crate::payload::trimmed_string(entry.get("name"))
            .unwrap_or(model_id)
            .to_owned(),
        limit,
        entry: entry.clone(),
        counter: None,
        estimate_reason: None,
    }))
}

/// The stored condensed prompt of one image (the result item's `condensed` object).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Condensed {
    pub(crate) target: String,
    pub(crate) limit: usize,
    pub(crate) text: String,
    /// The measured size in tokens; `None` when no reply was produced.
    pub(crate) tokens: Option<usize>,
    pub(crate) exact: bool,
    pub(crate) estimate_reason: Option<String>,
    pub(crate) characters: usize,
    pub(crate) over: bool,
    /// No usable prompt (a generation error, or a person label still missing after the retry):
    /// the batch block uses the seven fields.
    pub(crate) failed: bool,
    pub(crate) notice: Option<String>,
}

impl Condensed {
    pub(crate) fn to_json(&self) -> Value {
        json!({
            "target": self.target,
            "limitTokens": self.limit,
            "text": self.text,
            "tokens": self.tokens,
            "tokenCount": if self.exact { "exact" } else { "estimate" },
            "estimateReason": self.estimate_reason,
            "characters": self.characters,
            "over": self.over,
            "failed": self.failed,
            "notice": self.notice,
        })
    }
}

/// The person labels in `text`: every whole word (ASCII letters, digits, underscore) of the form
/// `<lowercase letters>_<digits>` — the same rule the web formatter wraps as `{{label}}`.
pub(crate) fn person_labels(text: &str) -> Vec<&str> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|word| {
            word.split_once('_').is_some_and(|(noun, number)| {
                !noun.is_empty()
                    && noun.bytes().all(|b| b.is_ascii_lowercase())
                    && !number.is_empty()
                    && number.bytes().all(|b| b.is_ascii_digit())
            })
        })
        .collect()
}

/// The fixed condensing instruction with its word target filled in.
pub(crate) fn condense_system(max_words: usize) -> String {
    crate::prompt_refine_jobs::magic_section_from(super::IMAGE_BATCH_DESCRIBE_V1, "CONDENSE")
        .replace("{max_words}", &max_words.to_string())
}

/// The user turn: the seven fields as `Field: value` lines (one `Pose:` line per person); empty
/// fields are left out.
pub(crate) fn condense_user(fields: &[String; DESCRIBE_FIELDS.len()]) -> String {
    const NAMES: [&str; 7] = [
        "Description",
        "Pose",
        "Style",
        "Camera",
        "Composition",
        "Lighting",
        "Environment",
    ];
    let mut lines = vec![
        "Rewrite these fields into one prompt.".to_owned(),
        String::new(),
    ];
    for (name, text) in NAMES.iter().zip(fields) {
        for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
            lines.push(format!("{name}: {line}"));
        }
    }
    lines.join("\n")
}

/// The measured size of one reply against the target's bounds.
struct Measured {
    text: String,
    tokens: usize,
    exact: bool,
    characters: usize,
    block_chars: usize,
    cut_off: bool,
    missing: Vec<String>,
}

impl Measured {
    fn over(&self, token_bound: usize) -> bool {
        self.cut_off || self.tokens > token_bound || self.block_chars > condense_char_bound()
    }
}

fn measure(
    fit: &FitTarget,
    reply: CondenseReply,
    labels: &std::collections::BTreeSet<&str>,
) -> Measured {
    let text = crate::text_clean::strip_reasoning_and_fence(&reply.text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let found = person_labels(&text);
    let characters = text.chars().count();
    let block_chars = characters + LABEL_WRAP_CHARS * found.len();
    let missing = labels
        .iter()
        .filter(|label| !found.contains(label))
        .map(|label| (*label).to_owned())
        .collect();
    let exact_count = fit
        .counter
        .as_ref()
        .and_then(|counter| counter.count(&text).ok());
    Measured {
        tokens: exact_count.unwrap_or(reply.generated_tokens),
        exact: exact_count.is_some(),
        text,
        characters,
        block_chars,
        cut_off: reply.cut_off,
        missing,
    }
}

/// Condense one image's fields for `fit`: one generation, and one stricter retry when the reply is
/// over a limit or misses a person label. Returns `None` when the job was canceled meanwhile.
pub(crate) async fn condense_fields(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    describer: &dyn BatchDescriber,
    fit: &FitTarget,
    fields: &[String; DESCRIBE_FIELDS.len()],
    (index, total): (usize, usize),
) -> WorkerResult<Option<Condensed>> {
    let labels: std::collections::BTreeSet<&str> = fields
        .iter()
        .flat_map(|field| person_labels(field))
        .collect();
    let token_bound = condense_token_bound(fit.limit, fit.counter.is_some());
    let mut words = condense_word_target(token_bound, condense_char_bound());
    let user = condense_user(fields);
    let mut last: Option<Measured> = None;
    for attempt in 0..2 {
        if attempt > 0 {
            // Honour a cancel before spending the retry.
            if cancel_requested_peek(api, &job.id).await {
                return Ok(None);
            }
            words = words * CONDENSE_RETRY_PERCENT / 100;
            update_job(
                api,
                &job.id,
                progress_payload_for_backend(
                    JobStatus::Running,
                    ProgressStage::Running,
                    super::image_progress(index, total),
                    &format!(
                        "Condensing image {} of {total} for {} again (attempt 2): the first \
                         prompt did not fit.",
                        index + 1,
                        fit.name
                    ),
                    None,
                    backend_label(&settings.gpu_id),
                ),
            )
            .await?;
        }
        let cancel = CancelFlag::new();
        let generation = describer.condense(condense_system(words), user.clone(), cancel.clone());
        let (was_canceled, reply) =
            StreamSupervisor::new(cancel, generation, progress_report_interval(settings), None)
                .wait(api, settings, &job.id)
                .await?;
        if was_canceled {
            return Ok(None);
        }
        let reply = match reply? {
            Ok(reply) => reply,
            Err(error) => {
                return Ok(Some(Condensed {
                    target: fit.model_id.clone(),
                    limit: fit.limit,
                    text: String::new(),
                    tokens: None,
                    exact: fit.counter.is_some(),
                    estimate_reason: fit.estimate_reason.clone(),
                    characters: 0,
                    over: false,
                    failed: true,
                    notice: Some(super::bounded_error(&format!(
                        "Condensing failed, so the seven fields are used: {error}"
                    ))),
                }))
            }
        };
        let measured = measure(fit, reply, &labels);
        let done = !measured.over(token_bound) && measured.missing.is_empty();
        last = Some(measured);
        if done {
            break;
        }
    }
    let measured = last.expect("at least one attempt ran");
    let over = measured.over(token_bound);
    let failed = !measured.missing.is_empty() || measured.text.is_empty();
    let notice = if !measured.missing.is_empty() {
        Some(format!(
            "The condensed prompt leaves out {} even after one retry, so the seven fields are used.",
            measured.missing.join(", ")
        ))
    } else if measured.text.is_empty() {
        Some("The condensed prompt came back empty, so the seven fields are used.".to_owned())
    } else if over {
        let mut reasons = Vec::new();
        if measured.cut_off {
            reasons.push("the reply was cut off at its length limit".to_owned());
        }
        if measured.tokens > token_bound {
            reasons.push(format!(
                "{}{} of {token_bound} tokens",
                if measured.exact { "" } else { "about " },
                measured.tokens
            ));
        }
        if measured.block_chars > condense_char_bound() {
            reasons.push(format!(
                "{} of {} characters",
                measured.block_chars,
                condense_char_bound()
            ));
        }
        Some(format!(
            "Still over the limit after one retry ({}). Shorten it before generating.",
            reasons.join("; ")
        ))
    } else {
        None
    };
    Ok(Some(Condensed {
        target: fit.model_id.clone(),
        limit: fit.limit,
        tokens: Some(measured.tokens),
        exact: measured.exact,
        estimate_reason: (!measured.exact).then(|| {
            fit.estimate_reason
                .clone()
                .unwrap_or_else(|| format!("{} tokens could not be counted exactly", fit.name))
        }),
        characters: measured.characters,
        text: super::bounded_text(&measured.text, super::MAX_DESCRIBE_FIELD_CHARS),
        over,
        failed,
        notice,
    }))
}

/// The exact counter for a target, keyed on its catalog entry's `adapter` (data, not a model id):
/// Krea 2 (`mlx_krea`) counts with its snapshot's tokenizer, template included. `Err` says why the
/// count will be an estimate instead.
pub(crate) fn exact_token_counter(
    settings: &Settings,
    fit: &FitTarget,
) -> Result<Box<dyn PromptTokenCounter>, String> {
    match fit.entry.get("adapter").and_then(Value::as_str) {
        Some("mlx_krea") => {
            let root = tokenizer_roots(settings, &fit.entry)
                .into_iter()
                .next()
                .ok_or_else(|| format!("{} is not installed", fit.name))?;
            KreaCounter::load(&root)
                .map(|counter| Box::new(counter) as Box<dyn PromptTokenCounter>)
                .map_err(|error| format!("{}'s tokenizer could not be read: {error}", fit.name))
        }
        _ => Err(format!(
            "SceneWorks has no tokenizer-only counter for {}",
            fit.name
        )),
    }
}

/// The installed snapshot roots of an entry's primary downloads that hold `tokenizer/tokenizer.json`
/// beside their own files (a `<tier>/tokenizer/*` pattern; the tiers share one tokenizer).
pub(crate) fn tokenizer_roots(settings: &Settings, entry: &Value) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for download in entry
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| download.get("coRequisite").and_then(Value::as_bool) != Some(true))
    {
        let (Some(repo), Some(revision)) = (
            download.get("repo").and_then(Value::as_str),
            download.get("revision").and_then(Value::as_str),
        ) else {
            continue;
        };
        let Some(snapshot) =
            crate::model_jobs::huggingface_pinned_snapshot_dir(&settings.data_dir, repo, revision)
        else {
            continue;
        };
        for pattern in download
            .get("files")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            let Some(prefix) = pattern.strip_suffix("tokenizer/*") else {
                continue;
            };
            // The entry rides in the job payload: only plain relative components may join.
            if !Path::new(prefix)
                .components()
                .all(|part| matches!(part, std::path::Component::Normal(_)))
            {
                continue;
            }
            let root = snapshot.join(prefix);
            if root.join("tokenizer").join("tokenizer.json").is_file() && !roots.contains(&root) {
                roots.push(root);
            }
        }
    }
    roots
}

/// Krea 2's condition tokenizer (tokenizer file and encoder config only, no weights, no GPU).
struct KreaCounter(KreaTokenizer);

#[cfg(not(target_os = "macos"))]
use runtime_cuda::providers::krea::KreaTokenizer;
#[cfg(target_os = "macos")]
use runtime_macos::providers::krea::KreaTokenizer;

impl KreaCounter {
    fn load(root: &Path) -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        let tokenizer = KreaTokenizer::from_snapshot(root);
        #[cfg(not(target_os = "macos"))]
        let tokenizer =
            KreaTokenizer::from_snapshot(root, &runtime_cuda::media::candle_core::Device::Cpu);
        tokenizer.map(Self).map_err(|error| error.to_string())
    }
}

impl PromptTokenCounter for KreaCounter {
    fn count(&self, text: &str) -> Result<usize, String> {
        self.0
            .ids(text)
            .map(|ids| ids.len())
            .map_err(|error| error.to_string())
    }
}
