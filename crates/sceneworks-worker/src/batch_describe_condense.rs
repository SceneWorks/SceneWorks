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
//! character limit beside its `[W:H]` line. Both bounds keep room for every person-label mention,
//! which Image Studio replaces with the user's text before it sends the prompt
//! (`VARIABLE_VALUE_TOKEN_RESERVE`, `VARIABLE_VALUE_CHAR_RESERVE` in sceneworks-core).
//!
//! **Retry.** A reply over either limit, cut off at its budget, not one plain prompt (a preamble,
//! field names, headings or a list), or naming a different set of person labels than the fields
//! (one left out, or one invented) is rewritten once with a stricter word target. A result still
//! over is stored with `over: true` and a plain notice; one still malformed is stored `failed`
//! (the batch block then uses the seven fields). Text is never truncated to make it fit.
//!
//! **Measure-only.** A user-edited condensed prompt is re-counted without generating: the exact
//! count needs only the target's tokenizer, never the describer, so a measure-only job loads no
//! model. Without the target's tokenizer the edited prompt stays unmeasured.

use super::*;

use sceneworks_core::dataset_batch_describe::{
    condense_char_bound, condense_token_bound, condense_word_target, CONDENSE_RETRY_PERCENT,
    VARIABLE_VALUE_CHAR_RESERVE, VARIABLE_VALUE_TOKEN_RESERVE,
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
    /// The measured size in tokens; `None` when no reply was produced or the size could not be
    /// counted (a measure-only request without the target's tokenizer).
    pub(crate) tokens: Option<usize>,
    pub(crate) exact: bool,
    pub(crate) estimate_reason: Option<String>,
    pub(crate) characters: usize,
    /// Person-label mentions in the text: each keeps [`VARIABLE_VALUE_TOKEN_RESERVE`] tokens and
    /// [`VARIABLE_VALUE_CHAR_RESERVE`] characters of room for its filled-in value.
    pub(crate) mentions: usize,
    pub(crate) over: bool,
    /// No usable prompt (a generation error, a reply that is not one plain prompt, or labels that
    /// differ from the fields' after the retry): the batch block uses the seven fields.
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
            "labelMentions": self.mentions,
            "reservedTokens": self.mentions * VARIABLE_VALUE_TOKEN_RESERVE,
            "over": self.over,
            "failed": self.failed,
            "notice": self.notice,
        })
    }
}

/// The person labels in `text`: every whole word (ASCII letters, digits, underscore) of the form
/// `<lowercase letters>_<digits>` — the same rule the web formatter wraps as `{{label}}`. A label
/// inside a longer word (`xwoman_1`, `woman_1b`) is not one.
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

/// The names the condensing user turn gives the seven fields, in field order.
const FIELD_NAMES: [&str; 7] = [
    "Description",
    "Pose",
    "Style",
    "Camera",
    "Composition",
    "Lighting",
    "Environment",
];

/// The user turn: the seven fields as `Field: value` lines (one `Pose:` line per person); empty
/// fields are left out.
pub(crate) fn condense_user(fields: &[String; DESCRIBE_FIELDS.len()]) -> String {
    let mut lines = vec![
        "Rewrite these fields into one prompt.".to_owned(),
        String::new(),
    ];
    for (name, text) in FIELD_NAMES.iter().zip(fields) {
        for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
            lines.push(format!("{name}: {line}"));
        }
    }
    lines.join("\n")
}

/// Why a reply is not one plain prompt, judged cheaply on its text before whitespace collapses:
/// a preamble ("Here is the prompt:"), a field name used as a heading ("Pose:"), a markdown
/// heading, or a list marker. `None` for a plain prompt.
pub(crate) fn shape_problem(raw: &str) -> Option<&'static str> {
    let text = crate::text_clean::strip_reasoning_and_fence(raw);
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let first = lines
        .first()
        .copied()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if ["here is", "here's", "sure", "certainly", "okay", "below is"]
        .iter()
        .any(|start| first.starts_with(start))
        || (lines.len() > 1 && first.ends_with(':'))
    {
        return Some("it starts with a preamble");
    }
    if lines.iter().any(|line| line.starts_with('#')) {
        return Some("it has a markdown heading");
    }
    let is_list_item = |line: &&str| {
        line.starts_with("- ")
            || line.starts_with("* ")
            || line.starts_with("• ")
            || line.split_once(". ").is_some_and(|(number, _)| {
                !number.is_empty() && number.bytes().all(|b| b.is_ascii_digit())
            })
    };
    if lines.iter().any(is_list_item) {
        return Some("it is a list");
    }
    let has_field_name = FIELD_NAMES.iter().any(|name| {
        let needle = format!("{name}:");
        text.match_indices(&needle).any(|(at, _)| {
            text[..at]
                .chars()
                .next_back()
                .is_none_or(|before| !before.is_alphanumeric())
        })
    });
    if has_field_name {
        return Some("it repeats the field names");
    }
    None
}

/// One reply's measurement against the target's bounds.
pub(crate) struct Measured {
    text: String,
    /// `None` when it could not be counted.
    tokens: Option<usize>,
    exact: bool,
    /// The bound this reply is judged against: the limit for an exact count, the estimate share
    /// of it otherwise (also when the exact counter failed on this reply).
    token_bound: usize,
    characters: usize,
    mentions: usize,
    cut_off: bool,
    missing: Vec<String>,
    extra: Vec<String>,
    shape: Option<&'static str>,
}

impl Measured {
    fn tokens_over(&self) -> bool {
        self.tokens.is_some_and(|tokens| {
            tokens + self.mentions * VARIABLE_VALUE_TOKEN_RESERVE > self.token_bound
        })
    }

    fn chars_over(&self) -> bool {
        self.characters + self.mentions * VARIABLE_VALUE_CHAR_RESERVE > condense_char_bound()
    }

    fn over(&self) -> bool {
        self.cut_off || self.tokens_over() || self.chars_over()
    }

    fn usable(&self) -> bool {
        !self.text.is_empty()
            && self.shape.is_none()
            && self.missing.is_empty()
            && self.extra.is_empty()
    }
}

/// Measure `raw` for `fit`: the target's exact count when its counter runs, else
/// `generated_tokens` (the describer's own count, an estimate), else unmeasured. `labels` are the
/// fields' person labels: the text's set must equal them.
pub(crate) fn measure(
    fit: &FitTarget,
    raw: &str,
    generated_tokens: Option<usize>,
    cut_off: bool,
    labels: &std::collections::BTreeSet<&str>,
) -> Measured {
    let shape = shape_problem(raw);
    let text = crate::text_clean::strip_reasoning_and_fence(raw)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let found = person_labels(&text);
    let found_set: std::collections::BTreeSet<&str> = found.iter().copied().collect();
    let exact_count = fit
        .counter
        .as_ref()
        .and_then(|counter| counter.count(&text).ok());
    Measured {
        tokens: exact_count.or(generated_tokens),
        exact: exact_count.is_some(),
        token_bound: condense_token_bound(fit.limit, exact_count.is_some()),
        characters: text.chars().count(),
        mentions: found.len(),
        cut_off,
        missing: labels
            .difference(&found_set)
            .map(|l| (*l).to_owned())
            .collect(),
        extra: found_set
            .difference(labels)
            .map(|l| (*l).to_owned())
            .collect(),
        shape,
        text,
    }
}

/// The stored result of a measurement, with the plain notice of whatever is wrong with it.
/// `generated` is false for a user-edited prompt, whose label or shape problems are reported but
/// never make it `failed` (it is the user's text).
pub(crate) fn condensed_from(fit: &FitTarget, measured: Measured, generated: bool) -> Condensed {
    let after = if generated {
        " even after one retry"
    } else {
        ""
    };
    let fallback = if generated {
        ", so the seven fields are used"
    } else {
        ""
    };
    let mut problems = Vec::new();
    if measured.text.is_empty() {
        problems.push("is empty".to_owned());
    }
    if let Some(shape) = measured.shape {
        problems.push(format!("is not one plain prompt ({shape})"));
    }
    if !measured.missing.is_empty() {
        problems.push(format!("leaves out {}", measured.missing.join(", ")));
    }
    if !measured.extra.is_empty() {
        problems.push(format!(
            "adds {}, which the fields do not have",
            measured.extra.join(", ")
        ));
    }
    let over = measured.over();
    let notice = if !problems.is_empty() {
        Some(format!(
            "The condensed prompt {}{after}{fallback}.",
            problems.join("; ")
        ))
    } else if over {
        let mut reasons = Vec::new();
        if measured.cut_off {
            reasons.push("the reply was cut off at its length limit".to_owned());
        }
        if measured.tokens_over() {
            reasons.push(format!(
                "{}{} tokens plus room for {} variable values is over {}",
                if measured.exact { "" } else { "about " },
                measured.tokens.unwrap_or_default(),
                measured.mentions,
                measured.token_bound
            ));
        }
        if measured.chars_over() {
            reasons.push(format!(
                "{} characters plus room for {} variable values is over {}",
                measured.characters,
                measured.mentions,
                condense_char_bound()
            ));
        }
        Some(format!(
            "Still over the limit{after} ({}). Shorten it before generating.",
            reasons.join("; ")
        ))
    } else {
        None
    };
    Condensed {
        target: fit.model_id.clone(),
        limit: fit.limit,
        tokens: measured.tokens,
        exact: measured.exact,
        estimate_reason: (!measured.exact).then(|| {
            fit.estimate_reason
                .clone()
                .unwrap_or_else(|| format!("{} tokens could not be counted exactly", fit.name))
        }),
        characters: measured.characters,
        mentions: measured.mentions,
        failed: generated && !measured.usable(),
        over,
        notice,
        text: super::bounded_text(&measured.text, super::MAX_DESCRIBE_FIELD_CHARS),
    }
}

/// The fields' person labels.
fn field_labels(fields: &[String; DESCRIBE_FIELDS.len()]) -> std::collections::BTreeSet<&str> {
    fields
        .iter()
        .flat_map(|field| person_labels(field))
        .collect()
}

/// Measure a user-edited condensed prompt without generating anything (a measure-only job): the
/// exact count when the target's tokenizer is installed, otherwise unmeasured.
pub(crate) fn measure_edited(
    fit: &FitTarget,
    text: &str,
    fields: &[String; DESCRIBE_FIELDS.len()],
) -> Condensed {
    condensed_from(
        fit,
        measure(fit, text, None, false, &field_labels(fields)),
        false,
    )
}

/// Condense one image's fields for `fit`: one generation, and one stricter retry when the reply
/// is over a limit (counting the variable-value reserve), is not one plain prompt, or names a
/// different set of person labels than the fields. Returns `None` when the job was canceled
/// meanwhile.
pub(crate) async fn condense_fields(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    describer: &dyn BatchDescriber,
    fit: &FitTarget,
    fields: &[String; DESCRIBE_FIELDS.len()],
    (index, total): (usize, usize),
) -> WorkerResult<Option<Condensed>> {
    let labels = field_labels(fields);
    let token_bound = condense_token_bound(fit.limit, fit.counter.is_some());
    // Before any reply, expect about three mentions of each person.
    let mut words = condense_word_target(token_bound, condense_char_bound(), labels.len() * 3);
    let user = condense_user(fields);
    let mut last: Option<Measured> = None;
    for attempt in 0..2 {
        if let Some(previous) = &last {
            // Honour a cancel before spending the retry.
            if cancel_requested_peek(api, &job.id).await {
                return Ok(None);
            }
            let retry = condense_word_target(
                previous.token_bound,
                condense_char_bound(),
                previous.mentions.max(labels.len() * 3),
            );
            words = retry.min(words) * CONDENSE_RETRY_PERCENT / 100;
            update_job(
                api,
                &job.id,
                progress_payload_for_backend(
                    JobStatus::Running,
                    ProgressStage::Running,
                    super::image_progress(index, total),
                    &format!(
                        "Condensing image {} of {total} for {} again (attempt {}): the first \
                         prompt did not fit.",
                        index + 1,
                        fit.name,
                        attempt + 1
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
        if let Some(fatal) = crate::process_fault(&reply) {
            return Err(fatal);
        }
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
                    mentions: 0,
                    over: false,
                    failed: true,
                    notice: Some(super::bounded_error(&format!(
                        "Condensing failed, so the seven fields are used: {error}"
                    ))),
                }))
            }
        };
        let measured = measure(
            fit,
            &reply.text,
            Some(reply.generated_tokens),
            reply.cut_off,
            &labels,
        );
        let done = measured.usable() && !measured.over();
        last = Some(measured);
        if done {
            break;
        }
    }
    Ok(Some(condensed_from(
        fit,
        last.expect("at least one attempt ran"),
        true,
    )))
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
