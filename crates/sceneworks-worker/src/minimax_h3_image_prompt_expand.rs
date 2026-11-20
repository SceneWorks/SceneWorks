//! MiniMax-H3-Image prompt expansion (SW-7102, feature SW-F363; model choice SW-7208).
//!
//! `MiniMaxAI/MiniMax-H3-Image-PE` expands a short prompt — or an edit instruction that names its
//! ordered references as `<Picture N>` — into the long caption MiniMax-H3-Image was trained on, and
//! proposes an aspect ratio. It is a Qwen3.8-27B fine-tune on the ordinary native TextLlm lane, so
//! [`crate::prompt_refine_jobs`] drives it as the `prompt_refine` task
//! [`sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPAND_TASK`]. The request assembly and the reply contract are the engine's
//! (`core_llm::minimax_h3_image_pe`); this module only finds the weights and the prompt files on
//! disk and shapes the job result.
//!
//! # Two models, one contract
//!
//! The user picks the model (SW-7208): MiniMax-H3-Image-PE (the default), or the abliterated
//! Qwen3.8-27B the dataset describer uses (`vision_describe_qwen3_8_27b`, any installed tier). Both
//! run the same engine request and reply contract from the same MiniMax prompt files; only the
//! weights directory differs. The job's `expanderModelId` names the model and its
//! `modelManifestEntry` is that model's entry; for the describer the API adds MiniMax's `prompt`
//! files to it as a co-requisite. The result names the model that ran.
//!
//! # The prompts are read, never copied
//!
//! The system and task prompts are MiniMax Materials under the MiniMax-H3-Image Community License.
//! The engine reads them from the upstream snapshot's `prompt/` directory and checks every file
//! against `prompt/manifest.json`, whose own SHA-256 it pins with the revision
//! (`PromptFiles::load_verified`). A converted tier never carries `prompt/`, so the catalog's
//! `prompt` co-requisite fetches those five files from the upstream repository for every tier, and
//! [`resolve_dirs`] reads them from there. Nothing licensed is in this repository; the tests build
//! synthetic trees.
//!
//! # Nothing is applied on its own
//!
//! The job result carries the original prompt and the expansion side by side, plus the proposed
//! aspect ratio in `rewriteSuggestion`. The UI shows the expansion for review and replaces the
//! prompt only when the user accepts it; the proposed size is a separate offer.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::{JsonObject, Settings, WorkerError, WorkerResult};
use gen_core::core_llm::minimax_h3_image_pe as pe;
use sceneworks_core::model_artifacts::artifact_selection::{
    is_pending_artifact_download, model_download_for_variant,
};

/// The `task` discriminator that selects this path on a `prompt_refine` job.
use sceneworks_core::contracts::MINIMAX_H3_IMAGE_MODEL_ID as TARGET_MODEL_ID;
/// The expander's catalog id.
pub(crate) use sceneworks_core::contracts::MINIMAX_H3_IMAGE_PE_MODEL_ID as PE_MODEL_ID;

/// The co-requisite component whose upstream snapshot holds `prompt/`.
const PROMPT_COMPONENT: &str = "prompt";
/// The prompt manifest inside a snapshot.
const PROMPT_MANIFEST: &str = "prompt/manifest.json";
/// The abliterated describer, the expander's other model (SW-7208).
pub(crate) use sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_MODEL_ID as DESCRIBER_MODEL_ID;

/// The user-facing name of expander model `model_id`, for errors.
fn model_label(model_id: &str) -> String {
    if model_id == DESCRIBER_MODEL_ID {
        format!("the abliterated Qwen3.8-27B describer ({DESCRIBER_MODEL_ID})")
    } else {
        format!("the MiniMax-H3-Image Prompt Expander ({PE_MODEL_ID})")
    }
}

/// The checkpoint repository an expansion with `model_id` names on its result when no local
/// snapshot overrides it.
pub(crate) fn default_repo(model_id: &str) -> &'static str {
    if model_id == DESCRIBER_MODEL_ID {
        sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_REPO
    } else {
        pe::REPO
    }
}

/// Whether the expander applies to `model` (MiniMax-H3-Image only).
pub(crate) fn is_expand_target_model(model: &str) -> bool {
    model == TARGET_MODEL_ID
}

/// The two upstream workflows, selected by the REQUEST: references present means edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Workflow {
    TextToImage,
    Edit,
}

impl Workflow {
    pub(crate) fn for_reference_count(references: usize) -> Self {
        if references == 0 {
            Self::TextToImage
        } else {
            Self::Edit
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::TextToImage => "t2i",
            Self::Edit => "edit",
        }
    }
}

/// Where one expansion reads its weights and its prompts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExpanderDirs {
    /// The expander model that runs: [`PE_MODEL_ID`] or [`DESCRIBER_MODEL_ID`].
    pub(crate) model_id: &'static str,
    /// The tier directory the TextLlm loads.
    pub(crate) weights: PathBuf,
    /// The upstream snapshot holding `prompt/`.
    pub(crate) prompt_root: PathBuf,
    /// The tier `weights` is, or `None` for an explicit local snapshot.
    pub(crate) tier: Option<&'static str>,
}

fn entry_value(entry: &JsonObject) -> Value {
    Value::Object(entry.clone())
}

fn row_identity(row: &Value) -> Option<(&str, &str)> {
    Some((
        row.get("repo").and_then(Value::as_str)?,
        row.get("revision").and_then(Value::as_str)?,
    ))
}

/// The tier subdirectory a converted row's `files` name (`q8/*` ⇒ `q8`), or `None` for the
/// upstream row whose weights sit at the snapshot root.
fn row_tier_subdir(row: &Value) -> Option<String> {
    let first = row.get("files")?.as_array()?.first()?.as_str()?;
    let (head, _) = first.split_once('/')?;
    sceneworks_core::ids::is_safe_id(head).then(|| head.to_owned())
}

/// Whether this host can run `tier`. NVFP4 needs Blackwell CUDA (sm_120+); the converted tiers
/// run on both lanes.
fn tier_runs_here(tier: &str) -> bool {
    if tier != "nvfp4" {
        return true;
    }
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        crate::gpu::compute_cap_meets_nvfp4(crate::gpu::cached_compute_cap())
    }
    #[cfg(not(all(not(target_os = "macos"), feature = "backend-candle")))]
    {
        false
    }
}

/// `tier`'s directory in the cache when its row is declared, published and its pinned snapshot is
/// present. The directory holds a loadable checkpoint when its `config.json` is present.
fn tier_dir(data_dir: &Path, entry: &JsonObject, tier: &str) -> Option<PathBuf> {
    let row = model_download_for_variant(&entry_value(entry), tier)
        .filter(|row| !is_pending_artifact_download(row))?;
    let (repo, revision) = row_identity(&row)?;
    let snapshot = crate::model_jobs::huggingface_pinned_snapshot_dir(data_dir, repo, revision)?;
    let dir = match row_tier_subdir(&row) {
        Some(subdir) => snapshot.join(subdir),
        None => snapshot,
    };
    dir.join("config.json").is_file().then_some(dir)
}

/// The upstream snapshot the `prompt` co-requisite pins, when its manifest is on disk.
fn prompt_root(data_dir: &Path, entry: &JsonObject, model_id: &str) -> WorkerResult<PathBuf> {
    let row = entry
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|row| {
            row.get("coRequisite").and_then(Value::as_bool) == Some(true)
                && row.get("componentId").and_then(Value::as_str) == Some(PROMPT_COMPONENT)
        })
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{model_id}: the expansion entry declares no `{PROMPT_COMPONENT}` co-requisite — \
                 the model manifest entry is misconfigured"
            ))
        })?;
    let (repo, revision) = row_identity(row).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{model_id}: the `{PROMPT_COMPONENT}` co-requisite declares no repo/revision"
        ))
    })?;
    let install = if model_id == DESCRIBER_MODEL_ID {
        "download MiniMax's prompt files from the Expand prompt panel (the prompt files alone, \
         no expander weights), or install the MiniMax-H3-Image Prompt Expander"
    } else {
        "install the MiniMax-H3-Image Prompt Expander from the Model Manager"
    };
    crate::model_jobs::huggingface_pinned_snapshot_dir(data_dir, repo, revision)
        .filter(|root| root.join(PROMPT_MANIFEST).is_file())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{model_id}: MiniMax's prompt files ({repo} @ {revision}, `prompt/`) are not \
                 installed — {install}"
            ))
        })
}

/// Resolve the expander's weights and prompt directories from the entry of the model it runs
/// (MiniMax-H3-Image-PE, or the abliterated describer with MiniMax's `prompt` co-requisite added).
///
/// `explicit_tier` (the payload's `peTier`) is honoured or refused by name, never substituted;
/// with none, the first installed tier this host can run, in the model's expander tier order
/// ([`sceneworks_core::contracts::minimax_h3_image_expander_tiers`]). `override_dir` is an
/// explicit local snapshot (the real-weight smoke's `payload.model`): its own `prompt/` is used
/// when it has one, else the catalog's.
pub(crate) fn resolve_dirs(
    settings: &Settings,
    entry: &JsonObject,
    explicit_tier: Option<&str>,
    override_dir: Option<PathBuf>,
) -> WorkerResult<ExpanderDirs> {
    resolve_dirs_for_os(
        settings,
        entry,
        explicit_tier,
        override_dir,
        std::env::consts::OS,
    )
}

/// [`resolve_dirs`] for a worker running on `os`.
fn resolve_dirs_for_os(
    settings: &Settings,
    entry: &JsonObject,
    explicit_tier: Option<&str>,
    override_dir: Option<PathBuf>,
    os: &str,
) -> WorkerResult<ExpanderDirs> {
    let Ok(model_id) = sceneworks_core::contracts::minimax_h3_image_expander_model(Some(
        entry.get("id").and_then(Value::as_str).unwrap_or_default(),
    )) else {
        return Err(WorkerError::InvalidPayload(format!(
            "prompt expansion needs the `{PE_MODEL_ID}` or `{DESCRIBER_MODEL_ID}` catalog entry on \
             the job payload (`modelManifestEntry`)"
        )));
    };
    let tiers = sceneworks_core::contracts::minimax_h3_image_expander_tiers(model_id);
    let label = model_label(model_id);
    // The API forwards the complete cross-platform entry, and each converted tier has one row per
    // layout (`q8/` for MLX on macOS, `candle-q8/` for Candle on Windows/Linux) under the same
    // variant name. Narrow it to this host's rows first, so the tier lookup below cannot pick
    // another platform's layout.
    let mut narrowed = Value::Object(entry.clone());
    sceneworks_core::model_artifacts::artifact_selection::retain_downloads_for_os(
        &mut narrowed,
        os,
    );
    let entry = narrowed.as_object().unwrap_or(entry);
    let data_dir = settings.data_dir.as_path();
    if let Some(weights) = override_dir {
        let prompt_root = if weights.join(PROMPT_MANIFEST).is_file() {
            weights.clone()
        } else {
            prompt_root(data_dir, entry, model_id)?
        };
        return Ok(ExpanderDirs {
            model_id,
            weights,
            prompt_root,
            tier: None,
        });
    }
    let (tier, weights) = match explicit_tier {
        Some(requested) => {
            let tier = tiers
                .iter()
                .copied()
                .find(|tier| tier.eq_ignore_ascii_case(requested.trim()))
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "{model_id} has no `{requested}` tier; its tiers are {}",
                        tiers.join(", ")
                    ))
                })?;
            if !tier_runs_here(tier) {
                return Err(WorkerError::InvalidPayload(format!(
                    "{model_id} tier {tier} needs a Blackwell NVIDIA GPU (compute capability \
                     12.0 or newer); pick another tier"
                )));
            }
            let dir = tier_dir(data_dir, entry, tier).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{model_id} tier {tier} is not installed or not published yet; install it \
                     in the Model Manager or pick an installed tier (an explicit tier selection is \
                     never substituted with another)"
                ))
            })?;
            (tier, dir)
        }
        None => tiers
            .iter()
            .copied()
            .filter(|tier| tier_runs_here(tier))
            .find_map(|tier| tier_dir(data_dir, entry, tier).map(|dir| (tier, dir)))
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{label} is not installed in a tier this host can run — MiniMax-H3-Image \
                     generates normally without prompt expansion; open the Models screen to \
                     download it"
                ))
            })?,
    };
    Ok(ExpanderDirs {
        model_id,
        weights,
        prompt_root: prompt_root(data_dir, entry, model_id)?,
        tier: Some(tier),
    })
}

/// Stage the cached image generator out before the expander loads cold. The 27B expander and the
/// MiniMax-H3-Image generator it serves do not fit together on most hosts, and the generator stays
/// in `generator_cache` for its idle window after a render; the expansion's own refine-cache load
/// evicts only a previous refiner. A warm expander (already resident, so the generator was staged
/// out when it loaded, or the render after it evicted the expander) needs no new memory, so
/// nothing is evicted for it — the same order as the Qwen3.8 describer's cold-load admission. The
/// cache probe and the eviction are injected so the order is testable without weights. Returns
/// whether a generator was evicted.
#[cfg_attr(not(native_lane), allow(dead_code))]
pub(crate) async fn evict_generator_before_cold_load<Access, Evict>(
    access: impl FnOnce() -> Access,
    evict_generator: impl FnOnce() -> Evict,
) -> WorkerResult<bool>
where
    Access: std::future::Future<Output = WorkerResult<crate::cache_thread::CacheAccess>>,
    Evict: std::future::Future<Output = WorkerResult<bool>>,
{
    if access().await? == crate::cache_thread::CacheAccess::Warm {
        return Ok(false);
    }
    let evicted = evict_generator().await?;
    if evicted {
        tracing::info!(
            event = "minimax_h3_image_expander_evicted_cached_generator",
            "MiniMax-H3-Image prompt expansion: evicted the cached generator before loading the expander"
        );
    }
    Ok(evicted)
}

/// The edit target-ratio selection from the payload's `aspectRatio`: absent or `adaptive` lets
/// the model choose; a positive-integer `W:H` must come back unchanged.
pub(crate) fn aspect_selection(value: Option<&str>) -> WorkerResult<pe::AspectSelection> {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None => Ok(pe::AspectSelection::Adaptive),
        Some(value) if value.eq_ignore_ascii_case("adaptive") => Ok(pe::AspectSelection::Adaptive),
        Some(value) if pe::is_ratio(value) => Ok(pe::AspectSelection::Explicit(value.to_owned())),
        Some(value) => Err(WorkerError::InvalidPayload(format!(
            "prompt expansion aspectRatio must be `adaptive` or a positive-integer W:H, got \
             {value:?}"
        ))),
    }
}

/// One edit reference: the image the vision tower reads and its SOURCE pixel size, whose reduced
/// ratio is the `<Picture N>` metadata (the downscaled copy's rounding would change it).
pub(crate) fn edit_reference(
    image: gen_core::core_llm::ImageRef,
    source_size: (u32, u32),
) -> pe::EditReference {
    pe::EditReference {
        image,
        aspect_ratio: pe::reduced_ratio(source_size.0, source_size.1),
    }
}

/// The engine's request for this expansion.
pub(crate) fn build_request(
    files: &pe::PromptFiles,
    prompt: &str,
    references: &[pe::EditReference],
    aspect: &pe::AspectSelection,
) -> WorkerResult<gen_core::core_llm::TextLlmRequest> {
    match Workflow::for_reference_count(references.len()) {
        Workflow::TextToImage => pe::t2i_request(files, prompt),
        Workflow::Edit => pe::edit_request(files, prompt, references, aspect),
    }
    .map_err(WorkerError::InvalidPayload)
}

/// Read and verify the prompt files of `prompt_root`.
pub(crate) fn load_prompt_files(prompt_root: &Path) -> WorkerResult<pe::PromptFiles> {
    #[cfg(test)]
    if let Some(digest) = tests::PINNED_MANIFEST_OVERRIDE.with(|pin| pin.borrow().clone()) {
        return pe::PromptFiles::load_with_manifest_digest(prompt_root, &digest)
            .map_err(WorkerError::InvalidPayload);
    }
    pe::PromptFiles::load_verified(prompt_root).map_err(WorkerError::InvalidPayload)
}

/// Parse the cleaned answer for `workflow`.
pub(crate) fn parse_reply(
    answer: &str,
    workflow: Workflow,
    allowed_ratios: &[String],
    aspect: &pe::AspectSelection,
) -> Result<pe::Expansion, String> {
    match workflow {
        Workflow::TextToImage => pe::parse_t2i(answer, allowed_ratios),
        Workflow::Edit => pe::parse_edit(answer, aspect),
    }
}

/// The error for a reply the decoder ran out of budget on — distinct from a parse failure, because
/// the model reasons first and emits the JSON object last, so a short budget loses the object.
pub(crate) fn truncated_reply_error(max_new_tokens: u32) -> String {
    format!(
        "the MiniMax-H3-Image prompt expander's reply was cut off at {max_new_tokens} tokens \
         before it finished — raise `maxNewTokens` and try again (the model reasons first and \
         emits the JSON object last, so a budget that runs out loses the whole object rather than \
         shortening it)"
    )
}

/// The suggestion block the job result carries beside `refinedPrompt`:
/// `{ whRatio, workflow, rewriterModelId, tier? }`, `rewriterModelId` being the expander model that
/// ran. The UI maps `whRatio` to a size of the image model's own envelope and offers it; nothing
/// here changes the request's size.
pub(crate) fn suggestion_result_block(
    expansion: &pe::Expansion,
    workflow: Workflow,
    model_id: &str,
    tier: Option<&str>,
) -> Map<String, Value> {
    let mut block = Map::new();
    block.insert(
        "whRatio".to_owned(),
        Value::from(expansion.aspect_ratio.clone()),
    );
    block.insert("workflow".to_owned(), Value::from(workflow.name()));
    block.insert("rewriterModelId".to_owned(), Value::from(model_id));
    if let Some(tier) = tier {
        block.insert("tier".to_owned(), Value::from(tier));
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sha2::{Digest, Sha256};

    thread_local! {
        /// The manifest digest [`load_prompt_files`] verifies against on THIS test thread in
        /// place of the pinned upstream one, so a job-level test can run on a synthetic prompt
        /// tree (no MiniMax prompt text is vendored). Unset, the pinned digest applies.
        pub(super) static PINNED_MANIFEST_OVERRIDE: std::cell::RefCell<Option<String>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Sets [`PINNED_MANIFEST_OVERRIDE`] for its lifetime and clears it on drop.
    struct ManifestOverride;

    impl ManifestOverride {
        fn set(digest: String) -> Self {
            PINNED_MANIFEST_OVERRIDE.with(|pin| *pin.borrow_mut() = Some(digest));
            Self
        }
    }

    impl Drop for ManifestOverride {
        fn drop(&mut self) {
            PINNED_MANIFEST_OVERRIDE.with(|pin| *pin.borrow_mut() = None);
        }
    }

    const PE_REPO: &str = "MiniMaxAI/MiniMax-H3-Image-PE";
    const PE_REVISION: &str = "35603d4b63949afa12cb098227fffefedf081afa";

    fn sha256(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// The shipped catalog entry, every platform's rows.
    fn catalog_entry() -> JsonObject {
        let manifest =
            sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc").unwrap();
        manifest["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["id"] == PE_MODEL_ID)
            .and_then(Value::as_object)
            .cloned()
            .expect("the built-in catalog declares the expander")
    }

    /// The shipped catalog entry as this host's worker receives it (rows narrowed to its OS).
    fn shipped_entry() -> JsonObject {
        let mut entry = Value::Object(catalog_entry());
        sceneworks_core::model_artifacts::artifact_selection::retain_downloads_for_os(
            &mut entry,
            std::env::consts::OS,
        );
        entry.as_object().unwrap().clone()
    }

    /// The catalog entry as a macOS worker receives it: the published q8/q4 MLX rows are macOS-only,
    /// so tests of those rows resolve as a macOS worker (`resolve_dirs_for_os(.., "macos")`) and
    /// must not depend on the host this suite runs on.
    fn macos_entry() -> JsonObject {
        let mut entry = Value::Object(catalog_entry());
        sceneworks_core::model_artifacts::artifact_selection::retain_downloads_for_os(
            &mut entry, "macos",
        );
        entry.as_object().unwrap().clone()
    }

    fn snapshot(data: &Path, repo: &str, revision: &str) -> PathBuf {
        let dir = crate::huggingface_hub_cache_dir(data)
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn settings(data: &Path) -> Settings {
        Settings::for_test(data.to_path_buf())
    }

    /// A data directory whose Hugging Face cache is private to the test. `HF_HOME` / `HF_HUB_CACHE`
    /// take precedence over the data directory, so without the guard a fixture would write into
    /// the host's real cache.
    fn isolated() -> (
        tempfile::TempDir,
        sceneworks_core::hf_home::test_support::CacheRootGuard,
    ) {
        let data = tempfile::tempdir().unwrap();
        let guard = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
            data.path().to_path_buf(),
            data.path().join("hub"),
        );
        (data, guard)
    }

    /// A synthetic prompt tree: this repository's own text, with a manifest that lists each file's
    /// digest. Returns the manifest's digest. No MiniMax prompt text is used anywhere.
    fn synthetic_prompt_tree(root: &Path) -> String {
        let files = [
            ("prompt/t2i/system_prompt.txt", "Synthetic system turn.\n\n"),
            (
                "prompt/t2i/task_prompt.txt",
                "Expand this prompt: {}\nAnswer with JSON.",
            ),
            (
                "prompt/t2i/request_params.json",
                r#"{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 4096,
                    "reasoning_effort": "xhigh",
                    "chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true},
                    "response_format": {"type": "json_schema", "json_schema": {"name": "t2i_reply",
                        "strict": true, "schema": {"type": "object",
                        "properties": {"expanded_caption": {"type": "string"},
                            "aspect_ratio": {"type": "string", "enum": ["1:1", "4:3", "16:9"]}},
                        "required": ["expanded_caption", "aspect_ratio"],
                        "additionalProperties": false}}}}"#,
            ),
            ("prompt/edit/task_prompt.txt", "Synthetic edit task."),
        ];
        let mut manifest = json!({ "t2i": { "files": {} }, "edit": { "files": {} } });
        for (rel, body) in files {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, body).unwrap();
            let mode = if rel.starts_with("prompt/edit/") {
                "edit"
            } else {
                "t2i"
            };
            manifest[mode]["files"][rel] = json!({ "sha256": sha256(body.as_bytes()) });
        }
        let bytes = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(root.join(PROMPT_MANIFEST), &bytes).unwrap();
        sha256(&bytes)
    }

    #[test]
    fn the_shipped_entry_declares_every_tier_and_the_prompt_co_requisite() {
        let entry = catalog_entry();
        let value = entry_value(&entry);
        let nvfp4 = model_download_for_variant(&value, "nvfp4").expect("nvfp4 row");
        assert_eq!(row_identity(&nvfp4), Some((PE_REPO, PE_REVISION)));
        assert_eq!(row_tier_subdir(&nvfp4), None);
        assert!(!is_pending_artifact_download(&nvfp4));
        for tier in ["bf16", "q8", "q4"] {
            let row = model_download_for_variant(&value, tier).expect("converted row");
            // Every converted tier is published and pinned to a hub revision. bf16 is the dense
            // `bf16/` snapshot; the macOS q8/q4 rows are the fully quantized `q8-full/` / `q4-full/`
            // packs (SW-7202) pinned to their publication revision.
            assert!(!is_pending_artifact_download(&row), "{tier}");
            let expected = if tier == "bf16" {
                tier.to_owned()
            } else {
                format!("{tier}-full")
            };
            assert_eq!(row_tier_subdir(&row).as_deref(), Some(expected.as_str()));
        }
        // The engine's pinned revision is the catalog's, for the weights and the prompts alike.
        assert_eq!(pe::REPO, PE_REPO);
        assert_eq!(pe::REVISION, PE_REVISION);
        assert_eq!(pe::MODEL_ID, PE_MODEL_ID);
        assert_eq!(
            pe::MAX_REFERENCES,
            sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPAND_MAX_REFERENCES
        );
    }

    #[test]
    fn a_pending_tier_resolves_to_nothing_and_its_absence_says_how_to_install() {
        let (data, _cache) = isolated();
        let entry = macos_entry();
        // Nothing installed: the error names the Models screen and that H3-Image needs none of it.
        let error =
            resolve_dirs_for_os(&settings(data.path()), &entry, None, None, "macos").unwrap_err();
        assert!(error.to_string().contains("Models screen"), "{error}");
        // An explicit pending tier is refused by name, never substituted.
        let error = resolve_dirs_for_os(&settings(data.path()), &entry, Some("q4"), None, "macos")
            .unwrap_err();
        assert!(error.to_string().contains("tier q4"), "{error}");
        let error = resolve_dirs_for_os(&settings(data.path()), &entry, Some("q6"), None, "macos")
            .unwrap_err();
        assert!(error.to_string().contains("no `q6` tier"), "{error}");
    }

    #[test]
    fn a_published_converted_tier_loads_its_weights_and_reads_upstream_prompts() {
        let (data, _cache) = isolated();
        let mut entry = macos_entry();
        // Stand in for the terminal item's publication of the q8 tier.
        let published = "1111111111111111111111111111111111111111";
        for row in entry["downloads"].as_array_mut().unwrap() {
            if row["variant"] == "q8" {
                row["revision"] = json!(published);
                row.as_object_mut().unwrap().remove("pendingArtifact");
            }
        }
        let tier_snapshot = snapshot(data.path(), "SceneWorks/minimax-h3-image-pe", published);
        std::fs::create_dir_all(tier_snapshot.join("q8-full")).unwrap();
        std::fs::write(tier_snapshot.join("q8-full").join("config.json"), "{}").unwrap();
        let upstream = snapshot(data.path(), PE_REPO, PE_REVISION);
        synthetic_prompt_tree(&upstream);

        let dirs =
            resolve_dirs_for_os(&settings(data.path()), &entry, None, None, "macos").unwrap();
        assert_eq!(dirs.weights, tier_snapshot.join("q8-full"));
        assert_eq!(dirs.prompt_root, upstream);
        assert_eq!(dirs.tier, Some("q8"));
        // The converted tier never carries `prompt/` of its own.
        assert!(!dirs.weights.join(PROMPT_MANIFEST).exists());
    }

    /// A CUDA (Windows/Linux) host receives the Candle layout of each converted tier — the
    /// `candle-{tier}/` directory — and a macOS host the MLX `{tier}/` one, from the same variant.
    /// Weights-free: only a `config.json` marker per directory.
    #[test]
    fn each_host_os_resolves_its_own_layout_of_a_converted_tier() {
        for (os, q8_dir) in [
            ("macos", "q8-full"),
            ("windows", "candle-q8-full"),
            ("linux", "candle-q8-full"),
        ] {
            let (data, _cache) = isolated();
            let mut entry = Value::Object(catalog_entry());
            sceneworks_core::model_artifacts::artifact_selection::retain_downloads_for_os(
                &mut entry, os,
            );
            let entry = entry.as_object().unwrap().clone();
            let row = model_download_for_variant(&entry_value(&entry), "q8").expect("q8 row");
            let (repo, revision) = row_identity(&row).unwrap();
            assert_eq!(row_tier_subdir(&row).as_deref(), Some(q8_dir), "{os}");
            assert!(!is_pending_artifact_download(&row), "{os}");
            let tier_snapshot = snapshot(data.path(), repo, revision);
            // Both layouts exist on disk; only the host's own may be chosen.
            for directory in ["q8-full", "candle-q8-full"] {
                std::fs::create_dir_all(tier_snapshot.join(directory)).unwrap();
                std::fs::write(tier_snapshot.join(directory).join("config.json"), "{}").unwrap();
            }
            synthetic_prompt_tree(&snapshot(data.path(), PE_REPO, PE_REVISION));
            let dirs =
                resolve_dirs_for_os(&settings(data.path()), &entry, Some("q8"), None, os).unwrap();
            assert_eq!(dirs.weights, tier_snapshot.join(q8_dir), "{os}");
            assert_eq!(dirs.tier, Some("q8"));
        }
    }

    /// The API forwards the expander's COMPLETE cross-platform entry (`resolve_model_manifest_entry`
    /// keeps every platform's rows), where each converted tier lists the macOS `{tier}/` row before
    /// the Candle `candle-{tier}/` row under the same variant. A Windows/Linux worker must still
    /// load its own layout, so the worker narrows the entry to its OS before choosing a row.
    #[test]
    fn an_unnarrowed_entry_resolves_the_executing_hosts_layout() {
        for (os, q4_dir) in [
            ("macos", "q4-full"),
            ("windows", "candle-q4-full"),
            ("linux", "candle-q4-full"),
        ] {
            let (data, _cache) = isolated();
            let entry = catalog_entry();
            let mut host = Value::Object(entry.clone());
            sceneworks_core::model_artifacts::artifact_selection::retain_downloads_for_os(
                &mut host, os,
            );
            let row = model_download_for_variant(&host, "q4").expect("q4 row");
            let (repo, revision) = row_identity(&row).unwrap();
            let tier_snapshot = snapshot(data.path(), repo, revision);
            // A shared cache holding both layouts: only the host's own may be chosen.
            for directory in ["q4-full", "candle-q4-full"] {
                std::fs::create_dir_all(tier_snapshot.join(directory)).unwrap();
                std::fs::write(tier_snapshot.join(directory).join("config.json"), "{}").unwrap();
            }
            synthetic_prompt_tree(&snapshot(data.path(), PE_REPO, PE_REVISION));
            for explicit in [Some("q4"), None] {
                let dirs = resolve_dirs_for_os(&settings(data.path()), &entry, explicit, None, os)
                    .unwrap();
                assert_eq!(
                    dirs.weights,
                    tier_snapshot.join(q4_dir),
                    "{os} {explicit:?}"
                );
                assert_eq!(dirs.tier, Some("q4"), "{os}");
            }
        }
    }

    /// PR #3418 review: render then expand. A cold expander load evicts the cached generator
    /// FIRST (before the load), and a warm expander evicts nothing. Weights-free: the cache probe
    /// and the eviction are recording stubs.
    #[tokio::test]
    async fn a_cold_expander_load_evicts_the_cached_generator_first() {
        use crate::cache_thread::CacheAccess;
        use std::sync::{Arc, Mutex};
        for (access, expect_evict) in [(CacheAccess::Cold, true), (CacheAccess::Warm, false)] {
            let log = Arc::new(Mutex::new(Vec::<&str>::new()));
            let (probe_log, evict_log) = (log.clone(), log.clone());
            let evicted = evict_generator_before_cold_load(
                move || async move {
                    probe_log.lock().unwrap().push("probe");
                    Ok(access)
                },
                move || async move {
                    evict_log.lock().unwrap().push("evict");
                    Ok(true)
                },
            )
            .await
            .unwrap();
            // The load runs after this returns.
            log.lock().unwrap().push("load");
            assert_eq!(evicted, expect_evict, "{access:?}");
            let expected: &[&str] = if expect_evict {
                &["probe", "evict", "load"]
            } else {
                &["probe", "load"]
            };
            assert_eq!(*log.lock().unwrap(), expected, "{access:?}");
        }
    }

    /// SW-7166: the expansion JOB — not only the helper above — stages the cached image
    /// generator out before its cold expander load. The job runs against a local stub API and a
    /// weightless local expander snapshot (prompt files only), so the load itself fails after the
    /// eviction point; no model runs. Holds the crate's env lock, as every test touching the
    /// process-wide generator cache does.
    #[cfg(native_lane)]
    #[test]
    fn an_expansion_job_stages_the_cached_generator_out_before_its_cold_load() {
        use crate::generator_cache::{global_generator_resident, load_stub_into_global_cache};
        use axum::{
            routing::{get, post},
            Json, Router,
        };
        let _env = crate::test_env::env_lock();
        let temporary = tempfile::tempdir().unwrap();
        let data = temporary.path().canonicalize().unwrap();
        let _cache = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
            data.clone(),
            data.join("hub"),
        );
        let local = data.join("local-pe");
        std::fs::create_dir_all(&local).unwrap();
        let digest = synthetic_prompt_tree(&local);
        let resident_weights = data.join("resident-generator");
        std::fs::create_dir_all(&resident_weights).unwrap();

        let job_value = json!({
            "id": "job_h3_expand_stage_out", "type": "prompt_refine", "status": "running",
            "payload": {
                "task": sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPAND_TASK,
                "prompt": "a lighthouse",
                "modelId": sceneworks_core::contracts::MINIMAX_H3_IMAGE_MODEL_ID,
                "model": local.to_string_lossy(),
                "modelManifestEntry": shipped_entry(),
            },
            "result": {}, "requestedGpu": "auto", "workerId": "w", "progress": 0,
            "stage": "running", "message": "", "attempts": 1, "cancelRequested": false,
            "createdAt": "2026-11-17T00:00:00Z", "updatedAt": "2026-11-17T00:00:00Z"
        });
        let job: crate::JobSnapshot = serde_json::from_value(job_value.clone()).unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // The job's own task runs on this thread (the current-thread runtime above).
        let _pin = ManifestOverride::set(digest);
        let (evicted, result) = runtime.block_on(async move {
            let reply = job_value.clone();
            let app = Router::new()
                .route(
                    "/api/v1/workers/:worker/heartbeat",
                    post(|| async { Json(json!({})) }),
                )
                .route(
                    "/api/v1/jobs/:job",
                    get({
                        let reply = reply.clone();
                        move || async move { Json(reply) }
                    }),
                )
                .route(
                    "/api/v1/jobs/:job/progress",
                    post(move || async move { Json(reply) }),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let mut settings = crate::test_env::offline_settings();
            settings.api_url = format!("http://{address}");
            settings.worker_id = "w".into();
            settings.data_dir = data.clone();
            let api = crate::ApiClient::new(&settings);

            load_stub_into_global_cache(&resident_weights)
                .await
                .unwrap();
            assert!(
                global_generator_resident().await.is_some(),
                "the generator is resident"
            );
            let result =
                crate::prompt_refine_jobs::run_prompt_refine_job(&api, &settings, &job).await;
            server.abort();
            (global_generator_resident().await.is_none(), result)
        });
        assert!(result.is_err(), "a weightless expander cannot load");
        assert!(
            evicted,
            "the expansion job evicted the generator before its cold load ({result:?})"
        );
    }

    /// The describer's shipped entry as the API forwards it for an expansion (SW-7208): every
    /// platform's tiers plus the expander's `prompt` co-requisite row.
    fn describer_expansion_entry() -> JsonObject {
        let mut entry = crate::test_env::shipped_catalog_entry(DESCRIBER_MODEL_ID);
        let prompt = catalog_entry()["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["coRequisite"] == true && row["componentId"] == PROMPT_COMPONENT)
            .cloned()
            .expect("the expander's prompt co-requisite");
        entry
            .get_mut("downloads")
            .and_then(Value::as_array_mut)
            .unwrap()
            .push(prompt);
        entry
    }

    /// SW-7208 (AC2/AC3): the abliterated describer runs the expansion from its OWN tier folder on
    /// each host's layout, with MiniMax's prompt files read from the expander's pinned upstream
    /// snapshot — no MiniMax-H3-Image-PE weights anywhere. Its tiers are its own (no NVFP4), and
    /// without the prompt files it says how to get them alone. Weights-free.
    #[test]
    fn the_abliterated_describer_expands_from_its_own_tier_with_minimax_prompt_files() {
        for (os, q4_dir) in [
            ("macos", "q4-full"),
            ("windows", "q4-full"),
            ("linux", "q4-full"),
        ] {
            let (data, _cache) = isolated();
            let entry = describer_expansion_entry();
            let mut host = Value::Object(entry.clone());
            sceneworks_core::model_artifacts::artifact_selection::retain_downloads_for_os(
                &mut host, os,
            );
            let row = model_download_for_variant(&host, "q4").expect("describer q4 row");
            let (repo, revision) = row_identity(&row).unwrap();
            let tier_snapshot = snapshot(data.path(), repo, revision);
            std::fs::create_dir_all(tier_snapshot.join(q4_dir)).unwrap();
            std::fs::write(tier_snapshot.join(q4_dir).join("config.json"), "{}").unwrap();

            // The tier alone: the prompt files are missing, and the error names them alone.
            let error = resolve_dirs_for_os(&settings(data.path()), &entry, None, None, os)
                .unwrap_err()
                .to_string();
            assert!(error.contains("prompt files"), "{os}: {error}");
            assert!(error.contains("no expander weights"), "{os}: {error}");

            let upstream = snapshot(data.path(), PE_REPO, PE_REVISION);
            synthetic_prompt_tree(&upstream);
            for explicit in [Some("q4"), None] {
                let dirs = resolve_dirs_for_os(&settings(data.path()), &entry, explicit, None, os)
                    .unwrap();
                assert_eq!(dirs.model_id, DESCRIBER_MODEL_ID, "{os}");
                assert_eq!(
                    dirs.weights,
                    tier_snapshot.join(q4_dir),
                    "{os} {explicit:?}"
                );
                assert_eq!(dirs.prompt_root, upstream, "{os}");
                assert_eq!(dirs.tier, Some("q4"), "{os}");
            }
            assert_eq!(
                repo,
                if os == "macos" {
                    sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_MLX_REPO
                } else {
                    sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_CANDLE_REPO
                },
                "{os}: the describer's own re-host, never the expander's"
            );
            // An expander-only tier is not the describer's: refused by name.
            let error =
                resolve_dirs_for_os(&settings(data.path()), &entry, Some("nvfp4"), None, os)
                    .unwrap_err();
            assert!(error.to_string().contains("no `nvfp4` tier"), "{error}");
        }
        assert_eq!(
            default_repo(DESCRIBER_MODEL_ID),
            sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_REPO
        );
        assert_eq!(default_repo(PE_MODEL_ID), PE_REPO);
    }

    #[test]
    fn a_tier_without_its_prompt_files_is_refused_with_the_install_step() {
        let (data, _cache) = isolated();
        let entry = shipped_entry();
        let local = data.path().join("local-pe");
        std::fs::create_dir_all(&local).unwrap();
        let error =
            resolve_dirs(&settings(data.path()), &entry, None, Some(local.clone())).unwrap_err();
        assert!(error.to_string().contains("prompt files"), "{error}");
        // An explicit local snapshot that carries its own `prompt/` reads it.
        synthetic_prompt_tree(&local);
        let dirs = resolve_dirs(&settings(data.path()), &entry, None, Some(local.clone())).unwrap();
        assert_eq!(dirs.prompt_root, local);
        assert_eq!(dirs.tier, None);
    }

    #[test]
    fn the_request_needs_the_expanders_own_catalog_entry() {
        let (data, _cache) = isolated();
        let mut entry = shipped_entry();
        entry.insert("id".to_owned(), json!("qwen_image_2_1_pe_t2i"));
        let error = resolve_dirs(&settings(data.path()), &entry, None, None).unwrap_err();
        assert!(error.to_string().contains("modelManifestEntry"), "{error}");
    }

    /// SW-7204: the expansion request the job runs is the engine's, sending `auto` checked
    /// linearly (never an explicit block draft, never tree checking), its recipe untouched.
    #[test]
    fn the_expansion_request_the_job_runs_sends_auto_checked_linearly() {
        use gen_core::core_llm::{CancelFlag, KvCompressionPolicy, Speculative, SpeculativeVerify};
        let dir = tempfile::tempdir().unwrap();
        let digest = synthetic_prompt_tree(dir.path());
        let files = pe::PromptFiles::load_with_manifest_digest(dir.path(), &digest).unwrap();
        let engine = build_request(
            &files,
            "a harbour at dusk",
            &[],
            &pe::AspectSelection::Adaptive,
        )
        .unwrap();
        let request = crate::prompt_refine_jobs::prepare_expansion_request(
            engine.clone(),
            CancelFlag::new(),
            KvCompressionPolicy::default(),
        );
        assert_eq!(
            request.constraint, engine.constraint,
            "the reply schema is kept"
        );
        assert_eq!(request.speculative, Some(Speculative::Auto));
        assert_eq!(request.speculative_verify, SpeculativeVerify::Linear);
        assert_eq!(request.messages.len(), engine.messages.len());
        assert_eq!(request.max_new_tokens, engine.max_new_tokens);
        assert_eq!(request.sampling.temperature, engine.sampling.temperature);
    }

    #[test]
    fn a_verified_prompt_tree_builds_both_workflows_and_a_changed_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let digest = synthetic_prompt_tree(dir.path());
        let files = pe::PromptFiles::load_with_manifest_digest(dir.path(), &digest).unwrap();

        // Text-to-image: the system and task turns, the published sampling recipe.
        let request = build_request(
            &files,
            "a harbour at dusk",
            &[],
            &pe::AspectSelection::Adaptive,
        )
        .unwrap();
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.max_new_tokens, 4096);
        assert_eq!(request.sampling.temperature, 1.0);
        assert_eq!(request.sampling.top_p, 0.95);
        assert_eq!(request.sampling.top_k, 20);

        // Edit: one interleaved user turn carrying the references in order, each labelled with
        // its SOURCE ratio rather than the downscaled copy's.
        let image = gen_core::core_llm::ImageRef::new(2, 1, vec![0; 6]).unwrap();
        let references = [
            edit_reference(image.clone(), (1920, 1080)),
            edit_reference(image, (1000, 1000)),
        ];
        assert_eq!(references[0].aspect_ratio.as_deref(), Some("16:9"));
        let request = build_request(
            &files,
            "Put <Picture 1> on <Picture 2>",
            &references,
            &pe::AspectSelection::Adaptive,
        )
        .unwrap();
        assert_eq!(request.messages.len(), 1);
        let text: String = request.messages[0]
            .content
            .iter()
            .filter_map(|part| match part {
                gen_core::core_llm::Content::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(text.contains("<Picture 1> [aspect_ratio 16:9]"), "{text}");
        assert!(text.contains("<Picture 2> [aspect_ratio 1:1]"), "{text}");
        let images = request.messages[0]
            .content
            .iter()
            .filter(|part| matches!(part, gen_core::core_llm::Content::Image(_)))
            .count();
        assert_eq!(images, 2);

        // One changed byte in a prompt file and the tree is refused.
        std::fs::write(dir.path().join("prompt/edit/task_prompt.txt"), "Changed.").unwrap();
        assert!(pe::PromptFiles::load_with_manifest_digest(dir.path(), &digest).is_err());
        // And the production loader refuses any tree whose manifest is not the pinned revision's.
        assert!(load_prompt_files(dir.path()).is_err());
    }

    // The expander thinks first, so the worker's general rule for thinking runs
    // (`json_decode_constraint`: no JSON mask) must not reach its request. The engine's reply
    // schema is what makes the backends refuse an answer that is not the two-field object, and an
    // end of decode inside the reasoning; without it the job failed on both.
    #[test]
    fn the_job_runs_the_engines_reply_schema_not_the_thinking_rule() {
        use gen_core::core_llm::{Constraint, KvCompressionPolicy, ThinkingMode};

        let dir = tempfile::tempdir().unwrap();
        let digest = synthetic_prompt_tree(dir.path());
        let files = pe::PromptFiles::load_with_manifest_digest(dir.path(), &digest).unwrap();
        let image = gen_core::core_llm::ImageRef::new(2, 1, vec![0; 6]).unwrap();
        let edit = [edit_reference(image, (1920, 1080))];
        for references in [&[][..], &edit[..]] {
            let built = build_request(
                &files,
                "a harbour at dusk",
                references,
                &pe::AspectSelection::Adaptive,
            )
            .unwrap();
            assert_eq!(built.thinking, ThinkingMode::Enabled);
            assert!(
                matches!(built.constraint, Some(Constraint::JsonSchema(_))),
                "the engine constrains the reply to its schema"
            );
            let expected = built.constraint.clone();
            let cancel = gen_core::core_llm::CancelFlag::new();
            let run = crate::prompt_refine_jobs::adopt_prepared_request(
                built,
                cancel.clone(),
                KvCompressionPolicy::default(),
            );
            assert_eq!(run.constraint, expected, "the job keeps the reply schema");
            assert!(run.progress.is_some());
            cancel.cancel();
            assert!(run.cancel.is_cancelled(), "the job's cancel flag drives it");
        }
    }

    #[test]
    fn the_reply_becomes_an_expansion_and_a_size_offer_never_a_forced_size() {
        let ratios = vec!["1:1".to_owned(), "16:9".to_owned()];
        let expansion = parse_reply(
            r#"{"expanded_caption": "A wide harbour at dusk, boats on a stone quay.", "aspect_ratio": "16:9"}"#,
            Workflow::TextToImage,
            &ratios,
            &pe::AspectSelection::Adaptive,
        )
        .unwrap();
        assert_eq!(
            expansion.expanded_caption,
            "A wide harbour at dusk, boats on a stone quay."
        );
        let block = suggestion_result_block(
            &expansion,
            Workflow::TextToImage,
            PE_MODEL_ID,
            Some("nvfp4"),
        );
        assert_eq!(block.get("whRatio"), Some(&json!("16:9")));
        assert_eq!(block.get("workflow"), Some(&json!("t2i")));
        assert_eq!(block.get("rewriterModelId"), Some(&json!(PE_MODEL_ID)));
        // The block names a ratio only; the size is offered by the UI from the image model's own
        // presets, so nothing here can rewrite the request's geometry.
        assert!(!block.contains_key("resolution"));

        // A ratio outside the published enum, or extra keys, are refused, not repaired.
        assert!(parse_reply(
            r#"{"expanded_caption": "x", "aspect_ratio": "5:4"}"#,
            Workflow::TextToImage,
            &ratios,
            &pe::AspectSelection::Adaptive,
        )
        .is_err());
        // An explicit edit selection must come back unchanged.
        let explicit = pe::AspectSelection::Explicit("3:2".to_owned());
        assert!(parse_reply(
            r#"{"expanded_caption": "x", "target_aspect_ratio": "4:3"}"#,
            Workflow::Edit,
            &ratios,
            &explicit,
        )
        .is_err());
    }

    #[test]
    fn the_aspect_selection_is_adaptive_or_an_explicit_ratio() {
        assert_eq!(
            aspect_selection(None).unwrap(),
            pe::AspectSelection::Adaptive
        );
        assert_eq!(
            aspect_selection(Some("adaptive")).unwrap(),
            pe::AspectSelection::Adaptive
        );
        assert_eq!(
            aspect_selection(Some("21:9")).unwrap(),
            pe::AspectSelection::Explicit("21:9".to_owned())
        );
        assert!(aspect_selection(Some("wide")).is_err());
    }

    #[test]
    fn a_truncated_reply_names_the_ceiling() {
        let message = truncated_reply_error(4096);
        assert!(message.contains("cut off at 4096 tokens"), "{message}");
        assert!(message.contains("maxNewTokens"), "{message}");
    }

    #[test]
    fn only_minimax_h3_image_is_an_expansion_target() {
        assert!(is_expand_target_model("minimax_h3_image"));
        assert!(!is_expand_target_model("minimax_h3"));
        assert!(!is_expand_target_model("qwen_image_2_1"));
        assert_eq!(Workflow::for_reference_count(0), Workflow::TextToImage);
        assert_eq!(Workflow::for_reference_count(3), Workflow::Edit);
    }
}
