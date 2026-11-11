//! MiniMax-H3-Image prompt expansion (SW-7102, feature SW-F363).
//!
//! `MiniMaxAI/MiniMax-H3-Image-PE` expands a short prompt — or an edit instruction that names its
//! ordered references as `<Picture N>` — into the long caption MiniMax-H3-Image was trained on, and
//! proposes an aspect ratio. It is a Qwen3.8-27B fine-tune on the ordinary native TextLlm lane, so
//! [`crate::prompt_refine_jobs`] drives it as the `prompt_refine` task
//! [`EXPAND_TASK`]. The request assembly and the reply contract are the engine's
//! (`core_llm::minimax_h3_image_pe`); this module only finds the weights and the prompt files on
//! disk and shapes the job result.
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
#[cfg(any(test, native_lane))]
pub(crate) use sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPAND_TASK as EXPAND_TASK;
use sceneworks_core::contracts::MINIMAX_H3_IMAGE_MODEL_ID as TARGET_MODEL_ID;
/// The expander's catalog id.
pub(crate) use sceneworks_core::contracts::MINIMAX_H3_IMAGE_PE_MODEL_ID as PE_MODEL_ID;

/// The co-requisite component whose upstream snapshot holds `prompt/`.
const PROMPT_COMPONENT: &str = "prompt";
/// The prompt manifest inside a snapshot.
const PROMPT_MANIFEST: &str = "prompt/manifest.json";
/// Tiers in preference order: the published NVFP4 original first (where the device can run it),
/// then the converted tiers densest first.
const TIERS: [&str; 4] = ["nvfp4", "bf16", "q8", "q4"];

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
fn prompt_root(data_dir: &Path, entry: &JsonObject) -> WorkerResult<PathBuf> {
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
                "{PE_MODEL_ID}: the catalog entry declares no `{PROMPT_COMPONENT}` co-requisite — \
                 the model manifest entry is misconfigured"
            ))
        })?;
    let (repo, revision) = row_identity(row).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{PE_MODEL_ID}: the `{PROMPT_COMPONENT}` co-requisite declares no repo/revision"
        ))
    })?;
    crate::model_jobs::huggingface_pinned_snapshot_dir(data_dir, repo, revision)
        .filter(|root| root.join(PROMPT_MANIFEST).is_file())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{PE_MODEL_ID}: its prompt files ({repo} @ {revision}, `prompt/`) are not \
                 installed — install the MiniMax-H3-Image Prompt Expander from the Model Manager"
            ))
        })
}

/// Resolve the expander's weights and prompt directories from its catalog entry.
///
/// `explicit_tier` (the payload's `peTier`) is honoured or refused by name, never substituted;
/// with none, the first installed tier this host can run, in [`TIERS`] order. `override_dir` is an
/// explicit local snapshot (the real-weight smoke's `payload.model`): its own `prompt/` is used
/// when it has one, else the catalog's.
pub(crate) fn resolve_dirs(
    settings: &Settings,
    entry: &JsonObject,
    explicit_tier: Option<&str>,
    override_dir: Option<PathBuf>,
) -> WorkerResult<ExpanderDirs> {
    if entry.get("id").and_then(Value::as_str) != Some(PE_MODEL_ID) {
        return Err(WorkerError::InvalidPayload(format!(
            "prompt expansion needs the `{PE_MODEL_ID}` catalog entry on the job payload \
             (`modelManifestEntry`)"
        )));
    }
    let data_dir = settings.data_dir.as_path();
    if let Some(weights) = override_dir {
        let prompt_root = if weights.join(PROMPT_MANIFEST).is_file() {
            weights.clone()
        } else {
            prompt_root(data_dir, entry)?
        };
        return Ok(ExpanderDirs {
            weights,
            prompt_root,
            tier: None,
        });
    }
    let (tier, weights) = match explicit_tier {
        Some(requested) => {
            let tier = TIERS
                .into_iter()
                .find(|tier| tier.eq_ignore_ascii_case(requested.trim()))
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "{PE_MODEL_ID} has no `{requested}` tier; its tiers are {}",
                        TIERS.join(", ")
                    ))
                })?;
            if !tier_runs_here(tier) {
                return Err(WorkerError::InvalidPayload(format!(
                    "{PE_MODEL_ID} tier {tier} needs a Blackwell NVIDIA GPU (compute capability \
                     12.0 or newer); pick another tier"
                )));
            }
            let dir = tier_dir(data_dir, entry, tier).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{PE_MODEL_ID} tier {tier} is not installed or not published yet; install it \
                     in the Model Manager or pick an installed tier (an explicit tier selection is \
                     never substituted with another)"
                ))
            })?;
            (tier, dir)
        }
        None => TIERS
            .into_iter()
            .filter(|tier| tier_runs_here(tier))
            .find_map(|tier| tier_dir(data_dir, entry, tier).map(|dir| (tier, dir)))
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "the MiniMax-H3-Image Prompt Expander ({PE_MODEL_ID}) is not installed in a \
                     tier this host can run — MiniMax-H3-Image generates normally without it; open \
                     the Models screen to download it"
                ))
            })?,
    };
    Ok(ExpanderDirs {
        weights,
        prompt_root: prompt_root(data_dir, entry)?,
        tier: Some(tier),
    })
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
/// `{ whRatio, workflow, rewriterModelId, tier? }`. The UI maps `whRatio` to a size of the image
/// model's own envelope and offers it; nothing here changes the request's size.
pub(crate) fn suggestion_result_block(
    expansion: &pe::Expansion,
    workflow: Workflow,
    tier: Option<&str>,
) -> Map<String, Value> {
    let mut block = Map::new();
    block.insert(
        "whRatio".to_owned(),
        Value::from(expansion.aspect_ratio.clone()),
    );
    block.insert("workflow".to_owned(), Value::from(workflow.name()));
    block.insert("rewriterModelId".to_owned(), Value::from(PE_MODEL_ID));
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
                    "response_format": {"json_schema": {"schema": {"properties": {
                        "aspect_ratio": {"enum": ["1:1", "4:3", "16:9"]}}}}}}"#,
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
            assert!(is_pending_artifact_download(&row), "{tier}");
            assert_eq!(row_tier_subdir(&row).as_deref(), Some(tier));
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
        let entry = shipped_entry();
        // Nothing installed: the error names the Models screen and that H3-Image needs none of it.
        let error = resolve_dirs(&settings(data.path()), &entry, None, None).unwrap_err();
        assert!(error.to_string().contains("Models screen"), "{error}");
        // An explicit pending tier is refused by name, never substituted.
        let error = resolve_dirs(&settings(data.path()), &entry, Some("q4"), None).unwrap_err();
        assert!(error.to_string().contains("tier q4"), "{error}");
        let error = resolve_dirs(&settings(data.path()), &entry, Some("q6"), None).unwrap_err();
        assert!(error.to_string().contains("no `q6` tier"), "{error}");
    }

    #[test]
    fn a_published_converted_tier_loads_its_weights_and_reads_upstream_prompts() {
        let (data, _cache) = isolated();
        let mut entry = shipped_entry();
        // Stand in for the terminal item's publication of the q8 tier.
        let published = "1111111111111111111111111111111111111111";
        for row in entry["downloads"].as_array_mut().unwrap() {
            if row["variant"] == "q8" {
                row["revision"] = json!(published);
                row.as_object_mut().unwrap().remove("pendingArtifact");
            }
        }
        let tier_snapshot = snapshot(data.path(), "SceneWorks/minimax-h3-image-pe", published);
        std::fs::create_dir_all(tier_snapshot.join("q8")).unwrap();
        std::fs::write(tier_snapshot.join("q8").join("config.json"), "{}").unwrap();
        let upstream = snapshot(data.path(), PE_REPO, PE_REVISION);
        synthetic_prompt_tree(&upstream);

        let dirs = resolve_dirs(&settings(data.path()), &entry, None, None).unwrap();
        assert_eq!(dirs.weights, tier_snapshot.join("q8"));
        assert_eq!(dirs.prompt_root, upstream);
        assert_eq!(dirs.tier, Some("q8"));
        // The converted tier never carries `prompt/` of its own.
        assert!(!dirs.weights.join(PROMPT_MANIFEST).exists());
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
        let block = suggestion_result_block(&expansion, Workflow::TextToImage, Some("nvfp4"));
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
