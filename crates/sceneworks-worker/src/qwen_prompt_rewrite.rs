//! Qwen-Image 2.1 official prompt rewriting — sc-24113, epic 24107.
//!
//! # What upstream ships, and what this module is
//!
//! Qwen ships prompt rewriting for Qwen-Image 2.1 as a PAIR of separate checkpoints rather than as
//! anything inside the image model (`QwenLM/Qwen-Image-2.1` @ `fb7ae1d1f9611cd91524d03c53c5246b36ac8577`,
//! "Prompt Rewriting"):
//!
//! | repo | frozen revision | role |
//! |---|---|---|
//! | `Qwen/Qwen-Image-2.1-PE-T2I` | `f3ed7985c788ad75b3ab7223e0c4c51e2a43545b` | rewrite a text-to-image brief |
//! | `Qwen/Qwen-Image-2.1-PE-I2I` | `72927bc08afc99b7888ceb7d7d51a12db3700bbd` | rewrite an edit instruction, seeing the references |
//!
//! Both are fine-tuned Qwen3.5/3.6 (`model_type` `qwen3_5`) 9.41B VLM checkpoints under the Qwen
//! RESEARCH LICENSE (non-commercial). That is the SAME architecture the optional
//! `film_planner_qwen3_6_27b` catalog entry already runs, so they load on the EXISTING native
//! `core_llm::TextLlm` lane — `Architecture::Qwen35` → `Qwen35Config`/`Qwen35Model` on `mlx-llama`
//! (macOS) and `candle-llama` (Windows/CUDA), with `Qwen35VisionModel` + DeepStack serving the I2I
//! image turns. No second LLM runtime is built, and nothing here loads weights: this module is the
//! pure adapter — which rewriter, which template, what the reply means — that
//! [`crate::prompt_refine_jobs`] drives through the ordinary `prompt_refine` job.
//!
//! # The template is FROZEN BY REVISION, not vendored
//!
//! Each snapshot carries its own `system_prompt.txt`, and that file IS the published template. This
//! module reads it from the INSTALLED snapshot and verifies it against the SHA-256 of the pinned
//! revision's copy ([`T2I_SYSTEM_PROMPT_SHA256`] / [`I2I_SYSTEM_PROMPT_SHA256`]).
//!
//! Reading-and-verifying rather than copying the text into this repo is deliberate and is the
//! stronger of the two freezes:
//!
//! * The system prompts are Qwen Materials under a NON-COMMERCIAL licence whose §3 puts
//!   redistribution duties on anyone who passes them on. SceneWorks' whole posture for this family
//!   is that it never redistributes Qwen Materials — it pulls them from Qwen's own repository into
//!   the user's cache. Vendoring 10 KB and 18 KB of licensed prose into an AGPL repo would break
//!   that for no benefit.
//! * A digest check catches an upstream re-push (or a corrupted download) LOUDLY and by name. A
//!   vendored copy would silently keep using stale text against newer weights.
//!
//! # The reply contract
//!
//! Both rewriters emit a reasoning block and then ONE strictly-valid JSON object:
//!
//! ```text
//! T2I:  {"rewritten_prompt": "<long English description>", "wh_ratio": "3:2"}
//! I2I:  {"rewritten_prompt": "...", "wh_ratio": "", "ratio_follow": "<image1>"}
//! ```
//!
//! `wh_ratio` and `ratio_follow` are MUTUALLY EXCLUSIVE (the I2I system prompt states the rule):
//! the output either takes a stated aspect ratio or follows a named input image's resolution.
//!
//! # What this module deliberately does NOT do
//!
//! It never replaces the user's prompt. [`RewriteSuggestion`] is a SUGGESTION — the job result
//! carries the original and the rewrite side by side, the UI shows the rewrite in an editable box,
//! and nothing is applied until the user says so. The aspect-ratio suggestion is likewise offered
//! as a named preset the user may accept or ignore; it never re-writes the resolution control on
//! its own.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::{WorkerError, WorkerResult};

/// The `task` discriminator that selects this path on a `prompt_refine` job.
pub(crate) const REWRITE_TASK: &str = "qwen_image_rewrite";

/// The image model this rewriter pair belongs to. A rewrite request naming any other target model
/// is refused rather than served with the wrong template.
pub(crate) const REWRITE_TARGET_MODEL: &str = "qwen_image_2_1";

/// Text-to-image rewriter repo + frozen revision.
pub(crate) const PE_T2I_REPO: &str = "Qwen/Qwen-Image-2.1-PE-T2I";
pub(crate) const PE_T2I_REVISION: &str = "f3ed7985c788ad75b3ab7223e0c4c51e2a43545b";
/// Image-editing rewriter repo + frozen revision.
pub(crate) const PE_I2I_REPO: &str = "Qwen/Qwen-Image-2.1-PE-I2I";
pub(crate) const PE_I2I_REVISION: &str = "72927bc08afc99b7888ceb7d7d51a12db3700bbd";

/// The published template file inside each snapshot.
pub(crate) const SYSTEM_PROMPT_FILE: &str = "system_prompt.txt";

/// SHA-256 of `system_prompt.txt` at [`PE_T2I_REVISION`] (10,045 bytes).
pub(crate) const T2I_SYSTEM_PROMPT_SHA256: &str =
    "a77c9a06c59b120741141d9514b95682bb8761d02bec49ca61def7b2b3d9fb99";
/// SHA-256 of `system_prompt.txt` at [`PE_I2I_REVISION`] (18,344 bytes).
pub(crate) const I2I_SYSTEM_PROMPT_SHA256: &str =
    "e378fea686a1431581ba4c654d332ae96adad633f144ae738ec8ce9c4fd66439";

/// Upstream's published sampling for both rewriters (model cards: `do_sample=True`).
///
/// Frozen verbatim — these are part of the published recipe, not a SceneWorks tuning choice.
pub(crate) const REWRITE_TEMPERATURE: f32 = 1.0;
pub(crate) const REWRITE_TOP_P: f32 = 0.95;
pub(crate) const REWRITE_TOP_K: usize = 20;

/// Output budget for a rewrite.
///
/// Upstream publishes `max_new_tokens: 16256`, which is a CONTEXT ceiling for their batched vLLM
/// runner rather than a length the reply ever reaches: the object is one paragraph plus two short
/// fields, and it emits EOS far below any of these numbers. 4096 is the same generous ceiling the
/// worker's other JSON-emitting tasks take (`DEFAULT_CAPTION_MAX_NEW_TOKENS`), and it bounds a
/// runaway reasoning block on a single-request local decode instead of letting one occupy the GPU
/// for minutes. Callers may still raise it with `payload.maxNewTokens`.
pub(crate) const REWRITE_MAX_NEW_TOKENS: u32 = 4096;

/// Upper bound on reference images sent to the edit rewriter.
///
/// The same 10 the render takes (S3 edit contract): the rewrite must see EXACTLY the ordered list
/// the render will condition on, or its `ratio_follow: "<imageN>"` names a different picture than
/// the one the user attached.
pub(crate) const MAX_REWRITE_REFERENCES: usize = 10;

/// Which of the two rewriters a request selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rewriter {
    /// No references → the text-to-image rewriter.
    TextToImage,
    /// One or more ordered references → the editing rewriter.
    Editing,
}

impl Rewriter {
    /// Pick the rewriter from the REQUEST, never from a user-facing model picker (story AC:
    /// "Which rewriter applies is determined by the request (T2I vs edit with references)").
    pub(crate) fn for_reference_count(references: usize) -> Self {
        if references == 0 {
            Self::TextToImage
        } else {
            Self::Editing
        }
    }

    pub(crate) fn repo(self) -> &'static str {
        match self {
            Self::TextToImage => PE_T2I_REPO,
            Self::Editing => PE_I2I_REPO,
        }
    }

    pub(crate) fn revision(self) -> &'static str {
        match self {
            Self::TextToImage => PE_T2I_REVISION,
            Self::Editing => PE_I2I_REVISION,
        }
    }

    /// The catalog id, so the UI can ask whether THIS half is installed and offer that one
    /// download rather than both.
    pub(crate) fn catalog_id(self) -> &'static str {
        match self {
            Self::TextToImage => "qwen_image_2_1_pe_t2i",
            Self::Editing => "qwen_image_2_1_pe_i2i",
        }
    }

    fn expected_system_prompt_sha256(self) -> &'static str {
        match self {
            Self::TextToImage => T2I_SYSTEM_PROMPT_SHA256,
            Self::Editing => I2I_SYSTEM_PROMPT_SHA256,
        }
    }
}

/// Read and verify the frozen system prompt out of an installed snapshot.
///
/// A missing file, an unreadable file or a digest mismatch are all typed refusals naming the
/// revision: running a rewrite against text we did not freeze is exactly the silent behaviour
/// change the pin exists to prevent.
pub(crate) fn load_system_prompt(snapshot_dir: &Path, rewriter: Rewriter) -> WorkerResult<String> {
    load_system_prompt_verifying(
        snapshot_dir,
        rewriter,
        rewriter.expected_system_prompt_sha256(),
    )
}

/// [`load_system_prompt`] with the expected digest supplied explicitly.
///
/// Production always passes the frozen constant. Tests pass their own so BOTH halves of the freeze
/// — a matching snapshot is accepted, a re-pushed one is refused — are exercised without vendoring
/// the licensed 10 KB / 18 KB prose into this repo as a fixture.
pub(crate) fn load_system_prompt_verifying(
    snapshot_dir: &Path,
    rewriter: Rewriter,
    expected_sha256: &str,
) -> WorkerResult<String> {
    let path: PathBuf = snapshot_dir.join(SYSTEM_PROMPT_FILE);
    let bytes = std::fs::read(&path).map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "{} is missing its {SYSTEM_PROMPT_FILE} ({}): {error}. Re-download the rewriter — the \
             system prompt ships inside the snapshot and is not vendored in SceneWorks.",
            rewriter.repo(),
            path.display()
        ))
    })?;
    verify_system_prompt_digest(&bytes, rewriter, expected_sha256)?;
    String::from_utf8(bytes).map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "{}'s {SYSTEM_PROMPT_FILE} is not valid UTF-8: {error}",
            rewriter.repo()
        ))
    })
}

/// Verify a system-prompt blob against a frozen digest.
///
/// An EMPTY expected digest disables the check. Nothing in production passes one —
/// `the_frozen_digests_are_recorded` fails if either constant is ever blanked — but leaving the
/// branch explicit keeps the failure mode "no freeze" rather than "every snapshot refused" if a
/// future revision lands before its digest does.
pub(crate) fn verify_system_prompt_digest(
    bytes: &[u8],
    rewriter: Rewriter,
    expected: &str,
) -> WorkerResult<()> {
    if expected.is_empty() {
        return Ok(());
    }
    let actual = hex_digest(bytes);
    if actual == expected {
        return Ok(());
    }
    Err(WorkerError::InvalidPayload(format!(
        "{}'s {SYSTEM_PROMPT_FILE} does not match the frozen revision {}: expected SHA-256 \
         {expected}, found {actual}. The installed snapshot is not the one this rewrite path was \
         built against — re-download it at the pinned revision.",
        rewriter.repo(),
        rewriter.revision()
    )))
}

fn hex_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A parsed rewriter reply.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub(crate) struct RewriteSuggestion {
    /// The rewritten prompt. EDITABLE by the user; never applied on its own.
    pub(crate) rewritten_prompt: String,
    /// A stated aspect ratio ("3:2"), or empty when the output should follow an input image.
    pub(crate) wh_ratio: String,
    /// Which input image the output resolution should follow ("<image1>"), or empty when
    /// `wh_ratio` is stated. T2I never sets this.
    pub(crate) ratio_follow: String,
}

/// Parse a rewriter reply into a suggestion.
///
/// `text` is the RAW model reply: both checkpoints emit a reasoning block before the object, so the
/// caller's `clean_json_output` (which strips `<think>` blocks and a code fence, then takes the
/// outermost `{ … }` span) runs first and this takes the isolated object.
///
/// Refuses, rather than salvaging, when:
/// * the object does not parse, or is not an object;
/// * `rewritten_prompt` is missing, not a string, or blank after trimming — an empty rewrite
///   offered to the user as a suggestion is worse than an error, because Apply would silently
///   erase their prompt;
/// * both `wh_ratio` and `ratio_follow` are non-empty, which the I2I template forbids. Picking one
///   would be guessing at the model's intent about the output geometry.
pub(crate) fn parse_rewrite(text: &str) -> Result<RewriteSuggestion, String> {
    let parsed: Value = serde_json::from_str(text.trim())
        .map_err(|error| format!("the rewriter did not return a JSON object: {error}"))?;
    let object = parsed
        .as_object()
        .ok_or_else(|| "the rewriter returned JSON that is not an object".to_owned())?;

    let rewritten_prompt = string_field(object, "rewritten_prompt");
    if rewritten_prompt.is_empty() {
        return Err("the rewriter returned an empty `rewritten_prompt`".to_owned());
    }
    let wh_ratio = string_field(object, "wh_ratio");
    let ratio_follow = string_field(object, "ratio_follow");
    if !wh_ratio.is_empty() && !ratio_follow.is_empty() {
        return Err(format!(
            "the rewriter set both `wh_ratio` ({wh_ratio}) and `ratio_follow` ({ratio_follow}); \
             the template makes them mutually exclusive"
        ));
    }
    Ok(RewriteSuggestion {
        rewritten_prompt,
        wh_ratio,
        ratio_follow,
    })
}

fn string_field(object: &Map<String, Value>, key: &str) -> String {
    object
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned()
}

/// Qwen-Image 2.1's seven presets, keyed by the `wh_ratio` string the rewriters emit.
///
/// The right-hand values are the engine's own `PRESETS` (S1 contract, `config.rs`) and are exactly
/// the seven buckets `limits.resolutions` publishes, in the engine's order. Mapping to a PRESET
/// rather than computing a WxH from the ratio is the point: the suggestion the user is offered is
/// always a legal bucket they could have picked from the Aspect menu themselves.
const RATIO_PRESETS: &[(&str, &str)] = &[
    ("1:1", "2048x2048"),
    ("4:3", "2400x1792"),
    ("3:4", "1792x2400"),
    ("3:2", "2528x1696"),
    ("2:3", "1696x2528"),
    ("16:9", "2752x1536"),
    ("9:16", "1536x2752"),
];

/// The preset a suggestion's `wh_ratio` names, or `None`.
///
/// `None` covers three cases the caller treats identically — offer no aspect change:
/// * `wh_ratio` is empty (an edit rewrite that set `ratio_follow` instead);
/// * the ratio is one the engine has no preset for (the T2I template's own guidance only ever
///   emits the seven, but a model is not a parser);
/// * the reply is malformed.
///
/// Never an error: a rewrite whose prose is good but whose ratio we cannot map is still a useful
/// rewrite, and refusing the whole thing over an unrecognised ratio would throw the prompt away.
pub(crate) fn suggested_resolution(suggestion: &RewriteSuggestion) -> Option<&'static str> {
    let ratio = suggestion.wh_ratio.trim();
    if ratio.is_empty() {
        return None;
    }
    RATIO_PRESETS
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(ratio))
        .map(|(_, resolution)| *resolution)
}

/// Which attached reference the output resolution should follow, as a ZERO-BASED index.
///
/// `ratio_follow` is `"<image1>"`-style and 1-based, matching the template's own image numbering
/// (`<image1>` is the first reference in the ordered list). Returns `None` for an empty field, a
/// malformed one, or an index past the attached list — a reply naming `<image3>` when two
/// references were sent is a suggestion we cannot honour, not a reason to fail the rewrite.
pub(crate) fn follow_reference_index(
    suggestion: &RewriteSuggestion,
    reference_count: usize,
) -> Option<usize> {
    let raw = suggestion.ratio_follow.trim();
    let inner = raw
        .strip_prefix("<image")
        .and_then(|rest| rest.strip_suffix('>'))?;
    let ordinal: usize = inner.parse().ok()?;
    let index = ordinal.checked_sub(1)?;
    (index < reference_count).then_some(index)
}

/// The suggestion as the JSON block the job result carries beside `refinedPrompt`.
///
/// Shape: `{ whRatio, ratioFollow, resolution?, followsReferenceIndex?, rewriter, rewriterModelId }`.
/// `resolution` and `followsReferenceIndex` are OMITTED rather than null when there is nothing to
/// suggest, so the UI's "offer an aspect change" branch is a presence check.
pub(crate) fn suggestion_result_block(
    suggestion: &RewriteSuggestion,
    rewriter: Rewriter,
    reference_count: usize,
) -> Map<String, Value> {
    let mut block = Map::new();
    block.insert(
        "whRatio".to_owned(),
        Value::from(suggestion.wh_ratio.clone()),
    );
    block.insert(
        "ratioFollow".to_owned(),
        Value::from(suggestion.ratio_follow.clone()),
    );
    if let Some(resolution) = suggested_resolution(suggestion) {
        block.insert("resolution".to_owned(), Value::from(resolution));
    }
    if let Some(index) = follow_reference_index(suggestion, reference_count) {
        block.insert("followsReferenceIndex".to_owned(), Value::from(index));
    }
    block.insert(
        "rewriter".to_owned(),
        Value::from(match rewriter {
            Rewriter::TextToImage => "t2i",
            Rewriter::Editing => "i2i",
        }),
    );
    block.insert(
        "rewriterModelId".to_owned(),
        Value::from(rewriter.catalog_id()),
    );
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_request_picks_the_rewriter_no_references_means_text_to_image() {
        assert_eq!(Rewriter::for_reference_count(0), Rewriter::TextToImage);
        assert_eq!(Rewriter::for_reference_count(1), Rewriter::Editing);
        assert_eq!(Rewriter::for_reference_count(10), Rewriter::Editing);
        assert_eq!(Rewriter::TextToImage.repo(), PE_T2I_REPO);
        assert_eq!(Rewriter::Editing.repo(), PE_I2I_REPO);
        assert_eq!(Rewriter::TextToImage.catalog_id(), "qwen_image_2_1_pe_t2i");
        assert_eq!(Rewriter::Editing.catalog_id(), "qwen_image_2_1_pe_i2i");
    }

    #[test]
    fn a_t2i_reply_yields_an_editable_prompt_and_a_preset_aspect_suggestion() {
        let suggestion = parse_rewrite(
            r#"{"rewritten_prompt": "A wide harbour at dusk, ...", "wh_ratio": "16:9"}"#,
        )
        .unwrap();
        assert_eq!(suggestion.rewritten_prompt, "A wide harbour at dusk, ...");
        assert_eq!(suggestion.wh_ratio, "16:9");
        assert!(suggestion.ratio_follow.is_empty());
        // The aspect suggestion is a LEGAL BUCKET, not a computed WxH: 16:9 is the engine's own
        // 2752x1536 preset, which the user could have picked from the Aspect menu.
        assert_eq!(suggested_resolution(&suggestion), Some("2752x1536"));
    }

    #[test]
    fn every_engine_preset_is_reachable_from_a_ratio_the_rewriter_can_emit() {
        // The seven presets of the S1 contract, in the engine's own order. If the manifest's
        // `limits.resolutions` moves, this is the test that says the aspect suggestion can no
        // longer land on the bucket it names.
        let expected = [
            ("1:1", "2048x2048"),
            ("4:3", "2400x1792"),
            ("3:4", "1792x2400"),
            ("3:2", "2528x1696"),
            ("2:3", "1696x2528"),
            ("16:9", "2752x1536"),
            ("9:16", "1536x2752"),
        ];
        for (ratio, resolution) in expected {
            let suggestion = RewriteSuggestion {
                rewritten_prompt: "x".to_owned(),
                wh_ratio: ratio.to_owned(),
                ..Default::default()
            };
            assert_eq!(
                suggested_resolution(&suggestion),
                Some(resolution),
                "{ratio}"
            );
        }
    }

    #[test]
    fn an_edit_reply_follows_a_named_reference_instead_of_stating_a_ratio() {
        let suggestion = parse_rewrite(
            r#"{"rewritten_prompt": "Replace the sky with a sunset.", "wh_ratio": "", "ratio_follow": "<image2>"}"#,
        )
        .unwrap();
        assert!(suggestion.wh_ratio.is_empty());
        assert_eq!(suggestion.ratio_follow, "<image2>");
        // No aspect preset is offered — the geometry follows an attached image instead.
        assert_eq!(suggested_resolution(&suggestion), None);
        // `<image2>` is the SECOND reference in the ordered list, i.e. index 1. This is the
        // ordering claim: renumbering the references renames what the rewrite followed.
        assert_eq!(follow_reference_index(&suggestion, 3), Some(1));
    }

    #[test]
    fn a_follow_index_past_the_attached_list_is_dropped_rather_than_honoured() {
        let suggestion = RewriteSuggestion {
            rewritten_prompt: "x".to_owned(),
            ratio_follow: "<image3>".to_owned(),
            ..Default::default()
        };
        assert_eq!(follow_reference_index(&suggestion, 2), None);
        // ... and a malformed field is simply no suggestion, never a panic or a wrong index.
        for malformed in ["", "image1", "<image>", "<image0>", "<imageX>", "2"] {
            let suggestion = RewriteSuggestion {
                rewritten_prompt: "x".to_owned(),
                ratio_follow: malformed.to_owned(),
                ..Default::default()
            };
            assert_eq!(follow_reference_index(&suggestion, 10), None, "{malformed}");
        }
    }

    #[test]
    fn an_empty_rewritten_prompt_is_refused_so_apply_can_never_erase_the_users_prompt() {
        for body in [
            r#"{"wh_ratio": "1:1"}"#,
            r#"{"rewritten_prompt": "", "wh_ratio": "1:1"}"#,
            r#"{"rewritten_prompt": "   ", "wh_ratio": "1:1"}"#,
            r#"{"rewritten_prompt": 7}"#,
        ] {
            let error = parse_rewrite(body).unwrap_err();
            assert!(
                error.contains("empty `rewritten_prompt`"),
                "{body}: {error}"
            );
        }
    }

    #[test]
    fn both_geometry_fields_set_at_once_is_refused_rather_than_guessed() {
        let error = parse_rewrite(
            r#"{"rewritten_prompt": "x", "wh_ratio": "1:1", "ratio_follow": "<image1>"}"#,
        )
        .unwrap_err();
        assert!(error.contains("mutually exclusive"), "{error}");
    }

    #[test]
    fn a_non_object_or_unparseable_reply_is_refused_by_name() {
        assert!(parse_rewrite("not json").is_err());
        assert!(parse_rewrite("[1, 2]")
            .unwrap_err()
            .contains("not an object"));
    }

    #[test]
    fn an_unrecognised_ratio_loses_the_aspect_suggestion_but_keeps_the_rewrite() {
        // 21:9 is not one of the engine's seven buckets. The prose is still good; throwing the
        // whole rewrite away over a ratio we cannot map would be the wrong trade.
        let suggestion =
            parse_rewrite(r#"{"rewritten_prompt": "An ultrawide vista", "wh_ratio": "21:9"}"#)
                .unwrap();
        assert_eq!(suggestion.rewritten_prompt, "An ultrawide vista");
        assert_eq!(suggested_resolution(&suggestion), None);
    }

    #[test]
    fn the_result_block_omits_a_geometry_suggestion_it_cannot_make() {
        let plain = RewriteSuggestion {
            rewritten_prompt: "x".to_owned(),
            ..Default::default()
        };
        let block = suggestion_result_block(&plain, Rewriter::TextToImage, 0);
        assert!(!block.contains_key("resolution"));
        assert!(!block.contains_key("followsReferenceIndex"));
        assert_eq!(block.get("rewriter"), Some(&Value::from("t2i")));
        assert_eq!(
            block.get("rewriterModelId"),
            Some(&Value::from("qwen_image_2_1_pe_t2i"))
        );

        let ratio = RewriteSuggestion {
            rewritten_prompt: "x".to_owned(),
            wh_ratio: "4:3".to_owned(),
            ..Default::default()
        };
        let block = suggestion_result_block(&ratio, Rewriter::TextToImage, 0);
        assert_eq!(block.get("resolution"), Some(&Value::from("2400x1792")));

        let follow = RewriteSuggestion {
            rewritten_prompt: "x".to_owned(),
            ratio_follow: "<image1>".to_owned(),
            ..Default::default()
        };
        let block = suggestion_result_block(&follow, Rewriter::Editing, 2);
        assert_eq!(block.get("followsReferenceIndex"), Some(&Value::from(0)));
        assert_eq!(block.get("rewriter"), Some(&Value::from("i2i")));
    }

    #[test]
    fn a_snapshot_matching_the_frozen_digest_is_read_back_verbatim() {
        // The freeze accepts the pinned bytes. The digest is computed here rather than vendored
        // because the real 18 KB file is Qwen-licensed prose this repo deliberately does not carry.
        let dir = tempfile::tempdir().unwrap();
        let body = "# Edit Prompt Enhancer\n\nReturn one JSON object.\n";
        std::fs::write(dir.path().join(SYSTEM_PROMPT_FILE), body).unwrap();
        let frozen = hex_digest(body.as_bytes());
        assert_eq!(frozen.len(), 64);
        assert_eq!(
            load_system_prompt_verifying(dir.path(), Rewriter::Editing, &frozen).unwrap(),
            body
        );
    }

    #[test]
    fn a_snapshot_whose_system_prompt_is_not_the_frozen_one_is_refused_by_revision() {
        // ... and refuses a re-push. One changed byte is enough: the template IS the behaviour.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(SYSTEM_PROMPT_FILE),
            "# Edit Prompt Enhancer v2\n",
        )
        .unwrap();
        let frozen = hex_digest(b"# Edit Prompt Enhancer\n\nReturn one JSON object.\n");
        let message = load_system_prompt_verifying(dir.path(), Rewriter::Editing, &frozen)
            .unwrap_err()
            .to_string();
        assert!(message.contains(PE_I2I_REPO), "{message}");
        assert!(message.contains(PE_I2I_REVISION), "{message}");
        assert!(message.contains(&frozen), "{message}");
    }

    #[test]
    fn a_missing_system_prompt_names_the_repo_and_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let error = load_system_prompt(dir.path(), Rewriter::Editing).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(PE_I2I_REPO), "{message}");
        assert!(message.contains(SYSTEM_PROMPT_FILE), "{message}");
    }

    #[test]
    fn the_frozen_digests_are_recorded() {
        // The escape hatch in `verify_system_prompt_digest` skips the check while a digest is
        // blank. This test is what stops that from becoming permanent: both must be filled in
        // with the SHA-256 of the pinned revision's `system_prompt.txt`.
        for (rewriter, digest) in [
            (Rewriter::TextToImage, T2I_SYSTEM_PROMPT_SHA256),
            (Rewriter::Editing, I2I_SYSTEM_PROMPT_SHA256),
        ] {
            assert_eq!(
                digest.len(),
                64,
                "{} @ {}: the frozen system-prompt SHA-256 is not recorded",
                rewriter.repo(),
                rewriter.revision()
            );
            assert!(
                digest.chars().all(|c| c.is_ascii_hexdigit()),
                "{}: digest is not hex",
                rewriter.repo()
            );
        }
    }

    #[test]
    fn the_published_sampling_recipe_is_frozen_verbatim() {
        // From both model cards. These are upstream's numbers, not a SceneWorks tuning choice, so
        // a change here should be a deliberate divergence with a reason beside it.
        assert_eq!(REWRITE_TEMPERATURE, 1.0);
        assert_eq!(REWRITE_TOP_P, 0.95);
        assert_eq!(REWRITE_TOP_K, 20);
    }

    #[test]
    fn the_rewrite_sees_exactly_the_ten_references_the_render_takes() {
        // S3 edit contract: 1-10 ordered references. The rewrite's `<imageN>` numbering is only
        // meaningful if its list is the render's list.
        assert_eq!(MAX_REWRITE_REFERENCES, 10);
    }
}
