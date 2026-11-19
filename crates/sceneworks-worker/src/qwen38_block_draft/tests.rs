//! Request construction, admission fallback and record fields of the Qwen3.8-27B block drafter
//! (SW-7204). No model runs: the snapshots are empty files in a private hub cache.

use std::path::{Path, PathBuf};

use gen_core::core_llm::{
    DecodeDraft, DecodeReport, DraftReport, LoadReport, LoadSpec, ProposerCapabilities,
    ProposerKind, Quantize, Speculative, SpeculativeProposer, SpeculativeVerify,
    TextLlmCapabilities, TreeVerifyCapabilities, TreeVerifyReport,
};
use sceneworks_core::dataset_batch_describe::{
    qwen38_block_draft_rows, QWEN38_27B_BLOCK_DRAFT_REPOS, QWEN38_27B_DESCRIBER_MODEL_ID,
};

use super::*;

const DFLASH2: &str = QWEN38_27B_BLOCK_DRAFT_REPOS[0];
const DSPINE: &str = QWEN38_27B_BLOCK_DRAFT_REPOS[1];

/// A data directory whose Hugging Face cache is private to the test.
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

/// Write `repo`'s catalog draft snapshot at its pinned revision, every file its row names except
/// `skip`.
fn install(data: &Path, repo: &str, skip: Option<&str>) -> PathBuf {
    let row = qwen38_block_draft_rows(QWEN38_27B_DESCRIBER_MODEL_ID)
        .into_iter()
        .find(|row| row.repo == repo)
        .expect("the describer ships the draft");
    let dir = crate::huggingface_hub_cache_dir(data)
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(&row.revision);
    std::fs::create_dir_all(&dir).unwrap();
    for file in row.files.iter().filter(|file| Some(file.as_str()) != skip) {
        std::fs::write(dir.join(file), b"x").unwrap();
    }
    dir
}

#[test]
fn no_installed_draft_names_none_and_says_so() {
    let (data, _guard) = isolated();
    assert_eq!(
        installed_block_draft(data.path(), QWEN38_27B_DESCRIBER_MODEL_ID, Some("q8")),
        None
    );
    assert_eq!(
        draft_fallback(None, None).as_deref(),
        Some(NO_DRAFT_INSTALLED)
    );
}

#[test]
fn a_partial_draft_snapshot_is_not_installed() {
    let (data, _guard) = isolated();
    install(data.path(), DFLASH2, Some("model.safetensors"));
    assert_eq!(
        installed_block_draft(data.path(), QWEN38_27B_DESCRIBER_MODEL_ID, None),
        None
    );
}

#[test]
fn dflash2_is_preferred_and_the_tier_sets_the_draft_precision() {
    let (data, _guard) = isolated();
    let dspine = install(data.path(), DSPINE, None);
    let only_dspine =
        installed_block_draft(data.path(), QWEN38_27B_DESCRIBER_MODEL_ID, Some("q4")).unwrap();
    assert_eq!(only_dspine.repo, DSPINE);
    assert_eq!(only_dspine.dir, dspine);
    assert_eq!(only_dspine.precision, "q4");

    let dflash2 = install(data.path(), DFLASH2, None);
    for (model_id, tier, precision) in [
        (QWEN38_27B_DESCRIBER_MODEL_ID, Some("q8"), "q8"),
        (QWEN38_27B_DESCRIBER_MODEL_ID, Some("bf16"), "q8"),
        (
            sceneworks_core::contracts::MINIMAX_H3_IMAGE_PE_MODEL_ID,
            Some("q4"),
            "q4",
        ),
        (
            sceneworks_core::contracts::MINIMAX_H3_IMAGE_PE_MODEL_ID,
            None,
            "q8",
        ),
    ] {
        let draft = installed_block_draft(data.path(), model_id, tier).unwrap();
        assert_eq!(draft.repo, DFLASH2, "{model_id} {tier:?}");
        assert_eq!(draft.dir, dflash2);
        assert_eq!(draft.precision, precision, "{model_id} {tier:?}");
        assert_eq!(draft.revision.len(), 40);
    }
}

fn choice(precision: &'static str) -> BlockDraftChoice {
    BlockDraftChoice {
        repo: DFLASH2.to_owned(),
        revision: "9030fabfb5e3ca0c5c180f84927af86aa9f9a2e9".to_owned(),
        dir: PathBuf::from("/hub/dflash2"),
        precision,
    }
}

#[test]
fn the_load_names_the_draft_at_its_precision() {
    let base = LoadSpec {
        source: "/hub/tier".to_owned(),
        ..Default::default()
    };
    let plain = with_block_draft(base.clone(), None);
    assert_eq!(plain.draft_source, None);
    assert_eq!(plain.draft_quantize, None);
    for (precision, quantize) in [("q8", Quantize::Q8), ("q4", Quantize::Q4)] {
        let spec = with_block_draft(base.clone(), Some(&choice(precision)));
        assert_eq!(spec.source, "/hub/tier", "the target is unchanged");
        assert_eq!(spec.draft_source.as_deref(), Some("/hub/dflash2"));
        assert_eq!(spec.draft_quantize, Some(quantize));
    }
}

#[test]
fn an_advertised_block_draft_runs_with_tree_checking_else_the_products_option() {
    let mut caps = TextLlmCapabilities::default();
    assert_eq!(
        block_draft_speculation(&caps, Some(Speculative::Auto)),
        (Some(Speculative::Auto), SpeculativeVerify::Linear)
    );
    caps.speculative.push(ProposerCapabilities {
        proposer: SpeculativeProposer::BlockDraft,
        max_depth: 15,
        recommended_depth: 4,
    });
    // Advertised but no tree verification: linear checking.
    assert_eq!(
        block_draft_speculation(&caps, Some(Speculative::Auto)),
        (
            Some(Speculative::proposer(SpeculativeProposer::BlockDraft, 4)),
            SpeculativeVerify::Linear
        )
    );
    caps.tree_verify = Some(TreeVerifyCapabilities { max_rows: 16 });
    assert_eq!(
        block_draft_speculation(&caps, None),
        (
            Some(Speculative::proposer(SpeculativeProposer::BlockDraft, 4)),
            SpeculativeVerify::Tree { max_rows: 16 }
        )
    );
}

fn tree_report() -> DecodeReport {
    DecodeReport {
        proposer: ProposerKind::BlockDraft,
        draft: Some(DecodeDraft {
            source: "/hub/dflash2".to_owned(),
            format: "dflash2".to_owned(),
            precision: "q4".to_owned(),
        }),
        tree_verify: Some(TreeVerifyReport {
            max_rows: 16,
            passes: 10,
            tree_passes: 8,
            rows_verified: 120,
            accepted_tokens: 35,
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn the_record_names_the_proposer_draft_precision_and_per_pass_counts() {
    let load = LoadReport {
        draft: Some(DraftReport::resident("/hub/dflash2")),
        ..Default::default()
    };
    let draft = choice("q4");
    let mut tally = DecodeTally::default();
    tally.note(Some(&tree_report()), Some(&load), Some(&draft));
    tally.note(Some(&tree_report()), Some(&load), Some(&draft));
    let record = tally.record();
    assert_eq!(record["generations"], 2);
    assert_eq!(record["proposer"], "block_draft");
    assert_eq!(record["verify"], "tree");
    assert_eq!(record["draft"]["repo"], DFLASH2);
    assert_eq!(
        record["draft"]["revision"],
        "9030fabfb5e3ca0c5c180f84927af86aa9f9a2e9"
    );
    assert_eq!(record["draft"]["format"], "dflash2");
    assert_eq!(record["draft"]["precision"], "q4");
    assert_eq!(record["targetPasses"], 20);
    assert_eq!(record["treePasses"], 16);
    assert_eq!(record["rowsChecked"], 240);
    assert_eq!(record["rowsCheckedPerPass"], 12.0);
    assert_eq!(record["acceptedTokens"], 70);
    assert_eq!(record["acceptedTokensPerPass"], 3.5);
    assert_eq!(record["draftFallback"], Value::Null);
}

#[test]
fn a_draft_refused_at_load_is_recorded_with_the_engines_reason() {
    let reason = "draft model: its load needs an estimated 2000 bytes beside the target's 9000, \
                  but only 10000 bytes are available";
    let load = LoadReport {
        draft: Some(DraftReport::refused("/hub/dflash2", reason)),
        ..Default::default()
    };
    // The target ran alone: its native MTP head under `auto`, checked linearly.
    let decode = DecodeReport {
        proposer: ProposerKind::Mtp,
        verify_steps: 10,
        proposed_tokens: 30,
        accepted_tokens: 12,
        speculative_demoted_at: Some(40),
        fallbacks: vec!["speculative: demoted".to_owned()],
        ..Default::default()
    };
    let record = decode_record(Some(&decode), Some(&load), Some(&choice("q8")));
    assert_eq!(record["proposer"], "mtp");
    assert_eq!(record["draft"], Value::Null);
    assert_eq!(record["verify"], "linear");
    assert_eq!(record["targetPasses"], 10);
    assert_eq!(record["rowsChecked"], 40);
    assert_eq!(record["acceptedTokensPerPass"], 1.2);
    assert_eq!(record["demotions"], 1);
    assert_eq!(record["draftFallback"], reason);
    assert_eq!(record["fallbacks"], json!(["speculative: demoted"]));

    let none = decode_record(None, None, None);
    assert_eq!(none["draftFallback"], NO_DRAFT_INSTALLED);
    assert_eq!(none["acceptedTokensPerPass"], Value::Null);
}

#[test]
fn the_smoke_modes_parse_and_record_is_the_default() {
    assert_eq!(SmokeDraftCheck::parse(""), Ok(SmokeDraftCheck::Record));
    assert_eq!(
        SmokeDraftCheck::parse("record"),
        Ok(SmokeDraftCheck::Record)
    );
    assert_eq!(SmokeDraftCheck::parse(" on "), Ok(SmokeDraftCheck::On));
    assert_eq!(SmokeDraftCheck::parse("no_fit"), Ok(SmokeDraftCheck::NoFit));
    assert!(SmokeDraftCheck::parse("yes").is_err());
}

#[test]
fn the_on_check_needs_the_draft_its_precision_a_tree_and_more_than_one_token_per_pass() {
    use DraftRequest::{Image, Text};
    let draft = choice("q4");
    let load = LoadReport {
        draft: Some(DraftReport::resident("/hub/dflash2")),
        ..Default::default()
    };
    let ran = decode_record(Some(&tree_report()), Some(&load), Some(&draft));
    for request in [Image, Text] {
        assert_eq!(
            SmokeDraftCheck::On.check(&ran, Some(&draft), request),
            Ok(())
        );
    }
    assert_eq!(
        SmokeDraftCheck::Record.check(&Value::Null, None, Text),
        Ok(())
    );
    // An 8-bit draft is not the 4-bit tier's.
    assert!(SmokeDraftCheck::On
        .check(&ran, Some(&choice("q8")), Text)
        .is_err());
    // One accepted token per pass or fewer is no speed-up to show.
    let mut slow = tree_report();
    slow.tree_verify.as_mut().unwrap().accepted_tokens = 10;
    let slow = decode_record(Some(&slow), Some(&load), Some(&draft));
    assert!(SmokeDraftCheck::On
        .check(&slow, Some(&draft), Text)
        .is_err());
    // A draft that did not fit fails `on` and passes `no_fit`.
    let refused = LoadReport {
        draft: Some(DraftReport::refused(
            "/hub/dflash2",
            "draft model: its load needs an estimated 2 bytes beside the target's 9, but only \
             10 bytes are available",
        )),
        ..Default::default()
    };
    let alone = DecodeReport {
        proposer: ProposerKind::Mtp,
        verify_steps: 4,
        accepted_tokens: 6,
        ..Default::default()
    };
    let fell_back = decode_record(Some(&alone), Some(&refused), Some(&draft));
    assert!(SmokeDraftCheck::On
        .check(&fell_back, Some(&draft), Image)
        .is_err());
    assert_eq!(
        SmokeDraftCheck::NoFit.check(&fell_back, Some(&draft), Image),
        Ok(())
    );
    assert!(SmokeDraftCheck::NoFit
        .check(&ran, Some(&draft), Text)
        .is_err());
}

/// At the pinned inference an image request's tree checking runs linearly, named in the decode
/// fallbacks: `on` accepts that for an image request only, and a text request must check a tree.
#[test]
fn an_image_request_may_check_linearly_only_with_the_engines_multimodal_reason() {
    let draft = choice("q8");
    let load = LoadReport {
        draft: Some(DraftReport::resident("/hub/dflash2")),
        ..Default::default()
    };
    let linear = |fallbacks: Vec<String>| DecodeReport {
        proposer: ProposerKind::BlockDraft,
        draft: Some(DecodeDraft {
            source: "/hub/dflash2".to_owned(),
            format: "dflash2".to_owned(),
            precision: "q8".to_owned(),
        }),
        verify_steps: 10,
        proposed_tokens: 40,
        accepted_tokens: 18,
        fallbacks,
        ..Default::default()
    };
    let named = decode_record(
        Some(&linear(vec![format!(
            "speculative: {TREE_MULTIMODAL_LINEAR} (its prompt is prefilled from spliced \
             embeddings); verified linearly"
        )])),
        Some(&load),
        Some(&draft),
    );
    assert_eq!(named["verify"], "linear");
    assert_eq!(
        SmokeDraftCheck::On.check(&named, Some(&draft), DraftRequest::Image),
        Ok(())
    );
    assert!(SmokeDraftCheck::On
        .check(&named, Some(&draft), DraftRequest::Text)
        .is_err());
    let unnamed = decode_record(Some(&linear(Vec::new())), Some(&load), Some(&draft));
    assert!(SmokeDraftCheck::On
        .check(&unnamed, Some(&draft), DraftRequest::Image)
        .is_err());
}

/// Describe and condense get their own decode blocks: an image describe checked linearly and a
/// text condense checked as a tree are never blended.
#[test]
fn describe_and_condense_are_recorded_as_separate_blocks() {
    let draft = choice("q4");
    let load = LoadReport {
        draft: Some(DraftReport::resident("/hub/dflash2")),
        ..Default::default()
    };
    let describe = DecodeReport {
        proposer: ProposerKind::BlockDraft,
        verify_steps: 8,
        proposed_tokens: 32,
        accepted_tokens: 12,
        ..Default::default()
    };
    let mut job = DescribeDecode::default();
    job.note(
        DecodePhase::Describe,
        Some(&describe),
        Some(&load),
        Some(&draft),
    );
    job.note(
        DecodePhase::Condense,
        Some(&tree_report()),
        Some(&load),
        Some(&draft),
    );
    let record = job.record();
    assert_eq!(record["describe"]["verify"], "linear");
    assert_eq!(record["describe"]["targetPasses"], 8);
    assert_eq!(record["describe"]["acceptedTokensPerPass"], 1.5);
    assert_eq!(record["condense"]["verify"], "tree");
    assert_eq!(record["condense"]["targetPasses"], 10);
    assert_eq!(record["condense"]["acceptedTokensPerPass"], 3.5);
    assert_eq!(record["condense"]["draft"]["precision"], "q4");

    // A phase whose generations differ says so rather than keeping the last one.
    let mut mixed = DecodeTally::default();
    mixed.note(Some(&describe), Some(&load), Some(&draft));
    let mtp = DecodeReport {
        proposer: ProposerKind::Mtp,
        tree_verify: Some(TreeVerifyReport::default()),
        ..Default::default()
    };
    mixed.note(Some(&mtp), Some(&load), Some(&draft));
    let mixed = mixed.record();
    assert_eq!(mixed["proposer"], "mixed");
    assert_eq!(mixed["proposers"], json!(["block_draft", "mtp"]));
    assert_eq!(mixed["verify"], "mixed");
}

#[test]
fn the_no_fit_probe_reads_the_targets_load_estimate() {
    let error = "Qwen3.8 27B describer load failed: invalid request: load admission: loading \
                 this model requires an estimated 31234567890 bytes of resident weights and \
                 load-time conversions but only 1 bytes are available; free memory";
    assert_eq!(refused_load_estimate(error), Some(31_234_567_890));
    assert_eq!(refused_load_estimate("out of memory"), None);
}
