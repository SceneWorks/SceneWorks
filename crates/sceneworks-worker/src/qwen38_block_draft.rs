//! The Qwen3.8-27B block drafter on the describer and the H3 prompt expander (SW-7204, feature
//! SW-F374 E1 and E4).
//!
//! Both products load a Qwen3.8-27B tier through the refine cache. When one of the catalog's block
//! drafts (DFlash2, DSpine; soft co-requisites of both entries, SW-7203) is installed at its pinned
//! revision and `auto` prefers it on this backend and tier ([`auto_keeps_mtp`]), the load names it
//! in [`LoadSpec::draft_source`] at the tier's precision (the 4-bit tier runs a 4-bit draft, every
//! other an 8-bit one). Where `auto` keeps MTP the load leaves the draft out, so it is neither
//! loaded nor priced, and the job record says why.
//!
//! Every describe, condense and expansion request sends `speculative: auto`, checked linearly
//! ([`request_speculation`]). Since SW-7200 the engine's `auto` chooses the proposer from measured
//! decisions (`core_llm::AUTO_BLOCK_DRAFT_DECISIONS`): on MLX it runs a resident block draft ahead
//! of the native MTP head on the q4 and q8 tiers, where the paths campaign measured it faster; on
//! Candle/CUDA (block draft measured slower there) and on unmeasured tiers it keeps MTP. Only
//! `auto` arms the engine's acceptance monitor, which demotes a proposer to plain decoding on a
//! request where it measures slower. Tree checking is never requested: it measured slower than
//! linear checking on both backends (MLX 0.67-1.18x plain against linear 1.6-2.1x; CUDA
//! 0.45-0.70x).
//!
//! Memory: the engine's own load admission prices the draft beside the target (its weights at
//! that precision and the draft cache) and, when they do not fit, loads the target alone and names
//! the refusal in its load report. The tier is never refused for the draft's sake: the worker's
//! tier floor prices the target only. When a request's own memory refuses `auto`'s block draft, the
//! engine resolves `auto` again without it (MTP) and names
//! [`AUTO_BLOCK_DRAFT_NOT_ADMITTED`](gen_core::core_llm::AUTO_BLOCK_DRAFT_NOT_ADMITTED). Either way
//! the job record says why the draft did not run.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use gen_core::core_llm::{
    auto_block_draft_decision, BlockDraftFormat, DecodeBackend, DecodeReport, LoadReport, LoadSpec,
    Quantize, Speculative, SpeculativeVerify, AUTO_BLOCK_DRAFT_NOT_ADMITTED,
};
use serde_json::{json, Value};

/// The `draftFallback` reason when no block draft is installed.
pub(crate) const NO_DRAFT_INSTALLED: &str =
    "block draft: none installed (the DFlash2 and DSpine drafts download with the model's tiers)";

/// The decode backend this build's Qwen3.8-27B loads run on: MLX on macOS, Candle/CUDA elsewhere.
const DRAFT_BACKEND: DecodeBackend = if cfg!(target_os = "macos") {
    DecodeBackend::Mlx
} else {
    DecodeBackend::CandleCuda
};

/// An installed block draft of a Qwen3.8-27B entry, and whether `auto` runs it here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BlockDraftChoice {
    /// The draft's repository (`SceneWorks/qwen3.8-27b-dflash2`).
    pub(crate) repo: String,
    /// Its pinned revision.
    pub(crate) revision: String,
    /// The pinned snapshot folder the load reads.
    pub(crate) dir: PathBuf,
    /// The load precision (`q8`, `q4`).
    pub(crate) precision: &'static str,
    /// The backend and target tier `auto`'s decision was taken for (`Mlx q4`).
    pub(crate) decided_for: String,
    /// `None` when `auto` prefers this draft on [`decided_for`](Self::decided_for), so the load
    /// names it; otherwise why `auto` keeps MTP there, and the load leaves the draft out.
    pub(crate) auto_keeps_mtp: Option<String>,
}

impl BlockDraftChoice {
    fn quantize(&self) -> Quantize {
        if self.precision == "q4" {
            Quantize::Q4
        } else {
            Quantize::Q8
        }
    }

    /// Whether `auto` runs this draft (the load names it).
    pub(crate) fn auto_runs(&self) -> bool {
        self.auto_keeps_mtp.is_none()
    }
}

/// Why `auto` keeps MTP instead of a resident `format` draft beside a `tier` target on `backend`, or
/// `None` when the engine's measured decision (`core_llm::auto_block_draft_decision`, SW-7200)
/// prefers the draft.
pub(crate) fn auto_keeps_mtp(
    backend: DecodeBackend,
    format: Option<BlockDraftFormat>,
    tier: Option<&str>,
) -> Option<String> {
    let Some(format) = format else {
        return Some(
            "block draft: its config.json names no block-draft format; auto keeps MTP".to_owned(),
        );
    };
    let Some(tier) = tier else {
        return Some(format!(
            "block draft: auto keeps MTP on {backend:?} (no target tier named)"
        ));
    };
    match auto_block_draft_decision(backend, format, tier) {
        Some(decision) if decision.prefer => None,
        Some(decision) => Some(format!(
            "block draft: auto keeps MTP on {backend:?} {tier} ({})",
            decision.reason
        )),
        None => Some(format!(
            "block draft: auto keeps MTP on {backend:?} {tier} (no measured decision)"
        )),
    }
}

/// The first of `model_id`'s catalog block drafts (preference order:
/// [`sceneworks_core::dataset_batch_describe::QWEN38_27B_BLOCK_DRAFT_REPOS`]) whose pinned snapshot
/// holds every file its row names, at the precision `tier` runs it at, with `auto`'s decision for
/// it on this build's backend. `None` when none is installed.
pub(crate) fn installed_block_draft(
    data_dir: &Path,
    model_id: &str,
    tier: Option<&str>,
) -> Option<BlockDraftChoice> {
    installed_block_draft_on(DRAFT_BACKEND, data_dir, model_id, tier)
}

/// [`installed_block_draft`] with `auto`'s decision taken for `backend`.
pub(crate) fn installed_block_draft_on(
    backend: DecodeBackend,
    data_dir: &Path,
    model_id: &str,
    tier: Option<&str>,
) -> Option<BlockDraftChoice> {
    use sceneworks_core::dataset_batch_describe::{
        qwen38_block_draft_precision, qwen38_block_draft_rows,
    };
    qwen38_block_draft_rows(model_id)
        .into_iter()
        .find_map(|row| {
            let dir = crate::model_jobs::huggingface_pinned_snapshot_dir(
                data_dir,
                &row.repo,
                &row.revision,
            )?;
            row.files
                .iter()
                .all(|file| dir.join(file).is_file())
                .then(|| BlockDraftChoice {
                    auto_keeps_mtp: auto_keeps_mtp(backend, BlockDraftFormat::of_dir(&dir), tier),
                    decided_for: format!("{backend:?} {}", tier.unwrap_or("(no tier)")),
                    repo: row.repo,
                    revision: row.revision,
                    dir,
                    precision: qwen38_block_draft_precision(tier),
                })
        })
}

/// `spec` naming `draft` at its precision when there is one and `auto` runs it; a draft `auto`
/// keeps MTP over is left out of the load, so it is neither loaded nor priced.
pub(crate) fn with_block_draft(mut spec: LoadSpec, draft: Option<&BlockDraftChoice>) -> LoadSpec {
    if let Some(draft) = draft.filter(|draft| draft.auto_runs()) {
        spec.draft_source = Some(draft.dir.to_string_lossy().into_owned());
        spec.draft_quantize = Some(draft.quantize());
    }
    spec
}

/// The speculative option and verify layout a describe, condense or expansion request carries: the
/// product's option (`auto` for every Qwen3.8-27B request, so the engine picks the proposer and its
/// monitor can demote it; `None` keeps the backend default for the other describers), always
/// checked linearly.
pub(crate) fn request_speculation(
    product: Option<Speculative>,
) -> (Option<Speculative>, SpeculativeVerify) {
    (product, SpeculativeVerify::Linear)
}

/// The speculative option a Qwen3.8-27B prompt expansion (MiniMax-H3-Image-PE or the abliterated
/// describer, both Qwen3.8-27B) carries: `auto`, checked linearly.
pub(crate) fn expansion_speculation() -> (Option<Speculative>, SpeculativeVerify) {
    request_speculation(Some(Speculative::Auto))
}

/// Why the block draft did not run: none installed, `auto` keeps MTP over it on this backend and
/// tier (so the load left it out), the engine refused it at load (it did not fit beside the
/// target, or its config does not match; from the load report), or the request's own memory
/// withdrew `auto`'s block draft (the decode report's [`AUTO_BLOCK_DRAFT_NOT_ADMITTED`] fallback).
pub(crate) fn draft_fallback(
    draft: Option<&BlockDraftChoice>,
    load: Option<&LoadReport>,
    decode: Option<&DecodeReport>,
) -> Option<String> {
    let Some(draft) = draft else {
        return Some(NO_DRAFT_INSTALLED.to_owned());
    };
    if let Some(reason) = &draft.auto_keeps_mtp {
        return Some(reason.clone());
    }
    match load.and_then(|load| load.draft.as_ref()) {
        Some(report) => report.refusal.clone().or_else(|| {
            decode?
                .fallbacks
                .iter()
                .find(|reason| reason.contains(AUTO_BLOCK_DRAFT_NOT_ADMITTED))
                .cloned()
        }),
        None => Some("block draft: the engine reported no draft for this load".to_owned()),
    }
}

/// What the generations of one phase of a job (describe, condense, or one expansion) decoded with,
/// summed: one `decode` block (E1).
#[derive(Clone, Debug, Default)]
pub(crate) struct DecodeTally {
    generations: u64,
    proposers: Vec<String>,
    draft: Option<Value>,
    tree_generations: u64,
    linear_generations: u64,
    target_passes: u64,
    tree_passes: u64,
    rows_checked: u64,
    accepted_tokens: u64,
    demotions: u64,
    draft_fallback: Option<String>,
    fallbacks: Vec<String>,
}

/// Which phase of a describe job a generation belongs to: the image describe, or the text-only
/// condense. Each gets its own decode block, since their proposers and checking differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodePhase {
    Describe,
    Condense,
}

/// A describe job's decode blocks, one per phase.
#[derive(Clone, Debug, Default)]
pub(crate) struct DescribeDecode {
    describe: DecodeTally,
    condense: DecodeTally,
}

impl DescribeDecode {
    /// Add one generation of `phase` ([`DecodeTally::note`]).
    pub(crate) fn note(
        &mut self,
        phase: DecodePhase,
        decode: Option<&DecodeReport>,
        load: Option<&LoadReport>,
        draft: Option<&BlockDraftChoice>,
    ) {
        match phase {
            DecodePhase::Describe => &mut self.describe,
            DecodePhase::Condense => &mut self.condense,
        }
        .note(decode, load, draft);
    }

    /// The job record's `decode` block: `{describe, condense}`, each a [`DecodeTally::record`].
    pub(crate) fn record(&self) -> Value {
        json!({ "describe": self.describe.record(), "condense": self.condense.record() })
    }
}

/// A job's decode blocks, shared by the describer's describe and condense calls.
pub(crate) type SharedDecodeTally = Arc<Mutex<DescribeDecode>>;

impl DecodeTally {
    /// Add one generation: its decode report, the load report of the model that ran it, and the
    /// block draft the load named.
    pub(crate) fn note(
        &mut self,
        decode: Option<&DecodeReport>,
        load: Option<&LoadReport>,
        draft: Option<&BlockDraftChoice>,
    ) {
        self.generations += 1;
        // The phase's first fallback stays: a later generation never erases why an earlier one ran
        // without the draft.
        if self.draft_fallback.is_none() {
            self.draft_fallback = draft_fallback(draft, load, decode);
        }
        let Some(decode) = decode else {
            return;
        };
        let proposer = decode.proposer.label().to_owned();
        if !self.proposers.contains(&proposer) {
            self.proposers.push(proposer);
        }
        // Once any generation ran the draft, the phase names it.
        if let Some(ran) = &decode.draft {
            self.draft = Some(json!({
                "repo": draft.map(|d| d.repo.as_str()),
                "revision": draft.map(|d| d.revision.as_str()),
                "source": ran.source,
                "format": ran.format,
                "precision": ran.precision,
            }));
        }
        match &decode.tree_verify {
            Some(tree) => {
                self.tree_generations += 1;
                self.target_passes += tree.passes;
                self.tree_passes += tree.tree_passes;
                self.rows_checked += tree.rows_verified;
                self.accepted_tokens += tree.accepted_tokens;
            }
            None => {
                self.linear_generations += 1;
                self.target_passes += decode.verify_steps;
                // A linear pass checks the anchor and the drafts proposed for it.
                self.rows_checked += decode.verify_steps + decode.proposed_tokens;
                self.accepted_tokens += decode.accepted_tokens;
            }
        }
        if decode.speculative_demoted_at.is_some() {
            self.demotions += 1;
        }
        for reason in &decode.fallbacks {
            if !self.fallbacks.contains(reason) {
                self.fallbacks.push(reason.clone());
            }
        }
    }

    /// The `decode` block: proposer, draft (repo, revision, format, precision), verify (`tree`,
    /// `linear`, or `mixed` when the phase's generations differ), target passes, rows checked and
    /// tokens accepted (totals and per pass), demotions, the decode fallbacks, and why the draft
    /// did not run when it did not. `proposer` is `mixed` when the generations ran different
    /// proposers (all are in `proposers`).
    pub(crate) fn record(&self) -> Value {
        let per_pass =
            |count: u64| (self.target_passes > 0).then(|| count as f64 / self.target_passes as f64);
        let proposer = match self.proposers.as_slice() {
            [] => None,
            [one] => Some(one.as_str()),
            _ => Some("mixed"),
        };
        let verify = match (self.tree_generations, self.linear_generations) {
            (0, 0) => None,
            (_, 0) => Some("tree"),
            (0, _) => Some("linear"),
            _ => Some("mixed"),
        };
        json!({
            "generations": self.generations,
            "proposer": proposer,
            "proposers": self.proposers,
            "draft": self.draft,
            "verify": verify,
            "targetPasses": self.target_passes,
            "treePasses": self.tree_passes,
            "rowsChecked": self.rows_checked,
            "rowsCheckedPerPass": per_pass(self.rows_checked),
            "acceptedTokens": self.accepted_tokens,
            "acceptedTokensPerPass": per_pass(self.accepted_tokens),
            "demotions": self.demotions,
            "draftFallback": self.draft_fallback,
            "fallbacks": self.fallbacks,
        })
    }
}

/// One generation's `decode` block.
pub(crate) fn decode_record(
    decode: Option<&DecodeReport>,
    load: Option<&LoadReport>,
    draft: Option<&BlockDraftChoice>,
) -> Value {
    let mut tally = DecodeTally::default();
    tally.note(decode, load, draft);
    tally.record()
}

/// What a real-weight smoke checks about the block drafter (SW-7204):
/// `$SCENEWORKS_QWEN38_BLOCK_DRAFT` = `record` (the default: the decode block is recorded, nothing
/// is asserted about it), `on` (a draft must be installed and the target checked linearly; where
/// `auto` prefers the draft ([`BlockDraftChoice::auto_runs`]) the record must name `block_draft`,
/// the draft and its precision, more than one accepted token per target pass and no fallback, and
/// elsewhere the draft must not run and the record must say why) or `no_fit` (the engine's load
/// budget is capped so the draft cannot fit beside the target: the job must complete without it
/// and its record must say why).
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SmokeDraftCheck {
    Record,
    On,
    NoFit,
}

#[cfg(test)]
impl SmokeDraftCheck {
    pub(crate) const ENV: &'static str = "SCENEWORKS_QWEN38_BLOCK_DRAFT";

    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            "" | "record" => Ok(Self::Record),
            "on" => Ok(Self::On),
            "no_fit" => Ok(Self::NoFit),
            other => Err(format!(
                "${} must be record, on or no_fit, got {other:?}",
                Self::ENV
            )),
        }
    }

    /// `$SCENEWORKS_QWEN38_BLOCK_DRAFT`; an unknown value fails the run.
    pub(crate) fn from_env() -> Self {
        Self::parse(&std::env::var(Self::ENV).unwrap_or_default())
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Record => "record",
            Self::On => "on",
            Self::NoFit => "no_fit",
        }
    }

    /// The installed draft the production path will name; `on` and `no_fit` need one.
    pub(crate) fn installed_draft(
        self,
        data_dir: &Path,
        model_id: &str,
        tier: Option<&str>,
    ) -> Option<BlockDraftChoice> {
        let draft = installed_block_draft(data_dir, model_id, tier);
        assert!(
            self == Self::Record || draft.is_some(),
            "${}={} needs a block draft installed for {model_id} (provision the DFlash2/DSpine \
             co-requisites)",
            Self::ENV,
            self.label()
        );
        draft
    }

    /// Check one phase's `decode` block against this mode; a failure names the backend and tier
    /// `auto`'s decision was taken for.
    pub(crate) fn check(
        self,
        record: &Value,
        draft: Option<&BlockDraftChoice>,
    ) -> Result<(), String> {
        let fail = |what: &str| {
            let decided = draft.map_or("no draft installed", |draft| draft.decided_for.as_str());
            Err(format!("{what} (auto decided for {decided}): {record}"))
        };
        match self {
            Self::Record => Ok(()),
            Self::On => {
                let Some(draft) = draft else {
                    return fail("no block draft is installed");
                };
                if record["verify"] != "linear" {
                    return fail("the target did not check linearly");
                }
                if !draft.auto_runs() {
                    if record["proposer"] == "block_draft" || !record["draft"].is_null() {
                        return fail("auto ran the block draft where it keeps MTP");
                    }
                    if record["draftFallback"].is_null() {
                        return fail("the record does not say why the draft did not run");
                    }
                    return Ok(());
                }
                if !record["draftFallback"].is_null() {
                    return fail("the draft fell back");
                }
                if record["proposer"] != "block_draft" {
                    return fail("the proposer is not block_draft");
                }
                if record["draft"]["repo"] != draft.repo.as_str()
                    || record["draft"]["revision"] != draft.revision.as_str()
                {
                    return fail("the record names another draft");
                }
                if record["draft"]["precision"] != draft.precision {
                    return fail(&format!("the draft did not run at {}", draft.precision));
                }
                if !matches!(
                    record["draft"]["format"].as_str(),
                    Some("dflash2" | "dspine")
                ) {
                    return fail("the draft format is not named");
                }
                if !record["acceptedTokensPerPass"]
                    .as_f64()
                    .is_some_and(|per_pass| per_pass > 1.0)
                {
                    return fail("accepted tokens per target pass is not above 1.0");
                }
                Ok(())
            }
            Self::NoFit => {
                if record["proposer"] == "block_draft" || !record["draft"].is_null() {
                    return fail("the draft ran although it could not fit");
                }
                // Where `auto` keeps MTP the load never names the draft, so its reason stands in
                // for the load refusal.
                let named = record["draftFallback"].as_str().is_some_and(|why| {
                    why.contains("beside the target")
                        || draft.is_some_and(|draft| !draft.auto_runs())
                });
                if !named {
                    return fail("the record does not say the draft did not fit");
                }
                Ok(())
            }
        }
    }
}

/// `auto`'s decision about the installed draft, for a smoke's evidence (the CI validator branches
/// on `runsDraft`): whether it runs the draft, the backend and tier it decided for, and why it keeps
/// MTP when it does.
#[cfg(test)]
pub(crate) fn auto_evidence(draft: Option<&BlockDraftChoice>) -> Value {
    json!({
        "runsDraft": draft.is_some_and(BlockDraftChoice::auto_runs),
        "decidedFor": draft.map(|draft| draft.decided_for.as_str()),
        "keepsMtp": draft.and_then(|draft| draft.auto_keeps_mtp.as_deref()),
    })
}

/// The target's load estimate an engine load refusal names ("requires an estimated N bytes").
#[cfg(test)]
pub(crate) fn refused_load_estimate(error: &str) -> Option<u64> {
    const MARK: &str = "requires an estimated ";
    let at = error.find(MARK)? + MARK.len();
    error[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// Restores an environment variable on drop.
#[cfg(test)]
pub(crate) struct EnvOverride {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

#[cfg(test)]
impl EnvOverride {
    pub(crate) fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, previous }
    }
}

#[cfg(test)]
impl Drop for EnvOverride {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}

/// `no_fit` (smokes only): cap the engine's load budget so the target fits and the draft does
/// not. A probe load of `weights_dir` under a 1-byte budget is refused before any weight is read,
/// naming the target's load estimate `N`; the budget becomes `N + D - 1`, where `D` is the draft's
/// own load bytes at its precision (`BlockDraftConfig::load_bytes`), which both engines' draft
/// estimates are at least. Returns the guard that keeps the cap until it drops.
#[cfg(all(test, native_lane))]
pub(crate) async fn force_draft_no_fit(
    weights_dir: &Path,
    draft: &BlockDraftChoice,
) -> EnvOverride {
    use gen_core::core_llm::{BlockDraftConfig, BlockDraftPrecision, AVAILABLE_MEMORY_OVERRIDE};
    let (spec, requirements) =
        crate::batch_describe_jobs::vision_describer_load(weights_dir, Some(draft));
    let _probe_budget = EnvOverride::set(AVAILABLE_MEMORY_OVERRIDE, "1");
    let probe = crate::refine_model_cache::with_cached_refiner(
        spec,
        requirements,
        "block-draft no-fit probe",
        |_| Ok(()),
    )
    .await;
    drop(_probe_budget);
    let error = match probe {
        Ok(()) => panic!("a 1-byte load budget admitted the target"),
        Err(error) => error.to_string(),
    };
    let target = refused_load_estimate(&error)
        .unwrap_or_else(|| panic!("the probe refusal names no load estimate: {error}"));
    let precision = if draft.precision == "q4" {
        BlockDraftPrecision::Q4
    } else {
        BlockDraftPrecision::Q8
    };
    let draft_bytes = BlockDraftConfig::read(&draft.dir)
        .unwrap_or_else(|error| panic!("the draft config reads: {error}"))
        .load_bytes(precision);
    let budget = target + draft_bytes.saturating_sub(1);
    eprintln!(
        "BLOCK_DRAFT_NO_FIT target_estimate={target} draft_load_bytes={draft_bytes} \
         budget={budget}"
    );
    EnvOverride::set(AVAILABLE_MEMORY_OVERRIDE, &budget.to_string())
}

#[cfg(test)]
#[path = "qwen38_block_draft/tests.rs"]
mod tests;
