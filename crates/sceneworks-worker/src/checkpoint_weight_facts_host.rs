//! The worker's producer for [`CheckpointWeightFactsV1`] — where the three facts about an imported
//! checkpoint are gathered before they enter an asset receipt, a telemetry row, or a model fact
//! (sc-21484, epic 11037).
//!
//! `sceneworks-core` owns the *shape* of the facts and the set of lies it refuses to represent.
//! This module owns *where each fact comes from*, which is a different question per fact and the
//! reason they are three separate values rather than one:
//!
//! | Fact | Source | Available where |
//! |---|---|---|
//! | Source binding | the resolved checkpoint file's name + size | everywhere |
//! | Source codec | the header classification persisted as `importQuantFormat` | everywhere |
//! | Host capability | the `nvidia-smi` compute-cap probe `discover_gpu` runs at startup | candle lane; dense-only elsewhere |
//! | Materialization | the engine's load receipt | see [`materialization_from_runtime`] |
//!
//! # The capability is a probe rendering, not a second gate
//!
//! [`host_native_execution_capability`] does not invent a floor. It renders the *existing*
//! [`crate::gpu::compute_cap_meets_nvfp4`] decision — the sm_120 marker capability sc-11042 already
//! advertises to the api and the web tier picker — into the cross-repo
//! [`NativeExecutionCapabilityFact`] vocabulary. There is one floor constant in this repository and
//! this module reads it rather than restating it, so the fact a user reads and the tier the picker
//! offers can never disagree about what this box can do.
//!
//! That floor is `>= 12.0`, which **excludes datacenter Blackwell `sm_100`** (it probes `10.0`).
//! sm_100 has FP4 tensor cores, but this leg's kernel is neither built for it
//! (`CUDA_COMPUTE_CAP=120` emits sm_120 SASS + compute_120 PTX) nor validated on it, and the
//! engine's own `CandleCodecResidency` applies the same exclusion. The two repositories agree by
//! construction here and [`tests::the_sm_120_floor_excludes_sm_100`] pins it.
//!
//! # What this module refuses to do
//!
//! **It never synthesizes a materialization.** A host whose capability lists `nvfp4-v1` is a host
//! that *could* execute packed; whether it *did* is a property of the load, not of the box, and
//! only the engine's receipt knows it. Deriving one from the other is precisely the collapse the
//! story exists to undo, so [`imported_checkpoint_facts`] takes the materialization as an argument
//! rather than computing it, and the only producer of a `Reported` value is a real receipt.

use std::path::Path;

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
use sceneworks_core::checkpoint_weight_facts::NVFP4_CODEC_ID;
use sceneworks_core::checkpoint_weight_facts::{
    CheckpointWeightFactsV1, ExecutionRepresentation, MaterializationFact,
    MaterializationUnavailable, MaterializedCodecFact, NativeExecutionCapabilityFact,
    SourceBindingFact, SourceCodecFact,
};
use serde_json::{Map, Value};

/// This host's native-execution declaration, rendered from the startup compute-cap probe.
///
/// Dense-only on macOS/MLX, on a build without the candle lane, on a non-NVIDIA or CPU worker, and
/// on any NVIDIA GPU below the sm_120 floor — including datacenter `sm_100`. An unprobed host is
/// dense-only too: [`crate::gpu::cached_compute_cap`] answers `None`, which fails safe, so a host
/// whose probe never ran cannot license a native label by omission.
pub(crate) fn host_native_execution_capability() -> NativeExecutionCapabilityFact {
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        if crate::gpu::compute_cap_meets_nvfp4(crate::gpu::cached_compute_cap()) {
            return NativeExecutionCapabilityFact::new([NVFP4_CODEC_ID])
                .expect("NVFP4_CODEC_ID is a codec id");
        }
    }
    NativeExecutionCapabilityFact::dense_only()
}

/// What this host materialized the load as — the engine's receipt on the runtime handle translated
/// row-for-row into the SceneWorks wire vocabulary — or the explicit statement that nothing
/// measured the load.
///
/// # The translation this seam was designed for
///
/// The measurement is merged inference-side (PR #806) and, since the runtime-handle surface landed
/// (inference PR #813/#821), exposed on the object a worker actually holds:
/// `gen_core::Generator::checkpoint_weight_facts()` answers with the validated
/// `CheckpointWeightFacts` of the most recent materialization. A caller with a loaded generator
/// passes `model.checkpoint_weight_facts()` here; a caller with no handle passes `None` and gets
/// the same honest [`MaterializationUnavailable::NoRuntimeReceipt`] this function always returned.
///
/// Rows translate faithfully, one [`MaterializedCodecFact`] per measured `(codec,
/// representation)` row — including the genuine `dense-fallback` rows the engine reports for
/// `Packed`-priced projections it demoted to dense W4A16 at construction time. The representation
/// is matched exhaustively rather than round-tripped through labels, so a new engine
/// representation is a compile error here, never a silently dropped row. `complete` is the
/// engine's own whole-tensor-surface judgement (`CheckpointWeightFacts::is_complete`).
///
/// **What it must never do** is fill a `None` in. Returning a `Reported` row derived from
/// [`host_native_execution_capability`] would assert a native run nobody observed on exactly the
/// hosts where the claim is most tempting and least checkable; returning a `DenseFallback` row
/// would assert a dense run nobody observed either. `None` from the handle means "nothing measured
/// this load" — a directory-sourced import, a packed-tier variant resolved to a folder, a provider
/// that has not adopted the seam, or a lazy provider before its first materialization — and the
/// consumers are built to render it as "not measured".
pub(crate) fn materialization_from_runtime(
    facts: Option<gen_core::CheckpointWeightFacts>,
) -> MaterializationFact {
    let Some(facts) = facts else {
        return MaterializationFact::Unavailable {
            reason: MaterializationUnavailable::NoRuntimeReceipt,
        };
    };
    MaterializationFact::Reported {
        rows: facts
            .materialized()
            .iter()
            .map(|row| MaterializedCodecFact {
                codec_id: row.codec_id.to_owned(),
                representation: match row.representation {
                    gen_core::ExecutionRepresentation::NativePacked => {
                        ExecutionRepresentation::NativePacked
                    }
                    gen_core::ExecutionRepresentation::DenseFallback => {
                        ExecutionRepresentation::DenseFallback
                    }
                },
                tensor_count: row.tensor_count,
                source_bytes: row.source_bytes,
                resident_bytes: row.resident_bytes,
            })
            .collect(),
        complete: facts.is_complete(),
    }
}

/// The whole fact set for one imported checkpoint request **with the runtime handle in hand** —
/// the post-load sibling of [`imported_checkpoint_facts`] (sc-11045).
///
/// With engine facts, all three facts upgrade together, not just the materialization:
///
/// * The **source inventory** becomes the engine's compiled per-codec topology
///   ([`SourceCodecFact::counted`] rows), which is what lets the receipt's dense rows validate —
///   a real NVFP4 checkpoint also stores dense tensors (norms, embeddings), and a source list
///   holding only the header-classified codec would make every such receipt row read as aliasing.
/// * The **binding** prefers the engine's verified one (re-checked against the pin at read time)
///   over a fresh stat of the same path.
/// * The **materialization** is the translated receipt ([`materialization_from_runtime`]).
///
/// The one thing the engine cannot vouch for is *correlation with this entry*: an engine source
/// inventory that does not declare the codec admission resolved for the entry would mean the
/// receipt describes some other artifact (or the classification lies), so this refuses the receipt
/// and falls back to the honest header-declared, `no-runtime-receipt` fact set rather than pairing
/// facts about two different things. `None` engine facts (a lazy provider before its first read, a
/// non-adopting provider) take the same fallback.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(crate) fn runtime_checkpoint_facts(
    entry: &Map<String, Value>,
    checkpoint_path: Option<&Path>,
    engine_facts: Option<gen_core::CheckpointWeightFacts>,
) -> Option<CheckpointWeightFactsV1> {
    let no_receipt =
        || imported_checkpoint_facts(entry, checkpoint_path, materialization_from_runtime(None));
    let codec_id = sceneworks_core::jobs_store::imported_entry_source_codec(entry)?;
    let facts = match engine_facts {
        Some(facts) if facts.source().declares(codec_id) => facts,
        Some(facts) => {
            tracing::warn!(
                codec_id,
                engine_codecs = ?facts
                    .source()
                    .entries
                    .iter()
                    .map(|row| row.codec_id)
                    .collect::<Vec<_>>(),
                "engine receipt does not declare the entry's verified source codec; \
                 keeping the no-receipt fact set rather than correlating mismatched artifacts"
            );
            return no_receipt();
        }
        None => return no_receipt(),
    };
    let binding = facts
        .source_binding()
        .map(|binding| {
            SourceBindingFact::new(
                binding
                    .canonical_path()
                    .file_name()
                    .and_then(|name| name.to_str()),
                binding.size_bytes(),
            )
        })
        .or_else(|| checkpoint_path.and_then(source_binding_for));
    let source = facts
        .source()
        .entries
        .iter()
        .map(|row| SourceCodecFact::counted(row.codec_id, row.tensor_count, row.source_bytes))
        .collect();
    // The capability that rides a MEASURED fact set is the engine's own — the one that actually
    // licensed (or forbade) every native row in this receipt, validated against it at load time.
    // The worker probe and the engine floor agree by construction (`the_sm_120_floor_excludes_
    // sm_100`), but the probe is process-startup state; pairing the receipt with anything other
    // than the capability that judged it would let the two drift in exactly the tests and startup
    // windows where the probe has not run.
    let capability = if facts.capability().is_dense_only() {
        NativeExecutionCapabilityFact::dense_only()
    } else {
        match NativeExecutionCapabilityFact::new(facts.capability().native_codec_ids()) {
            Ok(capability) => capability,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    codec_id,
                    "engine capability failed to translate; keeping the no-receipt fact set"
                );
                return no_receipt();
            }
        }
    };
    let materialization = materialization_from_runtime(Some(facts));
    CheckpointWeightFactsV1::try_new(binding, capability, source, materialization)
        .map_err(|error| {
            // Refusal keeps the run honest, not silent: fall back to the header-declared,
            // no-receipt fact set so the asset still states its source and this host's capability.
            tracing::warn!(
                error = %error,
                codec_id,
                "runtime checkpoint weight facts refused; keeping the no-receipt fact set"
            );
        })
        .ok()
        .or_else(no_receipt)
}

/// The host-independent binding token for a resolved checkpoint file, matching the engine's
/// `SourceBinding::stable_token` rendering. `None` when the file cannot be stat'd — a token is an
/// identity claim and an unreadable file supports none.
pub(crate) fn source_binding_for(path: &Path) -> Option<SourceBindingFact> {
    let size_bytes = std::fs::metadata(path).ok()?.len();
    Some(SourceBindingFact::new(
        path.file_name().and_then(|name| name.to_str()),
        size_bytes,
    ))
}

/// Assemble the three facts for one imported checkpoint request.
///
/// `materialization` is passed in rather than computed here so that the only way a `Reported` value
/// enters is from a real receipt (see [`materialization_from_runtime`]). `None` is returned when
/// the entry declares no source codec at all — facts that cannot state what the source is have
/// nothing to correlate, and an absent fact set is what a consumer must handle for every
/// directory-sourced and builtin load anyway.
pub(crate) fn imported_checkpoint_facts(
    entry: &Map<String, Value>,
    checkpoint_path: Option<&Path>,
    materialization: MaterializationFact,
) -> Option<CheckpointWeightFactsV1> {
    // The SAME resolver admission gates on (`catalog::imported_entry_source_codec`), never a second
    // local copy of the rule. A worker-side duplicate is the sc-13542 resolver-drift class: the two
    // could disagree on an entry — a case difference is enough — and the run would route native on
    // admission's answer while the receipt and telemetry carried no source codec at all.
    let codec_id = sceneworks_core::jobs_store::imported_entry_source_codec(entry)?;
    CheckpointWeightFactsV1::try_new(
        checkpoint_path.and_then(source_binding_for),
        host_native_execution_capability(),
        vec![SourceCodecFact::declared(codec_id)],
        materialization,
    )
    .map_err(|error| {
        // Refusal is the designed outcome for an inconsistent set, not a reason to ship a
        // half-true one. Log it and carry no facts rather than carrying wrong ones.
        tracing::warn!(
            error = %error,
            codec_id,
            "checkpoint weight facts refused; omitting them from this run's receipts"
        );
    })
    .ok()
}

/// Render the fact set into the flat asset-receipt / telemetry object.
///
/// Three separate keys, never one collapsed label. `executionRepresentation` is **absent** when
/// nothing measured the load — a consumer that wants to render "unknown" reads the absence, and one
/// that wants to say "native NVFP4" cannot, because there is no value to read.
pub(crate) fn facts_receipt_object(facts: &CheckpointWeightFactsV1) -> Value {
    serde_json::to_value(facts).unwrap_or(Value::Null)
}

/// The flat asset-receipt key the three facts ride under. One nested object rather than three
/// loose keys, so a reader cannot pick up `sourceCodec` from a run and `executionRepresentation`
/// from nowhere and correlate two facts that never described the same load.
pub(crate) const FACTS_RAW_SETTINGS_KEY: &str = "checkpointWeightFacts";

/// Attach the fact set to a lane's flat raw-settings object (which becomes the asset receipt's
/// `rawAdapterSettings`). A `None` fact set inserts nothing at all — an absent key is how a
/// consumer distinguishes "this lane has no source classification" from any value it could invent.
pub(crate) fn insert_facts_into_raw_settings(
    raw: &mut Map<String, Value>,
    facts: Option<&CheckpointWeightFactsV1>,
) {
    if let Some(facts) = facts {
        raw.insert(
            FACTS_RAW_SETTINGS_KEY.to_owned(),
            facts_receipt_object(facts),
        );
    }
}

/// The telemetry pair `(source_codec, execution_representation)` for
/// [`sceneworks_core::contracts::GenerationMetrics`].
///
/// A `generation_metrics` row is one flat record per run, so it reports the checkpoint's **primary**
/// source codec — the first (and, for every source SceneWorks classifies from a header, the only)
/// inventory row — and that codec's representation. A load storing several codecs keeps the full
/// per-codec breakdown in the asset receipt, which is not flattened.
///
/// Both halves are `None` without facts. The second is `None` whenever the load was not measured,
/// and that stays `None`: an absent representation means "not measured", which the stats surface
/// renders as unknown rather than as dense. A load that ran PART of the codec packed and part dense
/// carries the distinct `"mixed:<packed>/<dense>"` value rather than either pure label — see
/// [`sceneworks_core::checkpoint_weight_facts::ExecutionMix`].
/// Gated to the lanes that build image metrics at all: `image_jobs/base.rs` is included only where
/// a media backend is linked, so on the "neither" build (no MLX, no candle) there is no metrics
/// producer to call this and an ungated definition would be a dead-code error on that PR lane.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn facts_metrics_pair(
    facts: Option<&CheckpointWeightFactsV1>,
) -> (Option<String>, Option<String>) {
    let Some(facts) = facts else {
        return (None, None);
    };
    let Some(entry) = facts.source().first() else {
        return (None, None);
    };
    (
        Some(entry.codec_id.clone()),
        facts.representation_label(&entry.codec_id),
    )
}

/// The telemetry pair for a request's model-manifest entry — the whole producer in one call, for
/// the cross-platform metrics path.
/// Gated with [`facts_metrics_pair`], its only consumer's consumer.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn manifest_entry_metrics_pair(
    entry: &Map<String, Value>,
) -> (Option<String>, Option<String>) {
    facts_metrics_pair(
        imported_checkpoint_facts(entry, None, materialization_from_runtime(None)).as_ref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(quant: Option<&str>) -> Map<String, Value> {
        let mut entry = Map::new();
        entry.insert("family".to_owned(), json!("krea_2"));
        if let Some(quant) = quant {
            entry.insert("importQuantFormat".to_owned(), json!(quant));
        }
        entry
    }

    /// The facts producer resolves the source codec through the SAME helper admission gates on,
    /// so the two cannot answer differently for one entry.
    ///
    /// The case rows are the witness: the shared resolver trims and lowercases, so `"NVFP4"` and
    /// `" NVFP4 "` are `nvfp4-v1` here exactly as they are at admission. A worker-local
    /// exact-match copy answered `None` for both while admission routed native — the drift this
    /// asserts against (sc-13542 class). A facts object is built iff a codec resolves, so the
    /// `Some`/`None` split below is also the presence/absence of the whole fact set.
    #[test]
    fn the_source_codec_comes_from_the_verified_classification_not_the_tier() {
        let codec = |quant: Option<&str>| {
            imported_checkpoint_facts(
                &entry(quant),
                None,
                MaterializationFact::Unavailable {
                    reason: MaterializationUnavailable::NoRuntimeReceipt,
                },
            )
            .map(|facts| facts.source()[0].codec_id.clone())
        };
        assert_eq!(codec(Some("nvfp4")).as_deref(), Some(NVFP4_CODEC_ID));
        assert_eq!(
            codec(Some("int8_tensorwise_per_row")).as_deref(),
            Some("int8-per-row-v1")
        );
        assert_eq!(codec(Some("bf16")).as_deref(), Some("dense-bf16-v1"));
        // Case and surrounding whitespace are normalized, matching admission exactly.
        assert_eq!(codec(Some("NVFP4")).as_deref(), Some(NVFP4_CODEC_ID));
        assert_eq!(codec(Some(" NVFP4 ")).as_deref(), Some(NVFP4_CODEC_ID));
        // No stamp, an unknown stamp, a TIER spelling, a CODEC spelling, and a classification with
        // no proved codec all yield nothing rather than a guess.
        assert_eq!(codec(None), None);
        assert_eq!(codec(Some("q4")), None);
        assert_eq!(codec(Some("nvfp4-v1")), None);
        assert_eq!(codec(Some("fp8_e4m3")), None);
    }

    /// Pins the identity directly, entry by entry: whatever admission resolves is what the facts
    /// producer stamps, including for the spellings that only a normalizing resolver accepts.
    #[test]
    fn the_facts_producer_and_admission_read_one_resolver() {
        for quant in [
            None,
            Some("nvfp4"),
            Some("NVFP4"),
            Some(" NVFP4 "),
            Some("bf16"),
            Some("int8_tensorwise_per_row"),
            Some("q4"),
            Some("fp8_e4m3"),
            Some("nvfp4-v1"),
        ] {
            let entry = entry(quant);
            let admission = sceneworks_core::jobs_store::imported_entry_source_codec(&entry);
            let stamped = imported_checkpoint_facts(
                &entry,
                None,
                MaterializationFact::Unavailable {
                    reason: MaterializationUnavailable::NoRuntimeReceipt,
                },
            )
            .map(|facts| facts.source()[0].codec_id.clone());
            assert_eq!(
                stamped.as_deref(),
                admission,
                "facts producer and admission disagree on {quant:?}"
            );
        }
    }

    /// The `sm_120` floor this module renders is the same constant the tier picker gates on, and it
    /// excludes datacenter Blackwell. Asserted against the floor predicate directly because the
    /// live probe on this box would answer differently depending on which runner ran the test.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn the_sm_120_floor_excludes_sm_100() {
        use crate::gpu::compute_cap_meets_nvfp4;
        // sm_120 — consumer Blackwell, the hardware this leg is built and validated for.
        assert!(compute_cap_meets_nvfp4(Some(12.0)));
        assert!(compute_cap_meets_nvfp4(Some(12.1)));
        // sm_100 — datacenter Blackwell. HAS FP4 tensor cores; still outside this leg's kernel.
        assert!(
            !compute_cap_meets_nvfp4(Some(10.0)),
            "sm_100 must stay off the native gate by construction, not by accident"
        );
        // Pre-Blackwell and unprobed (CPU / non-NVIDIA / nvidia-smi absent).
        assert!(!compute_cap_meets_nvfp4(Some(8.9)));
        assert!(!compute_cap_meets_nvfp4(Some(9.0)));
        assert!(!compute_cap_meets_nvfp4(None));
    }

    /// On a host with no candle lane there is nothing to probe and the capability is dense-only.
    /// This is the macOS/MLX, CPU, and plain-CUDA-build answer.
    #[cfg(not(all(not(target_os = "macos"), feature = "backend-candle")))]
    #[test]
    fn a_host_without_the_candle_lane_is_dense_only() {
        assert!(host_native_execution_capability().is_dense_only());
    }

    #[test]
    fn the_runtime_receipt_is_unavailable_and_never_a_synthesized_native_row() {
        let materialization = materialization_from_runtime(None);
        assert!(
            matches!(
                materialization,
                MaterializationFact::Unavailable {
                    reason: MaterializationUnavailable::NoRuntimeReceipt
                }
            ),
            "the seam must state that nothing measured the load: {materialization:?}"
        );

        // And the assembled facts answer `unknown` for execution regardless of what this host
        // could do — which is the property that survives running this test on the sm_120 rig.
        let facts = imported_checkpoint_facts(&entry(Some("nvfp4")), None, materialization)
            .expect("an NVFP4-stamped entry produces facts");
        assert!(
            facts.declares(NVFP4_CODEC_ID),
            "the source fact is stated on every host"
        );
        assert_eq!(
            facts.executes_natively(NVFP4_CODEC_ID),
            None,
            "an unmeasured load is never labelled native and never labelled dense"
        );
        assert_eq!(facts.representation_label(NVFP4_CODEC_ID), None);
    }

    #[test]
    fn an_entry_without_a_classification_carries_no_facts() {
        assert!(
            imported_checkpoint_facts(&entry(None), None, materialization_from_runtime(None))
                .is_none()
        );
        assert!(imported_checkpoint_facts(
            &entry(Some("q4")),
            None,
            materialization_from_runtime(None)
        )
        .is_none());
    }

    #[test]
    fn the_receipt_object_keeps_the_three_facts_apart() {
        let facts = imported_checkpoint_facts(
            &entry(Some("nvfp4")),
            None,
            materialization_from_runtime(None),
        )
        .expect("facts");
        let object = facts_receipt_object(&facts);

        assert_eq!(object["source"][0]["codecId"], json!(NVFP4_CODEC_ID));
        assert_eq!(object["materialization"]["status"], json!("unavailable"));
        assert_eq!(
            object["materialization"]["reason"],
            json!("no-runtime-receipt")
        );
        assert!(
            object["capability"].get("nativeCodecIds").is_some(),
            "the host capability rides beside the two checkpoint facts: {object}"
        );
        // The one thing a reader must not be able to find is a representation claim.
        assert!(
            !object.to_string().contains("native-packed"),
            "an unmeasured load must not render a native label anywhere: {object}"
        );
    }

    #[test]
    fn the_metrics_pair_reports_the_source_and_leaves_the_representation_absent() {
        let facts = imported_checkpoint_facts(
            &entry(Some("nvfp4")),
            None,
            materialization_from_runtime(None),
        )
        .expect("facts");
        let (source_codec, representation) = facts_metrics_pair(Some(&facts));
        assert_eq!(source_codec.as_deref(), Some(NVFP4_CODEC_ID));
        assert_eq!(representation, None);

        // No facts at all: both halves absent, never a tier substituted for the codec.
        assert_eq!(facts_metrics_pair(None), (None, None));
    }

    /// The whole metrics producer, as `image_settings_metrics` calls it.
    #[test]
    fn the_manifest_entry_metrics_pair_reports_the_codec_and_no_representation() {
        assert_eq!(
            manifest_entry_metrics_pair(&entry(Some("nvfp4"))),
            (Some(NVFP4_CODEC_ID.to_owned()), None),
            "the stored codec is reported; the unmeasured representation stays absent"
        );
        assert_eq!(
            manifest_entry_metrics_pair(&entry(Some("int8_tensorwise_per_row"))),
            (Some("int8-per-row-v1".to_owned()), None)
        );
        // A builtin / unclassified entry contributes neither field to the telemetry row.
        assert_eq!(manifest_entry_metrics_pair(&entry(None)), (None, None));
        // And a TIER stamp never becomes a source codec.
        assert_eq!(
            manifest_entry_metrics_pair(&entry(Some("q4"))),
            (None, None)
        );
    }

    #[test]
    fn a_measured_dense_run_reports_both_halves() {
        use sceneworks_core::checkpoint_weight_facts::{
            ExecutionRepresentation, MaterializedCodecFact,
        };

        // The shape the seam will produce once the runtime exposes a receipt: a dense-only host
        // that ran an NVFP4-stored checkpoint.
        let facts = imported_checkpoint_facts(
            &entry(Some("nvfp4")),
            None,
            MaterializationFact::Reported {
                rows: vec![MaterializedCodecFact {
                    codec_id: NVFP4_CODEC_ID.to_owned(),
                    representation: ExecutionRepresentation::DenseFallback,
                    tensor_count: 588,
                    source_bytes: 32_700_000_000,
                    resident_bytes: 32_700_000_000,
                }],
                complete: true,
            },
        )
        .expect("a dense-fallback receipt is valid on every host");

        let (source_codec, representation) = facts_metrics_pair(Some(&facts));
        assert_eq!(source_codec.as_deref(), Some(NVFP4_CODEC_ID));
        assert_eq!(
            representation.as_deref(),
            Some("dense-fallback"),
            "the source stays NVFP4 while the execution says dense — the two facts, separately"
        );
    }

    /// A validated engine fact set for a **mixed-policy NVFP4 load**: one projection executed the
    /// packed W4A4 operand, one was demoted to dense W4A16 (a genuine dense-fallback row, not a
    /// synthesized one), and a dense-BF16 row rode along — the exact shape the pinned engine
    /// documents for this leg. Built through `gen_core::CheckpointWeightFacts::new`, so everything
    /// downstream consumes a receipt the engine itself would have validated.
    fn engine_mixed_facts() -> gen_core::CheckpointWeightFacts {
        use gen_core::checkpoint_codec::{
            CodecResidencyReport, CompanionRole, CompanionTensorPlan, LogicalReadMaterialization,
            LogicalTensorPlan, LogicalWeightPlan, LogicalWeightReceipt, PlannedResidency,
            ResidencyMode, TensorCodecSpec, WeightEncoding, DENSE_BF16_CODEC, NVFP4_CODEC,
        };
        use gen_core::checkpoint_facts::NativeExecutionCapability;

        let nvfp4_tensor = |key: &str, mode: ResidencyMode, resident: u64| LogicalTensorPlan {
            logical_key: key.to_owned(),
            physical_key: key.to_owned(),
            encoding: WeightEncoding::UInt8,
            shape: vec![64, 64],
            source_bytes: 2048,
            codec_id: NVFP4_CODEC.codec_id,
            resident_encoding: WeightEncoding::DenseBf16,
            codec: TensorCodecSpec::Nvfp4 {
                block_scale: format!("{key}_scale"),
                global_scale: format!("{key}_scale_2"),
                input_scale: None,
                stored_shape: [64, 64],
                logical_shape: [64, 64],
                logical_shape_declared: true,
                full_precision_matrix_mult: false,
            },
            residency: PlannedResidency {
                mode,
                resident_bytes: resident,
            },
            transform: None,
        };
        let scale_companion = |owner: &str, resident: u64| CompanionTensorPlan {
            physical_key: format!("{owner}_scale"),
            role: CompanionRole::WeightScale,
            owner_physical_key: owner.to_owned(),
            source_bytes: 256,
            resident_bytes: resident,
        };
        let plan = LogicalWeightPlan {
            mapping_id: "sc-11045-host-translation-test-v1",
            tensors: vec![
                nvfp4_tensor("packed", ResidencyMode::Packed, 2048),
                nvfp4_tensor("fallback", ResidencyMode::Dense, 8192),
                LogicalTensorPlan {
                    logical_key: "plain".to_owned(),
                    physical_key: "plain".to_owned(),
                    encoding: WeightEncoding::DenseBf16,
                    shape: vec![8, 8],
                    source_bytes: 128,
                    codec_id: DENSE_BF16_CODEC.codec_id,
                    resident_encoding: WeightEncoding::DenseBf16,
                    codec: TensorCodecSpec::Dense,
                    residency: PlannedResidency {
                        mode: ResidencyMode::Dense,
                        resident_bytes: 128,
                    },
                    transform: None,
                },
            ],
            companions: vec![
                scale_companion("packed", 256),
                scale_companion("fallback", 0),
            ],
            source_bytes: 2048 + 2048 + 128 + 256 + 256,
        };
        let row = |codec_id: &'static str,
                   representation: gen_core::ExecutionRepresentation,
                   source_bytes: u64,
                   resident_bytes: u64| CodecResidencyReport {
            codec_id,
            representation,
            tensor_count: 1,
            source_bytes,
            resident_bytes,
        };
        let receipt = LogicalWeightReceipt {
            mapping_id: "sc-11045-host-translation-test-v1",
            tensor_count: 3,
            source_bytes: 2048 + 2048 + 128 + 256 + 256,
            materialization: LogicalReadMaterialization::Materialized,
            demotions: Vec::new(),
            residency: vec![
                row(
                    DENSE_BF16_CODEC.codec_id,
                    gen_core::ExecutionRepresentation::DenseFallback,
                    128,
                    128,
                ),
                row(
                    NVFP4_CODEC.codec_id,
                    gen_core::ExecutionRepresentation::DenseFallback,
                    2048 + 256,
                    8192,
                ),
                row(
                    NVFP4_CODEC.codec_id,
                    gen_core::ExecutionRepresentation::NativePacked,
                    2048 + 256,
                    2048 + 256,
                ),
            ],
        };
        gen_core::CheckpointWeightFacts::new(
            &plan,
            NativeExecutionCapability::new([NVFP4_CODEC.codec_id]),
            receipt,
        )
        .expect("the fixture receipt is the honest receipt of its own plan")
    }

    /// **The translation, row for row (sc-11045).** Every measured engine row — the packed one,
    /// the genuine demoted-dense one, and the dense-BF16 one — becomes exactly one wire row with
    /// the same codec, representation, counts and bytes, and `complete` carries the engine's own
    /// whole-surface judgement. This is the facts-present test the seam's docs anticipated:
    /// dropping the translation (reverting to the unconditional `Unavailable`) turns the expected
    /// `Reported` into `Unavailable` and this test reds.
    #[test]
    fn the_runtime_receipt_translates_row_for_row() {
        let translated = materialization_from_runtime(Some(engine_mixed_facts()));
        let expected_row = |codec_id: &str,
                            representation: ExecutionRepresentation,
                            source_bytes: u64,
                            resident_bytes: u64| MaterializedCodecFact {
            codec_id: codec_id.to_owned(),
            representation,
            tensor_count: 1,
            source_bytes,
            resident_bytes,
        };
        assert_eq!(
            translated,
            MaterializationFact::Reported {
                rows: vec![
                    expected_row(
                        "dense-bf16-v1",
                        ExecutionRepresentation::DenseFallback,
                        128,
                        128
                    ),
                    expected_row(
                        NVFP4_CODEC_ID,
                        ExecutionRepresentation::DenseFallback,
                        2304,
                        8192
                    ),
                    expected_row(
                        NVFP4_CODEC_ID,
                        ExecutionRepresentation::NativePacked,
                        2304,
                        2304
                    ),
                ],
                complete: true,
            },
        );
    }

    /// The whole runtime fact set for a mixed-policy load: the source inventory is the engine's
    /// counted topology (which is what lets the dense-BF16 receipt row validate at all), the
    /// capability is the one that licensed the receipt, and the rendered representation is the
    /// distinct `mixed:<packed>/<dense>` value — never either pure label.
    #[test]
    fn a_mixed_policy_load_lights_the_mixed_rendering() {
        let facts =
            runtime_checkpoint_facts(&entry(Some("nvfp4")), None, Some(engine_mixed_facts()))
                .expect("a receipt declaring the entry's codec produces facts");

        assert_eq!(
            facts.representation_label(NVFP4_CODEC_ID).as_deref(),
            Some("mixed:1/1"),
            "one packed + one demoted-dense NVFP4 tensor must render as the counted mixed label"
        );
        assert_eq!(facts.executes_natively(NVFP4_CODEC_ID), Some(true));
        assert_eq!(
            facts.representation_label("dense-bf16-v1").as_deref(),
            Some("dense-fallback"),
            "the dense rows keep the engine's own pure label"
        );
        // The engine's inventory is carried with its counts, not collapsed to the header codec.
        let declared: Vec<&str> = facts
            .source()
            .iter()
            .map(|row| row.codec_id.as_str())
            .collect();
        assert_eq!(declared, ["dense-bf16-v1", NVFP4_CODEC_ID]);
        assert_eq!(facts.resident_bytes(), Some(128 + 8192 + 2304));
    }

    /// No engine facts (a lazy provider before its first read, a non-adopting provider, a
    /// directory-sourced load) keeps the exact pre-existing no-receipt behaviour, and an engine
    /// receipt for a DIFFERENT artifact — one whose inventory does not declare the entry's
    /// verified codec — is refused rather than correlated.
    #[test]
    fn a_missing_or_mismatched_receipt_falls_back_to_no_runtime_receipt() {
        for facts in [
            runtime_checkpoint_facts(&entry(Some("nvfp4")), None, None),
            runtime_checkpoint_facts(&entry(Some("int8_tensorwise_per_row")), None, {
                // The mixed fixture declares nvfp4-v1 + dense-bf16-v1 — not int8-per-row-v1.
                Some(engine_mixed_facts())
            }),
        ] {
            let facts = facts.expect("the fallback still carries the header-declared facts");
            assert!(
                matches!(
                    facts.materialization(),
                    MaterializationFact::Unavailable {
                        reason: MaterializationUnavailable::NoRuntimeReceipt
                    }
                ),
                "no correlated receipt means no materialization claim: {:?}",
                facts.materialization()
            );
        }
        // And an entry with no verified classification still produces nothing at all.
        assert!(runtime_checkpoint_facts(&entry(None), None, Some(engine_mixed_facts())).is_none());
    }

    #[test]
    fn the_source_binding_token_is_the_file_name_and_size() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("kreamania_v7.safetensors");
        std::fs::write(&path, b"0123456789").expect("write");
        let binding = source_binding_for(&path).expect("a readable file has a binding");
        assert_eq!(binding.stable_token, "kreamania_v7.safetensors@10");
        assert_eq!(binding.size_bytes, 10);
        // An unreadable path supports no identity claim.
        assert!(source_binding_for(&dir.path().join("missing.safetensors")).is_none());
    }
}
