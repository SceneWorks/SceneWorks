use super::*;
use std::collections::BTreeSet;

const LTX25_CORPUS_PATH: &str = "docs/calibration/sc-18791/ltx25-mlx-evidence.seed.json";

/// EVERY retained corpus is compiled in (sc-22666, epic 22657 E5), and the packaged store
/// loads with the byte-exact handshake against all of them.
///
/// SHAPE, not a census: the assertion is that every compiled-in entry parses as a corpus and
/// that every anchor's cited source resolves inside the list — never how many corpora exist or
/// which models they name, which would red on a retirement rather than on a defect. The
/// packaging-lapse guard that keeps a RETAINED corpus from being left out lives in
/// `scripts/extract-memory-anchors.mjs` (`assertEveryDerivableCorpusIsPackaged`), which can see
/// the walked tree this crate cannot.
#[test]
fn every_packaged_corpus_parses_and_backs_the_store_it_is_compiled_in_for() {
    assert!(!PACKAGED_MEMORY_ANCHOR_SOURCES.is_empty());
    let mut paths = BTreeSet::new();
    for (path, raw) in PACKAGED_MEMORY_ANCHOR_SOURCES {
        assert!(paths.insert(*path), "{path} is compiled in twice");
        let parsed: serde_json::Value =
            serde_json::from_str(raw).unwrap_or_else(|error| panic!("{path}: {error}"));
        // sc-22738: a corpus carries retained RECORDS, measured lower BOUNDS, or both. A
        // footprint hard stop produces a bundle with no record at all — the render never got
        // far enough to decompose — and it is retained evidence all the same.
        let records = parsed["records"].as_array().is_some_and(|r| !r.is_empty());
        let bounds = parsed["exceededBounds"]
            .as_array()
            .is_some_and(|r| !r.is_empty());
        assert!(
            records || bounds,
            "{path} carries neither retained records nor measured bounds, so it is not a corpus"
        );
    }
    let sorted: Vec<&str> = paths.iter().copied().collect();
    let declared: Vec<&str> = PACKAGED_MEMORY_ANCHOR_SOURCES
        .iter()
        .map(|(path, _)| *path)
        .collect();
    assert_eq!(sorted, declared, "the compiled-in list must stay sorted");

    let store = packaged_memory_anchors().expect("the packaged anchor store must load");
    for (id, cited) in store
        .anchors
        .iter()
        .map(|anchor| (&anchor.id, &anchor.source.path))
        .chain(
            store
                .exceeded_bounds
                .iter()
                .map(|bound| (&bound.id, &bound.source.path)),
        )
    {
        assert!(
            paths.contains(cited.as_str()),
            "{id} cites {cited}, which is not compiled in",
        );
    }
}

fn corpus_raw() -> &'static str {
    PACKAGED_MEMORY_ANCHOR_SOURCES
        .iter()
        .find(|(path, _)| *path == LTX25_CORPUS_PATH)
        .map(|(_, raw)| *raw)
        .expect("the LTX-2.5 MLX retained corpus is compiled in")
}

struct CorpusRecord {
    id: String,
    tier: String,
    transformer_variant: Ltx25TransformerVariant,
    decoder: Ltx25Decoder,
    width: u32,
    height: u32,
    frames: u32,
    decode_tiled: bool,
    transformer_windowed: bool,
    deferred: bool,
    conditioning_active: u64,
    denoise_active: u64,
    decode_active: u64,
}

/// Every retained, runtime-complete LTX-2.5 MLX measured record — the validation corpus.
fn retained_corpus() -> Vec<CorpusRecord> {
    let source: serde_json::Value =
        serde_json::from_str(corpus_raw()).expect("retained corpus parses");
    let records = source["records"].as_array().expect("corpus records");
    records
        .iter()
        .filter(|record| {
            record["target"]["modelId"].as_str() == Some("ltx_2_5")
                && record["backend"].as_str() == Some("mlx")
                && record["status"].as_str() == Some("runtime_complete")
        })
        .map(|record| {
            let geometry = &record["target"]["geometry"];
            let engaged: Vec<&str> = record["strategy"]["engagedRungs"]
                .as_array()
                .expect("engaged rungs")
                .iter()
                .filter_map(|value| value.as_str())
                .collect();
            let measured: std::collections::BTreeMap<&str, u64> = record["diagnostics"]
                ["measurements"]
                .as_array()
                .expect("measurements")
                .iter()
                .filter_map(|entry| Some((entry["name"].as_str()?, entry["value"].as_u64()?)))
                .collect();
            CorpusRecord {
                id: record["id"].as_str().expect("record id").to_owned(),
                tier: record["target"]["tier"].as_str().expect("tier").to_owned(),
                transformer_variant: match record["target"]["transformerVariant"].as_str() {
                    Some("dev") => Ltx25TransformerVariant::Dev,
                    Some("distilled") => Ltx25TransformerVariant::Distilled,
                    other => panic!("unknown transformer variant {other:?}"),
                },
                decoder: match record["target"]["decoder"].as_str() {
                    Some("conv") => Ltx25Decoder::Conv,
                    Some("diffvae") => Ltx25Decoder::DiffVae,
                    other => panic!("unknown decoder {other:?}"),
                },
                width: geometry["width"].as_u64().expect("width") as u32,
                height: geometry["height"].as_u64().expect("height") as u32,
                frames: geometry["frames"].as_u64().expect("frames") as u32,
                decode_tiled: engaged.contains(&"bounded_decode"),
                transformer_windowed: engaged.contains(&"bounded_transformer_residency"),
                deferred: record["loadShape"].as_str() == Some("deferred_materialization"),
                conditioning_active: measured["conditioningActivePeak"],
                denoise_active: measured["denoiseActivePeak"],
                decode_active: measured["decodeActivePeak"],
            }
        })
        .collect()
}

fn store() -> &'static MemoryAnchorStore {
    packaged_memory_anchors().expect("the packaged anchor store loads")
}

/// Index of an LTX-2.5 anchor in the packaged store. Since sc-22510 the store spans the whole
/// catalog, so the doctoring tests below cannot assume `anchors[0]` is a video row.
fn ltx25_anchor_index() -> usize {
    store()
        .anchors
        .iter()
        .position(|anchor| anchor.model_id == "ltx_2_5" && anchor.source.path == LTX25_CORPUS_PATH)
        .expect("the store carries an LTX-2.5 anchor")
}

// -------------------------------------------------------------------------------------
// AC 1 (store half): every LTX-2.5 (tier, transformer variant, decoder) cell the retained
// corpus measures, ON EITHER LANE, carries exactly one anchor, and no cell it does not
// measure carries any.
// -------------------------------------------------------------------------------------

/// Every `(tier, transformer variant, decoder)` cell that ANY packaged corpus measures for
/// LTX-2.5 on `lane` — the sc-18791 seed corpus and, since the sc-22738 campaign, the
/// per-cell corpora under `docs/calibration/sc-22738/`. Derived from the compiled-in list
/// rather than pinned, so a campaign that lands a new pipeline cell (the rerun's dev/conv
/// bf16 and q4) extends the expectation instead of reddening it.
///
/// sc-22738: LANE-PARAMETERISED, and `status` is no longer read. Both changes are the same
/// correction. This used to hard-code `mlx` and `status == "runtime_complete"`, which held
/// only while MLX was the sole lane carrying LTX-2.5 anchors; the candle campaign landed
/// `ltx_2_5:candle` bf16/q4/q8, whose records are `status: "gated"`, so a candle lane bolted
/// onto the old predicate would have derived ZERO measured cells against three live anchors.
///
/// `status` is not a candidacy rule anywhere in the anchor pass — `anchorCandidate` in
/// `scripts/extract-memory-anchors.mjs` never reads the field, and all three spellings the
/// packaged corpora use (`gated`, `runtime_complete`, `complete`) produce anchors today.
/// Filtering on it here was a SECOND, narrower opinion about what anchors, and this is the
/// bug that made the candle rows invisible to the guard below. What a record must have to
/// anchor is its lane's phase measurements, and re-deriving that rule in Rust would just be
/// a third opinion — so the cell set is the population, and the store is what is checked
/// against it.
fn measured_ltx25_cells(lane: &str) -> std::collections::BTreeSet<(String, String, String)> {
    PACKAGED_MEMORY_ANCHOR_SOURCES
        .iter()
        .flat_map(|(_, raw)| {
            let source: serde_json::Value = serde_json::from_str(raw).expect("corpus parses");
            source["records"]
                .as_array()
                .map(|records| records.to_vec())
                .unwrap_or_default()
        })
        .filter(|record| {
            record["target"]["modelId"].as_str() == Some("ltx_2_5")
                && record["backend"].as_str() == Some(lane)
        })
        .map(|record| {
            (
                record["target"]["tier"].as_str().expect("tier").to_owned(),
                record["target"]["transformerVariant"]
                    .as_str()
                    .unwrap_or("-")
                    .to_owned(),
                record["target"]["decoder"]
                    .as_str()
                    .unwrap_or("-")
                    .to_owned(),
            )
        })
        .collect()
}

// Synthetic CPU protocol fixture, never a measurement or committed calibration receipt.
fn accounting_protocol_fixture() -> ObservedOperatingEnvelope {
    let phase = |first, last| {
        serde_json::json!({
            "dedicatedPeakBytes": 80, "sharedPeakBytes": 30, "totalCommittedPeakBytes": 90,
            "samples": 2, "firstSampleMs": first, "lastSampleMs": last, "maxGapMs": last-first
        })
    };
    serde_json::from_value(serde_json::json!({
            "id": "cpu-protocol-only", "modelId": "flux3_klein",
            "recipe": { "schemaVersion": 1, "backend": "candle", "provider": "flux3_klein", "tier": "bf16",
                "actualLoadSpecSha256": "1".repeat(64), "actualGenerationRequestSha256": "2".repeat(64),
                "invocationRequestSha256": "3".repeat(64), "measuredStrategySha256": "4".repeat(64) },
            "accounting": {
                "schemaVersion": 1, "kind": "flux3_owned_process_memory_accounting_v1",
                "source": "Windows GPU Process Memory performance counters",
                "semantics": "sampled per-process dedicated/shared/total-committed accounting; not physical residency or CUDA active/reserved allocator bytes",
                "attribution": "own PID/start-time/executable and CUDA adapter LUID; includes any cross-process-shared allocations charged to this process",
                "physicalResidencyClaim": false, "allocatorActivityClaim": false,
                "deviceTotalDeltaAttributionClaim": false, "exclusiveThroughputClaim": false,
                "sampling": { "intervalMs": 200, "maxAllowedGapMs": 1500, "maxAllowedReadSkewMs": 250, "maxObservedReadSkewMs": 10, "losslessPeakClaim": false },
                "identity": { "pid": 42, "startedAt": "2026-10-23T00:00:00.000Z", "executable": "cpu-fixture.exe",
                    "luid": "0x00000000_0x0001f363", "physicalGpuId": "1", "gpuUuid": "GPU-00000000-0000-0000-0000-000000000001", "pciBusId": "00000000:21:00.0", "logicalCudaOrdinal": 0 },
                "physicalNodes": [0, 1],
                "phases": { "load": phase(0,200), "conditioning": phase(200,400), "denoise": phase(400,600), "decode": phase(600,800) },
                "overall": { "dedicatedPeakBytes": 80, "sharedPeakBytes": 30, "totalCommittedPeakBytes": 90,
                    "samples": 5, "firstSampleMs": 0, "lastSampleMs": 800, "maxGapMs": 200 },
                "unavailableMetrics": { "physicalResidentBytes": null, "activeBytes": null, "allocatorBytes": null }
            },
            "operatingConditions": { "schemaVersion": 1, "sharingMode": "bounded-shared", "physicalGpuId": "1",
                "gpuUuid": "GPU-00000000-0000-0000-0000-000000000001", "pciBusId": "00000000:21:00.0", "observedAt": "2026-10-23T00:00:00.000Z",
                "totalDeviceBytes": 1000, "freeDeviceBytes": 800, "foreignDedicatedBytes": 20,
                "foreignProcessCount": 1, "foreignMaxProcessBytes": 20,
                "foreignProcessLimitBytes": 1073741824_u64, "foreignTotalLimitBytes": 3221225472_u64,
                "observationScope": "producer-reservation-admission", "fitAuthorization": false,
                "exclusiveThroughputClaim": false, "physicalResidencyClaim": false },
            "source": { "path": "cpu-fixture-only", "sha256": "5".repeat(64), "recordId": "cpu-only" }
        })).unwrap()
}

#[test]
fn sampled_accounting_is_distinct_typed_observation_not_residency() {
    let fixture = accounting_protocol_fixture();
    fixture.validate().unwrap();
    // Independent peaks are deliberately nonadditive: 80 + 30 != 90 remains valid.
    let value = serde_json::to_value(&fixture.accounting).unwrap();
    assert!(value.get("activeBytes").is_none());
    assert!(value["unavailableMetrics"]["physicalResidentBytes"].is_null());
    assert!(
        serde_json::from_value::<MemoryAnchor>(serde_json::to_value(&fixture).unwrap()).is_err()
    );
    let mut store = bound_store(Vec::new());
    store.observed_operating_envelopes.push(fixture.clone());
    assert_eq!(
        store
            .observed_operating_envelopes_for("flux3_klein", &fixture.recipe)
            .count(),
        1
    );
    let mut changed = fixture.recipe.clone();
    changed.measured_strategy_sha256 = "6".repeat(64);
    assert_eq!(
        store
            .observed_operating_envelopes_for("flux3_klein", &changed)
            .count(),
        0
    );
    changed = fixture.recipe.clone();
    changed.actual_generation_request_sha256 = "7".repeat(64);
    assert_eq!(
        store
            .observed_operating_envelopes_for("flux3_klein", &changed)
            .count(),
        0
    );
    assert!(store
        .image_anchor_for("flux3_klein", AnchorBackend::Candle, "bf16")
        .is_none());
    // A valid typed object is not enough: it still needs compiled, byte-exact provenance.
    assert!(validate_observed_operating_envelopes(&store)
        .unwrap_err()
        .contains("unpackaged"));
}

#[test]
fn sampled_accounting_rejects_identity_coverage_and_residency_mutants() {
    let fixture = accounting_protocol_fixture();
    for mutate in [
        |entry: &mut ObservedOperatingEnvelope| entry.accounting.identity.pid = 0,
        |entry: &mut ObservedOperatingEnvelope| {
            entry.accounting.identity.started_at = "2026-02-30T00:00:00.000Z".into()
        },
        |entry: &mut ObservedOperatingEnvelope| entry.accounting.identity.luid = "foreign".into(),
        |entry: &mut ObservedOperatingEnvelope| entry.accounting.physical_nodes.push(0),
        |entry: &mut ObservedOperatingEnvelope| entry.accounting.physical_residency_claim = true,
        |entry: &mut ObservedOperatingEnvelope| entry.accounting.allocator_activity_claim = true,
        |entry: &mut ObservedOperatingEnvelope| {
            entry.accounting.sampling.lossless_peak_claim = true
        },
        |entry: &mut ObservedOperatingEnvelope| entry.accounting.phases.denoise.samples = 1,
        |entry: &mut ObservedOperatingEnvelope| entry.accounting.phases.denoise.max_gap_ms = 1501,
        |entry: &mut ObservedOperatingEnvelope| {
            entry.accounting.phases.denoise.last_sample_ms = 400
        },
        |entry: &mut ObservedOperatingEnvelope| {
            entry.accounting.phases.denoise.dedicated_peak_bytes = 81
        },
        |entry: &mut ObservedOperatingEnvelope| entry.accounting.phases.denoise.first_sample_ms = 0,
        |entry: &mut ObservedOperatingEnvelope| entry.operating_conditions.fit_authorization = true,
        |entry: &mut ObservedOperatingEnvelope| {
            entry.operating_conditions.gpu_uuid = "GPU-other".into()
        },
        |entry: &mut ObservedOperatingEnvelope| entry.operating_conditions.free_device_bytes = 1001,
        |entry: &mut ObservedOperatingEnvelope| entry.recipe.measured_strategy_sha256.clear(),
    ] {
        let mut mutant = fixture.clone();
        mutate(&mut mutant);
        assert!(mutant.validate().is_err(), "accepted mutant {mutant:?}");
    }
    let mut value = serde_json::to_value(&fixture.accounting).unwrap();
    value["unavailableMetrics"]["activeBytes"] = serde_json::json!(80);
    assert!(serde_json::from_value::<OwnedProcessAccountingEvidence>(value).is_err());
    let mut value = serde_json::to_value(&fixture.accounting).unwrap();
    value["allocatorBytes"] = serde_json::json!(90);
    assert!(serde_json::from_value::<OwnedProcessAccountingEvidence>(value).is_err());
}

#[test]
fn ltx25_carries_exactly_one_anchor_per_measured_pipeline_cell() {
    // Shape, not a frozen count, and MLX-scoped on purpose: the MLX corpus spans more than
    // one variant and more than one decoder, so the guard below cannot silently degenerate
    // into a single-cell assertion on that lane. It is NOT asserted cross-lane — the candle
    // campaign measured one variant (`distilled`) and one decoder (`conv`), so demanding a
    // spread there would freeze a population rather than describe a shape.
    let mlx_measured = measured_ltx25_cells(AnchorBackend::Mlx.as_key());
    assert!(
        mlx_measured
            .iter()
            .map(|(_, variant, _)| variant.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            > 1,
        "the retained MLX corpus must span more than one transformer variant"
    );
    assert!(
        mlx_measured
            .iter()
            .map(|(_, _, decoder)| decoder.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            > 1,
        "the retained MLX corpus must span more than one decoder"
    );

    // sc-22738: BOTH lanes. The uniqueness half used to run on MLX alone while the citation
    // loop below had already been widened, so the three `ltx_2_5:candle` anchors the candle
    // campaign landed were checked for provenance but never for one-anchor-per-cell. A second
    // candle anchor on a cell that already had one was admissible.
    for lane in [AnchorBackend::Mlx, AnchorBackend::Candle] {
        let measured = measured_ltx25_cells(lane.as_key());
        assert!(
            !measured.is_empty(),
            "{}: the packaged corpora must measure at least one LTX-2.5 pipeline cell on \
                 this lane — an empty population makes every assertion below vacuous",
            lane.as_key()
        );
        for (tier, variant_key, decoder_key_str) in &measured {
            let matching = store()
                .anchors
                .iter()
                .filter(|anchor| {
                    anchor.model_id == "ltx_2_5"
                        && anchor.backend == lane
                        && anchor.tier == *tier
                        && variant_key_opt(anchor.transformer_variant) == variant_key.as_str()
                        && decoder_key_opt(anchor.decoder) == decoder_key_str.as_str()
                })
                .count();
            assert_eq!(
                matching,
                1,
                "{}: cell ({tier}, {variant_key}, {decoder_key_str}) must carry exactly one \
                     anchor",
                lane.as_key()
            );
        }
        assert_eq!(
            store()
                .anchors
                .iter()
                .filter(|anchor| anchor.model_id == "ltx_2_5" && anchor.backend == lane)
                .count(),
            measured.len(),
            "{}: the store must not carry an anchor for a pipeline cell the corpus never \
                 measured",
            lane.as_key()
        );
    }
    // Scoped to LTX-2.5: since sc-22510 the store spans the whole routing catalog, so this
    // asks that every LTX-2.5 row still comes from a packaged corpus that carries a completed
    // LTX-2.5 record ON THAT ROW'S OWN LANE — the seed corpus or a sc-22738 campaign corpus —
    // and nothing more.
    //
    // sc-22738: this loop used to demand an `mlx` record of EVERY LTX-2.5 anchor regardless of
    // the anchor's backend, which held only while MLX was the sole lane carrying LTX-2.5
    // anchors. The candle campaign landed ltx_2_5:candle bf16/q4/q8, each citing a candle
    // corpus, and the MLX-keyed literal reddened on them. Keying the required record on
    // `anchor.backend` keeps the guard on both lanes instead of exempting the new one: a
    // candle anchor citing an MLX-only corpus (or the reverse) still fails.
    for anchor in store()
        .anchors
        .iter()
        .filter(|anchor| anchor.model_id == "ltx_2_5")
    {
        let lane = anchor.backend.as_key();
        let cites_ltx25_corpus = PACKAGED_MEMORY_ANCHOR_SOURCES
            .iter()
            .find(|(path, _)| *path == anchor.source.path)
            .map(|(_, raw)| {
                let source: serde_json::Value = serde_json::from_str(raw).expect("corpus parses");
                source["records"].as_array().is_some_and(|records| {
                    records.iter().any(|record| {
                        record["target"]["modelId"].as_str() == Some("ltx_2_5")
                            && record["backend"].as_str() == Some(lane)
                    })
                })
            })
            .unwrap_or(false);
        assert!(
            cites_ltx25_corpus,
            "{}: an LTX-2.5 anchor must cite a packaged corpus carrying LTX-2.5 {lane} records",
            anchor.id
        );
    }
}

// -------------------------------------------------------------------------------------
// sc-22510: the store is catalog-wide, and every row is one of exactly two kinds.
// -------------------------------------------------------------------------------------

/// The store's shape, asserted without pinning any population.
///
/// sc-22512: this deliberately does NOT require the store to declare any analytic-only row. A
/// cell nobody measured and nobody classified is priced from the conservative analytic estimate
/// at admission time; its absence from this file is never a build failure. What IS asserted is
/// the shape of the rows that ARE present, which holds at any store size including zero.
#[test]
fn every_store_row_is_well_formed_whatever_the_store_holds() {
    // sc-22512 removed the three population floors that stood here: "the store must keep at
    // least one pipeline-keyed (video) anchor", "must carry anchors extracted from corpora that
    // state no pipeline axes", and "the migration covers more than one model". Each reddened for
    // exactly one reason — a MEASUREMENT not being in the store — so retiring a corpus, or
    // shipping a catalog nobody has captured yet, broke the build over bookkeeping. Under E8 an
    // unanchored cell is priced from the conservative analytic estimate at admission time; its
    // absence here is never a failure.
    //
    // Every claim below is universally quantified, so it keeps full force over whatever rows the
    // store holds and degrades to vacuous rather than to red when it holds none.
    for anchor in &store().anchors {
        assert!(
            !anchor.model_id.trim().is_empty() && !anchor.tier.trim().is_empty(),
            "an anchor must name the cell it prices"
        );
        // A pipeline-keyed anchor states BOTH axes or NEITHER. Half a key would answer a
        // variant-keyed lookup it was never measured under.
        assert_eq!(
            anchor.transformer_variant.is_some(),
            anchor.decoder.is_some(),
            "{}: an anchor's pipeline axes are stated together or not at all",
            anchor.id
        );
    }
    // Shape, not population: every analytic-only row that IS present states a reason and does
    // not collide with an anchored cell. Zero such rows is a legal store.
    for entry in &store().analytic_only {
        assert!(
            !entry.reason.trim().is_empty(),
            "{}: an analytic-only classification must state its reason",
            entry.id
        );
        assert!(
            !store().anchors.iter().any(|anchor| {
                anchor.model_id == entry.model_id
                    && anchor.backend == entry.backend
                    && anchor.tier == entry.tier
            }),
            "{}: a cell is classified exactly once — it is anchored and analytic-only",
            entry.id
        );
    }
}

/// An anchor whose source record stated NO pipeline axes must not answer a variant-keyed
/// lookup: it certifies neither variant, and standing in for one would price a request from a
/// render that never claimed to be it.
///
/// sc-22512 (E8): the subject is FOUND, not required. A packaged store in which every anchor
/// states its axes carries no axis-free row for this harness to interrogate — that is a corpus
/// nobody measured that way, not a defect, so the test withholds its question instead of
/// reddening. The lookup rule keeps full force on every axis-free row the store does carry.
#[test]
fn an_axis_free_anchor_answers_no_variant_keyed_lookup() {
    let Some(axis_free) = store()
        .anchors
        .iter()
        .find(|anchor| anchor.transformer_variant.is_none())
    else {
        return;
    };
    for variant in [
        Ltx25TransformerVariant::Dev,
        Ltx25TransformerVariant::Distilled,
    ] {
        for decoder in [Ltx25Decoder::Conv, Ltx25Decoder::DiffVae] {
            assert!(
                store()
                    .anchor_for(
                        &axis_free.model_id,
                        axis_free.backend,
                        &axis_free.tier,
                        variant,
                        decoder
                    )
                    .is_none(),
                "an axis-free anchor must not resolve for ({variant:?}, {decoder:?})"
            );
        }
    }
}

/// A cell is classified exactly once. Declaring an anchored cell analytic-only would demote a
/// measured cell to an architecture estimate with the anchor sitting unused beside it.
#[test]
fn an_analytic_entry_for_an_anchored_cell_is_rejected() {
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    let anchor = doctored["anchors"][0].clone();
    doctored["analyticOnly"]
        .as_array_mut()
        .expect("analytic-only rows")
        .push(serde_json::json!({
            "id": "analytic:collision",
            "modelId": anchor["modelId"],
            "modelFamily": anchor["modelFamily"],
            "route": anchor["route"],
            "backend": anchor["backend"],
            "tier": anchor["tier"],
            "basis": "no_retained_evidence",
            "reason": "collides with a measured anchor",
            "evidence": serde_json::Value::Null,
        }));
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("an anchored cell may not also be analytic-only");
    assert!(error.contains("classified exactly once"), "{error}");
}

#[test]
fn a_duplicate_or_unexplained_analytic_entry_is_rejected() {
    // sc-22512: this adversarial harness doctors a REAL row, so it can only ask its question
    // when the corpus supplies one. It used to index `analyticOnly[0]` unconditionally and
    // panicked on an empty list — a store with nothing classified reddened the suite, which is
    // the measurement-absence failure this story removes. Skipping is the E8 posture: absence
    // withholds the question, it never answers it with a failure. The loader rules themselves
    // are unchanged and keep full force on every row the store does carry.
    if store().analytic_only.is_empty() {
        return;
    }
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    let clone = doctored["analyticOnly"][0].clone();
    doctored["analyticOnly"]
        .as_array_mut()
        .expect("analytic-only rows")
        .push(clone);
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("a duplicated analytic cell must be rejected");
    assert!(error.contains("duplicate analytic-only entry"), "{error}");

    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    doctored["analyticOnly"][0]["reason"] = serde_json::json!("   ");
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("an unexplained classification must be rejected");
    assert!(error.contains("states no reason"), "{error}");
}

/// The basis and the cited evidence must agree in both directions, and a foreign-repo citation
/// must name the revision it was read at.
#[test]
fn an_analytic_basis_that_disagrees_with_its_evidence_is_rejected() {
    // sc-22512: same posture as the harness above. Each leg doctors a row of a specific KIND,
    // so it runs exactly when the corpus carries that kind and is skipped when it does not.
    // The `.expect("an evidenced analytic-only entry exists")` pair that stood here made a
    // corpus with nothing classified — or with only unevidenced classifications — red the
    // suite, which is failure-on-absence rather than on anything malformed.
    let (Some(evidenced), Some(unevidenced)) = (
        store()
            .analytic_only
            .iter()
            .position(|entry| entry.evidence.is_some()),
        store()
            .analytic_only
            .iter()
            .position(|entry| entry.evidence.is_none()),
    ) else {
        return;
    };

    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    doctored["analyticOnly"][evidenced]["basis"] = serde_json::json!("no_retained_evidence");
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("claiming no evidence while citing some must be rejected");
    assert!(error.contains("claims no retained evidence"), "{error}");

    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    doctored["analyticOnly"][unevidenced]["basis"] = serde_json::json!("measured_envelope");
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("naming a basis with no evidence must be rejected");
    assert!(error.contains("cites no evidence"), "{error}");

    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    doctored["analyticOnly"][evidenced]["evidence"]["repo"] =
        serde_json::json!("SceneWorks/inference");
    doctored["analyticOnly"][evidenced]["evidence"]["revision"] = serde_json::Value::Null;
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("a foreign citation without a revision must be rejected");
    assert!(error.contains("without the revision"), "{error}");
}

/// A repo-local analytic citation of a COMPILED-IN corpus gets the anchors' byte-exact
/// handshake: the store may not cite evidence it has drifted from.
#[test]
fn an_analytic_citation_of_a_compiled_source_binds_to_its_digest() {
    // The store need not contain such a row: which cells are analytic-only, and what they cite,
    // is a property of the retained corpus, and a measurement leaving it must not red this
    // test. The assertion is about rows that ARE present.
    let Some(index) = store().analytic_only.iter().position(|entry| {
        entry.evidence.as_ref().is_some_and(|evidence| {
            evidence.repo.is_none()
                && PACKAGED_MEMORY_ANCHOR_SOURCES
                    .iter()
                    .any(|(path, _)| *path == evidence.path)
        })
    }) else {
        return;
    };
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    doctored["analyticOnly"][index]["evidence"]["sha256"] = serde_json::json!("0".repeat(64));
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("a drifted analytic citation must be rejected");
    assert!(error.contains("source digest mismatch"), "{error}");
}

/// The pipeline cell is part of the lookup key, not a post-filter: the corpus contains no
/// dev-vs-distilled pair at a common regime, so the variant effect is unmeasurable and one
/// variant's anchor must never resolve for the other's request.
#[test]
fn a_foreign_pipeline_cell_resolves_no_anchor() {
    // Measured at q8: dev/diffvae and dev/conv. Never measured at q8: either distilled combo.
    assert!(store()
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q8",
            Ltx25TransformerVariant::Dev,
            Ltx25Decoder::DiffVae
        )
        .is_some());
    assert!(store()
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q8",
            Ltx25TransformerVariant::Distilled,
            Ltx25Decoder::Conv
        )
        .is_none());
    // Measured at q4: dev/diffvae (seed corpus) and, since the sc-22738 rerun, dev/conv.
    // Never measured at q4: either distilled combo.
    assert!(store()
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q4",
            Ltx25TransformerVariant::Dev,
            Ltx25Decoder::Conv
        )
        .is_some());
    assert!(store()
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q4",
            Ltx25TransformerVariant::Distilled,
            Ltx25Decoder::Conv
        )
        .is_none());
}

/// An anchor measured in a BOUNDED regime carries a truncated intercept for that phase. It may
/// not price the unbounded request that would reuse it.
///
/// sc-22512 (E8): the bf16 distilled/conv cell is looked up, not required. A corpus that never
/// measured that cell is absence — the request simply prices from the caller's floor — so the
/// harness withholds its question rather than reddening. Every regime-handshake assertion below
/// keeps full force whenever the cell IS measured.
#[test]
fn an_anchor_measured_in_a_bounded_regime_refuses_the_unbounded_request() {
    let Some(bounded) = store().anchor_for(
        "ltx_2_5",
        AnchorBackend::Mlx,
        "bf16",
        Ltx25TransformerVariant::Distilled,
        Ltx25Decoder::Conv,
    ) else {
        return;
    };
    assert_eq!(bounded.load_shape, AnchorLoadShape::DeferredMaterialization);
    assert!(bounded.measured_regime.decode_tiled);
    assert!(bounded.measured_regime.transformer_windowed);

    let matching = AnchorDeriveRequest {
        width: bounded.geometry.width,
        height: bounded.geometry.height,
        frames: bounded.geometry.frames,
        decode_tiled: true,
        transformer_windowed: true,
        deferred_materialization: true,
    };
    assert!(bounded.derive_video_phase_peaks(matching).is_some());
    for (label, request) in [
        (
            "eager request against a deferred anchor",
            AnchorDeriveRequest {
                deferred_materialization: false,
                ..matching
            },
        ),
        (
            "unwindowed request against a windowed anchor",
            AnchorDeriveRequest {
                transformer_windowed: false,
                ..matching
            },
        ),
        (
            "single-pass request against a tiled anchor",
            AnchorDeriveRequest {
                decode_tiled: false,
                ..matching
            },
        ),
    ] {
        assert!(
            bounded.derive_video_phase_peaks(request).is_none(),
            "{label} must fall open to the caller's floor"
        );
    }
}

/// The store schema may not be doctored into a regime it was not measured in — the handshake
/// binds `loadShape` and `measuredRegime` to the source record.
#[test]
fn a_doctored_load_shape_or_measured_regime_is_rejected() {
    let ltx = ltx25_anchor_index();
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    doctored["anchors"][ltx]["loadShape"] = serde_json::json!("deferred_materialization");
    let error =
        load_memory_anchors(&doctored.to_string()).expect_err("load shape must bind to source");
    assert!(error.contains("load shape disagrees"), "{error}");

    // Every regime flag is bound, not just the two the video law reads: `staged` decides the
    // candle conditioning intercept and `attentionChunked` the denoise one.
    for field in [
        "decodeTiled",
        "transformerWindowed",
        "staged",
        "attentionChunked",
    ] {
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        let current = doctored["anchors"][ltx]["measuredRegime"][field]
            .as_bool()
            .unwrap_or_else(|| panic!("{field} is a declared regime flag"));
        doctored["anchors"][ltx]["measuredRegime"][field] = serde_json::json!(!current);
        let Err(error) = load_memory_anchors(&doctored.to_string()) else {
            panic!("{field} must bind to the engaged rungs");
        };
        assert!(
            error.contains("measured regime disagrees"),
            "{field}: {error}"
        );
    }
}

/// The candle anchor's own bound fields: the phase peaks are read under the CANDLE measurement
/// names, and a still-image anchor may not invent an output rate.
#[test]
fn a_doctored_candle_anchor_measurement_binding_is_rejected() {
    let index = store()
        .anchors
        .iter()
        .position(|anchor| anchor.backend == AnchorBackend::Candle)
        .expect("the packaged store carries a candle anchor");
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    doctored["anchors"][index]["geometry"]["fps"] = serde_json::json!(24);
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("a still-image anchor must not declare an fps its record never measured");
    assert!(error.contains("disagrees with the outputFps"), "{error}");

    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    let peak = doctored["anchors"][index]["phaseActivePeakBytes"]["denoise"]
        .as_u64()
        .expect("denoise peak");
    doctored["anchors"][index]["phaseActivePeakBytes"]["denoise"] = serde_json::json!(peak + 1);
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("the candle phase peaks must bind to the device-delta measurements");
    assert!(error.contains("peak bytes disagree"), "{error}");

    // The lane's measurement names are chosen by the anchor's backend, so relabelling the
    // anchor as MLX must fail rather than silently reading a different measurement set. The
    // record's own `backend` field is the FIRST conjunct that sees the relabel, so the exact
    // error is the identity one — asserting merely "disagree" would also have accepted the
    // peak-bytes failure and could not tell which guard fired.
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    doctored["anchors"][index]["backend"] = serde_json::json!("mlx");
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("a relabelled lane must not resolve the other lane's measurements");
    assert!(error.contains("identity disagrees"), "{error}");

    // Overlay and reference shape are identity conjuncts of both anchor lookups, so they bind
    // to the record too rather than being stored and trusted.
    for (field, value) in [
        ("overlay", serde_json::json!("identity")),
        (
            "referenceCount",
            serde_json::json!(store().anchors[index].reference_count + 1),
        ),
    ] {
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][index][field] = value;
        let error = load_memory_anchors(&doctored.to_string())
            .err()
            .unwrap_or_else(|| panic!("{field} must bind to the source record"));
        assert!(
            error.contains("overlay/reference shape disagrees"),
            "{field}: {error}"
        );
    }
}

/// Identity fields that were previously stored-and-unchecked are bound to the source record.
#[test]
fn a_doctored_pipeline_identity_provider_route_or_fps_is_rejected() {
    let ltx = ltx25_anchor_index();
    for (field, value) in [
        ("transformerVariant", serde_json::json!("distilled")),
        ("decoder", serde_json::json!("conv")),
        ("provider", serde_json::json!("someone_else")),
        ("route", serde_json::json!("ltx_2_5_other")),
    ] {
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][ltx][field] = value;
        // The doctored identity may now name a cell another packaged anchor really occupies
        // (the sc-22738 rerun anchored dev/conv at bf16, so relabelling the seed dev/diffvae
        // row's decoder collides with it). Drop that other row so the ONLY defect the loader
        // can see is the doctored row disagreeing with its own record — the duplicate-cell
        // guard is a different invariant with its own test.
        let identity = |anchor: &serde_json::Value| {
            [
                "modelId",
                "backend",
                "tier",
                "transformerVariant",
                "decoder",
                "overlay",
                "referenceCount",
            ]
            .map(|key| anchor[key].clone())
        };
        let target = identity(&doctored["anchors"][ltx]);
        let anchors = doctored["anchors"].as_array().expect("anchors").clone();
        doctored["anchors"] = serde_json::Value::Array(
            anchors
                .into_iter()
                .enumerate()
                .filter(|(index, anchor)| *index == ltx || identity(anchor) != target)
                .map(|(_, anchor)| anchor)
                .collect(),
        );
        let error = load_memory_anchors(&doctored.to_string())
            .err()
            .unwrap_or_else(|| panic!("{field} must bind to the source record"));
        assert!(error.contains("identity disagrees"), "{field}: {error}");
    }

    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    let fps = doctored["anchors"][ltx]["geometry"]["fps"]
        .as_u64()
        .expect("fps");
    doctored["anchors"][ltx]["geometry"]["fps"] = serde_json::json!(fps + 1);
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("fps must bind to the outputFps measurement");
    assert!(error.contains("disagrees with the outputFps"), "{error}");

    // sc-22510 made `fps` and the pipeline axes OPTIONAL, so both are checked in BOTH
    // directions. These two cases exercise the direction a `is_some() && ...` guard would
    // silently drop: a record that DID state a value, dropped to null in the store.
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    doctored["anchors"][ltx]["geometry"]["fps"] = serde_json::Value::Null;
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("dropping a measured output rate must be rejected");
    assert!(error.contains("disagrees with the outputFps"), "{error}");

    for axis in ["transformerVariant", "decoder"] {
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][ltx][axis] = serde_json::Value::Null;
        let error = load_memory_anchors(&doctored.to_string())
            .err()
            .unwrap_or_else(|| panic!("dropping {axis} must be rejected"));
        assert!(error.contains("identity disagrees"), "{axis}: {error}");
    }

    // And the other direction: a record that measured NO output rate — every image capture —
    // must not acquire one in the store. Whether such an anchor exists depends on which
    // captures the corpus retains, so this asks its question only when one is there rather than
    // reding when the last rateless capture is retired.
    if let Some(rateless) = store()
        .anchors
        .iter()
        .position(|anchor| anchor.geometry.fps.is_none())
    {
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][rateless]["geometry"]["fps"] = serde_json::json!(24);
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("an invented output rate must be rejected");
        assert!(error.contains("disagrees with the outputFps"), "{error}");
    }

    if let Some(axis_free) = store()
        .anchors
        .iter()
        .position(|anchor| anchor.transformer_variant.is_none())
    {
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][axis_free]["transformerVariant"] = serde_json::json!("dev");
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("an invented pipeline variant must be rejected");
        assert!(error.contains("identity disagrees"), "{error}");
    }
}

#[test]
fn a_duplicate_anchor_identity_is_rejected() {
    let ltx = ltx25_anchor_index();
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    let clone = doctored["anchors"][ltx].clone();
    doctored["anchors"]
        .as_array_mut()
        .expect("anchors")
        .push(clone);
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("a duplicated (model, tier, backend) must be rejected");
    assert!(error.contains("duplicate memory anchor"), "{error}");
}

#[test]
fn an_anchor_that_drifts_from_its_source_record_is_rejected() {
    let ltx = ltx25_anchor_index();
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    let peak = doctored["anchors"][ltx]["phaseActivePeakBytes"]["conditioning"]
        .as_u64()
        .expect("conditioning peak");
    doctored["anchors"][ltx]["phaseActivePeakBytes"]["conditioning"] = serde_json::json!(peak + 1);
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("a store whose bytes drift from the retained evidence must be rejected");
    assert!(error.contains("disagree"), "{error}");
}

#[test]
fn a_stale_source_digest_or_foreign_source_path_is_rejected() {
    let ltx = ltx25_anchor_index();
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    doctored["anchors"][ltx]["source"]["sha256"] = serde_json::json!("0".repeat(64));
    let error = load_memory_anchors(&doctored.to_string()).expect_err("digest must bind");
    assert!(error.contains("digest mismatch"), "{error}");

    let mut foreign: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    foreign["anchors"][ltx]["source"]["path"] = serde_json::json!("docs/nowhere.json");
    let error = load_memory_anchors(&foreign.to_string()).expect_err("path must be compiled");
    assert!(
        error.contains("not a compiled retained-evidence file"),
        "{error}"
    );
}

// -------------------------------------------------------------------------------------
// Currency: the model's own loader closure (sc-22511, E9).
//
// The DERIVATION of the key — what a pin bump, a sibling edit and an unreached shared-crate
// edit do to it — is asked of the real pinned inference tree by
// `scripts/anchor-loader-closure.test.mjs`, which is where the walk lives. What is asked here
// is the COMPARISON: which anchors a rotated key stales, and which it leaves alone.
// -------------------------------------------------------------------------------------

/// The tooling's currency question, restated locally for the two config-consistency tests
/// below. Deliberately NOT a method on `MemoryAnchor`: the runtime carries no such function
/// (sc-22738), and `scripts/anchor-loader-closure.mjs` owns the real one.
fn anchor_is_current(anchor: &MemoryAnchor, closures: &AnchorLoaderClosures) -> bool {
    closures.digest_for(&anchor.model_id, anchor.backend)
        == Some(anchor.source.loader_closure_digest.as_str())
}

fn packaged_closures() -> AnchorLoaderClosures {
    load_anchor_loader_closures(PACKAGED_ANCHOR_LOADER_CLOSURES)
        .expect("the packaged loader closures parse")
}

/// sc-22724: a catalog alias declares the engine id it resolves to, and the strict parse
/// keeps the field (`deny_unknown_fields` would otherwise demote every anchor to the floor
/// the moment the derivation wrote it). A model that is its own engine carries none.
#[test]
fn a_catalog_alias_declaration_carries_its_engine_id() {
    let closures = packaged_closures();
    let alias = closures
        .models
        .get("z_image_edit:mlx")
        .expect("z_image_edit:mlx is declared");
    assert_eq!(alias.engine_id.as_deref(), Some("z_image_turbo"));
    let own = closures
        .models
        .get("z_image_turbo:mlx")
        .expect("z_image_turbo:mlx is declared");
    assert_eq!(own.engine_id, None);
    // The field round-trips through serde exactly as the derivation wrote it.
    let raw: serde_json::Value =
        serde_json::from_str(PACKAGED_ANCHOR_LOADER_CLOSURES).expect("closures parse");
    assert_eq!(
        raw["models"]["z_image_edit:mlx"]["engineId"],
        "z_image_turbo"
    );
    assert!(raw["models"]["z_image_turbo:mlx"].get("engineId").is_none());
}

#[test]
fn every_packaged_anchor_declares_a_loader_closure() {
    let store = load_memory_anchors(PACKAGED_MEMORY_ANCHORS).expect("packaged store loads");
    let closures = packaged_closures();
    assert!(!store.anchors.is_empty(), "the store must carry anchors");
    for anchor in &store.anchors {
        let key = anchor_loader_closure_key(&anchor.model_id, anchor.backend);
        assert!(
            closures.models.contains_key(&key),
            "anchor {} has no loader-closure declaration for {key}. Add the (model, lane) to \
                 config/anchor-loader-closures.json and derive it: node \
                 scripts/anchor-loader-closure.mjs --repo <inference clone> --write",
            anchor.id
        );
    }
}

/// CURRENCY IS REPORTED, NEVER ASSERTED — and that distinction is the point of E8.
///
/// A packaged anchor whose model's loader source has moved since the measurement is STALE BY
/// DESIGN, and since sc-22738 that is a fact only the probe tooling acts on: the runtime binds
/// and derives from it exactly as from a current one, and the render still runs. Asserting
/// currency here would turn the first pin bump that genuinely touches a loader into a red
/// `cargo test` on a change with nothing wrong in it — which is the
/// pin-bump-forces-re-measurement gate this epic dismantled, rebuilt one level down.
/// `bump-inference` regenerates the closures with `--write` automatically, so that red would
/// land on the bump itself.
///
/// What IS asserted is the loud half above: an anchor for a model nobody DECLARED is a
/// mistake in the store, not a designed state, and it fails.
#[test]
fn packaged_anchor_currency_is_reported_not_gated() {
    let store = load_memory_anchors(PACKAGED_MEMORY_ANCHORS).expect("packaged store loads");
    let closures = packaged_closures();
    let stale: Vec<&str> = store
        .anchors
        .iter()
        .filter(|anchor| !anchor_is_current(anchor, &closures))
        .map(|anchor| anchor.id.as_str())
        .collect();
    if !stale.is_empty() {
        eprintln!(
            "note: {} of {} packaged anchors are not current against their model's declared \
                 loader closure (a re-capture signal for the tooling; the runtime binds them \
                 regardless): {}",
            stale.len(),
            store.anchors.len(),
            stale.join(", ")
        );
    }
}

/// sc-23692: historical attestations remain valid provenance after either a pin-only
/// bump or a shared-loader edit. Test both anchors and bounds without requiring the
/// packaged store to carry any particular current/historical attestation population.
#[test]
fn historical_attestations_load_regardless_of_live_pin_or_closure() {
    let original = load_memory_anchors(PACKAGED_MEMORY_ANCHORS).expect("store loads");
    let mut store = original.clone();
    for source in store
        .anchors
        .iter_mut()
        .map(|anchor| &mut anchor.source)
        .chain(
            store
                .exceeded_bounds
                .iter_mut()
                .map(|bound| &mut bound.source),
        )
    {
        source.currency_attestation = Some(AnchorCurrencyAttestation {
            measured_revision: "1".repeat(40),
            attested_revision: "2".repeat(40),
            attested_at: "2026-09-16".into(),
            story: "sc-23692".into(),
            class: "accounting-only".into(),
            why: "fixture: no loading change".into(),
            witness: "fixture: source review".into(),
        });
    }
    let serialized = serde_json::to_string(&store).unwrap();
    let mut live: serde_json::Value =
        serde_json::from_str(PACKAGED_ANCHOR_LOADER_CLOSURES).unwrap();
    live["inferenceRevision"] = serde_json::json!("3".repeat(40));
    for change_closure in [false, true] {
        if change_closure {
            for entry in live["models"].as_object_mut().unwrap().values_mut() {
                entry["digest"] = serde_json::json!("0".repeat(64));
            }
        }
        let closures = load_anchor_loader_closures(&live.to_string()).unwrap();
        let loaded = load_memory_anchors(&serialized).expect("historical attestations load");
        assert_eq!(loaded, store);
        if change_closure {
            assert!(loaded
                .anchors
                .iter()
                .all(|anchor| !anchor_is_current(anchor, &closures)));
        }
    }
}

/// sc-22738: the attestation shape is validated on a BOUND exactly as on an anchor — a bound
/// is the row kind where a silently-accepted malformed justification would keep a refusal
/// current on nothing.
#[test]
fn a_malformed_currency_attestation_on_a_bound_is_rejected_at_load() {
    let store: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    let bounds = store["exceededBounds"]
        .as_array()
        .expect("the packaged store carries exceeded bounds");
    assert!(!bounds.is_empty());
    let attested = bounds
        .iter()
        .position(|bound| bound["source"]["currencyAttestation"].is_object());
    let index = attested.unwrap_or(0);
    let doctor = |patch: &dyn Fn(&mut serde_json::Value)| {
        let mut doctored = store.clone();
        let source = &mut doctored["exceededBounds"][index]["source"];
        if attested.is_none() {
            source["currencyAttestation"] = serde_json::json!({
                "measuredRevision": "1".repeat(40),
                "attestedRevision": "2".repeat(40),
                "attestedAt": "2026-09-07",
                "story": "sc-22738",
                "class": "accounting-only",
                "why": "test",
                "witness": "test",
            });
        }
        patch(&mut source["currencyAttestation"]);
        load_memory_anchors(&doctored.to_string()).expect_err("must reject")
    };
    let error = doctor(&|a| a["attestedRevision"] = serde_json::json!("abc"));
    assert!(error.contains("exceeded bound"), "{error}");
    assert!(error.contains("not a 40-hex revision"), "{error}");
    let error = doctor(&|a| a["attestedRevision"] = a["measuredRevision"].clone());
    assert!(
        error.contains("attests its own measurement revision"),
        "{error}"
    );
    for field in ["why", "witness", "class", "story", "attestedAt"] {
        let error = doctor(&|a| a[field] = serde_json::json!("   "));
        assert!(error.contains(&format!("states no {field}")), "{error}");
    }
    assert!(load_memory_anchors(&store.to_string()).is_ok());
}

/// The attestation's shape is validated at load like the digest's: a revision that is not a
/// revision, an attestation of its own measurement, or a missing justification is a load
/// failure, never a silently-current anchor.
#[test]
fn a_malformed_currency_attestation_is_rejected_at_load() {
    let store: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    let index = store["anchors"]
        .as_array()
        .unwrap()
        .iter()
        .position(|anchor| anchor["source"]["currencyAttestation"].is_object())
        .expect("the packaged store carries an attested anchor");
    let doctor = |patch: &dyn Fn(&mut serde_json::Value)| {
        let mut doctored = store.clone();
        patch(&mut doctored["anchors"][index]["source"]["currencyAttestation"]);
        load_memory_anchors(&doctored.to_string()).expect_err("must reject")
    };
    let error = doctor(&|a| a["attestedRevision"] = serde_json::json!("abc"));
    assert!(error.contains("not a 40-hex revision"), "{error}");
    let error = doctor(&|a| a["attestedRevision"] = a["measuredRevision"].clone());
    assert!(
        error.contains("attests its own measurement revision"),
        "{error}"
    );
    for field in ["why", "witness", "class", "story", "attestedAt"] {
        let error = doctor(&|a| a[field] = serde_json::json!("   "));
        assert!(error.contains(&format!("states no {field}")), "{error}");
    }
    let error = doctor(&|a| a["extra"] = serde_json::json!(1));
    assert!(error.contains("unknown field"), "{error}");
    // Well-formed as shipped.
    assert!(load_memory_anchors(&store.to_string()).is_ok());
}

#[test]
fn a_loader_closure_file_that_does_not_parse_or_declare_is_refused() {
    let mut version: serde_json::Value =
        serde_json::from_str(PACKAGED_ANCHOR_LOADER_CLOSURES).expect("closures parse");
    version["digestVersion"] = serde_json::json!("anchor-loader-closure v0");
    let error = load_anchor_loader_closures(&version.to_string())
        .expect_err("a version mismatch must be refused");
    assert!(error.contains("digest version"), "{error}");

    let mut digest: serde_json::Value =
        serde_json::from_str(PACKAGED_ANCHOR_LOADER_CLOSURES).expect("closures parse");
    digest["models"]["ltx_2_5:mlx"]["digest"] = serde_json::json!("not-a-digest");
    let error = load_anchor_loader_closures(&digest.to_string())
        .expect_err("a malformed digest must be refused");
    assert!(error.contains("is not a sha256"), "{error}");

    let mut rooted: serde_json::Value =
        serde_json::from_str(PACKAGED_ANCHOR_LOADER_CLOSURES).expect("closures parse");
    rooted["models"]["ltx_2_5:mlx"]["entryPoints"] =
        serde_json::json!(["crates/media/mlx-gen/mlx-gen-wan/src/model.rs"]);
    let error = load_anchor_loader_closures(&rooted.to_string())
        .expect_err("an entry point outside its own closure must be refused");
    assert!(error.contains("not in its own closure"), "{error}");

    let mut counted: serde_json::Value =
        serde_json::from_str(PACKAGED_ANCHOR_LOADER_CLOSURES).expect("closures parse");
    counted["models"]["ltx_2_5:mlx"]["closureFileCount"] = serde_json::json!(1);
    let error = load_anchor_loader_closures(&counted.to_string())
        .expect_err("a file count that disagrees with the list must be refused");
    assert!(error.contains("declares"), "{error}");
}

#[test]
fn an_anchor_whose_loader_digest_is_not_a_sha256_is_refused_at_load() {
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    doctored["anchors"][0]["source"]["loaderClosureDigest"] = serde_json::json!("nope");
    let error =
        load_memory_anchors(&doctored.to_string()).expect_err("the digest shape is checked");
    assert!(error.contains("loader closure digest"), "{error}");
}

// -------------------------------------------------------------------------------------
// Derivation properties.
// -------------------------------------------------------------------------------------

fn plain_request(width: u32, height: u32, frames: u32) -> AnchorDeriveRequest {
    AnchorDeriveRequest {
        width,
        height,
        frames,
        decode_tiled: false,
        transformer_windowed: false,
        deferred_materialization: false,
    }
}

#[test]
fn derivation_at_the_anchor_geometry_reproduces_active_peaks() {
    let anchor = store()
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q8",
            Ltx25TransformerVariant::Dev,
            Ltx25Decoder::DiffVae,
        )
        .expect("q8 dev/diffvae anchor");
    let derived = anchor
        .derive_video_phase_peaks(plain_request(
            anchor.geometry.width,
            anchor.geometry.height,
            anchor.geometry.frames,
        ))
        .expect("derivable at the anchor's own geometry");
    let expect =
        |measured: u64| widened_by(i128::from(measured), MLX_ACTIVE_RECAPTURE_SPREAD).unwrap();
    assert_eq!(
        derived.conditioning,
        expect(anchor.phase_active_peak_bytes.conditioning)
    );
    assert_eq!(
        derived.denoise,
        expect(anchor.phase_active_peak_bytes.denoise)
    );
    assert_eq!(
        derived.decode,
        expect(anchor.phase_active_peak_bytes.decode)
    );
}

#[test]
fn derived_peaks_grow_with_frames_and_area_in_the_plain_regime() {
    let anchor = store()
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q4",
            Ltx25TransformerVariant::Dev,
            Ltx25Decoder::DiffVae,
        )
        .expect("q4 dev/diffvae anchor");
    let small = anchor
        .derive_video_phase_peaks(plain_request(768, 512, 89))
        .expect("small derivable");
    let longer = anchor
        .derive_video_phase_peaks(plain_request(768, 512, 145))
        .expect("longer derivable");
    let larger = anchor
        .derive_video_phase_peaks(plain_request(1280, 704, 89))
        .expect("larger derivable");
    assert!(longer.peak_bytes() > small.peak_bytes());
    assert!(larger.peak_bytes() > small.peak_bytes());
}

/// E3: coefficient uncertainty is priced INSIDE the coefficients, not inside the allocator
/// margin. Every per-unit constant must be at or above the highest slope the retained corpus
/// measures within one identity cell and regime — a coefficient sitting mid-spread would cross
/// below the measured trend at large geometry, where nothing else would catch it.
///
/// The corpus validation test cannot ask this question: its records all sit near the anchors,
/// so a mid-spread coefficient still brackets them.
#[test]
fn every_per_unit_coefficient_bounds_the_highest_measured_within_cell_slope() {
    let corpus = retained_corpus();
    let mut cond_slopes: Vec<f64> = Vec::new();
    let mut denoise_slopes: Vec<f64> = Vec::new();
    let mut decode_slopes: Vec<f64> = Vec::new();
    for (index, left) in corpus.iter().enumerate() {
        for right in &corpus[index + 1..] {
            // Only a pair that differs in GEOMETRY ALONE measures a slope.
            if left.tier != right.tier
                || left.transformer_variant != right.transformer_variant
                || left.decoder != right.decoder
                || left.decode_tiled != right.decode_tiled
                || left.transformer_windowed != right.transformer_windowed
                || left.deferred != right.deferred
            {
                continue;
            }
            let left_tokens = latent_tokens(left.width, left.height, left.frames);
            let right_tokens = latent_tokens(right.width, right.height, right.frames);
            if left_tokens != right_tokens {
                let delta = (right_tokens - left_tokens) as f64;
                // Both intercept-priced phases; a bounded phase has no anchor slope to fit.
                if !left.deferred {
                    cond_slopes.push(
                        (right.conditioning_active as f64 - left.conditioning_active as f64)
                            / delta,
                    );
                }
                denoise_slopes
                    .push((right.denoise_active as f64 - left.denoise_active as f64) / delta);
            }
            let left_voxels = voxels(left.width, left.height, left.frames);
            let right_voxels = voxels(right.width, right.height, right.frames);
            if left_voxels != right_voxels && !left.decode_tiled {
                decode_slopes.push(
                    (right.decode_active as f64 - left.decode_active as f64)
                        / (right_voxels - left_voxels) as f64,
                );
            }
        }
    }
    // Shape: the corpus must actually measure each slope, or this test asks nothing.
    assert!(!cond_slopes.is_empty(), "no measured conditioning slope");
    assert!(!denoise_slopes.is_empty(), "no measured denoise slope");
    assert!(!decode_slopes.is_empty(), "no measured decode slope");

    let highest = |slopes: &[f64]| slopes.iter().cloned().fold(f64::MIN, f64::max);
    for (name, coefficient, slopes) in [
        ("conditioning", COND_PER_TOKEN_BYTES as f64, &cond_slopes),
        ("denoise", DENOISE_PER_TOKEN_BYTES as f64, &denoise_slopes),
        ("decode", DECODE_PER_VOXEL_BYTES as f64, &decode_slopes),
    ] {
        let observed = highest(slopes);
        assert!(
            coefficient >= observed,
            "{name} coefficient {coefficient} sits below the highest measured within-cell \
                 slope {observed}: coefficient uncertainty must be priced in the coefficient, not \
                 in the allocator margin"
        );
    }
}

/// The frames term is affine in `frames`, and `frames == 0` must not underflow the latent
/// frame count on its way there.
#[test]
fn latent_tokens_does_not_underflow_at_zero_frames() {
    assert_eq!(latent_tokens(640, 640, 0), latent_tokens(640, 640, 1));
}

/// An extrapolation driven non-positive has left the law's domain. Clamping it to zero would
/// mint a 0-byte peak that admits ANYTHING, so widening refuses instead and the caller keeps
/// its floor.
///
/// No packaged anchor can currently be driven negative by geometry alone — the smallest
/// derivable request still leaves every phase positive — so this asks the question of
/// `widened` directly rather than pretending a reachable geometry exercises it. It is the
/// guard that keeps a future anchor with a smaller intercept, or a larger coefficient, from
/// silently minting a zero.
#[test]
fn a_non_positive_phase_estimate_is_refused_rather_than_clamped_to_zero() {
    assert_eq!(widened(-1), None);
    assert_eq!(widened(0), None);
    assert_eq!(widened(1_000_000), Some(1_170_000));
}

#[test]
fn degenerate_geometry_is_not_derivable() {
    let anchor = store()
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q4",
            Ltx25TransformerVariant::Dev,
            Ltx25Decoder::DiffVae,
        )
        .expect("q4 dev/diffvae anchor");
    assert!(anchor
        .derive_video_phase_peaks(plain_request(0, 512, 89))
        .is_none());
    assert!(anchor
        .derive_video_phase_peaks(plain_request(768, 0, 89))
        .is_none());
    assert!(anchor
        .derive_video_phase_peaks(plain_request(768, 512, 0))
        .is_none());
}

// -------------------------------------------------------------------------------------
// AC 2: the derivation reproduces every retained LTX-2.5 MLX measured peak within the
// per-term justified margins (epic acceptance test 2, v1).
// -------------------------------------------------------------------------------------

#[test]
fn derivation_brackets_every_retained_corpus_peak() {
    let corpus = retained_corpus();
    // Shape, not a frozen count: every anchored tier must be exercised by the corpus, and
    // both bounded regimes must appear, or this test silently stops asking its question.
    for tier in ["q4", "q8", "bf16"] {
        assert!(
            corpus.iter().any(|record| record.tier == tier),
            "the retained corpus must cover tier {tier}"
        );
    }
    assert!(corpus.iter().any(|record| record.decode_tiled));
    assert!(corpus.iter().any(|record| record.transformer_windowed));
    assert!(corpus.iter().any(|record| record.deferred));

    for record in &corpus {
        let anchor = store()
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                &record.tier,
                record.transformer_variant,
                record.decoder,
            )
            .unwrap_or_else(|| {
                panic!(
                    "cell ({}, {}, {}) carries an anchor",
                    record.tier,
                    transformer_variant_key(record.transformer_variant),
                    decoder_key(record.decoder)
                )
            });
        let derived = anchor
            .derive_video_phase_peaks(AnchorDeriveRequest {
                width: record.width,
                height: record.height,
                frames: record.frames,
                decode_tiled: record.decode_tiled,
                transformer_windowed: record.transformer_windowed,
                deferred_materialization: record.deferred,
            })
            .unwrap_or_else(|| panic!("record {} is derivable", record.id));

        // Per-phase safety: the derived active phase covers the measured active peak.
        for (phase, derived_bytes, measured_bytes) in [
            (
                "conditioning",
                derived.conditioning,
                record.conditioning_active,
            ),
            ("denoise", derived.denoise, record.denoise_active),
            ("decode", derived.decode, record.decode_active),
        ] {
            assert!(
                derived_bytes >= measured_bytes,
                "record {} phase {phase}: derived {derived_bytes} under measured \
                     {measured_bytes}",
                record.id
            );
        }

        // Overall bracket: at or above the measured active peak, and within the
        // validation tightness budget so the margins stay falsifiable.
        let upper = derived.peak_bytes();
        assert!(
            upper
                >= record
                    .conditioning_active
                    .max(record.denoise_active)
                    .max(record.decode_active),
            "record {}: derived admission bound {upper} under the measured active \
                 peak {}",
            record.id,
            record
                .conditioning_active
                .max(record.denoise_active)
                .max(record.decode_active)
        );
        let tight_cap = ((record
            .conditioning_active
            .max(record.denoise_active)
            .max(record.decode_active) as f64)
            * (1.0 + ANCHOR_VALIDATION_TIGHTNESS_BUDGET))
            .ceil() as u64;
        assert!(
            upper <= tight_cap,
            "record {}: derived admission bound {upper} exceeds the tightness cap \
                 {tight_cap} over envelope {}",
            record.id,
            record
                .conditioning_active
                .max(record.denoise_active)
                .max(record.decode_active)
        );
    }

    // The tiled decode estimate must GROW with output voxels. The corpus has both of its
    // tiled captures at the same 130.7M output voxels, so this is the assertion that keeps a
    // geometry-blind flat constant from coming back: a flat bound would silently
    // under-admit a large tiled render whose output clip alone runs to gigabytes.
    let tiled_at = |voxels: i128| decode_tiled_bound_bytes(voxels);
    assert!(
        tiled_at(931_000_000) > tiled_at(130_662_400),
        "the tiled decode bound must grow with output voxels"
    );
    assert!(
        tiled_at(931_000_000) - tiled_at(130_662_400)
            >= DECODE_TILED_PER_VOXEL_BYTES * (931_000_000 - 130_662_400),
        "the tiled decode bound must grow at least at the output-clip rate"
    );

    // Corpus-support shape (informative, never a hard fail on the frozen corpus): the frames
    // and token terms are only validated where a cell carries a record that DIFFERS from its
    // anchor in latent token count. Where it does not — q4 today — the derivation's
    // extrapolation for that tier is unexercised, and this annotation is what makes the gap
    // visible instead of implied.
    for tier in ["q4", "q8", "bf16"] {
        let independent = corpus.iter().any(|record| {
            let Some(anchor) = store().anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                &record.tier,
                record.transformer_variant,
                record.decoder,
            ) else {
                return false;
            };
            record.tier == tier
                && record.id != anchor.source.record_id
                && latent_tokens(record.width, record.height, record.frames)
                    != latent_tokens(
                        anchor.geometry.width,
                        anchor.geometry.height,
                        anchor.geometry.frames,
                    )
        });
        if !independent {
            eprintln!(
                "corpus support gap: tier {tier} has no retained record that differs from its \
                     anchor in latent token count, so its token/frames extrapolation is validated \
                     by zero independent cells"
            );
        }
    }
}

// -------------------------------------------------------------------------------------
// Image derivation law (sc-22663, epic 22657 E3) — fixtures.
// -------------------------------------------------------------------------------------

const KREA_CANDLE_CORPUS_PATH: &str = "docs/generated/krea-candle-five-rung-sc-11045.json";

/// The sc-15859 Z-Image-Turbo q4 candle anchor capture: `staged_residency` at 1024x1024,
/// conditioning 3.10 GB / denoise 8.05 GB / decode 11.74 GB device peak deltas. Not a
/// compiled-in store source (the store is regenerated at epic end), so the test reads the
/// retained file at run time — deliberately not `include_str!`, which would make a test-only
/// fixture a production embed every Docker builder context must copy — and builds the anchor
/// row from the record.
const Z_IMAGE_Q4_CANDLE_ANCHOR_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/calibration/sc-15859/z-image-turbo-q4-candle-anchor.json"
);

fn z_image_q4_candle_anchor_raw() -> String {
    std::fs::read_to_string(Z_IMAGE_Q4_CANDLE_ANCHOR_PATH)
        .unwrap_or_else(|error| panic!("{Z_IMAGE_Q4_CANDLE_ANCHOR_PATH}: {error}"))
}

/// Z-Image-Turbo q4 component bytes on the candle lane: the `SceneWorks/z-image-turbo-mlx`
/// q4 tier the retained record's `artifact` names (revision bb2bc989), whose subdirectories
/// measure text_encoder (Qwen3) 2.26 GB, transformer 3.47 GB, vae 0.16 GB on disk. The
/// record carries no per-component inventory, so the figures are bound to the tier's
/// packaged `estimatedSizeBytes` instead:
/// `the_fixture_component_bytes_sum_to_their_packaged_tier_sizes` recomputes the sum against
/// the manifest within 2%.
const Z_IMAGE_Q4_COMPONENTS: ComponentBytes = ComponentBytes {
    conditioning: 2_260_000_000,
    transformer: 3_470_000_000,
    decoder: 160_000_000,
};

/// Z-Image architecture facts (`candle_transformers::models::z_image::transformer::Config::
/// z_image_turbo`): 30 heads of 128, 30 blocks, patch 2, 16 latent channels, x8 VAE, bf16
/// activations.
const Z_IMAGE_FACTS: ArchitectureFacts = ArchitectureFacts {
    attention_heads: Some(30),
    head_dim: Some(128),
    transformer_blocks: Some(30),
    patch_size: Some(2),
    latent_channels: Some(16),
    vae_spatial_scale: Some(8),
    vae_temporal_scale: Some(1),
    activation_dtype_width: Some(2),
};

/// Krea 2 Turbo architecture facts (`candle-gen-krea/src/config.rs` at the pinned inference
/// revision): 48 heads of 128, 28 single-stream blocks, patch 2 over the 16-channel x8
/// Qwen-Image VAE, bf16 activations.
const KREA_FACTS: ArchitectureFacts = ArchitectureFacts {
    attention_heads: Some(48),
    head_dim: Some(128),
    transformer_blocks: Some(28),
    patch_size: Some(2),
    latent_channels: Some(16),
    vae_spatial_scale: Some(8),
    vae_temporal_scale: Some(1),
    activation_dtype_width: Some(2),
};

/// Krea 2 Turbo q4 component bytes on the candle lane, read off the retained five-rung
/// corpus itself: the resident-minus-staged DENOISE delta is the text encoder that staging
/// drops (2,818,572,288 bytes; the decode delta agrees byte for byte), the resident-minus-
/// staged CONDITIONING delta is the DiT plus VAE that staging has not yet loaded
/// (8,891,924,480), split at the Qwen-Image VAE's bf16 safetensors size (~254 MB).
const KREA_Q4_COMPONENTS: ComponentBytes = ComponentBytes {
    conditioning: 2_818_572_288,
    transformer: 8_638_117_888,
    decoder: 253_806_592,
};

/// The Z-Image / Krea candle fully-engaged rung parameters the providers publish
/// (`bounded_decode` 512/128, `bounded_attention` 64 Mi scores for Z-Image, one-block
/// transformer window).
const FULLY_ENGAGED: RequestRegime = RequestRegime {
    staged: true,
    decode_tile: Some(DecodeTile {
        edge: 512,
        overlap: 128,
    }),
    attention_chunk_scores: Some(64 * 1024 * 1024),
    transformer_window: Some(1),
};

fn at(width: u32, height: u32, regime: RequestRegime) -> ImageDeriveRequest {
    ImageDeriveRequest::new(width, height, regime)
}

/// Build the anchor row for the single record of a memory-v5 candle capture file, exactly as
/// the extractor would (identity, geometry, regime, the three device peak deltas).
fn candle_record_anchor(raw: &str) -> MemoryAnchor {
    let source: serde_json::Value = serde_json::from_str(raw).expect("capture parses");
    let record = &source["records"][0];
    let target = &record["target"];
    let measured: BTreeMap<&str, u64> = record["diagnostics"]["measurements"]
        .as_array()
        .expect("measurements")
        .iter()
        .filter_map(|entry| Some((entry["name"].as_str()?, entry["value"].as_u64()?)))
        .collect();
    let engaged: Vec<&str> = record["strategy"]["engagedRungs"]
        .as_array()
        .expect("engaged rungs")
        .iter()
        .filter_map(|rung| rung.as_str())
        .collect();
    let str_at = |value: &serde_json::Value, key: &str| {
        value[key]
            .as_str()
            .unwrap_or_else(|| panic!("{key}"))
            .to_owned()
    };
    MemoryAnchor {
        id: format!("{}:{}", str_at(target, "modelId"), str_at(record, "id")),
        model_id: str_at(target, "modelId"),
        model_family: String::new(),
        route: str_at(target, "provider"),
        provider: str_at(target, "provider"),
        backend: match record["backend"].as_str() {
            Some("candle") => AnchorBackend::Candle,
            _ => AnchorBackend::Mlx,
        },
        tier: str_at(target, "tier"),
        transformer_variant: None,
        decoder: None,
        mode: str_at(target, "mode"),
        overlay: None,
        reference_count: 0,
        load_shape: match record["loadShape"].as_str() {
            Some("eager_materialization") => AnchorLoadShape::EagerMaterialization,
            _ => AnchorLoadShape::DeferredMaterialization,
        },
        measured_regime: AnchorMeasuredRegime {
            decode_tiled: engaged.contains(&"bounded_decode"),
            transformer_windowed: engaged.contains(&"bounded_transformer_residency"),
            staged: engaged.contains(&"staged_residency"),
            attention_chunked: engaged.contains(&"bounded_attention"),
        },
        source: AnchorSource {
            path: String::new(),
            sha256: String::new(),
            record_id: str_at(record, "id"),
            calibration_fingerprint: str_at(record, "calibrationFingerprint"),
            loader_closure_digest: "0".repeat(64),
            currency_attestation: None,
        },
        geometry: AnchorGeometry {
            width: target["geometry"]["width"].as_u64().expect("width") as u32,
            height: target["geometry"]["height"].as_u64().expect("height") as u32,
            frames: target["geometry"]["frames"].as_u64().expect("frames") as u32,
            fps: None,
        },
        phase_active_peak_bytes: AnchorPhaseBytes {
            conditioning: measured["conditioningDevicePeakDelta"],
            denoise: measured["denoiseDevicePeakDelta"],
            decode: measured["decodeDevicePeakDelta"],
        },
        phase_allocator_envelope_bytes: None,
        overall_allocator_envelope_bytes: measured["overallDevicePeakDelta"],
        underived_reason: None,
        component_bytes: None,
        // The retained corpus this fixture replays predates the structural-exemption fact and
        // carries no structurally-exempt cell, so every replayed row keeps the staged-only law.
        staged_residency_structurally_not_applicable: false,
    }
}

fn z_image_q4_anchor() -> MemoryAnchor {
    let anchor = candle_record_anchor(&z_image_q4_candle_anchor_raw());
    // Shape of the fixture the story cites: the shallow staged composition at 1024x1024.
    assert_eq!(anchor.model_id, "z_image_turbo");
    assert_eq!(anchor.tier, "q4");
    assert_eq!(anchor.backend, AnchorBackend::Candle);
    assert_eq!(
        (anchor.geometry.width, anchor.geometry.height),
        (1024, 1024)
    );
    assert_eq!(
        anchor.measured_regime,
        AnchorMeasuredRegime {
            decode_tiled: false,
            transformer_windowed: false,
            staged: true,
            attention_chunked: false,
        }
    );
    assert_eq!(anchor.phase_active_peak_bytes.conditioning, 3_097_493_504);
    assert_eq!(anchor.phase_active_peak_bytes.denoise, 8_050_966_528);
    assert_eq!(anchor.phase_active_peak_bytes.decode, 11_741_954_048);
    anchor
}

fn z_image_at(request: &ImageDeriveRequest) -> AnchorDerivedPhases {
    z_image_q4_anchor()
        .derive_phase_peaks(request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
        .unwrap_or_else(|| panic!("{request:?} must be derivable"))
}

/// One retained candle capture: the rung composition it executed, its parameters, and its
/// three measured phase peaks.
struct CandleCorpusRecord {
    id: String,
    engaged: Vec<String>,
    regime: RequestRegime,
    width: u32,
    height: u32,
    conditioning: u64,
    denoise: u64,
    decode: u64,
}

impl CandleCorpusRecord {
    fn peak(&self) -> u64 {
        self.conditioning.max(self.denoise).max(self.decode)
    }
}

fn krea_candle_corpus() -> Vec<CandleCorpusRecord> {
    let raw = PACKAGED_MEMORY_ANCHOR_SOURCES
        .iter()
        .find(|(path, _)| *path == KREA_CANDLE_CORPUS_PATH)
        .map(|(_, raw)| *raw)
        .expect("the Krea candle retained corpus is compiled in");
    let source: serde_json::Value =
        serde_json::from_str(raw).expect("retained candle corpus parses");
    source["records"]
        .as_array()
        .expect("candle corpus records")
        .iter()
        .filter(|record| {
            record["target"]["modelId"].as_str() == Some("krea_2_turbo")
                && record["backend"].as_str() == Some("candle")
        })
        .map(|record| {
            let geometry = &record["target"]["geometry"];
            let measured: BTreeMap<&str, u64> = record["diagnostics"]["measurements"]
                .as_array()
                .expect("measurements")
                .iter()
                .filter_map(|entry| Some((entry["name"].as_str()?, entry["value"].as_u64()?)))
                .collect();
            let engaged: Vec<String> = record["strategy"]["engagedRungs"]
                .as_array()
                .expect("engaged rungs")
                .iter()
                .filter_map(|rung| rung.as_str().map(str::to_owned))
                .collect();
            let parameters = &record["strategy"]["parameters"];
            let has = |rung: &str| engaged.iter().any(|engaged| engaged == rung);
            let regime = RequestRegime {
                staged: has("staged_residency"),
                decode_tile: has("bounded_decode").then(|| DecodeTile {
                    edge: parameters["decodeTileEdge"].as_u64().expect("tile edge") as u32,
                    overlap: parameters["decodeOverlap"].as_u64().expect("overlap") as u32,
                }),
                attention_chunk_scores: has("bounded_attention")
                    .then(|| parameters["attentionChunkSize"].as_u64().expect("chunk")),
                transformer_window: has("bounded_transformer_residency").then(|| {
                    parameters["transformerWindowSize"]
                        .as_u64()
                        .expect("window") as u32
                }),
            };
            CandleCorpusRecord {
                id: record["id"].as_str().expect("record id").to_owned(),
                engaged,
                regime,
                width: geometry["width"].as_u64().expect("width") as u32,
                height: geometry["height"].as_u64().expect("height") as u32,
                conditioning: measured["conditioningDevicePeakDelta"],
                denoise: measured["denoiseDevicePeakDelta"],
                decode: measured["decodeDevicePeakDelta"],
            }
        })
        .collect()
}

fn krea_candle_anchor() -> &'static MemoryAnchor {
    store()
        .image_anchor_for("krea_2_turbo", AnchorBackend::Candle, "q4")
        .expect("the Krea candle q4 anchor is packaged")
}

// -------------------------------------------------------------------------------------
// Image derivation law — the story's acceptance criteria.
// -------------------------------------------------------------------------------------

/// AC 1: from the sc-15859 q4 staged anchor and the Z-Image facts, the fully engaged
/// composition (staged + bounded_decode 512/128 + bounded_attention 64 Mi + window 1) at
/// 1024x1024 prices in [4, 6] GB overall; each deeper rung's per-phase peaks are at or below
/// the shallower rung's; and the resident composition prices ABOVE staged instead of `None`.
#[test]
fn z_image_q4_rungs_price_from_the_staged_anchor_in_the_restated_window_and_in_order() {
    const GB: u64 = 1_000_000_000;
    let staged = RequestRegime::staged();
    let tiled = RequestRegime {
        decode_tile: FULLY_ENGAGED.decode_tile,
        ..staged
    };
    let chunked = RequestRegime {
        attention_chunk_scores: FULLY_ENGAGED.attention_chunk_scores,
        ..tiled
    };
    let windowed = RequestRegime {
        transformer_window: FULLY_ENGAGED.transformer_window,
        ..chunked
    };
    assert_eq!(windowed, FULLY_ENGAGED);
    let ladder =
        [staged, tiled, chunked, windowed].map(|regime| z_image_at(&at(1024, 1024, regime)));

    // The staged derivation at the anchor's own geometry and composition IS the anchor.
    let anchor = z_image_q4_anchor();
    assert_eq!(
        ladder[0].conditioning,
        anchor.phase_active_peak_bytes.conditioning
    );
    assert_eq!(ladder[0].denoise, anchor.phase_active_peak_bytes.denoise);
    assert_eq!(ladder[0].decode, anchor.phase_active_peak_bytes.decode);

    // Each deeper rung is at or below the shallower one, phase for phase — and the rung that
    // engages a phase's bound moves that phase strictly, so the ratios are known to bite.
    for pair in ladder.windows(2) {
        let (shallow, deep) = (pair[0], pair[1]);
        assert!(deep.conditioning <= shallow.conditioning);
        assert!(deep.denoise <= shallow.denoise);
        assert!(deep.decode <= shallow.decode);
    }
    assert!(
        ladder[1].decode < ladder[0].decode,
        "the tile ratio must bite"
    );
    assert!(ladder[2].denoise < ladder[1].denoise, "the chunk must bite");
    assert!(
        ladder[3].denoise < ladder[2].denoise,
        "the window must bite"
    );

    let fully_engaged = ladder[3];
    let overall = fully_engaged.peak_bytes();
    // The restated AC-1 window (review of sc-22663): [3, 6] GB overall, strictly below the
    // staged anchor's overall, and never below the components the engaged composition keeps
    // resident in any phase — the text encoder plus its conditioning residue, the windowed
    // DiT share, the VAE. No pad sits anywhere in this arithmetic.
    assert!(
        (3 * GB..=6 * GB).contains(&overall),
        "the fully engaged 1024x1024 composition must price in [3, 6] GB overall, got \
             {overall} ({fully_engaged:?})"
    );
    assert!(overall < ladder[0].peak_bytes());
    let components = Z_IMAGE_Q4_COMPONENTS;
    let cond_residue = anchor.phase_active_peak_bytes.conditioning - components.conditioning;
    assert!(overall >= components.conditioning + cond_residue);
    assert!(overall >= components.transformer.div_ceil(30) + components.decoder);
    assert!(fully_engaged.denoise >= components.transformer.div_ceil(30));
    assert!(fully_engaged.decode >= components.decoder);
    // The arithmetic behind that window, stated so a drift is legible: one resident block
    // (3.47 GB / 30), the non-score denoise residue (8.05 - 3.47 GB minus the 30 x 4608² x
    // 2 B score tensor) plus the 64 Mi x 2 B chunk; and, over the VAE, the decode residue
    // (11.74 - 0.16 GB) split into the unscaled blender floor (1024² x (3 x 2 B + 4 B)) and
    // the remainder scaled by the decode chunk — the larger of the 512² on-device tile and
    // the (512 - 128) x 1024 host-transfer band, i.e. 384 x 1024 = 3/8 of the image.
    assert_eq!(fully_engaged.conditioning, 3_097_493_504);
    assert_eq!(
        fully_engaged.denoise,
        115_666_667 + 3_306_946_688 + 134_217_728
    );
    let blender_floor = 1_048_576 * (3 * 2 + 4);
    assert_eq!(
        fully_engaged.decode,
        160_000_000
            + blender_floor
            + div_ceil_i128(
                (11_581_954_048 - blender_floor as i128) * 393_216,
                1_048_576
            ) as u64
    );
    assert_eq!(fully_engaged.decode, 4_509_786_368);
    assert_eq!(overall, fully_engaged.decode);

    // Resident from the staged anchor: priced, and strictly above staged in every phase.
    let resident = z_image_at(&at(1024, 1024, RequestRegime::resident()));
    assert!(resident.conditioning > ladder[0].conditioning);
    assert!(resident.denoise > ladder[0].denoise);
    assert!(resident.decode > ladder[0].decode);
    // …by exactly the components staging keeps out of each phase.
    let components = Z_IMAGE_Q4_COMPONENTS;
    assert_eq!(
        resident.conditioning - ladder[0].conditioning,
        components.transformer + components.decoder
    );
    assert_eq!(
        resident.denoise - ladder[0].denoise,
        components.conditioning + components.decoder
    );
    assert_eq!(
        resident.decode - ladder[0].decode,
        components.conditioning + components.transformer
    );
}

/// AC 2, candle bullet: a 512x512 request prices below the 1024x1024 anchor — there is no
/// lower clamp. (The MLX bullet is `the_packaged_mlx_anchors_are_outside_the_laws_domain`
/// and `a_resident_mlx_anchor_whose_counters_saw_its_weights_prices_the_ladder_in_order`.)
#[test]
fn a_smaller_request_prices_below_the_anchor_with_no_lower_clamp() {
    let staged = RequestRegime::staged();
    let small = z_image_at(&at(512, 512, staged));
    let mid = z_image_at(&at(768, 768, staged));
    let anchored = z_image_at(&at(1024, 1024, staged));
    let large = z_image_at(&at(1536, 1536, staged));
    assert!(small.peak_bytes() < anchored.peak_bytes());
    assert!(small.denoise < anchored.denoise && small.decode < anchored.decode);
    // No clamp: 512x512 sits strictly below 768x768, which sits strictly below the anchor.
    assert!(small.denoise < mid.denoise && small.decode < mid.decode);
    assert!(mid.denoise < anchored.denoise && mid.decode < anchored.decode);
    assert!(anchored.peak_bytes() < large.peak_bytes());
    // …and never below the components the composition holds resident.
    assert!(small.conditioning >= Z_IMAGE_Q4_COMPONENTS.conditioning);
    assert!(small.denoise >= Z_IMAGE_Q4_COMPONENTS.transformer);
    assert!(small.decode >= Z_IMAGE_Q4_COMPONENTS.decoder);
    // Area, not aspect.
    assert_eq!(
        z_image_at(&at(1344, 768, staged)),
        z_image_at(&at(768, 1344, staged))
    );
}

/// The packaged tier size (`estimatedSizeBytes` of the `variant == tier` download row) of one
/// builtin model — the resident set an eager, whole-model anchor of that tier claims.
fn packaged_tier_bytes(model_id: &str, tier: &str) -> u64 {
    let raw = crate::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| crate::jsonc::strip_jsonc_comments(contents))
        .expect("the builtin models manifest is compiled in");
    let manifest: serde_json::Value = serde_json::from_str(&raw).expect("manifest parses");
    manifest["models"]
        .as_array()
        .expect("models")
        .iter()
        .find(|model| model["id"].as_str() == Some(model_id))
        .unwrap_or_else(|| panic!("{model_id} is a builtin model"))["downloads"]
        .as_array()
        .unwrap_or_else(|| panic!("{model_id} declares downloads"))
        .iter()
        .find(|download| download["variant"].as_str() == Some(tier))
        .and_then(|download| download["estimatedSizeBytes"].as_u64())
        .unwrap_or_else(|| panic!("{model_id} {tier} declares estimatedSizeBytes"))
}

/// AC 2, MLX bullet — what the packaged store can and cannot support. The story asked for
/// the packaged `qwen_image` MLX resident anchor to price rung 4 below rung 2; under the
/// domain guard (review of sc-22663: a residue is never clamped to zero) NO packaged MLX
/// image anchor is derivable with its tier's real component bytes, because every one of them
/// reports a conditioning-phase active peak that is a fraction of the eager resident set it
/// claims — the MLX active counters did not see the weights. The numbers, so the refusal is
/// legible: conditioning active peaks of 1.26 MB (flux2_dev q4 and q8), 43.9 KB (qwen_image
/// bf16/q4/q8) and 2.27 GB (z_image_turbo q4) against packaged resident sets of 33.6 GB,
/// 55.0 GB, 57.7 / 28.4 / 38.6 GB and 5.9 GB. The lane entry point therefore prices nothing
/// from the packaged store; the rung ordering itself is asserted on an in-domain anchor in
/// `a_resident_mlx_anchor_whose_counters_saw_its_weights_prices_the_ladder_in_order`.
///
/// ALLOCATOR BASIS (second review pass, D5): the MLX fit gate admits on the per-phase
/// ALLOCATOR envelope, and one might expect that level — unlike the active counter — to
/// carry the wired weight set. It does not, in the conditioning phase: the allocator
/// conditioning levels are 1.26 MB (flux2_dev q4/q8), 44.1 KB (qwen_image bf16/q4/q8) and
/// 2.27 GB (z_image_turbo q4), so the allocator-basis residues are, in GB
/// (cond / denoise / decode): flux2_dev q4 −33.60 / +7.31 / +55.20; flux2_dev q8 −55.00 /
/// +10.26 / +58.15; qwen_image bf16 −57.73 / +5.83 / +5.47; qwen_image q4 −28.39 / +2.92 /
/// +1.65; qwen_image q8 −38.58 / +2.84 / +1.62; z_image_turbo q4 −3.64 / +6.27 / +8.40.
/// Denoise and decode come back positive on the allocator basis, conditioning never does, so
/// no packaged MLX anchor is in the law's domain on either basis and the law keeps
/// decomposing the active peak — a switch to the allocator basis would move nothing. Both
/// bases are asserted below so a re-captured anchor that reports its weights lands here.
///
/// RE-CAPTURE (sc-22667, epic 22657 D3): the cause was the ADAPTER, not the counters — MLX
/// residency is request-scoped even under eager materialization, and the window opened on a
/// cold first request. `z_image_turbo` q4 was re-captured with the window opened above its
/// materialized resident set and is now IN domain
/// (`the_recaptured_z_image_mlx_anchor_is_inside_the_laws_domain`); the other MLX image
/// anchors of that pre-campaign bundle keep the cold-request shape until their own re-capture
/// and stay in this census.
///
/// CENSUS SCOPE (sc-22738): the cold-request rows are exactly the MLX image anchors still
/// cited from the pre-campaign bundle `docs/generated/memory-calibration-evidence.json`. The
/// sc-22738 campaign re-captured the MLX image catalog under the fixed adapter into
/// `docs/calibration/sc-22738/`, and those rows are NOT this census: some report the resident
/// set in their conditioning level (klein q4: 15.1 GB against a 21.7 GB set) and are in the
/// law's domain, others (sd3_5, mage_flow, chroma) legitimately hold their text encoder
/// staged below it. Selecting "every MLX image anchor but z_image_turbo" was a frozen census
/// of the five bundle rows and red the moment a re-captured anchor landed. The bundle rows
/// that a later campaign re-captures leave this census with their file, and an empty census
/// is the test retired, not a failure.
#[test]
fn the_packaged_mlx_anchors_are_outside_the_laws_domain() {
    const PRE_CAMPAIGN_BUNDLE: &str = "docs/generated/memory-calibration-evidence.json";
    let packaged: Vec<&MemoryAnchor> = store()
        .anchors
        .iter()
        .filter(|anchor| {
            anchor.backend == AnchorBackend::Mlx
                && anchor.geometry.frames == 1
                && anchor.model_id != "z_image_turbo"
                && anchor.source.path == PRE_CAMPAIGN_BUNDLE
        })
        .collect();
    if packaged.is_empty() {
        return;
    }
    for anchor in packaged {
        assert!(
            !anchor.measured_regime.staged,
            "{}: eager resident",
            anchor.id
        );
        let resident_set = packaged_tier_bytes(&anchor.model_id, &anchor.tier);
        // The split does not matter to a resident-regime decomposition (every phase is
        // decomposed against the sum); the sum is the manifest's.
        let components = ComponentBytes {
            conditioning: resident_set,
            transformer: 0,
            decoder: 0,
        };
        assert!(
            anchor.phase_active_peak_bytes.conditioning < resident_set / 2,
            "{}: conditioning active {} is not below the {resident_set} resident set it claims",
            anchor.id,
            anchor.phase_active_peak_bytes.conditioning
        );
        // …and so is the ALLOCATOR level of the same phase (D5): every packaged row carries
        // a per-phase allocator decomposition, and its conditioning level is a fraction of
        // the resident set too, while its denoise and decode levels sit above it.
        let allocators = anchor
            .phase_allocator_envelope_bytes
            .unwrap_or_else(|| panic!("{}: carries a per-phase allocator envelope", anchor.id));
        assert!(
            allocators.conditioning < resident_set / 2,
            "{}: conditioning allocator {} is not below the {resident_set} resident set",
            anchor.id,
            allocators.conditioning
        );
        assert!(
            allocators.denoise > resident_set && allocators.decode > resident_set,
            "{}: the denoise/decode allocator levels {}/{} must exceed the resident set — if \
                 this fails the census in the doc comment above has moved",
            anchor.id,
            allocators.denoise,
            allocators.decode
        );
        // A resident request with every bounded rung engaged (so neither the regime guard
        // nor a staged phase priced at a zero component refuses) is refused by the domain
        // guard alone…
        let request = at(
            anchor.geometry.width,
            anchor.geometry.height,
            RequestRegime {
                staged: false,
                ..FULLY_ENGAGED
            },
        );
        assert!(
            anchor
                .derive_phase_peaks(&request, components, ArchitectureFacts::default())
                .is_none(),
            "{}: an out-of-domain anchor must refuse, not clamp",
            anchor.id
        );
        // …and lifting every phase peak to at least the resident set is exactly what makes
        // it derivable again, so the refusal is the domain guard and nothing else. (The
        // conditioning phase is below on every packaged row; flux2_dev q4 and qwen_image
        // bf16 report a denoise active peak below their resident set as well.)
        let mut lifted = anchor.clone();
        let peaks = &mut lifted.phase_active_peak_bytes;
        peaks.conditioning = peaks.conditioning.max(resident_set);
        peaks.denoise = peaks.denoise.max(resident_set);
        peaks.decode = peaks.decode.max(resident_set);
        assert!(
            lifted
                .derive_phase_peaks(&request, components, ArchitectureFacts::default())
                .is_some(),
            "{}: with every phase peak at or above the resident set the anchor derives",
            anchor.id
        );
    }
}

#[test]
fn cached_conditioning_preserves_other_resident_mlx_phase_residues() {
    let mut anchor = store()
        .image_anchor_for("z_image_turbo", AnchorBackend::Mlx, "q4")
        .unwrap()
        .clone();
    let components = ComponentBytes {
        conditioning: 1,
        transformer: 1,
        decoder: 1,
    };
    let request = AnchorMlxImageDeriveRequest {
        width: 1024,
        height: 1024,
    };
    let complete = anchor.derive_mlx_image_activation_residues(request, components);
    assert!(complete.iter().all(Option::is_some));
    anchor.phase_active_peak_bytes.conditioning = 0;
    let partial = anchor.derive_mlx_image_activation_residues(request, components);
    assert_eq!(partial, [None, complete[1], complete[2]]);
    assert!(
        anchor
            .derive_mlx_image_phase_peaks(request, components)
            .is_none(),
        "whole-pipeline authority remains strict"
    );
    anchor.measured_regime.decode_tiled = true;
    assert_eq!(
        anchor.derive_mlx_image_activation_residues(request, components),
        [None; 3]
    );
    anchor.measured_regime.decode_tiled = false;
    anchor.load_shape = AnchorLoadShape::DeferredMaterialization;
    assert_eq!(
        anchor.derive_mlx_image_activation_residues(request, components),
        [None; 3]
    );
}

/// sc-22667 (epic 22657 D3, E6 "one MLX tier"): the packaged `z_image_turbo` q4 MLX anchor was
/// re-captured on the nax runner with the adapter's phase window opened ABOVE its
/// materialized resident set (`memory-mlx-adapter`, `open_resident_phase_window`), so every
/// phase level now includes the 5.9 GB the tier holds resident and the law decomposes it with
/// the tier's real component split: the resident derivation reproduces the anchor, the staged
/// rung prices below it, and the fully engaged rung 4 prices below the staged rung 2.
///
/// The components are the T2I RESIDENT set read off the q4 snapshot's safetensors headers
/// (`Z_IMAGE_Q4_MLX_RESIDENT_COMPONENTS`): text encoder, transformer, and the VAE DECODER —
/// a text-to-image render never materializes the 67,041,088-byte VAE encoder, and the
/// record's post-request live set (5,833,334,792 bytes, `preRungActiveAfterClear`) sits
/// 7.1 MB above that sum and 60 MB below the whole-VAE sum.
///
/// MUTATION: point the store at the cold-request record this replaced (conditioning active
/// peak 2,266,383,344 against the 5.83 GB resident set — the figures the census above still
/// asserts for the other five MLX anchors) and every `expect` here fails: the domain guard
/// refuses a conditioning residue below zero.
#[test]
fn the_recaptured_z_image_mlx_anchor_is_inside_the_laws_domain() {
    let anchor = store()
        .image_anchor_for("z_image_turbo", AnchorBackend::Mlx, "q4")
        .expect("the z_image_turbo q4 MLX anchor is packaged");
    assert!(!anchor.measured_regime.staged);
    assert_eq!(anchor.load_shape, AnchorLoadShape::EagerMaterialization);
    let components = Z_IMAGE_Q4_MLX_RESIDENT_COMPONENTS;
    let resident_set = components.total();
    let peaks = anchor.phase_active_peak_bytes;
    // The window opened above the resident set: every phase level carries the weights.
    for (phase, peak) in [
        ("conditioning", peaks.conditioning),
        ("denoise", peaks.denoise),
        ("decode", peaks.decode),
    ] {
        assert!(
            peak >= resident_set,
            "{phase}: the re-captured level {peak} must include the {resident_set} resident set"
        );
    }
    let facts = Z_IMAGE_FACTS;
    let derive = |regime| anchor.derive_phase_peaks(&at(768, 768, regime), components, facts);
    let resident = derive(RequestRegime::resident())
        .expect("the re-captured MLX anchor is inside the law's domain");
    assert_eq!(resident.conditioning, peaks.conditioning);
    assert_eq!(resident.denoise, peaks.denoise);
    assert_eq!(resident.decode, peaks.decode);
    let rung_2 = derive(RequestRegime::staged()).expect("the staged rung prices");
    let rung_4 = derive(FULLY_ENGAGED).expect("the fully engaged rung prices");
    assert!(rung_2.peak_bytes() <= resident.peak_bytes());
    assert!(
        rung_4.peak_bytes() < rung_2.peak_bytes(),
        "rung 4 {rung_4:?} must price below rung 2 {rung_2:?}"
    );
    assert!(rung_4.denoise < rung_2.denoise && rung_4.decode < rung_2.decode);
    // The physical-memory gate must use active bytes, not add reclaimable cache.
    let lane = anchor
        .derive_mlx_image_phase_peaks(
            AnchorMlxImageDeriveRequest {
                width: 768,
                height: 768,
            },
            components,
        )
        .expect("the MLX lane entry point prices the re-captured anchor");
    assert_eq!(lane.conditioning, resident.conditioning);
    assert_eq!(lane.denoise, resident.denoise);
    assert_eq!(lane.decode, resident.decode);
    let mut inflated_cache = anchor.clone();
    inflated_cache.phase_allocator_envelope_bytes = Some(AnchorPhaseBytes {
        conditioning: u64::MAX,
        denoise: u64::MAX,
        decode: u64::MAX,
    });
    assert_eq!(
        inflated_cache.derive_mlx_image_phase_peaks(
            AnchorMlxImageDeriveRequest {
                width: 768,
                height: 768
            },
            components,
        ),
        Some(lane),
        "elastic cache is not mandatory residency"
    );
}

/// The Z-Image-Turbo q4 MLX resident set a text-to-image render holds, as tensor bytes summed
/// off the packaged snapshot's safetensors headers (`SceneWorks/z-image-turbo-mlx@bb2bc989`,
/// `q4/`): every `text_encoder` tensor, every `transformer` tensor, and the `decoder.*`
/// tensors of `vae/model.safetensors` (97,583,622 of its 164,624,710 bytes; the other
/// 67,041,088 are the encoder, which a T2I render never materializes).
const Z_IMAGE_Q4_MLX_RESIDENT_COMPONENTS: ComponentBytes = ComponentBytes {
    conditioning: 2_262_920_192,
    transformer: 3_465_730_304,
    decoder: 97_583_622,
};

/// THE GAP IS CLOSED (sc-22667, inference c6d6a4db) — what remains here is a DOMAIN GUARD,
/// not a production residue.
///
/// It used to read: the worker prices the anchor with the provider's asset facts
/// (`video_admission::anchor_component_bytes`), whose `decoder_bytes` was the WHOLE `vae/`
/// directory — encoder included — so the resident set it subtracted (5,893,275,206) exceeded
/// the re-captured conditioning level (5,834,701,816) by 58.6 MB, the law refused, and the
/// production MLX cell stayed on its estimate floor. The pinned provider now splits that
/// directory by key prefix: `decoder_bytes` is the render-resident `decoder.*` half
/// (97,583,622 B, exactly [`Z_IMAGE_Q4_MLX_RESIDENT_COMPONENTS`]) and the `encoder.*` half a
/// `Reference` request opts into is a typed `MemoryComponentKind::ReferenceEncoder` auxiliary
/// carried in `overlay_bytes`. `anchor_component_bytes` hands the law the three BASE legs, and
/// `base_bytes` is exactly `text_encoder + dit + vae` with the auxiliary charged on top by the
/// contract's own `predicted_peak_from_base` — so production now hands this anchor the
/// decoder-only split and it is IN DOMAIN. `the_recaptured_z_image_mlx_anchor_is_inside_the_laws_domain`
/// above walks that path; the worker's
/// `the_packaged_z_image_mlx_cell_prices_rung_four_below_rung_two_from_the_contracts_base_legs`
/// grades the same cell through the production seam.
///
/// What survives is the guard: an OVERSTATED component — the whole directory, which is what a
/// provider that folded the encoder back into `decoder_bytes` would state — must still be
/// REFUSED rather than clamped to zero, because the law will not subtract a resident set the
/// measurement contradicts. If this test starts deriving, the domain guard has been weakened;
/// if the decoder-only split stops deriving, the pin regressed.
#[test]
fn an_overstated_whole_vae_component_is_still_refused_by_the_anchor_domain_guard() {
    let anchor = store()
        .image_anchor_for("z_image_turbo", AnchorBackend::Mlx, "q4")
        .expect("the z_image_turbo q4 MLX anchor is packaged");
    let whole_vae = ComponentBytes {
        decoder: 164_624_710,
        ..Z_IMAGE_Q4_MLX_RESIDENT_COMPONENTS
    };
    assert!(anchor.phase_active_peak_bytes.conditioning < whole_vae.total());
    assert!(anchor
        .derive_phase_peaks(
            &at(768, 768, RequestRegime::resident()),
            whole_vae,
            Z_IMAGE_FACTS
        )
        .is_none());
    assert!(anchor
        .derive_mlx_image_phase_peaks(
            AnchorMlxImageDeriveRequest {
                width: 768,
                height: 768,
            },
            whole_vae,
        )
        .is_none());
}

/// A synthetic eager, resident, unbounded MLX anchor at 1024x1024 whose counters saw its
/// weights: every phase peak is the whole component set plus a stated activation residue,
/// and the allocator level sits a stated envelope above each. Identity fields are borrowed
/// from the packaged flux2_dev q8 row (the lane entry point checks lane, load shape and
/// regime, not the id); the peaks are NOT the record's.
fn in_domain_resident_mlx_anchor(components: ComponentBytes) -> MemoryAnchor {
    let residues = AnchorPhaseBytes {
        conditioning: 500_000_000,
        denoise: 6_000_000_000,
        decode: 12_000_000_000,
    };
    let total = components.total();
    let mut anchor = flux2_mlx_anchor("q8").clone();
    anchor.id = "synthetic:mlx:resident:in-domain".to_owned();
    anchor.underived_reason = None;
    anchor.geometry = AnchorGeometry {
        width: 1024,
        height: 1024,
        frames: 1,
        fps: None,
    };
    anchor.phase_active_peak_bytes = AnchorPhaseBytes {
        conditioning: total + residues.conditioning,
        denoise: total + residues.denoise,
        decode: total + residues.decode,
    };
    anchor.phase_allocator_envelope_bytes = Some(AnchorPhaseBytes {
        conditioning: total + residues.conditioning + 100_000_000,
        denoise: total + residues.denoise + 1_000_000_000,
        decode: total + residues.decode + 2_000_000_000,
    });
    anchor.overall_allocator_envelope_bytes = total + residues.decode + 2_000_000_000;
    anchor
}

/// The rung ordering AC 2 asked of an MLX resident anchor, on an anchor inside the law's
/// domain (the packaged ones are not — see the test above): the windowed rung prices below
/// the staged rung, with every residue it scales asserted positive so the ordering is not
/// vacuous, and the resident derivation at the anchor geometry reproduces the anchor.
#[test]
fn a_resident_mlx_anchor_whose_counters_saw_its_weights_prices_the_ladder_in_order() {
    // Qwen-Image bf16 component bytes: the tier's packaged download split at the
    // Qwen2.5-VL-7B text encoder's bf16 size (16.6 GB) and the VAE's (~254 MB).
    let total = packaged_tier_bytes("qwen_image", "bf16");
    let components = ComponentBytes {
        conditioning: 16_600_000_000,
        transformer: total - 16_600_000_000 - 253_806_592,
        decoder: 253_806_592,
    };
    // Qwen-Image: 24 heads of 128, 60 blocks, patch 2 over the 16-channel x8 VAE, bf16.
    let facts = ArchitectureFacts {
        attention_heads: Some(24),
        transformer_blocks: Some(60),
        ..Z_IMAGE_FACTS
    };
    let anchor = in_domain_resident_mlx_anchor(components);
    assert!(!anchor.measured_regime.staged);
    assert_eq!(anchor.load_shape, AnchorLoadShape::EagerMaterialization);
    for (phase, peak) in [
        ("conditioning", anchor.phase_active_peak_bytes.conditioning),
        ("denoise", anchor.phase_active_peak_bytes.denoise),
        ("decode", anchor.phase_active_peak_bytes.decode),
    ] {
        assert!(
            peak > components.total(),
            "{phase}: the residue the law scales must be positive, peak {peak} against {}",
            components.total()
        );
    }
    let rung_2 = anchor
        .derive_phase_peaks(&at(1024, 1024, RequestRegime::staged()), components, facts)
        .expect("the resident MLX anchor prices the staged rung");
    let rung_4 = anchor
        .derive_phase_peaks(&at(1024, 1024, FULLY_ENGAGED), components, facts)
        .expect("the resident MLX anchor prices the windowed rung");
    assert!(
        rung_4.peak_bytes() < rung_2.peak_bytes(),
        "windowed {rung_4:?} must price below staged {rung_2:?}"
    );
    assert!(rung_4.denoise < rung_2.denoise && rung_4.decode < rung_2.decode);
    // The resident derivation at the anchor geometry IS the anchor: every component re-added
    // into every phase plus the positive residue, byte for byte.
    let resident = anchor
        .derive_phase_peaks(
            &at(1024, 1024, RequestRegime::resident()),
            components,
            facts,
        )
        .expect("resident");
    assert_eq!(
        resident.conditioning,
        anchor.phase_active_peak_bytes.conditioning
    );
    assert_eq!(resident.denoise, anchor.phase_active_peak_bytes.denoise);
    assert_eq!(resident.decode, anchor.phase_active_peak_bytes.decode);
    assert!(rung_2.peak_bytes() <= resident.peak_bytes());
    // And the same anchor with one phase pushed below its resident set is refused whole.
    let mut below = anchor.clone();
    below.phase_active_peak_bytes.denoise = components.total() - 1;
    assert!(below
        .derive_phase_peaks(&at(1024, 1024, FULLY_ENGAGED), components, facts)
        .is_none());
}

/// sc-22665 (E4): the activation half of the law is exactly the whole estimate minus the
/// components the request's regime holds resident in that phase — nothing is re-added, and a
/// request the law refuses hands out no residue either. Asserted on the same in-domain
/// resident MLX anchor the rung-ordering test uses, at three regimes, so the residue's own
/// rung ordering (chunk shrinks denoise, tile shrinks decode) is visible without the
/// components' rung ordering masking it.
#[test]
fn the_activation_residues_are_the_derived_peaks_minus_the_regimes_resident_components() {
    let total = packaged_tier_bytes("qwen_image", "bf16");
    let components = ComponentBytes {
        conditioning: 16_600_000_000,
        transformer: total - 16_600_000_000 - 253_806_592,
        decoder: 253_806_592,
    };
    let facts = ArchitectureFacts {
        attention_heads: Some(24),
        transformer_blocks: Some(60),
        ..Z_IMAGE_FACTS
    };
    let anchor = in_domain_resident_mlx_anchor(components);
    let chunked = RequestRegime {
        staged: true,
        attention_chunk_scores: Some(64 * 1024 * 1024),
        ..RequestRegime::staged()
    };
    let residues = |regime| {
        let request = at(1024, 1024, regime);
        let peaks = anchor
            .derive_phase_peaks(&request, components, facts)
            .expect("the in-domain anchor prices this regime");
        let residues = anchor
            .derive_phase_activation_residues(&request, components, facts)
            .expect("…and hands out the activation half of the same estimate");
        // The difference is a COMPONENT set, never an activation byte: every phase's
        // peak-minus-residue is a sum of the three component figures.
        for (phase, peak, residue) in [
            ("conditioning", peaks.conditioning, residues.conditioning),
            ("denoise", peaks.denoise, residues.denoise),
            ("decode", peaks.decode, residues.decode),
        ] {
            let resident = peak
                .checked_sub(residue)
                .unwrap_or_else(|| panic!("{phase}: the residue cannot exceed the peak"));
            assert!(
                resident <= components.total(),
                "{phase}: {resident} resident bytes exceed the whole component set"
            );
        }
        residues
    };
    let rung_2 = residues(RequestRegime::staged());
    let rung_3 = residues(chunked);
    let rung_4 = residues(FULLY_ENGAGED);
    assert_eq!(
        rung_2.conditioning, rung_4.conditioning,
        "no rung below bounds the conditioning phase"
    );
    assert!(
        rung_3.denoise < rung_2.denoise,
        "the attention chunk must shrink the denoise residue: {rung_3:?} vs {rung_2:?}"
    );
    assert_eq!(rung_3.decode, rung_2.decode, "rung 3 does not tile decode");
    assert!(
        rung_4.decode < rung_3.decode,
        "the decode tile must shrink the decode residue: {rung_4:?} vs {rung_3:?}"
    );
    assert_eq!(rung_4.denoise, rung_3.denoise);
    // A refused request hands out no residue either.
    let mut below = anchor.clone();
    below.phase_active_peak_bytes.denoise = components.total() - 1;
    assert!(below
        .derive_phase_activation_residues(&at(1024, 1024, FULLY_ENGAGED), components, facts)
        .is_none());
}

/// sc-22667 (epic 22657 feature-end round): the window share both image lanes state is ONE
/// function. `transformer x min(window, blocks) / blocks` rounded up when both are known; the
/// WHOLE transformer when the block count is unknown (the default facts at this pin) or no
/// window is engaged — never zero.
///
/// MUTATION: returning `0` (or `transformer.saturating_sub(..)`) for the unknown-blocks arm
/// reds the second assertion; dropping the `.min(blocks)` clamp reds the fourth; and the law
/// itself is graded through this helper by `z_image_q4_rungs_price_from_the_staged_anchor_*`,
/// so a divergence between the two reds there.
#[test]
fn the_windowed_transformer_share_is_the_laws_and_errs_large_without_a_block_count() {
    assert_eq!(
        windowed_transformer_bytes(3_470_000_000, Some(1), Some(30)),
        115_666_667,
        "one of thirty blocks, rounded up"
    );
    assert_eq!(
        windowed_transformer_bytes(3_470_000_000, Some(1), None),
        3_470_000_000,
        "a window with no block count keeps the whole transformer resident"
    );
    assert_eq!(
        windowed_transformer_bytes(3_470_000_000, None, Some(30)),
        3_470_000_000,
        "no window engaged: whole transformer"
    );
    assert_eq!(
        windowed_transformer_bytes(3_470_000_000, Some(64), Some(30)),
        3_470_000_000,
        "a window wider than the model is the whole model, never more"
    );
    assert_eq!(
        windowed_transformer_bytes(3_470_000_000, Some(0), Some(30)),
        3_470_000_000
    );
}

/// sc-22738: the shared floor weights arithmetic, one assertion per branch the worker's
/// `mlx_fit_gate::floor_weights_bytes` used to carry privately. Facts: conditioning 26,
/// transformer 20 (of which 6 evictable), decoder 4, base 50, one unbounded auxiliary of 3.
#[test]
fn the_shared_floor_weights_term_prices_every_composition_the_worker_prices() {
    let facts = FloorWeightsFacts {
        conditioning_bytes: 26,
        base_bytes: 50,
        transformer_bytes: 20,
        intra_transformer_evicted_bytes: 6,
        auxiliary_resident_bytes: 3,
    };
    let composition = |staged: bool, bounded_transformer: bool, resident_window_bytes: u64| {
        FloorWeightsComposition {
            staged,
            bounded_transformer,
            resident_window_bytes,
        }
    };
    // Resident: conditioning + heavy (24) + auxiliary.
    assert_eq!(floor_weights_bytes(facts, composition(false, false, 0)), 53);
    // Staged: max(conditioning 26, heavy 24 − 6 evicted clamped at the transformer 20 = 20)
    // = 26, plus the auxiliary.
    assert_eq!(floor_weights_bytes(facts, composition(true, false, 0)), 29);
    // Staged with a drop smaller than the clamp room: heavy 24 − 2 = 22 ≥ 20.
    let small_drop = FloorWeightsFacts {
        intra_transformer_evicted_bytes: 2,
        ..facts
    };
    assert_eq!(
        floor_weights_bytes(small_drop, composition(true, false, 0)),
        29
    );
    let tall_heavy = FloorWeightsFacts {
        base_bytes: 60,
        ..facts
    };
    // heavy 34 − 6 = 28 > conditioning 26.
    assert_eq!(
        floor_weights_bytes(tall_heavy, composition(true, false, 0)),
        31
    );
    // Rung 4 alone: the transformer leaves (24 − 20 = 4), the window comes back clamped at the
    // load-exact transformer; the eviction is NOT applied on top.
    assert_eq!(floor_weights_bytes(facts, composition(false, true, 0)), 33);
    assert_eq!(floor_weights_bytes(facts, composition(false, true, 5)), 38);
    assert_eq!(floor_weights_bytes(facts, composition(false, true, 99)), 53);
    // Staged + rung 4: exclusive — the rung-4 branch, then the staged max. heavy = 4 + 5 = 9;
    // max(26, 9) = 26; + 3.
    assert_eq!(floor_weights_bytes(facts, composition(true, true, 5)), 29);
    // The weights-free contract prices zero everywhere.
    assert_eq!(
        floor_weights_bytes(FloorWeightsFacts::default(), composition(true, true, 7)),
        0
    );
}

/// sc-22738: the byte forms of the two shared allowances are the worker's own conversion of
/// the GiB declarations, and nothing else — the adapter's admission budget and floor read
/// these, so a drift here is a drift in what ships.
#[test]
fn the_shared_allowances_convert_to_the_bytes_the_worker_presents() {
    assert_eq!(
        gib_to_bytes(LEGACY_UNIFIED_FALLBACK_RESERVE_GB),
        2_147_483_648
    );
    assert_eq!(gib_to_bytes(MLX_GENERIC_HEADROOM_GB), 19_327_352_832);
    assert_eq!(gib_to_bytes(0.0), 0);
    assert_eq!(gib_to_bytes(-1.0), 0);
    assert_eq!(gib_to_bytes(1.5), 1_610_612_736);
}

/// The Bernini stop, as a fixture: `bernini:bf16:mlx` text_to_video at 848x480x49, stopped at
/// 97,147,294,328 bytes of physical footprint on a 128 GiB host (sc-22738, measured 2026-09-06).
fn bernini_bound() -> ExceededBound {
    ExceededBound {
        id: "exceeded:bernini:mlx:bf16:base:base:fp:exc-0".to_owned(),
        model_id: "bernini".to_owned(),
        model_family: "bernini".to_owned(),
        route: "bernini".to_owned(),
        provider: "bernini".to_owned(),
        backend: AnchorBackend::Mlx,
        tier: "bf16".to_owned(),
        transformer_variant: None,
        decoder: None,
        mode: "text_to_video".to_owned(),
        overlay: None,
        reference_count: 0,
        load_shape: AnchorLoadShape::EagerMaterialization,
        geometry: AnchorGeometry {
            width: 848,
            height: 480,
            frames: 49,
            fps: None,
        },
        observed_footprint_bytes: 97_147_294_328,
        ceiling_bytes: 94_822_600_832,
        host_memory_bytes: 137_438_953_472,
        reason: "physical_footprint_at_or_above_94822600832:observed_97147294328".to_owned(),
        source: AnchorSource {
            path: "docs/calibration/sc-22738/bernini-bf16-mlx-exceeded-evidence.json".to_owned(),
            sha256: "0".repeat(64),
            record_id: "exc-0".to_owned(),
            calibration_fingerprint: "fp".to_owned(),
            loader_closure_digest: "b".repeat(64),
            currency_attestation: None,
        },
    }
}

fn bound_store(bounds: Vec<ExceededBound>) -> MemoryAnchorStore {
    MemoryAnchorStore {
        schema_version: MEMORY_ANCHOR_SCHEMA_VERSION,
        anchors: Vec::new(),
        analytic_only: Vec::new(),
        component_deltas: Vec::new(),
        exceeded_bounds: bounds,
        observed_operating_envelopes: Vec::new(),
    }
}

fn bernini_query(width: u32, height: u32, frames: u32) -> ExceededBoundQuery<'static> {
    ExceededBoundQuery {
        model_id: "bernini",
        model_family: Some("bernini"),
        route: Some("bernini"),
        provider: "bernini",
        backend: AnchorBackend::Mlx,
        tier: "bf16",
        transformer_variant: None,
        decoder: None,
        mode: "text_to_video",
        overlay: None,
        reference_count: 0,
        width,
        height,
        frames,
    }
}

/// sc-22738: a measured lower bound binds at and above its own geometry, on the identity it was
/// measured on — and regardless of whether its currency key still matches the packaged loader
/// closures (see `a_bound_whose_loader_closure_moved_still_binds`).
#[test]
fn a_measured_lower_bound_binds_at_and_above_its_geometry_on_its_own_identity() {
    let store = bound_store(vec![bernini_bound()]);
    let bind = |query| {
        store
            .binding_exceeded_bound(query)
            .map(|bound| bound.observed_footprint_bytes)
    };
    // The measured point itself, and every request at or above it on all three axes.
    assert_eq!(bind(bernini_query(848, 480, 49)), Some(97_147_294_328));
    assert_eq!(bind(bernini_query(848, 480, 97)), Some(97_147_294_328));
    assert_eq!(bind(bernini_query(1280, 720, 49)), Some(97_147_294_328));
    // SMALLER on any single axis is outside the claim: the inequality was measured at one
    // geometry and monotonicity only runs upwards.
    assert_eq!(bind(bernini_query(848, 480, 25)), None);
    assert_eq!(bind(bernini_query(640, 480, 49)), None);
    assert_eq!(bind(bernini_query(848, 256, 49)), None);
    // Every identity axis is a conjunct — a foreign tier, mode, provider, lane, overlay or
    // reference surface must not borrow this cell's measurement.
    for foreign in [
        ExceededBoundQuery {
            tier: "q4",
            ..bernini_query(848, 480, 49)
        },
        ExceededBoundQuery {
            mode: "image_to_video",
            ..bernini_query(848, 480, 49)
        },
        ExceededBoundQuery {
            provider: "bernini_image",
            ..bernini_query(848, 480, 49)
        },
        ExceededBoundQuery {
            backend: AnchorBackend::Candle,
            ..bernini_query(848, 480, 49)
        },
        ExceededBoundQuery {
            overlay: Some("control"),
            ..bernini_query(848, 480, 49)
        },
        ExceededBoundQuery {
            reference_count: 1,
            ..bernini_query(848, 480, 49)
        },
        ExceededBoundQuery {
            model_family: Some("wan"),
            ..bernini_query(848, 480, 49)
        },
        ExceededBoundQuery {
            route: Some("wan_2_2_t2v_a14b"),
            ..bernini_query(848, 480, 49)
        },
    ] {
        assert_eq!(bind(foreign), None, "a foreign identity borrowed the bound");
    }
    // A caller that cannot state the catalog axes still binds: both are functions of axes
    // already graded, and the capture-side mirror has no catalog to resolve them from.
    assert_eq!(
        bind(ExceededBoundQuery {
            model_family: None,
            route: None,
            ..bernini_query(848, 480, 49)
        }),
        Some(97_147_294_328),
    );
}

/// sc-22738 (Michael's standing rule): the runtime ALWAYS behaves as if the measurement were
/// valid. A bound whose `loaderClosureDigest` matches nothing the packaged closures declare —
/// or that is stamped at a digest the ledger has since moved past — binds exactly as a current
/// one. The lookup signature cannot even take the closures, so the only way to demote here is to
/// re-add the parameter. MUTATION: restoring `bound.is_current(closures)` as a conjunct (or
/// `let closures = closures?;`) turns every assertion below red.
#[test]
fn a_bound_whose_loader_closure_moved_still_binds() {
    let store = bound_store(vec![bernini_bound()]);
    // The store's bound is stamped at "b"*64; the packaged ledger declares something else for
    // bernini:mlx (or nothing at all — the lookup does not care which).
    let declared = load_anchor_loader_closures(PACKAGED_ANCHOR_LOADER_CLOSURES)
        .expect("packaged closures parse")
        .digest_for("bernini", AnchorBackend::Mlx)
        .map(str::to_owned);
    assert_ne!(
        declared.as_deref(),
        Some(
            store.exceeded_bounds[0]
                .source
                .loader_closure_digest
                .as_str()
        ),
        "the fixture must be stale against the ledger for this test to prove anything"
    );
    assert_eq!(
        store
            .binding_exceeded_bound(bernini_query(848, 480, 49))
            .map(|bound| bound.observed_footprint_bytes),
        Some(97_147_294_328),
    );
    // And a bound at a digest no ledger has ever declared binds too: currency is not a term.
    let mut undeclared = bernini_bound();
    undeclared.source.loader_closure_digest = "f".repeat(64);
    let store = bound_store(vec![undeclared]);
    assert_eq!(
        store
            .binding_exceeded_bound(bernini_query(1280, 720, 97))
            .map(|bound| bound.observed_footprint_bytes),
        Some(97_147_294_328),
    );
}

/// sc-22738: several bounds can cover one request once a cell has been stopped more than once.
/// The STRONGEST true inequality is the useful one.
#[test]
fn the_largest_covering_bound_wins() {
    let smaller = ExceededBound {
        id: "exceeded:bernini:mlx:bf16:base:base:fp:exc-small".to_owned(),
        observed_footprint_bytes: 80_000_000_000,
        geometry: AnchorGeometry {
            width: 848,
            height: 480,
            frames: 25,
            fps: None,
        },
        ..bernini_bound()
    };
    let store = bound_store(vec![smaller, bernini_bound()]);
    assert_eq!(
        store
            .binding_exceeded_bound(bernini_query(848, 480, 49))
            .map(|bound| bound.observed_footprint_bytes),
        Some(97_147_294_328),
    );
    // At a geometry only the smaller bound covers, the smaller one is the whole claim.
    assert_eq!(
        store
            .binding_exceeded_bound(bernini_query(848, 480, 25))
            .map(|bound| bound.observed_footprint_bytes),
        Some(80_000_000_000),
    );
}

/// sc-22738: the requirement a host is graded against is the measured footprint PLUS the lane's
/// activation allowance — never the bare reading, which was taken mid-climb.
#[test]
fn the_requirement_is_the_measured_footprint_plus_the_lanes_headroom() {
    let bound = bernini_bound();
    assert_eq!(bound.required_bytes(0), 97_147_294_328);
    assert_eq!(
        bound.required_bytes(gib_to_bytes(16.0)),
        97_147_294_328 + 17_179_869_184
    );
    assert_eq!(
        bound.required_bytes(u64::MAX),
        u64::MAX,
        "the sum saturates rather than wrapping into a requirement of nearly zero",
    );
}

/// sc-22738: THE predicate, and the disjunct that makes the Bernini fix bite.
///
/// A host no larger than the one the render failed on is refused outright. Grading such a host
/// on footprint-versus-budget alone would re-admit it — 90.5 GiB plus the lane's 16 GiB
/// allowance is 106.5 GiB, and an idle 128 GiB Mac appears to have that — which is exactly the
/// re-admission this story exists to stop.
#[test]
fn a_host_no_larger_than_the_one_that_failed_is_refused_outright() {
    let bound = bernini_bound();
    let headroom = gib_to_bytes(16.0);
    let host = 137_438_953_472_u64;
    assert!(
        bound.required_bytes(headroom) < host,
        "the requirement alone does NOT exceed an idle 128 GiB host; the host-size disjunct \
             is what refuses it",
    );
    assert!(
        bound.refuses_host(host, host, headroom),
        "the same 128 GiB machine that could not finish this render must not be given it again",
    );
    assert!(
        bound.refuses_host(host - 1, u64::MAX, headroom),
        "a SMALLER host is refused even with an impossible budget",
    );
    // A larger host is graded on what it can actually offer.
    let bigger = host * 4;
    assert!(
        !bound.refuses_host(bigger, bigger, headroom),
        "a machine four times the size may still run it",
    );
    assert!(
        bound.refuses_host(bigger, bound.required_bytes(headroom) - 1, headroom),
        "a larger but busy host that cannot offer the requirement is refused",
    );
    assert!(
        !bound.refuses_host(bigger, bound.required_bytes(headroom), headroom),
        "exact equality fits",
    );
}

/// sc-22738: the store refuses a bound that is not a hard stop's reading — one under its own
/// ceiling, one above the whole capture host, or one that states no reason.
#[test]
fn a_bound_that_is_not_a_hard_stops_reading_fails_the_load() {
    let refuse = |bound: ExceededBound, fragment: &str| {
        let error = validate_exceeded_bounds(&bound_store(vec![bound]))
            .expect_err("the store accepted a bound that is not a hard stop's reading");
        assert!(error.contains(fragment), "unexpected error: {error}");
    };
    refuse(
        ExceededBound {
            observed_footprint_bytes: 94_822_600_831,
            ..bernini_bound()
        },
        "under its own hard-stop ceiling",
    );
    refuse(
        ExceededBound {
            host_memory_bytes: 97_147_294_327,
            ..bernini_bound()
        },
        "larger than the whole capture host",
    );
    refuse(
        ExceededBound {
            reason: "  ".to_owned(),
            ..bernini_bound()
        },
        "states no hard-stop reason",
    );
    refuse(
        ExceededBound {
            geometry: AnchorGeometry {
                frames: 0,
                ..bernini_bound().geometry
            },
            ..bernini_bound()
        },
        "degenerate geometry",
    );
    refuse(
        ExceededBound {
            source: AnchorSource {
                loader_closure_digest: "not-a-digest".to_owned(),
                ..bernini_bound().source
            },
            ..bernini_bound()
        },
        "is not a sha256",
    );
    // The corpus handshake, both halves: a bound may only cite a compiled-in retained-evidence
    // file, and it must hash to what the row recorded — the store may not drift from the
    // evidence a refusal is drawn from.
    refuse(
        ExceededBound {
            source: AnchorSource {
                path: "docs/calibration/sc-22738/not-compiled-in.json".to_owned(),
                ..bernini_bound().source
            },
            ..bernini_bound()
        },
        "not a compiled retained-evidence file",
    );
    refuse(bernini_bound(), "source digest mismatch");
}

/// sc-22738: the packaged Bernini bound is bound BYTE-EXACTLY to the corpus it cites — this is
/// the row a production refusal is drawn from, so a store that drifted from its evidence would
/// refuse renders on a number nothing measured.
#[test]
fn the_packaged_bernini_bound_agrees_with_its_retained_evidence() {
    let store = packaged_memory_anchors().expect("the packaged anchor store must load");
    let bound = store
        .exceeded_bounds
        .iter()
        .find(|bound| bound.model_id == "bernini" && bound.tier == "bf16")
        .expect("the bernini:bf16:mlx footprint hard stop is retained");
    assert_eq!(bound.backend, AnchorBackend::Mlx);
    assert_eq!(bound.mode, "text_to_video");
    assert_eq!(bound.geometry.frames, 49);
    assert_eq!(bound.observed_footprint_bytes, 97_147_294_328);
    assert_eq!(bound.host_memory_bytes, 137_438_953_472);
    // `validate_exceeded_bounds` re-derived every one of those figures from the compiled-in
    // bundle at load; asserting the load succeeded plus the row's own content is the whole of
    // the handshake, stated without a second copy of the comparison.
    assert!(validate_exceeded_bounds(store).is_ok());
}

/// AC 3: `ArchitectureFacts::default()` leaves every residue unscaled, so the estimate is
/// never smaller than with full facts — for the whole set and for each fact dropped alone.
#[test]
fn missing_facts_leave_residues_unscaled_and_never_shrink_the_estimate() {
    let request = at(1024, 1024, FULLY_ENGAGED);
    let anchor = z_image_q4_anchor();
    let full = z_image_at(&request);
    let none = anchor
        .derive_phase_peaks(
            &request,
            Z_IMAGE_Q4_COMPONENTS,
            ArchitectureFacts::default(),
        )
        .expect("no facts still derives");
    assert!(none.conditioning >= full.conditioning);
    assert!(none.denoise >= full.denoise);
    assert!(none.decode >= full.decode);
    // With no facts the window and the chunk cannot scale anything: denoise is the anchor's
    // own measured denoise (whole transformer + whole residue), i.e. the staged rung's.
    assert_eq!(none.denoise, anchor.phase_active_peak_bytes.denoise);
    assert!(none.denoise > full.denoise);
    // The decode floor needs the activation width (the blender's output accumulator is in
    // that dtype), so without facts the whole decode residue stays unscaled: decode is the
    // anchor's own measured decode, i.e. the staged rung's.
    assert_eq!(none.decode, anchor.phase_active_peak_bytes.decode);
    assert!(none.decode > full.decode);

    // Dropping any ONE fact never shrinks the estimate below the full-facts one.
    type FactDrop = fn(&mut ArchitectureFacts);
    let drops: [(&str, FactDrop); 8] = [
        ("attention_heads", |facts| facts.attention_heads = None),
        ("head_dim", |facts| facts.head_dim = None),
        ("transformer_blocks", |facts| {
            facts.transformer_blocks = None
        }),
        ("patch_size", |facts| facts.patch_size = None),
        ("latent_channels", |facts| facts.latent_channels = None),
        ("vae_spatial_scale", |facts| facts.vae_spatial_scale = None),
        ("vae_temporal_scale", |facts| {
            facts.vae_temporal_scale = None
        }),
        ("activation_dtype_width", |facts| {
            facts.activation_dtype_width = None
        }),
    ];
    for (label, drop) in drops {
        let mut facts = Z_IMAGE_FACTS;
        drop(&mut facts);
        let partial = anchor
            .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, facts)
            .unwrap_or_else(|| panic!("without {label} the law must still derive"));
        for (phase, partial_bytes, full_bytes) in [
            ("conditioning", partial.conditioning, full.conditioning),
            ("denoise", partial.denoise, full.denoise),
            ("decode", partial.decode, full.decode),
        ] {
            assert!(
                partial_bytes >= full_bytes,
                "without {label}, {phase} {partial_bytes} fell below the full-facts \
                     {full_bytes}"
            );
        }
    }
    // …and the facts that carry a ratio actually move it (so the assertions above are not
    // vacuous): blocks drive the window, heads/width/patch/scale drive the score split, and
    // the activation width alone drives the decode floor.
    for (label, drop) in drops {
        let mut facts = Z_IMAGE_FACTS;
        drop(&mut facts);
        let partial = anchor
            .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, facts)
            .expect("derives");
        let moves_denoise = matches!(
            label,
            "attention_heads"
                | "transformer_blocks"
                | "patch_size"
                | "vae_spatial_scale"
                | "activation_dtype_width"
        );
        assert_eq!(
            partial.denoise > full.denoise,
            moves_denoise,
            "{label}: dropping it {} change the denoise estimate",
            if moves_denoise { "must" } else { "must not" }
        );
        let moves_decode = label == "activation_dtype_width";
        assert_eq!(
            partial.decode > full.decode,
            moves_decode,
            "{label}: dropping it {} change the decode estimate",
            if moves_decode { "must" } else { "must not" }
        );
    }
}

// -------------------------------------------------------------------------------------
// Image derivation law — the retained corpora as falsifiers.
// -------------------------------------------------------------------------------------

/// The Krea candle five-rung corpus derived from its own staged record: every rung's derived
/// admission peak brackets the measured one, per phase where the composition matches the
/// law's, inside a documented tightness budget — and the deeper rungs are priced strictly
/// below the staged rung, so the ratios are known to have bitten.
#[test]
fn candle_anchor_derivation_brackets_every_retained_candle_measurement() {
    /// Deeper-rung tightness. The binding case is the retained window-1 rung at 2.14x its
    /// measured 3.996 GB peak, and the phase that binds is DECODE: the law scales Krea's
    /// decode residue by the larger of the two decode chunk models — the (512 − 128) x 1024
    /// host-transfer band, 3/8 of the image — because it cannot know that Krea tiles its VAE
    /// on device at 512² (1/4 of the image; see `DecodeTile::chunk_pixels`), and prices that
    /// rung's decode at 8.55 GB against a measured 3.83 GB. The next rungs sit at 1.18x and
    /// 1.02x. Forgetting to subtract the components would put the window-1 rung above 5x.
    const DEEPER_RUNG_TIGHTNESS: f64 = 2.5;
    let anchor = krea_candle_anchor();
    let corpus = krea_candle_corpus();
    assert!(
        corpus.len() >= 5,
        "the retained candle corpus must span the five rungs, found {}",
        corpus.len()
    );
    let staged = anchor
        .derive_phase_peaks(
            &at(
                anchor.geometry.width,
                anchor.geometry.height,
                RequestRegime::staged(),
            ),
            KREA_Q4_COMPONENTS,
            KREA_FACTS,
        )
        .expect("the staged rung derives");
    let mut bracketed = 0usize;
    for record in &corpus {
        let derived = anchor
            .derive_phase_peaks(
                &at(record.width, record.height, record.regime),
                KREA_Q4_COMPONENTS,
                KREA_FACTS,
            )
            .unwrap_or_else(|| panic!("{} must be derivable", record.id));
        let peak = derived.peak_bytes();
        assert!(
            peak >= record.peak(),
            "{} ({:?}): derived peak {peak} under-predicts the measured {}",
            record.id,
            record.engaged,
            record.peak()
        );
        assert!(
            derived.conditioning >= record.conditioning,
            "{} conditioning: derived {} under the measured {}",
            record.id,
            derived.conditioning,
            record.conditioning
        );
        assert!(
            derived.denoise >= record.denoise,
            "{} denoise: derived {} under the measured {}",
            record.id,
            derived.denoise,
            record.denoise
        );
        // Decode per phase only where the composition is the law's: Krea's staged path keeps
        // the DiT resident through decode (the retained resident-minus-staged decode delta is
        // the text encoder alone), so its tiled decode phases carry ~8.6 GB of transformer
        // the VAE-only decode composition does not price. The admission PEAK still brackets
        // them above; the decode phase itself is asserted where the anchor's composition
        // holds.
        if record.regime.decode_tile.is_none() {
            assert!(
                derived.decode >= record.decode,
                "{} decode: derived {} under the measured {}",
                record.id,
                derived.decode,
                record.decode
            );
        }
        let ratio = peak as f64 / record.peak() as f64;
        let budget = if record.regime == RequestRegime::staged() {
            1.0
        } else {
            DEEPER_RUNG_TIGHTNESS
        };
        assert!(
            ratio <= budget,
            "{} ({:?}): derived peak {peak} is {ratio:.3}x the measured {}, over the {budget}x \
                 budget",
            record.id,
            record.engaged,
            record.peak()
        );
        if record.regime != RequestRegime::staged() && record.regime.staged {
            assert!(
                peak < staged.peak_bytes(),
                "{} ({:?}): a deeper rung must price below the staged rung {}",
                record.id,
                record.engaged,
                staged.peak_bytes()
            );
        }
        bracketed += 1;
    }
    assert!(
        bracketed >= 5,
        "all five retained candle compositions must be bracketed, got {bracketed}"
    );
}

/// sc-22734. A candle cell whose provider contract classifies `staged_residency`
/// STRUCTURALLY NOT APPLICABLE has no staged composition to have been captured in, so its
/// anchor is the RESIDENT render and the entry point prices it. The two shapes are mutually
/// exclusive, so no packaged row's disposition moves: the flag defaults to `false` and every
/// existing anchor keeps the staged-only guard exactly.
#[test]
fn a_structurally_staging_free_candle_cell_is_priced_from_its_resident_anchor() {
    let staged_anchor = krea_candle_anchor();
    // The exempt cell's anchor is a RESIDENT render, so the fixture must be one: the
    // re-captured z_image_turbo q4 row is the packaged anchor whose every phase level carries
    // the whole resident set (`the_recaptured_z_image_mlx_anchor_is_inside_the_laws_domain`),
    // which is exactly the domain a resident derivation needs. Re-labelled onto the candle
    // lane, because no candle capture of a structurally staging-free family exists yet — the
    // sc-22734 arms are what will produce the first one.
    let resident_capture = |exempt: bool| {
        let mut anchor = store()
            .image_anchor_for("z_image_turbo", AnchorBackend::Mlx, "q4")
            .expect("the z_image_turbo q4 MLX anchor is packaged")
            .clone();
        anchor.backend = AnchorBackend::Candle;
        anchor.staged_residency_structurally_not_applicable = exempt;
        anchor
    };
    let components = Z_IMAGE_Q4_MLX_RESIDENT_COMPONENTS;
    let resident = AnchorImageDeriveRequest {
        width: 768,
        height: 768,
        staged_residency: false,
    };
    let staged = AnchorImageDeriveRequest {
        staged_residency: true,
        ..resident
    };

    // The flag defaults off, so the packaged shape is untouched: staged anchor + staged
    // request prices, resident request does not.
    assert!(!staged_anchor.staged_residency_structurally_not_applicable);
    let krea = AnchorImageDeriveRequest {
        width: 1024,
        height: 1024,
        staged_residency: true,
    };
    assert!(staged_anchor
        .derive_image_phase_peaks(krea, KREA_Q4_COMPONENTS)
        .is_some());
    assert!(staged_anchor
        .derive_image_phase_peaks(
            AnchorImageDeriveRequest {
                staged_residency: false,
                ..krea
            },
            KREA_Q4_COMPONENTS,
        )
        .is_none());

    // The exempt cell: a resident anchor, which the law otherwise refuses outright.
    let exempt = resident_capture(true);
    assert!(!exempt.measured_regime.staged);
    let derived = exempt
        .derive_image_phase_peaks(resident, components)
        .expect("a staging-free cell prices from its resident anchor");
    assert_eq!(
        derived,
        AnchorDerivedPhases {
            conditioning: exempt.phase_active_peak_bytes.conditioning,
            denoise: exempt.phase_active_peak_bytes.denoise,
            decode: exempt.phase_active_peak_bytes.decode,
        },
        "at the anchor's own geometry the resident derivation is the anchor"
    );
    // ...and it answers ONLY the resident candidate. A staged candidate on a cell with no
    // staged composition is not a request this engine can be asked to honour.
    assert!(exempt
        .derive_image_phase_peaks(staged, components)
        .is_none());

    // The two mixed shapes stay refused, so the flag can never launder a record the law
    // cannot honestly price: a STAGED anchor on an exempt cell, and a RESIDENT anchor on a
    // cell that has a staged composition it should have been captured in.
    let mut staged_on_exempt = resident_capture(true);
    staged_on_exempt.measured_regime.staged = true;
    for request in [staged, resident] {
        assert!(staged_on_exempt
            .derive_image_phase_peaks(request, components)
            .is_none());
    }
    let resident_on_ordinary = resident_capture(false);
    assert!(!resident_on_ordinary.measured_regime.staged);
    for request in [staged, resident] {
        assert!(resident_on_ordinary
            .derive_image_phase_peaks(request, components)
            .is_none());
    }

    // Every deeper rung is still refused on the exempt lane, and so is an underived row.
    for deeper in ["decode_tiled", "attention_chunked", "transformer_windowed"] {
        let mut anchor = exempt.clone();
        match deeper {
            "decode_tiled" => anchor.measured_regime.decode_tiled = true,
            "attention_chunked" => anchor.measured_regime.attention_chunked = true,
            _ => anchor.measured_regime.transformer_windowed = true,
        }
        assert!(
            anchor
                .derive_image_phase_peaks(resident, components)
                .is_none(),
            "{deeper} is deeper than the law prices, exemption or not"
        );
    }
    let mut underived = exempt.clone();
    underived.underived_reason = Some("measured_envelope".to_owned());
    assert!(underived
        .derive_image_phase_peaks(resident, components)
        .is_none());

    // The flag rides the store round trip, and is absent from a row that does not set it.
    let json = serde_json::to_string(&exempt).expect("serialize");
    assert!(json.contains("\"stagedResidencyStructurallyNotApplicable\":true"));
    assert_eq!(
        serde_json::from_str::<MemoryAnchor>(&json).expect("round trip"),
        exempt.clone()
    );
    let packaged = serde_json::to_string(&staged_anchor).expect("serialize");
    assert!(
        !packaged.contains("stagedResidencyStructurallyNotApplicable"),
        "a row that does not set it stays byte-identical to the packaged shape"
    );
    assert_eq!(
        serde_json::from_str::<MemoryAnchor>(&packaged).expect("round trip"),
        staged_anchor.clone()
    );
}

/// The candle lane's entry point is the same law: at the anchor's geometry the staged
/// derivation equals the anchor, and it keeps the shallow-anchor guards its consumers were
/// written against — a resident request, a deeper-measured anchor, another lane, an
/// underived row are refused there, not by the law.
#[test]
fn the_candle_entry_point_keeps_its_guards_while_the_law_prices_what_it_refuses() {
    let anchor = krea_candle_anchor();
    let request = AnchorImageDeriveRequest {
        width: 1024,
        height: 1024,
        staged_residency: true,
    };
    let derived = anchor
        .derive_image_phase_peaks(request, KREA_Q4_COMPONENTS)
        .expect("the shallow staged anchor prices its own geometry");
    assert_eq!(
        derived,
        AnchorDerivedPhases {
            conditioning: anchor.phase_active_peak_bytes.conditioning,
            denoise: anchor.phase_active_peak_bytes.denoise,
            decode: anchor.phase_active_peak_bytes.decode,
        }
    );
    // The entry point refuses the resident composition; the law prices it, above staged.
    assert!(anchor
        .derive_image_phase_peaks(
            AnchorImageDeriveRequest {
                staged_residency: false,
                ..request
            },
            KREA_Q4_COMPONENTS,
        )
        .is_none());
    let resident = anchor
        .derive_phase_peaks(
            &at(1024, 1024, RequestRegime::resident()),
            KREA_Q4_COMPONENTS,
            KREA_FACTS,
        )
        .expect("the law prices the resident composition");
    assert!(resident.peak_bytes() > derived.peak_bytes());
    let measured_resident = krea_candle_corpus()
        .into_iter()
        .find(|record| record.engaged == ["resident"])
        .expect("the retained resident capture");
    assert!(resident.conditioning >= measured_resident.conditioning);
    assert!(resident.denoise >= measured_resident.denoise);
    assert!(resident.decode >= measured_resident.decode);

    for (label, mutate) in [
        (
            "a deeper measured regime",
            Box::new(|anchor: &mut MemoryAnchor| anchor.measured_regime.decode_tiled = true)
                as Box<dyn Fn(&mut MemoryAnchor)>,
        ),
        (
            "a resident measured regime",
            Box::new(|anchor: &mut MemoryAnchor| anchor.measured_regime.staged = false),
        ),
        (
            "the other backend lane",
            Box::new(|anchor: &mut MemoryAnchor| anchor.backend = AnchorBackend::Mlx),
        ),
        (
            "a stated underived reason",
            Box::new(|anchor: &mut MemoryAnchor| {
                anchor.underived_reason = Some("validation-only".to_owned());
            }),
        ),
    ] {
        let mut mutated = anchor.clone();
        mutate(&mut mutated);
        assert!(
            mutated
                .derive_image_phase_peaks(request, KREA_Q4_COMPONENTS)
                .is_none(),
            "the candle entry point must refuse an anchor carrying {label}"
        );
    }
}

/// The identity and regime guards of the law itself, each mutated individually onto an
/// otherwise-derivable anchor.
#[test]
fn the_law_refuses_video_identities_and_unbounded_requests_of_bounded_anchors() {
    let anchor = z_image_q4_anchor();
    let request = at(1024, 1024, FULLY_ENGAGED);
    assert!(anchor
        .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
        .is_some());
    for (label, mutate) in [
        (
            "a video pipeline variant",
            Box::new(|anchor: &mut MemoryAnchor| {
                anchor.transformer_variant = Some(Ltx25TransformerVariant::Dev);
            }) as Box<dyn Fn(&mut MemoryAnchor)>,
        ),
        (
            "a video decoder",
            Box::new(|anchor: &mut MemoryAnchor| anchor.decoder = Some(Ltx25Decoder::Conv)),
        ),
        (
            "a multi-frame measured geometry",
            Box::new(|anchor: &mut MemoryAnchor| anchor.geometry.frames = 145),
        ),
    ] {
        let mut mutated = anchor.clone();
        mutate(&mut mutated);
        assert!(
            mutated
                .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
                .is_none(),
            "the law must refuse an anchor carrying {label}"
        );
    }
    // The packaged LTX video anchors are refused for their axes, and the video law refuses
    // the image anchors in turn.
    let ltx = store()
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "bf16",
            Ltx25TransformerVariant::Dev,
            Ltx25Decoder::DiffVae,
        )
        .expect("the LTX bf16 dev/diffvae anchor");
    assert!(ltx
        .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
        .is_none());
    assert!(krea_candle_anchor()
        .derive_video_phase_peaks(plain_request(1024, 1024, 1))
        .is_none());

    // A phase the anchor measured BOUNDED cannot price the same phase unbounded…
    for (label, measured, unbounded) in [
        (
            "tiled",
            AnchorMeasuredRegime {
                decode_tiled: true,
                ..anchor.measured_regime
            },
            RequestRegime {
                decode_tile: None,
                ..FULLY_ENGAGED
            },
        ),
        (
            "chunked",
            AnchorMeasuredRegime {
                attention_chunked: true,
                ..anchor.measured_regime
            },
            RequestRegime {
                attention_chunk_scores: None,
                ..FULLY_ENGAGED
            },
        ),
        (
            "windowed",
            AnchorMeasuredRegime {
                transformer_windowed: true,
                ..anchor.measured_regime
            },
            RequestRegime {
                transformer_window: None,
                ..FULLY_ENGAGED
            },
        ),
    ] {
        let mut bounded = anchor.clone();
        bounded.measured_regime = measured;
        assert!(
            bounded
                .derive_phase_peaks(
                    &at(1024, 1024, unbounded),
                    Z_IMAGE_Q4_COMPONENTS,
                    Z_IMAGE_FACTS
                )
                .is_none(),
            "a {label}-measured anchor must refuse the unbounded request"
        );
        // …while the correspondingly bounded request reuses the bounded residue UNSCALED:
        // it prices at or above what the unbounded anchor's scaled ratio would give.
        let reused = bounded
            .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
            .unwrap_or_else(|| panic!("a {label}-measured anchor prices the bounded request"));
        let scaled = anchor
            .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
            .expect("the unbounded anchor prices the bounded request");
        assert!(reused.peak_bytes() >= scaled.peak_bytes());
    }

    // Degenerate inputs are refused, not clamped.
    for (label, request) in [
        ("a zero width", at(0, 1024, FULLY_ENGAGED)),
        (
            "a zero batch",
            ImageDeriveRequest {
                batch: 0,
                ..request
            },
        ),
        (
            "a zero tile edge",
            at(
                1024,
                1024,
                RequestRegime {
                    decode_tile: Some(DecodeTile {
                        edge: 0,
                        overlap: 0,
                    }),
                    ..FULLY_ENGAGED
                },
            ),
        ),
        (
            "a tile overlap not below its edge (the engines refuse the tuple too)",
            at(
                1024,
                1024,
                RequestRegime {
                    decode_tile: Some(DecodeTile {
                        edge: 512,
                        overlap: 512,
                    }),
                    ..FULLY_ENGAGED
                },
            ),
        ),
        (
            "a zero chunk",
            at(
                1024,
                1024,
                RequestRegime {
                    attention_chunk_scores: Some(0),
                    ..FULLY_ENGAGED
                },
            ),
        ),
        (
            "a zero window",
            at(
                1024,
                1024,
                RequestRegime {
                    transformer_window: Some(0),
                    ..FULLY_ENGAGED
                },
            ),
        ),
    ] {
        assert!(
            anchor
                .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
                .is_none(),
            "{label} must be refused"
        );
    }
}

/// The residue split never goes negative: an anchor whose denoise residue is smaller than
/// the full score tensor its facts imply caps the subtraction at the residue, so the chunked
/// estimate is the resident set plus the chunk and nothing less, and the unchunked estimate
/// at the anchor geometry re-prices the full tensor above the measured peak rather than below.
#[test]
fn the_score_subtraction_is_capped_at_the_residue() {
    let anchor = z_image_q4_anchor();
    // 300 heads make the 1024x1024 score tensor ~12.7 GB, far above the 4.58 GB residue.
    let facts = ArchitectureFacts {
        attention_heads: Some(300),
        ..Z_IMAGE_FACTS
    };
    let chunked = anchor
        .derive_phase_peaks(
            &at(
                1024,
                1024,
                RequestRegime {
                    attention_chunk_scores: Some(64 * 1024 * 1024),
                    ..RequestRegime::staged()
                },
            ),
            Z_IMAGE_Q4_COMPONENTS,
            facts,
        )
        .expect("derives");
    assert_eq!(
        chunked.denoise,
        Z_IMAGE_Q4_COMPONENTS.transformer + 64 * 1024 * 1024 * 2
    );
    let unchunked = anchor
        .derive_phase_peaks(
            &at(1024, 1024, RequestRegime::staged()),
            Z_IMAGE_Q4_COMPONENTS,
            facts,
        )
        .expect("derives");
    assert!(unchunked.denoise > anchor.phase_active_peak_bytes.denoise);
}

/// A chunk budget larger than the whole score tensor is capped at the tensor: the chunked
/// rung can never price above the unchunked one. At 256x256 the Z-Image tensor is
/// 30 x 1536² x 2 B = 141.6 MB, so a 1 Gi-score chunk (2 GiB) is capped to it; at 1024x1024
/// the 64 Mi chunk sits below the 1.27 GB tensor and must bite by exactly the difference.
#[test]
fn the_chunk_term_is_capped_at_the_unchunked_score_tensor() {
    let anchor = z_image_q4_anchor();
    let chunked = |width: u32, chunk: u64| {
        anchor
            .derive_phase_peaks(
                &at(
                    width,
                    width,
                    RequestRegime {
                        attention_chunk_scores: Some(chunk),
                        ..RequestRegime::staged()
                    },
                ),
                Z_IMAGE_Q4_COMPONENTS,
                Z_IMAGE_FACTS,
            )
            .expect("derives")
    };
    let unchunked = |width: u32| {
        anchor
            .derive_phase_peaks(
                &at(width, width, RequestRegime::staged()),
                Z_IMAGE_Q4_COMPONENTS,
                Z_IMAGE_FACTS,
            )
            .expect("derives")
    };
    // 256x256: 32² + 512 = 1536 tokens, tensor 30 x 1536² x 2 B = 141,557,760 B; a 1 Gi
    // chunk (2 GiB of scores) is far above it and is capped, so chunked == unchunked.
    let oversized = 1024 * 1024 * 1024;
    assert!(oversized * 2 > 141_557_760);
    assert_eq!(chunked(256, oversized).denoise, unchunked(256).denoise);
    // 1024x1024: the 64 Mi chunk is below the 1,274,019,840 B tensor and bites by exactly
    // the difference.
    let chunk_bytes = 64 * 1024 * 1024 * 2;
    assert!(chunk_bytes < 1_274_019_840);
    assert_eq!(
        unchunked(1024).denoise - chunked(1024, 64 * 1024 * 1024).denoise,
        1_274_019_840 - chunk_bytes
    );
}

/// The conditioning residue scales with the prompt's conditioning tokens over the default
/// — and never below 1.0, since the records do not state the prompt length they were
/// measured with. The joint sequence length moves the denoise estimate with it.
#[test]
fn conditioning_tokens_scale_the_conditioning_residue_and_never_credit_a_short_prompt() {
    let anchor = z_image_q4_anchor();
    let with_tokens = |tokens: Option<u32>| {
        anchor
            .derive_phase_peaks(
                &ImageDeriveRequest {
                    conditioning_tokens: tokens,
                    ..at(1024, 1024, RequestRegime::staged())
                },
                Z_IMAGE_Q4_COMPONENTS,
                Z_IMAGE_FACTS,
            )
            .expect("derives")
    };
    let default = with_tokens(None);
    assert_eq!(default, with_tokens(Some(DEFAULT_CONDITIONING_TOKENS)));
    let residue = anchor.phase_active_peak_bytes.conditioning - Z_IMAGE_Q4_COMPONENTS.conditioning;
    assert!(residue > 0);
    // Twice the tokens: twice the residue on top of the text encoder, and a longer joint
    // sequence for denoise.
    let doubled = with_tokens(Some(2 * DEFAULT_CONDITIONING_TOKENS));
    assert_eq!(
        doubled.conditioning,
        Z_IMAGE_Q4_COMPONENTS.conditioning + 2 * residue
    );
    assert!(doubled.denoise > default.denoise);
    // Half the tokens: no credit anywhere — the request's count is floored at the default
    // on every side it enters, so the estimate is the default's in every phase.
    let halved = with_tokens(Some(DEFAULT_CONDITIONING_TOKENS / 2));
    assert_eq!(halved, default);
}

/// The fixture component bytes are not free literals: each set sums to its tier's packaged
/// download (`estimatedSizeBytes` in the builtin manifest) within 2%, so a retyped figure
/// or a re-cut tier lands here.
#[test]
fn the_fixture_component_bytes_sum_to_their_packaged_tier_sizes() {
    for (label, components, model, tier) in [
        (
            "Z-Image-Turbo q4",
            Z_IMAGE_Q4_COMPONENTS,
            "z_image_turbo",
            "q4",
        ),
        ("FLUX.2 [dev] q4", flux2_components("q4"), "flux2_dev", "q4"),
        ("FLUX.2 [dev] q8", flux2_components("q8"), "flux2_dev", "q8"),
    ] {
        let packaged = packaged_tier_bytes(model, tier);
        let sum = components.total();
        let drift = (sum as f64 - packaged as f64).abs() / packaged as f64;
        assert!(
            drift <= 0.02,
            "{label}: components sum to {sum} against the packaged {packaged} ({drift:.4})"
        );
        assert!(components.conditioning > 0 && components.transformer > 0);
        assert!(components.decoder > 0);
    }
    assert_eq!(packaged_tier_bytes("z_image_turbo", "q4"), 5_909_406_487);
}

/// Batch multiplies the activation residues (and the score tensor), never the weights.
#[test]
fn batch_scales_the_residues_and_not_the_components() {
    let anchor = z_image_q4_anchor();
    let request = at(1024, 1024, RequestRegime::staged());
    let one = z_image_at(&request);
    let two = z_image_at(&ImageDeriveRequest {
        batch: 2,
        ..request
    });
    assert!(two.conditioning > one.conditioning);
    assert!(two.denoise > one.denoise);
    assert!(two.decode > one.decode);
    let components = Z_IMAGE_Q4_COMPONENTS;
    let residue = |peak: u64, resident: u64| peak - resident;
    let peaks = anchor.phase_active_peak_bytes;
    assert_eq!(
        two.conditioning,
        components.conditioning + 2 * residue(peaks.conditioning, components.conditioning)
    );
    assert_eq!(
        two.decode,
        components.decoder + 2 * residue(peaks.decode, components.decoder)
    );
    assert_eq!(
        two.denoise,
        components.transformer + 2 * residue(peaks.denoise, components.transformer)
    );
}

// -------------------------------------------------------------------------------------
// MLX image lane entry point.
// -------------------------------------------------------------------------------------

fn flux2_mlx_anchor(tier: &str) -> &'static MemoryAnchor {
    store()
        .image_anchor_for("flux2_dev", AnchorBackend::Mlx, tier)
        .unwrap_or_else(|| panic!("the flux2_dev {tier} MLX anchor is packaged"))
}

/// FLUX.2 [dev] MLX component bytes per tier: the tier's packaged download
/// (`estimatedSizeBytes` in the builtin manifest — 33.6 GB q4, 55.0 GB q8) split at the
/// 24B Mistral text encoder's 4-/8-bit size and the VAE. Bound to the manifest by
/// `the_fixture_component_bytes_sum_to_their_packaged_tier_sizes` (within 2%).
fn flux2_components(tier: &str) -> ComponentBytes {
    match tier {
        "q4" => ComponentBytes {
            conditioning: 13_400_000_000,
            transformer: 19_850_000_000,
            decoder: 350_000_000,
        },
        "q8" => ComponentBytes {
            conditioning: 25_400_000_000,
            transformer: 29_250_000_000,
            decoder: 350_000_000,
        },
        other => panic!("no flux2 components for {other}"),
    }
}

/// The MLX entry point's guards, each mutated individually on an otherwise-derivable anchor
/// (an in-domain synthetic one — the packaged rows are refused by the law's domain guard,
/// asserted here too), and its area scaling in both directions.
#[test]
fn the_mlx_entry_point_keeps_its_guards_and_tracks_output_area() {
    let request = AnchorMlxImageDeriveRequest {
        width: 1024,
        height: 1024,
    };
    let components = flux2_components("q4");
    let in_domain = in_domain_resident_mlx_anchor(components);
    assert!(in_domain
        .derive_mlx_image_phase_peaks(request, components)
        .is_some());
    for (label, mutate) in [
        (
            "the candle lane",
            Box::new(|anchor: &mut MemoryAnchor| anchor.backend = AnchorBackend::Candle)
                as Box<dyn Fn(&mut MemoryAnchor)>,
        ),
        (
            "deferred materialization",
            Box::new(|anchor: &mut MemoryAnchor| {
                anchor.load_shape = AnchorLoadShape::DeferredMaterialization;
            }),
        ),
        (
            "a staged measured regime",
            Box::new(|anchor: &mut MemoryAnchor| anchor.measured_regime.staged = true),
        ),
        (
            "a tiled measured regime",
            Box::new(|anchor: &mut MemoryAnchor| anchor.measured_regime.decode_tiled = true),
        ),
        (
            "a stated underived reason",
            Box::new(|anchor: &mut MemoryAnchor| {
                anchor.underived_reason = Some("validation-only".to_owned());
            }),
        ),
    ] {
        let mut mutated = in_domain.clone();
        mutate(&mut mutated);
        assert!(
            mutated
                .derive_mlx_image_phase_peaks(request, components)
                .is_none(),
            "the MLX entry point must refuse an anchor carrying {label}"
        );
    }
    // Every packaged MLX image anchor is refused at the entry point: the flux2 rows by the
    // law's domain guard (their conditioning counters saw no weights), the single-geometry
    // rows by their stated underived reason AND the domain guard.
    for (model, tier) in [
        ("flux2_dev", "q4"),
        ("flux2_dev", "q8"),
        ("qwen_image", "bf16"),
        ("z_image_turbo", "q4"),
    ] {
        let anchor = store()
            .image_anchor_for(model, AnchorBackend::Mlx, tier)
            .unwrap_or_else(|| panic!("{model} {tier} anchor"));
        let components = ComponentBytes {
            conditioning: packaged_tier_bytes(model, tier),
            transformer: 0,
            decoder: 0,
        };
        assert!(anchor
            .derive_mlx_image_phase_peaks(request, components)
            .is_none());
        assert!(anchor
            .derive_phase_peaks(
                &at(1024, 1024, RequestRegime::resident()),
                components,
                ArchitectureFacts::default()
            )
            .is_none());
    }
    // Output-area tracking through the entry point, on the in-domain anchor: the allocator
    // envelope is carried unscaled below the anchor geometry and pixel-scaled above it.
    let anchor = in_domain_resident_mlx_anchor(flux2_components("q8"));
    let at = |width: u32, height: u32| {
        anchor
            .derive_mlx_image_phase_peaks(
                AnchorMlxImageDeriveRequest { width, height },
                flux2_components("q8"),
            )
            .unwrap_or_else(|| panic!("{width}x{height} must be derivable"))
    };
    assert!(at(512, 512).peak_bytes() < at(768, 768).peak_bytes());
    assert!(at(768, 768).peak_bytes() < at(1024, 1024).peak_bytes());
    assert!(at(1024, 1024).peak_bytes() < at(1536, 1536).peak_bytes());
    assert_eq!(
        at(1024, 1024).peak_bytes(),
        anchor
            .phase_active_peak_bytes
            .conditioning
            .max(anchor.phase_active_peak_bytes.denoise)
            .max(anchor.phase_active_peak_bytes.decode),
        "at the anchor geometry the entry point reproduces the active working set"
    );
    assert_eq!(at(1344, 768).peak_bytes(), at(768, 1344).peak_bytes());
    assert!(anchor
        .derive_mlx_image_phase_peaks(
            AnchorMlxImageDeriveRequest {
                width: 0,
                height: 512,
            },
            flux2_components("q8"),
        )
        .is_none());
}

/// An anchor row may carry component bytes; a stated set that sums to zero is refused at
/// load, and the field round-trips through the store.
#[test]
fn component_bytes_round_trip_through_the_store_and_a_zero_set_is_refused() {
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
    let index = store()
        .anchors
        .iter()
        .position(|anchor| anchor.model_id == "krea_2_turbo")
        .expect("the krea anchor");
    doctored["anchors"][index]["componentBytes"] = serde_json::json!({
        "conditioning": KREA_Q4_COMPONENTS.conditioning,
        "transformer": KREA_Q4_COMPONENTS.transformer,
        "decoder": KREA_Q4_COMPONENTS.decoder,
    });
    let loaded =
        load_memory_anchors(&doctored.to_string()).expect("a store carrying component bytes loads");
    assert_eq!(
        loaded.anchors[index].component_bytes,
        Some(KREA_Q4_COMPONENTS)
    );
    let serialized = serde_json::to_string(&loaded).expect("serializes");
    assert!(serialized.contains("\"componentBytes\""));
    assert_eq!(
        load_memory_anchors(&serialized)
            .expect("round-trips")
            .anchors[index]
            .component_bytes,
        Some(KREA_Q4_COMPONENTS)
    );
    // Every packaged row states none today; `None` never serializes.
    assert!(store()
        .anchors
        .iter()
        .all(|anchor| anchor.component_bytes.is_none()));
    assert!(!PACKAGED_MEMORY_ANCHORS.contains("componentBytes"));

    doctored["anchors"][index]["componentBytes"] = serde_json::json!({
        "conditioning": 0,
        "transformer": 0,
        "decoder": 0,
    });
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("all-zero component bytes must be rejected");
    assert!(error.contains("sum to zero"), "{error}");
}

/// An underived VIDEO anchor prices nothing: the video law fits, so it honours the field.
#[test]
fn an_underived_video_anchor_prices_nothing() {
    let mut video = store()
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q8",
            Ltx25TransformerVariant::Dev,
            Ltx25Decoder::DiffVae,
        )
        .expect("q8 dev/diffvae anchor")
        .clone();
    assert!(video
        .derive_video_phase_peaks(plain_request(768, 512, 121))
        .is_some());
    video.underived_reason = Some("validation-only".to_owned());
    assert!(video
        .derive_video_phase_peaks(plain_request(768, 512, 121))
        .is_none());
}

// -------------------------------------------------------------------------------------
// sc-22736: axis-free VIDEO anchors (Wan 2.2, SCAIL-2, LTX-2.3) — the measured-point bound.
// -------------------------------------------------------------------------------------

/// A synthetic Wan 2.2 I2V candle anchor: multi-frame, no pipeline axes, one reference, and
/// stamped with the axis-free underived reason exactly as the extractor writes it.
fn axis_free_video_anchor() -> MemoryAnchor {
    MemoryAnchor {
        id: "wan_2_2_i2v_14b:candle:bf16:-:-:sc-22736-wan2-2-i2v-a14b-candle-dense-v1:imc-test"
            .to_owned(),
        model_id: "wan_2_2_i2v_14b".to_owned(),
        model_family: "wan-video".to_owned(),
        route: "wan2_2_i2v_14b".to_owned(),
        provider: "wan2_2_i2v_14b".to_owned(),
        backend: AnchorBackend::Candle,
        tier: "bf16".to_owned(),
        transformer_variant: None,
        decoder: None,
        mode: "image_to_video".to_owned(),
        overlay: None,
        reference_count: 1,
        load_shape: AnchorLoadShape::EagerMaterialization,
        measured_regime: AnchorMeasuredRegime {
            decode_tiled: false,
            transformer_windowed: false,
            staged: true,
            attention_chunked: false,
        },
        source: AnchorSource {
            path: String::new(),
            sha256: String::new(),
            record_id: "imc-test".to_owned(),
            calibration_fingerprint: "sc-22736-wan2-2-i2v-a14b-candle-dense-v1".to_owned(),
            loader_closure_digest: "0".repeat(64),
            currency_attestation: None,
        },
        geometry: AnchorGeometry {
            width: 1280,
            height: 720,
            frames: 77,
            fps: Some(16),
        },
        phase_active_peak_bytes: AnchorPhaseBytes {
            conditioning: 30_000_000_000,
            denoise: 62_000_000_000,
            decode: 48_000_000_000,
        },
        phase_allocator_envelope_bytes: None,
        overall_allocator_envelope_bytes: 62_000_000_000,
        underived_reason: Some(
            "the source record states no (transformer variant, decoder) pipeline axes and the \
                 video law's per-token coefficients are keyed on them; this anchor validates its \
                 measured point and prices nothing beyond it"
                .to_owned(),
        ),
        component_bytes: None,
        staged_residency_structurally_not_applicable: false,
    }
}

fn widened_measured(anchor: &MemoryAnchor) -> AnchorDerivedPhases {
    let expect = |measured: u64| {
        ((measured as f64) * (1.0 + ANCHOR_ALLOCATOR_ENVELOPE_MARGIN)).ceil() as u64
    };
    AnchorDerivedPhases {
        conditioning: expect(anchor.phase_active_peak_bytes.conditioning),
        denoise: expect(anchor.phase_active_peak_bytes.denoise),
        decode: expect(anchor.phase_active_peak_bytes.decode),
    }
}

/// An axis-free video anchor prices every request it dominates AT its measured point — no
/// per-token slope, none invented — and refuses everything beyond it. Mutations this kills:
/// dropping the pixel or the frame dominance check (the beyond-anchor rows derive); restoring
/// the axis-free `return None` (nothing derives); scaling the intercepts by any coefficient
/// (the equality against the widened measured peaks fails).
#[test]
fn an_axis_free_video_anchor_bounds_every_request_it_dominates_at_its_measured_point() {
    let anchor = axis_free_video_anchor();
    let expected = widened_measured(&anchor);
    // At the measured point itself.
    assert_eq!(
        anchor.derive_video_phase_peaks(plain_request(1280, 720, 77)),
        Some(expected)
    );
    // Dominated: fewer frames, fewer pixels, the portrait twin — each bounded by the same
    // measured peaks, never scaled down (no lower clamp is a per-token law's property; this
    // one has no slope).
    for (width, height, frames) in [
        (1280, 720, 45),
        (832, 480, 77),
        (720, 1280, 77),
        (16, 16, 1),
    ] {
        assert_eq!(
            anchor.derive_video_phase_peaks(plain_request(width, height, frames)),
            Some(expected),
            "{width}x{height} f{frames}"
        );
    }
    // Beyond the measured point in either axis: refused, the caller keeps its floor.
    assert_eq!(
        anchor.derive_video_phase_peaks(plain_request(1280, 720, 81)),
        None
    );
    assert_eq!(
        anchor.derive_video_phase_peaks(plain_request(1280, 736, 77)),
        None
    );
    assert_eq!(
        anchor.derive_video_phase_peaks(plain_request(1920, 1080, 45)),
        None
    );
    // Degenerate geometry is refused, as everywhere.
    assert_eq!(
        anchor.derive_video_phase_peaks(plain_request(0, 720, 77)),
        None
    );
    // A request bounding a phase the anchor measured UNBOUNDED is still priced at the
    // unbounded intercept: no LTX architecture constant is substituted for a Wan phase.
    let mut tiled = plain_request(832, 480, 45);
    tiled.decode_tiled = true;
    assert_eq!(anchor.derive_video_phase_peaks(tiled), Some(expected));
    // ...and the mirror is refused: a tiled anchor cannot price an untiled request.
    let mut tiled_anchor = anchor.clone();
    tiled_anchor.measured_regime.decode_tiled = true;
    assert_eq!(
        tiled_anchor.derive_video_phase_peaks(plain_request(832, 480, 45)),
        None
    );
    let mut deferred_anchor = anchor.clone();
    deferred_anchor.load_shape = AnchorLoadShape::DeferredMaterialization;
    assert_eq!(
        deferred_anchor.derive_video_phase_peaks(plain_request(832, 480, 45)),
        None
    );
    // The axis-free underived reason is the STATEMENT of this bound, not a refusal of it:
    // without the reason the answer is identical.
    let mut unstated = anchor.clone();
    unstated.underived_reason = None;
    assert_eq!(
        unstated.derive_video_phase_peaks(plain_request(832, 480, 45)),
        Some(expected)
    );
    // An axis-free SINGLE-FRAME row is an image anchor and is not a video anchor under any
    // spelling.
    let mut still = anchor.clone();
    still.geometry.frames = 1;
    assert_eq!(
        still.derive_video_phase_peaks(plain_request(832, 480, 1)),
        None
    );
}

/// The store resolves an axis-free video anchor for an axis-free request and for nothing
/// else, and the cell derivation goes through it with no sibling crossing. Mutation this kills:
/// dropping the `frames > 1` guard in `video_anchor_for` (the image row resolves as video), or
/// matching the axes as wildcards (the variant-keyed lookups resolve).
#[test]
fn the_store_resolves_an_axis_free_video_anchor_only_for_an_axis_free_request() {
    let anchor = axis_free_video_anchor();
    let mut image = anchor.clone();
    image.id = "z_image_turbo:candle:q4:-:-:x:imc-image".to_owned();
    image.model_id = "z_image_turbo".to_owned();
    image.geometry.frames = 1;
    let store = MemoryAnchorStore {
        schema_version: 1,
        anchors: vec![anchor.clone(), image],
        analytic_only: Vec::new(),
        component_deltas: Vec::new(),
        exceeded_bounds: Vec::new(),
        observed_operating_envelopes: Vec::new(),
    };
    assert_eq!(
        store
            .video_anchor_for("wan_2_2_i2v_14b", AnchorBackend::Candle, "bf16", None, None)
            .map(|found| found.id.as_str()),
        Some(anchor.id.as_str())
    );
    // Wrong lane, wrong tier, a variant-keyed request: nothing.
    assert!(store
        .video_anchor_for("wan_2_2_i2v_14b", AnchorBackend::Mlx, "bf16", None, None)
        .is_none());
    assert!(store
        .video_anchor_for("wan_2_2_i2v_14b", AnchorBackend::Candle, "q4", None, None)
        .is_none());
    for variant in [
        Ltx25TransformerVariant::Dev,
        Ltx25TransformerVariant::Distilled,
    ] {
        for decoder in [Ltx25Decoder::Conv, Ltx25Decoder::DiffVae] {
            assert!(store
                .video_anchor_for(
                    "wan_2_2_i2v_14b",
                    AnchorBackend::Candle,
                    "bf16",
                    Some(variant),
                    Some(decoder)
                )
                .is_none());
            assert!(store
                .anchor_for(
                    "wan_2_2_i2v_14b",
                    AnchorBackend::Candle,
                    "bf16",
                    variant,
                    decoder
                )
                .is_none());
        }
    }
    // The image row answers no video lookup even though it is axis-free.
    assert!(store
        .video_anchor_for("z_image_turbo", AnchorBackend::Candle, "bf16", None, None)
        .is_none());
    // Cell derivation: the exact anchor, delta 0, the measured-point bound.
    let derived = store
        .derive_video_phase_peaks_for_cell(
            "wan_2_2_i2v_14b",
            AnchorBackend::Candle,
            "bf16",
            None,
            None,
            plain_request(832, 480, 45),
        )
        .expect("the dominated request derives");
    assert_eq!(derived.anchor.id, anchor.id);
    assert_eq!(derived.delta_bytes, 0);
    assert_eq!(derived.phases, widened_measured(&anchor));
    // No sibling crossing exists for an axis-free cell: an unanchored tier keeps its floor.
    assert!(store
        .derive_video_phase_peaks_for_cell(
            "wan_2_2_i2v_14b",
            AnchorBackend::Candle,
            "q4",
            None,
            None,
            plain_request(832, 480, 45),
        )
        .is_none());
}

// -------------------------------------------------------------------------------------
// Variant/decoder component deltas (epic 22505 E2, feature-end fix round).
// -------------------------------------------------------------------------------------

/// E2's headline: an unmeasured (variant, decoder) cell of an anchored (model, tier, lane)
/// derives from the sibling anchor plus the bound file-size delta — and the delta is APPLIED,
/// byte for byte, so zeroing it reds this test.
///
/// The crossing exercised is DEV-from-distilled, because that is the direction the corrected
/// table prices above zero (the dev recipe materializes the distillation LoRA the distilled
/// checkpoint already contains — see [`LTX25_VARIANT_DELTA_COMPONENTS`]). The mirrored
/// distilled-from-dev crossing is a legitimate zero and is asserted by
/// `a_zero_crossing_derives_the_sibling_estimate_unchanged_rather_than_refusing`.
#[test]
fn an_unmeasured_variant_cell_derives_from_the_sibling_anchor_plus_the_bound_delta() {
    // Removing the bf16 dev anchors makes bf16 dev/diffvae an unmeasured cell whose only
    // one-axis sibling is bf16 distilled/diffvae — i.e. exactly the DEV-from-distilled variant
    // crossing, with no decoder crossing mixed in. The request runs the deferred and windowed
    // rungs the retained distilled measurement ran (an anchor never prices a request looser
    // than its own regime) and leaves decode untiled, so the decode phase is priced from that
    // anchor's measured intercept and the add has somewhere to land.
    let request = AnchorDeriveRequest {
        transformer_windowed: true,
        deferred_materialization: true,
        ..plain_request(1280, 704, 145)
    };
    let mut doctored = store().clone();
    doctored.anchors.retain(|anchor| {
        !(anchor.model_id == "ltx_2_5"
            && anchor.tier == "bf16"
            && anchor.transformer_variant == Some(Ltx25TransformerVariant::Dev))
    });
    assert!(doctored
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "bf16",
            Ltx25TransformerVariant::Dev,
            Ltx25Decoder::DiffVae
        )
        .is_none());
    let derived = doctored
        .derive_video_phase_peaks_for_cell(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "bf16",
            Some(Ltx25TransformerVariant::Dev),
            Some(Ltx25Decoder::DiffVae),
            request,
        )
        .expect("the unmeasured variant cell derives from its sibling");
    let sibling = doctored
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "bf16",
            Ltx25TransformerVariant::Distilled,
            Ltx25Decoder::DiffVae,
        )
        .expect("the sibling anchor");
    assert_eq!(derived.anchor.id, sibling.id);
    let delta = store()
        .component_delta_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "bf16",
            ComponentDeltaAxis::TransformerVariant,
            "dev",
        )
        .expect("the dev variant delta is bound");
    assert_eq!(derived.delta_bytes, delta.bytes);
    // The delta names the distillation LoRA the DEV recipe materializes on top of the shared
    // transformer, and its bytes are the shipped file's size — not a tuned number.
    assert_eq!(
        delta.files,
        vec!["distilled_lora/ltx-2.5-22b-distilled-lora-450-bf16.safetensors".to_owned()]
    );
    assert!(delta.bytes > 0);
    let raw = sibling
        .derive_video_phase_estimates_raw(request)
        .expect("the sibling prices the request");
    let expect = |estimate: RawPhaseEstimate| {
        if estimate.anchored {
            video_admission_bytes(estimate.bytes + i128::from(delta.bytes), AnchorBackend::Mlx)
                .expect("active estimate")
        } else {
            video_admission_bytes(estimate.bytes, AnchorBackend::Mlx).expect("active estimate")
        }
    };
    assert_eq!(derived.phases.conditioning, expect(raw.conditioning));
    assert_eq!(derived.phases.denoise, expect(raw.denoise));
    assert_eq!(derived.phases.decode, expect(raw.decode));
    // The conservative direction: the delta ADDS — every intercept-priced phase of the derived
    // cell sits strictly above the sibling's own derivation.
    let plain = sibling
        .derive_video_phase_peaks(request)
        .expect("the sibling's own derivation");
    let mut lifted = 0_usize;
    for (anchored, derived_bytes, plain_bytes) in [
        (
            raw.conditioning.anchored,
            derived.phases.conditioning,
            plain.conditioning,
        ),
        (raw.denoise.anchored, derived.phases.denoise, plain.denoise),
        (raw.decode.anchored, derived.phases.decode, plain.decode),
    ] {
        if anchored {
            assert!(derived_bytes > plain_bytes);
            lifted += 1;
        } else {
            assert_eq!(derived_bytes, plain_bytes);
        }
    }
    assert!(lifted > 0, "the crossing must lift at least one phase");
}

/// The mirrored crossing: DISTILLED-from-dev crosses no component, so the fall-through must
/// return the sibling's own estimate unchanged — NOT refuse the cell.
///
/// This is the whole point of representing a zero delta instead of rejecting one. Before the
/// feature-end correction the distilled row carried the dev recipe's LoRA, which made this
/// cell's bound 8.9 GB wider than the evidence entails while the mirrored dev cell was 8.9 GB
/// too narrow — the OOM direction.
#[test]
fn a_zero_crossing_derives_the_sibling_estimate_unchanged_rather_than_refusing() {
    // q8 distilled/diffvae carries no anchor; q8 dev/diffvae is its variant sibling.
    assert!(store()
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q8",
            Ltx25TransformerVariant::Distilled,
            Ltx25Decoder::DiffVae
        )
        .is_none());
    let request = plain_request(1280, 704, 145);
    let derived = store()
        .derive_video_phase_peaks_for_cell(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q8",
            Some(Ltx25TransformerVariant::Distilled),
            Some(Ltx25Decoder::DiffVae),
            request,
        )
        .expect("a zero crossing still derives — a zero delta is priced, not missing");
    let sibling = store()
        .anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q8",
            Ltx25TransformerVariant::Dev,
            Ltx25Decoder::DiffVae,
        )
        .expect("the sibling anchor");
    assert_eq!(derived.anchor.id, sibling.id);
    let delta = store()
        .component_delta_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q8",
            ComponentDeltaAxis::TransformerVariant,
            "distilled",
        )
        .expect("the distilled variant delta is bound — as an explicit zero");
    assert_eq!(delta.bytes, 0);
    assert!(delta.files.is_empty());
    assert_eq!(derived.delta_bytes, 0);
    let plain = sibling
        .derive_video_phase_peaks(request)
        .expect("the sibling's own derivation");
    assert_eq!(derived.phases.conditioning, plain.conditioning);
    assert_eq!(derived.phases.denoise, plain.denoise);
    assert_eq!(derived.phases.decode, plain.decode);
}

/// The fall-through prefers the exact anchor, then the fewest differing axes, then the
/// smallest anchor id — deterministically.
#[test]
fn the_delta_fall_through_prefers_the_exact_anchor_then_the_fewest_axes() {
    let request = plain_request(1280, 704, 145);
    let exact = store()
        .derive_video_phase_peaks_for_cell(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "q8",
            Some(Ltx25TransformerVariant::Dev),
            Some(Ltx25Decoder::DiffVae),
            request,
        )
        .expect("the exact cell derives");
    assert_eq!(exact.delta_bytes, 0);
    assert_eq!(
        exact.anchor.transformer_variant,
        Some(Ltx25TransformerVariant::Dev)
    );
    // Remove the bf16 distilled/diffvae anchor: bf16 distilled/diffvae then has TWO one-axis
    // siblings (dev/diffvae by variant, distilled/conv by decoder) and the smaller anchor id
    // (dev/diffvae) must win, with the variant delta.
    let mut doctored = store().clone();
    doctored.anchors.retain(|anchor| {
        !(anchor.model_id == "ltx_2_5"
            && anchor.tier == "bf16"
            && anchor.transformer_variant == Some(Ltx25TransformerVariant::Distilled)
            && anchor.decoder == Some(Ltx25Decoder::DiffVae))
    });
    let derived = doctored
        .derive_video_phase_peaks_for_cell(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "bf16",
            Some(Ltx25TransformerVariant::Distilled),
            Some(Ltx25Decoder::DiffVae),
            request,
        )
        .expect("the doctored store still derives the cell via a sibling");
    assert_eq!(
        derived.anchor.transformer_variant,
        Some(Ltx25TransformerVariant::Dev)
    );
    assert_eq!(derived.anchor.decoder, Some(Ltx25Decoder::DiffVae));
    // A cell with NO priced sibling axis fails to the caller's floor: strip the deltas and
    // the same lookup returns nothing.
    let mut unpriced = doctored.clone();
    unpriced.component_deltas.clear();
    assert!(
        unpriced
            .derive_video_phase_peaks_for_cell(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "bf16",
                Some(Ltx25TransformerVariant::Distilled),
                Some(Ltx25Decoder::DiffVae),
                request,
            )
            .is_none(),
        "an axis with no bound delta must fall to the floor, never a guessed size"
    );
}

/// Leave-one-out validation against the retained records that DO exist for cells the
/// fall-through can reach: with the bf16 distilled anchors removed, every retained bf16
/// distilled record must be bracketed by the sibling-plus-delta derivation within the delta
/// budget — diffvae via the variant crossing (a priced ZERO under the corrected table, since
/// distilled materializes nothing its dev sibling lacks), conv via that plus the decoder
/// crossing.
///
/// The mirrored direction — dev records derived from distilled siblings, which crosses the
/// 8.9 GB LoRA — is NOT reachable from the retained corpus and is deliberately not faked here:
/// every retained bf16 distilled measurement engages `bounded_transformer_residency` while
/// neither bf16 dev record does, and `derive_video_phase_estimates_raw` refuses to price a
/// non-windowed request from a windowed anchor. The direction is covered instead by
/// `an_unmeasured_variant_cell_derives_from_the_sibling_anchor_plus_the_bound_delta` (byte-for
/// -byte application) and by the engine-agreement tests (which key the table itself).
///
/// SCOPE (sc-22738): the validation is over the seed corpus's OWN topology, so the doctored
/// store keeps only the LTX-2.5 anchors that corpus produced. The sc-22738 rerun anchored
/// dev/conv at bf16 and q4 from its own corpora; left in, distilled/conv would derive from
/// dev/conv (a variant crossing priced zero) instead of dev/diffvae plus the decoder crossing,
/// and the nonzero-delta interrogation below would have nothing to interrogate.
#[test]
fn leave_one_out_sibling_delta_derivations_bracket_the_retained_distilled_records() {
    let mut doctored = store().clone();
    doctored.anchors.retain(|anchor| {
        anchor.model_id != "ltx_2_5"
            || (anchor.source.path == LTX25_CORPUS_PATH
                && anchor.transformer_variant != Some(Ltx25TransformerVariant::Distilled))
    });
    let corpus = retained_corpus();
    let distilled: Vec<_> = corpus
        .iter()
        .filter(|record| record.transformer_variant == Ltx25TransformerVariant::Distilled)
        .collect();
    assert!(
        distilled.len() >= 4,
        "the retained corpus must carry distilled records to validate against, got {}",
        distilled.len()
    );
    assert!(
        distilled
            .iter()
            .any(|record| record.decoder == Ltx25Decoder::Conv),
        "the distilled/conv records are part of the validation set"
    );
    let mut crossed_a_nonzero_delta = false;
    for record in distilled {
        let derived = doctored
            .derive_video_phase_peaks_for_cell(
                "ltx_2_5",
                AnchorBackend::Mlx,
                &record.tier,
                Some(record.transformer_variant),
                Some(record.decoder),
                AnchorDeriveRequest {
                    width: record.width,
                    height: record.height,
                    frames: record.frames,
                    decode_tiled: record.decode_tiled,
                    transformer_windowed: record.transformer_windowed,
                    deferred_materialization: record.deferred,
                },
            )
            .unwrap_or_else(|| panic!("{} derives via a sibling", record.id));
        assert_ne!(
            derived.anchor.transformer_variant,
            Some(Ltx25TransformerVariant::Distilled),
            "{}: the leave-one-out derivation must come from the surviving variant",
            record.id
        );
        // The delta the fall-through applied must be EXACTLY what the shipped table prices for
        // this crossing — never an invented add, never a dropped one. Zero is a real answer
        // here, not a skipped axis: the distilled target crosses no component its dev sibling
        // lacks, so the variant term contributes nothing and only a decoder crossing lifts it.
        let mut expected = 0_u64;
        if derived.anchor.transformer_variant != Some(record.transformer_variant) {
            expected += doctored
                .component_delta_for(
                    "ltx_2_5",
                    AnchorBackend::Mlx,
                    &record.tier,
                    ComponentDeltaAxis::TransformerVariant,
                    transformer_variant_key(record.transformer_variant),
                )
                .unwrap_or_else(|| panic!("{}: the variant crossing is priced", record.id))
                .bytes;
        }
        if derived.anchor.decoder != Some(record.decoder) {
            expected += doctored
                .component_delta_for(
                    "ltx_2_5",
                    AnchorBackend::Mlx,
                    &record.tier,
                    ComponentDeltaAxis::Decoder,
                    decoder_key(record.decoder),
                )
                .unwrap_or_else(|| panic!("{}: the decoder crossing is priced", record.id))
                .bytes;
        }
        assert_eq!(
            derived.delta_bytes, expected,
            "{}: the applied delta must equal the shipped table's price for the crossing",
            record.id
        );
        crossed_a_nonzero_delta |= derived.delta_bytes > 0;
        // Where the derivation reuses a measured intercept, a NONZERO crossed-axis delta must
        // be APPLIED to it — a zeroed delta reds here — and a ZERO one must leave it exactly
        // alone. Architecture-bounded phases take no add (the bounds are cell-blind by their
        // documented provenance), so the check names the intercept-priced phases of this
        // record's regime.
        let sibling_raw = derived
            .anchor
            .derive_video_phase_estimates_raw(AnchorDeriveRequest {
                width: record.width,
                height: record.height,
                frames: record.frames,
                decode_tiled: record.decode_tiled,
                transformer_windowed: record.transformer_windowed,
                deferred_materialization: record.deferred,
            })
            .expect("the sibling prices the request");
        for (phase, anchored, sibling_raw, derived_bytes) in [
            (
                "conditioning",
                sibling_raw.conditioning.anchored,
                sibling_raw.conditioning.bytes,
                derived.phases.conditioning,
            ),
            (
                "denoise",
                sibling_raw.denoise.anchored,
                sibling_raw.denoise.bytes,
                derived.phases.denoise,
            ),
            (
                "decode",
                sibling_raw.decode.anchored,
                sibling_raw.decode.bytes,
                derived.phases.decode,
            ),
        ] {
            if anchored {
                let undelta = video_admission_bytes(sibling_raw, AnchorBackend::Mlx)
                    .expect("active estimate");
                if derived.delta_bytes > 0 {
                    assert!(
                        derived_bytes > undelta,
                        "{} {phase}: the crossed-axis delta must be applied to the sibling's \
                             measured intercept",
                        record.id
                    );
                } else {
                    assert_eq!(
                        derived_bytes, undelta,
                        "{} {phase}: a zero crossing must add nothing to the sibling's \
                             measured intercept",
                        record.id
                    );
                }
            }
        }
        for (phase, derived_bytes, measured_bytes) in [
            (
                "conditioning",
                derived.phases.conditioning,
                record.conditioning_active,
            ),
            ("denoise", derived.phases.denoise, record.denoise_active),
            ("decode", derived.phases.decode, record.decode_active),
        ] {
            assert!(
                derived_bytes >= measured_bytes,
                "{} {phase}: sibling+delta derived {derived_bytes} under measured \
                     {measured_bytes}",
                record.id
            );
        }
        let upper = derived.phases.peak_bytes();
        assert!(
            upper
                >= record
                    .conditioning_active
                    .max(record.denoise_active)
                    .max(record.decode_active),
            "{}: derived bound {upper} under the measured envelope {}",
            record.id,
            record
                .conditioning_active
                .max(record.denoise_active)
                .max(record.decode_active)
        );
        let cap = ((record
            .conditioning_active
            .max(record.denoise_active)
            .max(record.decode_active) as f64)
            * (1.0 + ANCHOR_DELTA_VALIDATION_TIGHTNESS_BUDGET))
            .ceil() as u64;
        assert!(
            upper <= cap,
            "{}: derived bound {upper} exceeds the delta tightness cap {cap} over envelope {} \
                 (overshoot {:.4})",
            record.id,
            record
                .conditioning_active
                .max(record.denoise_active)
                .max(record.decode_active),
            (upper as f64)
                / (record
                    .conditioning_active
                    .max(record.denoise_active)
                    .max(record.decode_active) as f64)
                - 1.0
        );
    }
    // The budget is a claim about DELTA-priced bounds, so the leave-one-out set has to carry
    // at least one crossing the table prices above zero. Without this the suite could pass on
    // zero crossings alone and never interrogate the add at all.
    assert!(
        crossed_a_nonzero_delta,
        "the leave-one-out set must include a crossing the delta table prices above zero"
    );
}

/// The delta budget is DERIVED, not chosen: it is the smallest round figure that clears every
/// leave-one-out overshoot the corrected table produces, and it must stay meaningfully tight —
/// a budget far above the binding case would stop refusing runaway estimates, which is the
/// only job it has. Recomputed at the feature-end fix round, after the variant deltas were
/// re-keyed onto the engine's real materialization.
#[test]
fn the_delta_tightness_budget_is_the_binding_leave_one_out_overshoot_rounded_up() {
    let mut doctored = store().clone();
    doctored.anchors.retain(|anchor| {
        !(anchor.model_id == "ltx_2_5"
            && anchor.transformer_variant == Some(Ltx25TransformerVariant::Distilled))
    });
    let mut worst = 0.0_f64;
    let mut worst_id = String::new();
    for record in retained_corpus()
        .iter()
        .filter(|record| record.transformer_variant == Ltx25TransformerVariant::Distilled)
    {
        let Some(derived) = doctored.derive_video_phase_peaks_for_cell(
            "ltx_2_5",
            AnchorBackend::Mlx,
            &record.tier,
            Some(record.transformer_variant),
            Some(record.decoder),
            AnchorDeriveRequest {
                width: record.width,
                height: record.height,
                frames: record.frames,
                decode_tiled: record.decode_tiled,
                transformer_windowed: record.transformer_windowed,
                deferred_materialization: record.deferred,
            },
        ) else {
            continue;
        };
        let overshoot = (derived.phases.peak_bytes() as f64)
            / (record
                .conditioning_active
                .max(record.denoise_active)
                .max(record.decode_active) as f64)
            - 1.0;
        if overshoot > worst {
            worst = overshoot;
            worst_id = record.id.clone();
        }
    }
    assert!(
        worst > 0.0,
        "the leave-one-out set must produce a measurable overshoot to size the budget from"
    );
    assert!(
        ANCHOR_DELTA_VALIDATION_TIGHTNESS_BUDGET >= worst,
        "the budget {ANCHOR_DELTA_VALIDATION_TIGHTNESS_BUDGET} is under the binding \
             leave-one-out overshoot {worst:.4} ({worst_id})"
    );
    // Tight: no more than one percentage point of round-up slack above the binding case. This
    // is what stops the budget from quietly re-widening the next time a delta moves.
    assert!(
        ANCHOR_DELTA_VALIDATION_TIGHTNESS_BUDGET <= worst + 0.01,
        "the budget {ANCHOR_DELTA_VALIDATION_TIGHTNESS_BUDGET} is more than a point above the \
             binding leave-one-out overshoot {worst:.4} ({worst_id}) — re-derive it"
    );
}

/// The shipped variant-delta rows are keyed on what each variant MATERIALIZES.
///
/// This is the test the first cut of the table did not have, and its absence is why the
/// mapping shipped inverted: the delta rows were keyed on the variant's NAME (the "distilled"
/// row got the distillation LoRA) rather than on the engine's load spec, which adds that LoRA
/// for DEV. Re-inverting [`LTX25_VARIANT_DELTA_COMPONENTS`] or the extractor's `variantRows`
/// reds here.
#[test]
fn ltx25_variant_delta_matches_the_engine_materialization() {
    let expected: BTreeMap<&str, BTreeSet<&str>> = LTX25_VARIANT_DELTA_COMPONENTS
        .iter()
        .map(|(variant, components)| (*variant, components.iter().copied().collect()))
        .collect();
    let mut seen: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut tiers: BTreeSet<&str> = BTreeSet::new();
    for delta in store()
        .component_deltas
        .iter()
        .filter(|delta| delta.model_id == "ltx_2_5")
        .filter(|delta| delta.axis == ComponentDeltaAxis::TransformerVariant)
    {
        tiers.insert(delta.tier.as_str());
        // The component a row crosses is its file's top-level directory in the bundle.
        let components: BTreeSet<&str> = delta
            .files
            .iter()
            .map(|file| {
                file.split('/')
                    .next()
                    .expect("an inventory path has a leading segment")
            })
            .collect();
        let previous = seen.insert(delta.to.as_str(), components.clone());
        if let Some(previous) = previous {
            assert_eq!(
                previous, components,
                "{}: every tier must cross the same components for a variant",
                delta.id
            );
        }
    }
    assert_eq!(
        seen, expected,
        "the shipped variant deltas must cross exactly what the engine materializes"
    );
    // Every tier carries a row for BOTH variants — including the zero one, so the fall-through
    // can cross the variant axis in either direction rather than refusing half of it.
    assert!(!tiers.is_empty(), "the store must price the variant axis");
    for tier in &tiers {
        for (variant, _) in LTX25_VARIANT_DELTA_COMPONENTS {
            assert!(
                store()
                    .component_delta_for(
                        "ltx_2_5",
                        AnchorBackend::Mlx,
                        tier,
                        ComponentDeltaAxis::TransformerVariant,
                        variant,
                    )
                    .is_some(),
                "{tier}/{variant}: a missing row means \"unpriced axis\", which is a \
                     different claim from a zero crossing"
            );
        }
    }
}

/// The cross-check behind [`LTX25_VARIANT_DELTA_COMPONENTS`]: the mirror must still describe
/// what the two engine sources actually do. `sceneworks-core` cannot depend on the worker or
/// the capture arm, so the agreement is asserted against their SOURCE TEXT — an engine edit
/// that moves the adapter to the other variant, or makes the enhancer variant-conditional,
/// reds here instead of silently invalidating every variant delta.
#[test]
fn ltx25_variant_delta_components_mirror_the_engine_source() {
    const CAPTURE_ARM: &str =
        include_str!("../../../sceneworks-memory-adapter/src/bin/mlx_ltx25.rs");
    const WORKER: &str = include_str!("../../../sceneworks-worker/src/video_jobs/ltx.rs");

    // 1. The capture arm's adapter constant names the component the dev row prices.
    let adapter_decl = CAPTURE_ARM
        .lines()
        .find(|line| line.trim_start().starts_with("const DEV_ADAPTER: &str ="))
        .expect("the capture arm declares DEV_ADAPTER");
    let adapter_path = adapter_decl
        .split('"')
        .nth(1)
        .expect("DEV_ADAPTER is a string literal");
    let adapter_component = adapter_path
        .split('/')
        .next()
        .expect("DEV_ADAPTER is a bundle-relative path");
    let dev_components: BTreeSet<&str> = LTX25_VARIANT_DELTA_COMPONENTS
        .iter()
        .find(|(variant, _)| *variant == "dev")
        .map(|(_, components)| components.iter().copied().collect())
        .expect("the mirror describes dev");
    assert_eq!(
        dev_components,
        BTreeSet::from([adapter_component]),
        "the dev row must cross exactly the component DEV_ADAPTER names"
    );

    // 2. That adapter is pushed under the DEV arm — the nearest variant test above the push.
    let push = CAPTURE_ARM
        .find("spec.adapters.push(")
        .expect("the capture arm pushes the adapter");
    let gate = CAPTURE_ARM[..push]
        .rfind("if target.variant == TransformerVariant::")
        .expect("the push is gated on the transformer variant");
    assert!(
        CAPTURE_ARM[gate..push].contains("TransformerVariant::Dev"),
        "the capture arm must push DEV_ADAPTER under the Dev arm, not the Distilled one"
    );

    // 3. The enhancer is inserted BEFORE any variant test, i.e. for both variants — which is
    //    why it cancels out of every variant delta and appears in neither row.
    let spec_fn = CAPTURE_ARM
        .find("fn configured_spec(")
        .expect("the capture arm builds the load spec");
    let enhancer = CAPTURE_ARM[spec_fn..]
        .find("\"enhancer\".to_owned()")
        .expect("configured_spec inserts the enhancer component");
    let first_variant_test = CAPTURE_ARM[spec_fn..]
        .find("if target.variant == TransformerVariant::")
        .expect("configured_spec tests the variant");
    assert!(
        enhancer < first_variant_test,
        "the enhancer must be inserted unconditionally; a variant-conditional enhancer would \
             belong in a variant delta row"
    );
    for (_, components) in LTX25_VARIANT_DELTA_COMPONENTS {
        assert!(
            !components.contains(&"enhancer"),
            "the enhancer is resident in both variants and cancels out of the deltas"
        );
    }

    // 4. The production worker agrees: LTX-2.5 Distilled resolves NO adapter.
    let refusal = WORKER
        .find("if ltx25_variant == Some(Ltx25TransformerVariant::Distilled) {")
        .expect("the worker refuses the distill adapter for LTX-2.5 distilled");
    assert!(
        WORKER[refusal..refusal + 200].contains("return Ok(None);"),
        "the worker's distilled arm must return no adapter"
    );
    let distilled_components: BTreeSet<&str> = LTX25_VARIANT_DELTA_COMPONENTS
        .iter()
        .find(|(variant, _)| *variant == "distilled")
        .map(|(_, components)| components.iter().copied().collect())
        .expect("the mirror describes distilled");
    assert!(
        distilled_components.is_empty(),
        "distilled materializes nothing its dev sibling lacks, so its row crosses nothing"
    );
}

/// The delta rows are BOUND to the shipped inventory: a zeroed, inflated, re-pointed,
/// unknown-file or duplicated row fails the load.
#[test]
fn a_doctored_component_delta_is_rejected() {
    let raw = PACKAGED_MEMORY_ANCHORS;
    let base: serde_json::Value = serde_json::from_str(raw).expect("store parses");
    assert!(
        base["componentDeltas"]
            .as_array()
            .is_some_and(|rows| !rows.is_empty()),
        "the packaged store must carry component deltas for this harness to interrogate"
    );

    let mut zeroed = base.clone();
    zeroed["componentDeltas"][0]["bytes"] = serde_json::json!(0);
    let error =
        load_memory_anchors(&zeroed.to_string()).expect_err("a zeroed delta must be rejected");
    assert!(
        error.contains("must equal the shipped file sizes"),
        "{error}"
    );

    let mut inflated = base.clone();
    let bytes = inflated["componentDeltas"][0]["bytes"]
        .as_u64()
        .expect("bytes");
    inflated["componentDeltas"][0]["bytes"] = serde_json::json!(bytes + 1);
    let error =
        load_memory_anchors(&inflated.to_string()).expect_err("an inflated delta must be rejected");
    assert!(
        error.contains("must equal the shipped file sizes"),
        "{error}"
    );

    let mut repointed = base.clone();
    repointed["componentDeltas"][0]["source"]["path"] = serde_json::json!("docs/nowhere.json");
    let error = load_memory_anchors(&repointed.to_string())
        .expect_err("a foreign inventory citation must be rejected");
    assert!(
        error.contains("not a compiled weights file inventory"),
        "{error}"
    );

    let mut drifted = base.clone();
    drifted["componentDeltas"][0]["source"]["sha256"] = serde_json::json!("0".repeat(64));
    let error = load_memory_anchors(&drifted.to_string())
        .expect_err("a drifted inventory digest must be rejected");
    assert!(error.contains("inventory digest mismatch"), "{error}");

    let mut unknown = base.clone();
    unknown["componentDeltas"][0]["files"] = serde_json::json!(["not/a/file.safetensors"]);
    let error =
        load_memory_anchors(&unknown.to_string()).expect_err("an unknown file must be rejected");
    assert!(error.contains("does not size"), "{error}");

    // The zero row is representable but not a loophole: dropping a real row's files while
    // keeping its bytes must fail, and so must giving the honest zero row a byte count.
    let mut unfiled = base.clone();
    unfiled["componentDeltas"][0]["files"] = serde_json::json!([]);
    let error = load_memory_anchors(&unfiled.to_string())
        .expect_err("bytes without files must be rejected");
    assert!(error.contains("names no files"), "{error}");

    let zero_index = base["componentDeltas"]
        .as_array()
        .expect("rows")
        .iter()
        .position(|row| {
            row["files"]
                .as_array()
                .is_some_and(|files| files.is_empty())
        })
        .expect("the packaged store carries an explicit zero crossing");
    assert_eq!(
        base["componentDeltas"][zero_index]["bytes"].as_u64(),
        Some(0)
    );
    let mut paid_zero = base.clone();
    paid_zero["componentDeltas"][zero_index]["bytes"] = serde_json::json!(1);
    let error = load_memory_anchors(&paid_zero.to_string())
        .expect_err("a priced zero crossing must be rejected");
    assert!(error.contains("names no files"), "{error}");

    let mut duplicated = base.clone();
    let clone = duplicated["componentDeltas"][0].clone();
    duplicated["componentDeltas"]
        .as_array_mut()
        .expect("rows")
        .push(clone);
    let error = load_memory_anchors(&duplicated.to_string())
        .expect_err("a duplicated delta cell must be rejected");
    assert!(error.contains("duplicate component delta"), "{error}");
}

/// The new anchor fields are bound, not stored-and-trusted: a doctored phase allocator level,
/// a dropped decomposition the record reports, and an empty underived reason all fail.
#[test]
fn a_doctored_phase_allocator_or_empty_underived_reason_is_rejected() {
    let index = store()
        .anchors
        .iter()
        .position(|anchor| anchor.phase_allocator_envelope_bytes.is_some())
        .expect("an anchor with phase allocator levels");
    let mut doctored: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("store parses");
    let level = doctored["anchors"][index]["phaseAllocatorEnvelopeBytes"]["denoise"]
        .as_u64()
        .expect("denoise allocator level");
    doctored["anchors"][index]["phaseAllocatorEnvelopeBytes"]["denoise"] =
        serde_json::json!(level + 1);
    let error = load_memory_anchors(&doctored.to_string())
        .expect_err("a doctored allocator level must be rejected");
    assert!(
        error.contains("phase allocator envelopes disagree"),
        "{error}"
    );

    let mut dropped: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("store parses");
    dropped["anchors"][index]
        .as_object_mut()
        .expect("anchor object")
        .remove("phaseAllocatorEnvelopeBytes");
    let error = load_memory_anchors(&dropped.to_string())
        .expect_err("dropping a reported decomposition must be rejected");
    assert!(
        error.contains("phase allocator envelopes disagree"),
        "{error}"
    );

    let mut unexplained: serde_json::Value =
        serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("store parses");
    unexplained["anchors"][index]["underivedReason"] = serde_json::json!("   ");
    let error = load_memory_anchors(&unexplained.to_string())
        .expect_err("an empty underived reason must be rejected");
    assert!(error.contains("without a reason"), "{error}");
}
