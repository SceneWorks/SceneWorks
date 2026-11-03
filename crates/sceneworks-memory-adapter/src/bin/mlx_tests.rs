use super::*;

/// sc-22738. EVERY SCENARIO LIST IN THIS BINARY DECLARES AN `overlay` ENTRY.
///
/// `protocol::settle_plain_overlay_scenario` REPLACES an existing `overlay` scenario and
/// refuses — "provider fragment is missing the required overlay scenario" — a fragment that has
/// none. The refusal fires while the receipt is being built, i.e. AFTER a completed render, so
/// the whole cost of a capture is paid before the omission is visible: that is exactly how the
/// Mage arm cost eighteen 11-38 s renders. Nothing else in the binary asked this question
/// statically, so a nineteenth arm could repeat it.
///
/// Every scenario array literal in this file must therefore carry either the placeholder or an
/// `overlay_scenario` binding (the shape `run_flux_one` and `run_qwen_edit_provider` use, where
/// the verdict is computed above the literal). The one dynamically-assembled list — the LTX
/// arm's, which assigns a `scenarios` binding — is not a literal and is not scanned; it builds
/// its entries through the same helper.
///
/// MUTATION: deleting the `overlay` entry from any scenario literal, including
/// [`mage_scenarios`], reds this with that literal's line number.
#[test]
fn every_scenario_literal_declares_the_overlay_entry_the_protocol_settles() {
    let source = crate::mlx_source_for_tests();
    let mut scanned = 0_usize;
    let mut cursor = 0_usize;
    while let Some(offset) = source[cursor..].find("\"scenarios\": [") {
        let start = cursor + offset + "\"scenarios\": [".len();
        let mut depth = 1_usize;
        let mut index = start;
        for byte in source[start..].bytes() {
            match byte {
                b'[' => depth += 1,
                b']' => depth -= 1,
                _ => {}
            }
            index += 1;
            if depth == 0 {
                break;
            }
        }
        assert!(depth == 0, "an unbalanced scenario literal at byte {start}");
        let block = &source[start..index];
        let line = source[..start].lines().count();
        assert!(
            block.contains("\"name\": \"overlay\"") || block.contains("overlay_scenario"),
            "the scenario literal at line {line} declares no overlay entry, so \
                 settle_plain_overlay_scenario will refuse its fragment after the render"
        );
        scanned += 1;
        cursor = index;
    }
    // Non-vacuous: this binary carries many arms, and a scan that matched none would be green.
    assert!(
        scanned >= 15,
        "only {scanned} scenario literals were scanned; the guard has lost its subject"
    );
    // The one list that is NOT a literal here is the Mage arm's, hoisted into
    // [`mage_scenarios`] so the settler can be run over it — assert it by calling it, rather
    // than leaving the arm this guard exists for outside the guard's reach.
    assert!(
        mage_scenarios(0)
            .as_array()
            .expect("the Mage scenario list is an array")
            .iter()
            .any(|scenario| scenario["name"] == "overlay"),
        "mage_scenarios declares no overlay entry"
    );
}

#[test]
fn scoped_generation_preserves_generator_and_finish_failures() {
    let mut observed = Some(ScopedGenerationFailureKind::Error);
    let error = settle_scoped_generation(
        Err(mlx_gen::gen_core::Error::Msg(
            "generator exploded".to_owned(),
        )),
        Err(mlx_gen::gen_core::Error::Msg("finish exploded".to_owned())),
        &mut observed,
    )
    .expect_err("neither terminal failure may be discarded");
    assert_eq!(
            error,
            "generate calibrated request: generator exploded; finish calibrated request: finish exploded"
        );
    assert_eq!(observed, Some(ScopedGenerationFailureKind::Finish));

    let mut control = Some(ScopedGenerationFailureKind::Error);
    let error = settle_scoped_generation(
        Err(mlx_gen::gen_core::Error::Msg(
            "generator exploded".to_owned(),
        )),
        Ok(()),
        &mut control,
    )
    .expect_err("the generator failure remains terminal when finish succeeds");
    assert_eq!(error, "generator exploded");
    assert_eq!(control, Some(ScopedGenerationFailureKind::Error));
}

#[test]
fn wired_limit_prefers_explicit_bytes() {
    assert_eq!(
        resolve_wired_limit(Some("123"), Some("456"), Some("789"), 999).unwrap(),
        WiredLimit {
            bytes: 123,
            source: "SCENEWORKS_MLX_WIRED_LIMIT_BYTES",
        }
    );
}

#[test]
fn wired_limit_converts_iogpu_megabytes() {
    assert_eq!(
        resolve_wired_limit(None, Some("57344"), Some("789"), 999).unwrap(),
        WiredLimit {
            bytes: 57344 * 1024 * 1024,
            source: "iogpu.wired_limit_mb",
        }
    );
}

#[test]
fn wired_limit_rejects_iogpu_byte_overflow() {
    let megabytes = u64::MAX.to_string();
    assert!(resolve_wired_limit(None, Some(&megabytes), None, 1_000)
        .unwrap_err()
        .contains("iogpu.wired_limit_mb overflows bytes"));
}

#[test]
fn wired_limit_uses_kernel_bytes_when_iogpu_is_unset() {
    assert_eq!(
        resolve_wired_limit(None, Some("0"), Some("789"), 999).unwrap(),
        WiredLimit {
            bytes: 789,
            source: "kern.memorystatus_wired_mem_limit",
        }
    );
}

#[test]
fn wired_limit_derives_default_mlx_ceiling_without_host_override() {
    assert_eq!(
        resolve_wired_limit(None, Some("0"), None, 1_000).unwrap(),
        WiredLimit {
            bytes: 666,
            source: "mlx_default_memory_limit/1.5",
        }
    );
}

#[test]
fn wired_limit_rejects_invalid_explicit_override() {
    assert!(resolve_wired_limit(Some("not-a-number"), None, None, 1_000)
        .unwrap_err()
        .contains("SCENEWORKS_MLX_WIRED_LIMIT_BYTES"));
}

#[test]
fn wired_limit_rejects_zero_explicit_override() {
    assert!(
        resolve_wired_limit(Some("0"), Some("456"), Some("789"), 1_000)
            .unwrap_err()
            .contains("SCENEWORKS_MLX_WIRED_LIMIT_BYTES must be greater than zero")
    );
}

#[test]
fn wired_limit_rejects_zero_everywhere() {
    assert!(resolve_wired_limit(None, Some("0"), Some("0"), 0)
        .unwrap_err()
        .contains("cannot resolve a nonzero wired ceiling"));
}

#[test]
fn overall_memory_and_prediction_cover_every_componentwise_phase_peak() {
    let conditioning = PhaseMemory {
        active: 8,
        cache: 1,
    };
    let denoise = PhaseMemory {
        active: 16,
        cache: 15,
    };
    let decode = PhaseMemory {
        active: 19,
        cache: 0,
    };
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    assert_eq!(
        overall,
        PhaseMemory {
            active: 19,
            cache: 15,
        }
    );
    assert_eq!(overall.allocator_bytes(), 34);
    assert!(predicted_ceiling(overall.allocator_bytes()) >= overall.allocator_bytes());
}

#[test]
fn prediction_basis_keeps_image_admission_conservative_and_video_resident() {
    let phase = PhaseMemory {
        active: 16 * MIB,
        cache: 15 * MIB,
    };
    assert_eq!(
        IMAGE_PREDICTED_PEAK_BASIS,
        PredictedPeakBasis::HistoricalImageAllocatorBound
    );
    assert_eq!(
        VIDEO_PREDICTED_PEAK_BASIS,
        PredictedPeakBasis::ResidentActive
    );
    assert_eq!(
        predicted_phase_ceiling(phase, IMAGE_PREDICTED_PEAK_BASIS),
        predicted_ceiling(31 * MIB)
    );
    assert_eq!(
        predicted_phase_ceiling(phase, VIDEO_PREDICTED_PEAK_BASIS),
        predicted_ceiling(16 * MIB)
    );
}

/// sc-22735 fix pass. Every `crates/<crate>/src/<path>.rs` in THIS workspace that this file
/// cites in prose must be a file that exists.
///
/// A doc comment citing `pinned_engine_geometry.rs` under `sceneworks-core` shipped for a
/// module that lives in `sceneworks-worker`, and nothing caught it: a wrong path reads exactly
/// like a right one, and the reader who follows it finds nothing and re-derives the fact. The
/// paths are read out of this file's own source and resolved against the workspace root, so a
/// module that is renamed or moved reds here rather than leaving a citation pointing at air.
///
/// A citation is checked only when its crate segment names a REAL directory under `crates/`.
/// That is what separates the three things this scan must not confuse:
///
///  * a citation into THIS workspace (`crates/sceneworks-worker/src/…`) — checked, and the
///    `sceneworks-core` typo above is caught precisely because that crate does exist, so the
///    wrong-module citation is resolved and found missing rather than waved through;
///  * a cross-repo citation into the pinned inference tree (`crates/media/mlx-gen/…`) — skipped,
///    since that file is not in this checkout and its absence here says nothing;
///  * a prose placeholder (`crates/<name>/src/<...>.rs`) — skipped, being no path at all.
#[test]
fn every_cited_workspace_source_path_exists() {
    let source = crate::mlx_source_for_tests();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("the workspace root is two levels above this crate");
    // The crates this workspace actually contains, read off disk rather than listed, so a new
    // or renamed crate is covered without editing this test.
    let workspace_crates: std::collections::BTreeSet<String> =
        std::fs::read_dir(root.join("crates"))
            .expect("the workspace crates directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                entry
                    .file_type()
                    .ok()?
                    .is_dir()
                    .then(|| entry.file_name().to_string_lossy().into_owned())
            })
            .collect();
    assert!(
        workspace_crates.contains("sceneworks-worker"),
        "the crate scan found no sceneworks-worker; it is reading the wrong directory"
    );

    // `crates/<name>/src/<...>.rs`, wherever it appears — backticked in prose, inside a
    // `source:` string, or in a plain comment.
    let cited: std::collections::BTreeSet<&str> = source
        .match_indices("crates/")
        .filter_map(|(start, _)| {
            let rest = &source[start..];
            let end = rest.find(".rs")? + 3;
            let path = &rest[..end];
            // Stop at anything that cannot be part of a path: a citation may carry a line-range
            // suffix or trailing prose, but the path itself is one unbroken token.
            (!path.contains(char::is_whitespace)
                && !path.contains('`')
                && !path.contains('#')
                && path.contains("/src/"))
            .then_some(path)
        })
        // Only the citations that name a crate of THIS workspace; see the doc above.
        .filter(|path| {
            path.split('/')
                .nth(1)
                .is_some_and(|krate| workspace_crates.contains(krate))
        })
        .collect();
    assert!(
        cited.len() > 5,
        "the citation scan found almost nothing ({cited:?}); it no longer guards anything"
    );
    let missing: Vec<&str> = cited
        .iter()
        .copied()
        .filter(|path| !root.join(path).exists())
        .collect();
    assert!(
        missing.is_empty(),
        "these cited workspace source files do not exist: {missing:?}"
    );
}

#[test]
fn every_receipt_builder_is_bound_to_its_lane_prediction_wrapper() {
    let source = crate::mlx_source_for_tests();
    // A SHAPE claim, not a count: EVERY `predicted_peaks` binding in this file must come from
    // the image policy wrapper. Asserting a fixed number instead made adding an image arm
    // (sc-22728 added the Qwen edit one) fail this test for the right reason with the wrong
    // message — "there are seven, not six" — while a new arm that bypassed the wrapper entirely
    // would have kept the count and gone unnoticed.
    let receipts: Vec<&str> = source
        .lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("let predicted_peaks = "))
        .collect();
    assert!(
        !receipts.is_empty(),
        "the image receipt builders are gone; this test no longer guards anything"
    );
    for line in &receipts {
        assert!(
            line.starts_with("let predicted_peaks = image_predicted_peak_bytes(")
                || line.starts_with("let predicted_peaks = video_predicted_peak_bytes("),
            "an MLX receipt builder bypasses both lane policy wrappers: {line}"
        );
    }
    assert!(
        receipts
            .iter()
            .any(|line| line.starts_with("let predicted_peaks = image_predicted_peak_bytes(")),
        "no image receipt builder is left; this test no longer guards the image lane"
    );
    assert_eq!(
        source
            .lines()
            .filter(|line| {
                line.trim_start()
                    == "video_predicted_peak_bytes(conditioning, denoise, decode).json()"
            })
            .count(),
        1,
        "the LTX receipt builder must use the video policy wrapper"
    );
}

/// sc-19115. This is the load-bearing decision test, not merely a count of records with cache.
/// It evaluates the same host-reserve currency production uses over every committed MLX image
/// cell and proves that changing the image basis would only ever LOOSEN shipped admission —
/// never tighten it — which is the whole reason the image basis stays where it is.
///
/// The verdict is asserted; the corpus size is not. See the shape assertions at the end.
#[test]
fn resident_peak_counterfactual_only_loosens_shipped_image_admission() {
    use sceneworks_core::memory_calibration::{Backend, EvidenceBundle, RequiredNullable};

    fn phase(phase: &sceneworks_core::memory_calibration::Phase) -> PhaseMemory {
        PhaseMemory {
            active: phase.active_bytes,
            cache: phase.reclaimable_bytes,
        }
    }

    let evidence: EvidenceBundle = serde_json::from_str(include_str!(
        "../../../../docs/generated/memory-calibration-evidence.json"
    ))
    .expect("parse committed memory evidence");
    let host_bytes = [24_u64, 32, 48, 64, 96, 128].map(|gib| gib * 1024 * 1024 * 1024);
    let mut image_records = 0_usize;
    let mut changed_records = 0_usize;
    let mut flipped_cells = 0_usize;
    let mut flips_by_provider = std::collections::BTreeMap::<&str, usize>::new();

    for record in evidence
        .records
        .iter()
        .filter(|record| record.backend == Backend::Mlx && record.target.mode == "text_to_image")
    {
        let observed = match &record.observed_memory {
            RequiredNullable::Value(observed) => observed.full().expect("full image telemetry"),
            RequiredNullable::Null => panic!("{} has null image telemetry", record.id),
        };
        image_records += 1;
        let historical = image_predicted_peak_bytes(
            phase(&observed.conditioning),
            phase(&observed.denoise),
            phase(&observed.decode),
        )
        .overall;
        let resident = video_predicted_peak_bytes(
            phase(&observed.conditioning),
            phase(&observed.denoise),
            phase(&observed.decode),
        )
        .overall;
        let shipped = match &record.predicted_peak_bytes {
            RequiredNullable::Value(predicted) => predicted.overall(),
            RequiredNullable::Null => panic!("{} has null prediction", record.id),
        };
        assert_eq!(
            shipped, historical,
            "{} no longer carries the historical image prediction",
            record.id
        );
        if historical != resident {
            // The DIRECTION, per record, stated where the pinned totals used to be: the
            // resident basis drops the reclaimable term, so wherever the two bases disagree
            // the resident one must predict LESS. That is what makes every admission flip
            // below a loosening rather than a coincidence of this corpus, and unlike a count
            // it holds at any corpus size.
            assert!(
                resident < historical,
                "{}: the resident basis predicted {resident} against the historical \
                     {historical} — a resident basis that predicts MORE would tighten admission, \
                     which is the opposite of the sc-19115 finding",
                record.id
            );
            changed_records += 1;
        }

        let historical_envelope = record
            .mlx_admission_envelope()
            .unwrap_or_else(|| panic!("{} has no production MLX envelope", record.id));
        let mut counterfactual = record.clone();
        match &mut counterfactual.predicted_peak_bytes {
            RequiredNullable::Value(predicted) => {
                predicted.full_mut().expect("full image prediction").overall = resident;
            }
            RequiredNullable::Null => panic!("{} has null prediction", record.id),
        }
        let resident_envelope = counterfactual
            .mlx_admission_envelope()
            .expect("counterfactual production MLX envelope");
        for host in host_bytes {
            let historical_fits = historical_envelope.fits_scaled_host_bytes(host);
            let resident_fits = resident_envelope.fits_scaled_host_bytes(host);
            if historical_fits != resident_fits {
                assert!(
                    !historical_fits && resident_fits,
                    "the resident counterfactual must only loosen admission"
                );
                flipped_cells += 1;
                *flips_by_provider
                    .entry(record.target.provider.as_str())
                    .or_default() += 1;
            }
        }
    }

    // The MLX `text_to_image` corpus is NOT frozen. Every calibration campaign and every main
    // sync adds coordinates, and the counterfactual's derived totals move with it: 69 -> 74 at
    // the sc-18304 sync, 74 -> 93 at the 2026-08-19 epic-17137 sync that brought epic 18803's
    // captures across. Pinning those totals recorded only which campaign ran last and went red
    // for legitimate corpus growth, so this states the SHAPE of the counterfactual instead —
    // the same treatment the sibling calibration gates got (see `memory_calibration.rs`, where
    // the re-capture set is held to its shape while only frozen history stays pinned).
    //
    // The load-bearing claim is the only-loosens assertion in the loop above. These keep it
    // from passing vacuously on an empty or half-read corpus, and hold the derived tallies to
    // the identities that must be true at ANY corpus size.
    assert!(
        image_records > 0,
        "no MLX text_to_image record was read — the counterfactual is vacuous"
    );
    assert!(
        changed_records > 0 && changed_records <= image_records,
        "{changed_records} of {image_records} records changed basis: the resident basis must \
             move some image predictions, and can never move more than the corpus holds"
    );
    assert!(
        flipped_cells > 0,
        "the resident basis flipped no admission cell — the decision this test records would \
             then be a no-op, which is the opposite of what sc-19115 concluded"
    );
    assert!(
        flipped_cells <= changed_records * host_bytes.len(),
        "{flipped_cells} flips exceed the {} cells the {changed_records} changed records span \
             — the tally is counting something other than host cells",
        changed_records * host_bytes.len()
    );
    // The per-provider tally is a PARTITION of the flips, not a pinned census: it must sum to
    // the total and may only name providers this corpus actually carries.
    assert_eq!(
        flips_by_provider.values().sum::<usize>(),
        flipped_cells,
        "the per-provider tally must partition the flipped cells: {flips_by_provider:?}"
    );
    let image_providers = evidence
        .records
        .iter()
        .filter(|record| record.backend == Backend::Mlx && record.target.mode == "text_to_image")
        .map(|record| record.target.provider.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    for provider in flips_by_provider.keys() {
        assert!(
            image_providers.contains(provider),
            "{provider} flipped admission but carries no MLX text_to_image record"
        );
    }
}

/// sc-18864. The emitted phase object must carry ONE named MLX quantity per field. Before this
/// story it carried five keys for two readings: `deviceBytes` and `wiredBytes` were both set to
/// `active + cache`, which is how every committed MLX record claimed wired residency of
/// 99.7-159.6 GB against a probed 87.0 GB ceiling.
///
/// Asserting the key SET is the load-bearing half — an assertion that only checked the three
/// surviving values would still pass if the two aliases came back.
#[test]
fn phase_json_emits_one_named_mlx_quantity_per_field() {
    let phase = PhaseMemory {
        active: 35_678_641_896,
        cache: 106_969_676_964,
    };
    let value = phase.json();
    let object = value.as_object().expect("phase json object");
    let mut keys = object.keys().map(String::as_str).collect::<Vec<_>>();
    keys.sort_unstable();
    assert_eq!(keys, ["activeBytes", "allocatorBytes", "reclaimableBytes"]);
    assert_eq!(object["activeBytes"], json!(35_678_641_896u64));
    assert_eq!(object["reclaimableBytes"], json!(106_969_676_964u64));
    // `allocatorBytes` is DERIVED, and the derivation is the whole content of the field.
    assert_eq!(object["allocatorBytes"], json!(142_648_318_860u64));
    assert_eq!(
        object["allocatorBytes"].as_u64().expect("allocator bytes"),
        object["activeBytes"].as_u64().expect("active bytes")
            + object["reclaimableBytes"]
                .as_u64()
                .expect("reclaimable bytes")
    );
}

/// sc-18864. `predictedPeakBytes` was hardcoded `null` on the LTX arm, which failed both
/// `mlx_admission_envelope` and the worker's `RequiredNullable::Value` seed check. Nothing
/// about it is contract-dependent — it is `predicted_ceiling` over the arm's own phases.
///
/// The measured q8 LTX numbers below are `imc-2c064567893ea869006e` verbatim, and they are the
/// discriminating case: a ceiling taken over `allocator_bytes()` instead of `active` would
/// predict 149.8 GB of demand on a 137.4 GB host for a render that completed comfortably.
#[test]
fn ltx_predicted_peak_is_derived_from_the_resident_peak_not_the_co_existence_bound() {
    let conditioning = PhaseMemory {
        active: 35_678_641_896,
        cache: 4_371_718_590,
    };
    let denoise = PhaseMemory {
        active: 36_831_735_964,
        cache: 0,
    };
    let decode = PhaseMemory {
        active: 37_931_479_408,
        cache: 104_716_839_452,
    };
    let predicted = ltx_predicted_peak_bytes(conditioning, denoise, decode);
    assert_eq!(
        predicted,
        json!({
            "conditioning": predicted_ceiling(35_678_641_896),
            "denoise": predicted_ceiling(36_831_735_964),
            "decode": predicted_ceiling(37_931_479_408),
            "overall": predicted_ceiling(37_931_479_408),
        })
    );
    let host_bytes = 137_438_953_472u64;
    let overall = predicted["overall"].as_u64().expect("predicted overall");
    assert!(
        overall < host_bytes,
        "a resident-peak ceiling must fit the capture host, got {overall}"
    );
    let overall_phase = PhaseMemory::overall(&[conditioning, denoise, decode]);
    assert!(
        predicted_ceiling(overall_phase.allocator_bytes()) > host_bytes,
        "the co-existence bound must NOT fit, or this test does not discriminate"
    );
    // Monotonicity is what lets the `predictedOverallCeiling` diagnostic agree with the record.
    assert_eq!(overall, predicted_ceiling(overall_phase.active));
}

/// sc-18104: an unimplemented provider must be refused by name at dispatch, not fall through to
/// the Qwen arm. The regression this guards is silent MISROUTING, so asserting "it errored" is
/// not enough — the old code errored too, just with a Qwen-shaped message after entering the
/// wrong arm. Assert the provider is named, and that none of the Qwen arm's own failure
/// vocabulary appears, which is what distinguishes a dispatch refusal from a misroute.
#[test]
fn run_refuses_a_provider_the_mlx_adapter_does_not_implement() {
    // `flux2_dev` left this list when sc-18218 landed its arm, and `flux2_klein_9b` when
    // sc-22727 landed the klein members; `flux2_dev_edit` and the klein edit routes stay — a
    // contract-only provider is not a dispatchable lane.
    // `ltx_2_3` left this class when sc-18808 landed the video arm; `ltx_2_3_distilled` (the
    // CANDLE LTX engine id) and `ltx_2_3_eros` stay — neither is a dispatchable MLX lane.
    for provider in [
        "flux2_klein_9b_edit",
        "flux2_klein_9b_kv_edit",
        "flux2_dev_edit",
        "sana",
        "ltx_2_3_distilled",
        "ltx_2_3_eros",
    ] {
        let request = json!({ "planned": { "target": { "provider": provider } } });
        let error = run(&request).expect_err("unimplemented provider must not dispatch");
        assert_eq!(
            error,
            format!("MLX five-rung calibration does not implement provider {provider:?}")
        );
        assert!(
            !error.contains("SCENEWORKS_QWEN") && !error.contains("calibration mismatch"),
            "refusal leaked the Qwen arm's vocabulary, so dispatch misrouted: {error}"
        );
    }
}

/// The companion direction, and the one the refusal arm above makes necessary: every IMPLEMENTED
/// provider must still reach its own arm through `run`. A typo in any match key — `"qwen_imagee"`
/// — would drop a live lane into the refusal arm permanently, and `mlx:qwen_image` alone carries
/// 9 authoritative plan entries and 41 evidence records.
///
/// Two properties are necessary (sc-18250):
///
///   1. dispatch is probed with LITERAL provider ids, exactly the strings the plan carries, so a
///      corrupted constant or mis-keyed match arm is caught by the refusal complaint leaking
///      through;
///   2. the constants are pinned to those same literals, so an arm's internal use of its
///      constant cannot drift from the id dispatch routes on.
///
/// This must go through `run` to mean anything. Dispatch is cheap to probe here — each arm
/// rejects this minimal request on a missing field long before any env read, catalog build or
/// weight load, so the assertion is on WHICH complaint comes back, not on success.
#[test]
fn every_implemented_provider_still_reaches_its_own_arm_through_dispatch() {
    for provider in [
        "qwen_image",
        // sc-22728: the edit engine provider, which serves both shipped edit catalog ids.
        "qwen_image_edit",
        "z_image_turbo",
        // sc-22724: the undistilled base is its own registry id on the same arm.
        "z_image",
        "krea_2_turbo",
        "sdxl",
        // sc-22729: the bespoke InstantID route is its own provider id on its own arm.
        "instantid",
        "krea_2_turbo_control",
        "flux2_dev",
        // sc-22727: both klein catalog models ride this ONE registry id.
        "flux2_klein_9b",
        // sc-18808: the video arm rides the same dispatch, so it is covered by the same proof.
        "ltx_2_3",
        "ltx_2_5",
        // sc-18663: and the second video arm.
        "minimax_h3",
        // sc-22731: the SANA and Chroma1 families.
        "sana_1600m",
        "sana_sprint_1600m",
        "chroma1_hd",
        "chroma1_base",
        "chroma1_flash",
        // sc-24112: Qwen-Image 2.1 on the turnkey still arm.
        "qwen_image_2_1",
        // sc-25907: and its distilled Turbo sibling, its own registry id on the same arm.
        "qwen_image_2_1_turbo",
    ] {
        let request = json!({ "planned": { "target": { "provider": provider } } });
        let error = run(&request)
            .expect_err("the minimal request is incomplete, so every arm must complain");
        assert!(
            !error.contains("does not implement provider"),
            "{provider} is wired but dispatch refused it — a match key is mis-typed: {error}"
        );
    }
    assert_eq!(QWEN_PROVIDER, "qwen_image");
    assert_eq!(QWEN_IMAGE_2_1_PROVIDER, "qwen_image_2_1");
    assert_eq!(QWEN_IMAGE_2_1_TURBO_PROVIDER, "qwen_image_2_1_turbo");
    assert_eq!(QWEN_EDIT_PROVIDER, "qwen_image_edit");
    assert_eq!(QWEN_EDIT_MODEL, "qwen_image_edit_2511");
    assert_eq!(QWEN_EDIT_LIGHTNING_MODEL, "qwen_image_edit_2511_lightning");
    assert_eq!(Z_IMAGE_PROVIDER, "z_image_turbo");
    assert_eq!(Z_IMAGE_BASE_PROVIDER, "z_image");
    assert_eq!(KREA_BASE_PROVIDER, "krea_2_turbo");
    assert_eq!(SDXL_PROVIDER, "sdxl");
    assert_eq!(INSTANTID_PROVIDER, "instantid");
    assert_eq!(KREA_PROVIDER, "krea_2_turbo_control");
    assert_eq!(FLUX2_PROVIDER, "flux2_dev");
    assert_eq!(FLUX2_KLEIN_PROVIDER, "flux2_klein_9b");
    assert_eq!(LTX_PROVIDER, "ltx_2_3");
    assert_eq!(LTX25_PROVIDER, "ltx_2_5");
    assert_eq!(MINIMAX_PROVIDER, "minimax_h3");
    assert_eq!(SANA_PROVIDER, "sana_1600m");
    assert_eq!(SANA_SPRINT_PROVIDER, "sana_sprint_1600m");
    assert_eq!(CHROMA1_HD_PROVIDER, "chroma1_hd");
    assert_eq!(CHROMA1_BASE_PROVIDER, "chroma1_base");
    assert_eq!(CHROMA1_FLASH_PROVIDER, "chroma1_flash");
}

fn sdxl_planned(model_id: &str, tier: &str, width: u32) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": "sdxl",
                "modelId": model_id,
                "tier": tier,
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": { "width": width, "height": width, "batch": 1, "frames": 1 }
            },
            "backend": "mlx",
            "loadShape": "deferred_materialization",
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": SDXL_CALIBRATION_FINGERPRINT,
            "fixture": "unused"
        }
    })
}

fn sdxl_snapshot_root(repository: &str, revision: &str, tier: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir()
        .join(format!("sc-22729-sdxl-{}-{nonce}", std::process::id()))
        .join(format!("models--{}", repository.replace('/', "--")))
        .join("snapshots")
        .join(revision)
        .join(tier);
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// sc-22729: the family member is read off the plan's `modelId`, and a model id no member
/// serves is refused BY NAME rather than measured as base SDXL. The engine id alone is not an
/// artifact identity — all five load through the same `sdxl` provider.
#[test]
fn sdxl_arm_is_resolved_from_the_plans_model_id() {
    for (model_id, repository, prefix) in [
        ("sdxl", protocol::SDXL_REPOSITORY, "sdxl-base-mlx"),
        ("realvisxl", protocol::REALVISXL_REPOSITORY, "realvisxl-mlx"),
        (
            "realvisxl_lightning",
            protocol::REALVISXL_LIGHTNING_REPOSITORY,
            "realvisxl-lightning-mlx",
        ),
        (
            "illustrious_xl_v1",
            protocol::ILLUSTRIOUS_XL_V1_REPOSITORY,
            "illustrious-xl-v1-mlx",
        ),
        (
            "illustrious_xl_v2",
            protocol::ILLUSTRIOUS_XL_V2_REPOSITORY,
            "illustrious-xl-v2-mlx",
        ),
    ] {
        let arm = sdxl_arm(&sdxl_planned(model_id, "q4", 768)).unwrap();
        assert_eq!(arm.model_id, model_id);
        assert_eq!(arm.expected_repository, repository);
        assert_eq!(arm.fixture_prefix, prefix);
    }
    // Every arm's env family, execution path, calibration label and slug is its OWN, so one
    // member's plan can never be satisfied by another member's staged weights.
    for field in [
        SDXL_FAMILY.map(|arm| arm.repository_env).to_vec(),
        SDXL_FAMILY.map(|arm| arm.revision_env).to_vec(),
        SDXL_FAMILY.map(|arm| arm.root_env).to_vec(),
        SDXL_FAMILY.map(|arm| arm.expected_repository).to_vec(),
        SDXL_FAMILY.map(|arm| arm.execution_path).to_vec(),
        SDXL_FAMILY.map(|arm| arm.still_calibration).to_vec(),
        SDXL_FAMILY.map(|arm| arm.fixture_prefix).to_vec(),
        SDXL_FAMILY.map(|arm| arm.slug).to_vec(),
    ] {
        let mut unique = field.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), field.len(), "collision in {field:?}");
    }
    for absent in ["instantid_realvisxl", "sdxl_turbo", "z_image"] {
        let error = sdxl_arm(&sdxl_planned(absent, "q4", 768)).unwrap_err();
        assert!(error.contains(&format!("modelId {absent:?}")), "{error}");
    }
    let error = sdxl_arm(&json!({
        "planned": { "target": { "provider": "krea_2_turbo", "modelId": "sdxl" } }
    }))
    .unwrap_err();
    assert!(error.contains("does not implement provider"), "{error}");
}

/// sc-22729: the LoadSpec mirrors the worker's MLX still-image shape — the resolved catalog
/// route (`image_jobs/base.rs:8785`), the deferred load shape the registry's legacy shaper
/// applies to every MLX `sdxl` rule, and the tier's advisory quant
/// (`mlx_load_quant_for_resolved_artifact` forwards it for `sdxl`). A q8 plan is never
/// satisfied by a q4 export.
#[test]
fn sdxl_spec_binds_the_route_the_tier_and_the_worker_load_shape() {
    const REVISION: &str = "36699bb8a6353e61c920e3bf19f0e6f8e4151c55";
    for arm in SDXL_FAMILY {
        for (tier, quant) in [
            ("q4", Some(Quant::Q4)),
            ("q8", Some(Quant::Q8)),
            ("bf16", None),
        ] {
            let root = sdxl_snapshot_root(arm.expected_repository, REVISION, tier);
            let selection = MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: MemoryStrategyParameters::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant,
                    component_precision_floors: &[],
                },
            };
            let spec = sdxl_spec_at(arm, root, tier, &selection)
                .unwrap_or_else(|error| panic!("{}/{tier}: {error}", arm.model_id));
            assert_eq!(spec.quantize, quant, "{}/{tier} load quant", arm.model_id);
            assert_eq!(spec.load_shape, LoadShape::DeferredMaterialization);
            assert_eq!(spec.offload_policy, OffloadPolicy::Resident);
            assert_eq!(
                spec.resolved_route.as_deref(),
                Some(arm.model_id),
                "the capture must bind the CATALOG route, not the engine id"
            );
        }
        // A selection whose quant disagrees with the planned tier is refused, so a q8 plan can
        // never be satisfied by a q4 selection that merely reused the tier string.
        let root = sdxl_snapshot_root(arm.expected_repository, REVISION, "q8");
        let mismatched = MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: MemoryStrategyParameters::default(),
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
        };
        let error = sdxl_spec_at(arm, root, "q8", &mismatched).unwrap_err();
        assert!(error.contains("q8"), "{}: {error}", arm.model_id);
    }
    // Staging is independent of scratch bounds. SDXL rung 4 requires staging; bounded decode
    // keeps the heavy components resident before the measured request begins.
    let root = sdxl_snapshot_root(protocol::REALVISXL_REPOSITORY, REVISION, "bf16");
    let staged = MemorySelection {
        strategy: MemoryStrategy::StagedResidency,
        parameters: MemoryStrategyParameters::default(),
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        },
    };
    let spec = sdxl_spec_at(REALVISXL_ARM, root.clone(), "bf16", &staged).unwrap();
    assert_eq!(spec.offload_policy, OffloadPolicy::Sequential);
    let mut selection = staged;
    for (strategy, policy) in [
        (MemoryStrategy::BoundedDecode, OffloadPolicy::Resident),
        (
            MemoryStrategy::BoundedTransformerResidency,
            OffloadPolicy::Sequential,
        ),
    ] {
        selection.strategy = strategy;
        let spec = sdxl_spec_at(REALVISXL_ARM, root.clone(), "bf16", &selection).unwrap();
        assert_eq!(spec.offload_policy, policy);
        assert_eq!(spec.load_shape, LoadShape::DeferredMaterialization);
    }
}

/// sc-22729: the fixture stem names the MODEL. Reusing base SDXL's fixture on a `realvisxl`
/// plan is refused, which is the only thing that keeps a finetune's peaks from being filed
/// under the base model when both load through the same provider id.
#[test]
fn sdxl_fixture_binds_the_family_member_not_just_the_engine() {
    let fixture = |name: &str| json!({ "planned": { "target": { "provider": "sdxl", "modelId": "realvisxl" }, "fixture": name } });
    assert_eq!(
        planned_sdxl_seed(
            &fixture("realvisxl-mlx-q4-768-seed18379-step2"),
            REALVISXL_ARM,
            "q4",
            768
        )
        .unwrap(),
        SDXL_SEED
    );
    for wrong in [
        "sdxl-base-mlx-q4-768-seed18379-step2",
        "realvisxl-mlx-q8-768-seed18379-step2",
        "realvisxl-mlx-q4-1024-seed18379-step2",
    ] {
        assert!(
            planned_sdxl_seed(&fixture(wrong), REALVISXL_ARM, "q4", 768).is_err(),
            "{wrong} was accepted for realvisxl/q4/768"
        );
    }
    assert!(planned_sdxl_seed(
        &fixture("realvisxl-mlx-q4-768-seed18379-step30"),
        REALVISXL_ARM,
        "q4",
        768
    )
    .is_err());
}

/// sc-22729: the InstantID arm is the bespoke identity route and refuses every other identity
/// — a different provider, a different catalog model, a non-character mode, or a plan that
/// asks for the deferred shape the pinned provider can never publish.
#[test]
fn instantid_target_binds_the_bespoke_identity_route() {
    let planned = |provider: &str, model_id: &str, mode: &str, overlay: &str| {
        json!({
            "planned": {
                "target": {
                    "provider": provider, "modelId": model_id, "tier": "bf16", "mode": mode,
                    "overlay": overlay,
                    "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
                },
                "backend": "mlx",
                "loadShape": "eager_materialization",
                "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
                "calibrationFingerprint": INSTANTID_CALIBRATION_FINGERPRINT,
                "fixture": "instantid-realvisxl-mlx-bf16-1024-seed18381-step2"
            }
        })
    };
    validate_instantid_target(&planned(
        "instantid",
        "instantid_realvisxl",
        "character_image",
        "identity",
    ))
    .unwrap();
    for (provider, model_id, mode, overlay, needle) in [
        (
            "sdxl",
            "instantid_realvisxl",
            "character_image",
            "identity",
            "does not implement provider",
        ),
        (
            "instantid",
            "realvisxl",
            "character_image",
            "identity",
            "requires modelId",
        ),
        (
            "instantid",
            "instantid_realvisxl",
            "text_to_image",
            "identity",
            "character_image mode",
        ),
        (
            "instantid",
            "instantid_realvisxl",
            "character_image",
            "none",
            "identity",
        ),
    ] {
        let error =
            validate_instantid_target(&planned(provider, model_id, mode, overlay)).unwrap_err();
        assert!(
            error.contains(needle),
            "{provider}/{model_id}/{mode}/{overlay}: {error}"
        );
    }
    // The provider hard-codes `LoadShape::EagerMaterialization` on its calibration identity, so
    // a deferred plan is refused before any weight is opened.
    let mut deferred = planned(
        "instantid",
        "instantid_realvisxl",
        "character_image",
        "identity",
    );
    deferred["planned"]["loadShape"] = json!("deferred_materialization");
    let error = run_instantid(&deferred).unwrap_err();
    assert!(error.contains("eager_materialization"), "{error}");
    // The MLX lane honours all three packed tiers (the candle lane is dense-only).
    assert_eq!(instantid_numeric_tier("q4").quant, Some(Quant::Q4));
    assert_eq!(instantid_numeric_tier("q8").quant, Some(Quant::Q8));
    assert_eq!(instantid_numeric_tier("bf16").quant, None);
}

/// sc-22729: the artifact digest is over the ORDERED paths with a record separator, so a swap
/// between two roles changes it and two different splits of the same text cannot collide. This
/// is the axis the provider's `overlay_key()` carries, so it is what stops one InstantID
/// artifact set from pricing another.
#[test]
fn instantid_artifact_fingerprint_is_ordered_and_separated() {
    let a = Path::new("/w/a");
    let b = Path::new("/w/b");
    assert_ne!(
        instantid_artifact_fingerprint(&[a, b]),
        instantid_artifact_fingerprint(&[b, a]),
        "role order must be part of the digest"
    );
    assert_ne!(
        instantid_artifact_fingerprint(&[Path::new("/ab"), Path::new("/c")]),
        instantid_artifact_fingerprint(&[Path::new("/a"), Path::new("/bc")]),
        "the record separator must stop a split collision"
    );
    let digest = instantid_artifact_fingerprint(&[a, b]);
    assert_eq!(digest.len(), 64);
    assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
    assert!(
        !digest.contains('='),
        "the shared safety gate rejects = in an overlay"
    );
}

/// sc-22729 review: the MLX identity arm driven END TO END on a synthetic tree — the twin of
/// `candle.rs`'s `instantid_candle_identity_names_the_missing_identitynet_weight`, which can
/// only be COMPILE-checked on this host. Nothing used to reach past the environment reads, so
/// deleting `protocol::instantid_controlnet_dir()?`, or a file out of the identity bundle, or a
/// path out of the digest, reddened nothing.
#[test]
fn instantid_binding_names_the_missing_identitynet_weight_and_covers_five_ordered_paths() {
    use protocol::staging::StagedInstantId;

    let staged = StagedInstantId::install("mlx-instantid", protocol::REALVISXL_REPOSITORY, "bf16");
    let error = match instantid_binding("bf16") {
        Err(error) => error,
        Ok(_) => panic!("an IdentityNet directory carrying no weight file must be refused"),
    };
    assert!(
        error.contains(protocol::INSTANTID_CONTROLNET_WEIGHT_FILE),
        "{error}"
    );

    staged.stage_identitynet_weight();
    let binding = match instantid_binding("bf16") {
        Ok(binding) => binding,
        Err(error) => panic!("a complete staged tree must bind: {error}"),
    };
    assert_eq!(binding.repository, protocol::REALVISXL_REPOSITORY);
    assert_eq!(binding.revision, StagedInstantId::REVISION);

    let digest_of = |paths: &[PathBuf]| {
        instantid_artifact_fingerprint(&paths.iter().map(PathBuf::as_path).collect::<Vec<_>>())
    };
    // The digest covers EXACTLY these five, in exactly this order: a permutation and every
    // four-path subset both differ.
    let ordered = staged.ordered_paths();
    assert_eq!(binding.artifact_fingerprint, digest_of(&ordered));
    let mut swapped = ordered.clone();
    swapped.swap(3, 4);
    assert_ne!(
        binding.artifact_fingerprint,
        digest_of(&swapped),
        "the digest must be order-sensitive across the face stack"
    );
    for dropped in 0..ordered.len() {
        let mut short = ordered.clone();
        short.remove(dropped);
        assert_ne!(
            binding.artifact_fingerprint,
            digest_of(&short),
            "path {dropped} is not covered by the digest"
        );
    }
    // …and the bundle the binding resolved is the staged one, so a missing bundle file would
    // have refused above rather than silently binding an empty face stack.
    assert_eq!(
        binding.bundle.face_dir,
        std::fs::canonicalize(&staged.bundle).unwrap()
    );
}

/// sc-22729: the calibration fingerprints the plan may cite are exactly the strings the pinned
/// crates publish — never a hand-written label. `mlx-gen-sdxl` mints ONE route-independent
/// string for the whole family; `mlx-gen-instantid` reuses its request-contract revision.
///
/// The second half walks the COMMITTED plan: every MLX SDXL-family and InstantID row must cite
/// the string its crate can actually emit, and must declare the load shape that crate publishes
/// (SDXL takes it from the spec — the worker's deferred shape; InstantID hard-codes eager). A
/// plan row that cites a registry-conformance label or the wrong shape would be refused at
/// capture time, on a booked host, instead of here.
#[test]
fn planned_fingerprints_are_the_pinned_crates_own_strings() {
    assert_eq!(
        SDXL_CALIBRATION_FINGERPRINT,
        mlx_gen_sdxl::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
    );
    assert_eq!(
        INSTANTID_CALIBRATION_FINGERPRINT,
        mlx_gen_instantid::memory_strategy::REQUEST_EVIDENCE_REVISION
    );
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan is valid JSON");
    let mut checked: Vec<String> = Vec::new();
    for (key, anchor) in plan["anchors"].as_object().expect("anchors is an object") {
        if !key.ends_with(":mlx") {
            continue;
        }
        let (fingerprint, load_shape, mode) = match anchor["provider"].as_str() {
            Some(SDXL_PROVIDER) => (
                SDXL_CALIBRATION_FINGERPRINT,
                protocol::LOAD_SHAPE_DEFERRED,
                "text_to_image",
            ),
            Some(INSTANTID_PROVIDER) => (
                INSTANTID_CALIBRATION_FINGERPRINT,
                protocol::LOAD_SHAPE_EAGER,
                "character_image",
            ),
            _ => continue,
        };
        assert_eq!(
            anchor["calibrationFingerprint"],
            json!(fingerprint),
            "{key}"
        );
        assert_eq!(anchor["loadShape"], json!(load_shape), "{key}");
        assert_eq!(anchor["mode"], json!(mode), "{key}");
        let model_id = key.split(':').next().expect("anchor key shape");
        assert!(
            SDXL_FAMILY.iter().any(|arm| arm.model_id == model_id)
                || model_id == INSTANTID_MODEL_ID,
            "{key} names a model no arm serves"
        );
        checked.push(key.clone());
    }
    // Not a floor (the candle twin's `>= 6` had the same defect, sc-22729 review): the EXACT
    // set of MLX keys the family plans. A floor stayed green with a whole member deleted from
    // the plan, which is the one thing this case is here to notice.
    checked.sort();
    let mut expected = SDXL_FAMILY
        .iter()
        .map(|arm| arm.model_id)
        .chain(std::iter::once(INSTANTID_MODEL_ID))
        .flat_map(|model_id| {
            ["bf16", "q4", "q8"]
                .into_iter()
                .map(move |tier| format!("{model_id}:{tier}:mlx"))
        })
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(checked, expected);
}

#[test]
fn qwen_complete_sweep_verifies_only_the_exact_executed_case() {
    let request = json!({
        "planned": {
            "strategy": {
                "parameters": { "decodeTileEdge": 448, "decodeOverlap": 64 }
            }
        }
    });
    let sweep = qwen_complete_sweep(&request).unwrap();
    assert_eq!(sweep["rangeVerified"], true);
    let axes = sweep["axes"].as_array().unwrap();
    assert!(axes.iter().any(|axis| {
        axis["parameter"] == "decodeTileEdge" && axis["testedValues"] == json!([448])
    }));
    assert!(axes.iter().any(|axis| {
        axis["parameter"] == "decodeOverlap" && axis["testedValues"] == json!([64])
    }));
    assert_eq!(sweep["cases"].as_array().unwrap().len(), 1);
    assert_eq!(
        sweep["cases"][0]["parameters"],
        request["planned"]["strategy"]["parameters"]
    );
}

#[test]
fn z_image_complete_sweep_verifies_only_the_exact_executed_case() {
    let request = json!({
        "planned": {
            "strategy": {
                "parameters": { "decodeTileEdge": 768, "decodeOverlap": 64 }
            }
        }
    });
    let sweep = z_image_complete_sweep(&request).unwrap();
    assert_eq!(sweep["rangeVerified"], true);
    assert_eq!(sweep["cases"].as_array().unwrap().len(), 1);
    assert_eq!(
        sweep["cases"][0]["parameters"],
        request["planned"]["strategy"]["parameters"]
    );
}

fn z_image_planned(provider: &str, mode: &str, tier: &str) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "modelId": if mode == "edit_image" { "z_image_edit" } else { provider },
                "tier": tier,
                "mode": mode,
                "overlay": "none",
                "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 }
            },
            "backend": "mlx",
            "loadShape": "eager_materialization",
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": "unused",
            "fixture": "unused"
        }
    })
}

/// sc-22724: the family member is read off the plan's `(provider, mode)`, and a pair no member
/// serves is refused by name rather than measured as its nearest neighbour.
#[test]
fn z_image_arm_is_resolved_from_the_plans_provider_and_mode() {
    let turbo = z_image_arm(&z_image_planned("z_image_turbo", "text_to_image", "q4")).unwrap();
    assert_eq!(turbo, Z_IMAGE_TURBO_ARM);
    assert!(!turbo.edit);
    let edit = z_image_arm(&z_image_planned("z_image_turbo", "edit_image", "q4")).unwrap();
    assert_eq!(edit, Z_IMAGE_EDIT_ARM);
    assert!(edit.edit);
    assert_eq!(
        edit.provider, Z_IMAGE_PROVIDER,
        "the edit alias loads the Turbo provider"
    );
    assert_eq!(edit.expected_repository, protocol::Z_IMAGE_REPOSITORY);
    let base = z_image_arm(&z_image_planned("z_image", "text_to_image", "q4")).unwrap();
    assert_eq!(base, Z_IMAGE_BASE_ARM);
    assert_eq!(base.expected_repository, protocol::Z_IMAGE_BASE_REPOSITORY);
    assert_eq!(base.root_env, "SCENEWORKS_Z_IMAGE_BASE_ROOT");
    for (provider, mode) in [
        ("z_image", "edit_image"),
        ("z_image_turbo", "image_to_image"),
        ("z_image_control", "text_to_image"),
    ] {
        let error = z_image_arm(&z_image_planned(provider, mode, "q4")).unwrap_err();
        assert!(
            error.contains(&format!("provider {provider:?} in mode {mode:?}")),
            "{provider}/{mode}: {error}"
        );
    }
    assert!(
        z_image_arm(&json!({ "planned": { "target": { "provider": "z_image_turbo" } } }))
            .unwrap_err()
            .contains("planned.target.mode")
    );
}

fn z_image_snapshot_root(repository: &str, revision: &str, tier: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir()
        .join(format!("sc-22724-z-image-{}-{nonce}", std::process::id()))
        .join(format!("models--{}", repository.replace('/', "--")))
        .join("snapshots")
        .join(revision)
        .join(tier);
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// sc-22724: the tier is the PLAN's, the root must carry it, and the `LoadSpec` re-asserts it
/// — a q8 plan against a q4 export is refused naming q8, on every family member. This used to
/// be a literal `"q4"` in both the validator and the `LoadSpec`, which capped the MLX Z-Image
/// lane at one tier (the sc-17097 defect class on the other adapter).
#[test]
fn z_image_root_must_carry_the_planned_tier_and_the_spec_binds_it() {
    const REVISION: &str = "bb2bc9893b3c49ae96c813350775f791a2e8bc80";
    for (provider, mode, repository) in [
        (
            "z_image_turbo",
            "text_to_image",
            protocol::Z_IMAGE_REPOSITORY,
        ),
        ("z_image_turbo", "edit_image", protocol::Z_IMAGE_REPOSITORY),
        (
            "z_image",
            "text_to_image",
            protocol::Z_IMAGE_BASE_REPOSITORY,
        ),
    ] {
        let q4_root = z_image_snapshot_root(repository, REVISION, "q4");
        let error = z_image_load_spec_at(
            &z_image_planned(provider, mode, "q8"),
            LoadShape::EagerMaterialization,
            repository.to_owned(),
            REVISION.to_owned(),
            q4_root.clone(),
        )
        .expect_err("a q8 plan must not be satisfied by a q4 root");
        assert!(
            error.ends_with(&format!("/snapshots/{REVISION}/q8")),
            "{provider}/{mode}: {error}"
        );
        for (tier, quant) in [
            ("q4", Some(Quant::Q4)),
            ("q8", Some(Quant::Q8)),
            ("bf16", None),
        ] {
            let root = z_image_snapshot_root(repository, REVISION, tier);
            let artifact = z_image_load_spec_at(
                &z_image_planned(provider, mode, tier),
                LoadShape::DeferredMaterialization,
                repository.to_owned(),
                REVISION.to_owned(),
                root.clone(),
            )
            .unwrap_or_else(|error| panic!("{provider}/{mode}/{tier}: {error}"));
            assert_eq!(artifact.tier, tier);
            assert_eq!(artifact.spec.quantize, quant, "{tier} load quant");
            assert_eq!(artifact.spec.load_shape, LoadShape::DeferredMaterialization);
            assert_eq!(
                artifact.loadability_fingerprint(),
                format!("{repository}@{REVISION}:{tier}")
            );
            assert_eq!(
                artifact.arm.provider,
                if provider == "z_image" {
                    "z_image"
                } else {
                    "z_image_turbo"
                }
            );
        }
        // The wrong artifact family is refused before the root is even looked at.
        let other = if repository == protocol::Z_IMAGE_REPOSITORY {
            protocol::Z_IMAGE_BASE_REPOSITORY
        } else {
            protocol::Z_IMAGE_REPOSITORY
        };
        let error = z_image_load_spec_at(
            &z_image_planned(provider, mode, "q4"),
            LoadShape::EagerMaterialization,
            other.to_owned(),
            REVISION.to_owned(),
            q4_root,
        )
        .expect_err("the other family's repository must be refused");
        assert!(error.contains(repository), "{provider}: {error}");
    }
}

/// sc-22724: an edit capture is the worker's edit request — one reference at the target
/// geometry plus the production strength — and a text-to-image capture carries none.
#[test]
fn z_image_edit_request_carries_one_reference_at_the_target_geometry() {
    let edit = z_image_request(Z_IMAGE_EDIT_ARM, 768, 512);
    assert_eq!(edit.conditioning.len(), 1);
    match &edit.conditioning[0] {
        Conditioning::Reference { image, strength } => {
            assert_eq!((image.width, image.height), (768, 512));
            assert_eq!(image.pixels.len(), 768 * 512 * 3);
            assert_eq!(*strength, Some(Z_IMAGE_EDIT_STRENGTH));
        }
        other => panic!("expected one Reference, got {other:?}"),
    }
    // The worker sets ONLY the per-reference strength (`build_lane_conditioning`,
    // image_jobs/base.rs:7136); the request-level lever stays unset, and so does this arm's.
    assert_eq!(edit.strength, None);
    // floor(4 * 0.6) = 2, so two executed denoise steps remain behind the conditioning
    // boundary — the engine's `init_time_step` law (mlx-gen/src/img2img.rs), not this arm's.
    assert_eq!(edit.steps, Some(Z_IMAGE_EDIT_STEPS));
    for arm in [Z_IMAGE_TURBO_ARM, Z_IMAGE_BASE_ARM] {
        let plain = z_image_request(arm, 768, 768);
        assert!(plain.conditioning.is_empty());
        assert_eq!(plain.strength, None);
        assert_eq!(plain.steps, Some(2));
    }
}

fn qwen_edit_planned(model_id: &str, tier: &str, overlay: &str, fixture: &str) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": "qwen_image_edit",
                "modelId": model_id,
                "tier": tier,
                "mode": "edit_image",
                "overlay": overlay,
                "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 }
            },
            "backend": "mlx",
            "loadShape": "deferred_materialization",
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": "qwen-image-mlx-shared-ladder-2026-08-01-v1",
            "fixture": fixture
        }
    })
}

/// sc-22728: the two shipped edit catalog ids are ONE engine provider, so nothing in the request
/// but the model id separates them. The arm is resolved from `(provider, modelId)` and any other
/// pair is refused by name rather than measured as its nearest neighbour — which here would mean
/// publishing the distilled stack's peaks as the production path's.
#[test]
fn the_qwen_edit_arm_is_resolved_from_the_plans_provider_and_model_id() {
    let base = qwen_edit_arm(&qwen_edit_planned(
        "qwen_image_edit_2511",
        "q4",
        "none",
        "qwen-edit-mlx-q4-seed16353-step2",
    ))
    .unwrap();
    assert_eq!(base, QWEN_EDIT_ARM);
    assert!(!base.lightning);
    assert_eq!(base.steps, 2);
    assert_eq!(base.overlay, "none");
    let lightning = qwen_edit_arm(&qwen_edit_planned(
        "qwen_image_edit_2511_lightning",
        "q4",
        "lora",
        "qwen-edit-lightning-mlx-q4-seed16353-step4",
    ))
    .unwrap();
    assert_eq!(lightning, QWEN_EDIT_LIGHTNING_ARM);
    assert!(lightning.lightning);
    assert_eq!(lightning.steps, 4, "the official lightx2v 4-step recipe");
    assert_eq!(lightning.overlay, "lora");
    assert_ne!(base.slug, lightning.slug, "one diagnostics source each");
    for (provider, model_id) in [
        ("qwen_image_edit", "qwen_image_edit_2509"),
        ("qwen_image_edit", "qwen_image"),
        ("qwen_image", "qwen_image_edit_2511"),
    ] {
        let mut request = qwen_edit_planned(model_id, "q4", "none", "unused");
        request["planned"]["target"]["provider"] = json!(provider);
        let error = qwen_edit_arm(&request).unwrap_err();
        assert!(
            error.contains(&format!("provider {provider:?} for model {model_id:?}")),
            "{provider}/{model_id}: {error}"
        );
    }
    assert!(
        qwen_edit_arm(&json!({ "planned": { "target": { "provider": "qwen_image_edit" } } }))
            .unwrap_err()
            .contains("planned.target.modelId")
    );
}

/// sc-22728: the fixture binds the tier, the seed AND the recipe's step count, per member. The
/// production id's fixture cannot name the Lightning arm's four steps and vice versa, and neither
/// can borrow the other's prefix — the `planned_qwen_seed` rule, extended across the family.
#[test]
fn the_qwen_edit_fixture_binds_the_member_the_tier_and_the_step_count() {
    for (arm, model_id, prefix, steps) in [
        (
            QWEN_EDIT_ARM,
            "qwen_image_edit_2511",
            "qwen-edit-mlx",
            2_u32,
        ),
        (
            QWEN_EDIT_LIGHTNING_ARM,
            "qwen_image_edit_2511_lightning",
            "qwen-edit-lightning-mlx",
            4,
        ),
    ] {
        for tier in ["q4", "q8", "bf16"] {
            let fixture = format!("{prefix}-{tier}-seed16353-step{steps}");
            let request = qwen_edit_planned(model_id, tier, arm.overlay, &fixture);
            assert_eq!(
                planned_qwen_edit_seed(&request, arm, tier).unwrap(),
                16353,
                "{fixture}"
            );
            // The same fixture under another tier is refused naming the tier it belongs to.
            let other = if tier == "q4" { "q8" } else { "q4" };
            let error = planned_qwen_edit_seed(&request, arm, other).unwrap_err();
            assert!(error.contains(&format!("{prefix}-{other}-seed")), "{error}");
        }
        // The other member's step count is refused, so a 4-step distilled capture can never be
        // recorded under the 2-step production fixture.
        let wrong_steps = if steps == 2 { 4 } else { 2 };
        let fixture = format!("{prefix}-q4-seed16353-step{wrong_steps}");
        let error = planned_qwen_edit_seed(
            &qwen_edit_planned(model_id, "q4", arm.overlay, &fixture),
            arm,
            "q4",
        )
        .unwrap_err();
        assert!(error.contains(&format!("{steps}-step")), "{error}");
    }
    // And the members' prefixes do not overlap: the Lightning fixture is not a production one.
    let error = planned_qwen_edit_seed(
        &qwen_edit_planned(
            "qwen_image_edit_2511",
            "q4",
            "none",
            "qwen-edit-lightning-mlx-q4-seed16353-step2",
        ),
        QWEN_EDIT_ARM,
        "q4",
    )
    .unwrap_err();
    assert!(error.contains("qwen-edit-mlx-q4-seed"), "{error}");
}

/// sc-22728: every Qwen edit capture is an EDIT request — one `Conditioning::Reference` at the
/// target geometry, which the pinned engine hard-requires (`model_edit.rs` refuses a request with
/// no reference) — and only the Lightning member selects the distill's sampler and drops the
/// negative branch the distill was CFG-distilled away from.
#[test]
fn the_qwen_edit_request_carries_one_reference_and_only_lightning_selects_the_distill_recipe() {
    for arm in [QWEN_EDIT_ARM, QWEN_EDIT_LIGHTNING_ARM] {
        let request = qwen_edit_request(arm, 768, 512, 16353);
        assert_eq!(request.conditioning.len(), 1);
        match &request.conditioning[0] {
            Conditioning::Reference { image, strength } => {
                assert_eq!((image.width, image.height), (768, 512));
                assert_eq!(image.pixels.len(), 768 * 512 * 3);
                assert_eq!(
                    *strength, None,
                    "Qwen edit is dual-latent conditioning, not an img2img strength lever"
                );
            }
            other => panic!("expected one Reference, got {other:?}"),
        }
        assert_eq!(request.seed, Some(16353));
        assert_eq!(request.steps, Some(arm.steps));
        assert_eq!(
            request.sampler.as_deref(),
            arm.lightning.then_some("lightning")
        );
        assert_eq!(request.negative_prompt.is_some(), !arm.lightning);
    }
}

#[test]
fn z_image_parity_envelope_accepts_published_decode_and_rejects_mutation() {
    assert!(z_image_quality_passes(48.0 / 255.0, 2.82 / 255.0));
    assert!(!z_image_quality_passes(57.0 / 255.0, 2.82 / 255.0));
    assert!(!z_image_quality_passes(48.0 / 255.0, 4.01 / 255.0));

    let baseline = Image {
        width: 1,
        height: 1,
        pixels: vec![10, 20, 30],
    };
    let mutated = qwen_negative_mutation(&baseline);
    let (maximum, mean) = image_max_mean_abs(&mutated, &baseline).unwrap();
    assert!(!z_image_quality_passes(maximum, mean));
}

#[test]
fn z_image_cleanup_bounds_reject_retained_memory_and_warm_peak_mutations() {
    let clean = AllocatorState {
        active: 1_000,
        cache: 200,
    };
    let bounds = LifecycleMemoryBounds::from_clean_warm(10_000, clean);
    assert_eq!(bounds.tolerance_bytes, 200);
    assert!(bounds.allows_retained(AllocatorState {
        active: 1_200,
        cache: 400,
    }));
    assert!(bounds.allows_warm_peak(10_200));
    assert!(!bounds.allows_retained(AllocatorState {
        active: 1_201,
        cache: 400,
    }));
    assert!(!bounds.allows_retained(AllocatorState {
        active: 1_200,
        cache: 401,
    }));
    assert!(!bounds.allows_warm_peak(10_201));
}

#[test]
fn qwen_parity_envelope_accepts_measured_chunking_and_rejects_mutation() {
    let measured_maximum = 43.0 / 255.0;
    let measured_mean = 0.113 / 255.0;
    assert!(qwen_quality_passes(measured_maximum, measured_mean));
    assert!(!qwen_quality_passes(
        measured_maximum + 0.05,
        measured_mean + 0.05
    ));
    assert!(!qwen_quality_passes(49.0 / 255.0, measured_mean));
    assert!(!qwen_quality_passes(measured_maximum, 0.51 / 255.0));
}

#[test]
fn qwen_fixture_seed_is_bound_to_the_planned_tier() {
    let request = json!({
        "planned": {
            "fixture": "qwen-image-q4-seed16353-step2",
            "target": { "tier": "q4" }
        }
    });
    assert_eq!(planned_qwen_tier(&request).unwrap(), "q4");
    assert_eq!(planned_qwen_seed(&request, "q4").unwrap(), 16353);

    let error = planned_qwen_seed(&request, "q8").unwrap_err();
    assert!(error.contains("must start with"));
}

const QWEN_EDIT_TEST_REVISION: &str = "bb2bc9893b3c49ae96c813350775f791a2e8bc80";

fn scratch_dir(label: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("sc-22728-{label}-{}-{nonce}", std::process::id()))
}

/// A `models--<repo>/snapshots/<revision>/<tier>` root laid out exactly as the HF cache does, so
/// the tier suffix the validator pins is real rather than mocked.
fn qwen_edit_snapshot_root(tier: &str) -> PathBuf {
    let root = scratch_dir("qwen-edit-root")
        .join(format!(
            "models--{}",
            protocol::QWEN_EDIT_REPOSITORY.replace('/', "--")
        ))
        .join("snapshots")
        .join(QWEN_EDIT_TEST_REVISION)
        .join(tier);
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// The distill snapshot, with the one pinned file actually on disk — the adapter helper refuses
/// any other file name, so a fixture that skipped this would assert nothing.
fn qwen_edit_lightning_fixture() -> QwenEditLightningSource {
    let root = scratch_dir("qwen-edit-lightning");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join(protocol::QWEN_EDIT_LIGHTNING_FILE), b"distill").unwrap();
    QwenEditLightningSource {
        repository: protocol::QWEN_EDIT_LIGHTNING_REPOSITORY.to_owned(),
        revision: QWEN_EDIT_TEST_REVISION.to_owned(),
        root,
    }
}

fn qwen_edit_source(arm: QwenEditArm, tier: &str) -> QwenEditArtifactSource {
    QwenEditArtifactSource {
        repository: protocol::QWEN_EDIT_REPOSITORY.to_owned(),
        revision: QWEN_EDIT_TEST_REVISION.to_owned(),
        root: qwen_edit_snapshot_root(tier),
        lightning: arm.lightning.then(qwen_edit_lightning_fixture),
    }
}

fn qwen_edit_request_for(arm: QwenEditArm, model_id: &str, tier: &str) -> Value {
    let fixture = format!("{}-{tier}-seed16353-step{}", arm.fixture_prefix, arm.steps);
    qwen_edit_planned(model_id, tier, arm.overlay, &fixture)
}

const QWEN_EDIT_MEMBERS: [(QwenEditArm, &str); 2] = [
    (QWEN_EDIT_ARM, "qwen_image_edit_2511"),
    (QWEN_EDIT_LIGHTNING_ARM, "qwen_image_edit_2511_lightning"),
];

/// sc-22728: the ONE thing that makes the Lightning member Lightning at load time. Every
/// downstream claim — the record's `builtInAdapters` count, the overlay scenario's verdict, and
/// the run context's `overlay` — is read off `spec.adapters`, so if this attachment were lost the
/// arm would publish an authoritative record asserting a distill that never participated. The
/// stack itself is therefore asserted here, on BOTH members and at every shipped tier.
#[test]
fn the_qwen_edit_load_spec_carries_the_distill_on_exactly_the_lightning_member() {
    for (arm, model_id) in QWEN_EDIT_MEMBERS {
        for (tier, quant) in [
            ("q4", Some(Quant::Q4)),
            ("q8", Some(Quant::Q8)),
            ("bf16", None),
        ] {
            let request = qwen_edit_request_for(arm, model_id, tier);
            let selection = planned_selection(&request).unwrap();
            let source = qwen_edit_source(arm, tier);
            let artifact = qwen_edit_load_spec(
                arm,
                tier,
                &source,
                &selection,
                OffloadPolicy::Resident,
                LoadShape::EagerMaterialization,
            )
            .unwrap_or_else(|error| panic!("{model_id}/{tier}: {error}"));
            assert_eq!(
                artifact.spec.quantize, quant,
                "{model_id}/{tier} load quant"
            );
            assert_eq!(
                artifact.spec.adapters.len(),
                usize::from(arm.lightning),
                "{model_id}/{tier}: the distill stack must be exactly the member's"
            );
            if arm.lightning {
                let expected = source
                    .lightning
                    .as_ref()
                    .unwrap()
                    .root
                    .canonicalize()
                    .unwrap()
                    .join(protocol::QWEN_EDIT_LIGHTNING_FILE);
                let adapter = &artifact.spec.adapters[0];
                assert_eq!(adapter.path, expected, "{model_id}/{tier} distill path");
                assert!(
                    adapter.path.ends_with(protocol::QWEN_EDIT_LIGHTNING_FILE),
                    "{model_id}/{tier}: {}",
                    adapter.path.display()
                );
                assert_eq!(adapter.scale, 1.0, "{model_id}/{tier} distill scale");
                assert!(
                    matches!(adapter.kind, AdapterKind::Lora),
                    "{model_id}/{tier}: {:?}",
                    adapter.kind
                );
            }
        }
    }
}

/// sc-22728: the root the capture opens must end in the PLANNED tier's directory, per member —
/// a stale `…/q4` export cannot satisfy a q8 plan and quietly re-label another tier's peaks
/// (the sc-17097 defect class). Modelled on
/// `z_image_root_must_carry_the_planned_tier_and_the_spec_binds_it`.
#[test]
fn a_qwen_edit_root_of_another_tier_is_refused_naming_the_planned_tier() {
    for (arm, model_id) in QWEN_EDIT_MEMBERS {
        let request = qwen_edit_request_for(arm, model_id, "q8");
        let selection = planned_selection(&request).unwrap();
        // The plan is q8; the root on disk is the q4 export.
        let source = qwen_edit_source(arm, "q4");
        let error = qwen_edit_load_spec(
            arm,
            "q8",
            &source,
            &selection,
            OffloadPolicy::Resident,
            LoadShape::EagerMaterialization,
        )
        .expect_err("a q8 plan must not be satisfied by a q4 root");
        assert!(
            error.ends_with(&format!("/snapshots/{QWEN_EDIT_TEST_REVISION}/q8")),
            "{model_id}: {error}"
        );
        // The wrong artifact family is refused before the root is even looked at.
        let mut foreign = qwen_edit_source(arm, "q8");
        foreign.repository = protocol::QWEN_EDIT_LIGHTNING_REPOSITORY.to_owned();
        let error = qwen_edit_load_spec(
            arm,
            "q8",
            &foreign,
            &selection,
            OffloadPolicy::Resident,
            LoadShape::EagerMaterialization,
        )
        .expect_err("the distill repository is not the base artifact");
        assert!(
            error.contains(protocol::QWEN_EDIT_REPOSITORY),
            "{model_id}: {error}"
        );
    }
}

#[test]
fn qwen_load_spec_preserves_every_planned_numeric_tier() {
    for (tier, expected_quant) in [
        ("bf16", None),
        ("q4", Some(Quant::Q4)),
        ("q8", Some(Quant::Q8)),
    ] {
        let request = json!({
            "planned": {
                "strategy": {
                    "rung": "bounded_attention",
                    "parameters": {}
                },
                "target": { "tier": tier }
            }
        });
        let selection = planned_selection(&request).unwrap();
        let spec = qwen_load_spec(
            PathBuf::from(format!("/tmp/qwen-image-{tier}")),
            &selection,
            OffloadPolicy::Resident,
            LoadShape::EagerMaterialization,
        );
        assert_eq!(spec.quantize, expected_quant, "numeric tier {tier}");
    }
}

#[test]
fn qwen_capture_uses_the_plans_typed_load_shape_independently_of_rung() {
    for (rung, load_shape, expected) in [
        (
            "bounded_attention",
            protocol::LOAD_SHAPE_DEFERRED,
            LoadShape::DeferredMaterialization,
        ),
        (
            "bounded_transformer_residency",
            protocol::LOAD_SHAPE_DEFERRED,
            LoadShape::DeferredMaterialization,
        ),
        (
            "bounded_attention",
            protocol::LOAD_SHAPE_EAGER,
            LoadShape::EagerMaterialization,
        ),
    ] {
        let request = json!({
            "planned": {
                "loadShape": load_shape,
                "strategy": { "rung": rung, "parameters": {} }
            }
        });
        assert_eq!(planned_load_shape(&request).unwrap(), expected, "{rung}");
    }

    let missing = json!({ "planned": {} });
    assert!(planned_load_shape(&missing)
        .unwrap_err()
        .contains("planned.loadShape must be a string"));
    let mutated = json!({ "planned": { "loadShape": "deferred-ish" } });
    assert!(planned_load_shape(&mutated)
        .unwrap_err()
        .contains("deferred-ish"));
}
