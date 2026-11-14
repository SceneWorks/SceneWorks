use super::*;

#[test]
fn flux3_cfg_source_allowance_tracks_forward_rows_not_jobs_weights_or_decode() {
    assert_eq!(flux3_conditioning_forward_batch("flux3_klein_base"), 2);
    assert_eq!(flux3_conditioning_forward_batch("flux3_klein"), 1);
    let geometry = request_geometry(&request_inputs(1024, 768, 1));
    let full_base = flux3_source_workspaces("flux3_klein_base", 1024, 768);
    // 1024x768 -> 64x48 = 3072 image tokens + 4096 text tokens, two CFG rows.
    let tokens = 3072u64 + 4096;
    assert_eq!(full_base.0, 2 * 4096 * 20480 * 4);
    assert_eq!(
        full_base.1,
        tokens * 3072 * 4 * 28 * 2 + full_base.0 + 3072 * 128 * 4 * 5
    );
    let full_distilled = flux3_source_workspaces("flux3_klein", 1024, 768);
    assert_eq!(full_base.0, full_distilled.0 * 2);
    assert!(full_base.1 > full_distilled.1 * 2); // CFG merge buffers, not another weight copy.
    let phase = flux3_cfg_workspace("flux3_klein_base", geometry, 791);
    assert!(phase.conditioning > 791 && phase.denoise > 791);
    assert_eq!(phase.decode, 791);
    let mut base = request_plan();
    base.engine_id = "flux3_klein_base";
    base.model_id = "flux3_klein_base".into();
    let mut distilled = base.clone();
    distilled.engine_id = "flux3_klein";
    distilled.model_id = "flux3_klein".into();
    let extra = flux3_cfg_additional_workspaces("flux3_klein_base", 1024, 768);
    assert_eq!(
        base.generic_total_peak_bytes(geometry) - distilled.generic_total_peak_bytes(geometry),
        extra.0.max(extra.1)
    );
    assert_eq!(
        base.generic_total_peak_bytes(geometry) - base.generic_headroom_bytes(geometry),
        base.asset_bytes
    );
    for count in [1, 4, 8] {
        let inputs = request_inputs(1024, 768, count);
        assert_eq!(request_geometry(&inputs).batch, 1);
        assert_eq!(
            base.generic_total_peak_bytes(request_geometry(&inputs)),
            base.generic_total_peak_bytes(geometry)
        );
    }
}

// -------------------------------------------------------------------------------------
// Image-MLX anchor injection seam (epic 22505 feature-end fix round): the mirror of
// `vram_gate::tests`' seam, for the same reason — anchor currency is the PACKAGED
// loader-closure declaration, which no argument reaches, so tests that grade the derivation
// itself inject a store stamped at the live declared digest.
// -------------------------------------------------------------------------------------

thread_local! {
    static INJECTED_IMAGE_ANCHOR_STORE: std::cell::Cell<
        Option<&'static sceneworks_core::memory_anchor::MemoryAnchorStore>,
    > = const { std::cell::Cell::new(None) };
}

pub(super) fn injected_image_anchor_store(
) -> Option<&'static sceneworks_core::memory_anchor::MemoryAnchorStore> {
    INJECTED_IMAGE_ANCHOR_STORE.with(std::cell::Cell::get)
}

fn with_injected_image_anchor_store<T>(
    store: sceneworks_core::memory_anchor::MemoryAnchorStore,
    body: impl FnOnce() -> T,
) -> T {
    let leaked: &'static _ = Box::leak(Box::new(store));
    INJECTED_IMAGE_ANCHOR_STORE.with(|cell| cell.set(Some(leaked)));
    let outcome = body();
    INJECTED_IMAGE_ANCHOR_STORE.with(|cell| cell.set(None));
    outcome
}

const DECODE_QUALITY_TEST_STAMP: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn quality_policy_json(route: &str) -> Value {
    serde_json::json!({
        "qualityAbi": gen_core::MEMORY_DECODE_QUALITY_ABI,
        "family": "sdxl",
        "resolvedRoute": route,
        "backend": "mlx",
        "tier": "q4",
        "loadShape": "deferred_materialization",
        "artifact": {
            "repository": "SceneWorks/fixture",
            "revision": "c".repeat(40),
            "variant": "packed-q4",
            "fingerprint": format!("SceneWorks/fixture@{}:packed-q4", "c".repeat(40)),
        },
        "implementationFingerprint": DECODE_QUALITY_TEST_STAMP,
        "mode": "text_to_image",
        "overlay": null,
        "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1, "referenceCount": 0 },
        "usePid": false,
        "tileEdge": 896,
        "overlap": 192,
        "metric": "max_abs_rgb_u8",
        "maximumError": 48,
        "fixtures": [
            { "seed": 7, "productionLatentProvenance": "fixture@revision seed=7", "productionLatentSha256": "a".repeat(64), "denseOutputSha256": "b".repeat(64), "tiledOutputSha256": "c".repeat(64), "observedError": 31 },
            { "seed": 99, "productionLatentProvenance": "fixture@revision seed=99", "productionLatentSha256": "d".repeat(64), "denseOutputSha256": "e".repeat(64), "tiledOutputSha256": "f".repeat(64), "observedError": 42 }
        ],
        "productionEvidenceSha256": "0".repeat(64),
        "disposition": { "kind": "admitted" }
    })
}

#[test]
fn manifest_quality_reader_is_route_bound_and_keeps_absence_legacy() {
    assert!(
        decode_quality_policies_from_manifest(&JsonObject::new(), "realvisxl")
            .unwrap()
            .is_empty()
    );
    let legacy =
        bind_decode_quality_policies_from_manifest(&JsonObject::new(), "realvisxl", None).unwrap();
    assert!(!legacy.authoritative);
    assert!(matches!(
        legacy.decision,
        DecodeQualityBindingDecision::Legacy
    ));

    let manifest = serde_json::json!({
        "mlx": { "memoryStrategyContract": { "implementations": [{
            "rung": "bounded_decode",
            "parameterRanges": { "decodeGeometryPolicies": [quality_policy_json("realvisxl")] }
        }] } }
    });
    let manifest = manifest.as_object().unwrap();
    let rows = decode_quality_policies_from_manifest(manifest, "realvisxl").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].resolved_route, "realvisxl");
    assert_eq!(
        rows[0].load_shape,
        gen_core::LoadShape::DeferredMaterialization
    );
    assert!(decode_quality_policies_from_manifest(manifest, "realvisxl_lightning").is_err());
}

#[test]
fn manifest_quality_binding_is_exact_and_refusal_is_typed() {
    let manifest_value = serde_json::json!({
        "mlx": { "memoryStrategyContract": { "implementations": [{
            "rung": "bounded_decode",
            "parameterRanges": { "decodeGeometryPolicies": [quality_policy_json("realvisxl")] }
        }] } }
    });
    let manifest = manifest_value.as_object().unwrap();
    let provenance = fixture_provenance("q4", "packed-q4");

    let bound =
        bind_decode_quality_policies_from_manifest(manifest, "realvisxl", Some(&provenance))
            .expect("matching immutable artifact binds");
    assert!(bound.authoritative);
    assert!(matches!(
        &bound.decision,
        DecodeQualityBindingDecision::Bound { row_count: 1, .. }
    ));
    assert_eq!(
        bound.runtime_identity.as_ref().unwrap().artifact.repository,
        "SceneWorks/fixture"
    );
    let attached = attach_decode_quality_binding(
        LoadSpec::new(WeightsSource::Dir("/models/realvisxl".into())),
        bound.clone(),
        "realvisxl",
    );
    assert!(attached.decode_geometry_policy_authoritative);
    assert_eq!(attached.decode_geometry_policies.len(), 1);

    let unproven = bind_decode_quality_policies_from_manifest(manifest, "realvisxl", None).unwrap();
    assert!(unproven.authoritative);
    assert!(unproven.policies.is_empty());
    assert!(matches!(
        unproven.decision,
        DecodeQualityBindingDecision::Refused { ref reason }
            if reason.contains("no independently resolved artifact identity")
    ));
    let refused_spec = attach_decode_quality_binding(
        LoadSpec::new(WeightsSource::Dir("/models/realvisxl".into())),
        unproven.clone(),
        "realvisxl",
    );
    assert!(refused_spec.decode_geometry_policy_authoritative);
    assert!(refused_spec.decode_geometry_policies.is_empty());

    let mut drifted_identities = Vec::new();
    let mut drifted = provenance.clone();
    drifted.identity.repository = "SceneWorks/other".to_owned();
    drifted_identities.push(drifted);
    let mut drifted = provenance.clone();
    drifted.identity.revision = "d".repeat(40);
    drifted_identities.push(drifted);
    let mut drifted = provenance.clone();
    drifted.identity.variant = "packed-q8".to_owned();
    drifted_identities.push(drifted);
    let mut drifted = provenance.clone();
    drifted.identity.fingerprint = "SceneWorks/fixture@drifted:packed-q4".to_owned();
    drifted_identities.push(drifted);
    for drifted in &drifted_identities {
        let refused =
            bind_decode_quality_policies_from_manifest(manifest, "realvisxl", Some(drifted))
                .unwrap();
        assert!(refused.authoritative);
        assert!(refused.policies.is_empty());
        assert!(matches!(
            refused.decision,
            DecodeQualityBindingDecision::Refused { ref reason }
                if reason.contains("does not match resolved artifact")
        ));
    }

    let mut forensic_source_value = manifest_value;
    forensic_source_value["mlx"]["memoryStrategyContract"]["implementations"][0]
        ["parameterRanges"]["decodeGeometryPolicies"][0]["implementationFingerprint"] =
        Value::String("f".repeat(64));
    let forensic_source = bind_decode_quality_policies_from_manifest(
        forensic_source_value.as_object().unwrap(),
        "realvisxl",
        Some(&provenance),
    )
    .unwrap();
    assert!(forensic_source.authoritative);
    assert_eq!(forensic_source.policies.len(), 1);
    assert_eq!(
        forensic_source.policies[0].implementation_fingerprint,
        "f".repeat(64)
    );
    assert!(matches!(
        forensic_source.decision,
        DecodeQualityBindingDecision::Bound { row_count: 1, .. }
    ));
}

#[test]
fn prepared_component_accounting_uses_token_size_and_propagates_stale_errors() {
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("model.safetensors");
    std::fs::write(&path, vec![0_u8; 23]).expect("weights write");
    let pin = gen_core::PinnedWeightsFile::pin(&path).expect("weights pin");
    let mut spec = LoadSpec::new(WeightsSource::File(path.clone()));
    spec.prepare_with_file_pins([pin])
        .expect("prepared spec finalizes");

    // An EMPTY File slot is filled from the caller's prepared token, not from a fresh stat.
    let (total, _, _) = spec_component_bytes_with_provider_footprint_checked(
        "fixture",
        &spec,
        Some(PerComponentBytes {
            text_encoder: 0,
            dit: 0,
            vae: 0,
        }),
    )
    .expect("prepared accounting succeeds");
    assert_eq!(total, 23, "primary File size comes from its prepared token");

    // A provider that DID price the slot owns it: the prepared token never overwrites a fact.
    let (planned, _, _) = spec_component_bytes_with_provider_footprint_checked(
        "fixture",
        &spec,
        Some(PerComponentBytes {
            text_encoder: 0,
            dit: 9_999,
            vae: 0,
        }),
    )
    .expect("prepared accounting succeeds");
    assert_eq!(
        planned, 9_999,
        "a plan-priced dit survives prepared accounting for a File source"
    );

    std::fs::write(&path, vec![1_u8; 31]).expect("weights replacement writes");
    spec_component_bytes_with_provider_footprint_checked("fixture", &spec, None)
        .expect_err("stale prepared source errors are propagated, not restatted or swallowed");
}

/// sc-20651. The checkpoint-import lane prices the DECODED plan for a quantized single-file
/// import: an fp8 checkpoint is resident at its decode width, well above its on-disk width. The
/// prepared accounting seam used to overwrite that provider fact with the raw file length, so the
/// modeled side of the resident-attribution guard came in short by exactly the decode delta and
/// the guard refused a real import that fits.
///
/// The numbers are the measured kreamania_variant1_fp8 case scaled onto a synthetic plan: the
/// live resident total was 33_483_838_240 against a modeled 31_875_544_875 — a 1_608_293_365-byte
/// shortfall that is exactly the fp8→bf16 (`fpmm`) decode delta the provider had already priced.
///
/// Mutation: restore `footprint.dit = prepared_source_bytes(&spec.weights, &file_sizes);` and this
/// goes RED reporting `PLAN_DIT - ON_DISK_DIT` too few bytes.
#[test]
fn a_plan_priced_dit_is_not_overwritten_by_the_on_disk_file_length() {
    const ON_DISK_DIT: u64 = 64;
    const PLAN_DIT: u64 = ON_DISK_DIT + 1_608_293_365;
    const PLAN_TE: u64 = 5_000;
    const PLAN_VAE: u64 = 700;

    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("checkpoint.safetensors");
    std::fs::write(&path, vec![0_u8; ON_DISK_DIT as usize]).expect("weights write");
    let pin = gen_core::PinnedWeightsFile::pin(&path).expect("weights pin");
    let mut spec = LoadSpec::new(WeightsSource::File(path.clone()));
    spec.prepare_with_file_pins([pin])
        .expect("prepared spec finalizes");

    let (total, te, _) = spec_component_bytes_with_provider_footprint_checked(
        "fixture",
        &spec,
        Some(PerComponentBytes {
            text_encoder: PLAN_TE,
            dit: PLAN_DIT,
            vae: PLAN_VAE,
        }),
    )
    .expect("prepared accounting succeeds");

    let expected = PLAN_TE + PLAN_DIT + PLAN_VAE;
    assert_eq!(
        total,
        expected,
        "the modeled total must carry the plan-priced dit; on-disk substitution under-prices it \
             by {} bytes",
        expected.saturating_sub(total)
    );
    assert_eq!(te, PLAN_TE, "the provider's encoder split is untouched");
}

#[test]
fn ltx_video_tier_is_checkpoint_bound_when_the_load_assertion_is_absent() {
    let fixture = tempfile::tempdir().expect("ltx split fixture");
    let spec = || LoadSpec::new(WeightsSource::Dir(fixture.path().to_owned()));

    let dense = resolved_video_numeric_tier("ltx_2_3", &spec()).expect("absent manifest");
    assert_eq!(dense.quant, None);

    for (bits, expected) in [(4, gen_core::Quant::Q4), (8, gen_core::Quant::Q8)] {
        std::fs::write(
            fixture.path().join("split_model.json"),
            serde_json::json!({ "quantized": true, "quantization_bits": bits }).to_string(),
        )
        .unwrap();
        let tier = resolved_video_numeric_tier("ltx_2_3", &spec())
            .unwrap_or_else(|error| panic!("q{bits} checkpoint resolves: {error}"));
        assert_eq!(tier.quant, Some(expected));
    }

    std::fs::write(
        fixture.path().join("split_model.json"),
        r#"{"quantized":false,"quantization_bits":8}"#,
    )
    .unwrap();
    assert_eq!(
        resolved_video_numeric_tier("ltx_2_3", &spec())
            .unwrap()
            .quant,
        None,
        "quantized:false is a dense bf16 checkpoint even when a stale bits field remains"
    );
}

#[test]
fn ltx_video_tier_assertion_mismatch_fails_closed() {
    let fixture = tempfile::tempdir().expect("ltx split fixture");
    std::fs::write(
        fixture.path().join("split_model.json"),
        r#"{"quantized":true,"quantization_bits":8}"#,
    )
    .unwrap();
    let asserted_q4 = LoadSpec::new(WeightsSource::Dir(fixture.path().to_owned()))
        .with_quant(gen_core::Quant::Q4);
    let error = resolved_video_numeric_tier("ltx_2_3", &asserted_q4)
        .expect_err("a q4 assertion may not price a q8 checkpoint")
        .to_string();
    assert!(error.contains("disagrees"));
}

/// sc-18237: every shipped Qwen binding must describe a load shape the production route can
/// execute. Native Qwen is deliberately deferred under both Resident and Sequential policies;
/// q8 and the SC-18353 bf16/q4 ladder coordinates were captured under that exact
/// materialization contract, while the old eager BF16/Q4 records remain historical corpus
/// entries only.
/// This is deliberately mutation-sensitive: adding any eager binding makes the production-shape
/// assertion red. It is NOT sensitive to a binding being absent (sc-22512, E8) — a coordinate
/// nobody measured simply contributes nothing here and is priced analytically at admission.
#[test]
fn shipped_qwen_bindings_are_producible_by_the_production_deferred_route() {
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let Some(qwen) = manifest["models"]
        .as_array()
        .and_then(|models| models.iter().find(|model| model["id"] == "qwen_image"))
        .and_then(Value::as_object)
    else {
        return;
    };
    // A model that declares NO MLX calibration opt-in has no shipped bindings to shape-check.
    let Some(bindings) =
        MlxCalibrationBinding::from_manifest(qwen).expect("Qwen calibration bindings are valid")
    else {
        return;
    };

    // sc-22512: no pinned binding COUNT and no lane-declaration gate. Both reddened purely
    // because something was absent. What is asserted instead is the shape of whatever the
    // manifest declares — a claim that holds at any binding count, zero included.
    let mut coordinates = bindings
        .iter()
        .map(|binding| {
            format!(
                "{}|{:?}|{}|{}|{:?}",
                binding.tier, binding.rung, binding.mode, binding.overlay, binding.geometry
            )
        })
        .collect::<Vec<_>>();
    coordinates.sort();
    let declared_count = coordinates.len();
    coordinates.dedup();
    assert_eq!(
        coordinates.len(),
        declared_count,
        "the shipped opt-in must not declare the same (tier, rung, mode, overlay, geometry) \
             coordinate twice"
    );
    let declared = shipped_mlx_declared_closure_digest("qwen_image");
    assert!(bindings.iter().all(|binding| {
        binding.query.abi == sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI
                && binding.provider == "qwen_image"
                && ["bf16", "q4", "q8"].contains(&binding.tier.as_str())
                && binding.mode == "text_to_image"
                && binding.overlay == "none"
                && binding.geometry
                    == CalibrationGeometry {
                        width: 1024,
                        height: 1024,
                        batch: 1,
                        frames: 1,
                    }
                // A model with no declared closure has no currency term to be measured against
                // (sc-22512): skip the comparison rather than red on its absence.
                && declared
                    .as_deref()
                    .is_none_or(|digest| binding.query.inference_closure_digest == digest)
    }));
    assert!(bindings
        .iter()
        .all(|binding| { binding.query.load_shape == LoadShapeKey::DeferredMaterialization }));

    // No tier names the same rung twice. Holds for any subset of the ladder, including none —
    // which is the point: an absent rung is an unmeasured coordinate, never a failure.
    for tier in bindings
        .iter()
        .map(|binding| binding.tier.as_str())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let mut rungs = bindings
            .iter()
            .filter(|binding| binding.tier == tier)
            .map(|binding| format!("{:?}", binding.rung))
            .collect::<Vec<_>>();
        rungs.sort();
        let declared_rungs = rungs.len();
        rungs.dedup();
        assert_eq!(
            rungs.len(),
            declared_rungs,
            "{tier} declares the same rung twice"
        );
    }
}

/// sc-18408: the audited-model set is DERIVED from the manifest — every model declaring
/// `mlx.calibrations` is audited, so a new declaration (flux2_dev arrived via PR #2221 while
/// the old hand list silently ignored it) is covered the moment it ships. Each binding's
/// expected shape is COMPUTED by calling the production shaping functions for that lane —
/// `image_jobs::apply_measured_mlx_load_shape` (the entry shaping that opts Qwen/Lens/plain
/// Krea/plain SDXL into the deferred load-exact contract), then either
/// `apply_residency_policy` (the generator-cache cold-load Resident branch; the fixture path
/// has unmeasurable weights, which always admits Resident and leaves the spec unchanged) or
/// `with_selected_sequential_shape` (the Sequential branch that couples Z-Image's deferred
/// shape to rung 4). No arm hand-writes a shape or leans on the `LoadSpec` struct default, so
/// a flipped production route reds here instead of staying green against a stale literal.
/// macOS-only because the entry shaping IS the macOS MLX route; off-Mac there is no MLX
/// cold-load path for a binding to describe.
#[test]
#[cfg(target_os = "macos")]
fn every_shipped_audited_mlx_binding_has_a_producible_production_load_shape() {
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let models = manifest["models"].as_array().expect("models array");

    let mut audited: Vec<String> = Vec::new();
    for model in models {
        let model = model.as_object().expect("model entry is an object");
        let model_id = model["id"].as_str().expect("model id");
        let Some(bindings) = MlxCalibrationBinding::from_manifest(model)
            .unwrap_or_else(|error| panic!("{model_id} bindings parse: {error}"))
        else {
            continue;
        };
        audited.push(model_id.to_owned());
        for binding in bindings {
            let provider = binding.provider.clone();
            let control_lane = provider == format!("{model_id}_control");
            assert!(
                control_lane || provider == model_id,
                "{model_id} declares an mlx.calibrations binding for provider {provider}, \
                     which this audit cannot map to a production lane of the model (base or \
                     `_control`) — teach the audit that lane's production route before shipping \
                     the binding, or its load shape goes unchecked"
            );

            // The lane's entry spec, characterized the way the production route builds it:
            // a directory-native load at the binding's tier, with the control overlay set for
            // a `_control` provider (which is exactly what keeps the entry shaping from
            // opting the base model's plain-text-to-image deferred contract into this lane).
            let mut spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from("fixture")));
            spec = match binding.tier.as_str() {
                "q4" => spec.with_quant(gen_core::Quant::Q4),
                "q8" => spec.with_quant(gen_core::Quant::Q8),
                "bf16" => spec,
                other => panic!("{model_id} binding names unaudited tier {other}"),
            };
            if control_lane {
                spec = spec.with_control(WeightsSource::File(std::path::PathBuf::from(
                    "fixture-control",
                )));
            }
            let entry = crate::image_jobs::apply_measured_mlx_load_shape(&provider, spec);
            let expected = match binding.rung {
                // Rung 4 is reached through the cold-load branch that selects Sequential
                // residency; every lower rung rides the Resident branch.
                StrategyRung::BoundedTransformerResidency => {
                    with_selected_sequential_shape(&provider, entry).load_shape
                }
                _ => {
                    apply_residency_policy(entry, &provider)
                        .expect("unmeasurable fixture weights always admit Resident")
                        .load_shape
                }
            };
            let actual = crate::memory_strategy::load_shape_from_receipt(binding.query.load_shape);
            assert_eq!(
                actual, expected,
                "{model_id} {provider} {:?} binding names a shape no production cold-load \
                     branch produces",
                binding.rung
            );
        }
    }

    // The anti-vacuity guard, derived instead of pinned (sc-22512).
    //
    // This used to name four models — `qwen_image`, `z_image_turbo`, `krea_2_turbo`,
    // `flux2_dev` — and require each to appear in `audited`. That reddened whenever one of them
    // stopped DECLARING an opt-in, which is a measurement/declaration absence and not a defect:
    // retiring a calibration, or shipping a catalog that has not been captured yet, broke the
    // build over bookkeeping.
    //
    // The real question — "did the manifest loader stop seeing bindings?" — is answered without
    // any roster by comparing the audit against the manifest's OWN declaration set. Every entry
    // that declares `mlx.calibrations` must have been audited, and nothing else may have been.
    // That still fails loudly on a loader that reports "no bindings" for everything (the
    // declarations are visible here but the audit is empty), and it holds at any number of
    // declaring models, zero included.
    let declaring = manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .filter(|model| {
            model
                .get("mlx")
                .and_then(|mlx| mlx.get("calibrations"))
                .and_then(Value::as_array)
                .is_some_and(|calibrations| !calibrations.is_empty())
        })
        .filter_map(|model| model.get("id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        audited
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        declaring,
        "the audited set must be exactly the entries declaring mlx.calibrations; a mismatch \
             means the manifest loader and this audit disagree about what is declared"
    );
}

#[test]
fn shipped_decode_quality_corpus_is_exact_and_drives_the_physical_planner() {
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let models = manifest["models"].as_array().expect("manifest models");
    // Which routes ship a P9 corpus, and the family each resolves through. The per-model
    // (rows, admitted) pair that used to sit here, and the 69/69 total below it, were populations
    // of the frozen corpus: a re-collection that widened or narrowed a sweep failed here with
    // nothing to say about what changed, and the only available repair was to renew the numbers.
    // What the pair was standing in for is asserted directly instead — one row per distinct
    // geometry, every advertised resolution covered, and every shipped row admitted — which holds
    // however many coordinates a re-collection sweeps. `expected` stays exhaustive because the
    // final loop refuses P9 evidence on any route missing from it.
    let expected = [
        ("chroma1_base", "chroma"),
        ("chroma1_flash", "chroma"),
        ("chroma1_hd", "chroma"),
        ("illustrious_xl_v1", "sdxl"),
        ("illustrious_xl_v2", "sdxl"),
        ("kolors", "kolors"),
        ("realvisxl", "sdxl"),
        ("realvisxl_lightning", "sdxl"),
        ("sdxl", "sdxl"),
    ];
    let expected_ids = expected
        .iter()
        .map(|(id, ..)| *id)
        .collect::<std::collections::BTreeSet<_>>();

    for (model_id, family) in expected {
        let model = models
            .iter()
            .find(|model| model["id"] == model_id)
            .unwrap_or_else(|| panic!("shipped {model_id} manifest entry"));
        let model_object = model.as_object().expect("model object");
        let mlx = model["mlx"].as_object().expect("mlx object");
        let implementations = mlx["memoryStrategyContract"]["implementations"]
            .as_array()
            .expect("memory strategy implementations");
        let bounded = implementations
            .iter()
            .filter(|implementation| implementation["rung"] == "bounded_decode")
            .collect::<Vec<_>>();
        assert_eq!(
            bounded.len(),
            1,
            "{model_id}: one route-local decode declaration"
        );
        assert_eq!(bounded[0]["tiers"], serde_json::json!(["q4"]));
        assert_eq!(bounded[0]["modes"], serde_json::json!(["text_to_image"]));
        assert_eq!(bounded[0]["overlays"], serde_json::json!(["none"]));

        let rows = decode_quality_policies_from_manifest(model_object, model_id)
            .unwrap_or_else(|error| panic!("{model_id}: shipped quality corpus: {error}"));
        assert!(
            !rows.is_empty(),
            "{model_id}: sealed quality corpus is empty"
        );
        assert!(rows.iter().all(|row| {
            row.resolved_route == model_id
                && row.family == family
                && row.backend == gen_core::MemoryBackend::Mlx
                && row.tier.quant == Some(gen_core::Quant::Q4)
                && row.mode == MemoryMode::TextToImage
                && row.overlay.is_none()
                && !row.use_pid
                && row.geometry.batch == 1
                && row.geometry.frames == 1
                && row.geometry.reference_count == 0
                && row.implementation_fingerprint.len() == 64
                && row
                    .implementation_fingerprint
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        }));
        // Every shipped row is admitted evidence. A refused row on a route that still advertises
        // its coordinate would advertise a geometry the physical planner then declines to bound —
        // the condition the per-model admitted count was standing in for, stated as the property.
        assert!(
            rows.iter()
                .all(|row| matches!(row.disposition, MemoryDecodeQualityDisposition::Admitted)),
            "{model_id}: the sealed corpus must carry only admitted decode evidence"
        );

        let q4 = model["downloads"]
            .as_array()
            .expect("downloads")
            .iter()
            .find(|download| download["variant"] == "q4")
            .unwrap_or_else(|| panic!("{model_id}: q4 artifact"));
        assert!(rows.iter().all(|row| {
            row.artifact.repository == q4["repo"].as_str().unwrap()
                && row.artifact.revision == q4["revision"].as_str().unwrap()
                && row.artifact.variant == "q4"
                && row.artifact.fingerprint
                    == format!(
                        "{}@{}:q4",
                        q4["repo"].as_str().unwrap(),
                        q4["revision"].as_str().unwrap()
                    )
        }));
        let expected_shape = if family == "sdxl" {
            gen_core::LoadShape::DeferredMaterialization
        } else {
            gen_core::LoadShape::EagerMaterialization
        };
        assert!(rows.iter().all(|row| row.load_shape == expected_shape));

        let measured_resolutions = rows
            .iter()
            .map(|row| format!("{}x{}", row.geometry.width, row.geometry.height))
            .collect::<std::collections::BTreeSet<_>>();
        // Several admitted tile choices may share a geometry; the exact policy key must
        // remain unique, and every choice is independently exercised below.
        let keys = rows
            .iter()
            .map(|row| {
                (
                    row.geometry.width,
                    row.geometry.height,
                    row.tile_edge,
                    row.overlap,
                    row.use_pid,
                )
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            rows.len(),
            keys.len(),
            "{model_id}: duplicate physical decode policy"
        );
        for resolution in model["limits"]["resolutions"]
            .as_array()
            .expect("advertised resolutions")
        {
            assert!(
                measured_resolutions.contains(resolution.as_str().unwrap()),
                "{model_id}: advertised {resolution} must have an exact admitted-or-refused row"
            );
        }

        let mut generator = fixture_generator();
        let contract = generator.contract.as_mut().expect("fixture contract");
        contract.provider_id = model_id.to_owned();
        contract.load_shape = expected_shape;
        contract.calibration = None;
        contract
            .adopt_decode_geometry_policies(family, rows.clone())
            .unwrap_or_else(|error| panic!("{model_id}: adopt shipped corpus: {error}"));
        assert!(
            contract.conformance_errors().is_empty(),
            "{model_id}: {:?}",
            contract.conformance_errors()
        );
        let mut plan = fixture_plan();
        plan.engine_id = model_id;
        plan.model_id = model_id.to_owned();

        // EVERY admitted row, not just the first one found. With the admitted count gone this is
        // what carries the coverage claim, and it is strictly stronger than the count ever was: a
        // row that stopped reaching the planner is now a failure rather than an unvisited element
        // the count still added up correctly.
        for row in rows
            .iter()
            .filter(|row| matches!(row.disposition, MemoryDecodeQualityDisposition::Admitted))
        {
            let result = synthesize_estimate_ladder(
                contract,
                &plan,
                "text_to_image",
                None,
                row.geometry,
                false,
                None,
                &[],
            );
            assert!(
                result.estimates.iter().any(|candidate| {
                    candidate.selection.strategy == MemoryStrategy::BoundedDecode
                        && candidate.selection.parameters.decode_tile_edge == Some(row.tile_edge)
                        && candidate.selection.parameters.decode_overlap == Some(row.overlap)
                }),
                "{model_id}: admitted {}x{} row must create its exact physical decode candidate",
                row.geometry.width,
                row.geometry.height
            );
        }
        let unmeasured_geometry = MemoryGeometry {
            width: 64,
            height: 64,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let result = synthesize_estimate_ladder(
            contract,
            &plan,
            "text_to_image",
            None,
            unmeasured_geometry,
            false,
            None,
            &[],
        );
        assert!(
            result
                .estimates
                .iter()
                .all(|candidate| { candidate.selection.strategy != MemoryStrategy::BoundedDecode }),
            "{model_id}: an unmeasured coordinate must not create a physical decode candidate"
        );
        let matched = contract
            .decode_geometry_policies_for_request(MemoryDecodePolicyQuery {
                tier: plan.tier,
                load_shape: contract.load_shape,
                mode_key: "text_to_image",
                overlay: None,
                geometry: unmeasured_geometry,
                use_pid: false,
            })
            .expect("exact policy query");
        assert!(matched.is_empty(), "{model_id}: unmeasured policy query");
    }

    // `quality_stamps.len() == 10` used to sit here, over `implementation_fingerprint`. That axis
    // is a FORENSIC stamp, not an admission gate: gen-core seals it into
    // `production_evidence_sha256`, so it cannot be edited after collection, and inference sc-19728
    // deliberately removed the comparison against the running build because a closure spanning
    // whole crates plus `Cargo.lock` made every dependency bump restamp every row. Counting the
    // distinct stamps re-created exactly that — a population any re-collection invalidates, over a
    // value the production path already refuses to accept unsealed. The per-row shape check above
    // (64 lowercase hex) is the part that is ours to make; the binding is gen-core's, and the
    // `adopt_decode_geometry_policies` call above is where it fails closed.
    for model in models {
        let id = model["id"].as_str().expect("model id");
        if !expected_ids.contains(id) {
            let count = model["mlx"]["memoryStrategyContract"]["implementations"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|implementation| {
                    implementation["parameterRanges"]
                        .get("decodeGeometryPolicies")
                        .is_some()
                })
                .count();
            assert_eq!(
                count, 0,
                "{id}: unrelated route must not inherit P9 evidence"
            );
        }
    }
}

/// sc-18408 item (d): every MLX plan row must resolve through the shipped registry to a
/// weights-free provider contract that IMPLEMENTS its planned anchor rung. The walk itself is
/// lane-generic and lives in `inference_runtime` (sc-22736) so the candle lane runs the same
/// one under its own cfg; this is the MLX call.
#[test]
#[cfg(target_os = "macos")]
fn every_planned_mlx_lane_resolves_a_weights_free_provider_contract() {
    crate::inference_runtime::every_planned_lane_row_resolves_a_weights_free_contract_implementing_its_rung(
            "mlx",
            gen_core::MemoryBackend::Mlx,
        );
}

/// SC-16915 acceptance: the SHIPPED manifest opt-in and the SHIPPED evidence bundle must agree
/// well enough for a covered cell to take the calibrated path. Every other test in this module
/// builds its own fixture manifest and fixture bundle, so all of them stay green even when the
/// two real artefacts have drifted apart — which is exactly the state this story had to repair.
/// This one reads both real files and nothing else.
///
/// The manifest now ships only the q8 pair re-captured under the production deferred route.
/// BF16/Q4 continue through the estimate ladder until production-shaped measurements exist.
/// The count assertion makes a dropped q8 record fail rather than hiding behind its sibling.
///
/// sc-17774 split this into the two questions it had been conflating. AGREEMENT — do the two
/// shipped artefacts describe the same measurements — is graded at the closure they were both
/// captured under, and is therefore true regardless of where the pin has since moved. sc-22738
/// retired the CURRENCY half entirely: the admission route takes no closure argument any more,
/// so there is no "at the live closure" evaluation to contrast with. What survives here is the
/// agreement claim plus the config-consistency claim that the shipped opt-in names ONE captured
/// closure.
#[test]
fn shipped_qwen_manifest_and_packaged_evidence_agree_at_their_captured_closure() {
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let entry = manifest
        .get("models")
        .and_then(Value::as_array)
        .expect("models array")
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some("qwen_image"))
        .and_then(Value::as_object)
        .expect("qwen_image manifest entry");
    // sc-22512: an absent opt-in is not a failure — there is simply nothing to compare. The
    // `.expect("qwen_image declares mlx.calibrations")` that stood here made the absence of a
    // declaration panic before the graceful skip below could ever be reached.
    let Some(calibrations) = entry
        .get("mlx")
        .and_then(|mlx| mlx.get("calibrations"))
        .and_then(Value::as_array)
    else {
        return;
    };
    // sc-22512: with no declared opt-in there is no captured closure to agree on, so this half
    // of the test has no question to ask. Skipping is the E8 posture; failing would be a
    // measurement-absence gate.
    let Some(declared) = shipped_mlx_declared_closure_digest("qwen_image") else {
        return;
    };
    // The helper asserts uniformity across the bindings; this pins the shape so a truncated or
    // placeholder digest cannot satisfy it.
    assert_eq!(
        declared.len(),
        64,
        "a captured inference closure digest is a sha256 hex string"
    );

    let tier = "q8";
    let quant = Some(gen_core::Quant::Q8);
    let expected_rungs = 2_usize;
    // Filled from the unmutated route below, so the load-shape mutation check can compare
    // against the records the SHIPPED opt-in actually resolves rather than a restated list.
    let declared_record_ids: Vec<String>;
    {
        // Take the request identity from the opt-in itself rather than restating it, so the
        // test cannot drift from the manifest it is checking.
        let binding = calibrations
            .iter()
            .find(|item| item.get("tier").and_then(Value::as_str) == Some(tier))
            .unwrap_or_else(|| panic!("{tier} is part of the shipped opt-in"));
        let text = |key: &str| {
            binding
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("{tier} binding is missing {key}"))
                .to_owned()
        };
        let geometry = binding.get("geometry").expect("binding geometry");
        let dimension = |key: &str| {
            u32::try_from(
                geometry
                    .get(key)
                    .and_then(Value::as_u64)
                    .unwrap_or_else(|| panic!("geometry is missing {key}")),
            )
            .expect("geometry fits u32")
        };

        let mut spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(format!(
            "/cache/models--SceneWorks--qwen-image-mlx/snapshots/x/{tier}"
        ))));
        if let Some(quant) = quant {
            spec = spec.with_quant(quant);
        }
        let plan = MlxRequestPlan::for_spec_and_manifest(
            "qwen_image",
            "qwen_image",
            &spec,
            Some(entry),
            Some(ResolvedArtifactProvenance {
                identity: crate::model_jobs::ResolvedArtifactIdentity {
                    repository: text("artifactRepository"),
                    revision: text("artifactResolvedRevision"),
                    variant: text("artifactVariant"),
                    fingerprint: text("resolvedPathFingerprint"),
                },
                fixed_artifact_tier: Some(tier.to_owned()),
            }),
        );
        assert!(
            matches!(plan.calibration, MlxCalibrationConfig::Valid(_)),
            "{tier}: the shipped opt-in must parse as a valid binding set"
        );

        let mut inputs = fixture_inputs(dimension("width"), dimension("height"));
        inputs.overlay = None;
        // Graded at the closure both artefacts were captured under. This is the agreement claim
        // and it does not expire: whether the manifest opt-in and the bundle describe the same
        // measurements is a fact about the two files, not about the pin.
        let route = packaged_admission_route(&plan, &inputs, &text("mode"), fixture_budget(128.0))
            .expect("a covered cell must not error");
        assert_eq!(
            route.path,
            AdmissionPath::Evidence,
            "{tier}: shipped manifest + shipped evidence must reach calibrated admission at \
                 the closure they were captured under; got fallback {:?}",
            route.fallback_reason
        );
        assert_eq!(
            route.evidence.len(),
            expected_rungs,
            "{tier}: every declared rung must resolve to a promoted record, not just one"
        );
        assert!(
            route
                .evidence
                .iter()
                .all(|candidate| !candidate.record_id.is_empty()),
            "{tier}: each candidate names the exact record backing it"
        );

        declared_record_ids = route
            .evidence
            .iter()
            .map(|candidate| candidate.record_id.clone())
            .collect();

        // sc-22738 removed the second half of this block. It used to re-run the same route at
        // the LIVE closure to prove that a moved closure still admitted; the route no longer
        // takes a closure at all, so the re-run would be a byte-identical duplicate of the
        // evaluation above rather than a second question.
    }

    // Mutation check on the `loadShape` axis. Asserting only the route above is a FALSE GREEN:
    // the route matches whichever binding fits the request, so flipping every declared shape
    // must be shown to change WHICH measurements back the request.
    //
    // Before sc-22738 the flip degraded to `Legacy`/`IdentityMismatch`, because the eager q8
    // records the flipped opt-in matches were captured under a different inference closure and
    // the route compared closures. It no longer does — a measurement is used as measured — so
    // the flipped opt-in resolves the EAGER receipts instead, and the axis is graded on the
    // record set rather than on the path.
    let q8 = calibrations
        .iter()
        .find(|item| item.get("tier").and_then(Value::as_str) == Some("q8"))
        .expect("q8 binding");
    let text = |key: &str| {
        q8.get(key)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("binding is missing {key}"))
            .to_owned()
    };
    let mut mutated = entry.clone();
    for calibration in mutated
        .get_mut("mlx")
        .and_then(|mlx| mlx.get_mut("calibrations"))
        .and_then(Value::as_array_mut)
        .expect("calibrations array")
    {
        let shape = calibration
            .get("loadShape")
            .and_then(Value::as_str)
            .expect("every binding declares a loadShape");
        calibration["loadShape"] = Value::from(match shape {
            "eager_materialization" => "deferred_materialization",
            _ => "eager_materialization",
        });
    }
    let spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
        "/cache/models--SceneWorks--qwen-image-mlx/snapshots/x/q8",
    )))
    .with_quant(gen_core::Quant::Q8);
    let mut inputs = fixture_inputs(1024, 1024);
    inputs.overlay = None;
    let mutated_route = packaged_admission_route(
        &MlxRequestPlan::for_spec_and_manifest(
            "qwen_image",
            "qwen_image",
            &spec,
            Some(&mutated),
            Some(ResolvedArtifactProvenance {
                identity: crate::model_jobs::ResolvedArtifactIdentity {
                    repository: text("artifactRepository"),
                    revision: text("artifactResolvedRevision"),
                    variant: text("artifactVariant"),
                    fingerprint: text("resolvedPathFingerprint"),
                },
                fixed_artifact_tier: Some("q8".to_owned()),
            }),
        ),
        &inputs,
        "text_to_image",
        fixture_budget(128.0),
    )
    .expect("a load-shape mutation degrades or re-resolves, it never errors");
    assert!(
        !declared_record_ids.is_empty(),
        "precondition: the shipped opt-in resolved records to compare against"
    );
    let mutated_record_ids = mutated_route
        .evidence
        .iter()
        .map(|candidate| candidate.record_id.as_str())
        .collect::<Vec<_>>();
    assert!(
            mutated_record_ids
                .iter()
                .all(|id| !declared_record_ids.iter().any(|shipped| shipped == id)),
            "flipping every declared loadShape must stop the SHIPPED receipts from backing the \
             request — the shape is a real selector, not decoration. shipped={declared_record_ids:?} \
             mutated={mutated_record_ids:?}"
        );
}

/// The krea half of the same guarantee. `packaged_krea_plan` synthesizes its bindings FROM the
/// evidence records, so it is current-by-construction and structurally cannot notice the
/// shipped krea manifest disagreeing with the shipped bundle. krea_2_turbo_control is a covered
/// provider of sc-16915, so it gets the same real-manifest × real-evidence route check qwen has
/// — the agreement claim, which sc-22738 left as the only half there is: the admission route no
/// longer takes a closure argument, so there is no live-closure currency evaluation to contrast
/// it with.
#[test]
fn shipped_krea_manifest_and_packaged_evidence_agree_at_their_captured_closure() {
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    // sc-22512 (E8): the entry and its opt-in are LOOKED UP, not required. A catalog that has
    // retired `krea_2_turbo`, or that ships it with no `mlx.calibrations` block, is a model
    // nobody measured on this lane — there is nothing to grade currency against, so the harness
    // withholds its question instead of reddening. The `calibrations.len() == 2` pin went with
    // them for the same reason: it froze the shipped opt-in at the 768²/896² pose-control pair,
    // so ADDING a third measured cell reddened the suite. Every agreement assertion below is
    // universally quantified over whatever bindings are declared and keeps full force on them.
    let Some(entry) = manifest
        .get("models")
        .and_then(Value::as_array)
        .expect("models array")
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some("krea_2_turbo"))
        .and_then(Value::as_object)
    else {
        return;
    };
    let Some(calibrations) = entry
        .get("mlx")
        .and_then(|mlx| mlx.get("calibrations"))
        .and_then(Value::as_array)
    else {
        return;
    };
    let Some(declared) = shipped_mlx_declared_closure_digest("krea_2_turbo") else {
        return;
    };
    // The helper asserts the bindings name ONE captured closure; this pins its shape so a
    // truncated or placeholder digest cannot satisfy the consistency claim.
    assert_eq!(
        declared.len(),
        64,
        "a captured inference closure digest is a sha256 hex string"
    );

    for binding in calibrations {
        let text = |key: &str| {
            binding
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("krea binding is missing {key}"))
                .to_owned()
        };
        let geometry = binding.get("geometry").expect("binding geometry");
        let dimension = |key: &str| {
            u32::try_from(
                geometry
                    .get(key)
                    .and_then(Value::as_u64)
                    .expect("dimension"),
            )
            .expect("geometry fits u32")
        };
        let spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
            "/cache/models--SceneWorks--krea-2-turbo-mlx/snapshots/x/q4",
        )))
        .with_quant(gen_core::Quant::Q4);
        // `packaged_admission_route` filters on `binding.provider == plan.engine_id`, so the
        // engine id is the PROVIDER (`krea_2_turbo_control`) and the model id is the catalog
        // entry (`krea_2_turbo`) — the order production uses in `image_jobs/krea_control.rs`,
        // which passes `KREA_CONTROL_ENGINE_ID` then `&request.model`. Swapping them yields
        // `StaleIdentity` and a silent legacy fallback.
        let plan = MlxRequestPlan::for_spec_and_manifest(
            "krea_2_turbo_control",
            "krea_2_turbo",
            &spec,
            Some(entry),
            Some(ResolvedArtifactProvenance {
                identity: crate::model_jobs::ResolvedArtifactIdentity {
                    repository: text("artifactRepository"),
                    revision: text("artifactResolvedRevision"),
                    variant: text("artifactVariant"),
                    fingerprint: text("resolvedPathFingerprint"),
                },
                fixed_artifact_tier: Some(text("tier")),
            }),
        );
        assert!(
            matches!(plan.calibration, MlxCalibrationConfig::Valid(_)),
            "the shipped krea opt-in must parse as a valid binding set"
        );

        let mut inputs = fixture_inputs(dimension("width"), dimension("height"));
        inputs.overlay = Some(text("overlay"));
        inputs.has_reference = true;
        inputs.reference_count = 1;
        let route = packaged_admission_route(&plan, &inputs, &text("mode"), fixture_budget(128.0))
            .expect("a covered krea cell must not error");
        assert_eq!(
            route.path,
            AdmissionPath::Evidence,
            "{}x{}: shipped krea manifest + evidence must reach calibrated admission at the \
                 closure they were captured under; got fallback {:?}",
            dimension("width"),
            dimension("height"),
            route.fallback_reason
        );

        // sc-22738 retired the second evaluation that stood here. `mlx:krea_2_turbo_control`'s
        // closure spans `mlx-gen` and five sibling provider crates, so it moves on work that
        // has nothing to do with Krea — and the route no longer reads it at all, so re-running
        // at the live closure would be a byte-identical duplicate rather than a claim.
    }
}

/// SC-16915 acceptance, end to end on real weights (ignored — needs the qwen-image-mlx bf16
/// turnkey, ~57 GB, and peaks around 59 GiB active).
///
/// The non-gated `shipped_qwen_manifest_and_packaged_evidence_select_the_calibrated_path`
/// proves the shipped manifest and bundle agree. This closes the story's remaining acceptance
/// line — "calibrated admission selects verified rungs again on a real MLX render" — by driving
/// the production seam `image_jobs/base.rs` calls per generation, against a LIVE loaded
/// provider, and then rendering.
///
/// The load-bearing assertion is `process_limit_bytes.is_some()`. That ceiling is derived only
/// from an exact verified cell; a legacy admission leaves it `None` and falls back to the
/// process-global limit. So it distinguishes "the request was admitted" — which the legacy
/// estimator also does — from "an exact verified rung was selected", which is what regressed.
///
///   cargo test -p sceneworks-worker --lib -- --ignored --nocapture \
///     qwen_real_install_selects_a_verified_rung
#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight MLX render; needs the qwen-image-mlx bf16 turnkey (~57 GB)"]
fn qwen_real_install_selects_a_verified_rung_and_renders() {
    let hub = std::path::PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".cache/huggingface/hub");
    let snapshots = hub.join("models--SceneWorks--qwen-image-mlx/snapshots");
    let Some(bf16) = std::fs::read_dir(&snapshots).ok().and_then(|entries| {
        entries
            .flatten()
            .map(|entry| entry.path().join("bf16"))
            .find(|path| path.join("model_index.json").is_file())
    }) else {
        eprintln!(
            "SKIP: no qwen-image-mlx bf16 turnkey under {}",
            snapshots.display()
        );
        return;
    };

    // The SHIPPED manifest entry, not a fixture: the opt-in under test is the one that ships.
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let entry = manifest
        .get("models")
        .and_then(Value::as_array)
        .expect("models array")
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some("qwen_image"))
        .and_then(Value::as_object)
        .expect("qwen_image manifest entry")
        .clone();

    // Provenance is derived from the snapshot actually being loaded, NOT copied from the app's
    // install receipt — that receipt can name a different variant than the one under test.
    // Reading the revision out of the resolved snapshot path and asserting it against the
    // manifest's declared identity is the stronger check anyway: it proves the bytes on disk
    // are the ones the opt-in names, which is the whole job of provenance.
    let revision = bf16
        .parent()
        .and_then(|snapshot| snapshot.file_name())
        .and_then(|name| name.to_str())
        .expect("snapshot directory is <hub>/models--.../snapshots/<revision>/bf16")
        .to_owned();
    let binding = entry
        .get("mlx")
        .and_then(|mlx| mlx.get("calibrations"))
        .and_then(Value::as_array)
        .expect("qwen_image declares mlx.calibrations")
        .iter()
        .find(|item| item.get("tier").and_then(Value::as_str) == Some("bf16"))
        .expect("a bf16 binding");
    let declared = |key: &str| {
        binding
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("binding is missing {key}"))
            .to_owned()
    };
    assert_eq!(
        revision,
        declared("artifactResolvedRevision"),
        "the on-disk snapshot must be the exact artifact the shipped opt-in names"
    );
    let provenance = ResolvedArtifactProvenance {
        identity: crate::model_jobs::ResolvedArtifactIdentity {
            repository: declared("artifactRepository"),
            revision,
            variant: declared("artifactVariant"),
            fingerprint: declared("resolvedPathFingerprint"),
        },
        fixed_artifact_tier: Some(declared("tier")),
    };

    let spec = LoadSpec::new(WeightsSource::Dir(bf16.clone()));
    let plan = MlxRequestPlan::for_spec_and_manifest(
        "qwen_image",
        "qwen_image",
        &spec,
        Some(&entry),
        Some(provenance),
    );
    let inputs = fixture_inputs(1024, 1024);

    // Cheap pre-load check, so a routing regression fails before a 57 GB load.
    let route = packaged_admission_route(&plan, &inputs, "text_to_image", fixture_budget(128.0))
        .expect("covered cell must not error");
    assert_eq!(
        route.path,
        AdmissionPath::Evidence,
        "shipped opt-in must route to calibrated admission; got fallback {:?}",
        route.fallback_reason
    );

    eprintln!(
        "[acceptance] loading qwen_image bf16 from {}",
        bf16.display()
    );
    let generator = crate::inference_runtime::load("qwen_image", &spec).expect("load qwen_image");

    // THE production call: exactly what base.rs runs per generation, on a live generator.
    let evaluation = evaluate_request(
        &*generator,
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        0,
    )
    .expect("a covered qwen cell must be admitted");
    eprintln!("[acceptance] gate admitted: {evaluation:?}");
    assert!(
        evaluation.process_limit_bytes.is_some(),
        "a verified rung supplies a request-scoped ceiling; `None` means this degraded to the \
             legacy estimator, which is exactly the regression this story repaired"
    );

    let request = gen_core::GenerationRequest {
        prompt: "a red fox resting beside a blue ceramic vase, studio photograph".to_owned(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(15511),
        steps: Some(2),
        memory: Some(evaluation.memory),
        ..Default::default()
    };
    let image = match generator
        .generate(&request, &mut |_| {})
        .expect("real qwen render must succeed under the calibrated ceiling")
    {
        gen_core::GenerationOutput::Images(mut images) => {
            assert_eq!(images.len(), 1, "one requested image");
            images.pop().expect("one image")
        }
        other => panic!("qwen_image must return images, got {other:?}"),
    };
    assert!(
        image.width == 1024 && image.height == 1024,
        "rendered geometry must match the verified cell"
    );
    eprintln!(
        "[acceptance] rendered {}x{} under a verified rung",
        image.width, image.height
    );
}

/// Admission on a REAL shipped lane. SC-19753 captured all five Z-Image rungs at the inference
/// closure that was live when it ran; the epic's pin has since advanced past it, so this ladder
/// is now an ACCEPTED FLOOR rather than current verification. A pin bump staling calibration
/// records is the fail-closed design working, not a re-capture work order.
///
/// Renamed from `..._admits_current_mlx_ladder_rungs` because "current" is the one thing it no
/// longer proves. What it still proves exactly, and what keeps it from going vacuous:
///   * the manifest binding and the packaged evidence agree with each other on the digest;
///   * the ladder reaches calibrated admission on all five rungs;
///   * sc-22738: a binding that names a DIFFERENT closure than the record it resolves to routes
///     identically — the runtime keeps behaving as if the measurement were valid, and the moved
///     digest is a re-capture signal for the probe tooling alone.
///
/// Deliberately NOT asserted: that a moved closure degrades anything. Since sc-22738 the route
/// takes no closure argument and `evidence_for` compares none, so asserting a degradation would
/// encode a mechanism that does not exist.
#[test]
fn shipped_z_image_manifest_admits_the_accepted_mlx_ladder_floor() {
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    // sc-22512 (E8): the entry and its opt-in are LOOKED UP, not required. A catalog without
    // `z_image_turbo`, or one shipping it with no exact MLX bindings, is a lane nobody measured
    // — there is no accepted floor to admit — so the harness withholds its question. A PRESENT
    // but malformed binding block still fails (`from_manifest`'s `Err`): that is contradictory
    // data, not missing data. Everything below keeps full force on a declared opt-in.
    let Some(z_image) = manifest["models"]
        .as_array()
        .and_then(|models| models.iter().find(|model| model["id"] == "z_image_turbo"))
        .and_then(Value::as_object)
    else {
        return;
    };
    let Some(bindings) = MlxCalibrationBinding::from_manifest(z_image)
        .expect("Z-Image calibration bindings are valid")
    else {
        return;
    };

    assert_eq!(bindings.len(), 5);
    // The digest the SHIPPED EVIDENCE says this ladder was measured under, read from the bundle
    // rather than restated as a hex literal. A literal here is brittle in a way that matters:
    // `--restamp` legitimately re-derives every digest on a `CLOSURE_DIGEST_VERSION` change
    // without any measurement moving, and a pinned literal turns that no-op into a red test.
    // Reading it back is also STRICTER — it catches the manifest binding drifting away from the
    // record it claims, which a literal cannot see.
    //
    // Looked up rather than required, for the same reason: a bundle carrying no Z-Image record
    // (or a record with no digest) is an unmeasured lane, and the whole remainder of this test
    // is a claim ABOUT that measurement.
    let Some(captured) = packaged_bundle()
        .records
        .into_iter()
        .find(|record| {
            matches!(record.backend, CalibrationBackend::Mlx)
                && record.target.provider == "z_image_turbo"
                && record.calibration_fingerprint == "z-image-mlx-independent-materialization-v4"
        })
        .and_then(|record| record.repositories.inference.closure_digest)
    else {
        return;
    };
    assert!(bindings.iter().all(|binding| {
        binding.query.abi == sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI
                && binding.provider == "z_image_turbo"
                && binding.tier == "q4"
                && binding.mode == "text_to_image"
                && binding.overlay == "none"
                && binding.geometry
                    == CalibrationGeometry {
                        width: 768,
                        height: 768,
                        batch: 1,
                        frames: 1,
                    }
                && binding.query.fingerprint == "z-image-mlx-independent-materialization-v4"
                // Capture provenance remains an exact fact about the shipped opt-in.
                && binding.query.inference_revision == "dfd76b5aef62b2082ed4b18a8eebd2e3e2e07cfb"
                // The manifest binding and the packaged evidence must still agree exactly with each
                // other. This is a config-consistency claim about the two shipped files; sc-22738
                // made it irrelevant to ADMISSION, which is asserted below.
                && binding.query.inference_closure_digest == captured
    }));
    assert!(bindings.iter().all(|binding| {
        binding.query.load_shape
            == if binding.rung == StrategyRung::BoundedTransformerResidency {
                LoadShapeKey::DeferredMaterialization
            } else {
                LoadShapeKey::EagerMaterialization
            }
    }));

    let resolved_binding = &bindings[0].query;
    let resolved = ResolvedArtifactProvenance {
        identity: crate::model_jobs::ResolvedArtifactIdentity {
            repository: resolved_binding.artifact_repository.clone(),
            revision: resolved_binding.artifact_resolved_revision.clone(),
            variant: resolved_binding.artifact_variant.clone(),
            fingerprint: resolved_binding.resolved_path_fingerprint.clone(),
        },
        fixed_artifact_tier: Some("q4".to_owned()),
    };
    let spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
        "/packaged/z-image-turbo-mlx/q4",
    )))
    .with_quant(gen_core::Quant::Q4);
    let plan = MlxRequestPlan::for_spec_and_manifest(
        "z_image_turbo",
        "z_image_turbo",
        &spec,
        Some(z_image),
        Some(resolved),
    );
    let route = packaged_admission_route(
        &plan,
        &fixture_inputs(768, 768),
        "text_to_image",
        fixture_budget(128.0),
    )
    .expect("the accepted Z-Image floor must route without an error");

    assert_eq!(
        route.path,
        AdmissionPath::Evidence,
        "the ladder must reach the selector; got fallback {:?}",
        route.fallback_reason
    );
    assert_eq!(
        route.evidence.len(),
        5,
        "every rung must resolve to its promoted record"
    );

    // sc-22738: a manifest binding naming a closure nothing was measured under is no longer a
    // demotion. `evidence_for` compares artifact identity, fingerprint, shape and geometry —
    // never the closure — so the route is byte-for-byte the route above.
    let mut mismatched = z_image.clone();
    for calibration in mismatched
        .get_mut("mlx")
        .and_then(|mlx| mlx.get_mut("calibrations"))
        .and_then(Value::as_array_mut)
        .expect("calibrations array")
    {
        calibration["inferenceClosureDigest"] = Value::from("0".repeat(64));
    }
    let mismatched_route = packaged_admission_route(
        &MlxRequestPlan::for_spec_and_manifest(
            "z_image_turbo",
            "z_image_turbo",
            &spec,
            Some(&mismatched),
            Some(ResolvedArtifactProvenance {
                identity: crate::model_jobs::ResolvedArtifactIdentity {
                    repository: resolved_binding.artifact_repository.clone(),
                    revision: resolved_binding.artifact_resolved_revision.clone(),
                    variant: resolved_binding.artifact_variant.clone(),
                    fingerprint: resolved_binding.resolved_path_fingerprint.clone(),
                },
                fixed_artifact_tier: Some("q4".to_owned()),
            }),
        ),
        &fixture_inputs(768, 768),
        "text_to_image",
        fixture_budget(128.0),
    )
    .expect("a moved closure digest degrades nothing, and it does not error");
    assert_eq!(
        format!("{mismatched_route:?}"),
        format!("{route:?}"),
        "a binding naming a different closure must route identically — the runtime always \
             behaves as if the measurement were valid (sc-22738)"
    );
}

#[test]
fn resident_evidence_is_keyed_by_the_live_contract_composition() {
    use gen_core::{MemoryPrerequisiteScope, MemoryStrategyPrerequisite};

    let mut contract = MemoryProviderContract::compatibility_default(
        "fixture_provider",
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.additional_prerequisites.push((
        MemoryStrategy::Resident,
        MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));
    contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
        .expect("staged-residency capability")
        .support = gen_core::MemoryStrategySupport::Implemented;
    let (_, evidence) = resident_evidence(
        &contract,
        MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Q4),
            component_precision_floors: &[],
        },
        "text_to_image",
        None,
        MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        1,
        None,
    );
    assert_eq!(
        evidence.key.engaged_composition,
        vec![MemoryStrategy::Resident, MemoryStrategy::StagedResidency,],
        "the test must distinguish live derivation from the old hardcoded resident-only key"
    );
}

struct RequestGenerator {
    descriptor: gen_core::ModelDescriptor,
    contract: Option<MemoryProviderContract>,
}

impl gen_core::Generator for RequestGenerator {
    fn descriptor(&self) -> &gen_core::ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, _req: &gen_core::GenerationRequest) -> gen_core::Result<()> {
        Ok(())
    }

    fn generate(
        &self,
        _req: &gen_core::GenerationRequest,
        _on_progress: &mut dyn FnMut(gen_core::Progress),
    ) -> gen_core::Result<gen_core::GenerationOutput> {
        unreachable!("request selector tests do not execute tensors")
    }

    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        self.contract.as_ref()
    }
}

#[derive(Default)]
struct ReferenceRenderRecord {
    safety: Vec<gen_core::MemoryRunContext>,
    scopes: Vec<gen_core::MemoryRunContext>,
    generated: usize,
    finished: Vec<gen_core::MemoryRunOutcome>,
}

struct ReferenceRenderScope(std::sync::Arc<std::sync::Mutex<ReferenceRenderRecord>>);
impl gen_core::MemoryRequestScope for ReferenceRenderScope {
    fn configure_request(&mut self, _: &mut gen_core::GenerationRequest) -> gen_core::Result<()> {
        Ok(())
    }
    fn enter_phase(&mut self, _: gen_core::MemoryPhase) -> gen_core::Result<()> {
        Ok(())
    }
    fn leave_phase(&mut self, _: gen_core::MemoryPhase) -> gen_core::Result<()> {
        Ok(())
    }
    fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> gen_core::Result<()> {
        Ok(())
    }
    fn configure_attention(&mut self, _: u32) -> gen_core::Result<()> {
        Ok(())
    }
    fn materialize_transformer_window(&mut self, _: u32, _: u32) -> gen_core::Result<()> {
        Ok(())
    }
    fn finish(&mut self, outcome: gen_core::MemoryRunOutcome) -> gen_core::Result<()> {
        self.0.lock().unwrap().finished.push(outcome);
        Ok(())
    }
}

struct ReferenceRenderGenerator {
    descriptor: gen_core::ModelDescriptor,
    record: std::sync::Arc<std::sync::Mutex<ReferenceRenderRecord>>,
}
impl gen_core::Generator for ReferenceRenderGenerator {
    fn descriptor(&self) -> &gen_core::ModelDescriptor {
        &self.descriptor
    }
    fn validate(&self, _: &gen_core::GenerationRequest) -> gen_core::Result<()> {
        Ok(())
    }
    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        self.record.lock().unwrap().safety.push(context.clone());
        gen_core::MemorySafetyDecision::Accept
    }
    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        self.record.lock().unwrap().scopes.push(context.clone());
        Ok(Some(Box::new(ReferenceRenderScope(std::sync::Arc::clone(
            &self.record,
        )))))
    }
    fn generate(
        &self,
        request: &gen_core::GenerationRequest,
        _: &mut dyn FnMut(gen_core::Progress),
    ) -> gen_core::Result<gen_core::GenerationOutput> {
        let mut record = self.record.lock().unwrap();
        assert_eq!(
            record.safety.len(),
            1,
            "provider guard must run before rendering"
        );
        assert_eq!(
            record.scopes.len(),
            1,
            "the admitted scope must be open before rendering"
        );
        let context = &record.scopes[0];
        assert_eq!(record.safety[0], *context);
        assert_eq!(
            context
                .reference_resources
                .as_ref()
                .unwrap()
                .prepared_canvases
                .len(),
            request.conditioning.len()
        );
        assert_eq!(
            (request.width, request.height),
            (context.geometry.width, context.geometry.height)
        );
        assert!(request.flux3_reference_output.is_some());
        record.generated += 1;
        Ok(gen_core::GenerationOutput::Images(vec![gen_core::Image {
            width: request.width,
            height: request.height,
            pixels: vec![137; (request.width * request.height * 3) as usize],
        }]))
    }
}

fn request_generator(contract: Option<MemoryProviderContract>) -> RequestGenerator {
    RequestGenerator {
        descriptor: gen_core::ModelDescriptor {
            id: "mage_flow",
            family: "test",
            backend: "mlx",
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &[],
            control_kinds: None,
        },
        contract,
    }
}

fn request_plan() -> MlxRequestPlan {
    MlxRequestPlan {
        engine_id: "mage_flow",
        model_id: "mage_flow".to_owned(),
        tier: MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Q4),
            component_precision_floors: &[],
        },
        asset_bytes: gib_to_bytes(6.0),
        folded_control_bytes: 0,
        folded_adapter_bytes: 0,
        // Deliberately ABOVE the 2 GiB fixed reserve so the area term is non-zero: a fixture
        // sitting exactly on the reserve would model resolution-blind and silently stop
        // exercising the sc-16195 scaling at all.
        activation_headroom_bytes: gib_to_bytes(6.0),
        fixed_reserve_bytes: gib_to_bytes(2.0),
        calibration: MlxCalibrationConfig::Absent,
        load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
    }
}

#[cfg(target_os = "macos")]
#[test]
fn mage_q4_memory_identity_includes_descriptor_component_floors() {
    let source = WeightsSource::Dir(std::path::PathBuf::from("/nonexistent/mage-fixture"));
    let q4 = LoadSpec::new(source.clone()).with_quant(gen_core::Quant::Q4);
    let q4_plan = MlxRequestPlan::for_spec_and_manifest("mage_flow", "mage_flow", &q4, None, None);
    let advertised = crate::inference_runtime::media_descriptor("mage_flow")
        .unwrap()
        .capabilities
        .component_precision_floors;
    assert_eq!(q4_plan.tier.component_precision_floors, advertised);
    assert_eq!(advertised.len(), 2);

    let q8 = LoadSpec::new(source).with_quant(gen_core::Quant::Q8);
    let q8_plan = MlxRequestPlan::for_spec_and_manifest("mage_flow", "mage_flow", &q8, None, None);
    assert!(q8_plan.tier.component_precision_floors.is_empty());
    assert_ne!(q4_plan.tier, q8_plan.tier);
}

fn mage_request_contract() -> MemoryProviderContract {
    use gen_core::MemoryCalibrationIdentity;

    let mut contract = MemoryProviderContract::compatibility_default(
        "mage_flow",
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        mage_request_fingerprint(),
        gen_core::LoadShape::EagerMaterialization,
    ));
    contract.asset_facts.base_bytes = gib_to_bytes(6.0);
    contract.asset_facts.transformer_bytes = gib_to_bytes(6.0);
    contract
}

/// The (route, tier) cell `request_plan()` names: `mage_flow` at q4.
fn mage_request_fingerprint() -> String {
    mage_calibration_fingerprint("mage_flow", Some(gen_core::Quant::Q4))
        .expect("mage_flow q4 is a shipped Mage-Flow cell")
}

fn request_inputs(width: u32, height: u32, count: u32) -> MlxRequestInputs {
    MlxRequestInputs {
        native_batch: 1,
        width,
        height,
        count,
        mode: "edit_image".to_owned(),
        overlay: Some("references:2+mask+adapters:1".to_owned()),
        adapter_count: 0,
        has_reference: true,
        reference_count: 2,
        use_pid: false,
        has_phases: false,
        conditioning_windows: None,
        negative_branch: false,
        prefix_kv_cache_eligible: false,
        reference_resources: None,
    }
}

#[test]
fn krea_raw_true_cfg_and_reference_img2img_keep_exact_public_and_provider_shapes() {
    let plain = MlxRequestInputs {
        native_batch: 1,
        width: 1024,
        height: 1024,
        count: 4,
        mode: "image_generation".to_owned(),
        overlay: None,
        adapter_count: 0,
        has_reference: false,
        reference_count: 0,
        use_pid: false,
        has_phases: false,
        conditioning_windows: None,
        negative_branch: false,
        prefix_kv_cache_eligible: false,
        reference_resources: None,
    };
    let (mode, mode_key) = request_mode(&plain.mode);
    assert_eq!(mode, MemoryMode::TextToImage);
    assert_eq!(mode_key, "text_to_image");
    assert_eq!(
        provider_request_mode("krea_2_raw", &plain),
        (MemoryMode::TextToImage, "text_to_image")
    );
    assert_eq!(request_geometry(&plain).batch, 1);
    assert_eq!(request_geometry(&plain).reference_count, 0);

    let reference = MlxRequestInputs {
        native_batch: 1,
        overlay: Some("references:1".to_owned()),
        has_reference: true,
        reference_count: 1,
        ..plain.clone()
    };
    assert_eq!(
        request_mode(&reference.mode).0,
        MemoryMode::TextToImage,
        "the public catalog coordinate remains generation"
    );
    assert_eq!(
        provider_request_mode("krea_2_raw", &reference),
        (MemoryMode::ImageToImage, "image_to_image"),
        "the Raw provider behavior contract sees its one-reference latent-init route"
    );
    assert_eq!(
        provider_request_mode("krea_2_turbo", &reference),
        (MemoryMode::ImageToImage, "image_to_image"),
        "Turbo's one-reference latent-init route has the same provider-owned identity"
    );
    assert_eq!(
        provider_request_inputs("krea_2_raw", &reference)
            .as_ref()
            .overlay
            .as_deref(),
        None,
        "reference count is the exact Raw provider axis; no synthetic overlay is retained"
    );
    assert_eq!(
        provider_request_inputs("krea_2_turbo", &reference)
            .as_ref()
            .overlay
            .as_deref(),
        None,
        "Krea base-DiT reference cardinality is typed and never duplicated as an overlay"
    );
    assert_eq!(request_geometry(&reference).batch, 1);
    assert_eq!(request_geometry(&reference).reference_count, 1);

    let reference_low_rank_pid = MlxRequestInputs {
        native_batch: 1,
        overlay: Some("references:1+adapters:1".to_owned()),
        adapter_count: 1,
        use_pid: true,
        ..reference.clone()
    };
    let provider_inputs = provider_request_inputs("krea_2_raw", &reference_low_rank_pid);
    assert_eq!(provider_inputs.as_ref().overlay, None);
    assert_eq!(provider_inputs.as_ref().adapter_count, 1);
    assert!(provider_inputs.as_ref().use_pid);
    assert_eq!(provider_inputs.as_ref().reference_count, 1);

    let pid = MlxRequestInputs {
        native_batch: 1,
        use_pid: true,
        ..plain
    };
    assert_eq!(
        provider_request_mode("krea_2_raw", &pid),
        (MemoryMode::TextToImage, "text_to_image")
    );
    assert_eq!(request_geometry(&pid).batch, 1);
    assert!(pid.use_pid);

    let edit = MlxRequestInputs {
        native_batch: 1,
        mode: "edit_image".to_owned(),
        overlay: Some("adapters:1+pid".to_owned()),
        adapter_count: 1,
        has_reference: true,
        reference_count: 2,
        use_pid: true,
        ..reference
    };
    for provider in ["krea_2_edit", "krea_2_turbo_edit"] {
        assert_eq!(
            provider_request_mode(provider, &edit),
            (MemoryMode::Edit, "edit")
        );
        let normalized = provider_request_inputs(provider, &edit);
        assert_eq!(normalized.as_ref().overlay, None, "{provider}");
        assert_eq!(normalized.as_ref().adapter_count, 1, "{provider}");
        assert!(normalized.as_ref().use_pid, "{provider}");
        assert_eq!(normalized.as_ref().reference_count, 2, "{provider}");
    }
    assert_eq!(
        provider_request_inputs("krea_2_turbo_control", &edit)
            .as_ref()
            .overlay
            .as_deref(),
        Some("adapters:1+pid"),
        "the independently keyed Krea control provider keeps its established overlay"
    );
}

#[test]
fn flux1_provider_overlay_is_derived_from_the_complete_load_identity() {
    let plain = LoadSpec::new(WeightsSource::Dir("flux".into()));
    assert_eq!(
        provider_overlay_for_load_spec("flux1_dev", &plain, None),
        None
    );
    let low_rank = plain.clone().with_adapters(vec![gen_core::AdapterSpec::new(
        "adapter.safetensors".into(),
        1.0,
        gen_core::AdapterKind::Lora,
    )]);
    assert_eq!(
        provider_overlay_for_load_spec("flux1_dev", &low_rank, Some("lora".to_owned())).as_deref(),
        Some("adapters")
    );
    let low_rank_pid = low_rank.clone().with_pid(
        WeightsSource::File("pid.safetensors".into()),
        WeightsSource::Dir("gemma".into()),
    );
    assert_eq!(
        provider_overlay_for_load_spec("flux1_dev", &low_rank_pid, None).as_deref(),
        Some("adapters-pid")
    );
    let identity = low_rank_pid.with_ip_adapter(WeightsSource::Dir("ip-adapter".into()));
    assert_eq!(
        provider_overlay_for_load_spec("flux1_dev", &identity, Some("identity".to_owned()))
            .as_deref(),
        Some("adapters-ip-adapter-pid")
    );
    let control = low_rank.with_control(WeightsSource::File("control.safetensors".into()));
    assert_eq!(
        provider_overlay_for_load_spec("flux1_dev_control", &control, Some("control".into()))
            .as_deref(),
        Some("control-adapters")
    );

    let text = MlxRequestInputs {
        native_batch: 1,
        width: 1024,
        height: 1024,
        count: 1,
        mode: "text_to_image".to_owned(),
        overlay: None,
        adapter_count: 0,
        has_reference: false,
        reference_count: 0,
        use_pid: false,
        has_phases: false,
        conditioning_windows: None,
        negative_branch: false,
        prefix_kv_cache_eligible: false,
        reference_resources: None,
    };
    for provider in ["flux1_schnell", "flux1_dev"] {
        for mode in ["text_to_image", "style_variations"] {
            let inputs = MlxRequestInputs {
                native_batch: 1,
                mode: mode.to_owned(),
                ..text.clone()
            };
            assert_eq!(
                provider_request_mode(provider, &inputs),
                (MemoryMode::TextToImage, "text_to_image"),
                "{provider} {mode}"
            );
        }
    }

    let character = MlxRequestInputs {
        native_batch: 1,
        mode: "character_image".to_owned(),
        overlay: Some("ip-adapter".to_owned()),
        has_reference: true,
        reference_count: 1,
        ..text.clone()
    };
    assert_eq!(
        provider_request_mode("flux1_dev", &character),
        (MemoryMode::ImageToImage, "image_to_image")
    );
    for mode in ["text_to_image", "style_variations", "character_image"] {
        let control = MlxRequestInputs {
            native_batch: 1,
            mode: mode.to_owned(),
            overlay: Some("control".to_owned()),
            has_reference: true,
            reference_count: 1,
            ..text.clone()
        };
        assert_eq!(
            provider_request_mode("flux1_dev_control", &control),
            (MemoryMode::ImageToImage, "image_to_image"),
            "control {mode}"
        );
    }

    assert_eq!(
            provider_request_mode(
                "flux1_schnell",
                &MlxRequestInputs {
                    native_batch: 1,
                    mode: "character_image".to_owned(),
                    has_reference: true,
                    reference_count: 1,
                    ..text
                }
            ),
            (MemoryMode::ImageToImage, "image_to_image"),
            "Schnell has no declaration-owned character route; generic normalization remains visible to fail closed"
        );
}

/// The fixture record carries its own synthetic revision and [`FIXTURE_CLOSURE_DIGEST`].
///
/// This used to rewrite every record's `inference.revision` to the live Cargo pin, because
/// currency was pin equality and the fixture would otherwise have read as stale on load. Under
/// sc-17774 currency is the closure digest, which the fixture file already carries, so the
/// rewrite is dead — and keeping it would restate the pin as a currency term in the one place
/// every gate test builds its evidence from.
fn fixture_bundle() -> EvidenceBundle {
    sceneworks_core::memory_calibration::load_bundle(include_str!(
        "../../tests/fixtures/mlx-memory-calibration.json"
    ))
    .expect("valid MLX calibration fixture")
}

fn fixture_binding(tier: &str, variant: &str) -> MlxCalibrationBinding {
    let parameters = JsonObject::from_iter([
        ("decodeTileEdge".to_owned(), serde_json::json!(512)),
        ("decodeOverlap".to_owned(), serde_json::json!(128)),
    ]);
    fixture_binding_for(tier, variant, StrategyRung::BoundedDecode, parameters)
}

fn fixture_binding_for(
    tier: &str,
    variant: &str,
    rung: StrategyRung,
    parameters: JsonObject<String, Value>,
) -> MlxCalibrationBinding {
    MlxCalibrationBinding {
        query: CalibrationBinding {
            abi: sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI,
            load_shape: sceneworks_core::memory_calibration::LoadShapeKey::EagerMaterialization,
            fingerprint: "fixture-formula-v2".to_owned(),
            scene_works_revision: "a".repeat(40),
            matrix_source_revision: "source-tree:1111111".to_owned(),
            // Capture provenance only, and deliberately the fixture RECORD's own synthetic
            // revision rather than the live pin: nothing compares this field, and spelling the
            // pin here implied a currency that no longer exists (sc-17774).
            inference_revision: "b".repeat(40),
            inference_closure_digest: fixture_closure_digest(),
            artifact_repository: "SceneWorks/fixture".to_owned(),
            artifact_resolved_revision: "c".repeat(40),
            artifact_variant: variant.to_owned(),
            resolved_path_fingerprint: format!("SceneWorks/fixture@{}:{variant}", "c".repeat(40)),
        },
        provider: "fixture_provider".to_owned(),
        tier: tier.to_owned(),
        mode: "text_to_image".to_owned(),
        overlay: "none".to_owned(),
        geometry: CalibrationGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
        },
        rung,
        selection_parameters: parse_evidence_parameters(
            rung,
            &crate::memory_strategy::default_engaged_composition(rung),
            &parameters,
        )
        .expect("fixture parameters"),
        parameters,
    }
}

/// The closure digest the synthetic `fixture_provider` lane is stamped with.
///
/// `fixture_provider` is not a real inference crate, so it is deliberately NOT in
/// `config/inference-provider-closures.json`. Since sc-22738 that costs it nothing: the runtime
/// compares no closure digest anywhere on the MLX path, so an undeclared lane is graded exactly
/// like a declared one. The constant survives because the fixture records and bindings must
/// still agree with each other on the provenance they carry.
const FIXTURE_CLOSURE_DIGEST: &str =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

fn fixture_closure_digest() -> String {
    FIXTURE_CLOSURE_DIGEST.to_owned()
}

/// The compile-closure digest the SHIPPED `mlx.calibrations` opt-in for `model_id` declares —
/// the closure its bindings, and the packaged records behind them, were measured under.
///
/// sc-22738 made this a CONFIG-CONSISTENCY accessor only: no admission decision reads a closure
/// digest any more, so the remaining callers use it to check that the shipped artefacts agree
/// with each other, never to predict how a request routes.
///
/// Uniformity across the model's bindings is asserted rather than assumed: a split opt-in would
/// let one stale row hide behind a current one and make every comparison below ambiguous.
///
/// sc-22512 (E8): `None` ONLY when the model declares no `mlx.calibrations` opt-in at all, or
/// declares an empty one. Absence of an opt-in is a model nobody measured, which is legal —
/// callers SKIP the comparison rather than failing. A PRESENT but split or malformed opt-in
/// still fails: that is contradictory data, not missing data.
fn shipped_mlx_declared_closure_digest(model_id: &str) -> Option<String> {
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let declared = manifest["models"]
        .as_array()
        .and_then(|models| models.iter().find(|model| model["id"] == model_id))
        .and_then(|model| model["mlx"]["calibrations"].as_array())?
        .clone();
    // Absence — no `mlx.calibrations` block, or an empty one — is a model nobody measured, and
    // the callers legally skip. Anything PRESENT must be well formed: a binding that names no
    // string closure is contradictory data, so it panics here rather than being mapped to
    // `None` and dedup'd into a `[None]` that would sail through the uniformity check below and
    // silently make every call site skip.
    if declared.is_empty() {
        return None;
    }
    let mut declared = declared
        .iter()
        .map(|binding| {
            binding["inferenceClosureDigest"]
                .as_str()
                .unwrap_or_else(|| {
                    panic!(
                        "{model_id} ships an mlx.calibrations binding with no string \
                             inferenceClosureDigest: {binding}"
                    )
                })
                .to_owned()
        })
        .collect::<Vec<String>>();
    declared.sort();
    declared.dedup();
    assert_eq!(
        declared.len(),
        1,
        "{model_id}'s shipped bindings must all name ONE captured closure"
    );
    Some(declared.remove(0))
}

fn fixture_calibration_json(tier: &str, variant: &str) -> Value {
    serde_json::json!({
        "abi": sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI,
        "loadShape": "eager_materialization",
        "fingerprint": "fixture-formula-v2",
        "sceneWorksRevision": "a".repeat(40),
        "matrixSourceRevision": "source-tree:1111111",
        // Capture provenance only — see `fixture_binding_for`.
        "inferenceRevision": "b".repeat(40),
        // sc-17774: the fixture must present the digest the gate will look up for this lane,
        // otherwise every fit-ladder test silently becomes a currency-refusal test.
        "inferenceClosureDigest": fixture_closure_digest(),
        "provider": "fixture_provider",
        "tier": tier,
        "mode": "text_to_image",
        "overlay": "none",
        "geometry": {
            "width": 1024,
            "height": 1024,
            "batch": 1,
            "frames": 1
        },
        "artifactRepository": "SceneWorks/fixture",
        "artifactResolvedRevision": "c".repeat(40),
        "artifactVariant": variant,
        "resolvedPathFingerprint": format!(
            "SceneWorks/fixture@{}:{variant}",
            "c".repeat(40)
        ),
        "rung": "bounded_decode",
        "parameters": {
            "decodeTileEdge": 512,
            "decodeOverlap": 128
        }
    })
}

fn fixture_manifest(calibrations: Vec<Value>) -> JsonObject<String, Value> {
    serde_json::json!({ "mlx": { "calibrations": calibrations } })
        .as_object()
        .expect("manifest object")
        .clone()
}

fn fixture_spec(tier: gen_core::Quant, variant: &str) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(format!(
        "/cache/models--SceneWorks--fixture/snapshots/{}/{variant}/weights",
        "c".repeat(40)
    ))))
    .with_quant(tier)
}

fn fixture_provenance(tier: &str, variant: &str) -> ResolvedArtifactProvenance {
    ResolvedArtifactProvenance {
        identity: crate::model_jobs::ResolvedArtifactIdentity {
            repository: "SceneWorks/fixture".to_owned(),
            revision: "c".repeat(40),
            variant: variant.to_owned(),
            fingerprint: format!("SceneWorks/fixture@{}:{variant}", "c".repeat(40)),
        },
        fixed_artifact_tier: Some(tier.to_owned()),
    }
}

fn fixture_plan() -> MlxRequestPlan {
    let variant = "packed-q4";
    MlxRequestPlan {
        engine_id: "fixture_provider",
        model_id: "fixture_model".to_owned(),
        tier: MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Q4),
            component_precision_floors: &[],
        },
        asset_bytes: gib_to_bytes(3.0),
        folded_control_bytes: 0,
        folded_adapter_bytes: 0,
        // Deliberately ABOVE the 2 GiB fixed reserve so the area term is non-zero: a fixture
        // sitting exactly on the reserve would model resolution-blind and silently stop
        // exercising the sc-16195 scaling at all.
        activation_headroom_bytes: gib_to_bytes(6.0),
        fixed_reserve_bytes: gib_to_bytes(2.0),
        calibration: MlxCalibrationConfig::Valid(MlxCalibrationSet {
            bindings: vec![fixture_binding("q4", variant)],
            resolved: fixture_provenance("q4", variant),
        }),
        load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
    }
}

#[test]
fn owned_reference_pricing_precedes_candidate_fit_and_none_preserves_decisions() {
    fn descriptor() -> gen_core::ModelDescriptor {
        let mut descriptor = fixture_generator().descriptor;
        descriptor.capabilities.min_size = 16;
        descriptor.capabilities.max_size = 4096;
        descriptor.capabilities.max_count = 1;
        descriptor
    }
    fn no_load(_: &LoadSpec) -> gen_core::Result<Box<dyn gen_core::Generator>> {
        panic!("reference fit must not load a generator")
    }
    fn price(
        spec: &LoadSpec,
        facts: &gen_core::MemoryReferenceResourceFacts,
    ) -> gen_core::Result<u64> {
        if facts.conditioning_tokens != 16 || facts.prepared_canvases != [(64, 64)] {
            return Err(gen_core::Error::Unsupported(
                "invalid fixture reference facts".to_owned(),
            ));
        }
        let WeightsSource::Dir(path) = &spec.weights else {
            panic!("fixture directory")
        };
        Ok(if path.ends_with("q4") {
            gib_to_bytes(4.0)
        } else {
            gib_to_bytes(16.0)
        })
    }
    let registry = |priced: bool| {
        let builder = gen_core::ProviderRegistryBuilder::new().register_generator(
            gen_core::ModelRegistration {
                descriptor,
                load: no_load,
                footprint: None,
            },
        );
        let builder = if priced {
            builder.register_reference_workspace(gen_core::ReferenceWorkspaceRegistration {
                provider_id: "fixture_provider",
                reference_workspace_bytes: price,
            })
        } else {
            builder
        };
        builder.build().unwrap()
    };
    let registry = registry(true);
    let mut generator = fixture_generator();
    // A real staging opportunity: separate 30 GiB conditioning and 3 GiB transformer stacks.
    let contract = generator.contract.as_mut().unwrap();
    contract.asset_facts.base_bytes = gib_to_bytes(33.0);
    contract.asset_facts.conditioning_bytes = gib_to_bytes(30.0);
    contract.load_shape = gen_core::LoadShape::DeferredMaterialization;
    contract.calibration = Some(gen_core::MemoryCalibrationIdentity::new(
        "fixture-formula-v2",
        gen_core::LoadShape::DeferredMaterialization,
    ));
    let plan = MlxRequestPlan {
        asset_bytes: gib_to_bytes(33.0),
        calibration: MlxCalibrationConfig::Absent,
        ..fixture_plan()
    };
    let mut q4 = LoadSpec::new(WeightsSource::Dir(PathBuf::from("q4")));
    q4.offload_policy = OffloadPolicy::Sequential;
    let mut bf16 = LoadSpec::new(WeightsSource::Dir(PathBuf::from("bf16")));
    bf16.offload_policy = OffloadPolicy::Sequential;
    let plain = fixture_inputs(1024, 1024);
    let data = tempfile::tempdir().unwrap();
    let store =
        sceneworks_core::project_store::ProjectStore::new(data.path().to_path_buf(), "scope-test");
    let project = store.create_project("owned pre-load references").unwrap();
    let source = data.path().join("source.png");
    image::RgbaImage::from_pixel(64, 64, image::Rgba([17, 31, 47, 0]))
        .save(&source)
        .unwrap();
    let asset = store
        .import_asset(
            &project.id,
            sceneworks_core::project_store::UploadAsset {
                filename: "source.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source,
                source_asset_id: None,
                provenance: None,
            },
        )
        .unwrap();
    let payload = serde_json::json!({"projectId":project.id,"model":"flux3_klein","mode":"edit_image","sourceAssetId":asset["id"],"width":1024,"height":1024,"count":1});
    let authored =
        sceneworks_core::image_request::ImageRequest::from_payload(payload.as_object().unwrap());
    let references = crate::image_jobs::flux3_references::resolve_owned_references(
        &authored,
        data.path(),
        std::path::Path::new(&project.path),
    )
    .unwrap()
    .unwrap();
    let facts = references.resources.clone();
    assert_eq!(facts.prepared_canvases, [(64, 64)]);
    assert_eq!(facts.conditioning_tokens, 16);
    let mut referenced = plain.clone();
    referenced.has_reference = true;
    referenced.reference_count = 1;
    referenced.reference_resources = Some(facts.clone());
    let mut reference_legacy = referenced.clone();
    reference_legacy.reference_resources = None;
    let run = |inputs: &MlxRequestInputs,
               spec: Option<&LoadSpec>,
               total: f64,
               phase: MlxAdmissionPhase| {
        select_request_with_budget_using_bundle(
            phase,
            generator.descriptor.capabilities.component_precision_floors,
            generator.contract.as_ref(),
            None,
            &|_| None,
            |base| {
                generator
                    .contract
                    .as_ref()
                    .unwrap()
                    .predicted_peak_from_base(base)
            },
            &plan,
            inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Sequential,
            crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
            fixture_budget(total),
            plan.generic_total_peak_bytes(request_geometry(inputs)),
            0,
            &[],
            None,
            spec.map(|spec| ReferencePricing {
                spec,
                registry: &registry,
            }),
        )
    };
    for phase in [MlxAdmissionPhase::Generation, MlxAdmissionPhase::TierProbe] {
        let without = run(&plain, None, 128.0, phase)
            .unwrap()
            .into_evaluation()
            .unwrap();
        let with_unused_spec = run(&plain, Some(&q4), 128.0, phase)
            .unwrap()
            .into_evaluation()
            .unwrap();
        assert_eq!(
            format!("{:?}", without.context),
            format!("{:?}", with_unused_spec.context)
        );
        let base = run(&reference_legacy, None, 128.0, phase)
            .unwrap()
            .into_evaluation()
            .unwrap();
        let small = run(&referenced, Some(&q4), 128.0, phase)
            .unwrap()
            .into_evaluation()
            .unwrap();
        let large = run(&referenced, Some(&bf16), 128.0, phase)
            .unwrap()
            .into_evaluation()
            .unwrap();
        assert_eq!(
            small.context.predicted_peak_bytes,
            base.context.predicted_peak_bytes + gib_to_bytes(4.0)
        );
        assert_eq!(
            large.context.predicted_peak_bytes,
            base.context.predicted_peak_bytes + gib_to_bytes(16.0)
        );
        assert_eq!(small.context.reference_resources, Some(facts.clone()));
        let optimized = (8..128)
            .find_map(|total| {
                run(&referenced, Some(&q4), f64::from(total), phase)
                    .unwrap()
                    .into_evaluation()
                    .ok()
                    .filter(|evaluation| evaluation.context.selection.strategy.is_optimized())
            })
            .expect("fixture must exercise an actually admitted optimized reference request");
        // This is the SAME scope/render entry used by render_on_surface on both native lanes,
        // past real pre-load pricing and selection, not an assertion on a helper's stored field.
        for admission in [&small, &optimized] {
            for backend in ["candle", "mlx"] {
                for engine_id in ["flux3_klein", "flux3_klein_base"] {
                    let context = crate::memory_strategy::executable_context(
                        engine_id,
                        admission.context.clone(),
                    )
                    .expect("resource-bearing Resident and optimized contexts must survive");
                    let mut descriptor = fixture_generator().descriptor;
                    descriptor.id = engine_id;
                    descriptor.backend = backend;
                    let record = std::sync::Arc::new(std::sync::Mutex::new(
                        ReferenceRenderRecord::default(),
                    ));
                    // Construct the loaded handle ONLY after the provider-priced pre-load gate.
                    let loaded = ReferenceRenderGenerator {
                        descriptor,
                        record: std::sync::Arc::clone(&record),
                    };
                    let mut request = gen_core::GenerationRequest {
                        prompt: "reference-scoped render".to_owned(),
                        width: context.geometry.width,
                        height: context.geometry.height,
                        // The production reference-render seam, not the test, installs the
                        // resolved original carriers and authored output-size intent.
                        memory: Some(admission.memory),
                        ..Default::default()
                    };
                    crate::image_jobs::flux3_references::render_reference_request(
                        &loaded,
                        &mut request,
                        &references,
                        Some(&context),
                        &mut |_| {},
                    )
                    .unwrap();
                    fn carrier(
                        conditioning: &gen_core::Conditioning,
                    ) -> (u32, u32, &[u8], Option<f32>) {
                        match conditioning {
                            gen_core::Conditioning::Reference { image, strength } => {
                                (image.width, image.height, &image.pixels, *strength)
                            }
                            gen_core::Conditioning::ReferenceRgba { image, strength } => {
                                (image.width, image.height, &image.pixels, *strength)
                            }
                            other => panic!("unexpected owned carrier: {other:?}"),
                        }
                    }
                    assert_eq!(request.conditioning.len(), references.conditioning.len());
                    for (actual, expected) in
                        request.conditioning.iter().zip(&references.conditioning)
                    {
                        assert_eq!(carrier(actual), carrier(expected));
                    }
                    assert_eq!(request.flux3_reference_output, Some(references.output_size));
                    let observed = record.lock().unwrap();
                    assert_eq!(observed.safety, vec![admission.context.clone()]);
                    assert_eq!(observed.scopes, vec![admission.context.clone()]);
                    assert_eq!(observed.scopes[0].reference_resources, Some(facts.clone()));
                    assert_eq!(observed.generated, 1);
                    assert_eq!(
                        observed.finished,
                        vec![gen_core::MemoryRunOutcome::Complete]
                    );
                    drop(observed);
                    assert!(
                        crate::memory_strategy::generate_with_scope_report(
                            &loaded,
                            &mut request,
                            None,
                            &mut |_| {}
                        )
                        .is_err(),
                        "references must not take the legacy no-context shortcut"
                    );
                    let mut substituted = context.clone();
                    substituted
                        .reference_resources
                        .as_mut()
                        .unwrap()
                        .conditioning_tokens += 1;
                    let error = crate::image_jobs::flux3_references::render_reference_request(
                        &loaded,
                        &mut request,
                        &references,
                        Some(&substituted),
                        &mut |_| {},
                    )
                    .unwrap_err();
                    assert!(error.to_string().contains("same admitted reference"));
                    let mut missing_facts = context.clone();
                    missing_facts.reference_resources = None;
                    assert!(crate::memory_strategy::generate_with_scope_report(
                        &loaded,
                        &mut request,
                        Some(&missing_facts),
                        &mut |_| {}
                    )
                    .is_err());
                    assert_eq!(
                        record.lock().unwrap().generated,
                        1,
                        "missing facts must fail before rendering"
                    );
                }
            }
        }
        let mut rejected_only_with_references = 0;
        for total in 8..128 {
            let without = run(&plain, None, f64::from(total), phase).unwrap();
            let with = run(&referenced, Some(&bf16), f64::from(total), phase).unwrap();
            if matches!(without, MlxRequestAdmission::Admitted(_))
                && matches!(with, MlxRequestAdmission::Rejected(_))
            {
                rejected_only_with_references += 1;
            }
        }
        assert!(
            rejected_only_with_references > 0,
            "reference cost never changed the fit decision"
        );
    }
    let error = evaluate_request_with_budget(
        &generator,
        &plan,
        &referenced,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        fixture_budget(128.0),
        plan.generic_total_peak_bytes(request_geometry(&referenced)),
        0,
        &[],
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("reference-aware memory admission"));
}

/// `None` when the packaged bundle cannot supply this fixture — see the note beside the
/// selection below. The caller withholds its question rather than reddening.
fn packaged_krea_plan() -> Option<MlxRequestPlan> {
    let bundle = sceneworks_core::memory_calibration::load_packaged_bundle()
        .expect("packaged bundle must parse");
    // Select the records that ARE current instead of rewriting stale ones into looking current,
    // which is what the deleted `packaged_bundle_migrated_to_v4_for_tests` shim did. The bundle
    // still carries the superseded 96b13b66 Krea cells as history, so without this currency
    // predicate the filter matches four records and the count assertion below fails.
    //
    // sc-17774: the predicate is the provider's own closure digest, not the inference pin. The
    // pin form separated the same two records only by coincidence — it also expired them on any
    // unrelated inference commit, so this fixture went unbuildable on every bump.
    //
    // The digest is the one the shipped opt-in DECLARES, not the live one. Selecting on live
    // made the fixture empty the moment a shared `mlx-gen` crate moved — a legitimate closure
    // move — and this fixture exists to exercise refusal NAMING, which needs two exact cells to
    // choose between. `packaged_krea_closure_lookup` feeds the same digest to the gate, and the
    // test asserts the live-closure refusal separately so the demotion is not merely bypassed.
    // sc-22512: the opt-in is the PREFERRED source of the closure to select on, not a required
    // one. With no declared opt-in the fixture falls back to the closure of the LAST matching
    // record in the bundle — still exactly one coherent capture, never a mix of two — so an
    // absent declaration degrades the fixture's provenance rather than reddening the build.
    let matches_krea_control_cell =
        |record: &sceneworks_core::memory_calibration::EvidenceRecord| {
            matches!(record.backend, CalibrationBackend::Mlx)
                && record.target.model_id == "krea_2_turbo"
                && record.target.provider == "krea_2_turbo_control"
                && record.target.tier == "q4"
                && record.target.mode == "text_to_image"
                && record.target.overlay == "control:1"
        };
    let captured = shipped_mlx_declared_closure_digest("krea_2_turbo").or_else(|| {
        bundle
            .records
            .iter()
            .rev()
            .find(|record| matches_krea_control_cell(record))
            .and_then(|record| record.repositories.inference.closure_digest.clone())
    });
    let records = bundle
        .records
        .iter()
        .filter(|record| {
            matches_krea_control_cell(record)
                && captured.as_deref().is_none_or(|captured| {
                    record.repositories.inference.closure_digest.as_deref() == Some(captured)
                })
        })
        .collect::<Vec<_>>();
    // PRESENT-data claim, kept: whatever cells the bundle carries at this one closure must be
    // DISTINCT — two records for the same geometry and rung are a duplicated capture, which no
    // measurement could resolve and which would make the refusal-naming fixture ambiguous.
    let mut cells = records
        .iter()
        .map(|record| (record.target.geometry, record.strategy.rung))
        .collect::<Vec<_>>();
    let distinct = cells.len();
    cells.dedup_by(|left, right| left == right);
    assert_eq!(
        cells.len(),
        distinct,
        "the packaged Krea cells at one closure must not duplicate a (geometry, rung)"
    );
    // ABSENCE, converted (sc-22512): this fixture stands in for a two-cell measured ladder — it
    // exists to prove a refusal NAMES the largest fitting exact cell, which needs two exact
    // cells to choose between. A bundle that carries fewer is a lane nobody measured that far,
    // so the fixture reports itself unbuildable and its caller withholds the question. The old
    // `assert_eq!(records.len(), 2)` reddened on exactly that absence — and made the `.or_else`
    // closure fallback above decorative, since the count fired first either way.
    if records.len() != 2 {
        return None;
    }
    let first = records[0];
    let resolved_path_fingerprint =
        |record: &sceneworks_core::memory_calibration::EvidenceRecord| {
            let sceneworks_core::memory_calibration::RequiredNullable::Value(fingerprint) =
                &record.loadability.resolved_path_fingerprint
            else {
                panic!("complete Krea record must carry a resolved path fingerprint");
            };
            fingerprint.clone()
        };
    let resolved = ResolvedArtifactProvenance {
        identity: crate::model_jobs::ResolvedArtifactIdentity {
            repository: first.artifact.repository.clone(),
            revision: first.artifact.resolved_revision.clone(),
            variant: first.artifact.variant.clone(),
            fingerprint: resolved_path_fingerprint(first),
        },
        fixed_artifact_tier: Some(first.target.tier.clone()),
    };
    let bindings = records
        .into_iter()
        .map(|record| MlxCalibrationBinding {
            query: CalibrationBinding {
                abi: sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI,
                load_shape: record.load_shape,
                fingerprint: record.calibration_fingerprint.clone(),
                scene_works_revision: "sc-16099-contract-v1".to_owned(),
                matrix_source_revision: record
                    .repositories
                    .scene_works
                    .matrix_source_revision
                    .clone()
                    .expect("current matrix source revision"),
                inference_revision: record.repositories.inference.revision.clone(),
                inference_closure_digest: record
                    .repositories
                    .inference
                    .closure_digest
                    .clone()
                    .expect("captured provider closure digest"),
                artifact_repository: record.artifact.repository.clone(),
                artifact_resolved_revision: record.artifact.resolved_revision.clone(),
                artifact_variant: record.artifact.variant.clone(),
                resolved_path_fingerprint: resolved_path_fingerprint(record),
            },
            provider: record.target.provider.clone(),
            tier: record.target.tier.clone(),
            mode: record.target.mode.clone(),
            overlay: record.target.overlay.clone(),
            geometry: CalibrationGeometry {
                width: record.target.geometry.width,
                height: record.target.geometry.height,
                batch: record.target.geometry.batch,
                frames: record.target.geometry.frames,
            },
            rung: record.strategy.rung,
            selection_parameters: parse_evidence_parameters(
                record.strategy.rung,
                &record.strategy.engaged_rungs,
                &record.strategy.parameters,
            )
            .expect("packaged Krea strategy parameters"),
            parameters: record.strategy.parameters.clone(),
        })
        .collect();
    Some(MlxRequestPlan {
        engine_id: "krea_2_turbo_control",
        model_id: "krea_2_turbo".to_owned(),
        tier: MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Q4),
            component_precision_floors: &[],
        },
        asset_bytes: gib_to_bytes(30.0),
        folded_control_bytes: 0,
        folded_adapter_bytes: 0,
        activation_headroom_bytes: gib_to_bytes(2.0),
        fixed_reserve_bytes: 0,
        calibration: MlxCalibrationConfig::Valid(MlxCalibrationSet { bindings, resolved }),
        load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
    })
}

/// `None` when the packaged bundle carries no Krea control record at the fixture's closure.
fn packaged_krea_generator() -> Option<RequestGenerator> {
    use gen_core::MemoryCalibrationIdentity;

    let mut generator = fixture_generator();
    generator.descriptor.id = "krea_2_turbo_control";
    let contract = generator.contract.as_mut().expect("fixture contract");
    contract.provider_id = "krea_2_turbo_control".to_owned();
    // Taken from the packaged record this fixture stands in for, never restated as a literal.
    // The previous literal `krea-control-mlx-v4-q4-pose-bounded-decode-512-64` outlived the
    // provider's move to the full-ladder identity, and a stale copy here fails the handshake
    // and reports the cell as `Missing` — which is how it presented before this was fixed.
    // sc-22512: the opt-in is preferred, not required — with none declared the fixture falls
    // back to the last matching record's own closure rather than panicking.
    let captured = shipped_mlx_declared_closure_digest("krea_2_turbo").or_else(|| {
        packaged_bundle()
            .records
            .iter()
            .rev()
            .find(|record| {
                matches!(record.backend, CalibrationBackend::Mlx)
                    && record.target.provider == "krea_2_turbo_control"
            })
            .and_then(|record| record.repositories.inference.closure_digest.clone())
    });
    // sc-22512 (E8): looked up, not required. A bundle carrying no Krea control record at this
    // closure is a lane nobody measured; the fixture reports itself unbuildable and its caller
    // withholds the question rather than the suite reddening on an absent measurement.
    let record = packaged_bundle().records.into_iter().find(|record| {
        matches!(record.backend, CalibrationBackend::Mlx)
            && record.target.provider == "krea_2_turbo_control"
            && captured.as_deref().is_none_or(|captured| {
                record.repositories.inference.closure_digest.as_deref() == Some(captured)
            })
    })?;
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        record.calibration_fingerprint,
        match record.load_shape {
            LoadShapeKey::DeferredMaterialization => gen_core::LoadShape::DeferredMaterialization,
            LoadShapeKey::EagerMaterialization => gen_core::LoadShape::EagerMaterialization,
        },
    ));
    let bounded_decode = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
        .expect("bounded decode capability");
    bounded_decode.parameters.decode_overlaps = vec![64];
    Some(generator)
}

fn fixture_inputs(width: u32, height: u32) -> MlxRequestInputs {
    MlxRequestInputs {
        native_batch: 1,
        width,
        height,
        count: 1,
        mode: "text_to_image".to_owned(),
        overlay: None,
        adapter_count: 0,
        has_reference: false,
        reference_count: 0,
        use_pid: false,
        has_phases: false,
        conditioning_windows: None,
        negative_branch: false,
        prefix_kv_cache_eligible: false,
        reference_resources: None,
    }
}

fn fixture_generator() -> RequestGenerator {
    use gen_core::{
        MemoryCalibrationIdentity, MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryPhase,
        MemoryStrategyCapability, MemoryStrategySupport,
    };

    let mut contract = MemoryProviderContract::compatibility_default(
        "fixture_provider",
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        "fixture-formula-v2",
        gen_core::LoadShape::EagerMaterialization,
    ));
    contract.asset_facts.base_bytes = gib_to_bytes(3.0);
    contract.asset_facts.transformer_bytes = gib_to_bytes(3.0);
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: true,
        transformer_window_materialization: false,
    };
    contract.strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: if matches!(
                strategy,
                MemoryStrategy::Resident
                    | MemoryStrategy::StagedResidency
                    | MemoryStrategy::BoundedDecode
                    | MemoryStrategy::BoundedAttention
            ) {
                MemoryStrategySupport::Implemented
            } else {
                MemoryStrategySupport::Missing
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: vec![512],
                    decode_overlaps: vec![128],
                    ..Default::default()
                },
                MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                    attention_chunk_sizes: vec![256],
                    ..Default::default()
                },
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect();
    RequestGenerator {
        descriptor: gen_core::ModelDescriptor {
            id: "fixture_provider",
            family: "test",
            backend: "mlx",
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &[],
            control_kinds: None,
        },
        contract: Some(contract),
    }
}

/// The real packaged bundle, as production loads it. sc-16915 re-collected the MLX evidence at
/// the current pin, so the tests below no longer need the deleted
/// `packaged_bundle_migrated_to_v4_for_tests` shim, which forced `schemaVersion` to 4, stamped
/// every row `eager_materialization`, and rewrote every inference revision to the running one.
/// That shim could not distinguish a genuinely current bundle from a stale one — its whole job
/// was to erase the difference — so a regression that re-staled the evidence would not have
/// failed a single test that used it.
fn packaged_bundle() -> EvidenceBundle {
    sceneworks_core::memory_calibration::load_packaged_bundle().expect("packaged bundle must parse")
}

fn fixture_budget(total_gib: f64) -> MemoryBudget {
    MemoryBudget {
        total_bytes: gib_to_bytes(total_gib),
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    }
}

/// The `full_ladder_generator` fixture's estimate floors in GiB, BEFORE the margin. Derived from
/// the fixture facts (base 3 GiB all-transformer, headroom 2 fixed + 4 area) and spelled out in
/// full on [`unmeasured_provider_under_a_small_budget_selects_a_deep_estimate_rung`]:
/// resident / staged / bounded-decode / bounded-attention all floor at 3 + 6 = 9 GiB, while
/// rung 4 windows the transformer down to its RESIDENT WINDOW (sc-22667: the law's share,
/// `FULL_LADDER_WEIGHTS_GB x 2 / FULL_LADDER_FIXTURE_BLOCKS` = 2 GiB under the injected
/// fixture facts) and floors at 2 + 6 = 8 GiB.
const FIXTURE_DEEP_ESTIMATE_FLOOR_GB: f64 = 4.0 + FULL_LADDER_RESIDENT_WINDOW_GB;

/// The block count the full-ladder fixture states through the `architecture_facts_from_contract`
/// seam (sc-22667): with the smallest declared window (2) of these blocks resident, rung 4
/// keeps `FULL_LADDER_RESIDENT_WINDOW_GB` of its all-transformer weights. Under the DEFAULT
/// facts (no block count) the floor keeps the PRE-EPIC accounting (whole transformer out) —
/// `without_facts_the_deep_rung_floor_keeps_the_pre_epic_accounting_and_admits` grades that
/// arm; every deep-rung admission fixture below injects these facts to grade the share.
const FULL_LADDER_FIXTURE_BLOCKS: u32 = 40;
const FULL_LADDER_RESIDENT_WINDOW_GB: f64 =
    FULL_LADDER_WEIGHTS_GB * 2.0 / FULL_LADDER_FIXTURE_BLOCKS as f64;
const FULL_LADDER_FIXTURE_FACTS: sceneworks_core::memory_anchor::ArchitectureFacts =
    sceneworks_core::memory_anchor::ArchitectureFacts {
        attention_heads: None,
        head_dim: None,
        transformer_blocks: Some(FULL_LADDER_FIXTURE_BLOCKS),
        patch_size: None,
        latent_channels: None,
        vae_spatial_scale: None,
        vae_temporal_scale: None,
        activation_dtype_width: None,
    };

#[test]
fn unknown_stream_inventory_cannot_erase_transformer_weights() {
    let mut generator = full_ladder_generator();
    generator.contract.as_mut().unwrap().phase_facts = None;
    let contract = generator.contract.as_ref().unwrap();
    let selection = MemorySelection {
        strategy: MemoryStrategy::BoundedTransformerResidency,
        tier: fixture_plan().tier,
        parameters: gen_core::MemoryStrategyParameters {
            transformer_window_size: Some(2),
            ..Default::default()
        },
    };
    let engaged = contract.engaged_composition_for_selection(&selection);
    assert_eq!(
        mlx_fallback_weights_bytes(contract, &engaged, selection.parameters),
        gib_to_bytes(FULL_LADDER_WEIGHTS_GB)
    );
    assert!(evaluate_request_with_budget(
        &generator,
        &fixture_plan(),
        &fixture_inputs(1024, 1024),
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        fixture_budget(7.0),
        full_ladder_baseline_bytes(),
        0,
        &[]
    )
    .is_err());
}

/// The fixture's combined headroom (2 GiB fixed reserve + 4 GiB activations at 1024²).
/// Only the activation slice receives the allocator allowance.
const FIXTURE_HEADROOM_GB: f64 = 6.0;

/// The production allowance an EstimateFloor candidate receives before the fit check.
///
/// Computed through the policy rather than a literal on purpose. Before sc-22508 this was
/// `floor * (1 + MLX_ESTIMATE_MARGIN)`, and when that corpus-derived margin moved 5% ->
/// 50.4073% every fixture that had hardcoded a host budget against the 5% term silently FLIPPED
/// DIRECTION (a "reaches the deep rung" test became a refusal test, for the wrong reason).
/// Sizing budgets through this function keeps each test's direction fixed across any future
/// re-derivation.
fn widened_estimate_gb(floor_gb: f64) -> f64 {
    floor_gb * (1.0 + crate::ladder_margin_policy::MLX_RECAPTURE_SPREAD)
}

/// A host budget that admits EXACTLY the deepest (rung-4) estimate floor and nothing shallower —
/// the midpoint of the widened window, so it cannot sit on either boundary.
///
/// Both bounds are computed from `FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE` through
/// [`widened_estimate_gb`], never transcribed. Transcribed window numbers are exactly what went
/// stale here before: this comment used to quote a GiB window read off a superseded allowance,
/// and a reader checking the fixtures against it would have been checking them against a value
/// the policy no longer carries. The window's position moves with any re-derivation of the
/// allowance; the assertion below is what keeps its ORDERING — the only property these
/// fixtures depend on — honest at whatever value the corpus derives.
fn budget_admitting_only_the_deepest_estimate_rung() -> MemoryBudget {
    let deepest = widened_estimate_gb(FIXTURE_DEEP_ESTIMATE_FLOOR_GB);
    // The shallow bound is now the LOWER of the shallow floors and the resident baseline's
    // undeclared-arm ceiling (epic 22505 feature-end fix round): the re-derived activation
    // allowance made every declared floor far more expensive than the recapture-graded
    // resident baseline, so a deep-rung window only exists where the transformer removal
    // (rung 4) outweighs the allowance surplus — which is why `full_ladder_generator` carries
    // FULL_LADDER_WEIGHTS_GB of all-transformer weights and its tests pass
    // `full_ladder_baseline_bytes()` as the resident baseline.
    let shallow_floor = widened_estimate_gb(FULL_LADDER_WEIGHTS_GB + FIXTURE_HEADROOM_GB);
    let resident_undeclared = crate::memory_strategy::peak_bytes_to_gb(
        crate::memory_strategy::floor_admitted_peak_bytes(
            gen_core::MemoryBackend::Mlx,
            full_ladder_baseline_bytes(),
            None,
        ),
    );
    let shallowest = shallow_floor.min(resident_undeclared);
    assert!(
        deepest < shallowest,
        "the deep rung must stay strictly cheaper than every shallower candidate: \
             {deepest} vs {shallowest}"
    );
    fixture_budget((deepest + shallowest) / 2.0)
}

/// Weights for the full-ladder fixture contract, all transformer: rung 4 windows the whole
/// 40 GiB out of residency, which is what keeps a walk-down window open under the honest
/// activation-term allowance (see `budget_admitting_only_the_deepest_estimate_rung`).
const FULL_LADDER_WEIGHTS_GB: f64 = 40.0;

/// The resident-baseline peak the full-ladder tests hand the gate: the fixture weights plus
/// the 6 GiB fixture headroom, so the baseline scales with the contract it grades.
fn full_ladder_baseline_bytes() -> u64 {
    gib_to_bytes(FULL_LADDER_WEIGHTS_GB + FIXTURE_HEADROOM_GB)
}

fn fixture_ladder() -> (EvidenceBundle, MlxRequestPlan) {
    use sceneworks_core::memory_calibration::RequiredNullable;

    let mut bundle = fixture_bundle();
    let base = bundle.records.remove(0);
    let generator = fixture_generator();
    let contract = generator.contract.as_ref().expect("fixture contract");
    let rung = |rung, parameters: JsonObject<String, Value>, peak_gib: f64| {
        let mut record = base.clone();
        record.strategy.rung = rung;
        record.strategy.engaged_rungs = contract
            .engaged_composition(evidence_strategy(rung))
            .into_iter()
            .map(|strategy| match strategy {
                MemoryStrategy::Resident => StrategyRung::Resident,
                MemoryStrategy::StagedResidency => StrategyRung::StagedResidency,
                MemoryStrategy::BoundedDecode => StrategyRung::BoundedDecode,
                MemoryStrategy::BoundedAttention => StrategyRung::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency => {
                    StrategyRung::BoundedTransformerResidency
                }
            })
            .collect();
        record.strategy.parameters = parameters.clone();
        record.sweep.cases[0].parameters = parameters;
        if let RequiredNullable::Value(predicted) = &mut record.predicted_peak_bytes {
            let predicted = predicted
                .full_mut()
                .expect("fixture has full phase telemetry");
            predicted.conditioning = predicted.conditioning.min(gib_to_bytes(peak_gib));
            predicted.denoise = gib_to_bytes(peak_gib);
            predicted.decode = predicted.decode.min(gib_to_bytes(peak_gib));
            predicted.overall = gib_to_bytes(peak_gib);
        }
        record
    };
    let resident_parameters = JsonObject::new();
    let decode_parameters = JsonObject::from_iter([
        ("decodeTileEdge".to_owned(), serde_json::json!(512)),
        ("decodeOverlap".to_owned(), serde_json::json!(128)),
    ]);
    let attention_parameters = JsonObject::from_iter([
        ("decodeTileEdge".to_owned(), serde_json::json!(512)),
        ("decodeOverlap".to_owned(), serde_json::json!(128)),
        ("attentionChunkSize".to_owned(), serde_json::json!(256)),
    ]);
    bundle.records = vec![
        rung(StrategyRung::Resident, resident_parameters.clone(), 7.0),
        rung(StrategyRung::BoundedDecode, decode_parameters.clone(), 5.0),
        rung(
            StrategyRung::BoundedAttention,
            attention_parameters.clone(),
            4.0,
        ),
    ];
    let mut plan = fixture_plan();
    let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
        panic!("fixture calibration");
    };
    calibration.bindings = vec![
        fixture_binding_for(
            "q4",
            "packed-q4",
            StrategyRung::Resident,
            resident_parameters,
        ),
        fixture_binding_for(
            "q4",
            "packed-q4",
            StrategyRung::BoundedDecode,
            decode_parameters,
        ),
        fixture_binding_for(
            "q4",
            "packed-q4",
            StrategyRung::BoundedAttention,
            attention_parameters,
        ),
    ];
    (bundle, plan)
}

#[test]
fn verified_lower_geometry_requires_exact_identity_and_geometry() {
    let mut bundle = fixture_bundle();
    let mut lower_record = bundle.records.remove(0);
    lower_record.target.geometry = CalibrationGeometry {
        width: 768,
        height: 768,
        batch: 1,
        frames: 1,
    };
    bundle.records = vec![lower_record];

    let mut plan = fixture_plan();
    let mut lower_binding = fixture_binding("q4", "packed-q4");
    lower_binding.geometry = CalibrationGeometry {
        width: 768,
        height: 768,
        batch: 1,
        frames: 1,
    };
    let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
        panic!("fixture calibration");
    };
    calibration.bindings = vec![lower_binding.clone()];
    let inputs = fixture_inputs(1024, 1024);
    let MlxCalibrationConfig::Valid(calibration) = &plan.calibration else {
        panic!("fixture calibration");
    };
    assert_eq!(
        verified_lower_geometry(
            &bundle,
            calibration,
            &plan,
            &inputs,
            "text_to_image",
            fixture_budget(128.0),
        ),
        Some(CalibrationGeometry {
            width: 768,
            height: 768,
            batch: 1,
            frames: 1,
        })
    );

    // sc-22738: a MOVED provider closure is not one of the exactness conditions. Between
    // sc-17774 and this story the alternative was withheld when the binding's closure had
    // moved, on the argument that naming a geometry the next request would refuse for the same
    // staleness was worse than naming none. There is no such refusal any more — the ladder is
    // priced at its measured peak whatever closure it was captured under — so the advice is
    // honoured either way. Same binding, same bundle, same budget: only the digest moves.
    let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
        panic!("fixture calibration");
    };
    calibration.bindings[0].query.inference_closure_digest = "a".repeat(64);
    let MlxCalibrationConfig::Valid(calibration) = &plan.calibration else {
        panic!("fixture calibration");
    };
    assert_eq!(
        verified_lower_geometry(
            &bundle,
            calibration,
            &plan,
            &inputs,
            "text_to_image",
            fixture_budget(128.0),
        ),
        Some(CalibrationGeometry {
            width: 768,
            height: 768,
            batch: 1,
            frames: 1,
        }),
        "a moved provider closure must NOT stop the refusal from naming the lower geometry"
    );

    let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
        panic!("fixture calibration");
    };
    calibration.bindings[0].query.inference_closure_digest = fixture_closure_digest();
    calibration.bindings[0].query.fingerprint = "mutated".to_owned();
    let MlxCalibrationConfig::Valid(calibration) = &plan.calibration else {
        panic!("fixture calibration");
    };
    assert_eq!(
        verified_lower_geometry(
            &bundle,
            calibration,
            &plan,
            &inputs,
            "text_to_image",
            fixture_budget(128.0),
        ),
        None,
        "a fingerprint mutation must stop the refusal from naming the lower geometry"
    );

    let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
        panic!("fixture calibration");
    };
    calibration.bindings[0] = lower_binding;
    calibration.bindings[0].geometry.width = 640;
    let MlxCalibrationConfig::Valid(calibration) = &plan.calibration else {
        panic!("fixture calibration");
    };
    assert_eq!(
        verified_lower_geometry(
            &bundle,
            calibration,
            &plan,
            &inputs,
            "text_to_image",
            fixture_budget(128.0),
        ),
        None,
        "a geometry mutation must stop the refusal from naming the lower geometry"
    );
}

#[test]
fn exact_infeasible_geometry_refuses_before_provider_and_names_only_verified_lower_geometry() {
    use sceneworks_core::memory_calibration::RequiredNullable;

    let mut bundle = fixture_bundle();
    let mut high_record = bundle.records.remove(0);
    let mut lower_record = high_record.clone();
    lower_record.target.geometry = CalibrationGeometry {
        width: 768,
        height: 768,
        batch: 1,
        frames: 1,
    };
    let RequiredNullable::Value(predicted) = &mut high_record.predicted_peak_bytes else {
        panic!("fixture predicted peak");
    };
    let predicted = predicted
        .full_mut()
        .expect("fixture has full phase telemetry");
    predicted.denoise = gib_to_bytes(6.0);
    predicted.overall = gib_to_bytes(6.0);
    bundle.records = vec![high_record, lower_record];

    let mut plan = fixture_plan();
    let mut lower_binding = fixture_binding("q4", "packed-q4");
    lower_binding.geometry = CalibrationGeometry {
        width: 768,
        height: 768,
        batch: 1,
        frames: 1,
    };
    let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
        panic!("fixture calibration");
    };
    calibration.bindings.push(lower_binding);

    let error = evaluate_request_with_budget_using_bundle(
        &fixture_generator(),
        &plan,
        &fixture_inputs(1024, 1024),
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(8.0),
        gib_to_bytes(6.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect_err("the 6 GiB high record plus its 3 GiB foreign reserve must refuse");
    let message = error.to_string();
    assert!(
        message.contains("smallest verified MLX host boundary"),
        "the refusal must come from the post-handshake exact evidence precheck: {message}"
    );
    assert!(
        message.contains("current verified alternative: 768x768"),
        "the refusal must name the lower exact record: {message}"
    );
}

#[test]
fn packaged_krea_1024_admits_by_estimate_and_refuses_below_the_widened_margins() {
    use gen_core::MemoryCalibrationIdentity;

    // sc-22512 (E8): both fixtures are LOOKED UP. If the packaged bundle no longer supplies
    // the two-cell Krea control ladder this test stands on, the question is withheld rather
    // than the suite reddening on an absent measurement.
    let Some(plan) = packaged_krea_plan() else {
        return;
    };
    // Realistic component facts so the floor arithmetic is meaningful: conditioning 8 GiB,
    // transformer 60 GiB, decoder 4 GiB. Floors at 1024² (headroom = 2 GiB fixture anchor):
    //   staged (rung 1)       max(8, 64) + 2 = 66 GiB  -> widened 99.27 GiB
    //   bounded decode floor  (8 + 64)  + 2 = 74 GiB  -> widened 111.30 GiB (never used: the
    //                                                   measured 896² basis supplies a fitted
    //                                                   curve instead)
    // Fitted bounded-decode estimate from the 896² record: envelope 38.563 GiB scaled by
    // 1024²/896² = 50.37 GiB, widened by the measured MLX estimate margin to 75.76 GiB.
    let Some(mut generator) = packaged_krea_generator() else {
        return;
    };
    {
        let facts = &mut generator
            .contract
            .as_mut()
            .expect("Krea contract")
            .asset_facts;
        facts.conditioning_bytes = gib_to_bytes(8.0);
        facts.transformer_bytes = gib_to_bytes(60.0);
        facts.decoder_bytes = gib_to_bytes(4.0);
        facts.base_bytes = gib_to_bytes(72.0);
    }
    let generator = generator;
    let mut inputs = fixture_inputs(1024, 1024);
    inputs.overlay = Some("control:1".to_owned());
    let evaluate = |generator: &RequestGenerator, budget_gib: f64| {
        evaluate_request_with_budget_using_bundle(
            generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            crate::execution_planner::WarmPolicyProposal::inert("fixture"),
            fixture_budget(budget_gib),
            gib_to_bytes(130.0),
            0,
            &[],
            Some(&packaged_bundle()),
            None,
        )
    };

    // sc-18096 headline: 1024² has no measured cell, and before this story it REFUSED here.
    // Now the estimate ladder admits it. At 128 GiB the cheapest fitting rung is the staged
    // floor; the run context is legacy-scoped (no record id, no request-scoped ceiling).
    let admitted = evaluate(&generator, 128.0)
        .expect("an unmeasured geometry must be admitted by the estimate ladder");
    assert_eq!(
        admitted.context.selection.strategy,
        MemoryStrategy::StagedResidency,
        "the cheapest fitting estimate rung must win: {:?}",
        admitted.context.selection
    );
    assert_eq!(admitted.process_limit_bytes, None);
    assert_eq!(
        admitted.context.evidence_revision,
        REQUEST_EVIDENCE_REVISION
    );

    // At 65 GiB the staged floor no longer fits, and the FITTED bounded-decode estimate
    // extrapolated from the measured 896² cell (50.37 GiB raw, 56.71 GiB at its same-cell
    // recapture ceiling) is selected — with the measured cell's own sweep parameters, which a
    // floor synthesis (built from the smallest declared ranges) could not produce at this
    // budget.
    let fitted = evaluate(&generator, 65.0)
        .expect("the fitted-curve estimate must admit where the floors cannot");
    assert_eq!(
        fitted.context.selection.strategy,
        MemoryStrategy::BoundedDecode
    );
    assert_eq!(
        fitted.context.selection.parameters.decode_tile_edge,
        Some(512),
        "the fitted estimate must carry the measured basis' parameters"
    );
    assert_eq!(
        fitted.context.selection.parameters.decode_overlap,
        Some(64),
        "the fitted estimate must carry the measured basis' parameters"
    );

    // Below every admitted estimate the request still refuses — allowances are load-bearing. No
    // verified alternative fits 40 GiB (the 768² cell needs its captured foreign reserve too),
    // so none may be named.
    let message = evaluate(&generator, 40.0)
        .expect_err("below the widened estimates the refusal must remain")
        .to_string();
    assert!(
        message.contains("needs") && message.contains("safely available"),
        "the refusal keeps the actionable needs/available format: {message}"
    );
    assert!(
        !message.contains("current verified alternative"),
        "no verified cell fits 40 GiB, so none may be named: {message}"
    );

    let mut exact_896 = inputs.clone();
    exact_896.width = 896;
    exact_896.height = 896;
    let exact = evaluate_request_with_budget_using_bundle(
        &generator,
        &plan,
        &exact_896,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(83.0),
        gib_to_bytes(130.0),
        0,
        &[],
        Some(&packaged_bundle()),
        None,
    )
    .expect("the exact 896 cell fits once its 128 GiB-host reserve is normalized to 83 GiB");
    assert_eq!(
        exact.context.selection.strategy,
        MemoryStrategy::BoundedDecode
    );
    assert_eq!(
        exact.context.evidence_revision, "imc-2cd840a85ce33b4f22a9",
        "host normalization must admit the exact verified cell, not silently demote to estimates"
    );

    // A loaded provider whose calibration identity DRIFTED from the packaged records must not
    // receive fitted-curve candidates built from them (sc-18096 review, major finding). The
    // mutated fingerprint is deliberately WELL-FORMED (`-v1` satisfies the contract's version
    // token conformance rule), so the refusal below is the work of the basis identity gate in
    // `synthesize_estimate_ladder`, not an accidental contract-conformance `Invalid`. And it
    // runs at 76 GiB, a budget where NO lower alternative is named (both packaged cells need
    // their ~47 GiB captured foreign reserve), so `carries_verified_claim` is FALSE and the
    // fingerprint demotion at the admission seam never fires — the synthesis-side gate is the
    // only thing standing between the drifted provider and the fitted candidate. The
    // unmutated generator ADMITS at 76 (the fitted arm above), so the refusal is exactly the
    // mutation's doing.
    let Some(mut mismatched_generator) = packaged_krea_generator() else {
        return;
    };
    {
        let contract = mismatched_generator
            .contract
            .as_mut()
            .expect("Krea contract");
        contract.asset_facts = generator.contract.as_ref().expect("facts").asset_facts;
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            "mutated-loaded-provider-v1",
            gen_core::LoadShape::EagerMaterialization,
        ));
    }
    assert!(
        mismatched_generator
            .contract
            .as_ref()
            .expect("Krea contract")
            .conformance_errors()
            .is_empty(),
        "the mutated fingerprint must be conformance-CLEAN so this arm exercises the basis \
             identity gate, not format validation"
    );
    let message = evaluate(&mismatched_generator, 65.0)
        .expect_err("a fingerprint-drifted provider loses the measured basis and refuses")
        .to_string();
    assert!(
        message.contains("needs") && message.contains("safely available"),
        "the drifted provider's refusal is the floors-only Reject — the fitted candidate was \
             suppressed by the basis identity gate: {message}"
    );
    assert!(
        !message.contains("current verified alternative"),
        "a loaded-provider fingerprint mutation must suppress evidence-derived naming: {message}"
    );
    // The gate is precisely scoped to the MEASURED basis: the floors derive from the
    // contract's own asset facts, not from any record, so the drifted provider still admits
    // where a floor fits.
    let floor_admitted = evaluate(&mismatched_generator, 128.0)
        .expect("a drifted provider keeps its no-measured-basis floor estimates");
    assert_eq!(
        floor_admitted.context.selection.strategy,
        MemoryStrategy::StagedResidency
    );

    // Dropping the rung's Implemented support removes both the fitted candidate and its floor:
    // an unimplemented rung is never estimate-admissible.
    let Some(mut unimplemented_generator) = packaged_krea_generator() else {
        return;
    };
    {
        let contract = unimplemented_generator
            .contract
            .as_mut()
            .expect("Krea contract");
        contract.asset_facts = generator.contract.as_ref().expect("facts").asset_facts;
        contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
            .expect("bounded decode capability")
            .support = gen_core::MemoryStrategySupport::Missing;
    }
    let message = evaluate(&unimplemented_generator, 65.0)
        .expect_err("an unimplemented rung is never estimate-admissible")
        .to_string();
    assert!(
        !message.contains("current verified alternative"),
        "a loaded-provider composition mutation must suppress evidence-derived naming: {message}"
    );

    // sc-22738: this used to fork on the digest pair. A fitted-curve estimate could extrapolate
    // only from CLOSURE-CURRENT records, on the argument that the estimate margin was derived
    // over same-closure re-capture variance and could not also absorb closure drift; the same
    // request therefore admitted or refused depending on where the pin happened to sit. It no
    // longer forks: a measured cell is a legitimate extrapolation basis whatever closure it was
    // captured under, so the 60 GiB request always reaches the fitted bounded-decode rung.
    let admitted = evaluate(&generator, 60.0)
        .expect("the fitted estimate admits the 60 GiB request regardless of closure drift");
    assert_eq!(
        admitted.context.selection.strategy,
        MemoryStrategy::BoundedDecode
    );
}

/// The fixture generator with the FULL ladder implemented, including rung 4 with its
/// deferred-materialization prerequisite — the shape an unmeasured provider needs for the
/// sc-18096 estimate ladder to reach every rung.
fn full_ladder_generator() -> RequestGenerator {
    use gen_core::{MemoryCalibrationIdentity, MemoryStrategySupport};

    let mut generator = fixture_generator();
    let contract = generator.contract.as_mut().expect("fixture contract");
    contract.load_shape = gen_core::LoadShape::DeferredMaterialization;
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        "fixture-formula-v2",
        gen_core::LoadShape::DeferredMaterialization,
    ));
    contract.lifecycle.transformer_window_materialization = true;
    // All-transformer weights big enough that rung 4's floor undercuts the resident
    // baseline under the re-derived activation allowance — see FULL_LADDER_WEIGHTS_GB.
    contract.asset_facts.base_bytes = gib_to_bytes(FULL_LADDER_WEIGHTS_GB);
    contract.asset_facts.transformer_bytes = gib_to_bytes(FULL_LADDER_WEIGHTS_GB);
    contract.phase_facts = Some(gen_core::MemoryPhaseFacts {
        architecture: None,
        staged_weights: gen_core::StagedWeightSchedule::TwoStage,
        transformer_stream: Some(gen_core::StreamedWeightFacts {
            resident_bytes: 0,
            stacks: vec![vec![
                gib_to_bytes(
                    FULL_LADDER_WEIGHTS_GB / f64::from(FULL_LADDER_FIXTURE_BLOCKS)
                );
                FULL_LADDER_FIXTURE_BLOCKS as usize
            ]],
        }),
        decoder_workspace: None,
        native_quant: None,
    });
    let rung4 = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
        .expect("rung 4 capability");
    rung4.support = MemoryStrategySupport::Implemented;
    rung4.parameters.transformer_window_sizes = vec![2, 4];
    generator
}

#[test]
fn an_unregistered_gate_cannot_select_an_optimized_contract_or_borrow_a_family() {
    let mut generator = full_ladder_generator();
    generator.contract.as_mut().unwrap().provider_id = "unregistered_future_provider".into();
    let mut plan = fixture_plan();
    plan.engine_id = "unregistered_future_provider";
    plan.calibration = MlxCalibrationConfig::Absent;
    let inputs = fixture_inputs(1024, 1024);
    let evaluation = evaluate_request_with_budget(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        fixture_budget(128.0),
        full_ladder_baseline_bytes(),
        0,
        &[],
    )
    .expect("legacy resident fallback remains available");
    assert_eq!(
        evaluation.context.selection.strategy,
        MemoryStrategy::Resident
    );
    assert!(
        evaluate_request_with_budget(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            budget_admitting_only_the_deepest_estimate_rung(),
            full_ladder_baseline_bytes(),
            0,
            &[]
        )
        .is_err(),
        "an unknown family cannot inherit rung four"
    );
}

/// Unknown activation splits and declared splits receive the same active-peak uncertainty.
#[test]
fn the_legacy_resident_baseline_is_graded_without_a_declared_activation_split() {
    let generator = full_ladder_generator();
    let plan = fixture_plan();
    let inputs = fixture_inputs(1024, 1024);
    let raw = gib_to_bytes(FULL_LADDER_WEIGHTS_GB + FIXTURE_HEADROOM_GB);
    let admitted = crate::memory_strategy::floor_admitted_peak_bytes(MemoryBackend::Mlx, raw, None);
    assert_eq!(
        admitted,
        crate::memory_strategy::floor_admitted_peak_bytes(
            MemoryBackend::Mlx,
            raw,
            Some(gib_to_bytes(FIXTURE_HEADROOM_GB))
        )
    );
    let evaluation = evaluate_request_with_budget(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        fixture_budget(admitted as f64 / BYTES_PER_GIB + 1.0),
        raw,
        0,
        &[],
    )
    .unwrap();
    assert_eq!(
        evaluation.context.selection.strategy,
        MemoryStrategy::Resident
    );
}

#[test]
fn unmeasured_provider_under_a_small_budget_selects_a_deep_estimate_rung() {
    // sc-22667: the fixture's block count through the production facts seam.
    let _facts = crate::video_admission::inject_architecture_facts(FULL_LADDER_FIXTURE_FACTS);
    let generator = full_ladder_generator();
    let mut plan = fixture_plan();
    plan.calibration = MlxCalibrationConfig::Absent;
    let inputs = fixture_inputs(1024, 1024);
    let evaluation = evaluate_request_with_budget(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        budget_admitting_only_the_deepest_estimate_rung(),
        full_ladder_baseline_bytes(),
        0,
        &[],
    )
    .expect("the estimate ladder must admit the deep rung instead of refusing");
    assert_eq!(
        evaluation.context.selection.strategy,
        MemoryStrategy::BoundedTransformerResidency,
        "only rung 4's floor fits a 9.1 GiB budget: {:?}",
        evaluation.context.selection
    );
    assert_eq!(
        evaluation.context.optimization_authority,
        MemoryOptimizationAuthority::Estimated
    );
    // The floor synthesizes the most deeply bounding declared parameters.
    assert_eq!(
        evaluation.context.selection.parameters.decode_tile_edge,
        Some(512)
    );
    assert_eq!(
        evaluation.context.selection.parameters.decode_overlap,
        Some(128)
    );
    assert_eq!(
        evaluation.context.selection.parameters.attention_chunk_size,
        Some(256)
    );
    assert_eq!(
        evaluation
            .context
            .selection
            .parameters
            .transformer_window_size,
        Some(2),
        "the smallest declared window is the most bounding"
    );
    // And the selection translates to the engine knobs the engaged composition names:
    // rung 4 engages bounded decode, bounded attention, and block streaming — but NOT staged
    // residency, which the shared cost order keeps explicit-selection-only.
    assert!(evaluation.memory.tile_vae_decode);
    assert!(evaluation.memory.chunk_attention);
    assert!(evaluation.memory.stream_transformer_blocks);
    assert!(!evaluation.memory.stage_residency);
    // Legacy-scoped telemetry: an estimate admission never claims a verified record or a
    // request-scoped process ceiling.
    assert_eq!(evaluation.process_limit_bytes, None);
    assert_eq!(
        evaluation.context.evidence_revision,
        REQUEST_EVIDENCE_REVISION
    );

    // Mutation arm: at 7 GiB even the rung-4 admitted floor (6 GiB of pure activation headroom
    // widened by `FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE`) overflows, and the refusal is the honest
    // Reject quoting the admitted requirement — proving the per-term allowance is applied on
    // this path (a zeroed allowance would admit 6.0 <= 7.0). The budget is sized through the
    // policy rather than against a transcribed widened figure, which is what went stale here
    // when the allowance was re-derived.
    let error = evaluate_request_with_budget(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        fixture_budget(widened_estimate_gb(FIXTURE_DEEP_ESTIMATE_FLOOR_GB) - 0.1),
        full_ladder_baseline_bytes(),
        0,
        &[],
    )
    .expect_err("below every admitted estimate the request must refuse")
    .to_string();
    let rung4_admitted = format!(
        "needs {:.2} GiB",
        widened_estimate_gb(FIXTURE_DEEP_ESTIMATE_FLOOR_GB)
    );
    assert!(
        error.contains(&rung4_admitted),
        "the refusal must quote the ADMITTED rung-4 floor ({rung4_admitted}): {error}"
    );
}

#[test]
fn pulid_identity_route_is_estimated_only_and_refusal_is_terminal() {
    // sc-22667: the fixture's block count through the production facts seam.
    let _facts = crate::video_admission::inject_architecture_facts(FULL_LADDER_FIXTURE_FACTS);
    let mut generator = full_ladder_generator();
    generator.descriptor.id = "pulid_flux";
    let contract = generator.contract.as_mut().expect("fixture contract");
    contract.provider_id = "pulid_flux".to_owned();
    contract.calibration = None;
    let decode = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
        .expect("bounded decode capability");
    decode.parameters.decode_tile_edges = vec![768, 640, 512];
    decode.parameters.decode_overlaps = vec![64];
    let attention = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedAttention)
        .expect("bounded attention capability");
    attention.parameters.attention_chunk_sizes = vec![67_108_864];
    let transformer = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
        .expect("bounded transformer capability");
    transformer.parameters.transformer_window_sizes = vec![1];

    let mut plan = fixture_plan();
    plan.engine_id = "pulid_flux";
    plan.model_id = "pulid_flux_dev".to_owned();
    plan.calibration = MlxCalibrationConfig::Absent;
    plan.load_shape_declaration_result = LoadShapeDeclarationResult::Applied;
    let mut inputs = fixture_inputs(1024, 1024);
    inputs.mode = "character_image".to_owned();
    inputs.overlay = Some("identity".to_owned());
    inputs.has_reference = true;
    inputs.reference_count = 1;

    let evaluation = evaluate_request_with_budget(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Sequential,
        budget_admitting_only_the_deepest_estimate_rung(),
        full_ladder_baseline_bytes(),
        0,
        &[],
    )
    .expect("the exact unmeasured PuLID identity route must reach the estimate ladder");
    assert_eq!(
        evaluation.context.selection.strategy,
        MemoryStrategy::BoundedTransformerResidency
    );
    assert_eq!(
        evaluation.context.optimization_authority,
        MemoryOptimizationAuthority::Estimated,
        "PuLID must not inherit flux_dev measurements"
    );
    assert_eq!(evaluation.context.mode, MemoryMode::ImageToImage);
    assert_eq!(evaluation.context.overlay.as_deref(), Some("identity"));
    assert_eq!(evaluation.context.geometry.reference_count, 1);
    assert_eq!(
        evaluation.context.selection.parameters.decode_tile_edge,
        Some(512)
    );
    assert_eq!(
        evaluation.context.selection.parameters.decode_overlap,
        Some(64)
    );
    assert_eq!(
        evaluation.context.selection.parameters.attention_chunk_size,
        Some(67_108_864)
    );
    assert_eq!(
        evaluation
            .context
            .selection
            .parameters
            .transformer_window_size,
        Some(1)
    );
    assert!(evaluation.process_limit_bytes.is_none());

    let mut refused = plan.clone();
    refused.load_shape_declaration_result = LoadShapeDeclarationResult::Refused;
    let resident = evaluate_request_with_budget(
        &generator,
        &refused,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        fixture_budget(20.0),
        gib_to_bytes(9.0),
        0,
        &[],
    )
    .expect("a refused declaration keeps the safe roomy resident path");
    assert_eq!(
        resident.context.selection.strategy,
        MemoryStrategy::Resident
    );
    assert_eq!(
        resident.context.optimization_authority,
        MemoryOptimizationAuthority::Resident
    );
    // Deliberately the SAME budget as the admitted arm above: the only difference between the two
    // is `LoadShapeDeclarationResult::Applied` vs `Refused`, so the refusal is attributable to the
    // declaration rather than to a budget that would have refused either way. (It previously read
    // 8.0 GiB, which under the current margin refuses every rung and would have passed for the
    // wrong reason.)
    let error = evaluate_request_with_budget(
        &generator,
        &refused,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        budget_admitting_only_the_deepest_estimate_rung(),
        full_ladder_baseline_bytes(),
        0,
        &[],
    )
    .expect_err("a refused over-budget eager request cannot rediscover an optimized rung")
    .to_string();
    assert!(
        error.contains("needs") && error.contains("safely available"),
        "terminal refusal must surface the safe resident fit failure: {error}"
    );
}

#[test]
fn krea_raw_estimate_floor_selects_exact_native_and_pid_decode_domains() {
    // sc-22667: the fixture's block count through the production facts seam.
    let _facts = crate::video_admission::inject_architecture_facts(FULL_LADDER_FIXTURE_FACTS);
    let mut generator = full_ladder_generator();
    generator.descriptor.id = "krea_2_raw";
    let contract = generator.contract.as_mut().expect("fixture contract");
    contract.provider_id = "krea_2_raw".to_owned();
    let decode = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
        .expect("bounded decode capability");
    decode.parameters.decode_tile_edges = vec![2048, 512];
    decode.parameters.decode_overlaps = vec![256, 64];
    contract.pid_decode_routes = Some(gen_core::MemoryPidDecodeRoutes {
        native: gen_core::MemoryDecodeRouteDomain {
            tile_edges: vec![512],
            tile_overlap: 64,
        },
        pid: gen_core::MemoryDecodeRouteDomain {
            tile_edges: vec![2048],
            tile_overlap: 256,
        },
    });
    let mut plan = fixture_plan();
    plan.engine_id = "krea_2_raw";
    plan.model_id = "krea_2_raw".to_owned();
    plan.calibration = MlxCalibrationConfig::Absent;
    for (profile, adapter_count, use_pid, expected_edge, expected_overlap) in [
        ("plain", 0, false, 512, 64),
        ("low-rank", 1, false, 512, 64),
        ("plain+pid", 0, true, 2048, 256),
        ("low-rank+pid", 1, true, 2048, 256),
    ] {
        let mut inputs = fixture_inputs(1024, 1024);
        inputs.use_pid = use_pid;
        inputs.adapter_count = adapter_count;
        let evaluation = evaluate_request_with_budget(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Sequential,
            budget_admitting_only_the_deepest_estimate_rung(),
            full_ladder_baseline_bytes(),
            0,
            &[],
        )
        .expect("the exact Raw decode domain must admit the deep estimate rung");
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency
        );
        assert_eq!(
            evaluation.context.selection.parameters.decode_tile_edge,
            Some(expected_edge),
            "profile={profile}"
        );
        assert_eq!(
            evaluation.context.selection.parameters.decode_overlap,
            Some(expected_overlap),
            "profile={profile}"
        );
        assert_eq!(evaluation.context.use_pid, use_pid);
        assert_eq!(evaluation.context.overlay, None);
    }
}

#[test]
fn flux2_klein_provider_axes_and_estimate_floor_keep_native_and_pid_domains_exact() {
    // sc-22667: the fixture's block count through the production facts seam.
    let _facts = crate::video_admission::inject_architecture_facts(FULL_LADDER_FIXTURE_FACTS);
    let mut composed = LoadSpec::new(WeightsSource::Dir("klein".into()))
        .with_adapters(vec![gen_core::AdapterSpec::new(
            "adapter.safetensors".into(),
            1.0,
            gen_core::AdapterKind::Lora,
        )])
        .with_control(WeightsSource::File("control.safetensors".into()))
        .with_extra_control(WeightsSource::File("extra-control.safetensors".into()))
        .with_ip_adapter(WeightsSource::Dir("ip-adapter".into()))
        .with_text_encoder(WeightsSource::Dir("external-text-encoder".into()))
        .with_component(
            "unknown",
            WeightsSource::File("component.safetensors".into()),
        );
    composed.identity = Some(gen_core::IdentityWeights::default());
    assert_eq!(
        provider_overlay_for_load_spec("flux2_klein_9b", &composed, None).as_deref(),
        Some(
            "adapters-control-extra-controls-ip-adapter-identity-external-text-encoder-components"
        ),
        "Klein provider identity follows the pinned provider's complete closed axis order"
    );
    assert_eq!(
        provider_overlay_for_load_spec("krea_2_raw", &composed, Some("public-label".to_owned())),
        None,
        "Krea base-DiT provider overlay remains None for the same typed load axes"
    );

    let public_base = MlxRequestInputs {
        native_batch: 1,
        width: 1024,
        height: 1024,
        count: 1,
        mode: "text_to_image".to_owned(),
        overlay: Some("adapters".to_owned()),
        adapter_count: 1,
        has_reference: false,
        reference_count: 0,
        use_pid: false,
        has_phases: false,
        conditioning_windows: None,
        negative_branch: false,
        prefix_kv_cache_eligible: false,
        reference_resources: None,
    };
    assert_eq!(
        provider_request_mode("flux2_klein_9b", &public_base),
        (MemoryMode::TextToImage, "text_to_image")
    );
    assert_eq!(
        provider_request_inputs("flux2_klein_9b", &public_base)
            .as_ref()
            .overlay,
        Some("adapters".to_owned()),
        "Klein low-rank loads use the provider's one exact adapter identity"
    );

    let reference = MlxRequestInputs {
        native_batch: 1,
        has_reference: true,
        reference_count: 1,
        ..public_base.clone()
    };
    assert_eq!(
        provider_request_mode("flux2_klein_9b", &reference),
        (MemoryMode::ImageToImage, "image_to_image")
    );
    for provider in ["flux2_klein_9b_edit", "flux2_klein_9b_kv_edit"] {
        for references in 1..=8 {
            let edit = MlxRequestInputs {
                native_batch: 1,
                mode: "character_image".to_owned(),
                reference_count: references,
                ..reference.clone()
            };
            assert_eq!(
                provider_request_mode(provider, &edit),
                (MemoryMode::Edit, "edit"),
                "{provider} refs={references}"
            );
            assert_eq!(
                provider_request_inputs(provider, &edit).as_ref().overlay,
                Some("adapters".to_owned())
            );
        }
    }

    for (provider, mode, references) in [
        ("flux2_klein_9b", "text_to_image", 0),
        ("flux2_klein_9b", "text_to_image", 1),
        ("flux2_klein_9b_edit", "edit_image", 8),
        ("flux2_klein_9b_kv_edit", "character_image", 2),
    ] {
        let mut generator = full_ladder_generator();
        generator.descriptor.id = provider;
        let contract = generator.contract.as_mut().expect("fixture contract");
        contract.provider_id = provider.to_owned();
        contract.calibration = None;
        let decode = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
            .expect("bounded decode capability");
        decode.parameters.decode_tile_edges = vec![2048, 768, 640, 512];
        decode.parameters.decode_overlaps = vec![256, 128];
        contract.pid_decode_routes = Some(gen_core::MemoryPidDecodeRoutes {
            native: gen_core::MemoryDecodeRouteDomain {
                tile_edges: vec![768, 640, 512],
                tile_overlap: 128,
            },
            pid: gen_core::MemoryDecodeRouteDomain {
                tile_edges: vec![2048],
                tile_overlap: 256,
            },
        });

        let mut plan = fixture_plan();
        plan.engine_id = provider;
        plan.model_id = "flux2_klein_9b".to_owned();
        plan.calibration = MlxCalibrationConfig::Absent;
        plan.load_shape_declaration_result = LoadShapeDeclarationResult::Applied;
        plan.tier.quant = None;
        let plan = plan
            .with_resolved_artifact_tier(Some("q4"))
            .expect("packed q4 tier is explicit without load-time quantization");

        for (use_pid, edge, overlap) in [(false, 512, 128), (true, 2048, 256)] {
            let inputs = MlxRequestInputs {
                native_batch: 1,
                width: 1024,
                height: 1024,
                count: 1,
                mode: mode.to_owned(),
                overlay: None,
                adapter_count: 0,
                has_reference: references > 0,
                reference_count: references,
                use_pid,
                has_phases: false,
                conditioning_windows: None,
                negative_branch: false,
                prefix_kv_cache_eligible: false,
                reference_resources: None,
            };
            let evaluation = evaluate_request_with_budget(
                &generator,
                &plan,
                &inputs,
                MemoryCacheState::Cold,
                OffloadPolicy::Sequential,
                budget_admitting_only_the_deepest_estimate_rung(),
                full_ladder_baseline_bytes(),
                0,
                &[],
            )
            .expect("the exact Klein estimate route must reach BTR");
            assert_eq!(
                evaluation.context.selection.strategy,
                MemoryStrategy::BoundedTransformerResidency,
                "{provider} pid={use_pid}"
            );
            assert_eq!(
                evaluation.context.optimization_authority,
                MemoryOptimizationAuthority::Estimated,
                "Klein aliases must not inherit measured Flux or sibling authority"
            );
            assert_eq!(
                evaluation.context.selection.parameters.decode_tile_edge,
                Some(edge)
            );
            assert_eq!(
                evaluation.context.selection.parameters.decode_overlap,
                Some(overlap)
            );
            assert_eq!(evaluation.context.overlay, None);
            assert_eq!(evaluation.context.geometry.reference_count, references);
            assert_eq!(evaluation.context.use_pid, use_pid);
        }

        let low_rank = MlxRequestInputs {
            native_batch: 1,
            width: 1024,
            height: 1024,
            count: 1,
            mode: mode.to_owned(),
            overlay: Some("adapters".to_owned()),
            adapter_count: 1,
            has_reference: references > 0,
            reference_count: references,
            use_pid: true,
            has_phases: false,
            conditioning_windows: None,
            negative_branch: false,
            prefix_kv_cache_eligible: false,
            reference_resources: None,
        };
        let mut refused = plan.clone();
        refused.load_shape_declaration_result = LoadShapeDeclarationResult::Refused;
        let resident = evaluate_request_with_budget(
            &generator,
            &refused,
            &low_rank,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(20.0),
            gib_to_bytes(9.0),
            0,
            &[],
        )
        .expect("LoRA/LoKr plus PiD stays executable on the safe eager path");
        assert_eq!(
            resident.context.selection.strategy,
            MemoryStrategy::Resident
        );
        assert_eq!(
            resident.context.optimization_authority,
            MemoryOptimizationAuthority::Resident
        );
        // Same budget as the admitted arm on purpose — it admits rung 4 and nothing shallower, so
        // "eager cannot fit" is a statement about the eager path, not about a budget too small for
        // anything at all.
        let error = evaluate_request_with_budget(
            &generator,
            &refused,
            &low_rank,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            budget_admitting_only_the_deepest_estimate_rung(),
            full_ladder_baseline_bytes(),
            0,
            &[],
        )
        .expect_err("a refused low-rank route cannot rediscover BTR when eager cannot fit")
        .to_string();
        assert!(
            error.contains("needs") && error.contains("safely available"),
            "{error}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn flux2_klein_refused_low_rank_routes_remain_executable_through_real_provider_scope() {
    let registry = crate::inference_runtime::media();
    for (provider, public_mode, provider_mode, reference_count) in [
        (
            "flux2_klein_9b",
            "text_to_image",
            MemoryMode::TextToImage,
            0,
        ),
        ("flux2_klein_9b_edit", "edit_image", MemoryMode::Edit, 1),
    ] {
        let registration = registry
            .memory_strategy_registrations()
            .find(|registration| registration.provider_id == provider)
            .unwrap_or_else(|| panic!("{provider} memory registration"));
        let fixture = registry
            .memory_contract_fixture_registrations()
            .find(|fixture| fixture.provider_id == provider)
            .unwrap_or_else(|| panic!("{provider} weights-free contract fixture"));
        let behavior = registry
            .memory_behavior_registrations()
            .find(|behavior| behavior.provider_id == provider)
            .unwrap_or_else(|| panic!("{provider} behavior registration"));

        for (label, kind, external_text_encoder, expected_overlay) in [
            ("lora", Some(gen_core::AdapterKind::Lora), false, "adapters"),
            ("lokr", Some(gen_core::AdapterKind::Lokr), false, "adapters"),
            ("external-te", None, true, "external-text-encoder"),
            (
                "lora-external-te",
                Some(gen_core::AdapterKind::Lora),
                true,
                "adapters-external-text-encoder",
            ),
            (
                "lokr-external-te",
                Some(gen_core::AdapterKind::Lokr),
                true,
                "adapters-external-text-encoder",
            ),
        ] {
            let weights = tempfile::tempdir().expect("Klein weights fixture");
            let mut spec = LoadSpec::new(WeightsSource::Dir(weights.path().to_owned()));
            if let Some(kind) = kind {
                spec = spec.with_adapters(vec![gen_core::AdapterSpec::new(
                    weights.path().join("adapter.safetensors"),
                    1.0,
                    kind,
                )]);
            }
            if external_text_encoder {
                spec = spec.with_text_encoder(WeightsSource::Dir(
                    weights.path().join("external-text-encoder"),
                ));
            }
            let contract = (fixture.contract)(&spec)
                .unwrap_or_else(|error| panic!("{provider} weights-free contract: {error}"));
            let mut generator = fixture_generator();
            generator.descriptor.id = provider;
            generator.contract = Some(contract.clone());

            let mut plan = fixture_plan();
            plan.engine_id = provider;
            plan.model_id = provider.to_owned();
            plan.tier.quant = None;
            plan.calibration = MlxCalibrationConfig::Absent;
            plan.load_shape_declaration_result = LoadShapeDeclarationResult::Refused;
            let plan = plan
                .with_resolved_artifact_tier(Some("bf16"))
                .expect("Klein dense tier remains explicit");
            let inputs = MlxRequestInputs {
                native_batch: 1,
                width: 1024,
                height: 1024,
                count: 1,
                mode: public_mode.to_owned(),
                overlay: provider_overlay_for_load_spec(provider, &spec, None),
                adapter_count: usize::from(kind.is_some()),
                has_reference: reference_count > 0,
                reference_count,
                use_pid: false,
                has_phases: false,
                conditioning_windows: None,
                negative_branch: false,
                prefix_kv_cache_eligible: false,
                reference_resources: None,
            };
            let evaluation = evaluate_request_with_budget(
                &generator,
                &plan,
                &inputs,
                MemoryCacheState::Cold,
                OffloadPolicy::Resident,
                fixture_budget(20.0),
                gib_to_bytes(9.0),
                0,
                &[],
            )
            .unwrap_or_else(|error| panic!("{provider} {label} resident request: {error}"));

            assert_eq!(
                evaluation.context.selection.strategy,
                MemoryStrategy::Resident,
                "the refused low-rank declaration remains eager/resident"
            );
            assert_eq!(evaluation.context.mode, provider_mode);
            assert_eq!(
                evaluation.context.overlay.as_deref(),
                Some(expected_overlay)
            );
            assert_eq!(
                (registration.safety_check)(&spec, &contract, &evaluation.context),
                gen_core::MemorySafetyDecision::Accept,
                "{provider} {label} must pass the real pinned provider safety seam"
            );
            assert!(
                (behavior.begin_request)(&spec, &contract, &evaluation.context)
                    .unwrap_or_else(|error| { panic!("{provider} {label} begin_request: {error}") })
                    .is_some(),
                "{provider} {label} must reach a real request scope"
            );
            assert_eq!(
                plan.load_shape_declaration_result,
                LoadShapeDeclarationResult::Refused,
                "provider execution must not reauthorize BTR"
            );
        }
    }
}

#[test]
fn krea_turbo_and_edit_routes_use_provider_exact_estimated_authority() {
    // sc-22667: the fixture's block count through the production facts seam.
    let _facts = crate::video_admission::inject_architecture_facts(FULL_LADDER_FIXTURE_FACTS);
    for (provider, model_id, mode, references) in [
        ("krea_2_turbo", "krea_2_turbo", "image_generation", 0),
        ("krea_2_edit", "krea_2_raw", "edit_image", 2),
        ("krea_2_turbo_edit", "krea_2_turbo", "edit_image", 1),
    ] {
        let mut generator = full_ladder_generator();
        generator.descriptor.id = provider;
        generator.contract.as_mut().unwrap().provider_id = provider.to_owned();
        let mut plan = fixture_plan();
        plan.engine_id = provider;
        plan.model_id = model_id.to_owned();
        plan.calibration = MlxCalibrationConfig::Absent;
        let mut inputs = fixture_inputs(1024, 1024);
        inputs.mode = mode.to_owned();
        inputs.overlay = Some("adapters:1".to_owned());
        inputs.adapter_count = 1;
        inputs.has_reference = references > 0;
        inputs.reference_count = references;
        inputs.use_pid = false;
        let evaluation = evaluate_request_with_budget(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Sequential,
            budget_admitting_only_the_deepest_estimate_rung(),
            full_ladder_baseline_bytes(),
            0,
            &[],
        )
        .expect("the provider-exact estimate ladder must admit the deep rung");
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
            "{provider}"
        );
        assert_eq!(
            evaluation.context.optimization_authority,
            MemoryOptimizationAuthority::Estimated,
            "{provider} must not inherit a catalog sibling's measured authority"
        );
        assert_eq!(evaluation.context.overlay, None, "{provider}");
        assert_eq!(evaluation.context.geometry.reference_count, references);
        assert!(!evaluation.context.use_pid);
    }
}

#[test]
fn krea_prepacked_tier_is_explicit_and_not_load_time_quantization() {
    let mut plan = fixture_plan();
    plan.engine_id = "krea_2_edit";
    plan.model_id = "krea_2_raw".to_owned();
    plan.tier.quant = None;
    let q4 = plan
        .clone()
        .with_resolved_artifact_tier(Some("q4"))
        .unwrap();
    let q8 = plan.with_resolved_artifact_tier(Some("q8")).unwrap();
    assert_eq!(q4.tier.quant, Some(gen_core::Quant::Q4));
    assert_eq!(q8.tier.quant, Some(gen_core::Quant::Q8));
    assert!(q8.with_resolved_artifact_tier(Some("unknown")).is_err());
}

#[test]
fn krea_eager_provider_refusal_runs_only_when_the_resident_path_fits() {
    let mut generator = full_ladder_generator();
    generator.descriptor.id = "krea_2_turbo";
    let contract = generator.contract.as_mut().unwrap();
    contract.provider_id = "krea_2_turbo".to_owned();
    for capability in &mut contract.strategies {
        if capability.strategy != MemoryStrategy::Resident {
            capability.support = gen_core::MemoryStrategySupport::Missing;
        }
    }
    let mut plan = fixture_plan();
    plan.engine_id = "krea_2_turbo";
    plan.model_id = "krea_2_turbo".to_owned();
    plan.calibration = MlxCalibrationConfig::Absent;
    // This fixture Missings every rung but Resident, so the ONLY candidate is the 12 GiB
    // resident baseline, which declares no weights/activation split and is graded on the
    // policy's undeclared-floor (recapture) arm. The pair below brackets THAT admitted value
    // from either side (+/-10%), computed from the policy so any allowance re-derivation
    // moves the bracket with it.
    let resident_admitted_gb = crate::memory_strategy::peak_bytes_to_gb(
        crate::memory_strategy::floor_admitted_peak_bytes(
            gen_core::MemoryBackend::Mlx,
            gib_to_bytes(12.0),
            None,
        ),
    );
    let admitted = evaluate_request_with_budget(
        &generator,
        &plan,
        &fixture_inputs(1024, 1024),
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        fixture_budget(resident_admitted_gb * 1.10),
        gib_to_bytes(12.0),
        0,
        &[],
    )
    .expect("the safe eager fallback remains executable when the resident path fits");
    assert_eq!(
        admitted.context.selection.strategy,
        MemoryStrategy::Resident
    );
    let error = evaluate_request_with_budget(
        &generator,
        &plan,
        &fixture_inputs(1024, 1024),
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        fixture_budget(resident_admitted_gb * 0.90),
        gib_to_bytes(12.0),
        0,
        &[],
    )
    .expect_err("a safe eager fallback cannot exceed the live host budget")
    .to_string();
    assert!(error.contains("needs"), "{error}");
    assert!(error.contains("safely available"), "{error}");
}

#[test]
fn uncalibrated_chroma_routes_authorize_exact_quality_backed_estimates() {
    // sc-22667: the fixture's block count through the production facts seam.
    let _facts = crate::video_admission::inject_architecture_facts(FULL_LADDER_FIXTURE_FACTS);
    for route in ["chroma1_hd", "chroma1_flash"] {
        let mut generator = full_ladder_generator();
        generator.descriptor.id = route;
        let contract = generator.contract.as_mut().unwrap();
        contract.provider_id = route.to_owned();
        contract.calibration = None;
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let policy = MemoryDecodeGeometryPolicy {
            quality_abi: gen_core::MEMORY_DECODE_QUALITY_ABI,
            family: "chroma".to_owned(),
            resolved_route: route.to_owned(),
            backend: gen_core::MemoryBackend::Mlx,
            tier: fixture_plan().tier,
            load_shape: contract.load_shape,
            artifact: MemoryDecodeArtifactIdentity {
                repository: format!("SceneWorks/{route}-mlx"),
                revision: "d".repeat(40),
                variant: "q4".to_owned(),
                fingerprint: format!("SceneWorks/{route}-mlx@{}:q4", "d".repeat(40)),
            },
            implementation_fingerprint: DECODE_QUALITY_TEST_STAMP.to_owned(),
            mode: MemoryMode::TextToImage,
            overlay: None,
            geometry,
            use_pid: false,
            tile_edge: 512,
            overlap: 128,
            metric: "max_abs_rgb_u8".to_owned(),
            maximum_error: 48,
            fixtures: [(7, 'a', 'b', 'c'), (99, 'd', 'e', 'f')]
                .into_iter()
                .map(|(seed, latent, dense, tiled)| MemoryDecodeQualityFixture {
                    seed,
                    production_latent_provenance: format!("{route}@revision seed={seed}"),
                    production_latent_sha256: latent.to_string().repeat(64),
                    dense_output_sha256: dense.to_string().repeat(64),
                    tiled_output_sha256: tiled.to_string().repeat(64),
                    observed_error: 40,
                })
                .collect(),
            production_evidence_sha256: String::new(),
            disposition: MemoryDecodeQualityDisposition::Admitted,
        }
        .seal();
        contract
            .adopt_decode_geometry_policies("chroma", vec![policy])
            .unwrap();
        assert!(contract.conformance_errors().is_empty());

        let mut plan = fixture_plan();
        plan.engine_id = route;
        plan.model_id = route.to_owned();
        plan.calibration = MlxCalibrationConfig::Absent;
        let evaluation = evaluate_request_with_budget(
            &generator,
            &plan,
            &fixture_inputs(1024, 1024),
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            budget_admitting_only_the_deepest_estimate_rung(),
            full_ladder_baseline_bytes(),
            0,
            &[],
        )
        .expect("uncalibrated Chroma must retain its estimate-backed optimized route");
        assert!(evaluation.context.selection.strategy.is_optimized());
        assert_eq!(
            evaluation.context.optimization_authority,
            MemoryOptimizationAuthority::Estimated
        );
        assert!(evaluation.memory.tile_vae_decode);
        assert!(matches!(
            gen_core::default_memory_strategy_safety_check(
                generator.contract.as_ref().unwrap(),
                &evaluation.context,
            ),
            gen_core::MemorySafetyDecision::Accept
        ));
    }
}

#[cfg(target_os = "macos")]
#[test]
fn shipped_plain_krea_without_a_binding_preserves_the_request_on_estimate_admission() {
    fn contract(policy: OffloadPolicy) -> MemoryProviderContract {
        let mut contract = crate::inference_runtime::media()
            .memory_contract_surfaces()
            .expect("the shipped weights-free contract inventory")
            .into_iter()
            .find(|surface| {
                surface.contract.provider_id == "krea_2_turbo"
                    && surface.selector.tier == gen_core::MemoryContractSurfaceTier::Q4
                    && surface.selector.offload_policy == policy
                    && surface.selector.load_shape == gen_core::LoadShape::DeferredMaterialization
            })
            .expect("the shipped plain Krea q4/deferred contract surface")
            .contract;
        // Preserve the shipped contract, composition, parameters and load shape while making
        // the pure selector arithmetic legible: a 40 GiB base consists of a 5 GiB conditioner
        // and 35 GiB DiT. Under the re-derived activation allowance (epic 22505 feature-end
        // fix round) only rung 4's windowed floor — which removes the 35 GiB transformer from
        // residency — undercuts the resident baseline, so the constrained arm below exercises
        // exactly the walk-down this test pins.
        contract.asset_facts.base_bytes = gib_to_bytes(40.0);
        contract.asset_facts.conditioning_bytes = gib_to_bytes(5.0);
        contract.asset_facts.transformer_bytes = gib_to_bytes(35.0);
        contract.asset_facts.decoder_bytes = 0;
        // These are synthetic weights; provide their synthetic exact stream inventory too.
        contract.phase_facts.as_mut().unwrap().transformer_stream =
            Some(gen_core::StreamedWeightFacts {
                resident_bytes: 0,
                stacks: vec![vec![gib_to_bytes(0.5); 70]],
            });
        contract
    }

    fn generator(contract: MemoryProviderContract) -> RequestGenerator {
        RequestGenerator {
            descriptor: gen_core::ModelDescriptor {
                id: "krea_2_turbo",
                family: "krea",
                backend: "mlx",
                modality: gen_core::Modality::Image,
                capabilities: gen_core::Capabilities::default(),
                encoder_contract: None,
                denoiser_output_latent_space: None,
                required_components: &[],
                control_kinds: None,
            },
            contract: Some(contract),
        }
    }

    let plan = MlxRequestPlan {
        engine_id: "krea_2_turbo",
        model_id: "krea_2_turbo".to_owned(),
        tier: MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Q4),
            component_precision_floors: &[],
        },
        asset_bytes: gib_to_bytes(40.0),
        folded_control_bytes: 0,
        folded_adapter_bytes: 0,
        activation_headroom_bytes: gib_to_bytes(6.0),
        fixed_reserve_bytes: gib_to_bytes(2.0),
        calibration: MlxCalibrationConfig::Absent,
        load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
    };
    let inputs = fixture_inputs(1024, 1024);

    let resident = evaluate_request_with_budget(
        &generator(contract(OffloadPolicy::Resident)),
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        fixture_budget(55.0),
        gib_to_bytes(46.0),
        0,
        &[],
    )
    .expect("a roomy host must keep the exact plain Krea request on resident admission");
    assert_eq!(
        resident.context.selection.strategy,
        MemoryStrategy::Resident
    );
    assert_eq!(
        resident.context.load_shape,
        gen_core::LoadShape::DeferredMaterialization
    );

    let constrained = evaluate_request_with_budget(
        &generator(contract(OffloadPolicy::Sequential)),
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Sequential,
        // Between rung 4's admitted floor (5 + 6 raw, ~29.6 admitted) and the resident
        // baseline's undeclared ceiling (~51.8) — the staged floor sits far above both.
        fixture_budget(40.0),
        gib_to_bytes(46.0),
        0,
        &[],
    )
    .expect("the unmeasured shipped Krea contract must reach the deep estimate rung");
    assert_eq!(
        constrained.context.selection.strategy,
        MemoryStrategy::BoundedTransformerResidency
    );
    assert_eq!(
        constrained.context.selection.parameters.decode_tile_edge,
        Some(512)
    );
    assert_eq!(
        constrained.context.selection.parameters.decode_overlap,
        Some(64)
    );
    assert_eq!(
        constrained
            .context
            .selection
            .parameters
            .attention_chunk_size,
        Some(67_108_864)
    );
    assert_eq!(
        constrained
            .context
            .selection
            .parameters
            .transformer_window_size,
        Some(1)
    );
    assert!(constrained.memory.stage_residency);
    assert!(constrained.memory.tile_vae_decode);
    assert!(constrained.memory.chunk_attention);
    assert!(constrained.memory.stream_transformer_blocks);
    for evaluation in [&resident, &constrained] {
        assert_eq!(evaluation.context.mode, MemoryMode::TextToImage);
        assert!(!evaluation.context.has_reference);
        assert_eq!(evaluation.context.geometry.reference_count, 0);
        assert!(evaluation.context.overlay.is_none());
        assert!(!evaluation.context.use_pid);
        assert_eq!(
            evaluation.context.evidence_revision,
            REQUEST_EVIDENCE_REVISION
        );
        assert_eq!(evaluation.process_limit_bytes, None);
    }
}

/// The shipped plain-SDXL registry contract driven through the real request seam, so a
/// warm-policy grant is graded by production code rather than by a hand-built contract.
#[cfg(target_os = "macos")]
struct SdxlSelectorFixture {
    root: tempfile::TempDir,
}

#[cfg(target_os = "macos")]
impl SdxlSelectorFixture {
    fn new() -> Self {
        let root = tempfile::tempdir().expect("sdxl fixture tempdir");
        for component in ["text_encoder", "text_encoder_2", "unet", "vae"] {
            let directory = root.path().join(component);
            std::fs::create_dir_all(&directory).unwrap();
            let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.extend_from_slice(&0_f32.to_le_bytes());
            std::fs::write(directory.join("model.safetensors"), bytes).unwrap();
        }
        std::fs::write(
            root.path().join("unet").join("config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        Self { root }
    }

    fn spec(&self, policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(self.root.path().to_owned()))
            .with_quant(gen_core::Quant::Q4)
            .with_offload_policy(policy)
            .with_load_shape(gen_core::LoadShape::DeferredMaterialization)
    }

    fn generator(&self, policy: OffloadPolicy) -> RequestGenerator {
        let mut contract = crate::inference_runtime::media()
            .memory_strategy_contract("sdxl", &self.spec(policy))
            .unwrap()
            .expect("the shipped plain SDXL registry contract");
        // 40/20/20 rather than the historical 6/1/5 (epic 22505 feature-end fix round):
        // under the re-derived activation allowance a staged floor only undercuts the
        // resident baseline when the co-residency drop (max-of vs sum-of) outweighs the
        // allowance surplus, which needs weights that dwarf the 6 GiB headroom.
        contract.asset_facts.base_bytes = gib_to_bytes(40.0);
        contract.asset_facts.conditioning_bytes = gib_to_bytes(20.0);
        contract.asset_facts.transformer_bytes = gib_to_bytes(20.0);
        contract.asset_facts.decoder_bytes = 0;
        RequestGenerator {
            descriptor: gen_core::ModelDescriptor {
                id: "sdxl",
                family: "sdxl",
                backend: "mlx",
                modality: gen_core::Modality::Image,
                capabilities: gen_core::Capabilities::default(),
                encoder_contract: None,
                denoiser_output_latent_space: None,
                required_components: &[],
                control_kinds: None,
            },
            contract: Some(contract),
        }
    }

    /// A proposal in the exact state the cache seam produces for a granted switch: loaded
    /// resident/eager, request asking for staged/deferred, snapshot directory, contract implements
    /// staging.
    fn loaded_policy(&self) -> crate::generator_cache::ExecutionPolicy {
        crate::generator_cache::ExecutionPolicy {
            offload_policy: OffloadPolicy::Resident,
            load_shape: gen_core::LoadShape::EagerMaterialization,
            load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
        }
    }

    fn requested_policy(&self) -> crate::generator_cache::ExecutionPolicy {
        crate::generator_cache::ExecutionPolicy {
            offload_policy: OffloadPolicy::Sequential,
            load_shape: gen_core::LoadShape::DeferredMaterialization,
            load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
        }
    }

    fn granted_proposal(&self) -> crate::execution_planner::WarmPolicyProposal {
        let loaded = self.loaded_policy();
        let requested = self.requested_policy();
        let decision = crate::execution_planner::decide_warm_policy(
            loaded,
            requested,
            crate::execution_planner::SourceReopenability::Reopenable,
            crate::execution_planner::StagingAttestation::Implemented,
        );
        assert!(
            decision.grants_requested_policy(),
            "the fixture must actually be a grant, got {decision:?}"
        );
        crate::execution_planner::WarmPolicyProposal::new(
            "sdxl",
            decision,
            loaded,
            requested,
            crate::execution_planner::SourceReopenability::Reopenable,
            crate::execution_planner::StagingAttestation::Implemented,
        )
    }

    fn evaluate(
        &self,
        load_policy: OffloadPolicy,
        warm_policy: crate::execution_planner::WarmPolicyProposal,
        budget: MemoryBudget,
    ) -> WorkerResult<MlxRequestEvaluation> {
        let plan = MlxRequestPlan {
            engine_id: "sdxl",
            model_id: "sdxl".to_owned(),
            tier: MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: Some(gen_core::Quant::Q4),
                component_precision_floors: &[],
            },
            asset_bytes: gib_to_bytes(40.0),
            folded_control_bytes: 0,
            folded_adapter_bytes: 0,
            activation_headroom_bytes: gib_to_bytes(6.0),
            fixed_reserve_bytes: gib_to_bytes(2.0),
            calibration: MlxCalibrationConfig::Absent,
            load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
        };
        evaluate_request_with_budget_and_warm_policy(
            &self.generator(load_policy),
            &plan,
            &fixture_inputs(1024, 1024),
            MemoryCacheState::Warm,
            load_policy,
            warm_policy,
            budget,
            gib_to_bytes(46.0),
            0,
            &[],
            None,
        )
    }
}

/// Capture this thread's tracing output so a settled decision can be graded by the event it emits.
#[cfg(target_os = "macos")]
struct EventCapture {
    sink: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    _guard: tracing::subscriber::DefaultGuard,
}

#[cfg(target_os = "macos")]
impl EventCapture {
    fn install() -> Self {
        #[derive(Clone)]
        struct Sink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Sink {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let writer = Sink(std::sync::Arc::clone(&sink));
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .with_target(false)
            .without_time()
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        Self {
            sink,
            _guard: guard,
        }
    }

    fn text(&self) -> String {
        String::from_utf8(self.sink.lock().unwrap().clone()).expect("utf-8 tracing")
    }
}

/// sc-18317 MAJOR fix: a granted warm-policy switch must CHANGE WHAT RUNS.
///
/// The first version of this feature threaded a vetted "effective policy" to the run callback,
/// where all eight production consumers bound it as `_requested_policy` and dropped it. Per-request
/// staging is chosen HERE, by `select_strategy` over the candidate set, so the grant was inert and
/// a `rematerialized` event could fire while the request ran fully resident.
///
/// This drives the real seam on a ROOMY budget, where the baseline selector prefers `Resident`, and
/// asserts the grant floors the candidate set into a staged selection instead. It is the mutation
/// sentinel for the threading: delete the floor in `evaluate_request_with_budget_using_bundle` and
/// the staged assertions below go red.
#[cfg(target_os = "macos")]
#[test]
fn a_granted_warm_switch_moves_the_selection_and_stays_inside_the_admitted_peak() {
    let fixture = SdxlSelectorFixture::new();
    // Above the baseline's undeclared admitted ceiling (46 * (1 + recapture) ~ 51.8) so the
    // roomy arm genuinely prefers resident; the staged floor (20 + 6 raw) admits below it.
    let roomy = fixture_budget(55.0);

    let baseline = fixture
        .evaluate(
            OffloadPolicy::Resident,
            crate::execution_planner::WarmPolicyProposal::inert("sdxl"),
            roomy,
        )
        .expect("a roomy host must admit resident SDXL");
    assert_eq!(
        baseline.context.selection.strategy,
        MemoryStrategy::Resident,
        "the baseline must prefer resident, or this test proves nothing about the floor"
    );
    assert!(
        !baseline.memory.stage_residency,
        "a resident baseline must not stage"
    );

    let granted = fixture
        .evaluate(OffloadPolicy::Resident, fixture.granted_proposal(), roomy)
        .expect("a granted switch must still admit the request");
    assert_ne!(
        granted.context.selection.strategy,
        MemoryStrategy::Resident,
        "the grant must move the selection off resident, not merely be reported"
    );
    assert!(
        granted.memory.stage_residency,
        "the granted request must reach the provider asking for staged residency — that field is \
             what `gen_core::Residency::run_request_scoped` reads, and it is the whole point"
    );

    // Memory safety, asserted rather than assumed: the granted selection's predicted peak never
    // exceeds the baseline's, so it stays inside the envelope admission proved against
    // `loaded_policy`. Both runs were admitted under the SAME loaded policy and the same budget.
    assert!(
        granted.context.predicted_peak_bytes <= baseline.context.predicted_peak_bytes,
        "granted peak {} must not exceed the admitted baseline peak {}",
        granted.context.predicted_peak_bytes,
        baseline.context.predicted_peak_bytes
    );
    assert_eq!(
        granted.context.budget, baseline.context.budget,
        "a grant is an execution intent and must never move an admission input"
    );
}

/// The event may not claim a re-materialization the selector did not perform.
///
/// On a constrained budget the baseline ALREADY stages, so the floor is a no-op and the honest
/// report is `served_as_is` / `selection_already_staged`. Mutation sentinel for the false-fire:
/// settle a granted proposal with `SelectionMovedToStaged` unconditionally and this goes red.
#[cfg(target_os = "macos")]
#[test]
fn a_grant_that_changes_nothing_reports_served_as_is_rather_than_rematerialized() {
    let fixture = SdxlSelectorFixture::new();
    let capture = EventCapture::install();
    let constrained = fixture
        .evaluate(
            OffloadPolicy::Sequential,
            fixture.granted_proposal(),
            // Between the staged floor's admitted ceiling (~44.6) and the resident
            // baseline's (~51.8): the baseline already stages, so the grant is a no-op.
            fixture_budget(48.0),
        )
        .expect("a constrained host must still reach an estimate rung");
    assert!(
        constrained.memory.stage_residency,
        "the constrained baseline already stages"
    );
    let text = capture.text();
    assert!(
        text.contains("generator_cache_warm_policy_decision")
            && text.contains("decision=\"served_as_is\"")
            && text.contains("reason=\"selection_already_staged\""),
        "a no-op grant must report the truth: {text:?}"
    );
    assert!(
        !text.contains("decision=\"rematerialized\""),
        "no re-materialization happened, so none may be claimed: {text:?}"
    );
}

/// MINOR 3 (review cycle 2): a MULTI-ITEM job settles ONE decision and emits ONE event.
///
/// Every image lane calls `evaluate_request` once per seed or pose with the same `Copy` proposal, so
/// before `WarmPolicyOnce` a four-image warm job emitted four identical decision events for one
/// cache access — contradicting the one-settlement contract the seam test asserts. This drives four
/// evaluations through the holder exactly as a lane does.
#[cfg(target_os = "macos")]
#[test]
fn a_multi_item_job_settles_one_decision_and_emits_one_event() {
    let fixture = SdxlSelectorFixture::new();
    let capture = EventCapture::install();
    let mut warm_policy = crate::execution_planner::WarmPolicyOnce::new(fixture.granted_proposal());
    let roomy = fixture_budget(55.0);
    let mut staged_items = 0;
    for _ in 0..4 {
        let evaluation = fixture
            .evaluate(OffloadPolicy::Resident, warm_policy.take(), roomy)
            .expect("every item of the job must be admitted");
        if evaluation.memory.stage_residency {
            staged_items += 1;
        }
    }
    let text = capture.text();
    assert_eq!(
        text.matches("generator_cache_warm_policy_decision").count(),
        1,
        "four items, one warm hit, one decision event: {text:?}"
    );
    assert!(
        text.contains("decision=\"rematerialized\""),
        "the one event must be the granted outcome from the FIRST item: {text:?}"
    );
    // EVERY item is floored, not just the one that reported. The decision is a fact about the
    // resident generator, so it holds for the whole job; rationing the decision instead of only its
    // event would have let items 2..N run at the very peak the switch exists to avoid.
    assert_eq!(
        staged_items, 4,
        "the grant must apply to every item of the job, not only the reporting one"
    );
}

/// The same one-event property, driven the way the POSE-control lane drives it.
///
/// Review cycle 3 caught `generate_krea_imported_control_stream` still passing the bare `Copy`
/// proposal per pose inside `drive_gen_items_scored`, so a multi-pose warm job emitted N duplicate
/// warn events. Flooring was already correct there; only the reporting was wrong. This covers the
/// per-pose shape — an odd item count, and the refusing decision whose event is a warn rather than
/// an info, since duplicated warns were the actual complaint.
#[cfg(target_os = "macos")]
#[test]
fn a_multi_pose_job_emits_one_event_even_when_the_decision_is_a_refusal() {
    let fixture = SdxlSelectorFixture::new();
    let capture = EventCapture::install();
    // A refusal, not a grant: an imported single-file source cannot re-open its components.
    let refused = crate::execution_planner::WarmPolicyProposal::new(
        "sdxl",
        crate::execution_planner::decide_warm_policy(
            fixture.loaded_policy(),
            fixture.requested_policy(),
            crate::execution_planner::SourceReopenability::SingleFileNotReopenable,
            crate::execution_planner::StagingAttestation::Implemented,
        ),
        fixture.loaded_policy(),
        fixture.requested_policy(),
        crate::execution_planner::SourceReopenability::SingleFileNotReopenable,
        crate::execution_planner::StagingAttestation::Implemented,
    );
    let mut warm_policy = crate::execution_planner::WarmPolicyOnce::new(refused);
    for _ in 0..3 {
        fixture
            .evaluate(
                OffloadPolicy::Resident,
                warm_policy.take(),
                fixture_budget(55.0),
            )
            .expect("a refused switch must still serve every pose");
    }
    let text = capture.text();
    assert_eq!(
        text.matches("generator_cache_warm_policy_decision").count(),
        1,
        "three poses, one warm hit, one event: {text:?}"
    );
    assert!(
        text.contains("decision=\"refused_switch\"")
            && text.contains("reason=\"source_not_reopenable\""),
        "the one event must carry the refusal: {text:?}"
    );
}

/// A granted proposal on a route with no request-scoped memory seam must not report a switch.
#[cfg(target_os = "macos")]
#[test]
fn a_declined_grant_reports_the_route_limit_rather_than_a_switch() {
    let fixture = SdxlSelectorFixture::new();
    let capture = EventCapture::install();
    fixture
        .granted_proposal()
        .decline(crate::execution_planner::ServedAsIsReason::RouteHasNoRequestScopedMemory);
    let text = capture.text();
    assert!(
        text.contains("decision=\"served_as_is\"")
            && text.contains("reason=\"route_has_no_request_scoped_memory\""),
        "a declined grant must name the route limit: {text:?}"
    );
    assert!(!text.contains("decision=\"rematerialized\""), "{text:?}");
}

#[cfg(target_os = "macos")]
#[test]
fn shipped_plain_sdxl_without_a_binding_preserves_the_four_rung_estimate_path() {
    fn fixture_spec(root: &std::path::Path, policy: OffloadPolicy) -> LoadSpec {
        for component in ["text_encoder", "text_encoder_2", "unet", "vae"] {
            let directory = root.join(component);
            std::fs::create_dir_all(&directory).unwrap();
            let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.extend_from_slice(&0_f32.to_le_bytes());
            std::fs::write(directory.join("model.safetensors"), bytes).unwrap();
        }
        std::fs::write(
            root.join("unet").join("config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        LoadSpec::new(WeightsSource::Dir(root.to_owned()))
            .with_quant(gen_core::Quant::Q4)
            .with_offload_policy(policy)
            .with_load_shape(gen_core::LoadShape::DeferredMaterialization)
    }

    fn contract(root: &std::path::Path, policy: OffloadPolicy) -> MemoryProviderContract {
        let mut contract = crate::inference_runtime::media()
            .memory_strategy_contract("sdxl", &fixture_spec(root, policy))
            .unwrap()
            .expect("the shipped plain SDXL registry contract");
        // 40/5/35 for the same reason as the shipped Krea fixture above: rung 4's
        // windowed floor must undercut the resident baseline under the re-derived
        // activation allowance for the constrained walk-down arm to exist.
        contract.asset_facts.base_bytes = gib_to_bytes(40.0);
        contract.asset_facts.conditioning_bytes = gib_to_bytes(5.0);
        contract.asset_facts.transformer_bytes = gib_to_bytes(35.0);
        contract.asset_facts.decoder_bytes = 0;
        // These are synthetic weights; provide their synthetic exact stream inventory too.
        contract.phase_facts.as_mut().unwrap().transformer_stream =
            Some(gen_core::StreamedWeightFacts {
                resident_bytes: 0,
                stacks: vec![vec![gib_to_bytes(0.5); 70]],
            });
        contract
    }

    fn generator(contract: MemoryProviderContract) -> RequestGenerator {
        RequestGenerator {
            descriptor: gen_core::ModelDescriptor {
                id: "sdxl",
                family: "sdxl",
                backend: "mlx",
                modality: gen_core::Modality::Image,
                capabilities: gen_core::Capabilities::default(),
                encoder_contract: None,
                denoiser_output_latent_space: None,
                required_components: &[],
                control_kinds: None,
            },
            contract: Some(contract),
        }
    }

    let root = tempfile::tempdir().unwrap();
    let plan = MlxRequestPlan {
        engine_id: "sdxl",
        model_id: "sdxl".to_owned(),
        tier: MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Q4),
            component_precision_floors: &[],
        },
        asset_bytes: gib_to_bytes(40.0),
        folded_control_bytes: 0,
        folded_adapter_bytes: 0,
        activation_headroom_bytes: gib_to_bytes(6.0),
        fixed_reserve_bytes: gib_to_bytes(2.0),
        calibration: MlxCalibrationConfig::Absent,
        load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
    };
    let inputs = fixture_inputs(1024, 1024);

    let resident = evaluate_request_with_budget(
        &generator(contract(root.path(), OffloadPolicy::Resident)),
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        fixture_budget(55.0),
        gib_to_bytes(46.0),
        0,
        &[],
    )
    .expect("a roomy host must retain exact SDXL resident admission");
    assert_eq!(
        resident.context.selection.strategy,
        MemoryStrategy::Resident
    );

    let constrained = evaluate_request_with_budget(
        &generator(contract(root.path(), OffloadPolicy::Sequential)),
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Sequential,
        // Between rung 4's admitted floor and the resident baseline's undeclared ceiling,
        // for the same window arithmetic as the shipped Krea test above.
        fixture_budget(40.0),
        gib_to_bytes(46.0),
        0,
        &[],
    )
    .expect("unmeasured SDXL must reach its deepest implemented estimate rung");
    assert_eq!(
        constrained.context.selection.strategy,
        MemoryStrategy::BoundedTransformerResidency
    );
    assert_eq!(
        constrained
            .context
            .selection
            .parameters
            .transformer_window_size,
        Some(1),
        "the current selector chooses the smallest-memory cadence from the provider domain"
    );
    assert!(constrained.memory.stage_residency);
    assert!(constrained.memory.stream_transformer_blocks);
    assert!(constrained.memory.tile_vae_decode);
    let parameters = &constrained.context.selection.parameters;
    assert!([896, 768, 640, 512, 448, 384, 320, 256].contains(
        &parameters
            .decode_tile_edge
            .expect("rung 4 composes native decode")
    ));
    assert_eq!(parameters.decode_overlap, Some(64));
    assert_eq!(
        constrained.memory.decode_tile_edge,
        parameters.decode_tile_edge
    );
    assert_eq!(constrained.memory.decode_overlap, parameters.decode_overlap);
    contract(root.path(), OffloadPolicy::Sequential)
        .validate_selection(&constrained.context.selection)
        .expect("the estimate must select the provider's explicit v5 composition");
    assert!(!resident.memory.tile_vae_decode);
    assert!(!resident.memory.stage_residency);
    assert!(!resident.memory.stream_transformer_blocks);
    assert!(!constrained.memory.chunk_attention);
    for evaluation in [&resident, &constrained] {
        assert_eq!(evaluation.context.mode, MemoryMode::TextToImage);
        assert!(!evaluation.context.has_reference);
        assert_eq!(evaluation.context.geometry.reference_count, 0);
        assert!(evaluation.context.overlay.is_none());
        assert!(!evaluation.context.use_pid);
        assert_eq!(
            evaluation.context.load_shape,
            gen_core::LoadShape::DeferredMaterialization
        );
        assert_eq!(
            evaluation.context.evidence_revision,
            REQUEST_EVIDENCE_REVISION
        );
        assert_eq!(evaluation.process_limit_bytes, None);
    }
}

/// sc-18094/sc-18096: `ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE` is honored at the
/// synthesis seam. An extrapolation that moves the request peak onto a DIFFERENT phase than
/// the measured cell's is rejected — the rung falls back to the no-measured-basis floor,
/// which the constraint's scope sentence exempts — while a same-binding-phase extrapolation
/// produces the fitted candidate at the area-scaled envelope.
#[test]
fn fitted_estimates_honor_the_measured_binding_phase_constraint() {
    use crate::memory_strategy::CandidateBasis;

    let generator = fixture_generator();
    let contract = generator.contract.as_ref().expect("fixture contract");
    let plan = fixture_plan();
    // Measured cell at 1024²: the CONDITIONING phase binds (16 GiB text-encoder peak against
    // 12 GiB denoise and 5 GiB decode) — the exact shape of the corpus example behind the
    // constraint.
    let basis = MeasuredRungBasis {
        rung: StrategyRung::BoundedDecode,
        parameters: gen_core::MemoryStrategyParameters {
            decode_tile_edge: Some(512),
            decode_overlap: Some(128),
            ..Default::default()
        },
        engaged_composition: contract.engaged_composition(MemoryStrategy::BoundedDecode),
        load_shape: contract.load_shape,
        calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: "fixture-formula-v2".to_owned(),
        geometry: CalibrationGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
        },
        conditioning_peak_bytes: gib_to_bytes(16.0),
        denoise_peak_bytes: gib_to_bytes(12.0),
        decode_peak_bytes: gib_to_bytes(5.0),
        envelope_peak_bytes: gib_to_bytes(20.0),
        record_id: "imc-binding-phase-fixture".to_owned(),
    };

    // 2048²: area scale 4.0 pushes denoise to 48 GiB past the flat 16 GiB conditioning peak —
    // the binding phase flips, the fitted candidate is refused, and the rung falls back to
    // the weights+headroom floor.
    let flipped = synthesize_estimate_ladder(
        contract,
        &plan,
        "text_to_image",
        None,
        MemoryGeometry {
            width: 2048,
            height: 2048,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        false,
        None,
        std::slice::from_ref(&basis),
    );
    let decode = flipped
        .estimates
        .iter()
        .find(|estimate| estimate.selection.strategy == MemoryStrategy::BoundedDecode)
        .expect("the rung must still be estimate-admissible via the floor");
    assert_eq!(
        decode.basis,
        CandidateBasis::EstimateFloor,
        "a binding-phase flip must refuse the FITTED candidate per \
             ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE"
    );

    // 1120² (area scale ~1.196): denoise extrapolates to ~14.4 GiB, still under the
    // conditioning peak — no flip, so the fitted candidate is emitted at the area-scaled
    // envelope with the measured cell's parameters.
    let request_geometry = MemoryGeometry {
        width: 1120,
        height: 1120,
        batch: 1,
        frames: 1,
        reference_count: 0,
    };
    let fitted = synthesize_estimate_ladder(
        contract,
        &plan,
        "text_to_image",
        None,
        request_geometry,
        false,
        None,
        std::slice::from_ref(&basis),
    );
    let decode = fitted
        .estimates
        .iter()
        .find(|estimate| estimate.selection.strategy == MemoryStrategy::BoundedDecode)
        .expect("a same-binding-phase extrapolation must be admissible");
    assert_eq!(decode.basis, CandidateBasis::EstimateFittedCurve);
    assert_eq!(decode.selection.parameters.decode_tile_edge, Some(512));
    assert_eq!(decode.selection.parameters.decode_overlap, Some(128));
    let scale = (1120.0 * 1120.0) / (1024.0 * 1024.0);
    let expected_peak = (gib_to_bytes(20.0) as f64 * scale).ceil() as u64;
    assert_eq!(
        decode.evidence.predicted_peak_bytes, expected_peak,
        "the fitted estimate's raw peak is the area-scaled measured envelope (the estimate \
             margin is applied later, by the selector)"
    );

    let mut smaller_tile_basis = basis.clone();
    smaller_tile_basis.parameters.decode_tile_edge = Some(256);
    smaller_tile_basis.parameters.decode_overlap = Some(64);
    let changed = synthesize_estimate_ladder(
        contract,
        &plan,
        "text_to_image",
        None,
        request_geometry,
        false,
        None,
        &[smaller_tile_basis],
    );
    assert!(
        changed
            .estimates
            .iter()
            .all(|candidate| candidate.basis != CandidateBasis::EstimateFittedCurve),
        "a larger executed tile cannot borrow the smaller tile's peak"
    );
    let mut sdxl = contract.clone();
    sdxl.phase_facts = Some(gen_core::MemoryPhaseFacts {
        architecture: Some(gen_core::ImagePipelineArchitecture::SdxlUnetWithDualClip),
        staged_weights: gen_core::StagedWeightSchedule::TwoStage,
        transformer_stream: None,
        decoder_workspace: None,
        native_quant: None,
    });
    for windows in [None, Some(2)] {
        let longer = synthesize_estimate_ladder_with_conditioning(
            &sdxl,
            &plan,
            "text_to_image",
            None,
            request_geometry,
            false,
            None,
            std::slice::from_ref(&basis),
            windows,
        );
        assert!(
            longer
                .estimates
                .iter()
                .all(|candidate| candidate.basis != CandidateBasis::EstimateFittedCurve),
            "unknown or longer CLIP demand is outside the fitted cell"
        );
    }
}

// -------------------------------------------------------------------------------------
// Image-MLX anchor-derived candidates (epic 22505 feature-end fix round, E2/E7).
// -------------------------------------------------------------------------------------

/// A fixture generator whose contract answers for the REAL `flux2_dev` route, so the packaged
/// image anchor's identity conjuncts can hold.
fn flux2_generator() -> RequestGenerator {
    let mut generator = fixture_generator();
    let contract = generator.contract.as_mut().expect("fixture contract");
    contract.provider_id = "flux2_dev".to_owned();
    contract.load_shape = gen_core::LoadShape::EagerMaterialization;
    generator.descriptor.id = "flux2_dev";
    generator
}

fn flux2_plan() -> MlxRequestPlan {
    MlxRequestPlan {
        engine_id: "flux2_dev",
        model_id: "flux2_dev".to_owned(),
        ..fixture_plan()
    }
}

/// The packaged store, unmodified, as the source of the flux2 anchors these tests price from.
///
/// Until sc-22738 this re-stamped every flux2 anchor at the loader-closure digest the pin
/// DECLARED, because the gate would otherwise price a stale-reading anchor from the floor and
/// the derivation's arithmetic could never be graded on the measured numbers. The re-stamp is
/// gone with the mechanism that needed it: `MemoryAnchor::is_current` and the anchor-currency
/// conjunct on the MLX image path are both deleted, so a stale anchor prices the request
/// exactly as a current one does and the shipped rows can be read as they ship.
fn flux2_live_anchor_store() -> sceneworks_core::memory_anchor::MemoryAnchorStore {
    sceneworks_core::memory_anchor::packaged_memory_anchors()
        .expect("the packaged anchor store")
        .clone()
}

/// [`flux2_live_anchor_store`] with the flux2 anchors' phase peaks lifted INTO the core law's
/// domain for the fixture contract's component bytes. The packaged flux2 rows are outside it
/// (sc-22663 review, D3): their conditioning-phase active peak is 1.26 MB against the eager
/// resident set they claim, so `MemoryAnchor::derive_phase_peaks` refuses them with any
/// non-zero component set rather than clamp a negative residue to zero (core test
/// `the_packaged_mlx_anchors_are_outside_the_laws_domain`). This fixture adds the fixture's
/// resident set to every phase's active AND allocator level — the peaks are synthetic, the
/// identity, currency and regime are the packaged row's — so the ARM's wiring (identity
/// guards, currency, outranking the floor) is exercised on an anchor the law can price.
fn flux2_in_domain_anchor_store() -> sceneworks_core::memory_anchor::MemoryAnchorStore {
    let mut store = flux2_live_anchor_store();
    let lift = crate::video_admission::anchor_component_bytes(
        flux2_generator()
            .contract
            .as_ref()
            .expect("fixture contract")
            .asset_facts,
    )
    .total();
    assert!(lift > 0, "the fixture contract states a resident set");
    for anchor in &mut store.anchors {
        if anchor.model_id != "flux2_dev" {
            continue;
        }
        let peaks = &mut anchor.phase_active_peak_bytes;
        peaks.conditioning += lift;
        peaks.denoise += lift;
        peaks.decode += lift;
        let allocators = anchor
            .phase_allocator_envelope_bytes
            .as_mut()
            .expect("the flux2 rows carry a per-phase allocator decomposition");
        allocators.conditioning += lift;
        allocators.denoise += lift;
        allocators.decode += lift;
        anchor.overall_allocator_envelope_bytes += lift;
    }
    store
}

fn flux2_geometry() -> MemoryGeometry {
    MemoryGeometry {
        width: 1536,
        height: 1536,
        batch: 1,
        frames: 1,
        reference_count: 0,
    }
}

fn flux2_ladder(
    generator: &RequestGenerator,
    plan: &MlxRequestPlan,
    mode_key: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
) -> SynthesizedEstimateLadder {
    synthesize_estimate_ladder(
        generator.contract.as_ref().expect("contract"),
        plan,
        mode_key,
        overlay,
        geometry,
        false,
        None,
        &[],
    )
}

/// The PACKAGED flux2 anchors, verbatim and current, price nothing on this fixture: the core
/// law refuses them (their conditioning counters saw no weights — see
/// `flux2_in_domain_anchor_store`), and the ladder falls to the weights+headroom floor,
/// never to a refusal. The differential control for the in-domain test below.
#[test]
fn the_packaged_flux2_anchors_are_refused_and_the_ladder_keeps_its_floor() {
    use crate::memory_strategy::CandidateBasis;

    let generator = flux2_generator();
    let components = crate::video_admission::anchor_component_bytes(
        generator.contract.as_ref().expect("contract").asset_facts,
    );
    let store = flux2_live_anchor_store();
    let anchor = store
        .image_anchor_for(
            "flux2_dev",
            sceneworks_core::memory_anchor::AnchorBackend::Mlx,
            "q4",
        )
        .expect("the flux2 q4 anchor");
    assert!(
        anchor.phase_active_peak_bytes.conditioning < components.total(),
        "the packaged row's conditioning peak sits below the fixture's resident set"
    );
    assert!(anchor
        .derive_mlx_image_phase_peaks(
            sceneworks_core::memory_anchor::AnchorMlxImageDeriveRequest {
                width: flux2_geometry().width,
                height: flux2_geometry().height,
            },
            components,
        )
        .is_none());
    let ladder = with_injected_image_anchor_store(store, || {
        flux2_ladder(
            &generator,
            &flux2_plan(),
            "text_to_image",
            None,
            flux2_geometry(),
        )
    });
    assert!(
        !ladder.estimates.is_empty(),
        "fail to the floor, never refuse"
    );
    for estimate in &ladder.estimates {
        assert_eq!(
            estimate.basis,
            CandidateBasis::EstimateFloor,
            "{:?}: an out-of-domain anchor must leave the floor in place",
            estimate.selection.strategy
        );
    }
}

/// E2/E7 wired: on a legacy image-MLX route with an anchor the law can price (see
/// `flux2_in_domain_anchor_store`), every implemented optimized rung's estimate is the
/// anchor-derived candidate — at exactly the core law's derived admission peak, on the image
/// lane — and it OUTRANKS the generic weights+headroom floor. sc-22738: the anchor's recorded
/// loader closure is not one of the conjuncts that can send the ladder back to that floor.
#[test]
fn the_image_anchor_prices_the_mlx_ladder_ahead_of_the_floor() {
    use crate::memory_strategy::CandidateBasis;

    let generator = flux2_generator();
    let plan = flux2_plan();
    let geometry = flux2_geometry();
    let store = flux2_in_domain_anchor_store();
    let expected_peak = store
        .image_anchor_for(
            "flux2_dev",
            sceneworks_core::memory_anchor::AnchorBackend::Mlx,
            "q4",
        )
        .expect("the flux2 q4 anchor")
        .derive_mlx_image_phase_peaks(
            sceneworks_core::memory_anchor::AnchorMlxImageDeriveRequest {
                width: geometry.width,
                height: geometry.height,
            },
            crate::video_admission::anchor_component_bytes(
                generator.contract.as_ref().expect("contract").asset_facts,
            ),
        )
        .expect("the anchor prices the request")
        .peak_bytes();

    let ladder = with_injected_image_anchor_store(store, || {
        flux2_ladder(&generator, &plan, "text_to_image", None, geometry)
    });
    assert!(
        !ladder.estimates.is_empty(),
        "the implemented optimized rungs must synthesize candidates"
    );
    for estimate in &ladder.estimates {
        assert_eq!(
            estimate.basis,
            CandidateBasis::EstimateAnchorDerived {
                lane: crate::memory_strategy::AnchorDerivationLane::Image,
            },
            "{:?}: a current anchor must outrank the weights+headroom floor",
            estimate.selection.strategy
        );
        assert_eq!(
            estimate.evidence.predicted_peak_bytes, expected_peak,
            "{:?}: the candidate's peak is the core law's derived admission envelope",
            estimate.selection.strategy
        );
        assert_eq!(
            estimate.unmodeled_activation_bytes, None,
            "an anchor-derived peak does not decompose into weights+headroom"
        );
    }

    // sc-22738: currency is NOT one of the anchor's match conjuncts any more. Rotating the
    // recorded loader-closure digest — which used to demote the whole ladder to the generic
    // weights+headroom floor — now changes nothing: the anchor still prices the request at the
    // same derived peak, and the moved digest is a re-capture signal for the probe tooling.
    // `MemoryAnchor::is_current` and `anchor_currency_matches` are both deleted.
    let mut rotated = flux2_in_domain_anchor_store();
    for anchor in &mut rotated.anchors {
        if anchor.model_id == "flux2_dev" {
            anchor.source.loader_closure_digest = "d".repeat(64);
        }
    }
    let rotated_ladder = with_injected_image_anchor_store(rotated, || {
        flux2_ladder(&generator, &plan, "text_to_image", None, geometry)
    });
    assert!(!rotated_ladder.estimates.is_empty());
    for estimate in &rotated_ladder.estimates {
        assert_eq!(
            estimate.basis,
            CandidateBasis::EstimateAnchorDerived {
                lane: crate::memory_strategy::AnchorDerivationLane::Image,
            },
            "{:?}: a moved loader closure must still price from the anchor",
            estimate.selection.strategy
        );
        assert_eq!(
            estimate.evidence.predicted_peak_bytes, expected_peak,
            "{:?}: and at the very same derived peak",
            estimate.selection.strategy
        );
    }
}

/// The full guard set, one conjunct at a time: every identity/regime/overlay/reference
/// mismatch drops exactly to the floor (never to a refusal), and the unmutated arm above is
/// the control that keeps these from being vacuous.
#[test]
fn the_image_anchor_candidate_carries_the_full_guard_set() {
    use crate::memory_strategy::CandidateBasis;

    type LadderCase = (&'static str, Box<dyn Fn() -> SynthesizedEstimateLadder>);
    let cases: Vec<LadderCase> = vec![
        (
            "a multi-frame request",
            Box::new(|| {
                flux2_ladder(
                    &flux2_generator(),
                    &flux2_plan(),
                    "text_to_image",
                    None,
                    MemoryGeometry {
                        frames: 2,
                        ..flux2_geometry()
                    },
                )
            }),
        ),
        (
            "a batched request",
            Box::new(|| {
                flux2_ladder(
                    &flux2_generator(),
                    &flux2_plan(),
                    "text_to_image",
                    None,
                    MemoryGeometry {
                        batch: 2,
                        ..flux2_geometry()
                    },
                )
            }),
        ),
        (
            "a reference-conditioned request",
            Box::new(|| {
                flux2_ladder(
                    &flux2_generator(),
                    &flux2_plan(),
                    "text_to_image",
                    None,
                    MemoryGeometry {
                        reference_count: 1,
                        ..flux2_geometry()
                    },
                )
            }),
        ),
        (
            "an overlaid request",
            Box::new(|| {
                flux2_ladder(
                    &flux2_generator(),
                    &flux2_plan(),
                    "text_to_image",
                    Some("lora"),
                    flux2_geometry(),
                )
            }),
        ),
        (
            "a foreign mode",
            Box::new(|| {
                flux2_ladder(
                    &flux2_generator(),
                    &flux2_plan(),
                    "edit",
                    None,
                    flux2_geometry(),
                )
            }),
        ),
        (
            "a foreign provider contract",
            Box::new(|| {
                let mut generator = flux2_generator();
                generator.contract.as_mut().expect("contract").provider_id =
                    "someone_else".to_owned();
                flux2_ladder(
                    &generator,
                    &flux2_plan(),
                    "text_to_image",
                    None,
                    flux2_geometry(),
                )
            }),
        ),
        (
            "a foreign route",
            Box::new(|| {
                let plan = MlxRequestPlan {
                    engine_id: "flux2_other",
                    ..flux2_plan()
                };
                flux2_ladder(
                    &flux2_generator(),
                    &plan,
                    "text_to_image",
                    None,
                    flux2_geometry(),
                )
            }),
        ),
        (
            "a mismatched materialization shape",
            Box::new(|| {
                let mut generator = flux2_generator();
                generator.contract.as_mut().expect("contract").load_shape =
                    gen_core::LoadShape::DeferredMaterialization;
                flux2_ladder(
                    &generator,
                    &flux2_plan(),
                    "text_to_image",
                    None,
                    flux2_geometry(),
                )
            }),
        ),
        (
            "an unanchored tier",
            Box::new(|| {
                let plan = MlxRequestPlan {
                    tier: MemoryNumericTier {
                        precision: gen_core::Precision::Bf16,
                        quant: None,
                        component_precision_floors: &[],
                    },
                    ..flux2_plan()
                };
                flux2_ladder(
                    &flux2_generator(),
                    &plan,
                    "text_to_image",
                    None,
                    flux2_geometry(),
                )
            }),
        ),
    ];
    for (label, build) in cases {
        let ladder = with_injected_image_anchor_store(flux2_in_domain_anchor_store(), build);
        assert!(
            !ladder.estimates.is_empty(),
            "{label}: the ladder must fail to the floor, never refuse"
        );
        for estimate in &ladder.estimates {
            assert_ne!(
                estimate.basis,
                CandidateBasis::EstimateAnchorDerived {
                    lane: crate::memory_strategy::AnchorDerivationLane::Image,
                },
                "{label} ({:?}): the anchor must not price a mismatched request",
                estimate.selection.strategy
            );
        }
    }
}

/// The Qwen-Image bf16 component split and architecture facts the core law's own in-domain
/// MLX fixture uses (`a_resident_mlx_anchor_whose_counters_saw_its_weights_prices_the_ladder
/// _in_order`): 16.6 GB of Qwen2.5-VL text encoder, ~254 MB of VAE, the rest DiT; 24 heads of
/// 128 over 60 blocks, patch 2 on the 16-channel x8 VAE, bf16 activations. Restated here
/// because the worker fixture must state the SAME model on both sides of the derivation — the
/// contract's asset facts and the anchor's component bytes are one model's, not two.
const QWEN_IN_DOMAIN_COMPONENTS: sceneworks_core::memory_anchor::ComponentBytes =
    sceneworks_core::memory_anchor::ComponentBytes {
        conditioning: 16_600_000_000,
        transformer: 40_880_000_000,
        decoder: 253_806_592,
    };

const QWEN_IN_DOMAIN_FACTS: sceneworks_core::memory_anchor::ArchitectureFacts =
    sceneworks_core::memory_anchor::ArchitectureFacts {
        attention_heads: Some(24),
        head_dim: Some(128),
        transformer_blocks: Some(60),
        patch_size: Some(2),
        latent_channels: Some(16),
        vae_spatial_scale: Some(8),
        vae_temporal_scale: Some(1),
        activation_dtype_width: Some(2),
    };

/// The flux2 fixture generator with its asset facts restated as [`QWEN_IN_DOMAIN_COMPONENTS`],
/// so `anchor_component_bytes` off this contract is exactly the component set the in-domain
/// anchor below was built against.
fn qwen_in_domain_generator() -> RequestGenerator {
    let mut generator = flux2_generator();
    let facts = &mut generator
        .contract
        .as_mut()
        .expect("fixture contract")
        .asset_facts;
    facts.conditioning_bytes = QWEN_IN_DOMAIN_COMPONENTS.conditioning;
    facts.transformer_bytes = QWEN_IN_DOMAIN_COMPONENTS.transformer;
    facts.decoder_bytes = QWEN_IN_DOMAIN_COMPONENTS.decoder;
    facts.base_bytes = QWEN_IN_DOMAIN_COMPONENTS.total();
    generator
}

/// A store whose flux2 q4 row is the SYNTHETIC in-domain resident MLX anchor the core law's
/// fixture defines — measured at 1024x1024 with every phase peak the whole component set plus
/// a stated activation residue (+0.5 / +6 / +12 GB) and an allocator level a stated envelope
/// above each (+0.1 / +1 / +2 GB). The packaged MLX rows are outside the law's domain by
/// design (their conditioning counters never saw their weights — core test
/// `the_packaged_mlx_anchors_are_outside_the_laws_domain`), and sc-22667 owns re-capturing a
/// real one; the identity, route, provider, mode and currency here are the packaged row's, so
/// only the numbers are synthetic.
fn qwen_in_domain_anchor_store() -> sceneworks_core::memory_anchor::MemoryAnchorStore {
    use sceneworks_core::memory_anchor::{AnchorGeometry, AnchorPhaseBytes};

    let mut store = flux2_live_anchor_store();
    let total = QWEN_IN_DOMAIN_COMPONENTS.total();
    for anchor in &mut store.anchors {
        if anchor.model_id != "flux2_dev" {
            continue;
        }
        anchor.underived_reason = None;
        anchor.geometry = AnchorGeometry {
            width: 1024,
            height: 1024,
            frames: 1,
            fps: None,
        };
        anchor.phase_active_peak_bytes = AnchorPhaseBytes {
            conditioning: total + 500_000_000,
            denoise: total + 6_000_000_000,
            decode: total + 12_000_000_000,
        };
        anchor.phase_allocator_envelope_bytes = Some(AnchorPhaseBytes {
            conditioning: total + 500_000_000 + 100_000_000,
            denoise: total + 6_000_000_000 + 1_000_000_000,
            decode: total + 12_000_000_000 + 2_000_000_000,
        });
        anchor.overall_allocator_envelope_bytes = total + 14_000_000_000;
    }
    store
}

/// The estimate the FLOOR arm composes for one rung: the contract-decomposed weights term plus
/// the derived activation residue, exactly as `synthesize_estimate_ladder`'s third tier does.
/// `None` where the floor would keep `generic_headroom_bytes` instead.
fn qwen_in_domain_floor_estimate(
    generator: &RequestGenerator,
    engaged: &[MemoryStrategy],
    parameters: gen_core::MemoryStrategyParameters,
    facts: sceneworks_core::memory_anchor::ArchitectureFacts,
) -> Option<u64> {
    let contract = generator.contract.as_ref().expect("contract");
    let (residue, _) = mlx_image_anchor_activation_residue(
        contract,
        &flux2_plan(),
        "text_to_image",
        None,
        MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        engaged,
        parameters,
        facts,
    )?;
    Some(
        image_floor_weights_bytes(
            contract,
            engaged,
            parameters.transformer_window_size,
            facts,
            ImageFloorActivationTerm::LawResidue,
        )
        .saturating_add(residue),
    )
}

/// AC 1 (sc-22665, epic 22657 E4) — the MLX estimate floor's activation term is the sc-22663
/// law's per-phase residue for the RUNG's own regime, so the decode tile, the attention chunk
/// and the transformer window reach the estimate where `generic_headroom_bytes` charged the
/// same geometry-blind number to every rung.
///
/// Graded on the in-domain resident MLX anchor (see `qwen_in_domain_anchor_store`) at 1024²,
/// where the request's 4096 image tokens plus 512 prompt tokens make the full bf16 score
/// tensor `24 x 4608² x 2` ≈ 1.02 GB against the 64 Mi-score chunk's 128 MB — the condition
/// AC 1 names for the chunked rung to price below the staged one.
///
/// MUTATION (the story's): feed `ArchitectureFacts::default()` — what the contract states at
/// this pin — and the chunk scales nothing, so rung 3's estimate is rung 2's again. That is
/// the honest state of the lane until sc-22667, and it is asserted rather than hidden.
#[test]
fn the_mlx_estimate_floor_prices_each_rungs_own_regime_through_the_derivation_law() {
    let generator = qwen_in_domain_generator();
    let staged = [MemoryStrategy::Resident, MemoryStrategy::StagedResidency];
    // The ladder's compositions are CUMULATIVE, so the chunked rung tiles decode as well; the
    // chunk's own contribution is isolated against `tiled` below.
    let tiled = [
        MemoryStrategy::Resident,
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
    ];
    let chunked = [
        MemoryStrategy::Resident,
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
    ];
    let windowed = [
        MemoryStrategy::Resident,
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ];
    let bare = gen_core::MemoryStrategyParameters::default();
    let tile_only = gen_core::MemoryStrategyParameters {
        decode_tile_edge: Some(512),
        decode_overlap: Some(128),
        ..Default::default()
    };
    let tile_and_chunk = gen_core::MemoryStrategyParameters {
        attention_chunk_size: Some(64 * 1024 * 1024),
        ..tile_only
    };
    let fully_engaged = gen_core::MemoryStrategyParameters {
        decode_tile_edge: Some(512),
        decode_overlap: Some(128),
        attention_chunk_size: Some(64 * 1024 * 1024),
        transformer_window_size: Some(1),
        ..Default::default()
    };

    let estimate = |engaged: &[MemoryStrategy], parameters, facts| {
        with_injected_image_anchor_store(qwen_in_domain_anchor_store(), || {
            qwen_in_domain_floor_estimate(&generator, engaged, parameters, facts)
        })
        .expect("the in-domain anchor prices this rung")
    };

    let rung_2 = estimate(&staged, bare, QWEN_IN_DOMAIN_FACTS);
    let rung_3 = estimate(&chunked, tile_and_chunk, QWEN_IN_DOMAIN_FACTS);
    let rung_4 = estimate(&windowed, fully_engaged, QWEN_IN_DOMAIN_FACTS);
    assert!(
        rung_3 < rung_2,
        "the chunked rung must price below the unbounded rung: {rung_3} vs {rung_2}"
    );
    assert!(
        rung_4 < rung_2,
        "the windowed rung must price below the staged rung: {rung_4} vs {rung_2}"
    );
    assert!(
        rung_4 < rung_3,
        "the transformer window must price below the chunked rung: {rung_4} vs {rung_3}"
    );

    // The residue is what moved, not only the weights term: the same rungs graded with an
    // IDENTICAL weights term still order the same way, and the CHUNK's own contribution is
    // visible against the same composition tiled but unchunked — at 1024² the denoise phase
    // binds once decode is tiled, so replacing the ~1.02 GB score tensor with the 128 MB
    // chunk budget moves the estimate on its own.
    let residue = |engaged: &[MemoryStrategy], parameters| {
        with_injected_image_anchor_store(qwen_in_domain_anchor_store(), || {
            mlx_image_anchor_activation_residue(
                generator.contract.as_ref().expect("contract"),
                &flux2_plan(),
                "text_to_image",
                None,
                MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
                engaged,
                parameters,
                QWEN_IN_DOMAIN_FACTS,
            )
        })
        .expect("the in-domain anchor prices this rung")
        .0
    };
    assert!(residue(&tiled, tile_only) < residue(&staged, bare));
    assert!(
        residue(&chunked, tile_and_chunk) < residue(&tiled, tile_only),
        "the attention chunk alone must lower the residue"
    );
    // The window is a RESIDENCY bound: it moves the weights term, never the activation one.
    assert_eq!(
        residue(&windowed, fully_engaged),
        residue(&chunked, tile_and_chunk)
    );

    // THE FIXTURE PREMISE, spelled out and pinned (sc-22665 review round). This contract's
    // `base_bytes` IS the law's component set, with no auxiliary resident component, so the
    // floor's weights term is exactly the components the anchor's measured peaks were
    // decomposed against — which is what makes the identity below an identity. A real
    // contract that breaks the premise (`base_bytes` above the component sum, or an
    // auxiliary component the anchor never held) prices the floor ABOVE the anchor's measured
    // allocator level with nothing else in this file able to see it, and these three
    // assertions are what catches that the moment sc-22667 wires a real one.
    let contract = generator.contract.as_ref().expect("contract");
    let unbounded = [MemoryStrategy::Resident];
    let weights = estimate_floor_weights_bytes(contract, &unbounded);
    let store = qwen_in_domain_anchor_store();
    let anchor = store
        .image_anchor_for(
            "flux2_dev",
            sceneworks_core::memory_anchor::AnchorBackend::Mlx,
            "q4",
        )
        .expect("the injected q4 anchor");
    let rung_1 = estimate(&unbounded, bare, QWEN_IN_DOMAIN_FACTS);

    // THE IDENTITY THAT HOLDS, at the anchor's own geometry and its own unbounded resident
    // regime: the floor reproduces the anchor's measured ACTIVE peak in the binding phase
    // (decode) to the byte. It does NOT reproduce the anchor's ALLOCATOR level, and must not
    // — the gap between the two is the retained envelope, which the SELECTOR charges as
    // `FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE` over this activation term
    // (`ladder_margin_policy::admission_allowance`, `EstimateFloor` arm). An estimate that
    // already equalled the allocator level would be an envelope charged twice.
    assert_eq!(
        rung_1, anchor.phase_active_peak_bytes.decode,
        "the unbounded floor is the anchor's measured ACTIVE peak, not its allocator level"
    );
    // The premise restated as its own two assertions, so a future fixture edit that breaks it
    // names the reason rather than only reporting a byte mismatch above.
    assert_eq!(
        contract.asset_facts.base_bytes,
        QWEN_IN_DOMAIN_COMPONENTS.total(),
        "the fixture's weights are the law's component set, exactly"
    );
    assert_eq!(
        weights,
        QWEN_IN_DOMAIN_COMPONENTS.total(),
        "the unbounded resident weights term counts the component set and nothing else"
    );
    assert_eq!(
        anchor.overall_allocator_envelope_bytes - rung_1,
        anchor
            .phase_allocator_envelope_bytes
            .expect("the fixture anchor decomposes its allocator level")
            .decode
            - anchor.phase_active_peak_bytes.decode,
        "exactly the retained envelope separates the floor from the allocator level, so no \
             envelope byte is inside the estimate"
    );
    // And the quantity ADMISSION actually compares against still covers the allocator level:
    // the allowance-charged floor sits at or above the anchor's measured envelope.
    let charged = rung_1 as f64
        + crate::ladder_margin_policy::FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE
            * (rung_1 - weights) as f64;
    assert!(
        charged >= anchor.overall_allocator_envelope_bytes as f64,
        "the allowance-charged floor must cover the anchor's measured allocator level: \
             {charged} vs {}",
        anchor.overall_allocator_envelope_bytes
    );

    // MUTATION — with the facts the contract states at THIS pin neither the chunk nor the tile
    // scales anything, and every rung's residue is the unbounded one again.
    let blind = sceneworks_core::memory_anchor::ArchitectureFacts::default();
    assert_eq!(
        estimate(&chunked, tile_and_chunk, blind),
        estimate(&staged, bare, blind),
        "without architecture facts the chunk and the tile cannot shrink the residue"
    );
    assert_eq!(
        crate::video_admission::architecture_facts_from_contract(
            generator.contract.as_ref().expect("contract")
        ),
        blind,
        "the contract states no architecture facts at this pin (sc-22667 wires them)"
    );
}

/// sc-22667 round-2 review (b): an in-domain anchor whose contract states NO block count
/// still prices rung 4 through the law's residue — and that residue carries neither weights
/// nor the window slice. Before the fix the weights term removed the whole transformer
/// (the pre-epic accounting, whose slice lives in the GENERIC headroom that this rung is no
/// longer composed with), so rung 4's window was charged nowhere and the rung priced a whole
/// transformer below rung 3 with no fact behind the saving. Now the transformer stays
/// resident: rung 4 prices exactly as rung 3, at the law's own erring-large reading, until a
/// block count makes the share knowable — and with one it prices the share.
///
/// MUTATION: `(None, LawResidue) => 0` in `image_floor_weights_bytes` (the pre-fix code)
/// makes the blind rung 4 price `transformer_bytes` below rung 3 and reds the first and third
/// assertions.
#[test]
fn without_a_block_count_a_law_priced_rung_4_keeps_the_transformer_resident() {
    let generator = qwen_in_domain_generator();
    let contract = generator.contract.as_ref().expect("contract");
    let chunked = [
        MemoryStrategy::Resident,
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
    ];
    let windowed = [
        MemoryStrategy::Resident,
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ];
    let tile_and_chunk = gen_core::MemoryStrategyParameters {
        decode_tile_edge: Some(512),
        decode_overlap: Some(128),
        attention_chunk_size: Some(64 * 1024 * 1024),
        ..Default::default()
    };
    let fully_engaged = gen_core::MemoryStrategyParameters {
        transformer_window_size: Some(1),
        ..tile_and_chunk
    };
    // The in-domain facts with the block count withheld: every ratio but the window's is live,
    // so the residue is the same one the block-counted rung 4 is priced with.
    let no_blocks = sceneworks_core::memory_anchor::ArchitectureFacts {
        transformer_blocks: None,
        ..QWEN_IN_DOMAIN_FACTS
    };
    let estimate = |engaged: &[MemoryStrategy], parameters, facts| {
        with_injected_image_anchor_store(qwen_in_domain_anchor_store(), || {
            qwen_in_domain_floor_estimate(&generator, engaged, parameters, facts)
        })
        .expect("the in-domain anchor prices this rung")
    };

    let rung_3 = estimate(&chunked, tile_and_chunk, no_blocks);
    let rung_4_blind = estimate(&windowed, fully_engaged, no_blocks);
    let rung_4_counted = estimate(&windowed, fully_engaged, QWEN_IN_DOMAIN_FACTS);
    assert_eq!(
        rung_4_blind, rung_3,
        "without a block count the law-priced rung 4 keeps the whole transformer resident and \
             prices as rung 3: {rung_4_blind} vs {rung_3}"
    );
    assert!(
        rung_4_counted < rung_4_blind,
        "with the block count rung 4 prices the law's share below the blind reading: \
             {rung_4_counted} vs {rung_4_blind}"
    );
    // The weights term alone, against the pre-fix reading: the whole transformer is what the
    // fix keeps, and the generic-headroom regime is untouched by it.
    let weights = |facts, term| {
        image_floor_weights_bytes(
            contract,
            &windowed,
            fully_engaged.transformer_window_size,
            facts,
            term,
        )
    };
    assert_eq!(
        weights(no_blocks, ImageFloorActivationTerm::LawResidue),
        estimate_floor_weights_bytes(contract, &chunked),
        "the law-residue regime keeps the whole transformer: rung 4's weights term is rung \
             3's (staged: max(conditioning, transformer + decoder))"
    );
    assert!(
        weights(no_blocks, ImageFloorActivationTerm::LawResidue)
            > weights(no_blocks, ImageFloorActivationTerm::GenericHeadroom),
        "…and above the generic-headroom regime's, which hands the transformer to its \
             headroom term"
    );
    assert_eq!(
        weights(no_blocks, ImageFloorActivationTerm::GenericHeadroom),
        estimate_floor_weights_bytes(contract, &windowed),
        "the generic-headroom regime is still the pre-epic floor"
    );
    assert_eq!(
        weights(QWEN_IN_DOMAIN_FACTS, ImageFloorActivationTerm::LawResidue),
        weights(
            QWEN_IN_DOMAIN_FACTS,
            ImageFloorActivationTerm::GenericHeadroom
        ),
        "with a block count the activation term no longer matters: the share is the law's"
    );
}

#[test]
fn unknown_mlx_topology_does_not_borrow_chunked_or_tiled_workspace_savings() {
    use crate::memory_strategy::CandidateBasis;

    let generator = qwen_in_domain_generator();
    let contract = generator.contract.as_ref().expect("contract");
    let plan = flux2_plan();
    let geometry = MemoryGeometry {
        width: 1024,
        height: 1024,
        batch: 1,
        frames: 1,
        reference_count: 0,
    };
    let mut store = qwen_in_domain_anchor_store();
    for anchor in &mut store.anchors {
        if anchor.model_id == "flux2_dev" {
            anchor.measured_regime.attention_chunked = true;
        }
    }
    let ladder = with_injected_image_anchor_store(store.clone(), || {
        synthesize_estimate_ladder(
            contract,
            &plan,
            "text_to_image",
            None,
            geometry,
            false,
            None,
            &[],
        )
    });
    assert!(!ladder.estimates.is_empty());
    // A chunk-measured anchor cannot justify unbounded resident workspace. Unknown MLX
    // topology therefore keeps generic workspace even on tiled/chunked candidates.
    for estimate in &ladder.estimates {
        assert_eq!(estimate.basis, CandidateBasis::EstimateFloor);
        let engaged = contract.engaged_composition_for_selection(&estimate.selection);
        let weights = mlx_fallback_weights_bytes(contract, &engaged, estimate.selection.parameters);
        assert_eq!(
            estimate.evidence.predicted_peak_bytes,
            weights + plan.generic_headroom_bytes(geometry)
        );
        assert_eq!(
            estimate.unmodeled_activation_bytes,
            Some(
                plan.generic_headroom_bytes(geometry)
                    .saturating_sub(plan.fixed_reserve_bytes.min(plan.activation_headroom_bytes))
            )
        );
    }
}

#[test]
fn the_request_regimes_transformer_window_reaches_the_laws_measured_regime_guard() {
    let generator = qwen_in_domain_generator();
    let contract = generator.contract.as_ref().expect("contract");
    let plan = flux2_plan();
    let geometry = MemoryGeometry {
        width: 1024,
        height: 1024,
        batch: 1,
        frames: 1,
        reference_count: 0,
    };
    let mut store = qwen_in_domain_anchor_store();
    for anchor in &mut store.anchors {
        if anchor.model_id == "flux2_dev" {
            anchor.measured_regime.transformer_windowed = true;
        }
    }
    let residue = |engaged: &[MemoryStrategy], parameters| {
        with_injected_image_anchor_store(store.clone(), || {
            mlx_image_anchor_activation_residue(
                contract,
                &plan,
                "text_to_image",
                None,
                geometry,
                engaged,
                parameters,
                QWEN_IN_DOMAIN_FACTS,
            )
        })
    };
    let unwindowed = [MemoryStrategy::Resident, MemoryStrategy::StagedResidency];
    let windowed = [
        MemoryStrategy::Resident,
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedTransformerResidency,
    ];
    let bare = gen_core::MemoryStrategyParameters::default();
    let with_window = gen_core::MemoryStrategyParameters {
        transformer_window_size: Some(1),
        ..Default::default()
    };

    assert!(
        residue(&windowed, with_window).is_some(),
        "the windowed rung's declared window must reach the regime and satisfy the law's \
             measured-regime guard"
    );
    assert!(
        residue(&unwindowed, bare).is_none(),
        "a request that does not window cannot be priced from a windowed anchor"
    );
    // Control on the other half of the arm: engaging the STRATEGY without a declared window
    // size states no window either, so it is the parameter reaching the regime that matters,
    // not the strategy being present in the composition.
    assert!(
        residue(&windowed, bare).is_none(),
        "an undeclared window size states no window"
    );
}

/// sc-17153 — `synthesize_estimate_ladder` emits NO candidate for a non-implemented rung,
/// asserted at the synthesis seam itself.
///
/// The sc-17153 pre-dispatch survey flagged this property as unverified. What this pins is
/// the seam's BEHAVIOR, per non-implemented variant, with a control that proves each mutation
/// is the sole reason the candidate disappears:
///
///  * control: the fixture contract's implemented optimized rungs each synthesize a candidate
///    (floor basis — no measured bases are supplied), and its `Missing` rung 4 synthesizes
///    none;
///  * flipping one implemented rung to `Missing` removes exactly that rung's candidate;
///  * flipping it to `StructurallyNotApplicable` removes it identically — positive-match
///    semantics, not a denylist a new support variant could slip past.
///
/// ⚠️ What this test does NOT isolate: WHICH layer refuses. The synthesis loop's explicit
/// `matches!(.., Some(MemoryStrategySupport::Implemented))` filter is redundant
/// defense-in-depth with `contract.validate_selection`, which also refuses a selection on a
/// non-implemented rung — deleting the filter outright leaves this test green because the
/// deeper check catches the same mutation (measured during sc-17153's review). The same
/// caveat applies to the sc-18096 `Missing`-mutation arm above, which reaches the seam only
/// through `evaluate`'s full reject path. Do not read either test as the guard that keeps
/// `validate_selection`'s refusal removable, or vice versa: the pinned property is the
/// behavior, and BOTH layers currently enforce it.
#[test]
fn synthesize_estimate_ladder_admits_only_implemented_rungs() {
    let generator = fixture_generator();
    let contract = generator.contract.as_ref().expect("fixture contract");
    let plan = fixture_plan();
    let geometry = MemoryGeometry {
        width: 1024,
        height: 1024,
        batch: 1,
        frames: 1,
        reference_count: 0,
    };
    let strategies = |contract: &MemoryProviderContract| -> Vec<MemoryStrategy> {
        synthesize_estimate_ladder(
            contract,
            &plan,
            "text_to_image",
            None,
            geometry,
            false,
            None,
            &[],
        )
        .estimates
        .iter()
        .map(|estimate| estimate.selection.strategy)
        .collect()
    };

    // Control: every implemented optimized rung gets a floor candidate; the resident baseline
    // is deliberately absent (it exists on every legacy route already) and the fixture's
    // Missing rung 4 is never synthesized.
    let control = strategies(contract);
    assert_eq!(
        control,
        vec![
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedAttention,
        ],
        "the fixture's implemented rungs, and only those, are estimate-admissible"
    );

    for support in [
        gen_core::MemoryStrategySupport::Missing,
        gen_core::MemoryStrategySupport::StructurallyNotApplicable {
            reason: "sc-17153 mutation: the architecture lacks what the rung optimizes".to_owned(),
        },
    ] {
        let mut mutated = contract.clone();
        mutated
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
            .expect("bounded decode capability")
            .support = support.clone();
        let survivors = strategies(&mutated);
        assert!(
            !survivors.contains(&MemoryStrategy::BoundedDecode),
            "a rung declared {support:?} must synthesize no candidate"
        );
        assert_eq!(
            survivors,
            vec![
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedAttention,
            ],
            "the mutation removes exactly the de-implemented rung — the other implemented \
                 rungs keep their floor candidates, so the control pair isolates the filter"
        );
    }
}

#[test]
fn geometry_quality_policy_keys_estimated_decode_without_disabling_legacy_fallbacks() {
    use gen_core::{
        MemoryDecodeGeometryPolicy, MemoryDecodeQualityDisposition, MemoryDecodeQualityFixture,
        MEMORY_DECODE_QUALITY_ABI,
    };

    let mut generator = fixture_generator();
    let contract = generator.contract.as_mut().expect("fixture contract");
    let admitted_geometry = MemoryGeometry {
        width: 1280,
        height: 1280,
        batch: 1,
        frames: 1,
        reference_count: 0,
    };
    let quality = MemoryDecodeGeometryPolicy {
        quality_abi: MEMORY_DECODE_QUALITY_ABI,
        family: "test".to_owned(),
        resolved_route: contract.provider_id.clone(),
        backend: gen_core::MemoryBackend::Mlx,
        tier: fixture_plan().tier,
        load_shape: contract.load_shape,
        artifact: MemoryDecodeArtifactIdentity {
            repository: "SceneWorks/fixture".to_owned(),
            revision: "c".repeat(40),
            variant: "packed-q4".to_owned(),
            fingerprint: format!("SceneWorks/fixture@{}:packed-q4", "c".repeat(40)),
        },
        implementation_fingerprint: DECODE_QUALITY_TEST_STAMP.to_owned(),
        mode: MemoryMode::TextToImage,
        overlay: None,
        geometry: admitted_geometry,
        use_pid: false,
        tile_edge: 768,
        overlap: 192,
        metric: "max_abs_rgb_u8".to_owned(),
        maximum_error: 48,
        fixtures: [(7, 'a', 'b', 'c'), (99, 'd', 'e', 'f')]
            .into_iter()
            .map(|(seed, latent, dense, tiled)| MemoryDecodeQualityFixture {
                seed,
                production_latent_provenance: format!("fixture@revision seed={seed}"),
                production_latent_sha256: latent.to_string().repeat(64),
                dense_output_sha256: dense.to_string().repeat(64),
                tiled_output_sha256: tiled.to_string().repeat(64),
                observed_error: 40,
            })
            .collect(),
        production_evidence_sha256: String::new(),
        disposition: MemoryDecodeQualityDisposition::Admitted,
    }
    .seal();
    let decode = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
        .expect("bounded decode");
    decode.parameters.decode_tile_edges = vec![768];
    decode.parameters.decode_overlaps = vec![192];
    decode.parameters.decode_geometry_policies = vec![quality];
    assert!(
        contract.conformance_errors().is_empty(),
        "{:?}",
        contract.conformance_errors()
    );

    let plan = fixture_plan();
    let admitted = synthesize_estimate_ladder(
        contract,
        &plan,
        "text_to_image",
        None,
        admitted_geometry,
        false,
        None,
        &[],
    );
    let selected = admitted
        .estimates
        .iter()
        .find(|candidate| candidate.selection.strategy == MemoryStrategy::BoundedDecode)
        .expect("admitted coordinate gets an estimated decode candidate");
    assert_eq!(selected.selection.parameters.decode_tile_edge, Some(768));
    assert_eq!(selected.selection.parameters.decode_overlap, Some(192));

    let mut second = contract
        .capability(MemoryStrategy::BoundedDecode)
        .unwrap()
        .parameters
        .decode_geometry_policies[0]
        .clone();
    second.tile_edge = 640;
    second.overlap = 160;
    second.production_evidence_sha256.clear();
    second = second.seal();
    let decode = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
        .unwrap();
    decode.parameters.decode_tile_edges = vec![640, 768];
    decode.parameters.decode_overlaps = vec![160, 192];
    decode.parameters.decode_geometry_policies.push(second);
    assert!(
        contract.conformance_errors().is_empty(),
        "{:?}",
        contract.conformance_errors()
    );
    let paired = synthesize_estimate_ladder(
        contract,
        &plan,
        "text_to_image",
        None,
        admitted_geometry,
        false,
        None,
        &[],
    );
    let pairs = paired
        .estimates
        .iter()
        .filter(|candidate| candidate.selection.strategy == MemoryStrategy::BoundedDecode)
        .map(|candidate| {
            (
                candidate.selection.parameters.decode_tile_edge.unwrap(),
                candidate.selection.parameters.decode_overlap.unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(pairs, vec![(640, 160), (768, 192), (640, 160), (768, 192)]);

    let off_grid = synthesize_estimate_ladder(
        contract,
        &plan,
        "text_to_image",
        None,
        MemoryGeometry {
            width: 1536,
            height: 1536,
            ..admitted_geometry
        },
        false,
        None,
        &[],
    );
    assert!(off_grid.estimates.iter().any(|candidate| {
        candidate.selection.strategy != MemoryStrategy::BoundedDecode
            && !contract.engages_selection(&candidate.selection, MemoryStrategy::BoundedDecode)
    }));
    assert!(off_grid.estimates.iter().all(|candidate| !contract
        .engages_selection(&candidate.selection, MemoryStrategy::BoundedDecode)));
    assert!(off_grid
        .decode_quality_decisions
        .iter()
        .any(|decision| matches!(decision, DecodeQualityRequestDecision::Unmeasured { .. })));

    let refused_geometry = MemoryGeometry {
        width: 1536,
        height: 1536,
        ..admitted_geometry
    };
    let mut exact_refusal = contract
        .capability(MemoryStrategy::BoundedDecode)
        .unwrap()
        .parameters
        .decode_geometry_policies[0]
        .clone();
    exact_refusal.geometry = refused_geometry;
    exact_refusal.disposition = MemoryDecodeQualityDisposition::Refused {
        reason: "seed 99 exceeded the exact semantic threshold".to_owned(),
    };
    exact_refusal.fixtures.last_mut().unwrap().observed_error = 52;
    exact_refusal.production_evidence_sha256.clear();
    exact_refusal = exact_refusal.seal();
    contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
        .unwrap()
        .parameters
        .decode_geometry_policies
        .push(exact_refusal);
    assert!(
        contract.conformance_errors().is_empty(),
        "{:?}",
        contract.conformance_errors()
    );
    let exact_refused = synthesize_estimate_ladder(
        contract,
        &plan,
        "text_to_image",
        None,
        refused_geometry,
        false,
        None,
        &[],
    );
    assert!(exact_refused.estimates.iter().any(|candidate| {
        candidate.selection.strategy != MemoryStrategy::BoundedDecode
            && !contract.engages_selection(&candidate.selection, MemoryStrategy::BoundedDecode)
    }));
    assert!(exact_refused
        .decode_quality_decisions
        .iter()
        .any(|decision| matches!(
            decision,
            DecodeQualityRequestDecision::Refused { reason, .. }
                if reason == "seed 99 exceeded the exact semantic threshold"
        )));

    // A provider that has not adopted the P9 table retains the pre-P9 smallest-range path.
    let legacy = fixture_generator();
    let legacy_contract = legacy.contract.as_ref().expect("legacy contract");
    let legacy_candidates = synthesize_estimate_ladder(
        legacy_contract,
        &plan,
        "text_to_image",
        None,
        admitted_geometry,
        false,
        None,
        &[],
    );
    let legacy_decode = legacy_candidates
        .estimates
        .iter()
        .find(|candidate| candidate.selection.strategy == MemoryStrategy::BoundedDecode)
        .expect("empty P9 table preserves estimated fallback");
    assert_eq!(
        legacy_decode.selection.parameters.decode_tile_edge,
        Some(512)
    );
    assert_eq!(legacy_decode.selection.parameters.decode_overlap, Some(128));

    // Declared-but-empty is not legacy. This is the state produced when SceneWorks strips a
    // packaged table after an artifact/source binding refusal while preserving its authority.
    // Even if a future provider keeps a route-blind decode range implemented, rung 2 must not
    // be synthesized from it; independent higher rungs remain eligible without decode.
    let mut refused = fixture_generator();
    let refused_contract = refused.contract.as_mut().expect("refused contract");
    refused_contract.decode_geometry_policy_authoritative = true;
    let refused_candidates = synthesize_estimate_ladder(
        refused_contract,
        &plan,
        "text_to_image",
        None,
        admitted_geometry,
        false,
        None,
        &[],
    );
    assert!(refused_candidates.estimates.iter().all(|candidate| {
        candidate.selection.strategy != MemoryStrategy::BoundedDecode
            && !refused_contract
                .engages_selection(&candidate.selection, MemoryStrategy::BoundedDecode)
    }));
    assert!(refused_candidates
        .decode_quality_decisions
        .iter()
        .any(|decision| {
            matches!(
                decision,
                DecodeQualityRequestDecision::Refused { reason, .. }
                    if reason.contains("declared quality scope")
            )
        }));
}

#[test]
fn manifest_reader_distinguishes_absent_valid_unproven_and_malformed_opt_in() {
    let spec = fixture_spec(gen_core::Quant::Q4, "packed-q4");
    let absent = MlxRequestPlan::for_spec_and_manifest(
        "fixture_provider",
        "fixture_model",
        &spec,
        Some(&JsonObject::new()),
        None,
    );
    assert!(matches!(absent.calibration, MlxCalibrationConfig::Absent));

    let valid_manifest = fixture_manifest(vec![fixture_calibration_json("q4", "packed-q4")]);
    let valid = MlxRequestPlan::for_spec_and_manifest(
        "fixture_provider",
        "fixture_model",
        &spec,
        Some(&valid_manifest),
        Some(fixture_provenance("q4", "packed-q4")),
    );
    let MlxCalibrationConfig::Valid(calibration) = valid.calibration else {
        panic!("well-formed opt-in must be valid");
    };
    assert_eq!(calibration.bindings.len(), 1);
    assert_eq!(calibration.resolved.identity.variant, "packed-q4");
    assert_eq!(valid.tier.quant, Some(gen_core::Quant::Q4));

    // An opt-in the resolver cannot back with immutable provenance is a NON-COVERING state, not
    // a malformed manifest: the install simply cannot prove which artifact it holds (a receipt
    // written before download-time tree stamps is the common cause). It must degrade to the
    // conservative legacy selector rather than refuse the request, which would strand the
    // install with no repair path.
    let unavailable = MlxRequestPlan::for_spec_and_manifest(
        "fixture_provider",
        "fixture_model",
        &spec,
        Some(&valid_manifest),
        None,
    );
    assert!(matches!(
        unavailable.calibration,
        MlxCalibrationConfig::Unproven
    ));
    assert_eq!(
        packaged_admission_route(
            &unavailable,
            &fixture_inputs(1024, 1024),
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("an unproven opt-in must degrade, never refuse")
        .path,
        AdmissionPath::Legacy
    );

    let mut malformed = fixture_calibration_json("q4", "packed-q4");
    malformed
        .as_object_mut()
        .expect("calibration object")
        .remove("fingerprint");
    let malformed_manifest = fixture_manifest(vec![malformed]);
    let malformed = MlxRequestPlan::for_spec_and_manifest(
        "fixture_provider",
        "fixture_model",
        &spec,
        Some(&malformed_manifest),
        Some(fixture_provenance("q4", "packed-q4")),
    );
    assert!(matches!(
        malformed.calibration,
        MlxCalibrationConfig::Invalid(_)
    ));
    assert!(packaged_admission_route(
        &malformed,
        &fixture_inputs(1024, 1024),
        "text_to_image",
        fixture_budget(8.0),
    )
    .expect_err("a malformed present opt-in must not collapse to packaged-empty legacy")
    .to_string()
    .contains("invalid MLX calibration opt-in"));
}

#[test]
fn parameter_reader_is_closed_and_preserves_transformer_component() {
    let cumulative = crate::memory_strategy::default_engaged_composition(
        StrategyRung::BoundedTransformerResidency,
    );
    let parameters = JsonObject::from_iter([
        ("decodeTileEdge".to_owned(), serde_json::json!(512)),
        ("decodeOverlap".to_owned(), serde_json::json!(128)),
        ("attentionChunkSize".to_owned(), serde_json::json!(256)),
        ("transformerWindowSize".to_owned(), serde_json::json!(4)),
        (
            "transformerWindowComponent".to_owned(),
            serde_json::json!("both"),
        ),
    ]);
    let parsed = parse_evidence_parameters(
        StrategyRung::BoundedTransformerResidency,
        &cumulative,
        &parameters,
    )
    .expect("the complete exact parameter set is valid");
    assert_eq!(parsed.decode_tile_edge, Some(512));
    assert_eq!(parsed.decode_overlap, Some(128));
    assert_eq!(parsed.attention_chunk_size, Some(256));
    assert_eq!(parsed.transformer_window_size, Some(4));
    assert_eq!(
        parsed.transformer_window_component,
        Some(TransformerComponent::Both)
    );

    let mut unknown = parameters.clone();
    unknown.insert("unrecognized".to_owned(), serde_json::json!(1));
    assert!(parse_evidence_parameters(
        StrategyRung::BoundedTransformerResidency,
        &cumulative,
        &unknown,
    )
    .expect_err("unknown parameters fail closed")
    .contains("unknown"));

    let mut malformed = parameters.clone();
    malformed.insert(
        "transformerWindowComponent".to_owned(),
        serde_json::json!(12),
    );
    assert!(parse_evidence_parameters(
        StrategyRung::BoundedTransformerResidency,
        &cumulative,
        &malformed,
    )
    .expect_err("a non-string transformer component fails closed")
    .contains("transformerWindowComponent"));

    let mut unsupported = parameters;
    unsupported.insert(
        "transformerWindowComponent".to_owned(),
        serde_json::json!("vae"),
    );
    assert!(parse_evidence_parameters(
        StrategyRung::BoundedTransformerResidency,
        &cumulative,
        &unsupported,
    )
    .expect_err("an unknown transformer component fails closed")
    .contains("unsupported"));
}

/// One calibration opt-in shaped like a real published binding, with the rung, declared
/// composition, parameters and materialization shape under test.
fn calibration_json_for(
    rung: &str,
    engaged: Option<&[&str]>,
    parameters: Value,
    load_shape: &str,
) -> Value {
    let mut value = fixture_calibration_json("q4", "packed-q4");
    let object = value.as_object_mut().expect("calibration object");
    object.insert("rung".to_owned(), serde_json::json!(rung));
    object.insert("parameters".to_owned(), parameters);
    object.insert("loadShape".to_owned(), serde_json::json!(load_shape));
    if let Some(engaged) = engaged {
        object.insert("engagedRungs".to_owned(), serde_json::json!(engaged));
    }
    value
}

const SDXL_SHAPED_COMPOSITION: [&str; 3] = [
    "resident",
    "staged_residency",
    "bounded_transformer_residency",
];

/// sc-17728. SDXL (sc-15525) and Kolors (sc-15521) both land rungs 0/1/4 `Implemented` with
/// rungs 2 and 3 `Missing` — measured and withheld, not unattempted. Their published rung-4
/// composition is `[resident, staged_residency, bounded_transformer_residency]`, and there is no
/// honest `decodeTileEdge` or `attentionChunkSize` to name because no selectable strategy on
/// those providers sets one. Keying the required set on the rung ORDINAL made that shape
/// structurally unrepresentable, so neither family could ever record authoritative evidence.
#[test]
fn a_rung_four_binding_records_evidence_without_the_rungs_its_provider_withholds() {
    for component in ["dit", "text_encoder", "both"] {
        let binding = MlxCalibrationBinding::parse(
            &calibration_json_for(
                "bounded_transformer_residency",
                Some(&SDXL_SHAPED_COMPOSITION),
                serde_json::json!({
                    "transformerWindowSize": 1,
                    "transformerWindowComponent": component,
                }),
                "deferred_materialization",
            ),
            0,
        )
        .unwrap_or_else(|error| {
            panic!("the published SDXL/Kolors rung-4 shape must bind evidence: {error}")
        });
        assert_eq!(
            binding.selection_parameters.transformer_window_size,
            Some(1)
        );
        assert!(binding
            .selection_parameters
            .transformer_window_component
            .is_some());
        // The withheld rungs contribute nothing rather than a fabricated default.
        assert_eq!(binding.selection_parameters.decode_tile_edge, None);
        assert_eq!(binding.selection_parameters.decode_overlap, None);
        assert_eq!(binding.selection_parameters.attention_chunk_size, None);
    }
}

/// The Z-Image/Qwen shape: rung 4 over a cumulative scratch composition that never engages
/// staged residency. At the pinned contract revision rung 4's SHARED prerequisite is
/// `LoadShape::DeferredMaterialization`, and `MemoryStrategy::engages` states outright that rung
/// 4 does not engage rung 1; a rung-1 edge is provider-specific (mlx-gen-anima declares one,
/// mlx-gen-z-image does not). So the rung-1-free composition must keep binding.
#[test]
fn a_rung_four_binding_without_rung_one_is_legitimate_when_the_contract_says_so() {
    let binding = MlxCalibrationBinding::parse(
        &calibration_json_for(
            "bounded_transformer_residency",
            Some(&[
                "resident",
                "bounded_decode",
                "bounded_attention",
                "bounded_transformer_residency",
            ]),
            serde_json::json!({
                "decodeTileEdge": 512,
                "decodeOverlap": 64,
                "attentionChunkSize": 67_108_864_u64,
                "transformerWindowSize": 1,
            }),
            "deferred_materialization",
        ),
        0,
    )
    .expect("the checked-in Z-Image rung-4 plan composition must bind");
    assert_eq!(binding.selection_parameters.decode_tile_edge, Some(512));
    assert_eq!(
        binding.selection_parameters.transformer_window_size,
        Some(1)
    );
    // The shared cost-order default is the same composition, so the shape also binds with no
    // declaration at all — the rung-1-free reading is not an artefact of declaring it.
    assert!(crate::memory_strategy::default_engaged_composition(
        StrategyRung::BoundedTransformerResidency
    )
    .iter()
    .all(|rung| *rung != StrategyRung::StagedResidency));
}

/// Fail-closed in the other direction: the fix derives the required set, it does not make fields
/// optional. A rung the composition does not engage owns no parameter, so naming one is an
/// error rather than harmless noise.
#[test]
fn naming_a_withheld_rungs_parameters_is_an_error() {
    for (parameters, needle) in [
        (
            serde_json::json!({
                "transformerWindowSize": 1,
                "decodeTileEdge": 512,
                "decodeOverlap": 64,
            }),
            "forbids decodeTileEdge",
        ),
        (
            serde_json::json!({
                "transformerWindowSize": 1,
                "attentionChunkSize": 67_108_864_u64,
            }),
            "forbids attentionChunkSize",
        ),
    ] {
        let error = MlxCalibrationBinding::parse(
            &calibration_json_for(
                "bounded_transformer_residency",
                Some(&SDXL_SHAPED_COMPOSITION),
                parameters,
                "deferred_materialization",
            ),
            0,
        )
        .expect_err("a withheld rung's parameters must not be nameable");
        assert!(error.contains(needle), "{error}");
    }
    // The same rule the other way round: a rung-2 binding cannot name the transformer scope.
    let error = MlxCalibrationBinding::parse(
        &calibration_json_for(
            "bounded_decode",
            None,
            serde_json::json!({
                "decodeTileEdge": 512,
                "decodeOverlap": 64,
                "transformerWindowComponent": "dit",
            }),
            "deferred_materialization",
        ),
        0,
    )
    .expect_err("a rung-2 binding does not engage rung 4");
    assert!(
        error.contains("forbids transformerWindowComponent"),
        "{error}"
    );
}

/// An ENGAGED rung must still name every parameter it owns. This is the guard the fix had to
/// keep: derived, not relaxed.
#[test]
fn an_engaged_rung_must_still_name_every_parameter_it_owns() {
    for (engaged, parameters, needle) in [
        (
            SDXL_SHAPED_COMPOSITION.to_vec(),
            serde_json::json!({ "transformerWindowComponent": "dit" }),
            "requires transformerWindowSize",
        ),
        (
            vec![
                "resident",
                "bounded_decode",
                "bounded_transformer_residency",
            ],
            serde_json::json!({ "decodeOverlap": 64, "transformerWindowSize": 1 }),
            "requires decodeTileEdge",
        ),
        (
            vec![
                "resident",
                "bounded_attention",
                "bounded_transformer_residency",
            ],
            serde_json::json!({ "transformerWindowSize": 1 }),
            "requires attentionChunkSize",
        ),
    ] {
        let error = MlxCalibrationBinding::parse(
            &calibration_json_for(
                "bounded_transformer_residency",
                Some(&engaged),
                parameters,
                "deferred_materialization",
            ),
            0,
        )
        .expect_err("an engaged rung must name its own parameters");
        assert!(error.contains(needle), "{error}");
    }
}

/// The declaration itself is closed. A binding cannot dodge a required parameter by publishing
/// an incoherent composition.
#[test]
fn a_declared_composition_is_itself_closed() {
    for (engaged, needle) in [
        (
            vec!["staged_residency", "bounded_transformer_residency"],
            "must contain resident",
        ),
        (
            vec!["resident", "staged_residency"],
            "must contain resident and the selected rung",
        ),
        (
            vec![
                "resident",
                "bounded_transformer_residency",
                "staged_residency",
            ],
            "canonical ladder order",
        ),
        (
            vec!["resident", "resident", "bounded_transformer_residency"],
            "unique set",
        ),
    ] {
        let error = MlxCalibrationBinding::parse(
            &calibration_json_for(
                "bounded_transformer_residency",
                Some(&engaged),
                serde_json::json!({ "transformerWindowSize": 1 }),
                "deferred_materialization",
            ),
            0,
        )
        .expect_err("an incoherent composition must fail closed");
        assert!(error.contains(needle), "{error}");
    }
    // A cheaper selection cannot claim to have engaged a costlier rung's mechanism.
    let error = MlxCalibrationBinding::parse(
        &calibration_json_for(
            "bounded_decode",
            Some(&[
                "resident",
                "bounded_decode",
                "bounded_transformer_residency",
            ]),
            serde_json::json!({ "decodeTileEdge": 512, "decodeOverlap": 64 }),
            "deferred_materialization",
        ),
        0,
    )
    .expect_err("a rung-2 selection cannot engage rung 4");
    assert!(error.contains("costlier than the selected rung"), "{error}");
}

/// Regression guard for the shape that already worked. A provider with the whole ladder
/// implemented (Anima) publishes the cumulative composition, and a binding that declares no
/// composition at all still gets exactly the pre-sc-17728 fixed required set.
#[test]
fn the_full_ladder_shape_still_validates_unchanged() {
    let cumulative = serde_json::json!({
        "decodeTileEdge": 512,
        "decodeOverlap": 128,
        "attentionChunkSize": 256,
        "transformerWindowSize": 4,
        "transformerWindowComponent": "both",
    });
    for engaged in [
        None,
        Some(
            [
                "resident",
                "bounded_decode",
                "bounded_attention",
                "bounded_transformer_residency",
            ]
            .as_slice(),
        ),
        // Anima's MLX provider appends a rung-1 edge to the shared graph, so its rung-4
        // composition additionally engages staged residency.
        Some(
            [
                "resident",
                "staged_residency",
                "bounded_decode",
                "bounded_attention",
                "bounded_transformer_residency",
            ]
            .as_slice(),
        ),
    ] {
        let binding = MlxCalibrationBinding::parse(
            &calibration_json_for(
                "bounded_transformer_residency",
                engaged,
                cumulative.clone(),
                "deferred_materialization",
            ),
            0,
        )
        .expect("the full-ladder shape must keep binding");
        assert_eq!(binding.selection_parameters.decode_tile_edge, Some(512));
        assert_eq!(binding.selection_parameters.decode_overlap, Some(128));
        assert_eq!(binding.selection_parameters.attention_chunk_size, Some(256));
        assert_eq!(
            binding.selection_parameters.transformer_window_size,
            Some(4)
        );
        assert_eq!(
            binding.selection_parameters.transformer_window_component,
            Some(TransformerComponent::Both)
        );
    }
    // Omitting the composition keeps the cumulative default, so an omitted lower-rung parameter
    // is still rejected exactly as before.
    let error = MlxCalibrationBinding::parse(
        &calibration_json_for(
            "bounded_transformer_residency",
            None,
            serde_json::json!({ "transformerWindowSize": 4 }),
            "deferred_materialization",
        ),
        0,
    )
    .expect_err("the cumulative default still requires the lower rungs' parameters");
    assert!(error.contains("requires decodeTileEdge"), "{error}");
}

#[test]
fn fallback_reason_priority_is_independent_of_binding_order() {
    let fold = |reasons: &[LegacyAdmissionReason]| {
        reasons
            .iter()
            .copied()
            .fold(LegacyAdmissionReason::NoRecord, stronger_fallback_reason)
    };
    let forward = [
        LegacyAdmissionReason::OutOfEnvelope,
        LegacyAdmissionReason::FingerprintMismatch,
        LegacyAdmissionReason::IdentityMismatch,
    ];
    let mut reversed = forward;
    reversed.reverse();
    assert_eq!(
        fold(&forward),
        LegacyAdmissionReason::IdentityMismatch,
        "the strongest drift reason wins"
    );
    assert_eq!(
        fold(&forward),
        fold(&reversed),
        "reversing calibration binding order must not change the fallback reason"
    );
}

#[test]
fn calibration_collection_supports_tiers_and_cells_but_rejects_duplicate_selectors() {
    let mut second_cell = fixture_calibration_json("q4", "packed-q4");
    second_cell["geometry"]["width"] = serde_json::json!(512);
    second_cell["geometry"]["height"] = serde_json::json!(512);
    let manifest = fixture_manifest(vec![
        fixture_calibration_json("q4", "packed-q4"),
        second_cell,
        fixture_calibration_json("q8", "packed-q8"),
    ]);
    let bindings = MlxCalibrationBinding::from_manifest(&manifest)
        .expect("valid collection")
        .expect("present collection");
    assert_eq!(bindings.len(), 3);
    assert_eq!(
        bindings
            .iter()
            .filter(|binding| binding.tier == "q4")
            .count(),
        2,
        "one tier may contain multiple exact request cells"
    );
    assert!(bindings.iter().any(|binding| binding.tier == "q8"));

    let duplicate = fixture_manifest(vec![
        fixture_calibration_json("q4", "packed-q4"),
        fixture_calibration_json("q4", "packed-q4"),
    ]);
    assert!(MlxCalibrationBinding::from_manifest(&duplicate)
        .expect_err("duplicate request selectors are ambiguous")
        .contains("ambiguous duplicates"));
    let mut resident = fixture_calibration_json("q4", "packed-q4");
    resident["rung"] = serde_json::json!("resident");
    resident["parameters"] = serde_json::json!({});
    let ladder = fixture_manifest(vec![fixture_calibration_json("q4", "packed-q4"), resident]);
    assert_eq!(
        MlxCalibrationBinding::from_manifest(&ladder)
            .expect("distinct rungs in one cell are a ladder, not duplicates")
            .expect("present ladder")
            .len(),
        2
    );

    let mut bundle = fixture_bundle();
    let mut second_cell_record = bundle.records[0].clone();
    second_cell_record.target.geometry.width = 512;
    second_cell_record.target.geometry.height = 512;
    bundle.records.push(second_cell_record);

    let q4_plan = MlxRequestPlan {
        calibration: MlxCalibrationConfig::Valid(MlxCalibrationSet {
            bindings,
            resolved: fixture_provenance("q4", "packed-q4"),
        }),
        ..fixture_plan()
    };
    assert_eq!(
        evidence_admission_route(
            &bundle,
            &q4_plan,
            &fixture_inputs(512, 512),
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("the second exact cell is independently selectable")
        .path,
        AdmissionPath::Evidence
    );
}

#[test]
fn load_spec_tier_and_resolved_artifact_variant_are_independent() {
    let manifest = fixture_manifest(vec![
        fixture_calibration_json("q4", "packed-q4"),
        fixture_calibration_json("q8", "packed-q8"),
    ]);
    let q4 = MlxRequestPlan::for_spec_and_manifest(
        "fixture_provider",
        "fixture_model",
        &fixture_spec(gen_core::Quant::Q4, "packed-q4"),
        Some(&manifest),
        Some(fixture_provenance("q4", "packed-q4")),
    );
    let MlxCalibrationConfig::Valid(q4_calibration) = &q4.calibration else {
        panic!("q4 binding");
    };
    assert_eq!(q4.tier.quant, Some(gen_core::Quant::Q4));
    assert_eq!(q4_calibration.resolved.identity.variant, "packed-q4");
    assert_eq!(
        evidence_admission_route(
            &fixture_bundle(),
            &q4,
            &fixture_inputs(1024, 1024),
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("q4 packed artifact is independently verified")
        .path,
        AdmissionPath::Evidence
    );

    let dense_load_q4 = MlxRequestPlan::for_spec_and_manifest(
        "fixture_provider",
        "fixture_model",
        &LoadSpec::new(WeightsSource::Dir("/resolved/packed-q4".into())),
        Some(&manifest),
        Some(fixture_provenance("q4", "packed-q4")),
    );
    assert_eq!(
        dense_load_q4.tier.quant,
        Some(gen_core::Quant::Q4),
        "resolver tier must win when a packed transformer keeps LoadSpec.quantize=None"
    );
    let relabeled_packed_q4 = MlxRequestPlan::for_spec_and_manifest(
        "fixture_provider",
        "fixture_model",
        &fixture_spec(gen_core::Quant::Q8, "packed-q4"),
        Some(&manifest),
        Some(fixture_provenance("q4", "packed-q4")),
    );
    assert!(
        matches!(
            relabeled_packed_q4.calibration,
            MlxCalibrationConfig::Invalid(_)
        ),
        "request quantization must not relabel a fixed packed-q4 artifact as q8"
    );

    let wrong_variant = MlxRequestPlan::for_spec_and_manifest(
        "fixture_provider",
        "fixture_model",
        &fixture_spec(gen_core::Quant::Q4, "different-packed-q4"),
        Some(&manifest),
        Some(fixture_provenance("q4", "different-packed-q4")),
    );
    assert_eq!(
        evidence_admission_route(
            &fixture_bundle(),
            &wrong_variant,
            &fixture_inputs(1024, 1024),
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("unverified artifact variant uses legacy")
        .fallback_reason,
        Some(LegacyAdmissionReason::IdentityMismatch)
    );

    let q8 = MlxRequestPlan::for_spec_and_manifest(
        "fixture_provider",
        "fixture_model",
        &fixture_spec(gen_core::Quant::Q8, "packed-q8"),
        Some(&manifest),
        Some(fixture_provenance("q8", "packed-q8")),
    );
    let MlxCalibrationConfig::Valid(q8_calibration) = &q8.calibration else {
        panic!("q8 binding");
    };
    assert_eq!(q8.tier.quant, Some(gen_core::Quant::Q8));
    assert_eq!(q8_calibration.resolved.identity.variant, "packed-q8");

    let dense_load_q8 = MlxRequestPlan::for_spec_and_manifest(
        "fixture_provider",
        "fixture_model",
        &LoadSpec::new(WeightsSource::Dir("/resolved/packed-q8".into())),
        Some(&manifest),
        Some(fixture_provenance("q8", "packed-q8")),
    );
    assert_eq!(
        dense_load_q8.tier.quant,
        Some(gen_core::Quant::Q8),
        "resolved packed q8 must not be mislabeled bf16 by a dense text-encoder LoadSpec"
    );

    let mut q8_bundle = fixture_bundle();
    let mut q8_record = q8_bundle.records[0].clone();
    q8_record.target.tier = "q8".to_owned();
    q8_record.artifact.variant = "packed-q8".to_owned();
    q8_record.loadability.resolved_path_fingerprint =
        sceneworks_core::memory_calibration::RequiredNullable::Value(format!(
            "SceneWorks/fixture@{}:packed-q8",
            "c".repeat(40)
        ));
    q8_bundle.records.push(q8_record);
    assert_eq!(
        evidence_admission_route(
            &q8_bundle,
            &q8,
            &fixture_inputs(1024, 1024),
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("q8 record is selected independently of q4")
        .path,
        AdmissionPath::Evidence
    );
}

#[tokio::test]
async fn flat_default_receipt_keeps_identity_orthogonal_to_q4_q8_execution() {
    let data = tempfile::tempdir().expect("data dir");
    let hub = data.path().join("hub");
    let _env = crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().expect("hub path"))]);
    let repo = "SceneWorks/fixture";
    let revision = "c".repeat(40);
    let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(data.path(), repo)
        .expect("cache")
        .join("snapshots")
        .join(&revision);
    std::fs::create_dir_all(&snapshot).expect("snapshot");
    std::fs::write(snapshot.join("weights.safetensors"), b"flat-q8").expect("weights");
    let resolved_files = vec!["weights.safetensors".to_owned()];
    let stamp = crate::model_jobs::resolved_files_tree_stamp(&snapshot, &resolved_files)
        .expect("tree stamp");
    let payload = JsonObject::from_iter([
        ("modelId".to_owned(), serde_json::json!("fixture_model")),
        ("variant".to_owned(), serde_json::json!("default")),
        ("mlx".to_owned(), serde_json::json!({ "quantize": 8 })),
    ]);
    let resolved_tier = crate::model_jobs::download_payload_resolved_tier(&payload);
    assert_eq!(
        resolved_tier, None,
        "a flat dense artifact is not permanently labeled by its default runtime quantization"
    );
    let marker_dir = data
        .path()
        .join("models")
        .join(crate::paths::safe_download_dir("fixture_model"));
    crate::write_model_download_receipt(
        &marker_dir,
        &payload,
        repo,
        "job-flat-default",
        &resolved_files,
        Some(&revision),
        crate::imports::DownloadArtifactReceipt {
            resolved_tier: resolved_tier.as_deref(),
            tree_stamp: Some(&stamp),
            ..Default::default()
        },
    )
    .await
    .expect("real receipt writer");
    let resolved = crate::model_jobs::huggingface_receipt_weights(
        data.path(),
        repo,
        Some("fixture_model"),
        Some("default"),
        crate::model_jobs::ProvenanceRepair::Skip,
    )
    .expect("receipt resolver");
    assert_eq!(resolved.path, snapshot);
    let provenance = resolved.provenance.expect("trusted provenance");
    assert_eq!(provenance.fixed_artifact_tier.as_deref(), None);

    let calibration = |tier| {
        let mut calibration = fixture_calibration_json(tier, "default");
        calibration["artifactResolvedRevision"] = serde_json::json!(revision);
        calibration["artifactVariant"] = serde_json::json!("default");
        calibration["resolvedPathFingerprint"] = serde_json::json!(provenance.identity.fingerprint);
        calibration
    };
    let manifest = fixture_manifest(vec![calibration("q4"), calibration("q8")]);
    for quant in [gen_core::Quant::Q4, gen_core::Quant::Q8] {
        let plan = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &LoadSpec::new(WeightsSource::Dir(resolved.path.clone())).with_quant(quant),
            Some(&manifest),
            Some(provenance.clone()),
        );
        assert_eq!(plan.tier.quant, Some(quant));
        assert!(matches!(plan.calibration, MlxCalibrationConfig::Valid(_)));
    }
}

#[test]
fn decision_two_transition_paths_are_explicit_and_fail_closed_only_when_covered() {
    let bundle = fixture_bundle();
    let inputs = fixture_inputs(1024, 1024);

    let mut uncalibrated = fixture_plan();
    uncalibrated.calibration = MlxCalibrationConfig::Absent;
    let route = evidence_admission_route(
        &bundle,
        &uncalibrated,
        &inputs,
        "text_to_image",
        fixture_budget(8.0),
    )
    .expect("an uncalibrated model uses legacy");
    assert_eq!(route.path, AdmissionPath::Legacy);
    assert_eq!(
        route.fallback_reason,
        Some(LegacyAdmissionReason::NoBinding)
    );

    let uncovered = evidence_admission_route(
        &bundle,
        &fixture_plan(),
        &fixture_inputs(768, 768),
        "text_to_image",
        fixture_budget(8.0),
    )
    .expect("an uncovered geometry uses legacy");
    assert_eq!(uncovered.path, AdmissionPath::Legacy);
    assert_eq!(
        uncovered.fallback_reason,
        Some(LegacyAdmissionReason::OutOfEnvelope)
    );

    let mut drifted = fixture_plan();
    let MlxCalibrationConfig::Valid(calibration) = &mut drifted.calibration else {
        panic!("fixture calibration");
    };
    calibration.bindings[0].query.fingerprint = "different".to_owned();
    let drifted = evidence_admission_route(
        &bundle,
        &drifted,
        &inputs,
        "text_to_image",
        fixture_budget(8.0),
    )
    .expect("fingerprint drift uses legacy");
    assert_eq!(drifted.path, AdmissionPath::Legacy);
    assert_eq!(
        drifted.fallback_reason,
        Some(LegacyAdmissionReason::FingerprintMismatch)
    );

    let covered = evidence_admission_route(
        &bundle,
        &fixture_plan(),
        &inputs,
        "text_to_image",
        fixture_budget(8.0),
    )
    .expect("the exact covered cell fits its captured safe envelope");
    assert_eq!(covered.path, AdmissionPath::Evidence);
    assert_eq!(
        covered.process_limit_bytes, None,
        "the process ceiling is candidate-specific and cannot be chosen before strategy selection"
    );
    assert_eq!(
        budget_for_admission(fixture_budget(8.0), &covered).reserved_headroom_bytes,
        0,
        "no sibling candidate reserve may be imposed before strategy selection"
    );
    let evidence = covered
        .evidence
        .first()
        .expect("verified selector evidence")
        .evidence
        .clone();
    assert_eq!(evidence.predicted_peak_bytes, gib_to_bytes(5.0));
    // sc-18864: 3 GiB is the fixture's NON-RECLAIMABLE residency (`activeBytes`). It read 4 GiB
    // when the envelope recovered it as `wiredBytes - reclaimableBytes`, and `wiredBytes` was a
    // copy of the two-instant allocator bound. Still distinct from the 5 GiB prediction, so the
    // assertion below still discriminates observed telemetry from the predicted maximum.
    assert_eq!(
        evidence.observed_peak_bytes,
        Some(gib_to_bytes(3.0)),
        "observed telemetry remains the measured counter rather than the predicted maximum"
    );
    let evaluated = evaluate_request_with_budget_using_bundle(
        &fixture_generator(),
        &fixture_plan(),
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(8.0),
        gib_to_bytes(4.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect("selected exact candidate");
    assert_eq!(evaluated.process_limit_bytes, Some(gib_to_bytes(5.0)));
    assert_eq!(
        evaluated.context.budget.reserved_headroom_bytes,
        gib_to_bytes(3.0)
    );

    let unfit = evaluate_request_with_budget_using_bundle(
        &fixture_generator(),
        &fixture_plan(),
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(6.0),
        gib_to_bytes(5.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect_err("the exact covered 5 GiB cell must reject when only 3 GiB is safely available");
    let unfit = unfit.to_string();
    assert!(unfit.contains("smallest verified MLX host boundary"));
    assert!(
            unfit.contains("needs at least 8.00 GiB"),
            "the refusal must quote the static proportional boundary, not the 7.25 GiB reserve sum evaluated only at this 6 GiB host: {unfit}"
        );
}

#[test]
fn persisted_uniform_tier_evidence_cannot_authorize_component_precision_floors() {
    use gen_core::{ComponentPrecisionFloor, PrecisionFloorComponent};

    const FLOORS: &[ComponentPrecisionFloor] = &[ComponentPrecisionFloor {
        component: PrecisionFloorComponent::TransformerHead,
        selected_tier: gen_core::Quant::Q4,
        resident_tier: gen_core::Quant::Q8,
    }];
    let bundle = fixture_bundle();
    let mut generator = fixture_generator();
    generator.descriptor.capabilities.component_precision_floors = FLOORS;
    let evaluated = evaluate_request_with_budget_using_bundle(
        &generator,
        &fixture_plan(),
        &fixture_inputs(1024, 1024),
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(8.0),
        gib_to_bytes(4.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect("a mixed-precision provider falls back to its conservative resident estimate");

    assert_eq!(evaluated.process_limit_bytes, None);
    assert_eq!(
        evaluated.context.selection.strategy,
        MemoryStrategy::Resident,
        "a coarse persisted q4 record must not be relabeled as mixed q4/q8 evidence"
    );
    assert_eq!(
        evaluated.context.selection.tier.component_precision_floors,
        FLOORS
    );
}

#[test]
fn provider_floor_binding_stays_inactive_for_q8_and_bf16() {
    use gen_core::{ComponentPrecisionFloor, PrecisionFloorComponent};

    const Q4_ONLY_FLOORS: &[ComponentPrecisionFloor] = &[ComponentPrecisionFloor {
        component: PrecisionFloorComponent::TransformerHead,
        selected_tier: gen_core::Quant::Q4,
        resident_tier: gen_core::Quant::Q8,
    }];
    let mut generator = fixture_generator();
    generator.descriptor.capabilities.component_precision_floors = Q4_ONLY_FLOORS;

    for (quant, label) in [(Some(gen_core::Quant::Q8), "q8"), (None, "bf16")] {
        let mut plan = fixture_plan();
        plan.tier.quant = quant;
        plan.calibration = MlxCalibrationConfig::Absent;
        let evaluated = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &fixture_inputs(1024, 1024),
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            crate::execution_planner::WarmPolicyProposal::inert("fixture"),
            fixture_budget(8.0),
            gib_to_bytes(4.0),
            0,
            &[],
            None,
            None,
        )
        .unwrap_or_else(|error| panic!("{label} provider binding failed: {error}"));

        assert!(
            evaluated
                .context
                .selection
                .tier
                .component_precision_floors
                .is_empty(),
            "Q4-only component floors must not relabel a {label} request"
        );
    }
}

#[test]
fn target_tier_and_artifact_variant_are_independent_identity_mutations() {
    let bundle = fixture_bundle();
    let inputs = fixture_inputs(1024, 1024);

    let mut wrong_tier = fixture_plan();
    wrong_tier.tier.quant = Some(gen_core::Quant::Q8);
    let MlxCalibrationConfig::Valid(calibration) = &mut wrong_tier.calibration else {
        panic!("fixture calibration");
    };
    calibration.resolved.fixed_artifact_tier = Some("q8".to_owned());
    let wrong_tier = evidence_admission_route(
        &bundle,
        &wrong_tier,
        &inputs,
        "text_to_image",
        fixture_budget(8.0),
    )
    .expect("target-tier mismatch falls back");
    assert_eq!(
        wrong_tier.fallback_reason,
        Some(LegacyAdmissionReason::IdentityMismatch)
    );

    let mut wrong_artifact = fixture_plan();
    let MlxCalibrationConfig::Valid(calibration) = &mut wrong_artifact.calibration else {
        panic!("fixture calibration");
    };
    calibration.resolved.identity.variant = "different-packed-q4".to_owned();
    calibration.resolved.identity.fingerprint =
        format!("SceneWorks/fixture@{}:different-packed-q4", "c".repeat(40));
    let wrong_artifact = evidence_admission_route(
        &bundle,
        &wrong_artifact,
        &inputs,
        "text_to_image",
        fixture_budget(8.0),
    )
    .expect("artifact mismatch falls back as drift");
    assert_eq!(
        wrong_artifact.fallback_reason,
        Some(LegacyAdmissionReason::IdentityMismatch)
    );

    let mut wrong_provider = fixture_plan();
    let MlxCalibrationConfig::Valid(calibration) = &mut wrong_provider.calibration else {
        panic!("fixture calibration");
    };
    calibration.bindings[0].provider = "fixture_model".to_owned();
    let wrong_provider = evidence_admission_route(
        &bundle,
        &wrong_provider,
        &inputs,
        "text_to_image",
        fixture_budget(8.0),
    )
    .expect("provider mismatch falls back as drift");
    assert_eq!(
        wrong_provider.fallback_reason,
        Some(LegacyAdmissionReason::IdentityMismatch),
        "the binding provider must match the actual engine route, not the catalog model id"
    );
}

/// sc-18101 #0 REGRESSION GUARD: a load-shape mismatch must DEGRADE to the estimate ladder,
/// never refuse the request.
///
/// `MemoryCalibrationIdentity` carries three fields, and gen-core's `optimized_eligibility`
/// rejects a candidate whose `key.load_shape` disagrees with the contract's — returning
/// `FingerprintMismatch`. The identity demotion above `evaluate_request_with_budget_using_bundle`
/// used to compare only `abi` and `fingerprint`, so a shape mismatch sailed past it into
/// `AdmissionPath::Evidence`, lost its only candidate inside `select_strategy`, and refused with
/// "no structurally admissible MLX memory strategy" — with no estimate ladder behind it, because
/// synthesis runs only on the Legacy route.
///
/// That hole shipped: `mlx:qwen_image` q8 1024² was captured `eager_materialization` while
/// `image_jobs::apply_measured_mlx_load_shape` forces `DeferredMaterialization` on every
/// `qwen_image` directory load, so the flagship q8 route hard-refused its most common geometry on
/// a 128 GiB machine. Verified on real weights at this commit and at the epic's base commit
/// (`docs/epic-18093-end-to-end-validation-sc-18101.md`): pre-epic it was ADMITTED, post-epic
/// REFUSED, post-fix ADMITTED again at the same predicted peak.
///
/// The fixture reproduces it minimally: the bundle's records are `eager`, the contract is moved
/// to `deferred` with its fingerprint and abi UNCHANGED, so `load_shape` is the only field that
/// disagrees. The mutation arm below puts the shape back and shows the same request reaches
/// `AdmissionPath::Evidence`, which is what makes this test discriminate rather than pass
/// vacuously.
#[test]
fn a_load_shape_mismatch_degrades_to_estimates_instead_of_refusing() {
    use gen_core::MemoryCalibrationIdentity;

    let bundle = fixture_bundle();
    let plan = fixture_plan();
    let inputs = fixture_inputs(1024, 1024);

    let mut generator = fixture_generator();
    let contract = generator.contract.as_mut().expect("fixture contract");
    let fingerprint = contract
        .calibration
        .as_ref()
        .expect("fixture calibration")
        .fingerprint
        .clone();
    // ONLY the materialization shape moves. Same provider, same artifact, same abi, same
    // formula fingerprint — exactly the qwen_image production shape.
    contract.load_shape = gen_core::LoadShape::DeferredMaterialization;
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        fingerprint,
        gen_core::LoadShape::DeferredMaterialization,
    ));

    let admitted = evaluate_request_with_budget_using_bundle(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(64.0),
        gib_to_bytes(9.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect("a load-shape mismatch must degrade to the estimate ladder, not refuse the request");
    // Served from a synthesized estimate, not from a measurement: no verified record was
    // selected, so there is no request-scoped process ceiling.
    assert_eq!(
        admitted.process_limit_bytes, None,
        "a degraded cell must not claim a verified record's request-scoped ceiling"
    );

    // Mutation arm: restore the shape the records were captured under and the SAME request
    // reaches calibrated admission. Without this the test would also pass if the gate refused
    // everything for an unrelated reason.
    let mut matched = fixture_generator();
    matched
        .contract
        .as_mut()
        .expect("fixture contract")
        .load_shape = gen_core::LoadShape::EagerMaterialization;
    let route = evidence_admission_route(
        &bundle,
        &plan,
        &inputs,
        "text_to_image",
        fixture_budget(64.0),
    )
    .expect("the matching-shape cell routes without error");
    assert_eq!(
        route.path,
        AdmissionPath::Evidence,
        "with the captured shape restored the cell must reach calibrated admission: {:?}",
        route.fallback_reason
    );
    // …and the filter must SPARE a candidate the loaded provider can serve while dropping its
    // unusable sibling. This is the arm that separates the shipped per-candidate filter from a
    // whole-route demotion, and it is the fix's central design decision: `qwen_image` q8 ships
    // one eager and one deferred binding on the same route, so demoting the whole route would
    // silently discard the matching measurement and serve an estimate instead. An earlier
    // iteration of this fix did exactly that.
    //
    // Asserting `process_limit_bytes.is_some()` is NOT sufficient here and was the gap a review
    // caught: with only one record in the bundle there is no sibling to spare, so
    // `evidence.clear()` — a literal whole-route demotion — passed. The bundle now carries a
    // second record on `bounded_transformer_residency` captured `deferred_materialization`, and
    // the assertion is on the surviving SET.
    let mut two_binding_plan = plan.clone();
    let MlxCalibrationConfig::Valid(calibration) = &mut two_binding_plan.calibration else {
        panic!("the fixture plan opts in to calibration");
    };
    let mut sibling = fixture_binding_for(
        "q4",
        "packed-q4",
        StrategyRung::BoundedTransformerResidency,
        JsonObject::from_iter([
            ("decodeTileEdge".to_owned(), serde_json::json!(512)),
            ("decodeOverlap".to_owned(), serde_json::json!(128)),
            ("attentionChunkSize".to_owned(), serde_json::json!(65536)),
            ("transformerWindowSize".to_owned(), serde_json::json!(1)),
        ]),
    );
    // The binding must declare the shape its RECORD was captured under, exactly as the shipped
    // `qwen_image` q8 pair does — `EvidenceBundle::evidence_for` matches on it.
    sibling.query.load_shape =
        sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
    calibration.bindings.push(sibling);

    let route = evidence_admission_route(
        &bundle,
        &two_binding_plan,
        &inputs,
        "text_to_image",
        fixture_budget(64.0),
    )
    .expect("both bindings route without error");
    assert_eq!(
        route.evidence.len(),
        2,
        "the fixture must present BOTH a shape-matching and a mismatching candidate, or this \
             arm cannot tell a filter from a whole-route demotion"
    );

    let calibrated = evaluate_request_with_budget_using_bundle(
        &matched,
        &two_binding_plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(64.0),
        gib_to_bytes(9.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect("the matching-shape cell is admitted");
    assert!(
        calibrated.process_limit_bytes.is_some(),
        "a shape-matching measured cell must still be served from its RECORD (which supplies \
             the request-scoped ceiling), not degraded to an estimate"
    );
    // The survivor is the shape-matching rung, and it survived ALONE: `clear()` in place of
    // `retain()` empties this and degrades the route, which the selection below would then have
    // served from an estimate with no `process_limit_bytes`.
    assert_eq!(
        calibrated.context.selection.strategy,
        MemoryStrategy::BoundedDecode,
        "the eager `bounded_decode` cell is the one this eager contract can serve; selecting \
             anything else means the filter dropped it"
    );
    assert_eq!(
        calibrated
            .context
            .selection
            .parameters
            .transformer_window_size,
        None,
        "the deferred rung-4 sibling must not have been selected"
    );

    // Third arm: the Resident EXEMPTION. `optimized_eligibility` short-circuits `Ok(())` for a
    // non-optimized selection before it compares load shapes (pinned against gen-core by
    // `gen_core_accepts_a_resident_cell_whose_load_shape_disagrees`), so a resident cell measured
    // under the other shape is one the downstream gate ACCEPTS — and this filter must not be
    // stricter than the gate it anticipates. Dropping the exemption would silently discard a
    // usable measurement, which is the same failure mode as the whole-route demotion.
    //
    // The bundle carries a resident cell captured `deferred_materialization`; bind it alongside
    // the eager `bounded_decode` cell and select under the EAGER contract. If the exemption
    // holds, the resident record survives the filter and the selector takes it first (rung
    // order), serving a RECORD — `process_limit_bytes` is `Some`. Without the exemption it is
    // filtered out and `bounded_decode` is selected instead.
    let mut resident_plan = plan.clone();
    let MlxCalibrationConfig::Valid(resident_calibration) = &mut resident_plan.calibration else {
        panic!("the fixture plan opts in to calibration");
    };
    let mut resident_binding =
        fixture_binding_for("q4", "packed-q4", StrategyRung::Resident, JsonObject::new());
    resident_binding.query.load_shape =
        sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
    resident_calibration.bindings.push(resident_binding);

    let exempted = evaluate_request_with_budget_using_bundle(
        &matched,
        &resident_plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(64.0),
        gib_to_bytes(9.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect("the resident cell is admitted");
    assert_eq!(
        exempted.context.selection.strategy,
        MemoryStrategy::Resident,
        "a RESIDENT cell whose load shape disagrees must survive the filter, because \
             `optimized_eligibility` accepts it; filtering it would be stricter than the gate"
    );
    assert!(
        exempted.process_limit_bytes.is_some(),
        "the exempted resident cell must be served from its RECORD, not an estimate"
    );

    // Fourth arm: RUNG SUPPORT, load-bearing in production — on the real `qwen_image` q8 cell
    // the probe records `dropped=2 retained=0`, and the second drop is this one, not the shape
    // one. Since sc-18251 the `Implemented` requirement is enforced through the
    // `validate_selection` conjunct of `usable` (which fails for an undeclared or
    // non-`Implemented` rung before it looks at parameters); the composition conjunct also
    // drops this cell (a `Missing` rung is absent from its own live composition), so this arm
    // pins the OUTCOME — degrade, not a bare-`Missing` refusal — while each conjunct's own
    // isolating fixture lives in the sc-18251 tests below.
    //
    // Bind ONLY the deferred rung-4 cell and select under the DEFERRED contract, whose
    // `BoundedTransformerResidency` support is `Missing`. The shape now AGREES, so only the
    // new conjuncts can drop it. They must: otherwise the candidate reaches `select_strategy`,
    // which skips an unimplemented rung WITHOUT recording an exclusion, leaving every rung
    // empty and refusing with a bare `Missing` and no log line naming a cause.
    let mut rung_plan = plan.clone();
    let MlxCalibrationConfig::Valid(rung_calibration) = &mut rung_plan.calibration else {
        panic!("the fixture plan opts in to calibration");
    };
    let mut rung4_only = fixture_binding_for(
        "q4",
        "packed-q4",
        StrategyRung::BoundedTransformerResidency,
        JsonObject::from_iter([
            ("decodeTileEdge".to_owned(), serde_json::json!(512)),
            ("decodeOverlap".to_owned(), serde_json::json!(128)),
            ("attentionChunkSize".to_owned(), serde_json::json!(65536)),
            ("transformerWindowSize".to_owned(), serde_json::json!(1)),
        ]),
    );
    rung4_only.query.load_shape =
        sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
    rung_calibration.bindings = vec![rung4_only];
    assert_eq!(
        generator
            .contract
            .as_ref()
            .and_then(|contract| contract.capability(MemoryStrategy::BoundedTransformerResidency))
            .map(|capability| &capability.support),
        Some(&gen_core::MemoryStrategySupport::Missing),
        "precondition: the fixture contract does not implement rung 4"
    );
    let unsupported_rung = evaluate_request_with_budget_using_bundle(
        &generator,
        &rung_plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(64.0),
        gib_to_bytes(9.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect(
        "a measured cell on a rung the loaded contract does not implement must degrade to the \
             estimate ladder, not refuse with a bare `Missing`",
    );
    assert_eq!(
        unsupported_rung.process_limit_bytes, None,
        "nothing measured survived, so the request is served from an estimate"
    );
}

/// sc-18101: the PREMISE of the Resident exemption in the shape filter, pinned against
/// gen-core rather than restated.
///
/// The filter above skips its load-shape test for `MemoryStrategy::Resident` because
/// `optimized_eligibility` does: that gate short-circuits `Ok(())` for a non-optimized
/// selection before it ever compares load shapes, so a resident cell measured under the other
/// shape is one the gate ACCEPTS. The filter exists to anticipate the gate, not to tighten it —
/// dropping such a cell would silently discard a usable measurement, which is the failure mode
/// the per-candidate filter exists to avoid.
///
/// This test asserts the premise directly. If a pin bump ever makes gen-core reject a resident
/// cell on load shape, this reds and the exemption must be revisited — which is the only way a
/// mirrored predicate can be kept honest against a dependency it does not own.
#[test]
fn gen_core_accepts_a_resident_cell_whose_load_shape_disagrees() {
    use gen_core::MemoryCalibrationIdentity;

    let generator = fixture_generator();
    let contract = generator.contract.as_ref().expect("fixture contract");
    assert_eq!(
        contract.load_shape,
        gen_core::LoadShape::EagerMaterialization,
        "fixture precondition"
    );

    // A resident cell measured under the OTHER shape.
    let (selection, mut resident) = resident_evidence(
        contract,
        fixture_plan().tier,
        "text_to_image",
        None,
        request_geometry(&fixture_inputs(1024, 1024)),
        gib_to_bytes(4.0),
        Some("fixture-formula-v2"),
    );
    resident.key.load_shape = gen_core::LoadShape::DeferredMaterialization;
    resident.conformance = gen_core::MemoryConformanceState::Verified;
    resident.dimensions = gen_core::MemoryEvidenceDimensions::VERIFIED;
    assert!(
        !selection.strategy.is_optimized(),
        "the exemption is about the non-optimized rung"
    );
    assert_eq!(
        resident.optimized_eligibility(contract),
        Ok(()),
        "gen-core must still accept a RESIDENT cell whose load shape disagrees; the filter's \
             exemption is built on this"
    );

    // …and the same disagreement on an OPTIMIZED rung is what the gate rejects, which is what
    // the filter anticipates. Without this arm the assertion above could pass vacuously. The
    // optimized candidate is taken from the real admission route rather than hand-built, so it
    // is structurally valid in every dimension EXCEPT the one under test.
    let route = evidence_admission_route(
        &fixture_bundle(),
        &fixture_plan(),
        &fixture_inputs(1024, 1024),
        "text_to_image",
        fixture_budget(64.0),
    )
    .expect("the fixture route resolves");
    let mut optimized = route
        .evidence
        .iter()
        .map(|candidate| candidate.evidence.clone())
        .find(|evidence| evidence.key.strategy.is_optimized())
        .expect("the fixture bundle carries an optimized cell");
    assert_eq!(
        optimized.optimized_eligibility(contract),
        Ok(()),
        "precondition: the optimized cell is eligible BEFORE the shape is disturbed"
    );
    optimized.key.load_shape = gen_core::LoadShape::DeferredMaterialization;
    assert_eq!(
        optimized.optimized_eligibility(contract),
        Err(gen_core::MemoryEvidenceVerdict::FingerprintMismatch),
        "an OPTIMIZED cell with the same shape disagreement must be rejected"
    );

    // Belt and braces: the identity really does carry the shape as a third field, which is the
    // thing the worker demotion above compares only two of.
    let identity = MemoryCalibrationIdentity::new(
        "fixture-formula-v2",
        gen_core::LoadShape::EagerMaterialization,
    );
    assert_eq!(
        identity.load_shape,
        gen_core::LoadShape::EagerMaterialization
    );
}

/// sc-18251 leg 1: COMPOSITION DRIFT must degrade to the estimate ladder, never refuse — and
/// the filter must stay per-candidate, sparing a sibling whose composition still agrees.
///
/// The live provider grows a realization prerequisite that engages rung 1 in every rung-2
/// request, so the live composition for `BoundedDecode` becomes
/// `[Resident, StagedResidency, BoundedDecode]` while the record was captured under
/// `[resident, bounded_decode]`. Same shape, same abi, same fingerprint, and the captured
/// parameters still pass `validate_selection` (rung 1 owns no parameters, and the new edge is
/// satisfied by its own engagement) — the preconditions below pin that, so of the filter's
/// three legs ONLY the composition one can drop this candidate. That is what makes the
/// mutation "composition conjunct deleted" observable: the candidate then reaches
/// `select_strategy`, is excluded with `CompositionMismatch`, and the route refuses with "no
/// structurally admissible MLX memory strategy" instead of degrading — the exact pre-sc-18251
/// hole, which the retired `StaleIdentity` closure pre-demotion used to mask by demoting every
/// drifted binding to Legacy before eligibility ran.
#[test]
fn a_composition_drift_degrades_to_estimates_and_spares_agreeing_siblings() {
    use gen_core::{MemoryPrerequisiteScope, MemoryStrategyPrerequisite};

    let bundle = fixture_bundle();
    let plan = fixture_plan();
    let inputs = fixture_inputs(1024, 1024);

    let mut generator = fixture_generator();
    let contract = generator.contract.as_mut().expect("fixture contract");
    contract.additional_prerequisites.push((
        MemoryStrategy::BoundedDecode,
        MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));
    let contract = generator.contract.as_ref().expect("fixture contract");
    assert!(
        contract.conformance_errors().is_empty(),
        "the drifted contract must stay structurally conformant, or the selector refuses \
             everything for an unrelated reason and the arm passes vacuously"
    );
    // The captured composition no longer matches the live one…
    let captured = fixture_binding("q4", "packed-q4");
    assert_eq!(
        contract.engaged_composition(MemoryStrategy::BoundedDecode),
        vec![
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode
        ],
        "precondition: the realization edge moved the live rung-2 composition"
    );
    // …while the OTHER two legs still pass, so only the composition conjunct can drop it.
    let captured_selection = MemorySelection {
        strategy: MemoryStrategy::BoundedDecode,
        parameters: captured.selection_parameters,
        tier: plan.tier,
    };
    assert!(
        contract.validate_selection(&captured_selection).is_ok(),
        "precondition: the captured parameters remain valid under the drifted contract, \
             isolating the composition leg"
    );

    let degraded = evaluate_request_with_budget_using_bundle(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(64.0),
        gib_to_bytes(9.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect("a composition drift must degrade to the estimate ladder, not refuse the request");
    assert_eq!(
        degraded.process_limit_bytes, None,
        "a degraded cell must not claim a verified record's request-scoped ceiling"
    );

    // Per-candidate arm: alongside the drifted rung-2 binding, bind the bundle's resident cell
    // (whose composition `[resident]` is untouched by the rung-2 edge). The survivor must be
    // served from its RECORD — `evidence.clear()` in place of `retain(usable)` empties the
    // route and this becomes an estimate with no process ceiling.
    let mut sibling_plan = plan.clone();
    let MlxCalibrationConfig::Valid(calibration) = &mut sibling_plan.calibration else {
        panic!("the fixture plan opts in to calibration");
    };
    let mut resident_binding =
        fixture_binding_for("q4", "packed-q4", StrategyRung::Resident, JsonObject::new());
    // The bundle's resident record was captured deferred; the binding must declare the shape
    // its RECORD was captured under, and the Resident shape exemption keeps it usable.
    resident_binding.query.load_shape =
        sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
    calibration.bindings.push(resident_binding);

    let spared = evaluate_request_with_budget_using_bundle(
        &generator,
        &sibling_plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(64.0),
        gib_to_bytes(9.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect("the agreeing resident sibling is admitted");
    assert_eq!(
        spared.context.selection.strategy,
        MemoryStrategy::Resident,
        "the resident cell whose composition still agrees must survive the filter"
    );
    assert!(
        spared.process_limit_bytes.is_some(),
        "the spared sibling must be served from its RECORD, not an estimate"
    );
}

/// sc-18251 leg 2: PARAMETER-RANGE NARROWING must degrade to the estimate ladder, never
/// refuse — and the filter must stay per-candidate.
///
/// The live provider narrows its declared `decode_tile_edges` from `[512]` to `[256]`, so the
/// record captured at 512 no longer passes `contract.validate_selection`. Shape and
/// composition still agree (narrowing a range changes neither), pinned below — so only the
/// selection-validity conjunct can drop this candidate, which is what makes the mutation
/// "selection conjunct deleted" observable: the candidate then reaches `select_strategy`, is
/// excluded with `Invalid` by the same `validate_selection` call downstream, and the route
/// refuses instead of degrading.
#[test]
fn a_narrowed_parameter_range_degrades_to_estimates_and_spares_valid_siblings() {
    let bundle = fixture_bundle();
    let plan = fixture_plan();
    let inputs = fixture_inputs(1024, 1024);

    let mut generator = fixture_generator();
    let contract = generator.contract.as_mut().expect("fixture contract");
    let bounded_decode = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
        .expect("bounded decode capability");
    bounded_decode.parameters.decode_tile_edges = vec![256];
    let contract = generator.contract.as_ref().expect("fixture contract");
    assert!(
        contract.conformance_errors().is_empty(),
        "the narrowed contract must stay structurally conformant"
    );
    let captured = fixture_binding("q4", "packed-q4");
    let captured_selection = MemorySelection {
        strategy: MemoryStrategy::BoundedDecode,
        parameters: captured.selection_parameters,
        tier: plan.tier,
    };
    assert!(
        contract.validate_selection(&captured_selection).is_err(),
        "precondition: the captured tile edge fell outside the narrowed range"
    );
    assert_eq!(
        contract.engaged_composition(MemoryStrategy::BoundedDecode),
        vec![MemoryStrategy::Resident, MemoryStrategy::BoundedDecode],
        "precondition: narrowing a parameter range does not move the composition, isolating \
             the selection leg"
    );

    let degraded = evaluate_request_with_budget_using_bundle(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(64.0),
        gib_to_bytes(9.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect(
        "a parameter-range narrowing must degrade to the estimate ladder, not refuse the \
             request",
    );
    assert_eq!(
        degraded.process_limit_bytes, None,
        "a degraded cell must not claim a verified record's request-scoped ceiling"
    );

    // Per-candidate arm: the resident sibling owns no parameters, so no narrowing can strand
    // it; it must survive and be served from its record.
    let mut sibling_plan = plan.clone();
    let MlxCalibrationConfig::Valid(calibration) = &mut sibling_plan.calibration else {
        panic!("the fixture plan opts in to calibration");
    };
    let mut resident_binding =
        fixture_binding_for("q4", "packed-q4", StrategyRung::Resident, JsonObject::new());
    resident_binding.query.load_shape =
        sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
    calibration.bindings.push(resident_binding);

    let spared = evaluate_request_with_budget_using_bundle(
        &generator,
        &sibling_plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(64.0),
        gib_to_bytes(9.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect("the still-valid resident sibling is admitted");
    assert_eq!(
        spared.context.selection.strategy,
        MemoryStrategy::Resident,
        "the resident cell whose parameters remain valid must survive the filter"
    );
    assert!(
        spared.process_limit_bytes.is_some(),
        "the spared sibling must be served from its RECORD, not an estimate"
    );
}

// Exact LFS object sizes for the two control checkpoints the production FLUX router can actually
// select. The audit creates sparse files with these logical lengths so the production
// `weights_source_bytes` seam prices the real overlay without downloading it.
#[cfg(target_os = "macos")]
const FLUX1_CONTROL_BYTES: u64 = 4_281_779_224;
#[cfg(target_os = "macos")]
const FLUX2_CONTROL_BYTES: u64 = 8_232_506_680;
// The pinned Lens bf16 turnkeys share this exact three-shard MXFP4 encoder on disk. The provider
// footprint query, not a copied resident-byte constant, supplies its load-exact expansion.
#[cfg(target_os = "macos")]
const LENS_BF16_TEXT_ENCODER_DISK_BYTES: u64 = 4_845_744_456 + 4_774_186_632 + 4_154_656_824;

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ResidentOnlyAuditSurface {
    Base,
    Edit,
    StrictControl,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceBoundAuditSurface {
    manifest_id: String,
    surface: ResidentOnlyAuditSurface,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SourceBoundAuditCell {
    manifest_id: String,
    provider_id: &'static str,
    tier: String,
    surface: ResidentOnlyAuditSurface,
    base_asset_bytes: u64,
    control_bytes: u64,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceBoundShippedTier {
    tier: String,
    base_asset_bytes: u64,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SourceBoundManifestDisposition {
    GenericSelector,
    PulidIdentityExcluded,
    MageSplitComponentsExcluded,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SourceBoundManifestClassification {
    manifest_id: String,
    family: String,
    disposition: SourceBoundManifestDisposition,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SourceBoundExcludedCell {
    manifest_id: String,
    family: String,
    provider_id: &'static str,
    tier: String,
    base_asset_bytes: u64,
    disposition: SourceBoundManifestDisposition,
}

/// Story-scoped product families for the Resident-only estimate-band audit. Individual
/// manifest entries and tiers are deliberately not enumerated: the shipped manifest and
/// production routers expand this family scope into the exact candidate inventory.
#[cfg(target_os = "macos")]
const RESIDENT_ONLY_AUDIT_FAMILIES: &[&str] = &[
    "boogu",
    "chroma",
    "flux",
    "flux2-dev",
    "flux2-klein",
    "ideogram",
    "lens",
    "sd3",
];

#[cfg(target_os = "macos")]
fn source_bound_manifest_classifications(
    models: &[Value],
) -> Result<Vec<SourceBoundManifestClassification>, String> {
    use SourceBoundManifestDisposition::{
        GenericSelector, MageSplitComponentsExcluded, PulidIdentityExcluded,
    };

    let mut classifications = Vec::new();
    let mut unique = std::collections::BTreeSet::new();
    for model in models {
        if model["type"] != "image" || !model["mlx"].is_object() {
            continue;
        }
        let manifest_id = model["id"]
            .as_str()
            .ok_or_else(|| "shipped MLX image entry has no string id".to_owned())?;
        let family = model["family"]
            .as_str()
            .ok_or_else(|| format!("shipped MLX image entry {manifest_id} has no string family"))?;
        let resolved = crate::engines::mlx_model(manifest_id);
        let is_pulid = crate::image_jobs::is_pulid_flux_model(manifest_id);
        let routed_as_mage = resolved
            .as_ref()
            .is_some_and(|model| model.adapter_label() == "mlx_mage");
        let in_mage_family = family == "mage-flow";
        let in_audit_family = RESIDENT_ONLY_AUDIT_FAMILIES.contains(&family);
        if !in_audit_family && !is_pulid && !in_mage_family && !routed_as_mage {
            continue;
        }
        if !unique.insert(manifest_id) {
            return Err(format!(
                "shipped manifest declares duplicate classified MLX image id {manifest_id}"
            ));
        }
        if source_bound_shipped_tiers(model)?.is_empty() {
            return Err(format!(
                "production MLX image route {manifest_id} has no auditable shipped q4/q8/bf16 tier"
            ));
        }
        let disposition = if is_pulid {
            if family != "flux" || resolved.is_some() {
                return Err(format!(
                        "{manifest_id} PuLID bespoke classification drifted: expected family=flux and no MODEL_TABLE row"
                    ));
            }
            PulidIdentityExcluded
        } else if in_mage_family || routed_as_mage {
            if !in_mage_family || !routed_as_mage {
                return Err(format!(
                        "{manifest_id} Mage split-component classification drifted: expected family=mage-flow and a production mlx_mage route"
                    ));
            }
            MageSplitComponentsExcluded
        } else if in_audit_family {
            if resolved.is_none() {
                return Err(format!(
                        "{manifest_id} is in an audited family but has neither a production MODEL_TABLE route nor an explicit bespoke exclusion"
                    ));
            }
            GenericSelector
        } else {
            return Err(format!(
                "{manifest_id} reached the source classifier without a disposition"
            ));
        };
        classifications.push(SourceBoundManifestClassification {
            manifest_id: manifest_id.to_owned(),
            family: family.to_owned(),
            disposition,
        });
    }
    // sc-22512 (E8): an empty classification set is a legal answer, not an error. Nothing to
    // audit means nothing to audit; every check above still fires on a route that IS present
    // and misclassified.
    Ok(classifications)
}

#[cfg(target_os = "macos")]
fn source_bound_generic_manifest_ids(
    classifications: &[SourceBoundManifestClassification],
) -> impl Iterator<Item = &str> {
    classifications.iter().filter_map(|classification| {
        (classification.disposition == SourceBoundManifestDisposition::GenericSelector)
            .then_some(classification.manifest_id.as_str())
    })
}

#[cfg(target_os = "macos")]
fn source_bound_audit_surfaces_from_manifest_ids<'a>(
    manifest_ids: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<SourceBoundAuditSurface>, String> {
    use ResidentOnlyAuditSurface::{Base, Edit, StrictControl};

    let mut declared = std::collections::BTreeSet::new();
    let mut surfaces = Vec::new();
    for manifest_id in manifest_ids {
        if !declared.insert(manifest_id) {
            return Err(format!(
                "duplicate source-bound audit manifest declaration {manifest_id}"
            ));
        }
        if crate::engines::mlx_model(manifest_id).is_none() {
            return Err(format!(
                    "{manifest_id} no longer resolves through production MODEL_TABLE and the pinned registry"
                ));
        }
        surfaces.push(SourceBoundAuditSurface {
            manifest_id: manifest_id.to_owned(),
            surface: Base,
        });
        if let Some(edit_engine_id) = crate::image_jobs::flux2_edit_engine_id(manifest_id) {
            if !crate::image_jobs::flux2_edit_uses_provider_memory_safety(edit_engine_id) {
                surfaces.push(SourceBoundAuditSurface {
                    manifest_id: manifest_id.to_owned(),
                    surface: Edit,
                });
            }
        }
        if crate::image_jobs::mlx_flux_strict_control_engine_id(manifest_id).is_some() {
            surfaces.push(SourceBoundAuditSurface {
                manifest_id: manifest_id.to_owned(),
                surface: StrictControl,
            });
        }
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
fn source_bound_audit_surfaces(models: &[Value]) -> Result<Vec<SourceBoundAuditSurface>, String> {
    let classifications = source_bound_manifest_classifications(models)?;
    source_bound_audit_surfaces_from_manifest_ids(source_bound_generic_manifest_ids(
        &classifications,
    ))
}

/// Build the exact FLUX.2 encoder/config/tokenizer admission surface used by the source-bound
/// audit without loading a tensor. Klein's Qwen3 stays dense across every DiT tier; Dev follows
/// the selected tier and its base route also retains the builtin Pixtral vision surface.
#[cfg(target_os = "macos")]
fn write_audited_flux2_encoder_fixture(
    snapshot_root: &Path,
    provider_id: &str,
    tier: &str,
) -> Result<Option<u64>, String> {
    let (quant_bits, include_multimodal) = match provider_id {
        "flux2_klein_9b" | "flux2_klein_9b_edit" | "flux2_klein_9b_kv_edit" => (None, false),
        "flux2_dev" | "flux2_dev_edit" => (
            match tier {
                "q4" => Some(4),
                "q8" => Some(8),
                "bf16" => None,
                other => return Err(format!("unsupported audited FLUX.2 tier {other}")),
            },
            true,
        ),
        "flux2_dev_control" => (
            match tier {
                "q4" => Some(4),
                "q8" => Some(8),
                "bf16" => None,
                other => return Err(format!("unsupported audited FLUX.2 tier {other}")),
            },
            false,
        ),
        _ => return Ok(None),
    };
    let contract = crate::inference_runtime::media_encoder_contract(provider_id)
        .ok_or_else(|| format!("{provider_id} owns an encoder contract"))?;
    let encoder_root = snapshot_root.join("text_encoder");
    let result = if include_multimodal {
        gen_core_testkit::write_multimodal_encoder_contract_fixture_with_quant(
            &encoder_root,
            contract,
            runtime_macos::providers::flux2::config::DEV_VISION_ENCODER_CONTRACT,
            quant_bits,
        )
    } else {
        gen_core_testkit::write_encoder_contract_fixture_with_quant(
            &encoder_root,
            contract,
            quant_bits,
        )
    };
    result.map_err(|error| {
        format!("write registry-owned {provider_id} {tier} encoder fixture: {error}")
    })?;
    if include_multimodal {
        if let Some(bits) = quant_bits {
            // Mistral3 stores the language model under `text_config`, but selected-encoder
            // admission deliberately accepts packing evidence only from the authoritative
            // root marker. Keep both views aligned, as converted Dev snapshots do.
            let config_path = encoder_root.join("config.json");
            let mut config: Value = serde_json::from_slice(
                &std::fs::read(&config_path).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            let group_size = contract
                .packing
                .ok_or_else(|| {
                    format!("{provider_id} packed fixture has no encoder packing contract")
                })?
                .group_size;
            config["quantization"] = serde_json::json!({"bits": bits, "group_size": group_size});
            std::fs::write(
                config_path,
                serde_json::to_vec(&config).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
        }
    }
    Ok(Some(sum_safetensors_bytes(&encoder_root)))
}

#[cfg(target_os = "macos")]
fn source_bound_shipped_tiers(model: &Value) -> Result<Vec<SourceBoundShippedTier>, String> {
    let downloads = model["downloads"].as_array().ok_or_else(|| {
        format!(
            "{} has no shipped downloads",
            model["id"].as_str().unwrap_or("<unknown>")
        )
    })?;
    fn supports_macos(download: &Value) -> bool {
        download["platforms"].as_array().is_none_or(|platforms| {
            platforms
                .iter()
                .any(|platform| platform.as_str() == Some("macos"))
        })
    }
    fn supported_variant(download: &Value) -> Option<&str> {
        download["variant"]
            .as_str()
            .and_then(|variant| matches!(variant, "q4" | "q8" | "bf16").then_some(variant))
    }
    let has_explicit_tiers = downloads
        .iter()
        .any(|download| supports_macos(download) && supported_variant(download).is_some());
    let inferred_tier = match model["mlx"]["quantize"].as_u64() {
        Some(4) => "q4",
        Some(8) => "q8",
        None | Some(0) => "bf16",
        Some(other) => {
            return Err(format!(
                "{} has unsupported shipped mlx.quantize tier {other}",
                model["id"].as_str().unwrap_or("<unknown>")
            ));
        }
    };
    let mut tiers = std::collections::BTreeMap::<String, u64>::new();
    for download in downloads {
        if !supports_macos(download) {
            continue;
        }
        let tier = if has_explicit_tiers {
            let Some(tier) = supported_variant(download) else {
                continue;
            };
            tier
        } else if download["variant"].as_str() == Some("training") {
            continue;
        } else {
            inferred_tier
        };
        let bytes = download["estimatedSizeBytes"]
            .as_u64()
            .or_else(|| download["footprint"]["diskSizeBytes"].as_u64())
            .ok_or_else(|| {
                format!(
                    "{} {tier} needs an auditable shipped byte size",
                    model["id"].as_str().unwrap_or("<unknown>")
                )
            })?;
        let total = tiers.entry(tier.to_owned()).or_default();
        *total = total.checked_add(bytes).ok_or_else(|| {
            format!(
                "{} {tier} shipped byte total overflows u64",
                model["id"].as_str().unwrap_or("<unknown>")
            )
        })?;
    }
    Ok(tiers
        .into_iter()
        .map(|(tier, base_asset_bytes)| SourceBoundShippedTier {
            tier,
            base_asset_bytes,
        })
        .collect())
}

/// Expand every explicitly excluded manifest entry into the same route/tier accounting shape
/// as the generic-selector audit. These are exclusions from the generic sparse-`LoadSpec`
/// fixture, not exclusions from source-truth accounting: every shipped tier remains pinned.
#[cfg(target_os = "macos")]
fn source_bound_excluded_inventory_from_classifications(
    models: &[Value],
    classifications: &[SourceBoundManifestClassification],
) -> Result<Vec<SourceBoundExcludedCell>, String> {
    use SourceBoundManifestDisposition::{
        GenericSelector, MageSplitComponentsExcluded, PulidIdentityExcluded,
    };

    let mut cells = std::collections::BTreeSet::new();
    for classification in classifications {
        if classification.disposition == GenericSelector {
            continue;
        }
        let model = models
            .iter()
            .find(|model| model["id"] == classification.manifest_id)
            .ok_or_else(|| {
                format!(
                    "missing explicitly excluded {} manifest entry",
                    classification.manifest_id
                )
            })?;
        let provider_id = match classification.disposition {
            PulidIdentityExcluded => {
                if !crate::image_jobs::is_pulid_flux_model(&classification.manifest_id)
                    || classification.family != "flux"
                    || crate::engines::mlx_model(&classification.manifest_id).is_some()
                {
                    return Err(format!(
                        "{} no longer matches the PuLID identity-path exclusion",
                        classification.manifest_id
                    ));
                }
                "pulid_flux"
            }
            MageSplitComponentsExcluded => {
                let resolved =
                    crate::engines::mlx_model(&classification.manifest_id).ok_or_else(|| {
                        format!(
                            "{} no longer resolves through the production Mage route",
                            classification.manifest_id
                        )
                    })?;
                if classification.family != "mage-flow" || resolved.adapter_label() != "mlx_mage" {
                    return Err(format!(
                        "{} no longer matches the Mage split-component exclusion",
                        classification.manifest_id
                    ));
                }
                resolved.engine_id()
            }
            GenericSelector => unreachable!("generic entries were skipped above"),
        };
        if crate::inference_runtime::media_descriptor(provider_id).is_none() {
            return Err(format!(
                "explicitly excluded provider {provider_id} is absent from the pinned registry"
            ));
        }
        for shipped in source_bound_shipped_tiers(model)? {
            let cell = SourceBoundExcludedCell {
                manifest_id: classification.manifest_id.clone(),
                family: classification.family.clone(),
                provider_id,
                tier: shipped.tier,
                base_asset_bytes: shipped.base_asset_bytes,
                disposition: classification.disposition,
            };
            if !cells.insert(cell.clone()) {
                return Err(format!(
                    "source-bound excluded inventory resolves a duplicate cell {cell:?}"
                ));
            }
        }
    }
    Ok(cells.into_iter().collect())
}

/// Revalidate each exclusion against current production routing and current manifest tier
/// accounting. Exact equality with the source-derived inventory below prevents a caller from
/// silently dropping an excluded model or tier before this classifier runs.
#[cfg(target_os = "macos")]
fn source_bound_classified_excluded_inventory(
    models: &[Value],
    source_inventory: &[SourceBoundExcludedCell],
) -> Result<Vec<SourceBoundExcludedCell>, String> {
    let classifications = source_bound_manifest_classifications(models)?;
    let canonical = source_bound_excluded_inventory_from_classifications(models, &classifications)?
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let mut classified = Vec::new();
    for candidate in source_inventory {
        if !canonical.contains(candidate) {
            return Err(format!(
                    "source-bound exclusion no longer matches production source/routing/accounting: {candidate:?}"
                ));
        }
        classified.push(candidate.clone());
    }
    Ok(classified)
}

#[cfg(target_os = "macos")]
fn require_exact_source_bound_excluded_inventory(
    expected: &[SourceBoundExcludedCell],
    actual: &[SourceBoundExcludedCell],
) -> Result<(), String> {
    let expected = expected
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let actual = actual
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    if expected == actual {
        return Ok(());
    }
    let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
    let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
    Err(format!(
        "source-bound excluded inventory mismatch; missing={missing:?}; unexpected={unexpected:?}"
    ))
}

#[cfg(target_os = "macos")]
fn source_bound_audit_provider(surface: &SourceBoundAuditSurface) -> Result<&'static str, String> {
    match surface.surface {
        ResidentOnlyAuditSurface::Base => crate::engines::mlx_model(&surface.manifest_id)
            .map(|resolved| resolved.engine_id())
            .ok_or_else(|| {
                format!(
                    "{} no longer resolves through production MODEL_TABLE and the pinned registry",
                    surface.manifest_id
                )
            }),
        ResidentOnlyAuditSurface::Edit => {
            crate::image_jobs::flux2_edit_engine_id(&surface.manifest_id).ok_or_else(|| {
                format!(
                    "{} has no production FLUX.2 edit route",
                    surface.manifest_id
                )
            })
        }
        ResidentOnlyAuditSurface::StrictControl => {
            crate::image_jobs::mlx_flux_strict_control_engine_id(&surface.manifest_id).ok_or_else(
                || {
                    format!(
                        "{} has no production FLUX strict-control route",
                        surface.manifest_id
                    )
                },
            )
        }
    }
}

#[cfg(target_os = "macos")]
fn source_bound_control_bytes(
    surface: &SourceBoundAuditSurface,
    provider_id: &str,
) -> Result<u64, String> {
    if surface.surface != ResidentOnlyAuditSurface::StrictControl {
        return Ok(0);
    }
    match provider_id {
        "flux1_dev_control" => Ok(FLUX1_CONTROL_BYTES),
        "flux2_dev_control" => Ok(FLUX2_CONTROL_BYTES),
        other => Err(format!(
            "production strict-control provider {other} has no audited control checkpoint size"
        )),
    }
}

#[cfg(target_os = "macos")]
fn audit_load_spec(
    provider_id: &'static str,
    tier: &str,
    base_asset_bytes: u64,
    control_bytes: u64,
) -> Result<(std::sync::Arc<tempfile::TempDir>, LoadSpec), String> {
    use crate::test_fixture_disk::{create_sparse_weights, set_sparse_valid_safetensor};
    use gen_core::Quant;

    type FixtureKey = (String, String, u64, u64);
    type Fixture = (std::sync::Arc<tempfile::TempDir>, LoadSpec);
    struct Fixtures {
        root: tempfile::TempDir,
        specs: std::sync::Mutex<std::collections::BTreeMap<FixtureKey, Fixture>>,
    }
    // Only the read-only source-bound audits use this pool. Reuse the FILES, never a
    // classified contract or selector result: each walk still calls the production lookup,
    // including ArtifactSeal. Stable file identities let its ordinary digest memo avoid
    // hashing the same multi-GiB sparse encoder again for every equivalent walk. A mutation
    // test must build its own private fixture, rather than write into this shared pool.
    // Like the Krea memory fixtures, static guards cannot clean themselves up at exit, so
    // use the existing process-keyed, age-bounded cleanup discipline for this family.
    const FAMILY: &str = "sceneworks-resident-audit-";
    static FIXTURES: std::sync::OnceLock<Result<Fixtures, String>> = std::sync::OnceLock::new();
    let fixtures = FIXTURES
        .get_or_init(|| {
            crate::test_fixture_disk::sweep_stale_fixtures(FAMILY);
            let root = tempfile::Builder::new()
                .prefix(&crate::test_fixture_disk::process_keyed_prefix(FAMILY))
                .tempdir()
                .map_err(|error| error.to_string())?;
            Ok(Fixtures {
                root,
                specs: std::sync::Mutex::new(std::collections::BTreeMap::new()),
            })
        })
        .as_ref()
        .map_err(Clone::clone)?;
    let key = (
        provider_id.to_owned(),
        tier.to_owned(),
        base_asset_bytes,
        control_bytes,
    );
    let mut specs = fixtures.specs.lock().map_err(|error| error.to_string())?;
    if let Some(fixture) = specs.get(&key) {
        return Ok(fixture.clone());
    }
    let fixture = std::sync::Arc::new(
        tempfile::tempdir_in(fixtures.root.path()).map_err(|error| error.to_string())?,
    );
    let weights = fixture.path().join("weights");
    if provider_id == "lens" && tier == "q4" {
        // The production worker opts this one route into the load-exact deferred contract.
        // Build the same component/config shape its registry contract inspects; a flat sparse
        // file would make the lookup fall back and falsely classify measured Lens q4 as
        // Resident-only.
        let text_encoder_bytes = base_asset_bytes / 3;
        let transformer_bytes = base_asset_bytes / 3;
        let vae_bytes = base_asset_bytes - text_encoder_bytes - transformer_bytes;
        for (component, bytes) in [
            ("text_encoder", text_encoder_bytes),
            ("transformer", transformer_bytes),
            ("vae", vae_bytes),
        ] {
            set_sparse_valid_safetensor(&weights.join(component).join("model.safetensors"), bytes)?;
        }
        for component in ["text_encoder", "transformer"] {
            std::fs::write(
                weights.join(component).join("config.json"),
                r#"{"quantization":{"bits":4,"group_size":64}}"#,
            )
            .map_err(|error| error.to_string())?;
        }
    } else if matches!(provider_id, "lens" | "lens_turbo") && tier == "bf16" {
        if base_asset_bytes < LENS_BF16_TEXT_ENCODER_DISK_BYTES {
            return Err(format!(
                "Lens bf16 asset {base_asset_bytes} is smaller than its source encoder"
            ));
        }
        set_sparse_valid_safetensor(
            &weights.join("text_encoder/model.safetensors"),
            LENS_BF16_TEXT_ENCODER_DISK_BYTES,
        )?;
        set_sparse_valid_safetensor(
            &weights.join("transformer/model.safetensors"),
            base_asset_bytes - LENS_BF16_TEXT_ENCODER_DISK_BYTES,
        )?;
        std::fs::write(
            weights.join("text_encoder/config.json"),
            r#"{"quantization_config":{"quant_method":"mxfp4"}}"#,
        )
        .map_err(|error| error.to_string())?;
    } else if let Some(encoder_bytes) =
        write_audited_flux2_encoder_fixture(&weights, provider_id, tier)?
    {
        let remaining_bytes = base_asset_bytes.checked_sub(encoder_bytes).ok_or_else(|| {
            format!(
                "{provider_id} {tier} registry encoder fixture uses {encoder_bytes} bytes, \
                    above the shipped base total {base_asset_bytes}"
            )
        })?;
        let transformer_bytes = remaining_bytes.saturating_mul(9) / 10;
        let vae_bytes = remaining_bytes - transformer_bytes;
        set_sparse_valid_safetensor(
            &weights.join("transformer/model.safetensors"),
            transformer_bytes,
        )?;
        set_sparse_valid_safetensor(&weights.join("vae/model.safetensors"), vae_bytes)?;
    } else {
        create_sparse_weights(&weights.join("model.safetensors"), base_asset_bytes);
    }
    let spec = match tier {
        "q4" => LoadSpec::new(WeightsSource::Dir(weights)).with_quant(Quant::Q4),
        "q8" => LoadSpec::new(WeightsSource::Dir(weights)).with_quant(Quant::Q8),
        "bf16" => LoadSpec::new(WeightsSource::Dir(weights)),
        other => return Err(format!("unsupported audited tier {other}")),
    };
    let spec = if control_bytes == 0 {
        spec
    } else {
        let control = fixture.path().join("control.safetensors");
        create_sparse_weights(&control, control_bytes);
        spec.with_control(WeightsSource::File(control))
    };
    let fixture = (
        fixture,
        crate::image_jobs::apply_measured_mlx_load_shape(provider_id, spec),
    );
    // Publish only after the real lookup has sealed this immutable source. Otherwise parallel
    // tests can race their FIRST pin and redundantly hash the same encoder before the digest
    // memo is populated. Discard the result: each audit still derives its own contract below.
    source_bound_contract(provider_id, &fixture.1)?;
    specs.insert(key, fixture.clone());
    Ok(fixture)
}

#[cfg(target_os = "macos")]
#[test]
fn resident_only_audit_fixture_reuse_preserves_every_load_coordinate() {
    let (first, spec) = audit_load_spec("flux1_dev", "q4", 4096, 0).unwrap();
    let (same, _) = audit_load_spec("flux1_dev", "q4", 4096, 0).unwrap();
    assert!(std::sync::Arc::ptr_eq(&first, &same));
    assert_eq!(weights_source_bytes(&spec.weights), 4096);
    for (provider, tier, base, control) in [
        ("flux1_schnell", "q4", 4096, 0),
        ("flux1_dev", "q8", 4096, 0),
        ("flux1_dev", "q4", 8192, 0),
    ] {
        let (distinct, spec) = audit_load_spec(provider, tier, base, control).unwrap();
        assert_ne!(first.path(), distinct.path());
        assert_eq!(weights_source_bytes(&spec.weights), base);
        assert_eq!(
            spec.control.as_ref().map_or(0, weights_source_bytes),
            control
        );
    }
    let (control_a, _) = audit_load_spec("flux1_dev_control", "q4", 4096, 1024).unwrap();
    let (control_b, spec) = audit_load_spec("flux1_dev_control", "q4", 4096, 2048).unwrap();
    assert_ne!(control_a.path(), control_b.path());
    assert_eq!(spec.control.as_ref().map_or(0, weights_source_bytes), 2048);
    // Mutation coverage uses a PRIVATE copy. Sealing a reused source must still compare real
    // contents, and neither a changed inode's bytes nor its refused old seal may poison the
    // immutable source borrowed by another audit.
    let shared_path = first.path().join("weights/model.safetensors");
    let shared_seal = gen_core::ArtifactSeal::pin(&shared_path).unwrap();
    let private = tempfile::tempdir().unwrap();
    let private_path = private.path().join("model.safetensors");
    std::fs::copy(&shared_path, &private_path).unwrap();
    let before = gen_core::ArtifactSeal::pin(&private_path).unwrap();
    std::fs::write(&private_path, vec![1_u8; 4096]).unwrap();
    assert!(before.verify_unchanged().is_err());
    let after = gen_core::ArtifactSeal::pin(&private_path).unwrap();
    assert_ne!(before.content_sha256(), after.content_sha256());
    shared_seal.verify_unchanged().unwrap();
}

#[cfg(target_os = "macos")]
fn source_bound_contract(
    provider_id: &'static str,
    spec: &LoadSpec,
) -> Result<(MemoryProviderContract, bool), String> {
    crate::inference_runtime::media()
        .memory_strategy_contract(provider_id, spec)
        .map_err(|error| error.to_string())
        .map(|contract| {
            contract.map_or_else(
                || {
                    (
                        MemoryProviderContract::compatibility_default(
                            provider_id,
                            MemoryBackendRealization::MlxMetal {
                                bounded_wired_residency: true,
                                lazy_or_mmap_materialization: true,
                                explicit_evaluation_and_synchronization: true,
                                cache_eviction: true,
                            },
                        ),
                        false,
                    )
                },
                |contract| (contract, true),
            )
        })
}

#[cfg(target_os = "macos")]
fn implemented_optimized_strategies(contract: &MemoryProviderContract) -> Vec<MemoryStrategy> {
    MemoryStrategy::ALL
        .into_iter()
        .filter(|strategy| strategy.is_optimized())
        .filter(|strategy| {
            matches!(
                contract
                    .capability(*strategy)
                    .map(|capability| &capability.support),
                Some(gen_core::MemoryStrategySupport::Implemented)
            )
        })
        .collect()
}

/// The shipped manifest is the product-side binding of provider capability to a concrete tier.
/// Provider contract construction may inspect the on-disk component tree, which this audit
/// represents with sparse total-size files; use the manifest's explicit optimized-rung tier
/// declaration to prevent that deliberately minimal filesystem fixture from turning an
/// optimized clean route (Chroma/FLUX/Klein) into a fake Resident-only cell. A manifest may now
/// also declare an exhaustive Resident-only contract (FLUX.2 q4/q8, SC-18218); that is evidence
/// that the tier belongs in this audit, not a reason to exclude it.
#[cfg(target_os = "macos")]
fn manifest_declares_optimized_tier(model: &Value, tier: &str) -> bool {
    model["mlx"]["memoryStrategyContract"]["implementations"]
        .as_array()
        .is_some_and(|implementations| {
            implementations.iter().any(|implementation| {
                implementation["rung"]
                    .as_str()
                    .is_some_and(|rung| rung != "resident")
                    && implementation["tiers"]
                        .as_array()
                        .is_some_and(|tiers| tiers.iter().any(|item| item.as_str() == Some(tier)))
            })
        })
}

#[cfg(target_os = "macos")]
fn source_bound_audit_inventory_from_surfaces(
    models: &[Value],
    source_surfaces: &[SourceBoundAuditSurface],
) -> Result<Vec<SourceBoundAuditCell>, String> {
    let mut surfaces = std::collections::BTreeSet::new();
    let mut cells = std::collections::BTreeSet::new();
    let mut cell_keys = std::collections::BTreeSet::new();
    for expected in source_surfaces {
        let model = models
            .iter()
            .find(|model| model["id"] == expected.manifest_id)
            .ok_or_else(|| format!("missing shipped {} manifest entry", expected.manifest_id))?;
        if model["type"] != "image" {
            return Err(format!(
                "{} is no longer a shipped image model",
                expected.manifest_id
            ));
        }
        let provider_id = source_bound_audit_provider(expected)?;
        let control_bytes = source_bound_control_bytes(expected, provider_id)?;
        if !surfaces.insert((expected.manifest_id.clone(), expected.surface, provider_id)) {
            return Err(format!(
                    "source-bound resident-only inventory resolves a duplicate {:?} route {} ({provider_id})",
                    expected.surface, expected.manifest_id
                ));
        }
        if crate::inference_runtime::media_descriptor(provider_id).is_none() {
            return Err(format!(
                    "source-bound resident-only provider {provider_id} is absent from the pinned registry"
                ));
        }
        for shipped in source_bound_shipped_tiers(model)? {
            let tier = shipped.tier.as_str();
            let base_asset_bytes = shipped.base_asset_bytes;
            let key = (
                expected.manifest_id.to_owned(),
                provider_id,
                tier.to_owned(),
                expected.surface,
            );
            if !cell_keys.insert(key) {
                return Err(format!(
                        "source-bound resident-only inventory resolves a duplicate cell {} ({provider_id}) {tier} {:?}",
                        expected.manifest_id, expected.surface
                    ));
            }
            cells.insert(SourceBoundAuditCell {
                manifest_id: expected.manifest_id.to_owned(),
                provider_id,
                tier: tier.to_owned(),
                surface: expected.surface,
                base_asset_bytes,
                control_bytes,
            });
        }
    }
    Ok(cells.into_iter().collect())
}

#[cfg(target_os = "macos")]
fn require_exact_source_bound_inventory(
    expected: &[SourceBoundAuditCell],
    actual: &[SourceBoundAuditCell],
) -> Result<(), String> {
    let expected = expected
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let actual = actual
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    if expected == actual {
        return Ok(());
    }
    let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
    let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
    Err(format!(
        "source-bound audit inventory mismatch; missing={missing:?}; unexpected={unexpected:?}"
    ))
}

#[cfg(target_os = "macos")]
fn source_bound_resident_only_cells_from_inventory(
    models: &[Value],
    source_inventory: &[SourceBoundAuditCell],
) -> Result<(Vec<SourceBoundAuditCell>, Vec<SourceBoundAuditCell>), String> {
    let mut classified = Vec::new();
    let mut cells = Vec::new();
    for candidate in source_inventory {
        let model = models
            .iter()
            .find(|model| model["id"] == candidate.manifest_id)
            .ok_or_else(|| format!("missing shipped {} manifest entry", candidate.manifest_id))?;
        if candidate.surface != ResidentOnlyAuditSurface::StrictControl
            && manifest_declares_optimized_tier(model, &candidate.tier)
        {
            classified.push(candidate.clone());
            continue;
        }
        let (_fixture, spec) = audit_load_spec(
            candidate.provider_id,
            &candidate.tier,
            candidate.base_asset_bytes,
            candidate.control_bytes,
        )?;
        let (contract, _) =
            source_bound_contract(candidate.provider_id, &spec).map_err(|error| {
                format!(
                    "{} ({}) {} {:?} contract lookup failed: {error}",
                    candidate.manifest_id, candidate.provider_id, candidate.tier, candidate.surface
                )
            })?;
        if implemented_optimized_strategies(&contract).is_empty() {
            cells.push(candidate.clone());
        }
        classified.push(candidate.clone());
    }
    Ok((classified, cells))
}

/// sc-18251 resident-only audit: derive every reachable candidate through the production base,
/// edit, and FLUX strict-control routers, retain only tiers whose pinned loaded-provider contract
/// is actually Resident-only, then drive those exact cells through the production request plan and
/// selector. A base provider never receives an invented control checkpoint.
#[cfg(target_os = "macos")]
#[test]
fn shipped_resident_only_mlx_estimate_band_audit_uses_the_production_budget_path() {
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let models = manifest["models"].as_array().expect("manifest models");
    let classifications =
        source_bound_manifest_classifications(models).expect("source-bound classifications");
    assert_eq!(
        classifications.len(),
        26,
        "the source-derived generic-plus-excluded model inventory changed"
    );
    let generic_manifest_ids =
        source_bound_generic_manifest_ids(&classifications).collect::<Vec<_>>();
    assert_eq!(
        generic_manifest_ids.len(),
        19,
        "the generic-selector model inventory changed"
    );
    let surfaces = source_bound_audit_surfaces_from_manifest_ids(generic_manifest_ids)
        .expect("source-bound audit surfaces");
    let source_inventory = source_bound_audit_inventory_from_surfaces(models, &surfaces)
        .expect("source-bound candidate inventory");
    assert_eq!(
        source_inventory.len(),
        62,
        "the source-derived candidate inventory changed; update the recorded audit result"
    );
    let excluded_inventory =
        source_bound_excluded_inventory_from_classifications(models, &classifications)
            .expect("source-bound excluded inventory");
    let pulid_excluded = excluded_inventory
        .iter()
        .filter(|cell| cell.disposition == SourceBoundManifestDisposition::PulidIdentityExcluded)
        .count();
    let mage_excluded = excluded_inventory
        .iter()
        .filter(|cell| {
            cell.disposition == SourceBoundManifestDisposition::MageSplitComponentsExcluded
        })
        .count();
    assert_eq!(pulid_excluded, 3, "PuLID must account for q4/q8/bf16");
    assert_eq!(
        mage_excluded, 18,
        "the six Mage variants must each account for q4/q8/bf16"
    );
    assert_eq!(
        source_inventory.len() + pulid_excluded,
        65,
        "every route/tier cell in the eight-family story scope must be classified"
    );
    let classified_excluded =
        source_bound_classified_excluded_inventory(models, &excluded_inventory)
            .expect("classified source-bound exclusions");
    require_exact_source_bound_excluded_inventory(&excluded_inventory, &classified_excluded)
        .expect("the executable audit must classify the exact excluded inventory");
    let (classified_inventory, cells) =
        source_bound_resident_only_cells_from_inventory(models, &source_inventory)
            .expect("source-bound resident-only inventory");
    require_exact_source_bound_inventory(&source_inventory, &classified_inventory)
        .expect("the executable audit must classify the exact source-bound candidate inventory");
    // Resident-only means "this cell has no ladder to fall back to", so this population SHRINKS
    // monotonically as ladders are declared and made reachable: the pin advance brought the
    // Bernini, Lens/Lens-Turbo, SANA and SD3.5 rung-4 ladders (sc-18609, sc-18605, sc-18607,
    // sc-18606), and the SC-18460 contract chain declares more still. A RISE is the alarming
    // direction and the only thing this recorded result exists to catch. Asserting the exact
    // count instead froze the corpus: every re-capture and every newly declared ladder made a
    // green branch red for the right reason and got "fixed" by editing the integer. So bound it
    // — a ceiling that legitimately falls, plus the vacuity floor that keeps the budget walk
    // below from passing on an empty inventory.
    const RESIDENT_ONLY_CEILING: usize = 18;
    assert!(
        !cells.is_empty() && cells.len() <= RESIDENT_ONLY_CEILING,
        "the source-derived Resident-only inventory is {} cells; it must stay non-empty and \
             within the recorded ceiling of {RESIDENT_ONLY_CEILING}. A rise means a cell lost a \
             reachable ladder — fix the declaration, do not raise the ceiling.",
        cells.len()
    );
    for route in [
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_klein_9b_true_v2",
    ] {
        assert!(
            cells.iter().all(|cell| cell.manifest_id != route),
            "{route}'s exact optimized declaration must remove it from the Resident-only audit"
        );
    }
    let legacy_reserve_bytes = gib_to_bytes(crate::fit_gate::legacy_unified_reserve(48.0).gb);
    let hosts = [48_u64, 64, 96, 128];
    let mut flips = Vec::new();
    let mut audited_cells = std::collections::BTreeSet::new();

    for cell in &cells {
        assert!(
            audited_cells.insert((
                cell.manifest_id.clone(),
                cell.provider_id,
                cell.tier.clone(),
                cell.surface,
            )),
            "the executable audit must reject duplicate route-tier cells: {cell:?}"
        );
        let (_fixture, spec) = audit_load_spec(
            cell.provider_id,
            &cell.tier,
            cell.base_asset_bytes,
            cell.control_bytes,
        )
        .expect("source-bound audit spec");
        let media = crate::inference_runtime::media();
        assert!(
                !matches!(cell.provider_id, "lens" | "lens_turbo"),
                "Lens materialization can change the resident total and must not enter the raw-source audit"
            );
        let activation_anchor_bytes = media
            .activation_memory_bytes_1024(cell.provider_id)
            .expect("source-bound activation query");
        // Every audited spec is directory-backed, and for these non-Lens providers the
        // provider footprint can only refine the text-encoder split; it cannot change the
        // total used by the Resident path below. Let the production planner derive that total
        // from the sparse source sizes directly. Asking the provider registry for a redundant
        // split would correctly content-pin and SHA-256 these synthetic multi-GiB sparse files,
        // turning a metadata-only budget audit into tens of minutes of hashing.
        let plan = MlxRequestPlan::for_spec_and_manifest_with_provider_facts(
            cell.provider_id,
            &cell.manifest_id,
            &spec,
            None,
            None,
            None,
            activation_anchor_bytes,
        );
        assert_eq!(
            plan.folded_control_bytes, cell.control_bytes,
            "{} ({}) {}: deleting production control-source accounting must make this \
                     audit red",
            cell.manifest_id, cell.provider_id, cell.tier
        );
        assert!(
            plan.asset_bytes >= cell.base_asset_bytes + cell.control_bytes,
            "{} ({}) {}: provider materialization cannot erase shipped base/control bytes",
            cell.manifest_id,
            cell.provider_id,
            cell.tier
        );
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let raw_incremental_peak = plan.generic_headroom_bytes(geometry);
        let widened_incremental_peak = crate::memory_strategy::floor_admitted_peak_bytes(
            gen_core::MemoryBackend::Mlx,
            raw_incremental_peak,
            None,
        );
        // These are precisely the cells for which no optimized product contract applies. Model
        // their legacy Resident fallback with the same compatibility contract production uses
        // when a provider has no applicable adopted cell, and bind its aggregate base fact to
        // the source-derived load plan so post-load cache credit is exact.
        let mut contract = MemoryProviderContract::compatibility_default(
            cell.provider_id,
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        );
        contract.asset_facts.base_bytes = plan.asset_bytes;
        contract.asset_facts.transformer_bytes = plan.asset_bytes;
        assert!(
            contract.conformance_errors().is_empty(),
            "audit contract must stay conformant for {cell:?}: {:?}",
            contract.conformance_errors()
        );
        let generator = RequestGenerator {
            descriptor: crate::inference_runtime::media_descriptor(cell.provider_id)
                .expect("source-bound provider descriptor"),
            contract: Some(contract),
        };
        let legacy_total_peak = plan.generic_total_peak_bytes(geometry);

        for host_gib in hosts {
            let host_bytes = gib_to_bytes(host_gib as f64);
            // This is exactly the live legacy budget after the generator has loaded:
            // Express the full-pipeline production comparison in its equivalent incremental
            // form. The resident allowance still applies only to the incremental peak.
            let available_incremental = host_bytes
                .saturating_sub(plan.asset_bytes)
                .saturating_sub(legacy_reserve_bytes);
            let admitted_before_margin = raw_incremental_peak <= available_incremental;
            let admitted_now = widened_incremental_peak <= available_incremental;
            if admitted_before_margin && !admitted_now {
                flips.push((
                    cell.manifest_id.as_str(),
                    cell.provider_id,
                    cell.tier.as_str(),
                    host_gib,
                    plan.asset_bytes,
                    raw_incremental_peak,
                    widened_incremental_peak,
                ));
            }

            // Exercise the production selector seam too; the arithmetic above names the
            // historical no-margin counterfactual, while this call proves the current side
            // uses cache credit + reserve + EstimateFloor margin together.
            let evaluated = evaluate_request_with_budget(
                &generator,
                &plan,
                &fixture_inputs(1024, 1024),
                MemoryCacheState::Cold,
                OffloadPolicy::Resident,
                MemoryBudget {
                    total_bytes: host_bytes,
                    committed_bytes: plan.asset_bytes,
                    reclaimable_bytes: 0,
                    reserved_headroom_bytes: legacy_reserve_bytes,
                },
                legacy_total_peak,
                0,
                &[],
            );
            assert_eq!(
                evaluated.is_ok(),
                admitted_now,
                "production-path result drifted for {} ({}) {} on {host_gib} GiB: \
                         {evaluated:?}",
                cell.manifest_id,
                cell.provider_id,
                cell.tier
            );
        }
    }

    assert_eq!(
        audited_cells.len(),
        cells.len(),
        "the executable budget walk must cover every unique source-derived cell exactly"
    );

    assert_eq!(
        flips
            .iter()
            .map(|(model, provider, tier, host, ..)| (*model, *provider, *tier, *host))
            .collect::<Vec<_>>(),
        // The Lens and SD3.5 rows are gone with the inference pin advance, and for the same
        // reason the Resident-only inventory above shrank 32 -> 18: sc-18605 and sc-18606 gave
        // those providers reachable rung-4 ladders, so they are no longer Resident-only and
        // have no estimate band left to flip. sc-20799 removed the fourth row the same way:
        // retiring `flux2_dev`'s `exhaustive` reading gave its T2I route a declared
        // `staged_residency` rung across bf16/q4/q8, so the bf16 cell is no longer
        // Resident-only and has no band to flip. sc-22508 narrowed the band itself and emptied
        // the list.
        //
        // The blanket 50.41%-of-the-whole-peak widening is gone. What remains for the legacy
        // RESIDENT BASELINE audited here is the policy's undeclared-floor arm — the same-cell
        // recapture spread, 12.6% of the peak — because this candidate's peak reaches the
        // selector through `predicted_memory_peak_from_base` (and, on `mage_flow`, through the
        // provider's own `generation_peak_gb`) and is not guaranteed to decompose into counted
        // weights plus an activation term. `flux2_dev` q4 at 64 GiB and both `ideogram_4` and
        // `ideogram_4_turbo` q8 at 48 GiB now admit at every audited host size, so no cell has
        // a band left to flip.
        //
        // An EMPTY list is a real audited result here, not a disabled check: the walk above
        // still asserts coverage of every source-derived cell, and `admitted_now` is still
        // compared against the production selector for each one. A future manifest or policy
        // change that reopens a band reds this list rather than passing silently.
        Vec::<(&str, &str, &str, u64)>::new(),
        "the source-bound resident-only audit changed; update the recorded result, \
             not only this expectation: {flips:?}"
    );
}

/// Mutation proof for the source-truth failure found in the prior audit: no control checkpoint
/// may be injected into Chroma, FLUX.1 Schnell, or FLUX.2 Klein, while the two real Dev routes
/// must resolve to their dedicated control providers and remain covered.
#[cfg(target_os = "macos")]
#[test]
fn resident_only_audit_rejects_impossible_control_routes_and_keeps_real_ones() {
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let models = manifest["models"].as_array().expect("manifest models");
    for fake in [
        "chroma1_base",
        "chroma1_flash",
        "chroma1_hd",
        "flux_schnell",
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_klein_9b_true_v2",
    ] {
        let injected = SourceBoundAuditSurface {
            manifest_id: fake.to_owned(),
            surface: ResidentOnlyAuditSurface::StrictControl,
        };
        let error = source_bound_audit_inventory_from_surfaces(models, &[injected])
            .expect_err("an impossible strict-control route must fail closed");
        assert!(
            error.contains("has no production FLUX strict-control route"),
            "{fake} must fail at the production control router, got: {error}"
        );
    }

    let surfaces = source_bound_audit_surfaces(models).expect("production source surfaces");
    let source_inventory = source_bound_audit_inventory_from_surfaces(models, &surfaces)
        .expect("production source inventory");
    let (_, cells) = source_bound_resident_only_cells_from_inventory(models, &source_inventory)
        .expect("production resident-only cells");
    assert!(
        !cells.iter().any(|cell| {
            cell.manifest_id == "lens"
                && cell.tier == "q4"
                && cell.surface == ResidentOnlyAuditSurface::Base
        }),
        "base Lens q4 must remain on its production measured deferred-materialization contract"
    );
    assert!(
        !cells
            .iter()
            .any(|cell| cell.provider_id == "flux2_dev_edit"),
        "FLUX.2 Dev edit must remain on its provider-owned request-safety path"
    );
    assert!(cells.iter().any(|cell| {
        cell.manifest_id == "flux_dev"
            && cell.provider_id == "flux1_dev_control"
            && cell.surface == ResidentOnlyAuditSurface::StrictControl
            && cell.control_bytes == FLUX1_CONTROL_BYTES
    }));
    assert!(cells.iter().any(|cell| {
        cell.manifest_id == "flux2_dev"
            && cell.provider_id == "flux2_dev_control"
            && cell.surface == ResidentOnlyAuditSurface::StrictControl
            && cell.control_bytes == FLUX2_CONTROL_BYTES
    }));
    assert!(cells.iter().all(|cell| {
        cell.control_bytes == 0
            || matches!(cell.provider_id, "flux1_dev_control" | "flux2_dev_control")
    }));
}

/// Completeness mutation for the loophole found after the production-router rewrite. A
/// zero-cell route is still part of the candidate inventory: replacing FLUX.1 Schnell with an
/// already-declared Chroma entry must fail before deduplication, and simply deleting Schnell
/// must fail exact source-inventory equality even though the 32 Resident-only cells and 12
/// flips are unchanged.
#[cfg(target_os = "macos")]
#[test]
fn resident_only_audit_inventory_rejects_duplicate_and_zero_cell_drop_mutations() {
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let models = manifest["models"].as_array().expect("manifest models");
    let classifications =
        source_bound_manifest_classifications(models).expect("source classifications");
    let manifest_ids = source_bound_generic_manifest_ids(&classifications).collect::<Vec<_>>();
    let expected_surfaces =
        source_bound_audit_surfaces_from_manifest_ids(manifest_ids.iter().copied())
            .expect("source surfaces");
    let expected_inventory = source_bound_audit_inventory_from_surfaces(models, &expected_surfaces)
        .expect("source inventory");
    let (_, expected_resident) =
        source_bound_resident_only_cells_from_inventory(models, &expected_inventory)
            .expect("source Resident-only inventory");
    assert!(
        expected_inventory
            .iter()
            .any(|cell| cell.manifest_id == "flux_schnell"),
        "the mutation requires FLUX.1 Schnell in the full source inventory"
    );
    assert!(
        expected_resident
            .iter()
            .all(|cell| cell.manifest_id != "flux_schnell"),
        "the mutation requires FLUX.1 Schnell to be a representative zero-cell route"
    );

    let mut duplicate_replacement = manifest_ids.clone();
    let schnell = duplicate_replacement
        .iter()
        .position(|manifest_id| *manifest_id == "flux_schnell")
        .expect("FLUX.1 Schnell source declaration");
    duplicate_replacement[schnell] = "chroma1_base";
    let duplicate_error =
        source_bound_audit_surfaces_from_manifest_ids(duplicate_replacement.iter().copied())
            .expect_err("a duplicate replacement must fail before set conversion");
    assert!(
        duplicate_error.contains("duplicate source-bound audit manifest declaration chroma1_base"),
        "duplicate replacement failed for the wrong reason: {duplicate_error}"
    );

    let dropped_ids = manifest_ids
        .iter()
        .copied()
        .filter(|manifest_id| *manifest_id != "flux_schnell")
        .collect::<Vec<_>>();
    let dropped_surfaces =
        source_bound_audit_surfaces_from_manifest_ids(dropped_ids.iter().copied())
            .expect("dropped source surfaces still resolve");
    let dropped_inventory = source_bound_audit_inventory_from_surfaces(models, &dropped_surfaces)
        .expect("dropped source inventory still resolves");
    let (_, dropped_resident) =
        source_bound_resident_only_cells_from_inventory(models, &dropped_inventory)
            .expect("dropped Resident-only inventory still resolves");
    assert_eq!(
        dropped_resident, expected_resident,
        "precondition: dropping a zero-cell route preserves the old Resident-only summary"
    );
    let drop_error = require_exact_source_bound_inventory(&expected_inventory, &dropped_inventory)
        .expect_err("exact source equality must reject a dropped zero-cell route");
    assert!(
        drop_error.contains("flux_schnell"),
        "zero-cell drop must name its missing source route: {drop_error}"
    );
}

/// Explicit-exclusion mutation proof. PuLID is inside the eight-family story scope but takes
/// the identity-conditioned route outside MODEL_TABLE; Mage is outside that family scope but
/// must remain named because its split component tree cannot be represented by the generic
/// single-root sparse fixture. Neither exclusion may disappear or drift families silently.
#[cfg(target_os = "macos")]
#[test]
fn resident_only_audit_exclusions_reject_pulid_and_mage_scope_mutations() {
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let models = manifest["models"].as_array().expect("manifest models");
    let classifications =
        source_bound_manifest_classifications(models).expect("source classifications");
    let expected = source_bound_excluded_inventory_from_classifications(models, &classifications)
        .expect("source excluded inventory");

    let missing_pulid = classifications
        .iter()
        .filter(|item| item.manifest_id != "pulid_flux_dev")
        .cloned()
        .collect::<Vec<_>>();
    let actual = source_bound_excluded_inventory_from_classifications(models, &missing_pulid)
        .expect("the mutated exclusion list still resolves");
    let error = require_exact_source_bound_excluded_inventory(&expected, &actual)
        .expect_err("removing PuLID classification must fail exact equality");
    assert!(error.contains("pulid_flux_dev"), "wrong failure: {error}");

    let reclassified_pulid = classifications
        .iter()
        .cloned()
        .map(|mut item| {
            if item.manifest_id == "pulid_flux_dev" {
                item.disposition = SourceBoundManifestDisposition::GenericSelector;
            }
            item
        })
        .collect::<Vec<_>>();
    let actual = source_bound_excluded_inventory_from_classifications(models, &reclassified_pulid)
        .expect("the mutated disposition list still resolves");
    let error = require_exact_source_bound_excluded_inventory(&expected, &actual)
        .expect_err("changing PuLID classification must fail exact equality");
    assert!(error.contains("pulid_flux_dev"), "wrong failure: {error}");

    let missing_mage = classifications
        .iter()
        .filter(|item| {
            item.disposition != SourceBoundManifestDisposition::MageSplitComponentsExcluded
        })
        .cloned()
        .collect::<Vec<_>>();
    let actual = source_bound_excluded_inventory_from_classifications(models, &missing_mage)
        .expect("the mutated exclusion list still resolves");
    let error = require_exact_source_bound_excluded_inventory(&expected, &actual)
        .expect_err("removing Mage scope must fail exact equality");
    assert!(error.contains("mage_flow"), "wrong failure: {error}");

    let mut pulid_family_drift = models.clone();
    pulid_family_drift
        .iter_mut()
        .find(|model| model["id"] == "pulid_flux_dev")
        .expect("PuLID manifest entry")["family"] = Value::String("not-flux".to_owned());
    let error = source_bound_manifest_classifications(&pulid_family_drift)
        .expect_err("PuLID family drift must fail classification");
    assert!(error.contains("PuLID bespoke classification drifted"));

    let mut mage_family_drift = models.clone();
    mage_family_drift
        .iter_mut()
        .find(|model| model["id"] == "mage_flow")
        .expect("Mage manifest entry")["family"] = Value::String("flux".to_owned());
    let error = source_bound_manifest_classifications(&mage_family_drift)
        .expect_err("Mage family drift must fail classification");
    assert!(error.contains("Mage split-component classification drifted"));
}

/// sc-18251: the PREMISE of applying the composition leg to Resident, pinned against gen-core
/// rather than restated.
///
/// The shape leg of the `usable` filter exempts `MemoryStrategy::Resident` because
/// `optimized_eligibility` short-circuits `Ok(())` for a non-optimized selection before it
/// compares load shapes. The composition leg deliberately does NOT mirror that exemption,
/// because both of the gate's composition checks — canonical form (`Invalid`) and contract
/// agreement (`CompositionMismatch`) — run BEFORE the short-circuit, so gen-core rejects a
/// resident cell on either. If a pin bump ever moves those checks behind the Resident
/// short-circuit, this test reds and the filter's legs must be re-derived.
#[test]
fn gen_core_rejects_a_resident_cell_whose_composition_disagrees() {
    let generator = fixture_generator();
    let contract = generator.contract.as_ref().expect("fixture contract");

    let (selection, mut resident) = resident_evidence(
        contract,
        fixture_plan().tier,
        "text_to_image",
        None,
        request_geometry(&fixture_inputs(1024, 1024)),
        gib_to_bytes(4.0),
        Some("fixture-formula-v2"),
    );
    resident.conformance = gen_core::MemoryConformanceState::Verified;
    resident.dimensions = gen_core::MemoryEvidenceDimensions::VERIFIED;
    assert!(
        !selection.strategy.is_optimized(),
        "the premise is about the non-optimized rung"
    );
    assert_eq!(
        resident.optimized_eligibility(contract),
        Ok(()),
        "precondition: the resident cell is eligible before the composition is disturbed"
    );

    // Canonical but disagreeing: the gate's contract-agreement check must reject even a
    // RESIDENT cell, which is why the filter's composition leg carries no Resident exemption.
    resident.key.engaged_composition =
        vec![MemoryStrategy::Resident, MemoryStrategy::BoundedDecode];
    assert_eq!(
        resident.optimized_eligibility(contract),
        Err(gen_core::MemoryEvidenceVerdict::CompositionMismatch),
        "gen-core rejects a RESIDENT cell whose composition disagrees; the filter's \
             no-exemption composition leg is built on this"
    );

    // The canonical-form prefix also runs before the Resident short-circuit.
    resident.key.engaged_composition = Vec::new();
    assert_eq!(
        resident.optimized_eligibility(contract),
        Err(gen_core::MemoryEvidenceVerdict::Invalid),
        "gen-core rejects a RESIDENT cell whose composition is non-canonical"
    );
}

/// sc-18251 (review addendum): why the shape leg's `identity.load_shape` conjunct has no
/// isolating fixture of its own — the contract↔identity split it would need is impossible by
/// construction, and THAT premise is what this test pins against the pinned gen-core.
///
/// Every fixture moves `contract.load_shape` and `identity.load_shape` together, so deleting
/// either single conjunct of
/// `key.load_shape == contract.load_shape && key.load_shape == identity.load_shape` cannot be
/// distinguished by those fixtures alone. An isolating arm was written and it demonstrated the
/// impossibility empirically: gen-core's `conformance_errors` requires
/// `calibration.load_shape == contract.load_shape`, and `select_strategy` refuses EVERY
/// candidate on a non-conformant contract with `Unverified(Invalid)` before grading a single
/// one, so a split contract refuses the request no matter which conjunct the filter carries.
/// The pair is effectively one comparison, kept in both spellings only to mirror
/// `optimized_eligibility` literally.
///
/// If a pin bump ever legalizes the split, this test reds — and the identity conjunct then
/// needs its own fixture arm, because it will have become independently load-bearing.
#[test]
fn gen_core_forbids_a_contract_identity_load_shape_split() {
    use gen_core::MemoryCalibrationIdentity;

    let generator = fixture_generator();
    let contract = generator.contract.as_ref().expect("fixture contract");
    assert!(
        contract.conformance_errors().is_empty(),
        "precondition: the agreeing fixture contract is conformant"
    );
    let fingerprint = contract
        .calibration
        .as_ref()
        .expect("fixture calibration")
        .fingerprint
        .clone();

    // Identity moves, contract stays.
    let mut identity_split = contract.clone();
    identity_split.calibration = Some(MemoryCalibrationIdentity::new(
        fingerprint.clone(),
        gen_core::LoadShape::DeferredMaterialization,
    ));
    assert!(
        identity_split
            .conformance_errors()
            .iter()
            .any(|error| error.contains("load shape")),
        "gen-core must reject a contract whose calibration identity shape disagrees with its \
             live load shape; the filter's fused shape conjuncts are built on this"
    );

    // Contract moves, identity stays.
    let mut contract_split = contract.clone();
    contract_split.load_shape = gen_core::LoadShape::DeferredMaterialization;
    assert!(
        contract_split
            .conformance_errors()
            .iter()
            .any(|error| error.contains("load shape")),
        "the split must be rejected in both directions"
    );

    // And a non-conformant contract never grades candidates at all — the refusal outruns the
    // filter, which is why no fixture can observe the identity conjunct alone.
    let selection = crate::memory_strategy::select_strategy(
        crate::memory_strategy::RequestScope {
            resolved_route: "fixture_provider",
            backend: "mlx",
            tier: fixture_plan().tier,
            mode: "text_to_image",
            overlay: None,
            geometry: request_geometry(&fixture_inputs(1024, 1024)),
        },
        &identity_split,
        Some(crate::memory_strategy::Budget {
            available_gb: 64.0,
            reclaimable_gb: 0.0,
            total_gb: 64.0,
            reserved_headroom_gb: 0.0,
        }),
        &[],
    );
    assert_eq!(
        selection,
        crate::memory_strategy::Selection::Unverified {
            reason: gen_core::MemoryEvidenceVerdict::Invalid
        },
        "a split contract refuses everything before any candidate is graded"
    );
}

/// The sc-22738 mutation test: the App Runtime ALWAYS behaves as if the measurement were valid.
///
/// A binding whose provider compile-closure has MOVED since capture — the manifest opt-in names
/// a digest neither the packaged record nor the live ledger carries — routes byte-for-byte
/// identically to the same request with a matching digest. Identity, fingerprint, load shape and
/// geometry are held fixed, so the closure is the ONLY variable between the two evaluations:
/// same `AdmissionPath`, same selected strategy, same `needed_gb` in the refusal, and the same
/// request-scoped process ceiling. Measurement currency is a re-capture signal for the JS probe
/// tooling and nothing else.
///
/// MUTATION: re-adding a closure comparison ANYWHERE on the MLX path turns this red — a
/// `stale_admitted_peak_bytes` widening in the selector, a `StaleIdentity`/`StaleBundle`
/// pre-demotion in `evidence_admission_route`, or a closure conjunct in
/// `memory_calibration::evidence_for`. Each of those moves the path, the needed_gb or the
/// ceiling on the moved side and leaves the matching side alone, so no equality below survives.
#[test]
fn a_moved_provider_closure_admits_the_ladder_at_the_exact_measured_peak() {
    let bundle = fixture_bundle();
    let generator = fixture_generator();
    let plan = fixture_plan();
    let inputs = fixture_inputs(1024, 1024);

    // The moved plan. Every binding names a closure digest nothing was ever measured under,
    // while the records in the bundle keep the fixture digest — the exact shape a pin bump
    // produces once the provider's crates recompile to something new.
    let moved_digest = "a".repeat(64);
    assert_ne!(
        moved_digest,
        fixture_closure_digest(),
        "the moved digest must actually differ from the measured one"
    );
    let mut moved_plan = fixture_plan();
    let MlxCalibrationConfig::Valid(calibration) = &mut moved_plan.calibration else {
        panic!("fixture calibration");
    };
    for binding in &mut calibration.bindings {
        binding
            .query
            .inference_closure_digest
            .clone_from(&moved_digest);
    }

    // The seam. Before sc-22738 the moved binding was pre-demoted here, to
    // `AdmissionPath::Legacy` with a `StaleIdentity` reason, before any candidate was graded.
    let moved_route = evidence_admission_route(
        &bundle,
        &moved_plan,
        &inputs,
        "text_to_image",
        fixture_budget(9.0),
    )
    .expect("a moved closure routes, it does not error");
    let route = evidence_admission_route(
        &bundle,
        &plan,
        &inputs,
        "text_to_image",
        fixture_budget(9.0),
    )
    .expect("the matching closure routes");
    assert_eq!(
        moved_route.path,
        AdmissionPath::Evidence,
        "a moved-closure cell must reach the selector, not be pre-demoted: {:?}",
        moved_route.fallback_reason
    );
    assert!(!moved_route.evidence.is_empty());
    assert_eq!(
        format!("{moved_route:?}"),
        format!("{route:?}"),
        "the moved closure must produce an identical admission route"
    );

    // End to end, at the budget the retired widening was calibrated to split. The record's
    // envelope peak is exactly 5 GiB with a 3 GiB captured foreign reserve; at 8.5 GiB the
    // effective budget is 5.5 GiB, so the RAW peak fits and the old recapture-widened 5.63 GiB
    // did not. A re-added widening therefore refuses the moved side here and admits the
    // matching side, breaking every equality below.
    let evaluate =
        |plan: &MlxRequestPlan, inputs: &MlxRequestInputs, budget_gib: f64, total_peak_gib: f64| {
            evaluate_request_with_budget_using_bundle(
                &generator,
                plan,
                inputs,
                MemoryCacheState::Cold,
                OffloadPolicy::Resident,
                crate::execution_planner::WarmPolicyProposal::inert("fixture"),
                fixture_budget(budget_gib),
                gib_to_bytes(total_peak_gib),
                0,
                &[],
                Some(&bundle),
                None,
            )
        };
    let admitted_moved = evaluate(&moved_plan, &inputs, 8.5, 4.0)
        .expect("a moved closure is graded at the exact measured peak, so 8.5 GiB admits");
    let admitted = evaluate(&plan, &inputs, 8.5, 4.0)
        .expect("the matching closure admits the verified rung at the raw peak");
    assert_eq!(
        admitted_moved.context.selection.strategy,
        MemoryStrategy::BoundedDecode,
        "the measured rung itself must be selected"
    );
    assert!(
        admitted_moved.process_limit_bytes.is_some(),
        "a moved-closure exact cell still derives the request-scoped ceiling"
    );
    assert_eq!(
        format!("{admitted_moved:?}"),
        format!("{admitted:?}"),
        "the decision and the whole memory context must be byte-for-byte equal"
    );

    // `needed_gb`, read where it is actually published: the refusal message. At 4 GiB neither
    // side fits, and both must quote the same host requirement — the raw peak plus the captured
    // foreign reserve, never a widened one.
    let refused_moved = evaluate(&moved_plan, &inputs, 4.0, 4.0)
        .expect_err("4 GiB fits neither the peak nor the reserve")
        .to_string();
    let refused = evaluate(&plan, &inputs, 4.0, 4.0)
        .expect_err("4 GiB fits neither the peak nor the reserve")
        .to_string();
    assert!(
        refused_moved.contains("needs at least"),
        "the refusal must quote the host requirement: {refused_moved}"
    );
    assert_eq!(
        refused_moved, refused,
        "the moved closure must not change the quoted needed_gb"
    );

    // And off the measured geometry, where the record is an extrapolation BASIS rather than an
    // exact cell. sc-22738 retired the rule that only a closure-current record could seed a
    // fitted curve, which used to make this arm refuse on floors alone at 8 GiB.
    let off_geometry = fixture_inputs(768, 768);
    let fitted_moved = evaluate(&moved_plan, &off_geometry, 8.0, 12.0)
        .expect("a moved-closure record is a legitimate fitted basis");
    let fitted = evaluate(&plan, &off_geometry, 8.0, 12.0)
        .expect("the matching-closure record is a legitimate fitted basis");
    assert_eq!(
        fitted_moved.context.selection.strategy,
        MemoryStrategy::BoundedDecode,
        "the fitted estimate must admit: {:?}",
        fitted_moved.context.selection
    );
    assert_eq!(
        format!("{fitted_moved:?}"),
        format!("{fitted:?}"),
        "a fitted extrapolation must not depend on the basis record's closure"
    );
}

#[test]
fn covered_cell_selects_exact_strategy_cold_and_warm_without_resident_bypass() {
    let bundle = fixture_bundle();
    let generator = fixture_generator();
    let plan = fixture_plan();
    let inputs = fixture_inputs(1024, 1024);
    let expected = MemorySelection {
        strategy: MemoryStrategy::BoundedDecode,
        parameters: fixture_binding("q4", "packed-q4").selection_parameters,
        tier: plan.tier,
    };
    let exact = evidence_admission_route(
        &bundle,
        &plan,
        &inputs,
        "text_to_image",
        fixture_budget(8.0),
    )
    .expect("covered route")
    .evidence
    .into_iter()
    .next()
    .expect("exact evidence")
    .evidence;
    assert!(
        exact.validation_errors().is_empty(),
        "fixture evidence errors: {:?}",
        exact.validation_errors()
    );
    let contract = generator.contract.as_ref().expect("fixture contract");
    assert!(
        contract.validate_selection(&expected).is_ok(),
        "fixture contract rejection: {:?}",
        contract.validate_selection(&expected)
    );
    let evaluate = |cache_state, committed_bytes| {
        evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &inputs,
            cache_state,
            OffloadPolicy::Resident,
            crate::execution_planner::WarmPolicyProposal::inert("fixture"),
            MemoryBudget {
                committed_bytes,
                ..fixture_budget(8.0)
            },
            gib_to_bytes(4.0),
            0,
            &[],
            Some(&bundle),
            None,
        )
        .expect("the exact covered request must select its verified rung")
    };
    let cold = evaluate(MemoryCacheState::Cold, 0);
    let warm = evaluate(MemoryCacheState::Warm, gib_to_bytes(3.0));
    assert_eq!(cold.context.selection, expected);
    assert_eq!(warm.context.selection, expected);
    assert!(cold.memory.tile_vae_decode && warm.memory.tile_vae_decode);
    assert_eq!(cold.context.predicted_peak_bytes, gib_to_bytes(5.0));
    assert_eq!(
        warm.context.predicted_peak_bytes, cold.context.predicted_peak_bytes,
        "warm attribution credits resident assets in the budget without rewriting exact evidence"
    );
    assert_eq!(
        warm.context.budget.committed_bytes, 0,
        "the provider's three resident GiB are credited exactly once"
    );
}

#[test]
fn verified_ladder_selects_resident_decode_then_attention_as_budget_tightens() {
    let (bundle, plan) = fixture_ladder();
    let generator = fixture_generator();
    let inputs = fixture_inputs(1024, 1024);
    for (total_gib, expected) in [
        (10.0, MemoryStrategy::Resident),
        (8.0, MemoryStrategy::BoundedDecode),
        (7.0, MemoryStrategy::BoundedAttention),
    ] {
        let evaluation = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            crate::execution_planner::WarmPolicyProposal::inert("fixture"),
            fixture_budget(total_gib),
            gib_to_bytes(4.0),
            0,
            &[],
            Some(&bundle),
            None,
        )
        .unwrap_or_else(|error| panic!("{total_gib} GiB ladder failed: {error}"));
        assert_eq!(evaluation.context.selection.strategy, expected);
        assert!(
            bundle
                .records
                .iter()
                .any(|record| record.id == evaluation.context.evidence_revision),
            "telemetry must name the selected evidence record, not the whole candidate list"
        );
    }
}

/// SC-15808's reachability claim is structural: the Krea route enters the same admission path as
/// every other MLX provider and has no provider-local three-rung ceiling. The synthetic rows here
/// deliberately make no calibration claim; Krea's family/model stories still own implementation
/// and real-weight evidence for rungs 3 and 4. This test proves those future verified rows are not
/// hidden from `select_strategy` by route-specific admission code.
#[test]
fn krea_route_admission_reaches_later_rungs_when_verified_rows_exist() {
    use gen_core::{LoadShape, MemoryParameterRanges, MemoryStrategySupport};
    use sceneworks_core::memory_calibration::RequiredNullable;

    const KREA_CONTROL_ROUTE: &str = "krea_2_turbo_control";
    let (mut bundle, mut plan) = fixture_ladder();
    let mut generator = fixture_generator();
    generator.descriptor.id = KREA_CONTROL_ROUTE;
    let contract = generator.contract.as_mut().expect("fixture contract");
    contract.provider_id = KREA_CONTROL_ROUTE.to_owned();
    contract.load_shape = LoadShape::DeferredMaterialization;
    // ABI 2: the calibration identity, every receipt, and the binding must carry the same
    // materialization shape as the contract or eligibility fails closed.
    contract.calibration = Some(gen_core::MemoryCalibrationIdentity::new(
        "fixture-formula-v2",
        LoadShape::DeferredMaterialization,
    ));
    for record in &mut bundle.records {
        record.load_shape =
            sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
    }
    contract.lifecycle.transformer_window_materialization = true;
    let transformer = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
        .expect("transformer-residency capability");
    transformer.support = MemoryStrategySupport::Implemented;
    transformer.parameters = MemoryParameterRanges {
        transformer_window_sizes: vec![1],
        ..Default::default()
    };

    let transformer_parameters = JsonObject::from_iter([
        ("decodeTileEdge".to_owned(), serde_json::json!(512)),
        ("decodeOverlap".to_owned(), serde_json::json!(128)),
        ("attentionChunkSize".to_owned(), serde_json::json!(256)),
        ("transformerWindowSize".to_owned(), serde_json::json!(1)),
    ]);
    let mut transformer_record = bundle
        .records
        .last()
        .expect("attention fixture row")
        .clone();
    transformer_record.id = "imc-15808000000000000000".to_owned();
    transformer_record.logical_case_id = "implan-15808000000000000000".to_owned();
    transformer_record.strategy.rung = StrategyRung::BoundedTransformerResidency;
    transformer_record.strategy.engaged_rungs = contract
        .engaged_composition(MemoryStrategy::BoundedTransformerResidency)
        .into_iter()
        .map(|strategy| match strategy {
            MemoryStrategy::Resident => StrategyRung::Resident,
            MemoryStrategy::StagedResidency => StrategyRung::StagedResidency,
            MemoryStrategy::BoundedDecode => StrategyRung::BoundedDecode,
            MemoryStrategy::BoundedAttention => StrategyRung::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency => {
                StrategyRung::BoundedTransformerResidency
            }
        })
        .collect();
    transformer_record.strategy.parameters = transformer_parameters.clone();
    transformer_record.sweep.cases[0].parameters = transformer_parameters.clone();
    if let RequiredNullable::Value(predicted) = &mut transformer_record.predicted_peak_bytes {
        let predicted = predicted
            .full_mut()
            .expect("fixture has full phase telemetry");
        predicted.conditioning = predicted.conditioning.min(gib_to_bytes(3.0));
        predicted.denoise = gib_to_bytes(3.0);
        predicted.decode = predicted.decode.min(gib_to_bytes(3.0));
        predicted.overall = gib_to_bytes(3.0);
    }
    if let RequiredNullable::Value(observed) = &mut transformer_record.observed_memory {
        let observed = observed
            .full_mut()
            .expect("fixture has full phase telemetry");
        observed.overall.active_bytes = gib_to_bytes(3.0);
        observed.overall.allocator_bytes = gib_to_bytes(3.0);
        observed.overall.reclaimable_bytes = 0;
    }
    bundle.records.push(transformer_record);

    plan.engine_id = KREA_CONTROL_ROUTE;
    plan.model_id = "krea_2_turbo".to_owned();
    let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
        panic!("fixture calibration");
    };
    calibration.bindings.push(fixture_binding_for(
        "q4",
        "packed-q4",
        StrategyRung::BoundedTransformerResidency,
        transformer_parameters,
    ));
    for binding in &mut calibration.bindings {
        binding.provider = KREA_CONTROL_ROUTE.to_owned();
        binding.query.load_shape =
            sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
    }
    for record in &mut bundle.records {
        record.target.provider = KREA_CONTROL_ROUTE.to_owned();
        record.target.model_id = "krea_2_turbo".to_owned();
    }

    for (total_gib, expected) in [
        (7.0, MemoryStrategy::BoundedAttention),
        (6.0, MemoryStrategy::BoundedTransformerResidency),
    ] {
        let evaluation = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &fixture_inputs(1024, 1024),
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            crate::execution_planner::WarmPolicyProposal::inert("fixture"),
            fixture_budget(total_gib),
            gib_to_bytes(4.0),
            0,
            &[],
            Some(&bundle),
            None,
        )
        .unwrap_or_else(|error| panic!("{total_gib} GiB Krea route failed: {error}"));
        assert_eq!(evaluation.context.selection.strategy, expected);
    }
}

#[test]
fn candidate_specific_foreign_reserve_does_not_block_a_fitting_lower_rung() {
    use sceneworks_core::memory_calibration::Hardware;

    let (mut bundle, plan) = fixture_ladder();
    let set_reserve = |record: &mut sceneworks_core::memory_calibration::EvidenceRecord,
                       reserve_gib: f64| {
        let Hardware::Mlx(hardware) = &mut record.hardware else {
            panic!("MLX fixture hardware");
        };
        hardware.memory_bytes = gib_to_bytes(10.0);
        hardware.mlx_memory_limit_bytes = gib_to_bytes(10.0 - reserve_gib);
        hardware.wired_limit_bytes = hardware.memory_bytes;
    };
    set_reserve(&mut bundle.records[0], 4.0);
    set_reserve(&mut bundle.records[1], 2.0);
    set_reserve(&mut bundle.records[2], 2.0);

    let evaluation = evaluate_request_with_budget_using_bundle(
        &fixture_generator(),
        &plan,
        &fixture_inputs(1024, 1024),
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        fixture_budget(10.0),
        gib_to_bytes(4.0),
        0,
        &[],
        Some(&bundle),
        None,
    )
    .expect("the bounded-decode candidate's own 5+2 GiB boundary fits");
    assert_eq!(
        evaluation.context.selection.strategy,
        MemoryStrategy::BoundedDecode,
        "resident 7+4 GiB must be excluded without imposing its reserve on bounded decode"
    );
    assert_eq!(
        evaluation.process_limit_bytes,
        Some(gib_to_bytes(8.0)),
        "request ceiling comes from the selected candidate's 2 GiB reserve"
    );
}

#[test]
fn capture_host_reserve_scales_to_48_gib() {
    use sceneworks_core::memory_calibration::MlxAdmissionEnvelope;

    let capture_host = gib_to_bytes(128.0);
    let envelope = MlxAdmissionEnvelope {
        peak_bytes: gib_to_bytes(16.41),
        observed_non_reclaimable_wired_bytes: gib_to_bytes(15.0),
        capture_host_bytes: capture_host,
        foreign_reserve_bytes: gib_to_bytes(46.93),
    };
    let live_host = gib_to_bytes(48.0);
    assert!(
        envelope
            .peak_bytes
            .saturating_add(envelope.foreign_reserve_bytes)
            > live_host,
        "the old absolute 128 GiB-host reserve reproduces the false 48 GiB refusal"
    );
    assert!(
        envelope.required_host_bytes() <= live_host,
        "the true static boundary must agree that this candidate can fit below 48 GiB"
    );
    let live_reserve = envelope.foreign_reserve_for_host_bytes(live_host);
    // sc-22738 dropped the stale-margin half of this test with the widening it graded. The
    // measured peak is charged as measured, whatever closure it was captured under.
    assert!(
        envelope.peak_bytes.saturating_add(live_reserve) <= live_host,
        "the measured peak plus the normalized reserve must fit the live host"
    );
    let process_limit = live_host.saturating_sub(live_reserve);
    assert!(
        envelope.peak_bytes <= process_limit,
        "the request remains below the absolute MLX process limit used for OOM containment"
    );
    assert_eq!(
            live_reserve,
            18_896_513_925,
            "46.93 GiB reserved on 128 GiB scales, conservatively rounded up, to 17.59875 GiB on 48 GiB"
        );
}

#[test]
fn packaged_bundle_without_an_exact_record_is_a_normal_no_record_reason() {
    // The packaged evidence is current for schema v4 / ABI 3. An uncovered fixture therefore
    // degrades to the legacy path with the precise `NoRecord` reason, not bundle drift.
    let route = packaged_admission_route(
        &fixture_plan(),
        &fixture_inputs(1024, 1024),
        "text_to_image",
        fixture_budget(8.0),
    )
    .expect("a current promoted bundle with no exact record degrades, never errors");
    assert_eq!(route.path, AdmissionPath::Legacy);
    assert_eq!(route.fallback_reason, Some(LegacyAdmissionReason::NoRecord));
}

/// SC-15805: `memory_for_selection` is the live MLX memory-admission seam — the
/// [`GenerationMemory`] it builds reaches the engine via `image_jobs/base.rs`. It asks the
/// CONTRACT which rungs a selection engages instead of re-deriving the answer from
/// `MemoryStrategy`'s numeric order, and that difference is only observable when a provider
/// declares a cheaper rung unavailable.
///
/// Every other test in this module uses [`mage_request_contract`], whose optimized rungs are all
/// `Missing`, so it never selects past `Resident` and never sets these knobs at all — a test that
/// passes because a field was never set is a false green. This one sets the field.
///
/// Reverting `memory_for_selection` to `selection.strategy >= MemoryStrategy::BoundedDecode`
/// (etc.) must turn this RED.
#[test]
fn an_unimplemented_rung_is_not_engaged_by_a_deeper_selection() {
    use gen_core::{MemoryParameterRanges, MemoryStrategyCapability, MemoryStrategySupport};

    let contract_with = |missing: MemoryStrategy| {
        let mut contract = mage_request_contract();
        contract.strategies = MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: if strategy == missing {
                    MemoryStrategySupport::Missing
                } else {
                    MemoryStrategySupport::Implemented
                },
                parameters: MemoryParameterRanges::default(),
            })
            .collect();
        contract
    };
    let deepest = MemorySelection {
        strategy: MemoryStrategy::BoundedTransformerResidency,
        parameters: Default::default(),
        tier: request_plan().tier,
    };

    // Rung 2 unavailable: the deepest selection must not tile the decode.
    let memory = memory_for_selection(&contract_with(MemoryStrategy::BoundedDecode), deepest);
    assert!(
        !memory.tile_vae_decode,
        "BoundedDecode is declared Missing, so a BoundedTransformerResidency selection must \
             not switch decode tiling on underneath the provider: the ladder's numeric order is a \
             COST ordering, not a dependency"
    );
    // ...and the rungs the provider does declare are still engaged, so this is not the vacuous
    // "everything is false" green that would also pass if the function returned Default.
    assert!(memory.chunk_attention);
    assert!(memory.stream_transformer_blocks);

    // Rung 3 unavailable: same property, one knob over.
    let memory = memory_for_selection(&contract_with(MemoryStrategy::BoundedAttention), deepest);
    assert!(
        !memory.chunk_attention,
        "BoundedAttention is declared Missing, so a deeper selection must not chunk attention"
    );
    assert!(memory.tile_vae_decode);
    assert!(memory.stream_transformer_blocks);

    // Positive control: with every rung implemented, the cumulative default still applies in
    // full. Without this, a `memory_for_selection` that returned `Default` unconditionally would
    // satisfy both assertions above.
    let all_implemented = {
        let mut contract = mage_request_contract();
        contract.strategies = MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: MemoryStrategySupport::Implemented,
                parameters: MemoryParameterRanges::default(),
            })
            .collect();
        contract
    };
    let memory = memory_for_selection(&all_implemented, deepest);
    assert!(
        !memory.stage_residency,
        "rung 4 must not evict the warm cache by implicitly engaging rung 1"
    );
    assert!(memory.tile_vae_decode);
    assert!(memory.chunk_attention);
    assert!(memory.stream_transformer_blocks);

    let staged = memory_for_selection(
        &all_implemented,
        MemorySelection {
            strategy: MemoryStrategy::StagedResidency,
            parameters: Default::default(),
            tier: request_plan().tier,
        },
    );
    assert!(
        staged.stage_residency,
        "an explicit rung-1 selection must reach GenerationMemory"
    );
    assert!(!staged.tile_vae_decode);
    assert!(!staged.chunk_attention);
    assert!(!staged.stream_transformer_blocks);
}

#[test]
fn request_scope_reselects_a_b_a_without_fragmenting_request_axes() {
    let generator = request_generator(Some(mage_request_contract()));
    let plan = request_plan();
    let budget = MemoryBudget {
        total_bytes: gib_to_bytes(16.0),
        committed_bytes: gib_to_bytes(4.0),
        reclaimable_bytes: 0,
        reserved_headroom_bytes: gib_to_bytes(2.0),
    };
    let a = request_inputs(512, 512, 3);
    let b = request_inputs(1536, 1024, 3);
    let first_a = evaluate_request_with_budget(
        &generator,
        &plan,
        &a,
        MemoryCacheState::Cold,
        OffloadPolicy::Sequential,
        budget,
        gib_to_bytes(8.0),
        0,
        &[],
    )
    .unwrap();
    let selected_b = evaluate_request_with_budget(
        &generator,
        &plan,
        &b,
        MemoryCacheState::Warm,
        OffloadPolicy::Sequential,
        budget,
        gib_to_bytes(9.0),
        0,
        &[],
    )
    .unwrap();
    let second_a = evaluate_request_with_budget(
        &generator,
        &plan,
        &a,
        MemoryCacheState::Warm,
        OffloadPolicy::Sequential,
        budget,
        gib_to_bytes(8.0),
        0,
        &[],
    )
    .unwrap();

    // A 3-image job is three sequential forward passes, so the modeled batch stays 1 even though
    // `request_inputs` carried count 3 (see `request_batch`).
    assert_eq!(first_a.context.geometry.batch, 1);
    assert_eq!(first_a.context.mode, MemoryMode::Edit);
    assert_eq!(
        first_a.context.overlay.as_deref(),
        Some("references:2+mask+adapters:1")
    );
    assert_eq!(first_a.context.cache_state, MemoryCacheState::Cold);
    assert_eq!(selected_b.context.geometry.width, 1536);
    assert_eq!(selected_b.context.cache_state, MemoryCacheState::Warm);
    assert_eq!(first_a.context.selection, second_a.context.selection);
    assert_eq!(
        first_a.context.predicted_peak_bytes, second_a.context.predicted_peak_bytes,
        "the intervening geometry cannot poison a warm follow-up"
    );
}

/// A multi-image job is a sequential loop with an MLX cache release between items (sc-5567), so
/// its peak is one image's working set — the estimator must be INVARIANT in the job count.
///
/// The numbers are the real reported rejection: krea_2_turbo bf16 (33.22 GiB of safetensors) at
/// 1152x2048 count 4 on a 128 GiB Mac (126.00 GiB after the 2 GiB legacy unified reserve). The
/// count multiplier quoted 33.22 + 16 x 2.25 x 4 = 177.22 GiB — the activation term alone
/// exceeded the whole budget, so this cell rejected every model at every tier.
#[test]
fn generic_request_peak_is_invariant_in_the_job_image_count() {
    let plan = MlxRequestPlan {
        engine_id: "krea_2_turbo",
        model_id: "krea_2_turbo".to_owned(),
        tier: MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        },
        asset_bytes: 35_666_644_396,
        folded_control_bytes: 0,
        folded_adapter_bytes: 0,
        activation_headroom_bytes: gib_to_bytes(HEADROOM_GB - 2.0),
        fixed_reserve_bytes: gib_to_bytes(OS_APP_RESERVE_GB - 2.0),
        calibration: MlxCalibrationConfig::Absent,
        load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
    };
    // Go through `request_geometry` rather than hand-building a `batch: 1` geometry, so this
    // exercises the production count -> batch seam instead of asserting a value it supplies.
    let peak = |count: u32| {
        plan.generic_total_peak_bytes(request_geometry(&request_inputs(1152, 2048, count)))
    };

    let single = peak(1);
    for count in [2, 4, 8] {
        assert_eq!(
            peak(count),
            single,
            "count {count} is {count} sequential passes, not a batched one"
        );
    }

    let budget = gib_to_bytes(126.0);
    assert!(
        peak(4) <= budget,
        "the reported cell must admit: needed {:.2} GiB vs {:.2} GiB available",
        peak(4) as f64 / BYTES_PER_GIB,
        budget as f64 / BYTES_PER_GIB
    );

    // Non-constant control: resolution is still a real axis, so an estimator that stopped
    // scaling entirely would not pass by accident.
    assert!(
        peak(1) > plan.generic_total_peak_bytes(request_geometry(&request_inputs(512, 512, 1))),
        "the estimator must still grow with output resolution"
    );
}

/// sc-16209: the generic fallback must consume Krea's provider-owned, measured 1024²
/// activation anchor instead of charging the unrelated 14 GiB worst-case family anchor.
#[cfg(target_os = "macos")]
#[test]
fn krea_measured_anchor_admits_the_64_gib_motivating_cell() {
    let weights = tempfile::tempdir().expect("Krea weights fixture");
    std::fs::File::create(weights.path().join("fixture.safetensors"))
        .expect("Krea weights fixture file")
        .set_len(1024)
        .expect("Krea weights fixture size");
    let spec = LoadSpec::new(WeightsSource::Dir(weights.path().to_path_buf()));
    let mut plan =
        MlxRequestPlan::for_spec_and_manifest("krea_2_turbo", "krea_2_turbo", &spec, None, None);
    // Preserve the production-selected allowance while pinning the exact asset term from the
    // measured motivating cell. A tiny sparse fixture is enough to exercise descriptor lookup.
    plan.asset_bytes = gib_to_bytes(33.22);

    let peak_gb = plan.generic_total_peak_bytes(request_geometry(&request_inputs(1152, 2048, 1)))
        as f64
        / BYTES_PER_GIB;
    let expected_gb = 33.22 + 2.0 + 7.67 * 2.25;
    assert!(
        (peak_gb - expected_gb).abs() < 1e-5,
        "Krea's measured 7.67 GiB anchor should model {expected_gb:.4} GiB, got {peak_gb:.4}"
    );
    assert!(
        peak_gb <= 62.0,
        "a 64 GiB Mac's 62 GiB request budget must admit the measured Krea cell"
    );

    // Mutation guard: restoring the generic 14 GiB anchor through the production estimator
    // reproduces the original rejection.
    plan.activation_headroom_bytes = gib_to_bytes(16.0);
    plan.fixed_reserve_bytes = gib_to_bytes(2.0);
    let generic_peak_gb =
        plan.generic_total_peak_bytes(request_geometry(&request_inputs(1152, 2048, 1))) as f64
            / BYTES_PER_GIB;
    assert!(
        generic_peak_gb > 62.0,
        "the generic fallback must remain a rejecting estimate, got {generic_peak_gb:.4} GiB"
    );
}

/// An unmeasured route must retain the conservative generic allowance. Krea Turbo's measured
/// family anchor cannot silently authorize the distinct Krea Raw graph.
#[test]
fn unmeasured_krea_route_retains_the_generic_activation_fallback() {
    let weights = tempfile::tempdir().expect("Krea weights fixture");
    std::fs::File::create(weights.path().join("fixture.safetensors"))
        .expect("Krea weights fixture file")
        .set_len(1024)
        .expect("Krea weights fixture size");
    let spec = LoadSpec::new(WeightsSource::Dir(weights.path().to_path_buf()));
    let plan = MlxRequestPlan::for_spec_and_manifest("krea_2_raw", "krea_2_raw", &spec, None, None);

    assert_eq!(plan.fixed_reserve_bytes, gib_to_bytes(2.0));
    assert_eq!(plan.activation_headroom_bytes, gib_to_bytes(16.0));
    assert_eq!(
        plan.generic_total_peak_bytes(request_geometry(&request_inputs(1152, 2048, 1))),
        plan.asset_bytes
            .saturating_add(gib_to_bytes(2.0))
            .saturating_add(gib_to_bytes(14.0 * 2.25))
    );
}

/// sc-16195: the OS/app reserve inside the flat headroom is FIXED, so only the activation term
/// may scale with output area.
///
/// Every number here is pinned in absolute GiB rather than re-derived from the formula, so the
/// test fails if the split moves — a test that recomputed `fixed + area * mp` would pass against
/// any split, including the broken one.
#[test]
fn generic_request_peak_scales_the_area_term_but_never_the_os_reserve() {
    let asset_gb = 33.22_f64;
    let plan = MlxRequestPlan {
        engine_id: "krea_2_turbo",
        model_id: "krea_2_turbo".to_owned(),
        tier: MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        },
        asset_bytes: gib_to_bytes(asset_gb),
        folded_control_bytes: 0,
        folded_adapter_bytes: 0,
        activation_headroom_bytes: gib_to_bytes(HEADROOM_GB - 2.0),
        fixed_reserve_bytes: gib_to_bytes(OS_APP_RESERVE_GB - 2.0),
        calibration: MlxCalibrationConfig::Absent,
        load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
    };
    let peak_gb = |width, height| {
        plan.generic_total_peak_bytes(request_geometry(&request_inputs(width, height, 1))) as f64
            / BYTES_PER_GIB
    };
    // The split: 16 GiB of allowance = a 2 GiB fixed remainder of the 4 GiB OS/app reserve (the
    // other 2 are carried separately as the legacy unified reserve) + a 14 GiB area term.
    let close = |actual: f64, expected: f64| (actual - expected).abs() < 1e-6;

    // 1024² is the calibration anchor and must be BIT-IDENTICAL to the pre-sc-16195 estimator —
    // the sweep re-derived the shape, it did not re-cut the safety margin.
    assert!(
        close(peak_gb(1024, 1024), asset_gb + 16.0),
        "1024² must be unchanged at asset + 16: got {:.4}",
        peak_gb(1024, 1024)
    );

    // Above 1024² only the 14 GiB area term scales. The old estimator scaled all 16.
    for (width, height, megapixels) in [(1024, 1536, 1.5), (1152, 2048, 2.25), (2048, 2048, 4.0)] {
        let expected = asset_gb + 2.0 + 14.0 * megapixels;
        assert!(
            close(peak_gb(width, height), expected),
            "{width}x{height}: expected asset + 2 + 14*{megapixels} = {expected:.4}, got {:.4}",
            peak_gb(width, height)
        );
        // Mutation guard: the pre-sc-16195 formula scaled the whole 16, so it is strictly
        // larger everywhere above the anchor. If the reserve ever starts scaling again, this
        // fires even if the arithmetic above were loosened.
        assert!(
            peak_gb(width, height) < asset_gb + 16.0 * megapixels,
            "{width}x{height} must model below the old whole-headroom scaling"
        );
    }

    // Below 1024² the scale is floored at 1.0: the measured transient stops falling off
    // proportionally down there (illustrious 0.305x and qwen 0.512x of their anchors at 0.25x
    // area, both above proportional), so the floor is the conservative reading.
    assert!(
        close(peak_gb(512, 512), asset_gb + 16.0),
        "sub-anchor requests stay floored at the 1024² allowance: got {:.4}",
        peak_gb(512, 512)
    );
}

/// sc-16209: a family whose allowance was measured as a BARE activation transient keeps that
/// whole measurement in the area term, with the request budget's remaining OS reserve added as
/// a separate fixed term.
///
/// This is the lens dense path ([`HeadroomAllowance::LENS_DENSE`], sc-11924). Subtracting the
/// legacy 2 GiB request-budget reserve from its 29.88 GiB BARE activation measurement would
/// under-predict above 1024² by `2·MP`. The safe decomposition is fixed 2 + 29.88·MP.
#[test]
fn a_bare_transient_allowance_keeps_its_whole_area_term() {
    let asset_gb = 28.43_f64;
    let plan = |headroom: HeadroomAllowance| {
        let fixed_reserve_bytes = gib_to_bytes(
            (OS_APP_RESERVE_GB - crate::fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB).max(0.0),
        );
        let activation_anchor_bytes =
            gib_to_bytes((headroom.total_gb - headroom.os_reserve_gb).max(0.0));
        MlxRequestPlan {
            engine_id: "lens_turbo",
            model_id: "lens_turbo".to_owned(),
            tier: MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            asset_bytes: gib_to_bytes(asset_gb),
            folded_control_bytes: 0,
            folded_adapter_bytes: 0,
            activation_headroom_bytes: activation_anchor_bytes.saturating_add(fixed_reserve_bytes),
            fixed_reserve_bytes,
            calibration: MlxCalibrationConfig::Absent,
            load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
        }
    };
    let dense = plan(HeadroomAllowance::LENS_DENSE);
    let peak_gb = |plan: &MlxRequestPlan, width, height| {
        plan.generic_total_peak_bytes(request_geometry(&request_inputs(width, height, 1))) as f64
            / BYTES_PER_GIB
    };
    for (width, height, megapixels) in [
        (1024, 1024, 1.0),
        (1024, 1536, 1.5),
        (1152, 2048, 2.25),
        (2048, 2048, 4.0),
    ] {
        let expected = asset_gb + 2.0 + LENS_DENSE_HEADROOM_GB * megapixels;
        assert!(
                (peak_gb(&dense, width, height) - expected).abs() < 1e-6,
                "{width}x{height}: a bare-transient allowance must scale WHOLLY with area — \
                 expected fixed 2 + {LENS_DENSE_HEADROOM_GB}*{megapixels} = {expected:.4}, got {:.4}",
                peak_gb(&dense, width, height)
            );
    }

    // Control: the generic allowance, which really does contain a reserve, must NOT behave that
    // way — otherwise this test would pass against a build that had removed the split entirely.
    let generic = plan(HeadroomAllowance::GENERIC);
    let generic_whole = asset_gb + (HEADROOM_GB - 2.0) * 2.25;
    assert!(
        peak_gb(&generic, 1152, 2048) < generic_whole - 1e-6,
        "the generic allowance must still hold its reserve out of the area term"
    );
}

#[test]
fn request_scope_charges_external_committed_memory_and_never_accepts_zero_zero() {
    let generator = request_generator(Some(mage_request_contract()));
    let plan = request_plan();
    let inputs = request_inputs(512, 512, 1);
    let external = evaluate_request_with_budget(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Warm,
        OffloadPolicy::Resident,
        MemoryBudget {
            total_bytes: gib_to_bytes(10.0),
            committed_bytes: gib_to_bytes(8.0),
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        gib_to_bytes(8.0),
        gib_to_bytes(7.0),
        &[],
    );
    assert!(
        external.is_err(),
        "seven GiB of unrelated active memory must remain charged"
    );

    let overcommitted = evaluate_request_with_budget(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Warm,
        OffloadPolicy::Resident,
        MemoryBudget {
            total_bytes: gib_to_bytes(10.0),
            committed_bytes: gib_to_bytes(12.0),
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        gib_to_bytes(8.0),
        0,
        &[],
    );
    assert!(
        overcommitted.is_err(),
        "an overcommitted process must not pass through a saturated 0 <= 0 comparison"
    );
}

#[test]
fn legacy_provider_credits_only_the_known_generator_assets_on_a_warm_run() {
    let plan = MlxRequestPlan {
        engine_id: "flux_dev",
        model_id: "flux_dev".to_owned(),
        tier: request_plan().tier,
        asset_bytes: gib_to_bytes(6.0),
        folded_control_bytes: 0,
        folded_adapter_bytes: 0,
        // Deliberately ABOVE the 2 GiB fixed reserve so the area term is non-zero: a fixture
        // sitting exactly on the reserve would model resolution-blind and silently stop
        // exercising the sc-16195 scaling at all.
        activation_headroom_bytes: gib_to_bytes(6.0),
        fixed_reserve_bytes: gib_to_bytes(2.0),
        calibration: MlxCalibrationConfig::Absent,
        load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
    };
    let selected = evaluate_request_with_budget(
        &request_generator(None),
        &plan,
        &request_inputs(512, 512, 1),
        MemoryCacheState::Warm,
        OffloadPolicy::Resident,
        MemoryBudget {
            total_bytes: gib_to_bytes(11.0),
            committed_bytes: gib_to_bytes(6.0),
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        gib_to_bytes(8.0),
        gib_to_bytes(1.0),
        &[],
    )
    .unwrap();

    assert_eq!(
        selected.context.predicted_peak_bytes,
        gib_to_bytes(8.0),
        "the full peak and normalized budget use the same accounting domain"
    );
    assert_eq!(
        selected.context.budget.committed_bytes,
        gib_to_bytes(1.0),
        "only the unrelated GiB stays charged"
    );
}

#[test]
fn generic_floor_keeps_fixed_reserve_without_padding_it_as_activation() {
    let generator = fixture_generator();
    let contract = generator.contract.as_ref().unwrap();
    let plan = fixture_plan();
    for (edge, activation_gb) in [(1024, 4.0), (2048, 16.0)] {
        let geometry = MemoryGeometry {
            width: edge,
            height: edge,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let ladder = synthesize_estimate_ladder(
            contract,
            &plan,
            "text_to_image",
            None,
            geometry,
            false,
            None,
            &[],
        );
        let estimate = ladder
            .estimates
            .iter()
            .find(|candidate| candidate.selection.strategy == MemoryStrategy::StagedResidency)
            .unwrap();
        assert_eq!(
            estimate.unmodeled_activation_bytes,
            Some(gib_to_bytes(activation_gb))
        );
        let weights = estimate_floor_weights_bytes(
            contract,
            &contract.engaged_composition_for_selection(&estimate.selection),
        );
        assert_eq!(
            estimate.evidence.predicted_peak_bytes,
            weights + gib_to_bytes(activation_gb + 2.0)
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires installed SenseNova weights and exclusive Apple/Metal access"]
fn installed_sensenova_infographic_request_is_admitted_and_renders() {
    let root = std::env::var_os("SENSENOVA_ROOT").expect("SENSENOVA_ROOT");
    let requested_route = std::env::var("SENSENOVA_ROUTE").expect("SENSENOVA_ROUTE");
    let route = [
        "sensenova_u1_8b_infographic_v2",
        "sensenova_u1_8b_infographic_v2_fast",
        "sensenova_u1_8b_infographic_v3",
        "sensenova_u1_8b_infographic_v3_fast",
    ]
    .into_iter()
    .find(|route| *route == requested_route)
    .expect("infographic route");
    let engine = if route.ends_with("_fast") {
        "sensenova_u1_8b_fast"
    } else {
        "sensenova_u1_8b"
    };
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let entry = manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == route)
        .unwrap()
        .as_object()
        .unwrap();
    use crate::memory_route_registry::{MemoryRouteMode, MemoryRouteRequestContext};
    let mode = MemoryRouteMode::TextToImage;
    let context = MemoryRouteRequestContext {
        mode,
        reference_count: 0,
        use_pid: false,
        has_phases: false,
    };
    let spec = LoadSpec::new(WeightsSource::Dir(root.into())).with_resolved_route(route);
    let spec = crate::memory_route_registry::evaluate_declared_mlx_load_shape_for_request(
        route,
        Some("bf16"),
        Some(mode),
        entry,
        spec,
        context,
    );
    let mut spec = crate::memory_route_registry::apply_declared_mlx_load_policy_for_request(
        route,
        Some("bf16"),
        Some(mode),
        entry,
        spec,
        context,
    );
    if spec.load_shape_declaration_result == LoadShapeDeclarationResult::NotEvaluated {
        spec = crate::memory_route_registry::apply_registered_load_shape(
            crate::memory_route_registry::MemoryRouteBackend::Mlx,
            engine,
            mode,
            spec,
            false,
        );
    }
    let contract = runtime_macos::providers::sensenova::memory_strategy::memory_strategy_contract(
        engine, &spec,
    )
    .unwrap();
    assert_eq!(
        Some(
            contract
                .calibration
                .as_ref()
                .expect("installed production identity")
                .fingerprint
                .clone()
        ),
        runtime_macos::providers::sensenova::memory_strategy::production_calibration_fingerprint(
            engine, &spec
        )
    );
    assert!(
        contract.conformance_errors().is_empty(),
        "{:?}",
        contract.conformance_errors()
    );
    let candidate = RequestGenerator {
        descriptor: crate::inference_runtime::media_descriptor(engine)
            .unwrap()
            .clone(),
        contract: Some(contract),
    };
    let plan = MlxRequestPlan::for_spec_and_manifest(engine, route, &spec, Some(entry), None);
    let inputs = MlxRequestInputs {
        native_batch: 1,
        count: 2,
        ..fixture_inputs(2048, 2048)
    };
    let evaluation = evaluate_request_with_budget(
        &candidate,
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        spec.offload_policy,
        MemoryBudget {
            total_bytes: gib_to_bytes(128.0),
            committed_bytes: 28,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: gib_to_bytes(2.0),
        },
        request_total_peak_bytes(&plan, request_geometry(&inputs)),
        0,
        &[],
    )
    .expect("the reported infographic request must enter a valid memory strategy");
    assert!(
        evaluation.process_limit_bytes.is_none(),
        "historical identities must not claim a current measured ceiling"
    );
    assert_ne!(
        evaluation.context.optimization_authority,
        MemoryOptimizationAuthority::Calibrated
    );
    assert_eq!(
        evaluation.context.geometry.batch, 1,
        "job outputs run serially"
    );
    eprintln!(
        "{route}: {:?}, {:.2} GiB",
        evaluation.context.selection,
        evaluation.context.predicted_peak_bytes as f64 / BYTES_PER_GIB
    );
    if std::env::var_os("SENSENOVA_RENDER").is_none() {
        return;
    }
    let generator = crate::inference_runtime::load(engine, &spec).unwrap();
    for index in 0..2 {
        let mut scope = generator
            .begin_memory_strategy_request(&evaluation.context)
            .unwrap()
            .unwrap();
        let mut request = gen_core::GenerationRequest {
                prompt: "A simple infographic explaining the water cycle with clouds, rain, a lake and clear arrows".into(),
                width: 2048, height: 2048, count: 1, seed: Some(1234 + index),
                steps: Some(2), memory: Some(evaluation.memory), ..Default::default()
            };
        scope.configure_request(&mut request).unwrap();
        let output = generator
            .generate(&request, &mut |event| {
                if matches!(
                    event,
                    gen_core::Progress::Step { .. } | gen_core::Progress::Decoding
                ) {
                    eprintln!("{route} image {index}: {event:?}");
                }
            })
            .unwrap();
        let gen_core::GenerationOutput::Images(images) = output else {
            panic!("expected images")
        };
        assert_eq!(images.len(), 1);
        let image = &images[0];
        assert_eq!((image.width, image.height), (2048, 2048));
        assert!(image.pixels.iter().any(|pixel| *pixel != image.pixels[0]));
        if let Some(dir) = std::env::var_os("SENSENOVA_OUTPUT_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            image::save_buffer(
                dir.join(format!("{route}-{index}.png")),
                &image.pixels,
                image.width,
                image.height,
                image::ColorType::Rgb8,
            )
            .unwrap();
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires installed Flux2 Dev bf16 metadata; no tensors are loaded"]
fn installed_flux2_dev_request_uses_the_staged_ladder() {
    let root = std::env::var_os("FLUX2_DEV_DIR").expect("FLUX2_DEV_DIR");
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let entry = manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == "flux2_dev")
        .unwrap()
        .as_object()
        .unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir(root.into()))
        .with_resolved_route("flux2_dev")
        .with_offload_policy(OffloadPolicy::Resident);
    use crate::memory_route_registry::{MemoryRouteMode, MemoryRouteRequestContext};
    let mode = MemoryRouteMode::TextToImage;
    let context = MemoryRouteRequestContext {
        mode,
        reference_count: 0,
        use_pid: false,
        has_phases: false,
    };
    let spec = crate::memory_route_registry::evaluate_declared_mlx_load_shape_for_request(
        "flux2_dev",
        Some("bf16"),
        Some(mode),
        entry,
        spec,
        context,
    );
    let mut spec = crate::memory_route_registry::apply_declared_mlx_load_policy_for_request(
        "flux2_dev",
        Some("bf16"),
        Some(mode),
        entry,
        spec,
        context,
    );
    if spec.load_shape_declaration_result == gen_core::LoadShapeDeclarationResult::NotEvaluated {
        spec = crate::memory_route_registry::apply_registered_load_shape(
            crate::memory_route_registry::MemoryRouteBackend::Mlx,
            "flux2_dev",
            mode,
            spec,
            false,
        );
    }
    assert_ne!(
        spec.load_shape_declaration_result,
        gen_core::LoadShapeDeclarationResult::Refused
    );
    let contract =
        runtime_macos::providers::flux2::memory_strategy::registered_dev_t2i_contract(&spec)
            .unwrap();
    let generator = RequestGenerator {
        descriptor: crate::inference_runtime::media_descriptor("flux2_dev")
            .unwrap()
            .clone(),
        contract: Some(contract),
    };
    let plan =
        MlxRequestPlan::for_spec_and_manifest("flux2_dev", "flux2_dev", &spec, Some(entry), None);
    let evaluate = |count| {
        let inputs = MlxRequestInputs {
            native_batch: 1,
            count,
            ..fixture_inputs(1024, 1024)
        };
        evaluate_request_with_budget(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Warm,
            spec.offload_policy,
            MemoryBudget {
                total_bytes: gib_to_bytes(128.0),
                committed_bytes: gib_to_bytes(44.72),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: gib_to_bytes(2.0),
            },
            request_total_peak_bytes(
                &plan,
                MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
            ),
            0,
            &[],
        )
        .unwrap()
    };
    let one = evaluate(1);
    let two = evaluate(2);
    assert_eq!(
        one.context.selection.strategy,
        MemoryStrategy::StagedResidency
    );
    assert!(one.memory.stage_residency);
    assert_eq!(
        one.context.predicted_peak_bytes,
        two.context.predicted_peak_bytes
    );
    assert_eq!(two.context.geometry.batch, 1);
    eprintln!(
        "installed Flux2 Dev selected {:?}, full peak {:.2} GiB",
        one.context.selection.strategy,
        one.context.predicted_peak_bytes as f64 / BYTES_PER_GIB
    );
}

/// Epic 25899 (E6), through the real MLX request planner: the prefix KV cache is added
/// rung-for-rung — it never changes which rung a request runs on — and pricing it never turns
/// an admissible request into a refusal.
///
/// The same request is admitted twice at every budget on a scan from roomy to over-full — once
/// as a provider with no cache (`None`, today's admission) and once with a cache surcharge — so
/// each budget is graded against its own shed twin:
///
/// * at EVERY admitted budget the priced admission runs the shed twin's rung;
/// * (b) that rung also fits cached → `Cached` on it, written onto the memory block, with the
///   context's peak the shed peak PLUS the surcharge;
/// * (a) it fits only shed → `Prefill` on it, byte-for-byte the no-cache admission — including
///   at a budget where the cache WOULD fit on a lower rung (proven by the same request with that
///   rung withdrawn, which then runs `Cached` one rung down);
/// * (c) fits neither → refused with the no-cache admission's own diagnostic.
///
/// *Mutations that red this:* letting the cached pass pick any rung (cached-first over the whole
/// ladder: the (a) budget then runs `Cached` on the lower rung); skipping the cached attempt
/// (the `Cached` band disappears); writing `Cached` without the surcharge in the context;
/// dropping `write_onto`.
#[test]
fn mlx_prefix_kv_cache_keeps_the_shed_rung_and_never_refuses_an_admissible_request() {
    // A two-rung ladder whose staged rung is far below resident (it sheds 50 GiB of
    // conditioning weights), so a budget exists where the cache fits staged but not resident.
    let ladder = || {
        let mut generator = fixture_generator();
        let contract = generator.contract.as_mut().expect("fixture contract");
        contract.asset_facts = gen_core::MemoryAssetFacts {
            base_bytes: gib_to_bytes(100.0),
            conditioning_bytes: gib_to_bytes(50.0),
            transformer_bytes: gib_to_bytes(49.0),
            decoder_bytes: gib_to_bytes(1.0),
            overlay_bytes: 0,
        };
        for capability in &mut contract.strategies {
            if !matches!(
                capability.strategy,
                MemoryStrategy::Resident | MemoryStrategy::StagedResidency
            ) {
                capability.support = gen_core::MemoryStrategySupport::Missing;
            }
        }
        generator
    };
    let mut plan = MlxRequestPlan {
        calibration: MlxCalibrationConfig::Absent,
        ..fixture_plan()
    };
    plan.asset_bytes = gib_to_bytes(100.0);
    plan.activation_headroom_bytes = gib_to_bytes(18.0);
    plan.fixed_reserve_bytes = gib_to_bytes(4.0);
    let inputs = fixture_inputs(1024, 1024);
    let surcharge = gib_to_bytes(3.0);
    let total_peak = gib_to_bytes(130.0);
    let run = |generator: &RequestGenerator, committed: u64, surcharge: Option<u64>| {
        select_request_with_budget_using_bundle(
            MlxAdmissionPhase::Generation,
            generator.descriptor.capabilities.component_precision_floors,
            generator.contract.as_ref(),
            surcharge,
            &|_| None,
            |base| {
                generator
                    .contract
                    .as_ref()
                    .expect("fixture contract")
                    .predicted_peak_from_base(base)
            },
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
            MemoryBudget {
                total_bytes: gib_to_bytes(192.0),
                committed_bytes: committed,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: gib_to_bytes(2.0),
            },
            total_peak,
            // Everything committed is an unrelated allocation, charged in full.
            committed,
            &[],
            None,
            None,
        )
        .expect("the fixture request is structurally admissible")
    };
    let without = |strategy: MemoryStrategy| {
        let mut generator = ladder();
        for capability in &mut generator
            .contract
            .as_mut()
            .expect("fixture contract")
            .strategies
        {
            if capability.strategy == strategy {
                capability.support = gen_core::MemoryStrategySupport::Missing;
            }
        }
        generator
    };
    let generator = ladder();
    let (mut cached, mut prefill, mut refused, mut lower_rung_would_cache) = (0, 0, 0, 0);
    for step in 0..=256u64 {
        let committed = gib_to_bytes(0.5) * step;
        match (
            run(&generator, committed, None),
            run(&generator, committed, Some(surcharge)),
        ) {
            (MlxRequestAdmission::Rejected(shed), MlxRequestAdmission::Rejected(priced)) => {
                assert_eq!(
                    priced.to_string(),
                    shed.to_string(),
                    "(c) a request that fits neither is refused exactly as without the cache"
                );
                refused += 1;
            }
            (MlxRequestAdmission::Admitted(shed), MlxRequestAdmission::Admitted(priced)) => {
                assert_eq!(shed.prefix_kv_cache, None, "no surcharge, no plan");
                assert_eq!(
                    priced.context.selection, shed.context.selection,
                    "the cache never changes which rung a request runs on (at {committed})"
                );
                let decision = priced
                    .prefix_kv_cache
                    .expect("a priced request records its plan");
                assert_eq!(decision.priced_bytes, surcharge);
                match decision.mode {
                    gen_core::PrefixKvCacheMode::Cached => {
                        assert_eq!(
                            priced.memory.prefix_kv_cache,
                            Some(gen_core::PrefixKvCacheMode::Cached),
                            "(b) the admitted selection rides the memory block to the provider"
                        );
                        assert_eq!(
                            priced.context.predicted_peak_bytes,
                            shed.context.predicted_peak_bytes + surcharge,
                            "(b) the context states the cached demand the admission approved"
                        );
                        cached += 1;
                    }
                    gen_core::PrefixKvCacheMode::Prefill => {
                        assert_eq!(priced.memory, shed.memory, "prefill writes nothing");
                        assert_eq!(
                            priced.context, shed.context,
                            "a shed admission is the no-cache admission, byte for byte"
                        );
                        // (a): would the cache have fitted one rung down?
                        if let MlxRequestAdmission::Admitted(lower) = run(
                            &without(shed.context.selection.strategy),
                            committed,
                            Some(surcharge),
                        ) {
                            if lower.prefix_kv_cache.map(|plan| plan.mode)
                                == Some(gen_core::PrefixKvCacheMode::Cached)
                            {
                                assert_ne!(lower.context.selection, shed.context.selection);
                                lower_rung_would_cache += 1;
                            }
                        }
                        prefill += 1;
                    }
                }
            }
            (MlxRequestAdmission::Admitted(_), MlxRequestAdmission::Rejected(error)) => {
                panic!("pricing the cache refused an admissible request at {committed}: {error}")
            }
            (MlxRequestAdmission::Rejected(error), MlxRequestAdmission::Admitted(_)) => {
                panic!("pricing the cache admitted what the shed demand refuses at {committed}: {error}")
            }
        }
    }
    assert!(
        cached > 0 && prefill > 0 && refused > 0 && lower_rung_would_cache > 0,
        "the scan must cross every band: cached={cached} prefill={prefill} refused={refused} \
             prefill-where-a-lower-rung-would-cache={lower_rung_would_cache}"
    );
    // A provider that prices no cache is untouched at every budget (`None` above is that run).
    match run(&generator, 0, None) {
        MlxRequestAdmission::Admitted(evaluation) => {
            assert_eq!(evaluation.memory.prefix_kv_cache, None);
            assert_eq!(evaluation.prefix_kv_cache, None);
        }
        MlxRequestAdmission::Rejected(error) => panic!("roomy budget refused: {error}"),
    }
}

/// Epic 25899, through the PRODUCTION pricing site (`evaluate_request_with_budget_using_bundle`,
/// the call `evaluate_request` makes) with the LINKED `qwen_image_2_1_turbo` descriptor: an
/// eligible request is priced at exactly the provider's own
/// `derived::prefix_kv_cache_demand_bytes(tier, w, h, refs, negative)`, and an ineligible one
/// (hires fix, native prompt batch — `MlxRequestInputs::prefix_kv_cache_eligible`) is priced
/// no cache at all: no plan and no `Cached` on the memory block.
///
/// *Mutations that red this:* passing `None` for the surcharge at the production site (no
/// plan); dropping the eligibility filter (the hires request gets a plan); pricing a different
/// tier, geometry, reference count or branch flag (the byte equality).
#[cfg(target_os = "macos")]
#[test]
fn mlx_production_pricing_site_prices_the_linked_turbo_cache_only_for_an_eligible_request() {
    use runtime_macos::providers::qwen_image_2_1::{memory_strategy::derived, quant::Tier};
    let descriptor = crate::inference_runtime::media_descriptor("qwen_image_2_1_turbo")
        .expect("the linked MLX registry carries qwen_image_2_1_turbo");
    assert!(
        descriptor
            .capabilities
            .execution
            .prefix_kv_cache
            .accepts(gen_core::PrefixKvCacheMode::Cached),
        "the linked Turbo provider declares the cache"
    );
    let mut generator = RequestGenerator {
        descriptor,
        ..fixture_generator()
    };
    // The fixture ladder, conformed to the provider it now stands in for.
    generator
        .contract
        .as_mut()
        .expect("fixture contract")
        .provider_id = "qwen_image_2_1_turbo".to_owned();
    let generator = generator;
    let plan = MlxRequestPlan {
        engine_id: "qwen_image_2_1_turbo",
        calibration: MlxCalibrationConfig::Absent,
        ..fixture_plan()
    };
    let evaluate = |inputs: &MlxRequestInputs| {
        evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
            MemoryBudget {
                total_bytes: gib_to_bytes(512.0),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: gib_to_bytes(2.0),
            },
            plan.generic_total_peak_bytes(request_geometry(inputs)),
            0,
            &[],
            None,
            None,
        )
        .expect("a roomy budget admits the request")
    };
    let eligible = MlxRequestInputs {
        reference_count: 2,
        has_reference: true,
        negative_branch: true,
        prefix_kv_cache_eligible: true,
        ..fixture_inputs(1024, 768)
    };
    let priced = evaluate(&eligible)
        .prefix_kv_cache
        .expect("an eligible Turbo request is priced for its cache");
    assert_eq!(
        priced.priced_bytes,
        derived::prefix_kv_cache_demand_bytes(Tier::Q4, 1024, 768, 2, true),
        "priced from the linked provider at the request's own tier, geometry, refs and branches"
    );
    assert_eq!(
        priced.mode,
        gen_core::PrefixKvCacheMode::Cached,
        "a roomy budget caches"
    );

    let hires = evaluate(&MlxRequestInputs {
        prefix_kv_cache_eligible: false,
        ..eligible.clone()
    });
    assert_eq!(
        hires.prefix_kv_cache, None,
        "a hires request is priced no cache"
    );
    assert_eq!(
        hires.memory.prefix_kv_cache, None,
        "and no cached selection reaches either hires pass"
    );
}

/// sc-25909, through the PRODUCTION pricing site with the LINKED Qwen-Image 2.1 descriptors: a
/// reference request's rungs carry the generic area-scaled headroom as their activation, which
/// at ≤1024² with many references on a packed tier is BELOW the provider's own cache-shed
/// transient (`derived::request_transient_budget_bytes`). OWNER-APPROVED E6 EXCEPTION
/// (2026-10-16): the shed ladder itself — prefill and cached paths alike — now grades each such
/// rung at max(generic activation slice, derived transient) + the fixed reserve, so a request
/// admitted before the fix may shed to a further rung or be refused. The correction itself never
/// changes the tier, the references or the size; a request with no explicit tier pick whose
/// requested tier no longer fits the (corrected) tier probe goes through the pre-existing
/// downtier ladder (`downtier_candidate_tiers`, bounded by `min_quality_floor`), and an explicit
/// tier pick is refused — that ladder is not exercised here.
///
/// Every budget on a scan from roomy to over-full is admitted three ways: the pre-fix shed
/// admission (no transient seam — exactly the pre-fix ladder), the corrected shed admission (the
/// seam, no cache), and the production site (seam + cache). The pre-load tier probe
/// (`preflight_request_with_contract`, the production probe past its registry lookup) is also
/// run at every budget, with and without the seam. At every budget:
///
/// * (a) q8/q4 at ten references: the corrected admission never admits what the pre-fix one
///   refused; on the fixture weights (3 GiB, all transformer — every rung floors alike) a band
///   exists where pre-fix admitted and the corrected ladder refuses (`newly_refused`); on
///   Qwen-shaped weights (20 GiB DiT + 8 GiB text encoder, `StagedResidency` implemented) a band
///   also exists where both admit but the corrected ladder sheds to a FURTHER rung
///   (`shed_further`); every refusal is the existing "needs … GiB for the complete pipeline"
///   reason. The cache is asserted taken only on the fixture-weights scan;
/// * (probe) the tier probe's admit/refuse equals the loaded gate's at every budget, and a band
///   exists where the pre-fix probe (`&|_| None`) admits a tier the corrected probe refuses;
/// * (b) where the provider's transient is under the generic slice (bf16 at ten references, q4
///   with no reference) production is the pre-fix admission byte for byte, refusals included;
/// * (c) production runs the corrected shed rung; a `Prefill` is the corrected shed admission
///   byte for byte, and a `Cached` records `predicted_peak_bytes` = corrected shed peak + cache
///   surcharge — the figure it was graded at, with the shortfall counted once; the corrected
///   shed peak of a rung the pre-fix ladder also ran is that peak + the rung's own shortfall.
///
/// (d) measured / anchor-derived / fitted-curve / law-residue rungs are NOT reached by this
/// fixture: production passes no additional evidence, the qwen reference route has no packaged
/// cells (so no `Evidence` path) and reference geometry skips the anchor, law and phase arms.
/// Their exclusion is structural (`candidate_charges_generic_headroom` is `false` for them).
///
/// *Mutations that red this:* dropping the shed-ladder correction (no tightened band; the
/// corrected-peak equality); keeping the old cached-only correction as well (the cached peak
/// equality); charging the whole derived transient instead of its shortfall (the bf16 / no-ref
/// byte identity); dropping the probe's seam (probe/gate disagreement); recording a peak other
/// than the graded candidate's (the corrected-peak and cached-peak equalities).
#[cfg(target_os = "macos")]
#[test]
fn mlx_prefix_kv_cache_shed_ladder_prices_the_provider_transient_above_the_generic_headroom() {
    use gen_core::Generator as _;
    /// Per-scan tallies. `shed_further`: both ladders admit, on different rungs.
    /// `newly_refused`: the pre-fix ladder admits, the corrected one refuses. `probe_tightened`:
    /// the pre-fix tier probe admits, the corrected probe refuses.
    #[derive(Debug, Default)]
    struct ScanCounts {
        cached: u32,
        shed_further: u32,
        newly_refused: u32,
        refused: u32,
        probe_tightened: u32,
    }
    const STEP_GIB: f64 = 1.0;
    const TOTAL_GIB: f64 = 96.0;
    const EXISTING_REFUSAL: &str = "for the complete pipeline but only";
    let scan = |engine_id: &'static str,
                quant: Option<gen_core::Quant>,
                reference_count: u32,
                expect_correction: bool,
                realistic_weights: bool| {
        let descriptor = crate::inference_runtime::media_descriptor(engine_id)
            .expect("the linked MLX registry carries the Qwen-Image 2.1 ids");
        let mut generator = RequestGenerator {
            descriptor,
            ..fixture_generator()
        };
        let fixture_contract = generator.contract.as_mut().expect("fixture contract");
        fixture_contract.provider_id = engine_id.to_owned();
        if realistic_weights {
            // Qwen-Image 2.1-shaped residency: a ~20 GiB DiT, an ~8 GiB text encoder and a
            // small VAE. `StagedResidency` (implemented on the fixture) drops the text encoder
            // from the denoise-phase floor, so the staged rung sits ~8 GiB under resident.
            fixture_contract.asset_facts = gen_core::MemoryAssetFacts {
                base_bytes: gib_to_bytes(28.5),
                conditioning_bytes: gib_to_bytes(8.0),
                transformer_bytes: gib_to_bytes(20.0),
                decoder_bytes: gib_to_bytes(0.5),
                overlay_bytes: 0,
            };
        }
        let generator = generator;
        let contract = generator
            .memory_strategy_contract()
            .expect("fixture contract");
        let mut plan = MlxRequestPlan {
            engine_id,
            calibration: MlxCalibrationConfig::Absent,
            ..fixture_plan()
        };
        plan.tier.quant = quant;
        if realistic_weights {
            plan.asset_bytes = gib_to_bytes(28.5);
        }
        // The production generic figure: 2 GiB fixed reserve + 14 GiB at 1024².
        plan.activation_headroom_bytes = gib_to_bytes(16.0);
        plan.fixed_reserve_bytes = gib_to_bytes(2.0);
        let plan = plan;
        let inputs = MlxRequestInputs {
            reference_count,
            has_reference: reference_count > 0,
            prefix_kv_cache_eligible: true,
            ..fixture_inputs(1024, 1024)
        };
        let geometry = request_geometry(&inputs);
        // The ACTIVATION slice of the generic headroom: its fixed OS/app reserve is held back,
        // not available to the render.
        let generic = plan.generic_headroom_bytes(geometry) - plan.fixed_reserve_bytes;
        let transient = |forced_tile_edge: Option<u32>| {
            mlx_prefix_kv_cache_shed_transient_bytes(engine_id, quant, &inputs, forced_tile_edge)
        };
        let shortfall_of = |rung: &MemorySelection| {
            let tile = contract
                .engages_selection(rung, MemoryStrategy::BoundedDecode)
                .then_some(rung.parameters.decode_tile_edge)
                .flatten();
            transient(tile)
                .expect("the provider prices its cache-shed transient")
                .saturating_sub(generic)
        };
        let derived = transient(None).expect("the provider prices its cache-shed transient");
        assert_eq!(
            derived > generic,
            expect_correction,
            "{engine_id} {quant:?} refs {reference_count}: derived {:.2} GiB vs generic {:.2} GiB",
            derived as f64 / BYTES_PER_GIB,
            generic as f64 / BYTES_PER_GIB,
        );
        let surcharge = crate::execution_planner::prefix_kv_cache_surcharge_bytes(
            engine_id,
            &generator.descriptor.capabilities.execution,
            quant,
            inputs.width,
            inputs.height,
            inputs.reference_count,
            inputs.negative_branch,
        )
        .expect("the linked provider prices its cache");
        let total_peak = plan.generic_total_peak_bytes(geometry);
        let budget = |committed| MemoryBudget {
            total_bytes: gib_to_bytes(TOTAL_GIB),
            committed_bytes: committed,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: gib_to_bytes(2.0),
        };
        let select =
            |committed: u64, surcharge: Option<u64>, seam: &dyn Fn(Option<u32>) -> Option<u64>| {
                select_request_with_budget_using_bundle(
                    MlxAdmissionPhase::Generation,
                    generator.descriptor.capabilities.component_precision_floors,
                    generator.memory_strategy_contract(),
                    surcharge,
                    seam,
                    |base| generator.predicted_memory_peak_from_base(base),
                    &plan,
                    &inputs,
                    MemoryCacheState::Cold,
                    OffloadPolicy::Resident,
                    crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
                    budget(committed),
                    total_peak,
                    committed,
                    &[],
                    None,
                    None,
                )
                .expect("structurally admissible")
            };
        let production = |committed: u64| match evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
            budget(committed),
            total_peak,
            committed,
            &[],
            None,
            None,
        ) {
            Ok(evaluation) => MlxRequestAdmission::Admitted(Box::new(evaluation)),
            Err(error) => MlxRequestAdmission::Rejected(error),
        };
        let render = |admission: &MlxRequestAdmission| match admission {
            MlxRequestAdmission::Admitted(evaluation) => format!("{evaluation:?}"),
            MlxRequestAdmission::Rejected(error) => error.to_string(),
        };
        // The pre-load tier probe (`preflight_request` past its registry lookup) and the loaded
        // gate it stands in for, on the same budget and the same (probe-shaped, zero external
        // baseline) attribution. `uncorrected_probe` is the probe as it was before the fix.
        let probe_total_peak = request_total_peak_bytes(&plan, geometry);
        let probe = |committed: u64| {
            preflight_request_with_contract(
                generator.memory_strategy_contract(),
                OffloadPolicy::Resident,
                &plan,
                &inputs,
                budget(committed),
                None,
            )
            .expect("structurally admissible")
        };
        let uncorrected_probe = |committed: u64| {
            select_request_with_budget_using_bundle(
                MlxAdmissionPhase::TierProbe,
                declared_component_floors(plan.engine_id),
                generator.memory_strategy_contract(),
                None,
                &|_| None,
                |base| generator.predicted_memory_peak_from_base(base),
                &plan,
                &inputs,
                MemoryCacheState::Cold,
                OffloadPolicy::Resident,
                crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
                budget(committed),
                probe_total_peak,
                0,
                &[],
                None,
                None,
            )
            .expect("structurally admissible")
        };
        let loaded_gate = |committed: u64| {
            evaluate_request_with_budget_using_bundle(
                &generator,
                &plan,
                &inputs,
                MemoryCacheState::Cold,
                OffloadPolicy::Resident,
                crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
                budget(committed),
                probe_total_peak,
                0,
                &[],
                None,
                None,
            )
            .is_ok()
        };
        let mut counts = ScanCounts::default();
        for step in 0..=(TOTAL_GIB / STEP_GIB) as u64 {
            let committed = gib_to_bytes(STEP_GIB) * step;
            let probe_admits = matches!(probe(committed), MlxRequestAdmission::Admitted(_));
            assert_eq!(
                probe_admits,
                loaded_gate(committed),
                "{engine_id} {quant:?} refs {reference_count}: the tier probe and the loaded \
                     gate agree on admit/refuse (at {committed})"
            );
            if !probe_admits
                && matches!(
                    uncorrected_probe(committed),
                    MlxRequestAdmission::Admitted(_)
                )
            {
                counts.probe_tightened += 1;
            }
            let pre_fix_shed = select(committed, None, &|_| None);
            let corrected_shed = select(committed, None, &transient);
            let produced = production(committed);
            if !expect_correction {
                // (b): no shortfall, so the seam changes nothing at all.
                assert_eq!(
                    render(&produced),
                    render(&select(committed, Some(surcharge), &|_| None)),
                    "{engine_id} {quant:?} refs {reference_count}: production is the pre-fix \
                         admission byte for byte (at {committed})"
                );
                assert_eq!(render(&corrected_shed), render(&pre_fix_shed));
            }
            match (&pre_fix_shed, &corrected_shed, &produced) {
                (
                    _,
                    MlxRequestAdmission::Rejected(corrected),
                    MlxRequestAdmission::Rejected(error),
                ) => {
                    assert_eq!(error.to_string(), corrected.to_string());
                    assert!(
                        error.to_string().contains(EXISTING_REFUSAL),
                        "an existing refusal reason (at {committed}): {error}"
                    );
                    if matches!(pre_fix_shed, MlxRequestAdmission::Admitted(_)) {
                        counts.newly_refused += 1;
                    }
                    counts.refused += 1;
                }
                (MlxRequestAdmission::Rejected(error), MlxRequestAdmission::Admitted(_), _) => {
                    panic!(
                        "{engine_id} {quant:?}: the correction admitted what the pre-fix ladder \
                         refused at {committed}: {error}"
                    )
                }
                (
                    MlxRequestAdmission::Admitted(pre),
                    MlxRequestAdmission::Admitted(shed),
                    MlxRequestAdmission::Admitted(corrected),
                ) => {
                    assert_eq!(
                        corrected.context.selection, shed.context.selection,
                        "production runs the corrected shed rung (at {committed})"
                    );
                    if pre.context.selection == shed.context.selection {
                        assert_eq!(
                            shed.context.predicted_peak_bytes,
                            pre.context.predicted_peak_bytes
                                + shortfall_of(&shed.context.selection),
                            "the rung is graded at its own shortfall once (at {committed})"
                        );
                    } else {
                        // Both admitted on a different rung: the corrected ladder shed further.
                        counts.shed_further += 1;
                    }
                    let decision = corrected
                        .prefix_kv_cache
                        .expect("a priced request records its plan");
                    if shortfall_of(&shed.context.selection) > 0 {
                        // (c) the cache is taken exactly when the corrected shed rung + the
                        // surcharge fits: a seam that also carries the surcharge prices this
                        // rung at exactly that figure (and never cheapens an earlier rung), so
                        // its ladder lands on the same rung iff the cached pass does.
                        let shed_plus_cache = select(committed, None, &|tile| {
                            transient(tile).map(|bytes| bytes + surcharge)
                        });
                        assert_eq!(
                            decision.mode == gen_core::PrefixKvCacheMode::Cached,
                            matches!(&shed_plus_cache, MlxRequestAdmission::Admitted(rung)
                                    if rung.context.selection == shed.context.selection),
                            "cached iff the corrected shed rung + the cache fits (at \
                                 {committed})"
                        );
                    }
                    match decision.mode {
                        gen_core::PrefixKvCacheMode::Cached => {
                            assert_eq!(
                                corrected.context.predicted_peak_bytes,
                                shed.context.predicted_peak_bytes + surcharge,
                                "a cached selection records the corrected shed peak + the \
                                     cache it was graded at (at {committed})"
                            );
                            counts.cached += 1;
                        }
                        gen_core::PrefixKvCacheMode::Prefill => {
                            assert_eq!(
                                format!("{:?}", (&corrected.memory, &corrected.context)),
                                format!("{:?}", (&shed.memory, &shed.context)),
                                "a production prefill is the corrected shed admission byte \
                                     for byte (at {committed})"
                            );
                        }
                    }
                }
                _ => panic!(
                    "{engine_id} {quant:?}: production and the corrected shed ladder disagree \
                         on admit/refuse at {committed}"
                ),
            }
        }
        counts
    };
    for engine_id in ["qwen_image_2_1", "qwen_image_2_1_turbo"] {
        for quant in [gen_core::Quant::Q8, gen_core::Quant::Q4] {
            // Fixture weights (3 GiB, all transformer): every rung floors alike, so no rung can
            // absorb the shortfall — the correction only ever refuses here.
            let counts = scan(engine_id, Some(quant), 10, true, false);
            assert!(
                counts.cached > 0
                    && counts.newly_refused > 0
                    && counts.refused > 0
                    && counts.probe_tightened > 0,
                "{engine_id} {quant:?}: {counts:?}"
            );
            // Realistic weights: the staged rung sits a text encoder under resident, so a band
            // exists where the corrected ladder admits on a FURTHER rung instead of refusing.
            let counts = scan(engine_id, Some(quant), 10, true, true);
            assert!(
                counts.shed_further > 0 && counts.newly_refused > 0 && counts.probe_tightened > 0,
                "{engine_id} {quant:?} realistic weights: {counts:?}"
            );
        }
        for (quant, refs) in [(None, 10), (Some(gen_core::Quant::Q4), 0)] {
            let counts = scan(engine_id, quant, refs, false, false);
            assert!(
                counts.cached > 0
                    && counts.shed_further == 0
                    && counts.newly_refused == 0
                    && counts.probe_tightened == 0,
                "{engine_id} {quant:?} refs {refs}: {counts:?}"
            );
        }
    }
}

/// sc-25909: each rung's correction is the provider transient AT THAT RUNG'S forced decode tile.
/// A synthetic seam prices every untiled rung far over budget and the fixture's 512-px
/// `BoundedDecode` tile at `generic + EXTRA`, so only a rung that engages bounded decode with
/// `decode_tile_edge = Some(512)` can be admitted, and its graded (and recorded) peak is the
/// same rung's zero-shortfall peak + exactly `transient(Some(512)) − generic`; the zero-shortfall
/// peak is pinned to the fixture's weights + generic headroom floor.
///
/// *Mutations that red this:* passing `None` as the forced tile edge (every rung is refused);
/// recording any peak other than the graded candidate's (the pinned floor, or the `+ EXTRA`).
#[cfg(target_os = "macos")]
#[test]
fn mlx_shed_transient_correction_prices_each_rung_at_its_own_forced_decode_tile() {
    use gen_core::Generator as _;
    const TILE: u32 = 512;
    let engine_id = "qwen_image_2_1";
    let descriptor = crate::inference_runtime::media_descriptor(engine_id)
        .expect("the linked MLX registry carries the Qwen-Image 2.1 ids");
    let mut generator = RequestGenerator {
        descriptor,
        ..fixture_generator()
    };
    generator
        .contract
        .as_mut()
        .expect("fixture contract")
        .provider_id = engine_id.to_owned();
    let generator = generator;
    let contract = generator
        .memory_strategy_contract()
        .expect("fixture contract");
    let mut plan = MlxRequestPlan {
        engine_id,
        calibration: MlxCalibrationConfig::Absent,
        ..fixture_plan()
    };
    plan.tier.quant = Some(gen_core::Quant::Q8);
    plan.activation_headroom_bytes = gib_to_bytes(16.0);
    plan.fixed_reserve_bytes = gib_to_bytes(2.0);
    let plan = plan;
    let inputs = MlxRequestInputs {
        reference_count: 10,
        has_reference: true,
        ..fixture_inputs(1024, 1024)
    };
    let geometry = request_geometry(&inputs);
    let generic = plan.generic_headroom_bytes(geometry) - plan.fixed_reserve_bytes;
    let extra = gib_to_bytes(1.0);
    let over_budget = gib_to_bytes(1024.0);
    let select = |tiled_transient: u64| {
        select_request_with_budget_using_bundle(
            MlxAdmissionPhase::Generation,
            generator.descriptor.capabilities.component_precision_floors,
            Some(contract),
            None,
            &|forced_tile_edge| {
                Some(if forced_tile_edge == Some(TILE) {
                    tiled_transient
                } else {
                    over_budget
                })
            },
            |base| generator.predicted_memory_peak_from_base(base),
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
            fixture_budget(96.0),
            plan.generic_total_peak_bytes(geometry),
            0,
            &[],
            None,
            None,
        )
        .expect("structurally admissible")
        .into_evaluation()
        .expect("the tiled rung is admitted")
    };
    let at_generic = select(generic);
    let corrected = select(generic + extra);
    let rung = corrected.context.selection;
    assert!(
        contract.engages_selection(&rung, MemoryStrategy::BoundedDecode)
            && rung.parameters.decode_tile_edge == Some(TILE),
        "only the rung tiled at {TILE} px escapes the untiled pricing: {rung:?}"
    );
    assert_eq!(at_generic.context.selection, rung);
    // The recorded peak is the GRADED candidate's: the fixture's 3 GiB (all transformer — no
    // rung sheds weights) + the generic headroom, with no correction at a zero shortfall.
    assert_eq!(
        at_generic.context.predicted_peak_bytes,
        contract.asset_facts.base_bytes + plan.generic_headroom_bytes(geometry),
        "a zero-shortfall rung records its weights + generic headroom floor"
    );
    assert_eq!(
        corrected.context.predicted_peak_bytes,
        at_generic.context.predicted_peak_bytes + extra,
        "the tiled rung is charged transient(Some({TILE})) − generic, not the untiled figure"
    );
}

#[test]
fn resident_loaded_staged_ladder_credits_only_its_own_allocations_before_selection() {
    let mut generator = fixture_generator();
    let contract = generator.contract.as_mut().unwrap();
    contract.asset_facts = gen_core::MemoryAssetFacts {
        base_bytes: gib_to_bytes(100.0),
        conditioning_bytes: gib_to_bytes(50.0),
        transformer_bytes: gib_to_bytes(49.0),
        decoder_bytes: gib_to_bytes(1.0),
        overlay_bytes: 0,
    };
    for capability in &mut contract.strategies {
        if !matches!(
            capability.strategy,
            MemoryStrategy::Resident | MemoryStrategy::StagedResidency
        ) {
            capability.support = gen_core::MemoryStrategySupport::Missing;
        }
    }
    let mut plan = fixture_plan();
    plan.asset_bytes = gib_to_bytes(100.0);
    plan.activation_headroom_bytes = gib_to_bytes(18.0);
    plan.fixed_reserve_bytes = gib_to_bytes(4.0);
    let evaluate = |count, committed, external, peak| {
        let inputs = fixture_inputs(1024, 1024);
        let inputs = MlxRequestInputs { count, ..inputs };
        evaluate_request_with_budget(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Warm,
            OffloadPolicy::Resident,
            MemoryBudget {
                total_bytes: gib_to_bytes(128.0),
                committed_bytes: gib_to_bytes(committed),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: gib_to_bytes(2.0),
            },
            gib_to_bytes(peak),
            gib_to_bytes(external),
            &[],
        )
    };
    for committed in [44.72, 100.0] {
        let one = evaluate(1, committed, 0.0, 130.0).expect("the staged complete pipeline fits");
        let two = evaluate(2, committed, 0.0, 130.0).expect("count remains sequential");
        assert_eq!(
            one.context.selection.strategy,
            MemoryStrategy::StagedResidency
        );
        assert!(one.memory.stage_residency);
        assert_eq!(one.context.predicted_peak_bytes, gib_to_bytes(68.0));
        assert_eq!(one.context.budget.committed_bytes, 0);
        assert_eq!(two.context.geometry.batch, 1);
        assert_eq!(
            one.context.predicted_peak_bytes,
            two.context.predicted_peak_bytes
        );
        assert_eq!(one.context.selection, two.context.selection);
    }
    assert_eq!(
        evaluate(2, 100.0, 0.0, 118.0)
            .unwrap()
            .context
            .selection
            .strategy,
        MemoryStrategy::Resident,
        "normalization must preserve the legacy resident allowance"
    );
    assert!(
        evaluate(2, 60.0, 60.0, 130.0).is_err(),
        "unrelated allocations must remain charged"
    );
}

#[test]
fn production_control_spec_replaces_raw_checkpoint_with_one_typed_residency() {
    use gen_core::{
        MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable,
        MemoryResidentComponent,
    };
    use std::fs::File;

    const BASE_SOURCE_BYTES: u64 = 4_000;
    const CONTROL_SOURCE_BYTES: u64 = 2_000;
    const CONTROL_RESIDENT_BYTES: u64 = 1_000;
    let root_guard = tempfile::Builder::new()
        .prefix("sceneworks-mlx-fit-sc-16065-")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let base = root.join("base.safetensors");
    let control = root.join("control.safetensors");
    File::create(&base)
        .expect("base fixture")
        .set_len(BASE_SOURCE_BYTES)
        .expect("base fixture size");
    File::create(&control)
        .expect("control fixture")
        .set_len(CONTROL_SOURCE_BYTES)
        .expect("control fixture size");
    let spec = LoadSpec::new(WeightsSource::File(base)).with_control(WeightsSource::File(control));
    let mut plan = MlxRequestPlan::for_spec_and_manifest(
        "fixture_provider",
        "fixture_model",
        &spec,
        None,
        None,
    );
    // Isolate asset accounting from the separately-tested activation allowance. Crucially, the
    // plan itself still comes from the production LoadSpec path that folds control into total.
    plan.activation_headroom_bytes = 0;
    plan.fixed_reserve_bytes = 0;
    assert_eq!(plan.asset_bytes, BASE_SOURCE_BYTES + CONTROL_SOURCE_BYTES);
    assert_eq!(plan.folded_control_bytes, CONTROL_SOURCE_BYTES);

    let mut generator = fixture_generator();
    {
        let contract = generator.contract.as_mut().expect("fixture contract");
        let phases = contract.lifecycle.phases.clone();
        contract.asset_facts.base_bytes = BASE_SOURCE_BYTES;
        contract.asset_facts.transformer_bytes = BASE_SOURCE_BYTES;
        contract.asset_facts.overlay_bytes = CONTROL_RESIDENT_BYTES;
        contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::OverlayBytes,
            ],
            resident_components: vec![MemoryResidentComponent {
                id: "fixture.control".to_owned(),
                kind: MemoryComponentKind::ControlBranch,
                resident_bytes: CONTROL_RESIDENT_BYTES,
                bounded_by: None,
                residency: MemoryComponentResidency::WholeRender,
            }],
        };
    }
    let inputs = fixture_inputs(1024, 1024);
    let legacy_total_peak = plan.generic_total_peak_bytes(request_geometry(&inputs));
    assert_eq!(
        legacy_total_peak,
        BASE_SOURCE_BYTES + CONTROL_SOURCE_BYTES,
        "the production legacy estimate includes the raw control checkpoint"
    );

    let cold = evaluate_request_with_budget(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        fixture_budget(1.0),
        legacy_total_peak,
        0,
        &[],
    )
    .expect("the normalized base plus one typed control branch fits");

    assert_eq!(
        cold.context.predicted_peak_bytes,
        BASE_SOURCE_BYTES + CONTROL_RESIDENT_BYTES,
        "the raw control source must be replaced, not double-counted"
    );
    let contract = generator.contract.as_ref().expect("fixture contract");
    // Ask the CONTRACT for the typed decomposition, not the run context (sc-17037).
    //
    // gen-core `8ffa211a` changed `MemoryRunContext::predicted_peak_breakdown` to ignore
    // its contract argument (the parameter is literally `_contract` now) and report the
    // whole scalar as unattributed. That is deliberate and correct: the same bump renamed
    // `MemoryBudget::fits(predicted_peak_bytes)` to `fits(incremental_live_demand_bytes)`,
    // so the context scalar is now INCREMENTAL demand after the caller has already credited
    // request-owned resident bytes — subtracting the provider's formula components from it
    // again would credit the control branch twice.
    //
    // `MemoryProviderContract::decompose_predicted_peak` is unchanged and is where typed
    // attribution still lives, which is what these four assertions are actually about. This
    // fixture zeroes activation headroom and fixed reserve, so the context scalar equals the
    // absolute peak here and the decomposition is exact.
    let breakdown = contract.decompose_predicted_peak(cold.context.predicted_peak_bytes);
    assert_eq!(
        breakdown.predicted_peak_bytes(),
        BASE_SOURCE_BYTES + CONTROL_RESIDENT_BYTES
    );
    assert_eq!(breakdown.unattributed_bytes, BASE_SOURCE_BYTES);
    assert_eq!(breakdown.components.len(), 1);
    assert_eq!(
        breakdown.components[0].kind,
        MemoryComponentKind::ControlBranch
    );
    // Pin the context-level behaviour too. Without this the next gen-core bump could quietly
    // reinstate contract-aware decomposition here — re-introducing exactly the double-credit
    // the upstream change removed — and nothing in this repo would notice: this test is the
    // ONLY caller of `predicted_peak_breakdown` in SceneWorks.
    let context_breakdown = cold.context.predicted_peak_breakdown(contract);
    assert_eq!(
        context_breakdown.unattributed_bytes,
        BASE_SOURCE_BYTES + CONTROL_RESIDENT_BYTES,
        "the run context reports incremental demand whole; it must not re-decompose it"
    );
    assert!(context_breakdown.components.is_empty());

    let warm = evaluate_request_with_budget(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Warm,
        OffloadPolicy::Resident,
        MemoryBudget {
            total_bytes: gib_to_bytes(1.0),
            committed_bytes: contract.total_resident_bytes(),
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        legacy_total_peak,
        0,
        &[],
    )
    .expect("a fully warm base and control branch require no duplicate incremental bytes");
    assert_eq!(
        warm.context.predicted_peak_bytes,
        BASE_SOURCE_BYTES + CONTROL_RESIDENT_BYTES,
        "the complete pipeline peak keeps the base and typed control exactly once"
    );
    assert_eq!(
        warm.context.budget.committed_bytes, 0,
        "provider assets are credited once"
    );

    let legacy = evaluate_request_with_budget(
        &fixture_generator(),
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        fixture_budget(1.0),
        legacy_total_peak,
        0,
        &[],
    )
    .expect("a non-adopting provider preserves the legacy whole-spec estimate");
    assert_eq!(legacy.context.predicted_peak_bytes, legacy_total_peak);
}

#[test]
fn post_load_external_delta_never_reclassifies_or_subtracts_resident_bytes() {
    assert_eq!(add_post_load_external_delta(10, 100, 125), 35);
    assert_eq!(
        add_post_load_external_delta(10, 125, 100),
        10,
        "allocator cleanup cannot reduce the cache-recorded external baseline"
    );
    assert_eq!(add_post_load_external_delta(u64::MAX - 1, 0, 10), u64::MAX);
}

#[test]
fn mage_resident_path_requires_the_provider_calibration_handshake() {
    let error = evaluate_request_with_budget(
        &request_generator(None),
        &request_plan(),
        &request_inputs(512, 512, 1),
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        MemoryBudget {
            total_bytes: gib_to_bytes(16.0),
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        gib_to_bytes(8.0),
        0,
        &[],
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains(&mage_request_fingerprint()),
        "the production Resident path must compare the loaded provider fingerprint: {error}"
    );
}

#[test]
fn mage_resident_path_refuses_a_contract_from_another_tier() {
    // A loaded q8 Mage contract carries a real production identity — but not the cell the q4
    // plan names. The handshake must key on the plan's tier, not merely on "some Mage cell".
    let q8_fingerprint = mage_calibration_fingerprint("mage_flow", Some(gen_core::Quant::Q8))
        .expect("mage_flow q8 is a shipped Mage-Flow cell");
    assert_ne!(q8_fingerprint, mage_request_fingerprint());
    let mut contract = mage_request_contract();
    contract.calibration = Some(gen_core::MemoryCalibrationIdentity::new(
        q8_fingerprint.clone(),
        gen_core::LoadShape::EagerMaterialization,
    ));
    let evaluate = |contract: MemoryProviderContract| {
        evaluate_request_with_budget(
            &request_generator(Some(contract)),
            &request_plan(),
            &request_inputs(512, 512, 1),
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            MemoryBudget {
                total_bytes: gib_to_bytes(64.0),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            gib_to_bytes(8.0),
            0,
            &[],
        )
    };
    let error = evaluate(contract).unwrap_err().to_string();
    assert!(
        error.contains("does not match") && error.contains(&mage_request_fingerprint()),
        "a q8 identity must be refused against a q4 plan by naming the q4 cell: {error}"
    );
    assert!(
        !error.contains(&q8_fingerprint),
        "the refusal names the EXPECTED cell, not the loaded one: {error}"
    );
    // Positive control: the same contract carrying the plan's own cell is admitted.
    evaluate(mage_request_contract())
        .expect("the matching (route, tier) identity passes the Resident-path handshake");
}

#[test]
fn mage_calibration_cells_are_the_eighteen_distinct_shipped_pairs() {
    let routes = [
        "mage_flow",
        "mage_flow_base",
        "mage_flow_turbo",
        "mage_flow_edit",
        "mage_flow_edit_base",
        "mage_flow_edit_turbo",
    ];
    let tiers = [None, Some(gen_core::Quant::Q4), Some(gen_core::Quant::Q8)];
    let mut cells = std::collections::BTreeSet::new();
    for route in routes {
        for tier in tiers {
            let cell = mage_calibration_fingerprint(route, tier)
                .unwrap_or_else(|| panic!("{route} {tier:?} is a shipped cell"));
            assert!(
                cells.insert(cell),
                "{route} {tier:?} collides with another cell"
            );
        }
        assert_eq!(
            mage_calibration_fingerprint(route, Some(gen_core::Quant::Nvfp4)),
            None,
            "{route}: NVFP4 is not a Mage-Flow MLX tier and must stay unnameable"
        );
    }
    assert_eq!(cells.len(), 18);
    assert_eq!(
        mage_calibration_fingerprint("mage_flow_lora", Some(gen_core::Quant::Q4)),
        None,
        "an id the engine does not serve names no cell"
    );
    assert!(!cells.contains("mage-flow-mlx-shared-ladder-2026-08-03-v1"));
}

#[cfg(target_os = "macos")]
#[test]
fn mage_estimator_fingerprint_matches_the_linked_provider_contract() {
    use runtime_macos::providers::mage::model::{production_calibration_fingerprint, MODEL_IDS};
    let tiers = [
        None,
        Some(gen_core::Quant::Q4),
        Some(gen_core::Quant::Q8),
        Some(gen_core::Quant::Nvfp4),
    ];
    for engine_id in MODEL_IDS {
        for tier in tiers {
            assert_eq!(
                mage_calibration_fingerprint(engine_id, tier),
                production_calibration_fingerprint(engine_id, tier),
                "{engine_id} {tier:?}: the worker's Mage peak estimator must fail at test time \
                     when the linked provider identity table moves",
            );
        }
    }
    assert_eq!(
        production_calibration_fingerprint("mage_flow_lora", None),
        None,
        "the engine serves exactly MODEL_IDS; the worker table must not name more"
    );
}

#[test]
fn mage_adapter_requests_require_and_consume_declared_residency() {
    use gen_core::{
        ComponentPrecisionFloor, MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind,
        MemoryFormulaVariable, MemoryResidentComponent, PrecisionFloorComponent,
    };

    let mut inputs = request_inputs(512, 512, 1);
    inputs.adapter_count = 1;
    let root_guard = tempfile::Builder::new()
        .prefix("mage-request-adapter-plan-")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let base = root.join("base.safetensors");
    let adapter = root.join("adapter.safetensors");
    std::fs::write(&base, vec![0_u8; 100]).unwrap();
    std::fs::write(&adapter, vec![0_u8; 5_750]).unwrap();
    let spec = LoadSpec::new(WeightsSource::File(base))
        .with_quant(gen_core::Quant::Q4)
        .with_adapters(vec![gen_core::AdapterSpec::new(
            adapter,
            1.0,
            gen_core::AdapterKind::Lora,
        )]);
    let plan = MlxRequestPlan::for_spec_and_manifest("mage_flow", "mage_flow", &spec, None, None);
    assert_eq!(plan.folded_adapter_bytes, 5_750);
    let error = evaluate_request_with_budget(
        &request_generator(Some(mage_request_contract())),
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        MemoryBudget {
            total_bytes: gib_to_bytes(128.0),
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        gib_to_bytes(8.0),
        0,
        &[],
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("did not declare load-exact adapter residency"));

    const BASE_PEAK_GIB: f64 = 18.73;
    const ADAPTER_GIB: f64 = 5.75;
    let mut contract = mage_request_contract();
    let phases = contract.lifecycle.phases.clone();
    contract.asset_facts.overlay_bytes = gib_to_bytes(ADAPTER_GIB);
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases,
        variables: vec![
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::OverlayBytes,
        ],
        resident_components: vec![MemoryResidentComponent {
            id: "adapter_stack".to_owned(),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: gib_to_bytes(ADAPTER_GIB),
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        }],
    };
    const PRECISION_FLOORS: &[ComponentPrecisionFloor] = &[ComponentPrecisionFloor {
        component: PrecisionFloorComponent::TransformerHead,
        selected_tier: gen_core::Quant::Q4,
        resident_tier: gen_core::Quant::Q8,
    }];
    let mut generator = request_generator(Some(contract.clone()));
    generator.descriptor.capabilities.component_precision_floors = PRECISION_FLOORS;
    let evaluated = evaluate_request_with_budget(
        &generator,
        &plan,
        &inputs,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        MemoryBudget {
            total_bytes: gib_to_bytes(128.0),
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        gib_to_bytes(BASE_PEAK_GIB),
        0,
        &[],
    )
    .expect("load-exact adapter residency narrows Mage's refusal");
    assert_eq!(
        evaluated.context.predicted_peak_bytes,
        gib_to_bytes(BASE_PEAK_GIB).saturating_add(gib_to_bytes(ADAPTER_GIB)),
        "the measured 18.73 GiB base must become 24.48 GiB with additive adapters"
    );
    assert_eq!(
        evaluated.context.selection.tier.component_precision_floors, PRECISION_FLOORS,
        "provider precision floors must participate in the selected evidence identity"
    );

    contract.asset_facts.overlay_bytes = 0;
    if let MemoryFormulaKind::ComponentPhaseEnvelope {
        resident_components,
        ..
    } = &mut contract.formula
    {
        resident_components[0].resident_bytes = 0;
    }
    assert!(
        evaluate_request_with_budget(
            &request_generator(Some(contract)),
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            MemoryBudget {
                total_bytes: gib_to_bytes(128.0),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            gib_to_bytes(BASE_PEAK_GIB),
            0,
            &[],
        )
        .is_err(),
        "mutation guard: zero adapter bytes must restore the refusal"
    );
}

#[test]
fn optimized_fingerprint_mismatch_fails_closed() {
    use gen_core::{
        MemoryCalibrationIdentity, MemoryParameterRanges, MemoryStrategyCapability,
        MemoryStrategyParameters, MemoryStrategySupport,
    };

    let plan = request_plan();
    let inputs = request_inputs(1024, 1024, 1);
    let mut contract = MemoryProviderContract::compatibility_default(
        "mage_flow",
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        mage_request_fingerprint(),
        gen_core::LoadShape::EagerMaterialization,
    ));
    contract.asset_facts.base_bytes = gib_to_bytes(6.0);
    contract.asset_facts.transformer_bytes = gib_to_bytes(6.0);
    contract.lifecycle.decode_tiling = true;
    contract.strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
                MemoryStrategy::StagedResidency => {
                    MemoryStrategySupport::StructurallyNotApplicable {
                        reason: "single resident pipeline".to_owned(),
                    }
                }
                MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
                _ => MemoryStrategySupport::Missing,
            },
            parameters: if strategy == MemoryStrategy::BoundedDecode {
                MemoryParameterRanges {
                    decode_tile_edges: vec![512],
                    decode_overlaps: vec![32],
                    ..Default::default()
                }
            } else {
                Default::default()
            },
        })
        .collect();
    let selection = MemorySelection {
        strategy: MemoryStrategy::BoundedDecode,
        parameters: MemoryStrategyParameters {
            decode_tile_edge: Some(512),
            decode_overlap: Some(32),
            ..Default::default()
        },
        tier: plan.tier,
    };
    let evidence = MemoryEvidence {
        key: MemoryEvidenceKey {
            model_family: "mage_flow".to_owned(),
            resolved_route: "mage_flow".to_owned(),
            backend: gen_core::MemoryBackend::Mlx,
            tier: plan.tier,
            load_shape: gen_core::LoadShape::EagerMaterialization,
            mode: memory_mode_from_mode_key("edit"),
            reference_shape: if inputs.reference_count == 0 {
                MemoryReferenceShape::None
            } else {
                MemoryReferenceShape::Image
            },
            overlay: inputs.overlay.clone(),
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: inputs.reference_count,
            },
            frames_per_second: None,
            strategy: selection.strategy,
            engaged_composition: contract.engaged_composition(selection.strategy),
            parameters: selection.parameters,
        },
        conformance: MemoryConformanceState::Verified,
        dimensions: MemoryEvidenceDimensions {
            static_implementation: MemoryEvidenceVerdict::Satisfied,
            declared_calibration: MemoryEvidenceVerdict::Satisfied,
            historical_verification: MemoryEvidenceVerdict::Satisfied,
            current_environment_verification: MemoryEvidenceVerdict::Satisfied,
            canonical_route_loadability: MemoryEvidenceVerdict::Satisfied,
            exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
        },
        calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: "wrong-provider-fingerprint".to_owned(),
        sceneworks_revision: REQUEST_EVIDENCE_REVISION.to_owned(),
        inference_revision: INFERENCE_REVISION.to_owned(),
        harness_version: "test".to_owned(),
        predicted_peak_bytes: gib_to_bytes(5.0),
        observed_peak_bytes: Some(gib_to_bytes(5.0)),
        parity: MemoryParityContract::Exact,
        parity_result: MemoryParityResult::Passed,
    };
    let error = evaluate_request_with_budget(
        &request_generator(Some(contract)),
        &plan,
        &inputs,
        MemoryCacheState::Warm,
        OffloadPolicy::Resident,
        MemoryBudget {
            total_bytes: gib_to_bytes(10.0),
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        gib_to_bytes(12.0),
        0,
        &[evidence],
    )
    .unwrap_err()
    .to_string();
    // sc-18096 repin. The mismatched record's 5 GiB raw peak fits the 10 GiB budget easily, so
    // ANY error here proves the record was excluded rather than authorizing the fit — the
    // fail-closed property this test owns. The refusal is the honest "no rung fits with its
    // allowance" `Reject`, quoting the smallest admitted requirement — the 12 GiB resident
    // baseline graded on the undeclared (recapture) arm — recomputed from the policy so an
    // allowance re-derivation moves the quote with it.
    let smallest_admitted = format!(
        "needs {:.2} GiB",
        crate::memory_strategy::peak_bytes_to_gb(
            crate::memory_strategy::floor_admitted_peak_bytes(
                gen_core::MemoryBackend::Mlx,
                gib_to_bytes(12.0),
                None,
            )
        )
    );
    assert!(
        error.contains(&smallest_admitted),
        "the mismatched record must not authorize the fit; the refusal must quote the \
             estimate floor ({smallest_admitted}) instead: {error}"
    );
}

/// The dense Mage-Flow-Base DiT (`transformer/diffusion_pytorch_model.safetensors`, bf16 on disk).
const MAGE_DIT_BYTES: u64 = 8_231_536_784;

#[test]
fn full_finetune_tokens_track_the_square_latent_grid() {
    // edge buckets to a multiple of 32, /16 stride, patch_size 1 ⇒ grid² tokens.
    assert_eq!(full_finetune_tokens(1024), 4096.0); // 64²
    assert_eq!(full_finetune_tokens(64), 16.0); // 4²
    assert_eq!(full_finetune_tokens(512), 1024.0); // 32²
                                                   // Sub-32 buckets up to one 32-edge grid (never zero tokens).
    assert_eq!(full_finetune_tokens(16), 4.0); // edge 32 → grid 2 → 4
}

#[test]
fn full_finetune_peak_is_no_signal_without_weights() {
    // Nothing installed ⇒ no signal ⇒ never block (the fit-gate invariant).
    assert_eq!(full_finetune_peak_gb(0, 1024, 1), None);
    assert!(full_finetune_peak_gb(MAGE_DIT_BYTES, 1024, 1).is_some());
}

#[test]
fn full_finetune_rejects_production_resolution_even_on_a_large_mac() {
    // A 1024² full fine-tune of the 4B Mage DiT exceeds even a 128 GB Mac without gradient
    // checkpointing — the exact "don't advertise production-res as fitting" case (sc-14056).
    let msg = full_finetune_fit_message(
        "Mage-Flow Base",
        MAGE_DIT_BYTES,
        1024,
        1,
        Some(MlxMemoryBudget { total_gb: 128.0 }),
    )
    .expect("1024² full-tune must be rejected on 128 GB");
    assert!(
        msg.contains("sc-14989"),
        "message must point at grad-checkpointing: {msg}"
    );
    assert!(
        msg.contains("gradient"),
        "message must name the missing lever: {msg}"
    );
    assert!(
        msg.contains("1024px"),
        "message must name the resolution: {msg}"
    );
    assert!(
        msg.contains("Mage-Flow Base"),
        "message must name the model: {msg}"
    );
}

#[test]
fn full_finetune_permits_a_tiny_resolution_on_a_large_mac() {
    // A tiny-resolution full-tune (the e2e regime) fits a 128 GB Mac — the gate permits it.
    assert_eq!(
        full_finetune_fit_message(
            "Mage-Flow Base",
            MAGE_DIT_BYTES,
            64,
            1,
            Some(MlxMemoryBudget { total_gb: 128.0 }),
        ),
        None
    );
}

#[test]
fn full_finetune_rejects_any_resolution_on_a_small_mac() {
    // The optimizer state alone (~8× the bf16 DiT bytes ≈ 61 GB) exceeds a 32 GB Mac, so a 4B
    // full fine-tune is refused there even at the tiniest resolution — resolution-independent.
    assert!(full_finetune_fit_message(
        "Mage-Flow Base",
        MAGE_DIT_BYTES,
        64,
        1,
        Some(MlxMemoryBudget { total_gb: 32.0 }),
    )
    .is_some());
}

/// sc-14056 review — the gradient-accumulation term. The first cut of this gate called the 8×
/// state multiplier "exact" while omitting the accumulator map `accumulate_grads` holds across
/// the whole window whenever `gradient_accumulation > 1` (a user-exposed Training Studio knob).
/// That under-prediction is the dangerous direction: a 64–80 GB Mac was predicted at ~63 GiB,
/// ADMITTED, and then hard-killed — an MLX overcommit is a SIGKILL nothing downstream can catch.
///
/// This test is written to DISCRIMINATE rather than to restate the formula: it pins a budget in
/// the exact band where the two multipliers disagree, and asserts the accumulation-1 run is still
/// permitted there while the accumulation-4 run — identical in every other respect — is refused.
/// Delete the accumulation term and the second half fails; make the term unconditional and the
/// first half fails.
#[test]
fn full_finetune_accumulation_window_rejects_where_a_single_step_run_fits() {
    // 8× the 7.665 GiB bf16 DiT ≈ 61.3 GiB of state; 10× ≈ 76.7 GiB. At a tiny resolution the
    // activation terms are negligible, so a 72 GB budget sits squarely between them — the band a
    // 64–80 GB Mac lands in, and the band the old formula got wrong.
    const BUDGET: Option<MlxMemoryBudget> = Some(MlxMemoryBudget { total_gb: 72.0 });

    let single = full_finetune_peak_gb(MAGE_DIT_BYTES, 64, 1).expect("weights present");
    let windowed = full_finetune_peak_gb(MAGE_DIT_BYTES, 64, 4).expect("weights present");
    assert!(
        windowed > single,
        "an accumulation window must cost MORE than a single-step run ({windowed} vs {single})"
    );

    assert_eq!(
        full_finetune_fit_message("Mage-Flow Base", MAGE_DIT_BYTES, 64, 1, BUDGET),
        None,
        "accumulation 1 still fits a 72 GB Mac at a tiny resolution — the gate must not become \
             unconditionally stricter"
    );
    let msg = full_finetune_fit_message("Mage-Flow Base", MAGE_DIT_BYTES, 64, 4, BUDGET).expect(
        "accumulation 4 holds an extra full f32 gradient map and must now be REFUSED on a \
                 72 GB Mac — this is the run that previously got an uncatchable SIGKILL",
    );
    assert!(
        msg.contains("Gradient accumulation is 4"),
        "the rejection must name the accumulation lever it can actually act on: {msg}"
    );
    // …and must NOT mention it when accumulation is not what is costing the memory.
    let single_step_reject = full_finetune_fit_message(
        "Mage-Flow Base",
        MAGE_DIT_BYTES,
        64,
        1,
        Some(MlxMemoryBudget { total_gb: 32.0 }),
    )
    .expect("32 GB cannot hold the optimizer state at any resolution");
    assert!(
        !single_step_reject.contains("Gradient accumulation"),
        "at accumulation 1 there is no accumulator buffer, so the advice would be noise: \
             {single_step_reject}"
    );
}

/// `0` and `1` both mean "step every micro-step" to the engine, so neither may pay the
/// accumulator term — a legacy payload that omits the key must not be gated more strictly than
/// one that sets it to 1.
#[test]
fn full_finetune_accumulation_zero_and_one_are_the_same_envelope() {
    assert_eq!(
        full_finetune_peak_gb(MAGE_DIT_BYTES, 64, 0),
        full_finetune_peak_gb(MAGE_DIT_BYTES, 64, 1)
    );
    assert_ne!(
        full_finetune_peak_gb(MAGE_DIT_BYTES, 64, 1),
        full_finetune_peak_gb(MAGE_DIT_BYTES, 64, 2)
    );
}

#[test]
fn full_finetune_never_blocks_without_a_budget() {
    // No platform budget (off-macOS, or an unreadable probe) ⇒ never block, exactly like the
    // generation gate's `Unknown`.
    assert_eq!(
        full_finetune_fit_message("Mage-Flow Base", MAGE_DIT_BYTES, 1024, 1, None),
        None
    );
}

#[test]
fn full_finetune_memory_error_no_signal_when_transformer_absent() {
    // The live entry point permits when the base isn't installed (transformer/ sums to 0) — the
    // base-installed gate is what reports "not installed".
    let empty_guard = tempfile::Builder::new()
        .prefix("mage_full_gate_")
        .tempdir()
        .expect("temp dir");
    let empty = empty_guard.path();
    assert_eq!(
        full_finetune_memory_error(empty, 1024, 1, "Mage-Flow Base"),
        None
    );
}

#[test]
fn parse_memory_cap_accepts_positive_numbers_only() {
    assert_eq!(parse_memory_cap(Some("16")), Some(16.0));
    assert_eq!(parse_memory_cap(Some("  32.5 ")), Some(32.5));
    assert_eq!(parse_memory_cap(Some("0")), None);
    assert_eq!(parse_memory_cap(Some("-8")), None);
    assert_eq!(parse_memory_cap(Some("nan")), None);
    assert_eq!(parse_memory_cap(Some("inf")), None);
    assert_eq!(parse_memory_cap(Some("abc")), None);
    assert_eq!(parse_memory_cap(Some("")), None);
    assert_eq!(parse_memory_cap(None), None);
}

#[test]
fn resolve_budget_prefers_the_emulation_cap() {
    // Cap overrides the real total (emulate a smaller Mac on big hardware).
    assert_eq!(
        resolve_budget(Some(128.0), Some(16.0)),
        Some(MlxMemoryBudget { total_gb: 16.0 })
    );
    // No cap ⇒ the real total.
    assert_eq!(
        resolve_budget(Some(128.0), None),
        Some(MlxMemoryBudget { total_gb: 128.0 })
    );
    // A cap with no real reading still yields a budget (exercisable in a no-probe unit test).
    assert_eq!(
        resolve_budget(None, Some(16.0)),
        Some(MlxMemoryBudget { total_gb: 16.0 })
    );
    // No signal at all ⇒ no budget ⇒ gate no-ops.
    assert_eq!(resolve_budget(None, None), None);
}

#[test]
fn predicted_peak_is_weights_plus_headroom_and_zero_is_no_signal() {
    // 20 GiB of weights ⇒ 20 + headroom.
    let bytes = 20 * 1024 * 1024 * 1024_u64;
    assert_eq!(predicted_peak_gb(bytes), Some(20.0 + HEADROOM_GB));
    // No measurable weights ⇒ no signal.
    assert_eq!(predicted_peak_gb(0), None);
}

/// NOTE (sc-16014): these numbers remain the bf16/MXFP4 calibration; they no longer describe q4/q8,
/// whose re-hosted affine packs eliminated the tier-integrity exception. The ledger and this file
/// share the `sc-16014-resolution: rehosted-q4-q8` marker so a future artifact change must reconcile
/// both sources rather than reviving or erasing the exception silently.
#[test]
fn lens_dense_calibration_covers_the_measured_full_peak_without_blanket_inflation() {
    let gib = BYTES_PER_GIB;
    let lens_materialized = (45.67 * gib).ceil() as u64;
    let lens_peak = predicted_peak_gb_with_headroom(lens_materialized, LENS_DENSE_HEADROOM_GB)
        .expect("lens signal");
    assert!(lens_peak >= 75.55, "{lens_peak} GiB must cover 75.55 GiB");

    // A bf16-on-disk family keeps the generic calculation; no global dense-tier multiplier.
    let ordinary_bf16 = (28.43 * gib).ceil() as u64;
    let ordinary_peak = predicted_peak_gb(ordinary_bf16).expect("ordinary signal");
    assert!((ordinary_peak - 46.43).abs() < 1e-6);
}

#[cfg(target_os = "macos")]
#[test]
fn lens_provider_footprint_distinguishes_storage_format() {
    let root_guard = tempfile::Builder::new()
        .prefix("mlx_fit_gate_sc11924_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let (te_disk, dit_disk, vae_disk) = (256_u64, 272_u64, 288_u64);
    let disk_total = te_disk + dit_disk + vae_disk;
    for (component, bytes) in [
        ("text_encoder", te_disk),
        ("transformer", dit_disk),
        ("vae", vae_disk),
    ] {
        crate::test_fixture_disk::set_sparse_valid_safetensor(
            &root.join(component).join("model.safetensors"),
            bytes,
        )
        .expect("valid structural fixture for the provider's strict header reader");
    }
    std::fs::write(
        root.join("text_encoder").join("config.json"),
        r#"{"dtype":"bfloat16","quantization_config":{"quant_method":"mxfp4"}}"#,
    )
    .expect("mxfp4 marker");
    let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
    let (dense_total, dense_te, dense_headroom) = spec_component_bytes("lens_turbo", &spec);
    let expected_te = (30.07 * BYTES_PER_GIB).ceil() as u64;
    assert_eq!(dense_te, expected_te);
    assert_eq!(dense_total, expected_te + dit_disk + vae_disk);
    assert_eq!(dense_headroom, HeadroomAllowance::LENS_DENSE);
    // sc-16195: a bare measured transient carries NO OS reserve, so the request estimator
    // must leave the whole allowance in its area term for this path.
    assert_eq!(dense_headroom.os_reserve_gb, 0.0);
    for engine_id in ["lens", "lens_turbo"] {
        let plan = MlxRequestPlan::for_spec_and_manifest(engine_id, engine_id, &spec, None, None);
        assert_eq!(plan.fixed_reserve_bytes, gib_to_bytes(2.0));
        assert_eq!(
            plan.activation_headroom_bytes,
            gib_to_bytes(2.0 + LENS_DENSE_HEADROOM_GB),
            "{engine_id} dense/MXFP4 must preserve the format-aware 29.88 GiB activation fallback"
        );
    }

    std::fs::write(
        root.join("text_encoder").join("config.json"),
        r#"{"dtype":"bfloat16"}"#,
    )
    .expect("bf16 marker");
    let (bf16_total, bf16_te, bf16_headroom) = spec_component_bytes("lens_turbo", &spec);
    assert_eq!(
        (bf16_total, bf16_te),
        (disk_total, te_disk),
        "bf16-on-disk has no MXFP4 materialization delta"
    );
    assert_eq!(
        bf16_headroom,
        HeadroomAllowance::LENS_DENSE,
        "the Lens activation transient remains architecture-specific even without a weight upcast"
    );

    std::fs::write(
            root.join("text_encoder").join("config.json"),
            // The re-hosted q4/q8 configs retain this inherited MXFP4 marker. The load-bearing MLX
            // affine marker must win, or the fit gate would inflate the already-packed experts.
            r#"{"quantization":{"bits":8,"group_size":64},"quantization_config":{"quant_method":"mxfp4"}}"#,
        )
        .expect("packed marker with inherited MXFP4 metadata");
    let (packed_total, packed_te, packed_headroom) = spec_component_bytes("lens_turbo", &spec);
    assert_eq!((packed_total, packed_te), (disk_total, te_disk));
    assert_eq!(packed_headroom, HeadroomAllowance::GENERIC);
    for engine_id in ["lens", "lens_turbo"] {
        let plan = MlxRequestPlan::for_spec_and_manifest(engine_id, engine_id, &spec, None, None);
        assert_eq!(plan.fixed_reserve_bytes, gib_to_bytes(2.0));
        assert_eq!(
            plan.activation_headroom_bytes,
            gib_to_bytes(16.0),
            "{engine_id} packed q4/q8 must preserve the generic 14 GiB activation fallback"
        );
    }
}

#[test]
fn fit_decision_rejects_only_a_genuine_overflow() {
    let budget = MlxMemoryBudget { total_gb: 16.0 };
    // qwen-image q8: ~36 GiB weights + 18 headroom (sc-10863) = 54 ⇒ too big for a 16 GB Mac.
    assert_eq!(
        fit_decision(predicted_peak_gb(36 * 1024 * 1024 * 1024_u64), Some(budget)),
        FitDecision::TooBig {
            needed_gb: 36.0 + HEADROOM_GB,
            available_gb: 16.0,
        }
    );
    // A ~3 GiB model fits a roomy budget (3 + 18 headroom = 21 < 32).
    assert_eq!(
        fit_decision(
            predicted_peak_gb(3 * 1024 * 1024 * 1024_u64),
            Some(MlxMemoryBudget { total_gb: 32.0 })
        ),
        FitDecision::Fits
    );
    // Exactly-fits is not a rejection: budget 46, need 46.
    assert_eq!(
        fit_decision(Some(46.0), Some(MlxMemoryBudget { total_gb: 46.0 })),
        FitDecision::Fits
    );
    // Missing either input ⇒ never block.
    assert_eq!(fit_decision(None, Some(budget)), FitDecision::Unknown);
    assert_eq!(fit_decision(Some(8.0), None), FitDecision::Unknown);
}

#[test]
fn sum_safetensors_skips_appledouble_and_nonweights_and_recurses() {
    let root_guard = tempfile::Builder::new()
        .prefix("mlx_fit_gate_sum_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let te = root.join("text_encoder");
    let dit = root.join("transformer");
    std::fs::create_dir_all(&te).expect("mk te");
    std::fs::create_dir_all(&dit).expect("mk dit");
    std::fs::write(te.join("model.safetensors"), vec![0u8; 1000]).expect("te weights");
    std::fs::write(dit.join("diffusion.safetensors"), vec![0u8; 2000]).expect("dit weights");
    // AppleDouble sidecar + a non-weight file must NOT be counted.
    std::fs::write(te.join("._model.safetensors"), vec![0u8; 500]).expect("sidecar");
    std::fs::write(dit.join("config.json"), vec![0u8; 700]).expect("config");

    assert_eq!(sum_safetensors_bytes(root), 3000);
    // Missing dir ⇒ 0 (no signal).
    assert_eq!(sum_safetensors_bytes(&root.join("nope")), 0);
}

/// Adapter factors are resident only when the provider keeps them as additive residuals.
#[test]
fn spec_adapter_bytes_distinguish_additive_and_folded_loads() {
    let root_guard = tempfile::Builder::new()
        .prefix("mlx_fit_gate_adapters_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let weights = root.join("weights");
    std::fs::create_dir_all(&weights).expect("weights dir");
    std::fs::write(weights.join("model.safetensors"), vec![0_u8; 1_000]).expect("base weights");
    let adapter = root.join("adapter.safetensors");
    std::fs::write(&adapter, vec![0_u8; 250]).expect("adapter weights");
    let adapter_spec = gen_core::AdapterSpec::new(adapter, 1.0, gen_core::AdapterKind::Lora);

    let additive = LoadSpec::new(WeightsSource::Dir(weights.clone()))
        .with_adapters(vec![adapter_spec.clone()]);
    assert_eq!(spec_component_bytes("mage_flow", &additive).0, 1_250);
    assert_eq!(
        MlxRequestPlan::for_spec_and_manifest("mage_flow", "mage_flow", &additive, None, None)
            .folded_adapter_bytes,
        250
    );

    let dense_wan = LoadSpec::new(WeightsSource::Dir(weights.clone()))
        .with_quant(gen_core::Quant::Q4)
        .with_adapters(vec![adapter_spec.clone()]);
    assert_eq!(
        spec_component_bytes("wan2_2_t2v_14b", &dense_wan).0,
        1_000,
        "dense Wan folds factors before load-time quantization"
    );
    assert_eq!(
        spec_component_bytes("wan_vace", &dense_wan).0,
        1_000,
        "Wan VACE always folds its dense adapter factors"
    );
    assert_eq!(
        spec_component_bytes("wan2_2_vace_fun_14b", &dense_wan).0,
        1_000,
        "Wan VACE-Fun folds each expert's factors before quantization"
    );

    std::fs::write(
        weights.join("config.json"),
        r#"{"quantization":{"bits":4,"group_size":64}}"#,
    )
    .expect("packed marker");
    let packed_wan = LoadSpec::new(WeightsSource::Dir(weights)).with_adapters(vec![adapter_spec]);
    assert_eq!(
        spec_component_bytes("wan2_2_t2v_14b", &packed_wan).0,
        1_250,
        "pre-packed Wan retains adapter factors as additive residuals"
    );
}

/// sc-15154: a split-layout tier's staged co-requisites are part of what it loads. Mage-Flow's
/// per-tier dir holds the DiT alone; the shared text encoder and VAE must still count.
#[test]
fn a_staged_component_counts_toward_the_tier_that_loads_it() {
    let root_guard = tempfile::Builder::new()
        .prefix("mlx_fit_gate_staged_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let write = |dir: std::path::PathBuf, bytes: usize| {
        std::fs::create_dir_all(&dir).expect("mk dir");
        std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).expect("write");
        dir
    };
    // The variant mirror's tier dir (DiT only) + the shared components mirror.
    let tier = root.join("q4");
    write(tier.join("transformer"), 3_000);
    let te = write(root.join("shared/q4/text_encoder"), 5_000);
    let vae = write(root.join("shared/q4/vae"), 1_000);

    // Nothing staged — the pre-fix reading, and still correct for a flat snapshot.
    let bare = LoadSpec::new(WeightsSource::Dir(tier.clone()));
    assert_eq!(spec_component_bytes("mage_flow_edit", &bare).0, 3_000);

    // Staged the way the load stages them.
    let staged = bare
        .clone()
        .with_component("text_encoder", WeightsSource::Dir(te))
        .with_component("vae", WeightsSource::Dir(vae));
    let (total, te_bytes, _) = spec_component_bytes("mage_flow_edit", &staged);
    println!(
        "mage_flow_edit split tier: bare total={} staged total={total} staged te={te_bytes}",
        spec_component_bytes("mage_flow_edit", &bare).0
    );
    assert_eq!(
        total, 9_000,
        "the staged text encoder and VAE are weights this tier holds resident"
    );
    // The TEXT-ENCODER split comes from the provider's own footprint at the pinned inference
    // revision, which must likewise resolve the STAGED dir and not `<tier>/text_encoder`. A zero
    // here would collapse the sequential schedule's `max(te, rest)` onto the whole model and
    // silently over-reject, so it is asserted rather than inferred from the total.
    //
    // macOS ONLY: the MLX media registry — and therefore any provider footprint — is compiled in
    // on macOS alone. Off-macOS `footprint()` yields `None`, `resolve_text_encoder_bytes` falls
    // back to the diffusers `text_encoder*` subdir scan of the TIER dir, and that is correctly 0
    // here. The `total` assertions above are registry-free and hold on every platform, which is
    // what this test is really for.
    #[cfg(target_os = "macos")]
    assert_eq!(
        te_bytes, 5_000,
        "mage_flow's per-component footprint must follow the staged component dirs"
    );

    // A component that resolves INSIDE the weights dir was already counted by the scan, so it
    // must not be added twice — the flat published layout, staged explicitly.
    let flat_te = write(tier.join("text_encoder"), 700);
    let flat = LoadSpec::new(WeightsSource::Dir(tier))
        .with_component("text_encoder", WeightsSource::Dir(flat_te));
    assert_eq!(spec_component_bytes("mage_flow_edit", &flat).0, 3_700);
}

#[test]
fn external_selected_text_encoder_is_counted_once_across_containment_shapes() {
    let root_guard = tempfile::Builder::new()
        .prefix("mlx_fit_gate_selected_te_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let write = |dir: std::path::PathBuf, bytes: usize| {
        std::fs::create_dir_all(&dir).expect("mk dir");
        std::fs::write(dir.join("model.safetensors"), vec![0_u8; bytes]).expect("write");
        dir
    };
    let base = write(root.join("base"), 1_000);

    // A nested named component is not part of the selected encoder's direct-shard inventory.
    // Both sources are loaded and priced, but only the direct shard belongs to the TE phase.
    let external_te = write(root.join("selected-te"), 500);
    let nested_component = write(external_te.join("projection"), 200);
    let external = LoadSpec::new(WeightsSource::Dir(base.clone()))
        .with_component("projection", WeightsSource::Dir(nested_component))
        .with_text_encoder(WeightsSource::Dir(external_te));
    let (total, te_bytes, _) = spec_component_bytes("unregistered_fixture", &external);
    assert_eq!(total, 1_000 + 500 + 200);
    assert_eq!(te_bytes, 500);

    // A selected source nested under the primary directory is already covered by the base scan.
    let nested_te = write(base.join("selected-te"), 300);
    let nested = LoadSpec::new(WeightsSource::Dir(base.clone()))
        .with_text_encoder(WeightsSource::Dir(nested_te.clone()));
    let (total, te_bytes, _) = spec_component_bytes("unregistered_fixture", &nested);
    assert_eq!(total, 1_000 + 300);
    assert_eq!(te_bytes, 300);

    // Equal is likewise already covered, but the TE phase is only the base directory's direct
    // shard and excludes the nested selected-te descendant.
    let equal =
        LoadSpec::new(WeightsSource::Dir(base.clone())).with_text_encoder(WeightsSource::Dir(base));
    let (total, te_bytes, _) = spec_component_bytes("unregistered_fixture", &equal);
    assert_eq!(total, 1_000 + 300);
    assert_eq!(te_bytes, 1_000);

    // Parent is intentionally asymmetric: the exact selected inventory contains only direct
    // shards, so the nested primary remains separately priced and is never subtracted from TE.
    let selected_parent = write(root.join("selected-parent"), 400);
    let nested_base = write(selected_parent.join("base"), 600);
    let parent = LoadSpec::new(WeightsSource::Dir(nested_base))
        .with_text_encoder(WeightsSource::Dir(selected_parent));
    let (total, te_bytes, _) = spec_component_bytes("unregistered_fixture", &parent);
    assert_eq!(total, 600 + 400);
    assert_eq!(te_bytes, 400);
}

#[test]
fn complete_snapshot_selection_prices_only_resolved_encoder_direct_shards() {
    let root_guard = tempfile::Builder::new()
        .prefix("mlx_fit_gate_selected_snapshot_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let write = |path: std::path::PathBuf, bytes: usize| {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mk dir");
        std::fs::write(&path, vec![0_u8; bytes]).expect("write");
    };
    let base = root.join("base");
    write(base.join("model.safetensors"), 1_000);
    let snapshot = root.join("complete-selected-snapshot");
    write(snapshot.join("text_encoder/model.safetensors"), 500);
    write(snapshot.join("text_encoder/config.json"), 2);
    write(snapshot.join("transformer/model.safetensors"), 700);
    write(snapshot.join("vae/model.safetensors"), 300);

    let spec =
        LoadSpec::new(WeightsSource::Dir(base)).with_text_encoder(WeightsSource::Dir(snapshot));
    let (total, te_bytes, _) = spec_component_bytes("unregistered_fixture", &spec);
    assert_eq!(total, 1_000 + 500);
    assert_eq!(te_bytes, 500);

    // A selection-aware provider owns materialized residency. Its smaller projection must not
    // be overwritten by either the exact stored bytes or the snapshot's unrelated siblings.
    let (_, projected_te, _) = spec_component_bytes_with_provider_footprint(
        "unregistered_fixture",
        &spec,
        Some(PerComponentBytes {
            text_encoder: 450,
            dit: 1_000,
            vae: 0,
        }),
    );
    assert_eq!(projected_te, 450);
}

/// sc-17137 — MiniMax-H3's INVERTED staging must not price the dense upstream it supersedes.
///
/// Unlike Mage-Flow (weights dir = the tier DiT, TE/VAE staged on top), MiniMax-H3 points
/// `spec.weights` at the SHARED UPSTREAM snapshot and stages the tiered DiT and packed encoder in
/// as `components["transformer"]` / `components["text_encoder"]` from a different repo
/// (`video_load_spec`, sc-19508 / sc-19120). That upstream snapshot also physically carries the
/// DENSE `transformer/`, `transformer_ref/` and `text_encoder/`, which a q4/q8 render never loads.
/// The flat `sum_safetensors_bytes(weights)` counted all three — quoting ~250 GB (dense 62+62+62 +
/// VAEs 11 + staged 18+18) for a model whose real q4 resident set is ~65 GB, so a 128 GB Mac
/// refused a model that fits with room to spare (the sc-17137 first-run report). The staged
/// components must SUPERSEDE the same-named upstream subtrees, ref partition included.
#[test]
fn minimax_h3_shared_root_does_not_price_the_dense_partitions_the_tier_supersedes() {
    use crate::test_fixture_disk::create_sparse_weights;
    const GIB: u64 = 1024 * 1024 * 1024;

    let fixture = tempfile::tempdir().expect("minimax fixture");
    // The shared UPSTREAM snapshot `spec.weights` points at: the VAEs a render loads, PLUS the
    // dense DiT partitions and dense encoder the tiered load never touches.
    let base = fixture.path().join("upstream");
    create_sparse_weights(&base.join("transformer/model.safetensors"), 62 * GIB);
    create_sparse_weights(&base.join("transformer_ref/model.safetensors"), 62 * GIB);
    create_sparse_weights(&base.join("text_encoder/model.safetensors"), 62 * GIB);
    create_sparse_weights(&base.join("vae/model.safetensors"), 10 * GIB);
    create_sparse_weights(&base.join("audio_vae/model.safetensors"), GIB);
    // The per-tier rehost the DiT and packed encoder are staged from (a DIFFERENT root).
    let tier = fixture.path().join("tier/q4");
    create_sparse_weights(&tier.join("transformer/model.safetensors"), 18 * GIB);
    create_sparse_weights(&tier.join("text_encoder/model.safetensors"), 18 * GIB);

    let spec = LoadSpec::new(WeightsSource::Dir(base))
        .with_component("transformer", WeightsSource::Dir(tier.join("transformer")))
        .with_component(
            "text_encoder",
            WeightsSource::Dir(tier.join("text_encoder")),
        );

    let (total, te, _) = spec_component_bytes("minimax_h3", &spec);
    // Only what the tiered render actually loads: the two VAEs from upstream plus the staged DiT
    // and packed encoder. The dense 62+62+62 GiB of superseded transformer / transformer_ref /
    // encoder is gone.
    assert_eq!(
        total,
        (10 + 1 + 18 + 18) * GIB,
        "the dense upstream partitions the tier supersedes must not be priced"
    );
    // The resident (sequential-droppable) encoder is the staged packed one, not the dense subtree
    // the dir scan measured — registry-free, so it holds on every platform.
    assert_eq!(
        te,
        18 * GIB,
        "te must follow the staged packed encoder, not the dense upstream"
    );

    // The admission flip this fixes: 47 GiB weights + 18 headroom = 65 fits a 128 GB Mac as
    // Resident; pricing the dense partitions (the pre-fix sum) is what forced the Reject.
    let budget = Some(MlxMemoryBudget { total_gb: 128.0 });
    assert!(
        !matches!(
            decide_residency(total, te, budget, true),
            ResidencyOutcome::Reject { .. }
        ),
        "the real q4 footprint fits 128 GB — the gate must stop refusing it"
    );
    assert!(
        matches!(
            decide_residency((62 * 3 + 11 + 36) * GIB, 62 * GIB, budget, true),
            ResidencyOutcome::Reject { .. }
        ),
        "guard: the pre-fix sum (dense partitions included) is what over-rejected"
    );
}

/// sc-15154 — the Mage q4 admit boundary, from the REAL published install sizes.
///
/// The epic-14034 acceptance run found emulated caps of **both 6 GB and 7 GB admitting** a q4
/// edit job. That was not the memory model being wrong; it was this gate seeing 2.17 GiB of
/// weights (the DiT alone) where the tier installs 6.52 GiB, so the permissive weights-fit floor
/// (`Σstaged ≤ budget − legacy reserve`) cleared caps the tier cannot hold.
///
/// Bytes are the manifest's `estimatedSizeBytes` for `mage_flow_edit` q4:
/// DiT 2,326,294,167 + shared TE 4,331,077,508 + shared VAE 345,053,168 = 7,002,424,843
/// (7.00 GB decimal / 6.52 GiB), against inference's measured `calibration_peak_gb(Q4)` of
/// **7.868 GB** at 512² — the anchor sc-15071 re-derived and this branch's pin bump finally
/// brings into the app. The two gates are different instruments (this one bounds WIRED weights
/// under a staged schedule; mage's own `ensure_generation_fits` bounds the complete peak against
/// MLX's live limit), but they must not disagree about direction: a 6 GB machine cannot hold a
/// tier that peaks at 7.87 GB, and this gate must stop saying it can.
#[test]
fn the_mage_q4_admit_boundary_moves_with_the_real_install_size() {
    const DIT: u64 = 2_326_294_167;
    const TE: u64 = 4_331_077_508;
    const VAE: u64 = 345_053_168;
    let total = DIT + TE + VAE;
    let budget = |gb: f64| Some(MlxMemoryBudget { total_gb: gb });
    for gb in [6.0, 7.0] {
        println!(
            "q4 @ {gb} GB cap: fixed={:?}  pre-fix(DiT-only)={:?}",
            decide_residency(total, TE, budget(gb), true),
            decide_residency(DIT, 0, budget(gb), true),
        );
    }

    // 6 GB: below the 7.868 GB measured q4 peak ⇒ must refuse.
    assert!(
        matches!(
            decide_residency(total, TE, budget(6.0), true),
            ResidencyOutcome::Reject { .. }
        ),
        "a 6 GB Mac cannot hold a tier whose measured peak is 7.87 GB"
    );
    // 7 GB: the staged schedule's wired high-water is the text encoder (4.03 GiB), which fits
    // 7 − the legacy reserve. Admitting here is the transition override working as designed — a machine
    // that can hold the weights runs, paging the activation transient.
    assert_eq!(
        decide_residency(total, TE, budget(7.0), true),
        ResidencyOutcome::Sequential
    );

    // The pre-fix reading — the DiT alone, no text-encoder split — cleared BOTH caps. Asserted
    // so the numbers above are visibly about the footprint and not about the thresholds.
    assert_eq!(
        decide_residency(DIT, 0, budget(6.0), true),
        ResidencyOutcome::Sequential,
        "this is the acceptance-run symptom: 6 GB admitted because the gate saw a third of the \
             tier. If it now rejects, the floor moved and this test's premise must be re-derived."
    );
}

#[test]
fn weights_source_bytes_counts_both_file_and_dir_control_checkpoints() {
    // The qwen_image_control VACE branch ships either as a single `.safetensors` File or as a Dir
    // of shards; both must be counted so `apply_residency_policy` folds the control branch into the
    // heavy side of the staged-peak split (else the DiT-phase working set is under-counted).
    let root_guard = tempfile::Builder::new()
        .prefix("mlx_fit_gate_ctrl_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();

    // Single-file control checkpoint ⇒ its file length.
    let file = root.join("control.safetensors");
    std::fs::write(&file, vec![0u8; 4096]).expect("control file");
    assert_eq!(
        weights_source_bytes(&WeightsSource::File(file.clone())),
        4096
    );

    // Dir control checkpoint ⇒ the recursive `.safetensors` sum (AppleDouble sidecars skipped).
    let dir = root.join("control_dir");
    std::fs::create_dir_all(&dir).expect("mk control dir");
    std::fs::write(dir.join("part-1.safetensors"), vec![0u8; 1000]).expect("shard 1");
    std::fs::write(dir.join("part-2.safetensors"), vec![0u8; 2000]).expect("shard 2");
    std::fs::write(dir.join("._part-1.safetensors"), vec![0u8; 999]).expect("sidecar");
    assert_eq!(weights_source_bytes(&WeightsSource::Dir(dir)), 3000);
}

#[test]
fn imported_file_plan_prices_consumed_companions_not_replaced_snapshot_transformer() {
    let root = tempfile::tempdir().expect("imported plan tempdir");
    let imported = root.path().join("imported.safetensors");
    let base = root.path().join("base");
    for component in ["text_encoder", "vae", "transformer", "tokenizer"] {
        std::fs::create_dir_all(base.join(component)).expect("component dir");
    }
    std::fs::write(&imported, vec![0_u8; 11]).expect("imported DiT");
    std::fs::write(
        base.join("text_encoder").join("model.safetensors"),
        vec![0_u8; 13],
    )
    .expect("text encoder");
    std::fs::write(base.join("vae").join("model.safetensors"), vec![0_u8; 17]).expect("vae");
    std::fs::write(
        base.join("transformer").join("model.safetensors"),
        vec![0_u8; 19],
    )
    .expect("replaced snapshot transformer");
    std::fs::write(base.join("tokenizer").join("tokenizer.json"), b"{}").expect("tokenizer config");
    std::fs::write(base.join("transformer").join("config.json"), b"{}")
        .expect("transformer config");

    let spec = LoadSpec::new(WeightsSource::File(imported))
        .with_component(gen_core::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(base));
    let consumed = 11 + 13 + 17;
    let (fallback_total, fallback_te, _) =
        spec_component_bytes_with_provider_footprint("krea_2_turbo", &spec, None);
    assert_eq!((fallback_total, fallback_te), (consumed, 13));

    let fallback = MlxRequestPlan::for_spec_and_manifest_with_provider_facts(
        "krea_2_turbo",
        "imported_krea",
        &spec,
        None,
        None,
        None,
        None,
    );
    assert_eq!(
        fallback.asset_bytes, consumed,
        "the compatibility path must enumerate only File + TE + VAE; tokenizer/config carry no \
             tensors and the snapshot transformer is replaced"
    );

    let provider_owned = MlxRequestPlan::for_spec_and_manifest_with_provider_facts(
        "krea_2_turbo",
        "imported_krea",
        &spec,
        None,
        None,
        Some(PerComponentBytes {
            text_encoder: 13,
            dit: 11,
            vae: 17,
        }),
        None,
    );
    assert_eq!(provider_owned.asset_bytes, consumed);
    assert_eq!(provider_owned.asset_bytes + 19, 60);
    assert_ne!(
        provider_owned.asset_bytes, 60,
        "recursively charging base_snapshot would add the unused 19-byte transformer"
    );
}

/// The gate derives sequential-capability from each engine's REGISTERED descriptor bit
/// (`Capabilities::supports_sequential_offload`) rather than a hand-maintained allowlist (sc-10840,
/// epic 10834). This exercises the LIVE registry, so it must see the force-linked `mlx_gen_*`
/// providers — anchored (`use mlx_gen_* as _;` in `image_jobs`) only on macOS, the sole platform the
/// MLX gate runs on. Off-Mac the image registry is empty, so this is macOS-gated exactly like the
/// `engines.rs` descriptor sweeps. The assertions below mirror the capability bits of the
/// currently pinned providers, so the shared registry query stays the only source of truth:
/// selectable engines resolve true through that query, while a registered provider that stages
/// unconditionally remains false for this specific request-selectable control.
#[cfg(target_os = "macos")]
#[test]
fn engine_supports_sequential_is_derived_from_the_registered_capability() {
    // The earlier-wired families (sdxl / z-image / qwen / lens / krea) still resolve true — proving
    // dropping the hardcoded allowlist introduced no regression for the already-covered engines.
    for id in [
        "sdxl",
        "z_image",
        "z_image_control",
        "z_image_turbo",
        "z_image_turbo_control",
        "qwen_image",
        "qwen_image_2_1",
        "qwen_image_2_1_turbo",
        "qwen_image_edit",
        "qwen_image_control",
        "lens",
        "lens_turbo",
        "krea_2_turbo",
        "krea_2_raw",
        "krea_2_edit",
        "krea_2_turbo_edit",
        "krea_2_turbo_control",
    ] {
        assert!(
            engine_supports_sequential(id),
            "{id}: earlier-wired family must stay sequential-capable"
        );
    }
    // The sc-10840 selectable Phase-1 fan-out families are AUTO-covered by the capability query with no
    // allowlist edit here — the whole point of deriving from the descriptor bit. A newly-wired
    // engine (e.g. `sd3_5_large`) is sequential-capable the moment its provider advertises the bit.
    for id in [
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
        "sana_1600m",
        "sana_sprint_1600m",
        "flux1_schnell",
        "flux1_dev",
        "flux1_dev_control",
        "flux2_klein_9b",
        "flux2_klein_9b_edit",
        "flux2_klein_9b_kv_edit",
        "flux2_dev",
        "flux2_dev_control",
        "flux2_dev_edit",
        "chroma1_base",
        "chroma1_flash",
        "chroma1_hd",
        "ideogram_4",
        "ideogram_4_turbo",
        "anima_base",
        "anima_aesthetic",
        "anima_turbo",
        "boogu_image",
        "boogu_image_turbo",
        "boogu_image_edit",
    ] {
        assert!(
            engine_supports_sequential(id),
            "{id}: sc-10840 fan-out engine must advertise selectable sequential residency at the \
                 pinned inference revision"
        );
    }
    // Bernini's provider is registered and physically stages every request, but it has no
    // Resident-warm mode and does not consume `OffloadPolicy`. Its descriptor must therefore
    // expose unconditional staging without falsely advertising the selectable Sequential control.
    let bernini = crate::inference_runtime::media()
        .generators()
        .find(|registration| (registration.descriptor)().id == "bernini")
        .expect("Bernini provider must be registered in the MLX runtime");
    let bernini_capabilities = (bernini.descriptor)().capabilities;
    assert!(!bernini_capabilities.supports_sequential_offload);
    assert!(bernini_capabilities.unconditionally_engages_staged_residency);
    // The DESCRIPTOR still does not advertise the selectable Sequential control — that is the
    // assertion directly above, and it is what keeps the knob off Bernini. `engine_supports_
    // sequential` is a different question: it feeds `decide_residency_with_headroom`, i.e. how
    // much memory to CHARGE. sc-19721 widened it to the disjunction because reading only
    // `supports_sequential_offload` made Bernini charge the SUM of planner + UMT5-XXL + both
    // experts — a co-residency `generate_impl` never creates. Unconditional staging is
    // sequential for accounting purposes even though it is not offerable as a control.
    assert!(engine_supports_sequential("bernini"));
    // A REGISTERED engine that does NOT advertise the bit stays false: sensenova's encoder is fused
    // into a unified MoT (`footprint` te=0) — no separable text encoder to drop, so residency buys
    // nothing and Sequential would be a no-op that OOMs. This proves the query reads the descriptor
    // BIT, not mere registry membership.
    //
    // BOTH SenseNova engine ids, and REGISTRATION IS ASSERTED FIRST (sc-22734 review):
    // `engine_supports_sequential` is `is_some_and`, so it answers `false` for an id that is not
    // in the registry at all. Asserting only the `false` would therefore stay green if the
    // provider were renamed, dropped from the pinned bundle, or misspelled here — the exact
    // failure this case claims to rule out. Look the descriptor up by name, prove BOTH bits are
    // clear on it, and only then ask the predicate.
    //
    // The 8-step distill (`sensenova_u1_8b_fast`) is a separate registration with the same fused
    // MoT encoder, so it must answer the same way; the adapter's Candle spec builder cites this
    // fact for BOTH ids to justify `OffloadPolicy::Resident`.
    for id in ["sensenova_u1_8b", "sensenova_u1_8b_fast"] {
        let capabilities = crate::inference_runtime::media()
            .generators()
            .find(|reg| (reg.descriptor)().id == id)
            .map(|reg| (reg.descriptor)().capabilities)
            .unwrap_or_else(|| panic!("{id} is registered in the pinned bundle"));
        assert!(
            !capabilities.supports_sequential_offload,
            "{id}: the fused MoT exposes no separable text encoder to offload"
        );
        assert!(
            !capabilities.unconditionally_engages_staged_residency,
            "{id}: nor does it stage unconditionally — neither bit of the disjunction is set"
        );
        assert!(!engine_supports_sequential(id));
    }

    // sc-19721: WHICH engines reach `true` through the second bit rather than the first, pinned
    // as a set so the disjunction cannot quietly widen. Every one of these declares
    // `unconditionally_engages_staged_residency: true` and `supports_sequential_offload: false`
    // at the pinned revision: they stage physically on every generation and expose no selectable
    // control to honour. Bernini is here because inference moved it between the two bits, which
    // is what made the disjunction necessary.
    for id in ["bernini", "krea_realtime_14b", "ltx_2_3", "scail2_14b"] {
        let capabilities = crate::inference_runtime::media()
            .generators()
            .find(|reg| (reg.descriptor)().id == id)
            .map(|reg| (reg.descriptor)().capabilities)
            .unwrap_or_else(|| panic!("{id} is registered in the pinned bundle"));
        assert!(
            !capabilities.supports_sequential_offload,
            "{id}: this set is specifically the engines the FIRST bit does not cover"
        );
        assert!(
            capabilities.unconditionally_engages_staged_residency,
            "{id}: reaches the gate only through the unconditional-staging bit"
        );
        assert!(engine_supports_sequential(id));
    }
}

/// Static ladder publication needs the broader descriptor fact: a provider may physically
/// stage every request without exposing a selectable Sequential control (SC-18816). This list
/// deliberately contains both selectable and unconditional providers. The memory-matrix
/// generator parses this exact positive sweep, while runtime load selection remains pinned to
/// `engine_supports_sequential` above.
#[cfg(target_os = "macos")]
#[test]
fn engine_engages_staged_residency_is_derived_from_the_registered_capability() {
    // Bound to a local rather than iterated inline so the completeness loop at the bottom can
    // test MEMBERSHIP of this exact list. That matters: `scripts/generate-memory-matrix.mjs`
    // (`parseMlxStagedResidencyEngines`) parses the STRING LITERALS in this function body as
    // the MLX staged-residency authority, so an engine that stages but is missing HERE still
    // publishes `Missing` in the matrix no matter what its descriptor says. Checking the
    // predicate instead of the list would therefore pass while the artifact stayed wrong.
    // The literals stay inside the function, ahead of the first negative assert, because that
    // span is exactly what the generator's regex reads.
    let swept = [
        "sdxl",
        // Iris-3B (sc-25679): `supports_sequential_offload` — encoder, drop, then backbone.
        "iris_3b",
        // FLUX 3 Image Klein (sc-26234): the same staged shape — tower, drop, then DiT + AE.
        "flux3_klein",
        "flux3_klein_base",
        "z_image",
        "z_image_control",
        "z_image_turbo",
        "z_image_turbo_control",
        "qwen_image",
        "qwen_image_2_1",
        "qwen_image_2_1_turbo",
        "qwen_image_edit",
        "qwen_image_control",
        "lens",
        "lens_turbo",
        "krea_2_turbo",
        "krea_2_raw",
        "krea_2_edit",
        "krea_2_turbo_edit",
        "krea_2_turbo_control",
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
        "sana_1600m",
        "sana_sprint_1600m",
        "flux1_schnell",
        "flux1_dev",
        "flux1_dev_control",
        "flux2_klein_9b",
        "flux2_klein_9b_edit",
        "flux2_klein_9b_kv_edit",
        "flux2_dev",
        "flux2_dev_control",
        "flux2_dev_edit",
        "chroma1_base",
        "chroma1_flash",
        "chroma1_hd",
        "ideogram_4",
        "ideogram_4_turbo",
        "kolors",
        "anima_base",
        "anima_aesthetic",
        "anima_turbo",
        "boogu_image",
        "boogu_image_turbo",
        "boogu_image_edit",
        "bernini",
        "bernini_renderer",
        "krea_realtime_14b",
        "ltx_2_3",
        "scail2_14b",
        "wan2_2_i2v_14b",
        "wan2_2_t2v_14b",
        "wan2_2_ti2v_5b",
        "wan2_2_vace_fun_14b",
        "wan_vace",
    ];
    for id in swept {
        assert!(
            engine_engages_staged_residency(id),
            "{id}: registered staged-residency declaration must remain visible to the ladder"
        );
    }
    assert!(!engine_engages_staged_residency("sensenova_u1_8b"));
    assert!(!engine_engages_staged_residency("no_such_engine_xyz"));
    for id in ENGINES_WITHOUT_A_STAGED_RESIDENCY_CAPABILITY {
        assert!(
            !engine_engages_staged_residency(id),
            "{id}: registered descriptor must continue to advertise no staged residency"
        );
    }

    // COMPLETENESS (sc-24108). The sweep above only asserts the ids it LISTS, so a newly
    // registered engine that belongs in it is invisible: the list stays green, the memory-matrix
    // generator — which parses this exact block as its authority for MLX staged residency
    // (scripts/generate-memory-matrix.mjs) — publishes `Missing` for it, and the census fixture
    // pins that `Missing` as if it were a decision. That is a silent, un-armed deferment: no
    // test anywhere turns red when the provider finally registers.
    //
    // So: every MODEL_TABLE row whose engine the PINNED runtime actually resolves must appear
    // in the positive list or in the explicit negative one. An engine that registers and is in
    // neither fails HERE, which is what forces both the sweep entry and the matrix
    // regeneration at the moment the pin lands rather than whenever someone notices.
    //
    // Both lists are read back off the source of truth — `engine_engages_staged_residency`
    // itself — rather than restated, so this cannot drift from the assertions above.
    let mut unclassified: Vec<&str> = crate::engines::MODEL_TABLE
        .iter()
        // Not registered by this pin: nothing to classify yet. The moment a provider registers,
        // this loop demands its classification, which is the point.
        .filter(|row| crate::engines::mlx_model(row.sceneworks_id).is_some())
        .map(|row| row.engine_id)
        .filter(|engine_id| {
            !swept.contains(engine_id)
                && !ENGINES_WITHOUT_A_STAGED_RESIDENCY_CAPABILITY.contains(engine_id)
        })
        .collect();
    unclassified.sort_unstable();
    unclassified.dedup();
    assert_eq!(
        unclassified,
        Vec::<&str>::new(),
        "these engines are registered by the pinned runtime but appear in neither the \
             staged-residency sweep above nor ENGINES_WITHOUT_A_STAGED_RESIDENCY_CAPABILITY. \
             Classify each: add it to the sweep and re-run `npm run generate:memory-matrix` (the \
             generator reads the sweep as its MLX authority, so the cells publish Missing until \
             you do), or record its descriptor as advertising no staging bit (sc-24108)."
    );
}

/// Engines the pinned MLX runtime registers whose DESCRIPTOR advertises no staged-residency
/// capability — neither `supports_sequential_offload` nor the SC-18816
/// `unconditionally_engages_staged_residency` bit — so
/// [`engine_engages_staged_residency_is_derived_from_the_registered_capability`]'s completeness
/// loop does not demand a sweep entry for them. This is an OBSERVED fact about each descriptor,
/// read back from `engine_engages_staged_residency` in that same test, not a claim that the
/// provider never stages anything.
///
/// That distinction matters for the six Mage engines. The generated matrix publishes their
/// `staged_residency` cells as `Anchored`, which folds to "implemented" in the census — but it
/// does so from MEASURED ANCHORS, not from this predicate, which the generator consults only to
/// choose between `Implemented` and `Missing`. So the matrix and this list do not disagree:
/// Mage has evidence and no advertised capability bit. Whether `mlx-gen-mage`'s descriptor
/// SHOULD advertise one is a real question about that provider and predates sc-24108, which
/// added this loop; it is recorded here rather than silently absorbed.
#[cfg(target_os = "macos")]
const ENGINES_WITHOUT_A_STAGED_RESIDENCY_CAPABILITY: &[&str] = &[
    "mage_flow",
    "mage_flow_base",
    "mage_flow_edit",
    "mage_flow_edit_base",
    "mage_flow_edit_turbo",
    "mage_flow_turbo",
    // sc-26325: native MOSS accepts resident/eager loading and advertises neither staging bit.
    "moss_image_base",
    "moss_image_sft",
    "sensenova_u1_8b",
    "sensenova_u1_8b_fast",
    // sc-24213: Supra2-IMG loads resident and eager; its loader refuses sequential offload and
    // deferred materialization, so the descriptor advertises neither staging bit.
    "supra2_img",
];

/// An id with no registered generator is never sequential-capable (the safe default: never select a
/// residency policy the provider won't honor) — a cross-platform invariant.
#[test]
fn engine_supports_sequential_is_false_for_an_unregistered_id() {
    assert!(!engine_supports_sequential("no_such_engine_xyz"));
}

#[test]
fn production_residency_policies_materialize_each_provider_under_its_bound_shape() {
    let eager = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from("fixture")));
    let z_image = with_selected_sequential_shape("z_image_turbo", eager.clone());
    assert_eq!(z_image.offload_policy, OffloadPolicy::Sequential);
    assert_eq!(
        z_image.load_shape,
        gen_core::LoadShape::DeferredMaterialization,
        "the shipped Z-Image rung-4 binding must be producible by the production cold-load route"
    );
    let eager_edit = eager
        .clone()
        .with_resolved_route("z_image_edit")
        .with_refused_load_shape_declaration();
    let refused_edit = with_selected_sequential_shape("z_image_turbo", eager_edit);
    assert_eq!(refused_edit.offload_policy, OffloadPolicy::Sequential);
    assert_eq!(
        refused_edit.load_shape,
        gen_core::LoadShape::EagerMaterialization,
        "the shared provider id must not bypass z_image_edit's declaration/predicate refusal"
    );
    let admitted_edit = eager
        .clone()
        .with_resolved_route("z_image_edit")
        .with_applied_load_shape_declaration();
    assert_eq!(
        with_selected_sequential_shape("z_image_turbo", admitted_edit).load_shape,
        gen_core::LoadShape::DeferredMaterialization,
    );
    for provider in ["anima_base", "chroma1_base", "kolors", "z_image"] {
        let refused = eager
            .clone()
            .with_resolved_route(provider)
            .with_refused_load_shape_declaration();
        let refused = with_selected_sequential_shape(provider, refused);
        assert_eq!(refused.offload_policy, OffloadPolicy::Sequential);
        assert_eq!(
            refused.load_shape,
            gen_core::LoadShape::EagerMaterialization,
            "{provider}: a same-id declaration refusal must survive the late sequential shaper",
        );
        assert_eq!(
            refused.load_shape_declaration_result,
            gen_core::LoadShapeDeclarationResult::Refused,
        );
    }

    let qwen_resident = eager
        .clone()
        .with_load_shape(gen_core::LoadShape::DeferredMaterialization);
    let qwen_resident = apply_residency_policy(qwen_resident, "qwen_image")
        .expect("production deferred Qwen resident route");
    assert_eq!(qwen_resident.offload_policy, OffloadPolicy::Resident);
    assert_eq!(
        qwen_resident.load_shape,
        gen_core::LoadShape::DeferredMaterialization
    );
    let qwen_sequential = with_selected_sequential_shape("qwen_image", qwen_resident.clone());
    assert_eq!(qwen_sequential.offload_policy, OffloadPolicy::Sequential);
    assert_eq!(
        qwen_sequential.load_shape,
        gen_core::LoadShape::DeferredMaterialization
    );

    let krea = with_selected_sequential_shape("krea_2_turbo_control", eager);
    assert_eq!(krea.offload_policy, OffloadPolicy::Sequential);
    assert_eq!(
        krea.load_shape,
        gen_core::LoadShape::EagerMaterialization,
        "Krea's shipped bounded-decode bindings remain on its production eager route"
    );
}

/// Candle's sc-12130 twin of the macOS registry sweep above. These are the generic generator ids that
/// reach the Candle fit gate after route diversion, including the registered Krea pose generator.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn candle_sequential_capability_is_derived_from_the_registered_descriptor() {
    for id in [
        "flux1_dev",
        "flux1_schnell",
        "flux2_dev",
        "flux2_klein_9b",
        "qwen_image",
        "z_image_turbo",
        "z_image",
        "krea_2_turbo",
        "krea_2_raw",
        "krea_2_edit",
        "krea_2_turbo_control",
    ] {
        assert!(
            engine_supports_sequential(id),
            "{id}: wired Candle provider must advertise sequential residency"
        );
    }
}

#[test]
fn predicted_sequential_peak_is_largest_component_plus_headroom() {
    let gib = 1024 * 1024 * 1024_u64;
    // illustrious q8-class: total ~5 GiB, text encoders ~1 GiB ⇒ staged = max(1, 4) + headroom.
    let total = 5 * gib;
    let te = gib;
    assert_eq!(
        predicted_sequential_peak_gb(total, te),
        Some(4.0 + HEADROOM_GB)
    );
    // TE-dominant (lens-class): total 17, TE 13 ⇒ staged = max(13, 4) + headroom = 13 + headroom.
    assert_eq!(
        predicted_sequential_peak_gb(17 * gib, 13 * gib),
        Some(13.0 + HEADROOM_GB)
    );
    // Unmeasured text encoders ⇒ staged == resident sum (no claimed saving), the safe fallback.
    assert_eq!(
        predicted_sequential_peak_gb(20 * gib, 0),
        Some(20.0 + HEADROOM_GB)
    );
    // Nothing measured ⇒ no signal.
    assert_eq!(predicted_sequential_peak_gb(0, 0), None);
}

#[test]
fn resolve_offload_rewrites_toobig_only_when_capable() {
    let too_big = FitDecision::TooBig {
        needed_gb: 46.0,
        available_gb: 16.0,
    };
    // Sequential-capable provider ⇒ Offload (carrying the resident numbers).
    assert_eq!(
        resolve_offload(too_big.clone(), true),
        FitDecision::Offload {
            needed_gb: 46.0,
            available_gb: 16.0,
        }
    );
    // Non-capable ⇒ still a reject.
    assert!(matches!(
        resolve_offload(too_big, false),
        FitDecision::TooBig { .. }
    ));
    // Fits / Unknown are never rewritten.
    assert_eq!(resolve_offload(FitDecision::Fits, true), FitDecision::Fits);
    assert_eq!(
        resolve_offload(FitDecision::Unknown, true),
        FitDecision::Unknown
    );
}

#[test]
fn sequential_overflow_rejects_only_a_genuine_staged_overflow() {
    let budget = Some(MlxMemoryBudget { total_gb: 16.0 });
    // Staged still needs 20 > 16 ⇒ reject even sequentially.
    assert_eq!(sequential_overflow_gb(Some(20.0), budget), Some(20.0));
    // Staged fits (14 <= 16) ⇒ run sequentially, no reject.
    assert_eq!(sequential_overflow_gb(Some(14.0), budget), None);
    // Exactly-fits is not an overflow.
    assert_eq!(sequential_overflow_gb(Some(16.0), budget), None);
    // No staged estimate or no budget ⇒ best-effort run (no reject).
    assert_eq!(sequential_overflow_gb(None, budget), None);
    assert_eq!(sequential_overflow_gb(Some(20.0), None), None);
}

#[test]
fn too_big_error_names_model_budget_and_optional_staged() {
    // Plain resident reject (non-staging provider): no staged note.
    let WorkerError::InvalidPayload(resident) = too_big_error("qwen-image", 46.0, 16.0, None)
    else {
        panic!("expected InvalidPayload");
    };
    assert!(
        resident.contains("qwen-image"),
        "names the model: {resident}"
    );
    assert!(resident.contains("unified memory"), "explains: {resident}");
    assert!(resident.contains("46"), "states what it needs: {resident}");
    assert!(resident.contains("16"), "states the budget: {resident}");
    assert!(
        !resident.contains("one component at a time"),
        "no staged note when not attempted: {resident}"
    );
    // Staged reject: the message also names the one-at-a-time requirement.
    let WorkerError::InvalidPayload(staged) = too_big_error("sdxl", 46.0, 16.0, Some(24.0)) else {
        panic!("expected InvalidPayload");
    };
    assert!(
        staged.contains("one component at a time"),
        "names the staged path: {staged}"
    );
    assert!(
        staged.contains("24"),
        "states the staged requirement: {staged}"
    );
}

// The PEAK layer (`decide_residency_by_peak`) still selects Resident / Sequential / Reject exactly
// as the pre-sc-12179 gate did — the Decision 2 legacy override is layered on top only for
// unverified cells, so this proves the peak selection is intact.
#[test]
fn decide_residency_by_peak_picks_resident_sequential_or_reject_by_budget() {
    let gib = 1024 * 1024 * 1024_u64;
    // illustrious q8-class: total ~5 GiB (TE ~1, DiT+VAE ~4). With HEADROOM_GB=18 (sc-10863):
    // resident peak = 5+18 = 23; staged peak = max(1, 4)+18 = 22.
    let total = 5 * gib;
    let te = gib;

    // Roomy budget (128 GB Mac) ⇒ Resident (keep the warm path).
    assert_eq!(
        decide_residency_by_peak(total, te, Some(MlxMemoryBudget { total_gb: 128.0 }), true),
        ResidencyOutcome::Resident
    );
    // Budget between staged (22) and resident (23): resident won't fit, staged will, provider
    // stages ⇒ Sequential. This is the fit-gate SELECTING sequential residency.
    assert_eq!(
        decide_residency_by_peak(total, te, Some(MlxMemoryBudget { total_gb: 22.5 }), true),
        ResidencyOutcome::Sequential
    );
    // Same budget but a provider that can't stage ⇒ reject (never a silent Resident that OOMs).
    assert!(matches!(
        decide_residency_by_peak(total, te, Some(MlxMemoryBudget { total_gb: 22.5 }), false),
        ResidencyOutcome::Reject {
            staged_gb: None,
            ..
        }
    ));
    // Budget below even the staged peak (22) ⇒ reject, naming the staged requirement.
    assert!(matches!(
        decide_residency_by_peak(total, te, Some(MlxMemoryBudget { total_gb: 20.0 }), true),
        ResidencyOutcome::Reject {
            staged_gb: Some(_),
            ..
        }
    ));
    // No budget signal ⇒ Resident (never block).
    assert_eq!(
        decide_residency_by_peak(total, te, None, true),
        ResidencyOutcome::Resident
    );
    // Unmeasured weights ⇒ Resident (no signal).
    assert_eq!(
        decide_residency_by_peak(0, 0, Some(MlxMemoryBudget { total_gb: 8.0 }), true),
        ResidencyOutcome::Resident
    );
}

#[test]
fn generic_mlx_adopts_shared_selector_without_an_optimized_claim() {
    let weights_bytes = 4 * 1024 * 1024 * 1024;
    // 96 GB: above the floor's admitted requirement (4 + 18 raw plus the activation-term
    // allowance on the 18 GiB headroom), which the assertion below recomputes exactly.
    let observation = generic_mlx_shared_observation(
        weights_bytes,
        Some(MlxMemoryBudget { total_gb: 96.0 }),
        HEADROOM_GB,
    );
    let crate::memory_strategy::Selection::Selected {
        selection,
        needed_gb,
        ..
    } = observation
    else {
        panic!("the generic cold-load floor must reach a selection: {observation:?}");
    };
    assert_eq!(selection.strategy, gen_core::MemoryStrategy::Resident);
    // sc-22508: the declared activation term must be a SLICE OF THE GRADED PEAK, never a
    // second charge sitting beside it. This route's headroom used to live only in the budget's
    // `reserved_headroom_gb`, which made `unmodeled_activation_bytes` an out-of-band charge
    // contradicting the field's contract. Pin that the graded peak now contains it: a peak
    // rebuilt from the weights alone lands 18 GiB lower and reds here.
    let headroom_bytes = (HEADROOM_GB * BYTES_PER_GIB).ceil() as u64;
    assert_eq!(
        needed_gb,
        crate::memory_strategy::peak_bytes_to_gb(
            crate::memory_strategy::floor_admitted_peak_bytes(
                gen_core::MemoryBackend::Mlx,
                weights_bytes + headroom_bytes,
                Some(headroom_bytes),
            )
        ),
        "the floor's peak is weights + headroom, and its allowance is a fraction of the \
             headroom term inside that peak"
    );
}

/// The load-time weights floor (`weights_floor_load_admission`, formerly
/// `legacy_admission_override`) must preserve its settled small-Mac outcomes byte-for-byte
/// through sc-18096's retirement of the request-path freeze: a spec whose bare staged weights
/// fit still loads, and one whose weights cannot be held resident under any policy still
/// rejects at load.
#[test]
fn weights_floor_load_admission_preserves_small_mac_behavior() {
    let gib = 1024 * 1024 * 1024_u64;

    // SANA-class small model on an 8 GB Mac: total 2 GiB (TE 1, rest 1). Peak = 2+18 = 20 ≫ 8 and
    // staged peak = 1+18 = 19 ≫ 8, so the PEAK layer rejects outright...
    let (total, te) = (2 * gib, gib);
    let budget = Some(MlxMemoryBudget { total_gb: 8.0 });
    assert!(matches!(
        decide_residency_by_peak(total, te, budget, true),
        ResidencyOutcome::Reject { .. }
    ));
    // ...but the staged weights (1 GiB) fit 8 − 2 = 6, so legacy runs it Sequential instead of
    // walling off the machine. This is exactly the model that generated fine on 0.7.3.
    assert_eq!(
        decide_residency(total, te, budget, true),
        ResidencyOutcome::Sequential
    );
    // A non-staging provider whose whole 2 GiB weights fit 6 loads Resident best-effort (not reject).
    assert_eq!(
        decide_residency(total, te, budget, false),
        ResidencyOutcome::Resident
    );

    // A genuinely-too-big model on 8 GB still rejects: 40 GiB weights (staged max-component 30) can
    // NOT be held resident under any policy ⇒ the override returns None and the reject stands.
    let (big_total, big_te) = (40 * gib, 10 * gib);
    assert!(matches!(
        decide_residency(big_total, big_te, budget, true),
        ResidencyOutcome::Reject { .. }
    ));

    // The override never fabricates a decision without a budget signal.
    assert_eq!(
        decide_residency(total, te, None, true),
        ResidencyOutcome::Resident
    );
}

/// Decision 1's existing 8 GiB policy guard, preserved byte-for-byte through sc-18096: the
/// LOAD-time weights floor still admits this real on-disk q4 footprint (the request-scoped
/// estimate ladder owns transient bounding, not the loader). This is a policy outcome only,
/// not a model calibration or implementation claim. Measured
/// tier layout: total 5.49 GiB (text_encoder 2.11, transformer 3.23, vae 0.15), so the largest
/// single component the Sequential schedule ever holds wired is ~3.38 GiB. Against an 8 GB budget
/// (legacy reserve 2 ⇒ 6 GiB weight budget) that admits with ~2.6 GiB of margin.
#[test]
fn unverified_q4_8gb_guard_remains_on_the_legacy_transition() {
    // Bytes rounded from the measured tier (…/z-image-turbo-mlx/…/q4), following HF-cache symlinks.
    let mib = 1024 * 1024_u64;
    let total = 5624 * mib; // ~5.49 GiB whole model
    let te = 2161 * mib; //    ~2.11 GiB Qwen text encoder (dropped first under Sequential)
    let budget = Some(MlxMemoryBudget { total_gb: 8.0 });

    // The flat-headroom peak layer would reject it (5.49 + 18 = 23.49 ≫ 8)...
    assert!(matches!(
        decide_residency_by_peak(total, te, budget, true),
        ResidencyOutcome::Reject { .. }
    ));
    // ...but z-image-turbo stages components, and its largest (transformer ≈ 3.38 GiB) fits the
    // 6 GiB weight budget ⇒ Sequential. This is the #1544 baseline that must keep working.
    assert_eq!(
        decide_residency(total, te, budget, true),
        ResidencyOutcome::Sequential
    );
}

#[test]
fn sum_text_encoder_bytes_sums_only_text_encoder_subdirs() {
    let root_guard = tempfile::Builder::new()
        .prefix("mlx_fit_gate_te_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    // SDXL-shaped tree: two CLIP encoders + the U-Net + VAE.
    for (sub, bytes) in [
        ("text_encoder", 1000usize),
        ("text_encoder_2", 3000),
        ("unet", 9000),
        ("vae", 400),
    ] {
        let dir = root.join(sub);
        std::fs::create_dir_all(&dir).expect("mk subdir");
        std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).expect("weights");
    }
    // Only the two text-encoder subdirs count (1000 + 3000); unet/vae are excluded.
    assert_eq!(sum_text_encoder_bytes(root), 4000);
    // The whole-model sum includes everything.
    assert_eq!(sum_safetensors_bytes(root), 13400);
    // Missing dir ⇒ 0.
    assert_eq!(sum_text_encoder_bytes(&root.join("nope")), 0);
}

// HF cache stores each shard as a symlink into `blobs/`; the gate must follow those to the real
// byte size. The synthetic test above uses plain files, so exercise the symlink layout here.
#[cfg(unix)]
#[test]
fn sum_safetensors_follows_hf_cache_symlinks() {
    use std::os::unix::fs::symlink;
    let root_guard = tempfile::Builder::new()
        .prefix("mlx_fit_gate_symlink_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let blobs = root.join("blobs");
    let snap = root.join("snapshots/hash/transformer");
    std::fs::create_dir_all(&blobs).expect("mk blobs");
    std::fs::create_dir_all(&snap).expect("mk snap");
    std::fs::write(blobs.join("weightblob"), vec![0u8; 4096]).expect("weight blob");
    std::fs::write(blobs.join("sidecarblob"), vec![0u8; 500]).expect("sidecar blob");
    symlink(blobs.join("weightblob"), snap.join("diffusion.safetensors")).expect("weight link");
    symlink(
        blobs.join("sidecarblob"),
        snap.join("._diffusion.safetensors"),
    )
    .expect("sidecar link");

    // Summing the SNAPSHOT dir follows the symlink to the 4096-byte blob and skips the `._`
    // sidecar; the `blobs/` dir is not under the snapshot, so nothing is double-counted.
    assert_eq!(sum_safetensors_bytes(&root.join("snapshots/hash")), 4096);
}

#[cfg(unix)]
#[test]
fn sum_safetensors_terminates_on_directory_symlink_cycles() {
    use std::os::unix::fs::symlink;
    let root_guard = tempfile::Builder::new()
        .prefix("mlx_fit_gate_cycle_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let weights = root.join("weights");
    std::fs::create_dir_all(&weights).expect("mk weights");
    std::fs::write(weights.join("model.safetensors"), vec![0_u8; 4096]).expect("write weights");
    symlink(root, weights.join("cycle")).expect("create directory cycle");

    assert_eq!(sum_safetensors_bytes(root), 4096);
}

/// sc-10894: on a boogu-style snapshot (text encoder under `mllm/`, not `text_encoder*`), the
/// subdir scan reads ZERO, but `resolve_text_encoder_bytes` PREFERS a provider footprint value when
/// present and only falls back to the scan when it is `None`.
#[test]
fn resolve_text_encoder_prefers_footprint_over_subdir_scan() {
    let root_guard = tempfile::Builder::new()
        .prefix("mlx_fit_gate_resolve_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    // Encoder under `mllm/`, DiT `transformer/`, VAE `vae/` — NO `text_encoder*` subdir.
    for (sub, bytes) in [("mllm", 1500usize), ("transformer", 9000), ("vae", 400)] {
        let dir = root.join(sub);
        std::fs::create_dir_all(&dir).expect("mk subdir");
        std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).expect("weights");
    }
    // The historical subdir scan finds no `text_encoder*` → 0 (the bug this seam fixes).
    assert_eq!(sum_text_encoder_bytes(root), 0);
    // The whole-model sum still sees every component.
    assert_eq!(sum_safetensors_bytes(root), 10900);
    // No footprint declared ⇒ fall back to the (zero) subdir scan.
    assert_eq!(resolve_text_encoder_bytes(None, root), 0);
    // A provider footprint (the `mllm/` bytes) is preferred, even though the scan reads zero.
    assert_eq!(resolve_text_encoder_bytes(Some(1500), root), 1500);
}

/// Exercise the production declarations and footprint readers without allocating model tensors.
/// Unlike the old 40 GiB fixture, these are the installed q4 artifacts from the reported failures.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires shipped Z-Image, SDXL and SANA Sprint q4 snapshots; SC23584_RENDER=all also executes on exclusive Metal"]
fn reported_q4_requests_credit_the_implemented_memory_ladder() {
    use crate::ladder_margin_policy::{admission_allowance, AdmissionSubject};
    let prompt = std::env::var("SC23584_PROMPT")
            .unwrap_or_else(|_| "A cinematic frame of a woman in a red raincoat holding a red umbrella standing in the middle of a neon street on a rainy night".to_owned());
    let negative_prompt = std::env::var("SC23584_NEGATIVE_PROMPT").unwrap_or_default();
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("builtin model manifest parses")
            .clone();
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface/hub")
        });
    let mut failures = Vec::new();
    for (id, count) in [("z_image_turbo", 2), ("sdxl", 1), ("sana_sprint_1600m", 1)] {
        let model = manifest["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["id"] == id)
            .unwrap()
            .as_object()
            .unwrap();
        let download = model["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|download| {
                download["variant"] == "q4"
                    && download["repo"]
                        .as_str()
                        .unwrap_or("")
                        .to_lowercase()
                        .contains("mlx")
            })
            .unwrap();
        let root = hub
            .join(format!(
                "models--{}",
                download["repo"].as_str().unwrap().replace('/', "--")
            ))
            .join("snapshots")
            .join(download["revision"].as_str().unwrap())
            .join("q4");
        assert!(
            root.is_dir(),
            "required snapshot missing: {}",
            root.display()
        );
        let root = std::fs::canonicalize(root).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_resolved_route(id)
            .with_quant(gen_core::Quant::Q4);
        let mode = crate::memory_route_registry::MemoryRouteMode::TextToImage;
        let context = crate::memory_route_registry::MemoryRouteRequestContext {
            mode,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
        };
        let supports_sequential = crate::inference_runtime::media_descriptor(id)
            .unwrap()
            .capabilities
            .supports_sequential_offload;
        let mut spec = crate::test_env::temp_env_var(MLX_MEMORY_CAP_ENV, "8", || {
            crate::image_jobs::prepare_mlx_load_policy(
                id,
                Some("q4"),
                Some(mode),
                model,
                spec,
                context,
                true,
                supports_sequential,
            )
            .unwrap()
        });
        eprintln!(
            "{id}: production shape={:?}, declaration={:?}, offload={:?}",
            spec.load_shape, spec.load_shape_declaration_result, spec.offload_policy
        );
        // Exercise the receipt and path resolution used by generation. Copy bookkeeping into
        // a scratch data directory so this test never rewrites the operator's installed receipt.
        let source_data = PathBuf::from(
            std::env::var_os("SC23648_INSTALL_DATA")
                .expect("SC23648_INSTALL_DATA must name the installed SceneWorks data directory"),
        );
        let data = tempfile::tempdir().unwrap();
        let repository = download["repo"].as_str().unwrap();
        let marker_dir = PathBuf::from("models").join(repository.replace('/', "__"));
        std::fs::create_dir_all(data.path().join(&marker_dir)).unwrap();
        std::fs::copy(
            source_data.join(&marker_dir).join(crate::INSTALL_MARKER),
            data.path().join(&marker_dir).join(crate::INSTALL_MARKER),
        )
        .unwrap();
        let settings = crate::image_jobs::resolved_artifact_provenance_tests::settings(data.path());
        let image_request = sceneworks_core::image_request::ImageRequest::from_payload(
            serde_json::json!({"model":id,"advanced":{"mlxQuantize":4},
                    "modelManifestEntry":model})
            .as_object()
            .unwrap(),
        );
        let resolved = crate::image_jobs::resolve_weights_dir(&image_request, &settings)
            .unwrap()
            .unwrap();
        assert_eq!(resolved, root);
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(
                crate::model_jobs::receipt_verification::ensure_huggingface_receipt_provenance(
                    &settings,
                    repository,
                    id,
                    Some("q4"),
                    &root,
                ),
            )
            .unwrap();
        let provenance = crate::image_jobs::resolved_mlx_artifact_provenance(
            &image_request,
            &settings,
            repository,
            &root,
            Some("q4"),
        )
        .unwrap()
        .unwrap();
        let binding =
            bind_decode_quality_policies_from_manifest(model, id, Some(&provenance)).unwrap();
        spec = attach_decode_quality_binding(spec, binding, id);
        let conditioning_windows = clip_windows_for_spec(id, &spec, &prompt, &negative_prompt);
        if id == "sdxl" {
            assert_eq!(clip_windows_for_spec(id, &spec, "fox", ""), 1);
            assert_eq!(clip_windows_for_spec(id, &spec, &"fox ".repeat(76), ""), 2);
            assert_eq!(
                clip_windows_for_spec(id, &spec, "fox", &"fox ".repeat(76)),
                2
            );
            assert_eq!(clip_windows_for_spec(id, &spec, "café ☔", ""), 1);
        }
        eprintln!("{id}: prompt demand {conditioning_windows} windows");
        let plan =
            MlxRequestPlan::for_spec_and_manifest(id, id, &spec, Some(model), Some(provenance));
        let contract = crate::inference_runtime::media()
            .memory_strategy_contract(id, &spec)
            .unwrap()
            .unwrap();
        if std::env::var("SC23648_DIAG_ROUTE").is_ok_and(|route| route == id) {
            // Diagnostic on the development host's real capacity, independent of the emulated
            // admission budget. Exercise only the provider's advertised staging/decode knobs.
            mlx_rs::memory::clear_cache();
            mlx_rs::memory::reset_peak_memory();
            let generator = crate::inference_runtime::load(id, &spec).unwrap();
            let request = gen_core::GenerationRequest {
                prompt: prompt.clone(),
                width: 1024,
                height: 1024,
                count: 1,
                steps: Some(2),
                seed: Some(1234),
                memory: Some(GenerationMemory {
                    stage_residency: true,
                    tile_vae_decode: true,
                    decode_tile_edge: Some(192),
                    decode_overlap: Some(48),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let mut phase_peaks = Vec::new();
            let output = generator
                .generate(&request, &mut |event| {
                    if matches!(
                        event,
                        gen_core::Progress::Loading(_) | gen_core::Progress::Decoding
                    ) {
                        phase_peaks.push((
                            format!("{event:?}"),
                            mlx_rs::memory::get_peak_memory(),
                            mlx_rs::memory::get_active_memory(),
                        ));
                        mlx_rs::memory::reset_peak_memory();
                    }
                })
                .unwrap();
            phase_peaks.push((
                "complete".to_owned(),
                mlx_rs::memory::get_peak_memory(),
                mlx_rs::memory::get_active_memory(),
            ));
            eprintln!("SC23648 diagnostic phases {id}: {phase_peaks:?}");

            eprintln!(
                "SC23648 diagnostic {id}: {:?}, final decode-phase peak {} bytes",
                std::mem::discriminant(&output),
                mlx_rs::memory::get_peak_memory()
            );
        }
        assert!(
            contract.asset_facts.base_bytes > 0,
            "production contracts must price loaded components"
        );
        let mut inputs = fixture_inputs(1024, 1024);
        inputs.count = count;
        let geometry = request_geometry(&inputs);
        assert_eq!(geometry.batch, 1, "job count is sequential");
        let resident = plan.generic_total_peak_bytes(geometry);
        let ladder = synthesize_estimate_ladder_with_conditioning(
            &contract,
            &plan,
            "text_to_image",
            None,
            geometry,
            false,
            None,
            &[],
            Some(conditioning_windows),
        );
        let mut minimum = u64::MAX;
        eprintln!(
            "{id}: facts={:?}, architecture={:?}, resident={:.3} GiB",
            contract.asset_facts,
            contract.architecture_facts,
            resident as f64 / 1073741824.0
        );
        for candidate in ladder.estimates {
            let raw = candidate.evidence.predicted_peak_bytes;
            let allowance = admission_allowance(AdmissionSubject {
                backend: MemoryBackend::Mlx,
                basis: candidate.basis,
                unmodeled_activation_bytes: candidate.unmodeled_activation_bytes,
            });
            let admitted = raw.saturating_add(
                allowance.bytes(raw, candidate.unmodeled_activation_bytes.unwrap_or(raw)),
            );
            minimum = minimum.min(admitted);
            eprintln!(
                "  {:?} {:?}: {:.3} + {:.3} = {:.3} GiB ({:?})",
                candidate.selection.strategy,
                candidate.selection.parameters,
                raw as f64 / 1073741824.0,
                (admitted - raw) as f64 / 1073741824.0,
                admitted as f64 / 1073741824.0,
                candidate.basis
            );
        }
        if minimum > gib_to_bytes(6.0) {
            failures.push(format!(
                "{id}: the optimized ladder costs {minimum} bytes, above the 6 GiB request budget"
            ));
        }
        inputs.conditioning_windows = Some(conditioning_windows);
        let result = preflight_request(
            &spec,
            &plan,
            &inputs,
            MemoryBudget {
                total_bytes: gib_to_bytes(8.0),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: gib_to_bytes(2.0),
            },
        )
        .unwrap();
        let insufficient = preflight_request(
            &spec,
            &plan,
            &inputs,
            MemoryBudget {
                total_bytes: gib_to_bytes(4.0),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: gib_to_bytes(2.0),
            },
        )
        .unwrap();
        assert!(
            matches!(insufficient, MlxRequestAdmission::Rejected(_)),
            "{id}: valid refusals must remain"
        );
        match result {
            MlxRequestAdmission::Rejected(error) => {
                eprintln!("{id}: production preflight refused: {error}");
                failures.push(format!("{id}: production preflight refused: {error}"))
            }
            MlxRequestAdmission::Admitted(evaluation)
                if std::env::var("SC23584_RENDER")
                    .is_ok_and(|route| route == "all" || route == id) =>
            {
                mlx_rs::memory::clear_cache();
                mlx_rs::memory::reset_peak_memory();
                let generator = crate::inference_runtime::load(id, &spec).unwrap();
                for index in 0..count {
                    let mut scope = generator
                        .begin_memory_strategy_request(&evaluation.context)
                        .unwrap()
                        .unwrap();
                    let mut request = gen_core::GenerationRequest {
                        prompt: prompt.clone(),
                        negative_prompt: (!negative_prompt.is_empty())
                            .then(|| negative_prompt.clone()),
                        width: 1024,
                        height: 1024,
                        count: 1,
                        steps: Some(4),
                        seed: Some(1234 + u64::from(index)),
                        memory: Some(evaluation.memory),
                        ..Default::default()
                    };
                    scope.configure_request(&mut request).unwrap();
                    let output = generator.generate(&request, &mut |_| {}).unwrap();
                    let gen_core::GenerationOutput::Images(images) = output else {
                        panic!("expected images")
                    };
                    assert_eq!(images.len(), 1);
                    assert_eq!((images[0].width, images[0].height), (1024, 1024));
                    assert!(images[0]
                        .pixels
                        .iter()
                        .any(|pixel| *pixel != images[0].pixels[0]));
                    let peak = mlx_rs::memory::get_peak_memory() as u64;
                    eprintln!("SC23584 {id} sequential item {index}: cold-inclusive active peak {peak} bytes");
                    assert!(
                        peak <= gib_to_bytes(6.0),
                        "{id}: active runtime exceeded the admitted budget"
                    );
                }
            }
            _ => {}
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// #1544 baseline through the LIVE gate path on REAL weights (ignored — needs the model on disk +
/// the force-linked registry). Drives `residency_for_dir` — the exact seam the worker's cold load
/// uses — against the real z-image-turbo q4 tier under an emulated 8 GB Mac
/// (`SCENEWORKS_MLX_MEMORY_CAP_GB=8`), so it exercises the real on-disk `.safetensors` scan, the
/// provider footprint TE split, the registered `supports_sequential_offload` capability, AND the
/// budget resolution together. Must come back Sequential, not Reject. Run explicitly (alone, since
/// it sets a process env var):
///   cargo test -p sceneworks-worker --lib -- --ignored --nocapture z_image_turbo_q4_live_gate
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs z-image-turbo q4 weights on disk + the force-linked mlx-gen registry"]
fn z_image_turbo_q4_live_gate_admits_under_an_emulated_8gb_cap() {
    // Resolve the q4 snapshot dir from the HF cache (HF_HOME or ~/.cache/huggingface).
    let hf_home = std::env::var("HF_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").expect("HOME"))
                .join(".cache/huggingface")
        });
    let snapshots = hf_home.join("hub/models--SceneWorks--z-image-turbo-mlx/snapshots");
    let Some(q4) = std::fs::read_dir(&snapshots).ok().and_then(|entries| {
        entries
            .flatten()
            .map(|e| e.path().join("q4"))
            .find(|p| p.is_dir())
    }) else {
        eprintln!(
            "SKIP: z-image-turbo q4 not found under {}",
            snapshots.display()
        );
        return;
    };

    // Emulate an 8 GB Mac through the same env the epic added for exactly this (sc-10835).
    // Through the crate-wide seam: `set_var` is process-global, and the hand-rolled
    // set/`remove_var` pair this replaces both skipped the lock AND left the cap set for the rest
    // of the process if the assertion below panicked between them (sc-12380).
    let outcome = crate::test_env::temp_env_var(MLX_MEMORY_CAP_ENV, "8", || {
        residency_for_dir("z_image_turbo", &q4)
    });

    eprintln!("live gate on {} @ 8 GB → {outcome:?}", q4.display());
    assert_eq!(
        outcome,
        ResidencyOutcome::Sequential,
        "z-image-turbo q4 (the #1544 0.7.3 baseline) must run on an 8 GB Mac, not be rejected"
    );
}

/// The REAL shipped `krea_2_turbo` manifest entry, for the two live sc-16482 tests below.
///
/// `SCENEWORKS_KREA_MANIFEST_JSON` points at the entry extracted from
/// `config/manifests/builtin.models.jsonc` (that file is JSONC; this crate has no JSONC reader,
/// and hand-copying the entry into the test would let it drift from what actually ships):
/// ```text
/// node --input-type=module -e "
///   import { readFileSync, writeFileSync } from 'node:fs';
///   const { stripJsoncComments } = await import('./scripts/lib/jsonc.mjs');
///   const doc = JSON.parse(stripJsoncComments(readFileSync('config/manifests/builtin.models.jsonc','utf8')));
///   const list = Array.isArray(doc) ? doc : (doc.models ?? Object.values(doc).find(Array.isArray));
///   writeFileSync(process.env.OUT, JSON.stringify(list.find(m => m && m.id === 'krea_2_turbo')));
/// "
/// ```
/// Returns `None` (with a SKIP note) rather than panicking, so `--ignored` on a machine without
/// the setup reports honestly instead of failing.
#[cfg(target_os = "macos")]
fn real_krea_manifest_entry() -> Option<JsonObject<String, Value>> {
    let Ok(path) = std::env::var("SCENEWORKS_KREA_MANIFEST_JSON") else {
        eprintln!("SKIP: set SCENEWORKS_KREA_MANIFEST_JSON (see doc comment for extraction)");
        return None;
    };
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("read SCENEWORKS_KREA_MANIFEST_JSON {path}: {error}"));
    let manifest: JsonObject<String, Value> =
        serde_json::from_slice(&bytes).expect("manifest entry object");
    assert_eq!(
        manifest.get("id").and_then(Value::as_str),
        Some("krea_2_turbo"),
        "SCENEWORKS_KREA_MANIFEST_JSON must be the krea_2_turbo entry"
    );
    Some(manifest)
}

/// sc-16482 on the REAL install (ignored — needs krea-2-turbo-mlx on disk and its actual receipt).
///
/// Reproduces the reported failure end to end against the artifacts that produced it, rather than
/// a fixture: the user's own `.sceneworks-download-complete.json` (three `backfilled: true`
/// receipts, no `artifactTreeStamp`) plus the real `SceneWorks/krea-2-turbo-mlx` snapshot in the HF
/// cache and the real `krea_2_turbo` manifest entry (2 calibrations, both `krea_2_turbo_control`).
///
/// The receipt is COPIED into a temp data dir, so running this never mutates the real install.
/// `HF_HUB_CACHE` still points at the real cache, so the snapshot and every stat'ed file are real.
///
/// Before the fix this returned `Err("... invalid MLX calibration opt-in: the resolver supplied no
/// immutable artifact provenance")`. Run explicitly:
///   SCENEWORKS_KREA_MANIFEST_JSON=<extracted entry>.json \
///     cargo test -p sceneworks-worker --lib -- --ignored --nocapture krea_real_install
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs the real krea-2-turbo-mlx install + SCENEWORKS_KREA_MANIFEST_JSON"]
fn krea_real_install_backfilled_receipt_now_admits() {
    let home = std::path::PathBuf::from(std::env::var("HOME").expect("HOME"));
    let real_marker = home
        .join("SceneWorks/data/models/SceneWorks__krea-2-turbo-mlx")
        .join(crate::INSTALL_MARKER);
    let Ok(receipt_bytes) = std::fs::read(&real_marker) else {
        eprintln!("SKIP: no real receipt at {}", real_marker.display());
        return;
    };
    let Some(manifest) = real_krea_manifest_entry() else {
        return;
    };
    let calibrations = manifest["mlx"]["calibrations"]
        .as_array()
        .expect("real manifest still declares mlx.calibrations");
    assert!(
        !calibrations.is_empty(),
        "this test is meaningless if the opt-in was removed"
    );

    // The real artifact, untouched.
    let hub = home.join(".cache/huggingface/hub");
    let snapshots = hub.join("models--SceneWorks--krea-2-turbo-mlx/snapshots");
    let Some(snapshot) = std::fs::read_dir(&snapshots)
        .ok()
        .and_then(|entries| entries.flatten().map(|e| e.path()).find(|p| p.is_dir()))
    else {
        eprintln!("SKIP: no krea snapshot under {}", snapshots.display());
        return;
    };
    let q8 = snapshot.join("q8");
    if !q8.is_dir() {
        eprintln!("SKIP: no q8 tier at {}", q8.display());
        return;
    }

    // Copy-on-read data dir: the receipt is real, the file we may stamp is a throwaway.
    let data = tempfile::tempdir().expect("data dir");
    let marker_dir = data
        .path()
        .join("models")
        .join(crate::paths::safe_download_dir(
            "SceneWorks/krea-2-turbo-mlx",
        ));
    std::fs::create_dir_all(&marker_dir).expect("marker dir");
    std::fs::write(marker_dir.join(crate::INSTALL_MARKER), &receipt_bytes).expect("copy");

    let outcome =
        crate::test_env::temp_env_var("HF_HUB_CACHE", hub.to_str().expect("hub path"), || {
            let unrepaired = crate::model_jobs::huggingface_receipt_weights(
                data.path(),
                "SceneWorks/krea-2-turbo-mlx",
                Some("krea_2_turbo"),
                Some("q8"),
                crate::model_jobs::ProvenanceRepair::Skip,
            )
            .expect("the real backfilled receipt still resolves loadable weights");
            assert_eq!(unrepaired.path, q8, "resolves the real q8 tier");
            assert_eq!(
                unrepaired.provenance, None,
                "the real receipt carries no artifactTreeStamp — this IS the reported cause"
            );

            let repaired = crate::model_jobs::huggingface_receipt_weights(
                data.path(),
                "SceneWorks/krea-2-turbo-mlx",
                Some("krea_2_turbo"),
                Some("q8"),
                crate::model_jobs::ProvenanceRepair::Allow,
            )
            .expect("resolves")
            .provenance
            .expect("repair must establish provenance for the real install");

            let spec =
                LoadSpec::new(WeightsSource::Dir(q8.clone())).with_quant(gen_core::Quant::Q8);
            let plan = MlxRequestPlan::for_spec_and_manifest(
                "krea_2_turbo",
                "krea_2_turbo",
                &spec,
                Some(&manifest),
                Some(repaired.clone()),
            );
            (
                repaired,
                packaged_admission_route(
                    &plan,
                    &fixture_inputs(1024, 1024),
                    "text_to_image",
                    fixture_budget(128.0),
                ),
            )
        });

    let (provenance, route) = outcome;
    eprintln!(
        "real install → repo={} revision={} variant={} tier={:?}",
        provenance.identity.repository,
        provenance.identity.revision,
        provenance.identity.variant,
        provenance.fixed_artifact_tier
    );
    let route = route.expect("the reported generation must no longer be refused");
    eprintln!(
        "admission → path={:?} fallback={:?}",
        route.path, route.fallback_reason
    );
    // The base t2i lane has no bindings of its own (every calibration names
    // `krea_2_turbo_control`), so legacy is the CORRECT route here. What matters is that it is
    // reached by evaluation rather than by a refusal.
    assert_eq!(route.path, AdmissionPath::Legacy);
    assert_ne!(
        route.fallback_reason,
        Some(LegacyAdmissionReason::NoProvenance),
        "provenance was repaired, so the route must not still report it missing"
    );
}

/// sc-16482 FULL end-to-end on real weights (ignored — needs the krea-2-turbo-mlx q8 turnkey).
///
/// The sibling test above proves the admission route stops refusing. This one closes the last gap:
/// it drives the REAL production seam — `evaluate_request`, the call `image_jobs/base.rs` makes
/// immediately before generation, against a live loaded generator — and then actually renders. The
/// base `krea_2_turbo` text-to-image lane at q8/1024² is the exact configuration that was reported
/// broken, so a written PNG is the end of the causal chain the bug report started.
///
///   SCENEWORKS_KREA_MANIFEST_JSON=<entry>.json KREA_BASE_OUT_DIR=<dir> \
///     cargo test -p sceneworks-worker --lib -- --ignored --nocapture krea_real_install_renders
#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight MLX render; needs the krea-2-turbo-mlx q8 turnkey + SCENEWORKS_KREA_MANIFEST_JSON"]
fn krea_real_install_renders_through_the_live_gate() {
    let home = std::path::PathBuf::from(std::env::var("HOME").expect("HOME"));
    let hub = home.join(".cache/huggingface/hub");
    let snapshots = hub.join("models--SceneWorks--krea-2-turbo-mlx/snapshots");
    let Some(q8) = std::fs::read_dir(&snapshots).ok().and_then(|entries| {
        entries.flatten().map(|e| e.path().join("q8")).find(|p| {
            p.join("transformer/diffusion_pytorch_model.safetensors")
                .is_file()
        })
    }) else {
        eprintln!("SKIP: no krea q8 turnkey under {}", snapshots.display());
        return;
    };
    let Some(manifest) = real_krea_manifest_entry() else {
        return;
    };
    let out_dir = std::path::PathBuf::from(crate::smoke_support::env_or(
        "KREA_BASE_OUT_DIR",
        "/tmp/krea_base_smoke",
    ));
    std::fs::create_dir_all(&out_dir).expect("out dir");

    // Real receipt, copied so the real install is never mutated.
    let data = tempfile::tempdir().expect("data dir");
    let marker_dir = data
        .path()
        .join("models")
        .join(crate::paths::safe_download_dir(
            "SceneWorks/krea-2-turbo-mlx",
        ));
    std::fs::create_dir_all(&marker_dir).expect("marker dir");
    std::fs::copy(
        home.join("SceneWorks/data/models/SceneWorks__krea-2-turbo-mlx")
            .join(crate::INSTALL_MARKER),
        marker_dir.join(crate::INSTALL_MARKER),
    )
    .expect("copy real receipt");

    let provenance =
        crate::test_env::temp_env_var("HF_HUB_CACHE", hub.to_str().expect("hub path"), || {
            crate::model_jobs::huggingface_receipt_weights(
                data.path(),
                "SceneWorks/krea-2-turbo-mlx",
                Some("krea_2_turbo"),
                Some("q8"),
                crate::model_jobs::ProvenanceRepair::Allow,
            )
            .expect("resolves")
            .provenance
            .expect("repair establishes provenance for the real install")
        });

    let spec = LoadSpec::new(WeightsSource::Dir(q8.clone())).with_quant(gen_core::Quant::Q8);
    let plan = MlxRequestPlan::for_spec_and_manifest(
        "krea_2_turbo",
        "krea_2_turbo",
        &spec,
        Some(&manifest),
        Some(provenance),
    );
    eprintln!("[smoke] loading krea_2_turbo q8 from {}", q8.display());
    let generator =
        crate::inference_runtime::load("krea_2_turbo", &spec).expect("load krea_2_turbo");

    // THE production call: exactly what base.rs runs per generation, on a live generator.
    let evaluation = evaluate_request(
        &*generator,
        &plan,
        &fixture_inputs(1024, 1024),
        MemoryCacheState::Warm,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert("fixture"),
        0,
    )
    .expect("the reported generation must be admitted, not refused");
    eprintln!("[smoke] gate admitted: {evaluation:?}");

    let request = gen_core::GenerationRequest {
        prompt: "a windswept basalt sea stack at dawn, low mist, long exposure water".to_owned(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(20260801),
        steps: Some(8),
        // Krea Turbo is CFG-free (distilled).
        guidance: None,
        cancel: gen_core::CancelFlag::new(),
        ..Default::default()
    };
    let image = match generator
        .generate(&request, &mut |_| {})
        .expect("krea_2_turbo generate")
    {
        gen_core::GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected Images, got {other:?}"),
    };
    let path = out_dir.join("krea_base_q8_1024.png");
    crate::smoke_support::save_png(&image, &path);
    let std_dev = crate::smoke_support::image_std(&image);
    eprintln!("[smoke] rendered {} (std {std_dev:.2})", path.display());
    assert_eq!((image.width, image.height), (1024, 1024));
    assert!(
        !crate::smoke_support::is_all_zero(&image),
        "render is entirely black"
    );
    assert!(
        std_dev > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
        "render looks degenerate (std {std_dev:.2})"
    );
}

/// sc-10894 end-to-end: a non-zero footprint text encoder flips the residency decision from Reject to
/// Sequential where the zero-reading subdir scan (the fallback) would reject. This is the whole point
/// of the seam — the staged working set is only real when the text-encoder split is measured. Post
/// sc-12179 the flip runs through the weights-fit floor: the measured TE lowers the staged WEIGHTS
/// (the wired residency), which is what legacy admits against `budget − reserve`.
#[test]
fn footprint_text_encoder_flips_reject_to_sequential() {
    let gib = 1024 * 1024 * 1024_u64;
    // boogu-class: whole model 22 GiB (mllm 13 + transformer 8 + vae 1). No `text_encoder*` subdir,
    // so the subdir scan reads 0.
    let total = 22 * gib;
    // Budget where the staged WEIGHTS decide it: floor ceiling = 22 − 2 = 20 GiB. te=0 ⇒ staged
    // weights = 22 > 20 (reject); te=13 ⇒ staged weights = max(13, 9) = 13 ≤ 20 (Sequential).
    let budget = Some(MlxMemoryBudget { total_gb: 22.0 });

    // Fallback path (footprint None on a dir with no `text_encoder*`) → te = 0 → staged weights ==
    // whole model (22 GiB) > 20 → Reject even under the floor: one component IS the whole model.
    let te_fallback = resolve_text_encoder_bytes(None, std::path::Path::new("/nonexistent"));
    assert_eq!(te_fallback, 0);
    assert!(matches!(
        decide_residency(total, te_fallback, budget, true),
        ResidencyOutcome::Reject { .. }
    ));

    // Provider footprint (te = 13 GiB) → staged weights = max(13, 22 − 13 = 9) = 13 ≤ 20 → Sequential.
    let te_footprint = resolve_text_encoder_bytes(Some(13 * gib), std::path::Path::new("/ignored"));
    assert_eq!(te_footprint, 13 * gib);
    assert_eq!(
        decide_residency(total, te_footprint, budget, true),
        ResidencyOutcome::Sequential
    );
}

// -----------------------------------------------------------------------------------------
// Mochi 1 frame-aware decode gate (epic 1788 / sc-11992)
// -----------------------------------------------------------------------------------------

/// Mochi's q4 resident weights (GiB): DiT 9.007 + T5-XXL bf16 8.871 + VAE 0.856 = 18.73 (the
/// exact hosted bytes B1's manifest derivation pins). Used as the weight signal in the gate tests.
const MOCHI_Q4_RESIDENT_BYTES: u64 = 9_670_883_602 + 9_524_669_250 + 919_551_200;

// The pure decode arithmetic (`mochi_decode_peak_gb`: linearity in frames + pixels, the f32
// anchor) is pinned in `crate::fit_gate`'s tests alongside the formula, which moved there in
// sc-12306 when the candle video lane grew the same gate. What stays here is what is genuinely
// MLX: the unified-memory budget shape, the typed unified reserve, and the Mac-worded message.

/// THE gate behavior: on ONE fixed machine, a short Mochi clip is admitted and a long one is
/// rejected. A frame-blind gate (the plausible wrong implementation — reusing `predicted_peak_gb`
/// or dropping the frames term) cannot pass this: it would give both clips the same verdict.
#[test]
fn mochi_gate_admits_a_short_clip_and_rejects_a_long_one_on_the_same_mac() {
    // A 64 GB Mac — the machine B1's `mlx.minMemoryGb: 96` says is under-provisioned, and the one
    // the epic's crash report names.
    let mac_64 = Some(MlxMemoryBudget { total_gb: 64.0 });

    // 19 frames (the engine's own DEFAULT_FRAMES, ~0.6 s): 18.73 weights + 9.32 decode + 2 OS
    // ≈ 30.1 GiB ⇒ admitted. (All GiB — see `mochi_decode_peak_gb`.)
    assert!(
        mochi_fit_error("mochi_1", MOCHI_Q4_RESIDENT_BYTES, 19, 848, 480, mac_64).is_none(),
        "a 19-frame clip fits a 64 GB Mac and must NOT be rejected"
    );

    // 151 frames (the shipped 5 s default): 18.73 + 60.56 + 2 ≈ 81.3 GiB ⇒ rejected BEFORE the
    // untiled decode can trip MLX's `exit(-1)`.
    assert!(
        mochi_fit_error("mochi_1", MOCHI_Q4_RESIDENT_BYTES, 151, 848, 480, mac_64).is_some(),
        "a 151-frame clip needs ~81 GB and MUST be rejected on a 64 GB Mac — this is the \
             unmappable exit(-1) the gate exists to prevent"
    );

    // The same 151-frame clip is admitted on a 128 GB Mac — the gate rejects by BUDGET, it does
    // not blanket-ban the default duration.
    let mac_128 = Some(MlxMemoryBudget { total_gb: 128.0 });
    assert!(
        mochi_fit_error("mochi_1", MOCHI_Q4_RESIDENT_BYTES, 151, 848, 480, mac_128).is_none(),
        "a 151-frame clip fits a 128 GB Mac"
    );
}

/// The rejection message must be self-contained + actionable, following the `too_big_error`
/// convention (name the model, explain the constraint, state need vs budget) and additionally
/// naming Mochi's one real lever: clip length.
#[test]
fn mochi_too_big_error_names_model_budget_and_the_clip_length_lever() {
    let mac_64 = Some(MlxMemoryBudget { total_gb: 64.0 });
    let Some(WorkerError::InvalidPayload(message)) =
        mochi_fit_error("mochi_1", MOCHI_Q4_RESIDENT_BYTES, 151, 848, 480, mac_64)
    else {
        panic!("expected an InvalidPayload rejection");
    };
    assert!(message.contains("mochi_1"), "names the model: {message}");
    assert!(
        message.contains("unified memory"),
        "explains the constraint: {message}"
    );
    assert!(
        message.contains("81"),
        "states what it needs (~81 GB): {message}"
    );
    assert!(message.contains("64"), "states the budget: {message}");
    assert!(
        message.contains("151"),
        "names the clip length that drove it: {message}"
    );
    assert!(message.contains("848x480"), "names the geometry: {message}");
    assert!(
        message.contains("Shorten the clip"),
        "gives the actionable lever: {message}"
    );
}

/// No signal ⇒ never block, matching `fit_decision`/`predicted_peak_gb`. An unmeasurable model
/// dir or a machine with no probe must not wall off generation.
#[test]
fn mochi_gate_never_blocks_without_a_signal() {
    let mac_64 = Some(MlxMemoryBudget { total_gb: 64.0 });
    // Weights unmeasurable (empty/missing dir ⇒ 0 bytes) ⇒ admit.
    assert!(mochi_fit_error("mochi_1", 0, 151, 848, 480, mac_64).is_none());
    assert!(mochi_needed_gb(
        0,
        151,
        848,
        480,
        crate::fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB
    )
    .is_none());
    // No budget (off-Mac / probe failed) ⇒ admit.
    assert!(mochi_fit_error("mochi_1", MOCHI_Q4_RESIDENT_BYTES, 151, 848, 480, None).is_none());
}

/// `mochi_resident_bytes` must fold the SHARED `text_encoder/` + `vae/` siblings resolved from the
/// tier dir's PARENT — they are over half the resident footprint, and summing only the tier dir
/// would under-count by ~9.7 GiB and admit a job that then dies.
#[test]
fn mochi_resident_bytes_folds_the_shared_parent_siblings() {
    let root_guard = tempfile::Builder::new()
        .prefix("mochi_resident_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    // The A6 installed layout: tier dirs as siblings of the shared components.
    for (sub, bytes) in [
        ("q4/transformer", 400_usize),
        ("text_encoder", 300),
        ("vae", 200),
        // A sibling tier that must NOT be counted into the q4 load.
        ("q8/transformer", 999),
    ] {
        let dir = root.join(sub);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).unwrap();
    }

    let tier_dir = root.join("q4");
    assert_eq!(
        mochi_resident_bytes(&tier_dir),
        400 + 300 + 200,
        "tier transformer + the shared text_encoder/vae siblings, and NOT the q8 tier"
    );

    // A self-contained dir (the raw snapshot: components UNDER the dir) is summed once, not
    // double-counted via the parent scan.
    let flat = root.join("flat");
    for (sub, bytes) in [
        ("transformer", 400_usize),
        ("text_encoder", 300),
        ("vae", 200),
    ] {
        let dir = flat.join(sub);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).unwrap();
    }
    assert_eq!(
        mochi_resident_bytes(&flat),
        900,
        "a self-contained snapshot counts its own components exactly once"
    );
}

/// sc-19721 — the AdaLN exclusion, at the only consumer that can honour it.
///
/// gen-core landed `MemoryComponentKind::TransformerSubStack` +
/// `MemoryComponentResidency::PrecomputedThenEvicted` (SC-18665) with **zero** non-test callers,
/// so the declaration moved no estimate by a byte. These grade the consumer, and they grade the
/// half that is easy to get wrong in the OOM direction: the drop lowers the STEADY STATE, and
/// the declaring phase still holds the whole sub-stack at the precompute instant.
///
/// Every figure is MiniMax-H3's own declaration, tied to the pinned engine's public constants by
/// [`the_h3_eviction_figures_are_the_pinned_engines_own`]. Nothing is asserted against a default:
/// `evicted_component_bytes()` is asserted non-zero and equal to `resident − retained` first, so
/// a contract that declared no component (or a build with the feature deleted) cannot pass by
/// comparing zero with zero.
mod adaln_exclusion_reaches_the_estimate_floor {
    use super::*;

    /// Qwen3-VL-32B, the dense conditioning stack — `mlx_gen_minimax_h3::TEXT_ENCODER_BYTES`.
    const TEXT_ENCODER_BYTES: u64 = 66_714_912_872;
    /// One bf16 33B DiT partition — `DIT_BF16_BYTES`. A render loads exactly one.
    const DIT_BF16_BYTES: u64 = 66_280_504_216;
    /// Video VAE + audio VAE: the decode-phase components, i.e. the part of the `heavy` lump
    /// that is NOT the transformer.
    const VAE_BYTES: u64 = 10_415_558_888 + 605_429_340;
    /// The 50-block `adaln_proj` stack at bf16 — `ADALN_EVICTED_BYTES`.
    const ADALN_RESIDENT_BF16_BYTES: u64 = 26_020_915_200;
    /// The same 50 projections on the packed q4 tier — `ADALN_EVICTED_Q4_BYTES`. **The lever is
    /// tier-scaled**, which is why q4 is graded here and not only bf16.
    const ADALN_RESIDENT_Q4_BYTES: u64 = 7_325_337_600;
    /// What the precompute keeps in the projections' place —
    /// `ADALN_MODULATION_TABLE_MAX_BYTES`. Deliberately **not** tier-scaled: the table's dtype is
    /// the compute dtype, not the tier's bit width. Applying one factor to both would be wrong at
    /// every tier but bf16, and this pair is what makes that visible.
    const ADALN_RETAINED_BYTES: u64 = 3_870_720_000;

    /// A packed conditioning tier (sc-19120 re-hosts one). Used as a STAND-IN so the DiT-side
    /// arithmetic is observable at all: with the dense encoder above, `max(conditioning, heavy)`
    /// is pinned by the conditioner at q4 and the whole exclusion is invisible at the floor —
    /// itself a finding, graded by [`q4_with_the_dense_conditioner_moves_nothing`].
    const PACKED_TEXT_ENCODER_BYTES: u64 = TEXT_ENCODER_BYTES / 4;

    /// A conditioning stack SHORTER than the decoder — the stand-in
    /// [`rung_four_deducts_the_transformer_once_not_twice`] needs, one step past
    /// [`PACKED_TEXT_ENCODER_BYTES`].
    ///
    /// The staged branch's floor is `conditioning.max(heavy)`. Above `VAE_BYTES` the conditioner
    /// wins that `max` and `heavy` is INVISIBLE — which is exactly how the rung-4 test passed
    /// for arithmetic that had erased the decoder from `heavy` entirely (sc-18650 pre-merge
    /// review). Below `VAE_BYTES`, `heavy` is the binding leg and the assertion reads the number
    /// it names.
    const SUB_DECODER_CONDITIONER_BYTES: u64 = PACKED_TEXT_ENCODER_BYTES / 2;

    const STAGED: &[MemoryStrategy] = &[MemoryStrategy::StagedResidency];
    const CO_RESIDENT: &[MemoryStrategy] = &[];
    const STAGED_PLUS_RUNG4: &[MemoryStrategy] = &[
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedTransformerResidency,
    ];

    /// MiniMax-H3's contract shape: three base components, and one `TransformerSubStack`
    /// component inside `transformer_bytes` that is precomputed and evicted during Denoise.
    /// `evicting = false` builds the byte-identical contract a provider had before SC-18665
    /// existed — the control every delta below is measured against.
    fn h3_shaped_contract(
        conditioning_bytes: u64,
        dit_bytes: u64,
        adaln_resident_bytes: u64,
        evicting: bool,
    ) -> MemoryProviderContract {
        let mut contract = MemoryProviderContract::compatibility_default(
            "minimax_h3",
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: false,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: false,
                cache_eviction: true,
            },
        );
        contract.asset_facts.conditioning_bytes = conditioning_bytes;
        contract.asset_facts.transformer_bytes = dit_bytes;
        contract.asset_facts.decoder_bytes = VAE_BYTES;
        contract.asset_facts.base_bytes = conditioning_bytes + dit_bytes + VAE_BYTES;
        let phases = contract.lifecycle.phases.clone();
        contract.formula = gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables: vec![gen_core::MemoryFormulaVariable::AssetBytes],
            resident_components: vec![gen_core::MemoryResidentComponent {
                id: "dit_adaln_proj_stack".to_owned(),
                kind: gen_core::MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
                resident_bytes: adaln_resident_bytes,
                bounded_by: None,
                residency: if evicting {
                    gen_core::MemoryComponentResidency::PrecomputedThenEvicted {
                        precomputed_in: gen_core::MemoryPhase::Denoise,
                        retained_bytes: ADALN_RETAINED_BYTES,
                        evidence: "sc-19721 fixture mirroring \
                                       mlx-gen-minimax-h3::memory_strategy::adaln_component"
                            .to_owned(),
                    }
                } else {
                    gen_core::MemoryComponentResidency::WholeRender
                },
            }],
        };
        contract
    }

    /// The declared drop, read off the contract rather than recomputed — and proved non-zero and
    /// equal to the provider's own `resident − retained`, so none of the deltas below can be a
    /// zero-versus-zero pass.
    fn declared_eviction(contract: &MemoryProviderContract, adaln_resident: u64) -> u64 {
        let evicted = contract.evicted_component_bytes();
        assert_eq!(
            evicted,
            adaln_resident - ADALN_RETAINED_BYTES,
            "the fixture must declare the NET drop the provider declares: the precompute keeps \
                 a modulation table in the projections' place, and a gross declaration claims a \
                 saving the runtime does not deliver"
        );
        assert!(
            evicted > 0,
            "a zero drop would make every delta below vacuous"
        );
        assert_eq!(
            contract.steady_state_transformer_bytes(),
            contract.asset_facts.transformer_bytes - evicted,
            "the accessor under test must correct transformer_bytes by exactly that drop"
        );
        evicted
    }

    /// bf16 + the dense Qwen3-VL-32B conditioner: the shipped cell today.
    ///
    /// The lump `heavy = transformer + decoder` is 77.30 GB against a 66.71 GB conditioner, so
    /// the staged floor is the lump. The drop cancels against the decoder's bytes — which the
    /// denoise phase does not hold — until it hits the load-exact transformer, and there the
    /// clamp stops it. The conditioner is then the taller leg, so the floor lands on it.
    #[test]
    fn bf16_staged_floor_falls_to_the_conditioning_stack_and_never_below_the_transformer() {
        let legacy = h3_shaped_contract(
            TEXT_ENCODER_BYTES,
            DIT_BF16_BYTES,
            ADALN_RESIDENT_BF16_BYTES,
            false,
        );
        let adopted = h3_shaped_contract(
            TEXT_ENCODER_BYTES,
            DIT_BF16_BYTES,
            ADALN_RESIDENT_BF16_BYTES,
            true,
        );
        let evicted = declared_eviction(&adopted, ADALN_RESIDENT_BF16_BYTES);
        assert_eq!(
            legacy.evicted_component_bytes(),
            0,
            "the control declares WholeRender, so it drops nothing"
        );

        let before = estimate_floor_weights_bytes(&legacy, STAGED);
        let after = estimate_floor_weights_bytes(&adopted, STAGED);
        assert_eq!(
            before,
            DIT_BF16_BYTES + VAE_BYTES,
            "before: the staged floor is the transformer+decoder lump"
        );
        assert_eq!(
            after, TEXT_ENCODER_BYTES,
            "after: the lump falls below the conditioning stack, which becomes the binding leg"
        );
        assert!(
            before > after,
            "a configuration between {after} and {before} bytes was refused and now fits"
        );
        assert!(
            after >= DIT_BF16_BYTES,
            "the precompute instant holds the WHOLE DiT ({DIT_BF16_BYTES} B, sub-stack \
                 included). A floor below it under-charges that instant by up to {evicted} B — the \
                 OOM direction, and the exact asymmetry ADALN_MODULATION_TABLE_MAX_BYTES exists to \
                 avoid."
        );
    }

    /// bf16 + a packed conditioner: the clamp is the ONLY thing stopping the fall, and the floor
    /// lands exactly on the load-exact transformer. This is the cell that would go silently
    /// wrong if the eviction were subtracted raw.
    #[test]
    fn bf16_staged_floor_stops_exactly_at_the_load_exact_transformer() {
        let legacy = h3_shaped_contract(
            PACKED_TEXT_ENCODER_BYTES,
            DIT_BF16_BYTES,
            ADALN_RESIDENT_BF16_BYTES,
            false,
        );
        let adopted = h3_shaped_contract(
            PACKED_TEXT_ENCODER_BYTES,
            DIT_BF16_BYTES,
            ADALN_RESIDENT_BF16_BYTES,
            true,
        );
        let evicted = declared_eviction(&adopted, ADALN_RESIDENT_BF16_BYTES);
        assert!(
            evicted > VAE_BYTES,
            "this cell only means something while the drop is bigger than the decode-phase \
                 bytes it can cancel against"
        );

        let before = estimate_floor_weights_bytes(&legacy, STAGED);
        let after = estimate_floor_weights_bytes(&adopted, STAGED);
        assert_eq!(before, DIT_BF16_BYTES + VAE_BYTES);
        assert_eq!(
            after, DIT_BF16_BYTES,
            "the fall stops at the load-exact transformer — the precompute instant — so the \
                 reduction is the decode-phase bytes ({VAE_BYTES}), never the raw drop ({evicted})"
        );
        assert_eq!(before - after, VAE_BYTES);
    }

    /// q4 + a packed conditioner: neither clamp binds, so the floor falls by EXACTLY
    /// `evicted_component_bytes()`. Grading a second tier is not redundant — the resident side
    /// is tier-scaled (26.02 → 7.33 GB) while the retained table is not, so a single factor
    /// applied to both would be right at bf16 and wrong here.
    #[test]
    fn q4_staged_floor_falls_by_exactly_the_declared_eviction() {
        let legacy = h3_shaped_contract(
            PACKED_TEXT_ENCODER_BYTES,
            DIT_BF16_BYTES / 4,
            ADALN_RESIDENT_Q4_BYTES,
            false,
        );
        let adopted = h3_shaped_contract(
            PACKED_TEXT_ENCODER_BYTES,
            DIT_BF16_BYTES / 4,
            ADALN_RESIDENT_Q4_BYTES,
            true,
        );
        let evicted = declared_eviction(&adopted, ADALN_RESIDENT_Q4_BYTES);
        assert!(
            evicted < VAE_BYTES,
            "at q4 the drop is smaller than the decode-phase bytes, so the clamp does not bind \
                 and the whole declared exclusion reaches the floor"
        );
        assert_ne!(
            evicted,
            ADALN_RESIDENT_BF16_BYTES - ADALN_RETAINED_BYTES,
            "the exclusion must be tier-scaled; a q4 drop equal to bf16's is the error this \
                 cell exists to catch"
        );

        let before = estimate_floor_weights_bytes(&legacy, STAGED);
        let after = estimate_floor_weights_bytes(&adopted, STAGED);
        assert_eq!(
            before - after,
            evicted,
            "the whole declared exclusion reaches the fit gate at this cell"
        );
        assert_eq!(after, DIT_BF16_BYTES / 4 + VAE_BYTES - evicted);
    }

    /// q4 + the dense conditioner: the floor does NOT move, because the 66.71 GB conditioning
    /// stack is taller than the whole packed DiT pipeline. Pinned rather than left unsaid: it is
    /// why sc-19120's packed text encoder and this change are a pair, and why a q4 assertion
    /// against the shipped dense encoder would have looked like the fix doing nothing.
    #[test]
    fn q4_with_the_dense_conditioner_moves_nothing() {
        let legacy = h3_shaped_contract(
            TEXT_ENCODER_BYTES,
            DIT_BF16_BYTES / 4,
            ADALN_RESIDENT_Q4_BYTES,
            false,
        );
        let adopted = h3_shaped_contract(
            TEXT_ENCODER_BYTES,
            DIT_BF16_BYTES / 4,
            ADALN_RESIDENT_Q4_BYTES,
            true,
        );
        declared_eviction(&adopted, ADALN_RESIDENT_Q4_BYTES);
        assert_eq!(
            estimate_floor_weights_bytes(&legacy, STAGED),
            TEXT_ENCODER_BYTES
        );
        assert_eq!(
            estimate_floor_weights_bytes(&adopted, STAGED),
            TEXT_ENCODER_BYTES,
            "the dense conditioning stack still binds; the DiT-side win is real but invisible \
                 here"
        );
    }

    /// Without `StagedResidency` nothing is staged out of the precompute instant: the
    /// conditioner, the transformer and the decoder are charged as ONE co-residency, and that
    /// co-residency includes the instant. The floor must not move by a byte.
    #[test]
    fn a_co_resident_composition_takes_no_reduction_at_all() {
        for (conditioning, dit, adaln) in [
            (
                TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES,
                ADALN_RESIDENT_BF16_BYTES,
            ),
            (
                PACKED_TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES,
                ADALN_RESIDENT_BF16_BYTES,
            ),
            (
                PACKED_TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES / 4,
                ADALN_RESIDENT_Q4_BYTES,
            ),
        ] {
            let legacy = h3_shaped_contract(conditioning, dit, adaln, false);
            let adopted = h3_shaped_contract(conditioning, dit, adaln, true);
            declared_eviction(&adopted, adaln);
            assert_eq!(
                estimate_floor_weights_bytes(&adopted, CO_RESIDENT),
                estimate_floor_weights_bytes(&legacy, CO_RESIDENT),
                "co-resident floor moved for conditioning={conditioning} dit={dit}: the \
                     precompute instant is inside this charge and would be under-charged"
            );
            assert_eq!(
                estimate_floor_weights_bytes(&adopted, CO_RESIDENT),
                conditioning + dit + VAE_BYTES
            );
        }
    }

    /// Declare the rung-4 → rung-1 edge on an [`h3_shaped_contract`], so the composition under
    /// test is one a SELECTOR can produce rather than a hand-written slice.
    ///
    /// `MemoryStrategy::engages` deliberately does not make rung 4 engage rung 1 — the shared
    /// prerequisite is `LoadShape::DeferredMaterialization`, which needs no residency policy —
    /// so `staged + rung 4` exists only where a provider appends the edge through
    /// `additional_prerequisites`. Thirteen shipped MLX providers do (sana, flux, flux2, chroma,
    /// kolors, krea ×2, lens, mage, qwen-image, sd3, sdxl, anima), each pushing exactly this
    /// pair. MiniMax-H3 itself declares `additional_prerequisites: Vec::new()`, which is why
    /// this composition moves no live admission TODAY and why the arithmetic it grades is one
    /// provider declaration away from mattering.
    fn rung4_edge_contract(mut contract: MemoryProviderContract) -> MemoryProviderContract {
        for rung in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            contract
                .strategies
                .iter_mut()
                .find(|capability| capability.strategy == rung)
                .expect("the compatibility contract declares every rung's capability")
                .support = gen_core::MemoryStrategySupport::Implemented;
        }
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            gen_core::MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
        contract
    }

    /// Rung 4 windows the WHOLE transformer, sub-stack included, so the eviction must not be
    /// deducted a second time on top of it. BOTH legs remove the transformer, and what is left
    /// is the DECODER — the component rung 4 does not window and which stays resident.
    ///
    /// # This test was a false green until sc-18650's pre-merge review
    ///
    /// It ran on [`PACKED_TEXT_ENCODER_BYTES`], which is TALLER than [`VAE_BYTES`], so the
    /// staged branch's `conditioning.max(heavy)` threw `heavy` away and both arms collapsed onto
    /// the conditioner. It passed whether the arithmetic deducted the transformer once or
    /// twice — and it was deducting it twice, leaving `max(0, decoder − evicted)`, which at bf16
    /// is exactly **0**: MiniMax-H3's whole 11.02 GB video+audio VAE pair gone from the floor.
    /// Its own expected value, `PACKED_TEXT_ENCODER_BYTES.max(VAE_BYTES)`, was inert for the
    /// same reason — the `.max` never chose its right-hand side.
    ///
    /// So the conditioner is now deliberately BELOW the decoder, which puts `heavy` on the
    /// binding side of that `max`, and the composition is read off `engaged_composition` on a
    /// contract that declares the edge instead of being written out by hand.
    #[test]
    fn rung_four_deducts_the_transformer_once_not_twice() {
        let legacy = rung4_edge_contract(h3_shaped_contract(
            SUB_DECODER_CONDITIONER_BYTES,
            DIT_BF16_BYTES,
            ADALN_RESIDENT_BF16_BYTES,
            false,
        ));
        let adopted = rung4_edge_contract(h3_shaped_contract(
            SUB_DECODER_CONDITIONER_BYTES,
            DIT_BF16_BYTES,
            ADALN_RESIDENT_BF16_BYTES,
            true,
        ));
        // The two properties that make this fixture able to SEE the arithmetic, asserted off the
        // contract's own facts so a later constant edit cannot quietly re-mask the test.
        assert!(
            adopted.asset_facts.conditioning_bytes < adopted.asset_facts.decoder_bytes,
            "the conditioner has to be the SHORTER leg of `conditioning.max(heavy)`, or that \
                 `max` hides the answer this test exists to read ({} vs {})",
            adopted.asset_facts.conditioning_bytes,
            adopted.asset_facts.decoder_bytes
        );
        let evicted = declared_eviction(&adopted, ADALN_RESIDENT_BF16_BYTES);
        assert!(
            evicted > adopted.asset_facts.decoder_bytes,
            "the eviction must be able to swallow the decoder whole ({evicted} vs {}), or a \
                 double deduction would understate the floor partially rather than erasing the \
                 decode phase outright",
            adopted.asset_facts.decoder_bytes
        );

        // The composition is the PROVIDER'S, not this test's. Without the declared edge rung 4
        // engages rungs 2 and 3 but never rung 1, so `StagedResidency` appearing here is the
        // edge doing its job — and `estimate_floor_weights_bytes` reads exactly these two.
        let composition = adopted.engaged_composition(MemoryStrategy::BoundedTransformerResidency);
        assert_eq!(
            composition,
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedTransformerResidency,
            ],
            "the declared rung-4 -> rung-1 edge is what makes this composition reachable"
        );
        assert_eq!(
            legacy.engaged_composition(MemoryStrategy::BoundedTransformerResidency),
            composition,
            "adopting the sub-stack vocabulary must not change which rungs engage"
        );

        assert_eq!(
            estimate_floor_weights_bytes(&adopted, &composition),
            estimate_floor_weights_bytes(&legacy, &composition),
            "the declared eviction lives INSIDE the window rung 4 removes, so adopting the \
                 vocabulary must not move this floor by a byte"
        );
        assert_eq!(
            estimate_floor_weights_bytes(&adopted, &composition),
            VAE_BYTES,
            "the windowed transformer leaves the floor exactly once, and what remains is the \
                 decoder rung 4 does not window"
        );
    }

    /// The 23 providers that never adopted the sub-stack vocabulary must be byte-identical.
    /// `steady_state_transformer_bytes()` returns `asset_facts.transformer_bytes` unchanged for
    /// them, which is the property that makes this change safe to land at the shared floor.
    #[test]
    fn a_provider_that_declares_no_sub_stack_is_byte_identical() {
        let mut contract = MemoryProviderContract::compatibility_default(
            "fixture_provider",
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: false,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: false,
                cache_eviction: true,
            },
        );
        contract.asset_facts.conditioning_bytes = gib_to_bytes(1.0);
        contract.asset_facts.transformer_bytes = gib_to_bytes(5.0);
        contract.asset_facts.decoder_bytes = gib_to_bytes(2.0);
        contract.asset_facts.base_bytes = gib_to_bytes(8.0);
        assert!(contract.resident_components().is_empty());
        assert_eq!(contract.evicted_component_bytes(), 0);
        assert_eq!(
            contract.steady_state_transformer_bytes(),
            contract.asset_facts.transformer_bytes
        );
        assert_eq!(intra_transformer_evicted_bytes(&contract), 0);

        assert_eq!(
            estimate_floor_weights_bytes(&contract, STAGED),
            gib_to_bytes(7.0),
            "staged: max(conditioning, transformer + decoder)"
        );
        assert_eq!(
            estimate_floor_weights_bytes(&contract, CO_RESIDENT),
            gib_to_bytes(8.0),
            "co-resident: the whole base"
        );
        assert_eq!(
            estimate_floor_weights_bytes(&contract, STAGED_PLUS_RUNG4),
            gib_to_bytes(2.0),
            "rung 4: max(conditioning, decoder) with the transformer windowed out"
        );
    }

    /// sc-22667 (epic 22657 feature-end round, E3/E4): with a block count on the facts the
    /// IMAGE lane's rung-4 floor keeps the law's window share of the transformer resident —
    /// the same `windowed_transformer_bytes` the candle ladder and the law price — where the
    /// video lane's form removes the whole transformer (its window is carried by its headroom
    /// term, sc-18096). With the DEFAULT facts (no block count, this pin) the image floor
    /// keeps that pre-epic accounting too, so admission on real Macs is unchanged until the
    /// pin bump supplies block counts.
    ///
    /// MUTATION: passing `0` as the resident window whatever the facts reds the first two
    /// assertions; keeping the whole transformer under the default facts with the generic
    /// headroom (dropping the `GenericHeadroom` arm) reds the third; removing it under the
    /// default facts with the law's residue (the pre-round-2 code, window charged nowhere)
    /// reds the fourth; a divergence from the law's helper reds the last.
    #[test]
    fn the_image_floor_keeps_the_laws_window_share_resident_and_errs_large_without_facts() {
        use ImageFloorActivationTerm::{GenericHeadroom, LawResidue};

        let mut contract = h3_shaped_contract(
            gib_to_bytes(1.0),
            gib_to_bytes(5.0),
            ADALN_RESIDENT_Q4_BYTES,
            false,
        );
        contract.asset_facts.conditioning_bytes = gib_to_bytes(1.0);
        contract.asset_facts.transformer_bytes = gib_to_bytes(5.0);
        contract.asset_facts.decoder_bytes = gib_to_bytes(2.0);
        contract.asset_facts.base_bytes = gib_to_bytes(8.0);
        let facts = sceneworks_core::memory_anchor::ArchitectureFacts {
            transformer_blocks: Some(10),
            ..Default::default()
        };
        let blind = sceneworks_core::memory_anchor::ArchitectureFacts::default();
        for term in [LawResidue, GenericHeadroom] {
            assert_eq!(
                image_floor_weights_bytes(&contract, STAGED_PLUS_RUNG4, Some(1), facts, term),
                gib_to_bytes(2.5),
                "{term:?}: rung 4 with one of ten blocks resident: max(conditioning, decoder \
                     + 0.5 GiB)"
            );
            assert_eq!(
                image_floor_weights_bytes(&contract, STAGED_PLUS_RUNG4, Some(4), facts, term),
                gib_to_bytes(4.0),
                "{term:?}: four of ten blocks: decoder + 2 GiB"
            );
        }
        assert_eq!(
            image_floor_weights_bytes(
                &contract,
                STAGED_PLUS_RUNG4,
                Some(1),
                blind,
                GenericHeadroom
            ),
            gib_to_bytes(2.0),
            "no block count under the generic headroom: the PRE-EPIC accounting — the whole \
                 transformer out, its window carried by the headroom term — so real-Mac admission \
                 is unchanged"
        );
        assert_eq!(
            image_floor_weights_bytes(&contract, STAGED_PLUS_RUNG4, Some(1), blind, LawResidue),
            gib_to_bytes(7.0),
            "no block count under the law's residue: the whole transformer stays resident \
                 (decoder + 5 GiB), because the residue carries no window slice"
        );
        assert_eq!(
            image_floor_weights_bytes(&contract, STAGED_PLUS_RUNG4, Some(1), blind, LawResidue),
            image_floor_weights_bytes(&contract, STAGED, Some(1), blind, LawResidue),
            "…which is rung 3's weights term: no saving is promised until the share is knowable"
        );
        for term in [LawResidue, GenericHeadroom] {
            assert_eq!(
                image_floor_weights_bytes(&contract, STAGED, Some(1), facts, term),
                estimate_floor_weights_bytes(&contract, STAGED),
                "{term:?}: a composition that does not window is unchanged by the window inputs"
            );
        }
        assert_eq!(
            estimate_floor_weights_bytes(&contract, STAGED_PLUS_RUNG4),
            gib_to_bytes(2.0),
            "the video lane's form keeps its sc-18096 accounting: the whole transformer leaves"
        );
        assert_eq!(
            image_floor_weights_bytes(&contract, STAGED_PLUS_RUNG4, Some(3), facts, LawResidue)
                - estimate_floor_weights_bytes(&contract, STAGED_PLUS_RUNG4),
            sceneworks_core::memory_anchor::windowed_transformer_bytes(
                gib_to_bytes(5.0),
                Some(3),
                Some(10)
            ),
            "the share is the law's helper, byte for byte"
        );
    }

    /// An eviction declared on an AUXILIARY network is not inside `transformer_bytes`, so it
    /// must not be subtracted from the transformer's term. This is the distinction between
    /// `steady_state_transformer_bytes()` and `evicted_component_bytes()`, and swapping one for
    /// the other is invisible on MiniMax-H3 (whose only component is the sub-stack) — which is
    /// exactly why it is graded on a contract that has both.
    #[test]
    fn an_evicting_auxiliary_component_does_not_move_the_transformer_term() {
        const CONTROL_RESIDENT: u64 = 4_000_000_000;
        const CONTROL_RETAINED: u64 = 1_000_000_000;
        let mut contract = h3_shaped_contract(
            PACKED_TEXT_ENCODER_BYTES,
            DIT_BF16_BYTES / 4,
            ADALN_RESIDENT_Q4_BYTES,
            true,
        );
        let sub_stack_only = estimate_floor_weights_bytes(&contract, STAGED);
        let intra = intra_transformer_evicted_bytes(&contract);

        if let gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
            resident_components,
            ..
        } = &mut contract.formula
        {
            resident_components.push(gen_core::MemoryResidentComponent {
                id: "fixture.control".to_owned(),
                kind: gen_core::MemoryComponentKind::ControlBranch,
                resident_bytes: CONTROL_RESIDENT,
                bounded_by: None,
                residency: gen_core::MemoryComponentResidency::PrecomputedThenEvicted {
                    precomputed_in: gen_core::MemoryPhase::Denoise,
                    retained_bytes: CONTROL_RETAINED,
                    evidence: "sc-19721 fixture: an evicting auxiliary network".to_owned(),
                },
            });
        } else {
            panic!("fixture declares a ComponentPhaseEnvelope formula");
        }

        assert_eq!(
            contract.evicted_component_bytes(),
            intra + (CONTROL_RESIDENT - CONTROL_RETAINED),
            "the contract-wide accessor now sums BOTH drops"
        );
        assert_eq!(
            intra_transformer_evicted_bytes(&contract),
            intra,
            "…but only the sub-stack's drop is inside transformer_bytes"
        );
        assert_eq!(
            estimate_floor_weights_bytes(&contract, STAGED),
            sub_stack_only + CONTROL_RESIDENT,
            "the auxiliary network adds its LOAD-EXACT residency and removes nothing from the \
                 transformer term; reading evicted_component_bytes() here would take {} B off a \
                 stack that never held them",
            CONTROL_RESIDENT - CONTROL_RETAINED
        );
    }

    /// The fixture figures above are the PINNED engine's own constants, not transcriptions that
    /// can drift from it. Gated to the lanes that have a provider bundle in scope, exactly like
    /// `pinned_engine_geometry`.
    #[cfg(native_lane)]
    #[test]
    fn the_h3_eviction_figures_are_the_pinned_engines_own() {
        use platform_runtime::providers::minimax_h3::memory_strategy as h3;
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        use runtime_cuda as platform_runtime;
        #[cfg(target_os = "macos")]
        use runtime_macos as platform_runtime;

        assert_eq!(TEXT_ENCODER_BYTES, h3::TEXT_ENCODER_BYTES);
        assert_eq!(DIT_BF16_BYTES, h3::DIT_BF16_BYTES);
        assert_eq!(VAE_BYTES, h3::VIDEO_VAE_BYTES + h3::AUDIO_VAE_BYTES);
        assert_eq!(ADALN_RESIDENT_BF16_BYTES, h3::ADALN_EVICTED_BYTES);
        // The tier-scaled sub-stack figures are declared by the MLX engine only: the candle
        // sibling's `memory_strategy` carries the bf16 constant plus a private
        // `resolved_adaln_bytes` that scales it from the staged tier, and publishes no q4
        // `pub const` to bind to. Nothing to tie on that lane, so the tie is macOS-only.
        #[cfg(target_os = "macos")]
        assert_eq!(ADALN_RESIDENT_Q4_BYTES, h3::ADALN_EVICTED_Q4_BYTES);
        assert_eq!(
            ADALN_RETAINED_BYTES,
            h3::ADALN_MODULATION_TABLE_MAX_BYTES,
            "the retained table is NOT tier-scaled; if this constant moves, every tier cell \
                 above changes and must be re-derived rather than re-stamped"
        );
    }
}

/// sc-22667, second pin bump (inference c6d6a4db): the packaged `z_image_turbo` q4 MLX cell is
/// IN DOMAIN through the production seam, and prices rung 4 below rung 2.
///
/// THE FLIP. At a5f643ae the MLX provider's `decoder_bytes` was the WHOLE `vae/` directory —
/// the `decoder.*` tensors a text-to-image render materializes AND the `encoder.*` tensors it
/// never does. `anchor_component_bytes` therefore handed the law a 5,893,275,206-byte resident
/// set against a re-captured conditioning level of 5,834,701,816, the domain guard refused the
/// negative residue, and the production cell sat on its generic weights+headroom estimate
/// floor (core test, then named `..._is_refused_against_the_whole_vae_asset_fact`). c6d6a4db
/// splits the directory by key prefix: `decoder_bytes` is the render-resident half and the
/// encoder becomes a typed `MemoryComponentKind::ReferenceEncoder` auxiliary in
/// `overlay_bytes`.
///
/// WHY THIS TEST LIVES IN THE WORKER, not in core. The core law only ever sees a
/// `ComponentBytes`; what could regress is the SEAM that builds one. `anchor_component_bytes`
/// must read the three BASE legs and NOT add `overlay_bytes` — `base_bytes` is exactly
/// `conditioning + transformer + decoder`, and the auxiliary is charged on top by the
/// contract's own `predicted_peak_from_base` at admission. A seam that summed base + overlay
/// would hand the law 5,893,275,206 again and put this cell straight back on the floor, which
/// is the exact defect the flip removed. The facts below are the pinned provider's own
/// (`mlx-gen-z-image::memory_strategy::component_asset_facts`) for the q4 T2I route; MLX
/// cannot be built on Windows, so they are stated rather than read off a linked registry, and
/// `base_bytes` is asserted to equal the three legs so a restatement cannot drift silently.
///
/// MUTATION (the pre-c6d6a4db shape): fold `overlay_bytes` back into `decoder_bytes` and the
/// derivation returns `None` on every rung — asserted below, so this test fails if the split
/// stops being what admits the cell.
#[test]
fn the_packaged_z_image_mlx_cell_prices_rung_four_below_rung_two_from_the_contracts_base_legs() {
    use sceneworks_core::memory_anchor::{AnchorBackend, AnchorMlxImageDeriveRequest};

    // The pinned MLX contract's asset facts for z_image_turbo q4, text-to-image, no control:
    // the three base legs plus the reference-encoder auxiliary.
    let facts = gen_core::MemoryAssetFacts {
        base_bytes: 2_262_920_192 + 3_465_730_304 + 97_583_622,
        conditioning_bytes: 2_262_920_192,
        transformer_bytes: 3_465_730_304,
        decoder_bytes: 97_583_622,
        overlay_bytes: 67_041_088,
    };
    assert_eq!(
        facts.base_bytes,
        facts.conditioning_bytes + facts.transformer_bytes + facts.decoder_bytes,
        "base_bytes is the three legs; the auxiliary rides in overlay_bytes"
    );

    let components = crate::video_admission::anchor_component_bytes(facts);
    assert_eq!(components.conditioning, facts.conditioning_bytes);
    assert_eq!(components.transformer, facts.transformer_bytes);
    assert_eq!(components.decoder, facts.decoder_bytes);
    assert_eq!(
        components.total(),
        facts.base_bytes,
        "the seam must pass the BASE legs — adding the auxiliary overlay here is the defect \
             that floored this cell at a5f643ae"
    );

    let store = sceneworks_core::memory_anchor::packaged_memory_anchors()
        .expect("the packaged anchor store");
    let anchor = store
        .image_anchor_for("z_image_turbo", AnchorBackend::Mlx, "q4")
        .expect("the packaged z_image_turbo q4 MLX anchor");
    let request = AnchorMlxImageDeriveRequest {
        width: 768,
        height: 768,
    };

    let priced = anchor
        .derive_mlx_image_phase_peaks(request, components)
        .expect(
            "the packaged MLX cell is in the law's domain through the production seam \
                 (this is the sc-22667 flip; it returned None at a5f643ae)",
        );
    assert!(priced.peak_bytes() > 0);

    // Rung ordering on the same cell, through the law the lane prices with.
    let facts_axes = sceneworks_core::memory_anchor::ArchitectureFacts {
        attention_heads: Some(30),
        head_dim: Some(128),
        transformer_blocks: Some(30),
        patch_size: Some(2),
        latent_channels: Some(16),
        vae_spatial_scale: Some(8),
        vae_temporal_scale: None,
        activation_dtype_width: Some(2),
    };
    let derive = |regime| {
        anchor.derive_phase_peaks(
            &sceneworks_core::memory_anchor::ImageDeriveRequest {
                width: 768,
                height: 768,
                batch: 1,
                conditioning_tokens: None,
                regime,
            },
            components,
            facts_axes,
        )
    };
    let rung_2 = derive(sceneworks_core::memory_anchor::RequestRegime::staged())
        .expect("rung 2 prices from the contract's base legs");
    let rung_4 = derive(sceneworks_core::memory_anchor::RequestRegime {
        decode_tile: Some(sceneworks_core::memory_anchor::DecodeTile {
            edge: 512,
            overlap: 128,
        }),
        attention_chunk_scores: Some(64 * 1024 * 1024),
        transformer_window: Some(1),
        staged: true,
    })
    .expect("rung 4 prices from the contract's base legs");
    eprintln!(
        "sc-22667 z_image_turbo q4 mlx 768x768: lane peak {} | rung 2 {rung_2:?} peak {} | \
             rung 4 {rung_4:?} peak {}",
        priced.peak_bytes(),
        rung_2.peak_bytes(),
        rung_4.peak_bytes()
    );
    assert!(
        rung_4.peak_bytes() < rung_2.peak_bytes(),
        "rung 4 {rung_4:?} must price below rung 2 {rung_2:?}"
    );

    // The mutation: the pre-c6d6a4db whole-directory decoder floors the cell again.
    let whole_vae = crate::video_admission::anchor_component_bytes(gen_core::MemoryAssetFacts {
        base_bytes: facts.base_bytes + facts.overlay_bytes,
        decoder_bytes: facts.decoder_bytes + facts.overlay_bytes,
        overlay_bytes: 0,
        ..facts
    });
    assert!(
        anchor
            .derive_mlx_image_phase_peaks(request, whole_vae)
            .is_none(),
        "folding the reference encoder back into decoder_bytes must refuse — otherwise this \
             test would pass with or without the split it exists to pin"
    );
}
