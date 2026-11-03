use super::*;
use mlx_gen::gen_core::MemoryStrategySupport;

/// An otherwise-valid t2va plan row: the shipped 16:9 canvas at EXACTLY the pixel budget, the
/// shortest legal clip, and the one legal cadence. `modelId` stays `minimax_h3` even when
/// `provider` is foreign, so the provider guard is the check that fires rather than a
/// missing-field complaint standing in for it.
fn minimal_request(provider: &str) -> Value {
    json!({
        "hardware": {
            "memoryBytes": 137_438_953_472_u64,
            "wiredLimitBytes": 96_000_000_000_u64,
        },
        "planned": {
            "target": {
                "provider": provider,
                "modelId": MINIMAX_PROVIDER,
                "tier": "q4",
                "mode": "text_to_video",
                "overlay": "none",
                "geometry": { "width": 1344, "height": 768, "batch": 1, "frames": 124 }
            },
            "backend": "mlx",
            "loadShape": protocol::LOAD_SHAPE_EAGER,
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": minimax_calibration_fingerprint("q4")
                .expect("q4 is a shipped MiniMax-H3 tier"),
            "fixture": "minimax-h3-mlx-q4-1344x768-f124-fps24-seed17137"
        }
    })
}

fn weights_free_spec(quant: Option<Quant>, load_shape: LoadShape) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(load_shape);
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    spec
}

/// The provider-owned weights-free contract fixture — the same one catalog conformance resolves
/// when a snapshot is unavailable, so these pins touch no filesystem and no GPU.
fn fixture_contract(
    quant: Option<Quant>,
    load_shape: LoadShape,
) -> mlx_gen::gen_core::MemoryProviderContract {
    let registry = mlx_gen_minimax_h3::provider_registry().unwrap();
    let fixture = registry
        .memory_contract_fixture_registrations()
        .find(|fixture| fixture.provider_id == MINIMAX_PROVIDER)
        .expect("the provider-owned MiniMax-H3 contract fixture");
    (fixture.contract)(&weights_free_spec(quant, load_shape)).unwrap()
}

/// The capture harness must probe under the SAME evidence key the shipped runtime queries.
///
/// The MiniMax twin of `the_ltx_runtime_context_binds_the_exact_video_route_and_selection`,
/// which pins this correspondence for the family that already had it right. sc-18663: this arm
/// probed admission under `text_to_image` while the plan it validates, the record it emits and
/// `video_admission` all say `text_to_video` — see [`minimax_context`] for why that split is
/// currently inert and what makes it stop being inert.
///
/// NEITHER SIDE IS RESTATED AS A LITERAL. A test spelling `"text_to_video"` on both sides still
/// passes when both sides move together, which is the drift this exists to catch.
#[test]
fn the_minimax_capture_context_binds_the_mode_key_the_runtime_video_route_sends() {
    // The runtime side, derived: `video_jobs::wan` resolves every video job's admission mode
    // with `payload_video_mode`, and `video_admission` types that string with
    // `memory_mode_from_mode_key`, whose contract is `as_key(from(key)) == key`. So the string
    // this returns IS the `mode_key` the runtime's decode-policy query carries.
    let payload = json!({
        "model": MINIMAX_PROVIDER,
        "mode": sceneworks_core::contracts::ContractMode::TextToVideo.as_str(),
    });
    let runtime_mode_key = sceneworks_core::video_request::payload_video_mode(
        payload.as_object().expect("the probe payload is an object"),
    );
    assert!(
        sceneworks_core::video_request::is_minimax_h3_model(MINIMAX_PROVIDER),
        "the payload above must be a MiniMax-H3 video job, or it derives another family's key"
    );

    // And that really is the only route this arm records: the plan-target gate admits the
    // derived spelling and refuses the image spelling this context used to carry.
    let plan_under = |mode: &str| {
        let mut request = minimal_request(MINIMAX_PROVIDER);
        request["planned"]["target"]["mode"] = json!(mode);
        request
    };
    assert!(validate_minimax_target(&plan_under(&runtime_mode_key)).is_ok());
    assert!(validate_minimax_target(&plan_under(MemoryMode::TextToImage.as_key())).is_err());

    // The capture side, from the builder every MiniMax admission probe runs through.
    let contract = fixture_contract(Some(Quant::Q4), LoadShape::EagerMaterialization);
    let calibration = contract.calibration.as_ref().unwrap();
    let context = minimax_context(
        MINIMAX_BASE_MEMBER,
        planned_selection(&minimal_request(MINIMAX_PROVIDER)).unwrap(),
        calibration,
        &calibration.fingerprint,
        validate_minimax_geometry(1344, 768, 124).unwrap(),
        1 << 40,
        1,
    );
    assert_eq!(context.mode.as_key(), runtime_mode_key);
}

/// The per-arm twin of the sc-18104 provider guard: dispatch routes by name today, but this arm
/// hardcodes the MiniMax-H3 contract, so a misrouted target must be refused BY NAME inside the
/// arm — and before any environment variable is read, so the refusal cannot read like a
/// provisioning problem.
#[test]
fn the_arm_refuses_a_foreign_provider_by_name_before_any_environment_work() {
    for provider in ["ltx_2_3", "flux2_dev", "qwen_image", "minimax_h3_ref"] {
        let error = validate_minimax_target(&minimal_request(provider))
            .expect_err("a foreign provider must not reach the MiniMax-H3 contract");
        assert_eq!(
            error,
            format!("{MINIMAX_LABEL} does not implement provider {provider:?}")
        );
        assert!(
            !error.contains("SCENEWORKS_")
                && !error.contains("fingerprint")
                && !error.contains("contract"),
            "the refusal came from the environment or the contract, so validation let it \
                 through: {error}"
        );
    }
    assert!(validate_minimax_target(&minimal_request(MINIMAX_PROVIDER)).is_ok());
}

/// The dispatcher must ROUTE `minimax_h3` into this arm rather than refuse it, and must not
/// misroute it into a sibling. Proved by the two refusals it does NOT produce, which keeps the
/// assertion deterministic whether or not the capture environment happens to be exported.
#[test]
fn dispatch_routes_the_minimax_provider_into_its_own_arm() {
    let error = run(&minimal_request(MINIMAX_PROVIDER))
        .expect_err("no MiniMax-H3 weights are staged during a unit test");
    assert_ne!(
        error,
        format!("MLX five-rung calibration does not implement provider {MINIMAX_PROVIDER:?}"),
        "the dispatcher still refuses the provider this arm implements"
    );
    for foreign in ["LTX", "FLUX.2", "Qwen", "SDXL", "Krea", "Z-Image"] {
        assert!(
            !error.contains(foreign),
            "the request was misrouted into the {foreign} arm: {error}"
        );
    }
}

/// The envelope's engine constants, pinned so a pin bump that moves one reds HERE rather than
/// silently widening or narrowing what this arm will capture. The lattice is CHECKED as
/// `17n + 5` rather than only tabulated, so a table that drifted into a different progression
/// could not pass by matching itself.
#[test]
fn minimax_envelope_is_the_pinned_engines_own() {
    assert_eq!(mlx_gen_minimax_h3::SPATIAL_STRIDE, 32);
    assert_eq!(mlx_gen_minimax_h3::CANVAS_MAX_PIXELS, 1_032_192);
    assert_eq!(mlx_gen_minimax_h3::MINIMAX_H3_FPS, 24.0);
    assert_eq!(
        mlx_gen_minimax_h3::LEGAL_FRAME_COUNTS,
        [124, 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345]
    );
    for (index, frames) in mlx_gen_minimax_h3::LEGAL_FRAME_COUNTS.iter().enumerate() {
        assert_eq!(
            *frames,
            17 * index as i32 + 124,
            "the legal frame counts are 17n + 5, not a tabulated list"
        );
    }
    // The budget is a PRODUCT, and the published resolution list proves it cannot be a per-edge
    // cap: two shipped canvases whose long edges differ by 192px have the identical area.
    assert_eq!(1536 * 672, mlx_gen_minimax_h3::CANVAS_MAX_PIXELS);
    assert_eq!(1344 * 768, mlx_gen_minimax_h3::CANVAS_MAX_PIXELS);
}

#[test]
fn the_geometry_envelope_refuses_off_lattice_off_stride_and_over_budget_canvases() {
    let admitted = validate_minimax_geometry(1344, 768, 124).unwrap();
    assert_eq!(admitted.frames, 124);
    // 124 = 17·7 + 5 ⇒ 5·7 + 2 video latents, and round(124/24·40) audio tokens.
    assert_eq!(admitted.video_latent_frames, 37);
    assert_eq!(admitted.audio_latent_frames, 207);

    // Off-lattice, including a still (`T = 1` does not render at all) and LTX's own `1 + 8k`
    // lattice, which shares no member with this one inside the legal range.
    for frames in [0_u32, 1, 97, 121, 123, 125, 305, 346, u32::MAX] {
        let error = validate_minimax_geometry(1344, 768, frames)
            .expect_err("an off-lattice frame count must be refused");
        assert!(error.contains("17n+5 lattice"), "{frames}: {error}");
    }

    // Off-stride, on both axes, checked before the area so the message names the real defect.
    for (width, height) in [(1352_u32, 768_u32), (1344, 776), (1330, 768)] {
        let error = validate_minimax_geometry(width, height, 124)
            .expect_err("an off-stride canvas must be refused");
        assert!(error.contains("32px stride"), "{width}x{height}: {error}");
    }

    // Over budget as a PRODUCT. 1536x704 is on-stride and its long edge is one the model ships,
    // so only the area check can refuse it.
    assert!(validate_minimax_geometry(1536, 672, 124).is_ok());
    assert!(validate_minimax_geometry(576, 320, 345).is_ok());
    let error = validate_minimax_geometry(1536, 704, 124)
        .expect_err("an over-budget canvas must be refused");
    assert!(error.contains("canvas budget"), "{error}");
}

#[test]
fn the_geometry_envelope_pins_batch_to_one_clip_per_request() {
    let mut request = minimal_request(MINIMAX_PROVIDER);
    for batch in [0_u64, 2, 4] {
        request["planned"]["target"]["geometry"]["batch"] = json!(batch);
        let error = minimax_target_geometry(&request)
            .expect_err("this engine renders one clip per request");
        assert!(error.contains("geometry.batch == 1"), "{batch}: {error}");
    }
    // A missing or non-integer axis fails closed rather than defaulting.
    request["planned"]["target"]["geometry"]["batch"] = json!("1");
    assert!(minimax_target_geometry(&request)
        .unwrap_err()
        .contains("planned.target.geometry.batch must fit u32"));
}

/// This arm is the SECOND allowed to accept a multi-frame geometry, and it buys that with its
/// own envelope rather than by dropping the axis: the shared still guard would refuse the exact
/// geometry this arm admits.
#[test]
fn the_minimax_arm_accepts_a_multi_frame_geometry_the_image_guard_refuses() {
    let request = minimal_request(MINIMAX_PROVIDER);
    assert!(validate_minimax_target(&request).is_ok());
    assert_eq!(
        protocol::validate_still_geometry(&request, "MLX Qwen base calibration").unwrap_err(),
        "MLX Qwen base calibration requires geometry.frames == 1, got 124"
    );
}

/// A base (`minimax_h3`) plan row must not carry a reference surface: the reference-carrying
/// entry is `minimax_h3_ref`, which denoises on the `transformer_ref` partition, and a record
/// measured on one partition may never be filed against the other. The refusal names the field,
/// the entry and the partition fact, so a reader of the failure knows which row to fix.
#[test]
fn a_reference_carrying_or_wrong_mode_target_is_refused_because_ref2va_is_another_checkpoint() {
    for (field, value) in [
        ("referenceCount", json!(1)),
        ("reference_count", json!(2)),
        ("hasReference", json!(true)),
        ("has_reference", json!(true)),
    ] {
        let mut request = minimal_request(MINIMAX_PROVIDER);
        request["planned"]["target"][field] = value;
        let error = validate_minimax_target(&request)
            .expect_err("a reference surface must be refused by this arm");
        assert!(
            error.contains(field)
                && error.contains(MINIMAX_BASE_MEMBER.model_id)
                && error.contains("different DiT partitions"),
            "{error}"
        );
    }
    for mode in [
        "text_to_image",
        "image_to_video",
        "first_last_frame",
        "ref2va",
    ] {
        let mut request = minimal_request(MINIMAX_PROVIDER);
        request["planned"]["target"]["mode"] = json!(mode);
        let error = validate_minimax_target(&request).expect_err("only t2va is capturable here");
        assert!(
            error.contains("text_to_video") && error.contains(mode),
            "{error}"
        );
    }
}

#[test]
fn a_material_overlay_target_is_refused_rather_than_recorded_as_no_overlay() {
    for overlay in ["lora", "control", "identity", "control:1"] {
        let mut request = minimal_request(MINIMAX_PROVIDER);
        request["planned"]["target"]["overlay"] = json!(overlay);
        let error = run_minimax_h3(&request).expect_err("a material overlay must be refused");
        assert!(error.contains(MINIMAX_PLAIN_EXECUTION_PATH), "{error}");
        assert!(error.contains("refusing"), "{error}");
    }
}

#[test]
fn the_fixture_binds_the_tier_geometry_cadence_and_seed() {
    let request = minimal_request(MINIMAX_PROVIDER);
    let (member, geometry) = validate_minimax_target(&request).unwrap();
    assert_eq!(member, MINIMAX_BASE_MEMBER);
    assert_eq!(
        planned_minimax_capture(&request, member, "q4", geometry).unwrap(),
        (24, MINIMAX_SEED)
    );
    // The tier and every geometry axis are part of the binding.
    for tier in ["q8", "bf16"] {
        assert!(planned_minimax_capture(&request, member, tier, geometry)
            .unwrap_err()
            .contains("must start with"));
    }
    let mut off = request.clone();
    off["planned"]["fixture"] = json!("minimax-h3-mlx-q4-1344x768-f141-fps24-seed17137");
    assert!(planned_minimax_capture(&off, member, "q4", geometry)
        .unwrap_err()
        .contains("must start with"));
    // The cadence axis the geometry envelope cannot carry has exactly one legal value here.
    off["planned"]["fixture"] = json!("minimax-h3-mlx-q4-1344x768-f124-fps30-seed17137");
    assert!(planned_minimax_capture(&off, member, "q4", geometry)
        .unwrap_err()
        .contains("24 fps only"));
    off["planned"]["fixture"] = json!("minimax-h3-mlx-q4-1344x768-f124-fps24-seed1234");
    assert!(planned_minimax_capture(&off, member, "q4", geometry)
        .unwrap_err()
        .contains("does not match the MiniMax-H3 calibration seed"));
}

/// Fingerprint check 1 of 3 — the plan against this arm — must fire BEFORE the environment
/// contract, so a re-fingerprinted provider cannot be diagnosed as a provisioning failure.
/// Checks 2 and 3 (provider-vs-arm and plan-vs-provider) need the loaded registry contract;
/// `the_pinned_minimax_contract_declares_rung_support_by_load_shape_not_by_tier` pins the
/// constant those two compare against, weights-free.
#[test]
fn a_plan_naming_another_fingerprint_is_refused_before_any_environment_work() {
    let mut request = minimal_request(MINIMAX_PROVIDER);
    request["planned"]["calibrationFingerprint"] = json!("minimax-h3-mlx-deferred-v1");
    let error = run_minimax_h3(&request).expect_err("a foreign fingerprint must be refused");
    assert!(
        error.starts_with("plan/adapter calibration mismatch"),
        "{error}"
    );
    assert!(
        !error.contains("SCENEWORKS_"),
        "the refusal must precede the env contract: {error}"
    );
}

/// The arm's load-bearing premises, pinned against the PINNED provider crate, weights-free.
///
/// The finding the campaign has to plan around is that rung support here is a function of the
/// resolved LOAD SHAPE, not of the tier: `bounded_attention` is `StructurallyNotApplicable` on
/// every shape and tier, and `bounded_transformer_residency` is `Implemented` ONLY under
/// `deferred_materialization` and `Missing` under eager. Asserted across all three tiers so a
/// tier-shaped assumption cannot survive here either.
#[test]
fn the_pinned_minimax_contract_declares_rung_support_by_load_shape_not_by_tier() {
    let production: Vec<String> = MINIMAX_TIERS
        .iter()
        .map(|tier| minimax_calibration_fingerprint(tier).unwrap())
        .collect();
    for quant in [Some(Quant::Q4), Some(Quant::Q8), None] {
        for (load_shape, rung4_implemented) in [
            (LoadShape::EagerMaterialization, false),
            (LoadShape::DeferredMaterialization, true),
        ] {
            let contract = fixture_contract(quant, load_shape);
            let where_we_are = format!("{quant:?} at {load_shape:?}");
            assert_eq!(contract.provider_id, MINIMAX_PROVIDER);
            for strategy in [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
            ] {
                assert!(
                    matches!(
                        contract.capability(strategy).unwrap().support,
                        MemoryStrategySupport::Implemented
                    ),
                    "{strategy:?} is no longer Implemented ({where_we_are})"
                );
            }
            assert!(
                matches!(
                    contract
                        .capability(MemoryStrategy::BoundedAttention)
                        .unwrap()
                        .support,
                    MemoryStrategySupport::StructurallyNotApplicable { .. }
                ),
                "bounded_attention changed disposition ({where_we_are})"
            );
            let rung4 = contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap();
            if rung4_implemented {
                assert!(
                    matches!(rung4.support, MemoryStrategySupport::Implemented),
                    "rung 4 must be Implemented on a deferred load ({where_we_are})"
                );
                assert_eq!(rung4.parameters.transformer_window_sizes, vec![1]);
                assert_eq!(
                    rung4.parameters.transformer_window_components,
                    vec![TransformerComponent::Both],
                    "rung 4 streams BOTH transformers here, not the DiT alone"
                );
            } else {
                assert!(
                    matches!(rung4.support, MemoryStrategySupport::Missing),
                    "rung 4 must be Missing on a resident load ({where_we_are})"
                );
            }
            // Rung 2's domain is a SINGLETON — the tile geometry is an output-correctness input
            // copied from the published VAE, not a memory lever a sweep may vary.
            let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
            assert_eq!(decode.parameters.decode_tile_edges, vec![256]);
            assert_eq!(decode.parameters.decode_overlaps, vec![64]);
            // The weights-free fixture publishes the engine's DECLARATION key, never a
            // production identity (sc-22737): a contract that read no weights must not be able
            // to satisfy a plan row naming a measured cell. Its identity still carries the
            // RESOLVED shape, which is what the record's `loadShape` is taken from.
            let calibration = contract.calibration.as_ref().unwrap();
            assert_eq!(
                calibration.fingerprint,
                mlx_gen_minimax_h3::memory_strategy::STATIC_CALIBRATION_FINGERPRINT,
                "{where_we_are}"
            );
            assert!(
                !production.contains(&calibration.fingerprint),
                "{where_we_are}: the weights-free fixture published a production identity"
            );
            assert_eq!(calibration.load_shape, load_shape, "{where_we_are}");
        }
    }
}

/// The plan's `engagedRungs` must equal the provider's engaged composition, or
/// `attested_strategy` fails every capture at plan/measured comparison time — so the exact
/// compositions are pinned here rather than assumed from the ordinal rung order.
///
/// TWO of them are NOT what a reader would guess, and a plan written from the LTX shape would
/// be rejected by every MiniMax-H3 capture:
///
/// * `bounded_decode` engages `[resident, bounded_decode]` — **`staged_residency` is not in
///   it**, unlike LTX's `[resident, staged_residency, bounded_decode]`;
/// * `bounded_transformer_residency` engages `bounded_decode`, so a rung-4 row must ALSO carry
///   `decodeTileEdge` / `decodeOverlap` or `validate_selection` refuses it with
///   "decode_tile_edge is required by a strategy rung this selection engages".
#[test]
fn the_engaged_composition_is_what_the_plan_rows_must_declare() {
    use MemoryStrategy::{BoundedDecode, BoundedTransformerResidency, Resident, StagedResidency};
    for load_shape in [
        LoadShape::EagerMaterialization,
        LoadShape::DeferredMaterialization,
    ] {
        let contract = fixture_contract(Some(Quant::Q4), load_shape);
        assert_eq!(contract.engaged_composition(Resident), vec![Resident]);
        assert_eq!(
            contract.engaged_composition(StagedResidency),
            vec![Resident, StagedResidency]
        );
        assert_eq!(
            contract.engaged_composition(BoundedDecode),
            vec![Resident, BoundedDecode],
            "bounded_decode does NOT engage staged_residency here ({load_shape:?})"
        );
    }
    assert_eq!(
        fixture_contract(Some(Quant::Q4), LoadShape::DeferredMaterialization)
            .engaged_composition(BoundedTransformerResidency),
        vec![Resident, BoundedDecode, BoundedTransformerResidency],
        "rung 4 drags rung 2 in, so a rung-4 plan row must declare the decode parameters too"
    );
}

/// Rung 4 is reachable ONLY under the deferred shape, and it needs the `"both"` window spelling
/// the shared parser learned for this provider. Both halves are asserted, in both directions.
#[test]
fn rung_four_is_capturable_only_under_the_deferred_load_shape() {
    let mut request = minimal_request(MINIMAX_PROVIDER);
    request["planned"]["loadShape"] = json!(protocol::LOAD_SHAPE_DEFERRED);
    // Rung 4 ENGAGES rung 2 here, so the decode parameters are mandatory on a rung-4 row —
    // omitting them is refused with "decode_tile_edge is required by a strategy rung this
    // selection engages", which reads like a plan typo and is not one.
    request["planned"]["strategy"] = json!({
        "rung": "bounded_transformer_residency",
        "engagedRungs": ["resident", "bounded_decode", "bounded_transformer_residency"],
        "parameters": {
            "decodeTileEdge": 256,
            "decodeOverlap": 64,
            "transformerWindowSize": 1,
            "transformerWindowComponent": "both"
        }
    });
    let selection = planned_selection(&request).expect("the shared parser accepts \"both\"");
    assert_eq!(
        selection.parameters.transformer_window_component,
        Some(TransformerComponent::Both)
    );
    fixture_contract(Some(Quant::Q4), LoadShape::DeferredMaterialization)
        .validate_selection(&selection)
        .expect("a deferred load implements rung 4");
    assert!(
        fixture_contract(Some(Quant::Q4), LoadShape::EagerMaterialization)
            .validate_selection(&selection)
            .is_err(),
        "a resident load does NOT have rung 4's prerequisite shape"
    );

    // And the decode parameters really are the reason, not incidental: dropping them fails the
    // deferred contract too.
    request["planned"]["strategy"]["parameters"] =
        json!({ "transformerWindowSize": 1, "transformerWindowComponent": "both" });
    let bare = planned_selection(&request).unwrap();
    assert!(
        fixture_contract(Some(Quant::Q4), LoadShape::DeferredMaterialization)
            .validate_selection(&bare)
            .unwrap_err()
            .to_string()
            .contains("decode_tile_edge")
    );
}

/// A snapshot root whose DiT carries the packed marker the engine's `resolved_artifact_tier`
/// reads (`transformer/config.json`, the same file `mlx_gen::quant::packed_quant_bits_at`
/// parses on the load path). `None` writes a DENSE marker — a real `config.json` with no
/// `quantization` block — which is what makes a bf16 root distinguishable from an absent one.
fn minimax_marked_root(bits: Option<u32>) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "sc-22737-minimax-{}-{nonce}-{bits:?}",
        std::process::id()
    ));
    let dit = root.join(mlx_gen_minimax_h3::model::DIT_COMPONENT);
    std::fs::create_dir_all(&dit).unwrap();
    let body = match bits {
        Some(bits) => format!("{{\"quantization\":{{\"bits\":{bits},\"group_size\":64}}}}"),
        None => "{}".to_owned(),
    };
    std::fs::write(dit.join("config.json"), body).unwrap();
    root
}

/// The pinned engine mints exactly this arm's identity family, and withholds it where the
/// artifact cannot prove a tier (sc-22737, epic 22723 E4).
///
/// * A weights-free spec publishes NO calibration identity: the contract resolves, but its
///   `calibration` is `None` rather than a fabricated production key, so a runner that loaded
///   nothing can never satisfy a plan row naming a measured cell.
/// * A root whose DiT carries a real tier marker publishes, per tier, byte-for-byte the key
///   [`minimax_calibration_fingerprint`] expects — fingerprint check 2 of 3, proven for all
///   three tiers on a CPU-only host — and a `spec.quantize` that disagrees with the marker is
///   withheld too, never re-labelled.
#[test]
fn the_pinned_minimax_engine_mints_this_arms_identity_family_per_artifact_tier() {
    let registry = mlx_gen_minimax_h3::provider_registry().unwrap();
    let withheld = registry
        .memory_strategy_contract(
            MINIMAX_PROVIDER,
            &weights_free_spec(Some(Quant::Q4), LoadShape::EagerMaterialization),
        )
        .expect("the registry resolves a MiniMax-H3 memory-strategy contract")
        .expect("the MiniMax-H3 memory-strategy registration is present at the pin");
    assert_eq!(withheld.provider_id, MINIMAX_PROVIDER);
    assert!(
        withheld.calibration.is_none(),
        "a spec whose weights nobody can see must publish no identity, got {:?}",
        withheld.calibration
    );
    for (tier, bits, quant) in [
        ("q4", Some(4), Some(Quant::Q4)),
        ("q8", Some(8), Some(Quant::Q8)),
        ("bf16", None, None),
    ] {
        let root = minimax_marked_root(bits);
        let spec = |quant: Option<Quant>| {
            let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
                .with_offload_policy(OffloadPolicy::Resident)
                .with_load_shape(LoadShape::EagerMaterialization);
            if let Some(quant) = quant {
                spec = spec.with_quant(quant);
            }
            spec
        };
        let contract = registry
            .memory_strategy_contract(MINIMAX_PROVIDER, &spec(quant))
            .unwrap()
            .unwrap();
        let calibration = contract
            .calibration
            .as_ref()
            .unwrap_or_else(|| panic!("{tier}: a marked artifact publishes its identity"));
        assert_eq!(
            calibration.fingerprint,
            minimax_calibration_fingerprint(tier).unwrap(),
            "{tier}: the pinned engine mints a key this arm would refuse"
        );
        assert_eq!(
            calibration.load_shape,
            LoadShape::EagerMaterialization,
            "{tier}"
        );
        // The tier is the ARTIFACT's: asking a marked root for ANOTHER packed width — or a
        // dense root for a packed one — is a load-time requantization whose peak no anchor
        // measured, and it publishes nothing. (`quantize: None` on a packed root is not a
        // disagreement: it loads whatever the sidecars prove, which is the production shape.)
        let disagreeing = match quant {
            Some(Quant::Q4) => Some(Quant::Q8),
            _ => Some(Quant::Q4),
        };
        let mismatch = registry
            .memory_strategy_contract(MINIMAX_PROVIDER, &spec(disagreeing))
            .unwrap()
            .unwrap();
        assert!(
            mismatch.calibration.is_none(),
            "{tier}: a spec.quantize that disagrees with the marker must be withheld, got {:?}",
            mismatch.calibration
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}

/// **The provider contract, pinned in code.** At the permanent inference pin the MiniMax-H3
/// provider publishes `memory_strategy_contract` and its admission check is non-vacuous in both
/// directions — INCLUDING the calibration handshake, which is only asked when the contract
/// carries an identity (sc-22737): the check is driven from a marked q4 root, the same contract
/// shape `run_minimax_h3` reads, so the handshake this arm relies on is proven rather than
/// skipped over a weights-free contract that publishes none.
#[test]
fn the_minimax_h3_memory_strategy_contract_enforces_non_vacuous_admission() {
    let registry = mlx_gen_minimax_h3::provider_registry().unwrap();
    let root = minimax_marked_root(Some(4));
    let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(LoadShape::EagerMaterialization)
        .with_quant(Quant::Q4);
    let contract = registry
        .memory_strategy_contract(MINIMAX_PROVIDER, &spec)
        .expect("the registry resolves a MiniMax-H3 memory-strategy contract")
        .expect("the MiniMax-H3 memory-strategy registration is present at the pin");
    assert_eq!(contract.provider_id, MINIMAX_PROVIDER);
    let calibration = contract
        .calibration
        .as_ref()
        .expect("a marked q4 root publishes the q4 identity");
    assert_eq!(
        calibration.fingerprint,
        minimax_calibration_fingerprint("q4").unwrap()
    );
    let selection = planned_selection(&minimal_request(MINIMAX_PROVIDER)).unwrap();
    let geometry = validate_minimax_geometry(1344, 768, 124).unwrap();
    let context = |fingerprint: &str, total: u64| {
        minimax_context(
            MINIMAX_BASE_MEMBER,
            selection,
            calibration,
            fingerprint,
            geometry,
            total,
            1,
        )
    };
    assert!(matches!(
        mlx_gen_minimax_h3::memory_strategy::safety_check(
            &spec,
            &contract,
            &context(&calibration.fingerprint, 1 << 40)
        ),
        MemorySafetyDecision::Accept
    ));
    assert!(matches!(
        mlx_gen_minimax_h3::memory_strategy::safety_check(
            &spec,
            &contract,
            &context(&calibration.fingerprint, 0)
        ),
        MemorySafetyDecision::Reject { .. }
    ));
    // The handshake is REAL: another tier's identity — a plausible member of this very family,
    // not a garbage string — is refused against the loaded q4 contract.
    let stale = minimax_calibration_fingerprint("q8").unwrap();
    match mlx_gen_minimax_h3::memory_strategy::safety_check(
        &spec,
        &contract,
        &context(&stale, 1 << 40),
    ) {
        MemorySafetyDecision::Reject { reason } => {
            assert!(
                reason.contains("calibration handshake mismatch"),
                "{reason}"
            );
        }
        other => panic!("a foreign identity must be refused, got {other:?}"),
    }
    std::fs::remove_dir_all(&root).unwrap();
}

/// The three `mlx:minimax_h3` identities are ONE family minted off the engine's retained key
/// (`MEMORY_CALIBRATION_FINGERPRINT`, kept by the engine's `CALIBRATED_TIER`), and each other
/// shipped tier is that key with its own tier token — distinct per tier, so a q4 record can
/// never be matched against a bf16 load. A tier the family does not ship is refused by name.
#[test]
fn the_minimax_identity_family_reproduces_the_engines_retained_key_with_a_tier_token() {
    use mlx_gen_minimax_h3::memory_strategy as engine;
    assert_eq!(MINIMAX_CALIBRATED_TIER, engine::CALIBRATED_TIER);
    assert!(MINIMAX_TIERS.contains(&MINIMAX_CALIBRATED_TIER));
    let retained = engine::MEMORY_CALIBRATION_FINGERPRINT;
    assert_eq!(
        minimax_calibration_fingerprint(MINIMAX_CALIBRATED_TIER).unwrap(),
        retained,
        "the calibrated cell must reproduce the engine's retained key byte-for-byte"
    );
    let (stem, remainder) = retained
        .split_once("-mlx-")
        .unwrap_or_else(|| panic!("{retained} carries no lane token"));
    assert_eq!(
        stem, "minimax-h3",
        "the retained key carries no tier token: {retained}"
    );
    let mut identities = std::collections::BTreeSet::new();
    for tier in MINIMAX_TIERS {
        let minted = minimax_calibration_fingerprint(tier).unwrap();
        if tier != MINIMAX_CALIBRATED_TIER {
            assert_eq!(
                minted,
                format!("{stem}-{tier}-mlx-{remainder}"),
                "the {tier} cell must be the engine's key carrying its own tier token"
            );
        }
        assert!(
            identities.insert(minted),
            "{tier} reuses another tier's identity"
        );
    }
    assert_eq!(identities.len(), MINIMAX_TIERS.len());
    assert_ne!(
        retained,
        engine::STATIC_CALIBRATION_FINGERPRINT,
        "the weights-free declaration key is never a production identity"
    );
    let error = minimax_calibration_fingerprint("fp8").unwrap_err();
    assert!(
        error.contains("\"fp8\"") && error.contains("bf16"),
        "{error}"
    );
}

/// Both catalog entries share the loaded provider's per-tier identity (see
/// [`minimax_calibration_fingerprint`]), so every planned MLX row of either member must name
/// exactly the key this arm mints for the row's tier — the binding that reds when the plan
/// carries the bare key on a tier the engine mints a tokened one for (sc-22737 review).
#[test]
fn every_planned_minimax_mlx_row_names_the_identity_this_arm_mints_for_its_tier() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan is valid JSON");
    let mut planned = std::collections::BTreeSet::new();
    for (key, row) in plan["anchors"].as_object().expect("anchors is an object") {
        let parts: Vec<&str> = key.split(':').collect();
        if parts.len() != 3 || parts[2] != "mlx" {
            continue;
        }
        let Some(member) = MINIMAX_MEMBERS
            .iter()
            .find(|member| member.model_id == parts[0])
        else {
            continue;
        };
        let tier = parts[1];
        assert_eq!(row["provider"].as_str(), Some(MINIMAX_PROVIDER), "{key}");
        assert_eq!(row["mode"].as_str(), Some(member.mode), "{key}");
        assert_eq!(
            row["calibrationFingerprint"]
                .as_str()
                .expect("fingerprint is a string"),
            minimax_calibration_fingerprint(tier).unwrap(),
            "{key} names an identity the pinned provider does not mint for tier {tier}"
        );
        assert!(
            planned.insert((member.model_id, tier.to_owned())),
            "{key} is planned twice"
        );
    }
    let expected: std::collections::BTreeSet<(&str, String)> = MINIMAX_MEMBERS
        .iter()
        .flat_map(|member| MINIMAX_TIERS.map(|tier| (member.model_id, tier.to_owned())))
        .collect();
    assert_eq!(
        planned, expected,
        "every member has one planned MLX row per shipped tier"
    );
}

/// A plan row naming ANOTHER TIER's identity — a plausible member of this family — is refused
/// before any environment work, naming both the plan's key and the tier's, so a copied row
/// cannot re-label one tier's peaks as another's.
#[test]
fn a_plan_naming_another_tiers_identity_is_refused_before_any_environment_work() {
    let mut request = minimal_request(MINIMAX_PROVIDER);
    request["planned"]["calibrationFingerprint"] =
        json!(minimax_calibration_fingerprint("bf16").unwrap());
    let error = run_minimax_h3(&request).expect_err("a q4 row may not carry the bf16 key");
    assert!(
        error.starts_with("plan/adapter calibration mismatch"),
        "{error}"
    );
    assert!(error.contains("for tier q4"), "{error}");
    assert!(
        error.contains(&minimax_calibration_fingerprint("q4").unwrap()),
        "{error}"
    );
    assert!(!error.contains("SCENEWORKS_"), "{error}");
}

/// The runtime-complete sweep contract the harness checks: `rangeVerified`, exactly one passed
/// case, and parameters equal to the plan's own. Rung 4's string component belongs in the case
/// and never as a swept axis.
#[test]
fn the_runtime_complete_sweep_carries_one_passed_case_matching_the_planned_parameters() {
    let mut request = minimal_request(MINIMAX_PROVIDER);
    let sweep = minimax_complete_sweep(&request).unwrap();
    assert_eq!(sweep["rangeVerified"], json!(true));
    assert_eq!(sweep["axes"], json!([]));
    assert_eq!(sweep["cases"].as_array().unwrap().len(), 1);
    assert_eq!(sweep["cases"][0]["result"], json!("passed"));
    assert_eq!(
        sweep["cases"][0]["parameters"],
        request["planned"]["strategy"]["parameters"]
    );

    request["planned"]["strategy"]["parameters"] =
        json!({ "transformerWindowSize": 1, "transformerWindowComponent": "both" });
    let sweep = minimax_complete_sweep(&request).unwrap();
    assert_eq!(
        sweep["axes"],
        json!([{ "parameter": "transformerWindowSize", "testedValues": [1] }]),
        "a component name is not a swept numeric range"
    );
    assert_eq!(
        sweep["cases"][0]["parameters"],
        request["planned"]["strategy"]["parameters"]
    );
}

/// A tier is a WHOLE-PIPELINE contract here: the conditioning stage's tier is derived from the
/// DiT's rather than being a free axis the record could not describe.
#[test]
fn the_text_encoder_tier_is_derived_from_the_dit_tier() {
    assert_eq!(
        minimax_text_encoder_source("q4"),
        MINIMAX_TIERED_TEXT_ENCODER
    );
    assert_eq!(
        minimax_text_encoder_source("q8"),
        MINIMAX_TIERED_TEXT_ENCODER
    );
    assert_eq!(
        minimax_text_encoder_source("bf16"),
        MINIMAX_UPSTREAM_TEXT_ENCODER,
        "the rehost publishes no bf16 text encoder; the manifest's bf16 co-requisite row is \
             the upstream repository"
    );
}

/// The output unwrapper refuses the two shapes an image or audio provider would return, under
/// this arm's OWN name — so a misrouted still could never be recorded as a joint A/V clip.
#[test]
fn the_minimax_output_unwrapper_refuses_image_and_audio_shaped_output() {
    let frame = Image {
        width: 2,
        height: 1,
        pixels: vec![0, 0, 0, 255, 255, 255],
    };
    assert_eq!(
        diagnostic_video_frames(
            GenerationOutput::Images(vec![frame.clone()]),
            MINIMAX_VIDEO_LABEL
        )
        .unwrap_err(),
        "MLX MiniMax-H3 render returned images, not a video clip"
    );
    assert_eq!(
        diagnostic_video_frames(
            GenerationOutput::Video {
                frames: Vec::new(),
                fps: 24,
                audio: None,
            },
            MINIMAX_VIDEO_LABEL
        )
        .unwrap_err(),
        "MLX MiniMax-H3 render returned no frames"
    );
    // And the LTX arm's own refusals are unchanged by the label becoming a parameter.
    assert_eq!(
        diagnostic_video_frames(GenerationOutput::Images(vec![frame]), LTX_VIDEO_LABEL)
            .unwrap_err(),
        "MLX LTX-2.3 render returned images, not a video clip"
    );
}

/// The determinism envelope must be falsifiable in both directions: an identical clip passes,
/// and the broad bias the arm's negative mutation applies breaches all three metrics.
#[test]
fn the_determinism_envelope_admits_an_identical_clip_and_breaches_on_the_mutation() {
    let clip = vec![
        Image {
            width: 2,
            height: 1,
            pixels: vec![10, 20, 30, 200, 210, 220],
        },
        Image {
            width: 2,
            height: 1,
            pixels: vec![40, 50, 60, 70, 80, 90],
        },
    ];
    let (maximum, mean, rms) = video_max_mean_rms_abs(&clip, &clip).unwrap();
    assert_eq!((maximum, mean, rms), (0.0, 0.0, 0.0));
    assert!(minimax_quality_passes(maximum, mean, rms));

    let mutated = clip.iter().map(qwen_negative_mutation).collect::<Vec<_>>();
    let (maximum, mean, rms) = video_max_mean_rms_abs(&mutated, &clip).unwrap();
    assert!(maximum > MINIMAX_MAX_THRESHOLD);
    assert!(mean > MINIMAX_MEAN_THRESHOLD);
    assert!(rms > MINIMAX_RMS_THRESHOLD);
    assert!(!minimax_quality_passes(maximum, mean, rms));

    // The thresholds are this provider's own literals, pinned in the 0-255 unit an operator
    // reads them in. They are spelled as MiniMax-named literals rather than aliased onto
    // another lane's constants, so a receipt embedding them cannot be traced to a constant
    // asserting a different provider's provenance.
    let per_255 = |threshold: f64| (threshold * 255.0).round() as u64;
    assert_eq!(
        (
            per_255(MINIMAX_MAX_THRESHOLD),
            per_255(MINIMAX_MEAN_THRESHOLD),
            per_255(MINIMAX_RMS_THRESHOLD),
        ),
        (3, 1, 2)
    );
}
