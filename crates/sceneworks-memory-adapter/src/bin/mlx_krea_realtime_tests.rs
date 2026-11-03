use super::*;

/// The geometry the plan declares for all three `krea_realtime_14b:*:mlx` rows.
const PLANNED: (u32, u32, u32) = (832, 480, 45);

fn minimal_request(provider: &str, width: u32, height: u32, frames: u32, batch: u32) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "modelId": provider,
                "tier": "q4",
                "mode": "text_to_video",
                "overlay": "none",
                "geometry": {
                    "width": width, "height": height, "batch": batch, "frames": frames
                }
            },
            "loadShape": "eager_materialization",
            "calibrationFingerprint": "krea-realtime-14b-q4-mlx-resident-ladder-v1",
            "fixture": format!(
                "krea-realtime-14b-mlx-q4-{width}x{height}-f{frames}-fps24-seed22735"
            ),
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} }
        }
    })
}

fn planned_request() -> Value {
    minimal_request(KREA_REALTIME_PROVIDER, PLANNED.0, PLANNED.1, PLANNED.2, 1)
}

/// (e) The plan's own geometry is ACCEPTED, and every derived quantity the record reports is the
/// engine's own arithmetic rather than a restated literal.
#[test]
fn the_planned_video_geometry_is_accepted_and_derives_the_engines_own_latents() {
    let geometry =
        validate_krea_realtime_target(&planned_request()).expect("the planned row renders");
    assert_eq!((geometry.width, geometry.height), (832, 480));
    assert_eq!(geometry.frames, 45);
    // 45 output frames ⇒ (45 − 1)/4 + 1 = 12 latent frames ⇒ exactly four whole AR blocks.
    assert_eq!(geometry.latent_frames, 12);
    assert_eq!(geometry.autoregressive_blocks, 4);
}

/// The frame lattice this arm transcribes is SceneWorks-core's own Wan rule, driven over the
/// whole accepted and refused range through the real symbol (a dev-dependency, so the arm itself
/// cannot call it). A transcription drift reds here.
#[test]
fn the_realtime_frame_lattice_is_sceneworks_cores_own_wan_rule() {
    for frames in 1..=200_u32 {
        let accepted = validate_krea_realtime_geometry(PLANNED.0, PLANNED.1, frames).is_ok();
        let on_lattice = sceneworks_core::video_request::wan_frame_count(frames) == frames;
        assert_eq!(
            accepted, on_lattice,
            "frames={frames}: the arm and wan_frame_count must agree on the lattice"
        );
    }
}

/// The latent derivation matches the formula the engine documents on `KreaRealtimeJob` and
/// implements in the `pub(crate)` `latent_frame_count` — re-derived here because there is no
/// symbol to read.
#[test]
fn the_realtime_latent_derivation_matches_the_engines_own_arithmetic() {
    for frames in (5..=997_u32).step_by(4) {
        let Ok(geometry) = validate_krea_realtime_geometry(PLANNED.0, PLANNED.1, frames) else {
            continue;
        };
        assert_eq!(
            geometry.latent_frames,
            (frames - 1) / KREA_REALTIME_TEMPORAL_STRIDE + 1,
            "frames={frames}"
        );
        assert!(geometry.latent_frames <= KREA_REALTIME_MAX_LATENT_FRAMES);
    }
    // 1,028 output frames is exactly the 257-latent maximum; the next lattice step is over it.
    assert_eq!(
        validate_krea_realtime_geometry(PLANNED.0, PLANNED.1, 1_025)
            .unwrap()
            .latent_frames,
        257
    );
}

/// (e) Each geometry refusal fires with its OWN message, and the guard is not a blanket one.
#[test]
fn each_geometry_clause_refuses_with_its_own_message() {
    let capabilities = mlx_gen_krea_realtime::descriptor().capabilities;
    // Off the 4k+1 lattice.
    let error = validate_krea_realtime_geometry(PLANNED.0, PLANNED.1, 46).unwrap_err();
    assert!(error.contains("Wan lattice"), "{error}");
    // On the lattice, but past the engine's own latent ceiling.
    let error = validate_krea_realtime_geometry(PLANNED.0, PLANNED.1, 1_029).unwrap_err();
    assert!(error.contains("latent-frame maximum"), "{error}");
    // An edge off the engine's advertised grid.
    let error = validate_krea_realtime_geometry(840, PLANNED.1, PLANNED.2).unwrap_err();
    assert!(error.contains("grid"), "{error}");
    // An on-grid edge above the engine's advertised per-edge maximum.
    let over = capabilities.max_size + 16;
    let error = validate_krea_realtime_geometry(over, PLANNED.1, PLANNED.2).unwrap_err();
    assert!(
        error.contains("per-edge range") && error.contains("width"),
        "{error}"
    );
    // ...and below its advertised minimum, on the same clause but the other axis.
    let error = validate_krea_realtime_geometry(PLANNED.0, 0, PLANNED.2).unwrap_err();
    assert!(
        error.contains("per-edge range") && error.contains("height"),
        "{error}"
    );
    // batch != 1.
    let error = validate_krea_realtime_target(&minimal_request(
        KREA_REALTIME_PROVIDER,
        PLANNED.0,
        PLANNED.1,
        PLANNED.2,
        2,
    ))
    .unwrap_err();
    assert!(error.contains("geometry.batch"), "{error}");
}

/// (e) THE ENGINE HAS NO AREA CAP (`pinned_engine_geometry.rs` lists `krea_realtime_14b` as
/// `PinnedAreaCap::EngineHasNone`), so a large-area geometry inside the per-edge range must be
/// ACCEPTED. A refusal here would be an invented gate that made this harness reject geometries
/// the shipped engine renders.
#[test]
fn a_large_area_geometry_is_not_refused_because_the_engine_has_no_area_cap() {
    let max = mlx_gen_krea_realtime::descriptor().capabilities.max_size;
    assert!(max.is_multiple_of(16), "the advertised maximum is on-grid");
    let geometry = validate_krea_realtime_geometry(max, max, PLANNED.2)
        .expect("the maximum square canvas is inside the engine's own envelope");
    assert_eq!((geometry.width, geometry.height), (max, max));
    // Far above MiniMax-H3's canvas budget, which this arm deliberately does not apply.
    assert!(u64::from(max) * u64::from(max) > 1_000_000);
}

/// (c/f) A foreign provider, a foreign mode and a declared reference are each refused BY NAME,
/// before any environment variable is read or any path canonicalized.
#[test]
fn the_realtime_arm_refuses_a_foreign_target_before_environment_or_weight_work() {
    for provider in ["krea_2_raw", "minimax_h3", "ltx_video", "wan_2_1_t2v_14b"] {
        let request = minimal_request(provider, PLANNED.0, PLANNED.1, PLANNED.2, 1);
        let error = run_krea_realtime(&request)
            .expect_err("a foreign provider must not reach the Krea Realtime arm");
        assert_eq!(
            error,
            format!("{KREA_REALTIME_LABEL} does not implement provider {provider:?}")
        );
        for leaked in ["SCENEWORKS_", "canonicalize", "contract"] {
            assert!(!error.contains(leaked), "{leaked:?} leaked into {error}");
        }
    }

    for mode in [
        "image_to_video",
        "video_to_video",
        "text_to_image",
        "edit_image",
    ] {
        let mut request = planned_request();
        request["planned"]["target"]["mode"] = json!(mode);
        let error = run_krea_realtime(&request).expect_err("only t2v is capturable here");
        assert_eq!(
            error,
            format!(
                "{KREA_REALTIME_LABEL} requires reference-free text_to_video mode, got {mode:?}"
            )
        );
    }

    for (field, value) in [
        ("referenceCount", json!(1)),
        ("reference_count", json!(2)),
        ("hasReference", json!(true)),
        ("has_reference", json!(true)),
    ] {
        let mut request = planned_request();
        request["planned"]["target"][field] = value;
        let error = run_krea_realtime(&request)
            .expect_err("a reference routes the engine to i2v and is not this record");
        assert!(
            error.contains(field) && error.contains("i2v"),
            "{field}: {error}"
        );
    }
}

/// The load-shape guard: the plan must declare the eager shape the worker executes.
#[test]
fn the_realtime_arm_refuses_a_deferred_plan_row() {
    let mut request = planned_request();
    request["planned"]["loadShape"] = json!(protocol::LOAD_SHAPE_DEFERRED);
    let error = run_krea_realtime(&request).expect_err("the plan declares the wrong shape");
    assert!(
        error.contains("must use the production eager_materialization"),
        "{error}"
    );
}

/// (Part 3) The pre-load fingerprint check fires inside the arm, before any environment read.
#[test]
fn the_realtime_arm_refuses_an_unproducible_planned_identity_before_weight_work() {
    let mut request = planned_request();
    request["planned"]["calibrationFingerprint"] =
        json!("krea-realtime-14b-bf16-mlx-resident-ladder-v1");
    let error =
        run_krea_realtime(&request).expect_err("a bf16 identity must not satisfy a q4 plan row");
    assert!(
        error.contains("krea-realtime-14b-q4-mlx-resident-ladder-v1"),
        "{error}"
    );
    assert!(!error.contains("SCENEWORKS_"), "{error}");
}

/// The fixture binds the tier, the full rendered geometry, the single legal cadence and the
/// seed.
#[test]
fn the_realtime_fixture_binds_the_tier_geometry_cadence_and_seed() {
    let geometry = validate_krea_realtime_target(&planned_request()).unwrap();
    assert_eq!(
        planned_krea_realtime_capture(&planned_request(), "q4", geometry).unwrap(),
        (KREA_REALTIME_FPS, KREA_REALTIME_SEED)
    );
    for (fixture, expected) in [
        (
            "krea-realtime-14b-mlx-bf16-832x480-f45-fps24-seed22735",
            "must start with",
        ),
        (
            "krea-realtime-14b-mlx-q4-832x480-f45-fps30-seed22735",
            "generates at 24 fps only",
        ),
        (
            "krea-realtime-14b-mlx-q4-832x480-f45-fps24-seed17137",
            "does not match the Krea Realtime calibration seed",
        ),
    ] {
        let mut request = planned_request();
        request["planned"]["fixture"] = json!(fixture);
        let error = planned_krea_realtime_capture(&request, "q4", geometry)
            .expect_err("a mis-bound fixture must be refused");
        assert!(error.contains(expected), "{fixture}: {error}");
    }
}

/// The arm is reachable through `run`'s dispatch under exactly the plan's provider id.
#[test]
fn the_realtime_provider_id_is_the_engines_own() {
    assert_eq!(KREA_REALTIME_PROVIDER, mlx_gen_krea_realtime::MODEL_ID);
    assert_eq!(KREA_REALTIME_PROVIDER, "krea_realtime_14b");
}

/// sc-22738. THE decision this fix is about: for every planned `krea_realtime_14b:*:mlx` cell
/// the arm's answer to "does this capture ask Krea Realtime's admission gate anything?" must be
/// the answer PRODUCTION gives for the same request shape — and today, on every cell this arm
/// can plan, that answer is NO.
///
/// sc-22735 asked the gate on the anchor's own reference-free t2v surface, which the engine's
/// route gate refuses by name, and all three anchors died with "admission rejected a fitting
/// probe budget" after a full 14B weight load. The remedy is not a surface that makes the gate
/// answer — production never presents one for this request — it is to skip the probe exactly
/// where production skips the context.
#[test]
fn the_arm_asks_the_admission_gate_exactly_what_production_asks_it() {
    // The plan's own rows, read rather than restated: every one is reference-free t2v, so the
    // arm's decision on every schedulable cell is the one asserted below.
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut planned_rows = 0;
    for (key, row) in plan["anchors"].as_object().expect("anchors object") {
        if row["provider"].as_str() != Some(KREA_REALTIME_PROVIDER) || !key.ends_with(":mlx") {
            continue;
        }
        let mode = row["mode"].as_str().expect("a plan row declares a mode");
        let references = row["referenceCount"].as_u64().unwrap_or(0) as u32;
        assert!(
            !krea_realtime_probes_admission(mode, references),
            "{key}: a reference-free Krea Realtime capture must not probe admission"
        );
        planned_rows += 1;
    }
    assert_eq!(
        planned_rows, 3,
        "the plan must still carry the three MLX Krea Realtime rows this case decides for"
    );

    // The predicate over both of its axes, not just today's rows: the two routes the engine's
    // gate names probe, every other mode key does not, and a reference count above zero probes
    // on any mode — so a reference-carrying anchor can never inherit the skip silently.
    // (The route list itself is bound to the pinned engine's own `safety_check` by
    // `the Krea Realtime arm's admitted routes are the pinned engine's own` in
    // scripts/measure-memory-catalog.test.mjs, which reads the engine source.)
    for mode in [
        "image_to_video",
        "video_to_video",
        "text_to_video",
        "text_to_image",
        "edit",
    ] {
        for reference_count in [0, 1, 4] {
            let admitted_route = matches!(mode, "image_to_video" | "video_to_video");
            assert_eq!(
                krea_realtime_probes_admission(mode, reference_count),
                admitted_route || reference_count > 0,
                "{mode}/refs={reference_count}: the arm's mirror of the engine's route gate \
                     has drifted"
            );
        }
    }

    // The arm's own target validation refuses every surface the predicate answers `true` for,
    // so the refusal branch is unreachable from a plan row rather than merely unused: a
    // reference-carrying Krea Realtime capture is a different anchor, not a mode flag.
    for (field, value) in [
        ("mode", json!("image_to_video")),
        ("referenceCount", json!(1)),
    ] {
        let mut request = planned_request();
        request["planned"]["target"][field] = value;
        assert!(
            validate_krea_realtime_target(&request).is_err(),
            "{field}: a surface the gate would answer for must be refused by name"
        );
    }

    // …and the arm ACTS on that decision. The predicate is only the answer; a body that still
    // called the provider's admission gate, or a record that still claimed the three admission
    // scenarios passed, would be the sc-22735 defect wearing the sc-22738 predicate. The
    // measured render needs real weights, so the body is read as source.
    let source = include_str!("mlx.rs");
    let start = source
        .find("\nfn run_krea_realtime(")
        .expect("the arm still exists");
    let body = &source[start
        ..start
            + source[start..]
                .find("\n}\n")
                .expect("the arm's body closes")];
    assert!(
        !body.contains("safety_check"),
        "the Krea Realtime arm asks the provider's admission gate something again; production \
             asks it nothing for a reference-free t2v request"
    );
    assert!(
        body.contains("\"status\": \"gated\""),
        "a capture that ran no admission probe cannot file a runtime-activating record"
    );
    for scenario in ["exact_fit", "unknown_budget", "stale_evidence"] {
        assert!(
            body.contains(&format!(
                "{{ \"name\": \"{scenario}\", \"result\": \"not_run\", \"reason\": \
                     KREA_REALTIME_ADMISSION_BLOCKER }}"
            )),
            "the {scenario} scenario must be reported unexecuted, with the reason production \
                 gives for it"
        );
    }
    assert!(
        KREA_REALTIME_ADMISSION_BLOCKER
            .contains("reference-free Krea Realtime has NO admission surface")
            && KREA_REALTIME_ADMISSION_BLOCKER.contains("prices the RESIDENT load"),
        "the record must state why it ran no probe and what it does price: \
             {KREA_REALTIME_ADMISSION_BLOCKER}"
    );
}
