use super::*;

fn flux_one_planned(provider: &str, mode: &str, tier: &str) -> Value {
    let overlay = if provider == PULID_FLUX_PROVIDER {
        "identity"
    } else {
        "none"
    };
    let model_id = match provider {
        FLUX1_DEV_PROVIDER => "flux_dev",
        FLUX1_SCHNELL_PROVIDER => "flux_schnell",
        PULID_FLUX_PROVIDER => "pulid_flux_dev",
        other => other,
    };
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "modelId": model_id,
                "tier": tier,
                "mode": mode,
                "overlay": overlay,
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
            },
            "backend": "mlx",
            "loadShape": "deferred_materialization",
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": "unused",
            "fixture": "unused"
        }
    })
}

fn flux_one_temp_dir(label: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("sc-22726-{label}-{}-{nonce}", std::process::id()))
}

fn flux_one_snapshot_root(repository: &str, revision: &str, tier: &str) -> PathBuf {
    let root = flux_one_temp_dir("flux1")
        .join(format!("models--{}", repository.replace('/', "--")))
        .join("snapshots")
        .join(revision)
        .join(tier);
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// A complete staged PuLID bundle: all five loose files in ONE directory, which is what both
/// worker lanes stage under `SCENEWORKS_PULID_WEIGHTS` and what both engines require.
fn staged_pulid_bundle() -> protocol::PulidIdentityBundle {
    let root = flux_one_temp_dir("pulid-bundle");
    std::fs::create_dir_all(&root).unwrap();
    for file in protocol::PULID_IDENTITY_BUNDLE_FILES {
        std::fs::write(root.join(file), b"weights").unwrap();
    }
    protocol::pulid_identity_bundle_at(root).unwrap()
}

/// The family member is read off the plan's `(provider, mode)`, and a pair no member serves is
/// refused by name rather than measured as its nearest neighbour. `flux1_schnell` has no
/// identity route at all, and PuLID is text-to-image-WITH-A-FACE only (the worker's
/// `pulid_flux_available` requires `character_image` plus a reference), so both of those
/// crossings must fail.
#[test]
fn the_flux_one_arm_is_resolved_from_the_planned_provider_and_mode() {
    let dev = flux_one_arm(&flux_one_planned(FLUX1_DEV_PROVIDER, "text_to_image", "q4")).unwrap();
    assert_eq!(dev, FLUX1_DEV_ARM);
    assert_eq!(dev.expected_repository, protocol::FLUX1_DEV_REPOSITORY);
    assert!(!dev.identity);
    // Each base member binds ITS OWN env family — the one `measure-memory-catalog.mjs`
    // exports for it (`PROVIDER_FAMILIES.flux1_dev.env` / `.flux1_schnell.env`).
    assert_eq!(dev.repository_env, "SCENEWORKS_FLUX1_DEV_REPOSITORY");
    assert_eq!(dev.revision_env, "SCENEWORKS_FLUX1_DEV_REVISION");
    assert_eq!(dev.root_env, "SCENEWORKS_FLUX1_DEV_ROOT");
    let schnell = flux_one_arm(&flux_one_planned(
        FLUX1_SCHNELL_PROVIDER,
        "text_to_image",
        "q8",
    ))
    .unwrap();
    assert_eq!(schnell, FLUX1_SCHNELL_ARM);
    assert_eq!(
        schnell.expected_repository,
        protocol::FLUX1_SCHNELL_REPOSITORY
    );
    assert_eq!(
        schnell.repository_env,
        "SCENEWORKS_FLUX1_SCHNELL_REPOSITORY"
    );
    assert_eq!(schnell.revision_env, "SCENEWORKS_FLUX1_SCHNELL_REVISION");
    assert_eq!(schnell.root_env, "SCENEWORKS_FLUX1_SCHNELL_ROOT");
    let pulid = flux_one_arm(&flux_one_planned(
        PULID_FLUX_PROVIDER,
        "character_image",
        "bf16",
    ))
    .unwrap();
    assert_eq!(pulid, PULID_FLUX_ARM);
    assert!(pulid.identity);
    // The PuLID backbone IS the FLUX.1-dev artifact, so it binds the same env family — that is
    // the fact the closure's `engineId` alias and the shared PROVIDER_FAMILIES row encode.
    assert_eq!(pulid.expected_repository, protocol::FLUX1_DEV_REPOSITORY);
    assert_eq!(pulid.repository_env, "SCENEWORKS_FLUX1_DEV_REPOSITORY");
    assert_eq!(pulid.revision_env, "SCENEWORKS_FLUX1_DEV_REVISION");
    assert_eq!(pulid.root_env, "SCENEWORKS_FLUX1_DEV_ROOT");
    // ...but never the same record: the diagnostics source and the execution path differ.
    assert_ne!(pulid.slug, FLUX1_DEV_ARM.slug);
    assert_ne!(pulid.execution_path, FLUX1_DEV_ARM.execution_path);
    assert_ne!(pulid.still_calibration, FLUX1_DEV_ARM.still_calibration);

    for (provider, mode) in [
        (FLUX1_SCHNELL_PROVIDER, "character_image"),
        (PULID_FLUX_PROVIDER, "text_to_image"),
        (FLUX1_DEV_PROVIDER, "edit_image"),
        ("flux1_dev_control", "text_to_image"),
    ] {
        let error = flux_one_arm(&flux_one_planned(provider, mode, "q4")).unwrap_err();
        assert!(
            error.contains(&format!("provider {provider:?} in mode {mode:?}")),
            "{provider}/{mode}: {error}"
        );
    }
    assert!(
        flux_one_arm(&json!({ "planned": { "target": { "provider": FLUX1_DEV_PROVIDER } } }))
            .unwrap_err()
            .contains("planned.target.mode")
    );
}

/// The dispatcher routes all three ids into this arm, and still refuses everything else by
/// name. `run` used to send an unimplemented provider into the Qwen arm (sc-18104), so the
/// refusal sentence is asserted rather than assumed.
#[test]
fn run_dispatches_every_flux_one_id_and_still_refuses_a_foreign_one() {
    for provider in [
        FLUX1_DEV_PROVIDER,
        FLUX1_SCHNELL_PROVIDER,
        PULID_FLUX_PROVIDER,
    ] {
        let mode = if provider == PULID_FLUX_PROVIDER {
            "character_image"
        } else {
            "text_to_image"
        };
        let error = run(&flux_one_planned(provider, mode, "q4"))
            .expect_err("the weights-free plan cannot complete a real capture");
        assert!(
            !error.contains("does not implement provider"),
            "{provider} was not dispatched into the FLUX.1 arm: {error}"
        );
    }
    // Two ids no MLX arm serves: a control route the engine does not register, and the CANDLE
    // LTX-2.3 engine id. `chroma1_hd` used to stand here; sc-22731 armed it, which is exactly
    // the drift this loop exists to catch.
    for provider in ["flux1_dev_control", "ltx_2_3_distilled"] {
        let error = run(&flux_one_planned(provider, "text_to_image", "q4")).unwrap_err();
        assert_eq!(
            error,
            format!("MLX five-rung calibration does not implement provider {provider:?}")
        );
    }
}

/// The still-geometry guard fires BEFORE any environment or weight work, under each member's
/// own label — so a video geometry can never reach a FLUX.1 load (sc-18808).
#[test]
fn every_flux_one_member_refuses_a_multi_frame_geometry_under_its_own_label() {
    for arm in [FLUX1_DEV_ARM, FLUX1_SCHNELL_ARM, PULID_FLUX_ARM] {
        for frames in [0_u64, 2, 97] {
            let mut request = flux_one_planned(arm.provider, arm.mode, "q4");
            request["planned"]["target"]["geometry"]["frames"] = json!(frames);
            let error = run(&request).expect_err("a video geometry must be refused");
            assert_eq!(
                error,
                format!(
                    "{} requires geometry.frames == 1, got {frames}",
                    arm.still_calibration
                ),
                "{}/{frames}",
                arm.provider
            );
        }
    }
}

/// The tier is the PLAN's on every member, the root must carry it, and the `LoadSpec` re-asserts
/// it — a q8 plan against a q4 export is refused NAMING the tier it wanted (the sc-17097 defect
/// class). All three tiers must round-trip, so the arm cannot be quietly capped at one.
#[test]
fn flux_one_root_must_carry_the_planned_tier_and_the_spec_binds_it() {
    const REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
    for arm in [FLUX1_DEV_ARM, FLUX1_SCHNELL_ARM, PULID_FLUX_ARM] {
        let repository = arm.expected_repository;
        let identity = || arm.identity.then(staged_pulid_bundle);
        let q4_root = flux_one_snapshot_root(repository, REVISION, "q4");
        let error = flux_one_load_spec_at(
            &flux_one_planned(arm.provider, arm.mode, "q8"),
            LoadShape::DeferredMaterialization,
            repository.to_owned(),
            REVISION.to_owned(),
            q4_root.clone(),
            identity(),
        )
        .expect_err("a q8 plan must not be satisfied by a q4 root");
        assert!(
            error.ends_with(&format!("/snapshots/{REVISION}/q8")),
            "{}: {error}",
            arm.provider
        );
        for (tier, quant) in [
            ("q4", Some(Quant::Q4)),
            ("q8", Some(Quant::Q8)),
            ("bf16", None),
        ] {
            let root = flux_one_snapshot_root(repository, REVISION, tier);
            let artifact = flux_one_load_spec_at(
                &flux_one_planned(arm.provider, arm.mode, tier),
                LoadShape::DeferredMaterialization,
                repository.to_owned(),
                REVISION.to_owned(),
                root,
                identity(),
            )
            .unwrap_or_else(|error| panic!("{}/{tier}: {error}", arm.provider));
            assert_eq!(artifact.tier, tier);
            assert_eq!(artifact.spec.quantize, quant, "{tier} load quant");
            // Resident: the anchor rung, and the policy the worker loads every FLUX.1 route
            // under at that rung (no `requiredOffloadPolicy` on the manifests' `resident`
            // implementation). The identity is per (route, tier) and does not depend on it.
            assert_eq!(artifact.spec.offload_policy, OffloadPolicy::Resident);
            assert_eq!(artifact.spec.load_shape, LoadShape::DeferredMaterialization);
            assert_eq!(artifact.arm, arm);
            assert!(artifact
                .loadability_fingerprint()
                .starts_with(&format!("{repository}@{REVISION}:{tier}")));
        }
        // The wrong artifact family is refused before the root is even looked at.
        let other = if repository == protocol::FLUX1_DEV_REPOSITORY {
            protocol::FLUX1_SCHNELL_REPOSITORY
        } else {
            protocol::FLUX1_DEV_REPOSITORY
        };
        let error = flux_one_load_spec_at(
            &flux_one_planned(arm.provider, arm.mode, "q4"),
            LoadShape::DeferredMaterialization,
            other.to_owned(),
            REVISION.to_owned(),
            q4_root,
            identity(),
        )
        .expect_err("the other family's repository must be refused");
        assert!(error.contains(repository), "{}: {error}", arm.provider);
    }
}

/// The identity stack rides the `LoadSpec` exactly as the worker threads it
/// (`image_jobs/pulid.rs` `pulid_identity_weights`), the base routes carry none, and the
/// resolved bundle is part of the loadability fingerprint — two identity stacks over one
/// backbone are two different measurements.
#[test]
fn the_pulid_route_binds_the_identity_stack_and_the_base_routes_do_not() {
    const REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
    let bundle = staged_pulid_bundle();
    let root = flux_one_snapshot_root(protocol::FLUX1_DEV_REPOSITORY, REVISION, "q4");
    let artifact = flux_one_load_spec_at(
        &flux_one_planned(PULID_FLUX_PROVIDER, "character_image", "q4"),
        LoadShape::DeferredMaterialization,
        protocol::FLUX1_DEV_REPOSITORY.to_owned(),
        REVISION.to_owned(),
        root.clone(),
        Some(bundle.clone()),
    )
    .unwrap();
    let identity = artifact.spec.identity.as_ref().expect("identity stack");
    assert_eq!(
        identity.encoder,
        Some(WeightsSource::File(
            bundle.root.join(protocol::PULID_ADAPTER_FILE)
        ))
    );
    assert_eq!(
        identity.eva,
        Some(WeightsSource::File(
            bundle.root.join(protocol::PULID_EVA_FILE)
        ))
    );
    // The engine reads scrfd / arcface / bisenet out of `face_dir` BY NAME, so the bundle root
    // IS the face dir — not a sibling of it.
    assert_eq!(
        identity.face_dir,
        Some(WeightsSource::Dir(bundle.root.clone()))
    );
    // The fingerprint names the stack by CONTENT, never by the host path it was staged at.
    assert!(
        artifact
            .loadability_fingerprint()
            .ends_with(&format!("+identity:{}", bundle.composite_sha256)),
        "{}",
        artifact.loadability_fingerprint()
    );
    assert!(!artifact
        .loadability_fingerprint()
        .contains(&bundle.root.display().to_string()));
    // ...and the schema-compatible inventory digest binds every file's content.
    let captured = artifact.artifact_json();
    serde_json::from_value::<sceneworks_core::memory_calibration::Artifact>(captured.clone())
        .expect("PuLID capture artifact must pass the production reader");
    assert_eq!(
        captured["inventorySha256"].as_str(),
        Some(bundle.composite_sha256.as_str())
    );
    // Same bytes at another path: the same identity. Different bytes: a different one.
    let restaged = staged_pulid_bundle();
    assert_ne!(restaged.root, bundle.root);
    assert_eq!(restaged.composite_sha256, bundle.composite_sha256);
    std::fs::write(restaged.root.join(protocol::PULID_BISENET_FILE), b"other").unwrap();
    let restaged = protocol::pulid_identity_bundle_at(restaged.root).unwrap();
    assert_ne!(restaged.composite_sha256, bundle.composite_sha256);
    assert_ne!(
        restaged.file_sha256[4].1, bundle.file_sha256[4].1,
        "the bisenet digest is the one that moved"
    );
    assert_eq!(restaged.file_sha256[..4], bundle.file_sha256[..4]);

    let base = flux_one_load_spec_at(
        &flux_one_planned(FLUX1_DEV_PROVIDER, "text_to_image", "q4"),
        LoadShape::DeferredMaterialization,
        protocol::FLUX1_DEV_REPOSITORY.to_owned(),
        REVISION.to_owned(),
        root.clone(),
        None,
    )
    .unwrap();
    assert!(base.spec.identity.is_none());
    assert!(!base.loadability_fingerprint().contains("+identity"));

    // Neither crossing is expressible: an identity stack on a base route, or a PuLID plan with
    // no stack, is refused rather than measured as the other route.
    for (provider, mode, supplied) in [
        (FLUX1_DEV_PROVIDER, "text_to_image", Some(bundle.clone())),
        (PULID_FLUX_PROVIDER, "character_image", None),
    ] {
        let error = flux_one_load_spec_at(
            &flux_one_planned(provider, mode, "q4"),
            LoadShape::DeferredMaterialization,
            protocol::FLUX1_DEV_REPOSITORY.to_owned(),
            REVISION.to_owned(),
            root.clone(),
            supplied,
        )
        .expect_err("the identity stack and the route must agree");
        assert!(error.contains("identity"), "{provider}: {error}");
    }
}

/// `flux_one_load_spec` reads the identity bundle from `SCENEWORKS_PULID_WEIGHTS` on the
/// PuLID route and from nothing on a base route. Serialized on [`FLUX_ONE_ENV_LOCK`]: the
/// process environment is global, and every other FLUX.1 test in this binary fails before it
/// reads an env var, so nothing else needs the lock.
#[test]
fn the_env_bound_load_spec_reads_the_identity_bundle_from_its_env_on_the_pulid_route_only() {
    const REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
    let _guard = FLUX_ONE_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let restore = FluxOneEnv::capture();
    let bundle = staged_pulid_bundle();
    let root = flux_one_snapshot_root(protocol::FLUX1_DEV_REPOSITORY, REVISION, "q4");
    std::env::set_var(
        "SCENEWORKS_FLUX1_DEV_REPOSITORY",
        protocol::FLUX1_DEV_REPOSITORY,
    );
    std::env::set_var("SCENEWORKS_FLUX1_DEV_REVISION", REVISION);
    std::env::set_var("SCENEWORKS_FLUX1_DEV_ROOT", &root);
    std::env::remove_var(protocol::PULID_IDENTITY_BUNDLE_ENV);
    let pulid = flux_one_planned(PULID_FLUX_PROVIDER, "character_image", "q4");
    let error = flux_one_load_spec(&pulid, LoadShape::DeferredMaterialization)
        .expect_err("the PuLID route must require the bundle env");
    assert!(
        error.contains(protocol::PULID_IDENTITY_BUNDLE_ENV),
        "{error}"
    );
    // A base route reads no bundle env at all: it binds with the variable absent.
    let dev = flux_one_planned(FLUX1_DEV_PROVIDER, "text_to_image", "q4");
    let base = flux_one_load_spec(&dev, LoadShape::DeferredMaterialization).unwrap();
    assert!(base.identity.is_none());

    std::env::set_var(protocol::PULID_IDENTITY_BUNDLE_ENV, &bundle.root);
    let artifact = flux_one_load_spec(&pulid, LoadShape::DeferredMaterialization).unwrap();
    let identity = artifact.identity.as_ref().expect("the bundle env was read");
    assert_eq!(identity.composite_sha256, bundle.composite_sha256);
    assert_eq!(
        identity.root,
        std::fs::canonicalize(&bundle.root).unwrap(),
        "the env value is canonicalized, not used as spelled"
    );
    // The schnell family's env is NOT consulted on the dev backbone: pointing it somewhere
    // unusable changes nothing for either dev-backed member.
    std::env::set_var("SCENEWORKS_FLUX1_SCHNELL_ROOT", "/nonexistent/sc-22726");
    std::env::set_var("SCENEWORKS_FLUX1_SCHNELL_REPOSITORY", "not/a-repo");
    std::env::set_var("SCENEWORKS_FLUX1_SCHNELL_REVISION", "junk");
    flux_one_load_spec(&pulid, LoadShape::DeferredMaterialization).unwrap();
    flux_one_load_spec(&dev, LoadShape::DeferredMaterialization).unwrap();
    // ...and a schnell plan reads ITS family, so it is refused on those values by name.
    let schnell = flux_one_planned(FLUX1_SCHNELL_PROVIDER, "text_to_image", "q4");
    let error = flux_one_load_spec(&schnell, LoadShape::DeferredMaterialization)
        .expect_err("the schnell env family is the one a schnell plan reads");
    assert!(
        error.contains(protocol::FLUX1_SCHNELL_REPOSITORY),
        "{error}"
    );
    drop(restore);
}

/// Serializes the tests that mutate the FLUX.1 env families and the PuLID bundle env.
static FLUX_ONE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Every env var the FLUX.1 arm reads, restored on drop — including on a panic, so a failing
/// assertion cannot leak a binding into a later test.
struct FluxOneEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl FluxOneEnv {
    const NAMES: [&'static str; 7] = [
        "SCENEWORKS_FLUX1_DEV_REPOSITORY",
        "SCENEWORKS_FLUX1_DEV_REVISION",
        "SCENEWORKS_FLUX1_DEV_ROOT",
        "SCENEWORKS_FLUX1_SCHNELL_REPOSITORY",
        "SCENEWORKS_FLUX1_SCHNELL_REVISION",
        "SCENEWORKS_FLUX1_SCHNELL_ROOT",
        protocol::PULID_IDENTITY_BUNDLE_ENV,
    ];

    fn capture() -> Self {
        Self(
            Self::NAMES
                .iter()
                .map(|name| (*name, std::env::var_os(name)))
                .collect(),
        )
    }
}

impl Drop for FluxOneEnv {
    fn drop(&mut self) {
        for (name, value) in &self.0 {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

/// An incomplete staged bundle is refused HERE, naming the missing files, rather than several
/// gigabytes into a capture as an opaque loader error.
#[test]
fn an_incomplete_pulid_bundle_is_refused_naming_the_missing_files() {
    let root = flux_one_temp_dir("pulid-partial");
    std::fs::create_dir_all(&root).unwrap();
    for skipped in protocol::PULID_IDENTITY_BUNDLE_FILES {
        for file in protocol::PULID_IDENTITY_BUNDLE_FILES {
            let path = root.join(file);
            if file == skipped {
                let _ = std::fs::remove_file(&path);
            } else {
                std::fs::write(&path, b"weights").unwrap();
            }
        }
        let error = protocol::pulid_identity_bundle_at(root.clone())
            .expect_err("an incomplete bundle must be refused");
        assert!(error.contains(skipped), "{skipped}: {error}");
        assert!(error.contains(protocol::PULID_IDENTITY_BUNDLE_ENV));
    }
}

/// The overlay axis is read off the declared target on both routes: a base plan that declares
/// `identity` is refused, and a PuLID plan that declares `none` is too — neither may acquire
/// the other's overlay coverage.
#[test]
fn the_flux_one_overlay_axis_is_bound_to_the_declared_target() {
    const REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
    let root = flux_one_snapshot_root(protocol::FLUX1_DEV_REPOSITORY, REVISION, "q4");
    let mut base = flux_one_planned(FLUX1_DEV_PROVIDER, "text_to_image", "q4");
    base["planned"]["target"]["overlay"] = json!("identity");
    let error = flux_one_load_spec_at(
        &base,
        LoadShape::DeferredMaterialization,
        protocol::FLUX1_DEV_REPOSITORY.to_owned(),
        REVISION.to_owned(),
        root.clone(),
        None,
    )
    .expect_err("a base route must refuse a declared overlay");
    assert!(error.contains("only executes the base target"), "{error}");

    let mut pulid = flux_one_planned(PULID_FLUX_PROVIDER, "character_image", "q4");
    pulid["planned"]["target"]["overlay"] = json!("none");
    let error = flux_one_load_spec_at(
        &pulid,
        LoadShape::DeferredMaterialization,
        protocol::FLUX1_DEV_REPOSITORY.to_owned(),
        REVISION.to_owned(),
        root,
        Some(staged_pulid_bundle()),
    )
    .expect_err("the identity route must require its own overlay");
    assert!(error.contains("executes exactly \"identity\""), "{error}");
}

/// The measured request is the worker's: two steps on every member, and exactly one reference
/// face at the target geometry on the identity route (never on a base route).
#[test]
fn the_flux_one_request_carries_a_reference_only_on_the_identity_route() {
    let pulid = flux_one_request(PULID_FLUX_ARM, 1024, 768);
    assert_eq!(pulid.conditioning.len(), 1);
    match &pulid.conditioning[0] {
        Conditioning::Reference { image, strength } => {
            assert_eq!((image.width, image.height), (1024, 768));
            assert_eq!(image.pixels.len(), 1024 * 768 * 3);
            assert_eq!(*strength, Some(PULID_REFERENCE_STRENGTH));
        }
        other => panic!("expected one Reference, got {other:?}"),
    }
    assert_eq!(pulid.steps, Some(2));
    assert_eq!(pulid.seed, Some(FLUX1_SEED));
    for arm in [FLUX1_DEV_ARM, FLUX1_SCHNELL_ARM] {
        let base = flux_one_request(arm, 1024, 1024);
        assert!(base.conditioning.is_empty(), "{}", arm.provider);
        assert_eq!(base.steps, Some(2));
        assert_eq!(base.seed, Some(FLUX1_SEED));
    }
}

/// The admission context is the shape each route is admitted under. The identity route is the
/// PROVIDER mode `image_to_image` with one reference and `overlay: identity` — exactly what
/// `mlx-gen-pulid`'s route gate checks and what the manifest's `requestContexts` declare — and
/// the base routes are reference-free text-to-image.
#[test]
fn the_flux_one_admission_context_matches_each_declared_route() {
    let calibration = MemoryCalibrationIdentity::new(
        "flux-one-static-registry-behavior-v2-dev",
        LoadShape::DeferredMaterialization,
    );
    let selection =
        planned_selection(&flux_one_planned(FLUX1_DEV_PROVIDER, "text_to_image", "q4")).unwrap();
    let pulid = flux_one_context(
        PULID_FLUX_ARM,
        &selection,
        &calibration,
        &calibration.fingerprint,
        1024,
        1024,
        1,
        1,
    );
    assert_eq!(pulid.mode, MemoryMode::ImageToImage);
    assert!(pulid.has_reference);
    assert_eq!(pulid.geometry.reference_count, 1);
    assert_eq!(pulid.overlay.as_deref(), Some("identity"));
    assert!(!pulid.use_pid);
    assert!(!pulid.has_phases);
    for arm in [FLUX1_DEV_ARM, FLUX1_SCHNELL_ARM] {
        let base = flux_one_context(
            arm,
            &selection,
            &calibration,
            &calibration.fingerprint,
            1024,
            1024,
            1,
            1,
        );
        assert_eq!(base.mode, MemoryMode::TextToImage, "{}", arm.provider);
        assert!(!base.has_reference);
        assert_eq!(base.geometry.reference_count, 0);
        assert_eq!(base.overlay, None);
    }
}

/// The fixture names the member, the tier, the edge and the step count, so a bf16 record can
/// never be emitted against a q4 capture that merely reused the fixture string.
#[test]
fn the_flux_one_fixture_binds_the_member_tier_geometry_and_step_count() {
    let request = flux_one_planned(FLUX1_DEV_PROVIDER, "text_to_image", "q4");
    validate_flux_one_fixture(
        &json!({ "planned": {
            "fixture": format!("flux1-dev-mlx-q4-1024-seed{FLUX1_SEED}-step2"),
            "target": request["planned"]["target"].clone(),
        }}),
        FLUX1_DEV_ARM,
        "q4",
    )
    .unwrap();
    for (fixture, expected) in [
        (
            format!("flux1-dev-mlx-q8-1024-seed{FLUX1_SEED}-step2"),
            "must start with",
        ),
        (
            format!("pulid-flux-mlx-q4-1024-seed{FLUX1_SEED}-step2"),
            "must start with",
        ),
        (
            format!("flux1-dev-mlx-q4-768-seed{FLUX1_SEED}-step2"),
            "must start with",
        ),
        (
            format!("flux1-dev-mlx-q4-1024-seed{FLUX1_SEED}-step3"),
            "two-step",
        ),
        (
            "flux1-dev-mlx-q4-1024-seed1-step2".to_owned(),
            "does not match",
        ),
    ] {
        let error = validate_flux_one_fixture(
            &json!({ "planned": {
                "fixture": fixture,
                "target": request["planned"]["target"].clone(),
            }}),
            FLUX1_DEV_ARM,
            "q4",
        )
        .expect_err("the fixture must be bound to its cell");
        assert!(error.contains(expected), "{fixture}: {error}");
    }
}

/// Every FLUX.1 cell the committed plan declares must resolve to an implemented member, and its
/// fixture must satisfy the same binding a capture would apply. This is the plan/arm agreement
/// E3 asks for, checked against the checked-in plan rather than a hand-written sample.
#[test]
fn every_planned_flux_one_mlx_cell_resolves_to_an_implemented_member() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let anchors = plan["anchors"].as_object().expect("anchors object");
    let mut seen = std::collections::BTreeSet::new();
    for (key, entry) in anchors {
        if !key.ends_with(":mlx") {
            continue;
        }
        let provider = entry["provider"].as_str().unwrap();
        if !matches!(
            provider,
            FLUX1_DEV_PROVIDER | FLUX1_SCHNELL_PROVIDER | PULID_FLUX_PROVIDER
        ) {
            continue;
        }
        seen.insert(key.clone());
        let (_, rest) = key.split_once(':').unwrap();
        let tier = rest.split_once(':').unwrap().0;
        let request = json!({ "planned": {
            "target": {
                "provider": provider,
                "tier": tier,
                "mode": entry["mode"].clone(),
                "overlay": entry["overlay"].clone(),
                "geometry": entry["geometry"].clone(),
            },
            "loadShape": entry["loadShape"].clone(),
            "fixture": entry["fixture"].clone(),
        }});
        let arm = flux_one_arm(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        validate_flux_one_fixture(&request, arm, tier)
            .unwrap_or_else(|error| panic!("{key}: {error}"));
        assert_eq!(
            planned_load_shape(&request).unwrap(),
            LoadShape::DeferredMaterialization,
            "{key}: the FLUX.1 arm attests the deferred load shape"
        );
    }
    // The exact cell set, not a count: three members x three tiers, each named once.
    let expected: std::collections::BTreeSet<String> =
        ["flux_dev", "flux_schnell", "pulid_flux_dev"]
            .iter()
            .flat_map(|model| {
                ["bf16", "q4", "q8"]
                    .iter()
                    .map(move |tier| format!("{model}:{tier}:mlx"))
            })
            .collect();
    assert_eq!(seen, expected);
}

/// Every FLUX.1 MLX plan row names the production calibration identity its loaded generator
/// publishes (inference PR 943's per-(route, tier) table): the two measured keys are the
/// engine constants byte-for-byte, no row names a weights-free conformance string, and the
/// nine identities are distinct. The existing member-resolution test never looked at
/// `calibrationFingerprint`, which is how seven rows shipped naming strings no production load
/// can return.
#[test]
fn every_planned_flux_one_mlx_cell_names_the_identity_its_loaded_generator_publishes() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut identities = std::collections::BTreeMap::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        let provider = entry["provider"].as_str().unwrap();
        if !key.ends_with(":mlx")
            || !matches!(
                provider,
                FLUX1_DEV_PROVIDER | FLUX1_SCHNELL_PROVIDER | PULID_FLUX_PROVIDER
            )
        {
            continue;
        }
        let tier = key.split(':').nth(1).unwrap();
        let mode = entry["mode"].as_str().unwrap();
        let arm = flux_one_arm(&flux_one_planned(provider, mode, tier)).unwrap();
        let planned = entry["calibrationFingerprint"]
            .as_str()
            .unwrap_or_else(|| panic!("{key}: calibrationFingerprint must be a string"));
        assert!(
            !is_flux_one_weights_free_fingerprint(planned),
            "{key}: {planned} is a weights-free conformance identity no production load returns"
        );
        assert_eq!(
            planned,
            flux_one_calibration_fingerprint(arm, tier),
            "{key}: the plan row must name the loaded generator's production identity"
        );
        assert!(
            identities.insert(planned.to_owned(), key.clone()).is_none(),
            "{key}: {planned} is already claimed by {}",
            identities[planned]
        );
    }
    assert_eq!(identities.len(), 9, "nine distinct FLUX.1 MLX identities");
    // The preserved measured keys, byte-for-byte, so the table cannot drift off the engine.
    assert_eq!(
        identities["flux1-dev-q4-mlx-shared-ladder-2026-08-03-v1"],
        "flux_dev:q4:mlx"
    );
    assert_eq!(
        identities["pulid-flux-dev-q4-mlx-shared-ladder-2026-08-04-v1"],
        "pulid_flux_dev:q4:mlx"
    );
    assert_eq!(
        runtime_macos::providers::flux::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
        "flux1-dev-q4-mlx-shared-ladder-2026-08-03-v1"
    );
    assert_eq!(
        runtime_macos::providers::pulid::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
        "pulid-flux-dev-q4-mlx-shared-ladder-2026-08-04-v1"
    );
    // The seven new cells, spelled out: the strings inference PR 943 mints.
    for (key, identity) in [
        ("flux_dev:q8:mlx", "flux1-dev-q8-mlx-shared-ladder-v1"),
        ("flux_dev:bf16:mlx", "flux1-dev-bf16-mlx-shared-ladder-v1"),
        (
            "flux_schnell:q4:mlx",
            "flux1-schnell-q4-mlx-shared-ladder-v1",
        ),
        (
            "flux_schnell:q8:mlx",
            "flux1-schnell-q8-mlx-shared-ladder-v1",
        ),
        (
            "flux_schnell:bf16:mlx",
            "flux1-schnell-bf16-mlx-shared-ladder-v1",
        ),
        (
            "pulid_flux_dev:q8:mlx",
            "pulid-flux-dev-q8-mlx-shared-ladder-v1",
        ),
        (
            "pulid_flux_dev:bf16:mlx",
            "pulid-flux-dev-bf16-mlx-shared-ladder-v1",
        ),
    ] {
        assert_eq!(identities[identity], key);
    }
}

/// The arm binds the plan row to the table BEFORE any environment or weight work, naming
/// both strings, so a stale row cannot cost a load.
#[test]
fn a_plan_row_naming_a_foreign_identity_is_refused_before_the_load() {
    let mut request = flux_one_planned(FLUX1_SCHNELL_PROVIDER, "text_to_image", "q8");
    request["planned"]["fixture"] =
        json!(format!("flux1-schnell-mlx-q8-1024-seed{FLUX1_SEED}-step2"));
    request["planned"]["calibrationFingerprint"] =
        json!("flux-one-static-registry-behavior-v2-schnell");
    let error = run(&request).expect_err("a conformance identity is not capturable");
    assert!(
        error.contains("plan=flux-one-static-registry-behavior-v2-schnell")
            && error.contains("flux1-schnell-q8-mlx-shared-ladder-v1"),
        "{error}"
    );
    // Not an env error: the refusal fired before `SCENEWORKS_FLUX1_SCHNELL_*` was read.
    assert!(!error.contains("required environment variable"), "{error}");
}
