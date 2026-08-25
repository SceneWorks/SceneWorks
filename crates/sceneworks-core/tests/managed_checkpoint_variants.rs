//! Managed NVFP4 checkpoint variants (sc-11043, epic 11037).
//!
//! The story registers two pinned upstream artifacts as explicitly named NVFP4 managed variants
//! and routes every verb through epic 20398's existing ownership contracts. These tests hold the
//! three acceptance criteria and their mutations:
//!
//! 1. both artifacts are registered, explicitly NVFP4, never q4, never replacing a tier — and
//!    installing one compiles the SAME semantic plan as linking the identical bytes;
//! 2. install and removal delegate to the managed lifecycle while a linked source stays external
//!    and byte-for-byte untouched;
//! 3. no conversion happens at install, and provenance records the pinned revision and the
//!    verified checksum.
//!
//! Plus the property that makes NVFP4 a deliberate choice rather than a default: nothing in the
//! registry maps a model onto a variant, so auto-selection has nothing to reach.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use sceneworks_core::checkpoint_import::SourceLocatorV1;
use sceneworks_core::checkpoint_ingest::ManagedIngestError;
use sceneworks_core::checkpoint_plan_store::{linked_checkpoint_id, CheckpointPlanStore};
use sceneworks_core::jobs_store::{
    imported_image_request_provider_eligible, imported_provider_routes,
};
use sceneworks_core::managed_checkpoint_variants::{
    managed_nvfp4_variant, managed_nvfp4_variants, ManagedCheckpointVariantV1, PinnedArtifactV1,
    MANAGED_VARIANT_PROVENANCE_SOURCE, MANAGED_VARIANT_SOURCE_SHAPE,
};

const KREA_VARIANT: &str = "nvfp4-krea-2-turbo";
const KLEIN_VARIANT: &str = "nvfp4-flux2-klein-9b-true-v2";

// ---- fixtures ---------------------------------------------------------------------------------

/// A minimal but REAL NVFP4 single file: the `_quantization_metadata` declaration both pinned
/// artifacts carry, plus complete U8 / F8_E4M3 / F32 triplets whose shapes satisfy the header
/// classifier. `marker` selects the architecture family, and `descriptors` reproduces the FLUX.2
/// Klein `nvfp4mixed` shape, which carries a `.comfy_quant` blob beside every packed projection.
fn write_nvfp4_file(path: &Path, marker: &str, descriptors: bool, fill: u8) {
    let layers = 6;
    let mut declared = Map::new();
    for index in 0..layers {
        declared.insert(
            format!("blocks.{index}.attn.wq"),
            json!({"format": "nvfp4", "group_size": 16}),
        );
    }
    let mut entries: Vec<(String, &'static str, Vec<u64>)> =
        vec![(marker.to_owned(), "BF16", vec![128, 128])];
    for index in 0..layers {
        let base = format!("blocks.{index}.attn.wq");
        entries.push((format!("{base}.weight"), "U8", vec![128, 32]));
        entries.push((format!("{base}.weight_scale"), "F8_E4M3", vec![128, 4]));
        entries.push((format!("{base}.weight_scale_2"), "F32", vec![]));
        if descriptors {
            entries.push((format!("{base}.comfy_quant"), "U8", vec![64]));
        }
    }

    let mut header = Map::new();
    header.insert(
        "__metadata__".to_owned(),
        json!({
            "format": "pt",
            "_quantization_metadata":
                serde_json::to_string(&json!({"layers": declared})).unwrap(),
        }),
    );
    let mut offset = 0_u64;
    for (name, dtype, shape) in &entries {
        let width = match *dtype {
            "F32" => 4,
            "BF16" | "F16" => 2,
            _ => 1,
        };
        let count = shape.iter().product::<u64>().max(1);
        let bytes = count * width;
        header.insert(
            name.clone(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        offset += bytes;
    }

    let encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.resize(bytes.len() + offset as usize, fill);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

fn sha256_of(path: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(fs::read(path).unwrap());
    format!("{:x}", hasher.finalize())
}

struct Fixture {
    _data: TempDir,
    _library: TempDir,
    store: CheckpointPlanStore,
    library_dir: PathBuf,
}

fn fixture(label: &str) -> Fixture {
    let data = tempfile::Builder::new()
        .prefix(&format!("variant-{label}-data-{}-", std::process::id()))
        .tempdir()
        .unwrap();
    let library = tempfile::Builder::new()
        .prefix(&format!("variant-{label}-lib-{}-", std::process::id()))
        .tempdir()
        .unwrap();
    let library_dir = fs::canonicalize(library.path()).unwrap();
    Fixture {
        store: CheckpointPlanStore::open(data.path()),
        library_dir,
        _data: data,
        _library: library,
    }
}

/// The registered variant with a fixture file standing in for its multi-gigabyte pinned bytes.
///
/// Everything except the digest and size is the REAL registration — the same id, provider, family,
/// repo, revision, and installed path — so the store, the ingest, and the provenance under test
/// are exactly the ones production uses. Only the two facts that describe the bytes themselves are
/// re-pinned to the bytes actually on disk, which is what lets the pinned-checksum verification
/// path run for real instead of being stubbed out.
fn variant_over(
    registered: &ManagedCheckpointVariantV1,
    file: &Path,
) -> ManagedCheckpointVariantV1 {
    ManagedCheckpointVariantV1::new(
        &registered.variant_id,
        &registered.display_name,
        &registered.provider,
        &registered.family,
        PinnedArtifactV1::new(
            &registered.repo,
            &registered.revision,
            &registered.repo_file,
            sha256_of(file),
            fs::metadata(file).unwrap().len(),
        ),
    )
    .expect("a fixture-backed variant is still a well-formed registration")
}

fn registered(variant_id: &str) -> &'static ManagedCheckpointVariantV1 {
    managed_nvfp4_variant(variant_id).expect("the variant is registered")
}

/// The family marker and descriptor shape each registered variant's real artifact carries.
fn fixture_shape(variant_id: &str) -> (&'static str, bool) {
    match variant_id {
        KREA_VARIANT => ("model.diffusion_model.txtfusion.projector.weight", false),
        KLEIN_VARIANT => ("double_stream_modulation_lin.weight", true),
        other => panic!("no fixture shape for {other}"),
    }
}

/// A manifest entry as the import job records one for this variant.
fn manifest_entry(variant: &ManagedCheckpointVariantV1) -> Map<String, Value> {
    let mut entry = Map::new();
    entry.insert("family".to_owned(), json!(variant.family));
    entry.insert(
        "importSourceShape".to_owned(),
        json!(MANAGED_VARIANT_SOURCE_SHAPE),
    );
    entry.insert("importQuantFormat".to_owned(), json!(variant.quant_tier));
    entry.insert(
        "importPlan".to_owned(),
        json!({ "checkpointId": variant.checkpoint_id() }),
    );
    entry
}

fn generate_request(variant: &ManagedCheckpointVariantV1, advanced: Value) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("mode".to_owned(), json!("text_to_image"));
    payload.insert(
        "modelManifestEntry".to_owned(),
        Value::Object(manifest_entry(variant)),
    );
    payload.insert("advanced".to_owned(), advanced);
    payload
}

// ---- AC1: registration, explicit naming, and linked/managed plan equality ----------------------

/// The exact artifacts epic 11037 pins, registered and named for what they are.
///
/// The pins are asserted literally. They are the whole point of a managed variant: the app is
/// promising to fetch THESE bytes from THIS revision, and a silent edit to any of them is a
/// different artifact wearing the same name.
#[test]
fn both_pinned_artifacts_are_registered_as_explicitly_named_nvfp4_variants() {
    let variants = managed_nvfp4_variants();
    assert_eq!(
        variants
            .iter()
            .map(|variant| variant.variant_id.as_str())
            .collect::<Vec<_>>(),
        [KREA_VARIANT, KLEIN_VARIANT]
    );

    let krea = registered(KREA_VARIANT);
    assert_eq!(krea.provider, "krea_2_turbo");
    assert_eq!(krea.family, "krea_2");
    assert_eq!(krea.repo, "Comfy-Org/Krea-2");
    assert_eq!(krea.revision, "952f49d49653cb42e7d6cf7cbfad74738073ec7d");
    assert_eq!(
        krea.repo_file,
        "diffusion_models/krea2_turbo_nvfp4.safetensors"
    );
    assert_eq!(krea.relative_path, "krea2_turbo_nvfp4.safetensors");
    assert_eq!(
        krea.sha256,
        "61527003b2d537055494d01bc8efe51d6e86e64192ba23e3721a5647231fe394"
    );
    assert_eq!(krea.size_bytes, 7_673_668_448);

    let klein = registered(KLEIN_VARIANT);
    assert_eq!(klein.provider, "flux2_klein_9b");
    assert_eq!(klein.family, "flux2");
    assert_eq!(klein.repo, "wikeeyang/Flux2-Klein-9B-True-V2");
    assert_eq!(klein.revision, "9c9fe9880029a4e0c4af5ca7d86e83cdb83eea83");
    assert_eq!(
        klein.repo_file,
        "Flux2-Klein-9B-True-v2-nvfp4mixed.safetensors"
    );
    assert_eq!(
        klein.sha256,
        "32ab833377c6a6052508ee3d29c1cb0f5cd2eeb369518fb6e740ee35645ecadb"
    );
    assert_eq!(klein.size_bytes, 5_616_278_928);

    for variant in variants {
        assert_eq!(variant.quant_tier, "nvfp4", "{}", variant.variant_id);
        assert_eq!(variant.source_codec, "nvfp4-v1", "{}", variant.variant_id);
        assert!(
            variant.display_name.contains("NVFP4"),
            "{} must name its tier in the product: {:?}",
            variant.variant_id,
            variant.display_name
        );
        assert!(
            variant.validate().is_ok(),
            "{} must be a valid registration",
            variant.variant_id
        );
    }
}

/// A registered variant carries ONE identity for its bytes, whoever owns them.
///
/// Asserted against the real pinned digests, which needs no bytes on disk: the semantic identity
/// of a source IS its content digest, and ownership lives only in the binding identity. This is
/// the contract half of "installing compiles the same semantic plan as linking"; the store half
/// is below.
#[test]
fn a_registered_variant_has_one_semantic_identity_and_two_bindings() {
    for variant in managed_nvfp4_variants() {
        let managed = variant.managed_locator().unwrap();
        let linked = variant
            .linked_locator(
                "user-library",
                &format!("krea/v2/{}", variant.relative_path),
            )
            .unwrap();

        assert_eq!(
            managed.semantic_identity().unwrap(),
            linked.semantic_identity().unwrap(),
            "{}: equal bytes are the same checkpoint under either ownership",
            variant.variant_id
        );
        assert_ne!(
            managed.source_binding_identity().unwrap(),
            linked.source_binding_identity().unwrap(),
            "{}: ownership, path, and provenance still separate the two bindings",
            variant.variant_id
        );
        assert_eq!(managed.content_digest().unwrap(), variant.sha256);
        assert_eq!(linked.content_digest().unwrap(), variant.sha256);
    }
}

/// The store half, run through the real production compiles for BOTH variants' architectures: a
/// managed install of the file and a linked reference to the identical bytes in the user's own
/// library produce the same semantic plan, and therefore find each other as duplicates.
///
/// Neither digest is derived from the other — one comes out of `compile_linked` through an
/// approved root, the other out of `ManagedIngest::finalize`'s staging/commit path.
#[test]
fn installing_a_variant_compiles_the_same_semantic_plan_as_linking_the_identical_file() {
    for variant_id in [KREA_VARIANT, KLEIN_VARIANT] {
        let fixture = fixture(&format!("parity-{variant_id}"));
        let registered = registered(variant_id);
        let (marker, descriptors) = fixture_shape(variant_id);

        let nested = format!("vendors/{}/{}", registered.family, registered.relative_path);
        let source = fixture.library_dir.join(&nested);
        write_nvfp4_file(&source, marker, descriptors, 0x5a);
        let variant = variant_over(registered, &source);

        let root = fixture.store.approve_root(&fixture.library_dir).unwrap();
        let linked = fixture
            .store
            .compile_linked(&root.root_id, &nested)
            .unwrap();

        let ingest = variant.begin_install(&fixture.store).unwrap();
        let staged = ingest.staged_path(&variant.relative_path).unwrap();
        fs::copy(&source, &staged).unwrap();
        let install = variant.finalize_install(ingest).unwrap();

        variant
            .confirm_installed(&install)
            .expect("the committed install is the pinned variant");
        assert_eq!(
            install.compiled.plan.semantic_digest().unwrap(),
            linked.plan.semantic_digest().unwrap(),
            "{variant_id}: a managed install and a linked reference are the same checkpoint"
        );
        assert_eq!(
            install.compiled.record.summary.semantic_digest, linked.record.summary.semantic_digest,
            "{variant_id}: the catalog summaries agree too"
        );
        assert_ne!(
            install.compiled.plan.source_binding_identity().unwrap(),
            linked.plan.source_binding_identity().unwrap(),
            "{variant_id}: the two bindings stay distinct"
        );
        assert_eq!(
            install.duplicate_checkpoint_ids(),
            [linked_checkpoint_id(&root.root_id, &nested)],
            "{variant_id}: the two copies find each other"
        );

        // The plan describes the checkpoint, never the tier: NVFP4 identity is a fact about the
        // bytes, read from the header, and is deliberately not restated in the plan document.
        let plan_json = install.compiled.plan.canonical_json().unwrap();
        assert!(
            !plan_json.contains("q4"),
            "{variant_id}: a plan must never carry a q4 spelling: {plan_json}"
        );
        assert_eq!(install.compiled.plan.layers.len(), 1);
        assert_eq!(
            install.compiled.plan.layers[0].target_path,
            variant.relative_path
        );
    }
}

// ---- AC2: ownership and lifecycle delegation ---------------------------------------------------

/// Removal drops exactly the state SceneWorks owns, and the user's file — which a LINKED import of
/// the same variant references in place — is still there, byte for byte.
#[test]
fn install_and_removal_delegate_to_the_managed_ownership_contract() {
    let fixture = fixture("lifecycle");
    let registered = registered(KREA_VARIANT);
    let (marker, descriptors) = fixture_shape(KREA_VARIANT);

    let source = fixture.library_dir.join(&registered.relative_path);
    write_nvfp4_file(&source, marker, descriptors, 0x11);
    let before = fs::read(&source).unwrap();
    let variant = variant_over(registered, &source);

    let root = fixture.store.approve_root(&fixture.library_dir).unwrap();
    let linked = fixture
        .store
        .compile_linked(&root.root_id, &registered.relative_path)
        .unwrap();

    let ingest = variant.begin_install(&fixture.store).unwrap();
    fs::copy(&source, ingest.staged_path(&variant.relative_path).unwrap()).unwrap();
    let install = variant.finalize_install(ingest).unwrap();
    assert!(install.install_path.is_dir());
    assert!(fixture.store.resolve(&variant.checkpoint_id()).is_ok());

    assert!(variant.remove(&fixture.store).unwrap());

    // SceneWorks-owned state is gone...
    assert!(!install.install_path.exists());
    assert!(fixture.store.resolve(&variant.checkpoint_id()).is_err());
    assert!(fixture.store.record(&variant.checkpoint_id()).is_err());
    // ...and the external source is untouched, still linked, still byte-identical.
    assert_eq!(fs::read(&source).unwrap(), before);
    assert!(fixture.store.resolve(&linked.checkpoint_id).is_ok());
    assert_eq!(fixture.store.inventory().unwrap().records.len(), 1);

    // Removal is idempotent, and re-installing the same variant is allowed again afterwards: the
    // install id it holds is released with it. (`cancel` reports the cancellation as its Err, so
    // an abandoned session reads as a decision rather than a fall-through.)
    assert!(!variant.remove(&fixture.store).unwrap());
    let reopened = variant
        .begin_install(&fixture.store)
        .expect("the install id is free once the install is gone");
    assert!(matches!(
        reopened.cancel(),
        Err(ManagedIngestError::Cancelled { .. })
    ));
}

// ---- AC3: no conversion, verified provenance ---------------------------------------------------

/// Provenance records the pinned revision and the exact file, and nothing that could re-point them.
#[test]
fn provenance_records_the_pinned_revision_and_file() {
    for variant in managed_nvfp4_variants() {
        let provenance = variant.provenance();
        assert_eq!(provenance.source, MANAGED_VARIANT_PROVENANCE_SOURCE);
        assert_eq!(
            provenance.reference.as_deref(),
            Some(format!("{}@{}", variant.repo, variant.revision).as_str())
        );
        assert_eq!(
            provenance.version_id.as_deref(),
            Some(variant.revision.as_str())
        );
        assert_eq!(
            provenance.file_id.as_deref(),
            Some(variant.repo_file.as_str())
        );
        // Public, ungated repos: no stored credential authorizes them, so none is claimed.
        assert_eq!(provenance.credential_host, None);
        let url = provenance.url.clone().expect("a pinned artifact has a URL");
        assert!(
            url.contains(&variant.revision),
            "{url} must pin the revision"
        );
        assert!(!url.contains('@'), "{url} must carry no credential");
        provenance.validate().unwrap();
    }
}

/// The pinned checksum is VERIFIED against the bytes that arrive, and a mismatch produces no
/// install at all — not a partial one, and never a silent re-quantization of what did arrive.
#[test]
fn a_checksum_mismatch_against_the_pin_produces_no_install() {
    let fixture = fixture("checksum");
    let registered = registered(KREA_VARIANT);
    let (marker, descriptors) = fixture_shape(KREA_VARIANT);

    let pinned = fixture.library_dir.join("pinned.safetensors");
    write_nvfp4_file(&pinned, marker, descriptors, 0x11);
    let variant = variant_over(registered, &pinned);

    // Different bytes, same shape — exactly what a substituted or re-quantized artifact looks like.
    let substituted = fixture.library_dir.join("substituted.safetensors");
    write_nvfp4_file(&substituted, marker, descriptors, 0x22);
    assert_ne!(sha256_of(&substituted), variant.sha256);

    let ingest = variant.begin_install(&fixture.store).unwrap();
    fs::copy(
        &substituted,
        ingest.staged_path(&variant.relative_path).unwrap(),
    )
    .unwrap();
    let error = variant
        .finalize_install(ingest)
        .expect_err("bytes that are not the pinned artifact must not install");
    assert!(
        matches!(error, ManagedIngestError::HashMismatch { .. }),
        "expected a hash mismatch, got {error:?}"
    );

    assert!(!fixture
        .store
        .install_dir(&variant.variant_id)
        .unwrap()
        .exists());
    assert!(fixture.store.record(&variant.checkpoint_id()).is_err());
    assert_eq!(fixture.store.inventory().unwrap().records.len(), 0);
}

// ---- E2: explicit selection only ---------------------------------------------------------------

/// The registry offers exactly one lookup, and it needs an id.
///
/// This is the structural half of "auto-select must remain impossible": there is no function here
/// that takes a model, a family, a backend, or a host capability and returns a variant, so a tier
/// chooser has no entry point to call. The behavioural half is below.
#[test]
fn nothing_maps_a_model_onto_a_variant() {
    assert!(managed_nvfp4_variant("nvfp4-krea-2-turbo").is_some());
    for stranger in [
        "krea_2_turbo",
        "krea_2",
        "flux2",
        "nvfp4",
        "q4",
        "",
        "NVFP4-KREA-2-TURBO",
    ] {
        assert!(
            managed_nvfp4_variant(stranger).is_none(),
            "{stranger:?} must not resolve to a variant"
        );
    }
}

/// A generation request against an installed Krea variant is admitted ONLY when it names `nvfp4`.
///
/// The legacy MLX bit count is refused rather than coerced, and so is any other named tier: an
/// NVFP4 checkpoint is never served as q4, and `mlxQuantize: 4` never reaches it.
#[test]
fn an_installed_variant_is_reachable_only_by_naming_nvfp4() {
    let variant = registered(KREA_VARIANT);
    let admitted = generate_request(variant, json!({"quantTier": "nvfp4"}));
    assert!(
        imported_image_request_provider_eligible(variant.imported_model_id(), &admitted, "candle"),
        "an explicit NVFP4 request must be admitted"
    );

    // A request that names NO tier is admitted, and that is not auto-selection: the checkpoint is
    // NVFP4 bytes and there is no other tier of it to choose. What must never happen is a DIFFERENT
    // tier reaching it, or the legacy bit count being coerced into one.
    assert!(imported_image_request_provider_eligible(
        variant.imported_model_id(),
        &generate_request(variant, json!({})),
        "candle"
    ));

    for refused in [
        json!({"mlxQuantize": 4}),
        json!({"mlxQuantize": 8}),
        json!({"quantTier": "q4"}),
        json!({"quantTier": "q8"}),
        json!({"quantTier": "bf16"}),
        json!({"quantTier": "nvfp4", "mlxQuantize": 4}),
    ] {
        assert!(
            !imported_image_request_provider_eligible(
                variant.imported_model_id(),
                &generate_request(variant, refused.clone()),
                "candle"
            ),
            "{refused} must not reach an NVFP4 checkpoint"
        );
    }

    // MLX has no consumer for packed E2M1 weights, so the backend never wins the family route.
    assert!(!imported_image_request_provider_eligible(
        variant.imported_model_id(),
        &admitted,
        "mlx"
    ));
}

/// FLUX.2 Klein admission is DATA, not code: the `flux2` + `transformer_file` provider row arrives
/// in the checked-in engine dump when the pin advances to sc-21485's registration (it is not in
/// the inference commit SceneWorks pins today, which only has `flux2` + `comfy_ui_tree`).
///
/// Asserted under BOTH pins, so the terminal re-pin flips it without touching a line here: with no
/// row the gate must fail closed rather than half-admit, and with one it must be NVFP4-only.
///
/// **The two states are written out as branches on purpose.** A single
/// `assert_eq!(eligible, route_registered)` is satisfied by `false == false` at today's pin, which
/// makes it a tautology that proves nothing about either state; each arm below asserts something
/// the other cannot. At today's pin (inference `1caa686`, whose `flux2` provider rows are
/// `comfy_ui_tree` only) the closed arm runs. When the epic's terminal re-pin lands sc-21485's
/// `flux2` + `transformer_file` registration, the checked-in engine dump this reads changes and
/// the open arm runs instead — no edit here, but this test IS re-run against the new dump, and it
/// is the assertion that catches a Klein row that admits more than NVFP4.
#[test]
fn klein_admission_follows_the_engine_facts_and_never_aliases_q4() {
    let variant = registered(KLEIN_VARIANT);
    let model = variant.imported_model_id();
    let named = generate_request(variant, json!({"quantTier": "nvfp4"}));
    let route_registered = imported_provider_routes("candle", &variant.family)
        .any(|route| route.source == MANAGED_VARIANT_SOURCE_SHAPE);
    let admits_named = imported_image_request_provider_eligible(model, &named, "candle");

    if route_registered {
        // Open state: the engine serves single-file Klein, so an explicitly NVFP4 request is
        // admitted, and a request naming no tier at all is admitted too (the bytes are NVFP4 and
        // there is no other tier of this checkpoint to pick between).
        assert!(
            admits_named,
            "a registered transformer_file route must admit the explicitly NVFP4 request"
        );
        assert!(
            imported_image_request_provider_eligible(
                model,
                &generate_request(variant, json!({})),
                "candle"
            ),
            "an untiered request against NVFP4-only bytes is not auto-selection"
        );
    } else {
        // Closed state: no route, so the gate fails CLOSED — not half-admitted, and not admitted
        // on the strength of the manifest entry's own `importQuantFormat`.
        assert!(
            !admits_named,
            "with no transformer_file route the gate must refuse rather than half-admit"
        );
        assert!(
            !imported_image_request_provider_eligible(
                model,
                &generate_request(variant, json!({})),
                "candle"
            ),
            "an untiered request must not slip past a gate that has no route to serve it"
        );
    }
    for refused in [json!({"quantTier": "q4"}), json!({"mlxQuantize": 4})] {
        assert!(
            !imported_image_request_provider_eligible(
                model,
                &generate_request(variant, refused.clone()),
                "candle"
            ),
            "{refused} must never reach the Klein NVFP4 checkpoint, at either pin"
        );
    }
}

// ---- mutations ---------------------------------------------------------------------------------

/// Redirecting a registered entry to q4 is refused at registration, so it can never be offered,
/// installed, or recorded as a q4 install of NVFP4 bytes.
#[test]
fn redirecting_a_variant_to_q4_is_refused() {
    for registered in managed_nvfp4_variants() {
        for tier in ["q4", "q8", "bf16", "4", "nvfp4-v1", ""] {
            let mut mutated = registered.clone();
            mutated.quant_tier = tier.to_owned();
            let error = mutated
                .validate()
                .expect_err("a variant is only ever the NVFP4 tier");
            assert!(
                error.reason().contains("never represented as q4"),
                "unexpected reason for {tier:?}: {error}"
            );
            assert!(mutated.managed_locator().is_err());
        }

        let mut miscoded = registered.clone();
        miscoded.source_codec = "int8-per-row-v1".to_owned();
        assert!(miscoded.validate().is_err());
    }
}

/// Changing a pin is refused, and a pin that merely LOOKS well formed still cannot claim an
/// install of different bytes.
#[test]
fn changing_the_pinned_provenance_is_refused() {
    let registered = registered(KREA_VARIANT);

    for revision in ["main", "952f49d4", "", &"z".repeat(40)] {
        let mut mutated = registered.clone();
        mutated.revision = revision.to_owned();
        assert!(
            mutated.validate().is_err(),
            "{revision:?} is not a pinned commit"
        );
    }
    for checksum in ["", "deadbeef", &"F".repeat(64), &"g".repeat(64)] {
        let mut mutated = registered.clone();
        mutated.sha256 = checksum.to_owned();
        assert!(
            mutated.validate().is_err(),
            "{checksum:?} is not a pinned SHA-256"
        );
    }
    for repo in ["Comfy-Org", "/Krea-2", "Comfy-Org/", ""] {
        let mut mutated = registered.clone();
        mutated.repo = repo.to_owned();
        assert!(mutated.validate().is_err(), "{repo:?} is not owner/name");
    }

    // A well-formed but DIFFERENT pin: the registration validates, and the install it produced
    // still refuses to answer to it.
    let fixture = fixture("repin");
    let (marker, descriptors) = fixture_shape(KREA_VARIANT);
    let source = fixture.library_dir.join(&registered.relative_path);
    write_nvfp4_file(&source, marker, descriptors, 0x33);
    let variant = variant_over(registered, &source);

    let ingest = variant.begin_install(&fixture.store).unwrap();
    fs::copy(&source, ingest.staged_path(&variant.relative_path).unwrap()).unwrap();
    let install = variant.finalize_install(ingest).unwrap();
    variant.confirm_installed(&install).unwrap();

    let mut repinned = variant.clone();
    repinned.sha256 = "0".repeat(64);
    repinned.validate().expect("still a well-formed pin");
    let error = repinned
        .confirm_installed(&install)
        .expect_err("a re-pinned variant must not claim bytes it did not pin");
    assert!(error.reason().contains("not the pinned"), "{error}");

    // `size_bytes` is CHECKED, not decorative: it is the download estimate a client renders, so a
    // registration whose size does not describe the artifact is a defect, not a rounding.
    //
    // Failing mutation: delete the `observed != self.size_bytes` branch in `confirm_installed`.
    let mut resized = variant.clone();
    resized.size_bytes = variant.size_bytes + 1;
    resized
        .validate()
        .expect("a non-zero size is still a well-formed registration");
    let error = resized
        .confirm_installed(&install)
        .expect_err("a variant that mis-states its download size must not confirm");
    assert!(
        error.reason().contains("not the pinned"),
        "unexpected size-mismatch reason: {error}"
    );
    // ...and the honest registration still confirms, so the check is a comparison and not a
    // blanket refusal.
    variant.confirm_installed(&install).unwrap();
}

/// The linked locator is derived from the pin, so a mutated pin cannot be laundered through the
/// linked side to fake plan equality.
#[test]
fn linked_and_managed_locators_share_the_pin_they_are_derived_from() {
    let mut mutated = registered(KLEIN_VARIANT).clone();
    mutated.sha256 = "1".repeat(64);
    let managed = mutated.managed_locator().unwrap();
    let linked = mutated.linked_locator("root", "a.safetensors").unwrap();
    assert_eq!(
        managed.semantic_identity().unwrap(),
        linked.semantic_identity().unwrap()
    );
    assert_ne!(
        managed.semantic_identity().unwrap(),
        registered(KLEIN_VARIANT)
            .managed_locator()
            .unwrap()
            .semantic_identity()
            .unwrap(),
        "a different pin is a different checkpoint"
    );
    match linked {
        SourceLocatorV1::Linked { fingerprint, .. } => assert_eq!(fingerprint, mutated.sha256),
        other => panic!("expected a linked locator, got {other:?}"),
    }
}
