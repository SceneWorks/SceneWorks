//! Persisted checkpoint plans + approved roots (sc-20634): compile determinism, locator
//! independence, and every typed refusal a resolve can raise before a loader exists.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use tempfile::TempDir;

use sceneworks_core::checkpoint_import::{
    CheckpointCatalogRecordV1, CheckpointInventoryV1, ManagedProvenanceV1, SourceLocatorV1,
};
use sceneworks_core::checkpoint_inspector::{
    inspect_checkpoint, CheckpointDiagnosticCodeV1, CheckpointInspectionRequestV1,
};
use sceneworks_core::checkpoint_plan_store::{
    linked_checkpoint_id, managed_checkpoint_id, CheckpointPlanError, CheckpointPlanStore,
    APPROVED_ROOTS_FILE, BINDINGS_DIR, CHECKPOINTS_DIR, INVENTORY_FILE, PLANS_DIR, PLAN_ID_PREFIX,
};

fn fixture_dir(label: &str) -> TempDir {
    tempfile::Builder::new()
        .prefix(&format!("plan-store-{label}-{}-", std::process::id()))
        .tempdir()
        .unwrap()
}

/// A minimal single-file Krea 2 native DiT: the `txtfusion.` marker the family detector keys on,
/// every tensor dense bf16, bytes deterministic.
fn write_krea_native_file(path: &Path, fill: u8) {
    write_safetensors(
        path,
        &[
            ("model.diffusion_model.txtfusion.projector.weight", "BF16"),
            ("model.diffusion_model.blocks.0.attn.wq.weight", "BF16"),
            ("model.diffusion_model.first.weight", "BF16"),
        ],
        fill,
    );
}

fn write_safetensors(path: &Path, entries: &[(&str, &str)], fill: u8) {
    let mut header = Map::new();
    let mut offset = 0_u64;
    for (name, dtype) in entries {
        let width = match *dtype {
            "F16" | "BF16" => 2,
            "F32" => 4,
            _ => 1,
        };
        header.insert(
            (*name).to_owned(),
            json!({"dtype": dtype, "shape": [1], "data_offsets": [offset, offset + width]}),
        );
        offset += width;
    }
    let encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.resize(bytes.len() + offset as usize, fill);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

struct Fixture {
    _data: TempDir,
    _library: TempDir,
    store: CheckpointPlanStore,
    data_dir: PathBuf,
    library_dir: PathBuf,
}

fn fixture(label: &str) -> Fixture {
    let data = fixture_dir(&format!("{label}-data"));
    let library = fixture_dir(&format!("{label}-library"));
    let data_dir = data.path().to_path_buf();
    let library_dir = fs::canonicalize(library.path()).unwrap();
    Fixture {
        store: CheckpointPlanStore::open(&data_dir),
        data_dir,
        library_dir,
        _data: data,
        _library: library,
    }
}

#[test]
fn approve_root_is_idempotent_and_derives_an_opaque_stable_id() {
    let fx = fixture("roots");
    let first = fx.store.approve_root(&fx.library_dir).unwrap();
    let second = fx.store.approve_root(&fx.library_dir).unwrap();
    assert_eq!(
        first, second,
        "re-approving the same directory is idempotent"
    );
    assert!(first.root_id.starts_with("root-"));
    assert_eq!(first.root_id.len(), 21);
    assert!(
        !first
            .root_id
            .contains(fx.library_dir.file_name().unwrap().to_str().unwrap()),
        "the root id is opaque: {}",
        first.root_id
    );
    assert_eq!(first.path, fx.library_dir);
    let on_disk: Value = serde_json::from_slice(
        &fs::read(fx.data_dir.join(CHECKPOINTS_DIR).join(APPROVED_ROOTS_FILE)).unwrap(),
    )
    .unwrap();
    assert_eq!(on_disk["schemaVersion"], 1);
    assert_eq!(on_disk["roots"].as_array().unwrap().len(), 1);
    assert_eq!(
        fx.store.approved_roots().unwrap().roots,
        vec![first.clone()]
    );

    assert_eq!(
        fx.store.resolve_root(&first.root_id).unwrap(),
        fx.library_dir
    );
    assert_eq!(
        fx.store.resolve_root("root-doesnotexist"),
        Err(CheckpointPlanError::UnknownRoot {
            root_id: "root-doesnotexist".to_owned()
        })
    );
    assert!(matches!(
        fx.store.approve_root(Path::new("relative/dir")),
        Err(CheckpointPlanError::RootNotApprovable { .. })
    ));
    assert!(matches!(
        fx.store.approve_root(&fx.library_dir.join("absent")),
        Err(CheckpointPlanError::RootNotApprovable { .. })
    ));

    // A second, distinct directory gets a distinct id.
    let other = fixture_dir("roots-other");
    let other_root = fx
        .store
        .approve_root(&fs::canonicalize(other.path()).unwrap())
        .unwrap();
    assert_ne!(other_root.root_id, first.root_id);
    assert_eq!(fx.store.approved_roots().unwrap().roots.len(), 2);

    // An approved root whose directory vanished is unavailable, not unknown.
    drop(other);
    assert!(matches!(
        fx.store.resolve_root(&other_root.root_id),
        Err(CheckpointPlanError::RootUnavailable { .. })
    ));
}

#[test]
fn compile_linked_persists_plan_record_and_bindings_deterministically() {
    let fx = fixture("compile");
    write_krea_native_file(
        &fx.library_dir.join("checkpoints/kreamania.safetensors"),
        0x5a,
    );
    let root = fx.store.approve_root(&fx.library_dir).unwrap();

    let first = fx
        .store
        .compile_linked(&root.root_id, "checkpoints/kreamania.safetensors")
        .unwrap();
    let second = fx
        .store
        .compile_linked(&root.root_id, "checkpoints/kreamania.safetensors")
        .unwrap();
    assert_eq!(first, second, "compiling unchanged bytes is deterministic");

    assert_eq!(
        first.checkpoint_id,
        linked_checkpoint_id(&root.root_id, "checkpoints/kreamania.safetensors")
    );
    assert_eq!(
        first.checkpoint_id,
        format!("linked/{}/checkpoints/kreamania.safetensors", root.root_id)
    );
    assert_eq!(first.plan.family, "krea_2");
    assert_eq!(first.plan.layers.len(), 1);
    let layer = &first.plan.layers[0];
    assert_eq!(layer.role, "transformer");
    assert_eq!(layer.target_path, "checkpoints/kreamania.safetensors");
    match &layer.source {
        SourceLocatorV1::Linked {
            root_id,
            relative_path,
            ..
        } => {
            assert_eq!(root_id, &root.root_id);
            assert_eq!(relative_path, "checkpoints/kreamania.safetensors");
        }
        other => panic!("linked compile must produce linked locators, got {other:?}"),
    }
    assert_eq!(first.record.checkpoint_id, first.checkpoint_id);
    assert_eq!(first.record.plan.plan_id, first.plan.plan_id);
    first.record.validate_loaded_plan(&first.plan).unwrap();

    let checkpoints = fx.data_dir.join(CHECKPOINTS_DIR);
    assert!(checkpoints
        .join(PLANS_DIR)
        .join(format!("{}.json", first.plan.plan_id))
        .is_file());
    assert!(checkpoints
        .join(BINDINGS_DIR)
        .join(format!("{}.json", first.plan.plan_id))
        .is_file());
    let inventory = fx.store.inventory().unwrap();
    assert_eq!(inventory.records, vec![first.record.clone()]);
    let raw_inventory: Value =
        serde_json::from_slice(&fs::read(checkpoints.join(INVENTORY_FILE)).unwrap()).unwrap();
    assert_eq!(raw_inventory["schemaVersion"], 1);
    assert!(
        raw_inventory["records"][0]["plan"].get("layers").is_none(),
        "catalog records carry a plan reference + summary, never inline layers"
    );
    let persisted_plan = fx
        .store
        .plan(&first.checkpoint_id, &first.plan.plan_id)
        .unwrap();
    assert_eq!(persisted_plan, first.plan);
    // No absolute path leaks into any persisted plan/record document.
    let library_text = fx.library_dir.to_string_lossy().into_owned();
    for file in [
        checkpoints.join(INVENTORY_FILE),
        checkpoints
            .join(PLANS_DIR)
            .join(format!("{}.json", first.plan.plan_id)),
    ] {
        let text = fs::read_to_string(&file).unwrap();
        assert!(
            !text.contains(&library_text),
            "{} embeds the library path",
            file.display()
        );
    }

    // Relative-path hygiene is refused before any inspection.
    for bad in [
        "../escape.safetensors",
        "/abs.safetensors",
        "",
        "a\\b.safetensors",
    ] {
        assert!(
            matches!(
                fx.store.compile_linked(&root.root_id, bad),
                Err(CheckpointPlanError::InvalidRelativePath { .. })
            ),
            "{bad:?}"
        );
    }
    assert!(matches!(
        fx.store
            .compile_linked("root-unknown", "checkpoints/kreamania.safetensors"),
        Err(CheckpointPlanError::UnknownRoot { .. })
    ));
}

#[test]
fn semantic_digest_is_locator_independent_but_source_binding_is_not() {
    let fx = fixture("digest");
    write_krea_native_file(&fx.library_dir.join("kreamania.safetensors"), 0x5a);
    let root = fx.store.approve_root(&fx.library_dir).unwrap();
    let linked = fx
        .store
        .compile_linked(&root.root_id, "kreamania.safetensors")
        .unwrap();

    // The managed twin: identical bytes under an app-owned install, under its OWN managed
    // checkpoint identity. Compiling it under the LINKED checkpoint id (as this test first did)
    // held constant the very thing the two ownership modes disagree about, and the inspector
    // derives its plan id from the checkpoint id — so the assertion below passed without the
    // digest actually being locator-independent (sc-20636).
    let install = fixture_dir("digest-managed");
    write_krea_native_file(&install.path().join("kreamania.safetensors"), 0x5a);
    let managed = inspect_checkpoint(
        &CheckpointInspectionRequestV1::managed(
            managed_checkpoint_id("install-7"),
            install.path(),
            "kreamania.safetensors",
            "install-7",
            ManagedProvenanceV1 {
                source: "civitai".to_owned(),
                reference: Some("model-version-1".to_owned()),
                ..ManagedProvenanceV1::default()
            },
        )
        .unwrap(),
    );
    assert!(managed.is_runnable(), "{:?}", managed.diagnostics);
    let managed_plan = &managed.plans[0];
    assert!(managed_plan
        .layers
        .iter()
        .all(|layer| matches!(layer.source, SourceLocatorV1::Managed { .. })));
    assert_eq!(
        managed_plan.semantic_digest().unwrap(),
        linked.plan.semantic_digest().unwrap(),
        "the semantic digest must not see root/install ids, paths, or provenance"
    );
    assert_ne!(
        managed_plan.source_binding_identity().unwrap(),
        linked.plan.source_binding_identity().unwrap(),
        "the source binding must see physical ownership"
    );

    // Different bytes at the same identity rotate the semantic digest.
    let other = fixture_dir("digest-other-bytes");
    write_krea_native_file(&other.path().join("kreamania.safetensors"), 0x11);
    let other_root = fx
        .store
        .approve_root(&fs::canonicalize(other.path()).unwrap())
        .unwrap();
    let other_plan = fx
        .store
        .compile_linked(&other_root.root_id, "kreamania.safetensors")
        .unwrap();
    assert_ne!(
        other_plan.plan.semantic_digest().unwrap(),
        linked.plan.semantic_digest().unwrap()
    );
}

#[test]
fn resolve_verifies_stamps_and_refuses_drifted_or_missing_sources() {
    let fx = fixture("resolve");
    let file = fx.library_dir.join("kreamania.safetensors");
    write_krea_native_file(&file, 0x5a);
    let root = fx.store.approve_root(&fx.library_dir).unwrap();
    let compiled = fx
        .store
        .compile_linked(&root.root_id, "kreamania.safetensors")
        .unwrap();

    let resolved = fx.store.resolve(&compiled.checkpoint_id).unwrap();
    assert_eq!(resolved.checkpoint_id, compiled.checkpoint_id);
    assert_eq!(resolved.plan, compiled.plan);
    assert_eq!(resolved.family(), "krea_2");
    assert_eq!(resolved.layers.len(), 1);
    assert_eq!(resolved.layers[0].path, file);
    assert!(
        !resolved.layers[0].rehashed,
        "an untouched source is accepted on its stamp"
    );
    assert_eq!(
        resolved
            .layers_with_role("transformer")
            .map(|layer| layer.path.clone())
            .collect::<Vec<_>>(),
        vec![file.clone()]
    );
    assert!(resolved.layers_with_role("vae").next().is_none());

    // Same bytes, new entry (a re-copy): the stamp differs, the bytes are re-hashed and accepted,
    // and the refreshed stamp makes the next resolve cheap again.
    let bytes = fs::read(&file).unwrap();
    let staging = fx.library_dir.join("kreamania.safetensors.staging");
    fs::write(&staging, &bytes).unwrap();
    fs::rename(&staging, &file).unwrap();
    let rehashed = fx.store.resolve(&compiled.checkpoint_id).unwrap();
    assert!(rehashed.layers[0].rehashed, "a new entry must be re-hashed");
    let again = fx.store.resolve(&compiled.checkpoint_id).unwrap();
    assert!(!again.layers[0].rehashed, "the stamp was refreshed");

    // Same size, different bytes (an in-place edit or a retargeted root): drift, refused.
    let mut mutated = bytes.clone();
    let last = mutated.len() - 1;
    mutated[last] ^= 0xff;
    fs::write(&file, &mutated).unwrap();
    match fx.store.resolve(&compiled.checkpoint_id) {
        Err(CheckpointPlanError::SourceDrifted {
            checkpoint_id,
            relative_path,
            expected_sha256,
            actual_sha256,
        }) => {
            assert_eq!(checkpoint_id, compiled.checkpoint_id);
            assert_eq!(relative_path, "kreamania.safetensors");
            assert_ne!(expected_sha256, actual_sha256);
            match &compiled.plan.layers[0].source {
                SourceLocatorV1::Linked { fingerprint, .. } => {
                    assert_eq!(&expected_sha256, fingerprint)
                }
                other => panic!("{other:?}"),
            }
        }
        other => panic!("drifted bytes must refuse with SourceDrifted, got {other:?}"),
    }
    let error = fx.store.resolve(&compiled.checkpoint_id).unwrap_err();
    assert_eq!(error.code(), "source-drifted");
    assert!(error
        .to_string()
        .starts_with("[checkpoint-plan:source-drifted]"));

    // Restoring the exact bytes is accepted again (the plan is about bytes, not timestamps).
    fs::write(&file, &bytes).unwrap();
    assert!(fx.store.resolve(&compiled.checkpoint_id).is_ok());

    // A missing source is typed as missing, not drifted.
    fs::remove_file(&file).unwrap();
    assert!(matches!(
        fx.store.resolve(&compiled.checkpoint_id),
        Err(CheckpointPlanError::SourceMissing { ref relative_path, .. })
            if relative_path == "kreamania.safetensors"
    ));

    // A retargeted root (the approved directory now points somewhere else) fails closed too:
    // the library directory itself disappearing is RootUnavailable.
    fs::remove_dir_all(&fx.library_dir).unwrap();
    assert!(matches!(
        fx.store.resolve(&compiled.checkpoint_id),
        Err(CheckpointPlanError::RootUnavailable { .. })
    ));
}

#[test]
fn resolve_refuses_tampered_or_missing_plan_documents() {
    let fx = fixture("tamper");
    write_krea_native_file(&fx.library_dir.join("kreamania.safetensors"), 0x5a);
    let root = fx.store.approve_root(&fx.library_dir).unwrap();
    let compiled = fx
        .store
        .compile_linked(&root.root_id, "kreamania.safetensors")
        .unwrap();
    let plan_path = fx
        .data_dir
        .join(CHECKPOINTS_DIR)
        .join(PLANS_DIR)
        .join(format!("{}.json", compiled.plan.plan_id));

    assert!(matches!(
        fx.store.resolve("linked/root-nope/x.safetensors"),
        Err(CheckpointPlanError::UnknownCheckpoint { .. })
    ));

    // Edit the persisted plan: swap the layer role. The record's semantic digest no longer matches.
    let original = fs::read_to_string(&plan_path).unwrap();
    let edited = original.replace("\"role\":\"transformer\"", "\"role\":\"vae\"");
    assert_ne!(
        original, edited,
        "fixture must actually edit the plan document"
    );
    fs::write(&plan_path, edited).unwrap();
    match fx.store.resolve(&compiled.checkpoint_id) {
        Err(CheckpointPlanError::PlanTampered { checkpoint_id, .. }) => {
            assert_eq!(checkpoint_id, compiled.checkpoint_id)
        }
        other => panic!("an edited plan must refuse as tampered, got {other:?}"),
    }

    // A plan document that is not valid v1 JSON is corrupt, not silently re-derived.
    fs::write(&plan_path, "{not json").unwrap();
    assert!(matches!(
        fx.store.resolve(&compiled.checkpoint_id),
        Err(CheckpointPlanError::Corrupt { .. })
    ));

    fs::remove_file(&plan_path).unwrap();
    assert!(matches!(
        fx.store.resolve(&compiled.checkpoint_id),
        Err(CheckpointPlanError::MissingPlan { ref plan_id, .. }) if *plan_id == compiled.plan.plan_id
    ));

    // Recompiling restores a usable plan.
    let recompiled = fx
        .store
        .compile_linked(&root.root_id, "kreamania.safetensors")
        .unwrap();
    assert_eq!(recompiled, compiled);
    assert!(fx.store.resolve(&compiled.checkpoint_id).is_ok());
}

#[test]
fn unknown_families_and_malformed_sources_refuse_at_compile_with_typed_diagnostics() {
    let fx = fixture("unknown");
    let root = fx.store.approve_root(&fx.library_dir).unwrap();

    // A well-formed safetensors nobody recognises: no family evidence.
    write_safetensors(
        &fx.library_dir.join("mystery.safetensors"),
        &[("foo.weight", "BF16"), ("bar.bias", "BF16")],
        0x01,
    );
    match fx
        .store
        .compile_linked(&root.root_id, "mystery.safetensors")
    {
        Err(CheckpointPlanError::UnrunnableSource {
            checkpoint_id,
            diagnostics,
        }) => {
            assert_eq!(
                checkpoint_id,
                linked_checkpoint_id(&root.root_id, "mystery.safetensors")
            );
            assert!(
                diagnostics.iter().any(|d| d.code
                    == CheckpointDiagnosticCodeV1::MissingFamilyEvidence
                    || d.code == CheckpointDiagnosticCodeV1::AmbiguousComponentRole),
                "{diagnostics:?}"
            );
        }
        other => panic!("unknown family must refuse with typed diagnostics, got {other:?}"),
    }
    assert!(
        fx.store.inventory().unwrap().records.is_empty(),
        "a refused compile persists nothing"
    );

    // A truncated container refuses with the inspector's container diagnostic.
    let truncated = fx.library_dir.join("truncated.safetensors");
    write_krea_native_file(&truncated, 0x5a);
    let bytes = fs::read(&truncated).unwrap();
    fs::write(&truncated, &bytes[..bytes.len() - 3]).unwrap();
    match fx
        .store
        .compile_linked(&root.root_id, "truncated.safetensors")
    {
        Err(CheckpointPlanError::UnrunnableSource { diagnostics, .. }) => {
            assert!(!diagnostics.is_empty());
            let error = CheckpointPlanError::UnrunnableSource {
                checkpoint_id: "x".to_owned(),
                diagnostics,
            };
            assert_eq!(error.code(), "unrunnable-source");
            assert!(error.to_string().contains("truncated.safetensors"));
        }
        other => panic!("truncated source must refuse, got {other:?}"),
    }

    // A path that does not exist under the root.
    assert!(matches!(
        fx.store.compile_linked(&root.root_id, "absent.safetensors"),
        Err(CheckpointPlanError::UnrunnableSource { .. })
    ));
    assert!(fx.store.inventory().unwrap().records.is_empty());
}

#[test]
fn invalidate_and_recompile_manage_plan_documents_without_touching_the_source() {
    let fx = fixture("lifecycle");
    let file = fx.library_dir.join("kreamania.safetensors");
    write_krea_native_file(&file, 0x5a);
    let root = fx.store.approve_root(&fx.library_dir).unwrap();
    let first = fx
        .store
        .compile_linked(&root.root_id, "kreamania.safetensors")
        .unwrap();
    let checkpoints = fx.data_dir.join(CHECKPOINTS_DIR);
    let plan_file = |plan_id: &str| checkpoints.join(PLANS_DIR).join(format!("{plan_id}.json"));
    let bindings_file = |plan_id: &str| {
        checkpoints
            .join(BINDINGS_DIR)
            .join(format!("{plan_id}.json"))
    };

    // New bytes at the same identity: the plan id rotates and the old documents are dropped.
    write_krea_native_file(&file, 0x22);
    let second = fx
        .store
        .compile_linked(&root.root_id, "kreamania.safetensors")
        .unwrap();
    assert_eq!(second.checkpoint_id, first.checkpoint_id);
    assert_ne!(second.plan.plan_id, first.plan.plan_id);
    assert!(!plan_file(&first.plan.plan_id).exists());
    assert!(!bindings_file(&first.plan.plan_id).exists());
    assert!(plan_file(&second.plan.plan_id).is_file());
    assert_eq!(fx.store.inventory().unwrap().records.len(), 1);
    assert_eq!(
        fx.store.record(&first.checkpoint_id).unwrap().plan.plan_id,
        second.plan.plan_id
    );

    assert!(fx.store.invalidate(&first.checkpoint_id).unwrap());
    assert!(!fx.store.invalidate(&first.checkpoint_id).unwrap());
    assert!(fx.store.inventory().unwrap().records.is_empty());
    assert!(!plan_file(&second.plan.plan_id).exists());
    assert!(!bindings_file(&second.plan.plan_id).exists());
    assert!(
        file.is_file(),
        "invalidation never deletes or modifies the linked source"
    );
    assert!(matches!(
        fx.store.resolve(&first.checkpoint_id),
        Err(CheckpointPlanError::UnknownCheckpoint { .. })
    ));
}

/// A symlink planted inside an approved root after the plan compiled must not smuggle a file from
/// outside the root into a load (sc-20634 review). The decoy carries the SAME bytes as the planned
/// source, so the byte-fingerprint check cannot catch it: only path confinement can. The inspector
/// applies the same confinement at discovery time, so planting the link first refuses at compile.
#[test]
fn resolve_refuses_a_layer_whose_symlink_escapes_the_approved_root() {
    let fx = fixture("escape");
    let file = fx.library_dir.join("kreamania.safetensors");
    write_krea_native_file(&file, 0x5a);
    let root = fx.store.approve_root(&fx.library_dir).unwrap();
    let compiled = fx
        .store
        .compile_linked(&root.root_id, "kreamania.safetensors")
        .unwrap();
    assert!(fx.store.resolve(&compiled.checkpoint_id).is_ok());

    // Outside the root, byte-for-byte identical: a drift check alone would accept this.
    let outside = fixture_dir("escape-outside");
    let decoy = fs::canonicalize(outside.path())
        .unwrap()
        .join("kreamania.safetensors");
    write_krea_native_file(&decoy, 0x5a);
    assert_eq!(fs::read(&decoy).unwrap(), fs::read(&file).unwrap());

    fs::remove_file(&file).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(&decoy, &file).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(&decoy, &file).unwrap();

    match fx.store.resolve(&compiled.checkpoint_id) {
        Err(CheckpointPlanError::PathEscapesRoot {
            ref checkpoint_id,
            ref relative_path,
            ref resolved_path,
            ref root_path,
        }) => {
            assert_eq!(checkpoint_id, &compiled.checkpoint_id);
            assert_eq!(relative_path, "kreamania.safetensors");
            assert_eq!(resolved_path, &decoy);
            assert_eq!(root_path, &fx.library_dir);
        }
        other => panic!("an escaping symlink must refuse with PathEscapesRoot, got {other:?}"),
    }
    let error = fx.store.resolve(&compiled.checkpoint_id).unwrap_err();
    assert_eq!(error.code(), "path-escapes-root");
    assert!(error
        .to_string()
        .starts_with("[checkpoint-plan:path-escapes-root]"));

    // Planted BEFORE the compile, the inspector's own confinement refuses first, so an escaping
    // source can never reach a persisted plan in the first place.
    let fresh = fixture("escape-compile");
    let planted = fresh.library_dir.join("planted.safetensors");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&decoy, &planted).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(&decoy, &planted).unwrap();
    let fresh_root = fresh.store.approve_root(&fresh.library_dir).unwrap();
    let error = fresh
        .store
        .compile_linked(&fresh_root.root_id, "planted.safetensors")
        .unwrap_err();
    match &error {
        CheckpointPlanError::UnrunnableSource { diagnostics, .. } => assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == CheckpointDiagnosticCodeV1::PathEscapesRoot),
            "{diagnostics:?}"
        ),
        other => panic!("compiling an escaping source must refuse, got {other:?}"),
    }
    assert!(
        fresh.store.inventory().unwrap().records.is_empty(),
        "a refused compile persists nothing"
    );
}

/// `plan_id` is read back from a user-writable `inventory.json` and then interpolated into a
/// filename, so a crafted traversal id would make `plan()` read and `invalidate()` delete a
/// document outside the store. Every path builder validates the id first (sc-20634 review).
#[test]
fn a_traversal_plan_id_can_neither_read_nor_delete_outside_the_store() {
    let fx = fixture("plan-id");
    let file = fx.library_dir.join("kreamania.safetensors");
    write_krea_native_file(&file, 0x5a);
    let root = fx.store.approve_root(&fx.library_dir).unwrap();
    let compiled = fx
        .store
        .compile_linked(&root.root_id, "kreamania.safetensors")
        .unwrap();
    assert!(
        compiled.plan.plan_id.starts_with(PLAN_ID_PREFIX),
        "the inspector's shape is what the validator pins: {}",
        compiled.plan.plan_id
    );

    // A file the store must never be able to address, one level above `<data>/checkpoints/plans/`.
    let victim = fx.data_dir.join("victim.json");
    fs::write(&victim, b"{\"keep\":true}").unwrap();
    let traversal = "../../victim";

    let error = fx
        .store
        .plan(&compiled.checkpoint_id, traversal)
        .unwrap_err();
    assert!(
        matches!(error, CheckpointPlanError::InvalidPlanId { ref plan_id, .. } if plan_id == traversal),
        "{error:?}"
    );
    assert_eq!(error.code(), "invalid-plan-id");

    // The same id arriving through a tampered inventory record cannot delete the victim either.
    let inventory_path = fx.data_dir.join(CHECKPOINTS_DIR).join(INVENTORY_FILE);
    let mut inventory: Value = serde_json::from_slice(&fs::read(&inventory_path).unwrap()).unwrap();
    inventory["records"][0]["plan"]["planId"] = json!(traversal);
    fs::write(&inventory_path, serde_json::to_vec(&inventory).unwrap()).unwrap();

    let error = fx.store.resolve(&compiled.checkpoint_id).unwrap_err();
    assert_eq!(error.code(), "invalid-plan-id", "{error}");
    let error = fx.store.invalidate(&compiled.checkpoint_id).unwrap_err();
    assert_eq!(error.code(), "invalid-plan-id", "{error}");
    assert!(
        victim.is_file(),
        "a traversal plan id must not delete a document outside the store"
    );

    for rejected in [
        "",
        "checkpoint-plan-",
        "checkpoint-plan-../x",
        "checkpoint-plan-ZZZZ",
        "checkpoint-plan-abc/def",
        "plans",
    ] {
        assert!(
            fx.store.plan(&compiled.checkpoint_id, rejected).is_err(),
            "plan id {rejected:?} must be rejected"
        );
    }
}

/// `derive_root_id` truncates its digest, so a matching id is not proof of a matching directory.
/// Approving a second directory that collides must refuse rather than silently hand back the first
/// directory's binding (sc-20634 review).
#[test]
fn approving_a_colliding_root_id_refuses_instead_of_rebinding() {
    let fx = fixture("collision");
    let first = fx.store.approve_root(&fx.library_dir).unwrap();

    // Forge the collision the truncated id makes possible: the same id bound to another directory.
    let other = fixture_dir("collision-other");
    let other_dir = fs::canonicalize(other.path()).unwrap();
    let roots_path = fx.data_dir.join(CHECKPOINTS_DIR).join(APPROVED_ROOTS_FILE);
    let mut roots: Value = serde_json::from_slice(&fs::read(&roots_path).unwrap()).unwrap();
    roots["roots"][0]["path"] = json!(other_dir.to_str().unwrap());
    fs::write(&roots_path, serde_json::to_vec(&roots).unwrap()).unwrap();

    match fx.store.approve_root(&fx.library_dir) {
        Err(CheckpointPlanError::RootIdCollision {
            ref root_id,
            ref existing_path,
            ref path,
        }) => {
            assert_eq!(root_id, &first.root_id);
            assert_eq!(existing_path, &other_dir);
            assert_eq!(path, &fx.library_dir);
        }
        other => panic!("a colliding root id must refuse, got {other:?}"),
    }

    // The refusal is specific to the collision, not a blanket failure: `other_dir` derives its own
    // id and approves normally, and the forged entry is still bound to the directory it names.
    let other_root = fx.store.approve_root(&other_dir).unwrap();
    assert_ne!(other_root.root_id, first.root_id);
    assert_eq!(other_root.path, other_dir);
    assert_eq!(
        fx.store.resolve_root(&first.root_id).unwrap(),
        other_dir,
        "the forged binding still resolves to the directory it names; approve refused to rebind it"
    );
    assert!(
        fx.store.approve_root(&fx.library_dir).is_err(),
        "the collision is not cleared by approving the other directory"
    );
}

/// A root retargeted at a *different* library holding a same-named file is drift, not absence
/// (sc-20634 review): the previous coverage only removed the library dir, which is
/// `RootUnavailable`. Also pins the two remaining resolve refusals: a `Managed` locator is
/// `UnsupportedLocator`, and bindings belonging to another plan are `PlanTampered`.
#[test]
fn a_retargeted_root_managed_locator_and_foreign_bindings_each_refuse_typed() {
    let fx = fixture("retarget");
    let file = fx.library_dir.join("kreamania.safetensors");
    write_krea_native_file(&file, 0x5a);
    let root = fx.store.approve_root(&fx.library_dir).unwrap();
    let compiled = fx
        .store
        .compile_linked(&root.root_id, "kreamania.safetensors")
        .unwrap();
    assert!(fx.store.resolve(&compiled.checkpoint_id).is_ok());

    // ---- retargeted root: the id now names a second library with a same-named DIFFERENT file ----
    let second_library = fixture_dir("retarget-second");
    let second_dir = fs::canonicalize(second_library.path()).unwrap();
    write_krea_native_file(&second_dir.join("kreamania.safetensors"), 0x11);
    let roots_path = fx.data_dir.join(CHECKPOINTS_DIR).join(APPROVED_ROOTS_FILE);
    let mut roots: Value = serde_json::from_slice(&fs::read(&roots_path).unwrap()).unwrap();
    roots["roots"][0]["path"] = json!(second_dir.to_str().unwrap());
    fs::write(&roots_path, serde_json::to_vec(&roots).unwrap()).unwrap();

    match fx.store.resolve(&compiled.checkpoint_id) {
        Err(CheckpointPlanError::SourceDrifted {
            ref checkpoint_id,
            ref relative_path,
            ref expected_sha256,
            ref actual_sha256,
        }) => {
            assert_eq!(checkpoint_id, &compiled.checkpoint_id);
            assert_eq!(relative_path, "kreamania.safetensors");
            assert_ne!(expected_sha256, actual_sha256);
        }
        other => panic!("a retargeted root holding different bytes must be drift, got {other:?}"),
    }

    // Point the root back and confirm the plan is usable again: the refusal was about the bytes.
    roots["roots"][0]["path"] = json!(fx.library_dir.to_str().unwrap());
    fs::write(&roots_path, serde_json::to_vec(&roots).unwrap()).unwrap();
    assert!(fx.store.resolve(&compiled.checkpoint_id).is_ok());

    // ---- Managed locator filed under a LINKED identity: refused, never resolved ----
    // Managed locators became resolvable in sc-20636, but only under their OWN
    // `managed/<installId>` identity: the install a layer names must be the install the checkpoint
    // id names. A managed layer swapped into a linked checkpoint's record names no reachable
    // install, so it refuses rather than reading some other install's bytes.
    let install = fixture_dir("retarget-managed");
    write_krea_native_file(&install.path().join("kreamania.safetensors"), 0x5a);
    let managed = inspect_checkpoint(
        &CheckpointInspectionRequestV1::managed(
            compiled.checkpoint_id.clone(),
            install.path(),
            "kreamania.safetensors",
            "install-7",
            ManagedProvenanceV1 {
                source: "civitai".to_owned(),
                reference: None,
                ..ManagedProvenanceV1::default()
            },
        )
        .unwrap(),
    );
    assert!(managed.is_runnable(), "{:?}", managed.diagnostics);
    let managed_plan = managed.plans[0].clone();
    assert!(managed_plan
        .layers
        .iter()
        .all(|layer| matches!(layer.source, SourceLocatorV1::Managed { .. })));
    let managed_record =
        CheckpointCatalogRecordV1::from_plan(&compiled.checkpoint_id, &managed_plan).unwrap();
    let checkpoints = fx.data_dir.join(CHECKPOINTS_DIR);
    fs::write(
        checkpoints
            .join(PLANS_DIR)
            .join(format!("{}.json", managed_plan.plan_id)),
        managed_plan.canonical_json().unwrap(),
    )
    .unwrap();
    fs::write(
        checkpoints.join(INVENTORY_FILE),
        CheckpointInventoryV1::new(vec![managed_record])
            .unwrap()
            .canonical_json()
            .unwrap(),
    )
    .unwrap();
    match fx.store.resolve(&compiled.checkpoint_id) {
        Err(CheckpointPlanError::UnsupportedLocator {
            ref checkpoint_id,
            kind,
            ..
        }) => {
            assert_eq!(checkpoint_id, &compiled.checkpoint_id);
            assert_eq!(kind, "foreign-install managed");
        }
        other => panic!(
            "a managed locator under a linked identity must refuse as unsupported, got {other:?}"
        ),
    }

    // ---- foreign bindings: a bindings document naming another plan is tampering ----
    let fresh = fixture("retarget-bindings");
    write_krea_native_file(&fresh.library_dir.join("kreamania.safetensors"), 0x5a);
    let fresh_root = fresh.store.approve_root(&fresh.library_dir).unwrap();
    let fresh_plan = fresh
        .store
        .compile_linked(&fresh_root.root_id, "kreamania.safetensors")
        .unwrap();
    let bindings_path = fresh
        .data_dir
        .join(CHECKPOINTS_DIR)
        .join(BINDINGS_DIR)
        .join(format!("{}.json", fresh_plan.plan.plan_id));
    let mut bindings: Value = serde_json::from_slice(&fs::read(&bindings_path).unwrap()).unwrap();
    let foreign = format!("{PLAN_ID_PREFIX}{}", "0".repeat(64));
    assert_ne!(foreign, fresh_plan.plan.plan_id);
    bindings["planId"] = json!(foreign);
    fs::write(&bindings_path, serde_json::to_vec(&bindings).unwrap()).unwrap();
    match fresh.store.resolve(&fresh_plan.checkpoint_id) {
        Err(CheckpointPlanError::PlanTampered {
            ref checkpoint_id,
            ref reason,
        }) => {
            assert_eq!(checkpoint_id, &fresh_plan.checkpoint_id);
            assert!(
                reason.contains("source bindings belong to plan"),
                "{reason}"
            );
        }
        other => panic!("foreign bindings must refuse as tampered, got {other:?}"),
    }
}
