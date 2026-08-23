//! Persisted checkpoint plans + approved roots (sc-20634): compile determinism, locator
//! independence, and every typed refusal a resolve can raise before a loader exists.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::{json, Map, Value};
use tempfile::TempDir;

use sceneworks_core::checkpoint_import::{
    CheckpointCatalogRecordV1, CheckpointInventoryV1, ManagedProvenanceV1, SourceLocatorV1,
};
use sceneworks_core::checkpoint_inspector::{
    inspect_checkpoint, CheckpointDiagnosticCodeV1, CheckpointInspectionRequestV1,
};
use sceneworks_core::checkpoint_plan_store::{
    linked_checkpoint_id, CheckpointPlanError, CheckpointPlanStore, APPROVED_ROOTS_FILE,
    BINDINGS_DIR, CHECKPOINTS_DIR, INVENTORY_FILE, PLANS_DIR, PLAN_ID_PREFIX, STORE_LOCK_FILE,
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

    // The managed twin: identical bytes under an app-owned install, same checkpoint identity.
    let install = fixture_dir("digest-managed");
    write_krea_native_file(&install.path().join("kreamania.safetensors"), 0x5a);
    let managed = inspect_checkpoint(
        &CheckpointInspectionRequestV1::managed(
            linked.checkpoint_id.clone(),
            install.path(),
            "kreamania.safetensors",
            "install-7",
            ManagedProvenanceV1 {
                source: "civitai".to_owned(),
                reference: Some("model-version-1".to_owned()),
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

    // The refusal is specific to the collision, not a blanket failure: `other_dir` still approves,
    // and it approves to the binding it ALREADY has rather than to a second id (sc-20635 — one
    // directory is one root, which is also what makes `relink_root` safe: after a relink a
    // directory is bound to an id `derive_root_id` would not produce for it, and re-approving it
    // must return that binding).
    let other_root = fx.store.approve_root(&other_dir).unwrap();
    assert_eq!(other_root.root_id, first.root_id);
    assert_eq!(other_root.path, other_dir);
    assert_eq!(fx.store.approved_roots().unwrap().roots.len(), 1);
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

    // ---- Managed locator: a fully self-consistent managed checkpoint is not resolvable yet ----
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
            assert_eq!(kind, "managed");
        }
        other => panic!("a managed locator must refuse as unsupported, got {other:?}"),
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

// =============================================================================================
// Linked-library lifecycle and confinement probes (sc-20635, AC1 + AC2)
// =============================================================================================

// Used only by the `#[cfg(unix)]` symlink probe, and this target now COMPILES on the Windows lane
// (sc-20635 added it to desktop-windows.yml), where an ungated import is an unused-import warning.
#[cfg(unix)]
use sceneworks_core::checkpoint_inspector::CheckpointDiagnosticSeverityV1;
use sceneworks_core::checkpoint_plan_store::{
    parse_linked_checkpoint_id, validate_linked_relative_path, LinkedCheckpointStateV1,
};
use std::collections::BTreeMap;
use std::time::SystemTime;

/// Every file under `root` as `(size, mtime, bytes)`. A lifecycle action that leaves this equal
/// wrote nothing, deleted nothing and re-stamped nothing.
fn tree_snapshot(root: &Path) -> BTreeMap<PathBuf, (u64, SystemTime, Vec<u8>)> {
    let mut snapshot = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in fs::read_dir(&directory).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path).unwrap();
            if metadata.is_dir() {
                stack.push(path);
                continue;
            }
            snapshot.insert(
                path.strip_prefix(root).unwrap().to_path_buf(),
                (
                    metadata.len(),
                    metadata.modified().unwrap(),
                    fs::read(&path).unwrap_or_default(),
                ),
            );
        }
    }
    snapshot
}

/// A library holding two Krea checkpoints, one of them compiled.
fn library_fixture(label: &str) -> (Fixture, String, String) {
    let fx = fixture(label);
    write_krea_native_file(&fx.library_dir.join("alpha.safetensors"), 0x11);
    write_krea_native_file(&fx.library_dir.join("nested/beta.safetensors"), 0x22);
    let root = fx.store.approve_root(&fx.library_dir).unwrap();
    let compiled = fx
        .store
        .compile_linked(&root.root_id, "alpha.safetensors")
        .unwrap();
    (fx, root.root_id, compiled.checkpoint_id)
}

/// AC1: a candidate is VISIBLE from a header-only scan and UNSELECTABLE until a full-content
/// compile succeeds. Discovery never promotes anything on its own.
///
/// Failing mutation: make `scan_root` set `selectable: true` unconditionally, and the
/// "uncompiled candidates are not selectable" assertion goes red.
#[test]
fn a_scanned_candidate_is_visible_but_unselectable_until_it_compiles() {
    let (fx, root_id, compiled_id) = library_fixture("scan");
    let scan = fx.store.scan_root(&root_id).unwrap();
    assert!(scan.available);
    let paths: Vec<&str> = scan
        .candidates
        .iter()
        .map(|candidate| candidate.candidate.relative_path.as_str())
        .collect();
    assert_eq!(
        paths,
        vec!["alpha.safetensors", "nested/beta.safetensors"],
        "every weight file under the root is its own candidate, at its portable relative path"
    );
    // Header evidence is present without any full-content pass having run.
    assert!(
        scan.candidates[1].candidate.header_family.is_some(),
        "header-only discovery still classifies the candidate: {:?}",
        scan.candidates[1].candidate
    );

    let compiled = &scan.candidates[0];
    assert_eq!(compiled.checkpoint_id, compiled_id);
    assert!(compiled.selectable, "the compiled candidate is selectable");
    assert_eq!(
        compiled.status.as_ref().unwrap().state,
        LinkedCheckpointStateV1::Ready
    );

    let uncompiled = &scan.candidates[1];
    assert!(
        !uncompiled.selectable,
        "a candidate with no persisted plan is visible but NOT selectable"
    );
    assert_eq!(uncompiled.status, None);
    assert!(scan.unmatched.is_empty());
    assert!(
        scan.diagnostics.is_empty(),
        "a clean library scans clean: {:?}",
        scan.diagnostics
    );

    // Compiling it is what promotes it — nothing else does.
    fx.store
        .compile_linked(&root_id, "nested/beta.safetensors")
        .unwrap();
    let scan = fx.store.scan_root(&root_id).unwrap();
    assert!(scan.candidates.iter().all(|candidate| candidate.selectable));
}

/// AC1: rename, rescan, relocate, remove and relink all leave the library byte-for-byte and
/// mtime-for-mtime untouched. SceneWorks never modifies or deletes a linked file.
///
/// Failing mutation: make `remove_root` also `fs::remove_file` the checkpoint under the root (the
/// obvious "clean up what we imported" mistake) and the final snapshot comparison goes red.
#[test]
fn no_lifecycle_action_writes_or_deletes_anything_under_the_library() {
    let (fx, root_id, checkpoint_id) = library_fixture("no-writes");
    let before = tree_snapshot(&fx.library_dir);
    assert_eq!(before.len(), 2);

    fx.store.rename_root(&root_id, "My Comfy Library").unwrap();
    assert_eq!(
        tree_snapshot(&fx.library_dir),
        before,
        "rename wrote nothing"
    );

    fx.store.scan_root(&root_id).unwrap();
    assert_eq!(tree_snapshot(&fx.library_dir), before, "scan wrote nothing");

    fx.store.rescan_checkpoint(&checkpoint_id).unwrap();
    assert_eq!(
        tree_snapshot(&fx.library_dir),
        before,
        "rescan re-READS the bytes; it never writes them back"
    );

    // Relocate: move the whole library, then relink the root at its new home.
    let moved = fixture_dir("no-writes-moved");
    let moved_dir = fs::canonicalize(moved.path()).unwrap().join("library");
    fs::create_dir_all(&moved_dir).unwrap();
    for relative in before.keys() {
        let destination = moved_dir.join(relative);
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        fs::copy(fx.library_dir.join(relative), &destination).unwrap();
    }
    let moved_before = tree_snapshot(&moved_dir);
    let relinked = fx.store.relink_root(&root_id, &moved_dir).unwrap();
    assert_eq!(
        relinked.root_id, root_id,
        "a relink keeps the root id, so every compiled plan stays valid"
    );
    assert_eq!(relinked.path, moved_dir);
    assert_eq!(
        relinked.label, "My Comfy Library",
        "a relink does not disturb the label"
    );
    assert_eq!(
        fx.store.checkpoint_status(&checkpoint_id).unwrap().state,
        LinkedCheckpointStateV1::Ready,
        "the same bytes at the new location resolve"
    );
    assert_eq!(tree_snapshot(&moved_dir), moved_before);

    let removal = fx.store.remove_root(&root_id).unwrap();
    assert_eq!(removal.removed_checkpoints, vec![checkpoint_id.clone()]);
    assert!(fx.store.approved_roots().unwrap().roots.is_empty());
    assert_eq!(
        tree_snapshot(&moved_dir),
        moved_before,
        "removing a linked library forgets it; it never deletes it"
    );
    assert_eq!(
        tree_snapshot(&fx.library_dir),
        before,
        "and the original library is untouched too"
    );
    // What removal DID delete is store-owned only.
    assert!(matches!(
        fx.store.record(&checkpoint_id),
        Err(CheckpointPlanError::UnknownCheckpoint { .. })
    ));
}

/// AC1: an absent root is Needs Relink (the plans stay valid, the library moved); a source that is
/// gone or changed under a present root is Needs Rescan.
#[test]
fn missing_and_changed_content_surface_as_needs_relink_and_needs_rescan() {
    let (fx, root_id, checkpoint_id) = library_fixture("states");
    assert_eq!(
        fx.store.checkpoint_status(&checkpoint_id).unwrap().state,
        LinkedCheckpointStateV1::Ready
    );

    // Content changed in place.
    write_krea_native_file(&fx.library_dir.join("alpha.safetensors"), 0x33);
    let status = fx.store.checkpoint_status(&checkpoint_id).unwrap();
    assert_eq!(status.state, LinkedCheckpointStateV1::NeedsRescan);
    assert!(
        status
            .detail
            .as_deref()
            .unwrap()
            .contains("[checkpoint-plan:source-drifted]"),
        "{status:?}"
    );
    assert_eq!(status.root_id, root_id);
    assert_eq!(status.relative_path, "alpha.safetensors");
    assert!(!status.is_selectable());
    // The scan reports it against its candidate rather than hiding it.
    let scan = fx.store.scan_root(&root_id).unwrap();
    assert!(!scan.candidates[0].selectable);

    // A rescan recompiles from the bytes that are there now, keeping the identity.
    let recompiled = fx.store.rescan_checkpoint(&checkpoint_id).unwrap();
    assert_eq!(recompiled.checkpoint_id, checkpoint_id);
    assert_eq!(
        fx.store.checkpoint_status(&checkpoint_id).unwrap().state,
        LinkedCheckpointStateV1::Ready
    );

    // Source deleted under a present root.
    fs::remove_file(fx.library_dir.join("alpha.safetensors")).unwrap();
    let status = fx.store.checkpoint_status(&checkpoint_id).unwrap();
    assert_eq!(status.state, LinkedCheckpointStateV1::NeedsRescan);
    let scan = fx.store.scan_root(&root_id).unwrap();
    assert!(
        scan.candidates
            .iter()
            .all(|candidate| candidate.candidate.relative_path != "alpha.safetensors"),
        "the deleted checkpoint is no longer a candidate"
    );
    assert_eq!(
        scan.unmatched
            .iter()
            .map(|status| status.checkpoint_id.as_str())
            .collect::<Vec<_>>(),
        vec![checkpoint_id.as_str()],
        "but it does not silently vanish from the catalog either"
    );

    // Root gone entirely: Needs Relink, not Needs Rescan — the plans are fine, the library moved.
    let unavailable = fx.library_dir.join("gone");
    fs::create_dir(&unavailable).unwrap();
    fx.store.relink_root(&root_id, &unavailable).unwrap();
    fs::remove_dir(&unavailable).unwrap();
    let status = fx.store.checkpoint_status(&checkpoint_id).unwrap();
    assert_eq!(status.state, LinkedCheckpointStateV1::NeedsRelink);
    let scan = fx.store.scan_root(&root_id).unwrap();
    assert!(!scan.available);
    assert!(scan.candidates.is_empty());
    assert_eq!(
        scan.unmatched
            .iter()
            .map(|status| status.state)
            .collect::<Vec<_>>(),
        vec![LinkedCheckpointStateV1::NeedsRelink]
    );
}

/// AC1: a relink can retarget a root at a DIFFERENT library. That is allowed (the user may have
/// reorganised), but the plans are never trusted across it — the bytes are re-verified and a
/// mismatch is Needs Rescan rather than a silent load of the wrong weights.
#[test]
fn relinking_to_a_different_library_refuses_to_serve_the_old_plan() {
    let (fx, root_id, checkpoint_id) = library_fixture("retarget");
    let other = fixture_dir("retarget-other");
    let other_dir = fs::canonicalize(other.path()).unwrap();
    // Same path, different bytes: the shape a retarget-at-the-wrong-library attack needs.
    write_krea_native_file(&other_dir.join("alpha.safetensors"), 0x44);

    fx.store.relink_root(&root_id, &other_dir).unwrap();
    let status = fx.store.checkpoint_status(&checkpoint_id).unwrap();
    assert_eq!(status.state, LinkedCheckpointStateV1::NeedsRescan);
    assert!(
        status
            .detail
            .as_deref()
            .unwrap()
            .contains("[checkpoint-plan:source-drifted]"),
        "{status:?}"
    );
    assert!(matches!(
        fx.store.resolve(&checkpoint_id),
        Err(CheckpointPlanError::SourceDrifted { .. })
    ));
}

/// AC1: one directory is one root. Re-approving a relinked directory returns the EXISTING binding
/// rather than minting a second id for the same files, and relinking onto a directory another root
/// already owns refuses.
#[test]
fn one_directory_is_bound_to_exactly_one_root_id() {
    let (fx, root_id, _) = library_fixture("one-root");
    let second = fixture_dir("one-root-second");
    let second_dir = fs::canonicalize(second.path()).unwrap();

    fx.store.relink_root(&root_id, &second_dir).unwrap();
    let reapproved = fx.store.approve_root(&second_dir).unwrap();
    assert_eq!(
        reapproved.root_id, root_id,
        "re-approving a relinked directory returns its existing id"
    );
    assert_eq!(fx.store.approved_roots().unwrap().roots.len(), 1);

    // A second root, then a relink of it onto the first's directory, refuses.
    let third = fixture_dir("one-root-third");
    let third_dir = fs::canonicalize(third.path()).unwrap();
    let third_root = fx.store.approve_root(&third_dir).unwrap();
    assert_ne!(third_root.root_id, root_id);
    match fx.store.relink_root(&third_root.root_id, &second_dir) {
        Err(CheckpointPlanError::RootAlreadyApproved {
            existing_root_id, ..
        }) => assert_eq!(existing_root_id, root_id),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// AC1: labels are persisted, bounded and presentation-only.
#[test]
fn a_root_can_be_labelled_and_relabelled() {
    let fx = fixture("labels");
    let root = fx.store.approve_root(&fx.library_dir).unwrap();
    assert_eq!(root.label, "");
    assert_eq!(
        root.display_label(),
        fx.library_dir.file_name().unwrap().to_str().unwrap(),
        "an unlabelled root shows its own directory name"
    );

    let renamed = fx
        .store
        .rename_root(&root.root_id, "ComfyUI models")
        .unwrap();
    assert_eq!(renamed.display_label(), "ComfyUI models");
    assert_eq!(
        fx.store
            .approved_roots()
            .unwrap()
            .get(&root.root_id)
            .unwrap()
            .label,
        "ComfyUI models",
        "the label round-trips through the persisted document"
    );

    for bad in ["", "   ", "a\nb", &"x".repeat(129)] {
        assert!(
            matches!(
                fx.store.rename_root(&root.root_id, bad),
                Err(CheckpointPlanError::InvalidRootLabel { .. })
            ),
            "label {bad:?} must be refused"
        );
    }
    assert!(matches!(
        fx.store.rename_root("root-nope", "x"),
        Err(CheckpointPlanError::UnknownRoot { .. })
    ));
}

/// The persisted identity round-trips, and only a linked id parses.
#[test]
fn a_linked_checkpoint_id_parses_back_to_its_root_and_relative_path() {
    assert_eq!(
        parse_linked_checkpoint_id(&linked_checkpoint_id("root-abc", "a/b/c.safetensors")),
        Some(("root-abc", "a/b/c.safetensors"))
    );
    assert_eq!(parse_linked_checkpoint_id("managed/install-1/x"), None);
    assert_eq!(parse_linked_checkpoint_id("linked/root-abc"), None);
    assert_eq!(parse_linked_checkpoint_id("linked//x"), None);
    assert_eq!(parse_linked_checkpoint_id("linked/root-abc/"), None);
}

/// AC2: every traversal shape a caller can name is refused lexically, by the SAME validator the
/// API and the UI call, so no second copy of the rules can drift.
#[test]
fn traversal_shapes_are_refused_by_the_shared_relative_path_validator() {
    let (fx, root_id, _) = library_fixture("traversal");
    for bad in [
        "",
        "   ",
        "../outside.safetensors",
        "nested/../../outside.safetensors",
        "./alpha.safetensors",
        "/etc/passwd",
        "nested\\beta.safetensors",
    ] {
        assert!(
            matches!(
                validate_linked_relative_path(bad),
                Err(CheckpointPlanError::InvalidRelativePath { .. })
            ),
            "{bad:?} must not validate"
        );
        assert!(
            matches!(
                fx.store.compile_linked(&root_id, bad),
                Err(CheckpointPlanError::InvalidRelativePath { .. })
            ),
            "{bad:?} must not compile"
        );
    }
}

/// AC2 (macOS/Linux): a symlink INSIDE the library pointing outside it is lexically clean, so only
/// the canonicalizing confinement check can catch it. It must be caught at compile, and never
/// offered by a scan.
///
/// Failing mutation: delete the `!canonical_target.starts_with(&canonical_root)` refusal in
/// `confined_root_join` (and the inspector's matching discovery-time check) and this goes red.
#[cfg(unix)]
#[test]
fn a_symlink_out_of_the_library_fails_closed_everywhere() {
    let (fx, root_id, _) = library_fixture("symlink");
    let outside = fixture_dir("symlink-outside");
    let outside_dir = fs::canonicalize(outside.path()).unwrap();
    write_krea_native_file(&outside_dir.join("secret.safetensors"), 0x55);

    // A link inside the library that points at a file outside it.
    let link = fx.library_dir.join("escape.safetensors");
    std::os::unix::fs::symlink(outside_dir.join("secret.safetensors"), &link).unwrap();

    // Compile refuses.
    let error = fx
        .store
        .compile_linked(&root_id, "escape.safetensors")
        .unwrap_err();
    match &error {
        CheckpointPlanError::UnrunnableSource { diagnostics, .. } => assert!(
            diagnostics.iter().any(|diagnostic| matches!(
                diagnostic.code,
                CheckpointDiagnosticCodeV1::PathEscapesRoot
            )),
            "{diagnostics:?}"
        ),
        CheckpointPlanError::PathEscapesRoot { .. } => {}
        other => panic!("a symlink out of the library must fail closed, got {other:?}"),
    }

    // A scan never offers it, and says why.
    let scan = fx.store.scan_root(&root_id).unwrap();
    assert!(
        scan.candidates
            .iter()
            .all(|candidate| candidate.candidate.relative_path != "escape.safetensors"),
        "an escaping candidate is never offered: {:?}",
        scan.candidates
    );
    assert!(
        scan.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == CheckpointDiagnosticCodeV1::PathEscapesRoot
                && diagnostic.severity == CheckpointDiagnosticSeverityV1::Error
        }),
        "{:?}",
        scan.diagnostics
    );
}

/// AC2 (macOS/Linux): the REPLACEMENT probe — a checkpoint that compiled honestly is swapped for a
/// symlink pointing outside the library AFTER the plan was persisted. The fingerprint check alone
/// cannot see this (the outside bytes are identical); only re-confining the canonical path the
/// resolve is about to OPEN can.
#[cfg(unix)]
#[test]
fn replacing_a_compiled_source_with_an_escaping_link_fails_closed() {
    let (fx, _root_id, checkpoint_id) = library_fixture("replacement");
    let outside = fixture_dir("replacement-outside");
    let outside_dir = fs::canonicalize(outside.path()).unwrap();
    // Byte-IDENTICAL content outside the root: the fingerprint check alone would pass this, so
    // only the confinement check refuses it.
    write_krea_native_file(&outside_dir.join("alpha.safetensors"), 0x11);
    assert_eq!(
        fs::read(outside_dir.join("alpha.safetensors")).unwrap(),
        fs::read(fx.library_dir.join("alpha.safetensors")).unwrap()
    );

    let target = fx.library_dir.join("alpha.safetensors");
    fs::remove_file(&target).unwrap();
    std::os::unix::fs::symlink(outside_dir.join("alpha.safetensors"), &target).unwrap();

    match fx.store.resolve(&checkpoint_id) {
        Err(CheckpointPlanError::PathEscapesRoot {
            ref relative_path,
            ref resolved_path,
            ..
        }) => {
            assert_eq!(relative_path, "alpha.safetensors");
            assert!(resolved_path.starts_with(&outside_dir), "{resolved_path:?}");
        }
        other => panic!("a swapped-in escaping link must fail closed, got {other:?}"),
    }
    assert_eq!(
        fx.store.checkpoint_status(&checkpoint_id).unwrap().state,
        LinkedCheckpointStateV1::NeedsRescan,
        "and it surfaces as a lifecycle state the user can act on"
    );
    // Even a rescan refuses rather than adopting the outside bytes.
    assert!(fx.store.rescan_checkpoint(&checkpoint_id).is_err());
}

/// AC2 (macOS/Linux): a symlinked ROOT is resolved to its real directory once, at approval, so a
/// checkpoint reached through it is confined against the real directory rather than the link.
#[cfg(unix)]
#[test]
fn an_approved_root_is_bound_to_its_real_directory() {
    let fx = fixture("symlinked-root");
    write_krea_native_file(&fx.library_dir.join("alpha.safetensors"), 0x66);
    let alias = fixture_dir("symlinked-root-alias");
    let link = fs::canonicalize(alias.path()).unwrap().join("library-link");
    std::os::unix::fs::symlink(&fx.library_dir, &link).unwrap();

    let root = fx.store.approve_root(&link).unwrap();
    assert_eq!(
        root.path, fx.library_dir,
        "the approved root is the real directory, never the link"
    );
    assert_eq!(
        root.root_id,
        fx.store.approve_root(&fx.library_dir).unwrap().root_id,
        "reaching one directory through a link is not a second library"
    );
    fx.store
        .compile_linked(&root.root_id, "alpha.safetensors")
        .unwrap();
}

/// AC2 (Windows): a junction/reparse point inside the library that targets a directory outside it
/// is the Windows shape of the symlink probe. `fs::canonicalize` resolves reparse points, so the
/// same `confined_root_join` refusal covers it — proven here, in the Windows CI lane.
#[cfg(windows)]
#[test]
fn a_junction_out_of_the_library_fails_closed() {
    use std::process::Command;

    /// `fs::canonicalize` returns a VERBATIM path (`\\?\C:\…`), and `cmd /C mklink /J` rejects
    /// that form outright — so both arguments are handed over in their ordinary `C:\…` shape.
    /// Without this the junction is never created and the probe asserts nothing.
    fn mklink_argument(path: &std::path::Path) -> String {
        let text = path.to_string_lossy().into_owned();
        text.strip_prefix(r"\\?\")
            .map(str::to_owned)
            .unwrap_or(text)
    }

    let (fx, root_id, _) = library_fixture("junction");
    let outside = fixture_dir("junction-outside");
    let outside_dir = fs::canonicalize(outside.path()).unwrap();
    write_krea_native_file(&outside_dir.join("secret.safetensors"), 0x77);

    let junction = fx.library_dir.join("escape");
    let status = Command::new("cmd")
        .args([
            "/C",
            "mklink",
            "/J",
            &mklink_argument(&junction),
            &mklink_argument(&outside_dir),
        ])
        .status()
        .expect("mklink runs");
    assert!(status.success(), "creating a junction must succeed");
    assert!(
        junction.is_dir(),
        "SANITY: the junction must exist, or this test proves nothing"
    );

    // Compiling THROUGH the junction refuses: its canonical target is outside the root.
    let error = fx
        .store
        .compile_linked(&root_id, "escape/secret.safetensors")
        .unwrap_err();
    assert!(
        matches!(
            error,
            CheckpointPlanError::PathEscapesRoot { .. }
                | CheckpointPlanError::UnrunnableSource { .. }
        ),
        "a junction out of the library must fail closed, got {error:?}"
    );

    // And a scan never offers anything reached through it.
    let scan = fx.store.scan_root(&root_id).unwrap();
    assert!(
        scan.candidates
            .iter()
            .all(|candidate| !candidate.candidate.relative_path.starts_with("escape/")),
        "{:?}",
        scan.candidates
    );
}

// ---------------------------------------------------------------------------------------------
// cross-process serialisation of the store's read-modify-writes (sc-20635)
// ---------------------------------------------------------------------------------------------

/// The lock file every store mutator serialises on, opened the way another PROCESS would open it.
///
/// `fs2` locks an open file description, so a second descriptor on the same file contends even
/// inside one process — which is what lets these tests stand in for the API-process /
/// worker-process race without spawning a child.
fn hold_store_lock(data_dir: &Path) -> fs::File {
    let path = data_dir.join(CHECKPOINTS_DIR).join(STORE_LOCK_FILE);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .unwrap();
    fs2::FileExt::lock_exclusive(&file).unwrap();
    file
}

/// Run `operation` on another thread and report whether it finished inside `window`.
fn finishes_within<T: Send + 'static>(
    operation: impl FnOnce() -> T + Send + 'static,
    window: Duration,
) -> (bool, std::thread::JoinHandle<T>) {
    let (done_tx, done_rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let value = operation();
        let _ = done_tx.send(());
        value
    });
    (done_rx.recv_timeout(window).is_ok(), handle)
}

/// The API process (root lifecycle) and the worker process (compile, invalidate) both read a whole
/// shared document, edit it, and write it back. `write_atomic` makes each WRITE atomic; it does
/// nothing for the read-modify-write PAIR, so without a lock the later writer silently discards
/// whatever the other one committed in between.
///
/// This is that exact interleaving: the test holds the store lock the way the other process would,
/// starts `remove_root` (which must block), commits a record for a DIFFERENT root while it is
/// blocked, then releases. With the lock, `remove_root` reads the inventory the other process left
/// and removes only its own root's checkpoint.
///
/// Failing mutation: delete the `let _guard = self.lock_store()?;` line from `remove_root`.
/// `remove_root` then completes immediately (the first assertion goes red) against the pre-write
/// inventory, and the record this test commits afterwards is resurrected by nothing — the removal
/// wrote `[]` before it existed — so the last assertion goes red too.
#[test]
fn a_mutator_cannot_lose_a_concurrent_writers_record() {
    let fx = fixture("lock-lost-update");
    write_krea_native_file(&fx.library_dir.join("alpha.safetensors"), 0x11);
    let other_library = fixture_dir("lock-lost-update-other");
    let other_dir = fs::canonicalize(other_library.path()).unwrap();
    write_krea_native_file(&other_dir.join("beta.safetensors"), 0x22);

    let doomed = fx.store.approve_root(&fx.library_dir).unwrap();
    let survivor = fx.store.approve_root(&other_dir).unwrap();
    // Compile the survivor's checkpoint to capture a REAL record, then take it back out: the
    // inventory the blocked `remove_root` must not clobber is the one written below, while it is
    // already blocked.
    let survivor_checkpoint = fx
        .store
        .compile_linked(&survivor.root_id, "beta.safetensors")
        .unwrap();
    let survivor_record = survivor_checkpoint.record.clone();
    assert!(fx
        .store
        .invalidate(&survivor_checkpoint.checkpoint_id)
        .unwrap());
    let doomed_checkpoint = fx
        .store
        .compile_linked(&doomed.root_id, "alpha.safetensors")
        .unwrap();
    assert_eq!(
        fx.store
            .inventory()
            .unwrap()
            .records
            .iter()
            .map(|record| record.checkpoint_id.clone())
            .collect::<Vec<_>>(),
        vec![doomed_checkpoint.checkpoint_id.clone()],
        "SANITY: only the doomed root's checkpoint is on disk before the race"
    );

    let held = hold_store_lock(&fx.data_dir);
    let store = CheckpointPlanStore::open(&fx.data_dir);
    let doomed_id = doomed.root_id.clone();
    let (finished, handle) = finishes_within(
        move || store.remove_root(&doomed_id),
        Duration::from_millis(750),
    );
    assert!(
        !finished,
        "a mutator must wait for the store lock another process holds"
    );

    // The other process commits its own read-modify-write while the remover is parked.
    let inventory = CheckpointInventoryV1::new(vec![
        doomed_checkpoint.record.clone(),
        survivor_record.clone(),
    ])
    .unwrap();
    let mut payload = inventory.canonical_json().unwrap().into_bytes();
    payload.push(b'\n');
    fs::write(
        fx.data_dir.join(CHECKPOINTS_DIR).join(INVENTORY_FILE),
        payload,
    )
    .unwrap();
    fs2::FileExt::unlock(&held).unwrap();
    drop(held);

    let removal = handle.join().unwrap().unwrap();
    assert_eq!(
        removal.removed_checkpoints,
        vec![doomed_checkpoint.checkpoint_id.clone()]
    );
    assert_eq!(
        fx.store
            .inventory()
            .unwrap()
            .records
            .iter()
            .map(|record| record.checkpoint_id.clone())
            .collect::<Vec<_>>(),
        vec![survivor_record.checkpoint_id.clone()],
        "the concurrent writer's record must survive the removal"
    );
}

/// Every mutator — not just the one the lost-update test drives — takes the lock.
///
/// Failing mutation: delete `let _guard = self.lock_store()?;` from any one of `approve_root_inner`,
/// `rename_root`, `relink_root`, `remove_root`, `upsert_record` (reached through `compile_linked`)
/// or `invalidate`, and that mutator's `!finished` assertion goes red.
#[test]
fn every_store_mutator_serialises_on_the_store_lock() {
    let fx = fixture("lock-every-mutator");
    write_krea_native_file(&fx.library_dir.join("alpha.safetensors"), 0x11);
    let spare = fixture_dir("lock-every-mutator-spare");
    let spare_dir = fs::canonicalize(spare.path()).unwrap();
    let unapproved = fixture_dir("lock-every-mutator-new");
    let unapproved_dir = fs::canonicalize(unapproved.path()).unwrap();
    let root = fx.store.approve_root(&fx.library_dir).unwrap();
    let compiled = fx
        .store
        .compile_linked(&root.root_id, "alpha.safetensors")
        .unwrap();

    type Mutator = Box<dyn FnOnce(CheckpointPlanStore) -> bool + Send>;
    let mutators: Vec<(&str, Mutator)> = vec![
        (
            "approve_root",
            Box::new(move |store: CheckpointPlanStore| store.approve_root(&unapproved_dir).is_ok()),
        ),
        (
            "rename_root",
            Box::new({
                let root_id = root.root_id.clone();
                move |store: CheckpointPlanStore| store.rename_root(&root_id, "renamed").is_ok()
            }),
        ),
        (
            "relink_root",
            Box::new({
                let root_id = root.root_id.clone();
                move |store: CheckpointPlanStore| store.relink_root(&root_id, &spare_dir).is_ok()
            }),
        ),
        (
            "invalidate",
            Box::new({
                let checkpoint_id = compiled.checkpoint_id.clone();
                move |store: CheckpointPlanStore| store.invalidate(&checkpoint_id).is_ok()
            }),
        ),
        (
            "remove_root",
            Box::new({
                let root_id = root.root_id.clone();
                move |store: CheckpointPlanStore| store.remove_root(&root_id).is_ok()
            }),
        ),
    ];

    for (name, mutator) in mutators {
        let held = hold_store_lock(&fx.data_dir);
        let store = CheckpointPlanStore::open(&fx.data_dir);
        let (finished, handle) =
            finishes_within(move || mutator(store), Duration::from_millis(500));
        assert!(!finished, "{name} must wait for the store lock");
        fs2::FileExt::unlock(&held).unwrap();
        drop(held);
        assert!(handle.join().unwrap(), "{name} must succeed once unblocked");
    }

    // `compile_linked` reaches the lock through `upsert_record`, after the (unlocked) inspection
    // and the plan/bindings writes, so it is driven separately.
    let fresh = fixture("lock-compile");
    write_krea_native_file(&fresh.library_dir.join("alpha.safetensors"), 0x33);
    let fresh_root = fresh.store.approve_root(&fresh.library_dir).unwrap();
    let held = hold_store_lock(&fresh.data_dir);
    let store = CheckpointPlanStore::open(&fresh.data_dir);
    let (finished, handle) = finishes_within(
        move || {
            store
                .compile_linked(&fresh_root.root_id, "alpha.safetensors")
                .is_ok()
        },
        Duration::from_millis(750),
    );
    assert!(!finished, "compile_linked must wait for the store lock");
    fs2::FileExt::unlock(&held).unwrap();
    drop(held);
    assert!(handle.join().unwrap(), "compile_linked succeeds unblocked");
}

/// Re-approving a directory that is already a root under a NEW label relabels it.
///
/// The idempotent-by-path branch exists so a relinked library is not re-identified as a second
/// one; it must not also silently swallow the label the caller asked for — that is the whole
/// payload of `approve_root_with_label`.
///
/// Failing mutation: restore the plain `return Ok(existing.clone())` in the `get_by_path` branch
/// of `approve_root_inner`.
#[test]
fn re_approving_a_root_under_a_new_label_relabels_it() {
    let fx = fixture("relabel");
    let first = fx
        .store
        .approve_root_with_label(&fx.library_dir, "ComfyUI")
        .unwrap();
    assert_eq!(first.label, "ComfyUI");

    let renamed = fx
        .store
        .approve_root_with_label(&fx.library_dir, "Shared checkpoints")
        .unwrap();
    assert_eq!(renamed.root_id, first.root_id, "identity is unchanged");
    assert_eq!(renamed.label, "Shared checkpoints");
    assert_eq!(
        fx.store.approved_roots().unwrap().roots,
        vec![renamed.clone()],
        "the new label is PERSISTED, not just returned"
    );

    // An unlabelled re-approval is still a no-op: it expresses no preference, so it must not wipe
    // the label the user chose.
    assert_eq!(
        fx.store.approve_root(&fx.library_dir).unwrap().label,
        "Shared checkpoints"
    );
    // And an invalid label never reaches the store.
    assert!(matches!(
        fx.store.approve_root_with_label(&fx.library_dir, "  "),
        Err(CheckpointPlanError::InvalidRootLabel { .. })
    ));
    assert_eq!(
        fx.store.approved_roots().unwrap().roots[0].label,
        "Shared checkpoints"
    );
}

/// One unreadable plan document must not unlist the whole library.
///
/// `scan_root` classifies every persisted checkpoint under the root; propagating a `Corrupt` from
/// any one of them made a single damaged file hide every OTHER checkpoint — including the ones a
/// rescan could repair, and including the candidates that have no persisted plan at all.
///
/// Failing mutation: restore the `?` on `self.checkpoint_status(&checkpoint_id)` in `scan_root`.
#[test]
fn one_corrupt_plan_document_does_not_unlist_the_library() {
    let fx = fixture("corrupt-one");
    write_krea_native_file(&fx.library_dir.join("alpha.safetensors"), 0x11);
    write_krea_native_file(&fx.library_dir.join("beta.safetensors"), 0x22);
    let root = fx.store.approve_root(&fx.library_dir).unwrap();
    let broken = fx
        .store
        .compile_linked(&root.root_id, "alpha.safetensors")
        .unwrap();
    let healthy = fx
        .store
        .compile_linked(&root.root_id, "beta.safetensors")
        .unwrap();

    let bindings = fx
        .data_dir
        .join(CHECKPOINTS_DIR)
        .join(BINDINGS_DIR)
        .join(format!("{}.json", broken.plan.plan_id));
    assert!(bindings.is_file(), "SANITY: the bindings document is there");
    fs::write(&bindings, b"{ this is not json").unwrap();
    assert!(
        matches!(
            fx.store.checkpoint_status(&broken.checkpoint_id),
            Err(CheckpointPlanError::Corrupt { .. })
        ),
        "SANITY: the damaged checkpoint's own status still refuses"
    );

    let scan = fx.store.scan_root(&root.root_id).unwrap();
    assert!(
        scan.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == CheckpointDiagnosticCodeV1::Io
                && diagnostic.relative_path.as_deref() == Some("alpha.safetensors")
                && diagnostic.message.contains("[checkpoint-plan:corrupt]")
        }),
        "the damaged checkpoint surfaces as a diagnostic: {:?}",
        scan.diagnostics
    );
    let healthy_candidate = scan
        .candidates
        .iter()
        .find(|candidate| candidate.checkpoint_id == healthy.checkpoint_id)
        .expect("the healthy checkpoint is still listed");
    assert!(
        healthy_candidate.selectable,
        "an undamaged sibling stays selectable: {healthy_candidate:?}"
    );
    // The damaged one is still OFFERED as a header-only candidate, just not selectable: its plan
    // is unreadable, so a rescan is exactly what the user has to do.
    let broken_candidate = scan
        .candidates
        .iter()
        .find(|candidate| candidate.checkpoint_id == broken.checkpoint_id)
        .expect("the damaged checkpoint is still visible as a candidate");
    assert!(!broken_candidate.selectable);
}
