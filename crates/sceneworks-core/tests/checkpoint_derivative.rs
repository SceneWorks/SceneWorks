//! Checkpoint derivatives in the epic-19703 resolved-artifact cache (sc-20635, AC3).
//!
//! What these prove: the derivative key really is content + semantic plan + adapter/codec version +
//! backend representation; a stale or partial derivative is never admitted; a leased one is never
//! evicted; a pinned one is never evicted; and producing one never writes anything under the
//! linked library.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use tempfile::TempDir;

use sceneworks_core::checkpoint_derivative::{
    CheckpointDerivativeError, CheckpointDerivativeKindV1, CheckpointDerivativeOutcomeV1,
    CheckpointDerivativeRequestV1, CheckpointDerivativeStore,
};
use sceneworks_core::checkpoint_plan_store::{CheckpointPlanStore, ResolvedCheckpointV1};
use sceneworks_core::model_artifacts::resolved_cache::{ResolvedCachePolicy, ResolvedCacheStore};

// ---------------------------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------------------------

fn fixture_dir(label: &str) -> TempDir {
    tempfile::Builder::new()
        .prefix(&format!("ckpt-derivative-{label}-{}-", std::process::id()))
        .tempdir()
        .unwrap()
}

/// A minimal single-file Krea 2 native DiT (the same shape `checkpoint_plan_store.rs`'s fixtures
/// use): the `txtfusion.` marker the family detector keys on, dense bf16, deterministic bytes.
fn write_krea_native_file(path: &Path, fill: u8) {
    let mut header = Map::new();
    let mut offset = 0_u64;
    for name in [
        "model.diffusion_model.txtfusion.projector.weight",
        "model.diffusion_model.blocks.0.attn.wq.weight",
        "model.diffusion_model.first.weight",
    ] {
        header.insert(
            name.to_owned(),
            json!({"dtype": "BF16", "shape": [1], "data_offsets": [offset, offset + 2]}),
        );
        offset += 2;
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
    data_dir: PathBuf,
    library_dir: PathBuf,
    store: CheckpointDerivativeStore,
    checkpoint_id: String,
}

/// An approved library holding one compiled Krea checkpoint, plus an open derivative store.
fn fixture(label: &str) -> Fixture {
    fixture_with_fill(label, 7)
}

fn fixture_with_fill(label: &str, fill: u8) -> Fixture {
    let data = fixture_dir(&format!("{label}-data"));
    let library = fixture_dir(&format!("{label}-library"));
    let data_dir = fs::canonicalize(data.path()).unwrap();
    let library_dir = fs::canonicalize(library.path()).unwrap();
    write_krea_native_file(&library_dir.join("dit.safetensors"), fill);
    let plans = CheckpointPlanStore::open(&data_dir);
    let root = plans.approve_root(&library_dir).unwrap();
    let compiled = plans
        .compile_linked(&root.root_id, "dit.safetensors")
        .unwrap();
    Fixture {
        store: CheckpointDerivativeStore::open(&data_dir).unwrap(),
        checkpoint_id: compiled.checkpoint_id,
        data_dir,
        library_dir,
        _data: data,
        _library: library,
    }
}

fn request() -> CheckpointDerivativeRequestV1 {
    CheckpointDerivativeRequestV1 {
        kind: CheckpointDerivativeKindV1::DerivedIndex,
        adapter_id: "krea_2".to_owned(),
        adapter_version: "1".to_owned(),
        codec_id: "dense-bf16".to_owned(),
        codec_version: "1".to_owned(),
        backend: "mlx".to_owned(),
        outputs: vec!["index.json".to_owned()],
    }
}

fn resolve(fx: &Fixture) -> ResolvedCheckpointV1 {
    fx.store.plans().resolve(&fx.checkpoint_id).unwrap()
}

/// A producer that writes one deterministic index file.
fn write_index(
    outputs: &sceneworks_core::checkpoint_derivative::CheckpointDerivativeOutputs<'_>,
) -> Result<(), String> {
    let mut file = outputs.create("index.json").map_err(|e| e.to_string())?;
    let payload = json!({
        "checkpointId": outputs.resolved().checkpoint_id,
        "family": outputs.resolved().plan.family,
        "layers": outputs.resolved().plan.layers.len(),
    });
    file.write_all(serde_json::to_string(&payload).unwrap().as_bytes())
        .map_err(|error| error.to_string())?;
    Ok(())
}

/// Path → (len, modified) for every file under `root`, so a test can prove a whole tree is
/// byte-for-byte and timestamp-for-timestamp untouched.
fn tree_snapshot(root: &Path) -> BTreeMap<PathBuf, (u64, std::time::SystemTime, Vec<u8>)> {
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

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// ---------------------------------------------------------------------------------------------
// keying
// ---------------------------------------------------------------------------------------------

/// AC3: the derivative key covers content, semantic plan, adapter/codec version AND backend
/// representation — each one alone moves the key, and none of them is ignored.
///
/// Failing mutation: drop any one `field(...)` from `derivative_revision` (for example the
/// `request.codec_version` line) and the corresponding assertion below fails, because the two
/// requests then collide on one key and one entry would serve both.
#[test]
fn the_derivative_key_covers_content_plan_adapter_codec_and_backend() {
    let fx = fixture("key");
    let resolved = resolve(&fx);
    let base = fx.store.cache_key(&resolved, &request()).unwrap();
    assert!(base.starts_with("sha256:"), "{base}");

    let vary = |mutate: fn(&mut CheckpointDerivativeRequestV1)| {
        let mut request = request();
        mutate(&mut request);
        fx.store.cache_key(&resolved, &request).unwrap()
    };
    assert_ne!(
        base,
        vary(|request| request.kind = CheckpointDerivativeKindV1::BackendRepack),
        "the derivative KIND is key material"
    );
    assert_ne!(
        base,
        vary(|request| request.adapter_version = "2".to_owned()),
        "the ADAPTER VERSION is key material"
    );
    assert_ne!(
        base,
        vary(|request| request.codec_version = "2".to_owned()),
        "the CODEC VERSION is key material"
    );
    assert_ne!(
        base,
        vary(|request| request.codec_id = "fp8-e4m3".to_owned()),
        "the CODEC is key material"
    );
    assert_ne!(
        base,
        vary(|request| request.backend = "candle_cuda".to_owned()),
        "the BACKEND REPRESENTATION is key material"
    );
    assert_ne!(
        base,
        vary(|request| request.outputs = vec!["index.json".to_owned(), "map.json".to_owned()]),
        "the declared output shape is key material"
    );

    // The same request against a checkpoint with DIFFERENT CONTENT keys differently, even though
    // both are `dit.safetensors` under an approved root with the same family and layer shape.
    let other = fixture_with_fill("key-other", 9);
    let other_key = other.store.cache_key(&resolve(&other), &request()).unwrap();
    assert_ne!(base, other_key, "the checkpoint's CONTENT is key material");

    // The SAME bytes in a second LIBRARY key IDENTICALLY — full E1 locator independence.
    //
    // This assertion was `assert_ne!` when sc-20635 landed, because `semantic_digest` then folded
    // in `plan_id`, which the inspector derives from the checkpoint id, and a checkpoint id names
    // its root. sc-20636 removed `plan_id` from `ImportPlanV1::semantic_form`, so the digest is now
    // content-and-routing only, and two approved roots holding byte-identical checkpoints resolve
    // to ONE derivative entry instead of producing the same bytes twice. That is what E1 asks for
    // and what a content-addressed cache is for, so the test follows the contract rather than
    // pinning the behaviour it used to have.
    //
    // Sharing is safe in both directions: `derived_from` records only which checkpoint's producer
    // created the bundle, so forgetting one of the twins can cost the other a re-production but
    // can never lose data — a derivative is reproducible from its input by definition.
    let twin = fixture_with_fill("key-twin", 7);
    let twin_resolved = resolve(&twin);
    assert_ne!(fx.library_dir, twin.library_dir);
    let fingerprint = |checkpoint: &ResolvedCheckpointV1| match &checkpoint.plan.layers[0].source {
        sceneworks_core::checkpoint_import::SourceLocatorV1::Linked {
            relative_path,
            fingerprint,
            ..
        } => (relative_path.clone(), fingerprint.clone()),
        other => panic!("a linked checkpoint carries a linked locator, got {other:?}"),
    };
    assert_eq!(
        fingerprint(&resolved),
        fingerprint(&twin_resolved),
        "SANITY: the two libraries hold byte-identical checkpoints at the same relative path"
    );
    assert_eq!(
        base,
        twin.store.cache_key(&twin_resolved, &request()).unwrap(),
        "identical bytes reached through a DIFFERENT approved root are the same content, so they \
         share one derivative entry (E1)"
    );
}

// ---------------------------------------------------------------------------------------------
// production, reuse, and the linked source
// ---------------------------------------------------------------------------------------------

/// AC3: a produced derivative lands in the SHARED cache (same store root, same receipt), is reused
/// on the next call without re-running the producer, and the linked library is untouched
/// throughout — no file added, removed, rewritten, or even re-stamped.
#[test]
fn producing_a_derivative_publishes_into_the_shared_cache_and_never_writes_the_library() {
    let fx = fixture("produce");
    let before = tree_snapshot(&fx.library_dir);
    assert_eq!(before.len(), 1, "the library holds exactly the checkpoint");

    let outcome = fx
        .store
        .ensure(&fx.checkpoint_id, &request(), write_index)
        .unwrap();
    let metadata = match outcome {
        CheckpointDerivativeOutcomeV1::Produced(metadata) => *metadata,
        other => panic!("expected a fresh production, got {other:?}"),
    };
    assert!(
        metadata.production.is_derived(),
        "the entry records that it was produced, not copied"
    );

    // It lives in the epic-19703 store, under its own key, with the completion receipt that store
    // writes — not in a second cache of this module's own.
    let cache_root = fx.data_dir.join("models").join("resolved");
    let digest = metadata.cache_key.strip_prefix("sha256:").unwrap();
    let entry = cache_root.join("entries").join(digest);
    assert!(entry.join("complete.receipt.json").is_file(), "{entry:?}");
    assert!(fs::read_to_string(entry.join("bundle").join("index.json"))
        .unwrap()
        .contains(&fx.checkpoint_id));
    assert!(
        cache_root.join(".sceneworks-resolved-cache-v1").is_file(),
        "the derivative is in the shared resolved-artifact store"
    );

    // Second call: served from the cache, producer never runs.
    let reused = fx
        .store
        .ensure(&fx.checkpoint_id, &request(), |_| {
            panic!("the producer must not run for a cached derivative")
        })
        .unwrap();
    match reused {
        CheckpointDerivativeOutcomeV1::AlreadyPresent(cached) => {
            assert_eq!(cached.cache_key, metadata.cache_key)
        }
        other => panic!("expected a cache hit, got {other:?}"),
    }

    assert_eq!(
        tree_snapshot(&fx.library_dir),
        before,
        "conversion never writes, deletes, or restamps a linked source file"
    );
}

/// AC3: a producer cannot write outside the staging directory, and a failed producer publishes
/// nothing — the entry stays absent and the library stays untouched.
#[test]
fn a_failing_or_escaping_producer_publishes_nothing() {
    let fx = fixture("producer-refusals");
    let before = tree_snapshot(&fx.library_dir);

    // Traversal out of the bundle, and an undeclared output, both refuse at `create`.
    let escapes = fx.store.ensure(&fx.checkpoint_id, &request(), |outputs| {
        outputs
            .create("../../escaped.json")
            .map(|_| ())
            .map_err(|error| error.to_string())
    });
    let message = escapes.unwrap_err().to_string();
    assert!(message.contains("producer-failed"), "{message}");
    assert!(message.contains("'..'"), "{message}");

    let undeclared = fx.store.ensure(&fx.checkpoint_id, &request(), |outputs| {
        outputs
            .create("not-declared.json")
            .map(|_| ())
            .map_err(|error| error.to_string())
    });
    assert!(
        undeclared
            .unwrap_err()
            .to_string()
            .contains("was not declared"),
        "an output the request never declared cannot be created"
    );

    // A producer that simply fails leaves no entry behind either.
    let failed = fx
        .store
        .ensure(&fx.checkpoint_id, &request(), |_| {
            Err("adapter blew up".to_owned())
        })
        .unwrap_err();
    assert!(matches!(failed, CheckpointDerivativeError::Producer { .. }));

    let resolved = resolve(&fx);
    assert_eq!(
        fx.store.lookup(&resolved, &request()).unwrap(),
        None,
        "a stale or partial derivative is never admitted"
    );
    assert_eq!(tree_snapshot(&fx.library_dir), before);

    // And the entry is still usable afterwards: the failed attempt did not wedge the key.
    assert!(matches!(
        fx.store
            .ensure(&fx.checkpoint_id, &request(), write_index)
            .unwrap(),
        CheckpointDerivativeOutcomeV1::Produced(_)
    ));
}

/// AC3: a producer that does not write a file it declared is a PARTIAL bundle and is refused at
/// publication rather than published with a hole in it.
#[test]
fn a_partial_bundle_is_never_admitted() {
    let fx = fixture("partial");
    let mut request = request();
    request.outputs = vec!["index.json".to_owned(), "map.json".to_owned()];

    let error = fx
        .store
        .ensure(&fx.checkpoint_id, &request, |outputs| {
            // Writes only the first of the two declared outputs.
            let mut file = outputs.create("index.json").map_err(|e| e.to_string())?;
            file.write_all(b"{}").map_err(|error| error.to_string())
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("map.json"), "{error}");
    assert!(error.contains("missing"), "{error}");
    assert_eq!(fx.store.lookup(&resolve(&fx), &request).unwrap(), None);
}

/// AC3 / E7: a checkpoint whose bytes changed since its plan compiled refuses BEFORE any
/// derivative work — so a derivative can never be produced from, or served for, content the plan
/// no longer describes.
#[test]
fn a_drifted_checkpoint_refuses_before_any_derivative_is_produced() {
    let fx = fixture("drift");
    fx.store
        .ensure(&fx.checkpoint_id, &request(), write_index)
        .unwrap();
    let stale_key = fx.store.cache_key(&resolve(&fx), &request()).unwrap();

    // The user edits the checkpoint in place.
    write_krea_native_file(&fx.library_dir.join("dit.safetensors"), 11);

    let error = fx
        .store
        .ensure(&fx.checkpoint_id, &request(), |_| {
            panic!("no producer may run for a drifted checkpoint")
        })
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("[checkpoint-plan:source-drifted]"),
        "{error}"
    );

    // After a rescan the SAME request keys somewhere else, so the derivative of the old bytes is
    // never handed out for the new ones.
    fx.store
        .plans()
        .rescan_checkpoint(&fx.checkpoint_id)
        .unwrap();
    let fresh_key = fx.store.cache_key(&resolve(&fx), &request()).unwrap();
    assert_ne!(stale_key, fresh_key);
    assert!(
        fx.store
            .cache()
            .lookup_complete(&fresh_key)
            .unwrap()
            .is_none(),
        "the new content has no derivative yet"
    );
}

// ---------------------------------------------------------------------------------------------
// leases, pins, eviction
// ---------------------------------------------------------------------------------------------

/// AC3: a LEASED derivative is never evicted, and the same entry is evictable the moment the lease
/// is dropped.
///
/// The lease here is taken BEFORE the sweep, so it is covered by TWO independent guards and this
/// test cannot distinguish them:
///
/// * the scan phase's `artifact_lock_is_contended` probe (`retention.rs`, inside `scan_entry`),
///   which retains the entry before it ever becomes an eviction candidate; and
/// * `evict_candidate`'s phase-two `FileExt::try_lock_exclusive` contended arm.
///
/// Measured, one mutation at a time: with only the scan-phase probe no-opped this stays GREEN
/// (phase two catches it); with only the phase-two arm no-opped it also stays GREEN (the scan
/// probe catches it); with BOTH removed it goes RED, evicting the leased entry. So the honest
/// claim is "at least one of the two guards is present", and the two guards are pinned separately:
/// the scan probe by `the_retention_scan_classifies_without_holding_the_lock_that_blocks_status_reads`
/// and phase two, alone, by
/// `retention::tests::the_phase_two_artifact_lock_retains_an_entry_leased_after_it_was_classified`,
/// which leases in the window between them.
#[test]
fn a_leased_derivative_is_never_evicted_and_an_unleased_one_is() {
    let fx = fixture("lease");
    let resolved = resolve(&fx);
    fx.store
        .ensure(&fx.checkpoint_id, &request(), write_index)
        .unwrap();
    let cache_key = fx.store.cache_key(&resolved, &request()).unwrap();

    // A one-byte budget and a one-second idle window, swept from an hour in the future: every
    // entry retention is ALLOWED to reclaim is reclaimed on this pass.
    let policy = ResolvedCachePolicy {
        enabled: true,
        max_bytes: 1,
        inactivity_seconds: 1,
    };

    let lease = fx
        .store
        .acquire(&resolved, &request())
        .unwrap()
        .expect("a published derivative leases");
    let report = fx
        .store
        .cache()
        .enforce_retention(&policy, now() + 3600)
        .unwrap();
    assert!(
        report.evicted.is_empty(),
        "a leased derivative must never be evicted: {report:?}"
    );
    assert!(
        fx.store
            .cache()
            .lookup_complete(&cache_key)
            .unwrap()
            .is_some(),
        "the leased bundle is still there"
    );
    drop(lease);

    let report = fx
        .store
        .cache()
        .enforce_retention(&policy, now() + 3600)
        .unwrap();
    assert_eq!(
        report
            .evicted
            .iter()
            .map(|record| record.cache_key.as_str())
            .collect::<Vec<_>>(),
        vec![cache_key.as_str()],
        "with no lease the derivative is reclaimable: {report:?}"
    );
    assert_eq!(fx.store.cache().lookup_complete(&cache_key).unwrap(), None);

    // Nothing about the library changed while the cache reclaimed its own derivative.
    assert!(fx.library_dir.join("dit.safetensors").is_file());
    // And it can simply be produced again.
    assert!(matches!(
        fx.store
            .ensure(&fx.checkpoint_id, &request(), write_index)
            .unwrap(),
        CheckpointDerivativeOutcomeV1::Produced(_)
    ));
}

/// AC3: a derivative is evictable at all only because retention judges it as DERIVED. A source
/// copy has to prove a complete second copy in the source library; a derivative has none by
/// construction, so it would be permanently unevictable if it were judged by that rule.
///
/// Failing mutation: delete the `Some(scanned) if scanned.production.is_derived() => None` arm in
/// `evict_candidate` (and the matching `metadata.production.is_derived()` branch) and this goes
/// red with a `SourceUnverified` hold instead of an eviction.
#[test]
fn a_derived_entry_is_evicted_without_a_source_library_second_copy() {
    let fx = fixture("derived-eviction");
    fx.store
        .ensure(&fx.checkpoint_id, &request(), write_index)
        .unwrap();
    let cache_key = fx.store.cache_key(&resolve(&fx), &request()).unwrap();
    // There is deliberately no source library at all: `<data>/models/hub` holds nothing that
    // could be a second copy of this bundle.
    let report = fx
        .store
        .cache()
        .enforce_retention(
            &ResolvedCachePolicy {
                enabled: true,
                max_bytes: 1,
                inactivity_seconds: 1,
            },
            now() + 3600,
        )
        .unwrap();
    assert!(
        report.retained.is_empty(),
        "nothing should be held back: {report:?}"
    );
    assert_eq!(
        report
            .evicted
            .iter()
            .map(|record| record.cache_key.as_str())
            .collect::<Vec<_>>(),
        vec![cache_key.as_str()]
    );
}

/// AC3: a PINNED derivative is never evicted, and removal by lifecycle action still reaches it.
#[test]
fn a_pinned_derivative_is_retained_and_removal_still_reaches_it() {
    let fx = fixture("pin");
    fx.store
        .ensure(&fx.checkpoint_id, &request(), write_index)
        .unwrap();
    let cache_key = fx.store.cache_key(&resolve(&fx), &request()).unwrap();
    fx.store.cache().set_artifact_pin(&cache_key, true).unwrap();

    let report = fx
        .store
        .cache()
        .enforce_retention(
            &ResolvedCachePolicy {
                enabled: true,
                max_bytes: 1,
                inactivity_seconds: 1,
            },
            now() + 3600,
        )
        .unwrap();
    assert!(report.evicted.is_empty(), "{report:?}");
    assert!(fx
        .store
        .cache()
        .lookup_complete(&cache_key)
        .unwrap()
        .is_some());

    // While it is pinned, removal REFUSES rather than quietly dropping a pinned bundle — the
    // shared store's own rule, applied unchanged to a derivative.
    let pinned_refusal = fx
        .store
        .remove_derivatives_for_checkpoint(&fx.checkpoint_id)
        .unwrap_err()
        .to_string();
    assert!(pinned_refusal.contains("pinned"), "{pinned_refusal}");
    assert!(fx
        .store
        .cache()
        .lookup_complete(&cache_key)
        .unwrap()
        .is_some());

    // Unpinned, forgetting the checkpoint drops its plan AND its derivatives — SceneWorks-owned
    // state only.
    fx.store
        .cache()
        .set_artifact_pin(&cache_key, false)
        .unwrap();
    let library_before = tree_snapshot(&fx.library_dir);
    let (removed, derivatives) = fx.store.invalidate_checkpoint(&fx.checkpoint_id).unwrap();
    assert!(removed, "the plan record was dropped");
    assert_eq!(
        derivatives
            .iter()
            .map(|outcome| outcome.cache_key.as_str())
            .collect::<Vec<_>>(),
        vec![cache_key.as_str()]
    );
    assert_eq!(fx.store.cache().lookup_complete(&cache_key).unwrap(), None);
    assert_eq!(
        tree_snapshot(&fx.library_dir),
        library_before,
        "removal never deletes a linked file"
    );
}

/// AC3 / AC1: removing the approved ROOT drops every plan and every derivative produced from it,
/// and still never touches the library.
#[test]
fn removing_a_root_drops_its_derivatives_and_leaves_the_library_intact() {
    let fx = fixture("root-removal");
    fx.store
        .ensure(&fx.checkpoint_id, &request(), write_index)
        .unwrap();
    let cache_key = fx.store.cache_key(&resolve(&fx), &request()).unwrap();
    let root_id = fx.store.plans().approved_roots().unwrap().roots[0]
        .root_id
        .clone();
    let before = tree_snapshot(&fx.library_dir);

    let (removal, derivatives) = fx.store.remove_root(&root_id).unwrap();
    assert_eq!(removal.removed_checkpoints, vec![fx.checkpoint_id.clone()]);
    assert_eq!(derivatives.len(), 1);
    assert_eq!(fx.store.cache().lookup_complete(&cache_key).unwrap(), None);
    assert!(fx.store.plans().approved_roots().unwrap().roots.is_empty());
    assert_eq!(
        tree_snapshot(&fx.library_dir),
        before,
        "removing a linked library forgets it; it never deletes it"
    );
}

/// A published derivative that is altered on disk is refused at the lease boundary rather than
/// handed to a loader: the shared store's own full re-hash covers derivatives too.
#[test]
fn an_altered_derivative_bundle_is_refused_at_the_lease_boundary() {
    let fx = fixture("altered");
    let resolved = resolve(&fx);
    fx.store
        .ensure(&fx.checkpoint_id, &request(), write_index)
        .unwrap();
    let cache_key = fx.store.cache_key(&resolved, &request()).unwrap();
    let digest = cache_key.strip_prefix("sha256:").unwrap();
    let staged = fx
        .data_dir
        .join("models")
        .join("resolved")
        .join("entries")
        .join(digest)
        .join("bundle")
        .join("index.json");
    assert!(fx.store.acquire(&resolved, &request()).unwrap().is_some());

    let mut tampered = fs::read(&staged).unwrap();
    tampered.reverse();
    fs::write(&staged, tampered).unwrap();
    // The TYPED refusal, not `is_err() || is_none()`: an or over two outcomes passes whichever one
    // happens, so it could never have told a re-hash failure apart from the entry simply being
    // gone, and it would have kept passing if the store stopped checking at all.
    match fx.store.acquire(&resolved, &request()) {
        Err(CheckpointDerivativeError::Cache(error)) => {
            let message = error.to_string();
            assert!(
                message.contains("index.json") || message.contains("digest"),
                "the refusal names what failed re-validation: {message}"
            );
        }
        other => panic!("an altered derivative bundle must refuse at the lease, got {other:?}"),
    }
}

/// The request validator refuses key material it cannot key on, rather than digesting it and
/// producing a key two different requests could share.
#[test]
fn unkeyable_requests_refuse() {
    let fx = fixture("validation");
    let resolved = resolve(&fx);
    let refuse = |mutate: fn(&mut CheckpointDerivativeRequestV1)| {
        let mut request = request();
        mutate(&mut request);
        fx.store
            .cache_key(&resolved, &request)
            .expect_err("must refuse")
            .to_string()
    };
    assert!(refuse(|request| request.backend = String::new()).contains("backend"));
    assert!(refuse(|request| request.backend = "mlx/cuda".to_owned()).contains("backend"));
    assert!(refuse(|request| request.outputs.clear()).contains("at least one output"));
    assert!(refuse(|request| request.outputs = vec!["../x".to_owned()]).contains("'..'"));
    assert!(refuse(|request| request.outputs = vec!["/abs".to_owned()]).contains("relative"));
    // The derivative output validator is the shared one now, so the rules the document contract
    // enforces reach here too rather than stopping at the four that used to be duplicated.
    assert!(refuse(|request| request.outputs = vec!["c:idx.json".to_owned()]).contains("':'"));
    assert!(
        refuse(|request| request.outputs = vec!["a//b.json".to_owned()])
            .contains("empty path component")
    );
    assert!(
        refuse(|request| request.outputs = vec!["bad\u{0007}.json".to_owned()])
            .contains("control characters")
    );
    assert!(
        refuse(|request| request.outputs = vec!["a.json".to_owned(), "a.json".to_owned()])
            .contains("declared twice")
    );
}

/// Two DIFFERENT derivatives of one checkpoint coexist as separate entries; neither is served for
/// the other.
#[test]
fn two_derivative_kinds_of_one_checkpoint_are_separate_entries() {
    let fx = fixture("kinds");
    let index = request();
    let mut repack = request();
    repack.kind = CheckpointDerivativeKindV1::BackendRepack;
    repack.outputs = vec!["repacked.safetensors".to_owned()];

    fx.store
        .ensure(&fx.checkpoint_id, &index, write_index)
        .unwrap();
    fx.store
        .ensure(&fx.checkpoint_id, &repack, |outputs| {
            let mut file = outputs
                .create("repacked.safetensors")
                .map_err(|e| e.to_string())?;
            file.write_all(b"repacked")
                .map_err(|error| error.to_string())
        })
        .unwrap();

    let resolved = resolve(&fx);
    let index_key = fx.store.cache_key(&resolved, &index).unwrap();
    let repack_key = fx.store.cache_key(&resolved, &repack).unwrap();
    assert_ne!(index_key, repack_key);
    assert!(fx.store.lookup(&resolved, &index).unwrap().is_some());
    assert!(fx.store.lookup(&resolved, &repack).unwrap().is_some());

    // Both belong to this checkpoint, so invalidating it reaches both.
    let (_, removed) = fx.store.invalidate_checkpoint(&fx.checkpoint_id).unwrap();
    assert_eq!(removed.len(), 2);
}

/// The derivative store never adopts a foreign directory as its cache: it is the shared store's
/// own marker-guarded root or nothing.
#[test]
fn the_derivative_store_uses_the_shared_marked_cache_root() {
    let fx = fixture("root");
    let shared = ResolvedCacheStore::open(&fx.data_dir).unwrap();
    assert_eq!(fx.store.cache().root(), shared.root());
    assert_eq!(
        shared.root(),
        fx.data_dir.join("models").join("resolved").as_path()
    );
}

/// BLOCKER 6 (sc-20651): a MANAGED install produces derivatives.
///
/// `ensure` used to parse a LINKED checkpoint id to find the directory the source bytes sit under,
/// and refused everything else with "is not a linked checkpoint" before a producer ever ran — so
/// every checkpoint the user brought in through "Add to SceneWorks" silently had no derived index,
/// no normalized layout and no backend repack, for the whole life of the feature.
///
/// The install below is built through the real staging/commit ingest, not by hand.
#[test]
fn a_managed_install_produces_and_serves_derivatives() {
    use sceneworks_core::checkpoint_import::ManagedProvenanceV1;
    use sceneworks_core::checkpoint_ingest::ManagedIngest;

    let data = fixture_dir("managed-derivative-data");
    let source_dir = fixture_dir("managed-derivative-source");
    let data_dir = fs::canonicalize(data.path()).unwrap();
    let source = fs::canonicalize(source_dir.path())
        .unwrap()
        .join("dit.safetensors");
    write_krea_native_file(&source, 11);

    let plans = CheckpointPlanStore::open(&data_dir);
    let ingest = ManagedIngest::begin(
        &plans,
        "krea-managed",
        ManagedProvenanceV1 {
            source: "local_path".to_owned(),
            ..ManagedProvenanceV1::default()
        },
    )
    .unwrap();
    ingest.stage_copy_file(&source, "dit.safetensors").unwrap();
    let install = ingest.finalize("dit.safetensors", None).unwrap();
    assert_eq!(install.checkpoint_id, "managed/krea-managed");

    let store = CheckpointDerivativeStore::open(&data_dir).unwrap();
    let outcome = store
        .ensure(&install.checkpoint_id, &request(), write_index)
        .unwrap_or_else(|error| panic!("a managed install must produce derivatives: {error}"));
    assert!(
        matches!(outcome, CheckpointDerivativeOutcomeV1::Produced(_)),
        "{outcome:?}"
    );

    // The published entry is keyed to this checkpoint, resolvable and leasable — i.e. the
    // derivative really was produced against the MANAGED install directory, which is the only
    // directory these bytes exist in.
    let resolved = plans.resolve(&install.checkpoint_id).unwrap();
    let metadata = store
        .lookup(&resolved, &request())
        .unwrap()
        .expect("the produced derivative is published and complete");
    assert_eq!(
        metadata.derived_from.as_deref(),
        Some(install.checkpoint_id.as_str())
    );
    assert!(store.acquire(&resolved, &request()).unwrap().is_some());

    // And removal reaches it by the same identity.
    let removed = store
        .remove_derivatives_for_checkpoint(&install.checkpoint_id)
        .unwrap();
    assert_eq!(removed.len(), 1);
}

/// sc-20651 feature-end review: tearing down a MANAGED install goes through the derivative store,
/// derivatives FIRST, so a derivative still in use holds the whole teardown rather than being
/// orphaned by it.
///
/// The API's model-delete route used to call `CheckpointPlanStore::remove_managed` directly, which
/// drops the plan documents and then `remove_dir_all`s the install tree with nothing asking about
/// derivatives. A pinned or leased derivative survived that — but its plan and its source bytes did
/// not, and its `derivedFrom` checkpoint id, the only handle any later invalidation has, no longer
/// resolved to anything. `remove_managed_install` applies the ordering `invalidate_checkpoint`
/// already had on the linked side.
///
/// Failing mutation: swap the two statements in `remove_managed_install` so `plans.remove_managed`
/// runs first — the pinned refusal still comes back, but the plan and the install bytes are gone.
#[test]
fn a_pinned_derivative_holds_a_managed_teardown_instead_of_being_orphaned_by_it() {
    use sceneworks_core::checkpoint_import::ManagedProvenanceV1;
    use sceneworks_core::checkpoint_ingest::ManagedIngest;

    let data = fixture_dir("managed-teardown-data");
    let source_dir = fixture_dir("managed-teardown-source");
    let data_dir = fs::canonicalize(data.path()).unwrap();
    let source = fs::canonicalize(source_dir.path())
        .unwrap()
        .join("dit.safetensors");
    write_krea_native_file(&source, 13);

    let plans = CheckpointPlanStore::open(&data_dir);
    let ingest = ManagedIngest::begin(
        &plans,
        "krea-teardown",
        ManagedProvenanceV1 {
            source: "local_path".to_owned(),
            ..ManagedProvenanceV1::default()
        },
    )
    .unwrap();
    ingest.stage_copy_file(&source, "dit.safetensors").unwrap();
    let install = ingest.finalize("dit.safetensors", None).unwrap();
    let install_dir = plans.install_dir("krea-teardown").unwrap();
    assert!(
        install_dir.join("dit.safetensors").is_file(),
        "fixture check: the install tree holds the only copy of these bytes"
    );

    let store = CheckpointDerivativeStore::open(&data_dir).unwrap();
    store
        .ensure(&install.checkpoint_id, &request(), write_index)
        .unwrap();
    let resolved = plans.resolve(&install.checkpoint_id).unwrap();
    let cache_key = store.cache_key(&resolved, &request()).unwrap();
    store.cache().set_artifact_pin(&cache_key, true).unwrap();

    // Pinned: the teardown REFUSES, and neither the plan nor the install bytes are touched.
    let refusal = store
        .remove_managed_install("krea-teardown")
        .unwrap_err()
        .to_string();
    assert!(
        refusal.contains("pinned"),
        "a pinned derivative must hold the managed teardown, got: {refusal}"
    );
    assert!(
        plans.resolve(&install.checkpoint_id).is_ok(),
        "the plan must survive a refused teardown: the pinned derivative's only handle is its id"
    );
    assert!(
        install_dir.join("dit.safetensors").is_file(),
        "the install bytes must survive a refused teardown"
    );
    assert!(store.cache().lookup_complete(&cache_key).unwrap().is_some());

    // Unpinned: derivative, plan and install tree all go, in that order.
    store.cache().set_artifact_pin(&cache_key, false).unwrap();
    let (removed, derivatives) = store.remove_managed_install("krea-teardown").unwrap();
    assert!(removed, "the plan record was dropped");
    assert_eq!(
        derivatives
            .iter()
            .map(|outcome| outcome.cache_key.as_str())
            .collect::<Vec<_>>(),
        vec![cache_key.as_str()]
    );
    assert_eq!(store.cache().lookup_complete(&cache_key).unwrap(), None);
    assert!(
        !install_dir.exists(),
        "a completed managed teardown removes the SceneWorks-owned install tree"
    );
}
