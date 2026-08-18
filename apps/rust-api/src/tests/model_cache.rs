//! Resolved-model hot-cache control surface (sc-19711).
//!
//! Every test here drives a REAL materialized cache entry through the REAL store operations —
//! never a stubbed status shape — because the whole point of the surface is that what the UI shows
//! and what the resolver would actually do cannot drift apart.

use super::support::*;
use sceneworks_core::model_artifacts::resolved_cache::{
    MaterializationCancellation, MaterializationOutcome, ResolvedCacheMaterializer,
    ResolvedCacheStore,
};
use sceneworks_core::model_artifacts::{
    ArtifactAvailability, ArtifactCompleteness, ArtifactFile, ArtifactIdentity, ArtifactLocation,
    ArtifactMemberRole, ArtifactProvenance, ArtifactSourceLibrary, PromotionCandidate,
    ResolvedBundleClosure, ResolvedBundleMember, ResolvedModelArtifact,
    MODEL_ARTIFACT_CONTRACT_VERSION,
};

/// The store's own tombstone file name. Writing garbage here is how a journal read fails for a
/// reason that does NOT prove the entry holds no recoverable state — the exact case whose pin
/// answer must stay `Unknown` rather than collapsing to "not pinned".
const EVICTED_MARKER_FILE: &str = "evicted.marker.json";

struct CacheFixture {
    temp_dir: tempfile::TempDir,
    store: ResolvedCacheStore,
    cache_key: String,
    entry: std::path::PathBuf,
}

/// Materialize one complete resolved-cache entry for `owner/cached-model` and register a catalog
/// model that declares the same source repository, so the status route's identity join has
/// something real to resolve.
fn cache_fixture(tier: &str) -> CacheFixture {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let data_dir = temp_dir.path().join("data");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        format!(
            r#"{{
                  "schemaVersion": 1,
                  "models": [{{
                    "id": "cached_model",
                    "name": "Cached Model",
                    "type": "image",
                    "family": "z-image",
                    "downloads": [{{ "repo": "owner/cached-model", "variant": "{tier}" }}]
                  }}]
                }}"#
        ),
    )
    .expect("user models writes");

    let revision = "a".repeat(40);
    let library = sceneworks_core::hf_home::model_source_library(&data_dir)
        .root()
        .to_owned();
    let snapshot = ArtifactSourceLibrary::new(&library)
        .expect("library root")
        .repository_root("owner/cached-model")
        .expect("repository root")
        .join("snapshots")
        .join(&revision);
    std::fs::create_dir_all(&snapshot).expect("snapshot dir creates");
    std::fs::write(snapshot.join("model.safetensors"), b"weights-0123456789")
        .expect("weights write");
    let identity =
        ArtifactIdentity::pinned("owner/cached-model", &revision, "default").expect("identity");
    let closure = ResolvedBundleClosure::new(vec![ResolvedBundleMember {
        role: ArtifactMemberRole::Primary,
        component_id: None,
        source: identity.clone(),
        tier: Some(tier.to_owned()),
        source_subpath: std::path::PathBuf::new(),
        destination: std::path::PathBuf::new(),
        files: vec![ArtifactFile::new("model.safetensors").expect("file")],
    }])
    .expect("closure");
    let artifact = ResolvedModelArtifact {
        schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
        identity: identity.clone(),
        location: ArtifactLocation::SourceLibrary { root: snapshot },
        closure,
        provenance: ArtifactProvenance {
            identity,
            fixed_artifact_tier: Some(tier.to_owned()),
        },
        completeness: ArtifactCompleteness::Complete,
        availability: ArtifactAvailability::Available,
    };
    let candidate = PromotionCandidate {
        cache_key: artifact.cache_key().expect("cache key"),
        artifact,
    };
    let store = ResolvedCacheStore::open(&data_dir).expect("store opens");
    let entry = store.entry_path(&candidate.cache_key).expect("entry path");
    match ResolvedCacheMaterializer::new(store.clone())
        .materialize(
            &candidate,
            &library,
            "test:cached_model",
            &MaterializationCancellation::default(),
        )
        .expect("materialization runs")
    {
        MaterializationOutcome::Published(_) => {}
        other => panic!("fixture must publish, got {other:?}"),
    }
    assert!(entry.join("bundle").join("model.safetensors").is_file());
    CacheFixture {
        temp_dir,
        store,
        cache_key: candidate.cache_key,
        entry,
    }
}

fn entry_for<'a>(body: &'a Value, cache_key: &str) -> &'a Value {
    body["entries"]
        .as_array()
        .expect("entries is an array")
        .iter()
        .find(|entry| entry["cacheKey"] == json!(cache_key))
        .expect("the materialized entry is listed")
}

/// The walking skeleton: a real entry is listed with its typed state, joined to the catalog model
/// that owns its repository, previewed with real measured bytes, and then removed through the real
/// `remove_entry`. Byte accounting must agree between the ledger listing and the removal.
#[tokio::test]
async fn status_lists_a_real_entry_and_removal_runs_through_the_real_preview() {
    let fixture = cache_fixture("q8");

    let (status, body) = request(
        create_app(test_settings(&fixture.temp_dir)).expect("app creates"),
        "GET",
        "/api/v1/model-cache",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["initialized"], json!(true));
    assert_eq!(body["error"], Value::Null);
    assert_eq!(body["entryCount"], json!(1));
    // The store→API isolation seam. The fixture puts the data directory and the source library
    // under one temp root, so they genuinely share a volume and the local copies buy no disconnect
    // or performance isolation at all — "a same-volume configuration clearly reports no isolation"
    // is an acceptance criterion, and it is asserted HERE because the web suite mocks this field.
    // Without this, `compare_source_volume` could be replaced by a hardcoded `Unknown` and every
    // endpoint test would stay green.
    assert_eq!(
        body["sourceVolumeRelation"],
        json!("same"),
        "the fixture's cache and source library share a volume; the API must say so rather than \
         reporting an unknown relation the UI would render as no warning at all"
    );
    // The configured external library location, so the UI can name the location the same-volume
    // warning is about instead of telling the user to move "the model library" they can't find.
    let configured_library =
        sceneworks_core::hf_home::model_source_library(&fixture.temp_dir.path().join("data"))
            .root()
            .to_path_buf();
    assert_eq!(
        body["sourceLibraryPath"],
        json!(configured_library),
        "the status read must report the library location it actually compared against"
    );
    let listed = entry_for(&body, &fixture.cache_key);
    assert_eq!(listed["state"], json!("complete"));
    assert_eq!(listed["repository"], json!("owner/cached-model"));
    assert_eq!(listed["tier"], json!("q8"));
    assert_eq!(listed["pinned"], json!(false));
    assert_eq!(
        listed["modelIds"],
        json!(["cached_model"]),
        "the identity join belongs to the backend, not to the UI"
    );
    let ledger_bytes = listed["bytes"].as_u64().expect("entry bytes");
    assert!(ledger_bytes > 0);
    assert_eq!(body["usedBytes"], json!(ledger_bytes));
    assert_eq!(
        body["reclaimableBytes"],
        json!(ledger_bytes),
        "a complete unpinned entry is reclaimable in full"
    );
    assert_eq!(body["pinnedBytes"], json!(0));

    let (status, preview) = request(
        create_app(test_settings(&fixture.temp_dir)).expect("app creates"),
        "POST",
        "/api/v1/model-cache/removal-preview",
        json!({ "cacheKey": fixture.cache_key }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(preview["blocked"], Value::Null);
    assert_eq!(preview["pins"]["kind"], json!("known"));
    assert_eq!(preview["pins"]["artifactPinned"], json!(false));
    assert_eq!(preview["pins"]["owners"], json!([]));
    assert_eq!(
        preview["sourceUnavailableWarning"],
        Value::Null,
        "the source library is present, so removal costs nothing but re-materialization"
    );
    let previewed_bytes = preview["reclaimableBytes"].as_u64().expect("preview bytes");
    assert!(previewed_bytes >= ledger_bytes);

    let (status, outcome) = request(
        create_app(test_settings(&fixture.temp_dir)).expect("app creates"),
        "POST",
        "/api/v1/model-cache/remove",
        json!({ "cacheKey": fixture.cache_key }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(outcome["reclaimedBytes"], json!(previewed_bytes));
    assert!(
        !fixture.entry.exists(),
        "removal must go through the real store operation, not just report success"
    );
    assert!(fixture
        .store
        .lookup_complete(&fixture.cache_key)
        .expect("lookup succeeds")
        .is_none());
}

/// Pinning is the "Keep locally" control. It must round-trip through the store AND change what the
/// removal preview says, so the UI can disable the destructive action for the right reason.
#[tokio::test]
async fn pinning_blocks_removal_and_moves_bytes_out_of_the_reclaimable_total() {
    let fixture = cache_fixture("q4");

    let (status, pinned) = request(
        create_app(test_settings(&fixture.temp_dir)).expect("app creates"),
        "POST",
        "/api/v1/model-cache/pin",
        json!({ "cacheKey": fixture.cache_key, "pinned": true }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(pinned["pinned"], json!(true));
    assert_eq!(pinned["artifactPinned"], json!(true));
    assert!(fixture
        .store
        .effective_pin(&fixture.cache_key)
        .expect("pin reads back"));

    let (_, body) = request(
        create_app(test_settings(&fixture.temp_dir)).expect("app creates"),
        "GET",
        "/api/v1/model-cache",
        Value::Null,
    )
    .await;
    let bytes = entry_for(&body, &fixture.cache_key)["bytes"]
        .as_u64()
        .expect("entry bytes");
    assert_eq!(body["pinnedBytes"], json!(bytes));
    assert_eq!(
        body["reclaimableBytes"],
        json!(0),
        "a pinned entry must never be counted as space the app can reclaim"
    );

    let (status, preview) = request(
        create_app(test_settings(&fixture.temp_dir)).expect("app creates"),
        "POST",
        "/api/v1/model-cache/removal-preview",
        json!({ "cacheKey": fixture.cache_key }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(preview["blocked"]
        .as_str()
        .expect("a pinned entry is blocked")
        .contains("pinned"));
    assert_eq!(preview["pins"]["artifactPinned"], json!(true));

    // Unpinning re-opens the action, and the entry is genuinely removable afterwards.
    let (status, unpinned) = request(
        create_app(test_settings(&fixture.temp_dir)).expect("app creates"),
        "POST",
        "/api/v1/model-cache/pin",
        json!({ "cacheKey": fixture.cache_key, "pinned": false }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(unpinned["pinned"], json!(false));
    let (status, _) = request(
        create_app(test_settings(&fixture.temp_dir)).expect("app creates"),
        "POST",
        "/api/v1/model-cache/remove",
        json!({ "cacheKey": fixture.cache_key }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!fixture.entry.exists());
}

/// An entry whose state cannot be read must reach the UI as `unknown` pins with a refusal — never
/// as "no pins". Collapsing the two would present a possibly-pinned entry as freely removable.
#[tokio::test]
async fn an_unreadable_entry_previews_unknown_pins_rather_than_no_pins() {
    let fixture = cache_fixture("q8");
    fixture
        .store
        .set_artifact_pin(&fixture.cache_key, true)
        .expect("pin applies");
    // A plain regular file, so the entry stays measurable: the refusal has to come from the unknown
    // pin state itself rather than from an incidental measurement failure.
    std::fs::write(
        fixture.entry.join(EVICTED_MARKER_FILE),
        b"garbage-tombstone",
    )
    .expect("tombstone writes");

    let (status, preview) = request(
        create_app(test_settings(&fixture.temp_dir)).expect("app creates"),
        "POST",
        "/api/v1/model-cache/removal-preview",
        json!({ "cacheKey": fixture.cache_key }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(preview["pins"]["kind"], json!("unknown"));
    assert_eq!(
        preview["pins"]["artifactPinned"],
        Value::Null,
        "the Unknown arm must carry no pin fields a client could misread as 'false'"
    );
    assert!(preview["reclaimableBytes"].as_u64().expect("bytes") > 0);
    assert!(preview["blocked"]
        .as_str()
        .expect("an unreadable entry is blocked")
        .contains("cannot be read"));

    // The removal fails CLOSED, and it does so because the entry's own stored state is damaged —
    // an unreadable tombstone is the host's problem, not a badly-formed request. So this is a 500:
    // the store could not answer, which is a different thing from the store declining to.
    let (status, _) = request(
        create_app(test_settings(&fixture.temp_dir)).expect("app creates"),
        "POST",
        "/api/v1/model-cache/remove",
        json!({ "cacheKey": fixture.cache_key }),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(fixture
        .entry
        .join("bundle")
        .join("model.safetensors")
        .is_file());
}

/// A damaged entry must still be LISTED (labelled corrupt) rather than erasing the whole status
/// read: a status surface that reports an empty cache while bytes sit on disk is worse than one
/// that names the damage.
#[tokio::test]
async fn a_damaged_entry_is_labelled_instead_of_blanking_the_listing() {
    let fixture = cache_fixture("q8");
    // A bundle whose bytes no longer match the closure. The store falls back to the last journal
    // generation it can still prove, so the entry stops being `complete` — what matters here is
    // that it is still LISTED and stops counting as space the app can reclaim.
    std::fs::write(
        fixture.entry.join("bundle").join("model.safetensors"),
        b"tampered",
    )
    .expect("bundle tamper writes");

    let (status, body) = request(
        create_app(test_settings(&fixture.temp_dir)).expect("app creates"),
        "GET",
        "/api/v1/model-cache",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["error"], Value::Null);
    assert_eq!(body["entryCount"], json!(1));
    assert_ne!(
        body["entries"][0]["state"],
        json!("complete"),
        "a bundle that no longer matches its closure must not read as a complete local copy"
    );
    assert_eq!(body["reclaimableBytes"], json!(0));

    // Residue whose journal cannot be read at all is labelled `corrupt` — and one damaged entry
    // must not take the rest of the listing (or the byte accounting) down with it.
    for slot in 0..=1 {
        std::fs::write(
            fixture.entry.join(format!("metadata.{slot}.json")),
            b"not-json",
        )
        .expect("journal garbage writes");
    }
    let (status, body) = request(
        create_app(test_settings(&fixture.temp_dir)).expect("app creates"),
        "GET",
        "/api/v1/model-cache",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["error"], Value::Null);
    assert_eq!(body["entryCount"], json!(1));
    assert_eq!(body["entries"][0]["state"], json!("corrupt"));
    assert_eq!(body["entries"][0]["repository"], Value::Null);
    assert_eq!(body["usedBytes"], json!(0));
}

/// With no cache ever created, the status read must report an empty, uninitialized cache without
/// creating the store as a side effect — and the mutating routes must refuse rather than create it.
#[tokio::test]
async fn an_uninitialized_cache_reports_empty_and_never_creates_the_store() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    std::fs::create_dir_all(temp_dir.path().join("config/manifests")).expect("config dir creates");

    let (status, body) = request(
        create_app(test_settings(&temp_dir)).expect("app creates"),
        "GET",
        "/api/v1/model-cache",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["initialized"], json!(false));
    assert_eq!(body["entryCount"], json!(0));
    assert_eq!(body["usedBytes"], json!(0));
    assert_eq!(body["policy"]["enabled"], json!(false));

    let (status, _) = request(
        create_app(test_settings(&temp_dir)).expect("app creates"),
        "POST",
        "/api/v1/model-cache/pin",
        json!({ "cacheKey": format!("sha256:{}", "b".repeat(64)), "pinned": true }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        !temp_dir
            .path()
            .join("data")
            .join("models")
            .join("resolved")
            .exists(),
        "a refused control must not materialize a cache root"
    );
}

/// Session slots a store handed out, by directory name. `ResolvedCacheStore::open` creates one
/// `sessions/<id>/` per handle and holds its lock for the handle's life; only a later `recover()`
/// reclaims the slots of processes that are gone.
fn session_slots(store: &ResolvedCacheStore) -> Vec<String> {
    let mut slots = std::fs::read_dir(store.root().join("sessions"))
        .expect("sessions dir reads")
        .filter_map(|item| {
            let item = item.expect("session entry reads");
            item.file_type()
                .expect("session file type")
                .is_dir()
                .then(|| item.file_name().to_string_lossy().into_owned())
        })
        .collect::<Vec<_>>();
    slots.sort();
    slots
}

/// Every control POST goes through ONE session for the API process, not one per request.
///
/// `ResolvedCacheStore::open` claims a session slot and holds its exclusive lock for the handle's
/// lifetime; the store's design is that a slot belongs to a process, and stale slots are reclaimed
/// by a later `recover()`. An API that opened a store per request would therefore strand a fresh
/// directory and lock file on every keep/preview/remove the UI performs — debris that accumulates
/// for as long as the app runs. Driving the SAME router (and therefore the same `AppState`) across
/// several actions is what makes that visible.
#[tokio::test]
async fn control_actions_share_one_cache_session_for_the_api_process() {
    let fixture = cache_fixture("q8");
    let fixture_slots = session_slots(&fixture.store);
    assert_eq!(
        fixture_slots.len(),
        1,
        "the fixture's own store holds exactly one session"
    );

    let app = create_app(test_settings(&fixture.temp_dir)).expect("app creates");
    for pinned in [true, false] {
        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/v1/model-cache/pin",
            json!({ "cacheKey": fixture.cache_key, "pinned": pinned }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/v1/model-cache/removal-preview",
            json!({ "cacheKey": fixture.cache_key }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/model-cache/remove",
        json!({ "cacheKey": fixture.cache_key }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let slots = session_slots(&fixture.store);
    assert_eq!(
        slots.len(),
        fixture_slots.len() + 1,
        "five control actions must leave ONE api session behind, not one per request; found {slots:?}"
    );
    // And it is a genuine, still-locked runtime session rather than a slot the API deleted behind
    // itself: the whole point is that the store's session semantics are unchanged, only reused.
    let api_slot = slots
        .iter()
        .find(|slot| !fixture_slots.contains(slot))
        .expect("the api session slot exists");
    assert!(
        fixture
            .store
            .root()
            .join("sessions")
            .join(format!("{api_slot}.lock"))
            .is_file(),
        "the api session must still hold its own session lock file"
    );
}

/// A store fault is a 500 and a caller's mistake is a 400. Reporting every failure as a client
/// error tells an operator to fix their request when the machine is what is broken, and hides a
/// real fault from anything watching 5xx rates.
#[tokio::test]
async fn store_faults_answer_500_while_a_bad_request_stays_a_client_error() {
    let fixture = cache_fixture("q8");
    let app = create_app(test_settings(&fixture.temp_dir)).expect("app creates");

    // Caller's mistake #1: a key that is not a cache key at all.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/model-cache/removal-preview",
        json!({ "cacheKey": "not-a-cache-key" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["detail"].as_str().expect("detail").contains("sha256:"));

    // Caller's mistake #2: a well-formed key for an entry this store does not hold.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/model-cache/pin",
        json!({ "cacheKey": format!("sha256:{}", "c".repeat(64)), "pinned": true }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Caller's mistake #3: a refusal from an intact store. The entry is pinned, so removal is
    // declined — nothing is broken, and the reply tells the caller what to change. This is the
    // guard against over-correcting: an implementation that answered 500 for every store `Err`
    // would turn ordinary refusals into fake server faults.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/model-cache/pin",
        json!({ "cacheKey": fixture.cache_key, "pinned": true }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/model-cache/remove",
        json!({ "cacheKey": fixture.cache_key }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["detail"].as_str().expect("detail").contains("pinned"));

    // Now a genuine store fault on a key the caller named correctly: both journal slots are
    // unreadable, so the store cannot answer at all. Nothing about the request is wrong, and a
    // 4xx here would send an operator to fix their request while real damage sits on the host.
    for slot in 0..=1 {
        std::fs::write(
            fixture.entry.join(format!("metadata.{slot}.json")),
            b"not-json",
        )
        .expect("journal garbage writes");
    }
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/model-cache/pin",
        json!({ "cacheKey": fixture.cache_key, "pinned": false }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "a damaged journal is the host's fault, not the caller's"
    );
}

/// The effective policy the API reports is the one THIS process is running under, so the UI can
/// tell a persisted-but-not-yet-applied setting from a live one.
#[tokio::test]
async fn status_reports_the_policy_this_process_is_running_under() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    std::fs::create_dir_all(temp_dir.path().join("config/manifests")).expect("config dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.resolved_cache =
        sceneworks_core::model_artifacts::resolved_cache::ResolvedCachePolicy {
            enabled: true,
            max_bytes: 32 * 1024 * 1024 * 1024,
            inactivity_seconds: 7 * 24 * 60 * 60,
        };

    let (status, body) = request(
        create_app(settings).expect("app creates"),
        "GET",
        "/api/v1/model-cache",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["policy"]["enabled"], json!(true));
    assert_eq!(
        body["policy"]["maxBytes"],
        json!(32 * 1024 * 1024 * 1024_u64)
    );
    assert_eq!(body["policy"]["inactivitySeconds"], json!(604_800));
}
