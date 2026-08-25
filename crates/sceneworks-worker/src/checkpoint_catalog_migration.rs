//! One-time migration of PRE-EPIC imported catalog entries onto the checkpoint plan store
//! (epic 20398, sc-20651).
//!
//! # What is migrated
//!
//! A model imported before sc-20636 is a `user.models.jsonc` entry whose `paths.model` names
//! `<data>/models/imports/<name>` — which is byte-for-byte
//! [`CheckpointPlanStore::installs_root`]`/<name>`. Such an entry can therefore be compiled by
//! [`CheckpointPlanStore::compile_managed`] IN PLACE: no approved root, no copy, no rename. The
//! only write to the entry is an `importPlan` stamp; its `id`, its `paths.model`, and every other
//! field are left exactly as they were, so no saved workflow and no model id changes and the
//! family's bespoke lane (keyed on the loadable path) keeps serving every request shape the plan
//! route declines.
//!
//! # Why a background pass at worker startup
//!
//! The mechanism was chosen against two alternatives:
//!
//! * **Not inline on a render request.** Compilation is genuinely expensive:
//!   `inspect_checkpoint` streams every artifact byte through SHA-256 three times and reads them
//!   once more for the per-layer source stamps, so a first render of a migrated model would stall
//!   for roughly four passes over a multi-gigabyte checkpoint before any pixels moved.
//! * **Not readiness-critical at API boot.** For the same reason: the API must answer
//!   `/models` immediately, and a catalog with N legacy imports would delay it by 4N checkpoint
//!   reads. The pass is spawned at worker startup and NEVER awaited (each compile runs on a
//!   blocking thread), and a worker that has not finished it — or crashed part-way through it — is
//!   fully functional: every unmigrated entry still routes through the bespoke lane it always did.
//!
//! # Idempotency and the completion record
//!
//! There is no ledger and no marker file: the completion record is the per-entry
//! `importPlan.checkpointId` itself, exactly as in
//! [`sceneworks_core::credentials::migrate_legacy_store`] — idempotency comes from the state
//! itself. A migrated entry is skipped on the next boot for free (a map lookup, no I/O); a failed
//! entry is left completely untouched and simply retried next boot.

use super::*;

use sceneworks_core::checkpoint_import::ManagedProvenanceV1;
use sceneworks_core::checkpoint_plan_store::CheckpointPlanStore;
use sceneworks_core::jobs_store::{checkpoint_plan_checkpoint_id, imported_entry_loadable_path};

/// The ingest source a pre-epic import is recorded as. The legacy job wrote its bytes into the
/// install directory itself from a download, an upload, or a user-selected file, and the entry
/// retains no discriminator that could tell those apart — so the provenance says only what is
/// actually known: the bytes are an application-owned local copy.
const LEGACY_IMPORT_PROVENANCE_SOURCE: &str = "local-copy";

/// What one migration pass did. Returned rather than logged-only so the pass is testable without a
/// filesystem side channel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CatalogMigrationSummary {
    /// Entries a compile was actually run for. Always `migrated + failed()`.
    pub attempted: usize,
    /// Entries whose `importPlan` stamp landed in the manifest.
    pub migrated: usize,
    /// Entries no compile was run for: already plan-backed, or not addressing a managed install
    /// directory in the single-file shape a pre-epic import has.
    pub skipped: usize,
    /// One VERBATIM diagnostic per entry that refused, `(modelId, message)`. The store's messages
    /// already begin `[checkpoint-plan:<code>]`, and the code is the only thing that distinguishes
    /// "the user deleted our copy" from "these bytes are not a runnable checkpoint" — flattening
    /// them to a count, or to "migration failed", would throw that away. The failed entry itself is
    /// left byte-identical.
    pub failures: Vec<(String, String)>,
    /// `(modelId, container)` per entry left on its bespoke lane because the plan route has no
    /// loader for its container. A DISTINCT outcome from a failure: the checkpoint is fine and the
    /// model still renders — conflating the two would make a healthy catalog look broken and would
    /// make a real refusal harder to find.
    pub declined_containers: Vec<(String, String)>,
}

impl CatalogMigrationSummary {
    pub fn failed(&self) -> usize {
        self.failures.len()
    }
}

/// Run one migration pass over `<config_dir>/manifests/user.models.jsonc`.
///
/// Never returns `Err` for a per-entry refusal — a single bad entry must not abort the pass for the
/// rest of the catalog — only for a catalog that cannot be read at all.
pub(crate) async fn migrate_legacy_checkpoint_catalog(
    config_dir: &Path,
    data_dir: &Path,
) -> WorkerResult<CatalogMigrationSummary> {
    let manifest_path = config_dir.join("manifests").join("user.models.jsonc");
    let entries = match read_user_model_entries(&manifest_path).await? {
        Some(entries) => entries,
        // No user catalog yet: a fresh install has nothing to migrate.
        None => return Ok(CatalogMigrationSummary::default()),
    };
    let store = CheckpointPlanStore::open(data_dir);
    let mut summary = CatalogMigrationSummary::default();
    // Sequential on purpose: each compile is ~4 passes over a whole checkpoint, and running them
    // concurrently would only make them contend for the same disk.
    for entry in entries {
        let Some(candidate) = migration_candidate(&store, &entry) else {
            summary.skipped += 1;
            continue;
        };
        // The plan-driven route has no loader for a non-safetensors container, and (E8) refuses one
        // rather than handing GGUF bytes to the safetensors loader. Stamping such an entry would
        // therefore move a model that renders TODAY through its family's bespoke lane onto a route
        // that declines it — a regression the migration itself would have introduced. Declined, not
        // failed: nothing is wrong with the checkpoint, and it keeps the lane that serves it.
        if let Some(container) = declined_container(&store, &candidate) {
            tracing::info!(
                event = "checkpoint_catalog_migration_declined",
                modelId = %candidate.model_id,
                installId = %candidate.install_id,
                container = %container,
                "left a pre-epic import on its bespoke lane: the plan route has no loader for its container"
            );
            summary
                .declined_containers
                .push((candidate.model_id.clone(), container));
            continue;
        }
        summary.attempted += 1;
        match compile_candidate(&store, &candidate).await {
            Ok(stamp) => {
                let mut update = JsonObject::new();
                update.insert("id".to_owned(), Value::String(candidate.model_id.clone()));
                update.insert("importPlan".to_owned(), stamp);
                match upsert_manifest_entry(&manifest_path, "models", update).await {
                    Ok(()) => summary.migrated += 1,
                    Err(error) => {
                        tracing::warn!(
                            event = "checkpoint_catalog_migration_failed",
                            modelId = %candidate.model_id,
                            error = %error,
                            "could not stamp the migrated checkpoint plan onto the catalog entry"
                        );
                        summary
                            .failures
                            .push((candidate.model_id.clone(), error.to_string()));
                    }
                }
            }
            Err(error) => {
                // The store's own `Display` already carries `[checkpoint-plan:<code>] …`; it is
                // surfaced verbatim so the code that identifies the refusal survives to the log.
                tracing::warn!(
                    event = "checkpoint_catalog_migration_failed",
                    modelId = %candidate.model_id,
                    installId = %candidate.install_id,
                    error = %error,
                    "could not compile a pre-epic imported model into the checkpoint plan store"
                );
                summary
                    .failures
                    .push((candidate.model_id.clone(), error.to_string()));
            }
        }
    }
    Ok(summary)
}

/// The `models` array of the user catalog, or `None` when the catalog does not exist yet.
async fn read_user_model_entries(path: &Path) -> WorkerResult<Option<Vec<JsonObject>>> {
    let payload = match tokio::fs::read_to_string(path).await {
        Ok(payload) => payload,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let manifest: Value = serde_json::from_str(&strip_jsonc_comments(&payload))?;
    let entries = manifest
        .get("models")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(Value::as_object)
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(Some(entries))
}

/// A pre-epic entry the pass will attempt to compile.
struct MigrationCandidate {
    model_id: String,
    install_id: String,
    relative_path: String,
}

/// Decide whether `entry` is a pre-epic managed import, doing NO expensive I/O.
///
/// `None` for every entry the pass leaves alone, including the already-migrated case, which costs a
/// single map lookup and no filesystem access at all.
fn migration_candidate(
    store: &CheckpointPlanStore,
    entry: &JsonObject,
) -> Option<MigrationCandidate> {
    // Already plan-backed: the completion record IS this key.
    if checkpoint_plan_checkpoint_id(entry).is_some() {
        return None;
    }
    // The manifest upsert is keyed on `id`; an entry without one cannot be written back, so it is
    // not a candidate rather than a failure.
    let model_id = entry
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())?
        .to_owned();
    // The path a bespoke lane would LOAD from — deliberately not the wider installed-path reading,
    // which accepts a provenance-only breadcrumb naming bytes this entry does not own.
    let declared = imported_entry_loadable_path(entry)?;
    let install_path = crate::paths::normalize_existing_or_absolute(Path::new(declared)).ok()?;
    if !install_path.is_dir() {
        return None;
    }
    // The install id is the directory's own name, and the store's own `install_dir` must reproduce
    // the very path we started from. This is the proof the id addresses the intended directory
    // (`model_jobs::managed_install_id_for_target` does exactly this); a lexical `starts_with`
    // would reject a legitimate install on macOS, where a data dir under `/var` canonicalizes to
    // `/private/var`.
    let install_id = install_path.file_name().and_then(|name| name.to_str())?;
    let derived = store.install_dir(install_id).ok()?;
    let derived = crate::paths::normalize_existing_or_absolute(&derived).ok()?;
    if derived != install_path {
        return None;
    }
    let relative_path = lone_top_level_weight_file(&install_path)?;
    // Validated the way the store validates it, so a name that could never address a layer is not
    // paid for with a full checkpoint read.
    sceneworks_core::checkpoint_plan_store::validate_linked_relative_path(&relative_path).ok()?;
    Some(MigrationCandidate {
        model_id,
        install_id: install_id.to_owned(),
        relative_path,
    })
}

/// The file name of the LONE top-level weight file in `dir`, or `None` when there is not exactly
/// one.
///
/// This is the shape the pre-epic import job wrote and the shape the bespoke imported lanes load
/// (`krea_imported::imported_dit_file`): the checkpoint plus an install marker. Requiring exactly
/// one is what makes the compiled plan's primary provably the same file the legacy lane loads,
/// rather than a guess among several.
///
/// BOTH weight containers count here, matching the inspector's own `is_weight_extension`. A GGUF
/// import must be RECOGNIZED so [`declined_container`] can decline it as the typed outcome it is;
/// leaving it out would silently fold it into the "not a managed install" skip and lose the
/// distinction.
fn lone_top_level_weight_file(dir: &Path) -> Option<String> {
    let mut found: Option<String> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let is_weights = path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                ext.eq_ignore_ascii_case("safetensors") || ext.eq_ignore_ascii_case("gguf")
            });
        if !is_weights {
            continue;
        }
        let name = path.file_name().and_then(|name| name.to_str())?.to_owned();
        if found.is_some() {
            return None;
        }
        found = Some(name);
    }
    found
}

/// The container name when `candidate`'s primary is NOT safetensors, `None` when it is (or when the
/// verdict cannot be reached, in which case the compile below is left to refuse on its own terms).
///
/// The verdict is the INSPECTOR'S, taken from `discover_checkpoint` — the same discovery pass the
/// compile itself runs, and the same header read whose answer becomes `ImportLayerV1::container` on
/// the compiled primary. It is never the file extension, which is exactly the thing a silent
/// substitution lies about.
///
/// Read here rather than off the compiled plan's primary layer for one reason: nothing is ever
/// stamped for a declined container, so the decline RECURS on every boot. Discovery reads only
/// container descriptors, so it costs one header read; deciding after the compile would cost four
/// full passes over the checkpoint, every boot, forever. The two verdicts cannot disagree for the
/// shape this pass migrates — a single-file install has exactly one candidate, and it is the file
/// that becomes the primary layer.
fn declined_container(
    store: &CheckpointPlanStore,
    candidate: &MigrationCandidate,
) -> Option<String> {
    let install_path = store.resolve_install(&candidate.install_id).ok()?;
    let request = sceneworks_core::checkpoint_inspector::CheckpointInspectionRequestV1::managed(
        sceneworks_core::checkpoint_plan_store::managed_checkpoint_id(&candidate.install_id),
        install_path,
        &candidate.relative_path,
        &candidate.install_id,
        ManagedProvenanceV1 {
            source: LEGACY_IMPORT_PROVENANCE_SOURCE.to_owned(),
            ..Default::default()
        },
    )
    .ok()?;
    let discovery = sceneworks_core::checkpoint_inspector::discover_checkpoint(&request);
    // The primary is the candidate discovery found for the very file this migration selected.
    let primary = discovery
        .candidates
        .iter()
        .find(|found| found.relative_path == candidate.relative_path)?;
    match primary.container {
        sceneworks_core::checkpoint_inspector::CheckpointContainerV1::Safetensors => None,
        other => Some(format!("{other:?}").to_ascii_lowercase()),
    }
}

/// Compile `candidate` in place and build the `importPlan` value to stamp.
async fn compile_candidate(
    store: &CheckpointPlanStore,
    candidate: &MigrationCandidate,
) -> Result<Value, sceneworks_core::checkpoint_plan_store::CheckpointPlanError> {
    let store = store.clone();
    let install_id = candidate.install_id.clone();
    let relative_path = candidate.relative_path.clone();
    // Four full reads of the checkpoint: off the async runtime.
    let compiled = tokio::task::spawn_blocking(move || {
        store.compile_managed(
            &install_id,
            &relative_path,
            ManagedProvenanceV1 {
                source: LEGACY_IMPORT_PROVENANCE_SOURCE.to_owned(),
                ..Default::default()
            },
        )
    })
    .await
    .map_err(|error| {
        sceneworks_core::checkpoint_plan_store::CheckpointPlanError::Corrupt {
            what: "checkpoint compile task".to_owned(),
            message: error.to_string(),
        }
    })??;
    Ok(json!({
        "checkpointId": compiled.checkpoint_id,
        "planId": compiled.plan.plan_id,
        "semanticDigest": compiled.record.plan.semantic_digest,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    /// A minimal but REAL safetensors file: the same header-length + JSON header + payload encoding
    /// the production inspector parses. The migration is driven through
    /// `CheckpointPlanStore::compile_managed`, which runs the real `inspect_checkpoint`, so a mock
    /// checkpoint would prove nothing about whether a legacy install actually compiles.
    fn write_tiny_safetensors(path: &Path, entries: &[(&str, &str)], fill: u8) {
        let mut header = serde_json::Map::new();
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
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// A minimal but REAL GGUF the production inspector accepts: v3 magic, one `general.architecture`
    /// / `general.alignment` metadata pair and one aligned F32 tensor. Ported verbatim from
    /// `sceneworks-core/tests/checkpoint_inspector.rs::write_tiny_gguf` (that file is a test target,
    /// so it cannot be imported) — nothing is weakened; the inspector really does classify this as
    /// `CheckpointContainerV1::Gguf`, which the test below asserts by way of the decline.
    fn write_tiny_gguf(path: &Path, architecture: &str) {
        fn push_u32(bytes: &mut Vec<u8>, value: u32) {
            bytes.extend(value.to_le_bytes());
        }
        fn push_u64(bytes: &mut Vec<u8>, value: u64) {
            bytes.extend(value.to_le_bytes());
        }
        fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
            push_u64(bytes, value.len() as u64);
            bytes.extend(value.as_bytes());
        }
        let mut bytes = b"GGUF".to_vec();
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 1); // tensor count
        push_u64(&mut bytes, 2); // metadata count
        push_gguf_string(&mut bytes, "general.architecture");
        push_u32(&mut bytes, 8); // string
        push_gguf_string(&mut bytes, architecture);
        push_gguf_string(&mut bytes, "general.alignment");
        push_u32(&mut bytes, 4); // u32
        push_u32(&mut bytes, 32);
        push_gguf_string(&mut bytes, "model.weight");
        push_u32(&mut bytes, 1); // dimensions
        push_u64(&mut bytes, 1);
        push_u32(&mut bytes, 0); // F32
        push_u64(&mut bytes, 0); // relative data offset
        let aligned = bytes.len().div_ceil(32) * 32;
        bytes.resize(aligned, 0);
        bytes.extend(1_f32.to_le_bytes());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// The tensor surface a native Krea 2 transformer file is recognized by — the same surface the
    /// image-job plan-route fixtures compile.
    fn krea_native_entries() -> Vec<(&'static str, &'static str)> {
        vec![
            ("model.diffusion_model.txtfusion.projector.weight", "BF16"),
            ("model.diffusion_model.blocks.0.attn.wq.weight", "BF16"),
            ("model.diffusion_model.first.weight", "BF16"),
        ]
    }

    struct MigrationFixture {
        _temp: TempDir,
        config_dir: PathBuf,
        data_dir: PathBuf,
    }

    impl MigrationFixture {
        fn new() -> Self {
            let temp = TempDir::new().unwrap();
            let config_dir = temp.path().join("config");
            let data_dir = temp.path().join("data");
            std::fs::create_dir_all(config_dir.join("manifests")).unwrap();
            std::fs::create_dir_all(&data_dir).unwrap();
            Self {
                _temp: temp,
                config_dir,
                data_dir,
            }
        }

        fn manifest_path(&self) -> PathBuf {
            self.config_dir.join("manifests").join("user.models.jsonc")
        }

        /// Lay down a pre-epic import exactly as the legacy job did: a directory that IS
        /// `installs_root()/<name>` holding one weight file, plus a catalog entry naming it in
        /// `paths.model` and carrying no `importPlan` key at all.
        fn install(&self, name: &str, entries: &[(&str, &str)]) -> PathBuf {
            let install = self.data_dir.join("models").join("imports").join(name);
            write_tiny_safetensors(&install.join(format!("{name}.safetensors")), entries, 0x5a);
            install
        }

        fn write_catalog(&self, entries: Value) {
            std::fs::write(
                self.manifest_path(),
                serde_json::to_vec_pretty(&json!({"schemaVersion": 1, "models": entries})).unwrap(),
            )
            .unwrap();
        }

        fn catalog_entry(&self, id: &str) -> JsonObject {
            let payload = std::fs::read_to_string(self.manifest_path()).unwrap();
            let manifest: Value = serde_json::from_str(&strip_jsonc_comments(&payload)).unwrap();
            manifest["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["id"] == json!(id))
                .unwrap_or_else(|| panic!("the catalog must still contain {id:?}"))
                .as_object()
                .unwrap()
                .clone()
        }

        async fn migrate(&self) -> CatalogMigrationSummary {
            migrate_legacy_checkpoint_catalog(&self.config_dir, &self.data_dir)
                .await
                .expect("the migration pass must not abort on a readable catalog")
        }
    }

    /// AC: the stamp is ADDITIVE. A migrated entry keeps its `id` and its `paths.model` byte for
    /// byte — that is the whole no-model-ID / no-saved-workflow-breakage promise — and `importPlan`
    /// is the ONLY key that appears.
    ///
    /// Failing mutation (run): make the stamp write `paths` as well, by adding
    /// `update.insert("paths".to_owned(), json!({}));` beside the `importPlan` insert.
    #[tokio::test]
    async fn migration_stamps_only_import_plan_and_leaves_every_other_field_byte_identical() {
        let fx = MigrationFixture::new();
        let install = fx.install("legacy-kreamania", &krea_native_entries());
        fx.write_catalog(json!([{
            "id": "imported_kreamania",
            "name": "Kreamania",
            "type": "image",
            "catalogScope": "user",
            "family": "krea_2",
            "importSourceShape": "transformer_file",
            "paths": { "model": install.to_str().unwrap() }
        }]));
        let before = fx.catalog_entry("imported_kreamania");
        assert!(
            !before.contains_key("importPlan"),
            "fixture check: this is the pre-epic shape — no importPlan key at all"
        );

        let summary = fx.migrate().await;
        assert_eq!(
            (summary.attempted, summary.migrated, summary.failed()),
            (1, 1, 0),
            "a legacy single-file install under installs_root must compile: {:?}",
            summary.failures
        );

        let after = fx.catalog_entry("imported_kreamania");
        let checkpoint_id = checkpoint_plan_checkpoint_id(&after)
            .expect("the migrated entry carries a checkpoint identity");
        assert!(
            !checkpoint_id.is_empty(),
            "the stamped checkpoint id must be a real identity"
        );
        let mut stripped = after.clone();
        stripped.remove("importPlan");
        assert_eq!(
            stripped, before,
            "importPlan must be the ONLY key the migration adds or changes"
        );
    }

    /// AC: idempotent. The completion record is the per-entry `importPlan.checkpointId` itself, so
    /// a second boot re-reads the catalog and attempts ZERO compiles — the assertion is on
    /// `attempted`, not on the resulting file, because a pass that recompiled and rewrote the same
    /// stamp would also leave the file identical while costing four full checkpoint reads.
    ///
    /// Failing mutation (run): delete the
    /// `if checkpoint_plan_checkpoint_id(entry).is_some() { return None; }` guard in
    /// `migration_candidate`.
    #[tokio::test]
    async fn a_second_migration_pass_attempts_no_compiles_and_rewrites_nothing() {
        let fx = MigrationFixture::new();
        let install = fx.install("legacy-kreamania", &krea_native_entries());
        fx.write_catalog(json!([{
            "id": "imported_kreamania",
            "catalogScope": "user",
            "family": "krea_2",
            "paths": { "model": install.to_str().unwrap() }
        }]));

        let first = fx.migrate().await;
        assert_eq!(first.migrated, 1, "{:?}", first.failures);
        let after_first = std::fs::read(fx.manifest_path()).unwrap();

        let second = fx.migrate().await;
        assert_eq!(
            (second.attempted, second.migrated, second.failed()),
            (0, 0, 0),
            "an already plan-backed entry must cost no compile at all"
        );
        assert_eq!(second.skipped, 1);
        assert_eq!(
            std::fs::read(fx.manifest_path()).unwrap(),
            after_first,
            "the second pass must not rewrite the catalog"
        );
    }

    /// AC: a refused compile leaves the entry EXACTLY as it was — the whole entry JSON, not merely
    /// the absence of `importPlan` — and the recorded diagnostic is the store's own typed message.
    ///
    /// The failure is a real one: an install directory holding a real safetensors file whose tensor
    /// surface matches no family the inspector recognizes, so `compile_managed` refuses with
    /// `UnrunnableSource` rather than being mocked into failing.
    ///
    /// Failing mutation (run): in `migrate_legacy_checkpoint_catalog`, replace the pushed
    /// `error.to_string()` with the flattened `"migration failed".to_owned()`.
    #[tokio::test]
    async fn a_refused_compile_leaves_the_entry_untouched_and_records_the_typed_diagnostic() {
        let fx = MigrationFixture::new();
        let install = fx.install("legacy-garbage", &[("not.a.recognized.tensor.name", "F32")]);
        fx.write_catalog(json!([{
            "id": "imported_garbage",
            "catalogScope": "user",
            "family": "krea_2",
            "importSourceShape": "transformer_file",
            "paths": { "model": install.to_str().unwrap() }
        }]));
        let before = fx.catalog_entry("imported_garbage");

        let summary = fx.migrate().await;
        assert_eq!(
            (summary.attempted, summary.migrated, summary.failed()),
            (1, 0, 1),
            "premise: the compile must be ATTEMPTED and must refuse, or this row is vacuous"
        );
        let (model_id, diagnostic) = &summary.failures[0];
        assert_eq!(model_id, "imported_garbage");
        assert!(
            diagnostic.starts_with("[checkpoint-plan:"),
            "the store's typed code must survive verbatim, got: {diagnostic}"
        );

        assert_eq!(
            fx.catalog_entry("imported_garbage"),
            before,
            "a failed entry must be left byte-identical, keeping its bespoke lane"
        );
    }

    /// AC (E8 interaction): a legacy GGUF-backed entry is DECLINED, not stamped and not failed.
    ///
    /// The plan route refuses a non-safetensors primary rather than handing GGUF bytes to the
    /// safetensors loader, so stamping such an entry would move a model that renders today onto a
    /// route that declines it — a regression the migration itself would have caused. The entry must
    /// come out byte-identical, and the outcome must be reported as its own thing: a healthy
    /// checkpoint on the lane that serves it, not a failure.
    ///
    /// Failing mutation (run): delete the `if let Some(container) = declined_container(...)` block
    /// from `migrate_legacy_checkpoint_catalog`.
    #[tokio::test]
    async fn a_legacy_gguf_backed_entry_is_declined_untouched_rather_than_stamped() {
        let fx = MigrationFixture::new();
        let install = fx
            .data_dir
            .join("models")
            .join("imports")
            .join("legacy-fluxmania");
        write_tiny_gguf(&install.join("fluxmania.gguf"), "flux");
        fx.write_catalog(json!([{
            "id": "imported_fluxmania",
            "name": "Fluxmania",
            "type": "image",
            "catalogScope": "user",
            "family": "flux_2",
            "importSourceShape": "transformer_file",
            "paths": { "model": install.to_str().unwrap() }
        }]));
        let before = fx.catalog_entry("imported_fluxmania");

        let summary = fx.migrate().await;
        assert_eq!(
            summary.declined_containers,
            vec![("imported_fluxmania".to_owned(), "gguf".to_owned())],
            "the GGUF entry must be reported as a declined container: {summary:?}"
        );
        assert_eq!(
            (summary.attempted, summary.migrated, summary.failed()),
            (0, 0, 0),
            "a declined container is neither an attempted compile nor a failure"
        );
        assert_eq!(
            summary.skipped, 0,
            "premise: the entry WAS recognized as a managed install — otherwise the decline is \
             indistinguishable from never having looked at it"
        );
        assert_eq!(
            fx.catalog_entry("imported_fluxmania"),
            before,
            "a declined entry must be left byte-identical, keeping the bespoke lane that loads it"
        );
    }

    /// A catalog entry whose `paths.model` is NOT the store's own install directory is not a
    /// managed install and is never compiled, however plausible the path looks. The proof is the
    /// store's `install_dir` round-trip, not a lexical prefix test.
    ///
    /// Failing mutation (run): replace the `derived != install_path` rejection in
    /// `migration_candidate` with a normalized lexical prefix test,
    /// `!install_path.starts_with(normalize_existing_or_absolute(store.installs_root())?)`.
    #[tokio::test]
    async fn a_nested_install_path_is_not_a_managed_install_and_is_never_compiled() {
        let fx = MigrationFixture::new();
        // A GRANDCHILD of installs_root: lexically "inside" the managed root, but its own name
        // (`inner`) addresses `installs_root()/inner`, which is a different directory.
        let nested = fx
            .data_dir
            .join("models")
            .join("imports")
            .join("outer")
            .join("inner");
        write_tiny_safetensors(
            &nested.join("kreamania.safetensors"),
            &krea_native_entries(),
            0x5a,
        );
        fx.write_catalog(json!([{
            "id": "imported_nested",
            "catalogScope": "user",
            "family": "krea_2",
            "paths": { "model": nested.to_str().unwrap() }
        }]));

        let summary = fx.migrate().await;
        assert_eq!(
            (summary.attempted, summary.skipped, summary.failed()),
            (0, 1, 0),
            "a path that is merely under the managed root is not an install id"
        );
        assert!(
            !fx.catalog_entry("imported_nested")
                .contains_key("importPlan"),
            "nothing may be stamped onto an entry no compile was run for"
        );
    }
}
