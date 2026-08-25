//! Transactional managed ingestion (sc-20636): the one staging → validation → atomic-finalize path
//! every managed source shares, its fault matrix (cancel, crash, disk-full, hash mismatch,
//! unrunnable content), linked/managed semantic parity with duplicate reporting, and
//! ownership-safe removal.
//!
//! Every fault case asserts the SAME three things, because they are the invariant: no install
//! directory, no catalog record, no plan document. A partial install that is merely "not selected"
//! is still a partial install; these tests refuse to accept one existing at all.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use tempfile::TempDir;

use sceneworks_core::checkpoint_import::{ManagedProvenanceV1, SourceLocatorV1};
use sceneworks_core::checkpoint_ingest::{
    sanitize_provenance_url, sweep_staging, ManagedIngest, ManagedIngestError,
};
// Its only consumer is `the_active_staging_set_keeps_a_live_transfer_and_releases_a_crash_orphan`,
// which is `#[cfg(unix)]` for the mtime backdating that separates a live transfer from an orphan.
// Ungated, the import is dead on Windows and `cargo clippy --all-targets -- -D warnings` fails
// there — a break no CI lane sees, because no lane runs clippy on Windows.
#[cfg(unix)]
use sceneworks_core::checkpoint_ingest::active_staging_ids;
use sceneworks_core::checkpoint_plan_store::{
    linked_checkpoint_id, managed_checkpoint_id, CheckpointPlanError, CheckpointPlanStore,
    BINDINGS_DIR, PLANS_DIR,
};

const KREA_FILE: &str = "kreamania.safetensors";

fn fixture_dir(label: &str) -> TempDir {
    tempfile::Builder::new()
        .prefix(&format!("ingest-{label}-{}-", std::process::id()))
        .tempdir()
        .unwrap()
}

/// A minimal single-file Krea 2 native DiT: the `txtfusion.` marker the family detector keys on,
/// every tensor dense bf16, bytes deterministic. Same shape the plan-store suite uses, so the two
/// suites compile the same checkpoint through the two ownership modes.
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
    library_dir: PathBuf,
}

fn fixture(label: &str) -> Fixture {
    let data = fixture_dir(&format!("{label}-data"));
    let library = fixture_dir(&format!("{label}-library"));
    let library_dir = fs::canonicalize(library.path()).unwrap();
    Fixture {
        store: CheckpointPlanStore::open(data.path()),
        library_dir,
        _data: data,
        _library: library,
    }
}

impl Fixture {
    /// The user's own copy of the checkpoint, outside anything SceneWorks owns.
    fn user_file(&self) -> PathBuf {
        let path = self.library_dir.join(KREA_FILE);
        if !path.is_file() {
            write_krea_native_file(&path, 0x21);
        }
        path
    }

    fn staging_root(&self) -> PathBuf {
        self.store.staging_root().to_path_buf()
    }

    fn plan_documents(&self) -> Vec<String> {
        let mut names = Vec::new();
        for dir in [PLANS_DIR, BINDINGS_DIR] {
            if let Ok(entries) = fs::read_dir(self.store.root().join(dir)) {
                for entry in entries.flatten() {
                    names.push(format!("{dir}/{}", entry.file_name().to_string_lossy()));
                }
            }
        }
        names.sort();
        names
    }

    /// The invariant every refusal must satisfy: nothing under `installs/`, no catalog record, no
    /// plan or bindings document. Asserted as a whole so a fault that leaves any ONE of them fails.
    fn assert_no_trace_of(&self, install_id: &str, context: &str) {
        let install_dir = self.store.install_dir(install_id).unwrap();
        assert!(
            !install_dir.exists(),
            "{context}: a partial install survived at {}",
            install_dir.display()
        );
        assert!(
            !self.store.installs_root().join(install_id).exists(),
            "{context}: an install directory survived"
        );
        assert!(
            self.store
                .record(&managed_checkpoint_id(install_id))
                .is_err(),
            "{context}: a catalog record survived"
        );
        assert_eq!(
            self.store.inventory().unwrap().records.len(),
            0,
            "{context}: the inventory is not empty"
        );
        assert!(
            self.plan_documents().is_empty(),
            "{context}: plan/bindings documents survived: {:?}",
            self.plan_documents()
        );
        assert!(
            self.store
                .resolve(&managed_checkpoint_id(install_id))
                .is_err(),
            "{context}: the checkpoint still resolves"
        );
    }
}

fn civitai_provenance() -> ManagedProvenanceV1 {
    ManagedProvenanceV1 {
        source: "civitai".to_owned(),
        reference: Some("Kreamania".to_owned()),
        url: sanitize_provenance_url(
            "https://user:t0ken@civitai.com/api/download/models/9931?type=Model&token=s3cret",
        ),
        version_id: Some("9931".to_owned()),
        file_id: Some("40277".to_owned()),
        credential_host: Some("civitai.com".to_owned()),
    }
}

/// A reader that hands over `fail_after` bytes and then reports the error a full disk surfaces.
struct FailingReader {
    bytes: Vec<u8>,
    position: usize,
    fail_after: usize,
}

impl Read for FailingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.position >= self.fail_after {
            // `ErrorKind::StorageFull` is stable only since 1.83 and the workspace MSRV is 1.80,
            // so the ENOSPC message is carried rather than the kind. The ingest treats every
            // transfer error the same way, and the assertion is on the message.
            return Err(io::Error::other("No space left on device (ENOSPC)"));
        }
        let take = buffer.len().min(self.fail_after - self.position);
        let take = take.min(self.bytes.len() - self.position);
        if take == 0 {
            // `ErrorKind::StorageFull` is stable only since 1.83 and the workspace MSRV is 1.80,
            // so the ENOSPC message is carried rather than the kind. The ingest treats every
            // transfer error the same way, and the assertion is on the message.
            return Err(io::Error::other("No space left on device (ENOSPC)"));
        }
        buffer[..take].copy_from_slice(&self.bytes[self.position..self.position + take]);
        self.position += take;
        Ok(take)
    }
}

// ---- AC1: one transactional path, and its fault matrix -------------------------------------

#[test]
fn a_managed_ingest_commits_atomically_and_publishes_a_resolvable_plan() {
    let fixture = fixture("commit");
    let source = fixture.user_file();
    let expected = sha256_of(&source);

    let ingest = ManagedIngest::begin(&fixture.store, "install-a1", civitai_provenance()).unwrap();
    // Before the commit there is nothing under `installs/` and nothing in the inventory: the
    // staged bytes exist, and they are not an install.
    ingest.stage_copy_file(&source, KREA_FILE).unwrap();
    assert!(ingest.staging_dir().join(KREA_FILE).is_file());
    assert!(!fixture.store.install_dir("install-a1").unwrap().exists());
    assert_eq!(fixture.store.inventory().unwrap().records.len(), 0);

    let install = ingest.finalize(KREA_FILE, Some(&expected)).unwrap();

    assert_eq!(install.checkpoint_id, "managed/install-a1");
    assert_eq!(install.primary_sha256, expected);
    assert!(install.install_path.join(KREA_FILE).is_file());
    assert!(
        install
            .install_path
            .starts_with(fixture.store.installs_root()),
        "a managed install must live under the SceneWorks-owned installs tree"
    );
    // The staging directory is consumed by the rename, not copied out of.
    assert!(!fixture.staging_root().join("install-a1").exists());

    // The published plan is a MANAGED plan and it resolves and re-verifies for loading.
    let resolved = fixture.store.resolve("managed/install-a1").unwrap();
    assert_eq!(resolved.family(), "krea_2");
    assert_eq!(resolved.layers.len(), 1);
    assert_eq!(
        resolved.layers[0].path,
        fs::canonicalize(install.install_path.join(KREA_FILE)).unwrap()
    );
    assert!(matches!(
        &resolved.plan.layers[0].source,
        SourceLocatorV1::Managed { install_id, relative_path, sha256, .. }
            if install_id == "install-a1" && relative_path == KREA_FILE && sha256 == &expected
    ));

    // Provenance travelled with the plan, and carries no secret.
    let SourceLocatorV1::Managed { provenance, .. } = &resolved.plan.layers[0].source else {
        panic!("managed locator");
    };
    assert_eq!(provenance.source, "civitai");
    assert_eq!(provenance.version_id.as_deref(), Some("9931"));
    assert_eq!(provenance.file_id.as_deref(), Some("40277"));
    assert_eq!(provenance.credential_host.as_deref(), Some("civitai.com"));
    assert_eq!(
        provenance.url.as_deref(),
        Some("https://civitai.com/api/download/models/9931?type=Model")
    );
    let plan_json = fs::read_to_string(
        fixture
            .store
            .root()
            .join(PLANS_DIR)
            .join(format!("{}.json", resolved.plan.plan_id)),
    )
    .unwrap();
    assert!(
        !plan_json.contains("s3cret") && !plan_json.contains("t0ken"),
        "a persisted plan must never contain a credential: {plan_json}"
    );
}

#[test]
fn a_cancelled_ingest_leaves_no_install_no_plan_and_no_record() {
    let fixture = fixture("cancel");
    let source = fixture.user_file();

    let ingest = ManagedIngest::begin(&fixture.store, "install-c1", civitai_provenance()).unwrap();
    ingest.stage_copy_file(&source, KREA_FILE).unwrap();
    let error = ingest.cancel().expect_err("cancel refuses");

    assert_eq!(error.code(), "cancelled");
    fixture.assert_no_trace_of("install-c1", "cancel");
    assert!(!fixture.staging_root().join("install-c1").exists());
    assert!(
        source.is_file(),
        "cancel must not touch the user's own file"
    );
}

#[test]
fn an_ingest_dropped_without_finalizing_leaves_no_install_no_plan_and_no_record() {
    let fixture = fixture("drop");
    let source = fixture.user_file();

    {
        let ingest =
            ManagedIngest::begin(&fixture.store, "install-d1", civitai_provenance()).unwrap();
        ingest.stage_copy_file(&source, KREA_FILE).unwrap();
        // An early return, a `?`, or a panic all reach here.
    }

    fixture.assert_no_trace_of("install-d1", "drop before finalize");
    assert!(!fixture.staging_root().join("install-d1").exists());
}

#[test]
fn a_crashed_ingest_leaves_no_runnable_install_and_sweep_reclaims_its_staging() {
    let fixture = fixture("crash");
    let source = fixture.user_file();

    let ingest = ManagedIngest::begin(&fixture.store, "install-x1", civitai_provenance()).unwrap();
    ingest.stage_copy_file(&source, KREA_FILE).unwrap();
    // A crash runs no destructor. `forget` is exactly that: the staged bytes stay on disk and
    // nothing cleans them up.
    std::mem::forget(ingest);

    fixture.assert_no_trace_of("install-x1", "crash before finalize");
    assert!(
        fixture.staging_root().join("install-x1").is_dir(),
        "a crash leaves the staging directory behind — that is what sweep_staging is for"
    );

    // An in-flight id is never swept out from under a live session.
    assert_eq!(sweep_staging(&fixture.store, &["install-x1"]).unwrap(), 0);
    assert!(fixture.staging_root().join("install-x1").is_dir());

    assert_eq!(sweep_staging(&fixture.store, &[]).unwrap(), 1);
    assert!(!fixture.staging_root().join("install-x1").exists());
    fixture.assert_no_trace_of("install-x1", "after sweep");
    assert!(source.is_file(), "sweep must not touch the user's own file");
}

/// sc-20636 review: the sweep has a production caller now
/// (`sceneworks_worker::reclaim_import_staging`, at worker startup), and a SceneWorks install runs
/// SEVERAL worker processes against ONE data dir. So the sweep's `in_flight` set has to be
/// derivable WITHOUT owning the sessions, or worker B's startup deletes the multi-gigabyte tree
/// worker A is still downloading into.
///
/// Unix-only for the mtime backdating, which is what separates a live transfer from an orphan.
#[cfg(unix)]
#[test]
fn the_active_staging_set_keeps_a_live_transfer_and_releases_a_crash_orphan() {
    let fixture = fixture("active");
    let source = fixture.user_file();

    // A live transfer: staged bytes written just now.
    let live = ManagedIngest::begin(&fixture.store, "install-live", civitai_provenance()).unwrap();
    live.stage_copy_file(&source, KREA_FILE).unwrap();

    // A crash orphan: the same shape on disk, backdated well past any plausible transfer.
    let orphan =
        ManagedIngest::begin(&fixture.store, "install-orphan", civitai_provenance()).unwrap();
    orphan.stage_copy_file(&source, KREA_FILE).unwrap();
    let orphan_root = fixture.staging_root().join("install-orphan");
    std::mem::forget(orphan);
    backdate_tree(&orphan_root, std::time::Duration::from_secs(6 * 60 * 60));

    let within = std::time::Duration::from_secs(60 * 60);
    let mut active = active_staging_ids(&fixture.store, within);
    active.sort();
    assert_eq!(
        active,
        vec!["install-live".to_owned()],
        "a tree written moments ago is a live transfer; a backdated one is a crash orphan"
    );

    // ...and that is exactly the set the sweep must be handed: the orphan goes, the live tree stays.
    let in_flight: Vec<&str> = active.iter().map(String::as_str).collect();
    assert_eq!(sweep_staging(&fixture.store, &in_flight).unwrap(), 1);
    assert!(
        fixture.staging_root().join("install-live").is_dir(),
        "the sweep must never delete another process's in-flight staging tree"
    );
    assert!(!orphan_root.exists(), "the crash orphan must be reclaimed");
    assert!(live.staging_dir().join(KREA_FILE).is_file());

    // A tree whose newest byte is inside the window is live even if its ROOT is old: a long
    // single-file download bumps the file, never the directory it sits in.
    let deep = fixture.staging_root().join("install-deep");
    fs::create_dir_all(&deep).unwrap();
    fs::write(deep.join("part.bin"), b"in flight").unwrap();
    set_mtime(&deep, std::time::Duration::from_secs(6 * 60 * 60));
    assert!(
        active_staging_ids(&fixture.store, within).contains(&"install-deep".to_owned()),
        "age must be the NEWEST mtime under the tree, not the root's own"
    );
}

/// sc-20636 review: `begin` used to `remove_dir_all` an existing staging tree before creating its
/// own. Two sessions for the same install id could then both "succeed": B's begin deleted A's
/// in-flight bytes, and A's Drop then deleted B's — two live transfers destroying each other with
/// no error on either side. The directory itself is now the mutual exclusion.
#[test]
fn a_second_session_for_the_same_install_id_refuses_instead_of_deleting_the_first() {
    let fixture = fixture("concurrent");
    let source = fixture.user_file();

    let first = ManagedIngest::begin(&fixture.store, "install-s1", civitai_provenance()).unwrap();
    first.stage_copy_file(&source, KREA_FILE).unwrap();
    let staged = first.staging_dir().join(KREA_FILE);
    let staged_bytes = fs::read(&staged).unwrap();

    let error = ManagedIngest::begin(&fixture.store, "install-s1", civitai_provenance())
        .expect_err("a second session for the same install id must refuse");
    assert!(
        matches!(
            error,
            ManagedIngestError::Plan(CheckpointPlanError::InstallIdTaken { .. })
        ),
        "expected InstallIdTaken, got {error}"
    );

    // The first session's bytes are untouched, and it still finalizes.
    assert_eq!(fs::read(&staged).unwrap(), staged_bytes);
    let install = first.finalize(KREA_FILE, None).unwrap();
    assert!(install.install_path.join(KREA_FILE).is_file());
    assert_eq!(
        fs::read(install.install_path.join(KREA_FILE)).unwrap(),
        staged_bytes
    );
}

/// sc-20636 review: `entry.file_type()` does not follow links, so a symlink was neither `is_dir`
/// nor `is_file` and fell out of the loop — SILENTLY SKIPPED. An HF cache snapshot dir is entirely
/// symlinks into `blobs/`, so it would have staged EMPTY with no error anywhere.
#[cfg(unix)]
#[test]
fn staging_a_directory_refuses_an_entry_that_is_neither_a_file_nor_a_directory() {
    let fixture = fixture("copydir");
    let source_dir = fixture.library_dir.join("snapshot");
    fs::create_dir_all(&source_dir).unwrap();
    write_krea_native_file(&source_dir.join(KREA_FILE), 0x44);

    let ingest = ManagedIngest::begin(&fixture.store, "install-cd1", civitai_provenance()).unwrap();
    // The all-regular-files case still copies.
    let copied = ingest.stage_copy_dir(&source_dir, "").unwrap();
    assert!(copied > 0);
    assert!(ingest.staging_dir().join(KREA_FILE).is_file());

    // Now the HF-cache shape: the payload reachable only through a symlink.
    let blobs = fixture.library_dir.join("blobs");
    fs::create_dir_all(&blobs).unwrap();
    let blob = blobs.join("deadbeef");
    write_krea_native_file(&blob, 0x55);
    std::os::unix::fs::symlink(&blob, source_dir.join("linked.safetensors")).unwrap();

    let error = ingest
        .stage_copy_dir(&source_dir, "again")
        .expect_err("a symlinked entry must refuse, not be skipped");
    assert_eq!(error.code(), "unsupported-source-entry");
    let message = error.to_string();
    assert!(
        message.contains("linked.safetensors") && message.contains("symbolic link"),
        "the refusal must name the entry and why it could not be staged: {message}"
    );
    assert!(
        !ingest
            .staging_dir()
            .join("again/linked.safetensors")
            .exists(),
        "the link must not be followed out of the directory the user pointed at"
    );
    assert!(blob.is_file(), "the user's own bytes are only ever read");
}

/// Backdate every mtime at or under `path` by `age`, so a tree written moments ago reads as one
/// abandoned that long ago. Bottom-up: setting a child's times never bumps its parent.
#[cfg(unix)]
fn backdate_tree(path: &Path, age: std::time::Duration) {
    if path.is_dir() {
        for entry in fs::read_dir(path).unwrap() {
            backdate_tree(&entry.unwrap().path(), age);
        }
    }
    set_mtime(path, age);
}

#[cfg(unix)]
fn set_mtime(path: &Path, age: std::time::Duration) {
    let when = std::time::SystemTime::now() - age;
    let handle = fs::File::open(path).unwrap();
    handle
        .set_times(fs::FileTimes::new().set_modified(when))
        .unwrap();
}

#[test]
fn a_disk_full_write_leaves_no_install_no_plan_and_no_record() {
    let fixture = fixture("diskfull");
    let bytes = fs::read(fixture.user_file()).unwrap();

    let ingest = ManagedIngest::begin(&fixture.store, "install-f1", civitai_provenance()).unwrap();
    let mut reader = FailingReader {
        fail_after: bytes.len() / 2,
        bytes,
        position: 0,
    };
    let error = ingest
        .stage_from_reader(KREA_FILE, &mut reader)
        .expect_err("a full disk refuses");
    assert_eq!(error.code(), "io");
    assert!(
        error.to_string().contains("No space left on device"),
        "the refusal must name the cause: {error}"
    );
    // The partial bytes are in STAGING, which is the whole point.
    assert!(ingest.staging_dir().join(KREA_FILE).is_file());
    fixture.assert_no_trace_of("install-f1", "disk full, before drop");
    drop(ingest);
    fixture.assert_no_trace_of("install-f1", "disk full");
    assert!(!fixture.staging_root().join("install-f1").exists());
}

/// The same invariant with a REAL write failure rather than a simulated one: the second artifact
/// cannot be created because its directory is unwritable, after the first has already landed.
#[cfg(unix)]
#[test]
fn a_real_write_failure_partway_through_a_multi_file_stage_leaves_no_install() {
    use std::os::unix::fs::PermissionsExt as _;

    let fixture = fixture("writefail");
    let source = fixture.user_file();

    let ingest = ManagedIngest::begin(&fixture.store, "install-w1", civitai_provenance()).unwrap();
    ingest.stage_copy_file(&source, KREA_FILE).unwrap();
    let staging = ingest.staging_dir().to_path_buf();
    fs::set_permissions(&staging, fs::Permissions::from_mode(0o555)).unwrap();
    let error = ingest
        .stage_copy_file(&source, "vae/diffusion_pytorch_model.safetensors")
        .expect_err("an unwritable staging directory refuses");
    assert_eq!(error.code(), "io");
    fs::set_permissions(&staging, fs::Permissions::from_mode(0o755)).unwrap();

    drop(ingest);
    fixture.assert_no_trace_of("install-w1", "write failure partway");
}

#[test]
fn a_hash_mismatch_leaves_no_install_no_plan_and_no_record() {
    let fixture = fixture("hash");
    let source = fixture.user_file();

    let ingest = ManagedIngest::begin(&fixture.store, "install-h1", civitai_provenance()).unwrap();
    ingest.stage_copy_file(&source, KREA_FILE).unwrap();
    let error = ingest
        .finalize(KREA_FILE, Some(&"a".repeat(64)))
        .expect_err("a declared digest that does not match the staged bytes refuses");

    assert_eq!(error.code(), "hash-mismatch");
    assert!(
        error.to_string().contains("the transfer was corrupted"),
        "the refusal must be actionable: {error}"
    );
    fixture.assert_no_trace_of("install-h1", "hash mismatch");
    assert!(!fixture.staging_root().join("install-h1").exists());
    assert!(
        source.is_file(),
        "a hash mismatch must not touch the user's own file"
    );
}

#[test]
fn an_unrunnable_staged_tree_rolls_the_commit_back_and_leaves_no_install() {
    let fixture = fixture("unrunnable");
    // A structurally valid safetensors file whose tensors carry no family evidence at all: the
    // inspector's full-content validation refuses it, so the committed tree is rolled back.
    let source = fixture.library_dir.join("mystery.safetensors");
    write_safetensors(&source, &[("some.unknown.weight", "BF16")], 0x11);

    let ingest = ManagedIngest::begin(&fixture.store, "install-u1", civitai_provenance()).unwrap();
    ingest
        .stage_copy_file(&source, "mystery.safetensors")
        .unwrap();
    let error = ingest
        .finalize("mystery.safetensors", None)
        .expect_err("an unrunnable staged tree refuses");

    assert_eq!(error.code(), "unrunnable-source");
    fixture.assert_no_trace_of("install-u1", "unrunnable content");
    assert!(!fixture.staging_root().join("install-u1").exists());
    assert!(source.is_file(), "the user's own file is untouched");
}

#[test]
fn an_install_id_already_in_use_refuses_before_anything_is_staged() {
    let fixture = fixture("taken");
    let source = fixture.user_file();

    ManagedIngest::begin(&fixture.store, "install-t1", civitai_provenance())
        .unwrap()
        .stage_and_finalize(&source);

    let error = ManagedIngest::begin(&fixture.store, "install-t1", civitai_provenance())
        .expect_err("a live install id cannot be re-staged over");
    assert_eq!(error.code(), "install-id-taken");
    // The first install is untouched.
    assert!(fixture
        .store
        .install_dir("install-t1")
        .unwrap()
        .join(KREA_FILE)
        .is_file());
    assert!(fixture.store.resolve("managed/install-t1").is_ok());
}

#[test]
fn a_staged_relative_path_can_never_escape_the_staging_directory() {
    let fixture = fixture("escape");
    let source = fixture.user_file();
    let ingest = ManagedIngest::begin(&fixture.store, "install-e1", civitai_provenance()).unwrap();

    for escape in [
        "../escaped.safetensors",
        "../../escaped.safetensors",
        "/tmp/escaped.safetensors",
        "./escaped.safetensors",
    ] {
        let error = ingest
            .stage_copy_file(&source, escape)
            .expect_err("an escaping relative path refuses");
        assert_eq!(error.code(), "invalid-relative-path", "{escape}");
    }
    assert!(!fixture
        .store
        .installs_root()
        .join("escaped.safetensors")
        .exists());
    assert!(!fixture.staging_root().join("escaped.safetensors").exists());
}

/// `stage_copy_dir` already refuses a symlink it meets while walking. `stage_copy_file` is also a
/// public entry point — a local-path import of a single file reaches it directly — and it followed
/// the link, reading bytes from wherever it led, outside the path the user named.
#[cfg(unix)]
#[test]
fn a_symbolic_link_at_the_staging_source_root_is_refused() {
    let fixture = fixture("symlink");
    let real = fixture.user_file();
    let link = fixture.library_dir.join("link-to-krea.safetensors");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let ingest = ManagedIngest::begin(&fixture.store, "install-s1", civitai_provenance()).unwrap();
    let error = ingest
        .stage_copy_file(&link, KREA_FILE)
        .expect_err("a symlinked source refuses");
    assert_eq!(error.code(), "unsupported-source-entry");
    assert!(
        !fixture
            .staging_root()
            .join("install-s1")
            .join(KREA_FILE)
            .exists(),
        "nothing was staged from the link"
    );
    // The real file behind the link is still stageable: the refusal is about the link, not the
    // bytes.
    assert!(ingest.stage_copy_file(&real, KREA_FILE).is_ok());
}

// ---- AC2: parity, duplicate reporting, ownership-safe removal --------------------------------

#[test]
fn a_managed_copy_and_its_linked_original_share_a_semantic_digest_and_report_each_other() {
    let fixture = fixture("parity");
    let source = fixture.user_file();

    let root = fixture.store.approve_root(&fixture.library_dir).unwrap();
    let linked = fixture
        .store
        .compile_linked(&root.root_id, KREA_FILE)
        .unwrap();
    assert!(
        linked.duplicate_checkpoint_ids.is_empty(),
        "the first compile of a digest has no duplicate"
    );

    let managed = ManagedIngest::begin(&fixture.store, "install-p1", civitai_provenance())
        .unwrap()
        .stage_and_finalize(&source);

    // E1: the SEMANTIC digest is locator-independent, so the managed copy and the linked original
    // compile to the same plan identity...
    assert_eq!(
        managed.compiled.record.summary.semantic_digest,
        linked.record.summary.semantic_digest
    );
    assert_eq!(
        managed.compiled.plan.semantic_digest().unwrap(),
        linked.plan.semantic_digest().unwrap()
    );
    // ...while the source-binding identity — which carries ownership, location and provenance — is
    // deliberately different, so lifecycle and invalidation can still tell them apart.
    assert_ne!(
        managed.compiled.record.plan.source_binding_identity,
        linked.record.plan.source_binding_identity
    );
    assert_ne!(managed.compiled.plan.plan_id, linked.plan.plan_id);

    // The duplicate is REPORTED, in both directions...
    assert_eq!(
        managed.duplicate_checkpoint_ids(),
        [linked_checkpoint_id(&root.root_id, KREA_FILE)]
    );
    assert_eq!(
        fixture
            .store
            .duplicates_of(
                &linked.record.summary.semantic_digest,
                &linked.checkpoint_id
            )
            .unwrap(),
        ["managed/install-p1"]
    );
    // ...and NEITHER copy is deleted: both still resolve, and both files are on disk.
    assert!(fixture.store.resolve(&linked.checkpoint_id).is_ok());
    assert!(fixture.store.resolve("managed/install-p1").is_ok());
    assert!(source.is_file());
    assert!(managed.install_path.join(KREA_FILE).is_file());
    assert_eq!(fixture.store.inventory().unwrap().records.len(), 2);
}

/// The plan's `target_path` is part of its semantic identity, so parity is between copies that hold
/// the same logical layout — not between any two files with equal bytes. Pinned so a future change
/// that dropped `target_path` from the digest (making unrelated layouts collide) is caught.
///
/// The difference exercised here is the checkpoint's own INTERNAL name, because that is what
/// `target_path` carries after sc-20651: the directory a single-file checkpoint happens to sit in
/// is location, not layout, and the two are no longer allowed to be confused (see
/// `a_checkpoint_linked_at_a_nested_path_is_the_same_checkpoint_as_its_managed_copy`).
#[test]
fn a_managed_copy_under_a_different_internal_name_is_a_different_plan_despite_identical_bytes() {
    let fixture = fixture("layout");
    let source = fixture.user_file();

    let root = fixture.store.approve_root(&fixture.library_dir).unwrap();
    let linked = fixture
        .store
        .compile_linked(&root.root_id, KREA_FILE)
        .unwrap();

    let renamed = "kreamania-v2.safetensors";
    let ingest = ManagedIngest::begin(&fixture.store, "install-l1", civitai_provenance()).unwrap();
    ingest.stage_copy_file(&source, renamed).unwrap();
    let managed = ingest.finalize(renamed, None).unwrap();

    assert_eq!(
        managed.compiled.plan.layers[0].target_path, renamed,
        "the layer carries the checkpoint's own internal name"
    );
    assert_ne!(
        managed.compiled.record.summary.semantic_digest,
        linked.record.summary.semantic_digest
    );
    assert!(managed.duplicate_checkpoint_ids().is_empty());
}

/// BLOCKER 5 (sc-20651): the semantic digest must not know where in the library the checkpoint sits.
///
/// A plan layer's `layer_id` and `target_path` used to be taken relative to the LIBRARY ROOT, so a
/// checkpoint kept a few directories down — the normal case for anyone with an organised library —
/// compiled to a different semantic digest than the byte-identical managed copy of it, which is a
/// managed install whose own directory is the root. Linked/managed parity therefore held only for
/// checkpoints sitting at the very top of a root, and nowhere else.
///
/// Both digests below are produced by real production compiles (`compile_linked` through the
/// approved root, `ManagedIngest::finalize` through the staging/commit path); neither side is
/// derived from the other.
#[test]
fn a_checkpoint_linked_at_a_nested_path_is_the_same_checkpoint_as_its_managed_copy() {
    let fixture = fixture("nested-parity");
    let nested_relative = format!("vendors/krea/v2/{KREA_FILE}");
    let nested = fixture.library_dir.join(&nested_relative);
    write_krea_native_file(&nested, 0x21);

    let root = fixture.store.approve_root(&fixture.library_dir).unwrap();
    let linked = fixture
        .store
        .compile_linked(&root.root_id, &nested_relative)
        .unwrap();
    let managed = ManagedIngest::begin(&fixture.store, "install-n1", civitai_provenance())
        .unwrap()
        .stage_and_finalize(&nested);

    assert_eq!(
        managed.compiled.record.summary.semantic_digest, linked.record.summary.semantic_digest,
        "a nested linked checkpoint and its managed copy are the same checkpoint"
    );
    assert_eq!(
        managed.compiled.plan.semantic_digest().unwrap(),
        linked.plan.semantic_digest().unwrap()
    );

    // The layer identity is the checkpoint's own, with no trace of the library path in it...
    assert_eq!(linked.plan.layers.len(), 1);
    assert_eq!(linked.plan.layers[0].target_path, KREA_FILE);
    assert_eq!(
        linked.plan.layers[0].layer_id,
        format!("artifact:{KREA_FILE}")
    );

    // ...while the LOCATOR — the thing that says where to open the bytes — is still root-relative,
    // because that is the one place the location genuinely belongs.
    match &linked.plan.layers[0].source {
        SourceLocatorV1::Linked { relative_path, .. } => {
            assert_eq!(relative_path, &nested_relative);
        }
        other => panic!("a linked compile produced {other:?}"),
    }

    // And because the digests now agree, the two copies find each other as duplicates.
    assert_eq!(
        managed.duplicate_checkpoint_ids(),
        [linked_checkpoint_id(&root.root_id, &nested_relative)]
    );
}

#[test]
fn removing_a_managed_install_deletes_only_sceneworks_owned_state() {
    let fixture = fixture("remove");
    let source = fixture.user_file();

    let root = fixture.store.approve_root(&fixture.library_dir).unwrap();
    let linked = fixture
        .store
        .compile_linked(&root.root_id, KREA_FILE)
        .unwrap();
    let managed = ManagedIngest::begin(&fixture.store, "install-r1", civitai_provenance())
        .unwrap()
        .stage_and_finalize(&source);
    let managed_documents = [
        format!("{PLANS_DIR}/{}.json", managed.compiled.plan.plan_id),
        format!("{BINDINGS_DIR}/{}.json", managed.compiled.plan.plan_id),
    ];
    for document in &managed_documents {
        assert!(fixture.plan_documents().contains(document));
    }

    assert!(fixture.store.remove_managed("install-r1").unwrap());

    // SceneWorks-owned state for that install is gone: bytes, record, plan, bindings.
    assert!(!managed.install_path.exists());
    assert!(fixture.store.record("managed/install-r1").is_err());
    for document in &managed_documents {
        assert!(
            !fixture.plan_documents().contains(document),
            "{document} survived removal"
        );
    }
    // The user's file, the approved root, and the LINKED checkpoint compiled from it are untouched.
    assert!(
        source.is_file(),
        "removal must never delete a linked source"
    );
    assert!(fixture.library_dir.is_dir());
    assert!(fixture.store.resolve(&linked.checkpoint_id).is_ok());
    assert_eq!(fixture.store.inventory().unwrap().records.len(), 1);
    assert!(fixture
        .plan_documents()
        .contains(&format!("{PLANS_DIR}/{}.json", linked.plan.plan_id)));

    // Idempotent: removing again reports nothing removed rather than erroring.
    assert!(!fixture.store.remove_managed("install-r1").unwrap());
}

#[test]
fn remove_managed_can_never_address_anything_outside_the_installs_tree() {
    let fixture = fixture("confine");
    let victim = fixture.library_dir.join("precious.safetensors");
    write_krea_native_file(&victim, 0x33);

    for install_id in [
        "..",
        "../..",
        "../library",
        "install/../../../library",
        "install/..",
        "/etc",
        "a..b",
        ".hidden",
        "-install",
        "",
        " ",
        "install a",
        &"a".repeat(129),
    ] {
        let error = fixture
            .store
            .remove_managed(install_id)
            .expect_err("a non-conforming install id refuses");
        assert!(
            matches!(error, CheckpointPlanError::InvalidInstallId { .. }),
            "{install_id:?} refused as {error}"
        );
        assert!(
            fixture.store.install_dir(install_id).is_err(),
            "{install_id:?} must not resolve to a directory at all"
        );
    }
    assert!(victim.is_file(), "nothing outside installs/ was touched");
    assert!(fixture.library_dir.is_dir());
}

#[test]
fn a_managed_plan_refuses_drifted_bytes_and_a_foreign_install_locator() {
    let fixture = fixture("drift");
    let source = fixture.user_file();
    let managed = ManagedIngest::begin(&fixture.store, "install-g1", civitai_provenance())
        .unwrap()
        .stage_and_finalize(&source);

    // Same size, different bytes: the stamp changes, the re-hash refuses.
    write_krea_native_file(&managed.install_path.join(KREA_FILE), 0x77);
    let error = fixture.store.resolve("managed/install-g1").unwrap_err();
    assert!(
        matches!(error, CheckpointPlanError::SourceDrifted { .. }),
        "drifted managed bytes refused as {error}"
    );

    // A hand-edited plan whose managed layer names a DIFFERENT install must not read that
    // install's bytes through this checkpoint's record.
    let plan_path = fixture
        .store
        .root()
        .join(PLANS_DIR)
        .join(format!("{}.json", managed.compiled.plan.plan_id));
    let mut plan: Value = serde_json::from_str(&fs::read_to_string(&plan_path).unwrap()).unwrap();
    plan["layers"][0]["source"]["installId"] = json!("install-other");
    fs::write(&plan_path, serde_json::to_vec(&plan).unwrap()).unwrap();
    let error = fixture.store.resolve("managed/install-g1").unwrap_err();
    assert!(
        matches!(error, CheckpointPlanError::PlanTampered { .. }),
        "a retargeted managed locator refused as {error}"
    );
}

#[test]
fn a_managed_install_whose_directory_was_deleted_outside_the_app_refuses_typed() {
    let fixture = fixture("gone");
    let source = fixture.user_file();
    let managed = ManagedIngest::begin(&fixture.store, "install-n1", civitai_provenance())
        .unwrap()
        .stage_and_finalize(&source);

    fs::remove_dir_all(&managed.install_path).unwrap();
    let error = fixture.store.resolve("managed/install-n1").unwrap_err();
    assert!(
        matches!(error, CheckpointPlanError::InstallUnavailable { .. }),
        "a vanished install refused as {error}"
    );
    assert_eq!(error.code(), "install-unavailable");
}

#[test]
fn provenance_refuses_a_url_that_carries_a_credential() {
    let mut provenance = civitai_provenance();
    provenance.url = Some("https://user:t0ken@civitai.com/api/download/models/1".to_owned());
    let error = provenance
        .validate()
        .expect_err("a credential-bearing url must never reach a persisted plan");
    assert!(error.to_string().contains("must not embed credentials"));

    for blank in ["", "   "] {
        let mut provenance = civitai_provenance();
        provenance.version_id = Some(blank.to_owned());
        assert!(provenance.validate().is_err(), "{blank:?}");
    }
}

// ---- helpers ---------------------------------------------------------------------------------

trait StageAndFinalize {
    fn stage_and_finalize(
        self,
        source: &Path,
    ) -> sceneworks_core::checkpoint_ingest::ManagedInstallV1;
}

impl StageAndFinalize for ManagedIngest {
    fn stage_and_finalize(
        self,
        source: &Path,
    ) -> sceneworks_core::checkpoint_ingest::ManagedInstallV1 {
        self.stage_copy_file(source, KREA_FILE).unwrap();
        let expected = sha256_of(source);
        self.finalize(KREA_FILE, Some(&expected)).unwrap()
    }
}

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(fs::read(path).unwrap());
    format!("{:x}", hasher.finalize())
}

#[test]
fn every_ingest_refusal_carries_its_stable_code_in_its_message() {
    // The codes callers branch on and the operator sees are the same strings.
    let error = ManagedIngestError::Cancelled {
        install_id: "install-z1".to_owned(),
    };
    assert!(error
        .to_string()
        .starts_with("[checkpoint-ingest:cancelled]"));
    let error = ManagedIngestError::Plan(CheckpointPlanError::InvalidInstallId {
        install_id: "..".to_owned(),
        reason: "must be lowercase alphanumeric or '-'",
    });
    // A plan refusal keeps its own `[checkpoint-plan:...]` prefix rather than being double-wrapped.
    assert!(error
        .to_string()
        .starts_with("[checkpoint-plan:invalid-install-id]"));
    assert_eq!(error.code(), "invalid-install-id");
}
