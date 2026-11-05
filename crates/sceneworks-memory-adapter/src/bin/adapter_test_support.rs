//! Filesystem fixtures shared by the two capture binaries, compiled only in test harnesses.
//! A test owns `Fixtures`; returned paths/source rows borrow its lifetime by convention and every
//! TempDir remains retained until that test finishes. No global paths or leaked directories.
use sceneworks_memory_adapter as protocol;
use std::{cell::RefCell, path::PathBuf};

#[derive(Default)]
pub(crate) struct Fixtures(RefCell<Vec<tempfile::TempDir>>);
impl Fixtures {
    /// Return an absent child path, retaining its private temporary parent for the whole test.
    pub(crate) fn path(&self, label: &str) -> PathBuf {
        let temporary = tempfile::Builder::new()
            .prefix(label)
            .tempdir()
            .expect("create fixture parent");
        let path = temporary.path().join("fixture");
        self.0.borrow_mut().push(temporary);
        path
    }
    pub(crate) fn snapshot(&self, repository: &str, revision: &str, tier: Option<&str>) -> PathBuf {
        let mut path = self
            .path("snapshot")
            .join(format!("models--{}", repository.replace('/', "--")))
            .join("snapshots")
            .join(revision);
        if let Some(tier) = tier {
            path.push(tier);
        }
        std::fs::create_dir_all(&path).expect("create snapshot fixture");
        path
    }
    pub(crate) fn pulid_bundle(&self) -> protocol::PulidIdentityBundle {
        let root = self.path("pulid-bundle");
        std::fs::create_dir_all(&root).unwrap();
        for file in protocol::PULID_IDENTITY_BUNDLE_FILES {
            std::fs::write(root.join(file), b"weights").unwrap();
        }
        protocol::pulid_identity_bundle_at(root).unwrap()
    }
    pub(crate) fn qwen_edit_source(
        &self,
        lightning: bool,
        tier: &str,
        revision: &str,
    ) -> protocol::QwenEditArtifactSource {
        protocol::QwenEditArtifactSource {
            repository: protocol::QWEN_EDIT_REPOSITORY.to_owned(),
            revision: revision.to_owned(),
            root: self.snapshot(protocol::QWEN_EDIT_REPOSITORY, revision, Some(tier)),
            lightning: lightning.then(|| {
                let root = self.path("qwen-edit-lightning");
                std::fs::create_dir_all(&root).unwrap();
                std::fs::write(root.join(protocol::QWEN_EDIT_LIGHTNING_FILE), b"distill").unwrap();
                protocol::QwenEditLightningSource {
                    repository: protocol::QWEN_EDIT_LIGHTNING_REPOSITORY.to_owned(),
                    revision: revision.to_owned(),
                    root,
                }
            }),
        }
    }
}

#[test]
fn fixture_scope_keeps_nested_paths_alive_and_removes_them_on_drop() {
    let path;
    {
        let fixtures = Fixtures::default();
        path = fixtures.snapshot("owner/model", "revision", Some("q4"));
        std::fs::write(path.join("sentinel"), b"fixture").unwrap();
        assert!(path.join("sentinel").is_file());
    }
    assert!(!path.exists());
}

pub(crate) mod staging {
    use super::protocol::{
        INSTANTID_CONTROLNET_ENV, INSTANTID_CONTROLNET_WEIGHT_FILE, INSTANTID_IDENTITY_BUNDLE_ENV,
        INSTANTID_IDENTITY_BUNDLE_FILES,
    };
    use std::path::PathBuf;
    use std::sync::{Mutex, MutexGuard, PoisonError};

    /// Serializes the process-global `SCENEWORKS_INSTANTID_*` swap. Both `instantid_binding` and
    /// `instantid_candle_identity` read five environment variables; two tests mutating them
    /// concurrently would each resolve half of the other's tree.
    static INSTANTID_ENV_LOCK: Mutex<()> = Mutex::new(());

    pub const INSTANTID_ENV_NAMES: [&str; 5] = [
        "SCENEWORKS_INSTANTID_REALVISXL_REPOSITORY",
        "SCENEWORKS_INSTANTID_REALVISXL_REVISION",
        "SCENEWORKS_INSTANTID_REALVISXL_ROOT",
        INSTANTID_IDENTITY_BUNDLE_ENV,
        INSTANTID_CONTROLNET_ENV,
    ];

    /// A synthetic InstantID staging tree, installed into the process environment for as long as
    /// this value lives. RAII rather than a restore statement so a panicking assertion cannot leak
    /// the injected environment into every later test in the process.
    pub struct StagedInstantId {
        _temporary: tempfile::TempDir,
        /// The tiered backbone snapshot root, in the exact Hugging Face layout the validators want.
        pub snapshot: PathBuf,
        /// The IdentityNet `ControlNetModel` directory. Created EMPTY — call
        /// [`StagedInstantId::stage_identitynet_weight`] to complete it.
        pub identitynet: PathBuf,
        /// The identity bundle root, carrying all three of `INSTANTID_IDENTITY_BUNDLE_FILES`.
        pub bundle: PathBuf,
        _guard: MutexGuard<'static, ()>,
    }

    impl StagedInstantId {
        /// A well-formed 40-hex revision. Any value works: the validators check SHAPE and the path
        /// suffix, never that the revision exists anywhere.
        pub const REVISION: &'static str = "e40202d63baef826c7df95a639a811698c1178d2";

        pub fn install(tag: &str, repository: &str, tier: &str) -> Self {
            let guard = INSTANTID_ENV_LOCK
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let temporary = tempfile::Builder::new()
                .prefix(tag)
                .tempdir()
                .expect("stage InstantID fixture");
            let root = temporary.path();
            let snapshot = root
                .join(format!("models--{}", repository.replace('/', "--")))
                .join("snapshots")
                .join(Self::REVISION)
                .join(tier);
            let identitynet = root.join("identitynet");
            let bundle = root.join("bundle");
            for dir in [&snapshot, &identitynet, &bundle] {
                std::fs::create_dir_all(dir).expect("create the staged tree");
            }
            for file in INSTANTID_IDENTITY_BUNDLE_FILES {
                std::fs::write(bundle.join(file), b"x").expect("stage a bundle file");
            }
            for (name, value) in [
                (INSTANTID_ENV_NAMES[0], repository.to_owned()),
                (INSTANTID_ENV_NAMES[1], Self::REVISION.to_owned()),
                (INSTANTID_ENV_NAMES[2], snapshot.display().to_string()),
                (INSTANTID_ENV_NAMES[3], bundle.display().to_string()),
                (INSTANTID_ENV_NAMES[4], identitynet.display().to_string()),
            ] {
                std::env::set_var(name, value);
            }
            Self {
                _temporary: temporary,
                snapshot,
                identitynet,
                bundle,
                _guard: guard,
            }
        }

        pub fn stage_identitynet_weight(&self) {
            std::fs::write(
                self.identitynet.join(INSTANTID_CONTROLNET_WEIGHT_FILE),
                b"x",
            )
            .expect("stage the IdentityNet weight");
        }

        /// The five artifact paths, canonicalized, in the role order the worker's
        /// `instantid_artifact_fingerprint` hashes them: backbone, IdentityNet, IP-Adapter, SCRFD,
        /// ArcFace.
        pub fn ordered_paths(&self) -> Vec<PathBuf> {
            let bundle = std::fs::canonicalize(&self.bundle).expect("canonicalize the bundle");
            let mut paths = vec![
                std::fs::canonicalize(&self.snapshot).expect("canonicalize the snapshot"),
                std::fs::canonicalize(&self.identitynet).expect("canonicalize the IdentityNet dir"),
            ];
            paths.extend(INSTANTID_IDENTITY_BUNDLE_FILES.map(|file| bundle.join(file)));
            paths
        }
    }

    impl Drop for StagedInstantId {
        fn drop(&mut self) {
            for name in INSTANTID_ENV_NAMES {
                std::env::remove_var(name);
            }
        }
    }
}
