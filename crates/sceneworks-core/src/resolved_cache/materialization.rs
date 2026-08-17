use super::*;
use crate::model_artifacts::{ArtifactFile, ResolvedBundleClosure, ResolvedBundleFileLocation};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::{Read, Write};
use std::path::Component;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const COPY_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Default)]
pub struct MaterializationCancellation {
    cancelled: Arc<AtomicBool>,
}

impl MaterializationCancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaterializationOutcome {
    Published(Box<ResolvedCacheMetadata>),
    AlreadyComplete(Box<ResolvedCacheMetadata>),
    Contended,
    Cancelled,
}

#[derive(Clone)]
pub struct ResolvedCacheMaterializer {
    store: ResolvedCacheStore,
    #[cfg(test)]
    test_hook: Option<Arc<TestCopyHook>>,
}

#[cfg(test)]
type TestCopyHook = dyn Fn(usize, &Path) -> std::io::Result<()> + Send + Sync;

impl std::fmt::Debug for ResolvedCacheMaterializer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedCacheMaterializer")
            .field("store", &self.store)
            .finish_non_exhaustive()
    }
}

impl ResolvedCacheMaterializer {
    pub fn new(store: ResolvedCacheStore) -> Self {
        Self {
            store,
            #[cfg(test)]
            test_hook: None,
        }
    }

    pub fn store(&self) -> &ResolvedCacheStore {
        &self.store
    }

    pub fn materialize(
        &self,
        candidate: &PromotionCandidate,
        source_configured_path: &Path,
        logical_model_owner: &str,
        cancellation: &MaterializationCancellation,
    ) -> Result<MaterializationOutcome, ResolvedCacheError> {
        validate_configured_source(candidate, source_configured_path)?;
        let reservation =
            match self
                .store
                .reserve(candidate, source_configured_path, logical_model_owner)?
            {
                ReservationOutcome::AlreadyComplete(metadata) => {
                    return Ok(MaterializationOutcome::AlreadyComplete(metadata));
                }
                ReservationOutcome::Contended => return Ok(MaterializationOutcome::Contended),
                ReservationOutcome::Acquired(reservation) => reservation,
            };
        reservation.prepare_for_materialization()?;
        if cancellation.is_cancelled() {
            return cancel_reservation(*reservation);
        }

        let primary_root = match &candidate.artifact.location {
            ArtifactLocation::SourceLibrary { root } => root,
            ArtifactLocation::ResolvedLocal { .. } => {
                return Err(ResolvedCacheError::new(
                    "only a source-library artifact can be materialized",
                ));
            }
        };
        let locations = match candidate
            .artifact
            .closure
            .source_file_locations(&candidate.artifact.identity, primary_root)
        {
            Ok(locations) => locations,
            Err(error) => {
                cleanup_failed_reservation(*reservation)?;
                return Err(ResolvedCacheError::new(error.to_string()));
            }
        };
        let sources = match locations
            .into_iter()
            .map(PlannedSourceFile::open)
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(sources) => sources,
            Err(error) => {
                cleanup_failed_reservation(*reservation)?;
                return Err(error);
            }
        };
        let result = self.copy_bundle(candidate, &reservation, sources, cancellation);
        let artifact = match result {
            Ok(artifact) => artifact,
            Err(_error) if cancellation.is_cancelled() => {
                cleanup_failed_reservation(*reservation)?;
                return Ok(MaterializationOutcome::Cancelled);
            }
            Err(error) => {
                if let Err(cleanup) = cleanup_failed_reservation(*reservation) {
                    return Err(ResolvedCacheError::new(format!(
                        "{error}; failed to clean interrupted staging: {cleanup}"
                    )));
                }
                return Err(error);
            }
        };
        if cancellation.is_cancelled() {
            cleanup_failed_reservation(*reservation)?;
            return Ok(MaterializationOutcome::Cancelled);
        }
        reservation
            .publish_staged(artifact)
            .map(|metadata| MaterializationOutcome::Published(Box::new(metadata)))
    }

    fn copy_bundle(
        &self,
        candidate: &PromotionCandidate,
        reservation: &ResolvedCacheReservation,
        sources: Vec<PlannedSourceFile>,
        cancellation: &MaterializationCancellation,
    ) -> Result<ResolvedModelArtifact, ResolvedCacheError> {
        let mut artifact = candidate.artifact.clone();
        artifact.location = ArtifactLocation::ResolvedLocal {
            root: reservation.staging_path().to_owned(),
        };
        let mut copied = BTreeMap::<(String, PathBuf), (PathBuf, VerifiedFile)>::new();
        #[cfg(test)]
        let mut copy_index = 0_usize;
        for mut source in sources {
            let location = source.location.clone();
            if cancellation.is_cancelled() {
                return Err(ResolvedCacheError::new("materialization cancelled"));
            }
            #[cfg(test)]
            if let Some(hook) = &self.test_hook {
                hook(copy_index, &location.source_path)?;
                copy_index += 1;
            }
            let member = candidate
                .artifact
                .closure
                .members
                .get(location.member_index)
                .ok_or_else(|| ResolvedCacheError::new("source plan member index changed"))?;
            let destination = reservation.staging_path().join(&location.output_path);
            let source_key = (
                member.source.fingerprint.clone(),
                location.source_path.clone(),
            );
            let expected = expected_file(candidate, &location)?;
            let verified = if let Some((existing, verified)) = copied.get(&source_key) {
                source.validate_unchanged()?;
                verify_expected(&location.source_path, expected, verified)?;
                match hard_link_confined(
                    reservation.staging_path(),
                    existing,
                    &location.output_path,
                ) {
                    Ok(()) => verified.clone(),
                    Err(_) => {
                        let output = create_confined_output(
                            reservation.staging_path(),
                            &location.output_path,
                        )?;
                        copy_and_verify(&mut source, output, &destination, expected, cancellation)?
                    }
                }
            } else {
                let output =
                    create_confined_output(reservation.staging_path(), &location.output_path)?;
                let verified =
                    copy_and_verify(&mut source, output, &destination, expected, cancellation)?;
                copied.insert(source_key, (location.output_path.clone(), verified.clone()));
                verified
            };
            let file = artifact
                .closure
                .members
                .get_mut(location.member_index)
                .and_then(|member| member.files.get_mut(location.file_index))
                .ok_or_else(|| ResolvedCacheError::new("source plan file index changed"))?;
            file.size_bytes = Some(verified.size_bytes);
            file.sha256 = Some(verified.sha256);
        }
        artifact.closure = ResolvedBundleClosure::new(artifact.closure.members)
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
        artifact
            .validate()
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
        if artifact
            .cache_key()
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?
            != candidate.cache_key
        {
            return Err(ResolvedCacheError::new(
                "verification enrichment changed the immutable cache key",
            ));
        }
        Ok(artifact)
    }

    #[cfg(test)]
    fn with_test_hook(
        mut self,
        hook: impl Fn(usize, &Path) -> std::io::Result<()> + Send + Sync + 'static,
    ) -> Self {
        self.test_hook = Some(Arc::new(hook));
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VerifiedFile {
    size_bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceFileIdentity {
    len: u64,
    modified: Option<std::time::SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
    #[cfg(windows)]
    volume_serial_number: u64,
    #[cfg(windows)]
    file_index: u64,
}

impl SourceFileIdentity {
    fn from_file(file: &File) -> Result<Self, ResolvedCacheError> {
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(ResolvedCacheError::new(
                "materialization source handle is not a regular file",
            ));
        }
        #[cfg(windows)]
        let information = winapi_util::file::information(file)
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            #[cfg(unix)]
            device: {
                use std::os::unix::fs::MetadataExt;
                metadata.dev()
            },
            #[cfg(unix)]
            inode: {
                use std::os::unix::fs::MetadataExt;
                metadata.ino()
            },
            #[cfg(unix)]
            changed_seconds: {
                use std::os::unix::fs::MetadataExt;
                metadata.ctime()
            },
            #[cfg(unix)]
            changed_nanoseconds: {
                use std::os::unix::fs::MetadataExt;
                metadata.ctime_nsec()
            },
            #[cfg(windows)]
            volume_serial_number: information.volume_serial_number(),
            #[cfg(windows)]
            file_index: information.file_index(),
        })
    }
}

#[derive(Debug)]
struct PlannedSourceFile {
    location: ResolvedBundleFileLocation,
    input: File,
    identity: SourceFileIdentity,
}

impl PlannedSourceFile {
    fn open(location: ResolvedBundleFileLocation) -> Result<Self, ResolvedCacheError> {
        validate_source_path_is_regular_canonical(&location.source_path)?;
        let input = open_source_no_follow(&location.source_path)?;
        let identity = SourceFileIdentity::from_file(&input)?;
        validate_source_path_identity(&location.source_path, &identity)?;
        Ok(Self {
            location,
            input,
            identity,
        })
    }

    fn validate_unchanged(&self) -> Result<(), ResolvedCacheError> {
        if SourceFileIdentity::from_file(&self.input)? != self.identity {
            return Err(ResolvedCacheError::new(format!(
                "materialization source mutated while copying {}",
                self.location.source_path.display()
            )));
        }
        validate_source_path_identity(&self.location.source_path, &self.identity)
    }
}

fn expected_file<'a>(
    candidate: &'a PromotionCandidate,
    location: &ResolvedBundleFileLocation,
) -> Result<&'a ArtifactFile, ResolvedCacheError> {
    candidate
        .artifact
        .closure
        .members
        .get(location.member_index)
        .and_then(|member| member.files.get(location.file_index))
        .ok_or_else(|| ResolvedCacheError::new("source plan no longer matches artifact closure"))
}

fn copy_and_verify(
    source: &mut PlannedSourceFile,
    mut output: File,
    destination: &Path,
    expected: &ArtifactFile,
    cancellation: &MaterializationCancellation,
) -> Result<VerifiedFile, ResolvedCacheError> {
    if expected
        .size_bytes
        .is_some_and(|size| size != source.identity.len)
    {
        return Err(ResolvedCacheError::new(format!(
            "materialization source size changed for {}",
            source.location.source_path.display()
        )));
    }
    let mut digest = Sha256::new();
    let mut size_bytes = 0_u64;
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    loop {
        if cancellation.is_cancelled() {
            return Err(ResolvedCacheError::new("materialization cancelled"));
        }
        let read = source.input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        digest.update(&buffer[..read]);
        size_bytes = size_bytes
            .checked_add(read as u64)
            .ok_or_else(|| ResolvedCacheError::new("materialized byte count overflow"))?;
    }
    output.sync_all()?;
    let sha256 = format!("sha256:{:x}", digest.finalize());
    let verified = VerifiedFile { size_bytes, sha256 };
    verify_expected(&source.location.source_path, expected, &verified)?;
    source.validate_unchanged()?;
    let destination_metadata = output.metadata()?;
    if !destination_metadata.is_file() || destination_metadata.len() != size_bytes {
        return Err(ResolvedCacheError::new(format!(
            "materialized destination verification failed for {}",
            destination.display()
        )));
    }
    Ok(verified)
}

fn verify_expected(
    source: &Path,
    expected: &ArtifactFile,
    verified: &VerifiedFile,
) -> Result<(), ResolvedCacheError> {
    if expected
        .size_bytes
        .is_some_and(|expected| expected != verified.size_bytes)
        || expected
            .sha256
            .as_ref()
            .is_some_and(|expected| expected != &verified.sha256)
    {
        return Err(ResolvedCacheError::new(format!(
            "materialization source verification changed for {}",
            source.display()
        )));
    }
    Ok(())
}

fn validate_source_path_identity(
    source: &Path,
    expected: &SourceFileIdentity,
) -> Result<(), ResolvedCacheError> {
    validate_source_path_is_regular_canonical(source)?;
    let current = open_source_no_follow(source)?;
    if SourceFileIdentity::from_file(&current)? != *expected {
        return Err(ResolvedCacheError::new(format!(
            "materialization source identity changed for {}",
            source.display()
        )));
    }
    Ok(())
}

fn validate_source_path_is_regular_canonical(source: &Path) -> Result<(), ResolvedCacheError> {
    let metadata = std::fs::symlink_metadata(source)?;
    if metadata.file_type().is_symlink()
        || metadata_is_reparse_point(&metadata)
        || !metadata.is_file()
    {
        return Err(ResolvedCacheError::new(format!(
            "materialization source {} is linked, reparsed, or not a file",
            source.display()
        )));
    }
    if std::fs::canonicalize(source)? != source {
        return Err(ResolvedCacheError::new(format!(
            "materialization source {} is no longer its planned canonical path",
            source.display()
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn open_source_no_follow(source: &Path) -> Result<File, ResolvedCacheError> {
    use std::os::unix::fs::OpenOptionsExt;
    Ok(OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(source)?)
}

#[cfg(windows)]
fn open_source_no_follow(source: &Path) -> Result<File, ResolvedCacheError> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    Ok(OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(source)?)
}

#[cfg(not(any(unix, windows)))]
fn open_source_no_follow(source: &Path) -> Result<File, ResolvedCacheError> {
    Ok(File::open(source)?)
}

#[cfg(unix)]
fn confined_parent(
    root: &Path,
    relative: &Path,
    create: bool,
) -> Result<(File, std::ffi::OsString), ResolvedCacheError> {
    use rustix::fs::{mkdirat, openat, Mode, OFlags};
    use std::os::unix::fs::OpenOptionsExt;

    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ResolvedCacheError::new(
            "materialized output path is not a confined relative path",
        ));
    }
    let mut components = relative.components().collect::<Vec<_>>();
    let file_name = components
        .pop()
        .ok_or_else(|| ResolvedCacheError::new("materialized output path has no file name"))?
        .as_os_str()
        .to_owned();
    let mut directory = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(root)?;
    for component in components {
        let name = component.as_os_str();
        let flags = OFlags::RDONLY | OFlags::CLOEXEC | OFlags::DIRECTORY | OFlags::NOFOLLOW;
        let descriptor = match openat(&directory, name, flags, Mode::empty()) {
            Ok(descriptor) => descriptor,
            Err(error) if create && error == rustix::io::Errno::NOENT => {
                if let Err(error) = mkdirat(&directory, name, Mode::RUSR | Mode::WUSR | Mode::XUSR)
                {
                    if error != rustix::io::Errno::EXIST {
                        return Err(ResolvedCacheError::new(error.to_string()));
                    }
                }
                openat(&directory, name, flags, Mode::empty())
                    .map_err(|error| ResolvedCacheError::new(error.to_string()))?
            }
            Err(error) => return Err(ResolvedCacheError::new(error.to_string())),
        };
        directory = File::from(descriptor);
    }
    Ok((directory, file_name))
}

#[cfg(unix)]
fn create_confined_output(root: &Path, relative: &Path) -> Result<File, ResolvedCacheError> {
    use rustix::fs::{openat, Mode, OFlags};

    let (parent, file_name) = confined_parent(root, relative, true)?;
    let descriptor = openat(
        &parent,
        &file_name,
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    Ok(File::from(descriptor))
}

#[cfg(unix)]
fn hard_link_confined(
    root: &Path,
    existing: &Path,
    destination: &Path,
) -> Result<(), ResolvedCacheError> {
    use rustix::fs::{linkat, AtFlags};

    let (source_parent, source_name) = confined_parent(root, existing, false)?;
    let (destination_parent, destination_name) = confined_parent(root, destination, true)?;
    linkat(
        &source_parent,
        &source_name,
        &destination_parent,
        &destination_name,
        AtFlags::empty(),
    )
    .map_err(|error| ResolvedCacheError::new(error.to_string()))
}

#[cfg(windows)]
fn create_confined_output(root: &Path, relative: &Path) -> Result<File, ResolvedCacheError> {
    use fs_at::OpenOptionsWriteMode;

    let (parent, file_name) = windows_confined_parent(root, relative, true)?;
    let mut options = fs_at::OpenOptions::default();
    options
        .follow(false)
        .write(OpenOptionsWriteMode::Write)
        .create_new(true);
    Ok(options.open_at(&parent, file_name)?)
}

#[cfg(windows)]
fn hard_link_confined(
    _root: &Path,
    _existing: &Path,
    _destination: &Path,
) -> Result<(), ResolvedCacheError> {
    // Windows has no std handle-relative hard-link API. Returning an error deliberately selects
    // the streamed-copy fallback rather than reopening a validated path and reintroducing a
    // junction race.
    Err(ResolvedCacheError::new(
        "safe staging hard links are unavailable on Windows",
    ))
}

#[cfg(not(any(unix, windows)))]
fn create_confined_output(_root: &Path, _relative: &Path) -> Result<File, ResolvedCacheError> {
    Err(ResolvedCacheError::new(
        "resolved-cache materialization is unsupported on this platform",
    ))
}

#[cfg(not(any(unix, windows)))]
fn hard_link_confined(
    _root: &Path,
    _existing: &Path,
    _destination: &Path,
) -> Result<(), ResolvedCacheError> {
    Err(ResolvedCacheError::new(
        "resolved-cache materialization is unsupported on this platform",
    ))
}

#[cfg(windows)]
fn validate_confined_relative_path(relative: &Path) -> Result<(), ResolvedCacheError> {
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ResolvedCacheError::new(
            "materialized output path is not a confined relative path",
        ));
    }
    Ok(())
}

#[cfg(windows)]
fn windows_confined_parent(
    root: &Path,
    relative: &Path,
    create: bool,
) -> Result<(File, std::ffi::OsString), ResolvedCacheError> {
    validate_confined_relative_path(relative)?;
    let mut components = relative.components().collect::<Vec<_>>();
    let file_name = components
        .pop()
        .ok_or_else(|| ResolvedCacheError::new("materialized output path has no file name"))?
        .as_os_str()
        .to_owned();
    let managed_parent = root
        .parent()
        .ok_or_else(|| ResolvedCacheError::new("staging directory has no managed parent"))?;
    let (_, mut directory, _) = windows_confined_directory(root, managed_parent)?;
    for component in components {
        let mut options = fs_at::OpenOptions::default();
        options.read(true).follow(false);
        directory = match options.open_dir_at(&directory, component.as_os_str()) {
            Ok(directory) => directory,
            Err(error) if create && error.kind() == std::io::ErrorKind::NotFound => options
                .mkdir_at(&directory, component.as_os_str())
                .map_err(ResolvedCacheError::from)?,
            Err(error) => return Err(error.into()),
        };
    }
    Ok((directory, file_name))
}

fn validate_configured_source(
    candidate: &PromotionCandidate,
    source_configured_path: &Path,
) -> Result<(), ResolvedCacheError> {
    let snapshot = match &candidate.artifact.location {
        ArtifactLocation::SourceLibrary { root } => root,
        ArtifactLocation::ResolvedLocal { .. } => {
            return Err(ResolvedCacheError::new(
                "promotion candidate is not source-library backed",
            ));
        }
    };
    let library = snapshot
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .ok_or_else(|| ResolvedCacheError::new("source snapshot has no library root"))?;
    if std::fs::canonicalize(library)? != std::fs::canonicalize(source_configured_path)? {
        return Err(ResolvedCacheError::new(
            "configured source library differs from promotion candidate provenance",
        ));
    }
    Ok(())
}

fn cleanup_failed_reservation(
    mut reservation: ResolvedCacheReservation,
) -> Result<(), ResolvedCacheError> {
    if reservation.staging_path().exists() {
        let staging_root = reservation.store.root().join("staging");
        remove_managed_tree(reservation.staging_path(), &staging_root)?;
    }
    reservation.mark_interrupted()?;
    Ok(())
}

fn cancel_reservation(
    reservation: ResolvedCacheReservation,
) -> Result<MaterializationOutcome, ResolvedCacheError> {
    cleanup_failed_reservation(reservation)?;
    Ok(MaterializationOutcome::Cancelled)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PromotionScheduleOutcome {
    Enqueued,
    Coalesced,
    Disabled,
    Full,
    Stopped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IdlePromotionOutcome {
    NotIdle,
    Empty,
    Stopped,
    Materialized(MaterializationOutcome),
}

#[derive(Clone, Debug)]
struct ScheduledPromotion {
    candidate: PromotionCandidate,
    source_configured_path: PathBuf,
    logical_model_owner: String,
}

#[derive(Debug, Default)]
struct PromotionQueueState {
    queue: VecDeque<ScheduledPromotion>,
    keys: BTreeSet<String>,
    active: Option<(String, MaterializationCancellation)>,
    stopped: bool,
}

#[derive(Clone, Debug)]
pub struct ResolvedCachePromotionScheduler {
    policy: ResolvedCachePolicy,
    materializer: ResolvedCacheMaterializer,
    capacity: usize,
    state: Arc<Mutex<PromotionQueueState>>,
}

impl ResolvedCachePromotionScheduler {
    pub fn new(
        policy: ResolvedCachePolicy,
        materializer: ResolvedCacheMaterializer,
        capacity: usize,
    ) -> Result<Self, ResolvedCacheError> {
        policy.validate()?;
        if capacity == 0 {
            return Err(ResolvedCacheError::new(
                "resolved-cache promotion queue capacity must be nonzero",
            ));
        }
        Ok(Self {
            policy,
            materializer,
            capacity,
            state: Arc::new(Mutex::new(PromotionQueueState::default())),
        })
    }

    pub fn schedule(
        &self,
        candidate: PromotionCandidate,
        source_configured_path: PathBuf,
        logical_model_owner: String,
    ) -> Result<PromotionScheduleOutcome, ResolvedCacheError> {
        if !self.policy.enabled {
            return Ok(PromotionScheduleOutcome::Disabled);
        }
        validate_model_owner(&logical_model_owner)?;
        cache_key_digest(&candidate.cache_key)?;
        let mut state = lock_scheduler(&self.state);
        if state.stopped {
            return Ok(PromotionScheduleOutcome::Stopped);
        }
        if state.keys.contains(&candidate.cache_key)
            || state
                .active
                .as_ref()
                .is_some_and(|(key, _)| key == &candidate.cache_key)
        {
            return Ok(PromotionScheduleOutcome::Coalesced);
        }
        if state.queue.len() + usize::from(state.active.is_some()) >= self.capacity {
            return Ok(PromotionScheduleOutcome::Full);
        }
        state.keys.insert(candidate.cache_key.clone());
        state.queue.push_back(ScheduledPromotion {
            candidate,
            source_configured_path,
            logical_model_owner,
        });
        Ok(PromotionScheduleOutcome::Enqueued)
    }

    /// Run one queued promotion only when the caller proves its worker is idle. The caller owns the
    /// thread or blocking-task lifetime; this type never detaches work, and `schedule` itself never
    /// performs source I/O or delays the successful first request.
    pub fn run_next_if_idle(&self, idle: bool) -> Result<IdlePromotionOutcome, ResolvedCacheError> {
        if !idle {
            return Ok(IdlePromotionOutcome::NotIdle);
        }
        let (scheduled, cancellation) = {
            let mut state = lock_scheduler(&self.state);
            if state.stopped {
                return Ok(IdlePromotionOutcome::Stopped);
            }
            if state.active.is_some() {
                return Ok(IdlePromotionOutcome::NotIdle);
            }
            let Some(scheduled) = state.queue.pop_front() else {
                return Ok(IdlePromotionOutcome::Empty);
            };
            state.keys.remove(&scheduled.candidate.cache_key);
            let cancellation = MaterializationCancellation::default();
            state.active = Some((scheduled.candidate.cache_key.clone(), cancellation.clone()));
            (scheduled, cancellation)
        };
        let result = self.materializer.materialize(
            &scheduled.candidate,
            &scheduled.source_configured_path,
            &scheduled.logical_model_owner,
            &cancellation,
        );
        lock_scheduler(&self.state).active = None;
        result.map(IdlePromotionOutcome::Materialized)
    }

    pub fn cancel_active(&self) -> bool {
        let cancellation = lock_scheduler(&self.state)
            .active
            .as_ref()
            .map(|(_, cancellation)| cancellation.clone());
        if let Some(cancellation) = cancellation {
            cancellation.cancel();
            true
        } else {
            false
        }
    }

    pub fn shutdown(&self) {
        let mut state = lock_scheduler(&self.state);
        state.stopped = true;
        state.queue.clear();
        state.keys.clear();
        if let Some((_, cancellation)) = &state.active {
            cancellation.cancel();
        }
    }

    pub fn queued_len(&self) -> usize {
        lock_scheduler(&self.state).queue.len()
    }
}

fn lock_scheduler<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
#[path = "materialization_tests.rs"]
mod tests;
