//! ProjectStore registry aggregate.

use super::*;

pub const PROJECT_FOLDERS: &[&str] = &[
    "assets/images",
    "assets/tensors",
    "assets/videos",
    "assets/uploads",
    "assets/frames",
    "assets/renders",
    "assets/documents",
    "assets/poses",
    "assets/keypoints",
    "assets/depth",
    "characters",
    "generation-sets",
    "loras",
    "person-tracks",
    "recipes",
    "timelines",
    "films/drafts",
    "films/draft-assets",
    "films/runs",
    "training/datasets",
    "training/uploads",
    "trash",
    "cache",
];

/// `recent-projects.json` contains only one small metadata record per project.
/// Eight MiB accommodates many thousands of projects even with long paths while
/// bounding every cache fingerprint/read if the file is externally corrupted
/// or replaced with an unexpectedly large payload.
pub(super) const MAX_REGISTRY_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(super) struct RegistryItem {
    id: Option<String>,
    pub(super) name: Option<String>,
    pub(super) path: Option<String>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RegistryFingerprint {
    modified: SystemTime,
    len: u64,
    // `modified + len` misses in-place same-length rewrites when a tool
    // preserves mtime. Unix ctime is the filesystem change revision; dev+ino
    // also distinguish atomic replacement by another process.
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
    // Stable Rust does not expose the filesystem ChangeTime on Windows, and
    // this workspace forbids unsafe Win32 calls. File identity covers atomic
    // replacement; a fixed-size digest from the same open handle covers every
    // in-place rewrite even when length and last-write time are restored.
    #[cfg(windows)]
    volume_serial_number: u64,
    #[cfg(windows)]
    file_index: u64,
    #[cfg(windows)]
    content_sha256: [u8; 32],
}

#[derive(Debug, Default)]
pub(super) struct RegistryCache {
    fingerprint: Option<RegistryFingerprint>,
    pub(super) items: Vec<RegistryItem>,
    loaded: bool,
    #[cfg(test)]
    pub(super) disk_reads: usize,
}

fn registry_fingerprint(path: &Path) -> ProjectStoreResult<Option<RegistryFingerprint>> {
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::RegistryOpen);
    match fs::File::open(path) {
        Ok(file) => {
            #[cfg(windows)]
            let mut file = file;
            record_asset_list_filesystem_operation(
                AssetListFilesystemOperation::RegistryMetadataRead,
            );
            let metadata = file.metadata()?;
            ensure_registry_size(metadata.len())?;
            #[cfg(windows)]
            let file_information = winapi_util::file::information(&file)?;
            #[cfg(windows)]
            let content_sha256 = registry_content_sha256(&mut file)?;
            Ok(Some(RegistryFingerprint {
                modified: metadata.modified()?,
                len: metadata.len(),
                #[cfg(unix)]
                device: metadata.dev(),
                #[cfg(unix)]
                inode: metadata.ino(),
                #[cfg(unix)]
                changed_seconds: metadata.ctime(),
                #[cfg(unix)]
                changed_nanoseconds: metadata.ctime_nsec(),
                #[cfg(windows)]
                volume_serial_number: file_information.volume_serial_number(),
                #[cfg(windows)]
                file_index: file_information.file_index(),
                #[cfg(windows)]
                content_sha256,
            }))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn ensure_registry_size(len: u64) -> ProjectStoreResult<()> {
    if len > MAX_REGISTRY_BYTES {
        return Err(ProjectStoreError::Io(registry_size_error(len)));
    }
    Ok(())
}

fn registry_size_error(len: u64) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!(
            "recent-projects.json is {len} bytes; the maximum supported size is \
             {MAX_REGISTRY_BYTES} bytes"
        ),
    )
}

#[derive(Default)]
struct RegistrySizeCounter {
    len: u64,
}

impl Write for RegistrySizeCounter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let next_len = self
            .len
            .saturating_add(u64::try_from(buffer.len()).unwrap_or(u64::MAX));
        if next_len > MAX_REGISTRY_BYTES {
            return Err(registry_size_error(next_len));
        }
        self.len = next_len;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn ensure_registry_serialized_size(value: &Value) -> ProjectStoreResult<()> {
    let mut counter = RegistrySizeCounter::default();
    serde_json::to_writer_pretty(&mut counter, value)?;
    counter.write_all(b"\n")?;
    Ok(())
}

#[cfg(windows)]
fn registry_content_sha256(file: &mut fs::File) -> ProjectStoreResult<[u8; 32]> {
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::RegistryContentRead);
    let mut hasher = Sha256::new();
    let mut limited = file.take(MAX_REGISTRY_BYTES + 1);
    crate::hashing::update_sha256(&mut limited, &mut hasher)?;
    ensure_registry_size(MAX_REGISTRY_BYTES + 1 - limited.limit())?;
    Ok(hasher.finalize().into())
}

pub(super) fn read_registry_payload(path: &Path) -> ProjectStoreResult<String> {
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::RegistryOpen);
    let file = fs::File::open(path)?;
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::RegistryMetadataRead);
    let len = file.metadata()?.len();
    ensure_registry_size(len)?;
    let mut payload = String::with_capacity(
        usize::try_from(len).expect("registry byte ceiling fits every supported target"),
    );
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::RegistryContentRead);
    file.take(MAX_REGISTRY_BYTES + 1)
        .read_to_string(&mut payload)?;
    ensure_registry_size(u64::try_from(payload.len()).unwrap_or(u64::MAX))?;
    Ok(payload)
}

pub(super) fn read_project_summary(project_path: &Path) -> ProjectStoreResult<ProjectSummary> {
    let project_file = project_path.join("project.json");
    if !project_file.exists() {
        return Err(ProjectStoreError::NotFound(
            "Project file not found".to_owned(),
        ));
    }
    let payload = read_json(&project_file)?;
    Ok(ProjectSummary {
        id: required_str(&payload, "id")?.to_owned(),
        name: required_str(&payload, "name")?.to_owned(),
        path: project_path.display().to_string(),
        created_at: required_str(&payload, "createdAt")?.to_owned(),
    })
}

fn write_project_file(
    app_version: &str,
    project_path: &Path,
    project_id: &str,
    name: &str,
) -> ProjectStoreResult<()> {
    let folders = PROJECT_FOLDERS
        .iter()
        .map(|folder| {
            (
                folder.replace('/', "_"),
                Value::String((*folder).to_owned()),
            )
        })
        .collect::<Map<_, _>>();
    write_json(
        &project_path.join("project.json"),
        &json!({
            "schemaVersion": 1,
            "appVersion": app_version,
            "id": project_id,
            "name": name,
            "createdAt": utc_now(),
            "folders": folders
        }),
    )
}

impl ProjectStore {
    pub fn new(data_dir: impl Into<PathBuf>, app_version: impl Into<String>) -> Self {
        Self {
            data_dir: data_dir.into(),
            app_version: app_version.into(),
            lock: Mutex::new(RegistryCache::default()),
        }
    }

    pub fn projects_dir(&self) -> PathBuf {
        self.data_dir.join("projects")
    }

    pub fn registry_path(&self) -> PathBuf {
        self.data_dir.join("recent-projects.json")
    }

    pub fn list_projects(&self) -> ProjectStoreResult<Vec<ProjectSummary>> {
        self.list_projects_inner(true)
    }

    /// List existing projects without initializing the app-managed data directories.
    /// Background source inventory uses this path while inspecting a fresh workspace.
    pub fn list_projects_read_only(&self) -> ProjectStoreResult<Vec<ProjectSummary>> {
        self.list_projects_inner(false)
    }

    fn list_projects_inner(
        &self,
        initialize_data_dirs: bool,
    ) -> ProjectStoreResult<Vec<ProjectSummary>> {
        let mut registry_cache = self.lock.lock();
        if initialize_data_dirs {
            self.ensure_data_dirs()?;
        }
        let mut projects = Vec::new();
        for item in self.load_registry(&mut registry_cache)? {
            // The reserved global pose + keypoint libraries are addressable by id but
            // hidden from the project switcher (epic 2282 / epic 4422).
            if matches!(
                item.id.as_deref(),
                Some(GLOBAL_POSES_PROJECT_ID) | Some(GLOBAL_KEYPOINTS_PROJECT_ID)
            ) {
                continue;
            }
            let Some(path) = item.path else {
                continue;
            };
            let project_path = PathBuf::from(path);
            if project_path.exists() {
                // Fail open per entry: one project whose `project.json` is missing,
                // unreadable, or short a required field must not take down the whole
                // listing and leave every healthy project unreachable. Skip the bad
                // entry with a structured warning and keep the rest.
                match read_project_summary(&project_path) {
                    Ok(summary) => projects.push(summary),
                    Err(error) => tracing::warn!(
                        event = "list_projects_summary_failed",
                        project = %project_path.display(),
                        error = %error,
                        "skipping project with unreadable project.json"
                    ),
                }
            }
        }
        Ok(projects)
    }

    pub fn create_project(&self, name: &str) -> ProjectStoreResult<ProjectSummary> {
        let name = name.trim();
        if name.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                "Project name is required".to_owned(),
            ));
        }
        if name.chars().count() > 120 {
            return Err(ProjectStoreError::BadRequest(
                "Project name must be at most 120 characters".to_owned(),
            ));
        }

        let mut registry_cache = self.lock.lock();
        self.ensure_data_dirs()?;
        // Fail fast with an actionable error if the workspace folder rejects writes,
        // rather than surfacing an opaque SQLITE_READONLY from deep in provisioning
        // after a half-built project dir already exists (issue #1435 / sc-11855).
        if let Err(error) = probe_storage_writable(&self.projects_dir()) {
            if is_write_denied(&error) {
                return Err(ProjectStoreError::StorageNotWritable(
                    storage_not_writable_message(&self.projects_dir()),
                ));
            }
            return Err(error);
        }
        let project_id = format!("project_{}", random_hex(16)?);
        let slug = slugify(name, "project", None);
        let mut project_path = self.projects_dir().join(format!("{slug}.sceneworks"));
        if project_path.exists() {
            let suffix = &project_id[project_id.len().saturating_sub(8)..];
            project_path = self
                .projects_dir()
                .join(format!("{slug}-{suffix}.sceneworks"));
        }

        self.provision_project_locked(&project_id, name, &project_path, &mut registry_cache)
    }

    /// Provision a project directory (folders + project file + db + registry entry)
    /// for a known id/path. Caller must hold `self.lock`. Shared by `create_project`
    /// (random id) and `ensure_global_poses_project` (fixed id).
    pub(super) fn provision_project_locked(
        &self,
        project_id: &str,
        name: &str,
        project_path: &Path,
        registry_cache: &mut RegistryCache,
    ) -> ProjectStoreResult<ProjectSummary> {
        fs::create_dir_all(project_path)?;
        for folder in PROJECT_FOLDERS {
            fs::create_dir_all(project_path.join(folder))?;
        }
        write_project_file(&self.app_version, project_path, project_id, name)?;
        // Backstop for the create_project preflight (and the only guard on the
        // lazily-provisioned global pose/keypoint projects): translate a write
        // denial on the project.db migration into the actionable, path-naming
        // error instead of a raw SQLITE_READONLY (issue #1435 / sc-11855).
        if let Err(error) = connect_project_db_migrated(project_path).and_then(|mut connection| {
            let conn = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            // A newly provisioned project has no legacy sidecars to backfill;
            // mark its empty index complete so the first indexed write is not
            // immediately erased by an unnecessary rebuild.
            conn.execute(
                "insert or replace into project_metadata (key, value) values (?1, ?2)",
                params![ASSET_INDEX_VERSION_KEY, PROJECT_SCHEMA_VERSION.to_string()],
            )?;
            conn.commit()?;
            Ok(())
        }) {
            return Err(if is_write_denied(&error) {
                ProjectStoreError::StorageNotWritable(storage_not_writable_message(
                    project_path.parent().unwrap_or(project_path),
                ))
            } else {
                error
            });
        }

        let mut registry = self
            .load_registry(registry_cache)?
            .into_iter()
            .filter(|item| item.id.as_deref() != Some(project_id))
            .collect::<Vec<_>>();
        registry.insert(
            0,
            RegistryItem {
                id: Some(project_id.to_owned()),
                name: Some(name.to_owned()),
                path: Some(project_path.display().to_string()),
                extra: Map::new(),
            },
        );
        self.save_registry(&registry, registry_cache)?;

        read_project_summary(project_path)
    }

    pub fn get_project(&self, project_id: &str) -> ProjectStoreResult<ProjectSummary> {
        let project_path = self.find_project_path(project_id)?;
        read_project_summary(&project_path)
    }

    pub fn project_stem(&self, project_id: &str) -> ProjectStoreResult<String> {
        let project_path = self.find_project_path(project_id)?;
        Ok(project_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_owned())
    }

    pub(super) fn ensure_data_dirs(&self) -> ProjectStoreResult<()> {
        fs::create_dir_all(self.projects_dir())?;
        for folder in ["models", "loras", "cache"] {
            fs::create_dir_all(self.data_dir.join(folder))?;
        }
        Ok(())
    }

    pub(super) fn load_registry(
        &self,
        registry_cache: &mut RegistryCache,
    ) -> ProjectStoreResult<Vec<RegistryItem>> {
        let registry_path = self.registry_path();
        let fingerprint = registry_fingerprint(&registry_path)?;
        if registry_cache.loaded && registry_cache.fingerprint == fingerprint {
            return Ok(registry_cache.items.clone());
        }
        let items = if fingerprint.is_some() {
            #[cfg(test)]
            {
                registry_cache.disk_reads += 1;
            }
            let payload = read_registry_payload(&registry_path)?;
            serde_json::from_str(&payload)?
        } else {
            Vec::new()
        };
        registry_cache.fingerprint = fingerprint;
        registry_cache.items = items.clone();
        registry_cache.loaded = true;
        Ok(items)
    }

    fn save_registry(
        &self,
        projects: &[RegistryItem],
        registry_cache: &mut RegistryCache,
    ) -> ProjectStoreResult<()> {
        fs::create_dir_all(&self.data_dir)?;
        // `write_json` stages to a unique temp and renames atomically (sc-1633),
        // so the registry write no longer needs its own intermediate temp file.
        let registry_path = self.registry_path();
        let registry_value = serde_json::to_value(projects)?;
        ensure_registry_serialized_size(&registry_value)?;
        write_json(&registry_path, &registry_value)?;
        registry_cache.fingerprint = registry_fingerprint(&registry_path)?;
        registry_cache.items = projects.to_vec();
        registry_cache.loaded = true;
        Ok(())
    }

    /// Resolves the project path and acquires the per-project file lock, so the
    /// caller's read-modify-write of that project's JSON/sidecar files is
    /// serialized against other threads mutating the same project (sc-1633).
    /// Read-only paths keep using [`find_project_path`] and don't take the lock.
    pub(super) fn lock_project(
        &self,
        project_id: &str,
    ) -> ProjectStoreResult<(PathBuf, ReentrantMutexGuard<'static, ()>)> {
        let project_path = self.find_project_path(project_id)?;
        let guard = lock_project_files(&project_path);
        Ok((project_path, guard))
    }

    pub(super) fn find_project_path(&self, project_id: &str) -> ProjectStoreResult<PathBuf> {
        let mut registry_cache = self.lock.lock();
        for item in self.load_registry(&mut registry_cache)? {
            if item.id.as_deref() == Some(project_id) {
                let Some(path) = item.path else {
                    break;
                };
                let project_path = PathBuf::from(path);
                record_asset_list_filesystem_operation(AssetListFilesystemOperation::PathStat);
                if project_path.exists() {
                    return Ok(project_path);
                }
                break;
            }
        }
        Err(ProjectStoreError::NotFound("Project not found".to_owned()))
    }
}
