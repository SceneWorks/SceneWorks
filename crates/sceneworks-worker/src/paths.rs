//! Path normalization and app-managed-directory confinement helpers (the worker trust boundary).
use super::*;

pub fn safe_download_dir(value: &str) -> String {
    let mut output = String::new();
    let mut in_replacement = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
            output.push(character);
            in_replacement = false;
        } else if !in_replacement {
            output.push_str("__");
            in_replacement = true;
        }
    }
    let output = output.trim_matches('_').to_owned();
    if output.is_empty() {
        "download".to_owned()
    } else {
        output
    }
}

pub(crate) fn safe_join(base: &Path, relative: &str) -> WorkerResult<PathBuf> {
    let mut target = base.to_path_buf();
    for component in Path::new(relative).components() {
        match component {
            std::path::Component::Normal(value) => target.push(value),
            _ => {
                return Err(WorkerError::InvalidPayload(format!(
                    "Unsafe snapshot path: {relative}"
                )))
            }
        }
    }
    Ok(target)
}

/// Confine a payload-supplied weight *filename* that is joined under a resolved HF
/// snapshot / app-cache directory (sc-8821 / F-019): `advanced.controlWeights.filename`
/// on every strict-control lane and `advanced.pidCheckpoint.filename` on the PiD
/// decoder. The repo half of those overrides is already sanitized
/// (`safe_repo_dir_name` slugs separators into `--`), but the filename was joined
/// raw, so a `../../…` (or absolute) filename escaped the snapshot and loaded an
/// arbitrary readable file as weights — reachable over the LAN jobs API (epic 4484),
/// the same primitive `normalize_app_managed_lora_path` closes for LoRA paths. A
/// weight file inside an HF repo snapshot is always a single plain path component,
/// so require exactly that: no separators (`/` or `\`), no `..`/`.`, not absolute.
///
/// Callers all live in `image_jobs` lanes gated macOS or off-Mac + `backend-candle`
/// (`image_jobs/{*_control,pid,*_candle}`), so the bare (non-macOS, non-candle) lib
/// build has none — silence dead_code only there, like `decode_image_any`.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn safe_weight_filename(filename: &str, label: &str) -> WorkerResult<String> {
    let filename = filename.trim();
    let mut components = Path::new(filename).components();
    let single_normal = matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none();
    if single_normal && !filename.contains('/') && !filename.contains('\\') {
        Ok(filename.to_owned())
    } else {
        Err(WorkerError::InvalidPayload(format!(
            "{label} must be a plain filename (no path separators or '..'): {filename}"
        )))
    }
}

pub(crate) fn normalize_absolute_path(path: &Path) -> WorkerResult<PathBuf> {
    let mut output = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir()?
    };
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => output.push(prefix.as_os_str()),
            std::path::Component::RootDir => output.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !output.pop() {
                    return Err(WorkerError::InvalidPayload(format!(
                        "Unsafe absolute path: {}",
                        path.display()
                    )));
                }
            }
            std::path::Component::Normal(value) => output.push(value),
        }
    }
    Ok(output)
}

pub(crate) fn normalized_data_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    normalize_absolute_path(&settings.data_dir)
}

pub(crate) fn ensure_path_under(
    path: PathBuf,
    roots: &[PathBuf],
    label: &str,
) -> WorkerResult<PathBuf> {
    if roots.iter().any(|root| path.starts_with(root)) {
        return Ok(path);
    }
    let allowed = roots
        .iter()
        .map(|root| root.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(WorkerError::InvalidPayload(format!(
        "{label} must be inside an app-managed directory ({allowed})."
    )))
}

pub(crate) fn normalize_app_managed_path(
    settings: &Settings,
    raw_path: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let raw_path = raw_path.trim();
    if raw_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    // Canonicalize before the confinement check so a symlink (or `..`) under the
    // data dir can't satisfy a purely-lexical `starts_with` and land the write
    // outside; allow either the lexical or the canonical root so a not-yet-created
    // target still matches (sc-8877 / F-075), mirroring `normalize_app_managed_lora_path`.
    let data_dir = normalized_data_dir(settings)?;
    let canonical_data_dir = normalize_existing_or_absolute(&settings.data_dir)?;
    let path = normalize_existing_or_absolute(Path::new(raw_path))?;
    ensure_path_under(path, &[data_dir, canonical_data_dir], label)
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn normalize_app_managed_cache_path(
    settings: &Settings,
    raw_path: &str,
    cache_dir: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let raw_path = raw_path.trim();
    if raw_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    let root = settings.data_dir.join("cache").join(cache_dir);
    let normalized_root = normalize_absolute_path(&root)?;
    let canonical_root = normalize_existing_or_absolute(&root)?;
    let normalized = normalize_absolute_path(Path::new(raw_path))?;
    let resolved = normalize_existing_or_absolute(&normalized)?;
    ensure_path_under(resolved, &[normalized_root, canonical_root], label)
}

/// A model's weights are a read-only source the rust-api resolves (e.g.
/// `resolve_base_model_path`) from either the app data dir *or* the shared
/// Hugging Face hub cache — the default `HF_HOME` the desktop injects points the
/// cache at `~/.cache/huggingface`, outside `data_dir`. Unlike output dirs and
/// dataset roots (write targets, confined to `data_dir`), model weights may
/// legitimately live in that cache, so they are allowed under either root. Used
/// for the training base model and every other read-only model dir (captioner,
/// image/InstantID). Without this, an HF-cache-resident model (e.g. z_image_turbo)
/// fails the data-dir-only check even though the install/resolve gates accepted it.
pub(crate) fn normalize_app_managed_model_path(
    settings: &Settings,
    raw_path: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let raw_path = raw_path.trim();
    if raw_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    // Canonicalize the input and allow either the lexical or canonical form of each
    // root, so a symlink/`..` can't slip past the confinement check (sc-8877 / F-075),
    // matching `normalize_app_managed_lora_path` (same roots).
    let data_dir = normalized_data_dir(settings)?;
    let canonical_data_dir = normalize_existing_or_absolute(&settings.data_dir)?;
    let hf_cache = normalize_absolute_path(&huggingface_hub_cache_dir(&settings.data_dir))?;
    let canonical_hf_cache =
        normalize_existing_or_absolute(&huggingface_hub_cache_dir(&settings.data_dir))?;
    let mut roots = vec![data_dir, canonical_data_dir, hf_cache, canonical_hf_cache];
    // Additionally admit the operator's external model roots (epic 10451 / sc-10668). Phase 1
    // widened only the LoRA lane; Phase 2 reads an external ComfyUI **base** model's component
    // files (DiT / text-encoder / VAE) in place from the configured tree, so those paths must
    // resolve under a declared root too. Same posture as `normalize_app_managed_lora_path`: the
    // roots come only from the process env (never a payload — a LAN caller, epic 4484, cannot
    // widen the allow-list), and the list is empty by default and on macOS, so confinement is
    // unchanged for every install that has not opted in. Both lexical + canonical forms are added.
    for root in &settings.external_model_roots {
        roots.push(normalize_absolute_path(root)?);
        if let Ok(canonical) = normalize_existing_or_absolute(root) {
            roots.push(canonical);
        }
    }
    let path = normalize_existing_or_absolute(Path::new(raw_path))?;
    ensure_path_under(path, &roots, label)
}

/// Confine a LoRA adapter path taken from a job payload to an app-managed root
/// (sc-5723 / WKA-002). The path arrives untrusted (`installedPath`/`sourcePath`/
/// `path`/`source.path` on a LoRA spec) and is loaded as adapter weights, so —
/// like every other on-disk model input — it must resolve under the app data dir
/// or the shared Hugging Face hub cache (installed LoRAs live in `<data>/loras` or
/// a project tree under `<data>`; HF-cached adapters live in the hub cache).
/// Without this a crafted payload could point a LoRA at any `.safetensors` on the
/// host, giving the worker an arbitrary-file read primitive across the API boundary.
///
/// Additionally admits `settings.external_model_roots` (epic 10451 / sc-10452) — an
/// operator's existing ComfyUI `models/` tree, read in place. Those roots come only
/// from the process environment, never from a payload, so a LAN caller (epic 4484)
/// still cannot widen the allow-list; they merely name directories the deployment
/// has already declared readable. The list is empty by default and on macOS, so the
/// confinement is unchanged for every install that has not opted in. Phase 2 (sc-10668)
/// widened `normalize_app_managed_model_path` the same way for external ComfyUI **base**
/// model component files; both lanes now admit the declared external roots.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn normalize_app_managed_lora_path(
    settings: &Settings,
    path: &Path,
) -> WorkerResult<PathBuf> {
    let data_dir = normalized_data_dir(settings)?;
    let canonical_data_dir = normalize_existing_or_absolute(&settings.data_dir)?;
    let hf_cache = normalize_absolute_path(&huggingface_hub_cache_dir(&settings.data_dir))?;
    let canonical_hf_cache =
        normalize_existing_or_absolute(&huggingface_hub_cache_dir(&settings.data_dir))?;
    let mut roots = vec![data_dir, canonical_data_dir, hf_cache, canonical_hf_cache];
    // Both the lexical and canonical form of each external root, matching the posture
    // above: `resolved` is canonical, and a canonical path never `starts_with` a
    // lexical root when the two differ (a symlinked or `..`-bearing root, macOS
    // `/var` -> `/private/var`). A root that cannot be canonicalized (unmounted drive)
    // contributes its lexical form only, and simply never matches.
    for root in &settings.external_model_roots {
        roots.push(normalize_absolute_path(root)?);
        if let Ok(canonical) = normalize_existing_or_absolute(root) {
            roots.push(canonical);
        }
    }
    let normalized = normalize_absolute_path(path)?;
    let resolved = normalize_existing_or_absolute(&normalized)?;
    ensure_path_under(resolved, &roots, "LoRA path")
}

pub(crate) fn normalize_existing_or_absolute(path: &Path) -> WorkerResult<PathBuf> {
    match std::fs::canonicalize(path) {
        Ok(canonical) => normalize_absolute_path(&canonical),
        // `canonicalize` fails `NotFound` as soon as *any* component (leaf or an
        // intermediate dir) is absent, so a purely-lexical fallback would leave
        // intermediate symlinks unresolved: a planted `<data>/loras/evil -> /outside`
        // with a not-yet-created leaf `<data>/loras/evil/newdir` would still satisfy a
        // lexical `starts_with(<data>)` and escape the managed root (sc-9812 / F-075
        // follow-up). Instead canonicalize the deepest *existing* ancestor — resolving
        // every symlink in the real portion of the path — then re-append the missing
        // tail lexically, so a legitimate not-yet-created leaf under the root still
        // resolves while an intermediate-symlink escape is caught by the confinement
        // check that runs on the returned path.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            normalize_nonexistent_via_ancestor(path)
        }
        Err(error) => Err(error.into()),
    }
}

/// Resolve a path whose leaf (and possibly deeper ancestors) does not yet exist by
/// canonicalizing the deepest existing ancestor and re-appending the missing tail.
/// This resolves any symlink in the *existing* portion of the path (closing the
/// intermediate-symlink escape lexical normalization missed) while still returning a
/// usable path for a legitimate not-yet-created target under a managed root.
fn normalize_nonexistent_via_ancestor(path: &Path) -> WorkerResult<PathBuf> {
    // Start from a lexically-absolute, `.`/`..`-collapsed form so the tail we peel off
    // is composed only of `Normal` components (no `..` to re-introduce a traversal
    // after the existing prefix is canonicalized).
    let absolute = normalize_absolute_path(path)?;
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut ancestor = absolute.as_path();
    loop {
        match std::fs::canonicalize(ancestor) {
            Ok(canonical) => {
                let mut resolved = canonical;
                for component in tail.iter().rev() {
                    resolved.push(component);
                }
                return normalize_absolute_path(&resolved);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match (ancestor.file_name(), ancestor.parent()) {
                    (Some(name), Some(parent)) => {
                        tail.push(name.to_owned());
                        ancestor = parent;
                    }
                    // Reached the filesystem root without finding an existing ancestor
                    // (e.g. a path on a non-mounted volume): fall back to the lexical
                    // form. There is no symlink to resolve, so this cannot escape.
                    _ => return Ok(absolute),
                }
            }
            Err(error) => return Err(error.into()),
        }
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn looks_like_huggingface_repo(value: &str) -> bool {
    let value = value.trim();
    if value.is_empty() || value.contains('\\') || Path::new(value).is_absolute() {
        return false;
    }
    let mut parts = value.split('/');
    let Some(owner) = parts.next() else {
        return false;
    };
    let Some(repo) = parts.next() else {
        return false;
    };
    !owner.is_empty()
        && !repo.is_empty()
        && parts.next().is_none()
        && ![owner, repo]
            .iter()
            .any(|part| *part == "." || *part == "..")
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn resolve_app_managed_model_dir(
    settings: &Settings,
    model_name_or_path: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let model_name_or_path = model_name_or_path.trim();
    if model_name_or_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, model_name_or_path) {
        return Ok(snapshot);
    }
    if looks_like_huggingface_repo(model_name_or_path) {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} snapshot is not cached for {model_name_or_path}."
        )));
    }
    let path = normalize_app_managed_model_path(settings, model_name_or_path, label)?;
    if path.is_dir() {
        return Ok(path);
    }
    if path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} must be a snapshot directory, not a file: {}",
            path.display()
        )));
    }
    Err(WorkerError::InvalidPayload(format!(
        "{label} is not installed at {}.",
        path.display()
    )))
}

pub(crate) fn resolve_training_output_dir(
    settings: &Settings,
    output_dir: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    // `normalize_app_managed_path` now returns the canonicalized path (sc-8877), so
    // build each sub-root in both its lexical and canonical form to keep this second
    // `starts_with` consistent (a canonical path never starts_with a lexical root
    // when the two differ, e.g. macOS `/var` -> `/private/var`).
    let path = normalize_app_managed_path(settings, output_dir, label)?;
    let data_dir = normalized_data_dir(settings)?;
    // Global-scope outputs land in `<data>/loras` (or `<data>/models` for full
    // fine-tunes); project-scope outputs — the default — land in the owning
    // project's tree, `<data>/projects/<slug>.sceneworks/loras/<lora_id>`, which
    // `resolve_training_output_location` computes API-side from trusted inputs.
    // All three stay inside the app data dir, so allow the projects tree too
    // rather than rejecting every project-scoped run.
    let mut allowed_roots = Vec::with_capacity(6);
    for sub in ["loras", "models", "projects"] {
        allowed_roots.push(data_dir.join(sub));
        allowed_roots.push(normalize_existing_or_absolute(
            &settings.data_dir.join(sub),
        )?);
    }
    ensure_path_under(path, &allowed_roots, label)
}

pub(crate) fn resolve_dataset_item_path(
    settings: &Settings,
    dataset_root: &str,
    image_path: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    // `root` is now canonicalized (sc-8877), so canonicalize the resolved image path
    // too — otherwise an absolute image path stays lexical (e.g. macOS `/var/...`)
    // and never starts_with a canonical root (`/private/var/...`). Canonicalizing the
    // image also closes the symlink-escape the lexical check missed.
    let root = normalize_app_managed_path(settings, dataset_root, "Dataset root")?;
    let raw_image = Path::new(image_path.trim());
    if image_path.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    let path = if raw_image.is_absolute() {
        normalize_existing_or_absolute(raw_image)?
    } else {
        normalize_existing_or_absolute(&root.join(raw_image))?
    };
    ensure_path_under(path, &[root], label)
}

pub(crate) fn project_path_for_payload(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<Option<PathBuf>> {
    let Some(project_id) = optional_payload_string(payload, "projectId") else {
        return Ok(None);
    };
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(project_id)?;
    Ok(Some(PathBuf::from(project.path)))
}

/// Confine a client-supplied import *source* path (sc-8803 / F-002). LoRA/model
/// import jobs arrive over the unauthenticated local jobs API (LAN-exposed via
/// epic 4484), and the worker copies — or with `uploadedSourcePath: true`, moves —
/// the source into an app-listable target dir, so an unconfined source is an
/// arbitrary-file-read/exfiltration primitive (and move mode deletes the original).
/// The rust-api validates `sourcePath` at job creation, but the worker is the
/// stated trust boundary and must re-confine:
/// - uploaded sources (move mode) must live in the API's staged-upload cache,
///   `<data>/cache/<upload_cache>` (`lora-uploads` / `model-uploads`), matching
///   `cleanup_uploaded_import_source`;
/// - copy-mode sources must live under the app data dir or, for project-scoped
///   imports, the owning project's `loras` tree (resolved from the trusted
///   project store, mirroring `resolve_lora_import_target`).
///
/// Symlinks resolve before the root check, like `normalize_app_managed_lora_path`.
pub(crate) fn resolve_import_source_path(
    settings: &Settings,
    payload: &JsonObject,
    raw_path: &str,
    upload_cache: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let raw_path = raw_path.trim();
    if raw_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    let mut roots = Vec::new();
    if payload_bool(payload, "uploadedSourcePath") {
        let upload_root = settings.data_dir.join("cache").join(upload_cache);
        roots.push(normalize_absolute_path(&upload_root)?);
        roots.push(normalize_existing_or_absolute(&upload_root)?);
    } else {
        roots.push(normalized_data_dir(settings)?);
        roots.push(normalize_existing_or_absolute(&settings.data_dir)?);
        if let Some(project_path) = project_path_for_payload(settings, payload)? {
            let project_loras = project_path.join("loras");
            roots.push(normalize_absolute_path(&project_loras)?);
            roots.push(normalize_existing_or_absolute(&project_loras)?);
        }
    }
    let normalized = normalize_absolute_path(Path::new(raw_path))?;
    let resolved = normalize_existing_or_absolute(&normalized)?;
    ensure_path_under(resolved, &roots, label)
}

pub(crate) fn resolve_lora_import_target(
    settings: &Settings,
    payload: &JsonObject,
    fallback_target: PathBuf,
) -> WorkerResult<PathBuf> {
    let requested = optional_payload_string(payload, "targetDir")
        .map(PathBuf::from)
        .unwrap_or(fallback_target);
    // Canonicalize the target before the confinement check so a symlink/`..` under
    // the managed root can't pass a purely-lexical `starts_with` and redirect the
    // write outside; allow each root's lexical or canonical form so a not-yet-created
    // target dir still matches (sc-8877 / F-075).
    let target = normalize_existing_or_absolute(&requested)?;
    let loras_root = settings.data_dir.join("loras");
    let mut allowed_roots = vec![
        normalize_absolute_path(&loras_root)?,
        normalize_existing_or_absolute(&loras_root)?,
    ];
    if let Some(project_path) = project_path_for_payload(settings, payload)? {
        let project_imports = project_path.join("loras").join("imports");
        allowed_roots.push(normalize_absolute_path(&project_imports)?);
        allowed_roots.push(normalize_existing_or_absolute(&project_imports)?);
    }
    if allowed_roots.iter().any(|root| target.starts_with(root)) {
        return Ok(target);
    }
    Err(WorkerError::InvalidPayload(
        "LoRA import targetDir must be inside app-managed data/loras or project/loras/imports"
            .to_owned(),
    ))
}

pub(crate) fn resolve_model_import_target(
    settings: &Settings,
    payload: &JsonObject,
    fallback_target: PathBuf,
) -> WorkerResult<PathBuf> {
    let requested = optional_payload_string(payload, "targetDir")
        .map(PathBuf::from)
        .unwrap_or(fallback_target);
    // Canonicalize before the confinement check so a symlink/`..` can't escape the
    // lexical `starts_with`; allow the managed root's lexical or canonical form so a
    // not-yet-created target still matches (sc-8877 / F-075).
    let target = normalize_existing_or_absolute(&requested)?;
    let models_root = settings.data_dir.join("models");
    let allowed_roots = [
        normalize_absolute_path(&models_root)?,
        normalize_existing_or_absolute(&models_root)?,
    ];
    if allowed_roots.iter().any(|root| target.starts_with(root)) {
        return Ok(target);
    }
    Err(WorkerError::InvalidPayload(
        "Model import targetDir must be inside app-managed data/models".to_owned(),
    ))
}

pub(crate) fn resolve_model_convert_output(
    settings: &Settings,
    output_dir: &str,
) -> WorkerResult<PathBuf> {
    // Canonicalize before the confinement check so a symlink/`..` can't escape the
    // lexical `starts_with`; allow the managed root's lexical or canonical form so a
    // not-yet-created output dir still matches (sc-8877 / F-075).
    let target = normalize_existing_or_absolute(Path::new(output_dir))?;
    let models_root = settings.data_dir.join("models");
    let allowed_roots = [
        normalize_absolute_path(&models_root)?,
        normalize_existing_or_absolute(&models_root)?,
    ];
    if allowed_roots.iter().any(|root| target.starts_with(root)) {
        return Ok(target);
    }
    Err(WorkerError::InvalidPayload(
        "Model convert outputDir must be inside app-managed data/models".to_owned(),
    ))
}

pub(crate) fn model_manifest_target(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<PathBuf> {
    let manifest_path = normalize_absolute_path(&PathBuf::from(required_payload_string(
        payload,
        "manifestPath",
    )?))?;
    let allowed = [normalize_absolute_path(
        &settings
            .config_dir
            .join("manifests")
            .join("user.models.jsonc"),
    )?];
    if allowed.iter().any(|path| path == &manifest_path) {
        return Ok(manifest_path);
    }
    Err(WorkerError::InvalidPayload(
        "Model manifestPath must target the global user model manifest".to_owned(),
    ))
}

pub(crate) fn lora_manifest_target(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<PathBuf> {
    let manifest_path = normalize_absolute_path(&PathBuf::from(required_payload_string(
        payload,
        "manifestPath",
    )?))?;
    let mut allowed = vec![normalize_absolute_path(
        &settings
            .config_dir
            .join("manifests")
            .join("user.loras.jsonc"),
    )?];
    if let Some(project_path) = project_path_for_payload(settings, payload)? {
        allowed.push(normalize_absolute_path(
            &project_path.join("loras").join("manifest.jsonc"),
        )?);
    }
    if allowed.iter().any(|path| path == &manifest_path) {
        return Ok(manifest_path);
    }
    Err(WorkerError::InvalidPayload(
        "LoRA manifestPath must target the global user manifest or the selected project's LoRA manifest"
            .to_owned(),
    ))
}

pub(crate) fn safe_project_path(project_path: &Path, relative: &str) -> WorkerResult<PathBuf> {
    if relative.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Project-relative path is required.".to_owned(),
        ));
    }
    let mut path = project_path.to_path_buf();
    for component in Path::new(relative).components() {
        match component {
            std::path::Component::Normal(value) => path.push(value),
            _ => {
                return Err(WorkerError::InvalidPayload(format!(
                    "Unsafe project-relative path: {relative}"
                )))
            }
        }
    }
    Ok(path)
}

pub(crate) fn relative_path(root: &Path, path: &Path) -> WorkerResult<String> {
    // Project media paths are app-created filenames; keep recipe metadata best-effort
    // if a host path contains non-UTF-8 bytes.
    Ok(path
        .strip_prefix(root)
        .map_err(|_| WorkerError::InvalidPayload("Path is outside project.".to_owned()))?
        .to_string_lossy()
        .replace('\\', "/"))
}
