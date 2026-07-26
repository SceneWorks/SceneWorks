use std::path::{Component, Path, PathBuf};

use sceneworks_core::project_store::ProjectStore;

pub(crate) struct ResolvedAssetMedia {
    pub path: PathBuf,
    pub display_name: Option<String>,
}

fn lexically_confined_media_path(project_path: &Path, relative: &str) -> Option<PathBuf> {
    let mut path = project_path.to_path_buf();
    for component in Path::new(relative).components() {
        match component {
            Component::Normal(value) => path.push(value),
            _ => return None,
        }
    }
    Some(path)
}

fn confined_media_path(project_path: &Path, relative: &str) -> Option<PathBuf> {
    let project_path = project_path.canonicalize().ok()?;
    let media_path = lexically_confined_media_path(&project_path, relative)?
        .canonicalize()
        .ok()?;
    (media_path.starts_with(&project_path) && media_path.is_file()).then_some(media_path)
}

/// Resolve an asset sidecar's project-relative media path without accepting root,
/// prefix, `.` or `..` components or symlinks that escape the canonical project
/// root. Segment, upscale, and pose all consume the same contract, so confinement
/// must not drift between separate copies.
pub(crate) fn resolve_asset_media(
    store: &ProjectStore,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> Option<ResolvedAssetMedia> {
    let asset = store.get_asset(project_id, asset_id).ok()?;
    let relative = asset.get("file")?.get("path")?.as_str()?;
    let path = confined_media_path(project_path, relative)?;
    Some(ResolvedAssetMedia {
        path,
        display_name: asset
            .get("displayName")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confines_asset_media_paths_to_normal_components() {
        let root = Path::new("/projects/example.sceneworks");
        assert_eq!(
            lexically_confined_media_path(root, "assets/images/source.png"),
            Some(root.join("assets/images/source.png"))
        );
        assert_eq!(lexically_confined_media_path(root, "../secret"), None);
        assert_eq!(lexically_confined_media_path(root, "/tmp/secret"), None);
        assert_eq!(
            lexically_confined_media_path(root, "./assets/source.png"),
            None
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_media_that_escapes_the_project() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().expect("tempdir");
        let project = temp.path().join("project.sceneworks");
        let outside = temp.path().join("outside.png");
        std::fs::create_dir(&project).expect("project dir");
        std::fs::write(&outside, b"not actually an image").expect("outside file");
        symlink(&outside, project.join("linked.png")).expect("symlink");

        assert_eq!(confined_media_path(&project, "linked.png"), None);
    }
}
