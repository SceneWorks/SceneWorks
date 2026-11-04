//! Shared relative-path syntax and filesystem confinement. Syntax alone never proves confinement.
use std::path::{Component, Path, PathBuf};
use std::{fs, io};

pub fn portable_relative_path_parts(relative_path: &str) -> Result<PathBuf, &'static str> {
    if relative_path.trim().is_empty() {
        return Err("empty");
    }
    if relative_path.contains('\\') {
        return Err("must use '/' separators");
    }
    if relative_path.contains(':') {
        return Err("must not contain ':'");
    }
    if relative_path.bytes().any(|byte| byte.is_ascii_control()) {
        return Err("must not contain control characters");
    }
    // Named before the generic empty-component rule below so an absolute path is refused as what it
    // is rather than as "a leading empty segment".
    if relative_path.starts_with('/') {
        return Err("must be relative");
    }
    // Split on '/' rather than walking `Path::components()`: components() silently NORMALISES
    // `a//b` and a trailing `/.` away, so it would accept documents the contract validator (which
    // splits) rejects. The looser reading is the one that has to go.
    let mut out = PathBuf::new();
    for part in relative_path.split('/') {
        match part {
            "" => return Err("empty path component"),
            "." => return Err("contains '.'"),
            ".." => return Err("contains '..'"),
            part => out.push(part),
        }
    }
    if out.as_os_str().is_empty() {
        return Err("empty");
    }
    Ok(out)
}

/// Legacy normal-component syntax accepts redundant separators, as Path::components does.
/// Portable document contracts use `portable_relative_path_parts` for strict spelling instead.
pub fn relative_normal_components(relative: &str) -> Result<PathBuf, &'static str> {
    if relative.contains('\\') || relative.contains(':') {
        return Err("must use portable relative components");
    }
    path_normal_components(Path::new(relative))
}

/// Validate a native Path while retaining the raw-string policy for portable document callers.
pub fn path_normal_components(path: &Path) -> Result<PathBuf, &'static str> {
    let portable = path
        .components()
        .map(|component| match component {
            Component::Normal(value) => value.to_str().ok_or("non-UTF-8 component"),
            _ => Err("must contain only normal relative components"),
        })
        .collect::<Result<Vec<_>, _>>()?
        .join("/");
    portable_relative_path_parts(&portable)
}

/// Windows directory junctions and other reparse points are link-bearing entries too.
pub fn is_link_or_reparse_point(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x0000_0400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

#[derive(Clone, Copy)]
pub enum Confinement {
    Lexical,
    SymlinkSafe,
}

/// Join an untrusted native relative path (including Windows native separators).
/// Portable document strings must first use `portable_relative_path_parts` or
/// `relative_normal_components`. SymlinkSafe requires an existing root and resolves existing
/// ancestors, so missing output files are supported without allowing a symlinked parent to
/// redirect them outside the root. Its result uses the canonical root spelling (including a
/// Windows verbatim prefix), not necessarily the caller's lexical spelling.
pub fn confine_under(root: &Path, relative: &str, policy: Confinement) -> io::Result<PathBuf> {
    let parts = path_normal_components(Path::new(relative))
        .map_err(|reason| io::Error::new(io::ErrorKind::InvalidInput, reason))?;
    let candidate = root.join(parts);
    match policy {
        Confinement::Lexical => Ok(candidate),
        Confinement::SymlinkSafe => {
            let root = root.canonicalize()?;
            let resolved = canonicalize_existing_ancestor(&candidate)?;
            if !resolved.starts_with(&root) {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "path escapes its root",
                ));
            }
            Ok(resolved)
        }
    }
}

fn canonicalize_existing_ancestor(path: &Path) -> io::Result<PathBuf> {
    let mut ancestor = path;
    let mut missing = Vec::new();
    loop {
        match fs::canonicalize(ancestor) {
            Ok(mut canonical) => {
                for component in missing.iter().rev() {
                    canonical.push(component);
                }
                return Ok(canonical);
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                // A dangling symlink is not a missing output component.
                if fs::symlink_metadata(ancestor).is_ok() {
                    return Err(error);
                }
                missing.push(ancestor.file_name().ok_or(error)?.to_os_string());
                ancestor = ancestor.parent().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::NotFound, "no existing ancestor")
                })?;
            }
            Err(error) => return Err(error),
        }
    }
}

pub fn resolve_local_file(root: &Path, relative: &str) -> io::Result<PathBuf> {
    let path = confine_under(root, relative, Confinement::SymlinkSafe)?;
    if !path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "confined file was not found",
        ));
    }
    Ok(path)
}

/// Film harness exports intentionally allow only URL-safe ASCII filename characters.
pub fn is_safe_media_path(path: &str) -> bool {
    portable_relative_path_parts(path).is_ok()
        && path
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'_' | b'-' | b'.'))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn portable_and_legacy_syntax_are_explicit() {
        assert!(relative_normal_components("images//frame.png").is_ok());
        assert!(portable_relative_path_parts("images//frame.png").is_err());
        for bad in ["", "../x", "/x", "./x", "C:/x", "a\\b", "a/../b", "a\n"] {
            assert!(relative_normal_components(bad).is_err(), "{bad:?}");
        }
        #[cfg(windows)]
        assert!(path_normal_components(Path::new("images\\frame.png")).is_ok());
        assert!(is_safe_media_path("images/a-1.png"));
        assert!(!is_safe_media_path("images/a b.png"));
    }
    #[test]
    fn existing_and_missing_files_stay_confined() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("inside"), b"valid").unwrap();
        #[cfg(windows)]
        assert_eq!(
            confine_under(temp.path(), "new\\file", Confinement::Lexical).unwrap(),
            temp.path().join("new/file")
        );
        assert!(resolve_local_file(temp.path(), "inside").is_ok());
        assert!(resolve_local_file(temp.path(), "missing").is_err());
        assert!(confine_under(temp.path(), "new/file", Confinement::SymlinkSafe).is_ok());
    }
    #[test]
    fn canonical_root_supports_missing_output_descendants_but_must_exist() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("project");
        let relative = "training/datasets/ds-1/upscaled/item_1.png";
        let error = confine_under(&root, relative, Confinement::SymlinkSafe).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        fs::create_dir(&root).unwrap();
        let canonical = root.canonicalize().unwrap();
        let resolved = confine_under(&root, relative, Confinement::SymlinkSafe).unwrap();
        assert_eq!(resolved, canonical.join(relative));
        assert_eq!(
            resolved.strip_prefix(canonical).unwrap(),
            Path::new(relative)
        );
        assert!(
            !resolved.exists(),
            "confinement must not create output paths"
        );
    }
    #[cfg(windows)]
    #[test]
    fn directory_junction_cannot_escape_confinement() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("root");
        fs::create_dir(&root).unwrap();
        let outside = temp.path().join("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("secret"), b"outside").unwrap();
        let link = root.join("junction");
        assert!(std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(&link)
            .arg(&outside)
            .status()
            .unwrap()
            .success());
        assert!(resolve_local_file(&root, "junction/secret").is_err());
        assert!(confine_under(&root, "junction/missing", Confinement::SymlinkSafe).is_err());
        fs::remove_dir(link).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn symlinks_cannot_escape_even_for_missing_outputs() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("root");
        fs::create_dir(&root).unwrap();
        fs::write(temp.path().join("secret"), b"outside").unwrap();
        std::os::unix::fs::symlink(temp.path(), root.join("escape")).unwrap();
        assert!(resolve_local_file(&root, "escape/secret").is_err());
        assert!(confine_under(&root, "escape/not-yet-created", Confinement::SymlinkSafe).is_err());
        fs::write(root.join("ok"), b"inside").unwrap();
        std::os::unix::fs::symlink(root.join("ok"), root.join("link")).unwrap();
        assert!(resolve_local_file(&root, "link").is_ok());
    }
}
