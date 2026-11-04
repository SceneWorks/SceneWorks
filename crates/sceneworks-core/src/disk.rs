//! Portable free bytes available to this process (including filesystem quotas).
use std::path::Path;

pub fn disk_free_bytes(path: &Path) -> std::io::Result<u64> {
    // fs2's Windows volume lookup can succeed for a nonexistent path. Require the
    // caller's path to exist so probe errors retain the same contract on every host.
    std::fs::metadata(path)?;
    fs2::available_space(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn probes_existing_filesystems_and_preserves_errors() {
        let dir = tempfile::tempdir().unwrap();
        assert!(disk_free_bytes(dir.path()).unwrap() > 0);
        let file = dir.path().join("existing-file");
        std::fs::write(&file, b"disk probe fixture").unwrap();
        assert!(disk_free_bytes(&file).unwrap() > 0);
        for invalid in [
            dir.path().join("missing"),
            dir.path().join("missing/descendant"),
            file.join("not-a-directory"),
        ] {
            let expected = std::fs::metadata(&invalid).unwrap_err();
            let actual = disk_free_bytes(&invalid).unwrap_err();
            assert_eq!(actual.kind(), expected.kind(), "{invalid:?}");
            assert_eq!(
                actual.raw_os_error(),
                expected.raw_os_error(),
                "{invalid:?}"
            );
        }
    }
}
