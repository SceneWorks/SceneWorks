//! Portable free bytes available to this process (including filesystem quotas).
use std::path::Path;

pub fn disk_free_bytes(path: &Path) -> std::io::Result<u64> {
    fs2::available_space(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn probes_existing_filesystems_and_preserves_errors() {
        let dir = tempfile::tempdir().unwrap();
        assert!(disk_free_bytes(dir.path()).unwrap() > 0);
        assert!(disk_free_bytes(&dir.path().join("missing")).is_err());
    }
}
