//! Exercise the production zip dependency against the archive extraction advisories.
#![cfg(any(windows, target_os = "linux"))]

use std::io::{Cursor, Write};
use zip::write::SimpleFileOptions;

fn archive(name: &str, contents: &[u8], mode: u32) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .start_file(name, SimpleFileOptions::default())
        .unwrap();
    writer.write_all(contents).unwrap();
    let mut bytes = writer.finish().unwrap().into_inner();
    // Model an untrusted producer: patch the central header's Unix mode directly.
    // FileOptions intentionally strips privileged bits when creating an archive.
    let header = bytes.windows(4).position(|b| b == b"PK\x01\x02").unwrap();
    bytes[header + 5] = 3; // "version made by": Unix
    bytes[header + 38..header + 42].copy_from_slice(&(mode << 16).to_le_bytes());
    bytes
}

#[test]
fn ordinary_archive_contents_are_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = archive("nested/installer.exe", b"installer fixture", 0o100755);
    zip::ZipArchive::new(Cursor::new(bytes))
        .unwrap()
        .extract(dir.path())
        .unwrap();
    assert_eq!(
        std::fs::read(dir.path().join("nested/installer.exe")).unwrap(),
        b"installer fixture"
    );
}

#[cfg(unix)]
#[test]
fn extracted_files_cannot_gain_privileged_permission_bits() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let bytes = archive("payload", b"fixture", 0o107777);
    zip::ZipArchive::new(Cursor::new(bytes))
        .unwrap()
        .extract(dir.path())
        .unwrap();
    let permissions = std::fs::metadata(dir.path().join("payload"))
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(permissions & 0o7000, 0, "SUID/SGID/sticky bits survived");
}

#[cfg(unix)]
#[test]
fn escaping_symlinks_cannot_modify_an_external_file() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let dest = root.path().join("extract");
    let outside = root.path().join("outside");
    std::fs::write(&outside, b"untouched").unwrap();
    std::fs::set_permissions(&outside, std::fs::Permissions::from_mode(0o600)).unwrap();
    let bytes = archive("escape", b"../outside", 0o120777);
    let result = zip::ZipArchive::new(Cursor::new(bytes))
        .unwrap()
        .extract(&dest);
    assert!(result.is_err(), "an escaping symlink must be rejected");
    assert!(!dest.join("escape").is_symlink());
    assert_eq!(std::fs::read(&outside).unwrap(), b"untouched");
    assert_eq!(
        std::fs::metadata(&outside).unwrap().permissions().mode() & 0o777,
        0o600,
        "extraction changed permissions through an escaping symlink"
    );
}
