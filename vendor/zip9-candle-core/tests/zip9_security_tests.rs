//! Synthetic container compatibility/security checks. No model, GPU, or real weights.
use candle_core::{Device, Tensor};
use std::io::{Cursor, Write};
use std::path::PathBuf;
use zip::{write::SimpleFileOptions, CompressionMethod, ZipArchive, ZipWriter};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let name = format!(
            "zip9-security-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(name);
        std::fs::create_dir(&root).unwrap();
        Self(root)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn archive(name: &str, bytes: &[u8]) -> Vec<u8> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .start_file(
            name,
            SimpleFileOptions::default().compression_method(CompressionMethod::Stored),
        )
        .unwrap();
    writer.write_all(bytes).unwrap();
    writer.finish().unwrap().into_inner()
}

#[test]
fn unicode_npz_roundtrip_and_lazy_names() {
    let scratch = Scratch::new();
    let file = scratch.0.join("unicode.npz");
    let tensor = Tensor::new(&[1f32, 2f32, 3f32], &Device::Cpu).unwrap();
    Tensor::write_npz(&[("音声_π", &tensor)], &file).unwrap();
    let loaded = Tensor::read_npz(&file).unwrap();
    assert_eq!(loaded[0].0, "音声_π");
    assert_eq!(loaded[0].1.to_vec1::<f32>().unwrap(), [1., 2., 3.]);
    let lazy = candle_core::npy::NpzTensors::new(&file).unwrap();
    assert_eq!(lazy.names(), [&"音声_π".to_string()]);
    assert_eq!(
        lazy.get("音声_π")
            .unwrap()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        [1., 2., 3.]
    );
}

#[test]
fn unicode_pth_container_prefix_preserves_tensor_values() {
    let scratch = Scratch::new();
    let file = scratch.0.join("unicode.pt");
    let mut source = ZipArchive::new(Cursor::new(include_bytes!("test.pt"))).unwrap();
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    for index in 0..source.len() {
        let mut entry = source.by_index(index).unwrap();
        let name = format!("π/{name}", name = entry.name().unwrap());
        writer
            .start_file(
                name,
                SimpleFileOptions::default().compression_method(CompressionMethod::Stored),
            )
            .unwrap();
        std::io::copy(&mut entry, &mut writer).unwrap();
    }
    std::fs::write(&file, writer.finish().unwrap().into_inner()).unwrap();
    let original = candle_core::pickle::PthTensors::new("tests/test.pt", None)
        .unwrap()
        .get("test")
        .unwrap()
        .unwrap();
    let loaded = candle_core::pickle::PthTensors::new(&file, None)
        .unwrap()
        .get("test")
        .unwrap()
        .unwrap();
    assert_eq!(original.dims(), loaded.dims());
    assert_eq!(
        original.flatten_all().unwrap().to_vec1::<i64>().unwrap(),
        loaded.flatten_all().unwrap().to_vec1::<i64>().unwrap()
    );
}

#[test]
fn legacy_cp437_names_are_preserved_without_lossy_replacement() {
    let scratch = Scratch::new();
    let file = scratch.0.join("invalid.npz");
    let mut bytes = archive("bad.npy", include_bytes!("test.npy"));
    for (signature, flags, name) in [(b"PK\x03\x04", 6, 30), (b"PK\x01\x02", 8, 46)] {
        let index = bytes.windows(4).position(|part| part == signature).unwrap();
        bytes[index + flags + 1] &= !8; // Legacy CP437, not Unicode replacement decoding.
        bytes[index + name] = 0xff;
    }
    std::fs::write(&file, bytes).unwrap();
    let tensors = Tensor::read_npz(&file).unwrap();
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].0, "\u{a0}ad");
    let lazy = candle_core::npy::NpzTensors::new(&file).unwrap();
    assert_eq!(lazy.names(), [&"\u{a0}ad".to_string()]);
}

#[test]
fn enclosed_path_and_updater_extract_reject_parent_escape() {
    let scratch = Scratch::new();
    let bytes = archive("../outside.dll", b"not an executable");
    let mut zip = ZipArchive::new(Cursor::new(bytes)).unwrap();
    assert!(zip.by_index(0).unwrap().enclosed_name().is_none());
    let destination = scratch.0.join("destination");
    std::fs::create_dir(&destination).unwrap();
    assert!(zip.extract(&destination).is_err());
    assert!(!scratch.0.join("outside.dll").exists());
}

#[test]
fn truncated_container_is_an_error_for_npz_and_pth() {
    let scratch = Scratch::new();
    let file = scratch.0.join("truncated.zip");
    let mut bytes = archive("tensor.npy", include_bytes!("test.npy"));
    bytes.truncate(bytes.len() / 2);
    std::fs::write(&file, bytes).unwrap();
    assert!(Tensor::read_npz(&file).is_err());
    assert!(candle_core::pickle::PthTensors::new(&file, None).is_err());
}
