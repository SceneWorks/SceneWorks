//! Pure shared helper contracts: no network, model loading or learned execution.
mod common;

use sceneworks_core::{
    display_util::{format_bytes, quote_path},
    model_artifacts::artifact_selection::{
        allow_pattern_matches, pattern_matches, safe_download_dir,
    },
    payload_util::json_size_to_u64,
};
use serde_json::json;

#[test]
fn byte_labels_keep_binary_thresholds_rounding_and_tb_ceiling() {
    for (bytes, expected) in [
        (0, "0 B"),
        (1, "1 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1024 * 1024 - 1, "1024.0 KB"),
        (1024 * 1024, "1.0 MB"),
        (1024_u64.pow(3), "1.0 GB"),
        (1024_u64.pow(4), "1.0 TB"),
        (1024_u64.pow(5), "1024.0 TB"),
        (u64::MAX, "16777216.0 TB"),
    ] {
        assert_eq!(format_bytes(bytes), expected, "{bytes}");
    }
}

#[test]
fn url_path_escaping_preserves_separators_and_encodes_utf8_bytes_once() {
    assert_eq!(
        quote_path("AZaz09-._~/owner/model"),
        "AZaz09-._~/owner/model"
    );
    assert_eq!(
        quote_path("a b+c%20?#&=:@\\\"'\n"),
        "a%20b%2Bc%2520%3F%23%26%3D%3A%40%5C%22%27%0A"
    );
    assert_eq!(quote_path("人物/é"), "%E4%BA%BA%E7%89%A9/%C3%A9");
    assert_eq!(quote_path(""), "");
}

#[test]
fn glob_contract_keeps_default_separators_invalid_patterns_and_platform_case() {
    assert!(allow_pattern_matches("any/file", &[]));
    for (pattern, path, expected) in [
        ("q8/*", "q8/nested/model.safetensors", true),
        ("*.safetensors", "nested/model.safetensors", true),
        ("model-[0-9].safetensors", "model-7.safetensors", true),
        ("model-?.bin", "model-12.bin", false),
        ("[", "[", false),
        ("q8/*", "q4/model", false),
        ("*.bin", ".hidden.bin", true),
        ("", "", true),
    ] {
        assert_eq!(pattern_matches(pattern, path), expected, "{pattern}/{path}");
        assert_eq!(allow_pattern_matches(path, &[pattern.to_owned()]), expected);
    }
    assert_eq!(
        pattern_matches("*.safetensors", "MODEL.SAFETENSORS"),
        cfg!(windows)
    );
    // Windows folds ASCII only, preserving the original Unicode distinction.
    assert!(!pattern_matches("é.bin", "É.bin"));
    assert!(allow_pattern_matches(
        "model.bin",
        &["[".into(), "*.bin".into()]
    ));
}

#[test]
fn size_parser_preserves_strict_unsigned_types_and_untrimmed_strings() {
    for (value, expected) in [
        (json!(0), Some(0)),
        (json!(u64::MAX), Some(u64::MAX)),
        (json!("200"), Some(200)),
        (json!("+200"), Some(200)),
        (json!("0002"), Some(2)),
        (json!(" 200 "), None),
        (json!("200.0"), None),
        (json!(200.0), None),
        (json!(-1), None),
        (json!(true), None),
        (json!(null), None),
        (json!("18446744073709551616"), None),
        (json!({}), None),
    ] {
        assert_eq!(json_size_to_u64(&value), expected, "{value}");
    }
}

#[test]
fn download_directory_keeps_historical_replacement_spelling() {
    assert_eq!(safe_download_dir("owner/model name"), "owner__model__name");
    assert_eq!(safe_download_dir("a//é  b"), "a__b");
    assert_eq!(safe_download_dir(""), "download");
    // Directory spelling is not the separate safe-id/confinement boundary.
    assert_eq!(safe_download_dir(".."), "..");
    assert_eq!(safe_download_dir("A_b-1.safetensors"), "A_b-1.safetensors");
}

#[test]
fn environment_policies_preserve_api_raw_worker_trimmed_and_path_defaults() {
    use sceneworks_core::env_util::{env_path_or, env_string, env_trimmed_string};
    use std::{path::Path, process::Command};
    const KEY: &str = "SCENEWORKS_ARTIFACT_HELPER_ENV_FIXTURE";
    const CHILD: &str = "SCENEWORKS_ARTIFACT_HELPER_ENV_CHILD";
    if let Ok(case) = std::env::var(CHILD) {
        let present = case == "value";
        assert_eq!(
            env_string(KEY, " raw default "),
            if present {
                "  custom path  "
            } else {
                " raw default "
            }
        );
        assert_eq!(
            env_trimmed_string(KEY, " trimmed default "),
            if present {
                "custom path"
            } else {
                " trimmed default "
            }
        );
        assert_eq!(
            env_path_or(KEY, Path::new(" default path ")),
            Path::new(if present {
                "custom path"
            } else {
                " default path "
            })
        );
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let default = std::path::PathBuf::from(std::ffi::OsString::from_vec(vec![0xff]));
            if !present {
                assert_eq!(env_path_or(KEY, &default), default);
            }
        }
        return;
    }
    for (case, value) in [
        ("value", Some("  custom path  ")),
        ("blank", Some(" \n\t ")),
        ("missing", None),
    ] {
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args([
            "--exact",
            "environment_policies_preserve_api_raw_worker_trimmed_and_path_defaults",
        ])
        .env(CHILD, case);
        if let Some(value) = value {
            cmd.env(KEY, value);
        } else {
            cmd.env_remove(KEY);
        }
        assert!(cmd.status().unwrap().success(), "{case}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        assert!(Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "environment_policies_preserve_api_raw_worker_trimmed_and_path_defaults"
            ])
            .env(CHILD, "nonunicode")
            .env(KEY, std::ffi::OsString::from_vec(vec![0xff]))
            .status()
            .unwrap()
            .success());
    }
}

// The shared fixtures interoperate with the production header reader; no tensor execution.

#[test]
fn shared_safetensors_fixture_preserves_metadata_fill_and_header_offsets() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("nested/metadata.safetensors");
    common::write_safetensors(
        &path,
        &[("half", "BF16"), ("full", "F32"), ("byte", "U8")],
        Some(json!({ "source": "fixture" })),
        0x5a,
    );
    let before = common::tree_snapshot(root.path());
    let header = common::safetensors_header(&path);
    assert_eq!(header.value["__metadata__"], json!({ "source": "fixture" }));
    assert_eq!(header.value["half"]["data_offsets"], json!([0, 2]));
    assert_eq!(header.value["full"]["data_offsets"], json!([2, 6]));
    assert_eq!(header.value["byte"]["data_offsets"], json!([6, 7]));
    assert_eq!(header.file_len - header.data_start, 7);
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[header.data_start as usize..], &[0x5a; 7]);
    assert_eq!(
        common::tree_snapshot(root.path()),
        before,
        "header read is immutable"
    );
}

#[test]
fn shared_krea_fixture_keeps_its_three_dense_bf16_markers() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("krea.safetensors");
    common::write_krea_native_file(&path, 0x22);
    let header = common::safetensors_header(&path);
    assert_eq!(header.value.as_object().unwrap().len(), 3);
    assert_eq!(
        header.value["model.diffusion_model.txtfusion.projector.weight"]["dtype"],
        "BF16"
    );
    assert_eq!(header.file_len - header.data_start, 6);
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[header.data_start as usize..], &[0x22; 6]);
}
