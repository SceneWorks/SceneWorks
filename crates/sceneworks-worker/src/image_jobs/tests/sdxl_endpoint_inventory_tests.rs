use super::*;
use sceneworks_core::memory_anchor::sdxl_cuda_memory_endpoints::EndpointArtifactFile;
use sha2::{Digest, Sha256};

fn fixture() -> (tempfile::TempDir, LoadSpec, Vec<EndpointArtifactFile>) {
    let temp = tempfile::tempdir().unwrap();
    let mut files = Vec::new();
    for path in [
        "base/unet/model.safetensors",
        "base/vae/model.safetensors",
        "components/tokenizer_clip_l/tokenizer.json",
        "components/tokenizer_clip_bigg/tokenizer.json",
        "components/vae_fp16_fix/diffusion_pytorch_model.safetensors",
    ] {
        let file = temp.path().join(path);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"fixture-a").unwrap();
        files.push(EndpointArtifactFile {
            path: path.to_owned(),
            bytes: 9,
            sha256: format!("{:x}", Sha256::digest(b"fixture-a")),
        });
    }
    let mut spec = LoadSpec::new(WeightsSource::Dir(temp.path().join("base")));
    for id in ["tokenizer_clip_l", "tokenizer_clip_bigg", "vae_fp16_fix"] {
        spec = spec.with_component(
            id,
            WeightsSource::Dir(temp.path().join("components").join(id)),
        );
    }
    (temp, spec, files)
}

// CI_IMPACT: tiny filesystem fixtures and SHA-256 only; no model, learned layer, or device.
#[test]
fn sdxl_endpoint_inventory_pins_the_actual_shared_vae_and_both_tokenizers() {
    let (_temp, mut spec, expected) = fixture();
    assert!(seal_sdxl_endpoint_load_spec(&mut spec, &expected).unwrap());
    assert_eq!(spec.prepared_file_pins().len(), expected.len());
    spec.validate_prepared_file_pins().unwrap();
    let shared_vae = match &spec.components["vae_fp16_fix"] {
        WeightsSource::Dir(root) => root.join("diffusion_pytorch_model.safetensors"),
        _ => unreachable!(),
    };
    assert!(spec.prepared_file_pins().get(&shared_vae).is_some());
    std::fs::write(shared_vae, b"fixture-b").unwrap();
    assert!(spec.validate_prepared_file_pins().is_err());
}

// CI_IMPACT: actual stock co-requisite factory over four tiny cached files; no model/device calls.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn sdxl_endpoint_inventory_accepts_the_certified_stock_component_factory() {
    let temp = tempfile::tempdir().unwrap();
    let _env = crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
        ("SCENEWORKS_DATA_DIR", temp.path().to_str().unwrap()),
        ("SCENEWORKS_CONFIG_DIR", temp.path().to_str().unwrap()),
    ]);
    let settings = Settings::from_env();
    let source = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, source)| *source)
        .unwrap();
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(source)).unwrap();
    let manifest = manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == "sdxl")
        .unwrap()
        .as_object()
        .unwrap();
    let descriptor = crate::inference_runtime::media_descriptor("sdxl").unwrap();
    let mut expected = Vec::new();
    let mut base = None;
    for download in manifest["downloads"].as_array().unwrap() {
        let component = download.get("componentId").and_then(Value::as_str);
        if component.is_some_and(|id| !descriptor.required_components.contains(&id))
            || (component.is_none() && download["variant"] != "q4")
        {
            continue;
        }
        let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(
            &settings.data_dir,
            download["repo"].as_str().unwrap(),
        )
        .unwrap()
        .join("snapshots")
        .join(download["revision"].as_str().unwrap());
        let (relative, file) = if let Some(component) = component {
            let file = download["files"][0].as_str().unwrap();
            (
                format!("components/{component}/{file}"),
                snapshot.join(file),
            )
        } else {
            base = Some(snapshot.join("q4"));
            (
                "base/fixture.bin".to_owned(),
                snapshot.join("q4/fixture.bin"),
            )
        };
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(file, b"fixture-a").unwrap();
        expected.push(EndpointArtifactFile {
            path: relative,
            bytes: 9,
            sha256: format!("{:x}", Sha256::digest(b"fixture-a")),
        });
    }
    let spec =
        load_spec(base.unwrap(), Some(Quant::Q4), Vec::new(), None).with_resolved_route("sdxl");
    let mut spec = attach_required_components(spec, "sdxl", manifest, &settings).unwrap();
    assert_eq!(spec.components.len(), 3);
    assert_eq!(spec.file_source_paths().len(), 3);
    assert_eq!(spec.directory_source_paths().len(), 1);
    assert!(candle_certified_load_spec(
        "sdxl", &settings, &spec, manifest, "q4"
    ));
    assert!(seal_sdxl_endpoint_load_spec(&mut spec, &expected).unwrap());
    assert_eq!(spec.prepared_file_pins().len(), expected.len());
    spec.validate_prepared_file_pins().unwrap();
    for source in spec.components.values() {
        let WeightsSource::File(path) = source else {
            panic!("the stock single-literal co-requisite must resolve to File");
        };
        assert!(spec.prepared_file_pins().get(path).is_some());
    }
}

#[test]
fn sdxl_endpoint_inventory_file_components_reject_changed_members_and_extra_sources() {
    for changed in 0..3 {
        let (temp, mut spec, expected) = fixture();
        for file in &expected[2..] {
            let component = file.path.split('/').nth(1).unwrap();
            spec =
                spec.with_component(component, WeightsSource::File(temp.path().join(&file.path)));
        }
        let path = temp.path().join(&expected[2 + changed].path);
        std::fs::write(&path, b"fixture-b").unwrap();
        assert!(!seal_sdxl_endpoint_load_spec(&mut spec, &expected).unwrap());
        std::fs::remove_file(path).unwrap();
        assert!(!seal_sdxl_endpoint_load_spec(&mut spec, &expected).unwrap());
    }
    let (temp, spec, expected) = fixture();
    let renamed = temp.path().join("components/tokenizer_clip_l/other.json");
    std::fs::write(&renamed, b"fixture-a").unwrap();
    let mut renamed_spec = spec
        .clone()
        .with_component("tokenizer_clip_l", WeightsSource::File(renamed));
    assert!(!seal_sdxl_endpoint_load_spec(&mut renamed_spec, &expected).unwrap());
    for source in [
        WeightsSource::File(temp.path().join(&expected[2].path)),
        WeightsSource::Dir(temp.path().join("base")),
    ] {
        let mut extra = spec.clone().with_text_encoder(source);
        assert!(!seal_sdxl_endpoint_load_spec(&mut extra, &expected).unwrap());
    }
}

#[test]
fn sdxl_endpoint_inventory_rejects_changed_missing_and_extra_members() {
    // Equal-size changes in every source must lose the measured price, including shared VAE
    // changes which a tier-root-only inventory would miss.
    for index in 0..5 {
        let (temp, mut spec, expected) = fixture();
        std::fs::write(temp.path().join(&expected[index].path), b"fixture-b").unwrap();
        assert!(!seal_sdxl_endpoint_load_spec(&mut spec, &expected).unwrap());
        assert!(!spec.prepared_file_pins().is_prepared());
    }
    let (temp, mut spec, expected) = fixture();
    std::fs::remove_file(temp.path().join(&expected[0].path)).unwrap();
    assert!(!seal_sdxl_endpoint_load_spec(&mut spec, &expected).unwrap());

    let (temp, mut spec, expected) = fixture();
    std::fs::write(temp.path().join("base/added.safetensors"), b"fixture-a").unwrap();
    assert!(!seal_sdxl_endpoint_load_spec(&mut spec, &expected).unwrap());

    let (temp, spec, expected) = fixture();
    let mut spec = spec.with_component("unexpected", WeightsSource::Dir(temp.path().join("base")));
    assert!(!seal_sdxl_endpoint_load_spec(&mut spec, &expected).unwrap());
}

#[test]
fn sdxl_endpoint_inventory_ignores_unused_shared_snapshot_members() {
    let (temp, mut spec, expected) = fixture();
    std::fs::write(
        temp.path()
            .join("components/tokenizer_clip_l/model.safetensors"),
        b"unused",
    )
    .unwrap();
    std::fs::write(
        temp.path().join("components/vae_fp16_fix/config.json"),
        b"{}",
    )
    .unwrap();
    assert!(seal_sdxl_endpoint_load_spec(&mut spec, &expected).unwrap());
    assert_eq!(spec.prepared_file_pins().len(), expected.len());
}
