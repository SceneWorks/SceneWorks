use super::super::LoadSpec;
use super::super::WeightsSource;
use super::candle_certified_hf_artifact_path;
use super::candle_certified_load_spec;
use crate::JsonObject;
use crate::Path;
use crate::PathBuf;
use crate::Settings;
use crate::Value;
use crate::DEFAULT_HUGGINGFACE_BASE_URL;
use crate::DEFAULT_MAX_LORA_URL_BYTES;
use crate::DEFAULT_MAX_MODEL_URL_BYTES;
use std::fs;

fn settings(data_dir: &Path) -> Settings {
    Settings {
        api_url: "http://127.0.0.1".to_owned(),
        config_dir: data_dir.join("config"),
        worker_id: "test-worker".to_owned(),
        gpu_id: "gpu-0".to_owned(),
        huggingface_base_url: DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
        max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
        max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
        backend_candle_enabled: true,
        ..crate::Settings::for_test(data_dir.to_path_buf())
    }
}

fn model_manifest(model_id: &str) -> JsonObject {
    let source = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, source)| *source)
        .expect("embedded model manifest");
    let root: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(source))
        .expect("model manifest parses");
    root["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|model| model["id"] == model_id)
        .and_then(Value::as_object)
        .cloned()
        .expect("model manifest entry")
}

fn cached_snapshot(data_dir: &Path, repo: &str, revision: &str) -> PathBuf {
    let root = sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir, repo)
        .expect("safe fixture repo")
        .join("snapshots")
        .join(revision);
    fs::create_dir_all(&root).expect("create fixture snapshot");
    root
}

#[test]
fn chroma_certification_binds_exact_route_revision_and_physical_tier() {
    let _env = crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
    ]);
    let data = tempfile::tempdir().expect("temp data dir");
    let settings = settings(data.path());
    for (model, repo, revision) in [
        (
            "chroma1_hd",
            "SceneWorks/chroma1-hd-mlx",
            "9d99afe1ebca67032476756bc70d4a7152bc1bd5",
        ),
        (
            "chroma1_base",
            "SceneWorks/chroma1-base-mlx",
            "e7330dda29d00ffdeeb719b28e92ee74cff0884c",
        ),
        (
            "chroma1_flash",
            "SceneWorks/chroma1-flash-mlx",
            "6a9cb6178709559461506bf247f708d0d1008d00",
        ),
    ] {
        let snapshot = cached_snapshot(data.path(), repo, revision);
        let tier = snapshot.join("q4");
        fs::create_dir_all(&tier).expect("tier dir");
        let spec = LoadSpec::new(WeightsSource::Dir(tier.clone())).with_resolved_route(model);
        let manifest = model_manifest(model);
        assert!(candle_certified_load_spec(
            model, &settings, &spec, &manifest, "q4"
        ));

        let wrong_tier =
            LoadSpec::new(WeightsSource::Dir(snapshot.join("q8"))).with_resolved_route(model);
        assert!(!candle_certified_load_spec(
            model,
            &settings,
            &wrong_tier,
            &manifest,
            "q4"
        ));
        let wrong_revision = data.path().join("wrong-revision/q4");
        fs::create_dir_all(&wrong_revision).expect("wrong revision dir");
        let wrong_revision =
            LoadSpec::new(WeightsSource::Dir(wrong_revision)).with_resolved_route(model);
        assert!(!candle_certified_load_spec(
            model,
            &settings,
            &wrong_revision,
            &manifest,
            "q4"
        ));
    }
}

#[test]
fn mage_certification_binds_backbone_and_every_required_component() {
    // The pinned-snapshot resolver consults the process-global HF cache env before
    // `settings.data_dir`. Hold the crate-wide env lock and clear those overrides so a
    // concurrent cache-resolution test cannot redirect this fixture mid-certification.
    let _env = crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
    ]);
    let data = tempfile::tempdir().expect("temp data dir");
    let settings = settings(data.path());
    let manifest = model_manifest("mage_flow_base");
    let downloads = manifest["downloads"].as_array().expect("downloads");
    let revision_for = |component_id: Option<&str>| {
        let download = downloads
            .iter()
            .find(|download| {
                download["variant"] == "q4"
                    && download.get("componentId").and_then(Value::as_str) == component_id
            })
            .expect("tier artifact");
        (
            download["repo"].as_str().expect("repo"),
            download["revision"].as_str().expect("revision"),
        )
    };
    let (backbone_repo, backbone_revision) = revision_for(None);
    let (components_repo, components_revision) = revision_for(Some("text_encoder"));
    let backbone = cached_snapshot(data.path(), backbone_repo, backbone_revision).join("q4");
    let components = cached_snapshot(data.path(), components_repo, components_revision).join("q4");
    let text_encoder = components.join("text_encoder");
    let vae = components.join("vae");
    for path in [&backbone, &text_encoder, &vae] {
        fs::create_dir_all(path).expect("create tier artifact");
    }

    let complete = LoadSpec::new(WeightsSource::Dir(backbone.clone()))
        .with_component("text_encoder", WeightsSource::Dir(text_encoder.clone()))
        .with_component("vae", WeightsSource::Dir(vae.clone()));
    assert!(candle_certified_load_spec(
        "mage_flow_base",
        &settings,
        &complete,
        &manifest,
        "q4",
    ));

    let substituted_vae = LoadSpec::new(WeightsSource::Dir(backbone.clone()))
        .with_component("text_encoder", WeightsSource::Dir(text_encoder))
        .with_component("vae", WeightsSource::Dir(data.path().join("other-vae")));
    assert!(!candle_certified_load_spec(
        "mage_flow_base",
        &settings,
        &substituted_vae,
        &manifest,
        "q4",
    ));

    let incomplete =
        LoadSpec::new(WeightsSource::Dir(backbone)).with_component("vae", WeightsSource::Dir(vae));
    assert!(!candle_certified_load_spec(
        "mage_flow_base",
        &settings,
        &incomplete,
        &manifest,
        "q4",
    ));
}

#[test]
fn strict_z_control_requires_independent_primary_and_control_certification() {
    let _env = crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
    ]);
    let data = tempfile::tempdir().expect("temp data dir");
    let settings = settings(data.path());
    let manifest = model_manifest("z_image");
    let primary = manifest["downloads"]
        .as_array()
        .expect("downloads")
        .iter()
        .find(|download| download["variant"] == "q4")
        .expect("q4 primary download");
    let primary_repo = primary["repo"].as_str().expect("primary repo");
    let primary_revision = primary["revision"].as_str().expect("primary revision");
    let primary_path = cached_snapshot(data.path(), primary_repo, primary_revision).join("q4");
    fs::create_dir_all(&primary_path).expect("primary tier");

    let control_root = cached_snapshot(
        data.path(),
        crate::image_jobs::zimage_control::ZIMAGE_CTRL_BASE_REPO,
        crate::image_jobs::zimage_control::ZIMAGE_CTRL_BASE_REVISION,
    );
    let control_path = control_root.join(crate::image_jobs::zimage_control::ZIMAGE_CTRL_BASE_FILE);
    fs::write(&control_path, b"control").expect("control file");
    let spec = LoadSpec::new(WeightsSource::Dir(primary_path.clone()))
        .with_resolved_route("z_image")
        .with_control(WeightsSource::File(control_path.clone()));

    assert!(candle_certified_load_spec(
        "z_image_control",
        &settings,
        &spec,
        &manifest,
        "q4",
    ));
    assert!(candle_certified_hf_artifact_path(
        &settings,
        crate::image_jobs::zimage_control::ZIMAGE_CTRL_BASE_REPO,
        crate::image_jobs::zimage_control::ZIMAGE_CTRL_BASE_REVISION,
        Path::new(crate::image_jobs::zimage_control::ZIMAGE_CTRL_BASE_FILE),
        &control_path,
    ));
    assert!(!candle_certified_load_spec(
        "z_image_control",
        &settings,
        &LoadSpec::new(WeightsSource::Dir(data.path().join("substituted-primary")))
            .with_resolved_route("z_image")
            .with_control(WeightsSource::File(control_path)),
        &manifest,
        "q4",
    ));
    assert!(!candle_certified_hf_artifact_path(
        &settings,
        crate::image_jobs::zimage_control::ZIMAGE_CTRL_BASE_REPO,
        crate::image_jobs::zimage_control::ZIMAGE_CTRL_BASE_REVISION,
        Path::new(crate::image_jobs::zimage_control::ZIMAGE_CTRL_BASE_FILE),
        &data.path().join("substituted-control.safetensors"),
    ));
}
