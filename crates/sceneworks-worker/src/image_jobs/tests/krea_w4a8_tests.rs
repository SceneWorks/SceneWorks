//! SW-6135 (feature SW-F287): the Krea 2 Turbo W4A8 tier on the generic Candle image stream —
//! request selection, artifact resolution, the native-file LoadSpec substitution, its tier
//! identity, telemetry label, and the typed refusals. Pure: no GPU, no weights.
use super::super::artifact_resolution::resolve_krea_w4a8;
use super::super::artifact_resolution::wants_krea_w4a8;
use super::super::candle::krea_w4a8_load_spec;
use super::super::candle::refuse_w4a8_block_streaming;
use super::super::krea_control_candle::refuse_krea_control_w4a8;
use super::super::load_spec::load_spec;
use super::super::memory::candle_stream_tier_key;
use super::super::metrics::effective_quant_label_gated;
use super::super::AdapterKind;
use super::super::AdapterSpec;
use super::super::ImageRequest;
use super::super::WeightsSource;
use super::super::W4A8_TIER;
use crate::Path;
use crate::PathBuf;
use crate::Settings;
use crate::Value;
use serde_json::json;
use std::fs;

fn bound_request(value: Value) -> ImageRequest {
    let payload = value.as_object().expect("request object");
    ImageRequest::from_resolved(payload, payload).expect("fixture options bind")
}

fn settings(data_dir: &Path) -> Settings {
    Settings {
        backend_candle_enabled: true,
        ..Settings::for_test(data_dir.to_path_buf())
    }
}

fn cached_snapshot(data_dir: &Path, repo: &str, revision: &str) -> PathBuf {
    let repo_root = sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir, repo)
        .expect("safe fixture repo");
    let root = repo_root.join("snapshots").join(revision);
    fs::create_dir_all(&root).expect("create fixture snapshot");
    fs::create_dir_all(repo_root.join("refs")).expect("create refs");
    fs::write(repo_root.join("refs").join("main"), revision).expect("write ref");
    root
}

fn isolate_hf_cache() -> crate::test_env::EnvVars {
    crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
    ])
}

#[test]
fn w4a8_is_selected_by_quant_tier_or_variant_on_krea_turbo_only() {
    for request in [
        json!({"model": "krea_2_turbo", "advanced": {"quantTier": "w4a8"}}),
        json!({"model": "krea_2_turbo", "advanced": {"quantTier": " W4A8 "}}),
        json!({"model": "krea_2_turbo", "variant": "w4a8"}),
    ] {
        assert!(
            wants_krea_w4a8(&bound_request(request.clone())),
            "{request}"
        );
    }
    for request in [
        json!({"model": "krea_2_turbo", "advanced": {"quantTier": "q4"}}),
        json!({"model": "krea_2_turbo", "advanced": {"convRot": true}}),
        json!({"model": "krea_2_turbo", "variant": "q8", "advanced": {"quantTier": "w4a8"}}),
        json!({"model": "krea_2_raw", "advanced": {"quantTier": "w4a8"}}),
    ] {
        assert!(
            !wants_krea_w4a8(&bound_request(request.clone())),
            "{request}"
        );
    }
}

#[test]
fn w4a8_resolves_the_published_transformer_and_the_dense_base_or_nothing() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let settings = settings(temp.path());
    let request =
        bound_request(json!({"model": "krea_2_turbo", "advanced": {"quantTier": "w4a8"}}));

    // Nothing installed: the worker refuses rather than rendering another tier.
    assert_eq!(resolve_krea_w4a8(&request, &settings), None);

    let w4a8 = cached_snapshot(
        temp.path(),
        "SceneWorks/krea-2-turbo-w4a8",
        "1e79167e1e8ad4c23dfbfa22b39f88a7ca2ba4d2",
    );
    fs::write(w4a8.join("krea2_turbo_w4a8.safetensors"), b"fixture").unwrap();
    // The transformer alone is not enough: the tokenizer / TE / VAE come from the bf16 base.
    assert_eq!(resolve_krea_w4a8(&request, &settings), None);

    let base = cached_snapshot(
        temp.path(),
        "SceneWorks/krea-2-turbo-mlx",
        "d009674080cc1bccf2b629d834c34bf5eccdb723",
    )
    .join("bf16");
    fs::create_dir_all(base.join("text_encoder")).unwrap();
    fs::create_dir_all(base.join("vae")).unwrap();
    fs::write(base.join("model_index.json"), "{}").unwrap();
    let (resolved_base, dit) = resolve_krea_w4a8(&request, &settings).expect("both installed");
    assert_eq!(resolved_base.file_name().unwrap(), "bf16");
    assert_eq!(dit.file_name().unwrap(), "krea2_turbo_w4a8.safetensors");

    // A request for any other tier never resolves the W4A8 pair.
    let q8 = bound_request(json!({"model": "krea_2_turbo", "advanced": {"quantTier": "q8"}}));
    assert_eq!(resolve_krea_w4a8(&q8, &settings), None);
}

#[test]
fn w4a8_load_spec_is_the_native_file_route_over_the_base_snapshot() {
    let base = PathBuf::from("/nonexistent/krea-2-turbo-mlx/bf16");
    let dit = PathBuf::from("/nonexistent/krea-2-turbo-w4a8/krea2_turbo_w4a8.safetensors");
    let adapter = AdapterSpec::new(
        PathBuf::from("/nonexistent/krea-style-lora.safetensors"),
        0.6,
        AdapterKind::Lora,
    );
    // Even a stray load quant is cleared: a packed W4A8 transformer is never requantized.
    let spec = load_spec(
        base.clone(),
        Some(gen_core::Quant::Q8),
        vec![adapter.clone()],
        None,
    );
    let spec = krea_w4a8_load_spec(spec, &base, &dit);

    assert!(matches!(&spec.weights, WeightsSource::File(path) if path == &dit));
    assert!(matches!(
        spec.components.get(gen_core::BASE_SNAPSHOT_COMPONENT),
        Some(WeightsSource::Dir(path)) if path == &base
    ));
    assert!(!spec
        .components
        .contains_key(gen_core::KREA_CONVROT_DIT_COMPONENT));
    assert_eq!(spec.quantize, None);
    assert!(spec.text_encoder.is_none());
    // LoRA/LoKr ride the tier as additive residuals (inference SW-6135), so the stack is kept.
    assert_eq!(spec.adapters.len(), 1);
    assert_eq!(spec.adapters[0].path, adapter.path);
}

#[test]
fn w4a8_is_gated_and_labelled_by_its_own_tier_identity() {
    let request =
        bound_request(json!({"model": "krea_2_turbo", "advanced": {"quantTier": "w4a8"}}));
    let base = Path::new("/models/krea-2-turbo-mlx/bf16");
    assert_eq!(W4A8_TIER, "w4a8");
    assert_eq!(
        candle_stream_tier_key(&request, base, false, true),
        W4A8_TIER,
        "the bf16 base basename must never size a W4A8 render as the bf16 tier"
    );
    // Unresolved, the same request sizes against the directory it would actually load.
    assert_eq!(candle_stream_tier_key(&request, base, false, false), "bf16");
    assert_eq!(
        effective_quant_label_gated(&request, false, Some(base)),
        (Some("w4a8".to_owned()), None),
        "no honest bit width: the label names the tier, never q4"
    );
}

#[test]
fn w4a8_refuses_block_streaming_but_admits_resident_and_staged_memory() {
    assert!(refuse_w4a8_block_streaming(None).is_ok());
    let staged = gen_core::GenerationMemory {
        stage_residency: true,
        ..Default::default()
    };
    assert!(refuse_w4a8_block_streaming(Some(&staged)).is_ok());
    let streamed = gen_core::GenerationMemory {
        stream_transformer_blocks: true,
        ..Default::default()
    };
    let error = refuse_w4a8_block_streaming(Some(&streamed))
        .expect_err("block streaming has no W4A8 form")
        .to_string();
    assert!(error.contains("block-streaming"), "{error}");
}

#[test]
fn w4a8_pose_control_is_refused_with_the_shared_reason() {
    let pose = bound_request(json!({
        "model": "krea_2_turbo",
        "advanced": {"quantTier": "w4a8"}
    }));
    let error = refuse_krea_control_w4a8(&pose)
        .expect_err("pose control does not run on the W4A8 tier")
        .to_string();
    assert!(error.contains("pose control"), "{error}");
    assert!(error.contains("W4A8"), "{error}");
    let q8 = bound_request(json!({
        "model": "krea_2_turbo",
        "advanced": {"quantTier": "q8"}
    }));
    assert!(refuse_krea_control_w4a8(&q8).is_ok());
}
