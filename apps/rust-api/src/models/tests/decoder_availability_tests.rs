use super::*;
use crate::tests::support::isolate_hf_cache;

fn model() -> Value {
    json!({
        "id": "qwen_image",
        "downloads": [{
            "provider": "huggingface",
            "repo": "SceneWorks/krea-realtime-14b-mlx",
            "revision": "e68e9a3d98187fdf6936838ffcf6df5aa48d6626",
            "coRequisite": true,
            "required": "soft",
            "componentId": "vae",
            "files": ["q4/vae.safetensors"],
            "estimatedSizeBytes": 507591212
        }],
        "decoders": { "byBackend": { "mlx": [{
            "id": "wan_2_1_vae", "componentId": "vae", "experimental": true
        }] } }
    })
}

fn donor_download() -> Value {
    model()["downloads"][0].clone()
}

#[test]
fn option_is_available_only_at_the_declared_pinned_single_file() {
    let _env = isolate_hf_cache();
    let data = tempfile::tempdir().unwrap();
    let mut model = model();
    apply_decoder_availability(model.as_object_mut().unwrap(), data.path());
    assert_eq!(model["decoders"]["byBackend"]["mlx"][0]["available"], false);

    let repo_cache =
        huggingface_repo_cache_path(data.path(), "SceneWorks/krea-realtime-14b-mlx").unwrap();
    let wrong = repo_cache.join("snapshots/0000000000000000000000000000000000000000/q4");
    std::fs::create_dir_all(&wrong).unwrap();
    std::fs::write(wrong.join("vae.safetensors"), b"stale donor").unwrap();
    apply_decoder_availability(model.as_object_mut().unwrap(), data.path());
    assert_eq!(model["decoders"]["byBackend"]["mlx"][0]["available"], false);

    let exact = repo_cache.join("snapshots/e68e9a3d98187fdf6936838ffcf6df5aa48d6626/q4");
    std::fs::create_dir_all(&exact).unwrap();
    std::fs::write(exact.join("vae.safetensors"), b"pinned donor").unwrap();
    apply_decoder_availability(model.as_object_mut().unwrap(), data.path());
    let option = &model["decoders"]["byBackend"]["mlx"][0];
    assert_eq!(option["available"], true);
    assert_eq!(option["estimatedSizeBytes"], 507591212_u64);
}

#[test]
fn imported_krea_alias_inherits_exact_provider_donor_and_fails_closed() {
    let _env = isolate_hf_cache();
    let data = tempfile::tempdir().unwrap();
    let builtin = vec![json!({
        "id": "krea_2_turbo",
        "downloads": [donor_download()]
    })];
    let mut imported = json!({
        "id": "user_kreamania_variant5",
        "name": "Kreamania Variant 5",
        "type": "image",
        "family": "krea_2",
        "downloads": []
    });

    inherit_provider_decoder_downloads(&mut imported, &builtin);
    let donor = &imported["downloads"][0];
    assert_eq!(donor["repo"], "SceneWorks/krea-realtime-14b-mlx");
    assert_eq!(
        donor["revision"],
        "e68e9a3d98187fdf6936838ffcf6df5aa48d6626"
    );
    assert_eq!(donor["files"], json!(["q4/vae.safetensors"]));

    sceneworks_core::decoder_support::apply_to_model_entry(imported.as_object_mut().unwrap());
    apply_decoder_availability(imported.as_object_mut().unwrap(), data.path());
    assert_eq!(
        imported["decoders"]["byBackend"]["mlx"][0]["available"], false,
        "an imported alias must not advertise an unresolved donor as selectable"
    );

    let repo_cache =
        huggingface_repo_cache_path(data.path(), "SceneWorks/krea-realtime-14b-mlx").unwrap();
    let wrong = repo_cache.join("snapshots/0000000000000000000000000000000000000000/q4");
    std::fs::create_dir_all(&wrong).unwrap();
    std::fs::write(wrong.join("vae.safetensors"), b"wrong revision").unwrap();
    apply_decoder_availability(imported.as_object_mut().unwrap(), data.path());
    assert_eq!(
        imported["decoders"]["byBackend"]["mlx"][0]["available"],
        false
    );

    let exact = repo_cache.join("snapshots/e68e9a3d98187fdf6936838ffcf6df5aa48d6626/q4");
    std::fs::create_dir_all(&exact).unwrap();
    std::fs::write(exact.join("vae.safetensors"), b"exact donor").unwrap();
    apply_decoder_availability(imported.as_object_mut().unwrap(), data.path());
    assert_eq!(
        imported["decoders"]["byBackend"]["mlx"][0]["available"],
        true
    );
}
